"""Transport layer of GatewayStreamConsumer: native frames, drafts, edit/send.

Mixin methods use only ``self`` state; see gateway/stream_consumer.py for the
state model and the drain loop that calls into these."""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from typing import Any, Optional

from gateway.platforms.base import BasePlatformAdapter as _BasePlatformAdapter
from gateway.stream_consumer_fences import ensure_closed_code_fences

logger = logging.getLogger("gateway.stream_consumer")


class StreamTransportMixin:
    """Send/edit/frame primitives and the transport-ordered ``_send_or_edit``."""

    _MIN_NEW_MSG_CHARS = 4

    async def _edit_message(
        self,
        *,
        message_id: str,
        content: str,
        finalize: bool = False,
    ):
        """Edit via the adapter, passing routing metadata when supported."""
        # Contract: adapters must accept finalize= even when False (test-guarded).
        kwargs = {
            "chat_id": self.chat_id,
            "message_id": message_id,
            "content": content,
            "finalize": finalize,
        }
        if self.metadata:
            try:
                params = inspect.signature(self.adapter.edit_message).parameters
                if "metadata" in params or any(
                    param.kind is inspect.Parameter.VAR_KEYWORD
                    for param in params.values()
                ):
                    kwargs["metadata"] = self.metadata
            except (TypeError, ValueError):
                pass
        return await self.adapter.edit_message(**kwargs)

    async def _send_seed_frame(self):
        """Open a native stream with an empty seed frame (typing indicator before any token)."""
        return await self.adapter.send_stream_frame(
            "",
            chat_id=self.chat_id,
            reply_to=self._initial_reply_to_id,
            turn_id=self._turn_id,
        )

    async def _send_frame(self, text: str, *, finalize: bool):
        """One native-stream frame; every frame carries the same chat/reply/turn routing."""
        return await self.adapter.send_stream_frame(
            text,
            finalize=finalize,
            chat_id=self.chat_id,
            reply_to=self._initial_reply_to_id,
            turn_id=self._turn_id,
        )

    def _close_native_state(self) -> None:
        """Mark the native stream closed (next content re-seeds or falls back)."""
        self._native_stream_opened = False
        self._native_last_pushed_len = 0

    def _degrade_native_to_buffered_send(self) -> None:
        """Leave native mode; post-boundary output goes out as ONE send() at got_done.

        buffer_only avoids mid-stream flushes that create multiple messages on
        non-editable platforms.
        """
        self._use_native_streaming = False
        self._close_native_state()
        self.cfg.buffer_only = True

    def _draft_metadata(self) -> dict | None:
        """Draft-frame metadata.

        Every frame must carry the same reply_to_message_id the final send gets
        from _metadata_for_send: the relay adapter keys draft/seal state on it,
        else the final can't find the open stream (flat DMs have no thread
        metadata and would key on the bare chat).
        """
        md = dict(self.metadata) if self.metadata else {}
        if self._initial_reply_to_id:
            md.setdefault("reply_to_message_id", self._initial_reply_to_id)
        return md or None

    def _stale_preview_ids(self, *, segment_only: bool = False) -> set:
        """Preview message ids a fresh final replaces.

        ``segment_only``: never delete an earlier finalized preamble.
        """
        if segment_only:
            stale_ids = set(self._segment_preview_message_ids)
            if self._message_id and self._message_id != "__no_edit__":
                stale_ids.add(str(self._message_id))
            return stale_ids
        stale_ids = set(self._preview_message_ids)
        if self._message_id and self._message_id != "__no_edit__":
            stale_ids.add(self._message_id)
        return stale_ids

    async def _delete_previews(
        self, stale_ids, *, skip=None, label: str, retry_on_false: bool = False,
        skip_sentinel: bool = True,
    ) -> None:
        """Best-effort delete of stale previews; never the message just sent (``skip``)."""
        delete_fn = getattr(self.adapter, "delete_message", None)
        if delete_fn is None:
            return
        for stale_id in stale_ids:
            if not stale_id or stale_id == skip or (skip_sentinel and stale_id == "__no_edit__"):
                continue
            try:
                deleted = await delete_fn(self.chat_id, stale_id)
                if retry_on_false and deleted is False:
                    await asyncio.sleep(1.0)
                    await delete_fn(self.chat_id, stale_id)
            except Exception as e:
                logger.debug("%s preview cleanup failed (%s): %s", label, stale_id, e)

    def _resolve_draft_streaming(self) -> bool:
        """Whether this run should use draft streaming per ``cfg.transport``.

        "edit"/"off" → False.  "draft"/"auto" → the adapter's
        supports_draft_streaming probe (chat type, platform-version gates);
        "draft" logs the downgrade when unsupported.
        """
        transport = (self.cfg.transport or "edit").lower()
        if transport in ("edit", "off"):
            return False
        # MagicMock test adapters default to edit.
        if not isinstance(self.adapter, _BasePlatformAdapter):
            return False
        try:
            try:
                # Per-chat probe (relay adapters resolve through the CHAT's
                # descriptor); older adapters without the kwarg keep the legacy probe.
                supported = self.adapter.supports_draft_streaming(
                    chat_type=self.cfg.chat_type or None,
                    metadata=self.metadata,
                    chat_id=self.chat_id,
                )
            except TypeError:
                supported = self.adapter.supports_draft_streaming(
                    chat_type=self.cfg.chat_type or None,
                    metadata=self.metadata,
                )
        except Exception:
            logger.debug("supports_draft_streaming probe raised", exc_info=True)
            supported = False
        if not supported:
            if transport == "draft":
                logger.debug(
                    "Draft streaming requested but unsupported (chat=%s, type=%r) — "
                    "falling back to edit",
                    self.chat_id, self.cfg.chat_type,
                )
            return False
        return True

    def _resolve_native_streaming(self) -> bool:
        """Whether to use native streaming (adapter.send_stream_frame for ALL frames).

        Requires a BasePlatformAdapter subclass with class-level
        SUPPORTS_NATIVE_STREAMING and a truthy supports_native_streaming probe.
        """
        if not isinstance(self.adapter, _BasePlatformAdapter):
            return False
        if not getattr(type(self.adapter), "SUPPORTS_NATIVE_STREAMING", False):
            return False
        probe = getattr(self.adapter, "supports_native_streaming", None)
        if probe is None:
            return False
        try:
            supported = probe(
                chat_type=self.cfg.chat_type or None,
                metadata=self.metadata,
            )
        except Exception:
            logger.debug(
                "supports_native_streaming probe raised", exc_info=True,
            )
            return False
        return bool(supported)

    async def _send_draft_frame(self, text: str) -> bool:
        """Emit one draft frame; any failure permanently disables drafts for this run.

        Drafts have no message_id and clear on the client when the final
        sendMessage lands.
        """
        if self._draft_id is None:
            # Should never happen (set in tandem with _use_draft_streaming in run()).
            self._use_draft_streaming = False
            return False
        # Every frame must carry the same reply_to_message_id the final send
        # gets from _metadata_for_send: the relay adapter keys draft/seal state
        # on it, else the final can't find the open stream (flat DMs have no
        # thread metadata and would key on the bare chat).
        try:
            result = await self.adapter.send_draft(
                chat_id=self.chat_id,
                draft_id=self._draft_id,
                content=text,
                metadata=self._draft_metadata(),
            )
        except Exception as e:
            logger.debug(
                "send_draft raised, disabling draft transport for this run: %s", e,
            )
            self._draft_failures += 1
            self._use_draft_streaming = False
            return False
        if not getattr(result, "success", False):
            logger.debug(
                "send_draft returned success=False, disabling draft transport: %s",
                getattr(result, "error", "unknown"),
            )
            self._draft_failures += 1
            self._use_draft_streaming = False
            return False
        self._last_sent_text = text  # parity with the edit-based no-op skip
        return True

    async def _abandon_native_stream(self) -> None:
        """Seal an orphaned draft stream in place on turn death (stale exit / cancel).

        Otherwise the message keeps its live indicator forever and the
        adapter's armed interception state leaks into the next turn.  Never
        sets delivery flags — the gateway's normal paths own what happens next.
        """
        if not self._use_draft_streaming:
            return
        abandon = getattr(type(self.adapter), "abandon_open_draft", None)
        if abandon is None:
            return
        try:
            await self.adapter.abandon_open_draft(
                self.chat_id,
                self._last_sent_text or self._clean_for_display(self._accumulated),
                metadata=self._draft_metadata(),
            )
        except Exception as e:
            logger.debug("abandon_open_draft failed (best-effort): %s", e)

    def _should_send_fresh_final(self) -> bool:
        """True when fresh-final is enabled and a real preview has been visible ≥ threshold."""
        threshold = getattr(self.cfg, "fresh_final_after_seconds", 0.0) or 0.0
        if threshold <= 0:
            return False
        if not self._message_id or self._message_id == "__no_edit__":
            return False
        if self._message_created_ts is None:
            return False
        age = time.monotonic() - self._message_created_ts
        return age >= threshold

    def _track_preview_id(self, message_id: Optional[str]) -> None:
        """Record a real preview message id for finalization cleanup."""
        if message_id and message_id != "__no_edit__":
            message_id = str(message_id)
            self._preview_message_ids.add(message_id)
            self._segment_preview_message_ids.add(message_id)

    def _track_preview_ids_from_result(self, result: Any) -> None:
        """Record the primary id plus any continuation ids from an oversized split."""
        self._track_preview_id(getattr(result, "message_id", None))
        for mid in (getattr(result, "continuation_message_ids", None) or ()):
            self._track_preview_id(mid)
        raw = getattr(result, "raw_response", None) or {}
        if isinstance(raw, dict):
            for mid in (raw.get("message_ids") or ()):
                self._track_preview_id(mid)

    def _adapter_prefers_fresh_final(self, text: str) -> bool:
        """Adapter's prefers_fresh_final_streaming hook (e.g. Telegram's richer send path).

        False when there's no real preview, no hook, or on any error.
        """
        if not self._message_id or self._message_id == "__no_edit__":
            return False
        fn = getattr(self.adapter, "prefers_fresh_final_streaming", None)
        if fn is None:
            return False
        try:
            try:
                # chat_id lets relay adapters decide via THIS chat's platform;
                # otherwise a Slack-primary relay misroutes fronted chats
                # through the fresh-send lane (duplicates: no delete op).
                result = fn(text, metadata=self.metadata, chat_id=self.chat_id)
            except TypeError:
                try:
                    result = fn(text, metadata=self.metadata)  # single-platform signature
                except TypeError:
                    result = fn(text)  # test doubles without the metadata kwarg
        except Exception as e:
            logger.debug("prefers_fresh_final_streaming check failed: %s", e)
            return False
        # ``is True`` keeps MagicMock auto-children from enabling fresh-final.
        return result is True

    async def _try_fresh_final(self, text: str, *, is_turn_final: bool = True) -> bool:
        """Send ``text`` as a fresh message and best-effort delete the preview(s).

        Returns False on any failure so the caller falls back to edit.
        ``is_turn_final=False`` (interim segment at a tool boundary) leaves the
        final-delivery flag unset so the gateway still delivers the real answer.
        """
        # Replacing every tracked preview is only sound while ``text`` holds
        # the whole answer; after a split, deleting sealed heads would erase
        # delivered text — take the edit path instead.
        if self._turn_split_delivery:
            return False
        stale_ids = self._stale_preview_ids()
        try:
            result = await self.adapter.send(
                chat_id=self.chat_id,
                content=text,
                metadata=self._metadata_for_send(final=True),
            )
        except Exception as e:
            logger.debug("Fresh-final send failed, falling back to edit: %s", e)
            return False
        if not getattr(result, "success", False):
            return False
        new_message_id = getattr(result, "message_id", None)
        # Best-effort preview cleanup; never delete the message just sent.
        await self._delete_previews(stale_ids, skip=new_message_id, label="Fresh-final")
        self._preview_message_ids = set()
        if new_message_id:
            self._message_id = new_message_id
            self._message_created_ts = time.monotonic()
        else:
            # No id returned: sentinel so we never try to edit it.
            self._message_id = "__no_edit__"
            self._message_created_ts = None
        self._already_sent = True
        self._last_sent_text = text
        if is_turn_final:
            self._final_response_sent = True
            self._record_turn_final_payload(text)
        return True

    async def _send_or_edit(
        self, text: str, *, finalize: bool = False, is_turn_final: bool = True,
    ) -> bool:
        """Send or edit the streaming message; True if delivered.

        ``finalize`` marks the last edit of a streaming sequence.  Callers such
        as the overflow split loop use the result to decide whether to advance.
        Transport order: native frame → draft frame → edit existing → first
        send; a transport returns None to fall through to the next.
        """
        text = self._clean_for_display(text)
        # Stream-is-the-message draft frames must stay prefix-stable: a closing
        # ``` appended to a mid-code-block frame makes frame N not a prefix of
        # N+1 and the connector re-appends the whole snapshot.  The final
        # message is still fence-closed below.
        pre_fence_text = text
        text = ensure_closed_code_fences(text)
        # A bare cursor renders as a stray tofu box on some clients.
        visible_stripped = text
        if self.cfg.cursor:
            visible_stripped = visible_stripped.replace(self.cfg.cursor, "")
        visible_stripped = visible_stripped.strip()
        if not visible_stripped:
            # Native streams MUST still get a finalize frame (placeholder) to
            # close the thinking bubble, e.g. for a MEDIA-only response.
            if finalize and self._use_native_streaming and self._native_stream_opened:
                try:
                    if await self._send_frame("✅", finalize=True):
                        self._mark_final_delivered()
                except Exception as e:
                    logger.debug("Finalize empty stream failed: %s", e)
            return True  # cursor-only / whitespace-only update
        if not text.strip():
            return True  # nothing to send is "success"
        # Don't open a new message for 1-2 tokens + cursor (rapid tool-calling):
        # if the cursor-strip edit is then rate-limited, "X ▉" stays forever.
        # Only first sends are gated.
        if (
            self._message_id is None
            and self.cfg.cursor
            and self.cfg.cursor in text
            and len(visible_stripped) < self._MIN_NEW_MSG_CHARS
        ):
            return True  # too short for a standalone message — accumulate more

        if self._use_native_streaming:
            ok = await self._native_push(text, finalize=finalize, is_turn_final=is_turn_final)
            if ok is not None:
                return ok
            # Fall through so accumulated text still reaches the user via edit/send.
        if self._use_draft_streaming and self._message_id is None:
            ok = await self._draft_push(
                text, pre_fence_text, finalize=finalize, is_turn_final=is_turn_final,
            )
            if ok is not None:
                return ok
            # Failure disabled drafts; fall through to edit/send.
        self._last_edit_overflowed = False
        try:
            if self._message_id is None:
                return await self._first_send(text, finalize=finalize)
            if not self._edit_supported:
                return False  # edits unsupported; fallback path sends the final
            return await self._edit_existing(text, finalize=finalize, is_turn_final=is_turn_final)
        except Exception as e:
            logger.error("Stream send/edit error: %s", e)
            return False

    async def _native_push(self, text: str, *, finalize: bool, is_turn_final: bool) -> Optional[bool]:
        """Native streaming: every frame goes through send_stream_frame().

        The adapter's send/edit paths are not touched in this mode.  Lazy
        re-seed here after a boundary closed the stream.  Returns None when
        native was disabled (seed/frame failure) so the caller falls through.
        """
        if not self._native_stream_opened and text:
            try:
                if await self._send_seed_frame():
                    self._native_stream_opened = True
                    self._awaiting_reopen_after_boundary = False
                    # Paired with the boundary-finalize INFO: typing-reappear latency.
                    logger.info(
                        "[latency] Re-opened native stream after boundary "
                        "(turn=%s, waited for first delta)",
                        self._turn_id,
                    )
                else:
                    self._use_native_streaming = False
            except Exception as e:
                logger.debug("Re-seed failed, disabling native streaming: %s", e)
                self._use_native_streaming = False
        if not self._use_native_streaming:
            return None

        # WeCom renders each finalize as a separate bubble: only the
        # turn-final and boundaries close the stream, not segment breaks.
        if finalize and not is_turn_final:
            finalize = False
        if not finalize and text == self._last_sent_text:
            return True  # unchanged — skip

        # Mark a finalize frame delivered OPTIMISTICALLY, before the ack wait:
        # the bytes hit the wire (and WeCom renders them) before the ack, so a
        # gateway join-cancel during the ack wait must not strand
        # final_content_delivered=False and cause a duplicate normal send
        # (docs/rca-wecom-stream-final-ack-timeout-duplicate.md).  A definitive
        # dispatch failure rolls the mark back below.  Residual window (cancel
        # between mark and wire write, sub-ms) is accepted.
        if finalize:
            # Recorded so a stale/partial frame can't suppress the corrective send.
            self._mark_final_delivered(record=text)
        try:
            ok = await self._send_frame(text, finalize=finalize)
        except Exception as e:
            logger.debug("send_stream_frame raised, disabling native streaming: %s", e)
            ok = False
        if ok:
            self._already_sent = True
            self._last_sent_text = text
            self._native_last_pushed_len = len(text)
            if finalize:
                self._mark_final_delivered()
            return True

        # Definitive failure: roll back the optimistic mark so the edit/send
        # fallback delivers exactly once.
        if finalize:
            self._final_response_sent = False
            self._final_content_delivered = False
            self._delivered_final_text = None
        # Subsequent frames take the edit/send fallback; the adapter marks the
        # chat expired so it doesn't retry the dead stream.
        self._use_native_streaming = False
        # Best-effort close of an opened bubble (seed frame has zero length but
        # still opens it — hence _native_stream_opened, not pushed_len).  DO NOT
        # mark delivered: the frame closes the bubble but WeCom may not render
        # the content (errcode 6000 race).
        if self._native_stream_opened:
            try:
                await self._send_frame(text, finalize=True)
                logger.debug("Native fallback: finalized stream (best-effort close)")
            except Exception as e:
                logger.debug("Native fallback: failed to finalize stream: %s", e)
        return None

    async def _draft_push(
        self, text: str, pre_fence_text: str, *, finalize: bool, is_turn_final: bool,
    ) -> Optional[bool]:
        """Draft frame while no message_id exists; None = not applicable / drafts just failed.

        Drafts have no message_id: the final answer goes through the regular
        send (which clears the draft client-side), so drafts are skipped when
        finalizing.  Exception: stream-is-the-message adapters keep ONE stream
        per turn, so a segment-break finalize must NOT become a real send
        (seal interception would seal the stream at every tool boundary);
        only got_done seals.
        """
        stream_is_msg = self._stream_is_message()
        if finalize and not (stream_is_msg and not is_turn_final):
            return None
        frame_text = pre_fence_text if stream_is_msg else text
        # Strip the cursor: native streams render their own indicator, and
        # "...text▉" is never a prefix of "...text more▉", which forces the
        # connector's whole-text re-append on EVERY tick (stacked copies).
        if self.cfg.cursor and frame_text.endswith(self.cfg.cursor):
            frame_text = frame_text[: -len(self.cfg.cursor)]
        if frame_text == self._last_sent_text:
            return True
        if await self._send_draft_frame(frame_text):
            # Deliberately NOT _already_sent: the gateway's fallback final send
            # must still fire so the user gets a real message.
            return True
        return None

    async def _first_send(self, text: str, *, finalize: bool) -> bool:
        """First send, threaded to the user's message (correct topic/thread)."""
        result = await self.adapter.send(
            chat_id=self.chat_id,
            content=text,
            reply_to=self._initial_reply_to_id,
            metadata=self._metadata_for_send(final=finalize, expect_edits=not finalize),
        )
        if not result.success:
            self._edit_supported = False
            return False
        if result.message_id:
            self._message_id = result.message_id
            self._message_created_ts = time.monotonic()
            self._track_preview_ids_from_result(result)
        else:
            self._edit_supported = False
        self._already_sent = True
        self._last_sent_text = text
        if not result.message_id:
            self._fallback_prefix = self._visible_prefix()
            self._fallback_final_send = True
            # Sentinel: no editable id, don't re-enter first-send on every
            # delta/tool boundary.
            self._message_id = "__no_edit__"
        self._notify_new_message()
        return True

    async def _edit_existing(self, text: str, *, finalize: bool, is_turn_final: bool) -> bool:
        """Edit the live preview (or replace it via fresh-final when finalizing)."""
        # REQUIRES_EDIT_FINALIZE adapters need the finalize=True edit even
        # when unchanged; everyone else short-circuits.
        if text == self._last_sent_text and not (finalize and self._adapter_requires_finalize):
            return True
        # Fresh-final: replace a long-lived preview with a fresh message
        # (timestamp reflects completion), or whenever the adapter prefers it
        # (Telegram's send path renders richer markdown than its edit path).
        # An explicit hook returning False must NOT be overridden by the time
        # threshold — on Telegram both messages would stay on screen since the
        # delete is best-effort.  Check the CLASS (MagicMock auto-creates
        # attrs) plus instance __dict__ (test doubles assign the hook explicitly).
        has_prefers_hook = (
            hasattr(type(self.adapter), "prefers_fresh_final_streaming")
            or "prefers_fresh_final_streaming" in getattr(self.adapter, "__dict__", {})
        )
        prefers_fresh = self._adapter_prefers_fresh_final(text)
        if (
            finalize
            and (prefers_fresh or (not has_prefers_hook and self._should_send_fresh_final()))
            and await self._try_fresh_final(text, is_turn_final=is_turn_final)
        ):
            return True
        result = await self._edit_message(
            message_id=self._message_id, content=text, finalize=finalize,
        )
        if not result.success:
            return await self._on_edit_failure(result, text, finalize=finalize, is_turn_final=is_turn_final)
        self._already_sent = True
        self._track_preview_ids_from_result(result)
        # Oversized edit split across continuations: message_id is now the
        # LAST continuation, which holds only the final chunk — retarget edits
        # and reset skip-if-same.  getattr keeps SimpleNamespace test mocks working.
        continuation_ids = getattr(result, "continuation_message_ids", ()) or ()
        if continuation_ids and result.message_id and result.message_id != self._message_id:
            self._last_edit_overflowed = True
            self._turn_split_delivery = True
            self._message_id = str(result.message_id)
            self._message_created_ts = time.monotonic()
            self._last_sent_text = ""
            self._notify_new_message()
        else:
            self._last_sent_text = text
        self._flood_strikes = 0
        return True

    async def _on_edit_failure(self, result, text: str, *, finalize: bool, is_turn_final: bool) -> bool:
        """Classify a failed edit: partial overflow, flood backoff, or fallback mode.

        Returns the _send_or_edit result (always False here; the caller's
        finalize path may still deliver the tail via _send_fallback_final).
        """
        immediate_final_fallback = False
        if (
            finalize
            and is_turn_final
            and self.cfg.cursor
            and self._last_sent_text.endswith(self.cfg.cursor)
            and self._visible_prefix() == text
        ):
            # Cosmetic final edit was rate-limited but the full answer is
            # already on screen (cursor stuck): mark delivered so the gateway
            # doesn't send it twice, and record the on-screen payload.
            self._final_content_delivered = True
            self._record_turn_final_payload(text)
        raw_response = getattr(result, "raw_response", None)
        if isinstance(raw_response, dict) and raw_response.get("partial_overflow"):
            # Some overflow chunks landed but not the whole response: preserve
            # the visible prefix so got_done sends the missing tail.
            self._message_id = str(
                raw_response.get("last_message_id") or result.message_id or self._message_id
            )
            delivered_prefix = raw_response.get("delivered_prefix")
            if isinstance(delivered_prefix, str) and delivered_prefix:
                self._last_sent_text = delivered_prefix
                self._fallback_prefix = delivered_prefix
                self._fallback_preserve_partial_messages = text.startswith(delivered_prefix)
            else:
                self._fallback_prefix = self._visible_prefix()
                self._fallback_preserve_partial_messages = False
            self._fallback_final_send = True
            self._edit_supported = False
            self._already_sent = True
            if getattr(result, "continuation_message_ids", ()):
                self._notify_new_message()
            return False

        # Flood control: adaptive backoff (double the interval); disable edits
        # only after _MAX_FLOOD_STRIKES in a row.
        if self._is_flood_error(result):
            self._flood_strikes += 1
            self._current_edit_interval = min(self._current_edit_interval * 2, 10.0)
            logger.debug(
                "Flood control on edit (strike %d/%d), backoff interval → %.1fs",
                self._flood_strikes, self._MAX_FLOOD_STRIKES, self._current_edit_interval,
            )
            immediate_final_fallback = (
                finalize
                and is_turn_final
                and getattr(self.adapter, "FALLBACK_ON_FINAL_EDIT_FLOOD", False) is True
            )
            if self._flood_strikes < self._MAX_FLOOD_STRIKES and not immediate_final_fallback:
                self._last_edit_time = time.monotonic()  # honor the new interval
                return False
            if immediate_final_fallback:
                logger.debug("Turn-final edit hit flood control; entering fallback immediately")

        # Fallback mode: send only the missing tail at got_done.
        logger.debug("Edit failed (strikes=%d), entering fallback mode", self._flood_strikes)
        self._fallback_prefix = self._visible_prefix()
        self._fallback_final_send = True
        self._edit_supported = False
        self._already_sent = True
        # A turn-final flood skips the cosmetic cursor strip: it would burn the
        # same flood budget and delay the answer.
        if not immediate_final_fallback:
            await self._try_strip_cursor()
        return False
