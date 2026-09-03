"""Gateway streaming consumer — bridges sync agent callbacks to async platform delivery.

The agent fires stream_delta_callback(text) synchronously from its worker thread;
on_delta() queues it (queue.Queue) and the async run() task buffers, rate-limits,
and progressively edits a single platform message (send, then editMessageText —
supported everywhere; draft/native transports are optional per adapter).

Credit: jobless0x (#774, #1312), OutThisLife (#798), clicksingh (#697).
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import inspect
import logging
import queue
import secrets
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

from gateway.platforms.base import BasePlatformAdapter as _BasePlatformAdapter
from gateway.platforms.base import _custom_unit_to_cp
from gateway.config import (
    DEFAULT_STREAMING_EDIT_INTERVAL as _DEFAULT_STREAMING_EDIT_INTERVAL,
    DEFAULT_STREAMING_BUFFER_THRESHOLD as _DEFAULT_STREAMING_BUFFER_THRESHOLD,
    DEFAULT_STREAMING_CURSOR as _DEFAULT_STREAMING_CURSOR,
)
from gateway.response_filters import (
    is_intentional_silence_response as _is_intentional_silence_response,
    is_partial_silence_marker as _is_partial_silence_marker,
)
from gateway.stream_consumer_fences import (  # noqa: F401  (re-exported)
    ensure_closed_code_fences,
    escape_code_fences_for_display,
)
from gateway.stream_consumer_transport import StreamTransportMixin
from gateway.stream_consumer_fallback import StreamFallbackMixin
from gateway.stream_consumer_think import StreamThinkFilterMixin
import contextlib

logger = logging.getLogger("gateway.stream_consumer")

# Queue sentinels (see run() for handling).
_DONE = object()          # stream complete
_NEW_SEGMENT = object()   # finalize current message, start a fresh one
_COMMENTARY = object()    # (_COMMENTARY, text): completed interim commentary
_TOOL_PROGRESS = object()  # (_TOOL_PROGRESS, line): overlay line for the native bubble
# (_FINAL_TEXT, text): authoritative completed final_response (incl. post-stream
# augmentation such as verifier footers) enqueued just before _DONE, so the
# finalize/seal delivers the TRUE final and the recorded payload reconciles.
_FINAL_TEXT = object()
# (_FLUSH, threading.Event): barrier — drain loop finalizes/delivers anything
# buffered, then sets the event.  Used by flush_pending_sync() before a blocking
# interactive prompt so the prompt lands below buffered prose, not above it.
_FLUSH = object()
# Interaction boundary (approval OR clarify prompt): finalize the current
# stream; post-prompt output goes via send() or a re-opened stream.
_APPROVAL_BOUNDARY = object()
# EAGER native re-seed right after a clarify answer, before the first post-answer
# delta.  On WeCom typing is driven by the seed frame (send_typing is a no-op)
# and lazy re-seed on first delta measured 48s of dead air.
_REOPEN_SEED = object()

# Boundary finalize text when nothing has accumulated yet; overridable per
# boundary via close_for_approval_prompt(placeholder=...).
_DEFAULT_BOUNDARY_PLACEHOLDER = "⏸ 等待审批中..."

@dataclass
class StreamConsumerConfig:
    """Runtime config for a single stream consumer instance."""
    edit_interval: float = _DEFAULT_STREAMING_EDIT_INTERVAL
    buffer_threshold: int = _DEFAULT_STREAMING_BUFFER_THRESHOLD
    cursor: str = _DEFAULT_STREAMING_CURSOR
    buffer_only: bool = False
    # When >0, deliver the final as a fresh message if the preview has been
    # visible at least this long, so the visible timestamp reflects completion
    # rather than first token.  0 = always edit in place.  Enabled per-platform.
    fresh_final_after_seconds: float = 0.0
    # "auto"/"draft": native draft streaming when adapter+chat support it, else
    # edit.  "edit": progressive editMessageText.  "off": handled by the gateway
    # before the consumer is built.
    transport: str = "edit"
    # Originating chat type ("dm", "group", ...); gates draft streaming, which
    # is platform-specific (Telegram drafts are DM-only).
    chat_type: str = ""

@dataclass
class _Tick:
    """Everything one drain of the queue decided (flag locals of the old run() loop)."""
    got_done: bool = False
    got_segment_break: bool = False
    got_flush: bool = False
    flush_event: Any = None
    got_reopen_seed: bool = False
    approval_boundary: Optional[tuple] = None  # (future, cancelled_flag)
    commentary_text: Optional[str] = None
    # Set by _push_update for _finalize_turn / _end_segment.
    update_visible: bool = False
    draft_final_fresh_send: bool = False

    @property
    def is_interim(self) -> bool:
        """Mid-stream tick: not finalizing, not a segment break, no commentary."""
        return not self.got_done and not self.got_segment_break and self.commentary_text is None

class GatewayStreamConsumer(
    StreamTransportMixin,
    StreamFallbackMixin,
    StreamThinkFilterMixin,
):
    """Async consumer that progressively edits a platform message with streamed tokens.

    Usage::

        consumer = GatewayStreamConsumer(adapter, chat_id, config, metadata=metadata)
        agent = AIAgent(..., stream_delta_callback=consumer.on_delta)
        task = asyncio.create_task(consumer.run())
        # ... run agent in thread pool ...
        consumer.finish()  # signal completion
        await task         # wait for final edit
    """

    # Consecutive flood-control failures before progressive edits are disabled
    # for the rest of the stream.
    _MAX_FLOOD_STRIKES = 3

    # Class-wide monotonic counter for draft ids (Telegram animates a draft only
    # when the same non-zero draft_id is reused within a response).  Seeded from
    # a RANDOM nonce, not 0 or the clock: draft_id is the wire identity for the
    # relay connector's per-(channel, draft_id) sealed-stream tombstones, which
    # outlive this (scale-to-zero) process — a replayed id is answered out of
    # the OLD tombstone and the reply is silently dropped.  Epoch-ms seeds still
    # collide on same-ms starts/forks/clock steps; 49 bits keeps ids + turn
    # counts inside the connector's JS number range (2^53).
    _draft_id_counter: int = secrets.randbits(49)

    def __init__(
        self,
        adapter: Any,
        chat_id: str,
        config: Optional[StreamConsumerConfig] = None,
        metadata: Optional[dict] = None,
        on_new_message: Optional[callable] = None,
        on_before_finalize: Optional[Callable[[], Any]] = None,
        initial_reply_to_id: Optional[str] = None,
        run_still_current: Optional[Callable[[], bool]] = None,
    ):
        self.adapter = adapter
        self.chat_id = chat_id
        self.cfg = config or StreamConsumerConfig()
        self.metadata = metadata
        # Fired whenever a fresh content bubble is created (first send,
        # commentary, overflow chunk, fallback continuation) so the gateway
        # opens the next tool-progress bubble BELOW it.  Exceptions swallowed.
        self._on_new_message = on_new_message
        # Fired once on entering finalization so the gateway can pause typing
        # refreshes before a slow rich-text final edit.
        self._on_before_finalize = on_before_finalize
        self._initial_reply_to_id = initial_reply_to_id
        # Per-turn id for adapter.send_stream_frame() so concurrent consumers
        # (/background, parallel subagents) don't interfere.
        import uuid
        self._turn_id = str(uuid.uuid4())
        # Returns False after /new or /stop; run() then abandons the stream.
        self._run_still_current = run_still_current or (lambda: True)
        # Only platforms needing an explicit finalize call (DingTalk AI Cards)
        # force a redundant final edit; ``is True`` keeps MagicMock adapters out.
        self._adapter_requires_finalize: bool = (
            getattr(adapter, "REQUIRES_EDIT_FINALIZE", False) is True
        )
        # Telegram bounds edit retries at 5s; a final-delivery fallback must not
        # hold the stream task through a longer flood cooldown.
        self._max_fallback_flood_retry_seconds = 5.0

        self._queue: queue.Queue = queue.Queue()
        self._accumulated = ""
        # Mirror of ``_accumulated`` NOT truncated when overflow splits seal
        # head chunks; records a reconcilable turn-final payload for splits.
        self._stream_ledger = ""
        self._message_id: Optional[str] = None
        # monotonic() when ``_message_id`` was first assigned (fresh-final age).
        self._message_created_ts: Optional[float] = None
        # Every real preview id on screen this response (first send +
        # continuations): fresh-final deletes them all so a reply split at the
        # edit limit leaves no stale fragments above the final.  The segment
        # set holds only the active text segment: failure recovery must never
        # delete an earlier finalized preamble/commentary message.
        self._preview_message_ids: "set[str]" = set()
        self._segment_preview_message_ids: "set[str]" = set()
        self._already_sent = False
        self._edit_supported = True  # False once progressive edits stop working
        self._last_edit_time = 0.0
        self._last_sent_text = ""    # skip redundant edits
        # Most recent _send_or_edit split across continuation messages (the
        # adapter adopted a new message id).
        self._last_edit_overflowed = False
        self._fallback_final_send = False
        self._fallback_prefix = ""
        # Fallback sends only the missing tail after a partial overflow
        # delivery: the visible prefix is content, not a stale preview.
        self._fallback_preserve_partial_messages = False
        self._flood_strikes = 0         # consecutive flood-control edit failures
        self._current_edit_interval = self.cfg.edit_interval  # adaptive backoff
        self._final_response_sent = False
        # Final content reached the user even if the cosmetic final edit
        # (cursor removal) then failed.
        self._final_content_delivered = False
        # Exact cleaned payload of the turn-final delivery that set the flags
        # above; the gateway compares it to the completed final_response before
        # trusting the flags (a successful finalize edit may carry only a stale
        # preview snapshot).  ``None`` = no record → legacy trust.
        self._delivered_final_text: Optional[str] = None
        # Answer delivered across multiple sealed messages (overflow split /
        # continuation adoption).  Payload-less split delivery must NOT inherit
        # legacy trust (it swallowed complete replies after partial splits).
        self._turn_split_delivery = False
        # A full-final send timed out in a way that MAY have reached the
        # platform — the only payload-less case that keeps legacy trust, since
        # re-sending risks a duplicate rather than recovering a loss.
        self._delivery_ambiguous = False
        self._delivered_commentary_texts: list[str] = []
        # Finalized visible text per segment, so has_delivered_text still
        # matches after _reset_segment_state clears _last_sent_text.
        self._delivered_segment_texts: list[str] = []
        # Think-block filter state (mirrors CLI's _stream_delta tag suppression).
        self._in_think_block = False
        self._think_buffer = ""
        self._before_finalize_notified = False

        # Transports, resolved at the start of run().  Draft: animated frames
        # via adapter.send_draft instead of edits; the final still uses the
        # first-send path (drafts have no message_id); the first draft failure
        # disables drafts for the response.  Native (WeCom msgtype "stream"):
        # the ONLY delivery channel — seed, cumulative updates and finish=true
        # all go through send_stream_frame(); any failure falls back to edit/send.
        self._use_draft_streaming = False
        self._draft_id: Optional[int] = None
        self._draft_failures = 0
        self._use_native_streaming = False
        # Seed frame sent (zero visible content but the bubble is open); the
        # fallback decides from this whether the stream must be finalized first.
        self._native_stream_opened = False
        # Visible chars last pushed; throttles under WeCom's 30 frames/min.
        self._native_last_pushed_len = 0
        # Boundary state from close_for_approval_prompt(); race-free because
        # boundaries are processed serially.  ``_boundary_reopen`` keeps native
        # enabled so post-prompt output re-opens a fresh stream via the lazy
        # re-seed (clarify: short waits) instead of degrading to send()
        # (approval: unbounded waits, stream may go stale).
        self._boundary_placeholder = _DEFAULT_BOUNDARY_PLACEHOLDER
        self._boundary_reason = "Approval"
        self._boundary_reopen = False
        # Reopen requested but nothing re-seeded yet: got_done must not open a
        # stream just to emit a lone "✅".  An EAGER re-seed (_REOPEN_SEED)
        # already opened a fresh bubble before any content: got_done must
        # actively finalize it or a blank typing bubble hangs forever.
        self._awaiting_reopen_after_boundary = False
        self._reopen_seeded_eagerly = False
        # Tool-progress overlay (native only): shown in the bubble until text arrives.
        self._tool_progress_lines: list[str] = []
        self._tool_progress_active: bool = False

    def _stream_is_message(self) -> bool:
        """Whether THIS chat's transport treats the stream as the message.

        Prefers the adapter's per-chat probe (a multi-platform relay adapter's
        class attribute can only reflect its primary identity), falling back to
        the legacy attribute.  Both are resolved on the CLASS to stay
        MagicMock-safe (auto-created instance attributes are truthy).
        """
        probe = getattr(type(self.adapter), "stream_is_message_for_chat", None)
        if callable(probe):
            try:
                return probe(self.adapter, str(self.chat_id)) is True
            except Exception:
                return False
        return getattr(self.adapter, "draft_stream_is_message", False) is True

    @property
    def accepts_tool_progress(self) -> bool:
        """True only when native streaming is active (gates in-stream tool progress)."""
        return self._use_native_streaming

    def on_tool_progress(self, line: str) -> None:
        """Thread-safe: overlay a tool-progress line in the native bubble until the next delta."""
        if line:
            self._queue.put((_TOOL_PROGRESS, line))

    def _compose_frame_content(self) -> str:
        """Native frame content: text, with any tool-progress lines below a rule."""
        if self._accumulated and self._tool_progress_lines:
            return self._accumulated + "\n\n---\n" + "\n".join(self._tool_progress_lines)
        elif self._accumulated:
            return self._accumulated
        elif self._tool_progress_lines:
            return "\n".join(self._tool_progress_lines)
        return ""

    def _metadata_for_send(
        self,
        *,
        final: bool = False,
        expect_edits: bool = False,
    ) -> dict | None:
        """Per-send metadata for stream-created messages.

        ``final`` sets notify=True (Mattermost treats notify-worthy sends as
        final content when deciding whether a broken thread root may fall back
        flat).  ``expect_edits`` keeps editable previews on Telegram's legacy
        send path while final sends may use richer delivery.
        """
        meta = dict(self.metadata) if self.metadata else {}
        if self._initial_reply_to_id:
            meta["reply_to_message_id"] = self._initial_reply_to_id
        if expect_edits:
            meta["expect_edits"] = True
        if final:
            meta["notify"] = True
        return meta or None

    @property
    def already_sent(self) -> bool:
        """True if at least one message was sent or edited during the run."""
        return self._already_sent

    @property
    def final_response_sent(self) -> bool:
        """True when the stream consumer delivered the final assistant reply."""
        return self._final_response_sent

    @property
    def message_id(self) -> str | None:
        """Message ID of the last-sent or edited message."""
        return self._message_id

    @property
    def final_content_delivered(self) -> bool:
        """True when the final content reached the user, even if the cosmetic final edit failed."""
        return self._final_content_delivered

    async def _notify_before_finalize(self) -> None:
        """Run the pre-finalize hook exactly once, swallowing hook errors."""
        if self._before_finalize_notified:
            return
        self._before_finalize_notified = True
        if self._on_before_finalize is None:
            return
        try:
            result = self._on_before_finalize()
            if inspect.isawaitable(result):
                await result
        except Exception:
            pass

    def _append_accumulated(self, text: str) -> None:
        """Append to the live buffer and the split-stable stream ledger."""
        if not text:
            return
        # Real text overwrites the tool-progress overlay.
        if self._tool_progress_lines:
            self._tool_progress_lines.clear()
            self._tool_progress_active = False
        self._accumulated += text
        self._stream_ledger += text

    def _mark_skip_redundant_finalize(self) -> None:
        """Mark the turn final as delivered by a prior mid-stream edit.

        Records what was ACKED on the wire, not ``_accumulated``: a throttled
        edit stream can reach this state with the last acked edit still holding
        an earlier (cursor-suffixed) preview, and recording the accumulator
        would let that frozen preview suppress the corrective send.
        """
        self._mark_final_delivered()
        acked = self._last_sent_text or self._accumulated
        if self.cfg.cursor and acked.endswith(self.cfg.cursor):
            acked = acked[: -len(self.cfg.cursor)]
        self._record_turn_final_payload(acked)

    def _mark_final_delivered(self, record: Optional[str] = None) -> None:
        """Set both turn-final flags; ``record`` also records the delivered payload."""
        self._final_response_sent = True
        self._final_content_delivered = True
        if record is not None:
            self._record_turn_final_payload(record)

    def _record_turn_final_payload(self, text: str) -> None:
        """Record what the user actually saw as this turn's final answer.

        Normalized like ``_send_or_edit`` output (media-directive strip + fence
        closing) so the gateway can compare it to the completed final_response.
        On a multi-message split ``text`` is only the trailing chunk (overflow
        truncates ``_accumulated`` as head chunks seal), so the un-truncated
        ``_stream_ledger`` is recorded instead — else the gateway sees a
        mismatch and re-sends an answer the user already received.
        """
        source = text or ""
        if self._turn_split_delivery and self._stream_ledger:
            source = self._stream_ledger
        self._delivered_final_text = ensure_closed_code_fences(
            self._clean_for_display(source)
        ).strip()

    def delivered_final_matches(self, final_text: str) -> Optional[bool]:
        """Tri-state reconcile of the recorded turn-final payload against ``final_text``.

        A *successful* finalize edit can still carry only a stale preview, so
        call success alone must not confirm delivery.
        True: recorded payload (or an earlier segment/commentary) matches —
        suppressing the normal final send is safe.  False: recorded payload
        differs, or payload-less multi-message split (flag alone must not
        suppress).  None: nothing recorded on a non-split legacy/ambiguous path;
        caller keeps its flag-trusting behavior.
        """
        target = ensure_closed_code_fences(
            self._clean_for_display(final_text or "")
        ).strip()
        if not target:
            return None
        if self._delivered_final_text is None:
            if self._turn_split_delivery:
                return False
            # No recorded payload: judge against the FINAL content rather than
            # trusting the flag — a consumer whose visible text lacks the
            # completed response (first-edit prefix, mid-stream truncation) has
            # demonstrably NOT delivered it.  ``_already_sent`` gates the match:
            # draft frames set ``_last_sent_text`` but are ephemeral and
            # deliberately don't set ``_already_sent``.
            if self._already_sent and self.has_delivered_text(final_text):
                return True
            # Only a timed-out full-final send that MAY have landed keeps legacy
            # trust — re-sending risks a duplicate.
            if self._delivery_ambiguous:
                return None
            return False
        if self._delivered_final_text.strip() == target:
            return True
        # A segment break / commentary may have delivered it under another record.
        return bool(self.has_delivered_text(final_text))

    def has_delivered_text(self, text: str) -> bool:
        """Return True if *text* was already delivered as visible chat content."""
        target = self._clean_for_display(text or "").strip()
        if not target:
            return False
        visible_prefix = self._visible_prefix().strip()
        if visible_prefix == target:
            return True
        return any(
            sent.strip() == target
            for sent in (*self._delivered_commentary_texts, *self._delivered_segment_texts)
        )

    def on_segment_break(self) -> None:
        """Finalize the current stream segment and start a fresh message."""
        self._queue.put(_NEW_SEGMENT)

    def close_for_approval_prompt(
        self,
        placeholder: str | None = None,
        reason: str = "Approval",
        reopen: bool = False,
    ) -> asyncio.Future:
        """Queue an interaction boundary (approval / clarify prompt) from sync context.

        run() processes it serially: finalize the current native stream with
        accumulated text (``placeholder`` when nothing accumulated yet), then
        per ``reopen``: False (approval; long unbounded waits, stream may go
        stale) disables native streaming and batches post-prompt output into
        one send(); True (clarify) keeps native enabled so post-prompt output
        re-opens a fresh stream via the lazy re-seed, degrading to send() if
        that fails.  ``reason`` labels log lines ("Approval"/"Clarify").

        Returns a (Future, cancelled_flag) tuple; the Future resolves True once
        processed.  cancelled_flag is kept for callers that set it on timeout
        but the handler no longer reads it (finalize always runs).  Without
        native streaming returns a bare, already-resolved Future.
        """
        loop = None
        with contextlib.suppress(RuntimeError):
            loop = asyncio.get_running_loop()

        if not self._use_native_streaming:
            f = asyncio.Future() if loop else concurrent.futures.Future()
            f.set_result(True)
            return f

        # Instance attributes are race-free: boundaries are processed one at a time.
        self._boundary_placeholder = placeholder or _DEFAULT_BOUNDARY_PLACEHOLDER
        self._boundary_reason = reason or "Approval"
        self._boundary_reopen = bool(reopen)

        boundary_future = loop.create_future() if loop else concurrent.futures.Future()

        cancelled_flag = {"cancelled": False}
        self._queue.put((_APPROVAL_BOUNDARY, boundary_future, cancelled_flag))
        return boundary_future, cancelled_flag

    def on_commentary(self, text: str) -> None:
        """Queue a completed interim assistant commentary message."""
        if text:
            self._queue.put((_COMMENTARY, text))

    def flush_pending_sync(self, timeout: float = 5.0) -> bool:
        """Block the agent worker thread until everything queued so far is delivered.

        Enqueues a ``(_FLUSH, Event)`` barrier; run() drains earlier items
        (FIFO), finalizes the current segment, then sets the event.  Returns
        False on timeout so the caller proceeds even if the consumer task is
        not running.  Used before a blocking interactive prompt (clarify poll)
        so the question doesn't land ABOVE its own explanation.
        """
        evt = threading.Event()
        try:
            self._queue.put((_FLUSH, evt))
        except Exception:
            return False
        return evt.wait(timeout=max(0.0, float(timeout)))

    def request_reopen_seed(self) -> None:
        """Thread-safe: request an EAGER native re-seed after a clarify answer.

        Posts _REOPEN_SEED so run() sends an empty seed frame before the first
        post-answer delta (WeCom typing bubble reappears immediately).  No-op
        unless reopen-pending on a native stream with no stream open, so a
        stray call can't open a spurious bubble mid-stream or on approval.
        """
        if (
            self._use_native_streaming
            and self._awaiting_reopen_after_boundary
            and not self._native_stream_opened
        ):
            self._queue.put(_REOPEN_SEED)

    def _notify_new_message(self) -> None:
        """Fire the on_new_message callback, swallowing any errors."""
        cb = self._on_new_message
        if cb is None:
            return
        try:
            cb()
        except Exception:
            logger.debug("on_new_message callback error", exc_info=True)

    @staticmethod
    def _signal_flush(flush_event) -> None:
        """Wake a thread blocked in flush_pending_sync(), swallowing errors.

        Every loop path that consumed a ``_FLUSH`` barrier (including early
        ``continue`` paths) must call this; a missed set isn't a deadlock
        (bounded wait) but stalls the caller for the full timeout.
        """
        if flush_event is None:
            return
        with contextlib.suppress(Exception):
            flush_event.set()

    def _reset_segment_state(self, *, preserve_no_edit: bool = False) -> None:
        if preserve_no_edit and self._message_id == "__no_edit__":
            return
        # Retain the segment's visible text so has_delivered_text still matches.
        if self._last_sent_text:
            finalized = self._clean_for_display(self._last_sent_text).strip()
            if finalized:
                self._delivered_segment_texts.append(finalized)
        self._message_id = None
        self._message_created_ts = None
        self._accumulated = ""
        self._stream_ledger = ""
        self._last_sent_text = ""
        self._fallback_final_send = False
        self._fallback_prefix = ""
        self._fallback_preserve_partial_messages = False
        self._segment_preview_message_ids = set()
        self._tool_progress_lines = []
        self._tool_progress_active = False
        # A segment boundary means what we delivered was an interim preamble —
        # clear the final flags so a premature setter can't fool the gateway.
        # Safe: got_done returns before any reset; run.py reads these only
        # after the consumer task exits.
        self._final_response_sent = False
        self._final_content_delivered = False
        self._delivered_final_text = None
        self._delivery_ambiguous = False
        self._turn_split_delivery = False
        # Telegram-shaped drafts: bump draft_id so the next segment animates as
        # a fresh preview below the tool-progress bubbles instead of over the
        # prior finalized draft.  Stream-is-the-message adapters (relay Slack)
        # keep ONE stream per turn — a bump there opened a new platform stream
        # per tool boundary, leaving one frozen cursor-suffixed message per
        # segment; the connector's suffix-delta logic appends segments instead.
        if self._use_draft_streaming and not self._stream_is_message():
            type(self)._draft_id_counter += 1
            self._draft_id = type(self)._draft_id_counter

    async def _handle_approval_boundary(self, boundary_future, cancelled_flag=None) -> None:
        """Serially process an interaction boundary dequeued by run().

        Finalize the current stream with accumulated text (stable message for
        pre-prompt content), then route post-prompt output per
        ``_boundary_reopen``.  The stream is not kept open across approval
        because the WeCom finalize ack only confirms server receipt: after a
        long idle gap the client may stop tracking the stream, and a suppressed
        final send would then leave the user with nothing.
        """
        _reason = self._boundary_reason or "Approval"
        try:
            boundary_ok = True
            if self._native_stream_opened:
                boundary_ok = await self._finalize_boundary_stream(_reason)
            if self._boundary_reopen:
                # Clarify: keep native enabled; marking the stream closed makes
                # the next post-prompt delta re-open a fresh one via the lazy
                # re-seed in _send_or_edit.  Do NOT set buffer_only — post-prompt
                # output should stream.  The gap between this INFO and the
                # "Re-opened native stream" INFO is the typing-reappear latency.
                self._close_native_state()
                self._awaiting_reopen_after_boundary = True
                self._reset_segment_state()
                logger.info(
                    "[latency] Clarify boundary finalized, awaiting first "
                    "post-answer delta to re-seed (chat=%s, turn=%s)",
                    self.chat_id, self._turn_id,
                )
            else:
                # Approval: post-approval output goes via one send() at got_done.
                self._degrade_native_to_buffered_send()
                self._reset_segment_state()
        except Exception as e:
            logger.warning("%s boundary processing failed: %s", _reason, e)
            boundary_ok = False
        finally:
            if isinstance(boundary_future, (asyncio.Future, concurrent.futures.Future)):
                with contextlib.suppress(Exception):
                    if not boundary_future.done():
                        boundary_future.set_result(boundary_ok)

    async def _finalize_boundary_stream(self, _reason: str) -> bool:
        """Close the open native stream at a boundary; send() the pre-prompt text if that fails.

        Returns False only when both finalize and the fallback send failed
        (pre-prompt text may not have been delivered).
        """
        finalize_text = self._accumulated or self._boundary_placeholder
        finalize_ok = False
        try:
            finalize_ok = bool(await self._send_frame(finalize_text, finalize=True))
        except Exception as e:
            logger.warning("%s boundary: finalize failed: %s", _reason, e)
        if finalize_ok:
            logger.debug(
                "%s boundary: finalized stream (chat=%s, turn=%s)",
                _reason, self.chat_id, self._turn_id,
            )
            return True
        # Typing bubble may still show partial content; deliver pre-prompt
        # text via send() so the user at least sees it.
        logger.warning(
            "%s boundary: finalize not confirmed, "
            "falling back to send() for pre-prompt text (chat=%s)",
            _reason, self.chat_id,
        )
        fallback_ok = False
        try:
            send_result = await self.adapter.send(self.chat_id, finalize_text)
            fallback_ok = getattr(send_result, "success", False)
        except Exception as send_err:
            logger.warning("%s boundary: fallback send also failed: %s", _reason, send_err)
        if not fallback_ok:
            logger.error(
                "%s boundary: both finalize and fallback send failed "
                "(chat=%s) — pre-prompt text may not have been delivered",
                _reason, self.chat_id,
            )
        return fallback_ok

    def on_delta(self, text: str) -> None:
        """Thread-safe callback — called from the agent's worker thread.

        When *text* is ``None``, signals a tool boundary: the current message
        is finalized and subsequent text will be sent as a new message so it
        appears below any tool-progress messages the gateway sent in between.
        """
        if text:
            self._queue.put(text)
        elif text is None:
            self.on_segment_break()

    def finish(self, final_text: Optional[str] = None) -> None:
        """Signal stream completion.

        ``final_text`` is the AUTHORITATIVE completed final_response (incl.
        post-stream augmentation the accumulator never saw); the drain loop
        adopts it as the finalize payload so no corrective send is needed.
        Interrupt/error paths that can't know the final call ``finish()`` bare.
        """
        if final_text is not None:
            self._queue.put((_FINAL_TEXT, final_text))
        self._queue.put(_DONE)

    async def run(self) -> None:
        """Async task that drains the queue and edits the platform message."""
        self._len_fn, self._safe_limit = self._resolve_length_budget()
        await self._start_transports()
        try:
            while True:
                # Session reset (/new, /stop): abandon rather than deliver stale deltas.
                if not self._run_still_current():
                    await self._abandon_native_stream()
                    return
                tick = self._drain_queue()

                # Boundary produces its own finalize and resets state, so it
                # must run before got_done/segment_break processing.
                if tick.approval_boundary is not None:
                    await self._handle_approval_boundary(*tick.approval_boundary)
                    continue
                if tick.got_reopen_seed:
                    await self._eager_reopen_seed()
                    continue

                if tick.got_done:
                    self._flush_think_buffer()
                    # A bare intentional-silence marker (NO_REPLY / [SILENT]):
                    # the gateway's whole-response filter runs too late for a
                    # streamed preview, so retract it here instead of finalizing.
                    if _is_intentional_silence_response(
                        self._clean_for_display(self._accumulated)
                    ):
                        await self._suppress_silence_marker()
                        return

                if self._should_edit(tick) and (
                    self._accumulated
                    or (self._use_native_streaming and self._tool_progress_active)
                ):
                    # Overflow split.  Native streaming bypasses this: the
                    # adapter truncates against the stream protocol's own limit.
                    if (
                        not self._use_native_streaming
                        and self._len_fn(self._accumulated) > self._safe_limit
                        and self._message_id is None
                    ):
                        verdict = await self._split_first_send(tick)
                        if verdict == "return":
                            return
                        continue
                    await self._seal_overflow_heads()
                    await self._push_update(tick)

                if tick.got_done:
                    await self._finalize_turn(tick)
                    return

                if tick.commentary_text is not None:
                    await self._deliver_commentary(tick.commentary_text)
                if tick.got_segment_break:
                    await self._end_segment(tick)

                # Done last so the waiter unblocks only once everything queued
                # before the barrier is on screen.
                if tick.got_flush:
                    self._signal_flush(tick.flush_event)

                await asyncio.sleep(0.05)  # Small yield to not busy-loop

        except asyncio.CancelledError:
            await self._on_cancelled()
        except Exception as e:
            logger.error("Stream consumer error: %s", e)
        finally:
            self._wake_flush_waiters()

    # ── run() collaborators ─────────────────────────────────────────────

    def _resolve_length_budget(self) -> "tuple[Callable[[str], int], int]":
        """Per-chat length function + overflow budget.

        A relay adapter fronting N platforms has different caps per chat, in
        the platform's unit (e.g. utf16 for Telegram).  isinstance gate:
        MagicMock auto-attributes aren't callables, so test doubles use len.
        """
        len_fn: "Callable[[str], int]" = (
            self.adapter.message_len_fn_for_chat(self.chat_id)
            if isinstance(self.adapter, _BasePlatformAdapter)
            else len
        )
        raw_limit = self._raw_message_limit()
        return len_fn, max(500, raw_limit - len_fn(self.cfg.cursor) - 100)

    async def _start_transports(self) -> None:
        """Resolve native/draft transport; native wins (adapters declaring it can't edit).

        Sends an empty seed frame so the user sees "typing" before the first
        token; on seed failure fall back to the edit path (the gateway's
        fallback send then handles it).  Drafts and native streaming target
        the same first-frame slot.
        """
        self._use_native_streaming = self._resolve_native_streaming()
        if self._use_native_streaming:
            logger.debug(
                "Stream consumer using native-stream transport (chat=%s)",
                self.chat_id,
            )
            try:
                seed_ok = await self._send_seed_frame()
                if seed_ok:
                    self._native_stream_opened = True
            except Exception:
                logger.debug(
                    "Native streaming seed frame raised; disabling native",
                    exc_info=True,
                )
                seed_ok = False
            if not seed_ok:
                self._use_native_streaming = False

        if self._use_native_streaming:
            self._use_draft_streaming = False
            return
        self._use_draft_streaming = self._resolve_draft_streaming()
        if self._use_draft_streaming:
            type(self)._draft_id_counter += 1
            self._draft_id = type(self)._draft_id_counter
            logger.debug(
                "Stream consumer using native-draft transport (chat=%s draft_id=%s)",
                self.chat_id, self._draft_id,
            )

    def _drain_queue(self) -> "_Tick":
        """Drain everything queued so far into one tick.

        Control sentinels stop the drain (they take effect this tick);
        _FINAL_TEXT / _TOOL_PROGRESS / text deltas fold into state and keep
        draining so simultaneous items batch.
        """
        tick = _Tick()
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                return tick
            if item is _DONE:
                tick.got_done = True
                return tick
            if item is _NEW_SEGMENT:
                tick.got_segment_break = True
                return tick
            if item is _REOPEN_SEED:
                tick.got_reopen_seed = True
                return tick
            if not isinstance(item, tuple):
                self._filter_and_accumulate(item)
                continue
            try:
                handler = self._QUEUE_TUPLE_HANDLERS.get((item[0], len(item)))
            except TypeError:  # unhashable head: not one of ours
                handler = None
            if handler is None:
                self._filter_and_accumulate(item)
                continue
            if handler(self, tick, item):
                return tick

    def _on_final_text(self, tick: "_Tick", item: tuple) -> bool:
        self._adopt_final_text(item[1])
        return False

    def _on_approval_boundary(self, tick: "_Tick", item: tuple) -> bool:
        tick.approval_boundary = (item[1], item[2])
        return True

    def _on_commentary(self, tick: "_Tick", item: tuple) -> bool:
        tick.commentary_text = item[1]
        return True

    def _on_flush(self, tick: "_Tick", item: tuple) -> bool:
        # Barrier: finalize like a tool boundary, signal at the end of the tick.
        tick.got_flush = True
        tick.got_segment_break = True
        tick.flush_event = item[1]
        return True

    def _on_tool_progress(self, tick: "_Tick", item: tuple) -> bool:
        if self._use_native_streaming:
            self._tool_progress_lines.append(item[1])
            self._tool_progress_active = True
        return False  # keep draining to batch simultaneous progress lines

    def _adopt_final_text(self, final_raw: str) -> None:
        """Adopt the authoritative final (see finish()) as the finalize content.

        Only if this consumer streamed something (a no-stream turn keeps the
        gateway's normal final-send ownership).  Split delivery: wholesale
        adoption would repeat sealed heads inside the tail, but refusing
        entirely makes the gateway resend the ENTIRE body+footer — if the
        final strictly prefix-extends the ledger, append only the missing
        suffix; non-prefix rewrites keep the full-resend fallback.
        """
        streamed_something = bool(
            self._accumulated or self._message_id or self._last_sent_text
        )
        if not streamed_something:
            return
        if not self._turn_split_delivery:
            final_payload = self._clean_for_display(final_raw)
            visible = self._clean_for_display(self._accumulated)
            if final_payload and final_payload != visible:
                self._accumulated = final_raw
                self._stream_ledger = final_raw
            return
        ledger = self._stream_ledger
        if ledger and final_raw.startswith(ledger) and len(final_raw) > len(ledger):
            self._accumulated += final_raw[len(ledger):]
            self._stream_ledger = final_raw

    async def _eager_reopen_seed(self) -> None:
        """Eager re-seed after a clarify answer.

        Re-check the gate (state may have advanced between put and dequeue).
        Trade-off: WeCom's ~6-minute stream limit (errcode 846608, counted
        from the FIRST frame) now starts at the reply instant rather than the
        first post-answer delta; if it expires, send_stream_frame returns
        False and we degrade to send() — only animation is lost.
        """
        if not (
            self._use_native_streaming
            and self._awaiting_reopen_after_boundary
            and not self._native_stream_opened
        ):
            return
        try:
            seed_ok = await self._send_seed_frame()
        except Exception as e:
            logger.debug("Eager reopen seed raised, disabling native: %s", e)
            seed_ok = False
        if seed_ok:
            self._native_stream_opened = True
            self._native_last_pushed_len = 0
            self._awaiting_reopen_after_boundary = False
            self._reopen_seeded_eagerly = True
            logger.info(
                "[latency] Eager re-seed after clarify answer "
                "(typing bubble reopened immediately, turn=%s)",
                self._turn_id,
            )
        else:
            # Degrade to a single buffered send() (one bubble, not per-tick
            # fragments), like the approval path.
            self._degrade_native_to_buffered_send()

    def _should_edit(self, tick: "_Tick") -> bool:
        """Decide whether this tick flushes an edit/frame."""
        should_edit = tick.got_done or tick.got_segment_break or tick.commentary_text is not None
        if not self.cfg.buffer_only:
            if self._use_native_streaming:
                # No platform edit-rate limit: push every delta immediately.
                should_edit = should_edit or bool(self._accumulated) or self._tool_progress_active
            else:
                elapsed = time.monotonic() - self._last_edit_time
                should_edit = should_edit or (
                    (elapsed >= self._current_edit_interval and self._accumulated)
                    # buffer_threshold is a codepoint debounce heuristic,
                    # not a platform-limit check (_len_fn is for overflow).
                    or len(self._accumulated) >= self.cfg.buffer_threshold
                )
        # Defer mid-stream edits while the buffer could still resolve to a
        # silence marker ("NO"→"NO_REPLY") so it never flashes on screen;
        # got_done always resolves the buffer, so marker-like prose is never lost.
        if (
            should_edit
            and tick.is_interim
            and _is_partial_silence_marker(self._clean_for_display(self._accumulated))
        ):
            return False
        return should_edit

    async def _split_first_send(self, tick: "_Tick") -> str:
        """No message to edit yet and the buffer overflows: seal only the head chunks.

        The tail stays in _accumulated so it becomes the active preview later
        deltas edit in place.  Returns "return" (turn finished) or "continue".
        """
        chunks = self._truncate_for_stream(self._accumulated, self._safe_limit, self._len_fn)
        if len(chunks) <= 1:
            # Malformed/legacy adapter result must still be splittable.
            chunks = self._split_text_chunks(self._accumulated, self._safe_limit, self._len_fn)
        chunks_delivered = False
        reply_to = self._initial_reply_to_id
        all_heads_delivered = len(chunks) > 1
        for chunk in chunks[:-1]:
            new_id = await self._send_new_chunk(chunk, reply_to, final=tick.got_done)
            if new_id is None or new_id == reply_to:
                # Keep the full text intact for the gateway fallback.
                all_heads_delivered = False
                chunks_delivered = False
                break
            chunks_delivered = True
            reply_to = new_id

        if all_heads_delivered:
            self._accumulated = chunks[-1]
        # Heads are sealed (or a later head failed): never edit a sealed
        # message with the unsplit payload — the tail is sent fresh, or the
        # fallback path retries.
        self._message_id = None
        self._message_created_ts = None
        self._last_sent_text = ""

        if chunks_delivered:
            # Flag BEFORE the tail send: fresh-final replaces every tracked
            # preview with one message, which is only valid while the active
            # message holds the whole answer — deleting sealed heads drops
            # delivered text.
            self._turn_split_delivery = True

        self._last_edit_time = time.monotonic()
        if tick.got_done:
            tail_delivered = True
            if self._accumulated:
                tail_delivered = await self._send_or_edit(self._accumulated, finalize=True)
            # ``_already_sent`` may be True from prior progress/fallback state
            # — only heads + tail count.
            self._final_response_sent = chunks_delivered and tail_delivered
            if self._final_response_sent:
                self._final_content_delivered = True
                self._turn_split_delivery = True
                self._record_turn_final_payload(self._accumulated)
            return "return"
        if tick.got_segment_break:
            self._message_id = None
            self._fallback_final_send = False
            self._fallback_prefix = ""
            if not self._accumulated:
                return "continue"
        # Early `continue` skips the bottom-of-loop flush signal.
        if tick.got_flush:
            self._signal_flush(tick.flush_event)
        return "continue"

    async def _seal_overflow_heads(self) -> None:
        """Existing message overflowing: seal it with the head, start a new message for the rest."""
        while (
            self._len_fn(self._accumulated) > self._safe_limit
            and self._message_id is not None
            and self._edit_supported
        ):
            cp_budget = _custom_unit_to_cp(self._accumulated, self._safe_limit, self._len_fn)
            split_at = self._accumulated.rfind("\n", 0, cp_budget)
            if split_at < cp_budget // 2:
                split_at = cp_budget
            chunk = self._accumulated[:split_at]
            # finalize=True: this sealed chunk is never edited again, so it
            # needs its rich-text pass now or it renders raw.
            # is_turn_final=False: a split head is not the answer, so
            # fresh-final must not mark the turn delivered on it.
            ok = await self._send_or_edit(chunk, finalize=True, is_turn_final=False)
            if self._fallback_final_send or not ok:
                # Keep the full text intact for the fallback final send.
                break
            self._accumulated = self._accumulated[split_at:].lstrip("\n")
            self._message_id = None
            self._last_sent_text = ""
            self._turn_split_delivery = True

    async def _push_update(self, tick: "_Tick") -> None:
        """Send/edit this tick's visible text (cursor-suffixed unless finalizing)."""
        display_text = self._accumulated
        if tick.is_interim:
            if self._use_native_streaming:
                display_text = self._compose_frame_content()
                if display_text and self.cfg.cursor:
                    display_text += self.cfg.cursor
            else:
                display_text += self.cfg.cursor

        # got_done update going out as a FRESH send via the draft transport
        # (drafts have no message id) already carries finalize=True — unlike an
        # EDIT during draft streaming, which still needs the explicit finalize
        # pass for REQUIRES_EDIT_FINALIZE adapters.
        tick.draft_final_fresh_send = (
            tick.got_done and self._use_draft_streaming and self._message_id is None
        )
        # Segment break finalizes so platforms needing explicit closure
        # (DingTalk AI Cards) don't leave the previous segment stuck loading;
        # got_done has its own finalize in _finalize_turn.
        tick.update_visible = await self._send_or_edit(
            display_text,
            finalize=(tick.got_done or tick.got_segment_break),
            # A segment-break finalize closes a preamble, not the answer.
            is_turn_final=tick.got_done,
        )
        self._last_edit_time = time.monotonic()
        # Lines stay in _tool_progress_lines for the next compose; only new
        # progress should trigger another should_edit.
        if self._tool_progress_active:
            self._tool_progress_active = False

    async def _finalize_turn(self, tick: "_Tick") -> None:
        """got_done: final edit without cursor, or one continuation send if edits failed mid-stream."""
        if self._accumulated or self._message_id is not None or self._already_sent:
            await self._notify_before_finalize()
        if (
            self._awaiting_reopen_after_boundary
            and not self._native_stream_opened
            and not self._accumulated
        ):
            # Lazy reopen, no post-prompt content: nothing is open on screen,
            # so don't re-seed just to emit a lone "✅".
            logger.debug(
                "Clarify reopen boundary with no post-prompt content "
                "— skipping lone-placeholder finalize (turn=%s)",
                self._turn_id,
            )
        elif (
            self._reopen_seeded_eagerly
            and self._native_stream_opened
            and not self._accumulated
            and not tick.update_visible
        ):
            # Eager seed, no content: the typing bubble IS on screen and would
            # hang forever — close it with an empty finalize (not a lone "✅").
            # Delivery flags untouched.
            try:
                await self._send_frame("", finalize=True)
            except Exception as e:
                logger.debug("Eager-seed empty finalize failed: %s", e)
            self._close_native_state()
            self._reopen_seeded_eagerly = False
            logger.debug(
                "Eager reopen seed but no post-answer content — "
                "closed empty typing bubble (turn=%s)",
                self._turn_id,
            )
        elif self._use_native_streaming:
            # Native streams MUST close with finish=true even when empty
            # (tool-only turns) — placeholder if needed.
            if not tick.update_visible:
                await self._finalize_edit(self._accumulated or "✅", record=False)
            else:
                self._mark_final_delivered()
        elif self._accumulated:
            await self._finalize_edit_path(tick)

    async def _finalize_edit_path(self, tick: "_Tick") -> None:
        """Edit-transport finalize (the non-native got_done branches)."""
        if self._fallback_final_send:
            await self._send_fallback_final(self._accumulated)
        elif self._final_response_sent:
            # Fresh-final already delivered above; a second finalize would
            # duplicate / re-delete.
            self._final_content_delivered = True
            self._record_turn_final_payload(self._accumulated)
        elif tick.update_visible and (
            not self._adapter_requires_finalize
            or self._last_edit_overflowed
            or tick.draft_final_fresh_send
        ):
            # The update above already delivered the final.  A second finalize
            # would re-edit an already-final message (Telegram: sendRichMessage
            # followed by editMessageText falls back to the legacy formatter),
            # or overflow-split again into an adopted continuation,
            # duplicating chunks.
            self._mark_skip_redundant_finalize()
        elif self._message_id:
            # No visible update this tick, or the adapter needs explicit finalize=True.
            ok = await self._finalize_edit(self._accumulated)
            if not ok and self._fallback_final_send:
                # This edit may have exhausted flood strikes and promoted
                # fallback mode; send only the unsent tail now so the gateway
                # doesn't duplicate the visible prefix.
                await self._send_fallback_final(self._accumulated)
        elif not self._already_sent:
            # Retry after the finalize tick failed.  finalize=True keeps
            # stream-is-the-message adapters out of the draft-frame branch,
            # whose dedupe against the last UNSEALED frame would report
            # success with no transport call (silent loss).
            await self._finalize_edit(self._accumulated)

    async def _finalize_edit(self, text: str, *, record: bool = True) -> bool:
        """finalize=True send_or_edit; on success mark the turn delivered (and record the payload)."""
        self._final_response_sent = await self._send_or_edit(text, finalize=True)
        if self._final_response_sent:
            self._mark_final_delivered(record=text if record else None)
        return self._final_response_sent

    def _cumulative_transport(self) -> bool:
        """Stream-is-the-message drafts and WeCom native: one append-only stream per turn."""
        return (
            self._stream_is_message() and self._use_draft_streaming
        ) or self._use_native_streaming

    async def _deliver_commentary(self, commentary_text: str) -> None:
        """Cumulative transports post commentary as its own message and the
        stream continues — resetting _accumulated would break the append-only
        invariant / lose pre-commentary text."""
        if self._cumulative_transport():
            await self._send_commentary(commentary_text)
            self._last_edit_time = time.monotonic()
        else:
            self._reset_segment_state()
            await self._send_commentary(commentary_text)
            self._last_edit_time = time.monotonic()
            self._reset_segment_state()

    async def _end_segment(self, tick: "_Tick") -> None:
        """Tool boundary: edit-based transports reset so the next chunk is a fresh message below tool progress.

        Cumulative transports must NOT reset — clearing _accumulated makes the
        next frame a non-prefix snapshot and the connector re-appends the
        whole answer.  preserve_no_edit: "__no_edit__" (platform never
        returned a real id — Signal, github_comment webhook) must keep its
        sentinel or every tool boundary posts a new message (155 comments on
        one PR); the continuation goes out once via _send_fallback_final.  A
        real id from a flood-failed edit still resets as intended.
        """
        if self._cumulative_transport():
            return
        # If the segment-break edit didn't land (flood control / fallback
        # mode), _accumulated holds unseen pre-boundary text — flush it
        # before the reset wipes it.
        if (
            self._accumulated
            and not tick.update_visible
            and self._message_id
            and self._message_id != "__no_edit__"
        ):
            await self._flush_segment_tail_on_edit_failure()
        self._reset_segment_state(preserve_no_edit=True)

    async def _on_cancelled(self) -> None:
        """Best-effort final edit on task cancel.

        finalize=True so REQUIRES_EDIT_FINALIZE platforms apply formatting
        (the flags set here suppress the gateway's formatted re-send);
        is_turn_final=False because this handler owns the flags, not
        _try_fresh_final.  Only a successful best-effort edit confirms
        delivery — a partial send (already_sent) may be just "Let me
        search…", not the answer.
        """
        best_effort_ok = False
        if self._accumulated and self._message_id:
            with contextlib.suppress(Exception):
                best_effort_ok = bool(
                    await self._send_or_edit(
                        self._accumulated, finalize=True, is_turn_final=False,
                    )
                )
        elif self._message_id is None:
            # Draft path keeps _message_id=None, so the edit above never runs
            # for it; seal in place (else the stream stays visibly live and
            # the adapter keeps armed interception state for the next turn).
            # Sets no delivery flags.
            await self._abandon_native_stream()
        if best_effort_ok and not self._final_response_sent:
            self._mark_final_delivered(record=self._accumulated)

    def _wake_flush_waiters(self) -> None:
        """Wake still-queued _FLUSH waiters so a consumer dying mid-flush
        doesn't stall flush_pending_sync() for its full timeout."""
        try:
            while True:
                item = self._queue.get_nowait()
                if isinstance(item, tuple) and len(item) == 2 and item[0] is _FLUSH:
                    self._signal_flush(item[1])
        except queue.Empty:
            pass
        except Exception:
            pass

    # Tuple-shaped queue items keyed on (sentinel, arity); handler returns True
    # to stop draining.  Order-insensitive: each sentinel is a distinct object.
    _QUEUE_TUPLE_HANDLERS = {
        (_FINAL_TEXT, 2): _on_final_text,
        (_APPROVAL_BOUNDARY, 3): _on_approval_boundary,
        (_COMMENTARY, 2): _on_commentary,
        (_FLUSH, 2): _on_flush,
        (_TOOL_PROGRESS, 2): _on_tool_progress,
    }

    @staticmethod
    def _clean_for_display(text: str) -> str:
        """Hide MEDIA:<path> / [[audio_as_voice]] directives; media is delivered after the stream."""
        return _BasePlatformAdapter.strip_media_directives_for_display(text)

