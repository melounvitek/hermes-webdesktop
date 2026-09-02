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
from gateway.platforms.base import MEDIA_TAG_CLEANUP_RE
from gateway.config import (
    DEFAULT_STREAMING_EDIT_INTERVAL as _DEFAULT_STREAMING_EDIT_INTERVAL,
    DEFAULT_STREAMING_BUFFER_THRESHOLD as _DEFAULT_STREAMING_BUFFER_THRESHOLD,
    DEFAULT_STREAMING_CURSOR as _DEFAULT_STREAMING_CURSOR,
)
from gateway.response_filters import (
    is_intentional_silence_response as _is_intentional_silence_response,
    is_partial_silence_marker as _is_partial_silence_marker,
)
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


def escape_code_fences_for_display(text: str) -> str:
    """Replace each ``` with \\`\\`\\` so text can be wrapped in an outer ``` block.

    Reasoning content that quotes code would otherwise break the outer fence.
    """
    if not isinstance(text, str) or "```" not in text:
        return text
    return text.replace("```", "\\`\\`\\`")


def ensure_closed_code_fences(text: str) -> str:
    """Append a closing ``` and/or ` if the text has orphaned code markers.

    Output truncated mid-code-block (token limit, finish_reason="length") would
    otherwise render everything after the orphan as one code block / inline
    span.  Trade-off: a spurious close creates a brief empty span at the end,
    far less harmful than the alternative.  Odd ``` count → append a fence on
    its own line; then, with complete ```…``` regions stripped, odd ` count →
    append a backtick.
    """
    if not isinstance(text, str) or not text:
        return text

    if text.count("```") % 2 == 1:
        text = text.rstrip("\n") + "\n```"

    # Strip complete fenced regions (and any trailing unclosed ``` that leaks
    # through) so their internal backticks don't pollute the standalone count.
    import re
    without_fences = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    without_fences = re.sub(r"```[^`]*$", "", without_fences)

    if without_fences.count("`") % 2 == 1:
        text = text + "`"

    return text


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


class GatewayStreamConsumer:
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

    # Must stay in sync with cli.py _OPEN_TAGS/_CLOSE_TAGS and
    # run_agent.py _strip_think_blocks() tag variants.
    _OPEN_THINK_TAGS = (
        "<REASONING_SCRATCHPAD>", "<think>", "<reasoning>",
        "<THINKING>", "<thinking>", "<thought>",
    )
    _CLOSE_THINK_TAGS = (
        "</REASONING_SCRATCHPAD>", "</think>", "</reasoning>",
        "</THINKING>", "</thinking>", "</thought>",
    )

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
        # No-arg callback fired whenever a fresh content bubble is created
        # (first send, commentary, overflow chunk, fallback continuation); the
        # gateway uses it to open the next tool-progress bubble BELOW resumed
        # content instead of editing the old one above.  Exceptions swallowed.
        self._on_new_message = on_new_message
        # Fired once on entering the finalization path so the gateway can pause
        # typing refreshes before a slow rich-text final edit.
        self._on_before_finalize = on_before_finalize
        self._initial_reply_to_id = initial_reply_to_id

        # Per-turn id passed to adapter.send_stream_frame() so concurrent
        # consumers (/background, parallel subagents) don't interfere.
        import uuid
        self._turn_id = str(uuid.uuid4())

        self._queue: queue.Queue = queue.Queue()
        self._accumulated = ""
        # Mirror of ``_accumulated`` NOT truncated when overflow splits seal
        # head chunks; records a reconcilable turn-final payload for splits.
        self._stream_ledger = ""
        self._message_id: Optional[str] = None
        # time.monotonic() when ``_message_id`` was first assigned; fresh-final
        # logic uses it to detect long-lived previews.
        self._message_created_ts: Optional[float] = None
        # Every real preview message id put on screen this response (first send
        # + continuation messages).  Fresh-final deletes them all so a reply
        # split at the edit limit leaves no stale fragments above the final.
        self._preview_message_ids: "set[str]" = set()
        # IDs from only the active text segment: failure recovery must never
        # delete an earlier finalized preamble/commentary message.
        self._segment_preview_message_ids: "set[str]" = set()
        self._already_sent = False
        self._edit_supported = True  # Disabled when progressive edits are no longer usable
        self._last_edit_time = 0.0
        self._last_sent_text = ""   # Track last-sent text to skip redundant edits
        # True when the most recent _send_or_edit split across continuation
        # messages (the adapter adopted a new message id).
        self._last_edit_overflowed = False
        self._fallback_final_send = False
        self._fallback_prefix = ""
        # True when fallback sends only the missing tail after a partial
        # overflow delivery: the visible prefix is content, not a stale preview.
        self._fallback_preserve_partial_messages = False
        # Telegram bounds edit retries at 5s; a final-delivery fallback must not
        # hold the stream task through a longer flood cooldown.
        self._max_fallback_flood_retry_seconds = 5.0
        self._flood_strikes = 0         # Consecutive flood-control edit failures
        self._current_edit_interval = self.cfg.edit_interval  # Adaptive backoff
        self._final_response_sent = False
        # Final content reached the user even if the cosmetic final edit
        # (cursor removal) then failed.
        self._final_content_delivered = False
        # Exact cleaned payload of the turn-final delivery that set the flags
        # above.  The gateway compares it against the completed final_response
        # before trusting the flags (a successful finalize edit may carry only a
        # stale preview snapshot).  ``None`` = no record → legacy trust.
        self._delivered_final_text: Optional[str] = None
        # Answer delivered across multiple sealed messages (overflow split /
        # continuation adoption).  With a recorded payload, delivered_final_matches
        # can still reconcile; payload-less split delivery must NOT inherit
        # legacy trust (it swallowed complete replies after partial splits).
        self._turn_split_delivery = False
        # A full-final send timed out in a way that MAY have reached the
        # platform — the only payload-less case that keeps legacy trust, since
        # re-sending risks a duplicate rather than recovering a loss.
        self._delivery_ambiguous = False
        self._delivered_commentary_texts: list[str] = []
        # Finalized visible text of each segment, so has_delivered_text still
        # matches after _reset_segment_state clears _last_sent_text.
        self._delivered_segment_texts: list[str] = []
        # Only platforms needing an explicit finalize call (e.g. DingTalk AI
        # Cards) force a redundant final edit.  ``is True`` keeps MagicMock
        # adapters in tests from enabling this path.
        self._adapter_requires_finalize: bool = (
            getattr(adapter, "REQUIRES_EDIT_FINALIZE", False) is True
        )

        # Returns False after /new or /stop; run() then abandons the stream
        # instead of delivering stale deltas.
        self._run_still_current = run_still_current or (lambda: True)

        # Think-block filter state (mirrors CLI's _stream_delta tag suppression)
        self._in_think_block = False
        self._think_buffer = ""

        # Draft streaming: resolved at the start of run().  Animated frames go
        # via adapter.send_draft instead of edits; the final answer still uses
        # the normal first-send path (drafts have no message_id).
        self._use_draft_streaming = False
        self._draft_id: Optional[int] = None
        # First draft failure permanently disables drafts for this response.
        self._draft_failures = 0
        self._before_finalize_notified = False
        # Native streaming (e.g. WeCom msgtype "stream"): the ONLY delivery
        # channel for the turn — seed, cumulative updates, and finish=true all
        # go through adapter.send_stream_frame().  Resolved at the start of
        # run(); disabled on any failure so the consumer falls back to edit/send.
        self._use_native_streaming = False
        # Seed frame sent (even though it has zero visible content); fallback
        # logic uses it to decide whether the stream must be finalized first.
        self._native_stream_opened = False
        # Visible chars last pushed to the native stream; throttles frames
        # under WeCom's 30 frames/min ceiling.
        self._native_last_pushed_len = 0
        # Boundary state, set by close_for_approval_prompt(); race-free because
        # boundaries are processed serially.  ``_boundary_reopen``: keep native
        # streaming enabled so post-prompt output re-opens a fresh stream via
        # the lazy re-seed (clarify: short waits) instead of degrading to send()
        # (approval: unbounded waits, stream may go stale).
        self._boundary_placeholder = _DEFAULT_BOUNDARY_PLACEHOLDER
        self._boundary_reason = "Approval"
        self._boundary_reopen = False
        # Boundary asked to reopen but nothing has re-seeded yet; keeps got_done
        # from opening a stream just to emit a lone "✅" placeholder.
        self._awaiting_reopen_after_boundary = False
        # An EAGER re-seed (_REOPEN_SEED) already opened a fresh bubble before
        # any content, so got_done must actively finalize it when the agent
        # produces nothing — otherwise a blank typing bubble hangs forever.
        self._reopen_seeded_eagerly = False

        # Tool-progress overlay (native streaming only): lines from
        # on_tool_progress() shown in the bubble until real text arrives.
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
        self._final_response_sent = True
        self._final_content_delivered = True
        acked = self._last_sent_text or self._accumulated
        if self.cfg.cursor and acked.endswith(self.cfg.cursor):
            acked = acked[: -len(self.cfg.cursor)]
        self._record_turn_final_payload(acked)

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
        delivery_failed = False
        try:
            if self._native_stream_opened:
                finalize_text = self._accumulated or self._boundary_placeholder
                finalize_ok = False
                try:
                    result = await self.adapter.send_stream_frame(
                        finalize_text,
                        finalize=True,
                        chat_id=self.chat_id,
                        reply_to=self._initial_reply_to_id,
                        turn_id=self._turn_id,
                    )
                    finalize_ok = bool(result)
                except Exception as e:
                    logger.warning("%s boundary: finalize failed: %s", _reason, e)

                if not finalize_ok:
                    # Typing bubble may still show partial content; deliver
                    # pre-prompt text via send() so the user at least sees it.
                    logger.warning(
                        "%s boundary: finalize not confirmed, "
                        "falling back to send() for pre-prompt text (chat=%s)",
                        _reason, self.chat_id,
                    )
                    fallback_ok = False
                    try:
                        send_result = await self.adapter.send(
                            self.chat_id, finalize_text,
                        )
                        fallback_ok = getattr(send_result, "success", False)
                    except Exception as send_err:
                        logger.warning(
                            "%s boundary: fallback send also failed: %s",
                            _reason, send_err,
                        )
                    if not fallback_ok:
                        logger.error(
                            "%s boundary: both finalize and fallback send failed "
                            "(chat=%s) — pre-prompt text may not have been delivered",
                            _reason, self.chat_id,
                        )
                        delivery_failed = True
                else:
                    logger.debug(
                        "%s boundary: finalized stream (chat=%s, turn=%s)",
                        _reason, self.chat_id, self._turn_id,
                    )

            if self._boundary_reopen:
                # Clarify: keep native enabled; marking the stream closed makes
                # the next post-prompt delta re-open a fresh one via the lazy
                # re-seed in _send_or_edit.  Do NOT set buffer_only — post-prompt
                # output should stream.  The gap between this INFO and the
                # "Re-opened native stream" INFO is the typing-reappear latency.
                self._native_stream_opened = False
                self._native_last_pushed_len = 0
                self._awaiting_reopen_after_boundary = True
                self._reset_segment_state()
                logger.info(
                    "[latency] Clarify boundary finalized, awaiting first "
                    "post-answer delta to re-seed (chat=%s, turn=%s)",
                    self.chat_id, self._turn_id,
                )
            else:
                # Approval: post-approval output goes via send().  buffer_only
                # delivers it in one shot on got_done, avoiding mid-stream
                # flushes that create multiple messages on non-editable platforms.
                self._use_native_streaming = False
                self._native_stream_opened = False
                self._native_last_pushed_len = 0
                self.cfg.buffer_only = True
                self._reset_segment_state()

            boundary_ok = not delivery_failed

        except Exception as e:
            logger.warning("%s boundary processing failed: %s", _reason, e)
            boundary_ok = False
        finally:
            if boundary_future is not None:
                try:
                    if (
                        isinstance(boundary_future, (asyncio.Future, concurrent.futures.Future))
                        and not boundary_future.done()
                    ):
                        boundary_future.set_result(boundary_ok)
                except Exception:
                    pass

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

    # ── Think-block filtering ────────────────────────────────────────
    # Some models emit inline <think>...</think> blocks in content.  The agent
    # strips them from the final response, but intermediate edits go out before
    # that, so mirror the CLI's _stream_delta state machine here.

    def _filter_and_accumulate(self, text: str) -> None:
        """Append a delta to the buffer, discarding think blocks.

        Partial tags at buffer boundaries are held in ``_think_buffer`` until
        enough characters arrive to decide.
        """
        buf = self._think_buffer + text
        self._think_buffer = ""

        while buf:
            # Case-insensitive: models emit <Think>, <THINKING>, …
            lower_buf = buf.lower()
            if self._in_think_block:
                best_idx = -1
                best_len = 0
                for tag in self._CLOSE_THINK_TAGS:
                    idx = lower_buf.find(tag.lower())
                    if idx != -1 and (best_idx == -1 or idx < best_idx):
                        best_idx = idx
                        best_len = len(tag)

                if best_len:
                    self._in_think_block = False
                    buf = buf[best_idx + best_len:]
                else:
                    # Hold a tail that could be a partial close tag; discard the rest.
                    max_tag = max(len(t) for t in self._CLOSE_THINK_TAGS)
                    self._think_buffer = buf[-max_tag:] if len(buf) > max_tag else buf
                    return
            else:
                # Earliest opening tag at a block boundary (start of text, or
                # newline + optional whitespace) — prose that merely *mentions*
                # a tag must not trigger.
                best_idx = -1
                best_len = 0
                for tag in self._OPEN_THINK_TAGS:
                    tag_lower = tag.lower()
                    search_start = 0
                    while True:
                        idx = lower_buf.find(tag_lower, search_start)
                        if idx == -1:
                            break
                        # Block-boundary check (mirrors cli.py logic)
                        if idx == 0:
                            is_boundary = (
                                not self._accumulated
                                or self._accumulated.endswith("\n")
                            )
                        else:
                            preceding = buf[:idx]
                            last_nl = preceding.rfind("\n")
                            if last_nl == -1:
                                is_boundary = (
                                    (not self._accumulated
                                     or self._accumulated.endswith("\n"))
                                    and preceding.strip() == ""
                                )
                            else:
                                is_boundary = preceding[last_nl + 1:].strip() == ""

                        if is_boundary and (best_idx == -1 or idx < best_idx):
                            best_idx = idx
                            best_len = len(tag)
                            break  # first boundary hit for this tag is enough
                        search_start = idx + 1

                if best_len:
                    self._append_accumulated(buf[:best_idx])
                    self._in_think_block = True
                    buf = buf[best_idx + best_len:]
                else:
                    # Hold back a partial open tag at the tail.
                    held_back = 0
                    for tag in self._OPEN_THINK_TAGS:
                        tag_lower = tag.lower()
                        for i in range(1, len(tag)):
                            if lower_buf.endswith(tag_lower[:i]) and i > held_back:
                                held_back = i
                    if held_back:
                        self._append_accumulated(buf[:-held_back])
                        self._think_buffer = buf[-held_back:]
                    else:
                        # An orphan </think> (thinking-mode toggle dropped the
                        # open, or incomplete upstream stripping) is noise.
                        self._append_accumulated(self._strip_orphan_close_tags(buf))
                    return

    @classmethod
    def _strip_orphan_close_tags(cls, text: str) -> str:
        """Remove close tags (plus trailing whitespace) that have no matching open.

        Mirrors ``agent/think_scrubber.py::StreamingThinkScrubber`` so the
        progressive display matches the post-stream scrubber.
        """
        if "</" not in text:
            return text
        text_lower = text.lower()
        out: list[str] = []
        i = 0
        while i < len(text):
            matched = False
            if text_lower[i:i + 2] == "</":
                for tag in cls._CLOSE_THINK_TAGS:
                    tag_lower = tag.lower()
                    tag_len = len(tag_lower)
                    if text_lower[i:i + tag_len] == tag_lower:
                        j = i + tag_len
                        while j < len(text) and text[j] in " \t\n\r":
                            j += 1
                        i = j
                        matched = True
                        break
            if not matched:
                out.append(text[i])
                i += 1
        return "".join(out)

    def _flush_think_buffer(self) -> None:
        """On stream end, flush text held back waiting for a possible open tag."""
        if self._think_buffer and not self._in_think_block:
            self._append_accumulated(self._strip_orphan_close_tags(self._think_buffer))
            self._think_buffer = ""

    async def run(self) -> None:
        """Async task that drains the queue and edits the platform message."""
        # Length function and limit resolve PER-CHAT (a relay adapter fronting
        # N platforms has different caps per chat) using the platform's unit
        # (e.g. utf16 for Telegram).  isinstance gate: MagicMock auto-attributes
        # aren't callables, so test doubles fall back to len.
        _len_fn: "Callable[[str], int]" = (
            self.adapter.message_len_fn_for_chat(self.chat_id)
            if isinstance(self.adapter, _BasePlatformAdapter)
            else len
        )
        _raw_limit = self._raw_message_limit()
        _safe_limit = max(500, _raw_limit - _len_fn(self.cfg.cursor) - 100)

        # Native streaming wins over draft: adapters that declare it (WeCom)
        # cannot edit at all.  Send an empty seed frame now so the user sees
        # "typing" before the first token; on seed failure fall back to the
        # edit path (which the gateway's fallback send then handles).
        self._use_native_streaming = self._resolve_native_streaming()
        if self._use_native_streaming:
            logger.debug(
                "Stream consumer using native-stream transport (chat=%s)",
                self.chat_id,
            )
            try:
                seed_ok = await self.adapter.send_stream_frame(
                    "",
                    chat_id=self.chat_id,
                    reply_to=self._initial_reply_to_id,
                    turn_id=self._turn_id,
                )
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

        # Drafts and native streaming target the same first-frame slot.
        if self._use_native_streaming:
            self._use_draft_streaming = False
        else:
            self._use_draft_streaming = self._resolve_draft_streaming()
            if self._use_draft_streaming:
                type(self)._draft_id_counter += 1
                self._draft_id = type(self)._draft_id_counter
                logger.debug(
                    "Stream consumer using native-draft transport (chat=%s draft_id=%s)",
                    self.chat_id, self._draft_id,
                )

        try:
            while True:
                # Session reset (/new, /stop): abandon rather than deliver stale deltas.
                if not self._run_still_current():
                    await self._abandon_native_stream()
                    return

                got_done = False
                got_segment_break = False
                got_flush = False
                flush_event = None
                got_approval_boundary = False
                got_reopen_seed = False
                approval_boundary_future = None
                approval_boundary_cancelled = None
                commentary_text = None
                while True:
                    try:
                        item = self._queue.get_nowait()
                        if item is _DONE:
                            got_done = True
                            break
                        if item is _NEW_SEGMENT:
                            got_segment_break = True
                            break
                        if isinstance(item, tuple) and len(item) == 2 and item[0] is _FINAL_TEXT:
                            # Adopt the authoritative final (see finish()) as the
                            # finalize content — only if this consumer streamed
                            # something (a no-stream turn keeps the gateway's
                            # normal final-send ownership).
                            _streamed_something = bool(
                                self._accumulated
                                or self._message_id
                                or self._last_sent_text
                            )
                            if _streamed_something and not self._turn_split_delivery:
                                _final_payload = self._clean_for_display(item[1])
                                _visible = self._clean_for_display(self._accumulated)
                                if _final_payload and _final_payload != _visible:
                                    self._accumulated = item[1]
                                    self._stream_ledger = item[1]
                            elif _streamed_something and self._turn_split_delivery:
                                # Split delivery: wholesale adoption would repeat
                                # sealed heads inside the tail, but refusing
                                # entirely makes the gateway resend the ENTIRE
                                # body+footer.  If the final strictly
                                # prefix-extends the ledger, append only the
                                # missing suffix; non-prefix rewrites keep the
                                # full-resend fallback.
                                _final_raw = item[1]
                                _ledger = self._stream_ledger
                                if (
                                    _ledger
                                    and _final_raw.startswith(_ledger)
                                    and len(_final_raw) > len(_ledger)
                                ):
                                    _suffix = _final_raw[len(_ledger):]
                                    self._accumulated += _suffix
                                    self._stream_ledger = _final_raw
                            continue
                        if item is _REOPEN_SEED:
                            got_reopen_seed = True
                            break
                        if isinstance(item, tuple) and len(item) == 3 and item[0] is _APPROVAL_BOUNDARY:
                            got_approval_boundary = True
                            approval_boundary_future = item[1]
                            approval_boundary_cancelled = item[2]
                            break
                        if isinstance(item, tuple) and len(item) == 2 and item[0] is _COMMENTARY:
                            commentary_text = item[1]
                            break
                        if isinstance(item, tuple) and len(item) == 2 and item[0] is _FLUSH:
                            # Barrier: finalize like a tool boundary, signal below.
                            got_flush = True
                            got_segment_break = True
                            flush_event = item[1]
                            break
                        if isinstance(item, tuple) and len(item) == 2 and item[0] is _TOOL_PROGRESS:
                            if self._use_native_streaming:
                                self._tool_progress_lines.append(item[1])
                                self._tool_progress_active = True
                            continue  # continue draining to batch simultaneous progress lines
                        self._filter_and_accumulate(item)
                    except queue.Empty:
                        break

                # Boundary produces its own finalize and resets state, so it
                # must run before got_done/segment_break processing.
                if got_approval_boundary:
                    await self._handle_approval_boundary(
                        approval_boundary_future, approval_boundary_cancelled
                    )
                    continue

                # Eager re-seed after a clarify answer.  Re-check the gate
                # (state may have advanced between put and dequeue).  Trade-off:
                # WeCom's ~6-minute stream limit (errcode 846608, counted from
                # the FIRST frame) now starts at the reply instant rather than
                # the first post-answer delta; if it expires, send_stream_frame
                # returns False and we degrade to send() — only animation is lost.
                if got_reopen_seed:
                    if (
                        self._use_native_streaming
                        and self._awaiting_reopen_after_boundary
                        and not self._native_stream_opened
                    ):
                        try:
                            seed_ok = await self.adapter.send_stream_frame(
                                "",
                                chat_id=self.chat_id,
                                reply_to=self._initial_reply_to_id,
                                turn_id=self._turn_id,
                            )
                        except Exception as e:
                            logger.debug(
                                "Eager reopen seed raised, disabling native: %s", e,
                            )
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
                            # Degrade to a single buffered send() (one bubble,
                            # not per-tick fragments), like the approval path.
                            self._use_native_streaming = False
                            self._native_stream_opened = False
                            self._native_last_pushed_len = 0
                            self.cfg.buffer_only = True
                    continue

                if got_done:
                    self._flush_think_buffer()

                    # A bare intentional-silence marker (NO_REPLY / [SILENT]):
                    # the gateway's whole-response filter runs too late for a
                    # streamed preview, so retract it here instead of finalizing.
                    if _is_intentional_silence_response(
                        self._clean_for_display(self._accumulated)
                    ):
                        await self._suppress_silence_marker()
                        return

                # Decide whether to flush an edit
                now = time.monotonic()
                elapsed = now - self._last_edit_time
                should_edit = (
                    got_done
                    or got_segment_break
                    or commentary_text is not None
                )
                if not self.cfg.buffer_only:
                    if self._use_native_streaming:
                        # No platform edit-rate limit: push every delta immediately.
                        should_edit = should_edit or bool(self._accumulated) or self._tool_progress_active
                    else:
                        should_edit = should_edit or (
                            (elapsed >= self._current_edit_interval
                                and self._accumulated)
                            # buffer_threshold is a codepoint debounce heuristic,
                            # not a platform-limit check (_len_fn is for overflow).
                            or len(self._accumulated) >= self.cfg.buffer_threshold
                        )

                current_update_visible = False
                # got_done update went out as a FRESH send via the draft
                # transport (drafts have no message id) already carrying
                # finalize=True — unlike an EDIT during draft streaming, which
                # still needs the explicit finalize pass for
                # REQUIRES_EDIT_FINALIZE adapters.
                draft_final_fresh_send = False
                # Defer mid-stream edits while the buffer could still resolve
                # to a silence marker ("NO"→"NO_REPLY") so it never flashes on
                # screen; got_done always resolves the buffer, so marker-like
                # prose is never lost.
                if (
                    should_edit
                    and not got_done
                    and not got_segment_break
                    and commentary_text is None
                    and _is_partial_silence_marker(
                        self._clean_for_display(self._accumulated)
                    )
                ):
                    should_edit = False
                if should_edit and (self._accumulated or (self._use_native_streaming and self._tool_progress_active)):
                    # Overflow split.  Native streaming bypasses this: the
                    # adapter truncates against the stream protocol's own limit.
                    if (
                        not self._use_native_streaming
                        and _len_fn(self._accumulated) > _safe_limit
                        and self._message_id is None
                    ):
                        # No message to edit yet: seal only the overflowing head
                        # chunks, keep the tail in _accumulated so it becomes the
                        # active preview that later deltas edit in place.
                        chunks = self._truncate_for_stream(
                            self._accumulated, _safe_limit, _len_fn,
                        )
                        if len(chunks) <= 1:
                            # Malformed/legacy adapter result must still be splittable.
                            chunks = self._split_text_chunks(
                                self._accumulated, _safe_limit, _len_fn,
                            )
                        chunks_delivered = False
                        reply_to = self._initial_reply_to_id
                        all_heads_delivered = len(chunks) > 1
                        for chunk in chunks[:-1]:
                            new_id = await self._send_new_chunk(
                                chunk,
                                reply_to,
                                final=got_done,
                            )
                            if new_id is None or new_id == reply_to:
                                # Keep the full text intact for the gateway fallback.
                                all_heads_delivered = False
                                chunks_delivered = False
                                break
                            chunks_delivered = True
                            reply_to = new_id

                        if all_heads_delivered:
                            self._accumulated = chunks[-1]
                        # Heads are sealed (or a later head failed): never edit a
                        # sealed message with the unsplit payload — the tail is
                        # sent fresh, or the fallback path retries.
                        self._message_id = None
                        self._message_created_ts = None
                        self._last_sent_text = ""

                        if chunks_delivered:
                            # Flag BEFORE the tail send: fresh-final replaces
                            # every tracked preview with one message, which is
                            # only valid while the active message holds the whole
                            # answer — deleting sealed heads drops delivered text.
                            self._turn_split_delivery = True

                        self._last_edit_time = time.monotonic()
                        if got_done:
                            tail_delivered = True
                            if self._accumulated:
                                tail_delivered = await self._send_or_edit(
                                    self._accumulated, finalize=True,
                                )
                            # ``_already_sent`` may be True from prior
                            # progress/fallback state — only heads + tail count.
                            self._final_response_sent = chunks_delivered and tail_delivered
                            if self._final_response_sent:
                                self._final_content_delivered = True
                                self._turn_split_delivery = True
                                self._record_turn_final_payload(self._accumulated)
                            return
                        if got_segment_break:
                            self._message_id = None
                            self._fallback_final_send = False
                            self._fallback_prefix = ""
                            if not self._accumulated:
                                continue

                        # Early `continue` skips the bottom-of-loop flush signal.
                        if got_flush:
                            self._signal_flush(flush_event)
                        continue
                    # Existing message: seal it with the head, start a new
                    # message for the remainder.
                    while (
                        _len_fn(self._accumulated) > _safe_limit
                        and self._message_id is not None
                        and self._edit_supported
                    ):
                        _cp_budget = _custom_unit_to_cp(
                            self._accumulated, _safe_limit, _len_fn,
                        )
                        split_at = self._accumulated.rfind("\n", 0, _cp_budget)
                        if split_at < _cp_budget // 2:
                            split_at = _cp_budget
                        chunk = self._accumulated[:split_at]
                        # finalize=True: this sealed chunk is never edited again,
                        # so it needs its rich-text pass now or it renders raw.
                        # is_turn_final=False: a split head is not the answer, so
                        # fresh-final must not mark the turn delivered on it.
                        ok = await self._send_or_edit(
                            chunk, finalize=True, is_turn_final=False,
                        )
                        if self._fallback_final_send or not ok:
                            # Keep the full text intact for the fallback final send.
                            break
                        self._accumulated = self._accumulated[split_at:].lstrip("\n")
                        self._message_id = None
                        self._last_sent_text = ""
                        self._turn_split_delivery = True

                    display_text = self._accumulated
                    if not got_done and not got_segment_break and commentary_text is None:
                        if self._use_native_streaming:
                            display_text = self._compose_frame_content()
                            if display_text and self.cfg.cursor:
                                display_text += self.cfg.cursor
                        else:
                            display_text += self.cfg.cursor

                    # Segment break finalizes so platforms needing explicit
                    # closure (DingTalk AI Cards) don't leave the previous
                    # segment stuck loading; got_done has its own finalize below.
                    draft_final_fresh_send = (
                        got_done
                        and self._use_draft_streaming
                        and self._message_id is None
                    )
                    current_update_visible = await self._send_or_edit(
                        display_text,
                        finalize=(got_done or got_segment_break),
                        # A segment-break finalize closes a preamble, not the answer.
                        is_turn_final=got_done,
                    )
                    self._last_edit_time = time.monotonic()
                    # Lines stay in _tool_progress_lines for the next compose;
                    # only new progress should trigger another should_edit.
                    if self._tool_progress_active:
                        self._tool_progress_active = False

                if got_done:
                    if self._accumulated or self._message_id is not None or self._already_sent:
                        await self._notify_before_finalize()
                    # Final edit without cursor; if progressive editing failed
                    # mid-stream, send one continuation here rather than letting
                    # the gateway resend the full response.
                    if (
                        self._awaiting_reopen_after_boundary
                        and not self._native_stream_opened
                        and not self._accumulated
                    ):
                        # Lazy reopen, no post-prompt content: nothing is open
                        # on screen, so don't re-seed just to emit a lone "✅".
                        logger.debug(
                            "Clarify reopen boundary with no post-prompt content "
                            "— skipping lone-placeholder finalize (turn=%s)",
                            self._turn_id,
                        )
                    elif (
                        self._reopen_seeded_eagerly
                        and self._native_stream_opened
                        and not self._accumulated
                        and not current_update_visible
                    ):
                        # Eager seed, no content: the typing bubble IS on screen
                        # and would hang forever — close it with an empty
                        # finalize (not a lone "✅").  Delivery flags untouched.
                        try:
                            await self.adapter.send_stream_frame(
                                "",
                                finalize=True,
                                chat_id=self.chat_id,
                                reply_to=self._initial_reply_to_id,
                                turn_id=self._turn_id,
                            )
                        except Exception as e:
                            logger.debug(
                                "Eager-seed empty finalize failed: %s", e,
                            )
                        self._native_stream_opened = False
                        self._native_last_pushed_len = 0
                        self._reopen_seeded_eagerly = False
                        logger.debug(
                            "Eager reopen seed but no post-answer content — "
                            "closed empty typing bubble (turn=%s)",
                            self._turn_id,
                        )
                    elif self._use_native_streaming:
                        # Native streams MUST close with finish=true even when
                        # empty (tool-only turns) — placeholder if needed.
                        if not current_update_visible:
                            close_text = self._accumulated or "✅"
                            self._final_response_sent = await self._send_or_edit(
                                close_text, finalize=True,
                            )
                            if self._final_response_sent:
                                self._final_content_delivered = True
                        else:
                            self._final_response_sent = True
                            self._final_content_delivered = True
                    elif self._accumulated:
                        if self._fallback_final_send:
                            await self._send_fallback_final(self._accumulated)
                        elif self._final_response_sent:
                            # Fresh-final already delivered above; a second
                            # finalize would duplicate / re-delete.
                            self._final_content_delivered = True
                            self._record_turn_final_payload(self._accumulated)
                        elif (
                            current_update_visible
                            and (
                                not self._adapter_requires_finalize
                                or self._last_edit_overflowed
                                or draft_final_fresh_send
                            )
                        ):
                            # The update above already delivered the final.  A
                            # second finalize would re-edit an already-final
                            # message (Telegram: sendRichMessage followed by
                            # editMessageText falls back to the legacy
                            # formatter), or overflow-split again into an
                            # adopted continuation, duplicating chunks.
                            self._mark_skip_redundant_finalize()
                        elif self._message_id:
                            # No visible update this tick, or the adapter needs
                            # explicit finalize=True.
                            self._final_response_sent = await self._send_or_edit(
                                self._accumulated, finalize=True,
                            )
                            if self._final_response_sent:
                                self._final_content_delivered = True
                                self._record_turn_final_payload(self._accumulated)
                            elif self._fallback_final_send:
                                # This edit may have exhausted flood strikes and
                                # promoted fallback mode; send only the unsent
                                # tail now so the gateway doesn't duplicate the
                                # visible prefix.
                                await self._send_fallback_final(self._accumulated)
                        elif not self._already_sent:
                            # Retry after the finalize tick failed.  finalize=True
                            # keeps stream-is-the-message adapters out of the
                            # draft-frame branch, whose dedupe against the last
                            # UNSEALED frame would report success with no
                            # transport call (silent loss).
                            self._final_response_sent = await self._send_or_edit(
                                self._accumulated, finalize=True,
                            )
                            if self._final_response_sent:
                                self._final_content_delivered = True
                                self._record_turn_final_payload(self._accumulated)
                    return

                if commentary_text is not None:
                    # Cumulative transports (stream-is-the-message drafts, WeCom
                    # native): commentary posts as its own message and the
                    # stream continues — resetting _accumulated would break the
                    # append-only invariant / lose pre-commentary text.
                    if (
                        self._stream_is_message() and self._use_draft_streaming
                    ) or self._use_native_streaming:
                        await self._send_commentary(commentary_text)
                        self._last_edit_time = time.monotonic()
                    else:
                        self._reset_segment_state()
                        await self._send_commentary(commentary_text)
                        self._last_edit_time = time.monotonic()
                        self._reset_segment_state()

                # Tool boundary: edit-based transports reset so the next chunk
                # is a fresh message below tool progress.  Cumulative transports
                # (stream-is-the-message drafts, WeCom native) must NOT reset —
                # clearing _accumulated makes the next frame a non-prefix
                # snapshot and the connector re-appends the whole answer.
                # preserve_no_edit: "__no_edit__" (platform never returned a
                # real id — Signal, github_comment webhook) must keep its
                # sentinel or every tool boundary posts a new message (155
                # comments on one PR); the continuation goes out once via
                # _send_fallback_final.  A real id from a flood-failed edit
                # still resets as intended.
                if got_segment_break:
                    if (
                        self._stream_is_message()
                        and self._use_draft_streaming
                    ) or self._use_native_streaming:
                        pass
                    else:
                        # If the segment-break edit didn't land (flood control /
                        # fallback mode), _accumulated holds unseen pre-boundary
                        # text — flush it before the reset wipes it.
                        if (
                            self._accumulated
                            and not current_update_visible
                            and self._message_id
                            and self._message_id != "__no_edit__"
                        ):
                            await self._flush_segment_tail_on_edit_failure()
                        self._reset_segment_state(preserve_no_edit=True)

                # Done last so the waiter unblocks only once everything queued
                # before the barrier is on screen.
                if got_flush:
                    self._signal_flush(flush_event)

                await asyncio.sleep(0.05)  # Small yield to not busy-loop

        except asyncio.CancelledError:
            # Best-effort final edit.  finalize=True so REQUIRES_EDIT_FINALIZE
            # platforms apply formatting (the flags below suppress the gateway's
            # formatted re-send); is_turn_final=False because this handler owns
            # the flags, not _try_fresh_final.
            _best_effort_ok = False
            if self._accumulated and self._message_id:
                with contextlib.suppress(Exception):
                    _best_effort_ok = bool(
                        await self._send_or_edit(
                            self._accumulated, finalize=True, is_turn_final=False,
                        )
                    )
            elif self._message_id is None:
                # Draft path keeps _message_id=None, so the edit above never
                # runs for it; seal in place (else the stream stays visibly
                # live and the adapter keeps armed interception state for the
                # next turn).  Sets no delivery flags.
                await self._abandon_native_stream()
            # Only a successful best-effort edit confirms delivery — a partial
            # send (already_sent) may be just "Let me search…", not the answer.
            if _best_effort_ok and not self._final_response_sent:
                self._final_response_sent = True
                self._final_content_delivered = True
                self._record_turn_final_payload(self._accumulated)
        except Exception as e:
            logger.error("Stream consumer error: %s", e)
        finally:
            # Wake any still-queued _FLUSH waiters so a consumer dying mid-flush
            # doesn't stall flush_pending_sync() for its full timeout.
            try:
                while True:
                    item = self._queue.get_nowait()
                    if (
                        isinstance(item, tuple)
                        and len(item) == 2
                        and item[0] is _FLUSH
                    ):
                        self._signal_flush(item[1])
            except queue.Empty:
                pass
            except Exception:
                pass

    # Shared with the non-streaming path so a MEDIA tag is treated identically
    # either way (only deliverable-extension paths are stripped).
    _MEDIA_RE = MEDIA_TAG_CLEANUP_RE

    @staticmethod
    def _clean_for_display(text: str) -> str:
        """Hide MEDIA:<path> / [[audio_as_voice]] directives; media is delivered after the stream."""
        return _BasePlatformAdapter.strip_media_directives_for_display(text)

    async def _send_new_chunk(
        self,
        text: str,
        reply_to_id: Optional[str],
        *,
        final: bool = False,
    ) -> Optional[str]:
        """Send a new chunk threaded to ``reply_to_id``; returns the new message_id."""
        text = self._clean_for_display(text)
        if not text.strip():
            return reply_to_id
        try:
            result = await self.adapter.send(
                chat_id=self.chat_id,
                content=text,
                reply_to=reply_to_id,
                metadata=self._metadata_for_send(
                    final=final,
                    expect_edits=not final,
                ),
            )
            if result.success and result.message_id:
                self._message_id = str(result.message_id)
                self._track_preview_ids_from_result(result)
                self._already_sent = True
                self._last_sent_text = text
                self._notify_new_message()
                return str(result.message_id)
            else:
                self._edit_supported = False
                return reply_to_id
        except Exception as e:
            logger.error("Stream send chunk error: %s", e)
            return reply_to_id

    def _visible_prefix(self) -> str:
        """Return the visible text already shown in the streamed message."""
        prefix = self._last_sent_text or ""
        if self.cfg.cursor and prefix.endswith(self.cfg.cursor):
            prefix = prefix[:-len(self.cfg.cursor)]
        return self._clean_for_display(prefix)

    def _continuation_text(self, final_text: str) -> str:
        """Return only the part of final_text the user has not already seen."""
        prefix = self._fallback_prefix or self._visible_prefix()
        if prefix and final_text.startswith(prefix):
            return final_text[len(prefix):].lstrip()
        return final_text

    @staticmethod
    def _split_text_chunks(
        text: str,
        limit: int,
        len_fn: "Callable[[str], int]" = len,
    ) -> list[str]:
        """Split text for fallback sends: newline-preferred, fence-balanced across chunks."""
        from gateway.platforms.helpers import split_text_fence_aware

        return split_text_fence_aware(
            text,
            limit,
            len_fn,
            prefer_paragraphs=False,
            balance_fences=True,
        )

    def _truncate_for_stream(
        self,
        text: str,
        limit: int,
        len_fn: "Callable[[str], int]",
    ) -> list[str]:
        """Split via the adapter's canonical truncate_message (platform-specific rules).

        Non-base test doubles / legacy adapters keep the two-argument call shape.
        """
        truncate = getattr(self.adapter, "truncate_message", None)
        if not callable(truncate):
            return self._split_text_chunks(text, limit, len_fn)

        if isinstance(self.adapter, _BasePlatformAdapter):
            chunks = truncate(text, limit, len_fn=len_fn)
        else:
            chunks = truncate(text, limit)
        if not isinstance(chunks, (list, tuple)) or not all(
            isinstance(chunk, str) for chunk in chunks
        ):
            return self._split_text_chunks(text, limit, len_fn)
        return list(chunks)

    async def _send_fallback_final(self, text: str) -> None:
        """Send the final continuation after streaming edits stop working.

        Retries each chunk once on flood-control failures with a short delay.
        """
        final_text = self._clean_for_display(text)
        # Balance fences BEFORE computing the continuation so the closing
        # fence reaches the user even when only the tail is delivered.
        final_text = ensure_closed_code_fences(final_text)
        continuation = self._continuation_text(final_text)
        self._fallback_final_send = False
        if not continuation.strip():
            # Telegram clients can lose (part of) a streamed preview after a
            # failed final edit, so opt-in adapters commit a fresh final send.
            if (
                final_text.strip()
                and final_text == self._visible_prefix()
                and getattr(
                    self.adapter,
                    "RESEND_FINAL_ON_EMPTY_STREAM_FALLBACK",
                    False,
                ) is True
            ):
                delivery = await self._send_empty_fallback_final(final_text)
                if delivery == "delivered":
                    return
                self._already_sent = True
                self._fallback_prefix = ""
                self._fallback_preserve_partial_messages = False
                if delivery in {"ambiguous", "preview"}:
                    # Timeout: Telegram may have accepted the send.  Flood
                    # rejection: the complete ACKed preview is authoritative.
                    # Keep duplicate suppression in both cases.
                    self._final_content_delivered = True
                    if delivery == "preview":
                        # Preview already shows the full final (checked above);
                        # record it so the gateway doesn't re-send next to it.
                        self._record_turn_final_payload(final_text)
                    else:
                        self._delivery_ambiguous = True
                else:
                    # Confirmed failure: gateway performs its normal final send.
                    self._final_response_sent = False
                    self._final_content_delivered = False
                return
            # The prefix may be from a *previous* segment (before a tool
            # boundary), wrongly reading as "already shown" — send final_text as-is.
            if final_text.strip() and final_text != self._visible_prefix():
                continuation = final_text
            else:
                # Best-effort strip of a cursor left stuck by the edit failure
                # that entered fallback mode.
                if (
                    self._message_id
                    and self._last_sent_text
                    and self.cfg.cursor
                    and self._last_sent_text.endswith(self.cfg.cursor)
                ):
                    clean_text = self._last_sent_text[:-len(self.cfg.cursor)]
                    try:
                        result = await self._edit_message(
                            message_id=self._message_id,
                            content=clean_text,
                        )
                        if result.success:
                            self._last_sent_text = clean_text
                    except Exception:
                        pass
                self._already_sent = True
                self._final_response_sent = True
                self._final_content_delivered = True
                # Recorder substitutes the full ledger on a split turn.
                self._record_turn_final_payload(final_text)
                return

        raw_limit = getattr(self.adapter, "MAX_MESSAGE_LENGTH", 4096)
        _len_fn: "Callable[[str], int]" = (
            self.adapter.message_len_fn
            if isinstance(self.adapter, _BasePlatformAdapter)
            else len
        )
        # Per-chat cap/unit (relay adapter fronting N platforms).
        if isinstance(self.adapter, _BasePlatformAdapter):
            try:
                raw_limit = self.adapter.max_message_length_for_chat(self.chat_id)
                _len_fn = self.adapter.message_len_fn_for_chat(self.chat_id)
            except Exception as e:
                logger.debug("per-chat limit resolution failed: %s", e)
        safe_limit = max(500, raw_limit - 100)
        chunks = self._split_text_chunks(continuation, safe_limit, len_fn=_len_fn)

        stale_message_id = self._message_id  # partial message to clean up
        last_message_id: Optional[str] = None
        last_successful_chunk = ""
        sent_any_chunk = False
        for chunk in chunks:
            result = None
            for attempt in range(2):
                result = await self.adapter.send(
                    chat_id=self.chat_id,
                    content=chunk,
                    metadata=self._metadata_for_send(final=True),
                )
                if result.success:
                    break
                retry_delay = self._fallback_flood_retry_delay(result)
                if attempt == 0 and retry_delay is not None:
                    logger.debug(
                        "Flood control on fallback send, retrying in %.1fs",
                        retry_delay,
                    )
                    await asyncio.sleep(retry_delay)
                else:
                    break  # non-flood error, long flood wait, or second failure

            if not result or not result.success:
                if sent_any_chunk:
                    # Partial continuation landed: do NOT set _final_response_sent
                    # (gateway must still deliver the full answer); _already_sent
                    # only prevents a duplicate of the partial.
                    self._already_sent = True
                    self._message_id = last_message_id
                    self._last_sent_text = last_successful_chunk
                    self._fallback_prefix = ""
                    return
                # Nothing landed — let the gateway final send try once more.
                self._already_sent = False
                self._message_id = None
                self._last_sent_text = ""
                self._fallback_prefix = ""
                return
            sent_any_chunk = True
            last_successful_chunk = chunk
            last_message_id = result.message_id or last_message_id
            self._notify_new_message()

        # Best-effort delete of the frozen partial — ONLY when the FULL final
        # was re-sent.  If only the missing tail went out, the partial IS the
        # head of the answer ("sent only the second half" symptom).
        if (
            stale_message_id
            and stale_message_id != last_message_id
            and not self._fallback_preserve_partial_messages
            and continuation == final_text
        ):
            delete_fn = getattr(self.adapter, "delete_message", None)
            if delete_fn is not None:
                try:
                    await delete_fn(self.chat_id, stale_message_id)
                except Exception as e:
                    logger.debug(
                        "Fallback partial cleanup failed (%s): %s",
                        stale_message_id, e,
                    )

        self._message_id = last_message_id
        self._already_sent = True
        self._final_response_sent = True
        self._final_content_delivered = True
        # Recorder substitutes the unsplit ledger on a split turn.
        self._record_turn_final_payload(final_text)
        self._last_sent_text = chunks[-1]
        self._fallback_prefix = ""
        self._fallback_preserve_partial_messages = False

    async def _send_empty_fallback_final(self, final_text: str) -> str:
        """Commit a completed answer after Telegram finalization fails.

        Returns "delivered", "failed" (gateway may retry), "ambiguous" (a
        timeout may have reached the platform), or "preview" (flood control
        leaves the complete streamed preview authoritative).
        """
        # Segment-scoped only: never delete an earlier finalized preamble.
        stale_ids = set(self._segment_preview_message_ids)
        if self._message_id and self._message_id != "__no_edit__":
            stale_ids.add(str(self._message_id))

        result = None
        for attempt in range(2):
            try:
                result = await self.adapter.send(
                    chat_id=self.chat_id,
                    content=final_text,
                    reply_to=self._initial_reply_to_id,
                    metadata=self._metadata_for_send(final=True),
                )
            except Exception as exc:
                logger.debug("Empty fallback final send failed: %s", exc)
                return (
                    "ambiguous"
                    if self._send_failure_may_have_delivered(exc)
                    else "failed"
                )

            if getattr(result, "success", False):
                break
            retry_delay = self._fallback_flood_retry_delay(result)
            if attempt == 0 and retry_delay is not None:
                logger.debug(
                    "Flood control on empty fallback final send; retrying in %.1fs",
                    retry_delay,
                )
                await asyncio.sleep(retry_delay)
                continue
            if self._is_flood_error(result):
                return "preview"
            return (
                "ambiguous"
                if self._send_failure_may_have_delivered(result)
                else "failed"
            )

        new_message_id = getattr(result, "message_id", None)
        delete_fn = getattr(self.adapter, "delete_message", None)
        if delete_fn is not None:
            for stale_id in stale_ids:
                if not stale_id or stale_id == new_message_id:
                    continue
                try:
                    deleted = await delete_fn(self.chat_id, stale_id)
                    if deleted is False:
                        # Telegram reports delete failure by returning False;
                        # the flood window that broke the finalize can reject
                        # this too.  One bounded retry, then best-effort.
                        await asyncio.sleep(1.0)
                        await delete_fn(self.chat_id, stale_id)
                except Exception as exc:
                    logger.debug(
                        "Empty fallback preview cleanup failed (%s): %s",
                        stale_id,
                        exc,
                    )

        self._segment_preview_message_ids = set()
        self._message_id = new_message_id or "__no_edit__"
        self._already_sent = True
        self._final_response_sent = True
        self._final_content_delivered = True
        # Record VERBATIM, not via _record_turn_final_payload: the sealed
        # previews were just deleted, so the ledger (which still holds sealed
        # heads) would claim delivery for text this path removed.
        self._delivered_final_text = ensure_closed_code_fences(
            self._clean_for_display(final_text or "")
        ).strip()
        self._last_sent_text = final_text
        self._fallback_prefix = ""
        self._fallback_preserve_partial_messages = False
        self._notify_new_message()
        return "delivered"

    @staticmethod
    def _send_failure_may_have_delivered(result_or_exc: Any) -> bool:
        """Return True for timeout failures where retrying may duplicate."""
        if getattr(result_or_exc, "retryable", None) is True:
            return False
        error = str(getattr(result_or_exc, "error", None) or result_or_exc).lower()
        name = result_or_exc.__class__.__name__.lower()
        return "timeout" in error or "timed out" in error or "timeout" in name

    def _fallback_flood_retry_delay(self, result: Any) -> float | None:
        """Return a bounded retry delay for a fallback send, if safe to retry."""
        if not self._is_flood_error(result):
            return None
        try:
            delay = float(getattr(result, "retry_after", None) or 3.0)
        except (TypeError, ValueError):
            delay = 3.0
        if delay > self._max_fallback_flood_retry_seconds:
            logger.debug(
                "Flood control requests %.1fs; leaving final delivery to the gateway",
                delay,
            )
            return None
        return max(0.0, delay)

    def _is_flood_error(self, result) -> bool:
        """Check if a SendResult failure is due to flood control / rate limiting."""
        err = getattr(result, "error", "") or ""
        err_lower = err.lower()
        return "flood" in err_lower or "retry after" in err_lower or "rate" in err_lower

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
        _md = dict(self.metadata) if self.metadata else {}
        if self._initial_reply_to_id:
            _md.setdefault("reply_to_message_id", self._initial_reply_to_id)
        try:
            result = await self.adapter.send_draft(
                chat_id=self.chat_id,
                draft_id=self._draft_id,
                content=text,
                metadata=_md or None,
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
            _md = dict(self.metadata) if self.metadata else {}
            if self._initial_reply_to_id:
                _md.setdefault("reply_to_message_id", self._initial_reply_to_id)
            await self.adapter.abandon_open_draft(
                self.chat_id,
                self._last_sent_text or self._clean_for_display(self._accumulated),
                metadata=_md or None,
            )
        except Exception as e:
            logger.debug("abandon_open_draft failed (best-effort): %s", e)

    async def _flush_segment_tail_on_edit_failure(self) -> None:
        """Send the unseen tail after the delivered prefix as a new message before a segment reset.

        Also best-effort strips the stuck cursor from the partial message.
        """
        if not self._fallback_final_send:
            await self._try_strip_cursor()
        visible = self._fallback_prefix or self._visible_prefix()
        tail = self._accumulated
        if visible and tail.startswith(visible):
            tail = tail[len(visible):].lstrip()
        tail = self._clean_for_display(tail)
        if not tail.strip():
            return
        try:
            # Interim: must never seal a native stream (see _send_commentary).
            _md = dict(self.metadata) if self.metadata else {}
            _md["_interim_send"] = True
            result = await self.adapter.send(
                chat_id=self.chat_id,
                content=tail,
                metadata=_md,
            )
            if result.success:
                self._already_sent = True
        except Exception as e:
            logger.error("Segment-break tail flush error: %s", e)

    async def _try_strip_cursor(self) -> None:
        """Best-effort edit removing a stuck cursor when entering fallback mode."""
        if not self._message_id or self._message_id == "__no_edit__":
            return
        prefix = self._visible_prefix()
        if not prefix or not prefix.strip():
            return
        try:
            result = await self._edit_message(
                message_id=self._message_id,
                content=prefix,
            )
            if getattr(result, "success", False):
                self._last_sent_text = prefix
        except Exception:
            pass  # best-effort — don't let this block the fallback path

    async def _send_commentary(self, text: str) -> bool:
        """Send a completed interim assistant commentary message."""
        text = self._clean_for_display(text)
        if not text.strip():
            return False
        try:
            # Interim: a stream-is-the-message adapter's seal-interception must
            # not turn this into draft(final=true), which would seal the live
            # stream with interim text and orphan the true final.
            _md = self._metadata_for_send(final=False) or {}
            _md["_interim_send"] = True
            # reply_to only for reply-anchored threading; Discord/Telegram use
            # thread_id metadata and reply_to on every commentary is spam.
            _plat = getattr(getattr(self.adapter, "platform", None), "value", None)
            _platform_name = str(_plat or getattr(self.adapter, "name", "")).lower()
            _needs_reply_anchor = _platform_name in ("buzz", "slack", "mattermost", "feishu")
            result = await self.adapter.send(
                chat_id=self.chat_id,
                content=text,
                reply_to=self._initial_reply_to_id if _needs_reply_anchor else None,
                metadata=_md,
            )
            # Do NOT set _already_sent: commentary is interim, and the flag
            # would suppress the real final after multiple tool calls.
            if result.success:
                self._notify_new_message()
                # Lets run.py confirm whether an interim send carried the final.
                self._delivered_commentary_texts.append(text)
            return result.success
        except Exception as e:
            logger.error("Commentary send error: %s", e)
            return False

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

    def _raw_message_limit(self) -> int:
        """Per-chat length budget (adapter ``message_len_fn`` units) before overflow splits.

        Rich-capable adapters may raise it via ``streaming_overflow_limit`` so a
        reply that fits one rich message isn't fragmented at the edit limit.
        """
        base = getattr(self.adapter, "MAX_MESSAGE_LENGTH", 4096)
        # isinstance gate keeps MagicMock adapters (mock attrs, not ints) on base.
        if isinstance(self.adapter, _BasePlatformAdapter):
            try:
                base = self.adapter.max_message_length_for_chat(self.chat_id)
            except Exception as e:
                logger.debug("max_message_length_for_chat failed: %s", e)
            try:
                cap = self.adapter.streaming_overflow_limit()
            except Exception as e:
                logger.debug("streaming_overflow_limit check failed: %s", e)
                cap = None
            if isinstance(cap, int) and cap > base:
                return cap
        return base

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
        stale_ids = set(self._preview_message_ids)
        if self._message_id and self._message_id != "__no_edit__":
            stale_ids.add(self._message_id)
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
        delete_fn = getattr(self.adapter, "delete_message", None)
        if delete_fn is not None:
            for stale_id in stale_ids:
                if not stale_id or stale_id == "__no_edit__" or stale_id == new_message_id:
                    continue
                try:
                    await delete_fn(self.chat_id, stale_id)
                except Exception as e:
                    logger.debug(
                        "Fresh-final preview cleanup failed (%s): %s",
                        stale_id, e,
                    )
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

    async def _suppress_silence_marker(self) -> None:
        """Retract any streamed preview when the final reply is a bare silence marker.

        Delivery flags and ``_already_sent`` are left False: nothing was
        delivered, and the gateway's whole-response filter turns the marker
        into "" so no fallback send happens either.
        """
        # A native-stream bubble isn't a deletable message — close an open one
        # (e.g. from an eager re-seed) with an empty finalize so it doesn't hang.
        if self._native_stream_opened:
            try:
                await self.adapter.send_stream_frame(
                    "",
                    finalize=True,
                    chat_id=self.chat_id,
                    reply_to=self._initial_reply_to_id,
                    turn_id=self._turn_id,
                )
            except Exception as e:
                logger.debug(
                    "Silence-marker native stream close failed: %s", e,
                )
            self._native_stream_opened = False
            self._native_last_pushed_len = 0
            self._reopen_seeded_eagerly = False

        stale_ids = set(self._preview_message_ids)
        if self._message_id and self._message_id != "__no_edit__":
            stale_ids.add(self._message_id)
        delete_fn = getattr(self.adapter, "delete_message", None)
        if delete_fn is not None:
            for stale_id in stale_ids:
                if not stale_id or stale_id == "__no_edit__":
                    continue
                try:
                    await delete_fn(self.chat_id, stale_id)
                except Exception as e:
                    logger.debug(
                        "Silence-marker preview cleanup failed (%s): %s",
                        stale_id, e,
                    )
        self._preview_message_ids = set()
        self._message_id = None
        self._accumulated = ""
        self._stream_ledger = ""
        self._last_sent_text = ""
        self._already_sent = False
        self._final_response_sent = False
        self._final_content_delivered = False
        self._delivered_final_text = None
        self._delivery_ambiguous = False
        self._turn_split_delivery = False
        logger.info(
            "Suppressed streamed intentional-silence marker (chat=%s)",
            self.chat_id,
        )

    async def _send_or_edit(
        self, text: str, *, finalize: bool = False, is_turn_final: bool = True,
    ) -> bool:
        """Send or edit the streaming message; True if delivered.

        ``finalize`` marks the last edit of a streaming sequence.  Callers such
        as the overflow split loop use the result to decide whether to advance.
        """
        text = self._clean_for_display(text)
        # Stream-is-the-message draft frames must stay prefix-stable: a closing
        # ``` appended to a mid-code-block frame makes frame N not a prefix of
        # N+1 and the connector re-appends the whole snapshot.  The final
        # message is still fence-closed below.
        _pre_fence_text = text
        text = ensure_closed_code_fences(text)
        # A bare cursor renders as a stray tofu box on some clients.
        visible_without_cursor = text
        if self.cfg.cursor:
            visible_without_cursor = visible_without_cursor.replace(self.cfg.cursor, "")
        _visible_stripped = visible_without_cursor.strip()
        if not _visible_stripped:
            # Native streams MUST still get a finalize frame (placeholder) to
            # close the thinking bubble, e.g. for a MEDIA-only response.
            if finalize and self._use_native_streaming and self._native_stream_opened:
                try:
                    ok = await self.adapter.send_stream_frame(
                        "✅",
                        finalize=True,
                        chat_id=self.chat_id,
                        reply_to=self._initial_reply_to_id,
                        turn_id=self._turn_id,
                    )
                    if ok:
                        self._final_response_sent = True
                        self._final_content_delivered = True
                except Exception as e:
                    logger.debug("Finalize empty stream failed: %s", e)
            return True  # cursor-only / whitespace-only update
        if not text.strip():
            return True  # nothing to send is "success"
        # Don't open a new message for 1-2 tokens + cursor (rapid tool-calling):
        # if the cursor-strip edit is then rate-limited, "X ▉" stays forever.
        # Only first sends are gated.
        _MIN_NEW_MSG_CHARS = 4
        if (self._message_id is None
                and self.cfg.cursor
                and self.cfg.cursor in text
                and len(_visible_stripped) < _MIN_NEW_MSG_CHARS):
            return True  # too short for a standalone message — accumulate more

        # Native streaming: every frame goes through send_stream_frame(); the
        # adapter's send/edit paths are not touched in this mode.  Lazy re-seed
        # here after a boundary closed the stream.
        if self._use_native_streaming and not self._native_stream_opened and text:
            try:
                seed_ok = await self.adapter.send_stream_frame(
                    "",
                    chat_id=self.chat_id,
                    reply_to=self._initial_reply_to_id,
                    turn_id=self._turn_id,
                )
                if seed_ok:
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

        if self._use_native_streaming:
            # WeCom renders each finalize as a separate bubble: only the
            # turn-final and boundaries close the stream, not segment breaks.
            if finalize and not is_turn_final:
                finalize = False

            if not finalize and text == self._last_sent_text:
                return True  # unchanged — skip

            # Mark a finalize frame delivered OPTIMISTICALLY, before the ack
            # wait: the bytes hit the wire (and WeCom renders them) before the
            # ack, so a gateway join-cancel during the ack wait must not strand
            # final_content_delivered=False and cause a duplicate normal send
            # (docs/rca-wecom-stream-final-ack-timeout-duplicate.md).  A
            # definitive dispatch failure rolls the mark back below.  Residual
            # window (cancel between mark and wire write, sub-ms) is accepted.
            _optimistic_finalize = bool(finalize)
            if _optimistic_finalize:
                self._final_response_sent = True
                self._final_content_delivered = True
                # Recorded so a stale/partial frame can't suppress the corrective send.
                self._record_turn_final_payload(text)

            ok = False
            try:
                ok = await self.adapter.send_stream_frame(
                    text,
                    finalize=finalize,
                    chat_id=self.chat_id,
                    reply_to=self._initial_reply_to_id,
                    turn_id=self._turn_id,
                )
            except Exception as e:
                logger.debug(
                    "send_stream_frame raised, disabling native streaming: %s", e,
                )
                ok = False

            if ok:
                self._already_sent = True
                self._last_sent_text = text
                self._native_last_pushed_len = len(text)
                if finalize:
                    self._final_response_sent = True
                    self._final_content_delivered = True
                return True

            # Definitive failure: roll back the optimistic mark so the
            # edit/send fallback delivers exactly once.
            if _optimistic_finalize:
                self._final_response_sent = False
                self._final_content_delivered = False
                self._delivered_final_text = None

            # Subsequent frames take the edit/send fallback; the adapter marks
            # the chat expired so it doesn't retry the dead stream.
            self._use_native_streaming = False

            # Best-effort close of an opened bubble (seed frame has zero length
            # but still opens it — hence _native_stream_opened, not pushed_len).
            if self._native_stream_opened:
                try:
                    await self.adapter.send_stream_frame(
                        text,
                        finalize=True,
                        chat_id=self.chat_id,
                        reply_to=self._initial_reply_to_id,
                        turn_id=self._turn_id,
                    )
                    logger.debug("Native fallback: finalized stream (best-effort close)")
                    # DO NOT mark delivered: the frame closes the bubble but
                    # WeCom may not render the content (errcode 6000 race).
                except Exception as e:
                    logger.debug(
                        "Native fallback: failed to finalize stream: %s", e,
                    )
            # Fall through so accumulated text still reaches the user via edit/send.

        # Drafts have no message_id: the final answer goes through the regular
        # send below (which clears the draft client-side).  Skip drafts when
        # finalizing or when an edit path is already established.  Exception:
        # stream-is-the-message adapters keep ONE stream per turn, so a
        # segment-break finalize must NOT become a real send (seal interception
        # would seal the stream at every tool boundary); only got_done seals.
        _stream_is_msg = self._stream_is_message()
        if (
            self._use_draft_streaming
            and self._message_id is None
            and (not finalize or (_stream_is_msg and not is_turn_final))
        ):
            _frame_text = _pre_fence_text if _stream_is_msg else text
            # Strip the cursor: native streams render their own indicator, and
            # "...text▉" is never a prefix of "...text more▉", which forces the
            # connector's whole-text re-append on EVERY tick (stacked copies).
            if self.cfg.cursor and _frame_text.endswith(self.cfg.cursor):
                _frame_text = _frame_text[: -len(self.cfg.cursor)]
            if _frame_text == self._last_sent_text:
                return True
            ok = await self._send_draft_frame(_frame_text)
            if ok:
                # Deliberately NOT _already_sent: the gateway's fallback final
                # send must still fire so the user gets a real message.
                return True
            # Failure disabled drafts; fall through to edit/send.
        self._last_edit_overflowed = False
        try:
            if self._message_id is not None:
                if self._edit_supported:
                    # REQUIRES_EDIT_FINALIZE adapters need the finalize=True
                    # edit even when unchanged; everyone else short-circuits.
                    if text == self._last_sent_text and not (
                        finalize and self._adapter_requires_finalize
                    ):
                        return True
                    # Fresh-final: replace a long-lived preview with a fresh
                    # message (timestamp reflects completion), or whenever the
                    # adapter prefers it (Telegram's send path renders richer
                    # markdown than its edit path).  An explicit hook returning
                    # False must NOT be overridden by the time threshold — on
                    # Telegram both messages would stay on screen since the
                    # delete is best-effort.  Check the CLASS (MagicMock
                    # auto-creates attrs) plus instance __dict__ (test doubles
                    # assign the hook explicitly).
                    _has_prefers_hook = (
                        hasattr(type(self.adapter),
                                "prefers_fresh_final_streaming")
                        or "prefers_fresh_final_streaming"
                            in getattr(self.adapter, "__dict__", {})
                    )
                    _prefers_fresh = self._adapter_prefers_fresh_final(text)
                    if (
                        finalize
                        and (
                            _prefers_fresh
                            or (
                                not _has_prefers_hook
                                and self._should_send_fresh_final()
                            )
                        )
                        and await self._try_fresh_final(
                            text, is_turn_final=is_turn_final,
                        )
                    ):
                        return True
                    # Edit existing message
                    result = await self._edit_message(
                        message_id=self._message_id,
                        content=text,
                        finalize=finalize,
                    )
                    if result.success:
                        self._already_sent = True
                        self._track_preview_ids_from_result(result)
                        # Oversized edit split across continuations: message_id
                        # is now the LAST continuation, which holds only the
                        # final chunk — retarget edits and reset skip-if-same.
                        # getattr keeps SimpleNamespace test mocks working.
                        _continuation_ids = getattr(result, "continuation_message_ids", ()) or ()
                        if (
                            _continuation_ids
                            and result.message_id
                            and result.message_id != self._message_id
                        ):
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
                    else:
                        immediate_final_fallback = False
                        if (
                            finalize
                            and is_turn_final
                            and self.cfg.cursor
                            and self._last_sent_text.endswith(self.cfg.cursor)
                            and self._visible_prefix() == text
                        ):
                            # Cosmetic final edit was rate-limited but the full
                            # answer is already on screen (cursor stuck): mark
                            # delivered so the gateway doesn't send it twice,
                            # and record the on-screen payload.
                            self._final_content_delivered = True
                            self._record_turn_final_payload(text)
                        raw_response = getattr(result, "raw_response", None)
                        if isinstance(raw_response, dict) and raw_response.get("partial_overflow"):
                            # Some overflow chunks landed but not the whole
                            # response: preserve the visible prefix so got_done
                            # sends the missing tail.
                            self._message_id = str(
                                raw_response.get("last_message_id")
                                or result.message_id
                                or self._message_id
                            )
                            delivered_prefix = raw_response.get("delivered_prefix")
                            if isinstance(delivered_prefix, str) and delivered_prefix:
                                self._last_sent_text = delivered_prefix
                                self._fallback_prefix = delivered_prefix
                                self._fallback_preserve_partial_messages = text.startswith(
                                    delivered_prefix
                                )
                            else:
                                self._fallback_prefix = self._visible_prefix()
                                self._fallback_preserve_partial_messages = False
                            self._fallback_final_send = True
                            self._edit_supported = False
                            self._already_sent = True
                            if getattr(result, "continuation_message_ids", ()):
                                self._notify_new_message()
                            return False

                        # Flood control: adaptive backoff (double the interval);
                        # disable edits only after _MAX_FLOOD_STRIKES in a row.
                        if self._is_flood_error(result):
                            self._flood_strikes += 1
                            self._current_edit_interval = min(
                                self._current_edit_interval * 2, 10.0,
                            )
                            logger.debug(
                                "Flood control on edit (strike %d/%d), "
                                "backoff interval → %.1fs",
                                self._flood_strikes,
                                self._MAX_FLOOD_STRIKES,
                                self._current_edit_interval,
                            )
                            immediate_final_fallback = (
                                finalize
                                and is_turn_final
                                and getattr(
                                    self.adapter,
                                    "FALLBACK_ON_FINAL_EDIT_FLOOD",
                                    False,
                                ) is True
                            )
                            if (
                                self._flood_strikes < self._MAX_FLOOD_STRIKES
                                and not immediate_final_fallback
                            ):
                                self._last_edit_time = time.monotonic()  # honor the new interval
                                return False

                            if immediate_final_fallback:
                                logger.debug(
                                    "Turn-final edit hit flood control; "
                                    "entering fallback immediately"
                                )

                        # Fallback mode: send only the missing tail at got_done.
                        logger.debug(
                            "Edit failed (strikes=%d), entering fallback mode",
                            self._flood_strikes,
                        )
                        self._fallback_prefix = self._visible_prefix()
                        self._fallback_final_send = True
                        self._edit_supported = False
                        self._already_sent = True
                        # A turn-final flood skips the cosmetic cursor strip: it
                        # would burn the same flood budget and delay the answer.
                        if not immediate_final_fallback:
                            await self._try_strip_cursor()
                        return False
                else:
                    return False  # edits unsupported; fallback path sends the final
            else:
                # First send, threaded to the user's message (correct topic/thread).
                result = await self.adapter.send(
                    chat_id=self.chat_id,
                    content=text,
                    reply_to=self._initial_reply_to_id,
                    metadata=self._metadata_for_send(
                        final=finalize,
                        expect_edits=not finalize,
                    ),
                )
                if result.success:
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
                        # Sentinel: no editable id, don't re-enter first-send
                        # on every delta/tool boundary.
                        self._message_id = "__no_edit__"
                    self._notify_new_message()
                    return True
                else:
                    self._edit_supported = False
                    return False
        except Exception as e:
            logger.error("Stream send/edit error: %s", e)
            return False
