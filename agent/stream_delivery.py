"""Streaming / interim-message delivery for ``AIAgent``.

Single-writer stream ownership, delta/reasoning hook fan-out, and interim assistant text dedup.
Extracted from ``run_agent.py``; every method resolves through ``AIAgent``'s MRO unchanged.
"""
import logging
import re
import threading
from typing import Any, Dict, List

from agent.memory_manager import sanitize_context
from agent.message_content import flatten_message_text
from agent.redact import redact_sensitive_text

# Same logger name as the origin module so log records / caplog filters are unchanged.
logger = logging.getLogger("run_agent")


class StreamDeliveryMixin:
    """Stream ownership, delta/reasoning hook fan-out and interim-text dedup (see module docstring)."""

    # ── stream text delivery ──

    def _deliver_to_stream_callbacks(self, text: str) -> bool:
        """Send ``text`` to the display + TTS delta callbacks; True if at least one accepted it."""
        delivered = False
        for cb in (self.stream_delta_callback, self._stream_callback):
            if cb is None:
                continue
            try:
                cb(text)
                delivered = True
            except Exception:
                pass
        return delivered

    def _enqueue_stream_hook(self, event: str, *, label: str | None = None, **fields: Any) -> None:
        """Best-effort plugin stream hook enqueue; never raises into the stream path."""
        try:
            from agent.plugin_stream_hooks import enqueue_plugin_stream_hook

            enqueue_plugin_stream_hook(event, **self._stream_hook_base_payload(), **fields)
        except Exception:
            logger.debug("%s plugin hook enqueue failed", label or event, exc_info=True)

    def _reset_stream_delivery_tracking(self) -> None:
        """Reset tracking for text delivered during the current model response.

        Flushes the think scrubber's benign partial-tag tail first, routed through the
        context scrubber (a span straddling the boundary must still be caught), then the
        context scrubber's own tail. Mid-span, ``flush()`` drops orphaned content.
        """
        think_scrubber = getattr(self, "_stream_think_scrubber", None)
        ctx_scrubber = getattr(self, "_stream_context_scrubber", None)
        if think_scrubber is not None:
            think_tail = think_scrubber.flush()
            if think_tail and ctx_scrubber is not None:
                think_tail = ctx_scrubber.feed(think_tail)
            if think_tail:
                self._deliver_to_stream_callbacks(think_tail)
                self._record_streamed_assistant_text(think_tail)
        if ctx_scrubber is not None:
            tail = ctx_scrubber.flush()
            if tail:
                self._deliver_to_stream_callbacks(tail)
                self._record_streamed_assistant_text(tail)
        self._current_streamed_assistant_text = ""

    def _record_streamed_assistant_text(self, text: str) -> None:
        """Accumulate visible assistant text emitted through stream callbacks.

        A superseded stream writer must not pollute the accumulated text, even via the
        tool-suppressed path.
        """
        if self._stream_writer_superseded():
            return
        if isinstance(text, str) and text:
            self._current_streamed_assistant_text = (
                getattr(self, "_current_streamed_assistant_text", "") + text
            )

    # ── interim assistant text ──

    @staticmethod
    def _normalize_interim_visible_text(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip() if isinstance(text, str) else ""

    def _interim_content_was_streamed(self, content: str) -> bool:
        visible_content = self._normalize_interim_visible_text(self._strip_think_blocks(content or ""))
        if not visible_content:
            return False
        streamed = self._normalize_interim_visible_text(
            self._strip_think_blocks(getattr(self, "_current_streamed_assistant_text", "") or "")
        )
        # Prefix match, not equality: the final may be streamed text plus a trailing delta. The
        # reverse (streamed longer) is NOT matched — it could suppress a needed resend.
        return bool(streamed) and visible_content.startswith(streamed)

    def _extract_codex_interim_visible_parts(self, assistant_msg: Dict[str, Any]) -> List[str]:
        """Visible Codex commentary, one string per message item.

        Codex keeps mid-turn narration as ``phase=commentary`` items while the final answer
        stays in ``content``; ``phase=analysis`` stays hidden (scratchpad). With
        ``display.show_commentary=false`` commentary stays on the reasoning channel.
        """
        if not getattr(self, "show_commentary", True):
            return []
        items = assistant_msg.get("codex_message_items")
        if not isinstance(items, list):
            return []

        messages: List[str] = []
        for item in items:
            if not isinstance(item, dict) or item.get("type") != "message":
                continue
            phase = item.get("phase")
            if not isinstance(phase, str) or phase.strip().lower() != "commentary":
                continue
            content_parts = item.get("content")
            if not isinstance(content_parts, list):
                continue
            visible = "".join(
                part["text"] for part in content_parts
                if isinstance(part, dict) and part.get("type") == "output_text"
                and isinstance(part.get("text"), str) and part["text"].strip()
            ).strip()
            if visible:
                visible = redact_sensitive_text(self._strip_think_blocks(visible).strip())
            if visible:
                messages.append(visible)
        return messages

    def _extract_codex_interim_visible_text(self, assistant_msg: Dict[str, Any]) -> str:
        """All visible Codex commentary joined, for comparison/fallback."""
        return "\n\n".join(self._extract_codex_interim_visible_parts(assistant_msg)).strip()

    def _interim_assistant_visible_text(self, assistant_msg: Dict[str, Any]) -> str:
        """The exact assistant text eligible for interim delivery.

        Prefers structured Codex commentary over top-level content — a response can hold
        commentary AND a partial final answer while tools are pending, and treating content
        as progress leaks the answer early. Content may be a parts list.
        """
        return (
            self._extract_codex_interim_visible_text(assistant_msg)
            or self._strip_think_blocks(flatten_message_text(assistant_msg.get("content"))).strip()
        )

    def _interim_text_was_delivered(self, text: str) -> bool:
        normalized = self._normalize_interim_visible_text(text)
        return bool(normalized) and normalized in getattr(self, "_delivered_interim_texts", set())

    def _record_delivered_interim_text(self, text: str) -> None:
        normalized = self._normalize_interim_visible_text(text)
        if normalized:
            delivered = getattr(self, "_delivered_interim_texts", None)
            if not isinstance(delivered, set):
                delivered = set()
                self._delivered_interim_texts = delivered
            delivered.add(normalized)

    def _deliver_interim(self, visible: str, *, already_streamed: bool, record: List[str]) -> None:
        """Hand ``visible`` to ``interim_assistant_callback`` and mark ``record`` delivered; swallows callback errors."""
        cb = getattr(self, "interim_assistant_callback", None)
        if cb is None:
            return
        try:
            cb(visible, already_streamed=already_streamed)
            for part in record:
                self._record_delivered_interim_text(part)
        except Exception:
            logger.debug("interim_assistant_callback error", exc_info=True)

    def _fire_streamed_codex_commentary(self, text: str) -> None:
        """Deliver a completed live Codex commentary message immediately."""
        if getattr(self, "interim_assistant_callback", None) is None or not isinstance(text, str):
            return
        visible = self._strip_think_blocks(text).strip()
        if visible:
            visible = redact_sensitive_text(visible)
        if not visible or visible == "(empty)" or self._interim_text_was_delivered(visible):
            return
        self._deliver_interim(visible, already_streamed=False, record=[visible])

    def _emit_interim_assistant_message(self, assistant_msg: Dict[str, Any]) -> None:
        """Surface a real mid-turn assistant commentary message to the UI layer.

        Does NOT set ``_response_was_previewed`` — that means "the final response was shown";
        setting it for narration would make the CLI suppress a different final summary.
        """
        if not isinstance(assistant_msg, dict):
            return
        commentary_parts = self._extract_codex_interim_visible_parts(assistant_msg)
        undelivered_parts: List[str] = []
        pending_keys: set[str] = set()
        for part in commentary_parts:
            key = self._normalize_interim_visible_text(part)
            if not key or key in pending_keys or self._interim_text_was_delivered(part):
                continue
            pending_keys.add(key)
            undelivered_parts.append(part)
        visible = (
            "\n\n".join(undelivered_parts).strip()
            if commentary_parts
            else self._interim_assistant_visible_text(assistant_msg)
        )
        if not visible or visible == "(empty)" or self._interim_text_was_delivered(visible):
            return
        already_streamed = self._interim_content_was_streamed(visible)
        self._enqueue_stream_hook("on_interim_message", text=visible, already_streamed=already_streamed)
        self._deliver_interim(visible, already_streamed=already_streamed, record=undelivered_parts or [visible])

    # ── single-writer stream fence ──

    def _ensure_stream_writer_state(self) -> None:
        """Lazily create the single-writer guard fields (``AIAgent.__new__``-built instances skip ``agent_init``)."""
        if getattr(self, "_stream_writer_lock", None) is None:
            self._stream_writer_lock = threading.Lock()
        if not hasattr(self, "_stream_writer_token"):
            self._stream_writer_token = 0
        if getattr(self, "_stream_writer_tls", None) is None:
            self._stream_writer_tls = threading.local()
        if not hasattr(self, "_stream_writer_dropped"):
            self._stream_writer_dropped = 0

    def _claim_stream_writer(self) -> int:
        """Claim exclusive ownership of the delta sink for this stream attempt; returns its writer token.

        Every attempt (each provider path, each retry) claims right before consuming. Claiming
        bumps the shared token, so an earlier attempt still alive on another thread is
        superseded and its late chunks fenced out. Stored per-thread: a thread that never
        claimed is never a writer and can never be fenced.
        """
        self._ensure_stream_writer_state()
        with self._stream_writer_lock:
            self._stream_writer_token = token = self._stream_writer_token + 1
        self._stream_writer_tls.token = token
        return token

    def _stream_writer_is_current(self, token: int) -> bool:
        """True when ``token`` is still the active writer, so a stream loop can bail the instant it is superseded."""
        return token == getattr(self, "_stream_writer_token", token)

    def _stream_writer_superseded(self) -> bool:
        """True when this thread claimed the sink but a newer attempt has since claimed it.

        A thread that never claimed (``token is None``) is never reported superseded.
        """
        token = getattr(getattr(self, "_stream_writer_tls", None), "token", None)
        return token is not None and token != getattr(self, "_stream_writer_token", token)

    def _note_dropped_stream_writer(self, where: str) -> None:
        """Record + log that a superseded stream's delta was discarded."""
        try:
            self._stream_writer_dropped = int(getattr(self, "_stream_writer_dropped", 0)) + 1
        except Exception:
            self._stream_writer_dropped = 1
        # Log sparsely (first drop, then powers of two) so a chatty superseded stream can't flood the log.
        _n = self._stream_writer_dropped
        if _n == 1 or (_n & (_n - 1)) == 0:
            logger.warning(
                "Dropped delta from a superseded stream writer at %s "
                "(discarded=%d this turn) — a stale stream tried to write into "
                "the turn after a retry superseded it.",
                where, _n,
            )

    # ── hook fan-out ──

    def _stream_hook_base_payload(self) -> Dict[str, Any]:
        return {
            "turn_id": getattr(self, "_current_turn_id", "") or "",
            "iteration": int(getattr(self, "_api_call_count", 0) or 0),
            "session_id": self.session_id or "",
            "model": self.model or "",
            "provider": self.provider or "",
            "surface": self.platform or "cli",
        }

    def _emit_stream_start(self) -> None:
        self._enqueue_stream_hook("on_stream_start")

    def _emit_stream_end(self, *, final_text: str, finished: bool, error: str | None) -> None:
        self._enqueue_stream_hook("on_stream_end", final_text=final_text, finished=finished, error=error)

    def _fire_stream_delta(self, text: str) -> None:
        """Fire all registered stream delta callbacks (display + TTS)."""
        # A superseded stream must not interleave its tokens alongside the retry that replaced it.
        if self._stream_writer_superseded():
            self._note_dropped_stream_writer("_fire_stream_delta")
            return
        # One paragraph break before the first text delta after a tool iteration, without
        # stacking blank lines across back-to-back tool iterations.
        prepended_break = bool(getattr(self, "_stream_needs_break", False) and text and text.strip())
        if prepended_break:
            self._stream_needs_break = False
            text = "\n\n" + text
        if isinstance(text, str):
            # Stateful scrubbers: per-delta regex stripping destroyed downstream state machines
            # when a tag was split across deltas ('<think>' sent alone); memory-context spans
            # split across chunks must not leak to the UI. Legacy callers lack the attributes.
            think_scrubber = getattr(self, "_stream_think_scrubber", None)
            text = think_scrubber.feed(text or "") if think_scrubber is not None else self._strip_think_blocks(text or "")
            scrubber = getattr(self, "_stream_context_scrubber", None)
            text = scrubber.feed(text) if scrubber is not None else sanitize_context(text)
            # Only strip leading newlines on the first delta — mid-stream "\n" is legitimate markdown.
            if not prepended_break and not getattr(self, "_current_streamed_assistant_text", ""):
                text = text.lstrip("\n")
        if not text:
            return
        delivered = self._deliver_to_stream_callbacks(text)
        self._enqueue_stream_hook("on_stream_delta", delta=text, kind="text")
        if delivered:
            self._record_streamed_assistant_text(text)

    def _fire_reasoning_delta(self, text: str) -> None:
        """Fire reasoning callback if registered; superseded writers are fenced like content deltas."""
        if self._stream_writer_superseded():
            self._note_dropped_stream_writer("_fire_reasoning_delta")
            return
        cb = self.reasoning_callback
        if cb is not None:
            try:
                cb(text)
            except Exception:
                pass
        try:
            from agent.plugin_stream_hooks import stream_reasoning_deltas_enabled

            enabled = stream_reasoning_deltas_enabled()
        except Exception:
            logger.debug("reasoning on_stream_delta plugin hook enqueue failed", exc_info=True)
            return
        if enabled:
            self._enqueue_stream_hook("on_stream_delta", label="reasoning on_stream_delta", delta=text, kind="reasoning")

    def _fire_tool_gen_started(self, tool_name: str) -> None:
        """Notify the display layer that the model is generating tool call arguments (spinner for large payloads)."""
        cb = self.tool_gen_callback
        if cb is not None:
            try:
                cb(tool_name)
            except Exception:
                pass

    def _has_stream_consumers(self) -> bool:
        """Return True if any streaming consumer is registered."""
        try:
            from agent.plugin_stream_hooks import has_stream_observer_hooks

            if has_stream_observer_hooks():
                return True
        except Exception:
            logger.debug("plugin stream hook consumer check failed", exc_info=True)
        return (
            self.stream_delta_callback is not None
            or getattr(self, "_stream_callback", None) is not None
        )
