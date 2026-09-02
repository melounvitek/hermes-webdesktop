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

    def _reset_stream_delivery_tracking(self) -> None:
        """Reset tracking for text delivered during the current model response."""
        # Flush the think scrubber's benign partial-tag tail first (#17924), then the context scrubber —
        # order matters, the think output feeds the context scrubber.
        think_scrubber = getattr(self, "_stream_think_scrubber", None)
        if think_scrubber is not None:
            think_tail = think_scrubber.flush()
            if think_tail:
                # Route the tail through the context scrubber so a span straddling the boundary is caught.
                ctx_scrubber = getattr(self, "_stream_context_scrubber", None)
                if ctx_scrubber is not None:
                    think_tail = ctx_scrubber.feed(think_tail)
                if think_tail:
                    callbacks = [cb for cb in (self.stream_delta_callback, self._stream_callback) if cb is not None]
                    for cb in callbacks:
                        try:
                            cb(think_tail)
                        except Exception:
                            pass
                    self._record_streamed_assistant_text(think_tail)
        # Flush the context scrubber's benign tail before clearing; mid-span, flush() drops orphaned content.
        scrubber = getattr(self, "_stream_context_scrubber", None)
        if scrubber is not None:
            tail = scrubber.flush()
            if tail:
                callbacks = [cb for cb in (self.stream_delta_callback, self._stream_callback) if cb is not None]
                for cb in callbacks:
                    try:
                        cb(tail)
                    except Exception:
                        pass
                self._record_streamed_assistant_text(tail)
        self._current_streamed_assistant_text = ""

    def _record_streamed_assistant_text(self, text: str) -> None:
        """Accumulate visible assistant text emitted through stream callbacks."""
        # Single-writer guard (#65991): a superseded stream must not pollute the accumulated text, even
        # when reached directly via the tool-suppressed path.
        if self._stream_writer_superseded():
            return
        if isinstance(text, str) and text:
            self._current_streamed_assistant_text = (
                getattr(self, "_current_streamed_assistant_text", "") + text
            )

    @staticmethod
    def _normalize_interim_visible_text(text: str) -> str:
        if not isinstance(text, str):
            return ""
        return re.sub(r"\s+", " ", text).strip()

    def _interim_content_was_streamed(self, content: str) -> bool:
        visible_content = self._normalize_interim_visible_text(
            self._strip_think_blocks(content or "")
        )
        if not visible_content:
            return False
        streamed = self._normalize_interim_visible_text(
            self._strip_think_blocks(getattr(self, "_current_streamed_assistant_text", "") or "")
        )
        # Prefix match, not equality: the final may be streamed text plus a trailing delta, or the stream
        # partial. The reverse (streamed longer) is NOT matched — it could suppress a needed resend (#65919).
        return bool(streamed) and visible_content.startswith(streamed)

    def _extract_codex_interim_visible_parts(
        self,
        assistant_msg: Dict[str, Any],
    ) -> List[str]:
        """Extract visible Codex commentary as one string per message item.

        Codex keeps mid-turn narration as ``phase=commentary`` items while the final answer stays in
        ``content``;
        non-streaming gateways need it via the interim callback. ``phase=analysis`` stays hidden (scratchpad).
        """
        if not getattr(self, "show_commentary", True):
            # display.show_commentary=false — commentary stays on the
            # reasoning channel (pre-commentary-channel behavior).
            return []
        items = assistant_msg.get("codex_message_items")
        if not isinstance(items, list):
            return []

        messages: List[str] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            if item.get("type") != "message":
                continue
            phase = item.get("phase")
            if not isinstance(phase, str) or phase.strip().lower() != "commentary":
                continue
            content_parts = item.get("content")
            if not isinstance(content_parts, list):
                continue
            item_parts: List[str] = []
            for part in content_parts:
                if not isinstance(part, dict):
                    continue
                if part.get("type") != "output_text":
                    continue
                text = part.get("text")
                if isinstance(text, str) and text.strip():
                    item_parts.append(text)
            visible = "".join(item_parts).strip()
            if visible:
                visible = self._strip_think_blocks(visible).strip()
                visible = redact_sensitive_text(visible)
            if visible:
                messages.append(visible)
        return messages

    def _extract_codex_interim_visible_text(self, assistant_msg: Dict[str, Any]) -> str:
        """Extract all visible Codex commentary for comparison/fallback."""
        return "\n\n".join(
            self._extract_codex_interim_visible_parts(assistant_msg)
        ).strip()

    def _interim_assistant_visible_text(self, assistant_msg: Dict[str, Any]) -> str:
        """Return the exact assistant text eligible for interim delivery.

        Prefers structured Codex commentary over top-level content — a response can hold commentary AND a
        partial final answer while tools are pending, and treating content as progress leaks the answer early.
        Content may be a parts list, so flatten before stripping reasoning.
        """
        visible = self._extract_codex_interim_visible_text(assistant_msg)
        if visible:
            return visible
        content = assistant_msg.get("content")
        return self._strip_think_blocks(flatten_message_text(content)).strip()

    def _interim_text_was_delivered(self, text: str) -> bool:
        normalized = self._normalize_interim_visible_text(text)
        if not normalized:
            return False
        return normalized in getattr(self, "_delivered_interim_texts", set())

    def _record_delivered_interim_text(self, text: str) -> None:
        normalized = self._normalize_interim_visible_text(text)
        if normalized:
            delivered = getattr(self, "_delivered_interim_texts", None)
            if not isinstance(delivered, set):
                delivered = set()
                self._delivered_interim_texts = delivered
            delivered.add(normalized)

    def _fire_streamed_codex_commentary(self, text: str) -> None:
        """Deliver a completed live Codex commentary message immediately."""
        cb = getattr(self, "interim_assistant_callback", None)
        if cb is None or not isinstance(text, str):
            return
        visible = self._strip_think_blocks(text).strip()
        if visible:
            visible = redact_sensitive_text(visible)
        if not visible or visible == "(empty)" or self._interim_text_was_delivered(visible):
            return
        try:
            cb(visible, already_streamed=False)
            self._record_delivered_interim_text(visible)
        except Exception:
            logger.debug("interim_assistant_callback error", exc_info=True)

    def _emit_interim_assistant_message(
        self, assistant_msg: Dict[str, Any]
    ) -> None:
        """Surface a real mid-turn assistant commentary message to the UI layer.

        Does NOT set ``_response_was_previewed`` — that means "the final response was shown"; setting it for
        narration would make the CLI suppress a different final summary (response-loss blocker).
        """
        if not isinstance(assistant_msg, dict):
            return
        commentary_parts = self._extract_codex_interim_visible_parts(assistant_msg)
        undelivered_parts: List[str] = []
        pending_keys: set[str] = set()
        for part in commentary_parts:
            key = self._normalize_interim_visible_text(part)
            if (
                not key
                or key in pending_keys
                or self._interim_text_was_delivered(part)
            ):
                continue
            pending_keys.add(key)
            undelivered_parts.append(part)
        visible = (
            "\n\n".join(undelivered_parts).strip()
            if commentary_parts
            else self._interim_assistant_visible_text(assistant_msg)
        )
        if (
            not visible
            or visible == "(empty)"
            or self._interim_text_was_delivered(visible)
        ):
            return
        already_streamed = self._interim_content_was_streamed(visible)
        try:
            from agent.plugin_stream_hooks import enqueue_plugin_stream_hook

            enqueue_plugin_stream_hook(
                "on_interim_message",
                turn_id=getattr(self, "_current_turn_id", "") or "",
                iteration=int(getattr(self, "_api_call_count", 0) or 0),
                session_id=self.session_id or "",
                model=self.model or "",
                provider=self.provider or "",
                surface=self.platform or "cli",
                text=visible,
                already_streamed=already_streamed,
            )
        except Exception:
            logger.debug("on_interim_message plugin hook enqueue failed", exc_info=True)
        cb = getattr(self, "interim_assistant_callback", None)
        if cb is None:
            return
        try:
            cb(visible, already_streamed=already_streamed)
            if undelivered_parts:
                for part in undelivered_parts:
                    self._record_delivered_interim_text(part)
            else:
                self._record_delivered_interim_text(visible)
        except Exception:
            logger.debug("interim_assistant_callback error", exc_info=True)

    def _ensure_stream_writer_state(self) -> None:
        """Lazily create the single-writer guard fields.

        Normally set in ``agent_init``; ``AIAgent.__new__``-built instances skip that path and must not crash.
        """
        if getattr(self, "_stream_writer_lock", None) is None:
            self._stream_writer_lock = threading.Lock()
        if not hasattr(self, "_stream_writer_token"):
            self._stream_writer_token = 0
        if getattr(self, "_stream_writer_tls", None) is None:
            self._stream_writer_tls = threading.local()
        if not hasattr(self, "_stream_writer_dropped"):
            self._stream_writer_dropped = 0

    def _claim_stream_writer(self) -> int:
        """Claim exclusive ownership of the streaming delta sink for this stream attempt; returns its writer
        token.

        Every attempt (each provider path, each retry) claims right before consuming. Claiming bumps the
        shared token, so an earlier attempt still alive on another thread is superseded and its late chunks
        fenced out. Stored per-thread: a thread that never claimed is never a writer and can never be fenced.
        """
        self._ensure_stream_writer_state()
        with self._stream_writer_lock:
            self._stream_writer_token += 1
            token = self._stream_writer_token
        self._stream_writer_tls.token = token
        return token

    def _stream_writer_is_current(self, token: int) -> bool:
        """True when ``token`` is still the active writer (no newer attempt claimed since).

        Lets a stream loop bail out the instant it is superseded.
        """
        return token == getattr(self, "_stream_writer_token", token)

    def _stream_writer_superseded(self) -> bool:
        """True when this thread claimed the sink but a newer attempt has since claimed it (stale writer, drop
        chunks).

        A thread that never claimed (``token is None``) is never reported superseded.
        """
        tls = getattr(self, "_stream_writer_tls", None)
        token = getattr(tls, "token", None) if tls is not None else None
        if token is None:
            return False
        return token != getattr(self, "_stream_writer_token", token)

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
        try:
            from agent.plugin_stream_hooks import enqueue_plugin_stream_hook

            enqueue_plugin_stream_hook("on_stream_start", **self._stream_hook_base_payload())
        except Exception:
            logger.debug("on_stream_start plugin hook enqueue failed", exc_info=True)

    def _emit_stream_end(self, *, final_text: str, finished: bool, error: str | None) -> None:
        try:
            from agent.plugin_stream_hooks import enqueue_plugin_stream_hook

            enqueue_plugin_stream_hook(
                "on_stream_end",
                **self._stream_hook_base_payload(),
                final_text=final_text,
                finished=finished,
                error=error,
            )
        except Exception:
            logger.debug("on_stream_end plugin hook enqueue failed", exc_info=True)

    def _fire_stream_delta(self, text: str) -> None:
        """Fire all registered stream delta callbacks (display + TTS)."""
        # Single-writer guard (#65991): a superseded stream must not interleave
        # its tokens into the turn alongside the retry that replaced it.
        if self._stream_writer_superseded():
            self._note_dropped_stream_writer("_fire_stream_delta")
            return
        # Prepend one paragraph break before the first text delta after a tool iteration, without
        # stacking blank lines across back-to-back tool iterations.
        if getattr(self, "_stream_needs_break", False) and text and text.strip():
            self._stream_needs_break = False
            text = "\n\n" + text
            prepended_break = True
        else:
            prepended_break = False
        if isinstance(text, str):
            # Stateful scrubber (#17924): per-delta regex stripping destroyed downstream state machines when a
            # tag was split across deltas (MiniMax-M2.7 sends '<think>' separately).
            think_scrubber = getattr(self, "_stream_think_scrubber", None)
            if think_scrubber is not None:
                text = think_scrubber.feed(text or "")
            else:
                # Defensive: legacy callers without the scrubber attribute.
                text = self._strip_think_blocks(text or "")
            # Then feed through the stateful context scrubber so memory-context
            # spans split across chunks cannot leak to the UI (#5719).
            scrubber = getattr(self, "_stream_context_scrubber", None)
            if scrubber is not None:
                text = scrubber.feed(text)
            else:
                # Defensive: legacy callers without the scrubber attribute.
                text = sanitize_context(text)
            # Only strip leading newlines on the first delta — mid-stream "\n" is legitimate markdown.
            if not prepended_break and not getattr(
                self, "_current_streamed_assistant_text", ""
            ):
                text = text.lstrip("\n")
        if not text:
            return
        callbacks = [cb for cb in (self.stream_delta_callback, self._stream_callback) if cb is not None]
        delivered = False
        for cb in callbacks:
            try:
                cb(text)
                delivered = True
            except Exception:
                pass
        try:
            from agent.plugin_stream_hooks import enqueue_plugin_stream_hook

            enqueue_plugin_stream_hook(
                "on_stream_delta",
                **self._stream_hook_base_payload(),
                delta=text,
                kind="text",
            )
        except Exception:
            logger.debug("on_stream_delta plugin hook enqueue failed", exc_info=True)
        if delivered:
            self._record_streamed_assistant_text(text)

    def _fire_reasoning_delta(self, text: str) -> None:
        """Fire reasoning callback if registered."""
        # Single-writer guard (#65991): fence out a superseded stream's
        # reasoning deltas the same way as content deltas.
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
            from agent.plugin_stream_hooks import enqueue_plugin_stream_hook, stream_reasoning_deltas_enabled

            if stream_reasoning_deltas_enabled():
                enqueue_plugin_stream_hook(
                    "on_stream_delta",
                    **self._stream_hook_base_payload(),
                    delta=text,
                    kind="reasoning",
                )
        except Exception:
            logger.debug("reasoning on_stream_delta plugin hook enqueue failed", exc_info=True)

    def _fire_tool_gen_started(self, tool_name: str) -> None:
        """Notify the display layer that the model is generating tool call arguments.

        Fires once per tool name so the TUI can show a spinner while a large payload (e.g. a 45 KB write_file)
        streams.
        """
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
