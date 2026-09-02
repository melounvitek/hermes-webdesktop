"""User-facing status / warning / notice plumbing for ``AIAgent``.

Safe printing, quiet-mode gating, deduped context-overflow warnings, and the buffered retry chatter that
is shown only when every retry/fallback is exhausted.
Extracted from ``run_agent.py``; every method resolves through ``AIAgent``'s MRO unchanged.
"""
import logging
import sys

from agent.session_activity import ActivityProvenance

# Same logger name as the origin module so log records / caplog filters are unchanged.
logger = logging.getLogger("run_agent")


class StatusOutputMixin:
    """Status/warning/notice emission and retry-chatter buffering (see module docstring)."""

    def _safe_print(self, *args, **kwargs):
        """Print that silently handles broken pipes / closed stdout.

        Headless stdout (systemd, Docker, nohup) can vanish mid-session and a raw ``print()`` would crash cron
        jobs.
        Routes through ``self._print_fn`` so the CLI can inject an ANSI-aware renderer.
        """
        try:
            fn = self._print_fn or print
            fn(*args, **kwargs)
        except (OSError, ValueError):
            pass

    def _vprint(self, *args, force: bool = False, **kwargs):
        """Verbose print — suppressed when actively streaming tokens.

        ``force=True`` always shows (errors/warnings). Printing is allowed during tool execution (no tokens
        streaming), muted after the main response (``_mute_post_response``), and fully suppressed under
        ``suppress_status_output`` (machine-readable ``hermes chat -q``).
        """
        if getattr(self, "suppress_status_output", False):
            return
        if not force and getattr(self, "_mute_post_response", False):
            return
        if not force and self._has_stream_consumers() and not self._executing_tools:
            return
        self._safe_print(*args, **kwargs)

    def _should_start_quiet_spinner(self) -> bool:
        """Return True when quiet-mode spinner output has a safe sink.

        A raw spinner falling back to ``sys.stdout`` can corrupt protocol streams (ACP JSON-RPC); allow it
        only when output is rerouted via ``_print_fn`` or stdout is a real TTY.
        """
        if self._print_fn is not None:
            return True
        stream = getattr(sys, "stdout", None)
        if stream is None:
            return False
        try:
            return bool(stream.isatty())
        except (AttributeError, ValueError, OSError):
            return False

    def _should_emit_quiet_tool_messages(self) -> bool:
        """Return True when quiet-mode tool summaries should print directly.

        The CLI wants compact hints when no callback owns rendering; embedded callers expect true silence.
        ``suppress_status_output`` always wins so ``[tool]``/``[done]`` lines never land in captured stdout.
        """
        if getattr(self, "suppress_status_output", False):
            return False
        return (
            self.quiet_mode
            and not self.tool_progress_callback
            and getattr(self, "platform", "") == "cli"
        )

    def _emit_status(self, message: str) -> None:
        """Emit a lifecycle status message to both CLI (``_vprint(force=True)``) and gateway
        (``status_callback``).

        Never raises — it must not interrupt the retry/fallback logic.
        """
        try:
            self._vprint(f"{self.log_prefix}{message}", force=True)
        except Exception:
            pass
        if self.status_callback:
            try:
                self.status_callback("lifecycle", message)
            except Exception:
                logger.debug("status_callback error in _emit_status", exc_info=True)

    def _emit_warning(self, message: str) -> None:
        """Emit a user-visible warning through the same status plumbing.

        For degraded side paths (auxiliary compression, memory flushes) where the turn continues but the user
        must know.
        """
        try:
            self._vprint(f"{self.log_prefix}{message}", force=True)
        except Exception:
            pass
        if self.status_callback:
            try:
                self.status_callback("warn", message)
            except Exception:
                logger.debug("status_callback error in _emit_warning", exc_info=True)

    def _warn_context_overflow_blocked(
        self, reason: str, preflight_tokens: int, threshold_tokens: int
    ) -> None:
        """Warn (deduped) when context is over the compression threshold but compression is blocked.

        Without this the session grows until the provider hard limit with no explanation. Dedup is on the
        block
        *kind* (``cooldown`` / ``ineffective``), not the countdown string; cleared by
        ``_clear_context_overflow_warn``.
        """
        _warn_kind = (reason or "unknown").split(":", 1)[0]
        _warn_key = ("ctx_overflow_blocked", _warn_kind)
        if getattr(self, "_last_ctx_overflow_warn", None) != _warn_key:
            self._last_ctx_overflow_warn = _warn_key
            from agent.conversation_compression import (
                CONTEXT_OVERFLOW_BLOCKED_WARNING_TEMPLATE,
            )
            # cooldown + anti-thrash (ineffective) are both "compression blocked".
            if _warn_kind in ("cooldown", "ineffective"):
                self._touch_activity(
                    f"compression blocked ({reason})",
                    provenance=ActivityProvenance.AGENT_COMPRESSION_COOLDOWN,
                )
            self._emit_warning(
                CONTEXT_OVERFLOW_BLOCKED_WARNING_TEMPLATE.format(
                    tokens=preflight_tokens,
                    threshold=threshold_tokens,
                    reason=reason,
                )
            )

    def _warn_uncompressed_context_overflow(
        self, preflight_tokens: int, context_length: int
    ) -> None:
        """Surface a deduped warning when uncompressed context exceeds the model limit.

        With compression disabled the session can outgrow the window silently; point the user at /compact.
        """
        _warn_key = ("uncompressed_ctx_overflow", context_length)
        if getattr(self, "_last_ctx_overflow_warn", None) != _warn_key:
            self._last_ctx_overflow_warn = _warn_key
            self._emit_warning(
                f"⚠️ Session context (~{preflight_tokens:,} tokens) exceeds the model "
                f"context window (~{context_length:,} tokens) with compression disabled "
                f"(compression.enabled: false). Use /compact to compress history or "
                f"enable compression in config.yaml."
            )

    def _clear_context_overflow_warn(self) -> None:
        """Reset the blocked-overflow warning dedup so it can re-fire on the next blocked turn."""
        self._last_ctx_overflow_warn = None

    def _emit_notice(self, notice) -> None:
        """Fire a structured ``AgentNotice`` to the active driver (TUI / CLI).

        Driver-agnostic; swallows callback errors — a notice must never break the agent loop.
        """
        if self.notice_callback:
            try:
                self.notice_callback(notice)
            except Exception:
                logger.debug("notice_callback error in _emit_notice", exc_info=True)

    def _emit_notice_clear(self, key: str) -> None:
        """Clear a previously-fired sticky notice by ``key`` (e.g. on recovery)."""
        if self.notice_clear_callback:
            try:
                self.notice_clear_callback(key)
            except Exception:
                logger.debug("notice_clear_callback error in _emit_notice_clear", exc_info=True)

    def _emit_wait_notice(self, text: str) -> None:
        """Surface a live wait-state explanation on every driver.

        Rewrites the live status line (CLI spinner via ``thinking_callback``, TUI/Desktop ``thinking.delta``,
        gateway ``_touch_activity`` description) so long provider waits are not an anonymous spinner. Never
        raises.
        """
        self._touch_activity(text)
        _thinking_cb = getattr(self, "thinking_callback", None)
        if _thinking_cb:
            try:
                _thinking_cb(text)
            except Exception:
                logger.debug("thinking_callback error in _emit_wait_notice", exc_info=True)

    # ── Buffered retry/fallback status ──
    # Retry chatter is buffered and shown only when every retry/fallback is exhausted; dropped on
    # success. Backend logs are unaffected (every site still logs).

    def _buffer_status(self, message: str) -> None:
        """Buffer a retry/fallback status message as ``(kind, text)``.

        ``kind`` is ``"status"`` (replays via ``_emit_status``), ``"vprint"`` (``_vprint(force=True)``) or
        ``"warn"`` (``_emit_warning``). Deferred until we know whether the turn recovered.
        """
        try:
            buf = getattr(self, "_retry_status_buffer", None)
            if buf is None:
                buf = []
                self._retry_status_buffer = buf
            buf.append(("status", message))
        except Exception:
            # Never break the retry loop on a buffer hiccup.
            pass

    def _buffer_vprint(self, message: str) -> None:
        """Buffer a vprint(force=True) retry/fallback line."""
        try:
            buf = getattr(self, "_retry_status_buffer", None)
            if buf is None:
                buf = []
                self._retry_status_buffer = buf
            buf.append(("vprint", message))
        except Exception:
            pass

    def _clear_status_buffer(self) -> None:
        """Drop buffered retry messages — call on successful recovery."""
        try:
            buf = getattr(self, "_retry_status_buffer", None)
            if buf:
                buf.clear()
        except Exception:
            pass

    def _emit_pending_fallback_notice(self) -> None:
        """Surface the one-shot fallback-switch notice on successful recovery.

        A provider switch is durable state operators must see, unlike the transient retry chatter
        ``_clear_status_buffer`` drops. Emitted exactly once, then cleared; on terminal failure the buffered
        switch line is flushed instead (``_flush_status_buffer``).
        """
        try:
            notice = getattr(self, "_pending_fallback_notice", None)
            if notice:
                # Clear before emitting so a (swallowed) callback error can't
                # leave the notice set for a stale re-emit on a later turn.
                self._pending_fallback_notice = None
                notices = notice if isinstance(notice, list) else [notice]
                for item in notices:
                    try:
                        self._emit_status(str(item))
                    except Exception:
                        # A single surface callback failure must not hide later
                        # switches from the same fallback chain.
                        continue
        except Exception:
            # Never break the conversation loop on a notice hiccup.
            pass

    def _flush_status_buffer(self) -> None:
        """Emit buffered retry messages — call on terminal failure so the user sees what was tried."""
        try:
            # The buffered trace already carries the switch line; drop the one-shot notice to avoid a stale
            # duplicate.
            self._pending_fallback_notice = None
            buf = getattr(self, "_retry_status_buffer", None)
            if not buf:
                return
            # Drain first so a callback exception doesn't double-emit.
            messages = list(buf)
            buf.clear()
            for kind, msg in messages:
                try:
                    if kind == "status":
                        self._emit_status(msg)
                    elif kind == "warn":
                        self._emit_warning(msg)
                    else:
                        self._vprint(f"{self.log_prefix}{msg}", force=True)
                except Exception:
                    pass
        except Exception:
            pass
