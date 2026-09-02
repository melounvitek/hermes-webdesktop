"""Interrupt / steer / redirect control surface for ``AIAgent``.

Soft/hard interrupt requests, tool-thread interrupt propagation, pending steer/redirect queues.
Extracted from ``run_agent.py``; every method resolves through ``AIAgent``'s MRO unchanged.
"""
import logging
import threading
from typing import Optional

from agent.interrupt_compat import request_hard_interrupt
from tools.interrupt import set_interrupt as _set_interrupt

# Same logger name as the origin module so log records / caplog filters are unchanged.
logger = logging.getLogger("run_agent")


class InterruptControlMixin:
    """interrupt()/hard_interrupt()/clear_interrupt()/steer()/redirect() (see module docstring)."""

    def interrupt(
        self,
        message: Optional[str] = None,
        *,
        hard_cancel: bool = False,
        tool_reason: Optional[str] = None,
        require_generation: Optional[int] = None,
    ) -> bool:
        """Request the agent to interrupt its current tool-calling loop (call from another thread).

        ``message``: new message to include in the response context. ``hard_cancel``: explicit stop;
        compression may honor it even while ordinary interrupts are masked. ``tool_reason``: trusted fixed
        category safe for tool output. ``require_generation``: activity-generation claim — the interrupt is
        published only if the turn's generation still matches at the final mutation edge (claim reserved under
        the activity lock, consumed together with the first observable publication); returns False if the turn
        resumed meanwhile.
        """
        if require_generation is not None:
            # RESERVE the abort's generation claim under the SAME lock `_touch_activity` stamps with. Real
            # progress invalidates it; it is CONSUMED at the final mutation edge, so a resumed turn abandons
            # the abort.
            with self._liveness_activity_lock():
                if (
                    getattr(self, "_turn_liveness_activity_generation", 0)
                    != require_generation
                ):
                    return False
                self._turn_liveness_abort_claim = require_generation

        # A hard stop and redirect share one lock so /stop cannot race with an
        # accepted correction and accidentally turn itself into a retry.
        def _wait_for_compression_commit() -> None:
            # Pre-claim half of hard-cancel admission (#99758 P1): wait out a commit that already crossed its
            # boundary but mutate NOTHING — cancelling a pending fence is irreversible and must wait until the
            # generation claim survived the final mutation edge (_cancel_pending_compression_commit).
            fence = vars(self).get("_active_compression_commit_fence")
            if fence is None:
                return
            if not getattr(fence, "commit_in_flight", False):
                # No commit in flight — cancel_before_commit here WOULD cancel the pending commit; leave it to
                # the
                # destructive half.
                return
            cancel_before_commit = getattr(
                type(fence), "cancel_before_commit", None
            )
            if callable(cancel_before_commit):
                try:
                    # A commit holds the fence lock through finish_commit: this blocks until it finishes and
                    # returns
                    # False WITHOUT setting _cancelled.
                    cancel_before_commit(fence)
                except Exception:
                    logger.debug(
                        "Compression hard-cancel fence wait failed",
                        exc_info=True,
                    )

        def _cancel_pending_compression_commit() -> None:
            # Destructive half of hard-cancel admission (#99758 P1): runs only AFTER the claim survived, so a
            # declined abort never leaves the fence cancelled. A commit that started meanwhile owns the fence
            # and completes on its own; only a still-pending commit is cancelled here.
            fence = vars(self).get("_active_compression_commit_fence")
            if fence is None:
                return
            if getattr(fence, "commit_in_flight", False):
                return
            cancel_before_commit = getattr(
                type(fence), "cancel_before_commit", None
            )
            if callable(cancel_before_commit):
                try:
                    # Marks the fence cancelled (or waits out a just-started commit) without touching the
                    # hard-stop
                    # Event, which was published at the claim edge.
                    cancel_before_commit(fence)
                except Exception:
                    logger.debug(
                        "Compression hard-cancel fence admission failed",
                        exc_info=True,
                    )

        def _publish_interrupt_state() -> None:
            self._interrupt_requested = True
            self._interrupt_message = message
            self._tool_interrupt_reason = tool_interrupt_reason
            if hard_cancel:
                _hard_event = getattr(
                    self, "_hard_interrupt_requested", None
                )
                if _hard_event is not None:
                    _hard_event.set()

        def _consume_claim_and_publish_first_state() -> bool:
            # Final mutation edge: claim consumption and the FIRST observable interrupt publication are ONE
            # activity-lock critical section, so either the claim survives and commits before any later
            # activity stamp, or the stamp landed first and the abort declines without publishing.
            if require_generation is None:
                # No claim to race: publish WITHOUT the liveness lock. Bare AIAgent stand-ins in other suites
                # lack
                # the liveness seam and would AttributeError.
                _publish_interrupt_state()
                return True
            with self._liveness_activity_lock():
                if (
                    getattr(self, "_turn_liveness_abort_claim", None)
                    != require_generation
                ):
                    return False
                self._turn_liveness_abort_claim = None
                _publish_interrupt_state()
            return True

        # Tool cancellation attribution stays separate from _interrupt_message, which may carry the user's
        # full next message.
        tool_interrupt_reason = (
            (tool_reason or "explicit stop requested")
            if hard_cancel
            else ("user sent a new message" if message else "user interrupt")
        )

        _redirect_lock = getattr(self, "_pending_redirect_lock", None)
        if _redirect_lock is not None:
            with _redirect_lock:
                # The blocking in-flight-commit wait runs BEFORE the atomic claim edge (redirect lock still
                # held);
                # the destructive pending-commit cancel runs AFTER the claim survives (#99758 P1).
                if hard_cancel:
                    _wait_for_compression_commit()
                if not _consume_claim_and_publish_first_state():
                    return False
                if hard_cancel:
                    _cancel_pending_compression_commit()
                self._pending_redirect = None
        else:
            if hard_cancel:
                _wait_for_compression_commit()
            if not _consume_claim_and_publish_first_state():
                return False
            if hard_cancel:
                _cancel_pending_compression_commit()
            self._pending_redirect = None

        # Codex app-server owns its model/tool loop and watches a private
        # interrupt event rather than Hermes' per-thread flag.
        if getattr(self, "api_mode", None) == "codex_app_server":
            _codex_session = getattr(self, "_codex_session", None)
            _request_interrupt = getattr(_codex_session, "request_interrupt", None)
            if callable(_request_interrupt):
                try:
                    _request_interrupt()
                except Exception:
                    logger.debug(
                        "Failed to interrupt Codex app-server turn",
                        exc_info=True,
                    )

        # Cron turns request on the conversation thread (no nested interrupt-worker deadlock); their client
        # is registered here so this cross-thread interrupt can still shut the sockets.
        _abort_active_request = getattr(self, "_active_request_abort", None)
        if callable(_abort_active_request):
            try:
                _abort_active_request("interrupt_abort")
            except Exception:
                logger.debug("Failed to abort active inline request", exc_info=True)
        # Scope the tool interrupt to this agent's execution thread so other in-process agents are unaffected.
        if self._execution_thread_id is not None:
            _set_interrupt(
                True,
                self._execution_thread_id,
                reason=tool_interrupt_reason,
            )
            self._interrupt_thread_signal_pending = False
        else:
            # Interrupt arrived before run_conversation bound the execution thread: defer the tool-level
            # signal instead of targeting the caller thread.
            self._interrupt_thread_signal_pending = True
        # Fan out to concurrent-tool worker tids: is_interrupted() inside a tool only sees its own tid, so
        # without this a hung concurrent tool runs to its own timeout. getattr covers __init__-less stubs.
        _tracker = getattr(self, "_tool_worker_threads", None)
        _tracker_lock = getattr(self, "_tool_worker_threads_lock", None)
        if _tracker is not None and _tracker_lock is not None:
            with _tracker_lock:
                _worker_tids = list(_tracker)
            for _wtid in _worker_tids:
                try:
                    _set_interrupt(True, _wtid, reason=tool_interrupt_reason)
                except Exception:
                    pass
        # Propagate interrupt to any running child agents (subagent delegation)
        with self._active_children_lock:
            children_copy = list(self._active_children)
        for child in children_copy:
            try:
                if hard_cancel:
                    request_hard_interrupt(
                        child,
                        message,
                        tool_reason=tool_interrupt_reason,
                    )
                else:
                    child.interrupt(message)
            except Exception as e:
                logger.debug("Failed to propagate interrupt to child agent: %s", e)
        if not self.quiet_mode:
            print("\n⚡ Interrupt requested" + (f": '{message[:40]}...'" if message and len(message) > 40 else f": '{message}'" if message else ""))
        return True

    def hard_interrupt(
        self,
        message: Optional[str] = None,
        *,
        tool_reason: Optional[str] = None,
    ) -> None:
        """Request an explicit stop while preserving the ``interrupt()`` ABI.

        Frontends feature-detect this and fall back to legacy ``interrupt()`` for third-party agents.
        """
        # Bypass dynamic dispatch: legacy subclasses may override interrupt(message=None) without hard_cancel.
        InterruptControlMixin.interrupt(
            self,
            message,
            hard_cancel=True,
            tool_reason=tool_reason,
        )

    def clear_interrupt(self, *, preserve_redirect: bool = False) -> bool:
        """Clear the interrupt request and per-thread tool signal.

        ``preserve_redirect`` is only for the conversation loop rebuilding the same logical turn after
        cancelling a model request; public hard-stop paths clear everything.
        """
        _redirect_lock = getattr(self, "_pending_redirect_lock", None)
        if _redirect_lock is not None:
            with _redirect_lock:
                if preserve_redirect and not self._pending_redirect:
                    return False
                self._interrupt_requested = False
                self._interrupt_message = None
                self._tool_interrupt_reason = None
                getattr(self, "_hard_interrupt_requested", threading.Event()).clear()
                if not preserve_redirect:
                    self._pending_redirect = None
        else:
            if preserve_redirect and not getattr(self, "_pending_redirect", None):
                return False
            self._interrupt_requested = False
            self._interrupt_message = None
            self._tool_interrupt_reason = None
            getattr(self, "_hard_interrupt_requested", threading.Event()).clear()
            if not preserve_redirect:
                self._pending_redirect = None
        self._interrupt_thread_signal_pending = False
        if self._execution_thread_id is not None:
            _set_interrupt(False, self._execution_thread_id)
        # Also clear worker-thread bits so no stale interrupt survives a turn boundary onto a recycled tid.
        # getattr covers __init__-less test stubs.
        _tracker = getattr(self, "_tool_worker_threads", None)
        _tracker_lock = getattr(self, "_tool_worker_threads_lock", None)
        if _tracker is not None and _tracker_lock is not None:
            with _tracker_lock:
                _worker_tids = list(_tracker)
            for _wtid in _worker_tids:
                try:
                    _set_interrupt(False, _wtid)
                except Exception:
                    pass
        # A hard interrupt supersedes any pending /steer — its target iteration will no longer happen.
        _steer_lock = getattr(self, "_pending_steer_lock", None)
        if _steer_lock is not None:
            with _steer_lock:
                self._pending_steer = None
        return True

    def steer(self, text: str) -> bool:
        """Inject user text into the next tool result without interrupting the current tool.

        The text is appended to the LAST tool result once the batch finishes, so the model sees it on its next
        iteration. Thread-safe; multiple calls concatenate with newlines. Returns False for empty text.
        """
        if not text or not text.strip():
            return False
        cleaned = text.strip()
        _lock = getattr(self, "_pending_steer_lock", None)
        if _lock is None:
            # __init__-less test stubs: fall back to a direct attribute set.
            existing = getattr(self, "_pending_steer", None)
            self._pending_steer = (existing + "\n" + cleaned) if existing else cleaned
            return True
        with _lock:
            if self._pending_steer:
                self._pending_steer = self._pending_steer + "\n" + cleaned
            else:
                self._pending_steer = cleaned
        return True

    def redirect(self, text: str) -> bool:
        """Redirect the active turn without converting it into a new task.

        During a model request this cancels only that request: completed messages/tool results are kept, the
        displayed partial reasoning becomes assistant context, the correction is appended as a real user
        message, and the loop retries. During tool execution it degrades to ``steer()``; Codex app-server uses
        native ``turn/steer``. Returns False when there is no live turn or the text is empty.
        """
        if not text or not text.strip():
            return False
        cleaned = text.strip()

        # Codex owns its internal reasoning/tool loop, so use its first-class
        # active-turn steering protocol rather than interrupting the subprocess.
        if getattr(self, "api_mode", None) == "codex_app_server":
            _codex_session = getattr(self, "_codex_session", None)
            _native_steer = getattr(_codex_session, "request_steer", None)
            if callable(_native_steer):
                _redirect_lock = getattr(self, "_pending_redirect_lock", None)
                if _redirect_lock is not None:
                    with _redirect_lock:
                        if self._interrupt_requested:
                            return False
                elif self._interrupt_requested:
                    return False
                try:
                    return bool(_native_steer(cleaned))
                except Exception:
                    logger.debug("Codex app-server turn/steer failed", exc_info=True)
                    return False

        # Never kill a tool to deliver guidance; the steer drain puts it on the final tool result.
        if getattr(self, "_executing_tools", False):
            return self.steer(cleaned)

        _model_active = getattr(self, "_model_request_active", None)
        _redirect_lock = getattr(self, "_pending_redirect_lock", None)
        if _redirect_lock is None:
            if _model_active is None or not _model_active.is_set():
                return False
            existing = getattr(self, "_pending_redirect", None)
            if self._interrupt_requested and not existing:
                return False
            self._pending_redirect = (
                f"{existing}\n\n[Additional user correction]\n{cleaned}"
                if existing
                else cleaned
            )
            self._interrupt_requested = True
            self._interrupt_message = None
        else:
            with _redirect_lock:
                if _model_active is None or not _model_active.is_set():
                    # The response completed before we acquired the state lock.
                    # Reject so the surface queues a new turn.
                    return False
                if self._interrupt_requested and not self._pending_redirect:
                    return False
                if self._pending_redirect:
                    self._pending_redirect = (
                        f"{self._pending_redirect}\n\n"
                        f"[Additional user correction]\n{cleaned}"
                    )
                else:
                    self._pending_redirect = cleaned
                self._interrupt_requested = True
                self._interrupt_message = None

        # Interrupt only the model request. Do not fan out to tool workers or
        # child agents as interrupt() does.
        _execution_thread_id = getattr(self, "_execution_thread_id", None)
        if _execution_thread_id is not None:
            _set_interrupt(True, _execution_thread_id)
            self._interrupt_thread_signal_pending = False
        else:
            self._interrupt_thread_signal_pending = True
        _abort_active_request = getattr(self, "_active_request_abort", None)
        if callable(_abort_active_request):
            try:
                _abort_active_request("redirect_abort")
            except Exception:
                logger.debug("Failed to abort request for redirect", exc_info=True)
        return True

    def _has_pending_redirect(self) -> bool:
        """Return whether an active-turn redirect is waiting to be applied."""
        _redirect_lock = getattr(self, "_pending_redirect_lock", None)
        if _redirect_lock is None:
            return bool(getattr(self, "_pending_redirect", None))
        with _redirect_lock:
            return bool(self._pending_redirect)

    def _drain_pending_redirect(self) -> Optional[str]:
        """Return and clear pending active-turn correction text."""
        _redirect_lock = getattr(self, "_pending_redirect_lock", None)
        if _redirect_lock is None:
            text = getattr(self, "_pending_redirect", None)
            self._pending_redirect = None
            return text
        with _redirect_lock:
            text = self._pending_redirect
            self._pending_redirect = None
        return text

    def _drain_pending_steer(self) -> Optional[str]:
        """Return the pending steer text (if any) and clear the slot; None when nothing is pending."""
        _lock = getattr(self, "_pending_steer_lock", None)
        if _lock is None:
            text = getattr(self, "_pending_steer", None)
            self._pending_steer = None
            return text
        with _lock:
            text = self._pending_steer
            self._pending_steer = None
        return text
