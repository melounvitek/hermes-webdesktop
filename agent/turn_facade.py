"""``AIAgent.run_conversation`` / ``chat`` façade.

Turn admission around ``conversation_loop.run_conversation``: durable cross-process session turn lease +
refresher thread, turn-liveness watchdog, relay/accounting/portal scopes, and balanced start/finish marks.
Extracted from ``run_agent.py``; every method resolves through ``AIAgent``'s MRO unchanged.
"""
import logging
import os
import threading
import uuid
from typing import Any, Dict, List, Optional

from agent.lazy_forward import forward as _forward
from tools.interrupt import set_interrupt as _set_interrupt

# Same logger name as the origin module so log records / caplog filters are unchanged.
logger = logging.getLogger("run_agent")


class TurnFacadeMixin:
    """run_conversation()/chat() (see module docstring)."""

    def run_conversation(
        self,
        user_message: Any,
        system_message: str = None,
        conversation_history: List[Dict[str, Any]] = None,
        task_id: str = None,
        stream_callback: Optional[callable] = None,
        persist_user_message: Optional[Any] = None,
        persist_user_timestamp: Optional[float] = None,
        persist_user_display_kind: Optional[str] = None,
        persist_user_display_metadata: Optional[Dict[str, Any]] = None,
        persist_user_platform_id: Optional[str] = None,
        moa_config: Optional[dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Forwarder — see ``agent.conversation_loop.run_conversation``."""
        # A review shares this session_id for cache parity: fence review startup or interrupt an admitted
        # request and await its exit before opening live-turn instrumentation (#84423).
        from agent.background_review import cancel_background_review_for_live_turn

        cancel_background_review_for_live_turn(self)

        # Turn liveness for the deferred-review idle queue; start-mark is the first statement of the try
        # so the finally's note_turn_finished balances every exit.
        from agent.review_idle_queue import QUEUE as _review_queue

        from agent.aux_accounting import (
            reset_accounting_context,
            set_accounting_context,
        )
        from agent import relay_runtime
        from agent.conversation_loop import run_conversation
        from agent.portal_tags import (
            reset_affinity_scope,
            reset_conversation_context,
            set_affinity_scope,
            set_conversation_context,
        )
        from agent.prompt_cache_scope import declared_conversation_scope_safe
        from hermes_cli.observability.relay_shared_metrics import (
            finish_task_run,
            start_task_run,
        )
        from agent.subagent_lifecycle import bind_subagent_parent
        effective_task_id = task_id or str(uuid.uuid4())
        session_id = str(getattr(self, "session_id", None) or "")
        task_context = {
            "session_id": session_id,
            "task_id": effective_task_id,
            "platform": getattr(self, "platform", None) or "",
        }
        relay_turn_id = (
            f"{session_id or 'session'}:{effective_task_id}:{uuid.uuid4().hex[:8]}"
        )
        self._relay_pending_turn_id = relay_turn_id
        relay_parent_session_id = (
            str(getattr(self, "_parent_session_id", None) or "")
            if task_context["platform"] == "subagent"
            else ""
        )
        relay_lease = None
        relay_turn = None
        durable_turn_lease = None
        durable_turn_lease_stop = None
        durable_turn_lease_thread = None
        durable_turn_liveness_thread = None
        durable_turn_lease_activity_lock = threading.Lock()
        durable_turn_lease_turn_active = False
        durable_turn_lease_interrupt_message = None
        token = None
        # Initialized alongside `token`: early returns leave the try before set_affinity_scope() and the
        # finally reads this unconditionally (PR #97158).
        affinity_token = None
        acct_token = None
        task_started = False
        task_finished = False
        relay_outcome = "failed"

        def _stop_durable_turn_lease_refresher() -> None:
            nonlocal durable_turn_lease_turn_active
            with durable_turn_lease_activity_lock:
                durable_turn_lease_turn_active = False
                if durable_turn_lease_stop is not None:
                    durable_turn_lease_stop.set()

        def _clear_durable_turn_lease_interrupt() -> None:
            """Clear only the interrupt admitted by this turn's refresher."""
            message = durable_turn_lease_interrupt_message
            if not message:
                return

            def _clear_if_owned() -> None:
                if getattr(self, "_interrupt_message", None) != message:
                    return
                self._interrupt_requested = False
                self._interrupt_message = None
                getattr(self, "_hard_interrupt_requested", threading.Event()).clear()
                self._interrupt_thread_signal_pending = False
                if self._execution_thread_id is not None:
                    _set_interrupt(False, self._execution_thread_id)

            redirect_lock = getattr(self, "_pending_redirect_lock", None)
            if redirect_lock is None:
                _clear_if_owned()
            else:
                with redirect_lock:
                    _clear_if_owned()

        try:
            _review_queue.note_turn_started()
            # Durable cross-process lease over load -> run -> flush (Desktop, CLI resume, gateway, background
            # delivery sharing state.db, #84234).
            _turn_db = getattr(self, "_session_db", None)
            _durable_session_exists = False
            if _turn_db is not None and session_id:
                try:
                    _durable_session_exists = _turn_db.get_session(session_id) is not None
                except Exception:
                    # A locked / non-WAL read is not proof the row is absent; treating probe failure as
                    # "fresh" ran fail-open at the exact contention point (#84234). Acquire, or fail closed.
                    logger.warning(
                        "Could not check durable session before turn lease; "
                        "will acquire rather than run without serialization",
                        exc_info=True,
                    )
                    _durable_session_exists = True
            if (
                _turn_db is not None
                and session_id
                and not getattr(self, "_persist_disabled", False)
                # A fresh session id has no durable transcript to race over, and callers may supply an in-
                # memory seed before the row exists — reloading would erase it.
                and _durable_session_exists
                # Check the concrete type: MagicMock-style shims accept any attribute without the protocol.
                and callable(
                    getattr(type(_turn_db), "acquire_session_turn_lease", None)
                )
            ):
                # Row proven to exist — suppress the redundant create attempt.
                self._session_db_created = True
                _durable_holder = (
                    f"pid={os.getpid()}:turn={relay_turn_id}:platform="
                    f"{task_context['platform'] or 'unknown'}"
                )
                _lease_ttl = 300.0
                _lease_waited = False

                def _on_session_turn_lease_wait(elapsed: float) -> None:
                    nonlocal _lease_waited
                    _lease_waited = True
                    if elapsed < 1.0:
                        self._emit_status(
                            "⏳ Another Hermes process is using this session; "
                            "waiting for it to finish before starting your turn..."
                        )
                    else:
                        self._emit_status(
                            "⏳ Still waiting for the other Hermes process on "
                            f"this session ({int(elapsed)}s)..."
                        )

                if not _turn_db.acquire_session_turn_lease(
                    session_id,
                    _durable_holder,
                    ttl_seconds=_lease_ttl,
                    wait_seconds=1800.0,
                    on_wait=_on_session_turn_lease_wait,
                    should_abort=lambda: getattr(self, "_interrupt_requested", False),
                ):
                    if getattr(self, "_interrupt_requested", False):
                        logger.info(
                            "session turn lease wait aborted by interrupt: %s",
                            session_id,
                        )
                        relay_outcome = "cancelled"
                        interrupt_msg = (
                            "Stopped waiting for another Hermes process on "
                            "this session. Your message was not processed."
                        )
                        interrupt_result = {
                            "final_response": interrupt_msg,
                            "messages": list(conversation_history or []),
                            "api_calls": 0,
                            "completed": False,
                            "interrupted": True,
                        }
                        interrupt_message = getattr(
                            self, "_interrupt_message", None
                        )
                        if interrupt_message:
                            interrupt_result["interrupt_message"] = (
                                interrupt_message
                            )
                        # The finalizer never runs on this early return; clear so a cached agent doesn't fail-
                        # close the next turn.
                        try:
                            self.clear_interrupt()
                        except Exception:
                            self._interrupt_requested = False
                            self._interrupt_message = None
                        return interrupt_result
                    # Fail closed like gateway TurnLeaseTimeoutError: surface a resend notice, not a bare
                    # TimeoutError.
                    timeout_msg = (
                        "⏳ Another Hermes process kept this session busy too "
                        "long. Your message was not processed - wait for the "
                        "other process to finish, then send it again."
                    )
                    logger.error(
                        "session turn lease wait timed out for %s",
                        session_id,
                    )
                    try:
                        self._emit_warning(timeout_msg)
                    except Exception:
                        logger.debug(
                            "Failed to emit session turn lease timeout warning",
                            exc_info=True,
                        )
                    relay_outcome = "timed_out"
                    return {
                        "final_response": timeout_msg,
                        "messages": list(conversation_history or []),
                        "api_calls": 0,
                        "completed": False,
                        "failed": True,
                        "error": f"session_turn_lease_timeout:{session_id}",
                    }

                # Assign only after admission so the finally cannot release a holder that never owned the row;
                # persist paths read the agent attr so a late flush is fenced in the same SQLite transaction.
                durable_turn_lease = _durable_holder
                self._active_session_turn_lease_holder = _durable_holder
                self._active_session_turn_lease_ttl_seconds = _lease_ttl
                if _lease_waited:
                    self._emit_status(
                        "Session is free; loading the latest transcript..."
                    )

                # The holder may have compressed/rotated the session while we waited: reload only AFTER
                # admission, and skip when acquisition was immediate (avoids a needless prompt-cache miss).
                if _lease_waited:
                    latest_session_id = _turn_db.resolve_resume_session_id(session_id)
                    if latest_session_id:
                        self.session_id = latest_session_id
                        task_context["session_id"] = latest_session_id
                    conversation_history = _turn_db.get_messages_as_conversation(
                        self.session_id,
                        repair_alternation=True,
                        include_row_ids=True,
                    )

                # Long turns outlive a fixed TTL: refresh in a daemon thread; holder-qualified UPDATE/DELETE
                # fence a late refresher from a successor lease.
                durable_turn_lease_stop = threading.Event()
                _lease_refresh_interval = float(
                    getattr(self, "_session_turn_lease_refresh_interval", 60.0)
                )

                # ── Turn liveness watchdog (#95548) ──
                # Lease renewal is NOT evidence of progress; a silently stalled turn would renew forever.
                # Policy
                # lives in agent/turn_liveness.py; this block only wires config + commit/deactivate callbacks.
                try:
                    from hermes_cli.config import (
                        load_config_readonly as _liveness_load_config,
                    )
                    _liveness_config = _liveness_load_config() or {}
                except Exception:
                    _liveness_config = {}
                from agent import turn_liveness

                _liveness_timeout, _liveness_poll = (
                    turn_liveness.resolve_turn_liveness_settings(_liveness_config)
                )

                def _interrupt_turn(message: str) -> None:
                    # Lease-loss interrupts fire UNCONDITIONALLY (no generation claim): a lost lease means
                    # this process no longer owns the session. Only the watchdog's stalls can be spuriously
                    # stale.
                    nonlocal durable_turn_lease_interrupt_message
                    with durable_turn_lease_activity_lock:
                        if (
                            durable_turn_lease_stop.is_set()
                            or not durable_turn_lease_turn_active
                        ):
                            return
                        durable_turn_lease_interrupt_message = message
                        try:
                            self.interrupt(message, hard_cancel=True)
                        except Exception:
                            self._interrupt_requested = True
                            self._interrupt_message = message

                def _commit_turn_liveness_abort(
                    snapshot: "turn_liveness.ActivitySnapshot",
                    message: str,
                ) -> bool:
                    """Commit point for the watchdog's stall observation.

                    Revalidates the observed ``(generation, timestamp)`` under the SAME lock
                    ``_touch_activity`` uses, so a turn
                    that resumed while the stall was logged is never hard-cancelled. The revalidated
                    generation is carried into
                    ``interrupt`` as ``require_generation``, which consumes it with the first publication in
                    ONE critical section.
                    If ``interrupt`` raises, the abort declines FAIL-CLOSED. Returns False when stale or
                    already winding down.
                    """
                    nonlocal durable_turn_lease_interrupt_message
                    with self._liveness_activity_lock():
                        current_generation = getattr(
                            self, "_turn_liveness_activity_generation", 0
                        )
                        if (
                            current_generation,
                            getattr(self, "_last_activity_ts", None),
                        ) != (snapshot.generation, snapshot.activity_ts):
                            return False
                    with durable_turn_lease_activity_lock:
                        if (
                            durable_turn_lease_stop.is_set()
                            or not durable_turn_lease_turn_active
                        ):
                            return False
                    try:
                        published = self.interrupt(
                            message,
                            hard_cancel=True,
                            require_generation=current_generation,
                        )
                    except Exception:
                        # Round-4 (#95663): fail closed — an exceptional path must not turn an unvalidated
                        # claim into unconditional abort authority.
                        logger.debug(
                            "Turn liveness abort interrupt raised; "
                            "declining the abort",
                            exc_info=True,
                        )
                        published = False
                    if published is False:
                        # Claim went stale between revalidation and the hammer: real progress landed, abandon
                        # the abort.
                        return False
                    with durable_turn_lease_activity_lock:
                        durable_turn_lease_interrupt_message = message
                    return True

                def _deactivate_turn_after_liveness_abort() -> None:
                    """Stop lease renewal after a committed liveness abort.

                    A wedge the hard interrupt cannot unwind must not keep the lease alive forever; TTL expiry
                    lets stale-turn cleanup reclaim the row.
                    """
                    nonlocal durable_turn_lease_turn_active
                    with durable_turn_lease_activity_lock:
                        durable_turn_lease_stop.set()
                        durable_turn_lease_turn_active = False

                def _turn_is_active() -> bool:
                    with durable_turn_lease_activity_lock:
                        return durable_turn_lease_turn_active

                def _refresh_durable_turn_lease() -> None:
                    while not durable_turn_lease_stop.wait(_lease_refresh_interval):
                        try:
                            if not _turn_db.refresh_session_turn_lease(
                                getattr(self, "session_id", None) or session_id,
                                durable_turn_lease,
                                ttl_seconds=_lease_ttl,
                            ):
                                # finally sets stop then releases; a late holder-fenced miss must not hard-
                                # interrupt the next turn.
                                if durable_turn_lease_stop.is_set():
                                    return
                                logger.error(
                                    "Lost session turn lease while turn is active: %s",
                                    getattr(self, "session_id", None) or session_id,
                                )
                                _interrupt_turn(
                                    "Session turn lease lost; stopping to protect "
                                    "the transcript."
                                )
                                return
                        except Exception:
                            if durable_turn_lease_stop.is_set():
                                return
                            logger.warning(
                                "Failed to refresh session turn lease: %s",
                                getattr(self, "session_id", None) or session_id,
                                exc_info=True,
                            )
                            _interrupt_turn(
                                "Session turn lease could not be refreshed; "
                                "stopping to protect the transcript."
                            )
                            return

                durable_turn_lease_thread = threading.Thread(
                    target=_refresh_durable_turn_lease,
                    name="session-turn-lease-refresh",
                    daemon=True,
                )
                if _liveness_timeout is not None:
                    durable_turn_liveness_thread = (
                        turn_liveness.TurnLivenessWatchdog(
                            self,
                            session_id=getattr(self, "session_id", None) or session_id,
                            timeout_s=_liveness_timeout,
                            poll_s=_liveness_poll,
                            stop_event=durable_turn_lease_stop,
                            activity_lock=self._liveness_activity_lock(),
                            is_turn_active=_turn_is_active,
                            commit_abort=_commit_turn_liveness_abort,
                            deactivate_turn=_deactivate_turn_after_liveness_abort,
                        ).make_thread()
                    )

            relay_lease = relay_runtime.SESSION_COORDINATOR.acquire_conversation(
                profile_key=relay_runtime.current_profile_key(),
                session_id=task_context["session_id"],
                platform=task_context["platform"],
                parent_session_id=relay_parent_session_id,
                model=str(getattr(self, "model", None) or ""),
            )
            relay_turn = relay_runtime.SESSION_COORDINATOR.begin_turn(
                relay_lease,
                turn_id=relay_turn_id,
                task_id=effective_task_id,
            )
            # Keep existing tests and external relay-runtime shims that return
            # a minimal turn object compatible with the new opt-out flag.
            if getattr(relay_turn, "relay_enabled", True):
                start_task_run(
                    **task_context,
                    parent_session_id=getattr(self, "_parent_session_id", None) or "",
                )
                task_started = True
            # Publish the conversation id for ambient Nous Portal tagging: every LLM call in this turn (loop,
            # compression, vision, MoA, review forks) inherits `conversation=<root>`.
            token = set_conversation_context(self._conversation_root_id())
            # Routing/affinity scope the HOST declared; providers fall back to the id above when unset
            # (#96811).
            affinity_token = set_affinity_scope(
                declared_conversation_scope_safe(self)
            )
            # Publish session accounting handles so auxiliary calls record usage into session_model_usage
            # (#23270).
            acct_token = set_accounting_context(
                getattr(self, "_session_db", None),
                getattr(self, "session_id", None),
            )
            from agent.auxiliary_client import scoped_runtime_main

            # Keep the ContextVar scope local (tokens on the agent may be observed from another thread).
            with bind_subagent_parent(self), scoped_runtime_main({}):
                try:
                    if durable_turn_lease_thread is not None:
                        with durable_turn_lease_activity_lock:
                            durable_turn_lease_turn_active = True
                        # Stamp the activity clock at turn entry (#95663): `_last_activity_ts` persists across
                        # turns, so without this the watchdog would measure idle from the PREVIOUS turn and
                        # abort a fresh one.
                        self._touch_activity("starting new turn")
                        durable_turn_lease_thread.start()
                        if durable_turn_liveness_thread is not None:
                            durable_turn_liveness_thread.start()
                    result = run_conversation(
                        self,
                        user_message,
                        system_message,
                        conversation_history,
                        effective_task_id,
                        stream_callback,
                        persist_user_message,
                        persist_user_timestamp=persist_user_timestamp,
                        persist_user_display_kind=persist_user_display_kind,
                        persist_user_display_metadata=persist_user_display_metadata,
                        persist_user_platform_id=persist_user_platform_id,
                        moa_config=moa_config,
                    )
                finally:
                    # Post-loop relay/task finalization must not receive a late refresh interrupt.
                    _stop_durable_turn_lease_refresher()
                    # Interrupt clear is deferred until after thread join (outer finally) so a refresher
                    # firing between stop and join cannot leave an interrupt behind.
            terminal = result if isinstance(result, dict) else {}
            if terminal.get("interrupted") is True:
                relay_outcome = "cancelled"
            elif terminal.get("failed") is True:
                relay_outcome = "failed"
            else:
                relay_outcome = "success"
            relay_runtime.SESSION_COORDINATOR.finish_logical_calls(
                relay_turn,
                outcome=relay_outcome,
            )
            if task_started:
                task_finished = True
                finish_task_run(**task_context, result=result)
            return result
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, InterruptedError)) or (
                type(exc).__name__ == "CancelledError"
            ):
                relay_outcome = "cancelled"
            elif isinstance(exc, TimeoutError):
                relay_outcome = "timed_out"
            if relay_turn is not None:
                relay_runtime.SESSION_COORDINATOR.finish_logical_calls(
                    relay_turn,
                    outcome=relay_outcome,
                )
            if task_started and not task_finished:
                task_finished = True
                finish_task_run(**task_context, error=exc)
            raise
        finally:
            try:
                if relay_turn is not None:
                    relay_runtime.SESSION_COORDINATOR.end_turn(
                        relay_turn,
                        outcome=relay_outcome,
                    )
            finally:
                try:
                    if relay_lease is not None:
                        relay_runtime.SESSION_COORDINATOR.release_conversation(
                            relay_lease
                        )
                finally:
                    _stop_durable_turn_lease_refresher()
                    for _durable_thread in (
                        durable_turn_lease_thread,
                        durable_turn_liveness_thread,
                    ):
                        if (
                            _durable_thread is not None
                            and _durable_thread.is_alive()
                        ):
                            _durable_thread.join(timeout=1.0)
                    # Clear any refresher interrupt fired between stop and join; must run AFTER join.
                    _clear_durable_turn_lease_interrupt()
                    if durable_turn_lease is not None:
                        try:
                            _turn_db.release_session_turn_lease(
                                session_id, durable_turn_lease
                            )
                        except Exception:
                            logger.error(
                                "Failed to release session turn lease: %s",
                                session_id,
                                exc_info=True,
                            )
                        if (
                            getattr(self, "_active_session_turn_lease_holder", None)
                            == durable_turn_lease
                        ):
                            self._active_session_turn_lease_holder = None
                            self._active_session_turn_lease_ttl_seconds = None
                    # Always clear mid-turn labels when the turn exits — including
                    # interrupted early returns that skip finalize_turn. Keep ts.
                    try:
                        self._reset_activity_labels_after_turn()
                    except Exception:
                        pass
                    if getattr(self, "_relay_pending_turn_id", None) == relay_turn_id:
                        self._relay_pending_turn_id = None
                    if acct_token is not None:
                        reset_accounting_context(acct_token)
                    if token is not None:
                        reset_conversation_context(token)
                    if affinity_token is not None:
                        reset_affinity_scope(affinity_token)
                    # Balance note_turn_started on every exit so the idle queue's live-turn count cannot leak.
                    try:
                        _review_queue.note_turn_finished()
                    except Exception:
                        pass

    def chat(self, message: str, stream_callback: Optional[callable] = None) -> str:
        """Simple chat interface that returns just the final response string.

        ``stream_callback`` is invoked with each text delta during streaming.
        """
        result = self.run_conversation(message, stream_callback=stream_callback)
        return result["final_response"]

    _run_codex_app_server_turn = _forward("agent.codex_runtime", "run_codex_app_server_turn")
