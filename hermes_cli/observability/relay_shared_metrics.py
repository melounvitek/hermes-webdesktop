"""Direct NeMo Relay integration for Hermes shared client metrics."""

from __future__ import annotations

import atexit
import contextlib
import contextvars
import logging
import threading
from collections import deque
from dataclasses import dataclass, field
from time import monotonic_ns
from typing import Any, Callable

from agent import relay_runtime
from hermes_cli import __version__

from .shared_metrics import SharedMetricsStore
from .shared_metrics_contract import (
    CLIENT_ACTIVE_MARK,
    MODEL_CALL_PROFILE_MODEL,
    MODEL_CALL_SCOPE,
    SCHEMA_KEY,
    SCHEMA_VERSION,
    SKILL_LIFECYCLE_MARK,
    SKILL_LOAD_MARK,
    SUBSCRIBER_NAME,
    TASK_SCOPE,
    TOOL_APPROVAL_MARK,
    TOOL_CALL_SCOPE,
    model_call_fields,
    skill_lifecycle_fields,
    skill_load_fields,
    task_start_fields,
    task_terminal_fields,
    task_terminal_state,
    tool_approval_outcome,
    tool_category,
    tool_terminal_fields,
)
from .shared_metrics_subscriber import SharedMetricsSubscriber

logger = logging.getLogger(__name__)

_RUNTIME_FAILED = object()
_RUNTIMES: dict[str, _Runtime | object] = {}
_RUNTIME_LOCK = threading.RLock()

_ABORTED = {"failed": True, "turn_exit_reason": "system_aborted"}


def _text(event: dict[str, Any], key: str) -> str:
    return str(event.get(key) or "")


def _session_pair(event: dict[str, Any], key: str) -> tuple[str, str] | None:
    """(session_id, event[key]) when both are non-empty."""
    session_id, value = _text(event, "session_id"), _text(event, key)
    return (session_id, value) if session_id and value else None


def _retry_ordinal(event: dict[str, Any]) -> int | None:
    value = event.get("retry_count")
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def _sole(items: Any) -> Any:
    """The single distinct element of ``items`` (identity-deduplicated), else None."""
    unique = {id(item): item for item in items}
    return next(iter(unique.values())) if len(unique) == 1 else None


@dataclass
class _ModelCall:
    handle: Any
    task_id: str
    fields: dict[str, str]
    retry_ordinal: int | None = None


@dataclass
class _ToolCall:
    handle: Any
    task_id: str
    category: str
    started_ns: int
    approval_outcome: str = "not_required"


@dataclass
class _TaskRun:
    task_id: str
    handle: Any
    context: contextvars.Context
    started_ns: int
    start_fields: dict[str, str]
    model_call_ids: set[str] = field(default_factory=set)
    tool_call_ids: set[tuple[str, str, str]] = field(default_factory=set)
    turn_ids: set[str] = field(default_factory=set)
    retired_turn_ids: frozenset[str] = field(default_factory=frozenset)
    completed_tool_call_ids: set[tuple[str, str, str]] = field(default_factory=set)
    unidentified_tool_calls: int = 0
    retry_count: int = 0


@dataclass
class _MetricsSession:
    session_id: str
    relay_session: relay_runtime.RelaySession
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    closing: bool = False
    model_calls: dict[tuple[str, str], _ModelCall] = field(default_factory=dict)
    tasks: dict[str, _TaskRun] = field(default_factory=dict)
    tool_calls: dict[tuple[str, str, str, str], _ToolCall] = field(default_factory=dict)
    retired_turn_ids: deque[str] = field(default_factory=lambda: deque(maxlen=256))


class _Runtime:
    """Own shared-metrics state layered on the Hermes core Relay host."""

    def __init__(self, host: relay_runtime.RelayRuntime | None = None) -> None:
        resolved_host = host or relay_runtime.get_runtime()
        if resolved_host is None:
            raise RuntimeError("Hermes core Relay runtime is unavailable")
        self.host: relay_runtime.RelayRuntime = resolved_host
        self.relay = self.host.relay
        self._sessions_lock = threading.RLock()
        self._active = True
        self._sessions: dict[str, _MetricsSession] = {}
        self._task_creation_lock = threading.RLock()
        self._task_sessions_lock = threading.RLock()
        # Guards the opt-in send pass: at most one in flight per process.
        self._send_lock = threading.RLock()
        self._send_thread: threading.Thread | None = None
        self._task_sessions: dict[tuple[str, str], _MetricsSession] = {}
        self._turn_sessions: dict[tuple[str, str], _MetricsSession] = {}
        self._subscriber_name = f"{SUBSCRIBER_NAME}.{self.host.runtime_id}"
        self.subscriber = SharedMetricsSubscriber(
            SharedMetricsStore(), __version__, runtime_id=self.host.runtime_id
        )
        self.relay.subscribers.register(self._subscriber_name, self.subscriber)
        self.host.retain_managed_execution(self._subscriber_name)
        self._registered = True
        atexit.register(self.shutdown)

    def ensure_session(self, event: dict[str, Any]) -> _MetricsSession | None:
        session_id = _text(event, "session_id")
        if not session_id:
            return None
        with self._sessions_lock:
            if not self._active:
                return None
            relay_session = self.host.ensure_session(event)
            if relay_session is None:
                return None
            session = self._sessions.get(session_id)
            if session is None:
                session = _MetricsSession(session_id=session_id, relay_session=relay_session)
                self._sessions[session_id] = session
        with session.lock:
            if session.closing:
                return None
        return session

    def record_client_active(self, event: dict[str, Any]) -> None:
        """Emit one payload-free activation attempt under the session scope."""
        session = self.ensure_session(event)
        if session is not None:
            self._emit_client_active(session)

    def _emit_client_active(self, session: _MetricsSession) -> None:
        with session.lock:
            if not session.closing:
                self._mark(session, None, CLIENT_ACTIVE_MARK, {})

    def _mark(
        self, session: _MetricsSession, task: _TaskRun | None, name: str, data: dict[str, str]
    ) -> None:
        """Emit one Relay mark under the task scope when given, else the session scope."""
        handle = task.handle if task is not None else session.relay_session.handle
        self._run_scoped(
            session, task, self.relay.scope.event, name,
            handle=handle, data=data, metadata=self._event_metadata(),
        )

    def _run_in_session(
        self, session: _MetricsSession, callback: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any:
        return self.host.run_in_session(session.relay_session, callback, *args, **kwargs)

    def start_task(self, event: dict[str, Any]) -> _TaskRun | None:
        """Open one Relay function scope for a Hermes task run."""
        task_key = _session_pair(event, "task_id")
        if task_key is None:
            return None
        _, task_id = task_key
        with self._task_creation_lock:
            owner = self._task_session(event)
            if owner is not None:
                with owner.lock:
                    if owner.closing:
                        return None
                    task = owner.tasks.get(task_id)
                    if task is not None and not self._admits(owner, task, event):
                        return None
                    return task

            session = self.ensure_session(event)
            if session is None:
                return None
            with session.lock:
                turn_id = _text(event, "turn_id")
                if (
                    session.closing
                    or (turn_id and turn_id in session.retired_turn_ids)
                    or session.relay_session.context is None
                ):
                    return None
                self._emit_client_active(session)
                task_context = session.relay_session.context.copy()
                start_fields = task_start_fields(event)
                active_turn = relay_runtime.active_turn(session.session_id)
                parent_handle = session.relay_session.handle
                if (
                    active_turn is not None
                    and active_turn.lease.session_id == session.session_id
                    and active_turn.task_id == task_id
                    and active_turn.handle is not None
                ):
                    parent_handle = active_turn.handle

                handle = task_context.run(
                    self._with_scope_stack, self.relay.scope.push,
                    TASK_SCOPE, self.relay.ScopeType.Function,
                    handle=parent_handle, input=start_fields, metadata=self._event_metadata(),
                )
                task = _TaskRun(
                    task_id=task_id,
                    handle=handle,
                    context=task_context,
                    started_ns=monotonic_ns(),
                    start_fields=start_fields,
                    retired_turn_ids=frozenset(session.retired_turn_ids),
                )
                session.tasks[task_id] = task
                with self._task_sessions_lock:
                    self._task_sessions[task_key] = session
                self._remember_turn(session, task, event)
                return task

    def _run_in_task(
        self, task: _TaskRun, callback: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any:
        return task.context.copy().run(self._with_scope_stack, callback, *args, **kwargs)

    def _with_scope_stack(self, callback: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        self.relay.get_scope_stack()
        return callback(*args, **kwargs)

    def start_model_call(self, event: dict[str, Any]) -> None:
        task_id = _text(event, "task_id")
        session, task = self._task_for(event, start=True)
        if task_id and task is None:
            return
        if session is None:
            session = self.ensure_session(event)
        if session is None:
            return
        request_id = _text(event, "api_request_id")
        if not request_id:
            return
        model_call_key = (task_id, request_id)
        fields = model_call_fields(event)
        retry_ordinal = _retry_ordinal(event)
        with session.lock:
            if session.closing:
                return
            if task is not None and not self._admits(session, task, event, current=True):
                return
            existing = session.model_calls.get(model_call_key)
            if existing is not None:
                existing.fields = fields
                if task is not None:
                    # Every repeated start for one logical request is another physical
                    # attempt. Provider fallback resets Hermes's provider-local retry
                    # ordinal, so ordinal deltas are not a reliable task-level counter.
                    task.retry_count += 1
                if retry_ordinal is not None:
                    existing.retry_ordinal = max(existing.retry_ordinal or 0, retry_ordinal)
                return
            if task is not None:
                task.model_call_ids.add(request_id)
                if retry_ordinal is not None and retry_ordinal > 0:
                    # A real Hermes retry can advance api_request_id while carrying the
                    # retry ordinal. Count that physical attempt.
                    task.retry_count += 1
            handle = self._run_scoped(
                session, task, self.relay.llm.call, MODEL_CALL_SCOPE, self.relay.LLMRequest({}, {}),
                handle=task.handle if task is not None else session.relay_session.handle,
                metadata=self._event_metadata(),
                model_name=MODEL_CALL_PROFILE_MODEL,
            )
            session.model_calls[model_call_key] = _ModelCall(
                handle=handle, task_id=task_id, fields=fields, retry_ordinal=retry_ordinal
            )

    def record_model_call_error(self, event: dict[str, Any]) -> None:
        """Retain the latest attempt error without closing the logical call."""
        self._update_model_call(event, finish=False)

    def end_model_call(self, event: dict[str, Any]) -> None:
        self._update_model_call(event, finish=True)

    def _update_model_call(self, event: dict[str, Any], *, finish: bool) -> None:
        """Refresh the located model call's fields from ``event``, optionally closing it."""
        session = self._any_session(event)
        if session is None:
            return
        with session.lock:
            if session.closing:
                return
            model_call_key = self._existing_model_call_key(session, event)
            model_call = session.model_calls.get(model_call_key) if model_call_key else None
            if model_call is None:
                return
            model_call.fields = model_call_fields(event)
            if finish:
                self._finish_model_call(session, model_call_key)

    def start_tool_call(self, event: dict[str, Any]) -> None:
        """Open one privacy-safe Relay tool lifecycle under its task."""
        task_id = _text(event, "task_id")
        session, task = self._task_for(event, start=True)
        if session is None or task is None or not _text(event, "tool_call_id"):
            return
        identity = self._tool_call_identity(event)
        with session.lock:
            if not self._admits(session, task, event):
                return
            key = (task_id, *identity)
            if identity in task.completed_tool_call_ids or key in session.tool_calls:
                return
            task.tool_call_ids.add(identity)
            session.tool_calls[key] = self._open_tool_call(task, event)

    def record_approval(self, event: dict[str, Any]) -> None:
        """Record one bounded approval result without approval text or commands."""
        session, task = self._approval_task(event)
        if session is None or task is None:
            return
        outcome = tool_approval_outcome(event)
        attribution = "unattributed"
        with session.lock:
            if session.closing or not self._event_matches_task_turn(task, event):
                return
            if _text(event, "tool_call_id"):
                identity = self._tool_call_identity(event)
                tool_call = session.tool_calls.get((task.task_id, *identity))
                if tool_call is None:
                    key = _sole(self._compatible_tool_call_keys(session, task.task_id, identity))
                    tool_call = session.tool_calls[key] if key is not None else None
                if tool_call is not None:
                    tool_call.approval_outcome = outcome
                    attribution = "tool_call"
            self._mark(
                session, task, TOOL_APPROVAL_MARK, {"attribution": attribution, "outcome": outcome}
            )

    def record_tool_call(self, event: dict[str, Any]) -> None:
        """Close and count one unique privacy-safe tool lifecycle."""
        task_id = _text(event, "task_id")
        session, task = self._task_for(event, start=False)
        if session is None or task is None:
            return
        with session.lock:
            if not self._admits(session, task, event):
                return
            tool_call = None
            if _text(event, "tool_call_id"):
                observed_identity = self._tool_call_identity(event)
                if observed_identity in task.completed_tool_call_ids:
                    return
                identity = observed_identity
                tool_call = session.tool_calls.pop((task_id, *identity), None)
                if tool_call is None:
                    if any(
                        self._tool_call_identities_are_compatible(completed, observed_identity)
                        for completed in task.completed_tool_call_ids
                    ):
                        return
                    matching_keys = self._compatible_tool_call_keys(
                        session, task_id, observed_identity
                    )
                    if len(matching_keys) > 1:
                        # Partial context cannot safely choose between concurrent calls
                        # that reused the provider-local ID.
                        return
                    if matching_keys:
                        identity = matching_keys[0][1:]
                        tool_call = session.tool_calls.pop(matching_keys[0])
                task.completed_tool_call_ids.update({identity, observed_identity})
                task.tool_call_ids.add(identity)
            else:
                task.unidentified_tool_calls += 1
            if tool_call is None:
                tool_call = self._open_tool_call(task, event)
            self._finish_tool_call(task, tool_call, event)

    def record_skill_lifecycle(self, event: dict[str, Any]) -> None:
        """Emit one allowlisted skill fact without its local identity."""
        if _text(event, "action").strip().lower() == "loaded":
            mark, fields = SKILL_LOAD_MARK, skill_load_fields(event)
        else:
            mark, fields = SKILL_LIFECYCLE_MARK, skill_lifecycle_fields(event)
        if fields is None:
            return

        session_id, task_id = _text(event, "session_id"), _text(event, "task_id")
        session, task = self._task_pair(event, allow_task_id_fallback=not session_id)
        if session is not None:
            if task is None:
                return
            with session.lock:
                if (
                    session.closing
                    or session.tasks.get(task.task_id) is not task
                    or not self._event_matches_task_turn(task, event)
                ):
                    return
                self._mark(session, task, mark, fields)
            return
        if session_id and task_id:
            return

        self._with_scope_stack(
            self.relay.scope.event, mark, data=fields, metadata=self._event_metadata()
        )

    def finish_task(self, event: dict[str, Any]) -> None:
        """Close one task scope exactly once with bounded terminal fields."""
        session = self._any_session(event)
        if session is None:
            return
        with session.lock:
            if session.closing:
                return
            finished = self._finish_task(session, _text(event, "task_id"), event)
        if finished:
            self._flush_and_export("Hermes shared-metrics task flush failed")

    def close_session(self, event: dict[str, Any]) -> None:
        session = self._session(event)
        if session is None:
            return
        with session.lock:
            if session.closing:
                return
            session.closing = True
            self._abort_tasks(
                session, {**event, **_ABORTED, "completed": False, "interrupted": False}
            )
        try:
            self.relay.subscribers.flush()
        except Exception as exc:
            logger.warning(
                "Hermes shared-metrics session %s closed with errors: subscriber flush failed: %s",
                session.session_id,
                exc,
            )
        else:
            self._export()
        with self._sessions_lock:
            if self._sessions.get(session.session_id) is session:
                self._sessions.pop(session.session_id, None)

    def shutdown(self) -> None:
        with self._sessions_lock:
            self._active = False
            session_ids = list(self._sessions)
        for session_id in session_ids:
            self._safe(self.close_session, {"session_id": session_id})
        if not self._registered:
            return
        self._flush_and_export("Hermes shared-metrics shutdown flush failed")
        self._deregister()
        # The final export may have started a send; give it the same bounded chance
        # deactivate() gets, or a short-lived CLI exits and kills the daemon thread mid-request.
        self._join_send_thread()
        self._unregister_atexit()

    def _deregister(self) -> None:
        self._safe(self.relay.subscribers.deregister, self._subscriber_name)
        self.host.release_managed_execution(self._subscriber_name)
        self._registered = False

    def deactivate(self) -> None:
        """Stop collection without exporting locally aggregated metrics."""
        with self._sessions_lock:
            self._active = False
        self.subscriber.deactivate()
        if self._registered:
            self._deregister()
        with self._sessions_lock:
            sessions = list(self._sessions.values())
        for session in sessions:
            with session.lock:
                if session.closing:
                    continue
                session.closing = True
                self._abort_tasks(session, {"session_id": session.session_id, **_ABORTED})
        with self._sessions_lock:
            self._sessions.clear()
        with self._task_sessions_lock:
            self._task_sessions.clear()
            self._turn_sessions.clear()
        self._join_send_thread()
        self._unregister_atexit()

    def _join_send_thread(self, timeout: float = 2.0) -> None:
        """Give an in-flight send a brief, bounded chance to finish at exit.

        Pending packages stay in SQLite and go out next run, so blocking on a slow network
        is the wrong trade; the daemon thread dies with the process.
        """
        with self._send_lock:
            thread = self._send_thread
        if thread is None or not thread.is_alive():
            return
        try:
            thread.join(timeout)
        except Exception:
            logger.debug("Shared-metrics send thread join failed", exc_info=True)

    def _session(self, event: dict[str, Any]) -> _MetricsSession | None:
        with self._sessions_lock:
            return self._sessions.get(_text(event, "session_id"))

    def _any_session(self, event: dict[str, Any]) -> _MetricsSession | None:
        """Owner session by task/turn correlation, else by session_id."""
        return self._task_session(event, allow_task_id_fallback=True) or self._session(event)

    def _task_for(
        self, event: dict[str, Any], *, start: bool
    ) -> tuple[_MetricsSession | None, _TaskRun | None]:
        """Resolve (session, task) for a task-scoped hook, optionally opening the task."""
        session, task = self._task_pair(event, allow_task_id_fallback=True)
        if task is None and start:
            task = self.start_task(event)
            session = self._task_session(event) if task is not None else None
        return session, task

    def _task_pair(
        self, event: dict[str, Any], **lookup: Any
    ) -> tuple[_MetricsSession | None, _TaskRun | None]:
        session = self._task_session(event, **lookup)
        task = session.tasks.get(_text(event, "task_id")) if session is not None else None
        return session, task

    def _run_scoped(
        self, session: _MetricsSession, task: _TaskRun | None, callback: Callable[..., Any],
        *args: Any, **kwargs: Any,
    ) -> Any:
        """Run under the task context when the call belongs to a task, else the session."""
        if task is not None:
            return self._run_in_task(task, callback, *args, **kwargs)
        return self._run_in_session(session, callback, *args, **kwargs)

    def _flush_and_export(self, failure_message: str) -> None:
        """Flush the Relay subscriber, then export; a failed flush skips the export."""
        if self._guarded(failure_message, self._flush_ok):
            self._export()

    def _flush_ok(self) -> bool:
        self.relay.subscribers.flush()
        return True

    def _abort_tasks(self, session: _MetricsSession, base_event: dict[str, Any]) -> None:
        """Close every open task of a closing session as system-aborted (caller holds the lock)."""
        for task_id in list(session.tasks):
            self._finish_task(session, task_id, {**base_event, "task_id": task_id})
        self._end_pending_model_calls(session, base_event)

    def _unregister_atexit(self) -> None:
        with contextlib.suppress(Exception):
            atexit.unregister(self.shutdown)

    def _task_session(
        self, event: dict[str, Any], *, allow_task_id_fallback: bool = False
    ) -> _MetricsSession | None:
        session_id, task_id = _text(event, "session_id"), _text(event, "task_id")
        if not task_id:
            return None
        turn_key = _session_pair(event, "turn_id")
        with self._task_sessions_lock:
            owner = self._turn_sessions.get(turn_key) if turn_key is not None else None
            if owner is None and session_id:
                owner = self._task_sessions.get((session_id, task_id))
            if owner is not None:
                return owner
            if not allow_task_id_fallback:
                return None
            return _sole(
                session
                for (_, candidate_task_id), session in self._task_sessions.items()
                if candidate_task_id == task_id
            )

    def _remember_turn(
        self, session: _MetricsSession, task: _TaskRun, event: dict[str, Any]
    ) -> None:
        turn_id = _text(event, "turn_id")
        if not turn_id:
            return
        task.turn_ids.add(turn_id)
        with self._task_sessions_lock:
            self._turn_sessions[(session.session_id, turn_id)] = session

    @staticmethod
    def _tool_call_identity(event: dict[str, Any]) -> tuple[str, str, str]:
        """Identify one provider-local tool call without exporting its IDs."""
        return _text(event, "api_request_id"), _text(event, "turn_id"), _text(event, "tool_call_id")

    @staticmethod
    def _tool_call_identities_are_compatible(
        candidate: tuple[str, str, str], observed: tuple[str, str, str]
    ) -> bool:
        """Match partial hook context without crossing known call boundaries."""
        if not observed[2] or candidate[2] != observed[2]:
            return False
        return all(
            not left or not right or left == right
            for left, right in zip(candidate[:2], observed[:2], strict=True)
        )

    @classmethod
    def _compatible_tool_call_keys(
        cls, session: _MetricsSession, task_id: str, identity: tuple[str, str, str]
    ) -> list[tuple[str, str, str, str]]:
        return [
            key
            for key in session.tool_calls
            if key[0] == task_id and cls._tool_call_identities_are_compatible(key[1:], identity)
        ]

    @staticmethod
    def _event_matches_task_turn(task: _TaskRun, event: dict[str, Any]) -> bool:
        """Reject delayed hooks from a prior run that reused the task ID."""
        turn_id = _text(event, "turn_id")
        if not turn_id:
            return True
        if turn_id in task.retired_turn_ids:
            return False
        return not task.turn_ids or turn_id in task.turn_ids

    def _admits(
        self,
        session: _MetricsSession,
        task: _TaskRun,
        event: dict[str, Any],
        *,
        current: bool = False,
    ) -> bool:
        """Whether ``event`` may act on ``task`` (caller holds ``session.lock``).

        Rejects closing sessions and stale turns; with ``current`` also requires ``task`` to
        still be the session's live run for its ID. Admitted events have their turn remembered.
        """
        if session.closing or not self._event_matches_task_turn(task, event):
            return False
        if current and session.tasks.get(task.task_id) is not task:
            return False
        self._remember_turn(session, task, event)
        return True

    def _approval_task(
        self, event: dict[str, Any]
    ) -> tuple[_MetricsSession | None, _TaskRun | None]:
        """Resolve approval correlation without guessing across ambiguous turns."""
        active = relay_runtime.active_turn()
        if active is not None:
            session, task = self._task_pair(
                {**event, "session_id": active.lease.session_id, "task_id": active.task_id}
            )
            if task is not None:
                return session, task

        session, task = self._task_pair(event)
        if task is not None:
            return session, task

        turn_id = _text(event, "turn_id")
        if not turn_id:
            return None, None
        with self._task_sessions_lock:
            session = _sole(
                candidate
                for (owner_id, candidate_turn_id), candidate in self._turn_sessions.items()
                if candidate_turn_id == turn_id and self._sessions.get(owner_id) is candidate
            )
        if session is None:
            return None, None
        task = _sole(task for task in session.tasks.values() if turn_id in task.turn_ids)
        return (None, None) if task is None else (session, task)

    def _open_tool_call(self, task: _TaskRun, event: dict[str, Any]) -> _ToolCall:
        handle = self._run_in_task(
            task, self.relay.tools.call, TOOL_CALL_SCOPE, {},
            handle=task.handle, metadata=self._event_metadata(),
        )
        return _ToolCall(
            handle=handle,
            task_id=task.task_id,
            category=tool_category(event),
            started_ns=monotonic_ns(),
        )

    def _finish_tool_call(
        self, task: _TaskRun, tool_call: _ToolCall, event: dict[str, Any]
    ) -> None:
        fields = tool_terminal_fields(
            event,
            category=tool_call.category,
            approval_outcome=tool_call.approval_outcome,
            fallback_duration_ms=max(0, (monotonic_ns() - tool_call.started_ns) // 1_000_000),
        )
        self._guarded(
            "Hermes shared-metrics tool call close failed",
            self._run_in_task, task, self.relay.tools.call_end, tool_call.handle, fields,
            metadata=self._event_metadata(),
        )

    def _end_pending_tool_calls(
        self, session: _MetricsSession, task: _TaskRun, event: dict[str, Any]
    ) -> None:
        pending_keys = [key for key in session.tool_calls if key[0] == task.task_id]
        task_outcome, _, _ = task_terminal_state(event)
        status = {"cancelled": "cancelled", "timed_out": "timeout"}.get(task_outcome, "error")
        for key in pending_keys:
            tool_call = session.tool_calls.pop(key, None)
            if tool_call is not None:
                self._finish_tool_call(task, tool_call, {**event, "status": status})

    def _finish_model_call(self, session: _MetricsSession, model_call_key: tuple[str, str]) -> None:
        model_call = session.model_calls.pop(model_call_key, None)
        if model_call is None:
            return
        self._guarded(
            "Hermes shared-metrics model call close failed",
            self._run_scoped, session, session.tasks.get(model_call.task_id),
            self.relay.llm.call_end, model_call.handle, model_call.fields,
            metadata=self._event_metadata(),
        )

    def _end_pending_model_calls(self, session: _MetricsSession, event: dict[str, Any]) -> None:
        task_id = _text(event, "task_id")
        pending = [
            key
            for key, call in session.model_calls.items()
            if not task_id or call.task_id == task_id
        ]
        for model_call_key in pending:
            self._finish_model_call(session, model_call_key)

    @staticmethod
    def _existing_model_call_key(
        session: _MetricsSession, event: dict[str, Any]
    ) -> tuple[str, str] | None:
        """(task_id, request_id) of an open call; a task-less event may match by request alone."""
        request_id = _text(event, "api_request_id")
        if not request_id:
            return None
        key = (_text(event, "task_id"), request_id)
        if key in session.model_calls:
            return key
        if key[0]:
            return None
        candidates = [candidate for candidate in session.model_calls if candidate[1] == request_id]
        return candidates[0] if len(candidates) == 1 else None

    def _finish_task(self, session: _MetricsSession, task_id: str, event: dict[str, Any]) -> bool:
        task = session.tasks.get(task_id)
        if task is None:
            return False
        self._end_pending_tool_calls(session, task, event)
        self._end_pending_model_calls(session, {**event, "task_id": task_id})
        fields = task_terminal_fields(
            {**task.start_fields, **event},
            duration_ms=max(0, (monotonic_ns() - task.started_ns) // 1_000_000),
            model_call_count=len(task.model_call_ids),
            tool_call_count=len(task.tool_call_ids) + task.unidentified_tool_calls,
            retry_count=task.retry_count,
        )
        try:
            self._guarded(
                "Hermes shared-metrics task close failed",
                self._run_in_task, task, relay_runtime.pop_relay_scope, self.relay, task.handle,
                output=fields, metadata=self._event_metadata(),
            )
        finally:
            session.tasks.pop(task_id, None)
            session.retired_turn_ids.extend(task.turn_ids)
            with self._task_sessions_lock:
                task_key = (session.session_id, task_id)
                if self._task_sessions.get(task_key) is session:
                    self._task_sessions.pop(task_key, None)
                for turn_id in task.turn_ids:
                    turn_key = (session.session_id, turn_id)
                    if self._turn_sessions.get(turn_key) is session:
                        self._turn_sessions.pop(turn_key, None)
        return True

    def _export(self) -> None:
        exported = self._safe(self.subscriber.store.create_and_export_package_if_due)
        # Sending must never delay the caller: _export runs on finish_task, the user's
        # interactive path. The thread is about latency, not correctness.
        if exported is not None:
            self._safe(self._send_exported_packages)

    def _observe_send_consent(self, send_enabled: bool) -> None:
        """Reconcile consent windows with observed config; failures never break the export
        hook but log at warning (an unclosed consent window is privacy-relevant)."""
        self._guarded(
            "Unable to record a shared-metrics consent transition",
            _reconcile_store_consent, self.subscriber.store, send_enabled,
        )

    def _send_exported_packages(self) -> None:
        try:
            resolved = _resolved_send_config()
        except Exception:
            logger.debug("Unable to read shared-metrics send policy", exc_info=True)
            return

        # Observe the consent EDGE before deciding whether to send: the dominant revocation
        # case is "sending turned off while no pass is running", invisible to the send loop.
        self._observe_send_consent(resolved.send)
        if not resolved.send:
            return

        with self._send_lock:
            # One in-flight pass per process; the next hook fire picks up what is pending.
            if self._send_thread is not None and self._send_thread.is_alive():
                return
            thread = threading.Thread(
                target=self._run_send_pass,
                args=(resolved.endpoint,),
                name="hermes-shared-metrics-send",
                daemon=True,
            )
            self._send_thread = thread
            thread.start()

    def _run_send_pass(self, endpoint: str) -> None:
        from hermes_cli.observability.shared_metrics_sender import SharedMetricsSender

        def still_consented() -> bool:
            """Re-read consent so revoking `send` stops an in-flight pass."""
            resolved = _resolved_send_config()
            return resolved.send and resolved.endpoint == endpoint

        sender = SharedMetricsSender(self.subscriber.store, endpoint, consent_check=still_consented)
        self._guarded("Shared-metrics send pass failed", sender.send_pending)

    def _event_metadata(self) -> dict[str, str]:
        return {
            SCHEMA_KEY: SCHEMA_VERSION,
            relay_runtime.RUNTIME_INSTANCE_KEY: self.host.runtime_id,
        }

    @staticmethod
    def _guarded(message: str, callback: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Run ``callback``; log-and-swallow any exception, returning None."""
        try:
            return callback(*args, **kwargs)
        except Exception:
            logger.warning(message, exc_info=True)
            return None

    @classmethod
    def _safe(cls, callback: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        return cls._guarded("Hermes shared metrics operation failed", callback, *args, **kwargs)


def _resolved_send_config():
    """Resolve the opt-in send policy from the read-only config snapshot."""
    from hermes_cli.config import read_raw_config_readonly
    from hermes_cli.observability.shared_metrics_send_config import resolve_send_config

    return resolve_send_config(read_raw_config_readonly() or {})


def _reconcile_store_consent(store: SharedMetricsStore, send_enabled: bool) -> None:
    from hermes_cli.observability.shared_metrics_sender import reconcile_send_consent
    from hermes_cli.sqlite_util import write_txn

    with store._connection() as connection:
        with write_txn(connection):
            reconcile_send_consent(connection, send_enabled)


def enabled() -> bool:
    """Return the shared-metrics policy for the active Hermes profile."""
    profile_key = relay_runtime.current_profile_key()
    try:
        from hermes_cli.config import read_raw_config_readonly

        # Collection consent is profile-owned: managed overlays cannot opt a profile in or
        # out. Read-only fast path — this gate runs 2-3x per agent turn and the mutable
        # read_raw_config() paid a full config deepcopy on every call.
        config = read_raw_config_readonly() or {}
    except Exception:
        logger.debug("Unable to read Hermes shared-metrics policy", exc_info=True)
        config = None
    telemetry = config.get("telemetry") if isinstance(config, dict) else None
    shared_metrics = telemetry.get("shared_metrics") if isinstance(telemetry, dict) else None
    if isinstance(shared_metrics, dict) and shared_metrics.get("enabled") is True:
        return True
    with _RUNTIME_LOCK:
        runtime = _RUNTIMES.pop(profile_key, None)
        if isinstance(runtime, _Runtime):
            runtime.deactivate()
    return False


def handles_hook(hook_name: str) -> bool:
    return hook_name in HANDLED_HOOKS and enabled()


_consent_reconcile_done = False


def _reconcile_send_consent_once() -> None:
    """Reconcile consent windows with config, once per process.

    Runs BEFORE and INDEPENDENT of the collection gate, so a user with ``enabled: false``
    still gets send-consent windows reconciled. Skipped only when there is no store on disk
    AND consent is off: nothing to protect, and creating ``~/.hermes/telemetry`` for every
    fully-disabled user would be the wrong behaviour change.
    """
    global _consent_reconcile_done
    if _consent_reconcile_done:
        return
    _consent_reconcile_done = True
    try:
        from hermes_cli.observability.shared_metrics import SharedMetricsStore
        from hermes_constants import get_hermes_home

        resolved = _resolved_send_config()
        # Probe WITHOUT constructing a store: the constructor creates the directory and
        # schema as a side effect, which would make the skip below dead code.
        default_path = get_hermes_home() / "telemetry" / "shared_metrics" / "metrics.sqlite3"
        if not resolved.send and not default_path.exists():
            return
        _reconcile_store_consent(SharedMetricsStore(), resolved.send)
    except Exception:
        logger.warning("Unable to reconcile shared-metrics send consent", exc_info=True)


def observe_lifecycle(hook_name: str, **kwargs: Any) -> None:
    """Project one Hermes lifecycle event into the core Relay integration."""
    _reconcile_send_consent_once()
    if not handles_hook(hook_name) or not relay_runtime.relay_instrumentation_enabled():
        return
    runtime = _get_runtime()
    if runtime is None:
        return
    try:
        _HOOK_HANDLERS[hook_name](runtime, kwargs)
    except Exception:
        logger.warning("Hermes shared metrics hook failed: %s", hook_name, exc_info=True)


def _with_runtime_toolset(event: dict[str, Any]) -> dict[str, Any]:
    """Attach the toolset already declared by Hermes's runtime registry."""
    tool_name = _text(event, "tool_name")
    if event.get("toolset") or not tool_name:
        return event
    try:
        from model_tools import get_toolset_for_tool

        toolset = get_toolset_for_tool(tool_name)
    except Exception:
        toolset = None
    return {**event, "toolset": toolset or "other"}


def _close_child_session(runtime: _Runtime, kwargs: dict[str, Any]) -> None:
    child_session_id = _text(kwargs, "child_session_id")
    if child_session_id:
        runtime.close_session({"session_id": child_session_id})


_HOOK_HANDLERS: dict[str, Callable[[_Runtime, dict[str, Any]], Any]] = {
    "on_session_start": lambda rt, kw: rt.record_client_active(kw),
    "pre_llm_call": lambda rt, kw: rt.start_task(kw),
    "pre_api_request": lambda rt, kw: rt.start_model_call(kw),
    "pre_tool_call": lambda rt, kw: rt.start_tool_call(_with_runtime_toolset(kw)),
    "post_tool_call": lambda rt, kw: rt.record_tool_call(_with_runtime_toolset(kw)),
    "post_approval_response": lambda rt, kw: rt.record_approval(kw),
    "on_skill_lifecycle": lambda rt, kw: rt.record_skill_lifecycle(kw),
    "post_api_request": lambda rt, kw: rt.end_model_call(kw),
    "api_request_error": lambda rt, kw: rt.record_model_call_error(kw),
    "on_session_end": lambda rt, kw: rt.finish_task(kw),
    "subagent_stop": _close_child_session,
    "on_session_finalize": lambda rt, kw: rt.close_session(kw),
    "on_session_reset": lambda rt, kw: rt.close_session(kw),
}
HANDLED_HOOKS = frozenset(_HOOK_HANDLERS)


def _prepare_core_session(host: relay_runtime.RelayRuntime, context: dict[str, Any]) -> None:
    """Prepare the profile subscriber before the coordinator opens a scope."""
    del context
    if host.profile_key == relay_runtime.current_profile_key() and enabled():
        _get_runtime(retry_failed=True, host=host)


def start_task_run(
    *, session_id: str, task_id: str, platform: str, parent_session_id: str = ""
) -> None:
    """Start task metrics at the outer Hermes execution boundary."""
    if not enabled():
        return
    runtime = _get_runtime(retry_failed=True)
    if runtime is not None:
        runtime._safe(
            runtime.start_task,
            {
                "session_id": session_id, "task_id": task_id, "platform": platform,
                "parent_session_id": parent_session_id,
            },
        )


def finish_task_run(
    *,
    session_id: str,
    task_id: str,
    platform: str,
    result: dict[str, Any] | None = None,
    error: BaseException | None = None,
) -> None:
    """Finish task metrics for every return or exception path."""
    if not enabled():
        return
    runtime = _get_runtime()
    if runtime is None:
        return

    terminal = result if isinstance(result, dict) else {}
    interrupted = terminal.get("interrupted") is True
    completed = terminal.get("completed") is True
    failed = terminal.get("failed") is True
    reason = str(terminal.get("turn_exit_reason") or terminal.get("failure_reason") or "")
    if error is not None:
        interrupted = (
            isinstance(error, (KeyboardInterrupt, InterruptedError))
            or type(error).__name__ == "CancelledError"
        )
        completed = False
        failed = not interrupted
        if interrupted:
            reason = "interrupted_by_user"
        else:
            reason = "timed_out" if isinstance(error, TimeoutError) else "system_aborted"
    elif not reason:
        reason = "failed" if failed else "unknown"

    runtime._safe(
        runtime.finish_task,
        {
            "session_id": session_id, "task_id": task_id, "platform": platform,
            "completed": completed, "failed": failed, "interrupted": interrupted,
            "turn_exit_reason": reason,
        },
    )


def _get_runtime(
    *, retry_failed: bool = False, host: relay_runtime.RelayRuntime | None = None
) -> _Runtime | None:
    profile_key = relay_runtime.current_profile_key()
    with _RUNTIME_LOCK:
        runtime = _RUNTIMES.get(profile_key)
        if isinstance(runtime, _Runtime):
            if host is None or runtime.host is host:
                return runtime
            runtime.deactivate()
            _RUNTIMES.pop(profile_key, None)
        elif runtime is _RUNTIME_FAILED:
            if not retry_failed:
                return None
            _RUNTIMES.pop(profile_key, None)
        try:
            runtime = _Runtime(host=host)
        except Exception:
            logger.warning("Hermes shared metrics initialization failed", exc_info=True)
            _RUNTIMES[profile_key] = _RUNTIME_FAILED
            return None
        _RUNTIMES[profile_key] = runtime
        return runtime


relay_runtime.SESSION_COORDINATOR.register_session_initializer(
    SUBSCRIBER_NAME, _prepare_core_session
)


def _reset_for_tests() -> None:
    """Reset all profile-scoped shared-metrics state for isolated tests."""
    with _RUNTIME_LOCK:
        runtimes = list(_RUNTIMES.values())
        _RUNTIMES.clear()
    for runtime in runtimes:
        if isinstance(runtime, _Runtime):
            runtime.shutdown()
