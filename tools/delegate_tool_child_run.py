"""Helpers for running ONE pre-built child agent: heartbeat, registry entry, workspace seeding, timeout/failure handling, result-entry assembly and cleanup.

Split out of ``tools/delegate_tool.py``; every moved name is re-imported there, so
``tools.delegate_tool.<name>`` keeps resolving (and monkeypatching) as before.
"""

from __future__ import annotations

import logging
import json
import threading
import time
from concurrent.futures import TimeoutError as FuturesTimeoutError
from typing import Any, Dict, List, Optional
from agent.interrupt_compat import request_hard_interrupt
from dataclasses import dataclass
from tools import file_state
from tools.delegate_tool_registry import (
    _capture_gateway_steer_authority, _close_subagent_steering, _register_subagent, _unregister_subagent,
)
from tools.delegate_tool_results import (
    _extract_output_tail, _looks_like_error_output, _stringify_tool_content, _summarize_tool_arguments,
)

# Log-record parity with the origin module.
logger = logging.getLogger("tools.delegate_tool")


def _dump_subagent_timeout_diagnostic(
    *,
    child: Any,
    task_index: int,
    timeout_seconds: float,
    duration_seconds: float,
    worker_thread: Optional[threading.Thread],
    goal: str,
) -> Optional[str]:
    """Write a structured diagnostic dump for a subagent that timed out
    before making any API call.

    See issue #14726: users hit "subagent timed out after 300s with no response"
    with zero API calls and no way to inspect what happened. This helper
    writes a dedicated log under ``~/.hermes/logs/subagent-<sid>-<ts>.log``
    capturing the child's config, system-prompt / tool-schema sizes, activity
    tracker snapshot, and the worker thread's Python stack at timeout.

    Returns the absolute path to the diagnostic file, or None on failure.
    """
    try:
        from hermes_constants import get_hermes_home
        import datetime as _dt
        import sys as _sys
        import traceback as _traceback
        import threading as _threading

        hermes_home = get_hermes_home()
        logs_dir = hermes_home / "logs"
        try:
            logs_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            return None

        subagent_id = getattr(child, "_subagent_id", None) or f"idx{task_index}"
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        dump_path = logs_dir / f"subagent-timeout-{subagent_id}-{ts}.log"

        lines: List[str] = []
        def _w(line: str = "") -> None:
            lines.append(line)

        _w("# Subagent timeout diagnostic — issue #14726")
        _w(f"# Generated: {_dt.datetime.now().isoformat()}")
        _w("")
        _w("## Timeout")
        _w(f"  task_index:        {task_index}")
        _w(f"  subagent_id:       {subagent_id}")
        _w(f"  configured_timeout: {timeout_seconds}s")
        _w(f"  actual_duration:   {duration_seconds:.2f}s")
        _w("")

        _w("## Goal")
        _goal_preview = (goal or "").strip()
        if len(_goal_preview) > 1000:
            _goal_preview = _goal_preview[:1000] + " ...[truncated]"
        _w(_goal_preview or "(empty)")
        _w("")

        _w("## Child config")
        for attr in (
            "model", "provider", "api_mode", "base_url", "max_iterations",
            "quiet_mode", "skip_memory", "skip_context_files", "platform",
            "_delegate_role", "_delegate_depth",
        ):
            try:
                val = getattr(child, attr, None)
                # Redact api_key-shaped values defensively
                if isinstance(val, str) and attr == "base_url":
                    pass
                _w(f"  {attr}: {val!r}")
            except Exception:
                _w(f"  {attr}: <unreadable>")
        _w("")

        _w("## Toolsets")
        enabled = getattr(child, "enabled_toolsets", None)
        _w(f"  enabled_toolsets:  {enabled!r}")
        tool_names = getattr(child, "valid_tool_names", None)
        if tool_names:
            _w(f"  loaded tool count: {len(tool_names)}")
            try:
                _w(f"  loaded tools:      {sorted(tool_names)}")
            except Exception:
                pass
        _w("")

        _w("## Prompt / schema sizes")
        try:
            sys_prompt = getattr(child, "ephemeral_system_prompt", None) \
                or getattr(child, "system_prompt", None) \
                or ""
            _w(f"  system_prompt_bytes: {len(sys_prompt.encode('utf-8')) if isinstance(sys_prompt, str) else 'n/a'}")
            _w(f"  system_prompt_chars: {len(sys_prompt) if isinstance(sys_prompt, str) else 'n/a'}")
        except Exception as exc:
            _w(f"  system_prompt: <error: {exc}>")
        try:
            tools_schema = getattr(child, "tools", None)
            if tools_schema is not None:
                _schema_json = json.dumps(tools_schema, default=str)
                _w(f"  tool_schema_count: {len(tools_schema)}")
                _w(f"  tool_schema_bytes: {len(_schema_json.encode('utf-8'))}")
        except Exception as exc:
            _w(f"  tool_schema: <error: {exc}>")
        _w("")

        _w("## Activity summary")
        try:
            summary = child.get_activity_summary()
            for k, v in summary.items():
                _w(f"  {k}: {v!r}")
        except Exception as exc:
            _w(f"  <get_activity_summary failed: {exc}>")
        _w("")

        _w("## Worker thread stack at timeout")
        if worker_thread is not None and worker_thread.is_alive():
            frames = _sys._current_frames()
            worker_frame = frames.get(worker_thread.ident)
            if worker_frame is not None:
                stack = _traceback.format_stack(worker_frame)
                for frame_line in stack:
                    for sub in frame_line.rstrip().split("\n"):
                        _w(f"  {sub}")
            else:
                _w("  <worker frame not available>")
        elif worker_thread is None:
            _w("  <no worker thread handle>")
        else:
            _w("  <worker thread already exited>")
        _w("")

        # All other live threads (bounded to 40): the worker is often parked on
        # a helper thread, so a pre-HTTP wedge is indistinguishable from a slow
        # provider without the full picture.
        _w("## All thread stacks at timeout")
        try:
            frames = _sys._current_frames()
            by_ident = {
                th.ident: th for th in _threading.enumerate() if th.ident
            }
            worker_ident = worker_thread.ident if worker_thread else None
            dumped = 0
            for ident, frame in frames.items():
                if ident == worker_ident:
                    continue  # already dumped above
                if dumped >= 40:
                    _w(f"  <{len(frames) - dumped - 1} more threads omitted>")
                    break
                th = by_ident.get(ident)
                name = th.name if th else f"ident={ident}"
                daemon = " daemon" if (th and th.daemon) else ""
                _w(f"  --- {name}{daemon} ---")
                for frame_line in _traceback.format_stack(frame):
                    for sub in frame_line.rstrip().split("\n"):
                        _w(f"    {sub}")
                dumped += 1
        except Exception as exc:
            _w(f"  <all-thread dump failed: {exc}>")
        _w("")

        _w("## Notes")
        _w("  This file is written ONLY when a subagent times out with 0 API calls.")
        _w("  0-API-call timeouts mean the child never reached its first LLM request.")
        _w("  Common causes: oversized prompt rejected by provider, transport hang,")
        _w("  credential resolution stuck. See issue #14726 for context.")

        dump_path.write_text("\n".join(lines), encoding="utf-8")
        return str(dump_path)
    except Exception as exc:
        logger.warning("Subagent timeout diagnostic dump failed: %s", exc)
        return None


def _start_heartbeat(child: Any, parent_agent: Any, task_index: int) -> tuple:
    """Build the parent-activity heartbeat thread for one child (not started).

    Returns ``(stop_event, thread)``. The caller starts the thread inside its
    ``try`` so a failed ``start()`` (OS thread exhaustion) leaves ``ident`` None
    and the finally-path join can be skipped safely.
    """
    from tools.delegate_tool import (
        _HEARTBEAT_INTERVAL,
        _HEARTBEAT_STALE_CYCLES_IDLE,
        _HEARTBEAT_STALE_CYCLES_IN_TOOL,
    )

    _heartbeat_stop = threading.Event()
    # Stale detection: a cycle counts as stale when (tool, iteration,
    # activity_ts) all froze; thresholds differ idle vs in-tool.
    _last_seen_iter = [0]
    _last_seen_tool = [None]  # type: list
    _last_seen_activity_ts = [None]  # type: list
    _stale_count = [0]

    def _heartbeat_loop():
        while not _heartbeat_stop.wait(_HEARTBEAT_INTERVAL):
            if parent_agent is None:
                continue
            touch = getattr(parent_agent, "_touch_activity", None)
            if not touch:
                continue
            # Pull detail from the child's own activity tracker
            desc = f"delegate_task: subagent {task_index} working"
            try:
                child_summary = child.get_activity_summary()
                child_tool = child_summary.get("current_tool")
                child_iter = child_summary.get("api_call_count", 0)
                child_max = child_summary.get("max_iterations", 0)
                child_activity_ts = child_summary.get("last_activity_ts")

                # A slow model wait refreshes last_activity_ts (direct_api_call
                # heartbeat), so it never looks stale at the idle threshold.
                iter_advanced = child_iter > _last_seen_iter[0]
                tool_changed = child_tool != _last_seen_tool[0]
                activity_advanced = (
                    child_activity_ts is not None
                    and (
                        _last_seen_activity_ts[0] is None
                        or child_activity_ts > _last_seen_activity_ts[0]
                    )
                )
                if iter_advanced or tool_changed or activity_advanced:
                    _last_seen_iter[0] = child_iter
                    _last_seen_tool[0] = child_tool
                    if child_activity_ts is not None:
                        _last_seen_activity_ts[0] = child_activity_ts
                    _stale_count[0] = 0
                else:
                    _stale_count[0] += 1

                stale_limit = (
                    _HEARTBEAT_STALE_CYCLES_IN_TOOL
                    if child_tool
                    else _HEARTBEAT_STALE_CYCLES_IDLE
                )
                if _stale_count[0] >= stale_limit:
                    logger.warning(
                        "Subagent %d appears stale (no progress for %d "
                        "heartbeat cycles, tool=%s) — stopping heartbeat",
                        task_index,
                        _stale_count[0],
                        child_tool or "<none>",
                    )
                    break  # stop touching parent, let gateway timeout fire

                if child_tool:
                    desc = (
                        f"delegate_task: subagent running {child_tool} "
                        f"(iteration {child_iter}/{child_max})"
                    )
                else:
                    child_desc = child_summary.get("last_activity_desc", "")
                    if child_desc:
                        desc = (
                            f"delegate_task: subagent {child_desc} "
                            f"(iteration {child_iter}/{child_max})"
                        )
            except Exception:
                pass
            try:
                touch(desc)
            except Exception:
                pass

    _heartbeat_thread = threading.Thread(target=_heartbeat_loop, daemon=True)
    return _heartbeat_stop, _heartbeat_thread


def _register_child(
    child: Any,
    parent_agent: Any,
    goal: str,
    *,
    owner_session_id: Optional[str],
    owner_transport: Any,
    owner_session_record: Any,
) -> Optional[str]:
    """Register the live child in the module registry; return its subagent_id.

    Test doubles without a stable string ``_subagent_id`` are not registered
    (returns None) and the caller skips every registry interaction for them.
    """
    _raw_sid = getattr(child, "_subagent_id", None)
    _subagent_id = _raw_sid if isinstance(_raw_sid, str) else None
    if _subagent_id:
        if owner_session_id is None:
            try:
                from gateway.session_context import get_session_env

                owner_session_id = get_session_env("HERMES_UI_SESSION_ID", "") or None
            except Exception:
                owner_session_id = None
        if owner_session_id and (
            owner_transport is None or owner_session_record is None
        ):
            owner_transport, owner_session_record = (
                _capture_gateway_steer_authority(owner_session_id)
            )
        _raw_depth = getattr(child, "_delegate_depth", 1)
        _tui_depth = max(0, _raw_depth - 1) if isinstance(_raw_depth, int) else 0
        _parent_sid = getattr(child, "_parent_subagent_id", None)
        # Owning conversation's durable session id (same lineage completion
        # delivery routes by); sourced from the child's stamp so it survives a
        # parent_agent rebuild between dispatch and run.
        _owner_agent_session_id = (
            str(getattr(child, "_parent_session_id", "") or "")
            or str(getattr(parent_agent, "session_id", "") or "")
        )
        _delegation_id = getattr(child, "_delegation_id", None)
        _register_subagent(
            {
                "subagent_id": _subagent_id,
                "parent_id": _parent_sid if isinstance(_parent_sid, str) else None,
                "depth": _tui_depth,
                "goal": goal,
                "delegation_id": (
                    _delegation_id if isinstance(_delegation_id, str) else None
                ),
                "model": (
                    getattr(child, "model", None)
                    if isinstance(getattr(child, "model", None), str)
                    else None
                ),
                "started_at": time.time(),
                "status": "running",
                "tool_count": 0,
                "agent": child,
                # list/steer/stop ownership when the weakref chain breaks
                # (CLI rebuilds its AIAgent mid-session).
                "owner_agent_session_id": _owner_agent_session_id or None,
                # Immutable live gateway/TUI session that commissioned this
                # child. Empty outside those hosts; RPC authority fails closed.
                "owner_session_id": owner_session_id,
                "owner_transport": owner_transport,
                "owner_session_record": owner_session_record,
            }
        )

    return _subagent_id


class _WorktreeReporter:
    """Holds the child's worktree-isolation state and reports it into result entries.

    ``info`` stays None until isolation engages, so ``attach`` is a no-op on
    every early error path.
    """

    def __init__(self) -> None:
        self.info: Optional[Dict[str, str]] = None

    def attach(self, entry_dict: Dict[str, Any]) -> None:
        """Inspect + prune the child worktree, reporting into the entry."""
        _worktree_info = self.info
        if _worktree_info is None:
            return
        try:
            from tools import subagent_worktree

            entry_dict["worktree"] = (
                subagent_worktree.finalize_subagent_worktree(_worktree_info)
            )
        except Exception as e:
            # State is unknown: emit the SAME flagged schema the parent expects,
            # via the shared factory so the two producers never drift.
            logger.warning("worktree finalize failed: %s", e)
            try:
                from tools import subagent_worktree as _sw

                entry_dict["worktree"] = _sw.unproven_worktree_payload(
                    _worktree_info, f"finalize raised: {e}"
                )
            except Exception:
                # Import itself failed — inline the same shape rather than
                # dropping the flag (the parent must still see the warning).
                entry_dict["worktree"] = {
                    "path": _worktree_info.get("path", ""),
                    "branch": _worktree_info.get("branch", ""),
                    "commits": 0,
                    "dirty": False,
                    "pruned": False,
                    "inspection_failed": True,
                    "note": (
                        f"worktree finalize raised ({e}) and the reporting "
                        "helper was unavailable: 'commits' and 'dirty' are "
                        "UNKNOWN, not zero/clean. Inspect "
                        f"{_worktree_info.get('path', '')} before assuming "
                        "no work."
                    ),
                }


@dataclass
class _ChildWorkspace:
    child_task_id: str
    parent_task_id: Optional[str]
    goal: str
    wall_start: float
    parent_reads_snapshot: list


def _seed_child_workspace(
    child: Any,
    parent_agent: Any,
    goal: str,
    task_index: int,
    subagent_id: Optional[str],
    worktree: _WorktreeReporter,
) -> _ChildWorkspace:
    """Seed cwd/container aliases and optional worktree isolation for the child.

    Returns the ids the run needs; ``goal`` comes back extended with the
    worktree contract note when isolation engaged.
    """
    from tools.delegate_tool import _get_worktree_isolation, _resolve_workspace_hint

    _subagent_id = subagent_id
    _worktree_info = None
    import uuid as _uuid

    child_task_id = _subagent_id or f"subagent-{task_index}-{_uuid.uuid4().hex[:8]}"
    parent_task_id = getattr(parent_agent, "_current_task_id", None)
    # Seed the child's cwd record from the parent's: same starting directory,
    # but the child's later `cd`s stay in its own record.
    try:
        from tools.terminal_tool import (
            get_session_cwd,
            record_session_cwd,
            register_container_alias,
        )

        record_session_cwd(child_task_id, get_session_cwd(parent_task_id))
        # Per-session container isolation keys containers by task_id; the
        # child must share the PARENT's container.
        register_container_alias(child_task_id, parent_task_id)
    except Exception as e:
        logger.debug("Child cwd seed failed: %s", e)

    # Opt-in worktree isolation: own git worktree off the parent's HEAD, terminal
    # started there. Git-only, local-backend-only; failures degrade silently.
    if _get_worktree_isolation():
        try:
            from tools import subagent_worktree

            if subagent_worktree.local_backend_active():
                _parent_cwd = None
                try:
                    from tools.terminal_tool import get_session_cwd as _gsc

                    _parent_cwd = _gsc(parent_task_id)
                except Exception:
                    pass
                _worktree_info = subagent_worktree.create_subagent_worktree(
                    _parent_cwd or _resolve_workspace_hint(parent_agent),
                    subagent_id=_subagent_id,
                )
            else:
                logger.debug(
                    "worktree isolation skipped: non-local terminal backend"
                )
        except Exception as e:
            logger.debug("worktree isolation setup failed: %s", e)
        if _worktree_info is not None:
            try:
                from tools.terminal_tool import record_session_cwd as _rsc

                _rsc(child_task_id, _worktree_info["path"])
            except Exception as e:
                logger.debug("worktree cwd seed failed: %s", e)
            # The child's context is already built; carry the isolation
            # contract on the goal message instead (same turn, no
            # system-prompt mutation).
            from tools.subagent_worktree import build_worktree_context_note

            goal = goal + build_worktree_context_note(_worktree_info)

    wall_start = time.time()
    parent_reads_snapshot = (
        list(file_state.known_reads(parent_task_id)) if parent_task_id else []
    )
    worktree.info = _worktree_info
    return _ChildWorkspace(child_task_id, parent_task_id, goal, wall_start, parent_reads_snapshot)


@dataclass
class _ChildFailure:
    entry: Dict[str, Any]
    close_deferred: bool


def _handle_child_wait_failure(
    exc: BaseException,
    *,
    child: Any,
    task_index: int,
    goal: str,
    subagent_id: Optional[str],
    child_future: Any,
    child_timeout: Optional[float],
    child_start: float,
    child_progress_cb: Any,
    worker_thread_holder: Dict[str, Optional[threading.Thread]],
    worktree: _WorktreeReporter,
) -> _ChildFailure:
    """Build the error entry for a child whose Future timed out or raised.

    Signals the child to stop, dumps the 0-API-call diagnostic, and on a
    timeout hands ``child.close()`` to a Future done-callback (returned as
    ``close_deferred=True``) because closing from this thread races the
    still-unwinding worker's finally path.
    """
    _timeout_exc = exc
    _subagent_id = subagent_id
    _child_future = child_future
    _worker_thread_holder = worker_thread_holder
    _attach_worktree = worktree.attach
    _child_close_deferred = False
    # No consumer boundary remains once this owner stops waiting for
    # the child. Close acceptance before any completion callback and
    # retain steer text that won the race with this failure/timeout.
    _late_pending_steer = (
        _close_subagent_steering(_subagent_id, child) if _subagent_id else None
    )
    # Signal the child to stop so its thread can exit cleanly.
    try:
        interrupted = child is not None and request_hard_interrupt(child)
        if not interrupted and child is not None and hasattr(child, "_interrupt_requested"):
            child._interrupt_requested = True
    except Exception:
        pass

    is_timeout = isinstance(_timeout_exc, (FuturesTimeoutError, TimeoutError))
    duration = round(time.monotonic() - child_start, 2)
    logger.warning(
        "Subagent %d %s after %.1fs",
        task_index,
        "timed out" if is_timeout else f"raised {type(_timeout_exc).__name__}",
        duration,
    )

    # When a subagent times out BEFORE making any API call, dump a
    # diagnostic to help users (and us) see what the child was doing.
    # See #14726 — without this, 0-API-call hangs are black boxes.
    diagnostic_path: Optional[str] = None
    child_api_calls = 0
    try:
        _summary = child.get_activity_summary()
        child_api_calls = int(_summary.get("api_call_count", 0) or 0)
    except Exception:
        pass
    if is_timeout and child_api_calls == 0:
        diagnostic_path = _dump_subagent_timeout_diagnostic(
            child=child,
            task_index=task_index,
            # is_timeout implies a cap was configured (result(timeout=None)
            # never raises FuturesTimeoutError); guard for the type checker.
            timeout_seconds=float(child_timeout or 0.0),
            duration_seconds=float(duration),
            worker_thread=_worker_thread_holder.get("t"),
            goal=goal,
        )
        if diagnostic_path:
            logger.warning(
                "Subagent %d 0-API-call timeout — diagnostic written to %s",
                task_index,
                diagnostic_path,
            )

    if child_progress_cb:
        try:
            child_progress_cb(
                "subagent.complete",
                preview=(
                    f"Timed out after {duration}s"
                    if is_timeout
                    else str(_timeout_exc)
                ),
                status="timeout" if is_timeout else "error",
                duration_seconds=duration,
                summary="",
            )
        except Exception:
            pass

    if is_timeout:
        if child_api_calls == 0:
            _err = (
                f"Subagent timed out after {child_timeout}s without "
                f"making any API call — the child never reached its "
                f"first LLM request (prompt construction, credential "
                f"resolution, or transport may be stuck)."
            )
            if diagnostic_path:
                _err += f" Diagnostic: {diagnostic_path}"
        else:
            _err = (
                f"Subagent timed out after {child_timeout}s with "
                f"{child_api_calls} API call(s) completed — likely "
                f"stuck on a slow API call, tool call, or unresponsive "
                f"network request."
            )
            if diagnostic_path:
                _err += f" Diagnostic: {diagnostic_path}"
    else:
        _err = str(_timeout_exc)

    _error_entry = {
        "task_index": task_index,
        "status": "timeout" if is_timeout else "error",
        "summary": None,
        "error": _err,
        "exit_reason": "timeout" if is_timeout else "error",
        "api_calls": child_api_calls,
        "duration_seconds": duration,
        "timeout_seconds": child_timeout if is_timeout else None,
        "timed_out_after_seconds": duration if is_timeout else None,
        "timeout_phase": (
            "before_first_llm_call" if is_timeout and child_api_calls == 0
            else "after_llm_calls" if is_timeout
            else None
        ),
        "_child_role": getattr(child, "_delegate_role", None),
        "diagnostic_path": diagnostic_path,
    }
    if _late_pending_steer:
        _error_entry["missed_steer"] = _late_pending_steer
        _error_entry["error"] += (
            " [steer did not land before the subagent stopped: "
            f"{_late_pending_steer}]"
        )
    _attach_worktree(_error_entry)
    if is_timeout and not _child_future.done():
        # The interrupt is cooperative: the worker still runs its finally path,
        # so closing the child now could close SQLite under its final write.
        # A Future done-callback is the first safe close boundary.
        def _close_after_timed_out_worker(_done_future) -> None:
            try:
                close = getattr(child, "close", None)
                if callable(close):
                    close()
            except Exception:
                logger.debug(
                    "Failed to close timed-out child after worker exit",
                    exc_info=True,
                )

        _child_future.add_done_callback(_close_after_timed_out_worker)
        _child_close_deferred = True

        # The abandoned worker is usually parked in an OpenSSL read. NEVER
        # hard-close that transport from this thread (cross-thread FD release
        # under a live SSL read corrupts native state); shutdown() the pooled
        # sockets instead — FD-safe, settles the read with EOF so the worker
        # unwinds. One immediate sweep + one delayed re-sweep for a connection
        # opened in between; a worker that still won't settle keeps its
        # resources until process exit.
        _drain = getattr(child, "_drain_transports_after_abandonment", None)
        if callable(_drain):
            def _drain_once(phase: str) -> None:
                try:
                    _drain(reason=f"delegate_timeout_{phase}")
                except Exception:
                    logger.debug(
                        "Timed-out child transport drain (%s) failed",
                        phase,
                        exc_info=True,
                    )

            _drain_once("immediate")

            def _drain_resweep() -> None:
                if not _child_future.done():
                    _drain_once("resweep")

            _resweep_timer = threading.Timer(5.0, _drain_resweep)
            _resweep_timer.daemon = True
            _resweep_timer.start()
    return _ChildFailure(_error_entry, _child_close_deferred)


@dataclass
class _SchemaOutcome:
    schema: Optional[Dict[str, Any]]
    valid: Optional[bool]
    errors: List[str]
    retries: int


def _validate_child_output_schema(
    child: Any, result: Dict[str, Any], task_index: int, child_task_id: str, relay_child_text: Any
) -> _SchemaOutcome:
    """Validate the final answer against the attached output_schema with ONE bounded retry.

    Schema-less children (no dict on ``child._delegate_output_schema``) take no
    branch here so their result entry stays byte-identical.
    """
    _relay_child_text = relay_child_text
    _output_schema = getattr(child, "_delegate_output_schema", None)
    _schema_valid: Optional[bool] = None
    _schema_errors: List[str] = []
    _schema_retries = 0
    if isinstance(_output_schema, dict):
        from tools.delegation_output_schema import (
            build_retry_message,
            validate_output,
        )

        _first_text = result.get("final_response") or ""
        _schema_valid, _schema_errors = validate_output(
            _first_text, _output_schema
        )
        if (
            not _schema_valid
            and _first_text.strip()
            and not result.get("interrupted", False)
        ):
            # Exactly one retry turn, carrying the validation errors
            # verbatim (no schema re-paste — the child already holds
            # the contract in its context).
            _schema_retries = 1
            _retry_result = None
            try:
                _retry_result = child.run_conversation(
                    user_message=build_retry_message(_schema_errors),
                    task_id=child_task_id,
                    stream_callback=_relay_child_text,
                )
            except Exception as _retry_exc:
                logger.warning(
                    "Subagent %d schema-retry turn failed: %s",
                    task_index,
                    _retry_exc,
                )
            if isinstance(_retry_result, dict):
                _retry_text = _retry_result.get("final_response") or ""
                if _retry_text.strip():
                    result["final_response"] = _retry_text
                try:
                    result["api_calls"] = int(
                        result.get("api_calls", 0) or 0
                    ) + int(_retry_result.get("api_calls", 0) or 0)
                except (TypeError, ValueError):
                    pass
                _retry_messages = _retry_result.get("messages")
                if isinstance(_retry_messages, list) and isinstance(
                    result.get("messages"), list
                ):
                    result["messages"] = result["messages"] + _retry_messages
                _schema_valid, _schema_errors = validate_output(
                    _retry_text, _output_schema
                )

    # Linearization boundary for registry steering. From this point on the
    # child cannot consume another steer. Closing under the registry lock
    # either rejects a concurrent caller or drains every previously accepted
    return _SchemaOutcome(_output_schema, _schema_valid, _schema_errors, _schema_retries)


def _build_result_entry(
    child: Any,
    result: Dict[str, Any],
    task_index: int,
    duration: float,
    schema: _SchemaOutcome,
) -> Dict[str, Any]:
    """Derive the parent-visible result entry (status, exit_reason, tool trace, tokens, cost).

    ``status`` / ``exit_reason`` / ``truncated`` follow the contract in the
    ``_run_single_child`` docstring; a structured failure always wins over the
    summary-presence heuristic (which is only a fallback for legacy/mock results).
    """
    _output_schema, _schema_valid, _schema_errors, _schema_retries = (
        schema.schema, schema.valid, schema.errors, schema.retries
    )
    summary = result.get("final_response") or ""
    completed = result.get("completed", False)
    interrupted = result.get("interrupted", False)
    api_calls = result.get("api_calls", 0)

    # "(empty)" is run_agent's give-up sentinel after repeated empty LLM
    # responses (usually a transport bug) — a failure, not a success.
    _empty_sentinel = summary.strip() == "(empty)"

    if interrupted:
        status = "interrupted"
    elif result.get("failed") or result.get("error"):
        # A structured failure WINS over the summary heuristic: the loop returns
        # the error text as final_response, which would otherwise read as
        # "completed". The heuristic only covers legacy/mock results.
        status = "failed"
    elif _schema_valid is False:
        # Declared schema still violated after the bounded retry: the summary
        # is unusable under the contract, so status must not say completed
        # (orchestrators reading only status/icon would accept an empty
        # verdict). None on schema-less runs, which never take this branch.
        status = "failed"
    elif summary and not _empty_sentinel:
        # A summary means the subagent produced usable output.
        # exit_reason ("completed" vs "max_iterations") already
        # tells the parent *how* the task ended.
        status = "completed"
    else:
        status = "failed"

    # Build tool trace from conversation messages (already in memory).
    # Uses tool_call_id to correctly pair parallel tool calls with results.
    tool_trace: list[Dict[str, Any]] = []
    trace_by_id: Dict[str, Dict[str, Any]] = {}
    messages = result.get("messages") or []
    if isinstance(messages, list):
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            if msg.get("role") == "assistant":
                for tc in msg.get("tool_calls") or []:
                    fn = tc.get("function", {})
                    arguments = fn.get("arguments", "")
                    entry_t = {
                        "tool": fn.get("name", "unknown"),
                        "args_bytes": len(arguments),
                        "input_summary": _summarize_tool_arguments(arguments),
                    }
                    tool_trace.append(entry_t)
                    tc_id = tc.get("id")
                    if tc_id:
                        trace_by_id[tc_id] = entry_t
            elif msg.get("role") == "tool":
                content = _stringify_tool_content(msg.get("content", ""))
                is_error = _looks_like_error_output(content)
                result_meta = {
                    "result_bytes": len(content),
                    "status": "error" if is_error else "ok",
                }
                # Match by tool_call_id for parallel calls
                tc_id = msg.get("tool_call_id")
                target = trace_by_id.get(tc_id) if tc_id else None
                if target is not None:
                    target.update(result_meta)
                elif tool_trace:
                    # Fallback for messages without tool_call_id
                    tool_trace[-1].update(result_meta)

    # Determine exit reason
    if interrupted:
        exit_reason = "interrupted"
    elif result.get("failed") or result.get("error"):
        # Provider rejection / terminal failure. Do NOT report this as
        # iteration-budget exhaustion — "max_iterations" is only truthful
        # when the child actually hit its per-delegation iteration cap.
        exit_reason = "error"
    elif completed:
        exit_reason = "completed"
    else:
        # Genuine budget exhaustion: completed=False with no failure.
        exit_reason = "max_iterations"

    # Extract token counts (safe for mock objects)
    _input_tokens = getattr(child, "session_prompt_tokens", 0)
    _output_tokens = getattr(child, "session_completion_tokens", 0)
    _model = getattr(child, "model", None)

    # Result entry contract: see the _run_single_child docstring.
    entry: Dict[str, Any] = {
        "task_index": task_index,
        "status": status,
        "summary": summary,
        "api_calls": api_calls,
        "duration_seconds": duration,
        "model": _model if isinstance(_model, str) else None,
        "exit_reason": exit_reason,
        # A budget-exhausted child still returns a summary (status stays
        # "completed"), so the parent needs this explicit flag.
        "truncated": exit_reason == "max_iterations",
        "tokens": {
            "input": (
                _input_tokens if isinstance(_input_tokens, (int, float)) else 0
            ),
            "output": (
                _output_tokens if isinstance(_output_tokens, (int, float)) else 0
            ),
        },
        "tool_trace": tool_trace,
        # Captured before the finally block calls child.close() so the
        # parent thread can fire subagent_stop with the correct role.
        # Stripped before the dict is serialised back to the model.
        "_child_role": getattr(child, "_delegate_role", None),
        # Folded into the parent's session cost by the aggregator; stripped
        # before serialisation back to the model.
        "_child_cost_usd": (
            float(getattr(child, "session_estimated_cost_usd", 0.0) or 0.0)
            if isinstance(
                getattr(child, "session_estimated_cost_usd", 0.0),
                (int, float),
            )
            else 0.0
        ),
    }
    # Model-visible per-delegation spend (unlike _child_cost_usd above).
    entry["cost_usd"] = round(entry["_child_cost_usd"], 6)
    _cost_status = getattr(child, "session_cost_status", None)
    entry["cost_status"] = (
        _cost_status if isinstance(_cost_status, str) and _cost_status
        else "unknown"
    )
    if status == "failed":
        if _schema_valid is False and summary and not _empty_sentinel:
            # The child DID respond; name the contract violation instead of
            # the generic "no response" error.
            entry["error"] = (
                "Final answer does not satisfy the declared "
                "output_schema (after 1 retry)."
                if _schema_retries
                else "Final answer does not satisfy the declared "
                "output_schema."
            )
        else:
            entry["error"] = result.get(
                "error", "Subagent did not produce a response."
            )
        # Classified reason from the child loop (e.g. "rate_limit",
        # "billing", "server_error") — lets the parent distinguish a
        # quota wall from a real task error without parsing prose.
        _failure_reason = result.get("failure_reason")
        if isinstance(_failure_reason, str) and _failure_reason:
            entry["failure_reason"] = _failure_reason

    # T1-24: schema-validation outcome — emitted ONLY when a schema was
    # requested, so legacy (schema-less) payloads keep their exact shape.
    if isinstance(_output_schema, dict):
        entry["schema_valid"] = bool(_schema_valid)
        if _schema_retries:
            entry["schema_retries"] = _schema_retries
        if not _schema_valid and _schema_errors:
            entry["schema_errors"] = _schema_errors

    # A steer queued after the final assistant turn had no tool batch to land
    # in; the finalizer hands it back as "pending_steer". Name it so the parent
    # sees it was MISSED rather than silently absorbed.
    _missed_steer = result.get("pending_steer")
    if isinstance(_missed_steer, str) and _missed_steer.strip():
        entry["missed_steer"] = _missed_steer
        _miss_note = (
            "[steer did not land — the subagent finished before it could "
            f"be delivered: {_missed_steer}]"
        )
        entry["summary"] = f"{summary}\n\n{_miss_note}" if summary else _miss_note
    return entry


def _append_sibling_write_reminder(entry: Dict[str, Any], ws: _ChildWorkspace) -> None:
    """Warn the parent when this child wrote files the parent had already read.

    Checks writes by ANY non-parent task_id (not just this child's) so nested
    orchestrator→worker chains are covered too.
    """
    parent_task_id, wall_start, parent_reads_snapshot = (
        ws.parent_task_id, ws.wall_start, ws.parent_reads_snapshot
    )
    try:
        if parent_task_id and parent_reads_snapshot:
            sibling_writes = file_state.writes_since(
                parent_task_id, wall_start, parent_reads_snapshot
            )
            if sibling_writes:
                mod_paths = sorted(
                    {p for paths in sibling_writes.values() for p in paths}
                )
                if mod_paths:
                    reminder = (
                        "\n\n[NOTE: subagent modified files the parent "
                        "previously read — re-read before editing: "
                        + ", ".join(mod_paths[:8])
                        + (
                            f" (+{len(mod_paths) - 8} more)"
                            if len(mod_paths) > 8
                            else ""
                        )
                        + "]"
                    )
                    if entry.get("summary"):
                        entry["summary"] = entry["summary"] + reminder
                    else:
                        entry["stale_paths"] = mod_paths
    except Exception:
        logger.debug("file_state sibling-write check failed", exc_info=True)


def _emit_child_complete(
    child: Any,
    result: Dict[str, Any],
    entry: Dict[str, Any],
    ws: _ChildWorkspace,
    duration: float,
    child_progress_cb: Any,
) -> None:
    """Fire ``subagent.complete`` with the per-branch observability payload.

    Tokens, cost, files touched and a tool-output tail feed the TUI overlay;
    every field is optional and degrades gracefully on the client.
    """
    if not child_progress_cb:
        return
    summary, status, api_calls = entry["summary"], entry["status"], entry["api_calls"]
    _input_tokens = getattr(child, "session_prompt_tokens", 0)
    _output_tokens = getattr(child, "session_completion_tokens", 0)
    child_task_id, wall_start = ws.child_task_id, ws.wall_start
    _cost_usd = getattr(child, "session_estimated_cost_usd", None)
    _reasoning_tokens = getattr(child, "session_reasoning_tokens", 0)
    try:
        _files_read = list(file_state.known_reads(child_task_id))[:40]
    except Exception:
        _files_read = []
    try:
        _files_written_map = file_state.writes_since(
            "", wall_start, []
        )  # all writes since wall_start
    except Exception:
        _files_written_map = {}
    _files_written = sorted(
        {
            p
            for tid, paths in _files_written_map.items()
            if tid == child_task_id
            for p in paths
        }
    )[:40]

    _output_tail = _extract_output_tail(result, max_entries=8, max_chars=600)

    complete_kwargs: Dict[str, Any] = {
        "preview": summary[:160] if summary else entry.get("error", ""),
        "status": status,
        "duration_seconds": duration,
        "summary": summary[:500] if summary else entry.get("error", ""),
        "input_tokens": (
            int(_input_tokens) if isinstance(_input_tokens, (int, float)) else 0
        ),
        "output_tokens": (
            int(_output_tokens) if isinstance(_output_tokens, (int, float)) else 0
        ),
        "reasoning_tokens": (
            int(_reasoning_tokens)
            if isinstance(_reasoning_tokens, (int, float))
            else 0
        ),
        "api_calls": int(api_calls) if isinstance(api_calls, (int, float)) else 0,
        "files_read": _files_read,
        "files_written": _files_written,
        "output_tail": _output_tail,
    }
    if _cost_usd is not None:
        try:
            complete_kwargs["cost_usd"] = float(_cost_usd)
        except (TypeError, ValueError):
            pass

    if child_progress_cb:
        try:
            child_progress_cb("subagent.complete", **complete_kwargs)
        except Exception as e:
            logger.debug("Progress callback completion failed: %s", e)


def _cleanup_child_run(
    child: Any,
    parent_agent: Any,
    *,
    subagent_id: Optional[str],
    heartbeat: tuple,
    child_pool: Any,
    leased_cred_id: Any,
    close_deferred: bool,
) -> None:
    """Finally-path teardown for one child run (idempotent, never raises).

    Order matters: stop heartbeat → drop registry entry → release credential
    lease → restore the parent's process-global tool names → detach from the
    parent's interrupt list → close the child (unless a timed-out worker still
    owns it) → pop the child's Relay scope if no turn is active.
    """
    _heartbeat_stop, _heartbeat_thread = heartbeat
    _subagent_id = subagent_id
    _child_close_deferred = close_deferred
    _heartbeat_stop.set()
    if _heartbeat_thread.ident is not None:
        _heartbeat_thread.join(timeout=5)

    # Drop the TUI-facing registry entry.  Safe to call even if the
    # child was never registered (e.g. ID missing on test doubles).
    if _subagent_id:
        _unregister_subagent(_subagent_id, agent=child)

    if child_pool is not None and leased_cred_id is not None:
        try:
            child_pool.release_lease(leased_cred_id)
        except Exception as exc:
            logger.debug("Failed to release credential lease: %s", exc)

    # Restore the parent's tool names so the process-global is correct
    # for any subsequent execute_code calls or other consumers.
    import model_tools

    saved_tool_names = getattr(child, "_delegate_saved_tool_names", None)
    if isinstance(saved_tool_names, list):
        model_tools._last_resolved_tool_names = list(saved_tool_names)

    # Remove child from active tracking

    # Unregister child from interrupt propagation
    if hasattr(parent_agent, "_active_children"):
        try:
            lock = getattr(parent_agent, "_active_children_lock", None)
            if lock:
                with lock:
                    parent_agent._active_children.remove(child)
            else:
                parent_agent._active_children.remove(child)
        except (ValueError, UnboundLocalError) as e:
            logger.debug("Could not remove child from active_children: %s", e)

    # Close tool resources (terminal sandboxes, browser daemons,
    # background processes, httpx clients) so subagent subprocesses
    # don't outlive the delegation.
    if not _child_close_deferred:
        try:
            close = getattr(child, "close", None)
            if callable(close):
                close()
        except Exception:
            logger.debug("Failed to close child agent after delegation")

    # The AIAgent turn boundary normally closes the child scope itself. This
    # fallback covers failures before that boundary starts, but must not pop
    # a scope while a timed-out child worker is still unwinding.
    try:
        from agent import relay_runtime

        runtime = relay_runtime.get_runtime(create=False)
        child_session_id = str(getattr(child, "session_id", "") or "")
        child_turn_is_active = relay_runtime.SESSION_COORDINATOR.has_active_turn(
            profile_key=relay_runtime.current_profile_key(),
            session_id=child_session_id,
        )
        if runtime is not None and child_session_id and not child_turn_is_active:
            runtime.unregister_subagent({"child_session_id": child_session_id})
    except Exception:
        logger.debug("Failed to close child Relay session after delegation")

