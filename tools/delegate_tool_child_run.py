"""Helpers for running ONE pre-built child agent: heartbeat, registry entry, workspace seeding, timeout/failure handling, result-entry assembly and cleanup.

Split out of ``tools/delegate_tool.py``; every moved name is re-imported there, so
``tools.delegate_tool.<name>`` keeps resolving (and monkeypatching) as before.
"""

from __future__ import annotations

import logging
import contextvars
import json
import threading
import time
from concurrent.futures import TimeoutError as FuturesTimeoutError
from typing import Any, Dict, List, Optional
from agent.interrupt_compat import request_hard_interrupt
from dataclasses import dataclass
from tools import file_state
from tools.delegate_tool_progress import _safe_progress
from tools.delegate_tool_registry import (
    _capture_gateway_steer_authority, _close_subagent_steering, _register_subagent, _unregister_subagent,
)
from tools.delegate_tool_results import (
    _extract_output_tail, _looks_like_error_output, _stringify_tool_content, _summarize_tool_arguments,
)

# Log-record parity with the origin module.
logger = logging.getLogger("tools.delegate_tool")


def _num(value: Any, default: int = 0) -> int:
    """int() for counters that may be mocks/None on test doubles."""
    return int(value) if isinstance(value, (int, float)) else default


def _fabricated_entry(idx: int, status: str, error: str, child: Any, duration: float = 0) -> Dict[str, Any]:
    """Result entry for a child that raised, never finished, or was abandoned."""
    return {
        "task_index": idx,
        "status": status,
        "summary": None,
        "error": error,
        "api_calls": 0,
        "duration_seconds": duration,
        "_child_role": getattr(child, "_delegate_role", None),
    }


def _append_missed_steer(entry: Dict[str, Any], late_steer: Optional[str]) -> None:
    """Record steer text that won the race with the child's failure/timeout."""
    if late_steer:
        entry["missed_steer"] = late_steer
        entry["error"] += (
            " [steer did not land before the subagent stopped: "
            f"{late_steer}]"
        )


def _close_child(child: Any, log_message: str) -> None:
    """Best-effort ``child.close()`` (tool sandboxes, browser daemons, httpx clients)."""
    try:
        close = getattr(child, "close", None)
        if callable(close):
            close()
    except Exception:
        logger.debug(log_message, exc_info=True)


def _attach_child(parent_agent: Any, child: Any) -> None:
    """Register the child for parent interrupt propagation."""
    if not hasattr(parent_agent, "_active_children"):
        return
    lock = getattr(parent_agent, "_active_children_lock", None)
    if lock:
        with lock:
            parent_agent._active_children.append(child)
    else:
        parent_agent._active_children.append(child)


def _detach_child(parent_agent: Any, child: Any) -> None:
    """Remove the child from parent interrupt propagation (no-op if absent)."""
    if not hasattr(parent_agent, "_active_children"):
        return
    try:
        lock = getattr(parent_agent, "_active_children_lock", None)
        if lock:
            with lock:
                parent_agent._active_children.remove(child)
        else:
            parent_agent._active_children.remove(child)
    except (ValueError, UnboundLocalError) as e:
        logger.debug("Could not remove child from active_children: %s", e)


def _signal_child_stop(child: Any, *reason: str) -> None:
    """Cooperative interrupt so the child's worker thread can exit cleanly."""
    try:
        if child is not None and not request_hard_interrupt(child, *reason) and hasattr(child, "_interrupt_requested"):
            child._interrupt_requested = True
    except Exception:
        pass


def _format_thread_stack(frame: Any, indent: str) -> List[str]:
    import traceback as _traceback

    return [
        f"{indent}{sub}"
        for frame_line in _traceback.format_stack(frame)
        for sub in frame_line.rstrip().split("\n")
    ]


def _dump_subagent_timeout_diagnostic(
    *,
    child: Any,
    task_index: int,
    timeout_seconds: float,
    duration_seconds: float,
    worker_thread: Optional[threading.Thread],
    goal: str,
) -> Optional[str]:
    """Write a structured diagnostic for a subagent that timed out before any
    API call (users hit "timed out with no response" and 0 API calls with no way
    to inspect it). Lands under ``~/.hermes/logs/subagent-timeout-<sid>-<ts>.log``
    with the child's config, prompt/schema sizes, activity snapshot and the
    worker thread's stack. Returns the path, or None on failure.
    """
    try:
        from hermes_constants import get_hermes_home
        import datetime as _dt
        import sys as _sys
        import threading as _threading

        logs_dir = get_hermes_home() / "logs"
        try:
            logs_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            return None

        subagent_id = getattr(child, "_subagent_id", None) or f"idx{task_index}"
        dump_path = logs_dir / f"subagent-timeout-{subagent_id}-{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        _goal_preview = (goal or "").strip()
        if len(_goal_preview) > 1000:
            _goal_preview = _goal_preview[:1000] + " ...[truncated]"
        lines: List[str] = [
            "# Subagent timeout diagnostic — issue #14726",
            f"# Generated: {_dt.datetime.now().isoformat()}",
            "",
            "## Timeout",
            f"  task_index:        {task_index}",
            f"  subagent_id:       {subagent_id}",
            f"  configured_timeout: {timeout_seconds}s",
            f"  actual_duration:   {duration_seconds:.2f}s",
            "",
            "## Goal",
            _goal_preview or "(empty)",
            "",
            "## Child config",
        ]
        _w = lines.append
        for attr in (
            "model", "provider", "api_mode", "base_url", "max_iterations",
            "quiet_mode", "skip_memory", "skip_context_files", "platform",
            "_delegate_role", "_delegate_depth",
        ):
            try:
                _w(f"  {attr}: {getattr(child, attr, None)!r}")
            except Exception:
                _w(f"  {attr}: <unreadable>")
        _w("")

        _w("## Toolsets")
        _w(f"  enabled_toolsets:  {getattr(child, 'enabled_toolsets', None)!r}")
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
            sys_prompt = getattr(child, "ephemeral_system_prompt", None) or getattr(child, "system_prompt", None) or ""
            _w(f"  system_prompt_bytes: {len(sys_prompt.encode('utf-8')) if isinstance(sys_prompt, str) else 'n/a'}")
            _w(f"  system_prompt_chars: {len(sys_prompt) if isinstance(sys_prompt, str) else 'n/a'}")
        except Exception as exc:
            _w(f"  system_prompt: <error: {exc}>")
        try:
            tools_schema = getattr(child, "tools", None)
            if tools_schema is not None:
                _w(f"  tool_schema_count: {len(tools_schema)}")
                _w(f"  tool_schema_bytes: {len(json.dumps(tools_schema, default=str).encode('utf-8'))}")
        except Exception as exc:
            _w(f"  tool_schema: <error: {exc}>")
        _w("")

        _w("## Activity summary")
        try:
            for k, v in child.get_activity_summary().items():
                _w(f"  {k}: {v!r}")
        except Exception as exc:
            _w(f"  <get_activity_summary failed: {exc}>")
        _w("")

        _w("## Worker thread stack at timeout")
        frames = _sys._current_frames()
        if worker_thread is not None and worker_thread.is_alive():
            worker_frame = frames.get(worker_thread.ident)
            lines.extend(
                _format_thread_stack(worker_frame, "  ") if worker_frame is not None
                else ["  <worker frame not available>"]
            )
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
            by_ident = {th.ident: th for th in _threading.enumerate() if th.ident}
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
                _w(f"  --- {name}{' daemon' if (th and th.daemon) else ''} ---")
                lines.extend(_format_thread_stack(frame, "    "))
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
    last_seen = {"iter": 0, "tool": None, "ts": None, "stale": 0}

    def _heartbeat_loop():
        while not _heartbeat_stop.wait(_HEARTBEAT_INTERVAL):
            touch = getattr(parent_agent, "_touch_activity", None) if parent_agent is not None else None
            if not touch:
                continue
            desc = f"delegate_task: subagent {task_index} working"
            try:
                child_summary = child.get_activity_summary()
                child_tool = child_summary.get("current_tool")
                child_iter = child_summary.get("api_call_count", 0)
                child_max = child_summary.get("max_iterations", 0)
                child_activity_ts = child_summary.get("last_activity_ts")

                # A slow model wait refreshes last_activity_ts (direct_api_call
                # heartbeat), so it never looks stale at the idle threshold.
                activity_advanced = child_activity_ts is not None and (
                    last_seen["ts"] is None or child_activity_ts > last_seen["ts"]
                )
                if child_iter > last_seen["iter"] or child_tool != last_seen["tool"] or activity_advanced:
                    last_seen["iter"], last_seen["tool"], last_seen["stale"] = child_iter, child_tool, 0
                    if child_activity_ts is not None:
                        last_seen["ts"] = child_activity_ts
                else:
                    last_seen["stale"] += 1

                stale_limit = _HEARTBEAT_STALE_CYCLES_IN_TOOL if child_tool else _HEARTBEAT_STALE_CYCLES_IDLE
                if last_seen["stale"] >= stale_limit:
                    logger.warning(
                        "Subagent %d appears stale (no progress for %d "
                        "heartbeat cycles, tool=%s) — stopping heartbeat",
                        task_index,
                        last_seen["stale"],
                        child_tool or "<none>",
                    )
                    break  # stop touching parent, let gateway timeout fire

                if child_tool:
                    desc = f"delegate_task: subagent running {child_tool} (iteration {child_iter}/{child_max})"
                elif child_summary.get("last_activity_desc", ""):
                    desc = (
                        f"delegate_task: subagent {child_summary.get('last_activity_desc', '')} "
                        f"(iteration {child_iter}/{child_max})"
                    )
            except Exception:
                pass
            try:
                touch(desc)
            except Exception:
                pass

    return _heartbeat_stop, threading.Thread(target=_heartbeat_loop, daemon=True)


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
    _subagent_id = getattr(child, "_subagent_id", None)
    if not isinstance(_subagent_id, str) or not _subagent_id:
        return None
    if owner_session_id is None:
        try:
            from gateway.session_context import get_session_env

            owner_session_id = get_session_env("HERMES_UI_SESSION_ID", "") or None
        except Exception:
            owner_session_id = None
    if owner_session_id and (owner_transport is None or owner_session_record is None):
        owner_transport, owner_session_record = _capture_gateway_steer_authority(owner_session_id)
    _raw_depth = getattr(child, "_delegate_depth", 1)
    _parent_sid = getattr(child, "_parent_subagent_id", None)
    _delegation_id = getattr(child, "_delegation_id", None)
    _model = getattr(child, "model", None)
    _register_subagent(
        {
            "subagent_id": _subagent_id,
            "parent_id": _parent_sid if isinstance(_parent_sid, str) else None,
            "depth": max(0, _raw_depth - 1) if isinstance(_raw_depth, int) else 0,
            "goal": goal,
            "delegation_id": _delegation_id if isinstance(_delegation_id, str) else None,
            "model": _model if isinstance(_model, str) else None,
            "started_at": time.time(),
            "status": "running",
            "tool_count": 0,
            "agent": child,
            # Owning conversation's durable session id (same lineage completion
            # delivery routes by), sourced from the child's stamp so it survives
            # a parent_agent rebuild between dispatch and run; used for
            # list/steer/stop ownership when the weakref chain breaks.
            "owner_agent_session_id": (
                str(getattr(child, "_parent_session_id", "") or "")
                or str(getattr(parent_agent, "session_id", "") or "")
                or None
            ),
            # Immutable live gateway/TUI session that commissioned this child.
            # Empty outside those hosts; RPC authority fails closed.
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
        info = self.info
        if info is None:
            return
        from tools import subagent_worktree

        try:
            entry_dict["worktree"] = subagent_worktree.finalize_subagent_worktree(info)
        except Exception as e:
            # State is unknown: emit the SAME flagged schema the parent expects,
            # via the shared factory so the two producers never drift.
            logger.warning("worktree finalize failed: %s", e)
            entry_dict["worktree"] = subagent_worktree.unproven_worktree_payload(info, f"finalize raised: {e}")


@dataclass
class _ChildWorkspace:
    child_task_id: str
    parent_task_id: Optional[str]
    goal: str
    wall_start: float
    parent_reads_snapshot: list


def _create_isolated_worktree(parent_agent: Any, parent_task_id: Any, subagent_id: Optional[str]):
    """Opt-in worktree isolation: own git worktree off the parent's HEAD (the
    child's terminal starts there). Git-only, local-backend-only; failures
    degrade silently to the shared workspace. Returns the worktree info or None."""
    from tools.delegate_tool import _get_worktree_isolation, _resolve_workspace_hint

    if not _get_worktree_isolation():
        return None
    try:
        from tools import subagent_worktree

        if not subagent_worktree.local_backend_active():
            logger.debug("worktree isolation skipped: non-local terminal backend")
            return None
        _parent_cwd = None
        try:
            from tools.terminal_tool import get_session_cwd as _gsc

            _parent_cwd = _gsc(parent_task_id)
        except Exception:
            pass
        return subagent_worktree.create_subagent_worktree(
            _parent_cwd or _resolve_workspace_hint(parent_agent), subagent_id=subagent_id,
        )
    except Exception as e:
        logger.debug("worktree isolation setup failed: %s", e)
        return None


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
    import uuid as _uuid

    child_task_id = subagent_id or f"subagent-{task_index}-{_uuid.uuid4().hex[:8]}"
    parent_task_id = getattr(parent_agent, "_current_task_id", None)
    # Seed the child's cwd record from the parent's: same starting directory,
    # but the child's later `cd`s stay in its own record. Per-session container
    # isolation keys containers by task_id; the child must share the PARENT's.
    try:
        from tools.terminal_tool import get_session_cwd, record_session_cwd, register_container_alias

        record_session_cwd(child_task_id, get_session_cwd(parent_task_id))
        register_container_alias(child_task_id, parent_task_id)
    except Exception as e:
        logger.debug("Child cwd seed failed: %s", e)

    _worktree_info = _create_isolated_worktree(parent_agent, parent_task_id, subagent_id)
    if _worktree_info is not None:
        try:
            from tools.terminal_tool import record_session_cwd as _rsc

            _rsc(child_task_id, _worktree_info["path"])
        except Exception as e:
            logger.debug("worktree cwd seed failed: %s", e)
        # The child's context is already built; carry the isolation contract on
        # the goal message instead (same turn, no system-prompt mutation).
        from tools.subagent_worktree import build_worktree_context_note

        goal = goal + build_worktree_context_note(_worktree_info)

    worktree.info = _worktree_info
    parent_reads_snapshot = list(file_state.known_reads(parent_task_id)) if parent_task_id else []
    return _ChildWorkspace(child_task_id, parent_task_id, goal, time.time(), parent_reads_snapshot)


@dataclass
class _ChildFailure:
    entry: Dict[str, Any]
    close_deferred: bool


def _defer_close_after_timeout(child: Any, child_future: Any) -> None:
    """Hand ``child.close()`` to a Future done-callback and drain its transports.

    The interrupt is cooperative: the worker still runs its finally path, so
    closing the child now could close SQLite under its final write — the
    done-callback is the first safe close boundary. The abandoned worker is
    usually parked in an OpenSSL read; NEVER hard-close that transport from this
    thread (cross-thread FD release under a live SSL read corrupts native
    state) — shutdown() the pooled sockets instead, which settles the read with
    EOF so the worker unwinds. One immediate sweep + one delayed re-sweep for a
    connection opened in between; a worker that still won't settle keeps its
    resources until process exit.
    """
    child_future.add_done_callback(
        lambda _done: _close_child(child, "Failed to close timed-out child after worker exit")
    )
    _drain = getattr(child, "_drain_transports_after_abandonment", None)
    if not callable(_drain):
        return

    def _drain_once(phase: str) -> None:
        try:
            _drain(reason=f"delegate_timeout_{phase}")
        except Exception:
            logger.debug("Timed-out child transport drain (%s) failed", phase, exc_info=True)

    _drain_once("immediate")

    def _drain_resweep() -> None:
        if not child_future.done():
            _drain_once("resweep")

    _resweep_timer = threading.Timer(5.0, _drain_resweep)
    _resweep_timer.daemon = True
    _resweep_timer.start()


def _lease_child_credential(child: Any) -> tuple[Any, Optional[str]]:
    """Lease a credential from the child's pool (if any) and bind it; ``(pool, lease_id)``."""
    child_pool = getattr(child, "_credential_pool", None)
    if child_pool is None:
        return None, None
    leased_cred_id = child_pool.acquire_lease()
    if leased_cred_id is not None:
        try:
            leased_entry = child_pool.current()
            if leased_entry is not None and hasattr(child, "_swap_credential"):
                child._swap_credential(leased_entry)
        except Exception as exc:
            logger.debug("Failed to bind child to leased credential: %s", exc)
    return child_pool, leased_cred_id


def _make_text_relay(child_progress_cb: Any):
    """Stream callback forwarding the child's reply text up the progress relay so
    gateway watch windows mirror it live (subagent.text → message.delta). Inert
    under CLI/TUI: their progress handlers ignore non-tool events."""

    def _relay_child_text(delta: str) -> None:
        if delta:
            _safe_progress(child_progress_cb, "subagent.text", preview=delta)

    return _relay_child_text


def _await_child(
    child: Any,
    goal: str,
    ws: "_ChildWorkspace",
    relay_child_text: Any,
    *,
    task_index: int,
    subagent_id: Optional[str],
    child_start: float,
    child_progress_cb: Any,
    worktree: _WorktreeReporter,
) -> tuple[Optional[Dict[str, Any]], Optional[_ChildFailure]]:
    """Run the child's conversation on a daemon worker and wait for it.

    Returns ``(result, None)`` or ``(None, failure)`` on timeout/exception.
    The hard timeout is off by default (``result(timeout=None)`` blocks until
    the child finishes; stuck-child protection is the heartbeat). The worker is
    a daemon: a timed-out child is abandoned and a stdlib non-daemon worker
    would block interpreter exit at atexit-join time. The worker installs a
    non-interactive approval callback so dangerous-command prompts never fall
    back to ``input()`` and deadlock the parent TUI (deny vs approve follows
    delegation.subagent_auto_approve).
    """
    from tools.delegate_tool import (
        _get_child_timeout, _get_subagent_approval_callback, _set_subagent_approval_cb,
    )
    from tools.daemon_pool import DaemonThreadPoolExecutor

    child_timeout = _get_child_timeout()
    executor = DaemonThreadPoolExecutor(
        max_workers=1, initializer=_set_subagent_approval_cb, initargs=(_get_subagent_approval_callback(),),
    )
    # Worker thread handle so the timeout diagnostic can dump its stack.
    worker_thread_holder: Dict[str, Optional[threading.Thread]] = {"t": None}

    def _run_with_thread_capture():
        worker_thread_holder["t"] = threading.current_thread()
        from agent.delegation_context import delegated_child_context

        with delegated_child_context(str(getattr(child, "session_id", "") or "")):
            return child.run_conversation(
                user_message=goal, task_id=ws.child_task_id, stream_callback=relay_child_text,
            )

    future = executor.submit(contextvars.copy_context().run, _run_with_thread_capture)
    try:
        return future.result(timeout=child_timeout), None
    except Exception as exc:
        return None, _handle_child_wait_failure(
            exc,
            child=child,
            task_index=task_index,
            goal=goal,
            subagent_id=subagent_id,
            child_future=future,
            child_timeout=child_timeout,
            child_start=child_start,
            child_progress_cb=child_progress_cb,
            worker_thread_holder=worker_thread_holder,
            worktree=worktree,
        )
    finally:
        # Shut down without waiting — a child stuck on blocking I/O would hang wait=True forever.
        executor.shutdown(wait=False)


def _merge_late_steer(result: Dict[str, Any], subagent_id: Optional[str], child: Any) -> None:
    """Linearization boundary for registry steering: from here the child cannot
    consume another steer. Closing under the registry lock either rejects a
    concurrent caller or drains every accepted exact text into the result
    before callbacks/result assembly run."""
    late = _close_subagent_steering(subagent_id, child) if subagent_id else None
    if late:
        existing = result.get("pending_steer")
        result["pending_steer"] = f"{existing}\n{late}" if isinstance(existing, str) and existing else late


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
    # No consumer boundary remains once this owner stops waiting for the
    # child. Close acceptance before any completion callback and retain steer
    # text that won the race with this failure/timeout.
    _late_pending_steer = _close_subagent_steering(subagent_id, child) if subagent_id else None
    _signal_child_stop(child)

    is_timeout = isinstance(exc, (FuturesTimeoutError, TimeoutError))
    duration = round(time.monotonic() - child_start, 2)
    logger.warning(
        "Subagent %d %s after %.1fs",
        task_index,
        "timed out" if is_timeout else f"raised {type(exc).__name__}",
        duration,
    )

    # A timeout BEFORE any API call is a black box without a diagnostic dump.
    diagnostic_path: Optional[str] = None
    child_api_calls = 0
    try:
        child_api_calls = int(child.get_activity_summary().get("api_call_count", 0) or 0)
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
            worker_thread=worker_thread_holder.get("t"),
            goal=goal,
        )
        if diagnostic_path:
            logger.warning(
                "Subagent %d 0-API-call timeout — diagnostic written to %s",
                task_index,
                diagnostic_path,
            )

    status = "timeout" if is_timeout else "error"
    _safe_progress(
        child_progress_cb,
        "subagent.complete",
        preview=f"Timed out after {duration}s" if is_timeout else str(exc),
        status=status,
        duration_seconds=duration,
        summary="",
    )

    if not is_timeout:
        _err = str(exc)
    elif child_api_calls == 0:
        _err = (
            f"Subagent timed out after {child_timeout}s without "
            f"making any API call — the child never reached its "
            f"first LLM request (prompt construction, credential "
            f"resolution, or transport may be stuck)."
        )
    else:
        _err = (
            f"Subagent timed out after {child_timeout}s with "
            f"{child_api_calls} API call(s) completed — likely "
            f"stuck on a slow API call, tool call, or unresponsive "
            f"network request."
        )
    if is_timeout and diagnostic_path:
        _err += f" Diagnostic: {diagnostic_path}"

    _error_entry = {
        "task_index": task_index,
        "status": status,
        "summary": None,
        "error": _err,
        "exit_reason": status,
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
    _append_missed_steer(_error_entry, _late_pending_steer)
    worktree.attach(_error_entry)
    close_deferred = is_timeout and not child_future.done()
    if close_deferred:
        _defer_close_after_timeout(child, child_future)
    return _ChildFailure(_error_entry, close_deferred)


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
    _output_schema = getattr(child, "_delegate_output_schema", None)
    if not isinstance(_output_schema, dict):
        return _SchemaOutcome(_output_schema, None, [], 0)
    from tools.delegation_output_schema import build_retry_message, validate_output

    _first_text = result.get("final_response") or ""
    _schema_valid, _schema_errors = validate_output(_first_text, _output_schema)
    if _schema_valid or not _first_text.strip() or result.get("interrupted", False):
        return _SchemaOutcome(_output_schema, _schema_valid, _schema_errors, 0)

    # Exactly one retry turn, carrying the validation errors verbatim (no
    # schema re-paste — the child already holds the contract in its context).
    _retry_result = None
    try:
        _retry_result = child.run_conversation(
            user_message=build_retry_message(_schema_errors),
            task_id=child_task_id,
            stream_callback=relay_child_text,
        )
    except Exception as _retry_exc:
        logger.warning("Subagent %d schema-retry turn failed: %s", task_index, _retry_exc)
    if isinstance(_retry_result, dict):
        _retry_text = _retry_result.get("final_response") or ""
        if _retry_text.strip():
            result["final_response"] = _retry_text
        try:
            result["api_calls"] = int(result.get("api_calls", 0) or 0) + int(_retry_result.get("api_calls", 0) or 0)
        except (TypeError, ValueError):
            pass
        _retry_messages = _retry_result.get("messages")
        if isinstance(_retry_messages, list) and isinstance(result.get("messages"), list):
            result["messages"] = result["messages"] + _retry_messages
        _schema_valid, _schema_errors = validate_output(_retry_text, _output_schema)
    return _SchemaOutcome(_output_schema, _schema_valid, _schema_errors, 1)


def _build_tool_trace(messages: Any) -> list[Dict[str, Any]]:
    """Tool trace from the child's conversation messages, pairing parallel
    tool calls with their results by tool_call_id."""
    tool_trace: list[Dict[str, Any]] = []
    trace_by_id: Dict[str, Dict[str, Any]] = {}
    if not isinstance(messages, list):
        return tool_trace
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
                if tc.get("id"):
                    trace_by_id[tc["id"]] = entry_t
        elif msg.get("role") == "tool":
            content = _stringify_tool_content(msg.get("content", ""))
            result_meta = {
                "result_bytes": len(content),
                "status": "error" if _looks_like_error_output(content) else "ok",
            }
            tc_id = msg.get("tool_call_id")
            target = trace_by_id.get(tc_id) if tc_id else None
            if target is not None:
                target.update(result_meta)
            elif tool_trace:
                tool_trace[-1].update(result_meta)  # no tool_call_id: pair with the latest call
    return tool_trace


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
    summary = result.get("final_response") or ""
    interrupted = result.get("interrupted", False)
    structured_failure = bool(result.get("failed") or result.get("error"))
    # "(empty)" is run_agent's give-up sentinel after repeated empty LLM
    # responses (usually a transport bug) — a failure, not a success.
    _empty_sentinel = summary.strip() == "(empty)"
    usable_summary = bool(summary) and not _empty_sentinel

    if interrupted:
        status, exit_reason = "interrupted", "interrupted"
    elif structured_failure:
        # The loop returns the error text as final_response, which would
        # otherwise read as "completed". Never report a provider rejection as
        # "max_iterations" — that is only truthful for real budget exhaustion.
        status, exit_reason = "failed", "error"
    else:
        # exit_reason ("completed" vs "max_iterations") tells the parent HOW
        # the task ended; completed=False with no failure = budget exhaustion.
        exit_reason = "completed" if result.get("completed", False) else "max_iterations"
        # A declared schema still violated after the bounded retry makes the
        # summary unusable under the contract, so status must not say completed
        # (orchestrators reading only status/icon would accept an empty verdict).
        status = "completed" if schema.valid is not False and usable_summary else "failed"

    _model = getattr(child, "model", None)
    _cost = getattr(child, "session_estimated_cost_usd", 0.0)
    _cost_status = getattr(child, "session_cost_status", None)
    # Result entry contract: see the _run_single_child docstring.
    entry: Dict[str, Any] = {
        "task_index": task_index,
        "status": status,
        "summary": summary,
        "api_calls": result.get("api_calls", 0),
        "duration_seconds": duration,
        "model": _model if isinstance(_model, str) else None,
        "exit_reason": exit_reason,
        # A budget-exhausted child still returns a summary (status stays
        # "completed"), so the parent needs this explicit flag.
        "truncated": exit_reason == "max_iterations",
        "tokens": {
            "input": _num(getattr(child, "session_prompt_tokens", 0)),
            "output": _num(getattr(child, "session_completion_tokens", 0)),
        },
        "tool_trace": _build_tool_trace(result.get("messages") or []),
        # Captured before the finally block calls child.close() so the parent
        # thread can fire subagent_stop with the correct role; stripped before
        # the dict is serialised back to the model (as is _child_cost_usd,
        # folded into the parent's session cost by the aggregator).
        "_child_role": getattr(child, "_delegate_role", None),
        "_child_cost_usd": float(_cost or 0.0) if isinstance(_cost, (int, float)) else 0.0,
    }
    # Model-visible per-delegation spend (unlike _child_cost_usd above).
    entry["cost_usd"] = round(entry["_child_cost_usd"], 6)
    entry["cost_status"] = _cost_status if isinstance(_cost_status, str) and _cost_status else "unknown"
    if status == "failed":
        if schema.valid is False and usable_summary:
            # The child DID respond; name the contract violation instead of
            # the generic "no response" error.
            entry["error"] = (
                "Final answer does not satisfy the declared output_schema"
                + (" (after 1 retry)." if schema.retries else ".")
            )
        else:
            entry["error"] = result.get("error", "Subagent did not produce a response.")
        # Classified reason from the child loop (e.g. "rate_limit", "billing")
        # lets the parent tell a quota wall from a task error without parsing prose.
        _failure_reason = result.get("failure_reason")
        if isinstance(_failure_reason, str) and _failure_reason:
            entry["failure_reason"] = _failure_reason

    # Schema-validation outcome — emitted ONLY when a schema was requested, so
    # legacy (schema-less) payloads keep their exact shape.
    if isinstance(schema.schema, dict):
        entry["schema_valid"] = bool(schema.valid)
        if schema.retries:
            entry["schema_retries"] = schema.retries
        if not schema.valid and schema.errors:
            entry["schema_errors"] = schema.errors

    # A steer queued after the final assistant turn had no tool batch to land
    # in; name it so the parent sees it was MISSED rather than silently absorbed.
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
    if not (ws.parent_task_id and ws.parent_reads_snapshot):
        return
    try:
        sibling_writes = file_state.writes_since(ws.parent_task_id, ws.wall_start, ws.parent_reads_snapshot)
        mod_paths = sorted({p for paths in sibling_writes.values() for p in paths}) if sibling_writes else []
        if not mod_paths:
            return
        reminder = (
            "\n\n[NOTE: subagent modified files the parent "
            "previously read — re-read before editing: "
            + ", ".join(mod_paths[:8])
            + (f" (+{len(mod_paths) - 8} more)" if len(mod_paths) > 8 else "")
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
    summary, status = entry["summary"], entry["status"]
    try:
        _files_read = list(file_state.known_reads(ws.child_task_id))[:40]
    except Exception:
        _files_read = []
    try:
        _files_written_map = file_state.writes_since("", ws.wall_start, [])  # all writes since wall_start
    except Exception:
        _files_written_map = {}
    _files_written = sorted(
        {p for tid, paths in _files_written_map.items() if tid == ws.child_task_id for p in paths}
    )[:40]

    complete_kwargs: Dict[str, Any] = {
        "preview": summary[:160] if summary else entry.get("error", ""),
        "status": status,
        "duration_seconds": duration,
        "summary": summary[:500] if summary else entry.get("error", ""),
        "input_tokens": _num(getattr(child, "session_prompt_tokens", 0)),
        "output_tokens": _num(getattr(child, "session_completion_tokens", 0)),
        "reasoning_tokens": _num(getattr(child, "session_reasoning_tokens", 0)),
        "api_calls": _num(entry["api_calls"]),
        "files_read": _files_read,
        "files_written": _files_written,
        "output_tail": _extract_output_tail(result, max_entries=8, max_chars=600),
    }
    _cost_usd = getattr(child, "session_estimated_cost_usd", None)
    if _cost_usd is not None:
        try:
            complete_kwargs["cost_usd"] = float(_cost_usd)
        except (TypeError, ValueError):
            pass
    _safe_progress(child_progress_cb, "subagent.complete", **complete_kwargs)


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
    _heartbeat_stop.set()
    if _heartbeat_thread.ident is not None:
        _heartbeat_thread.join(timeout=5)

    # Safe even if the child was never registered (ID missing on test doubles).
    if subagent_id:
        _unregister_subagent(subagent_id, agent=child)

    if child_pool is not None and leased_cred_id is not None:
        try:
            child_pool.release_lease(leased_cred_id)
        except Exception as exc:
            logger.debug("Failed to release credential lease: %s", exc)

    # Restore the parent's tool names so the process-global is correct for
    # any subsequent execute_code calls or other consumers.
    import model_tools

    saved_tool_names = getattr(child, "_delegate_saved_tool_names", None)
    if isinstance(saved_tool_names, list):
        model_tools._last_resolved_tool_names = list(saved_tool_names)

    _detach_child(parent_agent, child)

    # Close tool resources (terminal sandboxes, browser daemons, background
    # processes, httpx clients) so subagent subprocesses don't outlive the delegation.
    if not close_deferred:
        _close_child(child, "Failed to close child agent after delegation")

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
