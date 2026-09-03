"""Batch execution + background dispatch for delegate_task.

Split out of ``tools/delegate_tool.py`` (zero-back-ref regions of ``delegate_task``);
``delegate_task`` stays the orchestrator and calls these with its locals.
"""

from __future__ import annotations

import contextvars
import json
import logging
from concurrent.futures import FIRST_COMPLETED, wait as _cf_wait
from typing import Any, Dict, List, Optional

from tools.delegate_tool_child_run import _detach_child, _fabricated_entry, _signal_child_stop
from tools.delegate_tool_progress import (
    SUBAGENT_FAILURE_STATUSES,
    _clean_error_text,
    _print_completion_line,
    format_batch_tag,
)

# Log-record parity with the origin module.
logger = logging.getLogger("tools.delegate_tool")

def _future_entry(future: Any, idx: int, child: Any) -> Dict[str, Any]:
    """The finished Future's entry, or a fabricated error entry if it raised."""
    try:
        return future.result()
    except Exception as exc:
        return _fabricated_entry(idx, "error", str(exc), child)

def _report_child_done(parent_agent, spinner_ref, entry, tag, task_labels, n_tasks, remaining) -> None:
    """Print one completion line for a finished child and refresh the spinner text."""
    idx = entry["task_index"]
    label = task_labels[idx] if idx < len(task_labels) else f"Task {idx}"
    status = entry.get("status", "?")
    _slot = f"{tag} · {idx+1}/{n_tasks}" if tag else f"{idx+1}/{n_tasks}"
    completion_line = f"{'✓' if status == 'completed' else '✗'} [{_slot}] {label}  ({entry.get('duration_seconds', 0)}s)"
    # Failed/errored/timed-out children: say WHY on the same line — a bare ✗
    # reads as "silently dropped".
    if status in SUBAGENT_FAILURE_STATUSES:
        _err_line = _clean_error_text(entry.get("error"), max_chars=120)
        if _err_line:
            completion_line += f" — {_err_line}"
    _print_completion_line(parent_agent, spinner_ref, completion_line)
    if spinner_ref and remaining > 0:
        try:
            spinner_ref.update_text(
                f"🔀 {'[' + tag + '] ' if tag else ''}{remaining} task{'s' if remaining != 1 else ''} remaining"
            )
        except Exception as e:
            logger.debug("Spinner update_text failed: %s", e)

def _run_children_parallel(
    children: List[tuple],
    results: list,
    *,
    parent_agent: Any,
    n_tasks: int,
    max_children: int,
    task_labels: List[str],
    live_deleg_id: Optional[str],
    honor_parent_interrupt: bool,
    origin_ui_session_id: str,
    origin_owner_transport: Any,
    origin_owner_session_record: Any,
) -> None:
    """Run a batch of built children in parallel, appending entries to ``results``.

    Polls futures with a short ``wait()`` timeout instead of ``as_completed()``
    so a wedged child cannot block the parent forever after an interrupt;
    on parent interrupt the still-pending children are reported as
    ``interrupted`` and abandoned. Prints one completion line per child.
    ``results`` ends sorted by task_index.
    """
    from tools.delegate_tool import _run_single_child

    completed_count = 0
    spinner_ref = getattr(parent_agent, "_delegate_spinner", None)
    _tag = format_batch_tag(live_deleg_id)
    # Fabricated entries for still-pending futures carry the correct _delegate_role.
    _child_by_index = {i: child for (i, _, child) in children}

    # Daemon workers (tools.daemon_pool): the `with` block still joins normally,
    # but if the parent is interrupted while a child is wedged, the abandoned
    # worker must not block interpreter exit.
    from tools.daemon_pool import DaemonThreadPoolExecutor
    with DaemonThreadPoolExecutor(max_workers=max_children) as executor:
        futures = {}
        for i, t, child in children:
            future = executor.submit(
                contextvars.copy_context().run,
                _run_single_child,
                task_index=i,
                goal=t["goal"],
                child=child,
                parent_agent=parent_agent,
                owner_session_id=origin_ui_session_id or None,
                owner_transport=origin_owner_transport,
                owner_session_record=origin_owner_session_record,
            )
            futures[future] = i

        pending = set(futures.keys())
        while pending:
            if (honor_parent_interrupt and getattr(parent_agent, "_interrupt_requested", False) is True):
                # Parent interrupted — collect whatever finished and abandon the
                # rest (children already got the interrupt signal).
                for f in pending:
                    idx = futures[f]
                    if f.done():
                        entry = _future_entry(f, idx, _child_by_index.get(idx))
                    else:
                        entry = _fabricated_entry(
                            idx,
                            "interrupted",
                            "Parent agent interrupted — child did not finish in time",
                            _child_by_index.get(idx),
                        )
                    results.append(entry)
                    completed_count += 1
                break

            done, pending = _cf_wait(pending, timeout=0.5, return_when=FIRST_COMPLETED)
            for future in done:
                entry = _future_entry(future, futures[future], _child_by_index.get(futures[future]))
                results.append(entry)
                completed_count += 1

                _report_child_done(parent_agent, spinner_ref, entry, _tag, task_labels, n_tasks, n_tasks - completed_count)

    # Sort by task_index so results match input order
    results.sort(key=lambda r: r["task_index"])

_SYNC_FALLBACK_NOTES = {
    "no_async": (
        "background=true is not available in this session — it cannot "
        "receive a detached subagent result after the turn ends (a "
        "one-shot runner such as `hermes -z`, a cron job, a Kanban "
        "worker, or a stateless HTTP endpoint). The subagent(s) ran "
        "SYNCHRONOUSLY and the result is included above."
    ),
    "at_capacity": (
        "The background delegation pool was at capacity "
        "(delegation.max_concurrent_children), so the subagent(s) ran "
        "SYNCHRONOUSLY and the result is included above. Raise "
        "delegation.max_concurrent_children in config.yaml to allow "
        "more concurrent background delegations."
    ),
}

def _run_sync_with_note(execute_and_aggregate: Any, reason: str) -> str:
    """Inline fallback: run the batch now and explain why it was not detached."""
    result = execute_and_aggregate()
    if isinstance(result, dict):
        result["note"] = _SYNC_FALLBACK_NOTES[reason]
    return json.dumps(result, ensure_ascii=False)

def _resolve_async_wake_sid(origin_wake_sid: str) -> Optional[str]:
    """Wake target for a detached batch, or None to force synchronous execution.

    Finite sessions (stateless HTTP requests, one-shot Kanban workers) cannot
    route a detached result back after their turn/process ends. But if a raw
    session id is bound (the API server always binds one), gateway.wake can
    still reach the session by self-POSTing /v1/chat/completions with that id,
    so only fall back to sync when there is truly no session id to wake. Uses
    the origin captured BEFORE child construction — HERMES_SESSION_ID here
    would be the subagent's internal id.
    """
    try:
        from gateway.session_context import async_delivery_supported
        if async_delivery_supported():
            return ""
    except Exception:
        return ""
    if origin_wake_sid:
        logger.info(
            "delegate_task: async delivery unsupported on this "
            "session, but a session id is bound (%s) — dispatching "
            "in the background and waking the session via self-post "
            "when it completes instead of forcing synchronous "
            "execution.",
            origin_wake_sid,
        )
        return origin_wake_sid
    return None

def _resolve_async_session_key(parent_agent: Any, origin_ui_session_id: str) -> tuple[str, str]:
    """``(session_key, origin_ui_session_id)`` the async registry routes completions by.

    Desktop/TUI: the routable key is the durable AIAgent.session_id — compression
    can rotate it mid-turn before the TUI-side dict is re-anchored, and a stale
    approval-context key would orphan the completion. Gateway chats keep the
    platform conversation key (agent:main:...). The CLI has no bound approval
    contextvar and no HERMES_SESSION_KEY, so the key resolves empty; its drain
    is a positive-ownership filter keyed on the durable session_id, so an empty
    key would fail closed — stamp the parent's durable id.
    """
    from tools.approval import get_current_session_key

    session_key = get_current_session_key(default="")
    agent_session_id = str(getattr(parent_agent, "session_id", "") or "")
    try:
        from gateway.session_context import get_session_env

        source = get_session_env("HERMES_SESSION_SOURCE", "")
        # Refresh from the task-local source when available, else retain the
        # immutable value captured before child construction.
        origin_ui_session_id = get_session_env("HERMES_UI_SESSION_ID", "") or origin_ui_session_id
        if source == "tui" and agent_session_id:
            session_key = agent_session_id
    except Exception:
        pass
    if not session_key and agent_session_id:
        session_key = agent_session_id
    return session_key, origin_ui_session_id

def _batch_progress_token(child_agents: List[Any]) -> tuple:
    """Progress token for the async registry's stale monitor: every child's
    (api_call_count, current_tool, last_activity_ts). last_activity_ts ticks on
    streamed chunks, tool transitions and API-call start/completion, so a child
    streaming a long response counts as alive; a fully frozen token past the
    threshold means the batch is wedged. ``in_tool`` is True while ANY child is
    inside a tool so slow tools get the higher ceiling (mirrors the sync heartbeat)."""
    parts = []
    in_tool = False
    for c in child_agents:
        try:
            summary = c.get_activity_summary()
            tool = summary.get("current_tool")
            parts.append((summary.get("api_call_count", 0), tool, summary.get("last_activity_ts")))
            in_tool = in_tool or bool(tool)
        except Exception:
            parts.append(None)
    return tuple(parts), in_tool

def _dispatched_payload(dispatch: dict, goals: List[str], child_agents: List[Any], live_paths: List[str]) -> dict:
    """Model-facing handle for an accepted background batch."""
    n = len(goals)
    payload = {
        "status": "dispatched",
        "mode": "background",
        "count": n,
        "delegation_id": dispatch["delegation_id"],
        "goals": goals,
        "note": (
            "Subagent is running in the background. You and the user can "
            "keep working; its full result re-enters the conversation as a "
            "new message when it finishes. Do not wait or poll — just "
            "continue."
            if n == 1 else
            f"{n} subagents are running in parallel in the background. You "
            f"and the user can keep working; they wait on each other and "
            f"their consolidated results re-enter the conversation as a "
            f"single message once ALL of them finish. Do not wait or poll "
            f"— just continue."
        ),
    }
    sids = [getattr(c, "_subagent_id", None) for c in child_agents]
    if any(isinstance(s, str) and s for s in sids):
        payload["subagent_ids"] = sids
        payload["control_hint"] = (
            "While a child runs you can orchestrate it live with this "
            "same tool: delegate_task(action='list') to see live "
            "children, action='steer' with subagent_id + message to "
            "redirect one, action='stop' with subagent_id to end one "
            "early."
        )
    if live_paths:
        payload["live_transcripts"] = list(live_paths)
        payload["live_transcripts_hint"] = (
            "Each subagent streams a human-readable transcript of its "
            "operations to the file listed above (append-only, one per "
            "task). Read or `tail -f` these paths at any time to watch "
            "a child work while it runs."
        )
    return payload

def _dispatch_background(
    *,
    parent_agent: Any,
    context: Optional[str],
    task_list: List[Dict[str, Any]],
    children: List[tuple],
    creds: Dict[str, Any],
    top_role: str,
    live_deleg_id: Optional[str],
    live_paths: List[str],
    origin_wake_sid: str,
    origin_ui_session_id: str,
    execute_and_aggregate: Any,
) -> str:
    """Dispatch the WHOLE batch as one async unit and return the tool result JSON.

    ``execute_and_aggregate`` joins on every child and yields ONE consolidated
    results block that re-enters the conversation as a single message when ALL
    children finish. Falls back to running it synchronously (with an explanatory
    ``note``) when the session cannot receive detached completions or the async
    pool is at capacity.
    """
    from tools.delegate_tool import _get_max_async_children
    from tools.async_delegation import dispatch_async_delegation_batch

    wake_sid = _resolve_async_wake_sid(origin_wake_sid)
    if wake_sid is None:
        logger.info(
            "delegate_task: async delivery unsupported on this session "
            "runtime; running the batch synchronously instead."
        )
        return _run_sync_with_note(execute_and_aggregate, "no_async")

    session_key, origin_ui_session_id = _resolve_async_session_key(parent_agent, origin_ui_session_id)
    child_agents = [c for (_, _, c) in children]
    # The batch's lifecycle is owned by the async registry now: drop the children
    # from the parent's interrupt-propagation list (_build_child_agent attached
    # them, which is correct for sync runs).
    for c in child_agents:
        _detach_child(parent_agent, c)

    def _batch_interrupt():
        # Cancellation path for the detached batch (owned by the async registry).
        for c in child_agents:
            _signal_child_stop(c, "Async delegation cancelled")

    goals = [t["goal"] for t in task_list]
    dispatch = dispatch_async_delegation_batch(
        goals=goals,
        context=context,
        # Metadata for the completion block only; subagents inherit the
        # parent's toolsets (no model-facing toolsets arg).
        toolsets=None,
        role=top_role,
        model=creds["model"],
        session_key=session_key,
        origin_ui_session_id=origin_ui_session_id,
        origin_session_id=wake_sid,
        parent_session_id=getattr(parent_agent, "session_id", None),
        runner=lambda: execute_and_aggregate(honor_parent_interrupt=False),
        interrupt_fn=_batch_interrupt,
        max_async_children=_get_max_async_children(),
        # Reuse the live-transcript directory's id (when created) so the
        # returned delegation_id matches cache/delegation/live/<id>/.
        delegation_id=live_deleg_id,
        progress_fn=lambda: _batch_progress_token(child_agents),
    )
    if dispatch.get("status") == "dispatched":
        return json.dumps(_dispatched_payload(dispatch, goals, child_agents, live_paths), ensure_ascii=False)

    # Pool at capacity / schedule failure: the async unit was never accepted,
    # so just run inline (re-attaching to the parent list is not needed).
    logger.info(
        "delegate_task: async pool at capacity (%s); running the whole "
        "batch synchronously instead.",
        dispatch.get("error", "rejected"),
    )
    return _run_sync_with_note(execute_and_aggregate, "at_capacity")
