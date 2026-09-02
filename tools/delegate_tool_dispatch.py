"""Batch execution + background dispatch for delegate_task.

Split out of ``tools/delegate_tool.py`` (zero-back-ref regions of ``delegate_task``);
``delegate_task`` stays the orchestrator and calls these with its locals.
"""

from __future__ import annotations

import contextvars
import json
import logging
from typing import Any, Dict, List, Optional

from agent.interrupt_compat import request_hard_interrupt
from tools.delegate_tool_progress import (
    SUBAGENT_FAILURE_STATUSES,
    _clean_error_text,
    _emit_parent_console,
    format_batch_tag,
)

# Log-record parity with the origin module.
logger = logging.getLogger("tools.delegate_tool")


def _fabricated_entry(idx: int, status: str, error: str, child: Any) -> Dict[str, Any]:
    """Result entry for a child whose Future raised or never finished."""
    return {
        "task_index": idx,
        "status": status,
        "summary": None,
        "error": error,
        "api_calls": 0,
        "duration_seconds": 0,
        "_child_role": getattr(child, "_delegate_role", None),
    }


def _print_completion_line(parent_agent: Any, spinner_ref: Any, line: str) -> None:
    """Above-spinner line when a spinner exists (console fallback if it raises), else console."""
    if spinner_ref:
        try:
            spinner_ref.print_above(line)
            return
        except Exception:
            pass
    _emit_parent_console(parent_agent, f"  {line}")


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

    _origin_ui_session_id = origin_ui_session_id
    _origin_owner_transport = origin_owner_transport
    _origin_owner_session_record = origin_owner_session_record
    # Batch -- run in parallel with per-task progress lines
    completed_count = 0
    spinner_ref = getattr(parent_agent, "_delegate_spinner", None)

    # Daemon workers (tools.daemon_pool): the `with` block still joins
    # normally, but if the parent is interrupted while a child is
    # wedged, the abandoned worker must not block interpreter exit.
    from tools.daemon_pool import DaemonThreadPoolExecutor
    with DaemonThreadPoolExecutor(max_workers=max_children) as executor:
        futures = {}
        for i, t, child in children:
            child_context = contextvars.copy_context()
            future = executor.submit(
                child_context.run,
                _run_single_child,
                task_index=i,
                goal=t["goal"],
                child=child,
                parent_agent=parent_agent,
                owner_session_id=_origin_ui_session_id or None,
                owner_transport=_origin_owner_transport,
                owner_session_record=_origin_owner_session_record,
            )
            futures[future] = i

        # Poll futures with interrupt checking.  as_completed() blocks
        # until ALL futures finish — if a child agent gets stuck,
        # the parent blocks forever even after interrupt propagation.
        # Instead, use wait() with a short timeout so we can bail
        # when the parent is interrupted.
        # Map task_index -> child agent, so fabricated entries for
        # still-pending futures can carry the correct _delegate_role.
        _child_by_index = {i: child for (i, _, child) in children}

        pending = set(futures.keys())
        while pending:
            if (
                honor_parent_interrupt
                and getattr(parent_agent, "_interrupt_requested", False) is True
            ):
                # Parent interrupted — collect whatever finished and
                # abandon the rest.  Children already received the
                # interrupt signal; we just can't wait forever.
                for f in pending:
                    idx = futures[f]
                    if f.done():
                        try:
                            entry = f.result()
                        except Exception as exc:
                            entry = _fabricated_entry(idx, "error", str(exc), _child_by_index.get(idx))
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

            from concurrent.futures import wait as _cf_wait, FIRST_COMPLETED

            done, pending = _cf_wait(
                pending, timeout=0.5, return_when=FIRST_COMPLETED
            )
            for future in done:
                try:
                    entry = future.result()
                except Exception as exc:
                    idx = futures[future]
                    entry = _fabricated_entry(idx, "error", str(exc), _child_by_index.get(idx))
                results.append(entry)
                completed_count += 1

                # Print per-task completion line above the spinner
                idx = entry["task_index"]
                label = (
                    task_labels[idx] if idx < len(task_labels) else f"Task {idx}"
                )
                dur = entry.get("duration_seconds", 0)
                status = entry.get("status", "?")
                icon = "✓" if status == "completed" else "✗"
                remaining = n_tasks - completed_count
                _tag = format_batch_tag(live_deleg_id)
                _slot = f"{_tag} · {idx+1}/{n_tasks}" if _tag else f"{idx+1}/{n_tasks}"
                completion_line = f"{icon} [{_slot}] {label}  ({dur}s)"
                # Failed/errored/timed-out children: say WHY on the
                # same line, cleaned to one short human-readable
                # fragment — a bare ✗ reads as "silently dropped".
                if status in SUBAGENT_FAILURE_STATUSES:
                    _err_line = _clean_error_text(
                        entry.get("error"), max_chars=120
                    )
                    if _err_line:
                        completion_line += f" — {_err_line}"
                _print_completion_line(parent_agent, spinner_ref, completion_line)

                # Update spinner text to show remaining count
                if spinner_ref and remaining > 0:
                    try:
                        spinner_ref.update_text(
                            f"🔀 {'[' + _tag + '] ' if _tag else ''}{remaining} task{'s' if remaining != 1 else ''} remaining"
                        )
                    except Exception as e:
                        logger.debug("Spinner update_text failed: %s", e)

    # Sort by task_index so results match input order
    results.sort(key=lambda r: r["task_index"])


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

    _execute_and_aggregate = execute_and_aggregate
    _origin_wake_sid = origin_wake_sid
    _origin_ui_session_id = origin_ui_session_id
    from tools.async_delegation import dispatch_async_delegation_batch
    from tools.approval import get_current_session_key

    # Finite sessions cannot route a detached subagent result back to the
    # agent after their turn/process ends. This includes stateless HTTP
    # requests (#10760) and one-shot Kanban workers (#63169). Fall back to
    # SYNCHRONOUS execution so the result returns in this same turn instead
    # of handing out a handle with no durable consumer. Mirrors the
    # pool-at-capacity inline fallback below.
    try:
        from gateway.session_context import async_delivery_supported
        _async_ok = async_delivery_supported()
    except Exception:
        _async_ok = True

    _wake_sid = ""
    if not _async_ok:
        # The adapter itself cannot push, but if a raw session id is
        # bound (the API server always binds one — see
        # ApiServerAdapter._bind_api_server_session), gateway.wake can
        # still reach the session by self-POSTing /v1/chat/completions
        # with that id in X-Hermes-Session-Id once the batch completes.
        # Only fall back to forced-sync execution when there is truly no
        # session id to wake. Uses the origin captured before child
        # construction (see _origin_wake_sid above) — reading
        # HERMES_SESSION_ID here would return the subagent's internal id.
        _wake_sid = _origin_wake_sid
        if _wake_sid:
            logger.info(
                "delegate_task: async delivery unsupported on this "
                "session, but a session id is bound (%s) — dispatching "
                "in the background and waking the session via self-post "
                "when it completes instead of forcing synchronous "
                "execution.",
                _wake_sid,
            )
            _async_ok = True

    if not _async_ok:
        logger.info(
            "delegate_task: async delivery unsupported on this session "
            "runtime; running the batch synchronously instead."
        )
        _sync_result = _execute_and_aggregate()
        if isinstance(_sync_result, dict):
            _sync_result["note"] = (
                "background=true is not available in this session — it cannot "
                "receive a detached subagent result after the turn ends (a "
                "one-shot runner such as `hermes -z`, a cron job, a Kanban "
                "worker, or a stateless HTTP endpoint). The subagent(s) ran "
                "SYNCHRONOUSLY and the result is included above."
            )
        return json.dumps(_sync_result, ensure_ascii=False)

    _session_key = get_current_session_key(default="")
    try:
        from gateway.session_context import get_session_env

        _source = get_session_env("HERMES_SESSION_SOURCE", "")
        # Refresh from the same task-local source when available, but retain
        # the immutable value captured before child construction otherwise.
        _origin_ui_session_id = (
            get_session_env("HERMES_UI_SESSION_ID", "") or _origin_ui_session_id
        )
        # In desktop/TUI, the routable session key is the durable
        # AIAgent.session_id. Context compression can rotate that id during
        # the same turn before the TUI-side session dict is re-anchored;
        # if we capture the stale approval/session context key here, the
        # async completion becomes an orphan and any desktop poller may
        # consume it. Gateway chats are different: their session_key is the
        # platform conversation key (agent:main:...), so keep it there.
        if _source == "tui":
            _agent_session_id = str(getattr(parent_agent, "session_id", "") or "")
            if _agent_session_id:
                _session_key = _agent_session_id
    except Exception:
        _source = ""
    if not _session_key:
        # CLI (single-process) path: the approval contextvar is only bound
        # during gateway/TUI turns and HERMES_SESSION_KEY is not in the CLI
        # environment, so the key resolves empty here. Since #64240 the CLI
        # drains completions through a positive-ownership filter keyed on
        # the durable AIAgent.session_id — an empty session_key would fail
        # closed and the CLI could never claim its own completions, while
        # a restored foreign event with an empty key could leak into any
        # unfiltered consumer (#64484). Stamp the parent's durable session
        # id instead; compression rotations are handled on the drain side
        # via resolve_resume_session_id lineage resolution.
        _agent_session_id = str(getattr(parent_agent, "session_id", "") or "")
        if _agent_session_id:
            _session_key = _agent_session_id
    _parent_session_id = getattr(parent_agent, "session_id", None)
    _child_agents = [c for (_, _, c) in children]

    # Detach every child from the parent's interrupt-propagation list — the
    # batch's lifecycle is owned by the async registry now, not the parent
    # turn. _build_child_agent attached them (correct for sync runs).
    if hasattr(parent_agent, "_active_children"):
        _ac_lock = getattr(parent_agent, "_active_children_lock", None)
        for _c in _child_agents:
            try:
                if _ac_lock:
                    with _ac_lock:
                        parent_agent._active_children.remove(_c)
                else:
                    parent_agent._active_children.remove(_c)
            except ValueError:
                pass

    def _batch_runner():
        # This batch is detached from the foreground turn. Its lifecycle is
        # owned by the async registry and cancelled only via _batch_interrupt.
        return _execute_and_aggregate(honor_parent_interrupt=False)

    def _batch_interrupt():
        for _c in _child_agents:
            try:
                interrupted = request_hard_interrupt(_c, "Async delegation cancelled")
                if not interrupted and hasattr(_c, "_interrupt_requested"):
                    _c._interrupt_requested = True
            except Exception:
                pass

    def _batch_progress():
        # Progress token for the async registry's stale monitor: the
        # combined (api_call_count, current_tool, last_activity_ts) of
        # every child. last_activity_ts is ticked by _touch_activity on
        # every streamed chunk ("receiving stream response"), every tool
        # transition, and every API-call start/completion — so a child
        # streaming a long response is alive even though api_call_count
        # only advances when the call completes (same liveness signal as
        # the compaction inactivity budget, PR #71508). A fully frozen
        # token past the stale threshold means the detached batch is
        # wedged (e.g. stuck inside the first model API call — #60203).
        # in_tool=True while ANY child is inside a tool so legitimately
        # slow tools get the higher staleness ceiling, mirroring the
        # sync-path heartbeat monitor.
        parts = []
        in_tool = False
        for _c in _child_agents:
            try:
                _summary = _c.get_activity_summary()
                _tool = _summary.get("current_tool")
                parts.append(
                    (
                        _summary.get("api_call_count", 0),
                        _tool,
                        _summary.get("last_activity_ts"),
                    )
                )
                in_tool = in_tool or bool(_tool)
            except Exception:
                parts.append(None)
        return tuple(parts), in_tool

    _goals = [t["goal"] for t in task_list]
    dispatch = dispatch_async_delegation_batch(
        goals=_goals,
        context=context,
        # Metadata for the completion block only; subagents inherit the
        # parent's toolsets (no model-facing toolsets arg).
        toolsets=None,
        role=top_role,
        model=creds["model"],
        session_key=_session_key,
        origin_ui_session_id=_origin_ui_session_id,
        origin_session_id=_wake_sid,
        parent_session_id=_parent_session_id,
        runner=_batch_runner,
        interrupt_fn=_batch_interrupt,
        max_async_children=_get_max_async_children(),
        # Reuse the live-transcript directory's id (when created) so the
        # returned delegation_id matches cache/delegation/live/<id>/.
        delegation_id=live_deleg_id,
        progress_fn=_batch_progress,
    )

    if dispatch.get("status") == "dispatched":
        n = len(_goals)
        note = (
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
        )
        payload = {
            "status": "dispatched",
            "mode": "background",
            "count": n,
            "delegation_id": dispatch["delegation_id"],
            "goals": _goals,
            "note": note,
        }
        _sids = [
            getattr(_c, "_subagent_id", None) for _c in _child_agents
        ]
        if any(isinstance(s, str) and s for s in _sids):
            payload["subagent_ids"] = _sids
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
        return json.dumps(payload, ensure_ascii=False)

    # Pool at capacity / schedule failure — children are still attached
    # (we detach above only on the parent list, but the async unit was
    # never accepted, so re-attaching isn't needed: we just run inline).
    logger.info(
        "delegate_task: async pool at capacity (%s); running the whole "
        "batch synchronously instead.",
        dispatch.get("error", "rejected"),
    )
    _cap_result = _execute_and_aggregate()
    if isinstance(_cap_result, dict):
        _cap_result["note"] = (
            "The background delegation pool was at capacity "
            "(delegation.max_concurrent_children), so the subagent(s) ran "
            "SYNCHRONOUSLY and the result is included above. Raise "
            "delegation.max_concurrent_children in config.yaml to allow "
            "more concurrent background delegations."
        )
    return json.dumps(_cap_result, ensure_ascii=False)
