"""Human-readable rendering of background-process notification events.

Events come off ``ProcessRegistry.completion_queue`` (completion, watch_match,
watch_disabled, watch_overflow_*, async_delegation) and are turned into the
``[IMPORTANT: ...]`` / ``[ASYNC DELEGATION ...]`` text injected into the agent
conversation by the CLI drain loop, the gateway, and the TUI.
"""

import time


def _format_age(seconds: float) -> str:
    """Human-friendly elapsed string ('18m', '2h3m', '45s')."""
    try:
        s = int(max(0, seconds))
    except (TypeError, ValueError):
        return "?"
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m" if s == 0 else f"{m}m{s}s"
    h, m = divmod(m, 60)
    return f"{h}h" if m == 0 else f"{h}h{m}m"


def _model_not_found_patterns() -> "list[str]":
    """Model-not-found phrases shared with the failover classifier.

    Imported from ``agent.error_classifier`` so the batch renderer applies the
    same classification the failover path uses (no hand-copied list to drift).
    Fails open to a minimal built-in set so an import problem never hides the
    per-task blocks.
    """
    try:
        from agent.error_classifier import _MODEL_NOT_FOUND_PATTERNS

        return list(_MODEL_NOT_FOUND_PATTERNS)
    except Exception:
        return ["is not a valid model", "model not found", "model_not_found"]


def _delegation_config() -> dict:
    """Active delegation config (model/provider/fallbacks); ``{}`` on any error.

    Mirrors ``tools.delegate_tool._load_config`` lazily so the renderer sees the
    same model/provider the dispatcher used without importing the heavy
    delegation module at import time.
    """
    try:
        from tools.delegate_tool import _load_config as _cfg

        return _cfg() or {}
    except Exception:
        return {}


def _delegation_model_not_found(results, config) -> bool:
    """True when a result reflects a config-level model_not_found rejection.

    Requires both a model-not-found phrase AND the currently-configured model
    name in the same error/summary text, so a stale task failing on a
    different (removed) model is not mis-attributed to the config.
    """
    model = (config or {}).get("model")
    if not model:
        return False
    model = str(model).lower()
    for r in results or []:
        text = " ".join(
            str(part) for part in (r.get("error"), r.get("summary")) if part
        ).lower()
        if not text or model not in text:
            continue
        if any(p in text for p in _model_not_found_patterns()):
            return True
    return False


def _delegation_model_not_found_notice(results) -> "list[str] | None":
    """Config-level model_not_found notice lines, or None (fail-open) — once per batch."""
    config = _delegation_config()
    if not _delegation_model_not_found(results, config):
        return None
    model = config.get("model") or "?"
    provider = config.get("provider") or "configured provider"
    lines = [
        "⚠ SUBAGENT MODEL REJECTED: the configured Subagent Model "
        f'"{model}" was rejected by provider "{provider}" '
        "(HTTP 400: not a valid model ID).",
        "Every task in this batch failed for this reason before doing any work.",
        "Check Settings → Advanced → Subagent Model (or: "
        "hermes config get delegation.model).",
    ]
    try:
        from hermes_cli.fallback_config import get_fallback_chain

        if not get_fallback_chain(config):
            lines.append(
                "No fallback chain is configured, so no failover was attempted."
            )
    except Exception:
        pass
    return lines


_TRUNCATED_SUMMARY_NOTE = (
    "[TRUNCATED — subagent hit its iteration cap; the summary below "
    "may be incomplete. Verify before relying on it, or re-dispatch "
    "the unfinished part.]"
)


def _is_truncated(entry: dict) -> bool:
    return bool(entry.get("truncated") or entry.get("exit_reason") == "max_iterations")


def _dispatched_line(dispatched_at, completed_at) -> "str | None":
    if not isinstance(dispatched_at, (int, float)):
        return None
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(dispatched_at))
    return f"Dispatched: {ts} ({_format_age(completed_at - dispatched_at)} ago)"


def _task_source_lines(evt: dict) -> "list[str]":
    lines = []
    if evt.get("context"):
        lines.append(f"Context you provided: {evt['context']}")
    if evt.get("toolsets"):
        lines.append(f"Toolsets: {', '.join(evt['toolsets'])}")
    return lines


def _format_batch_delegation(evt: dict, deleg_id: str, completed_at: float) -> str:
    """Consolidated block for a delegate_task fan-out that finished as one unit."""
    results = evt.get("results") or []
    goals = evt.get("goals") or []
    n = len(results) if results else len(goals)
    total_dur = evt.get("total_duration_seconds", evt.get("duration_seconds", "?"))
    error = evt.get("error")
    lines = [
        f"[ASYNC DELEGATION BATCH COMPLETE — {deleg_id}]",
        f"A background fan-out of {n} subagent(s) you dispatched earlier "
        "has finished. All ran in parallel and waited on each other; their "
        "consolidated results are below. You may have moved on since "
        "dispatching — act on these or re-dispatch if things have changed.",
        "",
    ]
    dispatched = _dispatched_line(evt.get("dispatched_at"), completed_at)
    if dispatched:
        lines.append(dispatched)
    lines.extend(_task_source_lines(evt))
    lines.append(
        f"Role: {evt.get('role') or 'leaf'}   Model: {evt.get('model') or '?'}"
        f"   Total duration: {total_dur}s"
    )
    if error and not results:
        lines.append("--- ERROR ---")
        lines.append(f"The batch did not complete successfully: {error}")
        return "\n".join(lines)
    # Config-level rejection notice BEFORE the per-task wall — a rejected
    # delegation model fails every task identically and must not stay buried.
    _notice = _delegation_model_not_found_notice(results)
    if _notice:
        lines.append("")
        lines.extend(_notice)
    for r in sorted(results, key=lambda x: x.get("task_index", 0)):
        idx = r.get("task_index", 0)
        r_status = r.get("status", "?")
        r_summary = r.get("summary")
        r_error = r.get("error")
        r_goal = goals[idx] if idx < len(goals) else r.get("goal", "")
        r_truncated = _is_truncated(r)
        icon = "⚠" if r_truncated else ("✓" if r_status in ("completed", "success") else "✗")
        lines.append("")
        header = f"--- {icon} TASK {idx + 1}/{n}"
        if r_goal:
            header += f": {r_goal}"
        header += f"  (status={r_status}"
        if r.get("api_calls"):
            header += f", api_calls={r['api_calls']}"
        if r.get("duration_seconds") is not None:
            header += f", {r['duration_seconds']}s"
        if r_truncated:
            header += ", TRUNCATED: hit max_iterations — work may be incomplete"
        header += ") ---"
        lines.append(header)
        if r_status in ("completed", "success") and r_summary:
            if r_truncated:
                lines.append(_TRUNCATED_SUMMARY_NOTE)
            lines.append(r_summary)
        elif r_summary:
            if r_error:
                lines.append(f"({r_status}: {r_error})")
            lines.append("Partial output:")
            lines.append(r_summary)
        else:
            lines.append(
                f"(no summary — status={r_status}"
                + (f": {r_error}" if r_error else "")
                + ")"
            )
        r_live = r.get("live_transcript")
        if r_live:
            lines.append(
                f"Full live transcript (complete tool/assistant trace): {r_live}"
            )
    return "\n".join(lines)


def _format_async_delegation(evt: dict) -> str:
    """Format an async-delegation completion into a self-contained re-injection.

    Carries the FULL original task source (goal, context, toolsets, role,
    model) plus dispatch time, status, and the complete result summary: when
    this re-enters the conversation the agent may be deep in unrelated context
    and must be able to use the result OR re-dispatch without remembering why
    the subagent existed.
    """
    deleg_id = evt.get("delegation_id", "unknown")
    completed_at = evt.get("completed_at") or time.time()
    if evt.get("is_batch") or isinstance(evt.get("results"), list):
        return _format_batch_delegation(evt, deleg_id, completed_at)

    status = evt.get("status") or "completed"
    summary = evt.get("summary")
    error = evt.get("error")
    truncated = _is_truncated(evt)
    lines = [
        f"[ASYNC DELEGATION COMPLETE — {deleg_id}]",
        "A background subagent you dispatched earlier has finished. You may "
        "have moved on since dispatching it; the full task source is below so "
        "you can act on the result or re-dispatch if things have changed.",
        "",
    ]
    dispatched = _dispatched_line(evt.get("dispatched_at"), completed_at)
    if dispatched:
        lines.append(dispatched)
    lines.append(f"Original goal: {evt.get('goal', '') or ''}")
    lines.extend(_task_source_lines(evt))
    lines.append(f"Role: {evt.get('role') or 'leaf'}   Model: {evt.get('model') or '?'}")
    _notice = _delegation_model_not_found_notice([evt])
    if _notice:
        lines.append("")
        lines.extend(_notice)
    _trunc = " [TRUNCATED: hit max_iterations — work may be incomplete]" if truncated else ""
    lines.append(
        f"Status: {status}   API calls: {evt.get('api_calls', 0)}   "
        f"Duration: {evt.get('duration_seconds', '?')}s{_trunc}"
    )
    lines.append("--- RESULT ---")
    if status in ("completed", "success") and summary:
        if truncated:
            lines.append(_TRUNCATED_SUMMARY_NOTE)
        lines.append(summary)
    else:
        if status == "interrupted":
            lines.append(
                "The subagent was interrupted before completing"
                + (f": {error}" if error else ".")
            )
        else:  # error / timeout / failed
            lines.append(
                f"The subagent did not complete successfully (status={status})."
                + (f"\n{error}" if error else "")
            )
        if summary:
            lines.append("Partial output:")
            lines.append(summary)
    return "\n".join(lines)


def _delegation_attribution_line(evt: dict) -> "str | None":
    """One-line provenance for a subagent-owned process event, else None.

    Subagents run terminal sessions under ``task_id == subagent_id``; a
    background process they started outlives the child and is routed to the
    PARENT conversation, which otherwise sees an anonymous raw output wall.
    Judged on ``owner_task_id`` (the raw spawning id) — ``task_id`` is the
    container key and may be collapsed to the session key.
    """
    task_id = str(evt.get("owner_task_id") or evt.get("task_id") or "")
    if not task_id.startswith("sa-"):
        return None
    try:
        from tools.delegate_tool import get_subagent_attribution

        info = get_subagent_attribution(task_id)
    except Exception:
        info = None
    if not info:
        # Registry entry aged out — still attribute generically, not anonymously.
        return f"Started by subagent {task_id} (delegate_task)."
    goal = str(info.get("goal") or "").strip()
    if len(goal) > 120:
        goal = goal[:117] + "..."
    deleg = info.get("delegation_id")
    parts = [f"Started by subagent {task_id}"]
    if deleg:
        parts.append(f"of delegation {deleg}")
    line = " ".join(parts) + "."
    if goal:
        line += f' Task: "{goal}"'
    return line


def _completion_status(evt: dict) -> str:
    _exit = evt.get("exit_code", "?")
    _reason = evt.get("completion_reason") or "exited"
    if _reason == "killed":
        return f"terminated by {evt.get('termination_source') or 'Hermes'}"
    if _reason == "lost":
        return "marked lost because the process backend disappeared"
    if _reason == "failed_start":
        return "failed to start"
    if _exit == 0:
        return "completed normally"
    return "exited"


def format_process_notification(evt: dict) -> "str | None":
    """Format a completion_queue event into an ``[IMPORTANT: ...]`` message."""
    evt_type = evt.get("type", "completion")
    _sid = evt.get("session_id", "unknown")
    _cmd = evt.get("command", "unknown")
    _attribution = _delegation_attribution_line(evt)

    # watch_disabled and overflow events carry their own human-readable
    # `message`; without this branch overflow events would fall through to the
    # completion formatter as a phantom "process exited (exit code ?)".
    if evt_type in ("watch_disabled", "watch_overflow_tripped", "watch_overflow_released"):
        return f"[IMPORTANT: {evt.get('message', '')}]"

    if evt_type == "watch_match":
        _sup = evt.get("suppressed", 0)
        text = (
            f"[IMPORTANT: Background process {_sid} matched "
            f"watch pattern \"{evt.get('pattern', '?')}\".\n"
        )
        if _attribution:
            text += f"{_attribution}\n"
        text += f"Command: {_cmd}\nMatched output:\n{evt.get('output', '')}"
        if _sup:
            text += f"\n({_sup} earlier matches were suppressed by rate limit)"
        return text + "]"

    if evt_type == "async_delegation":
        return _format_async_delegation(evt)

    _exit = evt.get("exit_code", "?")
    _out = evt.get("output", "")
    _signal = ", SIGTERM" if _exit in {-15, 143, "-15", "143"} else ""
    text = (
        f"[IMPORTANT: Background process {_sid} {_completion_status(evt)} "
        f"(exit code {_exit}{_signal}).\n"
    )
    if _attribution:
        text += f"{_attribution}\n"
        # A subagent-owned process's full output belongs in the child's
        # transcript, not as a raw wall in the parent — trim hard but keep
        # enough tail to recognise failures.
        if isinstance(_out, str) and len(_out) > 600:
            _out = (
                "...(output trimmed — subagent-owned process; see the "
                "delegation's live transcript for full output)\n"
                + _out[-600:]
            )
    text += f"Command: {_cmd}\nOutput:\n{_out}]"
    return text
