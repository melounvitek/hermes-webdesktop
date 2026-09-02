"""Child progress relay, console formatting and child system-prompt construction for delegate_task.

Split out of ``tools/delegate_tool.py``; every moved name is re-imported there, so
``tools.delegate_tool.<name>`` keeps resolving (and monkeypatching) as before.
"""

from __future__ import annotations

import logging
import enum
import os
import threading
from typing import Any, Dict, List, Optional
from tools.delegate_tool_registry import _active_subagents, _active_subagents_lock

# Log-record parity with the origin module.
logger = logging.getLogger("tools.delegate_tool")

# Terminal child statuses that mean "the subagent did NOT deliver a usable
# result". Shared by the CLI spinner echo, the gateway failure notice, and
# the parent-facing failure summary so every surface agrees on what counts
# as a failure.
SUBAGENT_FAILURE_STATUSES = frozenset({"failed", "error", "timeout"})


def _clean_error_text(error: Any, max_chars: int = 200) -> str:
    """Reduce an arbitrary error payload to one clean human-readable line.

    Provider/SDK errors routinely arrive as multi-line tracebacks or JSON
    walls. For a chat-facing notice we want the single most informative
    line: the exception message (last line of a traceback) or the first
    non-empty line otherwise, hard-capped in length.
    """
    text = str(error or "").strip()
    if not text:
        return ""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return ""
    # A traceback's last line is the actual exception message.
    line = lines[-1] if lines[0].startswith("Traceback") else lines[0]
    if len(line) > max_chars:
        line = line[: max_chars - 3] + "..."
    return line


def format_subagent_failure_line(
    goal: Optional[str],
    status: Optional[str],
    error: Any = None,
    duration_seconds: Any = None,
) -> str:
    """One clean, human-readable line describing a failed subagent.

    Rendered directly to the user (CLI spinner echo, gateway platform
    notice) — no JSON, no traceback, no internal field names. Example:

        ⚠️ Subagent failed — "research competitor pricing": Error code: 404 —
        model not found (after 12s)
    """
    goal_label = (goal or "").strip().replace("\n", " ")
    if len(goal_label) > 60:
        goal_label = goal_label[:57] + "..."
    verb = "timed out" if status == "timeout" else "failed"
    line = f"⚠️ Subagent {verb}"
    if goal_label:
        line += f' — "{goal_label}"'
    err = _clean_error_text(error)
    if err:
        line += f": {err}"
    if isinstance(duration_seconds, (int, float)) and duration_seconds > 0:
        line += f" (after {round(duration_seconds)}s)"
    return line


class DelegateEvent(str, enum.Enum):
    """Formal event types emitted during delegation progress.

    _build_child_progress_callback normalises incoming legacy strings
    (``tool.started``, ``_thinking``, …) to these enum values via
    ``_LEGACY_EVENT_MAP``.  External consumers (gateway SSE, ACP adapter,
    CLI) still receive the legacy strings during the deprecation window.

    TASK_SPAWNED / TASK_COMPLETED / TASK_FAILED are reserved for
    future orchestrator lifecycle events and are not currently emitted.
    """

    TASK_SPAWNED = "delegate.task_spawned"
    TASK_PROGRESS = "delegate.task_progress"
    TASK_COMPLETED = "delegate.task_completed"
    TASK_FAILED = "delegate.task_failed"
    TASK_THINKING = "delegate.task_thinking"
    TASK_TOOL_STARTED = "delegate.tool_started"
    TASK_TOOL_COMPLETED = "delegate.tool_completed"


# Legacy event strings → DelegateEvent mapping.
# Incoming child-agent events use the old names; the callback normalises them.
_LEGACY_EVENT_MAP: Dict[str, DelegateEvent] = {
    "_thinking": DelegateEvent.TASK_THINKING,
    "reasoning.available": DelegateEvent.TASK_THINKING,
    "tool.started": DelegateEvent.TASK_TOOL_STARTED,
    "tool.completed": DelegateEvent.TASK_TOOL_COMPLETED,
    "subagent_progress": DelegateEvent.TASK_PROGRESS,
}


def _build_child_system_prompt(
    goal: str,
    context: Optional[str] = None,
    *,
    workspace_path: Optional[str] = None,
    role: str = "leaf",
    max_spawn_depth: int = 2,
    child_depth: int = 1,
) -> str:
    """Build a focused system prompt for a child agent.

    When role='orchestrator', appends a delegation-capability block
    modeled on OpenClaw's buildSubagentSystemPrompt (canSpawn branch at
    inspiration/openclaw/src/agents/subagent-system-prompt.ts:63-95).
    The depth note is literal truth (grounded in the passed config) so
    the LLM doesn't confabulate nesting capabilities that don't exist.
    """
    parts = [
        "You are a focused subagent working on a specific delegated task.",
        "",
        f"YOUR TASK:\n{goal}",
    ]
    if context and context.strip():
        parts.append(f"\nCONTEXT:\n{context}")
    if workspace_path and str(workspace_path).strip():
        parts.append(
            "\nWORKSPACE PATH:\n"
            f"{workspace_path}\n"
            "Use this exact path for local repository/workdir operations unless the task explicitly says otherwise."
        )
        # Project context files (AGENTS.md / CLAUDE.md / .cursorrules ...)
        # from the workspace, via the SAME discovery/priority/cap logic the
        # main agent's system prompt uses. Children are constructed with
        # skip_context_files=True (their prompt is this focused one), so
        # without this a subagent works in a repo without the repo's own
        # conventions unless it thinks to go read them. SOUL.md is skipped —
        # identity belongs to the parent. workspace_path comes only from
        # explicit sources (_resolve_workspace_hint: TERMINAL_CWD / agent cwd
        # hints, never bare getcwd), so the #64590 install-tree-fallback leak
        # doesn't apply here. Best-effort: on any failure the child prompt is
        # simply built without the block.
        try:
            from agent.prompt_builder import build_context_files_prompt

            _ctx_files = build_context_files_prompt(
                cwd=str(workspace_path), skip_soul=True
            )
        except Exception:
            logger.debug(
                "subagent: workspace context-files load failed", exc_info=True
            )
            _ctx_files = ""
        if _ctx_files.strip():
            parts.append(
                "\nThe workspace's project context files are reproduced "
                "below. Their conventions and invariants are binding for "
                "your work in this workspace.\n\n" + _ctx_files.strip()
            )
    parts.append(
        "\nComplete this task using the tools available to you. "
        "When finished, provide a clear, concise summary of:\n"
        "- What you did\n"
        "- What you found or accomplished\n"
        "- Any files you created or modified\n"
        "- Any issues encountered\n\n"
        "Important workspace rule: Never assume a repository lives at /workspace/... or any other container-style path unless the task/context explicitly gives that path. "
        "If no exact local path is provided, discover it first before issuing git/workdir-specific commands.\n\n"
        "Keep your final summary tight: lead with outcomes, prefer bullet "
        "points over paragraphs, and don't replay your whole process. Your "
        "response is returned to the parent agent as a summary, and overlong "
        "summaries crowd out the parent's context window."
    )
    if role == "orchestrator":
        child_note = (
            "Your own children MUST be leaves (cannot delegate further) "
            "because they would be at the depth floor — you cannot pass "
            "role='orchestrator' to your own delegate_task calls."
            if child_depth + 1 >= max_spawn_depth
            else "Your own children can themselves be orchestrators or leaves, "
            "depending on the `role` you pass to delegate_task. Default is "
            "'leaf'; pass role='orchestrator' explicitly when a child "
            "needs to further decompose its work."
        )
        parts.append(
            "\n## Subagent Spawning (Orchestrator Role)\n"
            "You have access to the `delegate_task` tool and CAN spawn "
            "your own subagents to parallelize independent work.\n\n"
            "WHEN to delegate:\n"
            "- The goal decomposes into 2+ independent subtasks that can "
            "run in parallel (e.g. research A and B simultaneously).\n"
            "- A subtask is reasoning-heavy and would flood your context "
            "with intermediate data.\n\n"
            "WHEN NOT to delegate:\n"
            "- Single-step mechanical work — do it directly.\n"
            "- Trivial tasks you can execute in one or two tool calls.\n"
            "- Re-delegating your entire assigned goal to one worker "
            "(that's just pass-through with no value added).\n\n"
            "Coordinate your workers' results and synthesize them before "
            "reporting back to your parent. You are responsible for the "
            "final summary, not your workers.\n\n"
            f"NOTE: You are at depth {child_depth}. The delegation tree "
            f"is capped at max_spawn_depth={max_spawn_depth}. {child_note}"
        )
    return "\n".join(parts)


def _resolve_workspace_hint(parent_agent) -> Optional[str]:
    """Best-effort local workspace hint for child prompts.

    We only inject a path when we have a concrete absolute directory. This avoids
    teaching subagents a fake container path while still helping them avoid
    guessing `/workspace/...` for local repo tasks.
    """
    candidates = [
        os.getenv("TERMINAL_CWD"),
        getattr(
            getattr(parent_agent, "_subdirectory_hints", None), "working_dir", None
        ),
        getattr(parent_agent, "terminal_cwd", None),
        getattr(parent_agent, "cwd", None),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        try:
            text = os.path.abspath(os.path.expanduser(str(candidate)))
        except Exception:
            continue
        if os.path.isabs(text) and os.path.isdir(text):
            return text
    return None


_BATCH_ORDINALS: Dict[str, int] = {}
_BATCH_ORDINALS_LOCK = threading.Lock()


def format_batch_tag(delegation_id: Optional[str]) -> str:
    """Short human tag identifying which delegation batch a line belongs to.

    ``deleg_6a664903`` → ``set 1`` (first batch seen in this process),
    the next distinct id → ``set 2``, and so on. Several batches (a parent's
    fan-out plus a child's nested fan-out, or two concurrent tools) print
    interleaved ``[n/N]`` progress lines to the same console; without a batch
    tag a ``✓ [3/3]`` and a ``✓ [3/9]`` are indistinguishable, and a raw hex
    slice (``[b2ac 3/9]``) is attributable but unreadable. Empty string when
    no id is known so callers can concatenate unconditionally.
    """
    if not isinstance(delegation_id, str) or not delegation_id:
        return ""
    with _BATCH_ORDINALS_LOCK:
        n = _BATCH_ORDINALS.get(delegation_id)
        if n is None:
            n = len(_BATCH_ORDINALS) + 1
            _BATCH_ORDINALS[delegation_id] = n
    return f"set {n}"


def _batch_prefix(delegation_id: Optional[str], task_index: int, task_count: int) -> str:
    """``[set 2 · 3/9] `` for batch children, ``[set 2] `` for a lone child,
    ``[3/9] `` / ``""`` when the batch id is unknown."""
    tag = format_batch_tag(delegation_id)
    if task_count > 1:
        inner = f"{tag} · {task_index + 1}/{task_count}" if tag else f"{task_index + 1}/{task_count}"
        return f"[{inner}] "
    return f"[{tag}] " if tag else ""


def _emit_parent_console(parent_agent, line: str) -> None:
    """Emit a human-readable progress line to the parent's console.

    Routes through ``parent_agent._safe_print`` when available so headless
    stdio hosts (ACP, gateway API) can redirect non-protocol output to
    stderr via their configured ``_print_fn``. A bare ``print()`` would
    otherwise land on stdout and corrupt JSON-RPC framing.
    """
    printer = getattr(parent_agent, "_safe_print", None)
    if callable(printer):
        try:
            printer(line)
            return
        except Exception:
            pass
    print(line)


def _build_child_progress_callback(
    task_index: int,
    goal: str,
    parent_agent,
    task_count: int = 1,
    *,
    subagent_id: Optional[str] = None,
    parent_id: Optional[str] = None,
    depth: Optional[int] = None,
    model: Optional[str] = None,
    toolsets: Optional[List[str]] = None,
    session_ref: Optional[Dict[str, Any]] = None,
) -> Optional[callable]:
    """Build a callback that relays child agent tool calls to the parent display.

    Two display paths:
      CLI:     prints tree-view lines above the parent's delegation spinner
      Gateway: batches tool names and relays to parent's progress callback

    The identity kwargs (``subagent_id``, ``parent_id``, ``depth``, ``model``,
    ``toolsets``) are threaded into every relayed event so the TUI can
    reconstruct the live spawn tree and route per-branch controls (kill,
    pause) back by ``subagent_id``.  All are optional for backward compat —
    older callers that ignore them still produce a flat list on the TUI.

    Returns None if no display mechanism is available, in which case the
    child agent runs with no progress callback (identical to current behavior).
    """
    spinner = getattr(parent_agent, "_delegate_spinner", None)
    parent_cb = getattr(parent_agent, "tool_progress_callback", None)

    if not spinner and not parent_cb:
        return None  # No display → no callback → zero behavior change

    # Show 1-indexed prefix only in batch mode (multiple tasks). The batch tag
    # (short delegation id) is resolved lazily from session_ref because the
    # callback is built before delegate_task stamps ``_delegation_id`` on the
    # child; delegate_task drops the id into the same shared ref.
    def _prefix() -> str:
        deleg = session_ref.get("delegation_id") if session_ref else None
        return _batch_prefix(deleg, task_index, task_count)

    goal_label = (goal or "").strip()

    # Gateway: batch tool names, flush periodically
    _BATCH_SIZE = 5
    _batch: List[str] = []
    _tool_count = [0]  # per-subagent running counter (list for closure mutation)

    def _identity_kwargs() -> Dict[str, Any]:
        kw: Dict[str, Any] = {
            "task_index": task_index,
            "task_count": task_count,
            "goal": goal_label,
        }
        if subagent_id is not None:
            kw["subagent_id"] = subagent_id
        if parent_id is not None:
            kw["parent_id"] = parent_id
        if depth is not None:
            kw["depth"] = depth
        if model is not None:
            kw["model"] = model
        if toolsets is not None:
            kw["toolsets"] = list(toolsets)
        # The child's own session id — filled into the shared ref once the
        # child agent exists (the callback is built first), so every relayed
        # event lets UIs open/inspect the subagent's session directly.
        if session_ref and session_ref.get("session_id"):
            kw["child_session_id"] = str(session_ref["session_id"])
        if session_ref and session_ref.get("delegation_id"):
            kw["delegation_id"] = str(session_ref["delegation_id"])
        kw["tool_count"] = _tool_count[0]
        return kw

    def _relay(
        event_type: str, tool_name: str = None, preview: str = None, args=None, **kwargs
    ):
        if not parent_cb:
            return
        payload = _identity_kwargs()
        payload.update(kwargs)  # caller overrides (e.g. status, duration_seconds)
        try:
            parent_cb(event_type, tool_name, preview, args, **payload)
        except Exception as e:
            logger.debug("Parent callback failed: %s", e)

    def _tree_line(text: str) -> None:
        """Print one tree-view line above the CLI spinner (no-op without a spinner)."""
        if not spinner:
            return
        try:
            spinner.print_above(f" {_prefix()}├─ {text}")
        except Exception as e:
            logger.debug("Spinner print_above failed: %s", e)

    def _flush_batch() -> None:
        if parent_cb and _batch:
            _relay("subagent.progress", preview=f"🔀 {_prefix()}{', '.join(_batch)}")
            _batch.clear()

    def _short(text: str, n: int) -> str:
        return (text[:n] + "...") if len(text) > n else text

    # ── Lifecycle events emitted by the orchestrator itself (not DelegateEvent) ──
    def _on_start(tool_name, preview, args, kwargs):
        if goal_label:
            _tree_line(f"🔀 {_short(goal_label, 55)}")
        _relay("subagent.start", preview=preview or goal_label or "", **kwargs)

    def _on_complete(tool_name, preview, args, kwargs):
        # Failed child: echo one clean reason line into the CLI tree so the human
        # sees WHY, not just a vanished branch. Gateway-side rendering happens in
        # TurnRunner.progress_callback off the relayed event.
        if kwargs.get("status") in SUBAGENT_FAILURE_STATUSES:
            _tree_line(
                format_subagent_failure_line(
                    goal_label,
                    kwargs.get("status"),
                    error=kwargs.get("summary") or preview,
                    duration_seconds=kwargs.get("duration_seconds"),
                )
            )
        _relay("subagent.complete", preview=preview, **kwargs)

    def _on_text(tool_name, preview, args, kwargs):
        # Streamed child reply text, relayed verbatim so a gateway watch window
        # mirrors the child "talking". No spinner echo: CLI/TUI progress
        # handlers ignore non-tool events, so this is inert there.
        _relay("subagent.text", preview=preview)

    # ── DelegateEvent handlers ──
    def _on_thinking(tool_name, preview, args, kwargs):
        text = preview or tool_name or ""
        _tree_line(f'💭 "{_short(text, 55)}"')
        _relay("subagent.thinking", preview=text)

    def _on_progress(tool_name, preview, args, kwargs):
        # Pre-batched summary from a nested orchestrator's grandchild; upstream
        # emits parent_cb("subagent_progress", summary) with the summary in the
        # tool_name slot. Pass through: render distinctly (not via the tool
        # emoji lookup) and relay upward without re-batching.
        summary_text = tool_name or preview or ""
        if summary_text:
            _tree_line(f"🔀 {summary_text}")
        if parent_cb:
            try:
                parent_cb("subagent_progress", f"{_prefix()}{summary_text}")
            except Exception as e:
                logger.debug("Parent callback relay failed: %s", e)

    def _on_tool_started(tool_name, preview, args, kwargs):
        _tool_count[0] += 1
        if subagent_id is not None:
            with _active_subagents_lock:
                rec = _active_subagents.get(subagent_id)
                if rec is not None:
                    rec["tool_count"] = _tool_count[0]
                    rec["last_tool"] = tool_name or ""
        if spinner:
            from agent.display import get_tool_emoji

            line = f"{get_tool_emoji(tool_name or '')} {tool_name}"
            short = _short(preview, 35) if preview else ""
            if short:
                line += f'  "{short}"'
            _tree_line(line)
        if parent_cb:
            _relay("subagent.tool", tool_name, preview, args)
            _batch.append(tool_name or "")
            if len(_batch) >= _BATCH_SIZE:
                _flush_batch()

    _lifecycle_handlers = {
        "subagent.start": _on_start,
        "subagent.complete": _on_complete,
        "subagent.text": _on_text,
    }
    # Any other DelegateEvent (TASK_TOOL_STARTED and the reserved TASK_* values)
    # takes the tool-started path, as before the table.
    _event_handlers = {
        DelegateEvent.TASK_THINKING: _on_thinking,
        DelegateEvent.TASK_PROGRESS: _on_progress,
        DelegateEvent.TASK_TOOL_COMPLETED: None,
    }

    def _callback(
        event_type, tool_name: str = None, preview: str = None, args=None, **kwargs
    ):
        handler = _lifecycle_handlers.get(event_type) if isinstance(event_type, str) else None
        if handler is None:
            # Normalise legacy strings, "delegate.*" strings and DelegateEvent
            # values to one DelegateEvent; unknown events are ignored.
            if isinstance(event_type, DelegateEvent):
                event = event_type
            else:
                event = _LEGACY_EVENT_MAP.get(event_type)
                if event is None:
                    try:
                        event = DelegateEvent(event_type)
                    except (ValueError, TypeError):
                        return
            handler = _event_handlers.get(event, _on_tool_started)
            if handler is None:
                return
        handler(tool_name, preview, args, kwargs)

    def _flush():
        """Flush remaining batched tool names to gateway on completion."""
        _flush_batch()

    _callback._flush = _flush
    return _callback
