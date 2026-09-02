"""Agent-level ("inline") tool executors shared by the sequential and concurrent tool paths.

These tools need live ``AIAgent`` state (stores, callbacks, session DB) and therefore
bypass the tool registry. Each executor is ``fn(agent, args, ctx) -> result``; the
table replaces two hand-maintained if/elif chains (``invoke_tool`` and
``execute_tool_calls_sequential``) that had drifted apart. Tool modules are imported
lazily inside the bodies so ``patch("tools.x.y")`` in tests keeps working.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional


def tool_hook_ids(agent, effective_task_id: str, tool_call_id: Optional[str]) -> Dict[str, str]:
    """Identity kwargs every tool hook/middleware call carries (all coerced to ``""``)."""
    return {
        "task_id": effective_task_id or "",
        "session_id": getattr(agent, "session_id", "") or "",
        "tool_call_id": tool_call_id or "",
        "turn_id": getattr(agent, "_current_turn_id", "") or "",
        "api_request_id": getattr(agent, "_current_api_request_id", "") or "",
    }


def emit_terminal_post_tool_call(
    agent,
    *,
    function_name: str,
    function_args: dict,
    result: Any,
    effective_task_id: str,
    tool_call_id: Optional[str],
    duration_ms: int = 0,
    status: Optional[str] = None,
    error_type: Optional[str] = None,
    error_message: Optional[str] = None,
    middleware_trace: Optional[list] = None,
) -> None:
    """Emit the one terminal ``post_tool_call`` hook for a tool_call_id (best-effort)."""
    try:
        from model_tools import _emit_post_tool_call_hook
        _emit_post_tool_call_hook(
            function_name=function_name,
            function_args=function_args,
            result=result,
            **tool_hook_ids(agent, effective_task_id, tool_call_id),
            duration_ms=duration_ms,
            status=status,
            error_type=error_type,
            error_message=error_message,
            middleware_trace=list(middleware_trace or []),
        )
    except Exception:
        pass


@dataclass
class InlineToolContext:
    """Per-call state an inline executor may need beyond its arguments."""

    effective_task_id: str
    tool_call_id: Optional[str] = None
    messages: Optional[list] = None


def _todo_list(agent, args: dict, ctx: InlineToolContext) -> Any:
    from tools.todo_tool import todo_tool as _todo_tool

    return _todo_tool(
        todos=args.get("todos"),
        merge=args.get("merge", False),
        store=agent._todo_store,
    )


def _message_agent(agent, args: dict, ctx: InlineToolContext) -> Any:
    # Bot Mode teammate DM is injected, not registered: only a canonical Bot
    # Chat session carries the schema, and the tool re-gates on the title.
    from tools.bot_mode_dm import message_agent_tool as _message_agent_tool

    return _message_agent_tool(
        target=args.get("target", ""),
        message=args.get("message", ""),
        task_id=ctx.effective_task_id,
        agent=agent,
    )


def _session_search(agent, args: dict, ctx: InlineToolContext) -> Any:
    session_db = agent._get_session_db_for_recall()
    if not session_db:
        from hermes_state import format_session_db_unavailable

        return json.dumps({"success": False, "error": format_session_db_unavailable()})
    from tools.session_search_tool import session_search as _session_search_tool

    return _session_search_tool(
        query=args.get("query", ""),
        role_filter=args.get("role_filter"),
        limit=args.get("limit", 3),
        session_id=args.get("session_id"),
        around_message_id=args.get("around_message_id"),
        window=args.get("window", 5),
        sort=args.get("sort"),
        detail=args.get("detail", "adaptive"),
        db=session_db,
        current_session_id=agent.session_id,
    )


def _memory(agent, args: dict, ctx: InlineToolContext) -> Any:
    from tools.memory_tool import memory_tool as _memory_tool

    result = _memory_tool(
        action=args.get("action"),
        target=args.get("target", "memory"),
        content=args.get("content"),
        old_text=args.get("old_text"),
        operations=args.get("operations"),
        store=agent._memory_store,
    )
    # Mirror built-in memory writes to external providers; gating lives in
    # MemoryManager.notify_memory_tool_write.
    if agent._memory_manager:
        agent._memory_manager.notify_memory_tool_write(
            result,
            args,
            build_metadata=lambda: agent._build_memory_write_metadata(
                task_id=ctx.effective_task_id,
                tool_call_id=ctx.tool_call_id,
            ),
        )
    return result


def _clarify(agent, args: dict, ctx: InlineToolContext) -> Any:
    from tools.clarify_tool import clarify_tool as _clarify_tool

    return _clarify_tool(
        question=args.get("question", ""),
        choices=args.get("choices"),
        multi_select=args.get("multi_select", False),
        questions=args.get("questions"),
        callback=agent.clarify_callback,
    )


def _read_terminal(agent, args: dict, ctx: InlineToolContext) -> Any:
    from tools.read_terminal_tool import read_terminal_tool as _read_terminal_tool

    return _read_terminal_tool(
        start_line=args.get("start_line"),
        count=args.get("count"),
        callback=getattr(agent, "read_terminal_callback", None),
    )


def _desktop_preview(agent, args: dict, ctx: InlineToolContext) -> Any:
    # action=read needs the GUI callback (agent-level); open/close go through the
    # registry handler like any other tool.
    if (args.get("action") or "").strip() == "read":
        from tools.read_preview_tool import read_preview_tool as _read_preview_tool

        return _read_preview_tool(
            start=args.get("start"),
            count=args.get("count"),
            callback=getattr(agent, "read_preview_callback", None),
        )
    from tools.preview_tool import _handle_preview

    return _handle_preview(args)


def _drive_preview(agent, args: dict, ctx: InlineToolContext) -> Any:
    from tools.drive_preview_tool import drive_preview_tool as _drive_preview_tool

    return _drive_preview_tool(
        action=args.get("action", ""),
        ref=args.get("ref"),
        selector=args.get("selector"),
        text=args.get("text"),
        key=args.get("key"),
        submit=args.get("submit"),
        amount=args.get("amount"),
        to=args.get("to"),
        limit=args.get("max"),
        callback=getattr(agent, "drive_preview_callback", None),
    )


def _annotate_preview(agent, args: dict, ctx: InlineToolContext) -> Any:
    from tools.annotate_preview_tool import annotate_preview_tool as _annotate_preview_tool

    return _annotate_preview_tool(
        action=args.get("action", "add"),
        ref=args.get("ref"),
        selector=args.get("selector"),
        label=args.get("label"),
        callback=getattr(agent, "drive_preview_callback", None),
    )


def _read_window_below(agent, args: dict, ctx: InlineToolContext) -> Any:
    from tools.read_window_tool import read_window_below_tool as _read_window_below_tool

    return _read_window_below_tool(
        callback=getattr(agent, "read_window_below_callback", None),
    )


def _gui_tour(agent, args: dict, ctx: InlineToolContext) -> Any:
    from tools.tour_tool import tour_tool as _tour_tool

    return _tour_tool(
        action=args.get("action", ""),
        surface=args.get("surface"),
        selector=args.get("selector"),
        title=args.get("title"),
        text=args.get("text"),
        side=args.get("side"),
        steps=args.get("steps"),
        step_index=args.get("step_index"),
        callback=getattr(agent, "tour_callback", None),
    )


def _setup_mcp(agent, args: dict, ctx: InlineToolContext) -> Any:
    from tools.setup_mcp_tool import setup_mcp_tool as _setup_mcp_tool

    return _setup_mcp_tool(
        server=args.get("server", ""),
        action=args.get("action", "install"),
        reason=args.get("reason", ""),
        callback=getattr(agent, "setup_mcp_callback", None),
    )


def _delegate_task(agent, args: dict, ctx: InlineToolContext) -> Any:
    return agent._dispatch_delegate_task(args)


InlineToolExecutor = Callable[[Any, dict, InlineToolContext], Any]

# Order is the historical if/elif order of ``execute_tool_calls_sequential``.
INLINE_TOOL_EXECUTORS: Dict[str, InlineToolExecutor] = {
    "todo_list": _todo_list,
    "message_agent": _message_agent,
    "session_search": _session_search,
    "memory": _memory,
    "clarify": _clarify,
    "read_terminal": _read_terminal,
    "desktop_preview": _desktop_preview,
    "drive_preview": _drive_preview,
    "annotate_preview": _annotate_preview,
    "read_window_below": _read_window_below,
    "gui_tour": _gui_tour,
    "setup_mcp": _setup_mcp,
    "delegate_task": _delegate_task,
}

# ``invoke_tool`` (concurrent path) historically consulted the memory manager right
# after these three names and before the remaining inline tools; it never handled
# ``message_agent`` inline (that name falls through to the registry there).
INVOKE_TOOL_PRE_MEMORY_MANAGER_NAMES = frozenset({"todo_list", "session_search", "memory"})


def memory_manager_executor(function_name: str) -> InlineToolExecutor:
    """Executor routing ``function_name`` through ``agent._memory_manager``."""

    def _run(agent, args: dict, ctx: InlineToolContext) -> Any:
        return agent._memory_manager.handle_tool_call(function_name, args)

    return _run


def resolve_invoke_tool_executor(agent, function_name: str) -> Optional[InlineToolExecutor]:
    """Inline executor for ``invoke_tool`` (concurrent path), or None for registry dispatch.

    Preserves the historical precedence: todo_list/session_search/memory, then memory
    manager tools, then the remaining inline tools (``message_agent`` excluded).
    """
    if function_name in INVOKE_TOOL_PRE_MEMORY_MANAGER_NAMES:
        return INLINE_TOOL_EXECUTORS[function_name]
    memory_manager = agent._memory_manager
    if memory_manager and memory_manager.has_tool(function_name):
        return memory_manager_executor(function_name)
    if function_name == "message_agent":
        return None
    return INLINE_TOOL_EXECUTORS.get(function_name)
