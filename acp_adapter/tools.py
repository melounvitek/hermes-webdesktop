"""ACP tool-call helpers for mapping hermes tools to ACP ToolKind and building content."""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Callable, Dict, List, Optional

import acp
from acp.schema import ToolCallLocation, ToolCallProgress, ToolCallStart, ToolKind

logger = logging.getLogger(__name__)

# Hermes tool name -> ACP ToolKind (anything unlisted is "other").
TOOL_KIND_MAP: Dict[str, ToolKind] = {
    name: kind
    for kind, names in {
        "read": ("read_file", "skill_view", "skills_list", "browser_snapshot", "browser_vision",
                 "browser_get_images", "vision_analyze"),
        "edit": ("write_file", "patch", "skill_manage"),
        "search": ("search_files",),
        "execute": ("terminal", "process", "execute_code", "browser_click", "browser_type", "browser_scroll",
                    "browser_press", "browser_back", "delegate_task", "image_generate", "text_to_speech"),
        "fetch": ("web_search", "web_extract", "browser_navigate"),
        "other": ("todo",),
        "think": ("_thinking",),
    }.items()
    for name in names
}

# Tools whose results render through the curated formatters below (raw JSON is
# suppressed for these); unknown/plugin tools stay conservative.
_POLISHED_TOOLS = {
    # Core operator loop
    "todo", "memory", "session_search", "delegate_task",
    # Files / execution
    "read_file", "write_file", "patch", "search_files", "terminal", "process", "execute_code",
    # Skills / web / browser / media
    "skill_view", "skills_list", "skill_manage", "web_search", "web_extract",
    "browser_navigate", "browser_click", "browser_type", "browser_press", "browser_scroll",
    "browser_back", "browser_snapshot", "browser_console", "browser_get_images", "browser_vision",
    "vision_analyze", "image_generate", "text_to_speech",
    # Schedulers / platform integrations
    "cronjob", "send_message", "clarify", "discord", "discord_admin",
    "ha_list_entities", "ha_get_state", "ha_list_services", "ha_call_service",
    "feishu_doc_read", "feishu_drive_list_comments", "feishu_drive_list_comment_replies",
    "feishu_drive_reply_comment", "feishu_drive_add_comment",
    "kanban_create", "kanban_show", "kanban_comment", "kanban_complete",
    "kanban_block", "kanban_request_review", "kanban_request_changes",
    "kanban_link", "kanban_heartbeat",
    "yb_query_group_info", "yb_query_group_members", "yb_search_sticker",
    "yb_send_dm", "yb_send_sticker",
}

_EMPTYISH = (None, "", [], {})
Args = Dict[str, Any]


def get_tool_kind(tool_name: str) -> ToolKind:
    """Return the ACP ToolKind for a hermes tool, defaulting to 'other'."""
    return TOOL_KIND_MAP.get(tool_name, "other")


def make_tool_call_id() -> str:
    return f"tc-{uuid.uuid4().hex[:12]}"


# --- small shared helpers ---------------------------------------------------


def _text(content: str) -> Any:
    return acp.tool_content(acp.text_block(content))


def _arg(args: Optional[Args], *keys: str, default: str = "") -> str:
    """First truthy ``args[key]`` as a stripped string, else ``default``."""
    a = args or {}
    value = next((a.get(k) for k in keys if a.get(k)), None)
    return str(value or default).strip() or default


def _clip(text: str, limit: int) -> str:
    """Hard-truncate to ``limit`` chars with a trailing ellipsis."""
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _fmt(value: Any, template: str, fallback: str) -> str:
    """``template.format(value)`` when value is truthy, else ``fallback``."""
    return template.format(value) if value else fallback


def _nonempty(result: Optional[str]) -> Optional[str]:
    return result if isinstance(result, str) and result.strip() else None


def _plural(count: int, word: str, suffix: str = "s") -> str:
    return f"{count} {word}{suffix if count != 1 else ''}"


def _failure(data: Args, prefix: str) -> Optional[str]:
    """Structured tool-level failure text (``success: false`` or ``error`` set)."""
    if data.get("success") is False or data.get("error"):
        return f"{prefix}: {data.get('error', 'unknown error')}"
    return None


def _args_json(arguments: Any) -> str:
    try:
        return json.dumps(arguments, indent=2, default=str)
    except (TypeError, ValueError):
        return str(arguments)


def _json_loads_maybe(value: Optional[str]) -> Any:
    """Decode a JSON string; non-strings pass through, undecodable strings yield None.

    Some Hermes tools append a human hint after the payload (``{...}\\n\\n[Hint: ...]``),
    so fall back to decoding the first JSON value to keep the structured rendering path.
    """
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except Exception:
        pass
    try:
        return json.JSONDecoder().raw_decode(value.lstrip())[0]
    except Exception:
        return None


def _truncate_text(text: str, limit: int = 5000) -> str:
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 100)] + f"\n... ({len(text)} chars total, truncated)"


def _fenced_text(text: str, language: str = "") -> str:
    """Return a Markdown fence that cannot be broken by backticks in text."""
    longest = max((len(run) for run in text.split("`")[1::2]), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{fence}{language}\n{text}\n{fence}"


def _tool_result_failed(result: Optional[str], tool_name: str | None = None) -> bool:
    """Return True when a structured Hermes tool result clearly failed.

    Deliberately conservative: plain text may legitimately contain "error", so
    only structured tool-level failures map to ACP failed status.
    """
    # The agent's tool executor wraps raised exceptions in a canonical
    # "Error executing tool '<name>': ..." prefix that well-behaved tool output
    # cannot produce; catch it so a tool that blew up is not shown green.
    if isinstance(result, str) and result.startswith("Error executing tool '"):
        return True
    data = _json_loads_maybe(result)
    if not isinstance(data, dict):
        return False
    if any(data.get(key) is False for key in ("success", "ok")):
        return True
    exit_code = data.get("exit_code", data.get("returncode"))
    if isinstance(exit_code, int) and exit_code != 0:
        return True
    # Polished tools report failures as {"error": ...} without a success flag;
    # generic/plugin payloads stay conservative so diagnostics aren't marked failed.
    return bool(tool_name in _POLISHED_TOOLS and data.get("error") and not data.get("content"))


# --- tool-call titles -------------------------------------------------------


def _title_web_extract(args: Args) -> str:
    urls = args.get("urls", [])
    if not urls:
        return "web extract"
    first = urls[0]
    if isinstance(first, dict):
        first = first.get("url") or first.get("href") or "?"
    elif not isinstance(first, str):
        first = "?"
    return f"extract: {first}" + (f" (+{len(urls)-1})" if len(urls) > 1 else "")


def _title_delegate(args: Args) -> str:
    tasks = args.get("tasks")
    if isinstance(tasks, list) and tasks:
        return f"delegate batch ({len(tasks)} tasks)"
    goal = args.get("goal", "")
    return f"delegate: {_clip(goal, 60)}" if goal else "delegate task"


def _title_execute_code(args: Args) -> str:
    first_line = next((line.strip() for line in _arg(args, "code").splitlines() if line.strip()), "")
    return _fmt(_clip(first_line, 70), "python: {}", "python code")


def _title_skill_manage(args: Args) -> str:
    name, file_path = _arg(args, "name", default="?"), _arg(args, "file_path")
    target = _clip(f"{name}/{file_path}" if file_path else name, 64)
    return f"skill {_arg(args, 'action', default='manage')}: {target}"


_TITLE_BUILDERS: Dict[str, Callable[[Args], str]] = {
    "terminal": lambda a: f"terminal: {_clip(a.get('command', ''), 80)}",
    "read_file": lambda a: f"read: {a.get('path', '?')}",
    "write_file": lambda a: f"write: {a.get('path', '?')}",
    "patch": lambda a: f"patch ({a.get('mode', 'replace')}): {a.get('path', '?')}",
    "search_files": lambda a: f"search: {a.get('pattern', '?')}",
    "web_search": lambda a: f"web search: {a.get('query', '?')}",
    "web_extract": _title_web_extract,
    "process": lambda a: _fmt(_arg(a, "session_id"), f"process {_arg(a, 'action', default='manage')}: {{}}",
                              f"process {_arg(a, 'action', default='manage')}"),
    "delegate_task": _title_delegate,
    "session_search": lambda a: _fmt(_arg(a, "query"), "session search: {}", "recent sessions"),
    "memory": lambda a: f"memory {_arg(a, 'action', default='manage')}: {_arg(a, 'target', default='memory')}",
    "execute_code": _title_execute_code,
    "todo": lambda a: f"todo ({_plural(len(a['todos']), 'item')})" if isinstance(a.get("todos"), list) else "todo",
    "skill_view": lambda a: f"skill view ({_arg(a, 'name', default='?')}{_fmt(_arg(a, 'file_path'), '/{}', '')})",
    "skills_list": lambda a: _fmt(_arg(a, "category"), "skills list ({})", "skills list"),
    "skill_manage": _title_skill_manage,
    "browser_navigate": lambda a: f"navigate: {a.get('url', '?')}",
    "browser_snapshot": lambda a: "browser snapshot",
    "browser_vision": lambda a: f"browser vision: {str(a.get('question', '?'))[:50]}",
    "browser_get_images": lambda a: "browser images",
    "vision_analyze": lambda a: f"analyze image: {str(a.get('question', '?'))[:50]}",
    "image_generate": lambda a: _fmt(_arg(a, "prompt", "description")[:50], "generate image: {}", "generate image"),
    "cronjob": lambda a: _fmt(_arg(a, "job_id", "id"), f"cron {_arg(a, 'action', default='manage')}: {{}}",
                              f"cron {_arg(a, 'action', default='manage')}"),
}


def build_tool_title(tool_name: str, args: Args) -> str:
    """Build a human-readable title for a tool call (defaults to the tool name)."""
    builder = _TITLE_BUILDERS.get(tool_name)
    return builder(args) if builder is not None else tool_name


# --- completion formatters; all share the signature (tool_name, result, args) --


def _format_todo_result(tool_name: str, result: Optional[str], args: Optional[Args]) -> Optional[str]:
    data = _json_loads_maybe(result)
    if not isinstance(data, dict) or not isinstance(data.get("todos"), list):
        return None
    summary = data.get("summary") if isinstance(data.get("summary"), dict) else {}
    icon = {"completed": "✅", "in_progress": "🔄", "pending": "⏳", "cancelled": "✗"}
    todos = [t for t in data["todos"] if isinstance(t, dict)]
    by_id = {str(t.get("id") or ""): t for t in todos}

    def _depth(item: Args) -> int:
        depth, seen, node = 0, set(), item
        while node is not None:
            parent = str(node.get("parent") or "")
            if not parent or parent not in by_id or parent in seen:
                break
            seen.add(parent)
            depth += 1
            node = by_id.get(parent)
        return min(depth, 4)

    lines = ["**Todo list**", ""]
    for item in todos:
        content = str(item.get("content") or item.get("id") or "").strip()
        if content:
            lines.append(f"{'  ' * _depth(item)}- {icon.get(str(item.get('status') or 'pending'), '•')} {content}")
    if summary:
        cancelled = summary.get("cancelled", 0)
        lines.extend([
            "",
            f"**Progress:** {summary.get('completed', 0)} completed, {summary.get('in_progress', 0)} in progress, "
            f"{summary.get('pending', 0)} pending" + (f", {cancelled} cancelled" if cancelled else ""),
        ])
    return "\n".join(lines)


def _format_read_file_result(tool_name: str, result: Optional[str], args: Optional[Args]) -> Optional[str]:
    data = _json_loads_maybe(result)
    if not isinstance(data, dict):
        return None
    if data.get("error") and not data.get("content"):
        return f"Read failed: {data.get('error')}"
    content = data.get("content")
    if not isinstance(content, str):
        return None
    a = args or {}
    range_bits = [f"from line {a['offset']}"] if a.get("offset") else []
    if a.get("limit"):
        range_bits.append(f"limit {a['limit']}")
    header = f"Read {str(a.get('path') or data.get('path') or 'file').strip()}"
    header += f" ({', '.join(range_bits)})" if range_bits else ""
    if data.get("total_lines") is not None:
        header += f" — {data.get('total_lines')} total lines"
    # read_file output is `|`-line-numbered; raw Markdown would let Zed parse
    # pipes as tables, so fence the payload to keep lines literal.
    return _truncate_text(f"{header}\n\n{_fenced_text(content)}")


def _format_search_files_result(tool_name: str, result: Optional[str], args: Optional[Args]) -> Optional[str]:
    data = _json_loads_maybe(result)
    if not isinstance(data, dict):
        return None
    files, matches = data.get("files"), data.get("matches")
    if isinstance(files, list):
        shown = min(len(files), 20)
        lines = ["File search results", f"Found {_plural(data.get('total_count', len(files)), 'file')}; showing {shown}.", ""]
        lines.extend(f"- {path}" for path in files[:shown])
        if bool(data.get("truncated")) or len(files) > shown:
            lines.extend(["", "Results truncated. Narrow the search, add path/file_glob, or use offset to page."])
        return _truncate_text("\n".join(lines), limit=7000)
    if not isinstance(matches, list):
        return None
    shown = min(len(matches), 12)
    lines = ["Search results", f"Found {_plural(data.get('total_count', len(matches)), 'match', 'es')}; showing {shown}.", ""]
    for match in matches[:shown]:
        if not isinstance(match, dict):
            lines.append(f"- {match}")
            continue
        path = str(match.get("path") or match.get("file") or match.get("filename") or "?")
        line = match.get("line") or match.get("line_number")
        content = str(match.get("content") or match.get("text") or "").strip()
        lines.append(f"- {path}:{line}" if line else f"- {path}")
        if content:
            lines.append(f"  {_truncate_text(' '.join(content.split()), 300)}")
    if bool(data.get("truncated")) or len(matches) > shown:
        lines.extend(["", "Results truncated. Narrow the search, add file_glob, or use offset to page."])
    return _truncate_text("\n".join(lines), limit=7000)


def _format_execute_code_result(tool_name: str, result: Optional[str], args: Optional[Args]) -> Optional[str]:
    data = _json_loads_maybe(result)
    if not isinstance(data, dict):
        return _nonempty(result)
    exit_code = data.get("exit_code")
    parts = [f"Exit code: {exit_code}" if exit_code is not None else "Execution complete"]
    if data.get("stdout_truncated"):
        total, captured, omitted = (data.get(k) for k in ("stdout_bytes_total", "stdout_bytes_captured", "stdout_bytes_omitted"))
        if all(isinstance(v, int) for v in (captured, total, omitted)):
            parts.extend(["", f"Output truncated: captured {captured:,} of {total:,} bytes ({omitted:,} omitted)."])
        else:
            parts.extend(["", "Output truncated."])
    for key, label, value in (
        ("warning", "Warning:", str(data.get("warning") or "").strip()),
        ("output", "Output:", str(data.get("output") or "")),
        ("error", "Error:", str(data.get("error") or "")),
    ):
        if value:
            parts.extend(["", label, value])
    return _truncate_text("\n".join(parts))


def _extract_markdown_headings(content: str, limit: int = 8) -> list[str]:
    headings: list[str] = []
    for line in content.splitlines():
        heading = line.strip().lstrip("#").strip() if line.strip().startswith("#") else ""
        if heading:
            headings.append(heading)
        if len(headings) >= limit:
            break
    return headings


def _format_skill_view_result(tool_name: str, result: Optional[str], args: Optional[Args]) -> Optional[str]:
    data = _json_loads_maybe(result)
    if not isinstance(data, dict):
        return None
    if data.get("success") is False:
        return f"Skill view failed: {data.get('error', 'unknown error')}"
    content = str(data.get("content") or "")
    linked = data.get("linked_files") if isinstance(data.get("linked_files"), dict) else None
    lines = ["**Skill loaded**", "", f"- **Name:** `{data.get('name') or 'skill'}`",
             f"- **File:** `{data.get('file') or data.get('path') or 'SKILL.md'}`"]
    description = str(data.get("description") or "").strip()
    if description:
        lines.append(f"- **Description:** {description}")
    if content:
        lines.append(f"- **Content:** {len(content):,} chars loaded into agent context")
    if linked:
        lines.append(f"- **Linked files:** {sum(len(v) for v in linked.values() if isinstance(v, list))}")
    headings = _extract_markdown_headings(content)
    if headings:
        lines.extend(["", "**Sections**", *(f"- {heading}" for heading in headings)])
    lines.extend(["", "_Full skill content is available to the agent but hidden here to keep ACP readable._"])
    return "\n".join(lines)


def _format_skill_manage_result(tool_name: str, result: Optional[str], args: Optional[Args]) -> Optional[str]:
    data = _json_loads_maybe(result)
    if not isinstance(data, dict):
        return None
    a = args or {}
    action = _arg(a, "action", default="manage")
    name = str(a.get("name") or data.get("name") or "skill").strip() or "skill"
    file_path = str(a.get("file_path") or data.get("file_path") or "SKILL.md").strip() or "SKILL.md"
    status = "✅ Skill updated" if data.get("success") is not False else "✗ Skill update failed"
    lines = [f"**{status}**", "", f"- **Action:** `{action}`", f"- **Skill:** `{name}`"]
    if action != "delete":
        lines.append(f"- **File:** `{file_path}`")
    message = str(data.get("message") or data.get("error") or "").strip()
    if message:
        lines.append(f"- **Result:** {message}")
    replacements = data.get("replacements") or data.get("replacement_count")
    if replacements is not None:
        lines.append(f"- **Replacements:** {replacements}")
    path = str(data.get("path") or "").strip()
    if path:
        lines.append(f"- **Path:** `{path}`")
    return "\n".join(lines)


def _format_web_search_result(tool_name: str, result: Optional[str], args: Optional[Args]) -> Optional[str]:
    data = _json_loads_maybe(result)
    if not isinstance(data, dict):
        return None
    web = data.get("data", {}).get("web") if isinstance(data.get("data"), dict) else data.get("web")
    if not isinstance(web, list):
        return None
    lines = [f"Web results: {len(web)}"]
    for item in web[:10]:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or "").strip()
        desc = str(item.get("description") or "").strip()
        lines.append(f"• {str(item.get('title') or item.get('url') or 'result').strip()}" + (f" — {url}" if url else ""))
        if desc:
            lines.append(f"  {desc}")
    return _truncate_text("\n".join(lines))


def _format_web_extract_result(result: Optional[str]) -> Optional[str]:
    """Return only web_extract errors for ACP; success stays compact via title."""
    data = _json_loads_maybe(result)
    if not isinstance(data, dict):
        return None
    if data.get("success") is False and data.get("error"):
        return f"Web extract failed: {data.get('error')}"
    results = data.get("results")
    if not isinstance(results, list):
        return None
    failures: list[str] = []
    for item in results[:10]:
        error = str(item.get("error") or "").strip() if isinstance(item, dict) else ""
        if not error or error in {"None", "null"}:
            continue
        url = str(item.get("url") or "").strip()
        title = str(item.get("title") or url or "Untitled").strip()
        failures.append(
            f"- {title}" + (f" — {url}" if url and url != title else "") + f"\n  Error: {_truncate_text(error, limit=500)}"
        )
    if not failures:
        return None
    return "\n".join([f"Web extract failed for {_plural(len(failures), 'URL')}", *failures])


def _format_process_result(tool_name: str, result: Optional[str], args: Optional[Args]) -> Optional[str]:
    data = _json_loads_maybe(result)
    if not isinstance(data, dict):
        return _nonempty(result)
    if data.get("success") is False and data.get("error"):
        return f"Process error: {data.get('error')}"
    action = _arg(args, "action", default="process")
    processes = data.get("processes")
    if isinstance(processes, list):
        lines = [f"Processes: {len(processes)}"]
        for proc in processes[:20]:
            if not isinstance(proc, dict):
                lines.append(f"- {proc}")
                continue
            cmd = str(proc.get("command") or "").strip()
            bits = [str(proc.get("status") or ("exited" if proc.get("exited") else "running"))]
            bits += [f"pid {proc['pid']}"] if proc.get("pid") is not None else []
            bits += [f"exit {proc['exit_code']}"] if proc.get("exit_code") is not None else []
            sid = proc.get("session_id") or proc.get("id") or "?"
            lines.append(f"- `{sid}` — {', '.join(bits)}" + (f" — {cmd[:120]}" if cmd else ""))
        if len(processes) > 20:
            lines.append(f"... {len(processes) - 20} more process(es)")
        return "\n".join(lines)

    status = str(data.get("status") or data.get("state") or action).strip()
    sid = str(data.get("session_id") or (args or {}).get("session_id") or "").strip()
    lines = [f"Process {action}: {status}" + (f" (`{sid}`)" if sid else "")]
    for key, label in (("command", "Command"), ("pid", "PID"), ("exit_code", "Exit code"), ("returncode", "Exit code"), ("lines", "Lines")):
        if data.get(key) is not None:
            lines.append(f"- **{label}:** {data.get(key)}")
    output = data.get("output") or data.get("new_output") or data.get("log") or data.get("stdout")
    error = data.get("error") or data.get("stderr")
    if output:
        lines.extend(["", "Output:", _truncate_text(str(output), limit=5000)])
    if error:
        lines.extend(["", "Error:", _truncate_text(str(error), limit=2000)])
    if data.get("message") and not output and not error:
        lines.append(str(data.get("message")))
    return _truncate_text("\n".join(lines), limit=7000)


def _format_delegate_result(tool_name: str, result: Optional[str], args: Optional[Args]) -> Optional[str]:
    data = _json_loads_maybe(result)
    if not isinstance(data, dict):
        return None
    results = data.get("results")
    if data.get("error") and not isinstance(results, list):
        return f"Delegation failed: {data.get('error')}"
    if not isinstance(results, list):
        return None
    total = data.get("total_duration_seconds")
    lines = [f"Delegation results: {_plural(len(results), 'task')}" + (f" in {total}s" if total is not None else "")]
    icon = {"completed": "✅", "failed": "✗", "error": "✗", "timeout": "⏱", "interrupted": "⚠"}
    for item in results:
        if not isinstance(item, dict):
            lines.append(f"- {item}")
            continue
        idx, status = item.get("task_index"), str(item.get("status") or "unknown")
        header = f"{icon.get(status, '•')} Task {idx + 1 if isinstance(idx, int) else '?'}: {status}"
        bits = [str(item["model"])] if item.get("model") else []
        bits += [f"role={item['_child_role']}"] if item.get("_child_role") else []
        bits += [f"{item['duration_seconds']}s"] if item.get("duration_seconds") is not None else []
        lines.extend(["", header + (" (" + ", ".join(bits) + ")" if bits else "")])
        summary, error = str(item.get("summary") or "").strip(), str(item.get("error") or "").strip()
        if summary:
            lines.append(_truncate_text(summary, limit=1200))
        if error:
            lines.append("Error: " + _truncate_text(error, limit=800))
        trace = item.get("tool_trace")
        names = [str(t.get("tool") or "?") for t in trace if isinstance(t, dict)] if isinstance(trace, list) else []
        if names:
            lines.append("Tools: " + ", ".join(names[:12]) + (f" (+{len(names)-12})" if len(names) > 12 else ""))
    return _truncate_text("\n".join(lines), limit=8000)


def _format_session_search_result(tool_name: str, result: Optional[str], args: Optional[Args]) -> Optional[str]:
    data = _json_loads_maybe(result)
    if not isinstance(data, dict):
        return None
    if data.get("success") is False:
        return f"Session search failed: {data.get('error', 'unknown error')}"
    results = data.get("results")
    if not isinstance(results, list):
        return None
    if (data.get("mode") or "search") == "recent":
        lines = ["Recent sessions"]
    else:
        lines = ["Session search results" + _fmt(data.get("query"), " for `{}`", "")]
    if not results:
        lines.append(str(data.get("message") or "No matching sessions found."))
        return "\n".join(lines)
    for item in results:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or item.get("when") or "Untitled session").strip()
        when = str(item.get("last_active") or item.get("started_at") or item.get("when") or "").strip()
        count = item.get("message_count")
        meta = ", ".join(str(x) for x in [when, str(item.get("source") or "").strip(), f"{count} msgs" if count is not None else ""] if x)
        lines.append(f"- **{title}** (`{item.get('session_id') or '?'}`)" + (f" — {meta}" if meta else ""))
        summary = str(item.get("summary") or item.get("preview") or "").strip()
        if summary:
            lines.append("  " + _truncate_text(" ".join(summary.split()), limit=500))
    return _truncate_text("\n".join(lines), limit=7000)


def _format_memory_result(tool_name: str, result: Optional[str], args: Optional[Args]) -> Optional[str]:
    data = _json_loads_maybe(result)
    if not isinstance(data, dict):
        return None
    action = _arg(args, "action", default="memory")
    target = str(data.get("target") or (args or {}).get("target") or "memory")
    if data.get("success") is False:
        lines = [f"✗ Memory {action} failed ({target})", str(data.get("error") or "unknown error")]
        matches = data.get("matches")
        if isinstance(matches, list) and matches:
            lines.extend(["Matches:", *(f"- {_truncate_text(str(m), 160)}" for m in matches[:5])])
        return "\n".join(lines)
    lines = [f"✅ Memory {action} saved ({target})"]
    if data.get("message"):
        lines.append(str(data.get("message")))
    if data.get("entry_count") is not None:
        lines.append(f"Entries: {data.get('entry_count')}")
    if data.get("usage"):
        lines.append(f"Usage: {data.get('usage')}")
    # Never dump all memory entries into the ACP UI; only preview the new value.
    preview = _arg(args, "content", "old_text")
    if preview:
        lines.append("Preview: " + _truncate_text(preview, limit=300))
    return "\n".join(lines)


def _format_edit_result(tool_name: str, result: Optional[str], args: Optional[Args]) -> Optional[str]:
    data = _json_loads_maybe(result)
    path = str((args or {}).get("path") or "file").strip()
    done = f"✅ {tool_name} completed" + (f" for `{path}`" if path else "")
    if not isinstance(data, dict):
        text = _nonempty(result)
        return _truncate_text(text, limit=3000) if text else done
    failed = _failure(data, f"{tool_name} failed for {path}")
    if failed:
        return failed
    lines = [done]
    message = str(data.get("message") or "").strip()
    if message:
        lines.append(message)
    replacements = data.get("replacements") or data.get("replacement_count")
    if replacements is not None:
        lines.append(f"Replacements: {replacements}")
    files = data.get("files_modified")
    if files and isinstance(files, list):
        lines.append("Files: " + ", ".join(f"`{f}`" for f in files[:8]))
    return "\n".join(lines)


def _format_browser_result(tool_name: str, result: Optional[str], args: Optional[Args]) -> Optional[str]:
    data = _json_loads_maybe(result)
    if not isinstance(data, dict):
        return _nonempty(result)
    failed = _failure(data, f"{tool_name} failed")
    if failed:
        return failed
    images = (data.get("images") or data.get("data")) if tool_name == "browser_get_images" else None
    if isinstance(images, list):
        lines = [f"Images found: {len(images)}"]
        for img in images[:12]:
            if isinstance(img, dict):
                url = str(img.get("url") or img.get("src") or "").strip()
                lines.append(f"- {str(img.get('alt') or '').strip() or 'image'}" + (f" — {url}" if url else ""))
        return _truncate_text("\n".join(lines), limit=5000)
    title = str(data.get("title") or data.get("url") or data.get("status") or tool_name)
    text = str(data.get("text") or data.get("content") or data.get("snapshot") or data.get("analysis") or data.get("message") or "").strip()
    lines = [title]
    if data.get("url") and data.get("url") != title:
        lines.append(str(data.get("url")))
    if text:
        lines.extend(["", _truncate_text(text, limit=5000)])
    return _truncate_text("\n".join(lines), limit=7000)


def _format_media_or_cron_result(tool_name: str, result: Optional[str], args: Optional[Args]) -> Optional[str]:
    data = _json_loads_maybe(result)
    if not isinstance(data, dict):
        return _nonempty(result)
    failed = _failure(data, f"{tool_name} failed")
    if failed:
        return failed
    keys = ("file_path", "path", "url", "image_url", "job_id", "id", "status", "message", "next_run")
    return "\n".join([f"✅ {tool_name} completed", *(f"- **{k}:** {data.get(k)}" for k in keys if data.get(k))])


def _format_structured_value(key: str, value: Any, *, indent: int = 0, max_depth: int = 3, max_items: int = 8) -> List[str]:
    """Render nested JSON-ish values as compact Markdown bullets, not inline blobs."""
    pad = "  " * indent
    bullet = f"{pad}- "
    label = f"**{key}:**" if key else ""

    def _line(text: str) -> str:
        return f"{bullet}{label} {text}" if label else f"{bullet}{text}"

    def _child(child_key: str, child_value: Any, extra_indent: int) -> List[str]:
        return _format_structured_value(
            child_key, child_value, indent=indent + extra_indent, max_depth=max_depth - 1, max_items=max_items,
        )

    if value in _EMPTYISH:
        return []
    if max_depth <= 0:
        preview = json.dumps(value, ensure_ascii=False, default=str) if isinstance(value, (dict, list)) else str(value)
        return [_line(_truncate_text(preview, limit=240))]

    if isinstance(value, dict):
        lines = [f"{bullet}{label}" if label else f"{bullet}{len(value)} fields"]
        shown = 0
        for child_key, child_value in value.items():
            if child_value in _EMPTYISH:
                continue
            lines.extend(_child(str(child_key), child_value, 1))
            shown += 1
            if shown >= max_items:
                if len(value) > shown:
                    lines.append(f"{pad}  - ... {len(value) - shown} more fields")
                break
        return lines

    if isinstance(value, list):
        lines = [_line(_plural(len(value), "item"))]
        for idx, item in enumerate(value[:max_items], 1):
            if isinstance(item, dict):
                headline = str(
                    item.get("content") or item.get("message") or item.get("title") or item.get("name") or item.get("id") or ""
                ).strip()
                if headline:
                    lines.append(f"{pad}  {idx}. {_truncate_text(headline, limit=220)}")
                    for child_key in ("id", "status", "type", "scope", "quality_score", "score", "path", "url"):
                        if item.get(child_key) not in _EMPTYISH:
                            lines.append(f"{pad}    - **{child_key}:** {_truncate_text(str(item[child_key]), limit=180)}")
                else:
                    lines.append(f"{pad}  {idx}.")
                    for child_key, child_value in list(item.items())[:max_items]:
                        lines.extend(_child(str(child_key), child_value, 2))
            elif isinstance(item, list):
                lines.append(f"{pad}  {idx}. {len(item)} items")
                for nested in item[:max_items]:
                    lines.extend(_child("", nested, 2))
            else:
                lines.append(f"{pad}  {idx}. {_truncate_text(str(item), limit=240)}")
        if len(value) > max_items:
            lines.append(f"{pad}  ... {len(value) - max_items} more items")
        return lines

    return [_line(_truncate_text(str(value), limit=500))]


_PRIORITY_KEYS = (
    "message", "status", "id", "task_id", "issue_id", "title", "name", "entity_id",
    "state", "service", "url", "path", "file_path", "count", "total", "next_run",
)


def _format_generic_structured_result(tool_name: str, result: Optional[str], *, fallback_to_text: bool = True) -> Optional[str]:
    data = _json_loads_maybe(result)
    if not isinstance(data, (dict, list)):
        return _nonempty(result) if fallback_to_text else None
    if isinstance(data, list):
        lines = [f"{tool_name}: {_plural(len(data), 'item')}"]
        for item in data[:12]:
            if isinstance(item, (dict, list)):
                lines.extend(_format_structured_value("", item, indent=0, max_depth=2, max_items=6))
            else:
                lines.append(f"- {_truncate_text(str(item), limit=240)}")
        if len(data) > 12:
            lines.append(f"... {len(data) - 12} more items")
        return _truncate_text("\n".join(lines), limit=5000)

    failed = _failure(data, f"{tool_name} failed")
    if failed:
        return failed
    lines = [f"✅ {tool_name} completed" if data.get("success") is True else f"{tool_name} result"]
    seen = {key for key in _PRIORITY_KEYS if data.get(key) not in _EMPTYISH}
    lines.extend(f"- **{key}:** {_truncate_text(str(data[key]), limit=500)}" for key in _PRIORITY_KEYS if key in seen)
    for key, value in data.items():
        if key in seen or key in {"success", "raw", "content", "entries"} or value in _EMPTYISH:
            continue
        lines.extend(_format_structured_value(str(key), value, indent=0, max_depth=3, max_items=8))
        if len(lines) >= 40:
            lines.append("- ... more fields truncated")
            break
    content = data.get("content")
    if isinstance(content, str) and content.strip():
        lines.extend(["", _truncate_text(content.strip(), limit=1500)])
    return _truncate_text("\n".join(lines), limit=7000)


_Formatter = Callable[[str, Optional[str], Optional[Args]], Optional[str]]

_COMPLETION_FORMATTERS: Dict[str, _Formatter] = {
    "todo": _format_todo_result,
    "read_file": _format_read_file_result,
    "write_file": _format_edit_result,
    "patch": _format_edit_result,
    "search_files": _format_search_files_result,
    "execute_code": _format_execute_code_result,
    "process": _format_process_result,
    "delegate_task": _format_delegate_result,
    "session_search": _format_session_search_result,
    "memory": _format_memory_result,
    "skill_view": _format_skill_view_result,
    "skill_manage": _format_skill_manage_result,
    "web_search": _format_web_search_result,
    "web_extract": lambda t, r, a: _format_web_extract_result(r),
    **{n: _format_browser_result for n in ("browser_navigate", "browser_snapshot", "browser_vision", "browser_get_images")},
    **{n: _format_media_or_cron_result for n in ("vision_analyze", "image_generate", "cronjob")},
}


def _build_polished_completion_content(tool_name: str, result: Optional[str], function_args: Optional[Args]) -> Optional[List[Any]]:
    formatter = _COMPLETION_FORMATTERS.get(tool_name)
    if formatter is not None:
        text = formatter(tool_name, result, function_args)
    else:
        text = _format_generic_structured_result(tool_name, result, fallback_to_text=tool_name in _POLISHED_TOOLS)
    return [_text(text)] if text else None


def _parse_unified_diff_content(diff_text: str) -> List[Any]:
    """Convert unified diff text into ACP diff content blocks (one per ``---``/``+++`` pair)."""
    content: List[Any] = []
    if not diff_text:
        return content
    state: Dict[str, Any] = {"old": None, "new": None, "old_lines": [], "new_lines": []}

    def _flush() -> None:
        old_path, new_path = state["old"], state["new"]
        if old_path is not None or new_path is not None:
            path = new_path if new_path and new_path != "/dev/null" else old_path
            if path and path != "/dev/null":
                path = str(path).strip()
                content.append(acp.tool_diff_content(
                    path=path[2:] if path.startswith(("a/", "b/")) else path,
                    old_text="\n".join(state["old_lines"]) if state["old_lines"] else None,
                    new_text="\n".join(state["new_lines"]),
                ))
        state.update(old=None, new=None, old_lines=[], new_lines=[])

    for line in diff_text.splitlines():
        if line.startswith("--- "):
            _flush()
            state["old"] = line[4:].strip()
        elif line.startswith("+++ "):
            state["new"] = line[4:].strip()
        elif line.startswith("@@") or (state["old"] is None and state["new"] is None):
            continue
        elif line.startswith("+"):
            state["new_lines"].append(line[1:])
        elif line.startswith("-"):
            state["old_lines"].append(line[1:])
        elif line.startswith(" "):
            state["old_lines"].append(line[1:])
            state["new_lines"].append(line[1:])
    _flush()
    return content


def _build_tool_complete_content(
    tool_name: str, result: Optional[str], *, function_args: Optional[Args] = None, snapshot: Any = None
) -> List[Any]:
    """Build structured ACP completion content, falling back to plain text."""
    if tool_name == "skill_manage":
        try:
            from agent.display import extract_edit_diff

            diff_text = extract_edit_diff(tool_name, result, function_args=function_args, snapshot=snapshot)
            if isinstance(diff_text, str) and diff_text.strip():
                diff_content = _parse_unified_diff_content(diff_text)
                if diff_content:
                    return diff_content
        except Exception:
            pass
    return _build_polished_completion_content(tool_name, result, function_args) or [_text(_truncate_text(result or ""))]


# --- ToolCallStart / ToolCallProgress events ---------------------------------


def _start_todo(args: Args) -> List[Any]:
    items = args.get("todos")
    if not isinstance(items, list):
        return [_text("Reading todo list")]
    lines = ["Updating todo list", ""]
    lines.extend(f"- {i.get('status', 'pending')}: {i.get('content', i.get('id', ''))}" for i in items[:8] if isinstance(i, dict))
    if len(items) > 8:
        lines.append(f"... {len(items) - 8} more")
    return [_text("\n".join(lines))]


def _start_skill_manage(args: Args) -> List[Any]:
    action = _arg(args, "action", default="manage")
    name = _arg(args, "name", default="?")
    file_path = _arg(args, "file_path", default="SKILL.md")
    path = f"skills/{name}/{file_path}"
    if action == "patch":
        old = str(args.get("old_string") or "")
        return [acp.tool_diff_content(path=path, old_text=old or None, new_text=str(args.get("new_string") or ""))]
    if action in {"edit", "create"}:
        return [acp.tool_diff_content(path=path, new_text=str(args.get("content") or ""))]
    if action == "write_file":
        target = str(args.get("file_path") or "file")
        return [acp.tool_diff_content(path=f"skills/{name}/{target}", new_text=str(args.get("file_content") or ""))]
    if action in {"delete", "remove_file"}:
        return [_text(f"Removing {str(args.get('file_path') or file_path)} from skill '{name}'")]
    return [_text(f"Running skill_manage action '{action}' on skill '{name}' ({file_path})")]


def _start_execute_code(args: Args) -> List[Any]:
    code = _arg(args, "code")
    preview = code[:1200] + (f"\n... ({len(code)} chars total, truncated)" if len(code) > 1200 else "")
    return [_text(_fmt(preview, "Running Python helper script:\n\n```python\n{}\n```", "Running Python helper script"))]


def _start_process(args: Args) -> List[Any]:
    text = f"Process action: {_arg(args, 'action', default='manage')}" + _fmt(_arg(args, "session_id"), "\nSession: {}", "")
    data_preview = _arg(args, "data")
    return [_text(text + (f"\nInput: {_truncate_text(data_preview, limit=500)}" if data_preview else ""))]


def _start_delegate(args: Args) -> List[Any]:
    tasks = args.get("tasks")
    if not (isinstance(tasks, list) and tasks):
        goal = _arg(args, "goal")
        return [_text("Delegating task" + (f":\n{_truncate_text(goal, limit=800)}" if goal else ""))]
    lines = [f"Delegating {len(tasks)} tasks", ""]
    for i, task in enumerate(tasks[:8], 1):
        if isinstance(task, dict):
            lines.append(f"{i}. " + _truncate_text(_arg(task, "goal"), limit=160) + _fmt(_arg(task, "role"), " ({})", ""))
    if len(tasks) > 8:
        lines.append(f"... {len(tasks) - 8} more")
    return [_text("\n".join(lines))]


def _start_memory(args: Args) -> List[Any]:
    text = f"Memory {_arg(args, 'action', default='manage')} ({_arg(args, 'target', default='memory')})"
    preview = _arg(args, "content", "old_text")
    return [_text(text + (f"\nPreview: {_truncate_text(preview, limit=500)}" if preview else ""))]


# Per-tool start-content builders. ``None`` means the title/location already
# identify the target (read_file, web_extract): a synthetic content block would
# make Zed render an unhelpful Output section before completion.
_START_CONTENT_BUILDERS: Dict[str, Optional[Callable[[Args], List[Any]]]] = {
    "patch": lambda a: [_text(
        f"Preparing {a.get('mode', 'replace')} edit for {a.get('path') or 'patch input'}. Approval prompt shows the diff."
    )],
    "write_file": lambda a: [_text(_fmt(a.get("path", ""), "Preparing write to {}. Approval prompt shows the diff.",
                                        "Preparing file write. Approval prompt shows the diff."))],
    "terminal": lambda a: [_text(f"$ {a.get('command', '')}")],
    "read_file": None,
    "search_files": lambda a: [_text(
        f"Searching for '{a.get('pattern', '')}' ({a.get('target', 'content')})" + _fmt(a.get("path"), " in {}", "")
    )],
    "todo": _start_todo,
    "skill_view": lambda a: [_text(f"Loading skill '{_arg(a, 'name', default='?')}' ({_arg(a, 'file_path', default='SKILL.md')})")],
    "skill_manage": _start_skill_manage,
    "execute_code": _start_execute_code,
    "web_search": lambda a: [_text(_fmt(_arg(a, "query"), "Searching the web for: {}", "Searching the web"))],
    "web_extract": None,
    "process": _start_process,
    "delegate_task": _start_delegate,
    "session_search": lambda a: [_text(_fmt(_arg(a, "query"), "Searching past sessions for: {}", "Loading recent sessions"))],
    "memory": _start_memory,
}


def build_tool_start(tool_call_id: str, tool_name: str, arguments: Args, *, edit_diff: Any = None) -> ToolCallStart:
    """Create a ToolCallStart event for the given hermes tool invocation.

    A malformed argument (e.g. a non-string ``command``/``path`` from a model
    ignoring the schema) must never abort the render — this runs on the live
    tool-progress callback and during history replay — so any failure in the
    title/content/location builders falls back to a minimal valid start event
    (mirrors ``get_cute_tool_message`` in ``agent/display.py``).
    """
    try:
        return _build_tool_start(tool_call_id, tool_name, arguments, edit_diff=edit_diff)
    except Exception as exc:  # noqa: BLE001 — a tool-call render must never abort the turn
        logger.debug("ACP tool-start render failed for %r: %s", tool_name, exc)
        safe_name = tool_name if isinstance(tool_name, str) and tool_name else "tool"
        return acp.start_tool_call(tool_call_id, safe_name, kind=get_tool_kind(safe_name), content=None, locations=[])


def _build_tool_start(tool_call_id: str, tool_name: str, arguments: Args, *, edit_diff: Any = None) -> ToolCallStart:
    """Build the ToolCallStart event (unguarded; see ``build_tool_start``)."""
    raw_input = None
    if tool_name in ("patch", "write_file") and edit_diff is not None:
        content = [acp.tool_diff_content(path=edit_diff.path, old_text=edit_diff.old_text, new_text=edit_diff.new_text)]
    elif tool_name in _START_CONTENT_BUILDERS:
        builder = _START_CONTENT_BUILDERS[tool_name]
        content = builder(arguments) if builder is not None else None
    elif tool_name in _POLISHED_TOOLS:
        content = [_text(_truncate_text(_args_json(arguments), limit=1200))]
    elif not arguments:
        content = None
    else:  # unknown tool with arguments: echo them as content and raw_input
        content = [_text(_args_json(arguments))]
        raw_input = arguments
    return acp.start_tool_call(
        tool_call_id, build_tool_title(tool_name, arguments), kind=get_tool_kind(tool_name),
        content=content, locations=extract_locations(arguments), raw_input=raw_input,
    )


def build_tool_complete(
    tool_call_id: str,
    tool_name: str,
    result: Optional[str] = None,
    function_args: Optional[Args] = None,
    snapshot: Any = None,
) -> ToolCallProgress:
    """Create a ToolCallUpdate (progress) event for a completed tool call."""
    if tool_name == "web_extract":
        error_text = _format_web_extract_result(result)
        content = [_text(error_text)] if error_text else None
    else:
        content = _build_tool_complete_content(tool_name, result, function_args=function_args, snapshot=snapshot)
    structured = isinstance(_json_loads_maybe(result), (dict, list))
    return acp.update_tool_call(
        tool_call_id,
        kind=get_tool_kind(tool_name),
        status="failed" if _tool_result_failed(result, tool_name) else "completed",
        content=content,
        raw_output=None if tool_name in _POLISHED_TOOLS or structured else result,
    )


def extract_locations(arguments: Args) -> List[ToolCallLocation]:
    """Extract file-system locations from tool arguments."""
    path = arguments.get("path")
    if not path:
        return []
    return [ToolCallLocation(path=path, line=arguments.get("offset") or arguments.get("line"))]
