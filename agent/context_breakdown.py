"""Live session context-window breakdown for UI surfaces.

Estimates how the next provider request is composed: system prompt tiers,
tool schemas, and conversation history. Uses the same rough char/4 heuristic
as ``agent.model_metadata.estimate_request_tokens_rough`` so numbers align
with compression thresholds — not exact tokenizer counts.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

_SKILLS_BLOCK_RE = re.compile(r"<available_skills>.*?</available_skills>", re.DOTALL)

_SUBAGENT_TOOL_NAMES = frozenset({"delegate_task"})

_CATEGORY_COLORS = {
    "system_prompt": "var(--context-usage-system)",
    "tool_definitions": "var(--context-usage-tools)",
    "rules": "var(--context-usage-rules)",
    "skills": "var(--context-usage-skills)",
    "mcp": "var(--context-usage-mcp)",
    "subagent_definitions": "var(--context-usage-subagents)",
    "memory": "var(--context-usage-memory)",
    "conversation": "var(--context-usage-conversation)",
}


def _chars_to_tokens(text: str) -> int:
    return (len(text) + 3) // 4 if text else 0


def _json_tokens(value: Any) -> int:
    return _chars_to_tokens(json.dumps(value, ensure_ascii=False)) if value else 0


def _bytes_to_tokens(size: Optional[int]) -> Optional[int]:
    return None if size is None else (int(size) + 3) // 4


def _tool_name(tool: dict) -> str:
    fn = tool.get("function") if isinstance(tool, dict) else None
    if isinstance(fn, dict):
        return str(fn.get("name") or "")
    return str(tool.get("name") or "")


def _skills_block(stable: str) -> str:
    """The live ``<available_skills>`` block inside the stable tier, or ''."""
    m = _SKILLS_BLOCK_RE.search(stable)
    return m.group(0) if m else ""


def _split_tools(tools: Sequence[dict]) -> Tuple[List[dict], List[dict], List[dict]]:
    builtin: List[dict] = []
    mcp: List[dict] = []
    subagent: List[dict] = []
    for tool in tools:
        name = _tool_name(tool)
        if name.startswith("mcp_"):
            mcp.append(tool)
        elif name in _SUBAGENT_TOOL_NAMES:
            subagent.append(tool)
        else:
            builtin.append(tool)
    return builtin, mcp, subagent


def _memory_blocks(agent: Any) -> Tuple[str, str]:
    memory_block = user_block = ""
    store = getattr(agent, "_memory_store", None)
    if store is None:
        return memory_block, user_block
    try:
        if getattr(agent, "_memory_enabled", True):
            memory_block = store.format_for_system_prompt("memory") or ""
        if getattr(agent, "_user_profile_enabled", True):
            user_block = store.format_for_system_prompt("user") or ""
    except Exception:
        pass
    return memory_block, user_block


def _strip_blocks(text: str, *blocks: str) -> str:
    out = text
    for block in blocks:
        if block:
            out = out.replace(block, "")
    return out.strip()


def compute_session_context_breakdown(
    agent: Any,
    messages: Optional[List[dict]] = None,
) -> Dict[str, Any]:
    """Return a Cursor-style context usage breakdown for one live agent."""
    from agent.model_metadata import estimate_messages_tokens_rough
    from agent.system_prompt import build_system_prompt_parts

    parts = build_system_prompt_parts(agent)
    stable = parts.get("stable", "") or ""
    context = parts.get("context", "") or ""
    volatile = parts.get("volatile", "") or ""
    skills_index = _skills_block(stable)

    memory_block, user_block = _memory_blocks(agent)
    memory_text = "\n\n".join(part for part in (memory_block, user_block) if part).strip()

    system_core = _strip_blocks(stable, skills_index)
    system_tail = _strip_blocks(volatile, memory_block, user_block)
    system_prompt_text = "\n\n".join(part for part in (system_core, system_tail) if part).strip()

    builtin_tools, mcp_tools, subagent_tools = _split_tools(list(getattr(agent, "tools", None) or []))
    categories = [
        ("system_prompt", "System prompt", _chars_to_tokens(system_prompt_text)),
        ("tool_definitions", "Tool definitions", _json_tokens(builtin_tools)),
        ("rules", "Rules", _chars_to_tokens(context)),
        ("skills", "Skills", _chars_to_tokens(skills_index)),
        ("mcp", "MCP", _json_tokens(mcp_tools)),
        ("subagent_definitions", "Subagent definitions", _json_tokens(subagent_tools)),
        ("memory", "Memory", _chars_to_tokens(memory_text)),
        ("conversation", "Conversation", estimate_messages_tokens_rough(messages or [])),
    ]
    estimated_total = sum(tokens for _, _, tokens in categories)

    comp = getattr(agent, "context_compressor", None)
    context_max = int(getattr(comp, "context_length", 0) or 0) if comp else 0
    # Usage-anchored figure (provider-exact tokens of a response + delta of what
    # was appended since) beats last_prompt_tokens (lags) and the heuristic.
    # Prefer the turn-base anchor: on reasoning models later same-turn
    # responses inflate prompt_tokens with replayed thinking that evaporates at
    # the turn boundary, so anchoring on the LAST response makes the meter
    # sawtooth.  Fall back to last-response anchor, then measured/estimated.
    from agent.model_metadata import anchored_context_tokens

    anchored_used = anchored_context_tokens(
        messages or [],
        getattr(agent, "_turn_base_usage_anchor", None),
        charge_stale_thinking=False,
    )
    if anchored_used is None:
        anchored_used = anchored_context_tokens(messages or [], getattr(agent, "_usage_anchor", None))
    measured_used = int(getattr(comp, "last_prompt_tokens", 0) or 0) if comp else 0
    if anchored_used is not None:
        context_used = anchored_used
    else:
        context_used = measured_used if measured_used > 0 else estimated_total
    context_percent = max(0, min(100, round(context_used / context_max * 100))) if context_max else 0

    return {
        "categories": [
            {
                "color": _CATEGORY_COLORS.get(category_id, "var(--ui-text-tertiary)"),
                "id": category_id,
                "label": label,
                "tokens": tokens,
            }
            for category_id, label, tokens in categories
            if tokens > 0
        ],
        "context_max": context_max,
        "context_percent": context_percent,
        "context_used": context_used,
        "estimated_total": estimated_total,
        "model": getattr(agent, "model", "") or "",
    }


# ── /context rendering (CLI + gateway) ──────────────────────────────────────
# Pure text renderers over the payload above.  The gateway skips the glyph grid
# (monospace is not guaranteed on messaging platforms).

_CATEGORY_GLYPHS = {
    "system_prompt": "■",
    "tool_definitions": "▣",
    "rules": "▩",
    "skills": "▤",
    "mcp": "▥",
    "subagent_definitions": "▦",
    "memory": "▧",
    "conversation": "▨",
}
_FREE_GLYPH = "·"
_GRID_COLUMNS = 20
_GRID_ROWS = 5  # 100 cells → 1 cell per percent of the context window
_DETAILS_TABLE_LIMIT = 15  # display cap only; the underlying data keeps everything


def compute_context_details(agent: Any) -> Dict[str, Any]:
    """Expanded per-skill / per-toolset cost listing for ``/context all``,
    reusing the ``hermes prompt-size`` attribution (index-line bytes from the
    live skills block; schema bytes via the registry's tool→toolset map)."""
    from hermes_cli.prompt_size import (
        _compute_skills_breakdown,
        _compute_toolsets_breakdown,
    )
    from agent.system_prompt import build_system_prompt_parts

    skills_block = _skills_block(build_system_prompt_parts(agent).get("stable", "") or "")
    skills = [
        {
            "name": entry.get("name", ""),
            "index_tokens": _bytes_to_tokens(entry.get("index_line_bytes")) or 0,
            "skill_md_tokens": _bytes_to_tokens(entry.get("skill_md_bytes")),
        }
        for entry in (_compute_skills_breakdown(skills_block) if skills_block else [])
    ]
    tools = list(getattr(agent, "tools", None) or [])
    toolsets = [
        {
            "toolset": group.get("toolset", ""),
            "tool_count": int(group.get("tool_count", 0) or 0),
            "schema_tokens": _bytes_to_tokens(group.get("json_bytes")) or 0,
        }
        for group in (_compute_toolsets_breakdown(tools) if tools else [])
    ]
    return {"skills": skills, "toolsets": toolsets}


def render_context_grid(payload: Dict[str, Any]) -> List[str]:
    """Glyph block grid: 100 cells, one per percent of the context window;
    categories fill in declaration order, the remainder is free space."""
    context_max = int(payload.get("context_max") or 0)
    categories = payload.get("categories") or []
    total_cells = _GRID_COLUMNS * _GRID_ROWS

    cells: List[str] = []
    if context_max > 0:
        for cat in categories:
            tokens = int(cat.get("tokens") or 0)
            n = round(tokens / context_max * total_cells)
            if tokens > 0 and n == 0:
                n = 1  # never render a nonzero category as invisible
            glyph = _CATEGORY_GLYPHS.get(str(cat.get("id") or ""), "▪")
            cells.extend([glyph] * n)
        cells = cells[:total_cells]
    cells.extend([_FREE_GLYPH] * (total_cells - len(cells)))

    return [
        " ".join(cells[row * _GRID_COLUMNS:(row + 1) * _GRID_COLUMNS])
        for row in range(_GRID_ROWS)
    ]


def render_context_category_lines(payload: Dict[str, Any]) -> List[str]:
    """Render the 'Estimated usage by category' table as plain-text lines."""
    categories = payload.get("categories") or []
    context_max = int(payload.get("context_max") or 0)
    estimated_total = int(payload.get("estimated_total") or 0)
    denom = context_max or estimated_total

    lines = ["Estimated usage by category"]
    if not categories:
        lines.append("  (no data yet — send a message first)")
        return lines

    width = max(len(str(cat.get("label") or "")) for cat in categories)
    width = max(width, len("Free space"))
    for cat in categories:
        tokens = int(cat.get("tokens") or 0)
        glyph = _CATEGORY_GLYPHS.get(str(cat.get("id") or ""), "▪")
        pct = tokens / denom * 100 if denom else 0.0
        label = str(cat.get("label") or cat.get("id") or "")
        lines.append(f"{glyph} {label:<{width}} {tokens:>9,} tokens {pct:>5.1f}%")
    if context_max > 0:
        free = max(0, context_max - estimated_total)
        pct = free / context_max * 100
        lines.append(f"{_FREE_GLYPH} {'Free space':<{width}} {free:>9,} tokens {pct:>5.1f}%")
    return lines


def _append_overflow(lines: List[str], count: int) -> None:
    remaining = count - _DETAILS_TABLE_LIMIT
    if remaining > 0:
        lines.append(f"  … and {remaining} more")


def render_context_details_lines(details: Dict[str, Any]) -> List[str]:
    """Render the expanded ``/context all`` per-skill / per-toolset tables."""
    lines: List[str] = []

    toolsets = details.get("toolsets") or []
    if toolsets:
        lines.append("Toolsets by schema cost (largest first)")
        for group in toolsets[:_DETAILS_TABLE_LIMIT]:
            lines.append(
                f"  {group['toolset']:<24} {group['tool_count']:>3} tools"
                f" {group['schema_tokens']:>8,} tokens"
            )
        _append_overflow(lines, len(toolsets))

    skills = details.get("skills") or []
    if skills:
        if lines:
            lines.append("")
        lines.append("Skills by cost (index = always-on; SKILL.md = cost when loaded)")
        for entry in skills[:_DETAILS_TABLE_LIMIT]:
            name = str(entry.get("name") or "")
            if len(name) > 28:
                name = name[:27] + "…"
            md = entry.get("skill_md_tokens")
            md_str = f"{md:>8,}" if md is not None else f"{'n/a':>8}"
            lines.append(
                f"  {name:<28} index {entry['index_tokens']:>6,}"
                f"  SKILL.md {md_str} tokens"
            )
        _append_overflow(lines, len(skills))

    return lines


def render_context_breakdown_lines(
    payload: Dict[str, Any],
    *,
    details: Optional[Dict[str, Any]] = None,
    grid: bool = True,
) -> List[str]:
    """Full /context view.  ``grid`` prepends the glyph grid (CLI; the gateway
    keeps its own gauge); ``details`` appends the expanded listings."""
    lines: List[str] = []
    if grid:
        lines.extend(render_context_grid(payload))
        lines.append("")
    lines.extend(render_context_category_lines(payload))

    context_max = int(payload.get("context_max") or 0)
    context_used = int(payload.get("context_used") or 0)
    if context_max > 0:
        pct = int(payload.get("context_percent") or 0)
        lines.append("")
        lines.append(f"Context window: {context_used:,} / {context_max:,} tokens ({pct}%)")

    if details is not None:
        detail_lines = render_context_details_lines(details)
        if detail_lines:
            lines.append("")
            lines.extend(detail_lines)
    else:
        lines.append("")
        lines.append("Use /context all for per-skill and per-toolset costs.")
    return lines
