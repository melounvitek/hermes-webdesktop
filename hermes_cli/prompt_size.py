"""Prompt-size diagnostic: ``hermes prompt-size``.

The diagnostic builds a real inspection agent (so the numbers match what actually ships on the wire)
but never makes a network call: it passes dummy credentials so ``AIAgent.__init__`` takes the
direct-construction path, then calls ``build_system_prompt_parts`` / inspects ``agent.tools``
offline.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# The skills index is wrapped in this tag pair inside the stable tier.
_SKILLS_BLOCK_RE = re.compile(r"<available_skills>.*?</available_skills>", re.DOTALL)

# A rendered skill entry inside <available_skills> is ``    - name: desc`` (or
# ``    - name`` when the skill has no description). Category headers use two
# leading spaces, so the four-space + ``- `` prefix isolates skill lines.
_SKILL_LINE_PREFIX = "    - "

# Posture-demoted categories render all visible skill names on one shared line.
_NAMES_ONLY_LINE_RE = re.compile(r"^  .+ \[names only\]: (?P<names>.+)$")

# Cap the human-readable "Skills by size" table; ``--json`` always has them all.
_SKILLS_TABLE_LIMIT = 20


def _bytes(s: str) -> int:
    return len(s.encode("utf-8"))


def _tool_name(tool: Any) -> str:
    """Return the callable name of a tool schema (OpenAI ``function`` shape)."""
    if not isinstance(tool, dict):
        return ""
    fn = tool.get("function")
    if isinstance(fn, dict) and fn.get("name"):
        return str(fn["name"])
    return str(tool.get("name", ""))


def _build_inspection_agent(platform: str) -> Any:
    """Construct an offline AIAgent for prompt inspection.

    Dummy ``api_key`` + ``base_url`` force the direct-construction path (no provider
    auto-detection, no network). Toolsets and platform come from the caller so the breakdown
    matches a real session.
    """
    from run_agent import AIAgent
    from hermes_cli.config import load_config
    from hermes_cli.tools_config import _get_platform_tools

    cfg = load_config()
    model_cfg = cfg.get("model", {}) if isinstance(cfg.get("model"), dict) else {}
    model = model_cfg.get("default") or model_cfg.get("model") or ""

    # Resolve platform-specific toolsets the same way the gateway does.
    enabled_toolsets = sorted(_get_platform_tools(cfg, platform))
    agent_cfg = cfg.get("agent") or {}
    from agent.skill_utils import parse_config_string_list

    disabled_toolsets = parse_config_string_list(agent_cfg.get("disabled_toolsets")) or None

    return AIAgent(
        model=model,
        api_key="inspect-only",
        base_url="https://openrouter.ai/api/v1",
        quiet_mode=True,
        save_trajectories=False,
        platform=platform,
        enabled_toolsets=enabled_toolsets,
        disabled_toolsets=disabled_toolsets,
    )


def _skill_md_paths_by_name() -> Dict[str, Path]:
    """Map each installed skill's name to its ``SKILL.md`` path on disk.

    Keyed by both the frontmatter ``name`` (what the index renders) and the skill directory name, so
    either resolves. Local skills win over external dirs (``get_all_skills_dirs`` yields local
    first), matching the index's own precedence.
    """
    from agent.skill_utils import (
        get_all_skills_dirs,
        iter_skill_index_files,
        parse_frontmatter,
    )

    mapping: Dict[str, Path] = {}
    for skills_dir in get_all_skills_dirs():
        if not skills_dir.exists():
            continue
        for skill_file in iter_skill_index_files(skills_dir, "SKILL.md"):
            dir_name = skill_file.parent.name
            try:
                frontmatter, _ = parse_frontmatter(skill_file.read_text(encoding="utf-8"))
                frontmatter_name = str(frontmatter.get("name") or dir_name)
            except Exception:
                frontmatter_name = dir_name
            # setdefault keeps the first (local) occurrence on name collisions.
            mapping.setdefault(frontmatter_name, skill_file)
            mapping.setdefault(dir_name, skill_file)
    return mapping


def _compute_skills_breakdown(skills_block: str) -> List[Dict[str, Any]]:
    """Per-skill byte breakdown parsed from the rendered ``<available_skills>``.

    * ``index_line_bytes`` — the skill's attributed bytes in the always-on index (the fixed per-call
    cost of *listing* the skill). For a compact ``[names only]`` line, each name keeps its own bytes
    and receives an even share of the category prefix and separators.
    """
    name_to_path = _skill_md_paths_by_name()
    entries: List[Dict[str, Any]] = []

    def append_entry(
        name: str, *, attributed_bytes: int, total_bytes: int, shared_bytes: int, skill_count: int
    ) -> None:
        path = name_to_path.get(name)
        md_bytes: Optional[int] = None
        if path is not None:
            try:
                md_bytes = path.stat().st_size
            except OSError:
                pass
        entries.append({
            "name": name,
            "index_line_bytes": attributed_bytes,
            "index_line_total_bytes": total_bytes,
            "index_line_shared_bytes": shared_bytes,
            "index_line_skill_count": skill_count,
            "skill_md_bytes": md_bytes,
            "path": str(path) if path is not None else "",
        })

    for line in skills_block.splitlines():
        compact_match = _NAMES_ONLY_LINE_RE.match(line)
        line_bytes = _bytes(line)
        if compact_match is not None:
            names = [n.strip() for n in compact_match.group("names").split(",") if n.strip()]
            if not names:
                continue
            name_bytes = [_bytes(name) for name in names]
            shared_base, shared_remainder = divmod(line_bytes - sum(name_bytes), len(names))
            for index, name in enumerate(names):
                shared_bytes = shared_base + (1 if index < shared_remainder else 0)
                append_entry(
                    name, attributed_bytes=name_bytes[index] + shared_bytes,
                    total_bytes=line_bytes, shared_bytes=shared_bytes, skill_count=len(names),
                )
            continue

        if not line.startswith(_SKILL_LINE_PREFIX):
            continue
        # ``name: desc`` — the first ``": "`` separates name from description.
        # Namespaced names (``codex:rescue``) have no space after their colon,
        # so partitioning on ``": "`` keeps the full name intact.
        name = line[len(_SKILL_LINE_PREFIX):].partition(": ")[0].strip()
        if name:
            append_entry(name, attributed_bytes=line_bytes, total_bytes=line_bytes,
                         shared_bytes=0, skill_count=1)
    entries.sort(key=lambda e: (-(e["skill_md_bytes"] or 0), e["name"]))
    return entries


def _compute_toolsets_breakdown(tools: List[Any]) -> List[Dict[str, Any]]:
    """Per-toolset schema-byte breakdown of the resolved tool list.

    Each tool is attributed to its single canonical toolset so ``json_bytes`` sums are fully
    attributable (grand total = sum of per-tool serializations). Sorted largest-first, tie-broken
    by toolset name.
    """
    from tools.registry import registry

    tool_to_toolset = registry.get_tool_to_toolset_map()
    groups: Dict[str, Dict[str, Any]] = {}
    for tool in tools:
        name = _tool_name(tool)
        toolset = tool_to_toolset.get(name) or "(unknown)"
        group = groups.setdefault(toolset, {"toolset": toolset, "tool_count": 0, "json_bytes": 0})
        group["tool_count"] += 1
        group["json_bytes"] += _bytes(json.dumps(tool, ensure_ascii=False))
    return sorted(groups.values(), key=lambda g: (-g["json_bytes"], g["toolset"]))


def compute_prompt_breakdown(platform: str = "cli") -> Dict[str, Any]:
    """Return a dict of prompt-size measurements for a fresh session.

    Keys: ``system_prompt``, ``skills_index``, ``memory``, ``user_profile``, ``tools``, ``sections``
    (the three prompt tiers), ``skills_breakdown`` and ``toolsets_breakdown`` (largest-first); the
    last two answer "what should I disable to cut tokens?".
    """
    from agent.system_prompt import build_system_prompt, build_system_prompt_parts

    agent = _build_inspection_agent(platform)

    parts = build_system_prompt_parts(agent)
    full = build_system_prompt(agent)

    stable = parts.get("stable", "")
    context = parts.get("context", "")
    volatile = parts.get("volatile", "")

    # Skills index — the <available_skills> block (the largest single block
    # when many skills are installed). Lives in the volatile tier (moved from
    # stable so skill edits don't invalidate the cached identity prefix).
    skills_match = _SKILLS_BLOCK_RE.search(volatile) or _SKILLS_BLOCK_RE.search(stable)
    skills_index = skills_match.group(0) if skills_match else ""

    # Memory + user profile live in the volatile tier. We re-derive their
    # blocks directly from the memory store so the numbers are attributable
    # even though they're joined into ``volatile``.
    memory_block = ""
    user_block = ""
    store = getattr(agent, "_memory_store", None)
    if store is not None:
        try:
            if getattr(agent, "_memory_enabled", True):
                memory_block = store.format_for_system_prompt("memory") or ""
            if getattr(agent, "_user_profile_enabled", True):
                user_block = store.format_for_system_prompt("user") or ""
        except Exception:
            pass

    # Tool-schema JSON — the other half of the fixed per-call payload.
    tools = getattr(agent, "tools", None) or []

    def _size(text: str) -> Dict[str, int]:
        return {"chars": len(text), "bytes": _bytes(text)}

    sections: List[Tuple[str, int, int]] = [
        ("stable (identity/guidance/skills)", len(stable), _bytes(stable)),
        ("context (AGENTS.md/cwd files)", len(context), _bytes(context)),
        ("volatile (memory/profile/timestamp)", len(volatile), _bytes(volatile)),
    ]

    return {
        "platform": platform,
        "model": getattr(agent, "model", "") or "",
        "system_prompt": _size(full),
        "skills_index": _size(skills_index),
        "memory": _size(memory_block),
        "user_profile": _size(user_block),
        "tools": {"count": len(tools), "json_bytes": _bytes(json.dumps(tools, ensure_ascii=False))},
        "sections": sections,
        "skills_breakdown": _compute_skills_breakdown(skills_index),
        "toolsets_breakdown": _compute_toolsets_breakdown(tools),
    }


def _fmt_kb(n: int) -> str:
    return f"{n / 1024:.1f} KB"


def render_breakdown(data: Dict[str, Any]) -> str:
    """Render the breakdown as plain text suitable for a terminal."""
    sp = data["system_prompt"]
    tools = data["tools"]
    lines: List[str] = [
        f"Prompt-size breakdown (platform={data['platform']}, model={data['model'] or 'unset'})",
        "",
        f"  System prompt total : {sp['bytes']:>8,} B  ({_fmt_kb(sp['bytes'])}, {sp['chars']:,} chars)",
        "",
        "  Major blocks:",
    ]
    for label, key in (("skills index", "skills_index"), ("memory", "memory"),
                       ("user profile", "user_profile")):
        byts = data[key]["bytes"]
        lines.append(f"    {label:<19}: {byts:>8,} B  ({_fmt_kb(byts)})")
    lines += ["", "  Prompt tiers:"]
    for label, chars, byts in data["sections"]:
        lines.append(f"    {label:<36}: {byts:>8,} B  ({_fmt_kb(byts)})")
    lines += ["", f"  Tool schemas         : {tools['json_bytes']:>8,} B  ({_fmt_kb(tools['json_bytes'])}, {tools['count']} tools)"]

    # Per-toolset schema cost — which toolset's tools cost the most to ship.
    toolsets = data.get("toolsets_breakdown") or []
    if toolsets:
        lines += ["", "  Toolsets by size (tool-schema JSON, largest first):",
                  f"    {'toolset':<22} {'tools':>5}  {'schema':>10}"]
        for ts in toolsets:
            lines.append(
                f"    {ts['toolset']:<22} {ts['tool_count']:>5}  "
                f"{ts['json_bytes']:>8,} B  ({_fmt_kb(ts['json_bytes'])})"
            )

    # Per-skill cost — index line (always shipped) vs SKILL.md (read on load).
    skills = data.get("skills_breakdown") or []
    if skills:
        lines += ["",
                  "  Skills by size (SKILL.md on-disk = read cost; index cost = "
                  "attributed always-on bytes, largest first):",
                  f"    {'skill':<28} {'SKILL.md':>10}  {'index cost':>10}"]
        shown = skills[:_SKILLS_TABLE_LIMIT]
        for sk in shown:
            md = sk["skill_md_bytes"]
            md_str = f"{md:>8,} B" if md is not None else f"{'n/a':>10}"
            name = sk["name"]
            if len(name) > 28:
                name = name[:27] + "…"
            lines.append(f"    {name:<28} {md_str}  {sk['index_line_bytes']:>8,} B")
        remaining = len(skills) - len(shown)
        if remaining > 0:
            lines.append(f"    … and {remaining} more (use --json for the full list)")
    return "\n".join(lines)


def cmd_prompt_size(args: Any) -> None:
    """Entry point for ``hermes prompt-size``."""
    platform = getattr(args, "platform", "cli") or "cli"
    as_json = getattr(args, "json", False)
    try:
        data = compute_prompt_breakdown(platform)
    except Exception as e:
        print(f"Could not compute prompt-size breakdown: {e}")
        return
    if as_json:
        print(json.dumps(data, ensure_ascii=False, indent=2))
    else:
        print(render_breakdown(data))
