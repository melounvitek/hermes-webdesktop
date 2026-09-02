"""Assemble the "learning made visible" graph for desktop.

Scoped to what a user actually learns over time: non-base, learned/profile
skills (agent-created or used) plus ``MEMORY.md`` / ``USER.md`` chunks as
first-class nodes. Skill links come from declared ``related_skills``;
memory→skill links are derived from lexical overlap.

``python -m agent.learning_graph`` prints edge-density stats against real data.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from hermes_constants import get_hermes_home

_SKIP_PARTS = {".archive", ".hub", "node_modules", ".git"}
_USAGE_TS_KEYS = ("last_activity_at", "last_used_at", "last_viewed_at", "last_patched_at", "created_at")


@dataclass
class SkillNode:
    name: str
    category: str
    source: str = "profile"
    timestamp: Optional[int] = None
    use_count: int = 0
    state: str = "active"
    created_by: Optional[str] = None
    pinned: bool = False
    related: list[str] = field(default_factory=list)


def _frontmatter(text: str) -> dict[str, Any]:
    try:
        from agent.skill_utils import parse_frontmatter

        fm, _ = parse_frontmatter(text)
        return fm or {}
    except Exception:
        return {}


def _fm_field(fm: dict[str, Any], key: str) -> Any:
    """Top-level ``key`` or ``metadata.hermes.<key>``; tolerant of the string-valued
    frontmatter that ``parse_frontmatter``'s malformed-YAML fallback produces."""
    if fm.get(key):
        return fm[key]
    meta = fm.get("metadata")
    hermes = meta.get("hermes") if isinstance(meta, dict) else None
    return hermes.get(key) if isinstance(hermes, dict) else None


def _related(fm: dict[str, Any]) -> list[str]:
    raw = _fm_field(fm, "related_skills")
    if isinstance(raw, list):
        return [str(r).strip() for r in raw if str(r).strip()]
    if isinstance(raw, str):
        return [r.strip() for r in raw.strip("[]").split(",") if r.strip()]
    return []


def _category(fm: dict[str, Any], skill_md: Path) -> str:
    cat = _fm_field(fm, "category")
    if cat:
        return str(cat)
    parts = skill_md.parts  # …/skills/<category>/<skill>/SKILL.md
    return parts[-3] if len(parts) >= 3 else "general"


def _load_usage() -> dict[str, dict[str, Any]]:
    try:
        from tools.skill_usage import load_usage

        return load_usage()
    except Exception:
        try:
            return json.loads((get_hermes_home() / "skills" / ".usage.json").read_text(encoding="utf-8"))
        except Exception:
            return {}


def _to_int_ts(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return int(value)
        s = str(value).strip()
        if not s:
            return None
        try:
            return int(float(s))
        except ValueError:
            parsed = datetime.fromisoformat(s.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return int(parsed.timestamp())
    except Exception:
        return None


def _usage_timestamp(rec: dict[str, Any]) -> Optional[int]:
    return next((ts for ts in (_to_int_ts(rec.get(k)) for k in _USAGE_TS_KEYS) if ts is not None), None)


def build_skill_nodes(skill_roots: list[tuple[str, Path]]) -> dict[str, SkillNode]:
    usage = _load_usage()
    nodes: dict[str, SkillNode] = {}
    for source, root in skill_roots:
        for skill_md in root.rglob("SKILL.md") if root.exists() else ():
            if _SKIP_PARTS.intersection(skill_md.parts):
                continue
            try:
                fm = _frontmatter(skill_md.read_text(encoding="utf-8")[:4000])
            except OSError:
                continue
            name = str(fm.get("name") or skill_md.parent.name).strip()
            if not name or name in nodes:
                continue
            rec = usage.get(name, {})
            nodes[name] = SkillNode(
                name=name,
                category=_category(fm, skill_md),
                source=source,
                timestamp=_usage_timestamp(rec) or _to_int_ts(skill_md.stat().st_mtime),
                use_count=int(rec.get("use_count", 0) or 0),
                state=str(rec.get("state", "active") or "active"),
                created_by=rec.get("created_by"),
                pinned=bool(rec.get("pinned", False)),
                related=_related(fm),
            )
    return nodes


def build_edges(nodes: dict[str, SkillNode]) -> list[tuple[str, str]]:
    """Undirected related_skills edges where BOTH endpoints exist (deduped, first-seen order)."""
    return list(dict.fromkeys(
        (min(node.name, target), max(node.name, target))
        for node in nodes.values()
        for target in node.related
        if target in nodes and target != node.name
    ))


def density_stats(nodes: dict[str, SkillNode], edges: list[tuple[str, str]]) -> dict[str, Any]:
    linked = {x for edge in edges for x in edge}
    cats = Counter(n.category for n in nodes.values())
    n = len(nodes) or 1
    return {
        "nodes": len(nodes),
        "related_edges": len(edges),
        "edges_per_node": round(len(edges) / n, 3),
        "linked_nodes": len(linked),
        "isolated_pct": round(100 * (n - len(linked)) / n, 1),
        "categories": len(cats),
        "agent_created": sum(1 for x in nodes.values() if x.created_by == "agent"),
        "used": sum(1 for x in nodes.values() if x.use_count > 0),
        "top_categories": sorted(cats.items(), key=lambda kv: -kv[1])[:8],
    }


def _memory_cards() -> list[dict[str, Any]]:
    """``MEMORY.md`` / ``USER.md`` prose split on bare ``§`` separators; every
    non-empty chunk becomes one card (MEMORY.md cards first, then USER.md)."""
    base = get_hermes_home() / "memories"
    cards: list[dict[str, Any]] = []
    for fname, source in (("MEMORY.md", "memory"), ("USER.md", "profile")):
        path = base / fname
        try:
            text = path.read_text(encoding="utf-8").strip()
            file_ts = _to_int_ts(path.stat().st_mtime)
        except OSError:
            continue
        for chunk_idx, chunk in enumerate(c.strip() for c in text.split("\n§\n")):
            if not chunk:
                continue
            first = chunk.splitlines()[0].strip().lstrip("# ").strip()
            cards.append({
                "source": source,
                "timestamp": file_ts + chunk_idx if file_ts is not None else None,
                "title": (first[:80] + "…") if len(first) > 80 else first,
                "body": chunk[:1200],
            })
    return cards


def _tokenize(text: str) -> set[str]:
    return {t for t in re.split(r"[^a-z0-9]+", text.lower()) if len(t) >= 3}


def _memory_skill_edges(memory_cards: list[dict[str, Any]], skills: list[SkillNode]) -> list[tuple[str, str]]:
    """Top-4 lexically overlapping skills per memory card (name hit weighs 6)."""
    edges: list[tuple[str, str]] = []
    skill_meta = [(s.name, _tokenize(s.name), s.name.lower()) for s in skills]
    for idx, card in enumerate(memory_cards):
        text = f"{card.get('title', '')}\n{card.get('body', '')}".lower()
        text_tokens = _tokenize(text)
        scored = []
        for name, tokens, name_lower in skill_meta:
            score = (6 if name_lower in text else 0) + len(tokens & text_tokens)
            if score > 0:
                scored.append((score, name))
        scored.sort(key=lambda x: (-x[0], x[1]))
        edges.extend((f"memory:{card['source']}:{idx}", name) for _, name in scored[:4])
    return edges


def _skill_roots() -> list[tuple[str, Path]]:
    repo = Path(__file__).resolve().parent.parent
    return [("base", repo / "skills"), ("profile", get_hermes_home() / "skills")]


def build_learning_graph() -> dict[str, Any]:
    """Full payload for the desktop learning panel: non-base skills with real
    learning signal (agent-created or used) plus memory chunks as graph nodes."""
    learned_skills = {
        name: node
        for name, node in build_skill_nodes(_skill_roots()).items()
        if node.source != "base" and (node.created_by == "agent" or node.use_count > 0)
    }
    skill_edges = build_edges(learned_skills)
    memory_cards = _memory_cards()
    memory_edges = _memory_skill_edges(memory_cards, list(learned_skills.values()))

    clusters = Counter(node.category for node in learned_skills.values())
    if memory_cards:
        clusters["memory"] = len(memory_cards)

    graph_nodes = [
        {
            "id": n.name,
            "label": n.name,
            "kind": "skill",
            "timestamp": n.timestamp,
            "category": n.category,
            "useCount": n.use_count,
            "state": n.state,
            "createdBy": n.created_by,
            "pinned": n.pinned,
        }
        for n in learned_skills.values()
    ] + [
        {
            "id": f"memory:{card['source']}:{i}",
            "label": card["title"],
            "kind": "memory",
            "memorySource": card["source"],
            "timestamp": card.get("timestamp"),
            "category": "memory",
            "useCount": 0,
            "state": "active",
            "createdBy": "memory",
            "pinned": False,
        }
        for i, card in enumerate(memory_cards)
    ]

    return {
        "nodes": graph_nodes,
        "edges": [{"source": a, "target": b} for a, b in skill_edges + memory_edges],
        "clusters": [
            {"category": c, "count": n}
            for c, n in sorted(clusters.items(), key=lambda kv: -kv[1])
        ],
        "memory": memory_cards,
        "stats": {
            **density_stats(learned_skills, skill_edges),
            "memory_nodes": len(memory_cards),
            "memory_skill_edges": len(memory_edges),
            "learned_skills": len(learned_skills),
        },
    }


if __name__ == "__main__":
    nodes = build_skill_nodes(_skill_roots())
    print(json.dumps(density_stats(nodes, build_edges(nodes)), indent=2))
