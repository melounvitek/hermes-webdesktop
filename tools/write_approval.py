#!/usr/bin/env python3
"""Write-approval gate + pending store for memory and skill writes.

The agent writes to two cross-session stores — **memory** (MEMORY.md / USER.md,
small entries) and **skills** (SKILL.md + files, potentially 10-100 KB) — from
two origins: **foreground** (a normal turn) and **background_review** (the
autonomous self-improvement fork). A per-subsystem boolean ``write_approval``
gates those writes: ``false`` (default) writes freely; ``true`` never commits
directly — it prompts inline (memory, interactive CLI only) or **stages** the
write to a pending store for out-of-band review.

Staging is mandatory for background writes (a daemon thread cannot block on a
prompt), gateway sessions (no inline channel — review via ``/memory pending``),
and all skill writes (too big to eyeball mid-loop). Memory shows full content;
skills show metadata + a gist + a ``diff`` escape hatch.

Pending records live under ``<HERMES_HOME>/pending/{memory,skills}/<id>.json``
so they survive restarts and can be reviewed from CLI, gateway, or dashboard.
"""

from __future__ import annotations

import difflib
import json
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

# Subsystem identifiers
MEMORY = "memory"
SKILLS = "skills"
_SUBSYSTEMS = (MEMORY, SKILLS)

# Per-subsystem config key. Intentionally a single boolean with no "block all
# writes" state — to disable a subsystem use its own enable flag
# (e.g. ``memory.memory_enabled: false``).
CONFIG_KEY = "write_approval"


# --- Config resolution ---

def write_approval_enabled(subsystem: str) -> bool:
    """Read ``<subsystem>.write_approval``; any unset/invalid value means gate off."""
    if subsystem not in _SUBSYSTEMS:
        return False
    try:
        from hermes_cli.config import load_config, cfg_get
        raw = cfg_get(load_config(), subsystem, CONFIG_KEY, default=False)
    except Exception:
        return False
    return _normalize_enabled(raw)


def _normalize_enabled(value: Any) -> bool:
    """Coerce a config value to bool; unknown → False (gate off).

    YAML already parses bare on/off/yes/no as bools; the string branch covers
    hand-edited configs.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"on", "true", "yes", "1", "approve", "enabled"}
    return False


# --- Pending store (file-backed) ---

def _pending_dir(subsystem: str) -> Path:
    return get_hermes_home() / "pending" / subsystem


def _pending_path(subsystem: str, pending_id: str) -> Path:
    return _pending_dir(subsystem) / f"{pending_id}.json"


def _read_record(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def stage_write(subsystem: str, payload: Dict[str, Any],
                *, summary: str, origin: str) -> Dict[str, Any]:
    """Persist a pending write and return its record (``id`` + metadata).

    ``payload`` is the exact kwargs to replay the write on approval; ``summary``
    is the one-line description shown in pending lists; ``origin`` is
    ``foreground`` or ``background_review`` (audit). Best-effort: on disk
    failure it logs and still returns a record — the write is lost, which is
    the safe failure for an approval gate (nothing silently committed).
    """
    pid = uuid.uuid4().hex[:8]
    record = {
        "id": pid,
        "subsystem": subsystem,
        "action": payload.get("action", ""),
        "summary": (summary or "").strip(),
        "origin": origin or "foreground",
        "created_at": time.time(),
        "payload": payload,
    }
    try:
        path = _pending_path(subsystem, pid)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, path)
    except Exception as e:  # pragma: no cover - disk failure path
        logger.error("Failed to stage pending %s write: %s", subsystem, e, exc_info=True)
    return record


def list_pending(subsystem: str) -> List[Dict[str, Any]]:
    """Return all pending records for ``subsystem``, oldest first."""
    d = _pending_dir(subsystem)
    if not d.exists():
        return []
    records: List[Dict[str, Any]] = []
    for p in d.glob("*.json"):
        try:
            records.append(_read_record(p))
        except Exception:
            logger.warning("Skipping unreadable pending record: %s", p)
    records.sort(key=lambda r: r.get("created_at", 0))
    return records


def get_pending(subsystem: str, pending_id: str) -> Optional[Dict[str, Any]]:
    """Return a single pending record by id, or None."""
    path = _pending_path(subsystem, pending_id)
    if not path.exists():
        return None
    try:
        return _read_record(path)
    except Exception:
        return None


def discard_pending(subsystem: str, pending_id: str) -> bool:
    """Delete a pending record. Returns True if it existed."""
    path = _pending_path(subsystem, pending_id)
    try:
        if path.exists():
            path.unlink()
            return True
    except Exception as e:  # pragma: no cover
        logger.error("Failed to discard pending %s/%s: %s", subsystem, pending_id, e)
    return False


def pending_count(subsystem: str) -> int:
    """Cheap count of pending records (for notification badges)."""
    d = _pending_dir(subsystem)
    if not d.exists():
        return 0
    try:
        return sum(1 for _ in d.glob("*.json"))
    except Exception:
        return 0


# --- Write origin ---

def current_origin() -> str:
    """Return ``foreground`` or ``background_review``.

    Reuses the skill-provenance ContextVar the background review fork sets;
    foreground turns leave it at the default.
    """
    try:
        from tools.skill_provenance import get_current_write_origin
        return get_current_write_origin()
    except Exception:
        return "foreground"


# --- Gate decision ---

@dataclass(slots=True, kw_only=True)
class GateDecision:
    """Result of evaluating the write gate. Exactly one flag is True.

    ``allow`` proceed with the real write; ``blocked`` the user denied an inline
    prompt (``message`` explains why); ``stage`` the caller must ``stage_write``
    the payload (``message`` is the user-facing "staged for approval" note).
    """

    allow: bool = False
    blocked: bool = False
    stage: bool = False
    message: str = ""


def _staged(subsystem: str) -> GateDecision:
    where = "/skills pending" if subsystem == SKILLS else "/memory pending"
    return GateDecision(
        stage=True,
        message=(
            f"Staged for approval ({subsystem}.write_approval is on). "
            f"Not yet saved — review with {where}."
        ),
    )


def evaluate_gate(subsystem: str, *, inline_summary: str = "",
                  inline_detail: str = "") -> GateDecision:
    """Decide what to do with a pending write for ``subsystem``.

    Decision matrix:
        gate off (default)                    → allow
        gate on, memory + interactive CLI     → inline approve/deny prompt
        gate on, memory + gateway/script/bg   → stage
        gate on, skills (any origin)          → stage (too big to review inline)

    The gate only ever delays a write, never silently refuses it; ``blocked``
    is produced only when the user actively denies the inline prompt.
    ``inline_summary``/``inline_detail`` feed the memory inline prompt.
    """
    if not write_approval_enabled(subsystem):
        return GateDecision(allow=True)

    # Skills always stage; a background write runs in a daemon thread with no user.
    if subsystem == SKILLS or current_origin() == "background_review":
        return _staged(subsystem)

    # Memory + foreground: prompt inline if an interactive channel exists;
    # otherwise (gateway, script, prompt failure) stage instead of blind-denying.
    granted = _prompt_inline_memory_approval(inline_summary, inline_detail)
    if granted is True:
        return GateDecision(allow=True)
    if granted is False:
        return GateDecision(
            blocked=True,
            message="Memory write denied by user. The change was not saved.",
        )
    return _staged(MEMORY)


def _prompt_inline_memory_approval(summary: str, detail: str) -> Optional[bool]:
    """Prompt inline for a memory write: True approved, False denied, None → stage.

    Uses the per-thread CLI approval callback registered for dangerous
    commands (``tools.terminal_tool.set_approval_callback``), invoked directly
    rather than via ``prompt_dangerous_approval``: that wrapper falls back to
    ``input()`` (deadlock-prone under prompt_toolkit; silent deny in gateway
    sessions, whose ``/approve`` round-trip lives in the pending-approval
    queue) and converts callback errors into a deny. Here a missing channel or
    failed prompt must stage instead.
    """
    try:
        from tools.terminal_tool import _get_approval_callback
    except Exception:
        return None
    callback = _get_approval_callback()
    if callback is None:
        return None

    header = summary.strip() or "Save to memory?"
    body = detail.strip()
    try:
        choice = callback(body or header, f"Save to memory: {header}", allow_permanent=False)
    except Exception as e:
        logger.error("Inline memory approval prompt failed: %s", e)
        return None

    if choice in {"once", "session"}:
        return True
    if choice == "deny":
        return False
    return None  # unknown outcome → no decision, stage rather than drop


# --- Skill-specific helpers (gist + diff for the review affordances) ---

def skill_gist(action: str, name: str, *, content: str = "",
               file_path: str = "", old_string: str = "",
               new_string: str = "") -> str:
    """Build a one-line heuristic gist (no model call) for a pending skill write.

    create/edit use the frontmatter ``description:``; patch/write_file describe
    the size of the change. The full diff stays behind /skills diff.
    """
    if action in {"create", "edit"} and content:
        desc = _frontmatter_description(content)
        size = f"{len(content) // 1024 + 1} KB" if len(content) >= 1024 else f"{len(content)} chars"
        verb = "create" if action == "create" else "rewrite"
        return f"{verb} '{name}' — {desc} ({size})" if desc else f"{verb} '{name}' ({size})"
    if action == "patch":
        removed = old_string.count("\n") + 1 if old_string else 0
        added = new_string.count("\n") + 1 if new_string else 0
        return f"patch '{name}' {file_path or 'SKILL.md'} (+{added}/-{removed} lines)"
    if action == "write_file":
        return f"write {file_path} in '{name}'"
    if action == "remove_file":
        return f"remove {file_path} from '{name}'"
    if action == "delete":
        return f"delete skill '{name}'"
    return f"{action} '{name}'"


def _frontmatter_description(content: str) -> str:
    """Extract the ``description:`` value from SKILL.md YAML frontmatter (≤140 chars)."""
    m = re.search(r"^description:\s*(.+)$", content, re.MULTILINE)
    return m.group(1).strip().strip("'\"")[:140] if m else ""


def _find_skill_path(name: str) -> Optional[Path]:
    """Directory of an installed skill, or None if unknown / lookup unavailable."""
    try:
        from tools.skill_manager_tool import _find_skill
        found = _find_skill(name)
        return found["path"] if found else None
    except Exception:
        return None


def skill_pending_diff(record: Dict[str, Any]) -> str:
    """Full content (create) or unified diff vs. the on-disk skill (edit/patch/write_file).

    Rendered by /skills diff <id> on surfaces that can show it (CLI pager,
    dashboard, pending JSON file).
    """
    payload = record.get("payload", {})
    action = payload.get("action", "")
    name = payload.get("name", "")

    if action == "create":
        return payload.get("content") or ""
    if action == "remove_file":
        return f"remove file: {payload.get('file_path')} from skill '{name}'"
    if action == "delete":
        return f"delete skill '{name}'"
    if action not in {"edit", "patch", "write_file"}:
        return f"({action} on '{name}')"

    # patch/write_file target a file inside the skill; edit always targets SKILL.md.
    target_label = "SKILL.md"
    current = ""
    skill_dir = _find_skill_path(name)
    if skill_dir:
        if action != "edit":
            target_label = payload.get("file_path") or "SKILL.md"
        try:
            p = skill_dir / target_label
            if p.exists():
                current = p.read_text(encoding="utf-8")
        except Exception:
            current = ""

    if action == "edit":
        new = payload.get("content") or ""
    elif action == "patch":
        old_s = payload.get("old_string") or ""
        new_s = payload.get("new_string") or ""
        new = current.replace(old_s, new_s) if current else f"(patch {old_s!r} → {new_s!r})"
    else:
        new = payload.get("file_content") or ""

    diff = difflib.unified_diff(
        current.splitlines(keepends=True),
        new.splitlines(keepends=True),
        fromfile=f"a/{target_label}",
        tofile=f"b/{target_label}",
    )
    return "".join(diff) or "(no textual change)"
