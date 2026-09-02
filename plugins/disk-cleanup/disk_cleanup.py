"""disk_cleanup — ephemeral file cleanup for Hermes Agent.

Library module behind the disk-cleanup plugin; ``__init__.py`` wires these
functions into ``post_tool_call`` / ``on_session_end`` hooks so tracking and
cleanup happen without the agent calling a tool or remembering a skill.

Rules: test files delete at task end (age >= 0); temp after 7 days; cron-output
after 14 days; empty dirs under HERMES_HOME always. Deep-only prompts: research
(keep 10 newest, > 30 days), chrome-profile > 14 days, any file > 500 MB.

Scope: strictly HERMES_HOME and /tmp/hermes-*. Never touches ~/.hermes/logs/ or
any system directory.
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

try:
    from hermes_constants import get_hermes_home
except Exception:  # pragma: no cover — plugin may load before constants resolves
    import os

    def get_hermes_home() -> Path:  # type: ignore[no-redef]
        val = (os.environ.get("HERMES_HOME") or "").strip()
        return Path(val).resolve() if val else (Path.home() / ".hermes").resolve()


logger = logging.getLogger(__name__)

_LARGE_FILE_BYTES = 500 * 1024 * 1024


# --- Paths / safety ---------------------------------------------------------

def _state_file(name: str) -> Path:
    """``$HERMES_HOME/disk-cleanup/<name>`` — state and audit log deliberately
    live outside ``$HERMES_HOME/logs/``."""
    return get_hermes_home() / "disk-cleanup" / name


def is_safe_path(path: Path) -> bool:
    """Accept only paths under HERMES_HOME or ``/tmp/hermes-*``.

    Rejects Windows mounts (``/mnt/c`` etc.) and any system directory.
    """
    try:
        path.resolve().relative_to(get_hermes_home())
        return True
    except (ValueError, OSError):
        pass
    parts = path.parts
    return len(parts) >= 3 and parts[1] == "tmp" and parts[2].startswith("hermes-")


def _log(message: str) -> None:
    """Append to the audit log; never let it break the agent loop."""
    try:
        log_file = _state_file("cleanup.log")
        log_file.parent.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"[{ts}] {message}\n")
    except OSError:
        pass


# --- tracked.json — atomic read/write, backup scoped to tracked.json only ----

def load_tracked() -> List[Dict[str, Any]]:
    """Load tracked.json.  Restores from ``.bak`` on corruption."""
    tf = _state_file("tracked.json")
    tf.parent.mkdir(parents=True, exist_ok=True)
    if not tf.exists():
        return []
    try:
        return json.loads(tf.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, ValueError):
        bak = tf.with_suffix(".json.bak")
        if bak.exists():
            try:
                data = json.loads(bak.read_text(encoding="utf-8"))
                _log("WARN: tracked.json corrupted — restored from .bak")
                return data
            except Exception:
                pass
        _log("WARN: tracked.json corrupted, no backup — starting fresh")
        return []


def save_tracked(tracked: List[Dict[str, Any]]) -> None:
    """Atomic write: ``.tmp`` → backup old → rename."""
    tf = _state_file("tracked.json")
    tf.parent.mkdir(parents=True, exist_ok=True)
    tmp = tf.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(tracked, indent=2), encoding="utf-8")
    if tf.exists():
        shutil.copy2(tf, tf.with_suffix(".json.bak"))
    tmp.replace(tf)


# --- Categories / protected trees -------------------------------------------

ALLOWED_CATEGORIES = {
    "temp", "test", "research", "download",
    "chrome-profile", "cron-output", "other",
}

# Top-level HERMES_HOME dirs whose empty subdirs are never swept. The last row
# is user-authored project trees (patches/, projects/, ...) — never sweep inside.
_EMPTY_DIR_PROTECTED_TOP_LEVEL = frozenset({
    "logs", "memories", "sessions", "cron", "cronjobs",
    "cache", "skills", "plugins", "disk-cleanup", "optional-skills",
    "hermes-agent", "backups", "profiles", ".worktrees",
    "patches", "projects", "skins", "themes", "contributors",
})

_EMPTY_DIR_SWEEP_PRUNE_DIRS = frozenset({
    ".git", "node_modules", "venv", ".venv",
    "site-packages", "__pycache__",
})

# Top-level entries under HERMES_HOME that guess_category() never auto-tracks:
# state dir, logs, memory, sessions, config/secrets, and user-authored project
# trees (a file named test_*/tmp_* inside patches/ or projects/ is not disposable).
_NEVER_TRACK_TOP_LEVEL = frozenset({
    "disk-cleanup", "logs", "memories", "sessions", "config.yaml",
    "skills", "plugins", ".env", "USER.md", "MEMORY.md", "SOUL.md",
    "auth.json", "hermes-agent",
    "patches", "projects", "skins", "themes", "contributors",
    "profiles", "backups", "optional-skills",
})

# Defense-in-depth for quick(): exact cron control-plane paths never deleted,
# regardless of stored category (guards stale tracked.json entries).
_PROTECTED_CRON_PATHS: set[str] = set()


def _is_protected_cron_path(p: Path) -> bool:
    """True if *p* is cron control-plane state that must never be deleted.

    Matches by EXACT path only: the ``cron/`` dir itself, ``jobs.json``,
    ``.tick.lock``, and the ``output/`` root. It must NOT be widened to
    everything under ``cron/output/`` — run artifacts there are disposable and
    cleaned by retention; only the ``output/`` root is protected because
    deleting it wholesale erases every job's retained run history.
    """
    if not _PROTECTED_CRON_PATHS:  # built lazily so HERMES_HOME resolves once
        hermes_home = get_hermes_home()
        for parent in ("cron", "cronjobs"):
            base = hermes_home / parent
            _PROTECTED_CRON_PATHS.update(
                str(x) for x in (base, base / "output", base / "jobs.json", base / ".tick.lock")
            )
    return str(p.resolve()) in _PROTECTED_CRON_PATHS


def fmt_size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


# --- Track / forget ---------------------------------------------------------

def track(path_str: str, category: str, silent: bool = False) -> bool:
    """Register a file for tracking. Returns True if newly tracked."""
    if category not in ALLOWED_CATEGORIES:
        _log(f"WARN: unknown category '{category}', using 'other'")
        category = "other"

    path = Path(path_str).resolve()
    if not path.exists():
        _log(f"SKIP: {path} (does not exist)")
        return False
    if not is_safe_path(path):
        _log(f"REJECT: {path} (outside HERMES_HOME)")
        return False

    size = path.stat().st_size if path.is_file() else 0
    tracked = load_tracked()
    if any(item["path"] == str(path) for item in tracked):
        return False

    tracked.append({
        "path": str(path),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "category": category,
        "size": size,
    })
    save_tracked(tracked)
    _log(f"TRACKED: {path} ({category}, {fmt_size(size)})")
    if not silent:
        print(f"Tracked: {path} ({category}, {fmt_size(size)})")
    return True


def forget(path_str: str) -> int:
    """Remove a path from tracking without deleting the file."""
    p = Path(path_str).resolve()
    tracked = load_tracked()
    before = len(tracked)
    tracked = [i for i in tracked if Path(i["path"]).resolve() != p]
    removed = before - len(tracked)
    if removed:
        save_tracked(tracked)
        _log(f"FORGOT: {p} ({removed} entries)")
    return removed


# --- Rules shared by dry_run / quick / deep ---------------------------------

def _live_items(tracked: List[Dict], now: datetime, *, log_stale: bool = False) -> Iterator[Tuple[Dict, Path, int]]:
    """Yield ``(item, path, age_days)`` for entries whose path still exists."""
    for item in tracked:
        p = Path(item["path"])
        if not p.exists():
            if log_stale:
                _log(f"STALE: {p} (removed from tracking)")
            continue
        yield item, p, (now - datetime.fromisoformat(item["timestamp"])).days


def _is_auto_delete(cat: str, age: int) -> bool:
    return cat == "test" or (cat == "temp" and age > 7) or (cat == "cron-output" and age > 14)


def _prompt_group(item: Dict, age: int) -> Optional[str]:
    """Deep-only bucket: ``research`` / ``chrome`` / ``large`` or None."""
    cat = item["category"]
    if cat == "research" and age > 30:
        return "research"
    if cat == "chrome-profile" and age > 14:
        return "chrome"
    if item["size"] > _LARGE_FILE_BYTES:
        return "large"
    return None


def _delete_item(item: Dict) -> Optional[str]:
    """Delete a tracked file/dir and audit-log it. Returns an error string on OSError, else None."""
    p = Path(item["path"])
    try:
        if p.is_file():
            p.unlink()
        elif p.is_dir():
            shutil.rmtree(p)
    except OSError as e:
        _log(f"ERROR deleting {p}: {e}")
        return f"{p}: {e}"
    _log(f"DELETED: {p} ({item['category']}, {fmt_size(item['size'])})")
    return None


# Stored categories that are re-validated against guess_category() before use.
# Old tracked.json entries can carry "cron-output" for control-plane files
# (cron/jobs.json) or "test" for files now under protected project trees;
# guess_category() was tightened later but existing entries were never re-checked.
_STALE_SKIP_NOTE = {"cron-output": "", "test": " — under protected tree"}


# --- Dry run / quick / deep -------------------------------------------------

def dry_run() -> Tuple[List[Dict], List[Dict]]:
    """Return (auto_delete_list, needs_prompt_list) without touching files."""
    auto: List[Dict] = []
    prompt: List[Dict] = []
    for item, p, age in _live_items(load_tracked(), datetime.now(timezone.utc)):
        cat = item["category"]
        # Stale cron-output entries are skipped by quick(); omit them here too.
        if cat == "cron-output" and guess_category(p) != "cron-output":
            continue
        if _is_auto_delete(cat, age):
            auto.append(item)
        elif _prompt_group(item, age):
            prompt.append(item)
    return auto, prompt


def quick() -> Dict[str, Any]:
    """Safe deterministic cleanup — no prompts.

    Returns: ``{"deleted": N, "empty_dirs": N, "freed": bytes, "errors": [str, ...]}``.
    """
    deleted = freed = 0
    new_tracked: List[Dict] = []
    errors: List[str] = []

    for item, p, age in _live_items(load_tracked(), datetime.now(timezone.utc), log_stale=True):
        cat = item["category"]
        if cat in _STALE_SKIP_NOTE and (re_cat := guess_category(p)) != cat:
            # Misclassified stale entry — drop it rather than delete the file.
            _log(f"SKIP stale {cat} entry: {p} (re-classified as {re_cat!r}{_STALE_SKIP_NOTE[cat]})")
            continue
        # Hard safety net even if re-validation above somehow let it through.
        if _is_protected_cron_path(p):
            _log(f"SKIP protected cron path: {p}")
            continue
        if not _is_auto_delete(cat, age):
            new_tracked.append(item)
            continue
        err = _delete_item(item)
        if err is None:
            freed += item["size"]
            deleted += 1
        else:
            errors.append(err)
            new_tracked.append(item)

    empty_removed = _sweep_empty_dirs(get_hermes_home())
    save_tracked(new_tracked)
    _log(f"QUICK_SUMMARY: {deleted} files, {empty_removed} dirs, {fmt_size(freed)}")
    return {"deleted": deleted, "empty_dirs": empty_removed, "freed": freed, "errors": errors}


def _subdirs(dirpath: Path, exclude: frozenset) -> List[Path]:
    try:
        return [c for c in dirpath.iterdir() if c.is_dir() and not c.is_symlink() and c.name not in exclude]
    except OSError:
        return []


def _sweep_empty_dirs(hermes_home: Path) -> int:
    """Remove empty dirs under HERMES_HOME, never recursing into durable state
    trees. Some installs keep the Hermes checkout, venv, and desktop build under
    HERMES_HOME; a full rglob there can stall the gateway event loop for minutes.
    Iterative post-order so parents emptied by child removal are caught."""
    removed = 0
    stack: List[Tuple[Path, bool]] = [
        (top, False) for top in _subdirs(hermes_home, _EMPTY_DIR_PROTECTED_TOP_LEVEL | _EMPTY_DIR_SWEEP_PRUNE_DIRS)
    ]
    while stack:
        dirpath, visited = stack.pop()
        if visited:
            try:
                if not any(dirpath.iterdir()):
                    dirpath.rmdir()
                    removed += 1
                    _log(f"DELETED: {dirpath} (empty dir)")
            except OSError:
                pass
            continue
        stack.append((dirpath, True))
        stack.extend((child, False) for child in _subdirs(dirpath, _EMPTY_DIR_SWEEP_PRUNE_DIRS))
    return removed


def deep(confirm: Optional[Callable[[Dict], bool]] = None) -> Dict[str, Any]:
    """Deep cleanup: :func:`quick`, then ask *confirm(item)* for each risky item
    (research > 30d beyond the 10 newest, chrome-profile > 14d, any file > 500 MB).

    Returns: ``{"quick": {...}, "deep_deleted": N, "deep_freed": bytes}``.
    """
    quick_result = quick()
    if confirm is None:  # no interactive confirmer — stop after the quick pass
        return {"quick": quick_result, "deep_deleted": 0, "deep_freed": 0}

    tracked = load_tracked()
    groups: Dict[str, List[Dict]] = {"research": [], "chrome": [], "large": []}
    for item, _p, age in _live_items(tracked, datetime.now(timezone.utc)):
        group = _prompt_group(item, age)
        if group:
            groups[group].append(item)

    groups["research"].sort(key=lambda x: x["timestamp"], reverse=True)
    del groups["research"][:10]  # keep the 10 newest research items

    removed = [item for group in groups.values() for item in group if confirm(item) and _delete_item(item) is None]
    if removed:
        remove_paths = {i["path"] for i in removed}
        save_tracked([i for i in tracked if i["path"] not in remove_paths])

    return {"quick": quick_result, "deep_deleted": len(removed), "deep_freed": sum(i["size"] for i in removed)}


# --- Status -----------------------------------------------------------------

def status() -> Dict[str, Any]:
    """Return per-category breakdown and top 10 largest tracked files."""
    tracked = load_tracked()
    cats: Dict[str, Dict] = {}
    for item in tracked:
        c = cats.setdefault(item["category"], {"count": 0, "size": 0})
        c["count"] += 1
        c["size"] += item["size"]

    existing = [(i["path"], i["size"], i["category"]) for i in tracked if Path(i["path"]).exists()]
    existing.sort(key=lambda x: x[1], reverse=True)
    return {"categories": cats, "top10": existing[:10], "total_tracked": len(tracked)}


def format_status(s: Dict[str, Any]) -> str:
    """Human-readable status string (for slash command output)."""
    lines = [f"{'Category':<20} {'Files':>6}  {'Size':>10}", "-" * 40]
    cats = s["categories"]
    for cat, d in sorted(cats.items(), key=lambda x: x[1]["size"], reverse=True):
        lines.append(f"{cat:<20} {d['count']:>6}  {fmt_size(d['size']):>10}")
    if not cats:
        lines.append("(nothing tracked yet)")

    lines += ["", "Top 10 largest tracked files:"]
    if not s["top10"]:
        lines.append("  (none)")
    for rank, (path, size, cat) in enumerate(s["top10"], 1):
        lines.append(f"  {rank:>2}. {fmt_size(size):>8}  [{cat}]  {path}")
    return "\n".join(lines)


# --- Auto-categorisation from tool-call inspection --------------------------

_TEST_PATTERNS = ("test_", "tmp_")
_TEST_SUFFIXES = (".test.py", ".test.js", ".test.ts", ".test.md")


def guess_category(path: Path) -> Optional[str]:
    """Return a category label for *path*, or None if we shouldn't track it.

    Used by the ``post_tool_call`` hook to auto-track ephemeral files.
    """
    if not is_safe_path(path):
        return None

    try:
        rel = path.resolve().relative_to(get_hermes_home())
        top = rel.parts[0] if rel.parts else ""
        if top in _NEVER_TRACK_TOP_LEVEL:
            return None
        if top in ("cron", "cronjobs"):
            # Only the disposable ``output/`` subtree is a candidate. Top-level
            # control-plane state (jobs.json, .tick.lock) must never be tracked —
            # deleting it wipes the live scheduler registry.
            if len(rel.parts) >= 3 and rel.parts[1] == "output":
                return "cron-output"
            return None
        if top == "cache":
            return "temp"
    except ValueError:
        pass  # not under HERMES_HOME (e.g. /tmp/hermes-*) — fall through to name rules

    name = path.name
    if name.startswith(_TEST_PATTERNS) or name.endswith(_TEST_SUFFIXES):
        return "test"
    return None
