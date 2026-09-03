"""Per-task read/search bookkeeping for the file tools.

Process-lifetime state behind read_file/search_files/write_file/patch;
``tools.file_tools`` re-imports every name here. Per task_id ``_read_tracker``
stores: ``last_key``/``consecutive`` (loop detection; reset by any OTHER tool
call), ``read_history`` (diagnostics), ``dedup`` (key -> mtime; cleared on
context compression), ``dedup_hits`` (stub-loop breaker), ``read_timestamps``
(staleness warnings) and ``not_found`` (short-TTL negative cache). Every
container is hard-capped (``_cap_read_tracker_data``) so long sessions stay small.
"""

import logging
import os
import threading
import time

from tools.file_state import _evict_oldest
from tools.file_tools_paths import _authoritative_workspace_root, _resolve_path_for_task

logger = logging.getLogger("tools.file_tools")

_read_tracker_lock = threading.Lock()
_read_tracker: dict = {}

# Consecutive patch failures per (task_id, resolved_path); escalates the hint
# when the model keeps failing the same file. Reset on a successful patch.
_patch_failure_lock = threading.Lock()
_patch_failure_tracker: dict = {}  # {task_id: {resolved_path: count}}
_PATCH_FAILURE_PATHS_CAP = 64

# Only the most recent reads matter for dedup, loop detection and external-edit
# warnings; caps bound accretion regardless of session length.
_READ_HISTORY_CAP = 500
_DEDUP_CAP = 1000
_READ_TIMESTAMPS_CAP = 1000
_NOT_FOUND_CAP = 500
_NOT_FOUND_TTL_SECONDS = 60.0  # a path that didn't exist may be created soon


def _task_data(task_id: str) -> dict:
    """Get-or-create the tracker entry for *task_id*, back-filling any missing keys.

    Must be called with ``_read_tracker_lock`` held. Entries created by older
    code paths (or injected by tests) may lack the newer containers.
    """
    task_data = _read_tracker.setdefault(task_id, {
        "last_key": None, "consecutive": 0, "read_history": set()})
    for key in ("dedup", "dedup_hits", "read_timestamps"):
        task_data.setdefault(key, {})
    return task_data


def _record_patch_failure(task_id: str, resolved_path: str) -> int:
    """Increment and return the consecutive-failure count for this path."""
    with _patch_failure_lock:
        task_failures = _patch_failure_tracker.setdefault(task_id, {})
        # Evict the oldest entry once a task has failed on many distinct files.
        if resolved_path not in task_failures:
            _evict_oldest(task_failures, _PATCH_FAILURE_PATHS_CAP - 1)
        task_failures[resolved_path] = task_failures.get(resolved_path, 0) + 1
        return task_failures[resolved_path]


def _reset_patch_failures(task_id: str, resolved_paths: list) -> None:
    """Clear consecutive-failure counts for the given paths."""
    if not resolved_paths:
        return
    with _patch_failure_lock:
        task_failures = _patch_failure_tracker.get(task_id)
        for rp in resolved_paths if task_failures else ():
            task_failures.pop(rp, None)


def _cap_read_tracker_data(task_data: dict) -> None:
    """Enforce size caps on the per-task sub-containers. Call with ``_read_tracker_lock`` held."""
    # Caps are read at call time so tests can monkeypatch the module constants.
    for key, cap in (
        ("read_history", _READ_HISTORY_CAP),
        ("dedup", _DEDUP_CAP),
        ("dedup_hits", _DEDUP_CAP),
        ("read_timestamps", _READ_TIMESTAMPS_CAP),
        ("not_found", _NOT_FOUND_CAP)):
        container = task_data.get(key)
        if container is not None and len(container) > cap:
            _evict_oldest(container, cap)


def _pop_not_found(op: str, resolved_str: str, task_id: str) -> None:
    """Drop the negative-cache entry for *(op, resolved_str)*. Lock must be held."""
    task_data = _read_tracker.get(task_id)
    nf = task_data.get("not_found") if task_data else None
    if nf:
        nf.pop((op, resolved_str), None)


def _check_not_found_cache(op: str, resolved_str: str, task_id: str) -> str | None:
    """Return cached not-found JSON for *(op, resolved_str)* if still fresh.

    Skips the subprocess + similar-name walk when the model retries the same
    missing path. *op* is "read" or "search" (different error JSON shapes).
    Evicted by TTL, by write_file/patch on the path, or by any other tool call.
    """
    with _read_tracker_lock:
        task_data = _read_tracker.get(task_id)
        entry = (task_data.get("not_found") or {}).get((op, resolved_str)) if task_data else None
        if entry is None:
            return None
        ts, cached_json = entry
        if time.monotonic() - ts > _NOT_FOUND_TTL_SECONDS:
            _pop_not_found(op, resolved_str, task_id)
            return None
    # The path may have been created since the miss was cached (terminal,
    # another agent, ...) — "check → create → read" is common, so serving a
    # stale miss breaks it. The stat runs OUTSIDE the global tracker lock: a
    # hung stat on a dead network mount must not stall every task.
    if os.path.exists(resolved_str):
        with _read_tracker_lock:
            _pop_not_found(op, resolved_str, task_id)
        return None
    return cached_json


def _record_not_found(op: str, resolved_str: str, task_id: str, error_json: str) -> None:
    """Cache a not-found error so the next *op* call for *resolved_str* skips I/O."""
    with _read_tracker_lock:
        task_data = _task_data(task_id)
        task_data.setdefault("not_found", {})[(op, resolved_str)] = (time.monotonic(), error_json)
        _cap_read_tracker_data(task_data)


def _bump_consecutive(task_data: dict, key: tuple) -> int:
    """Update last_key/consecutive for *key* and return the new count. Lock must be held."""
    if task_data["last_key"] == key:
        task_data["consecutive"] += 1
    else:
        task_data["last_key"] = key
        task_data["consecutive"] = 1
    return task_data["consecutive"]


def reset_file_dedup(task_id: str = None):
    """Clear the read-dedup cache (one task, or all when ``task_id`` is None).

    Called after context compression: the original read content was summarised
    away, so a "file unchanged" stub would point at content no longer in context.
    """
    with _read_tracker_lock:
        if task_id:
            targets = [_read_tracker[task_id]] if _read_tracker.get(task_id) else []
        else:
            targets = list(_read_tracker.values())
        for task_data in targets:
            for key in ("dedup", "dedup_hits"):
                if key in task_data:
                    task_data[key].clear()


def notify_other_tool_call(task_id: str = "default"):
    """Reset the consecutive read/search counter for a task.

    Called by the dispatcher for every tool OTHER than read_file/search_files,
    so loop detection only fires on truly consecutive repeats. Also clears the
    stub-hit counters and the not-found cache: any other tool may have created
    a previously-missing path (the serve-side stat covers most cases; clearing
    covers the rest, e.g. permission flips).
    """
    with _read_tracker_lock:
        task_data = _read_tracker.get(task_id)
        if task_data:
            task_data["last_key"] = None
            task_data["consecutive"] = 0
            for key in ("dedup_hits", "not_found"):
                if task_data.get(key):
                    task_data[key].clear()


def _invalidate_dedup_for_path(filepath: str, task_id: str) -> None:
    """Evict every dedup entry (all offset/limit ranges) and not-found entry for *filepath*.

    Called after write_file/patch so the next read returns fresh content
    instead of a stale "unchanged" stub. Acquires ``_read_tracker_lock`` itself.
    """
    try:
        resolved = str(_resolve_path_for_task(filepath, task_id))
    except (OSError, ValueError):
        return
    with _read_tracker_lock:
        task_data = _read_tracker.get(task_id)
        if task_data is None:
            return
        dedup = task_data.get("dedup")
        if dedup:
            for k in [k for k in dedup if k[0] == resolved]:
                del dedup[k]
        _pop_not_found("read", resolved, task_id)
        _pop_not_found("search", resolved, task_id)


def _update_read_timestamp(filepath: str, task_id: str) -> None:
    """After a successful write: invalidate dedup and refresh the stored mtime so
    consecutive edits by the same task don't trigger false staleness warnings."""
    _invalidate_dedup_for_path(filepath, task_id)
    try:
        resolved = str(_resolve_path_for_task(filepath, task_id))
        current_mtime = os.path.getmtime(resolved)
    except (OSError, ValueError):
        return
    with _read_tracker_lock:
        task_data = _read_tracker.get(task_id)
        if task_data is not None:
            task_data.setdefault("read_timestamps", {})[resolved] = current_mtime
            _cap_read_tracker_data(task_data)


def _check_file_staleness(filepath: str, task_id: str) -> str | None:
    """Warn (don't block) when the file's mtime changed since this task last read it.

    ``None`` when never read, fresh, or unstattable (a deleted file is the
    write's problem to report).
    """
    try:
        resolved = str(_resolve_path_for_task(filepath, task_id))
    except (OSError, ValueError):
        return None
    with _read_tracker_lock:
        task_data = _read_tracker.get(task_id)
        read_mtime = task_data.get("read_timestamps", {}).get(resolved) if task_data else None
    if read_mtime is None:
        return None
    try:
        current_mtime = os.path.getmtime(resolved)
    except OSError:
        return None
    if current_mtime != read_mtime:
        return (
            f"Warning: {filepath} was modified since you last read it "
            "(external edit or concurrent agent). The content you read may be "
            "stale. Consider re-reading the file to verify before writing.")
    return None


def _mark_verification_stale(task_id: str, resolved_paths: list[str],
                             session_id: str | None = None) -> None:
    """Best-effort note that successful edits made prior verification stale.

    The workspace cwd is the first edited path's project root when one is
    recognised, else the task's workspace root, else the first path's parent.
    """
    from pathlib import Path

    paths = [p for p in resolved_paths if p]
    if not paths:
        return
    try:
        from agent.coding_context import project_facts_for
        from agent.verification_evidence import mark_workspace_edited

        parents = [str(Path(p).parent) for p in paths]
        cwd = (next((c for c in parents if project_facts_for(c)), None)
               or _authoritative_workspace_root(task_id) or parents[0])
        mark_workspace_edited(session_id=session_id or task_id, cwd=cwd, paths=paths)
    except Exception:
        logger.debug("verification stale marker failed", exc_info=True)
