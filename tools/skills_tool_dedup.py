"""skill_view repeat-view dedup registry.

Per-task cache of (skill name, file_path) -> (skill file mtime+size). On a
repeat view of an UNCHANGED skill file, ``tools.skills_tool`` returns a short
stub instead of re-sending the full content — the earlier tool result in this
conversation already carries it verbatim. Cleared on context compression via
``reset_skill_view_dedup()`` (wired next to read_file's reset_file_dedup)
because after compression the original content is summarized away.

Every name is re-imported into ``tools.skills_tool``; the tracker state lives
here and only here.
"""

import json
import os
import threading
from typing import Dict

_skill_view_tracker: Dict[str, Dict[tuple, tuple]] = {}
_skill_view_tracker_lock = threading.Lock()
_SKILL_VIEW_DEDUP_CAP = 200

_SKILL_VIEW_DEDUP_MESSAGE = (
    "Skill content unchanged since it was loaded earlier in this "
    "conversation — refer to the earlier skill_view result; it is still "
    "current and complete. (Re-issued after context compression, this "
    "returns the full content again.)"
)


def _skill_view_fingerprint(payload: dict) -> tuple | None:
    """Stat the skill file a successful skill_view served, for change detection."""
    src = payload.get("_source_path")
    if not src:
        return None
    try:
        st = os.stat(src)
        return (src, st.st_mtime_ns, st.st_size)
    except OSError:
        return None


def _record_skill_view(task_id, name, file_path, payload: dict) -> None:
    """Record a served skill_view so an identical repeat can be deduped."""
    if not task_id:
        return
    # Never dedup setup-needed views: readiness depends on config/env state
    # that can change without the skill file changing, and the model must
    # see the refreshed setup status on a re-view.
    if payload.get("setup_needed") or payload.get("readiness_status") == "setup_needed":
        return
    fp = _skill_view_fingerprint(payload)
    if fp is None:
        return
    key = (str(payload.get("name") or name), file_path or "")
    with _skill_view_tracker_lock:
        cache = _skill_view_tracker.setdefault(str(task_id), {})
        cache[key] = fp
        while len(cache) > _SKILL_VIEW_DEDUP_CAP:
            try:
                cache.pop(next(iter(cache)))
            except (StopIteration, KeyError):
                break


def _check_skill_view_dedup(task_id, name, file_path) -> str | None:
    """Return a dedup stub when this exact skill file was already served
    to this task and is unchanged on disk; None otherwise."""
    if not task_id:
        return None
    with _skill_view_tracker_lock:
        cache = _skill_view_tracker.get(str(task_id))
        if not cache:
            return None
        # The record key uses the RESOLVED name; check both the raw arg and
        # resolved forms so 'category/skill' and bare-name views coalesce.
        for key, (src, mtime_ns, size) in list(cache.items()):
            rec_name, rec_fp = key
            if rec_fp != (file_path or ""):
                continue
            n = str(name)
            if rec_name != n and not n.endswith("/" + rec_name) \
                    and not rec_name.endswith("/" + n) and n.split(":")[-1] != rec_name:
                continue
            try:
                st = os.stat(src)
                if (st.st_mtime_ns, st.st_size) != (mtime_ns, size):
                    cache.pop(key, None)
                    return None
            except OSError:
                cache.pop(key, None)
                return None
            return json.dumps({
                "success": True, "status": "unchanged", "name": rec_name,
                "file": file_path or "SKILL.md", "dedup": True,
                "content_returned": False, "message": _SKILL_VIEW_DEDUP_MESSAGE,
            }, ensure_ascii=False)
    return None


def reset_skill_view_dedup(task_id: str | None = None) -> None:
    """Clear the skill_view dedup cache (all tasks when task_id is None). Called on
    context compression: the original content is summarized away, so a re-view
    must return full content again."""
    with _skill_view_tracker_lock:
        if task_id is None:
            _skill_view_tracker.clear()
        else:
            _skill_view_tracker.pop(str(task_id), None)
