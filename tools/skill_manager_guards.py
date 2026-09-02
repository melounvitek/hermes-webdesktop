"""Write/delete guards for ``skill_manage`` (extracted from skill_manager_tool).

Every guard returns ``None`` when the operation may proceed, otherwise a
refusal (error dict or message). Origin-owned state (``_find_skill``,
``_skills_dir``) is reached lazily through ``tools.skill_manager_tool`` so
test patches on that module keep working.
"""

import contextvars as _ctxvars
import logging
import threading
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("tools.skill_manager_tool")

_ERR_KEY = "error"


def _refusal(message: str, **extra: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {"success": False, _ERR_KEY: message}
    out.update(extra)
    return out


def _is_background_review() -> bool:
    """True inside the autonomous curator review fork; False on any lookup failure."""
    try:
        from tools.skill_provenance import is_background_review
        return bool(is_background_review())
    except Exception:
        return False


def _resolved_str(path: Path) -> str:
    try:
        return str(path.resolve())
    except Exception:
        return str(path)


# ---------------------------------------------------------------------------
# Background-review read marks
# ---------------------------------------------------------------------------

class _BackgroundReviewReadMarks:
    """Read marks shared by copied tool contexts within one review run."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._paths: set[str] = set()

    def add(self, path: str) -> None:
        with self._lock:
            self._paths.add(path)

    def contains(self, path: str) -> bool:
        with self._lock:
            return path in self._paths


_background_review_read_paths: (
    "_ctxvars.ContextVar[Optional[_BackgroundReviewReadMarks]]"
) = _ctxvars.ContextVar("background_review_read_paths", default=None)


def mark_background_review_skill_read(path: Path) -> None:
    """Record that the active background-review fork has read a skill file.

    The review fork may evolve skills but must not patch content it only
    inferred from the transcript: skill_view calls this after returning file
    content, and the write guards below require the target to be marked.
    """
    if not _is_background_review():
        return
    marks = _background_review_read_paths.get()
    if marks is None:
        marks = _BackgroundReviewReadMarks()
        _background_review_read_paths.set(marks)
    marks.add(_resolved_str(path))


def _background_review_has_read(path: Path) -> bool:
    marks = _background_review_read_paths.get()
    return marks is not None and marks.contains(_resolved_str(path))


def _reset_background_review_read_marks() -> None:
    """Start a fresh, isolated read set for the current review context."""
    _background_review_read_paths.set(_BackgroundReviewReadMarks())


# ---------------------------------------------------------------------------
# Delete-target safety
# ---------------------------------------------------------------------------

def _containing_skills_root(skill_path: Path) -> Path:
    """Skills root (local or external_dirs entry) containing ``skill_path``;
    falls back to the local skills dir when no root matches."""
    from agent.skill_utils import get_all_skills_dirs
    from tools import skill_manager_tool as _smt

    try:
        resolved = skill_path.resolve()
    except OSError:
        resolved = skill_path
    for root in get_all_skills_dirs():
        try:
            resolved.relative_to(root.resolve())
            return root
        except (ValueError, OSError):
            continue
    return _smt._skills_dir()


def _is_path_redirect(path: Path) -> bool:
    """True when ``path`` is a symlink or (Windows 3.12+) a junction — either
    lets a poisoned tree redirect ``shutil.rmtree`` outside the skills root."""
    try:
        return path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction())
    except OSError:
        return False


def _validate_delete_target(skill_dir: Path) -> Optional[str]:
    """Last-line guard before ``shutil.rmtree(skill_dir)``.

    ``_find_skill`` already restricts the dir to a real SKILL.md parent, but
    even a poisoned tree must never recursively delete (1) a path outside every
    known skills root, (2) a skills root itself, or (3) a symlink/junction
    (rmtree would follow it). Returns a refusal string or ``None``.
    """
    from agent.skill_utils import get_all_skills_dirs

    if _is_path_redirect(skill_dir):
        return (
            f"Refusing to delete '{skill_dir}': the skill directory is a "
            f"symlink/junction. Remove the link target manually if intended."
        )
    try:
        resolved = skill_dir.resolve()
    except OSError as exc:
        return f"Refusing to delete '{skill_dir}': could not resolve path ({exc})."

    for root in get_all_skills_dirs():
        try:
            root = root.resolve()
        except OSError:
            continue
        if resolved == root:
            return (
                f"Refusing to delete '{skill_dir}': resolves to the skills root "
                f"itself, which would remove every installed skill."
            )
        try:
            rel = resolved.relative_to(root)
        except ValueError:
            continue
        if rel.parts:
            return None
    return (
        f"Refusing to delete '{skill_dir}': path does not resolve inside any "
        f"known skills root."
    )


# ---------------------------------------------------------------------------
# Ownership / provenance guards
# ---------------------------------------------------------------------------

def _pinned_guard(name: str) -> Optional[str]:
    """Refusal message if *name* is pinned or essential, else None.

    Pin only guards against **deletion** (curator auto-archive and
    ``skill_manage(delete)``); patches/edits stay allowed. Essential skills
    (``ESSENTIAL_SKILLS``) are permanently pinned because the system prompt
    references them. Best-effort: an unreadable sidecar lets the delete through.
    """
    try:
        from agent.skill_utils import ESSENTIAL_SKILLS
        if name in ESSENTIAL_SKILLS:
            return (
                f"Skill '{name}' is essential to Hermes (the agent's own "
                f"operating manual referenced by the system prompt) and "
                f"cannot be deleted. Patches and edits are still allowed."
            )
    except Exception:
        logger.debug("essential-guard lookup failed for %s", name, exc_info=True)
    try:
        from tools import skill_usage
        if skill_usage.get_record(name).get("pinned"):
            return (
                f"Skill '{name}' is pinned and cannot be deleted by "
                f"skill_manage. Ask the user to run "
                f"`hermes curator unpin {name}` if they want to delete it. "
                f"Patches and edits are allowed on pinned skills; only "
                f"deletion is blocked."
            )
    except Exception:
        logger.debug("pinned-guard lookup failed for %s", name, exc_info=True)
    return None


def _background_review_write_guard(
    name: str,
    skill_dir: Path,
    action: str,
) -> Optional[Dict[str, Any]]:
    """Refuse autonomous curator writes to anything but curator-owned sediment.

    Foreground agents may edit external/bundled/hub skills at the user's
    direction; the background review fork has no user in the loop, so it is
    also blocked on pinned skills (stricter than ``_pinned_guard``).
    """
    if not _is_background_review():
        return None

    try:
        from tools import skill_usage
        if skill_usage.get_record(name).get("pinned"):
            return _refusal(
                f"Refusing background curator {action} for pinned skill "
                f"'{name}': pinned skills are off-limits to autonomous "
                "maintenance. Ask the user to run "
                f"`hermes curator unpin {name}` if they want it changed."
            )
    except Exception:
        logger.debug("pinned skill guard lookup failed for %s", name, exc_info=True)

    try:
        from agent.skill_utils import is_external_skill_path
        if is_external_skill_path(skill_dir):
            return _refusal(
                f"Refusing background curator {action} for skill '{name}': "
                "the skill lives in skills.external_dirs, which are "
                "externally owned and read-only to autonomous curation."
            )
    except Exception:
        logger.debug("external skill guard lookup failed for %s", name, exc_info=True)

    try:
        from tools import skill_usage
        for predicate, label in (
            (skill_usage.is_protected_builtin, "protected built-in"),
            (skill_usage.is_hub_installed, "hub-installed"),
            (skill_usage.is_bundled, "bundled"),
        ):
            if predicate(name):
                return _refusal(
                    f"Refusing background curator {action} for {label} "
                    f"skill '{name}'."
                )
        # Not curator-managed (no `created_by: "agent"` marker) => user-owned.
        # A MISSING record and an explicit `created_by: null` must resolve
        # IDENTICALLY: keying on record presence made the policy depend on the
        # guard's own side effect (the first successful write created a null
        # record and the next identical write was refused). Fail closed for
        # both; `hermes curator adopt <name>` is the supported way in.
        usage_rec = skill_usage.load_usage().get(name)
        if not skill_usage._is_curator_managed_record(usage_rec):
            if isinstance(usage_rec, dict):
                _detail = f"created_by={usage_rec.get('created_by')!r}"
            else:
                _detail = "no usage record"
            return _refusal(
                f"Refusing background curator {action} for skill "
                f"'{name}': the skill is not curator-managed ({_detail}). "
                "User-owned skills are off-limits to autonomous curation. "
                f"Run `hermes curator adopt {name}` to opt it in."
            )
    except Exception:
        logger.warning("owned skill guard lookup failed for %s", name, exc_info=True)
        return _refusal(
            f"Refusing background curator {action} for skill '{name}': "
            "agent ownership could not be verified because the provenance "
            "record is unavailable or unreadable."
        )
    return None


def _background_review_read_before_write_guard(
    name: str,
    target: Path,
    action: str,
    file_label: str,
) -> Optional[Dict[str, Any]]:
    """Require review forks to load the exact target before mutating it."""
    if not _is_background_review() or _background_review_has_read(target):
        return None
    return _refusal(
        f"Refusing background curator {action} for skill '{name}': "
        f"the current {file_label} content has not been loaded in this "
        "review turn. Call skill_view(name) for SKILL.md, or "
        "skill_view(name, file_path=...) for a supporting file, then "
        "retry the write using the content just returned.",
        _read_before_write_required=True,
    )


def _background_review_preflight(action: str, name: str) -> Optional[Dict[str, Any]]:
    if action not in {"edit", "patch", "delete", "write_file", "remove_file"}:
        return None
    from tools import skill_manager_tool as _smt

    existing = _smt._find_skill(name)
    if not existing:
        return None
    return _background_review_write_guard(name, existing["path"], action)


def _curator_consolidation_delete_guard(
    name: str, absorbed_into: Optional[str]
) -> Optional[Dict[str, Any]]:
    """Fail closed on unverified deletes during the curator consolidation pass.

    The review fork's only legitimate ``skill_manage(delete)`` is a verified
    consolidation declared via ``absorbed_into=<umbrella>`` (existence is
    validated in ``_delete_skill``). A bare delete (``None`` or ``""``) is the
    fail-open behavior that once archived whole clusters of active skills; the
    deterministic inactivity prune archives via ``skill_usage.archive_skill``
    without ever calling ``skill_manage``, so a bare prune here can only be the
    LLM pass pruning without evidence. Refuse it; keep the skill active.
    """
    if not _is_background_review():
        return None
    if isinstance(absorbed_into, str) and absorbed_into.strip():
        return None
    return _refusal(
        f"Refusing background curator delete of skill '{name}': the "
        "consolidation pass may only archive a skill it has absorbed into "
        "an umbrella. Pass absorbed_into=<umbrella> (the umbrella must "
        "already exist) to record a verified consolidation. Pruning a "
        "skill with no forwarding target is not permitted here — the "
        "deterministic inactivity prune handles staleness archival "
        "separately. Keeping '{name}' active.".format(name=name),
        _fail_closed=True,
    )


# ---------------------------------------------------------------------------
# Org-mirror handling
# ---------------------------------------------------------------------------

def _maybe_auto_propose_org_edit(name: str, skill_path: Path) -> Optional[str]:
    """Submit an org-skill edit upstream when `sync.org_auto_propose` is on.

    Returns a short note for the tool result, or None when nothing happened.
    Never raises: the edit is already saved locally and can be proposed later.
    """
    from tools import skill_manager_tool as _smt

    try:
        from agent.skill_utils import is_org_mirror_path
        from tools import skills_sync_client as ssc

        if not is_org_mirror_path(skill_path, _smt._skills_dir()):
            return None
        if not ssc.sync_org_auto_propose():
            return (
                f"This skill is shared by your organisation. Your edit is "
                f"saved locally and will not be overwritten by org updates. "
                f"Run `hermes sync propose {name}` to share it back."
            )
        result = ssc.propose_skill(name)
        if result.get("proposal_pending"):
            return (
                f"Auto-proposed to your organisation as proposal "
                f"#{result.get('proposal_id')} (pending admin review)."
            )
        return "Auto-proposed to your organisation (merged into the shared set)."
    except Exception as e:
        logger.debug("auto-propose skipped for %s: %s", name, e)
        return (
            f"Edit saved locally. Could not submit it to your organisation "
            f"right now — run `hermes sync propose {name}` to retry."
        )


def _org_mirror_write_guard(name: str, skill_path: Path, action: str) -> Optional[Dict[str, Any]]:
    """Org-shared skills are EDITABLE IN PLACE — this only blocks deletion.

    Refusing every write to `_org/` froze shared skills while personal ones
    kept improving (agents don't fork mid-task). Edits now land in the mirror,
    survive the next org pull (baseline sidecar in skills_sync_client), and
    reach the org via `hermes sync propose` or `sync.org_auto_propose`.
    Deletion stays refused: the mirror is a view of org HEAD, so a local
    delete just comes back, and removing for everyone is an admin action.
    """
    if action not in {"delete", "remove_file"}:
        return None
    from tools import skill_manager_tool as _smt

    try:
        from agent.skill_utils import is_org_mirror_path

        if is_org_mirror_path(skill_path, _smt._skills_dir()):
            return _refusal(
                f"Cannot {action} '{name}' locally: it is shared by your "
                "organisation, so a local delete would just come back on "
                "the next sync. Ask an org admin to remove it for "
                "everyone. (Editing it IS allowed — your changes are kept "
                "and can be proposed back with `hermes sync propose "
                f"{name}`.)"
            )
    except Exception:
        logger.debug("org mirror guard lookup failed for %s", name, exc_info=True)
    return None
