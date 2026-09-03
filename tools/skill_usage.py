"""Skill usage telemetry + provenance for the Curator: a sidecar ``~/.hermes/skills/.usage.json`` keyed by
skill name (never frontmatter — keeps telemetry out of user-authored SKILL.md and off bundled/hub skills).

Counter bumps are best-effort (failures log at DEBUG, never break the tool call); writes are atomic
(tempfile + os.replace) under a cross-process lock. Curator management is an explicit ``created_by: agent``
marker written by skill_manage — never inferred from location. Lifecycle: active -> stale -> archived
(moved to .archive/); ``pinned`` opts out of auto transitions, orthogonal to state.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from contextlib import contextmanager, suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Set, Tuple

from hermes_constants import get_hermes_home
from agent.skill_utils import is_excluded_skill_path, is_external_skill_path

logger = logging.getLogger(__name__)

# fcntl is Unix-only; on Windows use msvcrt for file locking.
msvcrt = None
try:
    import fcntl
except ImportError:  # pragma: no cover - platform-specific fallback
    fcntl = None
    try:
        import msvcrt
    except ImportError:
        pass


STATE_ACTIVE = "active"
STATE_STALE = "stale"
STATE_ARCHIVED = "archived"
_VALID_STATES = {STATE_ACTIVE, STATE_STALE, STATE_ARCHIVED}

# Load-bearing built-ins (by frontmatter ``name``) the curator must NEVER archive or consolidate, regardless
# of ``curator.prune_builtins``, pin state, or LLM judgment — silently archiving one turns its slash command
# into "Unknown command". Keep tiny; it is not a substitute for ``curator.prune_builtins: false``.
# (``plan`` used to live here; it is now a first-class command with no skill on disk.)
PROTECTED_BUILTIN_SKILLS: Set[str] = set()


def is_protected_builtin(skill_name: str) -> bool:
    """Exempt from archival/consolidation on every path: auto transitions, LLM pass, direct archive_skill."""
    return skill_name in PROTECTED_BUILTIN_SKILLS


def _skills_dir() -> Path:
    return get_hermes_home() / "skills"


def _usage_file() -> Path:
    return _skills_dir() / ".usage.json"


def _archive_dir() -> Path:
    return _skills_dir() / ".archive"


def _flock(fd, lock: bool) -> None:
    if fcntl:
        fcntl.flock(fd, fcntl.LOCK_EX if lock else fcntl.LOCK_UN)
    else:
        fd.seek(0)
        msvcrt.locking(fd.fileno(), msvcrt.LK_LOCK if lock else msvcrt.LK_UNLCK, 1)


@contextmanager
def _usage_file_lock():
    """Serialize .usage.json read-modify-write cycles across processes."""
    lock_path = _usage_file().with_suffix(".json.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if fcntl is None and msvcrt is None:
        yield
        return
    if msvcrt and (not lock_path.exists() or lock_path.stat().st_size == 0):
        lock_path.write_text(" ", encoding="utf-8")
    fd = open(lock_path, "r+" if msvcrt else "a+", encoding="utf-8")
    try:
        _flock(fd, True)
        yield
    finally:
        with suppress(OSError, IOError):
            _flock(fd, False)
        fd.close()


def _atomic_write(path: Path, prefix: str, write: Callable[[Any], None]) -> None:
    """Write *path* via tempfile + fsync + os.replace; the temp file is removed on failure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=prefix, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            write(f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        with suppress(OSError):
            os.unlink(tmp)
        raise


def _read_lines(path: Path, fail_log: str) -> List[str]:
    """Stripped, non-empty lines of a small metadata file ([] if missing/unreadable)."""
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as e:
        logger.debug(fail_log, e)
        return []
    return [s for s in (line.strip() for line in lines) if s]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_iso_timestamp(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed


def latest_activity_at(record: Dict[str, Any]) -> Optional[str]:
    """Newest use/view/patch timestamp. Creation time is excluded so never-active skills stay
    distinguishable; lifecycle code falls back to ``created_at`` itself."""
    stamps = [(dt, str(raw)) for raw in (record.get(k) for k in ("last_used_at", "last_viewed_at", "last_patched_at"))
              if (dt := _parse_iso_timestamp(raw)) is not None]
    return max(stamps, key=lambda t: t[0])[1] if stamps else None


def activity_count(record: Dict[str, Any]) -> int:
    """Total observed use+view+patch events."""
    total = 0
    for key in ("use_count", "view_count", "patch_count"):
        try:
            total += int(record.get(key) or 0)
        except (TypeError, ValueError):
            continue
    return total


# --- Provenance — which skills are agent-created (and thus eligible for curation) ---
def _read_bundled_manifest_names() -> Set[str]:
    """Names from ``.bundled_manifest`` ("name:hash" per line); empty if missing/unreadable."""
    lines = _read_lines(_skills_dir() / ".bundled_manifest", "Failed to read bundled manifest: %s")
    return {n for n in (line.split(":", 1)[0].strip() for line in lines) if n}


def _read_hub_installed_names() -> Set[str]:
    """Names installed via the Skills Hub (``.hub/lock.json``, see tools/skills_hub.py::HubLockFile), plus the
    frontmatter name of each ``install_path`` that resolves inside the skills dir."""
    lock_path = _skills_dir() / ".hub" / "lock.json"
    if not lock_path.exists():
        return set()
    try:
        # errors="replace": hub descriptions can carry Windows-1252 high bytes; a strict read raises
        # UnicodeDecodeError (a ValueError, not caught below) and would 500 the whole /api/skills endpoint.
        data = json.loads(lock_path.read_text(encoding="utf-8", errors="replace"))
        installed = (data.get("installed") or {}) if isinstance(data, dict) else None
        if not isinstance(installed, dict):
            return set()
        names = {str(k) for k in installed}
        skills_dir = _skills_dir()
        for entry in installed.values():
            install_path = entry.get("install_path") if isinstance(entry, dict) else None
            if not isinstance(install_path, str) or not install_path.strip():
                continue
            try:
                resolved = (skills_dir / install_path).resolve()
                resolved.relative_to(skills_dir.resolve())
            except (OSError, ValueError):
                continue
            if (resolved / "SKILL.md").exists():
                names.add(_read_skill_name(resolved / "SKILL.md", fallback=resolved.name))
        return names
    except (OSError, json.JSONDecodeError) as e:
        logger.debug("Failed to read hub lock file: %s", e)
    return set()


def _prune_builtins_enabled() -> bool:
    """``curator.prune_builtins`` (default True). Lazy config import keeps this module importable in the
    update/sync context. The real mass-prune safety is seed-on-first-sight, not this flag."""
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
        cur = cfg.get("curator") if isinstance(cfg, dict) else None
        if isinstance(cur, dict):
            return bool(cur.get("prune_builtins", True))
    except Exception as e:  # pragma: no cover — best-effort config read
        logger.debug("Failed to read curator.prune_builtins: %s", e)
    return True


def read_suppressed_names() -> Set[str]:
    """Built-ins the curator pruned (one name per line in ``.curator_suppressed``); the update-time re-seeder
    must leave these archived, otherwise ``hermes update`` would re-copy the bundled skill."""
    lines = _read_lines(_skills_dir() / ".curator_suppressed", "Failed to read curator suppression list: %s")
    return {line for line in lines if not line.startswith("#")}


def _toggle_suppressed_name(skill_name: str, *, add: bool) -> None:
    """Add (built-in pruned) or drop (restored) *skill_name* in the suppression list; no-op when unchanged."""
    if not skill_name or (skill_name in (names := read_suppressed_names())) == add:
        return
    (names.add if add else names.discard)(skill_name)
    data = "\n".join(sorted(names)) + ("\n" if names else "")
    try:
        _atomic_write(_skills_dir() / ".curator_suppressed", ".curator_suppressed_", lambda f: f.write(data))
    except Exception as e:
        logger.debug("Failed to write curator suppression list: %s", e, exc_info=True)


def _iter_skill_mds(base: Path, *, local_only: bool) -> Iterator[Tuple[str, Path]]:
    """``(frontmatter name, SKILL.md)`` for flat and category-nested skills under *base*, skipping metadata/VCS/
    venv/cache dirs. *local_only* also skips external skill dirs mounted below the tree — discovery may see
    them, autonomous curation must not."""
    for skill_md in base.rglob("SKILL.md"):
        if not (is_excluded_skill_path(skill_md) or (local_only and is_external_skill_path(skill_md))):
            yield _read_skill_name(skill_md, fallback=skill_md.parent.name), skill_md


def _scan_local_skills(keep: Callable[[str, Path, Set[str], Dict[str, Any]], bool]) -> List[str]:
    """Sorted unique names of local skills passing *keep(name, skill_md, bundled, usage)*. Hub-installed and
    protected built-ins never reach *keep*."""
    base = _skills_dir()
    if not base.exists():
        return []
    hub, bundled, usage = _read_hub_installed_names(), _read_bundled_manifest_names(), load_usage()
    return sorted({
        name for name, skill_md in _iter_skill_mds(base, local_only=True)
        if name not in hub and not is_protected_builtin(name) and keep(name, skill_md, bundled, usage)})


def list_agent_created_skill_names() -> List[str]:
    """Skills the curator may manage: agent-authored (``created_by: agent`` record) plus, when
    ``curator.prune_builtins`` is on, bundled built-ins (inactivity anchored on first sight). Never hub skills."""
    prune_builtins = _prune_builtins_enabled()  # read once, before the walk

    def _keep(name: str, _skill_md: Path, bundled: Set[str], usage: Dict[str, Any]) -> bool:
        # Built-ins never carry a curator-managed record, so the record gate applies only to local skills.
        return prune_builtins if name in bundled else _is_curator_managed_record(usage.get(name))
    return _scan_local_skills(_keep)


def list_archived_skill_names() -> List[str]:
    """Skills in ``.archive/`` — flat layout (``archive_skill`` flattens), so dir name == skill name."""
    archive_root = _archive_dir()
    return sorted({p.name for p in archive_root.iterdir() if p.is_dir()}) if archive_root.exists() else []


def _read_skill_name(skill_md: Path, fallback: str) -> str:
    """The frontmatter ``name:`` field of a SKILL.md (first 4000 chars), else *fallback*."""
    try:
        text = skill_md.read_text(encoding="utf-8", errors="replace")[:4000]
    except OSError:
        return fallback
    in_frontmatter = False
    for stripped in (line.strip() for line in text.split("\n")):
        if stripped == "---":
            if in_frontmatter:
                break
            in_frontmatter = True
        elif in_frontmatter and stripped.startswith("name:"):
            value = stripped.split(":", 1)[1].strip().strip("\"'")
            if value:
                return value
    return fallback


def is_agent_created(skill_name: str) -> bool:
    """Neither bundled nor hub-installed (and not only present in an external dir)."""
    if skill_name in _read_bundled_manifest_names() | _read_hub_installed_names():
        return False
    return _find_skill_dir(skill_name) is not None or _find_external_skill_dir(skill_name) is None


def is_hub_installed(skill_name: str) -> bool:
    return skill_name in _read_hub_installed_names()


def is_bundled(skill_name: str) -> bool:
    return skill_name in _read_bundled_manifest_names()


def _external_read_only_message(skill_name: str) -> str:
    return f"skill '{skill_name}' lives in skills.external_dirs; external skills are read-only to the curator"


def is_curation_eligible(skill_name: str, skill_path: Optional[Path] = None) -> bool:
    """May the curator track/archive this skill? Agent-created: yes. Bundled: only with
    ``curator.prune_builtins``. Hub-installed / external-dir / protected built-ins: never (external owner).
    Org-shared skills are eligible for improvement (edits stay local) but protected from ARCHIVE/DELETE elsewhere."""
    if ((skill_path is not None and is_external_skill_path(skill_path)) or is_protected_builtin(skill_name)
            or is_hub_installed(skill_name)):
        return False
    if is_bundled(skill_name):
        return _prune_builtins_enabled()
    local_dir = _find_skill_dir(skill_name)
    return not is_external_skill_path(local_dir) if local_dir is not None else _find_external_skill_dir(skill_name) is None


def _is_curator_managed_record(record: Any) -> bool:
    """The on-disk ``created_by`` field reads like provenance but is a curator-management OPT-IN policy flag:
    ``"agent"`` means "curator-managed", not proof of authorship (the user can flip it via ``hermes curator
    adopt``). The name is kept because it is already in every user's ``.usage.json``."""
    return isinstance(record, dict) and (record.get("created_by") == "agent" or record.get("agent_created") is True)


def is_curator_managed(skill_name: str) -> bool:
    """Policy-intent alias for the ``created_by`` marker check."""
    return _is_curator_managed_record(load_usage().get(skill_name))


def list_unmanaged_skill_names() -> List[str]:
    """Curation-ELIGIBLE skills with no provenance marker: records predating ``created_by``, or FOREGROUND
    ``skill_manage(action="create")`` results (skills a user asks for belong to the user). Invisible to
    ``curated_report()`` and every automatic transition; surfaced by ``hermes curator status`` and handed over
    only by explicit ``hermes curator adopt`` — provenance is a declaration, never inferred from activity."""
    def _keep(name: str, skill_md: Path, bundled: Set[str], usage: Dict[str, Any]) -> bool:
        return name not in bundled and not _is_curator_managed_record(usage.get(name)) and is_curation_eligible(name, skill_md)
    return _scan_local_skills(_keep)


def unmanaged_report() -> List[Dict[str, Any]]:
    """Rows for :func:`list_unmanaged_skill_names`. ``has_provenance_key`` is False when the record has no
    ``created_by`` key (pre-dates the mechanism), True when present but unset (foreground create) — explains
    WHY a skill is unmanaged; not a signal to adopt on."""
    usage = load_usage()
    return [
        _report_row(name, raw, has_provenance_key=isinstance(raw, dict) and "created_by" in raw,
                    has_record=isinstance(raw, dict))
        for name, raw in ((n, usage.get(n)) for n in list_unmanaged_skill_names())]


def adopt_skill(skill_name: str) -> Tuple[bool, str]:
    """Hand *skill_name* to the curator by user declaration — writes the same ``created_by: agent`` marker the
    background review fork writes. The inactivity clock is NOT reset. Refuses hub-installed, external, bundled
    and protected built-in skills. Returns (ok, message)."""
    if not skill_name:
        return False, "no skill name given"
    if is_protected_builtin(skill_name):
        return False, f"'{skill_name}' is a protected built-in; the curator never manages it"
    if is_hub_installed(skill_name):
        return False, f"'{skill_name}' is hub-installed; its upstream owns it"
    if is_bundled(skill_name):
        # Bundled skills are governed by prune_builtins; stamping created_by=agent would change nothing.
        return False, f"'{skill_name}' is a bundled built-in — it is governed by curator.prune_builtins, not by adoption"
    skill_dir = _find_skill_dir(skill_name)
    if skill_dir is None:
        if _find_external_skill_dir(skill_name) is not None:
            return False, f"'{skill_name}' lives in skills.external_dirs and is read-only to the curator"
        return False, f"skill '{skill_name}' not found"
    if is_external_skill_path(skill_dir):
        return False, _external_read_only_message(skill_name)
    if _is_curator_managed_record(load_usage().get(skill_name)):
        return True, f"'{skill_name}' is already curator-managed"
    mark_agent_created(skill_name)
    if not _is_curator_managed_record(load_usage().get(skill_name)):
        return False, f"could not mark '{skill_name}' as curator-managed"
    return True, f"adopted '{skill_name}' into curator management"


# --- Sidecar I/O ---
def _empty_record() -> Dict[str, Any]:
    return {
        "created_by": None, "use_count": 0, "view_count": 0, "last_used_at": None, "last_viewed_at": None,
        "patch_count": 0, "patch_generation": 0, "last_reused_patch_generation": 0, "last_patched_at": None,
        "created_at": _now_iso(), "state": STATE_ACTIVE, "pinned": False, "archived_at": None}


def _backfilled(rec: Any) -> Dict[str, Any]:
    """*rec* with every missing default key filled in (a fresh record when not a dict)."""
    if not isinstance(rec, dict):
        return _empty_record()
    for k, v in _empty_record().items():
        rec.setdefault(k, v)
    return rec


def _report_row(name: str, raw: Any, **extra: Any) -> Dict[str, Any]:
    row = {"name": name, **_backfilled(raw), **extra}
    row.update(last_activity_at=latest_activity_at(row), activity_count=activity_count(row))
    return row


def load_usage() -> Dict[str, Dict[str, Any]]:
    """The whole .usage.json map (non-dict values dropped); {} on missing/corrupt."""
    path = _usage_file()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return {str(k): v for k, v in data.items() if isinstance(v, dict)} if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError) as e:
        logger.debug("Failed to read %s: %s", path, e)
        return {}


def save_usage(data: Dict[str, Dict[str, Any]]) -> bool:
    """Write the usage map atomically; True when it committed."""
    path = _usage_file()
    try:
        _atomic_write(path, ".usage_", lambda f: json.dump(data, f, indent=2, sort_keys=True, ensure_ascii=False))
        return True
    except Exception as e:
        logger.debug("Failed to write %s: %s", path, e, exc_info=True)
        return False


def get_record(skill_name: str) -> Dict[str, Any]:
    """The (backfilled) record for *skill_name*; fresh defaults if missing."""
    return _backfilled(load_usage().get(skill_name))


def _locked_update(
    skill_name: str, op: Callable[[Dict[str, Dict[str, Any]]], Tuple[Any, bool]], fail_log: str,
    guard: Optional[Callable[[], bool]] = None) -> Any:
    """Run *op(data) -> (result, dirty)* on the usage map under the file lock; save only when dirty. *guard*
    runs before the lock is taken. Returns result, or None when the guard failed, the save did not land, or
    anything raised (logged at DEBUG via *fail_log*)."""
    try:
        if guard is not None and not guard():
            return None
        with _usage_file_lock():
            data = load_usage()
            result, dirty = op(data)
            return None if dirty and not save_usage(data) else result
    except Exception as e:
        logger.debug(fail_log, skill_name, e, exc_info=True)
        return None


def seed_record_if_missing(skill_name: str) -> None:
    """Persist a baseline record for a curation-eligible skill so its inactivity clock is anchored at first
    sight (``created_at`` = now), not at epoch. No-op if a record exists or the skill isn't eligible."""
    if not skill_name or not is_curation_eligible(skill_name):
        return

    def _seed(data):
        if missing := not isinstance(data.get(skill_name), dict):
            data[skill_name] = _empty_record()
        return None, missing
    _locked_update(skill_name, _seed, "skill_usage.seed_record_if_missing(%s) failed: %s")


def _mutate(skill_name: str, mutator, *, require_curation_eligible: bool = False) -> Any:
    """Load, apply *mutator(record)* in place, save; returns the mutator result (None if nothing landed).
    Telemetry is recorded for ANY skill (observability is orthogonal to curation); lifecycle mutators pass
    ``require_curation_eligible=True`` so they never write state onto a skill the curator can't manage."""
    if not skill_name:
        return None

    def _apply(data):
        rec = data[skill_name] = data[skill_name] if isinstance(data.get(skill_name), dict) else _empty_record()
        return mutator(rec), True
    guard = (lambda: is_curation_eligible(skill_name)) if require_curation_eligible else None
    return _locked_update(skill_name, _apply, "skill_usage._mutate(%s) failed: %s", guard)


def _set_field(skill_name: str, key: str, value: Any) -> bool:
    """Curation-gated single-field write; True only when the write landed."""
    return bool(_mutate(skill_name, lambda rec: rec.update({key: value}) or True, require_curation_eligible=True))


def _non_negative_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _bump(rec: Dict[str, Any], count_key: str, ts_key: str) -> None:
    rec[count_key] = _non_negative_int(rec.get(count_key)) + 1
    rec[ts_key] = _now_iso()


def telemetry_provenance(skill_name: str, record: Optional[Dict[str, Any]] = None) -> str:
    """Bounded provenance label for shared skill metrics."""
    if is_hub_installed(skill_name) or is_bundled(skill_name):
        return "installed"
    if ":" in skill_name:
        with suppress(Exception):
            from hermes_cli.plugins import get_plugin_manager
            if get_plugin_manager().find_plugin_skill(skill_name) is not None:
                return "installed"
    created_by = record.get("created_by") if isinstance(record, dict) else None
    if created_by in ("installed", "agent"):
        return {"installed": "installed", "agent": "agent_created"}[created_by]
    if _find_external_skill_dir(skill_name) is not None:
        return "external"
    if _find_skill_dir(skill_name) is not None or isinstance(record, dict):
        return "local"
    return "unknown"


def _emit_skill_lifecycle(
    skill_name: str, action: str, *, record: Optional[Dict[str, Any]] = None,
    task_id: Optional[str] = None, session_id: Optional[str] = None, **facts: Any) -> None:
    """Best-effort lifecycle hook after an authoritative state change; absent facts are sent as None."""
    try:
        from hermes_cli.lifecycle import has_hook, invoke_hook
        if not has_hook("on_skill_lifecycle"):
            return
        invoke_hook(
            "on_skill_lifecycle", action=action, skill_name=skill_name,
            provenance=telemetry_provenance(skill_name, record), task_id=task_id or "", session_id=session_id or "",
            use_count=facts.get("use_count"), reused=facts.get("reused"), reuse_after_patch=facts.get("reuse_after_patch"),
        )
    except Exception:
        logger.debug("skill_usage lifecycle hook failed for %s/%s", skill_name, action, exc_info=True)


def _mutate_and_emit(skill_name: str, action: str, mutator: Callable[[Dict[str, Any]], Dict[str, Any]],
                     **hook_kwargs: Any) -> None:
    """``_mutate`` then emit *action* with the mutator's facts as the record — only if the write landed."""
    facts = _mutate(skill_name, mutator)
    if isinstance(facts, dict):
        hook_kwargs.update({k: facts[k] for k in ("use_count", "reused", "reuse_after_patch") if k in facts})
        _emit_skill_lifecycle(skill_name, action, record=facts, **hook_kwargs)


# --- Public counter-bump helpers — telemetry for ALL skills regardless of provenance (observability only) ---
def bump_view(skill_name: str) -> None:
    _mutate(skill_name, lambda rec: _bump(rec, "view_count", "last_viewed_at"))


def bump_use(skill_name: str, *, task_id: Optional[str] = None, session_id: Optional[str] = None) -> None:
    """Skill actively used (loaded into the prompt path / referenced from an assistant turn)."""
    def _apply(rec: Dict[str, Any]) -> Dict[str, Any]:
        previous_use_count = _non_negative_int(rec.get("use_count"))
        patch_generation = _non_negative_int(rec.get("patch_generation"))
        last_reused_generation = min(_non_negative_int(rec.get("last_reused_patch_generation")), patch_generation)
        reused = previous_use_count > 0
        reuse_after_patch = reused and patch_generation > last_reused_generation
        rec.update(
            use_count=previous_use_count + 1, last_used_at=_now_iso(), patch_generation=patch_generation,
            last_reused_patch_generation=patch_generation if reuse_after_patch else last_reused_generation)
        return {
            "created_by": rec.get("created_by"), "use_count": rec["use_count"],
            "reused": reused, "reuse_after_patch": reuse_after_patch}
    _mutate_and_emit(skill_name, "loaded", _apply, task_id=task_id, session_id=session_id)


def bump_patch(
    skill_name: str, *, action: str = "patch", task_id: Optional[str] = None, session_id: Optional[str] = None) -> None:
    """Called from skill_manage (patch/edit)."""
    def _apply(rec: Dict[str, Any]) -> Dict[str, Any]:
        _bump(rec, "patch_count", "last_patched_at")
        rec["patch_generation"] = _non_negative_int(rec.get("patch_generation")) + 1
        return {"created_by": rec.get("created_by")}
    lifecycle_action = "patched" if action == "patch" else "edited"
    _mutate_and_emit(skill_name, lifecycle_action, _apply, task_id=task_id, session_id=session_id)


def record_created(
    skill_name: str, *, agent_created: bool, task_id: Optional[str] = None, session_id: Optional[str] = None) -> None:
    """Persist creation provenance and emit a create fact. A successful create is a new logical skill even
    if stale sidecar state survived an earlier deletion, so the record is reset."""
    def _apply(rec: Dict[str, Any]) -> Dict[str, Any]:
        rec.clear()
        rec.update(_empty_record(), **({"created_by": "agent"} if agent_created else {}))
        return {"created_by": rec["created_by"]}
    _mutate_and_emit(skill_name, "created", _apply, task_id=task_id, session_id=session_id)


def record_installed(skill_name: str) -> None:
    """Record a successful Skills Hub install without exporting its name."""
    _mutate_and_emit(skill_name, "installed",
                     lambda rec: rec.update(created_by="installed", state=STATE_ACTIVE, archived_at=None) or {"created_by": "installed"})


def mark_agent_created(skill_name: str) -> None:
    """Opt a skill into curator management — the only thing that makes it eligible for automatic curation."""
    _set_field(skill_name, "created_by", "agent")


def set_state(skill_name: str, state: str) -> None:
    """Set lifecycle state; no-op for an invalid state or a skill the curator can't manage. Emits
    archived/stale/restored (active<-archived); active<-stale emits nothing."""
    if state not in _VALID_STATES:
        logger.debug("set_state: invalid state %r for %s", state, skill_name)
        return

    def _apply(rec: Dict[str, Any]) -> Dict[str, Any]:
        previous_state = rec.get("state")
        facts = {"changed": previous_state != state, "created_by": rec.get("created_by")}
        if facts["changed"]:
            rec["state"] = state
            if state != STATE_STALE:
                rec["archived_at"] = _now_iso() if state == STATE_ARCHIVED else None
            facts["previous_state"] = previous_state
        return facts
    facts = _mutate(skill_name, _apply, require_curation_eligible=True)
    if not isinstance(facts, dict) or not facts.get("changed"):
        return
    restored = state == STATE_ACTIVE and facts.get("previous_state") == STATE_ARCHIVED
    action = "restored" if restored else {STATE_ARCHIVED: "archived", STATE_STALE: "stale"}.get(state)
    if action is not None:
        _emit_skill_lifecycle(skill_name, action, record=facts)


def set_pinned(skill_name: str, pinned: bool) -> bool:
    """Set/clear the pin flag; False when the write did not land (not curation-eligible) so callers can
    report failure instead of a false success."""
    return _set_field(skill_name, "pinned", bool(pinned))


def set_sync(skill_name: str, sync: bool) -> None:
    """Sync is OPT-IN (``sync: true`` on the record, read by ``tools.skills_sync_client.list_synced_skill_names``);
    gated on curation eligibility so bundled/hub/external skills can't be marked."""
    _set_field(skill_name, "sync", bool(sync))


def is_sync_enabled(skill_name: str) -> bool:
    return get_record(skill_name).get("sync") is True


def forget(skill_name: str) -> None:
    """Drop a skill's usage entry entirely (skill deleted)."""
    if skill_name:
        _locked_update(skill_name, lambda d: (None, d.pop(skill_name, None) is not None), "skill_usage.forget(%s) failed: %s")


# --- Archive / restore ---
def _relocate(src: Path, dest: Path, skill_name: str, action: str, **capture_kwargs: Any) -> Tuple[bool, str]:
    """Move *src* to *dest* for *action* ("archive" | "restore") with an audit-ledger entry around it, then
    apply the suppression + state side effects. Ledger capture is best-effort and never blocks the move; the
    rename falls back to shutil.move across devices. Returns (ok, message)."""
    try:
        from tools import skill_ledger as _ledger
        _ledger_before = _ledger.capture_before(src, **capture_kwargs)
    except Exception:
        _ledger = _ledger_before = None  # type: ignore[assignment]

    try:
        src.rename(dest)
    except OSError:
        import shutil
        try:
            shutil.move(str(src), str(dest))
        except Exception as e:
            return False, f"failed to {action}: {e}"

    if action == "archive":
        if is_bundled(skill_name):  # pruning a built-in only sticks if the re-seeder is told to leave it alone
            _toggle_suppressed_name(skill_name, add=True)
        set_state(skill_name, STATE_ARCHIVED)
    else:
        _toggle_suppressed_name(skill_name, add=False)
        set_state(skill_name, STATE_ACTIVE)
    with suppress(Exception):
        if _ledger is not None:
            _ledger.record_mutation(
                action, skill_name, before=_ledger_before if _ledger_before is not None else [], after_root=dest)
    return True, f"{action}d to {dest}"


def archive_skill(skill_name: str) -> Tuple[bool, str]:
    """Move a curator-eligible skill dir to ``.archive/`` (flattened; timestamp suffix on collision). Never
    hub skills; bundled built-ins only with ``curator.prune_builtins`` (and then suppressed from re-seeding)."""
    skill_dir = _find_skill_dir(skill_name)
    if skill_dir is None and _find_external_skill_dir(skill_name) is not None:
        return False, _external_read_only_message(skill_name)
    if not is_curation_eligible(skill_name, skill_dir):
        if is_protected_builtin(skill_name):
            return False, f"skill '{skill_name}' is a protected built-in; it backs load-bearing UX and is never archived or consolidated"
        if is_hub_installed(skill_name):
            return False, f"skill '{skill_name}' is hub-installed; never archive"
        return False, f"skill '{skill_name}' is a bundled built-in; enable curator.prune_builtins to allow pruning it"
    if skill_dir is None:
        return False, f"skill '{skill_name}' not found"
    if is_external_skill_path(skill_dir):
        return False, _external_read_only_message(skill_name)

    archive_root = _archive_dir()
    try:
        archive_root.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        return False, f"failed to create archive dir: {e}"
    dest = archive_root / skill_dir.name
    if dest.exists():
        dest = archive_root / f"{skill_dir.name}-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
    # complete_package: consolidation may have re-homed support files first, so a disk-only capture can come
    # back hollow; the fill from the newest curator backup keeps rollback restorable.
    return _relocate(skill_dir, dest, skill_name, "archive", complete_package=True, skill=skill_name)


def restore_skill(skill_name: str) -> Tuple[bool, str]:
    """Move an archived skill back to the flat top-level layout (category nesting is NOT reconstructed).

    Refuses names that now collide with a hub skill, or a bundled built-in unless ``curator.prune_builtins``
    is on (then restoring is the documented way to lift a prune) — either would shadow the upstream copy."""
    if is_hub_installed(skill_name):
        return False, f"skill '{skill_name}' is now hub-installed; restore would shadow the upstream version"
    if is_bundled(skill_name) and not _prune_builtins_enabled():
        return False, f"skill '{skill_name}' is now bundled; restore would shadow the upstream version"
    archive_root = _archive_dir()
    if not archive_root.exists():
        return False, "no archive directory"

    # Exact name first (recursive: older archive paths left nested layouts), then the timestamped-duplicate
    # fallback. Only "<skill>-YYYYMMDDHHMMSS" (14 digits) counts — a bare startswith("<skill>-") would let
    # restoring "git" pull an archived "git-helpers" out and rename it, destroying the sibling's only copy.
    candidates = [p for p in archive_root.rglob("*") if p.is_dir() and p.name == skill_name]
    if not candidates:
        prefix = f"{skill_name}-"
        candidates = sorted(
            (p for p in archive_root.rglob("*") if p.is_dir() and p.name.startswith(prefix)
             and len(p.name) - len(prefix) == 14 and p.name[len(prefix):].isdigit()),
            reverse=True)
    if not candidates:
        return False, f"skill '{skill_name}' not found in archive"

    dest = _skills_dir() / skill_name
    if dest.exists():
        return False, f"destination already exists: {dest}"
    return _relocate(candidates[0], dest, skill_name, "restore")


def _match_skill_dir(skill_mds: Iterable[Path], skill_name: str) -> Optional[Path]:
    return next((p.parent for p in skill_mds if _read_skill_name(p, fallback=p.parent.name) == skill_name), None)


def _find_skill_dir(skill_name: str) -> Optional[Path]:
    """Skill dir by frontmatter ``name`` (flat or category-nested). The gated index iterator makes org mirrors
    resolve ONLY for the active org (stale ``_org/<other>/`` trees never match)."""
    base = _skills_dir()
    if not base.exists():
        return None
    from agent.skill_utils import iter_skill_index_files
    return _match_skill_dir(
        (p for p in iter_skill_index_files(base, "SKILL.md") if not is_external_skill_path(p)), skill_name)


def _find_external_skill_dir(skill_name: str) -> Optional[Path]:
    """Skill dir under configured external dirs by frontmatter name."""
    from agent.skill_utils import get_all_skills_dirs
    for base in (b for b in get_all_skills_dirs()[1:] if b.exists()):
        found = _match_skill_dir((p for p in base.rglob("SKILL.md") if not is_excluded_skill_path(p)), skill_name)
        if found is not None:
            return found
    return None


# --- Reporting — for the curator CLI / slash command ---
def curated_report() -> List[Dict[str, Any]]:
    """One backfilled row per curator-managed skill with ``provenance`` ('agent'|'bundled'|'hub') and
    ``_persisted`` (a real record exists — the curator seeds the inactivity clock for fresh backfills instead of
    treating them as ancient). Bundled skills only with ``curator.prune_builtins``; hub skills never."""
    data = load_usage()
    names = set(list_agent_created_skill_names())
    # A pinned-but-unmanaged skill must stay visible or its pin silently vanishes from `curator status`; the
    # local-dir guard keeps stale records for deleted dirs from rendering as ghost rows (`curator unpin` cleans up).
    names.update(
        name for name, rec in data.items()
        if isinstance(rec, dict) and rec.get("pinned") and is_curation_eligible(name) and _find_skill_dir(name) is not None
    )
    rows = [_report_row(name, data.get(name), _persisted=isinstance(data.get(name), dict)) for name in sorted(names)]
    for row in rows:
        row["provenance"] = provenance(row["name"])
    return rows


def provenance(skill_name: str) -> str:
    """'hub', 'bundled', or 'agent' (the latter covers agent-authored AND local manually-authored skills)."""
    return "hub" if is_hub_installed(skill_name) else "bundled" if is_bundled(skill_name) else "agent"


def usage_report() -> List[Dict[str, Any]]:
    """Usage rows for EVERY skill on disk (built-ins and hub included), with ``provenance`` and ``_persisted`` —
    unlike ``curated_report()``, which is scoped to curator-managed candidates."""
    base = _skills_dir()
    if not base.exists():
        return []
    data = load_usage()
    rows: Dict[str, Dict[str, Any]] = {}
    for name, _skill_md in _iter_skill_mds(base, local_only=False):
        if name not in rows:
            rows[name] = _report_row(name, data.get(name), provenance=provenance(name),
                                     _persisted=isinstance(data.get(name), dict))
    return [rows[name] for name in sorted(rows)]
