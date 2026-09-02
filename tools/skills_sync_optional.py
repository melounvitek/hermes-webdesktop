"""Official optional-skill provenance: hub-lock backfill and restore.

Extracted from ``tools.skills_sync``. Profile-scoped paths and patchable
helpers are resolved through ``_ss()`` at call time so tests and multi-profile
runtimes that patch ``tools.skills_sync`` globals keep working.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Dict, Iterator, List, Optional, Set, Tuple

from agent.skill_utils import is_excluded_skill_path
from utils import atomic_write_text

logger = logging.getLogger("tools.skills_sync")


def _ss():
    from tools import skills_sync

    return skills_sync


def _content_hash(directory: Path) -> str:
    """Same hash style the skills hub lock uses; hashing is provenance metadata
    only, so fall back to the local MD5 if guard deps are unavailable."""
    try:
        from tools.skills_guard import content_hash

        return content_hash(directory)
    except Exception:
        return _ss()._dir_hash(directory)


def _safe_rel_install_path(path: Path, base: Path) -> str:
    """Return a normalized relative POSIX path, rejecting traversal/absolute paths."""
    posix = path.relative_to(base).as_posix()
    pure = PurePosixPath(posix)
    parts = [part for part in pure.parts if part not in {"", "."}]
    if pure.is_absolute() or not parts or ".." in parts:
        raise ValueError(f"Unsafe optional skill path: {posix}")
    return "/".join(parts)


def _skill_file_list(skill_dir: Path) -> List[str]:
    """List files inside a skill directory in lock-file format."""
    return [f.relative_to(skill_dir).as_posix() for f in sorted(skill_dir.rglob("*")) if f.is_file()]


def _hub_lock_path() -> Path:
    return _ss()._skills_dir() / ".hub" / "lock.json"


def _load_hub_lock() -> Optional[dict]:
    """Parse the skills-hub lock; None when missing or unreadable."""
    try:
        return json.loads(_hub_lock_path().read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def _hub_lock_entries(data: Optional[dict]) -> List[dict]:
    return [e for e in ((data or {}).get("installed") or {}).values() if isinstance(e, dict)]


def _read_hub_install_paths() -> Set[str]:
    """Install paths recorded in the hub lock, as POSIX strings. Hub-installed skills
    are owned by the hub, never by bundled sync: rename recovery must not move them even
    when content matches a bundled origin hash, or the lock's ``install_path`` dangles."""
    return {str(e["install_path"]).strip("/") for e in _hub_lock_entries(_load_hub_lock()) if e.get("install_path")}


def _write_hub_lock(lock_path: Path, data: dict) -> None:
    """Atomic write so a crash mid-write can't wipe all provenance (the
    JSONDecodeError fallback in the reader resets ``installed`` to empty)."""
    atomic_write_text(lock_path, json.dumps(data, indent=2, ensure_ascii=False) + "\n", tmp_prefix=".lock_")


def _iter_optional_skills(optional_dir: Path, *, root_relative: bool) -> Iterator[Tuple[Path, Path, str]]:
    """Yield ``(skill_md, src, install_path)`` for every safe official optional skill."""
    for skill_md in sorted(optional_dir.rglob("SKILL.md")):
        if root_relative and is_excluded_skill_path(skill_md.relative_to(optional_dir), root=optional_dir):
            continue
        if not root_relative and is_excluded_skill_path(skill_md):
            continue
        try:
            yield skill_md, skill_md.parent, _safe_rel_install_path(skill_md.parent, optional_dir)
        except ValueError as e:
            logger.debug("Skipping optional skill with unsafe path %s: %s", skill_md.parent, e)


def _optional_skill_index() -> Dict[str, Tuple[str, str, Path]]:
    """Official optional skills keyed by BOTH folder name and frontmatter name, so callers
    may pass either the hub-lock slug or the user-facing name. Values are
    ``(folder_name, install_path, source_dir)``."""
    ss = _ss()
    optional_dir = ss._get_optional_dir()
    index: Dict[str, Tuple[str, str, Path]] = {}
    if not optional_dir.exists():
        return index
    for skill_md, src, install_path in _iter_optional_skills(optional_dir, root_relative=True):
        value = (src.name, install_path, src)
        index[src.name] = value
        index[ss._read_skill_name(skill_md, src.name)] = value
    return index


def _move_to_restore_backup(path: Path, backup_root: Path) -> str:
    """Move an existing skill directory into a restore backup, preserving rel path."""
    rel = path.relative_to(_ss()._skills_dir())
    target = backup_root / rel
    suffix = 0
    while target.exists():
        suffix += 1
        target = (backup_root / rel).with_name(f"{rel.name}-{suffix}")
    _ss()._move_dir(path, target)
    return rel.as_posix()


def _find_active_copies(folder_name: str, src_frontmatter: str, dest: Path) -> List[Path]:
    """Active copies of an official skill (by frontmatter name or folder slug),
    even when the curator moved it into another category; excludes ``dest``."""
    ss = _ss()
    names = {folder_name, src_frontmatter}
    return [
        md.parent
        for md in ss._iter_active_skill_mds(sort=True)
        if md.parent != dest and (md.parent.name == folder_name or ss._read_skill_name(md, md.parent.name) in names)
    ]


def restore_official_optional_skill(name: str, *, restore: bool = False) -> dict:
    """Restore one or all official optional skills from repo source. ``restore=False``
    only performs exact-match provenance backfill; ``restore=True`` repairs mutated /
    reorganized skills by backing up matching active copies and copying the official
    source into its canonical path."""
    ss = _ss()

    def _fail(message: str) -> dict:
        return {"ok": False, "message": message, "restored": [], "backfilled": [], "backed_up": []}

    index = _optional_skill_index()
    if not index:
        return _fail("No official optional skills directory found.")
    if name in {"all", "*"}:
        targets = sorted(set(index.values()), key=lambda item: item[1])
    elif name in index:
        targets = [index[name]]
    else:
        return _fail(f"Official optional skill not found: {name}")

    restored: List[str] = []
    backed_up: List[str] = []
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    backup_root = ss._skills_dir() / ".restore-backups" / f"official-optional-{timestamp}"

    for folder_name, install_path, src in targets if restore else []:
        dest = ss._skills_dir() / Path(*install_path.split("/"))
        canonical_ok = dest.exists() and ss._dir_hash(dest) == ss._dir_hash(src)
        src_frontmatter = ss._read_skill_name(src / "SKILL.md", folder_name)
        for match in _find_active_copies(folder_name, src_frontmatter, dest):
            if match.exists():
                backed_up.append(_move_to_restore_backup(match, backup_root))
        if dest.exists() and not canonical_ok:
            backed_up.append(_move_to_restore_backup(dest, backup_root))
        if not dest.exists():
            ss._copy_dir(src, dest)
            restored.append(folder_name)

    return {
        "ok": True, "message": "Official optional skill repair complete.", "restored": restored,
        "backfilled": _backfill_optional_provenance(quiet=True), "backed_up": backed_up,
        "backup_dir": str(backup_root) if backed_up else "",
    }


def _index_installed_skill_dirs_by_name() -> Dict[str, List[Path]]:
    """Index installed skills by directory name with one active-tree scan,
    skipping anything that resolves outside the skills tree (symlinks/external)."""
    ss = _ss()
    index: Dict[str, List[Path]] = {}
    root = ss._skills_dir().resolve()
    for skill_md in ss._iter_active_skill_mds():
        try:
            skill_md.parent.resolve().relative_to(root)
        except (OSError, ValueError):
            continue
        index.setdefault(skill_md.parent.name, []).append(skill_md.parent)
    return index


def _relocated_dest(src_name: str, index: Dict[str, List[Path]]) -> Optional[Tuple[Path, str]]:
    """The active tree may hold a skill under a DIFFERENT category path than the
    repo (upstream reorganizes; the installed copy keeps its old location). Fall
    back to a UNIQUE same-directory-name match — an ambiguous name gives no basis
    to pick one. Returns ``(dest, install_path)`` or None."""
    candidates = index.get(src_name, [])
    if len(candidates) != 1:
        return None
    dest = candidates[0]
    try:
        return dest, _safe_rel_install_path(dest, _ss()._skills_dir())
    except ValueError as e:
        logger.debug("Skipping relocated optional skill %s: %s", dest, e)
        return None


def _backfill_optional_provenance(quiet: bool = False) -> List[str]:
    """Mark already-present official optional skills as hub-installed: skills that used
    to be bundled (or were hand-copied) and now live under optional-skills/ get official
    provenance when byte-identical to the source. Modified/local skills are left alone."""
    ss = _ss()
    optional_dir = ss._get_optional_dir()
    if not optional_dir.exists():
        return []

    data = _load_hub_lock()
    if data is None:
        data = {"version": 1, "installed": {}}
    installed = data.setdefault("installed", {})
    existing_paths = {entry.get("install_path") for entry in _hub_lock_entries(data)}

    backfilled: List[str] = []
    installed_dir_index: Optional[Dict[str, List[Path]]] = None
    for _skill_md, src, install_path in _iter_optional_skills(optional_dir, root_relative=False):
        lock_name = src.name
        if lock_name in installed or install_path in existing_paths:
            continue
        dest = ss._skills_dir() / Path(*install_path.split("/"))
        if not dest.is_dir():
            if installed_dir_index is None:
                installed_dir_index = _index_installed_skill_dirs_by_name()
            found = _relocated_dest(src.name, installed_dir_index)
            if found is None:
                continue
            dest, install_path = found  # still requires a byte-identical hash below
        if install_path in existing_paths or ss._dir_hash(dest) != ss._dir_hash(src):
            continue

        timestamp = datetime.now(timezone.utc).isoformat()
        installed[lock_name] = {
            "source": "official",
            "identifier": f"official/{install_path}",
            "trust_level": "builtin",
            "scan_verdict": "backfilled",
            "content_hash": _content_hash(dest),
            "install_path": install_path,
            "files": _skill_file_list(dest),
            "metadata": {"backfilled_from": "optional-skills"},
            "installed_at": timestamp,
            "updated_at": timestamp,
        }
        existing_paths.add(install_path)
        backfilled.append(lock_name)
        if not quiet:
            print(f"  = {lock_name} (official optional provenance backfilled)")

    if backfilled:
        _write_hub_lock(_hub_lock_path(), data)
    return backfilled
