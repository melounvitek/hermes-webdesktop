"""Per-mutation skill audit ledger + single-edit rollback.

Every skill mutation — any actor — appends one JSONL entry to
``~/.hermes/skills/.curator_ledger.jsonl`` with before/after file manifests
whose contents are stored content-addressed (sha256-deduped) under
``~/.hermes/.curator_backups/blobs/``.

Design decisions (Teknium-approved):
  - JSONL, not the state DB: durable, human-greppable, survives DB resets.
  - Covers ALL actors (``curator`` / ``agent`` / ``user``). The curator
    invariant (never hard-delete autonomously) applies only to autonomous
    actors; user deletes stay hard-delete but are ledgered and recoverable via
    ``hermes curator rollback <entry-id>``.
  - Per-file content-addressed blobs, not tarballs: a mutation usually touches
    one file, and identical content dedupes to one blob.

The ledger is TELEMETRY, NOT A GATE: a ledger failure must never block the
mutation it describes — every public write path swallows and logs. The one
exception is ``rollback_entry``, which FAILS CLOSED when its own pre-rollback
safety capture fails (consistent with agent/curator_backup.py).
"""

from __future__ import annotations

import contextvars
import hashlib
import json
import logging
import os
import re
import tarfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

ACTOR_CURATOR = "curator"
ACTOR_AGENT = "agent"
ACTOR_USER = "user"

# Snapshot-id shape used by agent.curator_backup (duplicated so the ledger can
# read the newest skills.tar.gz without importing the backup stack).
_BACKUP_ID_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}Z(-\d{2})?$")
# ".archive/<name>-YYYYMMDDHHMMSS" collision suffix added by archive_skill.
_ARCHIVE_TS_SUFFIX_RE = re.compile(r"^(.+)-\d{14}$")
# Actions whose rollback must restore a COMPLETE package: consolidation may
# have re-homed support files out of the tree first, so a disk-only capture
# would make rollback restore a hollow skill.
_PACKAGE_RESTORE_ACTIONS = frozenset({"delete", "archive", "purge"})
_VALID_ACTORS = {ACTOR_CURATOR, ACTOR_AGENT, ACTOR_USER}
_NON_PACKAGE_TOPS = {".curator_backups", ".hub", ".archive"}

# Explicit actor override: the CLI sets "user", the curator walk sets "curator".
_actor_override: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "skill_ledger_actor", default=None
)


def set_ledger_actor(actor: Optional[str]) -> contextvars.Token:
    """Bind an explicit actor for this context; reset_ledger_actor(token) in a finally."""
    return _actor_override.set(actor)


def reset_ledger_actor(token: contextvars.Token) -> None:
    _actor_override.reset(token)


def derive_actor() -> str:
    """Explicit override, else background-review provenance -> curator, else agent."""
    override = _actor_override.get()
    if override in _VALID_ACTORS:
        return override
    try:
        from tools.skill_provenance import is_background_review

        if is_background_review():
            return ACTOR_CURATOR
    except Exception:
        pass
    return ACTOR_AGENT


# ---------------------------------------------------------------------------
# Paths + config gate
# ---------------------------------------------------------------------------

def ledger_path() -> Path:
    return get_hermes_home() / "skills" / ".curator_ledger.jsonl"


def blobs_dir() -> Path:
    return get_hermes_home() / ".curator_backups" / "blobs"


def _skills_dir() -> Path:
    return get_hermes_home() / "skills"


def ledger_enabled() -> bool:
    """Config gate ``skills.ledger`` (default True). Lazy import so this
    module stays importable without the CLI config layer."""
    try:
        from hermes_cli.config import cfg_get, load_config

        return bool(cfg_get(load_config(), "skills", "ledger", default=True))
    except Exception as e:  # pragma: no cover — best-effort config read
        logger.debug("skill_ledger: config read failed (%s); defaulting on", e)
        return True


def _norm(path: Path | str) -> Path:
    return Path(os.path.normpath(str(path)))


def _rel_posix(path: Path | str, root: Path) -> Optional[str]:
    """POSIX path of ``path`` relative to ``root`` (both normalized), or None when outside."""
    try:
        return _norm(path).relative_to(_norm(root)).as_posix()
    except ValueError:
        return None


def _is_within(root: Path, path: Path) -> bool:
    """True when *path* (normalized, no symlink resolution) sits under *root*."""
    try:
        root_r, path_r = _norm(root), _norm(path)
        return path_r == root_r or root_r in path_r.parents
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Blob store (content-addressed, deduped)
# ---------------------------------------------------------------------------

def _store_blob(data: bytes) -> str:
    """Write *data* keyed by sha256 (existing blob left alone). Returns the hash."""
    digest = hashlib.sha256(data).hexdigest()
    dest = blobs_dir() / digest
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(f".tmp-{uuid.uuid4().hex[:8]}-{digest}")
        tmp.write_bytes(data)
        os.replace(tmp, dest)
    return digest


def read_blob(sha256: str) -> Optional[bytes]:
    """Return blob content or None when missing/invalid."""
    if not sha256 or not all(c in "0123456789abcdef" for c in sha256):
        return None
    p = blobs_dir() / sha256
    try:
        return p.read_bytes() if p.exists() else None
    except OSError:
        return None


def snapshot_paths(
    root: Optional[Path],
    *,
    complete_package: bool = False,
) -> List[Dict[str, str]]:
    """Capture {path, sha256} for every file under *root*, storing each as a blob.

    Empty when root is None/missing. Raises on I/O failure — callers decide
    whether that is fatal (rollback safety capture) or swallowed (telemetry).
    ``complete_package=True`` unions in files from the newest curator
    ``skills.tar.gz`` for this skill (disk hashes win)."""
    if root is None:
        return []
    root = Path(root)
    if root.is_file():
        files = [root]
    elif root.is_dir():
        files = sorted(p for p in root.rglob("*") if p.is_file())
    elif complete_package:
        files = []
    else:
        return []
    out = [{"path": str(f), "sha256": _store_blob(f.read_bytes())} for f in files]
    if complete_package:
        out = fill_snapshot_from_curator_backup(root, out)
    return out


# ---------------------------------------------------------------------------
# Package-completeness fill from the newest curator backup
# ---------------------------------------------------------------------------

def _package_rel(root: Path) -> Optional[str]:
    """Relative POSIX path of a skill dir under ``skills/``; None when outside
    it or under backup/hub/archive metadata roots (never a package)."""
    posix = (_rel_posix(root, _skills_dir()) or "").strip("/")
    if not posix or posix.split("/", 1)[0] in _NON_PACKAGE_TOPS:
        return None
    return posix


def _strip_archive_timestamp(name: str) -> str:
    match = _ARCHIVE_TS_SUFFIX_RE.match(name)
    return match.group(1) if match else name


def _skill_md_parent(items: Optional[List[Dict[str, str]]]) -> Optional[Path]:
    for item in items or []:
        path = Path(str(item.get("path", "")))
        if path.name == "SKILL.md":
            return path.parent
    return None


def package_prefixes(
    root: Optional[Path] = None,
    skill: Optional[str] = None,
    before: Optional[List[Dict[str, str]]] = None,
) -> List[str]:
    """Tar member prefixes that belong to this skill's package: its live
    location under ``skills/``, the package parent recorded in the before-state
    SKILL.md path (for rollback fills where *root* is gone), the bare skill
    name, and the name minus an archive collision suffix."""
    candidates = [_package_rel(Path(root)) if root is not None else None]
    for item in before or []:
        path = Path(str(item.get("path", "")))
        if path.name == "SKILL.md":
            candidates.append(_package_rel(path.parent))
    candidates += [skill, _strip_archive_timestamp(skill) if skill else None]
    found: List[str] = []
    for prefix in candidates:
        prefix = (prefix or "").strip("/")
        if prefix and prefix not in found:
            found.append(prefix)
    return found


def _latest_skills_tarball() -> Optional[Path]:
    """Newest ``skills.tar.gz`` under ``skills/.curator_backups/``."""
    backups = _skills_dir() / ".curator_backups"
    if not backups.is_dir():
        return None
    try:
        children = list(backups.iterdir())
    except OSError:
        return None
    candidates = [
        child / "skills.tar.gz"
        for child in children
        if child.is_dir() and _BACKUP_ID_RE.match(child.name) and (child / "skills.tar.gz").is_file()
    ]
    if not candidates:
        return None
    # Parent dirs sort lexicographically == chronologically for the id shape.
    return max(candidates, key=lambda p: p.parent.name)


def _read_package_files_from_latest_backup(prefixes: List[str]) -> Dict[str, bytes]:
    """``{posix-relpath: bytes}`` for files under *prefixes* in the newest
    snapshot. Malicious member names (absolute, ``..`` traversal) are rejected."""
    if not prefixes:
        return {}
    archive = _latest_skills_tarball()
    if archive is None:
        return {}
    prefixed = tuple(p if p.endswith("/") else p + "/" for p in prefixes)
    exact = set(prefixes)
    out: Dict[str, bytes] = {}
    try:
        with tarfile.open(archive, "r:gz") as tf:
            for member in tf.getmembers():
                if not member.isfile():
                    continue
                name = member.name.replace("\\", "/").lstrip("./")
                if not name or name.startswith("/") or ".." in Path(name).parts:
                    continue
                if name not in exact and not name.startswith(prefixed):
                    continue
                extracted = tf.extractfile(member)
                if extracted is not None:
                    out[name] = extracted.read()
    except (OSError, tarfile.TarError) as exc:
        logger.debug("skill_ledger: could not read curator backup package: %s", exc)
        return {}
    return out


def fill_snapshot_from_curator_backup(
    root: Optional[Path],
    existing: Optional[List[Dict[str, str]]] = None,
    *,
    skill: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Union missing skill-package files from the newest curator snapshot.

    Completeness fill, not a gate: failures are swallowed and *existing* is
    returned unchanged; the backup only fills paths ABSENT from it.

    Filled files are addressed where the rollback must restore them: under
    *root* when known (for purge that is ``.archive/<name>/``, NOT the live
    tree), else under the live skills dir. Backup members carry a leading
    package-dir segment, stripped when *root* already names the package.
    Every fill target must stay under ``skills/`` and HERMES_HOME.
    """
    out = list(existing or [])
    prefixes = package_prefixes(root, skill, out)
    if not prefixes:
        return out
    try:
        extra = _read_package_files_from_latest_backup(prefixes)
    except Exception as exc:
        logger.debug("skill_ledger: backup package fill failed: %s", exc)
        return out
    if not extra:
        return out
    skills = _skills_dir()
    dest_root = Path(root) if root is not None else None
    pkg_names = {dest_root.name, _strip_archive_timestamp(dest_root.name)} if dest_root else set()
    have = {
        rel for rel in (_rel_posix(str(item.get("path", "")), skills) for item in out) if rel is not None
    }
    for rel, data in extra.items():
        parts = rel.split("/")
        if dest_root is not None and parts and parts[0] in pkg_names:
            parts = parts[1:]
        if not parts:
            continue
        dest = (dest_root if dest_root is not None else skills).joinpath(*parts)
        if not _is_within(skills, dest) or not _is_within(get_hermes_home(), dest):
            continue
        rel_key = _rel_posix(dest, skills)
        if rel_key is None or rel_key in have:
            continue
        try:
            out.append({"path": str(dest), "sha256": _store_blob(data)})
        except Exception as exc:
            logger.debug("skill_ledger: backup blob store failed for %s: %s", rel, exc)
            continue
        have.add(rel_key)
    return out


# ---------------------------------------------------------------------------
# Append + read
# ---------------------------------------------------------------------------

def append_entry(
    action: str,
    skill: str,
    before: Optional[List[Dict[str, str]]] = None,
    after: Optional[List[Dict[str, str]]] = None,
    actor: Optional[str] = None,
    evidence: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Append one ledger entry. Returns the entry id, or None when the
    ledger is disabled or the write failed (never raises)."""
    if not ledger_enabled():
        return None
    try:
        entry = {
            "id": uuid.uuid4().hex[:12],
            "ts": datetime.now(timezone.utc).isoformat(),
            "actor": actor if actor in _VALID_ACTORS else derive_actor(),
            "action": action,
            "skill": skill,
            "evidence": evidence or {},
            "before": before or [],
            "after": after or [],
        }
        path = ledger_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return entry["id"]
    except Exception as e:
        logger.warning("skill_ledger: failed to append entry (%s) — mutation unaffected", e)
        return None


def record_mutation(
    action: str,
    skill: str,
    before_root: Optional[Path] = None,
    before: Optional[List[Dict[str, str]]] = None,
    after_root: Optional[Path] = None,
    actor: Optional[str] = None,
    evidence: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """One-stop hook for mutation call sites: capture after-state from
    *after_root* (pre-captured *before* list, or capture from *before_root*)
    and append. NEVER raises and never blocks the mutation.

    delete/archive/purge always capture a COMPLETE package (support files
    filled from the newest curator backup) so rollback never restores a shell."""
    if not ledger_enabled():
        return None
    try:
        _complete = action in _PACKAGE_RESTORE_ACTIONS
        if before is None:
            before = snapshot_paths(before_root, complete_package=_complete)
        elif _complete:
            before = fill_snapshot_from_curator_backup(before_root, before, skill=skill)
        after = snapshot_paths(after_root)
        return append_entry(
            action, skill, before=before, after=after, actor=actor, evidence=evidence
        )
    except Exception as e:
        logger.warning("skill_ledger: record_mutation failed (%s) — mutation unaffected", e)
        return None


def capture_before(
    root: Optional[Path],
    *,
    complete_package: bool = False,
    skill: Optional[str] = None,
) -> Optional[List[Dict[str, str]]]:
    """Best-effort pre-mutation capture; None on failure or when disabled
    (callers pass the result straight to record_mutation). Use
    ``complete_package=True`` for delete/archive/purge captures."""
    if not ledger_enabled():
        return None
    try:
        captured = snapshot_paths(root)
        if complete_package:
            captured = fill_snapshot_from_curator_backup(root, captured, skill=skill)
        return captured
    except Exception as e:
        logger.warning("skill_ledger: before-capture failed (%s) — mutation unaffected", e)
        return None


def list_entries(
    skill: Optional[str] = None, limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Read the ledger, newest first. Malformed lines are skipped."""
    path = ledger_path()
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict):
                    rows.append(row)
    except OSError:
        return []
    if skill:
        rows = [r for r in rows if r.get("skill") == skill]
    rows.reverse()
    if limit is not None and limit >= 0:
        rows = rows[:limit]
    return rows


def get_entry(entry_id: str) -> Optional[Dict[str, Any]]:
    if not entry_id:
        return None
    return next((row for row in list_entries() if row.get("id") == entry_id), None)


# ---------------------------------------------------------------------------
# Single-edit rollback
# ---------------------------------------------------------------------------

def _validate_entry_paths(entry: Dict[str, Any]) -> Optional[str]:
    """All paths in an entry must live under HERMES_HOME — a hand-edited
    ledger must not become a write-anywhere primitive."""
    home = get_hermes_home()
    for section in ("before", "after"):
        for item in entry.get(section) or []:
            p = Path(str(item.get("path", "")))
            if not _is_within(home, p):
                return f"entry references a path outside {home}: {p}"
    return None


def rollback_entry(entry_id: str) -> Tuple[bool, str]:
    """Restore the before-state of the single mutation *entry_id*.

    Fail-closed (mirrors agent/curator_backup.rollback):
      1. Every needed before-blob must exist — verified BEFORE any change.
      2. A pre-rollback safety entry capturing the CURRENT state of every
         touched path is appended first; if that fails, nothing is changed.
    """
    entry = get_entry(entry_id)
    if entry is None:
        return False, f"no ledger entry with id '{entry_id}'"

    path_err = _validate_entry_paths(entry)
    if path_err:
        return False, f"refusing rollback: {path_err}"

    before = list(entry.get("before") or [])
    after = list(entry.get("after") or [])

    # Historical hollow delete/archive/purge entries (``files: 1`` = SKILL.md):
    # fill the before-state from the newest curator backup so the rollback
    # restores the complete package. Entry hashes win; only missing paths are
    # added, and the filled set is re-validated against HERMES_HOME.
    if entry.get("action") in _PACKAGE_RESTORE_ACTIONS:
        before = fill_snapshot_from_curator_backup(
            _skill_md_parent(before), before, skill=str(entry.get("skill") or "") or None
        )
        path_err = _validate_entry_paths({**entry, "before": before, "after": after})
        if path_err:
            return False, f"refusing rollback: {path_err}"

    # Pre-check every blob we need so we never fail mid-restore.
    for item in before:
        if read_blob(str(item.get("sha256", ""))) is None:
            return False, (
                f"missing blob {item.get('sha256')} for {item.get('path')}; "
                "rollback aborted, nothing was changed"
            )

    # Touched paths = union of before/after. Capture their CURRENT state as
    # the safety entry so the rollback itself is undoable. FAIL CLOSED.
    touched = {str(i["path"]) for i in before + after if i.get("path")}
    try:
        safety_before: List[Dict[str, str]] = []
        for p in sorted(touched):
            fp = Path(p)
            if fp.is_file():
                safety_before.append({"path": p, "sha256": _store_blob(fp.read_bytes())})
        safety_id = append_entry(
            "pre-rollback",
            entry.get("skill", "?"),
            before=safety_before,
            after=safety_before,
            evidence={"rollback_target": entry_id},
        )
    except Exception as e:
        return False, (
            f"pre-rollback safety capture failed ({e}); rollback aborted and "
            "current skills were not changed"
        )
    if safety_id is None:
        return False, (
            "pre-rollback safety capture failed (ledger disabled or "
            "unwritable); rollback aborted and current skills were not changed"
        )

    # Restore: write every before-file, remove files the mutation created.
    before_paths = {str(i["path"]) for i in before}
    restored = 0
    removed = 0
    for item in before:
        fp = Path(str(item["path"]))
        data = read_blob(str(item["sha256"]))
        assert data is not None  # pre-checked above
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_bytes(data)
        restored += 1
    for item in after:
        p = str(item.get("path", ""))
        if p and p not in before_paths:
            fp = Path(p)
            try:
                if fp.is_file():
                    fp.unlink()
                    removed += 1
            except OSError as e:
                logger.warning("skill_ledger: could not remove %s during rollback: %s", p, e)

    append_entry(
        "rollback",
        entry.get("skill", "?"),
        before=safety_before,
        after=before,
        evidence={"rollback_target": entry_id, "restored": restored, "removed": removed},
    )
    return True, (
        f"rolled back entry {entry_id} ({entry.get('action')} on "
        f"'{entry.get('skill')}'): {restored} file(s) restored, {removed} removed. "
        f"Safety entry {safety_id} captured the pre-rollback state."
    )
