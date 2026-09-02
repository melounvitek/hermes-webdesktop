"""Org-shared skills: org pull + propose (``~/.hermes/skills/_org/<org_id>/``).

Org skills live under a DISTINCT local namespace (read-only to the runtime; a
local edit is a personal fork until proposed). The canonical set is
``refs/org/<org_id>/HEAD`` -- the SAME object model as personal sync.
PERSONAL-ORG GATE: NAS stamps ``org_role`` ONLY for multi-member orgs; no
claim => pull/propose raise SyncInertError and personal sync is untouched.
``propose_skill`` must stay non-interactive (automation will drive it).
Module state (``_skills_dir``, ``_org_dir``, base URL, device id) stays in
``tools.skills_sync_client`` and is read lazily so tests can monkeypatch it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Dict, List, Optional

from tools.skills_sync_client_wire import (
    DEFAULT_MAX_OBJECT_BYTES, ObjectSet, SyncClient, SyncConflict, SyncError, _check_version,
    assemble_root_from_skill_trees, build_commit, build_tree, materialize_tree, read_ref_hash,
    root_tree_of_commit, skill_trees_of_root,
)

logger = logging.getLogger("tools.skills_sync_client")

ORG_DIR_NAME = "_org"

# Propose re-splices onto a moved org HEAD at most this many times. Small:
# contention means other members are actively proposing; unbounded would spin.
_ORG_CAS_MAX_ATTEMPTS = 5


def _ssc():
    from tools import skills_sync_client

    return skills_sync_client


def org_head_ref(org_id: str) -> str:
    return f"refs/org/{org_id}/HEAD"


def resolve_org_identity() -> Dict[str, Any]:
    """``resolve_identity()`` extended with ``org_id`` + ``org_role``.

    Raises SyncInertError when the token carries no ``org_role`` claim (personal
    org / issuer predates org support): org sync is unavailable, NOT an error.
    """
    from tools.skills_sync_client import SyncInertError, resolve_identity

    identity = resolve_identity()
    claims = identity.get("claims") or {}
    org_id = claims.get("org_id")
    org_role = claims.get("org_role")
    if not org_id:
        raise SyncInertError("no organisation associated with this account")
    if not isinstance(org_role, str) or not org_role:
        raise SyncInertError("this account isn't a member of a shared organisation")
    identity["org_id"] = str(org_id)
    identity["org_role"] = org_role
    return identity


def _org_client(identity: Optional[Dict[str, Any]], client: Optional[SyncClient]):
    """Resolve (identity, client, caps) for an org operation; raises SyncInertError
    when the base URL is missing or the server lacks the ``org`` feature."""
    ssc = _ssc()
    identity = identity or resolve_org_identity()
    if client is None:
        base_url = ssc.resolve_sync_base_url()
        if not base_url:
            raise ssc.SyncInertError("no sync base URL configured")
        client = SyncClient(base_url, identity["api_key"])
    caps = client.capabilities()
    _check_version(caps)
    if "org" not in (caps.get("features") or []):
        raise ssc.SyncInertError("this server does not support org-shared skills")
    return identity, client, caps


def _read_org_head(client: SyncClient, org_id: str) -> Optional[str]:
    """Current org HEAD, or None. MUST read through the ORG endpoint."""
    return read_ref_hash(client, org_head_ref(org_id), org_scope=True)


# Local mirror sidecars

def _mirror_root(org_id: str) -> Path:
    return _ssc()._org_dir() / org_id


def _write_sidecar(what: str, path_fn: Callable[[], Path], text: str) -> None:
    """Best-effort sidecar write (path resolution included); never raises."""
    try:
        path = path_fn()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    except Exception as e:
        logger.debug("skills_sync_client: %s write failed: %s", what, e)


def _skill_dir_fingerprint(path: Path) -> str:
    """Stable content hash of a materialized skill dir (sorted relative path +
    bytes, independent of filesystem order and mtimes). "" on read failure."""
    h = hashlib.sha256()
    try:
        for f in sorted(p for p in path.rglob("*") if p.is_file()):
            h.update(str(f.relative_to(path)).replace("\\", "/").encode("utf-8"))
            h.update(b"\0")
            h.update(f.read_bytes())
            h.update(b"\0")
    except OSError as e:
        logger.debug("skills_sync_client: fingerprint failed for %s: %s", path, e)
        return ""
    return h.hexdigest()


def _sidecar_path(org_id: Optional[str], const: str) -> Path:
    """``<mirror>/<agent.skill_utils.<const>>`` (org-level when org_id is None)."""
    import agent.skill_utils as sku

    return (_mirror_root(org_id) if org_id else _ssc()._org_dir()) / getattr(sku, const)


def _org_baseline_path(org_id: str) -> Path:
    """Sidecar recording the upstream fingerprint of each mirrored skill."""
    return _sidecar_path(org_id, "ORG_BASELINE_FILE")


def _read_org_baseline(org_id: str) -> Dict[str, Any]:
    try:
        return json.loads(_org_baseline_path(org_id).read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_org_baseline(org_id: str, baseline: Dict[str, Any]) -> None:
    _write_sidecar(
        "baseline", lambda: _org_baseline_path(org_id), json.dumps(baseline, indent=2, sort_keys=True)
    )


def _write_org_provenance(org_id: str, data: Dict[str, Any]) -> None:
    _write_sidecar(
        "org provenance", lambda: _sidecar_path(org_id, "ORG_PROVENANCE_FILE"), json.dumps(data, indent=2)
    )


def _write_active_org_marker(org_id: str) -> None:
    """Record which org's mirror may resolve (agent/skill_utils.read_active_org_id)."""
    _write_sidecar("active-org marker", lambda: _sidecar_path(None, "ORG_ACTIVE_MARKER"), org_id)


def _clear_active_org_marker() -> None:
    """Remove the active-org marker so org skills stop resolving."""
    try:
        marker = _sidecar_path(None, "ORG_ACTIVE_MARKER")
        if marker.exists():
            marker.unlink()
            logger.info(
                "skills_sync_client: cleared active-org marker "
                "(token has no org workflow); org skills no longer resolve"
            )
    except Exception as e:
        logger.debug("skills_sync_client: marker clear failed: %s", e)


def org_skill_is_locally_modified(skill_rel_path: str, org_id: str) -> bool:
    """True when the local copy of an org skill differs from what upstream sent.
    No recorded baseline (pre-existing mirror) => unmodified; the next pull
    records one."""
    dest = _mirror_root(org_id) / PurePosixPath(skill_rel_path)
    if not dest.is_dir():
        return False
    entry = _read_org_baseline(org_id).get(skill_rel_path) or {}
    recorded = entry.get("fingerprint") if isinstance(entry, dict) else entry
    return bool(recorded) and _skill_dir_fingerprint(dest) != recorded


def list_locally_modified_org_skills(org_id: Optional[str] = None) -> List[str]:
    """Org skills with local edits that upstream has not seen."""
    try:
        from agent.skill_utils import read_active_org_id

        org_id = org_id or read_active_org_id(_ssc()._skills_dir())
        if not org_id:
            return []
        return sorted(rel for rel in _read_org_baseline(org_id) if org_skill_is_locally_modified(rel, org_id))
    except Exception as e:
        logger.debug("skills_sync_client: modified-scan failed: %s", e)
        return []


def list_org_skill_names() -> List[str]:
    """Skill names present in the local org mirror (empty when none pulled)."""
    names: List[str] = []
    try:
        from agent.skill_utils import read_active_org_id

        org_id = read_active_org_id(_ssc()._skills_dir())
        root = _mirror_root(org_id) if org_id else None
        if root and root.is_dir():
            for skill_md in root.rglob("SKILL.md"):
                rel = skill_md.parent.relative_to(root)
                if rel.parts:
                    names.append(str(rel).replace("\\", "/"))
    except Exception as e:
        logger.debug("skills_sync_client: org skill listing failed: %s", e)
    return sorted(names)


# Pull / propose

def pull_org_skills(
    client: Optional[SyncClient] = None, *, identity: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Pull the org canonical set into the local mirror (fast-forward only; no
    client merge on the org path). A mirrored skill with LOCAL edits is never
    clobbered: it is skipped, and reported in ``conflicted`` when upstream also
    moved; the member's change of record is ``propose_skill``.
    Returns ``{ok, org_id, head, updated, conflicted}``."""
    identity = identity or resolve_org_identity()
    if "org_id" not in identity:
        raise _ssc().SyncInertError("no organisation context available")
    identity, client, _caps = _org_client(identity, client)
    org_id = identity["org_id"]

    head = _read_org_head(client, org_id)
    # Token-gated marker: written HERE because this runs only after the token's
    # org_id + org_role were verified. A stale mirror from a previous org stops
    # resolving the moment a pull runs under a different org.
    _write_active_org_marker(org_id)
    if not head:
        return {"ok": True, "org_id": org_id, "head": None, "updated": []}

    head_commit = client.get_commit_json(head, org_scope=True)
    skill_trees = skill_trees_of_root(client, head_commit["tree"], org_scope=True)

    dest_root = _mirror_root(org_id)
    updated: List[str] = []
    conflicted: List[str] = []
    baseline = _read_org_baseline(org_id)
    for rel_path, tree_hash in sorted(skill_trees.items()):
        dest = dest_root / PurePosixPath(rel_path)
        try:
            if dest.exists():
                if org_skill_is_locally_modified(rel_path, org_id):
                    if (baseline.get(rel_path) or {}).get("tree") != tree_hash:
                        conflicted.append(rel_path)
                    continue
                shutil.rmtree(dest)
            dest.mkdir(parents=True, exist_ok=True)
            materialize_tree(client, tree_hash, dest, org_scope=True)
            baseline[rel_path] = {"fingerprint": _skill_dir_fingerprint(dest), "tree": tree_hash}
            updated.append(rel_path)
        except Exception as e:
            logger.warning("skills_sync_client: org skill materialize failed for %s: %s", rel_path, e)
    # Provenance for the skill_view header: the HEAD author is token-verified
    # by the plane at push time, so it is trustworthy to display.
    author = head_commit.get("author") or {}
    _write_org_provenance(org_id, {
        "org_id": org_id, "head": head, "author_user_id": author.get("owner", ""),
        "author_device": author.get("device", ""), "ts": head_commit.get("ts", ""), "skills": updated,
    })
    _write_org_baseline(org_id, baseline)
    if conflicted:
        logger.warning(
            "skills_sync_client: %d org skill(s) have local edits AND upstream "
            "changes; left untouched: %s", len(conflicted), ", ".join(conflicted),
        )
    return {"ok": True, "org_id": org_id, "head": head, "updated": updated, "conflicted": conflicted}


def propose_skill(
    skill_name: str,
    client: Optional[SyncClient] = None,
    *,
    identity: Optional[Dict[str, Any]] = None,
    message: Optional[str] = None,
) -> Dict[str, Any]:
    """Propose a local (personal) skill's content to the org canonical set.

    Snapshots the skill dir as an org-scoped commit splicing that ONE skill
    subtree into the current org HEAD (proposals are per-skill deltas, never a
    wholesale replace), uploads with ``?scope=org``, then CAS-es the org HEAD:
    ADMIN/OWNER -> server merges -> ``{ok, merged: True}``; MEMBER -> 202
    proposal -> ``{ok, proposal_pending: True, proposal_id, ref}``, never
    presented as live.

    If HEAD moves between read and CAS, the skill is re-spliced onto the NEW
    head (not replayed from the old root, which would drop the other member's
    skill) up to ``_ORG_CAS_MAX_ATTEMPTS`` times.
    """
    ssc = _ssc()
    identity, client, caps = _org_client(identity, client)
    org_id = identity["org_id"]
    max_bytes = int(caps.get("max_object_bytes") or DEFAULT_MAX_OBJECT_BYTES)

    rel = ssc._skill_rel_path(skill_name)
    if rel is None:
        raise SyncError(f"skill '{skill_name}' not found under the skills dir")
    skill_dir = ssc._skills_dir() / rel
    if not (skill_dir / "SKILL.md").exists():
        raise SyncError(f"skill '{skill_name}' has no SKILL.md")

    objects = ObjectSet()
    skill_tree = build_tree(skill_dir, objects, max_object_bytes=max_bytes)

    for attempt in range(1, _ORG_CAS_MAX_ATTEMPTS + 1):
        base_head = _read_org_head(client, org_id)
        skill_map = (
            skill_trees_of_root(client, root_tree_of_commit(client, base_head, org_scope=True), org_scope=True)
            if base_head
            else {}
        )
        skill_map[str(rel)] = skill_tree
        root_hash = assemble_root_from_skill_trees(skill_map, objects)
        commit_hash = build_commit(
            root_hash, [base_head] if base_head else [], owner=identity["owner"],
            device=ssc.stable_device_id(), message=message or f"propose {skill_name}", objects=objects,
        )
        client.put_objects(objects.objects, org_scope=True)
        try:
            result = client.cas_ref(org_head_ref(org_id), base_head, commit_hash)
            break
        except SyncConflict as conflict:
            if attempt >= _ORG_CAS_MAX_ATTEMPTS:
                raise SyncError(
                    "the organisation's skills changed while this was being "
                    f"proposed, and {attempt} attempts to catch up all lost "
                    "the race — run the command again",
                    status=409,
                ) from conflict
            logger.debug(
                "propose_skill: org HEAD moved (actual=%r), re-splicing (attempt %d)", conflict.actual, attempt
            )

    if result.get("proposal_pending"):
        return {
            "ok": True, "proposal_pending": True, "proposal_id": result.get("proposal_id"),
            "ref": result.get("ref"), "commit": commit_hash, "org_id": org_id,
        }
    return {
        "ok": True, "merged": True, "head": result.get("hash", commit_hash),
        "commit": commit_hash, "org_id": org_id,
    }


def maybe_pull_org_skills() -> Optional[Dict[str, Any]]:
    """Best-effort org pull if all gates hold (logged in, org_role claim,
    feature enabled, base URL). Never raises; None when inert.

    Marker hygiene: when the token VERIFIABLY lacks the org claim (personal org
    / left the org) the active-org marker is cleared so mirrored org skills stop
    resolving. When identity cannot be resolved at all (offline, logged out) the
    marker is left alone -- offline grace keeps pulled org skills working.
    """
    ssc = _ssc()
    try:
        identity = resolve_org_identity()
    except ssc.SyncInertError:
        try:
            if not (ssc.resolve_identity().get("claims") or {}).get("org_role"):
                _clear_active_org_marker()
        except Exception:
            pass
        return None
    except Exception as e:
        logger.debug("skills_sync_client: maybe_pull_org_skills inert/failed: %s", e)
        return None
    try:
        if not ssc.sync_feature_enabled() or not ssc.resolve_sync_base_url():
            return None
        return pull_org_skills(identity=identity)
    except Exception as e:
        logger.debug("skills_sync_client: maybe_pull_org_skills inert/failed: %s", e)
        return None
