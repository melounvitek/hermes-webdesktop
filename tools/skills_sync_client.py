#!/usr/bin/env python3
"""Skill Sync client -- the low-level sync layer.

Builds content-addressed objects from local skills and talks the sync wire
contract (push objects + CAS a ref, pull the owner's HEAD, three-way merge on
a 409). Driven by the debounced push hook in ``skill_manage``, the periodic
``maybe_pull_skills`` at the curator tick sites, and the ``hermes sync`` CLI.
Lives under tools/ (NOT hermes_cli/) so it never imports the CLI at module
load. ``skills_sync_client_wire`` (object model + HTTP client + tree walks) and
``skills_sync_client_org`` (org-shared skills) are re-exported here.

ACCESS GATE (pre-launch): sync is INERT unless the signed-in user is a Nous
admin, read off the ``tool_gateway_admin`` JWT claim. The claim name is NAS's
and misleading: it is the global portal-admin permission, NOT a tool-gateway
right. Replace with a real entitlement before shipping to users.

OPT-IN DEFAULT (provisional): nothing syncs unless the user marks a skill for
sync. Local intent is the ``sync`` flag in ``.usage.json``; the DURABLE
cross-device state is the committed ``sync-manifest`` blob in the plane. Only
agent-created + user-authored skills under ``~/.hermes/skills/`` are eligible.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from contextlib import suppress
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Dict, List, Optional, Tuple

from tools.skills_sync_client_wire import (  # noqa: F401  (re-exports)
    ARTIFACT_TYPE_SKILL, DEFAULT_MAX_OBJECT_BYTES, KIND_BLOB, KIND_COMMIT, KIND_TREE, MODE_DIR,
    MODE_EXEC, MODE_FILE, SYNC_MANIFEST_ENTRY_NAME, SYNC_MANIFEST_TYPE, SYNC_MANIFEST_VERSION,
    WIRE_VERSION, ObjectSet, SyncClient, SyncConflict, SyncError, _check_version,
    assemble_root_from_skill_trees, build_commit, build_root_tree, build_sync_manifest_bytes,
    build_tree, canonical_json_bytes, materialize_tree, merge_skill, nest_skill_tree,
    parse_sync_manifest, read_manifest_of_root, read_ref_hash, root_tree_of_commit,
    skill_trees_of_root, wire_address,
)

logger = logging.getLogger(__name__)

_merge_skill = merge_skill
_skill_trees_of_root = skill_trees_of_root


# Identity & access gate. The bearer comes from resolve_nous_runtime_credentials()
# (file lock, host allowlist, refresh -- not reimplemented); its payload is
# decoded unverified to read the gate claim.

# Wire claim name is NAS's; it means "Nous admin" (Permissions.ADMIN_ACCESS).
NOUS_ADMIN_CLAIM = "tool_gateway_admin"


class SyncInertError(RuntimeError):
    """Sync must no-op: not logged in, no bearer, or not a Nous admin.
    Caught by the gate-and-swallow hooks."""


def resolve_identity() -> Dict[str, Any]:
    """Resolve ``{api_key, base_url, owner, nous_admin, claims}``; raises
    SyncInertError if not logged in / no bearer. ``owner`` is advisory for local
    ref naming only -- the server derives the real owner from the bearer.

    The JWT payload is decoded WITHOUT signature verification. Safe: the claims
    are never used for authz here (the server re-verifies); they only decide
    whether to attempt sync at all.
    """
    try:
        from hermes_cli.auth import resolve_nous_runtime_credentials

        creds = resolve_nous_runtime_credentials() or {}
    except Exception as e:
        raise SyncInertError(f"no Nous credentials: {e}") from e

    api_key = creds.get("api_key")
    if not api_key:
        raise SyncInertError("no bearer token available")

    try:
        import jwt  # PyJWT, a core dependency

        claims = jwt.decode(api_key, options={"verify_signature": False, "verify_exp": False}) or {}
    except Exception as e:
        logger.debug("skills_sync_client: JWT payload decode failed: %s", e)
        claims = {}
    owner = claims.get("sub") or claims.get("privy_did") or claims.get("tid") or "unknown"
    return {
        "api_key": api_key,
        "base_url": creds.get("base_url"),
        "owner": str(owner),
        "nous_admin": claims.get(NOUS_ADMIN_CLAIM) is True,
        "claims": claims,
    }


# Configuration -- env-first so a Hermes Cloud instance can enable sync purely
# through environment variables. Every knob: HERMES_SYNC_<KEY> env -> config.yaml
# ``sync.<key>`` -> built-in default (keys: base_url = the sync plane, NOT the
# inference URL; enabled; default_opt_in; org_auto_propose).

#: Production Skill Sync plane; a normal user configures nothing.
DEFAULT_SYNC_BASE_URL = "https://gateway-gateway.nousresearch.com"

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off", ""}


def _sync_config(key: str) -> Any:
    """``sync.<key>`` from config.yaml, or None. Lazy import: this layer must not
    import the CLI at module load."""
    try:
        from hermes_cli.config import load_config

        return ((load_config() or {}).get("sync") or {}).get(key)
    except Exception as e:
        logger.debug("skills_sync_client: config sync.%s read failed: %s", key, e)
        return None


def resolve_sync_base_url() -> Optional[str]:
    """HERMES_SYNC_BASE_URL -> ``sync.base_url`` -> production plane, without a
    trailing slash (``/v1/sync/`` is appended by the client). None only if the
    default is blanked out."""
    env = os.getenv("HERMES_SYNC_BASE_URL")
    if env and env.strip():
        return env.strip().rstrip("/")
    base = _sync_config("base_url")
    if isinstance(base, str) and base.strip():
        return base.strip().rstrip("/")
    return DEFAULT_SYNC_BASE_URL or None


def _parse_bool(value: Any) -> Optional[bool]:
    """Parse a config/env bool; None if unrecognized so callers fall through to
    the next precedence layer."""
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    s = str(value).strip().lower()
    return True if s in _TRUE else False if s in _FALSE else None


def _sync_config_bool(env_var: str, config_key: str, *, default: bool) -> bool:
    """``env_var`` -> ``sync.<config_key>`` -> default."""
    env_val = _parse_bool(os.getenv(env_var))
    if env_val is not None:
        return env_val
    cfg_val = _parse_bool(_sync_config(config_key))
    return default if cfg_val is None else cfg_val


def sync_feature_enabled() -> bool:
    """Master switch. Checked by the gate-and-swallow entrypoints IN ADDITION to
    the Nous-admin gate and a configured base URL -- all three must hold."""
    return _sync_config_bool("HERMES_SYNC_ENABLED", "enabled", default=False)


def sync_org_auto_propose() -> bool:
    """False (default): edits to an org skill stay LOCAL until ``hermes sync
    propose``. True: every edit is proposed right away (an admin still approves
    unless the editor is one) -- for small high-trust teams."""
    return _sync_config_bool("HERMES_SYNC_ORG_AUTO_PROPOSE", "org_auto_propose", default=False)


def sync_default_opt_in() -> bool:
    """False (default): opt-IN -- a skill syncs only after ``hermes sync enable``
    or a plane manifest opting it in. True: opt-OUT -- every eligible skill
    syncs unless explicitly disabled (the Hermes Cloud default). Provisional."""
    return _sync_config_bool("HERMES_SYNC_DEFAULT_OPT_IN", "default_opt_in", default=False)


# Local skill eligibility + the personal opt-in flag

def _skills_dir() -> Path:
    from hermes_constants import get_hermes_home

    return get_hermes_home() / "skills"


def _org_dir() -> Path:
    """Local mirror root for org skills (read-only by convention)."""
    return _skills_dir() / ORG_DIR_NAME


def _rel_to_skills_dir(skill_dir: Path) -> Optional[Path]:
    """*skill_dir* relative to ~/.hermes/skills/, or None if outside/unresolvable."""
    try:
        return skill_dir.resolve().relative_to(_skills_dir().resolve())
    except (OSError, ValueError):
        return None


def _skill_rel_path(skill_name: str) -> Optional[PurePosixPath]:
    """The skill's path relative to ~/.hermes/skills/ (posix), or None."""
    try:
        from tools.skill_usage import _find_skill_dir
    except Exception:
        return None
    skill_dir = _find_skill_dir(skill_name)
    rel = _rel_to_skills_dir(skill_dir) if skill_dir is not None else None
    return PurePosixPath(rel.as_posix()) if rel is not None else None


def is_sync_eligible(skill_name: str) -> bool:
    """Candidate for sync (before the opt-in check): present locally, NOT
    bundled, NOT hub-installed, NOT external, and NOT under the ``_org/`` mirror
    (enterprise content must never ride a personal push). Mirrors the curator's
    exclusions (tools/skill_usage.py)."""
    try:
        from tools.skill_usage import is_bundled, is_hub_installed, _find_skill_dir
        from agent.skill_utils import is_external_skill_path
    except Exception:
        return False
    if is_bundled(skill_name) or is_hub_installed(skill_name):
        return False
    skill_dir = _find_skill_dir(skill_name)
    if skill_dir is None or is_external_skill_path(skill_dir):
        return False
    rel = _rel_to_skills_dir(skill_dir)
    return not (rel is not None and rel.parts and rel.parts[0] == ORG_DIR_NAME)


def list_synced_skill_names() -> List[str]:
    """Names of skills that should sync (sorted, deduped), per ``sync_default_opt_in()``:
    opt-in -> only eligible skills whose usage record has ``sync: true``;
    opt-out -> every eligible skill unless its record has ``sync: false``."""
    try:
        from tools.skill_usage import load_usage
    except Exception:
        return []
    usage = load_usage() or {}
    flags = {n: rec.get("sync") for n, rec in usage.items() if isinstance(rec, dict)}
    if sync_default_opt_in():
        names = [n for n in _all_local_skill_names() if flags.get(n) is not False and is_sync_eligible(n)]
    else:
        names = [n for n, f in flags.items() if f is True and is_sync_eligible(n)]
    return sorted(set(names))


def _all_local_skill_names() -> List[str]:
    """Every locally-present skill name (a dir under ~/.hermes/skills/ with a
    ``SKILL.md``; frontmatter ``name`` falling back to the dir name). Eligibility
    is applied by the caller."""
    names: List[str] = []
    root = _skills_dir()
    try:
        for skill_md in root.rglob("SKILL.md") if root.exists() else ():
            if skill_md.is_symlink():
                continue
            name = skill_md.parent.name
            try:
                from tools.skill_usage import _read_skill_name

                name = _read_skill_name(skill_md, name)
            except Exception:
                pass
            if name:
                names.append(name)
    except OSError as e:
        logger.debug("skills_sync_client: local skill enumeration failed: %s", e)
    return sorted(set(names))


def _opted_in_rel_paths() -> List[str]:
    """Relative posix paths of skills the user has opted into sync."""
    rels = (_skill_rel_path(name) for name in list_synced_skill_names())
    return [rel.as_posix() for rel in rels if rel is not None]


def _adopt_manifest_opt_ins(remote_manifest: Optional[Dict[str, bool]]) -> List[str]:
    """Enable local sync intent for skills the plane manifest has enabled and that
    are locally curation-eligible. Enables only -- a pull never silently disables.
    Returns the adopted names; best-effort."""
    adopted: List[str] = []
    if not remote_manifest:
        return adopted
    try:
        from tools.skill_usage import set_sync, is_curation_eligible, is_sync_enabled

        for sname, enabled in remote_manifest.items():
            if enabled and is_curation_eligible(sname) and not is_sync_enabled(sname):
                set_sync(sname, True)
                adopted.append(sname)
    except Exception as e:
        logger.debug("skills_sync_client: manifest opt-in reconcile failed: %s", e)
    return adopted


# Device label (commit ``author.device``; advisory, never an auth input)

def _default_device_label() -> str:
    """Short hostname + short random suffix (two machines can share a hostname);
    a bare uuid if the hostname is unusable."""
    import socket
    import uuid

    try:
        host = socket.gethostname() or ""
    except OSError:
        host = ""
    short = "".join(c for c in host.split(".")[0].strip() if c.isalnum() or c in "-_")
    return f"{short}-{uuid.uuid4().hex[:6]}" if short else uuid.uuid4().hex


def stable_device_id() -> str:
    """Stable per-device label, persisted at ~/.hermes/skills/.sync_device_id.
    An existing file always wins. Otherwise seeded from HERMES_SYNC_DEVICE_NAME
    (first use only; lets Hermes Cloud name hosted instances) or a friendly
    default, then persisted so a later ``set_device_name()`` still wins."""
    path = _skills_dir() / ".sync_device_id"
    try:
        if path.exists():
            val = path.read_text(encoding="utf-8").strip()
            if val:
                return val
    except OSError:
        pass
    val = (os.environ.get("HERMES_SYNC_DEVICE_NAME") or "").strip() or _default_device_label()
    try:
        _write_device_id(val)
    except OSError as e:
        logger.debug("skills_sync_client: could not persist device id: %s", e)
    return val


def _write_device_id(val: str) -> None:
    path = _skills_dir() / ".sync_device_id"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(val, encoding="utf-8")


def set_device_name(name: str) -> str:
    """Overwrite the device label with the trimmed *name*; any non-empty string
    is accepted. Returns the stored value; ValueError on empty."""
    cleaned = (name or "").strip()
    if not cleaned:
        raise ValueError("device name must be a non-empty string")
    _write_device_id(cleaned)
    return cleaned


# Local sync STATE: the last HEAD we pushed/pulled + the root tree at that
# point (FULL-digest namespace). Distinct from the bundled manifest
# (skills_sync.py) and from the plane's `sync-manifest` object. Lives at
# ~/.hermes/skills/.sync_state; a legacy `.sync_manifest` is migrated on read.

_EMPTY_STATE: Dict[str, Any] = {"head": None, "skills": {}}


def _sync_state_path() -> Path:
    return _skills_dir() / ".sync_state"


def _load_state_file(path: Path, what: str = "sync state read") -> Optional[Dict[str, Any]]:
    """Parse a state file; None if missing / corrupt / not a dict."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        logger.debug("skills_sync_client: %s failed: %s", what, e)
        return None
    if not isinstance(data, dict):
        return None
    data.setdefault("head", None)
    data.setdefault("skills", {})
    return data


def read_sync_state() -> Dict[str, Any]:
    """``{"head": "sha256:...|null", "skills": {...}}``; a default on missing/corrupt.
    If ``.sync_state`` is absent but the legacy ``.sync_manifest`` exists, it is
    read, rewritten to the new path and removed, so no head record is lost."""
    path = _sync_state_path()
    if path.exists():
        return _load_state_file(path) or dict(_EMPTY_STATE)
    legacy = _skills_dir() / ".sync_manifest"
    if legacy.exists():
        data = _load_state_file(legacy, "legacy sync state migrate")
        if data is not None:
            write_sync_state(data)
            with suppress(OSError):
                legacy.unlink()
            return data
    return dict(_EMPTY_STATE)


def write_sync_state(data: Dict[str, Any]) -> None:
    """Write the local sync state atomically. Best-effort."""
    path = _sync_state_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".sync_state_", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, sort_keys=True, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        except BaseException:
            with suppress(OSError):
                os.unlink(tmp)
            raise
    except Exception as e:
        logger.debug("skills_sync_client: sync state write failed: %s", e)


def _record_head(state: Dict[str, Any], head: str, root: str) -> None:
    state["head"] = head
    state["root"] = root
    write_sync_state(state)


# Profile snapshot -- the root tree mirrors each synced skill's relative path
# under ~/.hermes/skills/ (category dirs become intermediate trees).

def snapshot_profile(
    skill_names: List[str], *, max_object_bytes: int = DEFAULT_MAX_OBJECT_BYTES
) -> Tuple[ObjectSet, str, Dict[str, str]]:
    """Build all objects for *skill_names* + the profile-root tree; returns
    ``(objects, root_tree_hash, {skill_name: tree_hash})``. Skills whose blobs
    exceed *max_object_bytes* are skipped (logged). The root also carries the
    ``sync-manifest`` blob listing every included skill as ``enabled: true``."""
    from tools.skill_usage import _find_skill_dir

    objects = ObjectSet()
    skill_tree_map: Dict[str, str] = {}
    root: Dict[str, Any] = {}
    for name in sorted(set(skill_names)):
        rel = _skill_rel_path(name)
        skill_dir = _find_skill_dir(name)
        if rel is None or skill_dir is None:
            continue
        try:
            tree_hash = build_tree(skill_dir, objects, max_object_bytes=max_object_bytes)
        except ValueError as e:
            logger.warning("skills_sync_client: skipping %s: %s", name, e)
            continue
        skill_tree_map[name] = tree_hash
        nest_skill_tree(root, rel.parts, tree_hash)

    manifest_hash = objects.add(KIND_BLOB, build_sync_manifest_bytes(dict.fromkeys(skill_tree_map, True)))
    return objects, build_root_tree(root, objects, manifest_hash=manifest_hash), skill_tree_map


# Personal refs, push, pull

def user_head_ref(owner: str) -> str:
    return f"refs/user/{owner}/HEAD"


def _personal_client(
    identity: Optional[Dict[str, Any]], client: Optional[SyncClient]
) -> Tuple[Dict[str, Any], Optional[SyncClient]]:
    """Resolve identity + client for a personal sync op. ``client`` is None when
    no base URL is configured (callers return a no-op result)."""
    identity = identity if identity is not None else resolve_identity()
    if client is None:
        base = resolve_sync_base_url()
        client = SyncClient(base, identity["api_key"]) if base else None
    return identity, client


_NO_BASE_URL = {"ok": False, "reason": "no sync base url configured", "noop": True}


def push_skills(
    client: Optional[SyncClient] = None,
    *,
    skill_names: Optional[List[str]] = None,
    identity: Optional[Dict[str, Any]] = None,
    message: str = "hermes skill sync",
) -> Dict[str, Any]:
    """Push opted-in skills to ``refs/user/<owner>/HEAD``: upload new objects, CAS
    HEAD. A 409 with an actual head -> three-way merge + one retry; a 409 against
    a NON-EXISTENT ref (stale local head, e.g. carried over from another plane)
    -> redo the CAS as a create. Never raises for the inert / no-op cases."""
    identity, client = _personal_client(identity, client)
    if client is None:
        return dict(_NO_BASE_URL)
    owner = identity["owner"]

    if skill_names is None:
        skill_names = list_synced_skill_names()
    if not skill_names:
        return {"ok": True, "reason": "no skills opted into sync", "noop": True}

    caps = client.capabilities()
    _check_version(caps)
    max_bytes = int(caps.get("max_object_bytes") or DEFAULT_MAX_OBJECT_BYTES)
    objects, root_hash, _ = snapshot_profile(skill_names, max_object_bytes=max_bytes)

    state = read_sync_state()
    base_head = state.get("head")
    # Idempotency: objects are immutable, so an unchanged root tree hash means
    # identical content -- skip building an empty commit.
    if base_head and state.get("root") == root_hash:
        return {"ok": True, "head": base_head, "reason": "unchanged", "noop": True}

    commit_hash = build_commit(
        root_hash, [base_head] if base_head else [], owner=owner, device=stable_device_id(),
        message=message, objects=objects,
    )
    client.put_objects(objects.objects)
    ref = user_head_ref(owner)
    result = {"ok": True, "head": commit_hash, "pushed_objects": len(objects)}
    try:
        client.cas_ref(ref, base_head, commit_hash)
    except SyncConflict as conflict:
        if conflict.actual:
            return _resolve_push_conflict(
                client, identity, conflict.actual, root_hash, commit_hash, objects, message, base_head
            )
        client.cas_ref(ref, None, commit_hash)
        result["recovered_stale_head"] = True
    _record_head(state, commit_hash, root_hash)
    return result


# Three-way merge per skill against the base we forked from (see merge_skill):
# each side changing a DIFFERENT skill -> merge commit (2 parents) + CAS retry;
# both changing the SAME skill differently -> TRUE OVERLAP -> written to
# refs/user/<owner>/conflict/<n> and surfaced for out-of-band resolution.
def _resolve_push_conflict(
    client: SyncClient,
    identity: Dict[str, Any],
    actual_head: str,
    our_root: str,
    our_commit: str,
    objects: ObjectSet,
    message: str,
    base_head: Optional[str],
) -> Dict[str, Any]:
    owner = identity["owner"]
    ours_trees = skill_trees_of_root(client, our_root)
    theirs_trees = skill_trees_of_root(client, root_tree_of_commit(client, actual_head))
    base_trees = skill_trees_of_root(client, root_tree_of_commit(client, base_head)) if base_head else {}

    merged: Dict[str, str] = {}
    overlaps: List[str] = []
    for path in set(ours_trees) | set(theirs_trees) | set(base_trees):
        o, t = ours_trees.get(path), theirs_trees.get(path)
        decision = merge_skill(base_trees.get(path), o, t)
        if decision == "overlap":
            overlaps.append(path)
        # overlap keeps OURS on the surfaced conflict head (theirs stays
        # server-side); "none" = deleted on the winning side -> drop.
        pick = {"overlap": o, "ours": o, "theirs": t, "either": o if o is not None else t}.get(decision)
        if pick is not None:
            merged[path] = pick

    if overlaps:
        conflict_ref = f"refs/user/{owner}/conflict/{_next_conflict_index(client, owner)}"
        try:
            client.cas_ref(conflict_ref, None, our_commit)
        except SyncConflict:
            pass  # someone else grabbed this index; the head still exists
        return {
            "ok": False, "conflict": True, "conflict_ref": conflict_ref,
            "overlapping_skills": sorted(overlaps), "actual_head": actual_head,
            "message": (
                f"{len(overlaps)} skill(s) changed on both sides; wrote "
                f"{conflict_ref}. Resolve out-of-band (hermes sync / NAS UI)."
            ),
        }

    # Merge commit (parents: actual, ours); re-add our objects so the merge
    # push is self-contained.
    merge_objects = ObjectSet()
    merge_objects.objects.update(objects.objects)
    merged_root = assemble_root_from_skill_trees(merged, merge_objects)
    merge_commit = build_commit(
        merged_root, [actual_head, our_commit], owner=owner, device=stable_device_id(),
        message=f"merge: {message}", objects=merge_objects,
    )
    client.put_objects(merge_objects.objects)
    try:
        client.cas_ref(user_head_ref(owner), actual_head, merge_commit)
    except SyncConflict as c2:
        return {
            "ok": False, "conflict": True, "actual_head": c2.actual,
            "message": f"merge CAS lost again (head now {c2.actual}); retry sync.",
        }
    _record_head(read_sync_state(), merge_commit, merged_root)
    return {"ok": True, "head": merge_commit, "merged": True}


def _next_conflict_index(client: SyncClient, owner: str) -> int:
    """Next free ``conflict/<n>`` index for the owner."""
    try:
        refs = client.get_refs(f"refs/user/{owner}/conflict/")
    except SyncError:
        return 1
    used = [int(t) for t in (r.get("name", "").rsplit("/", 1)[-1] for r in refs) if t.isdigit()]
    return (max(used) + 1) if used else 1


def pull_skills(
    client: Optional[SyncClient] = None, *, identity: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Pull the owner's HEAD and materialize opted-in skills under ~/.hermes/skills/
    if it advanced past our recorded head. Opt-in intent is first adopted FROM
    the plane manifest, then only opted-in paths are written so a pull never
    resurrects a skill the user hasn't chosen. Best-effort; returns a result dict."""
    identity, client = _personal_client(identity, client)
    if client is None:
        return dict(_NO_BASE_URL)
    owner = identity["owner"]

    _check_version(client.capabilities())
    head = read_ref_hash(client, user_head_ref(owner))
    if not head:
        return {"ok": True, "reason": "no remote HEAD yet", "noop": True}

    state = read_sync_state()
    if head == state.get("head"):
        return {"ok": True, "reason": "already up to date", "head": head, "noop": True}

    root_tree = root_tree_of_commit(client, head)
    remote_trees = skill_trees_of_root(client, root_tree)

    adopted = _adopt_manifest_opt_ins(read_manifest_of_root(client, root_tree))
    opted_in = set(_opted_in_rel_paths())
    updated = [path for path in remote_trees if not opted_in or path in opted_in]
    for path in updated:
        materialize_tree(client, remote_trees[path], _skills_dir() / path)

    state["head"] = head
    write_sync_state(state)
    return {"ok": True, "head": head, "updated": sorted(updated), "opt_in_adopted": sorted(adopted)}


# Gated public entrypoints (gate-and-swallow, like the curator's
# maybe_run_curator): best-effort, never raise, return a result dict or None.

def _gate_and_swallow(op: str, run: Callable[[Dict[str, Any]], Optional[Dict[str, Any]]]):
    """Run *run(identity)* only if every background-sync gate holds (Nous admin,
    feature on, base URL); None when inert or on any error."""
    try:
        identity = resolve_identity()
        if not identity.get("nous_admin") or not sync_feature_enabled() or not resolve_sync_base_url():
            return None
        return run(identity)
    except Exception as e:
        logger.debug("skills_sync_client: %s failed: %s", op, e, exc_info=True)
        return None


def maybe_push_skills(*, message: str = "hermes skill sync") -> Optional[Dict[str, Any]]:
    """Best-effort push (debounced skill_manage hook). Never raises."""
    return _gate_and_swallow(
        "maybe_push_skills",
        lambda identity: push_skills(identity=identity, message=message) if list_synced_skill_names() else None,
    )


def maybe_pull_skills() -> Optional[Dict[str, Any]]:
    """Best-effort pull (curator tick sites: gateway housekeeping + CLI startup).
    Never raises."""
    return _gate_and_swallow("maybe_pull_skills", lambda identity: pull_skills(identity=identity))


def sync_status() -> Dict[str, Any]:
    """Status snapshot for ``hermes sync status``. Never raises.
    ``org_available`` False means the account isn't in a shared organisation
    (the org workflow does not apply), not that anything is broken."""
    status: Dict[str, Any] = {
        "nous_admin": False, "logged_in": False, "feature_enabled": sync_feature_enabled(),
        "default_opt_in": sync_default_opt_in(), "base_url": resolve_sync_base_url(),
        "opted_in_skills": [], "local_head": None, "owner": None, "org_available": False,
        "org_id": None, "org_role": None, "org_skills": [], "org_skills_modified": [],
    }

    try:
        identity = resolve_identity()
        status.update(logged_in=True, owner=identity.get("owner"), nous_admin=bool(identity.get("nous_admin")))
    except SyncInertError:
        pass
    except Exception as e:
        logger.debug("skills_sync_client: sync_status identity failed: %s", e)
    try:
        status["opted_in_skills"] = list_synced_skill_names()
        status["local_head"] = read_sync_state().get("head")
    except Exception:
        pass
    try:
        org_identity = resolve_org_identity()
        status.update(
            org_available=True, org_id=org_identity.get("org_id"), org_role=org_identity.get("org_role"),
            org_skills=list_org_skill_names(),
            org_skills_modified=list_locally_modified_org_skills(org_identity.get("org_id")),
        )
    except SyncInertError:
        pass
    except Exception as e:
        logger.debug("skills_sync_client: sync_status org lookup failed: %s", e)
    return status


# Org-shared skills live in skills_sync_client_org; imported last because that
# module reads this module's state lazily.
from tools.skills_sync_client_org import (  # noqa: E402,F401  (re-exports)
    ORG_DIR_NAME, _ORG_CAS_MAX_ATTEMPTS, _clear_active_org_marker, _org_baseline_path,
    _read_org_baseline, _read_org_head, _skill_dir_fingerprint, _write_active_org_marker,
    _write_org_baseline, _write_org_provenance, list_locally_modified_org_skills,
    list_org_skill_names, maybe_pull_org_skills, org_head_ref, org_skill_is_locally_modified,
    propose_skill, pull_org_skills, resolve_org_identity,
)
