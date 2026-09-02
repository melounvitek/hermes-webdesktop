"""Skill Sync wire model: content-addressed objects, the HTTP client, tree walks.

Everything here is independent of local skill state (no ``~/.hermes`` reads);
``tools/skills_sync_client.py`` orchestrates push/pull on top of it and
re-exports these names.

Wire contract (version 1). ``hsp_version`` / ``X-HSP-Object-Type`` are deployed
protocol identifiers and are NOT renamed with the product name "Skill Sync".
"""

from __future__ import annotations

import hashlib
import json
import logging
import stat as _stat
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("tools.skills_sync_client")

WIRE_VERSION = "1"
DEFAULT_MAX_OBJECT_BYTES = 26214400  # 25 MiB, mirrors capabilities default

KIND_BLOB = "blob"
KIND_TREE = "tree"
KIND_COMMIT = "commit"

MODE_FILE = "file"
MODE_EXEC = "exec"
MODE_DIR = "dir"

ARTIFACT_TYPE_SKILL = "skill"
_EXEC_BITS = _stat.S_IXUSR | _stat.S_IXGRP | _stat.S_IXOTH

# `sync-manifest`: per-skill opt-in is CONTENT in the object model, not a
# device-local flag -- a root-level blob in the tree at refs/user/<owner>/HEAD
# recording {name, enabled}. The plane manifest is authoritative; the local
# `.usage.json` `sync` flag is only the editable intent (reconciled FROM it on
# pull, TO it on push). Shape MUST match gateway-gateway src/sync/manifest.ts.
SYNC_MANIFEST_ENTRY_NAME = "sync-manifest"
SYNC_MANIFEST_TYPE = "sync-manifest"
SYNC_MANIFEST_VERSION = 1


# Content addressing. The wire uses the FULL 64-hex sha256 -- a different
# namespace from the truncated 16-hex local `content_hash` (skills_guard.py).

def wire_address(data: bytes) -> str:
    """Return ``sha256:<64-hex>`` -- the wire address of ``data``."""
    return "sha256:" + hashlib.sha256(data).hexdigest()


def canonical_json_bytes(obj: Dict[str, Any]) -> bytes:
    """Canonical JSON for tree/commit hashing: UTF-8, sorted keys, no whitespace,
    no trailing newline. Arrays must already be in contract order (tree entries
    by name, commit parents by significance). Client and server MUST produce
    byte-identical output or a push fails ``422 hash_mismatch``."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def build_sync_manifest_bytes(skills: Dict[str, bool]) -> bytes:
    """Serialize ``{name: enabled}`` into canonical ``sync-manifest`` bytes
    (entries sorted by name for a stable content address)."""
    return canonical_json_bytes({
        "type": SYNC_MANIFEST_TYPE, "version": SYNC_MANIFEST_VERSION,
        "skills": [{"name": name, "enabled": bool(enabled)} for name, enabled in sorted(skills.items())],
    })


def parse_sync_manifest(data: bytes) -> Optional[Dict[str, bool]]:
    """Parse ``sync-manifest`` bytes into ``{name: enabled}``, or None if malformed.

    Strict (mirrors gateway-gateway ``parseSyncManifest``): unknown type, version
    != 1, non-list skills, or a malformed entry all reject -- a malformed
    manifest must not be mistaken for "no skills opted in".
    """
    try:
        value = json.loads(data.decode("utf-8"))
    except Exception:
        return None
    if (
        not isinstance(value, dict)
        or value.get("type") != SYNC_MANIFEST_TYPE
        or value.get("version") != SYNC_MANIFEST_VERSION
        or not isinstance(value.get("skills"), list)
    ):
        return None
    out: Dict[str, bool] = {}
    for raw in value["skills"]:
        if not isinstance(raw, dict):
            return None
        name, enabled = raw.get("name"), raw.get("enabled")
        if not isinstance(name, str) or not name or not isinstance(enabled, bool):
            return None
        out[name] = enabled
    return out


# Object building

class ObjectSet:
    """Objects to push, ``hash -> (kind, bytes)``, deduped by content address."""

    def __init__(self) -> None:
        self.objects: Dict[str, Tuple[str, bytes]] = {}

    def add(self, kind: str, data: bytes) -> str:
        addr = wire_address(data)
        self.objects.setdefault(addr, (kind, data))
        return addr

    def __len__(self) -> int:
        return len(self.objects)


def _entry(name: str, kind: str, hash_: str, mode: str) -> Dict[str, str]:
    return {"name": name, "kind": kind, "hash": hash_, "mode": mode}


def _add_tree(entries: List[Dict[str, str]], objects: ObjectSet) -> str:
    """Canonicalize *entries* (sorted by name, byte order) into a tree object."""
    entries.sort(key=lambda e: e["name"])
    return objects.add(KIND_TREE, canonical_json_bytes({"type": KIND_TREE, "entries": entries}))


def _file_mode(path: Path) -> str:
    """``exec`` if +x else ``file``. No symlink / other modes are emitted."""
    with suppress(OSError):
        if path.stat().st_mode & _EXEC_BITS:
            return MODE_EXEC
    return MODE_FILE


def build_tree(dir_path: Path, objects: ObjectSet, *, max_object_bytes: int) -> str:
    """Recursively build objects for *dir_path*; return the tree address. Files
    -> blobs, subdirs -> nested trees; symlinks/special files are skipped
    (contract: no symlinks). A blob over *max_object_bytes* raises ValueError
    so the caller can surface / skip the artifact (server -> 413)."""
    entries: List[Dict[str, str]] = []
    for child in sorted(dir_path.iterdir(), key=lambda p: p.name):
        if child.is_symlink():
            logger.debug("skills_sync_client: skipping symlink %s", child)
        elif child.is_dir():
            sub_hash = build_tree(child, objects, max_object_bytes=max_object_bytes)
            entries.append(_entry(child.name, KIND_TREE, sub_hash, MODE_DIR))
        elif child.is_file():
            data = child.read_bytes()
            if len(data) > max_object_bytes:
                raise ValueError(
                    f"file {child} is {len(data)} bytes > max_object_bytes {max_object_bytes}"
                )
            entries.append(_entry(child.name, KIND_BLOB, objects.add(KIND_BLOB, data), _file_mode(child)))
    return _add_tree(entries, objects)


def build_commit(
    tree_hash: str, parents: List[str], *, owner: str, device: str, message: str,
    objects: ObjectSet, ts: Optional[str] = None,
) -> str:
    """Build a commit object and return its address.

    ``parents``: 0 for the first commit, 1 for an edit, 2 for a merge (order
    significant: parents[0] = base fast-forwarded from, parents[1] = other head).
    """
    return objects.add(KIND_COMMIT, canonical_json_bytes({
        "type": KIND_COMMIT, "tree": tree_hash, "parents": list(parents),
        "author": {"owner": owner, "device": device},
        "ts": ts or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "message": message, "artifact_type": ARTIFACT_TYPE_SKILL,
    }))


def build_root_tree(node: Dict[str, Any], objects: ObjectSet, *, manifest_hash: Optional[str] = None) -> str:
    """Canonicalize a nested ``{name: {"__tree__": hash} | subdict}`` root into trees.

    ``manifest_hash`` (top level only) adds the root ``sync-manifest`` BLOB entry.
    It cannot collide with a skill dir: skill entries are trees, this is a blob.
    """
    entries: List[Dict[str, str]] = []
    for name, child in node.items():
        if isinstance(child, dict) and "__tree__" in child and len(child) == 1:
            entries.append(_entry(name, KIND_TREE, child["__tree__"], MODE_DIR))
        else:
            entries.append(_entry(name, KIND_TREE, build_root_tree(child, objects), MODE_DIR))
    if manifest_hash is not None:
        entries.append(_entry(SYNC_MANIFEST_ENTRY_NAME, KIND_BLOB, manifest_hash, MODE_FILE))
    return _add_tree(entries, objects)


def nest_skill_tree(root: Dict[str, Any], rel_parts: Tuple[str, ...], tree_hash: str) -> None:
    """Insert a skill tree leaf into the nested root structure by path parts."""
    node = root
    for part in rel_parts[:-1]:
        node = node.setdefault(part, {})
    node[rel_parts[-1]] = {"__tree__": tree_hash}


def assemble_root_from_skill_trees(skill_trees: Dict[str, str], objects: ObjectSet) -> str:
    """Build a profile-root tree from ``{posix_rel_path: tree_hash}``.

    The skill trees are assumed already durable (they came from either side of
    a merge / the org HEAD); only the new intermediate/root trees are added.
    """
    root: Dict[str, Any] = {}
    for path, tree_hash in skill_trees.items():
        nest_skill_tree(root, PurePosixPath(path).parts, tree_hash)
    return build_root_tree(root, objects)


# HTTP client (routes under /v1/sync/)

class SyncError(RuntimeError):
    """A non-recoverable wire error (4xx the client can't retry)."""

    def __init__(self, message: str, *, status: Optional[int] = None):
        super().__init__(message)
        self.status = status


class SyncConflict(RuntimeError):
    """CAS lost (409). NOT a rejection -- pushed objects are already durable.

    ``actual`` is the current head to merge against, or None when the ref does
    not exist server-side (the server sends ""). None means "retry as a create";
    it must never be fetched as an object -- normalized here, not per call site.
    """

    def __init__(self, actual: Optional[str]):
        self.actual: Optional[str] = actual or None
        super().__init__(
            f"CAS conflict; actual head {self.actual}"
            if self.actual
            else "CAS conflict; the ref does not exist yet"
        )


def _check_version(caps: Dict[str, Any]) -> None:
    """Reject an incompatible server major version."""
    ver = str(caps.get("hsp_version") or "")  # wire field name
    if ver.split(".", 1)[0] != WIRE_VERSION:
        raise SyncError(
            f"this server speaks sync version {ver!r}, but this Hermes speaks "
            f"{WIRE_VERSION} — update Hermes to sync with it"
        )


class SyncClient:
    """Sync client bound to a base URL + Nous bearer.

    Org refs/objects live behind SEPARATE ``org/`` routes, not a prefix filter:
    the personal routes are hard-scoped to the token's owner and would silently
    answer an org query with personal data. Callers reading org content MUST
    pass ``org_scope=True`` on every hop.
    """

    def __init__(self, base_url: str, api_key: str, *, timeout: float = 30.0):
        self.base = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        import requests  # core dependency

        self._session = requests.Session()
        self._session.headers["Authorization"] = f"Bearer {api_key}"

    def _url(self, path: str) -> str:
        return f"{self.base}/v1/sync/{path.lstrip('/')}"

    @staticmethod
    def _check(r, op: str, ok=(200,), errors: Optional[Dict[int, str]] = None) -> None:
        """Raise SyncError unless the status is in *ok*; *errors* maps specific
        statuses to their message, anything else gets ``"<op> failed: <code>"``."""
        if r.status_code in ok:
            return
        msg = (errors or {}).get(r.status_code)
        raise SyncError(msg or f"{op} failed: {r.status_code}", status=r.status_code)

    def capabilities(self) -> Dict[str, Any]:
        """GET capabilities (no auth required)."""
        r = self._session.get(self._url("capabilities"), timeout=self.timeout)
        self._check(r, "capabilities")
        return r.json()

    def get_refs(self, prefix: str, *, org_scope: bool = False) -> List[Dict[str, str]]:
        """GET refs?prefix=... (or org/refs, filtered client-side by *prefix*)."""
        path = "org/refs" if org_scope else "refs"
        params = None if org_scope else {"prefix": prefix}
        r = self._session.get(self._url(path), params=params, timeout=self.timeout)
        self._check(r, "get_refs")
        refs = (r.json() or {}).get("refs", [])
        if org_scope:
            refs = [r_ for r_ in refs if str(r_.get("name", "")).startswith(prefix)]
        return refs

    def get_object(self, obj_hash: str, *, org_scope: bool = False) -> Tuple[str, bytes]:
        """GET objects/:hash -> ``(kind, bytes)``. Kind comes from the object-type
        header; a blob (octet-stream) is returned as ``blob``."""
        path = f"org/objects/{obj_hash}" if org_scope else f"objects/{obj_hash}"
        r = self._session.get(self._url(path), timeout=self.timeout)
        self._check(
            r, "get_object",
            errors={404: f"object {obj_hash} not found", 403: f"object {obj_hash} not readable"},
        )
        return r.headers.get("X-HSP-Object-Type") or KIND_BLOB, r.content

    def _get_json_of_kind(self, obj_hash: str, expected: str, org_scope: bool) -> Dict[str, Any]:
        kind, data = self.get_object(obj_hash, org_scope=org_scope)
        if kind != expected:
            raise SyncError(f"{obj_hash} is {kind}, expected {expected}")
        return json.loads(data.decode("utf-8"))

    def get_commit_json(self, commit_hash: str, *, org_scope: bool = False) -> Dict[str, Any]:
        return self._get_json_of_kind(commit_hash, KIND_COMMIT, org_scope)

    def get_tree_json(self, tree_hash: str, *, org_scope: bool = False) -> Dict[str, Any]:
        return self._get_json_of_kind(tree_hash, KIND_TREE, org_scope)

    def put_objects(self, objects: Dict[str, Tuple[str, bytes]], *, org_scope: bool = False) -> Dict[str, Any]:
        """POST objects -- batch upload as multipart/form-data: field name = the
        claimed ``sha256:<hex>``, ``filename`` = object type, body = raw bytes
        (the contract requires raw bytes, not base64-in-JSON). The server
        recomputes every hash and rejects the whole batch with 422 on mismatch;
        known hashes are idempotent no-ops. ``org_scope`` adds ``?scope=org`` so
        objects land org-readable (required before an org CAS/propose)."""
        files = [(h, (kind, data, "application/octet-stream")) for h, (kind, data) in objects.items()]
        r = self._session.post(
            self._url("objects"), files=files, params={"scope": "org"} if org_scope else None, timeout=self.timeout
        )
        self._check(
            r, "put_objects", ok=(200, 201),
            errors={413: "object too large (413)", 422: f"hash_mismatch (422): {r.text}"},
        )
        return r.json() if r.content else {}

    def cas_ref(self, name: str, from_hash: Optional[str], to_hash: str) -> Dict[str, Any]:
        """POST refs/:name -- atomic compare-and-swap. Raises SyncConflict on 409.

        A non-admin member's CAS on an org HEAD is converted server-side to a
        proposal (202) and surfaced as ``{"proposal_pending": True, ...}``: a
        SUCCESS-shaped outcome that must never be presented as live/merged.
        """
        r = self._session.post(
            self._url(f"refs/{name}"), json={"from": from_hash, "to": to_hash}, timeout=self.timeout
        )
        if r.status_code == 202:
            return {"proposal_pending": True, **(r.json() if r.content else {})}
        if r.status_code == 409:
            # "" actual = the ref does not exist server-side (SyncConflict -> None).
            raise SyncConflict((r.json() or {}).get("actual", ""))
        self._check(r, "cas_ref", errors={403: "forbidden (403) -- owner/permission"})
        return r.json() if r.content else {}


# Reading remote trees

def read_ref_hash(client: SyncClient, ref: str, *, org_scope: bool = False) -> Optional[str]:
    """Hash of *ref* (queried with itself as prefix), or None if absent."""
    refs = client.get_refs(ref, org_scope=org_scope)
    return next((r.get("hash") for r in refs if r.get("name") == ref), None)


def root_tree_of_commit(client: SyncClient, commit_hash: str, *, org_scope: bool = False) -> str:
    return client.get_commit_json(commit_hash, org_scope=org_scope)["tree"]


def skill_trees_of_root(client: SyncClient, root_tree_hash: str, *, org_scope: bool = False) -> Dict[str, str]:
    """Flatten a profile-root tree into ``{posix_rel_path: skill_tree_hash}``. A
    skill tree is any subtree containing a ``SKILL.md`` blob, keyed by its path
    so category nesting is preserved."""
    result: Dict[str, str] = {}

    def _walk(tree_hash: str, prefix: str) -> None:
        entries = client.get_tree_json(tree_hash, org_scope=org_scope).get("entries", [])
        if prefix and any(e.get("name") == "SKILL.md" and e.get("kind") == KIND_BLOB for e in entries):
            result[prefix] = tree_hash
            return
        for e in entries:
            if e.get("kind") == KIND_TREE:
                _walk(e["hash"], f"{prefix}/{e['name']}" if prefix else e["name"])

    _walk(root_tree_hash, "")
    return result


def read_manifest_of_root(client: SyncClient, root_tree_hash: str) -> Optional[Dict[str, bool]]:
    """``{name: enabled}`` from the root-level ``sync-manifest`` blob, or None if
    absent/malformed. This is how a device learns another device's opt-ins."""
    try:
        tree = client.get_tree_json(root_tree_hash)
    except Exception as e:
        logger.debug("skills_sync_client: manifest root read failed: %s", e)
        return None
    for e in tree.get("entries", []):
        if e.get("name") == SYNC_MANIFEST_ENTRY_NAME and e.get("kind") == KIND_BLOB:
            try:
                _kind, data = client.get_object(e["hash"])
            except Exception as ex:
                logger.debug("skills_sync_client: manifest blob fetch failed: %s", ex)
                return None
            return parse_sync_manifest(data)
    return None


def materialize_tree(client: SyncClient, tree_hash: str, dest: Path, *, org_scope: bool = False) -> None:
    """Write the tree at *tree_hash* into *dest* (created if needed): blobs ->
    files (+x restored for ``exec``), trees -> subdirectories. Does NOT delete
    files absent from the tree (caller decides). Refuses path traversal."""
    dest.mkdir(parents=True, exist_ok=True)
    for entry in client.get_tree_json(tree_hash, org_scope=org_scope).get("entries", []):
        name = entry.get("name", "")
        if not name or "/" in name or name in (".", ".."):
            logger.warning("skills_sync_client: skipping unsafe tree entry %r", name)
            continue
        target = dest / name
        kind = entry.get("kind")
        if kind == KIND_TREE:
            materialize_tree(client, entry["hash"], target, org_scope=org_scope)
        elif kind == KIND_BLOB:
            _, data = client.get_object(entry["hash"], org_scope=org_scope)
            target.write_bytes(data)
            if entry.get("mode") == MODE_EXEC:
                with suppress(OSError):
                    target.chmod(target.stat().st_mode | _EXEC_BITS)


def merge_skill(base: Optional[str], ours: Optional[str], theirs: Optional[str]) -> str:
    """Three-way decision for one skill's tree hash: ``ours`` / ``theirs`` /
    ``either`` / ``overlap`` / ``none``. A side "modified" the skill when its
    hash differs from the common base (same semantics as skills_sync.py's
    origin/user/incoming block)."""
    if ours == theirs:
        return "either" if ours is not None else "none"
    if theirs == base:  # only we moved
        return "ours"
    if ours == base:  # only they moved
        return "theirs"
    return "overlap"
