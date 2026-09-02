#!/usr/bin/env python3
"""
Skills Hub — hub state management and the public facade for the source adapters.

Library module (not an agent tool). Owns the hub paths, guarded HTTP, index
cache, lock file, taps and audit log. Install/uninstall/update live in
``skills_hub_install``, the index fetch/source router/search in
``skills_hub_search``, and the adapters in the other ``tools.skills_hub_*``
siblings; all are re-exported here so ``from tools.skills_hub import X`` keeps
working (and stays the test patch target).

Used by hermes_cli/skills_hub.py for CLI commands and the /skills slash command.
"""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

import httpx

from hermes_constants import get_hermes_home
from tools.url_safety import is_safe_url
from tools.website_policy import check_website_access
from tools.skills_hub_models import (  # noqa: F401  (re-exported public API)
    SkillMeta, SkillBundle, SkillSource, source_url_for_bundle,
    _referenced_support_paths, _normalize_bundle_path, _validate_skill_name,
    _validate_install_parent_path, _normalize_lock_install_path,
    _validate_bundle_rel_path, _skill_meta_to_dict, _parse_frontmatter,
    _dedupe_by_trust, TRUST_RANK,
)
from tools.skills_hub_github import (  # noqa: F401
    GITHUB_TAP_PROVIDERS, github_provider_for, _PROVIDER_FILTER_VALUES,
    _filter_results_by_provider, GitHubAuth, GitHubSource,
)
from tools.skills_hub_skillssh import SkillsShSource  # noqa: F401
from tools.skills_hub_clawhub import ClawHubSource  # noqa: F401
from tools.skills_hub_sources import (  # noqa: F401
    WellKnownSkillSource, UrlSource, LobeHubSource, BrowseShSource,
)
from tools.skills_hub_official import OptionalSkillSource, HermesIndexSource  # noqa: F401
from tools.skills_hub_search import (  # noqa: F401  (re-exported; tests patch tools.skills_hub.<name>)
    HERMES_INDEX_TTL,
    HERMES_INDEX_URL,
    _API_SOURCE_IDS,
    _hermes_index_cache_file,
    _load_hermes_index,
    _load_stale_index_cache,
    _search_one_source,
    _select_active_sources,
    create_source_router,
    parallel_search_sources,
    unified_search,
)
from tools.skills_hub_install import (  # noqa: F401  (re-exported; tests patch tools.skills_hub.<name>)
    _SOURCE_ID_ALIASES,
    _category_skill_dirs,
    _check_install_target,
    _is_path_redirect,
    _resolve_lock_install_path,
    _source_matches,
    bundle_content_hash,
    check_for_skill_updates,
    install_from_quarantine,
    quarantine_bundle,
    uninstall_skill,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Resolved per-call (not frozen at import) so the profile override is honored;
# import-time constants leaked across profiles in single-process multi-profile
# runtimes. Legacy names (SKILLS_DIR, ...) are re-exposed via __getattr__ below
# so external `from tools.skills_hub import SKILLS_DIR` callers still work.

INDEX_CACHE_TTL = 3600  # 1 hour


def _path_resolver(name: str, default):
    """Resolver for a hub path: a test-injected real module attribute
    (patch.object/monkeypatch on SKILLS_DIR etc.) wins over live resolution."""
    def resolve() -> Path:
        forced = globals().get(name)
        return Path(forced) if forced is not None else default()
    resolve.__name__ = f"_{name.lower()}"
    return resolve


def _hermes_home() -> Path:
    return get_hermes_home()


_skills_dir = _path_resolver("SKILLS_DIR", lambda: _hermes_home() / "skills")
_hub_dir = _path_resolver("HUB_DIR", lambda: _skills_dir() / ".hub")
_lock_file = _path_resolver("LOCK_FILE", lambda: _hub_dir() / "lock.json")
_quarantine_dir = _path_resolver("QUARANTINE_DIR", lambda: _hub_dir() / "quarantine")
_audit_log = _path_resolver("AUDIT_LOG", lambda: _hub_dir() / "audit.log")
_taps_file = _path_resolver("TAPS_FILE", lambda: _hub_dir() / "taps.json")
_index_cache_dir = _path_resolver("INDEX_CACHE_DIR", lambda: _hub_dir() / "index-cache")

_DYNAMIC_PATH_RESOLVERS = {
    "HERMES_HOME": _hermes_home,
    "SKILLS_DIR": _skills_dir,
    "HUB_DIR": _hub_dir,
    "LOCK_FILE": _lock_file,
    "QUARANTINE_DIR": _quarantine_dir,
    "AUDIT_LOG": _audit_log,
    "TAPS_FILE": _taps_file,
    "INDEX_CACHE_DIR": _index_cache_dir,
}


def __getattr__(name: str):
    """Resolve legacy path constants dynamically (PEP 562) so they reflect the
    active profile override; a test's patch.object-set real attribute shadows it."""
    resolver = _DYNAMIC_PATH_RESOLVERS.get(name)
    if resolver is not None:
        return resolver()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ---------------------------------------------------------------------------
# Install-path safety + guarded HTTP
# ---------------------------------------------------------------------------

_REDIRECT_STATUS_CODES = {301, 302, 303, 307, 308}
_MAX_SKILL_FETCH_REDIRECTS = 5


def _ssrf_safe_http_get(url: str, *, timeout: int = 20) -> httpx.Response:
    """Fetch one URL with connect-time SSRF validation and no automatic redirects."""
    from tools.url_safety import create_ssrf_safe_client

    with create_ssrf_safe_client(timeout=timeout, follow_redirects=False) as client:
        return client.get(url)


def _guarded_http_get(url: str, *, timeout: int = 20) -> Optional[httpx.Response]:
    """Fetch a URL with SSRF and redirect-target validation (each hop re-checked)."""
    from tools.url_safety import SSRFConnectionBlocked

    current_url = url

    for _ in range(_MAX_SKILL_FETCH_REDIRECTS + 1):
        if not is_safe_url(current_url):
            logger.warning("Blocked unsafe Skills Hub URL: %s", current_url)
            return None

        blocked = check_website_access(current_url)
        if blocked:
            logger.info(
                "Blocked Skills Hub fetch for %s by rule %s",
                blocked["host"],
                blocked["rule"],
            )
            return None

        try:
            resp = _ssrf_safe_http_get(current_url, timeout=timeout)
        except (SSRFConnectionBlocked, httpx.HTTPError) as exc:
            logger.debug("Skills Hub fetch failed for %s: %s", current_url, exc)
            return None

        if resp.status_code in _REDIRECT_STATUS_CODES:
            location = getattr(resp, "headers", {}).get("location")
            if not location:
                return None
            current_url = urljoin(current_url, location)
            continue

        return resp

    logger.warning("Skills Hub fetch exceeded redirect limit for %s", url)
    return None


# ---------------------------------------------------------------------------
# Shared index cache (used by every adapter)
# ---------------------------------------------------------------------------

def _read_json_if_fresh(path: Path, ttl: float) -> Optional[Any]:
    """Parsed JSON from ``path`` when it exists and is younger than ``ttl`` seconds."""
    if not path.exists():
        return None
    try:
        if time.time() - path.stat().st_mtime > ttl:
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _read_index_cache(key: str) -> Optional[Any]:
    return _read_json_if_fresh(_index_cache_dir() / f"{key}.json", INDEX_CACHE_TTL)


def _write_index_cache(key: str, data: Any) -> None:
    index_cache_dir = _index_cache_dir()
    index_cache_dir.mkdir(parents=True, exist_ok=True)
    # Cache files hold unvetted community text (possible prompt injection);
    # a .ignore keeps ripgrep and .ignore-aware tools out of the hub dir.
    ignore_file = _hub_dir() / ".ignore"
    if not ignore_file.exists():
        try:
            ignore_file.write_text("# Exclude hub internals from search tools\n*\n", encoding="utf-8")
        except OSError:
            pass
    try:
        (index_cache_dir / f"{key}.json").write_text(
            json.dumps(data, ensure_ascii=False, default=str), encoding="utf-8"
        )
    except OSError as e:
        logger.debug("Could not write cache: %s", e)


# ---------------------------------------------------------------------------
# Hub state files: lock.json, taps.json, audit.log
# ---------------------------------------------------------------------------

class _JsonStateFile:
    """A JSON file under the hub dir with a fixed empty shape."""

    EMPTY: dict = {}

    def __init__(self, path: Optional[Path] = None):
        self.path = path if path is not None else self._default_path()

    def _default_path(self) -> Path:
        raise NotImplementedError

    def _read(self) -> dict:
        if not self.path.exists():
            return json.loads(json.dumps(self.EMPTY))
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return json.loads(json.dumps(self.EMPTY))

    def _write(self, data: dict, **dumps_kwargs) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, indent=2, **dumps_kwargs) + "\n", encoding="utf-8")


class HubLockFile(_JsonStateFile):
    """skills/.hub/lock.json — provenance of installed hub skills."""

    EMPTY = {"version": 1, "installed": {}}

    def _default_path(self) -> Path:
        return _lock_file()

    def load(self) -> dict:
        return self._read()

    def save(self, data: dict) -> None:
        self._write(data, ensure_ascii=False)

    def record_install(
        self,
        name: str,
        source: str,
        identifier: str,
        trust_level: str,
        scan_verdict: str,
        skill_hash: str,
        install_path: str,
        files: List[str],
        metadata: Optional[Dict[str, Any]] = None,
        scan_provenance: Optional[Dict[str, Any]] = None,
    ) -> None:
        # Validate name and install-path SHAPE at write time: a poisoned lock
        # entry is the precondition for the uninstall_skill rmtree-escape.
        safe_name = _validate_skill_name(name)
        safe_install_path = _normalize_lock_install_path(install_path, safe_name)
        data = self.load()
        now = datetime.now(timezone.utc).isoformat()
        data["installed"][safe_name] = {
            "source": source,
            "identifier": identifier,
            "trust_level": trust_level,
            "scan_verdict": scan_verdict,
            "content_hash": skill_hash,
            "install_path": safe_install_path,
            "files": files,
            "metadata": metadata or {},
            "scan_provenance": scan_provenance or {},
            "installed_at": now,
            "updated_at": now,
        }
        self.save(data)

    def record_uninstall(self, name: str) -> None:
        data = self.load()
        data["installed"].pop(name, None)
        self.save(data)

    def get_installed(self, name: str) -> Optional[dict]:
        return self.load()["installed"].get(name)

    def list_installed(self) -> List[dict]:
        return [{"name": name, **entry} for name, entry in self.load()["installed"].items()]


class TapsManager(_JsonStateFile):
    """skills/.hub/taps.json — custom GitHub repo sources."""

    EMPTY = {"taps": []}

    def _default_path(self) -> Path:
        return _taps_file()

    def load(self) -> List[dict]:
        return self._read().get("taps", [])

    def save(self, taps: List[dict]) -> None:
        self._write({"taps": taps})

    def add(self, repo: str, path: str = "skills/") -> bool:
        """Add a tap. Returns False if already exists."""
        taps = self.load()
        if any(t["repo"] == repo for t in taps):
            return False
        taps.append({"repo": repo, "path": path})
        self.save(taps)
        return True

    def remove(self, repo: str) -> bool:
        """Remove a tap by repo name. Returns False if not found."""
        taps = self.load()
        new_taps = [t for t in taps if t["repo"] != repo]
        if len(new_taps) == len(taps):
            return False
        self.save(new_taps)
        return True

    def list_taps(self) -> List[dict]:
        return self.load()


def append_audit_log(action: str, skill_name: str, source: str,
                     trust_level: str, verdict: str, extra: str = "") -> None:
    """Append one space-separated line to the audit log (best-effort)."""
    audit_log = _audit_log()
    audit_log.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    parts = [timestamp, action, skill_name, f"{source}:{trust_level}", verdict]
    if extra:
        parts.append(extra)
    try:
        with open(audit_log, "a", encoding="utf-8") as f:
            f.write(" ".join(parts) + "\n")
    except OSError as e:
        logger.debug("Could not write audit log: %s", e)


# ---------------------------------------------------------------------------
# Hub operations (high-level)
# ---------------------------------------------------------------------------

def ensure_hub_dirs() -> None:
    """Create the .hub directory structure if it doesn't exist."""
    _hub_dir().mkdir(parents=True, exist_ok=True)
    _quarantine_dir().mkdir(exist_ok=True)
    _index_cache_dir().mkdir(exist_ok=True)
    for path, initial in (
        (_lock_file(), json.dumps(HubLockFile.EMPTY) + "\n"),
        (_audit_log(), ""),
        (_taps_file(), json.dumps(TapsManager.EMPTY) + "\n"),
    ):
        if not path.exists():
            path.write_text(initial, encoding="utf-8")


# ---------------------------------------------------------------------------
# Hermes centralized index (data source for HermesIndexSource)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Source router + parallel search
# ---------------------------------------------------------------------------

