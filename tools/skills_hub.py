#!/usr/bin/env python3
"""
Skills Hub — hub state management and the public facade for the source adapters.

Library module (not an agent tool). Owns the hub paths, index cache, lock file,
taps, audit log, quarantine/install/uninstall, update checks, and the source
router; the adapters live in the ``tools.skills_hub_*`` siblings and are
re-exported here so ``from tools.skills_hub import X`` keeps working.

Used by hermes_cli/skills_hub.py for CLI commands and the /skills slash command.
"""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
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


def _path(name: str, default):
    """A test-injected real module attribute (patch.object/monkeypatch on
    SKILLS_DIR etc.) wins over live resolution."""
    forced = globals().get(name)
    return Path(forced) if forced is not None else default()


def _hermes_home() -> Path:
    return get_hermes_home()


def _skills_dir() -> Path:
    return _path("SKILLS_DIR", lambda: _hermes_home() / "skills")


def _hub_dir() -> Path:
    return _path("HUB_DIR", lambda: _skills_dir() / ".hub")


def _lock_file() -> Path:
    return _path("LOCK_FILE", lambda: _hub_dir() / "lock.json")


def _quarantine_dir() -> Path:
    return _path("QUARANTINE_DIR", lambda: _hub_dir() / "quarantine")


def _audit_log() -> Path:
    return _path("AUDIT_LOG", lambda: _hub_dir() / "audit.log")


def _taps_file() -> Path:
    return _path("TAPS_FILE", lambda: _hub_dir() / "taps.json")


def _index_cache_dir() -> Path:
    return _path("INDEX_CACHE_DIR", lambda: _hub_dir() / "index-cache")


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
        (_lock_file(), '{"version": 1, "installed": {}}\n'),
        (_audit_log(), ""),
        (_taps_file(), '{"taps": []}\n'),
    ):
        if not path.exists():
            path.write_text(initial, encoding="utf-8")


# ---------------------------------------------------------------------------
# Hermes centralized index (data source for HermesIndexSource)
# ---------------------------------------------------------------------------

HERMES_INDEX_URL = "https://hermes-agent.nousresearch.com/docs/api/skills-index.json"
HERMES_INDEX_TTL = 6 * 3600  # 6 hours


def _hermes_index_cache_file() -> Path:
    return _index_cache_dir() / "hermes-index.json"


def _load_hermes_index() -> Optional[dict]:
    """Fetch the centralized skills index (docs site, rebuilt daily), cached
    locally for HERMES_INDEX_TTL; on any failure serve the stale cache.

    Brotli is deliberately NOT negotiated: the index is tens of MB and httpx's
    streaming Brotli decoder (brotlicffi, pinned for Discord attachments) raises
    DecodingError on payloads this size — which surfaced as a silently empty
    Skills Hub. gzip/deflate first; the identity retry covers proxies that
    ignore the header and return Brotli anyway.
    """
    cache_file = _hermes_index_cache_file()
    cached = _read_json_if_fresh(cache_file, HERMES_INDEX_TTL)
    if cached is not None:
        return cached

    data = None
    for accept_encoding in ("gzip, deflate", "identity"):
        try:
            resp = httpx.get(
                HERMES_INDEX_URL,
                timeout=15,
                follow_redirects=True,
                headers={"Accept-Encoding": accept_encoding},
            )
            if resp.status_code != 200:
                logger.debug("Hermes index fetch returned %d", resp.status_code)
                return _load_stale_index_cache()
            data = resp.json()
            break
        except httpx.DecodingError as e:
            logger.debug(
                "Hermes index decode failed (Accept-Encoding=%s): %s",
                accept_encoding,
                e,
            )
            continue
        except (httpx.HTTPError, json.JSONDecodeError) as e:
            logger.debug("Hermes index fetch failed: %s", e)
            return _load_stale_index_cache()

    if not isinstance(data, dict) or "skills" not in data:
        return _load_stale_index_cache()

    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(data), encoding="utf-8")
    except OSError:
        pass

    return data


def _load_stale_index_cache() -> Optional[dict]:
    """Fall back to the cache regardless of age when the network fetch fails."""
    return _read_json_if_fresh(_hermes_index_cache_file(), float("inf"))


# ---------------------------------------------------------------------------
# Source router + parallel search
# ---------------------------------------------------------------------------

# External API sources the centralized index already covers; skipped when the
# index is available and no source filter is active (~70 GitHub calls/search
# for unauthenticated users otherwise).
_API_SOURCE_IDS = frozenset({"github", "skills-sh", "clawhub", "lobehub", "well-known"})


def create_source_router(auth: Optional[GitHubAuth] = None) -> List[SkillSource]:
    """All configured source adapters, in priority order."""
    if auth is None:
        auth = GitHubAuth()

    return [
        OptionalSkillSource(auth=auth),   # official optional skills (highest priority)
        HermesIndexSource(auth=auth),     # centralized index (search + resolved install paths)
        SkillsShSource(auth=auth),
        WellKnownSkillSource(),
        UrlSource(),                      # direct HTTP(S) URL to a SKILL.md
        GitHubSource(auth=auth, extra_taps=TapsManager().list_taps()),
        ClawHubSource(),
        LobeHubSource(),
        BrowseShSource(),                 # browse.sh site-specific browser skills
    ]


def _search_one_source(
    src: SkillSource, query: str, limit: int
) -> Tuple[str, List[SkillMeta]]:
    """Search a single source.  Runs in a thread for parallelism."""
    try:
        return src.source_id(), src.search(query, limit=limit)
    except Exception as e:
        logger.debug("Search failed for %s: %s", src.source_id(), e)
        return src.source_id(), []


def _select_active_sources(sources: List[SkillSource], source_filter: str) -> List[SkillSource]:
    """Sources to query for ``source_filter``.

    A provider filter (nvidia/openai/...) is not a source id — the data lives
    in the index/github source under ``extra.provider`` — so it selects like
    "all"; the narrowing happens later on the merged results. "official" is
    always included alongside an explicit source filter.
    """
    effective = "all" if source_filter.strip().lower() in _PROVIDER_FILTER_VALUES else source_filter
    index_available = effective == "all" and any(
        src.source_id() == "hermes-index" and getattr(src, "is_available", False)
        for src in sources
    )
    active: List[SkillSource] = []
    for src in sources:
        sid = src.source_id()
        if effective != "all" and sid != effective and sid != "official":
            continue
        if index_available and sid in _API_SOURCE_IDS:
            continue
        active.append(src)
    return active


def parallel_search_sources(
    sources: List[SkillSource],
    query: str = "",
    per_source_limits: Optional[Dict[str, int]] = None,
    source_filter: str = "all",
    overall_timeout: float = 30,
    on_source_done: Optional[Any] = None,
) -> Tuple[List[SkillMeta], Dict[str, int], List[str]]:
    """Search all sources in parallel with an overall timeout.

    Returns ``(all_results, source_counts, timed_out_ids)``. *on_source_done*
    is an optional ``(source_id, count) -> None`` progress callback.
    """
    from concurrent.futures import as_completed

    per_source_limits = per_source_limits or {}
    active = _select_active_sources(sources, source_filter)

    all_results: List[SkillMeta] = []
    source_counts: Dict[str, int] = {}
    timed_out_ids: List[str] = []

    if not active:
        return all_results, source_counts, timed_out_ids

    # Not a ``with`` block: its shutdown(wait=True) would block on a slow source
    # (ClawHub) for minutes and defeat ``overall_timeout``. Daemon workers so an
    # abandoned source cannot block interpreter exit either.
    from tools.daemon_pool import DaemonThreadPoolExecutor
    pool = DaemonThreadPoolExecutor(max_workers=min(len(active), 8))
    futures = {
        pool.submit(_search_one_source, src, query, per_source_limits.get(src.source_id(), 50)): src.source_id()
        for src in active
    }

    try:
        try:
            for fut in as_completed(futures, timeout=overall_timeout):
                try:
                    sid, results = fut.result(timeout=0)
                    source_counts[sid] = len(results)
                    all_results.extend(results)
                    if on_source_done:
                        on_source_done(sid, len(results))
                except Exception:
                    pass
        except TimeoutError:
            timed_out_ids = [futures[f] for f in futures if not f.done()]
            if timed_out_ids:
                logger.debug(
                    "Skills browse timed out waiting for: %s",
                    ", ".join(timed_out_ids),
                )
    finally:
        pool.shutdown(wait=False, cancel_futures=True)

    return all_results, source_counts, timed_out_ids


def unified_search(query: str, sources: List[SkillSource],
                   source_filter: str = "all", limit: int = 10) -> List[SkillMeta]:
    """Search all sources (in parallel) and merge results."""
    all_results, _, _ = parallel_search_sources(
        sources,
        query=query,
        source_filter=source_filter,
        overall_timeout=30,
    )

    # Provider filters target ``extra.provider`` on the merged set, not a source id.
    if source_filter.strip().lower() in _PROVIDER_FILTER_VALUES:
        all_results = _filter_results_by_provider(all_results, source_filter)

    deduped = _dedupe_by_trust(all_results)
    # Stable-sort by trust before truncating so the limit cut never drops a
    # builtin/official entry because a high-volume community source finished
    # first; insertion order is preserved within each rank.
    deduped.sort(key=lambda r: -TRUST_RANK.get(r.trust_level, 0))
    return deduped[:limit]
