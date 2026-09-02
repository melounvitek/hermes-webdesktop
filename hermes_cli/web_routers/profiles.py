"""Profiles dashboard routes.

Two routers because route order matters: ``sessions_router``
(/api/profiles/sessions*, projects/tree, pull-requests) was registered long
before the generic ``/api/profiles/{name}`` routes on ``router``; the original
global registration order is preserved rather than relying on Starlette's
literal-before-param matching.

web_server-owned helpers are reached via the late-binding seam in
:mod:`hermes_cli.web_deps` so tests that ``monkeypatch.setattr(web_server,
"_helper", ...)`` keep working.
"""

import contextlib
import copy
import functools
import inspect
import json
import logging
import re
import subprocess
import sys
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, HTTPException, Query

from hermes_cli.web_deps import late
from hermes_cli.web_models import (
    ProfileCreate,
    ProfileActiveUpdate,
    ProfileExport,
    ProfileImport,
    ProfileRename,
    ProfileSoulUpdate,
    ProfileDescriptionUpdate,
    ProfileModelUpdate,
    ProfileDescribeAuto,
    SessionPrScanBody,
)

# Same logger the handlers used before extraction (identical logger object).
_log = logging.getLogger("hermes_cli.web_server")

# Per-profile session reads report failures in the response's ``errors``
# array, which the desktop sidebar does not surface — an empty sidebar can
# look healthy while nothing logs. Warn once per (profile, message) per
# process so a persistent failure is loud in errors.log without turning every
# sidebar poll into log spam.
_profile_read_warned: set = set()


def _warn_profile_read_error(profile: str, exc: Exception) -> None:
    key = (profile, str(exc))
    if key in _profile_read_warned:
        return
    _profile_read_warned.add(key)
    _log.warning(
        "profile session read failed for %r (reported only in the response "
        "errors array): %s", profile, exc,
    )

sessions_router = APIRouter()
router = APIRouter()

# Late-bound web_server helpers (resolved at call time; cycle-safe,
# monkeypatch-transparent).
_cron_profile_home = late("_cron_profile_home")
_fallback_profile_dicts = late("_fallback_profile_dicts")
_hub_action_name = late("_hub_action_name")
_open_session_db_at_path = late("_open_session_db_at_path")
_resolve_profile_dir = late("_resolve_profile_dir")
_spawn_hermes_action = late("_spawn_hermes_action")
run_in_threadpool = late("run_in_threadpool")
_strip_session_list_rows = late("_strip_session_list_rows")
_write_profile_mcp_servers = late("_write_profile_mcp_servers")
_apply_main_model_assignment = late("_apply_main_model_assignment")
_normalize_main_model_assignment = late("_normalize_main_model_assignment")


# ---------------------------------------------------------------------------
# Profile management endpoints (minimal — list/create/rename/delete + SOUL.md)
# ---------------------------------------------------------------------------


def _profile_attr(info, name: str, default: Any = None) -> Any:
    try:
        return getattr(info, name)
    except Exception:
        return default


def _profile_to_dict(info) -> Dict[str, Any]:
    return {
        "name": _profile_attr(info, "name", ""),
        "path": str(_profile_attr(info, "path", "")),
        "is_default": bool(_profile_attr(info, "is_default", False)),
        "model": _profile_attr(info, "model"),
        "provider": _profile_attr(info, "provider"),
        "has_env": bool(_profile_attr(info, "has_env", False)),
        "skill_count": int(_profile_attr(info, "skill_count", 0) or 0),
        "gateway_running": bool(_profile_attr(info, "gateway_running", False)),
        "description": _profile_attr(info, "description", "") or "",
        "description_auto": bool(_profile_attr(info, "description_auto", False)),
        "display_name": _profile_attr(info, "display_name", "") or "",
        "distribution_name": _profile_attr(info, "distribution_name"),
        "distribution_version": _profile_attr(info, "distribution_version"),
        "distribution_source": _profile_attr(info, "distribution_source"),
        "has_alias": _profile_attr(info, "alias_path") is not None,
    }


def _profile_setup_command(name: str) -> str:
    """Return the shell command used to configure a profile in the CLI."""
    _resolve_profile_dir(name)
    return "hermes setup" if name == "default" else f"{name} setup"


def _write_profile_model(profile_dir: Path, provider: str, model: str) -> None:
    """Write the main model assignment into a specific profile's config.yaml.

    Scopes ``load_config``/``save_config`` to ``profile_dir`` via the
    context-local HERMES_HOME override so the write lands in the target
    profile's config rather than the dashboard process's active profile.
    Clears any stale ``base_url`` / ``context_length`` the same way
    ``POST /api/model/set`` does, since the new model may differ.
    """
    from hermes_cli.web_server import load_config, save_config
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override

    token = set_hermes_home_override(str(profile_dir))
    try:
        provider, model = _normalize_main_model_assignment(provider, model)
        cfg = load_config()
        cfg["model"] = _apply_main_model_assignment(cfg.get("model", {}), provider, model)
        save_config(cfg)
    finally:
        reset_hermes_home_override(token)


def _disable_unselected_skills(profile_dir: Path, keep: List[str]) -> int:
    """Disable every installed skill in ``profile_dir`` not in ``keep``.

    Profiles manage skill activation via a *disabled* list — all installed
    skills are active by default and users opt out. The builder's skill step
    uses "replace" semantics: the user picks exactly which seeded built-in /
    optional skills stay active, and everything else gets added to the disabled
    list. (Hub skills are installed separately via subprocess and are active on
    install.) Scoped to the profile via the HERMES_HOME override. Returns the
    number of skills newly disabled.
    """
    from hermes_cli.web_server import load_config
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    from hermes_cli.skills_config import get_disabled_skills, save_disabled_skills

    keep_set = {s.strip() for s in keep if s and s.strip()}
    disabled_count = 0
    token = set_hermes_home_override(str(profile_dir))
    try:
        installed: List[str] = []
        skills_root = profile_dir / "skills"
        if skills_root.is_dir():
            for md in skills_root.rglob("SKILL.md"):
                installed.append(md.parent.name)
        cfg = load_config()
        disabled = get_disabled_skills(cfg)
        for name in installed:
            if name not in keep_set and name not in disabled:
                disabled.add(name)
                disabled_count += 1
        if disabled_count:
            save_disabled_skills(cfg, disabled)
    finally:
        reset_hermes_home_override(token)
    return disabled_count

# Returned by the offloaded file readers below to mean "the file is not there",
# which a plain ``None`` cannot express: ``desktop.json`` may legitimately hold
# the document ``null``, and that is an existing-but-empty overlay rather than
# an absent one.
_MISSING = object()


@contextlib.contextmanager
def _profile_errors(log_msg: str, *args, not_found=(FileNotFoundError,),
                    bad_request=(ValueError,)):
    """Map hermes_cli.profiles exceptions to HTTP: ``not_found`` -> 404,
    ``bad_request`` -> 400 (checked in that order), anything else is logged
    with ``log_msg`` and becomes a 500. ``HTTPException`` passes through."""
    try:
        yield
    except HTTPException:
        raise
    except not_found as e:
        raise HTTPException(status_code=404, detail=str(e))
    except bad_request as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        _log.exception(log_msg, *args)
        raise HTTPException(status_code=500, detail=str(e))


def _profile_targets(log_label: str, *, lightweight: bool) -> List[Tuple[str, Path]]:
    """(name, home) for every profile, falling back to ``default`` alone.

    ``lightweight`` uses ``profiles_to_serve`` (name/path only) instead of
    ``list_profiles``, which parses config/meta and probes gateways/skills
    per profile — too heavy for every sidebar refresh."""
    from hermes_cli import profiles as profiles_mod

    try:
        if lightweight:
            targets = list(profiles_mod.profiles_to_serve(multiplex=True))
        else:
            targets = [(info.name, info.path) for info in profiles_mod.list_profiles()]
    except Exception:
        _log.exception("%s: list_profiles failed", log_label)
        targets = []
    if not targets:
        targets.append(("default", profiles_mod.get_profile_dir("default")))
    return targets


def _tag_rows(rows: List[Dict[str, Any]], name: str, now: float) -> List[Dict[str, Any]]:
    """Stamp session rows with their owning profile and the 300s active
    heuristic; SQLite stores flags as 0/1 and the sidebar needs booleans."""
    for s in rows:
        s["profile"] = name
        s["is_default_profile"] = name == "default"
        s["is_active"] = (
            s.get("ended_at") is None
            and (now - s.get("last_active", s.get("started_at", 0))) < 300
        )
        s["archived"] = bool(s.get("archived"))
        s["pinned"] = bool(s.get("pinned"))
    return rows


def _pinned_window(rows: List[Dict[str, Any]], offset: int, cap: int) -> List[Dict[str, Any]]:
    """``rows[offset:offset+cap]`` plus every pinned row past it. The
    per-profile queries deliberately back-fill pinned rows past their LIMIT,
    so truncating on recency alone would drop exactly what they fetched."""
    window = rows[offset:offset + cap]
    if len(rows) > offset + cap:
        seen = {id(s) for s in window}
        window.extend(s for s in rows[offset + cap:] if s.get("pinned") and id(s) not in seen)
    return window


def _recency(s: Dict[str, Any]) -> Any:
    return s.get("last_active") or s.get("started_at") or 0


def _open_profile_db(name: str, home, errors: Optional[List[Dict[str, str]]]):
    """Read-only SessionDB for a profile's state.db, or None when the file is
    missing or the open fails (failure recorded in ``errors`` when given).

    Read-only on the healthy path: this runs on every sidebar refresh, so it
    must not routinely DDL/write-lock another profile's live DB. The helper's
    stale-schema probe performs a ONE-TIME writable open when the store
    predates a schema addition — read-only opens skip column reconciliation
    and would otherwise fail here on every refresh."""
    db_path = Path(home) / "state.db"
    if not db_path.exists():
        return None
    try:
        return _open_session_db_at_path(db_path, read_only=True)
    except Exception as exc:
        _warn_profile_read_error(name, exc)
        if errors is not None:
            errors.append({"profile": name, "error": str(exc)})
        return None


# Bounded cache lifetime for the expensive sidebar scan. Short enough that the
# UI never shows meaningfully stale data, long enough to coalesce the desktop's
# reconnect/focus/change poll bursts into one scan.
_SIDEBAR_CACHE_TTL_SECONDS = 5.0
_SIDEBAR_CACHE_MAX_ENTRIES = 32
_SIDEBAR_PROFILE_CACHE_MAX_ENTRIES = 256
_SIDEBAR_PROFILE_CACHE = OrderedDict()
_SIDEBAR_PROFILE_CACHE_LOCK = threading.Lock()


def _stat_fingerprint(path: Path):
    """Return identity + mutation metadata without opening the file."""
    try:
        stat = path.stat()
    except OSError:
        return None
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)


def _sidebar_db_fingerprint(db_path: Path):
    """Track SQLite content changes through the main DB and its WAL."""
    return (_stat_fingerprint(db_path), _stat_fingerprint(Path(f"{db_path}-wal")))


def _sidebar_profile_cache_get(key):
    with _SIDEBAR_PROFILE_CACHE_LOCK:
        value = _SIDEBAR_PROFILE_CACHE.get(key)
        if value is None:
            return None
        _SIDEBAR_PROFILE_CACHE.move_to_end(key)
        return copy.deepcopy(value)


def _sidebar_profile_cache_put(key, value):
    db_path, fingerprint = key[:2]
    snapshot = copy.deepcopy(value)
    with _SIDEBAR_PROFILE_CACHE_LOCK:
        # A changed DB/WAL makes all older parameter variants for that profile
        # obsolete. Remove them eagerly rather than waiting for LRU pressure.
        for existing in [k for k in _SIDEBAR_PROFILE_CACHE
                         if k[0] == db_path and k[1] != fingerprint]:
            _SIDEBAR_PROFILE_CACHE.pop(existing, None)
        _SIDEBAR_PROFILE_CACHE[key] = snapshot
        _SIDEBAR_PROFILE_CACHE.move_to_end(key)
        while len(_SIDEBAR_PROFILE_CACHE) > _SIDEBAR_PROFILE_CACHE_MAX_ENTRIES:
            _SIDEBAR_PROFILE_CACHE.popitem(last=False)


def _sidebar_profile_cache_clear():
    with _SIDEBAR_PROFILE_CACHE_LOCK:
        _SIDEBAR_PROFILE_CACHE.clear()


def _sidebar_singleflight_cache(func):
    """Coalesce concurrent sidebar scans and briefly reuse their response.

    Every uncached refresh opens every profile database and runs several
    session queries per profile. Desktop reconnect/focus/change bursts overlap
    identical scans in AnyIO worker threads, amplifying YAML/SQLite work and
    starving the uvicorn event loop for the GIL. The short TTL bounds UI
    staleness; the single-flight lock guarantees one expensive scan at a time.
    Cached values are copied on store and hit so FastAPI serialization or a
    caller cannot mutate shared state.
    """
    signature = inspect.signature(func)
    cache = OrderedDict()
    cache_lock = threading.Lock()
    refresh_lock = threading.Lock()
    miss = object()

    def _key(args, kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        return tuple(bound.arguments.items())

    def _lookup(key):
        now = time.monotonic()
        with cache_lock:
            item = cache.get(key)
            if item is None:
                return miss
            expires_at, value = item
            if now >= expires_at:
                cache.pop(key, None)
                return miss
            cache.move_to_end(key)
            return copy.deepcopy(value)

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        ttl = _SIDEBAR_CACHE_TTL_SECONDS
        if ttl <= 0:
            return func(*args, **kwargs)

        key = _key(args, kwargs)
        cached = _lookup(key)
        if cached is not miss:
            return cached

        # A plain Lock is intentional: FastAPI executes this sync handler in
        # the AnyIO worker pool, so contenders sleep without holding the GIL.
        with refresh_lock:
            cached = _lookup(key)
            if cached is not miss:
                return cached
            result = func(*args, **kwargs)
            # A 200 carrying errors[] is a FAILED profile scan, not a successful
            # empty page. Caching it would hold the empty recents in front of a
            # store that has already recovered, for the whole TTL.
            if isinstance(result, dict) and result.get("errors"):
                return result
            try:
                snapshot = copy.deepcopy(result)
            except Exception:
                _log.exception("sidebar response could not be cached")
                return result
            with cache_lock:
                cache[key] = (time.monotonic() + ttl, snapshot)
                cache.move_to_end(key)
                while len(cache) > _SIDEBAR_CACHE_MAX_ENTRIES:
                    cache.popitem(last=False)
            return result

    def cache_clear():
        with cache_lock:
            cache.clear()

    wrapped.cache_clear = cache_clear
    return wrapped


@sessions_router.get("/api/profiles/sessions")
def get_profiles_sessions(
    # ``le=500`` caps the per-request page size — this endpoint fans out across
    # EVERY profile's state.db, so an unbounded limit multiplies the damage.
    # 500 (not 100) because real desktop callers use limit=200 and the electron
    # remote-merge over-fetches ``limit + offset``.
    limit: int = Query(20, ge=0, le=500),
    offset: int = Query(0, ge=0),
    min_messages: int = 0,
    archived: str = "exclude",
    order: str = "recent",
    profile: str = "all",
    source: str = None,
    sources: str = None,
    exclude_sources: str = None,
    full: bool = False,
):
    """Unified, read-only session list aggregated across ALL profiles.

    Process-light: opens each profile's ``state.db`` directly from disk — it
    does NOT spawn a dashboard backend per profile. Each row is tagged with its
    owning ``profile`` so the desktop renders one list and only spins up a
    backend when the user interacts. Rows omit ``system_prompt`` /
    ``model_config`` unless ``full=1`` — same projection as ``/api/sessions``.
    """
    if archived not in ("exclude", "only", "include"):
        raise HTTPException(status_code=400, detail="archived must be one of: exclude, only, include")
    if order not in ("created", "recent"):
        raise HTTPException(status_code=400, detail="order must be one of: created, recent")

    if profile and profile != "all":
        targets = [_cron_profile_home(profile)]
    else:
        targets = _profile_targets("GET /api/profiles/sessions", lightweight=True)

    min_message_count = max(0, min_messages)
    archived_only = archived == "only"
    include_archived = archived == "include"
    # Source scoping (see /api/sessions): recents pass exclude_sources=cron,
    # the cron-jobs section passes source=cron — two independent lists so
    # newest cron sessions can't starve the recents page.
    source_filter = source or None
    source_list = [s.strip() for s in (sources or "").split(",") if s.strip()]
    exclude_list = [s.strip() for s in (exclude_sources or "").split(",") if s.strip()]
    # Over-fetch per profile so the merged+sorted window is correct for the
    # requested page. Capped so a huge profile can't blow up the response.
    per_profile = min(max(limit + offset, limit), 500)

    merged: List[Dict[str, Any]] = []
    total = 0
    profile_totals: Dict[str, int] = {}
    errors: List[Dict[str, str]] = []
    now = time.time()
    for name, home in targets:
        db = _open_profile_db(name, home, errors)
        if db is None:
            continue
        try:
            rows = db.list_sessions_rich(
                source=source_filter,
                sources=source_list or None,
                exclude_sources=exclude_list or None,
                limit=per_profile,
                offset=0,
                min_message_count=min_message_count,
                include_archived=include_archived,
                archived_only=archived_only,
                order_by_last_active=order == "recent",
                # Same SQL-level blob skip as /api/sessions.
                compact_rows=not full,
                include_pinned=True,
            )
            profile_total = db.session_count(
                source=source_filter,
                sources=source_list or None,
                exclude_sources=exclude_list or None,
                min_message_count=min_message_count,
                include_archived=include_archived,
                archived_only=archived_only,
                exclude_children=True,
            )
            total += profile_total
            profile_totals[name] = profile_total
            merged.extend(_tag_rows(rows, name, now))
        except Exception as exc:
            _warn_profile_read_error(name, exc)
            errors.append({"profile": name, "error": str(exc)})
        finally:
            db.close()

    sort_key = "last_active" if order == "recent" else "started_at"
    merged.sort(key=lambda s: s.get(sort_key) or s.get("started_at") or 0, reverse=True)
    window = _pinned_window(merged, offset, limit)
    if not full:
        _strip_session_list_rows(window)
    return {
        "sessions": window,
        "total": total,
        "profile_totals": profile_totals,
        "limit": limit,
        "offset": offset,
        "errors": errors,
    }


@sessions_router.get("/api/profiles/sessions/sidebar")
@_sidebar_singleflight_cache
def get_profiles_sessions_sidebar(
    recents_profile: str = "all",
    recents_limit: int = 20,
    recents_exclude: str = None,
    cron_limit: int = 50,
    messaging_limit: int = 100,
    messaging_exclude: str = None,
):
    """Batched sidebar session slices — one profile-DB open per refresh.

    The desktop sidebar needs three source-scoped windows per refresh: recents
    (local chats), cron sessions, and messaging-platform sessions. Served as
    three ``/api/profiles/sessions`` calls they reopened every profile's DB
    three times; this opens each once and runs the three queries together.
    Same row projection and 300s active heuristic as the per-slice endpoint.

    ``recents_profile`` scopes the WHOLE payload, not just recents — the
    sidebar has one scope, so a concrete profile must never show another
    profile's Telegram threads or cronjobs; ``all`` asks for everything.

    The caller passes the source taxonomy (``recents_exclude`` /
    ``messaging_exclude`` CSV, ``source=cron`` is implicit). All slices use
    ``min_messages=1`` / ``archived=exclude`` / recency order.
    """
    targets = _profile_targets("GET /api/profiles/sessions/sidebar", lightweight=True)

    recents_scope = (recents_profile or "all").strip() or "all"
    recents_exclude_list = [s for s in (recents_exclude or "").split(",") if s.strip()]
    messaging_exclude_list = [s for s in (messaging_exclude or "").split(",") if s.strip()]

    recents_cap = min(max(recents_limit, 1), 500)
    cron_cap = min(max(cron_limit, 1), 500)
    messaging_cap = min(max(messaging_limit, 1), 500)

    recents_rows: List[Dict[str, Any]] = []
    cron_rows: List[Dict[str, Any]] = []
    messaging_rows: List[Dict[str, Any]] = []
    recents_truncated: Dict[str, bool] = {}
    profile_totals: Dict[str, Dict[str, float]] = {}
    errors: List[Dict[str, str]] = []
    now = time.time()

    def _slice(db, *, source=None, exclude=None, cap):
        return db.list_sessions_rich(
            source=source,
            exclude_sources=exclude or None,
            limit=cap,
            offset=0,
            min_message_count=1,
            include_archived=False,
            archived_only=False,
            order_by_last_active=True,
            compact_rows=True,
            # A pinned conversation must reach the sidebar even when it has
            # aged past the window — otherwise its Pinned row renders empty.
            include_pinned=True,
        )

    for name, home in targets:
        if recents_scope != "all" and name != recents_scope:
            continue
        db_path = Path(home) / "state.db"
        if not db_path.exists():
            continue
        profile_cache_key = (
            str(db_path),
            _sidebar_db_fingerprint(db_path),
            recents_cap,
            tuple(recents_exclude_list),
            cron_cap,
            messaging_cap,
            tuple(messaging_exclude_list),
        )
        slices = _sidebar_profile_cache_get(profile_cache_key)
        if slices is None:
            db = _open_profile_db(name, home, errors)
            if db is None:
                continue
            try:
                slices = {
                    "recents": _slice(db, exclude=recents_exclude_list, cap=recents_cap),
                    # Aggregated in SQL rather than over the recents window: the
                    # window is a page, and a total that shrank when you scrolled
                    # would be worse than no total at all.
                    "usage": db.usage_totals(),
                    "cron": _slice(db, source="cron", cap=cron_cap),
                    "messaging": _slice(db, exclude=messaging_exclude_list, cap=messaging_cap),
                }
                _sidebar_profile_cache_put(profile_cache_key, slices)
            except Exception as exc:
                _warn_profile_read_error(name, exc)
                errors.append({"profile": name, "error": str(exc)})
                continue
            finally:
                db.close()

        profile_rows = slices["recents"]
        # A full window means more rows remain on disk — all "load more" needs,
        # at no cost beyond the rows already read. Discount pinned back-fills:
        # they arrive past the LIMIT and would fake a full page on a short list.
        unpinned_count = sum(1 for s in profile_rows if not s.get("pinned"))
        recents_truncated[name] = unpinned_count >= recents_cap
        recents_rows.extend(_tag_rows(profile_rows, name, now))
        profile_totals[name] = slices["usage"]
        cron_rows.extend(_tag_rows(slices["cron"], name, now))
        messaging_rows.extend(_tag_rows(slices["messaging"], name, now))

    def _window(rows: List[Dict[str, Any]], cap: int) -> List[Dict[str, Any]]:
        rows.sort(key=_recency, reverse=True)
        win = _pinned_window(rows, 0, cap)
        _strip_session_list_rows(win)
        return win

    return {
        "recents": {
            "sessions": _window(recents_rows, recents_cap),
            "profiles_truncated": recents_truncated,
            "profiles_usage": profile_totals,
        },
        "cron": {"sessions": _window(cron_rows, cron_cap)},
        "messaging": {
            "sessions": _window(messaging_rows, messaging_cap),
            "total": len(messaging_rows),
        },
        "errors": errors,
    }


def _merge_by_id(into: Dict[str, Dict[str, Any]], entries: List[Dict[str, Any]], child_key: str) -> None:
    """Fold ``entries`` into ``into`` by id, recursing through one child list.

    Repos merge their lanes, lanes merge their sessions. Counts add up and the
    newest activity wins; everything else is first-writer, since the entries
    describe the same path either way.
    """
    for entry in entries:
        existing = into.get(entry["id"])
        if existing is None:
            into[entry["id"]] = entry
            continue
        if child_key == "sessions":
            existing["sessions"].extend(entry.get("sessions") or [])
        else:
            children: Dict[str, Dict[str, Any]] = {c["id"]: c for c in existing.get(child_key) or []}
            _merge_by_id(children, entry.get(child_key) or [], "sessions")
            existing[child_key] = list(children.values())
        if "sessionCount" in existing:
            existing["sessionCount"] = (existing.get("sessionCount") or 0) + (entry.get("sessionCount") or 0)


def _merge_profile_tree(
    merged: Dict[str, Dict[str, Any]],
    projects: List[Dict[str, Any]],
    profile: str,
    preview_limit: int,
) -> None:
    """Fold one profile's projects into the shared tree, keyed by folder.

    The same checkout in two profiles is one group, as is ``__no_project__``
    (every profile has one, which would otherwise put a "Home" on screen per
    profile). Keying on the path also folds a declared project (``p_<hash>``)
    with the auto entry another profile grows for the same folder. Sessions
    carry the owning profile — the row badge and profile filter read that; a
    group header never claims a single owner.
    """
    for project in projects:
        lane_sessions = (s for r in project.get("repos") or []
                         for lane in r.get("groups") or []
                         for s in lane.get("sessions") or [])
        for session in [*lane_sessions, *(project.get("previewSessions") or [])]:
            session["profile"] = profile
            session["is_default_profile"] = profile == "default"

        key = project.get("path") or project["id"]
        existing = merged.get(key)
        if existing is None:
            merged[key] = project
            continue

        # A declared project carries the label, color and icon the user chose,
        # so it wins the identity when it meets another profile's auto entry.
        if existing.get("isAuto") and not project.get("isAuto"):
            existing, project = project, existing
            merged[key] = existing

        repos: Dict[str, Dict[str, Any]] = {r["id"]: r for r in existing.get("repos") or []}
        _merge_by_id(repos, project.get("repos") or [], "groups")
        existing["repos"] = list(repos.values())
        for total_key in ("sessionCount", "totalTokens", "totalCostUsd"):
            existing[total_key] = (existing.get(total_key) or 0) + (project.get(total_key) or 0)
        existing["lastActive"] = max(existing.get("lastActive") or 0, project.get("lastActive") or 0)
        previews = (existing.get("previewSessions") or []) + (project.get("previewSessions") or [])
        previews.sort(key=_recency, reverse=True)
        existing["previewSessions"] = previews[:preview_limit]


@sessions_router.get("/api/profiles/projects/tree")
def get_profiles_projects_tree(preview_limit: int = 3, session_limit: int = 2000):
    """Project tree for every profile at once, for the all-profiles sidebar.

    ``projects.tree`` over JSON-RPC answers for the backend's own profile only.
    This runs the same authoritative builder once per profile against that
    profile's ``state.db``, scoping its other inputs (projects.db, repo-scan
    policy, HERMES_HOME junk filters) through the context-local home override.
    Projects merge across profiles so a group stands for a checkout rather than
    a checkout-and-owner; the profile shows up per row for the filter.

    Discovery is off: a repo with zero sessions is the same repo in every
    profile (the disk scan would multiply empty lanes by the profile count),
    and it is the one part of the builder that writes (policy reconciliation),
    which this read-only fan-out must not do to a profile the user isn't driving.
    """
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from tui_gateway import server as gateway_server

    merged: Dict[str, Dict[str, Any]] = {}
    scoped_session_ids: List[str] = []
    errors: List[Dict[str, str]] = []

    for name, home in _profile_targets("GET /api/profiles/projects/tree", lightweight=False):
        db = _open_profile_db(name, home, errors)
        if db is None:
            continue
        token = set_hermes_home_override(str(home))
        try:
            tree, _active_id = gateway_server._build_project_tree(
                db,
                preview_limit=preview_limit,
                hydrate=False,
                session_limit=session_limit,
                include_discovered=False,
            )
            _merge_profile_tree(merged, tree["projects"], name, preview_limit)
            scoped_session_ids.extend(tree["scoped_session_ids"])
        except Exception as exc:
            _warn_profile_read_error(name, exc)
            errors.append({"profile": name, "error": str(exc)})
        finally:
            reset_hermes_home_override(token)
            db.close()

    return {
        "projects": sorted(merged.values(), key=lambda p: p.get("lastActive") or 0, reverse=True),
        # Ownership is per profile, so no single project is "the active one"
        # here; the desktop only reads active_id to bias its overview sort.
        "active_id": None,
        "scoped_session_ids": scoped_session_ids,
        "errors": errors,
    }


# `gh pr create` prints the PR url and nothing else, so a tool result whose
# whole output IS a PR url means this session opened that PR. Anything looser —
# a url inside prose, a `gh pr view` payload, an issue link — is a session
# TALKING about a PR, which is not the same claim.
_PR_URL_RE = re.compile(r"^https://github\.com/[\w.-]+/[\w.-]+/pull/(\d+)/?$")


def _pr_url_from_tool_output(content: str) -> Optional[Tuple[int, str]]:
    """The (number, url) a tool result announces, or None."""
    try:
        output = (json.loads(content) or {}).get("output")
    except (json.JSONDecodeError, TypeError, AttributeError):
        return None
    if not isinstance(output, str):
        return None
    match = _PR_URL_RE.match(output.strip())
    return (int(match.group(1)), match.group(0)) if match else None


@sessions_router.post("/api/profiles/sessions/pull-requests")
def post_profiles_sessions_pull_requests(body: SessionPrScanBody):
    """The PR each of these sessions opened, recovered from its own transcript.

    A session records the branch it started on, but one that starts in the main
    checkout and works in a worktree has no branch of its own, so its PR is
    invisible to that join. The evidence is in the conversation: ``gh pr
    create`` ran and its output is a bare PR url (see
    ``_pr_url_from_tool_output``). Read-only across every profile; the caller
    asks once per session and remembers the answer.
    """
    wanted = list(dict.fromkeys(s for s in (body.ids or []) if s))[:2000]
    if not wanted:
        return {"pull_requests": {}, "scanned": []}

    found: Dict[str, Dict[str, Any]] = {}
    for name, home in _profile_targets("POST /api/profiles/sessions/pull-requests", lightweight=False):
        db = _open_profile_db(name, home, None)
        if db is None:
            continue
        try:
            for pr in db.find_pr_url_messages(wanted):
                parsed = _pr_url_from_tool_output(pr["content"])
                if parsed:
                    # Ordered oldest-first, so a later `gh pr create` in the same
                    # conversation wins — the replacement PR is the one the
                    # session ended on.
                    found[pr["session_id"]] = {"number": parsed[0], "url": parsed[1]}
        except Exception as exc:
            _warn_profile_read_error(name, exc)
        finally:
            db.close()

    # Every id we looked at, so the caller can remember "asked, nothing there"
    # and never scan this session again.
    return {"pull_requests": found, "scanned": wanted}


@router.get("/api/profiles")
async def list_profiles_endpoint():
    from hermes_cli import profiles as profiles_mod
    try:
        profiles = await run_in_threadpool(profiles_mod.list_profiles)
        return {"profiles": [_profile_to_dict(p) for p in profiles]}
    except Exception:
        _log.exception("GET /api/profiles failed; falling back to profile directory scan")
        return {"profiles": _fallback_profile_dicts(profiles_mod)}


@router.post("/api/profiles")
async def create_profile_endpoint(body: ProfileCreate):
    from hermes_cli import profiles as profiles_mod
    explicit_source = (body.clone_from or "").strip()
    if explicit_source:
        # Duplicating a specific profile: clone its config/skills/SOUL (or full
        # state when clone_all) from the named source rather than "default".
        clone, clone_from, clone_config = True, explicit_source, not body.clone_all
    elif body.clone_all:
        # Historical dashboard clone-all behavior: a full-copy request with no
        # explicit dropdown source copies from default.
        clone, clone_from, clone_config = True, "default", False
    else:
        clone = body.clone_from_default
        clone_from = "default" if clone else None
        clone_config = clone
    with _profile_errors("POST /api/profiles failed", not_found=(),
                         bad_request=(ValueError, FileExistsError, FileNotFoundError)):
        path = profiles_mod.create_profile(
            name=body.name,
            clone_from=clone_from,
            clone_all=body.clone_all,
            clone_config=clone_config,
            no_skills=body.no_skills,
            description=body.description,
        )
        # Match the CLI's profile-create flow: fresh named profiles get the
        # bundled skills. Cloning already copied the source's skills (incl.
        # user-installed); no_skills wrote the opt-out marker so seeding no-ops.
        if not clone:
            profiles_mod.seed_profile_skills(path, quiet=True)

        # Match the CLI: named profiles get a ~/.local/bin wrapper when the
        # alias is safe to create.
        if not profiles_mod.check_alias_collision(body.name):
            profiles_mod.create_wrapper_script(body.name)

    # Everything below is best-effort: the profile already exists, so a hiccup
    # must not 500 the whole create — the user can fix it from the relevant
    # dashboard page or `<profile> setup` afterward.
    provider = (body.provider or "").strip()
    model = (body.model or "").strip()
    model_set = False
    if provider and model:
        try:
            _write_profile_model(path, provider, model)
            model_set = True
        except Exception:
            _log.exception("Setting model for new profile %s failed", body.name)

    mcp_written = 0
    if body.mcp_servers:
        try:
            mcp_written = _write_profile_mcp_servers(path, body.mcp_servers)
        except Exception:
            _log.exception("Writing MCP servers for new profile %s failed", body.name)

    # "keep" skill selection has replace semantics: disable every seeded skill
    # not in the list. Skipped when empty (legacy: keep the bundle).
    skills_disabled = 0
    if body.keep_skills:
        try:
            skills_disabled = _disable_unselected_skills(path, body.keep_skills)
        except Exception:
            _log.exception("Applying skill selection for new profile %s failed", body.name)

    # Skills-hub installs are spawned async, scoped to the new profile via
    # `-p <name>` (a fresh subprocess re-binds skills_hub.SKILLS_DIR to the
    # profile's HERMES_HOME at import). PIDs go back for the UI to poll.
    hub_installs: List[Dict[str, Any]] = []
    for identifier in body.hub_skills:
        ident = (identifier or "").strip()
        if not ident:
            continue
        try:
            proc = _spawn_hermes_action(
                ["-p", body.name, "skills", "install", ident, "--yes"],
                _hub_action_name("install", ident),
            )
            hub_installs.append({"identifier": ident, "pid": proc.pid})
        except Exception:
            _log.exception(
                "Spawning hub-skill install %s for new profile %s failed",
                ident,
                body.name,
            )
            hub_installs.append({"identifier": ident, "pid": None})

    return {
        "ok": True,
        "name": body.name,
        "path": str(path),
        "model_set": model_set,
        "mcp_written": mcp_written,
        "skills_disabled": skills_disabled,
        "hub_installs": hub_installs,
    }


@router.get("/api/profiles/active")
async def get_active_profile_endpoint():
    """``active`` is the sticky default written by ``hermes profile use`` (what
    new CLI invocations pick up); ``current`` is the profile this running
    dashboard/gateway is scoped to (derived from HERMES_HOME)."""
    from hermes_cli import profiles as profiles_mod

    def _run():
        # Both reads touch the filesystem: get_active_profile() reads the
        # active_profile state file and get_active_profile_name() resolves
        # HERMES_HOME against the profiles root. Batched into one hop so the
        # sidebar's polling costs a single executor round-trip, not two.
        try:
            active = profiles_mod.get_active_profile() or "default"
        except Exception:
            active = "default"
        try:
            current = profiles_mod.get_active_profile_name() or "default"
        except Exception:
            current = "default"
        return {"active": active, "current": current}

    return await run_in_threadpool(_run)


@router.post("/api/profiles/active")
async def set_active_profile_endpoint(body: ProfileActiveUpdate):
    """Set the sticky active profile (mirrors ``hermes profile use``). Does not
    retarget the already-running dashboard — it changes which profile
    subsequent CLI commands and gateways use."""
    from hermes_cli import profiles as profiles_mod
    with _profile_errors("POST /api/profiles/active failed"):
        # set_active_profile() stats the target profile, creates the state
        # directory and writes active_profile through a temp file + replace.
        await run_in_threadpool(profiles_mod.set_active_profile, body.name)
    return {"ok": True, "active": profiles_mod.normalize_profile_name(body.name)}


@router.get("/api/profiles/{name}/setup-command")
async def get_profile_setup_command(name: str):
    return {"command": _profile_setup_command(name)}


def _linux_terminal_commands(command: str) -> list:
    sh = ["sh", "-lc", command]
    quoted = f"sh -lc '{command}'"
    return [
        ("x-terminal-emulator", ["x-terminal-emulator", "-e", *sh]),
        ("gnome-terminal", ["gnome-terminal", "--", *sh]),
        ("konsole", ["konsole", "-e", *sh]),
        ("xfce4-terminal", ["xfce4-terminal", "-e", quoted]),
        ("mate-terminal", ["mate-terminal", "-e", quoted]),
        ("lxterminal", ["lxterminal", "-e", quoted]),
        ("tilix", ["tilix", "-e", *sh]),
        ("alacritty", ["alacritty", "-e", *sh]),
        ("kitty", ["kitty", *sh]),
        ("xterm", ["xterm", "-e", *sh]),
    ]


@router.post("/api/profiles/{name}/open-terminal")
async def open_profile_terminal_endpoint(name: str):
    with _profile_errors("POST /api/profiles/%s/open-terminal failed", name):
        command = _profile_setup_command(name)

        if sys.platform.startswith("win"):
            subprocess.Popen(["cmd.exe", "/c", "start", "", command])
        elif sys.platform == "darwin":
            escaped = command.replace("\\", "\\\\").replace('"', '\\"')
            applescript = (
                'tell application "Terminal"\n'
                "activate\n"
                f'do script "{escaped}"\n'
                "end tell"
            )
            subprocess.Popen(["osascript", "-e", applescript])
        else:
            for executable, popen_args in _linux_terminal_commands(command):
                if subprocess.call(
                    ["which", executable],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                ) == 0:
                    subprocess.Popen(popen_args)
                    break
            else:
                raise HTTPException(
                    status_code=400,
                    detail="No supported terminal emulator found",
                )
    return {"ok": True, "command": command}


@router.patch("/api/profiles/{name}")
async def rename_profile_endpoint(name: str, body: ProfileRename):
    from hermes_cli import profiles as profiles_mod
    with _profile_errors("PATCH /api/profiles/%s failed", name,
                         bad_request=(ValueError, FileExistsError)):
        # rename_profile() stops a running gateway through the same 10-second
        # _stop_gateway_process() poll that delete does, then renames the
        # profile directory, rewrites the Honcho host blocks and regenerates
        # the wrapper script.
        path = await run_in_threadpool(profiles_mod.rename_profile, name, body.new_name)
    # For the default profile the rename lands as a presentation-only
    # display_name; the canonical id ("default") is unchanged. Always return
    # the canonical id so callers keying on `name` stay correct.
    try:
        is_default = profiles_mod.normalize_profile_name(name) == "default"
    except ValueError:
        is_default = False
    if is_default:
        return {
            "ok": True,
            "name": "default",
            "display_name": body.new_name.strip(),
            "path": str(path),
        }
    return {
        "ok": True,
        "name": profiles_mod.normalize_profile_name(body.new_name),
        "path": str(path),
    }


@router.delete("/api/profiles/{name}")
async def delete_profile_endpoint(name: str):
    """The dashboard collects the user's confirmation in its own dialog, so
    ``yes=True`` always skips the CLI's interactive prompt."""
    from hermes_cli import profiles as profiles_mod
    with _profile_errors("DELETE /api/profiles/%s failed", name):
        # delete_profile() stops a running gateway by polling its PID once
        # every 500 ms for up to 10 s, then rmtree()s the profile directory;
        # on the loop that parks every request past the desktop's 10 s
        # WebSocket ready-probe.
        path = await run_in_threadpool(profiles_mod.delete_profile, name, yes=True)
    return {"ok": True, "path": str(path)}


@router.get("/api/profiles/{name}/soul")
async def get_profile_soul(name: str):
    soul_path = _resolve_profile_dir(name) / "SOUL.md"

    def _run():
        # Probe and read in the same hop: two round-trips would also widen the
        # window between the existence check and the read.
        if not soul_path.exists():
            return _MISSING
        return soul_path.read_text(encoding="utf-8")

    try:
        content = await run_in_threadpool(_run)
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"Could not read SOUL.md: {e}")
    if content is _MISSING:
        return {"content": "", "exists": False}
    return {"content": content, "exists": True}


@router.put("/api/profiles/{name}/soul")
async def update_profile_soul(name: str, body: ProfileSoulUpdate):
    soul_path = _resolve_profile_dir(name) / "SOUL.md"

    def _run():
        from utils import atomic_write_text

        # PUT replaces the whole persona document. A bare write_text() truncates
        # SOUL.md before the new body lands, and the paired GET reports an
        # unreadable file as ``{"content": "", "exists": False}`` — so an
        # interrupted save reads as "never set" and the editor's next Save
        # persists that empty document over it.
        #
        # preserve_mode keeps an existing file's mode/owner across the replace.
        # create_mode=0o644 covers the first save: named profiles seed SOUL.md
        # at the umask default (profiles chmods only .env to 0600) and SOUL.md
        # is not a secret. (The default profile's seeder runs on every
        # load_config, so its file already exists and preserve_mode applies.)
        atomic_write_text(
            soul_path, body.content, preserve_mode=True, create_mode=0o644
        )

    try:
        # atomic_write_text() writes a temp file, fsyncs it and replaces the
        # original — three syscalls that block for as long as the filesystem
        # takes to durably commit the persona document.
        await run_in_threadpool(_run)
    except OSError as e:
        _log.exception("PUT /api/profiles/%s/soul failed", name)
        raise HTTPException(status_code=500, detail=f"Could not write SOUL.md: {e}")
    return {"ok": True}


@router.put("/api/profiles/{name}/description")
async def update_profile_description_endpoint(name: str, body: ProfileDescriptionUpdate):
    """Set or clear a profile's role description (kanban routing signal).
    Non-empty stores it as user-authored (``description_auto: false``) so the
    auto-describer won't overwrite it on a sweep."""
    from hermes_cli import profiles as profiles_mod
    profile_dir = _resolve_profile_dir(name)
    text = (body.description or "").strip()
    with _profile_errors("PUT /api/profiles/%s/description failed", name,
                         not_found=(), bad_request=()):
        # write_profile_meta() reads profile.yaml, merges and rewrites it.
        await run_in_threadpool(
            profiles_mod.write_profile_meta,
            profile_dir,
            description=text,
            description_auto=False,
        )
    return {"ok": True, "description": text, "description_auto": False}


@router.put("/api/profiles/{name}/model")
async def update_profile_model_endpoint(name: str, body: ProfileModelUpdate):
    """Set the main model (``model.default`` + ``model.provider``) for a
    specific profile's config.yaml without touching the dashboard's own
    active profile. Mirrors ``POST /api/model/set`` (main scope) scoped to
    the named profile via the HERMES_HOME override."""
    profile_dir = _resolve_profile_dir(name)
    provider = (body.provider or "").strip()
    model = (body.model or "").strip()
    if not provider or not model:
        raise HTTPException(status_code=400, detail="provider and model are required")
    with _profile_errors("PUT /api/profiles/%s/model failed", name,
                         not_found=(), bad_request=()):
        # _write_profile_model() reads and rewrites the profile's config.yaml.
        await run_in_threadpool(_write_profile_model, profile_dir, provider, model)
    return {"ok": True, "provider": provider, "model": model}


@router.post("/api/profiles/{name}/describe-auto")
async def describe_profile_auto_endpoint(name: str, body: ProfileDescribeAuto):
    """Auto-generate a profile's description via the auxiliary LLM
    (``auxiliary.profile_describer``); mirrors ``hermes profile describe
    <name> --auto``. A failed generation (no aux client, LLM error, …) is
    ``ok: false`` with a reason rather than an HTTP error so the UI can
    surface it inline and let the operator fix config and retry."""
    # Resolution stays on the loop: a name check plus one stat, and it owns
    # the 400/404 mapping that the 500 fallback below would flatten.
    _resolve_profile_dir(name)

    def _run():
        from hermes_cli import profile_describer
        return profile_describer.describe_profile(name, overwrite=bool(body.overwrite))

    with _profile_errors("POST /api/profiles/%s/describe-auto failed", name,
                         not_found=(), bad_request=()):
        # describe_profile() is a synchronous LLM round-trip with a 60 s
        # ceiling; held on the loop it stalls every other dashboard request.
        outcome = await run_in_threadpool(_run)
    return {
        "ok": bool(outcome.ok),
        "reason": outcome.reason,
        "description": outcome.description,
        # Only a successful generation is an auto-authored description. A failed
        # sweep leaves any existing description untouched, so don't claim it's
        # auto-generated.
        "description_auto": bool(outcome.ok),
    }


# ── Export / Import ──────────────────────────────────────────────────────────
# Profile sharing for the desktop: wraps hermes_cli.profiles.export_profile /
# import_profile (the same machinery behind `hermes profile export|import`).
# Paths are exchanged, not bytes — the desktop's local and pooled backends
# share the filesystem with the native save/open dialogs that produce them.


def _read_desktop_overlay(profile_dir: Path) -> Any:
    """The desktop appearance overlay bundled with an imported profile
    (``desktop.json`` at the profile root); raises when unreadable."""
    return json.loads((profile_dir / "desktop.json").read_text(encoding="utf-8"))


@router.post("/api/profiles/{name}/export")
async def export_profile_endpoint(name: str, body: ProfileExport):
    from hermes_cli import profiles as profiles_mod

    output = (body.output or "").strip()
    if not output:
        try:
            output = str(profiles_mod.get_profile_export_path(name))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except OSError as exc:
            raise HTTPException(status_code=500, detail=f"Could not create export directory: {exc}")

    with _profile_errors("POST /api/profiles/%s/export failed", name):
        result = await run_in_threadpool(
            profiles_mod.export_profile, name, output, extra_files=body.extra_files or None)
    return {"ok": True, "archive": str(result)}


@router.post("/api/profiles/import")
async def import_profile_endpoint(body: ProfileImport):
    from hermes_cli import profiles as profiles_mod

    archive = (body.archive or "").strip()
    if not archive:
        raise HTTPException(status_code=400, detail="archive path is required")

    with _profile_errors("POST /api/profiles/import failed",
                         bad_request=(ValueError, FileExistsError)):
        profile_dir = await run_in_threadpool(
            profiles_mod.import_profile, archive, name=(body.name or "").strip() or None)

    imported = profile_dir.name
    # Match the CLI import flow: create the wrapper alias when it's safe.
    try:
        if not profiles_mod.check_alias_collision(imported):
            profiles_mod.create_wrapper_script(imported)
    except Exception:
        _log.exception("Creating wrapper for imported profile %s failed", imported)

    # Surface the bundled desktop appearance overlay (if the archive carried
    # one) so the desktop can apply theme/interface prefs without another
    # round-trip.
    desktop_overlay = None
    if (profile_dir / "desktop.json").is_file():
        try:
            desktop_overlay = _read_desktop_overlay(profile_dir)
        except Exception:
            _log.exception("Reading desktop.json from imported profile %s failed", imported)

    return {
        "ok": True,
        "name": imported,
        "path": str(profile_dir),
        "desktop": desktop_overlay,
    }


@router.get("/api/profiles/{name}/desktop-overlay")
async def get_profile_desktop_overlay(name: str):
    """The desktop appearance/interface overlay bundled with an imported
    profile (``desktop.json`` at the profile root), or ``exists: false``."""
    profile_dir = _resolve_profile_dir(name)

    def _run():
        # Probe and read in one hop; _MISSING (not None) because desktop.json
        # may legitimately hold the document ``null``.
        if not (profile_dir / "desktop.json").is_file():
            return _MISSING
        return _read_desktop_overlay(profile_dir)

    try:
        overlay = await run_in_threadpool(_run)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not read desktop.json: {e}")
    # _MISSING rather than None: an overlay file holding the document ``null``
    # exists, and must not be reported as absent.
    if overlay is _MISSING:
        return {"exists": False, "desktop": None}
    return {"exists": True, "desktop": overlay}
