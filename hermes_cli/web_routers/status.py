"""Status dashboard routes: health, /api/status, system stats, curator, learning graph, portal and diagnostics actions.

Extracted from ``hermes_cli.web_server``; helpers/state that tests monkeypatch on
``web_server`` stay there and are imported lazily at call time (cycle-safe).
"""

import concurrent.futures
import logging
import re
import asyncio
import os
import sys
import time
from fastapi import APIRouter
from fastapi import HTTPException, Request
from gateway.status import derive_gateway_busy, derive_gateway_drainable, normalize_updated_at, parse_active_agents, resolve_gateway_liveness
from hermes_cli import __version__, __release_date__
from hermes_cli.config import get_config_path, get_env_path
from hermes_cli.web_models import CuratorPause, LearningNodeRef, LearningNodeEdit, DebugShareRequest
from pathlib import Path
from typing import Any, Dict, Optional

_log = logging.getLogger("hermes_cli.web_server")
router = APIRouter()
# Mounted separately by web_server so /api/logs keeps its original route-table position.
logs_router = APIRouter()


@router.get("/api/ssh/ownership")
async def get_ssh_ownership(request: Request):
    from hermes_cli.web_server import _SSH_OWNER_NONCE, _require_token, _ssh_runtime_intact
    _require_token(request)
    if not _SSH_OWNER_NONCE:
        raise HTTPException(status_code=404, detail="SSH ownership is not active")
    return {
        "ok": True,
        "sshOwnerNonce": _SSH_OWNER_NONCE,
        "protocolVersion": 1,
        "runtimeIntact": _ssh_runtime_intact(),
    }


@router.get("/api/health")
async def get_health():
    """Lightweight process liveness for desktop/backend readiness probes."""
    from hermes_cli.web_server import app
    return {
        "ok": True,
        "version": __version__,
        "auth_required": bool(getattr(app.state, "auth_required", False)),
    }


_PROFILE_PLATFORM_STATUS_KEY_RE = re.compile(
    # Profile segment mirrors hermes_cli.profiles._PROFILE_ID_RE.  Platform
    # segment mirrors the Platform enum's normalized values: built-in members
    # plus plugin directory names (lowercased), which allow hyphens as well
    # as underscores (e.g. ``reviewer:foo-bar``).
    r"^[a-z0-9][a-z0-9_-]{0,63}:[a-z0-9][a-z0-9_-]{0,63}$"
)


def _is_profile_platform_status_key(key: object) -> bool:
    """Accept only the runner's public ``<profile>:<platform>`` key grammar."""
    return isinstance(key, str) and bool(_PROFILE_PLATFORM_STATUS_KEY_RE.fullmatch(key))


def _status_platform_key_allowed(
    key: object, configured: "set[str] | None"
) -> bool:
    """Decide whether a runtime-status platform key may appear publicly.

    Namespaced ``<profile>:<platform>`` keys are validated against the key
    grammar *unconditionally* — the config-set load failing must not fail
    open into projecting arbitrary colon-containing keys from a process-local
    JSON file onto the public endpoint.  Plain platform keys keep the
    long-standing behavior: checked against the configured set when it
    loaded, passed through when it did not.
    """
    if not isinstance(key, str):
        return False
    if ":" in key:
        return _is_profile_platform_status_key(key)
    return configured is None or key in configured


# Per-entry writer-identity stamps (added by gateway.status.write_runtime_status
# for the aggregation ownership check) are process recon — the same class of
# detail as the auth-gated top-level ``gateway_pid`` — and must not project
# onto the public endpoint.
_PRIVATE_PLATFORM_ENTRY_KEYS = frozenset({"writer_pid", "writer_start_time"})


def _public_platform_entry(value: Any) -> Any:
    """Strip writer-identity stamps from a platform entry before projection."""
    if not isinstance(value, dict):
        return value
    return {k: v for k, v in value.items() if k not in _PRIVATE_PLATFORM_ENTRY_KEYS}


def _merge_profile_gateway_platforms(
    gateway_platforms: dict, profile_platforms: dict
) -> dict:
    """Merge independent per-profile gateway platform states (OOF-3).

    Hosts that run separate gateway services per profile (``gateway_mode ==
    "multiple"``) persist each profile's platform failures in that profile's
    own ``gateway_state.json``.  The unparameterized ``/api/status`` — the
    machine-level probe NAS health monitoring reads — only read the active
    profile's file, so those failures were invisible to fleet health.  Fold
    them in under the same validated ``<profile>:<platform>`` grammar the
    multiplex path uses.  The active profile's own map is skipped (its
    entries are already present, including any multiplex-namespaced ones),
    and existing keys are never overwritten.
    """
    try:
        from hermes_cli.profiles import get_active_profile_name
        active = get_active_profile_name()
    except Exception:
        active = "default"
    merged = dict(gateway_platforms)
    for prof, plats in (profile_platforms or {}).items():
        if prof == active or not isinstance(plats, dict):
            continue
        for key, value in plats.items():
            if not isinstance(key, str) or ":" in key or not isinstance(value, dict):
                continue
            namespaced = f"{prof}:{key}"
            if not _is_profile_platform_status_key(namespaced):
                continue
            merged.setdefault(namespaced, _public_platform_entry(value))
    return merged


@router.get("/api/status")
async def get_status(profile: Optional[str] = None):
    from hermes_cli.web_server import (
        DASHBOARD_HEALTH,
        _GATEWAY_HEALTH_ROUTE_TIMEOUT,
        _GATEWAY_HEALTH_URL,
        _collect_profile_gateway_topology_cached,
        _config_profile_scope,
        _dashboard_local_update_managed_externally,
        _load_configured_gateway_platforms,
        _probe_gateway_health,
        _resolve_profile_dir,
        _resolve_restart_drain_timeout,
        _status_active_sessions,
        app,
        check_config_version,
        get_hermes_home,
        get_install_id,
        get_running_pid_cached,
        get_runtime_status_running_pid,
        read_runtime_status,
        run_in_threadpool,
    )
    status_scope = None
    requested_profile = (profile or "").strip()
    # Plain /api/status stays the machine-level public liveness probe. The
    # dashboard adds ?profile= when its management switcher targets another
    # profile, so its gateway badge reflects the selected profile.
    #
    # Use the config-only (contextvar) scope, NOT _profile_scope: this handler
    # awaits the remote-health probe, and _profile_scope swaps process-global
    # skills-module attributes that a concurrent request would cross-restore
    # across that await. Status only resolves get_hermes_home() at call time
    # (config/env/gateway state), which the task-local contextvar covers.
    profile_dir: Optional[Path] = None
    if requested_profile and requested_profile.lower() != "current":
        profile_dir = _resolve_profile_dir(requested_profile)
        status_scope = _config_profile_scope(requested_profile)
        status_scope.__enter__()

    try:
        current_ver, latest_ver = check_config_version()
        # --- Gateway liveness detection ---
        # Delegated to the single shared ladder in gateway.status so this
        # endpoint and /api/messaging/platforms can never disagree about
        # whether the gateway is up (they used to: sidebar "running" while
        # the Channels page rendered "The gateway is not running").
        #
        # When ?profile=<name> was given, scope PID and state reads to that
        # profile's directory — gateway identity files (PID, lock, runtime
        # status) are written to the per-profile home, not the process-level
        # HERMES_HOME (see issue #69143). Plain /api/status keeps the exact
        # zero-arg call so its behavior (and cache signature) is unchanged.
        #
        # The module-level probe references are handed to the resolver so the
        # long-standing `monkeypatch.setattr(web_server, "get_running_pid_cached", ...)`
        # seam used across the test-suite still intercepts them.
        def _bounded_health_probe():
            """Health probe with the route's blocking-call budget preserved.

            The resolver only reaches this rung when the local PID probe came
            up empty, so the timeout is paid at most once per request and only
            in the cross-container case that needs it.
            """
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(_probe_gateway_health)
                try:
                    return future.result(timeout=_GATEWAY_HEALTH_ROUTE_TIMEOUT)
                except concurrent.futures.TimeoutError:
                    _log.warning(
                        "/api/status gateway health probe exceeded %.2fs; "
                        "using local status",
                        _GATEWAY_HEALTH_ROUTE_TIMEOUT,
                    )
                    return False, None
                except Exception:
                    return False, None

        local_runtime = (
            read_runtime_status(path=profile_dir / "gateway_state.json")
            if profile_dir
            else read_runtime_status()
        )

        liveness = await run_in_threadpool(
            lambda: resolve_gateway_liveness(
                profile_dir=profile_dir,
                runtime=local_runtime,
                health_probe=_bounded_health_probe if _GATEWAY_HEALTH_URL else None,
                pid_probe=get_running_pid_cached,
                runtime_reader=read_runtime_status,
                runtime_pid_probe=get_runtime_status_running_pid,
            )
        )
        gateway_running = liveness.running
        gateway_pid = liveness.pid
        remote_health_body: dict | None = liveness.health_body

        gateway_state = None
        gateway_platforms: dict = {}
        gateway_exit_reason = None
        gateway_updated_at = None
        configured_gateway_platforms: set[str] | None = None
        try:
            configured_gateway_platforms = await run_in_threadpool(
                _load_configured_gateway_platforms
            )
        except Exception:
            configured_gateway_platforms = None

        # Prefer the detailed health endpoint response (has full state) when the
        # local runtime status file is absent or stale (cross-container).
        runtime = local_runtime
        if runtime is None and remote_health_body and remote_health_body.get("gateway_state"):
            runtime = remote_health_body

        if runtime:
            gateway_state = runtime.get("gateway_state")
            gateway_platforms = runtime.get("platforms") or {}
            # Namespaced entries are emitted by configured secondary-profile
            # adapters. The config set here belongs to the active/default
            # profile, so suffix-checking against it would incorrectly hide
            # secondary-only platforms. Colon-containing keys are validated
            # against the narrow key grammar UNCONDITIONALLY — a failed config
            # load must not fail open into projecting arbitrary keys from a
            # process-local JSON file onto this public endpoint.
            gateway_platforms = {
                key: _public_platform_entry(value)
                for key, value in gateway_platforms.items()
                if _status_platform_key_allowed(key, configured_gateway_platforms)
            }
            gateway_exit_reason = runtime.get("exit_reason")
            # Contract: gateway_updated_at is RFC3339 string | null, never a
            # number. ``runtime`` here may be the local gateway_state.json
            # (legacy gateways wrote epoch floats; hand edits can inject
            # anything) or a remote /health/detailed body — normalize both.
            gateway_updated_at = normalize_updated_at(runtime.get("updated_at"))
            if not gateway_running:
                gateway_state = gateway_state if gateway_state in {"stopped", "startup_failed"} else "stopped"
                # A cleanly stopped gateway's platform states are stale noise —
                # clear them so a dead process can't report "connected". But a
                # startup_failed gateway's FATAL entries are the diagnosis:
                # they carry per-profile credential collisions and auth
                # failures (multiplex entries under ``<profile>:<platform>``)
                # that the single exit_reason string can't express. Writer
                # -identity and freshness filtering upstream already dropped
                # entries from other/older processes, so keeping fatals here
                # cannot leak another gateway's live state (#80451 follow-up).
                if gateway_state == "startup_failed":
                    gateway_platforms = {
                        key: value
                        for key, value in gateway_platforms.items()
                        if isinstance(value, dict) and value.get("state") == "fatal"
                    }
                else:
                    gateway_platforms = {}
            elif gateway_running and remote_health_body is not None:
                # The health probe confirmed the gateway is alive, but the local
                # runtime status file may be stale (cross-container).  Override
                # stopped/None state so the dashboard shows the correct badge.
                if gateway_state in {None, "stopped"}:
                    gateway_state = "running"

        # If there was no runtime info at all but the health probe confirmed alive,
        # ensure we still report the gateway as running (no shared volume scenario).
        if gateway_running and gateway_state is None and remote_health_body is not None:
            gateway_state = "running"

        # Profile + gateway topology (cached, TTL 10s): fetched here — before
        # the platform rollup — because plain ``/api/status`` is the
        # machine-level probe NAS reads, and hosts running independent
        # per-profile gateway services (gateway_mode == "multiple") persist
        # each profile's platform failures in that profile's own
        # gateway_state.json.  Fold those in under the validated
        # ``<profile>:<platform>`` grammar so fleet health sees them (OOF-3).
        # A ``?profile=`` request targets one profile's view and is left
        # unmerged.
        topology = await run_in_threadpool(_collect_profile_gateway_topology_cached)
        if not requested_profile:
            gateway_platforms = _merge_profile_gateway_platforms(
                gateway_platforms, topology.get("profile_platforms") or {}
            )

        active_sessions = await _status_active_sessions()

        # Busy/drainable readout (NAS lifecycle-safety gate).  active_agents is
        # the in-flight gateway-turn count the gateway now persists at every
        # turn boundary; gateway_busy/gateway_drainable are derived from it +
        # liveness via the single shared contract in gateway.status.  Liveness
        # keys off gateway_running (a live PID/health probe), NEVER
        # gateway_updated_at — a healthy idle gateway never advances that.
        active_agents = parse_active_agents((runtime or {}).get("active_agents", 0))
        gateway_busy = derive_gateway_busy(
            gateway_running=gateway_running,
            gateway_state=gateway_state,
            active_agents=active_agents,
        )
        gateway_drainable = derive_gateway_drainable(
            gateway_running=gateway_running,
            gateway_state=gateway_state,
        )
        # Resolved drain timeout (seconds) so NAS can size its poll deadline
        # without out-of-band knowledge.  Offload to a thread: on a cold
        # Windows install the first import of hermes_cli.gateway blocks the
        # asyncio event loop for 15-30s (.pyc compilation + Defender scans),
        # exceeding the desktop handshake's 15s socket timeout.  After the
        # first call the module is in sys.modules and the worker call returns
        # in microseconds.
        restart_drain_timeout = await run_in_threadpool(_resolve_restart_drain_timeout)

        # Dashboard auth gate (Phase 7): surface whether the gate is engaged
        # and which providers are registered so ``hermes status`` and the
        # SPA's StatusPage can show "OAuth gate ON via Nous Research" or
        # "loopback only — no auth gate" with no extra round trips.
        auth_required = bool(getattr(app.state, "auth_required", False))
        auth_providers: list[str] = []
        # RFC 8252 native-app capability advertisement. The desktop reads this
        # to decide whether it can use the system-browser + loopback + PKCE
        # flow (no embedded webview, no session cookies) or must fall back to
        # the legacy embedded-webview cookie flow. "cookie" is always available
        # in gated mode; "native_pkce" is present when at least one interactive
        # session provider is registered — OAuth providers broker the upstream
        # IDP round trip, password providers complete interactively at /login
        # in the system browser (where OS password managers can autofill; an
        # embedded webview cannot reach them). Token-only credentials (e.g.
        # drain) don't count. Absent field / missing "native_pkce" ⇒ older
        # gateway ⇒ desktop falls back automatically.
        auth_flows: list[str] = []
        try:
            from hermes_cli.dashboard_auth import (
                list_providers as _list_providers,
                list_session_providers as _list_session_providers,
            )
            auth_providers = [p.name for p in _list_providers()]
            if auth_required:
                auth_flows.append("cookie")
                if _list_session_providers():
                    auth_flows.append("native_pkce")
        except Exception:
            # Module not importable yet (early startup) — leave as [].
            pass

        # Nous bootstrap-session validity for the NAS health sweep. A hosted
        # agent whose Nous auth dies terminally (invalid_grant / quarantine)
        # looks HEALTHY to every liveness/connectivity probe — the machine,
        # relay, and this dashboard all stay up — yet every inference turn
        # fails. This is the ONLY signal that surfaces that condition, and it
        # is determinable with no working token (local auth-store state). NAS
        # re-mints the bootstrap session when it reads "terminal". Best-effort:
        # never let auth classification break the public liveness probe.
        nous_session_valid = "unknown"
        try:
            from hermes_cli.auth import get_nous_session_validity
            nous_session_valid = get_nous_session_validity()
        except Exception:
            nous_session_valid = "unknown"

        # Always-public liveness + auth-gate shape. Safe for external uptime
        # probes (NAS's wildcard-subdomain liveness probe), the SPA's pre-login
        # bootstrap, and anyone who can curl the host — i.e. exactly the audience
        # ``PUBLIC_API_PATHS`` documents this endpoint as serving.
        status = {
            "version": __version__,
            "release_date": __release_date__,
            "config_version": current_ver,
            "latest_config_version": latest_ver,
            "can_update_hermes": not _dashboard_local_update_managed_externally(),
            "gateway_running": gateway_running,
            "gateway_state": gateway_state,
            "gateway_platforms": gateway_platforms,
            "gateway_exit_reason": gateway_exit_reason,
            "gateway_updated_at": gateway_updated_at,
            "active_agents": active_agents,
            "gateway_busy": gateway_busy,
            "gateway_drainable": gateway_drainable,
            "restart_drain_timeout": restart_drain_timeout,
            "active_sessions": active_sessions,
            "auth_required": auth_required,
            "auth_providers": auth_providers,
            "auth_flows": auth_flows,
            "nous_session_valid": nous_session_valid,
        }

        # Stable per-install identity (see get_install_id above). First call
        # may touch disk, so keep it off the event loop; afterwards it is a
        # process-global cache hit. Omitted (not null) when unpersistable so
        # older-client behavior and the no-identity fallback stay identical.
        install_id = await run_in_threadpool(get_install_id)
        if install_id:
            status["install_id"] = install_id

        # Component-level health rollup. Counts and status enums only — this
        # payload is public (PUBLIC_API_PATHS), so no messages, paths, or
        # other detail that could carry secrets. The storage probe reuses the
        # gateway readiness state_db check (read-only, 1s-bounded) in an
        # executor so a wedged DB can't stall the event loop.
        components: Dict[str, Any] = {
            "gateway": {
                "status": "ok" if gateway_running and gateway_state in {"running", "draining"} else "degraded",
                "state": gateway_state or ("running" if gateway_running else "stopped"),
            },
            "dashboard": DASHBOARD_HEALTH.snapshot(),
        }
        try:
            from gateway.readiness import _probe_state_db

            storage_check = await run_in_threadpool(_probe_state_db, get_hermes_home())
            components["storage"] = {"status": storage_check.get("status", "degraded")}
        except Exception:
            components["storage"] = {"status": "degraded"}
        platform_states = [
            str(value.get("state") or value.get("status") or "").lower()
            for value in gateway_platforms.values()
            if isinstance(value, dict)
        ]
        platforms_ok = all(
            state in {"connected", "running", "ok"} for state in platform_states
        )
        components["platforms"] = {
            "status": "ok" if platforms_ok else "degraded",
            "configured": len(gateway_platforms),
            "connected": sum(
                1 for state in platform_states if state in {"connected", "running", "ok"}
            ),
        }
        status["components"] = components
        status["overall"] = (
            "ok"
            if all(item.get("status") == "ok" for item in components.values())
            else "degraded"
        )

        # Memory-pressure rollup (NS-656). Distilled from the gateway's
        # 30s loop heartbeat + lifecycle sentinel — two small file reads,
        # no gateway IPC. Coarse MB numbers/enums/booleans only: this
        # endpoint is public (PUBLIC_API_PATHS), same disclosure class as
        # nous_session_valid above. Deliberately NOT folded into
        # components/overall — memory pressure is advisory (toast/notice
        # material), not a liveness verdict, and flipping `overall` to
        # "degraded" on it would page NAS's availability sweep for a
        # condition the valve is already handling.
        try:
            from gateway.memory_status import collect_memory_status

            status["memory"] = await run_in_threadpool(
                collect_memory_status,
                profile_dir if profile_dir else get_hermes_home(),
            )
        except Exception:
            status["memory"] = {"pressure": "unknown"}

        # Disk-usage rollup (NS-656, same lineage as OOF-2/OOF-107 fleet
        # disk-exhaustion incidents). One statvfs call on HERMES_HOME's
        # filesystem — coarse MB numbers + enum, same public disclosure
        # class as the memory block, and equally advisory: not folded
        # into components/overall.
        try:
            from gateway.disk_status import collect_disk_status

            status["disk"] = await run_in_threadpool(
                collect_disk_status,
                profile_dir if profile_dir else get_hermes_home(),
            )
        except Exception:
            status["disk"] = {"pressure": "unknown"}

        # Deferred FTS rebuild progress (schema v23): lets the desktop /
        # dashboard render a "search index rebuilding: N%" indicator instead
        # of users wondering why old-message search is slower after an
        # update. None/absent when no rebuild is pending (the common case).
        # Read-only probe, never blocks startup, never raises.
        try:
            from hermes_state import SessionDB as _SDB
            from hermes_constants import get_hermes_home as _ghh

            _db_path = _ghh() / "state.db"
            if _db_path.exists():
                _sdb = _SDB(db_path=_db_path, read_only=True)
                try:
                    _rebuild = _sdb.fts_rebuild_status()
                finally:
                    _sdb.close()
                if _rebuild is not None:
                    status["fts_rebuild"] = _rebuild
        except Exception:
            pass

        # Profile + gateway topology: which profiles exist, whether one
        # multiplexed gateway or several per-profile gateways serve them, and
        # (gated) which host ports the live gateways' port-binding platforms
        # listen on.  Enumerating profiles walks the filesystem and probes the
        # process table, so keep it off the event loop.
        #
        # Split by sensitivity: profile NAMES (``profiles``) and the gateway
        # ``gateway_mode`` are low-sensitivity PRODUCT surface — Hermes Cloud
        # renders the profile list in the Portal, which reads this endpoint over
        # the network (a gated bind), so they must survive the auth gate. The
        # per-gateway ``gateways[]`` detail carries host ports (deployment
        # recon), so it stays gated with the host paths / PID below.
        # (``topology`` was already fetched above, before the platform rollup,
        # so the per-profile platform merge could use it — the TTL cache makes
        # the earlier fetch the only real scan either way.)
        status["profiles"] = topology["profiles"]
        status["gateway_mode"] = topology["gateway_mode"]

        # Absolute host paths, the gateway PID, the internal gateway health
        # URL, and per-gateway ports are deployment recon a liveness probe never
        # needs. ``/api/status`` is in ``PUBLIC_API_PATHS`` so it bypasses
        # dashboard auth; on a network-exposed (gated) bind that means *any*
        # unauthenticated caller reaches it, and leaking host metadata there
        # contradicts the allowlist's own contract ("version, gateway state,
        # active session count, and the dashboard auth-gate shape. No bodies, no
        # session content, no secrets"). Surface this detail only on a loopback
        # / ``--insecure`` bind, where the dashboard is local-only and the
        # caller is already inside the trust envelope — the same loopback/gated
        # split ``should_require_auth`` draws.
        if not auth_required:
            status.update({
                "hermes_home": str(get_hermes_home()),
                "config_path": str(get_config_path()),
                "env_path": str(get_env_path()),
                "gateway_pid": gateway_pid,
                "gateway_health_url": _GATEWAY_HEALTH_URL,
                "gateways": topology["gateways"],
            })

        return status
    finally:
        if status_scope is not None:
            status_scope.__exit__(*sys.exc_info())


@router.get("/api/system/stats")
async def get_system_stats():
    """Host + process system stats for the System page.

    OS / Python / host identity from stdlib; CPU / memory / disk / uptime from
    psutil when available, with graceful degradation when it isn't.  Read-only
    and non-sensitive (no env values, no paths beyond the hermes home root).
    """
    from hermes_cli.web_server import _display_system_platform, get_hermes_home
    import platform as _platform

    info: Dict[str, Any] = {
        **_display_system_platform(
            system=_platform.system(),
            release=_platform.release(),
            version=_platform.version(),
            platform_label=_platform.platform(),
        ),
        "arch": _platform.machine(),
        "hostname": _platform.node(),
        "python_version": _platform.python_version(),
        "python_impl": _platform.python_implementation(),
        "hermes_version": __version__,
        "cpu_count": os.cpu_count(),
    }

    # psutil enriches the picture when present; everything below is optional.
    try:
        import psutil  # type: ignore

        vm = psutil.virtual_memory()
        info["memory"] = {
            "total": vm.total,
            "available": vm.available,
            "used": vm.used,
            "percent": vm.percent,
        }
        try:
            du = psutil.disk_usage(str(get_hermes_home()))
            info["disk"] = {
                "total": du.total,
                "used": du.used,
                "free": du.free,
                "percent": du.percent,
            }
        except Exception:
            pass
        try:
            info["cpu_percent"] = psutil.cpu_percent(interval=0.1)
            la = getattr(psutil, "getloadavg", None)
            if la:
                info["load_avg"] = list(la())
        except Exception:
            pass
        try:
            boot = psutil.boot_time()
            info["uptime_seconds"] = int(time.time() - boot)
        except Exception:
            pass
        try:
            proc = psutil.Process()
            info["process"] = {
                "pid": proc.pid,
                "rss": proc.memory_info().rss,
                "create_time": int(proc.create_time()),
                "num_threads": proc.num_threads(),
            }
        except Exception:
            pass
        info["psutil"] = True
    except Exception:
        info["psutil"] = False
        # stdlib-only fallbacks for load average + uptime where the kernel
        # exposes them.
        try:
            info["load_avg"] = list(os.getloadavg())
        except (OSError, AttributeError):
            pass

    return info


# ---------------------------------------------------------------------------
# Curator endpoints — background skill-maintenance status + controls.
#
# The curator periodically reviews skills (archive stale, prune, pin).  The
# dashboard surfaces its state and the pause/resume/run-now controls that
# `hermes curator` exposes.
# ---------------------------------------------------------------------------


@router.get("/api/curator")
async def get_curator_status():
    try:
        from agent import curator
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Curator unavailable: {exc}")
    try:
        state = curator.load_state()
    except Exception:
        state = {}
    return {
        "enabled": _safe_call(curator, "is_enabled", True),
        "paused": _safe_call(curator, "is_paused", False),
        "interval_hours": _safe_call(curator, "get_interval_hours", None),
        "last_run_at": state.get("last_run_at"),
        "min_idle_hours": _safe_call(curator, "get_min_idle_hours", None),
        "stale_after_days": _safe_call(curator, "get_stale_after_days", None),
        "archive_after_days": _safe_call(curator, "get_archive_after_days", None),
    }


@router.put("/api/curator/paused")
async def set_curator_paused(body: CuratorPause):
    from agent import curator

    curator.set_paused(bool(body.paused))
    return {"ok": True, "paused": bool(body.paused)}


@router.post("/api/curator/run")
async def run_curator():
    """Trigger a curator review now (backgrounded; tail via action status)."""
    from hermes_cli.web_server import _spawn_hermes_action
    try:
        proc = _spawn_hermes_action(["curator", "run"], "curator-run")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to run curator: {exc}")
    return {"ok": True, "pid": proc.pid, "name": "curator-run"}


@router.get("/api/learning/graph")
async def get_learning_graph(profile: Optional[str] = None):
    """Learning graph payload for the desktop panel.

    Profile-scoped view of learned, non-base skills plus memory chunks, with
    graph links derived from skill relations and memory-skill overlap.
    """
    from hermes_cli.web_server import _profile_scope
    def _run():
        from agent.learning_graph import build_learning_graph

        with _profile_scope(profile):
            return build_learning_graph()

    try:
        # _profile_scope takes _SKILLS_PROFILE_LOCK and the graph build reads
        # skills/memories from disk — keep it off the event loop.
        return await asyncio.to_thread(_run)
    except Exception:
        _log.exception("GET /api/learning/graph failed")
        raise HTTPException(status_code=500, detail="Failed to build learning graph")


@router.get("/api/learning/node")
async def get_learning_node(id: str, profile: Optional[str] = None):
    """Current content of a journey node (skill SKILL.md or memory chunk), for an edit prefill."""
    from hermes_cli.web_server import _profile_scope
    from agent.learning_mutations import node_detail

    def _run():
        with _profile_scope(profile):
            return node_detail(id)

    res = await asyncio.to_thread(_run)
    if not res.get("ok"):
        raise HTTPException(status_code=404, detail=res.get("message", "not found"))
    return res


@router.delete("/api/learning/node")
async def delete_learning_node(body: LearningNodeRef):
    """Delete a journey node — skills are archived (restorable), memories removed."""
    from hermes_cli.web_server import _profile_scope
    from agent.learning_mutations import delete_node

    def _run():
        with _profile_scope(body.profile):
            return delete_node(body.id)

    res = await asyncio.to_thread(_run)
    if not res.get("ok"):
        raise HTTPException(status_code=400, detail=res.get("message", "delete failed"))
    return res


@router.put("/api/learning/node")
async def update_learning_node(body: LearningNodeEdit):
    """Rewrite a journey node's content (SKILL.md or memory chunk)."""
    from hermes_cli.web_server import _profile_scope
    from agent.learning_mutations import edit_node

    def _run():
        with _profile_scope(body.profile):
            return edit_node(body.id, body.content)

    res = await asyncio.to_thread(_run)
    if not res.get("ok"):
        raise HTTPException(status_code=400, detail=res.get("message", "edit failed"))
    return res


def _safe_call(mod, fn_name: str, default):
    try:
        fn = getattr(mod, fn_name, None)
        return fn() if callable(fn) else default
    except Exception:
        return default


# ---------------------------------------------------------------------------
# Portal endpoint — Nous Portal auth + Tool Gateway routing status (read-only).
# ---------------------------------------------------------------------------


@router.get("/api/portal")
async def get_portal_status():
    # load_config() + auth/subscription snapshots are disk reads — this is a
    # polled endpoint, so keep them off the event loop.
    def _run():
        return _get_portal_status_sync()

    return await asyncio.to_thread(_run)


def _get_portal_status_sync():
    from hermes_cli.web_server import load_config
    cfg = load_config() or {}
    auth: Dict[str, Any] = {}
    try:
        from hermes_cli.auth import get_nous_auth_status_local

        # Read-only dashboard endpoint: refresh-free snapshot so polling
        # never performs an OAuth refresh or burns a refresh token.
        auth = get_nous_auth_status_local() or {}
    except Exception:
        auth = {}

    features = []
    try:
        from hermes_cli.nous_subscription import get_nous_subscription_features

        feats = get_nous_subscription_features(cfg)
        if feats is not None:
            for feat in feats.items():
                if getattr(feat, "managed_by_nous", False):
                    state = "via Nous Portal"
                elif getattr(feat, "active", False) and getattr(feat, "current_provider", None):
                    state = feat.current_provider
                elif getattr(feat, "active", False):
                    state = "active"
                else:
                    state = "not configured"
                features.append({"label": getattr(feat, "label", ""), "state": state})
    except Exception:
        _log.exception("portal features failed")

    model_cfg = cfg.get("model") if isinstance(cfg.get("model"), dict) else {}
    return {
        "logged_in": bool(auth.get("logged_in")),
        "portal_url": auth.get("portal_base_url"),
        "inference_url": auth.get("inference_base_url"),
        "provider": str((model_cfg or {}).get("provider") or ""),
        "subscription_url": "https://portal.nousresearch.com/manage-subscription",
        "features": features,
    }


# ---------------------------------------------------------------------------
# Diagnostics: prompt-size, support dump, debug upload, config migrate.
# All produce text output, so they spawn background actions tailed via
# /api/actions/<name>/status.
# ---------------------------------------------------------------------------


@router.post("/api/ops/prompt-size")
async def run_prompt_size():
    from hermes_cli.web_server import _spawn_hermes_action
    try:
        proc = _spawn_hermes_action(["prompt-size"], "prompt-size")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed: {exc}")
    return {"ok": True, "pid": proc.pid, "name": "prompt-size"}


@router.post("/api/ops/dump")
async def run_dump():
    from hermes_cli.web_server import _spawn_hermes_action
    try:
        proc = _spawn_hermes_action(["dump"], "dump")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed: {exc}")
    return {"ok": True, "pid": proc.pid, "name": "dump"}


@router.post("/api/ops/config-migrate")
async def run_config_migrate():
    from hermes_cli.web_server import _spawn_hermes_action
    try:
        proc = _spawn_hermes_action(["config", "migrate"], "config-migrate")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed: {exc}")
    return {"ok": True, "pid": proc.pid, "name": "config-migrate"}


@router.post("/api/ops/debug-share")
async def run_debug_share_endpoint(body: DebugShareRequest | None = None):
    """Upload a redacted debug report + full logs and return the paste URLs.

    Unlike the other diagnostics actions (doctor, dump, prompt-size) this is
    *synchronous*: the whole point of ``debug share`` is the set of shareable
    URLs it produces, so we run the upload in a worker thread and return the
    structured ``{urls, failures, redacted, ...}`` payload directly. The
    dashboard renders those as real, copyable links instead of scraping a log
    tail. Pastes auto-delete after 6 hours (handled inside the share core).
    """
    from hermes_cli.debug import build_debug_share

    req = body or DebugShareRequest()
    try:
        result = await asyncio.to_thread(
            build_debug_share,
            log_lines=max(1, min(int(req.lines), 5000)),
            redact=bool(req.redact),
        )
    except RuntimeError as exc:
        # Required summary-report upload failed (offline / paste service down).
        raise HTTPException(status_code=502, detail=f"Upload failed: {exc}")
    except Exception as exc:
        _log.exception("debug share failed")
        raise HTTPException(status_code=500, detail=f"Failed: {exc}")

    return {
        "ok": True,
        "urls": result.urls,
        "failures": result.failures,
        "redacted": result.redacted,
        "auto_delete_seconds": result.auto_delete_seconds,
    }


# ---------------------------------------------------------------------------
# Log viewer endpoint
# ---------------------------------------------------------------------------


@logs_router.get("/api/logs")
async def get_logs(
    file: str = "agent",
    lines: int = 100,
    level: Optional[str] = None,
    component: Optional[str] = None,
    search: Optional[str] = None,
):
    from hermes_cli.web_server import get_hermes_home
    from hermes_cli.logs import _read_tail, LOG_FILES

    log_name = LOG_FILES.get(file)
    if not log_name:
        raise HTTPException(status_code=400, detail=f"Unknown log file: {file}")
    log_path = get_hermes_home() / "logs" / log_name
    if not log_path.exists():
        return {"file": file, "lines": []}

    try:
        from hermes_logging import COMPONENT_PREFIXES
    except ImportError:
        COMPONENT_PREFIXES = {}

    # Normalize "ALL" / "all" / empty → no filter. _matches_filters treats an
    # empty tuple as "must match a prefix" (startswith(()) is always False),
    # so passing () instead of None silently drops every line.
    min_level = level if level and level.upper() != "ALL" else None
    if component and component.lower() != "all":
        comp_prefixes = COMPONENT_PREFIXES.get(component)
        if comp_prefixes is None:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown component: {component}. "
                       f"Available: {', '.join(sorted(COMPONENT_PREFIXES))}",
            )
    else:
        comp_prefixes = None

    has_filters = bool(min_level or comp_prefixes or search)
    result = _read_tail(
        log_path, min(lines, 500) if not search else 2000,
        has_filters=has_filters,
        min_level=min_level,
        component_prefixes=comp_prefixes,
    )
    # Post-filter by search term (case-insensitive substring match).
    # _read_tail doesn't support free-text search, so we filter here and
    # trim to the requested line count afterward.
    if search:
        needle = search.lower()
        result = [l for l in result if needle in l.lower()][-min(lines, 500):]
    return {"file": file, "lines": result}
