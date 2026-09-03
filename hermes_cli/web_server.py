"""Hermes Agent — Web UI server: FastAPI app assembly, auth/host middleware, ``start_server``.

Route handlers live in ``web_routers/``; their helpers live in the sibling
``web_server_<concern>`` modules and are re-imported here so ``web_server.<name>``
stays the single late-binding seam tests monkeypatch (``web_deps.late``).
Usage: ``python -m hermes_cli.main web [--port 8080]``.
"""

from contextlib import asynccontextmanager

import asyncio
from collections import deque
import hmac
import logging
import os
import re
import secrets
import shutil  # noqa: F401 — tests monkeypatch web_server.shutil.which
import subprocess
import sys
import sysconfig
import threading
import time
import urllib.parse

from hermes_cli.install_identity import get_install_id as _shared_get_install_id
from hermes_cli.pty_session import run_reaper
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hermes_cli import __version__
from hermes_cli.config import (  # noqa: F401 — late-bound by extracted routers/modules; tests monkeypatch web_server.<name>
    cfg_get,
    check_config_version,
    detect_install_method,
    get_hermes_home,
    load_config,
    load_env,
    remove_env_value,
    save_config,
    save_env_value,
)
from gateway.status import (  # noqa: F401 — late-bound by web_routers/status + tests monkeypatch web_server.<name>
    get_running_pid,
    get_running_pid_cached,
    get_runtime_status_running_pid,
    read_runtime_status,
)

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from starlette.concurrency import run_in_threadpool  # noqa: F401 — late-bound by web_server_cron/routers; tests patch web_server.run_in_threadpool
except ImportError:
    # First try lazy-installing the dashboard extras. Only the user actually
    # running `hermes dashboard` needs fastapi+uvicorn; lazy install keeps
    # them out of every other install path. After install, re-import.
    try:
        from tools.lazy_deps import ensure as _lazy_ensure
        _lazy_ensure("tool.dashboard", prompt=False)
        from fastapi import FastAPI, HTTPException, Request
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import JSONResponse
        from starlette.concurrency import run_in_threadpool  # noqa: F401
    except Exception:
        raise SystemExit(
            "Web UI requires fastapi and uvicorn.\n"
            f"Install with: {sys.executable} -m pip install 'fastapi' 'uvicorn[standard]'"
        )

WEB_DIST = Path(os.environ["HERMES_WEB_DIST"]) if "HERMES_WEB_DIST" in os.environ else Path(__file__).parent / "web_dist"
_log = logging.getLogger(__name__)


from hermes_cli.web_server_lifecycle import (  # noqa: E402,F401 — re-exported; routers/tests reach these via web_server.<name>
    PORT_IN_USE_EXIT_CODE,
    _dashboard_forwarded_allow_ips,
    _eager_reconcile_own_session_db,
    _is_addr_in_use_error,
    _is_serve_orphaned,
    _maybe_open_browser,
    _port_bind_conflict,
    _read_bound_port,
    _report_port_in_use,
    _resolve_restart_drain_timeout,
    _start_parent_death_watchdog,
    _valid_parent_start_marker,
    _warm_gateway_module,
    _write_dashboard_ready_file,
    _write_machine_sentinel_line,
)


def _start_desktop_cron_ticker(stop_event: "threading.Event", interval: int = 60) -> None:
    """Tick the cron scheduler from inside the desktop dashboard backend.

    The desktop spawns a ``hermes dashboard`` backend, not a gateway, so without
    this a cron created in the app would never fire (no live adapters; delivery
    falls back to the per-platform send path). The primary backend outlives the
    per-profile pool (reaped after ~10 idle minutes), so it ticks EVERY local
    profile's store like a multiplex gateway; external providers keep the
    single-store behavior (registries are not profile-scoped). Cross-process
    safe: the built-in tick takes the per-store ``cron/.tick.lock``.
    """
    from cron.scheduler_provider import InProcessCronScheduler, resolve_cron_scheduler

    provider = resolve_cron_scheduler()

    start_kwargs: dict = {"interval": interval}
    if isinstance(provider, InProcessCronScheduler):
        try:
            from hermes_cli.profiles import profiles_to_serve

            profile_homes = list(profiles_to_serve(multiplex=True))
            if len(profile_homes) > 1:
                start_kwargs["profile_homes"] = profile_homes
                # Stand down, per tick, for a profile whose OWN gateway runs:
                # it ticks with live adapters, and the tick-lock race would
                # otherwise deliver through the standalone path (#100489).
                from hermes_cli.profiles import _check_gateway_running

                start_kwargs["profile_gate"] = lambda _name, home: not _check_gateway_running(Path(home))
                from hermes_logging import enable_profile_log_routing

                enable_profile_log_routing(profile_homes)
                _log.info(
                    "Desktop cron scheduler will tick %d profile(s): %s",
                    len(profile_homes),
                    [name for name, _home in profile_homes],
                )
        except Exception:
            # Fail open to the single-store ticker so the active profile keeps firing.
            _log.exception("Desktop cron: profile enumeration failed; ticking active profile only")

    _log.info("Desktop cron scheduler started (provider=%s, interval=%ds)", provider.name, interval)
    provider.start(stop_event, **start_kwargs)


# Desktop `serve` only (start_server(start_mcp_discovery_after_bind=True)):
# seconds after the READY sentinel before the MCP discovery thread starts.
_DESKTOP_MCP_DISCOVERY_DELAY_S = 1.0


@asynccontextmanager
async def _lifespan(app: "FastAPI"):
    app.state.event_channels = {}  # dict[str, set]
    app.state.event_lock = asyncio.Lock()
    app.state.pty_active_session_files = {}  # dict[str, Path]
    # Serializes chat-argv resolution so concurrent /api/pty connections don't
    # overlap ``npm install`` / ``npm run build``. Locks live on app.state (not
    # module globals) so they bind to the running loop, not the import-time one.
    app.state.chat_argv_lock = asyncio.Lock()

    # Bring state.db schema current BEFORE the first session-list poll
    # (#79531/#80037): a store left behind by `hermes update` otherwise 500s
    # every poll while the read-probe heal loses to sibling lock contention.
    # Daemon thread so a locked store never delays the socket (Desktop
    # ready-probe times out at 10s, GH-73083).
    threading.Thread(
        target=_eager_reconcile_own_session_db,
        daemon=True,
        name="statedb-eager-reconcile",
    ).start()

    # Import hermes_cli.gateway *before* the yield: on Windows + 3.11 the
    # import holds the GIL, so run_in_executor still froze the loop 15-22s and
    # the Desktop's 10s ready-probe timed out (GH-73083).
    _warm_gateway_module()

    # Snapshot the checkout revision so lazy-import paths (model picker) can
    # refuse with "restart required" after `hermes update` replaced the code
    # (#86207); the update flow does not reliably restart the dashboard.
    from gateway.code_skew import record_boot_fingerprint

    record_boot_fingerprint()

    # Hosted Bot rooms belong to the backend process. Recovery may need a
    # contended state.db migration, so keep it off the pre-yield path: Group
    # Chat must degrade on its own rather than block every Desktop feature.
    from tui_gateway import methods_groups as _hosted_groups
    import tui_gateway.server  # noqa: F401

    hosted_room_start_cancel = threading.Event()

    def _start_hosted_rooms() -> None:
        try:
            _hosted_groups.start_hosted_room_service()
        except Exception:
            _log.exception("Hosted Group Chat recovery failed during backend startup")
        finally:
            if hosted_room_start_cancel.is_set():
                _hosted_groups.stop_hosted_room_service(timeout=1.0)

    hosted_room_start_thread = threading.Thread(
        target=_start_hosted_rooms,
        daemon=True,
        name="hosted-room-startup",
    )
    hosted_room_start_thread.start()

    # Desktop-spawned backends (HERMES_DESKTOP=1) fire cron jobs themselves,
    # since the app has no gateway running the scheduler. Server `hermes
    # dashboard` is unaffected — it relies on its own gateway.
    cron_stop: "threading.Event | None" = None
    cron_thread: "threading.Thread | None" = None
    if os.getenv("HERMES_DESKTOP") == "1":
        # Reap an orphaned gateway from an abnormal previous exit (reparented to
        # launchd, still holding the platform WebSocket) before forking a fresh
        # one that would race the same credential (#77276). Runs
        # unconditionally; protection of a healthy standalone gateway lives
        # INSIDE the reaper (registration probed with cleanup_stale=False).
        try:
            from hermes_cli.gateway import _reap_unsupervised_gateway_orphans

            _reap_unsupervised_gateway_orphans()
        except Exception:
            _log.exception("Desktop startup: orphan gateway reap failed")

        cron_stop = threading.Event()
        cron_thread = threading.Thread(
            target=_start_desktop_cron_ticker,
            args=(cron_stop,),
            daemon=True,
            name="desktop-cron-ticker",
        )
        cron_thread.start()

    # Reap idle/dead keep-alive PTY sessions (30-min TTL).
    pty_reaper_task = asyncio.create_task(run_reaper(PTY_REGISTRY))
    # Periodic authenticated self-test feeding the ``dashboard`` component on /api/status.
    selftest_task = asyncio.create_task(_dashboard_selftest_loop())
    # Live auto-archive timer, independent of list requests.
    auto_archive_task = asyncio.create_task(_auto_archive_ticker_loop())

    # Managed local runtime (local_runtime.enabled): bring llama-server back so a
    # restart doesn't strand a llamacpp main model. Off-thread and best-effort;
    # failure falls back to cloud providers like a cold start. Server only —
    # models load on first inference (an empty router holds no VRAM).
    def _boot_local_runtime():
        try:
            from hermes_cli.config import load_config
            from hermes_cli.local_runtime.bootstrap import ensure_local_runtime

            ensure_local_runtime(load_config())
        except Exception as exc:  # noqa: BLE001
            logging.getLogger(__name__).warning("local runtime boot failed: %s", exc)

    threading.Thread(target=_boot_local_runtime, daemon=True, name="local-runtime-boot").start()

    try:
        yield
    finally:
        hosted_room_start_cancel.set()
        _hosted_groups.stop_hosted_room_service(timeout=5.0)
        hosted_room_start_thread.join(timeout=1.0)
        if cron_stop is not None:
            cron_stop.set()
        pty_reaper_task.cancel()
        selftest_task.cancel()
        auto_archive_task.cancel()
        await PTY_REGISTRY.close_all()
        # Stop the managed llama-server with its parent (an orphan pins VRAM).
        try:
            from hermes_cli.local_runtime.bootstrap import shutdown_local_runtime

            shutdown_local_runtime()
        except Exception:  # noqa: BLE001
            pass
        if os.getenv("HERMES_DESKTOP") == "1":
            _terminate_desktop_managed_gateway()


def _app_state_default(app: "FastAPI", name: str, factory):
    """Return ``app.state.<name>``, lazily creating it for non-``with`` TestClient usages.

    The lifespan normally initialises these on the running event loop (an
    asyncio.Lock created at import time binds to whatever loop was active then).
    """
    try:
        return getattr(app.state, name)
    except AttributeError:
        value = factory()
        setattr(app.state, name, value)
        return value


def _get_chat_argv_lock(app: "FastAPI") -> asyncio.Lock:
    return _app_state_default(app, "chat_argv_lock", asyncio.Lock)


def _get_pty_active_session_files(app: "FastAPI") -> dict[str, Path]:
    return _app_state_default(app, "pty_active_session_files", dict)


app = FastAPI(title="Hermes Agent", version=__version__, lifespan=_lifespan)


# Memory-provider OAuth connect routes live in the memory layer, not here.
from hermes_cli.memory_oauth import router as _memory_oauth_router  # noqa: E402

app.include_router(_memory_oauth_router)

# Session token for sensitive endpoints. The desktop shell mints it via
# HERMES_DASHBOARD_SESSION_TOKEN; otherwise fresh per server start. It dies with
# the process and is injected into the SPA HTML so only the web UI can use it.
def _resolve_session_token() -> str:
    return os.environ.get("HERMES_DASHBOARD_SESSION_TOKEN") or secrets.token_urlsafe(32)


_SESSION_TOKEN = _resolve_session_token()
_SESSION_HEADER_NAME = "X-Hermes-Session-Token"
_SSH_OWNER_NONCE: Optional[str] = None
_SSH_RUNTIME_PURELIB: Optional[Tuple[str, int, int]] = None
_SSH_RUNTIME_MARKER: Optional[str] = None


def _apply_ssh_session_token(token: str) -> None:
    global _SESSION_TOKEN
    if token:
        _SESSION_TOKEN = token


def _apply_ssh_owner_nonce(nonce: Optional[str]) -> None:
    global _SSH_OWNER_NONCE, _SSH_RUNTIME_PURELIB, _SSH_RUNTIME_MARKER
    _SSH_OWNER_NONCE = nonce
    _SSH_RUNTIME_PURELIB = None
    _SSH_RUNTIME_MARKER = None
    if nonce:
        try:
            purelib = sysconfig.get_paths()["purelib"]
        except (KeyError, OSError):
            return
        # Primary identity: a marker FILE in site-packages. A replaced venv
        # loses it deterministically; pip installs leave it. A bare (dev, ino)
        # snapshot alone is NOT enough: ext4 reuses directory inodes at once,
        # so `rm -rf venv && uv venv` can land on the same inode undetected.
        try:
            marker = os.path.join(purelib, f".hermes-ssh-runtime-{nonce}")
            with open(marker, "w", encoding="utf-8") as fh:
                fh.write(f"pid={os.getpid()}\n")
            _SSH_RUNTIME_MARKER = marker
        except OSError:
            pass  # read-only site-packages — fall back to the stat snapshot
        try:
            st = os.stat(purelib)
            _SSH_RUNTIME_PURELIB = (purelib, st.st_dev, st.st_ino)
        except OSError:
            pass


def _ssh_runtime_intact() -> bool:
    if _SSH_RUNTIME_MARKER is not None:
        return os.path.isfile(_SSH_RUNTIME_MARKER)
    # Fallback (read-only site-packages): directory identity snapshot — weaker
    # (inode reuse) but catches cross-device moves and version-bump paths.
    if _SSH_RUNTIME_PURELIB is None:
        return True
    purelib, device, inode = _SSH_RUNTIME_PURELIB
    try:
        st = os.stat(purelib)
    except OSError:
        return False
    return (st.st_dev, st.st_ino) == (device, inode)


# In-browser Chat tab (/chat, /api/pty, /api/ws): always enabled. A module
# constant (not an inlined True) so the WS endpoints and SPA token injection
# share one testable seam.
_DASHBOARD_EMBEDDED_CHAT_ENABLED = True

# Desktop file.attach sends a whole base64 data URL in one JSON-RPC frame;
# uvicorn's 16 MiB default rejects files under the 256 MiB raw attach cap.
_DESKTOP_ATTACHMENT_WS_MAX_BYTES = 384 * 1024 * 1024


# CORS: localhost origins only — allow_origins=["*"] on 0.0.0.0 would let any
# website read/modify config and secrets.
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_methods=["*"],
    allow_headers=["*"],
)

# Endpoints that do NOT require the session token; everything else under /api/
# is gated below. Shared with the OAuth gate so the two allowlists cannot
# drift (/api/status once 401'd under the OAuth gate, breaking the portal probe).
from hermes_cli.dashboard_auth.public_paths import PUBLIC_API_PATHS as _PUBLIC_API_PATHS


def _has_valid_session_token(request: Request) -> bool:
    """True if the request carries a valid dashboard session token.

    The dedicated header avoids collisions with reverse proxies that already use
    ``Authorization`` (Caddy ``basic_auth``); the legacy Bearer path stays for
    older dashboard bundles.
    """
    session_header = request.headers.get(_SESSION_HEADER_NAME, "")
    if session_header and hmac.compare_digest(session_header.encode(), _SESSION_TOKEN.encode()):
        return True
    auth = request.headers.get("authorization", "")
    return hmac.compare_digest(auth.encode(), f"Bearer {_SESSION_TOKEN}".encode())


# Routes that may also authenticate via ``?token=`` (download links opened by
# the OS shell / a new tab, where no header can be set). Kept narrow.
_QUERY_TOKEN_API_PATHS: frozenset[str] = frozenset({"/api/files/download"})


def _has_valid_query_token(request: Request, path: str) -> bool:
    if path not in _QUERY_TOKEN_API_PATHS:
        return False
    token = request.query_params.get("token", "")
    return bool(token) and hmac.compare_digest(token.encode(), _SESSION_TOKEN.encode())


def _require_token(request: Request) -> None:
    """Authorize a sensitive endpoint, raising 401 if the caller isn't allowed.

    Loopback mode (``auth_required`` False): validate the SPA-injected
    ``_SESSION_TOKEN``. Gated mode: the token is NOT injected (cookie auth), and
    ``gated_auth_middleware`` already 401'd anything without a verified
    ``request.state.session`` — requiring the absent token here would make every
    ``_require_token`` endpoint unreachable behind the gate, so defer to it.
    """
    if getattr(request.app.state, "auth_required", False):
        ok = getattr(request.state, "session", None) is not None
    else:
        ok = _has_valid_session_token(request)
    if not ok:
        raise HTTPException(status_code=401, detail="Unauthorized")


# Accepted Host values for loopback binds. DNS rebinding TTL-flips an attacker
# hostname to 127.0.0.1 so the browser treats it as same-origin; validating Host
# at the app layer rejects it. See GHSA-ppp5-vxwm-4cf7.
_LOOPBACK_HOST_VALUES: frozenset = frozenset({"localhost", "127.0.0.1", "::1"})


def _dashboard_public_hosts() -> frozenset[str]:
    """Return the exact hostname declared by ``dashboard.public_url``.

    One source of truth for OAuth redirects, Host and WS Origin validation.
    Malformed or unset values fail closed as an empty set.
    """
    from hermes_cli.dashboard_auth.prefix import resolve_public_url

    public_url = resolve_public_url()
    try:
        hostname = urllib.parse.urlparse(public_url).hostname if public_url else None
    except ValueError:
        hostname = None
    return frozenset({hostname.lower()}) if hostname else frozenset()


def should_require_auth(host: str, allow_public: bool = False) -> bool:
    """True iff the auth gate must be active: any non-loopback bind.

    RFC1918 / CGNAT / link-local are deliberately PUBLIC — a hostile LAN device
    is the threat model. ``allow_public`` (legacy ``--insecure``) is accepted for
    old launch scripts but IGNORED since the June 2026 hermes-0day campaign.
    """
    return host not in _LOOPBACK_HOST_VALUES


def should_require_dashboard_auth(
    host: str,
    trusted_public_hosts: Optional[frozenset[str]] = None,
) -> bool:
    """Gate required for a non-loopback bind OR a non-loopback ``dashboard.public_url``.

    Callers may pass the already-resolved host set so startup and request
    validation share one snapshot.
    """
    if trusted_public_hosts is None:
        trusted_public_hosts = _dashboard_public_hosts()
    return should_require_auth(host) or any(h not in _LOOPBACK_HOST_VALUES for h in trusted_public_hosts)


def _desktop_loopback_auth_exempt(
    host: str,
    ssh_session_token: Optional[str] = None,
    ssh_owner_nonce: Optional[str] = None,
) -> bool:
    """True for a Desktop-owned loopback backend (#96490).

    A non-loopback ``dashboard.public_url`` would otherwise engage the
    ticket-only gate for the private loopback backends Desktop spawns, whose
    per-spawn session token the gate's WS path refuses — Desktop could not boot.
    The public dashboard is a separate non-loopback process that stays gated, so
    this never opens the public surface. Requires ALL of: loopback bind,
    ``HERMES_DESKTOP=1``, and an operator-minted credential (env token, SSH
    session token, or owner nonce).
    """
    return (
        host in _LOOPBACK_HOST_VALUES
        and os.environ.get("HERMES_DESKTOP") == "1"
        and bool(os.environ.get("HERMES_DASHBOARD_SESSION_TOKEN") or ssh_session_token or ssh_owner_nonce)
    )


def _host_header_hostname(host_header: str) -> str:
    """Return a normalized hostname from a valid HTTP Host authority.

    Host headers are authorities, not full URLs. Reject ambiguous ports,
    malformed IPv6 brackets, and URL syntax so validation always fails closed.
    """
    value = (host_header or "").strip()
    if not value or "://" in value or any(c in value for c in '"\'<> \n\r\t/?#@'):
        return ""

    if value.startswith("["):
        close = value.find("]")
        if close == -1:
            return ""
        hostname = value[1:close]
        # Bracket notation is reserved for IPv6 literals.
        if ":" not in hostname:
            return ""
        suffix = value[close + 1:]
        if suffix and not re.fullmatch(r":\d+", suffix):
            return ""
        return hostname.lower()

    # Unbracketed IPv6 authorities are ambiguous with a port separator.
    if value.count(":") > 1:
        return ""
    if ":" in value:
        hostname, port = value.rsplit(":", 1)
        if not hostname or not port.isdigit():
            return ""
        return hostname.lower()
    return value.lower()


def _is_accepted_host(
    host_header: str,
    bound_host: str,
    trusted_public_hosts: frozenset[str] = frozenset(),
) -> bool:
    """True if the Host header targets the interface we bound to.

    Accepts:
    - Exact bound host (with or without port suffix)
    - Loopback aliases when bound to loopback
    - Exact operator-declared public hosts (with or without port suffix)
    - Any host when bound to 0.0.0.0 (explicit opt-in to non-loopback,
      no protection possible at this layer)
    """
    host_only = _host_header_hostname(host_header)
    if not host_only:
        return False
    # All-interfaces bind: no Host-layer defence is possible; rely on operator
    # network controls.
    if host_only in trusted_public_hosts or bound_host in {"0.0.0.0", "::"}:
        return True
    bound_lc = bound_host.lower()
    if bound_lc in _LOOPBACK_HOST_VALUES:
        return host_only in _LOOPBACK_HOST_VALUES
    return host_only == bound_lc


@app.middleware("http")
async def host_header_middleware(request: Request, call_next):
    """Reject requests whose Host header doesn't match the bound interface (DNS rebinding, GHSA-ppp5-vxwm-4cf7)."""
    # app.state.bound_host is set by start_server() at listen time.
    bound_host = getattr(app.state, "bound_host", None)
    if bound_host and not _is_accepted_host(
        request.headers.get("host", ""), bound_host, getattr(app.state, "trusted_public_hosts", frozenset())
    ):
        return JSONResponse(
            status_code=400,
            content={
                "detail": (
                    "Invalid Host header. Dashboard requests must use the "
                    "bound hostname or the configured public hostname."
                ),
            },
        )
    return await call_next(request)


@app.middleware("http")
async def _plugin_api_runtime_gate(request: Request, call_next):
    """Block requests to disabled plugin API routes at request time.

    :func:`_mount_plugin_api_routes` gates at import time; a plugin disabled
    while running keeps its router mounted until restart, so enforce on every
    ``/api/plugins/{name}/...`` request. Registered BEFORE the auth middlewares
    (runs AFTER them): an unauthenticated caller must get auth's 401, never this
    404, or the status code becomes a plugin-name oracle.
    """
    path = request.url.path
    # parts: ['', 'api', 'plugins', '<name>', ...]
    parts = path.split("/")
    plugin_name = parts[3] if path.startswith("/api/plugins/") and len(parts) >= 4 else ""
    # Only gate authenticated requests. Unauthenticated ones fall through so
    # auth_middleware / the OAuth gate return 401 first and this route can't
    # be used as a plugin-name oracle.
    if plugin_name and (
        getattr(request.state, "token_authenticated", False)
        or getattr(request.app.state, "auth_required", False)
        or _has_valid_session_token(request)
        or _has_valid_query_token(request, path)
    ):
        try:
            from hermes_cli.plugins_cmd import _get_enabled_set, _get_disabled_set
            enabled_set = _get_enabled_set()
            disabled_set = _get_disabled_set()
        except Exception:
            enabled_set = set()
            disabled_set = set()
        # Source from the cached plugin list; unknown => user plugin (safe default — blocks).
        plugin = next((p for p in _get_dashboard_plugins() if p.get("name") == plugin_name), None)
        source = plugin.get("source") if plugin else "user"
        blocked = plugin_name in disabled_set or (source == "user" and plugin_name not in enabled_set)
        if blocked and source in ("user", "bundled"):
            return JSONResponse(status_code=404, content={"detail": "Plugin not found"})
    return await call_next(request)


@app.middleware("http")
async def _dashboard_auth_gate(request: Request, call_next):
    """OAuth gate — active only when start_server flags ``auth_required``; pass-through on loopback.

    Registered between host_header and auth_middleware: host check → cookie auth → token auth.
    """
    from hermes_cli.dashboard_auth.middleware import gated_auth_middleware
    return await gated_auth_middleware(request, call_next)


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    """Require the session token on all /api/ routes except the public list.

    Skipped for requests the token-auth seam already authenticated
    (``token_authenticated``) and when the OAuth gate is active — cookie auth is
    then authoritative and the loopback-only token path must not override it.
    """
    path = request.url.path
    if (
        not getattr(request.state, "token_authenticated", False)
        and not getattr(request.app.state, "auth_required", False)
        and path.startswith("/api/")
        and path not in _PUBLIC_API_PATHS
        and not path.startswith("/api/mcp/oauth/callback/")
        and not _has_valid_session_token(request)
        and not _has_valid_query_token(request, path)
    ):
        return JSONResponse(status_code=401, content={"detail": "Unauthorized"})
    return await call_next(request)


@app.middleware("http")
async def _token_auth_seam(request: Request, call_next):
    """Outermost auth seam: bearer-token auth for opted-in routes (registered LAST = runs FIRST).

    A registered token route is owned here — authenticate, attach the principal
    + ``token_authenticated`` so downstream gates skip enforcement. Non-token
    routes pass through untouched.
    """
    from hermes_cli.dashboard_auth.token_auth import token_auth_middleware
    return await token_auth_middleware(request, call_next)


_DASHBOARD_HEALTH_WINDOW_SECONDS = 300.0


class DashboardHealth:
    """Dashboard-process health: rolling unhandled-error/5xx window + periodic self-test result.

    Feeds ``components`` on the PUBLIC ``/api/status``, so :meth:`snapshot`
    exports counts and enums only — never ``last_error_type``/``last_error_path``.
    """

    def __init__(self, window_seconds: float = _DASHBOARD_HEALTH_WINDOW_SECONDS) -> None:
        self.window_seconds = window_seconds
        self._error_times: "deque[float]" = deque(maxlen=256)
        self.last_error_type: Optional[str] = None
        self.last_error_path: Optional[str] = None  # internal-only, never serialized
        self.last_error_at: Optional[float] = None
        self.selftest_status: str = "unknown"  # unknown | ok | failing
        self.selftest_http_status: Optional[int] = None
        self.selftest_at: Optional[float] = None

    def record_error(self, exc_type: str, path: str) -> None:
        now = time.time()
        self._error_times.append(now)
        self.last_error_type = exc_type
        self.last_error_path = path
        self.last_error_at = now

    def record_selftest(self, passed: bool, http_status: Optional[int]) -> None:
        self.selftest_status = "ok" if passed else "failing"
        self.selftest_http_status = http_status
        self.selftest_at = time.time()

    def recent_error_count(self) -> int:
        cutoff = time.time() - self.window_seconds
        while self._error_times and self._error_times[0] < cutoff:
            self._error_times.popleft()
        return len(self._error_times)

    def snapshot(self) -> Dict[str, Any]:
        """Public component payload: status enum + counts + timestamps only."""
        errors = self.recent_error_count()
        status = "degraded" if (errors or self.selftest_status == "failing") else "ok"
        return {
            "status": status,
            "recent_unhandled_errors": errors,
            "last_error_at": self.last_error_at,
            "selftest": self.selftest_status,
        }


DASHBOARD_HEALTH = DashboardHealth()


@app.middleware("http")
async def _dashboard_health_middleware(request: Request, call_next):
    """Outermost middleware (registered last): count unhandled exceptions and 5xx; re-raises, never alters."""
    try:
        response = await call_next(request)
    except Exception as exc:
        DASHBOARD_HEALTH.record_error(type(exc).__name__, request.url.path)
        raise
    if response.status_code >= 500:
        DASHBOARD_HEALTH.record_error(f"http_{response.status_code}", request.url.path)
    return response


# Authenticated-route self-test: one in-process request per minute against a
# cheap DB-touching route, catching "liveness fine but every authed request 500s".
_DASHBOARD_SELFTEST_INTERVAL_SECONDS = 60.0
_DASHBOARD_SELFTEST_ROUTE = "/api/sessions?limit=1"


async def _dashboard_selftest_once() -> None:
    """Run one authenticated in-process self-test request and record it."""
    try:
        import httpx
    except ImportError:
        return  # optional dependency — leave status "unknown"
    try:
        # Loopback base_url so the Host-header middleware accepts the request.
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1") as client:
            resp = await client.get(_DASHBOARD_SELFTEST_ROUTE, headers={_SESSION_HEADER_NAME: _SESSION_TOKEN})
        DASHBOARD_HEALTH.record_selftest(resp.status_code == 200, resp.status_code)
    except Exception:
        DASHBOARD_HEALTH.record_selftest(False, None)


async def _dashboard_selftest_loop() -> None:
    """Periodic self-test driver started from the lifespan."""
    try:
        import httpx  # noqa: F401
    except ImportError:
        _log.debug("httpx unavailable — dashboard self-test disabled")
        return
    while True:
        await asyncio.sleep(_DASHBOARD_SELFTEST_INTERVAL_SECONDS)
        # OAuth-gated binds don't honour the session token; the probe would false-alarm 401.
        if getattr(app.state, "auth_required", False):
            continue
        await _dashboard_selftest_once()


# Helpers extracted into web_server_<concern> modules, re-exported so
# ``web_server.<name>`` stays the late-binding seam (web_deps.late) routers use
# and tests monkeypatch. Every name here has a live web_server.<name> reference.
from hermes_cli.web_server_config import (  # noqa: E402,F401
    CONFIG_SCHEMA, _AUX_TASK_SLOTS, _apply_main_model_assignment, _apply_model_assignment_sync,
    _dashboard_code_skew_guard, _denormalize_config_from_web, _memory_provider_options,
    _normalize_config_for_web, _normalize_main_model_assignment,
    _schema_with_dynamic_provider_options, _timezone_options,
)
from hermes_cli.web_models import (  # noqa: E402,F401
    ConfigUpdate, WhatsAppOnboardingStart, WhatsAppOnboardingApply, MoaModelSlot, MoaPresetPayload,
    MoaConfigPayload, BulkDeleteSessions, CronJobCreate, CronJobUpdate,
    AutomationBlueprintInstantiate, MCPServerCreate,
)
from hermes_cli.web_server_gateway import (  # noqa: E402,F401
    _ACTION_COMMANDS, _ACTION_IDS, _ACTION_LOG_DIR, _ACTION_LOG_FILES, _ACTION_PROCS,
    _ACTION_RESULTS, _TOPOLOGY_CACHE, _TOPOLOGY_CACHE_TTL, _collect_profile_gateway_topology,
    _collect_profile_gateway_topology_cached, _dashboard_spawn_executable, _display_system_platform,
    _gateway_subcommand, _load_configured_gateway_platforms, _probe_gateway_health,
    _profile_gateway_writer_identity, _profile_platform_ports, _restart_gateway_after,
    _spawn_hermes_action, _split_text_for_speak_stream, _strip_session_list_rows,
    _terminate_desktop_managed_gateway,
)
from hermes_cli.web_server_files import (  # noqa: E402,F401
    _dashboard_local_update_managed_externally, _fs_path, _managed_file_entry,
    _managed_response_meta, _path_is_under, _resolve_managed_path,
)
from hermes_cli.web_server_memory import (  # noqa: E402,F401
    _coerce_bool, _dependency_importable, _discover_memory_provider_statuses, _field_default,
    _field_is_set, _field_value, _field_visible, _load_memory_provider, _memory_provider_manifest,
    _memory_provider_setup_info, _memory_provider_setup_manifest, _normalize_memory_provider_name,
    _normalize_memory_provider_schema, _read_memory_provider_existing_values,
    _require_memory_provider_ready, _run_setup_command,
)
from hermes_cli.web_server_profiles import (  # noqa: E402,F401
    _profile_cli_args, _hub_action_name, _installed_hub_identifiers, _SKILLS_PROFILE_LOCK,
    _TERMINAL_BACKENDS, _approval_mode_of, _aux_task_summary, _aux_usage_rows,
    _broadcast_gateway_session_info, _config_profile_scope, _fallback_profile_dicts,
    _is_other_profile, _merge_aux_into_by_model, _parse_model_ids, _plugin_terminal_backend_rows,
    _profile_scope, _resolve_profile_dir, _write_profile_mcp_servers,
)
from hermes_cli.web_server_messaging import (  # noqa: E402,F401
    _MESSAGING_KEYS_PAGE_KEYS, _TelegramOnboardingPairing, _WhatsAppOnboardingSession,
    _build_catalog_entry, _channel_managed_env_keys, _messaging_platform_catalog,
    _restart_gateway_after_whatsapp_onboarding, _telegram_onboarding_error_message,
    _telegram_onboarding_lock, _telegram_onboarding_pairings, _telegram_onboarding_request_sync,
    _whatsapp_onboarding_payload, _whatsapp_onboarding_sessions, _whatsapp_session_path,
    _write_platform_enabled,
)
from hermes_cli.web_server_oauth import (  # noqa: E402,F401
    _OAUTH_PROVIDER_CATALOG, _external_process_cli_command, _minimax_poller, _nous_poller,
    _oauth_profile_name, _oauth_sessions, _oauth_sessions_lock, _truncate_token, _xai_device_poller,
)
from hermes_cli.web_server_sessions import (  # noqa: E402,F401
    _auto_archive_ticker_loop, _last_auto_archive_check, _maybe_auto_archive_for_profile,
    _open_session_db_at_path, _open_session_db_for_profile, _session_db_heal_exhausted,
    _session_db_heal_warned, _session_db_read_probe_statements, _session_latest_descendant,
)
from hermes_cli.web_server_cron import (  # noqa: E402,F401
    _call_cron_for_profile, _create_cron_job_sync, _cron_default_profile, _cron_optional_text,
    _cron_profile_dicts, _cron_profile_home, _cron_string_list, _find_cron_job_profile,
    _fire_cron_job_for_profile, _forward_cron_fire_to_gateway, _gateway_fire_endpoint,
    _gateway_intentionally_stopped, _mutate_cron_for_profile, _normalize_dashboard_cron_script,
    _notify_cron_provider_for_profile, _raise_if_cron_registration_error, _run_cron_dashboard_io,
    _validate_dashboard_cron_context_from, _validate_dashboard_cron_effective_job,
)
from hermes_cli.web_server_mcp import (  # noqa: E402,F401
    _mcp_oauth_flows, _mcp_server_summary, _normalize_mcp_server_create, _run_dashboard_mcp_oauth,
)
from hermes_cli.web_server_chat import (  # noqa: E402,F401
    PTY_REGISTRY, PtyBridge, PtyUnavailableError, _GATEWAY_WS_PROTOCOL,
    _GATEWAY_WS_TICKET_PROTOCOL_PREFIX, _LOOPBACK_HOSTS, _PTY_BRIDGE_AVAILABLE, _RESIZE_RE,
    _active_session_file_for_channel, _build_gateway_ws_url, _build_sidecar_url,
    _get_console_executor, _legacy_pump, _resolve_chat_argv, _resolve_chat_argv_async,
    _resolve_client_ws_host, _ws_auth_ok, _ws_auth_reason, _ws_client_is_allowed, _ws_client_reason,
    _ws_host_origin_is_allowed, _ws_host_origin_reason, _ws_request_is_allowed,
)
from hermes_cli.web_server_dashboard import (  # noqa: E402,F401
    _BUILTIN_DASHBOARD_THEMES, _discover_dashboard_plugins, _discover_user_themes,
    _invalidate_plugins_hub_cache, _merged_plugins_hub, _mount_plugin_api_routes,
    _normalise_theme_definition, _render_active_theme_bootstrap_css, _safe_plugin_api_relpath,
    _schedule_check_fn_probe, mount_spa,
)


# Legacy re-exports of route handlers; tests call these via web_server.<name>.
from hermes_cli.web_routers.files import upload_managed_file_stream  # noqa: E402,F401
from hermes_cli.web_routers.status import get_status, run_dump  # noqa: E402,F401
from hermes_cli.web_routers.sessions import search_sessions  # noqa: E402,F401
from hermes_cli.web_routers.models import (  # noqa: E402,F401
    get_model_options, get_recommended_default_model, set_moa_models,
)
from hermes_cli.web_routers.messaging import (  # noqa: E402,F401
    apply_whatsapp_onboarding, start_whatsapp_onboarding,
)
from hermes_cli.web_routers.oauth import (  # noqa: E402,F401
    _codex_full_login_worker, _new_oauth_session, _resolve_provider_status,
)
from hermes_cli.web_routers.sessions import (  # noqa: E402,F401
    bulk_delete_sessions_endpoint, count_empty_sessions_endpoint, delete_empty_sessions_endpoint,
    get_session_latest_descendant, get_session_messages, delete_session_endpoint,
    export_session_endpoint, prune_sessions_endpoint,
)
from hermes_cli.web_routers.cron import (  # noqa: E402,F401
    list_cron_jobs, create_cron_job, update_cron_job, pause_cron_job, resume_cron_job,
    trigger_cron_job, delete_cron_job, instantiate_blueprint, _normalize_dashboard_cron_updates,
)
from hermes_cli.web_routers.ops import (  # noqa: E402,F401
    list_credential_pool, run_doctor, run_import,
)
from hermes_cli.web_routers.analytics import (  # noqa: E402,F401
    get_models_analytics, get_usage_analytics,
)
from hermes_cli.web_routers.chat_ws import (  # noqa: E402,F401
    _broadcast_event, _get_event_state, pty_ws,
)
from hermes_cli.web_routers.dashboard_ui import post_agent_plugin_install  # noqa: E402,F401


_GATEWAY_HEALTH_URL = os.getenv("GATEWAY_HEALTH_URL")
_GATEWAY_HEALTH_TIMEOUT_MAX = 1.0
try:
    _GATEWAY_HEALTH_TIMEOUT = float(os.getenv("GATEWAY_HEALTH_TIMEOUT", "1"))
except (ValueError, TypeError):
    _log.warning(
        "Invalid GATEWAY_HEALTH_TIMEOUT value %r — using default 1.0s",
        os.getenv("GATEWAY_HEALTH_TIMEOUT"),
    )
    _GATEWAY_HEALTH_TIMEOUT = 1.0
if _GATEWAY_HEALTH_TIMEOUT <= 0:
    _log.warning(
        "Invalid non-positive GATEWAY_HEALTH_TIMEOUT value %.3fs — using default 1.0s",
        _GATEWAY_HEALTH_TIMEOUT,
    )
    _GATEWAY_HEALTH_TIMEOUT = 1.0
elif _GATEWAY_HEALTH_TIMEOUT > _GATEWAY_HEALTH_TIMEOUT_MAX:
    _log.warning(
        "Capping GATEWAY_HEALTH_TIMEOUT %.3fs to %.3fs for dashboard liveness probes",
        _GATEWAY_HEALTH_TIMEOUT,
        _GATEWAY_HEALTH_TIMEOUT_MAX,
    )
    _GATEWAY_HEALTH_TIMEOUT = _GATEWAY_HEALTH_TIMEOUT_MAX


_MANAGED_FILE_MAX_BYTES = 100 * 1024 * 1024
_FS_DATA_URL_MAX_BYTES = 16 * 1024 * 1024
# Multipart uploads stream to a temp file in fixed chunks and rename into
# place: constant memory, no base64 inflation, no proxy body-size 502s (NS-501).
_UPLOAD_CHUNK_BYTES = 1024 * 1024

# Stable install identity for /api/status: one uuid4 hex per physical install,
# persisted under the ROOT Hermes home (not the profile HERMES_HOME) so every
# profile reports the same id and the desktop can collapse duplicate roster rows
# for one backend. Must never change across restarts, so cached per process.
_INSTALL_ID_CACHE: Dict[str, Optional[str]] = {"root": None, "value": None}


def get_install_id() -> Optional[str]:
    """Process-lifetime-cached stable install id."""
    return _shared_get_install_id(cache=_INSTALL_ID_CACHE)


# Serializes config.yaml read-modify-write cycles for handlers on worker threads
# (asyncio.to_thread): config.py's _CONFIG_LOCK covers each load/save call, not
# the span between them, so two off-loop updates could drop each other's writes.
# RLock so nested helpers that also take it can't self-deadlock.
_CONFIG_MUTATION_LOCK = threading.RLock()

# A finished ``gateway-restart`` child does not mean the gateway is back (it
# exits once the restart is handed off), so in-flight reuse stops coalescing
# exactly when a stale frontend re-fires every few seconds (#89034: 77 restarts,
# state.db corrupted mid-FTS5-write). MAINTAINER DECISION: a fixed window, not
# "until healthy" — a gateway that never returns must not leave the action
# inert. 10s is above the ~3.5s storm spacing and below an operator's retry.
GATEWAY_RESTART_COOLDOWN_SECONDS = 10.0

# ``(monotonic spawn time, Popen, command)`` of the last restart. Deliberately
# NOT read from ``_ACTION_PROCS``: entries there vanish when the child exits.
_LAST_GATEWAY_RESTART: Optional[Tuple[float, subprocess.Popen, Tuple[str, ...]]] = None


def _spawn_gateway_restart(profile: Optional[str] = None) -> Tuple[subprocess.Popen, bool]:
    """Spawn ``hermes gateway restart``, reusing an in-flight or recent restart.

    Concurrent children race each other on the kill-and-start path, so a live
    child is reused; requests within ``GATEWAY_RESTART_COOLDOWN_SECONDS`` for the
    same profile coalesce onto the last spawn too (#89034). Orphaned gateways
    are reaped first so the fresh one doesn't stack a duplicate (#77276).
    Returns ``(proc, reused)``.
    """
    try:
        from hermes_cli.gateway import _reap_unsupervised_gateway_orphans

        _reap_unsupervised_gateway_orphans()
    except Exception:
        pass  # best-effort — don't block the restart on a reap failure

    global _LAST_GATEWAY_RESTART

    subcommand = _gateway_subcommand(profile, "restart")
    existing = _ACTION_PROCS.get("gateway-restart")
    if existing is not None and existing.poll() is None:
        existing_command = _ACTION_COMMANDS.get("gateway-restart")
        if existing_command is None or existing_command == tuple(subcommand):
            return existing, True
        raise RuntimeError("gateway restart already in progress for another profile")

    recent = _LAST_GATEWAY_RESTART
    if recent is not None:
        spawned_at, recent_proc, recent_command = recent
        age = time.monotonic() - spawned_at if recent_command == tuple(subcommand) else None
        if age is not None and age < GATEWAY_RESTART_COOLDOWN_SECONDS:
            _log.info(
                "Coalescing gateway restart: one was started %.1fs ago "
                "(pid %s) and the gateway may still be coming back; not "
                "spawning another (#89034).",
                age,
                getattr(recent_proc, "pid", "?"),
            )
            return recent_proc, True

    proc = _spawn_hermes_action(subcommand, "gateway-restart")
    _LAST_GATEWAY_RESTART = (time.monotonic(), proc, tuple(subcommand))
    return proc, False


# Collapses repeated identical ElevenLabs voice-list failures (the desktop
# re-polls on every settings focus) to one log line; re-arms on success or a
# changed signature.
_voice_list_last_error: Optional[str] = None


def _voice_list_error_logged_once(signature: Optional[str]) -> bool:
    """True if ``signature`` is new and should be logged now; ``None`` clears the latch."""
    global _voice_list_last_error
    if signature is None:
        _voice_list_last_error = None
        return False
    if signature == _voice_list_last_error:
        return False
    _voice_list_last_error = signature
    return True


_ACTION_LOG_FILES.setdefault("computer-use-grant", "action-computer-use-grant.log")

# Cache discovered plugins per-process (refresh on explicit re-scan).
_dashboard_plugins_cache: Optional[list] = None


def _get_dashboard_plugins(force_rescan: bool = False) -> list:
    global _dashboard_plugins_cache
    stale = _dashboard_plugins_cache is None or force_rescan or any(
        not Path(p["_dir"]).is_dir() for p in _dashboard_plugins_cache
    )
    if stale:
        _dashboard_plugins_cache = _discover_dashboard_plugins()
    return _dashboard_plugins_cache


# Router mounting. ORDER IS ROUTE-MATCHING ORDER: literal paths must land before
# templated siblings (e.g. /api/sessions/bulk-delete before /api/sessions/{id}).
from hermes_cli.web_routers import (  # noqa: E402
    files as _files_routes,
    git as _git_routes,
    local_models as _local_models_routes,
    status as _status_routes,
    actions as _actions_routes,
    audio as _audio_routes,
    sessions as _sessions_routes,
    profiles as _profiles_routes,
    memory_providers as _memory_providers_routes,
    config_env as _config_env_routes,
    models as _models_routes,
    messaging as _messaging_routes,
    oauth as _oauth_routes,
    cron as _cron_routes,
    mcp as _mcp_routes,
    ops as _ops_routes,
    skills as _skills_routes,
    tools as _tools_routes,
    analytics as _analytics_routes,
    chat_ws as _chat_ws_routes,
    dashboard_ui as _dashboard_ui_routes,
)

app.include_router(_files_routes.router)
app.include_router(_git_routes.router)
app.include_router(_local_models_routes.router)
app.include_router(_status_routes.router)
app.include_router(_actions_routes.router)
app.include_router(_audio_routes.router)
app.include_router(_actions_routes.status_router)
app.include_router(_sessions_routes.list_router)
app.include_router(_profiles_routes.sessions_router)
app.include_router(_sessions_routes.search_router)
app.include_router(_memory_providers_routes.router)
app.include_router(_config_env_routes.config_router)
app.include_router(_models_routes.router)
app.include_router(_config_env_routes.router)
app.include_router(_messaging_routes.router)
app.include_router(_oauth_routes.router)
app.include_router(_sessions_routes.manage_router)
app.include_router(_status_routes.logs_router)
app.include_router(_cron_routes.router)
app.include_router(_mcp_routes.router)
app.include_router(_ops_routes.router)
app.include_router(_skills_routes.hub_router)
app.include_router(_profiles_routes.router)
app.include_router(_skills_routes.router)
app.include_router(_tools_routes.router)
app.include_router(_analytics_routes.router)
app.include_router(_chat_ws_routes.router)
app.include_router(_dashboard_ui_routes.router)

# Plugin API routes and the dashboard auth routes (/login, /auth/*, /api/auth/*)
# mount before the SPA catch-all so /{full_path:path} doesn't swallow them. Auth
# routes are always mounted — the gate middleware decides enforcement.
_mount_plugin_api_routes()
from hermes_cli.dashboard_auth.routes import router as _dashboard_auth_router  # noqa: E402

app.include_router(_dashboard_auth_router)
mount_spa(app)


def _no_auth_provider_message(host: str) -> str:
    """Actionable SystemExit text for a gated bind with no registered auth provider.

    Names the exact trigger: on a loopback bind the ONLY trigger is
    dashboard.public_url, so print the offending URL and the remove-it exit.
    Bundled providers expose ``LAST_SKIP_REASON`` so an installed-but-
    unconfigured provider is not reported as merely "no providers".
    """
    skip_reasons: list[str] = []
    try:
        from plugins.dashboard_auth import nous as _nous_plugin

        if _nous_plugin.LAST_SKIP_REASON:
            skip_reasons.append(f"  • nous: {_nous_plugin.LAST_SKIP_REASON}")
    except Exception:
        pass

    if host in _LOOPBACK_HOST_VALUES:
        public_url = ""
        try:
            from hermes_cli.dashboard_auth.prefix import resolve_public_url

            public_url = resolve_public_url()
        except Exception:
            pass
        gate_reason = (
            f"dashboard.public_url is set to "
            f"{public_url or '<a non-loopback URL>'} — an "
            f"operator-declared external URL engages the auth gate "
            f"even on a loopback bind"
        )
        fix_hint = (
            "If this dashboard should be LOCAL-ONLY (no reverse "
            "proxy), remove dashboard.public_url from config.yaml "
            "(and unset HERMES_DASHBOARD_PUBLIC_URL) to restore the "
            "unauthenticated loopback mode.\n"
        )
    else:
        gate_reason = f"the auth gate engages on non-loopback binds ({host})"
        fix_hint = ""

    fix_hint += (
        "Configure an auth provider before exposing the dashboard:\n"
        "  • Password: set dashboard.basic_auth.username + "
        "password_hash in config.yaml\n"
        "    (hash with: python -c \"from "
        "plugins.dashboard_auth.basic import hash_password; "
        "print(hash_password('your-password'))\")\n"
        "  • OAuth: run `hermes dashboard register` (Nous Portal) or "
        "install a DashboardAuthProvider plugin.\n"
        "There is no unauthenticated public-dashboard option. For "
        "local-only use, bind 127.0.0.1 and leave dashboard.public_url "
        "unset; a configured external public URL requires auth even "
        "when a local reverse proxy reaches a loopback backend."
    )
    # Credentials exist but the bundled provider is disabled (#54489). Basic
    # auth needs a username AND a credential; a half-configured block is silent.
    try:
        from hermes_cli.config import load_config as _load_cfg
        from hermes_cli.plugins_cmd import _BASIC_AUTH_PLUGIN_KEYS

        cfg = _load_cfg()
        ba = (cfg.get("dashboard") or {}).get("basic_auth") or {}
        disabled = (cfg.get("plugins") or {}).get("disabled") or []
        has_creds = bool(ba.get("username")) and bool(ba.get("password_hash") or ba.get("password"))
        if has_creds and (set(disabled) & _BASIC_AUTH_PLUGIN_KEYS):
            fix_hint = (
                "The 'basic' dashboard-auth plugin is in "
                "plugins.disabled but dashboard.basic_auth is "
                "configured.\n"
                "Remove 'basic' from plugins.disabled (or run "
                "`hermes plugins enable basic`), then restart the "
                "dashboard.\n\n"
            ) + fix_hint
    except Exception:
        pass
    msg = (
        f"Refusing to bind dashboard to {host} — {gate_reason}, "
        f"but no auth providers are registered.\n\n"
    )
    if skip_reasons:
        msg += "Bundled providers reported these issues:\n" + "\n".join(skip_reasons) + "\n\n"
    return msg + fix_hint


def _configure_auth_gate(
    host: str,
    allow_public: bool,
    ssh_session_token: Optional[str],
    ssh_owner_nonce: Optional[str],
) -> None:
    """Resolve the trusted public hosts + auth-gate flag onto ``app.state``.

    Fails closed (``SystemExit`` with an actionable message) when the gate
    engages but no dashboard auth provider is registered.
    """
    # dashboard.public_url is also the exact Host/Origin trust declaration for
    # reverse-proxy deployments; resolved once so middleware never reloads
    # config. A non-loopback public hostname engages the gate even on a loopback
    # backend, else the SPA's local session token becomes remotely reachable.
    app.state.trusted_public_hosts = _dashboard_public_hosts()
    # auth_required drives middleware, SPA-token injection, WS auth, the
    # startup refusal, the gate-on banner and uvicorn proxy_headers.
    if _desktop_loopback_auth_exempt(host, ssh_session_token, ssh_owner_nonce):
        # public_url describes the operator's PUBLIC deployment, not this
        # Desktop-owned loopback backend (#96490), which authenticates with the
        # per-spawn session token the ticket-only gate would refuse.
        app.state.auth_required = should_require_auth(host)
        _log.info(
            "Desktop-owned loopback backend: dashboard.public_url does not "
            "engage the ticket gate for this process; the public deployment "
            "keeps its own gate.",
        )
    else:
        app.state.auth_required = should_require_dashboard_auth(host, app.state.trusted_public_hosts)

    # ``--insecure`` no longer disables the gate (June 2026 hermes-0day
    # hardening); warn that it is a no-op rather than silently ignore it.
    if allow_public and host not in _LOOPBACK_HOST_VALUES:
        _log.warning(
            "--insecure no longer bypasses dashboard authentication. A "
            "non-loopback bind (%s) now ALWAYS requires an auth provider "
            "(OAuth or the bundled password provider). Configure one — see "
            "below — or bind to 127.0.0.1 and reach it over an SSH tunnel / "
            "Tailscale.", host,
        )

    if app.state.auth_required:
        # No escape hatch serves a gated dashboard without a provider.
        from hermes_cli.dashboard_auth import list_providers
        if not list_providers():
            raise SystemExit(_no_auth_provider_message(host))
        _log.info(
            "Dashboard binding to %s with auth gate enabled. Providers: %s",
            host,
            ", ".join(p.name for p in list_providers()),
        )


def _build_uvicorn_server(host: str, port: int):
    """Build the uvicorn ``Config`` + ``Server`` for this bind (reads ``app.state.auth_required``).

    uvicorn.Server is driven directly (not uvicorn.run) so startup is split from
    the main loop: after startup() the socket is bound and held by uvicorn, so the
    OS-assigned port can be read with no pre-bind-then-close TOCTOU. Explicit
    taken ports are caught by the #93608 preflight probe; uvicorn's own bind
    error stays the fallback for races.
    """
    import uvicorn

    # WS keepalive ping runs ON the agent event loop; a GIL-holding worker call
    # can starve it for minutes, so uvicorn misses the pong and drops a healthy
    # local socket (#53773/#48445/#50005). The ping only detects half-open
    # connections (proxy 524, dropped tunnels), impossible on loopback where a
    # dead client sends a real FIN/RST -> WebSocketDisconnect. So: no ping on
    # loopback; non-loopback sits behind a Cloudflare Tunnel (~100s idle) and
    # keeps a config-driven cadence (dashboard.ws_ping_interval/_timeout,
    # #79635) defaulting to 20/20.
    _is_loopback = host in _LOOPBACK_HOST_VALUES
    try:
        _dash_cfg = load_config().get("dashboard") or {}
    except Exception:
        _dash_cfg = {}

    def _ws_ping_setting(key: str, default: float = 20.0) -> float:
        try:
            return float(_dash_cfg.get(key, default))
        except (TypeError, ValueError):
            return default

    config = uvicorn.Config(
        app, host=host, port=port, log_level="warning",
        # Off by default so _ws_client_is_allowed sees the real peer, not
        # X-Forwarded-For. Gated mode runs behind a TLS terminator and needs
        # X-Forwarded-Proto for cookie Secure flags.
        proxy_headers=bool(app.state.auth_required),
        # Loopback-only unless the operator trusts a bounded upstream proxy, so
        # spoofed X-Forwarded-* from arbitrary callers is never honoured.
        forwarded_allow_ips=_dashboard_forwarded_allow_ips(_dash_cfg),
        ws_ping_interval=None if _is_loopback else _ws_ping_setting("ws_ping_interval"),
        ws_ping_timeout=None if _is_loopback else _ws_ping_setting("ws_ping_timeout"),
        ws_max_size=_DESKTOP_ATTACHMENT_WS_MAX_BYTES,
    )
    return config, uvicorn.Server(config)


def _best_effort(what: str, fn) -> None:
    """Run a best-effort startup step; any failure (import included) is a debug line."""
    try:
        fn()
    except Exception as exc:
        _log.debug("%s skipped: %s", what, exc)


def _on_server_started(
    server,
    *,
    host: str,
    port: int,
    headless: bool,
    open_browser: bool,
    initial_profile: str,
    start_mcp_discovery_after_bind: bool,
) -> None:
    """Post-bind arming on the serving loop right after ``server.startup()``.

    Reap prior corpses, parent-death watchdog, process identity, READY
    announcement, browser open, deferred MCP discovery, loop-noise filter,
    loop heartbeat.
    """
    # Clear corpses from a previous unclean Desktop exit (crash/SIGKILL/update
    # handoff leaves an orphaned backend + its MCP subtree) before stacking a
    # new tree (EMFILE / missing tabs). The watchdog only protects *this*
    # process going forward.
    def _reap_desktop_serves() -> None:
        from hermes_cli.dashboard_procs import _reap_orphaned_desktop_local_serves

        _reap_orphaned_desktop_local_serves()

    def _reap_mcp_helpers() -> None:
        from hermes_cli.process_identity import reap_orphaned_mcp_helpers

        reap_orphaned_mcp_helpers()

    if os.getenv("HERMES_DESKTOP") == "1":
        _best_effort("orphan desktop-local serve reap", _reap_desktop_serves)
    # Same sweep for stdio MCP helpers (#61514): positive identity only (spawn
    # ledger + spawner provably dead); anything alive or unprovable is untouched.
    _best_effort("orphan MCP helper reap", _reap_mcp_helpers)

    # No-op for standalone `hermes serve` (no HERMES_PARENT_PID).
    _start_parent_death_watchdog()

    actual_port = _read_bound_port(server, fallback=port)
    app.state.bound_port = actual_port

    # Positive process identity in the machine spawn ledger (+ Windows
    # kill-on-close job). Registered AFTER the bind so the entry carries the
    # ACTUAL port — what lets `hermes update` relaunch a manually-started serve
    # on its real endpoint (#63206).
    def _register_identity() -> None:
        from hermes_cli.process_identity import attach_self_to_kill_on_close_job, register_self

        register_self(
            "serve" if headless else "dashboard",
            detail={"host": host, "port": actual_port, "profile": initial_profile or ""},
        )
        attach_self_to_kill_on_close_job()

    _best_effort("process-identity registration", _register_identity)

    _write_dashboard_ready_file(actual_port)
    # Port-discovery sentinel parsed by the Desktop spawn (matches either
    # token). Written to fd 1: tui_gateway.server redirects sys.stdout to
    # stderr at import, and the Desktop watches child.stdout (#96282).
    ready_token = "HERMES_BACKEND_READY" if headless else "HERMES_DASHBOARD_READY"
    _write_machine_sentinel_line(f"{ready_token} port={actual_port}")
    if headless:
        # Auth-gated JSON-RPC/WS only — announce the bind, not a URL. flush:
        # a piped stdout otherwise surfaces this minutes after the sentinel.
        print(f"  Hermes backend listening on {host}:{actual_port}", flush=True)
    else:
        print(f"  Hermes Web UI → http://{host}:{actual_port}")
    _maybe_open_browser(host, actual_port, open_browser, initial_profile)

    if start_mcp_discovery_after_bind:
        # Desktop `serve`: the ~350ms `mcp` SDK import holds the GIL while the
        # renderer does its WS handshake + first hydration reads, so arm it one
        # second later when the shell is painted and idle. An agent build inside
        # that second fires the deferred start itself (wait_for_mcp_discovery).
        try:
            from hermes_cli.mcp_startup import defer_background_mcp_discovery

            defer_background_mcp_discovery(
                logger=_log,
                thread_name="dashboard-mcp-discovery",
                delay=_DESKTOP_MCP_DISCOVERY_DELAY_S,
            )
        except Exception:
            _log.debug("Deferred MCP discovery arm failed", exc_info=True)

    # Collapse the peer-hangup teardown flood (#50005): 50+ identical WinError
    # 10054 tracebacks per Desktop disconnect become one debug line.
    def _install_noise_filter() -> None:
        from tui_gateway.loop_noise import install_loop_noise_filter

        install_loop_noise_filter(asyncio.get_running_loop())

    _best_effort("loop noise filter install", _install_noise_filter)

    # Loop heartbeat watchdog (CF-1): a 2s call_later tick whose drift equals
    # any GIL stall, so a stalled-loop WS drop is diagnosable from the log.
    # call_later (not a task) dies with the loop — nothing to cancel.
    _hb_interval = 2.0
    _hb_stall_threshold = 5.0
    _hb_loop = asyncio.get_running_loop()

    def _loop_heartbeat(expected: float) -> None:
        now = _hb_loop.time()
        drift = now - expected
        if drift > _hb_stall_threshold:
            _log.warning("event loop stalled %.1fs (GIL pressure suspected)", drift)
        _hb_loop.call_later(_hb_interval, _loop_heartbeat, now + _hb_interval)

    _hb_loop.call_later(_hb_interval, _loop_heartbeat, _hb_loop.time() + _hb_interval)


def _run_serve(serve, config, host: str, port: int) -> None:
    """Drive ``serve()`` on the loop uvicorn expects.

    POSIX keeps ``asyncio.run`` (already a SelectorEventLoop / uvloop). On
    Windows ``asyncio.run`` defaults to a ProactorEventLoop, on which uvicorn
    binds a socket that never accepts (#50641), so mirror uvicorn's own runner +
    loop factory there (hand-installed selector policy for uvicorn < 0.36).
    Ctrl+C -> clean return; probe-to-bind port race -> sentinel + exit code.
    """
    runner = asyncio.run
    runner_kwargs: dict = {}
    if sys.platform == "win32":
        # Resolved FIRST; the serve call is outside this try so genuine
        # serve-time errors (port in use) propagate instead of double-running.
        try:
            from uvicorn._compat import asyncio_run as runner

            runner_kwargs = {"loop_factory": config.get_loop_factory()}
        except Exception:
            runner = asyncio.run
            runner_kwargs = {}
            try:
                asyncio.set_event_loop_policy(
                    asyncio.WindowsSelectorEventLoopPolicy()  # type: ignore[attr-defined]
                )
            except Exception:
                pass

    # ``capture_signals()`` re-raises the captured signal after graceful
    # shutdown; console Ctrl+C lands as KeyboardInterrupt = clean exit.
    # (Re-raised SIGTERM/SIGBREAK keep their terminate disposition.)
    try:
        runner(serve(), **runner_kwargs)
    except KeyboardInterrupt:
        return
    except SystemExit as exc:
        # Probe-to-bind race (#93608): uvicorn's bind_socket() exits 1 — re-check
        # and translate a confirmed conflict into the sentinel + distinct code.
        if exc.code == 1 and _port_bind_conflict(host, port):
            _report_port_in_use(host, port)
            raise SystemExit(PORT_IN_USE_EXIT_CODE) from None
        raise


def start_server(
    host: str = "127.0.0.1",
    port: int = 9119,
    open_browser: bool = True,
    allow_public: bool = False,
    initial_profile: str = "",
    headless: bool = False,
    ssh_session_token: Optional[str] = None,
    ssh_owner_nonce: Optional[str] = None,
    start_mcp_discovery_after_bind: bool = False,
):
    """Start the web UI server.

    ``initial_profile`` is appended to the auto-opened URL as ``?profile=<name>``
    (profile alias ``<profile> dashboard``). ``headless`` is the ``serve`` path:
    JSON-RPC/WS backend, no UI build, no SPA mount (``HERMES_SERVE_HEADLESS``).
    ``ssh_session_token``/``ssh_owner_nonce`` are process-local Desktop SSH
    bootstrap state, never persisted or exported to children.
    ``start_mcp_discovery_after_bind`` (Desktop ``serve``) defers MCP discovery
    until the ready sentinel is written so its SDK import can't hold the GIL
    against the pre-bind path.
    """
    _apply_ssh_session_token(ssh_session_token or "")
    _apply_ssh_owner_nonce(ssh_owner_nonce)

    # Dashboard-mode starts don't route through main.py's `serve` path, which
    # applies the same RLIMIT_NOFILE floor (policy in resource_limits, #81547).
    from hermes_cli.resource_limits import apply_nofile_soft_limit

    apply_nofile_soft_limit()

    import uvicorn  # noqa: F401 — fail fast (before any side effects) when the dashboard extra is missing

    try:
        from hermes_cli.nous_auth_keepalive import start_nous_auth_keepalive

        start_nous_auth_keepalive()
    except Exception as exc:
        _log.debug("Nous auth keepalive did not start: %s", exc)

    _configure_auth_gate(host, allow_public, ssh_session_token, ssh_owner_nonce)

    # host_header_middleware validates Host against this (DNS rebinding,
    # GHSA-ppp5-vxwm-4cf7).
    app.state.bound_host = host

    config, server = _build_uvicorn_server(host, port)

    # Flush-on-kill guard (#94724): chaining SIGTERM/SIGINT handlers persist
    # in-memory transcripts to state.db before shutdown. Installed BEFORE
    # uvicorn's capture_signals() so uvicorn re-raises into them as the
    # "original" handlers — kills outside the serve window are covered too.
    try:
        from tui_gateway.server import install_exit_flush_signal_handlers

        install_exit_flush_signal_handlers()
    except Exception as exc:
        _log.debug("exit-flush signal handlers not installed: %s", exc)

    # #93608: uvicorn's bind_socket() would exit 1 with a bare ERROR line,
    # indistinguishable from "backend broken". Probe first so a conflict
    # surfaces as the BACKEND_PORT_IN_USE sentinel + distinct exit code.
    # ``--port 0`` is skipped by the probe.
    if _port_bind_conflict(host, port):
        _report_port_in_use(host, port)
        raise SystemExit(PORT_IN_USE_EXIT_CODE)

    async def _serve():
        # startup split from main_loop so the bound (ephemeral) port is readable.
        if not config.loaded:
            config.load()
        server.lifespan = config.lifespan_class(config)
        with server.capture_signals():
            await server.startup()
            if server.should_exit:
                return

            _on_server_started(
                server,
                host=host,
                port=port,
                headless=headless,
                open_browser=open_browser,
                initial_profile=initial_profile,
                start_mcp_discovery_after_bind=start_mcp_discovery_after_bind,
            )

            await server.main_loop()
            if server.started:
                await server.shutdown()

    _run_serve(_serve, config, host, port)
