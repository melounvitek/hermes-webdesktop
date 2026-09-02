"""
Hermes Agent — Web UI server.

Provides a FastAPI backend serving the Vite/React frontend and REST API
endpoints for managing configuration, environment variables, and sessions.

Usage:
    python -m hermes_cli.main web          # Start on http://127.0.0.1:9119
    python -m hermes_cli.main web --port 8080
"""

from contextlib import asynccontextmanager

import asyncio
from collections import deque
import hashlib
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
from typing import Any, Dict, List, Optional, Tuple


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
    from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
    from fastapi.staticfiles import StaticFiles
    from starlette.concurrency import run_in_threadpool
except ImportError:
    # First try lazy-installing the dashboard extras. Only the user actually
    # running `hermes dashboard` needs fastapi+uvicorn; lazy install keeps
    # them out of every other install path. After install, re-import.
    try:
        from tools.lazy_deps import ensure as _lazy_ensure
        _lazy_ensure("tool.dashboard", prompt=False)
        from fastapi import (
            FastAPI, HTTPException, Request,
        )
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import JSONResponse
    except Exception:
        raise SystemExit(
            "Web UI requires fastapi and uvicorn.\n"
            f"Install with: {sys.executable} -m pip install 'fastapi' 'uvicorn[standard]'"
        )

WEB_DIST = Path(os.environ["HERMES_WEB_DIST"]) if "HERMES_WEB_DIST" in os.environ else Path(__file__).parent / "web_dist"
_log = logging.getLogger(__name__)


from hermes_cli.web_server_lifecycle import (  # noqa: E402,F401 — re-exported; routers/tests reach these via web_server.<name>
    _process_start_marker,
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


# ---------------------------------------------------------------------------
# Per-channel subscriber registry used by /api/pub (PTY-side gateway → dashboard)
# and /api/events (dashboard → browser sidebar).  Keyed by an opaque channel id
# the chat tab generates on mount; entries auto-evict when the last subscriber
# drops AND the publisher has disconnected.
#
# State lives on app.state (not module-level globals) so that asyncio.Lock is
# created on the running event loop during lifespan startup.  A module-level
# asyncio.Lock() binds to whatever loop was active at import time, which breaks
# when the same module is used across TestClient instances or uvicorn reloads.
# ---------------------------------------------------------------------------

def _start_desktop_cron_ticker(stop_event: "threading.Event", interval: int = 60) -> None:
    """Tick the cron scheduler from inside the desktop dashboard backend.

    The scheduler tick loop normally lives in ``hermes gateway run`` — but the
    desktop app spawns a ``hermes dashboard`` backend, not a gateway, so a cron
    a user creates in the app would never fire. We run the resolved cron
    scheduler provider here (no live adapters; delivery falls back to the
    per-platform send path).

    Every local profile's store is ticked, not just this backend's own
    (#69377's desktop sibling): the desktop pools per-profile backends and
    reaps them after ~10 idle minutes, so a secondary profile's ticker dies
    with its backend and that profile's jobs silently stop firing until the
    user next opens it ("tasks on the sleeping profile could be idle" —
    community report, Aug 2026). The primary backend outlives the pool, so it
    owns every profile's tick, exactly like a multiplex gateway. External
    providers keep the single-store behavior — their registries are not
    profile-scoped (see _notify_cron_provider_for_profile).

    Cross-process safe: the built-in provider's ``cron.scheduler.tick`` takes
    the per-store ``cron/.tick.lock`` file lock, so this never double-fires
    alongside a real gateway or a live pool backend on the same profile home —
    whichever process grabs the lock first wins the tick.
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
                # Stand down, per tick, for any profile whose OWN gateway is
                # running: that gateway ticks it with live adapters, and the
                # tick-lock race otherwise lets this adapter-less ticker win
                # and deliver the job through the standalone path (#100489).
                # Evaluated every cycle so a gateway starting/stopping later
                # is picked up without a dashboard restart.
                from hermes_cli.profiles import _check_gateway_running

                start_kwargs["profile_gate"] = (
                    lambda _name, home: not _check_gateway_running(Path(home))
                )
                from hermes_logging import enable_profile_log_routing

                enable_profile_log_routing(profile_homes)
                _log.info(
                    "Desktop cron scheduler will tick %d profile(s): %s",
                    len(profile_homes),
                    [name for name, _home in profile_homes],
                )
        except Exception:
            # Fail open to the single-store ticker — the active profile's
            # jobs must keep firing even if profile enumeration breaks.
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
    # Serializes chat-argv resolution so concurrent /api/pty connections
    # don't trigger overlapping ``npm install`` / ``npm run build`` work.
    # On app.state (not a module global) so the Lock binds to the running
    # event loop during lifespan startup — see _get_event_state's docstring.
    app.state.chat_argv_lock = asyncio.Lock()

    # Bring this profile's state.db schema current BEFORE the first
    # session-list poll (#79531/#80037). Migrations used to run lazily on
    # the first writable open — typically the user's first new session —
    # so a store left behind by `hermes update` kept 500ing every
    # /api/sessions poll (and the read-probe heal, while it retries per
    # poll, can lose repeatedly to lock contention from orphaned sibling
    # backends). One writable open here runs _init_schema →
    # _reconcile_columns with the full open-time lock patience. Runs in a
    # daemon thread so a locked store never delays the server socket (the
    # Desktop ready-probe times out at 10s, GH-73083); reads that land
    # before it finishes are still covered by the read-probe heal.
    threading.Thread(
        target=_eager_reconcile_own_session_db,
        daemon=True,
        name="statedb-eager-reconcile",
    ).start()

    # Import hermes_cli.gateway eagerly *before* the lifespan yield so the
    # GIL-heavy .pyc compilation and Defender scan cost is absorbed during
    # backend initialisation — before the server socket accepts probes.
    # On Windows + Python 3.11 the import does not release the GIL, so
    # run_in_executor still froze the event loop for 15-22 s, causing the
    # Desktop's 10-second WebSocket ready-probe to time out (GH-73083).
    _warm_gateway_module()

    # Snapshot the checkout revision at boot so risky lazy-import paths (the
    # model picker) can detect when `hermes update` replaced the code
    # underneath this long-lived process and refuse with a clear "restart
    # required" message instead of a stale-module ImportError (#86207).  This
    # mirrors the gateway's record_boot_fingerprint in gateway/run.py; the
    # dashboard is a separate process/unit that the update flow does not
    # reliably restart, so it must detect the drift itself.
    from gateway.code_skew import record_boot_fingerprint

    record_boot_fingerprint()

    # Hosted Bot rooms belong to the backend process, not to any connected
    # Desktop socket. Recovery may need a contended state.db migration, so keep
    # it off the lifespan's pre-yield path: Group Chat startup must degrade on
    # its own instead of preventing every dashboard/Desktop feature from booting.
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
        # Before forking a fresh gateway, reap any orphan left by a previous
        # serve session. Graceful shutdown reaps the managed child, but an
        # abnormal exit (crash, SIGKILL, power loss, forced update) reparents
        # the old gateway to launchd (PPID=1). It keeps holding the QQ
        # WebSocket, and a newly forked gateway then races the same credential,
        # splitting messages across parallel session trees (#77276).
        #
        # The sweep itself still runs unconditionally — a stale-but-present
        # registration must not veto the #77276 orphan reap. Protection for
        # a healthy standalone gateway (launched via `hermes gateway run`,
        # no service supervisor) lives INSIDE the reaper: it probes the
        # registration with cleanup_stale=False so the recorded PID always
        # joins the exclusion set, even when liveness validation would have
        # unlinked the record mid-sweep. That matters most on Windows, where
        # every layer of the launcher chain (stub -> venv python -> runtime
        # python) carries "gateway run" in its command line, so
        # find_gateway_pids() matches processes the pidfile exclusion cannot
        # see, and os.kill(SIGTERM) is a hard TerminateProcess — the
        # gateway's planned-stop watcher (0.5s poll) has no time to drain.
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

    # Reap idle/dead keep-alive PTY sessions in the background (30-min TTL).
    pty_reaper_task = asyncio.create_task(run_reaper(PTY_REGISTRY))

    # Periodic authenticated self-test (feeds the ``dashboard`` component on
    # /api/status).  The loop exits immediately when httpx is unavailable.
    selftest_task = asyncio.create_task(_dashboard_selftest_loop())

    # Live auto-archive timer — keeps a backend that stays up for days
    # sweeping stale sessions on schedule, independent of list requests.
    auto_archive_task = asyncio.create_task(_auto_archive_ticker_loop())

    # Managed local runtime: when the user opted in (local_runtime.enabled,
    # set by the Local Models 'Use' action), bring the llama-server back up
    # so a restart doesn't strand a llamacpp main model without a backend.
    # Off-thread and best-effort: binary check + spawn + health poll must
    # not delay the server socket, and failure falls back to configured
    # cloud providers exactly like a cold start.
    def _boot_local_runtime():
        try:
            from hermes_cli.config import load_config
            from hermes_cli.local_runtime.bootstrap import ensure_local_runtime

            # Server only — models load on first inference, always (residency
            # design: downloaded = available; demand loads; idleness
            # evicts). An empty router holds no VRAM; warming a model at
            # boot would reload gigabytes nobody asked for yet.
            ensure_local_runtime(load_config())
        except Exception as exc:  # noqa: BLE001
            logging.getLogger(__name__).warning("local runtime boot failed: %s", exc)

    threading.Thread(target=_boot_local_runtime, daemon=True,
                     name="local-runtime-boot").start()

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
        # Stop the managed llama-server with its parent — a supervisor-less
        # orphan would keep VRAM pinned after the app closes.
        try:
            from hermes_cli.local_runtime.bootstrap import shutdown_local_runtime

            shutdown_local_runtime()
        except Exception:  # noqa: BLE001
            pass
        if os.getenv("HERMES_DESKTOP") == "1":
            _terminate_desktop_managed_gateway()


def _get_chat_argv_lock(app: "FastAPI") -> asyncio.Lock:
    """Return the chat-argv resolution lock from app.state.

    Mirrors :func:`_get_event_state`: prefers the lifespan-initialised Lock
    (created on the correct event loop) but lazily initialises it for
    non-``with`` TestClient usages.
    """
    try:
        return app.state.chat_argv_lock
    except AttributeError:
        app.state.chat_argv_lock = asyncio.Lock()
        return app.state.chat_argv_lock


def _get_pty_active_session_files(app: "FastAPI") -> dict[str, Path]:
    """Return channel -> active-session-file state for dashboard PTYs."""
    try:
        return app.state.pty_active_session_files
    except AttributeError:
        app.state.pty_active_session_files = {}
        return app.state.pty_active_session_files


app = FastAPI(title="Hermes Agent", version=__version__, lifespan=_lifespan)


# Memory-provider OAuth connect routes live in the memory layer, not here.
from hermes_cli.memory_oauth import router as _memory_oauth_router  # noqa: E402

app.include_router(_memory_oauth_router)

# ---------------------------------------------------------------------------
# Session token for protecting sensitive endpoints (reveal).
# The desktop shell mints the token and injects it via
# HERMES_DASHBOARD_SESSION_TOKEN so its main process can authenticate the
# /api calls it makes on the user's behalf; otherwise we generate one fresh
# on every server start. Either way it dies when the process exits and is
# injected into the SPA HTML so only the legitimate web UI can use it.
# ---------------------------------------------------------------------------


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
        # Primary identity: a marker FILE written into site-packages now.
        # A replaced venv (rm -rf && recreate — same OR different Python
        # version) loses the marker deterministically, while pip installs
        # into the live venv leave it untouched (no false stales). A bare
        # (dev, ino) snapshot of the directory is NOT sufficient on its
        # own: ext4 reuses directory inodes immediately, so the exact
        # reported repro (`rm -rf venv && uv venv`) can land on the same
        # inode and pass undetected (proven live during salvage).
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
    # Marker file is the deterministic signal when we managed to write one.
    if _SSH_RUNTIME_MARKER is not None:
        return os.path.isfile(_SSH_RUNTIME_MARKER)
    # Fallback (read-only site-packages): directory identity snapshot.
    # Weaker — inode reuse can mask a same-filesystem recreate — but still
    # catches cross-device moves and version-bump path changes.
    if _SSH_RUNTIME_PURELIB is None:
        return True
    purelib, device, inode = _SSH_RUNTIME_PURELIB
    try:
        st = os.stat(purelib)
    except OSError:
        return False
    return (st.st_dev, st.st_ino) == (device, inode)

# In-browser Chat tab (/chat, /api/pty, /api/ws, …).  Always enabled: the
# desktop app and the dashboard's own Chat tab both drive the agent over the
# `/api/ws` + `/api/pty` WebSockets, so the embedded-chat surface is an
# unconditional part of the dashboard.  Kept as a module-level constant (rather
# than inlining ``True`` at every gate) so the WS endpoints and the SPA token
# injection share a single, testable seam.
_DASHBOARD_EMBEDDED_CHAT_ENABLED = True

# Desktop's file.attach compatibility transport sends a complete base64 data
# URL in one JSON-RPC frame. Uvicorn defaults to 16 MiB, which rejects files at
# the preview ceiling before the dispatcher sees them. Keep the gateway
# finite while allowing the 256 MiB raw Desktop attach cap plus base64/JSON
# overhead.
_DESKTOP_ATTACHMENT_WS_MAX_BYTES = 384 * 1024 * 1024


# CORS: restrict to localhost origins only.  The web UI is intended to run
# locally; binding to 0.0.0.0 with allow_origins=["*"] would let any website
# read/modify config and secrets.

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Endpoints that do NOT require the session token.  Everything else under
# /api/ is gated by the auth middleware below.
#
# This list is defined in ``hermes_cli.dashboard_auth.public_paths`` so the
# OAuth gate middleware can honour the same allowlist — keeping the two
# gates in lockstep avoids drift like the wildcard-subdomain regression
# where ``/api/status`` was public under the legacy gate but 401'd under
# the OAuth gate (breaking the portal's liveness probe).
#
# Keep the upstream list minimal — only truly non-sensitive, read-only
# endpoints belong there.
# ---------------------------------------------------------------------------
from hermes_cli.dashboard_auth.public_paths import (
    PUBLIC_API_PATHS as _PUBLIC_API_PATHS,
)


def _has_valid_session_token(request: Request) -> bool:
    """True if the request carries a valid dashboard session token.

    The dedicated session header avoids collisions with reverse proxies that
    already use ``Authorization`` (for example Caddy ``basic_auth``). We still
    accept the legacy Bearer path for backward compatibility with older
    dashboard bundles.
    """
    session_header = request.headers.get(_SESSION_HEADER_NAME, "")
    if session_header and hmac.compare_digest(
        session_header.encode(),
        _SESSION_TOKEN.encode(),
    ):
        return True

    auth = request.headers.get("authorization", "")
    expected = f"Bearer {_SESSION_TOKEN}"
    return hmac.compare_digest(auth.encode(), expected.encode())


# Routes that may also authenticate via a ``?token=`` query param, for download
# links opened by the OS shell or a new browser tab where the session header
# can't be set. Kept narrow — same query-token tradeoff as the /api/pty WS.
_QUERY_TOKEN_API_PATHS: frozenset[str] = frozenset({"/api/files/download"})


def _has_valid_query_token(request: Request, path: str) -> bool:
    if path not in _QUERY_TOKEN_API_PATHS:
        return False
    token = request.query_params.get("token", "")
    return bool(token) and hmac.compare_digest(token.encode(), _SESSION_TOKEN.encode())


def _require_token(request: Request) -> None:
    """Authorize a sensitive endpoint, raising 401 if the caller isn't allowed.

    Two auth schemes protect the dashboard, exactly one active per bind:

    * **Loopback / ``--insecure`` mode** (``auth_required`` False): the
      ephemeral ``_SESSION_TOKEN`` is injected into the SPA HTML and echoed
      back via ``X-Hermes-Session-Token`` (or the legacy ``Bearer`` header).
      Validate it here.
    * **Gated / OAuth mode** (``auth_required`` True): ``_SESSION_TOKEN`` is
      NOT injected (the SPA authenticates with a session cookie), so there is
      no token to check. The ``gated_auth_middleware`` has already verified the
      cookie before the request reached this handler — any non-public ``/api/``
      route it lets through carries a verified ``request.state.session``. The
      legacy ``auth_middleware`` likewise short-circuits in this mode. Requiring
      the (absent) token here would 401 every cookie-authenticated request,
      making plugin install/enable/disable and the other ``_require_token``
      endpoints permanently unreachable behind the gate. Defer to the gate.
    """
    if getattr(request.app.state, "auth_required", False):
        # Gate is authoritative. It attaches ``request.state.session`` on
        # success and 401s otherwise, so a request that reached us is already
        # authenticated. Belt-and-braces: confirm the session is present.
        if getattr(request.state, "session", None) is not None:
            return
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not _has_valid_session_token(request):
        raise HTTPException(status_code=401, detail="Unauthorized")


# Accepted Host header values for loopback binds. DNS rebinding attacks
# point a victim browser at an attacker-controlled hostname (evil.test)
# which resolves to 127.0.0.1 after a TTL flip — bypassing same-origin
# checks because the browser now considers evil.test and our dashboard
# "same origin". Validating the Host header at the app layer rejects any
# request whose Host isn't one we bound for. See GHSA-ppp5-vxwm-4cf7.
_LOOPBACK_HOST_VALUES: frozenset = frozenset({
    "localhost", "127.0.0.1", "::1",
})


def _dashboard_public_hosts() -> frozenset[str]:
    """Return the exact hostname declared by ``dashboard.public_url``.

    ``public_url`` is already Hermes' canonical browser-facing URL behind a
    reverse proxy. Reusing its validated hostname here keeps OAuth redirects,
    HTTP Host validation, and WebSocket Origin validation on one source of
    truth. Malformed or unset values fail closed as an empty set.
    """
    from hermes_cli.dashboard_auth.prefix import resolve_public_url

    public_url = resolve_public_url()
    if not public_url:
        return frozenset()
    try:
        hostname = urllib.parse.urlparse(public_url).hostname
    except ValueError:
        return frozenset()
    if not hostname:
        return frozenset()
    return frozenset({hostname.lower()})


def should_require_auth(host: str, allow_public: bool = False) -> bool:
    """Return True iff the dashboard auth gate must be active.

    Truth table:
      host == loopback        → False (no auth — local-only, trusted operator)
      host != loopback        → True  (gate engages — OAuth or password required)

    "Loopback" is 127.0.0.1, localhost, ::1. RFC1918 / CGNAT / link-local are
    deliberately treated as PUBLIC — a hostile device on the same LAN is exactly
    the threat model the gate is designed for.

    ``allow_public`` (the legacy ``--insecure`` escape hatch) NO LONGER disables
    the gate. It is accepted for backward-compat with old launch scripts and
    desktop shells but is ignored: a non-loopback bind ALWAYS requires an auth
    provider (OAuth or the bundled password provider). This closes the
    unauthenticated-public-dashboard hole behind the June 2026 ``hermes-0day``
    MCP-persistence campaign, where ``--insecure --host 0.0.0.0`` left the
    config/MCP/agent surface open to internet scanners.
    """
    return host not in _LOOPBACK_HOST_VALUES


def should_require_dashboard_auth(
    host: str,
    trusted_public_hosts: Optional[frozenset[str]] = None,
) -> bool:
    """Return whether the dashboard auth gate must be active.

    The browser-facing URL is part of the exposure boundary: a non-loopback
    ``dashboard.public_url`` requires authentication even when a reverse proxy
    reaches a backend bound to loopback. Callers may pass the already-resolved
    host set so startup and request validation use the same snapshot.
    """
    if trusted_public_hosts is None:
        trusted_public_hosts = _dashboard_public_hosts()
    return should_require_auth(host) or any(
        candidate not in _LOOPBACK_HOST_VALUES
        for candidate in trusted_public_hosts
    )


def _desktop_loopback_auth_exempt(
    host: str,
    ssh_session_token: Optional[str] = None,
    ssh_owner_nonce: Optional[str] = None,
) -> bool:
    """True for a Desktop-owned loopback backend (#96490).

    A non-loopback ``dashboard.public_url`` engages the ticket-only auth gate
    for EVERY ``hermes serve`` on the machine — including the private loopback
    backends the Desktop app spawns for itself. Those backends authenticate
    with the per-spawn session token (injected via
    ``HERMES_DASHBOARD_SESSION_TOKEN`` for local spawns, ``--ssh-session-token
    -file``/``--ssh-owner-nonce`` for Desktop SSH), which the gate's WS path
    refuses outright — Desktop could not boot with a ``public_url`` configured.

    The public_url describes a DIFFERENT deployment: the actual public
    dashboard is a separate process on a non-loopback bind, whose own startup
    computes ``should_require_dashboard_auth`` from its host and stays gated.
    Exempting this process therefore never opens the public surface.

    Exemption requires ALL of: loopback bind, ``HERMES_DESKTOP=1`` (set by
    every Desktop spawn path — local and SSH), and an operator-minted
    credential (env token, SSH session token, or owner nonce). A plain
    ``hermes serve`` with ``HERMES_DESKTOP=1`` exported but no credential is
    NOT exempt.
    """
    if host not in _LOOPBACK_HOST_VALUES:
        return False
    if os.environ.get("HERMES_DESKTOP") != "1":
        return False
    return bool(
        os.environ.get("HERMES_DASHBOARD_SESSION_TOKEN")
        or ssh_session_token
        or ssh_owner_nonce
    )


def _host_header_hostname(host_header: str) -> str:
    """Return a normalized hostname from a valid HTTP Host authority.

    Host headers are authorities, not full URLs. Reject ambiguous ports,
    malformed IPv6 brackets, and URL syntax so validation always fails closed.
    """
    value = (host_header or "").strip()
    if not value:
        return ""
    if any(char in value for char in ('"', "'", "<", ">", " ", "\n", "\r", "\t")):
        return ""
    if "://" in value or any(char in value for char in ("/", "?", "#", "@")):
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

    if host_only in trusted_public_hosts:
        return True

    # 0.0.0.0 bind means operator explicitly opted into all-interfaces
    # (requires --insecure per web_server.start_server). No Host-layer
    # defence can protect that mode; rely on operator network controls.
    if bound_host in {"0.0.0.0", "::"}:
        return True

    # Loopback bind: accept the loopback names
    bound_lc = bound_host.lower()
    if bound_lc in _LOOPBACK_HOST_VALUES:
        return host_only in _LOOPBACK_HOST_VALUES

    # Explicit non-loopback bind: require exact host match
    return host_only == bound_lc


@app.middleware("http")
async def host_header_middleware(request: Request, call_next):
    """Reject requests whose Host header doesn't match the bound interface.

    Defends against DNS rebinding: a victim browser on a localhost
    dashboard is tricked into fetching from an attacker hostname that
    TTL-flips to 127.0.0.1. CORS and same-origin checks don't help —
    the browser now treats the attacker origin as same-origin with the
    dashboard. Host-header validation at the app layer catches it.

    See GHSA-ppp5-vxwm-4cf7.
    """
    # Store the bound host on app.state so this middleware can read it —
    # set by start_server() at listen time.
    bound_host = getattr(app.state, "bound_host", None)
    if bound_host:
        host_header = request.headers.get("host", "")
        trusted_public_hosts = getattr(
            app.state, "trusted_public_hosts", frozenset()
        )
        if not _is_accepted_host(
            host_header, bound_host, trusted_public_hosts
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

    :func:`_mount_plugin_api_routes` gates at import time, but if a plugin
    is disabled *after* the dashboard is already running, its FastAPI router
    remains mounted until restart.  This middleware enforces the enabled/
    disabled policy on every request to ``/api/plugins/{name}/...`` so that
    runtime config changes take effect immediately.

    Registered BEFORE the auth middlewares (so it executes AFTER them): a
    request that hasn't cleared auth must get auth's 401 first, never this
    gate's 404 — otherwise an unauthenticated caller could fingerprint which
    plugins are installed/enabled by reading the status code. We only reach
    the enabled/disabled check for a request that auth already let through.
    """
    path = request.url.path
    if path.startswith("/api/plugins/"):
        # Only gate authenticated requests. Unauthenticated ones fall
        # through so auth_middleware / the OAuth gate return 401 first and
        # this route can't be used as a plugin-name oracle.
        _authed = (
            getattr(request.state, "token_authenticated", False)
            or getattr(request.app.state, "auth_required", False)
            or _has_valid_session_token(request)
            or _has_valid_query_token(request, path)
        )
        if _authed:
            # Extract plugin name from /api/plugins/<name>/...
            parts = path.split("/")
            # parts: ['', 'api', 'plugins', '<name>', ...]
            if len(parts) >= 4:
                plugin_name = parts[3]
                if plugin_name:
                    try:
                        from hermes_cli.plugins_cmd import (
                            _get_enabled_set,
                            _get_disabled_set,
                        )
                        enabled_set = _get_enabled_set()
                        disabled_set = _get_disabled_set()
                    except Exception:
                        enabled_set = set()
                        disabled_set = set()
                    # Determine plugin source.  Check the cached plugin list;
                    # if not found, assume user plugin (safe default — blocks).
                    plugins = _get_dashboard_plugins()
                    plugin = next(
                        (p for p in plugins if p.get("name") == plugin_name),
                        None,
                    )
                    source = plugin.get("source") if plugin else "user"
                    if source == "user":
                        if plugin_name in disabled_set or plugin_name not in enabled_set:
                            return JSONResponse(
                                status_code=404,
                                content={"detail": "Plugin not found"},
                            )
                    elif source == "bundled":
                        if plugin_name in disabled_set:
                            return JSONResponse(
                                status_code=404,
                                content={"detail": "Plugin not found"},
                            )
    return await call_next(request)


# ---------------------------------------------------------------------------
# Dashboard OAuth auth gate — engaged only when start_server flags the
# bind as non-loopback-without-insecure.  No-op pass-through in loopback
# mode so the legacy auth_middleware (below) handles those binds via
# the injected ``_SESSION_TOKEN``.  Registered between host_header and
# auth_middleware so the order is: host check → cookie auth → token auth.
# ---------------------------------------------------------------------------


@app.middleware("http")
async def _dashboard_auth_gate(request: Request, call_next):
    from hermes_cli.dashboard_auth.middleware import gated_auth_middleware
    return await gated_auth_middleware(request, call_next)


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    """Require the session token on all /api/ routes except the public list."""
    # A request already authenticated by the token-auth seam (a service caller
    # presenting a bearer token on a registered token route) carries
    # ``token_authenticated`` — never bounce it through the cookie/session gate.
    if getattr(request.state, "token_authenticated", False):
        return await call_next(request)
    # When the OAuth gate is active, cookie-based auth (gated_auth_middleware
    # above) is authoritative.  The legacy _SESSION_TOKEN path is loopback-only
    # and is skipped here so the gate's session attachment isn't overridden.
    if getattr(request.app.state, "auth_required", False):
        return await call_next(request)
    path = request.url.path
    is_mcp_oauth_callback = path.startswith("/api/mcp/oauth/callback/")
    if path.startswith("/api/") and path not in _PUBLIC_API_PATHS and not is_mcp_oauth_callback:
        if not _has_valid_session_token(request) and not _has_valid_query_token(request, path):
            return JSONResponse(
                status_code=401,
                content={"detail": "Unauthorized"},
            )
    return await call_next(request)


@app.middleware("http")
async def _token_auth_seam(request: Request, call_next):
    """Outermost auth seam: non-interactive bearer-token auth for opted-in routes.

    Registered LAST so it runs FIRST (Starlette middleware is outermost-last).
    A registered token route is fully owned here — authenticate by token,
    attach the principal + ``token_authenticated`` flag, and let the downstream
    cookie/session gates skip enforcement. Non-token routes pass straight
    through untouched.
    """
    from hermes_cli.dashboard_auth.token_auth import token_auth_middleware
    return await token_auth_middleware(request, call_next)


# ---------------------------------------------------------------------------
# Dashboard component health — in-process error/self-test counters that feed
# the ``components`` dict on ``/api/status``.  That endpoint is in
# ``PUBLIC_API_PATHS``, so everything exported from here must be counts and
# enums only: no exception messages, no request paths, no tokens.
# ---------------------------------------------------------------------------

_DASHBOARD_HEALTH_WINDOW_SECONDS = 300.0


class DashboardHealth:
    """Module-level holder for dashboard-process health signals.

    Tracks unhandled exceptions / 5xx responses seen by the outermost HTTP
    middleware (rolling window) and the result of the periodic authenticated
    self-test.  ``last_error_path`` and ``last_error_type`` are internal
    diagnostics for logs/debuggers — :meth:`snapshot` deliberately exports
    neither (public-payload no-secrets contract).
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
    """Outermost middleware: count unhandled exceptions and 5xx responses.

    Registered after ``_token_auth_seam`` so it is the outermost layer
    (Starlette middleware is outermost-last) — nothing below can raise past
    it unseen.  Records into :data:`DASHBOARD_HEALTH` and re-raises; never
    swallows or alters the response.
    """
    try:
        response = await call_next(request)
    except Exception as exc:
        DASHBOARD_HEALTH.record_error(type(exc).__name__, request.url.path)
        raise
    if response.status_code >= 500:
        DASHBOARD_HEALTH.record_error(f"http_{response.status_code}", request.url.path)
    return response


# ---------------------------------------------------------------------------
# Authenticated-route self-test: every minute, make one in-process request
# against a cheap DB-touching authenticated route with the real session
# token.  Catches the class of failure where liveness looks fine but every
# authenticated request 500s (e.g. wedged state DB).
# ---------------------------------------------------------------------------

_DASHBOARD_SELFTEST_INTERVAL_SECONDS = 60.0
_DASHBOARD_SELFTEST_ROUTE = "/api/sessions?limit=1"


async def _dashboard_selftest_once() -> None:
    """Run one authenticated in-process self-test request and record it."""
    try:
        import httpx
    except ImportError:
        return  # optional dependency — skip cleanly, leave status "unknown"
    try:
        transport = httpx.ASGITransport(app=app)
        # base_url uses a loopback name so the Host-header middleware accepts
        # the request on loopback binds.
        async with httpx.AsyncClient(
            transport=transport, base_url="http://127.0.0.1"
        ) as client:
            resp = await client.get(
                _DASHBOARD_SELFTEST_ROUTE,
                headers={_SESSION_HEADER_NAME: _SESSION_TOKEN},
            )
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
        # On OAuth-gated binds the legacy session token is not honoured, so
        # the probe would false-alarm 401 — skip until the gate is off.
        if getattr(app.state, "auth_required", False):
            continue
        await _dashboard_selftest_once()


from hermes_cli.web_server_config import (  # noqa: E402,F401 — re-exported; routers/tests reach these via web_server.<name>
    CONFIG_SCHEMA,
    _AUX_TASK_SLOTS,
    _SCHEMA_OVERRIDES,
    _apply_main_model_assignment,
    _apply_model_assignment_sync,
    _build_schema_from_config,
    _dashboard_code_skew_guard,
    _denormalize_config_from_web,
    _memory_provider_options,
    _normalize_config_for_web,
    _normalize_main_model_assignment,
    _schema_with_dynamic_provider_options,
    _timezone_options,
)


from hermes_cli.web_models import (  # noqa: F401
    ConfigUpdate,
    EnvVarUpdate,
    EnvVarDelete,
    EnvVarReveal,
    MemoryProviderConfigUpdate,
    MemoryProviderSetupRequest,
    CustomEndpointUpdate,
    MessagingPlatformUpdate,
    TelegramOnboardingStart,
    TelegramOnboardingApply,
    WhatsAppOnboardingStart,
    WhatsAppOnboardingApply,
    AudioTranscriptionRequest,
    ManagedFileUpload,
    ChatImageUpload,
    ManagedDirectoryCreate,
    ManagedFileDelete,
    ModelAssignment,
    MoaModelSlot,
    _MoaReferenceControls,
    MoaPresetPayload,
    MoaConfigPayload,
    FsWriteText,
    GitPathBody,
    GitFileBody,
    GitCommitBody,
    GitWorktreeAddBody,
    GitWorktreeRemoveBody,
    GitBranchSwitchBody,
    CuratorPause,
    LearningNodeRef,
    LearningNodeEdit,
    DebugShareRequest,
    TTSSpeakRequest,
    TTSLeaseRequest,
    OAuthSubmitBody,
    BulkDeleteSessions,
    SessionImport,
    SessionRename,
    SessionPrune,
    CronJobCreate,
    CronJobUpdate,
    AutomationBlueprintInstantiate,
    MCPServerCreate,
    MCPServersReplace,
    MCPEnabledToggle,
    MCPCatalogInstall,
    PairingApprove,
    PairingRevoke,
    WebhookCreate,
    WebhookEnabledToggle,
    CredentialPoolAdd,
    MemoryProviderSelect,
    MemoryReset,
    BackupRequest,
    ImportRequest,
    HookCreate,
    HookDelete,
    SkillInstallRequest,
    SkillUninstallRequest,
    SkillsUpdateRequest,
    ProfileCreate,
    ProfileRename,
    ProfileSoulUpdate,
    ProfileActiveUpdate,
    ProfileDescriptionUpdate,
    ProfileModelUpdate,
    ProfileDescribeAuto,
    SkillToggle,
    SkillCreate,
    SkillContentUpdate,
    ToolsetToggle,
    ToolsetProviderSelect,
    ToolsetModelSelect,
    ToolsetEnvUpdate,
    ToolsetPostSetup,
    TerminalBackendSelect,
    RawConfigUpdate,
    ThemeSetBody,
    FontSetBody,
    _AgentPluginInstallBody,
    _PluginProvidersPutBody,
    _PluginVisibilityBody,
)


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


from hermes_cli.web_server_gateway import (  # noqa: E402,F401 — re-exported; routers/tests reach these via web_server.<name>
    _ACTION_COMMANDS,
    _ACTION_IDS,
    _ACTION_LOG_DIR,
    _ACTION_LOG_FILES,
    _ACTION_PROCS,
    _ACTION_RESULTS,
    _TOPOLOGY_CACHE,
    _TOPOLOGY_CACHE_TTL,
    _collect_profile_gateway_topology,
    _collect_profile_gateway_topology_cached,
    _dashboard_spawn_executable,
    _display_system_platform,
    _gateway_subcommand,
    _load_configured_gateway_platforms,
    _probe_gateway_health,
    _profile_gateway_writer_identity,
    _profile_platform_ports,
    _restart_gateway_after,
    _spawn_hermes_action,
    _split_text_for_speak_stream,
    _strip_session_list_rows,
    _terminate_desktop_managed_gateway,
)


from hermes_cli.web_server_files import (  # noqa: E402,F401 — re-exported; routers/tests reach these via web_server.<name>
    _canonical_path,
    _dashboard_local_update_managed_externally,
    _fs_path,
    _managed_file_entry,
    _managed_response_meta,
    _path_is_under,
    _resolve_managed_path,
)


_MANAGED_FILE_MAX_BYTES = 100 * 1024 * 1024


from hermes_cli.web_routers import files as _files_routes  # noqa: E402

app.include_router(_files_routes.router)
from hermes_cli.web_routers.files import (  # noqa: E402,F401 — legacy re-exports; tests call these via web_server.<name>
    get_media,
    upload_managed_file_stream,
)


_FS_DATA_URL_MAX_BYTES = 16 * 1024 * 1024


# Stream uploads to disk in fixed-size chunks. The legacy JSON endpoint above
# buffers the whole file as a base64 data URL in a JSON body, which (a) inflates
# the payload ~33%, (b) holds the entire file (plus its decoded copy) in memory,
# and (c) reliably trips upstream proxy body-size/timeout limits with a 502 on
# large backup archives (NS-501). This multipart endpoint reads the request body
# in 1 MiB chunks straight to a temp file, enforces the size cap as it goes, and
# atomically renames into place — constant memory, no base64 inflation.
_UPLOAD_CHUNK_BYTES = 1024 * 1024


from hermes_cli.web_routers import git as _git_routes  # noqa: E402

app.include_router(_git_routes.router)

from hermes_cli.web_routers import local_models as _local_models_routes  # noqa: E402

app.include_router(_local_models_routes.router)
from hermes_cli.web_routers.git import (  # noqa: E402,F401 — legacy re-exports; tests call these via web_server.<name>
    git_status_route,
    git_worktrees_route,
    git_branches_route,
    git_base_branches_route,
    git_review_list_route,
    git_review_diff_route,
    git_file_diff_route,
    git_commit_context_route,
    git_rev_parse_route,
    git_ship_info_route,
    git_stage_route,
    git_unstage_route,
    git_revert_route,
    git_commit_route,
    git_push_route,
    git_create_pr_route,
    git_worktree_add_route,
    git_worktree_remove_route,
    git_branch_switch_route,
)


# Stable install identity for /api/status. One random opaque id per physical
# install, minted on first read and persisted under the ROOT Hermes home
# (get_default_hermes_root()) — NOT the profile-scoped HERMES_HOME — so every
# profile served by the same install reports the same id. Clients (the desktop
# connection registry) use it to recognize that two registered addresses
# (hostname + Tailscale IP, LAN + WAN) are one backend and collapse duplicate
# roster rows. Privacy: uuid4 hex, no hardware/user-derived material; the only
# fact it reveals is "these addresses are the same box", which is the feature.
# It must never change across restarts/updates, so reads are cached for the
# process lifetime and the file is written once, atomically.
_INSTALL_ID_CACHE: Dict[str, Optional[str]] = {"root": None, "value": None}


def get_install_id() -> Optional[str]:
    """Process-lifetime-cached stable install id."""
    return _shared_get_install_id(cache=_INSTALL_ID_CACHE)


# Serializes read-modify-write cycles over config.yaml for handlers that run
# in worker threads (asyncio.to_thread). config.py's _CONFIG_LOCK covers each
# load_config()/save_config() call individually, not the span between them —
# when these handlers ran on the event loop the loop itself serialized the
# whole cycle, but off-loop two concurrent updates could interleave
# load→mutate→save and silently drop one another's writes. Held only in
# worker threads, so it can never block the event loop. RLock so a locked
# section that calls helpers which also take it can't self-deadlock.
_CONFIG_MUTATION_LOCK = threading.RLock()


from hermes_cli.web_routers import status as _status_routes  # noqa: E402

app.include_router(_status_routes.router)
from hermes_cli.web_routers.status import (  # noqa: E402,F401 — legacy re-exports; tests call these via web_server.<name>
    get_status,
    run_dump,
)


# A finished ``gateway-restart`` child does not mean the gateway is back: the
# child exits as soon as it has handed the restart to the supervisor (or to the
# running gateway), while the gateway itself is still stopping and coming up.
# The in-flight reuse in :func:`_spawn_gateway_restart` therefore stops
# coalescing exactly when repeat requests do the most damage, so a stale cached
# frontend that re-fires its restart every few seconds gets a brand new restart
# every time (#89034: 77 restarts, 17 of them inside one minute, killing the
# gateway often enough mid-FTS5-write to corrupt state.db).  Suppress repeats
# for a short window after the last spawn as well.
#
# MAINTAINER DECISION: a fixed window, not "until the gateway reports healthy".
# Health-gating is what #89034 asks for, but it cannot be made to fail safe
# here — a gateway that never comes back would leave the restart action
# permanently inert, which is a worse failure than the flood it prevents.  A
# fixed window always releases.  10s is above the ~3.5s spacing of the reported
# storm and below the time an operator waits before deliberately retrying.
GATEWAY_RESTART_COOLDOWN_SECONDS = 10.0

# ``(monotonic spawn time, Popen, command)`` for the last gateway restart this
# process started.  Deliberately NOT read out of ``_ACTION_PROCS``: entries
# there are reaped once the child exits, and a guard that disappears when the
# child exits is the bug this exists to fix.
_LAST_GATEWAY_RESTART: Optional[Tuple[float, subprocess.Popen, Tuple[str, ...]]] = None


def _spawn_gateway_restart(profile: Optional[str] = None) -> Tuple[subprocess.Popen, bool]:
    """Spawn ``hermes gateway restart``, reusing an in-flight restart.

    Multiple dashboard paths can request a restart in quick succession
    (restart button double-click, or a stale cached frontend firing its own
    restart after the server already auto-restarted post-onboarding). Two
    concurrent ``hermes gateway restart`` children race each other on the
    manual kill-and-start path, so reuse the live one instead.

    Reusing only the *live* child is not enough. The child exits as soon as
    the restart has been handed off, long before the gateway is back, so a
    frontend re-firing every few seconds cleared that guard every time and
    kept restarting a gateway that was still coming up (#89034). Requests
    within ``GATEWAY_RESTART_COOLDOWN_SECONDS`` of the last spawn for the
    same profile are coalesced onto that spawn as well.

    Before spawning, sweep for orphaned gateway processes whose parent has
    exited (e.g. desktop-app restarts leaving a reparented gateway child
    under launchd/PPID=1).  Without this the orphan keeps its platform
    connection alive and the fresh gateway stacks a duplicate (#77276).

    Returns ``(proc, reused)``.
    """
    # Reap orphaned gateways before spawning a new one (#77276).
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


from hermes_cli.web_routers import actions as _actions_routes  # noqa: E402

app.include_router(_actions_routes.router)


from hermes_cli.web_routers import audio as _audio_routes  # noqa: E402

app.include_router(_audio_routes.router)
from hermes_cli.web_routers.audio import (  # noqa: E402,F401 — legacy re-exports; tests call these via web_server.<name>
    speak_text,
)


# Collapses repeated identical ElevenLabs voice-list failures (the desktop
# re-polls on every settings open/focus) to a single log line. Re-arms on
# success or when the error signature changes, so a real new failure is seen.
_voice_list_last_error: Optional[str] = None


def _voice_list_error_logged_once(signature: Optional[str]) -> bool:
    """Return True if ``signature`` is new and should be logged now.

    Passing ``None`` clears the latch (call on success). Idempotent per
    signature: the same error logs once until it changes.
    """
    global _voice_list_last_error
    if signature is None:
        _voice_list_last_error = None
        return False
    if signature == _voice_list_last_error:
        return False
    _voice_list_last_error = signature
    return True


app.include_router(_actions_routes.status_router)


from hermes_cli.web_routers import sessions as _sessions_routes  # noqa: E402

app.include_router(_sessions_routes.list_router)
from hermes_cli.web_routers.sessions import (  # noqa: E402,F401 — legacy re-exports; tests call these via web_server.<name>
    get_sessions,
)


from hermes_cli.web_routers import profiles as _profiles_routes  # noqa: E402

app.include_router(_profiles_routes.sessions_router)
from hermes_cli.web_routers.profiles import (  # noqa: E402,F401 — legacy re-exports; tests call these via web_server.<name>
    get_profiles_sessions,
    get_profiles_sessions_sidebar,
)


app.include_router(_sessions_routes.search_router)
from hermes_cli.web_routers.sessions import (  # noqa: E402,F401 — legacy re-exports; tests call these via web_server.<name>
    search_sessions,
)


from hermes_cli.web_server_memory import (  # noqa: E402,F401 — re-exported; routers/tests reach these via web_server.<name>
    _coerce_bool,
    _dependency_importable,
    _discover_memory_provider_statuses,
    _env_lookup,
    _field_default,
    _field_is_set,
    _field_value,
    _field_visible,
    _load_memory_provider,
    _memory_provider_manifest,
    _memory_provider_setup_info,
    _memory_provider_setup_manifest,
    _normalize_memory_provider_name,
    _normalize_memory_provider_schema,
    _read_json_file,
    _read_memory_provider_existing_values,
    _require_memory_provider_ready,
    _run_setup_command,
)


from hermes_cli.web_routers import memory_providers as _memory_providers_routes  # noqa: E402

app.include_router(_memory_providers_routes.router)


from hermes_cli.web_routers import config_env as _config_env_routes  # noqa: E402

app.include_router(_config_env_routes.config_router)
from hermes_cli.web_routers.config_env import (  # noqa: E402,F401 — legacy re-exports; tests call these via web_server.<name>
    get_config,
    get_schema,
)


from hermes_cli.web_routers import models as _models_routes  # noqa: E402

app.include_router(_models_routes.router)
from hermes_cli.web_routers.models import (  # noqa: E402,F401 — legacy re-exports; tests call these via web_server.<name>
    get_model_info,
    get_model_options,
    get_recommended_default_model,
    set_moa_models,
)


app.include_router(_config_env_routes.router)


from hermes_cli.web_server_profiles import (  # noqa: E402,F401 — re-exported; routers/tests reach these via web_server.<name>
    _SKILLS_PROFILE_LOCK,
    _TERMINAL_BACKENDS,
    _approval_mode_of,
    _aux_task_summary,
    _aux_usage_rows,
    _broadcast_gateway_session_info,
    _config_profile_scope,
    _fallback_profile_dicts,
    _is_other_profile,
    _merge_aux_into_by_model,
    _parse_model_ids,
    _plugin_terminal_backend_rows,
    _profile_scope,
    _resolve_profile_dir,
    _write_profile_mcp_servers,
)


from hermes_cli.web_server_messaging import (  # noqa: E402,F401 — re-exported; routers/tests reach these via web_server.<name>
    _MESSAGING_KEYS_PAGE_KEYS,
    _TelegramOnboardingPairing,
    _WhatsAppOnboardingSession,
    _build_catalog_entry,
    _channel_managed_env_keys,
    _messaging_platform_catalog,
    _restart_gateway_after_whatsapp_onboarding,
    _telegram_onboarding_error_message,
    _telegram_onboarding_lock,
    _telegram_onboarding_pairings,
    _telegram_onboarding_request_sync,
    _whatsapp_onboarding_payload,
    _whatsapp_onboarding_sessions,
    _whatsapp_session_path,
    _write_platform_enabled,
)


# Which per-platform knobs the setup UI hides, and why: see
# hermes_cli/setup_hidden_env.py. Shared with the `hermes setup gateway`
# wizard so the surfaces ask for the same things.


from hermes_cli.web_routers import messaging as _messaging_routes  # noqa: E402

app.include_router(_messaging_routes.router)
from hermes_cli.web_routers.messaging import (  # noqa: E402,F401 — legacy re-exports; tests call these via web_server.<name>
    apply_whatsapp_onboarding,
    start_whatsapp_onboarding,
)


from hermes_cli.web_server_oauth import (  # noqa: E402,F401 — re-exported; routers/tests reach these via web_server.<name>
    _OAUTH_PROVIDER_CATALOG,
    _external_process_cli_command,
    _minimax_poller,
    _nous_poller,
    _oauth_profile_name,
    _oauth_session_profile,
    _oauth_sessions,
    _oauth_sessions_lock,
    _truncate_token,
    _xai_device_poller,
)


from hermes_cli.web_routers import oauth as _oauth_routes  # noqa: E402

app.include_router(_oauth_routes.router)
from hermes_cli.web_routers.oauth import (  # noqa: E402,F401 — legacy re-exports; tests call these via web_server.<name>
    _codex_full_login_worker,
    _new_oauth_session,
    _resolve_provider_status,
    start_oauth_login,
)


from hermes_cli.web_server_sessions import (  # noqa: E402,F401 — re-exported; routers/tests reach these via web_server.<name>
    _auto_archive_ticker_loop,
    _last_auto_archive_check,
    _maybe_auto_archive_for_profile,
    _open_session_db_at_path,
    _open_session_db_for_profile,
    _session_db_heal_exhausted,
    _session_db_heal_warned,
    _session_db_read_probe_statements,
    _session_latest_descendant,
)


app.include_router(_sessions_routes.manage_router)
from hermes_cli.web_routers.sessions import (  # noqa: E402,F401 — legacy re-exports; tests call these via web_server.<name>
    bulk_delete_sessions_endpoint,
    import_sessions_endpoint,
    count_empty_sessions_endpoint,
    delete_empty_sessions_endpoint,
    get_session_stats,
    get_session_detail,
    get_session_latest_descendant,
    get_session_messages,
    delete_session_endpoint,
    rename_session_endpoint,
    export_session_endpoint,
    prune_sessions_endpoint,
)


app.include_router(_status_routes.logs_router)


from hermes_cli.web_server_cron import (  # noqa: E402,F401 — re-exported; routers/tests reach these via web_server.<name>
    _call_cron_for_profile,
    _create_cron_job_sync,
    _cron_default_profile,
    _cron_optional_text,
    _cron_profile_dicts,
    _cron_profile_home,
    _cron_string_list,
    _find_cron_job_profile,
    _fire_cron_job_for_profile,
    _forward_cron_fire_to_gateway,
    _gateway_fire_endpoint,
    _gateway_intentionally_stopped,
    _mutate_cron_for_profile,
    _normalize_dashboard_cron_script,
    _notify_cron_provider_for_profile,
    _raise_if_cron_registration_error,
    _run_cron_dashboard_io,
    _validate_dashboard_cron_context_from,
    _validate_dashboard_cron_effective_job,
)


from hermes_cli.web_routers import cron as _cron_routes  # noqa: E402

app.include_router(_cron_routes.router)
from hermes_cli.web_routers.cron import (  # noqa: E402,F401 — legacy re-exports; tests call these via web_server.<name>
    list_cron_jobs,
    get_cron_job,
    list_cron_job_runs,
    create_cron_job,
    get_cron_delivery_targets,
    update_cron_job,
    pause_cron_job,
    resume_cron_job,
    trigger_cron_job,
    delete_cron_job,
    cron_fire_webhook,
    list_cron_blueprints,
    instantiate_blueprint,
)


from hermes_cli.web_server_mcp import (  # noqa: E402,F401 — re-exported; routers/tests reach these via web_server.<name>
    _mcp_oauth_flows,
    _mcp_server_summary,
    _normalize_mcp_server_create,
    _run_dashboard_mcp_oauth,
)


from hermes_cli.web_routers import mcp as _mcp_routes  # noqa: E402

app.include_router(_mcp_routes.router)
from hermes_cli.web_routers.mcp import (  # noqa: E402,F401 — legacy re-exports; tests call these via web_server.<name>
    list_mcp_servers,
    add_mcp_server,
    replace_mcp_servers,
    remove_mcp_server,
    test_mcp_server,
    auth_mcp_server,
    mcp_oauth_flow_status,
    mcp_oauth_callback,
    set_mcp_server_enabled,
    list_mcp_catalog,
    install_mcp_catalog_entry,
)


_ACTION_LOG_FILES.setdefault("computer-use-grant", "action-computer-use-grant.log")


from hermes_cli.web_routers import ops as _ops_routes  # noqa: E402

app.include_router(_ops_routes.router)
from hermes_cli.web_routers.ops import (  # noqa: E402,F401 — legacy re-exports; tests call these via web_server.<name>
    delete_webhook,
    list_checkpoints,
    list_credential_pool,
    prune_checkpoints,
    run_backup,
    run_doctor,
    run_import,
    run_security_audit,
    start_gateway,
)


# ---------------------------------------------------------------------------
# Skills hub endpoints — search / install / uninstall / update.
#
# Search and install touch the network (GitHub, hub sources) and run the same
# complex source-router pipeline the CLI uses, so they're spawned as background
# actions whose logs the dashboard tails.  The already-installed skill list +
# enable/disable toggle live in the existing /api/skills endpoints.
# ---------------------------------------------------------------------------


def _profile_cli_args(profile: Optional[str]) -> List[str]:
    """Return ``["-p", <name>]`` for a validated non-default profile.

    Hub install/uninstall/update run in a fresh ``hermes`` subprocess, and
    ``_apply_profile_override()`` reads ``-p`` from argv in the child — the
    only mechanism that reaches import-time-bound globals like
    ``skills_hub.SKILLS_DIR``. Empty/"current" means the dashboard's own
    profile (no args, legacy behavior).
    """
    requested = (profile or "").strip()
    if not requested or requested.lower() in {"current", "default"}:
        return []
    from hermes_cli import profiles as profiles_mod
    _resolve_profile_dir(requested)
    return ["-p", profiles_mod.normalize_profile_name(requested)]


def _hub_action_name(verb: str, key: str) -> str:
    """Unique per-skill hub action name (+ registered log file).

    ``_spawn_hermes_action`` tracks one process/log per name, so a shared
    "skills-install"/"skills-uninstall" would make concurrent row-level actions
    overwrite each other's status/log while the UI polls per identifier. Slug
    (readable) + hash (collision-proof) keys each action to its own row.
    """
    slug = re.sub(r"[^a-z0-9]+", "-", key.lower()).strip("-")[:48] or "skill"
    digest = hashlib.sha1(key.encode()).hexdigest()[:8]
    name = f"skills-{verb}-{slug}-{digest}"
    _ACTION_LOG_FILES.setdefault(name, f"action-{name}.log")
    return name


from hermes_cli.web_routers import skills as _skills_routes  # noqa: E402

app.include_router(_skills_routes.hub_router)
from hermes_cli.web_routers.skills import (  # noqa: E402,F401 — legacy re-exports; tests call these via web_server.<name>
    install_skill_hub,
    uninstall_skill_hub,
    update_skills_hub,
    list_skills_hub_sources,
    search_skills_hub,
    preview_skill_hub,
    scan_skill_hub,
)


def _installed_hub_identifiers(profile: Optional[str] = None) -> dict:
    """Map identifier -> installed lock entry for hub-installed skills.

    Lets the UI mark search results that are already installed.  Scoped to
    ``profile``'s skills/.hub/lock.json when provided (HubLockFile takes an
    explicit path, sidestepping the import-time LOCK_FILE binding).
    Best-effort: returns an empty dict if the lock file can't be read.
    """
    try:
        from tools.skills_hub import HubLockFile

        requested = (profile or "").strip()
        if requested and requested.lower() != "current":
            profile_dir = _resolve_profile_dir(requested)
            lock = HubLockFile(profile_dir / "skills" / ".hub" / "lock.json")
        else:
            lock = HubLockFile()
        out = {}
        for entry in lock.list_installed():
            ident = entry.get("identifier")
            if ident:
                out[ident] = {
                    "name": entry.get("name"),
                    "trust_level": entry.get("trust_level"),
                    "scan_verdict": entry.get("scan_verdict"),
                }
        return out
    except Exception:
        return {}


app.include_router(_profiles_routes.router)
from hermes_cli.web_routers.profiles import (  # noqa: E402,F401 — legacy re-exports; tests call these via web_server.<name>
    list_profiles_endpoint,
    create_profile_endpoint,
    get_active_profile_endpoint,
    set_active_profile_endpoint,
    get_profile_setup_command,
    open_profile_terminal_endpoint,
    rename_profile_endpoint,
    delete_profile_endpoint,
    get_profile_soul,
    update_profile_soul,
    update_profile_description_endpoint,
    update_profile_model_endpoint,
    describe_profile_auto_endpoint,
)


app.include_router(_skills_routes.router)
from hermes_cli.web_routers.skills import (  # noqa: E402,F401 — legacy re-exports; tests call these via web_server.<name>
    get_skills,
    toggle_skill,
    get_skill_content,
    create_skill,
    update_skill_content,
)


from hermes_cli.web_routers import tools as _tools_routes  # noqa: E402

app.include_router(_tools_routes.router)
from hermes_cli.web_routers.tools import (  # noqa: E402,F401 — legacy re-exports; tests call these via web_server.<name>
    get_toolsets,
    toggle_toolset,
    get_toolset_config,
    get_toolset_models,
    select_toolset_model,
    select_toolset_provider,
    save_toolset_env,
    run_toolset_post_setup,
    get_terminal_backends,
    select_terminal_backend,
    get_computer_use_status,
    grant_computer_use_permissions,
)


from hermes_cli.web_routers import analytics as _analytics_routes  # noqa: E402

app.include_router(_analytics_routes.router)
from hermes_cli.web_routers.analytics import (  # noqa: E402,F401 — legacy re-exports; tests call these via web_server.<name>
    get_models_analytics,
    get_usage_analytics,
)


from hermes_cli.web_server_chat import (  # noqa: E402,F401 — re-exported; routers/tests reach these via web_server.<name>
    PTY_REGISTRY,
    PtyBridge,
    PtyUnavailableError,
    _GATEWAY_WS_PROTOCOL,
    _GATEWAY_WS_TICKET_PROTOCOL_PREFIX,
    _LOOPBACK_HOSTS,
    _PTY_BRIDGE_AVAILABLE,
    _RESIZE_RE,
    _WILDCARD_HOSTS,
    _active_session_file_for_channel,
    _build_gateway_ws_url,
    _build_sidecar_url,
    _get_console_executor,
    _legacy_pump,
    _resolve_chat_argv,
    _resolve_chat_argv_async,
    _resolve_client_ws_host,
    _ws_auth_ok,
    _ws_auth_reason,
    _ws_client_is_allowed,
    _ws_client_reason,
    _ws_host_origin_is_allowed,
    _ws_host_origin_reason,
    _ws_request_is_allowed,
)


from hermes_cli.web_routers import chat_ws as _chat_ws_routes  # noqa: E402

app.include_router(_chat_ws_routes.router)
from hermes_cli.web_routers.chat_ws import (  # noqa: E402,F401 — legacy re-exports; tests call these via web_server.<name>
    _broadcast_event,
    _get_event_state,
    pty_ws,
)


from hermes_cli.web_server_dashboard import (  # noqa: E402,F401 — re-exported; routers/tests reach these via web_server.<name>
    _BUILTIN_DASHBOARD_THEMES,
    _THEME_COMPONENT_BUCKETS,
    _THEME_NAMED_ASSET_KEYS,
    _discover_dashboard_plugins,
    _discover_user_themes,
    _invalidate_plugins_hub_cache,
    _merged_plugins_hub,
    _mount_plugin_api_routes,
    _normalise_theme_definition,
    _render_active_theme_bootstrap_css,
    _safe_plugin_api_relpath,
    _schedule_check_fn_probe,
    mount_spa,
)


from hermes_cli.web_routers import dashboard_ui as _dashboard_ui_routes  # noqa: E402

app.include_router(_dashboard_ui_routes.router)
from hermes_cli.web_routers.dashboard_ui import (  # noqa: E402,F401 — legacy re-exports; tests call these via web_server.<name>
    post_agent_plugin_install,
    serve_plugin_asset,
)


# Cache discovered plugins per-process (refresh on explicit re-scan).
_dashboard_plugins_cache: Optional[list] = None


def _get_dashboard_plugins(force_rescan: bool = False) -> list:
    global _dashboard_plugins_cache
    if _dashboard_plugins_cache is None or force_rescan:
        _dashboard_plugins_cache = _discover_dashboard_plugins()
    elif _dashboard_plugins_cache:
        if any(not Path(p["_dir"]).is_dir() for p in _dashboard_plugins_cache):
            _dashboard_plugins_cache = _discover_dashboard_plugins()
    return _dashboard_plugins_cache


# Mount plugin API routes before the SPA catch-all.
_mount_plugin_api_routes()

# Mount the dashboard auth routes (/login, /auth/*, /api/auth/*) before the
# SPA catch-all so /{full_path:path} doesn't swallow them.  These are
# always mounted — the gate middleware decides whether to enforce auth,
# not whether the routes exist.
from hermes_cli.dashboard_auth.routes import router as _dashboard_auth_router  # noqa: E402
app.include_router(_dashboard_auth_router)

mount_spa(app)


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

    ``initial_profile`` (when set) is appended to the auto-opened browser
    URL as ``?profile=<name>`` so the SPA's profile switcher preselects it
    — used when a profile alias (``<profile> dashboard``) routes to the
    machine dashboard.

    ``headless`` is the ``serve`` path: the JSON-RPC/WS backend with no UI
    build and no SPA mount (mount_spa() honours ``HERMES_SERVE_HEADLESS``), so
    the banner announces the bind rather than a browser URL.

    ``ssh_session_token`` and ``ssh_owner_nonce`` are process-local Desktop SSH
    bootstrap state. Neither is persisted or exported to child processes.

    ``start_mcp_discovery_after_bind`` (Desktop ``serve``) defers the
    background MCP discovery thread until the ready sentinel has been written,
    so its SDK import cannot hold the GIL against the pre-bind import path.
    """
    _apply_ssh_session_token(ssh_session_token or "")
    _apply_ssh_owner_nonce(ssh_owner_nonce)

    # Raise RLIMIT_NOFILE for dashboard-mode starts that don't route through
    # the `serve` path in main.py (which applies the same floor). Canonical
    # policy lives in resource_limits; #81547's motivating leak (iterdir fds)
    # is fixed above, this covers legitimate high fd demand.
    from hermes_cli.resource_limits import apply_nofile_soft_limit

    apply_nofile_soft_limit()

    import uvicorn

    try:
        from hermes_cli.nous_auth_keepalive import start_nous_auth_keepalive

        start_nous_auth_keepalive()
    except Exception as exc:
        _log.debug("Nous auth keepalive did not start: %s", exc)

    # A configured browser-facing URL is also the exact Host/Origin trust
    # declaration for reverse-proxy deployments. Resolve it once at startup so
    # request middleware never reloads config. Any non-loopback public hostname
    # engages the auth gate even when the backend itself remains on loopback;
    # otherwise the SPA's local session token would become remotely reachable.
    app.state.trusted_public_hosts = _dashboard_public_hosts()
    # Stash the auth-gate flag on app.state so middleware / SPA-token injection /
    # WS-auth paths can branch on it consistently. It also decides whether to
    # refuse startup, log the gate-on banner, and enable uvicorn proxy_headers.
    if _desktop_loopback_auth_exempt(host, ssh_session_token, ssh_owner_nonce):
        # A configured dashboard.public_url describes the operator's PUBLIC
        # deployment, not this private Desktop-owned loopback backend (#96490).
        # Desktop authenticates with the per-spawn session token; forcing the
        # ticket-only gate here broke every Desktop boot while the actual
        # public dashboard — a separate non-loopback process — stayed gated.
        app.state.auth_required = should_require_auth(host)
        _log.info(
            "Desktop-owned loopback backend: dashboard.public_url does not "
            "engage the ticket gate for this process; the public deployment "
            "keeps its own gate.",
        )
    else:
        app.state.auth_required = should_require_dashboard_auth(
            host, app.state.trusted_public_hosts
        )

    # ``--insecure`` no longer disables the auth gate (June 2026 hardening:
    # the hermes-0day MCP-persistence campaign abused unauthenticated public
    # dashboards). If a caller still passes it, warn that it is now a no-op
    # rather than silently changing their expectation of an open bind.
    if allow_public and host not in _LOOPBACK_HOST_VALUES:
        _log.warning(
            "--insecure no longer bypasses dashboard authentication. A "
            "non-loopback bind (%s) now ALWAYS requires an auth provider "
            "(OAuth or the bundled password provider). Configure one — see "
            "below — or bind to 127.0.0.1 and reach it over an SSH tunnel / "
            "Tailscale.", host,
        )

    if app.state.auth_required:
        # The gate engages on every non-loopback bind. Require at least one
        # provider to be registered, else fail closed — there is no longer an
        # escape hatch that serves the dashboard without authentication.
        from hermes_cli.dashboard_auth import list_providers
        if not list_providers():
            # Surface the *specific* reason any bundled provider declined
            # to register (e.g. missing HERMES_DASHBOARD_OAUTH_CLIENT_ID).
            # Each provider plugin that ships with Hermes Agent exposes a
            # module-level ``LAST_SKIP_REASON`` string for this purpose;
            # without it the operator would only see "no providers" which
            # is misleading when the provider IS installed but unconfigured.
            skip_reasons: list[str] = []
            try:
                from plugins.dashboard_auth import nous as _nous_plugin

                if _nous_plugin.LAST_SKIP_REASON:
                    skip_reasons.append(
                        f"  • nous: {_nous_plugin.LAST_SKIP_REASON}"
                    )
            except Exception:
                pass

            # Name the exact reason the gate engaged. When the bind itself is
            # loopback the ONLY trigger is dashboard.public_url — an operator
            # (or a stale config.yaml entry) declared external exposure. Say
            # so explicitly, print the offending URL, and give the two exits:
            # configure auth, or remove public_url to restore local-only mode.
            if host in _LOOPBACK_HOST_VALUES:
                _public_url_for_msg = ""
                try:
                    from hermes_cli.dashboard_auth.prefix import (
                        resolve_public_url as _rpu,
                    )

                    _public_url_for_msg = _rpu()
                except Exception:
                    pass
                _gate_reason = (
                    f"dashboard.public_url is set to "
                    f"{_public_url_for_msg or '<a non-loopback URL>'} — an "
                    f"operator-declared external URL engages the auth gate "
                    f"even on a loopback bind"
                )
                _local_only_hint = (
                    "If this dashboard should be LOCAL-ONLY (no reverse "
                    "proxy), remove dashboard.public_url from config.yaml "
                    "(and unset HERMES_DASHBOARD_PUBLIC_URL) to restore the "
                    "unauthenticated loopback mode.\n"
                )
            else:
                _gate_reason = (
                    f"the auth gate engages on non-loopback binds ({host})"
                )
                _local_only_hint = ""

            _fix_hint = (
                _local_only_hint
                + "Configure an auth provider before exposing the dashboard:\n"
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
            # Hint when credentials exist but the bundled provider is blocked
            # (#54489).
            try:
                from hermes_cli.config import load_config as _load_cfg
                from hermes_cli.plugins_cmd import _BASIC_AUTH_PLUGIN_KEYS

                _cfg = _load_cfg()
                _ba = (_cfg.get("dashboard") or {}).get("basic_auth") or {}
                _disabled = (_cfg.get("plugins") or {}).get("disabled") or []
                # Basic auth only activates with a username AND a credential
                # (plaintext password or password_hash); don't fire the hint on
                # a half-configured block.
                _has_creds = bool(_ba.get("username")) and bool(
                    _ba.get("password_hash") or _ba.get("password")
                )
                if _has_creds and (set(_disabled) & _BASIC_AUTH_PLUGIN_KEYS):
                    _fix_hint = (
                        "The 'basic' dashboard-auth plugin is in "
                        "plugins.disabled but dashboard.basic_auth is "
                        "configured.\n"
                        "Remove 'basic' from plugins.disabled (or run "
                        "`hermes plugins enable basic`), then restart the "
                        "dashboard.\n\n"
                    ) + _fix_hint
            except Exception:
                pass
            if skip_reasons:
                raise SystemExit(
                    f"Refusing to bind dashboard to {host} — {_gate_reason}, "
                    f"but no auth providers are registered.\n\n"
                    f"Bundled providers reported these issues:\n"
                    + "\n".join(skip_reasons)
                    + "\n\n"
                    + _fix_hint
                )
            raise SystemExit(
                f"Refusing to bind dashboard to {host} — {_gate_reason}, "
                f"but no auth providers are registered.\n\n" + _fix_hint
            )
        _log.info(
            "Dashboard binding to %s with auth gate enabled. Providers: %s",
            host,
            ", ".join(p.name for p in list_providers()),
        )

    # Record the bound host so host_header_middleware can validate incoming
    # Host headers against it. Defends against DNS rebinding (GHSA-ppp5-vxwm-4cf7).
    app.state.bound_host = host

    # ── Start uvicorn with direct Server API ─────────────────────────
    # We use uvicorn.Server directly (not uvicorn.run) so we can split
    # startup from the main loop.  After startup() the socket is actually
    # bound — we read the OS-assigned port from the live socket, print
    # HERMES_DASHBOARD_READY, open the browser, *then* serve.
    #
    # This eliminates the TOCTOU of the old pre-bind-then-close approach
    # (bind port 0 → close → uvicorn rebind): the socket is held by
    # uvicorn the entire time, so no other process can steal the port.
    #
    # For explicit non-zero ports, a taken port is detected by the #93608
    # preflight probe below (BACKEND_PORT_IN_USE sentinel + distinct exit
    # code); uvicorn's own bind error remains the fallback for races.
    # Loopback binds are the Desktop case: a single local client, no reverse
    # proxy in front. uvicorn's ws keepalive ping runs ON the same event loop
    # as agent turns, and a single synchronous GIL-holding call on a worker
    # thread (e.g. a regex/scrub over a large model output, or a long
    # delegate_task subagent turn) can starve that loop for *minutes* — the
    # loop cannot process the incoming pong, so uvicorn declares the socket
    # dead and closes it, dropping an otherwise-healthy local connection
    # (#53773: "event loop stalled 226.3s"; #48445/#50005). A longer timeout
    # only raises the threshold — a multi-minute stall sails past any finite
    # window. The keepalive ping exists to detect *half-open* connections
    # (reverse-proxy 524, dropped tunnels), which cannot happen on loopback:
    # there is no network or proxy in the path, and a dead local client tears
    # the socket down with a real FIN/RST that starlette surfaces as
    # WebSocketDisconnect regardless of the ping. So on loopback the ping
    # provides ~no liveness value while actively killing recoverable stalls —
    # disable it entirely. Non-loopback binds sit behind a Cloudflare Tunnel
    # (idle timeout ~100s) where half-open IS a real failure mode, so keep the
    # ping at 20/20 to detect it promptly and stay under the tunnel's idle
    # window.
    _is_loopback = host in ("127.0.0.1", "localhost", "::1")
    # Non-loopback ping cadence is config-driven (dashboard.ws_ping_interval /
    # dashboard.ws_ping_timeout, #79635); the 20/20 defaults keep the
    # Cloudflare-Tunnel-friendly behaviour when unset or invalid.
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
        # proxy_headers defaults to False so _ws_client_is_allowed sees
        # the real connection peer rather than X-Forwarded-For's rewritten
        # value (which would defeat the loopback gate when behind a reverse
        # proxy).  When the OAuth gate is active we are explicitly running
        # behind a TLS terminator (Fly.io) and need X-Forwarded-Proto to
        # decide cookie Secure flags, so we flip proxy_headers on for that
        # mode.
        proxy_headers=bool(app.state.auth_required),
        # Keep uvicorn's loopback-only default unless the operator explicitly
        # trusts the address or bounded network of an upstream proxy. This is
        # what lets a separate-container TLS terminator supply HTTPS/client
        # metadata without accepting spoofed X-Forwarded-* headers from every
        # caller.
        forwarded_allow_ips=_dashboard_forwarded_allow_ips(_dash_cfg),
        # Half-open detection for public binds only (see above). Loopback
        # disables the protocol ping (None) so an event-loop stall can never
        # trigger a false disconnect; a genuinely dead local client is still
        # reaped via the WebSocketDisconnect → disconnect/reap path.
        ws_ping_interval=None if _is_loopback else _ws_ping_setting("ws_ping_interval"),
        ws_ping_timeout=None if _is_loopback else _ws_ping_setting("ws_ping_timeout"),
        ws_max_size=_DESKTOP_ATTACHMENT_WS_MAX_BYTES,
    )
    server = uvicorn.Server(config)

    # Flush-on-kill guard (#94724 item 2): install chaining SIGTERM/SIGINT
    # handlers that first persist in-memory session transcripts to state.db
    # (bounded, best-effort) before the normal shutdown story runs. Installed
    # on the main thread BEFORE uvicorn's capture_signals() so uvicorn saves
    # these as the "original" handlers and re-raises into them after its own
    # graceful shutdown — kills outside the serve window are covered too.
    try:
        from tui_gateway.server import install_exit_flush_signal_handlers

        install_exit_flush_signal_handlers()
    except Exception as exc:
        _log.debug("exit-flush signal handlers not installed: %s", exc)

    # ── #93608: machine-readable port-conflict detection ──────────────
    # uvicorn's own bind_socket() would catch the EADDRINUSE and exit 1
    # with a bare ERROR line — indistinguishable from "backend broken".
    # Probe the exact bind first so a conflict surfaces as the stable
    # BACKEND_PORT_IN_USE sentinel + a distinct exit code instead.
    # ``--port 0`` (ephemeral) is skipped by the probe and unaffected.
    if _port_bind_conflict(host, port):
        _report_port_in_use(host, port)
        raise SystemExit(PORT_IN_USE_EXIT_CODE)

    async def _serve():
        # Split startup from main_loop so we can read the bound port
        # after the socket is live (ephemeral port discovery).
        if not config.loaded:
            config.load()
        server.lifespan = config.lifespan_class(config)
        with server.capture_signals():
            await server.startup()
            if server.should_exit:
                return

            # Parent-death watchdog. The desktop spawns us and is supposed to
            # SIGTERM us on quit, but a crash / SIGKILL / update handoff that
            # exits before reaping leaves us orphaned (ppid→1) yet still
            # serving — leaking the whole backend + its MCP child subtree
            # (each MCP watchdog is parented to THIS process, so os._exit here
            # cascades their teardown). Same pattern as
            # Clear corpses left by a previous unclean Desktop exit before we
            # stack another backend + MCP tree (EMFILE / missing tabs).
            # Parent-death watchdog only protects *this* process going forward.
            if os.getenv("HERMES_DESKTOP") == "1":
                try:
                    from hermes_cli.dashboard_procs import (
                        _reap_orphaned_desktop_local_serves,
                    )

                    _reap_orphaned_desktop_local_serves()
                except Exception as exc:
                    _log.debug("orphan desktop-local serve reap skipped: %s", exc)

            # Same sweep for stdio MCP helper children (#61514): ledger-
            # identified helpers whose recorded spawner is provably dead are
            # corpses from a prior unclean exit — reap them before this
            # backend stacks a fresh MCP tree on top. Positive identity only
            # (spawn ledger + spawner_is_dead); a helper whose spawner is
            # alive or unprovable is never touched.
            try:
                from hermes_cli.process_identity import reap_orphaned_mcp_helpers

                reap_orphaned_mcp_helpers()
            except Exception as exc:
                _log.debug("orphan MCP helper reap skipped: %s", exc)

            # tui_gateway/slash_worker.py::_start_parent_death_watchdog. No-op
            # for standalone `hermes serve` (no HERMES_PARENT_PID env).
            _start_parent_death_watchdog()

            actual_port = _read_bound_port(server, fallback=port)
            app.state.bound_port = actual_port

            # Positive process identity: record (pid, create_time, purpose,
            # spawner) in the machine spawn ledger and — on Windows — attach
            # to a kill-on-close job so this backend's whole child tree dies
            # with it. Both best-effort; failures degrade to legacy behavior.
            # Registered AFTER the bind so the entry carries the ACTUAL port
            # (ephemeral binds included) — the structured host/port/profile
            # is what lets `hermes update` relaunch a manually-started serve
            # on its real endpoint instead of dropping it (#63206).
            try:
                from hermes_cli.process_identity import (
                    attach_self_to_kill_on_close_job,
                    register_self,
                )

                register_self(
                    "serve" if headless else "dashboard",
                    detail={
                        "host": host,
                        "port": actual_port,
                        "profile": initial_profile or "",
                    },
                )
                attach_self_to_kill_on_close_job()
            except Exception as exc:
                _log.debug("process-identity registration skipped: %s", exc)

            _write_dashboard_ready_file(actual_port)
            # Port-discovery sentinel parsed by the desktop spawn. `serve` is a
            # plain backend, not a dashboard, so it announces a neutral token;
            # `dashboard` keeps the legacy one. The desktop matches either.
            ready_token = "HERMES_BACKEND_READY" if headless else "HERMES_DASHBOARD_READY"
            # tui_gateway.server (imported above for the flush-on-SIGTERM
            # handlers, #94724) redirects sys.stdout→sys.stderr at import time
            # to keep stray prints off the JSON-RPC protocol stream. fd 1 is
            # still the real stdout — and the Desktop spawn watches
            # child.stdout for this sentinel — so write to the fd, not to the
            # (redirected) sys.stdout, or the desktop times out after 90s
            # against a perfectly healthy backend (#96282).
            _write_machine_sentinel_line(f"{ready_token} port={actual_port}")
            if headless:
                # No SPA, and the JSON-RPC/WS endpoints are auth-gated — don't
                # advertise a paste-and-connect URL, just announce the bind.
                # flush: on a piped stdout (Desktop spawn) this line is
                # block-buffered and can surface MINUTES after the flushed
                # READY sentinel above, which reads as a slow boot in
                # support bundles when the backend was actually up.
                print(f"  Hermes backend listening on {host}:{actual_port}", flush=True)
            else:
                print(f"  Hermes Web UI → http://{host}:{actual_port}")
            _maybe_open_browser(host, actual_port, open_browser, initial_profile)

            if start_mcp_discovery_after_bind:
                # Deferred from cmd_dashboard for Desktop `serve` (see there).
                # Not started at the bind itself either: the ~350ms `mcp` SDK
                # import holds the GIL, and at bind time the renderer is doing
                # its WebSocket handshake + first hydration reads against this
                # loop (measured: starting it here gave back most of the
                # READY gain as a slower connect). One second later the shell
                # is painted and idle. An agent build inside that second fires
                # the deferred start itself (wait_for_mcp_discovery), so its
                # bounded join and the late-binding refresh are unchanged.
                try:
                    from hermes_cli.mcp_startup import defer_background_mcp_discovery

                    defer_background_mcp_discovery(
                        logger=_log,
                        thread_name="dashboard-mcp-discovery",
                        delay=_DESKTOP_MCP_DISCOVERY_DELAY_S,
                    )
                except Exception:
                    _log.debug("Deferred MCP discovery arm failed", exc_info=True)

            # Collapse the peer-hangup teardown flood (#50005). When the Desktop
            # forcibly closes its WebSocket mid-write, asyncio logs a full
            # traceback per pending connection-lost callback — 50+ identical
            # WinError 10054 (ConnectionResetError) lines per disconnect on
            # Windows. This filter downgrades exactly that class to one debug
            # line and passes every other loop error through unchanged.
            try:
                from tui_gateway.loop_noise import install_loop_noise_filter

                install_loop_noise_filter(asyncio.get_running_loop())
            except Exception as exc:  # pragma: no cover - best-effort
                _log.debug("loop noise filter install skipped: %s", exc)

            # ── Loop heartbeat watchdog (CF-1) ───────────────────────────
            # Confirm the GIL-pressure hypothesis in production. Re-arm a 2s
            # tick and measure the drift between when it *should* fire and
            # when it actually does: a healthy loop drifts ~0, but a turn that
            # holds the GIL blocks the loop and the next tick fires late by the
            # stall duration. We log that so a stalled-loop WS drop is
            # diagnosable from the gateway log. Uses loop.time() (monotonic)
            # for drift, and call_later (not a task) so it dies with the loop —
            # nothing to cancel on shutdown.
            _hb_interval = 2.0
            _hb_stall_threshold = 5.0
            _hb_loop = asyncio.get_running_loop()

            def _loop_heartbeat(expected: float) -> None:
                now = _hb_loop.time()
                drift = now - expected
                if drift > _hb_stall_threshold:
                    _log.warning(
                        "event loop stalled %.1fs (GIL pressure suspected)",
                        drift,
                    )
                _hb_loop.call_later(
                    _hb_interval, _loop_heartbeat, now + _hb_interval
                )

            _hb_loop.call_later(
                _hb_interval, _loop_heartbeat, _hb_loop.time() + _hb_interval
            )

            await server.main_loop()
            if server.started:
                await server.shutdown()

    # On POSIX, keep the long-standing ``asyncio.run(_serve())`` runner —
    # Python's default loop there is already a SelectorEventLoop (or uvloop when
    # uvicorn[standard] installs it), which is exactly what uvicorn serves on.
    # Uvicorn's ``capture_signals()`` restores the original SIGINT handler and
    # re-raises the captured signal after a graceful shutdown, which otherwise
    # leaks a noisy KeyboardInterrupt traceback for the normal foreground
    # dashboard Ctrl+C path. Treat that one signal as a clean user-requested
    # shutdown; other serve-time errors still propagate.
    #
    # On Windows it is broken: ``asyncio.run`` defaults to a ProactorEventLoop,
    # but uvicorn's socket-serving stack assumes a SelectorEventLoop on win32
    # (``uvicorn/loops/asyncio.py`` forces it, and ``uvicorn.Server.run`` threads
    # ``config.get_loop_factory()`` into its runner for exactly this reason).
    # Driving uvicorn on the proactor loop makes ``server.startup()`` bind a
    # socket that never accepts — the dashboard / desktop backend prints
    # "Skipping web UI build" and then hangs forever with the port LISTENING but
    # no TCP handshake completing (#50641). So *only on Windows* we mirror
    # uvicorn's own machinery and run on the loop factory it picks.
    if sys.platform != "win32":
        try:
            asyncio.run(_serve())
        except KeyboardInterrupt:
            return
        except SystemExit as exc:
            # Probe-to-bind race (#93608): another process grabbed the port
            # between our preflight probe and uvicorn's real bind. uvicorn's
            # bind_socket() exits 1 — re-check the bind and translate a
            # confirmed conflict into the sentinel + distinct exit code.
            if exc.code == 1 and _port_bind_conflict(host, port):
                _report_port_in_use(host, port)
                raise SystemExit(PORT_IN_USE_EXIT_CODE) from None
            raise
        return

    # Windows-only path. Resolve the runner + loop factory FIRST (and fall back
    # to a hand-installed Windows selector policy only when uvicorn predates the
    # loop-factory API, < 0.36). The actual serve call is then OUTSIDE this
    # import try/except so genuine serve-time errors (port in use) propagate
    # normally instead of being swallowed and double-run.
    try:
        from uvicorn._compat import asyncio_run as _runner

        _loop_factory = config.get_loop_factory()
    except Exception:
        _runner = None
        _loop_factory = None
        try:
            asyncio.set_event_loop_policy(
                asyncio.WindowsSelectorEventLoopPolicy()  # type: ignore[attr-defined]
            )
        except Exception:
            pass

    # Same clean Ctrl+C contract as the POSIX branch above: ``capture_signals()``
    # re-raises the captured signal after the graceful shutdown has already
    # completed. For console Ctrl+C the re-raised SIGINT lands as
    # ``KeyboardInterrupt`` — a clean user-requested exit here too. (Re-raised
    # SIGTERM/SIGBREAK keep their default terminate disposition and never reach
    # this except.)
    try:
        if _runner is not None:
            _runner(_serve(), loop_factory=_loop_factory)
        else:
            asyncio.run(_serve())
    except KeyboardInterrupt:
        return
    except SystemExit as exc:
        # Same probe-to-bind race translation as the POSIX branch (#93608).
        if exc.code == 1 and _port_bind_conflict(host, port):
            _report_port_in_use(host, port)
            raise SystemExit(PORT_IN_USE_EXIT_CODE) from None
        raise
