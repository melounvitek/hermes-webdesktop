"""
Hermes Agent — Web UI server.

Provides a FastAPI backend serving the Vite/React frontend and REST API
endpoints for managing configuration, environment variables, and sessions.

Usage:
    python -m hermes_cli.main web          # Start on http://127.0.0.1:9119
    python -m hermes_cli.main web --port 8080
"""

from contextlib import asynccontextmanager, contextmanager

import asyncio
import functools
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import hmac
import inspect
import ipaddress
import json
import logging
import mimetypes
import os
import re
import secrets
import shlex
import shutil  # noqa: F401 — tests monkeypatch web_server.shutil.which
import subprocess
import sys
import sysconfig
import tempfile
import threading
import time
import urllib.parse

from hermes_cli._subprocess_compat import windows_detach_flags
from hermes_cli.install_identity import get_install_id as _shared_get_install_id
from hermes_cli.pty_session import run_reaper
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hermes_cli import __version__
from hermes_cli.config import (
    build_cron_model_impact,
    cfg_get,
    DEFAULT_CONFIG,
    OPTIONAL_ENV_VARS,
    clear_model_endpoint_credentials,
    get_hermes_home,
    get_process_hermes_home,
    load_config,
    # Late-bound by extracted routers (tests monkeypatch web_server.<name>).
    check_config_version,  # noqa: F401
    remove_env_value,  # noqa: F401
    load_env,
    read_raw_config,
    resolve_cron_model_drift_defaults,
    save_config,
    save_env_value,  # noqa: F401 — late-bound by extracted routers
    find_provider_entry,
    detect_install_method,
    redact_key,
    write_platform_config_field,
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
        from starlette.concurrency import run_in_threadpool
    except Exception:
        raise SystemExit(
            "Web UI requires fastapi and uvicorn.\n"
            f"Install with: {sys.executable} -m pip install 'fastapi' 'uvicorn[standard]'"
        )

WEB_DIST = Path(os.environ["HERMES_WEB_DIST"]) if "HERMES_WEB_DIST" in os.environ else Path(__file__).parent / "web_dist"
_log = logging.getLogger(__name__)


def _process_start_marker(pid: int) -> str:
    """Return a cross-runtime marker for the current incarnation of ``pid``.

    ``ProcessLookupError`` means the process is absent. Other failures are left
    distinct so callers can fail safe rather than killing a healthy backend.
    """
    if sys.platform == "linux":
        try:
            stat_line = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise ProcessLookupError(pid) from exc

        # The command in field 2 may contain spaces or parentheses. Splitting
        # after its final ')' leaves field 3 at index zero and field 22 at 19.
        fields = stat_line.rsplit(")", 1)[1].strip().split()
        if len(fields) < 20 or not fields[19].isdigit():
            raise OSError(f"invalid /proc stat data for PID {pid}")
        return f"linux:{fields[19]}"

    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        process_query_limited_information = 0x1000
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.GetProcessTimes.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(wintypes.FILETIME),
            ctypes.POINTER(wintypes.FILETIME),
            ctypes.POINTER(wintypes.FILETIME),
            ctypes.POINTER(wintypes.FILETIME),
        ]
        kernel32.GetProcessTimes.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
        if not handle:
            error = ctypes.get_last_error()
            if error in (87, 1168):  # invalid parameter / not found
                raise ProcessLookupError(pid)
            raise OSError(error, f"OpenProcess failed for PID {pid}")

        creation = wintypes.FILETIME()
        exit_time = wintypes.FILETIME()
        kernel = wintypes.FILETIME()
        user = wintypes.FILETIME()
        try:
            if not kernel32.GetProcessTimes(
                handle,
                ctypes.byref(creation),
                ctypes.byref(exit_time),
                ctypes.byref(kernel),
                ctypes.byref(user),
            ):
                error = ctypes.get_last_error()
                raise OSError(error, f"GetProcessTimes failed for PID {pid}")
        finally:
            kernel32.CloseHandle(handle)

        filetime = (creation.dwHighDateTime << 32) | creation.dwLowDateTime
        return f"win:{filetime + 504911232000000000}"

    result = subprocess.run(
        ["ps", "-p", str(pid), "-o", "lstart="],
        capture_output=True,
        text=True,
        check=False,
    )
    marker = result.stdout.strip()
    if result.returncode == 0 and marker:
        return f"ps:{marker}"
    if result.returncode == 1 and not marker:
        raise ProcessLookupError(pid)
    raise OSError(f"ps could not inspect PID {pid}: {result.stderr.strip()}")


def _valid_parent_start_marker(marker: str) -> bool:
    prefix, separator, value = marker.partition(":")
    if not separator or not value or value != value.strip():
        return False
    if prefix in ("linux", "win", "winms"):
        return value.isdigit()
    return prefix == "ps"


def _parent_start_markers_match(actual: str, expected: str) -> bool:
    """Compare parent markers across Desktop protocol generations.

    Older Windows Desktop builds send .NET ticks (``win:``). New builds use
    Electron's native process creation time in Unix milliseconds (``winms:``)
    so startup does not need to launch PowerShell. The backend still reads the
    exact FILETIME and normalizes it only when the expected marker is ``winms``.
    """
    if actual == expected:
        return True
    if not actual.startswith("win:") or not expected.startswith("winms:"):
        return False

    try:
        dotnet_ticks = int(actual.removeprefix("win:"))
        expected_unix_ms = int(expected.removeprefix("winms:"))
    except ValueError:
        return False

    dotnet_ticks_at_unix_epoch = 621_355_968_000_000_000
    actual_unix_ms = (dotnet_ticks - dotnet_ticks_at_unix_epoch) // 10_000
    return actual_unix_ms == expected_unix_ms


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


def _warm_gateway_module() -> None:
    """Pre-import heavy modules so the event loop is not stalled on first use.

    On a cold Windows install, importing these module chains triggers .pyc
    compilation and Defender real-time scans that can stall the event loop
    for 15-30s. The original fix (pre-#60800) only warmed
    ``hermes_cli.gateway``. But the first WS connection and its initial
    RPC burst (``setup.status``, ``setup.runtime_check``,
    ``gateway.ready``→``resolve_skin``) pull in several *other* heavy
    chains that were still imported on the loop thread, contributing to
    the ~14s cold-start stall (#60800). Warm them all here so the cost
    is paid in a worker thread while the server socket is already open.
    """
    for mod in (
        "hermes_cli.gateway",
        # setup.status / setup.runtime_check resolve provider auth state,
        # which imports copilot_auth (→ subprocess module) and scans
        # credential files. First import is noticeably slow on Windows.
        "hermes_cli.auth",
        "hermes_cli.copilot_auth",
        "hermes_cli.runtime_provider",
        # resolve_skin() reads config + initialises the skin engine.
        # Even though handle_ws now calls it via asyncio.to_thread
        # (see tui_gateway/ws.py), warming it here avoids the first-call
        # import cost inside that thread.
        "hermes_cli.skin_engine",
        # model.options / picker context — parses provider catalogs and
        # the models.dev cache on first use.
        "hermes_cli.inventory",
        "hermes_cli.model_switch",
    ):
        try:
            __import__(mod)
        except Exception:
            pass


def _resolve_restart_drain_timeout() -> float:
    try:
        from hermes_cli.gateway import _get_restart_drain_timeout
        return _get_restart_drain_timeout()
    except ImportError:
        from gateway.restart import DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT
        return DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT


def _eager_reconcile_own_session_db() -> None:
    """One writable open of this process's own state.db at startup.

    ``SessionDB.__init__`` runs ``_init_schema`` → ``_reconcile_columns``,
    bringing a store left behind by `hermes update` current before the
    dashboard's first session-list poll, with the open-time lock patience
    (jittered retries) absorbing transient contention. Never raises: a
    store this cannot fix is still served through the read-probe heal in
    :func:`_open_session_db_at_path`, which retries on every poll.
    """
    try:
        from hermes_state import SessionDB, _default_db_path

        SessionDB(db_path=Path(_default_db_path()), read_only=False).close()
    except Exception as exc:
        _log.warning(
            "startup schema reconcile of state.db failed (%s); session "
            "reads will retry the heal per poll", exc,
        )


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


# ---------------------------------------------------------------------------
# Config schema — auto-generated from DEFAULT_CONFIG
# ---------------------------------------------------------------------------

# Manual overrides for fields that need select options or custom types
def _memory_provider_options() -> List[str]:
    """Discovered memory providers for the ``memory.provider`` select.

    Directory-scan only (no provider imports), so it's safe at module import
    time. ``""`` (built-in only) is always first; discovery failures degrade to
    the bundled defaults rather than dropping the field. The literal
    ``builtin`` alias is deliberately NOT offered — built-in memory is not a
    provider plugin, and ``_normalize_memory_provider_name`` already maps any
    legacy ``builtin``/``built-in``/``none`` value back to ``""`` (#49513).
    """
    options = [""]
    try:
        from plugins.memory import list_memory_provider_names

        options.extend(list_memory_provider_names())
    except Exception:
        options.extend(["honcho"])
    # Dedupe, preserve order
    return list(dict.fromkeys(options))


def _timezone_options() -> List[str]:
    """Return sorted IANA timezone identifiers, cached at import time."""
    try:
        import zoneinfo
        return sorted(zoneinfo.available_timezones()) or ["UTC"]
    except Exception:  # pragma: no cover
        return ["UTC"]


_SCHEMA_OVERRIDES: Dict[str, Dict[str, Any]] = {
    "timezone": {
        "type": "select",
        "description": "IANA timezone (e.g. America/New_York). Blank uses the system timezone.",
        "options": _timezone_options(),
        "searchable": True,
        "clearable": True,
    },
    "memory.provider": {
        "type": "select",
        "description": "Memory provider plugin",
        "options": _memory_provider_options(),
    },
    "model": {
        "type": "string",
        "description": "Default model (e.g. anthropic/claude-sonnet-4.6)",
        "category": "general",
    },
    "model_context_length": {
        "type": "number",
        "description": "Context window override (0 = auto-detect from model metadata)",
        "category": "general",
    },
    "terminal.backend": {
        "type": "select",
        "description": "Terminal execution backend",
        "options": ["local", "docker", "ssh", "modal", "daytona", "vercel_sandbox", "singularity"],
    },
    "terminal.vercel_runtime": {
        "type": "select",
        "description": "Vercel Sandbox runtime",
        "options": ["node24", "node22", "python3.13"],  # sync with _SUPPORTED_VERCEL_RUNTIMES in terminal_tool.py
    },
    "terminal.modal_mode": {
        "type": "select",
        "description": "Modal sandbox mode",
        "options": ["sandbox", "function"],
    },
    "proxy.enabled": {
        "type": "boolean",
        "description": (
            "Docker-only egress credential firewall. Requires `hermes egress setup` "
            "and `hermes egress start`; Modal/SSH/Daytona are not wired yet."
        ),
        "category": "security",
    },
    "proxy.credential_source": {
        "type": "select",
        "description": "Where iron-proxy loads real upstream secrets at start time",
        "options": ["env", "bitwarden"],
        "category": "security",
    },
    "proxy.enforce_on_docker": {
        "type": "boolean",
        "description": "Refuse Docker sandboxes when egress is enabled but not configured/running",
        "category": "security",
    },
    "tts.provider": {
        "type": "select",
        "description": "Text-to-speech provider",
        "options": ["edge", "elevenlabs", "openai", "xai", "minimax", "mistral", "gemini", "neutts", "kittentts", "piper"],
    },
    "stt.provider": {
        "type": "select",
        "description": "Speech-to-text provider",
        # "mistral" temporarily removed — mistralai PyPI package quarantined
        # (malicious 2.4.6 release on 2026-05-12). Restore once available.
        "options": ["local", "groq", "openai", "xai", "elevenlabs"],
    },
    "stt.local.model": {
        "type": "select",
        "description": "Local faster-whisper model size",
        "options": ["tiny", "base", "small", "medium", "large-v3"],
    },
    "stt.groq.model": {
        "type": "select",
        "description": "Groq Whisper model",
        "options": ["whisper-large-v3-turbo", "whisper-large-v3", "distil-whisper-large-v3-en"],
    },
    "stt.openai.model": {
        "type": "select",
        "description": "OpenAI transcription model",
        "options": ["whisper-1", "gpt-4o-mini-transcribe", "gpt-4o-transcribe", "gpt-transcribe"],
    },
    "stt.elevenlabs.model_id": {
        "type": "select",
        "description": "ElevenLabs Scribe model",
        "options": ["scribe_v2", "scribe_v1"],
    },
    "display.skin": {
        "type": "select",
        "description": "CLI visual theme",
        "options": ["default", "ares", "mono", "slate"],
    },
    "dashboard.theme": {
        "type": "select",
        "description": "Web dashboard visual theme",
        "options": ["default", "midnight", "ember", "mono", "cyberpunk", "rose"],
    },
    "display.resume_display": {
        "type": "select",
        "description": "How resumed sessions display history",
        "options": ["minimal", "full", "off"],
    },
    "display.busy_input_mode": {
        "type": "select",
        "description": "Input behavior while agent is running",
        "options": ["interrupt", "queue", "steer"],
    },
    "approvals.mode": {
        "type": "select",
        "description": "Dangerous command approval mode",
        "options": ["manual", "smart", "off"],
    },
    "context.engine": {
        "type": "select",
        "description": "Context management engine",
        "options": ["default", "custom"],
    },
    "human_delay.mode": {
        "type": "select",
        "description": "Simulated typing delay mode",
        "options": ["off", "typing", "fixed"],
    },
    "logging.level": {
        "type": "select",
        "description": "Log level for agent.log",
        "options": ["DEBUG", "INFO", "WARNING", "ERROR"],
    },
    "agent.service_tier": {
        "type": "select",
        "description": "Fast mode: fast = always, auto = first N seconds of each turn, cold = first turn only",
        "options": ["", "normal", "fast", "auto", "cold"],
    },
    "delegation.reasoning_effort": {
        "type": "select",
        "description": "Reasoning effort for delegated subagents",
        "options": ["", "minimal", "low", "medium", "high", "xhigh", "max", "ultra"],
    },
    "updates.non_interactive_local_changes": {
        "type": "select",
        "description": (
            "When the chat app / gateway updates Hermes (no terminal prompt), "
            "what to do with uncommitted local source edits. 'stash' keeps them "
            "and re-applies them after the update; 'discard' throws them away. "
            "Terminal updates always ask, regardless of this setting."
        ),
        "options": ["stash", "discard"],
    },
    "updates.refresh_cua_driver": {
        "type": "boolean",
        "description": (
            "Refresh an already-installed cua-driver during hermes update. "
            "Disable this on non-admin macOS accounts where /Applications is "
            "not writable."
        ),
    },
    "browser.headed": {
        "type": "boolean",
        "description": "Run the local browser in headed mode (visible window). Also keeps the window open between turns; idle sessions are still reaped after browser.inactivity_timeout.",
    },
    "plugins.hook_callback_timeout": {
        "type": "number",
        "description": (
            "Wall-clock cap (seconds) for timeout-bounded in-process Python "
            "plugin hook callbacks (hot-path observers + pre_tool_call). "
            "Timed-out pre_tool_call fails closed. 0 disables the cap; "
            "values above 600 are clamped. Caller-thread hooks such as "
            "subagent_stop are never moved onto a timeout worker."
        ),
    },
}

# Categories with fewer fields get merged into "general" to avoid tab sprawl.
_CATEGORY_MERGE: Dict[str, str] = {
    "privacy": "security",
    "context": "agent",
    "skills": "agent",
    "cron": "agent",
    "network": "agent",
    # `models_dev.url` (mirror override) is the only schema-surfaced
    # models_dev field — fold it in with the other network/agent plumbing
    # rather than spawning a one-field orphan tab.
    "models_dev": "agent",
    "checkpoints": "agent",
    "approvals": "security",
    "human_delay": "display",
    "dashboard": "display",
    "code_execution": "agent",
    "prompt_caching": "agent",
    # bot_mode holds a couple of relay tuning knobs — keep it folded into the
    # agent tab rather than spawning a tiny standalone category.
    "bot_mode": "agent",
    "goals": "agent",
    "updates": "general",
    # `onboarding.profile_build` is the only schema-surfaced onboarding field
    # (`onboarding.seen` is an internal latch dict, not a user setting), so fold
    # it into the agent tab rather than spawning a one-field orphan category.
    "onboarding": "agent",
    # Only `telegram.reactions` currently lives under telegram — fold it in
    # with the other messaging-platform config (discord) so it isn't an
    # orphan tab of one field.
    "telegram": "discord",
    # `mcp.auto_reload_on_config_change` is the only schema-surfaced mcp
    # runtime field (server definitions live under mcp_servers, edited via
    # the MCP tab) — fold it into the agent tab rather than spawning a
    # one-field orphan category.
    "mcp": "agent",
    # `computer_use.cua_telemetry` is the only schema-surfaced computer_use
    # field — fold it into the agent tab rather than spawning a one-field
    # orphan category.
    "computer_use": "agent",
    # `telemetry.shared_metrics.enabled` is the only schema-surfaced telemetry
    # field — fold it into security alongside the other privacy-posture toggles.
    "telemetry": "security",
    # `plugins.hook_callback_timeout` is the only schema-surfaced plugins field
    # (`enabled`/`disabled` are list allow-lists omitted from DEFAULT_CONFIG) —
    # fold it into the agent tab rather than spawning a one-field orphan category.
    "plugins": "agent",
    # `doctor.live_probe_timeout` is the only schema-surfaced doctor field —
    # fold it into general rather than spawning a one-field orphan category.
    "doctor": "general",
    # `runtime.nofile_soft_limit` (#78873) is the only schema-surfaced runtime
    # field — fold it into the agent tab rather than spawning a one-field
    # orphan category.
    "runtime": "agent",
    # `session.terminal_continue` is the only schema-surfaced session field —
    # fold it into general rather than spawning a one-field orphan category.
    "session": "general",
    # `nous.keepalive_interval_seconds` is the only schema-surfaced nous field
    # (Portal tokens live in auth.json) — fold it into the agent tab.
    "nous": "agent",
}


def _infer_type(value: Any) -> str:
    """Infer a UI field type from a Python value."""
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "number"
    if isinstance(value, float):
        return "number"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "object"
    return "string"


def _build_schema_from_config(
    config: Dict[str, Any],
    prefix: str = "",
) -> Dict[str, Dict[str, Any]]:
    """Walk DEFAULT_CONFIG and produce a flat dot-path → field schema dict."""
    schema: Dict[str, Dict[str, Any]] = {}
    for key, value in config.items():
        full_key = f"{prefix}.{key}" if prefix else key

        # Skip internal / version keys
        if full_key in {"_config_version"}:
            continue

        # Category is the first path component for nested keys, or "general"
        # for top-level scalar fields (model, toolsets, timezone, etc.).
        if prefix:
            category = prefix.split(".")[0]
        elif isinstance(value, dict):
            category = key
        else:
            category = "general"

        if isinstance(value, dict):
            # Recurse into nested dicts
            schema.update(_build_schema_from_config(value, full_key))
        else:
            entry: Dict[str, Any] = {
                "type": _infer_type(value),
                "description": full_key.replace(".", " → ").replace("_", " ").title(),
                "category": category,
            }
            # Apply manual overrides
            if full_key in _SCHEMA_OVERRIDES:
                entry.update(_SCHEMA_OVERRIDES[full_key])
            # Merge small categories
            entry["category"] = _CATEGORY_MERGE.get(entry["category"], entry["category"])
            schema[full_key] = entry
    return schema


CONFIG_SCHEMA = _build_schema_from_config(DEFAULT_CONFIG)

# Inject virtual fields that don't live in DEFAULT_CONFIG but are surfaced
# by the normalize/denormalize cycle.  Insert model_context_length right after
# the "model" key so it renders adjacent in the frontend.
_mcl_entry = _SCHEMA_OVERRIDES["model_context_length"]
_ordered_schema: Dict[str, Dict[str, Any]] = {}
for _k, _v in CONFIG_SCHEMA.items():
    _ordered_schema[_k] = _v
    if _k == "model":
        _ordered_schema["model_context_length"] = _mcl_entry
CONFIG_SCHEMA = _ordered_schema


def _is_command_provider_block(value: Any) -> bool:
    """Return True when *value* declares a command-type voice provider.

    Mirrors the runtime discriminators
    (``tools.tts_tool._is_command_provider_config`` /
    ``tools.transcription_tools._is_command_stt_provider_config``) and the
    desktop's ``isCommandProvider`` in
    ``apps/desktop/src/app/settings/helpers.ts``: ``type`` is OPTIONAL and
    case/space-insensitive (absent or normalizing to ``"command"``), and
    ``command`` MUST be a non-empty string. Built-in blocks (which carry
    ``voice``/``model`` and no ``command``) and the ``providers`` container
    itself are rejected.
    """
    if not isinstance(value, dict):
        return False
    ptype = str(value.get("type") or "").strip().lower()
    if ptype and ptype != "command":
        return False
    command = value.get("command")
    return isinstance(command, str) and bool(command.strip())


def _custom_provider_options(
    kind: str,
    builtin_names: List[str],
    cfg: Dict[str, Any],
) -> List[str]:
    """Return a merged provider option list without hard-coding vendor names.

    *kind* is ``"tts"`` or ``"stt"``. The result keeps the built-in display
    names first (original order — NOT re-sorted), then appends:

    1. Command-type providers declared under the canonical
       ``<kind>.providers.<name>`` location, plus the legacy top-level
       ``<kind>.<name>`` fallback — exactly the dual resolution the runtime
       performs in ``_get_named_provider_config`` /
       ``_get_named_stt_provider_config``. Names colliding with a RUNTIME
       built-in are excluded case-insensitively (the runtime rejects a
       built-in name as a command provider before any config lookup), so a
       ``providers.EDGE`` command block is not offered.
    2. Plugin-registered provider names from ``agent.tts_registry`` /
       ``agent.transcription_registry`` — opportunistic only: plugins
       register at runtime via ``ctx.register_tts_provider()``, and this
       process does not necessarily call ``discover_plugins()``, so the
       registry may legitimately be empty here. (There is no static
       ``provides: [tts]`` manifest convention to scan — real manifests only
       carry ``provides_tools``/``provides_hooks``.)
    3. The current ``<kind>.provider`` value when not already present — a
       custom name that only appears as the active provider stays
       selectable (matches desktop ``enumOptionsFor``'s current-value
       preservation).

    Guard semantics deliberately mirror
    ``apps/desktop/src/app/settings/helpers.ts:commandProviderNames`` so the
    backend schema (web dashboard) and the desktop client agree on which
    names are offered.
    """
    names = [str(n) for n in builtin_names]
    seen = {n.strip().lower() for n in names}

    # Guard against the RUNTIME built-in sets, not the display shortlist
    # above: the display list drifts from the runtime sets (e.g. omits
    # ``deepinfra``), and filtering on it would offer names the runtime
    # would never honour as command providers.
    if kind == "tts":
        from tools.tts_tool import BUILTIN_TTS_PROVIDERS as _runtime_builtins
    else:
        from tools.transcription_tools import BUILTIN_STT_PROVIDERS as _runtime_builtins

    def _add(name: Any) -> None:
        if not isinstance(name, str):
            return
        stripped = name.strip()
        key = stripped.lower()
        if stripped and key not in seen:
            names.append(stripped)
            seen.add(key)

    section = cfg.get(kind)
    if not isinstance(section, dict):
        section = {}

    # Canonical nested location first, then the legacy top-level fallback —
    # the same order the runtime resolves them in.
    candidate_blocks: List[Any] = []
    providers_map = section.get("providers")
    if isinstance(providers_map, dict):
        candidate_blocks.append(providers_map)
    candidate_blocks.append(
        {k: v for k, v in section.items() if k != "providers"}
    )
    for block in candidate_blocks:
        for name, value in block.items():
            if (
                isinstance(name, str)
                and name.strip().lower() not in _runtime_builtins
                and _is_command_provider_block(value)
            ):
                _add(name)

    # Plugin-registered providers (only populated when plugins are loaded in
    # this process). Registry names can never collide with built-ins — the
    # registries reject such registrations.
    try:
        if kind == "tts":
            from agent.tts_registry import list_providers as _list_voice_providers
        else:
            from agent.transcription_registry import list_providers as _list_voice_providers
        for _p in _list_voice_providers():
            _add(getattr(_p, "name", None))
    except Exception:  # pragma: no cover - registry import should not break schema
        pass

    # Current-value preservation (``cfg_get`` takes *keys*, not dotted paths).
    _add(cfg_get(cfg, kind, "provider"))

    return names


def _memory_provider_schema_options(cfg: Dict[str, Any]) -> List[str]:
    """Discovered memory providers for a per-request schema merge.

    Reuses the cheap directory scan of :func:`_memory_provider_options` and
    additionally preserves the currently-configured provider, so a value
    selected in config but not (yet) discoverable — e.g. a plugin removed from
    disk — never silently vanishes from the dropdown.
    """
    options = _memory_provider_options()

    memory = cfg.get("memory")
    configured = memory.get("provider") if isinstance(memory, dict) else None
    current = _normalize_memory_provider_name(configured)

    if current and current not in options:
        options = [*options, current]

    return options


def _schema_with_dynamic_provider_options() -> Dict[str, Dict[str, Any]]:
    """Return CONFIG_SCHEMA with per-request discovery-driven options merged.

    Some ``*.provider`` selects have options that are discovered at runtime
    (voice backends via the tts/stt registries + config.yaml command
    providers; memory providers via a plugin-dir scan). The module-level
    ``_SCHEMA_OVERRIDES`` freezes those lists at import time, so a provider
    installed after the server started never appears. This recomputes them at
    request time — reflecting the CURRENT config.yaml, the profile-scoped
    config when the request carries a ``profile`` param, and mid-session
    plugin installs — for every surface that reads the schema (desktop, CLI,
    dashboard), with no extra frontend round-trips.

    The module-level ``CONFIG_SCHEMA`` is never mutated; entries that change
    are shallow-copied onto a copied mapping.
    """
    try:
        cfg = load_config()
    except Exception:  # pragma: no cover - schema must survive config errors
        return CONFIG_SCHEMA

    overlay: Dict[str, Dict[str, Any]] = {}

    def merge(key: str, options: List[str]) -> None:
        entry = CONFIG_SCHEMA.get(key)

        if isinstance(entry, dict) and isinstance(entry.get("options"), list) and options != entry["options"]:
            overlay[key] = {**entry, "options": options}

    for kind in ("tts", "stt"):
        entry = CONFIG_SCHEMA.get(f"{kind}.provider")
        existing = entry.get("options") if isinstance(entry, dict) else None

        if isinstance(existing, list):
            merge(f"{kind}.provider", _custom_provider_options(kind, list(existing), cfg))

    merge("memory.provider", _memory_provider_schema_options(cfg))

    tb_entry = CONFIG_SCHEMA.get("terminal.backend")
    if isinstance(tb_entry, dict) and isinstance(tb_entry.get("options"), list):
        try:
            plugin_names = sorted(
                {row["name"] for row in _plugin_terminal_backend_rows()}
                - set(tb_entry["options"])
            )
        except Exception:
            plugin_names = []
        if plugin_names:
            merge("terminal.backend", [*tb_entry["options"], *plugin_names])

    if not overlay:
        return CONFIG_SCHEMA

    return {**CONFIG_SCHEMA, **overlay}


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


def _normalize_main_model_assignment(provider: str, model: str) -> tuple[str, str]:
    """Normalize a main-slot (provider, model) pair before persisting.

    The Models page has two assignment paths and only one of them was safe:

    - The "Change" picker sends a real Hermes provider slug — fine.
    - The per-card "Use as → Main model" menu sends ``entry.provider``
      from the analytics rows, falling back to the model's VENDOR prefix
      (``modelVendor("anthropic/claude-opus-4.6") == "anthropic"``) when
      the session row has no ``billing_provider`` (older sessions, NULL
      rows).  That wrote ``provider: anthropic`` +
      ``default: anthropic/claude-opus-4.6`` to config — a vendor-prefixed
      OpenRouter slug on the NATIVE Anthropic provider.  New sessions then
      400 against api.anthropic.com ("model: anthropic/claude-opus-4.6 not
      found") and the user reads it as "changing models does nothing".

    Two repairs, both at this single chokepoint so every caller inherits:

    1. Vendor-name → Hermes-provider mapping: when the provider string is
       not a known Hermes provider/alias (e.g. ``moonshotai``, ``x-ai`` is
       known but ``poolside`` isn't) but the model is a vendor-prefixed
       aggregator slug, keep the user's CURRENT aggregator if they're on
       one, else fall back to openrouter.

       Named custom providers (``custom:litellm``, etc.) are excluded from
       this fallback: ``_KNOWN_PROVIDER_NAMES`` only lists the bare
       ``"custom"`` bucket, never a specific ``custom:<name>`` slug, so
       without this exclusion every named custom provider paired with a
       slash-bearing model (e.g. ``ollama/glm-5.2`` behind a LiteLLM proxy)
       looked exactly like the stray-vendor-prefix case above and got
       silently reassigned to ``openrouter``.
    2. Model-format normalization for the resolved provider via
       ``normalize_model_for_provider`` (e.g. ``anthropic/claude-opus-4.6``
       on native anthropic → ``claude-opus-4-6``).
    """
    from hermes_cli.config import get_compatible_custom_providers
    from hermes_cli.models import _KNOWN_PROVIDER_NAMES, normalize_provider
    from hermes_cli.model_normalize import normalize_model_for_provider
    from hermes_cli.providers import resolve_custom_provider, resolve_user_provider

    prov_in = (provider or "").strip()
    model_in = (model or "").strip()
    canonical = normalize_provider(prov_in)

    # User-declared providers are real routing targets, not analytics vendor
    # labels. Resolve them before the unknown-vendor fallback. ``providers:``
    # keeps its declared bare slug; ``custom_providers:`` canonicalizes both a
    # bare display name and ``custom:<name>`` to the durable custom slug.
    try:
        cfg = load_config()
    except Exception:
        cfg = {}
    user_providers = cfg.get("providers") if isinstance(cfg, dict) else None
    user_provider = resolve_user_provider(
        prov_in, user_providers if isinstance(user_providers, dict) else {}
    )
    custom_provider = resolve_custom_provider(
        prov_in,
        get_compatible_custom_providers(cfg) if isinstance(cfg, dict) else [],
    )
    if user_provider is not None:
        return user_provider.id, model_in
    if custom_provider is not None:
        return custom_provider.id, model_in

    # A named custom provider that didn't resolve above (typo, config
    # mismatch, entry missing from custom_providers/providers) must still
    # not be treated as a stray vendor prefix -- it isn't a known Hermes
    # provider/alias, but it also isn't the analytics-vendor case this
    # fallback exists for. Match only the durable named-custom syntax
    # (bare "custom" bucket, or "custom:<name>" per
    # ``providers.custom_provider_slug``) -- a bare ``startswith("custom")``
    # would also swallow unrelated unconfigured vendor names that merely
    # happen to start with "custom" (e.g. "customproxy").
    is_custom_provider_slug = canonical == "custom" or canonical.startswith("custom:")
    if (
        canonical not in _KNOWN_PROVIDER_NAMES
        and not is_custom_provider_slug
        and "/" in model_in
    ):
        # Vendor prefix posing as a provider (analytics fallback). Resolve
        # against the user's current provider when it's an aggregator that
        # serves vendor-prefixed slugs; otherwise default to openrouter.
        try:
            cur_cfg = cfg.get("model", {})
            cur_provider = (
                str(cur_cfg.get("provider", "") or "").strip().lower()
                if isinstance(cur_cfg, dict) else ""
            )
        except Exception:
            cur_provider = ""
        from hermes_cli.models import _AGGREGATOR_PROVIDERS
        if cur_provider and normalize_provider(cur_provider) in _AGGREGATOR_PROVIDERS:
            canonical = normalize_provider(cur_provider)
            prov_in = cur_provider
        else:
            canonical = "openrouter"
            prov_in = "openrouter"

    # Custom/user-config providers keep the model verbatim — the registry
    # normalizer doesn't know their namespaces.
    if canonical in _KNOWN_PROVIDER_NAMES and not canonical.startswith("custom"):
        try:
            normalized_model = normalize_model_for_provider(model_in, canonical)
            if normalized_model:
                model_in = normalized_model
        except Exception:
            _log.debug("model normalization failed for %s/%s", prov_in, model_in, exc_info=True)

    return prov_in, model_in


def _apply_main_model_assignment(
    model_cfg: "Any", provider: str, model: str, base_url: str = "", api_key: str = ""
) -> dict:
    """Apply a main-slot model assignment to a ``model`` config dict in place.

    Sets ``provider``/``default``, then reconciles ``base_url``:

    - An explicitly supplied ``base_url`` is always persisted (covers
      ``custom``/local endpoints and any provider whose key is bound to a
      non-default host).
    - Otherwise, a stale ``base_url`` is cleared ONLY when switching to a
      *different* provider — that URL belonged to the old provider. When the
      provider is unchanged and no new URL is supplied, the existing
      ``base_url`` is preserved. This keeps a user's custom endpoint (e.g. a
      Xiaomi MiMo Token Plan host, ``https://token-plan-*.xiaomimimo.com/v1``)
      alive when they merely re-pick a model under the same provider — picking
      a model previously wiped it, forcing the registry default and breaking
      Token Plan keys.

    The runtime resolver reads ``model.base_url`` from config (it ignores
    ``OPENAI_BASE_URL``) and only honors it when the configured provider matches
    and the pool entry is on the registry default, so preserving it here is what
    lets the override actually route. The hardcoded ``context_length`` override
    is always dropped since the new model may have a different context window.

    Returns the same dict (coerced to a fresh dict if the input wasn't one) so
    callers can assign it straight back onto the model config.
    """
    if not isinstance(model_cfg, dict):
        model_cfg = {}
    prev_provider = str(model_cfg.get("provider") or "").strip().lower()
    new_provider = provider.strip().lower()
    model_cfg["provider"] = provider
    model_cfg["default"] = model
    if base_url.strip():
        model_cfg["base_url"] = base_url.strip()
    elif model_cfg.get("base_url") and new_provider != prev_provider:
        # Switching providers: the old URL belonged to the old provider, drop
        # it so the new provider's default endpoint is used. Same-provider
        # re-assignment keeps the user's configured base_url intact.
        model_cfg["base_url"] = ""
    # The endpoint key follows the same lifecycle as base_url: an explicit key
    # is always persisted; an existing key is dropped only when switching to a
    # different provider (it belonged to the old endpoint), and preserved on a
    # same-provider re-pick so re-selecting a model doesn't wipe the key.
    if api_key.strip():
        model_cfg["api_key"] = api_key.strip()
        model_cfg.pop("api", None)
    elif (model_cfg.get("api_key") or model_cfg.get("api")) and new_provider != prev_provider:
        # A stale endpoint secret can live under the legacy ``api`` alias with
        # no ``api_key`` (the resolver still reads ``model.api`` as a key), so
        # the switch-clears-the-key path must trigger on either field — else the
        # old endpoint's secret survives in config.yaml and contaminates a later
        # custom resolution. clear_model_endpoint_credentials scrubs both.
        clear_model_endpoint_credentials(model_cfg, clear_api_mode=False)
    if new_provider != prev_provider:
        clear_model_endpoint_credentials(model_cfg, clear_api_key=False)
    model_cfg.pop("context_length", None)
    return model_cfg


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


# DEPRECATED (scheduled for removal): GATEWAY_HEALTH_URL / GATEWAY_HEALTH_TIMEOUT.
# Cross-container / cross-host gateway liveness detection will be folded into a
# first-class dashboard config key so it's no longer Docker-adjacent lore buried
# in env vars.  The env vars still work for now so existing Compose deployments
# don't break.  Do not add new callers — wire new uses through the planned
# config surface.


def _probe_gateway_health() -> tuple[bool, dict | None]:
    """Probe the gateway via its HTTP health endpoint (cross-container).

    .. deprecated::
        Driven by the deprecated ``GATEWAY_HEALTH_URL`` /
        ``GATEWAY_HEALTH_TIMEOUT`` env vars.  Scheduled for removal alongside
        a move to a first-class dashboard config key.  See
        :data:`_GATEWAY_HEALTH_URL` for context.

    Uses ``/health/detailed`` first (returns full state), falling back to
    the simpler ``/health`` endpoint.  Returns ``(is_alive, body_dict)``.

    Accepts any of these as ``GATEWAY_HEALTH_URL``:
    - ``http://gateway:8642``                (base URL — recommended)
    - ``http://gateway:8642/health``         (explicit health path)
    - ``http://gateway:8642/health/detailed`` (explicit detailed path)

    This is a **blocking** call — run via ``run_in_executor`` from async code.
    """
    if not _GATEWAY_HEALTH_URL:
        return False, None

    # Normalise to base URL so we always probe the right paths regardless of
    # whether the user included /health or /health/detailed in the env var.
    base = _GATEWAY_HEALTH_URL.rstrip("/")
    if base.endswith("/health/detailed"):
        base = base[: -len("/health/detailed")]
    elif base.endswith("/health"):
        base = base[: -len("/health")]

    for path in (f"{base}/health/detailed", f"{base}/health"):
        try:
            req = urllib.request.Request(path, method="GET")
            with urllib.request.urlopen(req, timeout=_GATEWAY_HEALTH_TIMEOUT) as resp:
                if resp.status == 200:
                    body = json.loads(resp.read())
                    return True, body
        except Exception:
            continue
    return False, None


_MANAGED_FILES_ROOT_ENV = "HERMES_DASHBOARD_FILES_ROOT"
_MANAGED_FILE_MAX_BYTES = 100 * 1024 * 1024
_HOSTED_MANAGED_FILES_ROOT = Path("/opt/data")


@dataclass(frozen=True)
class ManagedFilesPolicy:
    default_path: Path
    locked_root: Path | None
    can_change_path: bool


from hermes_cli.web_routers import files as _files_routes  # noqa: E402

app.include_router(_files_routes.router)
from hermes_cli.web_routers.files import (  # noqa: E402,F401 — legacy re-exports; tests call these via web_server.<name>
    get_media,
    upload_managed_file_stream,
)


_FS_DATA_URL_MAX_BYTES = 16 * 1024 * 1024


def _fs_path(raw_path: str) -> Path:
    raw = str(raw_path or "").strip()
    if not raw:
        raise HTTPException(status_code=400, detail="Path is required")
    if "\0" in raw:
        raise HTTPException(status_code=400, detail="Invalid path")
    try:
        if raw.lower().startswith("file:"):
            parsed = urllib.parse.urlparse(raw)
            if parsed.netloc and parsed.netloc not in {"", "localhost"}:
                raise ValueError
            raw = urllib.request.url2pathname(parsed.path)
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
        return candidate.resolve(strict=False)
    except (OSError, RuntimeError, ValueError):
        raise HTTPException(status_code=400, detail="Invalid path")


def _canonical_path(path: Path, *, require_exists: bool = False) -> Path:
    try:
        return path.expanduser().resolve(strict=require_exists)
    except FileNotFoundError:
        if require_exists:
            raise HTTPException(status_code=404, detail="Path not found")
        raise
    except (OSError, RuntimeError):
        raise HTTPException(status_code=400, detail="Invalid path")


def _ensure_managed_root(raw_path: str | Path) -> Path:
    root = Path(raw_path).expanduser()
    try:
        root.mkdir(parents=True, exist_ok=True)
        resolved = root.resolve()
    except (OSError, RuntimeError) as exc:
        raise HTTPException(status_code=500, detail=f"Managed files root is unavailable: {exc}")
    if not resolved.is_dir():
        raise HTTPException(status_code=500, detail="Managed files root is not a directory")
    return resolved


def _path_is_under(root: Path, target: Path) -> bool:
    return target == root or root in target.parents


def _path_text(raw_path: str | None) -> str:
    text = str(raw_path or "").strip()
    if "\x00" in text:
        raise HTTPException(status_code=400, detail="Invalid path")
    return text


def _default_hermes_root_is_opt_data() -> bool:
    raw = os.environ.get("HERMES_HOME", "").strip()
    if not raw:
        return False
    try:
        from hermes_constants import get_default_hermes_root

        root = get_default_hermes_root().expanduser().resolve(strict=False)
    except (OSError, RuntimeError):
        root = Path(raw).expanduser().resolve(strict=False)
    return root == _HOSTED_MANAGED_FILES_ROOT


def _dashboard_local_update_managed_externally() -> bool:
    """Return true when the dashboard should not offer ``hermes update``.

    Containerized dashboards are updated by the outer launcher/image, not by an
    in-browser local update action. Keep this dashboard capability separate
    from install-method detection: manual git/pip installs inside containers can
    still behave like their actual install method in the CLI.

    However, when the install method is ``git`` (a bind-mounted checkout inside
    a container — e.g. the hermes-webui image sharing the Hermes source tree),
    the dashboard's ``hermes update`` button is the correct update path and
    should not be suppressed. Other containerized install methods remain
    externally managed unless their apply path is proven safe inside the
    running container filesystem.
    """
    if _default_hermes_root_is_opt_data():
        return True
    try:
        from hermes_constants import is_container

        if not is_container():
            return False
    except Exception:
        return False
    # We are inside a container, but the install may still be self-managed.
    # If the install method is git, the dashboard update button works against
    # the mounted checkout and should be offered. Keep pip blocked inside
    # containers: its apply path mutates the running container filesystem and
    # is not the bind-mounted checkout case this gate is meant to recover.
    try:
        method = detect_install_method(PROJECT_ROOT)
        if method == "git":
            return False
    except Exception:
        pass
    return True


def _managed_files_policy(request: Request, *, create_root: bool = True) -> ManagedFilesPolicy:
    raw_forced_root = os.environ.get(_MANAGED_FILES_ROOT_ENV, "").strip()
    if raw_forced_root:
        root = _ensure_managed_root(raw_forced_root) if create_root else _canonical_path(Path(raw_forced_root))
        return ManagedFilesPolicy(default_path=root, locked_root=root, can_change_path=False)

    # Remote/OAuth access does not imply a hosted container. Users can expose a
    # local dashboard through the auth gate (for example a macOS launchd install)
    # and still expect the Files page to browse their local home directory. Lock
    # to /opt/data only when the installation's Hermes root is actually /opt/data
    # (the container/hosted layout) or when HERMES_DASHBOARD_FILES_ROOT is set.
    if _default_hermes_root_is_opt_data():
        root = _ensure_managed_root(_HOSTED_MANAGED_FILES_ROOT) if create_root else _HOSTED_MANAGED_FILES_ROOT
        return ManagedFilesPolicy(default_path=root, locked_root=root, can_change_path=False)

    home = _canonical_path(Path.home())
    return ManagedFilesPolicy(default_path=home, locked_root=None, can_change_path=True)


def _resolve_managed_path(
    raw_path: str | None,
    request: Request,
    *,
    for_write: bool = False,
) -> tuple[ManagedFilesPolicy, Path, str]:
    policy = _managed_files_policy(request)
    text = _path_text(raw_path)
    root = policy.locked_root

    if root is not None and (not text or text in {".", "/"}):
        candidate = root
    elif not text:
        candidate = policy.default_path
    else:
        candidate = Path(text).expanduser()
        if root is not None and not candidate.is_absolute():
            if any(part == ".." for part in candidate.parts):
                raise HTTPException(status_code=400, detail="Path cannot contain '..'")
            candidate = root / candidate
        elif not candidate.is_absolute():
            raise HTTPException(status_code=400, detail="Path must be absolute")

    if ".." in candidate.parts:
        raise HTTPException(status_code=400, detail="Path cannot contain '..'")

    if for_write and not candidate.exists():
        parent = _canonical_path(candidate.parent)
        resolved = parent / candidate.name
    else:
        resolved = _canonical_path(candidate, require_exists=not for_write)

    if root is not None and not _path_is_under(root, resolved):
        raise HTTPException(status_code=403, detail="Path outside managed files root")

    return policy, resolved, str(resolved)


def _managed_response_meta(policy: ManagedFilesPolicy) -> Dict[str, Any]:
    locked_root = str(policy.locked_root) if policy.locked_root is not None else None
    return {
        "root": locked_root,
        "locked_root": locked_root,
        "can_change_path": policy.can_change_path,
    }


def _managed_file_entry(policy: ManagedFilesPolicy, target: Path) -> Dict[str, Any]:
    try:
        resolved = target.resolve()
    except (OSError, RuntimeError):
        raise HTTPException(status_code=400, detail="Invalid path")
    if policy.locked_root is not None and not _path_is_under(policy.locked_root, resolved):
        raise HTTPException(status_code=403, detail="Path outside managed files root")

    try:
        st = resolved.stat()
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Could not stat path: {exc}")

    is_dir = resolved.is_dir()
    mime_type = None if is_dir else (mimetypes.guess_type(resolved.name)[0] or "application/octet-stream")
    return {
        "name": target.name or resolved.name or str(resolved),
        "path": str(resolved),
        "is_directory": is_dir,
        "size": None if is_dir else st.st_size,
        "mtime": st.st_mtime,
        "mime_type": mime_type,
    }


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


# Host TCP ports each port-binding gateway platform listens on, as
# ``platform-name -> (config port key, adapter default)``.  Mirrors
# ``PORT_BINDING_PLATFORM_VALUES`` in gateway/config.py and each adapter's
# DEFAULT_PORT / DEFAULT_WEBHOOK_PORT constant.  Used only for the dashboard's
# gateway-topology readout — best-effort display data, not a bind source.
_PORT_BINDING_PLATFORM_PORTS: Dict[str, Tuple[str, int]] = {
    "webhook": ("port", 8644),
    "api_server": ("port", 8642),
    "msgraph_webhook": ("port", 8646),
    "feishu": ("webhook_port", 8765),
    "wecom_callback": ("port", 8645),
    "bluebubbles": ("webhook_port", 8645),
    "sms": ("webhook_port", 8080),
    "whatsapp_cloud": ("webhook_port", 8090),
    "line": ("port", 8646),
    "teams": ("port", 3978),
}

# Platform states that mean the adapter is NOT serving its port right now.
_PLATFORM_DEAD_STATES = frozenset({"fatal", "disconnected", "stopped"})


def _profile_platform_ports(profile_home: Path, runtime: Optional[dict]) -> Dict[str, int]:
    """Best-effort map of ``platform -> host TCP port`` for one profile's gateway.

    Reads the platforms the running gateway reported in its
    ``gateway_state.json`` and resolves each port-binding platform's port from
    the profile's ``config.yaml`` (top-level ``platforms:`` wins over
    ``gateway.platforms:``, matching ``load_gateway_config`` precedence),
    falling back to the adapter default.  Display-only: env-var port overrides
    (e.g. ``WEBHOOK_PORT`` in that profile's .env) are not resolved here.
    """
    platforms = (runtime or {}).get("platforms") or {}
    active = [
        name for name, state in platforms.items()
        if name in _PORT_BINDING_PLATFORM_PORTS
        and isinstance(state, dict)
        and state.get("state") not in _PLATFORM_DEAD_STATES
    ]
    if not active:
        return {}

    blocks: Dict[str, dict] = {}
    try:
        # Multi-profile probe: load_config() targets the ACTIVE profile's
        # home, so read the probed profile's file via the raw primitive.
        from hermes_cli.config import read_user_config_raw
        cfg = read_user_config_raw(profile_home / "config.yaml")
        gateway_cfg = cfg.get("gateway") if isinstance(cfg.get("gateway"), dict) else {}
        # gateway.platforms first, top-level platforms second — later wins,
        # matching the precedence in gateway.config.load_gateway_config().
        for src in ((gateway_cfg or {}).get("platforms"), cfg.get("platforms")):
            if not isinstance(src, dict):
                continue
            for plat_name, plat_block in src.items():
                if isinstance(plat_block, dict):
                    blocks.setdefault(plat_name, {}).update(plat_block)
    except Exception:
        blocks = {}

    ports: Dict[str, int] = {}
    for name in active:
        port_key, default_port = _PORT_BINDING_PLATFORM_PORTS[name]
        block = blocks.get(name) or {}
        extra = block.get("extra") if isinstance(block.get("extra"), dict) else {}
        raw = block.get(port_key, (extra or {}).get(port_key, default_port))
        try:
            ports[name] = int(raw)
        except (TypeError, ValueError):
            ports[name] = default_port
    return ports


def _profile_gateway_writer_identity(
    profile_home: Path, runtime: Optional[dict]
) -> Optional[tuple]:
    """``(pid, start_time)`` identity of the profile's LIVE gateway, or None.

    Reuses the validated-liveness helper — recorded PID checked against the
    live process table, the start-time PID-reuse fingerprint, and the
    profile's home — then reads the live process's fingerprint via the same
    ``_get_process_start_time`` that stamped it, so equality is exact (no
    unit or clock-source mismatch).  None when the record doesn't belong to
    a live gateway; nothing in it is current by definition then.
    """
    try:
        from gateway.status import (
            _get_process_start_time,
            get_runtime_status_running_pid,
        )

        pid = get_runtime_status_running_pid(runtime, expected_home=profile_home)
        if pid is None:
            return None
        start_time = _get_process_start_time(pid)
        if start_time is None:
            return None
        return (pid, start_time)
    except Exception:
        return None


def _owned_profile_platforms(
    writer_identity: Optional[tuple], platforms: dict
) -> dict:
    """Keep only platform entries the profile's CURRENT process wrote.

    Gateway startup deliberately preserves plain platform entries in
    ``gateway_state.json`` across restarts (the dashboard keeps showing
    last-known state while adapters reconnect), and the active-profile
    endpoint compensates by filtering them against the current
    configuration.  The cross-profile aggregation has no equivalent config
    context (a profile's platform set depends on tokens in that profile's
    ``.env`` behind its secret scope), so it demands strict process
    ownership instead: ``write_runtime_status`` stamps every platform write
    with the writer's ``(pid, start_time)`` identity, and an entry is
    aggregatable only when that identity equals the profile's live gateway
    process — exact match, no clock heuristics, so an entry written moments
    before a fast restart can never masquerade as current.  A fatal entry
    left behind by a platform the operator has since disabled/removed thus
    stops degrading fleet health as soon as that profile's gateway restarts
    (a config change requires that restart to take effect anyway).  Fail
    closed: entries without a writer identity (legacy records) or records
    with no live process are excluded — aggregation is a supplement, and a
    false "degraded forever" is the worse failure mode.
    """
    if writer_identity is None:
        return {}
    live_pid, live_start = writer_identity
    owned: Dict[str, dict] = {}
    for key, value in platforms.items():
        if not isinstance(value, dict):
            continue
        if (
            value.get("writer_pid") == live_pid
            and value.get("writer_start_time") == live_start
        ):
            owned[key] = value
    return owned


def _collect_profile_gateway_topology() -> Dict[str, Any]:
    """Enumerate profiles and the gateways serving them for ``/api/status``.

    Returns ``{"profiles": [...], "gateway_mode": ..., "gateways": [...]}``:

    * ``profiles`` — every profile on the host (default + named), from
      ``profiles_to_serve(True)`` (the cheap enumeration chokepoint — no
      per-profile config reads or skill counts).
    * ``gateways`` — one entry per profile with a LIVE gateway process:
      ``{"profile", "ports", "served_profiles"?}``.  Liveness reuses
      ``_check_gateway_running`` so this agrees with the profiles sidebar.
    * ``gateway_mode`` — ``"multiplex"`` when the default gateway serves
      multiple profiles (gateway.multiplex_profiles), ``"single"`` for one
      live gateway, ``"multiple"`` for independent per-profile gateways,
      ``"none"`` when nothing is running.
    * ``profile_platforms`` — ``{profile: platforms}`` runtime platform maps
      for each LIVE gateway, ownership-filtered to entries stamped by that
      profile's current process (stale preserved entries for since-removed
      platforms are excluded — see ``_owned_profile_platforms``).  Internal
      aggregation input for ``/api/status`` (independent per-profile gateways
      write failures to their own ``gateway_state.json``, which the
      unparameterized endpoint would otherwise never see).  Never exposed
      directly.
    """
    try:
        from hermes_cli.profiles import _check_gateway_running, profiles_to_serve
        from gateway.status import read_runtime_status
        homes = profiles_to_serve(True)
    except Exception:
        _log.debug("profile/gateway topology enumeration failed", exc_info=True)
        return {
            "profiles": [],
            "gateway_mode": "unknown",
            "gateways": [],
            "profile_platforms": {},
        }

    profile_names = [name for name, _home in homes]
    gateways: List[Dict[str, Any]] = []
    profile_platforms: Dict[str, dict] = {}
    multiplex = False
    for name, home in homes:
        try:
            if not _check_gateway_running(home):
                continue
        except Exception:
            continue
        try:
            runtime = read_runtime_status(home / "gateway_state.json")
        except Exception:
            runtime = None
        served = [str(p) for p in ((runtime or {}).get("served_profiles") or [])]
        if name == "default" and len(served) > 1:
            multiplex = True
        plats = (runtime or {}).get("platforms")
        if isinstance(plats, dict) and plats:
            # Ownership filter: gateway startup preserves plain platform
            # entries across restarts, so the raw map can carry fatal state
            # for platforms the operator has since disabled/removed.  Only
            # entries stamped with the profile's current live process's
            # writer identity are aggregation candidates (see
            # _owned_profile_platforms).
            owned = _owned_profile_platforms(
                _profile_gateway_writer_identity(home, runtime), plats
            )
            if owned:
                profile_platforms[name] = owned
        entry: Dict[str, Any] = {
            "profile": name,
            "ports": _profile_platform_ports(home, runtime),
        }
        if served:
            entry["served_profiles"] = served
        gateways.append(entry)

    if multiplex:
        mode = "multiplex"
    elif len(gateways) > 1:
        mode = "multiple"
    elif len(gateways) == 1:
        mode = "single"
    else:
        mode = "none"

    return {
        "profiles": profile_names,
        "gateway_mode": mode,
        "gateways": gateways,
        "profile_platforms": profile_platforms,
    }


# /api/status is polled ~1/s by the desktop app while it waits for the backend
# (and again by the dashboard badge). Each uncached call above walks 7+ profile
# homes (yaml.safe_load with the pure-Python loader + psutil process-table
# probes + realpath walks) inside the default executor; concurrent polls pile
# up and hold the GIL for 14-16s, starving the event loop — the desktop WS
# never receives gateway.ready and boot fails ("event loop stalled ... GIL
# pressure suspected"). Topology changes on gateway start/stop, so a short TTL
# cache with a collapse lock keeps the scan to one per window. The cache also
# remembers which collector produced the entry: tests monkeypatch
# _collect_profile_gateway_topology per case, and the identity check keeps
# them hermetic without needing a reset hook (a swapped collector is a miss).
_TOPOLOGY_CACHE: Dict[str, Any] = {"ts": 0.0, "data": None, "fn": None}
_TOPOLOGY_CACHE_LOCK = threading.Lock()
_TOPOLOGY_CACHE_TTL = 10.0

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


def _topology_cache_get(fn: Any) -> Optional[Dict[str, Any]]:
    if (
        _TOPOLOGY_CACHE["data"] is not None
        and _TOPOLOGY_CACHE["fn"] is fn
        and time.monotonic() - _TOPOLOGY_CACHE["ts"] < _TOPOLOGY_CACHE_TTL
    ):
        return _TOPOLOGY_CACHE["data"]
    return None


def _collect_profile_gateway_topology_cached() -> Dict[str, Any]:
    fn = _collect_profile_gateway_topology
    cached = _topology_cache_get(fn)
    if cached is not None:
        return cached
    with _TOPOLOGY_CACHE_LOCK:
        cached = _topology_cache_get(fn)
        if cached is not None:
            return cached
        data = fn()
        _TOPOLOGY_CACHE["data"] = data
        _TOPOLOGY_CACHE["fn"] = fn
        _TOPOLOGY_CACHE["ts"] = time.monotonic()
        return data


def _load_configured_gateway_platforms() -> set[str]:
    """Load connected platform names away from the asyncio event loop.

    The first ``load_gateway_config()`` call performs platform discovery and
    can take longer than Desktop's WebSocket connect timeout on Windows.  This
    helper is synchronous by design; ``get_status`` runs it in Starlette's
    worker pool so a concurrent ``/api/ws`` handshake can still complete.
    """
    from gateway.config import load_gateway_config

    gateway_config = load_gateway_config()
    return {platform.value for platform in gateway_config.get_connected_platforms()}


from hermes_cli.web_routers import status as _status_routes  # noqa: E402

app.include_router(_status_routes.router)
from hermes_cli.web_routers.status import (  # noqa: E402,F401 — legacy re-exports; tests call these via web_server.<name>
    get_status,
    run_dump,
)


_WINDOWS_11_MIN_BUILD = 22000


def _windows_build_number(version: str, platform_label: str) -> Optional[int]:
    """Extract the Windows NT build number from stdlib platform strings."""
    for value in (version or "", platform_label or ""):
        match = re.search(r"(?:^|[^\d])10\.0\.(\d{5,})(?:[^\d]|$)", value)
        if not match:
            continue
        try:
            return int(match.group(1))
        except ValueError:
            continue
    return None


def _display_system_platform(
    *,
    system: str,
    release: str,
    version: str,
    platform_label: str,
) -> Dict[str, str]:
    """Return host OS fields for display while preserving stdlib detail."""
    if system == "Windows" and release == "10":
        build = _windows_build_number(version, platform_label)
        if build is not None and build >= _WINDOWS_11_MIN_BUILD:
            platform_label = re.sub(
                r"^Windows-10(?=-)",
                "Windows-11",
                platform_label,
                count=1,
            )
            release = "11"

    return {
        "os": system,
        "os_release": release,
        "os_version": version,
        "platform": platform_label,
    }


# ---------------------------------------------------------------------------
# Gateway + update actions (invoked from the Status page).
#
# Both commands are spawned as detached subprocesses so the HTTP request
# returns immediately.  stdin is closed (``DEVNULL``) so any stray ``input()``
# calls fail fast with EOF rather than hanging forever.  stdout/stderr are
# streamed to a per-action log file under ``~/.hermes/logs/<action>.log`` so
# the dashboard can tail them back to the user.
# ---------------------------------------------------------------------------

_ACTION_LOG_DIR: Path = get_hermes_home() / "logs"

# Short ``name`` (from the URL) → absolute log file path.
_ACTION_LOG_FILES: Dict[str, str] = {
    "gateway-restart": "gateway-restart.log",
    "gateway-start": "gateway-start.log",
    "gateway-stop": "gateway-stop.log",
    "hermes-update": "hermes-update.log",
    "doctor": "action-doctor.log",
    "security-audit": "action-security-audit.log",
    "backup": "action-backup.log",
    "import": "action-import.log",
    "checkpoints-prune": "action-checkpoints-prune.log",
    "skills-install": "action-skills-install.log",
    "skills-uninstall": "action-skills-uninstall.log",
    "skills-update": "action-skills-update.log",
    "curator-run": "action-curator-run.log",
    "prompt-size": "action-prompt-size.log",
    "dump": "action-dump.log",
    "config-migrate": "action-config-migrate.log",
    "tools-post-setup": "action-tools-post-setup.log",
}

# ``name`` → most recently spawned Popen handle.  Used so ``status`` can
# report liveness and exit code without shelling out to ``ps``.
_ACTION_PROCS: Dict[str, subprocess.Popen] = {}
_ACTION_COMMANDS: Dict[str, Tuple[str, ...]] = {}
_ACTION_IDS: Dict[str, str] = {}

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


# ``name`` → completed synthetic action result for actions the server handled
# without spawning a subprocess (for example, unsupported Docker updates).
_ACTION_RESULTS: Dict[str, Dict[str, Any]] = {}


def _terminate_desktop_managed_gateway() -> None:
    """Stop a live gateway restart child when its Desktop backend shuts down."""
    proc = _ACTION_PROCS.get("gateway-restart")
    if proc is None:
        return
    try:
        if proc.poll() is None:
            proc.terminate()
    except OSError:
        # The child may have exited between poll() and terminate().
        pass


def _dashboard_spawn_executable() -> str:
    """Interpreter for detached dashboard actions.

    Prefers the install's own venv interpreter over ``sys.executable`` when
    they differ. Under an SSH remote backend the web server is launched by
    running the **uv base interpreter** with the venv's site-packages
    injected into ``sys.path`` at startup (``-c "sys.path[:0]=[...];
    runpy.run_module('hermes_cli.main', ...)"``) — so ``sys.executable`` is
    the dependency-less base python and a detached action spawned from it
    dies on the first third-party import (``ModuleNotFoundError: yaml``),
    because the injected path is a startup artifact of the parent and is
    not inherited (#90026). The venv launcher resolves the same dependency
    set on its own.

    Falls back to ``sys.executable`` when no venv interpreter exists next
    to the install (in-process dev runs, exotic layouts). On Windows the
    spawn below carries ``windows_detach_flags()`` (CREATE_NO_WINDOW), so
    the console python owns a single hidden console that its own subprocess
    spawns inherit — the action stays invisible without resorting to
    console-less pythonw.exe, which would make every console-subsystem
    descendant flash its own conhost (#54220/#56747).
    """
    exe = Path(sys.executable)
    try:
        for rel in ("venv/bin/python", "venv/Scripts/python.exe"):
            candidate = PROJECT_ROOT / rel
            if candidate.is_file():
                # Same interpreter → keep sys.executable (preserves the
                # docstring's console-ownership behavior verbatim). Compare
                # UNRESOLVED normalized paths: a venv's bin/python is
                # typically a SYMLINK to the base interpreter, so resolving
                # both sides makes the venv python and the dependency-less
                # base compare equal — exactly the SSH-runtime case this
                # function exists to fix. The unresolved path IS the venv's
                # identity (pyvenv.cfg discovery keys off argv0's location).
                if os.path.normcase(os.path.normpath(str(candidate))) == (
                    os.path.normcase(os.path.normpath(str(exe)))
                ):
                    return sys.executable
                # Return the candidate UNRESOLVED for the same reason:
                # invoking the resolved target would bypass pyvenv.cfg and
                # run the bare base interpreter again.
                return str(candidate)
    except OSError:
        pass
    return sys.executable


def _spawn_hermes_action(
    subcommand: List[str],
    name: str,
    *,
    env_overrides: Optional[Dict[str, str]] = None,
) -> subprocess.Popen:
    """Spawn ``hermes <subcommand>`` detached and record the Popen handle.

    Uses the running interpreter's ``hermes_cli.main`` module so the action
    inherits the same venv/PYTHONPATH the web server is using.
    """
    log_file_name = _ACTION_LOG_FILES[name]
    _ACTION_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = _ACTION_LOG_DIR / log_file_name
    log_file = open(log_path, "ab", buffering=0)
    log_file.write(
        f"\n=== {name} started {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n".encode()
    )

    cmd = [_dashboard_spawn_executable(), "-m", "hermes_cli.main", *subcommand]

    # The dashboard runs *inside* the gateway process, so os.environ carries
    # _HERMES_GATEWAY=1. Inheriting it makes a spawned `hermes gateway restart`
    # trip the in-process restart-loop guard and exit 1 — silently failing the
    # dashboard's auto-restart paths. The gateway's own restart watcher already
    # drops it (gateway/run.py); mirror that here (#52470).
    action_env = {**os.environ, "HERMES_NONINTERACTIVE": "1"}
    action_env.pop("_HERMES_GATEWAY", None)

    popen_kwargs: Dict[str, Any] = {
        "cwd": str(PROJECT_ROOT),
        "stdin": subprocess.DEVNULL,
        "stdout": log_file,
        "stderr": subprocess.STDOUT,
        "env": {**action_env, **(env_overrides or {})},
    }
    if sys.platform == "win32":
        popen_kwargs["creationflags"] = windows_detach_flags()
    else:
        popen_kwargs["start_new_session"] = True

    proc = subprocess.Popen(cmd, **popen_kwargs)
    # The child inherits its own duplicated fd for stdout/stderr, so the
    # parent's handle can be released immediately — otherwise we leak one
    # fd per spawned action.
    log_file.close()
    _ACTION_RESULTS.pop(name, None)
    _ACTION_COMMANDS[name] = tuple(subcommand)
    _ACTION_PROCS[name] = proc
    action_id = (env_overrides or {}).get("HERMES_ACTION_ID")
    if action_id:
        _ACTION_IDS[name] = action_id
    else:
        _ACTION_IDS.pop(name, None)
    return proc


def _gateway_subcommand(profile: Optional[str], verb: str) -> List[str]:
    return _profile_cli_args(profile) + ["gateway", verb]


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


def _restart_gateway_after(profile: Optional[str], *, what: str, label: str) -> dict[str, Any]:
    """Best-effort gateway restart after a config change (webhooks, onboarding).

    The config save stays authoritative; a failed spawn is reported in the
    result (``restart_started: False`` + ``restart_error``) so the UI can fall
    back to its manual restart banner instead of failing the request.
    """
    try:
        proc, reused = _spawn_gateway_restart(profile)
    except Exception as exc:
        _log.exception("Failed to auto-restart gateway after %s", what)
        return {"restart_started": False, "restart_error": str(exc)}
    if reused:
        _log.info("%s: reusing in-flight gateway restart (pid %s)", label, proc.pid)
    return {"restart_started": True, "restart_action": "gateway-restart", "restart_pid": proc.pid}


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


def _split_text_for_speak_stream(text: str, cap: int) -> list:
    """Split *text* into provider-cap-sized pieces on sentence boundaries.

    Deliberately NOT unified with gateway.platforms.helpers'
    split_text_fence_aware: this splitter reflows whitespace (sentences are
    re-joined with single spaces) and has no fence/markdown semantics, so
    expressing it as knobs on the fence-aware core would change behavior.
    """
    from tools.tts_streaming import SENTENCE_BOUNDARY_RE as _SENTENCE_BOUNDARY_RE

    cap = cap if cap and cap > 0 else 4000
    pieces, buf = [], ""
    for sentence in filter(str.strip, _SENTENCE_BOUNDARY_RE.split(text)):
        while len(sentence) > cap:
            pieces.append(sentence[:cap])
            sentence = sentence[cap:]
        if buf and len(buf) + len(sentence) + 1 > cap:
            pieces.append(buf)
            buf = sentence
        else:
            buf = f"{buf} {sentence}" if buf else sentence
    if buf:
        pieces.append(buf)
    return pieces


app.include_router(_actions_routes.status_router)


# Per-row fields that no session LIST consumer reads but that dominate the
# payload. ``system_prompt`` is the fully rendered prompt — tens of KB per
# row — and made a 21-row /api/sessions response 528KB (96% dead weight),
# re-fetched by the desktop sidebar on every refresh. The desktop's
# SessionInfo type doesn't declare either field and the web UI never touches
# them; ``GET /api/sessions/{id}`` detail reads stay complete. List callers
# that genuinely need the full rows can pass ``?full=1``.
_SESSION_LIST_HEAVY_FIELDS = ("system_prompt", "model_config")


def _strip_session_list_rows(sessions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    for s in sessions:
        for key in _SESSION_LIST_HEAVY_FIELDS:
            s.pop(key, None)
    return sessions


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


def _normalize_config_for_web(config: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize config for the web UI.

    Hermes supports ``model`` as either a bare string (``"anthropic/claude-sonnet-4"``)
    or a dict (``{default: ..., provider: ..., base_url: ...}``).  The schema is built
    from DEFAULT_CONFIG where ``model`` is a string, but user configs often have the
    dict form.  Normalize to the string form so the frontend schema matches.

    Also surfaces ``model_context_length`` as a top-level field so the web UI can
    display and edit it.  A value of 0 means "auto-detect".
    """
    config = dict(config)  # shallow copy
    model_val = config.get("model")
    if isinstance(model_val, dict):
        # Extract context_length before flattening the dict
        ctx_len = model_val.get("context_length", 0)
        config["model"] = model_val.get("default", model_val.get("name", ""))
        config["model_context_length"] = ctx_len if isinstance(ctx_len, int) else 0
    else:
        config["model_context_length"] = 0
    return config


def _normalize_memory_provider_name(name: Any) -> str:
    provider = str(name or "").strip()
    if provider.lower() in {"built-in", "builtin", "none"}:
        return ""
    return provider


def _load_memory_provider(name: str):
    try:
        from plugins.memory import load_memory_provider

        return load_memory_provider(name)
    except Exception:
        _log.debug("Failed to load memory provider %s", name, exc_info=True)
        return None


def _memory_provider_manifest(name: str) -> Dict[str, Any]:
    try:
        from plugins.memory import find_provider_dir

        provider_dir = find_provider_dir(name)
        if provider_dir is None:
            return {}
        manifest_path = provider_dir / "plugin.yaml"
        if not manifest_path.exists():
            return {}
        with manifest_path.open(encoding="utf-8-sig") as handle:
            manifest = yaml.safe_load(handle) or {}
        return manifest if isinstance(manifest, dict) else {}
    except Exception:
        _log.debug("Failed to read memory provider manifest for %s", name, exc_info=True)
        return {}


def _string_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _memory_provider_setup_manifest(name: str) -> Dict[str, Any]:
    manifest = _memory_provider_manifest(name)
    external_dependencies: List[Dict[str, str]] = []
    for raw in manifest.get("external_dependencies") or []:
        if not isinstance(raw, dict):
            continue
        dep = {
            "name": str(raw.get("name") or "").strip(),
            "install": str(raw.get("install") or "").strip(),
            "check": str(raw.get("check") or "").strip(),
        }
        if dep["name"] or dep["install"] or dep["check"]:
            external_dependencies.append(dep)

    return {
        "pip_dependencies": _string_list(manifest.get("pip_dependencies")),
        "external_dependencies": external_dependencies,
        "required_env": _string_list(manifest.get("requires_env")),
    }


def _memory_provider_setup_info(name: str) -> Dict[str, Any]:
    setup = _memory_provider_setup_manifest(name)
    setup["dependencies_installed"] = _memory_provider_dependencies_installed(setup)
    return setup


_MEMORY_PROVIDER_IMPORT_NAMES = {
    "honcho-ai": "honcho",
    "mem0ai": "mem0",
    "hindsight-client": "hindsight_client",
    "hindsight-all": "hindsight",
}


def _memory_provider_dependency_package(dep: str) -> str:
    return re.split(r"[\[<>=!~;]", dep, maxsplit=1)[0].strip()


def _memory_provider_import_name(dep: str) -> str:
    package = _memory_provider_dependency_package(dep)
    return _MEMORY_PROVIDER_IMPORT_NAMES.get(package, package.replace("-", "_"))


def _dependency_importable(dep: str) -> bool:
    import_name = _memory_provider_import_name(dep)
    if not import_name:
        return False
    try:
        __import__(import_name)
        return True
    except ImportError:
        return False


def _memory_provider_setup_env() -> Dict[str, str]:
    # External package-manager child (npm/uv/pip): exact env preservation —
    # scrubbing or HOME rewriting could break user tool auth/config.
    from tools.environments.local import build_subprocess_env
    env = build_subprocess_env(scrub_secrets=False, inherit_profile_home=False)
    home = Path.home()
    extra_bins = [
        home / ".brv-cli" / "bin",
        home / ".local" / "bin",
        home / ".npm-global" / "bin",
        Path("/usr/local/bin"),
    ]
    existing_path = env.get("PATH", "")
    prefix = os.pathsep.join(str(path) for path in extra_bins if path.exists())
    if prefix:
        env["PATH"] = prefix + os.pathsep + existing_path
    return env


def _run_setup_command(
    command: Any,
    *,
    display: str,
    shell: bool = False,
    timeout: int = 180,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        command,
        shell=shell,
        executable="/bin/bash" if shell else None,
        env=_memory_provider_setup_env(),
        capture_output=True,
        text=True,
        # Lossy UTF-8 decode — setup tools emit UTF-8; never let a
        # locale-mismatched byte raise in the reader thread (#52649).
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )


def _memory_provider_dependencies_installed(setup: Dict[str, Any]) -> bool:
    pip_dependencies = _string_list(setup.get("pip_dependencies"))
    external_dependencies = setup.get("external_dependencies") or []

    pip_ok = all(_dependency_importable(dep) for dep in pip_dependencies)
    external_ok = True
    for dep in external_dependencies:
        if not isinstance(dep, dict):
            continue
        check_cmd = str(dep.get("check") or "").strip()
        install_cmd = str(dep.get("install") or "").strip()
        if not check_cmd:
            if install_cmd:
                external_ok = False
            continue
        try:
            completed = _run_setup_command(
                shlex.split(check_cmd),
                display=check_cmd,
                timeout=20,
            )
        except Exception:
            external_ok = False
            continue
        if completed.returncode != 0:
            external_ok = False

    return pip_ok and external_ok


from hermes_cli.web_routers import memory_providers as _memory_providers_routes  # noqa: E402

app.include_router(_memory_providers_routes.router)


def _normalize_memory_provider_schema(name: str, provider: Any) -> List[Dict[str, Any]]:
    raw_schema: List[Dict[str, Any]] = []
    if provider is not None and hasattr(provider, "get_config_schema"):
        try:
            raw = provider.get_config_schema()
            if isinstance(raw, list):
                raw_schema = [field for field in raw if isinstance(field, dict)]
        except Exception:
            _log.warning("Failed to read memory provider schema for %s", name, exc_info=True)

    fields: List[Dict[str, Any]] = []
    for raw in raw_schema:
        key = str(raw.get("key") or "").strip()
        if not key:
            continue

        choices = raw.get("choices") or raw.get("options") or []
        if not isinstance(choices, list):
            choices = []

        explicit_kind = str(raw.get("kind") or raw.get("type") or "").strip().lower()
        if raw.get("secret"):
            kind = "secret"
        elif choices:
            kind = "select"
        elif explicit_kind in {"bool", "boolean"} or isinstance(raw.get("default"), bool):
            kind = "boolean"
        elif explicit_kind in {"int", "integer"} or (
            isinstance(raw.get("default"), int) and not isinstance(raw.get("default"), bool)
        ):
            kind = "integer"
        elif explicit_kind in {"float", "number"} or isinstance(raw.get("default"), float):
            kind = "number"
        else:
            kind = "text"

        options = []
        for choice in choices:
            value = str(choice)
            options.append({"value": value, "label": value, "description": ""})

        description = str(raw.get("description") or "")
        fields.append({
            "key": key,
            "label": str(raw.get("label") or key.replace("_", " ").title()),
            "kind": kind,
            "description": description,
            "placeholder": str(raw.get("placeholder") or ""),
            "required": bool(raw.get("required", False)),
            "default": raw.get("default", ""),
            "options": options,
            "url": str(raw.get("url") or ""),
            "when": raw.get("when") if isinstance(raw.get("when"), dict) else None,
            "minimum": raw.get("minimum"),
            "maximum": raw.get("maximum"),
            "step": raw.get("step"),
            "_env_key": str(raw.get("env_var") or "") or None,
        })

    return fields


def _read_json_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        _log.debug("Failed to read JSON config from %s", path, exc_info=True)
        return {}
    return data if isinstance(data, dict) else {}


def _read_memory_provider_existing_values(name: str) -> Dict[str, Any]:
    """Best-effort read of existing provider config across legacy/native stores."""

    hermes_home = get_hermes_home()
    values: Dict[str, Any] = {}

    # Common native provider stores.
    for path in (
        hermes_home / f"{name}.json",
        hermes_home / name / "config.json",
    ):
        values.update(_read_json_file(path))

    try:
        cfg = load_config()
    except Exception:
        cfg = {}

    memory_cfg = cfg.get("memory") if isinstance(cfg, dict) else {}
    if isinstance(memory_cfg, dict):
        provider_cfg = memory_cfg.get(name)
        if isinstance(provider_cfg, dict):
            values.update(provider_cfg)
        legacy_cfg = memory_cfg.get("provider_config")
        if isinstance(legacy_cfg, dict):
            values = {**legacy_cfg, **values}

    # Holographic stores under plugins.hermes-memory-store.
    plugins_cfg = cfg.get("plugins") if isinstance(cfg, dict) else {}
    if name == "holographic" and isinstance(plugins_cfg, dict):
        holographic_cfg = plugins_cfg.get("hermes-memory-store")
        if isinstance(holographic_cfg, dict):
            values.update(holographic_cfg)

    return values


def _env_lookup(env_key: Optional[str]) -> str:
    if not env_key:
        return ""
    env_on_disk = load_env()
    return str(env_on_disk.get(env_key) or os.environ.get(env_key) or "")


def _coerce_bool(value: Any, *, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or value == "":
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value}")


def _field_default(field: Dict[str, Any]) -> Any:
    default = field.get("default", "")
    if field["kind"] == "boolean":
        return _coerce_bool(default, default=False)
    return default


def _field_value(field: Dict[str, Any], data: Dict[str, Any]) -> Any:
    if field["kind"] == "secret":
        return ""

    value = data.get(field["key"])
    if value in (None, ""):
        value = _env_lookup(field.get("_env_key"))
    if value in (None, ""):
        value = _field_default(field)

    if field["kind"] == "select":
        allowed = {opt["value"] for opt in field.get("options", [])}
        value = str(value)
        return value if value in allowed else str(_field_default(field))
    if field["kind"] == "boolean":
        return _coerce_bool(value, default=_coerce_bool(_field_default(field), default=False))
    return str(value)


def _field_is_set(field: Dict[str, Any], data: Dict[str, Any]) -> bool:
    if field["kind"] == "secret":
        return bool(_env_lookup(field.get("_env_key")) or data.get(field["key"]))
    value = _field_value(field, data)
    return value not in (None, "")


def _field_visible(
    field: Dict[str, Any],
    data: Dict[str, Any],
    fields_by_key: Optional[Dict[str, Dict[str, Any]]] = None,
) -> bool:
    when = field.get("when")
    if not isinstance(when, dict) or not when:
        return True
    for dep_key, expected in when.items():
        dep_field = (fields_by_key or {}).get(str(dep_key)) or {
            "key": str(dep_key),
            "kind": "text",
            "default": "",
            "_env_key": None,
        }
        actual = _field_value(dep_field, data)
        if str(actual) != str(expected):
            return False
    return True


def _memory_provider_is_configured(name: str, provider: Any) -> bool:
    data = _read_memory_provider_existing_values(name)
    fields = _normalize_memory_provider_schema(name, provider)
    fields_by_key = {field["key"]: field for field in fields}
    visible_fields = [
        field for field in fields if _field_visible(field, data, fields_by_key)
    ]
    required_fields = [field for field in visible_fields if field.get("required")]
    if not required_fields:
        return True
    return all(_field_is_set(field, data) for field in required_fields)


def _discover_memory_provider_statuses() -> List[Dict[str, Any]]:
    discovered: Dict[str, Dict[str, Any]] = {}
    try:
        from plugins.memory import discover_memory_providers

        for name, description, available in discover_memory_providers():
            discovered[str(name)] = {
                "name": str(name),
                "description": str(description or ""),
                "available": bool(available),
                "missing": False,
            }
    except Exception:
        _log.exception("discover_memory_providers failed")

    cfg = load_config()
    active = ""
    mem = cfg.get("memory")
    if isinstance(mem, dict):
        active = _normalize_memory_provider_name(mem.get("provider"))
    if active and active not in discovered:
        discovered[active] = {
            "name": active,
            "description": "Configured provider was not found.",
            "available": False,
            "missing": True,
        }

    providers: List[Dict[str, Any]] = []
    for name in sorted(discovered):
        row = discovered[name]
        provider = None if row["missing"] else _load_memory_provider(name)
        setup = _memory_provider_setup_info(name)
        configured = False if row["missing"] else _memory_provider_is_configured(name, provider)
        schema_fields = [] if row["missing"] else _normalize_memory_provider_schema(name, provider)
        if row["missing"]:
            status = "missing"
        elif not row["available"] and not setup.get("dependencies_installed", True):
            status = "unavailable"
        elif not configured:
            status = "needs_config"
        elif not row["available"] and schema_fields:
            status = "needs_config"
        elif not row["available"]:
            status = "unavailable"
        else:
            status = "ready"
        providers.append({
            "name": name,
            "description": row["description"],
            "available": row["available"],
            "configured": configured,
            "status": status,
            "setup": setup,
        })
    return providers


def _require_memory_provider_ready(name: str) -> None:
    if not name:
        return
    statuses = {row["name"]: row for row in _discover_memory_provider_statuses()}
    row = statuses.get(name)
    if row is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown memory provider '{name}'.",
        )
    if row["status"] != "ready":
        raise HTTPException(
            status_code=400,
            detail=(
                f"Memory provider '{name}' is not ready "
                f"({row['status'].replace('_', ' ')}). Configure it in the dashboard first."
            ),
        )


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


# ---------------------------------------------------------------------------
# Model assignment — pick provider+model for main slot or auxiliary slots.
# Mirrors the model.options JSON-RPC from tui_gateway but uses REST so the
# Models page (which has no chat PTY open) can drive it.
# ---------------------------------------------------------------------------

# Canonical auxiliary task slots. Keep in sync with DEFAULT_CONFIG["auxiliary"]
# in hermes_cli/config.py — listed here for deterministic ordering in the UI.
_AUX_TASK_SLOTS: Tuple[str, ...] = (
    "vision",
    "compression",
    "skills_hub",
    "approval",
    "mcp",
    "title_generation",
    "review",
    "triage_specifier",
    "kanban_decomposer",
    "profile_describer",
    "curator",
)


def _dashboard_code_skew_guard() -> Optional[str]:
    """Return a clear \"restart required\" message when this process runs stale code.

    The dashboard and Desktop-owned ``hermes serve`` are long-lived; their
    ``sys.modules`` is frozen at boot.  When ``hermes update`` (or a manual
    ``git pull``) replaces the checkout underneath them, a first-time lazy
    import on a new code path can resolve a freshly-pulled consumer module
    against a stale cached dependency -> ImportError — e.g. ``/api/model/options``
    500 after the update added ``agent.model_metadata.is_grok_46_family`` while
    the running process kept serving the pre-update module (#86207).  Mirror
    the gateway's ``_model_switch_skew_guard``: refuse the risky call with an
    actionable, deployment-aware message instead of crashing with a cryptic
    import error (#97046).

    Returns None when no drift is detectable (fresh process, or a non-git
    install where the boot fingerprint could not be read — never a false
    positive).
    """
    from gateway.code_skew import detect_code_skew

    skew = detect_code_skew()
    if not skew:
        return None
    boot_rev, disk_rev = skew
    return (
        f"This process is running code from {boot_rev} but the checkout on "
        f"disk is now {disk_rev}. The model picker would risk a stale-module "
        f"crash — {_dashboard_skew_restart_hint()}"
    )


def _dashboard_skew_restart_hint() -> str:
    """Restart advice that matches how this process is actually owned.

    The same FastAPI app backs the browser dashboard *and* Desktop-owned
    ``hermes serve --isolated`` (local or SSH). Hardcoding a systemd unit
    misleads macOS/launchd hosts and Desktop SSH backends, which have no
    ``hermes-dashboard`` unit (#97046).
    """
    if os.environ.get("HERMES_SERVE_HEADLESS") == "1":
        return (
            "restart the Desktop-owned backend to load the new code "
            "(use Restart backend in Hermes Desktop, or quit and reopen the app)"
        )
    return (
        "restart this Hermes process to load the new code "
        "(hermes dashboard --port <port>, or the equivalent service restart for this install)"
    )


def _apply_model_assignment_sync(
    scope: str, provider: str, model: str, task: str, base_url: str, api_key: str = ""
):
    """Synchronous body of POST /api/model/set.

    Runs inside ``_profile_scope`` (in a worker thread) so every
    load_config/save_config lands in the requested profile.  Raises
    HTTPException for validation errors — the async wrapper re-raises them.
    """
    cfg = load_config()

    if scope == "main":
        if not provider or not model:
            raise HTTPException(status_code=400, detail="provider and model required for main")
        provider, model = _normalize_main_model_assignment(provider, model)
        providers_cfg = cfg.get("providers")
        provider_entry = providers_cfg.get(provider) if isinstance(providers_cfg, dict) else None
        if not base_url and isinstance(provider_entry, dict) and provider_entry.get("base_url"):
            base_url = str(provider_entry.get("base_url") or "").strip()
        model_cfg = _apply_main_model_assignment(
            cfg.get("model", {}), provider, model, base_url, api_key
        )
        _raw_assign_entry = None
        try:
            _stored, _raw_assign_entry = find_provider_entry(
                read_raw_config().get("providers"), provider
            )
        except Exception:
            _raw_assign_entry = None
        _assign_key_env = (
            str(_raw_assign_entry.get("key_env") or "").strip()
            if isinstance(_raw_assign_entry, dict)
            else ""
        )
        if _assign_key_env:
            # #88990: carry the credential POINTER, never a resolved secret.
            model_cfg["key_env"] = _assign_key_env
            model_cfg.pop("api_key", None)
        elif isinstance(provider_entry, dict) and provider_entry.get("api_key"):
            # #88990: provider_entry comes from load_config(), which expands
            # ${VAR} env refs to plaintext. Copying that resolved value into
            # model.api_key writes the SECRET into config.yaml (and recreates
            # it on every re-apply, even after the user deletes it by hand).
            # Prefer the raw ${VAR} template; only fall back to the expanded
            # value when the raw yaml itself stores the key as a literal (no
            # new exposure in that case).
            _raw_key = (
                str(_raw_assign_entry.get("api_key") or "").strip()
                if isinstance(_raw_assign_entry, dict)
                else ""
            )
            if _raw_key.startswith("${") and _raw_key.endswith("}"):
                model_cfg["api_key"] = _raw_key
            else:
                model_cfg["api_key"] = provider_entry["api_key"]
        cfg["model"] = model_cfg

        # When switching the main provider to Nous, mirror the CLI's
        # post-model-selection behaviour (hermes_cli/main.py
        # prompt_enable_tool_gateway / tools_config apply_nous_managed_defaults):
        # auto-route any *unconfigured* tools through the Nous Tool Gateway.
        # This is purely additive — apply_nous_managed_defaults skips every
        # tool where the user already has a direct key (FIRECRAWL_API_KEY,
        # FAL_KEY, etc.) or an explicit backend/provider in config, so it
        # never overwrites a user's own setup. GUI users thus land on the
        # gateway the same way CLI users do, without a separate prompt.
        gateway_tools: list[str] = []
        if provider.strip().lower() == "nous":
            try:
                from hermes_cli.nous_subscription import apply_nous_managed_defaults
                from hermes_cli.tools_config import _get_platform_tools

                enabled = _get_platform_tools(
                    cfg, "cli", include_default_mcp_servers=False
                )
                changed = apply_nous_managed_defaults(
                    cfg,
                    enabled_toolsets=enabled,
                    force_fresh=True,
                )
                gateway_tools = sorted(changed)
            except Exception:
                # Portal lookup hiccups / non-subscriber / non-nous gating
                # must never block saving the model assignment.
                _log.debug("apply_nous_managed_defaults skipped", exc_info=True)

        save_config(cfg)

        # Register a named ``custom_providers`` entry for a custom/local
        # endpoint, mirroring the ``hermes model`` custom flow
        # (_save_custom_provider). Without this the endpoint only lives in
        # ``model.*`` and the picker has no proper ready row for it — the
        # GUI then surfaces a "needs setup" dead-end on the bare ``custom``
        # provider. Dedups by base_url, so re-saving is idempotent.
        if provider.strip().lower() in {"custom", "local"} and base_url:
            try:
                from hermes_cli.main import _auto_provider_name, _save_custom_provider

                _save_custom_provider(
                    base_url,
                    api_key,
                    model,
                    name=_auto_provider_name(base_url),
                )
            except Exception:
                # Never block the assignment on the bookkeeping write —
                # model.* is already persisted and routable.
                _log.debug("custom_providers registration skipped", exc_info=True)

        # Surface auxiliary slots still pinned to a *different* provider than
        # the new main one. Switching the main model does NOT touch aux pins
        # (they're independent, sticky per-task overrides — see
        # auxiliary_client._resolve_auto). A user who switches main away from
        # a now-unpaid provider (e.g. nous with $0 balance) keeps paying 402s
        # on every background aux call until they reset those pins. We never
        # auto-clear them — pinning aux to a cheaper/different model is a
        # legitimate config — but we tell the caller so the UI can offer a
        # "reset to main" nudge instead of silently burning credits.
        new_provider = provider.strip().lower()
        stale_aux: list[dict] = []
        aux_cfg = cfg.get("auxiliary", {})
        if isinstance(aux_cfg, dict):
            for slot in _AUX_TASK_SLOTS:
                slot_cfg = aux_cfg.get(slot)
                if not isinstance(slot_cfg, dict):
                    continue
                slot_provider = str(slot_cfg.get("provider", "") or "").strip()
                if (
                    slot_provider
                    and slot_provider.lower() not in {"auto", ""}
                    and slot_provider.lower() != new_provider
                ):
                    stale_aux.append({
                        "task": slot,
                        "provider": slot_provider,
                        "model": str(slot_cfg.get("model", "") or ""),
                    })

        try:
            effective_config = load_config()
            effective_provider, effective_model = resolve_cron_model_drift_defaults(
                effective_config
            )
            cron_model_impact = build_cron_model_impact(
                current_provider=effective_provider or provider,
                current_model=effective_model or model,
                config=effective_config,
            )
        except Exception:
            _log.debug("cron model impact inspection failed", exc_info=True)
            cron_model_impact = build_cron_model_impact(config=cfg, jobs={})

        return {
            "ok": True,
            "scope": "main",
            "provider": provider,
            "model": model,
            "base_url": model_cfg.get("base_url", ""),
            "gateway_tools": gateway_tools,
            "stale_aux": stale_aux,
            "cron_model_impact": cron_model_impact,
        }

    # scope == "auxiliary"
    aux = cfg.get("auxiliary")
    if not isinstance(aux, dict):
        aux = {}

    if task == "__reset__":
        # Reset every slot to provider="auto", model="" — keeps other fields intact.
        for slot in _AUX_TASK_SLOTS:
            slot_cfg = aux.get(slot)
            if not isinstance(slot_cfg, dict):
                slot_cfg = {}
            slot_cfg["provider"] = "auto"
            slot_cfg["model"] = ""
            slot_cfg.pop("base_url", None)
            clear_model_endpoint_credentials(slot_cfg)
            aux[slot] = slot_cfg
        cfg["auxiliary"] = aux
        save_config(cfg)
        return {"ok": True, "scope": "auxiliary", "reset": True}

    if not provider:
        raise HTTPException(status_code=400, detail="provider required for auxiliary")

    targets = [task] if task else list(_AUX_TASK_SLOTS)
    for slot in targets:
        if slot not in _AUX_TASK_SLOTS:
            raise HTTPException(status_code=400, detail=f"unknown auxiliary task: {slot}")
        slot_cfg = aux.get(slot)
        if not isinstance(slot_cfg, dict):
            slot_cfg = {}
        prev_provider = str(slot_cfg.get("provider") or "").strip().lower()
        new_provider = provider.strip().lower()
        slot_cfg["provider"] = provider
        slot_cfg["model"] = model
        if base_url:
            # Sibling of the main-slot endpoint handling (#65254): an aux
            # assignment for a custom/local endpoint must carry its own
            # base_url, or the slot silently rebinds to whatever
            # model.base_url happens to hold — and breaks entirely once the
            # main slot switches away and clears it. The auxiliary resolver
            # already reads auxiliary.<task>.base_url/api_key
            # (_resolve_task_provider_model), so persisting them here is
            # what actually wires the endpoint in.
            slot_cfg["base_url"] = base_url
            if api_key:
                slot_cfg["api_key"] = api_key
        elif new_provider != prev_provider and new_provider != "custom":
            slot_cfg.pop("base_url", None)
            clear_model_endpoint_credentials(slot_cfg)
        aux[slot] = slot_cfg

    cfg["auxiliary"] = aux
    save_config(cfg)
    return {
        "ok": True,
        "scope": "auxiliary",
        "tasks": targets,
        "provider": provider,
        "model": model,
    }


def _infer_provider_on_model_change(model_val: str, prev_provider: str) -> tuple[str, str]:
    """Infer which provider serves ``model_val`` when the flat Config-page Model
    field changes, given the previously-saved ``prev_provider``.

    Returns ``(provider, model)``; ``provider`` is empty when no switch is
    warranted (leave the existing provider untouched). Two signals, in order:

    1. Curated-catalog detection (``detect_provider_for_model``) — handles the
       ~28 OpenRouter-curated models and direct provider-static catalogs.
    2. Vendor-slug heuristic — a ``vendor/model`` slug cannot belong to a
       single-model / non-aggregator provider (e.g. ``ollama-local``). When the
       current provider is not an aggregator that serves vendor-prefixed slugs,
       route to an aggregator. ``_normalize_main_model_assignment`` (called by
       the caller) keeps the user's current aggregator when they're already on
       one, else falls back to openrouter — the same chokepoint logic as
       ``POST /api/model/set``.
    """
    name = (model_val or "").strip()
    if not name:
        return "", name
    try:
        from hermes_cli.models import (
            _AGGREGATOR_PROVIDERS,
            detect_provider_for_model,
            normalize_provider,
        )
    except Exception:
        return "", name

    try:
        detected = detect_provider_for_model(name, prev_provider)
    except Exception:
        detected = None
    if detected:
        return detected[0], detected[1]

    # Vendor-prefixed slug under a non-aggregator provider → reassign. Use a
    # sentinel "openrouter" here; _normalize_main_model_assignment resolves the
    # real aggregator (keeps a current aggregator, else openrouter).
    if "/" in name:
        try:
            cur_is_aggregator = normalize_provider(prev_provider) in _AGGREGATOR_PROVIDERS
        except Exception:
            cur_is_aggregator = False
        if not cur_is_aggregator:
            return "openrouter", name

    return "", name


def _denormalize_config_from_web(config: Dict[str, Any]) -> Dict[str, Any]:
    """Reverse _normalize_config_for_web before saving.

    Reconstructs ``model`` as a dict by reading the current on-disk config
    to recover model subkeys (provider, base_url, api_mode, etc.) that were
    stripped from the GET response.  The frontend only sees model as a flat
    string; the rest is preserved transparently.

    Also handles ``model_context_length`` — writes it back into the model dict
    as ``context_length``.  A value of 0 means "auto-detect" (omitted from the
    dict so get_model_context_length() uses its normal resolution). ``config``
    may be a partial update (e.g. the Settings autosave diff) that omits
    ``model_context_length`` entirely when the user didn't touch it — that
    must leave the on-disk override untouched, not get treated the same as an
    explicit 0 and cleared.
    """
    config = dict(config)
    # Remove any _model_meta that might have leaked in (shouldn't happen
    # with the stripped GET response, but be defensive)
    config.pop("_model_meta", None)

    # Extract and remove model_context_length before processing model, but
    # remember whether it was actually present: a partial update omitting the
    # key means "unchanged", which is different from an explicit 0.
    ctx_sent = "model_context_length" in config
    ctx_override = config.pop("model_context_length", 0)
    if not isinstance(ctx_override, int):
        try:
            ctx_override = int(ctx_override)
        except (TypeError, ValueError):
            ctx_override = 0

    model_val = config.get("model")
    if (isinstance(model_val, str) and model_val) or ctx_sent:
        # Read the current disk config to recover model subkeys
        try:
            disk_config = load_config()
            disk_model = disk_config.get("model")
            if isinstance(disk_model, dict):
                if isinstance(model_val, str) and model_val:
                    prev_default = str(disk_model.get("default") or "").strip()
                    prev_provider = str(disk_model.get("provider") or "").strip()
                    # When the model name actually changed, re-detect which
                    # provider serves it. The Config-page Model field is a flat
                    # string with no provider info, so without this a user who
                    # picks an OpenRouter model while their default provider is
                    # ollama-local keeps the stale provider and 404s. Only fires
                    # on a real model change so saving unrelated config fields
                    # never overwrites an explicit provider.
                    if model_val != prev_default and prev_provider:
                        new_provider, resolved_model = _infer_provider_on_model_change(
                            model_val, prev_provider
                        )
                        if new_provider and new_provider.strip().lower() != prev_provider.lower():
                            # Route through the canonical assignment chokepoints so
                            # the model is normalized for the new provider and stale
                            # base_url/api_mode/api_key are cleared on the switch
                            # (and preserved on a same-provider re-pick).
                            norm_provider, norm_model = _normalize_main_model_assignment(
                                new_provider, resolved_model
                            )
                            disk_model = _apply_main_model_assignment(
                                disk_model, norm_provider, norm_model
                            )
                            model_val = norm_model
                    # Preserve all subkeys, update default with the new value
                    disk_model["default"] = model_val
                # Write context_length into the model dict (0 = remove/auto),
                # but only when the payload actually carried the key.
                if ctx_sent:
                    if ctx_override > 0:
                        disk_model["context_length"] = ctx_override
                    else:
                        disk_model.pop("context_length", None)
                config["model"] = disk_model
            # Model was previously a bare string (or absent) — upgrade to a
            # dict if the user is setting a context_length override.
            elif ctx_sent and ctx_override > 0:
                if isinstance(model_val, str) and model_val:
                    default = model_val
                elif isinstance(disk_model, str) and disk_model:
                    default = disk_model
                else:
                    default = ""
                config["model"] = {
                    "default": default,
                    "context_length": ctx_override,
                }
        except Exception:
            pass  # can't read disk config — just use the string form
    return config


app.include_router(_config_env_routes.router)


def _is_other_profile(profile: Optional[str]) -> bool:
    """True when ``profile`` names a profile other than this process's own."""
    requested = (profile or "").strip()
    if not requested or requested.lower() == "current":
        return False
    try:
        target = _resolve_profile_dir(requested)
    except HTTPException:
        return True
    return target.resolve() != get_process_hermes_home().resolve()


def _approval_mode_of(config: Dict[str, Any]) -> str:
    """Normalize approvals.mode from an in-memory config document.

    Both sides of the broadcast comparison use in-memory documents (the raw
    on-disk dict and the about-to-be-saved dict): re-reading through the
    config cache after a save can serve the pre-save document when the
    replacement file collides on the (mtime_ns, size) cache key, which would
    suppress the broadcast exactly when the mode changed. Absent block or
    key normalizes to the same default the approval gate uses.
    """
    from tools.approval import _normalize_approval_mode

    approvals = config.get("approvals")
    default_mode = (DEFAULT_CONFIG.get("approvals") or {}).get("mode", "manual")
    mode = approvals.get("mode", default_mode) if isinstance(approvals, dict) else default_mode
    return _normalize_approval_mode(mode)


def _broadcast_gateway_session_info() -> None:
    """Broadcast session.info on the in-process gateway when it's loaded.

    ``sys.modules`` guard, not an import: gateway never imported means no
    live sessions in this process to notify.
    """
    server = sys.modules.get("tui_gateway.server")
    if server is None:
        return
    try:
        server.broadcast_session_info()
    except Exception:
        _log.exception("session.info broadcast after config save failed")


def _parse_model_ids(resp: "Any") -> List[str]:
    """Extract model ids from an OpenAI-compatible ``/v1/models`` response.

    Tolerant of the common shapes: ``{"data": [{"id": ...}]}`` (OpenAI / vLLM /
    llama.cpp) and a bare ``{"data": ["id", ...]}``. Returns ``[]`` on any
    parse/HTTP error so a slightly non-standard endpoint never hard-blocks.
    """
    try:
        if not resp.is_success:
            return []
        payload = resp.json()
    except Exception:
        return []
    data = payload.get("data") if isinstance(payload, dict) else payload
    if not isinstance(data, list):
        return []
    ids: List[str] = []
    for item in data:
        if isinstance(item, dict):
            mid = str(item.get("id") or "").strip()
        else:
            mid = str(item or "").strip()
        if mid:
            ids.append(mid)
    return ids


# Entries omit fields they don't need to override; the catalog builder fills
# in env_vars from OPTIONAL_ENV_VARS via prefix matching when not specified,
# and pulls required_env from a plugin's PlatformEntry when available.
_PLATFORM_OVERRIDES: dict[str, dict[str, Any]] = {
    "telegram": {
        "name": "Telegram",
        "description": "Run Hermes from Telegram DMs, groups, and topics.",
        "docs_url": "https://core.telegram.org/bots/features#botfather",
        "env_vars": ("TELEGRAM_BOT_TOKEN", "TELEGRAM_ALLOWED_USERS", "TELEGRAM_PROXY"),
        "required_env": ("TELEGRAM_BOT_TOKEN",),
    },
    "discord": {
        "name": "Discord",
        "description": "Connect Hermes to Discord DMs, channels, and threads.",
        "docs_url": "https://discord.com/developers/applications",
        "env_vars": (
            "DISCORD_BOT_TOKEN",
            "DISCORD_ALLOWED_USERS",
        ),
        "required_env": ("DISCORD_BOT_TOKEN",),
    },
    "slack": {
        "name": "Slack",
        "description": "Use Hermes from Slack via Socket Mode. Add allowed Slack member IDs so connected bots can respond.",
        "docs_url": "https://api.slack.com/apps",
        "env_vars": ("SLACK_BOT_TOKEN", "SLACK_APP_TOKEN", "SLACK_ALLOWED_USERS"),
        "required_env": ("SLACK_BOT_TOKEN", "SLACK_APP_TOKEN"),
    },
    "mattermost": {
        "name": "Mattermost",
        "description": "Connect Hermes to Mattermost channels and direct messages.",
        "docs_url": "https://mattermost.com/deploy/",
        "env_vars": ("MATTERMOST_URL", "MATTERMOST_TOKEN", "MATTERMOST_ALLOWED_USERS"),
        "required_env": ("MATTERMOST_URL", "MATTERMOST_TOKEN"),
    },
    "matrix": {
        "name": "Matrix",
        "description": "Use Hermes in Matrix rooms and direct messages.",
        "docs_url": "https://matrix.org/ecosystem/servers/",
        "env_vars": (
            "MATRIX_HOMESERVER",
            "MATRIX_ACCESS_TOKEN",
            "MATRIX_USER_ID",
            "MATRIX_ALLOWED_USERS",
        ),
        "required_env": ("MATRIX_HOMESERVER", "MATRIX_ACCESS_TOKEN", "MATRIX_USER_ID"),
    },
    "signal": {
        "name": "Signal",
        "description": "Connect through a signal-cli REST bridge.",
        "docs_url": "https://github.com/bbernhard/signal-cli-rest-api",
        "env_vars": ("SIGNAL_HTTP_URL", "SIGNAL_ACCOUNT", "SIGNAL_ALLOWED_USERS"),
        "required_env": ("SIGNAL_HTTP_URL", "SIGNAL_ACCOUNT"),
    },
    "whatsapp": {
        "name": "WhatsApp",
        "description": "Use Hermes through the bundled WhatsApp bridge with QR-based auth.",
        "docs_url": "https://github.com/tulir/whatsmeow",
        "env_vars": (
            "WHATSAPP_ENABLED",
            "WHATSAPP_MODE",
            "WHATSAPP_DM_POLICY",
            "WHATSAPP_ALLOWED_USERS",
        ),
        "required_env": (),
    },
    "homeassistant": {
        "name": "Home Assistant",
        "description": "Control your smart home from Hermes via Home Assistant.",
        "docs_url": "https://www.home-assistant.io/docs/authentication/",
        "env_vars": ("HASS_URL", "HASS_TOKEN"),
        "required_env": ("HASS_URL", "HASS_TOKEN"),
    },
    "email": {
        "name": "Email",
        "description": "Talk to Hermes through an IMAP/SMTP mailbox.",
        "docs_url": "https://hermes-agent.nousresearch.com/docs/user-guide/messaging/",
        "env_vars": (
            "EMAIL_ADDRESS",
            "EMAIL_PASSWORD",
            "EMAIL_IMAP_HOST",
            "EMAIL_SMTP_HOST",
        ),
        "required_env": (
            "EMAIL_ADDRESS",
            "EMAIL_PASSWORD",
            "EMAIL_IMAP_HOST",
            "EMAIL_SMTP_HOST",
        ),
    },
    "sms": {
        "name": "SMS (Twilio)",
        "description": "Send and receive text messages via Twilio.",
        "docs_url": "https://www.twilio.com/console",
        "env_vars": ("TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN"),
        "required_env": ("TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN"),
    },
    "dingtalk": {
        "name": "DingTalk",
        "description": "Connect Hermes to DingTalk groups (钉钉).",
        "docs_url": "https://open.dingtalk.com/document/orgapp/the-robot-development-process",
        "env_vars": ("DINGTALK_CLIENT_ID", "DINGTALK_CLIENT_SECRET"),
        "required_env": ("DINGTALK_CLIENT_ID", "DINGTALK_CLIENT_SECRET"),
    },
    "feishu": {
        "name": "Feishu / Lark",
        "description": "Use Hermes inside Feishu / Lark.",
        "docs_url": "https://open.feishu.cn/document/uAjLw4CM/ukTMukTMukTM/reference/im-v1/intro",
        "env_vars": (
            "FEISHU_APP_ID",
            "FEISHU_APP_SECRET",
            "FEISHU_ENCRYPT_KEY",
            "FEISHU_VERIFICATION_TOKEN",
        ),
        "required_env": ("FEISHU_APP_ID", "FEISHU_APP_SECRET"),
    },
    "google_chat": {
        "name": "Google Chat",
        "description": "Connect Hermes to Google Chat via Cloud Pub/Sub.",
        "docs_url": "https://hermes-agent.nousresearch.com/docs/user-guide/messaging/google_chat",
    },
    "wecom": {
        "name": "WeCom (group bot)",
        "description": "Send-only WeCom group bot via webhook.",
        "docs_url": "https://developer.work.weixin.qq.com/document/path/91770",
        "env_vars": ("WECOM_BOT_ID", "WECOM_SECRET"),
        "required_env": ("WECOM_BOT_ID",),
    },
    "wecom_callback": {
        "name": "WeCom (app)",
        "description": "Two-way WeCom integration via callback app.",
        "docs_url": "https://developer.work.weixin.qq.com/document/path/90930",
        "env_vars": (
            "WECOM_CALLBACK_CORP_ID",
            "WECOM_CALLBACK_CORP_SECRET",
            "WECOM_CALLBACK_AGENT_ID",
            "WECOM_CALLBACK_TOKEN",
            "WECOM_CALLBACK_ENCODING_AES_KEY",
        ),
        "required_env": (
            "WECOM_CALLBACK_CORP_ID",
            "WECOM_CALLBACK_CORP_SECRET",
            "WECOM_CALLBACK_AGENT_ID",
        ),
    },
    "weixin": {
        "name": "Weixin / WeChat (Personal)",
        "description": "Connect a personal WeChat account through Tencent's iLink Bot API.",
        "docs_url": "https://hermes-agent.nousresearch.com/docs/user-guide/messaging/weixin/",
        "env_vars": ("WEIXIN_ACCOUNT_ID", "WEIXIN_TOKEN", "WEIXIN_BASE_URL"),
        "required_env": ("WEIXIN_ACCOUNT_ID", "WEIXIN_TOKEN"),
    },
    "bluebubbles": {
        "name": "BlueBubbles (iMessage)",
        "description": "Use Hermes through iMessage via a BlueBubbles server.",
        "docs_url": "https://bluebubbles.app/",
        "env_vars": (
            "BLUEBUBBLES_SERVER_URL",
            "BLUEBUBBLES_PASSWORD",
            "BLUEBUBBLES_ALLOWED_USERS",
        ),
        "required_env": ("BLUEBUBBLES_SERVER_URL", "BLUEBUBBLES_PASSWORD"),
    },
    "qqbot": {
        "name": "QQ Bot",
        "description": "Connect Hermes to a QQ Bot from the QQ Open Platform.",
        "docs_url": "https://q.qq.com",
        "env_vars": ("QQ_APP_ID", "QQ_CLIENT_SECRET", "QQ_ALLOWED_USERS"),
        "required_env": ("QQ_APP_ID", "QQ_CLIENT_SECRET"),
    },
    # Teams ships as a platform plugin, so its name/env vars come from the
    # plugin registry. Only the docs link needs an override here so the
    # Channels page can point at the Microsoft Teams setup guide.
    "teams": {
        "description": "Connect Hermes to Microsoft Teams chats via the Bot Framework.",
        "docs_url": "https://hermes-agent.nousresearch.com/docs/user-guide/messaging/teams",
    },
    # Bundled platform plugins: name comes from the plugin registry label;
    # give each a human description (the registry's install_hint is a
    # dependency note, not a description) and a docs link.
    "irc": {
        "description": "Relay messages between an IRC channel (or DMs) and Hermes.",
        "docs_url": "https://hermes-agent.nousresearch.com/docs/user-guide/messaging/irc",
    },
    "line": {
        "description": "Use Hermes from LINE via the LINE Messaging API webhook.",
        "docs_url": "https://hermes-agent.nousresearch.com/docs/user-guide/messaging/line",
    },
    "ntfy": {
        "description": "Chat with Hermes over ntfy push topics (ntfy.sh or self-hosted).",
        "docs_url": "https://hermes-agent.nousresearch.com/docs/user-guide/messaging/ntfy",
    },
    "photon": {
        "description": "Use Hermes through iMessage via Photon's managed Spectrum platform.",
        "docs_url": "https://hermes-agent.nousresearch.com/docs/user-guide/messaging/photon",
    },
    "raft": {
        "description": "Join a Raft workspace as an external agent.",
        "docs_url": "https://hermes-agent.nousresearch.com/docs/user-guide/messaging/raft",
    },
    "simplex": {
        "description": "Talk to Hermes over SimpleX Chat via a local simplex-chat daemon.",
        "docs_url": "https://hermes-agent.nousresearch.com/docs/user-guide/messaging/simplex",
    },
    "yuanbao": {
        "name": "Yuanbao (元宝)",
        "description": "Connect Hermes to Tencent Yuanbao.",
        "docs_url": "",
        "required_env": (),
    },
    "api_server": {
        "name": "API server",
        "description": "Expose Hermes as an OpenAI-compatible HTTP API for tools like Open WebUI.",
        "docs_url": "https://hermes-agent.nousresearch.com/docs/user-guide/messaging/",
        "env_vars": (
            "API_SERVER_ENABLED",
            "API_SERVER_KEY",
            "API_SERVER_PORT",
            "API_SERVER_HOST",
            "API_SERVER_MODEL_NAME",
        ),
        "required_env": (),
    },
    "webhook": {
        "name": "Webhooks",
        "description": "Receive events from GitHub, GitLab, and other webhook sources.",
        "docs_url": "https://hermes-agent.nousresearch.com/docs/user-guide/messaging/webhooks/",
        "env_vars": ("WEBHOOK_ENABLED", "WEBHOOK_PORT", "WEBHOOK_SECRET"),
        "required_env": (),
    },
    "msgraph_webhook": {
        "name": "Microsoft Graph Webhook",
        "description": "Receive Microsoft Graph change notifications (Teams meetings, Outlook, …).",
        "docs_url": "https://hermes-agent.nousresearch.com/docs/user-guide/messaging/msgraph-webhook",
        "required_env": (),
    },
    "whatsapp_cloud": {
        "name": "WhatsApp Cloud API",
        "description": "Use Hermes via Meta's hosted WhatsApp Cloud API (no local bridge).",
        "docs_url": "https://hermes-agent.nousresearch.com/docs/user-guide/messaging/whatsapp-cloud",
    },
    "relay": {
        "name": "Relay (experimental)",
        "description": "Generic relay adapter fronted by the Hermes Relay connector.",
        "docs_url": "",
        "required_env": (),
    },
}

# Display order: well-known platforms surface first; unknown plugins fall to
# the end alphabetically.
_PLATFORM_ORDER: tuple[str, ...] = (
    "telegram",
    "discord",
    "slack",
    "mattermost",
    "matrix",
    "whatsapp",
    "signal",
    "bluebubbles",
    "homeassistant",
    "email",
    "sms",
    "dingtalk",
    "feishu",
    "google_chat",
    "wecom",
    "wecom_callback",
    "weixin",
    "qqbot",
    "yuanbao",
    "api_server",
    "webhook",
)


def _messaging_platform_catalog() -> tuple[dict[str, Any], ...]:
    """Build the messaging catalog from the gateway's Platform enum + plugin registry.

    Built-in platforms come from ``gateway.config.Platform`` (LOCAL is excluded).
    Plugin platforms come from ``gateway.platform_registry.plugin_entries()``,
    which lets newly installed adapters (e.g. IRC) appear without a code change
    here. Per-platform UI metadata (description, docs URL, env-var picks) lives
    in :data:`_PLATFORM_OVERRIDES`; anything not overridden gets reasonable
    defaults derived from the platform id and required_env.
    """
    from gateway.config import Platform

    # Resolve plugin entries FIRST. Plugin platforms (irc, ntfy, photon, …)
    # leak into ``Platform.__members__`` as pseudo-members the moment any
    # earlier code path calls ``Platform("<plugin id>")`` — and iterating the
    # enum first would then claim them with no plugin metadata, rendering
    # nameless "Irc"/"Ntfy" cards with empty descriptions on the Channels
    # page while the real label/install-hint sat unused in the registry.
    plugin_map: dict[str, Any] = {}
    try:
        # Plugin discovery only runs as a side effect of importing
        # model_tools; this server process doesn't do that, so trigger it
        # explicitly (idempotent) or plugin_entries() is empty here and
        # every plugin platform renders nameless.
        from hermes_cli.plugins import discover_plugins

        discover_plugins()
        from gateway.platform_registry import platform_registry

        for plugin_entry in platform_registry.plugin_entries():
            plugin_map[plugin_entry.name] = plugin_entry
    except Exception:
        _log.debug("plugin platform registry unavailable", exc_info=True)

    seen: set[str] = set()
    entries: list[dict[str, Any]] = []

    for member in Platform.__members__.values():
        if member.value == "local":
            continue
        if member.value in seen:
            continue
        seen.add(member.value)
        entries.append(
            _build_catalog_entry(member.value, plugin_map.get(member.value))
        )

    for name, plugin_entry in plugin_map.items():
        if name in seen:
            continue
        seen.add(name)
        entries.append(_build_catalog_entry(name, plugin_entry))

    order = {pid: idx for idx, pid in enumerate(_PLATFORM_ORDER)}
    entries.sort(
        key=lambda e: (order.get(e["id"], len(_PLATFORM_ORDER)), e["name"].lower())
    )
    return tuple(entries)


def _channel_managed_env_keys() -> frozenset[str]:
    """Env-var keys owned by a Channels page platform card.

    The Channels page is the canonical surface for configuring messaging
    platform credentials (with connection status, test, enable toggle and
    gateway restart). The Keys/Env page consults this set to hide those vars
    so the same fields aren't duplicated in a plainer UI. Best-effort: if the
    gateway catalog can't be built, nothing is flagged and Keys shows it all.
    """
    try:
        keys: set[str] = set()
        for entry in _messaging_platform_catalog():
            keys.update(entry.get("env_vars", ()))
        return frozenset(keys)
    except Exception:
        _log.debug("could not build channel-managed env key set", exc_info=True)
        return frozenset()


# Cross-cutting gateway / relay knobs stay on the Keys → Settings tab even though
# they use the ``messaging`` category in OPTIONAL_ENV_VARS. Platform-scoped vars
# (``DISCORD_*``, ``MATRIX_*``, …) are owned by the Messaging UI instead.
_MESSAGING_KEYS_PAGE_KEYS = frozenset({
    "GATEWAY_ALLOW_ALL_USERS",
    "GATEWAY_PROXY_KEY",
    "GATEWAY_PROXY_URL",
})


def _platform_env_prefixes(platform_id: str) -> tuple[str, ...]:
    """Env-var prefixes owned by a messaging platform card."""
    aliases: dict[str, tuple[str, ...]] = {
        "email": ("EMAIL_",),
        "homeassistant": ("HASS_",),
        "qqbot": ("QQ_", "QQBOT_"),
        "sms": ("TWILIO_",),
        "wecom": ("WECOM_BOT_", "WECOM_SECRET"),
        "wecom_callback": ("WECOM_CALLBACK_",),
    }
    if platform_id in aliases:
        return aliases[platform_id]
    return (platform_id.upper().replace("-", "_") + "_",)


# Which per-platform knobs the setup UI hides, and why: see
# hermes_cli/setup_hidden_env.py. Shared with the `hermes setup gateway`
# wizard so the surfaces ask for the same things.
from hermes_cli.setup_hidden_env import (  # noqa: E402
    is_setup_hidden_env as _is_setup_hidden_env,
)


def _discover_platform_env_vars(platform_id: str) -> tuple[str, ...]:
    """All messaging-category env vars for a platform (override + plugin + prefix)."""
    prefixes = _platform_env_prefixes(platform_id)
    keys: list[str] = []
    for name, info in OPTIONAL_ENV_VARS.items():
        if info.get("category") != "messaging":
            continue
        if name in _MESSAGING_KEYS_PAGE_KEYS:
            continue
        if _is_setup_hidden_env(name):
            continue
        if not any(name.startswith(prefix) for prefix in prefixes):
            continue
        keys.append(name)
    return tuple(sorted(set(keys)))


def _merge_platform_env_vars(
    platform_id: str,
    override: dict[str, Any],
    plugin_entry: Any | None,
) -> tuple[str, ...]:
    """Canonical env-var list for a messaging platform card.

    Required credentials always survive: a platform that genuinely needs one of
    the hidden-suffix vars to connect keeps it, since hiding a required field
    would make the platform unconfigurable.
    """
    discovered = _discover_platform_env_vars(platform_id)
    if "env_vars" in override:
        explicit = tuple(
            key for key in override["env_vars"] if not _is_setup_hidden_env(key)
        )
        return tuple(dict.fromkeys((*explicit, *discovered)))
    if plugin_entry is not None and plugin_entry.required_env:
        return tuple(dict.fromkeys((*tuple(plugin_entry.required_env), *discovered)))
    return discovered


def _build_catalog_entry(
    platform_id: str, plugin_entry: Any | None = None
) -> dict[str, Any]:
    override = _PLATFORM_OVERRIDES.get(platform_id, {})

    env_vars = _merge_platform_env_vars(platform_id, override, plugin_entry)

    if "required_env" in override:
        required_env = tuple(override["required_env"])
    elif plugin_entry is not None:
        required_env = tuple(plugin_entry.required_env or ())
    else:
        required_env = ()

    if override.get("name"):
        name = override["name"]
    elif plugin_entry is not None and plugin_entry.label:
        name = plugin_entry.label
    else:
        name = platform_id.replace("_", " ").title()

    description = override.get("description")
    if not description and plugin_entry is not None:
        description = plugin_entry.install_hint or ""

    return {
        "id": platform_id,
        "name": name,
        "description": description or "",
        "docs_url": override.get("docs_url", ""),
        "env_vars": env_vars,
        "required_env": required_env,
    }


def _write_platform_enabled(platform_id: str, enabled: bool) -> None:
    write_platform_config_field(platform_id, "enabled", enabled)


from hermes_cli.web_routers import messaging as _messaging_routes  # noqa: E402

app.include_router(_messaging_routes.router)
from hermes_cli.web_routers.messaging import (  # noqa: E402,F401 — legacy re-exports; tests call these via web_server.<name>
    apply_whatsapp_onboarding,
    start_whatsapp_onboarding,
)


@dataclass
class _WhatsAppOnboardingSession:
    proc: subprocess.Popen | None
    mode: str
    allowed_users: str
    session_path: str
    expires_at: str
    expires_at_ts: float
    profile: str | None = None
    status: str = "starting"
    qr_payload: str | None = None
    account_id: str | None = None
    account_name: str | None = None
    account_phone: str | None = None
    error: str | None = None


_whatsapp_onboarding_sessions: dict[str, _WhatsAppOnboardingSession] = {}


def _whatsapp_session_path() -> Path:
    from hermes_constants import get_hermes_dir

    return get_hermes_dir("platforms/whatsapp/session", "whatsapp/session")


def _whatsapp_onboarding_payload(pairing_id: str, record: _WhatsAppOnboardingSession) -> dict[str, Any]:
    return {
        "pairing_id": pairing_id,
        "status": record.status,
        "qr_payload": record.qr_payload,
        "expires_at": record.expires_at,
        "mode": record.mode,
        "allowed_users": record.allowed_users,
        "account_id": record.account_id,
        "account_name": record.account_name,
        "account_phone": record.account_phone,
        "error": record.error,
    }


def _restart_gateway_after_whatsapp_onboarding(profile: Optional[str] = None) -> dict[str, Any]:
    return _restart_gateway_after(profile, what="WhatsApp onboarding", label="WhatsApp onboarding")


_TELEGRAM_ONBOARDING_DEFAULT_URL = "https://setup.hermes-agent.nousresearch.com"
_TELEGRAM_ONBOARDING_USER_AGENT = f"HermesDashboard/{__version__}"
@dataclass
class _TelegramOnboardingPairing:
    poll_token: str
    expires_at: str
    expires_at_ts: float
    bot_token: str | None = None
    bot_username: str | None = None
    owner_user_id: str | None = None


_telegram_onboarding_pairings: dict[str, _TelegramOnboardingPairing] = {}
_telegram_onboarding_lock = threading.RLock()


def _telegram_onboarding_base_url() -> str:
    return (
        os.getenv("TELEGRAM_ONBOARDING_URL", _TELEGRAM_ONBOARDING_DEFAULT_URL)
        .strip()
        .rstrip("/")
    )


def _telegram_onboarding_error_message(error: str, fallback: str) -> str:
    return {
        "not_found": "Telegram pairing was not found. Start a new setup.",
        "expired": "Telegram setup expired. Start a new setup.",
        "claimed": "Telegram setup was already claimed. Start a new setup.",
        "unauthorized": "Telegram setup service rejected this request.",
        "telegram_manager_bot_token_not_configured": "Telegram setup service is not configured.",
        "telegram_token_fetch_failed": "Telegram could not finish bot setup. Try again.",
    }.get(error, fallback)


def _telegram_onboarding_request_sync(
    method: str,
    path: str,
    *,
    body: dict[str, Any] | None = None,
    bearer_token: str | None = None,
) -> dict[str, Any]:
    import httpx

    headers = {
        "Accept": "application/json",
        "User-Agent": _TELEGRAM_ONBOARDING_USER_AGENT,
    }
    request_kwargs: dict[str, Any] = {}
    if body is not None:
        headers["Content-Type"] = "application/json"
        request_kwargs["json"] = body
    if bearer_token:
        headers["Authorization"] = f"Bearer {bearer_token}"

    url = f"{_telegram_onboarding_base_url()}{path}"
    try:
        with httpx.Client(timeout=httpx.Timeout(10.0)) as client:
            response = client.request(
                method,
                url,
                headers=headers,
                **request_kwargs,
            )
            response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        try:
            parsed = exc.response.json()
        except Exception:
            parsed = {}
        error = str(parsed.get("error") or parsed.get("status") or "")
        detail = _telegram_onboarding_error_message(
            error,
            "Telegram setup service returned an error.",
        )
        status_code = 404 if exc.response.status_code == 404 else 502
        if error in {"expired", "claimed"}:
            status_code = 410
        raise HTTPException(status_code=status_code, detail=detail) from exc
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=502,
            detail="Telegram setup service is unavailable. Try again shortly.",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail="Telegram setup service is unavailable. Try again shortly.",
        ) from exc

    try:
        parsed = response.json()
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail="Telegram setup service returned an invalid response.",
        ) from exc
    if not isinstance(parsed, dict):
        raise HTTPException(
            status_code=502,
            detail="Telegram setup service returned an invalid response.",
        )
    return parsed


# ---------------------------------------------------------------------------
# OAuth provider endpoints — status + disconnect (Phase 1)
# ---------------------------------------------------------------------------
#
# Phase 1 surfaces *which OAuth providers exist* and whether each is
# connected, plus a disconnect button. Anthropic subscription OAuth is
# deliberately delegated away from the dashboard: its card is external and
# points to the supported terminal path. Phase 2 adds in-browser device-code
# flows for providers that support them. For unconnected providers we return
# the canonical ``hermes auth add <provider>`` command so the dashboard can
# surface a one-click copy.


def _truncate_token(value: Optional[str], visible: int = 6) -> str:
    """Return ``...XXXXXX`` (last N chars) for safe display in the UI.

    We never expose more than the trailing ``visible`` characters of an
    OAuth access token. JWT prefixes (the part before the first dot) are
    stripped first when present so the visible suffix is always part of
    the signing region rather than a meaningless header chunk.

    Returns the Entra-ID placeholder when handed a callable (Azure Foundry
    bearer provider) — the callable is NEVER invoked here.
    """
    if not value:
        return ""
    if callable(value) and not isinstance(value, str):
        # Entra ID bearer provider — never reveal a minted token in the UI.
        return "<entra-id-bearer>"
    s = str(value)
    if "." in s and s.count(".") >= 2:
        # Looks like a JWT — show the trailing piece of the signature only.
        s = s.rsplit(".", 1)[-1]
    if len(s) <= visible:
        return s
    return f"…{s[-visible:]}"


def _anthropic_oauth_status() -> Dict[str, Any]:
    """Status for the "Anthropic API Key" catalog entry.

    Two sources, in priority order:
    1. ``~/.hermes/.anthropic_oauth.json`` — Hermes-managed terminal PKCE
       credentials (the dashboard no longer has a Connect button for this)
    2. ``ANTHROPIC_API_KEY`` → ``ANTHROPIC_TOKEN`` → ``CLAUDE_CODE_OAUTH_TOKEN``
       env vars (registry order) — from ``.env``, the shell, or an external
       secret source like Bitwarden (whose keys are injected into the process
       env during ``load_hermes_dotenv()``, so the same check covers them)

    Claude Code's ``~/.claude/.credentials.json`` is deliberately NOT read
    here — it has its own dedicated catalog entry (``claude-code`` →
    ``_claude_code_only_status``). Reporting it under the API-key entry
    double-counts the token and shadows a real ANTHROPIC_API_KEY.
    """
    try:
        from agent.anthropic_adapter import (
            read_hermes_oauth_credentials,
            _get_hermes_oauth_file,
        )
    except ImportError:
        read_hermes_oauth_credentials = None  # type: ignore
        _get_hermes_oauth_file = None  # type: ignore

    hermes_creds = None
    if read_hermes_oauth_credentials:
        try:
            hermes_creds = read_hermes_oauth_credentials()
        except Exception:
            hermes_creds = None
    if hermes_creds and hermes_creds.get("accessToken"):
        return {
            "logged_in": True,
            "source": "hermes_pkce",
            "source_label": f"Hermes PKCE ({_get_hermes_oauth_file() if _get_hermes_oauth_file else None})",
            "token_preview": _truncate_token(hermes_creds.get("accessToken")),
            "expires_at": hermes_creds.get("expiresAt"),
            "has_refresh_token": bool(hermes_creds.get("refreshToken")),
        }

    # Env-var / secret-source path. ``get_env_value`` checks the process
    # environment first (where Bitwarden-sourced secrets land) then .env.
    env_var_order: tuple = ("ANTHROPIC_API_KEY", "ANTHROPIC_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN")
    try:
        from hermes_cli.auth import PROVIDER_REGISTRY
        env_var_order = PROVIDER_REGISTRY["anthropic"].api_key_env_vars
    except (ImportError, KeyError):
        pass
    try:
        from hermes_cli.config import get_env_value
    except ImportError:
        get_env_value = None  # type: ignore
    try:
        from hermes_cli.env_loader import format_secret_source_suffix
    except ImportError:
        format_secret_source_suffix = None  # type: ignore

    for var in env_var_order:
        value = (get_env_value(var) if get_env_value else None) or os.getenv(var)
        if not value:
            continue
        suffix = format_secret_source_suffix(var) if format_secret_source_suffix else ""
        return {
            "logged_in": True,
            "source": "env_var",
            "source_label": f"{var}{suffix}",
            "token_preview": _truncate_token(value),
            "expires_at": None,
            "has_refresh_token": False,
        }
    return {"logged_in": False, "source": None}


def _claude_code_only_status() -> Dict[str, Any]:
    """Surface Claude Code CLI credentials as their own provider entry.

    Independent of the Anthropic entry above so users can see whether their
    Claude Code subscription tokens are actively flowing into Hermes even
    when they also have a separate Hermes-managed PKCE login.
    """
    try:
        from agent.anthropic_adapter import read_claude_code_credentials
        creds = read_claude_code_credentials()
    except Exception:
        creds = None
    if creds and creds.get("accessToken"):
        return {
            "logged_in": True,
            "source": "claude_code_cli",
            "source_label": "~/.claude/.credentials.json",
            "token_preview": _truncate_token(creds.get("accessToken")),
            "expires_at": creds.get("expiresAt"),
            "has_refresh_token": bool(creds.get("refreshToken")),
        }
    return {"logged_in": False, "source": None}


def _copilot_acp_status() -> Dict[str, Any]:
    """Status for copilot-acp — credentials are owned by the Copilot CLI.

    ``logged_in`` is claimed only on positive evidence (a supported env token
    or a known on-disk GitHub Copilot credential store, via
    ``auth.get_external_process_provider_status``). The Copilot CLI may also
    hold its session in an OS keychain Hermes can't read, so the unverified
    state is presented as "managed by the Copilot CLI" — never as signed out.
    """
    try:
        from hermes_cli.auth import get_external_process_provider_status
        status = get_external_process_provider_status("copilot-acp") or {}
    except Exception:
        status = {}
    verified = bool(status.get("auth_verified"))
    configured = bool(status.get("configured"))
    if verified:
        source_label = status.get("auth_source") or "Copilot credentials detected"
    elif configured:
        found = status.get("resolved_command") or status.get("command") or "copilot"
        source_label = f"Managed by the GitHub Copilot CLI ({found})"
    else:
        source_label = "GitHub Copilot CLI not found on PATH"
    return {
        "logged_in": verified,
        "source": "copilot_cli",
        "source_label": source_label,
        "token_preview": None,
        "expires_at": None,
        "has_refresh_token": False,
        "configured": configured,
    }


def _external_process_cli_command(provider_id: str, default: str) -> str:
    """Render an external-process provider's sign-in command with the CLI the
    user actually has configured.

    The static catalog assumes the default executable name; users who point
    Hermes at a custom binary (``HERMES_COPILOT_ACP_COMMAND`` /
    ``COPILOT_CLI_PATH``) would otherwise be told to run a command that isn't
    the one Hermes spawns. Non-external-process providers get ``default`` back
    untouched.
    """
    try:
        from hermes_cli.auth import PROVIDER_REGISTRY, get_external_process_provider_status
        pconfig = PROVIDER_REGISTRY.get(provider_id)
        if not pconfig or pconfig.auth_type != "external_process":
            return default
        status = get_external_process_provider_status(provider_id) or {}
        command = str(status.get("command") or "").strip()
        if command:
            parts = default.split(" ", 1)
            tail = f" {parts[1]}" if len(parts) > 1 else ""
            return f"{command}{tail}"
    except Exception:
        pass
    return default


# Explicit, hand-tuned OAuth/account provider cards. These carry the bits that
# can't be derived from the unified provider catalog: the OAuth ``flow`` shape,
# the per-provider ``status_fn``, the ``cli_command`` fallback, and curated
# display order. They are the OVERRIDE BASE for ``_build_oauth_catalog()``,
# which unions them with every accounts-tab provider in ``provider_catalog()``
# so newly-added OAuth/external providers appear automatically (no hand edit).
# This tuple also still includes two entries that are NOT catalog providers but
# must show on the Accounts tab: the Anthropic credential-status card and the
# synthetic ``claude-code`` subscription row.
# ``flow`` describes the account-management shape so the UI can pick the right
# behavior: ``device_code`` = show code + verification URL + poll, and
# ``external`` = read-only/delegated to a terminal or third-party CLI.
_OAUTH_PROVIDER_CATALOG: tuple[Dict[str, Any], ...] = (
    {
        "id": "nous",
        "name": "Nous Portal",
        "flow": "device_code",
        "cli_command": "hermes auth add nous",
        "docs_url": "https://portal.nousresearch.com",
        "status_fn": None,  # dispatched via auth.get_nous_auth_status
    },
    {
        "id": "openai-codex",
        "name": "ChatGPT or Codex Subscription",
        "flow": "device_code",
        "cli_command": "hermes auth add openai-codex",
        "docs_url": "https://platform.openai.com/docs",
        "status_fn": None,  # dispatched via auth.get_codex_auth_status
    },
    {
        "id": "qwen-oauth",
        "name": "Qwen (via Qwen CLI)",
        "flow": "external",
        "cli_command": "hermes auth add qwen-oauth",
        "docs_url": "https://github.com/QwenLM/qwen-code",
        "status_fn": None,  # dispatched via auth.get_qwen_auth_status
    },
    {
        "id": "minimax-oauth",
        "name": "MiniMax (OAuth)",
        # MiniMax's flow is structurally device-code (verification URI +
        # user code, backend polls the token endpoint) with a PKCE
        # extension for code-binding. The dashboard renders the same UX
        # as Nous's device-code flow; the PKCE bit is a security
        # extension that doesn't change the operator experience.
        "flow": "device_code",
        "cli_command": "hermes auth add minimax-oauth",
        "docs_url": "https://www.minimax.io",
        "status_fn": None,  # dispatched via auth.get_minimax_oauth_auth_status
    },
    {
        "id": "xai-oauth",
        "name": "xAI Grok OAuth (SuperGrok / Premium+)",
        # Device code is the default because it works in remote shells,
        # containers, and desktop installs without requiring a reachable
        # 127.0.0.1 callback.
        "flow": "device_code",
        "cli_command": "hermes auth add xai-oauth",
        "docs_url": "https://hermes-agent.nousresearch.com/docs/guides/xai-grok-oauth",
        "status_fn": None,  # dispatched via auth.get_xai_oauth_auth_status
    },
    {
        "id": "copilot-acp",
        "name": "GitHub Copilot (ACP)",
        "flow": "external",
        # `copilot login` is the CLI's non-interactive device-code login
        # subcommand; the previous `copilot /login` form is not a valid
        # invocation (slash-commands only exist inside an interactive
        # session, reachable as `copilot -i /login`).
        "cli_command": "copilot login",
        "docs_url": "https://docs.github.com/en/copilot",
        "status_fn": _copilot_acp_status,
    },
    # ── Anthropic / Claude entries sit at the bottom.
    #
    # This card is deliberately flow == "external" (no in-dashboard "Connect"
    # button walking the user through claude.ai/oauth/authorize from the web
    # server). Hermes previously reimplemented that subscription-OAuth PKCE
    # dance itself for the dashboard (issues #87887/#87888); that surface was
    # removed because it lets an unattended, scriptable HTTP endpoint mint
    # Claude Pro/Max subscription tokens outside Anthropic's own client,
    # which sits on the wrong side of Anthropic's usage policies for OAuth
    # credentials. Login still works via the terminal (`hermes auth add
    # anthropic`, unaffected by this change) or a plain API key below.
    {
        "id": "anthropic",
        "name": "Anthropic API Key",
        "flow": "external",
        "cli_command": "hermes auth add anthropic",
        "docs_url": "https://docs.claude.com/en/api/getting-started",
        "status_fn": _anthropic_oauth_status,
    },
    {
        "id": "claude-code",
        "name": "Anthropic OAuth: Required Extra Usage Credits to Use Subscription",
        "flow": "external",
        "cli_command": "claude setup-token",
        "docs_url": "https://docs.claude.com/en/docs/claude-code",
        "status_fn": _claude_code_only_status,
    },
)


from hermes_cli.web_routers import oauth as _oauth_routes  # noqa: E402

app.include_router(_oauth_routes.router)
from hermes_cli.web_routers.oauth import (  # noqa: E402,F401 — legacy re-exports; tests call these via web_server.<name>
    _codex_full_login_worker,
    _new_oauth_session,
    _resolve_provider_status,
    start_oauth_login,
)


_oauth_sessions: Dict[str, Dict[str, Any]] = {}
_oauth_sessions_lock = threading.Lock()


def _oauth_profile_name(profile: Optional[str]) -> Optional[str]:
    requested = (profile or "").strip()
    if not requested or requested.lower() == "current":
        return None
    return requested


def _oauth_session_profile(
    session_id: str,
    fallback: Optional[str] = None,
) -> Optional[str]:
    """Return the profile that owns an OAuth session, if one was provided."""
    with _oauth_sessions_lock:
        sess = _oauth_sessions.get(session_id)
        profile = sess.get("profile") if sess else None
    return profile or _oauth_profile_name(fallback)


def _oauth_poller(label: str):
    """Wrap a background device-code poller body ``fn(session_id, sess)``.

    Looks up the session (a vanished session is a no-op), marks it
    ``approved`` when the body returns, and on any exception records
    ``error`` + ``error_message`` on the session instead of raising — the
    thread has no caller to report to; the dashboard reads the status.
    """
    def deco(fn):
        @functools.wraps(fn)
        def poller(session_id: str) -> None:
            with _oauth_sessions_lock:
                sess = _oauth_sessions.get(session_id)
            if not sess:
                return
            try:
                fn(session_id, sess)
                with _oauth_sessions_lock:
                    sess["status"] = "approved"
                _log.info("oauth/device: %s login completed (session=%s)", label, session_id)
            except Exception as e:
                _log.warning("%s device-code poll failed (session=%s): %s", label, session_id, e)
                with _oauth_sessions_lock:
                    sess["status"] = "error"
                    sess["error_message"] = str(e)
        return poller
    return deco

@_oauth_poller("nous")
def _nous_poller(session_id: str, sess: Dict[str, Any]) -> None:
    """Background poller that drives a Nous device-code flow to completion."""
    from hermes_cli.auth import (
        _poll_for_token,
        refresh_nous_oauth_from_state,
    )
    from datetime import datetime, timezone
    import httpx
    portal_base_url = sess["portal_base_url"]
    client_id = sess["client_id"]
    device_code = sess["device_code"]
    interval = sess["interval"]
    scope = sess.get("scope")
    expires_in = max(60, int(sess["expires_at"] - time.time()))
    with httpx.Client(timeout=httpx.Timeout(15.0), headers={"Accept": "application/json"}) as client:
        token_data = _poll_for_token(
            client=client,
            portal_base_url=portal_base_url,
            client_id=client_id,
            device_code=device_code,
            expires_in=expires_in,
            poll_interval=interval,
        )
    # Same post-processing as _nous_device_code_login (validate/refresh JWT)
    now = datetime.now(timezone.utc)
    token_ttl = int(token_data.get("expires_in") or 0)
    auth_state = {
        "portal_base_url": portal_base_url,
        "inference_base_url": token_data.get("inference_base_url"),
        "client_id": client_id,
        "scope": token_data.get("scope") or scope,
        "token_type": token_data.get("token_type", "Bearer"),
        "access_token": token_data["access_token"],
        "refresh_token": token_data.get("refresh_token"),
        "obtained_at": now.isoformat(),
        "expires_at": (
            datetime.fromtimestamp(now.timestamp() + token_ttl, tz=timezone.utc).isoformat()
            if token_ttl else None
        ),
        "expires_in": token_ttl,
    }
    with _profile_scope(_oauth_session_profile(session_id)):
        full_state = refresh_nous_oauth_from_state(
            auth_state,
            timeout_seconds=15.0,
            force_refresh=False,
        )
        from hermes_cli.auth import persist_nous_credentials
        persist_nous_credentials(full_state)


@_oauth_poller("minimax")
def _minimax_poller(session_id: str, sess: Dict[str, Any]) -> None:
    """Background poller that drives a MiniMax OAuth flow to completion.

    Mirrors `_nous_poller` but calls the MiniMax-specific token endpoint,
    which uses a PKCE-style ``code_verifier`` + ``user_code`` rather than
    the ``device_code`` field used by Nous. On success, builds the same
    auth_state dict that ``_minimax_oauth_login`` (the CLI flow) builds
    and persists via ``_minimax_save_auth_state`` — so the dashboard
    path leaves the system in the same state as
    ``hermes auth add minimax-oauth``.
    """
    from hermes_cli.auth import (
        _minimax_poll_token,
        _minimax_resolve_token_expiry_unix,
        _minimax_save_auth_state,
        MINIMAX_OAUTH_GLOBAL_INFERENCE,
        MINIMAX_OAUTH_SCOPE,
    )
    from datetime import datetime, timezone
    import httpx
    portal_base_url = sess["portal_base_url"]
    client_id = sess["client_id"]
    user_code = sess["user_code"]
    code_verifier = sess["code_verifier"]
    interval_ms = sess.get("interval_ms")
    expired_in_raw = sess["expired_in_raw"]
    with httpx.Client(
        timeout=httpx.Timeout(15.0),
        headers={"Accept": "application/json"},
        follow_redirects=True,
    ) as client:
        token_data = _minimax_poll_token(
            client=client,
            portal_base_url=portal_base_url,
            client_id=client_id,
            user_code=user_code,
            code_verifier=code_verifier,
            expired_in=expired_in_raw,
            interval_ms=interval_ms,
        )
    # Build the auth_state dict in the same shape as the CLI flow's
    # `_minimax_oauth_login` so `_minimax_save_auth_state` writes
    # the canonical record. Region is fixed to "global" for the
    # dashboard path; cn-region operators can still use the CLI
    # flow which supports `--region cn`.
    now = datetime.now(timezone.utc)
    expires_at_ts = _minimax_resolve_token_expiry_unix(
        int(token_data["expired_in"]), now=now,
    )
    expires_in_s = max(0, int(expires_at_ts - now.timestamp()))
    auth_state = {
        "provider": "minimax-oauth",
        "region": sess.get("region", "global"),
        "portal_base_url": portal_base_url,
        "inference_base_url": MINIMAX_OAUTH_GLOBAL_INFERENCE,
        "client_id": client_id,
        "scope": MINIMAX_OAUTH_SCOPE,
        "token_type": token_data.get("token_type", "Bearer"),
        "access_token": token_data["access_token"],
        "refresh_token": token_data["refresh_token"],
        "resource_url": token_data.get("resource_url"),
        "obtained_at": now.isoformat(),
        "expires_at": datetime.fromtimestamp(
            expires_at_ts, tz=timezone.utc
        ).isoformat(),
        "expires_in": expires_in_s,
    }
    with _profile_scope(_oauth_session_profile(session_id)):
        _minimax_save_auth_state(auth_state)


@_oauth_poller("xai")
def _xai_device_poller(session_id: str, sess: Dict[str, Any]) -> None:
    """Background poller for xAI's OAuth device-code flow."""
    import httpx
    from hermes_cli.auth import (
        _save_xai_oauth_tokens,
        _xai_oauth_discovery,
        _xai_oauth_poll_device_token,
        mark_provider_active_if_unset,
        unsuppress_credential_source,
    )

    device_code = sess["device_code"]
    interval = int(sess["interval"])
    expires_in = max(60, int(sess["expires_at"] - time.time()))
    discovery = _xai_oauth_discovery(20.0)
    with httpx.Client(
        timeout=httpx.Timeout(20.0),
        headers={"Accept": "application/json"},
    ) as client:
        token_data = _xai_oauth_poll_device_token(
            client,
            token_endpoint=discovery["token_endpoint"],
            device_code=device_code,
            expires_in=expires_in,
            poll_interval=interval,
        )
    tokens = {
        "access_token": str(token_data.get("access_token", "") or "").strip(),
        "refresh_token": str(token_data.get("refresh_token", "") or "").strip(),
        "id_token": str(token_data.get("id_token", "") or "").strip(),
        "expires_in": token_data.get("expires_in"),
        "token_type": str(token_data.get("token_type") or "Bearer").strip() or "Bearer",
    }
    with _profile_scope(_oauth_session_profile(session_id)):
        _save_xai_oauth_tokens(
            tokens,
            discovery=discovery,
            last_refresh=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            auth_mode="oauth_device_code",
            # Persist credentials without hijacking an existing active
            # chat provider.
            set_active=False,
        )
        # Mirror `hermes auth add xai-oauth`: first credential may become
        # active when none is set yet; never overwrite an existing choice.
        mark_provider_active_if_unset("xai-oauth")
        # The singleton write above is the single source of truth: the
        # credential-pool load seeds it as the canonical ``device_code``
        # entry. Do NOT also insert a parallel ``manual:dashboard_*`` pool
        # entry — that duplicates the single-use refresh token across two
        # entries and triggers rotation churn / ``refresh_token_reused``.
        # An interactive dashboard login is also an explicit re-enable
        # signal, so clear any ``device_code`` suppression left by a
        # prior ``hermes auth remove xai-oauth`` (mirrors auth_add_command
        # and the ``hermes model`` re-login path in _login_xai_oauth).
        unsuppress_credential_source("xai-oauth", "device_code")


# ---------------------------------------------------------------------------
# Session detail endpoints
# ---------------------------------------------------------------------------


def _session_latest_descendant(session_id: str, db):
    """Resolve a session id to the newest child leaf session.

    /model may create child sessions. Dashboard refresh should continue the
    newest child instead of reopening the old parent.
    """
    def row_get(row, key, index):
        if isinstance(row, dict):
            return row.get(key)
        try:
            return row[key]
        except Exception:
            try:
                return row[index]
            except Exception:
                return None

    sid = db.resolve_session_id(session_id)
    if not sid or not db.get_session(sid):
        return None, []

    conn = (
        getattr(db, "conn", None)
        or getattr(db, "_conn", None)
        or getattr(db, "connection", None)
        or getattr(db, "_connection", None)
    )

    rows = []
    if conn is not None:
        raw_rows = conn.execute(
            """
            WITH RECURSIVE descendants(id, parent_session_id, started_at) AS (
                SELECT id, parent_session_id, started_at FROM sessions WHERE id = ?
                UNION
                SELECT s.id, s.parent_session_id, s.started_at
                FROM sessions s
                JOIN descendants d ON s.parent_session_id = d.id
            )
            SELECT id, parent_session_id, started_at FROM descendants
            """,
            (sid,),
        ).fetchall()
        for row in raw_rows:
            rows.append({
                "id": row_get(row, "id", 0),
                "parent_session_id": row_get(row, "parent_session_id", 1),
                "started_at": row_get(row, "started_at", 2),
            })
    else:
        rows = db.list_sessions_rich(limit=10000, offset=0, compact_rows=True)

    children = {}
    for row in rows:
        rid = row.get("id")
        parent = row.get("parent_session_id")
        if rid and parent:
            children.setdefault(parent, []).append(row)

    def started(row):
        try:
            return float(row.get("started_at") or 0)
        except Exception:
            return 0.0

    current = sid
    path = [sid]
    seen = {sid}

    while children.get(current):
        candidates = [r for r in children[current] if r.get("id") not in seen]
        if not candidates:
            break
        candidates.sort(key=started, reverse=True)
        current = candidates[0]["id"]
        path.append(current)
        seen.add(current)

    return current, path


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


# Serialises the one-time writable schema bootstrap for read-only opens.
# Concurrent first-load polls otherwise race sqlite file creation: the losers
# open mode=ro against a store whose schema is still being written and every
# query raises "no such table: sessions".
_session_db_bootstrap_lock = threading.Lock()


def _session_db_read_probe_statements() -> tuple:
    """Stale-schema probes for read-only opens, derived from SCHEMA_SQL.

    Read-only opens skip _reconcile_columns(), so an older store would
    otherwise 500 on every poll until something opened it writable. Derived
    from the same schema the writable reconciler applies, so any column
    added there is probed here automatically — the previous hand-written
    probe listed four columns and went stale the first time a new column
    (sessions.last_activity_at) shipped, leaving the desktop sidebar empty
    after `hermes update` until the first message forced a writable open.
    """
    from hermes_state_schema import schema_read_probe_statements

    return schema_read_probe_statements()


# Stores where a heal WRITABLE OPEN SUCCEEDED and the read probe still
# failed afterwards: the schema problem is one reconciliation cannot fix
# (e.g. a NOT-NULL-without-default column SQLite refuses to ADD). Retrying
# the full writable init on every poll would hammer a live DB for nothing,
# so such stores fall back to the raw read-only open until restart. A
# FAILED writable open (transient lock) is deliberately NOT recorded —
# the next poll retries the heal.
_session_db_heal_exhausted: set = set()

# Deduplicates the heal-failure warning per store per process, so a
# persistent problem is loud once instead of once per sidebar poll.
_session_db_heal_warned: set = set()


def _open_session_db_at_path(db_path: Path, *, read_only: bool):
    """Open a SessionDB at an explicit path with an explicit access mode.

    Writable opens keep the full init and repair path. Read-only opens
    bootstrap a missing or zero-byte store once, and heal an older or
    malformed schema through one writable open before reopening read-only.
    The healthy read path never takes a write lock or requests a checkpoint.

    Scope of the heal: the probe checks every table/column declared in
    SCHEMA_SQL (see ``schema_read_probe_statements``), so ANY schema
    addition escalates a stale store to a one-time writable open — the same
    reconcile the store's own backend runs at startup. Tables created
    outside SCHEMA_SQL (telemetry ``tel_*``, FTS shadow tables) are
    deliberately outside both the probe and the heal.
    """
    import sqlite3

    from hermes_state import SessionDB, is_malformed_schema_error

    if not read_only:
        return SessionDB(db_path=db_path, read_only=False)

    def _needs_bootstrap() -> bool:
        try:
            return db_path.stat().st_size == 0
        except FileNotFoundError:
            return True
        except OSError:
            return False

    if _needs_bootstrap():
        with _session_db_bootstrap_lock:
            if _needs_bootstrap():
                SessionDB(db_path=db_path, read_only=False).close()

    def _open_probed():
        db = SessionDB(db_path=db_path, read_only=True)
        # Unit-test fakes may replace SessionDB without exposing a raw
        # connection. Probe only real connections.
        conn = getattr(db, "_conn", None)
        if conn is not None and str(db_path) not in _session_db_heal_exhausted:
            try:
                for statement in _session_db_read_probe_statements():
                    conn.execute(statement).fetchone()
            except BaseException:
                db.close()
                raise
        return db

    try:
        return _open_probed()
    except (sqlite3.DatabaseError, UnicodeDecodeError) as exc:
        message = str(exc).lower()
        stale_schema = "no such table" in message or "no such column" in message
        if not stale_schema and not (
            # UnicodeDecodeError = pysqlite could not decode SQLite's own
            # error message because corrupt file bytes were embedded in it
            # (#98924). The one-writable-open heal is the only repair path,
            # so route it through the same dispatch as malformed schema.
            is_malformed_schema_error(exc) or isinstance(exc, UnicodeDecodeError)
        ):
            raise
        SessionDB(db_path=db_path, read_only=False).close()
        try:
            return _open_probed()
        except (sqlite3.DatabaseError, UnicodeDecodeError) as still_stale:
            message = str(still_stale).lower()
            if "no such table" not in message and "no such column" not in message:
                raise
            # The writable open succeeded but the store is STILL behind the
            # probe: reconciliation cannot fix this one. Serve reads without
            # the probe (queries touching the broken part will still fail,
            # everything else works) and stop paying the writable init per
            # poll.
            _session_db_heal_exhausted.add(str(db_path))
            if str(db_path) not in _session_db_heal_warned:
                _session_db_heal_warned.add(str(db_path))
                _log.warning(
                    "state.db at %s is missing schema that a writable "
                    "reconcile could not add (%s); read paths may partially "
                    "fail until the store is repaired",
                    db_path,
                    still_stale,
                )
            return _open_probed()


def _open_session_db_for_profile(profile: Optional[str], *, read_only: bool):
    """Open a SessionDB with an explicit access mode for a profile.

    ``profile`` None/empty selects this process's own ``state.db``. A named
    profile opens that profile's on-disk store directly. Access-mode
    semantics are documented on :func:`_open_session_db_at_path`.
    """
    from hermes_state import _default_db_path

    if profile:
        _name, home = _cron_profile_home(profile)
        db_path = Path(home) / "state.db"
    else:
        db_path = Path(_default_db_path())
    return _open_session_db_at_path(db_path, read_only=read_only)


# In-process throttle for the opportunistic auto-archive trigger, keyed by
# profile. Bounds the config.yaml read to at most once per this window per
# profile; the actual sweep is throttled far more coarsely by state_meta
# (sessions.min_interval_hours) inside maybe_auto_archive.
_AUTO_ARCHIVE_CHECK_INTERVAL_S = 300.0
_last_auto_archive_check: Dict[str, float] = {}


def _maybe_auto_archive_for_profile(profile: Optional[str]) -> None:
    """Run the config-gated stale-session auto-archive for ``profile``.

    The Desktop backend is spawned as ``hermes serve`` — it runs neither the
    interactive CLI nor the messaging gateway, so neither of those startup
    hooks fire for Desktop users. Triggering the (double-throttled, config-off
    by default) sweep from the session-list path is what makes
    ``sessions.auto_archive`` take effect there. Never raises.
    """
    try:
        key = profile or ""
        now = time.monotonic()
        last = _last_auto_archive_check.get(key)
        if last is not None and now - last < _AUTO_ARCHIVE_CHECK_INTERVAL_S:
            return
        _last_auto_archive_check[key] = now

        from hermes_cli.config import load_config as _load_full_config
        cfg = (_load_full_config().get("sessions") or {})
        if not cfg.get("auto_archive", False):
            return
        db = _open_session_db_for_profile(profile, read_only=False)
        try:
            db.maybe_auto_archive(
                idle_days=float(cfg.get("auto_archive_days", 3)),
                min_interval_hours=int(cfg.get("min_interval_hours", 24)),
            )
        finally:
            db.close()
    except Exception as exc:
        _log.debug("opportunistic auto-archive skipped: %s", exc)


async def _auto_archive_ticker_loop(
    interval_s: float = 3600.0, initial_delay_s: float = 90.0
) -> None:
    """Live timer for the stale-session auto-archive (primary profile).

    A long-running Desktop/serve backend must keep sweeping on schedule even
    when no ``/api/sessions`` request arrives to fire the opportunistic
    trigger — e.g. the app sits open for days on an idle chat. The real
    cadence is still owned by state_meta (``sessions.min_interval_hours``)
    inside ``maybe_auto_archive``; this loop is only the poll rate.
    """

    def _sweep() -> None:
        _maybe_auto_archive_for_profile(None)

    await asyncio.sleep(initial_delay_s)
    while True:
        try:
            await asyncio.to_thread(_sweep)
        except Exception as exc:
            _log.debug("auto-archive tick skipped: %s", exc)
        await asyncio.sleep(interval_s)


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


# ---------------------------------------------------------------------------
# Automation Blueprints — parameterized automation blueprints. The dashboard renders the
# slot schema as a form; submitting instantiates a real cron job via the same
# create_job path. See cron/blueprint_catalog.py for the single source of truth.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# MCP server endpoints — list / add / remove / test.
#
# Wraps the same config data layer the CLI uses (hermes_cli.mcp_config), so
# servers managed here show up under `hermes mcp list` and vice versa.  Secrets
# in stdio `env` blocks are redacted on read; the agent picks them up from
# config.yaml at session start exactly as with CLI-added servers.
# ---------------------------------------------------------------------------


def _normalize_mcp_server_create(
    body: MCPServerCreate,
) -> tuple[str, Dict[str, Any], Optional[str]]:
    """Validate a Dashboard MCP create request and build its safe config.

    The returned config never contains the submitted Bearer token. Callers
    persist the token with the shared Bearer helper only after they enter the
    intended profile scope. Keeping this conversion shared makes the
    standalone MCP page and the Profile Builder enforce the same
    transport/auth contract.
    """
    from hermes_cli.mcp_config import (
        _bearer_auth_headers,
        _strip_bearer_prefix,
    )
    from hermes_cli.mcp_security import validate_mcp_server_entry

    name = (body.name or "").strip()
    if not name:
        raise ValueError("Server name is required")

    url = (body.url or "").strip()
    command = (body.command or "").strip()
    auth = (body.auth or "none").strip().lower()
    bearer_token = (
        body.bearer_token.get_secret_value()
        if body.bearer_token is not None
        else None
    )

    if bool(url) == bool(command):
        raise ValueError("Provide exactly one of URL (HTTP/SSE) or command (stdio)")
    if auth not in {"none", "header", "oauth"}:
        raise ValueError(f"Unsupported auth mode: {auth}")

    server_config: Dict[str, Any] = {}
    if url:
        if body.args:
            raise ValueError("Arguments are only supported for stdio MCP servers")
        if body.env:
            raise ValueError(
                "Environment variables are only supported for stdio MCP servers"
            )
        if auth == "header":
            normalized = _strip_bearer_prefix(bearer_token) if bearer_token else ""
            if not normalized or normalized.lower() == "bearer":
                raise ValueError("Bearer token is required")
            server_config["headers"] = _bearer_auth_headers(name)
        elif body.bearer_token is not None:
            raise ValueError("Bearer token requires header authentication")

        server_config["url"] = url
        if auth == "oauth":
            server_config["auth"] = "oauth"
    else:
        if auth != "none" or body.bearer_token is not None:
            raise ValueError(
                "HTTP authentication is not supported for stdio MCP servers"
            )
        server_config["command"] = command
        if body.args:
            server_config["args"] = list(body.args)
        if body.env:
            server_config["env"] = dict(body.env)

    issues = validate_mcp_server_entry(name, server_config)
    if issues:
        raise ValueError(f"Server '{name}' rejected: {'; '.join(issues)}")
    return name, server_config, bearer_token


def _redact_mcp_env(env: Dict[str, Any]) -> Dict[str, str]:
    """Mask secret-shaped MCP env values for read responses."""
    out: Dict[str, str] = {}
    for k, v in (env or {}).items():
        try:
            out[str(k)] = redact_key(str(v)) if v else ""
        except Exception:
            out[str(k)] = "***"
    return out


def _mcp_server_summary(name: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    transport = "http" if cfg.get("url") else ("stdio" if cfg.get("command") else "unknown")
    auth = cfg.get("auth")
    headers = cfg.get("headers") or {}
    if not auth and isinstance(headers, dict) and any(
        str(key).lower() == "authorization" for key in headers
    ):
        auth = "header"
    return {
        "name": name,
        "transport": transport,
        "url": cfg.get("url"),
        "command": cfg.get("command"),
        "args": list(cfg.get("args") or []),
        "env": _redact_mcp_env(cfg.get("env") or {}),
        "auth": auth,
        "enabled": cfg.get("enabled", True) is not False,
        # Tool selection: list of enabled tool names, or None = all.
        "tools": cfg.get("tools"),
    }


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


_mcp_oauth_flows: dict[str, "DashboardOAuthFlow"] = {}
_mcp_oauth_transactions: dict[tuple[str, str], threading.Lock] = {}
_mcp_oauth_transactions_lock = threading.Lock()


def _mcp_oauth_transaction(flow) -> threading.Lock:
    key = (flow.hermes_home, flow.server_name)
    with _mcp_oauth_transactions_lock:
        return _mcp_oauth_transactions.setdefault(key, threading.Lock())


def _run_dashboard_mcp_oauth(flow, cfg: dict) -> None:
    """Run the normal MCP probe with dashboard redirect/callback handlers."""
    from hermes_cli.mcp_config import (
        _oauth_tokens_present,
        _probe_single_server,
        _save_mcp_server,
    )
    try:
        from agent.secret_scope import (
            build_profile_secret_scope,
            reset_secret_scope,
            set_secret_scope,
        )
        from hermes_constants import reset_hermes_home_override, set_hermes_home_override
        from tools.mcp_dashboard_oauth import dashboard_oauth_flow
        from tools.mcp_oauth import HermesTokenStorage, force_interactive_oauth
        from tools.mcp_oauth_manager import get_manager

        home_token = set_hermes_home_override(flow.hermes_home)
        secret_token = set_secret_scope(build_profile_secret_scope(Path(flow.hermes_home)))
        try:
            transaction = _mcp_oauth_transaction(flow)
            with transaction, force_interactive_oauth(), dashboard_oauth_flow(flow):
                manager = get_manager()
                storage = HermesTokenStorage(flow.server_name)
                backup = storage.snapshot()
                previous_entry = None
                try:
                    previous_entry = manager.remove(
                        flow.server_name,
                        hermes_home=flow.hermes_home,
                    )
                    tools = _probe_single_server(
                        flow.server_name,
                        cfg,
                        connect_timeout=max(float(cfg.get("connect_timeout", 0) or 0), 315),
                    )
                    if not _oauth_tokens_present(flow.server_name):
                        raise RuntimeError(
                            "The server responded, but no OAuth token was obtained — "
                            "this provider may require a manually-registered OAuth client."
                        )
                    _save_mcp_server(flow.server_name, cfg)
                    flow.tools = [{"name": t, "description": d} for t, d in tools]
                    flow.mark_approved()
                    if flow.reconnect_live:
                        from tools.mcp_tool import reconnect_mcp_server

                        reconnect_mcp_server(flow.server_name)
                except Exception:
                    storage.restore(backup, only_if_absent=True)
                    manager.restore_entry(
                        flow.server_name,
                        previous_entry,
                        hermes_home=flow.hermes_home,
                    )
                    raise
        finally:
            reset_secret_scope(secret_token)
            reset_hermes_home_override(home_token)
    except Exception as exc:
        msg = str(exc)
        # Providers that gate RFC 7591 registration to pre-approved clients
        # (Figma's MCP catalog, etc.) 403 the register call before any
        # authorization URL exists — surface what's actually happening
        # instead of a bare "403 Forbidden".
        try:
            from tools.mcp_oauth import humanize_oauth_registration_error

            humanized = humanize_oauth_registration_error(
                flow.server_name,
                exc,
                server_url=cfg.get("url") if isinstance(cfg, dict) else None,
            )
            if humanized:
                msg = humanized
        except Exception:
            pass
        flow.mark_error(msg)
    finally:
        flow.mark_worker_done()


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


def _fallback_profile_dicts(profiles_mod) -> List[Dict[str, Any]]:
    def _safe(callable_, default):
        try:
            return callable_()
        except Exception:
            return default

    profiles: List[Dict[str, Any]] = []
    default_home = profiles_mod._get_default_hermes_home()
    if default_home.is_dir():
        model, provider = _safe(lambda: profiles_mod._read_config_model(default_home), (None, None))
        profiles.append({
            "name": "default",
            "path": str(default_home),
            "is_default": True,
            "model": model,
            "provider": provider,
            "has_env": (default_home / ".env").exists(),
            "skill_count": _safe(lambda: profiles_mod._count_skills(default_home), 0),
            "gateway_running": _safe(lambda: profiles_mod._check_gateway_running(default_home), False),
            "description": _safe(lambda: profiles_mod.read_profile_meta(default_home).get("description", ""), ""),
            "description_auto": _safe(lambda: profiles_mod.read_profile_meta(default_home).get("description_auto", False), False),
            "distribution_name": None,
            "distribution_version": None,
            "distribution_source": None,
            "has_alias": False,
        })

    profiles_root = profiles_mod._get_profiles_root()
    if profiles_root.is_dir():
        # Use os.scandir (context-managed) instead of Path.iterdir to avoid
        # leaking directory fds when an exception interrupts iteration — the
        # sidebar polls every few seconds so an fd leak exhausts RLIMIT_NOFILE
        # within days (#81547).
        with os.scandir(profiles_root) as scan:
            entries = sorted(scan, key=lambda e: e.name)
        for entry in entries:
            entry_path = Path(entry.path)
            if not entry.is_dir() or not profiles_mod._PROFILE_ID_RE.match(entry.name):
                continue
            model, provider = _safe(lambda entry=entry_path: profiles_mod._read_config_model(entry), (None, None))
            profiles.append({
                "name": entry.name,
                "path": str(entry_path),
                "is_default": False,
                "model": model,
                "provider": provider,
                "has_env": _safe(lambda entry=entry_path: (entry / ".env").exists(), False),
                "skill_count": _safe(lambda entry=entry_path: profiles_mod._count_skills(entry), 0),
                "gateway_running": _safe(
                    lambda entry=entry_path, name=entry.name: (
                        profiles_mod._check_gateway_running(entry)
                        or profiles_mod._served_by_running_multiplexer(name)
                    ),
                    False,
                ),
                "description": _safe(lambda entry=entry_path: profiles_mod.read_profile_meta(entry).get("description", ""), ""),
                "description_auto": _safe(lambda entry=entry_path: profiles_mod.read_profile_meta(entry).get("description_auto", False), False),
                "distribution_name": None,
                "distribution_version": None,
                "distribution_source": None,
                "has_alias": False,
            })

    return profiles


def _resolve_profile_dir(name: str) -> Path:
    """Validate ``name`` and resolve to its directory or raise an HTTPException."""
    from hermes_cli import profiles as profiles_mod
    try:
        profiles_mod.validate_profile_name(name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not profiles_mod.profile_exists(name):
        raise HTTPException(status_code=404, detail=f"Profile '{name}' does not exist.")
    return profiles_mod.get_profile_dir(name)


def _write_profile_mcp_servers(profile_dir: Path, servers: List["MCPServerCreate"]) -> int:
    """Write MCP server entries into a specific profile's config.yaml.

    Scopes ``load_config``/``save_config`` to ``profile_dir`` via the
    context-local HERMES_HOME override (same mechanism as
    ``_write_profile_model``) so the entries land in the target profile's
    config rather than the dashboard process's active profile.

    Mirrors the per-server shape the ``POST /api/mcp/servers`` endpoint builds,
    but batched so the whole profile-create write is a single config save.
    Returns the number of servers written.
    """
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    from hermes_cli.mcp_config import _save_bearer_auth_token

    written = 0
    token = set_hermes_home_override(str(profile_dir))
    try:
        cfg = load_config()
        mcp = cfg.setdefault("mcp_servers", {})
        for server in servers:
            try:
                name, entry, bearer_token = _normalize_mcp_server_create(server)
            except ValueError as exc:
                display_name = (server.name or "").strip() or "<unnamed>"
                _log.warning(
                    "Profile-create: skipping MCP server '%s': %s",
                    display_name,
                    exc,
                )
                continue
            if bearer_token is not None:
                entry["headers"] = _save_bearer_auth_token(name, bearer_token)
            mcp[name] = entry
            written += 1
        if written:
            save_config(cfg)
        elif not mcp:
            # We created an empty mcp_servers dict but wrote nothing — don't
            # leave a stray empty key in the new profile's config.
            cfg.pop("mcp_servers", None)
            save_config(cfg)
    finally:
        reset_hermes_home_override(token)
    return written


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


# ---------------------------------------------------------------------------
# Skills & Tools endpoints
#
# Every read/write below accepts an optional ``profile`` query param so the
# dashboard can manage ANY profile's skills/toolsets, not just the profile
# the dashboard process happens to be running under. Without this, "Set as
# active" on the Profiles page (which only flips the sticky ``active_profile``
# file for FUTURE CLI/gateway invocations) misled users into thinking skill
# toggles would land in the activated profile — they silently wrote into the
# dashboard's own config instead. See _profile_scope() for the mechanism.
# ---------------------------------------------------------------------------


_SKILLS_PROFILE_LOCK = threading.RLock()


@contextmanager
def _profile_scope(profile: Optional[str]):
    """Scope config + skill-directory resolution to ``profile`` for one request.

    Two seams must be redirected for skills/toolsets endpoints:

    1. ``load_config``/``save_config`` resolve ``get_hermes_home()`` at call
       time — the context-local override from ``set_hermes_home_override``
       reaches them (same pattern as ``_write_profile_model``).
    2. ``tools.skills_tool`` and ``tools.skill_manager_tool`` bind
       ``SKILLS_DIR`` at import time, so the override CANNOT reach them.
       Like ``_call_cron_for_profile`` does for cron's module globals,
       temporarily retarget both under a lock and restore them
       immediately after.

    ``tools.skills_sync`` (reset/diff/list-modified/opt-in/opt-out/
    repair-official) needs NO retargeting: since #65828 its directory
    lookups resolve at call time through the same contextvar override
    set in step 1.

    ``profile`` of None/""/"current" means "the dashboard's own profile" —
    config resolution is untouched, but the skill-module globals are still
    retargeted to the *current* ``get_hermes_home()`` so writes land in the
    live home even when the import-time binding is stale (e.g. the process
    imported the modules before a HERMES_HOME override, or under test
    isolation).
    """
    requested = (profile or "").strip()

    from hermes_constants import (
        get_hermes_home,
        set_hermes_home_override,
        reset_hermes_home_override,
    )
    from tools import skills_tool as _skills_tool
    from tools import skill_manager_tool as _skill_mgr

    token = None
    if not requested or requested.lower() == "current":
        profile_dir = get_hermes_home()
    else:
        profile_dir = _resolve_profile_dir(requested)
        token = set_hermes_home_override(str(profile_dir))

    with _SKILLS_PROFILE_LOCK:
        old_home = _skills_tool.HERMES_HOME
        old_skills_dir = _skills_tool.SKILLS_DIR
        old_mgr_home = _skill_mgr.HERMES_HOME
        old_mgr_skills_dir = _skill_mgr.SKILLS_DIR
        _skills_tool.HERMES_HOME = profile_dir
        _skills_tool.SKILLS_DIR = profile_dir / "skills"
        _skill_mgr.HERMES_HOME = profile_dir
        _skill_mgr.SKILLS_DIR = profile_dir / "skills"
        try:
            yield profile_dir if token is not None else None
        finally:
            _skills_tool.HERMES_HOME = old_home
            _skills_tool.SKILLS_DIR = old_skills_dir
            _skill_mgr.HERMES_HOME = old_mgr_home
            _skill_mgr.SKILLS_DIR = old_mgr_skills_dir
            if token is not None:
                reset_hermes_home_override(token)


@contextmanager
def _config_profile_scope(profile: Optional[str]):
    """Await-safe, config-only profile scope for handlers that ``await``.

    Unlike ``_profile_scope`` this touches ONLY the context-local
    ``set_hermes_home_override`` contextvar — it does NOT swap the
    process-global ``skills_tool``/``skill_manager`` module attributes.
    Those globals are shared across all event-loop tasks, so holding them
    across an ``await`` lets a concurrent skills request restore THIS
    request's profile dir on its ``finally`` (cross-contamination). The
    contextvar override is task-local and survives an ``await`` cleanly,
    which is all endpoints that resolve ``get_hermes_home()`` at call time
    (config, env, gateway status) actually need.

    None/""/"current" means the dashboard's own profile — no override.
    """
    requested = (profile or "").strip()
    if not requested or requested.lower() == "current":
        yield None
        return

    from hermes_constants import (
        set_hermes_home_override,
        reset_hermes_home_override,
    )

    profile_dir = _resolve_profile_dir(requested)
    token = set_hermes_home_override(str(profile_dir))
    try:
        yield profile_dir
    finally:
        reset_hermes_home_override(token)


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


# ---------------------------------------------------------------------------
# Terminal execution backend picker — the GUI counterpart of terminal.backend
# in config.yaml. Each row carries a fast, defensive health probe (Docker
# daemon reachable, SSH host configured, Modal/Daytona credentials present) so
# the Capabilities panel can render Ready / Needs setup guidance instead of a
# bare enum (issues #57738 / #63783). Probes must never raise — a probe
# failure renders as a status, not a 500.
# ---------------------------------------------------------------------------

# Table-driven backend metadata — kept in sync with the dispatch ladder in
# tools/terminal_tool.py::_create_environment and the terminal.backend enum
# surfaced in the desktop raw-config settings.
_TERMINAL_BACKENDS: List[Dict[str, str]] = [
    {
        "name": "local",
        "label": "Local",
        "description": "Run commands directly on this machine. No isolation.",
    },
    {
        "name": "docker",
        "label": "Docker",
        "description": "Run commands in an isolated Docker container with a persistent workspace.",
    },
    {
        "name": "singularity",
        "label": "Singularity / Apptainer",
        "description": "Run commands in a Singularity/Apptainer container (HPC-friendly, rootless).",
    },
    {
        "name": "modal",
        "label": "Modal",
        "description": "Run commands in a Modal cloud sandbox.",
    },
    {
        "name": "daytona",
        "label": "Daytona",
        "description": "Run commands in a Daytona cloud sandbox.",
    },
    {
        "name": "ssh",
        "label": "SSH",
        "description": "Run commands on a remote host over SSH.",
    },
]


def _plugin_terminal_backend_rows() -> List[Dict[str, str]]:
    """Picker rows for plugin-registered terminal backends (fail-soft)."""
    rows: List[Dict[str, str]] = []
    try:
        from hermes_cli.plugins import discover_plugins

        discover_plugins()  # idempotent — plugin state may not be loaded yet
    except Exception:
        pass
    try:
        from agent.terminal_env_registry import list_providers

        for provider in list_providers():
            try:
                rows.append({
                    "name": provider.name.strip().lower(),
                    "label": provider.display_name,
                    "description": provider.description,
                })
            except Exception:
                continue
    except Exception:
        return rows
    return rows


from hermes_cli.web_routers import analytics as _analytics_routes  # noqa: E402

app.include_router(_analytics_routes.router)
from hermes_cli.web_routers.analytics import (  # noqa: E402,F401 — legacy re-exports; tests call these via web_server.<name>
    get_models_analytics,
    get_usage_analytics,
)


# ---------------------------------------------------------------------------
# Token / cost analytics endpoint
# ---------------------------------------------------------------------------


def _aux_usage_rows(db, cutoff: float) -> List[Dict[str, Any]]:
    """Per-(model, task) auxiliary usage within the window (issue #23270).

    Reads the task-dimension rows (task != '') that record_auxiliary_usage
    writes into session_model_usage. Returns [] when the table predates the
    task column (older DB opened read-only by newer code).
    """
    try:
        cur = db._conn.execute("""
            SELECT u.model,
                   u.task,
                   u.billing_provider,
                   SUM(u.input_tokens) as input_tokens,
                   SUM(u.output_tokens) as output_tokens,
                   SUM(u.cache_read_tokens) as cache_read_tokens,
                   SUM(u.reasoning_tokens) as reasoning_tokens,
                   COALESCE(SUM(u.estimated_cost_usd), 0) as estimated_cost,
                   COUNT(DISTINCT u.session_id) as sessions,
                   SUM(COALESCE(u.api_call_count, 0)) as api_calls,
                   MAX(u.last_seen) as last_used_at
            FROM session_model_usage u
            JOIN sessions s ON s.id = u.session_id
            WHERE s.started_at > ? AND u.task != ''
            GROUP BY u.model, u.task, u.billing_provider
            ORDER BY SUM(u.input_tokens) + SUM(u.output_tokens) DESC
        """, (cutoff,))
        return [dict(r) for r in cur.fetchall()]
    except Exception:
        # Table predates the task column (older DB opened by newer code) —
        # aux breakdown is simply unavailable.
        return []


def _merge_aux_into_by_model(
    by_model: List[Dict[str, Any]], aux_rows: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Fold aux usage rows into the sessions-derived per-model list.

    Aux usage lives only in session_model_usage (never in the sessions
    counters), so adding it here cannot double-count. Models that ONLY
    appear via aux calls (e.g. a dedicated vision model) get their own
    entry — previously they were entirely invisible.
    """
    if not aux_rows:
        return by_model
    merged: Dict[str, Dict[str, Any]] = {}
    for row in by_model:
        merged[row.get("model") or "unknown"] = row
    for aux in aux_rows:
        model = aux.get("model") or "unknown"
        target = merged.get(model)
        if target is None:
            target = {
                "model": model,
                "input_tokens": 0,
                "output_tokens": 0,
                "estimated_cost": 0,
                "sessions": 0,
                "api_calls": 0,
            }
            merged[model] = target
        target["input_tokens"] = (target.get("input_tokens") or 0) + (aux.get("input_tokens") or 0)
        target["output_tokens"] = (target.get("output_tokens") or 0) + (aux.get("output_tokens") or 0)
        target["estimated_cost"] = (target.get("estimated_cost") or 0) + (aux.get("estimated_cost") or 0)
        target["api_calls"] = (target.get("api_calls") or 0) + (aux.get("api_calls") or 0)
        tasks = target.setdefault("aux_tasks", [])
        tasks.append({
            "task": aux.get("task") or "",
            "input_tokens": aux.get("input_tokens") or 0,
            "output_tokens": aux.get("output_tokens") or 0,
            "estimated_cost": aux.get("estimated_cost") or 0,
            "api_calls": aux.get("api_calls") or 0,
        })
    result = list(merged.values())
    result.sort(
        key=lambda r: (r.get("input_tokens") or 0) + (r.get("output_tokens") or 0),
        reverse=True,
    )
    return result


def _aux_task_summary(aux_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Aggregate aux usage rows across models into a per-task summary."""
    by_task: Dict[str, Dict[str, Any]] = {}
    for aux in aux_rows:
        task = aux.get("task") or ""
        d = by_task.setdefault(task, {
            "task": task,
            "input_tokens": 0,
            "output_tokens": 0,
            "estimated_cost": 0,
            "api_calls": 0,
            "models": [],
        })
        d["input_tokens"] += aux.get("input_tokens") or 0
        d["output_tokens"] += aux.get("output_tokens") or 0
        d["estimated_cost"] += aux.get("estimated_cost") or 0
        d["api_calls"] += aux.get("api_calls") or 0
        model = aux.get("model") or "unknown"
        if model not in d["models"]:
            d["models"].append(model)
    result = list(by_task.values())
    result.sort(
        key=lambda r: (r.get("input_tokens") or 0) + (r.get("output_tokens") or 0),
        reverse=True,
    )
    return result


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


def _read_bound_port(server: "uvicorn.Server", fallback: int) -> int:
    """Read the OS-assigned port from a live uvicorn server socket.

    After ``server.startup()`` the socket is bound.  Returns the actual
    port so ephemeral (port-0) discovery works without a pre-bind TOCTOU.
    Falls back to *fallback* if the socket list is empty (shouldn't happen
    but guards against uvicorn internals changing).
    """
    if server.servers and server.servers[0].sockets:
        return server.servers[0].sockets[0].getsockname()[1]
    return fallback


def _write_dashboard_ready_file(actual_port: int) -> None:
    """Optionally publish the dashboard port through an atomic ready file.

    Windows Desktop can launch dashboard backends with ``pythonw.exe`` to avoid
    console flashes. That path cannot rely on stdout for the port announcement,
    so Electron passes ``HERMES_DESKTOP_READY_FILE`` and waits for this JSON.
    Normal CLI/dashboard launches still use the stdout READY line below.
    """
    target = os.environ.get("HERMES_DESKTOP_READY_FILE")
    if not target:
        return

    tmp_name = ""
    try:
        path = Path(target)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({"port": int(actual_port)}, separators=(",", ":"))
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=str(path.parent),
            prefix=f"{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
            tmp_name = fh.name
        os.replace(tmp_name, path)
    except Exception as exc:
        if tmp_name:
            try:
                Path(tmp_name).unlink(missing_ok=True)
            except Exception:
                pass
        _log.warning("Failed to write dashboard ready file %r: %s", target, exc)


def _maybe_open_browser(
    host: str, actual_port: int, open_browser: bool, initial_profile: str
) -> None:
    """Open the dashboard URL in the user's browser if appropriate.

    Skips on headless Linux (no ``DISPLAY`` / ``WAYLAND_DISPLAY``) to avoid
    TUI browsers (links, lynx) that would SIGHUP the server process.
    Maps ``0.0.0.0`` / ``::`` binds to ``127.0.0.1`` so the browser opens
    a reachable URL.
    """
    if not open_browser:
        return

    import webbrowser

    _has_display = (
        sys.platform != "linux"
        or bool(os.environ.get("DISPLAY"))
        or bool(os.environ.get("WAYLAND_DISPLAY"))
    )
    if not _has_display:
        _log.debug(
            "Skipping browser-open: no DISPLAY or WAYLAND_DISPLAY detected "
            "(headless Linux). Pass --no-open to suppress this detection."
        )
        return

    _display_host = host if host not in ("0.0.0.0", "::") else "127.0.0.1"
    _open_url = f"http://{_display_host}:{actual_port}"
    if initial_profile:
        from urllib.parse import quote
        _open_url += f"/?profile={quote(initial_profile)}"

    def _open():
        try:
            time.sleep(1.0)
            webbrowser.open(_open_url)
        except Exception:
            pass

    threading.Thread(target=_open, daemon=True).start()


def _is_serve_orphaned(
    desktop_pid: int,
    expected_start_marker: Optional[str] = None,
    *,
    pid_exists=None,
    process_start_marker=None,
) -> bool:
    """True when the exact Desktop process that owns this backend is gone.

    ``HERMES_PARENT_PID`` is the Electron Desktop PID, not necessarily this
    Python process's immediate PPID. On Windows the venv ``hermes.exe`` launcher
    introduces one or more shim processes, so comparing ``os.getppid()`` to the
    Electron PID incorrectly treats a healthy backend as orphaned and exits 0.

    New Desktop versions also provide the owner's process-start marker. This
    prevents a recycled PID from keeping an orphan alive. Older versions remain
    compatible through the PID-only probe. Any inconclusive probe failure is
    fail-safe: keep serving rather than killing a backend whose owner could not
    be conclusively shown to be dead.
    """
    try:
        if expected_start_marker is not None:
            probe = process_start_marker or _process_start_marker
            return not _parent_start_markers_match(
                probe(int(desktop_pid)), expected_start_marker
            )

        if pid_exists is None:
            from gateway.status import _pid_exists

            pid_exists = _pid_exists
        return not bool(pid_exists(int(desktop_pid)))
    except ProcessLookupError:
        return True
    except Exception:
        return False


def _start_parent_death_watchdog() -> None:
    """Exit when the exact desktop parent that spawned this backend dies.

    The desktop passes its PID and, in newer versions, its process-start marker
    plus a per-spawn nonce. The marker distinguishes a live owner from PID reuse;
    the nonce makes partial/mixed-version identity plumbing fail safe. Legacy
    Desktop versions that provide only ``HERMES_PARENT_PID`` retain PID-only
    tracking.
    """
    raw_pid = os.environ.get("HERMES_PARENT_PID")
    start_marker = os.environ.get("HERMES_PARENT_START_MARKER")
    nonce = os.environ.get("HERMES_PARENT_NONCE")

    try:
        desktop_pid = int(raw_pid or "")
    except (TypeError, ValueError):
        return
    if desktop_pid <= 0:
        return

    has_marker = start_marker is not None
    has_nonce = nonce is not None
    if has_marker != has_nonce:
        return
    if has_marker and (
        not _valid_parent_start_marker(start_marker or "")
        or not nonce
        or nonce != nonce.strip()
    ):
        return

    try:
        poll = max(0.5, float(os.environ.get("HERMES_SERVE_WATCHDOG_POLL_S", "2.0")))
    except (TypeError, ValueError):
        poll = 2.0

    def _loop() -> None:
        while not _is_serve_orphaned(desktop_pid, start_marker):
            time.sleep(poll)
        os._exit(0)

    threading.Thread(target=_loop, daemon=True, name="serve-parent-watchdog").start()


# ── Port-conflict sentinel (#93608) ─────────────────────────────────────────
# When the requested port is already bound, uvicorn's ``bind_socket()``
# catches the OSError itself and does ``logger.error(exc); sys.exit(1)`` — a
# bare ERROR line plus the same exit 1 as any real backend crash. The desktop
# spawn (and any script wrapping ``hermes serve``) cannot tell "port occupied"
# from "backend broken". So we probe the exact bind before handing the socket
# to uvicorn and, on conflict, emit ONE machine-readable stdout sentinel plus
# a human hint, then exit with a distinct code.
#
# 75 == BSD ``EX_TEMPFAIL`` (sysexits.h) — the codebase's existing convention
# for "transient environmental condition, not a code failure" (see
# gateway/restart.py and kanban_db.py's quota-wall sentinel).
PORT_IN_USE_EXIT_CODE = 75

# One line, stable format, parsed by machines — mirrors the shape of the
# HERMES_BACKEND_READY sentinel (which is NOT changed by any of this).
_PORT_IN_USE_SENTINEL = "BACKEND_PORT_IN_USE port={port}"


def _is_addr_in_use_error(exc: OSError) -> bool:
    """True when ``exc`` is the platform's address-in-use bind failure."""
    import errno

    codes = {errno.EADDRINUSE, 98, 48, 10048}  # POSIX, Linux, macOS, WinSock
    if exc.errno in codes:
        return True
    return getattr(exc, "winerror", None) == 10048  # WSAEADDRINUSE


def _port_bind_conflict(host: str, port: int) -> bool:
    """Probe whether binding ``host:port`` would fail with EADDRINUSE.

    ``port == 0`` (ephemeral) can never conflict — the kernel picks a free
    port — so the probe is skipped and ``--port 0`` behaves exactly as
    before. Any probe error other than address-in-use returns ``False`` so
    uvicorn surfaces it with its normal diagnostics (bad host, EACCES, …).
    """
    if not port:
        return False
    import socket as _socket

    family = _socket.AF_INET6 if ":" in host else _socket.AF_INET
    try:
        probe = _socket.socket(family, _socket.SOCK_STREAM)
    except OSError:
        return False
    try:
        import sys as _sys_mod

        _exclusive = getattr(_socket, "SO_EXCLUSIVEADDRUSE", None)
        if _sys_mod.platform == "win32" and _exclusive is not None:
            # Windows: SO_REUSEADDR means "bind over anyone" — a probe (or
            # uvicorn bind) with it SUCCEEDS on top of a live LISTEN socket,
            # so it can never detect a conflict. SO_EXCLUSIVEADDRUSE makes
            # the probe fail with WSAEADDRINUSE exactly when another socket
            # holds the port (the reporter's 10048 shape in #93608).
            probe.setsockopt(_socket.SOL_SOCKET, _exclusive, 1)
        else:
            # POSIX: match uvicorn's bind flags (uvicorn/config.py
            # bind_socket) so the probe conflicts exactly when uvicorn's own
            # bind would: SO_REUSEADDR lets TIME_WAIT remnants pass while a
            # live LISTEN socket still fails.
            probe.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1)
        probe.bind((host, port))
    except OSError as exc:
        return _is_addr_in_use_error(exc)
    except Exception:
        return False
    finally:
        probe.close()
    return False


def _write_machine_sentinel_line(line: str) -> None:
    """Write a machine-parsed sentinel line to the REAL stdout (fd 1).

    The serve startup path imports ``tui_gateway.server`` (flush-on-SIGTERM
    handlers, #94724) which redirects ``sys.stdout`` to ``sys.stderr`` at
    import time to keep stray prints off the JSON-RPC protocol stream. Any
    machine-readable sentinel printed after that import via ``print()`` lands
    on stderr — invisible to consumers that parse the child's stdout pipe
    (the Desktop spawn, scripts). fd 1 is untouched by the Python-level
    redirect, so write there.

    Best-effort by design: if fd 1 is unwritable (closed; invalid under
    pythonw.exe), fall back to ``print()`` for human visibility only — the
    redirected stream can't reach stdout-parsing consumers, and pythonw
    Desktop spawns rely on ``_write_dashboard_ready_file()`` (the
    HERMES_DESKTOP_READY_FILE channel) for port discovery instead. Never
    raises: a sentinel-delivery failure must not kill a healthy serve.
    """
    try:
        os.write(1, (line + "\n").encode())
    except OSError:
        try:
            print(line, flush=True)
        except Exception:
            pass


def _report_port_in_use(host: str, port: int) -> None:
    """Print the machine sentinel + a human hint naming likely holders."""
    _write_machine_sentinel_line(_PORT_IN_USE_SENTINEL.format(port=port))
    print(
        f"  Port {port} on {host} is already in use — likely another "
        "'hermes serve' / 'hermes dashboard' backend or the Hermes gateway. "
        "Stop the other process, or pass --port <other> "
        "(--port 0 picks a free ephemeral port).",
        flush=True,
    )


_DEFAULT_DASHBOARD_FORWARDED_ALLOW_IPS = ("127.0.0.1", "::1")


def _dashboard_forwarded_allow_ips(dashboard_config: dict[str, Any]) -> list[str]:
    """Return the bounded proxy addresses uvicorn may trust.

    Uvicorn's default trusts loopback. Preserve that behavior and extend it
    only with explicit IP addresses or CIDR networks from config. Invalid or
    unbounded entries fail closed instead of turning arbitrary client-supplied
    forwarding headers into request metadata.
    """
    configured = dashboard_config.get("trusted_proxies", [])
    if configured in (None, ""):
        configured = []
    elif isinstance(configured, str):
        configured = [configured]
    elif not isinstance(configured, (list, tuple)):
        _log.warning(
            "dashboard.trusted_proxies must be a list of IP addresses or CIDR networks; "
            "ignoring %r",
            configured,
        )
        configured = []

    trusted = list(_DEFAULT_DASHBOARD_FORWARDED_ALLOW_IPS)
    for raw_entry in configured:
        if not isinstance(raw_entry, str) or not raw_entry.strip():
            _log.warning(
                "Ignoring invalid dashboard.trusted_proxies entry %r; expected an IP "
                "address or CIDR network",
                raw_entry,
            )
            continue

        entry = raw_entry.strip()
        try:
            if "/" in entry:
                network = ipaddress.ip_network(entry, strict=False)
                if network.prefixlen == 0:
                    raise ValueError("unbounded network")
                normalized = str(network)
            else:
                normalized = str(ipaddress.ip_address(entry))
        except ValueError:
            _log.warning(
                "Ignoring unsafe dashboard.trusted_proxies entry %r; use a bounded IP "
                "address or CIDR network, never '*' or a /0 network",
                raw_entry,
            )
            continue

        if normalized not in trusted:
            trusted.append(normalized)

    if trusted != list(_DEFAULT_DASHBOARD_FORWARDED_ALLOW_IPS):
        _log.info("Dashboard trusted proxies: %s", ", ".join(trusted))

    return trusted


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
