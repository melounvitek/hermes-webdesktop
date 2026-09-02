import os
import sys

# Stop a ``utils/`` (or ``proxy/``, ``ui/``) package in the launch directory
# from shadowing Hermes's own top-level modules.  ``hermes_bootstrap`` lives at
# the repo root (its name can't collide with a user package), so importing it
# before the guard runs is safe.
import hermes_bootstrap

hermes_bootstrap.harden_import_path()

import json
import logging
import signal
import threading
import time
import traceback

from tui_gateway._env import env_float
from tui_gateway._stdin_recovery import handle_spurious_eof

from tui_gateway import server
from tui_gateway.event_replay import replay_epoch
from tui_gateway.server import _CRASH_LOG, dispatch, resolve_skin, write_json
from tui_gateway.transport import TeeTransport

logger = logging.getLogger(__name__)

# Handle for a background MCP discovery thread spawned by THIS module.  Stays
# None when discovery is delegated to the shared owner in hermes_cli.mcp_startup
# (the current path); the wait/in-flight/join helpers consult both owners.
_mcp_discovery_thread = None

# True once ensure_mcp_discovery_started found MCP servers configured and spawned
# discovery through the shared owner.  Lets wait_for_mcp_discovery re-invoke the
# idempotent spawn on later agent builds so the retry-after-zero-connected
# allowance can fire (otherwise a first run that connected nothing latches the
# process MCP-less).  A flag rather than a config re-probe so non-MCP sessions
# never pay the tools.mcp_tool import on the per-agent-build wait path.
_mcp_discovery_enabled = False


def _install_sidecar_publisher() -> None:
    """Mirror every dispatcher emit to the dashboard sidebar via WS.

    Activated by `HERMES_TUI_SIDECAR_URL`, set by the dashboard's ``/api/pty``
    endpoint when a chat tab passes a ``channel`` query param.  Best-effort:
    connect failure or runtime drop falls back to stdio-only.
    """
    url = os.environ.get("HERMES_TUI_SIDECAR_URL")
    if not url:
        return
    from tui_gateway.event_publisher import WsPublisherTransport

    server._stdio_transport = TeeTransport(server._stdio_transport, WsPublisherTransport(url))


# Grace for orderly shutdown (atexit + finalisers) before ``os._exit(0)`` so a
# wedged worker mid-flush can't strand the process.  1s covers the gateway's own
# shutdown work; ``HERMES_TUI_GATEWAY_SHUTDOWN_GRACE_S`` overrides for slower
# environments (a longer grace also means a longer wait on a real deadlock).
_DEFAULT_SHUTDOWN_GRACE_S = 1.0


def _shutdown_grace_seconds() -> float:
    value = env_float("HERMES_TUI_GATEWAY_SHUTDOWN_GRACE_S", _DEFAULT_SHUTDOWN_GRACE_S)
    return value if value > 0 else _DEFAULT_SHUTDOWN_GRACE_S


def _stamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _append_crash_log(header: str, dump=None) -> None:
    """Best-effort ``=== header ===`` entry in the crash log; ``dump(f)`` adds detail."""
    try:
        os.makedirs(os.path.dirname(_CRASH_LOG), exist_ok=True)
        with open(_CRASH_LOG, "a", encoding="utf-8") as f:
            f.write(f"\n=== {header} ===\n")
            if dump is not None:
                dump(f)
    except Exception:
        pass


def _log_signal(signum: int, frame) -> None:
    """Capture WHICH thread and WHERE a termination signal hit us, then exit.

    SIG_DFL for SIGPIPE kills the process silently the instant a background
    thread (TTS, beep, voice status) writes to a stdout the TUI stopped reading,
    leaving no trace in the crash log.  ``sys.exit(0)`` alone used to race the
    worker pool — a thread holding ``_stdout_lock`` mid-flush blocks interpreter
    shutdown indefinitely — so we log all thread stacks, give the process the
    configured grace to drain, and fall back to ``os._exit(0)``.
    """
    # SIGPIPE/SIGHUP don't exist on Windows — only look up attributes present.
    names = {
        int(sig): attr
        for attr in ("SIGPIPE", "SIGTERM", "SIGHUP", "SIGINT", "SIGBREAK")
        if (sig := getattr(signal, attr, None)) is not None
    }
    name = names.get(signum, f"signal {signum}")

    def _dump(f):
        if frame is not None:
            f.write("main-thread stack at signal delivery:\n")
            traceback.print_stack(frame, file=f)
        # All live threads — the signal may have come from a background writer.
        for tid, th in threading._active.items():
            f.write(f"\n--- thread {th.name} (id={tid}) ---\n")
            f.write("".join(traceback.format_stack(sys._current_frames().get(tid))))

    _append_crash_log(f"{name} received · {_stamp()}", _dump)
    print(f"[gateway-signal] {name}", file=sys.stderr, flush=True)

    # ``os._exit`` skips atexit but breaks the mid-flush deadlock; the crash log
    # + stderr line above are the forensic trail.
    timer = threading.Timer(_shutdown_grace_seconds(), lambda: os._exit(0))
    timer.daemon = True
    timer.start()

    # The atexit handler (_shutdown_sessions) can be blocked past the grace
    # window by a worker holding the GIL/_stdout_lock; finalize explicitly so
    # unpersisted messages reach state.db before the hard-exit timer fires.
    try:
        from tui_gateway.server import _shutdown_sessions

        _shutdown_sessions()
    except Exception:
        pass

    # Unwind the main thread so atexit + finalisers run inside the grace window;
    # the daemon timer is the safety net if that unwind hangs.
    sys.exit(0)


def _install_signal(signame, handler):
    """Install a signal handler if legal in this thread and platform.

    signal.signal() raises ValueError outside the main thread; skip silently so a
    worker-thread first import (Desktop build path: server._build does ``from
    tui_gateway.entry import ...``) doesn't abort.  Handlers are process-global,
    so any main-thread import installs them for everyone.  Missing signals
    (Windows: SIGPIPE/SIGHUP) are skipped too.
    """
    if threading.current_thread() is not threading.main_thread():
        return
    sig = getattr(signal, signame, None)
    if sig is None:
        return
    try:
        signal.signal(sig, handler)
    except (ValueError, OSError, RuntimeError):
        pass


# SIGPIPE: ignore, don't exit.  SIG_DFL killed the process silently whenever a
# *background* thread wrote to a pipe the TUI had gone quiet on, even with the
# main thread fine on stdin.  Ignoring lets the write raise BrokenPipeError
# (write_json handles it with a clean sys.exit(0) + _log_exit) so the gateway
# lives as long as the command pipe is readable.  Terminal signals route through
# _log_signal so kills/hangups are diagnosable; SIGBREAK (Windows Ctrl+Break) is
# the weaker SIGHUP equivalent.
_install_signal("SIGPIPE", signal.SIG_IGN)
_install_signal("SIGTERM", _log_signal)
if hasattr(signal, "SIGHUP"):
    _install_signal("SIGHUP", _log_signal)
elif hasattr(signal, "SIGBREAK"):
    _install_signal("SIGBREAK", _log_signal)
_install_signal("SIGINT", signal.SIG_IGN)


def _log_exit(reason: str) -> None:
    """Record why the gateway is shutting down.

    Every exit path (startup/parse-error/response write fail, stdin EOF)
    collapses into a silent sys.exit(0); without this trail the TUI shows
    "gateway exited" with no clue WHICH broken pipe or message triggered it.
    """
    _append_crash_log(f"gateway exit · {_stamp()} · reason={reason}")
    print(f"[gateway-exit] {reason}", file=sys.stderr, flush=True)


def wait_for_mcp_discovery(timeout: "float | None" = None) -> None:
    """Block until background MCP discovery finishes, up to the resolved bound.

    Discovery runs in a daemon thread so a slow/dead server can't freeze
    ``gateway.ready``, but the agent snapshots its tool list ONCE at build time.
    Joining with a bounded timeout before the first build lets already-spawning
    servers land (join returns the instant discovery completes, so no-MCP
    startups pay ~0s) without re-introducing the startup hang.  The bound is
    ``mcp_discovery_timeout`` from config (via ``hermes_cli.mcp_startup``);
    ``timeout`` overrides it.
    """
    thread = _mcp_discovery_thread
    if thread is not None and thread.is_alive():
        try:
            from hermes_cli.mcp_startup import _resolve_discovery_timeout

            bound = _resolve_discovery_timeout(timeout)
        except Exception:
            bound = timeout if timeout is not None else 0.75
        thread.join(timeout=bound)
        return
    # Shared-owner path.  Re-invoke the idempotent spawn first: if the previous
    # run connected zero servers, the retry allowance starts a fresh run instead
    # of leaving the process latched MCP-less.  It runs under the CALLER's
    # profile context (agent build binds the session profile's HERMES_HOME
    # first), so a launch profile without mcp_servers doesn't starve selected
    # profiles.  Gated so non-MCP sessions never pay the tools.mcp_tool import.
    if not _mcp_discovery_enabled:
        return
    try:
        from hermes_cli.mcp_startup import start_background_mcp_discovery

        start_background_mcp_discovery(logger=logger, thread_name="tui-mcp-discovery")
    except Exception:
        logger.debug("TUI MCP discovery retry-spawn failed", exc_info=True)
    try:
        from hermes_cli.mcp_startup import wait_for_mcp_discovery as _startup_wait

        _startup_wait(timeout)
    except Exception:
        pass


def mcp_discovery_in_flight() -> bool:
    """True if ANY background MCP discovery thread is still running.

    The agent-build path uses this to schedule a late tool-snapshot refresh when
    discovery didn't land within the bounded join.  There are two owners by
    surface — the stdio ``hermes --tui`` thread here and the
    ``hermes_cli.mcp_startup`` thread used by desktop/dashboard — and the
    late-refresh scheduler imports this regardless of surface, so it MUST
    consult both or slow MCP servers' tools never surface on desktop.
    """
    thread = _mcp_discovery_thread
    if thread is not None and thread.is_alive():
        return True
    try:
        from hermes_cli.mcp_startup import mcp_discovery_in_flight as _startup_in_flight

        return _startup_in_flight()
    except Exception:
        return False


def join_mcp_discovery(timeout: float | None = None) -> bool:
    """Join both discovery owners; True once neither is alive.

    Unlike ``wait_for_mcp_discovery`` this accepts an unbounded/long wait and
    reports the outcome (for the off-critical-path late-refresh waiter).
    ``timeout`` bounds EACH join: entry thread first, then the shared owner.
    """
    entry_done = True
    thread = _mcp_discovery_thread
    if thread is not None:
        thread.join(timeout=timeout)
        entry_done = not thread.is_alive()
    try:
        from hermes_cli.mcp_startup import join_mcp_discovery as _startup_join

        startup_done = _startup_join(timeout=timeout)
    except Exception:
        startup_done = True
    return entry_done and startup_done


# Spurious stdin-EOF recovery tracker (shared open-file-description O_NONBLOCK flip).
_recovery_times: list[float] = []


def _has_configured_mcp_servers() -> bool:
    """Delegate to the shared native and portable MCP startup gate."""
    from hermes_cli.mcp_startup import _has_configured_mcp_servers as configured

    return configured()


def ensure_mcp_discovery_started() -> None:
    """Start background MCP discovery for the current profile context, once.

    ``main()`` calls this for the stdio path.  WebSocket/Desktop entrypoints skip
    ``main()``, so ``server._start_agent_build`` also calls it AFTER binding the
    session profile's HERMES_HOME — the shared owner captures the caller's
    context-local override into the discovery thread, so discovery reads the
    SELECTED profile's ``mcp_servers``.  Delegating keeps the process-wide start
    lock, retry-after-zero-connected allowance, and interactive-OAuth
    suppression.  Limitation: MCP registration is process-global, so the FIRST
    profile to build an agent wins the discovery slot.
    """
    global _mcp_discovery_enabled

    if not _has_configured_mcp_servers():
        return
    _mcp_discovery_enabled = True
    try:
        from hermes_cli.mcp_startup import start_background_mcp_discovery

        start_background_mcp_discovery(logger=logger, thread_name="tui-mcp-discovery")
    except Exception:
        logger.warning("Background MCP tool discovery failed to start", exc_info=True)


def main():
    _install_sidecar_publisher()

    # Heartbeat row lets the orphan sweep tell "live but idle backend" from
    # "truly orphaned"; must run BEFORE the sweep so it sees our row.
    try:
        server._start_backend_heartbeat_refresher()
    except Exception:
        logger.warning("backend heartbeat refresher start failed", exc_info=True)

    # One-time sweep of rows orphaned by a previous gateway process (the
    # in-process reap timer dies with the process).  Once-per-process and
    # config-gated, so the handle_ws call site is a no-op when this ran.
    try:
        server._schedule_startup_orphan_sweep()
    except Exception:
        logger.warning("startup orphan sweep scheduling failed", exc_info=True)

    # Backgrounded so a dead MCP server (~7s of connect retries) can't freeze
    # startup; _make_agent briefly joins it (wait_for_mcp_discovery).  The config
    # gate inside keeps the ~200ms MCP SDK import off the no-mcp_servers path.
    ensure_mcp_discovery_started()

    if not write_json({
        "jsonrpc": "2.0",
        "method": "event",
        "params": {
            "type": "gateway.ready",
            # change_events: clients demote legacy polls (see tui_gateway/ws.py).
            # replay_epoch: WS restart detection; the stdio TUI ignores it.
            "payload": {
                "skin": resolve_skin(),
                "change_events": True,
                "replay_epoch": replay_epoch(),
            },
        },
    }):
        _log_exit("startup write failed (broken stdout pipe before first event)")
        sys.exit(0)

    # Live-apply skins Hermes activates mid-conversation.
    server._ensure_skin_watcher()

    # Warm the /model picker's provider-models cache during this idle window
    # (mirrors the classic CLI loop); otherwise the first /model open blocks on
    # serial /v1/models fetches.  Fire-and-forget, once-per-process.
    try:
        from hermes_cli.model_switch import prewarm_picker_cache_async
        prewarm_picker_cache_async()
    except Exception:
        logger.debug("picker cache prewarm (tui) failed to start", exc_info=True)

    while True:
        raw = sys.stdin.readline()
        if not raw:
            # Spurious (child flipped O_NONBLOCK on the shared description) or genuine EOF?
            if not handle_spurious_eof(_recovery_times, _log_exit):
                break
            continue

        line = raw.strip()
        if not line:
            continue

        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            if not write_json({"jsonrpc": "2.0", "error": {"code": -32700, "message": "parse error"}, "id": None}):
                _log_exit("parse-error-response write failed (broken stdout pipe)")
                sys.exit(0)
            continue

        method = req.get("method") if isinstance(req, dict) else None
        resp = dispatch(req)
        if resp is not None and not write_json(resp):
            _log_exit(f"response write failed for method={method!r} (broken stdout pipe)")
            sys.exit(0)


if __name__ == "__main__":
    main()
