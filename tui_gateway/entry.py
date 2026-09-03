import os
import sys

# Stop a ``utils/`` (or ``proxy/``, ``ui/``) package in the launch directory from
# shadowing Hermes's own top-level modules. ``hermes_bootstrap`` lives at the repo
# root (its name can't collide with a user package), so importing it first is safe.
import hermes_bootstrap

hermes_bootstrap.harden_import_path()

import json
import logging
import signal
import threading
import time
import traceback
from contextlib import suppress

from tui_gateway._env import env_float
from tui_gateway._stdin_recovery import handle_spurious_eof

from tui_gateway import server
from tui_gateway.event_replay import replay_epoch
from tui_gateway.server import _CRASH_LOG, dispatch, resolve_skin, write_json
from tui_gateway.transport import TeeTransport

logger = logging.getLogger(__name__)

# Discovery thread spawned by THIS module; None when delegated to the shared owner in
# hermes_cli.mcp_startup (current path). The wait/in-flight/join helpers consult both.
_mcp_discovery_thread = None
# Set once MCP servers are found configured, so wait_for_mcp_discovery can re-invoke
# the idempotent spawn on later builds (retry-after-zero-connected) without a config
# re-probe — non-MCP sessions never pay the tools.mcp_tool import per build.
_mcp_discovery_enabled = False


def _install_sidecar_publisher() -> None:
    """Mirror every dispatcher emit to the dashboard sidebar via WS.

    Activated by `HERMES_TUI_SIDECAR_URL` (set by the dashboard's ``/api/pty``
    endpoint). Best-effort: connect failure or runtime drop falls back to stdio-only.
    """
    url = os.environ.get("HERMES_TUI_SIDECAR_URL")
    if not url:
        return
    from tui_gateway.event_publisher import WsPublisherTransport

    server._stdio_transport = TeeTransport(server._stdio_transport, WsPublisherTransport(url))


# Grace for orderly shutdown before ``os._exit(0)`` so a worker wedged mid-flush can't
# strand the process; ``HERMES_TUI_GATEWAY_SHUTDOWN_GRACE_S`` overrides (a longer grace
# also means a longer wait on a real deadlock).
_DEFAULT_SHUTDOWN_GRACE_S = 1.0


def _shutdown_grace_seconds() -> float:
    value = env_float("HERMES_TUI_GATEWAY_SHUTDOWN_GRACE_S", _DEFAULT_SHUTDOWN_GRACE_S)
    return value if value > 0 else _DEFAULT_SHUTDOWN_GRACE_S


def _stamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _mcp_startup_call(name: str, *args, default=None, **kwargs):
    """Call ``hermes_cli.mcp_startup.<name>`` (lazy import); ``default`` on any failure."""
    try:
        from hermes_cli import mcp_startup

        return getattr(mcp_startup, name)(*args, **kwargs)
    except Exception:
        return default


def _append_crash_log(header: str, dump=None) -> None:
    """Best-effort ``=== header ===`` entry in the crash log; ``dump(f)`` adds detail."""
    with suppress(Exception):
        os.makedirs(os.path.dirname(_CRASH_LOG), exist_ok=True)
        with open(_CRASH_LOG, "a", encoding="utf-8") as f:
            f.write(f"\n=== {header} ===\n")
            if dump is not None:
                dump(f)


def _log_signal(signum: int, frame) -> None:
    """Capture WHICH thread and WHERE a termination signal hit us, then exit.

    ``sys.exit(0)`` alone raced the worker pool — a thread holding ``_stdout_lock``
    mid-flush blocks interpreter shutdown indefinitely — so log all thread stacks,
    give the configured grace to drain, then ``os._exit(0)``.
    """
    # SIGPIPE/SIGHUP don't exist on Windows — only look up attributes present.
    names = {int(sig): attr for attr in ("SIGPIPE", "SIGTERM", "SIGHUP", "SIGINT", "SIGBREAK")
             if (sig := getattr(signal, attr, None)) is not None}
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
    # The atexit handler (_shutdown_sessions) can be blocked past the grace window
    # by a worker holding the GIL/_stdout_lock; finalize explicitly so unpersisted
    # messages reach state.db before the hard-exit timer fires.
    with suppress(Exception):
        from tui_gateway.server import _shutdown_sessions

        _shutdown_sessions()

    # Unwind the main thread so atexit + finalisers run inside the grace window;
    # the daemon timer is the safety net if that unwind hangs.
    sys.exit(0)


def _install_signal(signame, handler):
    """Install a signal handler if legal in this thread and platform.

    signal.signal() raises ValueError outside the main thread; skip silently so a
    worker-thread first import (Desktop build path: server._build imports entry)
    doesn't abort. Handlers are process-global, so any main-thread import installs
    them for everyone. Missing signals (Windows: SIGPIPE/SIGHUP) are skipped too.
    """
    sig = getattr(signal, signame, None)
    if sig is None or threading.current_thread() is not threading.main_thread():
        return
    # Off the main thread despite the check, or handler rejected by the platform.
    with suppress(ValueError, OSError, RuntimeError):
        signal.signal(sig, handler)


# SIGPIPE: ignore, don't exit. SIG_DFL killed the process silently whenever a
# *background* thread (TTS, beep, voice status) wrote to a pipe the TUI had gone
# quiet on. Ignoring lets the write raise BrokenPipeError (write_json handles it with
# a clean sys.exit(0) + _log_exit) so the gateway lives as long as the command pipe
# is readable. Terminal signals route through _log_signal so kills/hangups are
# diagnosable; SIGBREAK (Windows Ctrl+Break) is the weaker SIGHUP.
_install_signal("SIGPIPE", signal.SIG_IGN)
_install_signal("SIGTERM", _log_signal)
if hasattr(signal, "SIGHUP"):
    _install_signal("SIGHUP", _log_signal)
elif hasattr(signal, "SIGBREAK"):
    _install_signal("SIGBREAK", _log_signal)
_install_signal("SIGINT", signal.SIG_IGN)


def _log_exit(reason: str) -> None:
    """Record why the gateway is shutting down: every exit path collapses into a
    silent sys.exit(0), and without this trail the TUI shows "gateway exited" with
    no clue WHICH broken pipe or message triggered it."""
    _append_crash_log(f"gateway exit · {_stamp()} · reason={reason}")
    print(f"[gateway-exit] {reason}", file=sys.stderr, flush=True)


def wait_for_mcp_discovery(timeout: "float | None" = None) -> None:
    """Block until background MCP discovery finishes, up to the resolved bound.

    The agent snapshots its tool list ONCE at build time, so a bounded join before
    the first build lets already-spawning servers land (no-MCP startups pay ~0s)
    without re-introducing the startup hang. Bound: ``mcp_discovery_timeout`` from
    config; ``timeout`` overrides it.
    """
    thread = _mcp_discovery_thread
    if thread is not None and thread.is_alive():
        fallback = timeout if timeout is not None else 0.75
        thread.join(timeout=_mcp_startup_call("_resolve_discovery_timeout", timeout, default=fallback))
        return
    # Shared-owner path. Re-invoke the idempotent spawn first so a previous
    # zero-connected run gets its retry instead of latching the process MCP-less.
    # It runs under the CALLER's profile context (agent build binds the session
    # profile's HERMES_HOME first), so a launch profile without mcp_servers doesn't
    # starve selected profiles. Gated so non-MCP sessions skip the mcp_tool import.
    if not _mcp_discovery_enabled:
        return
    try:
        from hermes_cli.mcp_startup import start_background_mcp_discovery

        start_background_mcp_discovery(logger=logger, thread_name="tui-mcp-discovery")
    except Exception:
        logger.debug("TUI MCP discovery retry-spawn failed", exc_info=True)
    _mcp_startup_call("wait_for_mcp_discovery", timeout)


def mcp_discovery_in_flight() -> bool:
    """True if ANY background MCP discovery thread is still running. Two owners by
    surface (stdio thread here, ``hermes_cli.mcp_startup`` for desktop/dashboard);
    the late-refresh scheduler calls this regardless of surface, so it MUST consult
    both or slow MCP servers' tools never surface on desktop."""
    thread = _mcp_discovery_thread
    if thread is not None and thread.is_alive():
        return True
    return _mcp_startup_call("mcp_discovery_in_flight", default=False)


def join_mcp_discovery(timeout: float | None = None) -> bool:
    """Join both discovery owners; True once neither is alive. Unlike
    ``wait_for_mcp_discovery`` this accepts an unbounded wait (off-critical-path
    late-refresh waiter); ``timeout`` bounds EACH join, entry thread first."""
    entry_done = True
    thread = _mcp_discovery_thread
    if thread is not None:
        thread.join(timeout=timeout)
        entry_done = not thread.is_alive()
    return entry_done and _mcp_startup_call("join_mcp_discovery", timeout=timeout, default=True)


# Spurious stdin-EOF recovery tracker (shared open-file-description O_NONBLOCK flip).
_recovery_times: list[float] = []


def _has_configured_mcp_servers() -> bool:
    """Delegate to the shared native and portable MCP startup gate."""
    from hermes_cli.mcp_startup import _has_configured_mcp_servers as configured

    return configured()


def ensure_mcp_discovery_started() -> None:
    """Start background MCP discovery for the current profile context, once.

    ``main()`` calls this for stdio; WS/Desktop skip ``main()``, so
    ``server._start_agent_build`` also calls it AFTER binding the session profile's
    HERMES_HOME (the shared owner captures that override, so discovery reads the
    SELECTED profile's ``mcp_servers``). Delegating keeps the process-wide start lock,
    retry-after-zero-connected allowance and interactive-OAuth suppression. MCP
    registration is process-global: the FIRST profile to build an agent wins.
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


def _write_or_exit(payload: dict, reason: str) -> None:
    if not write_json(payload):
        _log_exit(reason)
        sys.exit(0)


def main():
    _install_sidecar_publisher()

    # Heartbeat row lets the orphan sweep tell "live but idle backend" from "truly
    # orphaned"; must run BEFORE the sweep so it sees our row. The sweep itself is
    # once-per-process and config-gated (the handle_ws call site becomes a no-op).
    for start, what in (
        (server._start_backend_heartbeat_refresher, "backend heartbeat refresher start"),
        (server._schedule_startup_orphan_sweep, "startup orphan sweep scheduling"),
    ):
        try:
            start()
        except Exception:
            logger.warning("%s failed", what, exc_info=True)

    # Backgrounded so a dead MCP server (~7s of connect retries) can't freeze
    # startup; _make_agent briefly joins it (wait_for_mcp_discovery). The config
    # gate inside keeps the ~200ms MCP SDK import off the no-mcp_servers path.
    ensure_mcp_discovery_started()

    _write_or_exit({
        "jsonrpc": "2.0",
        "method": "event",
        "params": {
            "type": "gateway.ready",
            # change_events: clients demote legacy polls (see tui_gateway/ws.py).
            # replay_epoch: WS restart detection; the stdio TUI ignores it.
            "payload": {
                "skin": resolve_skin(), "change_events": True, "replay_epoch": replay_epoch(),
            },
        },
    }, "startup write failed (broken stdout pipe before first event)")

    # Live-apply skins Hermes activates mid-conversation.
    server._ensure_skin_watcher()

    # Warm the /model picker's provider-models cache during this idle window
    # (mirrors the classic CLI loop); otherwise the first /model open blocks on
    # serial /v1/models fetches. Fire-and-forget, once-per-process.
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
            _write_or_exit(
                {"jsonrpc": "2.0", "error": {"code": -32700, "message": "parse error"}, "id": None},
                "parse-error-response write failed (broken stdout pipe)")
            continue

        method = req.get("method") if isinstance(req, dict) else None
        resp = dispatch(req)
        if resp is not None:
            _write_or_exit(
                resp, f"response write failed for method={method!r} (broken stdout pipe)")


if __name__ == "__main__":
    main()
