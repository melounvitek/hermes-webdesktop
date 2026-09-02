"""Serve-process lifecycle helpers: parent start markers and death watchdog, port-conflict preflight, ready-file/sentinel announcement, browser auto-open, forwarded-IP resolution.

Split out of ``hermes_cli.web_server``; every externally used name is re-imported
there, so ``web_server.<name>`` keeps resolving (and monkeypatching) as before.
Helpers that tests patch on ``web_server`` are reached lazily through it.
"""

import logging
import ipaddress
import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Optional

# Same logger the code used before extraction (record parity).
_log = logging.getLogger("hermes_cli.web_server")


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
    from hermes_cli.web_server import _process_start_marker
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
    from hermes_cli.web_server import _write_machine_sentinel_line
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
