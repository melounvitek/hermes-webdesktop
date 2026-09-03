"""Gateway subcommand for hermes CLI.

Handles: hermes gateway [run|start|stop|restart|status|install|uninstall|setup]
"""

import asyncio
import contextlib
from hermes_cli.cli_output import line_input
import json
import logging
import os
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path

# UV's bundled Python ships a minimal PATH; ensure launchctl/systemctl are discoverable.
if os.name == "posix":
    _sys_dirs = {"/bin", "/usr/bin", "/usr/sbin", "/sbin"}
    _path_dirs = set(os.environ.get("PATH", "").split(os.pathsep))
    _missing = _sys_dirs - _path_dirs
    if _missing:
        os.environ["PATH"] = os.environ.get("PATH", "") + os.pathsep + os.pathsep.join(sorted(_missing))

PROJECT_ROOT = Path(__file__).parent.parent.resolve()

from gateway.config import coerce_systemd_watchdog_seconds, load_gateway_config
from gateway.status import terminate_pid
from gateway.restart import (
    DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT,
    EXTERNAL_GATEWAY_SUPERVISOR_ENV,
    GATEWAY_FATAL_CONFIG_EXIT_CODE,
    GATEWAY_SERVICE_RESTART_EXIT_CODE,
    is_gateway_supervisor_process,
    parse_cron_drain_timeout,
    parse_restart_after_turn_timeout,
    parse_restart_drain_timeout,
    resolve_restart_exit_wait_budget,
    resolve_systemd_timeout_stop_sec,
)
from hermes_cli.config import (
    get_env_value,
    get_hermes_home,
    is_managed,
    managed_error,
    read_raw_config,
    save_env_value,
    write_platform_config_field,
)

# display_hermes_home is imported lazily: hermes_constants may be a cached pre-update version.
from hermes_cli.setup import (
    print_header,
    print_info,
    print_success,
    print_warning,
    print_error,
    prompt,
    prompt_choice,
    prompt_yes_no,
)
from hermes_cli.colors import Colors, color

logger = logging.getLogger(__name__)

# Shared ``subprocess.run`` kwargs for text-mode probes (stdout/stderr captured, decode-tolerant).
_CAPTURE_TEXT = dict(capture_output=True, text=True, encoding="utf-8", errors="replace")

# =============================================================================
# Process Management (for manual gateway runs)
# =============================================================================


@dataclass(frozen=True)
class GatewayRuntimeSnapshot:
    manager: str
    service_installed: bool = False
    service_running: bool = False
    gateway_pids: tuple[int, ...] = ()
    service_scope: str | None = None

    @property
    def running(self) -> bool:
        return self.service_running or bool(self.gateway_pids)

    @property
    def has_process_service_mismatch(self) -> bool:
        return self.service_installed and self.running and not self.service_running


@dataclass(frozen=True)
class ProfileGatewayProcess:
    profile: str
    path: Path
    pid: int
    create_time: float = 0.0


@dataclass(frozen=True)
class WindowsGatewayService:
    """A real Windows service supervising a profile gateway process tree."""

    name: str
    profile: str
    service_pid: int
    gateway_pid: int
    descendant_pids: frozenset[int]
    descendant_identities: tuple[tuple[int, float], ...]
    service_create_time: float = 0.0
    gateway_create_time: float = 0.0


def _get_service_pids(all_profiles: bool = False) -> set:
    """Return PIDs managed by systemd/launchd gateway services (excluded from stale-process sweeps).

    Relies on the service manager committing the new PID before the restart command returns.
    Default scope covers only the current profile's unit/label; ``all_profiles`` widens to the
    whole ``hermes-gateway*`` / ``ai.hermes.gateway*`` fleet so the update path and orphan reaper
    never misclassify a sibling profile's service gateway as a manual process and kill it.
    """
    pids: set = set()

    # --- systemd (Linux): user and system scopes ---
    if supports_systemd_services():
        pattern = "hermes-gateway*" if all_profiles else get_service_name()
        for scope_args in [["systemctl", "--user"], ["systemctl"]]:
            try:
                result = subprocess.run(
                    scope_args
                    + ["list-units", pattern, "--plain", "--no-legend", "--no-pager"],
                    timeout=5,
                    **_CAPTURE_TEXT,
                )
                for line in result.stdout.strip().splitlines():
                    parts = line.split()
                    if not parts or not parts[0].endswith(".service"):
                        continue
                    svc = parts[0]
                    try:
                        show = subprocess.run(
                            scope_args + ["show", svc, "--property=MainPID", "--value"],
                            timeout=5,
                            **_CAPTURE_TEXT,
                        )
                        pid = int(show.stdout.strip())
                        if pid > 0:
                            pids.add(pid)
                    except (ValueError, subprocess.TimeoutExpired):
                        pass
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass

    # --- launchd (macOS) ---
    if is_macos():
        labels = {get_launchd_label()}
        if all_profiles:
            # Whole fleet, mirroring the systemd ``hermes-gateway*`` glob above.
            labels.update(launchd_gateway_labels_for_install())
        for label in sorted(labels):
            try:
                _domain, pid = _locate_launchd_gateway_service(label)
            except subprocess.TimeoutExpired:
                continue
            if pid is not None and pid > 0:
                pids.add(pid)
        if all_profiles:
            # Prefix scan also catches ai.hermes.gateway* agents the label derivation can't map
            # (renamed profiles, other installs). Over-inclusion is safe: PIDs are only protected.
            try:
                result = subprocess.run(["launchctl", "list"], timeout=5, **_CAPTURE_TEXT)
                if result.returncode == 0:
                    for line in result.stdout.strip().splitlines():
                        parts = line.split()
                        if len(parts) >= 3 and parts[-1].startswith("ai.hermes.gateway"):
                            try:
                                pid = int(parts[0])
                                if pid > 0:
                                    pids.add(pid)
                            except ValueError:
                                pass
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass

    return pids


def _get_parent_pid(pid: int) -> int | None:
    """Return the parent PID for ``pid``, or ``None``. psutil first (works on Windows, where ``ps`` doesn't)."""
    if pid <= 1:
        return None
    try:
        import psutil  # type: ignore

        return psutil.Process(pid).ppid() or None
    except ImportError:
        pass
    except Exception:
        return None
    # ps fallback, POSIX only: Git Bash's ps.exe would flash a console from the windowless backend.
    if is_windows():
        return None
    if not shutil.which("ps"):
        return None
    try:
        result = subprocess.run(["ps", "-o", "ppid=", "-p", str(pid)], timeout=5, **_CAPTURE_TEXT)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    raw = result.stdout.strip()
    if result.returncode != 0 or not raw:
        return None
    try:
        parent_pid = int(raw.splitlines()[-1].strip())
    except ValueError:
        return None
    return parent_pid if parent_pid > 0 else None


def _is_pid_ancestor_of_current_process(target_pid: int) -> bool:
    """Return True when ``target_pid`` is this process or one of its ancestors."""
    if target_pid <= 0:
        return False

    pid = os.getpid()
    seen: set[int] = set()
    while pid and pid not in seen:
        if pid == target_pid:
            return True
        seen.add(pid)
        pid = _get_parent_pid(pid) or 0
    return False


def _request_gateway_self_restart(pid: int) -> bool:
    """Ask a running gateway ancestor to restart itself asynchronously."""
    if not hasattr(signal, "SIGUSR1"):
        return False
    if not _is_pid_ancestor_of_current_process(pid):
        return False
    try:
        os.kill(pid, signal.SIGUSR1)  # windows-footgun: ok — POSIX signal, guarded by hasattr(signal, 'SIGUSR1') above
    except (ProcessLookupError, PermissionError, OSError):
        return False
    return True


def _graceful_restart_via_sigusr1(pid: int, drain_timeout: float) -> bool:
    """Send SIGUSR1 (drain-aware restart) to a gateway PID and wait for it to exit.

    gateway/run.py maps SIGUSR1 to ``request_restart(via_service=True)``: refuse new turns, wait
    for in-flight work, ``stop()``, exit; systemd/launchd then relaunch. ``drain_timeout`` must
    cover the after-turn wait plus the drain — pass ``resolve_restart_exit_wait_budget(...)``.
    Returns False if the signal couldn't be sent or the process outlived the timeout.
    """
    if not hasattr(signal, "SIGUSR1"):
        return False
    if pid <= 0:
        return False
    try:
        os.kill(pid, signal.SIGUSR1)  # windows-footgun: ok — POSIX signal, guarded by hasattr(signal, 'SIGUSR1') above
    except ProcessLookupError:
        return True
    except (PermissionError, OSError):
        return False

    return _wait_for_pid_exit(pid, max(drain_timeout, 1.0))


def _wait_for_pid_exit(pid: int, timeout: float) -> bool:
    """Wait up to ``timeout``s for ``pid`` to exit; True once gone, False on timeout.

    ``launchctl bootstrap`` fails with EIO while the previous instance is still draining, so
    teardown callers must wait for the real exit before re-bootstrapping.
    """
    if pid <= 0:
        return True

    import time as _time

    # ``os.kill(pid, 0)`` hard-kills on Windows (TerminateProcess); use _pid_exists instead.
    from gateway.status import _pid_exists

    deadline = _time.monotonic() + max(timeout, 0.0)
    while True:
        if not _pid_exists(pid):
            return True
        if _time.monotonic() >= deadline:
            return False
        _time.sleep(0.5)


# --- Wedged-gateway detection + bounded escalation ---------------------------
#
# A gateway whose asyncio loop is stalled cannot handle SIGTERM/SIGUSR1, so the drain wait burns
# its full budget and `hermes update` can deadlock. Two witnesses classify the loop BEFORE any
# drain wait: the heartbeat file ``state/gateway.heartbeat`` (rewritten every 30s, but on a thread
# — so freshness/staleness alone is not proof) and the loop-tick socket
# ``state/gateway.loop-tick.<pid>.sock``, answered by the loop itself; the payload records whether
# the socket is armed (``loop_tick_socket``).
#
# - ``alive``   — socket answered, or file fresh and not contradicted. Normal graceful drain,
#                 which honours the in-flight cron drain floor.
# - ``wedged``  — heartbeat is this PID's, stale past several beats, AND the armed socket stays
#                 silent across ``tick_strikes`` consecutive misses. Only then may callers
#                 escalate via ``_escalate_wedged_gateway``; one silent probe is never authority.
# - ``unknown`` — no/unreadable heartbeat, PID mismatch, or witness conflict. Treated as alive:
#                 never escalate on ambiguity.
#
# Legacy payloads (no ``loop_tick_socket`` flag) wrote on-loop, so staleness alone remains proof.

GATEWAY_LOOP_ALIVE = "alive"
GATEWAY_LOOP_WEDGED = "wedged"
GATEWAY_LOOP_UNKNOWN = "unknown"

# 3 missed 30s beats (gateway.shutdown_watchdog.DEFAULT_HEARTBEAT_INTERVAL_S): decisive, not one slow write.
DEFAULT_LOOP_LIVENESS_STALE_AFTER_S = 90.0

# Sentinel for "the producer never wrote the witness flag" (legacy payload).
_LOOP_TICK_ABSENT = object()


def _probe_loop_tick_socket(pid: int, home: Path | None, timeout: float = 1.0) -> bool | None:
    """Ping the loop-tick witness socket: True answered, False node present but silent, None no node (not evidence)."""
    try:
        from gateway.shutdown_watchdog import get_loop_tick_socket_path

        path = get_loop_tick_socket_path(home, pid)
        if not path.is_socket():
            return None
    except Exception:
        return None
    return _ping_loop_tick_witness(socket.AF_UNIX, str(path), timeout)


def _ping_loop_tick_witness(family: int, address, timeout: float) -> bool:
    """Connect to a loop-tick witness and expect one byte ``"1"``; False on refusal/timeout/any error."""
    sock = None
    try:
        sock = socket.socket(family, socket.SOCK_STREAM)
        sock.settimeout(max(float(timeout), 0.0))
        sock.connect(address)
        return sock.recv(1) == b"1"
    except Exception:
        return False
    finally:
        if sock is not None:
            with contextlib.suppress(Exception):
                sock.close()


def _probe_loop_tick_tcp(port: int, timeout: float = 1.0) -> bool | None:
    """TCP-loopback variant of the tick probe for Windows (no AF_UNIX in asyncio); same semantics, None
    on invalid port."""
    try:
        port_num = int(port)
        if port_num <= 0 or port_num > 65535:
            return None
    except (TypeError, ValueError):
        return None
    return _ping_loop_tick_witness(socket.AF_INET, ("127.0.0.1", port_num), timeout)


def _probe_loop_tick_socket_sustained(
    pid: int,
    home: Path | None,
    *,
    timeout: float = 1.0,
    strikes: int = 3,
    gap_s: float = 0.2,
    tcp_port: int | None = None,
) -> bool | None:
    """Probe the tick socket up to ``strikes`` times, ``gap_s`` apart, until a reply.

    One silent probe is not destructive evidence (a transient synchronous stall can outlast one
    recv timeout). True: some attempt answered. False: a node stayed silent the whole window.
    None: no socket node on some attempt (vanished / legacy producer) — not evidence.
    """
    total = max(int(strikes), 0)
    for attempt in range(total):
        if tcp_port is not None:
            result = _probe_loop_tick_tcp(tcp_port, timeout=timeout)
        else:
            result = _probe_loop_tick_socket(pid, home, timeout=timeout)
        if result is True:
            return True
        if result is None:
            # No node: ambiguity, never a wedge — absence is not a miss.
            return None
        if attempt < total - 1 and gap_s > 0:
            time.sleep(gap_s)
    return False



def probe_gateway_loop_liveness(
    pid: int,
    *,
    stale_after: float = DEFAULT_LOOP_LIVENESS_STALE_AFTER_S,
    home: Path | None = None,
    tick_timeout: float = 1.0,
    tick_strikes: int = 3,
    tick_gap_s: float = 0.2,
) -> str:
    """Classify a gateway PID's event loop as alive / wedged / unknown (see block comment above).

    A stale heartbeat is ``wedged`` only when the payload declares the tick socket armed AND the
    socket stays silent across ``tick_strikes`` consecutive misses; any answer is ``alive``; any
    conflict or ambiguity is ``unknown`` so callers keep the graceful-drain path.
    """
    try:
        stale_budget = max(float(stale_after), 0.0)
    except (TypeError, ValueError):
        stale_budget = DEFAULT_LOOP_LIVENESS_STALE_AFTER_S
    try:
        from gateway.shutdown_watchdog import get_loop_heartbeat_path

        path = get_loop_heartbeat_path(home)
        mtime = path.stat().st_mtime
        payload = json.loads(path.read_text(encoding="utf-8"))
        heartbeat_pid = int(payload.get("pid", 0))
    except Exception:
        return GATEWAY_LOOP_UNKNOWN
    if heartbeat_pid <= 0 or int(pid) <= 0 or heartbeat_pid != int(pid):
        # Heartbeat is not this process's (old version, starting up, stale file): not evidence.
        return GATEWAY_LOOP_UNKNOWN

    # TCP loopback witness (Windows) takes priority when published; else the AF_UNIX socket.
    tcp_port = payload.get("loop_tick_tcp_port")
    try:
        tcp_port_int = int(tcp_port) if tcp_port is not None else None
    except (TypeError, ValueError):
        tcp_port_int = None

    if tcp_port_int is not None and tcp_port_int > 0:
        witness = _probe_loop_tick_tcp(tcp_port_int, timeout=tick_timeout)
        tick_armed = True
    else:
        witness = _probe_loop_tick_socket(pid, home, timeout=tick_timeout)
        tick_armed = payload.get("loop_tick_socket", _LOOP_TICK_ABSENT)
    if witness is True:
        # Loop answered: a stale file is a stalled write, not a wedge.
        return GATEWAY_LOOP_ALIVE
    age = time.time() - mtime
    if age <= stale_budget:
        if witness is False:
            # Fresh file but silent loop: an off-loop write can land after the loop froze.
            return GATEWAY_LOOP_UNKNOWN
        return GATEWAY_LOOP_ALIVE

    # Stale past the budget; the verdict depends on what the producer promised about its witness.
    if tick_armed is _LOOP_TICK_ABSENT:
        # Legacy on-loop writer: staleness proves the loop stopped scheduling.
        return GATEWAY_LOOP_WEDGED
    if tick_armed is not True:
        # Witness could not be armed (bind failed); off-loop write means staleness is not proof.
        return GATEWAY_LOOP_UNKNOWN
    if witness is False:
        # First miss. The probe above is miss #1, so ``tick_strikes - 1`` more attempts follow.
        sustained = _probe_loop_tick_socket_sustained(
            pid,
            home,
            timeout=tick_timeout,
            strikes=tick_strikes - 1,
            gap_s=tick_gap_s,
            tcp_port=tcp_port_int,
        )
        if sustained is False:
            return GATEWAY_LOOP_WEDGED
        if sustained is True:
            # Transient stall, not a wedge.
            return GATEWAY_LOOP_ALIVE
        # Witness vanished mid-window: ambiguity — never kill on it.
        return GATEWAY_LOOP_UNKNOWN
    # Armed but unreachable socket: ambiguity — never kill on it.
    return GATEWAY_LOOP_UNKNOWN


def _escalate_wedged_gateway(pid: int, *, term_grace: float = 5.0, kill_wait: float = 5.0) -> bool:
    """Bounded stop (SIGTERM, ``term_grace``, SIGKILL, ``kill_wait``) for a provably dead loop.

    Callers MUST have classified the gateway ``GATEWAY_LOOP_WEDGED`` first: escalating a merely
    busy gateway bypasses the cron drain floor and SIGKILLs live work. True once the PID is gone.
    """
    from gateway.status import get_process_start_time

    expected_start_time = get_process_start_time(pid)
    try:
        terminate_pid(pid, force=False)
    except (ProcessLookupError, PermissionError, OSError):
        return _wait_for_pid_exit(pid, 1.0)
    if _wait_for_pid_exit(pid, max(float(term_grace), 0.0)):
        return True
    try:
        terminate_pid(pid, force=True, expected_start_time=expected_start_time)
        print(f"⚠ Gateway PID {pid} unresponsive to SIGTERM; sent SIGKILL")
    except (ProcessLookupError, PermissionError, OSError):
        pass
    return _wait_for_pid_exit(pid, max(float(kill_wait), 0.0))


def _get_ancestor_pids() -> set[int]:
    """PIDs of this process and its ancestors, so scans never count the invoking ``hermes`` CLI as a gateway."""
    ancestors: set[int] = set()
    pid = os.getpid()
    for _ in range(64):
        ancestors.add(pid)
        parent = _get_parent_pid(pid)
        if parent is None or parent <= 0 or parent in ancestors:
            break
        pid = parent
    return ancestors


def _append_unique_pid(pids: list[int], pid: int | None, exclude_pids: set[int]) -> None:
    if pid is None or pid <= 0:
        return
    if pid == os.getpid() or pid in exclude_pids or pid in pids:
        return
    pids.append(pid)


def _scan_gateway_pids(
    exclude_pids: set[int],
    all_profiles: bool = False,
    include_restart_managers: bool = False,
) -> list[int]:
    """Best-effort process-table scan for gateway PIDs (backs up a stale/missing PID file; ``--all`` sweeps)."""
    exclude_pids = exclude_pids | _get_ancestor_pids()
    pids: list[int] = []
    # Strict matcher shared with gateway.status: requires a real ``gateway run`` argv, so
    # ``gateway status``/``dashboard`` siblings and ``python -m tui_gateway`` don't match.
    from gateway.status import looks_like_gateway_command_line, looks_like_gateway_runtime_command_line
    current_home = str(get_hermes_home().resolve())
    # Forward slashes on both sides of the HERMES_HOME= match (mirrors gateway.status).
    current_home_lc = current_home.lower().replace("\\", "/")
    current_profile_arg = _profile_arg(current_home)
    current_profile_name = (current_profile_arg.split()[-1] if current_profile_arg else "")
    current_profile_name_lc = current_profile_name.lower()

    def _matches_current_profile(command: str) -> bool:
        command_lc = command.lower().replace("\\", "/")
        if current_profile_name:
            return (
                f"--profile {current_profile_name_lc}" in command_lc
                or f"-p {current_profile_name_lc}" in command_lc
                or f"hermes_home={current_home_lc}" in command_lc
            )

        # Default profile: accept unless argv advertises another profile. HERMES_HOME may come via
        # env (invisible to wmic/CIM), so only a non-matching explicit HERMES_HOME= disqualifies.
        if "--profile " in command_lc or " -p " in command_lc:
            return False
        return not ("hermes_home=" in command_lc and f"hermes_home={current_home_lc}" not in command_lc)

    def _matches_gateway_runtime(command: str) -> bool:
        if looks_like_gateway_command_line(command):
            return True
        return include_restart_managers and looks_like_gateway_runtime_command_line(command)

    def _consider(pid: int, command: str) -> None:
        if _matches_gateway_runtime(command) and (all_profiles or _matches_current_profile(command)):
            _append_unique_pid(pids, pid, exclude_pids)

    try:
        if is_windows():
            listing = _windows_process_listing()
            if listing is None:
                return []
            for pid, command in _iter_windows_list_processes(listing):
                _consider(pid, command)
        else:
            # /proc first (Docker without procps), then `ps -Aww`.
            _found_via_proc = False
            if os.path.isdir("/proc"):
                try:
                    my_pid = os.getpid()
                    for entry in os.listdir("/proc"):
                        if not entry.isdigit():
                            continue
                        pid = int(entry)
                        if pid == my_pid or pid in exclude_pids:
                            continue
                        try:
                            with open(f"/proc/{pid}/cmdline", "rb") as _f:
                                cmdline = _f.read().decode("utf-8", errors="replace")
                            _consider(pid, cmdline.replace("\x00", " "))
                        except (OSError, PermissionError):
                            continue
                    _found_via_proc = True
                except Exception:
                    pass

            if not _found_via_proc:
                # ``-Aww`` not ``-A eww``: BSD/macOS ps rejects ``e``; ``-ww`` = unlimited width.
                result = subprocess.run(["ps", "-Aww", "-o", "pid=,command="], timeout=10, **_CAPTURE_TEXT)
                if result.returncode != 0:
                    return []
                for line in result.stdout.split("\n"):
                    parsed = _parse_ps_line(line)
                    if parsed is not None:
                        _consider(*parsed)
    except (OSError, subprocess.TimeoutExpired):
        return []

    # Windows: a venv ``pythonw.exe`` is a launcher stub that spawns the base Python with the same
    # command line, so each gateway yields two matched PIDs. Drop a matched PID that parents another.
    if is_windows() and len(pids) > 1:
        pids = _filter_venv_launcher_stubs(pids)

    return pids


def _parse_ps_line(line: str) -> tuple[int, str] | None:
    """``(pid, command)`` from one ``ps -o pid=,command=`` line; also accepts ``ps aux`` rows."""
    stripped = line.strip()
    if not stripped or "grep" in stripped:
        return None
    parts = stripped.split(None, 1)
    if len(parts) == 2:
        with contextlib.suppress(ValueError):
            return int(parts[0]), parts[1]
    aux_parts = stripped.split()
    if len(aux_parts) > 10 and aux_parts[1].isdigit():
        return int(aux_parts[1]), " ".join(aux_parts[10:])
    return None


def _iter_windows_list_processes(listing: str):
    """Yield ``(pid, command_line)`` from wmic/CIM ``/FORMAT:LIST`` output."""
    current_cmd = ""
    for line in listing.split("\n"):
        line = line.strip()
        if line.startswith("CommandLine="):
            current_cmd = line[len("CommandLine=") :]
        elif line.startswith("ProcessId="):
            with contextlib.suppress(ValueError):
                yield int(line[len("ProcessId=") :]), current_cmd
            current_cmd = ""


def _windows_process_listing() -> str | None:
    """``CommandLine=``/``ProcessId=`` LIST output for every Windows process, or None.

    wmic when present, else Get-CimInstance emitting the same shape. ``bounded_probe_run``, NOT
    ``subprocess.run(timeout=...)``: on Windows run()'s post-timeout cleanup joins pipe readers
    unbounded and a conhost.exe holding duplicated handles wedges the caller forever; it also
    hides the console window this windowless pythonw backend would otherwise flash.
    """
    from hermes_cli._subprocess_compat import bounded_probe_run

    wmic_path = shutil.which("wmic")
    result = None
    if wmic_path is not None:
        result = bounded_probe_run(
            [wmic_path, "process", "get", "ProcessId,CommandLine", "/FORMAT:LIST"],
            timeout=10,
            errors="ignore",
        )
    if result is None or result.returncode != 0 or not (result.stdout or ""):
        powershell = shutil.which("powershell") or shutil.which("pwsh")
        if powershell is None:
            return None
        ps_cmd = (
            "Get-CimInstance Win32_Process | "
            "ForEach-Object { "
            "  'CommandLine=' + ($_.CommandLine -replace \"`r`n\",' ' -replace \"`n\",' '); "
            "  'ProcessId=' + $_.ProcessId; "
            "  '' "
            "}"
        )
        result = bounded_probe_run(
            [powershell, "-NoProfile", "-Command", ps_cmd],
            timeout=15,
            errors="ignore",
        )
        if result is None:
            return None
    if result.returncode != 0 or result.stdout is None:
        return None
    return result.stdout


def _filter_venv_launcher_stubs(pids: list[int]) -> list[int]:
    """Drop venv-launcher ``pythonw.exe`` stubs that parent another matched PID (see ``_scan_gateway_pids``)."""
    try:
        import psutil  # type: ignore
    except ImportError:
        return pids

    pid_set = set(pids)
    parent_of: dict[int, int | None] = {}
    for pid in pids:
        try:
            parent_of[pid] = psutil.Process(pid).ppid()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            parent_of[pid] = None

    drop: set[int] = set()
    for pid, ppid in parent_of.items():
        if ppid is not None and ppid in pid_set:
            drop.add(ppid)

    return [p for p in pids if p not in drop]


def find_gateway_pids(exclude_pids: set | None = None, all_profiles: bool = False) -> list:
    """Find running gateway PIDs for the current profile, or every profile with ``all_profiles`` (``hermes update``)."""
    _exclude = set(exclude_pids or set())
    pids: list[int] = []
    if not all_profiles:
        try:
            from gateway.status import get_running_pid

            _append_unique_pid(pids, get_running_pid(), _exclude)
        except Exception:
            pass
    for pid in _get_service_pids(all_profiles=all_profiles):
        _append_unique_pid(pids, pid, _exclude)
    try:
        include_restart_managers = not supports_systemd_services()
    except Exception:
        include_restart_managers = False
    for pid in _scan_gateway_pids(
        _exclude,
        all_profiles=all_profiles,
        include_restart_managers=include_restart_managers,
    ):
        _append_unique_pid(pids, pid, _exclude)
    return pids


def find_profile_gateway_processes(
    exclude_pids: set | None = None,
    *,
    strict: bool = False,
) -> list[ProfileGatewayProcess]:
    """Return running gateway PIDs mapped to Hermes profiles via PID files."""
    _exclude = set(exclude_pids or set())
    processes: list[ProfileGatewayProcess] = []
    try:
        from gateway.status import get_running_pid, get_running_pid_identity_strict
        from hermes_cli.profiles import list_profiles
    except Exception:
        if strict:
            raise
        return processes

    seen: set[int] = set()
    try:
        profiles = list_profiles()
    except Exception:
        if strict:
            raise
        return processes
    for profile in profiles:
        try:
            if strict:
                identity = get_running_pid_identity_strict(profile.path / "gateway.pid")
                pid = identity[0] if identity else None
                create_time = identity[1] if identity else 0.0
            else:
                pid = get_running_pid(profile.path / "gateway.pid", cleanup_stale=False)
                create_time = 0.0
        except Exception as exc:
            if strict:
                raise RuntimeError(f"Could not inspect gateway PID for profile {profile.name}") from exc
            continue
        if pid is None or pid <= 0 or pid in _exclude or pid in seen:
            continue
        seen.add(pid)
        processes.append(
            ProfileGatewayProcess(profile=profile.name, path=profile.path, pid=pid, create_time=create_time)
        )
    return processes


def find_windows_gateway_services(
    *,
    psutil_module=None,
    profile_processes: list[ProfileGatewayProcess] | None = None,
) -> list[WindowsGatewayService]:
    """Find profile gateways supervised by real Windows services.

    Service-logon processes may hide their command lines, so identity comes from Hermes's own PID
    file plus a parent chain ending at a running SCM service PID. The whole service subtree is
    returned so the Desktop preflight exempts exactly what the updater stops through the SCM.
    """
    if sys.platform != "win32":
        return []
    try:
        if psutil_module is None:
            import psutil as psutil_module  # type: ignore[no-redef]  # noqa: PLC0415
        if profile_processes is None:
            profile_processes = find_profile_gateway_processes(strict=True)
        service_names_by_pid: dict[int, set[str]] = {}
        indeterminate_services_by_pid: dict[int, list[tuple[str, object]]] = {}
        for service in psutil_module.win_service_iter():
            try:
                if all(callable(getattr(service, field, None)) for field in ("name", "status", "pid")):
                    service_name = str(service.name() or "")
                    service_status = service.status()
                    service_pid = int(service.pid() or 0)
                else:
                    data = service.as_dict()
                    service_name = str(data.get("name") or "")
                    service_status = data.get("status")
                    service_pid = int(data.get("pid") or 0)
            except FileNotFoundError:
                # Deleted between enumeration and inspection.
                continue
            except Exception as exc:
                raise RuntimeError("SCM service inspection failed") from exc
            if not service_name:
                raise RuntimeError("SCM service has an empty name")
            if service_status == "stopped":
                continue
            if service_status != "running":
                if service_pid > 0:
                    indeterminate_services_by_pid.setdefault(service_pid, []).append(
                        (service_name, service_status)
                    )
                continue
            if service_pid <= 0:
                raise RuntimeError(f"Running SCM service {service_name} has no valid process ID")
            service_names_by_pid.setdefault(service_pid, set()).add(service_name)
    except Exception as exc:
        raise RuntimeError("SCM service enumeration failed") from exc

    found: dict[str, WindowsGatewayService] = {}
    for profile_process in profile_processes:
        try:
            gateway_process = psutil_module.Process(int(profile_process.pid))
            gateway_create_time = float(gateway_process.create_time())
            if profile_process.create_time <= 0 or abs(
                gateway_create_time - profile_process.create_time
            ) > 0.001:
                raise RuntimeError("Gateway process identity changed during SCM discovery")
            ancestor_pids = [int(parent.pid) for parent in gateway_process.parents()]
            for pid in ancestor_pids:
                indeterminate_services = indeterminate_services_by_pid.get(pid, [])
                if indeterminate_services:
                    service_name, service_status = indeterminate_services[0]
                    raise RuntimeError(
                        f"SCM service {service_name} has indeterminate status: "
                        f"{service_status}"
                    )
            shared_service_pids = [
                pid
                for pid in ancestor_pids
                if len(service_names_by_pid.get(pid, set())) > 1
            ]
            if shared_service_pids:
                raise RuntimeError(
                    "Gateway ownership is ambiguous under shared SCM host PID(s): "
                    + ", ".join(str(pid) for pid in shared_service_pids)
                )
            service_pid = next(
                (pid for pid in ancestor_pids if len(service_names_by_pid.get(pid, set())) == 1),
                None,
            )
            if service_pid is None:
                continue
            service_name = next(iter(service_names_by_pid[service_pid]))
            service_process = psutil_module.Process(service_pid)
            service_create_time = float(service_process.create_time())
            descendant_processes = service_process.children(recursive=True)
            descendants = frozenset(int(child.pid) for child in descendant_processes)
            if int(profile_process.pid) not in descendants:
                continue
            descendant_identities = tuple(
                sorted((int(child.pid), float(child.create_time())) for child in descendant_processes)
            )
            found[service_name] = WindowsGatewayService(
                name=service_name,
                profile=str(profile_process.profile),
                service_pid=service_pid,
                gateway_pid=int(profile_process.pid),
                descendant_pids=descendants,
                descendant_identities=descendant_identities,
                service_create_time=service_create_time,
                gateway_create_time=gateway_create_time,
            )
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(
                "Could not determine SCM ownership for gateway profile "
                f"{profile_process.profile}"
            ) from exc
    return [found[name] for name in sorted(found)]


def _gateway_run_args_for_profile(profile: str) -> list[str]:
    args = [get_python_path(), "-m", "hermes_cli.main"]
    if profile != "default":
        args.extend(["--profile", profile])
    args.extend(["gateway", "run", "--replace"])
    return args


def _capture_gateway_argv(pid: int) -> list[str] | None:
    """Live argv of a running gateway (snapshotted before update kills so unmapped gateways can respawn).

    ``None`` if psutil is unavailable, the process is gone/denied, or the argv isn't a gateway command.
    """
    if pid <= 1:
        return None
    try:
        import psutil  # type: ignore
    except ImportError:
        return None
    try:
        argv = list(psutil.Process(pid).cmdline() or [])
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        return None
    except Exception:
        return None
    if not argv:
        return None
    # Never respawn an unrelated process the scan happened to report.
    try:
        from gateway.status import looks_like_gateway_command_line

        if not looks_like_gateway_command_line(" ".join(argv)):
            return None
    except Exception:
        pass
    return argv


def _prepare_profile_gateway_update_restart(profile: str, pid: int) -> str | None:
    """Choose who relaunches a profile gateway after ``hermes update``.

    ``--external-supervisor`` gateways must exit back to their manager (a detached watcher would
    race its replacement). Otherwise arm the profile-derived detached watcher, falling back to
    replaying the captured command line.
    """
    argv = _capture_gateway_argv(pid)
    if argv and "--external-supervisor" in argv:
        return "external-supervisor"
    if launch_detached_profile_gateway_restart(profile, pid):
        return "detached"
    if argv and launch_detached_gateway_restart_by_cmdline(pid, list(argv)):
        return "detached-cmdline"
    return None


def launch_detached_gateway_restart_by_cmdline(old_pid: int, run_argv: list[str]) -> bool:
    """Relaunch a gateway with no profile→PID-file mapping by replaying its captured argv after exit."""
    if old_pid <= 0 or not run_argv:
        return False
    return _spawn_gateway_restart_watcher(old_pid, list(run_argv))


def launch_detached_profile_gateway_restart(profile: str, old_pid: int) -> bool:
    """Relaunch a manually-run profile gateway after its current PID exits."""
    if old_pid <= 0:
        return False
    return _spawn_gateway_restart_watcher(old_pid, _gateway_run_args_for_profile(profile))


def _spawn_gateway_restart_watcher(old_pid: int, run_argv: list[str]) -> bool:
    """Spawn the detached watcher that respawns ``run_argv`` once ``old_pid`` exits."""
    if old_pid <= 0 or not run_argv:
        return False

    # Both watcher and respawned gateway need platform-appropriate detach: POSIX
    # ``start_new_session=True`` (setsid); on Windows that flag does NOT detach (the watcher would
    # die with the CLI console), so ``windows_detach_popen_kwargs()`` supplies the creationflags.
    from hermes_cli._subprocess_compat import (
        windows_detach_flags_without_breakaway,
        windows_detach_popen_kwargs,
    )

    # Windows: normalize the interpreter and capture a stable cwd + env overlay (HERMES_HOME,
    # VIRTUAL_ENV, PYTHONPATH) so the respawn doesn't depend on the watcher's cwd. No-op on POSIX.
    respawn_cwd = ""
    respawn_env_overlay: dict[str, str] = {}
    if sys.platform == "win32":
        try:
            from hermes_cli.gateway_windows import (windowless_gateway_restart_spec)

            run_argv, respawn_cwd, respawn_env_overlay = windowless_gateway_restart_spec(list(run_argv))
        except Exception:
            # Fall back to the original argv: a visible window beats a failed respawn.
            respawn_cwd = ""
            respawn_env_overlay = {}

    # Embedded as JSON literals in the watcher source (no extra argv plumbing).
    respawn_cwd_literal = json.dumps(respawn_cwd)
    respawn_env_literal = json.dumps(respawn_env_overlay)

    watcher = textwrap.dedent(
        """
        import os
        import subprocess
        import sys
        import time
        from hermes_cli._subprocess_compat import (
            _WINDOWS_GATEWAY_BREAKAWAY_ENV,
            windows_detach_flags,
            windows_detach_flags_without_breakaway,
        )

        pid = int(sys.argv[1])
        cmd = sys.argv[2:]
        _respawn_cwd = {respawn_cwd_literal}
        _respawn_env_overlay = {respawn_env_literal}
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            # ``os.kill(pid, 0)`` is not a no-op on Windows — use the
            # cross-platform existence check.
            from gateway.status import _pid_exists
            if not _pid_exists(pid):
                break
            time.sleep(0.2)

        # Route stray stdout/stderr from the respawned gateway to the same
        # sidecar log _spawn_detached uses.  DEVNULL here meant a gateway
        # killed moments after respawn (e.g. parent Job Object teardown when
        # breakaway is denied, #48820 4th repro) left ZERO trace anywhere —
        # no gateway.log line, no exit-diag record, nothing.  Best-effort:
        # fall back to DEVNULL when the log dir is unavailable.
        _stdio_target = subprocess.DEVNULL
        _stdio_fh = None
        try:
            from hermes_cli.config import get_hermes_home
            from pathlib import Path
            _log_dir = Path(get_hermes_home()) / "logs"
            _log_dir.mkdir(parents=True, exist_ok=True)
            _stdio_fh = open(_log_dir / "gateway-stdio.log", "ab", buffering=0)
            _stdio_target = _stdio_fh
        except Exception:
            pass

        # Platform-appropriate detach for the respawned gateway.  On POSIX
        # start_new_session=True maps to os.setsid; on Windows we need
        # explicit creationflags because start_new_session is a no-op there.
        # CREATE_BREAKAWAY_FROM_JOB is critical: the watcher itself may have
        # been spawned inside a job object (Electron/Tauri parent), and
        # without breakaway the respawned gateway would die when that job
        # tears down. See _subprocess_compat.windows_detach_flags().
        _popen_kwargs = {{
            "stdout": _stdio_target,
            "stderr": _stdio_target,
        }}
        # Anchor the respawned gateway at the stable working dir and overlay
        # the env (VIRTUAL_ENV / PYTHONPATH / HERMES_HOME) the windowless
        # base interpreter needs to import hermes_cli.  Empty on POSIX, where
        # the venv python resolves imports without help.
        if _respawn_cwd:
            _popen_kwargs["cwd"] = _respawn_cwd
        _base_env = {{**os.environ, **_respawn_env_overlay}}
        try:
            if sys.platform == "win32":
                try:
                    _popen_kwargs["creationflags"] = windows_detach_flags()
                    # Stamp the breakaway state exactly like the canonical
                    # gateway_windows._spawn_detached, so the respawned
                    # gateway's exit-diag / lifecycle records show whether it
                    # escaped the parent Job Object (#48820 4th repro:
                    # without the stamp, a job-teardown kill was
                    # indistinguishable from any other silent death).
                    _popen_kwargs["env"] = {{
                        **_base_env, _WINDOWS_GATEWAY_BREAKAWAY_ENV: "1",
                    }}
                    subprocess.Popen(cmd, **_popen_kwargs)
                except OSError:
                    # CREATE_BREAKAWAY_FROM_JOB can be rejected with
                    # ERROR_ACCESS_DENIED when the parent's job object refuses
                    # breakaway. Retry without it — DETACHED_PROCESS et al.
                    # alone are enough in most setups. Mirrors the canonical
                    # fallback in gateway_windows._spawn_detached.
                    _popen_kwargs["creationflags"] = (
                        windows_detach_flags_without_breakaway()
                    )
                    _popen_kwargs["env"] = {{
                        **_base_env, _WINDOWS_GATEWAY_BREAKAWAY_ENV: "0",
                    }}
                    subprocess.Popen(cmd, **_popen_kwargs)
            else:
                if _respawn_env_overlay:
                    _popen_kwargs["env"] = _base_env
                _popen_kwargs["start_new_session"] = True
                subprocess.Popen(cmd, **_popen_kwargs)
        finally:
            if _stdio_fh is not None:
                try:
                    _stdio_fh.close()
                except OSError:
                    pass
        """
    ).strip().format(
        respawn_cwd_literal=respawn_cwd_literal,
        respawn_env_literal=respawn_env_literal,
    )

    watcher_argv = [sys.executable, "-c", watcher, str(old_pid), *run_argv]

    # Same detach for the watcher itself, so closing the terminal doesn't kill it.
    try:
        subprocess.Popen(
            watcher_argv,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            **windows_detach_popen_kwargs(),
        )
    except OSError:
        # Parent job object rejected CREATE_BREAKAWAY_FROM_JOB; retry without it (Windows only —
        # ``start_new_session=True`` cannot raise OSError on POSIX).
        try:
            fallback_kwargs: dict = (
                {"creationflags": windows_detach_flags_without_breakaway()}
                if sys.platform == "win32"
                else {"start_new_session": True}
            )
            subprocess.Popen(
                watcher_argv,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                **fallback_kwargs,
            )
        except OSError:
            return False
    return True


def _systemd_unit_is_active(system: bool) -> bool:
    """``systemctl is-active`` == "active" for the installed unit in ``system`` scope, else False."""
    if not get_systemd_unit_path(system=system).exists():
        return False
    try:
        result = _run_systemctl(["is-active", get_service_name()], system=system, timeout=10, **_CAPTURE_TEXT)
    except (RuntimeError, subprocess.TimeoutExpired):
        return False
    return result.stdout.strip() == "active"


def _probe_systemd_service_running(system: bool = False) -> tuple[bool, bool]:
    selected_system = _select_systemd_scope(system)
    return selected_system, _systemd_unit_is_active(selected_system)


def _read_systemd_unit_environment(system: bool = False) -> dict[str, str]:
    """Parse ``systemctl show -p Environment`` (one line of unquoted space-separated KEY=VALUE pairs)."""
    body = _systemctl_show(("Environment",), system=system).get("Environment", "")
    parsed: dict[str, str] = {}
    for token in body.split():
        if "=" in token:
            key, value = token.split("=", 1)
            parsed[key] = value
    return parsed


def _systemctl_show(properties: tuple[str, ...], *, system: bool) -> dict[str, str]:
    """``systemctl show --property a,b`` for the gateway unit as ``{key: value}``; {} on failure."""
    try:
        result = _run_systemctl(
            ["show", get_service_name(), "--no-pager", "--property", ",".join(properties)],
            system=_select_systemd_scope(system),
            timeout=10,
            **_CAPTURE_TEXT,
        )
    except (RuntimeError, subprocess.TimeoutExpired, OSError):
        return {}
    if result.returncode != 0:
        return {}
    parsed: dict[str, str] = {}
    for line in result.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            parsed[key] = value.strip()
    return parsed


def _hermes_home_from_systemd_unit_file(system: bool = False) -> str | None:
    """``HERMES_HOME`` from the on-disk unit file — what refresh/compare already read, and reliable under ``sudo``."""
    unit_path = get_systemd_unit_path(system=system)
    if not unit_path.exists():
        return None
    try:
        text = unit_path.read_text(encoding="utf-8")
    except OSError:
        return None
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("Environment="):
            continue
        body = stripped[len("Environment=") :].strip().strip('"')
        if body.startswith("HERMES_HOME="):
            value = body.split("=", 1)[1].strip().strip('"')
            return value or None
    return None


def _sync_hermes_home_from_systemd_unit(system: bool) -> None:
    """For a system-scope unit, adopt its ``HERMES_HOME``.

    Under ``sudo`` HERMES_HOME is stripped and HOME=/root, so get_hermes_home() would pick the
    wrong profile; mirroring the unit's value makes runtime-status/PID reads hit the right files.
    """
    if not system:
        return
    # On-disk unit first; ``systemctl show`` for units that only exist in the manager.
    unit_home = (_hermes_home_from_systemd_unit_file(system=True) or "").strip()
    if not unit_home:
        unit_home = _read_systemd_unit_environment(system=True).get("HERMES_HOME", "").strip()
    if not unit_home:
        return
    current = os.environ.get("HERMES_HOME", "").strip()
    if current == unit_home:
        return
    os.environ["HERMES_HOME"] = unit_home


def _read_systemd_unit_properties(
    system: bool = False,
    properties: tuple[str, ...] = ("ActiveState", "SubState", "Result", "ExecMainStatus", "MainPID"),
) -> dict[str, str]:
    """Return selected ``systemctl show`` properties for the gateway unit."""
    return _systemctl_show(properties, system=system)


def _systemd_main_pid_from_props(props: dict[str, str]) -> int | None:
    try:
        pid = int(props.get("MainPID", "0") or "0")
    except (TypeError, ValueError):
        return None
    return pid if pid > 0 else None


def _systemd_main_pid(system: bool = False) -> int | None:
    return _systemd_main_pid_from_props(_read_systemd_unit_properties(system=system))


def _read_gateway_runtime_status() -> dict | None:
    try:
        from gateway.status import read_runtime_status

        state = read_runtime_status()
    except Exception:
        return None
    return state if isinstance(state, dict) else None


def _gateway_runtime_status_for_pid(pid: int | None) -> dict | None:
    if not pid:
        return None
    state = _read_gateway_runtime_status()
    if not state:
        return None
    try:
        state_pid = int(state.get("pid", 0) or 0)
    except (TypeError, ValueError):
        return None
    return state if state_pid == pid else None


def _wait_for_systemd_service_restart(
    *,
    system: bool = False,
    previous_pid: int | None = None,
    timeout: float | None = None,
    replacement_observed: list[bool] | None = None,
) -> bool:
    """Wait for the gateway service to become active after a restart handoff."""
    import time

    svc = get_service_name()
    scope_label = _service_scope_label(system).capitalize()
    if timeout is None:
        timeout = _systemd_restart_wait_timeout(system=system)
    deadline = time.monotonic() + timeout
    printed_runtime_wait = False

    while time.monotonic() < deadline:
        props = _read_systemd_unit_properties(system=system)
        active_state = props.get("ActiveState", "")
        sub_state = props.get("SubState", "")
        new_pid = None
        try:
            from gateway.status import get_running_pid

            new_pid = get_running_pid()
        except Exception:
            new_pid = None
        if not new_pid:
            new_pid = _systemd_main_pid_from_props(props)

        runtime_state = _read_gateway_runtime_status()
        try:
            runtime_pid = int((runtime_state or {}).get("pid", 0) or 0)
        except (TypeError, ValueError):
            runtime_pid = 0
        if (
            previous_pid is not None
            and replacement_observed is not None
            and not replacement_observed
            and any(
                candidate_pid > 0 and candidate_pid != previous_pid
                for candidate_pid in (new_pid or 0, runtime_pid)
            )
        ):
            replacement_observed.append(True)

        if active_state == "active" and new_pid and (previous_pid is None or new_pid != previous_pid):
            if runtime_pid != new_pid:
                runtime_state = _gateway_runtime_status_for_pid(new_pid)
            gateway_state = (runtime_state or {}).get("gateway_state")
            if gateway_state == "running":
                print(f"✓ {scope_label} service restarted (PID {new_pid})")
                return True
            if gateway_state == "startup_failed":
                reason = (runtime_state or {}).get("exit_reason") or "startup failed"
                print(
                    f"⚠ {scope_label} service process restarted (PID {new_pid}), but gateway startup failed: {reason}"
                )
                return False
            if not printed_runtime_wait:
                print(
                    f"⏳ {scope_label} service process started (PID {new_pid}); waiting for gateway runtime..."
                )
                printed_runtime_wait = True

        if active_state == "activating" and sub_state == "auto-restart":
            time.sleep(1)
            continue

        if _systemd_unit_is_start_limited(props):
            _print_systemd_start_limit_wait(system=system)
            return False

        time.sleep(2)

    print(
        f"⚠ {scope_label} service did not become active within {int(timeout)}s.\n"
        f"  Check status: {'sudo ' if system else ''}hermes gateway status\n"
        f"  Check logs:   journalctl {'--user ' if not system else ''}-u {svc} -l --since '2 min ago'"
    )
    return False


def _systemd_restart_wait_timeout(system: bool = False) -> float:
    """Cover systemd's relaunch delays before applying the runtime wait floor."""
    from gateway.shutdown_forensics import parse_systemd_duration_to_us

    props = _read_systemd_unit_properties(system=system, properties=("RestartUSec", "TimeoutStartUSec"))
    supervisor_budget = 0.0
    for name in ("RestartUSec", "TimeoutStartUSec"):
        raw = props.get(name, "")
        duration_us = (int(raw) if raw.isdigit() else parse_systemd_duration_to_us(raw))
        if duration_us is not None:
            supervisor_budget += duration_us / 1_000_000
    return 60.0 + supervisor_budget


def _systemd_unit_is_start_limited(props: dict[str, str]) -> bool:
    result = props.get("Result", "").lower()
    sub_state = props.get("SubState", "").lower()
    return result == "start-limit-hit" or sub_state == "start-limit-hit"


def _systemd_error_indicates_start_limit(exc: subprocess.CalledProcessError) -> bool:
    parts: list[str] = []
    for attr in ("stderr", "stdout", "output"):
        value = getattr(exc, attr, None)
        if not value:
            continue
        if isinstance(value, bytes):
            value = value.decode(errors="replace")
        parts.append(str(value))
    text = "\n".join(parts).lower()
    return "start-limit-hit" in text or "start request repeated too quickly" in text or "start-limit" in text


def _systemd_service_is_start_limited(system: bool = False) -> bool:
    return _systemd_unit_is_start_limited(_read_systemd_unit_properties(system=system))


def _print_systemd_start_limit_wait(system: bool = False) -> None:
    svc = get_service_name()
    scope_label = _service_scope_label(system).capitalize()
    scope_flag = " --system" if system else ""
    systemctl_prefix = "systemctl " if system else "systemctl --user "
    journal_prefix = "journalctl " if system else "journalctl --user "
    print(f"⏳ {scope_label} service is temporarily rate-limited by systemd.")
    print("  systemd is refusing another immediate start after repeated exits.")
    print(
        f"  Wait for the start-limit window to expire, then run: {'sudo ' if system else ''}hermes gateway restart{scope_flag}"
    )
    print(f"  Or clear the failed state manually: {systemctl_prefix}reset-failed {svc}")
    print(f"  Check logs: {journal_prefix}-u {svc} -l --since '5 min ago'")


def _recover_pending_systemd_restart(system: bool = False, previous_pid: int | None = None) -> bool:
    """Recover a planned service restart that is stuck in systemd state."""
    props = _read_systemd_unit_properties(system=system)
    if not props:
        return False

    try:
        from gateway.status import read_runtime_status
    except Exception:
        return False

    runtime_state = read_runtime_status() or {}
    if not runtime_state.get("restart_requested"):
        return False

    active_state = props.get("ActiveState", "")
    sub_state = props.get("SubState", "")
    exec_main_status = props.get("ExecMainStatus", "")
    result = props.get("Result", "")

    if active_state == "activating" and sub_state == "auto-restart":
        print("⏳ Service restart already pending — waiting for systemd relaunch...")
        return _wait_for_systemd_service_restart(system=system, previous_pid=previous_pid)

    if active_state == "failed" and (
        exec_main_status == str(GATEWAY_SERVICE_RESTART_EXIT_CODE)
        or result == "exit-code"
    ):
        svc = get_service_name()
        scope_label = _service_scope_label(system).capitalize()
        print(f"↻ Clearing failed state for pending {scope_label.lower()} service restart...")
        _run_systemctl(["reset-failed", svc], system=system, check=False, timeout=30)
        _run_systemctl(["start", svc], system=system, check=False, timeout=90)
        return _wait_for_systemd_service_restart(system=system, previous_pid=previous_pid)

    return False


def _parse_launchd_pid_from_list_output(output: str) -> int | None:
    """PID from ``launchctl list <label>`` (``"PID" = <n>;``); None if absent (registered, not running)
    or non-positive (crashed)."""
    for line in output.splitlines():
        stripped = line.strip()
        if stripped.startswith('"PID"') or stripped.startswith("PID"):
            parts = stripped.split("=", 1)
            if len(parts) == 2:
                val = parts[1].strip().rstrip(";").strip('"')
                try:
                    pid = int(val)
                    return pid if pid > 0 else None
                except ValueError:
                    return None
    return None


def _parse_launchd_pid_from_print_output(output: str) -> int | None:
    """Live PID from ``launchctl print`` (first ``pid = <N>`` line wins); None if absent or non-positive."""
    for line in output.splitlines():
        stripped = line.strip()
        if stripped.startswith("pid = "):
            try:
                pid = int(stripped[len("pid = "):].strip())
                return pid if pid > 0 else None
            except ValueError:
                return None
    return None


def _launchd_print_service_pid(domain: str, label: str) -> tuple[bool, int | None]:
    """``(loaded, pid)`` for ``domain/label`` via ``launchctl print``.

    Domain-explicit on purpose (``launchctl list`` infers domain from caller context).
    ``TimeoutExpired`` propagates: a wedged launchctl must be reported, not read as "unloaded".
    """
    try:
        result = subprocess.run(["launchctl", "print", f"{domain}/{label}"], timeout=5, **_CAPTURE_TEXT)
    except FileNotFoundError:
        return (False, None)
    if result.returncode != 0:
        return (False, None)
    return (True, _parse_launchd_pid_from_print_output(result.stdout))


def _launchd_service_registered(label: str, *, timeout: int = 5) -> bool:
    """True when launchd knows ``label`` (``launchctl list`` exit 0).

    Domain-agnostic, so it stays true on macOS 26+ hosts whose per-user domains reject management
    (``launchd_restart()`` owns that fallback). FileNotFoundError/TimeoutExpired propagate.
    """
    result = subprocess.run(["launchctl", "list", label], timeout=timeout, **_CAPTURE_TEXT)
    return result.returncode == 0


def _locate_launchd_gateway_service(label: str) -> tuple[str | None, int | None]:
    """``(domain, pid)`` for ``label``, probing ``gui/<uid>`` then ``user/<uid>``.

    Never consults the current profile's cached ``_launchd_domain()``: a fleet can mix domains
    (SSH installs land in ``user/<uid>``). ``TimeoutExpired`` propagates.
    """
    uid = os.getuid()  # windows-footgun: ok — POSIX launchd (macOS) helper, never invoked on Windows
    for domain in (f"gui/{uid}", f"user/{uid}"):
        loaded, pid = _launchd_print_service_pid(domain, label)
        if loaded:
            return (domain, pid)
    return (None, None)


def _probe_launchd_service_running() -> bool:
    """True when the plist exists AND launchd is running a process for the current label."""
    return get_launchd_plist_path().exists() and _launchctl_label_supervising_process(get_launchd_label())


def get_gateway_runtime_snapshot(system: bool = False) -> GatewayRuntimeSnapshot:
    """Return a unified view of gateway liveness for the current profile."""
    gateway_pids = tuple(find_gateway_pids())
    if is_termux():
        return GatewayRuntimeSnapshot(manager="Termux / manual process", gateway_pids=gateway_pids)

    from hermes_constants import is_container

    if is_linux() and is_container():
        # Report s6 supervision under our /init; other container runtimes keep "docker (foreground)".
        try:
            from hermes_cli.service_manager import detect_service_manager, get_service_manager
            if detect_service_manager() == "s6":
                profile = _profile_suffix() or "default"
                service_name = f"gateway-{profile}"
                mgr = get_service_manager()
                service_installed = False
                service_running = False
                try:
                    service_dir = getattr(mgr, "scandir", None)
                    if service_dir is not None:
                        service_installed = (service_dir / service_name).is_dir()
                except Exception:
                    service_installed = False
                if service_installed:
                    try:
                        service_running = bool(mgr.is_running(service_name))
                    except Exception:
                        service_running = False
                return GatewayRuntimeSnapshot(
                    manager="s6 (container supervisor)",
                    service_installed=service_installed,
                    service_running=service_running,
                    gateway_pids=gateway_pids,
                    service_scope="s6",
                )
        except Exception:
            pass  # Fall through to the legacy label on any detection error.
        return GatewayRuntimeSnapshot(manager="docker (foreground)", gateway_pids=gateway_pids)

    if supports_systemd_services():
        selected_system, service_running = _probe_systemd_service_running(system=system)
        scope_label = _service_scope_label(selected_system)
        return GatewayRuntimeSnapshot(
            manager=f"systemd ({scope_label})",
            service_installed=get_systemd_unit_path(system=selected_system).exists(),
            service_running=service_running,
            gateway_pids=gateway_pids,
            service_scope=scope_label,
        )

    if is_macos():
        return GatewayRuntimeSnapshot(
            manager="launchd",
            service_installed=get_launchd_plist_path().exists(),
            service_running=_probe_launchd_service_running(),
            gateway_pids=gateway_pids,
            service_scope="launchd",
        )

    return GatewayRuntimeSnapshot(manager="manual process", gateway_pids=gateway_pids)


def _format_gateway_pids(pids: tuple[int, ...] | list[int], *, limit: int | None = 3) -> str:
    rendered = (
        [str(pid) for pid in pids[:limit] if pid > 0]
        if limit is not None
        else [str(pid) for pid in pids if pid > 0]
    )
    if limit is not None and len(pids) > limit:
        rendered.append("...")
    return ", ".join(rendered)


def _print_gateway_process_mismatch(snapshot: GatewayRuntimeSnapshot) -> None:
    if not snapshot.has_process_service_mismatch:
        return
    print()
    # Managed detached fallback (macOS launchd exit-5 path) vs. a genuinely manual run.
    if _launchd_unsupported_marker_exists():
        print("⚠ Gateway is running as a detached fallback process — launchd cannot supervise it")
        print(f"  PID(s): {_format_gateway_pids(snapshot.gateway_pids, limit=None)}")
        print("  Auto-start at login and auto-restart on crash are NOT available.")
        print("  Stop it with: hermes gateway stop")
    else:
        print("⚠ Gateway process is running for this profile, but the service is not active")
        print(f"  PID(s): {_format_gateway_pids(snapshot.gateway_pids, limit=None)}")
        print("  This is usually a manual foreground/tmux/nohup run, so `hermes gateway`")
        print("  can refuse to start another copy until this process stops.")


def _print_other_profiles_gateway_status() -> None:
    """Print other profiles' running gateways at the bottom of ``hermes gateway status``."""
    try:
        from hermes_cli.profiles import get_active_profile_name

        current = get_active_profile_name()
        other_processes = [p for p in find_profile_gateway_processes() if p.profile != current]
        if not other_processes:
            return

        print()
        print("Other profiles:")
        for proc in other_processes:
            print(f"  ✓ {proc.profile:<16s} — PID {proc.pid}")
    except Exception:
        pass


def _gateway_list() -> None:
    """List every profile and whether its gateway is running."""
    try:
        from hermes_cli.profiles import list_profiles, get_active_profile_name
    except Exception:
        print("Unable to list profiles.")
        return

    profiles = list_profiles()
    if not profiles:
        print("No profiles found.")
        return

    current = get_active_profile_name()

    print("Gateways:")
    for prof in profiles:
        marker = "✓" if prof.gateway_running else "✗"
        label = prof.name
        if prof.name == current:
            label += " (current)"
        parts = [f"  {marker} {label:<24s}"]
        if prof.gateway_running:
            pid = None
            try:
                from gateway.status import get_running_pid

                pid = get_running_pid(prof.path / "gateway.pid", cleanup_stale=False)
            except Exception:
                pass
            if pid:
                parts.append(f"PID {pid}")
            elif named_profile_served_by_running_multiplexer(prof.name):
                parts.append("served by the default multiplexer")
        else:
            parts.append("not running")
        print(" — ".join(parts))


def kill_gateway_processes(
    force: bool = False, exclude_pids: set | None = None, all_profiles: bool = False
) -> int:
    """Kill running gateway processes (force-kill if ``force``); ``exclude_pids`` skips e.g. just-
    restarted service PIDs. Returns count killed."""
    pids = find_gateway_pids(exclude_pids=exclude_pids, all_profiles=all_profiles)
    killed = 0

    for pid in pids:
        try:
            expected_start_time = None
            if force:
                # Re-verify the LIVE cmdline at kill time: a PID recycled since the scan must
                # never be tree-killed.
                if _capture_gateway_argv(pid) is None:
                    continue
                from gateway.status import get_process_start_time

                expected_start_time = get_process_start_time(pid)
            terminate_pid(pid, force=force, expected_start_time=expected_start_time)
            killed += 1
        except ProcessLookupError:
            pass
        except PermissionError:
            print(f"⚠ Permission denied to kill PID {pid}")

        except OSError as exc:
            print(f"Failed to kill PID {pid}: {exc}")
    return killed


_REAPER_SUPERVISOR_WALK_LIMIT = 12


def _reaper_candidate_is_supervisor_owned(pid: int) -> bool:
    """True when ``pid``'s parent chain reaches ``services.exe`` (Windows Task Scheduler-owned gateway).

    Windows-only reaper backstop: ``_get_service_pids()`` is empty there, so a Scheduled-Task
    gateway with a missing/stale pidfile would look like an orphan. Fail-open: once the Task's
    bootstrap parent exits the chain breaks. Not applied on POSIX, where every process descends
    from PID 1 and would look supervised.
    """
    if not is_windows():
        return False
    try:
        import psutil  # type: ignore

        parent = psutil.Process(pid).parent()
        for _ in range(_REAPER_SUPERVISOR_WALK_LIMIT):
            if parent is None:
                break
            try:
                name = (parent.name() or "").lower()
            except Exception:
                name = ""
            if name == "services.exe":
                return True
            parent = parent.parent()
    except Exception:
        pass
    return False


def _reap_unsupervised_gateway_orphans(extra_exclude: set | None = None) -> bool:
    """Kill no-supervisor gateway orphans the pidfile/runtime record can't see.

    On WSL/no-systemd hosts the restart fallback runs the gateway in-process under a ``gateway
    restart`` argv; a stale pidfile then lets a live orphan keep the webhook port while a restart
    stacks a duplicate. No-op where a service supervisor exists — there ``gateway restart`` is a
    transient management command, not the gateway. ``extra_exclude``: PIDs the caller already killed.
    """
    try:
        supervised_host = supports_systemd_services()
    except Exception:
        supervised_host = True
    if supervised_host:
        return False

    # Windows Task Scheduler is a supervisor too; its task state is more reliable than a
    # parent-chain walk, which breaks once the VBS/conhost bootstrap exits (task then Ready, not
    # Running — a Running-only check would kill the detached gateway on every desktop start).
    if is_windows():
        try:
            # Task name is profile-aware (Hermes_Gateway_<profile>) — never hardcode it.
            from hermes_cli.gateway_windows import get_task_name

            _task_name = get_task_name()
        except Exception:
            _task_name = "Hermes_Gateway"
        if _windows_scheduled_task_supervises(_task_name):
            return False

    from gateway.status import _pid_exists, write_planned_stop_marker

    own = _reaper_exclusion_pids(extra_exclude)
    try:
        # On Windows also drop Task Scheduler-owned candidates (the pidfile-less gap).
        orphans = [
            p
            for p in find_gateway_pids(exclude_pids=own)
            if p and p > 0 and not _reaper_candidate_is_supervisor_owned(p)
        ]
    except Exception:
        return False
    if not orphans:
        return False

    # Pin each orphan's identity now: the delayed SIGKILL fires seconds later and a recycled PID
    # must never be force-killed. SIGTERM proceeds regardless; SIGKILL requires a matching fingerprint.
    from gateway.status import get_process_start_time

    orphan_identity: dict[int, int] = {}
    for pid in orphans:
        start = get_process_start_time(pid)
        if start is not None:
            orphan_identity[pid] = start

    reaped = False
    for pid in orphans:
        with contextlib.suppress(Exception):
            write_planned_stop_marker(pid)
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            continue
        except PermissionError:
            print(f"⚠ Permission denied to kill orphaned gateway PID {pid}")
            continue
        reaped = True

    # Wait, then force-kill survivors so the replacement can bind the port cleanly.
    survivors = _await_gateway_exit(orphans, pid_exists=_pid_exists)
    # Fail-closed: SIGKILL only a PID that still names the process fingerprinted at scan time.
    verified_survivors = []
    for pid in survivors:
        recorded = orphan_identity.get(pid)
        if recorded is None or get_process_start_time(pid) != recorded:
            continue
        verified_survivors.append(pid)
    _force_kill_survivors(verified_survivors)

    return reaped


def _reaper_exclusion_pids(extra_exclude: set | None) -> set[int]:
    """PIDs the orphan reaper must never kill: self, caller extras, service-managed, recorded."""
    own = {os.getpid()}
    if extra_exclude:
        own |= extra_exclude
    # Service-managed gateways are never orphans: on macOS supports_systemd_services() is False,
    # so without this a launchd gateway would be SIGTERM'd (and left down under
    # KeepAlive.SuccessfulExit=false). all_profiles=True because the scan sees every profile's
    # gateway; a sibling profile's launchd gateway must not be reaped.
    with contextlib.suppress(Exception):
        own |= _get_service_pids(all_profiles=True)
    # Exempt the recorded gateway PID and its parent chain (on Windows the Scheduled-Task
    # bootstrap's ``gateway run`` argv matches the scan; killing it takes the gateway down).
    # Evidence comes from the RAW pidfile + lock records, not the validated probe: get_running_pid
    # returns None on any validation hiccup — exactly when a healthy standalone gateway would be
    # hard-killed (Windows SIGTERM is TerminateProcess, no drain). For a KILL exclusion list a
    # stale PID at worst spares one process; a false-negative kills a live gateway. The validated
    # probe still supplies the runtime-status fallback PID when no pidfile exists.
    try:
        from gateway.status import (
            _pid_from_record,
            _read_gateway_lock_record,
            _read_pid_record,
            get_running_pid,
        )

        recorded_pids = set()
        for _record in (_read_pid_record(), _read_gateway_lock_record()):
            _raw_pid = _pid_from_record(_record)
            if _raw_pid and _raw_pid > 0:
                recorded_pids.add(_raw_pid)
        _probed = get_running_pid(cleanup_stale=False)
        if _probed and _probed > 0:
            recorded_pids.add(_probed)
        for recorded in recorded_pids:
            own.add(recorded)
            try:
                import psutil  # type: ignore

                parent = psutil.Process(recorded).parent()
                while parent is not None:
                    own.add(parent.pid)
                    parent = parent.parent()
            except Exception:
                pass
    except Exception:
        pass
    return own


# A retiring gateway runs a PASSIVE WAL checkpoint in ``SessionDB.close()``; a SIGKILL landing
# mid-checkpoint corrupts ``state.db``. The outgoing gateway keeps serving while we wait, so a
# long grace delays only the replacement's port bind, never traffic.
_ORPHAN_EXIT_GRACE_SECONDS = 30.0
_ORPHAN_EXIT_POLL_SECONDS = 0.2


def _await_gateway_exit(
    pids,
    *,
    pid_exists,
    sleep=None,
    grace_s: float = _ORPHAN_EXIT_GRACE_SECONDS,
    poll_s: float = _ORPHAN_EXIT_POLL_SECONDS,
):
    """Poll up to *grace_s* for *pids* to exit; return survivors. ``pid_exists``/``sleep`` injectable for tests."""
    if sleep is None:
        sleep = time.sleep
    survivors = [p for p in pids]
    for _ in range(max(1, int(grace_s / poll_s))):
        survivors = [p for p in survivors if pid_exists(p)]
        if not survivors:
            break
        sleep(poll_s)
    else:
        # Re-check after the LAST sleep, or a recycled PID could get the SIGKILL.
        survivors = [p for p in survivors if pid_exists(p)]
    return survivors


def _force_kill_survivors(survivors, *, kill=None) -> None:
    """SIGKILL processes that outlasted the grace period, loudly — a force-kill can tear the store, so
    it must leave a trace."""
    if not survivors:
        return
    if kill is None:
        kill = os.kill
    for pid in survivors:
        logger.warning(
            "Gateway PID %s did not exit within %.0fs of SIGTERM — sending "
            "SIGKILL. A kill during a WAL checkpoint can corrupt state.db; "
            "the next start will run an integrity check.",
            pid,
            _ORPHAN_EXIT_GRACE_SECONDS,
        )
        with contextlib.suppress((ProcessLookupError, PermissionError, OSError)):
            kill(pid, getattr(signal, "SIGKILL", signal.SIGTERM))


def _mark_planned_stop(pid: int | None = None) -> None:
    """Best-effort planned-stop marker for ``pid`` (default: the recorded gateway PID)."""
    try:
        from gateway.status import get_running_pid, write_planned_stop_marker

        if pid is None:
            pid = get_running_pid(cleanup_stale=False)
        if pid is not None:
            write_planned_stop_marker(pid)
    except Exception:
        pass


def stop_profile_gateway() -> bool:
    """Stop only this profile's gateway via its PID file; True if a process was stopped.

    Without a service supervisor the pidfile can be stale while a live orphan holds the webhook
    port, so fall back to the orphan-aware scan rather than stacking a duplicate.
    """
    try:
        from gateway.status import get_running_pid, remove_pid_file
    except ImportError:
        return False

    pid = get_running_pid()
    if pid is None:
        return _reap_unsupervised_gateway_orphans()

    _mark_planned_stop(pid)

    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass  # Already gone
    except PermissionError:
        print(f"⚠ Permission denied to kill PID {pid}")
        return False

    # ``_pid_exists``, NOT ``os.kill(pid, 0)`` (TerminateProcess on Windows).
    from gateway.status import _pid_exists

    for _ in range(20):
        if not _pid_exists(pid):
            break
        time.sleep(0.5)

    if get_running_pid() is None:
        remove_pid_file()

    # Reap orphans from prior restarts whose pidfile entry was overwritten; skip the PID just killed.
    try:
        _reap_unsupervised_gateway_orphans(extra_exclude={pid} if pid else None)
    except Exception as exc:
        logger.debug("orphan reap after stop_profile_gateway failed: %s", exc)

    return True


def is_linux() -> bool:
    return sys.platform.startswith("linux")


from hermes_constants import is_container, is_termux, is_wsl


def _wsl_systemd_operational() -> bool:
    """WSL2 with ``systemd=true`` in wsl.conf has working systemd; WSL1/without it does not."""
    return _systemd_operational(system=True)


def _systemd_operational(system: bool = False) -> bool:
    """Return True when the requested systemd scope is usable."""
    try:
        result = _run_systemctl(["is-system-running"], system=system, timeout=5, **_CAPTURE_TEXT)
        # "running", "degraded", "starting" all mean systemd is PID 1
        status = result.stdout.strip().lower()
        return status in {"running", "degraded", "starting", "initializing"}
    except (RuntimeError, subprocess.TimeoutExpired, OSError):
        return False


def supports_systemd_services() -> bool:
    if not is_linux() or is_termux():
        return False
    if shutil.which("systemctl") is None:
        return False
    if is_wsl():
        return _wsl_systemd_operational()
    if is_container():
        # A container whose init is systemd (nspawn, some k8s pods) behaves like a host.
        return _systemd_operational(system=False) or _systemd_operational(system=True)
    return True


def is_macos() -> bool:
    return sys.platform == "darwin"


def is_windows() -> bool:
    return sys.platform == "win32"


def _gw_windows():
    """Lazily import :mod:`hermes_cli.gateway_windows` (Windows-only service backend)."""
    from hermes_cli import gateway_windows

    return gateway_windows


# Task Scheduler states meaning "still supervised". Ready is the steady state after the launcher
# exits and leaves the detached gateway running; Disabled / MISSING are not supervisors.
_WINDOWS_TASK_SUPERVISOR_STATES = frozenset({"Running", "Ready", "Queued"})


def _windows_scheduled_task_state(task_name: str) -> str | None:
    """English ``Get-ScheduledTask`` State, or None on failure.

    PowerShell, not ``schtasks``: schtasks localizes its output and emits the local codepage,
    which utf-8 decoding mangles; the ``State`` enum is stable across locales.
    """
    if not is_windows():
        return None
    try:
        powershell = shutil.which("powershell") or shutil.which("pwsh")
        if powershell is None:
            return None
        ps_cmd = (
            f"$t = Get-ScheduledTask -TaskName '{task_name}' "
            "-ErrorAction SilentlyContinue; if ($t) { $t.State } else { 'MISSING' }"
        )
        result = subprocess.run(
            [powershell, "-NoProfile", "-Command", ps_cmd],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=10,
        )
        if result.returncode != 0:
            return None
        state = (result.stdout or "").strip()
        return state or None
    except (OSError, subprocess.TimeoutExpired):
        return None


def _windows_scheduled_task_supervises(task_name: str) -> bool:
    """True when Task Scheduler still owns this profile's gateway (Ready counts: the task is Ready, not Running, after bootstrap exits).

    Best-effort: any failure returns False so the caller falls back to pidfile / parent-chain exclusions.
    """
    state = _windows_scheduled_task_state(task_name)
    return state in _WINDOWS_TASK_SUPERVISOR_STATES


def _windows_gateway_should_absorb_console_controls() -> bool:
    """True for detached Windows gateway runs that should ignore Ctrl+C (``HERMES_GATEWAY_DETACHED=1``
    or no interactive stdin); foreground runs stay interruptible."""
    if not is_windows():
        return False

    detached = os.getenv("HERMES_GATEWAY_DETACHED", "").strip().lower()
    if detached in {"1", "true", "yes", "on"}:
        return True

    try:
        return not bool(sys.stdin and sys.stdin.isatty())
    except (ValueError, OSError):
        return True


def _windows_console_window_attached() -> bool | None:
    """Return whether Windows assigned this process a console window."""
    if not is_windows():
        return None
    try:
        import ctypes

        return bool(ctypes.windll.kernel32.GetConsoleWindow())  # type: ignore[attr-defined]
    except (OSError, AttributeError):
        return None


def _windows_gateway_breakaway_state() -> bool | None:
    """Consume private spawn metadata without guessing for older launchers."""
    if not is_windows():
        return None
    from hermes_cli._subprocess_compat import _WINDOWS_GATEWAY_BREAKAWAY_ENV

    value = os.environ.pop(_WINDOWS_GATEWAY_BREAKAWAY_ENV, None)
    if value == "1":
        return True
    if value == "0":
        return False
    return None


# =============================================================================
# Service Configuration
# =============================================================================

_SERVICE_BASE = "hermes-gateway"
SERVICE_DESCRIPTION = "Hermes Agent Gateway - Messaging Platform Integration"


def _profile_suffix() -> str:
    """Service-name suffix for HERMES_HOME: "" for the default root, the profile name for
    ``<root>/profiles/<name>``, else a short hash of the path."""
    import hashlib
    import re
    from hermes_constants import get_default_hermes_root

    home = get_hermes_home().resolve()
    default = get_default_hermes_root().resolve()
    if home == default:
        return ""
    # Detect <root>/profiles/<name> pattern → use the profile name
    profiles_root = (default / "profiles").resolve()
    try:
        rel = home.relative_to(profiles_root)
        parts = rel.parts
        if len(parts) == 1 and re.match(r"^[a-z0-9][a-z0-9_-]{0,63}$", parts[0]):
            return parts[0]
    except ValueError:
        pass
    # Fallback: short hash for arbitrary HERMES_HOME paths
    return hashlib.sha256(str(home).encode()).hexdigest()[:8]


def _profile_arg(hermes_home: str | None = None, default_root: str | Path | None = None) -> str:
    """Return ``--profile <name>`` for ``<root>/profiles/<name>``, else "" (default root or hash path).

    *hermes_home*/*default_root* let a sudo/root process generate a unit for another user, where
    ``get_hermes_home()``/``get_default_hermes_root()`` would otherwise refer to root.
    """
    import re
    from hermes_constants import get_default_hermes_root

    home = Path(hermes_home or str(get_hermes_home())).resolve()
    default = Path(default_root).resolve() if default_root else get_default_hermes_root().resolve()
    if home == default:
        return ""
    profiles_root = (default / "profiles").resolve()
    try:
        rel = home.relative_to(profiles_root)
        parts = rel.parts
        if len(parts) == 1 and re.match(r"^[a-z0-9][a-z0-9_-]{0,63}$", parts[0]):
            return f"--profile {parts[0]}"
    except ValueError:
        pass
    return ""


def _profile_arg_for_target_user(hermes_home: str, target_home_dir: str) -> str:
    """Return the profile arg for a system service running as another user."""
    target_root = Path(target_home_dir) / ".hermes"
    try:
        Path(hermes_home).resolve().relative_to(target_root.resolve())
        return _profile_arg(hermes_home, default_root=target_root)
    except ValueError:
        return _profile_arg(hermes_home)


def get_service_name() -> str:
    """Systemd service name: ``hermes-gateway`` for default HERMES_HOME, ``hermes-gateway-<profile>``
    or ``-<hash>`` otherwise."""
    suffix = _profile_suffix()
    if not suffix:
        return _SERVICE_BASE
    return f"{_SERVICE_BASE}-{suffix}"


def get_systemd_unit_path(system: bool = False) -> Path:
    name = get_service_name()
    if system:
        return Path("/etc/systemd/system") / f"{name}.service"
    return Path.home() / ".config" / "systemd" / "user" / f"{name}.service"


class UserSystemdUnavailableError(RuntimeError):
    """``systemctl --user`` cannot reach the user D-Bus session (fresh SSH sessions with linger off,
    so ``/run/user/$UID/bus`` never exists). ``args[0]`` is a user-facing remediation message."""


class SystemScopeRequiresRootError(RuntimeError):
    """System-scope gateway operation attempted as non-root.

    Typed (instead of ``sys.exit(1)``) so the setup wizard can print remediation instead of dying at
    a bare shell; ``gateway_command`` still exits 1. ``args[0]`` is the message, ``args[1]`` the
    action; ``str(e)`` returns only the message so ``f"Failed: {e}"`` renders cleanly.
    """

    def __str__(self) -> str:
        return self.args[0] if self.args else ""


def _user_dbus_socket_path() -> Path:
    """Return the expected per-user D-Bus socket path (regardless of existence)."""
    xdg = os.environ.get("XDG_RUNTIME_DIR") or f"/run/user/{os.getuid()}"  # windows-footgun: ok — POSIX systemd helper, never invoked on Windows
    return Path(xdg) / "bus"


def _user_systemd_private_socket_path() -> Path:
    """Return the per-user systemd private socket path (regardless of existence)."""
    xdg = os.environ.get("XDG_RUNTIME_DIR") or f"/run/user/{os.getuid()}"  # windows-footgun: ok — POSIX systemd helper, never invoked on Windows
    return Path(xdg) / "systemd" / "private"


def _path_exists_safe(path: Path) -> bool:
    """``Path.exists()`` that treats an inaccessible path as absent.

    ``Path.exists()`` lets ``EACCES`` propagate; a leaked ``XDG_RUNTIME_DIR`` from another user
    (``su``/``sudo -u`` from root, ``/run/user/0`` is 0700) would otherwise crash the preflight.
    """
    try:
        return path.exists()
    except OSError:  # e.g. EACCES on another user's runtime dir
        return False


def _runtime_dir_is_ours(runtime_dir: str) -> bool:
    """True when *runtime_dir* exists and is owned by our uid (a leaked foreign XDG_RUNTIME_DIR must not be trusted)."""
    try:
        return Path(runtime_dir).stat().st_uid == os.getuid()  # windows-footgun: ok — POSIX systemd helper, never invoked on Windows
    except OSError:
        return False


def _user_systemd_socket_ready() -> bool:
    """True when either the user D-Bus socket or the per-user systemd private socket exists.

    Some distros expose only the private socket and ``systemctl --user`` still works, so either
    counts. An inaccessible path counts as not-ready (falls through to UserSystemdUnavailableError).
    """
    return (
        _path_exists_safe(_user_dbus_socket_path())
        or _path_exists_safe(_user_systemd_private_socket_path())
    )


def _ensure_user_systemd_env() -> None:
    """Set XDG_RUNTIME_DIR / DBUS_SESSION_BUS_ADDRESS so ``systemctl --user`` works on headless hosts.

    Without them (SSH sessions, even with linger) systemctl fails "Failed to connect to bus". An
    XDG_RUNTIME_DIR leaked from another user is replaced with our own ``/run/user/{uid}``.
    """
    uid = os.getuid()  # windows-footgun: ok — POSIX systemd helper, never invoked on Windows
    xdg = os.environ.get("XDG_RUNTIME_DIR")
    if not xdg or not _runtime_dir_is_ours(xdg):
        runtime_dir = f"/run/user/{uid}"
        if _runtime_dir_is_ours(runtime_dir):
            os.environ["XDG_RUNTIME_DIR"] = runtime_dir

    if "DBUS_SESSION_BUS_ADDRESS" not in os.environ:
        xdg_runtime = os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{uid}")
        bus_path = Path(xdg_runtime) / "bus"
        if _path_exists_safe(bus_path):
            os.environ["DBUS_SESSION_BUS_ADDRESS"] = f"unix:path={bus_path}"


def _wait_for_user_dbus_socket(timeout: float = 3.0) -> bool:
    """Poll up to ``timeout`` s for a user systemd control socket (user@.service takes a moment after enable-linger)."""
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _user_systemd_socket_ready():
            _ensure_user_systemd_env()
            return True
        time.sleep(0.2)
    return _user_systemd_socket_ready()


def _preflight_user_systemd(*, auto_enable_linger: bool = True) -> None:
    """Ensure ``systemctl --user`` can reach user-scope systemd; raise UserSystemdUnavailableError otherwise.

    No-op when a control socket exists. Else: wait briefly if linger is on; if off and
    ``auto_enable_linger``, try ``loginctl enable-linger`` (non-root works when polkit permits).
    Callers should treat the exception as terminal for user-scope operations.
    """
    _ensure_user_systemd_env()
    if _user_systemd_socket_ready():
        return

    import getpass

    username = getpass.getuser()
    linger_enabled, linger_detail = get_systemd_linger_status()

    if linger_enabled is True:
        if _wait_for_user_dbus_socket(timeout=3.0):
            return
        # Linger is on but socket still missing — unusual; fall through to error.
        _raise_user_systemd_unavailable(
            username,
            reason="User systemd control sockets are missing even though linger is enabled.",
            fix_hint=(
                f"  systemctl start user@{os.getuid()}.service\n"  # windows-footgun: ok — POSIX systemd helper, never invoked on Windows
                "  (may require sudo; try again after the command succeeds)"
            ),
        )

    if auto_enable_linger and shutil.which("loginctl"):
        try:
            result = subprocess.run(
                ["loginctl", "enable-linger", username],
                check=False,
                timeout=30,
                **_CAPTURE_TEXT,
            )
        except Exception as exc:
            _raise_user_systemd_unavailable(
                username,
                reason=f"loginctl enable-linger failed ({exc}).",
                fix_hint=f"  sudo loginctl enable-linger {username}",
            )
        else:
            if result.returncode == 0:
                if _wait_for_user_dbus_socket(timeout=5.0):
                    print(f"✓ Enabled linger for {username} — user D-Bus now available")
                    return
                # enable-linger succeeded but the socket never appeared.
                _raise_user_systemd_unavailable(
                    username,
                    reason="Linger was enabled, but the user D-Bus socket did not appear.",
                    fix_hint=(
                        "  Log out and log back in, then re-run the command.\n"
                        f"  Or reboot and run: systemctl --user start {get_service_name()}"
                    ),
                )
            detail = (result.stderr or result.stdout or f"exit {result.returncode}").strip()
            _raise_user_systemd_unavailable(
                username,
                reason=f"loginctl enable-linger was denied: {detail}",
                fix_hint=f"  sudo loginctl enable-linger {username}",
            )

    _raise_user_systemd_unavailable(
        username,
        reason=("User D-Bus session is not available " f"({linger_detail or 'linger disabled'})."),
        fix_hint=f"  sudo loginctl enable-linger {username}",
    )


def _raise_user_systemd_unavailable(username: str, *, reason: str, fix_hint: str) -> None:
    """Build a user-facing error message and raise UserSystemdUnavailableError."""
    msg = (
        f"{reason}\n"
        "  systemctl --user cannot reach the user D-Bus session in this shell.\n"
        "\n"
        "  To fix:\n"
        f"{fix_hint}\n"
        "\n"
        "  Alternative: run the gateway in the foreground (stays up until\n"
        "  you exit / close the terminal):\n"
        "    hermes gateway run"
    )
    raise UserSystemdUnavailableError(msg)


def _systemctl_cmd(system: bool = False) -> list[str]:
    if not system:
        _ensure_user_systemd_env()
    return ["systemctl"] if system else ["systemctl", "--user"]


def _journalctl_cmd(system: bool = False) -> list[str]:
    return ["journalctl"] if system else ["journalctl", "--user"]


def _run_systemctl(args: list[str], *, system: bool = False, **kwargs) -> subprocess.CompletedProcess:
    """Run systemctl; raise RuntimeError (not raw FileNotFoundError) if missing, for callers bypassing
    ``supports_systemd_services()``."""
    try:
        return subprocess.run(_systemctl_cmd(system) + args, **kwargs)
    except FileNotFoundError:
        raise RuntimeError("systemctl is not available on this system") from None


def _service_scope_label(system: bool = False) -> str:
    return "system" if system else "user"


def get_installed_systemd_scopes() -> list[str]:
    scopes = []
    seen_paths: set[Path] = set()
    for system, label in ((False, "user"), (True, "system")):
        unit_path = get_systemd_unit_path(system=system)
        if unit_path in seen_paths:
            continue
        if unit_path.exists():
            scopes.append(label)
            seen_paths.add(unit_path)
    return scopes


def has_conflicting_systemd_units() -> bool:
    return len(get_installed_systemd_scopes()) > 1


# Legacy pre-rename service names. Explicit allowlist (NOT a glob) so profile units
# (hermes-gateway-*.service) and unrelated third-party "hermes" units never match.
_LEGACY_SERVICE_NAMES: tuple[str, ...] = ("hermes.service",)

# ExecStart content markers that identify a unit as running our gateway.
# A legacy unit is only flagged when its file contains one of these.
_LEGACY_UNIT_EXECSTART_MARKERS: tuple[str, ...] = (
    "hermes_cli.main gateway",
    "hermes_cli/main.py gateway",
    "gateway/run.py",
    " hermes gateway ",
    "/hermes gateway ",
)


def _legacy_unit_search_paths() -> list[tuple[bool, Path]]:
    """``[(is_system, base_dir), ...]`` to scan for legacy units; factored out so tests can monkeypatch."""
    return [(False, Path.home() / ".config" / "systemd" / "user"), (True, Path("/etc/systemd/system"))]


def _find_legacy_hermes_units() -> list[tuple[str, Path, bool]]:
    """Return ``[(unit_name, unit_path, is_system)]`` for legacy gateway units (e.g. ``hermes.service``).

    A legacy unit running alongside ``hermes-gateway.service`` fights over the same bot token
    (30s SIGTERM flap loop). Guards: explicit name allowlist (no globbing, so profile and
    third-party units never match), ExecStart marker check (an unrelated ``hermes.service`` is
    left alone), and no mutation — results are for caller inspection only.
    """
    results: list[tuple[str, Path, bool]] = []
    for is_system, base in _legacy_unit_search_paths():
        for name in _LEGACY_SERVICE_NAMES:
            unit_path = base / name
            try:
                if not unit_path.exists():
                    continue
                text = unit_path.read_text(encoding="utf-8", errors="ignore")
            except (OSError, PermissionError):
                continue
            if not any(marker in text for marker in _LEGACY_UNIT_EXECSTART_MARKERS):
                # Not our gateway — leave alone
                continue
            results.append((name, unit_path, is_system))
    return results


def has_legacy_hermes_units() -> bool:
    """Return True when any legacy Hermes gateway unit files exist."""
    return bool(_find_legacy_hermes_units())


def print_legacy_unit_warning() -> None:
    """Warn about installed legacy gateway units; prints nothing when there are none."""
    legacy = _find_legacy_hermes_units()
    if not legacy:
        return
    print_warning("Legacy Hermes gateway unit(s) detected from an older install:")
    for name, path, is_system in legacy:
        scope = "system" if is_system else "user"
        print_info(f"    {path}  ({scope} scope)")
    print_info("  These run alongside the current hermes-gateway service and")
    print_info("  cause SIGTERM flap loops — both try to use the same bot token.")
    print_info("  Remove them with:")
    print_info("    hermes gateway migrate-legacy")


def remove_legacy_hermes_units(interactive: bool = True, dry_run: bool = False) -> tuple[int, list[Path]]:
    """Stop, disable, and remove legacy gateway units found by ``_find_legacy_hermes_units()``.

    ``interactive=False`` skips the prompt (caller already confirmed); ``dry_run`` only lists.
    Returns ``(removed_count, remaining_paths)``; remaining includes units we couldn't remove
    (typically system-scope when not root).
    """
    legacy = _find_legacy_hermes_units()
    if not legacy:
        print("No legacy Hermes gateway units found.")
        return 0, []

    user_units = [(n, p) for n, p, is_sys in legacy if not is_sys]
    system_units = [(n, p) for n, p, is_sys in legacy if is_sys]

    print()
    print("Legacy Hermes gateway unit(s) found:")
    for name, path, is_system in legacy:
        scope = "system" if is_system else "user"
        print(f"  {path}  ({scope} scope)")
    print()

    if dry_run:
        print("(dry-run — nothing removed)")
        return 0, [p for _, p, _ in legacy]

    if interactive and not prompt_yes_no("Remove these legacy units?", True):
        print("Skipped. Run again with: hermes gateway migrate-legacy")
        return 0, [p for _, p, _ in legacy]

    removed = 0
    remaining: list[Path] = []

    def _remove_units(units: list[tuple[str, Path]], *, system: bool) -> None:
        nonlocal removed
        for name, path in units:
            try:
                _run_systemctl(["stop", name], system=system, check=False, timeout=90)
                _run_systemctl(["disable", name], system=system, check=False, timeout=30)
                path.unlink(missing_ok=True)
                print(f"  ✓ Removed {path}")
                removed += 1
            except (OSError, RuntimeError) as e:
                print(f"  ⚠ Could not remove {path}: {e}")
                remaining.append(path)
        with contextlib.suppress(RuntimeError):
            _run_systemctl(["daemon-reload"], system=system, check=False, timeout=30)

    if user_units:
        _remove_units(user_units, system=False)

    # System-scope removal (needs root)
    if system_units:
        if os.geteuid() != 0:  # windows-footgun: ok — Linux systemd removal path, guarded by `if system == "Linux"` / systemd-only branch
            print()
            print_warning("System-scope legacy units require root to remove.")
            print_info("  Re-run with: sudo hermes gateway migrate-legacy")
            remaining.extend(path for _, path in system_units)
        else:
            _remove_units(system_units, system=True)

    print()
    if remaining:
        print_warning(f"{len(remaining)} legacy unit(s) still present — see messages above.")
    else:
        print_success(f"Removed {removed} legacy unit(s).")

    return removed, remaining


def print_systemd_scope_conflict_warning() -> None:
    scopes = get_installed_systemd_scopes()
    if len(scopes) < 2:
        return

    rendered_scopes = " + ".join(scopes)
    print_warning(f"Both user and system gateway services are installed ({rendered_scopes}).")
    print_info("  This is confusing and can make start/stop/status behavior ambiguous.")
    print_info("  Default gateway commands target the user service unless you pass --system.")
    print_info("  Keep one of these:")
    print_info("    hermes gateway uninstall")
    print_info("    sudo hermes gateway uninstall --system")


def _require_root_for_system_service(action: str) -> None:
    if os.geteuid() != 0:  # windows-footgun: ok — POSIX systemd helper, never invoked on Windows
        raise SystemScopeRequiresRootError(
            f"System gateway {action} requires root. Re-run with sudo.",
            action,
        )


def _system_service_identity(run_as_user: str | None = None) -> tuple[str, str, str]:
    import getpass
    import grp
    import pwd

    username = (
        run_as_user
        or os.getenv("SUDO_USER")
        or os.getenv("USER")
        or os.getenv("LOGNAME")
        or getpass.getuser()
    ).strip()
    if not username:
        raise ValueError("Could not determine which user the gateway service should run as")
    if username == "root" and not run_as_user:
        raise ValueError(
            "Refusing to install the gateway system service as root; pass --run-as-user root to override (e.g. in LXC containers)"
        )
    if username == "root":
        print_warning("Installing gateway service to run as root.")
        print_info("  This is fine for LXC/container environments but not recommended on bare-metal hosts.")

    try:
        user_info = pwd.getpwnam(username)
    except KeyError as e:
        raise ValueError(f"Unknown user: {username}") from e

    group_name = grp.getgrgid(user_info.pw_gid).gr_name
    return username, group_name, user_info.pw_dir


def _read_systemd_user_from_unit(unit_path: Path) -> str | None:
    if not unit_path.exists():
        return None

    for line in unit_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("User="):
            value = line.split("=", 1)[1].strip()
            return value or None
    return None


def _default_system_service_user() -> str | None:
    for candidate in (os.getenv("SUDO_USER"), os.getenv("USER"), os.getenv("LOGNAME")):
        if candidate and candidate.strip() and candidate.strip() != "root":
            return candidate.strip()
    return None


def prompt_linux_gateway_install_scope() -> str | None:
    # Only root can create a boot-time system service, so that scope is offered only to root
    # sessions — a non-root user is never handed a "re-run under sudo" recipe.
    is_root = os.geteuid() == 0  # windows-footgun: ok — Linux systemd install wizard, never invoked on Windows
    if not is_root:
        choice = prompt_choice(
            "  Choose how the gateway should run in the background:",
            [
                "User service (no sudo; best for laptops/dev boxes; may need linger after logout)",
                "Skip service install for now",
            ],
            default=0,
        )
        if choice == 0:
            print_info(
                "  Tip: for a boot-time system service, re-run setup as root "
                "(e.g. from a root shell or `sudo -i`)."
            )
        return {0: "user", 1: None}[choice]

    choice = prompt_choice(
        "  Choose how the gateway should run in the background:",
        [
            "User service (no sudo; best for laptops/dev boxes; may need linger after logout)",
            "System service (starts on boot; runs as your chosen user)",
            "Skip service install for now",
        ],
        default=0,
    )
    return {0: "user", 1: "system", 2: None}[choice]


def install_linux_gateway_from_setup(force: bool = False, enable_on_startup: bool = True) -> tuple[str | None, bool]:
    scope = prompt_linux_gateway_install_scope()
    if scope is None:
        return None, False

    if scope == "system":
        run_as_user = _default_system_service_user()
        if os.geteuid() != 0:  # windows-footgun: ok — Linux systemd install wizard, never invoked on Windows
            # Unreachable from the wizard (system scope only offered to root). Defensive
            # guard for direct callers — no self-elevation recipe is printed.
            print_warning(
                "  System service install requires root. Re-run setup from a "
                "root shell, or install a user service instead: hermes gateway install"
            )
            return scope, False

        if not run_as_user:
            while True:
                run_as_user = prompt("  Run the system gateway service as which user?", default="")
                run_as_user = (run_as_user or "").strip()
                if run_as_user:
                    break
                print_error("  Enter a username.")

        systemd_install(force=force, system=True, run_as_user=run_as_user, enable_on_startup=enable_on_startup)
        return scope, True

    systemd_install(force=force, system=False, enable_on_startup=enable_on_startup)
    return scope, True


def ensure_gateway_service(context: str = "setup") -> bool:
    """Install and start a user-scope gateway service without prompting (``hermes setup``/``import``).

    A gateway with zero platforms is a supported degraded mode (cron runs, platforms picked up as
    tokens appear), so this never gates on messaging config. Never prompts, never raises; returns
    True when a service is installed and running.
    """
    from hermes_constants import is_container

    if is_container():
        # Containers use restart policies, not service managers.
        print_info("Start the gateway to bring your bots online:")
        print_info("   hermes gateway run          # Run as container main process")
        print_info("")
        print_info("For automatic restarts, use a Docker restart policy:")
        print_info("   docker run --restart unless-stopped ...")
        return False

    supports_systemd = supports_systemd_services()
    if not (supports_systemd or is_macos() or is_windows()):
        print_info("  No supported service manager found on this host.")
        print_info("  Run the gateway in the foreground with: hermes gateway")
        return False

    try:
        if _is_service_running():
            return True

        if not _is_service_installed():
            if supports_systemd and has_conflicting_systemd_units():
                # Both user and system units would fight over bot tokens.
                # Don't pile a fresh install onto a conflicted state.
                print_systemd_scope_conflict_warning()
                return False
            print_info("  Installing the gateway background service ...")
            if supports_systemd:
                systemd_install(force=False, non_interactive=True)
            elif is_macos():
                launchd_install(force=False)
            else:
                # Registers the Scheduled Task AND starts it.
                _gw_windows().install(force=False)
                print_success("  Gateway service installed and started.")
                return True

        if supports_systemd:
            systemd_start()
        elif is_macos():
            launchd_start()
        else:
            _gw_windows().start()
        print_success("  Gateway service running (cron jobs + messaging platforms).")
        return True
    except UserSystemdUnavailableError as e:
        print_warning("  Could not reach user systemd to start the gateway service:")
        _print_indented(str(e), print_info)
    except SystemScopeRequiresRootError as e:
        print_warning(f"  Gateway service needs root for this scope: {e}")
        _print_system_scope_remediation("start")
    except SystemExit:
        # Some install/start paths sys.exit() on hard failures (e.g. temp-HOME
        # guard). A background-service failure must never abort setup/import.
        print_warning("  Gateway service install did not complete.")
        print_info("  You can retry manually: hermes gateway install")
    except Exception as e:
        print_warning(f"  Gateway service install failed: {e}")
        print_info("  You can retry manually: hermes gateway install")
    return False


def get_systemd_linger_status() -> tuple[bool | None, str]:
    """Linger status for the current user: ``(True, "")``, ``(False, "")``, or ``(None, detail)`` when unknown."""
    if is_termux():
        return None, "not supported in Termux"
    if not is_linux():
        return None, "not supported on this platform"

    if not shutil.which("loginctl"):
        return None, "loginctl not found"

    username = os.getenv("USER") or os.getenv("LOGNAME")
    if not username:
        try:
            import pwd

            username = pwd.getpwuid(os.getuid()).pw_name  # windows-footgun: ok — POSIX loginctl helper, never invoked on Windows
        except Exception:
            return None, "could not determine current user"

    try:
        result = subprocess.run(
            ["loginctl", "show-user", username, "--property=Linger", "--value"],
            check=False,
            timeout=10,
            **_CAPTURE_TEXT,
        )
    except Exception as e:
        return None, str(e)

    if result.returncode != 0:
        detail = (result.stderr or result.stdout or f"exit {result.returncode}").strip()
        return None, detail or "loginctl query failed"

    value = (result.stdout or "").strip().lower()
    if value in {"yes", "true", "1"}:
        return True, ""
    if value in {"no", "false", "0"}:
        return False, ""

    rendered = value or "<empty>"
    return None, f"unexpected loginctl output: {rendered}"


def print_systemd_linger_guidance() -> None:
    """Print the current linger status and the fix when it is disabled."""
    linger_enabled, linger_detail = get_systemd_linger_status()
    if linger_enabled is True:
        print("✓ Systemd linger is enabled (service survives logout)")
    elif linger_enabled is False:
        print("⚠ Systemd linger is disabled (gateway may stop when you log out)")
        print("  Run: sudo loginctl enable-linger $USER")
    else:
        print(f"⚠ Could not verify systemd linger ({linger_detail})")
        print("  If you want the gateway user service to survive logout, run:")
        print("  sudo loginctl enable-linger $USER")


def _launchd_user_home() -> Path:
    """Real macOS account home for launchd artifacts (profile mode may point HOME at a profile dir)."""
    import pwd

    return Path(pwd.getpwuid(os.getuid()).pw_dir)  # windows-footgun: ok — POSIX launchd (macOS) helper, never invoked on Windows


def get_launchd_plist_path() -> Path:
    """launchd plist path: ``ai.hermes.gateway.plist`` for default HERMES_HOME,
    ``ai.hermes.gateway-<profile>.plist`` otherwise."""
    suffix = _profile_suffix()
    name = f"ai.hermes.gateway-{suffix}" if suffix else "ai.hermes.gateway"
    return _launchd_user_home() / "Library" / "LaunchAgents" / f"{name}.plist"


def launchd_gateway_labels_for_install() -> list[str]:
    """Launchd gateway labels for every profile of THIS install: root label first, then profiles by name.

    Derived from the install's profile layout, NOT by globbing the shared ``~/Library/LaunchAgents``,
    so a sandboxed HERMES_HOME never enumerates/restarts another install's fleet. Profiles whose
    names can't map to a service suffix are skipped; uninstalled profiles are harmless to include.
    """
    import re as _re

    from hermes_cli.profiles import list_profiles

    root_label: list[str] = []
    profile_labels: list[str] = []
    for profile in list_profiles():
        if profile.is_default:
            root_label.append("ai.hermes.gateway")
        elif _re.match(r"^[a-z0-9][a-z0-9_-]{0,63}$", profile.name):
            profile_labels.append(f"ai.hermes.gateway-{profile.name}")
    return root_label + sorted(profile_labels)


def _detect_venv_dir() -> Path | None:
    """Active virtualenv dir: ``sys.prefix``, then ``VIRTUAL_ENV`` (uv sets it without changing
    sys.prefix), then .venv/venv under PROJECT_ROOT; None if none found."""
    # If we're running inside a virtualenv, sys.prefix points to it.
    if sys.prefix != sys.base_prefix:
        venv = Path(sys.prefix)
        if venv.is_dir():
            return venv

    # uv and some other tools set VIRTUAL_ENV without changing sys.prefix. This catches `uv run`
    # where sys.prefix == sys.base_prefix but the environment IS a venv.
    _virtual_env = os.environ.get("VIRTUAL_ENV")
    if _virtual_env:
        venv = Path(_virtual_env)
        if venv.is_dir():
            return venv

    # Fallback: check common virtualenv directory names under the project root.
    for candidate in (".venv", "venv"):
        venv = PROJECT_ROOT / candidate
        if venv.is_dir():
            return venv

    return None


def get_python_path() -> str:
    venv = _detect_venv_dir()
    if venv is not None:
        try:
            from hermes_constants import venv_python_path
        except ImportError:
            # Update-boundary: a gateway restarted mid-update can hold a stale hermes_constants
            # without this symbol; see _reload_hermes_constants() in hermes_cli/managed_uv.py.
            from hermes_cli.managed_uv import _reload_hermes_constants

            venv_python_path = _reload_hermes_constants().venv_python_path

        venv_python = venv_python_path(venv, windows=is_windows())
        if venv_python.exists():
            return str(venv_python)
    return sys.executable


# =============================================================================
# Systemd (Linux)
# =============================================================================


def _build_user_local_paths(home: Path, path_entries: list[str]) -> list[str]:
    """Return user-local bin dirs that exist and aren't already in *path_entries*."""
    candidates = [
        str(home / ".local" / "bin"),  # uv, uvx, pip-installed CLIs
        str(home / ".cargo" / "bin"),  # Rust/cargo tools
        str(home / "go" / "bin"),  # Go tools
        str(home / ".npm-global" / "bin"),  # npm global packages
    ]
    return [p for p in candidates if p not in path_entries and Path(p).exists()]


def _build_wsl_interop_paths(path_entries: list[str]) -> list[str]:
    """WSL Windows-interop PATH entries for generated units: systemd services don't inherit the
    Windows PATH (``/mnt/c/WINDOWS/System32``…), so ``powershell.exe``/``cmd.exe`` break unless persisted."""
    if not is_wsl():
        return []

    candidates: list[str] = []
    for entry in os.environ.get("PATH", "").split(os.pathsep):
        if entry.startswith("/mnt/"):
            candidates.append(entry)

    for executable in ("powershell.exe", "cmd.exe", "explorer.exe", "wsl.exe"):
        resolved = shutil.which(executable)
        if resolved:
            candidates.append(str(Path(resolved).parent))

    for entry in (
        "/mnt/c/WINDOWS/system32",
        "/mnt/c/WINDOWS",
        "/mnt/c/WINDOWS/System32/Wbem",
        "/mnt/c/WINDOWS/System32/WindowsPowerShell/v1.0/",
        "/mnt/c/WINDOWS/System32/OpenSSH/",
    ):
        if Path(entry).exists():
            candidates.append(entry)

    result: list[str] = []
    seen = set(path_entries)
    for entry in candidates:
        if entry and entry not in seen:
            seen.add(entry)
            result.append(entry)
    return result


def _remap_path_for_user(path: str, target_home_dir: str) -> str:
    """Swap the ``Path.home()`` prefix of *path* for *target_home_dir*; other paths return unchanged.
    Intentionally does NOT resolve symlinks."""
    current_home = Path.home()
    p = Path(path).expanduser()
    try:
        relative = p.relative_to(current_home)
        return str(Path(target_home_dir) / relative)
    except ValueError:
        return str(p)


def _hermes_home_for_target_user(target_home_dir: str) -> str:
    """Remap the current HERMES_HOME (root's, under sudo) to the target user's equivalent:
    ``/root/.hermes[/profiles/x]`` → ``/home/alice/.hermes[/profiles/x]``; custom paths kept as-is."""
    current_hermes_raw = os.environ.get("HERMES_HOME", "").strip()
    current_hermes = Path(current_hermes_raw).expanduser() if current_hermes_raw else get_hermes_home()
    # Keep custom paths lexical: resolving a non-existent path can rewrite it through
    # host-specific mappings and bake a different HERMES_HOME into the unit.
    current_default = Path.home() / ".hermes"
    target_default = Path(target_home_dir) / ".hermes"

    # Default ~/.hermes → remap to target user's default
    if current_hermes == current_default:
        return str(target_default)

    # Profile or subdir of ~/.hermes → preserve the relative structure
    try:
        relative = current_hermes.relative_to(current_default)
        return str(target_default / relative)
    except ValueError:
        # Completely custom path (not under ~/.hermes) — keep as-is
        return str(current_hermes)


def _build_service_path_dirs(project_root: Path | None = None) -> list[str]:
    """Build PATH directory list for service units, excluding non-existent dirs."""
    if project_root is None:
        project_root = PROJECT_ROOT

    def _is_dir(path: Path) -> bool:
        try:
            return path.is_dir()
        except OSError:
            return False

    candidates = []

    venv_bin = project_root / "venv" / "bin"
    if _is_dir(venv_bin):
        candidates.append(str(venv_bin))
    elif sys.prefix != sys.base_prefix:
        candidates.append(str(Path(sys.prefix) / "bin"))

    node_bin = project_root / "node_modules" / ".bin"
    if _is_dir(node_bin):
        candidates.append(str(node_bin))

    hermes_home = get_hermes_home()
    hermes_node = hermes_home / "node" / "bin"
    if _is_dir(hermes_node):
        candidates.append(str(hermes_node))
    hermes_nm = hermes_home / "node_modules" / ".bin"
    if _is_dir(hermes_nm):
        candidates.append(str(hermes_nm))

    return candidates


def _stable_service_working_dir() -> str:
    """WorkingDirectory that won't disappear under systemd (HERMES_HOME, else PROJECT_ROOT).

    ExecStart uses an absolute interpreter + ``-m``, so cwd is irrelevant to module resolution.
    Pinning PROJECT_ROOT is harmful: a transient checkout (worktree, relocated by ``hermes update``)
    rots, systemd fails at CHDIR (status=200) before Python loads, the on-boot unit self-heal never
    runs, and Restart=always crash-loops forever.
    """
    try:
        home = get_hermes_home()
        if home and Path(home).is_dir():
            return str(Path(home).resolve())
    except Exception:
        pass
    return str(PROJECT_ROOT)


def _systemd_watchdog_seconds(hermes_home: str | Path | None = None) -> int:
    """Resolve the managed-overlay-aware watchdog setting for a service home."""
    override_token = None
    reset_home_override = None
    if hermes_home is not None:
        from hermes_constants import (reset_hermes_home_override, set_hermes_home_override)

        override_token = set_hermes_home_override(hermes_home)
        reset_home_override = reset_hermes_home_override
    try:
        config = load_gateway_config()
        return coerce_systemd_watchdog_seconds(getattr(config, "systemd_watchdog_seconds", 0))
    except Exception:
        logger.debug("Could not resolve effective systemd watchdog configuration", exc_info=True)
        return 0
    finally:
        if override_token is not None and reset_home_override is not None:
            reset_home_override(override_token)


def _systemd_watchdog_service_fields(hermes_home: str | Path | None = None) -> tuple[str, str]:
    """Return systemd service fields for the effective gateway config."""
    seconds = _systemd_watchdog_seconds(hermes_home)
    if seconds <= 0:
        return "simple", ""
    return "notify", f"NotifyAccess=main\nWatchdogSec={seconds}s\n"


def _append_node_dir_for_service(path_entries: list[str], hermes_root: Path | None = None) -> None:
    """Append the Node dir a service unit should use to *path_entries*.

    Managed Node under ``<hermes_root>/node`` first: a unit is written once and survives reboots,
    so baking a system Node that merely leads the installing shell's PATH is permanent breakage.
    Managed dirs are profile-scoped. PATH lookup is only the fallback when no managed Node exists.
    """
    from hermes_constants import (hermes_managed_node_tree_present, iter_hermes_node_dirs)

    managed_node_present = hermes_managed_node_tree_present(hermes_root)
    for directory in iter_hermes_node_dirs(hermes_root) if managed_node_present else ():
        entry = str(directory)
        try:
            present = directory.is_dir()
        except OSError:
            present = False
        if present and entry not in path_entries:
            path_entries.append(entry)

    # PATH is a fallback, not an extra rung: with managed Node present, consulting the invoker's
    # PATH would make a system unit differ between sudo/root and its service user.
    if managed_node_present:
        return

    resolved_node = shutil.which("node")
    if not resolved_node:
        return

    # Use the dir where node is FOUND on PATH, not the symlink target: ~/.local/bin/node often
    # links into one profile's node install, and resolving would bake it into every profile's unit.
    resolved_node_dir = str(Path(resolved_node).parent)
    if resolved_node_dir not in path_entries:
        path_entries.append(resolved_node_dir)


def generate_systemd_unit(system: bool = False, run_as_user: str | None = None) -> str:
    python_path = get_python_path()
    working_dir = _stable_service_working_dir()
    detected_venv = _detect_venv_dir()
    venv_dir = str(detected_venv) if detected_venv else str(PROJECT_ROOT / "venv")

    path_entries = _build_service_path_dirs()
    if not system:
        # System units add managed Node later, once the TARGET user's home is known —
        # probing here would bake the calling (sudo → root) user's Node into the unit.
        _append_node_dir_for_service(path_entries)

    common_bin_paths = ["/usr/local/sbin", "/usr/local/bin", "/usr/sbin", "/usr/bin", "/sbin", "/bin"]
    # TimeoutStopSec must cover the full stop budget: cron work may wait cron_drain_timeout plus
    # cleanup reserve, and systemd SIGKILLs past the deadline. +30s headroom, 60s floor.
    restart_timeout = resolve_systemd_timeout_stop_sec(
        _get_restart_drain_timeout(),
        _get_cron_drain_timeout(),
    )

    if system:
        username, group_name, home_dir = _system_service_identity(run_as_user)
        hermes_home = _hermes_home_for_target_user(home_dir)
        profile_arg = _profile_arg_for_target_user(hermes_home, home_dir)
        # Remap all paths that may resolve under the calling user's home (e.g. /root/) to the target
        # user's home so the service can actually access them.
        python_path = _remap_path_for_user(python_path, home_dir)
        # Anchor cwd to the target user's HERMES_HOME (stable) rather than a remapped checkout path that can rot.
        working_dir = str(hermes_home) if hermes_home else _remap_path_for_user(working_dir, home_dir)
        venv_dir = _remap_path_for_user(venv_dir, home_dir)
        path_entries = [_remap_path_for_user(p, home_dir) for p in path_entries]
        # Managed Node for the TARGET user's tree (probe the remapped hermes_home). Prepend so it
        # outranks remapped shell-PATH entries, matching the user-unit ordering.
        _target_node_entries: list[str] = []
        _append_node_dir_for_service(_target_node_entries, Path(hermes_home) if hermes_home else None)
        path_entries = [e for e in _target_node_entries if e not in path_entries] + path_entries
        user_home = Path(home_dir)
        identity_lines = f"User={username}\nGroup={group_name}\n"
        env_lines = (
            f'Environment="HOME={home_dir}"\n'
            f'Environment="USER={username}"\n'
            f'Environment="LOGNAME={username}"\n'
        )
        wanted_by = "multi-user.target"
    else:
        hermes_home = str(get_hermes_home().resolve())
        profile_arg = _profile_arg(hermes_home)
        user_home = Path.home()
        identity_lines = env_lines = ""
        wanted_by = "default.target"

    systemd_type, systemd_watchdog_directives = _systemd_watchdog_service_fields(hermes_home)
    path_entries.extend(_build_user_local_paths(user_home, path_entries))
    path_entries.extend(_build_wsl_interop_paths(path_entries))
    path_entries.extend(common_bin_paths)
    sane_path = ":".join(path_entries)
    return f"""[Unit]
Description={SERVICE_DESCRIPTION}
After=network-online.target
Wants=network-online.target
StartLimitIntervalSec=0

[Service]
Type={systemd_type}
{systemd_watchdog_directives}{identity_lines}ExecStart={python_path} -m hermes_cli.main{f" {profile_arg}" if profile_arg else ""} gateway run
WorkingDirectory={working_dir}
{env_lines}Environment="PATH={sane_path}"
Environment="VIRTUAL_ENV={venv_dir}"
Environment="HERMES_HOME={hermes_home}"
Environment="HERMES_SUPERVISED_CHILD=1"
Restart=always
RestartSec=5
RestartForceExitStatus={GATEWAY_SERVICE_RESTART_EXIT_CODE}
RestartPreventExitStatus={GATEWAY_FATAL_CONFIG_EXIT_CODE}
KillMode=mixed
KillSignal=SIGTERM
ExecReload=/bin/kill -USR1 $MAINPID
ExecStopPost=-{python_path} -m gateway.cgroup_cleanup
TimeoutStopSec={restart_timeout}
StandardOutput=journal
StandardError=journal

[Install]
WantedBy={wanted_by}
"""


def _normalize_service_definition(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.strip().splitlines())


# Directives older systemd silently strips; normalized out of stale-check comparisons so a
# unit differing only by these isn't perpetually flagged outdated.
_SYSTEMD_OPTIONAL_DIRECTIVES = ("RestartMaxDelaySec", "RestartSteps")


def _strip_optional_systemd_directives(text: str) -> str:
    """Remove systemd directives that older hosts silently drop."""
    lines = text.splitlines()
    filtered = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            key = stripped.split("=", 1)[0].strip()
            if key in _SYSTEMD_OPTIONAL_DIRECTIVES:
                continue
        filtered.append(line)
    return "\n".join(filtered)


def _normalize_launchd_plist_for_comparison(text: str) -> str:
    """Normalize plist text for staleness checks, ignoring the PATH payload: the generated PATH is
    captured from the invoking shell and varies across shells."""
    import re

    normalized = _normalize_service_definition(text)
    return re.sub(
        r"(<key>PATH</key>\s*<string>)(.*?)(</string>)",
        r"\1__HERMES_PATH__\3",
        normalized,
        flags=re.S,
    )


def systemd_unit_is_current(system: bool = False) -> bool:
    # HERMES_HOME sync chokepoint: every compare/regenerate path funnels through here
    # (refresh_systemd_unit_if_needed, systemd_status, systemd_install), so the operator's pinned
    # home is adopted before any compare at a single site. Under `sudo … --system` HERMES_HOME is
    # often stripped to /root/.hermes; without this, refresh rewrites a correct unit from root's
    # defaults and status warns forever. The sync is idempotent and its os.environ mutation
    # persists for later runtime reads (e.g. systemd_restart's get_running_pid / drain timeout).
    _sync_hermes_home_from_systemd_unit(system=system)

    unit_path = get_systemd_unit_path(system=system)
    if not unit_path.exists():
        return False

    installed = unit_path.read_text(encoding="utf-8")
    expected_user = _read_systemd_user_from_unit(unit_path) if system else None
    expected = generate_systemd_unit(system=system, run_as_user=expected_user)
    # Ignore directives older systemd drops (RestartMaxDelaySec, RestartSteps) to avoid a perpetual "outdated" flag.
    norm_installed = _normalize_service_definition(_strip_optional_systemd_directives(installed))
    norm_expected = _normalize_service_definition(_strip_optional_systemd_directives(expected))
    return norm_installed == norm_expected


def _temp_home_in_service_definition(definition: str) -> str | None:
    """Return the temp-dir HERMES_HOME baked into a systemd unit / launchd plist, or None.

    A temp HERMES_HOME means a test/E2E harness generated the definition; writing it to the real
    service file leaves the gateway "active (running)" but pointed at an empty home, deaf to every
    platform. Matches ``Environment="HERMES_HOME=..."`` and ``<key>HERMES_HOME</key><string>``.
    """
    import re
    import tempfile

    candidates = re.findall(r'HERMES_HOME=([^"\n]+)', definition)
    candidates += re.findall(r"<key>HERMES_HOME</key>\s*<string>(.*?)</string>", definition, flags=re.S)
    temp_roots = {
        Path(tempfile.gettempdir()).resolve(),
        Path("/tmp"),
        Path("/var/tmp"),
        Path("/private/tmp"),
        Path("/private/var/tmp"),
    }
    for raw in candidates:
        try:
            resolved = Path(raw.strip().strip('"')).resolve()
        except (OSError, ValueError):
            continue
        for root in temp_roots:
            if resolved == root or root in resolved.parents:
                return raw.strip()
    return None


def _refuse_temp_home_service_write(definition: str, kind: str) -> bool:
    """Refuse (with guidance) when a service definition carries a temp HERMES_HOME."""
    temp_home = _temp_home_in_service_definition(definition)
    if temp_home is None:
        return False
    print(
        f"✗ Refusing to write the gateway {kind}: HERMES_HOME resolves to a "
        f"temporary directory ({temp_home})."
    )
    print(
        "  This usually means a test/E2E environment exported HERMES_HOME. "
        "Unset it (or run from a clean shell) and retry."
    )
    return True


def refresh_systemd_unit_if_needed(system: bool = False) -> bool:
    """Rewrite the installed systemd unit when the generated definition has changed."""
    unit_path = get_systemd_unit_path(system=system)
    if not unit_path.exists():
        return False

    # systemd_unit_is_current is the HERMES_HOME-sync chokepoint; its env mutation persists for the regenerate below.
    if systemd_unit_is_current(system=system):
        return False

    expected_user = _read_systemd_user_from_unit(unit_path) if system else None
    new_unit = generate_systemd_unit(system=system, run_as_user=expected_user)

    # Test-environment safety belt: the user unit path is under Path.home(), which the test
    # conftest does NOT sandbox (only HERMES_HOME is). A pytest-tmp HERMES_HOME baked into the
    # developer's real unit silently breaks their gateway on next reboot (all platforms "not
    # configured"). Sniffing the unit body keeps tests that patch generate_systemd_unit working.
    if not system and (
        "/pytest-of-" in new_unit
        or '/hermes_test"' in new_unit
        or "/hermes_test/" in new_unit
    ):
        return False

    # Structural variant: refuse ANY temp-dir HERMES_HOME (manual E2E homes lack the pytest markers).
    if _refuse_temp_home_service_write(new_unit, "systemd unit"):
        return False

    unit_path.write_text(new_unit, encoding="utf-8")
    _run_systemctl(["daemon-reload"], system=system, check=True, timeout=30)
    print(
        f"↻ Updated gateway {_service_scope_label(system)} service definition to match the current Hermes install"
    )
    return True


def _print_linger_enable_warning(username: str, detail: str | None = None) -> None:
    print()
    print("⚠ Linger not enabled — gateway may stop when you close this terminal.")
    if detail:
        print(f"  Auto-enable failed: {detail}")
    print()
    print("  On headless servers (VPS, cloud instances) run:")
    print(f"    sudo loginctl enable-linger {username}")
    print()
    print("  Then restart the gateway:")
    print(f"    systemctl --user restart {get_service_name()}.service")
    print()


def _ensure_linger_enabled() -> None:
    """Enable linger when possible so the user gateway survives logout."""
    if is_termux() or not is_linux():
        return

    import getpass

    username = getpass.getuser()
    linger_file = Path(f"/var/lib/systemd/linger/{username}")
    if linger_file.exists():
        print("✓ Systemd linger is enabled (service survives logout)")
        return

    linger_enabled, linger_detail = get_systemd_linger_status()
    if linger_enabled is True:
        print("✓ Systemd linger is enabled (service survives logout)")
        return

    if not shutil.which("loginctl"):
        _print_linger_enable_warning(username, linger_detail or "loginctl not found")
        return

    print("Enabling linger so the gateway survives SSH logout...")
    try:
        result = subprocess.run(
            ["loginctl", "enable-linger", username],
            check=False,
            timeout=30,
            **_CAPTURE_TEXT,
        )
    except Exception as e:
        _print_linger_enable_warning(username, str(e))
        return

    if result.returncode == 0:
        print("✓ Linger enabled — gateway will persist after logout")
        return

    detail = (result.stderr or result.stdout or f"exit {result.returncode}").strip()
    _print_linger_enable_warning(username, detail or linger_detail)


def _select_systemd_scope(system: bool = False) -> bool:
    if system:
        return True
    return get_systemd_unit_path(system=True).exists() and not get_systemd_unit_path(system=False).exists()


def _system_scope_wizard_would_need_root(system: bool = False) -> bool:
    """True when the wizard would trigger a system-scope operation as non-root — mirrors
    ``_select_systemd_scope`` so the dead-end is detected BEFORE prompting."""
    if os.geteuid() == 0:  # windows-footgun: ok — systemd scope wizard decision, never invoked on Windows
        return False
    return _select_systemd_scope(system=system)


def _print_system_scope_remediation(action: str) -> None:
    """Print remediation when the wizard skips a system-scope action because the user isn't root."""
    svc = get_service_name()
    print_warning(f"Gateway is installed as a system-wide service — " f"{action} requires root.")
    print_info("  Options:")
    print_info(f"    1. {action.capitalize()} it this time:")
    print_info(f"         sudo systemctl {action} {svc}")
    print_info("    2. Switch to a per-user service (recommended for personal use):")
    print_info("         sudo hermes gateway uninstall --system")
    print_info("         hermes gateway install")
    print_info("         hermes gateway start")


def _get_restart_drain_timeout() -> float:
    """Return the configured gateway restart drain timeout in seconds."""
    raw = os.getenv("HERMES_RESTART_DRAIN_TIMEOUT", "").strip()
    if not raw:
        cfg = read_raw_config()
        agent_cfg = cfg.get("agent", {}) if isinstance(cfg, dict) else {}
        raw = str(agent_cfg.get("restart_drain_timeout", DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT))
    return parse_restart_drain_timeout(raw)


def _agent_timeout_setting(env_var: str, key: str, parse) -> float:
    """``parse(env)`` when the env var is non-empty, else ``parse(agent.<key>)`` (None if unset)."""
    env_raw = os.getenv(env_var)
    if env_raw is not None and str(env_raw).strip() != "":
        return parse(env_raw)
    cfg = read_raw_config()
    agent_cfg = cfg.get("agent", {}) if isinstance(cfg, dict) else {}
    if isinstance(agent_cfg, dict) and key in agent_cfg:
        return parse(agent_cfg.get(key))
    return parse(None)


def _get_cron_drain_timeout() -> float:
    """Return the configured cron-only drain floor in seconds."""
    return _agent_timeout_setting("HERMES_CRON_DRAIN_TIMEOUT", "cron_drain_timeout", parse_cron_drain_timeout)


def _get_restart_after_turn_timeout() -> float:
    """Return the in-band restart wait-for-idle timeout in seconds."""
    return _agent_timeout_setting(
        "HERMES_RESTART_AFTER_TURN_TIMEOUT", "restart_after_turn_timeout", parse_restart_after_turn_timeout
    )


def _get_restart_exit_wait_budget() -> float:
    """CLI wait for gateway exit after SIGUSR1 / self-restart (#77184)."""
    return resolve_restart_exit_wait_budget(_get_restart_drain_timeout(), _get_restart_after_turn_timeout())


def systemd_install(
    force: bool = False,
    system: bool = False,
    run_as_user: str | None = None,
    enable_on_startup: bool = True,
    non_interactive: bool = False,
):
    if system:
        _require_root_for_system_service("install")

    # Offer to remove legacy units first: left alongside the new unit they flap-fight for the bot
    # token on every start. Only allowlisted names with our ExecStart signature are touched.
    if has_legacy_hermes_units():
        print()
        print_legacy_unit_warning()
        print()
        if non_interactive or prompt_yes_no("Remove the legacy unit(s) before installing?", True):
            remove_legacy_hermes_units(interactive=False)
            print()

    unit_path = get_systemd_unit_path(system=system)
    scope_flag = " --system" if system else ""

    # Existing system units already pin HERMES_HOME; adopt it before any regenerate.
    if unit_path.exists():
        _sync_hermes_home_from_systemd_unit(system=system)

    if unit_path.exists() and not force:
        if not systemd_unit_is_current(system=system):
            print(f"↻ Repairing outdated {_service_scope_label(system)} systemd service at: {unit_path}")
            refresh_systemd_unit_if_needed(system=system)
            if enable_on_startup:
                _run_systemctl(["enable", get_service_name()], system=system, check=True, timeout=30)
            print(f"✓ {_service_scope_label(system).capitalize()} service definition updated")
            return
        print(f"Service already installed at: {unit_path}")
        print("Use --force to reinstall")
        return

    unit_path.parent.mkdir(parents=True, exist_ok=True)
    new_unit = generate_systemd_unit(system=system, run_as_user=run_as_user)
    if _refuse_temp_home_service_write(new_unit, "systemd unit"):
        return
    print(f"Installing {_service_scope_label(system)} systemd service to: {unit_path}")
    unit_path.write_text(new_unit, encoding="utf-8")

    _run_systemctl(["daemon-reload"], system=system, check=True, timeout=30)
    if enable_on_startup:
        _run_systemctl(["enable", get_service_name()], system=system, check=True, timeout=30)

    print()
    enable_label = "installed and enabled" if enable_on_startup else "installed"
    print(f"✓ {_service_scope_label(system).capitalize()} service {enable_label}!")
    print()
    print("Next steps:")
    print(f"  {'sudo ' if system else ''}hermes gateway start{scope_flag}              # Start the service")
    print(f"  {'sudo ' if system else ''}hermes gateway status{scope_flag}             # Check status")
    print(f"  {'journalctl' if system else 'journalctl --user'} -u {get_service_name()} -f  # View logs")
    print()

    if system:
        configured_user = _read_systemd_user_from_unit(unit_path)
        if configured_user:
            print(f"Configured to run as: {configured_user}")
    else:
        _ensure_linger_enabled()

    print_systemd_scope_conflict_warning()
    print_legacy_unit_warning()


def systemd_uninstall(system: bool = False):
    system = _select_systemd_scope(system)
    if system:
        _require_root_for_system_service("uninstall")

    _run_systemctl(["stop", get_service_name()], system=system, check=False, timeout=90)
    _run_systemctl(["disable", get_service_name()], system=system, check=False, timeout=30)

    unit_path = get_systemd_unit_path(system=system)
    if unit_path.exists():
        unit_path.unlink()
        print(f"✓ Removed {unit_path}")

    _run_systemctl(["daemon-reload"], system=system, check=True, timeout=30)
    print(f"✓ {_service_scope_label(system).capitalize()} service uninstalled")


def _require_service_installed(action: str, system: bool = False) -> None:
    unit_path = get_systemd_unit_path(system=system)
    if not unit_path.exists():
        scope_flag = " --system" if system else ""
        print("✗ Gateway service is not installed")
        print(f"  Run: {'sudo ' if system else ''}hermes gateway install{scope_flag}")
        sys.exit(1)


def systemd_start(system: bool = False):
    system = _select_systemd_scope(system)
    if system:
        _require_root_for_system_service("start")
    else:
        # Fail fast with guidance when the user D-Bus session is unreachable (raises UserSystemdUnavailableError).
        _preflight_user_systemd()
    _require_service_installed("start", system=system)
    # HERMES_HOME sync happens in refresh's systemd_unit_is_current gate; the unit is guaranteed to exist here.
    refresh_systemd_unit_if_needed(system=system)
    _run_systemctl(["start", get_service_name()], system=system, check=True, timeout=30)
    print(f"✓ {_service_scope_label(system).capitalize()} service started")


def systemd_stop(system: bool = False):
    system = _select_systemd_scope(system)
    if system:
        _require_root_for_system_service("stop")
    _require_service_installed("stop", system=system)
    _sync_hermes_home_from_systemd_unit(system=system)
    _mark_planned_stop()
    try:
        _run_systemctl(["stop", get_service_name()], system=system, check=True, timeout=90)
    except subprocess.TimeoutExpired:
        label = _service_scope_label(system)
        print(
            f"Gateway {label} service is still stopping after 90s; "
            "check `hermes gateway status` or logs for final shutdown state."
        )
        return
    print(f"✓ {_service_scope_label(system).capitalize()} service stopped")


def systemd_restart(system: bool = False):
    system = _select_systemd_scope(system)
    if system:
        _require_root_for_system_service("restart")
    else:
        _preflight_user_systemd()
    _require_service_installed("restart", system=system)
    # HERMES_HOME sync happens in refresh's systemd_unit_is_current gate; its os.environ mutation
    # persists for the get_running_pid / drain-timeout reads below.
    refresh_systemd_unit_if_needed(system=system)
    from gateway.status import get_running_pid

    pid = get_running_pid() or _systemd_main_pid(system=system)
    if pid is not None and probe_gateway_loop_liveness(pid) == GATEWAY_LOOP_WEDGED:
        # Event loop provably dead: SIGUSR1 can never drain it, so escalate (SIGTERM grace →
        # SIGKILL, ~10s) and let systemd relaunch. A busy-but-alive gateway keeps the full budget.
        print(
            f"⚠ Gateway PID {pid} event loop is unresponsive — "
            "skipping graceful drain and forcing a bounded stop..."
        )
        _escalate_wedged_gateway(pid)
        svc = get_service_name()
        _run_systemctl(["reset-failed", svc], system=system, check=False, timeout=30)
        _run_systemctl(["restart", svc], system=system, check=False, timeout=90)
        _wait_for_systemd_service_restart(system=system, previous_pid=pid)
        return
    if pid is not None:
        scope_label = _service_scope_label(system).capitalize()
        svc = get_service_name()
        wait_budget = _get_restart_exit_wait_budget()
        print(
            f"⏳ {scope_label} service restarting gracefully (PID {pid}) — "
            f"waiting up to {wait_budget:.0f}s for in-flight turns + drain..."
        )
        service_action = "restart"
        if _graceful_restart_via_sigusr1(pid, wait_budget):
            # Exit 75 hands restart ownership to systemd; observe that replacement rather than
            # issuing another restart that could stop the process systemd already brought up.
            replacement_observed: list[bool] = []
            if _wait_for_systemd_service_restart(
                system=system,
                previous_pid=pid,
                replacement_observed=replacement_observed,
            ):
                return
            if replacement_observed:
                return
            if _systemd_service_is_start_limited(system=system):
                return

            # A replacement may have started but not reached gateway runtime
            # readiness before the wait expired.  Never stop that generation.
            props = _read_systemd_unit_properties(system=system)
            if not props:
                return
            replacement_pid = _systemd_main_pid_from_props(props)
            if (
                props.get("ActiveState") in {"active", "activating", "reloading"}
                or props.get("SubState") == "auto-restart"
                or (replacement_pid is not None and replacement_pid != pid)
            ):
                return

            print(
                "⚠ Systemd did not relaunch the gateway after its graceful exit; "
                "starting the inactive service..."
            )
            # ``start`` is intentionally idempotent: if a replacement appears
            # after the snapshot, this must not stop that new generation.
            service_action = "start"
        else:
            print(
                f"⚠ Graceful restart did not complete within {int(wait_budget)}s; "
                "forcing a service restart..."
            )

        _systemd_reset_and_run(service_action, system=system, previous_pid=pid)
        return

    if _recover_pending_systemd_restart(system=system, previous_pid=pid):
        return
    _systemd_reset_and_run("restart", system=system, previous_pid=pid)


def _systemd_reset_and_run(action: str, *, system: bool, previous_pid) -> None:
    """``reset-failed`` then ``systemctl <action>``, then wait for the relaunch. Start-limit
    rejection prints the wait hint instead of raising; a 90s timeout prints where to look."""
    svc = get_service_name()
    _run_systemctl(["reset-failed", svc], system=system, check=False, timeout=30)
    try:
        _run_systemctl([action, svc], system=system, check=True, timeout=90)
    except subprocess.CalledProcessError as exc:
        if _systemd_error_indicates_start_limit(exc) or _systemd_service_is_start_limited(system=system):
            _print_systemd_start_limit_wait(system=system)
            return
        raise
    except subprocess.TimeoutExpired:
        label = _service_scope_label(system)
        print(
            f"Gateway {label} service is still restarting after 90s; "
            "check `hermes gateway status` or logs for final state."
        )
        return
    _wait_for_systemd_service_restart(system=system, previous_pid=previous_pid)


def systemd_status(deep: bool = False, system: bool = False, full: bool = False):
    system = _select_systemd_scope(system)
    unit_path = get_systemd_unit_path(system=system)
    scope_flag = " --system" if system else ""

    if not unit_path.exists():
        print("✗ Gateway service is not installed")
        print(f"  Run: {'sudo ' if system else ''}hermes gateway install{scope_flag}")
        return

    if has_conflicting_systemd_units():
        print_systemd_scope_conflict_warning()
        print()

    if has_legacy_hermes_units():
        print_legacy_unit_warning()
        print()

    if not systemd_unit_is_current(system=system):
        print("⚠ Installed gateway service definition is outdated")
        print(
            f"  Run: {'sudo ' if system else ''}hermes gateway restart{scope_flag}  # auto-refreshes the unit"
        )
        print()

    status_cmd = ["status", get_service_name(), "--no-pager"]
    if full:
        status_cmd.append("-l")

    _run_systemctl(status_cmd, system=system, capture_output=False, timeout=10)

    result = _run_systemctl(["is-active", get_service_name()], system=system, timeout=10, **_CAPTURE_TEXT)

    status = result.stdout.strip()

    if status == "active":
        print(f"✓ {_service_scope_label(system).capitalize()} gateway service is running")
    else:
        print(f"✗ {_service_scope_label(system).capitalize()} gateway service is stopped")
        print(f"  Run: {'sudo ' if system else ''}hermes gateway start{scope_flag}")

    configured_user = _read_systemd_user_from_unit(unit_path) if system else None
    if configured_user:
        print(f"Configured to run as: {configured_user}")

    _print_runtime_health()

    unit_props = _read_systemd_unit_properties(system=system)
    active_state = unit_props.get("ActiveState", "")
    sub_state = unit_props.get("SubState", "")
    exec_main_status = unit_props.get("ExecMainStatus", "")
    result_code = unit_props.get("Result", "")
    if active_state == "activating" and sub_state == "auto-restart":
        print("  ⏳ Restart pending: systemd is waiting to relaunch the gateway")
    elif _systemd_unit_is_start_limited(unit_props):
        print("  ⏳ Restart pending: systemd is temporarily rate-limiting starts")
        print(
            f"  Run after the start-limit window expires: {'sudo ' if system else ''}hermes gateway restart{scope_flag}"
        )
        print(
            f"  Or clear it manually: systemctl {'--user ' if not system else ''}reset-failed {get_service_name()}"
        )
    elif active_state == "failed" and exec_main_status == str(GATEWAY_SERVICE_RESTART_EXIT_CODE):
        print("  ⚠ Planned restart is stuck in systemd failed state (exit 75)")
        print(
            f"  Run: systemctl {'--user ' if not system else ''}reset-failed {get_service_name()} && {'sudo ' if system else ''}hermes gateway start{scope_flag}"
        )
    elif active_state == "failed" and result_code:
        print(f"  ⚠ Systemd unit result: {result_code}")

    if system:
        print("✓ System service starts at boot without requiring systemd linger")
    elif deep:
        print_systemd_linger_guidance()
    else:
        linger_enabled, _ = get_systemd_linger_status()
        if linger_enabled is True:
            print("✓ Systemd linger is enabled (service survives logout)")
        elif linger_enabled is False:
            print("⚠ Systemd linger is disabled (gateway may stop when you log out)")
            print("  Run: sudo loginctl enable-linger $USER")

    if deep:
        print()
        print("Recent logs:")
        log_cmd = _journalctl_cmd(system) + ["-u", get_service_name(), "-n", "20", "--no-pager"]
        if full:
            log_cmd.append("-l")
        subprocess.run(log_cmd, timeout=10)


# =============================================================================
# Launchd (macOS)
# =============================================================================


def get_launchd_label() -> str:
    """Return the launchd service label, scoped per profile."""
    suffix = _profile_suffix()
    return f"ai.hermes.gateway-{suffix}" if suffix else "ai.hermes.gateway"


# Cached launchd domain result — probing is cheap but should only run once per
# process invocation (each ``hermes gateway start/stop/status`` call).
_resolved_launchd_domain: str | None = None


def _probe_launchd_domain_for_label(label: str) -> str:
    """Resolve the launchd domain managing ``label`` (uncached): ``gui/<uid>`` (Aqua), then
    ``user/<uid>`` (Background/SSH), else ``launchctl managername`` heuristic.

    Sibling profiles may legitimately live in different domains, so never reuse the current
    profile's cached ``_launchd_domain()`` for another label.
    """
    uid = os.getuid()  # windows-footgun: ok — POSIX launchd (macOS) helper, never invoked on Windows
    gui_domain = f"gui/{uid}"
    user_domain = f"user/{uid}"

    # 1. Probe gui/<uid> first — in Aqua sessions the service is loaded here.
    # 2. Then user/<uid> — in Background/SSH sessions this is the working domain.
    for domain in (gui_domain, user_domain):
        try:
            subprocess.run(
                ["launchctl", "print", f"{domain}/{label}"],
                check=True,
                timeout=5,
                capture_output=True,
            )
            return domain
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
            pass

    # 3. Neither domain has the service loaded — use managername as heuristic.
    #    Aqua → gui/<uid>, anything else (Background, loginwindow) → user/<uid>.
    try:
        result = subprocess.run(["launchctl", "managername"], timeout=5, **_CAPTURE_TEXT)
        if "Aqua" in (result.stdout or ""):
            return gui_domain
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # 4. Default to user/<uid> (matches the pre-probing behavior for
    #    Background/SSH sessions and is the recommended domain on macOS 26+).
    return user_domain


def _launchd_domain() -> str:
    """Domain managing the current profile's gateway; cached per process so start/stop/restart agree."""
    global _resolved_launchd_domain
    if _resolved_launchd_domain is not None:
        return _resolved_launchd_domain
    _resolved_launchd_domain = _probe_launchd_domain_for_label(get_launchd_label())
    return _resolved_launchd_domain


# Exit 125 ("Domain does not support specified action") and 3/113 ("Could not find service") all
# mean the job isn't loaded in the target domain: re-bootstrap the plist and retry.
_LAUNCHD_JOB_UNLOADED_EXIT_CODES = frozenset({3, 113, 125})

# Exit 5 (EIO) or a persistent 125 is NOT on its own proof the domain is broken:
#   1. the label is still registered (stale load from an interrupted restart) — recoverable by
#      bootout + bootstrap again;
#   2. the domain genuinely can't manage services (macOS 26+) — degrade to a detached process.
# `_launchctl_bootstrap()` tries case 1 first; only when that retry ALSO returns 5/125 do callers
# treat the domain as unsupported via `_launchctl_domain_unsupported`.
_LAUNCHCTL_DOMAIN_UNSUPPORTED_CODES = frozenset({5, 125})


def _launchd_error_indicates_unloaded(exc: subprocess.CalledProcessError) -> bool:
    """True when launchctl failed because the job isn't loaded (retry bootstrap)."""
    return exc.returncode in _LAUNCHD_JOB_UNLOADED_EXIT_CODES


def _launchctl_domain_unsupported(returncode: int) -> bool:
    """True when launchctl can't manage the domain even after a fresh bootstrap (5/125 persist on macOS
    26+) — degrade to detached."""
    return returncode in _LAUNCHCTL_DOMAIN_UNSUPPORTED_CODES


# `launchctl bootstrap` returns EIO when the label is *already* registered (stale load). That is
# recoverable, NOT proof the domain is unmanageable; only a failed bootout + retry is.
_LAUNCHCTL_BOOTSTRAP_EIO = 5


def _launchctl_bootstrap(domain: str, plist_path, label: str, *, timeout: int = 30) -> None:
    """Bootstrap a launchd job, recovering from a stale already-loaded label.

    A still-registered label makes ``bootstrap`` fail EIO (5) — the *already loaded* case, distinct
    from an unmanageable domain. Without the bootout + retry we'd misclassify it as "launchd can't
    manage this macOS" and degrade to detached, silently losing auto-start and crash-restart.
    """
    try:
        subprocess.run(["launchctl", "bootstrap", domain, str(plist_path)], check=True, timeout=timeout)
        return
    except subprocess.CalledProcessError as exc:
        if exc.returncode != _LAUNCHCTL_BOOTSTRAP_EIO:
            raise
        # Stale registration — drop the leftover label and bootstrap once more.
        subprocess.run(["launchctl", "bootout", f"{domain}/{label}"], check=False, timeout=timeout)
        subprocess.run(["launchctl", "bootstrap", domain, str(plist_path)], check=True, timeout=timeout)


def _launchd_reload_log_path() -> Path:
    """Path the launchd reload watchdog tails for persistent-orphan detection."""
    return get_hermes_home() / "logs" / "launchd-reload.log"


def _append_launchd_reload_log(message: str) -> None:
    """Append a timestamped line to the launchd reload log (best-effort)."""
    path = _launchd_reload_log_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        from datetime import datetime as _dt

        stamp = _dt.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z")
        with path.open("a", encoding="utf-8") as fh:
            fh.write(f"[{stamp}] {message}\n")
    except OSError:
        pass


def _launchctl_label_supervising_process(label: str) -> bool:
    """True when launchd knows ``label`` AND runs a process for it. ``launchctl list`` exits 0 for a
    mere registered definition (``state = not running`` on macOS 26+), so a positive PID is required."""
    try:
        result = subprocess.run(["launchctl", "list", label], check=False, timeout=10, **_CAPTURE_TEXT)
    except (subprocess.TimeoutExpired, OSError):
        return False
    if result.returncode != 0:
        return False
    return _parse_launchd_pid_from_list_output(result.stdout) is not None


def _retry_launchctl_bootstrap_until_registered(
    domain: str, plist_path, label: str, *, deadline: float
) -> bool:
    """Retry ``_launchctl_bootstrap`` until the label supervises a process or ``deadline`` passes.

    Under load / a launchd race, bootstrap can fail even after bootout, orphaning the service from
    KeepAlive. This happens during a graceful drain (default 180s), so a fixed ~10s window is too short.
    """
    attempt = 0
    while True:
        attempt += 1
        try:
            _launchctl_bootstrap(domain, plist_path, label, timeout=30)
            if _launchctl_label_supervising_process(label):
                return True
            _append_launchd_reload_log(
                f"bootstrap attempt {attempt} exited 0 but {domain}/{label} "
                f"has no supervised process (launchctl list) — retrying"
            )
        except subprocess.CalledProcessError as exc:
            _append_launchd_reload_log(
                f"bootstrap attempt {attempt} failed (rc={exc.returncode}) "
                f"for {domain}/{label} — retrying"
            )
        except subprocess.TimeoutExpired:
            _append_launchd_reload_log(
                f"bootstrap attempt {attempt} timed out for {domain}/{label} "
                f"— retrying"
            )
        if time.monotonic() >= deadline:
            return False
        time.sleep(2)


# launchd-unsupported marker: persisted when the domain can't be managed (exit 5/125, macOS 26+)
# so `launchd_status()` can explain the missing supervision; cleared when bootstrap/kickstart
# succeeds so an OS fix recovers automatically.


def _launchd_unsupported_marker_path() -> Path:
    return get_hermes_home() / ".gateway-launchd-unsupported"


def _write_launchd_unsupported_marker() -> None:
    """Persist that launchd cannot supervise the gateway on this host."""
    import json
    from datetime import datetime, timezone

    try:
        _launchd_unsupported_marker_path().write_text(
            json.dumps({
                "written_at": datetime.now(timezone.utc).isoformat(),
                "reason": "launchd domain unsupported (exit 5/125)",
            }),
            encoding="utf-8",
        )
    except OSError:
        pass


def _clear_launchd_unsupported_marker() -> None:
    """Clear the unsupported marker when launchd bootstrap succeeds."""
    with contextlib.suppress(OSError):
        _launchd_unsupported_marker_path().unlink(missing_ok=True)


def _launchd_unsupported_marker_exists() -> bool:
    return _launchd_unsupported_marker_path().exists()


def _gateway_run_command() -> list[str]:
    """Build ``python -m hermes_cli.main [--profile X] gateway run --replace``, honoring the active profile."""
    cmd = [get_python_path(), "-m", "hermes_cli.main"]
    profile_arg = _profile_arg()
    if profile_arg:
        cmd.extend(profile_arg.split())
    cmd.extend(["gateway", "run", "--replace"])
    return cmd


def _timestamped_stderr_gateway_command(error_log: Path, *, external_supervisor: bool = False) -> list[str]:
    """Wrap gateway run so raw stderr lines are timestamped before file write.

    ``external_supervisor=True`` (launchd ProgramArguments only) adds ``--external-supervisor`` so
    ``hermes update`` hands the process back to launchd instead of a detached watcher, and drops
    ``--replace``: KeepAlive respawns would re-arm takeover on every respawn, so two profiles
    sharing a token would kill each other forever. The nohup fallback stays unmarked.
    """
    inner = _gateway_run_command()
    if external_supervisor and "--external-supervisor" not in inner:
        inner = [*inner, "--external-supervisor"]
    if external_supervisor and "--replace" in inner:
        inner = [part for part in inner if part != "--replace"]
    return [
        get_python_path(),
        "-m",
        "hermes_cli.stderr_timestamp",
        "--error-log",
        str(error_log),
        "--",
        *inner,
    ]


def _spawn_detached_gateway() -> bool:
    """Launch the gateway detached (launchd fallback for macOS 26+). CLI-managed nohup equivalent:
    stdout → gateway.log, timestamped stderr → gateway.error.log, PID via gateway.pid so stop/status work."""
    from hermes_cli._subprocess_compat import windows_detach_popen_kwargs

    log_dir = get_hermes_home() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    out_path = log_dir / "gateway.log"
    err_path = log_dir / "gateway.error.log"
    try:
        out = open(out_path, "ab")
    except OSError:
        return False
    try:
        with out:
            subprocess.Popen(
                _timestamped_stderr_gateway_command(err_path),
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=subprocess.DEVNULL,
                **windows_detach_popen_kwargs(),
            )
    except OSError:
        return False
    return True


def _launchd_fallback_to_detached(reason: str, *, exit_on_failure: bool = True) -> bool:
    """Start the gateway detached when launchd can't manage it; on failure print the manual workaround
    and (by default) exit 1."""
    from hermes_constants import display_hermes_home as _dhh

    _write_launchd_unsupported_marker()
    print(f"⚠ launchd cannot manage the gateway on this macOS version ({reason}).")
    if _spawn_detached_gateway():
        print("✓ Started gateway as a background process instead")
        print("  It will NOT auto-start at login or auto-restart on crash.")
        print(f"  Logs: {_dhh()}/logs/gateway.log")
        print("  Stop it with: hermes gateway stop")
        return True
    print_error("Failed to start the gateway as a background process.")
    print(f"  Try manually: nohup hermes gateway run --replace > {_dhh()}/logs/gateway.log 2>&1 &")
    if exit_on_failure:
        sys.exit(1)
    return False


def generate_launchd_plist() -> str:
    # Stable cwd anchor — never the volatile source checkout. See _stable_service_working_dir() for
    # the rationale (same rot risk applies to launchd's WorkingDirectory as to systemd's).
    working_dir = _stable_service_working_dir()
    hermes_home = str(get_hermes_home().resolve())
    log_dir = get_hermes_home() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    label = get_launchd_label()
    # launchd's default PATH (/usr/bin:/bin:/usr/sbin:/sbin) misses Homebrew, nvm, cargo…; prepend
    # venv/bin + node_modules/.bin (as in the systemd unit), then capture the user's shell PATH.
    detected_venv = _detect_venv_dir()
    venv_dir = str(detected_venv) if detected_venv else str(PROJECT_ROOT / "venv")
    # Resolve the directory containing the node binary (e.g. Homebrew, nvm)
    # so it's explicitly in PATH even if the user's shell PATH changes later.
    priority_dirs = _build_service_path_dirs()
    _append_node_dir_for_service(priority_dirs)
    sane_path = ":".join(
        dict.fromkeys(priority_dirs + [p for p in os.environ.get("PATH", "").split(":") if p])
    )

    err_path = log_dir / "gateway.error.log"

    # ProgramArguments (incl. --profile); the stderr wrapper keeps launchd restart semantics while timestamping stderr.
    prog_args = [
        f"<string>{part}</string>"
        for part in _timestamped_stderr_gateway_command(err_path, external_supervisor=True)
    ]
    prog_args_xml = "\n        ".join(prog_args)

    # Persist the configured RLIMIT_NOFILE floor: launchd defaults to soft 256, and every plist
    # rewrite would otherwise strip a manual limit and reintroduce EMFILE crashes.
    nofile_block = ""
    try:
        from hermes_cli.resource_limits import configured_nofile_soft_limit

        nofile_target = configured_nofile_soft_limit()
    except Exception:
        nofile_target = None
    if nofile_target:
        nofile_block = f"""
    <key>SoftResourceLimits</key>
    <dict>
        <key>NumberOfFiles</key>
        <integer>{nofile_target}</integer>
    </dict>
"""

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{label}</string>

    <key>ProgramArguments</key>
    <array>
        {prog_args_xml}
    </array>
    
    <key>WorkingDirectory</key>
    <string>{working_dir}</string>
    
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>{sane_path}</string>
        <key>VIRTUAL_ENV</key>
        <string>{venv_dir}</string>
        <key>HERMES_HOME</key>
        <string>{hermes_home}</string>
        <key>HERMES_SUPERVISED_CHILD</key>
        <string>1</string>
    </dict>

    <key>LimitLoadToSessionType</key>
    <array>
        <string>Aqua</string>
        <string>Background</string>
    </array>
    
    <key>RunAtLoad</key>
    <true/>
    
    <key>KeepAlive</key>
    <true/>

    <!-- ThrottleInterval raises launchd's default 10s minimum respawn interval
         to 30s so a crash-looping gateway can't hammer launchd into a rapid
         respawn storm; ExitTimeOut gives the gateway 25s of graceful-drain
         headroom before launchd escalates from SIGTERM to SIGKILL on stop. -->
    <key>ThrottleInterval</key>
    <integer>30</integer>

    <key>ExitTimeOut</key>
    <integer>25</integer>
{nofile_block}
    <key>StandardOutPath</key>
    <string>{log_dir}/gateway.log</string>
    
    <key>StandardErrorPath</key>
    <string>{log_dir}/gateway.error.log</string>
</dict>
</plist>
"""


def launchd_plist_is_current() -> bool:
    """Check if the installed launchd plist matches the currently generated one."""
    plist_path = get_launchd_plist_path()
    if not plist_path.exists():
        return False

    installed = plist_path.read_text(encoding="utf-8")
    expected = generate_launchd_plist()
    return _normalize_launchd_plist_for_comparison(
        installed
    ) == _normalize_launchd_plist_for_comparison(expected)


def _spawn_deferred_launchd_reload(
    *, domain: str, label: str, target: str, plist_path: Path, gateway_pid: int
) -> bool:
    """Hand the bootout/bootstrap cycle to a transient ``launchctl submit`` job; True if spawned.

    The helper waits for the OLD gateway to exit (bootout only SIGTERMs; bootstrap during drain
    fails EIO), then retries bootstrap until ``launchctl list`` shows a positive PID or the drain
    budget elapses, logging exhaustion for the reload watchdog.
    """
    reload_log_path = get_hermes_home() / "logs" / "launchd-reload.log"
    with contextlib.suppress(OSError):
        reload_log_path.parent.mkdir(parents=True, exist_ok=True)

    # Write a durable pre-bootout marker so we can distinguish "helper
    # never started" from "helper ran but bootout/bootstrap failed".
    _append_launchd_reload_log(f"Launchd reload helper started for {target}")

    # Retry until launchctl LISTS the label (not just exit 0), bounded by the drain budget: the
    # failure happens while the old gateway is still draining (default 180s), so ~10s is too short.
    _reload_budget = int(max(30.0, _get_restart_drain_timeout()))
    # Label for the transient one-shot job (see `launchctl submit` below).
    # Unique per reload so concurrent/repeated reloads never collide.
    submit_label = f"{label}.reload.{os.getpid()}.{int(time.time())}"
    reload_script = (
        f"sleep 2; "
        f"launchctl bootout {shlex.quote(target)} 2>/dev/null; "
        # Wait for the OLD gateway to exit: bootout only SIGTERMs, the gateway drains up to
        # agent.restart_drain_timeout, and every bootstrap during the drain fails EIO.
        f"_wait_deadline=$(($(date +%s) + {_reload_budget})); "
        f"while kill -0 {gateway_pid} 2>/dev/null; do "
        f"  if [ $(date +%s) -ge $_wait_deadline ]; then "
        f"    echo \"[$(date '+%Y-%m-%d %H:%M:%S %z')] old gateway pid {gateway_pid} still alive after {_reload_budget}s drain wait — bootstrapping anyway\" >> {shlex.quote(str(reload_log_path))}; "
        f"    break; "
        f"  fi; "
        f"  sleep 1; "
        f"done; "
        # Let launchd finish unregistering the label after the process exits.
        f"sleep 1; "
        f"_deadline=$(($(date +%s) + {_reload_budget})); "
        f"while :; do "
        f"  launchctl bootstrap {shlex.quote(domain)} {shlex.quote(str(plist_path))} 2>/dev/null; "
        # Require a POSITIVE PID: `launchctl list` also exits 0 for a registered-but-not-running
        # definition, and a crashed job reports `"PID" = -1` (mirrors _parse_launchd_pid_from_list_output).
        f"  if launchctl list {shlex.quote(label)} 2>/dev/null | grep -qE '\\\"PID\\\" = [0-9]+;'; then break; fi; "
        f"  echo \"[$(date '+%Y-%m-%d %H:%M:%S %z')] bootstrap not yet registered for {shlex.quote(target)} — retrying\" >> {shlex.quote(str(reload_log_path))}; "
        f"  if [ $(date +%s) -ge $_deadline ]; then break; fi; "
        f"  sleep 2; "
        f"done; "
        f"if ! launchctl list {shlex.quote(label)} 2>/dev/null | grep -qE '\\\"PID\\\" = [0-9]+;'; then "
        f"  echo \"[$(date '+%Y-%m-%d %H:%M:%S %z')] FAILED launchd reload for {shlex.quote(target)} — service NOT registered after {_reload_budget}s of retries\" >> {shlex.quote(str(reload_log_path))}; "
        f"fi; "
        # Submitted jobs stay registered after the script exits (one leaked dead label per reload);
        # removing our own label is the documented way to end a one-shot submit job.
        f"launchctl remove {shlex.quote(submit_label)} 2>/dev/null"
    )
    try:
        # `launchctl submit` (transient one-shot job) rather than start_new_session=True: setsid(2)
        # does NOT leave the launchd job's process coalition, and bootout kills ALL coalition members.
        subprocess.Popen(
            [
                "launchctl", "submit",
                "-l", submit_label,
                "-o", str(reload_log_path),
                "-e", str(reload_log_path),
                "--",
                "/bin/bash", "-c", reload_script,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as e:
        # Fall through to the in-process bootout/bootstrap: risky if we share the coalition, but
        # better than a never-reloaded plist.
        logger.warning("Deferred launchd reload could not be spawned: %s", e)
        _append_launchd_reload_log(
            f"FAILED to spawn launchd reload helper for {target}: {e} — "
            f"falling back to in-process bootout/bootstrap"
        )
        return False
    return True


def refresh_launchd_plist_if_needed() -> bool:
    """Rewrite the installed plist when the generated one differs, then bootout/bootstrap so launchd
    re-reads it immediately."""
    plist_path = get_launchd_plist_path()
    if not plist_path.exists() or launchd_plist_is_current():
        return False

    new_plist = generate_launchd_plist()
    if _refuse_temp_home_service_write(new_plist, "launchd plist"):
        return False

    plist_path.write_text(new_plist, encoding="utf-8")
    label = get_launchd_label()
    domain = _launchd_domain()
    target = f"{domain}/{label}"

    # If this runs INSIDE the gateway's launchd process tree (e.g. agent self-update), a direct
    # bootout kills THIS CLI before bootstrap runs, leaving the job unloaded with no KeepAlive.
    gateway_pid = None
    try:
        from gateway.status import get_running_pid
        gateway_pid = get_running_pid()
    except Exception:
        gateway_pid = None

    # POSIX ancestry is NOT a reliable "bootout will kill us" test: coalition membership is inherited
    # at spawn and survives reparenting to PID 1, so a misclassified process once died mid-bootstrap
    # with nothing left to re-register the label. The detached helper is also correct outside the
    # coalition, so always prefer it; in-process is only the fallback when it can't be spawned.
    if (
        gateway_pid is not None
        and hasattr(os, "setsid")  # POSIX-only; launchd is macOS so always true here
    ) and _spawn_deferred_launchd_reload(
        domain=domain, label=label, target=target, plist_path=plist_path, gateway_pid=gateway_pid
    ):
        print(
            "↻ Updated gateway launchd service definition; reload deferred to "
            "a transient launchd job (survives the bootout of this process)"
        )
        return True

    # Bootout/bootstrap so launchd reads the new definition. Bootstrap once failed silently under
    # load during a drain, leaving the job unregistered — KeepAlive can't revive an unknown job.
    subprocess.run(["launchctl", "bootout", target], check=False, timeout=90)
    # Size the retry window to the drain timeout (default 180s): the failure occurs while the old gateway drains.
    _reload_budget = max(30.0, _get_restart_drain_timeout())
    # Wait out the old gateway's drain first so the budget isn't burned on guaranteed EIO ("already loaded").
    if gateway_pid is not None and not _wait_for_pid_exit(gateway_pid, _reload_budget):
        _append_launchd_reload_log(
            f"old gateway pid {gateway_pid} still alive after "
            f"{int(_reload_budget)}s drain wait — bootstrapping {target} anyway"
        )
    _deadline = time.monotonic() + _reload_budget
    if not _retry_launchctl_bootstrap_until_registered(domain, plist_path, label, deadline=_deadline):
        _append_launchd_reload_log(
            f"FAILED launchd reload of {target} — service NOT registered after "
            f"retrying for {int(_reload_budget)}s (in-process fallback path)"
        )
        logger.error(
            "launchd reload of %s failed — service not registered after %ds of "
            "retries; see %s",
            target,
            int(_reload_budget),
            _launchd_reload_log_path(),
        )
    print("↻ Updated gateway launchd service definition to match the current Hermes install")
    return True


def launchd_install(force: bool = False):
    plist_path = get_launchd_plist_path()

    if plist_path.exists() and not force:
        if not launchd_plist_is_current():
            print(f"↻ Repairing outdated launchd service at: {plist_path}")
            refresh_launchd_plist_if_needed()
            print("✓ Service definition updated")
            return
        print(f"Service already installed at: {plist_path}")
        print("Use --force to reinstall")
        return

    plist_path.parent.mkdir(parents=True, exist_ok=True)
    new_plist = generate_launchd_plist()
    if _refuse_temp_home_service_write(new_plist, "launchd plist"):
        return
    print(f"Installing launchd service to: {plist_path}")
    plist_path.write_text(new_plist, encoding="utf-8")

    try:
        _launchctl_bootstrap(_launchd_domain(), plist_path, get_launchd_label(), timeout=30)
    except subprocess.CalledProcessError as e:
        if not _launchctl_domain_unsupported(e.returncode):
            raise
        _launchd_fallback_to_detached(f"launchctl bootstrap exit {e.returncode}")
        return

    print()
    print("✓ Service installed and loaded!")
    _clear_launchd_unsupported_marker()
    print()
    print("Next steps:")
    print("  hermes gateway status             # Check status")
    from hermes_constants import display_hermes_home as _dhh

    print(f"  tail -f {_dhh()}/logs/gateway.log  # View logs")


def launchd_uninstall():
    plist_path = get_launchd_plist_path()
    label = get_launchd_label()
    subprocess.run(["launchctl", "bootout", f"{_launchd_domain()}/{label}"], check=False, timeout=90)

    if plist_path.exists():
        plist_path.unlink()
        print(f"✓ Removed {plist_path}")

    print("✓ Service uninstalled")


def launchd_start():
    plist_path = get_launchd_plist_path()
    label = get_launchd_label()

    # Self-heal if the plist is missing entirely (e.g., manual cleanup, failed upgrade)
    if not plist_path.exists():
        new_plist = generate_launchd_plist()
        if _refuse_temp_home_service_write(new_plist, "launchd plist"):
            sys.exit(1)
        print("↻ launchd plist missing; regenerating service definition")
        plist_path.parent.mkdir(parents=True, exist_ok=True)
        plist_path.write_text(new_plist, encoding="utf-8")
        if not _launchd_bootstrap_and_kickstart(plist_path, label):
            return
        print("✓ Service started")
        _clear_launchd_unsupported_marker()
        return

    refresh_launchd_plist_if_needed()
    try:
        _launchctl_kickstart_current(label)
    except subprocess.CalledProcessError as e:
        if not _launchd_error_indicates_unloaded(e):
            raise
        # Job not loaded in this domain — re-bootstrap the plist and retry.
        print("↻ launchd job was unloaded; reloading service definition")
        if not _launchd_bootstrap_and_kickstart(plist_path, label):
            return
    print("✓ Service started")
    _clear_launchd_unsupported_marker()


def _launchctl_kickstart_current(label: str) -> None:
    subprocess.run(["launchctl", "kickstart", f"{_launchd_domain()}/{label}"], check=True, timeout=30)


def _launchd_bootstrap_and_kickstart(plist_path: Path, label: str) -> bool:
    """Bootstrap then kickstart; False after degrading to detached (domain unsupported). Other errors propagate."""
    try:
        _launchctl_bootstrap(_launchd_domain(), plist_path, label, timeout=30)
        _launchctl_kickstart_current(label)
    except subprocess.CalledProcessError as e:
        if not _launchctl_domain_unsupported(e.returncode):
            raise
        _launchd_fallback_to_detached(f"launchctl exit {e.returncode}")
        return False
    return True


def launchd_stop():
    label = get_launchd_label()
    target = f"{_launchd_domain()}/{label}"
    _mark_planned_stop()
    # bootout unloads the definition so KeepAlive doesn't respawn; a plain SIGTERM is immediately
    # undone by KeepAlive. `hermes gateway start` re-bootstraps when it sees the job unloaded.
    try:
        subprocess.run(["launchctl", "bootout", target], check=True, timeout=90)
    except subprocess.CalledProcessError as e:
        # Job already unloaded (3/113/125), or the domain can't be managed at all (5/125, macOS 26+
        # detached-fallback process) — in both cases just fall through to the PID-based kill below.
        if _launchd_error_indicates_unloaded(e) or _launchctl_domain_unsupported(e.returncode):
            pass
        else:
            raise
    _wait_for_gateway_exit(timeout=10.0, force_after=5.0)
    print("✓ Service stopped")


def _wait_for_gateway_exit(timeout: float = 10.0, force_after: float | None = 5.0) -> bool:
    """Wait up to ``timeout`` s for the gateway (by gateway.pid, not launchd labels, so multiple
    HERMES_HOMEs work) to exit; SIGKILL it after ``force_after`` s of graceful waiting."""
    import time
    from gateway.status import get_process_start_time, get_running_pid

    deadline = time.monotonic() + timeout
    force_deadline = ((time.monotonic() + force_after) if force_after is not None else None)
    force_sent = False

    while time.monotonic() < deadline:
        pid = get_running_pid()
        if pid is None:
            return True  # Process exited cleanly.

        if (force_after is not None and not force_sent and time.monotonic() >= force_deadline):
            # Grace period expired — force-kill the specific PID.
            try:
                terminate_pid(pid, force=True, expected_start_time=get_process_start_time(pid))
                print(f"⚠ Gateway PID {pid} did not exit gracefully; sent SIGKILL")
            except (ProcessLookupError, PermissionError, OSError):
                return True  # Already gone or we can't touch it.
            force_sent = True

        time.sleep(0.3)

    # Timed out even after force-kill.
    remaining_pid = get_running_pid()
    if remaining_pid is not None:
        print(f"⚠ Gateway PID {remaining_pid} still running after {timeout}s — restart may fail")
        return False
    return True


def _launchd_kickstart(label: str, domain: str) -> None:
    """``launchctl kickstart -k domain/label``; raises so callers own per-label failure accounting."""
    subprocess.run(
        ["launchctl", "kickstart", "-k", f"{domain}/{label}"],
        check=True,
        timeout=90,
        **_CAPTURE_TEXT,
    )


def _wait_for_launchd_service_pid(
    label: str, old_pid: int | None, timeout: float = 10.0, *, domain: str
) -> bool:
    """Poll ``domain/label`` (0.5s) until it runs on a fresh PID or ``timeout`` passes.

    KeepAlive respawn isn't instantaneous, so a one-shot check falsely reports the service down.
    launchctl ``TimeoutExpired`` propagates — callers own per-label failure accounting.
    """
    deadline = time.monotonic() + max(timeout, 0.5)
    while True:
        _loaded, pid = _launchd_print_service_pid(domain, label)
        if pid is not None and pid > 0 and pid != old_pid:
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.5)


def launchd_restart():
    label = get_launchd_label()
    domain = _launchd_domain()
    target = f"{domain}/{label}"
    from gateway.status import get_running_pid

    try:
        pid = get_running_pid()
        if pid is not None and _request_gateway_self_restart(pid):
            print("✓ Service restart requested")
            _clear_launchd_unsupported_marker()
            return
        if pid is not None and probe_gateway_loop_liveness(pid) == GATEWAY_LOOP_WEDGED:
            # Event loop provably dead: it can't process a graceful shutdown, so a full drain wait
            # only stalls the restart (and `hermes update`). Bounded SIGTERM → SIGKILL, ~10s.
            print(
                f"⚠ Gateway PID {pid} event loop is unresponsive — "
                "skipping drain and forcing a bounded stop..."
            )
            _escalate_wedged_gateway(pid)
            pid = None
        if pid is not None:
            # Graceful in-band restart via SIGUSR1 (mirrors systemd): refuse new turns, wait for
            # in-flight work (restart_after_turn_timeout), then stop() within restart_drain_timeout.
            # The budget must cover BOTH phases plus headroom. A bare SIGTERM would leave
            # restart_requested False (exit 1, "shutting down", lost resume_pending handoff).
            # Announce BEFORE waiting: it can last the full budget and streams into surfaces with
            # no other feedback (desktop updater), where silence reads as "update stuck".
            wait_budget = _get_restart_exit_wait_budget()
            print(f"→ Stopping gateway (PID {pid}) — draining in-flight runs (up to {wait_budget:.0f}s)...")
            if _graceful_restart_via_sigusr1(pid, wait_budget):
                # Planned-restart exit. When launchd supervises, KeepAlive revives it — do NOT
                # kickstart (-k would kill the replacement and restart twice). But a clean exit
                # doesn't prove supervision (detached fallback, unloaded jobs, already-gone PID),
                # so verify a replacement PID appears first.
                if _wait_for_launchd_service_pid(label, pid, timeout=15.0, domain=domain):
                    print("✓ Service restart requested")
                    _clear_launchd_unsupported_marker()
                    return
                print("⚠ launchd did not revive the gateway after its graceful exit — forcing restart")
            else:
                print(f"⚠ Gateway drain timed out after {wait_budget:.0f}s — forcing launchd restart")
        subprocess.run(["launchctl", "kickstart", "-k", target], check=True, timeout=90)
        print("✓ Service restarted")
        _clear_launchd_unsupported_marker()
    except subprocess.CalledProcessError as e:
        if not _launchd_error_indicates_unloaded(e):
            # Not "job unloaded": degrade to detached if the domain is unmanageable (old process
            # already stopped), else re-raise.
            if _launchctl_domain_unsupported(e.returncode):
                _launchd_fallback_to_detached(f"launchctl kickstart exit {e.returncode}")
                return
            raise
        # Job not loaded — bootstrap and start fresh
        print("↻ launchd job was unloaded; reloading")
        plist_path = get_launchd_plist_path()
        try:
            # After a drain the job is almost always still registered, so plain bootstrap would hit
            # EIO; boot the stale label out first rather than routing through _launchctl_bootstrap.
            subprocess.run(["launchctl", "bootout", target], check=False, timeout=90)
            subprocess.run(
                ["launchctl", "bootstrap", _launchd_domain(), str(plist_path)],
                check=True,
                timeout=30,
            )
            subprocess.run(["launchctl", "kickstart", target], check=True, timeout=30)
        except subprocess.CalledProcessError as e2:
            if not _launchctl_domain_unsupported(e2.returncode):
                raise
            _launchd_fallback_to_detached(f"launchctl exit {e2.returncode}")
            return
        print("✓ Service restarted")
        _clear_launchd_unsupported_marker()


# launchd relaunches a KeepAlive job at most ~once per 10s, so a prompt self-restart leaves the label
# with NO pid for most of that window; a verification budget shorter than that reports failure.
LAUNCHD_SUPERVISION_VERIFY_TIMEOUT = 20.0


def wait_for_launchd_gateway_supervision(
    *,
    timeout: float = LAUNCHD_SUPERVISION_VERIFY_TIMEOUT,
    label: str | None = None,
    poll_interval: float = 0.5,
) -> bool:
    """Poll launchd until it supervises a live gateway; True immediately if the detached fallback is active.

    ``launchd_restart`` returns once the restart is *requested* (self-restart or detached plist
    reload are asynchronous), so "returned without raising" can't see a helper that dies before
    bootstrap, nor a ``launchctl bootstrap`` that exits 0 without registering (seen on macOS 26.6.1).
    """
    if _launchd_unsupported_marker_exists():
        return True

    label = label or get_launchd_label()
    deadline = time.monotonic() + max(timeout, 0.0)
    while True:
        if _launchctl_label_supervising_process(label):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(max(poll_interval, 0.01))


def launchd_status(deep: bool = False):
    plist_path = get_launchd_plist_path()
    label = get_launchd_label()
    try:
        result = subprocess.run(["launchctl", "list", label], timeout=10, **_CAPTURE_TEXT)
        service_listed = result.returncode == 0
        list_output = result.stdout
    except subprocess.TimeoutExpired:
        service_listed = False
        list_output = ""

    # `launchctl list` exits 0 whenever the definition is registered — even `state = not running`
    # (macOS 26+) — so only a PID in the output confirms a live process.
    launchd_pid = _parse_launchd_pid_from_list_output(list_output) if service_listed else None

    # Hermes PID tracking — may be a detached fallback process spawned when
    # launchd cannot manage the domain on this host.
    from gateway.status import get_running_pid
    fallback_pid = get_running_pid(cleanup_stale=False)

    # Avoid double-counting: when launchd IS supervising, fallback_pid and launchd_pid point at the
    # same process (the gateway writes both the launchd PID and the Hermes PID file).
    if launchd_pid is not None and fallback_pid == launchd_pid:
        fallback_pid = None

    # Marker written when bootstrap/kickstart failed with 5/125: explains *why* launchd can't
    # supervise even with no fallback running.
    launchd_unsupported = _launchd_unsupported_marker_exists()

    # ── Report ──
    print(f"Launchd plist: {plist_path}")
    if launchd_plist_is_current():
        print("✓ Service definition matches the current Hermes install")
    else:
        print("⚠ Service definition is stale relative to the current Hermes install")
        print("  Run: hermes gateway start")

    if service_listed:
        if launchd_pid is not None:
            print(f"✓ Gateway is supervised by launchd (PID {launchd_pid})")
            print("  Auto-start at login and auto-restart on crash are available.")
            if launchd_unsupported:
                print("  (launchd domain was previously unavailable but is now working)")
        elif launchd_unsupported:
            print("⚠ Gateway service is registered but launchd is not supervising it")
            print("  launchd cannot manage the gateway on this macOS version.")
            if fallback_pid:
                print(f"✓ Detached fallback process is running (PID {fallback_pid})")
                print("  Cron jobs will fire. Stop with: hermes gateway stop")
            else:
                print("✗ No fallback process is running")
                print("  Run: hermes gateway start")
            print("  ⚠ Auto-start at login and auto-restart on crash are NOT available.")
        else:
            print("✓ Gateway service is registered with launchd")
            print(list_output)
            if fallback_pid:
                print(f"  Detached gateway process is running (PID {fallback_pid})")
    else:
        print("✗ Gateway service is not loaded")
        print("  Service definition exists locally but launchd has not loaded it.")
        print("  Run: hermes gateway start")
        if fallback_pid:
            print(f"  Note: a detached gateway process is running (PID {fallback_pid})")

    if deep:
        log_file = get_hermes_home() / "logs" / "gateway.log"
        if log_file.exists():
            print()
            print("Recent logs:")
            subprocess.run(["tail", "-20", str(log_file)], timeout=10)


# =============================================================================
# Gateway Runner
# =============================================================================


def _truthy_env(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _is_official_docker_checkout() -> bool:
    return str(PROJECT_ROOT) == "/opt/hermes" and (PROJECT_ROOT / "docker" / "entrypoint.sh").is_file()


def _running_under_gateway_supervisor() -> bool:
    """True when this process IS the supervisor-launched gateway, so the conflict guard never wedges
    the service into a respawn/refuse loop. Markers: systemd INVOCATION_ID, launchd XPC_SERVICE_NAME
    (shells inherit "0"), s6 HERMES_S6_SUPERVISED_CHILD, or ``--external-supervisor``."""
    return is_gateway_supervisor_process()


def named_profile_served_by_running_multiplexer(profile_name: str | None = None) -> bool:
    """True when a live default multiplexer already ticks this named profile.

    A satellite profile has no gateway.pid of its own; the default multiplexer's ticker fires its
    jobs and serves its platforms. ``profile_name`` defaults to the current HERMES_HOME profile.
    """
    try:
        suffix = profile_name if profile_name is not None else _profile_suffix()
    except Exception:
        return False
    if not suffix or suffix == "default":
        return False

    try:
        from hermes_constants import get_default_hermes_root
        default_root = get_default_hermes_root()
    except Exception:
        return False

    try:
        from gateway.status import _read_pid_record

        default_pid_path = default_root / "gateway.pid"
        rec = _read_pid_record(default_pid_path)
        if not rec:
            return False
        from gateway.status import _pid_exists, _pid_from_record
        pid = _pid_from_record(rec)
        if not pid or not _pid_exists(pid):
            return False

        from gateway.config import _env_multiplex_profiles_override

        cfg_path = default_root / "config.yaml"
        cfg = {}
        if cfg_path.exists():
            from hermes_cli.config import read_user_config_raw

            cfg = read_user_config_raw(cfg_path)

        env_multiplex = _env_multiplex_profiles_override()
        if env_multiplex is False:
            return False
        if env_multiplex is True:
            multiplex = True
        else:
            if not cfg_path.exists():
                return False
            multiplex = bool(
                cfg.get("multiplex_profiles")
                or (cfg.get("gateway", {}) or {}).get("multiplex_profiles")
            )
        if not multiplex:
            return False

        gateway_cfg = cfg.get("gateway", {}) or {}
        if "multiplex_profile_allowlist" in cfg:
            raw_allowlist = cfg.get("multiplex_profile_allowlist")
        else:
            raw_allowlist = gateway_cfg.get("multiplex_profile_allowlist")
        from gateway.config import _normalize_multiplex_profile_allowlist
        from hermes_cli.profiles import normalize_profile_name

        profile_allowlist = _normalize_multiplex_profile_allowlist(raw_allowlist)
        return profile_allowlist is None or normalize_profile_name(suffix) in profile_allowlist
    except Exception:
        logger.debug("Multiplexer-serving probe failed", exc_info=True)
        return False


def _guard_named_profile_under_multiplexer(force: bool = False) -> None:
    """Refuse a named-profile gateway when a multiplexing default gateway already serves it.

    A separate named-profile gateway would double-bind its platforms (two pollers on one bot
    token, port fights). Inert for the default profile or without a live multiplexer. ``--force`` overrides.
    """
    if force:
        return
    try:
        suffix = _profile_suffix()
    except Exception:
        return
    if not named_profile_served_by_running_multiplexer():
        return

    print_error(
        f"The default gateway is running as a profile multiplexer and already "
        f"serves profile '{suffix}'."
    )
    print(
        "  When gateway.multiplex_profiles is on, the default gateway is the\n"
        "  single inbound process for every profile. Starting a separate\n"
        "  gateway for this profile would double-bind its platforms (two\n"
        "  pollers on one bot token, port conflicts).\n"
    )
    print("  Manage the multiplexer instead (from the default profile):")
    print()
    print("    hermes gateway restart")
    print()
    print("  Pass --force to start a separate profile gateway anyway (not")
    print("  recommended while the multiplexer is running).")
    # EX_CONFIG, not 1: this refusal is decided purely by config, so it is permanent. The generated
    # systemd unit pairs Restart=always with StartLimitIntervalSec=0 and relies on
    # RestartPreventExitStatus=GATEWAY_FATAL_CONFIG_EXIT_CODE as its only backstop; exiting 1 left
    # it unarmed and turned a correct refusal into an unbounded restart loop. 78 also hits the s6
    # finish script's 125 "permanent failure" translation like the other fatal-config exits.
    sys.exit(GATEWAY_FATAL_CONFIG_EXIT_CODE)


def _guard_supervised_gateway_conflict(force: bool = False) -> None:
    """Refuse a foreground gateway when a service manager already supervises one.

    A shell-launched ``gateway run`` on a systemd/launchd host becomes a second dispatcher that
    escapes the service cgroup, survives ``systemctl restart``, and concurrently writes the shared
    kanban DB (multi-writer SQLite WAL corruption). ``--force`` starts anyway.
    """
    if force or _running_under_gateway_supervisor():
        return
    try:
        snapshot = get_gateway_runtime_snapshot()
    except Exception:
        logger.debug("Supervised-gateway conflict probe failed", exc_info=True)
        return
    if not (snapshot.service_installed and snapshot.service_running):
        return

    print_error(f"A gateway is already running under {snapshot.manager} for this profile.")
    print(
        "  Starting another one from a shell leaves an orphan dispatcher that\n"
        "  escapes the service, survives restarts, and writes to the same kanban\n"
        "  DB concurrently — which can corrupt it. Restart the supervised gateway\n"
        "  instead:"
    )
    print()
    print("    hermes gateway restart")
    print()
    print(
        "  Pass --force to start a foreground gateway anyway (not recommended\n"
        "  while the service is running)."
    )
    sys.exit(1)


def _guard_existing_gateway_process_conflict(replace: bool = False) -> None:
    """Cheap PID-file preflight before the expensive ``gateway.run`` import (authoritative lock check).

    Supervisor loops re-running bare ``gateway run`` burned memory on plugin discovery just to fail
    "already running". Same user-facing contract; never scans other HERMES_HOME roots.
    """
    if replace or _running_under_gateway_supervisor():
        return
    try:
        from gateway.status import get_running_pid

        pid = get_running_pid()
    except Exception:
        logger.debug("Existing-gateway process probe failed", exc_info=True)
        return
    if pid is None:
        # get_running_pid() filters by the current profile's HERMES_HOME; warn if the PID file
        # belongs to another profile (user switched profiles while the old gateway still runs).
        try:
            from gateway.status import _read_pid_record, _pid_record_belongs_to_current_profile

            stale = _read_pid_record()
            if stale is not None and not _pid_record_belongs_to_current_profile(stale):
                stale_home = stale.get("hermes_home", "<unknown>")
                logger.warning(
                    "PID file belongs to another profile (hermes_home=%s). "
                    "The old gateway may still be running under that profile.",
                    stale_home,
                )
        except Exception:
            pass
        return

    print_error(f"Another gateway instance is already running (PID {pid}).")
    print("  Use 'hermes gateway restart' to replace it,")
    print("  or 'hermes gateway stop' first.")
    print("  Or use 'hermes gateway run --replace' to auto-replace.")
    sys.exit(1)


def _guard_official_docker_root_gateway() -> None:
    """Refuse gateway startup when the official Docker privilege drop was bypassed."""
    if not hasattr(os, "geteuid") or os.geteuid() != 0:
        return
    if _truthy_env(os.getenv("HERMES_ALLOW_ROOT_GATEWAY")):
        return
    if not _is_official_docker_checkout():
        return

    print_error("Refusing to run the Hermes gateway as root inside the official Docker image.")
    print(
        "  The image entrypoint normally drops privileges to the 'hermes' user. "
        "If you override entrypoint in Docker Compose, include "
        "/opt/hermes/docker/entrypoint.sh before the Hermes command."
    )
    print(
        "  Running the gateway as root can leave root-owned files in "
        "$HERMES_HOME and break later non-root dashboard/gateway runs."
    )
    print("  Set HERMES_ALLOW_ROOT_GATEWAY=1 only if you intentionally accept this risk.")
    sys.exit(1)


def _apply_startup_watchdog_config() -> None:
    """Idempotent backstop arming of the startup-liveness watchdog (programmatic run_gateway callers).

    Must run AFTER the process-conflict guards (a --replace loser must not arm one). config.yaml
    gateway.startup_watchdog* is the user surface; env vars bridge it because the argv fast-path
    arms before config loads, and explicit env wins. arm() is idempotent, so a config timeout
    needs disarm+re-arm. GatewayRunner disarms once the event loop is live.
    """
    try:
        from hermes_startup_watchdog import (
            ENV_STARTUP_WATCHDOG,
            ENV_STARTUP_WATCHDOG_TIMEOUT_S,
            arm_startup_watchdog,
            disarm_startup_watchdog,
            startup_watchdog_disabled,
        )
        _sw_timeout_bridged = False
        try:
            from hermes_cli.config import load_config as _sw_load_config
            _gw_cfg = (_sw_load_config() or {}).get("gateway", {}) or {}
            if ENV_STARTUP_WATCHDOG not in os.environ and not _gw_cfg.get("startup_watchdog", True):
                os.environ[ENV_STARTUP_WATCHDOG] = "0"
            _sw_timeout = _gw_cfg.get("startup_watchdog_timeout_seconds")
            if (ENV_STARTUP_WATCHDOG_TIMEOUT_S not in os.environ and _sw_timeout is not None):
                os.environ[ENV_STARTUP_WATCHDOG_TIMEOUT_S] = str(_sw_timeout)
                _sw_timeout_bridged = True
        except Exception:
            pass
        if startup_watchdog_disabled():
            disarm_startup_watchdog()
        else:
            if _sw_timeout_bridged:
                disarm_startup_watchdog()
            arm_startup_watchdog()
    except Exception:
        pass


def _absorb_windows_console_controls() -> None:
    """Make a detached Windows gateway ignore console-control broadcasts from sibling CLIs."""
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        if hasattr(signal, "SIGBREAK"):
            signal.signal(signal.SIGBREAK, signal.SIG_IGN)
    except (OSError, ValueError):
        pass  # SetConsoleCtrlHandler unavailable (rare) — best-effort
    # signal only hooks SIGINT/SIGBREAK; SetConsoleCtrlHandler(NULL, TRUE) ignores ALL console
    # control events (CTRL_CLOSE/CTRL_LOGOFF included), as background services should.
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        kernel32.SetConsoleCtrlHandler(None, 1)
    except (OSError, AttributeError):
        pass


def _make_exit_diag():
    """Return an ``_exit_diag(tag, **extra)`` recorder writing to ``logs/gateway-exit-diag.log``.

    Captures every way ``asyncio.run()`` can return, for chasing silent Windows gateway deaths.
    Opt out with HERMES_GATEWAY_EXIT_DIAG=0.
    """
    from datetime import datetime as _dt, timezone as _tz

    def _exit_diag(tag: str, **extra: object) -> None:
        if os.environ.get("HERMES_GATEWAY_EXIT_DIAG", "1") != "1":
            return
        try:
            from hermes_constants import get_hermes_home as _ghh

            log_dir = _ghh() / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            ts = _dt.now(_tz.utc).isoformat()
            line = {
                "ts": ts,
                "tag": tag,
                "pid": os.getpid(),
                "python": sys.version.split()[0],
                "platform": sys.platform,
                **extra,
            }
            import json as _json

            with open(log_dir / "gateway-exit-diag.log", "a", encoding="utf-8") as f:
                f.write(_json.dumps(line, default=str) + "\n")
        except Exception:
            pass  # never let the diagnostic itself crash the gateway

    return _exit_diag


def _respawn_storm_backoff() -> None:
    """Portable app-level respawn-storm circuit breaker (works where supervisors lack a floor).

    Defaults mirror DEFAULT_CONFIG ``gateway.respawn_storm``; HERMES_GATEWAY_MAX_STARTS /
    HERMES_GATEWAY_START_WINDOW_S override. max_starts <= 0 disables. Never blocks startup.
    """
    try:
        import time as _time

        from gateway.status import record_start_and_check_storm

        _max_starts = 5
        _win = 120.0
        try:
            from hermes_cli.config import load_config

            _cfg = load_config()
            _gw = _cfg.get("gateway") if isinstance(_cfg, dict) else None
            _rs = _gw.get("respawn_storm") if isinstance(_gw, dict) else None
            if isinstance(_rs, dict):
                if isinstance(_rs.get("max_starts"), int):
                    _max_starts = _rs["max_starts"]
                if isinstance(_rs.get("window_seconds"), (int, float)):
                    _win = float(_rs["window_seconds"])
        except Exception:
            pass
        try:
            _env_starts = os.getenv("HERMES_GATEWAY_MAX_STARTS")
            if _env_starts is not None:
                _max_starts = int(_env_starts)
        except ValueError:
            pass
        try:
            _env_win = os.getenv("HERMES_GATEWAY_START_WINDOW_S")
            if _env_win is not None:
                _win = float(_env_win)
        except ValueError:
            pass
        _storm = (
            record_start_and_check_storm(max_starts=_max_starts, window_s=_win)
            if _max_starts > 0
            else None
        )
        if _storm is not None:
            logger.warning(
                "Gateway (re)started %d times in %.0fs — backing off %.0fs to break a respawn storm.",
                _storm.count,
                _storm.window_s,
                _storm.backoff_s,
            )
            # Tell the startup watchdog the backoff sleep is intentional, not a parked deadlock.
            try:
                from gateway.startup_watchdog import kick_startup_watchdog

                kick_startup_watchdog(extra_s=_storm.backoff_s)
            except Exception:
                pass
            _time.sleep(_storm.backoff_s)
    except Exception as _be:
        logger.debug("respawn-storm breaker check failed (non-fatal): %s", _be)


def run_gateway(verbose: int = 0, quiet: bool = False, replace: bool = False, force: bool = False):
    """Run the gateway in foreground. verbose: 1=INFO, 2+=DEBUG on stderr; quiet: no stderr logs;
    replace: kill an existing instance first (avoids systemd restart loops); force: skip the
    supervised-gateway conflict guard."""
    _guard_official_docker_root_gateway()
    _guard_named_profile_under_multiplexer(force=force)
    _guard_supervised_gateway_conflict(force=force)
    _guard_existing_gateway_process_conflict(replace=replace)
    sys.path.insert(0, str(PROJECT_ROOT))
    _apply_startup_watchdog_config()

    # Detached Windows runs (HERMES_GATEWAY_DETACHED=1, or non-TTY for older wrappers) ignore
    # console-control broadcasts from sibling CLIs; foreground runs keep Ctrl+C-to-stop.
    try:
        _stdin_is_tty = bool(sys.stdin and sys.stdin.isatty())
    except (ValueError, OSError):
        _stdin_is_tty = False
    _console_window_attached = _windows_console_window_attached()
    _gateway_detached = os.getenv("HERMES_GATEWAY_DETACHED", "").strip().lower() in {"1", "true", "yes", "on"}
    _breakaway = _windows_gateway_breakaway_state()
    _absorb = _windows_gateway_should_absorb_console_controls()
    if _absorb:
        _absorb_windows_console_controls()

    # Refresh the systemd unit on every boot so restart settings stay current even after an
    # exit-code-75 respawn (stale-code or /restart), which bypasses `hermes gateway restart`.
    if supports_systemd_services():
        try:
            refresh_systemd_unit_if_needed(system=False)
        except Exception:
            pass  # best-effort; don't block gateway startup

    from gateway.run import start_gateway

    print("┌─────────────────────────────────────────────────────────┐")
    print("│           ⚕ Hermes Gateway Starting...                 │")
    print("├─────────────────────────────────────────────────────────┤")
    print("│  Messaging platforms + cron scheduler                    │")
    print("│  Press Ctrl+C to stop                                   │")
    print("└─────────────────────────────────────────────────────────┘")
    print()

    # Exit 1 if no platform connects so systemd Restart=always retries transient errors.
    verbosity = None if quiet else verbose

    import atexit as _atexit
    import traceback as _traceback

    _exit_diag = _make_exit_diag()
    _exit_diag(
        "gateway.start",
        replace=replace,
        argv=sys.argv,
        stdin_is_tty=_stdin_is_tty,
        console_window_attached=_console_window_attached,
        detached=_gateway_detached,
        breakaway=_breakaway,
        absorb_windows_console_controls=_absorb,
    )
    _atexit.register(lambda: _exit_diag("atexit.hook", sys_exc=repr(sys.exc_info())))

    _respawn_storm_backoff()

    def _hard_exit_after_gateway_teardown(code: int) -> None:
        # Mirror gateway.run.main()'s wedge-proof exit: after graceful teardown, bypass Python
        # finalization so non-daemon threads (in-flight cron jobs) can't delay a /restart by minutes.
        from gateway.run import _exit_after_graceful_shutdown

        _exit_after_graceful_shutdown(code)

    success = False
    try:
        success = asyncio.run(start_gateway(replace=replace, verbosity=verbosity))
        _exit_diag("asyncio.run.returned", success=success)
    except KeyboardInterrupt:
        # Detached Windows runs absorb SIGINT above; keep the handler for console runs.
        _exit_diag("asyncio.run.KeyboardInterrupt", traceback=_traceback.format_exc())
        print("\nGateway stopped.")
        _hard_exit_after_gateway_teardown(0)
        return  # unreachable in production (os._exit); guard for test stubs
    except SystemExit as e:
        _exit_diag("asyncio.run.SystemExit", code=getattr(e, "code", None), traceback=_traceback.format_exc())
        if e.code is None:
            _code = 0
        elif isinstance(e.code, int):
            _code = e.code
        else:
            _code = 1
        _hard_exit_after_gateway_teardown(_code)
    except BaseException as e:
        # Everything else (CancelledError, exotic BaseExceptions): log the cause, then re-raise.
        _exit_diag(
            "asyncio.run.exception",
            exc_type=type(e).__name__,
            exc_repr=repr(e),
            traceback=_traceback.format_exc(),
        )
        raise
    if not success:
        _exit_diag("gateway.exit_nonzero")
        _hard_exit_after_gateway_teardown(1)
    _exit_diag("gateway.exit_clean")
    _hard_exit_after_gateway_teardown(0)


# =============================================================================
# Gateway Setup (Interactive Messaging Platform Configuration)
# =============================================================================

# Built-in per-platform setup config (env vars, instructions, prompts). Telegram, WhatsApp, Email,
# SMS, etc. live in plugins/platforms/<name>/ and are discovered via the platform registry.
_PLATFORMS = [
    {
        "key": "mattermost", "label": "Mattermost", "emoji": "💬", "token_var": "MATTERMOST_TOKEN",
        "setup_instructions": [
            "1. In Mattermost: Integrations → Bot Accounts → Add Bot Account",
            "   (System Console → Integrations → Bot Accounts must be enabled)",
            "2. Give it a username (e.g. hermes) and copy the bot token",
            "3. Works with any self-hosted Mattermost instance — enter your server URL",
            "4. To find your user ID: click your avatar (top-left) → Profile",
            "   Your user ID is displayed there — click it to copy.",
            "   ⚠ This is NOT your username — it's a 26-character alphanumeric ID.",
            "5. To get a channel ID: click the channel name → View Info → copy the ID",
        ],
        "vars": [
            {"name": "MATTERMOST_URL", "prompt": "Server URL (e.g. https://mm.example.com)",
             "password": False, "help": "Your Mattermost server URL. Works with any self-hosted instance."},
            {"name": "MATTERMOST_TOKEN", "prompt": "Bot token", "password": True,
             "help": "Paste the bot token from step 2 above."},
            {"name": "MATTERMOST_ALLOWED_USERS", "prompt": "Allowed user IDs (comma-separated)",
             "password": False, "is_allowlist": True, "help": "Your Mattermost user ID from step 4 above."},
            {"name": "MATTERMOST_HOME_CHANNEL",
             "prompt": "Home channel ID (for cron/notification delivery, or empty to set later with /set-home)",
             "password": False, "help": "Channel ID where Hermes delivers cron results and notifications."},
            {"name": "MATTERMOST_REPLY_MODE",
             "prompt": "Reply mode — 'off' for flat messages, 'thread' for threaded replies (default: off)",
             "password": False,
             "help": "off = flat channel messages, thread = replies nest under your message."},
        ],
    },
    {"key": "signal", "label": "Signal", "emoji": "📡", "token_var": "SIGNAL_HTTP_URL"},
    {"key": "weixin", "label": "Weixin / WeChat", "emoji": "💬", "token_var": "WEIXIN_ACCOUNT_ID"},
    {
        "key": "bluebubbles", "label": "BlueBubbles (iMessage)",
        "emoji": "💬", "token_var": "BLUEBUBBLES_SERVER_URL",
        "setup_instructions": [
            "1. Install BlueBubbles on a Mac that will act as your iMessage server:",
            "   https://bluebubbles.app/",
            "2. Complete the BlueBubbles setup wizard — sign in with your Apple ID",
            "3. In BlueBubbles Settings → API, note the Server URL and password",
            "4. The server URL is typically http://<your-mac-ip>:1234",
            "5. Hermes connects via the BlueBubbles REST API and receives",
            "   incoming messages via a local webhook",
            "6. To authorize users, use DM pairing: hermes pairing generate bluebubbles",
            "   Share the code — the user sends it via iMessage to get approved",
        ],
        "vars": [
            {"name": "BLUEBUBBLES_SERVER_URL",
             "prompt": "BlueBubbles server URL (e.g. http://192.168.1.10:1234)", "password": False,
             "help": "The URL shown in BlueBubbles Settings → API."},
            {"name": "BLUEBUBBLES_PASSWORD", "prompt": "BlueBubbles server password", "password": True,
             "help": "The password shown in BlueBubbles Settings → API."},
            {"name": "BLUEBUBBLES_ALLOWED_USERS",
             "prompt": "Pre-authorized phone numbers or iMessage IDs (comma-separated, or leave empty for DM pairing)",
             "password": False, "is_allowlist": True,
             "help": "Optional — pre-authorize specific users. Leave empty to use DM pairing instead (recommended)."},
            {"name": "BLUEBUBBLES_HOME_CHANNEL",
             "prompt": "Home channel (phone number or iMessage ID for cron/notifications, or empty)",
             "password": False,
             "help": "Phone number or Apple ID to deliver cron results and notifications to."},
        ],
    },
    {
        "key": "qqbot", "label": "QQ Bot", "emoji": "🐧", "token_var": "QQ_APP_ID",
        "setup_instructions": [
            "1. Register a QQ Bot application at q.qq.com",
            "2. Note your App ID and App Secret from the application page",
            "3. Enable the required intents (C2C, Group, Guild messages)",
            "4. Configure sandbox or publish the bot",
        ],
        "vars": [
            {"name": "QQ_APP_ID", "prompt": "QQ Bot App ID", "password": False,
             "help": "Your QQ Bot App ID from q.qq.com."},
            {"name": "QQ_CLIENT_SECRET", "prompt": "QQ Bot App Secret", "password": True,
             "help": "Your QQ Bot App Secret from q.qq.com."},
            {"name": "QQ_ALLOWED_USERS",
             "prompt": "Allowed user OpenIDs (comma-separated, leave empty for open access)",
             "password": False, "is_allowlist": True,
             "help": "Optional — restrict DM access to specific user OpenIDs."},
            {"name": "QQBOT_HOME_CHANNEL",
             "prompt": "Home channel (user/group OpenID for cron delivery, or empty)", "password": False,
             "help": "OpenID to deliver cron results and notifications to."},
        ],
    },
    {
        "key": "yuanbao", "label": "Yuanbao", "emoji": "💎", "token_var": "YUANBAO_APP_ID",
        "setup_instructions": [
            "1. Download the Yuanbao app from https://yuanbao.tencent.com/",
            "2. In the app, go to PAI → My Bot and create a new bot",
            "3. After the bot is created, copy the App ID and App Secret",
            "4. Enter them below and Hermes will connect automatically over WebSocket",
        ],
        "vars": [
            {"name": "YUANBAO_APP_ID", "prompt": "App ID", "password": False,
             "help": "The App ID from your Yuanbao IM Bot credentials."},
            {"name": "YUANBAO_APP_SECRET", "prompt": "App Secret", "password": True,
             "help": "The App Secret (used for HMAC signing) from your Yuanbao IM Bot."},
        ],
    },
]


def _all_platforms() -> list[dict]:
    """Built-in ``_PLATFORMS`` plus registry plugin platforms (adapted to the same dict shape, source
    in ``_registry_entry``). Plugins are discovered on first call so the setup menu works without a
    running gateway. Matrix is hidden on Windows: python-olm has no wheel or native build (use WSL).
    """
    # Idempotent. Bundled ``kind: platform`` plugins auto-load; user-installed ones under
    # ~/.hermes/plugins/ still need ``plugins.enabled`` (untrusted code).
    try:
        from hermes_cli.plugins import discover_plugins

        discover_plugins()
    except Exception as e:
        logger.debug("plugin discovery failed during platform enumeration: %s", e)

    platforms = [dict(p) for p in _PLATFORMS]

    if sys.platform == "win32":
        platforms = [p for p in platforms if p.get("key") != "matrix"]

    by_key = {p["key"]: p for p in platforms}

    try:
        from gateway.platform_registry import platform_registry
    except Exception:
        return platforms

    for entry in platform_registry.all_entries():
        if entry.name in by_key:
            continue  # built-in already covers it
        # Matrix hidden on Windows (python-olm has no wheel) for registry-discovered entries too.
        if sys.platform == "win32" and entry.name == "matrix":
            continue
        platforms.append(
            {
                "key": entry.name,
                "label": entry.label,
                "emoji": entry.emoji,
                "token_var": entry.required_env[0] if entry.required_env else "",
                "install_hint": entry.install_hint,
                "_registry_entry": entry,
            }
        )
    return platforms


def _platform_status(platform: dict) -> str:
    """Plain-text status string; uncolored because ANSI codes break curses menu width math."""
    entry = platform.get("_registry_entry")
    if entry is not None:
        configured = False
        # Prefer is_connected (env + config.yaml) over check_fn (deps/env presence only).
        if entry.is_connected is not None:
            try:
                from gateway.config import PlatformConfig

                synthetic = PlatformConfig(enabled=True)
                configured = bool(entry.is_connected(synthetic))
            except Exception:
                configured = False
        else:
            # No is_connected hook: check_fn is a coarse deps gate. Never fall back to it when
            # is_connected returned False, or "SDK installed" would override "no token".
            try:
                configured = bool(entry.check_fn())
            except Exception:
                configured = False
        return "configured" if configured else "not configured"

    token_var = platform.get("token_var", "")
    if not token_var:
        return "not configured"
    # Built-ins needing a second credential to count as fully configured.
    second_var = {"signal": "SIGNAL_ACCOUNT", "weixin": "WEIXIN_TOKEN"}.get(platform.get("key"))
    present = [bool(get_env_value(token_var))]
    if second_var:
        present.append(bool(get_env_value(second_var)))
    if all(present):
        return "configured"
    if any(present):
        return "partially configured"
    return "not configured"


def _runtime_health_lines() -> list[str]:
    """Summarize the latest persisted gateway runtime health state."""
    try:
        from gateway.status import read_runtime_status, runtime_status_is_stale, runtime_status_pid_is_live
    except Exception:
        return []

    state = read_runtime_status()
    if not state:
        return []

    lines: list[str] = []
    gateway_state = state.get("gateway_state")
    exit_reason = state.get("exit_reason")
    active_agents = state.get("active_agents")
    restart_requested = state.get("restart_requested")
    platforms = state.get("platforms", {}) or {}

    for platform, pdata in platforms.items():
        if pdata.get("state") == "fatal":
            message = pdata.get("error_message") or "unknown error"
            lines.append(f"⚠ {platform}: {message}")

    # A live-claiming snapshot can outlive an ungracefully killed gateway (taskkill /F, OOM). Past
    # the freshness TTL with the recorded PID gone, say so instead of rendering stale live state.
    if (
        gateway_state in ("running", "starting", "draining")
        and runtime_status_is_stale(state)
        and not runtime_status_pid_is_live(state)
    ):
        lines.append(
            f"⚠ Stale gateway_state.json: recorded state '{gateway_state}' but the "
            "recorded process is gone (likely an ungraceful shutdown)"
        )
        return lines

    if gateway_state == "startup_failed" and exit_reason:
        lines.append(f"⚠ Last startup issue: {exit_reason}")
    elif gateway_state == "draining":
        action = "restart" if restart_requested else "shutdown"
        from gateway.status import parse_active_agents

        count = parse_active_agents(active_agents)
        lines.append(f"⏳ Gateway draining for {action} ({count} active agent(s))")
    elif gateway_state == "stopped" and exit_reason:
        lines.append(f"⚠ Last shutdown reason: {exit_reason}")

    return lines


def _set_platform_unauthorized_dm_behavior(platform_key: str, behavior: str) -> None:
    """Persist a platform-specific unauthorized-DM policy in config.yaml."""
    write_platform_config_field(platform_key, "unauthorized_dm_behavior", behavior, raw=True)


def _confirm_reconfigure(label: str, *env_vars: str) -> bool:
    """False when ``label`` is already configured (all ``env_vars`` set) and the user declines."""
    if all(get_env_value(v) for v in env_vars):
        print()
        print_success(f"{label} is already configured.")
        if not prompt_yes_no(f"  Reconfigure {label}?", False):
            return False
    return True


def _offer_home_channel(home_var: str, user_id: str, what: str) -> None:
    """Offer to persist ``user_id`` as ``home_var`` (e.g. "your Telegram user ID")."""
    if prompt_yes_no(f"  Use {what} ({user_id}) as the home channel?", True):
        save_env_value(home_var, user_id)
        print_success(f"  Home channel set to {user_id}")


def _save_env_values(**values: str) -> None:
    for name, value in values.items():
        save_env_value(name, value)


def _prompt_unauthorized_access(*, is_email: bool) -> None:
    """No allowlist was given — ask open access vs DM pairing vs skip/silent, and persist."""
    print()
    if is_email:
        access_choices = [
            "Enable open access (any email sender can message the bot)",
            "Use DM pairing (unknown email senders receive a pairing code)",
            "Keep unknown senders silent",
        ]
        default_access_idx = 2
    else:
        access_choices = [
            "Enable open access (anyone can message the bot)",
            "Use DM pairing (unknown users request access, you approve with 'hermes pairing approve')",
            "Skip for now (bot will deny all users until configured)",
        ]
        default_access_idx = 1
    access_idx = prompt_choice(
        "  How should unauthorized users be handled?",
        access_choices,
        default_access_idx,
    )
    if access_idx == 0:
        save_env_value("EMAIL_ALLOW_ALL_USERS" if is_email else "GATEWAY_ALLOW_ALL_USERS", "true")
        print_warning("  Open access enabled — anyone can use your bot!")
    elif access_idx == 1:
        if is_email:
            _set_platform_unauthorized_dm_behavior("email", "pair")
        print_success("  DM pairing mode — users will receive a code to request access.")
        print_info("  Approve with: hermes pairing approve <platform> <code>")
    elif is_email:
        print_success("  Unknown email senders will be ignored.")
    else:
        print_info("  Skipped — configure later with 'hermes gateway setup'")


def _setup_standard_platform(platform: dict):
    """Interactive setup for Telegram, Discord, or Slack."""
    from hermes_cli.setup_hidden_env import is_setup_hidden_env as _is_setup_hidden_env

    emoji = platform["emoji"]
    label = platform["label"]
    token_var = platform["token_var"]

    print()
    print(color(f"  ─── {emoji} {label} Setup ───", Colors.CYAN))

    instructions = platform.get("setup_instructions")
    if instructions:
        print()
        for line in instructions:
            print_info(f"  {line}")

    if not _confirm_reconfigure(label, token_var):
        return

    auto_token_saved = False
    auto_owner_user_id = None
    if platform.get("key") == "telegram":
        print()
        print_info("  Telegram can be configured automatically with a managed bot:")
        print_info("  [1] Automatic (scan QR → confirm in Telegram → done)")
        print_info("  [2] Manual BotFather token")
        choice = prompt("  Choice [1/2]", default="1")
        if choice.strip() == "1":
            try:
                from hermes_cli.telegram_managed_bot import (
                    auto_setup_telegram_bot_result,
                    is_valid_telegram_bot_token,
                )
            except ImportError:
                print_warning("  Automatic setup is unavailable in this install.")
            else:
                result = auto_setup_telegram_bot_result()
                if result and is_valid_telegram_bot_token(result.token):
                    save_env_value(token_var, result.token)
                    print_success("  Saved TELEGRAM_BOT_TOKEN")
                    auto_token_saved = True
                    auto_owner_user_id = result.owner_user_id
                else:
                    if result:
                        print_warning("  Automatic setup returned an invalid Telegram token.")
                    print()
                    print_info("  Falling back to manual setup...")

    allowed_val_set = None  # Track if user set an allowlist (for home channel offer)

    # Skip knobs the setup forms hide (home channel, reply mode, proxy, mention behavior); they're
    # self-configuring (/sethome) and asking made a 2-question setup a 5-question one.
    required_names = {token_var}
    setup_vars = [
        v
        for v in platform["vars"]
        if v["name"] in required_names
        or v.get("is_allowlist")
        or not _is_setup_hidden_env(v["name"])
    ]

    for var in setup_vars:
        print()
        print_info(f"  {var['help']}")
        existing = get_env_value(var["name"])
        if existing and var["name"] != token_var:
            print_info(f"  Current: {existing}")

        if auto_token_saved and var["name"] == token_var:
            print_info("  Token saved by automatic setup.")
            continue

        if var.get("is_allowlist"):
            if "TELEGRAM" in var["name"] and auto_owner_user_id:
                detected_id = str(auto_owner_user_id)
                print_success(f"  Detected your Telegram user ID: {detected_id}")
                if prompt_yes_no("  Allow this Telegram account to use the bot?", True):
                    extra = prompt(
                        "  Additional allowed user IDs (comma-separated, optional)",
                        password=False,
                    )
                    ids = [detected_id]
                    for uid in extra.replace(" ", "").split(","):
                        if uid and uid not in ids:
                            ids.append(uid)
                    cleaned = ",".join(ids)
                    save_env_value(var["name"], cleaned)
                    print_success("  Saved — only these users can interact with the bot.")
                    allowed_val_set = cleaned
                    continue

            print_info("  The gateway DENIES all users by default for security.")
            print_info("  Enter user IDs to create an allowlist, or leave empty")
            print_info("  and you'll be asked about open access next.")
            value = prompt(f"  {var['prompt']}", password=False)
            if value:
                cleaned = value.replace(" ", "")
                # For Discord, strip common prefixes (user:123, <@123>, <@!123>)
                if "DISCORD" in var["name"]:
                    parts = []
                    for uid in cleaned.split(","):
                        uid = uid.strip()
                        if uid.startswith("<@") and uid.endswith(">"):
                            uid = uid.lstrip("<@!").rstrip(">")
                        if uid.lower().startswith("user:"):
                            uid = uid[5:]
                        if uid:
                            parts.append(uid)
                    cleaned = ",".join(parts)
                save_env_value(var["name"], cleaned)
                print_success("  Saved — only these users can interact with the bot.")
                allowed_val_set = cleaned
            else:
                _prompt_unauthorized_access(is_email=platform.get("key") == "email")
            continue

        value = prompt(f"  {var['prompt']}", password=var.get("password", False))
        if value:
            save_env_value(var["name"], value)
            print_success(f"  Saved {var['name']}")
        elif var["name"] == token_var:
            print_warning(f"  Skipped — {label} won't work without this.")
            return
        else:
            print_info("  Skipped (can configure later)")

    # Offer the first allowlisted user ID as home channel when none is set (Telegram DMs).
    home_var = f"{label.upper()}_HOME_CHANNEL"
    home_val = get_env_value(home_var)
    if allowed_val_set and not home_val and label == "Telegram":
        first_id = allowed_val_set.split(",")[0].strip()
        if first_id:
            _offer_home_channel(home_var, first_id, "your user ID")

    print()
    print_success(f"{emoji} {label} configured!")


# WhatsApp/DingTalk/WeCom/Feishu setup flows live in their plugins' adapter.py::interactive_setup.


def _running_under_s6() -> bool:
    from hermes_cli.service_manager import detect_service_manager

    return detect_service_manager() == "s6"


def _systemd_unit_installed() -> bool:
    return supports_systemd_services() and (
        get_systemd_unit_path(system=False).exists()
        or get_systemd_unit_path(system=True).exists()
    )


def _is_service_installed() -> bool:
    """Check if the gateway is installed as a system service."""
    return _installed_service_kind() is not None


def _is_service_running() -> bool:
    """Check if the gateway service is currently running."""
    if supports_systemd_services():
        return _systemd_unit_is_active(False) or _systemd_unit_is_active(True)
    elif is_macos() and get_launchd_plist_path().exists():
        try:
            return _launchd_service_registered(get_launchd_label(), timeout=10)
        except subprocess.TimeoutExpired:
            return False
    # Windows "installed" doesn't mean "running"; like manual runs, a live gateway process decides.
    return len(find_gateway_pids()) > 0


def _setup_weixin():
    """Interactive setup for Weixin / WeChat personal accounts."""
    print()
    print(color("  ─── 💬 Weixin / WeChat Setup ───", Colors.CYAN))
    print()
    print_info("  1. Hermes will open Tencent iLink QR login in this terminal.")
    print_info("  2. Use WeChat to scan and confirm the QR code.")
    print_info("  3. Hermes will store the returned account_id/token in ~/.hermes/.env.")
    print_info("  4. This adapter supports native text, image, video, and document delivery.")

    if not _confirm_reconfigure("Weixin", "WEIXIN_ACCOUNT_ID", "WEIXIN_TOKEN"):
        return

    try:
        from gateway.platforms.weixin import check_weixin_requirements, qr_login
    except Exception as exc:
        print_error(f"  Weixin adapter import failed: {exc}")
        print_info("  Install gateway dependencies first, then retry.")
        return

    if not check_weixin_requirements():
        print_error("  Missing dependencies: Weixin needs aiohttp and cryptography.")
        print_info("  Install them, then rerun `hermes gateway setup`.")
        return

    print()
    if not prompt_yes_no("  Start QR login now?", True):
        print_info("  Cancelled.")
        return

    import asyncio

    try:
        credentials = asyncio.run(qr_login(str(get_hermes_home())))
    except KeyboardInterrupt:
        print()
        print_warning("  Weixin setup cancelled.")
        return
    except Exception as exc:
        print_error(f"  QR login failed: {exc}")
        return

    if not credentials:
        print_warning("  QR login did not complete.")
        return

    account_id = credentials.get("account_id", "")
    token = credentials.get("token", "")
    base_url = credentials.get("base_url", "")
    user_id = credentials.get("user_id", "")

    save_env_value("WEIXIN_ACCOUNT_ID", account_id)
    save_env_value("WEIXIN_TOKEN", token)
    if base_url:
        save_env_value("WEIXIN_BASE_URL", base_url)
    save_env_value(
        "WEIXIN_CDN_BASE_URL",
        get_env_value("WEIXIN_CDN_BASE_URL") or "https://novac2c.cdn.weixin.qq.com/c2c",
    )

    print()
    access_choices = [
        "Use DM pairing approval (recommended)",
        "Allow all direct messages",
        "Only allow listed user IDs",
        "Disable direct messages",
    ]
    access_idx = prompt_choice("  How should direct messages be authorized?", access_choices, 0)
    if access_idx == 2:
        allowlist = prompt(
            "  Allowed Weixin user IDs (comma-separated)", user_id or "", password=False
        ).replace(" ", "")
        _save_env_values(
            WEIXIN_DM_POLICY="allowlist", WEIXIN_ALLOW_ALL_USERS="false", WEIXIN_ALLOWED_USERS=allowlist
        )
        print_success("  Weixin allowlist saved.")
    else:
        policy, allow_all = {0: ("pairing", "false"), 1: ("open", "true")}.get(
            access_idx, ("disabled", "false")
        )
        _save_env_values(WEIXIN_DM_POLICY=policy, WEIXIN_ALLOW_ALL_USERS=allow_all, WEIXIN_ALLOWED_USERS="")
        if access_idx == 0:
            print_success("  DM pairing enabled.")
            print_info(
                "  Unknown DM users can request access and you approve them with `hermes pairing approve`."
            )
        elif access_idx == 1:
            print_warning("  Open DM access enabled for Weixin.")
        else:
            print_warning("  Direct messages disabled.")

    print()
    for note_line in (
        "  Note: QR login connects an iLink bot identity (e.g. ...@im.bot), not a",
        "  scriptable personal WeChat account. Ordinary WeChat groups typically cannot",
        "  invite an @im.bot identity, and iLink does not deliver ordinary-group events",
        "  to most bot accounts. The settings below only apply when iLink actually",
        "  delivers group events for your account type — otherwise DM remains the only",
        "  working channel regardless of this choice.",
    ):
        print_info(note_line)
    group_choices = [
        "Disable group chats (recommended)",
        "Allow all group chats",
        "Only allow listed group chat IDs",
    ]
    group_idx = prompt_choice("  How should group chats be handled?", group_choices, 0)
    if group_idx == 0:
        _save_env_values(WEIXIN_GROUP_POLICY="disabled", WEIXIN_GROUP_ALLOWED_USERS="")
        print_info("  Group chats disabled.")
    elif group_idx == 1:
        _save_env_values(WEIXIN_GROUP_POLICY="open", WEIXIN_GROUP_ALLOWED_USERS="")
        print_warning("  All group chats enabled (only takes effect if iLink delivers group events).")
    else:
        allow_groups = prompt(
            "  Allowed group chat IDs (comma-separated, not member user IDs)",
            "",
            password=False,
        ).replace(" ", "")
        _save_env_values(WEIXIN_GROUP_POLICY="allowlist", WEIXIN_GROUP_ALLOWED_USERS=allow_groups)
        print_success("  Group allowlist saved (only takes effect if iLink delivers group events).")

    if user_id:
        print()
        _offer_home_channel("WEIXIN_HOME_CHANNEL", user_id, "your Weixin user ID")

    print()
    print_success("Weixin configured!")
    print_info(f"  Account ID: {account_id}")
    if user_id:
        print_info(f"  User ID: {user_id}")


def _setup_qqbot():
    """Interactive setup for QQ Bot — scan-to-configure or manual credentials."""
    print()
    print(color("  ─── 🐧 QQ Bot Setup ───", Colors.CYAN))

    if not _confirm_reconfigure("QQ Bot", "QQ_APP_ID", "QQ_CLIENT_SECRET"):
        return

    print()
    method_choices = [
        "Scan QR code to add bot automatically (recommended)",
        "Enter existing App ID and App Secret manually",
    ]
    method_idx = prompt_choice("  How would you like to set up QQ Bot?", method_choices, 0)

    credentials = None

    if method_idx == 0:
        try:
            from gateway.platforms.qqbot import qr_register

            credentials = qr_register()
        except KeyboardInterrupt:
            print()
            print_warning("  QQ Bot setup cancelled.")
            return
        if not credentials:
            print_info("  QR setup did not complete. Continuing with manual input.")

    if not credentials:
        print()
        print_info("  Go to https://q.qq.com to register a QQ Bot application.")
        print_info("  Note your App ID and App Secret from the application page.")
        print()
        app_id = prompt("  App ID", password=False)
        if not app_id:
            print_warning("  Skipped — QQ Bot won't work without an App ID.")
            return
        app_secret = prompt("  App Secret", password=True)
        if not app_secret:
            print_warning("  Skipped — QQ Bot won't work without an App Secret.")
            return
        credentials = {"app_id": app_id.strip(), "client_secret": app_secret.strip(), "user_openid": ""}

    save_env_value("QQ_APP_ID", credentials["app_id"])
    save_env_value("QQ_CLIENT_SECRET", credentials["client_secret"])

    user_openid = credentials.get("user_openid", "")

    print()
    access_choices = [
        "Use DM pairing approval (recommended)",
        "Allow all direct messages",
        "Only allow listed user OpenIDs",
    ]
    access_idx = prompt_choice("  How should direct messages be authorized?", access_choices, 0)
    if access_idx == 0:
        save_env_value("QQ_ALLOW_ALL_USERS", "false")
        allowed = ""
        if user_openid:
            print()
            if prompt_yes_no(f"  Add yourself ({user_openid}) to the allow list?", True):
                allowed = user_openid
                print_success(f"  Allow list set to {user_openid}")
        save_env_value("QQ_ALLOWED_USERS", allowed)
        print_success("  DM pairing enabled.")
        print_info("  Unknown users can request access; approve with `hermes pairing approve`.")
    elif access_idx == 1:
        _save_env_values(QQ_ALLOW_ALL_USERS="true", QQ_ALLOWED_USERS="")
        print_warning("  Open DM access enabled for QQ Bot.")
    else:
        allowlist = prompt(
            "  Allowed user OpenIDs (comma-separated)", user_openid or "", password=False
        ).replace(" ", "")
        _save_env_values(QQ_ALLOW_ALL_USERS="false", QQ_ALLOWED_USERS=allowlist)
        print_success("  Allowlist saved.")

    if user_openid:
        print()
        _offer_home_channel("QQBOT_HOME_CHANNEL", user_openid, "your QQ user ID")
    else:
        print()
        home_channel = prompt("  Home channel OpenID (for cron/notifications, or empty)", password=False)
        if home_channel:
            save_env_value("QQBOT_HOME_CHANNEL", home_channel.strip())
            print_success(f"  Home channel set to {home_channel.strip()}")

    print()
    print_success("🐧 QQ Bot configured!")
    print_info(f"  App ID: {credentials['app_id']}")


def _signal_line_input(prompt_text: str) -> str | None:
    """``line_input`` for the Signal wizard; None (after printing the cancel line) on EOF/Ctrl+C."""
    try:
        return line_input(prompt_text).strip()
    except (EOFError, KeyboardInterrupt):
        print("\n  Setup cancelled.")
        return None


def _setup_signal():
    """Interactive setup for Signal messenger."""
    print()
    print(color("  ─── 📡 Signal Setup ───", Colors.CYAN))

    existing_url = get_env_value("SIGNAL_HTTP_URL")
    existing_account = get_env_value("SIGNAL_ACCOUNT")
    if not _confirm_reconfigure("Signal", "SIGNAL_HTTP_URL", "SIGNAL_ACCOUNT"):
        return

    print()
    if shutil.which("signal-cli"):
        print_success("signal-cli found on PATH.")
    else:
        print_warning("signal-cli not found on PATH.")
        print_info("  Signal requires signal-cli running as an HTTP daemon.")
        print_info("  Install options:")
        print_info("    Linux:  download from https://github.com/AsamK/signal-cli/releases")
        print_info("    macOS:  brew install signal-cli")
        print_info("    Docker: bbernhard/signal-cli-rest-api")
        print()
        print_info("  After installing, link your account and start the daemon:")
        print_info('    signal-cli link -n "HermesAgent"')
        print_info("    signal-cli --account +YOURNUMBER daemon --http 127.0.0.1:8080")
        print()

    print()
    print_info("  Enter the URL where signal-cli HTTP daemon is running.")
    default_url = existing_url or "http://127.0.0.1:8080"
    url = _signal_line_input(f"  HTTP URL [{default_url}]: ")
    if url is None:
        return
    url = url or default_url

    print_info("  Testing connection...")
    try:
        import httpx

        resp = httpx.get(f"{url.rstrip('/')}/api/v1/check", timeout=10.0)
        if resp.status_code == 200:
            print_success("  signal-cli daemon is reachable!")
        else:
            print_warning(f"  signal-cli responded with status {resp.status_code}.")
            if not prompt_yes_no("  Continue anyway?", False):
                return
    except Exception as e:
        print_warning(f"  Could not reach signal-cli at {url}: {e}")
        if not prompt_yes_no("  Save this URL anyway? (you can start signal-cli later)", True):
            return

    save_env_value("SIGNAL_HTTP_URL", url)

    print()
    print_info("  Enter your Signal account phone number in E.164 format.")
    print_info("  Example: +15551234567")
    default_account = existing_account or ""
    account = _signal_line_input(f"  Account number{f' [{default_account}]' if default_account else ''}: ")
    if account is None:
        return
    account = account or default_account
    if not account:
        print_error("  Account number is required.")
        return

    save_env_value("SIGNAL_ACCOUNT", account)

    print()
    print_info("  The gateway DENIES all users by default for security.")
    print_info("  Enter phone numbers or UUIDs of allowed users (comma-separated).")
    existing_allowed = get_env_value("SIGNAL_ALLOWED_USERS") or ""
    default_allowed = existing_allowed or account
    allowed = _signal_line_input(f"  Allowed users [{default_allowed}]: ")
    if allowed is None:
        return
    save_env_value("SIGNAL_ALLOWED_USERS", allowed or default_allowed)

    print()
    if prompt_yes_no("  Enable group messaging? (disabled by default for security)", False):
        print()
        print_info("  Enter group IDs to allow, or * for all groups.")
        existing_groups = get_env_value("SIGNAL_GROUP_ALLOWED_USERS") or ""
        groups = _signal_line_input(f"  Group IDs [{existing_groups or '*'}]: ")
        if groups is None:
            return
        save_env_value("SIGNAL_GROUP_ALLOWED_USERS", groups or existing_groups or "*")

    print()
    print_success("Signal configured!")
    print_info(f"  URL: {url}")
    print_info(f"  Account: {account}")
    print_info("  DM auth: via SIGNAL_ALLOWED_USERS + DM pairing")
    print_info(f"  Groups: {'enabled' if get_env_value('SIGNAL_GROUP_ALLOWED_USERS') else 'disabled'}")


def _builtin_setup_fn(key: str):
    """Resolve a built-in platform's setup function; late-bound to dodge the hermes_cli.setup cycle."""
    from hermes_cli import setup as _s

    return {
        # telegram/discord/slack/whatsapp/dingtalk/feishu/wecom setup_fns come from their plugins.
        "bluebubbles": _s._setup_bluebubbles,
        "webhooks": _s._setup_webhooks,
        "signal": _setup_signal,
        "weixin": _setup_weixin,
        "qqbot": _setup_qqbot,
    }.get(key)


def _configure_platform(platform: dict) -> None:
    """Setup flow for one platform. Dispatch: plugin ``setup_fn`` -> built-in by key ->
    ``_setup_standard_platform`` when ``vars`` exists -> env-var hint fallback. Bundled plugins
    auto-load; user plugins must already be in ``plugins.enabled``."""
    entry = platform.get("_registry_entry")

    if entry is not None and entry.setup_fn is not None:
        entry.setup_fn()
        return

    fn = _builtin_setup_fn(platform["key"])
    if fn is not None:
        fn()
        return

    if platform.get("vars"):
        _setup_standard_platform(platform)
        return

    label = platform.get("label", platform["key"])
    emoji = platform.get("emoji", "🔌")
    print()
    print(color(f"  ─── {emoji} {label} Setup ───", Colors.CYAN))
    required = entry.required_env if entry else []
    if required:
        print_info(f"  Set these env vars in ~/.hermes/.env: {', '.join(required)}")
    else:
        print_info(f"  Configure {label} in config.yaml under gateway.platforms.{platform['key']}")
    if platform.get("install_hint"):
        print_info(f"  {platform['install_hint']}")


def _print_indented(text: str, emit=print) -> None:
    for line in text.splitlines():
        emit(f"  {line}")


def _wizard_offer_service_action(action: str, question: str, failed_label: str) -> None:
    """Wizard start/restart prompt; prints remediation instead when system scope would need root."""
    if supports_systemd_services() and _system_scope_wizard_would_need_root():
        _print_system_scope_remediation(action)
    elif prompt_yes_no(question, True):
        _setup_service_action(action, failed_label=failed_label)


def _setup_service_action(
    action: str, *, failed_label: str, windows: bool = True, system: bool = False
) -> None:
    """Run a wizard service start/restart, printing remediation instead of raising.

    ``windows=False`` skips Windows (pre-platform status block never offers it); ``system`` picks
    the systemd scope for a fresh install's first start.
    """
    try:
        if supports_systemd_services():
            if action == "restart":
                systemd_restart()
            else:
                systemd_start(system=system)
        elif is_macos():
            (launchd_restart if action == "restart" else launchd_start)()
        elif windows and is_windows():
            (_gw_windows().restart if action == "restart" else _gw_windows().start)()
        elif action == "restart" and windows:
            stop_profile_gateway()
            print_info("Start manually: hermes gateway")
    except UserSystemdUnavailableError as e:
        print_error(f"  {failed_label} — user systemd not reachable:")
        _print_indented(str(e))
    except SystemScopeRequiresRootError as e:
        # Defense in depth: the wizard's root pre-check should have caught this.
        print_error(f"  {failed_label}: {e}")
        _print_system_scope_remediation(action)
    except subprocess.CalledProcessError as e:
        print_error(f"  {failed_label}: {e}")


def gateway_setup():
    """Interactive setup for messaging platforms + gateway service."""
    if is_managed():
        managed_error("run gateway setup")
        return

    print()
    for banner_line in (
        "┌─────────────────────────────────────────────────────────┐",
        "│             ⚕ Gateway Setup                            │",
        "├─────────────────────────────────────────────────────────┤",
        "│  Configure messaging platforms and the gateway service. │",
        "│  Press Ctrl+C at any time to exit.                     │",
        "└─────────────────────────────────────────────────────────┘",
    ):
        print(color(banner_line, Colors.MAGENTA))

    # ── Gateway service status ──
    print()
    service_installed = _is_service_installed()
    service_running = _is_service_running()

    if supports_systemd_services() and has_conflicting_systemd_units():
        print_systemd_scope_conflict_warning()
        print()

    if supports_systemd_services() and has_legacy_hermes_units():
        print_legacy_unit_warning()
        print()

    if service_installed and service_running:
        print_success("Gateway service is installed and running.")
    elif service_installed:
        print_warning("Gateway service is installed but not running.")
        if supports_systemd_services() and _system_scope_wizard_would_need_root():
            _print_system_scope_remediation("start")
        elif prompt_yes_no("  Start it now?", True):
            _setup_service_action("start", failed_label="Failed to start", windows=False)
    else:
        print_info("Gateway service is not installed yet.")
        print_info("You'll be offered to install it after configuring platforms.")

    # ── Platform configuration loop ──
    while True:
        print()
        print_header("Messaging Platforms")

        platforms = _all_platforms()

        menu_items = [f"{p['emoji']} {p['label']}  ({_platform_status(p)})" for p in platforms]
        menu_items.append("Done")

        choice = prompt_choice("Select a platform to configure:", menu_items, len(menu_items) - 1)
        if choice == len(platforms):
            break

        _configure_platform(platforms[choice])

    # ── Post-setup: offer to install/restart gateway ──
    # Any platform (built-in or plugin) with meaningful progress; ``_platform_status`` already
    # handles plugin check_fn and dual states like WhatsApp's "enabled, not paired".
    def _is_progress(status: str) -> bool:
        s = status.lower()
        return not (s == "not configured" or s.startswith("partially") or s.startswith("plugin disabled"))

    any_configured = any(_is_progress(_platform_status(p)) for p in _all_platforms())

    if any_configured:
        print()
        print(color("─" * 58, Colors.DIM))
        service_installed = _is_service_installed()
        service_running = _is_service_running()

        if service_running:
            _wizard_offer_service_action("restart", "  Restart the gateway to pick up changes?", "Restart failed")
        elif service_installed:
            _wizard_offer_service_action("start", "  Start the gateway service?", "Start failed")
        else:
            print()
            if supports_systemd_services() or is_macos() or is_windows():
                if supports_systemd_services():
                    platform_name = "systemd"
                elif is_macos():
                    platform_name = "launchd"
                else:
                    platform_name = "Scheduled Task"
                wsl_note = " (note: services may not survive WSL restarts)" if is_wsl() else ""
                start_now = prompt_yes_no("  Start the gateway now?", True)
                start_on_login = prompt_yes_no(
                    f"  Start the gateway automatically on login/boot as a {platform_name} service?{wsl_note}",
                    True,
                )
                if start_now or start_on_login:
                    try:
                        installed_scope, did_install = None, True
                        if supports_systemd_services():
                            installed_scope, did_install = install_linux_gateway_from_setup(
                                force=False, enable_on_startup=start_on_login
                            )
                        elif is_macos():
                            launchd_install(force=False)
                        else:
                            _gw_windows().install(force=False)
                        print()
                        if did_install and start_now:
                            _setup_service_action(
                                "start", failed_label="Start failed", system=installed_scope == "system"
                            )
                    except subprocess.CalledProcessError as e:
                        print_error(f"  Install failed: {e}")
                        print_info("  You can try manually: hermes gateway install")
                else:
                    print_info("  Skipped start and auto-start setup.")
                    print_info("  You can install later: hermes gateway install")
                    if supports_systemd_services():
                        print_info("  Or as a boot-time service: sudo hermes gateway install --system")
                    print_info("  Or run in foreground:  hermes gateway run")
            elif is_wsl():
                print_info("  WSL detected but systemd is not running.")
                print_info("  Run in foreground: hermes gateway run")
                print_info("  For persistence:   tmux new -s hermes 'hermes gateway run'")
                print_info("  To enable systemd: add systemd=true to /etc/wsl.conf, then 'wsl --shutdown'")
            elif is_termux():
                from hermes_constants import display_hermes_home as _dhh

                print_info("  Termux does not use systemd/launchd services.")
                print_info("  Run in foreground: hermes gateway run")
                print_info(
                    f"  Or start it manually in the background (best effort): nohup hermes gateway run >{_dhh()}/logs/gateway.log 2>&1 &"
                )
            else:
                print_info("  Service install not supported on this platform.")
                print_info("  Run in foreground: hermes gateway run")
    else:
        print()
        print_info("No platforms configured. Run 'hermes gateway setup' when ready.")

    print()


# =============================================================================
# Main Command Handler
# =============================================================================

def _dispatch_via_service_manager_if_s6(action: str, profile: str | None = None) -> bool:
    """Dispatch start/stop/restart via s6 inside an s6 container; True iff dispatched (caller returns).

    Profile defaults to the current one. Missing slot / s6 errors become actionable CLI messages.
    """
    from hermes_cli.service_manager import (
        GatewayNotRegisteredError,
        S6CommandError,
        detect_service_manager,
        get_service_manager,
    )

    if detect_service_manager() != "s6":
        return False
    if profile is None:
        # _profile_suffix() is "" for the default root; map it to "default" so the default
        # gateway is reachable as gateway-default.
        profile = _profile_suffix() or "default"
    mgr = get_service_manager()
    service_name = f"gateway-{profile}"
    try:
        if action == "start":
            mgr.start(service_name)
        elif action == "stop":
            mgr.stop(service_name)
        elif action == "restart":
            mgr.restart(service_name)
        else:
            return False
    except GatewayNotRegisteredError as exc:
        print(f"✗ {exc}")
        sys.exit(1)
    except S6CommandError as exc:
        print(f"✗ {exc}")
        sys.exit(1)
    return True


def _dispatch_all_via_service_manager_if_s6(action: str) -> bool:
    """Dispatch ``--all`` stop/restart to every registered profile gateway under s6.

    Returns True iff dispatched (caller should ``return``). A bare pkill is seen by s6-supervise
    as a crash and restarted ~1s later (kicking, not stopping); the service manager flips
    ``want up``/``want down`` correctly. ``start --all`` is not a CLI surface.
    """
    from hermes_cli.service_manager import (detect_service_manager, get_service_manager)

    if detect_service_manager() != "s6":
        return False
    if action not in ("stop", "restart"):
        return False
    mgr = get_service_manager()
    profiles = mgr.list_profile_gateways()
    if not profiles:
        print("✗ No profile gateways registered under s6")
        return True
    fn = mgr.stop if action == "stop" else mgr.restart
    errors: list[tuple[str, Exception]] = []
    for profile in profiles:
        service_name = f"gateway-{profile}"
        try:
            fn(service_name)
        except Exception as exc:  # noqa: BLE001 — report and continue
            errors.append((profile, exc))
    succeeded = len(profiles) - len(errors)
    verb = "stopped" if action == "stop" else "restarted"
    if succeeded:
        print(f"✓ {verb.capitalize()} {succeeded} profile gateway(s) under s6")
    for profile, exc in errors:
        print(f"✗ Could not {action} gateway-{profile}: {exc}")
    return True



def gateway_command(args):
    """Handle gateway subcommands."""
    try:
        return _gateway_command_inner(args)
    except UserSystemdUnavailableError as e:
        # Actionable message, not a traceback, when the user D-Bus session is unreachable.
        print_error("User systemd not reachable:")
        _print_indented(str(e))
        sys.exit(1)
    except SystemScopeRequiresRootError as e:
        # System-scope action typed without sudo; the wizard intercepts this earlier with guidance.
        print(str(e))
        sys.exit(1)


def _maybe_redirect_run_to_s6_supervision(args) -> bool:
    """Inside an s6 container, upgrade bare ``gateway run`` to the supervised s6 longrun.

    Gates: ``_dispatch_via_service_manager_if_s6`` requires s6 as PID 1; ``HERMES_S6_SUPERVISED_CHILD``
    (set by ``S6ServiceManager._render_run_script``) marks the supervised child, which must run in
    foreground or we'd recurse run → start → run; ``--no-supervise`` / HERMES_GATEWAY_NO_SUPERVISE=1
    opts out (CI smoke, debugging). Returns True iff dispatched (caller should ``return``).
    """
    no_supervise = getattr(args, "no_supervise", False) or \
        os.environ.get("HERMES_GATEWAY_NO_SUPERVISE", "").lower() in ("1", "true", "yes")
    if no_supervise:
        return False
    if os.environ.get("HERMES_S6_SUPERVISED_CHILD"):
        # We ARE the supervised child; fall through so the gateway actually starts.
        return False
    if not _dispatch_via_service_manager_if_s6("start"):
        return False
    # Breadcrumb on stderr (keep stdout clean for scripts); the supervised gateway's own logs follow
    # via s6-log in `docker logs` and ${HERMES_HOME}/logs/gateways/<profile>/current.
    print(
        "→ gateway is now running under s6 supervision (auto-restart on crash,\n"
        "  dashboard supervised alongside if HERMES_DASHBOARD is set).\n"
        "  This is the recommended setup for the s6 container image — the\n"
        "  gateway will keep running even if it crashes.\n"
        "  Use `--no-supervise` (or HERMES_GATEWAY_NO_SUPERVISE=1) to opt out\n"
        "  and get the pre-s6 foreground behavior instead.",
        file=sys.stderr,
        flush=True,
    )
    # Keep the CMD process alive as a heartbeat so the container survives gateway flaps; `docker
    # stop` SIGTERMs it and /init runs stage-3 shutdown. Prefer `sleep infinity` (frees the
    # interpreter), but execvp's PATH lookup crashed containers with clobbered PATH / no `sleep`.
    try:
        os.execvp("sleep", ["sleep", "infinity"])
    except OSError:
        # execvp only returns by raising (ENOENT when `sleep` is missing, or any other exec error).
        print(
            "→ `sleep` is unavailable; keeping the s6 CMD process alive "
            "in-process until the container is stopped.",
            file=sys.stderr,
            flush=True,
        )
        _block_until_terminated()
    return True  # unreachable on the execvp success path


def _block_until_terminated() -> None:
    """Fallback heartbeat when ``execvp("sleep")`` fails. SIGTERM exits 128+signum so ``docker stop``
    is clean; ``Event().wait()`` covers platforms without ``signal.pause()`` (keeps it testable)."""
    signal.signal(signal.SIGTERM, lambda signum, _frame: sys.exit(128 + signum))
    pause = getattr(signal, "pause", None)
    if pause is not None:
        while True:
            pause()
    else:  # pragma: no cover - non-Unix fallback, not exercised in the s6 image
        import threading

        threading.Event().wait()


def _installed_service_kind() -> str | None:
    """``"systemd"`` / ``"launchd"`` / ``"windows"`` when that service is installed, else None."""
    if _systemd_unit_installed():
        return "systemd"
    if is_macos() and get_launchd_plist_path().exists():
        return "launchd"
    if is_windows() and _gw_windows().is_installed():
        return "windows"
    return None


def _stop_installed_service(system: bool) -> bool:
    """Stop the installed systemd/launchd/Windows service. Returns True if one was stopped."""
    kind = _installed_service_kind()
    if kind is None:
        return False
    # SystemScopeRequiresRootError is a RuntimeError and must propagate from systemd_stop.
    swallow = (subprocess.CalledProcessError, RuntimeError) if kind == "windows" else subprocess.CalledProcessError
    try:
        if kind == "systemd":
            systemd_stop(system=system)
        elif kind == "launchd":
            launchd_stop()
        else:
            _gw_windows().stop()
        return True
    except swallow:
        return False


def _refuse_from_inside_gateway(verb: str, reason: str) -> None:
    """Refuse self-targeting stop/restart/uninstall from inside the gateway process (#92560)."""
    from tools.process_registry import _is_supervised_gateway_process

    if _is_supervised_gateway_process():
        print_error(
            f"Refusing to {verb} the gateway from inside the gateway process.\n"
            f"This command was blocked to prevent {reason}.\n"
            f"Use `hermes gateway {verb}` from a shell outside the running gateway."
        )
        sys.exit(1)


def _print_wsl_foreground_hint(*, systemd_hint: bool) -> None:
    print()
    print("  hermes gateway run                              # direct foreground")
    print("  tmux new -s hermes 'hermes gateway run'         # persistent via tmux")
    print("  nohup hermes gateway run > ~/.hermes/logs/gateway.log 2>&1 &  # background")
    if systemd_hint:
        print()
        print(
            "To enable systemd: add systemd=true to /etc/wsl.conf and run 'wsl --shutdown' from PowerShell."
        )
    sys.exit(1)


def _print_runtime_health() -> None:
    runtime_lines = _runtime_health_lines()
    if runtime_lines:
        print()
        print("Recent gateway health:")
        for line in runtime_lines:
            print(f"  {line}")


def _cmd_run(args):
    if _maybe_redirect_run_to_s6_supervision(args):
        return  # unreachable; execvp doesn't return
    if getattr(args, "external_supervisor", False):
        os.environ[EXTERNAL_GATEWAY_SUPERVISOR_ENV] = "1"
    run_gateway(
        getattr(args, "verbose", 0),
        quiet=getattr(args, "quiet", False),
        replace=getattr(args, "replace", False),
        force=getattr(args, "force", False),
    )


def _cmd_setup(args):
    gateway_setup()


def _cmd_install(args):
    if is_managed():
        managed_error("install gateway service")
        return
    force = getattr(args, "force", False)
    system = getattr(args, "system", False)
    run_as_user = getattr(args, "run_as_user", None)
    if is_termux():
        print("Gateway service installation is not supported on Termux.")
        print("Run manually: hermes gateway")
        sys.exit(1)
    if supports_systemd_services():
        if is_wsl():
            print_warning("WSL detected — systemd services may not survive WSL restarts.")
            print_info("  Consider running in foreground instead: hermes gateway run")
            print_info("  Or use tmux/screen for persistence: tmux new -s hermes 'hermes gateway run'")
            print()
        # Honor --start-now/--start-on-login; else prompt on a TTY, default True headless.
        non_interactive = not (hasattr(sys.stdin, "isatty") and sys.stdin.isatty())
        _sn = getattr(args, "start_now", None)
        if _sn is not None:
            start_now = _sn
        elif not non_interactive:
            start_now = prompt_yes_no("Start the gateway now after installing the service?", True)
        else:
            start_now = True

        _sol = getattr(args, "start_on_login", None)
        if _sol is not None:
            start_on_login = _sol
        elif not non_interactive:
            start_on_login = prompt_yes_no("Start the gateway automatically on login/boot with systemd?", True)
        else:
            start_on_login = True
        systemd_install(
            force=force,
            system=system,
            run_as_user=run_as_user,
            enable_on_startup=start_on_login,
            non_interactive=non_interactive,
        )
        if start_now:
            systemd_start(system=system)
    elif is_macos():
        launchd_install(force)
    elif is_windows():
        _gw_windows().install(
            force=force,
            start_now=getattr(args, 'start_now', None),
            start_on_login=getattr(args, 'start_on_login', None),
            elevated_handoff=getattr(args, 'elevated_handoff', False),
        )
    elif is_wsl():
        print("WSL detected but systemd is not running.")
        print("Either enable systemd (add systemd=true to /etc/wsl.conf and restart WSL)")
        print("or run the gateway in foreground mode:")
        _print_wsl_foreground_hint(systemd_hint=False)
    elif is_container():
        # With s6 the gateway service is auto-registered when the profile is created.
        if _running_under_s6():
            print("Per-profile gateways are auto-registered when you create a profile.")
            print()
            print("  hermes profile create <name>     # creates the s6 service slot")
            print("  hermes -p <name> gateway start   # bring it up via s6")
            print("  hermes status                    # see currently-supervised gateways")
            return
        print("Service installation is not needed inside a Docker container.")
        print("The container runtime is your service manager — use Docker restart policies instead:")
        print()
        print("  docker run --restart unless-stopped ...   # auto-restart on crash/reboot")
        print("  docker restart <container>                # manual restart")
        print()
        print("To run the gateway: hermes gateway run")
        sys.exit(0)
    else:
        print("Service installation not supported on this platform.")
        print("Run manually: hermes gateway run")
        sys.exit(1)


def _cmd_uninstall(args):
    _refuse_from_inside_gateway("uninstall", "the gateway from terminating itself")
    if is_managed():
        managed_error("uninstall gateway service")
        return
    system = getattr(args, "system", False)
    if is_termux():
        print("Gateway service uninstall is not supported on Termux because there is no managed service to remove.")
        print("Stop manual runs with: hermes gateway stop")
        sys.exit(1)
    if supports_systemd_services():
        systemd_uninstall(system=system)
    elif is_macos():
        launchd_uninstall()
    elif is_windows():
        _gw_windows().uninstall()
    elif is_container():
        if _running_under_s6():
            print("Per-profile gateways are auto-unregistered when you delete the profile.")
            print()
            print("  hermes profile delete <name>     # tears down the s6 service slot")
            print("  hermes -p <name> gateway stop    # stop without deleting the profile")
            return
        print("Service uninstall is not applicable inside a Docker container.")
        print("To stop the gateway, stop or remove the container:")
        print()
        print("  docker stop <container>")
        print("  docker rm <container>")
        sys.exit(0)
    else:
        print("Not supported on this platform.")
        sys.exit(1)


def _cmd_start(args):
    system = getattr(args, "system", False)
    start_all = getattr(args, "all", False)
    if not start_all and _dispatch_via_service_manager_if_s6("start"):
        return
    if start_all:
        killed = kill_gateway_processes(all_profiles=True)
        if killed:
            print(f"✓ Killed {killed} stale gateway process(es) across all profiles")
            _wait_for_gateway_exit(timeout=10.0, force_after=5.0)

    if is_termux():
        print("Gateway service start is not supported on Termux because there is no system service manager.")
        print("Run manually: hermes gateway")
        sys.exit(1)
    if supports_systemd_services():
        systemd_start(system=system)
    elif is_macos():
        launchd_start()
    elif is_windows():
        _gw_windows().start()
    elif is_wsl():
        print("WSL detected but systemd is not available.")
        print("Run the gateway in foreground mode instead:")
        _print_wsl_foreground_hint(systemd_hint=True)
    elif is_container():
        # Reached only when s6 ISN'T running (the early dispatch above handles the s6 case).
        print("Service start is not applicable inside a Docker container.")
        print("The gateway runs as the container's main process.")
        print()
        print("  docker start <container>     # start a stopped container")
        print("  docker restart <container>   # restart a running container")
        print()
        print("Or run the gateway directly: hermes gateway run")
        sys.exit(0)
    else:
        print("Not supported on this platform.")
        sys.exit(1)


def _cmd_stop(args):
    _refuse_from_inside_gateway("stop", "restart loops")
    stop_all = getattr(args, "all", False)
    system = getattr(args, "system", False)
    # Under s6 a bare pkill is seen as a crash and restarted; go through the supervisor.
    if stop_all and _dispatch_all_via_service_manager_if_s6("stop"):
        return
    if not stop_all and _dispatch_via_service_manager_if_s6("stop"):
        return

    service_available = _stop_installed_service(system)
    if stop_all:
        killed = kill_gateway_processes(all_profiles=True)
        total = killed + (1 if service_available else 0)
        if total:
            print(f"✓ Stopped {total} gateway process(es) across all profiles")
        else:
            print("✗ No gateway processes found")
    elif not service_available:
        if stop_profile_gateway():
            print("✓ Stopped gateway for this profile")
        else:
            print("✗ No gateway running for this profile")
    else:
        print(f"✓ Stopped {get_service_name()} service")


def _cmd_restart(args):
    _refuse_from_inside_gateway("restart", "restart loops")
    service_available = False
    system = getattr(args, "system", False)
    restart_all = getattr(args, "all", False)
    service_configured = False
    if restart_all and _dispatch_all_via_service_manager_if_s6("restart"):
        return
    if not restart_all and _dispatch_via_service_manager_if_s6("restart"):
        return

    if restart_all:
        service_stopped = _stop_installed_service(system)
        killed = kill_gateway_processes(all_profiles=True)
        total = killed + (1 if service_stopped else 0)
        if total:
            print(f"✓ Stopped {total} gateway process(es) across all profiles")
        _wait_for_gateway_exit(timeout=10.0, force_after=5.0)

        print("Starting gateway...")
        if _systemd_unit_installed():
            systemd_start(system=system)
        elif is_macos() and get_launchd_plist_path().exists():
            launchd_start()
        elif is_windows():
            # Even without a registered task, gateway_windows.start() uses the detached launcher.
            _gw_windows().start()
        else:
            run_gateway(verbose=0)
        return

    if _systemd_unit_installed():
        service_configured = True
        try:
            systemd_restart(system=system)
            service_available = True
        except subprocess.CalledProcessError:
            pass
    elif is_macos() and get_launchd_plist_path().exists():
        service_configured = True
        try:
            launchd_restart()
            service_available = True
        except subprocess.CalledProcessError:
            pass
    elif is_windows():
        # The Windows restart path handles both registered installs and detached restarts.
        service_configured = _gw_windows().is_installed()
        try:
            _gw_windows().restart()
            return
        except (subprocess.CalledProcessError, RuntimeError, OSError):
            pass

    if service_available:
        return
    if supports_systemd_services():
        linger_ok, _detail = get_systemd_linger_status()
        if linger_ok is not True:
            import getpass

            _username = getpass.getuser()
            print()
            print("⚠ Cannot restart gateway as a service — linger is not enabled.")
            print("  The gateway user service requires linger to function on headless servers.")
            print()
            print(f"  Run:  sudo loginctl enable-linger {_username}")
            print()
            print("  Then restart the gateway:")
            print("    hermes gateway restart")
            return

    if service_configured:
        print()
        print("✗ Gateway service restart failed.")
        print("  The service definition exists, but the service manager did not recover it.")
        print("  Fix the service, then retry: hermes gateway start")
        sys.exit(1)

    if stop_profile_gateway():
        print("✓ Stopped gateway for this profile")
    _wait_for_gateway_exit(timeout=10.0, force_after=5.0)
    print("Starting gateway...")
    run_gateway(verbose=0)


def _cmd_status(args):
    deep = getattr(args, "deep", False)
    full = getattr(args, "full", False)
    system = getattr(args, "system", False)
    snapshot = get_gateway_runtime_snapshot(system=system)

    _windows_service_installed = is_windows() and _gw_windows().is_installed()
    if not snapshot.running and named_profile_served_by_running_multiplexer():
        # Satellite profile: the default multiplexer is the live inbound process for it.
        print("✓ Gateway is running via the default-profile multiplexer")
        print("  Manage it from the default profile: hermes gateway status")
    elif _systemd_unit_installed():
        systemd_status(deep, system=system, full=full)
        _print_gateway_process_mismatch(snapshot)
    elif is_macos() and get_launchd_plist_path().exists():
        launchd_status(deep)
        _print_gateway_process_mismatch(snapshot)
    elif _windows_service_installed:
        _gw_windows().status(deep=deep)
        _print_gateway_process_mismatch(snapshot)
    else:
        pids = list(snapshot.gateway_pids)
        if pids:
            print(f"✓ Gateway is running (PID: {', '.join(map(str, pids))})")
            print("  (Running manually, not as a system service)")
            _print_runtime_health()
            print()
            if is_termux():
                print("Termux note:")
                print("  Android may stop background jobs when Termux is suspended")
            elif is_wsl():
                print("WSL note:")
                print("  The gateway is running in foreground/manual mode (recommended for WSL).")
                print("  Use tmux or screen for persistence across terminal closes.")
            elif is_windows():
                print("To install as a Windows Scheduled Task (auto-start on login):")
                print("  hermes gateway install")
            else:
                print("To install as a service:")
                print("  hermes gateway install")
                print("  sudo hermes gateway install --system")
        else:
            print("✗ Gateway is not running")
            _print_runtime_health()
            print()
            print("To start:")
            print("  hermes gateway run      # Run in foreground")
            if is_termux():
                print("  nohup hermes gateway run > ~/.hermes/logs/gateway.log 2>&1 &  # Best-effort background start")
            elif is_wsl():
                print("  tmux new -s hermes 'hermes gateway run'         # persistent via tmux")
                print("  nohup hermes gateway run > ~/.hermes/logs/gateway.log 2>&1 &  # background")
            elif is_windows():
                print("  hermes gateway install  # Install as Windows Scheduled Task (auto-start on login)")
            else:
                print("  hermes gateway install  # Install as user service")
                print("  sudo hermes gateway install --system  # Install as boot-time system service")

    _print_other_profiles_gateway_status()


def _cmd_list(args):
    _gateway_list()


def _cmd_migrate_legacy(args):
    """Stop, disable, and remove legacy Hermes gateway unit files (e.g. hermes.service)."""
    dry_run = getattr(args, "dry_run", False)
    yes = getattr(args, "yes", False)
    if not supports_systemd_services() and not is_macos():
        print("Legacy unit migration only applies to systemd-based Linux hosts.")
        return
    remove_legacy_hermes_units(interactive=not yes, dry_run=dry_run)


_GATEWAY_SUBCOMMANDS = {
    None: _cmd_run,
    "run": _cmd_run,
    "setup": _cmd_setup,
    "install": _cmd_install,
    "uninstall": _cmd_uninstall,
    "start": _cmd_start,
    "stop": _cmd_stop,
    "restart": _cmd_restart,
    "status": _cmd_status,
    "list": _cmd_list,
    "migrate-legacy": _cmd_migrate_legacy,
}


def _gateway_command_inner(args):
    handler = _GATEWAY_SUBCOMMANDS.get(getattr(args, "gateway_command", None))
    if handler is not None:
        handler(args)
