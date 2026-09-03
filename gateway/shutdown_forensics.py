"""Shutdown forensics — capture context when the gateway receives SIGTERM/SIGINT.

``shutdown_signal_handler`` runs synchronously inside the asyncio loop, so
:func:`snapshot_shutdown_context` is a fast (<10ms) non-blocking probe it can log
immediately, and :func:`spawn_async_diagnostic` is a fire-and-forget ``ps`` walk in
a detached subprocess so it can't block teardown even if /proc is wedged. Anything
that waits belongs in the async helper, never in the synchronous probe.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from gateway.restart import DEFAULT_GATEWAY_CRON_DRAIN_TIMEOUT, resolve_systemd_timeout_stop_sec
import contextlib

_SIGNAL_NAME_BY_NUM: Dict[int, str] = {
    int(getattr(signal, _name)): _name
    for _name in ("SIGTERM", "SIGINT", "SIGHUP", "SIGQUIT", "SIGUSR1", "SIGUSR2")
    if getattr(signal, _name, None) is not None
}


def _signal_name(sig: Any) -> str:
    """Return a human-readable signal name (or ``str(sig)`` as fallback)."""
    if sig is None:
        return "UNKNOWN"
    try:
        sig_int = int(sig)
    except (TypeError, ValueError):
        return str(sig)
    return _SIGNAL_NAME_BY_NUM.get(sig_int, f"signal#{sig_int}")


def _read_proc_field(pid: int, key: str) -> Optional[str]:
    """Read a single field from /proc/<pid>/status.  Linux only; None elsewhere."""
    try:
        with open(f"/proc/{pid}/status", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith(key + ":"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return None


def _proc_summary(pid: int) -> Dict[str, Any]:
    """Compact /proc/<pid> snapshot (pid, ppid, state, uid, cmdline); missing fields omitted."""
    summary: Dict[str, Any] = {"pid": pid}
    if pid <= 0:
        return summary
    for out_key, proc_key in (("name", "Name"), ("state", "State")):
        value = _read_proc_field(pid, proc_key)
        if value is not None:
            summary[out_key] = value
    ppid = _read_proc_field(pid, "PPid")
    if ppid is not None:
        with contextlib.suppress(ValueError):
            summary["ppid"] = int(ppid)
    uid = _read_proc_field(pid, "Uid")
    if uid is not None:
        # "real effective saved fs"
        summary["uid"] = uid.split()[0] if uid else uid
    try:
        data = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        data = b""
    if data:
        # Truncate aggressively — these can be 4KB
        summary["cmdline"] = data.replace(b"\x00", b" ").decode("utf-8", errors="replace").strip()[:300]
    return summary


def _read_marker(path: Path) -> Optional[str]:
    """Return the marker file's text, or None if absent/unreadable."""
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def snapshot_shutdown_context(received_signal: Any = None) -> Dict[str, Any]:
    """Fast (<10ms) snapshot of who/what is asking us to shut down.

    Signal name/number, own + parent /proc summaries, systemd parentage,
    takeover/planned-stop markers, TracerPid, 1-min load, wall/monotonic
    timestamps. Pure stdlib, never raises, never blocks on subprocesses.
    """
    pid = os.getpid()
    ppid = os.getppid()

    ctx: Dict[str, Any] = {
        "ts": time.time(),
        "ts_monotonic": time.monotonic(),
        "signal": _signal_name(received_signal),
        "signal_num": int(received_signal) if received_signal is not None else None,
        "pid": pid,
        "ppid": ppid,
        "parent": _proc_summary(ppid),
        "self": _proc_summary(pid),
    }

    # INVOCATION_ID is set by systemd units; ppid==1 also strongly suggests
    # systemd reaped+forwarded the SIGTERM.
    for ctx_key, env_key in (("systemd_invocation_id", "INVOCATION_ID"), ("systemd_journal_stream", "JOURNAL_STREAM")):
        if os.environ.get(env_key):
            ctx[ctx_key] = os.environ[env_key]
    ctx["under_systemd"] = bool(os.environ.get("INVOCATION_ID")) or ppid == 1

    # High load points at "something crushing the box" rather than an external killer.
    with contextlib.suppress(OSError, AttributeError):
        ctx["loadavg_1m"] = os.getloadavg()[0]

    # Nonzero TracerPid means a debugger/strace is attached.
    try:
        tracer = _read_proc_field(pid, "TracerPid")
        if tracer is not None and tracer != "0":
            ctx["tracer_pid"] = int(tracer) if tracer.isdigit() else tracer
            ctx["tracer"] = _proc_summary(int(tracer)) if tracer.isdigit() else None
    except (TypeError, ValueError):
        pass

    # Race hint: a takeover marker on disk that does NOT name us is a smoking
    # gun for "another --replace instance is killing us". Filenames mirror
    # gateway.status; literals keep the signal-handler path import-light.
    try:
        hermes_home_str = os.environ.get("HERMES_HOME")
        if hermes_home_str:
            raw = _read_marker(Path(hermes_home_str) / ".gateway-takeover.json")
            if raw is not None:
                ctx["takeover_marker"] = raw[:300]
                ctx["takeover_marker_for_self"] = f'"target_pid": {pid}' in raw or f"'target_pid': {pid}" in raw
            raw = _read_marker(Path(hermes_home_str) / ".gateway-planned-stop.json")
            if raw is not None:
                ctx["planned_stop_marker"] = raw[:300]
    except Exception:  # noqa: BLE001 — never raise from a signal handler
        pass

    return ctx


def spawn_async_diagnostic(
    log_path: Path,
    signal_name: str,
    *,
    timeout_seconds: float = 5.0,
) -> Optional[int]:
    """Fire-and-forget ``ps``-style snapshot appended to ``log_path``.

    A detached subprocess (own ``timeout`` so a wedged ``ps`` self-cleans) rather
    than a blocking ``ps aux`` in the signal handler, which can freeze the loop >2s
    on a busy host. Returns the subprocess PID, or ``None`` on failure / Windows
    (bash -c is available on every POSIX target; Windows has no ps anyway).
    """
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    if sys.platform == "win32":
        return None

    script = (
        f"echo '=== shutdown diagnostic @ {signal_name} ==='; "
        "echo '--- date ---'; date -u +%Y-%m-%dT%H:%M:%SZ; "
        "echo '--- ps auxf (top 60 by cpu) ---'; "
        "ps auxf --sort=-pcpu 2>/dev/null | head -60; "
        "echo '--- pstree of self ---'; "
        f"pstree -plau {os.getpid()} 2>/dev/null | head -40 || true; "
        "echo '--- /proc/loadavg ---'; "
        "cat /proc/loadavg 2>/dev/null || true; "
        "echo '--- recent dmesg (oom/killed) ---'; "
        "dmesg -T 2>/dev/null | tail -20 || journalctl --user -n 20 --no-pager 2>/dev/null | tail -20 || true; "
        "echo '=== end ==='"
    )

    try:
        # O_APPEND so concurrent diagnostics from rapid signals don't trample each other.
        fd = os.open(str(log_path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    except OSError:
        return None

    try:
        # start_new_session: survive systemd killing our cgroup (KillMode=control-group)
        # long enough to flush.
        proc = subprocess.Popen(
            ["timeout", f"{timeout_seconds:.0f}", "bash", "-c", script],
            stdout=fd, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            start_new_session=True, close_fds=True,
        )
    except OSError:
        return None
    finally:
        # Subprocess inherited the fd; we can drop our handle.
        with contextlib.suppress(OSError):
            os.close(fd)

    return proc.pid


def format_context_for_log(ctx: Dict[str, Any]) -> str:
    """Render a shutdown context dict as a single, scannable log line."""
    parent = ctx.get("parent") or {}
    load = ctx.get("loadavg_1m")
    load_str = f"{load:.2f}" if isinstance(load, (int, float)) else "?"
    extras: List[str] = []
    if ctx.get("takeover_marker") is not None:
        extras.append(f"takeover_marker_present={'self' if ctx.get('takeover_marker_for_self') else 'other'}")
    if ctx.get("planned_stop_marker") is not None:
        extras.append("planned_stop_marker_present=yes")
    if ctx.get("tracer_pid"):
        extras.append(f"tracer_pid={ctx['tracer_pid']}")
    extras_str = (" " + " ".join(extras)) if extras else ""
    # Parent cmdline is the most useful single signal — log it prominently.
    return (
        f"signal={ctx.get('signal', '?')} under_systemd={'yes' if ctx.get('under_systemd') else 'no'} "
        f"parent_pid={parent.get('pid') or '?'} parent_name={parent.get('name') or '?'} "
        f"loadavg_1m={load_str}{extras_str} parent_cmdline={parent.get('cmdline', '(unknown)')!r}"
    )


def context_as_json(ctx: Dict[str, Any]) -> str:
    """JSON-serialise a context dict for structured ingestion.  Never raises."""
    try:
        return json.dumps(ctx, default=str, sort_keys=True)
    except (TypeError, ValueError):
        return "{}"


def check_systemd_timing_alignment(
    drain_timeout: float,
    cron_drain_timeout: float = DEFAULT_GATEWAY_CRON_DRAIN_TIMEOUT,
) -> Optional[Dict[str, Any]]:
    """At startup, sanity-check that systemd's TimeoutStopSec covers stop.

    A stale unit file (upgraded without re-running ``hermes setup``) can have
    ``TimeoutStopSec`` below the stop budget, so systemd SIGKILLs the cgroup
    mid-drain — a phantom ``code=killed status=9`` in the journal. Returns ``None``
    when aligned OR undeterminable (not under systemd, no ``systemctl``); otherwise
    a dict with ``timeout_stop_sec``/``drain_timeout``/``expected_min``/``mismatch``.
    """
    if not os.environ.get("INVOCATION_ID"):
        return None  # Not running under systemd (or at least not directly)

    # /proc/self/cgroup: "0::/user.slice/.../hermes-gateway.service"
    unit_name: Optional[str] = None
    try:
        with open("/proc/self/cgroup", encoding="utf-8") as fh:
            for line in fh:
                unit_name = next((p for p in reversed(line.strip().split("/")) if p.endswith(".service")), None)
                if unit_name:
                    break
    except OSError:
        pass
    if not unit_name:
        return None

    timeout_us = _systemd_timeout_stop_us(unit_name)
    if timeout_us is None:
        return None

    timeout_stop_sec = timeout_us / 1_000_000.0
    expected = float(resolve_systemd_timeout_stop_sec(drain_timeout, cron_drain_timeout))
    return {
        "unit": unit_name,
        "timeout_stop_sec": timeout_stop_sec,
        "drain_timeout": drain_timeout,
        "cron_drain_timeout": cron_drain_timeout,
        "expected_min": expected,
        "mismatch": timeout_stop_sec < expected,
    }


def _systemd_timeout_stop_us(unit_name: str) -> Optional[int]:
    """``TimeoutStopUSec`` of ``unit_name`` in microseconds; ``--user`` first (hermes' common case), then system."""
    for flag in (["--user"], []):
        try:
            result = subprocess.run(
                ["systemctl", *flag, "show", unit_name, "--property=TimeoutStopUSec"],
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=2.0,
            )
        except (subprocess.TimeoutExpired, OSError):
            continue
        if result.returncode != 0:
            continue
        # Output: "TimeoutStopUSec=1min 30s" or "TimeoutStopUSec=90000000"
        for line in result.stdout.splitlines():
            if line.startswith("TimeoutStopUSec="):
                value = line.split("=", 1)[1].strip()
                timeout_us = int(value) if value.isdigit() else parse_systemd_duration_to_us(value)
                if timeout_us is not None:
                    return timeout_us
    return None


def parse_systemd_duration_to_us(raw: str) -> Optional[int]:
    """Parse 'TimeoutStopUSec=1min 30s' / '90s' style values to microseconds.

    Covers the common units (us, ms, s, min, h); a bare number is seconds.
    Returns None on anything unexpected.  Never raises.

    Public: also consumed by hermes_cli.gateway's restart-wait sizing.
    """
    if not raw:
        return None
    units = {
        "us": 1, "ms": 1_000, "s": 1_000_000, "sec": 1_000_000,
        "min": 60_000_000, "h": 3_600_000_000, "hr": 3_600_000_000,
    }
    total_us = 0
    token = ""
    digits = ""

    def _flush() -> bool:
        """Fold the pending ``digits``/``token`` pair into ``total_us``."""
        nonlocal total_us, token, digits
        multiplier = units.get(token.lower()) if token else 1_000_000
        if multiplier is None or not digits:
            return False
        try:
            total_us += int(float(digits) * multiplier)
        except ValueError:
            return False
        digits = ""
        token = ""
        return True

    for ch in raw + " ":
        if ch.isdigit() or ch == ".":
            # A digit after a unit ends the previous number
            if token and not _flush():
                return None
            digits += ch
        elif ch.isalpha():
            token += ch
        elif digits and not _flush():
            return None
    return total_us if total_us > 0 else None


# Backward-compat private alias (pre-promotion name).
_parse_systemd_duration_to_us = parse_systemd_duration_to_us
