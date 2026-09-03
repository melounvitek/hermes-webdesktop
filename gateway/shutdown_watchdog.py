"""Out-of-loop shutdown and event-loop liveness backstops.

A frozen asyncio loop takes every asyncio-based recovery path down with it, and launchd/systemd
KeepAlive only restarts a *dead* process. Hence: (1) an OS-thread shutdown watchdog that dumps
stacks and ``os._exit``s past ``restart_drain_timeout + grace``; (2) a heartbeat file at
``<HERMES_HOME>/state/gateway.heartbeat`` so supervisors can tell "process alive" from "loop
frozen"; (3) a lifetime thread watchdog that hard-exits when the loop is too frozen to run its
own callbacks; (4) a self-rescheduling floor timer that keeps the selector timeout finite.
"""

from __future__ import annotations

import asyncio
import contextlib
import faulthandler
import json
import logging
import os
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from gateway.restart import GATEWAY_SERVICE_RESTART_EXIT_CODE
from hermes_constants import get_hermes_home
from utils import atomic_json_write

logger = logging.getLogger(__name__)

# Extra leash beyond ``agent.restart_drain_timeout`` so a slow-but-progressing drain survives.
DEFAULT_SHUTDOWN_WATCHDOG_GRACE_S = 60.0
DEFAULT_HEARTBEAT_INTERVAL_S = 30.0
DEFAULT_LOOP_FLOOR_TIMER_INTERVAL_S = 5.0
DEFAULT_LOOP_WATCHDOG_INTERVAL_S = 30.0
DEFAULT_LOOP_WATCHDOG_TIMEOUT_S = 10.0
# 3 sustained misses (~90-120s of loop block) escalate; stays tight because the heartbeat
# write is off-loop. Slow loops tune gateway.loop_watchdog_* in config.yaml.
DEFAULT_LOOP_WATCHDOG_MAX_STRIKES = 3
_HEARTBEAT_RELATIVE = ("state", "gateway.heartbeat")
_WATCHDOG_DUMP_RELATIVE = ("logs", "gateway-shutdown-watchdog.log")


def _coerce_float(value: Any, default: float, floor: float = 0.0) -> float:
    """``max(float(value), floor)``, or ``default`` when not coercible."""
    try:
        return max(float(value), floor)
    except (TypeError, ValueError):
        return default


class _LoopFloorTimerHandle:
    """Cancelable owner for the currently scheduled selector floor timer."""
    def __init__(self, loop: asyncio.AbstractEventLoop, interval: float):
        self._loop, self._interval, self._cancelled = loop, interval, False
        self._timer: Optional[asyncio.TimerHandle] = None
        self._tick()

    def _tick(self) -> None:
        if not self._cancelled:
            self._timer = self._loop.call_later(self._interval, self._tick)

    def cancel(self) -> None:
        self._cancelled = True
        if self._timer is not None:
            self._timer.cancel()


class _LoopLivenessWatchdogHandle:
    """Small lifecycle handle for the daemon liveness thread."""
    def __init__(self, stop_event: threading.Event, thread: threading.Thread):
        self._stop_event = stop_event
        self.stop, self.join, self.is_alive = stop_event.set, thread.join, thread.is_alive


def _arm_loop_floor_timer(
    loop: asyncio.AbstractEventLoop, interval: float = DEFAULT_LOOP_FLOOR_TIMER_INTERVAL_S
) -> _LoopFloorTimerHandle:
    """Keep at least one timer pending so selector waits remain bounded."""
    resolved = _coerce_float(interval, 0.0)
    return _LoopFloorTimerHandle(
        loop, resolved if resolved > 0 else DEFAULT_LOOP_FLOOR_TIMER_INTERVAL_S
    )


def start_loop_liveness_watchdog(
    loop: asyncio.AbstractEventLoop, *, probe_interval: float = DEFAULT_LOOP_WATCHDOG_INTERVAL_S,
    probe_timeout: float = DEFAULT_LOOP_WATCHDOG_TIMEOUT_S,
    max_strikes: int = DEFAULT_LOOP_WATCHDOG_MAX_STRIKES,
    exit_code: int = GATEWAY_SERVICE_RESTART_EXIT_CODE,
) -> Optional[_LoopLivenessWatchdogHandle]:
    """Start an out-of-loop watchdog that hard-exits after missed probes. The
    ``gateway.loop_watchdog: false`` opt-out is enforced by the caller
    (``GatewayRunner._start_loop_liveness_guards``).
    """
    stop_event = threading.Event()

    def _wait_for_probe(probe_event: threading.Event) -> Optional[bool]:
        """True/False = probe answered / timed out; None = stop requested mid-wait."""
        deadline = time.monotonic() + probe_timeout
        while not stop_event.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return probe_event.is_set()
            if probe_event.wait(timeout=min(remaining, 0.05)):
                return True
        return None

    def _watchdog() -> None:
        strikes = 0
        while not stop_event.wait(timeout=probe_interval):
            probe_event = threading.Event()
            try:
                loop.call_soon_threadsafe(probe_event.set)
            except RuntimeError:  # normally closed loop: nothing left to backstop
                return
            except Exception:
                logger.debug("Failed to schedule gateway loop liveness probe", exc_info=True)
                return
            responded = _wait_for_probe(probe_event)
            if responded is None:
                return
            if responded:
                strikes = 0
                continue
            # Re-check stop_event between each irreversible step: a late stop() during the dump must
            # win.
            if stop_event.is_set():
                return
            strikes += 1
            if strikes < max_strikes:
                continue
            if stop_event.is_set():
                return
            with contextlib.suppress(Exception):
                logger.critical(
                    "Gateway event loop missed %d consecutive liveness probes; dumping all thread "
                    "stacks and exiting with code %d so the service supervisor can restart it.",
                    strikes,
                    exit_code,
                )
            try:
                faulthandler.dump_traceback(all_threads=True)
            except Exception:
                logger.debug("Loop liveness faulthandler dump failed", exc_info=True)
            if stop_event.is_set():
                return
            # Stamp the lifecycle sentinel so the next boot reports "watchdog hard-exit", not
            # SIGKILL/OOM.
            _mark_exited_quietly(exit_code, "loop_liveness_watchdog")
            os._exit(exit_code)
    thread = threading.Thread(target=_watchdog, daemon=True, name="gateway-loop-liveness-watchdog")
    try:
        thread.start()
    except Exception:
        logger.debug("Failed to start gateway loop liveness watchdog", exc_info=True)
        return None
    return _LoopLivenessWatchdogHandle(stop_event, thread)


def _mark_exited_quietly(exit_code: int, reason: str) -> None:
    """Best-effort lifecycle-ledger stamp so the next boot names the watchdog, not SIGKILL/OOM."""
    with contextlib.suppress(Exception):
        from gateway.lifecycle_ledger import mark_exited
        mark_exited(exit_code, reason=reason)


def _process_hermes_home() -> Path:
    """HERMES_HOME for process-level identity files (ignore profile overrides)."""
    val = os.environ.get("HERMES_HOME", "").strip()
    return Path(val) if val else get_hermes_home()


def _home(home: Optional[Path]) -> Path:
    return home if home is not None else _process_hermes_home()


def get_loop_heartbeat_path(home: Optional[Path] = None) -> Path:
    """Return ``<HERMES_HOME>/state/gateway.heartbeat``."""
    return _home(home).joinpath(*_HEARTBEAT_RELATIVE)


def get_loop_tick_socket_path(home: Optional[Path] = None, pid: Optional[int] = None) -> Path:
    """``<HERMES_HOME>/state/gateway.loop-tick.<pid>.sock`` — PID-suffixed so a stale node from a
    dead process is never mistaken for this gateway's witness. Served by the loop itself
    (``_tick_socket_handler``), so an answer proves the loop dispatches — what the heartbeat can't.
    """
    return (
        _home(home)
        / "state"
        / f"gateway.loop-tick.{int(pid if pid is not None else os.getpid())}.sock"
    )


def get_shutdown_watchdog_dump_path(home: Optional[Path] = None) -> Path:
    """Return the faulthandler / metadata dump path for a fired watchdog."""
    return _home(home).joinpath(*_WATCHDOG_DUMP_RELATIVE)


def write_loop_heartbeat(
    *, pid: Optional[int] = None, start_time: Optional[float] = None,
    home: Optional[Path] = None, extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Atomically rewrite the loop-liveness heartbeat file; never raises.
    ``start_time`` (process start, epoch seconds) lets supervisors detect PID reuse."""
    path = get_loop_heartbeat_path(home)
    payload: Dict[str, Any] = {
        "pid": int(pid if pid is not None else os.getpid()),
        "updated_at": datetime.now(timezone.utc).isoformat(), "monotonic": time.monotonic(),
    }
    if start_time is not None:
        payload["start_time"] = float(start_time)
    # Cheap memory sample: after an unclean death the last heartbeat is the closest record of memory
    # pressure.
    with contextlib.suppress(Exception):
        from gateway.lifecycle_ledger import sample_memory
        if mem := sample_memory():
            payload["mem"] = mem
    if extra:
        payload.update(extra)
    try:
        atomic_json_write(path, payload, indent=None)
    except Exception:
        logger.debug("Failed to write gateway loop heartbeat", exc_info=True)
    return path


def resolve_shutdown_watchdog_delay(
    drain_timeout: float, *, grace_s: float = DEFAULT_SHUTDOWN_WATCHDOG_GRACE_S) -> float:
    """Return the wall-clock leash for the shutdown watchdog thread."""
    return _coerce_float(drain_timeout, 0.0) + _coerce_float(
        grace_s, DEFAULT_SHUTDOWN_WATCHDOG_GRACE_S
    )


def _write_watchdog_dump(
    dump_path: Path, *, delay_s: float, snapshot: Optional[Dict[str, Any]]
) -> None:
    """Best-effort faulthandler + metadata dump before hard-exit."""
    try:
        dump_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        return
    header = {"event": "shutdown_watchdog_fired", "pid": os.getpid(), "delay_s": delay_s,
              "fired_at": datetime.now(timezone.utc).isoformat(), "snapshot": snapshot or {}}
    with contextlib.suppress(Exception), open(dump_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(header, default=str) + "\n--- faulthandler dump (all threads) ---\n")
        fh.flush()
        try:
            faulthandler.dump_traceback(file=fh, all_threads=True)
        except Exception:
            fh.write("(faulthandler.dump_traceback failed)\n")
        fh.write("--- end dump ---\n")
        fh.flush()
    # Also to stderr so journald/launchd capture it even if the disk is wedged.
    with contextlib.suppress(Exception):
        sys.stderr.write(f"Gateway shutdown watchdog fired after {delay_s:.0f}s "
                         f"(pid={os.getpid()}); dumping all thread stacks.\n")
        sys.stderr.flush()
        faulthandler.dump_traceback(all_threads=True)


def arm_shutdown_watchdog(
    delay_s: float, *, done_event: Optional[threading.Event] = None,
    snapshot_fn: Optional[Callable[[], Dict[str, Any]]] = None, exit_code: int = 1,
    dump_path: Optional[Path] = None, name: str = "gateway-shutdown-watchdog",
) -> threading.Event:
    """Arm a daemon-thread hard-exit backstop for a wedged shutdown path.
    Exits quietly if ``done_event`` is set within ``delay_s``, else dumps diagnostics and
    ``os._exit(exit_code)``. Never raises; returns ``done_event`` for disarming."""
    done = done_event if done_event is not None else threading.Event()
    delay = _coerce_float(delay_s, DEFAULT_SHUTDOWN_WATCHDOG_GRACE_S)
    if delay <= 0:
        return done

    def _watchdog() -> None:
        deadline = time.monotonic() + delay  # chunked wait so a late disarm is observed within ~1s
        while time.monotonic() < deadline:
            if done.wait(timeout=min(deadline - time.monotonic(), 1.0)):
                return
        if done.is_set():
            return
        snapshot: Optional[Dict[str, Any]] = None
        if snapshot_fn is not None:
            try:
                snapshot = snapshot_fn()
            except Exception as exc:
                snapshot = {"snapshot_error": repr(exc)}
        target = dump_path if dump_path is not None else get_shutdown_watchdog_dump_path()
        _write_watchdog_dump(target, delay_s=delay, snapshot=snapshot)
        with contextlib.suppress(Exception):
            logger.critical("Shutdown watchdog fired after %.0fs — forcing process exit "
                            "(asyncio drain path appears wedged; see %s)", delay, target)
        for stream in (sys.stdout, sys.stderr):
            with contextlib.suppress(Exception):
                stream.flush()
        # Mirror _exit_after_graceful_shutdown: release PID file + runtime lock BEFORE the log drain
        # (locks must never be stranded), then drain the async log queue so logger.critical
        # reaches the file before os._exit skips atexit.
        with contextlib.suppress(Exception):
            from gateway.status import remove_pid_file, release_gateway_runtime_lock
            remove_pid_file()
            release_gateway_runtime_lock()
        with contextlib.suppress(Exception):
            from hermes_logging import drain_log_queue
            drain_log_queue(timeout=1.0)
        _mark_exited_quietly(exit_code, "shutdown_watchdog")
        os._exit(exit_code)
    try:
        threading.Thread(target=_watchdog, daemon=True, name=name).start()
    except Exception:
        logger.debug("Failed to arm shutdown watchdog", exc_info=True)
    return done


async def _tick_socket_handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    """Answer a liveness ping with one byte; never raises. Runs on the gateway loop, so a reply
    witnesses loop schedulability; the write is a socket-buffer copy (no fsync), immune to the
    stalls that age the heartbeat."""
    with contextlib.suppress(Exception):
        writer.write(b"1")
        await writer.drain()
    with contextlib.suppress(Exception):
        writer.close()


def _sweep_stale_tick_sockets(own_path: Path) -> None:
    """Unlink loop-tick socket nodes left by dead PIDs (POSIX only; never raises).
    create_unix_server removes a leftover node at OUR path (os._exit / SIGKILL skip the
    finally-unlink) but not SIBLING nodes from other dead PIDs. os.kill(pid, 0) is a liveness
    probe only on POSIX (Windows would TerminateProcess)."""
    try:
        for stale in (p for p in own_path.parent.glob("gateway.loop-tick.*.sock") if p != own_path):
            try:
                os.kill(
                    int(stale.name.split(".")[-2]), 0
                )  # windows-footgun: ok — POSIX-only caller
            except (ValueError, IndexError, OSError):
                stale.unlink(missing_ok=True)
    except Exception:
        logger.debug("stale loop-tick socket sweep failed", exc_info=True)


async def loop_heartbeat_forever(
    *, interval_s: float = DEFAULT_HEARTBEAT_INTERVAL_S, start_time: Optional[float] = None,
    home: Optional[Path] = None, should_continue: Optional[Callable[[], bool]] = None,
) -> None:
    """Rewrite the loop heartbeat file on a cadence until cancelled / gated off.

    Runs on the gateway loop so a frozen loop lets the file age for monitors. The write
    (``atomic_json_write`` -> ``os.fsync``) goes to a thread: inline, a stalled filesystem blocked
    the loop inside its own heartbeat and the liveness watchdog killed it (WSL2 fsync stalls p99 31s
    / max 112s vs a ~90-120s budget). The thread write is *awaited*, never fire-and-forget: an
    unawaited task would keep the file fresh while the loop was wedged, and one in-flight write at a
    time keeps a long stall from queuing a thread per interval. Off-loop, file freshness alone no
    longer proves loop schedulability, so this task also arms a loop-scheduling witness
    (``_tick_socket_handler``) recorded as ``loop_tick_socket``; probes must require the witness to
    agree before classifying WEDGED.
    """
    interval = _coerce_float(interval_s, DEFAULT_HEARTBEAT_INTERVAL_S, floor=1.0)
    # Arm the witness. Best-effort: a failed bind only disables it, and the payload flag makes
    # probes classify UNKNOWN, never WEDGED (drain backstop stays). asyncio AF_UNIX is POSIX-only
    # (ungated it raised AttributeError on native Windows), so non-POSIX binds TCP loopback and
    # publishes ``loop_tick_tcp_port``.
    tick_server = tick_socket_path = tick_tcp_port = None
    try:
        if os.name == "posix":
            tick_socket_path = get_loop_tick_socket_path(home)
            tick_socket_path.parent.mkdir(parents=True, exist_ok=True)
            _sweep_stale_tick_sockets(tick_socket_path)
            tick_server = await asyncio.start_unix_server(
                _tick_socket_handler, path=str(tick_socket_path)
            )
        else:
            tick_server = await asyncio.start_server(_tick_socket_handler, host="127.0.0.1", port=0)
            for _s in tick_server.sockets or []:
                with contextlib.suppress(Exception):
                    if isinstance(_sname := _s.getsockname(), tuple) and len(_sname) >= 2:
                        tick_tcp_port = int(_sname[1])
                        break
    except Exception:
        tick_server = tick_tcp_port = None
        logger.warning(
            "Loop tick socket unavailable — liveness probes will have no "
            "loop-scheduling witness and will not escalate on a stale heartbeat",
            exc_info=True,
        )

    async def _write_off_loop() -> None:
        # write_loop_heartbeat never raises, so a failure here is an executor problem and must not
        # kill the task.
        try:
            await asyncio.to_thread(
                write_loop_heartbeat,
                start_time=start_time,
                home=home,
                extra={
                    "loop_tick_socket": tick_server is not None,
                    "loop_tick_tcp_port": tick_tcp_port,
                },
            )
        except Exception:
            logger.debug("Loop heartbeat write failed off-loop", exc_info=True)
    try:
        await _write_off_loop()  # immediate first write so monitors see a fresh file at once
        while should_continue is None or should_continue():
            await asyncio.sleep(interval)
            if should_continue is not None and not should_continue():
                return
            await _write_off_loop()
    finally:
        if tick_server is not None:
            tick_server.close()
            with contextlib.suppress(Exception):
                await tick_server.wait_closed()
            if tick_socket_path is not None:
                with contextlib.suppress(Exception):
                    tick_socket_path.unlink(missing_ok=True)
