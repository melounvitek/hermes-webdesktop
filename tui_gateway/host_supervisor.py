"""Supervisor for the dashboard compute-host child process.

The dashboard process owns sockets and JSON-RPC dispatch.  When
``dashboard.turn_isolation`` is enabled, agent turns move behind one persistent
``python -m tui_gateway.compute_host`` child so compute-heavy agent threads do
not contend with the serving process' event loop for the same GIL.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import queue
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home
from tools.environments.local import hermes_subprocess_env

logger = logging.getLogger(__name__)
_Thread = threading.Thread

MUTATOR_ROUTE_TABLE: dict[str, str] = {
    "prompt.submit": "turn-path",
    "session.interrupt": "turn-path",
    "reload.mcp": "run-concurrent",
    "session.save": "run-concurrent",
    "session.compress": "idle-gated",
    "prompt.submit.truncate": "idle-gated",
    "slash.model": "idle-gated",
    "slash.personality": "idle-gated",
    "slash.prompt": "idle-gated",
    "slash.compress": "idle-gated",
    "session.reset": "idle-gated",
    "session.history.reload": "idle-gated",
    "slash.retry": "idle-gated",
}

_REGISTRY_NAME = "dashboard-compute-host.json"
_RESPAWN_WINDOW_SECS = 300.0
_SHUTDOWN_TIMEOUT_SECS = 10.0
# Late control-ack handlers: a compress that outlives its RPC waiter can run for
# the full compression ceiling plus a stall-fallback retry, so keep
# registrations well past that — but bounded.
_LATE_CONTROL_TTL_SECS = 1800.0
_LATE_CONTROL_MAX = 64
# Host frames whose ``request_id`` resolves a pending/late control waiter.
_CONTROL_REPLY_TYPES = frozenset(
    {"control.ack", "control.error", "respond.ack", "respond.error", "interrupt.ack", "reload_mcp.ack", "shutdown.ack"}
)


def append_log_record(path: str | Path, record: str) -> None:
    """Append one log record using O_APPEND and exactly one os.write call."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    text = record if record.endswith("\n") else f"{record}\n"
    data = text.encode("utf-8", errors="replace")
    fd = os.open(str(p), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(fd, data)
    finally:
        os.close(fd)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _build_sha() -> str:
    """Current checkout's HEAD sha, or ``"unknown"``. Shared with ``compute_host``
    so the hello handshake and the supervisor's expectation agree byte-for-byte."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(_repo_root()),
            text=True,
            encoding="utf-8",
            errors="replace",
            stderr=subprocess.DEVNULL,
            timeout=2,
        ).strip()
    except Exception:
        return "unknown"


def _default_registry_path() -> Path:
    return get_hermes_home() / "state" / _REGISTRY_NAME


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except Exception:
        return False


def _pid_command(pid: int) -> str:
    if pid <= 0:
        return ""
    with contextlib.suppress(Exception):  # Linux fast path
        data = (Path("/proc") / str(pid) / "cmdline").read_bytes()
        if data:
            return data.replace(b"\x00", b" ").decode("utf-8", errors="replace")
    try:
        return subprocess.check_output(
            ["ps", "-p", str(pid), "-o", "command="],
            text=True,
            encoding="utf-8",
            errors="replace",
            stderr=subprocess.DEVNULL,
            timeout=2,
        ).strip()
    except Exception:
        return ""


def is_compute_host_identity(pid: int) -> bool:
    return "tui_gateway.compute_host" in _pid_command(pid)


class HostSupervisor:
    """Own one persistent compute-host child and relay its frames."""

    def __init__(
        self,
        *,
        registry_path: str | Path | None = None,
        argv: list[str] | None = None,
        cwd: str | Path | None = None,
        env: dict[str, str] | None = None,
        rpc_sink: Callable[[dict], None] | None = None,
        respawn_max: int = 3,
        heartbeat_secs: int = 15,
        expected_build_sha: str | None = None,
        expected_hermes_home: str | None = None,
        autostart: bool = True,
    ) -> None:
        self.registry_path = Path(registry_path) if registry_path is not None else _default_registry_path()
        self.argv = argv or [sys.executable, "-m", "tui_gateway.compute_host"]
        self.cwd = Path(cwd) if cwd is not None else _repo_root()
        self.env = env
        self.rpc_sink = rpc_sink or (lambda _obj: None)
        self.respawn_max = max(0, int(respawn_max))
        self.heartbeat_secs = max(1, int(heartbeat_secs))
        self.expected_build_sha = expected_build_sha if expected_build_sha is not None else _build_sha()
        self.expected_hermes_home = expected_hermes_home if expected_hermes_home is not None else str(get_hermes_home())

        self._lock = threading.RLock()
        self._proc: subprocess.Popen[str] | None = None
        self._stdout_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._wait_thread: threading.Thread | None = None
        self._hello_event = threading.Event()
        self._hello: dict[str, Any] = {}
        self._closing = False
        self._stopped_respawning = False
        self._restart_times: list[float] = []
        self._pending_turns: dict[str, tuple[str, Callable[[dict], None] | None]] = {}
        self._pending_controls: dict[str, queue.Queue[dict]] = {}
        # request_id -> (registered_at, handler) for control waiters that timed
        # out while their host work still runs; without it the eventual
        # control.ack matched no queue and was silently dropped.
        self._late_control_handlers: dict[str, tuple[float, Callable[[dict], None]]] = {}
        self._stderr_tail: list[str] = []
        self._last_progress_counter = 0

        if autostart:
            self.start()

    @property
    def pid(self) -> int:
        proc = self._proc
        return int(proc.pid or 0) if proc is not None else 0

    @property
    def hello(self) -> dict[str, Any]:
        return dict(self._hello)

    def is_running(self) -> bool:
        proc = self._proc
        return proc is not None and proc.poll() is None and not self._stopped_respawning

    def start(self) -> None:
        with self._lock:
            if self.is_running():
                return
            self._closing = False
            self.reconcile_startup_orphan()
            self._spawn_locked(reason="startup")

    def shutdown(self) -> None:
        with self._lock:
            self._closing = True
            proc = self._proc
        if proc is None:
            return
        try:
            if proc.poll() is None and proc.stdin is not None:
                self._send_frame({"type": "shutdown", "request_id": f"shutdown-{uuid.uuid4().hex}"})
                proc.wait(timeout=_SHUTDOWN_TIMEOUT_SECS)
        except Exception:
            self._terminate_process(proc)
        finally:
            self._remove_registry()

    def reconcile_startup_orphan(self) -> str:
        """Terminate a stale registered host, guarding against PID reuse."""
        try:
            data = json.loads(self.registry_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return "none"
        except Exception:
            self._remove_registry()
            return "invalid-registry"

        try:
            pid = int(data.get("host_pid") or 0)
        except Exception:
            pid = 0
        if pid <= 0 or not _pid_alive(pid):
            self._remove_registry()
            return "not-running"
        if not self._pid_matches_compute_host(pid):
            # PID was reused by another process. Never signal it.
            self._remove_registry()
            return "pid-reuse-ignored"

        self._terminate_pid(pid, timeout=_SHUTDOWN_TIMEOUT_SECS)
        self._remove_registry()
        return "terminated"

    def submit_turn(
        self,
        frame: dict[str, Any],
        *,
        on_complete: Callable[[dict], None] | None = None,
    ) -> str:
        self.start()
        request_id = str(frame.get("request_id") or uuid.uuid4().hex)
        sid = str(frame.get("sid") or "")
        payload = {**frame, "type": "turn.start", "request_id": request_id}
        with self._lock:
            self._pending_turns[request_id] = (sid, on_complete)
        try:
            self._send_frame(payload)
        except Exception as exc:
            with self._lock:
                self._pending_turns.pop(request_id, None)
            if on_complete is not None:
                on_complete({"type": "turn.error", "sid": sid, "request_id": request_id, "reason": "send_failed", "message": str(exc)})
            raise
        return request_id

    def interrupt(self, sid: str, *, request_id: str | None = None) -> None:
        self.start()
        self._send_frame({"type": "interrupt", "sid": sid, "request_id": request_id or uuid.uuid4().hex})

    def respond(self, sid: str, params: dict[str, Any], *, timeout: float = 15.0) -> dict:
        """Deliver an interactive prompt response to the host that owns it."""
        self.start()
        request_id = uuid.uuid4().hex
        q: queue.Queue[dict] = queue.Queue(maxsize=1)
        with self._lock:
            self._pending_controls[request_id] = q
        try:
            self._send_frame({"type": "respond", "sid": sid, "request_id": request_id, "params": dict(params)})
            return q.get(timeout=timeout)
        finally:
            with self._lock:
                self._pending_controls.pop(request_id, None)

    def reload_mcp(self, sid: str, *, request_id: str | None = None) -> dict:
        return self.control(
            sid,
            route_name="reload.mcp",
            payload={"type": "reload_mcp", "sid": sid, "request_id": request_id or uuid.uuid4().hex},
            wait=True,
        )

    def control(
        self,
        sid: str,
        *,
        route_name: str,
        payload: dict[str, Any] | None = None,
        wait: bool = True,
        timeout: float = 30.0,
        on_late_ack: Callable[[dict], None] | None = None,
    ) -> dict:
        """Send a control frame; with ``wait`` block up to ``timeout`` for its ack.

        ``on_late_ack`` (only with ``wait``) keeps the request adoptable after the
        waiter gives up: the host's eventual ``control.ack``/``control.error``/
        ``error`` for this ``request_id`` fires the handler once instead of being
        dropped. Bounded by ``_LATE_CONTROL_TTL_SECS`` / ``_LATE_CONTROL_MAX``.
        """
        if route_name not in MUTATOR_ROUTE_TABLE:
            raise ValueError(f"unclassified host mutator route: {route_name}")
        self.start()
        request_id = str((payload or {}).get("request_id") or uuid.uuid4().hex)
        frame = {"type": "control", **(payload or {}), "sid": sid, "route_name": route_name, "request_id": request_id}
        q: queue.Queue[dict] | None = None
        if wait:
            q = queue.Queue(maxsize=1)
            with self._lock:
                self._pending_controls[request_id] = q
        self._send_frame(frame)
        if not wait or q is None:
            return {"status": "sent", "request_id": request_id}
        try:
            return q.get(timeout=timeout)
        except queue.Empty:
            if on_late_ack is not None:
                self._register_late_control_handler(request_id, on_late_ack)
            raise
        finally:
            with self._lock:
                self._pending_controls.pop(request_id, None)

    def _register_late_control_handler(self, request_id: str, handler: Callable[[dict], None]) -> None:
        now = time.monotonic()
        with self._lock:
            for rid in [r for r, (at, _cb) in self._late_control_handlers.items() if now - at > _LATE_CONTROL_TTL_SECS]:
                self._late_control_handlers.pop(rid, None)
            while len(self._late_control_handlers) >= _LATE_CONTROL_MAX:
                oldest = min(self._late_control_handlers, key=lambda rid: self._late_control_handlers[rid][0])
                self._late_control_handlers.pop(oldest, None)
            self._late_control_handlers[request_id] = (now, handler)

    def _deliver_control_frame(self, request_id: str, frame: dict[str, Any]) -> None:
        with self._lock:
            q = self._pending_controls.get(request_id)
            late = None if q is not None else self._late_control_handlers.pop(request_id, None)
        if q is not None:
            with contextlib.suppress(queue.Full):
                q.put_nowait(frame)
            return
        if late is None:
            return
        try:
            late[1](frame)
        except Exception:
            logger.exception("compute host late control ack handler failed (request_id=%s)", request_id)

    def _spawn_locked(self, *, reason: str) -> None:
        if self._stopped_respawning:
            raise RuntimeError("compute host respawn disabled after crash loop")
        self._hello_event.clear()
        self._hello = {}
        env = hermes_subprocess_env(inherit_credentials=True)
        env.update(os.environ)
        if self.env:
            env.update(self.env)
        env["HERMES_COMPUTE_HOST_HEARTBEAT_SECS"] = str(self.heartbeat_secs)
        root = str(_repo_root())
        env.setdefault("PYTHONPATH", root)
        if root not in env["PYTHONPATH"].split(os.pathsep):
            env["PYTHONPATH"] = root + os.pathsep + env["PYTHONPATH"]
        proc = subprocess.Popen(
            self.argv,
            cwd=str(self.cwd),
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            # Lossy UTF-8 decode: a locale-mismatched byte must not raise inside
            # the drain threads and kill the supervisor.
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            start_new_session=True,
        )
        self._proc = proc
        self._stdout_thread = _Thread(target=self._drain_stdout, args=(proc,), name="compute-host-stdout", daemon=True)
        self._stderr_thread = _Thread(target=self._drain_stderr, args=(proc,), name="compute-host-stderr", daemon=True)
        self._wait_thread = _Thread(target=self._wait_for_exit, args=(proc,), name="compute-host-wait", daemon=True)
        for t in (self._stdout_thread, self._stderr_thread, self._wait_thread):
            t.start()
        if not self._hello_event.wait(timeout=10.0):
            self._terminate_process(proc)
            raise RuntimeError(f"compute host did not send hello; stderr={self._stderr_tail[-5:]}")
        self._validate_hello()
        self._persist_registry()
        logger.info("compute host started pid=%s reason=%s", proc.pid, reason)

    def _validate_hello(self) -> None:
        hello = self._hello
        if not hello:
            raise RuntimeError("compute host missing hello")
        got_home = str(hello.get("hermes_home") or "")
        if got_home and got_home != self.expected_hermes_home:
            raise RuntimeError(f"compute host HERMES_HOME mismatch: {got_home} != {self.expected_hermes_home}")
        got_sha = str(hello.get("build_sha") or "")
        if self.expected_build_sha != "unknown" and got_sha not in {"", "unknown", self.expected_build_sha}:
            raise RuntimeError(f"compute host build mismatch: {got_sha} != {self.expected_build_sha}")

    def _persist_registry(self) -> None:
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.registry_path.with_suffix(self.registry_path.suffix + ".tmp")
        payload = {
            "host_pid": self.pid,
            "boot_id": self._hello.get("boot_id") or "",
            "build_sha": self._hello.get("build_sha") or "",
            "started_at": time.time(),
            "argv": self.argv,
        }
        tmp.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        tmp.replace(self.registry_path)

    def _remove_registry(self) -> None:
        try:
            self.registry_path.unlink(missing_ok=True)
        except Exception:
            logger.debug("failed to remove compute host registry", exc_info=True)

    def _send_frame(self, frame: dict[str, Any]) -> None:
        with self._lock:
            proc = self._proc
            if proc is None or proc.poll() is not None or proc.stdin is None:
                raise RuntimeError("compute host is not running")
            proc.stdin.write(json.dumps(frame, separators=(",", ":"), ensure_ascii=False) + "\n")
            proc.stdin.flush()

    def _drain_stdout(self, proc: subprocess.Popen[str]) -> None:
        assert proc.stdout is not None
        for raw in proc.stdout:
            try:
                frame = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning("compute host emitted invalid json: %r", raw[:200])
                continue
            if isinstance(frame, dict):
                self._handle_host_frame(frame)

    def _drain_stderr(self, proc: subprocess.Popen[str]) -> None:
        assert proc.stderr is not None
        for raw in proc.stderr:
            text = raw.rstrip("\n")
            if text:
                self._stderr_tail = (self._stderr_tail + [text])[-80:]
                logger.warning("compute host stderr: %s", text)

    def _handle_host_frame(self, frame: dict[str, Any]) -> None:
        ftype = str(frame.get("type") or "")
        if ftype in _CONTROL_REPLY_TYPES or (ftype == "error" and frame.get("request_id")):
            self._deliver_control_frame(str(frame.get("request_id") or ""), frame)
            return
        handler = self._HOST_FRAME_HANDLERS.get(ftype)
        if handler is not None:
            getattr(self, handler)(frame)

    # host frame ``type`` -> handler method name (see also _CONTROL_REPLY_TYPES).
    _HOST_FRAME_HANDLERS: dict[str, str] = {
        "hello": "_on_hello",
        "hb": "_on_heartbeat",
        "rpc": "_on_rpc",
        "turn.end": "_complete_turn",
        "turn.error": "_complete_turn",
    }

    def _on_hello(self, frame: dict[str, Any]) -> None:
        self._hello = dict(frame)
        self._hello_event.set()

    def _on_heartbeat(self, frame: dict[str, Any]) -> None:
        self._last_progress_counter = int(frame.get("progress_counter") or self._last_progress_counter)
        logger.debug("compute host heartbeat: %s", frame)

    def _on_rpc(self, frame: dict[str, Any]) -> None:
        message = frame.get("message")
        if isinstance(message, dict):
            self.rpc_sink(message)

    def _complete_turn(self, frame: dict[str, Any]) -> None:
        request_id = str(frame.get("request_id") or "")
        with self._lock:
            pending = self._pending_turns.pop(request_id, None)
        if pending is not None and pending[1] is not None:
            try:
                pending[1](frame)
            except Exception:
                logger.exception("compute host turn completion callback failed")

    def _wait_for_exit(self, proc: subprocess.Popen[str]) -> None:
        code = proc.wait()
        if self._closing:
            return
        with self._lock:
            if self._proc is not proc:
                return
            self._proc = None
        self._remove_registry()
        self._fail_pending_turns(reason="crash", message=f"compute host exited with code {code}")
        self._maybe_respawn_after_crash()

    def _fail_pending_turns(self, *, reason: str, message: str) -> None:
        with self._lock:
            pending = self._pending_turns
            self._pending_turns = {}
        for request_id, (sid, cb) in pending.items():
            self.rpc_sink(
                {
                    "jsonrpc": "2.0",
                    "method": "event",
                    "params": {"type": "error", "session_id": sid, "payload": {"message": message, "reason": reason}},
                }
            )
            if cb is not None:
                try:
                    cb({"type": "turn.error", "sid": sid, "request_id": request_id, "reason": reason, "message": message})
                except Exception:
                    logger.exception("compute host error callback failed")
        # A crashed host never emits the late acks timed-out control waiters still
        # expect; fail them too so the client's "still running" notice can't hang.
        with self._lock:
            late = self._late_control_handlers
            self._late_control_handlers = {}
        for request_id, (_registered_at, handler) in late.items():
            try:
                handler({"type": "control.error", "request_id": request_id, "reason": reason, "message": message})
            except Exception:
                logger.exception("compute host late control error handler failed")

    def _maybe_respawn_after_crash(self) -> None:
        now = time.monotonic()
        self._restart_times = [t for t in self._restart_times if now - t <= _RESPAWN_WINDOW_SECS]
        if len(self._restart_times) >= self.respawn_max:
            self._stopped_respawning = True
            logger.error("compute host crash loop: max %s restarts per 5min reached; not respawning", self.respawn_max)
            return
        self._restart_times.append(now)
        # Small bounded backoff; tests and first recovery stay quick.
        delay = min(5.0, 0.25 * (2 ** max(0, len(self._restart_times) - 1)))

        def _respawn() -> None:
            time.sleep(delay)
            with self._lock:
                if self._closing or self._stopped_respawning or self._proc is not None:
                    return
                try:
                    self._spawn_locked(reason="crash")
                except Exception:
                    logger.exception("compute host respawn failed")

        _Thread(target=_respawn, name="compute-host-respawn", daemon=True).start()

    _pid_matches_compute_host = staticmethod(is_compute_host_identity)

    def _terminate_pid(self, pid: int, *, timeout: float = _SHUTDOWN_TIMEOUT_SECS) -> None:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        except Exception:
            logger.debug("failed to SIGTERM compute host pid=%s", pid, exc_info=True)
            return
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not _pid_alive(pid):
                return
            time.sleep(0.05)
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        except Exception:
            logger.debug("failed to SIGKILL compute host pid=%s", pid, exc_info=True)

    def _terminate_process(self, proc: subprocess.Popen[str]) -> None:
        if proc.poll() is not None:
            return
        try:
            proc.terminate()
            proc.wait(timeout=_SHUTDOWN_TIMEOUT_SECS)
            return
        except Exception:
            pass
        with contextlib.suppress(Exception):
            proc.kill()
        with contextlib.suppress(Exception):
            proc.wait(timeout=2)


__all__ = [
    "MUTATOR_ROUTE_TABLE",
    "HostSupervisor",
    "append_log_record",
    "is_compute_host_identity",
]
