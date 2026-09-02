"""Session-persistent Python kernels for execute_code.

One Python child stays alive per (owner, mode, interpreter, cwd, tool-set) and
runs one code cell per call, so variables/imports/data survive across calls.

Design constraints, in order:

- **Same security envelope as per-call**: same ``_build_child_env`` (secret
  scrubbing, tool whitelist, PYTHONPATH rules), same ``_rpc_server_loop`` with
  the same token and per-cell tool budget, same ANSI strip + secret redaction.
  Nothing here widens what a script can reach — only how long it lives.
- **A wedged kernel dies, never hangs the agent.** Timeout or interrupt kills
  the whole kernel process tree and drops the registry entry; the next call
  spawns fresh. Losing state is deliberate: there is no reliable way to
  interrupt one cell in place without leaving the interpreter unknown.
- **The env is frozen at spawn.** Env passthrough registered after the kernel
  started is invisible until ``reset=true``; the result payload names the
  kernel so this is diagnosable.

Wire protocol (host <-> child): requests are one JSON object per stdin line
``{"id", "code"}``; responses are framed on stdout as
``<SENTINEL> <byte-length>\\n<json>`` with a per-kernel random SENTINEL from
the environment. Bytes outside frames are raw fd-level output (subprocesses
inherit the real stdout) and are attributed to the running cell — calls are
serialized per kernel, so attribution is unambiguous. Python-level
stdout/stderr are captured via ``contextlib.redirect_*`` into the payload; a
script forging a frame can only fake its own cell result (same trust position
as a per-call script printing a forged success message).

Also hosts what ``tools.code_kernel_remote`` shares: owner resolution, the
registry lifecycle, and the runner's cell-exec core.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import queue
import secrets
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_IS_WINDOWS = sys.platform == "win32"

# Runner-side cap on captured python-level output; the host applies its own
# MAX_STDOUT truncation again.
_RUNNER_CAPTURE_BYTES = 1_000_000

# Shared by both generated runners (which define _CAPTURE_LIMIT first): exec one
# request in the persistent GLOBALS namespace and build the response payload.
# `__name__` is `__main__` so scripts behave like the per-call path.
RUNNER_CELL_SOURCE = '''\
GLOBALS = {"__name__": "__main__", "__builtins__": __builtins__}


def _clip(text):
    return (text, False) if len(text) <= _CAPTURE_LIMIT else (text[:_CAPTURE_LIMIT], True)


def run_cell(request, execution_count):
    """Exec one cell; returns (response payload, FULL stdout text)."""
    out, err = io.StringIO(), io.StringIO()
    status, trace = "ok", ""
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            exec(compile(request["code"], "<cell>", "exec"), GLOBALS)
    except SystemExit as exc:
        status, trace = "exit", "SystemExit: " + repr(exc.code)
    except BaseException:
        status, trace = "error", traceback.format_exc()
    stdout_text, stdout_clipped = _clip(out.getvalue())
    stderr_text, stderr_clipped = _clip(err.getvalue())
    return {
        "id": request.get("id", ""), "status": status,
        "stdout": stdout_text, "stderr": stderr_text,
        "stdout_clipped": stdout_clipped, "stderr_clipped": stderr_clipped,
        "traceback": trace, "execution_count": execution_count,
    }, out.getvalue()
'''

KERNEL_RUNNER_SOURCE = '''\
"""Auto-generated Hermes session-kernel runner. One exec cell per request."""
import contextlib
import io
import json
import os
import sys
import traceback

_SENTINEL = os.environ["HERMES_KERNEL_SENTINEL"]
_CAPTURE_LIMIT = {capture_limit}
_SPILL_DIR = os.environ.get("HERMES_KERNEL_SPILL_DIR", "")
_SPILL_CAP = {spill_cap}
_real_stdout = sys.stdout

{cell_source}

def _spill(text, spill_name):
    """Best-effort: write the FULL clipped stdout to disk, return its path or ""."""
    if not _SPILL_DIR:
        return ""
    try:
        spill_path = os.path.join(_SPILL_DIR, spill_name)
        with open(spill_path, "w", encoding="utf-8", errors="replace") as f:
            f.write(text[:_SPILL_CAP])
            if len(text) > _SPILL_CAP:
                f.write("\\n\\n[... spill capped ...]")
        return spill_path
    except Exception:
        return ""


def _reply(payload):
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    _real_stdout.buffer.write(("\\n" + _SENTINEL + " " + str(len(body)) + "\\n").encode("utf-8"))
    _real_stdout.buffer.write(body)
    _real_stdout.buffer.flush()


def main():
    execution_count = 0
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except ValueError:
            continue
        execution_count += 1
        payload, full_stdout = run_cell(request, execution_count)
        payload["stdout_spill_path"] = (
            _spill(full_stdout, "cell_%06d_stdout.txt" % execution_count)
            if payload["stdout_clipped"] else ""
        )
        _reply(payload)
        if payload["status"] == "exit":
            break


if __name__ == "__main__":
    main()
'''.format(cell_source=RUNNER_CELL_SOURCE, capture_limit=_RUNNER_CAPTURE_BYTES, spill_cap=5_000_000)


class CellAuthority:
    """The approval/context identity of exactly one execute_code cell.

    Interpreter state persists across cells; RPC authority must not. Each cell
    installs a fresh authority — captured from the CALLING thread at cell
    start, exactly what ``propagate_context_to_thread`` would capture for a
    per-call RPC thread — and retires it when the cell settles, so a late tool
    call (a background thread the cell left behind, a raced client write) is
    refused instead of running under a stale approval/session/turn identity.
    """

    def __init__(self, task_id: str):
        import contextvars

        self.task_id = task_id
        self.ctx = contextvars.copy_context()
        self.active = True
        self._api = None  # (get_approval, get_sudo, set_approval, set_sudo)
        self._callbacks = (None, None)
        try:
            from tools.thread_context import _callback_api

            self._api = _callback_api()
            self._callbacks = (self._api[0](), self._api[1]())
        except Exception:
            # Fail-closed, mirroring propagate_context_to_thread: with no
            # callbacks installed, dangerous approvals deny.
            self._api = None

    def retire(self) -> None:
        self.active = False

    def dispatch(self, tool_name: str, tool_args: dict) -> str:
        """Run one tool call under THIS cell's context and callbacks."""
        from tools.code_execution_tool import tool_error

        if not self.active:
            return tool_error(
                "No active execute_code cell: the cell this kernel call "
                "belonged to has settled, so its tool authority is retired."
            )
        return self.ctx.run(self._invoke, tool_name, tool_args)

    def _invoke(self, tool_name: str, tool_args: dict) -> str:
        from model_tools import handle_function_call

        previous = None
        if self._api is not None:
            get_approval, get_sudo, set_approval, set_sudo = self._api
            try:
                previous = (get_approval(), get_sudo())
                set_approval(self._callbacks[0])
                set_sudo(self._callbacks[1])
            except Exception:
                previous = None
        try:
            return handle_function_call(tool_name, tool_args, task_id=self.task_id)
        finally:
            if previous is not None:
                try:
                    set_approval(previous[0])
                    set_sudo(previous[1])
                except Exception:
                    pass


class _BoundedBuffer:
    """Byte chunks capped at a total size; ``drain`` returns text and resets."""

    def __init__(self):
        self.chunks: List[bytes] = []
        self.total = 0

    def append(self, data: bytes, cap: int) -> None:
        if self.total >= cap:
            return
        keep = data[: cap - self.total]
        self.chunks.append(keep)
        self.total += len(keep)

    def drain(self) -> str:
        chunks, self.chunks, self.total = self.chunks, [], 0
        return b"".join(chunks).decode("utf-8", errors="replace")


class SessionKernel:
    """One live kernel process plus its RPC server and reader threads."""

    def __init__(self, key: Tuple):
        self.key = key
        self.owner: str = key[0]
        self.lock = threading.Lock()
        self.proc: Optional[subprocess.Popen] = None
        self.tmpdir = self.rpc_token = self.sentinel = ""
        self.sock_path: Optional[str] = None
        self.server_sock: Optional[socket.socket] = None
        self.stop_event = threading.Event()
        self.tool_call_log: List = []
        self.tool_call_counter: List[int] = [0]
        self.response_q: "queue.Queue[dict]" = queue.Queue()
        self.raw, self.stderr = _BoundedBuffer(), _BoundedBuffer()
        self.execution_count = 0
        self.last_used: float = time.monotonic()
        self.cell_authority: Optional[CellAuthority] = None

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None


class KernelRegistry:
    """Key -> kernel map plus its lock (shared with the remote registry).

    Kernels are popped under the lock and torn down outside it — teardown
    may block on the child process or the remote transport.
    """

    def __init__(self, teardown: Callable[[Any], None]):
        self.kernels: Dict[Tuple, Any] = {}
        self.lock = threading.Lock()
        self._teardown = teardown

    def pop_all(self, owner: Optional[str] = None) -> list:
        """Pop every kernel, or every kernel one owner (key[0]) holds."""
        with self.lock:
            doomed = [key for key in self.kernels if owner is None or key[0] == owner]
            return [self.kernels.pop(key) for key in doomed]

    def shutdown(self, owner: Optional[str] = None) -> None:
        for kernel in self.pop_all(owner):
            self._teardown(kernel)

    def discard(self, key: Tuple, kernel: Any) -> None:
        """Drop one registry entry and tear the kernel down."""
        with self.lock:
            self.kernels.pop(key, None)
        self._teardown(kernel)


_REGISTRY = KernelRegistry(lambda kernel: _teardown(kernel))
_KERNELS: Dict[Tuple, SessionKernel] = _REGISTRY.kernels
_KERNELS_LOCK = _REGISTRY.lock

# Bounded lifecycle defaults (config: code_execution.max_session_kernels /
# code_execution.kernel_idle_timeout). A long-lived gateway must never
# accumulate one live child per finished conversation: stable owner id,
# owner-teardown disposal, idle reaping, max-live bound.
DEFAULT_MAX_SESSION_KERNELS = 4
DEFAULT_KERNEL_IDLE_TIMEOUT = 1800


def _lifecycle_limits() -> Tuple[int, int]:
    from tools.code_execution_tool import _load_config

    config = _load_config()

    def limit(key: str, default: int) -> int:
        try:
            return max(1, int(config.get(key, default)))
        except (TypeError, ValueError):
            return default

    return (limit("max_session_kernels", DEFAULT_MAX_SESSION_KERNELS),
            limit("kernel_idle_timeout", DEFAULT_KERNEL_IDLE_TIMEOUT))


def _resolve_owner(task_id: str) -> str:
    """The stable identity a session kernel belongs to.

    The conversation's approval session key: context-propagated, stable across
    turns, distinct per session. ``run_agent`` mints a fresh task id per
    top-level turn, so a task-keyed kernel would neither survive the next turn
    nor ever be torn down with anything; the task id is only the last-resort
    owner for embeds and tests with no session context.

    Delegated children run in a copy of the parent's context and INHERIT its
    approval session key — without the ``::child::`` qualifier a child's
    execute_code would attach to the parent's kernel and read its in-memory
    state (verified live, both directions). Children get their own kernels,
    keyed by their delegation session id.
    """
    try:
        from tools.approval import get_current_session_key

        session_key = get_current_session_key(default="")
    except Exception:
        session_key = ""
    owner = session_key or (task_id or "")
    try:
        from agent.delegation_context import is_delegated_child_context

        if is_delegated_child_context():
            from gateway.session_context import get_session_env

            child_id = get_session_env("HERMES_SESSION_ID", "") or (task_id or "")
            owner = f"{owner}::child::{child_id}"
    except Exception:
        pass
    return owner


def _kernel_key(owner: str, mode: str, child_python: str, child_cwd: str,
                sandbox_tools: frozenset) -> Tuple:
    return (owner or "", mode, child_python, child_cwd, tuple(sorted(sandbox_tools)))


def shutdown_all_kernels() -> None:
    """Kill every session kernel. Registered via atexit; also used by tests."""
    _REGISTRY.shutdown()


def shutdown_kernels_for_owner(owner: str) -> None:
    """Dispose every kernel a session owns.

    Wired into ``tools.approval.clear_session`` so kernels die at the same
    session boundary that clears the owner's approval and yolo state
    (/new and session close).
    """
    if owner:
        _REGISTRY.shutdown(owner)


def _reap_unlocked() -> List[SessionKernel]:
    """Pop idle-expired kernels; caller tears them down outside the lock."""
    _, idle_timeout = _lifecycle_limits()
    now = time.monotonic()
    doomed = [key for key, kernel in _KERNELS.items() if now - kernel.last_used > idle_timeout]
    return [_KERNELS.pop(key) for key in doomed]


def _evict_over_cap_unlocked(keep: Tuple) -> List[SessionKernel]:
    """Pop least-recently-used kernels beyond the process-wide cap."""
    cap, _ = _lifecycle_limits()
    by_age = sorted((key for key in _KERNELS if key != keep), key=lambda key: _KERNELS[key].last_used)
    return [_KERNELS.pop(key) for key in by_age[: max(0, len(_KERNELS) - cap)]]


atexit.register(shutdown_all_kernels)


def _teardown(kernel: SessionKernel) -> None:
    kernel.stop_event.set()
    if kernel.proc is not None and kernel.proc.poll() is None:
        from tools.code_execution_tool import _kill_process_group

        _kill_process_group(kernel.proc, escalate=True)
    sock, kernel.server_sock = kernel.server_sock, None
    try:
        if sock is not None:
            sock.close()
        if kernel.sock_path:
            os.unlink(kernel.sock_path)
    except OSError:
        pass
    if kernel.tmpdir:
        import shutil

        shutil.rmtree(kernel.tmpdir, ignore_errors=True)


def _rpc_forever(kernel: SessionKernel, max_tool_calls: int,
                 sandbox_tools: frozenset) -> None:
    """Serve tool RPC for the kernel's whole life.

    ``_rpc_server_loop`` serves one connection and returns on disconnect or its
    300s idle timeout; a kernel legitimately idles longer between cells, so
    re-accept until teardown (the client stub reconnects: HERMES_RPC_PERSISTENT).

    The serving thread carries NO frozen authority: every dispatch routes
    through the CURRENT cell's ``CellAuthority``, so a later cell's tool calls
    run under that cell's context, not whatever the first cell captured.
    """
    from tools.code_execution_tool import _rpc_server_loop, tool_error

    def _dispatch(tool_name: str, tool_args: dict) -> str:
        authority = kernel.cell_authority
        if authority is None:
            return tool_error(
                "No active execute_code cell: this kernel has no cell "
                "authority installed."
            )
        return authority.dispatch(tool_name, tool_args)

    while not kernel.stop_event.is_set():
        _rpc_server_loop(kernel.server_sock, "", kernel.tool_call_log, kernel.tool_call_counter,
                         max_tool_calls, sandbox_tools, kernel.stop_event, kernel.rpc_token,
                         dispatch=_dispatch)


def _stdout_reader(kernel: SessionKernel) -> None:
    """Split the child's stdout into protocol frames and raw passthrough."""
    from tools.code_execution_tool import MAX_STDOUT_BYTES

    assert kernel.proc is not None and kernel.proc.stdout is not None
    stream = kernel.proc.stdout
    marker = ("\n" + kernel.sentinel + " ").encode("utf-8")

    def raw(data: bytes) -> None:
        kernel.raw.append(data, MAX_STDOUT_BYTES)

    buf = b""
    while True:
        # read1: return as soon as any bytes arrive. A plain read(n) on a
        # BufferedReader blocks until n bytes or EOF, which would sit on a
        # complete frame smaller than the buffer forever.
        chunk = stream.read1(4096)
        if not chunk:
            if buf:
                raw(buf)
            kernel.response_q.put({"status": "kernel-eof"})
            return
        buf += chunk
        while True:
            index = buf.find(marker)
            if index < 0:
                # Keep a marker-sized tail in case the marker is split
                # across reads; everything before it is raw output.
                spill = buf[: -len(marker)] if len(buf) > len(marker) else b""
                if spill:
                    raw(spill)
                    buf = buf[len(spill):]
                break
            if index:
                raw(buf[:index])
            rest = buf[index + len(marker):]
            newline = rest.find(b"\n")
            if newline < 0:
                buf = buf[index:]
                break
            try:
                length = int(rest[:newline])
            except ValueError:
                # Not a real frame header (user output that happens to
                # contain the marker bytes); treat the marker as raw.
                raw(marker)
                buf = rest
                continue
            body = rest[newline + 1:]
            while len(body) < length:
                more = stream.read1(length - len(body))
                if not more:
                    kernel.response_q.put({"status": "kernel-eof"})
                    return
                body += more
            try:
                kernel.response_q.put(json.loads(body[:length].decode("utf-8", errors="replace")))
            except ValueError:
                kernel.response_q.put({"status": "protocol-error"})
            buf = body[length:]


def _stderr_reader(kernel: SessionKernel) -> None:
    from tools.code_execution_tool import MAX_STDERR_BYTES

    assert kernel.proc is not None and kernel.proc.stderr is not None
    while True:
        chunk = kernel.proc.stderr.read1(4096)
        if not chunk:
            return
        kernel.stderr.append(chunk, MAX_STDERR_BYTES)


def _bind_rpc_socket(kernel: SessionKernel) -> str:
    """Bind the tool-RPC listener: loopback TCP on Windows, 0600 UDS elsewhere."""
    if _IS_WINDOWS:
        kernel.sock_path = None
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.bind(("127.0.0.1", 0))
        host, port = server_sock.getsockname()[:2]
        rpc_endpoint = f"tcp://{host}:{port}"
    else:
        sock_tmpdir = "/tmp" if sys.platform == "darwin" else tempfile.gettempdir()
        kernel.sock_path = os.path.join(sock_tmpdir, f"hermes_rpc_{uuid.uuid4().hex}.sock")
        server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server_sock.bind(kernel.sock_path)
        os.chmod(kernel.sock_path, 0o600)
        rpc_endpoint = kernel.sock_path
    server_sock.listen(1)
    kernel.server_sock = server_sock
    return rpc_endpoint


def _spawn(kernel: SessionKernel, *, child_python: str, child_cwd: str,
           sandbox_tools: frozenset, max_tool_calls: int) -> None:
    from tools.code_execution_tool import _build_child_env, generate_hermes_tools_module

    kernel.tmpdir = tempfile.mkdtemp(prefix="hermes_kernel_")
    kernel.rpc_token = secrets.token_urlsafe(32)
    kernel.sentinel = "@@HERMES-KERNEL-" + secrets.token_urlsafe(16) + "@@"
    rpc_endpoint = _bind_rpc_socket(kernel)

    for name, src in (("hermes_tools.py", generate_hermes_tools_module(list(sandbox_tools))),
                      ("hermes_kernel_runner.py", KERNEL_RUNNER_SOURCE)):
        with open(os.path.join(kernel.tmpdir, name), "w", encoding="utf-8") as f:
            f.write(src)

    child_env = _build_child_env(rpc_endpoint=rpc_endpoint, rpc_token=kernel.rpc_token,
                                 tmpdir=kernel.tmpdir, child_python=child_python)
    child_env["HERMES_KERNEL_SENTINEL"] = kernel.sentinel
    # Cells clip stdout to the inline cap; the full text spills to the
    # kernel's own tmpdir so the agent can read_file the middle instead of
    # re-running (host surfaces the path in the result).
    child_env["HERMES_KERNEL_SPILL_DIR"] = kernel.tmpdir
    # Tell the generated client to reconnect after the RPC server's idle
    # timeout — a kernel outlives the 300s window between cells.
    child_env["HERMES_RPC_PERSISTENT"] = "1"

    kernel.proc = subprocess.Popen(
        [child_python, os.path.join(kernel.tmpdir, "hermes_kernel_runner.py")],
        # Strict mode resolves an empty cwd: the kernel's own staging dir
        # then plays the per-call tmpdir's role.
        cwd=child_cwd or kernel.tmpdir, env=child_env,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE,
        start_new_session=True,
        creationflags=subprocess.CREATE_NO_WINDOW if _IS_WINDOWS else 0,
    )

    # Deliberately NOT propagate_context_to_thread: that would freeze the
    # spawning cell's context/callbacks into the server thread for the
    # kernel's whole life. Authority is rebound per cell via CellAuthority.
    for target, args in ((_rpc_forever, (kernel, max_tool_calls, sandbox_tools)),
                         (_stdout_reader, (kernel,)), (_stderr_reader, (kernel,))):
        threading.Thread(target=target, args=args, daemon=True).start()


def _acquire_kernel(key: Tuple, reset: bool) -> Tuple[SessionKernel, bool]:
    """Look up or register the kernel for *key*; returns (kernel, state_reset).

    Every entry also sweeps idle-expired kernels and enforces the
    process-wide cap, so a long-lived host stays bounded even for owners
    that never toggle or reset.
    """
    with _KERNELS_LOCK:
        expired = _reap_unlocked()
        kernel = _KERNELS.get(key)
        state_reset = kernel is not None and (reset or not kernel.alive())
        if state_reset:
            _KERNELS.pop(key, None)
            expired.append(kernel)
            kernel = None
        if kernel is None:
            kernel = SessionKernel(key)
            _KERNELS[key] = kernel
        kernel.last_used = time.monotonic()
        expired.extend(_evict_over_cap_unlocked(keep=key))
    for doomed in expired:
        _teardown(doomed)
    return kernel, state_reset


def _await_cell(kernel: SessionKernel, timeout: int, is_interrupted) -> Tuple[str, Dict[str, Any]]:
    """Wait for the cell's reply; returns (host status, payload)."""
    deadline = time.monotonic() + timeout if timeout else None
    while True:
        if is_interrupted():
            return "interrupted", {}
        if deadline is not None and time.monotonic() > deadline:
            return "timeout", {}
        try:
            payload = kernel.response_q.get(timeout=0.05)
        except queue.Empty:
            continue
        if payload.get("status") in ("kernel-eof", "protocol-error"):
            return "error", payload
        return "success", payload


def _with_stderr(stdout_text: str, stderr_text: str) -> str:
    return stdout_text + "\n--- stderr ---\n" + stderr_text


def _cell_result(kernel: SessionKernel, key: Tuple, status: str, payload: Dict[str, Any], *,
                 timeout: int, sandbox_tools: frozenset, reused: bool,
                 state_reset: bool, exec_start: float) -> Dict[str, Any]:
    """Assemble the tool result for one settled cell (disposing the kernel where the contract says so)."""
    from tools.code_execution_tool import _sandbox_failure_hint, _truncate_stdout_text
    from agent.redact import redact_sensitive_text
    from tools.ansi_strip import strip_ansi

    def clean(text: str) -> str:
        return redact_sensitive_text(strip_ansi(text), code_file=True)

    if status in ("timeout", "interrupted"):
        # No safe way to interrupt one cell in place: kill the kernel,
        # report the state loss, let the next call respawn.
        _REGISTRY.discard(key, kernel)

    duration = round(time.monotonic() - exec_start, 2)
    kernel.execution_count = int(payload.get("execution_count", kernel.execution_count + 1))

    stderr_raw = kernel.stderr.drain()
    stdout_text = clean(str(payload.get("stdout", "")) + kernel.raw.drain())
    cell_stderr = clean(str(payload.get("stderr", "")) + stderr_raw)
    stdout_text, stdout_metadata = _truncate_stdout_text(stdout_text)

    cell_status = payload.get("status", "")
    result: Dict[str, Any] = {
        "status": status, "output": stdout_text, "exit_code": 0,
        "tool_calls_made": kernel.tool_call_counter[0], "duration_seconds": duration,
        "kernel": {"mode": "session", "reused": reused,
                   "execution_count": kernel.execution_count, "state_reset": state_reset},
    }
    result.update(stdout_metadata)

    # Cell-side spill (runner clipped before replying): surface the full-output
    # path with the same read_file recipe as the host-side spill.
    cell_spill = str(payload.get("stdout_spill_path", "") or "")
    if cell_spill and payload.get("stdout_clipped"):
        result["stdout_spill_path"] = cell_spill
        result["warning"] = (
            f"Cell stdout exceeded the inline cap; head shown. FULL output saved to {cell_spill} "
            f'— page it with read_file(path="{cell_spill}", offset=...) instead of re-running. '
            "(Kernel state persists: printing a narrower slice next call is often cheaper.)"
        )

    if status == "timeout":
        message = (f"Cell timed out after {timeout}s; the session kernel was killed and its "
                   "state was lost. The next execute_code call starts a fresh kernel.")
        result.update(exit_code=-1, error=message,
                      output=(stdout_text + "\n\n⏰ " + message) if stdout_text else ("⏰ " + message))
    elif status == "interrupted":
        from tools.code_execution_tool import _format_interrupted_output

        result.update(exit_code=-1, output=_format_interrupted_output(stdout_text),
                      error="Interrupted; the session kernel was killed and its state was lost.")
    elif cell_status == "error":
        trace = clean(str(payload.get("traceback", "")))
        result.update(status="error", exit_code=1, error=trace or "Cell raised an exception.",
                      output=_with_stderr(stdout_text, cell_stderr + trace) if (cell_stderr or trace) else stdout_text)
        hint = _sandbox_failure_hint(trace, enabled_tools=sandbox_tools)
        if hint:
            result["hint"] = hint
    elif cell_status == "exit":
        # The cell called sys.exit(): honor it as end-of-kernel.
        _REGISTRY.discard(key, kernel)
        result["kernel"]["ended"] = True
        if cell_stderr:
            result["output"] = _with_stderr(stdout_text, cell_stderr)
    elif status == "error":
        _REGISTRY.discard(key, kernel)
        result.update(exit_code=-1, error="The session kernel died while running the cell"
                      + (": " + stderr_raw.strip() if stderr_raw.strip() else "."))
    elif cell_stderr:
        result["output"] = _with_stderr(stdout_text, cell_stderr)
    return result


def execute_in_session_kernel(
    code: str, *, task_id: str, mode: str, child_python: str, child_cwd: str,
    sandbox_tools: frozenset, timeout: int, max_tool_calls: int, reset: bool, is_interrupted,
) -> str:
    """Run one cell in the (owner, mode, python, cwd, tools) session kernel.

    The owner is the conversation's session key (``_resolve_owner``), not
    the per-turn task id, so state genuinely survives across user turns of
    one conversation and dies with the session.
    """
    key = _kernel_key(_resolve_owner(task_id), mode, child_python, child_cwd, sandbox_tools)
    exec_start = time.monotonic()
    kernel, state_reset = _acquire_kernel(key, reset)
    reused = kernel.proc is not None

    # Captured on the calling thread BEFORE the cell runs — the same
    # snapshot a per-call RPC thread would have received — and installed
    # atomically on the kernel so the serving thread dispatches this cell's
    # tool calls under this cell's approval/session/turn identity.
    authority = CellAuthority(task_id)

    with kernel.lock:
        try:
            if kernel.proc is None:
                _spawn(kernel, child_python=child_python, child_cwd=child_cwd,
                       sandbox_tools=sandbox_tools, max_tool_calls=max_tool_calls)
            assert kernel.proc is not None and kernel.proc.stdin is not None

            # Per-cell tool budget: the RPC loop enforces counter < max, so a
            # fresh cell starts from zero without restarting the server.
            kernel.tool_call_counter[0] = 0
            # Anything raw that leaked between cells belongs to no cell.
            kernel.raw.drain()
            kernel.stderr.drain()
            kernel.cell_authority = authority

            request = json.dumps({"id": uuid.uuid4().hex, "code": code}) + "\n"
            kernel.proc.stdin.write(request.encode("utf-8"))
            kernel.proc.stdin.flush()

            status, payload = _await_cell(kernel, timeout, is_interrupted)
            result = _cell_result(
                kernel, key, status, payload,
                timeout=timeout, sandbox_tools=sandbox_tools, reused=reused,
                state_reset=state_reset, exec_start=exec_start,
            )
            return json.dumps(result, ensure_ascii=False)
        except Exception as exc:  # pragma: no cover - defensive parity with per-call
            logger.error("session kernel failed: %s: %s", type(exc).__name__, exc, exc_info=True)
            _REGISTRY.discard(key, kernel)
            return json.dumps({
                "status": "error", "error": str(exc),
                "tool_calls_made": kernel.tool_call_counter[0],
                "duration_seconds": round(time.monotonic() - exec_start, 2),
            }, ensure_ascii=False)
        finally:
            # The cell has settled on every path (success, exception,
            # timeout, exit, kernel death): its tool authority retires with
            # it, so nothing the cell left running can dispatch under it.
            authority.retire()
