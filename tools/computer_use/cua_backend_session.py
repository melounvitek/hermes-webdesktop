"""cua-driver MCP session plumbing: the asyncio bridge thread and the lazily-started,
self-healing ``_CuaDriverSession`` (MCP transport with a ``cua-driver call`` CLI fallback).
Driver resolution / policy helpers are looked up lazily through
``tools.computer_use.cua_backend`` so tests that patch them there keep working.
"""

from __future__ import annotations

import asyncio
import base64
import concurrent.futures
import contextlib
import json
import logging
import os
import threading
from typing import Any, Dict, List, Optional, Tuple

from tools.computer_use.cua_backend_parse import _extract_tool_result, _mcp_field

logger = logging.getLogger("tools.computer_use.cua_backend")


class _AsyncBridge:
    """Runs one asyncio loop on a daemon thread; marshals coroutines from the caller."""

    def __init__(self) -> None:
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._ready = threading.Event()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._ready.clear()

        def _run() -> None:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._ready.set()
            try:
                self._loop.run_forever()
            finally:
                with contextlib.suppress(Exception):
                    self._loop.close()

        self._thread = threading.Thread(target=_run, daemon=True, name="cua-driver-loop")
        self._thread.start()
        if not self._ready.wait(timeout=5.0):
            raise RuntimeError("cua-driver asyncio bridge failed to start")

    def run(self, coro, timeout: Optional[float] = 30.0) -> Any:
        from agent.async_utils import safe_schedule_threadsafe
        if not self._loop or not self._thread or not self._thread.is_alive():
            if asyncio.iscoroutine(coro):
                coro.close()
            raise RuntimeError("cua-driver bridge not started")
        fut = safe_schedule_threadsafe(coro, self._loop)
        if fut is None:
            raise RuntimeError("cua-driver bridge not started")
        return fut.result(timeout=timeout)

    def stop(self) -> None:
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=2.0)
        self._thread = self._loop = None


# Fail-closed messages for calls whose effect on the remote screen is unknown. The action
# MAY have landed, so it is never replayed; the caller decides after taking fresh state.
_UNKNOWN_OUTCOME_MESSAGES = {
    "transport_outcome_unknown": (
        "cua-driver transport failed during {name}; the action outcome is unknown, so Hermes "
        "did not replay it. Take fresh state before deciding whether to act again."),
    "timeout_outcome_unknown": (
        "cua-driver MCP call {name} timed out; the action outcome is unknown and may still have "
        "taken effect on the remote screen. The session has been marked suspect and will be "
        "recreated before the next computer-use call. Take fresh state before deciding "
        "whether to act again."),
}


def _outcome_unknown(name: str, exc: Exception, code: str) -> Dict[str, Any]:
    """Fail-closed ``isError`` result for *code* (see ``_UNKNOWN_OUTCOME_MESSAGES``)."""
    message = _UNKNOWN_OUTCOME_MESSAGES[code].format(name=name)
    structured = {"ok": False, "code": code, "message": message, "operation": name,
                  "next_step": "fresh_state", "detail": str(exc)}
    return {"data": message, "images": [], "image_mime_types": [],
            "structuredContent": structured, "isError": True}


def _tool_field(obj: Any, *names: str) -> Any:
    """``_mcp_field`` plus the ``model_extra`` fallback some MCP SDKs (Pydantic v2) forward custom fields via."""
    value = _mcp_field(obj, names[0], names[-1])
    if value is None:
        value = (getattr(obj, "model_extra", None) or {}).get(names[-1])
    return value


_CLI_ATTEMPTS = 4  # CLI fallback transport retries (backoff 0.5s doubling)


def _cli_run_json(cmd: List[str], env: Dict[str, str], name: str, timeout: float) -> Any:
    """Run ``cua-driver call`` with backoff until it prints JSON; return the parsed value.
    "daemon is not running" is PERMANENT for this invocation (the CLI needs the machine-wide
    daemon socket, which Linux installs typically never start) -> fail fast, no ~3.5s backoff."""
    import subprocess as _subprocess
    import time as _time
    from tools.computer_use import cua_backend as _cb

    backoff = 0.5
    last_err = ""
    for attempt in range(_CLI_ATTEMPTS):
        try:
            proc = _subprocess.run(
                cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=max(15.0, timeout), creationflags=_cb.windows_hide_flags(), env=env)
        except Exception as e:  # pragma: no cover - subprocess spawn failure
            raise RuntimeError(f"cua-driver CLI fallback for {name} failed to spawn: {e}") from e

        out, err = (proc.stdout or "").strip(), proc.stderr or ""
        last_err = out[:200] or err[:200]
        if "daemon is not running" in out or "daemon is not running" in err:
            raise RuntimeError(
                f"cua-driver CLI fallback for {name} unavailable: the "
                "machine-wide cua-driver daemon is not running (the "
                "CLI transport requires it; the MCP runtime does not).")
        start = min((i for i in (out.find("{"), out.find("[")) if i != -1), default=-1)
        with contextlib.suppress(json.JSONDecodeError):
            if start != -1:
                return json.loads(out[start:])
        # No JSON (EAGAIN warning / empty) — retry with backoff.
        if attempt < _CLI_ATTEMPTS - 1:
            logger.warning("cua-driver CLI fallback for %s got no JSON (attempt %d/%d); "
                           "retrying in %.1fs", name, attempt + 1, _CLI_ATTEMPTS, backoff)
            _time.sleep(backoff)
            backoff *= 2
    raise RuntimeError(f"cua-driver CLI fallback for {name} returned no JSON after "
                       f"{_CLI_ATTEMPTS} attempts: {last_err}")


def _cli_result(parsed: Any, shot_file: Optional[str]) -> Dict[str, Any]:
    """Remap a ``cua-driver call`` JSON body into the ``_extract_tool_result`` shape."""
    if not isinstance(parsed, dict):
        return {"data": None, "images": [], "structuredContent": None, "isError": False}
    # In-band logical failures with exit 0 must still fail closed.
    is_error = parsed.get("isError") is True or parsed.get("is_error") is True
    shot = parsed.get("screenshot_png_b64")
    # Otherwise the screenshot was routed to a file (ours or the daemon's choice).
    fpath = parsed.get("screenshot_file_path") or shot_file
    if not shot and fpath and os.path.exists(fpath):
        try:
            with open(fpath, "rb") as fh:
                shot = base64.b64encode(fh.read()).decode("ascii")
        except Exception as e:
            logger.debug("cua-driver CLI fallback: failed reading %s: %s", fpath, e)
    data: Any = parsed.get("tree_markdown")
    if data is not None and parsed.get("element_count") is not None:
        data = f"{parsed['element_count']} elements\n{data}"
    return {"data": data, "images": [shot] if shot else [], "structuredContent": parsed, "isError": is_error}


class _CuaDriverSession:
    """Holds the mcp ClientSession. Spawned lazily; re-entered on drop.

    Lifecycle ownership: one long-running coroutine (`_lifecycle_coro`) opens the
    stdio_client + ClientSession contexts, populates capabilities, sets `_ready_event`,
    waits on `_shutdown_event`, then closes the contexts — enter and exit in the SAME
    task, as anyio's cancel-scope invariant requires (each `bridge.run(coro)` is a NEW
    task). Tool calls run in short-lived tasks touching only the session object.
    """

    # Handshake calls issued BY start()/stop() — exempt from call_tool's auto-restart
    # guard, or start() would recurse.
    _LIFECYCLE_CALLS = frozenset({"start_session", "end_session"})
    # Idempotent reads, safe to replay after a broken transport. Mutations stay out:
    # a lost response does not prove they failed.
    _TRANSPORT_REPLAY_SAFE_TOOLS = frozenset({
        "get_cursor_position", "get_displays", "get_screen_size",
        "get_window_state", "list_apps", "list_windows",
    })
    # A timed-out MCP session is wedged for later calls, so it is recreated before the
    # next non-lifecycle call_tool. Class-level default: tests that bypass __init__ see healthy.
    _timeout_suspect = False

    def __init__(self, bridge: _AsyncBridge, embedded_daemon: Optional[Any] = None) -> None:
        self._bridge, self._embedded_daemon = bridge, embedded_daemon
        self._session = None
        self._lock = threading.Lock()
        self._started = False
        # Per-tool capability-token sets from `tools/list` (read via supports_capability).
        self._capabilities: Dict[str, set] = {}
        # Raw input schemas are the source of truth for action properties: 0.9-era drivers
        # advertise delivery_mode in inputSchema without the ``input.delivery_mode`` token.
        self._tool_schemas: Dict[str, Dict[str, Any]] = {}
        self._capability_version: str = ""
        self._ready_event = threading.Event()
        self._shutdown_event: Optional[asyncio.Event] = None  # created on bridge loop
        self._lifecycle_future = None  # concurrent.futures.Future
        self._setup_error: Optional[BaseException] = None
        # Declared via start_session; revives an ended-session rejection non-re-entrantly.
        self._declared_session_id: Optional[str] = None
        self._transport_generation, self._transport_reset_callback = 0, None

    def _require_started(self) -> None:
        if not self._started:
            raise RuntimeError("cua-driver session not started")

    def _reset_capability_state(self) -> None:
        self._capabilities, self._tool_schemas, self._capability_version = {}, {}, ""

    async def _lifecycle_coro(self) -> None:
        """Owns the stdio MCP contexts: open, signal ready, block on shutdown, clean up —
        all in one task (see class docstring)."""
        import time as _time
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
        from tools.computer_use import cua_backend as _cb
        from tools.environments.local import _sanitize_subprocess_env

        self._shutdown_event = asyncio.Event()  # built on the loop's own thread
        _t0 = _time.monotonic()
        # Phase marker: the ready-timeout error reports HOW FAR a wedged startup got.
        self._startup_phase = "binary-check"
        try:
            driver_cmd = _cb.resolve_cua_driver_cmd()
            if not driver_cmd:
                raise RuntimeError(_cb.cua_driver_install_hint())
            self._startup_phase = "manifest-discovery"
            daemon = self._embedded_daemon
            if daemon is not None:
                (command, args), child_env = daemon.proxy_invocation(), daemon.child_env()
            else:
                (command, args), child_env = _cb._resolve_mcp_invocation(driver_cmd), _cb.cua_driver_child_env()
            _t_manifest = _time.monotonic()
            # Telemetry policy first (default: disabled), then strip Hermes secrets.
            params = StdioServerParameters(command=command, args=args,
                                           env=_sanitize_subprocess_env(child_env))
            async with stdio_client(params) as (read, write):
                self._startup_phase = "mcp-initialize"
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    _t_init = _time.monotonic()
                    # Capabilities BEFORE exposing the session: the first call sees them.
                    self._startup_phase = "capability-discovery"
                    await self._populate_capabilities(session)
                    self._session = session
                    self._startup_phase = "ready"
                    self._ready_event.set()
                    logger.info("cua-driver session ready in %.1fs (manifest=%.1fs, mcp_init=%.1fs)",
                                _time.monotonic() - _t0, _t_manifest - _t0, _t_init - _t_manifest)
                    await self._shutdown_event.wait()
        except BaseException as e:
            # Ordinary errors and anyio CancelledError alike: start() surfaces this.
            self._setup_error = e
            self._ready_event.set()
            raise
        finally:
            self._session = None
            # A session that dies for ANY reason must be re-enterable: the next call
            # sees _started False and rebuilds. Atomic bool write — stop() may hold _lock.
            self._started = False

    async def _populate_capabilities(self, session: Any) -> None:
        """Cache per-tool capability sets, input schemas and capability_version from
        tools/list. Soft prerequisite: on failure the map stays empty (capability False)."""
        self._reset_capability_state()
        try:
            tools_list = await session.list_tools()
            for tool in getattr(tools_list, "tools", []) or []:
                tool_name = getattr(tool, "name", None)
                if not isinstance(tool_name, str):
                    continue
                caps = _tool_field(tool, "capabilities")
                self._capabilities[tool_name] = (
                    {c for c in caps if isinstance(c, str)} if isinstance(caps, list) else set())
                schema = _tool_field(tool, "input_schema", "inputSchema")
                self._tool_schemas[tool_name] = dict(schema) if isinstance(schema, dict) else {}
            # capability_version is a sibling of `tools` in tools/list (NOT in initialize).
            cv = _tool_field(tools_list, "capability_version")
            if isinstance(cv, str):
                self._capability_version = cv
        except Exception as e:
            logger.debug("cua-driver tools/list capability discovery failed: %s", e)

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            self._bridge.start()
            self._start_lifecycle_locked()
            self._started = True

    def _start_lifecycle_locked(self) -> None:
        """Spawn the lifecycle owner and wait for ready. Caller holds self._lock."""
        self._ready_event = threading.Event()
        self._setup_error = self._shutdown_event = None
        # The future tracks the WHOLE lifecycle; readiness is signalled via _ready_event.
        loop = self._bridge._loop
        if loop is None:
            raise RuntimeError("cua-driver bridge not started")
        self._lifecycle_future = asyncio.run_coroutine_threadsafe(self._lifecycle_coro(), loop)
        if not self._ready_event.wait(timeout=30.0):
            self._signal_shutdown_locked()
            phase = getattr(self, "_startup_phase", "unknown")
            from hermes_constants import display_hermes_home
            raise RuntimeError(
                f"cua-driver session never reached ready (timeout 30s; stuck in phase: {phase}). "
                "Run `hermes computer-use doctor` and check "
                f"{display_hermes_home()}/logs/agent.log for the phase timings.")
        if self._setup_error is not None:
            raise RuntimeError(f"cua-driver session setup failed: {self._setup_error}") from self._setup_error
        self._transport_generation += 1
        if self._transport_generation > 1:
            self._notify_transport_reset()

    def stop(self) -> None:
        with self._lock:
            if not self._started:
                return
            self._started = False
            self._stop_lifecycle_locked()

    def set_transport_reset_callback(self, callback: Any) -> None:
        """Register a synchronous cache invalidation hook for transport swaps."""
        self._transport_reset_callback = callback

    def _notify_transport_reset(self) -> None:
        callback = getattr(self, "_transport_reset_callback", None)
        try:
            if callback is not None:
                callback()
        except Exception as exc:
            logger.debug("cua-driver transport reset callback failed: %s", exc)

    def _stop_lifecycle_locked(self) -> None:
        self._signal_shutdown_locked()
        fut, self._lifecycle_future = self._lifecycle_future, None
        try:
            if fut is not None:
                fut.result(timeout=5.0)
        except concurrent.futures.TimeoutError:
            logger.warning("cua-driver session shutdown timed out (5s)")
        except Exception as e:
            logger.warning("cua-driver shutdown error: %s", e)

    def _signal_shutdown_locked(self) -> None:
        """Set the asyncio shutdown event from the caller's thread."""
        loop, event = self._bridge._loop, self._shutdown_event
        if loop is not None and event is not None and loop.is_running():
            with contextlib.suppress(RuntimeError):  # loop closed — nothing to signal
                loop.call_soon_threadsafe(event.set)

    async def _call_tool_async(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        return _extract_tool_result(await self._session.call_tool(name, args))

    def _run_call(self, name: str, args: Dict[str, Any], timeout: float) -> Dict[str, Any]:
        return self._bridge.run(self._call_tool_async(name, args), timeout=timeout)

    # ── Capability detection ─────────────────────────────────────────
    def supports_capability(self, capability: str, tool: Optional[str] = None) -> bool:
        """Driver advertises *capability* for *tool* (or ANY tool). False before start."""
        caps = [self._capabilities.get(tool, set())] if tool is not None else self._capabilities.values()
        return any(capability in c for c in caps)

    def _has_tool(self, name: str) -> bool:
        """``tools/list`` advertised *name*. Routes capture() (PNG capture moved into
        ``get_window_state``). False before discovery — callers treat that as "unknown"."""
        return name in self._capabilities

    def supports_input_property(self, tool: str, property_name: str) -> bool:
        """Live tools/list schema accepts *property_name* (fails closed; no version guessing)."""
        schema = getattr(self, "_tool_schemas", {}).get(tool, {})
        properties = schema.get("properties") if isinstance(schema, dict) else None
        return isinstance(properties, dict) and property_name in properties

    @property
    def capabilities_discovered(self) -> bool:
        """tools/list populated the map; when False ``_has_tool`` is untrustworthy."""
        return bool(self._capabilities)

    @property
    def capability_version(self) -> str:
        """Driver-advertised capability vocabulary version ("" on old builds)."""
        return self._capability_version

    # ── Error classification ─────────────────────────────────────────
    @staticmethod
    def _logical_error_text(result: Dict[str, Any]) -> str:
        """Flatten a logical MCP error into text for narrow classification."""
        chunks: List[str] = []
        for value in (result.get("data"), result.get("structuredContent")):
            if value is None:
                continue
            try:
                chunks.append(value if isinstance(value, str) else json.dumps(value, sort_keys=True))
            except (TypeError, ValueError):
                chunks.append(str(value))
        return "\n".join(chunks)

    @classmethod
    def _is_ended_session_result(cls, result: Any) -> bool:
        """Recognise cua-driver's explicit recoverable ended-session result."""
        if not isinstance(result, dict) or result.get("isError") is not True:
            return False
        message = cls._logical_error_text(result).lower()
        return ("session" in message and "start_session" in message
                and ("has ended" in message or "session ended" in message))

    @staticmethod
    def _is_closed_session_error(exc: Exception) -> bool:
        """True for MCP/stdio failures that are recoverable by reconnecting."""
        name, module = exc.__class__.__name__, getattr(exc.__class__, "__module__", "")
        return (name in {"ClosedResourceError", "BrokenResourceError", "EndOfStream"}
                or (module.startswith("anyio") and "Resource" in name)
                or isinstance(exc, (BrokenPipeError, EOFError)))

    @staticmethod
    def _is_transient_daemon_error(exc: Exception) -> bool:
        """Daemon-proxy EAGAIN congestion: on macOS the ``cua-driver mcp`` bridge uses a
        non-blocking unix socket and heavy ops (``get_window_state``) fail with ``os error
        35`` when its buffer is full. A retry succeeds, so back off / fall back instead
        of surfacing an empty 0x0 capture."""
        msg = str(exc)
        return any(needle in msg for needle in ("Resource temporarily unavailable", "os error 35",
                                                "daemon transport error", "daemon proxy"))

    # ── Recovery ─────────────────────────────────────────────────────
    def _revive_declared_session_once(self, name: str, args: Dict[str, Any],
                                      first_result: Dict[str, Any], timeout: float) -> Dict[str, Any]:
        """Revive the stable session, replay the rejected call once; a 2nd rejection surfaces as-is."""
        session_id = self._declared_session_id
        if not session_id or name in self._LIFECYCLE_CALLS:
            return first_result
        logger.warning("cua-driver session %s ended during %s; reviving and retrying once", session_id, name)
        if not self._redeclare_session(timeout, "cua-driver session %s could not be revived: %s"):
            return first_result
        return self._run_call(name, args, timeout)

    def _redeclare_session(self, timeout: float, failure_msg: str) -> bool:
        """start_session with the declared id; log *failure_msg* and return False on rejection."""
        session_id = self._declared_session_id
        result = self._run_call("start_session", {"session": session_id}, timeout)
        if result.get("isError") is True:
            logger.warning(failure_msg, session_id, self._logical_error_text(result))
            return False
        return True

    def _restore_declared_session_after_transport_reset(self, timeout: float) -> None:
        """Re-attach the public label inside a replacement private lifecycle."""
        if getattr(self, "_declared_session_id", None):
            self._redeclare_session(timeout, "cua-driver public session label %s could not be restored: %s")

    def _restart_session_locked(self) -> None:
        """Recreate the MCP session after the transport closed. Caller holds self._lock."""
        try:
            if self._started:
                self._stop_lifecycle_locked()
        except Exception as e:
            logger.debug("cua-driver session cleanup before reconnect failed: %s", e)
        self._started = False
        self._reset_capability_state()  # repopulated from scratch by the next start
        self._start_lifecycle_locked()
        self._started = True

    def _recreate_session(self, timeout: float, *, clear_timeout_suspect: bool = False) -> None:
        """Restart the private lifecycle, then re-attach the declared public label."""
        with self._lock:
            self._restart_session_locked()
        if clear_timeout_suspect:
            self._timeout_suspect = False
        self._restore_declared_session_after_transport_reset(timeout)

    def _cli_command(self, name: str, args: Dict[str, Any]) -> Tuple[List[str], Dict[str, str], Optional[str]]:
        """Build ``(cmd, child_env, shot_file)`` for the CLI fallback. ``get_window_state`` routes
        its screenshot to a temp file (``screenshot_out_file``) so the daemon returns a tiny JSON
        body, not the multi-megabyte base64 blob that congests the socket; ``_cli_result`` reads it."""
        import tempfile as _tempfile
        from tools.computer_use import cua_backend as _cb

        call_args = dict(args)
        shot_file: Optional[str] = None
        if name == "get_window_state" and "screenshot_out_file" not in call_args:
            fd, shot_file = _tempfile.mkstemp(prefix="cua_shot_", suffix=".png")
            os.close(fd)
            call_args["screenshot_out_file"] = shot_file
        driver_command = _cb.resolve_cua_driver_cmd()
        if not driver_command:
            raise RuntimeError(_cb.cua_driver_install_hint())
        child_env, socket_args = _cb.cua_driver_child_env(), []
        daemon = getattr(self, "_embedded_daemon", None)
        if daemon is not None:
            driver_command, child_env = daemon.proxy_invocation()[0], daemon.child_env()
            socket_args = ["--socket", daemon.socket_path]
        return [driver_command, "call", name, json.dumps(call_args), *socket_args], child_env, shot_file

    def _call_tool_via_cli(self, name: str, args: Dict[str, Any], timeout: float) -> Dict[str, Any]:
        """Fallback transport: ``cua-driver call <tool> <json>`` subprocess. The MCP stdio bridge
        can persistently fail heavy calls (``get_window_state``) with EAGAIN while the plain CLI,
        on its own daemon socket, keeps working. Output is remapped to the ``_extract_tool_result`` shape."""
        from tools.environments.local import _sanitize_subprocess_env

        cmd, child_env, shot_file = self._cli_command(name, args)
        try:
            parsed = _cli_run_json(cmd, _sanitize_subprocess_env(child_env), name, timeout)
            return _cli_result(parsed, shot_file)
        finally:
            if shot_file and os.path.exists(shot_file):
                with contextlib.suppress(OSError):
                    os.remove(shot_file)

    def call_tool(self, name: str, args: Dict[str, Any], timeout: float = 30.0) -> Dict[str, Any]:
        # A prior MCP timeout marks the session suspect (possibly wedged): recreate it
        # so one timeout never poisons the run. Healthy sessions are never restarted here.
        if self._timeout_suspect and name not in self._LIFECYCLE_CALLS:
            logger.warning("cua-driver session suspect after earlier MCP timeout; recreating before %s", name)
            self._recreate_session(timeout, clear_timeout_suspect=True)
        # A prior session may have died (MCP drop / driver crash) and reset _started.
        if not self._started and name not in self._LIFECYCLE_CALLS:
            logger.warning("cua-driver session not active on %s; (re)starting before call", name)
            self.start()
            self._restore_declared_session_after_transport_reset(timeout)
        self._require_started()
        try:
            result = self._run_call(name, args, timeout)
        except Exception as e:
            if isinstance(e, concurrent.futures.TimeoutError):
                # Fail closed: the action may have landed, so never replay it.
                self._timeout_suspect = True
                logger.warning("cua-driver MCP timed out on %s; marking session suspect "
                               "for recreation before the next call", name)
                return _outcome_unknown(name, e, "timeout_outcome_unknown")
            if self._is_transient_daemon_error(e):
                if name not in self._TRANSPORT_REPLAY_SAFE_TOOLS:
                    self._notify_transport_reset()
                    return _outcome_unknown(name, e, "transport_outcome_unknown")
                logger.warning("cua-driver MCP transport failed on %s (%s); "
                               "falling back to CLI transport", name, e)
                return self._call_tool_via_cli(name, args, timeout)
            if not self._is_closed_session_error(e):
                raise
            logger.warning("cua-driver MCP session closed during %s; reconnecting once", name)
            self._recreate_session(timeout)
            if name not in self._TRANSPORT_REPLAY_SAFE_TOOLS:
                return _outcome_unknown(name, e, "transport_outcome_unknown")
            result = self._run_call(name, args, timeout)
        # Remember only a SUCCESSFULLY declared identity: no stale recovery state.
        declared_id = args.get("session")
        if (name == "start_session" and result.get("isError") is not True
                and isinstance(declared_id, str) and declared_id):
            self._declared_session_id = declared_id
        if self._is_ended_session_result(result):
            result = self._revive_declared_session_once(name, args, result, timeout)
        if (name == "end_session" and result.get("isError") is not True
                and args.get("session") == self._declared_session_id):
            self._declared_session_id = None
        return result
