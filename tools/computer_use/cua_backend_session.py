"""cua-driver MCP session plumbing: the asyncio bridge thread and the
lazily-started, self-healing ``_CuaDriverSession`` (MCP transport with a
``cua-driver call`` CLI fallback).

Driver resolution / policy helpers are looked up lazily through
``tools.computer_use.cua_backend`` so tests that patch them there keep working.
"""

from __future__ import annotations

import asyncio
import base64
import concurrent.futures
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
                try:
                    self._loop.close()
                except Exception:
                    pass

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
        self._thread = None
        self._loop = None


def _outcome_unknown(name: str, exc: Exception, code: str, message: str) -> Dict[str, Any]:
    """Fail-closed result for a call whose effect on the remote screen is unknown."""
    return {
        "data": message,
        "images": [],
        "image_mime_types": [],
        "structuredContent": {
            "ok": False,
            "code": code,
            "message": message,
            "operation": name,
            "next_step": "fresh_state",
            "detail": str(exc),
        },
        "isError": True,
    }


# ── CLI fallback transport helpers ───────────────────────────────────
_CLI_ATTEMPTS = 4


def _cli_run_json(cmd: List[str], env: Dict[str, str], name: str, timeout: float) -> Any:
    """Run ``cua-driver call`` with backoff until it prints JSON; return the parsed value.

    Fails fast on "daemon is not running": that is PERMANENT for this
    invocation (the CLI needs the machine-wide daemon socket, which Linux
    installs typically never start), so burning ~3.5s of backoff is pointless.
    """
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

        out = (proc.stdout or "").strip()
        err = proc.stderr or ""
        last_err = out[:200] or err[:200]
        if "daemon is not running" in out or "daemon is not running" in err:
            raise RuntimeError(
                f"cua-driver CLI fallback for {name} unavailable: the "
                "machine-wide cua-driver daemon is not running (the "
                "CLI transport requires it; the MCP runtime does not)."
            )
        start = min((i for i in (out.find("{"), out.find("[")) if i != -1), default=-1)
        if start != -1:
            try:
                return json.loads(out[start:])
            except json.JSONDecodeError:
                pass
        # No JSON (EAGAIN warning / empty) — retry with backoff.
        if attempt < _CLI_ATTEMPTS - 1:
            logger.warning(
                "cua-driver CLI fallback for %s got no JSON "
                "(attempt %d/%d); retrying in %.1fs",
                name, attempt + 1, _CLI_ATTEMPTS, backoff,
            )
            _time.sleep(backoff)
            backoff *= 2
    raise RuntimeError(f"cua-driver CLI fallback for {name} returned no JSON after "
                       f"{_CLI_ATTEMPTS} attempts: {last_err}")


def _cli_result(parsed: Any, shot_file: Optional[str]) -> Dict[str, Any]:
    """Remap a ``cua-driver call`` JSON body into the ``_extract_tool_result`` shape."""
    images: List[str] = []
    data: Any = None
    is_error = False
    if isinstance(parsed, dict):
        # Logical failures may be reported in-band even when the subprocess
        # exits 0 — preserve the bit so callers fail closed.
        is_error = parsed.get("isError") is True or parsed.get("is_error") is True
        shot = parsed.get("screenshot_png_b64")
        if not shot:
            # Screenshot was routed to a file (ours or the daemon's choice).
            fpath = parsed.get("screenshot_file_path") or shot_file
            if fpath and os.path.exists(fpath):
                try:
                    with open(fpath, "rb") as fh:
                        shot = base64.b64encode(fh.read()).decode("ascii")
                except Exception as e:
                    logger.debug("cua-driver CLI fallback: failed reading %s: %s", fpath, e)
        if shot:
            images.append(shot)
        tree = parsed.get("tree_markdown")
        if tree is not None:
            ec = parsed.get("element_count")
            summary = f"{ec} elements" if ec is not None else ""
            data = f"{summary}\n{tree}" if summary else tree
    structured = parsed if isinstance(parsed, dict) else None
    return {"data": data, "images": images, "structuredContent": structured, "isError": is_error}


class _CuaDriverSession:
    """Holds the mcp ClientSession. Spawned lazily; re-entered on drop.

    Lifecycle ownership: one long-running coroutine (`_lifecycle_coro`) opens
    the stdio_client and ClientSession contexts, populates capabilities, sets
    `_ready_event`, then waits on `_shutdown_event` and closes the contexts —
    enter and exit in the SAME task, which anyio's cancel-scope invariant
    requires (the bridge schedules each `bridge.run(coro)` as a NEW task).
    Tool calls run in their own short-lived tasks and only touch the session
    object, never the surrounding contexts.
    """

    # Handshake calls issued BY start()/stop() themselves — must not trigger
    # the auto-restart guard in call_tool, or start() would recurse.
    _LIFECYCLE_CALLS = frozenset({"start_session", "end_session"})

    # Safe to replay after a broken transport: no side effect or idempotent.
    # Mutations stay out — a lost response does not prove they failed.
    _TRANSPORT_REPLAY_SAFE_TOOLS = frozenset({
        "get_cursor_position",
        "get_displays",
        "get_screen_size",
        "get_window_state",
        "list_apps",
        "list_windows",
    })

    # Set when an MCP call timed out: a timed-out session is wedged for later
    # calls, so it is recreated before the next non-lifecycle call_tool.
    # Class-level default so tests that bypass __init__ see a healthy session.
    _timeout_suspect = False

    def __init__(self, bridge: _AsyncBridge, embedded_daemon: Optional[Any] = None) -> None:
        self._bridge = bridge
        self._embedded_daemon = embedded_daemon
        self._session = None
        self._lock = threading.Lock()
        self._started = False
        # Per-tool capability-token sets from `tools/list`; empty until the
        # session starts. Consumers call `supports_capability`, not this map.
        self._capabilities: Dict[str, set] = {}
        # Raw input schemas are the source of truth for action properties:
        # 0.9-era drivers advertise delivery_mode in inputSchema while
        # omitting the old fabricated ``input.delivery_mode`` capability token.
        self._tool_schemas: Dict[str, Dict[str, Any]] = {}
        self._capability_version: str = ""
        self._ready_event = threading.Event()
        self._shutdown_event: Optional[asyncio.Event] = None  # created on bridge loop
        self._lifecycle_future = None  # concurrent.futures.Future
        self._setup_error: Optional[BaseException] = None
        # Stable driver-side identity declared through start_session; used to
        # revive a logical ended-session rejection without re-entrant call_tool.
        self._declared_session_id: Optional[str] = None
        self._transport_generation = 0
        self._transport_reset_callback: Optional[Any] = None

    def _require_started(self) -> None:
        if not self._started:
            raise RuntimeError("cua-driver session not started")

    async def _lifecycle_coro(self) -> None:
        """Long-lived owner of the stdio MCP contexts: open, signal ready,
        block on shutdown, clean up — all in one task (see class docstring)."""
        import time as _time
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
        from tools.computer_use import cua_backend as _cb
        from tools.environments.local import _sanitize_subprocess_env

        # Built on the loop's thread so the primitive belongs to this loop.
        self._shutdown_event = asyncio.Event()
        _t0 = _time.monotonic()
        # Phase marker surfaced by the ready-timeout error so a wedged startup
        # reports HOW FAR it got instead of an opaque "never reached ready".
        self._startup_phase = "binary-check"

        try:
            driver_cmd = _cb.resolve_cua_driver_cmd()
            if not driver_cmd:
                raise RuntimeError(_cb.cua_driver_install_hint())

            self._startup_phase = "manifest-discovery"
            if self._embedded_daemon is not None:
                command, args = self._embedded_daemon.proxy_invocation()
                child_env = self._embedded_daemon.child_env()
            else:
                command, args = _cb._resolve_mcp_invocation(driver_cmd)
                child_env = _cb.cua_driver_child_env()
            _t_manifest = _time.monotonic()
            # Telemetry policy first (default: disabled), then strip Hermes secrets.
            params = StdioServerParameters(command=command, args=args,
                                           env=_sanitize_subprocess_env(child_env))

            async with stdio_client(params) as (read, write):
                self._startup_phase = "mcp-initialize"
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    _t_init = _time.monotonic()
                    # Populate capabilities BEFORE exposing the session so
                    # the first tool call already sees them.
                    self._startup_phase = "capability-discovery"
                    await self._populate_capabilities(session)
                    self._session = session
                    self._startup_phase = "ready"
                    self._ready_event.set()
                    logger.info("cua-driver session ready in %.1fs (manifest=%.1fs, mcp_init=%.1fs)",
                                _time.monotonic() - _t0, _t_manifest - _t0, _t_init - _t_manifest)
                    await self._shutdown_event.wait()
        except BaseException as e:
            # Ordinary errors and anyio CancelledError alike: start()
            # inspects this to surface setup failures synchronously.
            self._setup_error = e
            self._ready_event.set()
            raise
        finally:
            self._session = None
            # A session that dies for ANY reason (MCP drop, driver crash,
            # unexpected exit) must be re-enterable: the next call sees
            # _started False and rebuilds instead of hanging on a dead one.
            # Plain bool write is atomic — stop() may hold self._lock here.
            self._started = False

    async def _populate_capabilities(self, session: Any) -> None:
        """Cache per-tool capability sets, input schemas and capability_version
        from tools/list. Soft prerequisite — on failure the map stays empty and
        supports_capability degrades to False."""
        self._capabilities = {}
        self._tool_schemas = {}
        self._capability_version = ""

        def _field(obj: Any, *names: str) -> Any:
            # Some MCP SDKs forward custom fields via `model_extra` (Pydantic v2).
            value = _mcp_field(obj, names[0], names[-1])
            if value is None:
                value = (getattr(obj, "model_extra", None) or {}).get(names[-1])
            return value

        try:
            tools_list = await session.list_tools()
            for tool in getattr(tools_list, "tools", []) or []:
                tool_name = getattr(tool, "name", None)
                if not isinstance(tool_name, str):
                    continue
                caps = _field(tool, "capabilities")
                self._capabilities[tool_name] = (
                    {c for c in caps if isinstance(c, str)} if isinstance(caps, list) else set()
                )
                schema = _field(tool, "input_schema", "inputSchema")
                self._tool_schemas[tool_name] = dict(schema) if isinstance(schema, dict) else {}
            # capability_version is a top-level sibling of `tools` on the
            # tools/list response (cua-driver leaves it OUT of initialize).
            cv = _field(tools_list, "capability_version")
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
        self._setup_error = None
        self._shutdown_event = None
        # The future tracks the WHOLE lifecycle (open -> wait -> close);
        # readiness is signalled separately via _ready_event.
        loop = self._bridge._loop
        if loop is None:
            raise RuntimeError("cua-driver bridge not started")
        self._lifecycle_future = asyncio.run_coroutine_threadsafe(self._lifecycle_coro(), loop)
        if not self._ready_event.wait(timeout=30.0):
            self._signal_shutdown_locked()
            phase = getattr(self, "_startup_phase", "unknown")
            from hermes_constants import display_hermes_home
            raise RuntimeError(
                "cua-driver session never reached ready (timeout 30s; "
                f"stuck in phase: {phase}). "
                "Run `hermes computer-use doctor` and check "
                f"{display_hermes_home()}/logs/agent.log for the phase timings."
            )
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
        if callback is None:
            return
        try:
            callback()
        except Exception as exc:
            logger.debug("cua-driver transport reset callback failed: %s", exc)

    def _stop_lifecycle_locked(self) -> None:
        """Signal shutdown and wait (5s) for the lifecycle coroutine to unwind."""
        self._signal_shutdown_locked()
        fut = self._lifecycle_future
        if fut is None:
            return
        try:
            fut.result(timeout=5.0)
        except concurrent.futures.TimeoutError:
            logger.warning("cua-driver session shutdown timed out (5s)")
        except Exception as e:
            logger.warning("cua-driver shutdown error: %s", e)
        finally:
            self._lifecycle_future = None

    def _signal_shutdown_locked(self) -> None:
        """Set the asyncio shutdown event from the caller's thread."""
        loop = self._bridge._loop
        event = self._shutdown_event
        if loop is not None and event is not None and loop.is_running():
            try:
                loop.call_soon_threadsafe(event.set)
            except RuntimeError:  # loop closed — nothing to signal
                pass

    async def _call_tool_async(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        result = await self._session.call_tool(name, args)
        return _extract_tool_result(result)

    def _run_call(self, name: str, args: Dict[str, Any], timeout: float) -> Dict[str, Any]:
        return self._bridge.run(self._call_tool_async(name, args), timeout=timeout)

    # ── Capability detection ─────────────────────────────────────────
    def supports_capability(self, capability: str, tool: Optional[str] = None) -> bool:
        """True when the driver advertises *capability* — for *tool* when given,
        otherwise on ANY tool. Always False before the session started."""
        if tool is not None:
            return capability in self._capabilities.get(tool, set())
        return any(capability in caps for caps in self._capabilities.values())

    def _has_tool(self, name: str) -> bool:
        """True when ``tools/list`` advertised *name*. Routes capture(): cua-driver
        folded PNG capture into ``get_window_state`` and dropped ``screenshot``.
        False before discovery populated the map — callers treat that as "unknown"."""
        return name in self._capabilities

    def supports_input_property(self, tool: str, property_name: str) -> bool:
        """Whether the live action schema accepts *property_name* (fails closed).
        Inspects tools/list rather than guessing from the package version."""
        schema = getattr(self, "_tool_schemas", {}).get(tool, {})
        properties = schema.get("properties") if isinstance(schema, dict) else None
        return isinstance(properties, dict) and property_name in properties

    @property
    def capabilities_discovered(self) -> bool:
        """True once tools/list populated the map; when False, ``_has_tool``
        answers are untrustworthy and capture() should probe defensively."""
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
            if isinstance(value, str):
                chunks.append(value)
            elif value is not None:
                try:
                    chunks.append(json.dumps(value, sort_keys=True))
                except (TypeError, ValueError):
                    chunks.append(str(value))
        return "\n".join(chunks)

    @classmethod
    def _is_ended_session_result(cls, result: Any) -> bool:
        """Recognise cua-driver's explicit recoverable ended-session result."""
        if not isinstance(result, dict) or result.get("isError") is not True:
            return False
        message = cls._logical_error_text(result).lower()
        return (
            "session" in message
            and ("has ended" in message or "session ended" in message)
            and "start_session" in message
        )

    @staticmethod
    def _is_closed_session_error(exc: Exception) -> bool:
        """True for MCP/stdio failures that are recoverable by reconnecting."""
        name = exc.__class__.__name__
        module = getattr(exc.__class__, "__module__", "")
        return (
            name in {"ClosedResourceError", "BrokenResourceError", "EndOfStream"}
            or (module.startswith("anyio") and "Resource" in name)
            or isinstance(exc, (BrokenPipeError, EOFError))
        )

    @staticmethod
    def _is_transient_daemon_error(exc: Exception) -> bool:
        """True for the daemon-proxy EAGAIN congestion error: on macOS the
        ``cua-driver mcp`` bridge talks to the daemon over a non-blocking unix
        socket and heavy ops (``get_window_state``) fail with ``os error 35``
        when the buffer is full. A retry succeeds, so back off / fall back
        instead of surfacing an empty 0x0 capture."""
        msg = str(exc)
        return (
            "Resource temporarily unavailable" in msg
            or "os error 35" in msg
            or "daemon transport error" in msg
            or "daemon proxy" in msg
        )

    @classmethod
    def _transport_replay_is_safe(cls, name: str) -> bool:
        return name in cls._TRANSPORT_REPLAY_SAFE_TOOLS

    @staticmethod
    def _unknown_transport_outcome(name: str, exc: Exception) -> Dict[str, Any]:
        return _outcome_unknown(
            name, exc, "transport_outcome_unknown",
            f"cua-driver transport failed during {name}; the action outcome is "
            "unknown, so Hermes did not replay it. Take fresh state before "
            "deciding whether to act again.",
        )

    @staticmethod
    def _timeout_outcome(name: str, exc: Exception) -> Dict[str, Any]:
        """Fail-closed result for an MCP call that hit its deadline. The action
        MAY have landed, so it is never replayed here; the caller decides after
        taking fresh state."""
        return _outcome_unknown(
            name, exc, "timeout_outcome_unknown",
            f"cua-driver MCP call {name} timed out; the action outcome is "
            "unknown and may still have taken effect on the remote screen. "
            "The session has been marked suspect and will be recreated before "
            "the next computer-use call. Take fresh state before deciding "
            "whether to act again.",
        )

    # ── Recovery ─────────────────────────────────────────────────────
    def _revive_declared_session_once(
        self,
        name: str,
        args: Dict[str, Any],
        first_result: Dict[str, Any],
        timeout: float,
    ) -> Dict[str, Any]:
        """Revive the stable session and replay one rejected tool call once.
        A second rejection is surfaced as-is; no loop."""
        session_id = self._declared_session_id
        if not session_id or name in self._LIFECYCLE_CALLS:
            return first_result
        logger.warning("cua-driver session %s ended during %s; reviving and retrying once",
                       session_id, name)
        revive_result = self._run_call("start_session", {"session": session_id}, timeout)
        if revive_result.get("isError") is True:
            logger.warning("cua-driver session %s could not be revived: %s",
                           session_id, self._logical_error_text(revive_result))
            return first_result
        return self._run_call(name, args, timeout)

    def _restore_declared_session_after_transport_reset(self, timeout: float) -> None:
        """Re-attach the public label inside a replacement private lifecycle."""
        session_id = getattr(self, "_declared_session_id", None)
        if not session_id:
            return
        result = self._run_call("start_session", {"session": session_id}, timeout)
        if result.get("isError") is True:
            logger.warning("cua-driver public session label %s could not be restored: %s",
                           session_id, self._logical_error_text(result))

    def _restart_session_locked(self) -> None:
        """Recreate the MCP session after the transport closed. Caller holds self._lock."""
        if self._started:
            try:
                self._stop_lifecycle_locked()
            except Exception as e:
                logger.debug("cua-driver session cleanup before reconnect failed: %s", e)
        self._started = False
        # Stale capability state is repopulated from scratch by the next start.
        self._capabilities = {}
        self._tool_schemas = {}
        self._capability_version = ""
        self._start_lifecycle_locked()
        self._started = True

    def _restart_and_restore(self, timeout: float) -> None:
        with self._lock:
            self._restart_session_locked()
        self._restore_declared_session_after_transport_reset(timeout)

    def _cli_command(self, name: str, args: Dict[str, Any]) -> Tuple[List[str], Dict[str, str], Optional[str]]:
        """Build ``(cmd, child_env, shot_file)`` for the CLI fallback.

        For ``get_window_state`` the screenshot is routed to a temp file via
        ``screenshot_out_file`` so the daemon returns a tiny JSON body instead of
        a multi-megabyte base64 blob (the payload that congests the socket in
        the first place); ``_cli_result`` reads the PNG back.
        """
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
        child_env = _cb.cua_driver_child_env()
        socket_args: List[str] = []
        embedded_daemon = getattr(self, "_embedded_daemon", None)
        if embedded_daemon is not None:
            driver_command = embedded_daemon.proxy_invocation()[0]
            child_env = embedded_daemon.child_env()
            socket_args = ["--socket", embedded_daemon.socket_path]
        return [driver_command, "call", name, json.dumps(call_args), *socket_args], child_env, shot_file

    def _call_tool_via_cli(self, name: str, args: Dict[str, Any], timeout: float) -> Dict[str, Any]:
        """Fallback transport: ``cua-driver call <tool> <json>`` as a subprocess.

        The ``cua-driver mcp`` stdio bridge can persistently fail heavy calls
        (notably ``get_window_state``) with EAGAIN while the plain CLI — which
        talks to the daemon over its own socket — keeps working. The JSON is
        remapped into the ``_extract_tool_result`` dict shape so callers stay
        transport-agnostic; the call is retried with backoff since the socket
        may still be busy.
        """
        from tools.environments.local import _sanitize_subprocess_env

        cmd, child_env, shot_file = self._cli_command(name, args)
        try:
            parsed = _cli_run_json(cmd, _sanitize_subprocess_env(child_env), name, timeout)
            return _cli_result(parsed, shot_file)
        finally:
            if shot_file and os.path.exists(shot_file):
                try:
                    os.remove(shot_file)
                except OSError:
                    pass

    def call_tool(self, name: str, args: Dict[str, Any], timeout: float = 30.0) -> Dict[str, Any]:
        # A prior MCP timeout marks the session suspect (possibly wedged for
        # every later call): recreate it first so one timeout never poisons
        # the rest of the run. Healthy sessions are never restarted here.
        if self._timeout_suspect and name not in self._LIFECYCLE_CALLS:
            logger.warning("cua-driver session suspect after earlier MCP timeout; "
                           "recreating before %s", name)
            with self._lock:
                self._restart_session_locked()
            self._timeout_suspect = False
            self._restore_declared_session_after_transport_reset(timeout)

        # A prior session may have died (MCP drop / driver crash) and reset
        # _started in its lifecycle finally.
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
                return self._timeout_outcome(name, e)
            if self._is_transient_daemon_error(e):
                if not self._transport_replay_is_safe(name):
                    self._notify_transport_reset()
                    return self._unknown_transport_outcome(name, e)
                logger.warning("cua-driver MCP transport failed on %s (%s); "
                               "falling back to CLI transport", name, e)
                return self._call_tool_via_cli(name, args, timeout)
            if not self._is_closed_session_error(e):
                raise
            logger.warning("cua-driver MCP session closed during %s; reconnecting once", name)
            self._restart_and_restore(timeout)
            if not self._transport_replay_is_safe(name):
                return self._unknown_transport_outcome(name, e)
            result = self._run_call(name, args, timeout)

        # Remember only a successfully declared stable identity, so a failed
        # start_session cannot leave stale recovery state behind.
        if name == "start_session" and result.get("isError") is not True:
            declared_id = args.get("session")
            if isinstance(declared_id, str) and declared_id:
                self._declared_session_id = declared_id

        if self._is_ended_session_result(result):
            result = self._revive_declared_session_once(name, args, result, timeout)

        if (
            name == "end_session"
            and result.get("isError") is not True
            and args.get("session") == self._declared_session_id
        ):
            self._declared_session_id = None
        return result
