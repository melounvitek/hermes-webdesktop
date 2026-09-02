"""Background-loop plumbing for tools.mcp_tool: the cross-process discovery file lock,
scheduling coroutines onto the MCP loop from caller threads (with profile HOME override
and dashboard OAuth flow propagation) and the loop's exception handler. Split from
tools/mcp_tool.py; origin state (``_lock``, ``_mcp_loop``) is read through ``_core`` so
``mock.patch("tools.mcp_tool.X")`` keeps working."""

from __future__ import annotations

import asyncio
import concurrent.futures
import errno
import logging
import os
import threading
import time
from typing import Any, Coroutine, Optional
from tools.mcp_tool_common import _core

logger = logging.getLogger("tools.mcp_tool")


class _LockCookie:
    """Holds a cross-process file lock; ``release()`` drops it.

    The file object MUST stay open while the lock is held: both the fcntl and
    the portalocker lock are tied to the descriptor's lifetime.
    """

    def __init__(self, fh: Any) -> None:
        self._fh = fh

    def release(self) -> None:
        if self._fh is None:
            return
        # Best effort on every step: an unlock/close failure must never
        # propagate out of discovery.
        try:
            if os.name == "posix":
                import fcntl
                fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
            else:
                import portalocker
                portalocker.unlock(self._fh)
        except Exception:
            pass
        try:
            self._fh.close()
        except Exception:
            pass
        self._fh = None


def _acquire_lock_on_fh(fh: Any) -> bool:
    """Non-blocking exclusive lock (fcntl on POSIX, portalocker elsewhere).

    False when another process holds it; unexpected errors propagate so the
    caller can treat locking as unavailable.
    """
    fd = fh.fileno()
    if os.name == "posix":
        import fcntl
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError as e:
            if e.errno in (errno.EACCES, errno.EAGAIN, errno.EWOULDBLOCK):
                return False
            raise
    else:
        import portalocker
        try:
            portalocker.lock(fh, portalocker.LOCK_EX | portalocker.LOCK_NB)
            return True
        except portalocker.LockException:
            return False


def _try_acquire_mcp_discovery_lock() -> Any:
    """Return a ``_LockCookie`` (acquired), ``None`` (held by another process)
    or ``_LOCK_UNAVAILABLE`` (locking broken: run discovery unguarded)."""
    # The cached path lives on the ORIGIN module (tests reset
    # ``tools.mcp_tool._MCP_DISCOVERY_LOCK_PATH = None``), so write it there.
    from tools import mcp_tool as _origin
    try:
        from hermes_constants import get_hermes_home
        if _origin._MCP_DISCOVERY_LOCK_PATH is None:
            _origin._MCP_DISCOVERY_LOCK_PATH = str(
                get_hermes_home() / ".mcp-discovery.lock"
            )
        lock_path = _origin._MCP_DISCOVERY_LOCK_PATH
    except Exception:
        return _core._LOCK_UNAVAILABLE

    try:
        fh = open(lock_path, "w", encoding="utf-8")
    except Exception:
        return _core._LOCK_UNAVAILABLE

    try:
        acquired = _core._acquire_lock_on_fh(fh)
    except Exception:
        fh.close()
        return _core._LOCK_UNAVAILABLE

    if acquired:
        return _core._LockCookie(fh)
    fh.close()
    return None


def _mcp_loop_exception_handler(loop, context):
    """Suppress the benign 'Event loop is closed' RuntimeError that httpx
    finalizers raise against the dead loop during shutdown; forward the rest."""
    exc = context.get("exception")
    if isinstance(exc, RuntimeError) and "Event loop is closed" in str(exc):
        return
    loop.default_exception_handler(context)


def _wrap_with_home_override(coro: "Coroutine") -> "Coroutine":
    """Carry the caller's context-local HERMES_HOME override into ``coro``
    (task-local on the MCP loop, so concurrent scopes don't interfere)."""
    try:
        from hermes_constants import (
            get_hermes_home_override,
            reset_hermes_home_override,
            set_hermes_home_override,
        )

        home_override = get_hermes_home_override()
    except Exception:
        return coro
    if not home_override:
        return coro

    async def _scoped():
        token = set_hermes_home_override(home_override)
        try:
            return await coro
        finally:
            reset_hermes_home_override(token)

    return _scoped()


def _wrap_with_dashboard_oauth_flow(coro):
    """Propagate a dashboard OAuth flow onto the dedicated MCP loop task."""
    try:
        from tools.mcp_dashboard_oauth import (
            dashboard_oauth_flow,
            get_dashboard_oauth_flow,
        )

        flow = get_dashboard_oauth_flow()
    except Exception:
        return coro
    if flow is None:
        return coro

    async def _scoped():
        with dashboard_oauth_flow(flow):
            return await coro

    return _scoped()


def _running_loop() -> Optional[asyncio.AbstractEventLoop]:
    """The MCP loop when it is up, else None (read under ``_lock``)."""
    with _core._lock:
        loop = _core._mcp_loop
    return loop if loop is not None and loop.is_running() else None


def _run_on_mcp_loop(coro_or_factory, timeout: float = 30):
    """Schedule a coroutine on the MCP loop and block until done.

    Accepts a coroutine or a zero-arg factory (a factory avoids leaking a
    never-awaited coroutine when the loop is down). Polls in short intervals
    so the calling thread can honor user interrupts.
    """
    from tools.interrupt import is_interrupted
    from agent.async_utils import safe_schedule_threadsafe

    loop = _running_loop()
    if loop is None:
        if asyncio.iscoroutine(coro_or_factory):
            coro_or_factory.close()
        raise RuntimeError("MCP event loop is not running")

    coro = coro_or_factory() if callable(coro_or_factory) else coro_or_factory

    # Tasks created via run_coroutine_threadsafe copy the LOOP thread's
    # context, so a per-request profile scope would vanish here; re-establish
    # it inside the task's own context.
    coro = _core._wrap_with_home_override(coro)
    coro = _core._wrap_with_dashboard_oauth_flow(coro)

    future = safe_schedule_threadsafe(
        coro, loop,
        logger=logger,
        log_message="MCP scheduling failed",
    )
    if future is None:
        raise RuntimeError("MCP event loop unavailable (failed to schedule)")
    start_time = time.monotonic()
    deadline = None if timeout is None else start_time + timeout

    while True:
        if is_interrupted():
            future.cancel()
            raise InterruptedError("User sent a new message")

        wait_timeout = 0.1
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                future.cancel()
                elapsed = time.monotonic() - start_time
                raise TimeoutError(
                    f"MCP call timed out after {elapsed:.1f}s "
                    f"(configured timeout: {float(timeout):.1f}s)"
                )
            wait_timeout = min(wait_timeout, remaining)

        try:
            return future.result(timeout=wait_timeout)
        except concurrent.futures.TimeoutError:
            # Aliases builtin TimeoutError, so this also fires for the
            # coroutine's own timeout: a done future must yield its outcome.
            if future.done():
                return future.result()
            continue

def _signal_reconnect(server: Any) -> bool:
    """Ask a server task to rebuild its transport, thread-safely.

    Handlers run on caller threads while the event lives on the MCP loop, so
    it is set via ``call_soon_threadsafe`` when the loop runs (direct
    ``.set()`` otherwise). False when the server has no reconnect machinery.
    """
    event = getattr(server, "_reconnect_event", None)
    if event is None:
        return False
    loop = _core._mcp_loop
    if isinstance(event, asyncio.Event) and loop is not None and loop.is_running():
        loop.call_soon_threadsafe(event.set)
    else:
        event.set()
    return True


def reconnect_mcp_server(server_name: str) -> bool:
    """Ask a currently-live MCP server to rebuild after external re-auth."""
    with _core._lock:
        server = _core._servers.get(server_name)
    if server is None:
        return False
    return _core._signal_reconnect(server)


def _wait_for_server_session_ready(
    srv: Any,
    *,
    old_session: Any = None,
    timeout: float = 15.0,
) -> bool:
    """Poll until the server exposes a usable, ready session.

    During a reconnect ``srv.session`` is briefly None or still the stale
    object; retrying blindly there burns breaker strikes. With
    ``old_session`` the observed session must differ from it. Iteration-
    bounded, not deadline-bounded: tests freeze ``time.monotonic``.
    """
    poll_interval = 0.25
    iterations = max(1, int(max(float(timeout), 0.0) / poll_interval))
    for i in range(iterations):
        session = getattr(srv, "session", None)
        ready = getattr(srv, "_ready", None)
        is_ready = True
        if ready is not None and hasattr(ready, "is_set"):
            try:
                is_ready = bool(ready.is_set())
            except Exception:
                is_ready = True
        if session is not None and session is not old_session and is_ready:
            return True
        if i < iterations - 1:
            time.sleep(poll_interval)
    return False


def _signal_reconnect_and_wait(
    server_name: str,
    srv: Any,
    *,
    op_description: str,
    timeout: float = 15.0,
) -> bool:
    """Request a transport rebuild and wait for the fresh session.

    ``_ready`` is cleared on the loop BEFORE ``_reconnect_event`` is set;
    otherwise the readiness poll returns immediately and retries against the
    same dead session.
    """
    loop = _core._mcp_loop
    if loop is None or not loop.is_running():
        return False
    old_session = getattr(srv, "session", None)

    def _request_reconnect() -> None:
        ready = getattr(srv, "_ready", None)
        if ready is not None and hasattr(ready, "clear"):
            ready.clear()
        reconnect_event = getattr(srv, "_reconnect_event", None)
        if reconnect_event is not None and hasattr(reconnect_event, "set"):
            reconnect_event.set()

    logger.info(
        "MCP server '%s': %s requesting transport reconnect",
        server_name, op_description,
    )
    loop.call_soon_threadsafe(_request_reconnect)
    return _core._wait_for_server_session_ready(
        srv,
        old_session=old_session,
        timeout=timeout,
    )


def _ensure_mcp_loop():
    """Start the background event loop thread if not already running.

    The loop/thread handles live on the ORIGIN module (tests read and reset
    ``tools.mcp_tool._mcp_loop``), so they are written there, never here.
    """
    from tools import mcp_tool as _origin
    with _core._lock:
        if _origin._mcp_loop is not None and _origin._mcp_loop.is_running():
            return
        _origin._mcp_loop = asyncio.new_event_loop()
        _origin._mcp_loop.set_exception_handler(_core._mcp_loop_exception_handler)
        _origin._mcp_thread = threading.Thread(
            target=_origin._mcp_loop.run_forever,
            name="mcp-event-loop",
            daemon=True,
        )
        _origin._mcp_thread.start()


def _stop_mcp_loop(*, only_if_idle: bool = False) -> bool:
    """Stop the background event loop and join its thread."""
    from tools import mcp_tool as _origin
    with _core._lock:
        if only_if_idle and (_core._servers or _core._server_connecting):
            logger.debug("Leaving MCP event loop running; active servers are registered or connecting")
            return False
        loop = _origin._mcp_loop
        thread = _origin._mcp_thread
        _origin._mcp_loop = None
        _origin._mcp_thread = None
    if loop is not None:
        # Drain before stopping: tasks still suspended when the loop closes
        # get resumed by the GC against a closed loop. shutdown_mcp_servers
        # only reaps servers held in _servers; everything else ends up here.
        stop_owned_by_loop = False
        if loop.is_running():
            from agent.async_utils import safe_schedule_threadsafe

            future = safe_schedule_threadsafe(
                _core._drain_and_stop_mcp_loop(), loop,
                logger=logger,
                log_message="MCP loop drain: failed to schedule",
                log_level=logging.WARNING,
            )
            if future is not None:
                stop_owned_by_loop = True
                try:
                    future.result(timeout=_core._MCP_LOOP_DRAIN_TIMEOUT + 1)
                except TimeoutError:
                    logger.warning(
                        "Timed out waiting for MCP loop drain after %.1fs",
                        _core._MCP_LOOP_DRAIN_TIMEOUT + 1,
                    )
                except BaseException as exc:
                    logger.warning("Error draining MCP loop tasks: %s", exc)
        elif not loop.is_closed():
            try:
                loop.run_until_complete(
                    _core._drain_mcp_loop_tasks(timeout=_core._MCP_LOOP_DRAIN_TIMEOUT)
                )
            except BaseException as exc:
                logger.warning("Error draining stopped MCP loop tasks: %s", exc)

        if not stop_owned_by_loop and loop.is_running():
            loop.call_soon_threadsafe(loop.stop)
        if thread is not None:
            thread.join(timeout=5)
            if thread.is_alive():
                logger.warning("MCP event loop thread did not stop within 5.0s")
        try:
            loop.close()
        except Exception as exc:
            logger.warning("Unable to close MCP event loop cleanly: %s", exc)
        # The loop is gone, so no session can be in flight: reap active too.
        _core._kill_orphaned_mcp_children(include_active=True)
    return True
