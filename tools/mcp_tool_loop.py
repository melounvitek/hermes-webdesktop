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
import time
from typing import Any, Coroutine
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
        if self._fh is not None:
            try:
                fd = self._fh.fileno()
                if os.name == "posix":
                    import fcntl
                    try:
                        fcntl.flock(fd, fcntl.LOCK_UN)
                    except Exception:
                        pass
                else:
                    import portalocker
                    try:
                        portalocker.unlock(self._fh)
                    except Exception:
                        pass
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


def _run_on_mcp_loop(coro_or_factory, timeout: float = 30):
    """Schedule a coroutine on the MCP loop and block until done.

    Accepts a coroutine or a zero-arg factory (a factory avoids leaking a
    never-awaited coroutine when the loop is down). Polls in short intervals
    so the calling thread can honor user interrupts.
    """
    from tools.interrupt import is_interrupted
    from agent.async_utils import safe_schedule_threadsafe

    with _core._lock:
        loop = _core._mcp_loop
    if loop is None or not loop.is_running():
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
