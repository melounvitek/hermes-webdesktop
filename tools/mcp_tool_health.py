"""Session health for MCPServerTask: dynamic tool refresh on list_changed notifications, server log forwarding, keepalive probes, suspect-mark / lazy-verify, in-flight call fail-fast, stdio child liveness and stdio idle/lifetime recycling. Split from tools/mcp_tool.py."""

import asyncio
import json
import logging
import time
from typing import Iterable, Optional
from tools.mcp_tool_errors import _is_method_not_found_error, _unwrap_exception_group
from tools.mcp_tool_schema import mcp_prefixed_tool_name
from tools.mcp_tool_registration import _forget_mcp_tool_server
from tools.mcp_tool_common import _core

logger = logging.getLogger("tools.mcp_tool")

_KEEPALIVE_RPC_TIMEOUT = 30.0


def _stdio_children_dead_impl(pids, is_http: bool) -> bool:
    """True when every pid has exited. Best-effort: False (unknown → don't fail
    fast) for HTTP, no captured PIDs, missing psutil, or a failed probe."""
    if not pids or is_http:
        return False
    try:
        import psutil
    except ImportError:
        return False
    for pid in pids:
        # pid_exists handles Windows without signal-permission noise.
        try:
            if psutil.pid_exists(pid):
                return False
        except Exception:
            return False
    return True


class MCPServerHealthMixin:
    """Methods of :class:`tools.mcp_tool.MCPServerTask` (mixed in; relies on its attributes)."""

    __slots__ = ()

    def _is_http(self) -> bool:
        """Check if this server uses HTTP transport."""
        return "url" in self._config

    def _is_recycled_stdio(self) -> bool:
        """Return True when a stdio server was intentionally recycled."""
        return not self._is_http() and self._recycled_reason is not None

    def mark_tool_call(self) -> None:
        """Record that a user-visible MCP operation is starting."""
        self._last_tool_call_at = time.monotonic()

    def _mark_lifecycle_started(self) -> None:
        now = time.monotonic()
        self._lifecycle_started_at = now
        self._last_tool_call_at = now
        self._recycled_reason = None

    # ------------------------------------------------------- stdio recycling

    def _stdio_recycle_deadlines(self):
        """``[(deadline, reason), ...]`` for the configured lifetime/idle limits;
        empty for HTTP servers or while an RPC holds the lock."""
        if self._is_http() or self._rpc_lock.locked():
            return []
        deadlines = []
        if self._max_lifetime_seconds is not None:
            deadlines.append((self._lifecycle_started_at + self._max_lifetime_seconds, "max_lifetime_seconds"))
        if self._idle_timeout_seconds is not None:
            deadlines.append((self._last_tool_call_at + self._idle_timeout_seconds, "idle_timeout_seconds"))
        return deadlines

    def _stdio_recycle_reason(self, now: Optional[float] = None) -> Optional[str]:
        """Return the stdio recycle reason if idle/age limits have elapsed (lifetime wins)."""
        now = time.monotonic() if now is None else now
        for deadline, reason in self._stdio_recycle_deadlines():
            if now >= deadline:
                return reason
        return None

    def _next_stdio_recycle_deadline(self) -> Optional[float]:
        """Return the next monotonic recycle deadline for stdio, if any."""
        deadlines = self._stdio_recycle_deadlines()
        return min(d for d, _ in deadlines) if deadlines else None

    def _mark_stdio_recycled(self, reason: str) -> None:
        """Mark a stdio session dormant before its transport finishes closing."""
        self._recycled_reason = reason
        self.session = None

    # -------------------------------------------------- notifications / logs

    async def _refresh_tools_task(self):
        """Run a dynamic tool refresh and log failures from background tasks."""
        try:
            await self._refresh_tools()
        except Exception:
            logger.exception("MCP server '%s': dynamic tool refresh failed", self.name)

    def _schedule_tools_refresh(self) -> asyncio.Task:
        """Schedule a background tool refresh and keep it strongly referenced."""
        task = asyncio.create_task(self._refresh_tools_task())
        self._pending_refresh_tasks.add(task)
        task.add_done_callback(self._pending_refresh_tasks.discard)
        return task

    def _make_logging_callback(self):
        """Build a ``logging_callback`` that forwards server ``notifications/message``
        into Hermes logging tagged with the server name (the SDK default drops them)."""
        async def _on_log(params):
            try:
                level = _core._MCP_LOG_LEVEL_MAP.get(
                    str(getattr(params, "level", "info")).lower(), logging.INFO,
                )
                data = getattr(params, "data", None)
                if not isinstance(data, str):
                    try:
                        data = json.dumps(data, ensure_ascii=False, default=str)
                    except (TypeError, ValueError):
                        data = str(data)
                # Cap payloads so a chatty server can't flood agent.log.
                if len(data) > 2000:
                    data = data[:2000] + "... [truncated]"
                logger_name = getattr(params, "logger", None)
                origin = f"{self.name}/{logger_name}" if logger_name else self.name
                logger.log(level, "MCP server log [%s]: %s", origin, data)
            except Exception:
                logger.debug(
                    "Failed to handle MCP log notification from '%s'",
                    self.name, exc_info=True,
                )
        return _on_log

    def _make_message_handler(self):
        """Build a ``message_handler`` for ``ClientSession``: only
        ``ToolListChangedNotification`` triggers a refresh; prompt/resource changes are logged."""
        async def _handler(message):
            try:
                if isinstance(message, Exception):
                    logger.debug("MCP message handler (%s): exception: %s", self.name, message)
                    return
                if _core._MCP_NOTIFICATION_TYPES and isinstance(message, _core.ServerNotification):
                    # mcp 2.0 made ServerNotification a plain union (payload IS the
                    # message) instead of a RootModel (payload under ``.root``).
                    # ``isinstance`` accepts both; only the unwrap differs — without
                    # it ``.root`` raises into the catch-all and refreshes stop.
                    match getattr(message, "root", message):
                        case _core.ToolListChangedNotification():
                            logger.info(
                                "MCP server '%s': received tools/list_changed notification",
                                self.name,
                            )
                            # Refresh in a separate task: some servers emit
                            # list_changed right after initialize while another
                            # request is in flight, and refreshing synchronously
                            # inside the handler can wedge the stdio JSON-RPC stream.
                            self._schedule_tools_refresh()
                            # Yield one tick so short-lived notification contexts
                            # (and tests) can observe the scheduled refresh.
                            await asyncio.sleep(0)
                        case _core.PromptListChangedNotification():
                            logger.debug("MCP server '%s': prompts/list_changed (ignored)", self.name)
                        case _core.ResourceListChangedNotification():
                            logger.debug("MCP server '%s': resources/list_changed (ignored)", self.name)
                        case _:
                            pass
            except Exception:
                logger.exception("Error in MCP message handler for '%s'", self.name)
        return _handler

    def _deregister_owned(self, tool_names: Iterable[str]) -> None:
        """Deregister *tool_names* that this server's toolset still owns.
        Never removes a colliding name currently owned by another server."""
        from tools.registry import registry

        toolset_name = f"mcp-{self.name}"
        for tool_name in tool_names:
            if registry.get_toolset_for_tool(tool_name) != toolset_name:
                continue
            registry.deregister(tool_name, scope=_core._server_registry_scope(self.name))
            _forget_mcp_tool_server(tool_name)

    async def _refresh_tools(self):
        """Re-fetch tools on ``tools/list_changed`` and update the registry.

        The lock serializes rapid-fire notifications. After the list_tools
        ``await``, all mutations are synchronous — atomic on the event loop.
        """
        if not self._advertises_tools():
            # Shouldn't happen, but tools/list would raise MCPError(-32601).
            return

        async with self._refresh_lock:
            old_tool_names = set(self._registered_tool_names)

            # 1. Fetch the current tool list (follow nextCursor).
            async with self._rpc_lock:
                new_mcp_tools = await _core._paginate_full_list(
                    self.session.list_tools, "tools", self.name
                )

            # 2. Remove only stale names first — no nuke-and-repave: live agent
            # turns may hold tool-call IDs pointing at existing handlers, and
            # in-place replacement avoids transient "tool not connected" races.
            self._deregister_owned(old_tool_names - {
                mcp_prefixed_tool_name(self.name, tool.name) for tool in new_mcp_tools
            })

            # 3. Re-register; the helper may skip names ambiguous after normalization.
            self._tools = new_mcp_tools
            registered_names = _core._register_server_tools(
                self.name, self, self._config
            )
            # A raw name can become ambiguous without changing its normalized
            # name, so the pre-pass misses it: drop any old entry the final
            # collision-checked registration no longer owns.
            self._deregister_owned(old_tool_names - set(registered_names))
            self._registered_tool_names = registered_names

            # 4. Log what changed (user-visible).
            new_tool_names = set(self._registered_tool_names)
            added = new_tool_names - old_tool_names
            removed = old_tool_names - new_tool_names
            changes = []
            if added:
                changes.append(f"added: {', '.join(sorted(added))}")
            if removed:
                changes.append(f"removed: {', '.join(sorted(removed))}")
            if changes:
                logger.warning(
                    "MCP server '%s': tools changed dynamically — %s. "
                    "Verify these changes are expected.",
                    self.name, "; ".join(changes),
                )
            else:
                logger.info(
                    "MCP server '%s': dynamically refreshed %d tool(s) (no changes)",
                    self.name, len(self._registered_tool_names),
                )

    # ------------------------------------------------------ keepalive / health

    async def _keepalive_probe(self) -> None:
        """Exercise the session; raise on a genuine connection failure.

        ``ping`` first (cheap, OPTIONAL utility). On -32601 latch
        ``_ping_unsupported`` and fall back to ``list_tools`` when the server
        advertises tools; otherwise the -32601 propagates (no liveness primitive
        left). The latch resets on each fresh transport connection.
        """
        if not self._ping_unsupported:
            try:
                await asyncio.wait_for(self.session.send_ping(), timeout=_KEEPALIVE_RPC_TIMEOUT)
                return
            except Exception as exc:
                if _is_method_not_found_error(exc):
                    # Ping is definitively unsupported.
                    if not self._advertises_tools():
                        raise
                    self._ping_unsupported = True
                    logger.info(
                        "MCP server '%s': does not implement the optional "
                        "'ping' utility (-32601); using 'list_tools' for "
                        "keepalive on this connection.",
                        self.name,
                    )
                elif isinstance(exc, (TimeoutError, asyncio.TimeoutError)) and self._advertises_tools():
                    # A server that silently drops ping looks like a dead transport.
                    # Confirm with list_tools before declaring it dead; if that
                    # also fails, propagate the original failure.
                    try:
                        await asyncio.wait_for(self.session.list_tools(), timeout=_KEEPALIVE_RPC_TIMEOUT)
                    except Exception:
                        raise exc from None
                    # Transport alive; latch so later keepalives skip the 30s wait.
                    self._ping_unsupported = True
                    logger.info(
                        "MCP server '%s': ping timed out but list_tools "
                        "succeeded — server silently drops ping; using "
                        "'list_tools' for keepalive on this connection.",
                        self.name,
                    )
                    return
                else:
                    # Closed transport, expired session, etc. — real failure.
                    raise

        # Fallback probe for servers without ping support.
        await asyncio.wait_for(self.session.list_tools(), timeout=_KEEPALIVE_RPC_TIMEOUT)

    def _mark_session_proven(self) -> None:
        """Record that the session demonstrated real health (keepalive or tool-call success).

        Only then is the reconnect budget cleared: a handshake that drops moments
        later must keep consuming ``_reconnect_retries`` so a flapping transport
        still reaches the park instead of respawning forever.
        """
        if self._session_proven:
            return
        self._session_proven = True
        self._reconnect_retries = 0
        if self._was_parked:
            self._was_parked = False
            logger.warning(
                "MCP server '%s': revived — session healthy again after "
                "parking (state: parked → connected)",
                self.name,
            )
        # A proven fresh transport clears the one-time permanent-failure
        # grace and any race bookkeeping.
        self._permanent_grace_used = False
        self._teardown_race = False

    def mark_suspect(self, reason: str) -> None:
        """Latch a suspicion (no I/O). The NEXT call verifies via
        :meth:`ensure_healthy` and recycles the transport if the probe fails."""
        if self._suspect_reason is None and reason:
            logger.warning(
                "MCP server '%s': connection marked suspect (%s); next call "
                "will health-check it",
                self.name, reason,
            )
        self._suspect_reason = reason or None

    async def ensure_healthy(self, timeout: float = 5.0) -> bool:
        """Verify a suspect connection before reuse; recycle if dead.

        True when healthy (suspicion cleared). On failure requests a reconnect,
        drops the stale session so the caller's no-session path takes over, and
        returns False. Never raises.
        """
        reason = self._suspect_reason
        if not reason:
            return True
        if self.session is None:
            # Nothing to verify — the reconnect path owns recovery now.
            self._suspect_reason = None
            self._reconnect_event.set()
            return False
        try:
            await asyncio.wait_for(self._keepalive_probe(), timeout=timeout)
        except Exception as exc:
            root = _unwrap_exception_group(exc)
            logger.warning(
                "MCP server '%s': suspect connection (%s) failed health "
                "check (%s: %s) — requesting reconnect (state: suspect → "
                "degraded)",
                self.name, reason, type(root).__name__, root,
            )
            self._suspect_reason = None
            self.mark_suspect(f"health check failed after {reason}")
            self.session = None
            self._ready.clear()
            self._reconnect_event.set()
            return False
        logger.info(
            "MCP server '%s': suspect connection passed health check "
            "(%s) — clearing suspicion",
            self.name, reason,
        )
        self._suspect_reason = None
        self._mark_session_proven()
        return True

    def _fail_inflight_calls(self, reason: str) -> None:
        """Cancel every in-flight RPC on this connection.

        Called from lifecycle exits BEFORE the transport unwinds: the SDK does
        not always fail pending requests when streams close, so a call would
        otherwise wait out the full tool timeout. Cancelling anything flags
        ``_teardown_race`` so run() treats the next reconnect as recovery
        rather than charging the rapid-drop budget.
        """
        victims = [t for t in self._inflight_tasks if not t.done()]
        if not victims:
            return
        self._reconnecting = True
        self._teardown_race = True
        self.mark_suspect(f"{reason} tore down {len(victims)} in-flight call(s)")
        for task in victims:
            task.cancel()

    def _stdio_children_dead(self) -> bool:
        """True when every stdio child we spawned has exited (see :func:`_stdio_children_dead_impl`)."""
        return _stdio_children_dead_impl(getattr(self, "_stdio_child_pids", None), self._is_http())

    async def _watch_stdio_children(self) -> None:
        """Poll child liveness while a stdio RPC is in flight; resolves when a
        tracked child dies so the caller cancels the RPC instead of waiting out the timeout."""
        while not self._stdio_children_dead():
            await asyncio.sleep(0.25)
