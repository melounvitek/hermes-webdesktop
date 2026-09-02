"""Lifecycle of :class:`tools.mcp_tool.MCPServerTask`: the long-lived ``run`` state machine
(connect -> serve -> reconnect/park/recycle), keepalive-driven lifecycle waits, start/shutdown
and tool deregistration. Split from tools/mcp_tool.py; origin state and patchable helpers are
read through ``_core`` so ``mock.patch("tools.mcp_tool.X")`` keeps working."""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Optional
from tools.mcp_tool_common import _core

logger = logging.getLogger("tools.mcp_tool")


@dataclass
class _RetryBudget:
    """Per-run() retry counters shared by the branch helpers (``_reconnect_retries``
    lives on the task because handlers and tests read it)."""

    initial_retries: int = 0
    backoff: float = 1.0


class MCPServerRunMixin:
    """Methods of :class:`tools.mcp_tool.MCPServerTask` (mixed in; relies on its attributes)."""

    @staticmethod
    async def _cancel_waiters(*tasks: asyncio.Task) -> None:
        for t in tasks:
            if not t.done():
                t.cancel()
                try:
                    await t
                except (asyncio.CancelledError, Exception):
                    pass

    def _recycle_if_due(self) -> bool:
        """Latch a stdio idle/lifetime recycle when its deadline has passed."""
        recycle_reason = self._stdio_recycle_reason()
        if recycle_reason is None:
            return False
        self._mark_stdio_recycled(recycle_reason)
        return True

    async def _wait_for_lifecycle_event(self) -> str:
        """Serve the connection until a lifecycle event; return its kind.

        ``"shutdown"`` exits the run loop; ``"reconnect"`` tears the session
        down and re-enters the transport (event cleared before return);
        ``"recycle"`` means a stdio idle/lifetime limit elapsed and the
        transport restarts lazily on the next call. Shutdown wins a tie.

        Between events a keepalive (``ping``, list_tools fallback) runs every
        ``keepalive_interval`` — which must stay below the server's session
        TTL — and a failure triggers a reconnect. ``ping`` is a few bytes
        regardless of tool count; list_changed notifications still arrive
        out-of-band.
        """
        keepalive_interval = max(
            _core._MIN_KEEPALIVE_INTERVAL,
            float(self._config.get("keepalive_interval", _core._DEFAULT_KEEPALIVE_INTERVAL)),
        )

        shutdown_task = asyncio.create_task(self._shutdown_event.wait())
        reconnect_task = asyncio.create_task(self._reconnect_event.wait())
        try:
            while True:
                if self._recycle_if_due():
                    return "recycle"

                timeout = keepalive_interval
                recycle_deadline = self._next_stdio_recycle_deadline()
                if recycle_deadline is not None:
                    timeout = max(0.0, min(timeout, recycle_deadline - time.monotonic()))

                done, _pending = await asyncio.wait(
                    {shutdown_task, reconnect_task},
                    timeout=timeout,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if done:
                    break
                if self._recycle_if_due():
                    return "recycle"

                # Timeout: probe for a stale session — but NEVER while an RPC
                # is in flight (a concurrent ping can wedge the single stdio
                # stream, and a busy server is provably alive anyway).
                if self.session:
                    if self._rpc_lock.locked() or any(
                        not t.done() for t in self._inflight_tasks
                    ):
                        continue
                    try:
                        async with self._rpc_lock:
                            await self._keepalive_probe()
                    except Exception as exc:
                        root = _core._unwrap_exception_group(exc)
                        logger.warning(
                            "MCP server '%s' keepalive failed, triggering "
                            "reconnect (state: connected → degraded): %s: %s",
                            self.name, type(root).__name__, root,
                        )
                        self.mark_suspect(
                            f"keepalive failed: {type(root).__name__}: {root}"
                        )
                        self._reconnect_event.set()
                        break
                    # Survived a full keepalive interval: real proof of health.
                    self._mark_session_proven()
        finally:
            await self._cancel_waiters(shutdown_task, reconnect_task)

        if self._shutdown_event.is_set():
            self._fail_inflight_calls("shutdown")
            return "shutdown"
        # Deliberate teardown: fail in-flight RPCs NOW rather than letting
        # them ride the dying transport to the full tool timeout.
        self._fail_inflight_calls("reconnect")
        self._reconnect_event.clear()
        return "reconnect"

    async def _wait_for_reconnect_or_shutdown(
        self, timeout: Optional[float] = None
    ) -> str:
        """Wait, while parked, for a reconnect request or shutdown.

        Returns ``"shutdown"`` or ``"reconnect"`` (explicit request or, with
        ``timeout``, the periodic self-probe); the reconnect event is cleared
        first. Shutdown wins a tie.
        """
        shutdown_task = asyncio.ensure_future(self._shutdown_event.wait())
        reconnect_task = asyncio.ensure_future(self._reconnect_event.wait())
        try:
            await asyncio.wait(
                {shutdown_task, reconnect_task},
                return_when=asyncio.FIRST_COMPLETED,
                timeout=timeout,
            )
        finally:
            await self._cancel_waiters(shutdown_task, reconnect_task)
        if self._shutdown_event.is_set():
            return "shutdown"
        self._reconnect_event.clear()
        return "reconnect"

    async def _park(self, revival_reason: str) -> bool:
        """Drop this server's tools and wait for a reconnect request.

        The run task must NOT exit: it is the only listener on
        ``_reconnect_event``, so returning leaves the server unrevivable for
        the life of the process. Parking deregisters the tools, so no call
        can reach the breaker probe or ``_signal_reconnect``; the wait is
        therefore TIMED (one self-probe per ``_PARKED_RETRY_INTERVAL``), and
        an explicit ``_reconnect_event.set()`` wakes it immediately. Returns
        True when shutdown was requested instead.
        """
        self._was_parked = True
        self._deregister_tools()
        self._reconnect_event.clear()
        parked = await self._wait_for_reconnect_or_shutdown(
            timeout=_core._PARKED_RETRY_INTERVAL
        )
        if parked == "shutdown":
            return True
        logger.debug(
            "MCP server '%s': attempting revival %s (self-probe or explicit "
            "reconnect request); rebuilding transport.",
            self.name, revival_reason,
        )
        return False

    async def _prepare_run(self, config: dict) -> bool:
        """Bind config, build sampling/elicitation handlers, validate HTTP.

        Returns False when the server must not start: a bad remote URL or a
        non-MCP endpoint (both fail fast, non-retryably, with ``_error`` set
        and ``_ready`` fired) instead of burning the reconnect ladder inside
        the SDK's httpx layer on every retry.
        """
        self._config = config
        self.tool_timeout = _core._resolve_tool_timeout(config)
        self._auth_type = (config.get("auth") or "").lower().strip()
        self._idle_timeout_seconds = _core._get_lifecycle_seconds(config, "idle_timeout_seconds")
        self._max_lifetime_seconds = _core._get_lifecycle_seconds(config, "max_lifetime_seconds")

        # The _MCP_*_TYPES flags are False until the lazy SDK import runs.
        _core._ensure_mcp_sdk()

        sampling_config = config.get("sampling", {})
        if sampling_config.get("enabled", True) and _core._MCP_SAMPLING_TYPES:
            self._sampling = _core.SamplingHandler(self.name, sampling_config)
        else:
            self._sampling = None

        # elicitation/create lets a server ask for structured input mid-call;
        # the handler routes it through Hermes' approval system.
        elicitation_config = config.get("elicitation", {})
        if elicitation_config.get("enabled", True) and _core._MCP_ELICITATION_TYPES:
            self._elicitation = _core.ElicitationHandler(self.name, elicitation_config, owner=self)
        else:
            self._elicitation = None

        if "url" in config and "command" in config:
            logger.warning(
                "MCP server '%s' has both 'url' and 'command' in config. "
                "Using HTTP transport ('url'). Remove 'command' to silence "
                "this warning.",
                self.name,
            )

        if not self._is_http():
            return True
        try:
            _core._validate_remote_mcp_url(self.name, config.get("url"))
        except _core.InvalidMcpUrlError as exc:
            logger.warning("%s", exc)
            self._error = exc
            self._ready.set()
            return False

        # Content-type preflight (Streamable HTTP only; SSE legitimately serves
        # text/event-stream): a URL at a web-app root returns HTML and would
        # make the SDK hang for the full connect_timeout. Skipped once _ready
        # was ever set (endpoint already validated) and for OAuth servers,
        # where a token-less probe sees HTML/401 and would block the flow.
        if config.get("transport") != "sse" and not config.get("skip_preflight") and not self._ready.is_set() and self._auth_type != "oauth":
            try:
                _probe_headers = dict(config.get("headers") or {})
                await self._preflight_content_type(
                    config["url"],
                    headers=_probe_headers,
                    ssl_verify=config.get("ssl_verify", True),
                    client_cert=_core._resolve_client_cert(self.name, config),
                )
            except _core.NonMcpEndpointError as exc:
                logger.warning("%s", exc)
                self._error = exc
                self._ready.set()
                return False
        return True

    async def run(self, config: dict):
        """Long-lived coroutine: connect, discover, serve, reconnect.

        State machine: connecting -> connected -> (degraded -> parked ->
        revived)*. Unproven drops and transport errors charge a rapid-drop
        budget with jittered exponential backoff; exhausting it (or a
        permanent error) parks the server via :meth:`_park` rather than
        exiting, so it stays revivable. The branch helpers return True to
        keep looping and False to exit the loop.
        """
        if not await self._prepare_run(config):
            return

        self._reconnect_retries = 0
        budget = _RetryBudget()

        while True:
            try:
                if self._is_http():
                    lifecycle_reason = await self._run_http(config)
                else:
                    lifecycle_reason = await self._run_stdio(config)
                if not await self._on_clean_return(lifecycle_reason, budget):
                    break
            except asyncio.CancelledError:
                # Not a connection failure: re-raise so cancellation reaches
                # asyncio and shutdown()'s ``await self._task`` completes.
                self.session = None
                raise
            except Exception as exc:
                self.session = None
                if not await self._on_transport_error(exc, budget):
                    break
            finally:
                self.session = None
                # Stale PIDs must never fast-fail the NEXT transport's calls.
                self._stdio_child_pids = set()

    async def _on_clean_return(self, lifecycle_reason: str, budget: "_RetryBudget") -> bool:
        """Transport returned cleanly: shutdown, stdio recycle, or a requested
        rebuild (auth recovery / manual refresh / keepalive failure). A rebuild
        is not a failure for the retry counters."""
        if self._shutdown_event.is_set():
            return False
        if lifecycle_reason == "recycle":
            logger.info(
                "MCP server '%s': stdio session recycled after %s; "
                "waiting for lazy reconnect",
                self.name, self._recycled_reason,
            )
            self.session = None
            # Dormant until a lazy call wakes it (untimed: nothing to self-probe).
            return await self._wait_for_reconnect_or_shutdown() != "shutdown"
        # Per-cycle chatter stays DEBUG; WARNINGs mark state transitions.
        logger.debug(
            "MCP server '%s': reconnecting (OAuth recovery or "
            "manual refresh)",
            self.name,
        )
        # A clean return is NOT proof of health (a flapping transport
        # handshakes fine and drops moments later). Only a PROVEN
        # session clears the budget; a teardown race is recovery, not
        # a failure, and must never reach the park on its own.
        if self._teardown_race and not self._session_proven:
            logger.info(
                "MCP server '%s': reconnect after teardown race "
                "(in-flight calls were failed); not charging the "
                "rapid-drop budget",
                self.name,
            )
            self._teardown_race = False
            budget.backoff = 1.0
        elif self._session_proven:
            self._reconnect_retries = 0
            budget.backoff = 1.0
        else:
            self._reconnect_retries += 1
            if self._reconnect_retries > _core._MAX_RECONNECT_RETRIES:
                logger.warning(
                    "MCP server '%s': %d consecutive reconnects "
                    "without a healthy session (rapid-drop budget "
                    "exhausted), parking; will self-probe every %ds "
                    "until it recovers (state: degraded → parked)",
                    self.name, _core._MAX_RECONNECT_RETRIES,
                    _core._PARKED_RETRY_INTERVAL,
                )
                if not await self._park_and_rearm("from parked state", budget):
                    return False
        # Clear readiness too: a stale _ready lets handler-side
        # recovery mistake the old session for a fresh one.
        self._ready.clear()
        self.session = None
        return True

    async def _park_and_rearm(self, revival_reason: str, budget: "_RetryBudget") -> bool:
        """Park; on revival leave a budget of ONE probe per wake so a still-dead
        server parks again instead of burning 5 rapid retries. False on shutdown."""
        if await self._park(revival_reason):
            return False
        self._reconnect_retries = _core._MAX_RECONNECT_RETRIES
        budget.backoff = 1.0
        return True

    async def _park_initial_failure(self, exc: Exception, revival_reason: str,
                                    budget: "_RetryBudget") -> bool:
        """Publish ``exc`` to the waiting ``start()``, park, and on revival reset
        every counter so the ladder starts fresh. False on shutdown."""
        self._error = exc
        self._ready.set()
        if await self._park(revival_reason):
            return False
        budget.initial_retries = 0
        self._reconnect_retries = 0
        budget.backoff = 1.0
        self._error = None
        self._ready.clear()
        return True

    async def _backoff_sleep(self, budget: "_RetryBudget") -> None:
        await asyncio.sleep(_core._jittered(budget.backoff))
        budget.backoff = min(budget.backoff * 2, _core._MAX_BACKOFF_SECONDS)

    async def _on_transport_error(self, exc: Exception, budget: "_RetryBudget") -> bool:
        """Transport raised: classify, then run the initial-connect or the
        reconnect ladder. Returns False when the run loop must exit."""
        # Unwrap anyio TaskGroup wrappers: the group's str() is useless
        # and hides the root cause from the classification below.
        root = _core._unwrap_exception_group(exc)
        failure_class = _core._classify_mcp_failure(root)
        if self._is_recycled_stdio():
            logger.warning(
                "MCP server '%s': lazy reconnect after stdio recycle "
                "failed, marking unavailable while retrying: %s: %s",
                self.name, type(root).__name__, root,
            )
            self._recycled_reason = None

        # Initial-connect ladder: a transient blip at startup must not
        # kill the server. Gated on _ever_connected (never cleared),
        # not _ready (cleared every reconnect cycle).
        if not self._ever_connected:
            return await self._on_initial_connect_error(exc, root, failure_class, budget)

        # If shutdown was requested, don't reconnect
        if self._shutdown_event.is_set():
            logger.debug(
                "MCP server '%s' disconnected during shutdown: %s: %s",
                self.name, type(root).__name__, root,
            )
            return False

        if failure_class == "permanent":
            return await self._on_permanent_error(root, budget)

        self._reconnect_retries += 1
        if self._reconnect_retries > _core._MAX_RECONNECT_RETRIES:
            logger.warning(
                "MCP server '%s' failed after %d reconnection attempts, "
                "parking; will self-probe every %ds until it recovers "
                "(state: degraded → parked): %s: %s",
                self.name, _core._MAX_RECONNECT_RETRIES,
                _core._PARKED_RETRY_INTERVAL,
                type(root).__name__, root,
            )
            return await self._park_and_rearm("from parked state", budget)

        logger.debug(
            "MCP server '%s' connection lost (attempt %d/%d), "
            "reconnecting in %.0fs: %s: %s",
            self.name, self._reconnect_retries, _core._MAX_RECONNECT_RETRIES,
            budget.backoff, type(root).__name__, root,
        )
        await self._backoff_sleep(budget)
        # Check again after sleeping
        return not self._shutdown_event.is_set()

    async def _on_initial_connect_error(self, exc: Exception, root: BaseException,
                                        failure_class: str, budget: "_RetryBudget") -> bool:
        if failure_class == "permanent":
            # Deterministic failure (bad command, non-MCP URL,
            # 401/403): park at once instead of burning the ladder.
            # Auth failures park rather than return so the task
            # stays alive to pick up fresh tokens later.
            if _core._is_auth_error(root):
                logger.warning(
                    "MCP server '%s' failed initial authentication, "
                    "parking until credentials change; re-authenticate "
                    "with `hermes mcp login %s` "
                    "(state: connecting → parked): %s: %s",
                    self.name, self.name,
                    type(root).__name__, root,
                )
            else:
                logger.warning(
                    "MCP server '%s' failed initial connection with a "
                    "permanent error, parking without retries "
                    "(state: connecting → parked): %s: %s",
                    self.name, type(root).__name__, root,
                )
            return await self._park_initial_failure(exc, "after permanent initial failure", budget)

        budget.initial_retries += 1
        if budget.initial_retries > _core._MAX_INITIAL_CONNECT_RETRIES:
            logger.warning(
                "MCP server '%s' failed initial connection after "
                "%d attempts, parking until a reconnect is "
                "requested (state: connecting → parked): %s: %s",
                self.name, _core._MAX_INITIAL_CONNECT_RETRIES,
                type(root).__name__, root,
            )
            return await self._park_initial_failure(exc, "after initial connection failures", budget)

        logger.debug(
            "MCP server '%s' initial connection failed "
            "(attempt %d/%d), retrying in %.0fs: %s: %s",
            self.name, budget.initial_retries,
            _core._MAX_INITIAL_CONNECT_RETRIES, budget.backoff,
            type(root).__name__, root,
        )
        await self._backoff_sleep(budget)
        # Check if shutdown was requested during the sleep
        if self._shutdown_event.is_set():
            self._error = exc
            self._ready.set()
            return False
        return True

    async def _on_permanent_error(self, root: BaseException, budget: "_RetryBudget") -> bool:
        # An auth failure on a PROVEN session is often a corrupt
        # OAuth lock from a raced teardown, not revoked
        # credentials: grant ONE suspect+reconnect cycle first.
        if (
            _core._is_auth_error(root)
            and self._session_proven
            and not self._permanent_grace_used
        ):
            self._permanent_grace_used = True
            self.mark_suspect(
                f"auth error on proven session: {root}"
            )
            logger.warning(
                "MCP server '%s': auth error on a previously "
                "healthy session — marking suspect and forcing "
                "one reconnect instead of parking (state: "
                "connected → suspect): %s: %s",
                self.name, type(root).__name__, root,
            )
            self._reconnect_retries = 0
            budget.backoff = 1.0
            await asyncio.sleep(_core._jittered(1.0))
            return not self._shutdown_event.is_set()
        # Deterministic failure on a working server: park now.
        logger.warning(
            "MCP server '%s' hit a permanent error, parking "
            "without retries; will self-probe every %ds "
            "(state: connected → parked): %s: %s",
            self.name, _core._PARKED_RETRY_INTERVAL,
            type(root).__name__, root,
        )
        return await self._park_and_rearm("from parked state (permanent error)", budget)

    async def start(self, config: dict):
        """Create the background Task and wait until ready (or failed)."""
        self._task = asyncio.ensure_future(self.run(config))
        try:
            await self._ready.wait()
        except asyncio.CancelledError:
            # The caller's connect timeout (discover_mcp_tools wraps start()
            # in asyncio.wait_for) cancels *this* coroutine, but the
            # ensure_future'd run() task is independent and would otherwise
            # keep running detached — parked on a hung transport with no
            # owner to reap it (#59349). Propagate the cancellation so the
            # transport context managers unwind and their finally blocks
            # release the child process / FDs.
            if self._task and not self._task.done():
                self._task.cancel()
            raise
        if self._error:
            raise self._error

    async def shutdown(self):
        """Signal the Task to exit and wait for clean resource teardown."""
        self._shutdown_event.set()
        # Defensive: if _wait_for_lifecycle_event is blocking, we need ANY
        # event to unblock it. _shutdown_event alone is sufficient (the
        # helper checks shutdown first), but setting reconnect too ensures
        # there's no race where the helper misses the shutdown flag after
        # returning "reconnect".
        self._reconnect_event.set()
        if self._task and not self._task.done():
            try:
                await asyncio.wait_for(self._task, timeout=10)
            except asyncio.TimeoutError:
                logger.warning(
                    "MCP server '%s' shutdown timed out, cancelling task",
                    self.name,
                )
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    pass
        if self._pending_refresh_tasks:
            for task in list(self._pending_refresh_tasks):
                task.cancel()
            await asyncio.gather(*self._pending_refresh_tasks, return_exceptions=True)
            self._pending_refresh_tasks.clear()
        self._deregister_tools()
        self.session = None

    def _deregister_tools(self) -> None:
        """Drop this server's tools from the global registry (idempotent).

        Pulls the server's tool schemas out of the registry so the agent
        stops advertising them to the model. Called on shutdown AND when the
        reconnect budget is exhausted, so a dead server never leaves phantom
        tool definitions bloating the prompt cache and producing "not
        connected" errors on every turn.
        """
        from tools.registry import registry

        for tool_name in list(getattr(self, "_registered_tool_names", [])):
            registry.deregister(tool_name, scope=_core._server_registry_scope(self.name))
            _core._forget_mcp_tool_server(tool_name)
        self._registered_tool_names = []
