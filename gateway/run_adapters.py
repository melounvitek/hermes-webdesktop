"""Adapter connect/disconnect, fatal-error recovery, reconnect watcher and multiplex profile adapter methods for GatewayRunner.

Split out of ``gateway/run.py``; bound onto ``GatewayRunner`` via the MRO.
``gateway.run`` internals are imported lazily inside method bodies (import cycle),
so ``patch("gateway.run.X")`` keeps intercepting them at call time.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
import asyncio
import os
import time
import weakref as _weakref
from agent.async_utils import consume_detached_task_result
from contextvars import Context
from datetime import datetime, timedelta, timezone
from gateway.config import Platform, platform_binds_port as _platform_binds_port
from gateway.platforms.base import BasePlatformAdapter
from gateway.restart import is_global_startup_conflict
from gateway.session import SessionSource
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Optional

if TYPE_CHECKING:  # string annotations only; never imported at runtime (cycle)
    from gateway.run import GatewayRunner, TurnRunner  # noqa: F401

# Log-record parity with the origin module.
logger = logging.getLogger("gateway.run")


class GatewayAdapterLifecycleMixin:
    """Adapter connect/disconnect, fatal-error recovery, reconnect watcher and multiplex profile adapter methods for GatewayRunner."""

    async def _await_adapter_cleanup_with_timeout(
        self, awaitable: Awaitable[Any], timeout: float
    ) -> bool:
        """Wait for adapter cleanup without letting cancellation swallowing hang us.

        ``asyncio.wait_for`` cancels an overdue child but then waits for it to exit. An adapter
        close path that catches ``CancelledError`` can therefore block recovery forever. Keep
        ownership of the old task through its done callback, but release the runner at the deadline.
        """
        if timeout <= 0:
            await awaitable
            return True

        task = asyncio.ensure_future(awaitable)
        try:
            done, _pending = await asyncio.wait({task}, timeout=timeout)
        except asyncio.CancelledError:
            task.cancel()
            task.add_done_callback(consume_detached_task_result)
            raise
        if task in done:
            await task
            return True

        task.cancel()
        task.add_done_callback(consume_detached_task_result)
        return False

    async def _safe_adapter_disconnect(self, adapter, platform) -> None:
        """Call adapter.disconnect() defensively, swallowing any error.

        For a failed/raised connect(): partial resources (aiohttp.ClientSession, poll tasks, child
        subprocesses) would otherwise leak. Must tolerate partial-init state and never raise.
        """
        timeout = self._adapter_disconnect_timeout_secs()
        try:
            completed = await self._await_adapter_cleanup_with_timeout(
                adapter.disconnect(), timeout
            )
            if not completed:
                logger.warning(
                    "Timed out after %.1fs while disconnecting %s adapter; continuing shutdown",
                    timeout,
                    platform.value if platform is not None else "adapter",
                )
        except Exception as e:
            logger.debug(
                "Defensive %s disconnect after failed connect raised: %s",
                platform.value if platform is not None else "adapter",
                e,
            )

    async def _bounded_adapter_teardown(
        self, adapter, platform, *, profile: Optional[str] = None
    ) -> None:
        """Tear down one adapter on the shutdown path with bounded awaits.

        ``cancel_background_tasks()`` and ``disconnect()`` can block forever on half-dead network
        state (e.g. a wedged WebSocket thread), stalling shutdown past systemd's ``TimeoutStopSec``;
        the SIGKILL skips ``atexit`` PID-file cleanup and the next start dies with "PID file race
        lost". Each await uses ``HERMES_GATEWAY_ADAPTER_DISCONNECT_TIMEOUT``; on timeout the task is
        cancelled and detached so a cancellation-swallowing adapter can't hang the loop. Never raises.
        """
        timeout = self._adapter_disconnect_timeout_secs()
        suffix = f" (profile: {profile})" if profile else ""
        started_at = time.monotonic()
        try:
            cancelled = await self._await_adapter_cleanup_with_timeout(
                adapter.cancel_background_tasks(), timeout
            )
            if not cancelled:
                logger.warning(
                    "✗ %s background-task cancel timed out after %.1fs - forcing continue%s",
                    platform.value, timeout, suffix,
                )
        except Exception as e:
            logger.debug("✗ %s background-task cancel error%s: %s", platform.value, suffix, e)
        try:
            disconnected = await self._await_adapter_cleanup_with_timeout(
                adapter.disconnect(), timeout
            )
            if disconnected:
                logger.info(
                    "✓ %s disconnected (%.2fs)%s",
                    platform.value, time.monotonic() - started_at, suffix,
                )
            else:
                logger.warning(
                    "✗ %s disconnect timed out after %.1fs - forcing continue%s",
                    platform.value, timeout, suffix,
                )
        except Exception as e:
            logger.error(
                "✗ %s disconnect error after %.2fs%s: %s",
                platform.value, time.monotonic() - started_at, suffix, e,
            )

    def _adapter_disconnect_timeout_secs(self) -> float:
        """Return the per-adapter disconnect timeout used during shutdown."""
        from gateway.run import _ADAPTER_DISCONNECT_TIMEOUT_SECS_DEFAULT
        raw = os.getenv("HERMES_GATEWAY_ADAPTER_DISCONNECT_TIMEOUT", "").strip()
        if raw:
            try:
                timeout = float(raw)
            except ValueError:
                logger.warning(
                    "Ignoring invalid HERMES_GATEWAY_ADAPTER_DISCONNECT_TIMEOUT=%r",
                    raw,
                )
            else:
                return max(0.0, timeout)
        return _ADAPTER_DISCONNECT_TIMEOUT_SECS_DEFAULT

    def _platform_connect_timeout_secs(self, platform=None, *, initial: bool = False) -> float:
        """Return the per-platform connect timeout used during startup/retry.

        Telegram's full 180s connect budget is deliberately NOT spent at cold start: an unreachable
        Telegram would hold the gateway out of ``running`` for the whole budget. The cold-start wait
        is capped and the platform handed to the reconnect watcher, which retries with the full
        budget and ``is_reconnect=True`` (preserving the offline update queue).
        """
        from gateway.run import (
            _PLATFORM_CONNECT_TIMEOUT_SECS_DEFAULT,
            _TELEGRAM_CONNECT_TIMEOUT_SECS_DEFAULT,
            _TELEGRAM_INITIAL_CONNECT_TIMEOUT_SECS_DEFAULT,
        )
        raw = os.getenv("HERMES_GATEWAY_PLATFORM_CONNECT_TIMEOUT", "").strip()
        if raw:
            try:
                timeout = float(raw)
            except ValueError:
                logger.warning(
                    "Ignoring invalid HERMES_GATEWAY_PLATFORM_CONNECT_TIMEOUT=%r",
                    raw,
                )
            else:
                return max(0.0, timeout)
        if platform == Platform.TELEGRAM:
            if initial:
                return _TELEGRAM_INITIAL_CONNECT_TIMEOUT_SECS_DEFAULT
            return _TELEGRAM_CONNECT_TIMEOUT_SECS_DEFAULT
        return _PLATFORM_CONNECT_TIMEOUT_SECS_DEFAULT

    async def _connect_adapter_with_timeout(
        self, adapter, platform, *, is_reconnect: bool = False, initial: bool = False
    ) -> bool:
        """Connect an adapter without allowing one platform to block others.

        ``is_reconnect`` lets adapters distinguish a cold first boot (drop any stale server-side
        queue) from a watcher reconnect (preserve the queue so interim messages aren't dropped).
        ``initial`` selects the capped cold-start budget for platforms whose full connect budget is
        too long to spend before the gateway reaches ``running`` (Telegram's 180s).
        """
        timeout = self._platform_connect_timeout_secs(platform, initial=initial)
        if timeout <= 0:
            return await adapter.connect(is_reconnect=is_reconnect)
        # Detach-on-timeout rather than plain asyncio.wait_for: wait_for cancels the overdue task but
        # then waits for it to exit, so a connect() that catches CancelledError blocks recovery
        # forever (watcher never retries). Keep ownership via its done callback; release at deadline.
        task = asyncio.ensure_future(
            adapter.connect(is_reconnect=is_reconnect)
        )
        try:
            done, _pending = await asyncio.wait({task}, timeout=timeout)
        except asyncio.CancelledError:
            task.cancel()
            task.add_done_callback(consume_detached_task_result)
            raise
        if task in done:
            result = await task
            return bool(result)
        task.cancel()
        task.add_done_callback(consume_detached_task_result)
        raise TimeoutError(
            f"{platform.value} connect timed out after {timeout:g}s"
        )

    async def _connect_initial_adapter_with_timeout(self, adapter, platform) -> bool:
        """Connect one cold-start adapter with tightly scoped replace intent.

        The capability is visible only while this initial connect is awaited. Reconnects call
        ``_connect_adapter_with_timeout`` directly and adapters also default to deny, so a later
        network recovery can never evict a healthy token holder.
        """
        adapter._platform_lock_takeover_allowed = bool(
            self._platform_lock_takeover_on_start
        )
        try:
            return await self._connect_adapter_with_timeout(
                adapter, platform, initial=True
            )
        finally:
            adapter._platform_lock_takeover_allowed = False

    async def _handle_reaction_event(self, ctx: Dict[str, Any]) -> None:
        """Fan a normalised platform reaction event out to the HookRegistry.

        The adapter-supplied ``event_name`` ("reaction:added"/"reaction:removed") is the hook event,
        matching the ``agent:*`` naming scheme. Errors never block the adapter's event loop.
        """
        event_name = str(ctx.get("event_name") or "reaction:added")
        try:
            await self.hooks.emit(event_name, ctx)
        except Exception:
            logger.debug("[Gateway] reaction hook emit failed", exc_info=True)

    async def _handle_adapter_fatal_error(self, adapter: BasePlatformAdapter) -> None:
        """React to an adapter failure after startup.

        Retryable errors (network blip, DNS) queue the platform for background reconnection.
        The notification arrives on the failing adapter's own polling task, and the disconnect in
        the handler can cancel that task mid-flight (disconnect()'s current-task guard misses it
        because _safe_adapter_disconnect closes in a wrapper task), stranding the platform between
        the fatal log and the reconnect queue — so the real work runs in a detached task.
        """
        tasks = getattr(self, "_fatal_handler_tasks", None)
        if tasks is None:
            tasks = self._fatal_handler_tasks = set()
        task = asyncio.create_task(self._handle_adapter_fatal_error_detached(adapter))
        tasks.add(task)
        task.add_done_callback(tasks.discard)
        # Await so callers that expect completion still get it — but through shield(): Task.cancel()
        # on the caller also cancels the future it is awaiting (_fut_waiter), so a plain `await
        # task` would tunnel the cancellation straight into the "detached" task. shield() absorbs
        # it: the caller sees CancelledError, the handler runs to completion.
        await asyncio.shield(task)

    def _queue_retryable_fatal_platform(self, adapter: BasePlatformAdapter) -> bool:
        """Queue a retryable fatal adapter for background reconnection.

        Returns True when newly queued; idempotent if already queued. Must not await: callers
        invoke this *before* any disconnect await so a wedged close cannot strand the platform.
        """
        if not adapter.fatal_error_retryable:
            return False
        platform_config = self.config.platforms.get(adapter.platform)
        if not platform_config:
            return False
        if adapter.platform in self._failed_platforms:
            # Nothing to enqueue — but "already queued" is exactly when the watcher may have died,
            # and the enqueue branch below holds the ONLY _ensure_reconnect_watcher_running() call.
            # _spawn_supervised gives up after _MAX_SUPERVISED_RESTARTS; without this backstop a
            # queued platform is a silent permanent outage (nothing retries, and the stranded check
            # treats a queued platform as safe so the process never restarts either).
            self._ensure_reconnect_watcher_running()
            return False
        self._failed_platforms[adapter.platform] = {
            "config": platform_config,
            "attempts": 0,
            "next_retry": time.monotonic(),
            "queued_at": time.monotonic(),
            "credential_claim": self._adapter_credential_claim(
                adapter.platform, adapter
            ),
            "listener_claim": self._adapter_listener_claim(
                adapter.platform, adapter
            ),
        }
        logger.info(
            "%s queued for background reconnection",
            adapter.platform.value,
        )
        # Ensure the reconnect watcher is alive — respawn if it died (e.g. restart budget exhausted)
        # so queued platforms are not permanently stranded.
        self._ensure_reconnect_watcher_running()
        return True

    async def _handle_adapter_fatal_error_detached(
        self, adapter: BasePlatformAdapter
    ) -> None:
        """Run the fatal handler; if the platform still ends up stranded (not reconnected, not
        queued, not intentionally disabled), exit the gateway with failure so the service manager
        restarts it instead of leaving a silent partial outage."""
        try:
            # Outer hard deadline: even with queue-before-disconnect, a hang anywhere in the impl
            # (status write side effects, detach races, etc.) must not leave this task wedged
            # forever — the stranded check in ``finally`` only runs when we return.
            timeout = self._adapter_disconnect_timeout_secs()
            if timeout <= 0:
                await self._handle_adapter_fatal_error_impl(adapter)
            else:
                # Disconnect budget plus a little queue/status bookkeeping overhead; keep the extra
                # proportional so tests that shrink the disconnect timeout still finish promptly.
                outer = timeout + min(2.0, max(0.05, timeout))
                completed = await self._await_adapter_cleanup_with_timeout(
                    self._handle_adapter_fatal_error_impl(adapter),
                    outer,
                )
                if not completed:
                    logger.error(
                        "Fatal-error handling for %s timed out after %.1fs; "
                        "ensuring reconnect queue is populated",
                        adapter.platform.value,
                        outer,
                    )
                    self._queue_retryable_fatal_platform(adapter)
        except asyncio.CancelledError:
            # Best-effort queue before re-raising: a cancelled fatal handler
            # must not strand a retryable platform (#80598).
            try:
                self._queue_retryable_fatal_platform(adapter)
            except Exception:
                logger.debug(
                    "Failed to queue %s after fatal-handler cancellation",
                    adapter.platform.value,
                    exc_info=True,
                )
            raise
        except Exception:
            logger.exception(
                "Fatal-error handling for %s raised unexpectedly",
                adapter.platform.value,
            )
            # Best-effort queue so an unexpected raise mid-handler cannot
            # leave a retryable platform permanently deaf (#80598).
            try:
                self._queue_retryable_fatal_platform(adapter)
            except Exception:
                logger.debug(
                    "Failed to queue %s after fatal-handler exception",
                    adapter.platform.value,
                    exc_info=True,
                )
        finally:
            platform = adapter.platform
            shutdown_event = getattr(self, "_shutdown_event", None)
            stranded = (
                adapter.fatal_error_retryable
                and platform not in self.adapters
                and platform not in getattr(self, "_failed_platforms", {})
                and not (shutdown_event is not None and shutdown_event.is_set())
            )
            if stranded:
                logger.error(
                    "%s adapter was lost without entering the reconnection "
                    "queue; exiting gateway so the service manager restarts it.",
                    platform.value,
                )
                self._exit_reason = (
                    f"{platform.value} adapter lost without reconnection queue"
                )
                self._exit_with_failure = True
                await self.stop()

    async def _handle_adapter_fatal_error_impl(self, adapter: BasePlatformAdapter) -> None:
        # Snapshot this platform slot's current owner first: acting on a stale notification would
        # overwrite a healthy platform's runtime status and wrongly re-queue it for reconnection.
        existing = self.adapters.get(adapter.platform)
        if existing is not None and existing is not adapter:
            logger.debug(
                "Ignoring stale fatal error from a superseded %s adapter instance: %s",
                adapter.platform.value,
                adapter.fatal_error_code or "unknown",
            )
            return

        logger.error(
            "Fatal %s adapter error (%s): %s",
            adapter.platform.value,
            adapter.fatal_error_code or "unknown",
            adapter.fatal_error_message or "unknown error",
        )
        # A relay credential revoked by opt-out is not an error to retry: render a clean "disabled"
        # state, not red "fatal"/"retrying" (non-retryable code, so it also leaves the queue below).
        if adapter.fatal_error_code == "relay_disabled":
            platform_state = "disabled"
        elif adapter.fatal_error_retryable:
            platform_state = "retrying"
        else:
            platform_state = "fatal"
        self._update_platform_runtime_status(
            adapter.platform.value,
            platform_state=platform_state,
            error_code=adapter.fatal_error_code,
            error_message=adapter.fatal_error_message,
        )

        if existing is adapter:
            # Claim this adapter for teardown before awaiting disconnect(): a second fatal-error
            # notification for the same adapter (e.g. a concurrent recovery path) would otherwise
            # still see itself as "existing" during the await and disconnect() the same object twice.
            self.adapters.pop(adapter.platform, None)
            self.delivery_router.adapters = self.adapters

        # Queue retryable failures BEFORE any disconnect await: a half-dead transport can wedge
        # native close() (or swallow CancelledError), so "disconnect then queue" left platforms
        # permanently deaf in a live process after the network recovered. Populate the queue first so the
        # reconnect watcher always has work; teardown is best-effort after.
        self._queue_retryable_fatal_platform(adapter)

        if existing is adapter:
            # A half-closed transport can wedge native close() indefinitely; reuse the shutdown-path
            # timeout so this runtime fatal handler always returns to the stay-alive / stranded path.
            await self._safe_adapter_disconnect(adapter, adapter.platform)

        if not self.adapters and not self._failed_platforms:
            self._exit_reason = adapter.fatal_error_message or "All messaging adapters disconnected"
            if adapter.fatal_error_retryable:
                self._exit_with_failure = True
                logger.error("No connected messaging platforms remain. Shutting down gateway for service restart.")
            else:
                logger.error("No connected messaging platforms remain. Shutting down gateway cleanly.")
            await self.stop()
        elif not self.adapters and self._failed_platforms:
            # All platforms are down and queued for reconnection. Keep the gateway alive so cron jobs
            # still run and the watcher can recover platforms when the problem clears; exiting for a
            # systemd restart would turn a transient outage into a state-killing restart loop.
            logger.warning(
                "No connected messaging platforms remain, but %d platform(s) "
                "queued for reconnection — gateway staying alive, watcher will "
                "retry in background.",
                len(self._failed_platforms),
            )

    def _request_clean_exit(self, reason: str) -> None:
        self._exit_cleanly = True
        self._exit_reason = reason
        self._shutdown_event.set()

    @staticmethod
    def _supervised_backoff(attempt: int) -> float:
        """Delay before the supervisor's next respawn, in seconds (capped exponential).

        A method so tests can collapse the schedule instead of sleeping through the real curve.
        """
        return min(60, 2 ** min(attempt, 6))

    def _spawn_supervised(
        self, coro_factory, name, *, restart=True, _attempt=0, on_spawn=None,
        on_give_up=None,
    ):
        """Launch a long-lived background task with task-level supervision.

        Catches what a per-iteration try/except cannot — exceptions in the OUTER loop or pre-try
        setup — which a bare ``asyncio.create_task`` drops silently. Restarts with capped backoff up
        to ``_MAX_SUPERVISED_RESTARTS`` rapid failures; the counter resets after a run healthy for
        ``_SUPERVISED_HEALTHY_SECS``. Each spawn uses a fresh ``Context``: an inherited
        delegated-child marker would make the Kanban dispatcher reject its own writes.
        ``on_spawn`` fires on EVERY spawn incl. respawns; callers tracking the handle elsewhere
        (e.g. ``_reconnect_watcher_task``) MUST pass it or a respawn leaves a stale handle and a
        SECOND watcher. ``on_give_up(name)`` fires when the restart budget is spent.
        """
        if getattr(self, "_background_tasks", None) is None:
            self._background_tasks = set()

        # Monotonic spawn timestamp captured per spawn: the ``_done`` callback
        # uses it to distinguish a rapid crash-loop from a healthy-run-then-crash.
        _started = time.monotonic()

        # Deliberately no kwargs to create_task (some test doubles mock a narrow signature); calling
        # it from a fresh Context gives the same isolation as create_task(..., context=Context()).
        task = Context().run(lambda: asyncio.create_task(coro_factory()))
        # PERMANENT supervised watcher, not transient background WORK: the scale-to-zero idle check
        # must ignore process-lifetime watchers or the gateway counts itself busy forever. Transient
        # tasks added to _background_tasks elsewhere (startup-resume events etc.) stay counted.
        task._hermes_supervised_watcher = True  # type: ignore[attr-defined]
        self._background_tasks.add(task)
        if on_spawn is not None:
            # Record the live handle NOW so an external tracker (e.g. _reconnect_watcher_task)
            # points at the current task, not a dead one left by a prior supervised respawn.
            try:
                on_spawn(task)
            except Exception:  # pragma: no cover - defensive; a tracker must never kill the spawn
                logger.debug("on_spawn callback for %s raised", name, exc_info=True)

        def _done(t):
            self._background_tasks.discard(t)
            if t.cancelled():
                return
            exc = t.exception()
            if exc is None:
                # Clean return == deliberate shutdown or a self-disabling watcher (e.g. a gated
                # no-op returning at once); respawning would busy-spin it — NEVER restart on it.
                return
            logger.error("Supervised task %s died: %r", name, exc, exc_info=exc)
            if restart and self._running:
                ran_for = time.monotonic() - _started
                if ran_for >= self._SUPERVISED_HEALTHY_SECS:
                    # Ran healthily before crashing — a FRESH failure, not a rapid crash-loop. Reset
                    # the counter so a daemon crashing a few times over days is never abandoned.
                    effective_attempt = 0
                else:
                    effective_attempt = _attempt
                if effective_attempt >= self._MAX_SUPERVISED_RESTARTS:
                    logger.error(
                        "Supervised task %s died %d times in rapid succession "
                        "(each within %ds of restart) — giving up restarts",
                        name,
                        effective_attempt,
                        self._SUPERVISED_HEALTHY_SECS,
                    )
                    if on_give_up is not None:
                        try:
                            on_give_up(name)
                        except Exception:  # pragma: no cover - defensive
                            logger.debug(
                                "on_give_up callback for %s raised",
                                name, exc_info=True,
                            )
                    return
                backoff = self._supervised_backoff(effective_attempt)

                async def _respawn():
                    await asyncio.sleep(backoff)
                    if self._running:
                        self._spawn_supervised(
                            coro_factory,
                            name,
                            restart=restart,
                            _attempt=effective_attempt + 1,
                            on_spawn=on_spawn,
                            # Threaded through the recursion like on_spawn: only the LAST respawn's give-up
                            # matters, and dropping the callback leaves the exhaustion branch with no owner.
                            on_give_up=on_give_up,
                        )

                # The done callback retains its registration context, so isolate the backoff task
                # too; otherwise a restart could reintroduce the original caller's turn scope.
                respawn_task = Context().run(lambda: asyncio.create_task(_respawn()))
                self._background_tasks.add(respawn_task)
                respawn_task.add_done_callback(self._background_tasks.discard)

        task.add_done_callback(_done)
        return task

    async def _handoff_watcher(
        self, interval: float = 2.0, drain_timeout: float = 30.0,
    ) -> None:
        """Background task that processes pending CLI→gateway session handoffs.

        Polls ``state.db`` for ``handoff_state='pending'`` rows: claim atomically (pending →
        running), re-bind the home channel's session_key to the CLI session_id via
        ``switch_session``, dispatch a synthetic ``MessageEvent``, mark ``completed``/``failed``.
        """
        from gateway.run import _async_profile_runtime_scope, _handoff_watch_scopes, _reclaim_stale
        # Initial delay so the gateway is fully connected to its platforms
        # before we try to dispatch handoffs through them.
        await asyncio.sleep(5)

        # Does _process_handoff accept the profile argument? The real one does; test stand-ins bind
        # a one-parameter callable. Probed once, outside the loop.
        try:
            import inspect as _inspect
            _process_takes_profile = len(
                _inspect.signature(self._process_handoff).parameters
            ) >= 2
        except Exception:
            _process_takes_profile = False

        # In-flight dispatches keyed by session id. A handoff runs a FULL agent turn plus delivery
        # (far longer than the CLI's 60s wait); inline processing would let one slow handoff block
        # every other profile's poll and time them out. Fire-and-forget; the poll loop only claims.
        inflight: Dict[str, "asyncio.Task"] = {}

        async def _dispatch(row, session_id, session_db, profile_name) -> None:
            """Run one claimed handoff to a terminal state, off the poll path."""
            try:
                if _process_takes_profile:
                    await self._process_handoff(row, profile_name)
                else:
                    await self._process_handoff(row)
                await session_db.complete_handoff(session_id)
            except asyncio.CancelledError:
                # Gateway shutting down: leave the row 'running' so the next
                # start's reclaim marks it failed with a clear reason.
                raise
            except Exception as exc:
                logger.warning(
                    "Handoff for session %s failed: %s",
                    session_id, exc, exc_info=True,
                )
                try:
                    await session_db.fail_handoff(session_id, str(exc))
                except Exception:
                    logger.debug("Could not record handoff failure", exc_info=True)
            finally:
                inflight.pop(session_id, None)

        async def _tick(profile_name: Optional[str] = None) -> None:
            """One poll of the CURRENTLY-SCOPED session store.

            A closure over ``self``, not a method: unit tests bind ``_handoff_watcher`` onto a
            ``SimpleNamespace`` exposing only ``_session_db``, ``_running`` and ``_process_handoff``;
            any other ``self.<attr>`` would raise, be swallowed by the loop, and silently no-op the
            watcher. ``profile_name`` (``None`` = root) makes delivery use that profile's OWN adapter.
            """
            session_db = getattr(self, "_session_db", None)
            if session_db is None:
                return
            pending = await session_db.list_pending_handoffs()
            for row in pending:
                session_id = row.get("id")
                if not session_id or session_id in inflight:
                    continue
                if not await session_db.claim_handoff(session_id):
                    # Another tick or another gateway already claimed it.
                    continue
                # Positional, not keyword: tests bind a one-arg ``_process_handoff(row)`` stand-in and a
                # keyword call would TypeError into the failure branch (arity probed above).
                # INVARIANT (do not weaken): this task is created inside _profile_runtime_scope but
                # typically RUNS after it exits; it sees the profile's home/secret scope only because
                # those seams are ContextVar-based and ensure_future copies the Context.
                inflight[session_id] = asyncio.ensure_future(
                    _dispatch(row, session_id, session_db, profile_name)
                )

        # A row still 'running' at startup belongs to a gateway that died mid-dispatch: it can never
        # reach a terminal state, and request_handoff refuses new requests while it sits there.
        for _pname, _phome in _handoff_watch_scopes(self):
            try:
                if _phome is None:
                    await _reclaim_stale(self)
                else:
                    async with _async_profile_runtime_scope(_phome):
                        await _reclaim_stale(self)
            except Exception:
                logger.debug("Stale-handoff reclaim failed", exc_info=True)

        try:
            while self._running:
                try:
                    for profile_name, profile_home in _handoff_watch_scopes(self):
                        if profile_home is None:
                            await _tick(profile_name)
                        else:
                            async with _async_profile_runtime_scope(profile_home):
                                await _tick(profile_name)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.debug("Handoff watcher tick error: %s", exc, exc_info=True)
                await asyncio.sleep(interval)
        finally:
            # Drain in-flight dispatches before returning: cancelling would strand their rows in
            # 'running'; a bounded grace period lets an almost-done handoff record its own state.
            pending_tasks = [t for t in inflight.values() if not t.done()]
            if pending_tasks:
                try:
                    await asyncio.wait(pending_tasks, timeout=drain_timeout)
                except Exception:
                    logger.debug("Handoff drain raised", exc_info=True)
                for task in pending_tasks:
                    if not task.done():
                        task.cancel()

    def _on_reconnect_watcher_gave_up(self, name: str = "") -> None:
        """Own the reconnect invariant once supervision has abandoned it.

        Invariant: while running and ``_failed_platforms`` is non-empty, a reconnect watcher is live
        or a bounded respawn is scheduled. Event-coupled recovery is not enough: the failed adapter
        is dropped from the live map, so no later event may ever arrive to notice a dead watcher.
        Deliberately NOT done here: requesting a process restart when the slow tier is exhausted —
        a blast-radius policy call; a single loud error names the still-queued platforms instead.
        """
        if not getattr(self, "_running", False):
            return
        if not getattr(self, "_failed_platforms", None):
            # No queued work depends on the watcher; leaving it dead is correct — the enqueue path
            # spawns a fresh one the moment a platform is queued again.
            logger.warning(
                "Reconnect watcher supervision exhausted with an empty retry "
                "queue — leaving it down until a platform is queued."
            )
            return
        self._schedule_slow_reconnect_watcher_respawn(attempt=0)

    def _schedule_slow_reconnect_watcher_respawn(self, *, attempt: int) -> None:
        """Bounded slow-tier respawn of the reconnect watcher."""
        if attempt >= self._MAX_SLOW_WATCHER_RESPAWNS:
            logger.error(
                "Reconnect watcher could not be kept alive after %d slow "
                "respawns; %d platform(s) remain queued and unattended: %s. "
                "Manual intervention or a gateway restart is required.",
                attempt,
                len(self._failed_platforms),
                ", ".join(str(p) for p in self._failed_platforms),
            )
            return

        async def _slow_respawn() -> None:
            await asyncio.sleep(self._RECONNECT_WATCHER_SLOW_RETRY_SECS)
            if not getattr(self, "_running", False):
                return
            if not getattr(self, "_failed_platforms", None):
                # The queue drained while we waited -- something else healed
                # it. Nothing to own any more.
                return
            task = getattr(self, "_reconnect_watcher_task", None)
            if task is not None and not task.done():
                return  # a watcher came back on its own; stand down
            logger.warning(
                "Reconnect watcher still down with %d platform(s) queued — "
                "slow respawn %d/%d",
                len(self._failed_platforms),
                attempt + 1,
                self._MAX_SLOW_WATCHER_RESPAWNS,
            )
            self._spawn_reconnect_watcher(
                on_give_up=lambda _name: self._schedule_slow_reconnect_watcher_respawn(
                    attempt=attempt + 1
                )
            )

        respawn_task = asyncio.create_task(_slow_respawn())
        if getattr(self, "_background_tasks", None) is None:
            self._background_tasks = set()
        self._background_tasks.add(respawn_task)
        respawn_task.add_done_callback(self._background_tasks.discard)

    def _spawn_reconnect_watcher(self, *, on_give_up=None):
        """Single place that knows how to launch the reconnect watcher.

        ``on_spawn`` is load-bearing: without it the supervisor's own respawn leaves
        ``_reconnect_watcher_task`` at a dead handle and ``_ensure_...`` spawns a second watcher.
        """
        self._reconnect_watcher_task = self._spawn_supervised(
            self._platform_reconnect_watcher,
            "platform_reconnect_watcher",
            on_spawn=lambda t: setattr(self, "_reconnect_watcher_task", t),
            on_give_up=on_give_up or self._on_reconnect_watcher_gave_up,
        )
        return self._reconnect_watcher_task

    def _ensure_reconnect_watcher_running(self) -> None:
        """Ensure the platform reconnect watcher background task is alive.

        Respawns a dead watcher (exhausted restart budget, unrecoverable exception) so queued
        platforms are not stranded. Called on BOTH _queue_retryable_fatal_platform paths: the
        re-fatal of an already-queued platform is the only case where the budget can be exhausted.
        """
        if not getattr(self, "_running", False):
            return
        task = getattr(self, "_reconnect_watcher_task", None)
        if task is not None and not task.done():
            return  # already alive
        logger.warning(
            "Reconnect watcher task is dead (done=%s) — respawning",
            task.done() if task is not None else "N/A",
        )
        self._spawn_reconnect_watcher()

    async def _platform_reconnect_watcher(self) -> None:
        """Background task that periodically retries connecting failed platforms.

        Exponential backoff 30s → 300s cap; retryable failures (network/DNS) retry at the cap
        indefinitely so transient outages self-heal, non-retryable (bad auth) drop out immediately.
        The circuit breaker (``/platform pause``) is manual only — auto-pausing left bots dead.
        """
        from gateway.run import (
            _dispose_unused_adapter,
            _platform_has_bot_credential,
            _reconnect_backoff,
            _reconnect_needs_attention,
        )
        await asyncio.sleep(10)  # initial delay — let startup finish
        while self._running:
            if not self._failed_platforms:
                # Nothing to reconnect — sleep and check again
                for _ in range(30):
                    if not self._running:
                        return
                    if self._failed_platforms:
                        break
                    await asyncio.sleep(1)
                continue

            now = time.monotonic()
            for platform in list(self._failed_platforms.keys()):
                if not self._running:
                    return
                info = self._failed_platforms.get(platform)
                if info is None:
                    # Removed concurrently (/platform resume, reconnect via another path) between
                    # the snapshot above and this lookup — not an error, nothing to do this pass.
                    continue
                # Skip paused platforms entirely — they need explicit
                # /platform resume to come back.
                if info.get("paused"):
                    continue
                # Long-lived retry escalation: past the attention threshold flag the platform
                # NEEDS_ATTENTION in runtime status so a dead token/revoked intent doesn't look
                # like ordinary "retrying" forever. A signal, NOT a circuit breaker — retries continue.
                if not info.get("attention_flagged") and _reconnect_needs_attention(info, now):
                    info["attention_flagged"] = True
                    queued_for = now - info.get("queued_at", now)
                    retrying_since_iso = (
                        datetime.now(timezone.utc) - timedelta(seconds=queued_for)
                    ).isoformat()
                    logger.warning(
                        "%s has been failing/reconnecting continuously for "
                        "%.1f hours (%d attempts) — flagging NEEDS_ATTENTION. "
                        "Retries continue, but this usually means a permanent "
                        "problem (revoked credentials, missing intents, broken "
                        "sidecar). Check `hermes status` / `/platform list`.",
                        platform.value,
                        queued_for / 3600.0,
                        info.get("attempts", 0),
                    )
                    self._update_platform_runtime_status(
                        platform.value,
                        platform_state="retrying",
                        needs_attention=True,
                        retrying_since=retrying_since_iso,
                    )
                if now < info["next_retry"]:
                    continue  # not time yet

                platform_config = info["config"]
                attempt = info["attempts"] + 1
                # Empty-token primary configs can never reconnect; drop them so multiplex setups
                # where a secondary profile owns the bot do not spin forever.
                if not _platform_has_bot_credential(platform, platform_config):
                    logger.warning(
                        "Reconnect %s: no bot credential on queued config, "
                        "removing from retry queue",
                        platform.value,
                    )
                    del self._failed_platforms[platform]
                    continue
                logger.info(
                    "Reconnecting %s (attempt %d)...",
                    platform.value, attempt,
                )

                adapter = None
                try:
                    adapter = self._create_adapter(platform, platform_config)
                    if not adapter:
                        logger.warning(
                            "Reconnect %s: adapter creation returned None, removing from retry queue",
                            platform.value,
                        )
                        del self._failed_platforms[platform]
                        continue

                    adapter.set_message_handler(self._primary_message_handler())
                    adapter.set_fatal_error_handler(self._handle_adapter_fatal_error)
                    adapter.set_session_store(self.session_store)
                    adapter.set_busy_session_handler(self._handle_active_session_busy_message)
                    _set_reaction = getattr(adapter, "set_reaction_handler", None)
                    if callable(_set_reaction):
                        _set_reaction(self._handle_reaction_event)
                    adapter.set_topic_recovery_fn(self._recover_telegram_topic_thread_id)
                    adapter.set_authorization_check(self._make_adapter_auth_check(adapter.platform))
                    adapter.set_platform_event_handler(self._primary_platform_event_handler())
                    adapter._busy_text_mode = self._busy_text_mode

                    # Reconnect after outage: keep the platform's server-side update queue so
                    # messages sent while the bot was offline are delivered rather than dropped.
                    success = await self._connect_adapter_with_timeout(
                        adapter, platform, is_reconnect=True
                    )
                    if success:
                        self.adapters[platform] = adapter
                        self._sync_voice_mode_state_to_adapter(adapter)
                        # Wire voice input callback on reconnect as well (#60623).
                        self._bind_voice_input_callback(adapter)
                        self.delivery_router.adapters = self.adapters
                        del self._failed_platforms[platform]
                        self._update_platform_runtime_status(
                            platform.value,
                            platform_state="connected",
                            error_code=None,
                            error_message=None,
                            needs_attention=False,
                            retrying_since=None,
                        )
                        logger.info("✓ %s reconnected successfully", platform.value)

                        # Final responses rejected while this adapter was down are still owned by
                        # this live process, so startup recovery cannot claim them. Replay the
                        # explicitly transient subset now that the platform is usable.
                        try:
                            await self._redeliver_failed_obligations_for_platform(
                                platform
                            )
                        except Exception:
                            logger.debug(
                                "failed-obligation redelivery after %s reconnect failed",
                                platform.value,
                                exc_info=True,
                            )

                        # Rebuild channel directory with the new adapter
                        try:
                            from gateway.channel_directory import build_channel_directory
                            await build_channel_directory(self.adapters)
                        except Exception:
                            pass

                        # A platform that was offline at gateway startup never got its restart-
                        # interrupted sessions auto-resumed — the startup pass skips sessions whose
                        # adapter isn't connected yet.
                        try:
                            self._schedule_resume_pending_sessions(platform=platform)
                        except Exception:
                            logger.debug(
                                "resume-pending reschedule after %s reconnect failed",
                                platform.value,
                                exc_info=True,
                            )
                    # Check if the failure is non-retryable
                    elif adapter.has_fatal_error and not adapter.fatal_error_retryable:
                        self._update_platform_runtime_status(
                            platform.value,
                            platform_state="fatal",
                            error_code=adapter.fatal_error_code,
                            error_message=adapter.fatal_error_message,
                        )
                        logger.warning(
                            "Reconnect %s: non-retryable error (%s), removing from retry queue",
                            platform.value, adapter.fatal_error_message,
                        )
                        # The adapter is about to be dropped from the queue without ever being
                        # installed on self.adapters, so nothing else will call disconnect() on it.
                        # Dispose here or the resource owners built in __init__ (ResponseStore etc.)
                        # leak ~2 fds each; at the 300s cap the gateway hits the fd limit in ~12h.
                        await _dispose_unused_adapter(adapter)
                        del self._failed_platforms[platform]
                    else:
                        self._update_platform_runtime_status(
                            platform.value,
                            platform_state="retrying",
                            error_code=adapter.fatal_error_code,
                            error_message=adapter.fatal_error_message or "failed to reconnect",
                        )
                        backoff = _reconnect_backoff(attempt)
                        info["attempts"] = attempt
                        info["next_retry"] = time.monotonic() + backoff
                        logger.info(
                            "Reconnect %s failed, next retry in %ds",
                            platform.value, backoff,
                        )
                        # Same fd-leak concern as the non-retryable branch above: the adapter failed
                        # to connect and is being thrown away.
                        await _dispose_unused_adapter(adapter)
                        # Retryable failures (network/DNS blips) retry at the backoff cap forever,
                        # self-healing when connectivity returns. Never auto-pause them: a transient
                        # outage must not need `/platform resume`. Everything here is retryable.
                except Exception as e:
                    if adapter is not None:
                        # An exception escaping connect (DNS timeout, aiohttp server.start() crash,
                        # etc.) leaves the adapter in the same unowned state as the branches above.
                        await _dispose_unused_adapter(adapter)
                    self._update_platform_runtime_status(
                        platform.value,
                        platform_state="retrying",
                        error_code=None,
                        error_message=str(e),
                    )
                    backoff = _reconnect_backoff(attempt)
                    info["attempts"] = attempt
                    info["next_retry"] = time.monotonic() + backoff
                    logger.warning(
                        "Reconnect %s error: %s, next retry in %ds",
                        platform.value, e, backoff,
                    )
                    # A reconnect exception (connect timeout, DNS failure, ...) is transient; keep
                    # retrying at the backoff cap rather than auto-pausing.

            # Check every 10 seconds for platforms that need reconnection
            for _ in range(10):
                if not self._running:
                    return
                await asyncio.sleep(1)

    async def _cancel_secondary_profile_reconnect_tasks(self) -> None:
        """Cancel profile-scoped reconnects before tearing down their registry.

        A reconnect can be waiting in adapter setup while shutdown begins. It must not republish
        an adapter after the secondary registry is drained. Waiting is bounded by the adapter-
        cleanup budget; a task that overruns is still blocked by the stopped runner state.
        """
        pending = self._profile_failed_platforms
        if not isinstance(pending, dict):
            return
        current = asyncio.current_task()
        tasks: list[asyncio.Task] = []
        for profile_pending in pending.values():
            if not isinstance(profile_pending, dict):
                continue
            for task in profile_pending.values():
                if isinstance(task, asyncio.Task) and task is not current and not task.done():
                    tasks.append(task)
        for task in tasks:
            task.cancel()
        timeout = self._adapter_disconnect_timeout_secs()
        if tasks and timeout > 0:
            _done, unfinished = await asyncio.wait(tasks, timeout=timeout)
            if unfinished:
                logger.warning(
                    "Timed out waiting for %d secondary profile reconnect task(s) during shutdown",
                    len(unfinished),
                )
        pending.clear()

    async def _start_secondary_profile_adapters(self) -> int:
        """Bring up adapters for every non-active profile this gateway serves.

        Returns the count of connected secondary adapters; 0 unless ``gateway.multiplex_profiles``.
        Each profile's adapters connect under its HERMES_HOME + secret scope, live in
        ``self._profile_adapters[profile]``, and get a handler stamping ``source.profile``. Same-
        platform credential collisions are refused here — the only point seeing every profile's
        resolved credentials together.
        """
        from gateway.run import (
            MultiplexConfigError,
            SecondaryPortBindingConfigError,
            _multiplex_profile_homes,
        )
        if not getattr(self.config, "multiplex_profiles", False):
            return 0

        try:
            from hermes_cli.profiles import get_active_profile_name
        except Exception:
            return 0

        active = get_active_profile_name() or "default"
        connected = 0
        # Resource claim -> owning profile. Credential claims stop two profiles polling the same
        # account; listener claims stop sidecars with distinct credentials binding one endpoint.
        claimed: Dict[tuple, str] = {}
        for _plat, _ad in self.adapters.items():
            fp = self._adapter_credential_fingerprint(_ad)
            if fp is not None:
                claimed[(_plat, fp)] = active
            listener_claim = self._adapter_listener_claim(_plat, _ad)
            if listener_claim is not None:
                claimed[listener_claim] = active
        # A retryable primary still owns its credential and listener; reserve both while queued
        # so a secondary cannot take the endpoint before the reconnect watcher retries it.
        for retry_info in getattr(self, "_failed_platforms", {}).values():
            for claim_name in ("credential_claim", "listener_claim"):
                retry_claim = retry_info.get(claim_name)
                if isinstance(retry_claim, tuple):
                    claimed[retry_claim] = active

        profile_homes = _multiplex_profile_homes(self.config)
        for profile_name, profile_home in profile_homes:
            if profile_name == active:
                continue  # handled by the primary startup loop
            try:
                connected += await self._start_one_profile_adapters(
                    profile_name, profile_home, claimed
                )
            except SecondaryPortBindingConfigError as e:
                logger.warning(
                    "Skipping secondary profile '%s' due to port-binding config error: %s",
                    profile_name,
                    e,
                )
            except MultiplexConfigError:
                raise
            except Exception as e:
                logger.error(
                    "Failed to start adapters for profile '%s': %s",
                    profile_name, e, exc_info=True,
                )

        # Record the authoritative served set in runtime status for `hermes status`. "Served"
        # means eligible for shared routing, HTTP prefixes, cron, and profile runtime scope —
        # intentionally broader than profiles with a connected (or any) secondary adapter.
        try:
            from gateway.status import write_runtime_status
            from gateway.pairing import PairingStore
            served = [active] + sorted(
                name for name, _home in profile_homes if name != active
            )
            # Per-profile PairingStores so authz_mixin routes pairing checks to the right whitelist;
            # the active profile's store is at its HERMES_HOME, other served profiles at their own.
            for name in served:
                if name and name not in self.pairing_stores:
                    self.pairing_stores[name] = (
                        self.pairing_store
                        if name == active
                        else PairingStore(profile=name)
                    )
            write_runtime_status(served_profiles=served)
        except Exception:
            logger.debug("could not record served_profiles", exc_info=True)

        return connected

    async def _start_one_profile_adapters(
        self, profile_name: str, profile_home: "Path", claimed: Dict[tuple, str]
    ) -> int:
        """Create+connect one profile's adapters under its runtime scope."""
        from gateway.run import (
            MultiplexConfigError,
            SecondaryPortBindingConfigError,
            _load_gateway_runtime_config,
            _own_policy_open_startup_violation,
            _platform_has_bot_credential,
            _profile_runtime_scope,
        )
        from gateway.config import load_gateway_config
        from hermes_cli.env_loader import hydrate_profile_secret_sources

        # Hydrate external secret sources (1Password/vault/...) off-loop ONCE, then enter the scope
        # without re-hydrating: the sync hydration is network-bound and would otherwise stall every
        # other profile's heartbeat while this one boots (same class as the reconnect path).
        await asyncio.to_thread(hydrate_profile_secret_sources, profile_home)

        with _profile_runtime_scope(profile_home, hydrate_secrets=False):
            profile_runtime_cfg = _load_gateway_runtime_config()
            from hermes_cli.plugins import discover_plugins

            discover_plugins()

            # Register this profile's own declarative shell hooks and outbound webhooks. The
            # registration in start() runs before any profile scope exists and only sees the root
            # profile's config, so without this a secondary profile's `hooks:` block is silently
            # inert (its turns use a plugin manager keyed by resolved home).
            try:
                from hermes_cli.config import load_config as _load_profile_config
                from agent.shell_hooks import (
                    register_from_config as _register_shell_hooks,
                )
                from agent.outbound_webhooks import (
                    register_from_config as _register_outbound_webhooks,
                )

                _profile_hooks_cfg = _load_profile_config()
                _register_shell_hooks(_profile_hooks_cfg, accept_hooks=False)
                _register_outbound_webhooks(_profile_hooks_cfg)
            except Exception:
                logger.warning(
                    "shell-hook/webhook registration failed for profile '%s'",
                    profile_name,
                    exc_info=True,
                )

            profile_cfg = load_gateway_config()
            violation = _own_policy_open_startup_violation(profile_cfg)
        self._snapshot_profile_busy_modes(profile_name, profile_runtime_cfg)
        if violation:
            raise MultiplexConfigError(
                f"Profile '{profile_name}' enables {violation}. "
                "Enable GATEWAY_ALLOW_ALL_USERS or the platform allow-all flag "
                "for that profile, or change dm_policy/group_policy away from "
                "'open'."
            )

        port_binding_platforms = sorted(
            platform.value
            for platform, platform_config in profile_cfg.platforms.items()
            if platform_config.enabled
            and _platform_binds_port(platform.value, platform_config.extra)
        )
        if port_binding_platforms:
            joined = ", ".join(port_binding_platforms)
            raise SecondaryPortBindingConfigError(
                f"Profile '{profile_name}' enables port-binding platform(s) "
                f"{joined}, but gateway.multiplex_profiles is on. The default "
                f"profile owns the single shared HTTP listener and serves every "
                f"profile through the /p/{profile_name}/ URL prefix. Remove "
                f"these platform entries from profile '{profile_name}'s config.yaml "
                f"or configure them only on the default profile."
            )

        profile_map = self._profile_adapters.setdefault(profile_name, {})
        connected = 0
        for platform, platform_config in profile_cfg.platforms.items():
            if not platform_config.enabled:
                continue
            # A platform enabled in a secondary profile's config.yaml may have no credential in that
            # profile's secret scope — the shared YAML enables it for the default profile only.
            # Building an adapter anyway would fan one inbound message out across every
            # credential-less profile; mirror the primary loop's credential gate and skip.
            if (
                getattr(self.config, "multiplex_profiles", False)
                and not _platform_has_bot_credential(platform, platform_config)
            ):
                logger.info(
                    "[MULTIPLEX] Profile '%s': skipping %s - no bot credential "
                    "in this profile's secrets",
                    profile_name,
                    platform.value,
                )
                continue
            # Relay and WhatsApp are shared process-level ingress in multiplex mode (one connection
            # owned by the active profile, route-stamped source.profile fans out). WhatsApp is one
            # session per phone number; a secondary adapter would only retry-loop and stall startup.
            if (
                getattr(self.config, "multiplex_profiles", False)
                and platform in (Platform.RELAY, Platform.WHATSAPP)
            ):
                continue
            try:
                with _profile_runtime_scope(profile_home, hydrate_secrets=False):
                    adapter = self._create_adapter(platform, platform_config)
            except Exception as e:
                logger.error(
                    "[MULTIPLEX] Profile '%s': _create_adapter('%s') raised %s",
                    profile_name,
                    platform.value,
                    e,
                    exc_info=True,
                )
                continue
            if not adapter:
                logger.warning(
                    "[MULTIPLEX] Profile '%s': skipping platform '%s' - adapter creation returned None",
                    profile_name,
                    platform.value,
                )
                continue

            # Same-token conflict detection — refuse a duplicate poll.
            credential_claim = self._adapter_credential_claim(platform, adapter)
            if credential_claim is not None:
                owner = claimed.get(credential_claim)
                if owner is not None:
                    message = (
                        f"Profile '{owner}' and '{profile_name}' both configure "
                        f"{platform.value} with the same credential. Give each "
                        f"profile its own {platform.value} credential."
                    )
                    logger.error(
                        "Profile '%s' and '%s' both configure %s with the same "
                        "credential — refusing to start the duplicate (one "
                        "credential cannot be consumed twice). Give each profile "
                        "its own %s credential.",
                        owner, profile_name, platform.value, platform.value,
                    )
                    self._update_platform_runtime_status(
                        f"{profile_name}:{platform.value}",
                        platform_state="fatal",
                        error_code="duplicate_credential",
                        error_message=message,
                    )
                    # This adapter has not connected and therefore owns no resources to clean up.
                    # Calling disconnect here can mutate the shared platform state and, for a same-
                    # credential Photon adapter, shut down the primary profile's live sidecar.
                    continue

            listener_claim = self._adapter_listener_claim(platform, adapter)
            if listener_claim is not None:
                owner = claimed.get(listener_claim)
                if owner is not None:
                    bind, port = listener_claim[-2:]
                    message = (
                        f"Profile '{owner}' and '{profile_name}' both configure "
                        f"{platform.value} sidecars on the same listener. Configure "
                        f"a distinct listener for profile '{profile_name}'."
                    )
                    logger.error(
                        "Profile '%s' and '%s' both configure %s sidecars on "
                        "%s:%s — refusing to start the duplicate listener. "
                        "Set platforms.%s.extra.sidecar_port to a distinct port "
                        "for profile '%s'.",
                        owner,
                        profile_name,
                        platform.value,
                        bind,
                        port,
                        platform.value,
                        profile_name,
                    )
                    self._update_platform_runtime_status(
                        f"{profile_name}:{platform.value}",
                        platform_state="fatal",
                        error_code="duplicate_listener",
                        error_message=message,
                    )
                    # Like credential conflicts, this adapter never connected
                    # and owns no resources that should be disconnected.
                    continue

            self._configure_profile_adapter(adapter, profile_name, platform)

            try:
                with _profile_runtime_scope(profile_home, hydrate_secrets=False):
                    success = await self._connect_initial_adapter_with_timeout(
                        adapter, platform
                    )
                if success:
                    profile_map[platform] = adapter
                    # Restore persisted /voice state for this bot (#84872) —
                    # primary startup and every reconnect path already do.
                    self._sync_voice_mode_state_to_adapter(adapter)
                    if credential_claim is not None:
                        claimed[credential_claim] = profile_name
                    if listener_claim is not None:
                        claimed[listener_claim] = profile_name
                    connected += 1
                    logger.info("✓ %s connected (profile: %s)", platform.value, profile_name)
                else:
                    logger.warning("✗ %s failed to connect (profile: %s)", platform.value, profile_name)
                    await self._safe_adapter_disconnect(adapter, platform)
                    self._schedule_secondary_profile_startup_reconnect(
                        profile_name, platform, adapter
                    )
            except Exception as e:
                logger.error("✗ %s error (profile: %s): %s", platform.value, profile_name, e)
                await self._safe_adapter_disconnect(adapter, platform)
                self._schedule_secondary_profile_startup_reconnect(
                    profile_name, platform, adapter
                )
        return connected

    def _configure_profile_adapter(
        self,
        adapter: BasePlatformAdapter,
        profile_name: str,
        platform: Platform,
    ) -> None:
        """Install the profile-scoped handlers shared by startup and reconnect."""
        # Runtime status is process-scoped while message/config work is profile-scoped. Keep both
        # dimensions in the key so dashboard/NAS health aggregation sees which secondary failed.
        adapter._runtime_status_platform_key = f"{profile_name}:{platform.value}"
        adapter.set_message_handler(self._make_profile_message_handler(profile_name))
        adapter.set_fatal_error_handler(
            self._make_profile_fatal_error_handler(profile_name, platform)
        )
        adapter.set_session_store(self.session_store)
        # Declare credential ownership BEFORE any inbound event can be handled: adapter-level
        # session keys (batching, _active_sessions, busy guard) are derived at ingress, before the
        # handler stamps source.profile — without this every secondary bot would key into the
        # default profile's `agent:main:` lane (see BasePlatformAdapter._session_key_profile).
        _set_owner = getattr(adapter, "set_owner_profile", None)
        if callable(_set_owner):
            _set_owner(profile_name)
        adapter.set_busy_session_handler(
            self._make_profile_busy_session_handler(profile_name)
        )
        _set_reaction = getattr(adapter, "set_reaction_handler", None)
        if callable(_set_reaction):
            _set_reaction(self._handle_reaction_event)
        adapter.set_topic_recovery_fn(self._recover_telegram_topic_thread_id)
        adapter.set_authorization_check(
            self._make_adapter_auth_check(platform, profile_name=profile_name)
        )
        adapter.set_platform_event_handler(
            self._make_profile_platform_event_handler(profile_name)
        )
        # Voice transcripts from this bot's channels dispatch through THIS
        # adapter (primary wiring lives at connect time; see #75198).
        self._bind_voice_input_callback(adapter)
        text_modes = getattr(self, "_busy_text_modes_by_profile", None)
        adapter._busy_text_mode = (
            text_modes.get(profile_name, self._busy_text_mode)
            if isinstance(text_modes, dict)
            else self._busy_text_mode
        )
        # Secondary adapters always carry the profile they serve so prune
        # paths namespace topic bindings correctly under multiplex (#76423).
        adapter._hermes_profile_name = profile_name

    async def _run_secondary_profile_reconnect(
        self, profile_name: str, platform: Platform
    ) -> None:
        """Reconnect a retryable secondary adapter under its own profile scope."""
        from gateway.run import _platform_has_bot_credential, _profile_runtime_scope, _reconnect_backoff
        attempts = 0
        current_task = asyncio.current_task()
        try:
            while self._running:
                adapter = None
                try:
                    from hermes_cli.profiles import get_profile_dir
                    from hermes_cli.env_loader import hydrate_profile_secret_sources
                    from gateway.config import load_gateway_config

                    profile_home = get_profile_dir(profile_name)
                    # Like the #16856 MCP discovery path, hydrate external secret
                    # sources off-loop so they cannot starve platform heartbeats.
                    await asyncio.to_thread(
                        hydrate_profile_secret_sources, profile_home
                    )
                    with _profile_runtime_scope(profile_home, hydrate_secrets=False):
                        profile_config = load_gateway_config().platforms.get(platform)
                        if profile_config is None or not profile_config.enabled:
                            return
                        # Mirrors the startup credential gate: a credential removed from this
                        # profile's scope must not rebuild an adapter that would fan out turns.
                        if not _platform_has_bot_credential(platform, profile_config):
                            logger.info(
                                "Secondary %s reconnect skipped: no bot credential "
                                "(profile: %s)",
                                platform.value,
                                profile_name,
                            )
                            return
                        adapter = self._create_adapter(platform, profile_config)
                        if adapter is None:
                            logger.warning(
                                "Secondary %s reconnect skipped: adapter unavailable (profile: %s)",
                                platform.value,
                                profile_name,
                            )
                            return
                        self._configure_profile_adapter(
                            adapter, profile_name, platform
                        )
                        success = await self._connect_adapter_with_timeout(
                            adapter, platform, is_reconnect=True
                        )

                    if success and self._running:
                        profile_map = self._profile_adapters.setdefault(profile_name, {})
                        if platform not in profile_map:
                            profile_map[platform] = adapter
                            self._sync_voice_mode_state_to_adapter(adapter)
                            logger.info(
                                "✓ %s reconnected (profile: %s)",
                                platform.value,
                                profile_name,
                            )
                            await self._redeliver_failed_obligations_for_platform(
                                platform, profile=profile_name
                            )
                            return
                        # A newer reconnect already won the slot while this
                        # attempt was awaiting connect; do not replace it.
                        await self._safe_adapter_disconnect(adapter, platform)
                        return

                    # Shutdown can begin mid-connect(): never republish a newly connected adapter
                    # after the registry has been drained; release its partial resources instead.
                    if success:
                        await self._safe_adapter_disconnect(adapter, platform)
                        return

                    await self._safe_adapter_disconnect(adapter, platform)
                    if (
                        getattr(adapter, "has_fatal_error", False)
                        and not getattr(adapter, "fatal_error_retryable", True)
                    ):
                        return
                except asyncio.CancelledError:
                    if adapter is not None:
                        await self._safe_adapter_disconnect(adapter, platform)
                    raise
                except Exception:
                    if adapter is not None:
                        await self._safe_adapter_disconnect(adapter, platform)
                    logger.debug(
                        "Secondary %s reconnect attempt failed (profile: %s)",
                        platform.value,
                        profile_name,
                        exc_info=True,
                    )

                if not self._running:
                    return
                attempts += 1
                backoff = _reconnect_backoff(attempts)
                logger.info(
                    "Secondary %s reconnect retry in %ds (profile: %s)",
                    platform.value,
                    backoff,
                    profile_name,
                )
                await asyncio.sleep(backoff)
        finally:
            pending = self._profile_failed_platforms
            if isinstance(pending, dict):
                profile_pending = pending.get(profile_name)
                task = profile_pending.get(platform) if isinstance(profile_pending, dict) else None
                if not isinstance(task, asyncio.Task) or task is current_task:
                    if isinstance(profile_pending, dict):
                        profile_pending.pop(platform, None)
                        if not profile_pending:
                            pending.pop(profile_name, None)

    def _schedule_secondary_profile_startup_reconnect(
        self, profile_name: str, platform: Platform, adapter: BasePlatformAdapter
    ) -> None:
        """Queue a cold-start reconnect for a secondary adapter.

        Startup failures happen BEFORE ``self._running`` flips True, so the regular scheduler's
        guard would drop the request. Park a task across startup and hand off to the scheduler once
        live (``_profile_failed_platforms`` dedupes); release it if shutdown begins first.
        Non-retryable failures are dropped as the regular scheduler would.
        """
        if not getattr(adapter, "fatal_error_retryable", True):
            return
        if is_global_startup_conflict(getattr(adapter, "fatal_error_code", None)):
            # Same startup contract as the primary path: a live foreign holder of this profile's
            # token/identity is an ownership conflict, not a transient blip. Park it fatal (like
            # ``duplicate_credential``) instead of retry-storming the token every backoff.
            logger.error(
                "[MULTIPLEX] Profile '%s': %s credential is held by another "
                "gateway (%s) — parked, not retried. %s",
                profile_name,
                platform.value,
                adapter.fatal_error_code,
                adapter.fatal_error_message or "",
            )
            self._update_platform_runtime_status(
                f"{profile_name}:{platform.value}",
                platform_state="fatal",
                error_code=adapter.fatal_error_code,
                error_message=adapter.fatal_error_message,
            )
            return

        async def _await_running_then_schedule() -> None:
            if self._running:
                try:
                    self._schedule_secondary_profile_reconnect(
                        profile_name, platform, adapter
                    )
                except Exception:
                    # Same GC-time-exception hazard as the post-poll handoff
                    # below; surface it in gateway.log instead.
                    logger.exception(
                        "secondary-startup-reconnect handoff failed "
                        "(profile=%s platform=%s)",
                        profile_name,
                        platform.value,
                    )
                return
            # Modest poll: startup completion has no dedicated event, and the reconnect runner's own
            # backoff makes sub-100ms precision irrelevant. Bounded so a wedged startup cannot spin.
            while not self._running and not self._shutdown_event.is_set():
                await asyncio.sleep(0.1)
            if self._running and not self._shutdown_event.is_set():
                try:
                    self._schedule_secondary_profile_reconnect(
                        profile_name, platform, adapter
                    )
                except Exception:
                    # The handoff touches live registries; if it raises, the parked task dies as an
                    # unretrieved-task exception logged only at GC. Surface it where operators look.
                    logger.exception(
                        "secondary-startup-reconnect handoff failed "
                        "(profile=%s platform=%s)",
                        profile_name,
                        platform.value,
                    )

        task = asyncio.create_task(
            _await_running_then_schedule(),
            name=f"secondary-startup-reconnect:{profile_name}:{platform.value}",
        )
        background_tasks = getattr(self, "_background_tasks", None)
        if not isinstance(background_tasks, set):
            background_tasks = set()
            self._background_tasks = background_tasks
        background_tasks.add(task)
        task.add_done_callback(background_tasks.discard)

    def _schedule_secondary_profile_reconnect(
        self, profile_name: str, platform: Platform, adapter: BasePlatformAdapter
    ) -> None:
        """Schedule one runner-owned reconnect without sharing primary secrets."""
        if not self._running or not adapter.fatal_error_retryable:
            return
        pending = self._profile_failed_platforms
        if not isinstance(pending, dict):
            pending = {}
            self._profile_failed_platforms = pending
        profile_pending = pending.setdefault(profile_name, {})
        if platform in profile_pending:
            return
        task = asyncio.create_task(
            self._run_secondary_profile_reconnect(profile_name, platform),
            name=f"secondary-reconnect:{profile_name}:{platform.value}",
        )
        profile_pending[platform] = task
        background_tasks = getattr(self, "_background_tasks", None)
        if not isinstance(background_tasks, set):
            background_tasks = set()
            self._background_tasks = background_tasks
        background_tasks.add(task)
        task.add_done_callback(background_tasks.discard)

    def _make_profile_fatal_error_handler(
        self, profile_name: str, platform: Platform
    ) -> Callable[[BasePlatformAdapter], Awaitable[None]]:
        """Route a secondary-profile fatal error to that profile's reconnect slot."""
        async def _handler(adapter: BasePlatformAdapter) -> None:
            await self._handle_profile_adapter_fatal_error(profile_name, platform, adapter)

        return _handler

    async def _handle_profile_adapter_fatal_error(
        self,
        profile_name: str,
        platform: Platform,
        adapter: BasePlatformAdapter,
    ) -> None:
        """Remove a failed multiplexed adapter without touching the primary slot.

        Secondaries live in ``_profile_adapters``, which the primary-only fatal handler ignores;
        without this route a fatal secondary Discord client stayed live forever.
        """
        profile_map = getattr(self, "_profile_adapters", {}).get(profile_name)
        if not isinstance(profile_map, dict) or profile_map.get(platform) is not adapter:
            logger.debug(
                "Ignoring stale fatal error from secondary %s adapter (profile: %s)",
                platform.value,
                profile_name,
            )
            return
        profile_map.pop(platform, None)
        await self._safe_adapter_disconnect(adapter, platform)
        if not self._running:
            return
        self._schedule_secondary_profile_reconnect(profile_name, platform, adapter)
        logger.error(
            "Fatal %s adapter error for multiplexed profile %s (%s)",
            platform.value,
            profile_name,
            adapter.fatal_error_code or "unknown",
        )

    def _make_profile_message_handler(self, profile_name: str):
        """Return a message handler that stamps source.profile then delegates.

        Auth runs inside ``_handle_message`` *before* the agent-turn scope is installed. For
        secondary profiles under multiplex, wrap the whole handler in ``_profile_runtime_scope``
        so allowlists/tokens from that profile's ``.env`` are visible to ``get_secret`` / authz.
        """
        from gateway.run import _async_profile_runtime_scope
        from hermes_cli.profiles import get_profile_dir

        try:
            profile_home = get_profile_dir(profile_name)
        except Exception:
            profile_home = None

        async def _handler(event):
            try:
                if getattr(event, "source", None) is not None and not event.source.profile:
                    event.source.profile = profile_name
            except Exception:
                pass
            if profile_home is not None:
                async with _async_profile_runtime_scope(profile_home):
                    return await self._handle_message(event)
            return await self._handle_message(event)

        return _handler

    def _make_profile_busy_session_handler(self, profile_name: str):
        """Stamp an owning adapter's profile before resolving busy policy."""
        async def _handler(event, _session_key):
            try:
                if getattr(event, "source", None) is not None and not event.source.profile:
                    event.source.profile = profile_name
            except Exception:
                pass
            routed_session_key = self._session_key_for_source(event.source)
            return await self._handle_active_session_busy_message(
                event, routed_session_key
            )

        return _handler

    def _make_default_profile_message_handler(self):
        """Scope primary-adapter messages to their routed multiplex profile.

        Resolve the home per event so session lookup and transcript loading use the same profile
        store as the agent run. Authorization stays with the transport profile (a routed profile
        may intentionally have no bot credential/allowlist): the transport home is preserved on the
        live source and never re-checked against the routed scope. Unrouted events keep the default.
        """
        from gateway.run import _async_profile_runtime_scope, get_hermes_home
        default_home = Path(get_hermes_home())

        async def _handler(event):
            source = event.source
            # In-process only (SessionSource serialization ignores dynamic attrs). The route selects
            # agent/session state, not which bot admitted the message — separate trust domains.
            source._authorization_profile_home = default_home
            if (
                not getattr(source, "profile", None)
                and getattr(source, "profile_route_rejected", False) is not True
            ):
                from gateway.profile_routing import ProfileRouteRejected

                try:
                    source.profile = self._profile_name_for_source(source)
                except ProfileRouteRejected:
                    # NOT write-only: the ``_handle_message`` ingress gate reads this exact marker
                    # and drops the message fail-closed (explicit route to an unserved profile).
                    source.profile_route_rejected = True

            profile_home = (
                self._resolve_profile_home_for_source(source)
                if getattr(source, "profile", None)
                else default_home
            )
            async with _async_profile_runtime_scope(profile_home):
                return await self._handle_message(event)

        return _handler

    def _primary_message_handler(self):
        """Return the correctly scoped handler for a primary adapter."""
        if getattr(self.config, "multiplex_profiles", False):
            return self._make_default_profile_message_handler()
        return self._handle_message

    async def _handle_gateway_platform_event(self, event: dict, source) -> None:
        """Authorize and publish one normalized adapter event to plugin hooks."""
        try:
            from hermes_cli.lifecycle import has_hook, invoke_hook

            if not has_hook("gateway_platform_event"):
                return
            if not self._is_user_authorized_for_source(source):
                return
            invoke_hook("gateway_platform_event", **event)
        except Exception:
            # Observer failures must never break the adapter's update loop.
            logger.debug("gateway_platform_event hook dispatch failed", exc_info=True)

    def _make_profile_platform_event_handler(self, profile_name: str):
        """Bind platform-event auth and hook dispatch to one multiplex profile."""
        from gateway.run import _profile_runtime_scope
        from hermes_cli.profiles import get_profile_dir

        try:
            profile_home = get_profile_dir(profile_name)
        except Exception:
            profile_home = None

        async def _handler(event, source):
            if getattr(source, "profile", None) is None:
                source.profile = profile_name
            if profile_home is not None:
                with _profile_runtime_scope(profile_home):
                    return await self._handle_gateway_platform_event(event, source)
            return await self._handle_gateway_platform_event(event, source)

        return _handler

    def _make_default_profile_platform_event_handler(self):
        """Scope primary-transport events to their routed multiplex profile."""
        from gateway.run import _profile_runtime_scope, get_hermes_home
        default_home = Path(get_hermes_home())

        async def _handler(event, source):
            source._authorization_profile_home = default_home
            with _profile_runtime_scope(self._resolve_profile_home_for_source(source)):
                return await self._handle_gateway_platform_event(event, source)

        return _handler

    def _primary_platform_event_handler(self):
        if getattr(self.config, "multiplex_profiles", False):
            return self._make_default_profile_platform_event_handler()
        return self._handle_gateway_platform_event

    @staticmethod
    def _adapter_credential_claim(
        platform: Platform, adapter: Any
    ) -> Optional[tuple]:
        """Return the exclusive credential resource claimed by an adapter."""
        from gateway.run import GatewayRunner
        fingerprint = GatewayRunner._adapter_credential_fingerprint(adapter)
        if fingerprint is None:
            return None
        return (platform, fingerprint)

    @staticmethod
    def _adapter_listener_claim(platform: Platform, adapter: Any) -> Optional[tuple]:
        """Return the exclusive listener resource claimed by an adapter.

        Sidecars with different credentials still cannot share a bind+port; expose it as a claim so
        multiplex startup rejects the later adapter before connect()/disconnect() disturb the first.
        """
        if getattr(platform, "value", None) != "photon":
            return None
        bind = getattr(adapter, "_sidecar_bind", None)
        port = getattr(adapter, "_sidecar_port", None)
        if not isinstance(bind, str) or not bind.strip():
            return None
        try:
            port = int(port)
        except (TypeError, ValueError):
            return None
        return ("listener", "photon", bind.strip().lower(), port)

    @staticmethod
    def _adapter_credential_fingerprint(adapter: Any) -> Optional[str]:
        """Return a stable, log-safe fingerprint of an adapter's credential.

        Salted hash (never the credential) used to detect two profiles sharing one platform
        credential; None when no credential is discoverable (conflict detection is then skipped).
        """
        token = None
        for attr in (
            "token",
            "bot_token",
            "_token",
            "api_token",
            "_bot_token",
            # Photon/Spectrum authenticates with project credentials, not a bot token; including
            # its secret stops multiplexed profiles spawning rival sidecars for one account/port.
            "_project_secret",
            # Feishu/Lark authenticates with an app_id/app_secret pair (one WebSocket per app).
            # app_id is stable, log-safe and already the adapter's _app_lock_identity, so including
            # it lets the multiplex guard refuse cloned profiles competing for the same app.
            "_app_id",
            # Same class: Teams (client_id/client_secret) and WeCom
            # (bot_id/secret) authenticate with an app-style id pair too.
            "_client_id",
            "_bot_id",
        ):
            val = getattr(adapter, attr, None)
            if isinstance(val, str) and val.strip():
                token = val.strip()
                break
        # Many adapters (e.g. Discord) store the token on their `config` sub-object. Without this
        # lookup they return None, the same-token check is silently skipped, and every profile's
        # adapter polls the same bot token — a per-message race over which one answers.
        if not token:
            cfg = getattr(adapter, "config", None)
            if cfg is not None:
                for attr in ("token", "bot_token"):
                    val = getattr(cfg, attr, None)
                    if isinstance(val, str) and val.strip():
                        token = val.strip()
                        break
        if not token:
            config = getattr(adapter, "config", None)
            val = getattr(config, "token", None)
            if isinstance(val, str) and val.strip():
                token = val.strip()
        if not token:
            return None
        import hashlib
        return hashlib.sha256(("hermes-mux:" + token).encode("utf-8")).hexdigest()[:16]

    def _create_adapter(
        self,
        platform: Platform,
        config: Any,
    ) -> Optional[BasePlatformAdapter]:
        """Create an adapter and bind it to this gateway runner.

        Every lifecycle path (primary/secondary startup, reconnect) uses this method; keep runner
        binding here so adapters can resolve inbound profile routes before handlers or connect().
        """
        adapter = self._instantiate_adapter(platform, config)
        if adapter is not None:
            adapter.gateway_runner = self
        return adapter

    def _instantiate_adapter(
        self,
        platform: Platform,
        config: Any,
    ) -> Optional[BasePlatformAdapter]:
        """Instantiate the appropriate adapter for a platform.

        Checks platform_registry (plugin adapters) first, then the built-in table of core platforms.
        """
        from gateway.run import _instantiate_builtin_adapter
        if hasattr(config, "extra") and isinstance(config.extra, dict):
            config.extra.setdefault(
                "group_sessions_per_user",
                self.config.group_sessions_per_user,
            )
            config.extra.setdefault(
                "thread_sessions_per_user",
                getattr(self.config, "thread_sessions_per_user", False),
            )

        # ── Plugin-registered platforms (checked first) ───────────────────
        try:
            from gateway.platform_registry import platform_registry
            if platform_registry.is_registered(platform.value):
                adapter = platform_registry.create_adapter(platform.value, config)
                if adapter is not None:
                    return adapter
                # Registered but failed to instantiate — don't silently fall
                # through to built-ins (there are none for plugin platforms).
                logger.error(
                    "Platform '%s' is registered but adapter creation failed "
                    "(check dependencies and config)",
                    platform.value,
                )
                return None
        except Exception as e:
            logger.debug("Platform registry lookup for '%s' failed: %s", platform.value, e)
        # Fall through to built-in adapters below

        return _instantiate_builtin_adapter(platform, config)

    def _make_adapter_auth_check(
        self,
        platform: Platform,
        profile_name: Optional[str] = None,
    ) -> Callable[[str, Optional[str], Optional[str]], bool]:
        """Build a platform-bound auth callback for adapter use.

        Adapters fetching external context (e.g. Slack ``conversations.replies``) use it via
        ``_is_sender_authorized`` to mark non-allowlisted senders unverified (prompt-injection
        mitigation). Delegates to :meth:`_is_user_authorized` so the full auth chain stays the single
        source of truth. ``profile_name`` binds a secondary adapter to its own secret scope; for the
        shared primary (None) the ``profile_routes`` match is stamped on the source so the routed
        profile's pairing store is consulted while allowlist reads stay under the transport home.
        """
        from gateway.run import get_hermes_home
        multiplex = bool(getattr(self.config, "multiplex_profiles", False))
        transport_home = (
            Path(get_hermes_home()) if multiplex and profile_name is None else None
        )

        def check(
            user_id: str,
            chat_type: Optional[str] = None,
            chat_id: Optional[str] = None,
            *,
            is_bot: bool = False,
            thread_id: Optional[str] = None,
        ) -> bool:
            if not user_id:
                return False
            source = SessionSource(
                platform=platform,
                chat_id=chat_id or "",
                chat_type=chat_type or "group",
                user_id=user_id,
                thread_id=thread_id,
                is_bot=bool(is_bot),
                profile=profile_name,
            )
            # Same in-process transport provenance ``build_source`` retains, so adapter-level policy
            # reads (config.yaml group_allowed_chats, allow_from) resolve the receiving adapter even
            # once the routed profile is stamped below.
            registry = (
                (getattr(self, "_profile_adapters", None) or {}).get(profile_name)
                if profile_name
                else getattr(self, "adapters", None)
            ) or {}
            adapter = registry.get(platform)
            if adapter is not None:
                source._transport_adapter_ref = _weakref.ref(adapter)
            if transport_home is None:
                return self._is_user_authorized(source)
            source._authorization_profile_home = transport_home
            from gateway.profile_routing import ProfileRouteRejected

            try:
                source.profile = self._profile_name_for_source(source)
            except ProfileRouteRejected:
                # Same fail-closed outcome as the ingress gate in
                # ``_handle_message`` for a route to an unserved profile.
                return False
            return self._is_user_authorized_for_source(source)
        return check
