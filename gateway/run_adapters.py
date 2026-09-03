"""Adapter connect/disconnect, fatal-error recovery, reconnect watcher and multiplex profile
adapters for GatewayRunner (mixin bound via the MRO).

``gateway.run`` internals are imported lazily inside method bodies (import cycle), so
``patch("gateway.run.X")`` keeps intercepting them at call time.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
import asyncio
import contextlib
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
    """Adapter lifecycle: connect/teardown, fatal recovery, reconnect watcher, multiplex profiles."""

    @staticmethod
    async def _wait_or_detach(task: "asyncio.Future", timeout: float) -> bool:
        """Wait up to ``timeout`` for ``task``; on deadline (or our own cancellation) detach it.

        Not ``asyncio.wait_for``: that WAITS for the cancelled child to exit, so a connect()/close()
        that swallows ``CancelledError`` blocks recovery forever. Returns True if it finished in time.
        """
        try:
            done, _pending = await asyncio.wait({task}, timeout=timeout)
        except asyncio.CancelledError:
            task.cancel()
            task.add_done_callback(consume_detached_task_result)
            raise
        if task in done:
            return True
        task.cancel()
        task.add_done_callback(consume_detached_task_result)
        return False

    async def _await_adapter_cleanup_with_timeout(self, awaitable: Awaitable[Any], timeout: float) -> bool:
        """Await adapter cleanup with a detach-on-deadline bound; True when it completed."""
        if timeout <= 0:
            await awaitable
            return True
        task = asyncio.ensure_future(awaitable)
        if await self._wait_or_detach(task, timeout):
            await task
            return True
        return False

    async def _safe_adapter_disconnect(self, adapter, platform) -> None:
        """Call adapter.disconnect() defensively (bounded, never raises).

        After a failed connect() partial resources (ClientSession, poll tasks, subprocesses) would
        otherwise leak; must tolerate partial-init state.
        """
        timeout = self._adapter_disconnect_timeout_secs()
        label = platform.value if platform is not None else "adapter"
        try:
            if not await self._await_adapter_cleanup_with_timeout(adapter.disconnect(), timeout):
                logger.warning(
                    "Timed out after %.1fs while disconnecting %s adapter; continuing shutdown",
                    timeout, label,
                )
        except Exception as e:
            logger.debug("Defensive %s disconnect after failed connect raised: %s", label, e)

    async def _bounded_adapter_teardown(self, adapter, platform, *, profile: Optional[str] = None) -> None:
        """Tear down one adapter on the shutdown path with bounded awaits (never raises).

        Unbounded, a half-dead transport stalls shutdown past systemd's ``TimeoutStopSec``; the
        SIGKILL skips ``atexit`` PID-file cleanup and the next start dies with "PID file race lost".
        """
        timeout = self._adapter_disconnect_timeout_secs()
        suffix = f" (profile: {profile})" if profile else ""
        started_at = time.monotonic()
        try:
            if not await self._await_adapter_cleanup_with_timeout(
                adapter.cancel_background_tasks(), timeout
            ):
                logger.warning(
                    "✗ %s background-task cancel timed out after %.1fs - forcing continue%s",
                    platform.value, timeout, suffix,
                )
        except Exception as e:
            logger.debug("✗ %s background-task cancel error%s: %s", platform.value, suffix, e)
        try:
            if await self._await_adapter_cleanup_with_timeout(adapter.disconnect(), timeout):
                logger.info(
                    "✓ %s disconnected (%.2fs)%s", platform.value, time.monotonic() - started_at, suffix,
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

    @staticmethod
    def _env_timeout_override(name: str) -> Optional[float]:
        """Non-negative float from env var ``name``; None when unset or unparseable (warned)."""
        raw = os.getenv(name, "").strip()
        if not raw:
            return None
        try:
            return max(0.0, float(raw))
        except ValueError:
            logger.warning("Ignoring invalid %s=%r", name, raw)
            return None

    def _adapter_disconnect_timeout_secs(self) -> float:
        """Return the per-adapter disconnect timeout used during shutdown."""
        from gateway.run import _ADAPTER_DISCONNECT_TIMEOUT_SECS_DEFAULT
        override = self._env_timeout_override("HERMES_GATEWAY_ADAPTER_DISCONNECT_TIMEOUT")
        return _ADAPTER_DISCONNECT_TIMEOUT_SECS_DEFAULT if override is None else override

    def _platform_connect_timeout_secs(self, platform=None, *, initial: bool = False) -> float:
        """Per-platform connect timeout. Telegram's full 180s is NOT spent at cold start (it would
        hold the gateway out of ``running``); the watcher retries with the full budget."""
        from gateway.run import (
            _PLATFORM_CONNECT_TIMEOUT_SECS_DEFAULT, _TELEGRAM_CONNECT_TIMEOUT_SECS_DEFAULT,
            _TELEGRAM_INITIAL_CONNECT_TIMEOUT_SECS_DEFAULT,
        )
        override = self._env_timeout_override("HERMES_GATEWAY_PLATFORM_CONNECT_TIMEOUT")
        if override is not None:
            return override
        if platform == Platform.TELEGRAM:
            if initial:
                return _TELEGRAM_INITIAL_CONNECT_TIMEOUT_SECS_DEFAULT
            return _TELEGRAM_CONNECT_TIMEOUT_SECS_DEFAULT
        return _PLATFORM_CONNECT_TIMEOUT_SECS_DEFAULT

    async def _connect_adapter_with_timeout(
        self, adapter, platform, *, is_reconnect: bool = False, initial: bool = False
    ) -> bool:
        """Connect with a bound so one platform can't block others. ``is_reconnect``: cold boot
        drops the stale server-side queue, a reconnect keeps it. ``initial``: capped budget."""
        timeout = self._platform_connect_timeout_secs(platform, initial=initial)
        if timeout <= 0:
            return await adapter.connect(is_reconnect=is_reconnect)
        task = asyncio.ensure_future(adapter.connect(is_reconnect=is_reconnect))
        if await self._wait_or_detach(task, timeout):
            return bool(await task)
        raise TimeoutError(f"{platform.value} connect timed out after {timeout:g}s")

    async def _connect_initial_adapter_with_timeout(self, adapter, platform) -> bool:
        """Cold-start connect with replace intent visible ONLY during this await, so a later
        network recovery can never evict a healthy token holder."""
        adapter._platform_lock_takeover_allowed = bool(self._platform_lock_takeover_on_start)
        try:
            return await self._connect_adapter_with_timeout(adapter, platform, initial=True)
        finally:
            adapter._platform_lock_takeover_allowed = False

    async def _handle_reaction_event(self, ctx: Dict[str, Any]) -> None:
        """Fan a normalised reaction event out to the HookRegistry; errors never block the adapter."""
        event_name = str(ctx.get("event_name") or "reaction:added")
        try:
            await self.hooks.emit(event_name, ctx)
        except Exception:
            logger.debug("[Gateway] reaction hook emit failed", exc_info=True)

    async def _handle_adapter_fatal_error(self, adapter: BasePlatformAdapter) -> None:
        """React to an adapter failure after startup (retryable → background reconnect queue).

        Detached task: the notification arrives on the failing adapter's own polling task, which
        the handler's disconnect can cancel mid-flight, stranding the platform half-handled.
        """
        tasks = getattr(self, "_fatal_handler_tasks", None)
        if tasks is None:
            tasks = self._fatal_handler_tasks = set()
        task = asyncio.create_task(self._handle_adapter_fatal_error_detached(adapter))
        tasks.add(task)
        task.add_done_callback(tasks.discard)
        # shield(): a plain `await task` would tunnel the caller's cancellation into the detached
        # task; with shield the caller sees CancelledError and the handler runs to completion.
        await asyncio.shield(task)

    def _queue_retryable_fatal_platform(self, adapter: BasePlatformAdapter) -> bool:
        """Queue a retryable fatal adapter for background reconnection (True when newly queued).

        Must not await: callers run this BEFORE any disconnect so a wedged close can't strand it.
        """
        if not adapter.fatal_error_retryable:
            return False
        platform_config = self.config.platforms.get(adapter.platform)
        if not platform_config:
            return False
        if adapter.platform in self._failed_platforms:
            # Already queued is exactly when the watcher may have died (_spawn_supervised gives up
            # after _MAX_SUPERVISED_RESTARTS); without this backstop the platform is a silent
            # permanent outage — nothing retries and the stranded check treats "queued" as safe.
            self._ensure_reconnect_watcher_running()
            return False
        self._failed_platforms[adapter.platform] = {
            "config": platform_config,
            "attempts": 0,
            "next_retry": time.monotonic(),
            "queued_at": time.monotonic(),
            "credential_claim": self._adapter_credential_claim(adapter.platform, adapter),
            "listener_claim": self._adapter_listener_claim(adapter.platform, adapter),
        }
        logger.info("%s queued for background reconnection", adapter.platform.value)
        self._ensure_reconnect_watcher_running()
        return True

    async def _handle_adapter_fatal_error_detached(self, adapter: BasePlatformAdapter) -> None:
        """Run the fatal handler; a platform left stranded (not reconnected, not queued, not
        intentionally disabled) exits the gateway with failure so the service manager restarts it."""
        try:
            # Outer hard deadline: a hang anywhere in the impl must not wedge this task — the
            # stranded check in ``finally`` only runs when we return.
            timeout = self._adapter_disconnect_timeout_secs()
            if timeout <= 0:
                await self._handle_adapter_fatal_error_impl(adapter)
            else:
                # Disconnect budget + proportional bookkeeping overhead (tests shrink the timeout).
                outer = timeout + min(2.0, max(0.05, timeout))
                if not await self._await_adapter_cleanup_with_timeout(
                    self._handle_adapter_fatal_error_impl(adapter), outer
                ):
                    logger.error(
                        "Fatal-error handling for %s timed out after %.1fs; "
                        "ensuring reconnect queue is populated", adapter.platform.value, outer,
                    )
                    self._queue_retryable_fatal_platform(adapter)
        except asyncio.CancelledError:
            # A cancelled or raising fatal handler must not strand a retryable platform.
            self._queue_retryable_best_effort(adapter, "cancellation")
            raise
        except Exception:
            logger.exception("Fatal-error handling for %s raised unexpectedly", adapter.platform.value)
            self._queue_retryable_best_effort(adapter, "exception")
        finally:
            platform = adapter.platform
            shutdown_event = getattr(self, "_shutdown_event", None)
            if (
                adapter.fatal_error_retryable
                and platform not in self.adapters
                and platform not in getattr(self, "_failed_platforms", {})
                and not (shutdown_event is not None and shutdown_event.is_set())
            ):
                logger.error(
                    "%s adapter was lost without entering the reconnection "
                    "queue; exiting gateway so the service manager restarts it.", platform.value,
                )
                self._exit_reason = f"{platform.value} adapter lost without reconnection queue"
                self._exit_with_failure = True
                await self.stop()

    def _queue_retryable_best_effort(self, adapter: BasePlatformAdapter, why: str) -> None:
        try:
            self._queue_retryable_fatal_platform(adapter)
        except Exception:
            logger.debug(
                "Failed to queue %s after fatal-handler %s", adapter.platform.value, why, exc_info=True,
            )

    async def _handle_adapter_fatal_error_impl(self, adapter: BasePlatformAdapter) -> None:
        # Snapshot the slot owner first: a stale notification must not overwrite a healthy
        # platform's runtime status or re-queue it.
        existing = self.adapters.get(adapter.platform)
        if existing is not None and existing is not adapter:
            logger.debug(
                "Ignoring stale fatal error from a superseded %s adapter instance: %s",
                adapter.platform.value, adapter.fatal_error_code or "unknown",
            )
            return

        logger.error(
            "Fatal %s adapter error (%s): %s", adapter.platform.value,
            adapter.fatal_error_code or "unknown", adapter.fatal_error_message or "unknown error",
        )
        # relay_disabled (credential revoked by opt-out) renders "disabled", not red fatal/retrying.
        self._update_platform_runtime_status(
            adapter.platform.value,
            platform_state=(
                "disabled" if adapter.fatal_error_code == "relay_disabled"
                else "retrying" if adapter.fatal_error_retryable else "fatal"
            ),
            error_code=adapter.fatal_error_code,
            error_message=adapter.fatal_error_message,
        )

        if existing is adapter:
            # Claim for teardown BEFORE awaiting disconnect(), else a concurrent second fatal
            # notification for the same adapter would disconnect() it twice.
            self.adapters.pop(adapter.platform, None)
            self.delivery_router.adapters = self.adapters

        # Queue BEFORE any disconnect await: a half-dead transport can wedge native close() (or
        # swallow CancelledError), and "disconnect then queue" left platforms permanently deaf.
        self._queue_retryable_fatal_platform(adapter)

        if existing is adapter:
            # Bounded by the shutdown-path timeout so this always returns to the stranded check.
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
            # Everything is down but queued: stay alive (cron still runs, watcher recovers) rather
            # than turn a transient outage into a state-killing systemd restart loop.
            logger.warning(
                "No connected messaging platforms remain, but %d platform(s) "
                "queued for reconnection — gateway staying alive, watcher will "
                "retry in background.", len(self._failed_platforms),
            )

    def _retain_background_task(self, task: "asyncio.Task") -> "asyncio.Task":
        """Register ``task`` in ``_background_tasks`` (created lazily for bare test runners)."""
        tasks = getattr(self, "_background_tasks", None)
        if not isinstance(tasks, set):
            tasks = self._background_tasks = set()
        tasks.add(task)
        task.add_done_callback(tasks.discard)
        return task

    def _request_clean_exit(self, reason: str) -> None:
        self._exit_cleanly = True
        self._exit_reason = reason
        self._shutdown_event.set()

    @staticmethod
    def _supervised_backoff(attempt: int) -> float:
        """Capped exponential respawn delay (a method so tests can collapse the schedule)."""
        return min(60, 2 ** min(attempt, 6))

    def _spawn_supervised(
        self, coro_factory, name, *, restart=True, _attempt=0, on_spawn=None, on_give_up=None
    ):
        """Launch a long-lived background task with task-level supervision.

        Catches outer-loop/pre-try exceptions a bare ``create_task`` drops silently. Restarts with
        capped backoff up to ``_MAX_SUPERVISED_RESTARTS`` rapid failures; the counter resets after
        ``_SUPERVISED_HEALTHY_SECS`` of healthy running. Each spawn uses a fresh ``Context`` (an
        inherited delegated-child marker would make the Kanban dispatcher reject its own writes).
        ``on_spawn`` fires on EVERY spawn incl. respawns — callers tracking the handle elsewhere
        MUST pass it or a respawn leaves a stale handle and a SECOND watcher. ``on_give_up(name)``
        fires when the restart budget is spent.
        """
        # Spawn timestamp lets ``_done`` tell a rapid crash-loop from a healthy-run-then-crash.
        _started = time.monotonic()

        # No create_task kwargs (test doubles mock a narrow signature); running it from a fresh
        # Context gives the same isolation as create_task(..., context=Context()).
        task = Context().run(lambda: asyncio.create_task(coro_factory()))
        # Marks a PERMANENT watcher so the scale-to-zero idle check ignores it (else the gateway
        # counts itself busy forever); transient _background_tasks stay counted.
        task._hermes_supervised_watcher = True  # type: ignore[attr-defined]
        self._retain_background_task(task)
        if on_spawn is not None:
            # Record the live handle NOW so external trackers don't point at a dead prior task.
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
                # Clean return = deliberate shutdown or a self-disabling watcher; NEVER respawn.
                return
            logger.error("Supervised task %s died: %r", name, exc, exc_info=exc)
            if not (restart and self._running):
                return
            # A healthy run before the crash is a FRESH failure, not a crash-loop: reset so a
            # daemon crashing a few times over days is never abandoned.
            healthy = time.monotonic() - _started >= self._SUPERVISED_HEALTHY_SECS
            effective_attempt = 0 if healthy else _attempt
            if effective_attempt >= self._MAX_SUPERVISED_RESTARTS:
                logger.error(
                    "Supervised task %s died %d times in rapid succession "
                    "(each within %ds of restart) — giving up restarts", name,
                    effective_attempt, self._SUPERVISED_HEALTHY_SECS,
                )
                if on_give_up is not None:
                    try:
                        on_give_up(name)
                    except Exception:  # pragma: no cover - defensive
                        logger.debug("on_give_up callback for %s raised", name, exc_info=True)
                return
            backoff = self._supervised_backoff(effective_attempt)

            async def _respawn():
                await asyncio.sleep(backoff)
                if self._running:
                    self._spawn_supervised(
                        coro_factory, name, restart=restart, _attempt=effective_attempt + 1,
                        on_spawn=on_spawn, on_give_up=on_give_up,  # only the LAST give-up matters
                    )

            # The done callback runs in its registration context; isolate the backoff task too.
            self._retain_background_task(Context().run(lambda: asyncio.create_task(_respawn())))

        task.add_done_callback(_done)
        return task

    async def _handoff_watcher(self, interval: float = 2.0, drain_timeout: float = 30.0) -> None:
        """Process pending CLI→gateway session handoffs from ``state.db``.

        Claim atomically (pending → running), re-bind the home channel to the CLI session_id,
        dispatch a synthetic ``MessageEvent``, mark ``completed``/``failed``.
        """
        from gateway.run import _async_profile_runtime_scope, _handoff_watch_scopes, _reclaim_stale
        await asyncio.sleep(5)  # let platforms connect before dispatching through them

        # Does _process_handoff accept the profile argument? Test stand-ins bind a one-arg callable.
        try:
            import inspect as _inspect
            _process_takes_profile = len(_inspect.signature(self._process_handoff).parameters) >= 2
        except Exception:
            _process_takes_profile = False

        # In-flight dispatches keyed by session id: a handoff runs a FULL agent turn, so inline
        # processing would let one slow handoff time out every other profile's poll.
        inflight: Dict[str, "asyncio.Task"] = {}

        async def _dispatch(row, session_id, session_db, profile_name) -> None:
            """Run one claimed handoff to a terminal state, off the poll path."""
            try:
                await self._process_handoff(*((row, profile_name) if _process_takes_profile else (row,)))
                await session_db.complete_handoff(session_id)
            except asyncio.CancelledError:
                # Gateway shutting down: leave the row 'running' so the next
                # start's reclaim marks it failed with a clear reason.
                raise
            except Exception as exc:
                logger.warning("Handoff for session %s failed: %s", session_id, exc, exc_info=True)
                try:
                    await session_db.fail_handoff(session_id, str(exc))
                except Exception:
                    logger.debug("Could not record handoff failure", exc_info=True)
            finally:
                inflight.pop(session_id, None)

        async def _tick(profile_name: Optional[str] = None) -> None:
            """One poll of the CURRENTLY-SCOPED store; ``profile_name`` (None = root) routes
            delivery to that profile's OWN adapter.

            A closure, not a method: tests bind ``_handoff_watcher`` onto a ``SimpleNamespace``
            with only ``_session_db``/``_running``/``_process_handoff``.
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
                # INVARIANT (do not weaken): this task is created inside _profile_runtime_scope but
                # typically RUNS after it exits; it sees the profile's home/secret scope only because
                # those seams are ContextVar-based and ensure_future copies the Context.
                inflight[session_id] = asyncio.ensure_future(
                    _dispatch(row, session_id, session_db, profile_name)
                )

        # A row still 'running' at startup belongs to a gateway that died mid-dispatch; it blocks
        # request_handoff until reclaimed.
        def _scope(profile_home):
            if profile_home is None:
                return contextlib.nullcontext()
            return _async_profile_runtime_scope(profile_home)

        for _pname, _phome in _handoff_watch_scopes(self):
            try:
                async with _scope(_phome):
                    await _reclaim_stale(self)
            except Exception:
                logger.debug("Stale-handoff reclaim failed", exc_info=True)

        try:
            while self._running:
                try:
                    for profile_name, profile_home in _handoff_watch_scopes(self):
                        async with _scope(profile_home):
                            await _tick(profile_name)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.debug("Handoff watcher tick error: %s", exc, exc_info=True)
                await asyncio.sleep(interval)
        finally:
            # Bounded drain: cancelling would strand in-flight rows in 'running'.
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
        """Own the reconnect invariant once supervision gives up: while running with queued
        platforms, a watcher is live or a bounded respawn is scheduled (no later event can notice
        a dead watcher). Slow-tier exhaustion logs loudly; deliberately NOT a process restart."""
        if not getattr(self, "_running", False):
            return
        if getattr(self, "_failed_platforms", None):
            self._schedule_slow_reconnect_watcher_respawn(attempt=0)
        else:
            # Nothing depends on the watcher; the enqueue path spawns a fresh one when needed.
            logger.warning(
                "Reconnect watcher supervision exhausted with an empty retry "
                "queue — leaving it down until a platform is queued."
            )

    def _schedule_slow_reconnect_watcher_respawn(self, *, attempt: int) -> None:
        """Bounded slow-tier respawn of the reconnect watcher."""
        if attempt >= self._MAX_SLOW_WATCHER_RESPAWNS:
            logger.error(
                "Reconnect watcher could not be kept alive after %d slow "
                "respawns; %d platform(s) remain queued and unattended: %s. "
                "Manual intervention or a gateway restart is required.", attempt,
                len(self._failed_platforms), ", ".join(str(p) for p in self._failed_platforms),
            )
            return

        async def _slow_respawn() -> None:
            await asyncio.sleep(self._RECONNECT_WATCHER_SLOW_RETRY_SECS)
            task = getattr(self, "_reconnect_watcher_task", None)
            if (
                not getattr(self, "_running", False)
                or not getattr(self, "_failed_platforms", None)  # queue drained while waiting
                or (task is not None and not task.done())  # a watcher came back; stand down
            ):
                return
            logger.warning(
                "Reconnect watcher still down with %d platform(s) queued — "
                "slow respawn %d/%d", len(self._failed_platforms), attempt + 1,
                self._MAX_SLOW_WATCHER_RESPAWNS,
            )
            self._spawn_reconnect_watcher(
                on_give_up=lambda _name: self._schedule_slow_reconnect_watcher_respawn(attempt=attempt + 1)
            )

        self._retain_background_task(asyncio.create_task(_slow_respawn()))

    def _spawn_reconnect_watcher(self, *, on_give_up=None):
        """Launch the reconnect watcher. ``on_spawn`` is load-bearing: without it a supervised
        respawn leaves ``_reconnect_watcher_task`` dead and ``_ensure_...`` spawns a second one."""
        self._reconnect_watcher_task = self._spawn_supervised(
            self._platform_reconnect_watcher, "platform_reconnect_watcher",
            on_spawn=lambda t: setattr(self, "_reconnect_watcher_task", t),
            on_give_up=on_give_up or self._on_reconnect_watcher_gave_up,
        )
        return self._reconnect_watcher_task

    def _ensure_reconnect_watcher_running(self) -> None:
        """Respawn a dead reconnect watcher (called on BOTH _queue_retryable_fatal_platform paths:
        the re-fatal of an already-queued platform is the only case that exhausts the budget)."""
        task = getattr(self, "_reconnect_watcher_task", None)
        if not getattr(self, "_running", False) or (task is not None and not task.done()):
            return  # not running, or already alive
        logger.warning(
            "Reconnect watcher task is dead (done=%s) — respawning",
            task.done() if task is not None else "N/A",
        )
        self._spawn_reconnect_watcher()

    async def _platform_reconnect_watcher(self) -> None:
        """Periodically retry failed platforms: backoff 30s → 300s cap, retryable failures retry
        forever (self-heal), non-retryable drop out. Pausing is manual only (``/platform pause``)."""
        async def _idle(seconds: int, until_queued: bool = False) -> bool:
            """Sleep in 1s steps; False once the runner stops (or, if asked, once work is queued)."""
            for _ in range(seconds):
                if not self._running:
                    return False
                if until_queued and self._failed_platforms:
                    break
                await asyncio.sleep(1)
            return True

        await asyncio.sleep(10)  # initial delay — let startup finish
        while self._running:
            if not self._failed_platforms:
                if not await _idle(30, until_queued=True):
                    return
                continue
            now = time.monotonic()
            for platform in list(self._failed_platforms.keys()):
                if not self._running:
                    return
                await self._reconnect_failed_platform(platform, now)
            if not await _idle(10):  # re-check every 10 seconds
                return

    def _flag_reconnect_needs_attention(self, platform, info: dict, now: float) -> None:
        """Flag NEEDS_ATTENTION (once) past the threshold — a signal, NOT a circuit breaker."""
        from gateway.run import _reconnect_needs_attention
        if info.get("attention_flagged") or not _reconnect_needs_attention(info, now):
            return
        info["attention_flagged"] = True
        queued_for = now - info.get("queued_at", now)
        retrying_since_iso = (datetime.now(timezone.utc) - timedelta(seconds=queued_for)).isoformat()
        logger.warning(
            "%s has been failing/reconnecting continuously for "
            "%.1f hours (%d attempts) — flagging NEEDS_ATTENTION. "
            "Retries continue, but this usually means a permanent "
            "problem (revoked credentials, missing intents, broken "
            "sidecar). Check `hermes status` / `/platform list`.", platform.value,
            queued_for / 3600.0, info.get("attempts", 0),
        )
        self._update_platform_runtime_status(
            platform.value, platform_state="retrying", needs_attention=True,
            retrying_since=retrying_since_iso,
        )

    def _bump_reconnect_backoff(
        self, platform, info: dict, attempt: int, error_code, error_message: str
    ) -> int:
        """Mark the platform retrying and record the failed attempt; returns the backoff applied."""
        from gateway.run import _reconnect_backoff
        self._update_platform_runtime_status(
            platform.value, platform_state="retrying", error_code=error_code, error_message=error_message,
        )
        backoff = _reconnect_backoff(attempt)
        info["attempts"] = attempt
        info["next_retry"] = time.monotonic() + backoff
        return backoff

    async def _reconnect_failed_platform(self, platform, now: float) -> None:
        """One watcher pass for a queued platform: gate, attempt, and record the outcome."""
        from gateway.run import _dispose_unused_adapter, _platform_has_bot_credential
        info = self._failed_platforms.get(platform)
        # None: removed concurrently since the caller's snapshot. Paused needs /platform resume.
        if info is None or info.get("paused"):
            return
        self._flag_reconnect_needs_attention(platform, info, now)
        if now < info["next_retry"]:
            return  # not time yet

        platform_config = info["config"]
        attempt = info["attempts"] + 1
        # Empty-token primary configs can never reconnect; drop them so multiplex setups
        # where a secondary profile owns the bot do not spin forever.
        if not _platform_has_bot_credential(platform, platform_config):
            self._drop_from_reconnect_queue(platform, "no bot credential on queued config")
            return
        logger.info("Reconnecting %s (attempt %d)...", platform.value, attempt)

        adapter = None
        try:
            adapter = self._create_adapter(platform, platform_config)
            if not adapter:
                self._drop_from_reconnect_queue(platform, "adapter creation returned None")
                return

            self._wire_adapter_handlers(adapter)
            # Reconnect after outage: keep the platform's server-side update queue so
            # messages sent while the bot was offline are delivered rather than dropped.
            success = await self._connect_adapter_with_timeout(adapter, platform, is_reconnect=True)
            if success:
                await self._install_reconnected_adapter(platform, adapter)
            elif adapter.has_fatal_error and not adapter.fatal_error_retryable:
                self._update_platform_runtime_status(
                    platform.value, platform_state="fatal", error_code=adapter.fatal_error_code,
                    error_message=adapter.fatal_error_message,
                )
                logger.warning(
                    "Reconnect %s: non-retryable error (%s), removing from retry queue",
                    platform.value, adapter.fatal_error_message,
                )
                # Never installed on self.adapters, so nothing else will disconnect() it: dispose or
                # the resource owners built in __init__ leak ~2 fds each (fd limit in ~12h at the cap).
                await _dispose_unused_adapter(adapter)
                del self._failed_platforms[platform]
            else:
                # Retryable failures retry at the backoff cap forever; never auto-pause them — a
                # transient outage must not need `/platform resume`. Same fd-leak dispose as above.
                backoff = self._bump_reconnect_backoff(
                    platform, info, attempt, adapter.fatal_error_code,
                    adapter.fatal_error_message or "failed to reconnect",
                )
                logger.info("Reconnect %s failed, next retry in %ds", platform.value, backoff)
                await _dispose_unused_adapter(adapter)
        except Exception as e:
            if adapter is not None:
                # An exception escaping connect leaves the adapter in the same unowned state.
                await _dispose_unused_adapter(adapter)
            # A reconnect exception is transient; keep retrying at the cap rather than auto-pausing.
            backoff = self._bump_reconnect_backoff(platform, info, attempt, None, str(e))
            logger.warning("Reconnect %s error: %s, next retry in %ds", platform.value, e, backoff)

    def _drop_from_reconnect_queue(self, platform, reason: str) -> None:
        logger.warning("Reconnect %s: %s, removing from retry queue", platform.value, reason)
        del self._failed_platforms[platform]

    async def _install_reconnected_adapter(self, platform, adapter) -> None:
        """Publish a freshly reconnected primary adapter and replay what it missed while down."""
        self.adapters[platform] = adapter
        self._sync_voice_mode_state_to_adapter(adapter)
        self._bind_voice_input_callback(adapter)
        self.delivery_router.adapters = self.adapters
        del self._failed_platforms[platform]
        self._update_platform_runtime_status(
            platform.value, platform_state="connected", error_code=None, error_message=None,
            needs_attention=False, retrying_since=None,
        )
        logger.info("✓ %s reconnected successfully", platform.value)

        # Final responses rejected while this adapter was down are still owned by this live
        # process, so startup recovery cannot claim them. Replay the transient subset now.
        try:
            await self._redeliver_failed_obligations_for_platform(platform)
        except Exception:
            logger.debug(
                "failed-obligation redelivery after %s reconnect failed", platform.value, exc_info=True,
            )

        # Rebuild channel directory with the new adapter
        try:
            from gateway.channel_directory import build_channel_directory
            await build_channel_directory(self.adapters)
        except Exception:
            pass

        # A platform offline at gateway startup never got its restart-interrupted sessions
        # auto-resumed — the startup pass skips sessions whose adapter isn't connected yet.
        try:
            self._schedule_resume_pending_sessions(platform=platform)
        except Exception:
            logger.debug(
                "resume-pending reschedule after %s reconnect failed", platform.value, exc_info=True
            )

    async def _cancel_secondary_profile_reconnect_tasks(self) -> None:
        """Cancel profile-scoped reconnects before tearing down their registry, so a reconnect
        mid-setup cannot republish an adapter after the registry drains (bounded wait)."""
        pending = self._profile_failed_platforms
        if not isinstance(pending, dict):
            return
        current = asyncio.current_task()
        tasks = [
            task
            for profile_pending in pending.values()
            if isinstance(profile_pending, dict)
            for task in profile_pending.values()
            if isinstance(task, asyncio.Task) and task is not current and not task.done()
        ]
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
        """Bring up adapters for every non-active profile (multiplex only); returns connected count.

        Each profile connects under its own HERMES_HOME + secret scope; credential/listener
        collisions are refused here — the only point seeing every profile's credentials together.
        """
        from gateway.run import (
            MultiplexConfigError, SecondaryPortBindingConfigError, _multiplex_profile_homes
        )
        if not getattr(self.config, "multiplex_profiles", False):
            return 0

        try:
            from hermes_cli.profiles import get_active_profile_name
        except Exception:
            return 0

        active = get_active_profile_name() or "default"
        connected = 0
        claimed = self._primary_resource_claims(active)
        profile_homes = _multiplex_profile_homes(self.config)
        for profile_name, profile_home in profile_homes:
            if profile_name == active:
                continue  # handled by the primary startup loop
            try:
                connected += await self._start_one_profile_adapters(profile_name, profile_home, claimed)
            except SecondaryPortBindingConfigError as e:
                logger.warning(
                    "Skipping secondary profile '%s' due to port-binding config error: %s", profile_name, e,
                )
            except MultiplexConfigError:
                raise
            except Exception as e:
                logger.error(
                    "Failed to start adapters for profile '%s': %s", profile_name, e, exc_info=True
                )

        self._record_served_profiles(active, profile_homes)
        return connected

    def _primary_resource_claims(self, active: str) -> Dict[tuple, str]:
        """Resource claim -> owning profile for every live or queued primary adapter (credential:
        one account polled once; listener: one bind+port). A queued retryable primary owns both."""
        claimed: Dict[tuple, str] = {}
        for _plat, _ad in self.adapters.items():
            fp = self._adapter_credential_fingerprint(_ad)
            for claim in ((_plat, fp) if fp is not None else None, self._adapter_listener_claim(_plat, _ad)):
                if claim is not None:
                    claimed[claim] = active
        for retry_info in getattr(self, "_failed_platforms", {}).values():
            for claim_name in ("credential_claim", "listener_claim"):
                retry_claim = retry_info.get(claim_name)
                if isinstance(retry_claim, tuple):
                    claimed[retry_claim] = active
        return claimed

    def _record_served_profiles(self, active: str, profile_homes) -> None:
        """Record the served set (eligible for routing/HTTP prefixes/cron/runtime scope — broader
        than "has a connected adapter") for `hermes status`; seed per-profile PairingStores."""
        try:
            from gateway.status import write_runtime_status
            from gateway.pairing import PairingStore
            served = [active] + sorted(name for name, _home in profile_homes if name != active)
            for name in served:
                if name and name not in self.pairing_stores:
                    self.pairing_stores[name] = (
                        self.pairing_store if name == active else PairingStore(profile=name)
                    )
            write_runtime_status(served_profiles=served)
        except Exception:
            logger.debug("could not record served_profiles", exc_info=True)

    async def _load_secondary_profile_config(self, profile_name: str, profile_home: "Path"):
        """Hydrate + enter ``profile_home``'s scope once; return its gateway config.

        Raises ``MultiplexConfigError`` (open dm/group policy) or ``SecondaryPortBindingConfigError``
        (the default profile owns the single shared HTTP listener).
        """
        from gateway.run import (
            MultiplexConfigError, SecondaryPortBindingConfigError, _load_gateway_runtime_config,
            _own_policy_open_startup_violation, _profile_runtime_scope,
        )
        from gateway.config import load_gateway_config
        from hermes_cli.env_loader import hydrate_profile_secret_sources

        # Hydrate external secret sources (1Password/vault/...) off-loop ONCE: the sync hydration is
        # network-bound and would stall every other profile's heartbeat while this one boots.
        await asyncio.to_thread(hydrate_profile_secret_sources, profile_home)

        with _profile_runtime_scope(profile_home, hydrate_secrets=False):
            profile_runtime_cfg = _load_gateway_runtime_config()
            from hermes_cli.plugins import discover_plugins
            discover_plugins()

            # Register this profile's shell hooks / outbound webhooks: start() registers before any
            # profile scope exists, so a secondary profile's `hooks:` block would be silently inert.
            try:
                from hermes_cli.config import load_config as _load_profile_config
                from agent.shell_hooks import register_from_config as _register_shell_hooks
                from agent.outbound_webhooks import (register_from_config as _register_outbound_webhooks)

                _profile_hooks_cfg = _load_profile_config()
                _register_shell_hooks(_profile_hooks_cfg, accept_hooks=False)
                _register_outbound_webhooks(_profile_hooks_cfg)
            except Exception:
                logger.warning(
                    "shell-hook/webhook registration failed for profile '%s'", profile_name, exc_info=True,
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
            if platform_config.enabled and _platform_binds_port(platform.value, platform_config.extra)
        )
        if port_binding_platforms:
            raise SecondaryPortBindingConfigError(
                f"Profile '{profile_name}' enables port-binding platform(s) "
                f"{', '.join(port_binding_platforms)}, but gateway.multiplex_profiles is on. The default "
                f"profile owns the single shared HTTP listener and serves every "
                f"profile through the /p/{profile_name}/ URL prefix. Remove "
                f"these platform entries from profile '{profile_name}'s config.yaml "
                f"or configure them only on the default profile."
            )
        return profile_cfg

    def _refuse_duplicate_claim(
        self, claim, claimed: Dict[tuple, str], profile_name: str, platform: Platform, kind: str
    ) -> bool:
        """Log + park a secondary adapter whose credential/listener another profile owns (True when
        refused). NOT disconnected: it never connected, and for a same-credential Photon adapter
        disconnect() would shut down the primary profile's live sidecar."""
        owner = claimed.get(claim) if claim is not None else None
        if owner is None:
            return False
        if kind == "credential":
            message = (
                f"Profile '{owner}' and '{profile_name}' both configure "
                f"{platform.value} with the same credential. Give each "
                f"profile its own {platform.value} credential."
            )
            logger.error(
                "Profile '%s' and '%s' both configure %s with the same "
                "credential — refusing to start the duplicate (one "
                "credential cannot be consumed twice). Give each profile "
                "its own %s credential.", owner, profile_name, platform.value, platform.value,
            )
        else:
            bind, port = claim[-2:]
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
                owner, profile_name, platform.value, bind, port, platform.value, profile_name,
            )
        self._update_platform_runtime_status(
            f"{profile_name}:{platform.value}", platform_state="fatal",
            error_code=f"duplicate_{kind}", error_message=message,
        )
        return True

    async def _start_one_profile_adapters(
        self, profile_name: str, profile_home: "Path", claimed: Dict[tuple, str]
    ) -> int:
        """Create+connect one profile's adapters under its runtime scope."""
        from gateway.run import _platform_has_bot_credential, _profile_runtime_scope
        profile_cfg = await self._load_secondary_profile_config(profile_name, profile_home)
        multiplex = getattr(self.config, "multiplex_profiles", False)
        profile_map = self._profile_adapters.setdefault(profile_name, {})
        connected = 0
        for platform, platform_config in profile_cfg.platforms.items():
            if not platform_config.enabled:
                continue
            # Enabled in the shared YAML but no credential in THIS profile's scope: building an
            # adapter anyway would fan one inbound message out across every credential-less profile.
            if multiplex and not _platform_has_bot_credential(platform, platform_config):
                logger.info(
                    "[MULTIPLEX] Profile '%s': skipping %s - no bot credential "
                    "in this profile's secrets", profile_name, platform.value,
                )
                continue
            # Relay/WhatsApp are shared process-level ingress under multiplex (the active profile's
            # one connection fans out by route); a secondary would only retry-loop and stall startup.
            if multiplex and platform in (Platform.RELAY, Platform.WHATSAPP):
                continue
            try:
                with _profile_runtime_scope(profile_home, hydrate_secrets=False):
                    adapter = self._create_adapter(platform, platform_config)
            except Exception as e:
                logger.error(
                    "[MULTIPLEX] Profile '%s': _create_adapter('%s') raised %s", profile_name,
                    platform.value, e, exc_info=True,
                )
                continue
            if not adapter:
                logger.warning(
                    "[MULTIPLEX] Profile '%s': skipping platform '%s' - adapter creation returned None",
                    profile_name, platform.value,
                )
                continue
            # Same-token / same-listener conflict detection — refuse a duplicate poll or bind.
            credential_claim = self._adapter_credential_claim(platform, adapter)
            listener_claim = self._adapter_listener_claim(platform, adapter)
            if self._refuse_duplicate_claim(
                credential_claim, claimed, profile_name, platform, "credential"
            ) or self._refuse_duplicate_claim(listener_claim, claimed, profile_name, platform, "listener"):
                continue
            self._configure_profile_adapter(adapter, profile_name, platform)
            try:
                with _profile_runtime_scope(profile_home, hydrate_secrets=False):
                    success = await self._connect_initial_adapter_with_timeout(adapter, platform)
                if not success:
                    logger.warning("✗ %s failed to connect (profile: %s)", platform.value, profile_name)
            except Exception as e:
                logger.error("✗ %s error (profile: %s): %s", platform.value, profile_name, e)
                success = False
            if not success:
                await self._safe_adapter_disconnect(adapter, platform)
                self._schedule_secondary_profile_startup_reconnect(profile_name, platform, adapter)
                continue
            profile_map[platform] = adapter
            # Restore persisted /voice state for this bot (primary startup and reconnects do too).
            self._sync_voice_mode_state_to_adapter(adapter)
            for claim in (credential_claim, listener_claim):
                if claim is not None:
                    claimed[claim] = profile_name
            connected += 1
            logger.info("✓ %s connected (profile: %s)", platform.value, profile_name)
        return connected

    def _wire_adapter_handlers(
        self, adapter: BasePlatformAdapter, *, message_handler=None, fatal_error_handler=None,
        busy_session_handler=None, authorization_check=None, platform_event_handler=None,
        busy_text_mode: Optional[str] = None,
    ) -> None:
        """Install the runner callbacks every adapter needs (defaults = primary handlers;
        secondary wiring passes profile-scoped variants). ``set_reaction_handler`` is optional."""
        adapter.set_message_handler(message_handler or self._primary_message_handler())
        adapter.set_fatal_error_handler(fatal_error_handler or self._handle_adapter_fatal_error)
        adapter.set_session_store(self.session_store)
        adapter.set_busy_session_handler(busy_session_handler or self._handle_active_session_busy_message)
        _set_reaction = getattr(adapter, "set_reaction_handler", None)
        if callable(_set_reaction):
            _set_reaction(self._handle_reaction_event)
        adapter.set_topic_recovery_fn(self._recover_telegram_topic_thread_id)
        adapter.set_authorization_check(
            authorization_check or self._make_adapter_auth_check(adapter.platform)
        )
        adapter.set_platform_event_handler(platform_event_handler or self._primary_platform_event_handler())
        adapter._busy_text_mode = (self._busy_text_mode if busy_text_mode is None else busy_text_mode)

    def _configure_profile_adapter(
        self, adapter: BasePlatformAdapter, profile_name: str, platform: Platform
    ) -> None:
        """Install the profile-scoped handlers shared by startup and reconnect."""
        # Runtime status is process-scoped while message/config work is profile-scoped. Keep both
        # dimensions in the key so dashboard/NAS health aggregation sees which secondary failed.
        adapter._runtime_status_platform_key = f"{profile_name}:{platform.value}"
        # Declare ownership BEFORE any inbound event: adapter-level session keys are derived at
        # ingress, before the handler stamps source.profile — without this every secondary bot
        # would key into the default profile's `agent:main:` lane.
        _set_owner = getattr(adapter, "set_owner_profile", None)
        if callable(_set_owner):
            _set_owner(profile_name)
        text_modes = getattr(self, "_busy_text_modes_by_profile", None)
        self._wire_adapter_handlers(
            adapter,
            message_handler=self._make_profile_message_handler(profile_name),
            fatal_error_handler=self._make_profile_fatal_error_handler(profile_name, platform),
            busy_session_handler=self._make_profile_busy_session_handler(profile_name),
            authorization_check=self._make_adapter_auth_check(platform, profile_name=profile_name),
            platform_event_handler=self._make_profile_platform_event_handler(profile_name),
            busy_text_mode=(
                text_modes.get(profile_name, self._busy_text_mode)
                if isinstance(text_modes, dict)
                else self._busy_text_mode
            ),
        )
        # Voice transcripts from this bot's channels dispatch through THIS adapter (primary
        # wiring lives at connect time).
        self._bind_voice_input_callback(adapter)
        # Secondary adapters always carry the profile they serve so prune paths namespace topic
        # bindings correctly under multiplex.
        adapter._hermes_profile_name = profile_name

    async def _secondary_reconnect_attempt(self, profile_name: str, platform: Platform):
        """One scoped attempt to rebuild+connect a secondary adapter → ``(adapter, success)``.

        ``(None, None)`` = give up for good (disabled, credential removed, adapter unavailable).
        Caller tears down a RETURNED adapter; one whose configure/connect raised is torn down here.
        """
        from gateway.run import _platform_has_bot_credential, _profile_runtime_scope
        # Lazy + per-attempt: keeps test monkeypatches on these modules live.
        from hermes_cli.profiles import get_profile_dir
        from hermes_cli.env_loader import hydrate_profile_secret_sources
        from gateway.config import load_gateway_config
        profile_home = get_profile_dir(profile_name)
        # Hydrate external secret sources off-loop so they cannot starve heartbeats.
        await asyncio.to_thread(hydrate_profile_secret_sources, profile_home)
        with _profile_runtime_scope(profile_home, hydrate_secrets=False):
            profile_config = load_gateway_config().platforms.get(platform)
            if profile_config is None or not profile_config.enabled:
                return None, None
            # Startup credential gate mirror: a removed credential must not rebuild.
            if not _platform_has_bot_credential(platform, profile_config):
                logger.info(
                    "Secondary %s reconnect skipped: no bot credential (profile: %s)",
                    platform.value, profile_name,
                )
                return None, None
            adapter = self._create_adapter(platform, profile_config)
            if adapter is None:
                logger.warning(
                    "Secondary %s reconnect skipped: adapter unavailable (profile: %s)",
                    platform.value, profile_name,
                )
                return None, None
            try:
                self._configure_profile_adapter(adapter, profile_name, platform)
                success = await self._connect_adapter_with_timeout(adapter, platform, is_reconnect=True)
            except BaseException:
                # Caller never sees this adapter; release its partial resources here.
                await self._safe_adapter_disconnect(adapter, platform)
                raise
            return adapter, success

    async def _run_secondary_profile_reconnect(self, profile_name: str, platform: Platform) -> None:
        """Reconnect a retryable secondary adapter under its own profile scope."""
        from gateway.run import _reconnect_backoff
        attempts = 0
        current_task = asyncio.current_task()
        try:
            while self._running:
                adapter = None
                try:
                    adapter, success = await self._secondary_reconnect_attempt(profile_name, platform)
                    if adapter is None:
                        return
                    if success and self._running:
                        profile_map = self._profile_adapters.setdefault(profile_name, {})
                        if platform not in profile_map:
                            profile_map[platform] = adapter
                            self._sync_voice_mode_state_to_adapter(adapter)
                            logger.info("✓ %s reconnected (profile: %s)", platform.value, profile_name)
                            await self._redeliver_failed_obligations_for_platform(
                                platform, profile=profile_name
                            )
                            return
                    # Not installed: a newer reconnect won the slot mid-connect (never replace it),
                    # shutdown began mid-connect (never republish after the registry drained), or
                    # connect failed. Release partial resources; stop only for a non-retryable fatal.
                    await self._safe_adapter_disconnect(adapter, platform)
                    if success or (
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
                        "Secondary %s reconnect attempt failed (profile: %s)", platform.value,
                        profile_name, exc_info=True,
                    )

                if not self._running:
                    return
                attempts += 1
                backoff = _reconnect_backoff(attempts)
                logger.info(
                    "Secondary %s reconnect retry in %ds (profile: %s)", platform.value, backoff,
                    profile_name,
                )
                await asyncio.sleep(backoff)
        finally:
            # Release our slot unless a newer task already owns it.
            pending = self._profile_failed_platforms
            profile_pending = pending.get(profile_name) if isinstance(pending, dict) else None
            if isinstance(profile_pending, dict):
                task = profile_pending.get(platform)
                if not isinstance(task, asyncio.Task) or task is current_task:
                    profile_pending.pop(platform, None)
                    if not profile_pending:
                        pending.pop(profile_name, None)

    def _schedule_secondary_profile_startup_reconnect(
        self, profile_name: str, platform: Platform, adapter: BasePlatformAdapter
    ) -> None:
        """Queue a cold-start reconnect: startup failures happen BEFORE ``_running`` flips True (the
        regular scheduler would drop them), so park a task and hand off once live."""
        if not getattr(adapter, "fatal_error_retryable", True):
            return
        if is_global_startup_conflict(getattr(adapter, "fatal_error_code", None)):
            # A live foreign holder of this profile's token is an ownership conflict, not a blip:
            # park it fatal (like ``duplicate_credential``) instead of retry-storming.
            logger.error(
                "[MULTIPLEX] Profile '%s': %s credential is held by another "
                "gateway (%s) — parked, not retried. %s", profile_name, platform.value,
                adapter.fatal_error_code, adapter.fatal_error_message or "",
            )
            self._update_platform_runtime_status(
                f"{profile_name}:{platform.value}", platform_state="fatal",
                error_code=adapter.fatal_error_code, error_message=adapter.fatal_error_message,
            )
            return

        def _handoff() -> None:
            try:
                self._schedule_secondary_profile_reconnect(profile_name, platform, adapter)
            except Exception:
                # The handoff touches live registries; if it raises, the parked task dies as an
                # unretrieved-task exception logged only at GC. Surface it where operators look.
                logger.exception(
                    "secondary-startup-reconnect handoff failed (profile=%s platform=%s)",
                    profile_name, platform.value,
                )

        async def _await_running_then_schedule() -> None:
            if self._running:
                _handoff()
                return
            # Poll (startup completion has no event); bounded so a wedged startup cannot spin.
            while not self._running and not self._shutdown_event.is_set():
                await asyncio.sleep(0.1)
            if self._running and not self._shutdown_event.is_set():
                _handoff()

        self._retain_background_task(asyncio.create_task(
            _await_running_then_schedule(),
            name=f"secondary-startup-reconnect:{profile_name}:{platform.value}",
        ))

    def _schedule_secondary_profile_reconnect(
        self, profile_name: str, platform: Platform, adapter: BasePlatformAdapter
    ) -> None:
        """Schedule one runner-owned reconnect without sharing primary secrets."""
        if not self._running or not adapter.fatal_error_retryable:
            return
        pending = self._profile_failed_platforms
        if not isinstance(pending, dict):
            pending = self._profile_failed_platforms = {}
        profile_pending = pending.setdefault(profile_name, {})
        if platform in profile_pending:
            return
        profile_pending[platform] = self._retain_background_task(asyncio.create_task(
            self._run_secondary_profile_reconnect(profile_name, platform),
            name=f"secondary-reconnect:{profile_name}:{platform.value}",
        ))

    def _make_profile_fatal_error_handler(
        self, profile_name: str, platform: Platform
    ) -> Callable[[BasePlatformAdapter], Awaitable[None]]:
        """Route a secondary-profile fatal error to that profile's reconnect slot."""
        async def _handler(adapter: BasePlatformAdapter) -> None:
            await self._handle_profile_adapter_fatal_error(profile_name, platform, adapter)

        return _handler

    async def _handle_profile_adapter_fatal_error(
        self, profile_name: str, platform: Platform, adapter: BasePlatformAdapter
    ) -> None:
        """Remove a failed multiplexed adapter (the primary-only fatal handler ignores them)."""
        profile_map = getattr(self, "_profile_adapters", {}).get(profile_name)
        if not isinstance(profile_map, dict) or profile_map.get(platform) is not adapter:
            logger.debug(
                "Ignoring stale fatal error from secondary %s adapter (profile: %s)",
                platform.value, profile_name,
            )
            return
        profile_map.pop(platform, None)
        await self._safe_adapter_disconnect(adapter, platform)
        if not self._running:
            return
        self._schedule_secondary_profile_reconnect(profile_name, platform, adapter)
        logger.error(
            "Fatal %s adapter error for multiplexed profile %s (%s)", platform.value, profile_name,
            adapter.fatal_error_code or "unknown",
        )

    @staticmethod
    def _profile_home_or_none(profile_name: str):
        from hermes_cli.profiles import get_profile_dir
        try:
            return get_profile_dir(profile_name)
        except Exception:
            return None

    @staticmethod
    def _stamp_event_profile(event, profile_name: str) -> None:
        """Best-effort: stamp ``source.profile`` on an inbound event that has none yet."""
        try:
            if getattr(event, "source", None) is not None and not event.source.profile:
                event.source.profile = profile_name
        except Exception:
            pass

    def _make_profile_message_handler(self, profile_name: str):
        """Message handler that stamps source.profile, then delegates under the profile scope
        (auth runs BEFORE the agent-turn scope, so the profile's ``.env`` must be visible here)."""
        from gateway.run import _async_profile_runtime_scope
        profile_home = self._profile_home_or_none(profile_name)

        async def _handler(event):
            self._stamp_event_profile(event, profile_name)
            async with (
                _async_profile_runtime_scope(profile_home) if profile_home is not None
                else contextlib.nullcontext()
            ):
                return await self._handle_message(event)

        return _handler

    def _make_profile_busy_session_handler(self, profile_name: str):
        """Stamp an owning adapter's profile before resolving busy policy."""
        async def _handler(event, _session_key):
            self._stamp_event_profile(event, profile_name)
            routed_session_key = self._session_key_for_source(event.source)
            return await self._handle_active_session_busy_message(event, routed_session_key)

        return _handler

    def _make_default_profile_message_handler(self):
        """Scope primary-adapter messages to their routed multiplex profile. Authorization stays
        with the transport profile (a routed profile may have no credential/allowlist)."""
        from gateway.run import _async_profile_runtime_scope, get_hermes_home
        default_home = Path(get_hermes_home())

        async def _handler(event):
            source = event.source
            # In-process only (serialization ignores dynamic attrs); route ≠ admitting bot.
            source._authorization_profile_home = default_home
            if (
                not getattr(source, "profile", None)
                and getattr(source, "profile_route_rejected", False) is not True
                and not self._stamp_routed_profile(source)
            ):
                # Read by the ``_handle_message`` ingress gate, which drops fail-closed.
                source.profile_route_rejected = True
            profile_home = (
                self._resolve_profile_home_for_source(source)
                if getattr(source, "profile", None) else default_home
            )
            async with _async_profile_runtime_scope(profile_home):
                return await self._handle_message(event)

        return _handler

    def _stamp_routed_profile(self, source) -> bool:
        """Stamp ``source.profile`` from ``profile_routes``; False when the route is rejected."""
        from gateway.profile_routing import ProfileRouteRejected
        try:
            source.profile = self._profile_name_for_source(source)
        except ProfileRouteRejected:
            return False
        return True

    def _primary_message_handler(self):
        """Return the correctly scoped handler for a primary adapter."""
        if getattr(self.config, "multiplex_profiles", False):
            return self._make_default_profile_message_handler()
        return self._handle_message

    async def _handle_gateway_platform_event(self, event: dict, source) -> None:
        """Authorize and publish one normalized adapter event to plugin hooks."""
        try:
            from hermes_cli.lifecycle import has_hook, invoke_hook
            if has_hook("gateway_platform_event") and self._is_user_authorized_for_source(source):
                invoke_hook("gateway_platform_event", **event)
        except Exception:
            # Observer failures must never break the adapter's update loop.
            logger.debug("gateway_platform_event hook dispatch failed", exc_info=True)

    def _make_profile_platform_event_handler(self, profile_name: str):
        """Bind platform-event auth and hook dispatch to one multiplex profile."""
        from gateway.run import _profile_runtime_scope
        profile_home = self._profile_home_or_none(profile_name)

        async def _handler(event, source):
            if getattr(source, "profile", None) is None:
                source.profile = profile_name
            with (
                _profile_runtime_scope(profile_home) if profile_home is not None
                else contextlib.nullcontext()
            ):
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
    def _adapter_credential_claim(platform: Platform, adapter: Any) -> Optional[tuple]:
        """Return the exclusive credential resource claimed by an adapter."""
        from gateway.run import GatewayRunner
        fingerprint = GatewayRunner._adapter_credential_fingerprint(adapter)
        return None if fingerprint is None else (platform, fingerprint)

    @staticmethod
    def _adapter_listener_claim(platform: Platform, adapter: Any) -> Optional[tuple]:
        """Exclusive listener claim (Photon sidecar bind+port): distinct credentials still cannot
        share a port, so the later adapter is rejected before connect() disturbs the first."""
        bind = getattr(adapter, "_sidecar_bind", None)
        if getattr(platform, "value", None) != "photon" or not isinstance(bind, str) or not bind.strip():
            return None
        try:
            port = int(getattr(adapter, "_sidecar_port", None))
        except (TypeError, ValueError):
            return None
        return ("listener", "photon", bind.strip().lower(), port)

    @staticmethod
    def _adapter_credential_fingerprint(adapter: Any) -> Optional[str]:
        """Salted, log-safe hash of an adapter's credential; None when none is discoverable
        (conflict detection is then skipped)."""
        # Many adapters (e.g. Discord) keep the token on `config`; without that fallback the
        # same-token check is silently skipped and every profile polls the same bot.
        candidates = [
            (adapter, attr) for attr in (
                "token", "bot_token", "_token", "api_token", "_bot_token",
                "_project_secret",  # Photon/Spectrum: project credentials, not a bot token
                "_app_id",  # Feishu/Lark app_id — stable, log-safe, already the _app_lock_identity
                "_client_id", "_bot_id",  # Teams / WeCom app-style id pairs
            )
        ] + [(getattr(adapter, "config", None), attr) for attr in ("token", "bot_token")]
        for obj, attr in candidates:
            val = getattr(obj, attr, None)
            if isinstance(val, str) and val.strip():
                import hashlib
                return hashlib.sha256(("hermes-mux:" + val.strip()).encode("utf-8")).hexdigest()[:16]
        return None

    def _create_adapter(self, platform: Platform, config: Any) -> Optional[BasePlatformAdapter]:
        """Create an adapter bound to this runner (every lifecycle path goes through here so
        adapters can resolve inbound profile routes before handlers or connect())."""
        adapter = self._instantiate_adapter(platform, config)
        if adapter is not None:
            adapter.gateway_runner = self
        return adapter

    def _instantiate_adapter(self, platform: Platform, config: Any) -> Optional[BasePlatformAdapter]:
        """Instantiate the adapter for a platform: plugin registry first, then built-ins."""
        from gateway.run import _instantiate_builtin_adapter
        if hasattr(config, "extra") and isinstance(config.extra, dict):
            config.extra.setdefault("group_sessions_per_user", self.config.group_sessions_per_user)
            config.extra.setdefault(
                "thread_sessions_per_user", getattr(self.config, "thread_sessions_per_user", False)
            )

        try:
            from gateway.platform_registry import platform_registry
            if platform_registry.is_registered(platform.value):
                adapter = platform_registry.create_adapter(platform.value, config)
                if adapter is not None:
                    return adapter
                # Registered but failed — never fall through to built-ins.
                logger.error(
                    "Platform '%s' is registered but adapter creation failed "
                    "(check dependencies and config)", platform.value,
                )
                return None
        except Exception as e:
            logger.debug("Platform registry lookup for '%s' failed: %s", platform.value, e)
        return _instantiate_builtin_adapter(platform, config)

    def _make_adapter_auth_check(
        self, platform: Platform, profile_name: Optional[str] = None
    ) -> Callable[[str, Optional[str], Optional[str]], bool]:
        """Platform-bound auth callback for adapters (prompt-injection mitigation for fetched
        context); delegates to :meth:`_is_user_authorized`. ``profile_name`` binds a secondary to
        its scope; for the shared primary (None) the routed profile is stamped so its pairing store
        is consulted while allowlist reads stay under the transport home."""
        from gateway.run import get_hermes_home
        multiplex = bool(getattr(self.config, "multiplex_profiles", False))
        transport_home = (Path(get_hermes_home()) if multiplex and profile_name is None else None)

        def check(
            user_id: str, chat_type: Optional[str] = None, chat_id: Optional[str] = None, *,
            is_bot: bool = False, thread_id: Optional[str] = None,
        ) -> bool:
            if not user_id:
                return False
            source = SessionSource(
                platform=platform, chat_id=chat_id or "", chat_type=chat_type or "group",
                user_id=user_id, thread_id=thread_id, is_bot=bool(is_bot), profile=profile_name,
            )
            # Same transport provenance ``build_source`` retains, so adapter-level policy reads
            # resolve the receiving adapter even once the routed profile is stamped below.
            registry = (
                (getattr(self, "_profile_adapters", None) or {}).get(profile_name)
                if profile_name else getattr(self, "adapters", None)
            ) or {}
            adapter = registry.get(platform)
            if adapter is not None:
                source._transport_adapter_ref = _weakref.ref(adapter)
            if transport_home is None:
                return self._is_user_authorized(source)
            source._authorization_profile_home = transport_home
            if not self._stamp_routed_profile(source):
                return False  # fail-closed, like the ``_handle_message`` ingress gate
            return self._is_user_authorized_for_source(source)
        return check
