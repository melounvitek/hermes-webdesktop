"""Session expiry / stall / catalog-refresh watcher loops for GatewayRunner.

Split out of ``gateway/run.py``; bound onto ``GatewayRunner`` via the MRO.
``gateway.run`` internals are imported lazily inside method bodies (import cycle),
so ``patch("gateway.run.X")`` keeps intercepting them at call time.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import Counter
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:  # string annotations only; never imported at runtime (cycle)
    from gateway.run import GatewayRunner, TurnRunner  # noqa: F401

# Log-record parity with the origin module.
logger = logging.getLogger("gateway.run")

_MAX_FINALIZE_RETRIES = 3
_SESSION_STORE_PRUNE_INTERVAL = 3600.0  # once per hour


def _platform_of_key(key: str, default: str = "") -> str:
    """Session keys look like ``agent:main:telegram:dm:12345`` — platform is field [2]."""
    parts = key.split(":")
    return parts[2] if len(parts) > 2 else default


async def _interruptible_sleep(runner, seconds: int) -> None:
    """Sleep in 1s increments so the watcher stops quickly when ``runner._running`` flips."""
    for _ in range(seconds):
        if not runner._running:
            break
        await asyncio.sleep(1)


class GatewaySessionWatchersMixin:
    """Session expiry / stall / catalog-refresh watcher loops for GatewayRunner."""

    async def _session_expiry_watcher(self, interval: int = 300):
        """Finalize expired sessions (``on_session_finalize`` hooks, cached agent teardown, cache
        eviction, ``expiry_finalized`` flag) and run the cache/store sweeps."""
        await asyncio.sleep(60)  # initial delay — let the gateway fully start
        finalize_failures: dict[str, int] = {}  # session_id -> consecutive failure count
        while self._running:
            try:
                if expired := await self._collect_expired_sessions():
                    platforms = Counter(_platform_of_key(k, "unknown") for k, _ in expired)
                    logger.info(
                        "Session expiry: %d sessions to finalize (%s)",
                        len(expired), ", ".join(f"{p}:{c}" for p, c in sorted(platforms.items())),
                    )
                    await self._finalize_expired_sessions(expired, finalize_failures)
                    done = sum(1 for _, e in expired if e.expiry_finalized)
                    if failed := len(expired) - done:
                        logger.info(
                            "Session expiry done: %d finalized, %d pending retry", done, failed
                        )
                    else:
                        logger.info("Session expiry done: %d finalized", done)
                await self._expiry_housekeeping()
            except Exception as e:
                logger.debug("Session expiry watcher error: %s", e)
            await _interruptible_sleep(self, interval)

    async def _collect_expired_sessions(self) -> list:
        """Return ``[(session_key, entry)]`` for expired, not-yet-finalized sessions."""
        store = self.async_session_store
        await store._ensure_loaded()
        return [
            (key, entry) for key, entry in list(self.session_store._entries.items())
            if not entry.expiry_finalized and await store._is_session_expired(entry)
        ]

    async def _finalize_expired_sessions(self, expired: list, failures: dict[str, int]) -> None:
        """Finalize each entry; after ``_MAX_FINALIZE_RETRIES`` consecutive failures mark it
        finalized anyway (without clearing the model override) to stop an infinite retry loop."""
        for key, entry in expired:
            sid = entry.session_id
            try:
                await self._finalize_expired_session(key, entry)
            except Exception as e:
                count = failures[sid] = failures.get(sid, 0) + 1
                if count < _MAX_FINALIZE_RETRIES:
                    logger.debug(
                        "Session finalize failed (%d/%d) for %s: %s",
                        count, _MAX_FINALIZE_RETRIES, sid, e,
                    )
                    continue
                logger.warning(
                    "Session finalize gave up after %d attempts for %s: %s. "
                    "Marking as finalized to prevent infinite retry loop.",
                    count, sid, e,
                )
                store = self.async_session_store
                await store.set_expiry_finalized(entry, clear_model_override=False)
            failures.pop(sid, None)

    def _agent_for_expired_session(self, key: str):
        """Idle agents live in _agent_cache (not _running_agents); fall back to the running turn's
        agent in case the session is still mid-turn when the expiry fires."""
        cache_lock = getattr(self, "_agent_cache_lock", None)  # tests build runners without it
        if cache_lock is not None:
            with cache_lock:
                cached = self._agent_cache.get(key)
            agent = cached[0] if isinstance(cached, tuple) else cached if cached else None
            if agent is not None:
                return agent
        state = self._peek_session_state(key)
        return state.turn.agent if state else None

    async def _finalize_expired_session(self, key: str, entry) -> None:
        """Run finalize hooks, tear down the cached agent, clear conversation scope, persist."""
        from gateway.run import _AGENT_PENDING_SENTINEL

        try:
            # Off-loop + bounded: plugin finalize hooks can block arbitrarily, and this
            # watcher runs on the gateway event loop.
            await self._finalize_session_off_loop(
                session_id=entry.session_id,
                platform=_platform_of_key(key),
                reason="session_expired",
            )
        except Exception:
            pass
        agent = self._agent_for_expired_session(key)
        if agent and agent is not _AGENT_PENDING_SENTINEL:
            await self._cleanup_agent_resources_off_loop(agent, context="session expiry")
        # Evict so the AIAgent (LLM clients, tool schemas, memory refs) can be GC'd, then drop every
        # conversation-scoped dict AND boundary security state — only finalize, /new, /reset may
        # do this (idle-cache eviction must NOT: a resumed turn rebuilds from those overrides). The
        # persisted flag also drops the /model override — finalization is a conversation boundary.
        self._evict_cached_agent(key)
        self._clear_conversation_scope(key, reason="expiry_finalized")
        await self.async_session_store.set_expiry_finalized(entry)
        logger.debug("Session expiry finalized for %s", entry.session_id)

    async def _expiry_housekeeping(self) -> None:
        """Idle/pressure agent-cache sweeps plus the hourly SessionStore prune."""
        # Sweep agents idle beyond the TTL regardless of session reset policy: long / "never"
        # reset windows would otherwise pin memory for the gateway's life.
        try:
            if evicted := self._sweep_idle_cached_agents():
                logger.info("Agent cache idle sweep: evicted %d agent(s)", evicted)
        except Exception as e:
            logger.debug("Idle agent sweep failed: %s", e)
        # Neither LRU cap nor idle TTL knows what a cached transcript costs in memory.
        try:
            self._sweep_agent_cache_under_pressure()
        except Exception as e:
            logger.debug("Agent cache pressure sweep failed: %s", e)
        # Prune stale SessionStore entries; the in-memory dict (and sessions.json) would
        # otherwise grow unbounded with many rotating chats / threads / users.
        last_prune = getattr(self, "_last_session_store_prune_ts", 0.0)
        if time.time() - last_prune > _SESSION_STORE_PRUNE_INTERVAL:
            try:
                max_age = int(getattr(self.config, "session_store_max_age_days", 0) or 0)
                if max_age > 0:
                    pruned = await self.async_session_store.prune_old_entries(max_age)
                    if pruned:
                        logger.info("SessionStore prune: dropped %d stale entries", pruned)
            except Exception as e:
                logger.debug("SessionStore prune failed: %s", e)
            self._last_session_store_prune_ts = time.time()

    def _session_stall_timeout_seconds(self) -> float:
        """Return configured stall timeout (seconds); 0 disables the watchdog."""
        from gateway.run import _float_env
        return _float_env("HERMES_SESSION_STALL_TIMEOUT", 300)

    def _iter_gateway_adapters(self):
        """Yield every live platform adapter (default + multiplex profiles), deduped by identity."""
        seen: set[int] = set()
        maps = (getattr(self, "adapters", {}), *getattr(self, "_profile_adapters", {}).values())
        for amap in maps:
            for adapter in list(amap.values()):
                if adapter is not None and id(adapter) not in seen:
                    seen.add(id(adapter))
                    yield adapter

    def _session_activity_for_stall(self, session_key: str) -> Optional[dict]:
        """Activity snapshot for stall progress: the single source is
        ``AIAgent.get_activity_summary()``; no turn-start or pending-inbound clocks."""
        from gateway.run import _AGENT_PENDING_SENTINEL
        agent = (getattr(self, "_running_agents", None) or {}).get(session_key)
        if agent is None or agent is _AGENT_PENDING_SENTINEL:
            return None
        if not hasattr(agent, "get_activity_summary"):
            return None
        try:
            summary = agent.get_activity_summary()
        except Exception:
            return None
        return summary if isinstance(summary, dict) else None

    def _stall_candidates(self) -> Dict[str, tuple[Any, Any]]:
        """Map session_key -> (adapter, pending event) from adapter pending slots, then from the
        runner's overflow queues (first occurrence wins)."""
        candidates: Dict[str, tuple[Any, Any]] = {}
        for adapter in self._iter_gateway_adapters():
            pending_slot = getattr(adapter, "_pending_messages", None) or {}
            for session_key, event in list(pending_slot.items()):
                if session_key and session_key not in candidates and event is not None:
                    candidates[session_key] = (adapter, event)
        for session_key, overflow in list((getattr(self, "_queued_events", None) or {}).items()):
            if not session_key or session_key in candidates or not overflow:
                continue
            source = getattr(overflow[0], "source", None)
            adapter = self._adapter_for_source(source) if source is not None else None
            if adapter is not None:
                candidates[session_key] = (adapter, overflow[0])
        return candidates

    def _session_still_pending(self, adapter, session_key: str) -> bool:
        return (
            (getattr(adapter, "_pending_messages", None) or {}).get(session_key) is not None
            or bool((getattr(self, "_queued_events", None) or {}).get(session_key))
        )

    async def _check_session_stalls(self, timeout_seconds: float) -> int:
        """Scan pending inbound sessions and notify once per stall episode; returns the number of
        notifications sent this pass (for tests)."""
        from gateway.session_stall import (
            resolve_session_idle_seconds_from_activity,
            should_clear_session_stall_notification,
            should_emit_session_stall_notification,
        )

        notified_map = getattr(self, "_session_stall_notified", None)
        if notified_map is None:
            notified_map = self._session_stall_notified = {}
        sent, now, candidates = 0, time.time(), self._stall_candidates()
        # Every candidate carries a non-None pending event, so has_pending_inbound is always True.
        for session_key, (adapter, pending_event) in list(candidates.items()):
            activity = self._session_activity_for_stall(session_key)
            idle_seconds = resolve_session_idle_seconds_from_activity(activity, now=now)
            already = bool(notified_map.get(session_key))
            if should_clear_session_stall_notification(
                timeout_seconds=timeout_seconds, idle_seconds=idle_seconds, has_pending_inbound=True
            ):
                notified_map.pop(session_key, None)
                already = False
            if idle_seconds is None or not should_emit_session_stall_notification(
                timeout_seconds=timeout_seconds, idle_seconds=idle_seconds,
                has_pending_inbound=True, already_notified=already,
            ):
                continue
            if await self._notify_session_stall(
                session_key, adapter, pending_event, idle_seconds, activity or {},
                timeout_seconds, notified_map,
            ):
                sent += 1
        # Drop latches for sessions that no longer appear in any pending map.
        for key in list(notified_map.keys()):
            if key not in candidates:
                notified_map.pop(key, None)
        return sent

    async def _send_stall_notice(self, session_key: str, adapter, source, idle_seconds) -> bool:
        """Deliver one stall notice, bounded and failure-tolerant; True only when delivered."""
        from gateway.run import _STALL_NOTIFY_SEND_TIMEOUT_SECONDS
        from gateway.session_stall import format_session_stall_notification

        try:
            metadata = self._thread_metadata_for_source(source)
            notice = format_session_stall_notification(idle_seconds)
            # Bound the send: a wedged adapter transport (network hang, dead websocket) must not
            # block the watcher pass — siblings would go unevaluated and the watcher stop.
            try:
                result = await asyncio.wait_for(
                    adapter.send(str(source.chat_id), notice, metadata=metadata),
                    timeout=_STALL_NOTIFY_SEND_TIMEOUT_SECONDS,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "Session stall notify send timed out after %.0fs for %s; will retry next tick",
                    _STALL_NOTIFY_SEND_TIMEOUT_SECONDS, session_key,
                )
                return False
            # Adapters often return SendResult(success=False) instead of raising.
            if result is not None and getattr(result, "success", True) is False:
                logger.warning(
                    "Session stall notify failed for %s: %s",
                    session_key, getattr(result, "error", "send returned success=False"),
                )
                return False
        except Exception as exc:
            logger.warning("Session stall notify failed for %s: %s", session_key, exc)
            return False
        return True

    async def _notify_session_stall(
        self, session_key: str, adapter, pending_event, idle_seconds: float, activity: dict,
        timeout_seconds: float, notified_map: dict,
    ) -> bool:
        """Log one stall episode and deliver the notice. True only when sent (latched);
        undeliverable (no chat_id) latches without sending; send failures never latch."""
        from gateway.session_stall import resolve_session_idle_seconds_from_activity

        logger.warning(
            "Session stall detected: session=%s idle=%.0fs (timeout=%.0fs, ~%d min); pending "
            "inbound present | last_activity=%s | provenance=%s (agent.session_stall_timeout)",
            session_key, idle_seconds, timeout_seconds, max(1, int(idle_seconds // 60)),
            activity.get("last_activity_desc") or activity.get("last_activity_description")
            or "unknown",
            activity.get("provenance") or activity.get("last_activity_provenance") or "unknown",
        )
        source = getattr(pending_event, "source", None)
        if not getattr(source, "chat_id", None):
            logger.warning("Session stall notify skipped (no chat_id): session=%s", session_key)
            notified_map[session_key] = True  # cannot deliver; latch to avoid log spam every tick
            return False
        # Re-read pending state + activity IMMEDIATELY before delivery: the snapshot ages while
        # earlier candidates await sends; an agent that progressed (or drained its queue) must not
        # get a false stall notice. Abort with the latch un-set so the next tick re-evaluates.
        still_pending = self._session_still_pending(adapter, session_key)
        fresh_idle = resolve_session_idle_seconds_from_activity(
            self._session_activity_for_stall(session_key), now=time.time()
        )
        if not still_pending or (fresh_idle is not None and fresh_idle < timeout_seconds):
            logger.info(
                "Session stall notify aborted (no longer stale): "
                "session=%s pending=%s fresh_idle=%s",
                session_key, still_pending, fresh_idle,
            )
            notified_map.pop(session_key, None)  # re-arm so a FUTURE genuine stall notifies again
            return False
        if not await self._send_stall_notice(session_key, adapter, source, idle_seconds):
            return False
        notified_map[session_key] = True
        return True

    async def _model_catalog_refresh_watcher(self) -> None:
        """Refresh the /model picker's remote catalogs every TTL window. The picker itself only
        refreshes on a cold/stale open, so if nobody opens ``/model`` the cache never updates."""
        from hermes_cli.model_catalog import refresh_catalogs, refresh_interval_seconds

        await asyncio.sleep(30)  # let startup settle
        while self._running:
            try:
                await asyncio.to_thread(refresh_catalogs)
            except Exception as exc:
                logger.debug("Model catalog refresh failed: %s", exc)
            try:
                interval = refresh_interval_seconds()
            except Exception:
                interval = 1200.0
            deadline = time.monotonic() + interval
            while self._running and time.monotonic() < deadline:
                await asyncio.sleep(min(30.0, max(0.0, deadline - time.monotonic())))

    async def _session_stall_watcher(self, interval: float = 30.0):
        """Periodic pending-inbound + stale-activity stall watchdog. Progress comes only from
        ``get_activity_summary()``; pending inbound is a notify policy gate, not a progress clock.
        Notify-only: does not kill the turn (contrast ``gateway_timeout`` / ``shutdown_watchdog``).
        """
        # Short initial delay so startup reconnect noise does not false-fire.
        await asyncio.sleep(min(30.0, max(1.0, float(interval))))
        while self._running:
            try:
                if (timeout := self._session_stall_timeout_seconds()) > 0:
                    await self._check_session_stalls(timeout)
            except Exception as exc:
                logger.debug("Session stall watcher error: %s", exc)
            await _interruptible_sleep(self, max(1, int(float(interval))))
