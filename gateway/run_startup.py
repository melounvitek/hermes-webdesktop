"""Startup sequence, resume/restore and handoff methods for GatewayRunner.

Split out of ``gateway/run.py``; bound onto ``GatewayRunner`` via the MRO.
``gateway.run`` internals are imported lazily inside method bodies (import cycle),
so ``patch("gateway.run.X")`` keeps intercepting them at call time.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
import asyncio
import dataclasses
import faulthandler
import os
import signal
import time
from contextlib import suppress
from datetime import datetime
from gateway.config import Platform
from gateway.delivery import looks_like_telegram_private_chat_id
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType
from gateway.session import SessionSource, build_session_key
from gateway.restart import (
    DEFAULT_GATEWAY_CRON_DRAIN_TIMEOUT,
    GATEWAY_FATAL_CONFIG_EXIT_CODE,
    is_global_startup_conflict,
)
from gateway.shutdown_watchdog import (
    DEFAULT_HEARTBEAT_INTERVAL_S,
    DEFAULT_LOOP_WATCHDOG_INTERVAL_S,
    DEFAULT_LOOP_WATCHDOG_MAX_STRIKES,
    DEFAULT_LOOP_WATCHDOG_TIMEOUT_S,
    loop_heartbeat_forever,
)
from typing import Any, Dict, Optional, Tuple

if TYPE_CHECKING:  # string annotations only; never imported at runtime (cycle)
    from gateway.run import GatewayRunner, TurnRunner  # noqa: F401

# Log-record parity with the origin module.
logger = logging.getLogger("gateway.run")


class GatewayStartupMixin:
    """Startup sequence, resume/restore and handoff methods for GatewayRunner."""

    async def _run_startup_resume_event(
        self,
        adapter: BasePlatformAdapter,
        event: MessageEvent,
        session_key: str,
    ) -> None:
        """Dispatch one synthetic startup resume and wait for its agent turn.

        Inbound messages stay queued until the resumed turn finishes, else a user message can race it.
        """
        from gateway.run import _AGENT_PENDING_SENTINEL
        try:
            await adapter.handle_message(event)
            session_tasks = getattr(adapter, "_session_tasks", {})
            task = session_tasks.get(session_key) if isinstance(session_tasks, dict) else None
            if task is not None:
                await asyncio.shield(task)
        finally:
            # The runner slot was pre-claimed before this task spawned; release it if handle_message
            # raises before _handle_message takes ownership, else the real run's cleanup owns it.
            _pre_state = self._peek_session_state(session_key)
            if (_pre_state.turn.agent if _pre_state else None) is _AGENT_PENDING_SENTINEL:
                self._release_running_agent_state(session_key)

    def _queue_startup_restore_event(self, event: MessageEvent) -> None:
        queue = getattr(self, "_startup_restore_queue", None)
        if queue is None:
            queue = []
            self._startup_restore_queue = queue
        queue.append(event)
        try:
            source = event.source
            logger.info(
                "Queued inbound message during gateway startup restore: platform=%s chat=%s",
                source.platform.value if source and source.platform else "unknown",
                source.chat_id if source else "unknown",
            )
        except Exception:
            pass

    async def _drain_startup_restore_queue(self) -> int:
        """Replay inbound messages queued while startup auto-resume ran."""
        drained = 0
        queue = getattr(self, "_startup_restore_queue", None)
        if queue is None:
            return 0
        while queue:
            event = queue.pop(0)
            source = getattr(event, "source", None)
            adapter = self._adapter_for_source(source)
            if adapter is None:
                logger.debug(
                    "Dropping startup-restore queued message: adapter unavailable for %s",
                    getattr(getattr(source, "platform", None), "value", None),
                )
                continue
            # Mark this replay so _handle_message does not queue it again while
            # the restore gate remains closed for any fresh inbound arrivals.
            with suppress(Exception):
                setattr(event, "_hermes_startup_restore_replay", True)
            await adapter.handle_message(event)
            drained += 1
        return drained

    def _start_startup_warmup(self) -> None:
        """Kick off the boot turn-machinery warm-up in the background.

        Called from ``start()`` right after the startup-restore gate closes so the warm-up overlaps
        the network-bound platform connects; ``_finish_startup_restore`` awaits it (bounded).
        """
        from gateway.run import _startup_warmup_timeout_secs
        timeout = _startup_warmup_timeout_secs()
        if timeout <= 0:
            self._startup_warmup_task = None
            return
        self._startup_warmup_task = asyncio.ensure_future(
            self._warm_turn_prerequisites()
        )

    async def _warm_turn_prerequisites(self) -> None:
        """Initialize turn machinery on an executor thread before the gate opens.

        Never raises: a failed warm-up degrades to lazy init and must not block startup.
        """
        from gateway.run import _warm_turn_machinery_sync
        try:
            loop = asyncio.get_running_loop()
            t0 = time.monotonic()
            tool_count = await loop.run_in_executor(None, _warm_turn_machinery_sync)
            logger.info(
                "Turn machinery warmed in %.1fs (%d tool schema(s) materialized)",
                time.monotonic() - t0,
                tool_count,
            )
        except Exception:
            logger.warning(
                "Turn-machinery warm-up failed; first inbound turn will "
                "initialize lazily",
                exc_info=True,
            )

    async def _await_startup_warmup(self) -> None:
        """Bounded wait for the boot warm-up before the inbound gate opens.

        On timeout the gate opens anyway (availability outranks prompt completeness for a WEDGED
        init) and the warm-up continues in the background; a late failure is still logged.
        """
        from gateway.run import GatewayRunner, _startup_warmup_timeout_secs
        task = getattr(self, "_startup_warmup_task", None)
        if task is None or task.done():
            return
        timeout = _startup_warmup_timeout_secs()
        if timeout <= 0:
            return
        done, pending = await asyncio.wait({task}, timeout=timeout)
        if pending:
            logger.warning(
                "Turn-machinery warm-up still running after %.0fs; opening "
                "inbound gate anyway — the first turn may see lazily "
                "initialized machinery (#99373). Warm-up continues in the "
                "background.",
                timeout,
            )
            task.add_done_callback(
                lambda t: GatewayRunner._log_late_background_failure(
                    t,
                    "boot turn-machinery warm-up failed after gate release",
                    level=logging.DEBUG,
                )
            )

    async def _finish_startup_restore(self) -> None:
        """Wait (BOUNDED) for startup auto-resume, then release + drain inbound.

        Bounded by ``_startup_restore_drain_timeout_secs`` so one pathological boot-resume turn
        cannot hold the gate shut for every channel; on timeout the gate opens and resume turns
        finish in the background (NOT cancelled). Safe because ``_schedule_resume_pending_sessions``
        claims each ``_running_agents`` slot SYNCHRONOUSLY first, so drained inbound queues behind.
        """
        from gateway.run import _startup_restore_drain_timeout_secs
        tasks = list(getattr(self, "_startup_restore_tasks", []) or [])
        if tasks:
            timeout = _startup_restore_drain_timeout_secs()
            if timeout > 0:
                # asyncio.wait (unlike wait_for / gather+timeout) does NOT cancel pending tasks on
                # timeout — the slow resume turn keeps running in the background.
                done, pending = await asyncio.wait(tasks, timeout=timeout)
                if pending:
                    logger.warning(
                        "Startup-restore gate released after %.0fs with %d boot "
                        "auto-resume turn(s) still running; draining inbound "
                        "queue now (resume slots already claimed, so no "
                        "duplicate agents). Slow turn(s) continue in the "
                        "background.",
                        timeout,
                        len(pending),
                    )
                    # These tasks outlive the gate. Their normal done-callback only discards them
                    # from _background_tasks, so a LATER failure would be silently swallowed.
                    for task in pending:
                        task.add_done_callback(self._log_background_resume_result)
            else:
                # Non-positive timeout => opt out of the bound (historical
                # "wait forever" behaviour).
                await asyncio.gather(*tasks, return_exceptions=True)
                done = set(tasks)
            for task in done:
                if task.cancelled():
                    continue
                exc = task.exception()
                if exc is not None:
                    logger.debug(
                        "startup auto-resume task failed",
                        exc_info=(type(exc), exc, exc.__traceback__),
                    )
        self._startup_restore_tasks = []
        # Warm the turn machinery BEFORE the queue drains: replayed (and
        # fresh) inbound turns must not build skeleton prompts (#99373).
        await self._await_startup_warmup()
        drained = await self._drain_startup_restore_queue()
        self._startup_restore_in_progress = False
        if drained:
            logger.info("Drained %d inbound message(s) queued during startup restore", drained)

    @staticmethod
    def _log_background_resume_result(task: "asyncio.Task") -> None:
        """Done-callback for a boot-resume turn that outlived the startup-restore gate."""
        from gateway.run import GatewayRunner
        GatewayRunner._log_late_background_failure(
            task,
            "background startup auto-resume task failed after gate release",
            level=logging.DEBUG,
        )

    @staticmethod
    def _log_late_background_failure(
        task: "asyncio.Task", message: str, *, level: int = logging.WARNING
    ) -> None:
        """Shared done-callback body for boot-path tasks that outlive the startup-restore gate:
        surface a late failure otherwise swallowed once the task leaves ``_background_tasks``.
        Cancellation (shutdown) is expected, not an error."""
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.log(
                level,
                message,
                exc_info=(type(exc), exc, exc.__traceback__),
            )

    async def _await_startup_boot_sends(
        self,
        *,
        planned_restart_notification_pending: bool,
    ) -> None:
        """Run boot-path sends without letting them pin the inbound restore gate.

        Awaiting ``_send_restart_notification`` / ``_redeliver_pending_obligations`` inline before
        the gate releases lets one Telegram flood-control sleep freeze inbound on every platform.
        Same bounded ``asyncio.wait`` as the resume gate: on timeout return and let the sends finish
        in the background (not cancelled). The ledger claim + ``resume_pending`` clear run INLINE
        before the send task exists: bounded DB work, and deferring it let a hung notification
        expire the gate with zero rows claimed, so answered turns were replayed AND redelivered.
        """
        from gateway.run import _clear_planned_restart_notification, _startup_restore_drain_timeout_secs
        claimed = await self._claim_pending_obligations()

        async def _boot_sends() -> None:
            await self._send_restart_notification()
            if planned_restart_notification_pending:
                try:
                    await self._send_home_channel_startup_notifications(
                        skip_targets=None,
                    )
                finally:
                    _clear_planned_restart_notification()
            await self._redeliver_claimed_obligations(claimed)

        boot_task = asyncio.create_task(_boot_sends())
        timeout = _startup_restore_drain_timeout_secs()
        if timeout > 0:
            _done, pending = await asyncio.wait({boot_task}, timeout=timeout)
            if pending:
                logger.warning(
                    "Boot-path sends still running after %.0fs; releasing "
                    "inbound gate so other platforms are not frozen. "
                    "Restart notification / obligation redelivery continue "
                    "in the background.",
                    timeout,
                )
                boot_task.add_done_callback(self._log_background_boot_send_result)
                tasks = getattr(self, "_background_tasks", None)
                if tasks is None:
                    self._background_tasks = set()
                    tasks = self._background_tasks
                tasks.add(boot_task)
                boot_task.add_done_callback(tasks.discard)
        else:
            await boot_task

    @staticmethod
    def _log_background_boot_send_result(task: "asyncio.Task") -> None:
        """Done-callback for boot-path sends that outlived the restore gate."""
        from gateway.run import GatewayRunner
        GatewayRunner._log_late_background_failure(
            task, "background boot-path send failed after gate release: see traceback"
        )

    async def _clear_resume_pending_for_claimed_obligations(
        self, claimed: list, *, require_success: bool = False
    ) -> list:
        """Clear resume flags and return rows safe to redeliver.

        Startup recovery stays best-effort. Runtime reconnect recovery is stricter: if the
        session-store write fails the response must not be sent, or the turn could be resumed too.
        """
        sendable = []
        for row in claimed:
            session_key = row.get("session_key") or ""
            if not session_key:
                sendable.append(row)
                continue
            try:
                await self.async_session_store.clear_resume_pending(session_key)
            except Exception:
                logger.debug(
                    "clear_resume_pending failed for %s", session_key,
                    exc_info=True,
                )
                if not require_success:
                    sendable.append(row)
            else:
                sendable.append(row)
        return sendable

    async def _claim_pending_obligations(self) -> list:
        """Claim recoverable delivery-ledger rows and clear their ``resume_pending`` flags.

        Pure DB work, no sends. Must run INLINE at startup BEFORE ``_schedule_resume_pending_sessions``
        and before the abandonable boot-send task exists: these sessions already produced their
        answer, so clearing ``resume_pending`` here stops the resume path from re-running (and
        re-paying for) the turn however long the sends take. Rows that were mid-send or previously
        rejected carry a visible recovered-reply marker so a possible duplicate is labeled, never
        silent (gateway/delivery_ledger.py). Returns the claimed rows for redelivery.
        """
        try:
            from gateway.delivery_ledger import (
                ledger_enabled,
                sweep_recoverable,
            )

            if not await asyncio.to_thread(ledger_enabled):
                return []
            # Only claim rows whose exact transport owner is connected this boot. A multiplexed
            # gateway can host several bot identities for one platform; platform-only filtering
            # would spend a disconnected bot's retry budget merely because another bot is online.
            _profile_adapters = getattr(self, "_profile_adapters", None) or {}
            _deliverable_targets = {
                (getattr(p, "value", str(p)), "default") for p in self.adapters
            }
            # Legacy rows predate adapter_profile. They are unambiguous only in a non-multiplexed
            # gateway; fail closed when multiple bot identities share the process.
            if not _profile_adapters:
                _deliverable_targets.update(
                    (getattr(p, "value", str(p)), None) for p in self.adapters
                )
            for _profile, _adapters in _profile_adapters.items():
                _deliverable_targets.update(
                    (getattr(p, "value", str(p)), _profile) for p in _adapters
                )
            _deliverable = {platform for platform, _ in _deliverable_targets}
            claimed = await asyncio.to_thread(
                sweep_recoverable,
                None,
                deliverable_platforms=_deliverable,
                deliverable_targets=_deliverable_targets,
            )
        except Exception:
            logger.debug("delivery ledger sweep failed", exc_info=True)
            return []
        if not claimed:
            return []

        # Clear resume_pending for EVERY claimed row before any send: claiming already spent one
        # redelivery attempt and the answer is in the ledger, so the resume path must never re-run.
        await self._clear_resume_pending_for_claimed_obligations(claimed)
        return claimed

    async def _redeliver_claimed_obligations(self, claimed: list) -> int:
        """Redeliver final responses for rows claimed by :meth:`_claim_pending_obligations`.

        Network half of the split: runs inside the bounded boot-send task, so a flood-limited send
        can be abandoned by the restore gate without reopening the turn-replay window. Returns count.
        """
        if not claimed:
            return 0
        try:
            from gateway.delivery_ledger import (
                RECOVERED_MARKER,
                mark_delivered,
                mark_failed,
                release_runtime_claim,
            )
        except Exception:
            logger.debug("delivery ledger import failed", exc_info=True)
            return 0

        redelivered = 0
        for row in claimed:
            try:
                platform = Platform(row["platform"])
            except Exception:
                logger.debug(
                    "obligation %s: unknown platform %r",
                    row["obligation_id"], row.get("platform"),
                )
                continue
            if "profile" in row:
                adapter = self._authorization_adapter(
                    platform, row.get("profile")
                )
            else:
                # Startup rows preserve the historical default-adapter route.
                adapter = self.adapters.get(platform)
            if adapter is None:
                # Runtime claims have not reached a transport yet. If the
                # reconnect vanished before dispatch, release the claim without
                # spending an attempt so the next reconnect can retry it.
                if row.get("runtime_recovery"):
                    try:
                        await asyncio.to_thread(
                            release_runtime_claim,
                            row["obligation_id"],
                            "send_path_degraded",
                        )
                    except Exception:
                        logger.debug(
                            "failed to release undispatched runtime obligation %s",
                            row["obligation_id"],
                            exc_info=True,
                        )
                # Startup claims preserve their historical state; attempts cap
                # + stale cutoff bound later retries.
                continue
            content = row["content"]
            if row.get("needs_marker"):
                content = row.get("marker", RECOVERED_MARKER) + content
            metadata = (
                {"thread_id": row["thread_id"]} if row.get("thread_id") else None
            )

            try:
                result = await adapter.send(
                    chat_id=row["chat_id"],
                    content=content,
                    metadata=metadata,
                )
            except Exception as send_err:
                logger.warning(
                    "obligation %s: redelivery send raised: %s",
                    row["obligation_id"], send_err,
                )
                result = None
            try:
                if result is not None and getattr(result, "success", False):
                    await asyncio.to_thread(mark_delivered, row["obligation_id"])
                    redelivered += 1
                    logger.info(
                        "Redelivered recovered final response to %s:%s "
                        "(obligation %s, attempt %d)",
                        row["platform"], row["chat_id"],
                        row["obligation_id"], row["attempts"],
                    )
                else:
                    await asyncio.to_thread(
                        mark_failed,
                        row["obligation_id"],
                        str(getattr(result, "error", "") or "send failed"),
                    )
            except Exception:
                logger.debug("delivery ledger update failed", exc_info=True)
        return redelivered

    async def _redeliver_pending_obligations(self) -> int:
        """Claim + redeliver in one call (:meth:`_claim_pending_obligations` then
        :meth:`_redeliver_claimed_obligations`). Stable public shape for tests/external callers;
        the startup path calls the halves separately so the DB half runs inline before the
        abandonable send task.
        """
        return await self._redeliver_claimed_obligations(
            await self._claim_pending_obligations()
        )

    async def _redeliver_failed_obligations_for_platform(
        self,
        platform: Platform,
        *,
        profile: Optional[str] = None,
    ) -> int:
        """Replay one adapter identity's transient failures after reconnect.

        The startup sweep cannot claim live-owner rows, and an adapter can reconnect without the
        process exiting, so ``send_path_degraded`` responses would otherwise stay failed until the
        next restart. Claim/clear/send are best-effort and reuse the startup redelivery contract.
        """
        try:
            from gateway.delivery_ledger import (
                ledger_enabled,
                release_runtime_claim,
                sweep_failed_for_runtime,
            )

            if not await asyncio.to_thread(ledger_enabled):
                return 0
            claimed = await asyncio.to_thread(
                sweep_failed_for_runtime,
                platform.value,
                profile=profile,
            )
        except Exception:
            logger.debug(
                "runtime delivery ledger sweep failed after %s reconnect",
                platform.value,
                exc_info=True,
            )
            return 0
        if not claimed:
            return 0

        # Clear before any send so the reconnect path cannot both redeliver an
        # already-produced answer and schedule the same agent turn for resume.
        sendable = await self._clear_resume_pending_for_claimed_obligations(
            claimed, require_success=True
        )
        sendable_ids = {row["obligation_id"] for row in sendable}
        for row in claimed:
            if row["obligation_id"] in sendable_ids:
                continue
            try:
                await asyncio.to_thread(
                    release_runtime_claim,
                    row["obligation_id"],
                    "send_path_degraded",
                )
            except Exception:
                logger.debug(
                    "failed to release runtime delivery claim %s",
                    row["obligation_id"],
                    exc_info=True,
                )
        return await self._redeliver_claimed_obligations(sendable)

    def _schedule_resume_pending_sessions(self, platform=None) -> int:
        """Auto-continue fresh restart-interrupted sessions after startup.

        Synthesizes the next turn once adapters are back online; the event text is empty so the
        existing ``_is_resume_pending`` injection path owns the recovery wording. Sessions whose
        adapter is not in ``self.adapters`` stay ``resume_pending`` for the reconnect watcher, which
        re-calls this scoped to that ``platform`` (a reconnecting platform never touches another's
        recoveries); sessions with a running agent are skipped so none is resumed twice.
        """
        from gateway.run import _AGENT_PENDING_SENTINEL, _auto_continue_freshness_window
        window = _auto_continue_freshness_window()
        try:
            with self.session_store._lock:  # noqa: SLF001 — snapshot under lock
                self.session_store._ensure_loaded_locked()  # noqa: SLF001
                candidates = [
                    entry for entry in self.session_store._entries.values()  # noqa: SLF001
                    if entry.resume_pending
                    and not entry.suspended
                    and entry.origin is not None
                    and entry.resume_reason in self._AUTO_RESUME_REASONS
                    and (platform is None or entry.origin.platform == platform)
                ]
        except Exception as exc:
            logger.warning("Failed to enumerate resume-pending sessions: %s", exc)
            return 0

        # Defense-3: break the SIGTERM-respawn loop. Only count this boot when there are restart-
        # interrupted sessions to resume — a clean boot must not accrue toward the breaker. If too
        # many such boots hit the window, skip auto-resume for THIS boot only: the gateway still
        # serves inbound; the session stays resume_pending so a real user message can continue it.
        if candidates:
            try:
                from gateway import restart_loop_guard as _rlg

                _max_restarts, _window, _max_gap = self._restart_loop_guard_config()
                if _rlg.check_and_record(
                    _max_restarts, _window, max_gap_seconds=_max_gap
                ):
                    return 0
            except Exception as exc:  # noqa: BLE001 — breaker must fail OPEN
                logger.debug("Restart-loop guard check skipped: %s", exc)

        now = datetime.now()
        scheduled = 0
        for entry in candidates:
            marker = entry.last_resume_marked_at or entry.updated_at
            if marker is not None and (now - marker).total_seconds() > window:
                continue

            # Already being resumed (e.g. scheduled at startup and still
            # in-flight) — don't synthesize a second continuation turn.
            if self._is_session_running(entry.session_key):
                continue

            source = entry.origin
            adapter = self._adapter_for_source(source)
            if adapter is None:
                logger.debug(
                    "Skipping auto-resume for %s: adapter not ready for %s",
                    entry.session_key,
                    getattr(source.platform, "value", source.platform),
                )
                continue

            # Validate the session owner against the current allowlist before auto-resuming: a
            # session created before the allowlist existed (or whose owner was since removed) must
            # not silently receive a full agent response just because it carries a resume marker.
            try:
                if not self._is_user_authorized(source):
                    logger.warning(
                        "Skipping auto-resume for %s: session owner is no "
                        "longer authorized under the current allowlist",
                        entry.session_key,
                    )
                    continue
            except Exception as exc:
                logger.warning(
                    "Skipping auto-resume for %s: authorization check failed: %s",
                    entry.session_key, exc,
                )
                continue

            # Claim the session slot *before* spawning the task so an inbound message arriving
            # between task creation and the task's first await (where _process_message_background
            # sets the real sentinel) sees the slot occupied and queues, not a duplicate AIAgent.
            _resume_state = self._session_state(entry.session_key)
            _resume_state.turn.agent = _AGENT_PENDING_SENTINEL
            _resume_state.turn.started_ts = time.time()
            self._persist_active_agents()

            # Empty-text internal event: the _is_resume_pending branch in _handle_message_with_agent
            # prepends the reason-aware system note before the turn runs.
            event = MessageEvent(
                text="",
                message_type=MessageType.TEXT,
                source=source,
                internal=True,
            )
            task = asyncio.create_task(
                self._run_startup_resume_event(adapter, event, entry.session_key)
            )
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)
            if getattr(self, "_startup_restore_in_progress", False):
                tasks = getattr(self, "_startup_restore_tasks", None)
                if tasks is None:
                    tasks = []
                    self._startup_restore_tasks = tasks
                tasks.append(task)
            scheduled += 1
        if scheduled:
            logger.info(
                "Scheduled auto-resume for %d restart-interrupted session(s)",
                scheduled,
            )
        return scheduled

    def _startup_should_abort(self) -> bool:
        return (
            self._restart_requested
            or self._draining
            or self._shutdown_event.is_set()
        )

    async def _startup_teardown_adapter(self, adapter, platform) -> None:
        """Cancel an adapter's background tasks (best-effort) then disconnect it."""
        try:
            await adapter.cancel_background_tasks()
        except Exception as e:
            logger.debug("✗ %s background-task cancel error: %s", platform.value, e)
        await self._safe_adapter_disconnect(adapter, platform)

    def _startup_retry_entry(self, platform, adapter, platform_config, *, queued: bool = True) -> dict:
        """Build a ``_failed_platforms`` entry for a platform that failed at startup."""
        entry = {
            "config": platform_config,
            "attempts": 1,
            "next_retry": time.monotonic() + 30,
        }
        if queued:
            entry["queued_at"] = time.monotonic()
        entry["credential_claim"] = self._adapter_credential_claim(platform, adapter)
        entry["listener_claim"] = self._adapter_listener_claim(platform, adapter)
        return entry

    async def _abort_startup_if_shutdown_requested(
        self,
        adapter: Optional[BasePlatformAdapter] = None,
        platform: Optional[Platform] = None,
    ) -> bool:
        """Clean up and exit startup when restart/shutdown begins mid-startup."""
        if not self._startup_should_abort():
            return False
        if adapter is not None and platform is not None:
            await self._startup_teardown_adapter(adapter, platform)
        stop_task = self._stop_task
        current_task = asyncio.current_task()
        if stop_task is not None and stop_task is not current_task:
            await stop_task
        elif not self._shutdown_event.is_set():
            await self.stop(
                restart=self._restart_requested,
                detached_restart=self._restart_detached,
                service_restart=self._restart_via_service,
            )
        return True

    def _start_loop_liveness_guards(self, loop: asyncio.AbstractEventLoop) -> None:
        """Arm the selector floor and out-of-loop watchdog before adapters.

        Disabled entirely with ``gateway.loop_watchdog: false`` in config.yaml (config-only knob).
        """
        from gateway.run import _arm_loop_floor_timer, start_loop_liveness_watchdog
        config = getattr(self, "config", None)
        if config is not None and not getattr(config, "loop_watchdog", True):
            return
        if getattr(self, "_loop_floor_timer_handle", None) is None:
            try:
                self._loop_floor_timer_handle = _arm_loop_floor_timer(loop)
            except Exception:
                logger.debug("Failed to arm gateway loop floor timer", exc_info=True)

        watchdog = getattr(self, "_loop_liveness_watchdog", None)
        if watchdog is None or not watchdog.is_alive():
            try:
                # getattr defaults cover the config=None / bare-object test path; config-loaded
                # values are already validated+clamped by GatewayConfig.from_dict; no re-clamping.
                interval = getattr(
                    config,
                    "loop_watchdog_probe_interval_s",
                    DEFAULT_LOOP_WATCHDOG_INTERVAL_S,
                )
                timeout = getattr(
                    config,
                    "loop_watchdog_probe_timeout_s",
                    DEFAULT_LOOP_WATCHDOG_TIMEOUT_S,
                )
                strikes = getattr(
                    config,
                    "loop_watchdog_max_strikes",
                    DEFAULT_LOOP_WATCHDOG_MAX_STRIKES,
                )
                self._loop_liveness_watchdog = start_loop_liveness_watchdog(
                    loop,
                    probe_interval=float(interval),
                    probe_timeout=float(timeout),
                    max_strikes=int(strikes),
                )
            except Exception:
                logger.debug("Failed to start gateway loop liveness watchdog", exc_info=True)

    def _stop_loop_liveness_guards(self) -> None:
        """Disarm lifetime liveness guards before shutdown can load the loop.

        Also disarms the heartbeat writer: once shutdown starts loading the loop, a heartbeat that
        keeps refreshing the file makes a draining gateway look healthy to external probes.
        """
        for attr, method, what in (
            ("_loop_liveness_watchdog", "stop", "stop gateway loop liveness watchdog"),
            ("_loop_floor_timer_handle", "cancel", "cancel gateway loop floor timer"),
            ("_loop_heartbeat_task", "cancel", "cancel gateway loop heartbeat task"),
        ):
            guard = getattr(self, attr, None)
            setattr(self, attr, None)
            if guard is not None:
                try:
                    getattr(guard, method)()
                except Exception:
                    logger.debug("Failed to %s", what, exc_info=True)

    async def _consume_clean_shutdown_marker(self, marker_path) -> int:
        """Discard orphan turn markers before consuming a clean-exit receipt.

        If persistence or marker removal fails, startup must fail closed: continuing with the old
        receipt would let a later unclean exit masquerade as clean and discard interrupted turns.
        """
        discarded = await self.async_session_store.discard_active_turn_markers()
        marker_path.unlink()
        return discarded

    async def _recover_unclean_sessions(self) -> tuple[int, int]:
        """Recover exact active turns, then run the legacy recency fallback."""
        from gateway.run import _float_env
        exact = 0
        fallback = 0
        try:
            agent_timeout = max(1.0, _float_env("HERMES_AGENT_TIMEOUT", 1800))
            marker_max_age = max(60 * 60, int(agent_timeout * 2))
            exact = await self.async_session_store.recover_interrupted_turns(
                max_age_seconds=marker_max_age
            )
        except Exception as exc:
            logger.warning("Exact active-turn recovery on startup failed: %s", exc)
        try:
            fallback = await self.async_session_store.suspend_recently_active(
                max_age_seconds=120
            )
        except Exception as exc:
            logger.warning("Legacy session recovery on startup failed: %s", exc)
        return exact, fallback

    @staticmethod
    def _start_hosted_room_worker_sync():
        """Start the local Group Chat worker without importing the dashboard."""

        import tui_gateway.server  # noqa: F401
        from tui_gateway import methods_groups

        service = methods_groups.get_hosted_room_service()
        if service is None:
            service = methods_groups.start_hosted_room_service()
        if service is None:
            raise RuntimeError("Group Chat worker has no bound session backend")
        status = service.runtime.status()
        if not status.get("running") or status.get("stopping"):
            raise RuntimeError("Group Chat worker did not start")
        return service

    async def _ensure_hosted_room_worker(self):
        return await asyncio.to_thread(self._start_hosted_room_worker_sync)

    async def _hosted_room_worker_watcher(self, interval: float = 1.0) -> None:
        """Keep the room worker alive for the messaging gateway lifetime."""

        while self._running:
            await self._ensure_hosted_room_worker()
            await asyncio.sleep(interval)

    async def _stop_hosted_room_worker(self, timeout: float = 5.0) -> bool:
        """Pause room execution durably without interrupting accepted turns."""

        from tui_gateway import methods_groups

        return await asyncio.to_thread(
            methods_groups.stop_hosted_room_service,
            timeout=timeout,
        )

    def _start_loop_heartbeat_task(self) -> None:
        """Start the loop-liveness heartbeat task, idempotent.

        An asyncio task so a frozen loop stops refreshing ``state/gateway.heartbeat``; cancelled
        with the other background tasks in stop(). Best-effort — must never abort startup.
        """
        try:
            _existing_hb = getattr(self, "_loop_heartbeat_task", None)
            if _existing_hb is not None and not _existing_hb.done():
                return
            self._loop_heartbeat_task = asyncio.create_task(
                loop_heartbeat_forever(
                    interval_s=DEFAULT_HEARTBEAT_INTERVAL_S,
                    start_time=getattr(self, "_gateway_started_at", 0.0),
                )
            )
            # PERMANENT for the process lifetime, same as a _spawn_supervised watcher — tag it so
            # _scale_to_zero_has_live_background_work() doesn't treat an armed, otherwise-idle
            # gateway as busy forever.
            self._loop_heartbeat_task._hermes_supervised_watcher = True  # type: ignore[attr-defined]
            _bg = getattr(self, "_background_tasks", None)
            if _bg is not None:
                _bg.add(self._loop_heartbeat_task)
                self._loop_heartbeat_task.add_done_callback(_bg.discard)
        except Exception:
            logger.debug("Failed to start gateway loop heartbeat", exc_info=True)

    def _open_faulthandler_log(self):
        """Open (append) ``<log_dir>/gateway_faulthandler.log``, creating the directory."""
        from gateway.run import get_hermes_home
        log_dir = getattr(self.config, "log_dir", None) or os.path.join(
            str(get_hermes_home()), "logs",
        )
        os.makedirs(log_dir, exist_ok=True)
        return open(os.path.join(log_dir, "gateway_faulthandler.log"), "a", encoding="utf-8")

    def _start_install_faulthandler(self) -> None:
        """Enable faulthandler (stderr or a log file) plus the SIGUSR2 stack-dump hook."""
        # Falls back to a log file when sys.stderr is None (Windows VBS / pythonw / detached
        # service) — otherwise the gateway would die here and take every adapter offline.
        try:
            faulthandler.enable()
        except (RuntimeError, ValueError, OSError):
            try:
                faulthandler.enable(file=self._open_faulthandler_log(), all_threads=True)
            except Exception:
                logger.debug("faulthandler.enable() unavailable", exc_info=True)
        # Also dump stacks to a file on SIGUSR2 for off-line analysis under a service manager that
        # doesn't capture stderr. faulthandler.register()/SIGUSR2 are POSIX-only: skip on Windows
        # (faulthandler.enable() above still covers fatal errors).
        _sigusr2 = getattr(signal, "SIGUSR2", None)
        if _sigusr2 is not None and hasattr(faulthandler, "register"):
            try:
                faulthandler.register(
                    _sigusr2, file=self._open_faulthandler_log(), all_threads=True, chain=True,
                )
            except Exception:
                logger.debug("Could not set up faulthandler file logging", exc_info=True)

    def _start_log_startup_environment(self) -> None:
        """Bind the gateway loop, disarm the startup watchdog, and log the startup environment."""
        try:
            self._gateway_loop = asyncio.get_running_loop()
        except RuntimeError:
            self._gateway_loop = None
        if self._gateway_loop is not None:
            self._start_loop_liveness_guards(self._gateway_loop)
            # Loop confirmed live: the startup-liveness watchdog is done and the loop-liveness
            # watchdog (armed above) takes over. Disarm even when loop guards are config-disabled —
            # the startup watchdog covers only the pre-loop window. Deliberately inside this branch:
            # if the loop isn't live, startup has NOT reached the milestone and it must stay armed.
            try:
                from gateway.startup_watchdog import disarm_startup_watchdog

                disarm_startup_watchdog()
            except Exception:
                logger.debug("Startup watchdog disarm failed", exc_info=True)
        logger.info("Session storage: %s", self.config.sessions_dir)
        self._start_log_systemd_timing_alignment()
        # Log the resolved max_iterations budget so operators can verify the config.yaml → env
        # bridge at a glance (instead of silently running at a stale .env value for weeks).
        with suppress(Exception):
            logger.info(
                "Agent budget: max_iterations=%d (agent.max_turns from config.yaml, "
                "or HERMES_MAX_ITERATIONS from .env, or default 500)",
                int(os.getenv("HERMES_MAX_ITERATIONS", "500")),
            )
        # Redaction is ON by default; warn prominently when an operator has explicitly opted out so
        # the downgrade isn't forgotten. The redactor snapshots its state at import time, so this
        # log line is the source of truth for the process lifetime.
        with suppress(Exception):
            _redact_raw = os.getenv("HERMES_REDACT_SECRETS", "true")
            if _redact_raw.lower() in {"1", "true", "yes", "on"}:
                logger.info(
                    "Secret redaction: ENABLED (tool output, logs, and chat "
                    "responses are scrubbed before delivery)"
                )
            else:
                logger.warning(
                    "Secret redaction: DISABLED (HERMES_REDACT_SECRETS=%s). "
                    "API keys and tokens may appear verbatim in chat output, "
                    "session JSONs, and logs. Set security.redact_secrets: true "
                    "in config.yaml to re-enable.",
                    _redact_raw,
                )
        with suppress(Exception):
            from hermes_cli.profiles import get_active_profile_name
            _profile = get_active_profile_name()
            if _profile and _profile != "default":
                logger.info("Active profile: %s", _profile)
        with suppress(Exception):
            from gateway.status import write_runtime_status
            write_runtime_status(
                gateway_state="starting",
                exit_reason=None,
                clear_profile_platforms=True,
            )
        try:
            from hermes_cli.config import load_config
            from agent.monitoring.gateway_health_export import start_gateway_health_export
            self._gateway_health_export_runtime = start_gateway_health_export(load_config())
            if getattr(self._gateway_health_export_runtime, "enabled", False):
                logger.info("Gateway health OTLP export: enabled")
        except Exception:
            logger.debug("gateway health OTLP export startup failed", exc_info=True)

        # Log any active supply-chain security advisories. Deliberately does NOT block startup or
        # surface inline to users — only the operator can act (uninstall, rotate credentials).
        try:
            from hermes_cli.security_advisories import (
                detect_compromised,
                gateway_log_message,
            )
            _adv_msg = gateway_log_message(detect_compromised())
            if _adv_msg:
                logger.warning("%s", _adv_msg)
                logger.warning(
                    "Run `hermes doctor` on the gateway host for full "
                    "remediation steps."
                )
        except Exception:
            logger.debug(
                "security advisory check failed at gateway startup",
                exc_info=True,
            )

    def _start_log_systemd_timing_alignment(self) -> None:
        """Warn when systemd's TimeoutStopSec does not cover the drain window. Never raises.

        A unit file from before an upgrade (no ``hermes setup`` re-run) may encode the old default,
        so SIGKILL hits mid-drain and looks like a phantom kill in the journal.
        """
        try:
            from gateway.shutdown_forensics import check_systemd_timing_alignment
            _alignment = check_systemd_timing_alignment(
                self._restart_drain_timeout,
                getattr(self, "_cron_drain_timeout", DEFAULT_GATEWAY_CRON_DRAIN_TIMEOUT),
            )
            if _alignment is not None and _alignment.get("mismatch"):
                logger.warning(
                    "Stale systemd unit detected: %s has TimeoutStopSec=%.0fs but "
                    "drain_timeout=%.0fs cron_drain_timeout=%.0fs (expected >=%.0fs). "
                    "systemd may SIGKILL the gateway mid-drain. Run "
                    "`hermes gateway install --force` to regenerate the unit, or "
                    "shorten agent.restart_drain_timeout / agent.cron_drain_timeout.",
                    _alignment.get("unit", "(unknown)"),
                    _alignment["timeout_stop_sec"],
                    _alignment["drain_timeout"],
                    _alignment.get(
                        "cron_drain_timeout", DEFAULT_GATEWAY_CRON_DRAIN_TIMEOUT
                    ),
                    _alignment["expected_min"],
                )
        except Exception as _e:
            logger.debug("check_systemd_timing_alignment failed: %s", _e)

    # Env vars that count as "an allowlist is configured" / "open access opted in" for builtin
    # platforms; plugin-registered platforms are appended at check time.
    _BUILTIN_ALLOWED_USERS_VARS = (
        "TELEGRAM_ALLOWED_USERS", "DISCORD_ALLOWED_USERS",
        "WHATSAPP_ALLOWED_USERS", "WHATSAPP_CLOUD_ALLOWED_USERS",
        "SLACK_ALLOWED_USERS",
        "SIGNAL_ALLOWED_USERS", "SIGNAL_GROUP_ALLOWED_USERS",
        "TELEGRAM_GROUP_ALLOWED_USERS",
        "TELEGRAM_GROUP_ALLOWED_CHATS",
        "EMAIL_ALLOWED_USERS",
        "SMS_ALLOWED_USERS", "MATTERMOST_ALLOWED_USERS",
        "MATRIX_ALLOWED_USERS", "DINGTALK_ALLOWED_USERS",
        "FEISHU_ALLOWED_USERS",
        "WECOM_ALLOWED_USERS",
        "WECOM_CALLBACK_ALLOWED_USERS",
        "WEIXIN_ALLOWED_USERS",
        "BLUEBUBBLES_ALLOWED_USERS",
        "QQ_ALLOWED_USERS",
        "YUANBAO_ALLOWED_USERS",
        "GATEWAY_ALLOWED_USERS",
    )
    _BUILTIN_ALLOW_ALL_VARS = (
        "TELEGRAM_ALLOW_ALL_USERS", "DISCORD_ALLOW_ALL_USERS",
        "WHATSAPP_ALLOW_ALL_USERS", "WHATSAPP_CLOUD_ALLOW_ALL_USERS",
        "SLACK_ALLOW_ALL_USERS",
        "SIGNAL_ALLOW_ALL_USERS", "EMAIL_ALLOW_ALL_USERS",
        "SMS_ALLOW_ALL_USERS", "MATTERMOST_ALLOW_ALL_USERS",
        "MATRIX_ALLOW_ALL_USERS", "DINGTALK_ALLOW_ALL_USERS",
        "FEISHU_ALLOW_ALL_USERS",
        "WECOM_ALLOW_ALL_USERS",
        "WECOM_CALLBACK_ALLOW_ALL_USERS",
        "WEIXIN_ALLOW_ALL_USERS",
        "BLUEBUBBLES_ALLOW_ALL_USERS",
        "QQ_ALLOW_ALL_USERS",
        "YUANBAO_ALLOW_ALL_USERS",
    )

    def _start_check_access_policy(self) -> bool:
        """Warn about missing allowlists; return True when startup must be refused."""
        from gateway.run import (
            _OWN_POLICY_OPEN_ENV,
            _own_policy_open_startup_violation,
            _write_runtime_status_quiet,
        )
        # Plugin-registered platforms declare their own allowed_users_env / allow_all_env, so the
        # warning stays accurate as plugins (IRC) arrive.
        _plugin_allowed_vars: tuple = ()
        _plugin_allow_all_vars: tuple = ()
        try:
            from gateway.platform_registry import platform_registry
            _plugin_allowed_vars = tuple(
                e.allowed_users_env for e in platform_registry.plugin_entries()
                if e.allowed_users_env
            )
            _plugin_allow_all_vars = tuple(
                e.allow_all_env for e in platform_registry.plugin_entries()
                if e.allow_all_env
            )
        except Exception:
            pass
        _any_allowlist = any(
            os.getenv(v) for v in self._BUILTIN_ALLOWED_USERS_VARS + _plugin_allowed_vars
        )
        _allow_all = os.getenv("GATEWAY_ALLOW_ALL_USERS", "").lower() in {"true", "1", "yes"} or any(
            os.getenv(v, "").lower() in {"true", "1", "yes"}
            for v in self._BUILTIN_ALLOW_ALL_VARS + _plugin_allow_all_vars
        )
        if not _any_allowlist and not _allow_all:
            logger.warning(
                "No env user allowlists configured. Messaging platforms default to "
                "pairing/allowlist policies and will deny unknown senders unless you "
                "configure platform allowlists (e.g., TELEGRAM_ALLOWED_USERS=your_id) "
                "or explicitly opt in with GATEWAY_ALLOW_ALL_USERS=true plus "
                "dm_policy/group_policy: open on the platform."
            )

        reason = _own_policy_open_startup_violation(self.config)
        if reason:
            platform_value = reason.split(":", 1)[0]
            allow_all_env = None
            for platform, open_env in _OWN_POLICY_OPEN_ENV.items():
                if platform.value == platform_value:
                    allow_all_env = open_env[2]
                    break
            logger.error(
                "Refusing to start: %s has dm_policy/group_policy set to 'open' "
                "but neither GATEWAY_ALLOW_ALL_USERS nor %s is enabled.",
                platform_value,
                allow_all_env or "a platform allow-all flag",
            )
            _write_runtime_status_quiet(gateway_state="startup_failed", exit_reason=reason)
            self._request_clean_exit(reason)
            return True
        return False

    async def _start_recover_previous_run(self) -> None:
        """Plugins, relay, hooks, then crash/clean-exit recovery of processes and sessions."""
        from gateway.run import _hermes_home
        # Discover Python plugins before shell hooks so plugin block decisions take precedence in
        # tie cases. Explicit here because the gateway lazily imports run_agent per request, so
        # the discover_plugins() side-effect in model_tools.py is NOT guaranteed to have run yet.
        try:
            from hermes_cli.plugins import discover_plugins
            discover_plugins()
        except Exception:
            logger.warning(
                "plugin discovery failed at gateway startup", exc_info=True,
            )

        # Register the generic relay adapter only if GATEWAY_RELAY_URL / gateway.relay_url is set.
        # No URL -> no-op, so direct/single-tenant deployments are unaffected.
        try:
            from gateway.relay import (
                register_relay_adapter,
                relay_url,
                self_provision_relay,
                send_relay_policy,
            )

            # Boot-time relay self-provision: resolve the agent's NAS token -> POST /relay/provision
            # -> set GATEWAY_RELAY_* in os.environ BEFORE registration reads them. Never raises.
            self_provision_relay()

            if register_relay_adapter():
                logger.info("relay adapter registered (connector at %s)", relay_url())
                # Declare this gateway's relevance policy (mention-gating / free-response / allow-
                # bots) to the connector so the SAME behavior governs relay delivery (Phase 6 Unit
                # ζ). Runs after the secret is resolved; never raises, never blocks boot.
                send_relay_policy()
        except Exception:
            logger.warning(
                "relay adapter registration failed at gateway startup", exc_info=True,
            )

        # Register declarative shell hooks from cli-config.yaml. Gateway has no TTY, so consent must
        # come from --accept-hooks, HERMES_ACCEPT_HOOKS, or hooks_auto_accept: true; pass
        # accept_hooks=False and let register_from_config resolve env + config. Never blocks startup.
        try:
            from hermes_cli.config import load_config
            from agent.shell_hooks import register_from_config
            _hooks_cfg = load_config()
            register_from_config(_hooks_cfg, accept_hooks=False)

            from agent.outbound_webhooks import (
                register_from_config as register_outbound_webhooks,
            )
            register_outbound_webhooks(_hooks_cfg)
        except Exception:
            logger.debug(
                "shell-hook registration failed at gateway startup",
                exc_info=True,
            )

        # Discover and load event hooks
        self.hooks.discover_and_load()

        # Recover background processes from checkpoint (crash recovery)
        try:
            from tools.process_registry import process_registry
            recovered = process_registry.recover_from_checkpoint()
            if recovered:
                logger.info("Recovered %s background process(es) from previous run", recovered)
        except Exception as e:
            logger.warning("Process checkpoint recovery: %s", e)

        # Recover sessions active when the gateway last exited. Exact durable turn markers cover
        # long-running work; the 120s recency heuristic remains as a fallback for turns from older
        # versions without markers. SKIP after a clean shutdown — the previous process already drained.
        _clean_marker = _hermes_home / ".clean_shutdown"
        if _clean_marker.exists():
            logger.info("Previous gateway exited cleanly — skipping session suspension")
            try:
                discarded = await self._consume_clean_shutdown_marker(_clean_marker)
            except Exception as exc:
                logger.error(
                    "Clean-start marker cleanup failed; refusing startup so the "
                    "clean-exit receipt cannot mask a later unclean exit: %s",
                    exc,
                )
                raise RuntimeError("clean-start recovery cleanup failed") from exc
            if discarded:
                logger.info(
                    "Discarded %d orphan active-turn marker(s) after clean shutdown",
                    discarded,
                )
        else:
            exact, fallback = await self._recover_unclean_sessions()
            recovered = exact + fallback
            if recovered:
                logger.info(
                    "Marked %d in-flight session(s) as resumable from previous run "
                    "(%d exact, %d legacy)",
                    recovered,
                    exact,
                    fallback,
                )

        # Stuck-loop detection: a session active across 3+ consecutive restarts is probably looping
        # (its history keeps hanging the agent); auto-suspend so the next message starts clean.
        try:
            stuck = self._suspend_stuck_loop_sessions()
            if stuck:
                logger.warning("Auto-suspended %d stuck-loop session(s)", stuck)
        except Exception as e:
            logger.debug("Stuck-loop detection failed: %s", e)

    async def _start_prefilter_platforms(self) -> Tuple[bool, int, list, list]:
        """Create + wire an adapter per enabled platform (serial pre-filter, no connects).

        Returns (aborted, enabled_platform_count, multiplex_skipped_platforms, pending_connects).
        """
        from gateway.run import _platform_has_bot_credential
        enabled_platform_count = 0
        _multiplex_on = bool(getattr(self.config, "multiplex_profiles", False))
        _multiplex_skipped_platforms: list[Platform] = []
        # Initialize and connect each configured platform. connect() calls run concurrently so one
        # slow/failing platform (e.g. Telegram behind a dead proxy) cannot delay the others by a
        # full timeout window; the cheap serial pre-filter and per-platform timeouts are unchanged.
        _pending_connects = []  # (platform, platform_config, adapter)
        for platform, platform_config in self.config.platforms.items():
            if await self._abort_startup_if_shutdown_requested():
                return True, enabled_platform_count, _multiplex_skipped_platforms, _pending_connects
            if not platform_config.enabled:
                continue
            # Under multiplexing, a platform may be enabled on the default profile's config.yaml
            # while its bot token lives only in a secondary profile's .env. Starting the primary with
            # an empty token fails at once and queues a reconnect loop that can never heal; the
            # secondary starts its own adapter with the real token, so skip the empty primary.
            if _multiplex_on and not _platform_has_bot_credential(platform, platform_config):
                logger.info(
                    "Skipping %s on default profile: no bot credential in this "
                    "profile's secrets. Secondary multiplexed profiles that "
                    "provide the token will still connect.",
                    platform.value,
                )
                _multiplex_skipped_platforms.append(platform)
                continue
            enabled_platform_count += 1

            adapter = self._create_adapter(platform, platform_config)
            if not adapter:
                # Distinguish between missing builtin deps and missing plugin
                _pval = platform.value
                _builtin_names = {m.value for m in Platform.__members__.values()}
                if _pval not in _builtin_names:
                    logger.warning(
                        "No adapter for '%s' -- is the plugin installed? "
                        "(platform is enabled in config.yaml but no plugin registered it)",
                        _pval,
                    )
                else:
                    logger.warning("No adapter available for %s", _pval)
                continue

            # Set up message + fatal error handlers. Under multiplexing the default profile needs
            # the same whole-handler runtime scope as a secondary profile: authorization and prompt
            # rendering both run before the narrower agent-turn scope is installed.
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
            _pending_connects.append((platform, platform_config, adapter))
        return False, enabled_platform_count, _multiplex_skipped_platforms, _pending_connects

    async def _start_connect_pending(self, _pending_connects: list) -> Optional[list]:
        """Connect the pre-filtered adapters concurrently.

        Returns the raw per-platform results, or None when a restart/shutdown aborted startup
        mid-connect (adapters already torn down).
        """
        async def _connect_one_startup(p, p_cfg, adp):
            """Connect a single platform; never let one block the others (#83791)."""
            if await self._abort_startup_if_shutdown_requested(adp, p):
                return (p, adp, p_cfg, "aborted", None)
            logger.info("Connecting to %s...", p.value)
            self._update_platform_runtime_status(
                p.value, platform_state="connecting", error_code=None, error_message=None,
            )
            try:
                ok = await self._connect_initial_adapter_with_timeout(adp, p)
            except Exception as _exc:  # noqa: BLE001 - surfaced below as a retryable error
                return (p, adp, p_cfg, "exception", _exc)
            return (p, adp, p_cfg, "ok" if ok else "failed", None)

        if _pending_connects:
            # Abort-aware concurrent wait (parity with the serial loop's between-platforms check): a
            # restart/shutdown requested mid-connect must cancel still-pending connects, clean up the
            # ones already completed, and abort startup.
            _task_map: dict = {}
            for (p, c, a) in _pending_connects:
                _t = asyncio.ensure_future(_connect_one_startup(p, c, a))
                _task_map[_t] = (p, c, a)
            _pending_tasks = set(_task_map)
            _abort_mid_connect = False
            while _pending_tasks:
                _done, _pending_tasks = await asyncio.wait(
                    _pending_tasks, timeout=0.05
                )
                if _pending_tasks and self._startup_should_abort():
                    _abort_mid_connect = True
                    break
            if _abort_mid_connect:
                # Cancel and fully settle the in-flight connects FIRST, so a completed adapter's
                # disconnect cannot unblock a sibling's connect() before the sibling is cancelled.
                for _t in _pending_tasks:
                    _t.cancel()
                await asyncio.gather(*_pending_tasks, return_exceptions=True)
                for _t in _pending_tasks:
                    _p, _c, _a = _task_map[_t]
                    await self._startup_teardown_adapter(_a, _p)
                # Tear down adapters whose connect already succeeded — they
                # were never registered, so stop() won't reach them.
                for _t, (_p, _c, _a) in _task_map.items():
                    if _t in _pending_tasks or _t.cancelled():
                        continue
                    _res = _t.exception() is None and _t.result() or None
                    if _res and _res[3] == "ok":
                        await self._startup_teardown_adapter(_a, _p)
                await self._abort_startup_if_shutdown_requested()
                return None
            _raw = [
                _t.exception() or _t.result() for _t in _task_map
            ]
        else:
            _raw = []
        return _raw

    async def _start_aggregate_connect_results(
        self,
        _raw: list,
        startup_retryable_errors: list,
        startup_nonretryable_errors: list,
    ) -> int:
        """Apply connect outcomes to shared state; returns the connected adapter count.

        Aggregated single-threaded so shared state (self.adapters, self._failed_platforms, the
        error lists) is mutated exactly as the original serial loop did.
        """
        connected_count = 0
        for _item in _raw:
            if isinstance(_item, Exception):
                # Unexpected escape from _connect_one_startup (shouldn't happen); log and skip.
                logger.error("Unexpected startup connect error: %s", _item)
                continue
            platform, adapter, platform_config, outcome, exc = _item
            if outcome == "aborted":
                continue
            if outcome == "exception":
                logger.error("\u2717 %s error: %s", platform.value, exc)
                # Same defensive cleanup path for exceptions -- an adapter that raised mid-connect
                # may still have a live aiohttp.ClientSession or child subprocess.
                await self._safe_adapter_disconnect(adapter, platform)
                self._update_platform_runtime_status(
                    platform.value, platform_state="retrying", error_code=None, error_message=str(exc),
                )
                startup_retryable_errors.append(f"{platform.value}: {exc}")
                # Unexpected exceptions are typically transient -- queue for retry
                self._failed_platforms[platform] = self._startup_retry_entry(platform, adapter, platform_config)
                continue
            if outcome == "ok":
                self.adapters[platform] = adapter
                self._sync_voice_mode_state_to_adapter(adapter)
                # Wire voice input at connect time so transcription works without /voice join.
                self._bind_voice_input_callback(adapter)
                connected_count += 1
                self._update_platform_runtime_status(
                    platform.value, platform_state="connected", error_code=None, error_message=None,
                )
                logger.info("\u2713 %s connected", platform.value)
                continue
            # outcome == "failed"
            logger.warning("\u2717 %s failed to connect", platform.value)
            # Defensive cleanup: a failed connect() may have allocated resources
            # (aiohttp.ClientSession, poll tasks, bridge subprocesses) before giving up.
            await self._safe_adapter_disconnect(adapter, platform)
            if not adapter.has_fatal_error:
                self._update_platform_runtime_status(
                    platform.value, platform_state="retrying", error_code=None, error_message="failed to connect",
                )
                startup_retryable_errors.append(f"{platform.value}: failed to connect")
                # No fatal error info means likely a transient issue -- queue for retry
                self._failed_platforms[platform] = self._startup_retry_entry(platform, adapter, platform_config)
                continue
            # A live foreign holder of this bot token is a single-writer ownership conflict, not a
            # blip — ``_acquire_platform_lock`` emits it retryable only so a MID-RUN reconnect can
            # recover. At startup route it non-retryable: with nothing connected the gateway exits
            # 78 instead of sitting alive and deaf in the retry queue.
            _retryable = adapter.fatal_error_retryable and not (
                is_global_startup_conflict(adapter.fatal_error_code)
            )
            self._update_platform_runtime_status(
                platform.value,
                platform_state="retrying" if _retryable else "fatal",
                error_code=adapter.fatal_error_code,
                error_message=adapter.fatal_error_message,
            )
            target = startup_retryable_errors if _retryable else startup_nonretryable_errors
            target.append(f"{platform.value}: {adapter.fatal_error_message}")
            if _retryable:
                self._failed_platforms[platform] = self._startup_retry_entry(
                    platform, adapter, platform_config, queued=False
                )
        return connected_count

    async def _start_secondary_profiles(
        self, connected_count: int, _multiplex_skipped_platforms: list
    ) -> Tuple[bool, int]:
        """Bring up multiplexed secondary-profile adapters. Returns (aborted, connected_count)."""
        from gateway.run import MultiplexConfigError, _write_runtime_status_quiet
        # Multi-profile multiplexing: bring up adapters for every OTHER profile this gateway serves.
        # Each profile's adapters connect under that profile's home + credential scope and stamp
        # their inbound events with the profile so the agent turn resolves correctly.
        try:
            _secondary_connected = await self._start_secondary_profile_adapters()
            connected_count += _secondary_connected
        except MultiplexConfigError as e:
            # Invalid multiplexer config — abort startup cleanly so the operator
            # fixes config.yaml rather than running a half-wired gateway.
            reason = str(e)
            logger.error("Gateway multiplexer config error: %s", reason)
            _write_runtime_status_quiet(gateway_state="startup_failed", exit_reason=reason)
            self._exit_code = GATEWAY_FATAL_CONFIG_EXIT_CODE
            self._request_clean_exit(reason)
            self._startup_restore_in_progress = False
            return True, connected_count
        except Exception as e:
            logger.error("Secondary-profile adapter startup failed: %s", e, exc_info=True)
        finally:
            # Startup authority is one phase, not a persistent runner mode.
            # From this point onward every adapter retry is non-evicting.
            self._platform_lock_takeover_on_start = False

        # A platform skipped on the primary for a missing credential should have been picked up by
        # a secondary profile owning the token. If none did, it is enabled in config.yaml yet
        # silently unserved — surface it loudly instead of leaving a quiet dead channel.
        for _skipped in _multiplex_skipped_platforms:
            _served_by_secondary = any(
                _skipped in _profile_map
                for _profile_map in self._profile_adapters.values()
            )
            if not _served_by_secondary:
                logger.warning(
                    "%s is enabled but no profile (default or secondary) "
                    "provided a bot credential for it — the platform is not "
                    "being served. Add its token to the profile that should "
                    "own it, or disable the platform.",
                    _skipped.value,
                )
        return False, connected_count

    def _start_handle_no_connections(
        self,
        connected_count: int,
        enabled_platform_count: int,
        startup_retryable_errors: list,
        startup_nonretryable_errors: list,
    ) -> bool:
        """Log/degrade when nothing connected; return True when startup must exit."""
        from gateway.run import _write_runtime_status_quiet
        if connected_count == 0:
            if startup_nonretryable_errors and not startup_retryable_errors:
                reason = "; ".join(startup_nonretryable_errors)
                logger.error("Gateway hit a non-retryable startup conflict: %s", reason)
                _write_runtime_status_quiet(gateway_state="startup_failed", exit_reason=reason)
                self._exit_code = GATEWAY_FATAL_CONFIG_EXIT_CODE
                self._request_clean_exit(reason)
                self._startup_restore_in_progress = False
                return True
            if startup_nonretryable_errors:
                # Mixed failure mode: some platforms fatally misconfigured (e.g. WhatsApp never
                # paired), others merely transient (e.g. Telegram TimedOut). Exiting 78 here would
                # let exit-78 supervisors take the gateway PERMANENTLY down over a network blip and
                # deny the retryable ones their retry. Log the fatal side loudly, then fall through to
                # the degraded/retry path: the watcher recovers the retryable; the rest stay parked.
                logger.error(
                    "%d platform(s) fatally misconfigured and parked: %s. "
                    "Staying alive so retryable platforms can recover.",
                    len(startup_nonretryable_errors),
                    "; ".join(startup_nonretryable_errors),
                )
            if enabled_platform_count > 0:
                if startup_retryable_errors:
                    # All enabled platforms hit retryable failures (network blip, bridge not paired,
                    # npm install timeout...). Keep the gateway alive so cron jobs still run and the
                    # reconnect watcher can recover the platforms once the cause is fixed; exiting
                    # here would turn one misconfigured platform into an infinite systemd restart loop.
                    reason = "; ".join(startup_retryable_errors)
                    logger.warning(
                        "Gateway started with no connected platforms — "
                        "%d platform(s) queued for retry: %s",
                        len(self._failed_platforms), reason,
                    )
                    try:
                        from gateway.status import write_runtime_status
                        write_runtime_status(
                            gateway_state="degraded",
                            exit_reason=None,
                        )
                    except Exception:
                        pass
                    # Fall through to the normal "running" state — reconnect watcher takes it from here.
                # All enabled platforms had no adapter (missing library or credentials). Fleet nodes
                # share one config.yaml but hold credentials for only a subset of platforms, so
                # degrade gracefully and let cron jobs run.
                logger.warning(
                    "No adapter could be created for any of the %d configured platform(s). "
                    "Check that required dependencies are installed and credentials are set. "
                    "Gateway will continue for cron job execution.",
                    enabled_platform_count,
                )
            else:
                logger.warning("No messaging platforms enabled.")
                logger.info("Gateway will continue running for cron job execution.")
        return False

    async def _start_finish_wiring(self, connected_count: int) -> None:
        """Post-connect wiring: room worker, heartbeat, hooks, notifications, restore, watchers."""
        from gateway.run import (
            _hermes_home,
            _planned_restart_notification_pending,
            _restart_notification_pending,
        )
        try:
            await self._ensure_hosted_room_worker()
        except Exception:
            logger.error(
                "Group Chat worker failed to start; mutating Group Chat commands "
                "will fail closed until supervision recovers it",
                exc_info=True,
            )
        self._spawn_supervised(
            self._hosted_room_worker_watcher,
            "hosted_room_worker",
        )

        self._start_loop_heartbeat_task()

        # Emit gateway:startup hook
        hook_count = len(self.hooks.loaded_hooks)
        if hook_count:
            logger.info("%s hook(s) loaded", hook_count)
        await self.hooks.emit("gateway:startup", {
            "platforms": [p.value for p in self.adapters],
        })

        if connected_count > 0:
            logger.info("Gateway running with %s platform(s)", connected_count)

        # Build initial channel directory for send_message name resolution
        try:
            from gateway.channel_directory import build_channel_directory
            directory = await build_channel_directory(self.adapters)
            ch_count = sum(len(chs) for chs in directory.get("platforms", {}).values())
            logger.info("Channel directory built: %d target(s)", ch_count)
        except Exception as e:
            logger.warning("Channel directory build failed: %s", e)

        # Check if we're restarting after a /update command. If the update is
        # still running, keep watching so we notify once it actually finishes.
        notified = await self._send_update_notification()
        if not notified and any(
            path.exists()
            for path in (
                _hermes_home / ".update_pending.json",
                _hermes_home / ".update_pending.claimed.json",
            )
        ):
            self._schedule_update_notification_watch()

        # Give freshly connected adapters a brief moment to settle before sending restart/startup
        # lifecycle messages; in practice this helps Discord thread deliveries after reconnect.
        if connected_count > 0:
            await asyncio.sleep(1.0)

        # Notify the chat that initiated /restart that the gateway is back.
        chat_restart_notification_pending = _restart_notification_pending()
        planned_restart_notification_pending = _planned_restart_notification_pending()
        # Capture, before _send_restart_notification() unlinks the marker, whether this process
        # booted from a chat-originated /restart. One-shot signal for the /restart redelivery
        # guard (_is_stale_restart_redelivery): a missing dedup marker only suppresses a /restart
        # when we KNOW we just came out of a restart cycle.
        if chat_restart_notification_pending:
            self._booted_from_restart = True
        # Restart notification, home-channel startup notice, and obligation redelivery all call
        # adapter.send(). Those sends must not pin the inbound restore gate — a Telegram flood-
        # control sleep on this path froze every platform for the full penalty.
        await self._await_startup_boot_sends(
            planned_restart_notification_pending=planned_restart_notification_pending,
        )

        # Auto-continue fresh sessions interrupted by the previous restart/shutdown. resume_pending
        # is cleared by the normal successful-turn path, so a failed auto-resume stays visible on the
        # next user message. _await_startup_boot_sends already cleared sessions answered in the ledger.
        self._schedule_resume_pending_sessions()
        await self._finish_startup_restore()

        # Surface state.db init failures to the user's messaging platforms
        # so they know persistence is broken before losing data (#88235).
        await self._send_session_db_warning_notifications()

        # Drain any recovered process watchers (from crash recovery checkpoint)
        try:
            from tools.process_registry import process_registry
            # Detach the current batch atomically: reassigning to a fresh list takes ownership of
            # exactly the watchers present now, so any watcher appended concurrently during the
            # yield below isn't silently dropped by a clear() on the shared list.
            watchers = process_registry.pending_watchers
            process_registry.pending_watchers = []
            # Process in batches of 100 with event-loop yield points to avoid
            # O(n^2) event-loop blocking when recovering thousands of watchers.
            for i, watcher in enumerate(watchers):
                self._spawn_supervised(
                    lambda w=watcher: self._run_process_watcher(w),
                    f"process_watcher:{watcher.get('session_id')}",
                    restart=False,
                )
                logger.info("Resumed watcher for recovered process %s", watcher.get("session_id"))
                if i % 100 == 99:
                    await asyncio.sleep(0)
        except Exception as e:
            logger.error("Recovered watcher setup error: %s", e)

    # Long-lived supervised watchers spawned at the end of start(), in order. (method, name)
    # - session_expiry_watcher: finalize expired sessions.
    # - model_catalog_refresh_watcher: keep /model picker remote catalogs warm on disk so a
    #   delisted/new model reaches the picker within one TTL (model_catalog.ttl_minutes).
    # - session_stall_watcher: pending inbound + stale agent activity → warn user to /new
    #   (does not kill the turn; agent.session_stall_timeout).
    # - kanban_notifier_watcher: deliver events for subscriptions owned by the profiles whose
    #   adapters this gateway hosts, even when another gateway owns the dispatcher.
    # - kanban_dispatcher_watcher: spawn workers for ready tasks; gated by
    #   kanban.dispatch_in_gateway (default True), no-op when false.
    _PRE_RECONNECT_WATCHERS = (
        ("_session_expiry_watcher", "session_expiry_watcher"),
        ("_model_catalog_refresh_watcher", "model_catalog_refresh_watcher"),
        ("_session_stall_watcher", "session_stall_watcher"),
        ("_kanban_notifier_watcher", "kanban_notifier_watcher"),
        ("_kanban_dispatcher_watcher", "kanban_dispatcher_watcher"),
    )
    # - handoff_watcher: re-bind CLI sessions marked handoff_state='pending' to the destination
    #   platform's home channel and forge a synthetic user turn.
    # - async_delegation_watcher: inject delegate_task(background=true) completions into their
    #   originating session as a new turn (covers the idle, no-turn case).
    # - loop_wakeup_watcher: inject due /loop wakeup prompts into idle originating chats.
    _POST_RECONNECT_WATCHERS = (
        ("_handoff_watcher", "handoff_watcher"),
        ("_async_delegation_watcher", "async_delegation_watcher"),
        ("_loop_wakeup_watcher", "loop_wakeup_watcher"),
    )

    def _start_spawn_background_watchers(self) -> None:
        """Spawn the long-lived supervised background watchers."""
        for method, name in self._PRE_RECONNECT_WATCHERS:
            self._spawn_supervised(getattr(self, method), name)

        if self._failed_platforms:
            logger.info(
                "Starting reconnection watcher for %d failed platform(s): %s",
                len(self._failed_platforms),
                ", ".join(p.value for p in self._failed_platforms),
            )
        # Spawned via _spawn_supervised so an exception escaping the watcher's OUTER loop is caught,
        # logged, and restarted with backoff instead of silently killing it (else a platform already
        # queued in _failed_platforms stays stranded: the ensure hook only runs on a NEW fatal-error
        # arrival). ``on_spawn`` keeps ``_reconnect_watcher_task`` on the CURRENT live task across
        # backoff respawns so a superseded handle never looks like a dead watcher.
        self._spawn_reconnect_watcher()

        for method, name in self._POST_RECONNECT_WATCHERS:
            self._spawn_supervised(getattr(self, method), name)

        # Scale-to-zero idle watcher ONLY when opted in (HERMES_SCALE_TO_ZERO stamp), messaging is
        # relay-only/absent, and a wakeUrl is registered. When armed it drives the relay dormant on
        # sustained idle, then suspends via flaps — Fly autostop is inbound-only, job-blind.
        try:
            if self._scale_to_zero_should_arm():
                logger.info(
                    "scale-to-zero: armed (idle timeout %.0fs) — watching for idle",
                    self._scale_to_zero_idle_timeout_seconds(),
                )
                self._spawn_supervised(self._scale_to_zero_watcher, "scale_to_zero_watcher")
            else:
                # Surface WHY an OPTED-IN instance didn't arm (non-opted not arming is normal —
                # stay silent); otherwise a failed arm is invisible and needs a box-dive.
                self._log_scale_to_zero_not_armed_reason()
        except Exception:  # noqa: BLE001 - arming must never block startup
            logger.debug("scale-to-zero: arm check failed at startup", exc_info=True)

        # Drain-control watcher: reconciles the gateway's new-turn accept-state with the external
        # ``.drain_request.json`` marker the dashboard begin/cancel-drain endpoint writes. A marker
        # from a prior instantiation (durable-volume restart) is ignored via its epoch.
        self._spawn_supervised(self._drain_control_watcher, "drain_control_watcher")

    async def start(self) -> bool:
        """Start the gateway and all configured platform adapters.

        Returns True if at least one adapter connected successfully.
        """
        logger.info("Starting Hermes Gateway...")
        self._start_install_faulthandler()
        self._start_log_startup_environment()
        if await self._abort_startup_if_shutdown_requested():
            return True
        if self._start_check_access_policy():
            return True
        await self._start_recover_previous_run()

        # Serialize startup restore against inbound dispatch: adapters can receive messages as soon
        # as they connect, but restart-interrupted sessions are not auto-resumed until all startup
        # wiring below completes, so inbound queues until every synthetic resume turn has finished.
        self._startup_restore_in_progress = True
        self._startup_restore_queue = []
        self._startup_restore_tasks = []
        # Fresh-boot readiness: with no resume_pending sessions the gate opens almost immediately
        # while the turn machinery is still cold, so a message in that window got a skeleton system
        # prompt. Warm NOW to overlap the connects below; _finish_startup_restore awaits it (bounded).
        self._start_startup_warmup()

        startup_nonretryable_errors: list[str] = []
        startup_retryable_errors: list[str] = []
        (
            _aborted,
            enabled_platform_count,
            _multiplex_skipped_platforms,
            _pending_connects,
        ) = await self._start_prefilter_platforms()
        if _aborted:
            return True

        if await self._abort_startup_if_shutdown_requested():
            return True
        _raw = await self._start_connect_pending(_pending_connects)
        if _raw is None:
            return True
        connected_count = await self._start_aggregate_connect_results(
            _raw, startup_retryable_errors, startup_nonretryable_errors
        )

        if await self._abort_startup_if_shutdown_requested():
            return True
        _aborted, connected_count = await self._start_secondary_profiles(
            connected_count, _multiplex_skipped_platforms
        )
        if _aborted:
            return True
        if self._start_handle_no_connections(
            connected_count,
            enabled_platform_count,
            startup_retryable_errors,
            startup_nonretryable_errors,
        ):
            return True

        # Update delivery router with adapters
        if await self._abort_startup_if_shutdown_requested():
            return True
        self.delivery_router.adapters = self.adapters
        self._wire_teams_pipeline_runtime()

        self._running = True
        self._install_plugin_message_injector()
        self._update_runtime_status("running")
        await self._start_finish_wiring(connected_count)
        self._start_spawn_background_watchers()

        logger.info("Press Ctrl+C to stop")

        return True

    @dataclasses.dataclass
    class _HandoffDestination:
        """Resolved destination for one handoff row."""
        platform: Platform
        platform_name: str
        transport: Any
        home: Any
        home_chat_id: str
        effective_thread_id: Optional[str]
        source: SessionSource
        handoff_config: Any

    def _handoff_resolve_scope(self, profile_name: Optional[str]):
        """Return (config, adapters) for the profile that queued the handoff.

        Single-profile gateways (or a default-profile handoff) use self.config/self.adapters. For
        a secondary profile the watcher already entered _profile_runtime_scope, so a fresh load
        resolves THAT profile's config; fail closed — self.config would deliver to the WRONG chat.
        """
        from gateway.run import load_gateway_config
        if not profile_name or profile_name == "default":
            return self.config, self.adapters
        secondary = (self._profile_adapters or {}).get(profile_name)
        if not secondary:
            raise RuntimeError(
                f"profile '{profile_name}' has no live adapters in this gateway"
            )
        try:
            return load_gateway_config(), secondary
        except Exception as exc:
            logger.error(
                "Handoff: could not load config for profile %s; "
                "failing the handoff instead of delivering via the "
                "primary's config",
                profile_name, exc_info=True,
            )
            raise RuntimeError(
                f"could not load config for profile '{profile_name}': {exc}"
            ) from exc

    async def _handoff_resolve_destination(
        self, row: Dict[str, Any], profile_name: Optional[str]
    ) -> "GatewayStartupMixin._HandoffDestination":
        """Resolve platform, transport, home channel, thread and destination source for a row."""
        from gateway.run import resolve_delivery_transport
        cli_session_id = row["id"]
        platform_name = (row.get("handoff_platform") or "").strip().lower()
        if not platform_name:
            raise RuntimeError("handoff_platform is empty")
        try:
            platform = Platform(platform_name)
        except (ValueError, KeyError):
            raise RuntimeError(f"unknown platform '{platform_name}'")

        handoff_config, handoff_adapters = self._handoff_resolve_scope(profile_name)

        # Adapter must be live. A relay-fronted gateway registers ONE adapter under Platform.RELAY
        # fronting N logical platforms, so a literal adapters.get(discord) misses a deliverable
        # platform; resolve_delivery_transport is the alias-aware resolver (native adapter wins).
        transport = resolve_delivery_transport(platform, handoff_config, handoff_adapters)
        if not transport:
            raise RuntimeError(
                f"platform '{platform_name}' is not active in this gateway"
            )
        home = handoff_config.get_home_channel(platform)
        if not home or not home.chat_id:
            raise RuntimeError(
                f"no home channel configured for {platform_name}; "
                f"run /sethome on the desired chat first"
            )

        # Fresh thread on the destination so the handoff has its own scrollback. Adapter returns
        # None if threading is unsupported (Matrix/WhatsApp/Signal/SMS) or creation failed.
        cli_title = row.get("title") or cli_session_id[:8]
        try:
            new_thread_id = await transport.adapter.create_handoff_thread(
                str(home.chat_id), f"Hermes — {cli_title}",
            )
        except Exception as exc:
            logger.debug(
                "Handoff: create_handoff_thread raised on %s: %s",
                platform_name, exc, exc_info=True,
            )
            new_thread_id = None
        effective_thread_id = new_thread_id or (
            str(home.thread_id) if home.thread_id else None
        )

        # Telegram private-chat DM topics are shaped differently from group/forum threads by the
        # inbound adapter: a handoff-created topic in a positive chat_id must use the DM-topic source
        # shape, or the synthetic turn binds a `thread` key while real replies arrive on a `dm` key.
        home_chat_id = str(home.chat_id)
        is_telegram_private_chat = (
            platform == Platform.TELEGRAM
            and looks_like_telegram_private_chat_id(home_chat_id)
        )
        if new_thread_id and not is_telegram_private_chat:
            dest_chat_type = "thread"
            dest_user_id = "system:handoff"
        else:
            # No thread — assume DM-style. For Telegram private-chat topics use the real user id
            # (== chat_id) so topic-mode checks and binding persistence match later inbound turns.
            dest_chat_type = "dm"
            dest_user_id = home_chat_id if is_telegram_private_chat else "system:handoff"
        # Discord (unlike Slack/Telegram) builds in-thread messages with ``chat_id == thread id``,
        # so key on the thread's OWN id; keying on the parent would make the next reply spawn anew.
        if platform == Platform.DISCORD and dest_chat_type == "thread" and effective_thread_id:
            dest_chat_id = str(effective_thread_id)
        else:
            dest_chat_id = home_chat_id
        dest_source = SessionSource(
            platform=platform,
            chat_id=dest_chat_id,
            chat_name=home.name,
            chat_type=dest_chat_type,
            user_id=dest_user_id,
            user_name="Handoff",
            thread_id=effective_thread_id,
            profile=profile_name,
        )
        return self._HandoffDestination(
            platform=platform,
            platform_name=platform_name,
            transport=transport,
            home=home,
            home_chat_id=home_chat_id,
            effective_thread_id=effective_thread_id,
            source=dest_source,
            handoff_config=handoff_config,
        )

    def _handoff_session_key(self, dest, profile_name: Optional[str]) -> str:
        """Build the destination session_key with the adapters' own rules.

        Thread keys omit user_id (thread_sessions_per_user default) so the next message shares it.
        The key is namespaced to the queuing profile: a multiplexed gateway would otherwise build
        ``agent:main:...`` while the profile's adapter routes inbound on ``agent:<profile>:...``.
        The store resolver is only the root fallback (None when multiplexing is off; old key
        unchanged). The isinstance check is load-bearing: a Mock store returns a truthy MagicMock.
        """
        platform_cfg = dest.handoff_config.platforms.get(dest.platform)
        extra = platform_cfg.extra if platform_cfg else {}
        handoff_profile = profile_name if (profile_name and profile_name != "default") else None
        if handoff_profile is None:
            try:
                store = getattr(self.async_session_store, "_store", self.async_session_store)
                resolver = getattr(store, "_resolve_profile_for_key", None)
                if callable(resolver):
                    resolved = resolver(dest.source)
                    if isinstance(resolved, str) and resolved.strip():
                        handoff_profile = resolved
            except Exception:
                logger.debug("Handoff: could not resolve profile namespace", exc_info=True)
        return build_session_key(
            dest.source,
            group_sessions_per_user=extra.get("group_sessions_per_user", True),
            thread_sessions_per_user=extra.get("thread_sessions_per_user", False),
            profile=handoff_profile,
        )

    async def _process_handoff(
        self, row: Dict[str, Any], profile_name: Optional[str] = None,
    ) -> None:
        """Execute one handoff row. Raises on failure (caller marks failed).

        ``profile_name`` (``None`` = root) is the profile whose store queued this handoff. Under
        multiplex it is load-bearing: ``self.adapters``/``self.config`` are the primary's (secondaries
        live in ``_profile_adapters``), and the session key must be namespaced ``agent:<profile>:...``
        or it binds a key nobody reads. Passing the name beats re-deriving it from the contextvar.
        """
        cli_session_id = row["id"]
        dest = await self._handoff_resolve_destination(row, profile_name)
        session_key = self._handoff_session_key(dest, profile_name)

        # Ensure a session_store entry exists for this key (get_or_create_session creates one for a
        # never-used home channel); switch_session then re-points it.
        await self.async_session_store.get_or_create_session(dest.source)
        # Re-bind the destination key to the CLI session_id: switch_session ends the prior session
        # in SQLite and reopens the CLI session under the new key; its transcript is now active.
        switched = await self.async_session_store.switch_session(session_key, cli_session_id)
        if switched is None:
            raise RuntimeError(
                f"could not switch session key {session_key} → {cli_session_id}"
            )
        # Evict any cached AIAgent for this key so the next dispatch rebuilds it against the CLI
        # session_id (mirrors /resume / /branch), and clear stale running-agent state so the
        # synthetic turn isn't queued behind it.
        self._evict_cached_agent(session_key)
        self._release_running_agent_state(session_key)

        cli_title = row.get("title") or cli_session_id[:8]
        synthetic_event = MessageEvent(
            text=(
                f"[Session was just handed off from CLI (\"{cli_title}\") to this "
                f"channel. The full prior conversation history is loaded above. "
                f"Briefly confirm you're working here and summarize what we were "
                f"working on, so the user can continue from this device.]"
            ),
            source=dest.source,
            internal=True,
        )
        logger.info(
            "Handoff: dispatching synthetic turn for CLI session %s → %s "
            "(home=%s, thread=%s, session_key=%s)",
            cli_session_id, dest.platform_name, dest.home.chat_id, dest.effective_thread_id,
            session_key,
        )
        # Dispatch through the runner directly: adapter.handle_message would spawn a background task
        # and lose error visibility; inline _handle_message keeps success/failure observable.
        response_text = await self._handle_message(synthetic_event)
        if not response_text:
            # Streaming may have already delivered the response inline; the agent ran without
            # raising either way — count as success.
            return

        # Send the reply to the new thread if we created one, else the configured home channel
        # (which may carry a thread_id). Use the resolved transport (not adapter.send) so a
        # relay-fronted logical platform is stamped on the outbound frame (send_for_platform).
        send_metadata = {"thread_id": dest.effective_thread_id} if dest.effective_thread_id else None
        try:
            result = await dest.transport.send(
                dest.platform, str(dest.home.chat_id), response_text, send_metadata,
            )
        except Exception as exc:
            raise RuntimeError(f"adapter.send failed: {exc}") from exc
        if not getattr(result, "success", True):
            err = getattr(result, "error", "send returned success=False")
            raise RuntimeError(f"adapter.send failed: {err}")
