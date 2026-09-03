"""Stop/drain/restart, scale-to-zero and active-work accounting methods for GatewayRunner.

Split out of ``gateway/run.py``; bound onto ``GatewayRunner`` via the MRO.
``gateway.run`` internals are imported lazily inside method bodies (import cycle),
so ``patch("gateway.run.X")`` keeps intercepting them at call time.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
import asyncio
import os
import shlex
import sys
import threading
import time
from contextlib import suppress
from gateway.config import Platform
from gateway.restart import (
    DEFAULT_GATEWAY_CRON_DRAIN_TIMEOUT,
    GATEWAY_SERVICE_RESTART_EXIT_CODE,
    resolve_cron_drain_budget,
)
from gateway.run_common import _UNSET
from gateway.shutdown_watchdog import arm_shutdown_watchdog, resolve_shutdown_watchdog_delay
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

if TYPE_CHECKING:  # string annotations only; never imported at runtime (cycle)
    from gateway.run import GatewayRunner, TurnRunner  # noqa: F401

# Log-record parity with the origin module.
logger = logging.getLogger("gateway.run")


class GatewayShutdownMixin:
    """Stop/drain/restart, scale-to-zero and active-work accounting methods for GatewayRunner."""

    def _active_work_count(self) -> int:
        """All agent work the gateway must expose and drain as one total."""
        return (
            self._running_agent_count()
            + self._active_cron_job_count()
            + self._active_api_run_count()
            + self._active_deferred_agent_worker_count()
        )

    def _active_cron_job_count(self) -> int:
        """Count of cron jobs currently executing (``cron.scheduler._running_job_ids``).

        Cron jobs run on the scheduler's own thread pool, outside ``self._running_agents`` which
        every OTHER active-work check reads; without this the shutdown drain can kill a cron job's
        tool subprocess mid-run. Best-effort: returns 0 if the cron module can't be imported.
        """
        try:
            from cron.scheduler import get_running_job_ids
            return len(get_running_job_ids())
        except Exception:
            return 0

    def _active_api_run_count(self) -> int:
        """Count API-server work that is outside ``_running_agents``.

        Only the primary API server owns the HTTP listener (secondary multiplex profiles cannot
        bind a port), so only the primary registry is a source of this work.
        """
        try:
            adapter = getattr(self, "adapters", {}).get(Platform.API_SERVER)
            helper = getattr(adapter, "active_agent_work_count", None)
            return max(0, int(helper())) if callable(helper) else 0
        except Exception:
            return 0

    def _interrupt_api_server_runs(self, reason: str) -> int:
        """Interrupt API-server agents that are not in ``_running_agents``.

        Counterpart of ``_active_api_run_count()``: must reach the same agents when the drain times
        out. Duck-typed so an adapter (or test double) without the hook is skipped, not raised on.
        """
        try:
            adapter = getattr(self, "adapters", {}).get(Platform.API_SERVER)
            helper = getattr(adapter, "interrupt_active_runs", None)
            return max(0, int(helper(reason))) if callable(helper) else 0
        except Exception as exc:
            logger.debug("Failed interrupting api_server runs during shutdown: %s", exc)
            return 0

    def _active_deferred_agent_worker_count(self) -> int:
        """Count executor workers that outlived their owning gateway turn.

        A timed-out hygiene compression keeps running in its executor thread.
        Some paths defer agent cleanup; the live Codex path keeps its cached
        agent. In both cases the turn can finish before the worker does, so
        ``_running_agents`` no longer represents it. Count the worker itself.
        """
        workers = getattr(self, "_deferred_agent_workers", None)
        if not isinstance(workers, dict):
            return 0
        return sum(1 for future in list(workers) if not future.done())

    def _track_deferred_agent_worker(
        self,
        future: asyncio.Future,
        agent: Any,
    ) -> None:
        """Expose an executor worker to drain/interrupt until it really exits."""
        workers = getattr(self, "_deferred_agent_workers", None)
        if workers is None:
            workers = {}
            self._deferred_agent_workers = workers
        workers[future] = agent

        def _discard_worker(done_future: asyncio.Future) -> None:
            workers.pop(done_future, None)
            # Some tracked workers intentionally outlive the coroutine that
            # started them and therefore have no later waiter. Consume their
            # terminal exception so asyncio does not emit an unhandled-future
            # warning after the worker eventually unwinds (#98973).
            if not done_future.cancelled():
                try:
                    done_future.exception()
                except Exception:
                    pass

        future.add_done_callback(_discard_worker)

    def _interrupt_deferred_agent_workers(self, reason: str) -> int:
        """Request cancellation of detached executor-backed agent work."""
        from gateway.run import request_hard_interrupt
        workers = getattr(self, "_deferred_agent_workers", None)
        if not isinstance(workers, dict):
            return 0
        interrupted = 0
        seen: set[int] = set()
        for future, agent in list(workers.items()):
            if future.done() or agent is None or id(agent) in seen:
                continue
            seen.add(id(agent))
            try:
                request_hard_interrupt(agent, reason)
                interrupted += 1
            except Exception as exc:
                logger.debug(
                    "Failed interrupting deferred agent worker during shutdown: %s",
                    exc,
                )
        return interrupted

    def _scale_to_zero_has_live_background_work(self) -> bool:
        """Live background work that must block a suspend.

        Backgrounded delegate_task / kanban / terminal(background=true) are NOT counted by
        _running_agent_count() but suspending loses them; checks tracked tasks + process registry +
        pending completion watchers. PERMANENT supervised watchers (_hermes_supervised_watcher) are
        excluded — they live for the whole process (including the scale-to-zero watcher itself), so
        counting them would make this True forever and the gateway could never go dormant.
        """
        if any(
            not t.done() and not getattr(t, "_hermes_supervised_watcher", False)
            for t in self._background_tasks
        ):
            return True
        try:
            from tools.async_delegation import active_count

            if active_count() > 0:
                return True
        except Exception:  # noqa: BLE001 - never let the idle check raise
            logger.debug("scale-to-zero async-delegation check failed", exc_info=True)
        try:
            from tools.process_registry import process_registry

            if process_registry.has_any_active():
                return True
            if process_registry.pending_watchers:
                return True
        except Exception:  # noqa: BLE001 - never let the idle check raise
            logger.debug("scale-to-zero bg-work check failed", exc_info=True)
        return False

    def _scale_to_zero_idle_timeout_seconds(self) -> float:
        from gateway.run import _load_gateway_config
        from gateway.scale_to_zero import parse_idle_timeout_seconds

        raw = None
        try:
            user_cfg = _load_gateway_config()
            gw = user_cfg.get("gateway") if isinstance(user_cfg, dict) else None
            stz = gw.get("scale_to_zero") if isinstance(gw, dict) else None
            if isinstance(stz, dict):
                raw = stz.get("idle_timeout_minutes")
        except Exception:  # noqa: BLE001
            raw = None
        return parse_idle_timeout_seconds(raw)

    def _restart_loop_guard_config(self) -> tuple:
        """Return ``(max_restarts, window_seconds, max_gap_seconds)`` for the auto-resume
        restart-loop breaker, from ``gateway.restart_loop_guard`` with module defaults as fallback.

        ``max_restarts <= 0`` disables the breaker. ``max_gap_seconds`` is the longest spacing
        between consecutive restart-interrupted boots that still counts as the same loop, so a
        crash cycle slower than ``window_seconds`` stays visible.
        """
        from gateway.run import _load_gateway_config
        from gateway import restart_loop_guard as _rlg

        max_restarts = _rlg.DEFAULT_MAX_RESTARTS
        window_seconds = _rlg.DEFAULT_WINDOW_SECONDS
        max_gap_seconds = _rlg.DEFAULT_MAX_GAP_SECONDS
        try:
            user_cfg = _load_gateway_config()
            gw = user_cfg.get("gateway") if isinstance(user_cfg, dict) else None
            rlg = gw.get("restart_loop_guard") if isinstance(gw, dict) else None
            if isinstance(rlg, dict):
                if isinstance(rlg.get("max_restarts"), int):
                    max_restarts = rlg["max_restarts"]
                if isinstance(rlg.get("window_seconds"), int) and rlg["window_seconds"] > 0:
                    window_seconds = rlg["window_seconds"]
                if (
                    isinstance(rlg.get("max_gap_seconds"), int)
                    and rlg["max_gap_seconds"] > 0
                ):
                    max_gap_seconds = rlg["max_gap_seconds"]
        except Exception:  # noqa: BLE001
            pass
        return max_restarts, window_seconds, max_gap_seconds

    def _scale_to_zero_active_messaging_platforms(self) -> list:
        """ENABLED platforms that count for the relay-only arm gate.

        Two load-bearing filters: enabled only (config.platforms is pre-seeded with disabled
        placeholders for the whole catalog) and MESSAGING only (the api_server is a loopback listener
        force-enabled on every hosted container with no outbound socket; counting it silently
        disarmed the feature everywhere). Mirrors the non-messaging exclusion in _connect_platforms.
        """
        if not self.config:
            return []
        non_messaging = {Platform.LOCAL, Platform.API_SERVER, Platform.WEBHOOK}
        try:
            return [
                p
                for p, pc in self.config.platforms.items()
                if getattr(pc, "enabled", False) and p not in non_messaging
            ]
        except Exception:  # noqa: BLE001
            return []

    def _scale_to_zero_should_arm(self) -> bool:
        """Whether to start the idle watcher (D1/D11/§3.4(1))."""
        from gateway.relay import relay_wake_url
        from gateway.scale_to_zero import (
            messaging_is_relay_only_or_absent,
            scale_to_zero_enabled,
            should_arm,
        )

        platforms = self._scale_to_zero_active_messaging_platforms()
        try:
            wake_url = relay_wake_url()
        except Exception:  # noqa: BLE001
            wake_url = None
        return should_arm(
            enabled=scale_to_zero_enabled(),
            relay_only_or_absent=messaging_is_relay_only_or_absent(platforms),
            wake_url=wake_url,
        )

    def _log_scale_to_zero_not_armed_reason(self) -> None:
        """Log why the idle watcher did NOT arm — but only for an OPTED-IN instance.

        A non-opted instance (no HERMES_SCALE_TO_ZERO stamp) not arming is normal and stays silent;
        with the stamp set, the surprise earns one INFO line so the answer is a log grep.
        """
        from gateway.relay import relay_wake_url
        from gateway.scale_to_zero import (
            messaging_is_relay_only_or_absent,
            scale_to_zero_enabled,
        )

        try:
            enabled = scale_to_zero_enabled()
            if not enabled:
                return  # not opted in — normal, stay quiet
            active = [
                getattr(p, "value", p)
                for p in self._scale_to_zero_active_messaging_platforms()
            ]
            relay_only = messaging_is_relay_only_or_absent(active)
            try:
                wake_url = relay_wake_url()
            except Exception:  # noqa: BLE001
                wake_url = None
            logger.info(
                "scale-to-zero: NOT armed despite opt-in — "
                "relay_only_or_absent=%s (enabled platforms=%s), wake_url=%s. "
                "Need relay-only messaging + a registered wake URL.",
                relay_only,
                active or "none",
                "set" if wake_url else "MISSING",
            )
        except Exception:  # noqa: BLE001 - diagnostics must never block startup
            logger.debug("scale-to-zero: not-armed reason logging failed", exc_info=True)

    def _scale_to_zero_is_idle(self) -> bool:
        from gateway.scale_to_zero import is_idle

        # The FULL work aggregate, not _running_agent_count(): cron jobs and API-server runs live
        # outside _running_agents, so counting agents alone let a suspend land mid-cron-job.
        # Fail-AWAKE accounting: the shutdown-drain counters swallow exceptions to 0, which is fine
        # for a drain but unsafe for a suspend predicate (a transient read failure would look idle).
        # Here an unreadable source counts as work (sentinel 1) so the machine stays awake.
        try:
            from cron.scheduler import get_running_job_ids

            cron_count = len(get_running_job_ids())
        except Exception:  # noqa: BLE001 - unreadable source => assume busy
            logger.debug("scale-to-zero: cron work count unreadable — staying awake", exc_info=True)
            cron_count = 1
        try:
            adapter = getattr(self, "adapters", {}).get(Platform.API_SERVER)
            helper = getattr(adapter, "active_agent_work_count", None)
            api_count = max(0, int(helper())) if callable(helper) else 0
        except Exception:  # noqa: BLE001 - unreadable source => assume busy
            logger.debug("scale-to-zero: api work count unreadable — staying awake", exc_info=True)
            api_count = 1
        # An attached dashboard/desktop/TUI client is inbound activity too; it lives in the DASHBOARD
        # process and reaches us as a file mtime refreshed on every WS frame (gateway/scale_to_zero.py).
        # Folded into the inbound clock rather than a conjunct: same idle_timeout grace after
        # disconnect as a chat message, and a lingering marker cannot pin the box.
        last_inbound = self._last_inbound_at
        try:
            from gateway.scale_to_zero import dashboard_client_last_seen

            seen = dashboard_client_last_seen()
        except Exception:  # noqa: BLE001 - unreadable source => assume busy
            logger.debug("scale-to-zero: dashboard heartbeat unreadable — staying awake", exc_info=True)
            seen = time.time()
        if seen is not None and seen > last_inbound:
            last_inbound = seen
        return is_idle(
            active_work_count=self._running_agent_count() + cron_count + api_count,
            seconds_since_last_inbound=time.time() - last_inbound,
            idle_timeout_seconds=self._scale_to_zero_idle_timeout_seconds(),
            has_live_background_work=self._scale_to_zero_has_live_background_work(),
        )

    def _scale_to_zero_note_real_inbound(self) -> None:
        """Stamp real inbound and restore lifecycle after a dormant wake.

        Dormancy marks status `draining` but is not the stop/restart drain: the process stays alive
        and should present as running once real traffic wakes it. Internal completion/replay events
        deliberately do not call this, so they don't keep an idle gateway awake.
        """
        self._last_inbound_at = time.time()
        if getattr(self, "_scale_to_zero_cooldown_until", 0.0) > 0:
            try:
                self._update_runtime_status("running")
            except Exception:  # noqa: BLE001 - status restoration is best-effort
                logger.debug("scale-to-zero: status restore failed", exc_info=True)
            self._scale_to_zero_cooldown_until = 0.0

    def _relay_adapter_for_dormancy(self):
        """Return the connected RELAY adapter, if any (the one go_dormant targets)."""
        try:
            from gateway.platforms.base import Platform
        except Exception:  # noqa: BLE001
            return None
        return self.adapters.get(Platform.RELAY)

    async def _scale_to_zero_watcher(self, interval: float = 30.0) -> None:
        """Watch for idle, drive the relay dormant, then self-suspend the machine.

        Armed ONLY via _scale_to_zero_should_arm() (HERMES_SCALE_TO_ZERO stamp + relay-only/absent
        messaging + wakeUrl). On sustained idle: mark status `draining` (NOT _running=False), relay
        adapter.go_dormant() (supervisor-preserving socket close, NOT disconnect()), NO
        mark_resume_pending (suspend preserves RAM), THEN suspend via the local flaps socket. The
        gateway owns the suspend because Fly autostop sees only INBOUND connections and would freeze
        mid-job or before the relay flip (machines run autostop:"off"); autostart stays platform-side.
        A re-arm cooldown keeps a wake's drained backlog from being re-quiesced. Off-Fly (no flaps
        socket) the watcher does not quiesce at all.
        """
        await asyncio.sleep(min(interval, 30.0))  # let startup settle
        while self._running:
            try:
                await asyncio.sleep(interval)
                if not self._running:
                    return
                if time.time() < self._scale_to_zero_cooldown_until:
                    continue
                if not self._scale_to_zero_is_idle():
                    continue
                adapter = self._relay_adapter_for_dormancy()
                if adapter is None:
                    continue
                go_dormant = getattr(adapter, "go_dormant", None)
                if not callable(go_dormant):
                    continue
                # Quiesce only when a suspend can follow. Off-Fly the platform owns the freeze and
                # go_dormant()'s socket close arms the reconnect supervisor (re-dial ~1.4s, unflipped
                # at freeze, inbound dropped not buffered); stay connected, orphan detection adopts it.
                from gateway.scale_to_zero import self_suspend_available

                if not self_suspend_available():
                    if not self._scale_to_zero_no_suspend_logged:
                        self._scale_to_zero_no_suspend_logged = True
                        logger.info(
                            "scale-to-zero: idle, but this platform suspends on "
                            "its own timer (no in-machine suspend API); staying "
                            "connected rather than quiescing"
                        )
                    continue
                logger.info(
                    "scale-to-zero: gateway idle for >= %.0fs — going dormant "
                    "(relay buffered, socket closed) then self-suspending",
                    self._scale_to_zero_idle_timeout_seconds(),
                )
                try:
                    self._update_runtime_status("draining")
                except Exception:  # noqa: BLE001 - status is best-effort
                    logger.debug("scale-to-zero: status mark failed", exc_info=True)
                dormant_ok = True
                try:
                    result = go_dormant()
                    if asyncio.iscoroutine(result):
                        await result
                except Exception:  # noqa: BLE001 - dormancy is best-effort
                    dormant_ok = False
                    logger.debug("scale-to-zero: go_dormant failed", exc_info=True)
                # After a wake the drained inbound updates _last_inbound_at; give it a window so we
                # don't immediately re-go-dormant on the same idle reading before traffic lands.
                self._scale_to_zero_cooldown_until = time.time() + max(interval, 60.0)
                # Self-suspend ONLY after a clean quiesce: the relay flip (buffered delivery + wake
                # poke armed) must be set before the freeze, or inbound black-holes while we sleep.
                # Re-check idle one last time — inbound may have landed during the quiesce await.
                if not dormant_ok:
                    continue
                if not self._scale_to_zero_is_idle():
                    logger.info(
                        "scale-to-zero: inbound arrived during quiesce — skipping suspend"
                    )
                    continue
                await self._scale_to_zero_self_suspend()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - the watcher must never crash the gateway
                logger.debug("scale-to-zero watcher iteration error", exc_info=True)

    async def _scale_to_zero_self_suspend(self) -> None:
        """Suspend this Fly machine via the local flaps socket (fail-awake).

        Blocking unix-socket call runs in a worker thread so the loop stays live until the kernel
        freeze; nothing meaningful runs until wake. Off-Fly this is a silent no-op.
        """
        from gateway.scale_to_zero import self_suspend_available, suspend_self

        try:
            if not self_suspend_available():
                logger.debug(
                    "scale-to-zero: flaps socket / machine identity absent — "
                    "dormant without platform suspend"
                )
                return
            accepted = await asyncio.to_thread(suspend_self)
            if not accepted:
                logger.warning(
                    "scale-to-zero: self-suspend not accepted — machine stays "
                    "awake (fail-awake); will retry on the next idle window"
                )
        except Exception:  # noqa: BLE001 - suspend is best-effort, never crash
            logger.debug("scale-to-zero: self-suspend failed", exc_info=True)

    # ------------------------------------------------------------------
    # External drain control (NAS-driven quiesce-without-restart). The dashboard's
    # begin/cancel-drain endpoint writes/removes the ``.drain_request.json`` marker
    # (gateway/drain_control.py); this watcher flips the gateway between accepting and refusing
    # NEW turns WITHOUT exiting. Reversible: NAS begins drain, polls /api/status until
    # active_agents hits 0, acts; on cancel/abort the marker is removed and turns resume.
    # ------------------------------------------------------------------
    def _enter_external_drain(self) -> None:
        """Begin external drain: refuse NEW turns (in-flight ones are NOT interrupted), flip state.

        Idempotent: re-entry only re-writes status.
        """
        if self._external_drain_active:
            return
        self._external_drain_active = True
        logger.info(
            "External drain ENGAGED (.drain_request.json present) — refusing "
            "new turns; %d in-flight turn(s) will finish. Process stays up.",
            self._active_work_count(),
        )
        # Flip persisted lifecycle state so /api/status.gateway_busy / gateway_drainable track the
        # drain; active_agents is preserved (read-merge keeps the live count), only state changes.
        self._update_runtime_status("draining")

    def _exit_external_drain(self) -> None:
        """Cancel external drain: revert state, re-accept new turns.

        Idempotent. Reverts to ``running`` only when actually mid-drain AND not shutting down —
        a real shutdown ``_draining`` must win; never resurrect a stopping gateway.
        """
        if not self._external_drain_active:
            return
        self._external_drain_active = False
        if self._draining or not self._running:
            # A shutdown drain is in progress / the loop has stopped — do not
            # clobber the terminal state back to running.
            logger.info(
                "External drain marker cleared during shutdown — not reverting "
                "to running (shutdown takes precedence)."
            )
            return
        logger.info(
            "External drain RELEASED (.drain_request.json removed) — "
            "re-accepting new turns; gateway_state -> running."
        )
        self._update_runtime_status("running")

    async def _drain_control_watcher(self, interval: float = 1.0) -> None:
        """Background task: reconcile gateway accept-state with the drain marker.

        Polls ``.drain_request.json`` (presence-based) at 1s: present -> enter drain, absent -> exit;
        reconciles once at startup. A marker from a PRIOR instantiation epoch (survived a machine
        restart) is treated as absent. Best-effort: tick errors are logged and the loop continues.
        """
        from gateway.drain_control import drain_requested

        while self._running:
            try:
                # drain_requested() does a synchronous read_text() on the marker file: at 1s cadence
                # that is a blocking disk read on the event loop ~86k times/day, and under host I/O
                # pressure one read can stall 30s+ and take every platform heartbeat down. Off-thread it.
                if await asyncio.to_thread(drain_requested):
                    self._enter_external_drain()
                    # API and cron work live outside messaging's _running_agents map; refresh the
                    # aggregate while an external caller polls this reversible drain state.
                    self._persist_active_agents()
                else:
                    self._exit_external_drain()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.debug("Drain-control watcher tick error: %s", exc, exc_info=True)
            await asyncio.sleep(interval)

    def _update_platform_runtime_status(
        self,
        platform: str,
        *,
        platform_state: Optional[str] = None,
        error_code: Optional[str] = None,
        error_message: Optional[str] = None,
        needs_attention: Optional[bool] = None,
        retrying_since: Any = _UNSET,
    ) -> None:
        try:
            from gateway.status import write_runtime_status
            extra: Dict[str, Any] = {}
            if needs_attention is not None:
                extra["needs_attention"] = needs_attention
            if retrying_since is not _UNSET:
                extra["retrying_since"] = retrying_since
            write_runtime_status(
                platform=platform,
                platform_state=platform_state,
                error_code=error_code,
                error_message=error_message,
                **extra,
            )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Per-platform circuit breaker (pause/resume): reconnect watcher + /platform pause|resume.
    # ------------------------------------------------------------------
    def _pause_failed_platform(self, platform, *, reason: str = "") -> None:
        """Mark a queued platform as paused — stays in ``_failed_platforms`` but the reconnect
        watcher stops hammering it.

        Manual (``/platform pause <name>``) only: the watcher never auto-pauses — retryable failures
        keep retrying at the backoff cap so a transient outage self-heals.
        """
        info = getattr(self, "_failed_platforms", {}).get(platform)
        if info is None:
            return
        if info.get("paused"):
            return
        info["paused"] = True
        info["pause_reason"] = reason or "auto-paused after repeated failures"
        # Push next_retry far enough out that even if "paused" is missed
        # by a stale code path, the watcher won't fire on it.
        info["next_retry"] = float("inf")
        with suppress(Exception):
            self._update_platform_runtime_status(
                platform.value,
                platform_state="paused",
                error_code=None,
                error_message=info["pause_reason"],
            )
        logger.warning(
            "%s paused after %d consecutive failures (%s) — "
            "fix the underlying issue then run `/platform resume %s` "
            "to retry, or `hermes gateway restart` to restart the gateway.",
            platform.value, info.get("attempts", 0),
            info["pause_reason"], platform.value,
        )

    def _resume_paused_platform(self, platform) -> bool:
        """Unpause a platform — reset its attempt counter and schedule an
        immediate retry.  Returns True if the platform was paused and is
        now queued; False if it wasn't paused (or wasn't in the queue).
        """
        info = getattr(self, "_failed_platforms", {}).get(platform)
        if info is None:
            return False
        if not info.get("paused"):
            return False
        info["paused"] = False
        info.pop("pause_reason", None)
        info["attempts"] = 0
        info["next_retry"] = time.monotonic()  # retry on next watcher tick
        with suppress(Exception):
            self._update_platform_runtime_status(
                platform.value,
                platform_state="retrying",
                error_code=None,
                error_message=None,
            )
        logger.info("%s resumed — retrying on next watcher tick", platform.value)
        return True

    async def _drain_active_agents(
        self, timeout: float, cron_timeout: Optional[float] = None
    ) -> tuple[Dict[str, Any], bool]:
        snapshot = self._snapshot_running_agents()
        last_active_count = self._running_agent_count()
        last_cron_count = self._active_cron_job_count()
        last_api_count = self._active_api_run_count()
        last_deferred_count = self._active_deferred_agent_worker_count()
        last_status_at = 0.0

        def _maybe_update_status(force: bool = False) -> None:
            nonlocal last_active_count, last_cron_count, last_api_count
            nonlocal last_deferred_count, last_status_at
            now = asyncio.get_running_loop().time()
            active_count = self._running_agent_count()
            cron_count = self._active_cron_job_count()
            api_count = self._active_api_run_count()
            deferred_count = self._active_deferred_agent_worker_count()
            if (
                force
                or active_count != last_active_count
                or cron_count != last_cron_count
                or api_count != last_api_count
                or deferred_count != last_deferred_count
                or (now - last_status_at) >= 1.0
            ):
                self._update_runtime_status("draining")
                last_active_count = active_count
                last_cron_count = cron_count
                last_api_count = api_count
                last_deferred_count = deferred_count
                last_status_at = now

        # Cron jobs run on the scheduler's pool, outside ``self._running_agents`` — fold their in-flight
        # count into this wait, or a cron job's tool work is killed without warning once it's the only
        # active thing running. API-server/desk sessions and detached deferred workers share the gap.
        if (
            not self._running_agents
            and last_cron_count == 0
            and last_api_count == 0
            and last_deferred_count == 0
        ):
            _maybe_update_status(force=True)
            return snapshot, False

        _maybe_update_status(force=True)

        # Cron drains on its own deadline: ``timeout`` (``restart_drain_timeout``) defaults to 0 since
        # an interrupted chat turn is announced and resumable, while a cron run killed mid-flight is a
        # permanent failure nobody is waiting on. One shared budget would kill cron after 0.00s.
        loop = asyncio.get_running_loop()
        started = loop.time()
        deadline = started + timeout
        cron_deadline = started + (timeout if cron_timeout is None else cron_timeout)

        def _still_draining() -> bool:
            now = loop.time()
            if (
                len(self._running_agents)
                or self._active_api_run_count()
                or self._active_deferred_agent_worker_count()
            ) and now < deadline:
                return True
            return bool(self._active_cron_job_count()) and now < cron_deadline

        # Both budgets at 0 leave this loop unentered ("interrupt immediately") as an expired deadline,
        # not a special case, so timed_out below is always computed from real state.
        while _still_draining():
            _maybe_update_status()
            await asyncio.sleep(0.1)
        timed_out = (
            bool(len(self._running_agents))
            or bool(self._active_cron_job_count())
            or bool(self._active_api_run_count())
            or bool(self._active_deferred_agent_worker_count())
        )
        _maybe_update_status(force=True)
        return snapshot, timed_out

    def _interrupt_running_agents(self, reason: str) -> None:
        from gateway.run import _AGENT_PENDING_SENTINEL, request_hard_interrupt
        for session_key, agent in list(self._running_agents.items()):
            if agent is _AGENT_PENDING_SENTINEL:
                continue
            try:
                request_hard_interrupt(agent, reason)
                logger.debug("Interrupted running agent for session %s during shutdown", session_key)
            except Exception as e:
                logger.debug("Failed interrupting agent during shutdown: %s", e)
        # API-server / desk turns are adapter-owned and never enter _running_agents, so the loop above
        # cannot see them even though _drain_active_agents() waited for them.
        interrupted_api = self._interrupt_api_server_runs(reason)
        if interrupted_api:
            logger.debug("Interrupted %d api_server run(s) during shutdown", interrupted_api)
        interrupted_deferred = self._interrupt_deferred_agent_workers(reason)
        if interrupted_deferred:
            logger.debug(
                "Interrupted %d deferred agent worker(s) during shutdown",
                interrupted_deferred,
            )

    async def _notify_interrupted_cron_jobs(self, job_ids) -> int:
        """Tell the owner of each just-interrupted cron job that its run died.

        The cron worker can't: its thread reaches ``_deliver_result`` after teardown closed the
        transport. Must run post-interrupt while adapters are still connected (the window
        ``_notify_active_sessions_of_shutdown`` uses, which is blind to cron work). Best-effort: every
        failure is swallowed so a wedged adapter can't extend shutdown. Returns notices sent.
        """
        if not job_ids:
            return 0
        try:
            from cron.jobs import get_job
            from cron.scheduler import _resolve_delivery_targets
        except Exception as e:
            logger.debug("Cron interrupt notification unavailable: %s", e)
            return 0

        action = "restarting" if self._restart_requested else "shutting down"
        notified: set = set()
        for job_id in job_ids:
            try:
                job = get_job(job_id)
                if not job:
                    continue
                # deliver=local jobs, and deliver=origin jobs with no resolvable origin, resolve to zero
                # targets and must stay silent rather than fall back to a home channel. Interrupted
                # notices are failure-category engine status, so they honor failure_deliver.
                targets = _resolve_delivery_targets(job, for_failure=True)
            except Exception as e:
                logger.debug("Cron interrupt targets unresolved for %s: %s", job_id, e)
                continue
            if not targets:
                continue

            msg = (
                f"⚠️ Cron job '{job.get('name') or job_id}' was interrupted — "
                f"the gateway is {action} and killed the run before it "
                "finished. No result was produced for this run."
            )
            for target in targets:
                try:
                    platform = Platform(str(target.get("platform", "")).lower())
                except Exception:
                    continue
                adapter = self.adapters.get(platform)
                if adapter is None:
                    continue
                platform_cfg = self.config.platforms.get(platform)
                if platform_cfg is not None and not platform_cfg.gateway_restart_notification:
                    continue

                chat_id = str(target.get("chat_id"))
                thread_id = target.get("thread_id")
                dedup_key = (
                    job_id,
                    platform.value,
                    chat_id,
                    str(thread_id) if thread_id else None,
                )
                if dedup_key in notified:
                    continue
                try:
                    metadata = self._thread_metadata_for_target(
                        platform, chat_id, thread_id, adapter=adapter
                    )
                    result = await adapter.send(chat_id, msg, metadata=metadata)
                    if result is not None and getattr(result, "success", True) is False:
                        logger.debug(
                            "Cron interrupt notice to %s:%s failed: %s",
                            platform.value, chat_id,
                            getattr(result, "error", "send returned success=False"),
                        )
                        continue
                    notified.add(dedup_key)
                except Exception as e:
                    logger.debug(
                        "Cron interrupt notice to %s:%s raised: %s",
                        platform.value, chat_id, e,
                    )
        if notified:
            logger.info(
                "Shutdown: delivered %d interrupted-cron-job notice(s)",
                len(notified),
            )
        return len(notified)

    async def _notify_active_sessions_of_shutdown(self) -> None:
        """Send shutdown/restart notifications to active chats and home channels.

        Called at the start of stop() while adapters are connected; send failures never block shutdown.
        """
        from gateway.run import _parse_session_key
        active = self._snapshot_running_agents()
        restart_source = self._restart_command_source if self._restart_requested else None

        action = "restarting" if self._restart_requested else "shutting down"
        hint = (
            "Your current task will be interrupted. "
            "Send any message after restart and I'll try to resume where you left off."
            if self._restart_requested
            else "Your current task will be interrupted."
        )
        msg = f"⚠️ Gateway {action} — {hint}"

        notified: set[tuple[str, str, Optional[str]]] = set()
        for session_key in active:
            source = None
            try:
                if getattr(self, "session_store", None) is not None:
                    await self.async_session_store._ensure_loaded()
                    entry = self.session_store._entries.get(session_key)
                    source = getattr(entry, "origin", None) if entry else None
            except Exception as e:
                logger.debug(
                    "Failed to load session origin for shutdown notification %s: %s",
                    session_key,
                    e,
                )

            if source is None:
                source = self._get_cached_session_source(session_key)

            if source is not None:
                platform_str = source.platform.value
                chat_id = str(source.chat_id)
                thread_id = source.thread_id
            else:
                # Fall back to parsing the session key when no persisted
                # origin is available (legacy sessions/tests).
                _parsed = _parse_session_key(session_key)
                if not _parsed:
                    continue
                platform_str = _parsed["platform"]
                chat_id = _parsed["chat_id"]
                thread_id = _parsed.get("thread_id")

            # Dedupe only identical targets: thread/topic platforms share a parent chat yet route to
            # distinct destinations via metadata.
            dedup_key = (platform_str, chat_id, str(thread_id) if thread_id else None)
            if dedup_key in notified:
                continue

            try:
                platform = Platform(platform_str)
                adapter = self.adapters.get(platform)
                if not adapter:
                    continue

                platform_cfg = self.config.platforms.get(platform)
                if platform_cfg is not None and not platform_cfg.gateway_restart_notification:
                    logger.info(
                        "Shutdown notification suppressed for active session: %s has gateway_restart_notification=false",
                        platform_str,
                    )
                    continue

                reply_to_message_id = getattr(source, "message_id", None) if source is not None else None
                if reply_to_message_id is None and restart_source is not None:
                    try:
                        restart_platform = restart_source.platform.value
                        restart_chat_id = str(restart_source.chat_id)
                        restart_thread_id = str(restart_source.thread_id) if restart_source.thread_id else None
                        if (restart_platform, restart_chat_id, restart_thread_id) == dedup_key:
                            reply_to_message_id = getattr(restart_source, "message_id", None)
                    except Exception:
                        pass

                metadata = self._thread_metadata_for_target(
                    platform,
                    chat_id,
                    thread_id,
                    chat_type=getattr(source, "chat_type", None) if source is not None else None,
                    reply_to_message_id=reply_to_message_id,
                    adapter=adapter,
                )

                result = await adapter.send(chat_id, msg, metadata=metadata)
                if result is not None and getattr(result, "success", True) is False:
                    logger.debug(
                        "Failed to send shutdown notification to %s:%s: %s",
                        platform_str,
                        chat_id,
                        getattr(result, "error", "send returned success=False"),
                    )
                    continue

                notified.add(dedup_key)
                logger.info(
                    "Sent shutdown notification to active chat %s:%s",
                    platform_str, chat_id,
                )
            except Exception as e:
                logger.debug(
                    "Failed to send shutdown notification to %s:%s: %s",
                    platform_str, chat_id, e,
                )

        if self._restart_requested and restart_source is not None:
            logger.debug("Skipping home-channel shutdown notifications for in-chat restart")
            return

        # Suppress ONLY the home-channel broadcast when the drain asked to be quiet (e.g. routine
        # auto-update on an always-on fleet). Per-session interrupt pings above are NOT gated: empty by
        # construction on a drained shutdown, and useful ("task cut off, message me to resume") on a
        # force-interrupt. Honoured only for a CURRENT-epoch marker (staleness check inside
        # drain_notification_suppressed), so an orphaned marker can't silence a fresh gateway.
        try:
            from gateway.drain_control import drain_notification_suppressed
            if drain_notification_suppressed():
                logger.info(
                    "Home-channel shutdown broadcast suppressed by drain marker "
                    "(suppress_notification=true)"
                )
                return
        except Exception as e:
            # Never let the suppression check block the shutdown broadcast —
            # fail toward the louder, more-visible behaviour.
            logger.debug("drain_notification_suppressed check failed: %s", e)

        # Snapshot adapters: adapter.send() can hit a fatal path (_handle_fatal) that pops the adapter
        # from self.adapters -> ``RuntimeError: dictionary changed size during iteration``.
        for platform, adapter in list(self.adapters.items()):
            home = self.config.get_home_channel(platform)
            if not home or not home.chat_id:
                continue

            platform_cfg = self.config.platforms.get(platform)
            if platform_cfg is not None and not platform_cfg.gateway_restart_notification:
                logger.info(
                    "Shutdown notification suppressed for home channel: %s has gateway_restart_notification=false",
                    platform.value,
                )
                continue

            dedup_key = (platform.value, str(home.chat_id), str(home.thread_id) if home.thread_id else None)
            if dedup_key in notified:
                continue

            try:
                metadata = self._thread_metadata_for_target(
                    platform,
                    home.chat_id,
                    home.thread_id,
                    adapter=adapter,
                )
                if metadata:
                    result = await adapter.send(str(home.chat_id), msg, metadata=metadata)
                else:
                    result = await adapter.send(str(home.chat_id), msg)
                if result is not None and getattr(result, "success", True) is False:
                    logger.debug(
                        "Failed to send shutdown notification to home channel %s:%s: %s",
                        platform.value,
                        home.chat_id,
                        getattr(result, "error", "send returned success=False"),
                    )
                    continue

                notified.add(dedup_key)
                logger.info(
                    "Sent shutdown notification to home channel %s:%s",
                    platform.value,
                    home.chat_id,
                )
            except Exception as e:
                logger.debug(
                    "Failed to send shutdown notification to home channel %s:%s: %s",
                    platform.value,
                    home.chat_id,
                    e,
                )

    async def _finalize_shutdown_agents(self, active_agents: Dict[str, Any]) -> None:
        for agent in active_agents.values():
            # Persist in-flight transcripts before teardown: a force-interrupted agent may never reach
            # finalize_turn (the only mid-turn flush), so its tool rounds would vanish from
            # load_transcript() on resume (resume already tolerates a pending-tool-result tail). The
            # flush is idempotent (identity-tracked); gracefully finished agents re-flush nothing.
            try:
                _flush = getattr(agent, "_flush_messages_to_session_db", None)
                _session_messages = getattr(agent, "_session_messages", None)
                if callable(_flush) and isinstance(_session_messages, list) and _session_messages:
                    # Strip empty-response retry scaffolding from the tail first (as ``_persist_session``
                    # does) so a resumed turn doesn't replay synthetic recovery nudges.
                    _strip = getattr(
                        agent, "_drop_trailing_empty_response_scaffolding", None
                    )
                    if callable(_strip):
                        with suppress(Exception):
                            _strip(_session_messages)
                    try:
                        _flush(_session_messages)
                    except Exception as _flush_err:
                        # Transcript could not be persisted (e.g. FTS/SQLite index corruption). A log
                        # line alone loses the conversation at exit, so dump the live history to an
                        # external JSON recovery snapshot. Non-fatal: shutdown never blocks on a backup.
                        logger.warning(
                            "Shutdown transcript flush failed (%s); preserving "
                            "%d in-memory message(s) to recovery snapshot",
                            _flush_err,
                            len(_session_messages),
                        )
                        from gateway.shutdown_flush import flush_agent_history_to_file
                        flush_agent_history_to_file(
                            getattr(agent, "session_id", None),
                            _session_messages,
                        )
            except Exception as _e:
                logger.debug("Shutdown transcript flush failed: %s", _e)
            # Off-loop + bounded: plugin on_session_finalize hooks can do arbitrary synchronous work
            # (e.g. a full-session trace export) — same hang class as the memory provider below.
            await self._finalize_session_off_loop(
                session_id=getattr(agent, "session_id", None),
                platform="gateway",
                reason="shutdown",
            )
            # Off-loop + bounded: a wedged memory provider here used to hang
            # the whole shutdown so SIGTERM never completed (#53175).
            await self._cleanup_agent_resources_off_loop(
                agent, context="shutdown finalize"
            )

    def _should_emit_long_running_notification(
        self,
        session_key: Optional[str],
        agent: Any,
        executor_task: Optional[Any],
    ) -> bool:
        """Only emit the heartbeat while this task still owns the live run.

        Stop once the executor finishes, the agent is gone, or the session key was rebound (e.g.
        ``/new`` mid-run) — else a stale ``running: delegate_task`` heartbeat outlives its run.
        """
        if agent is None:
            return False
        if executor_task is not None and executor_task.done():
            return False
        if session_key:
            _hb_state = self._peek_session_state(session_key)
            if (_hb_state.turn.agent if _hb_state else None) is not agent:
                return False
        return True

    def _defer_agent_cleanup_until_future_done(
        self,
        future: asyncio.Future,
        agent: Any,
        *,
        context: str,
    ) -> None:
        """Clean up ``agent`` only after its executor future has finished.

        A timed-out executor call keeps running in its worker thread; closing the agent first can
        tear down clients it still uses, so hold a strong task ref and await the real future.
        """

        async def _cleanup_when_done() -> None:
            try:
                await asyncio.shield(future)
            except asyncio.CancelledError:
                # Loop shutdown can cancel this waiter while the executor still
                # runs. Never turn that cancellation into premature cleanup.
                return
            except Exception as exc:
                logger.debug(
                    "Deferred agent worker%s finished with an error: %s",
                    f" ({context})" if context else "",
                    exc,
                )
            await self._cleanup_agent_resources_off_loop(agent, context=context)

        self._track_deferred_agent_worker(future, agent)

        task = asyncio.create_task(_cleanup_when_done())
        tasks = getattr(self, "_deferred_agent_cleanup_tasks", None)
        if tasks is None:
            tasks = set()
            self._deferred_agent_cleanup_tasks = tasks
        tasks.add(task)
        task.add_done_callback(tasks.discard)

    async def _finalize_session_off_loop(
        self,
        *,
        session_id: Any,
        platform: str,
        reason: str,
        **extra: Any,
    ) -> None:
        """Run hermes_cli.lifecycle.finalize_session off the event loop, bounded.

        On timeout the worker thread is left to finish (or leak) and the caller proceeds.
        """

        def _call() -> None:
            from hermes_cli.lifecycle import finalize_session

            finalize_session(
                session_id=session_id,
                platform=platform,
                reason=reason,
                **extra,
            )

        try:
            await asyncio.wait_for(
                self._run_in_executor_with_context(_call),
                timeout=self._FINALIZE_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Session finalize hooks (%s, reason=%s) exceeded %ss; "
                "proceeding without blocking the event loop (the worker "
                "thread is left to finish on its own).",
                session_id,
                reason,
                self._FINALIZE_TIMEOUT_S,
            )
        except Exception as finalize_exc:
            logger.debug(
                "Session finalize hooks (%s, reason=%s) failed: %s",
                session_id,
                reason,
                finalize_exc,
            )

    async def _cleanup_agent_resources_off_loop(
        self, agent: Any, *, context: str = ""
    ) -> None:
        """Run _cleanup_agent_resources in a worker thread with a bounded wait.

        On timeout the worker thread is left to finish (or leak) and the caller proceeds, as /new does.
        """
        if agent is None:
            return
        if context.startswith("shutdown") or context == "session expiry":
            with suppress(Exception):
                agent._end_session_on_close = False
        try:
            await asyncio.wait_for(
                self._run_in_executor_with_context(
                    self._cleanup_agent_resources, agent
                ),
                timeout=self._CLEANUP_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Agent resource cleanup%s exceeded %ss; proceeding without "
                "blocking the event loop (the worker thread is left to finish "
                "on its own). (#53175)",
                f" ({context})" if context else "",
                self._CLEANUP_TIMEOUT_S,
            )
        except Exception as cleanup_exc:
            logger.warning(
                "Agent resource cleanup%s failed: %s (#53175)",
                f" ({context})" if context else "",
                cleanup_exc,
            )

    def _cleanup_agent_resources(self, agent: Any) -> None:
        """Best-effort cleanup for temporary or cached agent instances."""
        if agent is None:
            return
        try:
            if hasattr(agent, "shutdown_memory_provider"):
                # Drain queued memory writes BEFORE tearing the provider down: shutdown_all() gives
                # the serialized memory worker only ~5s and cancels the rest, so a /reset or rotation
                # could drop handed-off writes and the next session loads stale memory. Bounded head
                # start via the manager's own barrier (mirrors CLI exit); a failure never blocks teardown.
                _mm = getattr(agent, "_memory_manager", None)
                if _mm is not None and hasattr(_mm, "flush_pending"):
                    with suppress(Exception):
                        _mm.flush_pending(timeout=10)
                # Pass the real transcript so ``on_session_end`` hooks don't see the empty default.
                # ``_session_messages`` may be absent on ``object.__new__`` test stubs, hence getattr.
                session_messages = getattr(agent, "_session_messages", None)
                if isinstance(session_messages, list):
                    agent.shutdown_memory_provider(session_messages)
                else:
                    agent.shutdown_memory_provider()
        except Exception:
            pass
        # Close tool resources (sandboxes, browser daemons, background processes, httpx clients).
        try:
            if hasattr(agent, "close"):
                agent.close()
        except Exception:
            pass
        # Auxiliary async clients live in a process-global cache created from worker threads; drop
        # entries whose event loop is dead so httpx transports don't accumulate across turns.
        try:
            from agent.auxiliary_client import cleanup_stale_async_clients
            cleanup_stale_async_clients()
        except Exception:
            pass

    def _increment_restart_failure_counts(self, active_session_keys: set) -> None:
        """Increment restart-failure counters for sessions active at shutdown.

        Persists to a JSON file so counters survive across restarts. Sessions NOT in
        active_session_keys are removed (they completed successfully, so the loop is broken).
        """
        from gateway.run import _hermes_home, atomic_json_write
        import json

        path = _hermes_home / self._STUCK_LOOP_FILE
        try:
            counts = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        except Exception:
            counts = {}

        # Increment active sessions, remove inactive ones (loop broken)
        new_counts = {}
        for key in active_session_keys:
            new_counts[key] = counts.get(key, 0) + 1
        # Keep any entries that are still above 0 even if not active now
        # (they might become active again next restart)

        with suppress(Exception):
            atomic_json_write(path, new_counts, indent=None)

    def _suspend_stuck_loop_sessions(self) -> int:
        """Suspend sessions active across too many restarts (load → stuck → restart loop).

        Runs at startup AFTER suspend_recently_active(). Returns the number suspended.
        """
        from gateway.run import _hermes_home
        import json

        path = _hermes_home / self._STUCK_LOOP_FILE
        if not path.exists():
            return 0

        try:
            counts = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return 0

        suspended = 0
        stuck_keys = [k for k, v in counts.items() if v >= self._STUCK_LOOP_THRESHOLD]

        for session_key in stuck_keys:
            try:
                entry = self.session_store._entries.get(session_key)
                if entry and not entry.suspended:
                    entry.suspended = True
                    suspended += 1
                    logger.warning(
                        "Auto-suspended stuck session %s (active across %d "
                        "consecutive restarts — likely a stuck loop)",
                        session_key, counts[session_key],
                    )
            except Exception:
                pass

        if suspended:
            with suppress(Exception):
                self.session_store._save()

        # Clear the file — counters start fresh after suspension
        with suppress(Exception):
            path.unlink(missing_ok=True)

        return suspended

    async def _clear_restart_failure_count(self, session_key: str) -> None:
        """Clear a completed session's restart-failure counter off-loop (atomic_json_write fsyncs)."""
        from gateway.run import _hermes_home, atomic_json_write
        import json

        path = _hermes_home / self._STUCK_LOOP_FILE
        if not path.exists():
            return
        try:
            counts = json.loads(path.read_text(encoding="utf-8"))
            if session_key in counts:
                del counts[session_key]
                if counts:
                    await asyncio.to_thread(atomic_json_write, path, counts, indent=None)
                else:
                    path.unlink(missing_ok=True)
        except Exception:
            pass

    async def _launch_detached_restart_command(self) -> None:
        from gateway.run import _resolve_hermes_bin
        import shutil
        import subprocess

        hermes_cmd = _resolve_hermes_bin()
        if not hermes_cmd:
            logger.error("Could not locate hermes binary for detached /restart")
            return
        if self._detached_restart_helper_started:
            return
        self._detached_restart_helper_started = True

        current_pid = os.getpid()
        restart_after_s = max(float(getattr(self, "_restart_drain_timeout", 0.0) or 0.0) + 5.0, 5.0)

        # On Windows there's no bash/setsid chain — spawn a tiny Python watcher directly via
        # sys.executable instead.
        if sys.platform == "win32":
            import textwrap
            from hermes_cli._subprocess_compat import (
                windows_detach_flags_without_breakaway,
                windows_detach_popen_kwargs,
            )

            cmd_argv = [*hermes_cmd, "gateway", "restart"]
            watcher = textwrap.dedent(
                """
                import os, subprocess, sys, time
                from hermes_cli._subprocess_compat import windows_detach_flags_without_breakaway
                pid = int(sys.argv[1])
                restart_after_s = float(sys.argv[2])
                cmd = sys.argv[3:]
                deadline = time.monotonic() + restart_after_s

                def _alive(p):
                    # On Windows, os.kill(pid, 0) is NOT a no-op — it maps to
                    # GenerateConsoleCtrlEvent(0, pid) (bpo-14484). Use the
                    # Win32 handle-based existence check instead.
                    if os.name == 'nt':
                        import ctypes
                        k32 = ctypes.windll.kernel32
                        k32.OpenProcess.restype = ctypes.c_void_p
                        k32.WaitForSingleObject.restype = ctypes.c_uint
                        k32.GetLastError.restype = ctypes.c_uint
                        h = k32.OpenProcess(0x1000 | 0x100000, False, int(p))
                        if not h:
                            return k32.GetLastError() != 87
                        try:
                            return k32.WaitForSingleObject(h, 0) == 0x102
                        finally:
                            k32.CloseHandle(h)
                    try:
                        os.kill(int(p), 0)
                        return True
                    except ProcessLookupError:
                        return False
                    except PermissionError:
                        return True
                    except OSError:
                        return False

                while time.monotonic() < deadline:
                    if not _alive(pid):
                        break
                    time.sleep(0.2)
                subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=windows_detach_flags_without_breakaway(),
                )
                """
            ).strip()
            from tools.environments.local import build_subprocess_env
            watcher_env = build_subprocess_env(scrub_secrets=False, inherit_profile_home=True)
            # The watcher must not inherit the gateway marker, else `hermes gateway restart` refuses to
            # run (self-restart loop guard) and the gateway stays stopped.
            watcher_env.pop("_HERMES_GATEWAY", None)
            project_root = Path(__file__).resolve().parent.parent
            # Console python under CREATE_NO_WINDOW owns one hidden console inherited by the restart
            # child, so nothing flashes. Do NOT swap in pythonw.exe — a console-less watcher forces
            # every console-subsystem descendant to allocate a visible conhost.
            watcher_python = sys.executable
            venv_dir = Path(watcher_env.get("VIRTUAL_ENV") or project_root / "venv")
            site_packages = venv_dir / "Lib" / "site-packages"
            if site_packages.exists():
                watcher_env["VIRTUAL_ENV"] = str(venv_dir)
                pythonpath = [str(project_root), str(site_packages)]
                if watcher_env.get("PYTHONPATH"):
                    pythonpath.append(watcher_env["PYTHONPATH"])
                watcher_env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(pythonpath))
            watcher_argv = [
                watcher_python,
                "-c",
                watcher,
                str(current_pid),
                str(restart_after_s),
                *cmd_argv,
            ]
            # The watcher must break away from any job object the parent CLI lives in (Desktop
            # wrappers, Windows Terminal, schtasks), else it is reaped when the CLI exits and the
            # gateway never respawns. windows_detach_popen_kwargs() sets CREATE_BREAKAWAY_FROM_JOB,
            # but a job without JOB_OBJECT_LIMIT_BREAKAWAY_OK rejects it (ERROR_ACCESS_DENIED as
            # OSError); retry once without the bit, preserving argv and the scrubbed watcher_env.
            try:
                subprocess.Popen(
                    watcher_argv,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    env=watcher_env,
                    **windows_detach_popen_kwargs(),
                )
            except OSError:
                try:
                    subprocess.Popen(
                        watcher_argv,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        env=watcher_env,
                        creationflags=windows_detach_flags_without_breakaway(),
                    )
                except OSError as exc:
                    # Both spawns failed. Log only the interpreter basename and numeric errno — never
                    # argv, env, watcher source, or str(exc) (may carry a full path) — and return.
                    winerror = getattr(exc, "winerror", None)
                    error_code = winerror if winerror is not None else exc.errno
                    error_field = "winerror" if winerror is not None else "errno"
                    logger.warning(
                        "Detached restart watcher was not started after the "
                        "no-breakaway retry (%s; %s=%r). The gateway will not "
                        "be respawned by this restart attempt.",
                        os.path.basename(watcher_python),
                        error_field,
                        error_code,
                    )
            return

        cmd = " ".join(shlex.quote(part) for part in hermes_cmd)
        shell_cmd = (
            f"deadline=$(( $(date +%s) + {int(restart_after_s)} )); "
            f"while kill -0 {current_pid} 2>/dev/null && [ $(date +%s) -lt $deadline ]; do sleep 0.2; done; "
            f"{cmd} gateway restart"
        )
        # Same marker scrub as the Windows watcher: an inherited _HERMES_GATEWAY=1 makes the CLI's
        # self-restart loop guard refuse silently (DEVNULL), so the gateway stops and never comes back.
        from tools.environments.local import build_subprocess_env
        watcher_env = build_subprocess_env(scrub_secrets=False, inherit_profile_home=True)
        watcher_env.pop("_HERMES_GATEWAY", None)
        setsid_bin = shutil.which("setsid")
        if setsid_bin:
            subprocess.Popen(
                [setsid_bin, "bash", "-lc", shell_cmd],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=watcher_env,
                start_new_session=True,
            )
        else:
            subprocess.Popen(
                ["bash", "-lc", shell_cmd],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=watcher_env,
                start_new_session=True,
            )

    def _wedged_agent_count(self) -> int:
        """Count running chat agents already past the inactivity timeout.

        No activity (API bytes, tool progress) for ``agent.gateway_timeout`` = wedged (the turn reaper's
        threshold). Returns 0 when the timeout is disabled (the after-turn cap still bounds the wait).
        Cron/API-server work has no activity clock and pending sentinels are brand-new, so neither
        counts. Fail-open per agent: an unreadable activity summary means "not wedged".
        """
        from gateway.run import _AGENT_PENDING_SENTINEL, _float_env
        timeout = _float_env("HERMES_AGENT_TIMEOUT", 1800)
        if timeout <= 0:
            return 0
        wedged = 0
        for agent in list((getattr(self, "_running_agents", None) or {}).values()):
            if agent is None or agent is _AGENT_PENDING_SENTINEL:
                continue
            summary_fn = getattr(agent, "get_activity_summary", None)
            if not callable(summary_fn):
                continue
            try:
                summary = summary_fn()
                if not isinstance(summary, dict):
                    continue
                idle = float(summary.get("seconds_since_activity", 0.0))
            except Exception:
                continue
            if idle >= timeout:
                wedged += 1
        return wedged

    def _awaitable_work_count(self) -> int:
        """Active work minus wedged turns — what the restart wait waits on."""
        return max(0, self._active_work_count() - self._wedged_agent_count())

    async def _await_active_work_before_restart(self) -> bool:
        """Wait for in-flight work to finish before entering ``stop()``.

        Calling ``stop()`` immediately would fold the requesting turn into the drain set and
        force-interrupt it at ``restart_drain_timeout``; instead refuse new turns, wait for active
        agents/cron/api work to reach zero, then ``stop()`` an idle gateway. Wedged turns
        (``_wedged_agent_count``) are excluded — restart is the remedy, so ``stop()``'s drain
        interrupts them. Returns True when drained to zero, False when the safety cap elapsed or
        only wedged work remains (caller proceeds to ``stop()``).
        """
        active = self._active_work_count()
        if active <= 0:
            return True

        awaitable = self._awaitable_work_count()
        if awaitable <= 0:
            logger.warning(
                "Restart requested with %d active work unit(s), all wedged "
                "past the inactivity timeout; skipping the after-turn wait "
                "and proceeding to stop()/drain which will interrupt them",
                active,
            )
            return False

        timeout = float(getattr(self, "_restart_after_turn_timeout", 0.0) or 0.0)
        if timeout <= 0:
            logger.info(
                "Restart requested with %d active work unit(s); "
                "restart_after_turn_timeout=0 — entering stop()/drain immediately",
                active,
            )
            return False

        logger.info(
            "Restart requested with %d active work unit(s); "
            "deferring stop() until they finish (cap=%.0fs) so in-flight "
            "turns are not amputated (#77184)",
            active,
            timeout,
        )
        with suppress(Exception):
            self._update_runtime_status("draining")

        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        last_status_at = 0.0
        while self._awaitable_work_count() > 0:
            now = loop.time()
            if now >= deadline:
                logger.warning(
                    "Restart after-turn wait timed out after %.0fs with %d "
                    "still active; proceeding to stop()/drain which may "
                    "interrupt remaining work (#77184)",
                    timeout,
                    self._active_work_count(),
                )
                return False
            if (now - last_status_at) >= 30.0:
                logger.info(
                    "Restart deferred: waiting on %d active work unit(s) "
                    "(%d wedged and excluded; %.0fs remaining before force drain)",
                    self._awaitable_work_count(),
                    self._wedged_agent_count(),
                    deadline - now,
                )
                with suppress(Exception):
                    self._update_runtime_status("draining")
                last_status_at = now
            await asyncio.sleep(0.1)

        if self._active_work_count() > 0:
            logger.warning(
                "Restart deferred wait: %d wedged work unit(s) remain; "
                "proceeding to stop()/drain which will interrupt them",
                self._active_work_count(),
            )
            return False

        logger.info(
            "Restart deferred wait complete — active work drained; "
            "proceeding to stop()"
        )
        return True

    def request_restart(self, *, detached: bool = False, via_service: bool = False) -> bool:
        if self._restart_task_started:
            return False
        self._restart_requested = True
        self._restart_detached = detached
        self._restart_via_service = via_service
        self._restart_task_started = True
        # Refuse new turns while in-flight work finishes. Keep ``_running`` True so adapters stay
        # connected and the active turn can still deliver its final response.
        self._draining = True

        async def _run_restart() -> None:
            await self._await_active_work_before_restart()
            # Launch the detached helper only AFTER the after-turn wait: its drain_timeout+5 deadline
            # covers stop() teardown; earlier it would fire the restart mid-turn.
            if detached:
                try:
                    await self._launch_detached_restart_command()
                except Exception as e:
                    logger.error("Failed to launch detached gateway restart helper: %s", e)
            await asyncio.sleep(0.05)
            await self.stop(restart=True, detached_restart=detached, service_restart=via_service)

        # Do NOT add _run_restart to _background_tasks: _stop_impl cancels every entry there, which
        # would cancel it while awaiting _stop_task and propagate CancelledError into _stop_impl,
        # skipping _shutdown_event.set() / _exit_code = 75. Keep a strong ref in self._restart_task.
        self._restart_task = asyncio.create_task(_run_restart())
        return True

    def _start_systemd_watchdog(self) -> bool:
        """Start sd_notify only after a configured gateway is truly running."""
        if not self._running or self.config.systemd_watchdog_seconds <= 0:
            return False
        if self._systemd_watchdog is not None:
            return True

        from gateway.systemd_notify import SystemdWatchdog

        watchdog = SystemdWatchdog(config_enabled=True)
        if not watchdog.start():
            return False
        self._systemd_watchdog = watchdog
        watchdog.ready("Hermes Gateway running")
        return True

    async def _stop_systemd_watchdog(self) -> None:
        """Stop heartbeats before any potentially long shutdown drain."""
        watchdog = self._systemd_watchdog
        if watchdog is None:
            return
        self._systemd_watchdog = None
        await watchdog.stop()

    @staticmethod
    def _stop_kill_tool_subprocesses(phase: str) -> list:
        """Kill tool subprocesses + tear down terminal envs + browsers.

        Returns the cron job IDs marked interrupted so the caller can notify owners while
        adapters are still up. Called twice: eagerly after a drain timeout forces interrupt
        (reclaim children before systemd SIGKILLs) and as a final catch-all in _stop_impl().
        Best-effort; exceptions swallowed so one subsystem cannot block the rest.
        """
        try:
            from tools.process_registry import process_registry
            _killed = process_registry.kill_all()
            if _killed:
                logger.info(
                    "Shutdown (%s): killed %d tool subprocess(es)",
                    phase, _killed,
                )
        except Exception as _e:
            logger.debug("process_registry.kill_all (%s) error: %s", phase, _e)
        _marked_cron_jobs: list = []
        try:
            # kill_all() is a global sweep, so any cron job dispatched right now lost its tool
            # subprocess; its agent thread may still emit a plausible response from truncated
            # output. Mark the run interrupted so it can never be reported as success.
            from cron.scheduler import mark_running_jobs_interrupted
            _interrupted = _marked_cron_jobs = mark_running_jobs_interrupted(
                f"Gateway shutdown ({phase}) killed the job's tool "
                "subprocess before the run finished."
            )
            if _interrupted:
                logger.warning(
                    "Shutdown (%s): marked %d in-flight cron job(s) interrupted: %s",
                    phase, len(_interrupted), ", ".join(_interrupted),
                )
        except Exception as _e:
            logger.debug("mark_running_jobs_interrupted (%s) error: %s", phase, _e)
        try:
            from tools.async_delegation import interrupt_all as _interrupt_async
            _async_n = _interrupt_async(reason=f"gateway shutdown ({phase})")
            if _async_n:
                logger.info(
                    "Shutdown (%s): interrupted %d background delegation(s)",
                    phase, _async_n,
                )
        except Exception as _e:
            logger.debug("async interrupt_all (%s) error: %s", phase, _e)
        try:
            from tools.terminal_tool import cleanup_all_environments
            cleanup_all_environments()
        except Exception as _e:
            logger.debug("cleanup_all_environments (%s) error: %s", phase, _e)
        try:
            from tools.browser_tool import cleanup_all_browsers
            cleanup_all_browsers()
        except Exception as _e:
            logger.debug("cleanup_all_browsers (%s) error: %s", phase, _e)
        return _marked_cron_jobs

    async def _stop_begin_teardown(
        self, _stop_started_at_box: dict
    ) -> Tuple[Callable[[], int], Callable[[], float]]:
        """Flag teardown, stop room worker/watchdog, notify sessions. Returns the phase clocks."""
        # Shutdown-path tests and third-party runner doubles may only
        # implement the older drain-count surface.
        _deferred_worker_count = getattr(
            self,
            "_active_deferred_agent_worker_count",
            lambda: 0,
        )
        logger.info(
            "Stopping gateway%s...",
            " for restart" if self._restart_requested else "",
        )
        _stop_started_at = time.monotonic()
        _stop_started_at_box["t"] = _stop_started_at

        def _phase_elapsed() -> float:
            return time.monotonic() - _stop_started_at

        self._running = False
        self._clear_plugin_message_injector()
        self._draining = True

        stop_room_worker = getattr(self, "_stop_hosted_room_worker", None)
        if callable(stop_room_worker):
            try:
                stopped = await stop_room_worker(timeout=5.0)
                if not stopped:
                    logger.warning(
                        "Group Chat worker is still settling durable work; "
                        "the next gateway start will recover it"
                    )
            except Exception:
                logger.warning(
                    "Group Chat worker could not stop cleanly; the next gateway "
                    "start will recover durable work",
                    exc_info=True,
                )

        stop_watchdog = getattr(self, "_stop_systemd_watchdog", None)
        if callable(stop_watchdog):
            await stop_watchdog()

        await self._cancel_secondary_profile_reconnect_tasks()

        # Notify all chats with active agents BEFORE draining.
        # Adapters are still connected here, so messages can be sent.
        await self._notify_active_sessions_of_shutdown()
        logger.info(
            "Shutdown phase: notify_active_sessions done at +%.2fs",
            _phase_elapsed(),
        )
        return _deferred_worker_count, _phase_elapsed

    async def _stop_drain_active_work(
        self,
        timeout: float,
        _deferred_worker_count: Callable[[], int],
        _phase_elapsed: Callable[[], float],
    ) -> Tuple[dict, bool, float]:
        """Pre-mark resume_pending, drain agents/cron/API work. Returns (active_agents, timed_out, drain_elapsed)."""
        from gateway.run import _AGENT_PENDING_SENTINEL
        # Pre-mark sessions resume_pending BEFORE the drain wait: if the service manager kills
        # the process mid-drain, the durable marker already lets the next boot recover them.
        _pre_drain_keys: list[str] = []
        for _sk, _agent in list(self._running_agents.items()):
            if _agent is _AGENT_PENDING_SENTINEL:
                continue
            try:
                await self.async_session_store.mark_resume_pending(
                    _sk,
                    "restart_timeout" if self._restart_requested else "shutdown_timeout",
                )
                _pre_drain_keys.append(_sk)
            except Exception as _e:
                logger.debug("pre-drain mark_resume_pending failed for %s: %s", _sk, _e)

        _cron_at_start = self._active_cron_job_count()
        _api_at_start = self._active_api_run_count()
        _deferred_at_start = _deferred_worker_count()
        # In-flight cron work gets its own floor, clamped to the watchdog leash so the extra
        # wait never costs the post-drain cleanup window. getattr-guard: shutdown-path tests
        # drive _stop_impl_body from bare doubles (not GatewayRunner) lacking the class default.
        _cron_drain_cfg = getattr(
            self, "_cron_drain_timeout", DEFAULT_GATEWAY_CRON_DRAIN_TIMEOUT
        )
        _cron_timeout = resolve_cron_drain_budget(
            timeout,
            _cron_drain_cfg,
            watchdog_delay=resolve_shutdown_watchdog_delay(timeout),
            elapsed=_phase_elapsed(),
        )
        if _cron_at_start and _cron_timeout > timeout:
            logger.info(
                "Shutdown drain: %d in-flight cron job(s) — waiting up to "
                "%.0fs for them (cron_drain_timeout=%.0fs, "
                "restart_drain_timeout=%.0fs)",
                _cron_at_start,
                _cron_timeout,
                _cron_drain_cfg,
                timeout,
            )
        _drain_started_at = time.monotonic()
        active_agents, timed_out = await self._drain_active_agents(
            timeout, _cron_timeout
        )
        _drain_elapsed = time.monotonic() - _drain_started_at
        logger.info(
            "Shutdown phase: drain done at +%.2fs (drain took %.2fs, "
            "timed_out=%s, active_at_start=%d, active_now=%d, "
            "cron_at_start=%d, cron_now=%d, "
            "api_at_start=%d, api_now=%d, "
            "deferred_at_start=%d, deferred_now=%d)",
            _phase_elapsed(),
            _drain_elapsed,
            timed_out,
            len(active_agents),
            self._running_agent_count(),
            _cron_at_start,
            self._active_cron_job_count(),
            _api_at_start,
            self._active_api_run_count(),
            _deferred_at_start,
            _deferred_worker_count(),
        )

        if not timed_out:
            # Graceful drain: clear the pre-drain resume_pending markers so sessions that
            # finished during the drain window don't carry a stale flag.
            for _sk in _pre_drain_keys:
                if _sk not in self._running_agents:
                    try:
                        await self.async_session_store.clear_resume_pending(_sk)
                    except Exception as _e:
                        logger.debug(
                            "clear_resume_pending after drain failed for %s: %s",
                            _sk, _e,
                        )
        return active_agents, timed_out, _drain_elapsed

    async def _stop_interrupt_remaining_work(
        self,
        _drain_elapsed: float,
        _deferred_worker_count: Callable[[], int],
        _phase_elapsed: Callable[[], float],
    ) -> None:
        """Drain timed out: mark resume_pending, interrupt, settle, kill tool subprocesses, notify cron."""
        from gateway.run import (
            GatewayRunner,
            _AGENT_PENDING_SENTINEL,
            _INTERRUPT_REASON_GATEWAY_RESTART,
            _INTERRUPT_REASON_GATEWAY_SHUTDOWN,
        )
        logger.warning(
            "Gateway drain timed out after %.1fs with %d active agent(s), "
            "%d in-flight cron job(s), %d api_server run(s), and "
            "%d deferred agent worker(s); "
            "interrupting remaining work.",
            _drain_elapsed,
            self._running_agent_count(),
            self._active_cron_job_count(),
            self._active_api_run_count(),
            _deferred_worker_count(),
        )
        # Mark forcibly-interrupted sessions resume_pending BEFORE interrupting, so the next
        # message on the same session_key auto-resumes instead of being converted to a fresh
        # session by suspend_recently_active(). Genuinely stuck sessions still escalate via
        # ``.restart_failure_counts`` (threshold 3), which sets ``suspended=True`` and wins.
        #
        # Iterate self._running_agents (current), not the drain-start snapshot: sessions that
        # finished cleanly during the drain would otherwise get a stray interruption note.
        # Skip pending sentinels as _interrupt_running_agents() does — nothing has started.
        _resume_reason = (
            "restart_timeout" if self._restart_requested else "shutdown_timeout"
        )
        for _sk, _agent in list(self._running_agents.items()):
            if _agent is _AGENT_PENDING_SENTINEL:
                continue
            try:
                await self.async_session_store.mark_resume_pending(_sk, _resume_reason)
            except Exception as _e:
                logger.debug(
                    "mark_resume_pending failed for %s: %s",
                    _sk, _e,
                )
        self._interrupt_running_agents(
            _INTERRUPT_REASON_GATEWAY_RESTART if self._restart_requested else _INTERRUPT_REASON_GATEWAY_SHUTDOWN
        )
        interrupt_grace_timeout = (
            GatewayRunner._post_interrupt_grace_timeout(self)
        )
        interrupt_deadline = (
            asyncio.get_running_loop().time() + interrupt_grace_timeout
        )
        logger.info(
            "Shutdown phase: allowing %.1fs for interrupted agents to unwind",
            interrupt_grace_timeout,
        )
        # Wait on API-server work too: the interrupt is cooperative, and without this the
        # settle window closes as soon as _running_agents is empty, so an API turn just asked
        # to stop has its tool subprocesses killed below before it can unwind.
        while (
            self._running_agents
            or self._active_api_run_count()
            or _deferred_worker_count()
        ) and asyncio.get_running_loop().time() < interrupt_deadline:
            self._update_runtime_status("draining")
            await asyncio.sleep(0.1)

        # The interrupt fires once, but work can materialize AFTER it: a /v1/runs task enters
        # _active_run_agents only when _create_agent returns, and a _AGENT_PENDING_SENTINEL
        # entry is promoted by track_agent() on its own schedule. Re-signal anything still
        # live so it gets a cooperative interrupt instead of a bare tool-subprocess kill.
        if (
            self._running_agents
            or self._active_api_run_count()
            or _deferred_worker_count()
        ):
            self._interrupt_running_agents(
                _INTERRUPT_REASON_GATEWAY_RESTART
                if self._restart_requested
                else _INTERRUPT_REASON_GATEWAY_SHUTDOWN
            )
            logger.debug(
                "Re-signaled interrupt for work still live at settle-window exit"
            )

        # Kill lingering tool subprocesses NOW, before adapter disconnect / DB close: under
        # systemd (TimeoutStopSec ≈ drain_timeout + headroom) deferring risks the cgroup
        # SIGKILL reaping orphaned children instead of us. The final catch-all still runs.
        _interrupted_cron_jobs = GatewayRunner._stop_kill_tool_subprocesses("post-interrupt")
        logger.info(
            "Shutdown phase: post-interrupt tool kill done at +%.2fs",
            _phase_elapsed(),
        )
        # Last window where the transport is still up. The cron worker whose run we just
        # killed will try to deliver its own "interrupted" notice, but it gets there after
        # the adapter teardown below and the message is lost.
        try:
            await self._notify_interrupted_cron_jobs(_interrupted_cron_jobs)
        except Exception as _e:
            logger.debug("Cron interrupt notification failed: %s", _e)
        logger.info(
            "Shutdown phase: cron interrupt notices done at +%.2fs",
            _phase_elapsed(),
        )

    async def _stop_finalize_agents_and_adapters(
        self, active_agents: dict, _phase_elapsed: Callable[[], float]
    ) -> None:
        """Detached restart launch, agent finalization, idle-cache cleanup, adapter teardown."""
        if self._restart_requested and self._restart_detached:
            try:
                await self._launch_detached_restart_command()
            except Exception as e:
                logger.error("Failed to launch detached gateway restart: %s", e)

        await self._finalize_shutdown_agents(active_agents)

        # Also shut down memory providers on idle cached agents. _finalize_shutdown_agents only
        # handles agents that were mid-turn at drain time; the _agent_cache may still hold idle
        # agents whose MemoryProviders never received on_session_end().
        _cache_lock = getattr(self, "_agent_cache_lock", None)
        _cache = getattr(self, "_agent_cache", None)
        if _cache_lock is not None and _cache is not None:
            with _cache_lock:
                _idle_agents = list(_cache.values())
                _cache.clear()
            for _entry in _idle_agents:
                _agent = (
                    _entry[0] if isinstance(_entry, tuple) else _entry
                )
                # Bounded + off-loop so a wedged memory provider can't hang shutdown forever
                # (this path is why SIGTERM once failed to kill the process).
                await self._cleanup_agent_resources_off_loop(
                    _agent, context="shutdown idle-cache"
                )

        # Completion flush tasks can be sleeping in their fan-in window or blocked in adapter
        # delivery. Cancel and await them while adapters are still alive so every watcher
        # receives a retryable result before platform teardown begins.
        cancel_completion_batches = getattr(
            self, "_cancel_process_completion_batch_tasks", None
        )
        if cancel_completion_batches is not None:
            await cancel_completion_batches()

        for platform, adapter in list(self.adapters.items()):
            await self._bounded_adapter_teardown(adapter, platform)

        # Disconnect secondary-profile adapters (multiplex mode).
        for _prof, _amap in list(getattr(self, "_profile_adapters", {}).items()):
            for platform, adapter in list(_amap.items()):
                await self._bounded_adapter_teardown(
                    adapter, platform, profile=_prof
                )
            _amap.clear()
        if hasattr(self, "_profile_adapters"):
            self._profile_adapters.clear()
        logger.info(
            "Shutdown phase: all adapters disconnected at +%.2fs",
            _phase_elapsed(),
        )

    def _stop_release_runtime_state(self, _phase_elapsed: Callable[[], float]) -> None:
        """Cancel background tasks, flush pending messages, clear per-session state, final tool kill."""
        from gateway.run import GatewayRunner
        for _task in list(self._background_tasks):
            if _task is self._stop_task:
                continue
            if _task is self._restart_task:
                # The restart orchestration task is awaiting _stop_task right now; cancelling it
                # would propagate CancelledError into this _stop_impl and skip
                # _shutdown_event.set() / _exit_code = 75. It self-terminates anyway.
                continue
            _task.cancel()
        self._background_tasks.clear()

        self.adapters.clear()
        for _session_key in list(self._running_agents):
            self._release_running_agent_state(_session_key)
        # Flush pending messages to disk before clearing: under FTS5 corruption the in-memory
        # pending text is the only surviving copy; clearing unflushed loses it permanently.
        try:
            from gateway.shutdown_flush import flush_pending_to_file
            flush_pending_to_file(dict(self._pending_messages), reason="shutdown")
        except Exception:
            pass
        # The FIFO tail lives in SessionState.conversation.queued_events, not the slot dict
        # above — flush it too or every follow-up parked in overflow at restart time is lost.
        try:
            from gateway.shutdown_flush import flush_overflow_to_file
            flush_overflow_to_file(
                {
                    _k: list(_v)
                    for _k, _v in dict(getattr(self, "_queued_events", None) or {}).items()
                    if _v
                },
                reason="shutdown",
            )
        except Exception:
            pass
        # On the real runner these are live SessionState views whose clear() resets one field
        # per session — never a wholesale dict swap, so a concurrent writer on another session
        # can't lose its entry. Test fakes borrowing _stop_impl keep plain dicts.
        self._running_agents.clear()
        self._running_agents_ts.clear()
        if hasattr(self, "_active_session_leases"):
            self._active_session_leases.clear()
        self._pending_messages.clear()
        self._pending_approvals.clear()
        if hasattr(self, '_busy_ack_ts'):
            self._busy_ack_ts.clear()
        self._shutdown_event.set()

        # Global catch-all subprocess kill (safe to repeat): covers the graceful path and
        # anything respawned since the drain-timeout path's post-interrupt kill.
        GatewayRunner._stop_kill_tool_subprocesses("final-cleanup")
        logger.info(
            "Shutdown phase: final-cleanup tool kill done at +%.2fs",
            _phase_elapsed(),
        )

        # Reap the process-global auxiliary-client cache once at the end of teardown. Per-turn
        # cleanup misses clients bound to worker-thread loops that died with their executor
        # (cron ticks); without this sweep async httpx transports accumulate until EMFILE.
        try:
            from agent.auxiliary_client import shutdown_cached_clients
            shutdown_cached_clients()
        except Exception as _e:
            logger.debug("shutdown_cached_clients error: %s", _e)

    def _stop_quiesce_and_close_session_dbs(
        self, timeout: float, _phase_elapsed: Callable[[], float]
    ) -> None:
        """Quiesce the executor, then close SessionDB handles only if no worker is still live."""
        from gateway.run import GatewayRunner, _EXECUTOR_QUIESCE_TIMEOUT
        # Quiesce the gateway thread pool BEFORE the session databases are closed. Running it
        # after the close left two holes: (a) ``_executor_closing`` was still False, so any
        # coroutine reaching ``_run_in_executor_with_context`` minted a fresh pool and ran more
        # blocking DB work against just-closed handles; (b) cancelling ``self._background_tasks``
        # does not stop a ``run_in_executor`` future that already started — the task dies, the
        # worker keeps writing. Either way a write lands after ``SessionDB.close()`` has
        # checkpointed the WAL and let SQLite unlink the sidecar; the late write silently
        # reopens the handle and mints a fresh WAL generation behind that checkpoint, so
        # teardown checkpoints the same file twice from an unaccounted connection
        # (close-time page-write corruption / split WAL generation).
        # The wait is bounded and clamped to what is left of the shutdown watchdog leash
        # (minus a second for the close itself), so a stuck worker can never cost us the
        # post-close cleanup window.
        _exec_quiesce_budget = max(
            0.0,
            min(
                _EXECUTOR_QUIESCE_TIMEOUT,
                resolve_shutdown_watchdog_delay(timeout)
                - _phase_elapsed()
                - 1.0,
            ),
        )
        _exec_live = GatewayRunner._shutdown_executor(
            self, drain_timeout=_exec_quiesce_budget
        )
        if _exec_live:
            # A live worker can still be mid-write against a SessionDB
            # handle. Checkpointing/closing it now is exactly the
            # sequence that produced the wrong-page-number corruption in
            # #101093, so the close path below is skipped entirely
            # rather than raced — the handle is left open for SQLite to
            # recover from its own WAL on the next open, which is a
            # transient "database is locked" on an immediate --replace
            # at worst, not a corrupt file.
            logger.warning(
                "Shutdown phase: %d executor worker(s) still running after "
                "a %.2fs quiesce — skipping the SessionDB close/checkpoint "
                "to avoid racing a live write (#101093); handles are left "
                "open for SQLite to recover on next open",
                _exec_live,
                _exec_quiesce_budget,
            )
        else:
            logger.info(
                "Shutdown phase: executor quiesced at +%.2fs",
                _phase_elapsed(),
            )

            # Close SQLite session DBs so the WAL lock is released; otherwise --replace leaves the old
            # connection holding it until exit and the new gateway gets 'database is locked'.
            # ``_session_db`` is an AsyncSessionDB facade — unwrap; ``session_store`` holds ``_db``.
            _self_db = getattr(self, "_session_db", None)
            _self_db = getattr(_self_db, "_db", _self_db)
            for _db in (_self_db, getattr(getattr(self, "session_store", None), "_db", None)):
                if _db is None or not hasattr(_db, "close"):
                    continue
                try:
                    _db.close()
                except Exception as _e:
                    logger.debug("SessionDB close error: %s", _e)
            # A multiplexed session_store caches one SessionDB per profile; ``_db`` above only covered
            # the root scope. Sweep the rest so secondary WAL locks are released before --replace.
            _sweep = getattr(
                getattr(self, "session_store", None), "close_all_db_handles", None
            )
            if _sweep is not None:
                try:
                    _sweep()
                except Exception as _e:
                    logger.debug("SessionDB handle sweep error: %s", _e)
            # Same sweep for the runner's own per-profile session_search
            # handles (slash commands resolve them under profile scopes).
            try:
                GatewayRunner.close_all_session_db_handles(self)
            except Exception as _e:
                logger.debug("Runner SessionDB handle sweep error: %s", _e)
            # Final sweep: close shared SessionDB instances still held by the process-wide registry
            # (tools, cron, mirror, etc. opened via get_shared_session_db but not released above).
            try:
                from hermes_state import close_shared_session_dbs
                closed = close_shared_session_dbs()
                if closed:
                    logger.debug("Closed %d shared SessionDB instance(s) at shutdown", closed)
            except Exception as _e:
                logger.debug("Shared SessionDB close error: %s", _e)
            logger.info(
                "Shutdown phase: SessionDB close done at +%.2fs",
                _phase_elapsed(),
            )

    def _stop_persist_exit_state(
        self, timed_out: bool, active_agents: dict, _phase_elapsed: Callable[[], float]
    ) -> None:
        """PID/lock release, clean-shutdown marker, restart markers, terminal runtime status."""
        from gateway.run import (
            _hermes_home,
            _planned_restart_notification_path,
            _shutdown_gateway_health_export,
            atomic_json_write,
        )
        from gateway.status import remove_pid_file, release_gateway_runtime_lock
        remove_pid_file()
        release_gateway_runtime_lock()

        # Clean-shutdown marker: suspend_recently_active() need only run after unexpected exits.
        # If the drain timed out and agents were force-interrupted, sessions may be half-finished
        # — skip the marker so the next startup suspends them.
        if not timed_out:
            with suppress(Exception):
                (_hermes_home / ".clean_shutdown").touch()
        else:
            logger.info(
                "Skipping .clean_shutdown marker — drain timed out with "
                "interrupted agents; next startup will suspend recently "
                "active sessions."
            )

        # Stuck-loop detection: the counter increments for sessions active at each restart; at
        # the threshold (3 consecutive) the next startup auto-suspends the session.
        if active_agents:
            self._increment_restart_failure_counts(set(active_agents.keys()))

        if self._restart_requested and self._restart_command_source is None:
            try:
                atomic_json_write(
                    _planned_restart_notification_path(),
                    {
                        "requested_at": time.time(),
                        "via_service": bool(self._restart_via_service),
                        "detached": bool(self._restart_detached),
                    },
                    indent=None,
                )
            except Exception as e:
                logger.debug("Failed to write planned restart notification marker: %s", e)

        if self._restart_requested and self._restart_via_service:
            # Service manager owns restarts: exit 75 + ``RestartForceExitStatus=75`` has systemd
            # replace this process without a second helper racing the unit's stop/start job.
            self._exit_code = GATEWAY_SERVICE_RESTART_EXIT_CODE
            self._exit_reason = self._exit_reason or "Gateway restart requested"

        self._draining = False
        # Persist terminal gateway_state: "stopped" by default, but "running" on an UNEXPECTED
        # external signal (s6 SIGTERM on docker restart, OOM-kill, kill) — container_boot.py
        # only auto-starts gateways last seen "running", so "stopped"/"draining" after a routine
        # recreate would leave channels dark. Operator stops write a planned-stop marker BEFORE
        # signalling and persist "stopped"; a restart also persists "stopped".
        if getattr(self, "_signal_initiated_shutdown", False) and not self._restart_requested:
            logger.info(
                "Gateway stopped by an unexpected signal — persisting "
                "gateway_state=running so container_boot auto-starts on "
                "the next boot (issue #42675)"
            )
            self._update_runtime_status("running", self._exit_reason)
        else:
            self._update_runtime_status("stopped", self._exit_reason)
        _shutdown_gateway_health_export(self)
        logger.info("Gateway stopped (total teardown %.2fs)", _phase_elapsed())

    async def stop(
        self,
        *,
        restart: bool = False,
        detached_restart: bool = False,
        service_restart: bool = False,
    ) -> None:
        """Stop the gateway and disconnect all adapters."""
        from gateway.run import GatewayRunner
        # getattr-guard: shutdown-path tests build bare runners via
        # object.__new__ that lack the liveness-guard machinery.
        _stop_guards = getattr(self, "_stop_loop_liveness_guards", None)
        if callable(_stop_guards):
            _stop_guards()
        if restart:
            self._restart_requested = True
            self._restart_detached = detached_restart
            self._restart_via_service = service_restart
        if self._stop_task is not None:
            await self._stop_task
            return

        async def _stop_impl() -> None:
            # Thread-based shutdown watchdog: asyncio timeouts cannot recover a frozen loop. Arm a
            # plain OS thread at the start of stop(); if teardown never finishes within drain+grace
            # it dumps faulthandler stacks and os._exit so KeepAlive/systemd can revive. Skipped
            # under pytest so stop()-driving tests don't get a delayed hard-exit in the worker.
            _watchdog_done = threading.Event()
            self._shutdown_watchdog_done = _watchdog_done
            _stop_started_at_box: dict[str, float] = {}

            def _shutdown_watchdog_snapshot() -> dict:
                started = _stop_started_at_box.get("t")
                return {
                    "restart_requested": bool(self._restart_requested),
                    "draining": bool(self._draining),
                    "running": bool(self._running),
                    "active_agents": self._running_agent_count(),
                    "active_cron_jobs": self._active_cron_job_count(),
                    "active_api_runs": self._active_api_run_count(),
                    "active_deferred_agent_workers": getattr(
                        self,
                        "_active_deferred_agent_worker_count",
                        lambda: 0,
                    )(),
                    "restart_drain_timeout": self._restart_drain_timeout,
                    "watchdog_delay_s": resolve_shutdown_watchdog_delay(
                        self._restart_drain_timeout
                    ),
                    "phase_elapsed_s": (
                        time.monotonic() - started if started is not None else None
                    ),
                }

            if not os.environ.get("PYTEST_CURRENT_TEST"):
                arm_shutdown_watchdog(
                    resolve_shutdown_watchdog_delay(self._restart_drain_timeout),
                    done_event=_watchdog_done,
                    snapshot_fn=_shutdown_watchdog_snapshot,
                    exit_code=1,
                )

            try:
                await _stop_impl_body(_stop_started_at_box)
            finally:
                _watchdog_done.set()

        async def _stop_impl_body(_stop_started_at_box) -> None:
            _deferred_worker_count, _phase_elapsed = await GatewayRunner._stop_begin_teardown(
                self, _stop_started_at_box
            )

            timeout = self._restart_drain_timeout
            active_agents, timed_out, _drain_elapsed = await GatewayRunner._stop_drain_active_work(
                self, timeout, _deferred_worker_count, _phase_elapsed
            )

            if timed_out:
                await GatewayRunner._stop_interrupt_remaining_work(
                    self, _drain_elapsed, _deferred_worker_count, _phase_elapsed
                )

            await GatewayRunner._stop_finalize_agents_and_adapters(
                self, active_agents, _phase_elapsed
            )
            GatewayRunner._stop_release_runtime_state(self, _phase_elapsed)
            GatewayRunner._stop_quiesce_and_close_session_dbs(self, timeout, _phase_elapsed)
            GatewayRunner._stop_persist_exit_state(self, timed_out, active_agents, _phase_elapsed)

        self._stop_task = asyncio.create_task(_stop_impl())
        await self._stop_task

    async def wait_for_shutdown(self) -> None:
        """Wait for shutdown signal."""
        await self._shutdown_event.wait()
