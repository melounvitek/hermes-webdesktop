"""Goal/heartbeat continuation, post-turn hooks and loop-wakeup watcher methods for GatewayRunner.

Split out of ``gateway/run.py``; bound onto ``GatewayRunner`` via the MRO.
``gateway.run`` internals are imported lazily inside method bodies (import cycle),
so ``patch("gateway.run.X")`` keeps intercepting them at call time.
"""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import suppress
from typing import TYPE_CHECKING, Any

from gateway.platforms.base import MessageEvent, MessageType

if TYPE_CHECKING:  # string annotations only; never imported at runtime (cycle)
    from gateway.run import GatewayRunner, TurnRunner  # noqa: F401

# Log-record parity with the origin module.
logger = logging.getLogger("gateway.run")


class GatewayGoalsMixin:
    """Goal/heartbeat continuation, post-turn hooks and loop-wakeup watcher methods for GatewayRunner."""

    # ── /goal — persistent cross-turn goals (Ralph-style loop) ──────────
    def _goal_max_turns_from_config(self) -> int:
        """Resolve the configured /goal turn budget.

        GatewayRunner.config is a GatewayConfig dataclass, not the full user config, so the
        top-level ``goals`` block is only reachable via hermes_cli.config.load_config().
        """
        try:
            goals_cfg = (
                (self.config or {}).get("goals", {})
                if isinstance(self.config, dict)
                else getattr(self.config, "goals", {}) or {}
            )
            if not goals_cfg:
                from hermes_cli.config import load_config

                goals_cfg = (load_config() or {}).get("goals") or {}
            return int(goals_cfg.get("max_turns", 20) or 20)
        except Exception:
            return 20

    async def _warm_goals_session_db(self, label: str) -> None:
        """Warm the goals SessionDB cache off-loop (best-effort).

        A cold cache runs the state.db init on the loop thread and freezes the loop. The executor
        hop keeps the profile home override alive under multiplex. On failure the caller falls
        back to the bootstrap windows, so a dropped warm-up is a bounded stall, never a crash.
        """
        try:
            from hermes_cli.goals import _get_session_db as _warm_goals_db

            await self._run_in_executor_with_context(_warm_goals_db)
        except Exception as exc:
            logger.warning("%s: session DB warm-up failed: %s", label, exc)

    async def _session_entry_for_manager(self, event: "MessageEvent", label: str):
        """Session entry for a /goal or /heartbeat manager, or None when lookup fails.

        Warms the SessionDB cache first (a cold cache drops the first write while the reply
        claims it was set). Internal events never touch activity, so they don't advance the
        idle/daily reset clock.
        """
        await self._warm_goals_session_db(label)
        try:
            session_entry = await self.async_session_store.get_or_create_session(
                event.source, touch_activity=not bool(getattr(event, "internal", False)),
            )
        except Exception as exc:
            logger.debug("%s: session lookup failed: %s", label, exc)
            return None
        if not (getattr(session_entry, "session_id", None) or ""):
            return None
        return session_entry

    async def _get_goal_manager_for_event(self, event: "MessageEvent"):
        """Return ``(GoalManager, session_entry)`` for this event, or ``(None, None)``."""
        try:
            from hermes_cli.goals import GoalManager
        except Exception as exc:
            logger.debug("goal manager unavailable: %s", exc)
            return None, None
        session_entry = await self._session_entry_for_manager(event, "goal manager")
        if session_entry is None:
            return None, None
        max_turns = self._goal_max_turns_from_config()
        return GoalManager(session_id=session_entry.session_id, default_max_turns=max_turns), session_entry

    async def _get_heartbeat_manager_for_event(self, event: "MessageEvent"):
        """Return ``(HeartbeatManager, session_entry)`` for this event, or ``(None, None)``."""
        try:
            from hermes_cli.heartbeat import HeartbeatManager
        except Exception as exc:
            logger.debug("heartbeat manager unavailable: %s", exc)
            return None, None
        session_entry = await self._session_entry_for_manager(event, "heartbeat manager")
        if session_entry is None:
            return None, None
        return HeartbeatManager(session_id=session_entry.session_id), session_entry

    @staticmethod
    def _synthetic_prompt_event(source: Any, text: str, *, internal: bool = False) -> MessageEvent:
        """Build the TEXT event used to inject a goal/heartbeat/loop prompt into a session."""
        return MessageEvent(text=text, message_type=MessageType.TEXT, source=source, internal=internal)

    def _register_heartbeat_watch(self, quick_key: str, source: Any, session_id: str) -> None:
        """Track a session with an active heartbeat and start the poller.

        The registry maps ``quick_key`` → ``(source, session_id)`` so the poller can rebuild a
        MessageEvent and enqueue via the adapter FIFO. In-memory by design: heartbeat STATE
        survives restarts in SessionDB, but firing resumes only when the user touches /heartbeat
        again (durable schedules belong to cron).
        """
        watch = getattr(self, "_heartbeat_watch", None)
        if watch is None:
            watch = {}
            self._heartbeat_watch = watch
        watch[quick_key] = (source, session_id)
        self._start_heartbeat_poller()

    def _unregister_heartbeat_watch(self, quick_key: str) -> None:
        watch = getattr(self, "_heartbeat_watch", None)
        if watch:
            watch.pop(quick_key, None)

    async def _heartbeat_poll_once(self, watch: dict) -> None:
        """One heartbeat poll pass: enqueue every due prompt of a non-busy watched session."""
        # Warm the cache off-loop once per poll: this only covers the degraded path where the
        # /heartbeat command's own warm-up failed.
        await self._warm_goals_session_db("heartbeat poll")
        for quick_key, (source, session_id) in list(watch.items()):
            try:
                # Busy sessions coalesce their tick to the next idle poll.
                if quick_key in self._running_agents:
                    continue
                from hermes_cli.heartbeat import HeartbeatManager

                mgr = HeartbeatManager(session_id=session_id)
                if not mgr.has_heartbeat():
                    watch.pop(quick_key, None)
                    continue
                prompt = mgr.due_prompt()
                if not prompt:
                    continue
                adapter = self._adapter_for_source(source)
                if adapter is None:
                    continue
                self._enqueue_fifo(quick_key, self._synthetic_prompt_event(source, prompt), adapter)
            except Exception as exc:
                logger.debug("heartbeat poll for %s failed: %s", quick_key, exc)

    def _start_heartbeat_poller(self) -> None:
        """Start the single gateway-wide heartbeat poll task (idempotent)."""
        existing = getattr(self, "_heartbeat_poll_task", None)
        if existing is not None and not existing.done():
            return

        from hermes_cli.heartbeat import POLL_SECONDS

        async def _poll_loop():
            while True:
                await asyncio.sleep(POLL_SECONDS)
                watch = getattr(self, "_heartbeat_watch", None)
                if watch:
                    await self._heartbeat_poll_once(watch)

        try:
            task = asyncio.create_task(_poll_loop())
            self._heartbeat_poll_task = task
            # PERMANENT once started (infinite loop) — same as a _spawn_supervised watcher. Tag it
            # so _scale_to_zero_has_live_background_work() doesn't treat the gateway as busy forever.
            task._hermes_supervised_watcher = True  # type: ignore[attr-defined]
            _bg = getattr(self, "_background_tasks", None)
            if _bg is not None:
                _bg.add(task)
                task.add_done_callback(_bg.discard)
        except Exception:
            logger.debug("Failed to start heartbeat poller", exc_info=True)

    async def _send_goal_status_notice(self, source: Any, message: str) -> None:
        """Send a /goal judge status line back to the originating chat/thread."""
        adapter = self._adapter_for_source(source)
        if not adapter:
            logger.debug("goal continuation: no adapter for %s", getattr(source, "platform", None))
            return
        try:
            metadata = self._thread_metadata_for_source(source)
        except Exception:
            metadata = None
        result = await adapter.send(source.chat_id, message, metadata=metadata)
        if result is not None and not getattr(result, "success", True):
            logger.warning(
                "goal continuation: status send failed: %s", getattr(result, "error", "unknown error"),
            )

    async def _defer_goal_status_notice_after_delivery(self, source: Any, message: str) -> None:
        """Send a /goal status line after the main response is delivered.

        The adapter sends the agent response after this caller returns, so for reading order the
        status must follow that send: use the adapter's one-shot post-delivery callback when
        available, else fall back to direct awaited delivery rather than dropping the notice.
        """
        adapter = self._adapter_for_source(source)
        if not adapter:
            logger.debug("goal continuation: no adapter for %s", getattr(source, "platform", None))
            return

        async def _deliver() -> None:
            try:
                await self._send_goal_status_notice(source, message)
            except Exception as exc:
                logger.warning("goal continuation: status send failed: %s", exc, exc_info=True)

        try:
            session_key = self._session_key_for_source(source)
        except Exception:
            session_key = None

        if session_key and hasattr(adapter, "register_post_delivery_callback"):
            try:
                generation = None
                active = getattr(adapter, "_active_sessions", {}).get(session_key)
                if active is not None:
                    generation = getattr(active, "_hermes_run_generation", None)
                adapter.register_post_delivery_callback(session_key, _deliver, generation=generation)
                return
            except Exception as exc:
                logger.debug("goal continuation: post-delivery callback registration failed: %s", exc)

        await _deliver()

    async def _post_turn_goal_continuation(
        self, *, session_entry: Any, source: Any, final_response: str,
    ) -> None:
        """Run the goal judge after a gateway turn and, if still active, enqueue a continuation.

        Called at turn boundary AFTER delivery. Uses the adapter's pending-message/FIFO machinery
        so a simultaneous real user message is handled by the same queue and takes priority.
        """
        try:
            from hermes_cli.goals import GoalManager
        except Exception as exc:
            logger.debug("goal continuation: goals module unavailable: %s", exc)
            return

        sid = getattr(session_entry, "session_id", None) or ""
        if not sid:
            return

        max_turns = self._goal_max_turns_from_config()
        # Cold cache at the turn boundary: a slow state.db init on the loop thread can drop the
        # goal read and silently end the goal loop.
        await self._warm_goals_session_db("goal continuation")

        mgr = GoalManager(session_id=sid, default_max_turns=max_turns)
        if not mgr.is_active():
            return

        try:
            from hermes_cli.goals import gather_background_processes as _gather_bg
            _bg_procs = _gather_bg()
        except Exception:
            _bg_procs = None

        # evaluate_after_turn → judge_goal() is a synchronous aux-LLM HTTP call; on the loop thread
        # it blocks Discord heartbeats 10-40 s. _run_in_executor_with_context (not bare
        # run_in_executor) carries the profile secret scope / aux runtime contextvars, without which
        # aux credential resolution fails under multiplexing.
        decision = await self._run_in_executor_with_context(
            lambda: mgr.evaluate_after_turn(
                final_response or "", user_initiated=True, background_processes=_bg_procs,
            ),
        )
        msg = decision.get("message") or ""

        # Status line is deferred until the adapter has delivered the visible final response,
        # otherwise "✓ Goal achieved" would show before the answer itself.
        if msg and source is not None:
            await self._defer_goal_status_notice_after_delivery(source, msg)

        if not decision.get("should_continue"):
            return

        prompt = decision.get("continuation_prompt") or ""
        if not prompt or source is None:
            return

        # Enqueue via the adapter's FIFO so a user message already in flight preempts naturally.
        try:
            adapter = self._adapter_for_source(source)
            _quick_key = self._session_key_for_source(source)
            if adapter and _quick_key:
                self._enqueue_fifo(_quick_key, self._synthetic_prompt_event(source, prompt), adapter)
        except Exception as exc:
            logger.debug("goal continuation: enqueue failed: %s", exc)

    async def _run_post_turn_hooks(
        self, *, agent_result: Any, source: Any, is_internal: bool, event: Any = None,
    ) -> None:
        """Run goal and loop bookkeeping after an agent turn returns."""
        final_text = self._final_text_for_post_turn_hooks(agent_result, event)

        try:
            session_entry = await self.async_session_store.get_or_create_session(
                source, touch_activity=not is_internal,
            )
        except Exception as exc:
            logger.debug("post-turn session resolution failed: %s", exc)
            return

        # Empty interrupted/errored responses must not drive /goal, but an in-flight /loop tick
        # still needs to be released and rescheduled.
        if final_text.strip():
            try:
                await self._post_turn_goal_continuation(
                    session_entry=session_entry, source=source, final_response=final_text,
                )
            except Exception as exc:
                logger.debug("goal continuation hook failed: %s", exc)
        try:
            await self._post_turn_loop_completion(
                session_entry=session_entry, source=source, final_response=final_text,
            )
        except Exception as exc:
            logger.debug("loop completion hook failed: %s", exc)

    @staticmethod
    def _final_text_for_post_turn_hooks(agent_result, event=None) -> str:
        """Text for /goal and /loop after a gateway turn.

        Streamed turns return None from _handle_message_with_agent (already_sent); the delivered
        reply is stashed on the event so those hooks still see it.
        """
        text = ""
        if isinstance(agent_result, dict):
            text = str(agent_result.get("final_response") or "")
        elif isinstance(agent_result, str):
            text = agent_result
        if text.strip():
            return text
        streamed = getattr(event, "_streamed_final_response", None)
        if isinstance(streamed, str) and streamed.strip():
            return streamed
        return text

    async def _post_turn_loop_completion(
        self, *, session_entry: Any, source: Any, final_response: str,
    ) -> None:
        """Complete a /loop wakeup tick after a gateway turn.

        No-op unless the session has a loop whose tick is in flight (``awaiting_response`` — set
        when the wakeup was injected). Applies the LOOP_COMPLETE marker / --until judge / caps
        and schedules the next tick; the idle wakeup watcher fires it when due.
        """
        try:
            from hermes_cli.loops import LoopManager
        except Exception as exc:
            logger.debug("loop completion: loops module unavailable: %s", exc)
            return

        sid = getattr(session_entry, "session_id", None) or ""
        if not sid:
            return

        # Cold cache at the turn boundary can drop the tick-completion write (see /goal above).
        await self._warm_goals_session_db("loop completion")

        mgr = LoopManager(session_id=sid)
        state = mgr.state
        if state is None or not state.awaiting_response:
            return

        # The --until judge is a sync aux-LLM call — keep it off the event loop.
        decision = await asyncio.get_running_loop().run_in_executor(
            None, mgr.complete_tick, final_response or ""
        )
        msg = decision.get("message") or ""
        if msg and source is not None:
            await self._defer_goal_status_notice_after_delivery(source, msg)

    async def _loop_wakeup_fire_one(self, sid: str, state: Any, now: float, warned_no_route: set) -> None:
        """Inject one due /loop wakeup into its session, applying every deferral rule."""
        from hermes_cli.loops import LoopManager, goal_blocks_loop_tick

        if state.awaiting_response or now < state.next_due_at:
            return
        route = state.route or {}
        platform_name = route.get("platform", "")
        chat_id = route.get("chat_id", "")
        if not platform_name or not chat_id:
            return  # CLI / TUI-owned loop — their own schedulers drive it.
        adapter = next((a for p, a in self.adapters.items() if p.value == platform_name), None)
        if adapter is None:
            if sid not in warned_no_route:
                warned_no_route.add(sid)
                logger.debug(
                    "loop wakeup: no adapter for platform %r (session %s)", platform_name, sid,
                )
            return

        # Build the source + session key to check business.
        source = self._build_process_event_source({
            "session_key": "",
            "platform": platform_name,
            "chat_id": chat_id,
            "chat_type": route.get("chat_type", ""),
            "thread_id": route.get("thread_id", ""),
            "user_id": route.get("user_id", ""),
            "user_name": route.get("user_name", ""),
        })
        if source is None:
            return
        try:
            session_key = self._session_key_for_source(source)
        except Exception:
            session_key = None
        if session_key and session_key in self._running_agents:
            return  # busy — stays due, next scan retries
        if goal_blocks_loop_tick(sid):
            return

        mgr = LoopManager(session_id=sid)
        if not mgr.is_due(now):
            return
        wakeup = mgr.fire_tick()
        if not wakeup:
            return
        try:
            logger.info(
                "loop wakeup #%s — injecting for %s chat=%s thread=%s",
                mgr.state.ticks_fired if mgr.state else "?",
                platform_name, source.chat_id, source.thread_id,
            )
            await adapter.handle_message(self._synthetic_prompt_event(source, wakeup, internal=True))
            # Slash-command loops dispatch through the command path and never hit the post-turn
            # completion hook — complete the tick immediately (caps + scheduling).
            if wakeup.lstrip().startswith("/"):
                mgr.complete_tick("")
        except Exception as exc:
            logger.warning("loop wakeup injection failed for %s: %s", sid, exc)
            with suppress(Exception):
                mgr.abandon_tick()

    async def _loop_wakeup_watcher(self, interval: float = 15.0) -> None:
        """Fire due /loop wakeups for idle gateway sessions.

        The gateway has no per-session scheduler thread, so a coarse ticker scans persisted loops
        (SessionDB ``loop:*`` rows) and injects the wakeup prompt into each due session's chat
        via the same synthetic-message path used by watch notifications. Deferrals: session
        currently running a turn → skip (the FIFO would race the live turn); active non-parked
        /goal → skip (goal owns the idle boundary); no routing metadata → skip with a one-time
        warning (CLI/TUI loops carry no route).
        """
        await asyncio.sleep(5)  # let platforms finish connecting
        warned_no_route: set = set()
        while self._running:
            try:
                from hermes_cli.loops import list_active_loops

                # Warm the cache off-loop once per scan: the scan reads every persisted loop, so a
                # cold cache would run the state.db init on the loop thread before the first read.
                await self._warm_goals_session_db("loop wakeup")

                now = time.time()
                for sid, state in list_active_loops():
                    await self._loop_wakeup_fire_one(sid, state, now, warned_no_route)
            except Exception as exc:
                logger.debug("loop wakeup watcher error: %s", exc)
            await asyncio.sleep(interval)
