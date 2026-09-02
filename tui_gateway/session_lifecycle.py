"""Session lifecycle: active-session slot leases, finalize/teardown/close, turn interrupt, WS-orphan reap scheduling, transport-scoped close.

Bodies are rebound onto server.py's globals at install time (see
method_ctx.bind_module), so they reference server.py globals bare.
"""

from __future__ import annotations

import contextlib

from .method_ctx import HandlerRegistry, bind_module

_registry = HandlerRegistry()


def _notify_session_boundary(
    event_type: str, session_id: str | None, platform: str | None = None
) -> None:
    """Fire session lifecycle hooks with CLI parity."""
    try:
        from hermes_cli.lifecycle import finalize_session, invoke_hook

        if event_type == "on_session_finalize":
            finalize_session(
                session_id=session_id,
                platform=_resolve_agent_platform(platform),
            )
        else:
            invoke_hook(
                event_type,
                session_id=session_id,
                platform=_resolve_agent_platform(platform),
            )
    except Exception:
        pass


_SESSION_OWNERSHIP_UNAVAILABLE = (
    "Hermes could not safely reserve this session. Try again."
)

_AUTOMATIC_SESSION_END_REASONS = frozenset({
    "ws_orphan_reap",
    "ws_disconnect",
    "idle_timeout",
    "lru_evict",
    "tui_shutdown",
})


def _claim_active_session_slot(
    session_key: str,
    *,
    live_session_id: str,
    surface: str = "tui",
    profile_home: str | Path | None = None,
) -> tuple[Any, str | None]:
    track_liveness = str(surface or "").strip().lower() == "desktop"
    try:
        from hermes_cli.active_sessions import try_acquire_active_session

        return try_acquire_active_session(
            session_id=session_key,
            surface=surface,
            config=_load_cfg(),
            metadata={"live_session_id": live_session_id},
            registry_home=profile_home,
            track_liveness=track_liveness,
        )
    except Exception as exc:
        logger.warning("Failed to claim active session slot: %s", exc)
        # Fail CLOSED regardless of surface: per-session exclusivity is a
        # correctness guarantee (see PER_SESSION_EXCLUSIVE_SUBMIT), and a
        # claim that errors out has NOT proven the session is unowned.
        # Proceeding without a lease here is the silent double-writer hole
        # flagged in the #94595 review (blocker 2).
        return (None, _SESSION_OWNERSHIP_UNAVAILABLE)


def _ensure_active_session_slot(sid: str, session: dict) -> str | None:
    """Claim this session's cap slot on its first real turn; None when ok.

    session.create / session.resume deliberately do NOT claim one. Every
    desktop tile paint, background reconnect-resume and abandoned draft opens a
    session just to paint a composer, and a slot held by one of those is
    invisible everywhere: an unprompted draft has no DB row, and the sidebar
    filters it out with min_messages=1. Idle desktop tabs therefore silently
    starved the messaging gateway, which shares this cap — five parked tabs on
    a websocket-flappy host locked a Discord bot out of a 5-slot cap while
    running no agents at all. Claiming on the first turn mirrors the lazy
    contract _ensure_session_db_row already uses for the row itself, and keeps
    the invariant that anything holding a slot is something the user can see.
    """
    if session.get("active_session_lease") is not None:
        return None
    lease, limit_message = _claim_active_session_slot(
        str(session.get("session_key") or ""),
        live_session_id=sid,
        surface=_session_source(session),
        profile_home=session.get("profile_home"),
    )
    if limit_message is not None:
        return limit_message
    session["active_session_lease"] = lease
    return None


def _release_active_session_lease(lease) -> bool:
    if lease is None:
        return True
    attempts = 3 if getattr(lease, "track_liveness", False) else 1
    for attempt in range(attempts):
        try:
            lease.release()
            break
        except Exception:
            if attempt + 1 >= attempts:
                logger.warning("Failed to release active session slot", exc_info=True)
                return False
            time.sleep(0.05 * (attempt + 1))
    released = getattr(lease, "released", True)
    enabled = getattr(lease, "enabled", True)
    return bool(released or not enabled)


def _release_active_session_slot(session: dict | None) -> bool:
    if not session:
        return True
    lease = session.get("active_session_lease")
    if _release_active_session_lease(lease):
        if session.get("active_session_lease") is lease:
            session.pop("active_session_lease", None)
        return True
    return False


@contextlib.contextmanager
def _other_runtime_lease_guard(session_id: str, session: dict):
    """Release this runtime and lock sibling ownership through the DB write."""
    lease = session.get("active_session_lease")
    try:
        from hermes_cli.active_sessions import (
            active_session_liveness_guard,
            release_active_session_liveness_guard,
        )
    except Exception as exc:
        logger.warning(
            "Failed to load active session ownership guard; preserving session %s: %s",
            session_id,
            exc,
        )
        yield True
        return

    last_error: Exception | None = None
    stack = contextlib.ExitStack()
    for attempt in range(3):
        try:
            if lease is not None and getattr(lease, "enabled", False):
                guard = release_active_session_liveness_guard(lease, session_id)
            else:
                guard = active_session_liveness_guard(
                    session_id, registry_home=session.get("profile_home")
                )
            active = stack.enter_context(guard)
            break
        except Exception as exc:
            stack.close()
            stack = contextlib.ExitStack()
            last_error = exc
            if attempt < 2:
                time.sleep(0.05 * (attempt + 1))
    else:
        logger.warning(
            "Failed to inspect active session leases; preserving session %s: %s",
            session_id,
            last_error,
        )
        yield True
        return
    try:
        yield active
    finally:
        stack.close()
        if (
            lease is not None
            and getattr(lease, "released", False)
            and session.get("active_session_lease") is lease
        ):
            session.pop("active_session_lease", None)


def _transfer_active_session_slot(
    sid: str,
    session: dict,
    *,
    new_session_id: str,
) -> bool:
    if not new_session_id:
        return False
    lease = session.get("active_session_lease")
    if lease is None:
        return True
    try:
        from hermes_cli.active_sessions import transfer_active_session

        if transfer_active_session(
            lease,
            session_id=new_session_id,
            metadata={"live_session_id": sid},
        ):
            return True
    except Exception:
        logger.debug("Failed to transfer active session slot", exc_info=True)

    if getattr(lease, "track_liveness", False):
        return False

    # Fallback: the in-place transfer could not move the lease (entry pruned /
    # pid-check transiently failed). Reserve the new slot BEFORE releasing the
    # old one, so a concurrent gateway at the session cap cannot grab the freed
    # slot in a release-then-reacquire window and leave this session with no
    # lease at all (#49041 review). If the reserve fails, KEEP the old lease.
    new_lease, limit_message = _claim_active_session_slot(
        new_session_id,
        live_session_id=sid,
        surface=_session_source(session),
        profile_home=session.get("profile_home"),
    )
    if new_lease is not None:
        old_lease = session.pop("active_session_lease", None)
        if old_lease is not None:
            try:
                old_lease.release()
            except Exception:
                logger.debug("Failed to release stale active session slot", exc_info=True)
        session["active_session_lease"] = new_lease
        return True
    # Reserve failed — retain the existing lease rather than dropping it.
    if limit_message:
        logger.warning(
            "Compression session lease re-anchor failed (kept old lease): "
            "sid=%s new_session_id=%s reason=%s",
            sid,
            new_session_id,
            limit_message,
        )
    return False


# Session sources the TUI/desktop backend must never end in state.db: the
# messaging gateway owns those sessions' lifecycle — the TUI is only a viewer
# (a resume of a Telegram/Discord/... session).  Ending one creates the
# #60609 Groundhog Day routing loop (see _finalize_session).  Sources the
# TUI backend itself creates ("tui", plus whatever a client passes as its
# own ``source``) and the CLI's own sessions are NOT gateway-owned.
_NON_GATEWAY_SOURCES = frozenset({
    "", "tui", "cli", "webui", "desktop", "cron", "kanban", "subagent", "test",
    "local", "acp", "webhook", "api_server", "msgraph_webhook",
})


def _is_gateway_owned_source(source: str) -> bool:
    """True when ``source`` names a messaging-gateway platform whose session
    lifecycle belongs to the gateway, not to this TUI backend.

    Structural rather than a hardcoded platform list: any source that
    resolves to a known gateway ``Platform`` (built-in enum member OR a
    registered platform plugin, via ``Platform._missing_``) counts, so new
    platforms are covered automatically.  Local/self-owned sources are
    excluded explicitly — ``local``/``webhook``/``api_server`` are Platform
    members but their sessions are not owned by a remote chat surface that
    routes by session_key, so reaping them is safe and keeps /resume clean.
    """
    src = (source or "").strip().lower()
    if src in _NON_GATEWAY_SOURCES:
        return False
    try:
        from gateway.config import Platform

        Platform(src)  # raises ValueError for arbitrary non-platform strings
        return True
    except Exception:
        return False


def _finalize_session(session: dict | None, end_reason: str = "tui_close") -> None:
    """Best-effort finalize hook + memory commit for a session.

    Fires ``on_session_end`` plugin hook and attempts to persist any
    unflushed messages before closing the session.  This mirrors the
    CLI's exit-path behaviour and prevents data loss when the TUI is
    force-quit (double Ctrl‑C, terminal‑close, SIGHUP) while the agent
    is mid‑turn.
    """
    if not session or session.get("_finalized"):
        return
    session["_finalized"] = True
    history_ready = session.get("resume_history_ready")
    if history_ready is not None and not history_ready.is_set():
        session["resume_history_error"] = "session resume cancelled"
        history_ready.set()
    _desktop_automatic_cleanup = bool(
        end_reason in _AUTOMATIC_SESSION_END_REASONS
        and _session_source(session).strip().lower() == "desktop"
    )
    # Automatic Desktop cleanup removes its lease inside the lock-held lifecycle
    # guard below. Explicit close and non-Desktop paths keep force/end semantics.
    if not _desktop_automatic_cleanup:
        _release_active_session_slot(session)
    stop_event = session.get("_notif_stop")
    if stop_event is not None:
        stop_event.set()

    agent = session.get("agent")
    lock = session.get("history_lock")
    if lock is not None:
        with lock:
            history = list(session.get("history", []))
    else:
        history = list(session.get("history", []))

    # ── Persist unflushed messages to SQLite ──────────────────────────
    # Flush ``agent._session_messages`` via ``_persist_session``'s marker-based
    # dedup (same contract as the gateway-shutdown flush, #13121). Do NOT pass
    # ``conversation_history``: ``session["history"]`` and ``_session_messages``
    # alias the SAME list once a turn completes, so passing it made
    # ``_flush_messages_to_session_db`` treat every message as already-durable
    # and skip it — a data-loss bug when finalize is the sole persist path after
    # a WS disconnect/restart (e.g. the in-turn flush hit a transient SQLite
    # failure). Markers persist the genuinely-unflushed tail without duplicating
    # durable rows (including a resumed-but-not-run session's already-in-DB
    # transcript, which stays in ``session["history"]`` only).
    if agent is not None and hasattr(agent, "_persist_session"):
        snapshot = getattr(agent, "_session_messages", None)
        if snapshot:
            try:
                agent._persist_session(snapshot)
            except Exception:
                pass

    # ── Plugin hook: on_session_end ────────────────────────────────────
    # Signals every plugin that the session is closing, with
    # interrupted=True so crash‑recovery plugins can flush buffers,
    # persist state, or close connections before the gateway exits.
    # Mirrors cli.py's atexit handler that fires the same hook when
    # the user Ctrl‑C's mid‑turn.
    if agent is not None:
        try:
            from hermes_cli.lifecycle import invoke_hook

            invoke_hook(
                "on_session_end",
                session_id=getattr(agent, "session_id", None)
                or session.get("session_key", ""),
                completed=False,
                interrupted=True,
                model=getattr(agent, "model", "unknown"),
                platform=getattr(agent, "platform", None) or "tui",
            )
        except Exception:
            pass

    if agent is not None and history and hasattr(agent, "commit_memory_session"):
        try:
            agent.commit_memory_session(history)
        except Exception:
            pass

    session_key = session.get("session_key")
    session_id = getattr(agent, "session_id", None) or session_key
    _notify_session_boundary("on_session_finalize", session_id, _session_source(session))

    # Mark session ended in DB so it doesn't linger as a ghost row in /resume.
    # Use session_id (from agent.session_id) not session_key — after compression,
    # session_key may be stale (the ended parent) while session_id is the live
    # continuation. Fix for #20001.
    if _desktop_automatic_cleanup and not session_id:
        _release_active_session_slot(session)
    _lifecycle_guard = (
        _other_runtime_lease_guard(session_id, session)
        if _desktop_automatic_cleanup and session_id
        else contextlib.nullcontext(False)
    )
    with _lifecycle_guard as _other_runtime_owns_lifecycle:
        _tui_owns_lifecycle = not _other_runtime_owns_lifecycle
        if _other_runtime_owns_lifecycle:
            logger.info(
                "Preserving session %s during %s: another backend owns an active lease",
                session_id,
                end_reason,
            )
        if session_id:
            try:
                # End the row in the *session's* profile state.db (app-global
                # remote mode), not the launch profile's shared handle.
                with _session_db(session) as db:
                    if db is not None:
                        # Don't end gateway-originated sessions — the gateway owns
                        # their lifecycle.  The TUI is a viewer, not the owner.
                        # Ending a gateway session in state.db triggers a Groundhog
                        # Day routing loop: the gateway's #54878 self-heal detects
                        # the stale entry, recovers to the parent session, context
                        # compression splits back to the reaped child, and the cycle
                        # repeats on every inbound message.  (#60609)
                        row = db.get_session(session_id)
                        source = (row or {}).get("source", "")
                        if _is_gateway_owned_source(source):
                            _tui_owns_lifecycle = False
                        elif _tui_owns_lifecycle:
                            db.end_session(session_id, end_reason)
            except Exception:
                pass

    # A session's in-flight async delegations end WITH the session (#55578):
    # once nobody owns the return address, a still-running background subagent
    # can only burn tokens and park an orphaned completion on the shared
    # queue. Always interrupt delegations commissioned by THIS live UI session
    # (its sid); additionally interrupt by durable session_key, but only when
    # the TUI owns the lifecycle — closing a viewer tab on a live gateway
    # session must not kill the gateway's own background work.
    try:
        from tools.async_delegation import interrupt_for_session

        _own_sid = str(session.get("_sid") or "")
        if not _own_sid:
            try:
                with _sessions_lock:
                    for _cand_sid, _cand in _sessions.items():
                        if _cand is session:
                            _own_sid = _cand_sid
                            break
            except Exception:
                _own_sid = ""
        interrupt_for_session(
            session_key=str(session_key or "") if _tui_owns_lifecycle else "",
            origin_ui_session_id=_own_sid,
            reason=end_reason,
        )
    except Exception:
        pass

    # Close the slash-worker subprocess as part of finalize itself, not just
    # in the callers. Defense-in-depth: every session-end path goes through
    # _finalize_session (it's the single ``_finalized``-guarded chokepoint), so
    # folding worker cleanup in here means a future code path that calls
    # _finalize_session directly — without the surrounding _teardown_session /
    # _shutdown_sessions worker.close() — can't reintroduce the #38095 leak.
    # Idempotent: _SlashWorker.close() is poll()-guarded, so the explicit
    # close() still in those callers is harmless.
    try:
        worker = session.get("slash_worker")
        if worker:
            worker.close()
    except Exception:
        pass


# End reasons where the BACKEND reclaimed a session the client never asked to
# close: the idle-TTL reaper, the LRU cap, and the WS-orphan reap. A client
# holding that live session id gets no signal today — its next prompt fails
# against an id the backend has already forgotten, which reads as the session
# silently vanishing rather than being reclaimed. ``tui_close`` and friends are
# deliberately absent: the client initiated those and already knows.
_RECLAIM_END_REASONS = frozenset({"idle_timeout", "lru_evict", "ws_orphan_reap"})


def _announce_session_reclaimed(session: dict, end_reason: str) -> None:
    """Tell connected clients a session was reclaimed out from under them.

    Broadcast rather than session-targeted: the reap paths run on background
    timer threads with no contextvar binding, and the WS-orphan case has by
    definition lost its own transport — ``_emit`` would bottom out on stdio and
    the peer that owns the session would never see it. Best-effort; a failed
    notify must never break teardown.
    """
    if end_reason not in _RECLAIM_END_REASONS:
        return
    try:
        _broadcast_global_event(
            "session.reclaimed",
            {
                "session_id": str(session.get("_sid") or ""),
                "stored_session_id": str(session.get("session_key") or ""),
                "reason": end_reason,
            },
        )
    except Exception:
        logger.debug("session.reclaimed broadcast failed", exc_info=True)


def _teardown_session(session: dict | None, *, end_reason: str = "tui_close") -> None:
    """Fully tear down a session: finalize, unregister, close agent + worker.

    Shared by ``session.close`` and the orphaned-WS-session reaper. The
    slash-worker subprocess is closed inside ``_finalize_session`` (the single
    finalize chokepoint); this still unregisters the approval notifier and
    closes the in-process agent. Idempotent: the ``_finalized`` guard in
    ``_finalize_session`` and the ``poll()`` guard in ``_SlashWorker.close``
    make repeat calls harmless.
    """
    if not session:
        return
    _finalize_session(session, end_reason=end_reason)
    _announce_session_reclaimed(session, end_reason)
    try:
        from tools.approval import unregister_gateway_notify

        if key := session.get("session_key"):
            unregister_gateway_notify(key)
    except Exception:
        pass
    try:
        agent = session.get("agent")
        if agent is not None and hasattr(agent, "close"):
            agent.close()
    except Exception:
        pass
    # NOTE: the slash-worker is closed inside _finalize_session (the single
    # _finalized-guarded chokepoint that main folded it into), exactly once.
    # We deliberately do NOT re-close it here — _teardown_session's job beyond
    # finalize is unregistering the notifier and closing the in-process agent.


def _attach_worker(sid: str, session: dict, worker) -> None:
    """Store worker on session iff sid still maps to it, else close it — a
    concurrent teardown already popped the session and would orphan the
    worker. Closes the create/close race at every slash-worker spawn site."""
    with _sessions_lock:
        if _sessions.get(sid) is session:
            session["slash_worker"] = worker
            return
    worker.close()


def _pop_session_by_id(sid: str) -> dict | None:
    """Atomically detach one live session from the registry.

    Detaching is the ownership claim for teardown: once the record is no
    longer in ``_sessions``, a concurrent close/reaper becomes a no-op.  Keep
    this operation separate from ``_teardown_session`` because finalization can
    flush SQLite state, invoke plugins, commit memory, interrupt delegations,
    and close agents/workers.  None of that slow external work belongs under
    the global ``_session_resume_lock``.
    """
    with _sessions_lock:
        session = _sessions.get(sid)
        if session is not None:
            session["_closing"] = True
            _sessions.pop(sid, None)
    if session is None:
        return None
    # The session is already out of _sessions here, so downstream teardown
    # (e.g. _finalize_session's per-session async-delegation interrupt) can't
    # recover its live id by scanning the dict — stamp it on the record.
    session["_sid"] = sid
    return session


def _teardown_popped_session(
    session: dict | None, *, end_reason: str = "tui_close"
) -> bool:
    """Finish a close after the caller has atomically detached the session."""
    if session is None:
        return False
    run_thread = session.get("_run_thread")
    if (
        end_reason != "tui_shutdown"
        and run_thread is not None
        and run_thread is not threading.current_thread()
    ):
        try:
            if run_thread.is_alive():
                run_thread.join(timeout=_TURN_SETTLE_BEFORE_CLOSE_SECONDS)
            if run_thread.is_alive():
                logger.warning(
                    "session turn thread still alive after %.1fs teardown grace",
                    _TURN_SETTLE_BEFORE_CLOSE_SECONDS,
                )
        except Exception:
            logger.debug("failed waiting for session turn thread", exc_info=True)
    _teardown_session(session, end_reason=end_reason)
    return True


def _close_session_by_id(
    sid: str,
    *,
    end_reason: str = "tui_close",
    predicate: Callable[[dict], bool] | None = None,
) -> bool:
    """Single idempotent teardown funnel for callers needing no resume race.

    Resume-sensitive callers first pop under ``_session_resume_lock`` and then
    call ``_teardown_popped_session`` after releasing it.  Other reapers can use
    this convenience wrapper directly.  The pop remains the single atomic
    ownership claim, so concurrent/repeat close attempts stay harmless.

    Automatic reapers can pass ``predicate`` to revalidate under
    ``_sessions_lock`` immediately before the ownership claim. This prevents a
    stale scan result from closing a session that reattached or gained active
    delegated work before teardown.
    """
    if predicate is None:
        session = _pop_session_by_id(sid)
    else:
        with _sessions_lock:
            current = _sessions.get(sid)
            if current is None or not predicate(current):
                return False
            session = _pop_session_by_id(sid)
    return _teardown_popped_session(session, end_reason=end_reason)


def _ws_session_is_detached(session: dict | None) -> bool:
    """True if a live session is still bound to the disconnected-WS sentinel."""
    return bool(
        session
        and not session.get("_finalized")
        and session.get("transport") is _detached_ws_transport
    )


def _ws_session_is_orphaned(session: dict | None) -> bool:
    """True if a WS session has no live transport and no in-flight turn.

    After ``handle_ws`` detaches a disconnected client it points the session at
    ``_detached_ws_transport``. A session left on that transport (and not
    mid-turn) is genuinely orphaned and safe to reap.
    """
    return bool(
        _ws_session_is_detached(session)
        and session is not None
        and not session.get("running")
    )


def _interrupt_session_turn(
    sid: str, session: dict, *, request_id: str | None = None
) -> bool:
    """Apply the shared ``session.interrupt`` contract to one claimed session.

    Returns whether the interrupt used the compute-host control channel. The WS
    orphan reaper calls this same helper after its reconnect grace expires, so a
    dead client gets the same partial-history and queued-prompt semantics as an
    explicit user interrupt.
    """
    use_compute_host = _session_uses_compute_host(session)
    should_interrupt = bool(session.get("running"))
    run_thread_alive = False

    if use_compute_host:
        # The host owns the live turn. Parent `running` is only a mirror and
        # can lag behind a blocked interactive tool (a clarify parked on its
        # Event keeps the host turn alive after the parent flag went stale),
        # so let the host decide whether there is work to interrupt.
        # Gate on `_compute_host_active` too: `_session_uses_compute_host`
        # is also true for lazy sessions that never ran a hosted turn, and
        # `HostSupervisor.interrupt()` calls `start()` — forwarding
        # unconditionally would spawn a compute-host child just to deliver
        # an interrupt no session ever submitted work to.
        if should_interrupt or session.get("_compute_host_active"):
            _get_compute_host_supervisor().interrupt(sid, request_id=request_id)
    else:
        run_thread = session.get("_run_thread")
        run_thread_alive = run_thread is not None and run_thread.is_alive()

    with session["history_lock"]:
        session["_turn_cancel_requested"] = True
        session["queued_prompt"] = None
        session.pop("queued_prompts", None)
        session["_queued_prompt_generation"] = int(
            session.get("_queued_prompt_generation", 0)
        ) + 1

    if not use_compute_host:
        if should_interrupt:
            from agent.interrupt_compat import request_hard_interrupt

            request_hard_interrupt(session.get("agent"))
        if not run_thread_alive:
            with session["history_lock"]:
                if session.get("running"):
                    session["running"] = False
                    _clear_inflight_turn(session)

    _clear_pending(sid)
    try:
        from tools.approval import resolve_gateway_approval

        resolve_gateway_approval(session["session_key"], "deny", resolve_all=True)
    except Exception:
        pass
    return use_compute_host


def _session_owns_durable_lifecycle(session_id: str | None) -> bool:
    """Whether this TUI/desktop session may end its durable DB row by key."""
    if not session_id:
        return True
    try:
        db = _get_db()
        if db is None:
            return True
        # Don't end gateway-originated sessions — the gateway owns their
        # lifecycle. The TUI is only a viewer there (#60609).
        row = db.get_session(session_id)
        source = (row or {}).get("source", "")
        return not _is_gateway_owned_source(source)
    except Exception:
        return True


def _session_async_delegation_selectors(
    session: dict | None, *, sid_hint: str = ""
) -> tuple[str, str]:
    """Ownership selectors for async background work tied to one UI session."""
    if not session:
        return "", ""
    own_sid = str(sid_hint or session.get("_sid") or "")
    if not own_sid:
        try:
            with _sessions_lock:
                for _cand_sid, _cand in _sessions.items():
                    if _cand is session:
                        own_sid = _cand_sid
                        break
        except Exception:
            own_sid = ""
    agent = session.get("agent")
    session_key = str(session.get("session_key") or "")
    session_id = getattr(agent, "session_id", None) or session_key
    owned_session_key = session_key if _session_owns_durable_lifecycle(session_id) else ""
    return own_sid, owned_session_key


def _session_has_active_delegations(sid: str, session: dict | None = None) -> bool:
    """True when UI session ``sid`` still owns live background work.

    Matches by the live UI sid AND — when the TUI owns the durable lifecycle
    (never for gateway-viewer tabs, #60609) — by the durable session_key, so a
    delegation dispatched from an earlier tab of the same resumed session still
    keeps it alive.
    """
    if session is None:
        with _sessions_lock:
            session = _sessions.get(sid)
    own_sid, owned_session_key = _session_async_delegation_selectors(
        session, sid_hint=sid
    )
    if not own_sid and not owned_session_key:
        return False
    try:
        from tools.async_delegation import has_live_for_session

        return has_live_for_session(
            session_key=owned_session_key,
            origin_ui_session_id=own_sid,
        )
    except Exception:
        logger.debug(
            "Failed to query active delegations for UI session %s",
            sid,
            exc_info=True,
        )
        # A transient registry/import failure must not turn into destructive
        # cleanup. Conservatively keep the detached session and let the next
        # orphan timer retry the lookup.
        return True


# One pending WS-orphan reap Timer per live sid. Registered by
# _schedule_ws_orphan_reap, popped when its _reap fires, and cancelled by
# _cancel_ws_orphan_reap from every resume/reuse/transport-rebind path. Without
# this cancellation the reap could fire against an already-reattached session,
# broadcast session.reclaimed, and trigger the client's auto-re-resume — a
# reap->broadcast->resume feedback storm. Guarded by _sessions_lock.
_pending_ws_reaps: dict[str, threading.Timer] = {}


def _cancel_ws_orphan_reap(sid: str) -> None:
    """Cancel a pending WS-orphan reap for ``sid`` (client came back).

    Called from every path that re-binds a live transport onto the session:
    the session.resume fast-path reuse, _claim_or_reuse_live winners, and the
    _live_session_payload transport rebind. Cancelling here (rather than
    relying on the reap's own orphan re-check) removes the window where a
    fired-but-not-yet-run Timer races the resume, and stops dead Timers from
    accumulating for sessions that reconnect frequently.
    """
    with _sessions_lock:
        timer = _pending_ws_reaps.pop(sid, None)
    if timer is not None:
        try:
            timer.cancel()
        except Exception:
            pass


def _ws_orphan_turn_activity_is_fresh(session: dict) -> bool:
    """Whether a detached RUNNING turn's activity clock is still fresh.

    Reuses the agent's existing activity summary (``_touch_activity`` is
    stamped by API waits, stream tokens, and tool heartbeats — the same
    clock the turn-liveness watchdog samples; see agent/turn_liveness.py).
    Fresh means the WS-orphan reaper must NOT interrupt the turn yet
    (#98028/#100325): deliberate client absence (closed laptop, backgrounded
    mobile app, desktop update/relaunch) keeps healthy work running detached.

    Conservative fallbacks preserve the wedged-turn safety net: a disabled
    threshold (<= 0), a missing/opaque agent, an unreadable summary, or a
    never-stamped clock all report NOT fresh, i.e. eligible for the
    interrupt-at-grace path exactly as before.
    """
    if _WS_ORPHAN_ACTIVITY_STALE_S <= 0:
        return False
    agent = session.get("agent")
    summary_fn = getattr(agent, "get_activity_summary", None)
    if not callable(summary_fn):
        return False
    try:
        elapsed = summary_fn().get("seconds_since_activity")
        return elapsed is not None and float(elapsed) < _WS_ORPHAN_ACTIVITY_STALE_S
    except Exception:
        return False


def _schedule_ws_orphan_reap(sid: str, *, delay_s: float | None = None) -> None:
    """After a grace window, reap session ``sid`` iff it's still orphaned.

    Called from the WS-disconnect path. The grace window lets a transient
    reconnect (or a ``session.resume`` that reattaches the transport) cancel
    the reap by re-binding a live transport. Disabled when the grace is 0.
    """
    if _WS_ORPHAN_REAP_GRACE_S <= 0:
        return

    def _reap() -> None:
        # Serialize the orphan re-check against session.resume (which re-binds a
        # live transport under _session_resume_lock and would make this session
        # non-orphaned). Claim teardown by popping under both lifecycle locks,
        # then release the global resume lock before the slow finalization work.
        # The dict mutation still happens under _sessions_lock — consistent
        # with every other _sessions mutator
        # (#39591: _reap previously popped under _session_resume_lock, giving no
        # mutual exclusion against _init_session / _close_session_by_id, which
        # guard with _sessions_lock). _sessions_lock is an RLock and the global
        # ordering is always resume_lock -> sessions_lock, so nesting is safe.
        reschedule_delay = None
        interrupt_session = None
        session = None
        with _session_resume_lock:
            # This Timer is running: drop its registration so a concurrent
            # _cancel_ws_orphan_reap doesn't cancel a dead Timer object while
            # a rescheduled one (registered below) is the live owner.
            with _sessions_lock:
                _pending_ws_reaps.pop(sid, None)
            current = _sessions.get(sid)
            if current is None or not _ws_session_is_detached(current):
                return
            if _session_has_active_delegations(sid, current):
                reschedule_delay = _WS_ORPHAN_REAP_GRACE_S
            elif current.get("running"):
                if not current.get(
                    "_client_gone_interrupt_requested"
                ) and _ws_orphan_turn_activity_is_fresh(current):
                    # Client-absent but actively producing (#98028/#100325):
                    # the turn keeps running detached (the sentinel transport
                    # already buffers emits) and the reaper re-checks each
                    # grace interval. Only a turn whose activity clock has
                    # gone stale — genuinely wedged, the case the interrupt
                    # was added for — falls through to the interrupt below.
                    logger.debug(
                        "client_gone sid=%s action=defer (turn activity "
                        "fresh; stale threshold %.0fs)",
                        sid,
                        _WS_ORPHAN_ACTIVITY_STALE_S,
                    )
                    reschedule_delay = _WS_ORPHAN_REAP_GRACE_S
                else:
                    # Mid-turn detached sessions must never drop the single
                    # Timer (#85578): after the reconnect grace the turn is
                    # interrupted once, then the reap keeps polling until the
                    # normal turn-finalization path settles.
                    polls = int(current.get("_client_gone_interrupt_polls") or 0) + 1
                    current["_client_gone_interrupt_polls"] = polls
                    if polls > _WS_ORPHAN_INTERRUPT_REAP_MAX_POLLS:
                        # The interrupted turn never settled inside the budget
                        # — force-reap rather than parking the session + a
                        # timer chain forever. Loud by design: this only fires
                        # when a turn is genuinely stuck past interrupt.
                        logger.error(
                            "client_gone sid=%s: turn did not settle after %d "
                            "interrupt polls (%.0fs) — force-reaping detached "
                            "session",
                            sid, polls - 1,
                            (polls - 1) * _WS_ORPHAN_INTERRUPT_REAP_POLL_S,
                        )
                        session = _pop_session_by_id(sid)
                    else:
                        if not current.get("_client_gone_interrupt_requested"):
                            current["_client_gone_interrupt_requested"] = True
                            interrupt_session = current
                        reschedule_delay = _WS_ORPHAN_INTERRUPT_REAP_POLL_S
            else:
                session = _pop_session_by_id(sid)

        if interrupt_session is not None:
            try:
                isolated = _interrupt_session_turn(
                    sid,
                    interrupt_session,
                    request_id=f"client-gone-{sid}",
                )
                logger.info(
                    "client_gone sid=%s action=interrupt turn_isolation=%s",
                    sid,
                    isolated,
                )
            except Exception:
                logger.exception("client_gone interrupt failed sid=%s", sid)
                with _sessions_lock:
                    if _sessions.get(sid) is interrupt_session:
                        interrupt_session.pop(
                            "_client_gone_interrupt_requested", None
                        )

        if reschedule_delay is not None:
            _schedule_ws_orphan_reap(sid, delay_s=reschedule_delay)
            return
        if session is not None and session.get(
            "_client_gone_interrupt_requested"
        ):
            logger.info("client_gone sid=%s action=reap", sid)
        _teardown_popped_session(session, end_reason="ws_orphan_reap")

    timer = threading.Timer(
        _WS_ORPHAN_REAP_GRACE_S if delay_s is None else max(0.0, delay_s),
        _reap,
    )
    timer.daemon = True
    with _sessions_lock:
        prior = _pending_ws_reaps.pop(sid, None)
        _pending_ws_reaps[sid] = timer
    if prior is not None:
        try:
            prior.cancel()
        except Exception:
            pass
    timer.start()


def _close_sessions_for_transport(
    transport, *, end_reason: str = "ws_disconnect"
) -> tuple[int, int]:
    """On transport disconnect, reap the sessions that opted into
    close_on_disconnect (sidecar/dashboard) immediately and re-point the rest
    at the detached transport so later emits don't hit a dead socket.

    Non-flagged detached sessions are handed to the grace-windowed WS-orphan
    reaper (``_schedule_ws_orphan_reap``): a quick reconnect / session.resume
    that re-binds a live transport cancels the reap, otherwise the orphan is
    torn down through the same idempotent ``_teardown_session`` path. This is
    the single WS-disconnect teardown entry point — there is no second
    independent reap loop in ``handle_ws``.

    Returns ``(reaped, detached)`` counts for disconnect-path observability."""
    with _sessions_lock:
        owned = [(sid, s) for sid, s in _sessions.items() if s.get("transport") is transport]
    reaped = 0
    detached = 0
    for sid, session in owned:
        claimed_for_teardown = None
        should_schedule_reap = False
        # A session.resume fast-path rebinds its live session while holding
        # _session_resume_lock. Take that lock before re-checking the snapshot
        # so a reconnect cannot move the transport between this check and the
        # close/detach ownership claim. Keep the slow teardown below both locks.
        with _session_resume_lock:
            with _sessions_lock:
                current = _sessions.get(sid)
                if current is not session:
                    continue
                if current.get("transport") is not transport:
                    # The reconnect owns this session now. Drop only the old
                    # viewer registration; it must not affect the new owner.
                    viewers = current.get("viewers")
                    if viewers:
                        viewers.pop(transport, None)
                    continue
                if current.get("close_on_disconnect"):
                    claimed_for_teardown = _pop_session_by_id(sid)
                else:
                    # Point detached sessions at the drop sentinel (NOT real
                    # stdio) so _ws_session_is_orphaned recognizes them and
                    # the grace-reap can actually fire; a standalone
                    # `hermes --tui` keeps real _stdio. UNLESS another window
                    # still shows the session: multi-window pop-outs all
                    # register as viewers, so on disconnect re-bind the
                    # session to the most recent surviving viewer instead of
                    # stranding the original window on the sentinel (#83716).
                    viewers = current.get("viewers")
                    if viewers:
                        viewers.pop(transport, None)
                    remaining = [
                        (ts, viewer_transport)
                        for viewer_transport, ts in (viewers or {}).items()
                        if (
                            viewer_transport is not transport
                            and not _transport_is_dead(viewer_transport)
                        )
                    ]
                    if remaining:
                        remaining.sort(key=lambda item: item[0])
                        current["transport"] = remaining[-1][1]
                    else:
                        current["transport"] = _detached_ws_transport
                        current.pop("_client_gone_interrupt_requested", None)
                        should_schedule_reap = True
        if claimed_for_teardown is not None:
            if _teardown_popped_session(claimed_for_teardown, end_reason=end_reason):
                reaped += 1
        elif should_schedule_reap:
            detached += 1
            try:
                _schedule_ws_orphan_reap(sid)
            except Exception:
                pass
    return reaped, detached


def register(server) -> None:
    """Publish this module's helpers + handlers onto ``server``, rebound to its globals."""
    bind_module(globals(), server, skip=("_",))
