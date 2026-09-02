"""Per-session notification poller: kanban/loop/delegation events routed to the owning session, desktop UI wiring, HUD surface note.

Bodies are rebound onto server.py's globals at install time (see
method_ctx.bind_module), so they reference server.py globals bare.
"""

from __future__ import annotations


from .method_ctx import HandlerRegistry, bind_module

_registry = HandlerRegistry()


def _notification_event_belongs_elsewhere(sid: str, session: dict, evt: dict) -> bool:
    """True if ``evt`` is owned by a *different* live session.

    Background completions carry the ``session_key`` of the session that started
    the work. Async delegation completions from the desktop also carry
    ``origin_ui_session_id``: the live TUI tab/window that commissioned them.
    Since all desktop sessions share one process-wide completion queue, each
    poller must skip events it doesn't own so a detached result surfaces in the
    launching session, not whichever poller happened to dequeue first.
    """
    evt_ui_sid = str(evt.get("origin_ui_session_id") or "")
    if evt_ui_sid:
        if evt_ui_sid == str(sid or "") and not session.get("_finalized"):
            return False
        try:
            with _sessions_lock:
                owner_live = evt_ui_sid in _sessions and not _sessions[evt_ui_sid].get("_finalized")
        except Exception:
            owner_live = False
        if owner_live:
            return True
        # If the exact UI tab is gone, fall through to durable session_key
        # routing. That avoids wrong-session delivery while still allowing a
        # resumed continuation with the same durable key/lineage to claim it.

    evt_key = str(evt.get("session_key") or "")
    if not evt_key:
        return False

    current_keys = {
        str(session.get("session_key") or ""),
        _session_lookup_key(session, fallback=sid),
    }

    # Compression can rotate AIAgent.session_id while the detached child is
    # still running. Resolve the event's original key to its continuation tip so
    # an event captured before or after compression still maps to the same live
    # desktop session instead of becoming an orphan that any poller may consume.
    resolved_key = evt_key
    try:
        db = _get_db()
        if db is not None:
            resolved_key = db.resolve_resume_session_id(evt_key) or evt_key
    except Exception:
        resolved_key = evt_key

    # If the key has a live continuation, prefer that continuation over the
    # compressed parent. Otherwise a stale parent tab could consume the event
    # before the real current conversation sees it.
    if resolved_key != evt_key:
        if resolved_key in current_keys:
            return False
        try:
            with _sessions_lock:
                continuation_live = any(
                    not s.get("_finalized")
                    and (
                        str(s.get("session_key") or "") == resolved_key
                        or _session_lookup_key(s, fallback="") == resolved_key
                    )
                    for s in _sessions.values()
                )
        except Exception:
            continuation_live = False
        if continuation_live:
            return True

    if evt_key in current_keys:
        return False

    try:
        with _sessions_lock:
            snapshot = list(_sessions.values())
    except Exception:
        # If we can't safely enumerate live sessions, fail open so we don't
        # crash the poller thread or drop the event.
        return False

    return any(
        s is not session
        and not s.get("_finalized")
        and (
            str(s.get("session_key") or "") in {evt_key, resolved_key}
            or _session_lookup_key(s, fallback="") in {evt_key, resolved_key}
        )
        for s in snapshot
    )


def _session_owns_notification_event(sid: str, session: dict, evt: dict) -> bool:
    """True iff *this* session PROVABLY owns ``evt``.

    Positive ownership — the mirror of ``_notification_event_belongs_elsewhere``
    minus its orphan-adoption fallback. An event owns-matches when its
    ``origin_ui_session_id`` is this live session, or its ``session_key``
    (raw or resolved through the compression chain) matches this session's
    key/lineage. Used as the fail-closed gate for every addressed notification:
    "not provably elsewhere" is NOT good enough to inject a payload into this
    chat (#55578).
    """
    if session.get("_finalized"):
        return False
    if str(evt.get("origin_ui_session_id") or "") == str(sid or ""):
        return True
    evt_key = str(evt.get("session_key") or "")
    if not evt_key:
        return False
    current_keys = {
        str(session.get("session_key") or ""),
        _session_lookup_key(session, fallback=sid),
    }
    if evt_key in current_keys:
        return True
    try:
        db = _get_db()
        resolved_key = (
            db.resolve_resume_session_id(evt_key) if db is not None else evt_key
        ) or evt_key
    except Exception:
        resolved_key = evt_key
    return resolved_key in current_keys


def _notification_event_requires_owner(evt: dict) -> bool:
    """Whether ``evt`` must be positively claimed before TUI delivery."""
    return evt.get("type") == "async_delegation" or bool(
        str(evt.get("origin_ui_session_id") or "")
        or str(evt.get("session_key") or "")
    )


def _notification_event_dedup_key(evt: dict) -> tuple:
    """Return the UI-emission identity for a process notification event.

    Completion events are terminal notifications for a background process, so
    they remain one-shot per process session. Watch-match events are not
    terminal: a single background process can legitimately match the same or
    different patterns many times, so include event-specific content to avoid
    suppressing later distinct matches from the same process.
    """
    evt_type = evt.get("type", "completion")
    evt_sid = evt.get("session_id", "")
    if evt_type == "watch_match":
        return (
            evt_sid,
            evt_type,
            evt.get("command", ""),
            evt.get("pattern", ""),
            evt.get("output", ""),
            evt.get("suppressed", 0),
            evt.get("message_id", ""),
        )
    if evt_type.startswith("watch_overflow_") or evt_type == "watch_disabled":
        return (
            evt_sid,
            evt_type,
            evt.get("command", ""),
            evt.get("message", ""),
            evt.get("suppressed", 0),
        )
    if evt_type == "async_delegation":
        # Async-delegation completions have no process session_id; without
        # this the fallthrough keys every one as ("", "async_delegation")
        # and the second completion's status update is suppressed forever.
        return (evt.get("delegation_id", ""), evt_type)
    return (evt_sid, evt_type)


# Mirror gateway/kanban_watchers.py TERMINAL_KINDS: claim silent kinds too so
# the cursor advances past them and they can't wedge a later completed/blocked
# event behind an unclaimed row.
_KANBAN_NOTIFY_KINDS = (
    "completed", "blocked", "gave_up", "crashed", "timed_out",
    "status", "archived", "unblocked",
)
_KANBAN_SILENT_KINDS = frozenset({"archived", "unblocked"})
_KANBAN_POLL_SECONDS = 5.0
_LOOP_POLL_SECONDS = 5.0


def _maybe_fire_tui_loop_tick(sid: str, session: dict) -> None:
    """Fire a due /loop wakeup for an idle TUI/Desktop/dashboard session.

    Called from the per-session notification poller thread on a coarse
    cadence. Claims the session under history_lock (running=True) before
    dispatching so a racing user prompt wins cleanly. The post-turn hook
    in the turn dispatcher completes the tick.
    """
    try:
        from hermes_cli.loops import LoopManager, goal_blocks_loop_tick
    except Exception:
        return

    sid_key = session.get("session_key") or ""
    if not sid_key:
        return
    mgr = LoopManager(session_id=sid_key)
    if not mgr.is_due():
        return
    if goal_blocks_loop_tick(sid_key):
        return

    with session["history_lock"]:
        if session.get("running"):
            return  # busy — stays due, next poll retries
        session["running"] = True

    wakeup = mgr.fire_tick()
    if not wakeup:
        with session["history_lock"]:
            session["running"] = False
        return

    tick_no = mgr.state.ticks_fired if mgr.state else "?"
    rid = f"__loop__{int(time.time() * 1000)}"
    try:
        _emit(
            "status.update",
            sid,
            {"kind": "loop", "text": f"↻ /loop wakeup #{tick_no} firing…"},
        )
        if wakeup.lstrip().startswith("/"):
            # Slash-command loop: route through the slash pipeline instead of
            # the model. No model reply to evaluate — complete immediately.
            with session["history_lock"]:
                session["running"] = False
            try:
                parts = wakeup.lstrip()[1:].split(None, 1)
                resp = _methods["command.dispatch"](
                    rid,
                    {
                        "name": parts[0] if parts else "",
                        "arg": parts[1] if len(parts) > 1 else "",
                        "session_id": sid,
                    },
                )
                payload = (resp or {}).get("result") or {}
                out = str(payload.get("output") or "").strip()
                if out:
                    _emit("status.update", sid, {"kind": "loop", "text": out})
                if payload.get("type") == "send" and payload.get("message"):
                    # The command resolves to a prompt (skill command etc.) —
                    # run it as a normal turn; the post-turn hook completes
                    # the tick.
                    with session["history_lock"]:
                        if session.get("running"):
                            mgr.abandon_tick()
                            return
                        session["running"] = True
                    _emit("message.start", sid)
                    _run_prompt_submit(rid, sid, session, payload["message"])
                    return
            except Exception:
                pass
            decision = mgr.complete_tick("")
            if decision.get("message"):
                _emit("status.update", sid, {"kind": "loop", "text": decision["message"]})
            return
        _emit("message.start", sid)
        _run_prompt_submit(rid, sid, session, wakeup)
    except Exception as exc:
        print(
            f"[tui_gateway] loop wakeup dispatch failed: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        with session["history_lock"]:
            session["running"] = False
        try:
            mgr.abandon_tick()
        except Exception:
            pass


def _format_kanban_event_text(sub: dict, task, ev, board_slug: str) -> Optional[str]:
    """Single-line notification text for one kanban event.

    Wording mirrors the gateway notifier (gateway/kanban_watchers.py) so a
    task completion reads the same in the TUI as it does on Telegram.
    Returns None for kinds that are claimed but intentionally silent.
    """
    kind = getattr(ev, "kind", "")
    if not kind or kind in _KANBAN_SILENT_KINDS:
        return None
    task_id = sub.get("task_id", "")
    title = (getattr(task, "title", None) or task_id)[:120]
    board_tag = f"[{board_slug}] " if board_slug else ""
    who = getattr(task, "assignee", None) or ""
    tag = f"@{who} " if who else ""
    payload = getattr(ev, "payload", None) or {}
    if kind == "completed":
        handoff = ""
        summary = payload.get("summary")
        if summary:
            lines = str(summary).strip().splitlines()
            handoff = f"\n{lines[0][:200]}" if lines else ""
        elif getattr(task, "result", None):
            lines = str(task.result).strip().splitlines()
            handoff = f"\n{lines[0][:160]}" if lines else ""
        return f"✔ {board_tag}{tag}Kanban {task_id} done — {title}{handoff}"
    if kind == "blocked":
        reason = f": {str(payload.get('reason'))[:160]}" if payload.get("reason") else ""
        return f"⏸ {board_tag}{tag}Kanban {task_id} blocked{reason}"
    if kind == "gave_up":
        err = f"\n{str(payload.get('error'))[:200]}" if payload.get("error") else ""
        return f"✖ {board_tag}{tag}Kanban {task_id} gave up after repeated spawn failures{err}"
    if kind == "crashed":
        return f"✖ {board_tag}{tag}Kanban {task_id} worker crashed (pid gone); dispatcher will retry"
    if kind == "timed_out":
        limit = 0
        try:
            limit = int(payload.get("limit_seconds") or 0)
        except (TypeError, ValueError):
            pass
        return f"⏱ {board_tag}{tag}Kanban {task_id} timed out (max_runtime={limit}s); will retry"
    if kind == "status":
        return f"🔄 {board_tag}{tag}Kanban {task_id} → {payload.get('status') or ''}"
    return None


def _collect_kanban_notifications(session: dict) -> list:
    """Claim unseen terminal kanban events for this TUI session's subscriptions.

    ``kanban_create`` auto-subscribes TUI/desktop sessions with
    ``platform="tui"`` and ``chat_id=HERMES_SESSION_KEY`` (see
    tools/kanban_tools.py ``_maybe_auto_subscribe``). The gateway notifier
    can't deliver those — there is no "tui" messaging adapter — so this
    poller is the delivery path for them (issue #59890). Uses the same
    atomic cursor-claim (``claim_unseen_events_for_sub``) as the gateway
    notifier, so a subscription is delivered exactly once even if a gateway
    and a TUI poll the same board DB.

    Returns the list of formatted notification texts (may be empty).
    """
    session_key = str(session.get("session_key") or "")
    if not session_key or session.get("_finalized"):
        return []
    try:
        from hermes_cli import kanban_db as _kb
    except Exception:
        return []
    texts: list = []
    try:
        boards = _kb.list_boards(include_archived=False)
    except Exception:
        try:
            boards = [_kb.read_board_metadata(_kb.DEFAULT_BOARD)]
        except Exception:
            return []
    # Poll each resolved DB path once — multiple slugs can point at the same
    # DB when HERMES_KANBAN_DB pins the board path (same guard as the gateway
    # notifier).
    seen_db_paths: set = set()
    for board_meta in boards:
        slug = (board_meta or {}).get("slug") or _kb.DEFAULT_BOARD
        db_path = (board_meta or {}).get("db_path")
        try:
            resolved = (
                str(Path(db_path).expanduser().resolve())
                if db_path else str(_kb.kanban_db_path(slug).resolve())
            )
        except Exception:
            resolved = f"slug:{slug}"
        if resolved in seen_db_paths:
            continue
        seen_db_paths.add(resolved)
        # A poller runs per live TUI/Desktop session. Avoid opening this board
        # writable unless it has a subscription owned by this exact session;
        # subscriptions for gateways or other sessions are not actionable here.
        try:
            if _kb.count_notify_subs(
                board=slug,
                platform="tui",
                chat_id=session_key,
            ) == 0:
                continue
        except Exception:
            # Preserve delivery if the read-only probe cannot inspect a
            # locked, corrupt, or otherwise unusual database.
            pass
        try:
            conn = _kb.connect(board=slug)
        except Exception:
            continue
        try:
            try:
                subs = _kb.list_notify_subs(conn)
            except Exception:
                continue
            for sub in subs:
                if (sub.get("platform") or "").lower() != "tui":
                    continue
                if sub.get("chat_id") != session_key:
                    continue
                _old, _new, events = _kb.claim_unseen_events_for_sub(
                    conn,
                    task_id=sub["task_id"],
                    platform=sub["platform"],
                    chat_id=sub["chat_id"],
                    thread_id=sub.get("thread_id") or "",
                    kinds=_KANBAN_NOTIFY_KINDS,
                )
                if not events:
                    continue
                task = _kb.get_task(conn, sub["task_id"])
                for ev in events:
                    text = _format_kanban_event_text(sub, task, ev, slug)
                    if text:
                        texts.append(text)
                # Unsubscribe only on archive. ``done`` is reversible in
                # review/controller flows, so retaining the subscription lets
                # a later reopen notify the same originating TUI/Desktop
                # session. The claimed cursor prevents historical replay.
                if task and getattr(task, "status", "") == "archived":
                    try:
                        _kb.remove_notify_sub(
                            conn,
                            task_id=sub["task_id"],
                            platform=sub["platform"],
                            chat_id=sub["chat_id"],
                            thread_id=sub.get("thread_id") or "",
                        )
                    except Exception:
                        pass
        finally:
            conn.close()
    return texts


def _notification_poller_loop(
    stop_event: threading.Event, sid: str, session: dict
) -> None:
    """Poll completion_queue and dispatch notifications autonomously.

    Runs in a daemon thread started by _init_session(). Emits a
    status.update (kind=process) for user visibility, then chains an
    agent turn via _run_prompt_submit if the session is idle.

    The completion_queue is process-global. In multi-session Desktop each
    poller requeues events owned by another live session and drops addressed
    events whose owner is gone; ownerless legacy notifications remain global.

    Also polls ``kanban_notify_subs`` every ``_KANBAN_POLL_SECONDS`` for this
    session's TUI kanban subscriptions and delivers terminal task events the
    same way (status.update + agent turn) — the delivery path
    tools/kanban_tools.py documents for platform="tui" rows (issue #59890).
    """
    from tools.process_registry import process_registry, format_process_notification

    _emitted = set()  # dedup re-queued events so same completion isn't emitted 50 times while session is busy
    _last_kanban_poll = 0.0
    _last_loop_poll = 0.0
    while not stop_event.is_set() and not session.get("_finalized"):
        _now = time.monotonic()
        # ── /loop wakeup driver ──────────────────────────────────────
        # Fire a due /loop tick for THIS session while it's idle. Same
        # claim-under-lock pattern as the kanban dispatch below. Active
        # non-parked /goal owns the idle boundary and defers the tick.
        if _now - _last_loop_poll >= _LOOP_POLL_SECONDS:
            _last_loop_poll = _now
            try:
                _maybe_fire_tui_loop_tick(sid, session)
            except Exception as _loop_exc:
                print(
                    f"[tui_gateway] loop wakeup poll failed: "
                    f"{type(_loop_exc).__name__}: {_loop_exc}",
                    file=sys.stderr,
                )
        if _now - _last_kanban_poll >= _KANBAN_POLL_SECONDS:
            _last_kanban_poll = _now
            try:
                _kanban_texts = _collect_kanban_notifications(session)
            except Exception as _kb_exc:
                print(
                    f"[tui_gateway] kanban notification poll failed: "
                    f"{type(_kb_exc).__name__}: {_kb_exc}",
                    file=sys.stderr,
                )
                _kanban_texts = []
            if _kanban_texts:
                for _kb_text in _kanban_texts:
                    _emit("status.update", sid, {"kind": "process", "text": _kb_text})
                # Events are cursor-claimed (never re-queued), so buffer them
                # until the session is idle instead of dropping the agent turn.
                session.setdefault("_kanban_pending", []).extend(_kanban_texts)
            _pending = session.get("_kanban_pending") or []
            if _pending:
                _batch: list = []
                with session["history_lock"]:
                    if not session.get("running"):
                        session["running"] = True
                        _batch = list(_pending)
                        session["_kanban_pending"] = []
                if _batch:
                    rid = f"__notif__{int(time.time() * 1000)}"
                    try:
                        _emit("message.start", sid)
                        _run_prompt_submit(rid, sid, session, "\n".join(_batch))
                    except Exception as exc:
                        print(
                            f"[tui_gateway] kanban notification dispatch failed: "
                            f"{type(exc).__name__}: {exc}",
                            file=sys.stderr,
                        )
                        with session["history_lock"]:
                            session["running"] = False
        try:
            evt = process_registry.completion_queue.get(timeout=0.5)
        except Exception:
            continue

        # Multiple desktop sessions share this one process-wide queue. Only
        # consume events that belong to *this* session — otherwise a background
        # process started in session A would surface its completion in whichever
        # session's poller happened to wake first (Ben's "reported in a
        # different session" bug). Leave foreign events for their owner.
        if _notification_event_belongs_elsewhere(sid, session, evt):
            process_registry.completion_queue.put(evt)
            time.sleep(0.1)
            continue

        # What reaches here is not owned by another LIVE session. Addressed
        # events still require positive proof before injection: exact UI origin,
        # direct durable key, or compression lineage. If none proves ownership,
        # the event is orphaned and must not be adopted by this chat. Truly
        # ownerless ordinary notifications retain legacy global delivery.
        requires_owner = _notification_event_requires_owner(evt)
        if requires_owner and not _session_owns_notification_event(sid, session, evt):
            log = (
                logger.warning
                if evt.get("type") == "async_delegation"
                else logger.debug
            )
            log(
                "Dropping unowned %s notification (origin=%r key=%r) instead "
                "of delivering to session %s",
                evt.get("type", "completion"),
                str(evt.get("origin_ui_session_id") or ""),
                str(evt.get("session_key") or ""),
                sid,
            )
            continue

        _evt_sid = evt.get("session_id", "")
        if evt.get("type") == "completion" and process_registry.is_completion_consumed(_evt_sid):
            continue

        text = format_process_notification(evt)
        if not text:
            continue

        # Only emit the same notification identity to TUI once — re-queued
        # completions get re-emitted every 0.5s otherwise when session is busy,
        # while distinct watch_match events from the same process must remain
        # visible independently.
        _dedup_key = _notification_event_dedup_key(evt)
        if _dedup_key not in _emitted:
            _emit("status.update", sid, {"kind": "process", "text": text})
            _emitted.add(_dedup_key)

        _requeued = False
        with session["history_lock"]:
            if session.get("running"):
                process_registry.completion_queue.put(evt)
                _requeued = True
            else:
                session["running"] = True
        if _requeued:
            # Back off before re-polling: the re-queued event keeps the queue
            # non-empty, so without a sleep this loop spins at full speed
            # (100% CPU, GIL churn) for as long as the session stays busy.
            time.sleep(0.25)
            continue

        rid = f"__notif__{int(time.time() * 1000)}"
        from tools.async_delegation import (
            claim_event_delivery, complete_event_delivery, release_event_delivery,
        )
        _claim = claim_event_delivery(evt, "tui-poller")
        if _claim is None:
            continue
        try:
            _emit("message.start", sid)
            if evt.get("type") == "async_delegation":
                _run_prompt_submit(
                    rid,
                    sid,
                    session,
                    text,
                    display_kind="async_delegation_complete",
                    display_metadata=_async_delegation_display_metadata(evt),
                )
            else:
                _run_prompt_submit(rid, sid, session, text)
            complete_event_delivery(evt, _claim)
        except Exception as exc:
            release_event_delivery(evt, _claim)
            print(
                f"[tui_gateway] notification poller dispatch failed: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            with session["history_lock"]:
                session["running"] = False

    # Drain any remaining events after stop signal (process all pending
    # before exiting so nothing is lost on shutdown). Events owned by other
    # live sessions are set aside and re-queued so their poller still sees them.
    # Orphaned events (owner gone) are dropped — same guard as the main loop.
    deferred: list = []
    while not process_registry.completion_queue.empty():
        try:
            evt = process_registry.completion_queue.get_nowait()
        except Exception:
            break
        if _notification_event_belongs_elsewhere(sid, session, evt):
            deferred.append(evt)
            continue
        # Same positive-proof rule as the live loop. Preserve the existing
        # shutdown behavior for orphaned delegation payloads by deferring them
        # for a later resume; ordinary addressed orphans are dropped.
        requires_owner = _notification_event_requires_owner(evt)
        if requires_owner and not _session_owns_notification_event(sid, session, evt):
            if evt.get("type") == "async_delegation":
                deferred.append(evt)
            else:
                logger.debug(
                    "Dropping unowned %s notification during shutdown drain "
                    "(origin=%r key=%r)",
                    evt.get("type", "completion"),
                    str(evt.get("origin_ui_session_id") or ""),
                    str(evt.get("session_key") or ""),
                )
            continue
        _evt_sid = evt.get("session_id", "")
        if evt.get("type") == "completion" and process_registry.is_completion_consumed(_evt_sid):
            continue
        text = format_process_notification(evt)
        if not text:
            continue

        _dedup_key = _notification_event_dedup_key(evt)
        if _dedup_key not in _emitted:
            _emit("status.update", sid, {"kind": "process", "text": text})
            _emitted.add(_dedup_key)

        with session["history_lock"]:
            if session.get("running"):
                process_registry.completion_queue.put(evt)
                break
            session["running"] = True

        rid = f"__notif__{int(time.time() * 1000)}"
        from tools.async_delegation import (
            claim_event_delivery, complete_event_delivery, release_event_delivery,
        )
        _claim = claim_event_delivery(evt, "tui-poller")
        if _claim is None:
            continue
        try:
            _emit("message.start", sid)
            if evt.get("type") == "async_delegation":
                _run_prompt_submit(
                    rid,
                    sid,
                    session,
                    text,
                    display_kind="async_delegation_complete",
                    display_metadata=_async_delegation_display_metadata(evt),
                )
            else:
                _run_prompt_submit(rid, sid, session, text)
            complete_event_delivery(evt, _claim)
        except Exception as exc:
            release_event_delivery(evt, _claim)
            print(
                f"[tui_gateway] notification poller dispatch failed: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            with session["history_lock"]:
                session["running"] = False

    # Hand any other sessions' events back to the shared queue.
    for evt in deferred:
        process_registry.completion_queue.put(evt)


def _async_delegation_display_metadata(evt: dict) -> dict:
    """Build display-only metadata before the completion event is formatted."""
    raw_results = evt.get("results")
    results: list[dict] = [
        result for result in raw_results if isinstance(result, dict)
    ] if isinstance(raw_results, list) else []
    task_count = len(results) or 1
    completed_count = sum(
        1 for result in results
        if result.get("status") in {"completed", "success"}
    )
    failed_count = sum(
        1 for result in results
        if result.get("status") in {"failed", "error"}
    )
    metadata = {
        "delegation_id": str(evt.get("delegation_id") or ""),
        "task_count": task_count,
        "completed_count": completed_count or task_count - failed_count,
        "failed_count": failed_count,
    }
    duration = evt.get("total_duration_seconds") or evt.get("duration_seconds")
    if isinstance(duration, (int, float)):
        metadata["duration_seconds"] = duration
    return metadata


def _wire_agent_terminal_output() -> None:
    """Idempotently route background-process output (and tab-close requests) to
    the desktop, keyed by process id. Read-only agent terminal tabs stream
    `agent.terminal.output` chunks live instead of polling the output tail, and
    `process_registry.request_close_terminal` emits `terminal.close` so the agent
    can drop a tab without killing the process. Events are routed to the window
    that owns the process (its gateway session); `_emit`/`write_json` is
    `_stdout_lock`-guarded, so calling it from the registry's reader threads is
    safe."""
    from tools.process_registry import process_registry

    has_output_sink = getattr(process_registry, "on_output", None) is not None
    has_close_sink = getattr(process_registry, "on_close", None) is not None
    if has_output_sink and has_close_sink:
        return

    def _owner_sid_for_process(session) -> str:
        session_key = str(getattr(session, "session_key", "") or "")
        if not session_key:
            return ""
        with _sessions_lock:
            for sid, tui_session in _sessions.items():
                if str(tui_session.get("session_key") or "") == session_key:
                    return sid
        return ""

    def _emit_agent_terminal_output(session, chunk):
        _emit(
            "agent.terminal.output",
            _owner_sid_for_process(session),
            {"process_id": session.id, "chunk": chunk},
        )

    def _emit_agent_terminal_close(session, process_id):
        # session may be None (process already finished/pruned) — the tab can
        # still linger and be closed; route to the owning window when we can.
        sid = _owner_sid_for_process(session) if session is not None else ""
        _emit("terminal.close", sid, {"process_id": process_id})

    if not has_output_sink:
        process_registry.on_output = _emit_agent_terminal_output
    if not has_close_sink:
        process_registry.on_close = _emit_agent_terminal_close


_desktop_ui_wired = False


def _wire_desktop_ui() -> None:
    """Bridge desktop-only tools (open_preview, close_preview, focus_pane) to renderer events.

    Idempotent. The tool hands back the turn's ``HERMES_UI_SESSION_ID`` as
    ``sid`` so the event routes to the window that asked (``_emit`` /
    ``write_json`` is ``_stdout_lock``-guarded, so calling it from the tool's
    thread is safe)."""
    global _desktop_ui_wired
    if _desktop_ui_wired:
        return
    try:
        from tools import desktop_ui
    except Exception:
        return

    desktop_ui.set_emitter(lambda sid, event, payload: _emit(event, sid, payload))
    _desktop_ui_wired = True


# (stop_event, thread) for every poller ever started in this process.
# Pruned of dead threads on each spawn; consumed by test teardowns to reap
# leaked pollers (see _start_notification_poller).
_notification_pollers: list = []


def _start_notification_poller(sid: str, session: dict) -> threading.Event:
    """Start the background notification poller for a TUI session."""
    _wire_agent_terminal_output()
    _wire_desktop_ui()
    stop = threading.Event()
    t = threading.Thread(
        target=_notification_poller_loop,
        args=(stop, sid, session),
        daemon=True,
        # Stable, greppable name for debuggers and test teardowns.
        name=f"tui-notif-poller-{sid}",
    )
    # Registry of (stop, thread) pairs so test teardowns can reap pollers
    # leaked by session.init/create tests — an unjoined poller steals
    # events off the process-global completion_queue mid-assertion in a
    # LATER test (flaky test_run_prompt_submit_requeues_all_unstarted_...).
    # Bounded: entries for dead threads are pruned on each spawn.
    _notification_pollers[:] = [
        (s, th) for (s, th) in _notification_pollers if th.is_alive()
    ]
    _notification_pollers.append((stop, t))
    t.start()
    return stop


def _hud_surface_note(session: dict) -> str:
    """The HUD-mode note for this turn, or "" when it was not typed there."""
    if session.get("client_surface") != "hud":
        return ""
    from agent.prompt_builder import hud_surface_note

    return hud_surface_note(getattr(session.get("agent"), "valid_tool_names", None))


def _prepend_note(run_message: Any, note: str) -> Any:
    """Prefix a per-turn note onto the MODEL INPUT, leaving the prompt alone.

    Everything the model needs to know about the turn but the user did not
    type — an interrupted reply, reactions, the surface they typed into —
    arrives this way. persist_user_message keeps the clean prompt, so no
    scaffolding reaches the transcript, and annotating the NEW turn never
    rewrites an already-sent message, so the cached prefix survives.
    """
    if not note:
        return run_message
    if isinstance(run_message, str):
        return f"{note}\n\n{run_message}"
    if isinstance(run_message, list):
        return [{"type": "text", "text": note}, *run_message]
    return run_message


def register(server) -> None:
    """Publish this module's helpers + handlers onto ``server``, rebound to its globals."""
    bind_module(globals(), server, skip=("_",))
