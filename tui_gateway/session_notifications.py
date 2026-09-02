"""Per-session notification poller: kanban/loop/delegation events routed to the owning session, desktop UI wiring, HUD surface note.

Bodies are rebound onto server.py's globals at install time (see
method_ctx.bind_module), so they reference server.py globals bare.
"""

from __future__ import annotations


import contextlib

from .method_ctx import HandlerRegistry, bind_module

_registry = HandlerRegistry()


def _notif_locked_sessions(fn, default):
    """Run ``fn(_sessions)`` under ``_sessions_lock``; ``default`` if that fails (the poller
    thread must never crash on a lock/enumeration failure)."""
    try:
        with _sessions_lock:
            return fn(_sessions)
    except Exception:
        return default


def _notif_current_keys(sid: str, session: dict) -> set:
    return {str(session.get("session_key") or ""), _session_lookup_key(session, fallback=sid)}


def _notif_session_matches(s: dict, keys) -> bool:
    return str(s.get("session_key") or "") in keys or _session_lookup_key(s, fallback="") in keys


def _notif_resolve_event_key(evt_key: str) -> str:
    """Resolve a compression-rotated session key to its continuation tip (or itself)."""
    try:
        db = _get_db()
        return (db.resolve_resume_session_id(evt_key) if db is not None else evt_key) or evt_key
    except Exception:
        return evt_key


def _notification_event_belongs_elsewhere(sid: str, session: dict, evt: dict) -> bool:
    """True if ``evt`` is owned by a *different* live session.

    Background completions carry the ``session_key`` of the session that started the work; async
    delegation completions also carry ``origin_ui_session_id`` (the live TUI tab that commissioned
    them). All desktop sessions share one process-wide completion queue, so each poller must skip
    events it doesn't own or a detached result surfaces in whichever poller dequeued first."""
    evt_ui_sid = str(evt.get("origin_ui_session_id") or "")
    if evt_ui_sid:
        if evt_ui_sid == str(sid or "") and not session.get("_finalized"):
            return False
        if _notif_locked_sessions(lambda ss: evt_ui_sid in ss and not ss[evt_ui_sid].get("_finalized"), False):
            return True
        # Exact UI tab gone: fall through to durable session_key routing so a
        # resumed continuation with the same key/lineage can still claim it.

    evt_key = str(evt.get("session_key") or "")
    if not evt_key:
        return False

    current_keys = _notif_current_keys(sid, session)

    # Compression can rotate AIAgent.session_id while the detached child is still running: map
    # the event's original key to its continuation tip so it reaches the live session instead of
    # becoming an orphan any poller may consume. A live continuation wins over the compressed
    # parent, else a stale parent tab could consume the event before the current conversation.
    resolved_key = _notif_resolve_event_key(evt_key)
    if resolved_key != evt_key:
        if resolved_key in current_keys:
            return False
        if _notif_locked_sessions(
            lambda ss: any(not s.get("_finalized") and _notif_session_matches(s, {resolved_key}) for s in ss.values()),
            False,
        ):
            return True

    if evt_key in current_keys:
        return False

    snapshot = _notif_locked_sessions(lambda ss: list(ss.values()), None)
    if snapshot is None:
        return False  # can't enumerate: fail open rather than drop the event

    keys = {evt_key, resolved_key}
    return any(s is not session and not s.get("_finalized") and _notif_session_matches(s, keys) for s in snapshot)


def _session_owns_notification_event(sid: str, session: dict, evt: dict) -> bool:
    """True iff *this* session PROVABLY owns ``evt``: positive mirror of
    ``_notification_event_belongs_elsewhere`` minus its orphan-adoption fallback (UI origin is
    this live session, or ``session_key`` raw/compression-resolved matches this session). Fail-
    closed gate for every addressed notification — "not provably elsewhere" is NOT enough."""
    if session.get("_finalized"):
        return False
    if str(evt.get("origin_ui_session_id") or "") == str(sid or ""):
        return True
    evt_key = str(evt.get("session_key") or "")
    if not evt_key:
        return False
    current_keys = _notif_current_keys(sid, session)
    if evt_key in current_keys:
        return True
    return _notif_resolve_event_key(evt_key) in current_keys


def _notification_event_requires_owner(evt: dict) -> bool:
    """Whether ``evt`` must be positively claimed before TUI delivery."""
    return evt.get("type") == "async_delegation" or bool(
        str(evt.get("origin_ui_session_id") or "") or str(evt.get("session_key") or "")
    )


def _notification_event_dedup_key(evt: dict) -> tuple:
    """UI-emission identity for a process notification event. Completions are terminal (one-shot
    per process session); watch events are not — one process can match patterns many times, so
    include event content to avoid suppressing later distinct matches."""
    evt_type = evt.get("type", "completion")
    evt_sid = evt.get("session_id", "")
    if evt_type == "watch_match":
        return (evt_sid, evt_type, evt.get("command", ""), evt.get("pattern", ""), evt.get("output", ""),
                evt.get("suppressed", 0), evt.get("message_id", ""))
    if evt_type.startswith("watch_overflow_") or evt_type == "watch_disabled":
        return (evt_sid, evt_type, evt.get("command", ""), evt.get("message", ""), evt.get("suppressed", 0))
    if evt_type == "async_delegation":
        # No process session_id: without this every completion keys as
        # ("", "async_delegation") and the second one is suppressed forever.
        return (evt.get("delegation_id", ""), evt_type)
    return (evt_sid, evt_type)


# Mirror gateway/kanban_watchers.py TERMINAL_KINDS: claim silent kinds too so
# the cursor advances past them and they can't wedge a later completed/blocked
# event behind an unclaimed row.
_KANBAN_NOTIFY_KINDS = ("completed", "blocked", "gave_up", "crashed", "timed_out", "status", "archived", "unblocked")
_KANBAN_SILENT_KINDS = frozenset({"archived", "unblocked"})
_KANBAN_POLL_SECONDS = 5.0
_LOOP_POLL_SECONDS = 5.0


def _notif_release_turn(session: dict) -> None:
    with session["history_lock"]:
        session["running"] = False


def _notif_log_failure(what: str, exc: BaseException) -> None:
    print(f"[tui_gateway] {what}: {type(exc).__name__}: {exc}", file=sys.stderr)


def _maybe_fire_tui_loop_tick(sid: str, session: dict) -> None:
    """Fire a due /loop wakeup for an idle TUI/Desktop/dashboard session (per-session poller,
    coarse cadence). Claims the session under history_lock (running=True) before dispatching so
    a racing user prompt wins cleanly; the turn dispatcher's post-turn hook completes the tick."""
    try:
        from hermes_cli.loops import LoopManager, goal_blocks_loop_tick
    except Exception:
        return

    sid_key = session.get("session_key") or ""
    if not sid_key:
        return
    mgr = LoopManager(session_id=sid_key)
    if not mgr.is_due() or goal_blocks_loop_tick(sid_key):
        return

    with session["history_lock"]:
        if session.get("running"):
            return  # busy — stays due, next poll retries
        session["running"] = True

    wakeup = mgr.fire_tick()
    if not wakeup:
        _notif_release_turn(session)
        return

    tick_no = mgr.state.ticks_fired if mgr.state else "?"
    rid = f"__loop__{int(time.time() * 1000)}"
    try:
        _emit("status.update", sid, {"kind": "loop", "text": f"↻ /loop wakeup #{tick_no} firing…"})
        if wakeup.lstrip().startswith("/"):
            # Slash-command loop: route through the slash pipeline, not the
            # model. No model reply to evaluate — complete immediately.
            _notif_release_turn(session)
            try:
                parts = wakeup.lstrip()[1:].split(None, 1)
                resp = _methods["command.dispatch"](
                    rid,
                    {"name": parts[0] if parts else "", "arg": parts[1] if len(parts) > 1 else "", "session_id": sid},
                )
                payload = (resp or {}).get("result") or {}
                out = str(payload.get("output") or "").strip()
                if out:
                    _emit("status.update", sid, {"kind": "loop", "text": out})
                if payload.get("type") == "send" and payload.get("message"):
                    # Command resolves to a prompt (skill command etc.) — run it
                    # as a normal turn; the post-turn hook completes the tick.
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
        _notif_log_failure("loop wakeup dispatch failed", exc)
        _notif_release_turn(session)
        with contextlib.suppress(Exception):
            mgr.abandon_tick()


def _format_kanban_event_text(sub: dict, task, ev, board_slug: str) -> Optional[str]:
    """Single-line notification text for one kanban event; wording mirrors
    gateway/kanban_watchers.py so it reads the same as on Telegram. None for silent kinds."""
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
        try:
            limit = int(payload.get("limit_seconds") or 0)
        except (TypeError, ValueError):
            limit = 0
        return f"⏱ {board_tag}{tag}Kanban {task_id} timed out (max_runtime={limit}s); will retry"
    if kind == "status":
        return f"🔄 {board_tag}{tag}Kanban {task_id} → {payload.get('status') or ''}"
    return None


def _collect_kanban_notifications(session: dict) -> list:
    """Claim unseen terminal kanban events for this TUI session's subscriptions.

    ``kanban_create`` auto-subscribes TUI/desktop sessions with ``platform="tui"`` and
    ``chat_id=HERMES_SESSION_KEY``; there is no "tui" messaging adapter, so this poller is the
    delivery path. Same atomic cursor-claim (``claim_unseen_events_for_sub``) as the gateway
    notifier, so a sub is delivered exactly once even if a gateway and a TUI poll the same board
    DB. Returns formatted notification texts (may be empty)."""
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
    # DB when HERMES_KANBAN_DB pins the board path (same guard as the gateway).
    seen_db_paths: set = set()
    for board_meta in boards:
        slug = (board_meta or {}).get("slug") or _kb.DEFAULT_BOARD
        db_path = (board_meta or {}).get("db_path")
        try:
            resolved = str(Path(db_path).expanduser().resolve() if db_path else _kb.kanban_db_path(slug).resolve())
        except Exception:
            resolved = f"slug:{slug}"
        if resolved in seen_db_paths:
            continue
        seen_db_paths.add(resolved)
        # One poller per live session: don't open this board writable unless it
        # has a subscription owned by this exact session. If the read-only probe
        # fails (locked/corrupt DB), preserve delivery and fall through.
        try:
            if _kb.count_notify_subs(board=slug, platform="tui", chat_id=session_key) == 0:
                continue
        except Exception:
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
                if (sub.get("platform") or "").lower() != "tui" or sub.get("chat_id") != session_key:
                    continue
                sub_ident = dict(
                    task_id=sub["task_id"], platform=sub["platform"], chat_id=sub["chat_id"],
                    thread_id=sub.get("thread_id") or "",
                )
                _old, _new, events = _kb.claim_unseen_events_for_sub(conn, kinds=_KANBAN_NOTIFY_KINDS, **sub_ident)
                if not events:
                    continue
                task = _kb.get_task(conn, sub["task_id"])
                for ev in events:
                    text = _format_kanban_event_text(sub, task, ev, slug)
                    if text:
                        texts.append(text)
                # Unsubscribe only on archive: ``done`` is reversible in review/controller flows,
                # so keeping the sub lets a later reopen notify the same session. The claimed
                # cursor prevents replay.
                if task and getattr(task, "status", "") == "archived":
                    with contextlib.suppress(Exception):
                        _kb.remove_notify_sub(conn, **sub_ident)
        finally:
            conn.close()
    return texts


def _notif_poll_kanban(sid: str, session: dict) -> None:
    """One kanban poll: emit new texts, buffer them, and run the buffered batch as a turn if idle.
    Events are cursor-claimed (never re-queued), so they are buffered until the session is idle
    instead of dropping the agent turn."""
    try:
        _kanban_texts = _collect_kanban_notifications(session)
    except Exception as _kb_exc:
        _notif_log_failure("kanban notification poll failed", _kb_exc)
        _kanban_texts = []
    if _kanban_texts:
        for _kb_text in _kanban_texts:
            _emit("status.update", sid, {"kind": "process", "text": _kb_text})
        session.setdefault("_kanban_pending", []).extend(_kanban_texts)
    _pending = session.get("_kanban_pending") or []
    if not _pending:
        return
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
            _notif_log_failure("kanban notification dispatch failed", exc)
            _notif_release_turn(session)


def _notif_render_event(sid: str, evt: dict, emitted: set, process_registry, format_process_notification) -> Optional[str]:
    """Format ``evt`` and emit its status.update once; None means skip the event (consumed
    completion or unformattable). Re-queued completions would otherwise re-emit every 0.5s while
    the session is busy, while distinct watch_match events from one process must stay visible."""
    if evt.get("type") == "completion" and process_registry.is_completion_consumed(evt.get("session_id", "")):
        return None
    text = format_process_notification(evt)
    if not text:
        return None
    dedup_key = _notification_event_dedup_key(evt)
    if dedup_key not in emitted:
        _emit("status.update", sid, {"kind": "process", "text": text})
        emitted.add(dedup_key)
    return text


def _notif_dispatch_event(sid: str, session: dict, evt: dict, text: str) -> None:
    """Run the claimed (running=True) agent turn for one notification event."""
    from tools.async_delegation import claim_event_delivery, complete_event_delivery, release_event_delivery

    rid = f"__notif__{int(time.time() * 1000)}"
    claim = claim_event_delivery(evt, "tui-poller")
    if claim is None:
        return
    try:
        _emit("message.start", sid)
        if evt.get("type") == "async_delegation":
            _run_prompt_submit(
                rid, sid, session, text, display_kind="async_delegation_complete",
                display_metadata=_async_delegation_display_metadata(evt),
            )
        else:
            _run_prompt_submit(rid, sid, session, text)
        complete_event_delivery(evt, claim)
    except Exception as exc:
        release_event_delivery(evt, claim)
        _notif_log_failure("notification poller dispatch failed", exc)
        _notif_release_turn(session)


def _notification_poller_loop(stop_event: threading.Event, sid: str, session: dict) -> None:
    """Poll completion_queue and dispatch notifications autonomously (daemon thread started by
    _init_session()): emit a status.update (kind=process), then chain an agent turn via
    _run_prompt_submit if the session is idle. The queue is process-global: each poller requeues
    events owned by another live session and drops addressed events whose owner is gone;
    ownerless legacy notifications remain global. Also polls ``kanban_notify_subs`` every
    ``_KANBAN_POLL_SECONDS`` — the delivery path for platform="tui" rows."""
    from tools.process_registry import process_registry, format_process_notification

    _emitted = set()  # dedup re-queued events so one completion isn't emitted 50 times while busy
    _last_kanban_poll = 0.0
    _last_loop_poll = 0.0
    while not stop_event.is_set() and not session.get("_finalized"):
        _now = time.monotonic()
        # /loop wakeup driver: fire a due tick for THIS session while idle (same claim-under-lock
        # as kanban dispatch). An active non-parked /goal owns the idle boundary and defers it.
        if _now - _last_loop_poll >= _LOOP_POLL_SECONDS:
            _last_loop_poll = _now
            try:
                _maybe_fire_tui_loop_tick(sid, session)
            except Exception as _loop_exc:
                _notif_log_failure("loop wakeup poll failed", _loop_exc)
        if _now - _last_kanban_poll >= _KANBAN_POLL_SECONDS:
            _last_kanban_poll = _now
            _notif_poll_kanban(sid, session)
        try:
            evt = process_registry.completion_queue.get(timeout=0.5)
        except Exception:
            continue

        # Leave foreign events for their owner — otherwise a process started in
        # session A surfaces its completion in whichever poller wakes first.
        if _notification_event_belongs_elsewhere(sid, session, evt):
            process_registry.completion_queue.put(evt)
            time.sleep(0.1)
            continue

        # Not owned by another LIVE session, but addressed events still need positive proof
        # (exact UI origin, direct durable key, or compression lineage) — an orphan must not be
        # adopted by this chat. Truly ownerless ordinary notifications keep legacy global delivery.
        if _notification_event_requires_owner(evt) and not _session_owns_notification_event(sid, session, evt):
            log = logger.warning if evt.get("type") == "async_delegation" else logger.debug
            log(
                "Dropping unowned %s notification (origin=%r key=%r) instead of delivering to session %s",
                evt.get("type", "completion"), str(evt.get("origin_ui_session_id") or ""),
                str(evt.get("session_key") or ""), sid,
            )
            continue

        text = _notif_render_event(sid, evt, _emitted, process_registry, format_process_notification)
        if not text:
            continue

        _requeued = False
        with session["history_lock"]:
            if session.get("running"):
                process_registry.completion_queue.put(evt)
                _requeued = True
            else:
                session["running"] = True
        if _requeued:
            # Back off: the re-queued event keeps the queue non-empty, so
            # without a sleep this loop spins at 100% CPU while busy.
            time.sleep(0.25)
            continue

        _notif_dispatch_event(sid, session, evt, text)

    # Drain remaining events after the stop signal so nothing is lost on shutdown. Other live
    # sessions' events are set aside and re-queued; orphaned events (owner gone) are dropped —
    # same guard as the main loop, except orphaned delegation payloads are deferred for a resume.
    deferred: list = []
    while not process_registry.completion_queue.empty():
        try:
            evt = process_registry.completion_queue.get_nowait()
        except Exception:
            break
        if _notification_event_belongs_elsewhere(sid, session, evt):
            deferred.append(evt)
            continue
        if _notification_event_requires_owner(evt) and not _session_owns_notification_event(sid, session, evt):
            if evt.get("type") == "async_delegation":
                deferred.append(evt)
            else:
                logger.debug(
                    "Dropping unowned %s notification during shutdown drain (origin=%r key=%r)",
                    evt.get("type", "completion"), str(evt.get("origin_ui_session_id") or ""),
                    str(evt.get("session_key") or ""),
                )
            continue
        text = _notif_render_event(sid, evt, _emitted, process_registry, format_process_notification)
        if not text:
            continue

        with session["history_lock"]:
            if session.get("running"):
                process_registry.completion_queue.put(evt)
                break
            session["running"] = True

        _notif_dispatch_event(sid, session, evt, text)

    # Hand any other sessions' events back to the shared queue.
    for evt in deferred:
        process_registry.completion_queue.put(evt)


def _async_delegation_display_metadata(evt: dict) -> dict:
    """Build display-only metadata before the completion event is formatted."""
    raw_results = evt.get("results")
    results: list[dict] = [r for r in raw_results if isinstance(r, dict)] if isinstance(raw_results, list) else []
    task_count = len(results) or 1
    completed_count = sum(1 for r in results if r.get("status") in {"completed", "success"})
    failed_count = sum(1 for r in results if r.get("status") in {"failed", "error"})
    metadata = {
        "delegation_id": str(evt.get("delegation_id") or ""), "task_count": task_count,
        "completed_count": completed_count or task_count - failed_count, "failed_count": failed_count,
    }
    duration = evt.get("total_duration_seconds") or evt.get("duration_seconds")
    if isinstance(duration, (int, float)):
        metadata["duration_seconds"] = duration
    return metadata


def _wire_agent_terminal_output() -> None:
    """Idempotently route background-process output (`agent.terminal.output` chunks) and
    `process_registry.request_close_terminal` (`terminal.close`, drops a tab without killing the
    process) to the window owning the process (its gateway session), keyed by process id.
    `_emit` is `_stdout_lock`-guarded, so the registry's reader threads may call it."""
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
        _emit("agent.terminal.output", _owner_sid_for_process(session), {"process_id": session.id, "chunk": chunk})

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
    Idempotent. The tool hands back the turn's ``HERMES_UI_SESSION_ID`` as ``sid`` so the event
    routes to the window that asked (``_emit`` is ``_stdout_lock``-guarded; tool thread may call it)."""
    global _desktop_ui_wired
    if _desktop_ui_wired:
        return
    try:
        from tools import desktop_ui
    except Exception:
        return

    desktop_ui.set_emitter(lambda sid, event, payload: _emit(event, sid, payload))
    _desktop_ui_wired = True


# (stop_event, thread) for every poller started in this process, pruned of dead threads on each
# spawn. Test teardowns reap leaked pollers through it: an unjoined poller steals events off the
# process-global completion_queue mid-assertion in a LATER test.
_notification_pollers: list = []


def _start_notification_poller(sid: str, session: dict) -> threading.Event:
    """Start the background notification poller for a TUI session."""
    _wire_agent_terminal_output()
    _wire_desktop_ui()
    stop = threading.Event()
    # Thread name is greppable for debuggers/test teardowns.
    t = threading.Thread(
        target=_notification_poller_loop, args=(stop, sid, session), daemon=True, name=f"tui-notif-poller-{sid}"
    )
    _notification_pollers[:] = [(s, th) for (s, th) in _notification_pollers if th.is_alive()]
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
    """Prefix a per-turn note onto the MODEL INPUT, leaving the prompt alone. Everything the model
    must know about the turn that the user did not type (interrupted reply, reactions, surface)
    arrives this way; persist_user_message keeps the clean prompt, so no scaffolding reaches the
    transcript, and annotating the NEW turn never rewrites a sent message — cached prefix survives."""
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
