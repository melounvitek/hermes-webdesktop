"""Auto-continue: resume a turn killed by a process/machine death, plus queued-prompt drain and busy-submit handling.

Bodies are rebound onto server.py's globals at install time (see
method_ctx.bind_module), so they reference server.py globals bare.
"""

from __future__ import annotations


from .method_ctx import HandlerRegistry, bind_module

_registry = HandlerRegistry()


# ── Auto-continue: resume a turn killed by a process/machine death ────
#
# A turn that concludes — success, handled error, interrupt — clears its
# durable marker (see tui_gateway/turn_marker.py) in _run_prompt_submit's
# finally. Only a process death leaves the marker behind, so a marker found
# at session.resume time is positive proof the turn never finished AND the
# client never saw a terminal frame. If the interruption is fresh, re-submit
# the interrupted prompt automatically (the messaging gateway has done this
# for restart-interrupted sessions since #27856); if it's stale, clear the
# marker and let the recovered partial transcript speak for itself — the
# user can ask to continue manually.

_AUTO_CONTINUE_ENABLED_DEFAULT = True
_AUTO_CONTINUE_FRESHNESS_MINUTES_DEFAULT = 15
_AUTO_CONTINUE_MAX_ATTEMPTS_DEFAULT = 2


def _auto_continue_config() -> tuple[bool, float, int]:
    """(enabled, freshness window in seconds, max attempts) from config.yaml."""
    desktop = _load_cfg().get("desktop")
    cfg = desktop.get("auto_continue") if isinstance(desktop, dict) else None
    if not isinstance(cfg, dict):
        cfg = {}
    try:
        minutes = float(cfg.get("freshness_minutes", _AUTO_CONTINUE_FRESHNESS_MINUTES_DEFAULT))
    except (TypeError, ValueError):
        minutes = float(_AUTO_CONTINUE_FRESHNESS_MINUTES_DEFAULT)
    return (
        is_truthy_value(cfg.get("enabled"), default=_AUTO_CONTINUE_ENABLED_DEFAULT),
        max(0.0, minutes) * 60.0,
        _coerce_int_config_value(
            cfg.get("max_attempts"), _AUTO_CONTINUE_MAX_ATTEMPTS_DEFAULT, min_value=0
        ),
    )


def _session_home(session: dict) -> Path:
    """The HERMES_HOME the session's durable state lives in (profile-aware)."""
    profile_home = session.get("profile_home")
    return Path(profile_home) if profile_home else Path(_hermes_home)


def _retire_turn_marker(session: dict, *keys: str) -> None:
    """Drop the crash marker for a turn whose outcome is about to reach the client.

    Called immediately before the terminal frame rather than at the end of the
    turn thread: post-turn work (titles, memory sync, goal hooks) runs for a
    second or more after the client has its answer, and quitting inside that
    window would leave a marker that looks like a crash — re-running a finished
    turn on the next launch. Extra ``keys`` cover a session_key that
    compression rotated mid-turn.
    """
    home = _session_home(session)
    for key in dict.fromkeys((*keys, str(session.get("session_key") or ""))):
        if key:
            clear_turn_marker(home, key)


def _auto_continue_note(prompt: str) -> str:
    # Same opening as the messaging gateway's recovery notes so transcript
    # tooling recognizes both. The original prompt is embedded because a hard
    # crash persists nothing of the interrupted turn to the session DB — this
    # note is the only copy the model will see.
    return (
        f"{_AUTO_CONTINUE_NOTE_PREFIX} — the app or its backend process "
        "stopped before the turn could finish. Some of the work may already "
        "be complete; check the current state before redoing anything, then "
        "finish the task. The interrupted request was:]\n\n"
        f"{prompt}"
    )


def _maybe_schedule_auto_continue(sid: str, session: dict, session_key: str) -> dict | None:
    """Kick off a continuation turn for a crash-interrupted session.

    Called from session.resume's cold paths after the live record is
    registered. Returns a small descriptor for the resume payload when a
    continuation was scheduled, else None. The turn itself runs on a
    background thread after the (deferred) agent build finishes, through the
    same _run_prompt_submit machinery as every other synthesized turn — so
    the client that just resumed streams it live.
    """
    # Hosted room turns are recovered by their durable task/lease state
    # machine. Generic session auto-continue would bypass its execution
    # generation and can duplicate work after a process restart.
    if session.get("source") == "bot_room":
        return None

    home = _session_home(session)
    marker = read_turn_marker(home, session_key)
    if marker is None:
        return None
    enabled, freshness_secs, max_attempts = _auto_continue_config()
    age = time.time() - marker["started_at"]
    if not enabled or age > freshness_secs or marker["attempts"] >= max_attempts:
        # Stale, disabled, or crash-looping: stop trying. The journal/partial
        # transcript still shows what happened; a manual message continues it.
        clear_turn_marker(home, session_key)
        return None
    if session.get("_auto_continue_scheduled"):
        return None
    session["_auto_continue_scheduled"] = True
    attempt = marker["attempts"] + 1
    text = _auto_continue_note(marker["prompt"])

    def kickoff() -> None:
        rid = f"__auto_continue__{int(time.time() * 1000)}"
        try:
            _start_agent_build(sid, session)
            err = _wait_agent(session, rid, timeout=120.0)
        except Exception:
            logger.warning("auto-continue agent build failed for %s", sid, exc_info=True)
            err = {"error": {"message": "agent build failed"}}
        if err:
            # Leave the marker: the next resume retries (bounded by attempts).
            session["_auto_continue_scheduled"] = False
            return
        with session["history_lock"]:
            if session.get("running") or session.get("_turn_cancel_requested") or session.get("_finalized"):
                # A real user prompt beat us to it — their turn wins, and its
                # own conclusion clears the marker.
                session["_auto_continue_scheduled"] = False
                return
            session["running"] = True
            session["last_active"] = time.time()
        # Ownership admission BEFORE message.start: the interrupted-turn
        # marker this continuation is recovering may have been written by a
        # sibling backend that is still alive and mid-turn (#94778 — two
        # backends share one HERMES_HOME; B resumes S while A runs it and
        # sees A's fresh marker). Running the continuation anyway would be
        # the double-writer this fence exists to prevent. Leave the marker:
        # once the owner finishes or dies, a later resume retries.
        if _ensure_active_session_slot(sid, session) is not None:
            logger.info(
                "auto-continue for %s refused: session has another live owner",
                session_key,
            )
            with session["history_lock"]:
                session["running"] = False
                session["_auto_continue_scheduled"] = False
            return
        with session["history_lock"]:
            # Hand this turn its own marker inputs (read back by
            # _run_prompt_submit): count the attempt so a crash during the
            # continuation trips the breaker, and re-record the ORIGINAL
            # prompt so a second crash doesn't nest note inside note. Set
            # here, not at schedule time, so a bail above leaves nothing
            # behind for a racing user turn to inherit.
            session["_auto_continue_attempt"] = attempt
            session["_auto_continue_prompt"] = marker["prompt"]
        try:
            _emit(
                "status.update",
                sid,
                {"kind": "process", "text": "Resuming interrupted turn…"},
            )
            _emit("message.start", sid)
            _run_prompt_submit(rid, sid, session, text, display_kind="auto_continue")
        except Exception as exc:
            print(
                f"[tui_gateway] auto-continue dispatch failed: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            with session["history_lock"]:
                session["running"] = False

    threading.Thread(target=kickoff, daemon=True).start()
    logger.info(
        "auto-continue scheduled for session %s (attempt %d, interrupted %.0fs ago)",
        session_key,
        attempt,
        age,
    )
    return {"attempt": attempt, "interrupted_at": marker["started_at"]}


def _enqueue_prompt(
    session: dict,
    text: Any,
    transport: Any,
    image_paths: list[str] | None = None,
) -> None:
    """Stash a message to run as the very next turn once the live one ends.

    Used when a prompt arrives mid-turn (see ``_handle_busy_submit``). Text-only
    arrivals share a slot and merge losslessly (mirroring the consecutive-user
    merge in ``repair_message_sequence``). Image-bearing submissions stay as
    separate envelopes, so their attachment ownership and chronology survive.
    ``transport`` is pinned so the drained turn streams back to the client that
    sent it even if the session transport is rebound meanwhile.
    """
    image_paths = list(image_paths or [])
    # #84417: scrub any live-turn self-duplicates first so the consecutive-text
    # merge below cannot glue "{original}\\n\\n{later}" and re-fire original
    # on drain after a later correction settles.
    _drop_queued_duplicates_of_inflight_user(session)
    # Never queue a text-only self-copy of the live inflight user prompt. The
    # live turn already owns that text; draining it after settle would restart
    # the same user turn as a fresh agent invocation.
    if not image_paths and isinstance(text, str):
        turn = session.get("inflight_turn")
        original = (
            str(turn.get("user") or "").strip() if isinstance(turn, dict) else ""
        )
        if original and text.strip() == original:
            return
    queued = {"text": text, "transport": transport}
    if image_paths:
        queued["image_paths"] = image_paths
    existing = session.get("queued_prompt")
    if (
        existing
        and isinstance(existing.get("text"), str)
        and isinstance(text, str)
        and not existing.get("image_paths")
        and not image_paths
        and not session.get("queued_prompts")
    ):
        prev = existing["text"]
        existing["text"] = f"{prev}\n\n{text}" if prev and text else (prev or text)
        return
    if existing:
        session.setdefault("queued_prompts", []).append(queued)
        return
    session["queued_prompt"] = queued


def _sanitize_queued_entry_vs_inflight_user(
    entry: Any, original: str
) -> dict | None:
    """Drop or rewrite a queue envelope that re-carries the live user text.

    Returns ``None`` to drop the envelope, or a (possibly rewritten) dict to
    keep. Text-only self-duplicates of ``original`` are dropped. A merged
    slot ``"{original}\\n\\n{later}"`` (from ``_enqueue_prompt``'s consecutive
    text merge) is rewritten to just ``later`` so a later correction is not
    lost and the original is not re-fired (#84417). Image-bearing envelopes
    are left alone — their chronology/ownership is load-bearing.
    """
    if not original or not isinstance(entry, dict):
        return entry if isinstance(entry, dict) else None
    if entry.get("image_paths"):
        return entry
    text = entry.get("text")
    if not isinstance(text, str):
        return entry
    stripped = text.strip()
    if not stripped:
        return None
    if stripped == original:
        return None
    # Lossless text-merge glued the live original onto a later follow-up.
    for sep in ("\n\n", "\n"):
        prefix = original + sep
        if text.startswith(prefix):
            rest = text[len(prefix) :].strip()
            if not rest or rest == original:
                return None
            cleaned = dict(entry)
            cleaned["text"] = rest
            return cleaned
    return entry


def _drop_queued_duplicates_of_inflight_user(session: dict) -> None:
    """Remove server-queue copies of the live turn's original user text.

    A mid-turn ``prompt.submit`` of the same text can land in
    ``queued_prompt`` when redirect is not yet available (model not active,
    build window, tool boundary). If the user then corrects the turn with a
    different prompt via redirect, that stale self-duplicate must not
    ``_drain_queued_prompt`` after the redirected turn completes — otherwise
    the original prompt restarts as a fresh agent turn (#84417).

    Unrelated follow-ups (different text, image-bearing envelopes) stay.
    Merged ``original + later`` slots are rewritten to ``later`` only.
    """
    turn = session.get("inflight_turn")
    if not isinstance(turn, dict):
        return
    original = str(turn.get("user") or "").strip()
    if not original:
        return

    head = session.get("queued_prompt")
    rest = list(session.get("queued_prompts") or [])
    kept: list[dict] = []
    for entry in ([head] if head else []) + rest:
        cleaned = _sanitize_queued_entry_vs_inflight_user(entry, original)
        if cleaned is not None:
            kept.append(cleaned)

    if not kept:
        session["queued_prompt"] = None
        session.pop("queued_prompts", None)
        return
    session["queued_prompt"] = kept[0]
    if len(kept) > 1:
        session["queued_prompts"] = kept[1:]
    else:
        session.pop("queued_prompts", None)


def _interrupt_busy_session(sid: str, session: dict, agent: Any) -> None:
    """Interrupt a busy turn without blocking the RPC reader or session lock.

    Some providers cannot apply ``interrupt()`` until a synchronous tool or
    network call returns. Running that call inline used to leave
    ``prompt.submit`` holding ``history_lock`` for the whole wait, which in turn
    blocked ``session.resume`` and delayed the queued prompt itself. Keep at
    most one interrupt worker per session so repeated steering cannot leak an
    unbounded number of blocked threads.
    """
    use_agent = agent is not None and hasattr(agent, "interrupt")
    use_compute_host = not use_agent and _session_uses_compute_host(session)
    if not use_agent and not use_compute_host:
        return

    with session["history_lock"]:
        if session.get("_busy_interrupt_pending"):
            return
        session["_busy_interrupt_pending"] = True

    def interrupt() -> None:
        try:
            if use_agent:
                agent.interrupt()
            else:
                _get_compute_host_supervisor().interrupt(sid)
        except Exception:
            pass
        finally:
            with session["history_lock"]:
                session["_busy_interrupt_pending"] = False

    threading.Thread(target=interrupt, daemon=True, name=f"busy-interrupt-{sid}").start()


def _handle_busy_submit(
    rid, sid: str, session: dict, text: Any, transport: Any, queued: bool = False
) -> dict | None:
    """Apply the ``display.busy_input_mode`` policy to a prompt that lands while
    a turn is in flight, instead of rejecting it with ``session busy``.

    The old rejection forced clients into a deadline-bounded busy-retry that
    silently dropped the send when turn teardown outlived the deadline. The
    default policy now redirects a capable core agent in place; older agents
    retain the proven interrupt-and-queue path drained from ``run``'s tail.

    Modes: ``interrupt`` (default) → redirect the live turn, falling back to
    hard interrupt + queue for older agents; ``queue`` → queue without
    interrupting; ``steer`` → inject after the current atomic action.

    ``queued=True`` (client's queue drain, ``prompt.submit`` param) overrides
    the mode entirely: the message was explicitly queued as "run after", so it
    must NEVER become a live-turn correction or interrupt. Without this, a
    drain that loses the settle race (client observed idle, server still
    unwinding the turn) redirected the live turn with next-turn text — queue
    semantics betrayed by a millisecond race the user can't see.
    """
    mode = "queue" if queued else _load_busy_input_mode()
    agent = session.get("agent")
    with session["history_lock"]:
        if not session.get("running"):
            # The turn ended between prompt.submit's first busy check and this
            # helper. Let the caller retry and claim the now-idle session.
            return None
    with session["history_lock"]:
        if not session.get("running"):
            return None
        image_paths = list(session.get("attached_images", []))
        if image_paths:
            # Claim at submission time. A later paste must not be consumed by
            # this prompt after the active turn finally yields.
            session["attached_images"] = []
    text_only = not image_paths and _is_text_only_busy_payload(text)
    plain_text = _coerce_message_text(text).strip() if text_only else ""
    if mode == "steer" and text_only and plain_text and agent is not None and hasattr(agent, "steer"):
        try:
            if agent.steer(plain_text):
                with session["history_lock"]:
                    _record_inflight_correction(session, plain_text)
                    _drop_queued_duplicates_of_inflight_user(session)
                    session["last_active"] = time.time()
                return _ok(rid, {"status": "steered"})
        except Exception:
            pass  # fall through to queue
    # Text-only corrections redirect the live turn in place when the runtime
    # supports it; media/attachment payloads and older agents fall through to
    # the proven interrupt + queue path below.
    if (
        mode == "interrupt"
        and text_only
        and plain_text
        and agent is not None
        and getattr(agent, "_supports_active_turn_redirect", False) is True
        and hasattr(agent, "redirect")
    ):
        try:
            if agent.redirect(plain_text):
                with session["history_lock"]:
                    _record_inflight_correction(session, plain_text)
                    # #84417: do not re-fire the live turn's original user text
                    # from a stale server-queue self-duplicate after settle.
                    _drop_queued_duplicates_of_inflight_user(session)
                    session["last_active"] = time.time()
                return _ok(rid, {"status": "redirected"})
        except Exception:
            pass  # preserve the proven interrupt + queue fallback below
    # Queue before asking the live turn to stop. In particular, never call a
    # provider or compute-host method while holding history_lock: an interrupt
    # can wait behind the very operation it is trying to cancel.
    with session["history_lock"]:
        if not session.get("running"):
            if image_paths:
                session["attached_images"] = image_paths + list(session.get("attached_images", []))
            return None
        _enqueue_prompt(session, text, transport, image_paths=image_paths)
        session["last_active"] = time.time()

    # Attachments need a separate model invocation. Queue them without
    # cancelling the active turn so the user gets both results in order.
    #
    # #86134: ``steer`` mode must NEVER escalate to a hard interrupt. A burst
    # of user messages while the agent is busy can land as a mix of accepted
    # steers (stashed in ``AIAgent._pending_steer``) and fall-through queue
    # envelopes (payload not steerable, ``steer()`` rejected/raised). A hard
    # interrupt here kills the live turn AND ``AIAgent.interrupt()`` drops
    # the pending steer buffer — silently destroying the earlier messages of
    # the burst. Steer-mode fall-throughs keep queue semantics: preserved
    # FIFO in ``queued_prompt``/``queued_prompts`` and drained on turn end.
    if mode == "interrupt" and not image_paths:
        _interrupt_busy_session(sid, session, agent)
    return _ok(rid, {"status": "queued"})


def _drain_queued_prompt(rid, sid: str, session: dict) -> bool:
    """Fire a queued next-turn prompt if one is waiting and the session is idle.

    Returns True if a queued prompt was dispatched (the caller should then skip
    lower-priority follow-ups this cycle — the user's message wins). Mirrors the
    claim-under-lock pattern used by the goal-continuation re-fire.
    """
    with session["history_lock"]:
        if session.get("_closing"):
            return False
        queued = session.get("queued_prompt")
        if not queued or session.get("running"):
            return False
        queue_generation = int(session.get("_queued_prompt_generation", 0))
        queued_prompts = session.get("queued_prompts") or []
        session["queued_prompt"] = queued_prompts.pop(0) if queued_prompts else None
        if not queued_prompts:
            session.pop("queued_prompts", None)
        session["running"] = True
        if queued.get("transport") is not None:
            session["transport"] = queued["transport"]
    use_compute_host = _session_uses_compute_host(session)
    with session["history_lock"]:
        if int(session.get("_queued_prompt_generation", 0)) != queue_generation:
            # Generation cancelled the claim (Stop, compress re-anchor, …).
            # Do not dispatch — but put the claimed envelope back so a
            # legitimate follow-up is not silently dropped. Order: claimed
            # head first, then whatever advanced into the slot while we held
            # the claim (#84417 belt accuracy).
            rest: list = []
            advanced = session.get("queued_prompt")
            if advanced:
                rest.append(advanced)
            rest.extend(session.get("queued_prompts") or [])
            session["queued_prompt"] = queued
            if rest:
                session["queued_prompts"] = rest
            else:
                session.pop("queued_prompts", None)
            session["running"] = False
            return True
    dispatch_failed = False
    try:
        if use_compute_host:
            if queued.get("image_paths"):
                resp = _submit_prompt_to_compute_host(
                    rid,
                    sid,
                    session,
                    queued["text"],
                    image_paths=queued["image_paths"],
                    queued_prompt_generation=queue_generation,
                )
            else:
                resp = _submit_prompt_to_compute_host(
                    rid, sid, session, queued["text"], queued_prompt_generation=queue_generation
                )
            if resp.get("error"):
                message = str(((resp.get("error") or {}).get("message")) or "queued prompt failed")
                with session["history_lock"]:
                    session["running"] = False
                    _clear_inflight_turn(session)
                _emit("error", sid, {"message": message})
                dispatch_failed = True
        else:
            if queued.get("image_paths"):
                _run_prompt_submit(
                    rid,
                    sid,
                    session,
                    queued["text"],
                    image_paths=queued["image_paths"],
                    queued_prompt_generation=queue_generation,
                )
            else:
                _run_prompt_submit(
                    rid,
                    sid,
                    session,
                    queued["text"],
                    queued_prompt_generation=queue_generation,
                )
    except Exception as exc:
        print(
            f"[tui_gateway] queued prompt dispatch failed: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        with session["history_lock"]:
            session["running"] = False
        dispatch_failed = True
    if dispatch_failed:
        with session["history_lock"]:
            drain_next = bool(session.get("queued_prompt")) and not session.get(
                "_turn_cancel_requested"
            )
        if drain_next:
            _drain_queued_prompt(rid, sid, session)
    return True


def _inflight_snapshot(session: dict) -> dict | None:
    turn = session.get("inflight_turn")
    if not isinstance(turn, dict):
        return None
    user = str(turn.get("user") or "").strip()
    assistant = str(turn.get("assistant") or "")
    streaming = bool(turn.get("streaming"))
    error = str(turn.get("error") or "").strip()
    if not user and not assistant and not streaming and not error:
        return None
    snapshot = {
        "assistant": assistant,
        "streaming": streaming,
        "user": user,
    }
    raw_corrections = turn.get("corrections") or []
    raw_offsets = turn.get("correction_offsets") or []
    correction_pairs = [
        (str(c), raw_offsets[i] if i < len(raw_offsets) else None)
        for i, c in enumerate(raw_corrections)
        if str(c).strip()
    ]
    if correction_pairs:
        # Mid-turn redirects. Carried alongside the original prompt (not over
        # it) so resume can rebuild every user bubble the turn produced.
        snapshot["corrections"] = [c for c, _ in correction_pairs]
        # Assistant-text lengths at each correction boundary (parallel list).
        # Only sent when every correction has one, so clients can trust the
        # pairing; older in-memory turns without offsets omit the field and
        # clients fall back to placing corrections after the assistant dump.
        if all(isinstance(offset, int) and offset >= 0 for _, offset in correction_pairs):
            snapshot["correction_offsets"] = [int(offset) for _, offset in correction_pairs]  # type: ignore[arg-type]
    if error:
        # Retained failed turn (see _fail_inflight_turn): carry the error
        # semantics so a resuming client can rebuild the failed-turn bubble
        # instead of rendering the partial text as a healthy reply.
        snapshot["error"] = error
        snapshot["status"] = str(turn.get("status") or "error")
        snapshot["recoverable"] = bool(turn.get("recoverable"))
        surface = turn.get("error_surface")
        if isinstance(surface, dict) and surface:
            snapshot["error_surface"] = surface
    return snapshot


def _emit_terminal_turn_error(
    sid: str,
    session: dict,
    error: Any,
    error_surface: Optional[dict] = None,
    *,
    retire_marker: bool = True,
) -> None:
    """Close a failed turn with a terminal ``message.complete`` frame.

    Emits the same ``status: "error"`` frame shape the returned-error path in
    ``_run_prompt_submit`` already produces (so TUI/desktop handling is
    uniform), and retains the failed turn via ``_fail_inflight_turn`` so a
    client that missed this frame (disconnect window) can recover it from
    ``session.resume``'s ``inflight`` payload.

    ``error_surface`` lets callers that already know the failing layer (e.g.
    agent-init failures = local runtime) pass it explicitly; exception
    callers leave it None and the classifier derives it here.
    """
    agent = session.get("agent")
    # Classify the failure into a {layer, code, retryable} descriptor so the
    # desktop can say "Provider error" / "Gateway error" with matching
    # recovery actions instead of a generic toast. Never raises (advisory).
    if error_surface is None and isinstance(error, BaseException):
        try:
            from agent.error_surface import build_error_surface_from_exception

            error_surface = build_error_surface_from_exception(
                error,
                provider=str(getattr(agent, "provider", "") or ""),
                model=str(getattr(agent, "model", "") or ""),
            )
        except Exception:
            error_surface = None
    with session["history_lock"]:
        _fail_inflight_turn(session, error, error_surface=error_surface)
        turn = session.get("inflight_turn") or {}
        message = str(turn.get("error") or "turn failed")
        partial = str(turn.get("assistant") or "")
        cols = int(session.get("cols", 80))
    text = partial or f"Error: {message}"
    payload = {
        "text": text,
        "usage": _get_usage(agent) if agent is not None else {},
        "status": "error",
        "error": message,
        "recoverable": True,
    }
    if error_surface:
        payload["error_surface"] = error_surface
    if partial:
        payload["partial"] = True
    try:
        rendered = render_message(text, cols)
    except Exception:
        rendered = ""
    if rendered:
        payload["rendered"] = rendered
    if retire_marker:
        _retire_turn_marker(session)
    _emit("message.complete", sid, payload)


def _restore_agent_history_after_turn_error(session: dict, agent) -> bool:
    """Keep a failed turn's working transcript in the gateway session.

    ``AIAgent`` persists its working messages independently of the gateway's
    history snapshot. If the turn raises after that persistence, the next
    prompt must see the working transcript instead of the pre-turn snapshot.
    """
    agent_messages = getattr(agent, "_session_messages", None)
    if not isinstance(agent_messages, list):
        return False
    with session["history_lock"]:
        session["history"] = list(agent_messages)
        session["history_version"] = int(session.get("history_version", 0)) + 1
    return True


def _queued_prompt_snapshot(session: dict) -> dict | None:
    """Return the accepted next-turn prompt without its transport handle.

    A busy ``prompt.submit`` lives only in ``session["queued_prompt"]`` until
    the current turn winds down. Desktop may reconnect or restart during that
    window, so the live-session projection must carry the user-visible text;
    otherwise the accepted prompt disappears until it finally drains.
    """
    queued = session.get("queued_prompt")
    if not isinstance(queued, dict):
        return None
    user = _inflight_text(queued.get("text"))
    return {"user": user} if user else None


def register(server) -> None:
    """Publish this module's helpers + handlers onto ``server``, rebound to its globals."""
    bind_module(globals(), server, skip=("_",))
