"""Auto-continue: resume a turn killed by a process/machine death, plus queued-prompt drain and busy-submit handling.

Bodies are rebound onto server.py's globals at install time (see
method_ctx.bind_module), so they reference server.py globals bare.
"""

from __future__ import annotations


from .method_ctx import HandlerRegistry, bind_module

_registry = HandlerRegistry()


# A concluded turn (success, handled error, interrupt) clears its durable marker
# (turn_marker.py) in _run_prompt_submit's finally; only a process death leaves it
# behind, so a marker at session.resume proves the turn never finished AND the
# client never saw a terminal frame. Fresh: re-submit automatically (as the
# messaging gateway does). Stale: clear it and let the partial transcript speak.

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

    Called right before the terminal frame, not at turn-thread end: post-turn work
    (titles, memory sync, goal hooks) outlives the client's answer, and quitting in
    that window would leave a marker that re-runs a finished turn on next launch.
    Extra ``keys`` cover a session_key that compression rotated mid-turn.
    """
    home = _session_home(session)
    for key in dict.fromkeys((*keys, str(session.get("session_key") or ""))):
        if key:
            clear_turn_marker(home, key)


def _auto_continue_note(prompt: str) -> str:
    # Same opening as the gateway's recovery notes (transcript tooling recognizes
    # both). The prompt is embedded: a hard crash persists nothing else of the turn.
    return (
        f"{_AUTO_CONTINUE_NOTE_PREFIX} — the app or its backend process "
        "stopped before the turn could finish. Some of the work may already "
        "be complete; check the current state before redoing anything, then "
        "finish the task. The interrupted request was:]\n\n"
        f"{prompt}"
    )


def _ac_release_turn(session: dict, *, unschedule: bool = False) -> None:
    with session["history_lock"]:
        session["running"] = False
        if unschedule:
            session["_auto_continue_scheduled"] = False


def _maybe_schedule_auto_continue(sid: str, session: dict, session_key: str) -> dict | None:
    """Kick off a continuation turn for a crash-interrupted session.

    Called from session.resume's cold paths once the live record is registered.
    Returns a descriptor for the resume payload when scheduled, else None. The turn
    runs on a background thread after the deferred agent build via the normal
    _run_prompt_submit path, so the client that just resumed streams it live.
    """
    # Hosted room turns are recovered by their durable task/lease state machine;
    # generic auto-continue would bypass its execution generation and duplicate work.
    if session.get("source") == "bot_room":
        return None

    home = _session_home(session)
    marker = read_turn_marker(home, session_key)
    if marker is None:
        return None
    enabled, freshness_secs, max_attempts = _auto_continue_config()
    age = time.time() - marker["started_at"]
    if not enabled or age > freshness_secs or marker["attempts"] >= max_attempts:
        # Stale, disabled, or crash-looping: stop trying; a manual message continues.
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
                # A real user prompt beat us; its own conclusion clears the marker.
                session["_auto_continue_scheduled"] = False
                return
            session["running"] = True
            session["last_active"] = time.time()
        # Ownership admission BEFORE message.start: a sibling backend sharing this
        # HERMES_HOME may have written the marker and still be mid-turn. Leave the
        # marker so a later resume retries once the owner finishes or dies.
        if _ensure_active_session_slot(sid, session) is not None:
            logger.info("auto-continue for %s refused: session has another live owner", session_key)
            _ac_release_turn(session, unschedule=True)
            return
        with session["history_lock"]:
            # Marker inputs read back by _run_prompt_submit: count the attempt (crash
            # breaker) and re-record the ORIGINAL prompt (no nested notes). Set here,
            # not at schedule time, so a bail above leaves nothing for a racing user turn.
            session["_auto_continue_attempt"] = attempt
            session["_auto_continue_prompt"] = marker["prompt"]
        try:
            _emit("status.update", sid, {"kind": "process", "text": "Resuming interrupted turn…"})
            _emit("message.start", sid)
            _run_prompt_submit(rid, sid, session, text, display_kind="auto_continue")
        except Exception as exc:
            print(f"[tui_gateway] auto-continue dispatch failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            _ac_release_turn(session)

    threading.Thread(target=kickoff, daemon=True).start()
    logger.info(
        "auto-continue scheduled for session %s (attempt %d, interrupted %.0fs ago)",
        session_key, attempt, age,
    )
    return {"attempt": attempt, "interrupted_at": marker["started_at"]}


def _enqueue_prompt(session: dict, text: Any, transport: Any, image_paths: list[str] | None = None) -> None:
    """Stash a message to run as the very next turn once the live one ends.

    Text-only arrivals share a slot and merge losslessly (like the consecutive-user
    merge in ``repair_message_sequence``); image-bearing ones stay separate envelopes
    so attachment ownership/chronology survive. ``transport`` is pinned so the drained
    turn streams back to its sender even if the session transport is rebound.
    """
    image_paths = list(image_paths or [])
    # Scrub live-turn self-duplicates first so the text merge below can't glue
    # "{original}\n\n{later}" and re-fire the original after a correction settles.
    _drop_queued_duplicates_of_inflight_user(session)
    # Never queue a text-only self-copy of the live prompt: draining it would restart it.
    if not image_paths and isinstance(text, str):
        turn = session.get("inflight_turn")
        original = (str(turn.get("user") or "").strip() if isinstance(turn, dict) else "")
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


def _sanitize_queued_entry_vs_inflight_user(entry: Any, original: str) -> dict | None:
    """Drop (``None``) or rewrite a queue envelope that re-carries the live user text.

    Text-only self-duplicates are dropped; a merged slot ``"{original}\\n\\n{later}"``
    is rewritten to ``later`` so the correction survives without re-firing the
    original. Image-bearing envelopes are left alone (chronology is load-bearing).
    """
    if not original or not isinstance(entry, dict):
        return entry if isinstance(entry, dict) else None
    if entry.get("image_paths"):
        return entry
    text = entry.get("text")
    if not isinstance(text, str):
        return entry
    stripped = text.strip()
    if not stripped or stripped == original:
        return None
    # Lossless text-merge glued the live original onto a later follow-up.
    for sep in ("\n\n", "\n"):
        prefix = original + sep
        if text.startswith(prefix):
            rest = text[len(prefix) :].strip()
            if not rest or rest == original:
                return None
            return {**entry, "text": rest}
    return entry


def _drop_queued_duplicates_of_inflight_user(session: dict) -> None:
    """Remove server-queue copies of the live turn's original user text.

    A mid-turn ``prompt.submit`` of the same text can be queued while redirect is
    unavailable (build window, tool boundary); after a later redirect it must not
    drain and restart the original as a fresh turn. Unrelated follow-ups stay.
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

    _ac_set_queue(session, kept)


def _ac_set_queue(session: dict, entries: list) -> None:
    """Write ``entries`` back as queued_prompt (head) + queued_prompts (rest)."""
    session["queued_prompt"] = entries[0] if entries else None
    if len(entries) > 1:
        session["queued_prompts"] = entries[1:]
    else:
        session.pop("queued_prompts", None)


def _interrupt_busy_session(sid: str, session: dict, agent: Any) -> None:
    """Interrupt a busy turn on a worker thread, never under ``history_lock``.

    Some providers can't apply ``interrupt()`` until a blocking tool/network call
    returns; doing it inline stalled ``session.resume`` and the queued prompt.
    At most one interrupt worker per session so repeated steering can't leak threads.
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


def _ac_record_inflight_correction(session: dict, plain_text: str) -> None:
    """Record an accepted steer/redirect; scrub stale self-duplicates so the
    live turn's original text is not re-fired from the queue after settle."""
    with session["history_lock"]:
        _record_inflight_correction(session, plain_text)
        _drop_queued_duplicates_of_inflight_user(session)
        session["last_active"] = time.time()


def _handle_busy_submit(
    rid, sid: str, session: dict, text: Any, transport: Any, queued: bool = False
) -> dict | None:
    """Apply ``display.busy_input_mode`` to a prompt that lands mid-turn instead of
    rejecting it with ``session busy`` (rejection made clients busy-retry and
    silently drop sends when teardown outlived their deadline).

    Modes: ``interrupt`` (default) → redirect the live turn, falling back to hard
    interrupt + queue for older agents; ``queue`` → queue only; ``steer`` → inject
    after the current atomic action. ``queued=True`` (client queue drain) forces
    queue mode: a "run after" message must NEVER become a live-turn correction,
    even when the drain loses the settle race against a still-unwinding turn.
    """
    mode = "queue" if queued else _load_busy_input_mode()
    agent = session.get("agent")
    with session["history_lock"]:
        if not session.get("running"):
            # Turn ended since prompt.submit's busy check; caller retries on the idle session.
            return None
        image_paths = list(session.get("attached_images", []))
        if image_paths:
            # Claim now so a later paste isn't consumed by this prompt when the turn yields.
            session["attached_images"] = []
    text_only = not image_paths and _is_text_only_busy_payload(text)
    plain_text = _coerce_message_text(text).strip() if text_only else ""
    if mode == "steer" and text_only and plain_text and agent is not None and hasattr(agent, "steer"):
        try:
            if agent.steer(plain_text):
                _ac_record_inflight_correction(session, plain_text)
                return _ok(rid, {"status": "steered"})
        except Exception:
            pass  # fall through to queue
    # Text-only corrections redirect in place when supported; media payloads and
    # older agents fall through to the proven interrupt + queue path.
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
                _ac_record_inflight_correction(session, plain_text)
                return _ok(rid, {"status": "redirected"})
        except Exception:
            pass  # preserve the proven interrupt + queue fallback below
    # Queue before asking the live turn to stop. Never call a provider/compute-host
    # method under history_lock: an interrupt can wait behind the op it cancels.
    with session["history_lock"]:
        if not session.get("running"):
            if image_paths:
                session["attached_images"] = image_paths + list(session.get("attached_images", []))
            return None
        _enqueue_prompt(session, text, transport, image_paths=image_paths)
        session["last_active"] = time.time()

    # Attachments need their own model invocation: queue without cancelling so the
    # user gets both results in order. ``steer`` must NEVER escalate to a hard
    # interrupt: it would kill the live turn AND drop ``AIAgent._pending_steer``,
    # destroying earlier accepted steers; steer fall-throughs stay FIFO-queued.
    if mode == "interrupt" and not image_paths:
        _interrupt_busy_session(sid, session, agent)
    return _ok(rid, {"status": "queued"})


def _drain_queued_prompt(rid, sid: str, session: dict) -> bool:
    """Fire a queued next-turn prompt if one is waiting and the session is idle.

    True when dispatched: the caller skips lower-priority follow-ups this cycle
    (the user's message wins). Claim-under-lock like the goal-continuation re-fire.
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
            # Generation bump cancelled the claim (Stop, compress re-anchor, …): don't
            # dispatch, but restore the envelope (claimed head first, then whatever
            # advanced into the slot) so a legitimate follow-up isn't dropped.
            rest: list = []
            advanced = session.get("queued_prompt")
            if advanced:
                rest.append(advanced)
            rest.extend(session.get("queued_prompts") or [])
            _ac_set_queue(session, [queued, *rest])
            session["running"] = False
            return True
    kwargs: dict = {"queued_prompt_generation": queue_generation}
    if queued.get("image_paths"):
        kwargs["image_paths"] = queued["image_paths"]
    dispatch_failed = False
    try:
        if use_compute_host:
            resp = _submit_prompt_to_compute_host(rid, sid, session, queued["text"], **kwargs)
            if resp.get("error"):
                message = str(((resp.get("error") or {}).get("message")) or "queued prompt failed")
                with session["history_lock"]:
                    session["running"] = False
                    _clear_inflight_turn(session)
                _emit("error", sid, {"message": message})
                dispatch_failed = True
        else:
            _run_prompt_submit(rid, sid, session, queued["text"], **kwargs)
    except Exception as exc:
        print(f"[tui_gateway] queued prompt dispatch failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        _ac_release_turn(session)
        dispatch_failed = True
    if dispatch_failed:
        with session["history_lock"]:
            drain_next = bool(session.get("queued_prompt")) and not session.get("_turn_cancel_requested")
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
    snapshot = {"assistant": assistant, "streaming": streaming, "user": user}
    raw_corrections = turn.get("corrections") or []
    raw_offsets = turn.get("correction_offsets") or []
    correction_pairs = [
        (str(c), raw_offsets[i] if i < len(raw_offsets) else None)
        for i, c in enumerate(raw_corrections)
        if str(c).strip()
    ]
    if correction_pairs:
        # Mid-turn redirects alongside (not over) the original prompt so resume can
        # rebuild every user bubble; offsets only when every correction has one so
        # clients can trust the pairing.
        snapshot["corrections"] = [c for c, _ in correction_pairs]
        if all(isinstance(offset, int) and offset >= 0 for _, offset in correction_pairs):
            snapshot["correction_offsets"] = [int(offset) for _, offset in correction_pairs]  # type: ignore[arg-type]
    if error:
        # Retained failed turn (_fail_inflight_turn): a resuming client must rebuild
        # the failed bubble, not render the partial text as a healthy reply.
        snapshot["error"] = error
        snapshot["status"] = str(turn.get("status") or "error")
        snapshot["recoverable"] = bool(turn.get("recoverable"))
        surface = turn.get("error_surface")
        if isinstance(surface, dict) and surface:
            snapshot["error_surface"] = surface
    return snapshot


def _emit_terminal_turn_error(
    sid: str, session: dict, error: Any, error_surface: Optional[dict] = None, *, retire_marker: bool = True
) -> None:
    """Close a failed turn with the same ``status: "error"`` ``message.complete``
    frame as ``_run_prompt_submit``'s returned-error path, retaining the turn via
    ``_fail_inflight_turn`` so a client that missed the frame recovers it from
    ``session.resume``'s ``inflight``. Callers that know the failing layer pass
    ``error_surface``; exception callers leave it None and it is classified here.
    """
    agent = session.get("agent")
    # {layer, code, retryable} descriptor so the desktop can say "Provider error" /
    # "Gateway error" with matching recovery actions. Advisory: never raises.
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
    """Keep a failed turn's working transcript: ``AIAgent`` persists its messages
    independently, so after a raise the next prompt must see them, not the
    pre-turn snapshot."""
    agent_messages = getattr(agent, "_session_messages", None)
    if not isinstance(agent_messages, list):
        return False
    with session["history_lock"]:
        session["history"] = list(agent_messages)
        session["history_version"] = int(session.get("history_version", 0)) + 1
    return True


def _queued_prompt_snapshot(session: dict) -> dict | None:
    """The accepted next-turn prompt without its transport handle, for the
    live-session projection (Desktop may reconnect while it is still queued)."""
    queued = session.get("queued_prompt")
    if not isinstance(queued, dict):
        return None
    user = _inflight_text(queued.get("text"))
    return {"user": user} if user else None


def register(server) -> None:
    """Publish this module's helpers + handlers onto ``server``, rebound to its globals."""
    bind_module(globals(), server, skip=("_",))
