"""The prompt turn: ``_run_prompt_submit`` and the per-phase helpers it drives.

Bodies are rebound onto server.py's globals at install time (method_ctx.bind_module),
so they reference server.py globals bare.  Turn shape (one fresh daemon thread):
admit -> crash marker -> bind scopes -> resolve message -> run_conversation ->
commit history / message.complete -> goal & loop hooks -> release scopes ->
post-turn follow-ups (queued prompt, goal continuation, notifications).
"""

from __future__ import annotations

import dataclasses

from .method_ctx import HandlerRegistry, bind_module

_registry = HandlerRegistry()


def _hook_failure(what: str, exc: BaseException) -> None:
    print(f"[tui_gateway] {what} failed: {type(exc).__name__}: {exc}", file=sys.stderr)


def _is_successful_goal_turn(result: Any, status: str, raw: Any) -> bool:
    """Whether a turn produced a real response the goal judge can use."""
    return bool(
        status == "complete" and isinstance(raw, str) and raw.strip()
        and not (isinstance(result, dict) and result.get("failed"))
        and not (isinstance(result, dict) and result.get("completed") is False))


def _goal_max_turns() -> int:
    try:
        goals_cfg = _load_cfg().get("goals") or {}
        return int(goals_cfg.get("max_turns", 20) or 20)
    except Exception:
        return 20


def _plan_goal_compression_recovery(
    session: dict, result: Any, *, status: str, raw: Any) -> tuple[str | None, str | None]:
    """Plan a bounded active-goal retry after compression exhaustion.

    Exhaustion is a failed turn: never judge input, never a spent goal turn.  One
    fresh continuation is allowed; if that also exhausts, pause the goal instead of
    spinning until a random user message wakes it.  Returns
    ``(continuation_prompt, status_notice)``; no active goal -> ``(None, None)``.
    """
    if not (isinstance(result, dict) and result.get("compression_exhausted")):
        if _is_successful_goal_turn(result, status, raw):
            session.pop(_GOAL_COMPRESSION_RECOVERY_ATTEMPTS, None)
        return None, None
    from hermes_cli.goals import GoalManager
    sid_key = str(session.get("session_key") or "")
    if not sid_key:
        return None, None
    goal_mgr = GoalManager(session_id=sid_key, default_max_turns=_goal_max_turns())
    if not goal_mgr.is_active():
        session.pop(_GOAL_COMPRESSION_RECOVERY_ATTEMPTS, None)
        return None, None
    goal_created_at = float(getattr(goal_mgr.state, "created_at", 0.0) or 0.0)
    goal_text = getattr(goal_mgr.state, "goal", "")
    recovery_state = session.get(_GOAL_COMPRESSION_RECOVERY_ATTEMPTS)
    attempts = 0
    if (
        isinstance(recovery_state, dict)
        and recovery_state.get("goal_created_at") == goal_created_at
        and recovery_state.get("goal") == goal_text):
        with contextlib.suppress(TypeError, ValueError):
            attempts = int(recovery_state.get("attempts", 0) or 0)
    continuation_prompt = goal_mgr.next_continuation_prompt()
    if attempts < _GOAL_COMPRESSION_RECOVERY_LIMIT and continuation_prompt:
        session[_GOAL_COMPRESSION_RECOVERY_ATTEMPTS] = {
            "goal_created_at": goal_created_at, "goal": goal_text, "attempts": attempts + 1}
        return (
            continuation_prompt,
            "Context compression was exhausted. Retrying the active goal once.")
    goal_mgr.pause(reason="context compression exhausted twice consecutively")
    # A later explicit /goal resume gets a fresh bounded recovery cycle.
    session.pop(_GOAL_COMPRESSION_RECOVERY_ATTEMPTS, None)
    return (
        None,
        "Goal paused after context compression was exhausted twice. "
        "Run /compress, then /goal resume to continue.")


def _admit_prompt_turn(
    sid: str, session: dict, text: Any, image_paths: list[str] | None,
    queued_prompt_generation: int | None) -> tuple[list[str], Any] | None:
    """Ownership + liveness gate every fresh turn source must cross.

    prompt.submit claims the slot in its RPC handler, but auto-continue, wake-ups
    and other synthesized turns call ``_run_prompt_submit`` directly — the bypass
    that once let a second backend run a duplicate turn.  Returns
    ``(images, agent)`` or None when refused (``running`` already reset).
    """
    if (ownership_refusal := _ensure_active_session_slot(sid, session)) is not None:
        logger.info(
            "Refusing turn for session %s at _run_prompt_submit: %s",
            session.get("session_key") or sid,
            getattr(ownership_refusal, "reason", None) or "refused")
        with session["history_lock"]:
            session["running"] = False
        _emit("error", sid, {"message": str(ownership_refusal)})
        return None
    with session["history_lock"]:
        if session.get("_closing") or (
            queued_prompt_generation is not None
            and int(session.get("_queued_prompt_generation", 0)) != queued_prompt_generation):
            session["running"] = False
            return None
        if image_paths is None:
            images = list(session.get("attached_images", []))
            session["attached_images"] = []
        else:
            images = list(image_paths)
        inflight = session.get("inflight_turn")
        # A retained failed turn (see _fail_inflight_turn) is a stale leftover
        # by the time a new turn starts — replace it, never append onto it.
        if not isinstance(inflight, dict) or inflight.get("status") == "error":
            _start_inflight_turn(session, text)
        agent = session["agent"]
        if hasattr(agent, "clear_interrupt"):
            with contextlib.suppress(Exception):
                agent.clear_interrupt()
    return images, agent


def _record_turn_marker(session: dict, text: Any) -> str:
    """Write the durable crash marker; returns the session key it was written under.

    Retired when the outcome reaches the client; a surviving marker means the
    process died mid-turn and session.resume auto-continues from it.  Compression
    can rotate session_key mid-turn, so the caller keeps this key.  The key is
    published before the disk write so an interrupt racing startup can retire it;
    the post-write cancel check closes the inverse race (Stop landed first, no
    file to clear yet).
    """
    marker_home = _session_home(session)
    marker_key = str(session.get("session_key") or "")
    marker_attempt = int(session.pop("_auto_continue_attempt", 0) or 0)
    marker_text = session.pop("_auto_continue_prompt", None) or text
    if isinstance(marker_text, str) and marker_text.strip():
        with session["history_lock"]:
            session["_active_turn_marker_key"] = marker_key
        record_turn_start(marker_home, marker_key, marker_text, attempts=marker_attempt)
        with session["history_lock"]:
            marker_cancelled = bool(session.get("_turn_cancel_requested"))
        if marker_cancelled:
            clear_turn_marker(marker_home, marker_key)
    return marker_key


@dataclasses.dataclass(slots=True)
class _TurnScopes:
    """Reset tokens for the thread/context scopes a turn binds (filled incrementally)."""

    approval: Any = None
    session_tokens: list = dataclasses.field(default_factory=list)
    home: Any = None  # per-turn HERMES_HOME override for a resumed remote profile
    secret: Any = None
    terminal: Any = None


def _bind_turn_scopes(sid: str, session: dict, scopes: _TurnScopes) -> None:
    """Bind approval/session/profile/terminal scopes for this turn thread.

    Fills ``scopes`` field by field so a failure midway still leaves every bound
    token for ``_release_turn_scopes``.  The profile's COMPLETE terminal policy is
    bound too: terminal_tool otherwise reads the launch process's pinned env, and
    a failed install leaves a refusal scope so terminal tools fail closed.
    """
    from tools.approval import set_current_session_key
    scopes.approval = set_current_session_key(session["session_key"])
    scopes.session_tokens = _set_session_context(session["session_key"], ui_session_id=sid)
    profile_home = session.get("profile_home")
    if profile_home:
        scopes.home = set_hermes_home_override(profile_home)
        scopes.secret = set_secret_scope(build_profile_secret_scope(Path(profile_home)))
        from tools.terminal_scope import install_profile_terminal_scope
        scopes.terminal = install_profile_terminal_scope(Path(profile_home))
    # The sudo password callback is thread-local: the build thread's wiring doesn't
    # reach this turn thread and sudo prompts would fall through to /dev/tty and
    # hang the headless gateway (secret capture is a module global; re-run is a no-op).
    _wire_callbacks(sid)


def _release_turn_scopes(scopes: _TurnScopes) -> None:
    with contextlib.suppress(Exception):
        if scopes.approval is not None:
            from tools.approval import reset_current_session_key
            reset_current_session_key(scopes.approval)
    if scopes.home is not None:
        reset_hermes_home_override(scopes.home)
    if scopes.secret is not None:
        reset_secret_scope(scopes.secret)
    if scopes.terminal is not None:
        from tools.terminal_scope import reset_terminal_scope
        reset_terminal_scope(scopes.terminal)
    _clear_session_context(scopes.session_tokens)


def _expand_context_references(agent, prompt: str, cwd: str):
    """Expand ``@file`` references; returns the preprocess result (``.blocked``/``.message``)."""
    from agent.context_references import preprocess_context_references
    from agent.model_metadata import get_model_context_length
    ctx_len = get_model_context_length(
        getattr(agent, "model", "") or _resolve_model(),
        base_url=getattr(agent, "base_url", "") or "", api_key=getattr(agent, "api_key", "") or "",
        provider=getattr(agent, "provider", "") or "",
        config_context_length=getattr(agent, "_config_context_length", None))
    return preprocess_context_references(prompt, cwd=cwd, allowed_root=cwd, context_length=ctx_len)


def _route_turn_images(agent, prompt: Any, images: list[str]) -> Any:
    """Build the run message for a turn with attached images.

    "native" passes pixels as OpenAI-style content parts; "text" references the
    paths so the agent analyzes them in-loop with vision_analyze, never blocking
    the submit path on vision calls.  Decision table: agent/image_routing.py.
    """
    try:
        from agent.image_routing import build_native_content_parts, decide_image_input_mode
        from hermes_cli.config import load_config as _tui_load_config
        _provider, _model = _active_image_routing_identity(agent)
        mode = decide_image_input_mode(
            _provider, _model, _tui_load_config(),
            requested_provider=getattr(agent, "requested_provider", ""))
        if getattr(agent, "api_mode", "") == "codex_app_server":
            mode = "text"
    except Exception as _img_exc:
        print(f"[tui_gateway] image_routing decision failed, defaulting to text: {_img_exc}",
              file=sys.stderr)
        mode = "text"
    if mode != "native":
        return _build_image_ref_message(prompt, images)
    try:
        parts, skipped = build_native_content_parts(prompt, images)
        if skipped:
            print(
                f"[tui_gateway] native image attachment skipped {len(skipped)} unreadable path(s)",
                file=sys.stderr)
        if any(p.get("type") == "image_url" for p in parts):
            return parts
    except Exception as _img_exc:
        print(f"[tui_gateway] native attach failed, falling back to text: {_img_exc}",
              file=sys.stderr)
    return _build_image_ref_message(prompt, images)


def _start_turn_voice() -> tuple[Any, bool]:
    """Arm voice-mode turn audio; returns ``(tts_queue, thinking_started)``.

    ``_tts_stream_begin`` goes first: cutting a still-speaking previous turn IS
    this turn's barge-in, so it must latch before the caller consumes the latch.
    The full-duplex listener lets the user interject DURING generation.  The
    "thinking" sound keeps long silences from reading as a dead session; its
    gate skips while TTS plays or the mic captures; stopped in the turn's finally.
    """
    tts_queue = _tts_stream_begin()
    if not _voice_mode_enabled():
        return tts_queue, False
    if _voice_cfg_dict().get("barge_in", True):
        _arm_full_duplex_listener()
    try:
        from tools.voice_mode import is_audio_output_active, start_thinking_sound

        def _thinking_should_play() -> bool:
            if is_audio_output_active():
                return False
            try:
                from hermes_cli.voice import is_continuous_active
                return not is_continuous_active()
            except Exception:
                return True
        return tts_queue, start_thinking_sound(should_play=_thinking_should_play)
    except Exception:
        return tts_queue, False


def _apply_turn_notes(run_message: Any, session: dict) -> Any:
    """Prepend the per-turn API-message notes (same enrichment channel as images):
    barge mid-speech, reactions since the last turn, then which window the message
    was typed into (HUD mode is per-turn state; not for the byte-stable system prompt)."""
    from tools.tts_streaming import SPEECH_INTERRUPTED_NOTE, take_speech_interrupted
    if take_speech_interrupted():
        run_message = _prepend_note(run_message, SPEECH_INTERRUPTED_NOTE)
    run_message = _prepend_note(run_message, _pending_reaction_notes(session))
    return _prepend_note(run_message, _hud_surface_note(session))


def _build_run_kwargs(
    agent, session: dict, history: list, prompt: Any, images: list[str], run_message: Any,
    stream_cb, display_kind: str | None, display_metadata: dict | None) -> dict:
    """Assemble ``run_conversation`` kwargs, feature-detecting optional parameters.

    A synthesized turn is typed at turn START so the crash persist writes its row
    as a timeline event, not a raw user bubble (forever, if the turn never ends).
    The post-turn stamp is the fallback for an older agent; re-stamping is a no-op.
    """
    run_kwargs = {
        "conversation_history": list(history),
        "stream_callback": stream_cb,
        "persist_user_message": (
            _build_persist_user_message(prompt, images, run_message) if images else prompt)}
    try:
        run_params = inspect.signature(agent.run_conversation).parameters
    except (TypeError, ValueError):
        run_params = {}
    if "task_id" in run_params:
        run_kwargs["task_id"] = session["session_key"]
    if display_kind and "persist_user_display_kind" in run_params:
        run_kwargs["persist_user_display_kind"] = display_kind
        run_kwargs["persist_user_display_metadata"] = display_metadata
    return run_kwargs


def _stamp_synthetic_display_kind(
    agent, session: dict, result: Any, text: str, display_kind: str, display_metadata: dict | None
) -> None:
    """Post-turn fallback stamp of a synthesized turn's display kind (DB row + result)."""
    db = getattr(agent, "_session_db", None)
    current_session_id = getattr(agent, "session_id", None) or session.get("session_key")
    if db is not None:
        try:
            db.set_latest_matching_message_display_kind(
                current_session_id, role="user", content=text, display_kind=display_kind,
                display_metadata=display_metadata)
        except Exception:
            logger.debug("failed to stamp synthetic display kind", exc_info=True)
    if isinstance(result, dict) and isinstance(result.get("messages"), list):
        for message in reversed(result["messages"]):
            if message.get("role") == "user" and message.get("content") == text:
                message["display_kind"] = display_kind
                if display_metadata:
                    message["display_metadata"] = display_metadata
                break


def _restore_moa_one_shot(sid: str, session: dict) -> None:
    """Undo a /moa one-shot after its turn — through the switch path, because the
    one-shot did a real in-place ``agent.switch_model()``; resetting
    ``model_override`` alone would leave the live client pinned to MoA."""
    _restore = session.pop("moa_one_shot_restore", None)
    if isinstance(_restore, dict):
        _prev_override = _restore.get("override")
        _prev_model = _restore.get("model")
        _prev_provider = _restore.get("provider")
        if _prev_override is None:
            session.pop("model_override", None)
        else:
            session["model_override"] = _prev_override
        if _prev_model:
            _raw = f"{_prev_model} --provider {_prev_provider}" if _prev_provider else _prev_model
            try:
                _apply_model_switch(
                    sid, session, _raw, confirm_expensive_model=False,
                    pin_session_override=bool(_prev_override),
                    persist_override=False)  # session-internal restore, never config.yaml
            except Exception as _moa_restore_exc:
                logger.warning("MoA one-shot model restore failed: %s", _moa_restore_exc)
    elif _restore is None:
        session.pop("model_override", None)
    else:
        session["model_override"] = _restore


def _commit_turn_history(
    session: dict, result: dict, history: list, history_version: int) -> str | None:
    """Write the agent's messages back to session history; returns a client warning or None.

    Caller holds no lock.  If history_version moved during the turn, the only
    tolerated mutation is a pivot marker the gateway itself inserted mid-turn
    (model switch, /personality); then the output is merged after the current
    history.  ``_append_model_switch_marker`` strips prior markers in place then
    appends, so the delta is NOT a tail slice — compare content, not indices.
    Any other desync (undo/compress/retry/rollback) is surfaced, never dropped.
    """
    with session["history_lock"]:
        current_version = int(session.get("history_version", 0))
        if current_version == history_version:
            session["history"] = result["messages"]
            session["history_version"] = history_version + 1
            return None
        current_history = list(session["history"])
        history_no_markers = [e for e in history if not _is_pivot_marker(e)]
        current_no_markers = [e for e in current_history if not _is_pivot_marker(e)]
        if current_no_markers == history_no_markers and any(
                _is_pivot_marker(e) for e in current_history):
            # Auto-compression can make result["messages"] shorter than the
            # turn-start history; then the full result is the base.
            if len(result["messages"]) > len(history):
                new_messages = result["messages"][len(history):]
            else:
                new_messages = list(result["messages"])
            session["history"] = current_history + new_messages
            session["history_version"] = current_version + 1
            return None
        print(
            f"[tui_gateway] prompt.submit: history_version mismatch "
            f"(expected={history_version} current={current_version}) — "
            f"agent output NOT written to session history",
            file=sys.stderr)
        return (
            "History changed during this turn — the response above is visible "
            "but was not saved to session history.")


def _result_status(result: dict) -> str:
    return (
        "interrupted" if result.get("interrupted")
        else "error" if result.get("error") else "complete")


def _turn_outcome(result: Any) -> tuple[Any, str, str | None]:
    """Reduce a run_conversation result to ``(raw_text, status, last_reasoning)``."""
    if not isinstance(result, dict):
        return str(result), "complete", None
    raw = result.get("final_response", "")
    status = _result_status(result)
    # No visible response AND a real error (e.g. invalid model slug -> provider
    # 4xx): surface the error as the text (classic CLI parity) instead of an
    # empty turn.  An empty successful turn still renders as empty.
    if (not raw) and result.get("error") and (result.get("failed") or result.get("partial")):
        raw = f"Error: {result.get('error')}"
    # "Operation interrupted: waiting for model response (…)" is cancellation
    # metadata, not assistant prose (gateway/run.py and ACP suppress it too).
    if status == "interrupted" and isinstance(raw, str) and raw.strip().startswith(
            INTERRUPT_WAITING_FOR_MODEL_PREFIX):
        raw = ""
    lr = result.get("last_reasoning")
    last_reasoning = lr.strip() if isinstance(lr, str) and lr.strip() else None
    return raw, status, last_reasoning


def _turn_error_surface(agent, result: Any) -> Any:
    """{layer, code, retryable} descriptor for an error result (advisory, never raises)."""
    try:
        from agent.error_surface import build_error_surface_from_result
        return build_error_surface_from_result(
            result, provider=str(getattr(agent, "provider", "") or ""),
            model=str(getattr(agent, "model", "") or ""))
    except Exception:
        return None


def _goal_followup_after_turn(
    sid: str, session: dict, result: Any, status: str, raw: Any) -> str | None:
    """/goal continuation (mirrors gateway/run._post_turn_goal_continuation).

    Asks the judge whether the goal is done and, if not and under budget, returns
    the continuation prompt to chain once ``running`` is released.  The verdict is
    surfaced as a status line either way.  Compression failures are never judge
    input: the error text is not work toward the goal, and judging it spends a turn.
    """
    goal_followup = None
    compression_exhausted = bool(isinstance(result, dict) and result.get("compression_exhausted"))
    try:
        recovery_prompt, recovery_notice = _plan_goal_compression_recovery(
            session, result, status=status, raw=raw)
        if recovery_notice:
            _emit("status.update", sid, {"kind": "goal", "text": recovery_notice})
        if recovery_prompt:
            goal_followup = recovery_prompt
    except Exception as _goal_recovery_exc:
        _hook_failure("goal compression recovery", _goal_recovery_exc)
    if compression_exhausted or not _is_successful_goal_turn(result, status, raw):
        return goal_followup
    try:
        from hermes_cli.goals import GoalManager
        sid_key = session.get("session_key") or ""
        if sid_key and (
            goal_mgr := GoalManager(session_id=sid_key, default_max_turns=_goal_max_turns())
        ).is_active():
            try:
                from hermes_cli.goals import gather_background_processes as _gather_bg
                _bg_procs = _gather_bg()
            except Exception:
                _bg_procs = None
            decision = goal_mgr.evaluate_after_turn(
                raw, user_initiated=True, background_processes=_bg_procs)
            if verdict_msg := decision.get("message") or "":
                _emit("status.update", sid, {"kind": "goal", "text": verdict_msg})
            if decision.get("should_continue") and (
                cont_prompt := decision.get("continuation_prompt") or ""):
                goal_followup = cont_prompt
    except Exception as _goal_exc:
        _hook_failure("goal continuation hook", _goal_exc)
    return goal_followup


def _complete_loop_tick(sid: str, session: dict, raw: Any) -> None:
    """If this turn was a /loop wakeup, evaluate it (LOOP_COMPLETE, --until judge, caps, next)."""
    try:
        from hermes_cli.loops import LoopManager
        loop_sid_key = session.get("session_key") or ""
        if loop_sid_key:
            loop_mgr = LoopManager(session_id=loop_sid_key)
            loop_state = loop_mgr.state
            if loop_state is not None and loop_state.awaiting_response:
                loop_decision = loop_mgr.complete_tick(raw if isinstance(raw, str) else "")
                if loop_msg := loop_decision.get("message") or "":
                    _emit("status.update", sid, {"kind": "loop", "text": loop_msg})
    except Exception as _loop_exc:
        _hook_failure("loop completion hook", _loop_exc)


def _apply_pending_title(sid: str, session: dict) -> None:
    """Apply pending_title now that the DB row exists — in the session-owned profile store."""
    _pending = session.get("pending_title")
    if not _pending:
        return
    _session_key = session.get("session_key") or sid
    try:
        with _session_db(session) as _pdb:
            if _pdb and _pdb.set_session_title(_session_key, _pending):
                session["pending_title"] = None
    except ValueError as exc:
        # Invalid/duplicate title — non-retryable, drop it; auto-title takes over.
        session["pending_title"] = None
        logger.info("Dropping pending title for session %s: %s", _session_key, exc)
    except Exception:
        pass  # transient DB failure — keep pending_title for retry


def _dispatch_followup_turn(rid, sid: str, session: dict, prompt: Any, what: str, *,
                            on_done=None, on_error=None) -> None:
    """Chain one follow-up turn (caller already set ``running``); a dispatch failure
    runs ``on_error``, logs, and releases ``running``."""
    try:
        _emit("message.start", sid)
        _run_prompt_submit(rid, sid, session, prompt)
        if on_done is not None:
            on_done()
    except Exception as exc:
        if on_error is not None:
            on_error()
        _hook_failure(what, exc)
        with session["history_lock"]:
            session["running"] = False


def _run_post_turn_followups(
    rid, sid: str, session: dict, result: Any, goal_followup: str | None) -> None:
    """Chain whatever should run after ``running`` was released.

    Order: a user prompt that arrived mid-turn wins over every auto follow-up —
    drain it and skip the rest this cycle.  A leftover /steer the agent couldn't
    inject is requeued first so it isn't dropped (a real queued prompt still wins:
    ``_enqueue_prompt`` merges both).  Then the goal continuation, then completion
    notifications that arrived mid-turn.  Each nested ``_run_prompt_submit`` checks
    ``running`` under the lock first, so a racing user prompt wins.
    """
    _leftover_steer = result.get("pending_steer") if isinstance(result, dict) else None
    if isinstance(_leftover_steer, str) and _leftover_steer.strip():
        with session["history_lock"]:
            _enqueue_prompt(session, _leftover_steer, session.get("transport"))
    if _drain_queued_prompt(rid, sid, session):
        return
    if goal_followup:
        with session["history_lock"]:
            if session.get("running"):
                return  # user already sent something — their turn wins
            session["running"] = True
        _dispatch_followup_turn(rid, sid, session, goal_followup, "goal continuation dispatch")
    # Safety net for completion events that arrived mid-turn (the poller handles
    # between-turn delivery).  Ownership is positive-proof and compression-chain
    # aware (same fail-closed gate as the poller): session B must not consume
    # session A's event; a post-compression session still claims its
    # pre-compression dispatches.  Unclaimable events are requeued for the poller.
    try:
        from tools.process_registry import process_registry
        drained = process_registry.drain_notifications(
            session_key=session.get("session_key", ""),
            owns_event=lambda e: _session_owns_notification_event(sid, session, e),
            skip_poll_observed=False)
        for index, (_evt, synth) in enumerate(drained):
            with session["history_lock"]:
                if session.get("running"):
                    for pending_evt, _pending_synth in drained[index:]:
                        process_registry.completion_queue.put(pending_evt)
                    break
                session["running"] = True
            from tools.async_delegation import (
                claim_event_delivery, complete_event_delivery, release_event_delivery)
            _claim = claim_event_delivery(_evt, "tui-post-turn")
            if _claim is None:
                continue
            _dispatch_followup_turn(
                rid, sid, session, synth, "completion notification dispatch",
                on_done=lambda: complete_event_delivery(_evt, _claim),
                on_error=lambda: release_event_delivery(_evt, _claim))
    except Exception as _drain_exc:
        _hook_failure("completion queue drain", _drain_exc)


@dataclasses.dataclass(slots=True)
class _TurnRun:
    """Mutable state the phase helpers of one turn thread share.

    ``agent`` is bound eagerly so except/finally always have one even if setup
    throws (re-read after ``_sync_bot_capabilities`` may swap in a rebuilt agent).
    ``error_retained``: the finally skips the inflight clear (failed snapshot stays
    for resume replay).  ``error_detail``: cause for the "tui turn finished" bookend,
    stashed by both failure paths (the finally sees neither ``result`` nor the
    exception reliably); ``prompt_text`` is the post-@-expansion prompt the cause
    is checked against for quoting it back.
    """

    agent: Any
    one_turn_restore: Any
    terminal_callback: Any
    receipt_committed: bool
    scopes: _TurnScopes = dataclasses.field(default_factory=_TurnScopes)
    goal_followup: Any = None
    result: Any = None  # read after the finally for leftover /steer
    tts_queue: Any = None
    thinking_started: bool = False
    history: list = dataclasses.field(default_factory=list)
    history_version: int = 0
    run_kwargs: Any = None
    error_retained: bool = False
    error_detail: str = ""
    prompt_text: str = ""
    marker_key: str = ""
    receipt_attempted: bool = False


def _prepare_turn_input(sid: str, session: dict, st: _TurnRun, text: Any, images: list[str]):
    """Bind scopes, sync the agent, snapshot history and build the run message.

    Returns ``(prompt, run_message, cols, streamer)``, or None when @-expansion
    was refused (error already emitted).  The config-model sync is skipped while
    a /model --once override is active: the once-model is deliberately not pinned
    as model_override, so the sync would clobber it (a config.yaml change is
    adopted NEXT turn).  A model picked mid-turn was queued, not applied — apply
    it before the config sync so the explicit pick wins over a config change.
    """
    _bind_turn_scopes(sid, session, st.scopes)
    if not st.one_turn_restore:
        _apply_pending_model_switch(sid, session)
        _sync_agent_model_with_config(sid, session)
        _sync_agent_compression_with_config(sid, session)
    # Bot Chat: adopt Settings->Capabilities edits into the eternal bot session first.
    _sync_bot_capabilities(sid, session)
    st.agent = agent = session["agent"]
    # Snapshot after turn-start model sync: a deferred switch mutates history
    # and its version, and that mutation belongs to this turn.
    with session["history_lock"]:
        st.history = list(session["history"])
        st.history_version = int(session.get("history_version", 0))
    cwd = _session_cwd(session)
    _register_session_cwd(session)
    cols = session.get("cols", 80)
    streamer = make_stream_renderer(cols)
    prompt = text
    if isinstance(prompt, str) and "@" in prompt:
        ctx = _expand_context_references(agent, prompt, cwd)
        if ctx.blocked:
            _emit(
                "error", sid, {"message": "\n".join(ctx.warnings) or "Context injection refused."})
            return None
        prompt = ctx.message
    st.prompt_text = prompt if isinstance(prompt, str) else ""
    run_message: Any = _route_turn_images(agent, prompt, images) if images else prompt
    st.tts_queue, st.thinking_started = _start_turn_voice()
    return prompt, _apply_turn_notes(run_message, session), cols, streamer


def _invoke_agent(
    sid: str, session: dict, st: _TurnRun, prompt: Any, run_message: Any, streamer,
    images: list[str], display_kind: str | None, display_metadata: dict | None) -> None:
    """Wire the streaming callbacks and run the conversation into ``st.result``."""
    agent = st.agent

    def _stream(delta):
        with session["history_lock"]:
            _append_inflight_delta(session, delta)
        payload = {"text": delta}
        if streamer and (r := streamer.feed(delta)) is not None:
            payload["rendered"] = r
        if st.tts_queue is not None and isinstance(delta, str):
            st.tts_queue.put(delta)
        _emit("message.delta", sid, payload)

    # Interim assistant text (commentary beside tool calls, or a pre-nudge final
    # answer) is sealed by the desktop as its own segment instead of being lost
    # when message.complete replaces the streaming buffer.  Gated on
    # display.interim_assistant_messages (default true).
    if _load_interim_assistant_messages():
        def _interim_assistant_cb(text: str, *, already_streamed: bool = False) -> None:
            _emit("message.interim", sid, {"text": text, "already_streamed": already_streamed})
        agent.interim_assistant_callback = _interim_assistant_cb
    else:
        agent.interim_assistant_callback = None
    st.run_kwargs = _build_run_kwargs(
        agent, session, st.history, prompt, images, run_message, _stream, display_kind,
        display_metadata)
    # Auto-titling fires inside the turn prologue; this live-rename hook
    # repaints the sidebar the moment a title lands.
    _title_key = session.get("session_key") or sid
    agent._on_session_title = lambda t, _src, _k=_title_key: _emit(
        "session.title", sid, {"session_id": _k, "title": t})
    _usage_stop, _usage_thread = _start_usage_ticker(sid, agent)
    try:
        st.result = agent.run_conversation(run_message, **st.run_kwargs)
    finally:
        # Stop AND join before anything below emits: a tick surviving past
        # message.complete would roll the client's final usage back to a stale
        # snapshot.  The join is deliberately unbounded — once stop is set it only
        # waits out one in-flight _get_usage/_emit, whose worst case (a stalled
        # transport write) would stall the message.complete emit just the same.
        _usage_stop.set()
        _usage_thread.join()


def _absorb_turn_result(
    sid: str, session: dict, st: _TurnRun, text: Any, display_kind: str | None, display_metadata
) -> str | None:
    """Stamp, restore /moa, commit history, re-sync the session key; returns the history warning."""
    result = st.result
    if display_kind and isinstance(text, str):
        _stamp_synthetic_display_kind(
            st.agent, session, result, text, display_kind, display_metadata)
    if "moa_one_shot_restore" in session:
        _restore_moa_one_shot(sid, session)
    status_note = None
    if isinstance(result, dict):
        if isinstance(result.get("messages"), list):
            status_note = _commit_turn_history(session, result, st.history, st.history_version)
        # Auto-compression inside run_conversation() may have rotated
        # agent.session_id: sync session_key before title/goal/finalize use it,
        # keep pending_title (user intent), and restart the slash worker so
        # worker-backed commands (/title etc.) target the live session.
        _sync_session_key_after_compress(
            sid, session, clear_pending_title=False, restart_slash_worker=True)
    return status_note


def _complete_turn_payload(session: dict, st: _TurnRun, status_note: str | None, cols: int):
    """Build the ``message.complete`` payload, retain/clear the inflight turn and
    settle the hosted-room terminal receipt.  Returns ``(payload, raw, status)``."""
    result, agent = st.result, st.agent
    raw, status, last_reasoning = _turn_outcome(result)
    payload = {"text": raw, "usage": _get_usage(agent), "status": status}
    if last_reasoning:
        payload["reasoning"] = last_reasoning
    if status_note:
        payload["warning"] = status_note
    if result.get("response_previewed"):
        payload["response_previewed"] = True
    # Structured billing-wall descriptor so the client renders a
    # billing-specific recovery surface instead of re-parsing text.
    _billing_block = result.get("billing_block") if isinstance(result, dict) else None
    if _billing_block:
        payload["billing"] = _billing_block
        payload["failure_reason"] = result.get("failure_reason")
    if rendered := render_message(raw, cols):
        payload["rendered"] = rendered
    # Layer descriptor computed before the retain below so resume replay
    # carries the same one (advisory; older clients ignore it).
    _error_surface = _turn_error_surface(agent, result) if status == "error" else None
    _result_error = result.get("error") if isinstance(result, dict) else None
    error_value = _result_error if isinstance(result, dict) else raw
    with session["history_lock"]:
        if status == "error":
            # Retain the failed turn for resume replay: if this terminal frame
            # is lost to a disconnect, resume's inflight payload is the only
            # carrier of the failure.
            _fail_inflight_turn(session, error_value, error_surface=_error_surface)
            st.error_retained = True
            st.error_detail = _turn_failure_detail(
                error_value, result.get("failure_reason") if isinstance(result, dict) else None,
                st.prompt_text)
        else:
            _clear_inflight_turn(session)
    if status == "error":
        payload["error"] = str(error_value or raw)
        payload["recoverable"] = True
        if _error_surface:
            payload["error_surface"] = _error_surface
    if st.terminal_callback is not None:
        st.receipt_attempted = True
        st.terminal_callback({
            "status": (
                "cancelled" if status == "interrupted"
                else "failed" if status == "error" else "settled"),
            "text": raw if isinstance(raw, str) else str(raw),
            **(
                {"error": str(_result_error or raw)}
                if status == "error" and isinstance(result, dict) else {})})
        st.receipt_committed = True
    if st.receipt_committed:
        _retire_turn_marker(session, st.marker_key)
    return payload, raw, status


def _recover_turn_exception(sid: str, session: dict, st: _TurnRun, e: BaseException) -> None:
    """Except-path of the turn: crash log, history restore, terminal error frame."""
    import traceback
    with contextlib.suppress(Exception):
        os.makedirs(os.path.dirname(_CRASH_LOG), exist_ok=True)
        with open(_CRASH_LOG, "a", encoding="utf-8") as f:
            f.write(
                f"\n=== turn-dispatcher exception · "
                f"{time.strftime('%Y-%m-%d %H:%M:%S')} · sid={sid} ===\n")
            f.write(traceback.format_exc())
    print(f"[gateway-turn] {type(e).__name__}: {e}", file=sys.stderr, flush=True)
    # An exception in the agent's finalizer can leave the gateway's in-memory
    # history at the turn-start snapshot; keep the partial turn available to
    # the next prompt (the durable inflight record still carries the
    # recoverable error state).
    _restore_agent_history_after_turn_error(session, st.agent)
    if st.terminal_callback is not None and not st.receipt_attempted:
        st.receipt_attempted = True
        try:
            st.terminal_callback({"status": "failed", "text": "", "error": str(e)})
            st.receipt_committed = True
        except Exception:
            logger.exception("hosted room terminal receipt commit failed")
    try:
        # Same terminal error frame shape as the returned-error path (uniform
        # client handling), retaining the turn for replay.
        _emit_terminal_turn_error(sid, session, e, retire_marker=st.receipt_committed)
        st.error_retained = True
        st.error_detail = _turn_failure_detail(e, type(e).__name__, st.prompt_text)
    except Exception as emit_exc:
        print(
            f"[gateway-turn] terminal error emit failed: {type(emit_exc).__name__}: {emit_exc}",
            file=sys.stderr, flush=True)
        _emit("error", sid, {"message": str(e)})


def _finish_turn(sid: str, session: dict, st: _TurnRun) -> None:
    """Finally-path of the turn: release everything, then the "tui turn finished" bookend."""
    agent, history, run_kwargs = st.agent, st.history, st.run_kwargs
    # Drop both snapshots of the pre-turn history before asking glibc to return
    # pages; session["history"] already points at the new/pruned result.
    history.clear()
    if isinstance(run_kwargs, dict):
        run_kwargs.clear()
    # While any profile-specific HERMES_HOME override is still active, so
    # context.memory_trim resolves from the session's own config.
    try:
        from hermes_cli.mem_trim import trim_memory
        trim_memory(reason="tui turn completion")
    except Exception:
        logger.debug("post-turn memory trim failed", exc_info=True)
    if st.thinking_started:
        with contextlib.suppress(Exception):
            from tools.voice_mode import stop_thinking_sound
            stop_thinking_sound()
    if st.tts_queue is not None:
        st.tts_queue.put(None)  # end-of-text sentinel — flush + finish speaking
    if st.one_turn_restore:
        try:
            _restore_agent_model_runtime(agent, st.one_turn_restore)
            _restart_slash_worker(sid, session)
            _persist_live_session_runtime(session)
            _persist_live_session_system_prompt(session)
        except Exception:
            logger.debug("TUI one-turn model restore failed", exc_info=True)
    _release_turn_scopes(st.scopes)


def _log_turn_finished(sid: str, session: dict, st: _TurnRun, started_monotonic: float) -> None:
    """Closing bookend of "tui prompt accepted" — fires on every path, so one
    accepted prompt produces exactly one finished record.  agent.session_id is
    re-read because compression may have rotated it mid-turn (an accepted/finished
    pair whose id changed IS a rotation trace).  A missing finished record means
    the thread died before the finally."""
    result = st.result
    if isinstance(result, dict):
        status = _result_status(result)
    else:
        status = "error" if st.error_retained else "complete"
    logger.info(
        "tui turn finished: ui_session=%s session_key=%s "
        "agent_session_id=%s status=%s error_retained=%s duration=%.1fs"
        "%s",
        sid, session.get("session_key") or "", getattr(st.agent, "session_id", "") or "", status,
        st.error_retained, time.monotonic() - started_monotonic, st.error_detail)


def _run_prompt_submit(
    rid, sid: str, session: dict, text: Any, *, display_kind: str | None = None,
    display_metadata: dict | None = None, image_paths: list[str] | None = None,
    queued_prompt_generation: int | None = None,
    terminal_callback: Callable[[dict[str, Any]], None] | None = None) -> bool:
    admitted = _admit_prompt_turn(sid, session, text, image_paths, queued_prompt_generation)
    if admitted is None:
        return False
    images, agent = admitted
    # The ONE INFO record proving a Desktop/TUI prompt was accepted by THIS
    # process; ties the UI session id, gateway session_key and the agent's live
    # session_id (compression rotates the last independently) together for a
    # rotation-mute trace.  No prompt content is logged.
    _turn_started_monotonic = time.monotonic()
    logger.info(
        "tui prompt accepted: ui_session=%s session_key=%s agent_session_id=%s "
        "kind=%s chars=%s images=%d",
        sid, session.get("session_key") or "", getattr(agent, "session_id", "") or "",
        display_kind or "user", len(text) if isinstance(text, str) else "-", len(images))
    _emit("message.start", sid)

    def run():
        # ContextVars from the RPC dispatcher do not follow onto this thread:
        # rebind the exact transport stored on this session generation before any
        # tool can commission a child (delegate_task captures it as authority).
        transport_token = bind_transport(session.get("transport"))
        runtime_session_token = _current_runtime_session_record.set(session)
        st = _TurnRun(
            session["agent"], session.pop("one_turn_model_restore", None), terminal_callback,
            receipt_committed=terminal_callback is None)
        st.marker_key = _record_turn_marker(session, text)
        try:
            prepared = _prepare_turn_input(sid, session, st, text, images)
            if prepared is None:
                return
            prompt, run_message, cols, streamer = prepared
            _invoke_agent(
                sid, session, st, prompt, run_message, streamer, images, display_kind,
                display_metadata)
            status_note = _absorb_turn_result(
                sid, session, st, text, display_kind, display_metadata)
            payload, raw, status = _complete_turn_payload(session, st, status_note, cols)
            _emit("message.complete", sid, payload)
            st.goal_followup = _goal_followup_after_turn(sid, session, st.result, status, raw)
            if status == "complete":
                _complete_loop_tick(sid, session, raw)
                _apply_pending_title(sid, session)
                # Voice fallback when the streaming pipeline couldn't start (the
                # streaming path already spoke everything via tts_queue); barge-aware
                # so spoken interruptions cut this playback too.
                if (
                    st.tts_queue is None and isinstance(raw, str) and raw.strip()
                    and _voice_tts_enabled()):
                    try:
                        threading.Thread(
                            target=_speak_text_with_barge, args=(raw,), daemon=True).start()
                    except ImportError:
                        logger.warning("voice TTS skipped: hermes_cli.voice unavailable")
                    except Exception as e:
                        logger.warning("voice TTS dispatch failed: %s", e)
        except Exception as e:
            _recover_turn_exception(sid, session, st, e)
        finally:
            _finish_turn(sid, session, st)
            _current_runtime_session_record.reset(runtime_session_token)
            reset_transport(transport_token)
            # A stale interim closure must not fire during a later turn.
            st.agent.interim_assistant_callback = None
            with session["history_lock"]:
                session["running"] = False
                session["last_active"] = time.time()
                if not st.error_retained:
                    _clear_inflight_turn(session)
            _log_turn_finished(sid, session, st, _turn_started_monotonic)
            # Backstop for turns that never reached a terminal frame.
            if st.receipt_committed:
                _retire_turn_marker(session, st.marker_key)
                with session["history_lock"]:
                    if session.get("_active_turn_marker_key") == st.marker_key:
                        session.pop("_active_turn_marker_key", None)
                    session.pop("_hosted_room_task", None)
            session.pop("_auto_continue_scheduled", None)
            _emit_settled_session_info(sid, session, st.agent)
        _run_post_turn_followups(rid, sid, session, st.result, st.goal_followup)
    run_thread = threading.Thread(target=run, daemon=True)
    with _sessions_lock:
        registered = _sessions.get(sid)
        can_start = not session.get("_closing") and (registered is None or registered is session)
        if can_start:
            session["_run_thread"] = run_thread
            run_thread.start()
    if not can_start:
        with session["history_lock"]:
            session["running"] = False
    return can_start


def register(server) -> None:
    """Publish this module's helpers onto ``server``, rebound to its globals."""
    bind_module(globals(), server, skip=("_",))
