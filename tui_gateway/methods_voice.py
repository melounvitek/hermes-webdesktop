"""Voice / TTS / wake-word JSON-RPC handlers and their process-global state (one microphone, one speaker per process).

Bodies are rebound onto server.py's globals at install time (see
method_ctx.bind_module), so they reference server.py globals bare.
"""

from __future__ import annotations

import threading

from .method_ctx import HandlerRegistry, bind_module

_registry = HandlerRegistry()
method = _registry.method


# ── Methods: voice ───────────────────────────────────────────────────


_voice_sid_lock = threading.Lock()
_voice_event_sid: str = ""
_voice_wake_owner: "Optional[Transport]" = None


def _voice_emit(event: str, payload: dict | None = None) -> None:
    """Emit a voice event toward the session that most recently turned the
    mode on. Voice is process-global (one microphone), so there's only ever
    one sid to target; the TUI handler treats an empty sid as "active
    session". Kept separate from _emit to make the lack of per-call sid
    argument explicit."""
    with _voice_sid_lock:
        sid = _voice_event_sid
    _emit(event, sid, payload)


def _resume_voice_wake() -> None:
    global _voice_wake_owner
    with _voice_sid_lock:
        owner, _voice_wake_owner = _voice_wake_owner, None
    if owner is not None:
        _wake_resume_if_owner(owner)


def _voice_mode_enabled() -> bool:
    """Current voice-mode flag (runtime-only, CLI parity).

    cli.py initialises ``_voice_mode = False`` at startup and only flips
    it via ``/voice on``; it never reads a persisted enable bit from
    config.yaml.  We match that: no config lookup, env var only.  This
    avoids the TUI auto-starting in REC the next time the user opens it
    just because they happened to enable voice in a prior session.
    """
    return os.environ.get("HERMES_VOICE", "").strip() == "1"


def _voice_tts_enabled() -> bool:
    """Whether agent replies should be spoken back via TTS (runtime only)."""
    return os.environ.get("HERMES_VOICE_TTS", "").strip() == "1"


def _tts_lease_async(lease: str, active: bool) -> None:
    """Acquire/release a TTS engine lease off the RPC thread.

    Speech-output toggles are the signal that TTS is about to be needed (or
    no longer is). Acquiring warms the configured provider — for local
    engines that is a model load, possibly a voice download — so it must not
    block the toggle's RPC reply. Release is cheap but rides the same thread
    for symmetry. Best-effort: a failure here never affects the toggle.
    """

    def _run():
        try:
            from tools.tts_tool import acquire_tts_lease, release_tts_lease

            if active:
                acquire_tts_lease(lease)
            else:
                release_tts_lease(lease)
        except Exception as e:
            logger.debug("voice: tts lease %s active=%s failed: %s", lease, active, e)

    threading.Thread(target=_run, name=f"tts-lease-{lease}", daemon=True).start()


def _any_session_running() -> bool:
    """True while any session's agent turn is in flight.

    Registered as the voice busy-probe (``hermes_cli.voice.set_voice_busy_probe``)
    so silent capture cycles during a long agent turn don't count toward the
    no-speech limit — the user is correctly quiet while the agent works.
    Voice is process-global (one microphone), so any running session holds.
    """
    try:
        with _sessions_lock:
            return any(s.get("running") for s in _sessions.values())
    except Exception:
        return False


# ── Streaming TTS (one active pipeline per process — one speaker) ──────────
# Token deltas from the running turn feed a sentence-buffering consumer
# (tools.tts_tool.stream_tts_to_speaker) so speech starts on the first
# sentence instead of after the full reply. Voice is process-global, so a
# single slot suffices; starting a new turn's pipeline barges in on the
# previous one.

_tts_stream_lock = threading.Lock()
_tts_stream_state: Optional[dict] = None


def _tts_stream_begin() -> Optional[queue.Queue]:
    """Start a per-turn streaming TTS consumer; None when TTS can't stream."""
    if not _voice_tts_enabled():
        return None
    try:
        from tools.tts_tool import check_tts_requirements, stream_tts_to_speaker

        if not check_tts_requirements():
            return None
    except Exception:
        return None

    _tts_stream_stop()
    text_queue: queue.Queue = queue.Queue()
    stop = threading.Event()
    done = threading.Event()
    threading.Thread(
        target=stream_tts_to_speaker, args=(text_queue, stop, done), daemon=True
    ).start()

    global _tts_stream_state
    with _tts_stream_lock:
        _tts_stream_state = {"stop": stop, "done": done}

    if _voice_mode_enabled() and _voice_cfg_dict().get("barge_in", True):
        _arm_full_duplex_listener()

    return text_queue


def _tts_stream_stop(user_barge: bool = True) -> None:
    """Cut any in-flight streaming TTS (new turn, interrupt, /voice off).

    *user_barge* latches the interruption for the next turn's model note
    (``mark_speech_interrupted``) — pass ``False`` for mode changes like
    ``/voice off`` where the user isn't talking over the reply.
    """
    global _tts_stream_state
    with _tts_stream_lock:
        state, _tts_stream_state = _tts_stream_state, None
    if state is None:
        return
    if user_barge and not state["done"].is_set():
        import traceback as _tb
        logger.debug(
            "TTS CUT: _tts_stream_stop(user_barge=True) — new turn or "
            "interrupt cutting in-flight TTS\n%s",
            "".join(_tb.format_stack()),
        )
        from tools.tts_streaming import mark_speech_interrupted

        mark_speech_interrupted()
    state["stop"].set()
    try:
        from tools.voice_mode import stop_playback

        stop_playback()
    except Exception:
        pass


# ── Full-duplex agent-turn listener (one mic, whole turn) ──────────────────
# Replaces the per-playback barge monitors: those only opened the mic once
# TTS playback started (deaf during LLM generation) and calibrated the VAD
# floor against active speaker bleed (deaf during playback too, in practice).
# This listener arms at utterance-submit, spans generation AND playback, and
# disarms when no session is running, no TTS is pending, and no audio flows.

_fd_listener_lock = threading.Lock()
_fd_listener_active = False
# (stop, done) pairs for fallback whole-reply speak paths currently active —
# the listener must cut THEIR private stop events too, and must keep
# listening while any of them is still speaking.
_fd_speak_pipelines: "set[tuple[threading.Event, threading.Event]]" = set()


def _arm_full_duplex_listener() -> None:
    """Arm the process-global full-duplex listener (idempotent — one mic)."""
    global _fd_listener_active
    with _fd_listener_lock:
        if _fd_listener_active:
            return
        _fd_listener_active = True
    threading.Thread(
        target=_full_duplex_listener, daemon=True, name="voice-full-duplex"
    ).start()


def _fd_tts_pending() -> bool:
    """True while any TTS (streaming pipeline or fallback speak) is unfinished."""
    with _tts_stream_lock:
        state = _tts_stream_state
    if state is not None and not state["done"].is_set():
        return True
    with _fd_listener_lock:
        pipelines = list(_fd_speak_pipelines)
    return any(not done.is_set() for _stop, done in pipelines)


def _full_duplex_listener() -> None:
    """Mic live from utterance-submit to turn-complete; phase-aware trip.

    * generation phase (no TTS audio flowing): user speech interrupts every
      running session's agent turn — the same ``agent.interrupt()`` seam
      ``session.interrupt`` uses — and cuts any pending TTS pipeline so the
      stale reply never plays. The captured utterance is transcribed and
      emitted as ``voice.transcript`` (the TUI submits it as the next turn).
    * playback phase: cuts TTS (streaming pipeline + fallback speak paths +
      file player) and submits the captured interruption.

    Stop phrase is honored in both phases: mid-generation it interrupts the
    turn AND ends the voice chat ("stop everything").
    """
    global _fd_listener_active
    try:
        from tools.tts_streaming import mark_speech_interrupted
        from tools.voice_mode import (
            full_duplex_listen,
            is_audio_output_active,
            stop_playback,
            transcribe_recording,
        )

        cfg = _voice_cfg_dict()
        try:
            _mult = float(cfg.get("barge_in_threshold_multiplier", 0) or 0)
        except (TypeError, ValueError):
            _mult = 0.0
        try:
            _grace_ms = int(float(cfg.get("barge_in_grace_seconds", 0.5)) * 1000)
        except (TypeError, ValueError):
            _grace_ms = 500

        def _should_stop() -> bool:
            if not _voice_mode_enabled():
                return True
            if _any_session_running():
                return False
            if _fd_tts_pending():
                return False
            return not is_audio_output_active()

        tripped = threading.Event()

        def _cut_all_tts() -> None:
            # Streaming pipeline (private stop event + player).
            _tts_stream_stop(user_barge=True)
            # Fallback whole-reply speak paths (their own stop events).
            with _fd_listener_lock:
                pipelines = list(_fd_speak_pipelines)
            for _stop, _done in pipelines:
                _stop.set()
            stop_playback()

        def _on_trigger(phase: str) -> None:
            tripped.set()
            mark_speech_interrupted()
            if phase == "playback":
                logger.debug(
                    "TTS CUT: full-duplex listener tripped during playback"
                )
                _cut_all_tts()
            else:
                logger.debug(
                    "full-duplex listener tripped during generation — "
                    "interrupting running turn(s)"
                )
                # Cut pending TTS FIRST so the stale reply can never speak.
                _cut_all_tts()
                # Interrupt every running session's turn — voice is
                # process-global, and the same seam session.interrupt uses.
                try:
                    with _sessions_lock:
                        running = [
                            s for s in _sessions.values() if s.get("running")
                        ]
                    for s in running:
                        agent = s.get("agent")
                        if agent is not None and hasattr(agent, "interrupt"):
                            try:
                                agent.interrupt()
                            except Exception:
                                pass
                except Exception as e:
                    logger.debug("voice interjection interrupt failed: %s", e)
            _voice_emit("voice.interrupted")

        wav_path = full_duplex_listen(
            _should_stop,
            is_playing=is_audio_output_active,
            on_trigger=_on_trigger,
            multiplier=_mult or None,
            grace_ms=max(0, _grace_ms),
        )
        if not (wav_path and tripped.is_set()):
            return
        try:
            result = transcribe_recording(wav_path)
            text = (result.get("transcript") or "").strip() if result.get("success") else ""
            if text:
                # Stop-check must never break transcript delivery — if the
                # helper is unavailable (stubbed voice_mode in tests, partial
                # installs), treat as not-a-stop-phrase.
                try:
                    from tools.voice_mode import is_voice_stop_phrase
                    _is_stop = is_voice_stop_phrase(text)
                except Exception:
                    _is_stop = False

                if _is_stop:
                    # Bare stop phrase — in EITHER phase the user means
                    # "stop everything": the turn was already interrupted /
                    # TTS cut at trip time; now end the voice chat.
                    os.environ["HERMES_VOICE"] = "0"
                    os.environ["HERMES_VOICE_TTS"] = "0"
                    try:
                        from hermes_cli.voice import stop_continuous

                        stop_continuous()
                    except Exception:
                        pass
                    _voice_emit("voice.transcript", {"stop_phrase": True, "text": text})
                else:
                    _voice_emit("voice.transcript", {"text": text})
        finally:
            try:
                os.unlink(wav_path)
            except OSError:
                pass
    except Exception as e:
        logger.debug("full-duplex listener failed: %s", e)
    finally:
        with _fd_listener_lock:
            _fd_listener_active = False


def _speak_text_with_barge(text: str) -> None:
    """Speak *text* via hermes_cli.voice.speak_text with spoken barge-in.

    The fallback whole-reply path (streaming couldn't start) and the
    ``voice.tts`` RPC previously called ``speak_text`` bare — speech over
    those paths was UNINTERRUPTIBLE by voice. The full-duplex agent-turn
    listener covers this path too: the (stop, done) pair is registered in
    ``_fd_speak_pipelines`` so the listener can cut the private stop event
    on a playback trip and keeps listening while this speak is pending.
    """
    from hermes_cli.voice import speak_text

    stop = threading.Event()
    done = threading.Event()
    with _fd_listener_lock:
        _fd_speak_pipelines.add((stop, done))

    def _speak():
        try:
            speak_text(text, stop)
        except TypeError:
            # Older wrapper without the stop_event parameter.
            speak_text(text)
        finally:
            done.set()
            with _fd_listener_lock:
                _fd_speak_pipelines.discard((stop, done))

    threading.Thread(target=_speak, daemon=True).start()
    if _voice_mode_enabled() and _voice_cfg_dict().get("barge_in", True):
        _arm_full_duplex_listener()


def _voice_cfg_dict() -> dict:
    """Shape-safe accessor for the ``voice:`` block in config.yaml.

    ``_load_cfg()`` does not deep-merge DEFAULT_CONFIG, so both the
    root AND ``voice`` may be any YAML scalar / list / None. A hand-edit
    like ``voice: true`` or a malformed top-level config that parses to
    a scalar would otherwise break ``.get("…")`` and take every
    ``voice.*`` branch down with it (Copilot round-3..7 review on
    #19835). Coerce through ``isinstance`` at every level so malformed
    config falls back to an empty dict instead of crashing /voice.
    """
    cfg = _load_cfg()
    voice_cfg = cfg.get("voice") if isinstance(cfg, dict) else None

    return voice_cfg if isinstance(voice_cfg, dict) else {}


def _voice_record_key() -> str:
    """Current ``voice.record_key`` value, documented default on error."""
    record_key = _voice_cfg_dict().get("record_key")

    return str(record_key) if isinstance(record_key, str) and record_key else "ctrl+b"


# ── Wake word ("Hey Hermes") ──────────────────────────────────────────────
# The detector is process-global (one mic), like voice. The first eligible
# transport to call wake.start owns it until stop, disconnect, or stream failure.
# On detection we emit wake.detected; the client opens a new session and starts
# its own voice capture. The detector yields the mic to gateway voice.record
# (pause/resume below) and to the desktop's browser mic (wake.pause/resume RPCs).
_wake_lock = threading.Lock()
_wake_owner_transport: "Optional[Transport]" = None
_wake_owner_surface = ""


def _wake_owner_snapshot():
    with _wake_lock:
        return _wake_owner_transport, _wake_owner_surface


def _release_wake_for_transport(transport: "Transport") -> bool:
    """Release the wake lease iff ``transport`` is the current gateway owner."""
    global _wake_owner_transport, _wake_owner_surface
    with _wake_lock:
        if _wake_owner_transport is not transport:
            return False
        _wake_owner_transport = None
        _wake_owner_surface = ""
    try:
        from tools.wake_word import stop_listening

        stop_listening(owner=transport)
    except Exception as e:
        logger.debug("wake stop failed: %s", e)
    return True


def _release_gateway_wake_owner() -> bool:
    owner, _surface = _wake_owner_snapshot()
    return owner is not None and _release_wake_for_transport(owner)


_wake_resume_retry_lock = threading.Lock()
_wake_resume_retry_active = False


def _wake_resume_if_owner(owner: "Transport", *, retry_seconds: float = 15.0,
                          retry_interval: float = 1.0) -> bool:
    """Resume the wake detector for ``owner``; self-heal a busy microphone.

    Reopening the mic right after a voice turn can fail while the capture
    device is still being released (browser WebRTC tracks release async).
    The CLI covers this with its idle watchdog; the gateway had nothing, so
    one failed resume left the listener silently dead until the user toggled
    it by hand — despite ``wake_word.enabled: true``. On an exception (mic
    open failure) we retry in a background thread until it sticks, the lease
    changes hands, or ``retry_seconds`` elapses. ``False`` from
    ``resume_listening`` (lease gone / different owner) is final — never
    retried, so this can't steal another surface's mic.
    """
    from tools.wake_word import resume_listening

    try:
        return resume_listening(owner=owner)
    except Exception as e:
        logger.debug("wake resume failed (will retry): %s", e)

    global _wake_resume_retry_active
    with _wake_resume_retry_lock:
        if _wake_resume_retry_active:
            return False
        _wake_resume_retry_active = True

    def _retry() -> None:
        global _wake_resume_retry_active
        deadline = time.monotonic() + retry_seconds
        try:
            while time.monotonic() < deadline:
                time.sleep(retry_interval)
                try:
                    if resume_listening(owner=owner):
                        logger.info("wake: detector resumed after retry")
                        return
                except Exception:
                    continue
                # False — detector gone or lease moved: stop, don't fight it.
                return
            logger.warning(
                "wake: could not resume detector after voice turn "
                "(microphone still busy?) — toggle the wake word to re-arm"
            )
        finally:
            with _wake_resume_retry_lock:
                _wake_resume_retry_active = False

    threading.Thread(target=_retry, daemon=True, name="wake-resume-retry").start()
    return False


def _persist_wake_enabled(enabled: bool) -> bool:
    """Write ``wake_word.enabled`` to config.yaml.

    Only called for explicit user gestures (the desktop ear toggle, ``/wake
    on|off``) — never from passive auto-arm paths, so a mic can't become
    persistently enabled without a deliberate click.
    """
    try:
        from cli import save_config_value

        return bool(save_config_value("wake_word.enabled", enabled))
    except Exception as e:
        logger.warning("wake: failed to persist wake_word.enabled=%s: %s", enabled, e)
        return False


@method("gateway.capabilities")
def _(rid, params: dict) -> dict:
    """What guarantees THIS BUILD enforces, for a client that must not assume.

    An automated client cannot tell a gateway that fences concurrent writers to
    one session from one that silently allows them -- both accept the same calls
    and both answer prompt.submit the same way. It only finds out by corrupting a
    conversation. So the guarantee is advertised, and a client that does not see
    it advertised is expected to withhold rather than hope.

    Sourced from the module that performs the enforcement, never from config: a
    capability an operator can switch on without also having the mechanism is
    worse than no capability at all, because it is believed.
    """
    from hermes_cli.active_sessions import PER_SESSION_EXCLUSIVE_SUBMIT

    return _ok(rid, {"per_session_exclusive_submit": bool(PER_SESSION_EXCLUSIVE_SUBMIT)})


@method("ping")
def _(rid, params: dict) -> dict:
    """Cheapest possible liveness probe for the desktop client.

    Answered synchronously on the WS reader thread, so it works even while
    every agent is mid-turn or the GIL is contended — the round-trip only
    measures socket health, not backend load. A desktop client uses it after
    sleep/wake to distinguish a half-open TCP connection (no close event, so
    ``connectionState`` still reads ``open`` while every RPC hangs until its
    per-call timeout) from a genuinely healthy socket, and forces a reconnect
    in the former case instead of letting the next ``prompt.submit`` hang.
    """
    return _ok(rid, {"pong": True})


@method("wake.start")
def _(rid, params: dict) -> dict:
    """Arm the wake-word listener for the calling surface ("tui" | "gui").

    Idempotent and gated: returns ``{started: False, reason}`` when the wake
    word is disabled, scoped to another surface, or its deps/mic aren't ready.

    ``persist: true`` marks an explicit user gesture (toggle click, /wake on):
    when the feature is disabled in config, it flips ``wake_word.enabled`` on
    and saves it before arming, so the choice sticks for future sessions.
    Passive auto-arm callers omit it and keep getting the config-gated refusal.
    """
    surface = str(params.get("surface") or "auto").strip().lower()
    persist = bool(params.get("persist"))
    transport = current_transport() or _stdio_transport
    try:
        from tools.wake_word import (
            WakeWordInUse,
            check_wake_word_requirements,
            detector_frame_info,
            load_wake_word_config,
            owns_listener,
            resolve_capture_mode,
            start_listening,
            wake_phrase,
            wake_surface_enabled,
        )
    except Exception as e:
        return _err(rid, 5026, f"wake module unavailable: {e}")

    cfg = load_wake_word_config()
    # Desktop remote (gui) prefers client capture: Mac mic → wake.feed PCM,
    # while the engine still runs on the backend. CLI/TUI stay local.
    prefer_client = surface in ("gui", "desktop") or bool(params.get("client_capture"))
    capture_mode = resolve_capture_mode(cfg, prefer_client=prefer_client)
    external_audio = capture_mode == "client"
    # Requirements first: a gesture on an unarmed-able setup (no STT/TTS, no
    # mic, missing key) must refuse WITHOUT flipping wake_word.enabled — else
    # config says on while nothing can ever arm, and auto-arm paths churn.
    # Temporarily stamp capture so the probe matches the arm mode.
    probe_cfg = dict(cfg)
    probe_cfg["capture"] = capture_mode
    reqs = check_wake_word_requirements(probe_cfg)
    if not reqs["available"]:
        logger.warning("wake.start(%s): not available — %s", surface, reqs.get("hint"))
        return _ok(rid, {
            "started": False,
            "reason": "unavailable",
            "hint": reqs.get("hint") or "",
            "capture": capture_mode,
        })
    enabled_persisted = False
    if persist and not cfg.get("enabled"):
        enabled_persisted = _persist_wake_enabled(True)
        if enabled_persisted:
            cfg = dict(cfg)
            cfg["enabled"] = True
    if not wake_surface_enabled(surface, cfg):
        # Distinguish "feature off in config" (reason: disabled — a persist:true
        # retry can turn it on) from "scoped to a different surface" (reason:
        # disabled_for_surface — respects an explicit wake_word.surface choice,
        # which persist does NOT override).
        reason = "disabled" if not cfg.get("enabled") else "disabled_for_surface"
        logger.info("wake.start(%s): %s (enabled=%s, surface=%s)",
                    surface, reason, cfg.get("enabled"), cfg.get("surface"))
        return _ok(rid, {"started": False, "reason": reason})

    existing_owner, existing_surface = _wake_owner_snapshot()
    if existing_owner is not None and (
        _transport_is_dead(existing_owner) or not owns_listener(existing_owner)
    ):
        _release_wake_for_transport(existing_owner)
        existing_owner = None
        existing_surface = ""
    if existing_owner is not None and existing_owner is not transport:
        return _ok(rid, {
            "started": False,
            "reason": "owned",
            "owner_surface": existing_surface,
        })

    sid = str(params.get("session_id") or "")
    phrase = wake_phrase(cfg)
    new_session = bool(cfg.get("start_new_session", True))

    def _on_detect() -> None:
        from tools.wake_word import get_last_match, owns_listener, pause_listening

        if not pause_listening(owner=transport):
            return
        if not owns_listener(transport):
            return
        if _transport_is_dead(transport):
            _release_wake_for_transport(transport)
            return
        # Multi-phrase engines report WHICH phrase fired and the profile it
        # belongs to, so one listener can wake any enrolled profile. Falls
        # back to the owner's configured phrase / no profile for
        # single-phrase engines.
        matched_phrase, matched_profile = get_last_match() or (phrase, "")
        logger.info("wake.detected: emitting to sid=%r (transport=%s, profile=%r)",
                    sid, type(transport).__name__, matched_profile)
        token = bind_transport(transport)
        try:
            _emit("wake.detected", sid, {
                "phrase": matched_phrase or phrase,
                "profile": matched_profile or None,
                "start_new_session": new_session,
            })
        finally:
            reset_transport(token)

    try:
        start_listening(
            _on_detect,
            owner=transport,
            config=cfg,
            external_audio=external_audio,
        )
    except WakeWordInUse:
        return _ok(rid, {
            "started": False,
            "reason": "owned",
            "owner_surface": existing_surface or None,
        })
    except Exception as e:
        logger.warning("wake.start(%s): failed to start listener: %s", surface, e)
        return _err(rid, 5026, str(e))
    global _wake_owner_transport, _wake_owner_surface
    with _wake_lock:
        _wake_owner_transport = transport
        _wake_owner_surface = surface
    frame = detector_frame_info()
    logger.info(
        "wake.start(%s): listening for %r (%s) capture=%s frame=%s",
        surface, reqs["phrase"], reqs["provider"], capture_mode, frame.get("frame_length"),
    )
    return _ok(rid, {
        "started": True,
        "phrase": reqs["phrase"],
        "provider": reqs["provider"],
        "owner_surface": surface,
        "enabled_persisted": enabled_persisted,
        "capture": capture_mode,
        "sample_rate": frame.get("sample_rate", 16000),
        "frame_length": frame.get("frame_length", 1280),
    })


@method("wake.stop")
def _(rid, params: dict) -> dict:
    """Stop this surface's listener.

    ``persist: true`` (explicit user gesture) also writes
    ``wake_word.enabled: false`` to config.yaml so auto-arm stays off in
    future sessions — the toggle is the config, not just the live listener.
    """
    transport = current_transport() or _stdio_transport
    stopped = _release_wake_for_transport(transport)
    disabled_persisted = False
    if bool(params.get("persist")):
        try:
            from tools.wake_word import load_wake_word_config

            currently_enabled = bool(load_wake_word_config().get("enabled"))
        except Exception:
            currently_enabled = True
        if currently_enabled:
            disabled_persisted = _persist_wake_enabled(False)
    return _ok(rid, {
        "stopped": stopped,
        "reason": None if stopped else "not_owner",
        "disabled_persisted": disabled_persisted,
    })


@method("wake.pause")
def _(rid, params: dict) -> dict:
    """Release the mic (e.g. while the desktop's browser captures audio)."""
    transport = current_transport() or _stdio_transport
    try:
        from tools.wake_word import pause_listening

        paused = pause_listening(owner=transport)
        logger.info("wake.pause: detector paused=%s", paused)
    except Exception as e:
        logger.debug("wake.pause failed: %s", e)
        paused = False
    return _ok(rid, {
        "paused": paused,
        "reason": None if paused else "not_owner",
    })


@method("wake.resume")
def _(rid, params: dict) -> dict:
    """Reclaim the mic after a pause; no-op if the listener isn't armed."""
    transport = current_transport() or _stdio_transport
    resumed = _wake_resume_if_owner(transport)
    logger.info("wake.resume: detector resumed=%s", resumed)
    return _ok(rid, {
        "resumed": resumed,
        "reason": None if resumed else "not_owner",
    })


@method("wake.status")
def _(rid, params: dict) -> dict:
    try:
        from tools.wake_word import (
            audio_is_silent,
            check_wake_word_requirements,
            detector_frame_info,
            get_input_device_status,
            is_listening,
            load_wake_word_config,
            owns_listener,
            resolve_capture_mode,
            silent_audio_hint,
        )
        cfg = load_wake_word_config()
        # Prefer client when the GUI asks (desktop remote re-arm / status).
        prefer_client = bool(params.get("client_capture")) or str(
            params.get("surface") or ""
        ).strip().lower() in ("gui", "desktop")
        probe_cfg = dict(cfg)
        probe_cfg["capture"] = resolve_capture_mode(cfg, prefer_client=prefer_client)
        reqs = check_wake_word_requirements(probe_cfg)
        transport = current_transport() or _stdio_transport
        owner, owner_surface = _wake_owner_snapshot()
        owned_by_caller = owns_listener(transport)
        listening = owned_by_caller and is_listening()
        silent = listening and audio_is_silent()
        input_device = get_input_device_status(cfg)
        hint = reqs.get("hint", "")
        if input_device.get("error") and not hint:
            hint = f"Wake-word input device could not be resolved: {input_device['error']}"
        if silent and not hint:
            hint = silent_audio_hint(input_device)
        # Effective capture: prefer the *armed* detector over config/auto.
        # With capture:auto the GUI arms client mode, but a bare status probe
        # would otherwise report "local" and the desktop would not reattach
        # the PCM feeder after wake.detected.
        frame = detector_frame_info()
        if owned_by_caller and frame.get("external_audio"):
            capture = "client"
        elif owned_by_caller and listening:
            capture = "local"
        else:
            capture = probe_cfg.get("capture") or reqs.get("capture") or str(
                cfg.get("capture") or "auto"
            )
        return _ok(rid, {
            "listening": listening,
            "owned_by_caller": owned_by_caller,
            "owner_surface": owner_surface if owner is not None else None,
            "phrase": reqs["phrase"],
            "provider": reqs["provider"],
            "configured_surface": str(cfg.get("surface") or "auto"),
            "input_device": input_device,
            "available": reqs["available"],
            "hint": hint,
            # Config truth: clients use this to re-arm after a voice turn
            # ("permanent on") without guessing from runtime listener state.
            "enabled": bool(cfg.get("enabled")),
            # Armed but deaf despite an open stream; see platform-specific hint.
            "audio_silent": silent,
            "capture": capture,
            "local_input_available": bool(reqs.get("local_input_available")),
            "sample_rate": frame.get("sample_rate", 16000),
            "frame_length": frame.get("frame_length", 1280),
        })
    except Exception as e:
        return _err(rid, 5026, str(e))


@method("wake.feed")
def _(rid, params: dict) -> dict:
    """Push client-captured PCM into the armed wake detector.

    Params:
      pcm: base64-encoded int16 mono little-endian samples (preferred), OR
      pcm_b64: alias of pcm
    Optional:
      sample_rate: must be 16000 (ignored if missing; mismatched rates rejected)

    Used when ``wake.start`` returned ``capture: "client"`` so remote backends
    without a microphone can still run openWakeWord on Mac/desktop audio.
    """
    transport = current_transport() or _stdio_transport
    raw_b64 = params.get("pcm") or params.get("pcm_b64") or ""
    if not isinstance(raw_b64, str) or not raw_b64.strip():
        return _err(rid, 4001, "wake.feed requires base64 pcm")
    try:
        import base64
        pcm = base64.b64decode(raw_b64, validate=False)
    except Exception as e:
        return _err(rid, 4001, f"invalid base64 pcm: {e}")
    if not pcm:
        return _ok(rid, {"fed": False, "reason": "empty"})
    # Soft size cap: 64000 bytes = 2s of 16 kHz int16 mono
    if len(pcm) > 64000:
        return _err(rid, 4001, "pcm frame too large")
    sr = params.get("sample_rate")
    if sr is not None and int(sr) not in (0, 16000):
        return _err(rid, 4001, "wake.feed only accepts 16 kHz PCM")
    try:
        from tools.wake_word import feed_audio
        ok = feed_audio(owner=transport, pcm_int16=pcm)
    except Exception as e:
        logger.debug("wake.feed failed: %s", e)
        return _err(rid, 5026, str(e))
    return _ok(rid, {"fed": bool(ok), "reason": None if ok else "not_owner"})


@method("voice.toggle")
def _(rid, params: dict) -> dict:
    """CLI parity for the ``/voice`` slash command.

    Subcommands:

    * ``status`` — report mode + TTS flags (default when action is unknown).
    * ``on`` / ``off`` — flip voice *mode* (the umbrella bit). Turning it
      off also tears down any active continuous recording loop. Does NOT
      start recording on its own; recording is driven by ``voice.record``
      (Ctrl+B) after mode is on, matching cli.py's enable/Ctrl+B split.
    * ``tts`` — toggle speech-output of agent replies. Requires mode on
      (mirrors CLI's _toggle_voice_tts guard).
    """
    action = params.get("action", "status")

    if action == "status":
        # Mirror CLI's _show_voice_status: include STT/TTS provider
        # availability so the user can tell at a glance *why* voice mode
        # isn't working ("STT provider: MISSING ..." is the common case).
        # ``record_key`` mirrors the configured ``voice.record_key`` so the
        # TUI can both bind it (frontend ``isVoiceToggleKey``) and display
        # it in /voice status — previously the TUI hardcoded Ctrl+B and
        # ignored the config (#18994).
        payload: dict = {
            "enabled": _voice_mode_enabled(),
            "record_key": _voice_record_key(),
            "tts": _voice_tts_enabled(),
        }
        try:
            from tools.voice_mode import check_voice_requirements

            reqs = check_voice_requirements()
            payload["available"] = bool(reqs.get("available"))
            payload["audio_available"] = bool(reqs.get("audio_available"))
            payload["stt_available"] = bool(reqs.get("stt_available"))
            payload["details"] = reqs.get("details") or ""
        except Exception as e:
            # check_voice_requirements pulls optional transcription deps —
            # swallow so /voice status always returns something useful.
            logger.warning("voice.toggle status: requirements probe failed: %s", e)

        return _ok(rid, payload)

    if action in {"on", "off"}:
        enabled = action == "on"
        # Runtime-only flag (CLI parity) — no _write_config_key, so the
        # next TUI launch starts with voice OFF instead of auto-REC from a
        # persisted stale toggle.
        os.environ["HERMES_VOICE"] = "1" if enabled else "0"

        stop_hint = ""
        if enabled:
            # Spoken-stop hint for the client to render on voice-mode start.
            # Sourced from voice.stop_phrases (custom phrases render
            # correctly); empty when the feature is disabled.
            try:
                from tools.voice_mode import voice_stop_hint

                stop_hint = voice_stop_hint()
            except Exception:
                stop_hint = ""

            # Voice mode with speech output already on (voice.auto_tts /
            # prior /voice tts) means replies will be spoken — warm the
            # engine now rather than on the first reply.
            if _voice_tts_enabled():
                _tts_lease_async("tui:voice-tts", True)

        if not enabled:
            # Disabling the mode must tear the continuous loop down; the
            # loop holds the microphone and would otherwise keep running.
            try:
                from hermes_cli.voice import stop_continuous

                stop_continuous()
            except ImportError:
                pass
            except Exception as e:
                logger.warning("voice: stop_continuous failed during toggle off: %s", e)

            # Clear TTS so it can be toggled independently after voice is off,
            # and silence any in-flight streaming speech.
            os.environ["HERMES_VOICE_TTS"] = "0"
            _tts_stream_stop(user_barge=False)
            _tts_lease_async("tui:voice-tts", False)

        return _ok(
            rid,
            {
                "enabled": enabled,
                "record_key": _voice_record_key(),
                "tts": _voice_tts_enabled(),
                "stop_hint": stop_hint,
            },
        )

    if action == "tts":
        if not _voice_mode_enabled():
            return _err(rid, 4014, "enable voice mode first: /voice on")
        new_value = not _voice_tts_enabled()
        # Runtime-only flag (CLI parity) — see voice.toggle on/off above.
        os.environ["HERMES_VOICE_TTS"] = "1" if new_value else "0"
        if not new_value:
            _tts_stream_stop(user_barge=False)
        # The TTS toggle is the "speech is about to be needed" signal: on →
        # pre-load the configured engine so the first reply starts hot; off →
        # release the lease (last holder gone = resident local model freed).
        _tts_lease_async("tui:voice-tts", new_value)
        # Include ``record_key`` on every branch so a /voice tts toggle
        # doesn't reset the TUI's cached shortcut to the default when a
        # user has a custom binding configured (Copilot review, round 2
        # on #19835). Keeps parity with the status/on/off branches above.
        return _ok(
            rid,
            {
                "enabled": True,
                "record_key": _voice_record_key(),
                "tts": new_value,
            },
        )

    return _err(rid, 4013, f"unknown voice action: {action}")


@method("voice.record")
def _(rid, params: dict) -> dict:
    """VAD-bounded push-to-talk capture, CLI-parity.

    ``start`` begins one VAD-bounded capture and emits ``voice.transcript``
    after silence stops the recorder. ``stop`` forces transcription of the
    active buffer, matching classic CLI push-to-talk. The voice wrapper retains
    no-speech counts across single-shot starts, so three consecutive silent
    captures emit ``voice.transcript`` with ``no_speech_limit=True``.
    """
    action = params.get("action", "start")
    wake_paused = False

    if action not in {"start", "stop"}:
        return _err(rid, 4019, f"unknown voice action: {action}")

    transport = current_transport() or _stdio_transport
    wake_owner, _surface = _wake_owner_snapshot()
    if wake_owner is not None and wake_owner is not transport:
        return _ok(rid, {"status": "busy", "reason": "wake_owned"})

    try:
        if action == "start":
            if not _voice_mode_enabled():
                return _err(rid, 4015, "voice mode is off — enable with /voice on")

            with _voice_sid_lock:
                global _voice_event_sid, _voice_wake_owner
                _voice_event_sid = params.get("session_id") or _voice_event_sid

            from hermes_cli.voice import start_continuous

            # Register the agent-busy probe so the shared voice wrapper can
            # hold the no-speech counter during long agent turns (item:
            # silence must not end the chat while the agent works). Safe to
            # re-register on every start; older wrappers without the setter
            # are tolerated.
            try:
                from hermes_cli.voice import set_voice_busy_probe

                set_voice_busy_probe(_any_session_running)
            except Exception:
                pass

            # Shape-safe lookups: malformed ``voice:`` YAML (bool/scalar/list)
            # must not crash /voice with a 5025 — fall back to VAD defaults.
            #
            # Exclude ``bool`` from the numeric check since Python's bool is
            # a subclass of int — a hand-edit like ``silence_threshold: true``
            # would otherwise forward as ``1`` instead of falling back to
            # the documented 200 / 3.0 defaults (Copilot round-12 on #19835).
            voice_cfg = _voice_cfg_dict()
            threshold = voice_cfg.get("silence_threshold")
            duration = voice_cfg.get("silence_duration")
            safe_threshold = (
                threshold
                if isinstance(threshold, (int, float))
                and not isinstance(threshold, bool)
                else 200
            )
            safe_duration = (
                duration
                if isinstance(duration, (int, float)) and not isinstance(duration, bool)
                else 3.0
            )
            # Hand the mic to STT if the wake-word detector holds it; resume
            # once a terminal capture event fires (one-shot transcript / silence
            # limit), so wake-triggered and manual captures both coexist.
            try:
                from tools.wake_word import pause_listening

                wake_paused = pause_listening(owner=transport)
            except Exception:
                wake_paused = False
            if wake_paused:
                with _voice_sid_lock:
                    _voice_wake_owner = transport

            def _on_transcript(t):
                _voice_emit("voice.transcript", {"text": t})
                _resume_voice_wake()

            def _on_silent():
                _voice_emit("voice.transcript", {"no_speech_limit": True})
                _resume_voice_wake()

            def _on_stop_phrase(t):
                # Explicit user intent: the user SAID a bare stop phrase
                # ("stop"). End the voice chat exactly like a manual
                # /voice off — flip the mode flags and silence any live
                # streaming TTS — and emit a distinct signal so clients
                # (TUI, desktop) end the conversation instead of treating
                # it as a no-speech timeout. The continuous loop has
                # already halted before this callback fires.
                os.environ["HERMES_VOICE"] = "0"
                os.environ["HERMES_VOICE_TTS"] = "0"
                try:
                    _tts_stream_stop(user_barge=False)
                except Exception:
                    pass
                _voice_emit("voice.transcript", {"stop_phrase": True, "text": t})
                _resume_voice_wake()

            def _on_status(state):
                _voice_emit("voice.status", {"state": state})
                if state == "idle":
                    _resume_voice_wake()

            # voice.max_recording_seconds — hard cap on a single recording's
            # length. Same guard as the silence params: non-numeric / bool /
            # missing falls back to the documented 120 default, while an
            # explicit numeric value <= 0 disables the cap (0.0).
            max_rec = voice_cfg.get("max_recording_seconds")
            safe_max_rec = (
                (max_rec if max_rec > 0 else 0.0)
                if isinstance(max_rec, (int, float)) and not isinstance(max_rec, bool)
                else 120.0
            )
            started = start_continuous(
                on_transcript=_on_transcript,
                on_status=_on_status,
                on_silent_limit=_on_silent,
                silence_threshold=safe_threshold,
                silence_duration=safe_duration,
                auto_restart=False,
                max_recording_seconds=safe_max_rec,
                on_stop_phrase=_on_stop_phrase,
            )
            if started is False:
                _resume_voice_wake()
                return _ok(rid, {"status": "busy"})
            return _ok(rid, {"status": "recording"})

        # action == "stop"
        with _voice_sid_lock:
            _voice_event_sid = params.get("session_id") or _voice_event_sid

        from hermes_cli.voice import stop_continuous

        stop_continuous(force_transcribe=True)
        _resume_voice_wake()
        return _ok(rid, {"status": "stopped"})
    except ImportError:
        if wake_paused or action == "stop":
            _resume_voice_wake()
        return _err(
            rid, 5025, "voice module not available — install audio dependencies"
        )
    except Exception as e:
        if wake_paused or action == "stop":
            _resume_voice_wake()
        return _err(rid, 5025, str(e))


@method("voice.tts")
def _(rid, params: dict) -> dict:
    text = params.get("text", "")
    if not text:
        return _err(rid, 4020, "text required")
    try:
        # Import check up front so a missing voice module still returns the
        # documented 5026 instead of failing silently in the thread.
        import hermes_cli.voice  # noqa: F401

        threading.Thread(
            target=_speak_text_with_barge, args=(text,), daemon=True
        ).start()
        return _ok(rid, {"status": "speaking"})
    except ImportError:
        return _err(rid, 5026, "voice module not available")
    except Exception as e:
        return _err(rid, 5026, str(e))


def register(server) -> None:
    """Publish this module's helpers + handlers onto ``server``, rebound to its globals."""
    bind_module(globals(), server, skip=("_",))
