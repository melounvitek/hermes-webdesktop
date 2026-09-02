"""Compute-host (turn isolation) bridge: relay prompts/controls to the per-session child process and mirror its metadata/clarify/compress acks back into the session.

Bodies are rebound onto server.py's globals at install time (see
method_ctx.bind_module), so they reference server.py globals bare.
"""

from __future__ import annotations

import threading

from .method_ctx import HandlerRegistry, bind_module

_registry = HandlerRegistry()


_compute_host_supervisor = None
_compute_host_supervisor_lock = threading.Lock()
# Cap on how long session.compress blocks its RPC on the compute host. Must stay
# below the desktop's SESSION_COMPRESS_TIMEOUT_MS (660s) so the client gets the
# `pending` answer, not its own timeout; the late-ack path covers anything slower.
_COMPUTE_HOST_COMPRESS_WAIT_CAP_SECS = 630.0


def _inside_compute_host_child() -> bool:
    return os.environ.get("HERMES_COMPUTE_HOST_CHILD") == "1"


def _turn_isolation_enabled(cfg: dict | None = None) -> bool:
    if _inside_compute_host_child():
        return False
    isolation_cfg = cfg or _load_dashboard_process_isolation_config()
    return bool(isolation_cfg.get("turn_isolation"))


def _session_uses_compute_host(session: dict, cfg: dict | None = None) -> bool:
    if not _turn_isolation_enabled(cfg):
        return False
    # Routes lazy sessions whose AIAgent was never built in-process; already-built
    # sessions keep the in-process path unless a prior isolated turn marked host ownership.
    return bool(session.get("_compute_host_active")) or (
        session.get("agent") is None and session.get("agent_ready") is not None
    )


def _get_compute_host_supervisor(cfg: dict | None = None):
    global _compute_host_supervisor
    isolation_cfg = cfg or _load_dashboard_process_isolation_config()
    with _compute_host_supervisor_lock:
        if _compute_host_supervisor is None:
            from tui_gateway.host_supervisor import HostSupervisor

            _compute_host_supervisor = HostSupervisor(
                rpc_sink=_relay_compute_host_rpc,
                heartbeat_secs=int(isolation_cfg.get("compute_host_heartbeat_secs") or 15),
                respawn_max=int(isolation_cfg.get("compute_host_respawn_max") or 3),
            )
        return _compute_host_supervisor


def _compute_host_turn_frame(
    rid: str, sid: str, session: dict, text: Any, image_paths: list[str] | None = None,
    queued_prompt_generation: int | None = None, display_kind: str | None = None,
) -> dict:
    with session["history_lock"]:
        history = list(session.get("history", []))
        history_version = int(session.get("history_version", 0))
        attached_images = list(image_paths if image_paths is not None else session.get("attached_images", []))
    return {
        "type": "turn.start",
        "sid": sid,
        "request_id": rid,
        "session_key": session.get("session_key") or sid,
        "text": text,
        **({"display_kind": display_kind} if display_kind else {}),
        "history": history,
        "history_version": history_version,
        "cols": int(session.get("cols", 80) or 80),
        "cwd": _session_cwd(session),
        "context_cwd_is_launch_artifact": _context_cwd_is_launch_artifact(session),
        "profile_home": session.get("profile_home") or "",
        "model_override": session.get("model_override"),
        "reasoning_config_override": session.get("create_reasoning_override"),
        "service_tier_override": session.get("create_service_tier_override"),
        "source": _session_source(session),
        "attached_images": attached_images,
        "queued_prompt_generation": queued_prompt_generation,
    }


def _metadata_mirror(session: dict | None) -> dict:
    mirror = (session or {}).get("_metadata_mirror")
    return mirror if isinstance(mirror, dict) else {}


def _compute_host_session_info(session: dict) -> dict:
    # Tolerate a legacy one-arg _session_info (tests patch it that way).
    try:
        return _session_info(session.get("agent"), session)
    except TypeError:
        return _session_info(session.get("agent"))


def _compute_host_adopt_frame_meta(session: dict, frame: dict) -> None:
    """Adopt a host frame's session_key / history_version. Caller holds history_lock."""
    if frame.get("session_key"):
        session["session_key"] = str(frame.get("session_key"))
    if frame.get("history_version") is not None:
        try:
            session["history_version"] = max(
                int(session.get("history_version", 0)),
                int(frame.get("history_version") or 0),
            )
        except Exception:
            pass


def _relay_compute_host_rpc(message: dict) -> bool:
    """Relay host events while retaining the clarify snapshot needed on resume."""
    params = message.get("params") if isinstance(message, dict) else None
    kind = params.get("type") if isinstance(params, dict) else None
    if kind in {"clarify.request", "clarify.expire"}:
        session = _sessions.get(str(params.get("session_id") or ""))
        payload = params.get("payload")
        request_id = payload.get("request_id") if isinstance(payload, dict) else None
        if session is not None and request_id:
            with session.get("history_lock", threading.Lock()):
                if kind == "clarify.request":
                    session["_compute_host_pending_clarify"] = dict(payload)
                else:
                    pending = session.get("_compute_host_pending_clarify")
                    if isinstance(pending, dict) and pending.get("request_id") == request_id:
                        session.pop("_compute_host_pending_clarify", None)
    return write_json(message)


def _compute_host_clarify_session(request_id: str) -> tuple[str, dict] | None:
    """Find the parent mirror for one host-owned clarify request."""
    if not request_id:
        return None
    for sid, session in list(_sessions.items()):
        with session.get("history_lock", threading.Lock()):
            pending = session.get("_compute_host_pending_clarify")
            if isinstance(pending, dict) and pending.get("request_id") == request_id:
                return sid, session
    return None


def _update_compute_host_clarify_snapshot(sid: str, session: dict, params: dict, result: dict) -> None:
    """Keep reconnect snapshots accurate while a batch clarify is answered."""
    request_id = str(params.get("request_id") or "")
    with session.get("history_lock", threading.Lock()):
        pending = session.get("_compute_host_pending_clarify")
        if not isinstance(pending, dict) or pending.get("request_id") != request_id:
            return
        if result.get("status") == "expired" or not result.get("remaining") and not params.get("question_id"):
            session.pop("_compute_host_pending_clarify", None)
            return
        question_id = str(params.get("question_id") or "")
        if question_id and isinstance(result.get("remaining"), list):
            answers = dict(pending.get("answers") or {})
            answers[question_id] = str(params.get("answer") or "")
            pending["answers"] = answers
            if not result["remaining"]:
                session.pop("_compute_host_pending_clarify", None)


def _respond_compute_host_clarify(rid: str, params: dict) -> dict | None:
    """Proxy a clarify answer into the process that owns its pending Event."""
    located = _compute_host_clarify_session(str(params.get("request_id") or ""))
    if located is None:
        return None
    sid, session = located
    if not _session_uses_compute_host(session):
        return None
    try:
        ack = _get_compute_host_supervisor().respond(sid, params)
    except Exception as exc:
        return _err(rid, 5019, f"compute-host clarify response failed: {exc}")
    if ack.get("type") == "respond.error":
        return _err(rid, 5019, str(ack.get("message") or "compute-host clarify response failed"))
    response = ack.get("response")
    if not isinstance(response, dict):
        return _err(rid, 5019, "compute-host clarify response returned an invalid response")
    if "error" in response:
        error = response["error"] if isinstance(response["error"], dict) else {}
        return _err(rid, int(error.get("code") or 5000), str(error.get("message") or "clarify response failed"))
    result = response.get("result")
    if not isinstance(result, dict):
        return _err(rid, 5019, "compute-host clarify response returned an invalid result")
    _update_compute_host_clarify_snapshot(sid, session, params, result)
    return _ok(rid, result)


def _apply_compute_host_metadata_mirror(session: dict, frame: dict | None) -> None:
    """Mirror host-owned session metadata: while turn isolation is active the host is
    the only writer of live agent/history state, and UI reads must not build a
    second in-process agent."""
    if not isinstance(frame, dict):
        return
    with session.get("history_lock", threading.Lock()):
        _compute_host_adopt_frame_meta(session, frame)
        if frame.get("message_count") is not None:
            try:
                session["_metadata_message_count"] = int(frame.get("message_count") or 0)
            except Exception:
                pass
    info = frame.get("session_info")
    if isinstance(info, dict):
        mirror = dict(_metadata_mirror(session))
        mirror.update(info)
        session["_metadata_mirror"] = mirror
        session["_metadata_mirror_updated_at"] = time.time()


def _on_compute_host_turn_done(rid: str, sid: str, session: dict, frame: dict) -> None:
    with session["history_lock"]:
        _compute_host_adopt_frame_meta(session, frame)
        session["running"] = False
        session["last_active"] = time.time()
        _clear_inflight_turn(session)
        session.pop("_compute_host_pending_clarify", None)
    if frame.get("type") == "turn.error":
        message = str(frame.get("message") or "compute host turn failed")
        _emit("message.complete", sid, {"text": f"Error: {message}", "status": "error"})
    _apply_compute_host_metadata_mirror(session, frame)
    info = _compute_host_session_info(session)
    if not frame.get("session_info_emitted"):
        _emit("session.info", sid, info)
    _drain_queued_prompt(rid, sid, session)


def _submit_prompt_to_compute_host(
    rid: str, sid: str, session: dict, text: Any, image_paths: list[str] | None = None,
    queued_prompt_generation: int | None = None, display_kind: str | None = None,
) -> dict:
    cfg = _load_dashboard_process_isolation_config()
    frame = _compute_host_turn_frame(
        rid, sid, session, text, image_paths=image_paths,
        queued_prompt_generation=queued_prompt_generation, display_kind=display_kind,
    )

    def _complete(done: dict) -> None:
        # submit_turn reports a synchronous pipe failure via the callback before
        # re-raising; leave the session untouched so prompt.submit can fail open
        # to the in-process path without a duplicate terminal error.
        if done.get("reason") == "send_failed":
            return
        _on_compute_host_turn_done(rid, sid, session, done)

    try:
        _get_compute_host_supervisor(cfg).submit_turn(frame, on_complete=_complete)
    except Exception as exc:
        return _err(rid, 5019, f"compute-host dispatch failed: {exc}")
    with session["history_lock"]:
        session["_compute_host_active"] = True
        if image_paths is None:
            session["attached_images"] = []
    return _ok(rid, {"status": "streaming", "turn_isolation": True})


def _send_compute_host_control(
    sid: str, *, route_name: str, command: str = "", payload: dict | None = None,
    wait: bool = True, timeout: float = 30.0, on_late_ack=None,
) -> dict:
    frame = dict(payload or {})
    frame.setdefault("type", "control")
    frame.setdefault("command", command)
    return _get_compute_host_supervisor().control(
        sid, route_name=route_name, payload=frame, wait=wait, timeout=timeout, on_late_ack=on_late_ack
    )


def _compute_host_compress_wait_seconds(cfg: dict | None = None) -> float:
    """RPC wait budget for a compute-host compress control: the configured
    ``compression.context_total_ceiling_seconds`` plus slack, capped below the
    desktop's RPC timeout (a fixed waiter reported false timeouts while the host
    kept working); anything slower lands via the late-ack path."""
    from agent.conversation_compression import resolve_context_compression_timeouts

    try:
        compression_cfg = (cfg if cfg is not None else _load_cfg()).get("compression", {})
    except Exception:
        compression_cfg = {}
    _idle, ceiling = resolve_context_compression_timeouts(compression_cfg if isinstance(compression_cfg, dict) else {})
    return float(min(max(ceiling + 30.0, 120.0), _COMPUTE_HOST_COMPRESS_WAIT_CAP_SECS))


def _announce_compute_host_compress_done(sid: str, session: dict, ack: dict) -> None:
    """Mirror a compress ack and push the same ``session.info`` + ``compacted`` edges
    the in-process /compress path emits, so a client whose RPC wait expired still
    learns the transcript changed."""
    _apply_compute_host_metadata_mirror(session, ack)
    _emit("session.info", sid, _compute_host_session_info(session))
    _status_update(sid, "compacted", "✓ Context compression complete")


def _adopt_late_compute_host_compress_ack(sid: str, session: dict, ack: dict, *, route_name: str) -> None:
    """Adopt a compress ack that arrived after its RPC waiter answered ``pending``:
    the only place the rotated session_key / history_version / mirror can land and
    the client's only signal. A late ``control.error`` goes out via ``error``."""
    with _sessions_lock:
        live = _sessions.get(sid)
    if live is not session:
        return
    if not isinstance(ack, dict) or ack.get("type") in {"control.error", "error"}:
        message = str((ack or {}).get("message") or f"compute-host {route_name} failed")
        _emit("error", sid, {"message": f"compression failed: {message}"})
        _status_update(sid, "ready")
        return
    _announce_compute_host_compress_done(sid, session, ack)


def register(server) -> None:
    """Publish this module's helpers + handlers onto ``server``, rebound to its globals."""
    bind_module(globals(), server, skip=("_",))
