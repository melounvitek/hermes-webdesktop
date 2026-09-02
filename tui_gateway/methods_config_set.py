"""``config.set`` — one JSON-RPC method, dispatched on ``key`` through a table.

Bodies are rebound onto server.py's globals (method_ctx.bind_module) and reference them bare.
Each ``_set_*`` handler takes ``(rid, params, key, value, session)`` and returns the JSON-RPC
envelope. Keys match exactly except ``details_mode.<section>`` (prefix) and ``_DISPLAY_TOGGLE_KEYS``.
"""

import os

from hermes_constants import INDICATOR_STYLES

from .method_ctx import HandlerRegistry, bind_module

_registry = HandlerRegistry()
method = _registry.method
_profile_scoped = _registry.profile_scoped


# ── shared helpers ────────────────────────────────────────────────────


def _display_section(cfg: dict) -> dict:
    display = cfg.get("display")
    return display if isinstance(display, dict) else {}


def _write_display_sections(*, sections=None, drop_sections=(), **display_fields) -> None:
    """Persist ``display.<field>`` + ``display.sections`` edits via the raw (uncached) config write-back."""
    cfg = _load_cfg_raw()
    display = _display_section(cfg)
    cur = display.get("sections")
    cur = cur if isinstance(cur, dict) else {}
    display.update(display_fields)
    cur.update(sections or {})
    for name in drop_sections:
        cur.pop(name, None)
    display["sections"] = cur
    cfg["display"] = display
    _save_cfg(cfg)


def _emit_session_info(sid: str, session: dict) -> None:
    agent = session.get("agent")
    if agent is not None:
        _emit("session.info", sid, _session_info(agent, session))


def _emit_all_session_info() -> None:
    for sid, sess in list(_sessions.items()):
        _emit_session_info(sid, sess)


def _toggle_display_bool(rid, key, value, *, cfg_key, on_words, off_words):
    """Shared body of the on/off/toggle display booleans (``density``, ``battery``)."""
    raw = str(value or "").strip().lower()
    cur_b = bool(_display_section(_load_cfg()).get(cfg_key, False))
    if raw in {"", "toggle"}:
        nv_b = not cur_b
    elif raw in on_words:
        nv_b = True
    elif raw in off_words:
        nv_b = False
    else:
        return _err(rid, 4002, f"unknown {key} value: {value}")
    _write_config_key(f"display.{cfg_key}", nv_b)
    return _ok(rid, {"key": key, "value": "on" if nv_b else "off"})


def _cfgset_await_agent(session, rid):
    """Wait for an in-progress agent build; the error envelope if it failed, else None."""
    init_err = _wait_agent(session, rid)
    if init_err:
        return init_err
    if session.get("agent") is None:
        return _err(rid, 5032, "agent initialization failed")
    return None


def _cfgset_model_ok(rid, key, value, warning, confirm_required, confirm_message, scope, **extra):
    return _ok(rid, {"key": key, "value": value, "warning": warning, "confirm_required": confirm_required,
                     "confirm_message": confirm_message, "scope": scope, **extra})


# ── per-key handlers ──────────────────────────────────────────────────


def _set_model(rid, params, key, value, session):
    """Live/deferred model switch; see _apply_model_switch and _apply_pending_model_switch."""
    try:
        if not value:
            return _err(rid, 4002, "model value required")
        confirmed = bool(params.get("confirm_expensive_model", False))
        if session:
            from hermes_cli.model_switch import parse_model_switch_args

            sid = params.get("session_id", "")
            # No live swap while a turn streams: agent.switch_model() mutates model/provider/
            # base_url/client that the worker thread reads every iteration. Stash the pick and
            # apply it at the NEXT turn start (_apply_pending_model_switch).
            if session.get("running"):
                parsed = parse_model_switch_args(value)
                try:
                    pending_model = parsed.model_input
                except Exception:
                    pending_model = str(value)
                pending_provider = (getattr(parsed, "explicit_provider", "") or "").strip()
                # Selection guards run HERE (the only moment a confirm round-trip is possible);
                # otherwise an unconfirmed stashed pick is dropped at turn start, never confirmed.
                if not confirmed:
                    pending_warning = _pending_switch_selection_warning(pending_model, pending_provider)
                    if pending_warning is not None:
                        # Nothing stashed; the client re-sends with confirm_expensive_model.
                        # `confirm_message` is canonical, `warning` its legacy alias — identical.
                        return _cfgset_model_ok(
                            rid, key, pending_model, pending_warning, True, pending_warning, "session", deferred=False
                        )
                session["pending_model_switch"] = {
                    "raw": value,
                    "confirm_expensive_model": confirmed,
                    # _session_info reports these while pending so the end-of-turn settle keeps
                    # showing the user's pick instead of the still-live old model.
                    "display_model": pending_model,
                    "display_provider": pending_provider,
                }
                return _cfgset_model_ok(rid, key, pending_model, "", False, "", "session", deferred=True)
            parsed_flags = parse_model_switch_args(value)
            explicit_provider = parsed_flags.explicit_provider
            failed_agent_init = session.get("agent") is None and session.get("agent_error") is not None
            failed_ready = session.get("agent_ready") if failed_agent_init else None
            if failed_agent_init:
                if failed_ready is None:
                    return _err(rid, 5032, session.get("agent_error") or "agent initialization failed")
                if not failed_ready.wait(timeout=30.0):
                    return _err(rid, 5032, "agent initialization timed out")
            failed_agent_init = (
                failed_agent_init and session.get("agent") is None and session.get("agent_error") is not None
                and session.get("agent_ready") is failed_ready and failed_ready.is_set()
            )
            if session.get("agent") is None and not explicit_provider.strip() and not failed_agent_init:
                _start_agent_build(sid, session)
                init_err = _cfgset_await_agent(session, rid)
                if init_err:
                    return init_err
            with _session_profile_runtime_scope(session):
                result = _apply_model_switch(
                    sid, session, value, confirm_expensive_model=confirmed, parsed_flags=parsed_flags
                )
            if failed_agent_init and not result.get("confirm_required"):
                _restart_completed_failed_agent_build(sid, session, failed_ready)
                init_err = _cfgset_await_agent(session, rid)
                if init_err:
                    return init_err
                with _session_profile_runtime_scope(session):
                    _persist_live_session_runtime(session)
        else:
            result = _apply_model_switch("", {"agent": None}, value, confirm_expensive_model=confirmed)
        return _cfgset_model_ok(
            rid, key, result["value"], result["warning"], result.get("confirm_required", False),
            result.get("confirm_message", ""), result.get("scope", "session"),
        )
    except Exception as e:
        return _err(rid, 5001, str(e))


def _set_fast(rid, params, key, value, session):
    raw = str(value or "").strip().lower()
    agent = session.get("agent") if session else None
    if agent is not None:
        current_tier = getattr(agent, "service_tier", None)
    elif session is not None and session.get("create_service_tier_override") is not None:
        # Pre-build session with a pinned tier: report/toggle from the pin, not the global.
        current_tier = session["create_service_tier_override"] or None
    else:
        current_tier = _load_service_tier()
    current_fast = current_tier == "priority"

    if raw in {"status"}:
        return _ok(rid, {"key": key, "value": {"priority": "fast", None: "normal"}.get(current_tier, current_tier)})
    if raw in {"", "toggle"}:
        nv = "normal" if current_fast else "fast"
    elif raw in {"fast", "on"}:
        nv = "fast"
    elif raw in {"normal", "off"}:
        nv = "normal"
    elif raw in {"auto", "cold"}:
        nv = raw
    else:
        return _err(rid, 4002, f"unknown fast mode: {value}")

    overrides = None
    if nv == "fast":
        from hermes_cli.models import resolve_fast_mode_overrides

        if agent is not None:
            target_model = getattr(agent, "model", None)
        else:
            # A pre-build session may carry a picked model (desktop draft) — validate against THAT.
            session_override = (session or {}).get("model_override") or {}
            target_model = (isinstance(session_override, dict) and session_override.get("model")) or _resolve_model()
        if not target_model:
            return _err(rid, 4002, "fast mode is not available without a selected model")
        overrides = resolve_fast_mode_overrides(
            target_model, provider=getattr(agent, "provider", None), base_url=getattr(agent, "base_url", None)
        )
        if overrides is None:
            return _err(rid, 4002, "fast mode is not available for this model")

    if session is not None:
        # Session-scoped like `reasoning` (global persistence is `--global` / Settings → Model):
        # writing config.yaml here flipped fast mode for every other session/profile/CLI/gateway.
        # The create override keeps the choice across lazy builds and rebuilds; "" pins normal.
        session["create_service_tier_override"] = {"fast": "priority", "normal": ""}.get(nv, nv)
    else:
        _write_config_key("agent.service_tier", nv)
    if agent is not None:
        agent.service_tier = {"fast": "priority", "normal": None}.get(nv, nv)
        current_overrides = dict(getattr(agent, "request_overrides", {}) or {})
        current_overrides.pop("service_tier", None)
        current_overrides.pop("speed", None)
        if nv == "fast":
            current_overrides.update(overrides)
        agent.request_overrides = current_overrides
        _persist_live_session_runtime(session)
        _emit_session_info(params.get("session_id", ""), session)
    return _ok(rid, {"key": key, "value": nv})


def _set_busy(rid, params, key, value, session):
    raw = str(value or "").strip().lower()
    if raw in {"", "status"}:
        return _ok(rid, {"key": key, "value": _load_busy_input_mode()})
    if raw not in {"queue", "steer", "interrupt"}:
        return _err(rid, 4002, f"unknown busy mode: {value}")
    _write_config_key("display.busy_input_mode", raw)
    return _ok(rid, {"key": key, "value": raw})


def _set_verbose(rid, params, key, value, session):
    cycle = ["off", "new", "all", "verbose"]
    cur = session.get("tool_progress_mode", _load_tool_progress_mode()) if session else _load_tool_progress_mode()
    if value and value != "cycle":
        nv = str(value).strip().lower()
        if nv not in cycle:
            return _err(rid, 4002, f"unknown verbose mode: {value}")
    else:
        idx = cycle.index(cur) if cur in cycle else 2
        nv = cycle[(idx + 1) % len(cycle)]
    _write_config_key("display.tool_progress", nv)
    if session:
        session["tool_progress_mode"] = nv
        agent = session.get("agent")
        if agent is not None:
            agent.verbose_logging = nv == "verbose"
    return _ok(rid, {"key": key, "value": nv})


def _set_focus(rid, params, key, value, session):
    # Focus view (/focus): display-only reduced output composed with tool_progress — enabling
    # stashes the configured mode and pins tool_progress "off"; disabling restores the stash.
    from hermes_cli.focus_view import FOCUS_TOOL_PROGRESS_MODE, normalize_tool_progress_mode, resolve_focus_arg

    d_f = _display_section(_load_cfg())
    cur_focus = bool(d_f.get("focus_view", False))
    action, target = resolve_focus_arg(str(value or ""), cur_focus)
    if action == "usage":
        return _err(rid, 4002, f"unknown focus value: {value} (use on|off|status)")
    if action == "status" or target is None:
        return _ok(rid, {"key": key, "value": "on" if cur_focus else "off", "tool_progress": _load_tool_progress_mode()})

    if target:
        saved = (cur_focus and d_f.get("focus_saved_tool_progress")) or _load_tool_progress_mode()
        _write_config_key("display.focus_saved_tool_progress", normalize_tool_progress_mode(saved))
        _write_config_key("display.tool_progress", FOCUS_TOOL_PROGRESS_MODE)
        effective = FOCUS_TOOL_PROGRESS_MODE
    else:
        effective = normalize_tool_progress_mode(d_f.get("focus_saved_tool_progress") or "all")
        _write_config_key("display.tool_progress", effective)
    _write_config_key("display.focus_view", bool(target))

    if session:
        session["focus_view"] = bool(target)
        session["tool_progress_mode"] = effective
        agent_f = session.get("agent")
        if agent_f is not None:
            with contextlib.suppress(Exception):
                agent_f.tool_progress_mode = effective
    return _ok(rid, {"key": key, "value": "on" if target else "off", "tool_progress": effective})


def _set_approval_mode(rid, params, key, value, session):
    raw = str(value or "").strip().lower()
    if raw not in _APPROVAL_MODES:
        return _err(rid, 4002, f"unknown approval mode: {value}; pick one of manual|smart|off")
    _write_config_key("approvals.mode", raw)
    _emit_all_session_info()
    return _ok(rid, {"key": "approvals.mode", "value": raw})


def _set_yolo(rid, params, key, value, session):
    # Approval bypass. scope="session" (default; TUI Shift+Tab) toggles ONLY this session's flag.
    # scope="global" (Shift+click the zap) flips persistent approvals.mode between "off" (bypass
    # on) and "manual" (bypass off) for every surface, surviving restarts.
    scope = str(params.get("scope") or "session").strip().lower()
    try:
        from tools.approval import disable_session_yolo, enable_session_yolo, is_session_yolo_enabled

        raw = str(value or "").strip().lower()

        def _resolve_toggle(current: bool) -> bool:
            return _BOOL_WORDS.get(raw, not current)

        if scope == "global":
            from tools.approval import _normalize_approval_mode

            cfg = _load_cfg()
            appr = cfg.get("approvals") if isinstance(cfg, dict) else None
            appr = appr if isinstance(appr, dict) else {}
            enable = _resolve_toggle(_normalize_approval_mode(appr.get("mode", "manual")) == "off")
            # Binary affordance: no restore of a prior "smart"/custom mode (those live in config.yaml).
            _write_config_key("approvals.mode", "off" if enable else "manual")
            _emit_all_session_info()  # reflect the flip in every live indicator
            return _ok(rid, {"key": key, "value": "1" if enable else "0", "scope": "global"})

        if session:
            skey = session["session_key"]
            enable = _resolve_toggle(is_session_yolo_enabled(skey))
            (enable_session_yolo if enable else disable_session_yolo)(skey)
            _emit_session_info(params.get("session_id", ""), session)
        else:
            enable = _resolve_toggle(is_truthy_value(os.environ.get("HERMES_YOLO_MODE")))
            if enable:
                os.environ["HERMES_YOLO_MODE"] = "1"
            else:
                os.environ.pop("HERMES_YOLO_MODE", None)
        return _ok(rid, {"key": key, "value": "1" if enable else "0", "scope": "session"})
    except Exception as e:
        return _err(rid, 5001, str(e))


# /reasoning display words: (accepted inputs, reported value, display field, sections.thinking,
# session show_reasoning or None). full/clamp mirror the CLI's reasoning_full toggle; the TUI
# renders thinking as an expand/collapse section and display.reasoning_full is persisted too.
_REASONING_DISPLAY_WORDS = (
    ({"show", "on"}, "show", {"show_reasoning": True}, "expanded", True),
    ({"hide", "off"}, "hide", {"show_reasoning": False}, "hidden", False),
    ({"full", "all"}, "full", {"reasoning_full": True}, "expanded", None),
    ({"clamp", "collapse", "short"}, "clamp", {"reasoning_full": False}, "collapsed", None),
)


def _set_reasoning(rid, params, key, value, session):
    try:
        from hermes_constants import parse_reasoning_effort

        arg = str(value or "").strip().lower()
        scope = str(params.get("scope") or "").strip().lower()
        for words, reported, fields, thinking, show in _REASONING_DISPLAY_WORDS:
            if arg in words:
                _write_display_sections(sections={"thinking": thinking}, **fields)
                if show is not None and session:
                    session["show_reasoning"] = show
                return _ok(rid, {"key": key, "value": reported})

        parsed = parse_reasoning_effort(arg)
        if parsed is None:
            return _err(rid, 4002, f"unknown reasoning value: {value}")
        if scope == "global" or session is None:
            _write_config_key("agent.reasoning_effort", arg)
            if session is not None:
                session.pop("create_reasoning_override", None)
        else:
            # Session-scoped like the messaging gateway's `/reasoning <level>`; otherwise every
            # desktop model-menu pick rewrote the global default.
            session["create_reasoning_override"] = parsed
        if session and session.get("agent") is not None:
            session["agent"].reasoning_config = parsed
            _persist_live_session_runtime(session)
            _emit_session_info(params.get("session_id", ""), session)
        return _ok(rid, {"key": key, "value": arg})
    except Exception as e:
        return _err(rid, 5001, str(e))


def _set_details_mode(rid, params, key, value, session):
    nv = str(value or "").strip().lower()
    if nv not in _DETAIL_MODES:
        return _err(rid, 4002, f"unknown details_mode: {value}")
    _write_display_sections(sections={section: nv for section in _DETAIL_SECTION_NAMES}, details_mode=nv)
    return _ok(rid, {"key": key, "value": nv})


def _set_details_section(rid, params, key, value, session):
    # `details_mode.<section>` -> `display.sections.<section>`. Empty value clears the explicit
    # override so the frontend applies built-in section defaults before the global details_mode.
    section = key.split(".", 1)[1]
    if section not in _DETAIL_SECTION_NAMES:
        return _err(rid, 4002, f"unknown section: {section}")
    nv = str(value or "").strip().lower()
    if not nv:
        _write_display_sections(drop_sections=(section,))
    elif nv not in _DETAIL_MODES:
        return _err(rid, 4002, f"unknown details_mode: {value}")
    else:
        _write_display_sections(sections={section: nv})
    return _ok(rid, {"key": key, "value": nv})


def _set_thinking_mode(rid, params, key, value, session):
    nv = str(value or "").strip().lower()
    if nv not in {"collapsed", "truncated", "full"}:
        return _err(rid, 4002, f"unknown thinking_mode: {value}")
    _write_config_key("display.thinking_mode", nv)
    # Backward compatibility bridge: keep details_mode aligned.
    _write_config_key("display.details_mode", "expanded" if nv == "full" else "collapsed")
    return _ok(rid, {"key": key, "value": nv})


def _set_density(rid, params, key, value, session):
    return _toggle_display_bool(rid, key, value, cfg_key="tui_compact", on_words={"on"}, off_words={"off"})


def _set_battery(rid, params, key, value, session):
    return _toggle_display_bool(
        rid, key, value, cfg_key="battery", on_words={"on", "true", "yes"}, off_words={"off", "false", "no"}
    )


def _set_theme(rid, params, key, value, session):
    # 'light'/'dark' pin beats background auto-detection (xterm.js hosts misreport OSC 11).
    raw = str(value or "").strip().lower()
    if raw not in {"auto", "light", "dark"}:
        return _err(rid, 4002, f"unknown theme value: {value} (use auto|light|dark)")
    _write_config_key("display.tui_theme", raw)
    return _ok(rid, {"key": key, "value": raw})


def _set_statusbar(rid, params, key, value, session):
    raw = str(value or "").strip().lower()
    current = _coerce_statusbar(_display_section(_load_cfg()).get("tui_statusbar", "top"))
    if raw in {"", "toggle"}:
        nv = "top" if current == "off" else "off"
    elif raw == "on":
        nv = "top"
    elif raw in _STATUSBAR_MODES:
        nv = raw
    else:
        return _err(rid, 4002, f"unknown statusbar value: {value}")
    _write_config_key("display.tui_statusbar", nv)
    return _ok(rid, {"key": key, "value": nv})


def _set_mouse(rid, params, key, value, session):
    # Explicit None check (not `value or ""`) so falsy non-string inputs (0, False from
    # programmatic callers) reach the alias map as themselves (-> 'off') instead of toggling.
    raw = ("" if value is None else str(value)).strip().lower()
    current = _display_mouse_tracking(_display_section(_load_cfg()))
    if raw in {"", "toggle"}:
        nv = "all" if current == "off" else "off"
    elif raw in _MOUSE_TRACKING_ALIASES:
        nv = _MOUSE_TRACKING_ALIASES[raw]
    else:
        return _err(rid, 4002, f"unknown mouse value: {value}")
    _write_config_key("display.mouse_tracking", nv)
    return _ok(rid, {"key": key, "value": nv})


def _set_indicator(rid, params, key, value, session):
    # Explicit None check so falsy non-string inputs (0, False, []) surface in the error message.
    raw = ("" if value is None else str(value)).strip().lower()
    if raw not in INDICATOR_STYLES:
        return _err(rid, 4002, f"unknown indicator: {raw!r}; pick one of {'|'.join(INDICATOR_STYLES)}")
    _write_config_key("display.tui_status_indicator", raw)
    return _ok(rid, {"key": key, "value": raw})


def _set_cwd(rid, params, key, value, session):
    raw = str(value or "").strip()
    if not raw:
        return _err(rid, 4002, "cwd required")
    cwd = os.path.abspath(os.path.expanduser(raw))
    if not os.path.isdir(cwd):
        return _err(rid, 4002, f"working directory does not exist: {raw}")
    _write_config_key("terminal.cwd", cwd)
    os.environ["TERMINAL_CWD"] = cwd
    return _ok(rid, {"key": "terminal.cwd", "value": cwd, "cwd": cwd, "branch": _git_branch_for_cwd(cwd)})


def _set_prompt_like(rid, params, key, value, session):
    try:
        cfg = _load_cfg_raw()  # write-back round-trip ("prompt" saves cfg)
        resp = {"key": key, "value": value}
        if key == "prompt":
            if value == "clear":
                cfg.pop("custom_prompt", None)
                resp["value"] = ""
            else:
                cfg["custom_prompt"] = value
            _save_cfg(cfg)
        elif key == "personality":
            pname, new_prompt = _validate_personality(str(value or ""), cfg)
            # Personality text is an in-session overlay; persistence goes through
            # hermes_cli.personality (single owner), never the user-owned global system prompt.
            from hermes_cli.personality import persist_personality

            persist_personality(pname)
            resp["value"] = str(value or "none")
            history_reset, info = _apply_personality_to_session(params.get("session_id", ""), session, new_prompt, pname)
            resp["history_reset"] = history_reset
            if info is not None:
                resp["info"] = info
        else:
            _write_config_key(f"display.{key}", value)
            if key == "skin":
                # Every connected surface repaints; then sync the watcher baseline so the poll
                # loop doesn't re-broadcast the skin this RPC just applied.
                _broadcast_global_event("skin.changed", resolve_skin())
                _note_skin_broadcast()
        return _ok(rid, resp)
    except Exception as e:
        return _err(rid, 5001, str(e))


def _set_display_toggle(rid, params, key, value, session):
    on = _BOOL_WORDS.get(str(value).strip().lower())
    if on is None:
        return _err(rid, 4002, f"{key} takes true or false")
    _write_config_key(key, on)
    return _ok(rid, {"key": key, "value": on})


# ── dispatch ──────────────────────────────────────────────────────────

_CONFIG_SETTERS = {
    "model": _set_model, "fast": _set_fast, "busy": _set_busy, "verbose": _set_verbose, "focus": _set_focus,
    "approval_mode": _set_approval_mode, "approvals.mode": _set_approval_mode, "yolo": _set_yolo,
    "reasoning": _set_reasoning, "details_mode": _set_details_mode, "thinking_mode": _set_thinking_mode,
    "density": _set_density, "battery": _set_battery, "theme": _set_theme, "statusbar": _set_statusbar,
    "mouse": _set_mouse, "indicator": _set_indicator,
    "cwd": _set_cwd, "terminal.cwd": _set_cwd, "workdir": _set_cwd,
    "prompt": _set_prompt_like, "personality": _set_prompt_like, "skin": _set_prompt_like,
}


def _config_setter(key: str):
    handler = _CONFIG_SETTERS.get(key)
    if handler is not None:
        return handler
    if key.startswith("details_mode."):
        return _set_details_section
    if key in _DISPLAY_TOGGLE_KEYS:
        return _set_display_toggle
    return None


@method("config.set")
@_profile_scoped
def _(rid, params: dict) -> dict:
    key, value = params.get("key", ""), params.get("value", "")
    session = _sessions.get(params.get("session_id", ""))
    handler = _config_setter(key)
    if handler is None:
        return _err(rid, 4002, f"unknown config key: {key}")
    return handler(rid, params, key, value, session)


def register(server) -> None:
    """Publish helpers + the config.set handler onto ``server``, rebound to its globals."""
    bind_module(globals(), server, skip=("_",))
