"""``config.set`` — one JSON-RPC method, dispatched on ``key`` through a table.

Handlers are rebound onto server.py's globals at install time (see
method_ctx.bind_module), so bodies reference server.py globals bare
(``_ok``, ``_err``, ``_load_cfg``, ``_sessions``, ``_write_config_key``, ...).

Each ``_set_*`` handler takes ``(rid, params, key, value, session)`` and returns
the JSON-RPC envelope. Table order does not matter: keys are exact matches
except ``details_mode.<section>`` (prefix) and ``_DISPLAY_TOGGLE_KEYS`` (set),
which are tried after the exact table, in that order.
"""

import os

from hermes_constants import INDICATOR_STYLES

from .method_ctx import HandlerRegistry, bind_module

_registry = HandlerRegistry()
method = _registry.method
_profile_scoped = _registry.profile_scoped


# ── shared helpers ────────────────────────────────────────────────────


def _display_cfg(cfg: dict) -> dict:
    display = cfg.get("display")
    return display if isinstance(display, dict) else {}


def _write_display_sections(*, thinking=None, **display_fields) -> None:
    """Persist ``display.<field>`` values and optionally ``display.sections.thinking``.

    Write-back round-trip through the raw (uncached) config so other keys survive.
    """
    cfg = _load_cfg_raw()
    display = _display_cfg(cfg)
    sections = display.get("sections") if isinstance(display.get("sections"), dict) else {}
    display.update(display_fields)
    if thinking is not None:
        sections["thinking"] = thinking
    display["sections"] = sections
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
    cur_b = bool(_display_cfg(_load_cfg()).get(cfg_key, False))
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


# ── per-key handlers ──────────────────────────────────────────────────


def _set_model(rid, params, key, value, session):
    """Live/deferred model switch; see _apply_model_switch and _apply_pending_model_switch."""
    try:
        if not value:
            return _err(rid, 4002, "model value required")
        if session:
            from hermes_cli.model_switch import parse_model_switch_args

            # A live swap can't run in-place while a turn streams:
            # agent.switch_model() mutates self.model / self.provider /
            # self.base_url / self.client, and the worker thread running
            # agent.run_conversation reads those every iteration — a
            # mid-turn swap can fire an HTTP request with the new base_url
            # but old model (400/404s).  So instead of rejecting the pick
            # (the old 4009), stash it and apply it at the NEXT turn start
            # (_apply_pending_model_switch), where nothing is in flight.
            # The user gets to pick, keep typing, and send the next turn on
            # the new model without waiting for the swap or interrupting.
            if session.get("running"):
                parsed = parse_model_switch_args(value)
                try:
                    pending_model = parsed.model_input
                except Exception:
                    pending_model = str(value)
                pending_provider = (
                    getattr(parsed, "explicit_provider", "") or ""
                ).strip()
                confirmed = bool(params.get("confirm_expensive_model", False))
                # Run the selection guards HERE, not only at apply time.
                # This branch used to answer confirm_required=False without
                # consulting them, so a client that implements the confirm
                # round-trip was told no consent was needed. It stashed the
                # pick, and _apply_pending_model_switch -- which calls the
                # guards with the stashed (unconfirmed) flag -- dropped the
                # switch at the next turn start. The model reverted with no
                # confirm ever offered, because the one moment a round-trip
                # was possible had already passed.
                if not confirmed:
                    pending_warning = _pending_switch_selection_warning(
                        pending_model, pending_provider
                    )
                    if pending_warning is not None:
                        # Nothing is stashed: an unconfirmed guarded pick
                        # leaves the session exactly as it was, and the
                        # client re-sends with confirm_expensive_model to
                        # queue it for real.
                        return _ok(
                            rid,
                            {
                                "key": key,
                                "value": pending_model,
                                # `confirm_message` is the field to read.
                                # `warning` carries the same text only so
                                # clients written before the confirm
                                # round-trip existed still show something;
                                # `_apply_pending_model_switch` already
                                # prefers confirm_message and falls back to
                                # warning. Keep them identical or drop
                                # `warning` -- do not let them diverge.
                                "warning": pending_warning,
                                "confirm_required": True,
                                "confirm_message": pending_warning,
                                "scope": "session",
                                "deferred": False,
                            },
                        )
                session["pending_model_switch"] = {
                    "raw": value,
                    "confirm_expensive_model": confirmed,
                    # The resolved model/provider the next turn will run on.
                    # _session_info reports these while the switch is pending
                    # so the end-of-turn settle keeps showing the user's pick
                    # instead of blipping back to the still-live old model.
                    "display_model": pending_model,
                    "display_provider": pending_provider,
                }
                return _ok(
                    rid,
                    {
                        "key": key,
                        "value": pending_model,
                        "warning": "",
                        "confirm_required": False,
                        "confirm_message": "",
                        "scope": "session",
                        "deferred": True,
                    },
                )
            parsed_flags = parse_model_switch_args(value)
            explicit_provider = parsed_flags.explicit_provider
            failed_agent_init = (
                session.get("agent") is None
                and session.get("agent_error") is not None
            )
            failed_ready = session.get("agent_ready") if failed_agent_init else None
            if failed_agent_init:
                if failed_ready is None:
                    return _err(
                        rid,
                        5032,
                        session.get("agent_error")
                        or "agent initialization failed",
                    )
                if not failed_ready.wait(timeout=30.0):
                    return _err(rid, 5032, "agent initialization timed out")
            failed_agent_init = (
                failed_agent_init
                and session.get("agent") is None
                and session.get("agent_error") is not None
                and session.get("agent_ready") is failed_ready
                and failed_ready.is_set()
            )
            if (
                session.get("agent") is None
                and not explicit_provider.strip()
                and not failed_agent_init
            ):
                session_id = params.get("session_id", "")
                _start_agent_build(session_id, session)
                init_err = _wait_agent(session, rid)
                if init_err:
                    return init_err
                if session.get("agent") is None:
                    return _err(rid, 5032, "agent initialization failed")
            with _session_profile_runtime_scope(session):
                result = _apply_model_switch(
                    params.get("session_id", ""),
                    session,
                    value,
                    confirm_expensive_model=bool(
                        params.get("confirm_expensive_model", False)
                    ),
                    parsed_flags=parsed_flags,
                )
            if failed_agent_init and not result.get("confirm_required"):
                _restart_completed_failed_agent_build(
                    params.get("session_id", ""), session, failed_ready
                )
                init_err = _wait_agent(session, rid)
                if init_err:
                    return init_err
                if session.get("agent") is None:
                    return _err(rid, 5032, "agent initialization failed")
                with _session_profile_runtime_scope(session):
                    _persist_live_session_runtime(session)
        else:
            result = _apply_model_switch(
                "",
                {"agent": None},
                value,
                confirm_expensive_model=bool(
                    params.get("confirm_expensive_model", False)
                ),
            )
        return _ok(
            rid,
            {
                "key": key,
                "value": result["value"],
                "warning": result["warning"],
                "confirm_required": result.get("confirm_required", False),
                "confirm_message": result.get("confirm_message", ""),
                "scope": result.get("scope", "session"),
            },
        )
    except Exception as e:
        return _err(rid, 5001, str(e))


def _set_fast(rid, params, key, value, session):
    raw = str(value or "").strip().lower()
    agent = session.get("agent") if session else None
    if agent is not None:
        current_tier = getattr(agent, "service_tier", None)
    elif session is not None and session.get("create_service_tier_override") is not None:
        # Pre-build session with a pinned tier (desktop draft pick or an
        # earlier session-scoped toggle) — report/toggle from the pin, not
        # the global default.
        current_tier = session["create_service_tier_override"] or None
    else:
        current_tier = _load_service_tier()
    current_fast = current_tier == "priority"

    if raw in {"status"}:
        return _ok(
            rid,
            {"key": key, "value": {"priority": "fast", None: "normal"}.get(current_tier, current_tier)},
        )

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
            # A pre-build session may already have a picked model riding in
            # model_override (desktop draft) — validate fast support against
            # THAT model, not the global default it will never use.
            session_override = (session or {}).get("model_override") or {}
            target_model = (
                session_override.get("model")
                if isinstance(session_override, dict)
                else None
            ) or _resolve_model()
        if not target_model:
            return _err(
                rid,
                4002,
                "fast mode is not available without a selected model",
            )
        overrides = resolve_fast_mode_overrides(
            target_model,
            provider=getattr(agent, "provider", None),
            base_url=getattr(agent, "base_url", None),
        )
        if overrides is None:
            return _err(
                rid,
                4002,
                "fast mode is not available for this model",
            )

    if session is not None:
        # Session-scoped, like `reasoning` below (global persistence is
        # `--global` / Settings → Model territory). Writing config.yaml
        # here let every desktop model-menu selection (per-model fast
        # preset) rewrite the user's global agent.service_tier — flipping
        # fast mode for every OTHER session, profile, CLI, and gateway
        # build ("switch one session, switches everywhere"). Pin the
        # create override so lazily-built sessions and rebuilds (/new,
        # deferred resume) keep the choice; "" pins normal explicitly.
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
        _emit("session.info", params.get("session_id", ""), _session_info(agent, session))
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
    cur = (
        session.get("tool_progress_mode", _load_tool_progress_mode())
        if session
        else _load_tool_progress_mode()
    )
    if value and value != "cycle":
        nv = str(value).strip().lower()
        if nv not in cycle:
            return _err(rid, 4002, f"unknown verbose mode: {value}")
    else:
        try:
            idx = cycle.index(cur)
        except ValueError:
            idx = 2
        nv = cycle[(idx + 1) % len(cycle)]
    _write_config_key("display.tool_progress", nv)
    if session:
        session["tool_progress_mode"] = nv
        agent = session.get("agent")
        if agent is not None:
            agent.verbose_logging = nv == "verbose"
    return _ok(rid, {"key": key, "value": nv})


def _set_focus(rid, params, key, value, session):
    # Focus view — display-only reduced-output mode (/focus). Composes with
    # the tool_progress machinery rather than duplicating it: enabling it
    # pins tool_progress to "off" (the same value /verbose off uses) after
    # stashing the configured mode, and disabling it restores that mode.
    # Nothing about the request payload changes.
    from hermes_cli.focus_view import (
        FOCUS_TOOL_PROGRESS_MODE,
        normalize_tool_progress_mode,
        resolve_focus_arg,
    )

    d_f = _display_cfg(_load_cfg())
    cur_focus = bool(d_f.get("focus_view", False))
    action, target = resolve_focus_arg(str(value or ""), cur_focus)
    if action == "usage":
        return _err(rid, 4002, f"unknown focus value: {value} (use on|off|status)")
    if action == "status" or target is None:
        return _ok(
            rid,
            {
                "key": key,
                "value": "on" if cur_focus else "off",
                "tool_progress": _load_tool_progress_mode(),
            },
        )

    if target:
        saved = normalize_tool_progress_mode(
            (d_f.get("focus_saved_tool_progress") or _load_tool_progress_mode())
            if cur_focus
            else _load_tool_progress_mode()
        )
        _write_config_key("display.focus_saved_tool_progress", saved)
        _write_config_key("display.tool_progress", FOCUS_TOOL_PROGRESS_MODE)
        effective = FOCUS_TOOL_PROGRESS_MODE
    else:
        saved = normalize_tool_progress_mode(
            d_f.get("focus_saved_tool_progress") or "all"
        )
        _write_config_key("display.tool_progress", saved)
        effective = saved
    _write_config_key("display.focus_view", bool(target))

    if session:
        session["focus_view"] = bool(target)
        session["tool_progress_mode"] = effective
        agent_f = session.get("agent")
        if agent_f is not None:
            try:
                agent_f.tool_progress_mode = effective
            except Exception:
                pass
    return _ok(
        rid,
        {
            "key": key,
            "value": "on" if target else "off",
            "tool_progress": effective,
        },
    )


def _set_approval_mode(rid, params, key, value, session):
    raw = str(value or "").strip().lower()
    if raw not in _APPROVAL_MODES:
        return _err(
            rid,
            4002,
            f"unknown approval mode: {value}; pick one of manual|smart|off",
        )

    _write_config_key("approvals.mode", raw)
    _emit_all_session_info()
    return _ok(rid, {"key": "approvals.mode", "value": raw})


def _set_yolo(rid, params, key, value, session):
    # Approval bypass. Two scopes:
    #   scope="session" (default) — same as the TUI's Shift+Tab. Toggles
    #     ONLY this session's _session_yolo flag; never touches global
    #     config, so CLI / TUI / cron behavior is unaffected.
    #   scope="global" (Shift+click the zap) — flips the persistent global
    #     approvals.mode in config.yaml between "off" (bypass on) and
    #     "manual" (bypass off). This DOES affect every session, the CLI,
    #     the TUI, and cron, and survives restarts.
    scope = str(params.get("scope") or "session").strip().lower()
    try:
        from tools.approval import (
            disable_session_yolo,
            enable_session_yolo,
            is_session_yolo_enabled,
        )

        raw = str(value or "").strip().lower()

        def _resolve_toggle(current: bool) -> bool:
            if raw in {"1", "on", "true", "yes"}:
                return True
            if raw in {"0", "off", "false", "no"}:
                return False
            return not current

        if scope == "global":
            from tools.approval import _normalize_approval_mode

            cfg = _load_cfg()
            appr = cfg.get("approvals") if isinstance(cfg, dict) else None
            if not isinstance(appr, dict):
                appr = {}
            current = _normalize_approval_mode(appr.get("mode", "manual")) == "off"
            enable = _resolve_toggle(current)
            # Toggle between full bypass and the default manual gate. We do
            # not try to restore a prior "smart"/custom mode — the zap is a
            # binary on/off affordance; users with bespoke modes set them in
            # config.yaml.
            _write_config_key("approvals.mode", "off" if enable else "manual")
            nv = "1" if enable else "0"
            # Reflect the global flip in every live session's indicator.
            _emit_all_session_info()
            return _ok(rid, {"key": key, "value": nv, "scope": "global"})

        if session:
            current = is_session_yolo_enabled(session["session_key"])
            enable = _resolve_toggle(current)
            if enable:
                enable_session_yolo(session["session_key"])
                nv = "1"
            else:
                disable_session_yolo(session["session_key"])
                nv = "0"
            _emit_session_info(params.get("session_id", ""), session)
        else:
            current = is_truthy_value(os.environ.get("HERMES_YOLO_MODE"))
            enable = _resolve_toggle(current)
            if enable:
                os.environ["HERMES_YOLO_MODE"] = "1"
                nv = "1"
            else:
                os.environ.pop("HERMES_YOLO_MODE", None)
                nv = "0"
        return _ok(rid, {"key": key, "value": nv, "scope": "session"})
    except Exception as e:
        return _err(rid, 5001, str(e))


def _set_reasoning(rid, params, key, value, session):
    try:
        from hermes_constants import parse_reasoning_effort

        arg = str(value or "").strip().lower()
        scope = str(params.get("scope") or "").strip().lower()
        global_scope = scope == "global"
        if arg in {"show", "on"}:
            _write_display_sections(show_reasoning=True, thinking="expanded")
            if session:
                session["show_reasoning"] = True
            return _ok(rid, {"key": key, "value": "show"})
        if arg in {"hide", "off"}:
            _write_display_sections(show_reasoning=False, thinking="hidden")
            if session:
                session["show_reasoning"] = False
            return _ok(rid, {"key": key, "value": "hide"})

        # /reasoning full | clamp — parity with the classic CLI's reasoning_full
        # toggle. The TUI renders thinking as an expand/collapse section, so full
        # maps to sections.thinking=expanded and clamp to collapsed;
        # display.reasoning_full is persisted too so CLI and TUI stay consistent.
        if arg in {"full", "all"}:
            _write_display_sections(reasoning_full=True, thinking="expanded")
            return _ok(rid, {"key": key, "value": "full"})
        if arg in {"clamp", "collapse", "short"}:
            _write_display_sections(reasoning_full=False, thinking="collapsed")
            return _ok(rid, {"key": key, "value": "clamp"})

        parsed = parse_reasoning_effort(arg)
        if parsed is None:
            return _err(rid, 4002, f"unknown reasoning value: {value}")
        if global_scope or session is None:
            _write_config_key("agent.reasoning_effort", arg)
            if session is not None:
                session.pop("create_reasoning_override", None)
        else:
            # Session-scoped, like the messaging gateway's `/reasoning <level>`
            # (global persistence is `--global` / Settings → Model territory).
            # Writing config.yaml here let every desktop model-menu selection
            # rewrite the user's global agent.reasoning_effort to the preset default.
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
    cfg = _load_cfg_raw()  # write-back round-trip
    display = _display_cfg(cfg)
    sections = display.get("sections") if isinstance(display.get("sections"), dict) else {}
    display["details_mode"] = nv
    for section in _DETAIL_SECTION_NAMES:
        sections[section] = nv
    display["sections"] = sections
    cfg["display"] = display
    _save_cfg(cfg)
    return _ok(rid, {"key": key, "value": nv})


def _set_details_section(rid, params, key, value, session):
    # Per-section override: `details_mode.<section>` writes to
    # `display.sections.<section>`. Empty value clears the explicit override and
    # lets frontend resolution apply built-in section defaults before the global
    # details_mode.
    section = key.split(".", 1)[1]
    if section not in _DETAIL_SECTION_NAMES:
        return _err(rid, 4002, f"unknown section: {section}")

    cfg = _load_cfg_raw()  # write-back round-trip
    display = _display_cfg(cfg)
    sections_cfg = display.get("sections") if isinstance(display.get("sections"), dict) else {}

    nv = str(value or "").strip().lower()
    if not nv:
        sections_cfg.pop(section, None)
    elif nv not in _DETAIL_MODES:
        return _err(rid, 4002, f"unknown details_mode: {value}")
    else:
        sections_cfg[section] = nv
    display["sections"] = sections_cfg
    cfg["display"] = display
    _save_cfg(cfg)
    return _ok(rid, {"key": key, "value": nv})


def _set_thinking_mode(rid, params, key, value, session):
    nv = str(value or "").strip().lower()
    allowed_tm = frozenset({"collapsed", "truncated", "full"})
    if nv not in allowed_tm:
        return _err(rid, 4002, f"unknown thinking_mode: {value}")
    _write_config_key("display.thinking_mode", nv)
    # Backward compatibility bridge: keep details_mode aligned.
    _write_config_key(
        "display.details_mode", "expanded" if nv == "full" else "collapsed"
    )
    return _ok(rid, {"key": key, "value": nv})


def _set_density(rid, params, key, value, session):
    return _toggle_display_bool(rid, key, value, cfg_key="tui_compact", on_words={"on"}, off_words={"off"})


def _set_battery(rid, params, key, value, session):
    return _toggle_display_bool(
        rid, key, value, cfg_key="battery", on_words={"on", "true", "yes"}, off_words={"off", "false", "no"}
    )


def _set_theme(rid, params, key, value, session):
    # TUI light/dark mode pin: 'light'/'dark' beat background
    # auto-detection (xterm.js hosts misreport OSC 11); 'auto' trusts it.
    raw = str(value or "").strip().lower()
    if raw not in {"auto", "light", "dark"}:
        return _err(rid, 4002, f"unknown theme value: {value} (use auto|light|dark)")
    _write_config_key("display.tui_theme", raw)
    return _ok(rid, {"key": key, "value": raw})


def _set_statusbar(rid, params, key, value, session):
    raw = str(value or "").strip().lower()
    current = _coerce_statusbar(_display_cfg(_load_cfg()).get("tui_statusbar", "top"))

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
    # Explicit None check rather than `value or ""` so falsy non-string
    # inputs (0, False) reach the alias map as themselves — both map to
    # 'off' via _MOUSE_TRACKING_ALIASES — instead of being collapsed to
    # '' and triggering the toggle path. The slash command always passes
    # a string, but programmatic JSON-RPC callers may send booleans.
    raw = ("" if value is None else str(value)).strip().lower()
    current = _display_mouse_tracking(_display_cfg(_load_cfg()))

    if raw in {"", "toggle"}:
        nv = "all" if current == "off" else "off"
    elif raw in _MOUSE_TRACKING_ALIASES:
        nv = _MOUSE_TRACKING_ALIASES[raw]
    else:
        return _err(rid, 4002, f"unknown mouse value: {value}")

    _write_config_key("display.mouse_tracking", nv)
    return _ok(rid, {"key": key, "value": nv})


def _set_indicator(rid, params, key, value, session):
    # Use an explicit None check rather than `value or ""` so falsy
    # non-string inputs (0, False, []) still surface as themselves
    # in the error message instead of looking like a blank value.
    raw = ("" if value is None else str(value)).strip().lower()
    if raw not in INDICATOR_STYLES:
        return _err(
            rid,
            4002,
            f"unknown indicator: {raw!r}; pick one of {'|'.join(INDICATOR_STYLES)}",
        )
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
    return _ok(
        rid,
        {"key": "terminal.cwd", "value": cwd, "cwd": cwd, "branch": _git_branch_for_cwd(cwd)},
    )


def _set_prompt_like(rid, params, key, value, session):
    try:
        cfg = _load_cfg_raw()  # write-back round-trip ("prompt" saves cfg)
        if key == "prompt":
            if value == "clear":
                cfg.pop("custom_prompt", None)
                nv = ""
            else:
                cfg["custom_prompt"] = value
                nv = value
            _save_cfg(cfg)
        elif key == "personality":
            sid_key = params.get("session_id", "")
            pname, new_prompt = _validate_personality(str(value or ""), cfg)
            # Personality text is an in-session overlay. Persistence goes
            # through hermes_cli.personality (single owner) and never
            # touches the user-owned global system prompt.
            from hermes_cli.personality import persist_personality

            persist_personality(pname)
            nv = str(value or "none")
            history_reset, info = _apply_personality_to_session(
                sid_key, session, new_prompt, pname
            )
        else:
            _write_config_key(f"display.{key}", value)
            nv = value
            if key == "skin":
                # Every connected surface repaints, not just the RPC's
                # client; then sync the watcher baseline so the poll loop
                # doesn't re-broadcast the skin this RPC just applied.
                _broadcast_global_event("skin.changed", resolve_skin())
                _note_skin_broadcast()
        resp = {"key": key, "value": nv}
        if key == "personality":
            resp["history_reset"] = history_reset
            if info is not None:
                resp["info"] = info
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
    "model": _set_model,
    "fast": _set_fast,
    "busy": _set_busy,
    "verbose": _set_verbose,
    "focus": _set_focus,
    "approval_mode": _set_approval_mode,
    "approvals.mode": _set_approval_mode,
    "yolo": _set_yolo,
    "reasoning": _set_reasoning,
    "details_mode": _set_details_mode,
    "thinking_mode": _set_thinking_mode,
    "density": _set_density,
    "battery": _set_battery,
    "theme": _set_theme,
    "statusbar": _set_statusbar,
    "mouse": _set_mouse,
    "indicator": _set_indicator,
    "cwd": _set_cwd,
    "terminal.cwd": _set_cwd,
    "workdir": _set_cwd,
    "prompt": _set_prompt_like,
    "personality": _set_prompt_like,
    "skin": _set_prompt_like,
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
