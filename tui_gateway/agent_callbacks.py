"""Agent callback wiring: child-session live mirror, per-session agent callbacks, personality overlay, background/preview agent kwargs, agent reset.

Bodies are rebound onto server.py's globals at install time (see
method_ctx.bind_module), so they reference server.py globals bare.
"""

from __future__ import annotations

import threading

from .method_ctx import HandlerRegistry, bind_module

_registry = HandlerRegistry()


# Child-session live mirror: a delegated child's activity reaches the gateway only
# as relayed ``subagent.*`` events on the PARENT sid, so a window opened on the
# child's own session would sit silent until the run persists. Translate them into
# the native stream events emitted on the CHILD sid (write_json routes by sid).
_child_mirrors: dict[str, dict] = {}
_child_mirrors_lock = threading.Lock()
# Child session ids with a run in flight (refreshed per relayed event, popped on
# complete) so a lazy watch resume reports running=true during a silent long tool.
_active_child_runs: dict[str, float] = {}
# Anything quiet this long lost its completion event (callback raised, parent
# crashed) — don't pin "running".
_CHILD_RUN_STALE_S = 3600.0


def _child_run_active(child_key: str) -> bool:
    ts = _active_child_runs.get(child_key)
    return ts is not None and (time.time() - ts) < _CHILD_RUN_STALE_S


def _mirror_subagent_to_child(event_type: str, payload: dict) -> None:
    child_key = str(payload.get("child_session_id") or "")
    if not child_key:
        return
    # Liveness registry first: accurate with no window open, so one opened mid-run
    # immediately knows the child is busy.
    if event_type == "subagent.complete":
        _active_child_runs.pop(child_key, None)
    else:
        _active_child_runs[child_key] = time.time()
    # Mirror only into a live watch session NOT upgraded to a full agent: an
    # upgraded one owns a real native stream and mirroring would interleave two
    # turns on one sid. Either way drop state so a reopened window starts fresh.
    live = _find_live_session_by_key(child_key)
    if live is None or live[1].get("agent") is not None:
        with _child_mirrors_lock:
            _child_mirrors.pop(child_key, None)
        return
    csid = live[0]
    text = str(payload.get("text") or "")
    with _child_mirrors_lock:
        st = _child_mirrors.setdefault(child_key, {"seq": 0, "open_tool": None, "started": False})
        if not st["started"]:
            st["started"] = True
            _emit("message.start", csid)
        if event_type == "subagent.thinking":
            if text:
                _emit("reasoning.delta", csid, {"text": text})
        elif event_type == "subagent.text":
            if text:
                _emit("message.delta", csid, {"text": text})
        elif event_type == "subagent.start":
            # One-time header (the child's goal) so a fresh window has context first.
            if text:
                _emit("message.delta", csid, {"text": f"{text}\n"})
        elif event_type == "subagent.tool":
            if st["open_tool"]:
                _emit("tool.complete", csid, st["open_tool"])
            st["seq"] += 1
            tool = {
                "name": str(payload.get("tool_name") or "tool"),
                "tool_id": f"submirror:{child_key}:{st['seq']}",
                "args": {},
            }
            if preview := str(payload.get("tool_preview") or payload.get("text") or ""):
                tool["preview"] = preview
            st["open_tool"] = tool
            _emit("tool.start", csid, tool)
        elif event_type == "subagent.complete":
            if st["open_tool"]:
                _emit("tool.complete", csid, st["open_tool"])
            summary = str(payload.get("summary") or payload.get("text") or "")
            _emit("message.complete", csid, {"text": summary})
            _child_mirrors.pop(child_key, None)


def _agent_cbs(sid: str) -> dict:
    def _read_block(event: str, timeout: int):
        # read_terminal / read_preview (desktop GUI): blocking bridge like clarify; the
        # preview read gets longer since a URL tab extracts text from a live page.
        return lambda start=None, count=None: _block(
            event, sid, {k: v for k, v in (("start", start), ("count", count)) if v is not None}, timeout=timeout
        )

    callbacks = {
        "tool_start_callback": lambda tc_id, name, args: _on_tool_start(sid, tc_id, name, args),
        "tool_complete_callback": lambda tc_id, name, args, result: _on_tool_complete(sid, tc_id, name, args, result),
        "tool_progress_callback": lambda event_type, name=None, preview=None, args=None, **kwargs: _on_tool_progress(
            sid, event_type, name, preview, args, **kwargs
        ),
        "tool_gen_callback": lambda name: _tool_progress_enabled(sid) and _emit("tool.generating", sid, {"name": name}),
        "thinking_callback": lambda text: _emit("thinking.delta", sid, {"text": text}),
        # Affection reaction (ily / <3 / good bot) → hearts; core-detected so TUI and desktop share it.
        "reaction_callback": lambda kind: _emit("reaction", sid, {"kind": kind}),
        "reasoning_callback": lambda text: _emit(
            "reasoning.delta", sid, {"text": text, **({"verbose": True} if _session_verbose(sid) else {})}
        ),
        "status_callback": lambda kind, text=None: _status_update(sid, str(kind), None if text is None else str(text)),
        # Credits/notice spine: AgentNotice → notification.show; recovery clear → notification.clear.
        "notice_callback": lambda n: _emit(
            "notification.show",
            sid,
            {"text": n.text, "level": n.level, "kind": n.kind, "ttl_ms": n.ttl_ms, "key": n.key, "id": n.id},
        ),
        "notice_clear_callback": lambda key: _emit("notification.clear", sid, {"key": key}),
        "clarify_callback": lambda q, c, multi_select=False, questions=None: (
            _clarify_block(sid, q, c, multi_select=multi_select, questions=questions)
        ),
        "read_terminal_callback": _read_block("terminal.read.request", 30),
        "read_preview_callback": _read_block("preview.read.request", 45),
        # drive_preview / annotate_preview (desktop GUI): renderer drives the preview webview and
        # answers with outcome + refreshed element inventory; same budget as the preview read it ends with.
        "drive_preview_callback": lambda payload: _block("preview.act.request", sid, dict(payload), timeout=45),
        # read_window_below (desktop GUI): main process enumerates native windows.
        "read_window_below_callback": lambda: _block("window.read.request", sid, {}, timeout=30),
        # setup_mcp (desktop GUI): consent card + install/enable/OAuth. Long timeout on purpose (typing
        # an API key, browser OAuth); like clarify, timeout returns "unanswered" and a late answer is tolerated.
        "setup_mcp_callback": lambda server, action, reason: _block(
            "mcp.setup.request", sid, {"server": server, "action": action, "reason": reason}, timeout=600
        ),
        # tour (desktop GUI): renderer drives driver.js and answers tour.respond.
        "tour_callback": lambda payload: _tour_request(sid, payload),
    }

    # Interim assistant commentary (text alongside tool calls). Gated on
    # display.interim_assistant_messages (default true); _run_prompt_submit overwrites
    # it per turn and clears it in its finally so a stale closure can't fire.
    if _load_interim_assistant_messages():
        callbacks["interim_assistant_callback"] = lambda text, *, already_streamed=False: _emit(
            "message.interim", sid, {"text": str(text), "already_streamed": bool(already_streamed)}
        )

    return callbacks


def _apply_project_workspace(task_id: str, path: str, _name: str = "") -> None:
    """Intentional workspace move from the project_* tools: re-anchor the live
    session's cwd and push session.info so the desktop follows. This is the ONLY
    auto-cwd path — driven by an explicit tool call, never a terminal `cd`."""
    if not path:
        return
    # task_id is the durable session_key; _sessions (and desktop event routing) key by sid.
    key = str(task_id or "")
    sid, session = "", None
    with _sessions_lock:
        if key in _sessions:
            sid, session = key, _sessions[key]
        else:
            for cand_sid, cand in _sessions.items():
                if cand.get("session_key") == key or getattr(cand.get("agent"), "session_id", None) == key:
                    sid, session = cand_sid, cand
                    break
    if session is None:
        return
    resolved = os.path.abspath(os.path.expanduser(str(path)))
    if not os.path.isdir(resolved):
        return
    session["cwd"] = resolved
    session["explicit_cwd"] = True
    session["cwd_from_settle"] = False  # explicit switch supersedes a settle-adopted cwd
    _register_session_cwd(session)
    _persist_session_cwd_and_schedule_git_meta(session, resolved)
    try:
        agent = session.get("agent")
        if agent is not None:
            info = _session_info(agent, session)
        else:
            info = {
                "cwd": resolved, "branch": _git_branch_for_cwd(resolved),
                "project": _project_info_for_cwd(resolved), "lazy": True,
            }
        _emit("session.info", sid, info)
    except Exception:
        logger.debug("failed to emit session.info after project workspace move", exc_info=True)


def _wire_callbacks(sid: str):
    from tools.terminal_tool import set_sudo_password_callback
    from tools.skills_tool import set_secret_capture_callback
    from tools.project_tools import set_project_workspace_callback

    set_sudo_password_callback(lambda: _block("sudo.request", sid, {}, timeout=120))
    set_project_workspace_callback(_apply_project_workspace)

    def secret_cb(env_var, prompt, metadata=None):
        pl = {"prompt": prompt, "env_var": env_var}
        if metadata:
            pl["metadata"] = metadata
        val = _block("secret.request", sid, pl)
        if not val:
            return {"success": True, "stored_as": env_var, "validated": False, "skipped": True, "message": "skipped"}
        from hermes_cli.config import save_env_value_secure

        return {**save_env_value_secure(env_var, val), "skipped": False, "message": "ok"}

    set_secret_capture_callback(secret_cb)


def _render_personality_prompt(value) -> str:
    """Delegates to hermes_cli.personality (single owner of rendering)."""
    from hermes_cli.personality import render_personality_prompt

    return render_personality_prompt(value)


def _available_personalities(cfg: dict | None = None) -> dict:
    """Built-ins + user overrides, via hermes_cli.personality (single owner)."""
    from hermes_cli.personality import available_personalities

    return available_personalities(_load_cfg() if cfg is None else cfg)


def _validate_personality(value: str, cfg: dict | None = None) -> tuple[str, str]:
    """Resolve a requested personality to (name, prompt) or raise ValueError.

    Same contract as hermes_cli.personality.resolve_personality, but goes through
    the module-level _available_personalities so tests keep a single patch point.
    """
    from hermes_cli.personality import normalize_personality_name

    name = normalize_personality_name(value)
    if not name:
        return "", ""
    personalities = _available_personalities(cfg)
    if name not in personalities:
        names = ", ".join(f"`{n}`" for n in sorted(personalities))
        raise ValueError(f"Unknown personality: `{str(value).strip()}`.\n\nAvailable: `none`, {names}")
    return name, _render_personality_prompt(personalities[name])


def _prompt_text(value) -> str:
    """Normalize config prompt values from YAML for AIAgent (hermes_cli.personality owns this)."""
    from hermes_cli.personality import prompt_text

    return prompt_text(value)


def _apply_personality_to_session(
    sid: str, session: dict, new_prompt: str, personality: str = ""
) -> tuple[bool, dict | None]:
    """Apply a personality change to a live session without resetting history.

    Updates the ephemeral system prompt in place (appended at API-call time, so
    prompt-cache hits survive) and injects a pivot marker so the model stops
    pattern-matching its earlier tone. Returns (history_reset=False, info).
    """
    if not session:
        return False, None
    session["personality"] = personality

    agent = session.get("agent")
    if not agent:
        return False, None
    agent.ephemeral_system_prompt = new_prompt or None
    if new_prompt:
        marker = (
            "[System: The user has changed the assistant's personality. "
            "From this point forward, adopt the following persona and respond "
            f"accordingly: {new_prompt}]"
        )
    else:
        marker = (
            "[System: The user has cleared the personality overlay. "
            "From this point forward, respond in your normal default style.]"
        )
    # Like the model-switch marker: role=user so strict providers accept it
    # mid-conversation, but `display_kind` keeps it out of the
    # `truncate_before_user_ordinal` addressing space (untagged, every rewind would
    # land one turn early and `replace_messages` hard-delete the difference).
    with session["history_lock"]:
        session["history"].append({"role": "user", "content": marker, "display_kind": "personality_switch"})
        session["history_version"] = int(session.get("history_version", 0)) + 1
    info = _session_info(agent)
    _emit("session.info", sid, info)
    return False, info


def _cfg_max_turns(cfg: dict, default: int) -> int:
    from hermes_cli.config import resolve_turn_limit as _resolve_turn_limit
    # Env override wins; resolve_turn_limit makes "none"/"unlimited"/0 first-class spellings.
    env_val = os.environ.get("HERMES_TUI_MAX_TURNS")
    if env_val:
        return _resolve_turn_limit(env_val, default=default)
    raw = (cfg.get("agent") or {}).get("max_turns")
    if raw is None:
        raw = cfg.get("max_turns")
    if raw is not None:
        return _resolve_turn_limit(raw, default=default)
    return default


def _parse_tui_skills_env() -> list[str]:
    raw = os.environ.get("HERMES_TUI_SKILLS", "")
    skills: list[str] = []
    for part in raw.replace("\n", ",").split(","):
        item = part.strip()
        if item and item not in skills:
            skills.append(item)
    return skills


def _load_fallback_model():
    """Configured fallback chain for TUI-created agents, via the shared
    ``get_fallback_chain`` (parity with HermesCLI/gateway: ``fallback_providers``
    first in order, legacy ``fallback_model`` merged after, deduped)."""
    from hermes_cli.fallback_config import get_fallback_chain

    return get_fallback_chain(_load_cfg())


def _agent_fallback_model(agent):
    """Return an agent's fallback chain without rehydrating deliberately empty chains."""
    if hasattr(agent, "_fallback_chain"):
        return agent._fallback_chain or []
    if hasattr(agent, "_fallback_model"):
        return agent._fallback_model
    return _load_fallback_model()


def _background_agent_kwargs(agent, task_id: str) -> dict:
    cfg = _load_cfg()

    return {
        "base_url": getattr(agent, "base_url", None) or None,
        "api_key": getattr(agent, "api_key", None) or None,
        "provider": getattr(agent, "provider", None) or None,
        "api_mode": getattr(agent, "api_mode", None) or None,
        "acp_command": getattr(agent, "acp_command", None) or None,
        "acp_args": getattr(agent, "acp_args", None) or None,
        "model": getattr(agent, "model", None) or _resolve_model(),
        "max_iterations": _cfg_max_turns(cfg, 25),
        # Detached tasks declare platform="tui" (no UI sid for renderer-routed
        # events), so resolve toolsets against it — never GUI schema they can't use.
        "enabled_toolsets": getattr(agent, "enabled_toolsets", None) or _load_enabled_toolsets("tui"),
        "quiet_mode": True,
        "verbose_logging": False,
        "ephemeral_system_prompt": getattr(agent, "ephemeral_system_prompt", None) or None,
        "providers_allowed": getattr(agent, "providers_allowed", None),
        "providers_ignored": getattr(agent, "providers_ignored", None),
        "providers_order": getattr(agent, "providers_order", None),
        "provider_sort": getattr(agent, "provider_sort", None),
        "provider_require_parameters": getattr(agent, "provider_require_parameters", False),
        "provider_data_collection": getattr(agent, "provider_data_collection", None),
        "openrouter_min_coding_score": getattr(agent, "openrouter_min_coding_score", None),
        "session_id": task_id,
        "reasoning_config": getattr(agent, "reasoning_config", None)
        or _load_reasoning_config(str(getattr(agent, "model", "") or "")),
        "service_tier": getattr(agent, "service_tier", None) or _load_service_tier(),
        "request_overrides": dict(getattr(agent, "request_overrides", {}) or {}),
        "platform": "tui",
        "session_db": _get_db(),
        "fallback_model": _agent_fallback_model(agent),
    }


def _ephemeral_preview_agent_kwargs(agent, task_id: str) -> dict:
    kwargs = _background_agent_kwargs(agent, task_id)
    kwargs.update({"enabled_toolsets": ["terminal", "file"], "session_db": None, "skip_memory": True})
    return kwargs


def _preview_restart_history(session: dict, max_messages: int = 24, max_tool_chars: int = 1200) -> list[dict]:
    """Distill recent parent history for the ephemeral preview-restart agent, which
    otherwise would guess app/server/cwd/port from the bare URL + console logs.
    Keeps the last ``max_messages`` (always back to the last user turn); tool
    results are truncated so file dumps don't blow the context window."""
    try:
        with session["history_lock"]:
            history = list(session.get("history", []) or [])
    except Exception:
        history = list(session.get("history", []) or [])

    if not history:
        return []
    start = max(0, len(history) - max_messages)
    for idx in range(len(history) - 1, -1, -1):
        if history[idx].get("role") == "user":
            start = min(start, idx)
            break

    trimmed: list[dict] = []
    for msg in history[start:]:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role not in ("user", "assistant", "tool", "system"):
            continue
        copy = {k: v for k, v in msg.items() if k != "reasoning"}
        if role == "tool":
            content = copy.get("content")
            if isinstance(content, str) and len(content) > max_tool_chars:
                copy["content"] = content[:max_tool_chars] + f"\n... (truncated, original {len(content)} chars)"
        trimmed.append(copy)
    return trimmed


def _preview_tool_result_preview(name: str, result: str) -> str:
    try:
        data = json.loads(result)
    except Exception:
        return ""
    if not isinstance(data, dict):
        return ""
    if name == "terminal":
        output = str(data.get("output") or "").strip()
        if output:
            return output[-1200:]
        if data.get("session_id"):
            return f"Background process started: {data.get('session_id')}"
        if data.get("exit_code") is not None:
            return f"terminal exited with code {data.get('exit_code')}"
    return str(data.get("error") or "").strip()[:1200]


def _preview_restart_callbacks(parent: str, task_id: str) -> dict:
    started_at: dict[str, float] = {}

    def progress(message: str, level: str = "info") -> None:
        text = str(message or "").strip()
        if text:
            _emit("preview.restart.progress", parent, {"task_id": task_id, "level": level, "text": text})

    def tool_start(tool_call_id: str, name: str, args: dict) -> None:
        started_at[tool_call_id] = time.time()
        ctx = _tool_ctx(name, args)
        progress(f"Running {name}{f': {ctx}' if ctx else ''}")

    def tool_complete(tool_call_id: str, name: str, _args: dict, result: str) -> None:
        duration_s = time.time() - started_at.get(tool_call_id, time.time())
        summary = _tool_summary(name, result, duration_s) or f"Finished {name}{f' in {_fmt_tool_duration(duration_s)}' if duration_s else ''}"
        output = _preview_tool_result_preview(name, result)
        progress(summary + (f"\n{output}" if output else ""))

    def tool_progress(event_type: str, name: str | None = None, preview: str | None = None, **_kwargs) -> None:
        if preview:
            progress(str(preview))
        elif name:
            progress(f"{event_type.replace('.', ' ')}: {name}")

    return {
        "tool_start_callback": tool_start,
        "tool_complete_callback": tool_complete,
        "tool_progress_callback": tool_progress,
        "tool_gen_callback": lambda name: progress(f"Preparing {name}"),
        "status_callback": lambda kind, text=None: progress(text if text is not None else kind),
    }


def _reset_session_agent(sid: str, session: dict) -> dict:
    tokens = _set_session_context(session["session_key"])
    try:
        # /new is a full conversation boundary: session-scoped runtime overrides
        # (/model, /reasoning, /fast) do NOT carry forward — the fresh agent
        # re-derives them from config.yaml, and the pins are cleared so a rebuild
        # can't resurrect them. Global process state is never touched (see the
        # cross-session-contamination note in _apply_model_switch).
        for k in ("model_override", "create_reasoning_override", "create_service_tier_override", "one_turn_model_restore"):
            session.pop(k, None)
        new_agent = _make_agent(
            sid,
            session["session_key"],
            session_id=session["session_key"],
            platform_override=_session_source(session),
            context_cwd_is_launch_artifact=_context_cwd_is_launch_artifact(session),
        )
    finally:
        _clear_session_context(tokens)
    session.update(
        agent=new_agent, config_model_seen=_config_model_target(), attached_images=[], queued_prompt=None
    )
    session.pop("queued_prompts", None)
    session["_queued_prompt_generation"] = int(session.get("_queued_prompt_generation", 0)) + 1
    session.update(
        edit_snapshots={}, image_counter=0, running=False, show_reasoning=_load_show_reasoning(),
        tool_progress_mode=_load_tool_progress_mode(), tool_started_at={},
    )
    with session["history_lock"]:
        session["history"] = []
        session["history_version"] = int(session.get("history_version", 0)) + 1
    info = _session_info(new_agent, session)
    _emit("session.info", sid, info)
    _restart_slash_worker(sid, session)
    return info


def register(server) -> None:
    """Publish this module's helpers + handlers onto ``server``, rebound to its globals."""
    bind_module(globals(), server, skip=("_",))
