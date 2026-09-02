"""Agent callback wiring: child-session live mirror, per-session agent callbacks, personality overlay, background/preview agent kwargs, agent reset.

Bodies are rebound onto server.py's globals at install time (see
method_ctx.bind_module), so they reference server.py globals bare.
"""

from __future__ import annotations

import threading

from .method_ctx import HandlerRegistry, bind_module

_registry = HandlerRegistry()


# ── Child-session live mirror ────────────────────────────────────────
# A delegated child is not a live gateway session — it runs synchronously
# inside the parent's turn, and its activity reaches the gateway only as
# relayed ``subagent.*`` events on the PARENT sid. When a UI opens the child's
# own session (session.resume on ``child_session_id``, e.g. the desktop's
# open-in-new-window), that window would otherwise sit silent until the run
# persists. Translate the relayed events into the native stream events the
# window already renders — emitted on the CHILD sid, routed to its transport
# by write_json — so the window shows a real midstream turn.
_child_mirrors: dict[str, dict] = {}
_child_mirrors_lock = threading.Lock()
# Stored child session ids with a delegation run currently in flight (refreshed
# on every relayed subagent.* event, popped on subagent.complete). Lets a lazy
# watch resume report running=true so the window shows a busy indicator even
# while the child is silent inside a long tool call (no events for 25s+).
_active_child_runs: dict[str, float] = {}
# Staleness bound for the registry: entries refresh on every relayed event, so
# anything this quiet means the completion event was lost (callback raised,
# parent crashed) — don't let a leaked entry pin "running" forever.
_CHILD_RUN_STALE_S = 3600.0


def _child_run_active(child_key: str) -> bool:
    ts = _active_child_runs.get(child_key)
    return ts is not None and (time.time() - ts) < _CHILD_RUN_STALE_S


def _mirror_subagent_to_child(event_type: str, payload: dict) -> None:
    child_key = str(payload.get("child_session_id") or "")
    if not child_key:
        return
    # Liveness registry first — it must be accurate even when no window is
    # open, so a window opened mid-run can immediately know the child is busy.
    if event_type == "subagent.complete":
        _active_child_runs.pop(child_key, None)
    else:
        _active_child_runs[child_key] = time.time()
    # Mirror only into a live watch session (keyed by session_key; its live sid
    # differs from the stored id) that has NOT been upgraded to a full agent.
    # No window / closed → nothing to mirror; an upgraded session owns a real
    # native stream and mirroring on top would interleave two turns on one sid.
    # Either way drop state so a reopened window starts a fresh synthetic turn.
    live = _find_live_session_by_key(child_key)
    if live is None or live[1].get("agent") is not None:
        with _child_mirrors_lock:
            _child_mirrors.pop(child_key, None)
        return
    csid = live[0]
    with _child_mirrors_lock:
        st = _child_mirrors.setdefault(child_key, {"seq": 0, "open_tool": None, "started": False})
        if not st["started"]:
            st["started"] = True
            _emit("message.start", csid)
        if event_type == "subagent.thinking":
            if text := str(payload.get("text") or ""):
                _emit("reasoning.delta", csid, {"text": text})
        elif event_type == "subagent.text":
            # The child's streamed reply text — the actual "agent talking".
            # Relayed token-by-token from the child's run_conversation
            # stream_callback, so the watch window streams the reply live.
            if text := str(payload.get("text") or ""):
                _emit("message.delta", csid, {"text": text})
        elif event_type == "subagent.start":
            # One-time header line (the child's goal) so a freshly opened window
            # shows immediate context before the first reply token streams.
            if text := str(payload.get("text") or ""):
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
    callbacks = {
        "tool_start_callback": lambda tc_id, name, args: _on_tool_start(
            sid, tc_id, name, args
        ),
        "tool_complete_callback": lambda tc_id, name, args, result: _on_tool_complete(
            sid, tc_id, name, args, result
        ),
        "tool_progress_callback": lambda event_type, name=None, preview=None, args=None, **kwargs: _on_tool_progress(
            sid, event_type, name, preview, args, **kwargs
        ),
        "tool_gen_callback": lambda name: _tool_progress_enabled(sid)
        and _emit("tool.generating", sid, {"name": name}),
        "thinking_callback": lambda text: _emit("thinking.delta", sid, {"text": text}),
        # Affection reaction (ily / <3 / good bot) → hearts. Core-detected, so
        # the TUI heart and desktop floating hearts share one signal.
        "reaction_callback": lambda kind: _emit("reaction", sid, {"kind": kind}),
        "reasoning_callback": lambda text: _emit(
            "reasoning.delta",
            sid,
            {"text": text, **({"verbose": True} if _session_verbose(sid) else {})},
        ),
        "status_callback": lambda kind, text=None: _status_update(
            sid, str(kind), None if text is None else str(text)
        ),
        # Credits/notice spine (L1): an AgentNotice fired by the agent becomes a
        # notification.show WS event; a recovery clear becomes notification.clear.
        # Snake_case payload to match the existing gateway-event convention.
        "notice_callback": lambda n: _emit(
            "notification.show",
            sid,
            {
                "text": n.text,
                "level": n.level,
                "kind": n.kind,
                "ttl_ms": n.ttl_ms,
                "key": n.key,
                "id": n.id,
            },
        ),
        "notice_clear_callback": lambda key: _emit(
            "notification.clear", sid, {"key": key}
        ),
        "clarify_callback": lambda q, c, multi_select=False, questions=None: (
            _clarify_block(sid, q, c, multi_select=multi_select, questions=questions)
        ),
        # read_terminal tool (desktop GUI): same blocking bridge as clarify — the
        # renderer answers terminal.read.respond with the serialized buffer.
        "read_terminal_callback": lambda start=None, count=None: _block(
            "terminal.read.request",
            sid,
            {k: v for k, v in (("start", start), ("count", count)) if v is not None},
            timeout=30,
        ),
        # read_preview tool (desktop GUI): the renderer serializes the active
        # preview tab (a Browser webview's readable text, a file's identity)
        # and answers preview.read.respond. Longer timeout than the terminal
        # read — a URL tab extracts text from a live page.
        "read_preview_callback": lambda start=None, count=None: _block(
            "preview.read.request",
            sid,
            {k: v for k, v in (("start", start), ("count", count)) if v is not None},
            timeout=45,
        ),
        # drive_preview tool (desktop GUI): the renderer injects the interaction
        # engine into the preview pane's webview (or drives the pane's history)
        # and answers preview.act.respond with the outcome plus a refreshed
        # element inventory. Same budget as the preview read, which it ends
        # with — a click on a slow page pays for the settle and the re-scan.
        # annotate_preview rides this same callback: it resolves a target
        # through the same engine and differs only in the verb it sends, so it
        # needs a tool of its own but not a channel of its own.
        "drive_preview_callback": lambda payload: _block(
            "preview.act.request",
            sid,
            dict(payload),
            timeout=45,
        ),
        # read_window_below tool (desktop GUI): the renderer asks its main
        # process (which owns native window enumeration) which OS window sits
        # directly underneath the Hermes window, and answers
        # window.read.respond with the serialized metadata.
        "read_window_below_callback": lambda: _block(
            "window.read.request",
            sid,
            {},
            timeout=30,
        ),
        # setup_mcp tool (desktop GUI): the renderer shows an inline consent
        # card and walks the user through install/enable/OAuth via the REST
        # endpoints, then answers mcp.setup.respond with the JSON outcome.
        # Long timeout on purpose — the flow can include typing an API key or
        # a browser OAuth round-trip. Same lifecycle as clarify: on timeout
        # the tool returns "unanswered" and a late answer is tolerated.
        "setup_mcp_callback": lambda server, action, reason: _block(
            "mcp.setup.request",
            sid,
            {"server": server, "action": action, "reason": reason},
            timeout=600,
        ),
        # tour tool (desktop GUI): the renderer drives driver.js — highlighting
        # elements in the app's own DOM or injecting the engine into the
        # preview pane's webview — and answers tour.respond with the outcome
        # (did the selector match, which step is active).
        "tour_callback": lambda payload: _tour_request(sid, payload),
    }

    # Interim assistant commentary (text alongside tool calls, or the attempted
    # final answer before a verify-on-stop nudge). Gated on
    # display.interim_assistant_messages (default true). Also set per-turn in
    # _run_prompt_submit as defense-in-depth — the per-turn set overwrites
    # this, and the finally block clears it so a stale closure can't fire.
    if _load_interim_assistant_messages():
        callbacks["interim_assistant_callback"] = (
            lambda text, *, already_streamed=False: _emit(
                "message.interim",
                sid,
                {"text": str(text), "already_streamed": bool(already_streamed)},
            )
        )

    return callbacks


def _apply_project_workspace(task_id: str, path: str, _name: str = "") -> None:
    """Intentional workspace move from the project_* tools: re-anchor the live
    session's cwd to the chosen project's folder and push session.info so the
    desktop follows (refresh tree + scope into the project). This is the ONLY
    auto-cwd path — driven by an explicit tool call, never a terminal `cd`."""
    if not path:
        return

    # The tool's task_id is the durable session_key, but _sessions is keyed by a
    # short sid uuid (and the desktop routes events by that sid). Resolve it.
    key = str(task_id or "")
    sid = ""
    session = None
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
    # An explicit project switch supersedes any earlier settle-adopted cwd.
    session["cwd_from_settle"] = False
    _register_session_cwd(session)

    _persist_session_cwd_and_schedule_git_meta(session, resolved)

    try:
        agent = session.get("agent")
        info = (
            _session_info(agent, session)
            if agent is not None
            else {
                "cwd": resolved,
                "branch": _git_branch_for_cwd(resolved),
                "project": _project_info_for_cwd(resolved),
                "lazy": True,
            }
        )
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
            return {
                "success": True,
                "stored_as": env_var,
                "validated": False,
                "skipped": True,
                "message": "skipped",
            }
        from hermes_cli.config import save_env_value_secure

        return {
            **save_env_value_secure(env_var, val),
            "skipped": False,
            "message": "ok",
        }

    set_secret_capture_callback(secret_cb)


def _render_personality_prompt(value) -> str:
    """Delegates to hermes_cli.personality (single owner of rendering)."""
    from hermes_cli.personality import render_personality_prompt

    return render_personality_prompt(value)


def _available_personalities(cfg: dict | None = None) -> dict:
    """Built-ins + user overrides, via hermes_cli.personality (single owner)."""
    from hermes_cli.personality import available_personalities

    if cfg is None:
        cfg = _load_cfg()
    return available_personalities(cfg)


def _validate_personality(value: str, cfg: dict | None = None) -> tuple[str, str]:
    """Resolve a requested personality against _available_personalities.

    Same contract as hermes_cli.personality.resolve_personality — (name,
    prompt) or ValueError — but resolves through the module-level
    _available_personalities so tests (and future gateway-side overrides)
    keep a single patch point.
    """
    from hermes_cli.personality import normalize_personality_name

    name = normalize_personality_name(value)
    if not name:
        return "", ""
    personalities = _available_personalities(cfg)
    if name not in personalities:
        names = ", ".join(f"`{n}`" for n in sorted(personalities))
        raise ValueError(
            f"Unknown personality: `{str(value).strip()}`.\n\nAvailable: `none`, {names}"
        )
    return name, _render_personality_prompt(personalities[name])


def _prompt_text(value) -> str:
    """Normalize config prompt values from YAML before handing them to AIAgent.

    Delegates to hermes_cli.personality (single owner).
    """
    from hermes_cli.personality import prompt_text

    return prompt_text(value)


def _apply_personality_to_session(
    sid: str, session: dict, new_prompt: str, personality: str = ""
) -> tuple[bool, dict | None]:
    """Apply a personality change to an existing session without resetting history.

    Updates the agent's ephemeral system prompt in-place so the new personality
    takes effect on the next turn.  The cached base system prompt is left intact
    (ephemeral_system_prompt is appended at API-call time, not baked into the
    cache), which preserves prompt-cache hits.

    Also injects a system-role marker into the conversation history so the model
    knows to pivot its style from this point forward (without this, LLMs tend to
    continue the tone established by earlier messages in the transcript).

    Returns (history_reset, info) — history_reset is always False since we
    preserve the conversation.
    """
    if not session:
        return False, None
    session["personality"] = personality

    agent = session.get("agent")
    if agent:
        agent.ephemeral_system_prompt = new_prompt or None
        # Inject a pivot marker into history so the model sees the change point.
        # This prevents it from pattern-matching its prior style.
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
        # Tagged like the model-switch marker (`_append_model_switch_marker`):
        # the marker rides as role=user so strict OpenAI-compatible providers
        # accept it mid-conversation, but `display_kind` keeps it out of the
        # `truncate_before_user_ordinal` addressing space. Untagged, it counts
        # as a real user turn on the gateway side while no client counts it, so
        # every later rewind resolves one turn too early and `replace_messages`
        # hard-deletes the difference (#82756).
        with session["history_lock"]:
            session["history"].append(
                {"role": "user", "content": marker, "display_kind": "personality_switch"}
            )
            session["history_version"] = int(session.get("history_version", 0)) + 1
        info = _session_info(agent)
        _emit("session.info", sid, info)
        return False, info
    return False, None


def _cfg_max_turns(cfg: dict, default: int) -> int:
    from hermes_cli.config import resolve_turn_limit as _resolve_turn_limit
    # Env var override (highest priority)
    env_val = os.environ.get("HERMES_TUI_MAX_TURNS")
    if env_val:
        return _resolve_turn_limit(env_val, default=default)
    # Config file value — route through resolve_turn_limit so that
    # "none"/"unlimited"/0 are first-class spellings, not int() crashes.
    agent_cfg = cfg.get("agent") or {}
    raw = agent_cfg.get("max_turns")
    if raw is None:
        raw = cfg.get("max_turns")
    if raw is not None:
        return _resolve_turn_limit(raw, default=default)
    return default


def _parse_tui_skills_env() -> list[str]:
    raw = os.environ.get("HERMES_TUI_SKILLS", "")
    skills: list[str] = []
    seen: set[str] = set()
    for part in raw.replace("\n", ",").split(","):
        item = part.strip()
        if item and item not in seen:
            seen.add(item)
            skills.append(item)
    return skills


def _load_fallback_model():
    """Return the configured fallback chain for TUI-created agents.

    Delegates to the shared ``get_fallback_chain`` helper so the TUI path
    stays in parity with ``HermesCLI.__init__`` and ``gateway/run.py``:
    ``fallback_providers`` is the primary source of truth and keeps its
    order, with legacy ``fallback_model`` entries merged in afterwards
    (deduped on provider/model/base_url).
    """
    from hermes_cli.fallback_config import get_fallback_chain

    return get_fallback_chain(_load_cfg())


def _agent_fallback_model(agent):
    """Return an agent's fallback chain without rehydrating deliberately empty chains."""
    if hasattr(agent, "_fallback_chain"):
        return getattr(agent, "_fallback_chain") or []
    if hasattr(agent, "_fallback_model"):
        return getattr(agent, "_fallback_model", None)
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
        "enabled_toolsets": getattr(agent, "enabled_toolsets", None)
        # Detached background tasks declare platform="tui" below: they have no
        # UI session id, so a renderer-routed event has nowhere to land. Resolve
        # their toolsets against that same platform rather than the gateway
        # process's, so they never carry GUI schema they cannot use.
        or _load_enabled_toolsets("tui"),
        "quiet_mode": True,
        "verbose_logging": False,
        "ephemeral_system_prompt": getattr(agent, "ephemeral_system_prompt", None)
        or None,
        "providers_allowed": getattr(agent, "providers_allowed", None),
        "providers_ignored": getattr(agent, "providers_ignored", None),
        "providers_order": getattr(agent, "providers_order", None),
        "provider_sort": getattr(agent, "provider_sort", None),
        "provider_require_parameters": getattr(
            agent, "provider_require_parameters", False
        ),
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
    kwargs.update(
        {
            "enabled_toolsets": ["terminal", "file"],
            "session_db": None,
            "skip_memory": True,
        }
    )
    return kwargs


def _preview_restart_history(session: dict, max_messages: int = 24, max_tool_chars: int = 1200) -> list[dict]:
    """Distill the parent session's recent history into a context the
    ephemeral preview-restart agent can actually use.

    The restart agent has no idea what app the user was building, what
    server they ran, what cwd was active, or which port belongs to which
    project. Without this, it would take the bare URL + console logs and
    guess — usually starting the wrong thing.

    We keep the last ``max_messages`` messages from the parent session so
    the restart agent sees recent user prompts, assistant replies, and
    most importantly any terminal/tool calls. Tool result payloads are
    truncated so we don't blow the context window with file dumps.
    """
    try:
        with session["history_lock"]:
            history = list(session.get("history", []) or [])
    except Exception:
        history = list(session.get("history", []) or [])

    if not history:
        return []

    # Anchor on the last user turn so we always include at least the most
    # recent request and the assistant/tool work that followed it. Then
    # extend backwards up to max_messages so we capture the prior context.
    last_user_idx = None
    for idx in range(len(history) - 1, -1, -1):
        if history[idx].get("role") == "user":
            last_user_idx = idx
            break

    start = max(0, len(history) - max_messages)
    if last_user_idx is not None:
        start = min(start, last_user_idx)

    trimmed: list[dict] = []
    for msg in history[start:]:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role not in ("user", "assistant", "tool", "system"):
            continue

        copy = {k: v for k, v in msg.items() if k != "reasoning"}
        # Truncate heavy tool outputs so a single 50KB file read doesn't
        # crowd out the rest of the context.
        if role == "tool":
            content = copy.get("content")
            if isinstance(content, str) and len(content) > max_tool_chars:
                copy["content"] = (
                    content[:max_tool_chars]
                    + f"\n... (truncated, original {len(content)} chars)"
                )
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
        exit_code = data.get("exit_code")
        if output:
            return output[-1200:]
        if data.get("session_id"):
            return f"Background process started: {data.get('session_id')}"
        if exit_code is not None:
            return f"terminal exited with code {exit_code}"

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
        # /new is a full conversation boundary: session-scoped runtime
        # overrides (/model, /reasoning, /fast) do NOT carry forward — the
        # fresh agent re-derives model/provider, reasoning, and service tier
        # from config.yaml (#48055, #23131). Session pins are cleared below so
        # a rebuild can't resurrect them. (Global process state is still never
        # touched — see the cross-session-contamination note in
        # _apply_model_switch.)
        session.pop("model_override", None)
        session.pop("create_reasoning_override", None)
        session.pop("create_service_tier_override", None)
        session.pop("one_turn_model_restore", None)
        new_agent = _make_agent(
            sid,
            session["session_key"],
            session_id=session["session_key"],
            platform_override=_session_source(session),
            context_cwd_is_launch_artifact=(
                _context_cwd_is_launch_artifact(session)
            ),
        )
    finally:
        _clear_session_context(tokens)
    session["agent"] = new_agent
    session["config_model_seen"] = _config_model_target()
    session["attached_images"] = []
    session["queued_prompt"] = None
    session.pop("queued_prompts", None)
    session["_queued_prompt_generation"] = int(session.get("_queued_prompt_generation", 0)) + 1
    session["edit_snapshots"] = {}
    session["image_counter"] = 0
    session["running"] = False
    session["show_reasoning"] = _load_show_reasoning()
    session["tool_progress_mode"] = _load_tool_progress_mode()
    session["tool_started_at"] = {}
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
