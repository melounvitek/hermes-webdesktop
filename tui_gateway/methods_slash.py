"""slash.exec helpers: live-session command output and side-effect mirroring after a slash command ran in the worker.

Bodies are rebound onto server.py's globals at install time (see
method_ctx.bind_module), so they reference server.py globals bare.
"""

from __future__ import annotations


import contextlib

from .method_ctx import HandlerRegistry, bind_module

_registry = HandlerRegistry()


# ── Live-session slash output ────────────────────────────────────────

_LIVE_SESSION_DIRECT_COMMANDS = frozenset(
    {"clear", "compress", "effort", "history", "models", "prompt", "rename", "review", "status", "usage"}
)
# Answered from the live session ONLY when the agent lives on a compute host.
_ISOLATED_SESSION_READ_COMMANDS = frozenset({"context", "tools", "help"})

_NO_AGENT_USAGE = "(._.) No active agent -- send a message first."
_NO_AGENT = "No active agent -- send a message first."


def _format_live_review_output(session: Optional[dict], arg: str) -> str:
    """Dispatch /review against the live session's agent.

    The reviewer subagent runs on the async delegation rail; the TUI notification
    poller drains its completion back into this chat. The dispatch stamps the
    parent's durable session_id as the completion's session_key, which is what
    ``_session_owns_notification_event`` matches against.
    """
    if session is None:
        return "Nothing to review yet — send a message first."
    if _session_uses_compute_host(session):
        return "/review runs on the local agent only for now — this session's agent lives on a remote compute host."
    agent = session.get("agent")
    if agent is None:
        return "Nothing to review yet — send a message first."
    if session.get("running"):
        return "session busy — wait for the current turn to finish, then /review"

    with session.get("history_lock") or contextlib.nullcontext():
        snapshot = list(session.get("history", []))
    if not snapshot:
        snapshot = list(getattr(agent, "_session_messages", None) or [])

    try:
        from agent.review_engine import format_dispatch_note, start_review

        result = start_review(agent, snapshot, arg or "")
    except ValueError as exc:
        return str(exc)
    except Exception as exc:
        return f"/review failed to start: {exc}"
    return format_dispatch_note(result, arg or "")


def _format_live_usage_output(session: dict) -> str:
    agent = session.get("agent")
    usage = _session_usage_snapshot(session)
    if agent is None and not usage:
        return _NO_AGENT_USAGE
    if session.get("_metadata_message_count") is not None:
        message_count = int(session.get("_metadata_message_count") or 0)
    else:
        with session["history_lock"]:
            message_count = len(session.get("history", []))

    def n(key: str) -> str:
        return f"{int(usage.get(key) or 0):,}"

    lines = [
        "Session Token Usage",
        "────────────────────────────────────────",
        f"Model: {usage.get('model') or _metadata_mirror(session).get('model') or getattr(agent, 'model', '') or '(unknown)'}",
        f"Input tokens:                 {n('input')}",
        f"Output tokens:                {n('output')}",
    ]
    if int(usage.get("reasoning") or 0):
        lines.append(f"Reasoning tokens:             {n('reasoning')}")
    lines += [
        f"Prompt tokens:                {n('prompt')}",
        f"Completion tokens:            {n('completion')}",
        f"Total tokens:                 {n('total')}",
        f"API calls:                    {n('calls')}",
    ]
    if usage.get("context_max"):
        lines.append(
            f"Current context:              {n('context_used')} / {n('context_max')} "
            f"({int(usage.get('context_percent') or 0)}%)"
        )
    lines += [f"Messages:                     {message_count:,}", f"Compressions:                 {n('compressions')}"]
    return "\n".join(lines)


def _live_session_messages(session: dict) -> Optional[list]:
    """Session-scoped transcript read; None when no db/key or the read fails. Uses
    ``_session_db`` (not ``_get_db()``): a profile session's rows live in its own
    profile's state.db, and through the launch handle this read comes back empty."""
    with _session_db(session) as db:
        if db is not None and session.get("session_key"):
            try:
                return db.get_messages_as_conversation(
                    session["session_key"], include_ancestors=True, include_row_ids=True
                )
            except Exception:
                pass
    return None


def _format_live_history_output(session: dict) -> str:
    with session["history_lock"]:
        history = list(session.get("history", []))
    db_history = _live_session_messages(session)
    messages = _history_to_messages(history if db_history is None else db_history)
    if not messages:
        return "No conversation history yet."
    lines = ["Conversation History", "────────────────────────────────────────"]
    for idx, message in enumerate(messages, start=1):
        role = str(message.get("role") or "unknown")
        label = "You" if role == "user" else "Hermes" if role == "assistant" else role.title()
        text = str(message.get("text") or message.get("context") or "").strip()
        if len(text) > 400:
            text = f"{text[:400]}..."
        lines.append(f"[{label} #{idx}] {text or '(no text)'}")
    return "\n".join(lines)


def _format_live_prompt_output(session: dict) -> str:
    agent = session.get("agent")
    mirror = _metadata_mirror(session)
    if agent is None and "system_prompt" not in mirror:
        return _NO_AGENT
    prompt = (
        mirror.get("system_prompt")
        or getattr(agent, "ephemeral_system_prompt", None)
        or getattr(agent, "_cached_system_prompt", None)
        or ""
    )
    if not prompt:
        return "Current system prompt is not built yet; send a message first."
    return f"Current system prompt:\n{prompt}"


def _format_live_context_output(session: dict) -> str:
    try:
        messages = _history_to_messages(_live_session_messages(session) or [])
    except Exception:
        messages = []  # malformed db rows fall back to the live history below
    if not messages:
        with session["history_lock"]:
            messages = _history_to_messages(list(session.get("history", [])))
    usage = _session_usage_snapshot(session)
    mirror = _metadata_mirror(session)
    lines = [f"Conversation: {len(messages)} messages" if messages else "Conversation is empty (no messages yet)."]
    roles: dict[str, int] = {}
    for msg in messages:
        role = str(msg.get("role") or "unknown")
        roles[role] = roles.get(role, 0) + 1
    lines.append(
        f"  user: {roles.get('user', 0)}, assistant: {roles.get('assistant', 0)}, "
        f"tool: {roles.get('tool', 0)}, system: {roles.get('system', 0)}"
    )
    model = mirror.get("model") or usage.get("model") or ""
    if model:
        lines.append(f"Model: {model}")
    lines.append(f"Provider: {mirror.get('provider') or 'auto'}")
    context_used = int(usage.get("context_used") or usage.get("total") or 0)
    context_max = int(usage.get("context_max") or 0)
    if context_used and context_max:
        lines.append(
            f"Context usage: ~{context_used:,} / {context_max:,} tokens ({(context_used / context_max) * 100:.1f}%)"
        )
    elif context_used:
        lines.append(f"Context usage: ~{context_used:,} tokens")
    if usage.get("compressions"):
        lines.append(f"Compressions: {int(usage.get('compressions') or 0):,}")
    return "\n".join(lines)


def _format_live_tools_output(session: dict) -> str:
    info = _session_info(session.get("agent"), session)
    groups = info.get("tools") if isinstance(info, dict) else {}
    if not isinstance(groups, dict) or not groups:
        return "No tools available."
    names = sorted({str(n) for g in groups.values() if isinstance(g, list) for n in g})
    if not names:
        return "No tools available."
    return "Available tools ({}):\n{}".format(len(names), "\n".join(f"  {name}" for name in names))


def _format_live_help_output() -> str:
    try:
        from hermes_cli.commands import COMMANDS_BY_CATEGORY

        lines = ["Available commands:", ""]
        for category, commands in COMMANDS_BY_CATEGORY.items():
            lines.append(f"{category}:")
            lines.extend(f"  {cmd:<15} {desc}" for cmd, desc in commands.items())
        return "\n".join(lines)
    except Exception as exc:
        return f"help unavailable: {exc}"


def _format_live_model_output(session: dict) -> str:
    agent = session.get("agent")
    model = getattr(agent, "model", "") if agent is not None else ""
    provider = getattr(agent, "provider", "") if agent is not None else ""
    if model and provider:
        return f"Current model: {model} ({provider})"
    return f"Current model: {model}" if model else "Current model: (unknown)"


def _format_live_status_output(sid: str) -> str:
    response = _methods["session.status"]("status", {"session_id": sid})
    if response.get("error"):
        return str(response["error"].get("message") or "status unavailable")
    return str(response.get("result", {}).get("output") or "")


# name → (reply when there is no session, formatter(sid, session, arg)). A None
# no-session reply means the formatter handles a missing session itself.
_LIVE_SLASH_OUTPUT = {
    "compress": ("no active session for /compress", lambda sid, s, a: _mirror_slash_side_effects(sid, s, f"/compress {a}".strip())),
    "usage": (_NO_AGENT_USAGE, lambda sid, s, a: _format_live_usage_output(s)),
    "review": (None, lambda sid, s, a: _format_live_review_output(s, a)),
    "history": ("No conversation history yet.", lambda sid, s, a: _format_live_history_output(s)),
    "prompt": (_NO_AGENT, lambda sid, s, a: _format_live_prompt_output(s)),
    "status": (None, lambda sid, s, a: _format_live_status_output(sid)),
    "context": ("Conversation is empty (no messages yet).", lambda sid, s, a: _format_live_context_output(s)),
    "tools": ("No tools available.", lambda sid, s, a: _format_live_tools_output(s)),
    "help": (None, lambda sid, s, a: _format_live_help_output()),
    "clear": (None, lambda sid, s, a: "Screen clear is terminal-only; desktop/TUI chat left unchanged."),
    "models": (None, lambda sid, s, a: "Use /model to view or switch the current model; desktop users can also open the model picker."),
    "rename": (None, lambda sid, s, a: "Use /title <name> to rename this session."),
    "effort": (None, lambda sid, s, a: "Use /reasoning <effort> to change reasoning effort."),
}


def _live_slash_command_output(sid: str, session: Optional[dict], name: str, arg: str) -> Optional[str]:
    """Answer a slash command from the live session instead of the slash worker; None = not ours."""
    name = (name or "").lstrip("/").lower()
    arg = arg or ""
    if name == "model" and not arg.strip():
        return _format_live_model_output(session or {})
    if name in _ISOLATED_SESSION_READ_COMMANDS:
        if not (session is not None and _session_uses_compute_host(session)):
            return None
    elif name not in _LIVE_SESSION_DIRECT_COMMANDS:
        return None
    entry = _LIVE_SLASH_OUTPUT.get(name)
    if entry is None:
        return None
    no_session_reply, fmt = entry
    if session is None and no_session_reply is not None:
        return no_session_reply
    return fmt(sid, session, arg)


# ── Side-effect mirroring ────────────────────────────────────────────

# Read-then-mutate live agent/session state that a running turn is using; rejected
# while running (parity with session.compress / session.undo and the gateway's
# running-agent /model guard).
_MUTATES_WHILE_RUNNING = frozenset({"model", "personality", "prompt", "compress"})


def _compress_live_with_feedback(sid: str, session: dict, agent, arg: str, *, snapshot_kwargs: bool) -> dict:
    """Compress the live session; return the ``summarize_manual_compression`` dict.

    Shared by command.dispatch /compress and the slash mirror so every route shows
    "compressed N → M messages / ~X → ~Y tokens". ``snapshot_kwargs`` forwards the
    pre-read snapshot (approx_tokens/before_messages/history_version) to
    ``_compress_session_history``; the slash mirror passes only the raw arg. The raw
    arg goes through unparsed — the choke point parses ``here [N]`` / ``--keep N``.
    CompressionLockHeld and other errors propagate to the caller, which finalizes
    the deferred context-engine notification.
    """
    from agent.conversation_compression import finalize_context_engine_compression_notification
    from agent.manual_compression_feedback import summarize_manual_compression
    from agent.model_metadata import estimate_request_tokens_rough

    with session["history_lock"]:
        before_messages = list(session.get("history", []))
        history_version = int(session.get("history_version", 0))
    sys_prompt = getattr(agent, "_cached_system_prompt", "") or ""
    tools = getattr(agent, "tools", None) or None
    before_tokens = (
        estimate_request_tokens_rough(before_messages, system_prompt=sys_prompt, tools=tools) if before_messages else 0
    )
    if snapshot_kwargs:
        _compress_session_history(
            session,
            arg.strip() or None,
            approx_tokens=before_tokens,
            before_messages=before_messages,
            history_version=history_version,
        )
    else:
        _compress_session_history(session, arg)
    _sync_session_key_after_compress(sid, session)
    with session["history_lock"]:
        after_messages = list(session.get("history", []))
    after_tokens = (
        estimate_request_tokens_rough(
            after_messages,
            system_prompt=getattr(agent, "_cached_system_prompt", "") or sys_prompt,
            tools=getattr(agent, "tools", None) or tools,
        )
        if after_messages
        else 0
    )
    _emit("session.info", sid, _session_info(agent, session))
    fb = summarize_manual_compression(
        before_messages,
        after_messages,
        before_tokens,
        after_tokens,
        compression_state=getattr(agent, "context_compressor", None),
    )
    finalize_context_engine_compression_notification(agent, committed=True)
    return fb


def _mirror_model(sid, session, agent, arg) -> str:
    if arg and agent:
        return _apply_model_switch(sid, session, arg).get("warning", "")
    return ""


def _mirror_approvals(sid, session, agent, arg) -> str:
    # The worker already persisted approvals.mode; the bare read-only form needs no repaint.
    if arg:
        broadcast_session_info()
    return ""


def _mirror_personality(sid, session, agent, arg) -> str:
    if arg and agent:
        pname, new_prompt = _validate_personality(arg, _load_cfg())
        # Persist through the single owner so this surface never drifts from the others.
        from hermes_cli.personality import persist_personality

        persist_personality(pname)
        _apply_personality_to_session(sid, session, new_prompt, pname)
    return ""


def _mirror_prompt(sid, session, agent, arg) -> str:
    if agent:
        cfg = _load_cfg()
        agent.ephemeral_system_prompt = _prompt_text((cfg.get("agent") or {}).get("system_prompt", "")) or None
        agent._cached_system_prompt = None
    return ""


def _mirror_compress(sid, session, agent, arg) -> str:
    if not agent:
        return ""
    try:
        fb = _compress_live_with_feedback(sid, session, agent, arg, snapshot_kwargs=False)
    except CompressionLockHeld as e:
        from agent.manual_compression_feedback import describe_compression_lock_skip

        return describe_compression_lock_skip(e.holder)
    lines = [fb["headline"], fb["token_line"]]
    if fb.get("note"):
        lines.append(fb["note"])
    return "\n".join(lines)


def _mirror_fast(sid, session, agent, arg) -> str:
    if agent:
        mode = arg.lower()
        if mode in {"fast", "on"}:
            agent.service_tier = "priority"
        elif mode in {"normal", "off"}:
            agent.service_tier = None
        elif mode in {"auto", "cold"}:
            agent.service_tier = mode
        _emit("session.info", sid, _session_info(agent, session))
    return ""


def _mirror_reload_mcp(sid, session, agent, arg) -> str:
    if agent and hasattr(agent, "reload_mcp_tools"):
        agent.reload_mcp_tools()
    return ""


def _mirror_stop(sid, session, agent, arg) -> str:
    from tools.process_registry import process_registry

    process_registry.kill_all()
    return ""


_SLASH_MIRRORS = {
    "model": _mirror_model,
    "approvals": _mirror_approvals,
    "personality": _mirror_personality,
    "prompt": _mirror_prompt,
    "compress": _mirror_compress,
    "fast": _mirror_fast,
    "reload-mcp": _mirror_reload_mcp,
    "stop": _mirror_stop,
}


def _compute_host_slash(sid: str, session: dict, name: str, command: str) -> tuple[str, str]:
    """Forward a mutating slash command to the session's compute host.

    Returns ``(status, text)``: ``pending`` (compress still running after the wait),
    ``failed`` (transport error/timeout), ``rejected`` (host control.error), ``ok``
    (host output; metadata mirror already applied). Compress waits longer and installs
    a late-ack adopter so a slow compression still lands in this session.
    """
    route_name = f"slash.{name}"
    is_compress = name == "compress"
    _late_session = session

    def _on_late_ack(late: dict, _sid=sid) -> None:
        _adopt_late_compute_host_compress_ack(_sid, _late_session, late, route_name=route_name)

    try:
        ack = _send_compute_host_control(
            sid,
            route_name=route_name,
            command=command,
            wait=True,
            **({"timeout": _compute_host_compress_wait_seconds(), "on_late_ack": _on_late_ack} if is_compress else {}),
        )
    except queue.Empty:
        if is_compress:
            return "pending", "compression still running in the background; the transcript will refresh when it finishes"
        return "failed", f"compute-host {route_name} failed: timed out"
    except Exception as exc:
        return "failed", f"compute-host {route_name} failed: {exc}"
    if ack.get("type") in {"control.error", "error"}:
        return "rejected", str(ack.get("message") or f"compute-host {route_name} failed")
    _apply_compute_host_metadata_mirror(session, ack)
    return "ok", str(ack.get("output") or "")


def _mirror_slash_side_effects(sid: str, session: dict, command: str) -> str:
    """Apply side effects that must also hit the gateway's live agent."""
    parts = command.lstrip("/").split(None, 1)
    if not parts:
        return ""
    name, arg, agent = parts[0], (parts[1].strip() if len(parts) > 1 else ""), session.get("agent")
    if name == "compact":
        # /compact aliases /compress everywhere; the compute-host control forwards the
        # raw alias verbatim, so without this the child mirror silently no-ops.
        name = "compress"

    if _session_uses_compute_host(session) and name in _MUTATES_WHILE_RUNNING:
        return _compute_host_slash(sid, session, name, command)[1]
    if name in _MUTATES_WHILE_RUNNING and session.get("running"):
        return f"session busy — /interrupt the current turn before running /{name}"

    mirror = _SLASH_MIRRORS.get(name)
    if mirror is None:
        return ""
    try:
        return mirror(sid, session, agent, arg)
    except Exception as e:
        if name == "compress" and agent:
            from agent.conversation_compression import finalize_context_engine_compression_notification

            finalize_context_engine_compression_notification(agent, committed=False)
        return f"live session sync failed: {e}"


def register(server) -> None:
    """Publish this module's helpers + handlers onto ``server``, rebound to its globals."""
    bind_module(globals(), server, skip=("_",))
