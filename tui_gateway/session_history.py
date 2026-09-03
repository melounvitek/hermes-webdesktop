"""Session history/message shaping: image-ref messages, content coercion, history->wire messages,
in-flight turn tracking and turn-failure detail.

Bodies are rebound onto server.py's globals (method_ctx.bind_module) and reference them bare.
"""

from __future__ import annotations

from .method_ctx import bind_module


def _active_image_routing_identity(agent: Any) -> tuple[str, str]:
    """Return the live provider/model, falling back before agent startup."""
    from agent.auxiliary_client import _read_main_model, _read_main_provider

    return (getattr(agent, "provider", "") or _read_main_provider(), getattr(agent, "model", "") or _read_main_model())


def _build_image_ref_message(user_text: str, image_paths: list[str]) -> str:
    """Reference attached images by path so the agent analyzes them in-loop with ``vision_analyze``.
    Pre-analyzing with the auxiliary vision model blocked submit 60-90s per photo and poisoned
    auto-titles with the description."""
    prefix = "\n\n".join(
        f"[The user attached an image: {p.name}]\n[Examine it with the vision_analyze tool using image_url: {p}]"
        for p in map(Path, image_paths) if p.exists()
    )
    text = user_text or ""
    if prefix:
        return f"{prefix}\n\n{text}" if text else prefix
    return text or "What do you see in this image?"


def _build_persist_message_with_image_refs(user_text: str, image_paths: list[str]) -> str:
    """Persisted form of the user's message: ``@image:<path>`` directives (the desktop renders them
    as images); ``_build_image_ref_message``'s ``image_url:`` hint is model-only, never persisted.
    Caption first, directives last: session previews are the first 60 chars of the first user
    message, so a leading directive would label the session with a truncated path."""
    from agent.context_references import format_reference_value

    text = user_text or ""
    refs = "\n".join(f"@image:{format_reference_value(p)}" for p in image_paths if Path(p).exists())
    if not refs:
        return text
    return f"{text}\n{refs}" if text else refs


def _build_persist_user_message(user_text: str, image_paths: list[str], run_message: Any) -> Any:
    """Shape the persisted user turn like the model payload: ``_flush_messages_to_session_db`` ignores
    a plain-string override for a list (native-vision) payload, so swap only the text part for the
    ``@image:`` form, keep image parts, and drop API-only text parts (barge-in note)."""
    persist_text = _build_persist_message_with_image_refs(user_text, image_paths)
    if not isinstance(run_message, list):
        return persist_text
    image_parts = [p for p in run_message if not (isinstance(p, dict) and p.get("type") == "text")]
    return [{"type": "text", "text": persist_text}, *image_parts]


_HISTORY_TEXT_KINDS = frozenset({"text", "input_text", "output_text"})
_HISTORY_IMAGE_KINDS = frozenset({"image_url", "input_image", "image"})
_HISTORY_AUDIO_KINDS = frozenset({"input_audio", "audio"})


def _history_part_image_url(part: dict) -> str:
    """The URL carried by an image part (``image_url`` dict or str), else ""."""
    image_url = part.get("image_url")
    if isinstance(image_url, dict):
        image_url = image_url.get("url")
    return image_url if isinstance(image_url, str) else ""


def _history_dict_text(content: dict, *, image_urls: bool) -> str:
    """Placeholder/text rendering of one structured content dict."""
    kind = content.get("type")
    if kind in _HISTORY_TEXT_KINDS:
        return str(content.get("text") or content.get("content") or "")
    if kind in _HISTORY_IMAGE_KINDS:
        return (_history_part_image_url(content) if image_urls else "") or "[image]"
    if kind in _HISTORY_AUDIO_KINDS:
        return "[audio]"
    if kind:
        return f"[{kind}]"
    if "text" in content:
        return str(content.get("text") or "")
    return "[structured content]"


def _content_display_text(content: Any) -> str:
    if isinstance(content, list):
        parts = (_content_display_text(part).strip() for part in content)
        return "\n".join(text for text in parts if text)
    if isinstance(content, dict):
        return _history_dict_text(content, image_urls=False)
    return "" if content is None else str(content)


def _coerce_message_text(content: Any) -> str:
    """Render ``message['content']`` (str, parts list, or one structured dict) as a plain string.
    Image parts keep their URL inline so the desktop's ``extractEmbeddedImages`` and the resume payload
    agree with the cached message (else the inline image flashed, then vanished); other structured
    shapes become a bracketed placeholder so resume doesn't drop the message."""
    if isinstance(content, list):
        chunks: list[str] = []
        for part in content:
            if isinstance(part, str):
                chunks.append(part)
                continue
            if not isinstance(part, dict):
                continue
            text = part.get("text")
            if isinstance(text, str):
                chunks.append(text)
                continue
            kind = part.get("type")
            if kind in _HISTORY_TEXT_KINDS:
                t = part.get("text") or part.get("content") or ""
                if t:
                    chunks.append(str(t))
            elif kind in _HISTORY_IMAGE_KINDS:
                chunks.append(f"\n{_history_part_image_url(part) or '[image]'}")
            elif kind in _HISTORY_AUDIO_KINDS:
                chunks.append("\n[audio]")
            elif kind:
                chunks.append(f"\n[{kind}]")
        return "".join(chunks)
    if isinstance(content, dict):
        return _history_dict_text(content, image_urls=True)
    return "" if content is None else str(content)


def _history_text_only_part(part: dict) -> bool:
    kind = part.get("type")
    return kind in _HISTORY_TEXT_KINDS or (kind is None and isinstance(part.get("text"), str))


def _is_text_only_busy_payload(content: Any) -> bool:
    """True when a busy submit carries only plain text, not attachments/media."""
    if isinstance(content, (str, int, float)):
        return True
    if isinstance(content, list):
        return bool(content) and all(
            isinstance(part, str) or (isinstance(part, dict) and _history_text_only_part(part)) for part in content
        )
    return isinstance(content, dict) and _history_text_only_part(content)


def _is_display_hidden_marker(role: str | None, text: str) -> bool:
    """Gateway notices (model-switch, personality) persist as role=user ``[System: …]`` rows so strict
    providers accept them mid-history; they must never render as a user bubble. Filtering in this one
    projection hides them everywhere (raw marker stays in ``session["history"]``) and keeps them from
    shifting the user-message ordinals the desktop reconciles against."""
    return role == "user" and text.lstrip().startswith("[System:")


def _skill_scaffold_projection(content_text: str) -> str:
    """The invocation a slash-skill-expanded turn came from, else "" — every UI renders
    ``/work fix the leak`` instead of the embedded skill body."""
    return describe_skill_invocation(content_text, separator=" ") or ""


def _expand_skill_invocation_for_replay(text: str, task_id: str) -> str:
    """Inverse of :func:`_skill_scaffold_projection`: rewind/regenerate hands back the projected
    invocation, and re-running it verbatim would drop the skill. Unchanged when not resolvable."""
    head, _, arg = (text or "").strip().partition(" ")
    if not head.startswith("/"):
        return text
    try:
        from agent.skill_commands import build_skill_invocation_message, resolve_skill_command_key

        cmd_key = resolve_skill_command_key(head.lstrip("/"))
        if cmd_key is None:
            return text
        return build_skill_invocation_message(cmd_key, arg.strip(), task_id=task_id) or text
    except Exception:
        # A skill that no longer resolves must not break the rewind.
        logger.debug("skill re-expansion failed for replay", exc_info=True)
        return text


# Opening of the crash-recovery note synthesized by _auto_continue_note; matched (not just built) for
# rows persisted before display typing existed and for the messaging gateway's twin note.
_AUTO_CONTINUE_NOTE_PREFIX = "[System note: Your previous turn was interrupted mid-run"


def _legacy_display_kind(role: str, text: str) -> str | None:
    """Infer the display type of a synthetic row persisted without one. New rows are typed at turn
    start (``persist_user_display_kind``); this prefix sniff migrates untyped rows already on disk (a
    turn killed mid-run never reached the stamp), which would otherwise paint as a user bubble."""
    if role == "user" and text.lstrip().startswith(_AUTO_CONTINUE_NOTE_PREFIX):
        return "auto_continue"
    return None


_HISTORY_REASONING_KEYS = ("reasoning", "reasoning_content", "reasoning_details", "codex_reasoning_items")
_HISTORY_ROLES = frozenset({"user", "assistant", "tool", "system"})


def _history_to_messages(history: list[dict]) -> list[dict]:
    messages = []
    tool_call_args = {}
    for m in history:
        if not isinstance(m, dict):
            continue
        m = project_compaction_message_for_display(m)
        if m is None:
            continue
        role = m.get("role")
        # display_kind="hidden": model-facing scaffolding the "[System:" sniff does not catch.
        if role not in _HISTORY_ROLES or m.get("display_kind") == "hidden":
            continue
        content_text = _coerce_message_text(m.get("content"))
        if _is_display_hidden_marker(role, content_text):
            continue
        if role == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                fn = tc.get("function", {})
                tc_id = tc.get("id", "")
                if tc_id and fn.get("name"):
                    try:
                        args = json.loads(fn.get("arguments", "{}"))
                    except (json.JSONDecodeError, TypeError):
                        args = {}
                    tool_call_args[tc_id] = (fn["name"], args)
            if not content_text.strip():
                continue
        if role == "tool":
            tc_id = m.get("tool_call_id", "")
            tc_info = tool_call_args.get(tc_id) if tc_id else None
            name = (tc_info[0] if tc_info else None) or m.get("tool_name") or "tool"
            args = (tc_info[1] if tc_info else None) or {}
            tool_msg = {"role": "tool", "name": name, "context": _tool_ctx(name, args)}
            # `context` is an 80-char preview; ship args so a full-call renderer isn't truncated.
            if args:
                tool_msg["args"] = args
            messages.append(tool_msg)
            continue
        # A reasoning-only assistant turn is kept so "Thinking…" still shows after resume/reload.
        has_reasoning = role == "assistant" and any(m.get(key) for key in _HISTORY_REASONING_KEYS)
        if not content_text.strip() and not has_reasoning:
            continue
        msg = {"role": role, "text": content_text}
        # Authoring time (Unix seconds) for display.timestamps; display-only.
        ts = m.get("timestamp")
        if isinstance(ts, (int, float)) and ts > 0:
            msg["timestamp"] = float(ts)
        # Durable row identity (_rows_to_conversation); reactions etc. address persisted messages by it.
        if m.get("_row_id") is not None:
            msg["row_id"] = m["_row_id"]
        if role == "user":
            invocation = _skill_scaffold_projection(content_text)
            if invocation:
                # The invocation, never the expanded body (rewind re-sends by ordinal).
                msg["text"] = invocation
                msg["display_kind"] = "skill_invocation"
        if role == "assistant":
            for key in _HISTORY_REASONING_KEYS:
                if m.get(key) is not None:
                    msg[key] = m[key]
        # Display-only timeline metadata (model switches, delegation events).
        display_kind = m.get("display_kind") or _legacy_display_kind(role, content_text)
        if display_kind:
            msg["display_kind"] = display_kind
        if m.get("display_metadata"):
            msg["display_metadata"] = m["display_metadata"]
        messages.append(msg)
    return messages


def _coerce_seed_history(value: Any) -> list[dict]:
    if not isinstance(value, list):
        return []
    history = []
    for item in value:
        if not isinstance(item, dict) or item.get("role") not in ("user", "assistant", "system"):
            continue
        content = item.get("content")
        if content is None:
            content = item.get("text")
        if isinstance(content, str) and content.strip():
            history.append({"role": item["role"], "content": content})
    return history


def _inflight_text(value: Any) -> str:
    return _content_display_text(value).strip()


def _start_inflight_turn(session: dict, text: Any) -> None:
    now = time.time()
    session["inflight_turn"] = {
        "assistant": "", "started_at": now, "streaming": True, "updated_at": now, "user": _inflight_text(text),
    }


def _append_inflight_delta(session: dict, delta: Any) -> None:
    text = "" if delta is None else str(delta)
    if not text:
        return
    turn = session.get("inflight_turn")
    if not isinstance(turn, dict):
        turn = {"assistant": "", "streaming": True, "user": ""}
    turn["assistant"] = f"{turn.get('assistant') or ''}{text}"
    turn["streaming"] = True
    turn["updated_at"] = time.time()
    session["inflight_turn"] = turn


def _record_inflight_correction(session: dict, text: Any) -> None:
    """Record an accepted mid-turn correction on the live turn — appended, never written over ``user``,
    so a resuming client can rebuild BOTH bubbles."""
    correction = _inflight_text(text)
    turn = session.get("inflight_turn")
    if not correction or not isinstance(turn, dict):
        return
    turn = dict(turn)
    turn["corrections"] = [*(turn.get("corrections") or []), correction]
    # Arrival-order boundary (assistant chars already streamed) so resuming clients place the bubble
    # between the output seen and the output redirected.
    turn["correction_offsets"] = [*(turn.get("correction_offsets") or []), len(str(turn.get("assistant") or ""))]
    turn["updated_at"] = time.time()
    session["inflight_turn"] = turn


def _clear_inflight_turn(session: dict) -> None:
    session["inflight_turn"] = None


def _fail_inflight_turn(session: dict, error: Any, error_surface: Optional[dict] = None) -> None:
    """Mark the in-flight turn terminal-error but keep it replayable: a failure's terminal frame can be
    lost on WS disconnect and the turn may never have been committed, so the snapshot lets
    ``session.resume`` replay prompt, partial text and error instead of stranding the client on a
    spinner. Lives until the next turn starts or the session closes. Caller holds history_lock."""
    message = str(error) if not isinstance(error, BaseException) else (str(error) or type(error).__name__)
    now = time.time()
    turn = session.get("inflight_turn")
    if not isinstance(turn, dict):
        turn = {"assistant": "", "user": "", "started_at": now}
    turn["assistant"] = str(turn.get("assistant") or "")
    turn["user"] = str(turn.get("user") or "")
    turn["error"] = message or "turn failed"
    turn["status"] = "error"
    turn["recoverable"] = True
    if error_surface:
        # {layer, code, retryable} so a reconnect renders the same layered error card.
        turn["error_surface"] = dict(error_surface)
    else:
        turn.pop("error_surface", None)
    turn["streaming"] = False
    turn["updated_at"] = now
    session["inflight_turn"] = turn


_TURN_FAILURE_DETAIL_LIMIT = 240
# Shortest prompt run counting as a quote-back: above shared boilerplate, below a quoted sentence.
_TURN_PROMPT_ECHO_WINDOW = 24
# Ceiling on the prompt we shingle (an @-expanded prompt can carry a whole file).
_TURN_PROMPT_ECHO_MAX_PROMPT = 65536


def _strip_prompt_echo(message: str, prompt: Any) -> str:
    """Blank runs of the submitted prompt that ``message`` quotes back: secret redaction is pattern-based
    and a provider 4xx echoing the request carries private prose matching no pattern. Any run of
    ``_TURN_PROMPT_ECHO_WINDOW``+ chars shared with the prompt (or its JSON-escaped form) becomes
    ``<prompt>``. Shingle-set matching keeps it linear. Only verbatim echo is stopped — a floor."""
    if not message or not prompt:
        return message
    needle = " ".join(str(prompt).split())[:_TURN_PROMPT_ECHO_MAX_PROMPT]
    window = _TURN_PROMPT_ECHO_WINDOW
    if len(needle) < window or len(message) < window:
        return message
    shingles = {needle[i:i + window] for i in range(len(needle) - window + 1)}
    try:
        escaped = json.dumps(needle)[1:-1]
    except Exception:
        escaped = ""
    if escaped and escaped != needle:
        shingles.update(escaped[i:i + window] for i in range(len(escaped) - window + 1))
    out: list[str] = []
    i = 0
    n = len(message)
    while i <= n - window:
        if message[i:i + window] in shingles:
            j = i + window
            while j < n and message[j - window + 1:j + 1] in shingles:
                j += 1
            out.append("<prompt>")
            i = j
        else:
            out.append(message[i])
            i += 1
    out.append(message[i:])
    return "".join(out)


def _turn_failure_detail(error: Any, reason: Any = None, prompt: Any = None) -> str:
    """Why a turn failed, for the ``tui turn finished`` bookend: ``""`` when nothing to say, else a
    fragment with its own leading space (distinguishes a provider 4xx from a budget wall or crashed
    finalizer). Two content contracts: ``redact_sensitive_text`` removes credentials;
    ``_strip_prompt_echo`` removes a 4xx body quoting ``prompt`` back. Invariant: this record may gain
    failure classification and provider detail, never the user's own content."""
    reason_text = str(reason or "").strip()
    message = str(error or "").strip()
    if isinstance(error, BaseException):
        message = message or type(error).__name__
    if not message and not reason_text:
        return ""
    try:
        from agent.redact import redact_sensitive_text

        message = redact_sensitive_text(message, force=True)
    except Exception:
        message = "<unredactable>"  # never fail open
    message = " ".join(message.split())
    # After the collapse (same shape both sides), before truncation (a quote must not survive the cut).
    message = _strip_prompt_echo(message, prompt)
    if len(message) > _TURN_FAILURE_DETAIL_LIMIT:
        message = message[:_TURN_FAILURE_DETAIL_LIMIT] + "\u2026"
    out = ""
    if reason_text:
        out += " failure_reason=%s" % " ".join(reason_text.split())
    if message:
        out += " cause=%r" % message
    return out


def register(server) -> None:
    """Publish this module's helpers + handlers onto ``server``, rebound to its globals."""
    bind_module(globals(), server, skip=("_",))
