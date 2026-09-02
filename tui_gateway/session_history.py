"""Session history/message shaping: image-ref messages, content coercion, history->wire messages, in-flight turn tracking and turn-failure detail.

Bodies are rebound onto server.py's globals at install time (see
method_ctx.bind_module), so they reference server.py globals bare.
"""

from __future__ import annotations


from .method_ctx import HandlerRegistry, bind_module

_registry = HandlerRegistry()


def _active_image_routing_identity(agent: Any) -> tuple[str, str]:
    """Return the live provider/model, falling back before agent startup."""
    from agent.auxiliary_client import _read_main_model, _read_main_provider

    return (
        getattr(agent, "provider", "") or _read_main_provider(),
        getattr(agent, "model", "") or _read_main_model(),
    )


def _build_image_ref_message(user_text: str, image_paths: list[str]) -> str:
    """Reference attached images by path so the agent analyzes them in-loop.

    This used to pre-analyze every image with the auxiliary vision model
    *before* the turn was dispatched (``_enrich_with_attached_images``):
    serial blocking calls on the submit path — 60-90s per large photo —
    with failures silently swallowed and an interrupt during the window
    killing the turn with zero API calls (#83291). It also prepended the
    vision description to the first user message, poisoning session
    auto-titles (#82339). The CLI never gates turn dispatch on vision
    like this, which is why the same message was seconds there and
    minutes on desktop.

    Now the turn starts immediately. The agent examines each image itself
    with ``vision_analyze`` — its own retries, visible tool progress —
    exactly how the ``@folder:`` reference path already behaves, which
    responds in seconds for the same images.
    """
    parts: list[str] = []
    for path in image_paths:
        p = Path(path)
        if not p.exists():
            continue
        parts.append(
            f"[The user attached an image: {p.name}]\n"
            f"[Examine it with the vision_analyze tool using image_url: {p}]"
        )

    text = user_text or ""
    prefix = "\n\n".join(parts)
    if prefix:
        return f"{prefix}\n\n{text}" if text else prefix
    return text or "What do you see in this image?"


def _build_persist_message_with_image_refs(user_text: str, image_paths: list[str]) -> str:
    """Build the clean, UI-recognizable version of the user's message for
    persisting to session history. Uses ``@image:<path>`` directives — the
    format the desktop client (directive-text.tsx / HERMES_DIRECTIVE_RE)
    actually parses and renders as an image — unlike
    ``_build_image_ref_message``, which embeds an
    ``image_url:`` hint meant only for the model and must never be
    persisted as-is (it silently breaks image rendering after a full
    restart, and reorders image/text on live session-switch reconciliation).

    The caption leads and the directives trail: session previews are the first
    60 characters of the first user message (``list_sessions_rich``), so a
    leading directive would label the session with a truncated file path in the
    sidebar, switcher, and command palette. Clients lift the refs out of the
    body by line, so their position does not affect how the turn renders.
    """
    from agent.context_references import format_reference_value

    text = user_text or ""
    refs = "\n".join(f"@image:{format_reference_value(p)}" for p in image_paths if Path(p).exists())
    if not refs:
        return text
    return f"{text}\n{refs}" if text else refs


def _build_persist_user_message(user_text: str, image_paths: list[str], run_message: Any) -> Any:
    """Shape the persisted user turn to match what was sent to the model.

    Native-vision turns send ``content`` as a parts list, and
    ``_flush_messages_to_session_db`` deliberately ignores a plain-string
    override for a list payload (a text override must not erase a turn's
    image/audio summary). So mirror the shape: replace only the text part with
    the ``@image:`` ref form and keep the image parts, so the model still has
    the pixels for the rest of the session. Any API-only text part (the
    barge-in note) is dropped along the way, which is the point of the override.
    """
    persist_text = _build_persist_message_with_image_refs(user_text, image_paths)
    if not isinstance(run_message, list):
        return persist_text
    image_parts = [p for p in run_message if not (isinstance(p, dict) and p.get("type") == "text")]
    return [{"type": "text", "text": persist_text}, *image_parts]


def _content_display_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, (int, float)):
        return str(content)
    if isinstance(content, list):
        parts = []
        for part in content:
            text = _content_display_text(part).strip()
            if text:
                parts.append(text)
        return "\n".join(parts)
    if isinstance(content, dict):
        kind = content.get("type")
        if kind in {"text", "input_text", "output_text"}:
            return str(content.get("text") or content.get("content") or "")
        if kind in {"image_url", "input_image", "image"}:
            return "[image]"
        if kind in {"input_audio", "audio"}:
            return "[audio]"
        if kind:
            return f"[{kind}]"
        if "text" in content:
            return str(content.get("text") or "")
        return "[structured content]"
    return str(content)


def _coerce_message_text(content: Any) -> str:
    """Render ``message['content']`` as a plain string for transport.

    Provider-side, ``content`` may be a string (most common), a list of
    multimodal parts (e.g. ``[{"type": "text", "text": "..."},
    {"type": "image_url", "image_url": {...}}]``), or a single structured
    dict. Calling ``.strip()`` on a list raises ``'list' object has no
    attribute 'strip'`` and breaks session resume entirely.

    Image parts (``image_url``) are preserved by appending the underlying
    URL (data: or http:) into the text. The desktop renderer pulls these
    back out via ``extractEmbeddedImages`` so the user sees the image
    instead of the URL — and it stops the resume payload from disagreeing
    with the cached message (which would otherwise cause the inline image
    to flash, then disappear when the resume payload overwrites the cache).

    Other structured dict shapes (audio, unknown types) fall back to a
    bracketed placeholder so resume doesn't drop the message entirely.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, (int, float)):
        return str(content)
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
            if kind in {"text", "input_text", "output_text"}:
                t = part.get("text") or part.get("content") or ""
                if t:
                    chunks.append(str(t))
                continue
            if kind in {"image_url", "input_image", "image"}:
                image_url = part.get("image_url")
                url = ""
                if isinstance(image_url, dict):
                    candidate = image_url.get("url")
                    if isinstance(candidate, str):
                        url = candidate
                elif isinstance(image_url, str):
                    url = image_url
                if url:
                    chunks.append(f"\n{url}")
                else:
                    chunks.append("\n[image]")
                continue
            if kind in {"input_audio", "audio"}:
                chunks.append("\n[audio]")
                continue
            if kind:
                chunks.append(f"\n[{kind}]")
        return "".join(chunks)
    if isinstance(content, dict):
        kind = content.get("type")
        if kind in {"text", "input_text", "output_text"}:
            return str(content.get("text") or content.get("content") or "")
        if kind in {"image_url", "input_image", "image"}:
            image_url = content.get("image_url")
            url = ""
            if isinstance(image_url, dict):
                candidate = image_url.get("url")
                if isinstance(candidate, str):
                    url = candidate
            elif isinstance(image_url, str):
                url = image_url
            return url or "[image]"
        if kind in {"input_audio", "audio"}:
            return "[audio]"
        if kind:
            return f"[{kind}]"
        if "text" in content:
            return str(content.get("text") or "")
        return "[structured content]"
    return str(content)


_TEXT_ONLY_BUSY_PART_KINDS = frozenset({"text", "input_text", "output_text"})


def _is_text_only_busy_payload(content: Any) -> bool:
    """True when a busy submit carries only plain text, not attachments/media."""
    if content is None:
        return False
    if isinstance(content, (str, int, float)):
        return True
    if isinstance(content, list):
        if not content:
            return False
        for part in content:
            if isinstance(part, str):
                continue
            if not isinstance(part, dict):
                return False
            kind = part.get("type")
            if kind in _TEXT_ONLY_BUSY_PART_KINDS:
                continue
            if kind is None and isinstance(part.get("text"), str):
                continue
            return False
        return True
    if isinstance(content, dict):
        kind = content.get("type")
        if kind in _TEXT_ONLY_BUSY_PART_KINDS:
            return True
        return kind is None and isinstance(content.get("text"), str)
    return False


def _is_display_hidden_marker(role: str | None, text: str) -> bool:
    """Gateway bookkeeping notices (model-switch, personality) are persisted as
    role=user ``[System: …]`` rows so strict providers accept them mid-history.
    They are model-facing runtime metadata, not user turns, and must never
    render as a user bubble in ANY client transcript (desktop, TUI, CLI, web).

    Filtering here — the single display projection every surface reads — hides
    them everywhere while the raw marker stays in ``session["history"]`` for the
    model. It also removes the stored marker from the payload the desktop
    reconciles against, so it can no longer shift user-message ordinals and
    duplicate the optimistic prompt (#67603)."""
    return role == "user" and text.lstrip().startswith("[System:")


def _skill_scaffold_projection(content_text: str) -> str:
    """Return the invocation a slash-skill-expanded turn came from, else "".

    A ``/skill`` invocation expands into a model-facing message that embeds the
    whole skill body. That payload belongs to the agent — every UI renders the
    invocation (``/work fix the leak``) instead, so no surface can leak the
    body into a chat bubble.
    """
    return describe_skill_invocation(content_text, separator=" ") or ""


def _expand_skill_invocation_for_replay(text: str, task_id: str) -> str:
    """Re-expand a projected `/skill` invocation before re-running that turn.

    The inverse of :func:`_skill_scaffold_projection`. Because a skill turn is
    displayed as its invocation, a rewind/regenerate hands us back
    ``/work fix the leak`` rather than the body the agent originally saw —
    re-running that verbatim would drop the skill. Re-expanding here keeps the
    body server-side (no client ever holds it) and makes the replayed turn
    identical to the original.

    Returns *text* unchanged when it isn't a resolvable skill invocation.
    """
    head, _, arg = (text or "").strip().partition(" ")
    if not head.startswith("/"):
        return text

    try:
        from agent.skill_commands import (
            build_skill_invocation_message,
            resolve_skill_command_key,
        )

        cmd_key = resolve_skill_command_key(head.lstrip("/"))
        if cmd_key is None:
            return text

        return build_skill_invocation_message(cmd_key, arg.strip(), task_id=task_id) or text
    except Exception:
        # A skill that no longer resolves (renamed, disabled, external dir
        # gone) must not break the rewind — replay the text as typed.
        logger.debug("skill re-expansion failed for replay", exc_info=True)
        return text


# Opening of the crash-recovery note synthesized by _auto_continue_note.
# Matched (not just built) so a row persisted before the display type was
# stamped at turn start still reads as a timeline event, and to recognize the
# messaging gateway's twin note.
_AUTO_CONTINUE_NOTE_PREFIX = "[System note: Your previous turn was interrupted mid-run"


def _legacy_display_kind(role: str, text: str) -> str | None:
    """Infer the display type of a synthetic row persisted without one.

    Turn-start typing (see ``persist_user_display_kind``) covers everything
    written from here on. Sessions already on disk carry untyped rows — and a
    turn killed mid-run never reached the post-turn stamp at all, which is
    exactly the auto-continue case — so the raw recovery note would paint as a
    user bubble forever. Sniffing the one fixed synthetic prefix is the
    migration for those rows; it is not how new rows get typed.
    """
    if role == "user" and text.lstrip().startswith(_AUTO_CONTINUE_NOTE_PREFIX):
        return "auto_continue"
    return None


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
        if role not in {"user", "assistant", "tool", "system"}:
            continue
        # An explicit display_kind="hidden" row is model-facing scaffolding
        # (compaction references, interrupted-turn checkpoints). The string
        # sniff below only catches the "[System:" convention; honor the
        # declared field too, or scaffolding reaches every surface that reads
        # this projection.
        if m.get("display_kind") == "hidden":
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
            # This is the display projection, so keep it faithful. `context`
            # is an 80-char preview for collapsed row titles. A renderer that
            # shows the full call (the expanded `$` transcript in the desktop)
            # rebuilds it from args. When only the preview shipped, that
            # truncation was permanent.
            if args:
                tool_msg["args"] = args
            messages.append(tool_msg)
            continue
        # An assistant turn may carry only reasoning/thinking content with no
        # visible text (extended-thinking turns, thinking-only recovery
        # responses). Such a turn is persisted with its reasoning fields and is
        # recallable from the transcript, but dropping it here as "empty" makes
        # it vanish from the resumed/reloaded session view while the desktop's
        # reasoning disclosure has nothing to render. Keep it when it carries
        # reasoning so the "Thinking…" block still shows. (#44022)
        reasoning_keys = (
            "reasoning",
            "reasoning_content",
            "reasoning_details",
            "codex_reasoning_items",
        )
        has_reasoning = role == "assistant" and any(
            m.get(key) for key in reasoning_keys
        )
        if not content_text.strip() and not has_reasoning:
            continue
        msg = {"role": role, "text": content_text}
        # Persisted authoring time (Unix seconds) for display.timestamps
        # renderers (#41531). Display-only: never fed back into model context.
        ts = m.get("timestamp")
        if isinstance(ts, (int, float)) and ts > 0:
            msg["timestamp"] = float(ts)
        # Durable row identity, stamped by _rows_to_conversation. The renderer's
        # own message ids are ephemeral (timestamp+index derived, and a
        # different shape for live vs rehydrated vs optimistic rows), so
        # anything that addresses a specific persisted message later — message
        # reactions — needs this instead.
        if m.get("_row_id") is not None:
            msg["row_id"] = m["_row_id"]
        if role == "user":
            invocation = _skill_scaffold_projection(content_text)
            if invocation:
                # Show the invocation, never the expanded skill body. The raw
                # payload stays server-side: a rewind/regenerate re-sends the
                # turn by ordinal, so no client needs it.
                msg["text"] = invocation
                msg["display_kind"] = "skill_invocation"
        if role == "assistant":
            for key in reasoning_keys:
                if key in m and m.get(key) is not None:
                    msg[key] = m.get(key)
        # Forward display-only timeline metadata so the TUI can render
        # model switches and delegation completions as events instead of
        # opaque user messages, and hide compaction handoffs entirely.
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
        if not isinstance(item, dict):
            continue

        role = item.get("role")
        if role not in ("user", "assistant", "system"):
            continue

        content = item.get("content")
        if content is None:
            content = item.get("text")
        if not isinstance(content, str) or not content.strip():
            continue

        history.append({"role": role, "content": content})

    return history


def _inflight_text(value: Any) -> str:
    return _content_display_text(value).strip()


def _start_inflight_turn(session: dict, text: Any) -> None:
    now = time.time()
    session["inflight_turn"] = {
        "assistant": "",
        "started_at": now,
        "streaming": True,
        "updated_at": now,
        "user": _inflight_text(text),
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
    """Record an accepted mid-turn correction on the live turn.

    The correction is appended, never written over ``user``: a resuming client
    must be able to rebuild BOTH bubbles. Overwriting the slot erased the
    prompt that started the turn from the only snapshot resume can read, so a
    reconnect (or a dev hot-reload that wipes the renderer cache) repainted the
    thread with the user's original message missing.
    """
    correction = _inflight_text(text)
    if not correction:
        return
    turn = session.get("inflight_turn")
    if not isinstance(turn, dict):
        return
    turn = dict(turn)
    corrections = list(turn.get("corrections") or [])
    corrections.append(correction)
    turn["corrections"] = corrections
    # Arrival-order boundary: how much assistant text had already streamed
    # when this correction was accepted. Resuming clients use it to place the
    # correction bubble AFTER the output the user had already seen and BEFORE
    # the output it redirected (#73793) instead of above the whole reply.
    offsets = list(turn.get("correction_offsets") or [])
    offsets.append(len(str(turn.get("assistant") or "")))
    turn["correction_offsets"] = offsets
    turn["updated_at"] = time.time()
    session["inflight_turn"] = turn


def _clear_inflight_turn(session: dict) -> None:
    session["inflight_turn"] = None


def _fail_inflight_turn(
    session: dict, error: Any, error_surface: Optional[dict] = None
) -> None:
    """Mark the in-flight turn terminal-error but keep it replayable.

    Normal completion clears ``inflight_turn`` because the response is now in
    canonical history. Failures are different: the terminal frame can be lost
    on a WS disconnect, and the failed turn may never have been committed.
    Retaining a compact error snapshot lets ``session.resume`` replay the
    user's prompt, any partial assistant text, and the error itself instead of
    leaving the client stranded on a spinner or hydrating from stale DB state.
    The snapshot lives until the next turn starts (``_start_inflight_turn``
    overwrites it) or the session closes.

    Caller must hold ``session["history_lock"]``.
    """
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
        # Structured {layer, code, retryable} descriptor — replayed to
        # resuming clients via the resume snapshot so a reconnect renders the
        # same layered error card the live frame carried.
        turn["error_surface"] = dict(error_surface)
    else:
        turn.pop("error_surface", None)
    turn["streaming"] = False
    turn["updated_at"] = now
    session["inflight_turn"] = turn


_TURN_FAILURE_DETAIL_LIMIT = 240
# Shortest run of the submitted prompt that counts as the provider quoting it
# back. Long enough that shared boilerplate ("Invalid request for model ") does
# not trip it, short enough to catch a quoted sentence.
_TURN_PROMPT_ECHO_WINDOW = 24
# Ceiling on the prompt we shingle. An @-expanded prompt can carry a whole
# file; the failure path must stay cheap.
_TURN_PROMPT_ECHO_MAX_PROMPT = 65536


def _strip_prompt_echo(message: str, prompt: Any) -> str:
    """Blank runs of the submitted prompt that ``message`` quotes back.

    Secret redaction and prompt omission are different contracts, and only the
    first one is pattern-based. A provider 4xx that echoes the request carries
    ordinary private prose -- a paragraph about a person, a pasted file from an
    ``@`` reference -- that matches no credential pattern and would otherwise
    reach the log intact. This closes that path directly: anything the message
    shares with the prompt for ``_TURN_PROMPT_ECHO_WINDOW`` characters or more
    becomes ``<prompt>``.

    Shingle-set matching, not a diff: cost is linear in both strings, which
    matters because this runs on every failed turn and an ``@`` reference can
    make the prompt arbitrarily long. The JSON-escaped form of the prompt is
    shingled too, since a provider that hands back its own request body often
    hands it back escaped.

    Verbatim echo is what this stops. A paraphrase, a re-encoding (base64, a
    different unicode normalization) or a summary of the prompt would survive,
    so this is a floor and not a proof; the guarantee it does give is that the
    prompt cannot reach the record by being quoted.
    """
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
        shingles.update(
            escaped[i:i + window] for i in range(len(escaped) - window + 1)
        )
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
    """Render why a turn failed, for the ``tui turn finished`` bookend.

    Returns ``""`` when there is nothing to say, otherwise a fragment that
    already carries its own leading space, so the caller can append it to the
    record unconditionally.

    #86865 added the bookend to trace compression rotations, so it logs
    identities and a coarse ``status`` and deliberately logs no content.
    #89117 is what the missing cause costs: a report consisting of two lines
    reading ``status=error error_retained=True duration=0.9s`` with no way to
    tell a provider 4xx from a budget wall from a crashed finalizer. The
    returned-error path -- the one a 0.9 s failure almost always takes --
    emits no other log line at all; only the exception path prints to stderr,
    which is why the quiet failures are the ones that get filed.

    Content discipline follows #86865's, and it takes two separate steps
    because it is two separate contracts. ``redact_sensitive_text`` removes
    credentials, which are pattern-shaped. It does nothing about a 4xx body
    that quotes the request back, because ordinary private prose is not
    pattern-shaped -- so ``_strip_prompt_echo`` removes that separately, using
    the submitted ``prompt`` itself as the thing to look for. The invariant the
    two of them keep is: this record may gain failure classification and
    provider detail, and may not newly persist the user's own content.

    ``prompt`` is optional so the helper stays callable from a path that has no
    prompt in scope, but the turn paths always pass it; without it, only the
    secret contract is enforced.
    """
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
        # A redactor that cannot run must not be able to leak the raw
        # message into the log by failing open.
        message = "<unredactable>"
    message = " ".join(message.split())
    # After the collapse, so both sides are compared in the same shape, and
    # before the truncation, so a quote that starts inside the kept prefix
    # cannot survive by being cut mid-run.
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
