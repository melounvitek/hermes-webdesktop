"""Tool lifecycle callbacks (tool.start/complete/progress events), verbose-text capping/redaction, todo-state projection.

Bodies are rebound onto server.py's globals at install time (see
method_ctx.bind_module), so they reference server.py globals bare.
"""

from __future__ import annotations


from .method_ctx import HandlerRegistry, bind_module

_registry = HandlerRegistry()


# waste AND feeds the Ink render-tree blowup that silently OOM-killed the TUI
# parent (#34095). Cap here to match the render budget (a hair more, so the
# "[omitted …]" label is still informative when output is genuinely large).
# Full output stays in the agent context and the SQLite session, untouched.
_TUI_VERBOSE_TEXT_MAX_CHARS = 1_000
_TUI_VERBOSE_TEXT_MAX_LINES = 16


def _cap_tui_verbose_text(text: str) -> str:
    if (
        len(text) <= _TUI_VERBOSE_TEXT_MAX_CHARS
        and text.count("\n") < _TUI_VERBOSE_TEXT_MAX_LINES
    ):
        return text

    idx = len(text)
    start = 0
    for _ in range(_TUI_VERBOSE_TEXT_MAX_LINES):
        idx = text.rfind("\n", 0, idx)
        if idx < 0:
            start = 0
            break
        start = idx + 1

    line_start = start
    start = max(line_start, len(text) - _TUI_VERBOSE_TEXT_MAX_CHARS)
    if start > line_start:
        next_break = text.find("\n", start)
        if 0 <= next_break < len(text) - 1:
            start = next_break + 1

    tail = text[start:].lstrip()
    omitted_chars = max(0, len(text) - len(tail))
    omitted_lines = text[:start].count("\n")
    if omitted_lines:
        label = (
            "[showing verbose tail; omitted "
            f"{omitted_lines} lines / {omitted_chars} chars]\n"
        )
    else:
        label = f"[showing verbose tail; omitted {omitted_chars} chars]\n"
    return f"{label}{tail}"


def _redact_tui_verbose_text(text: str) -> str:
    try:
        from agent.redact import redact_sensitive_text

        redacted = redact_sensitive_text(str(text), force=True)
    except Exception:
        return ""
    return _cap_tui_verbose_text(redacted)


def _tool_args_text(args: dict) -> str:
    try:
        raw = json.dumps(args or {}, indent=2, ensure_ascii=False, default=str)
    except Exception:
        raw = str(args or {})
    return _redact_tui_verbose_text(raw)


def _tool_result_text(result: object) -> str:
    try:
        from agent.tool_dispatch_helpers import _multimodal_text_summary

        raw = _multimodal_text_summary(result)
    except Exception:
        raw = str(result)
    return _redact_tui_verbose_text(raw)


def _fmt_tool_duration(seconds: float | None) -> str:
    if seconds is None:
        return ""
    if seconds < 10:
        return f"{seconds:.1f}s"
    if seconds < 60:
        return f"{round(seconds)}s"
    mins, secs = divmod(int(round(seconds)), 60)
    return f"{mins}m {secs}s" if secs else f"{mins}m"


def _count_list(obj: object, *path: str) -> int | None:
    cur = obj
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return len(cur) if isinstance(cur, list) else None


def _tool_summary(name: str, result: str, duration_s: float | None) -> str | None:
    try:
        data = json.loads(result)
    except Exception:
        data = None

    dur = _fmt_tool_duration(duration_s)
    suffix = f" in {dur}" if dur else ""
    text = None

    if name == "web_search" and isinstance(data, dict):
        n = _count_list(data, "data", "web")
        if n is not None:
            text = f"Did {n} {'search' if n == 1 else 'searches'}"

    elif name == "web_extract" and isinstance(data, dict):
        n = _count_list(data, "results") or _count_list(data, "data", "results")
        if n is not None:
            text = f"Extracted {n} {'page' if n == 1 else 'pages'}"

    if isinstance(data, dict) and data.get("fallback_warning"):
        warning = str(data.get("fallback_warning") or "").strip()
        if warning:
            return f"{warning}{suffix}"

    return f"{text}{suffix}" if text else None


def _normalize_todo_state(value: object) -> dict | None:
    """Return a client-safe full todo snapshot or ``None`` when malformed."""
    if not isinstance(value, dict) or not isinstance(value.get("todos"), list):
        return None
    try:
        revision = max(0, int(value.get("revision") or 0))
    except (TypeError, ValueError):
        return None
    todos = list(value["todos"])
    # Unused TodoStore snapshot() is {todos: [], revision: 0}. Attaching
    # that on resume stamps a client watermark and blocks unversioned
    # tool.start merges. An empty list at revision >= 1 is a real clear.
    if not todos and revision == 0:
        return None
    return {"todos": todos, "revision": revision}


def _session_todo_state(session: dict) -> dict | None:
    """Return the newest live/cached todo snapshot for a runtime session."""
    cached = _normalize_todo_state(session.get("todo_state"))
    live = None
    agent = session.get("agent")
    store = getattr(agent, "_todo_store", None)
    snapshot = getattr(store, "snapshot", None)
    if callable(snapshot):
        try:
            live = _normalize_todo_state(snapshot())
        except Exception:
            logger.debug("failed to read live todo state", exc_info=True)

    if live is not None and (
        cached is None or live["revision"] >= cached["revision"]
    ):
        cached = live
    if cached is not None:
        session["todo_state"] = cached
    return cached


def _attach_todo_state(payload: dict, session: dict) -> dict:
    """Attach the authoritative todo snapshot to a session response."""
    state = _session_todo_state(session)
    if state is not None:
        payload["todo_state"] = state
    return payload


def _todo_state_from_history(history) -> dict | None:
    """Derive the latest todo snapshot from an already-loaded transcript.

    Used by resume paths that answer before an AIAgent (and its live
    TodoStore) exists. The canonical todo tool results already persist in
    conversation history as ordinary tool messages, so the latest one paired
    with an assistant ``todo`` tool call IS the durable snapshot — no side
    table and no extra transcript read (each resume path passes the history
    it already loaded).
    """
    if not isinstance(history, list) or not history:
        return None
    try:
        from tools.todo_tool import MAX_TODO_RESULT_CHARS

        todo_call_ids: set[str] = set()
        for msg in history:
            if not isinstance(msg, dict):
                continue
            for call in msg.get("tool_calls") or []:
                if (call.get("function") or {}).get("name") in ("todo_list", "todo"):
                    cid = call.get("id")
                    if cid:
                        todo_call_ids.add(cid)
        if not todo_call_ids:
            return None
        for msg in reversed(history):
            if not isinstance(msg, dict) or msg.get("role") != "tool":
                continue
            if msg.get("tool_call_id") not in todo_call_ids:
                continue
            content = msg.get("content", "")
            if (
                not isinstance(content, str)
                or len(content) > MAX_TODO_RESULT_CHARS
                or '"todos"' not in content
            ):
                continue
            try:
                return _normalize_todo_state(json.loads(content))
            except Exception:
                continue
        return None
    except Exception:
        logger.debug("failed to derive todo state from history", exc_info=True)
        return None


def _on_tool_start(sid: str, tool_call_id: str, name: str, args: dict):
    session = _sessions.get(sid)
    if session is not None:
        try:
            from agent.display import capture_local_edit_snapshot

            snapshot = capture_local_edit_snapshot(name, args)
            if snapshot is not None:
                session.setdefault("edit_snapshots", {})[tool_call_id] = snapshot
        except Exception:
            pass
        session.setdefault("tool_started_at", {})[tool_call_id] = time.time()
    if _tool_progress_enabled(sid) or _tool_lifecycle_required_for_ui(name):
        payload: dict[str, object] = {
            "tool_id": tool_call_id,
            "name": name,
            "context": _tool_ctx(name, args),
        }
        # The desktop renders the expanded tool row (the `$` transcript) from
        # the args of the part, and `context` is an 80-char display preview.
        # tool.complete already ships full args to every client. When
        # tool.start ships them too, the expanded row is complete while the
        # tool runs, at the cost of one duplicate transient payload per call.
        if args:
            payload["args"] = args
        if _session_verbose(sid):
            args_text = _tool_args_text(args)
            if args_text:
                payload["args_text"] = args_text
        # tool.complete is the source of truth for todos (full list from the
        # tool result). args.todos here may be a partial merge update.
        _emit("tool.start", sid, payload)


def _on_tool_complete(sid: str, tool_call_id: str, name: str, args: dict, result: str):
    payload = {"tool_id": tool_call_id, "name": name, "args": args}
    session = _sessions.get(sid)
    snapshot = None
    started_at = None
    if session is not None:
        snapshot = session.setdefault("edit_snapshots", {}).pop(tool_call_id, None)
        started_at = session.setdefault("tool_started_at", {}).pop(tool_call_id, None)
    duration_s = time.time() - started_at if started_at else None
    if duration_s is not None:
        payload["duration_s"] = duration_s
    try:
        payload["result"] = json.loads(result)
    except Exception:
        payload["result"] = result
    summary = _tool_summary(name, result, duration_s)
    if summary:
        payload["summary"] = summary
    if _session_verbose(sid):
        result_text = _tool_result_text(result)
        if result_text:
            payload["result_text"] = result_text
    todo_state = None
    if name in ("todo_list", "todo"):  # legacy alias: pre-rename replays
        todo_state = _normalize_todo_state(payload.get("result"))
        if todo_state is not None:
            payload.update(todo_state)
            if session is not None:
                cached = _normalize_todo_state(session.get("todo_state"))
                if cached is None or todo_state["revision"] >= cached["revision"]:
                    session["todo_state"] = todo_state
    try:
        from agent.display import render_edit_diff_with_delta

        rendered: list[str] = []
        if render_edit_diff_with_delta(
            name,
            result,
            function_args=args,
            snapshot=snapshot,
            print_fn=rendered.append,
        ):
            payload["inline_diff"] = "\n".join(rendered)
    except Exception:
        pass
    if (
        _tool_progress_enabled(sid)
        or payload.get("inline_diff")
        or _tool_lifecycle_required_for_ui(name)
        or name in ("todo_list", "todo")
    ):
        _emit("tool.complete", sid, payload)
    # Task state is application data, not optional tool-progress chrome. A
    # dedicated full-snapshot event lets every client reconcile immediately
    # without interpreting provider text or partial merge arguments.
    if todo_state is not None:
        _emit("todo.updated", sid, todo_state)


def _on_tool_progress(
    sid: str,
    event_type: str,
    name: str | None = None,
    preview: str | None = None,
    _args: dict | None = None,
    **_kwargs,
):
    if not _tool_progress_enabled(sid):
        return
    if event_type == "tool.started" and name:
        # `_on_tool_start` already emits the authoritative `tool.start` with
        # the stable tool id and args. Emitting another id-less progress row
        # here makes the desktop live view diverge from hydrated history.
        return
    if event_type == "tool.output_risk" and name:
        metadata = _kwargs.get("risk_metadata")
        if not isinstance(metadata, dict):
            return
        payload: dict[str, object] = {
            "tool_id": str(_kwargs.get("tool_call_id") or ""),
            "name": str(name),
            "risk": str(metadata.get("risk") or "low"),
            "findings": [str(item) for item in metadata.get("findings", [])],
            "redacted": bool(metadata.get("redacted", False)),
        }
        _emit("tool.output_risk", sid, payload)
        return
    if event_type == "reasoning.available" and preview:
        payload: dict[str, object] = {"text": str(preview)}
        if _session_verbose(sid):
            payload["verbose"] = True
        _emit("reasoning.available", sid, payload)
        return
    if event_type == "moa.reference" and name:
        # MoA reference-model output — relay as a labelled block the Ink/desktop
        # client renders before the aggregator's response (like a thinking
        # block, tagged with the source model). `name` is the slot label,
        # `preview` is the reference text.
        ref_payload: dict[str, object] = {
            "label": str(name),
            "text": str(preview or ""),
        }
        if _kwargs.get("moa_index") is not None:
            ref_payload["index"] = _kwargs.get("moa_index")
        if _kwargs.get("moa_count") is not None:
            ref_payload["count"] = _kwargs.get("moa_count")
        _emit("moa.reference", sid, ref_payload)
        return
    if event_type == "moa.aggregating":
        _emit("moa.aggregating", sid, {"aggregator": str(name or "")})
        return
    if event_type == "moa.progress":
        # Per-reference completion — drives the status-bar progress indicator
        # (`MOA: 2/3 refs done`) requested in issue #59546. Only emitted when
        # both counters are present so the client can render deterministically.
        refs_done = _kwargs.get("moa_refs_done")
        refs_total = _kwargs.get("moa_refs_total")
        if refs_done is None or refs_total is None:
            return
        _emit(
            "moa.progress",
            sid,
            {
                "label": str(name or ""),
                "refs_done": int(refs_done),
                "refs_total": int(refs_total),
            },
        )
        return
    if event_type == "moa.phase":
        # Phase transition — currently only ``phase="aggregator"`` fires once
        # the fan-out completes and the aggregator is about to act. Tells the
        # client which phase of the MoA pipeline is currently running so it
        # can swap status-bar copy accordingly.
        phase = _kwargs.get("moa_phase")
        if not phase:
            return
        phase_payload: dict[str, object] = {"phase": str(phase)}
        refs_done = _kwargs.get("moa_refs_done")
        refs_total = _kwargs.get("moa_refs_total")
        if refs_done is not None:
            phase_payload["refs_done"] = int(refs_done)
        if refs_total is not None:
            phase_payload["refs_total"] = int(refs_total)
        if name:
            phase_payload["aggregator"] = str(name)
        _emit("moa.phase", sid, phase_payload)
        return
    if event_type.startswith("subagent."):
        payload = {
            "goal": str(_kwargs.get("goal") or ""),
            "task_count": int(_kwargs.get("task_count") or 1),
            "task_index": int(_kwargs.get("task_index") or 0),
        }
        # Identity fields for the TUI spawn tree.  All optional — older
        # emitters that omit them fall back to flat rendering client-side.
        if _kwargs.get("subagent_id"):
            payload["subagent_id"] = str(_kwargs["subagent_id"])
        if _kwargs.get("parent_id"):
            payload["parent_id"] = str(_kwargs["parent_id"])
        if _kwargs.get("child_session_id"):
            payload["child_session_id"] = str(_kwargs["child_session_id"])
        if _kwargs.get("delegation_id"):
            payload["delegation_id"] = str(_kwargs["delegation_id"])
        if _kwargs.get("depth") is not None:
            payload["depth"] = int(_kwargs["depth"])
        if _kwargs.get("model"):
            payload["model"] = str(_kwargs["model"])
        if _kwargs.get("tool_count") is not None:
            payload["tool_count"] = int(_kwargs["tool_count"])
        if _kwargs.get("toolsets"):
            payload["toolsets"] = [str(t) for t in _kwargs["toolsets"]]
        # Per-branch rollups emitted on subagent.complete (features 1+2+4).
        for int_key in (
            "input_tokens",
            "output_tokens",
            "reasoning_tokens",
            "api_calls",
        ):
            val = _kwargs.get(int_key)
            if val is not None:
                try:
                    payload[int_key] = int(val)
                except (TypeError, ValueError):
                    pass
        if _kwargs.get("files_read"):
            payload["files_read"] = [str(p) for p in _kwargs["files_read"]]
        if _kwargs.get("files_written"):
            payload["files_written"] = [str(p) for p in _kwargs["files_written"]]
        if _kwargs.get("output_tail"):
            payload["output_tail"] = list(_kwargs["output_tail"])  # list of dicts
        if name:
            payload["tool_name"] = str(name)
        if preview:
            payload["text"] = str(preview)
        if _kwargs.get("status"):
            payload["status"] = str(_kwargs["status"])
        if _kwargs.get("summary"):
            payload["summary"] = str(_kwargs["summary"])
        if _kwargs.get("duration_seconds") is not None:
            payload["duration_seconds"] = float(_kwargs["duration_seconds"])
        if preview and event_type == "subagent.tool":
            payload["tool_preview"] = str(preview)
            payload["text"] = str(preview)
        # subagent.text is the child's per-token reply, relayed solely to feed a
        # watch window's live mirror. It is meaningless on the parent session
        # (which shows the child via the spawn tree, not its reply body), so
        # skip the parent emit — sending hundreds of ignored token frames there
        # is wasted traffic and a trap for any future parent-side subagent
        # catch-all. The mirror keys off the child sid and is unaffected.
        if event_type != "subagent.text":
            _emit(event_type, sid, payload)
        _mirror_subagent_to_child(event_type, payload)


def register(server) -> None:
    """Publish this module's helpers + handlers onto ``server``, rebound to its globals."""
    bind_module(globals(), server, skip=("_",))
