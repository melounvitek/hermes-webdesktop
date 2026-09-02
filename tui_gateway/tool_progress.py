"""Tool lifecycle callbacks (tool.start/complete/progress events), verbose-text capping/redaction, todo-state projection.

Bodies are rebound onto server.py's globals at install time (see
method_ctx.bind_module), so they reference server.py globals bare.
"""

from __future__ import annotations


from .method_ctx import HandlerRegistry, bind_module

_registry = HandlerRegistry()


# Verbose tool text is capped to the Ink render budget (a hair more, so the
# "[omitted …]" label stays informative): unbounded output fed a render-tree
# blowup that OOM-killed the TUI parent. Full output stays in the agent context
# and the SQLite session, untouched.
_TUI_VERBOSE_TEXT_MAX_CHARS = 1_000
_TUI_VERBOSE_TEXT_MAX_LINES = 16

_TODO_TOOL_NAMES = ("todo_list", "todo")  # legacy alias: pre-rename replays


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
            "[showing verbose tail; omitted " f"{omitted_lines} lines / {omitted_chars} chars]\n"
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
    # Unused TodoStore snapshot() is {todos: [], revision: 0}: attaching it on
    # resume stamps a client watermark and blocks unversioned tool.start merges.
    # An empty list at revision >= 1 is a real clear.
    if not todos and revision == 0:
        return None
    return {"todos": todos, "revision": revision}


def _cache_todo_state(session: dict, state: dict | None) -> None:
    """Keep the newest snapshot on the session (revision-monotonic)."""
    if state is None:
        return
    cached = _normalize_todo_state(session.get("todo_state"))
    if cached is None or state["revision"] >= cached["revision"]:
        session["todo_state"] = state


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
    """Latest todo snapshot from an already-loaded transcript, for resume paths
    that answer before an AIAgent (and its live TodoStore) exists: the newest
    tool result paired with an assistant ``todo`` call IS the durable snapshot."""
    if not isinstance(history, list) or not history:
        return None
    try:
        from tools.todo_tool import MAX_TODO_RESULT_CHARS

        todo_call_ids: set[str] = set()
        for msg in history:
            if not isinstance(msg, dict):
                continue
            for call in msg.get("tool_calls") or []:
                if (call.get("function") or {}).get("name") in _TODO_TOOL_NAMES:
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
        # Full args here (not just the 80-char `context` preview) so the desktop's
        # expanded tool row is complete while the tool runs; tool.complete ships
        # them again. args.todos may be a partial merge — tool.complete is the
        # source of truth for todos.
        if args:
            payload["args"] = args
        if _session_verbose(sid):
            args_text = _tool_args_text(args)
            if args_text:
                payload["args_text"] = args_text
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
    if name in _TODO_TOOL_NAMES:
        todo_state = _normalize_todo_state(payload.get("result"))
        if todo_state is not None:
            payload.update(todo_state)
            if session is not None:
                _cache_todo_state(session, todo_state)
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
        or name in _TODO_TOOL_NAMES
    ):
        _emit("tool.complete", sid, payload)
    # Task state is application data, not tool-progress chrome: a dedicated
    # full-snapshot event lets every client reconcile without parsing tool args.
    if todo_state is not None:
        _emit("todo.updated", sid, todo_state)


# ── _on_tool_progress dispatch ─────────────────────────────────────────────
# Each handler takes (sid, name, preview, kw). `tool.started` is dropped on
# purpose: _on_tool_start already emits the authoritative tool.start with the
# stable id and args; an id-less duplicate row makes the desktop live view
# diverge from hydrated history.


def _progress_output_risk(sid, name, preview, kw):
    metadata = kw.get("risk_metadata")
    if not isinstance(metadata, dict):
        return
    _emit("tool.output_risk", sid, {
        "tool_id": str(kw.get("tool_call_id") or ""), "name": str(name),
        "risk": str(metadata.get("risk") or "low"),
        "findings": [str(item) for item in metadata.get("findings", [])],
        "redacted": bool(metadata.get("redacted", False)),
    })


def _progress_reasoning(sid, name, preview, kw):
    payload: dict[str, object] = {"text": str(preview)}
    if _session_verbose(sid):
        payload["verbose"] = True
    _emit("reasoning.available", sid, payload)


def _progress_moa_reference(sid, name, preview, kw):
    # MoA reference-model output, rendered as a labelled block before the
    # aggregator's response. `name` is the slot label, `preview` the text.
    ref_payload: dict[str, object] = {
        "label": str(name),
        "text": str(preview or ""),
    }
    if kw.get("moa_index") is not None:
        ref_payload["index"] = kw.get("moa_index")
    if kw.get("moa_count") is not None:
        ref_payload["count"] = kw.get("moa_count")
    _emit("moa.reference", sid, ref_payload)


def _progress_moa_aggregating(sid, name, preview, kw):
    _emit("moa.aggregating", sid, {"aggregator": str(name or "")})


def _progress_moa_progress(sid, name, preview, kw):
    # Drives the status-bar `MOA: 2/3 refs done`; both counters required so the
    # client renders deterministically.
    refs_done = kw.get("moa_refs_done")
    refs_total = kw.get("moa_refs_total")
    if refs_done is None or refs_total is None:
        return
    _emit("moa.progress", sid, {
        "label": str(name or ""), "refs_done": int(refs_done), "refs_total": int(refs_total),
    })


def _progress_moa_phase(sid, name, preview, kw):
    # Currently only phase="aggregator" fires, once fan-out completes.
    phase = kw.get("moa_phase")
    if not phase:
        return
    phase_payload: dict[str, object] = {"phase": str(phase)}
    refs_done = kw.get("moa_refs_done")
    refs_total = kw.get("moa_refs_total")
    if refs_done is not None:
        phase_payload["refs_done"] = int(refs_done)
    if refs_total is not None:
        phase_payload["refs_total"] = int(refs_total)
    if name:
        phase_payload["aggregator"] = str(name)
    _emit("moa.phase", sid, phase_payload)


# Per-branch rollups emitted on subagent.complete.
_SUBAGENT_INT_FIELDS = ("input_tokens", "output_tokens", "reasoning_tokens", "api_calls")


def _progress_subagent(sid, name, preview, kw, event_type):
    # Identity fields are all optional: older emitters omit them and the TUI
    # spawn tree falls back to flat rendering.
    payload = {
        "goal": str(kw.get("goal") or ""), "task_count": int(kw.get("task_count") or 1),
        "task_index": int(kw.get("task_index") or 0),
    }
    for key in ("subagent_id", "parent_id", "child_session_id", "delegation_id"):
        if kw.get(key):
            payload[key] = str(kw[key])
    if kw.get("depth") is not None:
        payload["depth"] = int(kw["depth"])
    if kw.get("model"):
        payload["model"] = str(kw["model"])
    if kw.get("tool_count") is not None:
        payload["tool_count"] = int(kw["tool_count"])
    if kw.get("toolsets"):
        payload["toolsets"] = [str(t) for t in kw["toolsets"]]
    for int_key in _SUBAGENT_INT_FIELDS:
        val = kw.get(int_key)
        if val is not None:
            try:
                payload[int_key] = int(val)
            except (TypeError, ValueError):
                pass
    for key in ("files_read", "files_written"):
        if kw.get(key):
            payload[key] = [str(p) for p in kw[key]]
    if kw.get("output_tail"):
        payload["output_tail"] = list(kw["output_tail"])  # list of dicts
    if name:
        payload["tool_name"] = str(name)
    if preview:
        payload["text"] = str(preview)
    if kw.get("status"):
        payload["status"] = str(kw["status"])
    if kw.get("summary"):
        payload["summary"] = str(kw["summary"])
    if kw.get("duration_seconds") is not None:
        payload["duration_seconds"] = float(kw["duration_seconds"])
    if preview and event_type == "subagent.tool":
        payload["tool_preview"] = str(preview)
        payload["text"] = str(preview)
    # subagent.text is the child's per-token reply, relayed solely to feed a
    # watch window's live mirror (keyed off the child sid); on the parent it's
    # hundreds of ignored frames, so skip that emit.
    if event_type != "subagent.text":
        _emit(event_type, sid, payload)
    _mirror_subagent_to_child(event_type, payload)


# event_type -> (handler, requires) where `requires` names the arg that must be
# truthy for the row to be emitted at all ("name" / "preview" / None).
_PROGRESS_HANDLERS = {
    "tool.output_risk": (_progress_output_risk, "name"),
    "reasoning.available": (_progress_reasoning, "preview"),
    "moa.reference": (_progress_moa_reference, "name"),
    "moa.aggregating": (_progress_moa_aggregating, None),
    "moa.progress": (_progress_moa_progress, None), "moa.phase": (_progress_moa_phase, None),
}


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
        return
    entry = _PROGRESS_HANDLERS.get(event_type)
    if entry is not None:
        handler, requires = entry
        if requires is None or {"name": name, "preview": preview}[requires]:
            handler(sid, name, preview, _kwargs)
        return
    if event_type.startswith("subagent."):
        _progress_subagent(sid, name, preview, _kwargs, event_type)


def register(server) -> None:
    """Publish this module's helpers + handlers onto ``server``, rebound to its globals."""
    bind_module(globals(), server, skip=("_",))
