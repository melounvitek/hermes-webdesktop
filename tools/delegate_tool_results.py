"""Subagent result post-processing: summary budget/spill, tool-trace summaries, lifecycle hooks and cost rollup.

Split out of ``tools/delegate_tool.py``, which re-imports every name (patch targets stay valid).
"""

from __future__ import annotations

import logging
import json
import threading
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit, urlunsplit

# Log-record parity with the origin module.
logger = logging.getLogger("tools.delegate_tool")

def _extract_output_tail(
    result: Dict[str, Any], *, max_entries: int = 12, max_chars: int = 8000,
) -> List[Dict[str, Any]]:
    """Pull the last N tool-call results from a child's conversation.

    Powers the overlay's "Output" section — the cc-swarm-parity feature.
    We reuse the same messages list the trajectory saver walks, taking
    only the tail to keep event payloads small.  Each entry is
    ``{tool, preview, is_error}``.
    """
    messages = result.get("messages") if isinstance(result, dict) else None
    if not isinstance(messages, list):
        return []

    # Walk in reverse to build a tail; stop when we have enough.
    tail: List[Dict[str, Any]] = []
    pending_call_by_id: Dict[str, str] = {}

    # First pass (forward): build tool_call_id -> tool_name map
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "assistant":
            for tc in msg.get("tool_calls") or []:
                tc_id = tc.get("id")
                fn = tc.get("function") or {}
                if tc_id:
                    pending_call_by_id[tc_id] = str(fn.get("name") or "tool")

    # Second pass (reverse): pick tool results, newest first
    for msg in reversed(messages):
        if len(tail) >= max_entries:
            break
        if not isinstance(msg, dict) or msg.get("role") != "tool":
            continue
        # Flatten content-block lists/dicts to text so the overlay shows real
        # output (not a "[{'type': 'text'...}]" blob) and error detection can
        # see markers buried inside content blocks. Crude str() here would
        # mislabel a block-wrapped "Error: ..." result as is_error=False.
        content = _stringify_tool_content(msg.get("content") or "")
        is_error = _looks_like_error_output(content)
        tool_name = pending_call_by_id.get(msg.get("tool_call_id") or "", "tool")
        # Preserve line structure so the overlay's wrapped scroll region can
        # show real output rather than a whitespace-collapsed blob. We still
        # cap the payload size to keep events bounded.
        preview = content[:max_chars]
        tail.append({"tool": tool_name, "preview": preview, "is_error": is_error})

    tail.reverse()  # restore chronological order for display
    return tail

def _stringify_tool_content(content: Any) -> str:
    """Return a stable text representation for tool-result content.

    Most providers store tool results as strings, but some OpenAI-compatible
    paths can return content-block lists. Delegate observability must never
    crash while summarising a child run just because the transport used blocks.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            item["text"] if isinstance(item, dict) and isinstance(item.get("text"), str)
            else json.dumps(item, ensure_ascii=False, default=str) if isinstance(item, dict)
            else str(item)
            for item in content
        )
    if isinstance(content, dict):
        return json.dumps(content, ensure_ascii=False, default=str)
    return str(content)

_TOOL_INPUT_TARGET_KEYS = frozenset({
    "cwd",
    "destination_path",
    "directory",
    "dst",
    "endpoint",
    "file_path",
    "new_path",
    "old_path",
    "path",
    "source_path",
    "src",
    "target_path",
    "url",
    "urls",
})

_TOOL_INPUT_URL_KEYS = frozenset({"endpoint", "url", "urls"})

def _sanitize_tool_target(key: str, value: Any) -> Any:
    """Keep bounded side-effect targets while dropping URL secrets."""
    if isinstance(value, list):
        cleaned = [item for item in (_sanitize_tool_target(key, item) for item in value[:16]) if item is not None]
        return cleaned or None
    if not isinstance(value, str) or not value:
        return None
    bounded = value[:1024]
    if key in _TOOL_INPUT_URL_KEYS:
        try:
            parsed = urlsplit(bounded)
            if parsed.scheme and parsed.netloc:
                hostname = parsed.hostname
                if not hostname:
                    return None
                # ``SplitResult.netloc`` includes ``user:password@``. Rebuild
                # the authority from parsed host/port so hook-visible history
                # cannot carry URL credentials. Bracket IPv6 literals before
                # appending a validated port.
                host = f"[{hostname}]" if ":" in hostname else hostname
                port = parsed.port
                netloc = f"{host}:{port}" if port is not None else host
                return urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))
        except ValueError:
            return None
    return bounded

def _empty_input_summary() -> Dict[str, Any]:
    return {"argument_keys": [], "targets": {}}

def _sanitize_targets(mapping: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only known side-effect target keys, each sanitized (URL secrets dropped)."""
    targets: Dict[str, Any] = {}
    for raw_key, value in mapping.items():
        key = str(raw_key).lower()
        if key in _TOOL_INPUT_TARGET_KEYS:
            cleaned = _sanitize_tool_target(key, value)
            if cleaned is not None:
                targets[key] = cleaned
    return targets

def _summarize_tool_arguments(arguments: Any) -> Dict[str, Any]:
    """Summarize argument names and side-effect targets without raw payloads."""
    if not isinstance(arguments, str):
        return _empty_input_summary()
    try:
        parsed = json.loads(arguments)
    except (TypeError, ValueError):
        return _empty_input_summary()
    if not isinstance(parsed, dict):
        return _empty_input_summary()
    keys = sorted(str(key)[:128] for key in parsed)[:64]
    return {"argument_keys": keys, "targets": _sanitize_targets(parsed)}

def _sanitize_tool_input_summary(summary: Any) -> Dict[str, Any]:
    """Re-sanitize a stored input summary before handing it to lifecycle hooks."""
    if not isinstance(summary, dict):
        return _empty_input_summary()
    keys = summary.get("argument_keys")
    safe_keys = [str(key)[:128] for key in keys[:64]] if isinstance(keys, list) else []
    targets = summary.get("targets")
    safe_targets = _sanitize_targets(targets) if isinstance(targets, dict) else {}
    return {"argument_keys": safe_keys, "targets": safe_targets}

def _subagent_stop_tool_call_history(tool_trace: Any) -> List[Dict[str, Any]]:
    """Build a detached, metadata-only tool history for lifecycle hooks."""
    if not isinstance(tool_trace, list):
        return []

    history: List[Dict[str, Any]] = []
    for item in tool_trace:
        if not isinstance(item, dict):
            continue
        tool_name = str(item.get("tool") or "unknown")[:256]
        status = str(item.get("status") or "unknown").lower()
        if status not in {"ok", "error"}:
            status = "unknown"

        def _byte_count(key: str) -> int:
            value = item.get(key, 0)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                return 0
            return max(0, int(value))

        history.append({
            "tool_name": tool_name,
            "tool_input": _sanitize_tool_input_summary(item.get("input_summary")),
            "input_bytes": _byte_count("args_bytes"),
            "output_bytes": _byte_count("result_bytes"),
            "status": status,
        })
    return history

def _looks_like_error_output(content: Any) -> bool:
    """Conservative stderr/error detector for tool-result previews.

    The old heuristic flagged any preview containing the substring "error",
    which painted perfectly normal terminal/json output red.  We now only
    mark output as an error when there is stronger evidence:
      - structured JSON with an ``error`` key
      - structured JSON with ``status`` of error/failed
      - first line starts with a classic error marker
    """
    content = _stringify_tool_content(content)
    if not content:
        return False

    head = content.lstrip()
    if head.startswith("{") or head.startswith("["):
        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                if parsed.get("error"):
                    return True
                status = str(parsed.get("status") or "").strip().lower()
                if status in {"error", "failed", "failure", "timeout"}:
                    return True
        except Exception:
            pass

    first = content.splitlines()[0].strip().lower() if content.splitlines() else ""
    return first.startswith(("error:", "failed:", "traceback ", "exception:"))

# Hard per-summary character ceiling layered on top of the dynamic
# headroom budget (see _apply_summary_budget). Belt-and-suspenders for
# models that ignore the "be concise" instruction. 0 disables the ceiling.
DEFAULT_MAX_SUMMARY_CHARS = 24000

# Fraction of the parent's *remaining* context headroom that the whole batch
# of subagent summaries is allowed to consume. The per-summary budget is this
# slice divided across the batch, so N children can't collectively blow the
# parent's window (the compression/429 death-spiral in issue/PR #9126).
_SUMMARY_HEADROOM_FRACTION = 0.5

# Floor so a single summary always gets a usable slice even when the parent is
# already nearly full — below this we'd be truncating to noise.
_MIN_SUMMARY_CHARS = 2000

def _spill_summary_to_file(task_index: int, summary: str) -> Optional[str]:
    """Write a subagent's full summary to the delegation cache and return path.

    Mirrors web_extract's ``_store_full_text``: the file lands in
    ``cache/delegation`` which is mounted read-only into remote backends
    (Docker/Modal/SSH) via ``credential_files._CACHE_DIRS``, so the parent's
    terminal/``read_file`` tools can page through the complete text on any
    backend. Returns the absolute path, or None on failure (best-effort:
    the trimmed head+tail is still returned to the parent regardless).
    """
    try:
        from hermes_constants import get_hermes_dir
        import datetime as _dt
        cache_dir = get_hermes_dir("cache/delegation", "delegation_cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = cache_dir / f"subagent-summary-{task_index}-{ts}.txt"
        from tools.spill_safety import write_text_exclusive
        # Exclusive symlink-refusing create; not private because
        # cache/delegation is bind-mounted read-only into remote backends
        # whose container UID must be able to read it.
        write_text_exclusive(path, summary, private=False)
        return str(path)
    except Exception as exc:
        logger.debug("Failed to spill subagent summary to file: %s", exc)
        return None

def _trim_summary_with_footer(summary: str, cap: int, task_index: int) -> tuple[str, Optional[str]]:
    """Return (model_text, spill_path) for one over-budget summary.

    Mirrors web_extract's ``_truncate_with_footer``: keep a head+tail window
    (~75% head / ~25% tail, snapped to line boundaries) so the subagent's
    opening AND its closing (outcomes / files-changed / issues, which live at
    the end) both survive, spill the full text to disk, and append a footer
    telling the parent exactly how much it's seeing and the precise
    ``read_file offset=`` to page into the omitted middle. Deterministic.
    """
    original_len = len(summary)
    head_budget = int(cap * 0.75)
    tail_budget = cap - head_budget

    head = summary[:head_budget]
    tail = summary[-tail_budget:]
    # Snap the head cut back to the last newline so we don't slice mid-line.
    nl = head.rfind("\n")
    if nl > head_budget * 0.5:
        head = head[:nl]
    # Snap the tail cut forward to the next newline for the same reason.
    nl = tail.find("\n")
    if 0 <= nl < tail_budget * 0.5:
        tail = tail[nl + 1:]

    spill_path = _spill_summary_to_file(task_index, summary)

    footer_lines = [
        "", "─" * 8 + " [SUMMARY TRUNCATED] " + "─" * 8,
        f"Showing {len(head):,} chars (head) + {len(tail):,} chars (tail) "
        f"of {original_len:,} total — trimmed to protect the parent's context window.",
    ]
    if spill_path:
        # read_file is 1-indexed; +2 moves past the last head line shown.
        middle_start_line = head.count("\n") + 2
        footer_lines.append(f"Full subagent output saved to: {spill_path}")
        footer_lines.append(
            f'To read the omitted middle: read_file path="{spill_path}" '
            f"offset={middle_start_line} limit=200  (the file is the complete "
            f"summary; raise/lower offset to page through it)."
        )
    else:
        footer_lines.append("Full output could not be stored to disk; the head+tail above is all that was preserved.")
    footer_lines.append("─" * 37)

    model_text = head + "\n\n[... middle omitted — see footer ...]\n\n" + tail + "\n".join(footer_lines)
    return model_text, spill_path

def _parent_summary_char_budget(parent_agent, n_summaries: int) -> Optional[int]:
    """Per-summary char budget from the parent's *remaining* context headroom
    (context length minus prompt tokens minus the compressor's output reserve),
    a fraction of it split across the batch at ~4 chars/token. Guards against N
    summaries landing at once, not one large summary. None when the parent's
    context state is unknown (no compressor / token count) — caller then uses
    the static ceiling only."""
    try:
        compressor = getattr(parent_agent, "context_compressor", None)
        context_length = getattr(compressor, "context_length", None)
        if not isinstance(context_length, int) or context_length <= 0:
            return None

        used_tokens = getattr(parent_agent, "session_prompt_tokens", 0)
        if not isinstance(used_tokens, (int, float)) or used_tokens < 0:
            used_tokens = 0

        # Reserve the compressor's output budget so we measure INPUT headroom.
        reserved = getattr(compressor, "max_tokens", 0) or 0
        headroom_tokens = context_length - int(used_tokens) - int(reserved)
        if headroom_tokens <= 0:
            # Parent is already over budget — give each summary only the floor.
            return _MIN_SUMMARY_CHARS

        batch_token_budget = int(headroom_tokens * _SUMMARY_HEADROOM_FRACTION)
        per_summary_tokens = batch_token_budget // max(1, n_summaries)
        per_summary_chars = per_summary_tokens * 4  # ~4 chars/token
        return max(_MIN_SUMMARY_CHARS, per_summary_chars)
    except Exception:
        logger.debug("Summary budget computation failed", exc_info=True)
        return None

def _apply_summary_budget(results: List[Dict[str, Any]], parent_agent) -> None:
    """Trim subagent summaries in-place so a batch can't overflow the parent's
    context window; full text is spilled to disk so nothing is lost.

    Per-summary cap = MIN(dynamic headroom budget: remaining parent context ÷
    batch size, static ``delegation.max_summary_chars`` ceiling; 0 = disabled).
    Over-cap summaries become a head+tail slice plus a pointer to the spill file.
    Without this, fan-out returned N full summaries verbatim, blowing the parent
    context and (on rate-limited providers) triggering a compression/429 death spiral.
    """
    from tools.delegate_tool import _load_config
    summaries = [r for r in results if isinstance(r, dict) and isinstance(r.get("summary"), str) and r["summary"]]
    if not summaries:
        return

    cfg = _load_config()
    try:
        static_ceiling = int(cfg.get("max_summary_chars", DEFAULT_MAX_SUMMARY_CHARS))
    except (TypeError, ValueError):
        static_ceiling = DEFAULT_MAX_SUMMARY_CHARS

    dynamic_budget = _parent_summary_char_budget(parent_agent, len(summaries))

    # Combine the two caps. Either can be absent/disabled.
    candidates = [c for c in (static_ceiling, dynamic_budget) if c and c > 0]
    if not candidates:
        return  # both disabled / unknown → leave summaries untouched
    cap = min(candidates)

    for entry in summaries:
        summary = entry["summary"]
        if len(summary) <= cap:
            continue
        original_len = len(summary)
        model_text, spill_path = _trim_summary_with_footer(summary, cap, entry.get("task_index", -1))
        entry["summary"] = model_text
        entry["summary_truncated"] = True
        if spill_path:
            entry["summary_full_path"] = spill_path
        logger.debug(
            "[subagent-%s] summary trimmed %d → ~%d chars (spill=%s)", entry.get("task_index", "?"), original_len, cap,
            spill_path or "none",
        )

_PARENT_FINALIZATION_LOCK_GUARD = threading.Lock()

_PARENT_FINALIZATION_FALLBACK_LOCK = threading.RLock()

_CHILD_CONSTRUCTION_LOCK = threading.RLock()

def _build_child_preserving_parent_tools(**kwargs):
    """Build a child without leaking its resolved toolset into the parent."""
    from tools.delegate_tool import _build_child_agent
    import model_tools
    with _CHILD_CONSTRUCTION_LOCK:
        parent_tool_names = list(model_tools._last_resolved_tool_names)
        try:
            child = _build_child_agent(**kwargs)
        finally:
            model_tools._last_resolved_tool_names = parent_tool_names
    child._delegate_saved_tool_names = parent_tool_names
    return child

def _parent_finalization_lock(parent_agent) -> threading.RLock:
    """Return the per-parent lock that serializes lifecycle side effects."""
    if parent_agent is None:
        return _PARENT_FINALIZATION_FALLBACK_LOCK
    lock = getattr(parent_agent, "_subagent_finalization_lock", None)
    if lock is not None:
        return lock
    with _PARENT_FINALIZATION_LOCK_GUARD:
        lock = getattr(parent_agent, "_subagent_finalization_lock", None)
        if lock is None:
            lock = threading.RLock()
            try:
                setattr(parent_agent, "_subagent_finalization_lock", lock)
            except Exception:
                return _PARENT_FINALIZATION_FALLBACK_LOCK
    return lock

def _notify_memory_manager(results, task_list, child_by_index, parent_agent) -> None:
    memory = getattr(parent_agent, "_memory_manager", None) if parent_agent else None
    if not memory:
        return
    for entry in results:
        try:
            task_index = entry.get("task_index", -1)
            in_range = isinstance(task_index, int) and 0 <= task_index < len(task_list)
            memory.on_delegation(
                task=task_list[task_index]["goal"] if in_range else "", result=entry.get("summary", "") or "",
                child_session_id=getattr(child_by_index.get(task_index), "session_id", ""),
            )
        except Exception:
            pass

def _fire_subagent_stop_hooks(results, child_by_index, parent_agent) -> float:
    """Pop the model-hidden ``_child_role`` / ``_child_cost_usd`` fields from every
    entry, fire ``subagent_stop`` per child, and return the summed child cost."""
    try:
        from hermes_cli.plugins import invoke_hook as invoke_hook
    except Exception:
        invoke_hook = None

    children_cost_total = 0.0
    for entry in results:
        child_role = entry.pop("_child_role", None)
        child_cost = entry.pop("_child_cost_usd", 0.0)
        try:
            if child_cost:
                children_cost_total += float(child_cost)
        except (TypeError, ValueError):
            pass
        if invoke_hook is None:
            continue
        try:
            child = child_by_index.get(entry.get("task_index", -1))
            invoke_hook(
                "subagent_stop", parent_session_id=getattr(parent_agent, "session_id", None),
                parent_turn_id=getattr(parent_agent, "_current_turn_id", "") or "",
                child_session_id=getattr(child, "session_id", None), child_role=child_role,
                child_summary=entry.get("summary"), child_status=entry.get("status"),
                tool_call_history=_subagent_stop_tool_call_history(entry.get("tool_trace")),
                duration_ms=int((entry.get("duration_seconds") or 0) * 1000),
            )
        except Exception:
            logger.debug("subagent_stop hook invocation failed", exc_info=True)
    return children_cost_total

def _rollup_children_cost(parent_agent, children_cost_total: float) -> None:
    """Fold the children's spend into the parent's session cost (source/status
    only set when the parent had none of its own)."""
    if children_cost_total <= 0.0:
        return
    try:
        current = float(getattr(parent_agent, "session_estimated_cost_usd", 0.0) or 0.0)
        parent_agent.session_estimated_cost_usd = current + children_cost_total
        if getattr(parent_agent, "session_cost_source", "none") in {None, "", "none"}:
            parent_agent.session_cost_source = "subagent"
        if getattr(parent_agent, "session_cost_status", "unknown") in {None, "", "unknown"}:
            parent_agent.session_cost_status = "estimated"
    except Exception:
        logger.debug("Subagent cost rollup failed", exc_info=True)

def _finalize_child_results(
    results: List[Dict[str, Any]], task_list: List[Dict[str, Any]], children: List[tuple[int, Dict[str, Any], Any]],
    parent_agent,
) -> None:
    """Apply host-owned summary, memory, hook, and cost contracts once."""
    with _parent_finalization_lock(parent_agent):
        _apply_summary_budget(results, parent_agent)
        child_by_index = {index: child for index, _task, child in children}
        _notify_memory_manager(results, task_list, child_by_index, parent_agent)
        _rollup_children_cost(parent_agent, _fire_subagent_stop_hooks(results, child_by_index, parent_agent))

def _run_child_lifecycle(task_index: int, goal: str, child=None, parent_agent=None) -> Dict[str, Any]:
    """Run one child and apply the same host lifecycle used by delegate_task."""
    from tools.delegate_tool import _run_single_child
    result = _run_single_child(task_index, goal, child, parent_agent)
    result.setdefault("task_index", task_index)
    task = {"goal": goal}
    _finalize_child_results(
        [result],
        [{"goal": ""} for _ in range(task_index)] + [task],
        [(task_index, task, child)],
        parent_agent,
    )
    return result
