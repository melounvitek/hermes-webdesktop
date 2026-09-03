#!/usr/bin/env python3
"""Session Search Tool - long-term conversation recall over the SQLite session DB.

Single-shape tool; the mode is inferred from the args: DISCOVERY (``query``;
FTS5 deduped by lineage, adaptive detail hydrates only the top result),
SCROLL (``session_id`` + ``around_message_id``; ±window around the anchor),
READ (``session_id`` alone; whole session or head/tail), BROWSE (no args).
No LLM calls — every shape returns actual DB messages.
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from hermes_state_common import _RESET_END_REASONS

# Hidden from browsing/searching: integrations (HERMES_SESSION_SOURCE=tool),
# delegate subagent runs, kanban workers — not the user's history.
_HIDDEN_SESSION_SOURCES = ("kanban", "subagent", "tool")

# Searchable but DEMOTED below interactive sessions: cron vocabulary dominates bare
# BM25 and starves out the user's own sessions ("recall blindness").
_DEMOTED_SESSION_SOURCES = ("cron",)

# FTS rows scanned before dedup-by-lineage — well above the distinct sessions a query
# returns, so interactive matches buried under cron hits survive the demotion pass.
_DISCOVER_SCAN_LIMIT = 300

# Raw FTS rows are only a plan input; the response hydrates its own window/bookends.
_DISCOVER_SEARCH_FIELDS = ("id", "session_id", "role", "snippet", "source", "model", "session_started")

# Compaction handoff summaries (agent/context_compressor.py); excluded from bookends.
_COMPACTION_PREFIXES = ("[CONTEXT COMPACTION", "[CONTEXT SUMMARY]:")

# /new, /reset, idle/daily expiry and CLI /new ("new_session") end the predecessor
# WITHOUT carrying its transcript forward — unlike compression continuations and
# live delegation children. Derived from the gateway set so the two cannot drift.
_FRESH_RESET_END_REASONS = frozenset(_RESET_END_REASONS) | {"new_session"}


def _quiet(fn, default, msg, *log_args, with_exc: bool = False):
    """``fn()``, or *default* after debug-logging *msg* (exception appended as a
    final ``%s`` arg when *with_exc*) on any exception."""
    try:
        return fn()
    except Exception as e:
        logging.debug(msg, *(log_args + (e,) if with_exc else log_args), exc_info=True)
        return default


def _loud(fn, log_msg, error_prefix, *log_args):
    """``(value, None)`` from ``fn()``, or ``(None, tool_error_json)`` after an
    error-level log — for DB calls whose failure the model must see."""
    try:
        return fn(), None
    except Exception as e:
        logging.error(log_msg, *log_args, e, exc_info=True)
        return None, tool_error(f"{error_prefix}: {e}", success=False)


def _format_timestamp(ts: Union[int, float, str, None]) -> str:
    """Unix timestamp (number / numeric string) -> readable date; ISO strings pass
    through; "unknown" for None; str(ts) if conversion fails."""
    if ts is None:
        return "unknown"
    if isinstance(ts, str) and not ts.replace(".", "").replace("-", "").isdigit():
        return ts
    try:
        return datetime.fromtimestamp(float(ts)).strftime("%B %d, %Y at %I:%M %p")
    except Exception as e:
        logging.debug("Failed to format timestamp %s: %s", ts, e, exc_info=True)
    return str(ts)


def _session_meta_block(meta: Dict[str, Any]) -> Dict[str, Any]:
    """The ``session_meta`` sub-object shared by read/scroll responses."""
    return {"when": _format_timestamp(meta.get("started_at")), "source": meta.get("source"),
            "model": meta.get("model"), "title": meta.get("title")}


def _ok(**payload) -> str:
    """Serialize a successful tool result (``success`` first, then *payload* in order)."""
    return json.dumps({"success": True, **payload}, ensure_ascii=False)


def _is_compaction_summary(content: str) -> bool:
    """Return True if *content* looks like a generated compaction handoff."""
    return bool(content) and content.lstrip().startswith(_COMPACTION_PREFIXES)


def _resolve_to_parent(db, session_id: str) -> tuple[str, bool]:
    """Walk parent_session_id to the root -> ``(root_id, has_compression_hop)``. The
    flag separates a compression-split lineage (parent summarised away) from a
    delegation lineage (child still visible to the parent). Errors -> ``(session_id, False)``."""
    visited: set[str] = set()
    cur, has_compression = session_id, False
    while cur and cur not in visited:
        visited.add(cur)
        s = _quiet(lambda: db.get_session(cur), None, "Error resolving parent for %s: %s", cur, with_exc=True)
        if not s:
            break
        has_compression = has_compression or s.get("end_reason") == "compression"
        if not s.get("parent_session_id"):
            break
        cur = s["parent_session_id"]
    return cur, has_compression


def _resolve_lineage(db, session_id: str) -> str:
    """Return only the lineage root (ignores the compression hop)."""
    return _resolve_to_parent(db, session_id)[0]


def _session_left_live_context(db, session_id: str) -> bool:
    """True when the transcript left everyone's live context: ``compression``
    (summarised into the child) or a fresh reset (child starts empty). Live delegation
    children (``end_reason is None``) and ``branched`` parents (copied verbatim into
    the branch) ARE the current context, so they stay excluded from recall."""
    s = session_id and _quiet(lambda: db.get_session(session_id), None, "get_session failed for %s", session_id)
    end_reason = (s.get("end_reason") or None) if s else None
    return end_reason == "compression" or end_reason in _FRESH_RESET_END_REASONS


def _get_message_storage_state(db, message_id) -> Optional[Dict[str, Any]]:
    """Return the owning session and visibility flags for *message_id*."""
    if not message_id:
        return None

    def _lookup():
        with db._lock:
            return db._conn.execute(
                "SELECT session_id, active, compacted FROM messages WHERE id = ?", (message_id,)).fetchone()
    row = _quiet(_lookup, None, "message storage-state lookup failed for %s", message_id)
    return dict(row) if row is not None else None


def _is_compacted_state(state: Optional[Dict[str, Any]]) -> bool:
    """Compaction archives are ``active=0, compacted=1``; rewind/undo rows are
    ``active=0, compacted=0`` and must stay hidden."""
    return state is not None and state["active"] == 0 and state["compacted"] == 1


def _is_compacted_message(db, message_id) -> bool:
    """True for a compaction-archived row — content no longer in live context, so
    it stays discoverable even on the current session. False on any error."""
    return _is_compacted_state(_get_message_storage_state(db, message_id))


def _annotate_rebuild_status(db, payload: Dict[str, Any]) -> None:
    """Note rebuild progress while the deferred FTS backfill runs so the agent can
    explain thin results instead of treating them as ground truth. Never raises."""
    status = _quiet(db.fts_rebuild_status, None, "fts_rebuild_status failed")
    if status is None:
        return
    payload["index_rebuild"] = {"percent": status["percent"], "note": (
        f"The search index is rebuilding in the background ({status['percent']}% done, "
        f"{status['indexed']:,} of {status['total']:,} messages). Results from older messages "
        f"may be incomplete until it finishes.")}


def _order_for_recall(raw_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Stable-sort so interactive sessions rank above automation; BM25 order is
    kept within each class, so a cron hit never displaces an interactive one."""
    return sorted(raw_results, key=lambda r: 1 if (r.get("source") or "") in _DEMOTED_SESSION_SOURCES else 0)


def _shape_message(m: Dict[str, Any], anchor_id: Optional[int] = None,
                   max_content_len: Optional[int] = None) -> Dict[str, Any]:
    """Slim a message row. Keeps ``content`` even when empty (tool-call-only
    assistant turns); with *max_content_len* truncates and flags it."""
    content = m.get("content")
    if isinstance(content, str) and "\x1b" in content:  # archived terminal output carries ANSI
        from tools.ansi_strip import strip_ansi
        content = strip_ansi(content)
    entry = {"id": m.get("id"), "role": m.get("role"), "content": content, "timestamp": m.get("timestamp")}
    entry.update({k: m.get(k) for k in ("tool_name", "tool_calls", "tool_call_id") if m.get(k)})
    if anchor_id is not None and m.get("id") == anchor_id:
        entry["anchor"] = True
    if max_content_len and content and len(content) > max_content_len:
        entry.update(content=content[:max_content_len] + "…", content_truncated=True, original_content_chars=len(content))
    return {k: v for k, v in entry.items() if v is not None or k == "content"}


def _session_link(session_id: str, profile: str = None) -> str:
    """The reference the agent writes for a session — same value the desktop composer
    emits, so it renders as a titled link. The profile segment is omitted when it
    can't be named confidently (a bare id still resolves, just not across profiles)."""
    name = (profile or "").strip()
    if not name:
        def _active():
            from hermes_cli.profiles import get_active_profile_name
            resolved = get_active_profile_name()
            return "" if resolved == "custom" else resolved
        name = _quiet(_active, "", "get_active_profile_name failed for session link")
    return f"@session:{name}/{session_id}" if name else f"@session:{session_id}"


def _title_match_result(db, query: str, current_lineage_root: Optional[str]) -> Optional[Dict[str, Any]]:
    """Return a discovery-shaped result when the query matches a session title."""
    title_query = query.strip().strip("`'\"")  # models often quote a remembered title
    if not title_query:
        return None
    session_id = _quiet(lambda: db.resolve_session_by_title(title_query), None,
                        "resolve_session_by_title failed for %r", title_query)
    if not session_id:
        return None
    lineage_root = _resolve_lineage(db, session_id)
    # Same-lineage title hits are in-context only while the session is live;
    # /new-reset and compression-ended parents are not.
    if current_lineage_root and lineage_root == current_lineage_root and not _session_left_live_context(db, session_id):
        return None
    session_meta = _quiet(lambda: db.get_session(lineage_root) or db.get_session(session_id), None,
                          "get_session failed for title match %s", session_id) or {}
    if session_meta.get("source") in _HIDDEN_SESSION_SOURCES:
        return None
    messages = _quiet(lambda: db.get_messages(session_id), [], "get_messages failed for title match %s", session_id)
    anchor_id = messages[0].get("id") if messages else None
    view = {} if anchor_id is None else _quiet(
        lambda: db.get_anchored_view(session_id, anchor_id, window=5, bookend=3), {},
        "get_anchored_view failed for title match %s/%s", session_id, anchor_id)
    title = session_meta.get("title") or title_query
    entry = _discovery_entry(
        lineage_root, session_id=session_id, when=_format_timestamp(session_meta.get("started_at")),
        source=session_meta.get("source", "unknown"), model=session_meta.get("model") or "unknown",
        title=title, matched_role="session_title", match_message_id=anchor_id,
        snippet=f"Session title matched: {title}",
        bookend_start=[_shape_message(m) for m in (view.get("bookend_start") or messages[:3])],
        messages=[_shape_message(m, anchor_id=anchor_id) for m in (view.get("window") or messages[:5])],
        bookend_end=[_shape_message(m) for m in (view.get("bookend_end") or messages[-3:])],
        messages_before=view.get("messages_before", 0),
        messages_after=view.get("messages_after", max(len(messages) - 5, 0)), detail="full")
    entry["_lineage_root"] = lineage_root
    return entry


def _discovery_entry(lineage_root: Optional[str], **fields) -> Dict[str, Any]:
    """One discovery result in canonical key order; ``parent_session_id`` is set
    when the hit lives in a child of its lineage root."""
    entry = {k: fields[k] for k in (
        "session_id", "when", "source", "model", "title", "matched_role", "match_message_id", "snippet",
        "bookend_start", "messages", "bookend_end", "messages_before", "messages_after", "detail")}
    if lineage_root and lineage_root != entry["session_id"]:
        entry["parent_session_id"] = lineage_root
    return entry


def _discover_payload(db, query: str, detail: str, results: list, **extra) -> str:
    payload = {"success": True, "mode": "discover", "query": query, "detail": detail,
               "results": results, "count": len(results), **extra}
    _annotate_rebuild_status(db, payload)
    return json.dumps(payload, ensure_ascii=False)


def _dedupe_by_lineage(db, raw_results, limit, seen_sessions, current_session_id, current_lineage_root) -> None:
    """Fill *seen_sessions* (lineage_root -> first surviving FTS row) up to *limit*.
    The raw owning session_id stays on the row — only it pairs validly with the FTS
    match id. Current-lineage hits are skipped UNLESS the transcript left live
    context (compression-ended, /new-reset predecessor, or an in-place compacted row
    on the SAME session); a live delegation child (end_reason=None) stays excluded."""
    for r in raw_results:
        if len(seen_sessions) >= limit:
            break
        raw_sid = r["session_id"]
        resolved_sid = _resolve_lineage(db, raw_sid)
        is_compacted_hit = _is_compacted_message(db, r.get("id"))
        in_current_lineage = bool(current_lineage_root) and resolved_sid == current_lineage_root
        if in_current_lineage and not (_session_left_live_context(db, raw_sid) or is_compacted_hit):
            continue
        if current_session_id and raw_sid == current_session_id and not is_compacted_hit:
            continue
        seen_sessions.setdefault(resolved_sid, {**r, "_lineage_root": resolved_sid})


def _bookend(view: Dict[str, Any], key: str) -> List[Dict[str, Any]]:
    return [_shape_message(m, max_content_len=1200) for m in (view.get(key) or [])
            if not _is_compaction_summary(m.get("content", ""))]


def _hydrate_hit(db, lineage_root: str, match_info: Dict[str, Any], result_detail: str) -> Optional[Dict[str, Any]]:
    """One discovery result from a surviving FTS row; None (hit dropped) if the
    anchored view can't be loaded."""
    hit_sid = match_info.get("session_id") or lineage_root
    msg_id = match_info.get("id")
    try:
        view = db.get_anchored_view(hit_sid, msg_id, window=5, bookend=3)
    except Exception as e:
        logging.warning("get_anchored_view failed for %s/%s: %s", hit_sid, msg_id, e, exc_info=True)
        return None
    session_meta = _quiet(lambda: db.get_session(lineage_root), None, "get_session failed for %s", lineage_root) or {}
    full = result_detail == "full"
    window_messages = [m for m in (view.get("window") or []) if full or m.get("id") == msg_id]
    return _discovery_entry(
        lineage_root, session_id=hit_sid,
        when=_format_timestamp(session_meta.get("started_at") or match_info.get("session_started")),
        source=session_meta.get("source") or match_info.get("source", "unknown"),
        model=session_meta.get("model") or match_info.get("model") or "unknown",
        title=session_meta.get("title") or None, matched_role=match_info.get("role"),
        match_message_id=msg_id, snippet=match_info.get("snippet") or "",
        bookend_start=_bookend(view, "bookend_start") if full else [],
        messages=[_shape_message(m, anchor_id=msg_id, max_content_len=4000) for m in window_messages],
        bookend_end=_bookend(view, "bookend_end") if full else [],
        messages_before=view.get("messages_before", 0), messages_after=view.get("messages_after", 0),
        detail=result_detail)


def _discover(db, query: str, role_filter: Optional[List[str]], limit: int, sort: Optional[str],
              detail: str, current_session_id: str = None, link_profile: str = None) -> str:
    """Discovery shape: FTS5 plus adaptive or full result hydration."""
    current_lineage_root = _resolve_lineage(db, current_session_id) if current_session_id else None
    title_result = _title_match_result(db, query, current_lineage_root)
    raw_results, err = _loud(lambda: db.search_messages(
        query=query, role_filter=role_filter or ["user", "assistant"],
        exclude_sources=list(_HIDDEN_SESSION_SOURCES), limit=_DISCOVER_SCAN_LIMIT, offset=0, sort=sort,
        fields=_DISCOVER_SEARCH_FIELDS), "FTS5 search failed: %s", "Search failed")
    if err:
        return err
    # Demote cron rows below interactive ones BEFORE dedup so a high-volume cron
    # corpus can't starve the user's own sessions out of the top `limit`.
    raw_results = _order_for_recall(raw_results)
    if not raw_results and not title_result:
        return _discover_payload(db, query, detail, [], message=(
            "No matching sessions found. FTS5 ANDs all terms by default — "
            "broaden with OR (`alpha OR beta`), exact-match with quoted "
            "phrases, exclude with NOT, or prefix-match with `deploy*`."))

    seen_sessions: Dict[str, Dict[str, Any]] = {}
    results = []
    if title_result:
        title_lineage = title_result.pop("_lineage_root", None)
        if title_lineage:
            seen_sessions[title_lineage] = {"_title_only": True}
        results.append(title_result)
    _dedupe_by_lineage(db, raw_results, limit, seen_sessions, current_session_id, current_lineage_root)
    for lineage_root, match_info in seen_sessions.items():
        if match_info.get("_title_only"):
            continue
        # Adaptive: only the top-ranked result is fully hydrated.
        entry = _hydrate_hit(db, lineage_root, match_info, "full" if detail == "full" or not results else "compact")
        if entry is not None:
            results.append(entry)
    for entry in results:
        entry["link"] = _session_link(entry["session_id"], link_profile)
    return _discover_payload(db, query, detail, results, sessions_searched=len(seen_sessions), link_hint=(
        "When referring the user to a session, write its `link` value "
        "verbatim inline mid-sentence (it renders as a titled link) — never "
        "as markdown, in backticks, on its own line, or next to the "
        "title/id/date. To read more around a compact result, scroll: "
        "session_search(session_id=..., around_message_id=match_message_id)."))


def _resolve_profile_db(profile: str):
    """Another profile's ``state.db`` opened read-only (safe on a live DB), or None
    for the current profile."""
    if profile is None or not str(profile).strip():
        return None
    from hermes_cli import profiles as profiles_mod
    from hermes_state import SessionDB
    canon = profiles_mod.normalize_profile_name(profile)
    profiles_mod.validate_profile_name(canon)
    if not profiles_mod.profile_exists(canon):
        raise ValueError(f"profile '{canon}' does not exist")
    return SessionDB(db_path=profiles_mod.get_profile_dir(canon) / "state.db", read_only=True)


def _locate_session_db(session_id: str):
    """Scan every profile's ``state.db`` for a session id -> ``(db, profile_name)`` or
    ``(None, None)``. Ids are globally unique, so the first hit is authoritative."""
    from pathlib import Path
    try:
        from hermes_cli import profiles as profiles_mod
        from hermes_state import SessionDB
    except Exception:
        return None, None
    targets = [("default", profiles_mod.get_profile_dir("default"))]
    targets += _quiet(lambda: [(info.name, info.path) for info in profiles_mod.list_profiles()],
                      [], "list_profiles failed during session locate")
    seen: set = set()
    for name, home in targets:
        db_path = Path(home) / "state.db"
        if str(db_path) in seen or not db_path.exists():
            continue
        seen.add(str(db_path))
        pdb = _quiet(lambda: SessionDB(db_path=db_path, read_only=True), None, "open %s failed", db_path)
        if pdb and _quiet(lambda: pdb.get_session(session_id), None,
                          "get_session probe failed for %s in %s", session_id, name):
            return pdb, name
        if pdb:
            pdb.close()
    return None, None


def _get_session_meta(db, session_id: str) -> dict:
    """``db.get_session`` that degrades to ``{}`` on error."""
    return _quiet(lambda: db.get_session(session_id), None,
                  "get_session failed for %s: %s", session_id, with_exc=True) or {}


def _read_session(db, session_id: str, head: int = 20, tail: int = 10, link_profile: str = None) -> str:
    """Read shape: whole session, or ``head`` + ``tail`` messages with a scroll pointer."""
    meta = _get_session_meta(db, session_id)
    if not meta:
        return tool_error(f"session_id not found: {session_id}", success=False)
    rows, err = _loud(lambda: db.get_messages(session_id), "get_messages failed for %s: %s", "failed to load session",
                      session_id)
    if err:
        return err
    shaped = [_shape_message(m) for m in rows]
    total = len(shaped)
    truncated = total > head + tail
    extra = {"message": (f"Session has {total} messages; showing first {head} + last {tail}. "
                         "Pass around_message_id (any id above) to scroll the middle.")} if truncated else {}
    return _ok(mode="read", session_id=session_id, link=_session_link(session_id, link_profile),
               session_meta=_session_meta_block(meta), message_count=total, truncated=truncated,
               messages=shaped[:head] + shaped[-tail:] if truncated else shaped, **extra)


def _list_recent_sessions(db, limit: int, current_session_id: str = None, link_profile: str = None) -> str:
    """Browse shape: metadata for the most recent sessions (no LLM, no FTS5)."""
    def _browse():
        # Never use list_sessions_rich(order_by_last_active=True) here: it walks every
        # compression chain and derives activity/previews before LIMIT, which can
        # monopolise a gateway callback for minutes on a multi-GB state.db. The
        # bounded browse query preselects an indexed candidate set and carries a
        # cooperative SQLite VM cancellation deadline. Fail closed rather than
        # silently falling back to the whole-database query shape.
        bounded_list = getattr(db, "list_recent_sessions_bounded", None)
        if bounded_list is None:
            raise RuntimeError("session database does not support bounded recent-session browse")
        sessions = bounded_list(
            limit=limit + 15,  # extra so we can skip current / compression roots
            exclude_sources=list(_HIDDEN_SESSION_SOURCES), timeout_seconds=3.0)
        current_root, has_compression_hop = (
            _resolve_to_parent(db, current_session_id) if current_session_id else (None, False))
        results = []
        for s in sessions:
            sid = s.get("id", "")
            # Compression continuation: the root was summarised into the live child, so
            # hide it. /new-reset children carry no transcript — keep that root browsable.
            if sid == current_session_id or (has_compression_hop and current_root and sid == current_root):
                continue
            results.append({
                "session_id": sid, "link": _session_link(sid, link_profile), "title": s.get("title") or None,
                **{k: s.get(k, "") for k in ("source", "started_at", "last_active")},
                "message_count": s.get("message_count", 0), "preview": s.get("preview", "")})
            if len(results) >= limit:
                break
        return _ok(mode="browse", results=results, count=len(results), message=(
            f"Showing {len(results)} most recent sessions. Pass a query= to search, "
            "or session_id+around_message_id to scroll."))

    out, err = _loud(_browse, "Error listing recent sessions: %s", "Failed to list recent sessions")
    return err or out


def _clamp_int(value, default: int, lo: int, hi: int) -> int:
    try:
        value = int(value)
    except (TypeError, ValueError):
        value = default
    return max(lo, min(value, hi))


def _anchor_in_live_context(db, anchor_state, anchor_session_id: str, current_session_id: str) -> bool:
    """True when the scroll anchor is still in the caller's active context (reject).
    Same-lineage history that LEFT live context (compacted rows, compression-ended
    parents, /new-reset predecessors) is allowed, so scroll never rejects a result
    discovery just returned."""
    if not _same_lineage(db, anchor_session_id, current_session_id) or _is_compacted_state(anchor_state):
        return False
    # Rewind/undo rows (active=0, compacted!=1) never count as out-of-context history.
    inactive_non_compacted = anchor_state is not None and anchor_state["active"] == 0 and anchor_state["compacted"] != 1
    return inactive_non_compacted or not _session_left_live_context(db, anchor_session_id)


def _same_lineage(db, a: str, b: str) -> bool:
    a_root, b_root = _resolve_lineage(db, a), _resolve_lineage(db, b)
    return bool(a_root and b_root and a_root == b_root)


def _rebind_to_owner(db, session_id: str, owning: str, around_message_id: int, window: int):
    """Lineage rebind when the caller paired a parent session_id with a message id
    living in a descendant. ``(view, warning)`` from the owner, or ``(None, None)``."""
    rebind_view = _same_lineage(db, session_id, owning) and _quiet(
        lambda: db.get_messages_around(owning, around_message_id, window=window),
        None, "rebind get_messages_around failed: %s", with_exc=True)
    if not (rebind_view and rebind_view.get("window")):
        return None, None
    return rebind_view, f"around_message_id {around_message_id} lives in {owning} (child of {session_id}); rebound transparently"


def _scroll(db, session_id: str, around_message_id: int, window: int = 5,
            current_session_id: str = None) -> str:
    """Scroll shape: a window centered on an anchor (no FTS5, no bookends);
    rebinds silently if the anchor lives in a same-lineage child."""
    if not isinstance(session_id, str) or not session_id.strip():
        return tool_error("scroll requires session_id", success=False)
    session_id = session_id.strip()
    try:
        around_message_id = int(around_message_id)
    except (TypeError, ValueError):
        return tool_error("scroll requires integer around_message_id", success=False)
    window = _clamp_int(window, 5, 1, 20)
    # Locate the anchor BEFORE the current-lineage guard (see _anchor_in_live_context).
    anchor_state = _get_message_storage_state(db, around_message_id)
    owning_session_id = anchor_state.get("session_id") if anchor_state is not None else None
    if current_session_id and _anchor_in_live_context(
        db, anchor_state, owning_session_id or session_id, current_session_id):
        return tool_error("scroll rejected: anchor lives in the current session lineage (already in your active context)", success=False)
    session_meta = _get_session_meta(db, session_id)
    if not session_meta:
        return tool_error(f"session_id not found: {session_id}", success=False)
    view, err = _loud(lambda: db.get_messages_around(session_id, around_message_id, window=window),
                      "get_messages_around failed: %s", "failed to load messages")
    if err:
        return err
    messages = view.get("window") or []
    rebind_warning = None
    if not messages and owning_session_id and owning_session_id != session_id:
        rebind_view, rebind_warning = _rebind_to_owner(db, session_id, owning_session_id, around_message_id, window)
        if rebind_view is not None:
            view, messages = rebind_view, rebind_view["window"]
            session_meta = _get_session_meta(db, owning_session_id) or session_meta
            session_id = owning_session_id
    if not messages:
        return tool_error(f"around_message_id {around_message_id} not in session_id {session_id}", success=False)
    return _ok(
        mode="scroll", session_id=session_id, around_message_id=around_message_id,
        session_meta=_session_meta_block(session_meta), window=window,
        messages=[_shape_message(m, anchor_id=around_message_id) for m in messages],
        messages_before=view.get("messages_before", 0), messages_after=view.get("messages_after", 0),
        hint=("Scroll forward: re-call with around_message_id = the LAST message's "
              "id; backward: the FIRST message's id (the boundary message repeats "
              "as an orientation marker). messages_before/messages_after < window "
              "means you've hit that end of the session."),
        **({"warning": rebind_warning} if rebind_warning else {}))


def _read_with_profile_fallback(db, sid: str, profile: Optional[str]) -> str:
    """Read shape; on a miss scan every profile (the model may have dropped the
    owning profile from the link) and tag the result with where it was found."""
    result = _read_session(db, sid, link_profile=profile)
    if json.loads(result).get("success"):
        return result
    located, owner = _locate_session_db(sid)
    if located is None:
        return result
    try:
        found = json.loads(_read_session(located, sid, link_profile=owner))
    finally:
        located.close()
    if not found.get("success"):
        return result
    found["profile"] = owner
    return json.dumps(found, ensure_ascii=False)


def _dispatch(query, role_filter, limit, db, current_session_id, session_id,
              around_message_id, window, sort, profile, detail, owned_dbs) -> str:
    """Mode dispatch (see module docstring); scroll wins when an anchor is set.
    Profile DBs opened here are appended to *owned_dbs* for the caller to close."""
    # A raw `@session:<profile>/<id>` link as session_id: ids never contain "/", so
    # split on it and adopt the embedded profile only when none was passed.
    if isinstance(session_id, str) and "/" in session_id:
        emb_profile, _, emb_id = session_id.partition("/")
        if emb_id:
            session_id = emb_id
            if emb_profile and (profile is None or not str(profile).strip()):
                profile = emb_profile
    # Cross-profile: swap in the named profile's DB (read-only) for every shape;
    # current-lineage guards key off ids that won't collide, so they stay inert.
    try:
        profile_db = _resolve_profile_db(profile)
    except Exception as e:
        return tool_error(f"profile '{profile}': {e}", success=False)
    if profile_db is not None:
        db, current_session_id = profile_db, None
        owned_dbs.append(profile_db)
    if isinstance(session_id, str) and session_id.strip():
        if around_message_id is not None:
            return _scroll(db, session_id, around_message_id, window, current_session_id)
        return _read_with_profile_fallback(db, session_id.strip(), profile)
    limit = _clamp_int(limit, 3, 1, 10)
    if not query or not isinstance(query, str) or not query.strip():
        return _list_recent_sessions(db, limit, current_session_id, link_profile=profile)
    role_list = ([r.strip() for r in role_filter.split(",") if r.strip()] or None) if isinstance(role_filter, str) else None
    sort_norm = sort.strip().lower() if isinstance(sort, str) else None
    sort_norm = sort_norm if sort_norm in ("newest", "oldest") else None
    detail_norm = "full" if isinstance(detail, str) and detail.strip().lower() == "full" else "adaptive"
    return _discover(
        db=db, query=query.strip(), role_filter=role_list, limit=limit, sort=sort_norm,
        detail=detail_norm, current_session_id=current_session_id, link_profile=profile)


def session_search(query: str = "", role_filter: str = None, limit: int = 3, db=None,
                   current_session_id: str = None, session_id: str = None, around_message_id: int = None,
                   window: int = 5, sort: str = None, profile: str = None, detail: str = "adaptive") -> str:
    """Run session search, closing DBs opened here. Positional order is frozen for old callers."""
    owned_dbs: List[Any] = []
    if db is None:
        try:
            from hermes_state import get_shared_session_db
            db = get_shared_session_db()
            owned_dbs.append(db)
        except Exception:
            logging.debug("SessionDB unavailable for session_search", exc_info=True)
            from hermes_state import format_session_db_unavailable
            return tool_error(format_session_db_unavailable(), success=False)
    try:
        return _dispatch(query, role_filter, limit, db, current_session_id, session_id,
                         around_message_id, window, sort, profile, detail, owned_dbs)
    finally:
        from hermes_state import release_or_close
        for owned_db in reversed(owned_dbs):
            _quiet(lambda: release_or_close(owned_db), None, "Failed to close session_search SessionDB")


def check_session_search_requirements() -> bool:
    """Requires the SQLite state database."""
    try:
        from hermes_state import _default_db_path
        return _default_db_path().parent.exists()
    except ImportError:
        return False


SESSION_SEARCH_SCHEMA = {
    "name": "session_search",
    "description": (
        "Recall past conversations: search or read old Hermes sessions (FTS5), or "
        "scroll inside one. Four shapes, picked by args: `query` = discovery "
        "(top-N matching sessions, top result fully hydrated); `session_id` + "
        "`around_message_id` = scroll (window of messages around an anchor); "
        "`session_id` alone = read a whole session — how you resolve an "
        "`@session:<profile>/<id>` link (split on '/' into profile + id); no "
        "args = browse recent sessions. Results are actual DB messages, no LLM. "
        "Searches conversation history ONLY — when the user gave a direct "
        "source (URL, file, contact, live system), inspect that first; never "
        "conclude 'not found' from history alone. Use for questions about past "
        "conversations: 'what did we do about X', 'where did we leave Y'. When "
        "referring the user to a session, write its `link` value verbatim "
        "inline (it renders as a titled link)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Search query (discovery shape). Keywords, phrases, or boolean "
                    "expressions to find in past sessions. Omit to browse recent "
                    "sessions. Ignored when session_id + around_message_id are set "
                    "(scroll shape)."
                ),
            },
            "limit": {
                "type": "integer",
                "description": (
                    "Discovery shape only. Max sessions to return (default 3, max 10). "
                    "Bump to 5–10 when the topic likely spans several sessions and you "
                    "want to pick the right one to scroll into."
                ),
                "default": 3,
            },
            "sort": {
                "type": "string",
                "enum": ["newest", "oldest"],
                "description": (
                    "Discovery shape only. Temporal bias on top of FTS5 ranking: omit "
                    "for relevance-only (exploratory recall), 'newest' for "
                    "\"where did we leave X\", 'oldest' for \"how did X start\"."
                ),
            },
            "detail": {
                "type": "string",
                "enum": ["adaptive", "full"],
                "description": (
                    "Discovery shape only. 'adaptive' (default) fully hydrates the "
                    "top-ranked result and returns only the exact anchor message for "
                    "lower-ranked results. 'full' returns bookends and the complete "
                    "anchored window for every result."
                ),
                "default": "adaptive",
            },
            "session_id": {
                "type": "string",
                "description": (
                    "Scroll shape. Session to read inside. Use the session_id returned "
                    "from a prior discovery call. Must be paired with "
                    "around_message_id."
                ),
            },
            "around_message_id": {
                "type": "integer",
                "description": (
                    "Scroll shape. Message id to center the window on — use "
                    "match_message_id from a discovery result, or any id from a "
                    "prior window."
                ),
            },
            "window": {
                "type": "integer",
                "description": (
                    "Scroll shape only. Messages to return on each side of the anchor "
                    "(anchor itself always included). Clamped to [1, 20]. Default 5."
                ),
                "default": 5,
            },
            "role_filter": {
                "type": "string",
                "description": (
                    "Optional. Comma-separated roles to include. Discovery defaults to "
                    "'user,assistant' (tool output is usually noise). Pass "
                    "'user,assistant,tool' to include tool output (debugging tool "
                    "behaviour) or 'tool' to search tool output only."
                ),
            },
            "profile": {
                "type": "string",
                "description": (
                    "Optional. Read sessions from another Hermes profile's database "
                    "(read-only). Use when resolving an `@session:<profile>/<id>` link: "
                    "pass the profile segment here with session_id as the id segment. "
                    "Omit to use the current profile."
                ),
            },
        },
        "required": [],
    },
}


# --- Registry ---
from tools.registry import registry, tool_error

registry.register(
    name="session_search",
    toolset="session_search",
    schema=SESSION_SEARCH_SCHEMA,
    handler=lambda args, **kw: session_search(
        query=args.get("query") or "", limit=args.get("limit", 3), window=args.get("window", 5),
        detail=args.get("detail", "adaptive"), db=kw.get("db"), current_session_id=kw.get("current_session_id"),
        **{k: args.get(k) for k in ("role_filter", "session_id", "around_message_id", "sort", "profile")}),
    check_fn=check_session_search_requirements,
    emoji="🔍")
