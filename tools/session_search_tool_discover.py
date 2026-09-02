"""Discovery shape of session_search: FTS5 query, title match, lineage dedup
and adaptive/full hydration of the surviving results."""

import json
import logging
from typing import Any, Dict, List, Optional

from tools.registry import tool_error
from tools.session_search_tool_common import (
    _DISCOVER_SCAN_LIMIT, _DISCOVER_SEARCH_FIELDS, _HIDDEN_SESSION_SOURCES, _annotate_rebuild_status,
    _format_timestamp, _is_compacted_message, _is_compaction_summary, _order_for_recall, _quiet,
    _resolve_lineage, _resolve_to_parent, _session_left_live_context, _session_link, _shape_message,
)


def _normalize_title_query(query: str) -> str:
    """Strip common quoting the model may include around a remembered title."""
    return query.strip().strip("`'\"")


def _title_match_result(db, query: str, current_lineage_root: Optional[str]) -> Optional[Dict[str, Any]]:
    """Return a discovery-shaped result when the query matches a session title."""
    title_query = _normalize_title_query(query)
    if not title_query:
        return None
    session_id = _quiet(lambda: db.resolve_session_by_title(title_query), None,
                        "resolve_session_by_title failed for %r", title_query)
    if not session_id:
        return None
    lineage_root = _resolve_lineage(db, session_id)
    # Same-lineage title hits are in-context only while the session is live;
    # /new-reset and compression-ended parents are not.
    if (
        current_lineage_root
        and lineage_root == current_lineage_root
        and not _session_left_live_context(db, session_id)
    ):
        return None

    session_meta = _quiet(lambda: db.get_session(lineage_root) or db.get_session(session_id), None,
                          "get_session failed for title match %s", session_id) or {}
    if session_meta.get("source") in _HIDDEN_SESSION_SOURCES:
        return None
    messages = _quiet(lambda: db.get_messages(session_id), [], "get_messages failed for title match %s", session_id)
    anchor_id = messages[0].get("id") if messages else None
    view = {}
    if anchor_id is not None:
        view = _quiet(lambda: db.get_anchored_view(session_id, anchor_id, window=5, bookend=3), {},
                      "get_anchored_view failed for title match %s/%s", session_id, anchor_id)
    entry = {
        "session_id": session_id, "when": _format_timestamp(session_meta.get("started_at")),
        "source": session_meta.get("source", "unknown"), "model": session_meta.get("model") or "unknown",
        "title": session_meta.get("title") or title_query, "matched_role": "session_title",
        "match_message_id": anchor_id,
        "snippet": f"Session title matched: {session_meta.get('title') or title_query}",
        "bookend_start": [_shape_message(m) for m in (view.get("bookend_start") or messages[:3])],
        "messages": [_shape_message(m, anchor_id=anchor_id) for m in (view.get("window") or messages[:5])],
        "bookend_end": [_shape_message(m) for m in (view.get("bookend_end") or messages[-3:])],
        "messages_before": view.get("messages_before", 0),
        "messages_after": view.get("messages_after", max(len(messages) - 5, 0)),
        "detail": "full", "_lineage_root": lineage_root,
    }
    if lineage_root and lineage_root != session_id:
        entry["parent_session_id"] = lineage_root
    return entry


def _discover_payload(db, query: str, detail: str, results: list, **extra) -> str:
    payload = {"success": True, "mode": "discover", "query": query, "detail": detail,
               "results": results, "count": len(results), **extra}
    _annotate_rebuild_status(db, payload)
    return json.dumps(payload, ensure_ascii=False)


def _dedupe_by_lineage(db, raw_results, limit, seen_sessions, current_session_id, current_lineage_root) -> None:
    """Fill *seen_sessions* (lineage_root -> first surviving FTS row) up to *limit*.

    The raw owning session_id stays on the row — only it pairs validly with the
    FTS match id for the anchored window. Current-lineage hits are skipped
    UNLESS the transcript left live context: compression-ended session, /new-
    reset predecessor (hiding it made gateway recall blind after every /new),
    or an in-place compacted row on the SAME session_id. A live delegation
    child has end_reason=None, so it stays excluded.
    """
    for r in raw_results:
        if len(seen_sessions) >= limit:
            break
        raw_sid = r["session_id"]
        resolved_sid, _ = _resolve_to_parent(db, raw_sid)
        is_compacted_hit = _is_compacted_message(db, r.get("id"))
        is_ended_session = _session_left_live_context(db, raw_sid)
        if (
            current_lineage_root
            and resolved_sid == current_lineage_root
            and not (is_ended_session or is_compacted_hit)
        ):
            continue
        if current_session_id and raw_sid == current_session_id and not is_compacted_hit:
            continue
        seen_sessions.setdefault(resolved_sid, {**r, "_lineage_root": resolved_sid})


def _bookend(view: Dict[str, Any], key: str) -> List[Dict[str, Any]]:
    return [_shape_message(m, max_content_len=1200) for m in (view.get(key) or [])
            if not _is_compaction_summary(m.get("content", ""))]


def _hydrate_hit(db, lineage_root: str, match_info: Dict[str, Any], result_detail: str) -> Optional[Dict[str, Any]]:
    """Build one discovery result from a surviving FTS row; None if the anchored
    view can't be loaded (the hit is dropped)."""
    hit_sid = match_info.get("session_id") or lineage_root
    msg_id = match_info.get("id")
    try:
        view = db.get_anchored_view(hit_sid, msg_id, window=5, bookend=3)
    except Exception as e:
        logging.warning("get_anchored_view failed for %s/%s: %s", hit_sid, msg_id, e, exc_info=True)
        return None
    session_meta = _quiet(lambda: db.get_session(lineage_root), None, "get_session failed for %s", lineage_root) or {}
    full = result_detail == "full"
    window_messages = view.get("window") or []
    if not full:
        window_messages = [m for m in window_messages if m.get("id") == msg_id]
    entry = {
        "session_id": hit_sid,
        "when": _format_timestamp(session_meta.get("started_at") or match_info.get("session_started")),
        "source": session_meta.get("source") or match_info.get("source", "unknown"),
        "model": session_meta.get("model") or match_info.get("model") or "unknown",
        "title": session_meta.get("title") or None, "matched_role": match_info.get("role"),
        "match_message_id": msg_id, "snippet": match_info.get("snippet") or "",
        "bookend_start": _bookend(view, "bookend_start") if full else [],
        "messages": [_shape_message(m, anchor_id=msg_id, max_content_len=4000) for m in window_messages],
        "bookend_end": _bookend(view, "bookend_end") if full else [],
        "messages_before": view.get("messages_before", 0), "messages_after": view.get("messages_after", 0),
        "detail": result_detail,
    }
    if lineage_root and lineage_root != hit_sid:
        entry["parent_session_id"] = lineage_root
    return entry


def _discover(db, query: str, role_filter: Optional[List[str]], limit: int, sort: Optional[str],
              detail: str, current_session_id: str = None, link_profile: str = None) -> str:
    """Discovery shape: FTS5 plus adaptive or full result hydration."""
    role_list = role_filter if role_filter else ["user", "assistant"]
    current_lineage_root = _resolve_lineage(db, current_session_id) if current_session_id else None
    title_result = _title_match_result(db, query, current_lineage_root)

    try:
        raw_results = db.search_messages(
            query=query, role_filter=role_list, exclude_sources=list(_HIDDEN_SESSION_SOURCES),
            limit=_DISCOVER_SCAN_LIMIT, offset=0, sort=sort, fields=_DISCOVER_SEARCH_FIELDS,
        )
    except Exception as e:
        logging.error("FTS5 search failed: %s", e, exc_info=True)
        return tool_error(f"Search failed: {e}", success=False)

    # Demote cron rows below interactive ones BEFORE dedup so a high-volume
    # cron corpus can't starve the user's own sessions out of the top `limit`.
    raw_results = _order_for_recall(raw_results)

    if not raw_results and not title_result:
        return _discover_payload(db, query, detail, [], message=(
            "No matching sessions found. FTS5 ANDs all terms by default — "
            "broaden with OR (`alpha OR beta`), exact-match with quoted "
            "phrases, exclude with NOT, or prefix-match with `deploy*`."
        ))

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
        "session_search(session_id=..., around_message_id=match_message_id)."
    ))
