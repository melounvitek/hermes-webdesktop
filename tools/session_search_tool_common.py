"""Shared helpers for the session_search tool: source classification, lineage
resolution, message storage state, and response shaping. Imported by
``tools.session_search_tool`` (which re-exports the names) and
``tools.session_search_tool_discover``."""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from hermes_state_common import _RESET_END_REASONS

# Hidden from browsing/searching: integrations (HERMES_SESSION_SOURCE=tool),
# delegate subagent runs, kanban workers — not the user's history.
_HIDDEN_SESSION_SOURCES = ("kanban", "subagent", "tool")

# Searchable but DEMOTED below interactive sessions: cron sessions' repetitive
# vocabulary dominates bare BM25 and starves out the user's own sessions
# ("recall blindness"). Demoting keeps them reachable when they're the only match.
_DEMOTED_SESSION_SOURCES = ("cron",)

# FTS rows scanned before dedup-by-lineage — well above the handful of distinct
# sessions a query returns, so interactive matches buried under cron hits are
# still in hand for the demotion pass.
_DISCOVER_SCAN_LIMIT = 300

# Raw FTS rows are only a discovery-plan input; the response hydrates its own
# anchored window and bookends after lineage dedup.
_DISCOVER_SEARCH_FIELDS = ("id", "session_id", "role", "snippet", "source", "model", "session_started")

# Generated context-compaction handoff summaries (agent/context_compressor.py);
# excluded from bookends so huge compaction payloads aren't re-introduced.
_COMPACTION_PREFIXES = ("[CONTEXT COMPACTION", "[CONTEXT SUMMARY]:")

# /new, /reset, idle/daily expiry and CLI /new ("new_session") end the
# predecessor WITHOUT carrying its transcript forward — unlike compression
# continuations and live delegation children. Derived from the gateway set so
# this tool and the recovery fence cannot drift.
_FRESH_RESET_END_REASONS = frozenset(_RESET_END_REASONS) | {"new_session"}


def _quiet(fn, default, msg, *log_args, with_exc: bool = False):
    """Call ``fn()``; on any exception debug-log *msg* (appending the exception
    as a final ``%s`` arg when *with_exc*) and return *default*."""
    try:
        return fn()
    except Exception as e:
        logging.debug(msg, *(log_args + (e,) if with_exc else log_args), exc_info=True)
        return default


def _format_timestamp(ts: Union[int, float, str, None]) -> str:
    """Unix timestamp (number or numeric string) or ISO string -> readable date.
    "unknown" for None; str(ts) if conversion fails."""
    if ts is None:
        return "unknown"
    try:
        value = ts
        if isinstance(ts, str):
            if not ts.replace(".", "").replace("-", "").isdigit():
                return ts
            value = float(ts)
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(value).strftime("%B %d, %Y at %I:%M %p")
    except (ValueError, OSError, OverflowError) as e:
        logging.debug("Failed to format timestamp %s: %s", ts, e, exc_info=True)
    except Exception as e:
        logging.debug("Unexpected error formatting timestamp %s: %s", ts, e, exc_info=True)
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
    """Walk parent_session_id to the lineage root -> ``(root_id, has_compression_hop)``.
    The flag distinguishes a compression-split lineage (parent content summarised
    away) from a delegation lineage (child content still visible to the parent).
    Falls back to ``(session_id, False)`` on errors."""
    if not session_id:
        return session_id, False
    visited: set[str] = set()
    cur, has_compression = session_id, False
    while cur and cur not in visited:
        visited.add(cur)
        s = _quiet(lambda: db.get_session(cur), None, "Error resolving parent for %s: %s", cur, with_exc=True)
        if not s:
            break
        if s.get("end_reason") == "compression":
            has_compression = True
        if not s.get("parent_session_id"):
            break
        cur = s["parent_session_id"]
    return cur, has_compression


def _resolve_lineage(db, session_id: str) -> str:
    """Return only the lineage root (ignores the compression hop)."""
    return _resolve_to_parent(db, session_id)[0]


def _session_end_reason(db, session_id: str) -> Optional[str]:
    """Return the session's ``end_reason``, or None if missing/unended/error."""
    if not session_id:
        return None
    try:
        s = db.get_session(session_id)
        return (s.get("end_reason") or None) if s else None
    except Exception:
        return None


def _session_left_live_context(db, session_id: str) -> bool:
    """True when *session_id*'s transcript is no longer in anyone's live context:
    ``compression`` (summarised into the continuation child) or a fresh reset
    (child starts empty). Everything else stays excluded from same-lineage
    recall — live delegation children (``end_reason is None``) are visible to
    the parent agent, and ``branched`` parents were copied verbatim into the
    branch child, so their content IS the current context."""
    end_reason = _session_end_reason(db, session_id)
    return end_reason == "compression" or end_reason in _FRESH_RESET_END_REASONS


def _get_message_storage_state(db, message_id) -> Optional[Dict[str, Any]]:
    """Return the owning session and visibility flags for *message_id*."""
    if not message_id:
        return None

    def _lookup():
        with db._lock:
            return db._conn.execute(
                "SELECT session_id, active, compacted FROM messages WHERE id = ?", (message_id,)
            ).fetchone()

    row = _quiet(_lookup, None, "message storage-state lookup failed for %s", message_id)
    return dict(row) if row is not None else None


def _is_compacted_state(state: Optional[Dict[str, Any]]) -> bool:
    """Compaction archives are ``active=0, compacted=1`` (content summarised
    away by archive_and_compact). Rewind/undo rows are ``active=0, compacted=0``
    and must stay hidden."""
    return state is not None and state["active"] == 0 and state["compacted"] == 1


def _is_compacted_message(db, message_id) -> bool:
    """True if *message_id* is a compaction-archived row — pre-compaction content
    no longer in live context, so it should stay discoverable even on the
    current session. False on any error (caller falls back to skipping)."""
    return _is_compacted_state(_get_message_storage_state(db, message_id))


def _annotate_rebuild_status(db, payload: Dict[str, Any]) -> None:
    """Add a rebuild-progress note while the deferred FTS backfill is running,
    so the agent can explain thin/slow results instead of treating them as
    ground truth. No-op (never raises) when no rebuild is pending."""
    try:
        status = db.fts_rebuild_status()
    except Exception:
        return
    if status is None:
        return
    payload["index_rebuild"] = {"percent": status["percent"], "note": (
        f"The search index is rebuilding in the background ({status['percent']}% done, "
        f"{status['indexed']:,} of {status['total']:,} messages). Results from older messages "
        f"may be incomplete until it finishes."
    )}


def _order_for_recall(raw_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Stable-sort FTS rows so interactive sessions rank above automation.
    BM25 order is preserved within each class; only cross-class order changes,
    so a cron hit never displaces an interactive hit during lineage dedup."""
    return sorted(raw_results, key=lambda r: 1 if (r.get("source") or "") in _DEMOTED_SESSION_SOURCES else 0)


def _shape_message(m: Dict[str, Any], anchor_id: Optional[int] = None,
                   max_content_len: Optional[int] = None) -> Dict[str, Any]:
    """Slim a message row for the tool response. Keeps content even if empty
    (absent content is meaningful — tool-call-only assistant turns). With
    *max_content_len*, content is truncated and ``content_truncated`` /
    ``original_content_chars`` added."""
    content = m.get("content")
    if isinstance(content, str) and "\x1b" in content:
        # Recalled messages can carry ANSI escapes (archived terminal output).
        from tools.ansi_strip import strip_ansi

        content = strip_ansi(content)
    original_chars = None
    if max_content_len and content and len(content) > max_content_len:
        original_chars = len(content)
        content = content[:max_content_len] + "…"
    entry = {"id": m.get("id"), "role": m.get("role"), "content": content, "timestamp": m.get("timestamp")}
    entry.update({k: m.get(k) for k in ("tool_name", "tool_calls", "tool_call_id") if m.get(k)})
    if anchor_id is not None and m.get("id") == anchor_id:
        entry["anchor"] = True
    if original_chars is not None:
        entry["content_truncated"] = True
        entry["original_content_chars"] = original_chars
    return {k: v for k, v in entry.items() if v is not None or k == "content"}


def _session_link(session_id: str, profile: str = None) -> str:
    """The reference the agent writes to point the user at a session — same
    value the desktop composer emits, so it renders as a titled link. The
    profile segment is omitted when it can't be named confidently (a bare id
    still resolves, it just can't disambiguate across profiles)."""
    name = (profile or "").strip()
    if not name:
        def _active():
            from hermes_cli.profiles import get_active_profile_name

            resolved = get_active_profile_name()
            return "" if resolved == "custom" else resolved

        name = _quiet(_active, "", "get_active_profile_name failed for session link")
    return f"@session:{name}/{session_id}" if name else f"@session:{session_id}"
