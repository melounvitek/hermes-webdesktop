#!/usr/bin/env python3
"""Session Search Tool - long-term conversation recall over the SQLite session DB.

Single-shape tool; the mode is inferred from the args: DISCOVERY (``query``;
FTS5 deduped by lineage, adaptive detail hydrates only the top result),
SCROLL (``session_id`` + ``around_message_id``; ±window around the anchor),
READ (``session_id`` alone; whole session or head/tail), BROWSE (no args).
No LLM calls — every shape returns actual DB messages. Helpers live in
``session_search_tool_common`` / ``_discover`` and are re-exported here.
"""

import json
import logging
from typing import Any, List, Optional

from tools.session_search_tool_common import (  # noqa: F401  (re-exports)
    _COMPACTION_PREFIXES, _DEMOTED_SESSION_SOURCES, _DISCOVER_SCAN_LIMIT,
    _DISCOVER_SEARCH_FIELDS, _FRESH_RESET_END_REASONS, _HIDDEN_SESSION_SOURCES,
    _annotate_rebuild_status, _format_timestamp, _get_message_storage_state,
    _is_compacted_message, _is_compacted_state, _is_compaction_summary,
    _ok, _order_for_recall, _quiet, _resolve_lineage, _resolve_to_parent, _session_end_reason,
    _session_left_live_context, _session_link, _session_meta_block, _shape_message,
)
from tools.session_search_tool_discover import (  # noqa: F401  (re-exports)
    _discover, _normalize_title_query, _title_match_result,
)


def _resolve_profile_db(profile: str):
    """Open another profile's ``state.db`` read-only (no write lock — safe on a
    live DB), or None for the current profile."""
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
    """Scan every profile's ``state.db`` for a session id -> ``(db, profile_name)``
    or ``(None, None)``. Ids are globally unique, so the first hit is authoritative."""
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
        try:
            pdb = SessionDB(db_path=db_path, read_only=True)
        except Exception:
            continue
        if _quiet(lambda: pdb.get_session(session_id), None,
                  "get_session probe failed for %s in %s", session_id, name):
            return pdb, name
        pdb.close()
    return None, None


def _get_session_meta(db, session_id: str) -> dict:
    """``db.get_session`` that degrades to ``{}`` on error."""
    return _quiet(lambda: db.get_session(session_id), None,
                  "get_session failed for %s: %s", session_id, with_exc=True) or {}


def _read_session(db, session_id: str, head: int = 20, tail: int = 10, link_profile: str = None) -> str:
    """Read shape: whole session, or first ``head`` + last ``tail`` messages
    with a pointer to scroll the middle."""
    meta = _get_session_meta(db, session_id)
    if not meta:
        return tool_error(f"session_id not found: {session_id}", success=False)

    try:
        rows = db.get_messages(session_id)
    except Exception as e:
        logging.error("get_messages failed for %s: %s", session_id, e, exc_info=True)
        return tool_error(f"failed to load session: {e}", success=False)

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
    try:
        # list_sessions_rich (include_children=False) already applies the
        # canonical child classifier: roots, /branch children and /new-reset
        # children are admitted, delegation/compression children hidden.
        # Re-classifying here re-hid legacy reset children — trust the query.
        sessions = db.list_sessions_rich(
            limit=limit + 15,  # extra so we can skip current / compression roots
            exclude_sources=list(_HIDDEN_SESSION_SOURCES),
            order_by_last_active=True,
        )

        current_root, has_compression_hop = (
            _resolve_to_parent(db, current_session_id)
            if current_session_id else (None, False)
        )
        results = []
        for s in sessions:
            sid = s.get("id", "")
            if sid == current_session_id:
                continue
            # Compression continuation: the root's turns were summarised into
            # the live child, so hide the root. /new-reset children share a
            # root but carry no transcript — keep that root browsable.
            if has_compression_hop and current_root and sid == current_root:
                continue
            results.append({
                "session_id": sid, "link": _session_link(sid, link_profile), "title": s.get("title") or None,
                "source": s.get("source", ""), "started_at": s.get("started_at", ""),
                "last_active": s.get("last_active", ""), "message_count": s.get("message_count", 0),
                "preview": s.get("preview", ""),
            })
            if len(results) >= limit:
                break
        return _ok(mode="browse", results=results, count=len(results), message=(
            f"Showing {len(results)} most recent sessions. Pass a query= to search, "
            "or session_id+around_message_id to scroll."))
    except Exception as e:
        logging.error("Error listing recent sessions: %s", e, exc_info=True)
        return tool_error(f"Failed to list recent sessions: {e}", success=False)


def _clamp_int(value, default: int, lo: int, hi: int) -> int:
    if not isinstance(value, int):
        try:
            value = int(value)
        except (TypeError, ValueError):
            value = default
    return max(lo, min(value, hi))


def _anchor_in_live_context(db, anchor_state, anchor_session_id: str, current_session_id: str) -> bool:
    """True when the scroll anchor is still in the caller's active context and
    must be rejected. Same-lineage history that has LEFT live context (compacted
    rows, compression-ended parents, /new-reset predecessors) is allowed, so
    scroll never rejects a result discovery just returned."""
    a_root = _resolve_lineage(db, anchor_session_id)
    c_root = _resolve_lineage(db, current_session_id)
    if not (a_root and c_root and a_root == c_root):
        return False
    if _is_compacted_state(anchor_state):
        return False
    # Rewind/undo rows (active=0, compacted!=1) never count as out-of-context history.
    is_inactive_non_compacted = (
        anchor_state is not None
        and anchor_state["active"] == 0
        and anchor_state["compacted"] != 1
    )
    return is_inactive_non_compacted or not _session_left_live_context(db, anchor_session_id)


def _rebind_to_owner(db, session_id: str, owning: str, around_message_id: int, window: int):
    """Lineage rebind: the caller paired a parent session_id with a message id
    that lives in a descendant (compaction / delegation create child sessions).
    Returns ``(view, warning)`` from the owning session, or ``(None, None)``."""
    a_root = _resolve_lineage(db, session_id)
    o_root = _resolve_lineage(db, owning)
    if not (a_root and o_root and a_root == o_root):
        return None, None
    rebind_view = _quiet(lambda: db.get_messages_around(owning, around_message_id, window=window),
                         None, "rebind get_messages_around failed: %s", with_exc=True)
    if not (rebind_view and rebind_view.get("window")):
        return None, None
    return rebind_view, f"around_message_id {around_message_id} lives in {owning} (child of {session_id}); rebound transparently"


def _scroll(db, session_id: str, around_message_id: int, window: int = 5,
            current_session_id: str = None) -> str:
    """Scroll shape: a window of messages centered on an anchor (no FTS5, no
    bookends). Rebinds silently if the anchor lives in a same-lineage child."""
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
        db, anchor_state, owning_session_id or session_id, current_session_id
    ):
        return tool_error("scroll rejected: anchor lives in the current session lineage (already in your active context)", success=False)

    session_meta = _get_session_meta(db, session_id)
    if not session_meta:
        return tool_error(f"session_id not found: {session_id}", success=False)

    try:
        view = db.get_messages_around(session_id, around_message_id, window=window)
    except Exception as e:
        logging.error("get_messages_around failed: %s", e, exc_info=True)
        return tool_error(f"failed to load messages: {e}", success=False)

    messages = view.get("window") or []
    rebind_warning = None
    if not messages and owning_session_id and owning_session_id != session_id:
        rebind_view, rebind_warning = _rebind_to_owner(
            db, session_id, owning_session_id, around_message_id, window
        )
        if rebind_view is not None:
            view = rebind_view
            messages = view["window"]
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
        **({"warning": rebind_warning} if rebind_warning else {}),
    )


def _read_with_profile_fallback(db, sid: str, profile: Optional[str]) -> str:
    """Read shape. On a miss in the target profile, scan every profile (the
    model may have dropped the owning profile from the link) and tag the result
    with the profile it was found in."""
    result = _read_session(db, sid, link_profile=profile)
    if json.loads(result).get("success"):
        return result
    located, owner = _locate_session_db(sid)
    if located is not None:
        try:
            found = json.loads(_read_session(located, sid, link_profile=owner))
        finally:
            located.close()
        if found.get("success"):
            found["profile"] = owner
            return json.dumps(found, ensure_ascii=False)
    return result


def _dispatch(query, role_filter, limit, db, current_session_id, session_id,
              around_message_id, window, sort, profile, detail, owned_dbs) -> str:
    """Mode dispatch (see module docstring). Scroll wins over read/discovery when
    an anchor is set — the agent asked for a specific slice. Profile DBs opened
    here are appended to *owned_dbs* for the caller to close."""
    # A raw `@session:<profile>/<id>` link passed as session_id: ids never
    # contain "/", so a slash means profile/id — strip the prefix and adopt the
    # embedded profile only when none was passed explicitly.
    if isinstance(session_id, str) and "/" in session_id:
        emb_profile, _, emb_id = session_id.partition("/")
        if emb_id:
            session_id = emb_id
            if emb_profile and (profile is None or not str(profile).strip()):
                profile = emb_profile

    # Cross-profile read: swap in the named profile's DB (read-only) for every
    # shape. Current-lineage guards key off ids that won't collide, so they
    # stay inert.
    if profile is not None and str(profile).strip():
        try:
            profile_db = _resolve_profile_db(profile)
        except Exception as e:
            return tool_error(f"profile '{profile}': {e}", success=False)
        if profile_db is not None:
            db, current_session_id = profile_db, None
            owned_dbs.append(profile_db)

    has_session = isinstance(session_id, str) and bool(session_id.strip())
    if has_session and around_message_id is not None:
        return _scroll(db, session_id, around_message_id, window, current_session_id)
    if has_session:
        return _read_with_profile_fallback(db, session_id.strip(), profile)

    limit = _clamp_int(limit, 3, 1, 10)
    if not query or not isinstance(query, str) or not query.strip():
        return _list_recent_sessions(db, limit, current_session_id, link_profile=profile)

    role_list = ([r.strip() for r in role_filter.split(",") if r.strip()] or None) if isinstance(role_filter, str) else None
    sort_norm = sort.strip().lower() if isinstance(sort, str) else None
    if sort_norm not in ("newest", "oldest"):
        sort_norm = None
    detail_norm = "full" if isinstance(detail, str) and detail.strip().lower() == "full" else "adaptive"
    return _discover(
        db=db, query=query.strip(), role_filter=role_list, limit=limit, sort=sort_norm,
        detail=detail_norm, current_session_id=current_session_id, link_profile=profile,
    )


def session_search(query: str = "", role_filter: str = None, limit: int = 3, db=None,
                   current_session_id: str = None, session_id: str = None, around_message_id: int = None,
                   window: int = 5, sort: str = None, profile: str = None, detail: str = "adaptive") -> str:
    """Run session search and close databases opened by this invocation.
    Parameter order is positional-compatible with older callers."""
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
        query=args.get("query") or "",
        role_filter=args.get("role_filter"),
        limit=args.get("limit", 3),
        session_id=args.get("session_id"),
        around_message_id=args.get("around_message_id"),
        window=args.get("window", 5),
        sort=args.get("sort"),
        detail=args.get("detail", "adaptive"),
        profile=args.get("profile"),
        db=kw.get("db"),
        current_session_id=kw.get("current_session_id"),
    ),
    check_fn=check_session_search_requirements,
    emoji="🔍",
)
