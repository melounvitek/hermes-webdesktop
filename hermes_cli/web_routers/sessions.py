"""Session dashboard routes.

Three routers because global route order matters: ``list_router``
(GET /api/sessions) was registered before the profiles ``sessions_router``
include, ``search_router`` (GET /api/sessions/search) right after it, and
``manage_router`` (mutation/detail endpoints) much later — each is mounted at
its original registration point.  web_server-owned helpers are reached via
the late-binding seam so ``monkeypatch.setattr(web_server, ...)`` keeps working.
"""

import asyncio
import json
import re
import sqlite3
import time
from typing import Callable, List, Optional

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import StreamingResponse

from hermes_cli.web_deps import late
from hermes_cli.web_models import (
    BulkDeleteSessions,
    SessionImport,
    SessionOwnerBackfill,
    SessionPrune,
    SessionRename,
)
from hermes_cli.web_routers._common import log as _log, http_failure
from hermes_state import is_malformed_db_error, is_transient_sqlite_error
from typing import Any, Dict

list_router = APIRouter()
search_router = APIRouter()
manage_router = APIRouter()

_cron_default_profile = late("_cron_default_profile")
_cron_profile_home = late("_cron_profile_home")
_maybe_auto_archive_for_profile = late("_maybe_auto_archive_for_profile")
_open_session_db_for_profile = late("_open_session_db_for_profile")
_session_latest_descendant = late("_session_latest_descendant")
_strip_session_list_rows = late("_strip_session_list_rows")


# CRITICAL — every literal-path route below MUST be declared BEFORE the
# templated ``/api/sessions/{session_id}`` family that follows. FastAPI/
# Starlette match routes in registration order, and the ``{session_id}``
# pattern is unconstrained — it would otherwise swallow e.g.
# ``DELETE /api/sessions/empty``, ``POST /api/sessions/bulk-delete``, or
# ``GET /api/sessions/stats`` as "operate on the session with id
# 'empty'" / "'bulk-delete'" / "'stats'", which would 404 (or worse,
# succeed and delete the wrong row). Same story as the older
# ``/api/sessions/search`` endpoint up at line ~1191. If you split or
# reorder this block, move every route in it together.
# Keep the dashboard import endpoint stream-safe: FastAPI otherwise parses and
# buffers an arbitrarily large JSON body before SessionDB can enforce its own
# per-session and transaction-work limits.
_SESSION_IMPORT_MAX_BYTES = 25 * 1024 * 1024


async def _read_session_import_body(request: Request) -> bytes:
    body = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > _SESSION_IMPORT_MAX_BYTES:
            raise HTTPException(status_code=413, detail="Session import payload is too large")
        body.extend(chunk)
    return bytes(body)


def _import_sessions_for_profile(profile: Optional[str], sessions: List[Dict[str, Any]]) -> Dict[str, Any]:
    db = _open_session_db_for_profile(profile, read_only=False)
    try:
        return db.import_sessions(sessions)
    finally:
        db.close()


def _prune_sessions(body: SessionPrune):
    """Delete ended sessions matching filters (mirrors `hermes sessions prune`)."""
    from hermes_cli.web_server import get_hermes_home
    has_window = (
        body.started_before is not None or body.started_after is not None
    )
    if body.older_than_days is not None and body.older_than_days < 1 and not has_window:
        raise HTTPException(status_code=400, detail="older_than_days must be >= 1")
    # Mirror the CLI: the implicit 90-day cutoff only applies to a truly bare
    # prune. Any attribute filter (source, title, model, ...) suppresses it
    # unless the caller explicitly sent older_than_days.
    _attr_filters_set = any(
        getattr(body, f) is not None
        for f in (
            "source", "title_like", "end_reason", "cwd_prefix",
            "min_messages", "max_messages", "model_like", "provider",
            "user_id", "chat_id", "chat_type", "branch_like",
            "min_tokens", "max_tokens", "min_cost", "max_cost",
            "min_tool_calls", "max_tool_calls",
        )
    )
    _older_than_explicit = "older_than_days" in body.model_fields_set
    _effective_older_than = body.older_than_days
    if has_window or (_attr_filters_set and not _older_than_explicit):
        _effective_older_than = None
    profile_home = _cron_profile_home(body.profile)[1] if body.profile else get_hermes_home()
    db = _open_session_db_for_profile(body.profile, read_only=False)
    try:
        filters = dict(
            older_than_days=_effective_older_than,
            source=(body.source or None),
            started_before=body.started_before,
            started_after=body.started_after,
            title_like=(body.title_like or None),
            end_reason=(body.end_reason or None),
            cwd_prefix=(body.cwd_prefix or None),
            min_messages=body.min_messages,
            max_messages=body.max_messages,
            model_like=(body.model_like or None),
            provider=(body.provider or None),
            user_id=(body.user_id or None),
            chat_id=(body.chat_id or None),
            chat_type=(body.chat_type or None),
            branch_like=(body.branch_like or None),
            min_tokens=body.min_tokens,
            max_tokens=body.max_tokens,
            min_cost=body.min_cost,
            max_cost=body.max_cost,
            min_tool_calls=body.min_tool_calls,
            max_tool_calls=body.max_tool_calls,
            archived=None if body.include_archived else False,
        )
        skipped_open = db.count_open_prune_matches(**filters)
        if body.dry_run:
            rows = db.list_prune_candidates(**filters)
            return {
                "ok": True,
                "removed": 0,
                "matched": len(rows),
                "skipped_open": skipped_open,
                # Rows are ordered by last activity, not creation time.
                "oldest_last_active": rows[0]["last_active"] if rows else None,
                "newest_last_active": rows[-1]["last_active"] if rows else None,
                "oldest_started_at": (
                    min(r["started_at"] for r in rows) if rows else None
                ),
                "newest_started_at": (
                    max(r["started_at"] for r in rows) if rows else None
                ),
                "sessions": [
                    {
                        "id": r["id"],
                        "source": r["source"],
                        "title": r.get("title"),
                        "model": r.get("model"),
                        "started_at": r["started_at"],
                        "last_active": r["last_active"],
                        "message_count": r["message_count"],
                    }
                    for r in rows
                ],
            }
        sessions_dir = profile_home / "sessions"
        removed = db.prune_sessions(
            sessions_dir=sessions_dir if sessions_dir.exists() else None,
            **filters,
        )
        return {"ok": True, "removed": removed, "skipped_open": skipped_open}
    finally:
        db.close()

_ACTIVE_WINDOW_S = 300


def _csv(value: Optional[str]) -> List[str]:
    """Split a comma-separated query param into stripped, non-empty items."""
    return [s.strip() for s in (value or "").split(",") if s.strip()]


def _is_active(row: dict, now: float) -> bool:
    return (
        row.get("ended_at") is None
        and (now - row.get("last_active", row.get("started_at", 0))) < _ACTIVE_WINDOW_S
    )


def _with_db(profile: Optional[str], fn: Callable, *, read_only: bool):
    """Open the profile's session DB, run ``fn(db)``, always close."""
    db = _open_session_db_for_profile(profile, read_only=read_only)
    try:
        return fn(db)
    finally:
        db.close()


def _serving_profile(profile: Optional[str]) -> str:
    """The profile name rows are stamped with: the requested one, else the
    serving process's own — so default-profile rows never circulate unowned."""
    return _cron_profile_home(profile)[0] if profile else _cron_default_profile()


def _resolve_session_id(db, session_id: str) -> Optional[str]:
    """Resolve *session_id*, distinguishing "absent" from "unreadable".

    On a corrupt ``state.db`` the exact-match lookup (primary-key index) just
    misses, while the prefix fallback scans the b-tree and raises "database
    disk image is malformed".  Both used to end at 404; report corruption as
    503 with the actual problem instead of an empty store.
    """
    try:
        return db.resolve_session_id(session_id)
    except sqlite3.DatabaseError as exc:
        if not is_malformed_db_error(exc):
            raise
        _log.error(
            "state.db is corrupt while resolving session %s: %s", session_id, exc
        )
        raise HTTPException(
            status_code=503,
            detail=(
                "Session store is corrupt (database disk image is malformed). "
                "Sessions cannot be read until it is repaired — run "
                "`hermes doctor` for diagnosis."
            ),
        ) from exc


@list_router.get("/api/sessions")
def get_sessions(
    # ``le=100``: an unbounded limit lets one request drag every session row
    # (plus correlated-subquery preview work) out of SQLite in a single hit.
    limit: int = Query(20, ge=0, le=100),
    offset: int = Query(0, ge=0),
    min_messages: int = 0,
    archived: str = "exclude",
    order: str = "created",
    source: str = None,
    sources: str = None,
    exclude_sources: str = None,
    cwd_prefix: str = None,
    full: bool = False,
    profile: Optional[str] = None,
):
    """List sessions.

    ``archived``: ``exclude`` (default) / ``only`` / ``include``.  ``order``:
    ``created`` (start time) or ``recent`` (latest activity across the
    compression chain, so a long-running chat stays on page one after it
    auto-compresses onto a fresh id).  Rows omit ``system_prompt`` /
    ``model_config`` unless ``full=1``.
    """
    if archived not in ("exclude", "only", "include"):
        raise HTTPException(
            status_code=400,
            detail="archived must be one of: exclude, only, include",
        )
    if order not in ("created", "recent"):
        raise HTTPException(
            status_code=400,
            detail="order must be one of: created, recent",
        )
    profile_name: Optional[str] = None
    if profile:
        profile_name, _ = _cron_profile_home(profile)
    try:
        # Auto-archive is the only write on this GET path: run it on its own
        # maintenance connection, then open the listing connection read-only.
        _maybe_auto_archive_for_profile(profile)
        db = _open_session_db_for_profile(profile, read_only=True)
        try:
            min_message_count = max(0, min_messages)
            archived_only = archived == "only"
            include_archived = archived == "include"
            # Source scoping: the desktop splits recents (exclude=cron) from
            # the cron-jobs section (source=cron) into two independent lists.
            source_list = _csv(sources)
            exclude_list = _csv(exclude_sources)
            sessions = db.list_sessions_rich(
                source=source or None,
                sources=source_list or None,
                exclude_sources=exclude_list or None,
                cwd_prefix=(cwd_prefix or None),
                limit=limit,
                offset=offset,
                min_message_count=min_message_count,
                include_archived=include_archived,
                archived_only=archived_only,
                order_by_last_active=order == "recent",
                # Skip the system_prompt blob inside SQLite too (pairs with
                # _strip_session_list_rows below).
                compact_rows=not full,
                include_pinned=True,
            )
            total = db.session_count(
                source=source or None,
                sources=source_list or None,
                cwd_prefix=(cwd_prefix or None),
                exclude_sources=exclude_list or None,
                min_message_count=min_message_count,
                include_archived=include_archived,
                archived_only=archived_only,
                exclude_children=True,
            )
            now = time.time()
            row_profile = profile_name or _cron_default_profile()
            for s in sessions:
                s["is_active"] = _is_active(s, now)
                s["profile"] = row_profile
                s["is_default_profile"] = row_profile == "default"
                # SQLite stores the flags as 0/1; expose real JSON booleans.
                s["archived"] = bool(s.get("archived"))
                s["pinned"] = bool(s.get("pinned"))
            if not full:
                _strip_session_list_rows(sessions)
            return {"sessions": sessions, "total": total, "limit": limit, "offset": offset}
        finally:
            db.close()
    except HTTPException:
        raise
    except sqlite3.OperationalError as exc:
        _log.exception("GET /api/sessions failed")
        # 503, not 500: the store is busy, not gone — the desktop keeps its
        # sidebar instead of reading a 500 as an authoritative empty list.
        # The bounded open-retry lives in SessionDB's read-only constructor.
        transient = is_transient_sqlite_error(exc)
        raise HTTPException(
            status_code=503 if transient else 500,
            detail=(
                "Session store is busy (disk I/O or lock). Retry; the list was not cleared."
                if transient
                else "Internal server error"
            ),
        ) from exc
    except Exception:
        _log.exception("GET /api/sessions failed")
        raise HTTPException(status_code=500, detail="Internal server error")


@search_router.get("/api/sessions/search")
async def search_sessions(
    q: str = "",
    limit: int = 20,
    profile: Optional[str] = None,
    source: str = None,
    sources: str = None,
    exclude_sources: str = None,
):
    """Search sessions by ID plus FTS5 message content.

    ID matches first, then content matches.  Results are deduped by
    compression lineage, not raw ``session_id``: auto-compression rotates a
    chat onto a fresh id and leaves the old segment in the FTS index, so one
    logical chat owns many rows.  Branches also use ``parent_session_id`` but
    are real alternate conversations — they are NOT collapsed into the parent.
    """
    if not q or not q.strip():
        return {"results": []}
    with http_failure("GET /api/sessions/search failed", 500, detail="Search failed"):
        db = _open_session_db_for_profile(profile, read_only=True)
        try:
            safe_limit = max(1, min(int(limit or 20), 100))
            source_filter = source or None
            source_list = _csv(sources)
            include_sources = [source_filter] if source_filter else (source_list or None)
            exclude_list = _csv(exclude_sources)
            now = time.time()

            # Walk parent_session_id to the compression root, memoized per
            # chain; stops at branch/delegate edges (those stay searchable).
            root_cache: dict = {}

            def compression_root(session_id: str) -> str:
                if not session_id:
                    return session_id
                if session_id in root_cache:
                    return root_cache[session_id]
                chain = []
                cur = session_id
                visited = set()
                root = session_id
                while cur and cur not in visited:
                    visited.add(cur)
                    chain.append(cur)
                    if cur in root_cache:
                        root = root_cache[cur]
                        break
                    try:
                        s = db.get_session(cur)
                    except Exception:
                        s = None
                    if not s:
                        root = cur
                        break
                    parent = s.get("parent_session_id") if isinstance(s, dict) else None
                    if not parent:
                        root = cur
                        break
                    try:
                        parent_session = db.get_session(parent)
                    except Exception:
                        parent_session = None
                    if not parent_session:
                        root = cur
                        break
                    parent_ended_at = parent_session.get("ended_at")
                    started_at = s.get("started_at")
                    is_compression_edge = (
                        parent_session.get("end_reason") == "compression"
                        and parent_ended_at is not None
                        and started_at is not None
                        and started_at >= parent_ended_at
                    )
                    if not is_compression_edge:
                        root = cur
                        break
                    cur = parent
                for node in chain:
                    root_cache[node] = root
                return root

            tip_cache: dict = {}

            def lineage_tip(root_id: str) -> str:
                if root_id in tip_cache:
                    return tip_cache[root_id]
                tip = root_id
                try:
                    resolved = db.get_compression_tip(root_id)
                    if resolved:
                        tip = resolved
                except Exception:
                    pass
                tip_cache[root_id] = tip
                return tip

            # One keyspace for id-hits and content-hits, keyed by lineage root;
            # first hit wins, and ID matches run first.
            seen: dict = {}

            def add_lineage_result(raw_sid: str, payload: dict) -> None:
                if not raw_sid:
                    return
                root = compression_root(raw_sid)
                if root in seen or len(seen) >= safe_limit:
                    return
                payload = dict(payload)
                sid = lineage_tip(root)
                payload["session_id"] = sid
                payload["lineage_root"] = root
                try:
                    row = db.get_session_rich_row(sid)
                except Exception:
                    row = None
                if row:
                    payload.update(
                        {
                            "id": row.get("id") or sid,
                            "source": row.get("source"),
                            "model": row.get("model"),
                            "title": row.get("title"),
                            "started_at": row.get("started_at"),
                            "ended_at": row.get("ended_at"),
                            "last_active": row.get("last_active") or row.get("started_at"),
                            "is_active": (
                                row.get("ended_at") is None
                                and (now - (row.get("last_active") or row.get("started_at") or 0)) < 300
                            ),
                            "message_count": row.get("message_count") or 0,
                            "tool_call_count": row.get("tool_call_count") or 0,
                            "input_tokens": row.get("input_tokens") or 0,
                            "output_tokens": row.get("output_tokens") or 0,
                            "preview": row.get("preview"),
                            "parent_session_id": row.get("parent_session_id"),
                            "archived": bool(row.get("archived")),
                        }
                    )
                else:
                    payload["id"] = sid
                seen[root] = payload

            # Direct ID matches first (pasted ids never appear in message text).
            for row in db.search_sessions_by_id(
                q,
                limit=safe_limit,
                include_archived=True,
                source=source_filter,
                sources=source_list or None,
                exclude_sources=exclude_list or None,
            ):
                sid = row.get("id")
                preview = (row.get("preview") or "").strip()
                snippet = preview or f"Session ID: {sid}"
                add_lineage_result(
                    sid,
                    {
                        "snippet": snippet,
                        "role": None,
                        "source": row.get("source"),
                        "model": row.get("model"),
                        "session_started": row.get("started_at"),
                    },
                )

            # Prefix wildcards so partial words match ("nimb" -> "nimb*");
            # quoted phrases and existing wildcards are kept as-is.
            prefix_query = " ".join(
                tok if tok.startswith('"') or tok.endswith("*") else tok + "*"
                for tok in re.findall(r'"[^"]*"|\S+', q.strip())
            )
            # Over-fetch so lineage dedup can still surface `limit` distinct
            # conversations when several hits collapse onto one root.
            fetch_limit = max(safe_limit * 5, 50)
            matches = db.search_messages(
                query=prefix_query,
                source_filter=include_sources,
                exclude_sources=exclude_list or None,
                limit=fetch_limit,
                fields=(
                    "session_id",
                    "role",
                    "snippet",
                    "source",
                    "model",
                    "session_started",
                ),
            )

            for m in matches:
                if len(seen) >= safe_limit:
                    break
                add_lineage_result(
                    m["session_id"],
                    {
                        "snippet": m.get("snippet", ""),
                        "role": m.get("role"),
                        "source": m.get("source"),
                        "model": m.get("model"),
                        "session_started": m.get("session_started"),
                    },
                )
            return {"results": list(seen.values())}
        finally:
            db.close()


@manage_router.post("/api/sessions/bulk-delete")
async def bulk_delete_sessions_endpoint(body: BulkDeleteSessions):
    """Delete every session in ``body.ids`` in one transaction (POST because
    many clients refuse a DELETE body).

    Contract matches :meth:`SessionDB.delete_sessions`: unknown ids are
    skipped (``deleted`` reports what really happened — selection state can
    race another tab's delete), children are orphaned not cascaded, active
    and archived rows ARE deleted because the user hand-picked them, and
    on-disk transcript cleanup is left to the next prune pass.
    """
    # Hard cap so a runaway selection can't lock the writer for long; 500
    # covers "select all on every page of a reasonable scrollback".
    if len(body.ids) > 500:
        raise HTTPException(
            status_code=400,
            detail="ids must contain at most 500 entries",
        )
    deleted = await asyncio.to_thread(
        _with_db, body.profile, lambda db: db.delete_sessions(body.ids), read_only=False
    )
    return {"ok": True, "deleted": deleted}


@manage_router.post("/api/sessions/import")
async def import_sessions_endpoint(request: Request):
    """Import sessions exported from the dashboard or CLI (session rows only —
    ``/api/ops/import`` restores a whole backup archive)."""
    try:
        raw_body = await _read_session_import_body(request)
        body = SessionImport.model_validate_json(raw_body)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid session import payload") from exc

    try:
        result = await asyncio.to_thread(_import_sessions_for_profile, body.profile, body.sessions)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not result.get("ok", False):
        raise HTTPException(status_code=400, detail=result)
    return result


@manage_router.get("/api/sessions/empty/count")
async def count_empty_sessions_endpoint(profile: Optional[str] = None):
    """Count of empty, ended, non-archived sessions (drives the "Delete empty
    (N)" button, hidden when N is 0)."""
    count = await asyncio.to_thread(
        _with_db, profile, lambda db: db.count_empty_sessions(), read_only=True
    )
    return {"count": count}


@manage_router.delete("/api/sessions/empty")
async def delete_empty_sessions_endpoint(profile: Optional[str] = None):
    """Delete every empty, ended, non-archived session in one transaction.

    Mirrors :meth:`SessionDB.delete_empty_sessions`: "empty" means NO
    ``messages`` rows at all (a rewound/compacted chat reads
    ``message_count == 0`` while its soft-archived rows are the only copy of
    the transcript); active and archived sessions are skipped; children are
    orphaned; on-disk cleanup is left to the next prune pass.
    """
    deleted = await asyncio.to_thread(
        _with_db, profile, lambda db: db.delete_empty_sessions(), read_only=False
    )
    return {"ok": True, "deleted": deleted}


@manage_router.get("/api/sessions/stats")
async def get_session_stats(profile: Optional[str] = None):
    """Session-store statistics (mirrors `hermes sessions stats`).  Registered
    before ``/api/sessions/{session_id}`` so ``stats`` isn't taken as an id."""
    def _stats(db):
        out = {
            "total": db.session_count(include_archived=True),
            "active_store": db.session_count(include_archived=False),
            "archived": db.session_count(archived_only=True),
            "messages": db.message_count(),
            "by_source": {},
        }
        try:
            out["by_source"] = db.session_count_by_source(
                include_archived=True,
                exclude_children=True,
            )
        except Exception:
            pass
        return out

    return _with_db(profile, _stats, read_only=True)


@manage_router.get("/api/sessions/{session_id}")
async def get_session_detail(session_id: str, profile: Optional[str] = None):
    def _detail(db):
        sid = _resolve_session_id(db, session_id)
        session = db.get_session(sid) if sid else None
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        # Always stamp the owner: stamping only on explicit ``?profile=`` left
        # default-profile rows unowned, so multi-profile clients resolved them
        # to whichever gateway happened to be active.
        session["profile"] = _serving_profile(profile)
        session["is_default_profile"] = session["profile"] == "default"
        return session

    return _with_db(profile, _detail, read_only=True)


@manage_router.get("/api/sessions/{session_id}/latest-descendant")
async def get_session_latest_descendant(
    session_id: str,
    profile: Optional[str] = None,
):
    latest, path = await asyncio.to_thread(
        _with_db, profile, lambda db: _session_latest_descendant(session_id, db), read_only=True
    )
    if not latest:
        raise HTTPException(status_code=404, detail="Session not found")
    return {
        "requested_session_id": path[0] if path else session_id,
        "session_id": latest,
        "path": path,
        "changed": bool(path and latest != path[0]),
    }


@manage_router.get("/api/sessions/{session_id}/messages")
async def get_session_messages(
    session_id: str,
    profile: Optional[str] = None,
    limit: Optional[int] = Query(None, ge=0),
    offset: int = Query(0, ge=0),
    order: Optional[str] = Query(None),
    include_compacted: bool = Query(False),
):
    if order not in (None, "oldest", "latest"):
        raise HTTPException(
            status_code=400,
            detail="order must be one of: oldest, latest",
        )

    def _read(db):
        sid = _resolve_session_id(db, session_id)
        if not sid:
            return None
        sid = db.resolve_resume_session_id(sid)
        # Always page: an omitted limit used to load whole transcripts (hundreds
        # of thousands of rows for a runaway session).  Explicit pagination
        # anchors at the start; the default view is the latest page.
        default_page = limit is None
        latest_page = order == "latest" or (order is None and default_page)
        _limit = 500 if default_page else min(limit, 500)
        return sid, _limit, db.get_messages(
            sid,
            limit=_limit,
            offset=offset,
            latest=latest_page,
            include_compacted=include_compacted,
        )

    result = await asyncio.to_thread(_with_db, profile, _read, read_only=True)
    if result is None:
        raise HTTPException(status_code=404, detail="Session not found")
    sid, _limit, messages = result
    from agent.compaction_display import project_compaction_message_for_display
    from agent.context_compressor import is_compaction_summary_message

    projected_messages = []
    for message in messages:
        if not is_compaction_summary_message(message):
            projected_messages.append(message)
            continue
        display_view = project_compaction_message_for_display(message)
        projected = message.copy()
        if display_view is None:
            if not projected.get("display_kind"):
                projected["display_kind"] = "hidden"
        else:
            # Keep the physical content for inspection/export compatibility;
            # Desktop consumes this display-only projection. A legacy hidden
            # wrapper must not hide a successfully recovered live ask.
            projected["display_content"] = display_view.get("content")
            projected.pop("display_kind", None)
        projected_messages.append(projected)
    return {
        "session_id": sid,
        "messages": projected_messages,
        "pagination": {
            "limit": _limit,
            "offset": offset,
            "order": order or ("latest" if limit is None else "oldest"),
            "returned": len(projected_messages),
        },
    }


@manage_router.delete("/api/sessions/{session_id}")
async def delete_session_endpoint(session_id: str, profile: Optional[str] = None):
    # ``profile`` opens another local profile's state.db directly (remote
    # profiles never reach here — the desktop routes those to the remote backend).
    def _delete(db):
        # An already-absent session is an idempotent success: the desktop
        # optimistically removes the row and RESTORES it on any error, so a 404
        # here resurrected ghost rows (transient empties from /goal +
        # auto-compression churn racing the sidebar snapshot).
        sid = _resolve_session_id(db, session_id)
        if not sid:
            return {"ok": True, "already_absent": True}
        db.delete_session(sid)
        return {"ok": True}

    return await asyncio.to_thread(_with_db, profile, _delete, read_only=False)


@manage_router.post("/api/sessions/owner-backfill")
async def backfill_session_owner_profiles(body: SessionOwnerBackfill):
    """Stamp legacy ``profile_name = NULL`` rows with this store's own
    serving-profile identity.

    A multi-connection Desktop fails closed on unowned rows, leaving legacy
    sessions unresumable.  Each ``state.db`` belongs to exactly one profile,
    so this is a single-match backfill (the same identity ``get_sessions``
    stamps on outgoing rows), idempotent: non-NULL owners are never overwritten.
    """
    stamp = _serving_profile(body.profile)

    with http_failure("POST /api/sessions/owner-backfill failed", 500, detail="Internal server error"):
        stamped = await asyncio.to_thread(
            _with_db, body.profile, lambda db: db.backfill_null_session_profiles(stamp), read_only=False
        )

    if stamped:
        _log.info(
            "owner-backfill: stamped %d legacy NULL-profile session row(s) with profile %r",
            stamped,
            stamp,
        )
    return {"ok": True, "stamped": stamped, "profile": stamp}


@manage_router.patch("/api/sessions/{session_id}")
async def rename_session_endpoint(session_id: str, body: SessionRename):
    """Update a session: ``title`` (empty clears), ``archived``, ``hidden``,
    ``pinned`` (exempts from the auto-archive sweep), ``unread`` (True =
    explicitly unread, False = read up to now).  Any field may be omitted."""
    flags = ("archived", "hidden", "pinned", "unread")

    def _update(db):
        sid = _resolve_session_id(db, session_id)
        if not sid:
            raise HTTPException(status_code=404, detail="Session not found")
        if body.title is None and all(getattr(body, f) is None for f in flags):
            raise HTTPException(
                status_code=400,
                detail="Nothing to update; provide 'title', 'archived', 'hidden', 'pinned', and/or 'unread'.",
            )
        if body.title is not None:
            try:
                db.set_session_title(sid, body.title or "")
            except ValueError as e:
                # Title too long, invalid characters, or already in use.
                raise HTTPException(status_code=400, detail=str(e))
        if body.archived is not None:
            db.set_session_archived(sid, body.archived)
        if body.hidden is not None:
            db.set_session_hidden(sid, body.hidden)
        if body.pinned is not None:
            db.set_session_pinned(sid, body.pinned)
        if body.unread is not None:
            db.set_session_read(sid, read=not body.unread)
        result = {"ok": True, "title": db.get_session_title(sid) or ""}
        for f in flags:
            if getattr(body, f) is not None:
                result[f] = bool(getattr(body, f))
        return result

    return _with_db(body.profile, _update, read_only=False)


@manage_router.get("/api/sessions/{session_id}/export")
async def export_session_endpoint(session_id: str, profile: Optional[str] = None):
    """Stream a single session (metadata + messages) as JSON."""
    def _prepare_export(db):
        sid = _resolve_session_id(db, session_id)
        return (sid, db.get_session(sid)) if sid else None

    prepared = await asyncio.to_thread(_with_db, profile, _prepare_export, read_only=True)
    if prepared is None or prepared[1] is None:
        raise HTTPException(status_code=404, detail="Session not found")

    sid, session = prepared

    def _stream_export():
        db = _open_session_db_for_profile(profile, read_only=True)
        try:
            metadata = json.dumps(
                jsonable_encoder(session),
                ensure_ascii=False,
                separators=(",", ":"),
            )
            yield metadata[:-1] + ',"messages":['

            # Keyset pagination (id > last_seen): O(n) total over the
            # transcript, vs OFFSET's O(n²) on huge sessions.
            last_id = None
            first = True
            while True:
                messages = db.get_messages(
                    sid,
                    limit=500,
                    after_id=last_id if last_id is not None else 0,
                )
                for message in messages:
                    if not first:
                        yield ","
                    yield json.dumps(
                        jsonable_encoder(message),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    first = False
                if len(messages) < 500:
                    break
                last_id = messages[-1].get("id")
                if last_id is None:
                    break  # defensive: cannot keyset without row ids

            yield "]}"
        finally:
            db.close()

    return StreamingResponse(
        _stream_export(),
        media_type="application/json",
    )


@manage_router.post("/api/sessions/prune")
async def prune_sessions_endpoint(body: SessionPrune):
    """Delete ended sessions matching filters without blocking the event loop."""
    return await asyncio.to_thread(_prune_sessions, body)
