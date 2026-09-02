"""Kanban dashboard plugin — backend API routes.

Mounted at /api/plugins/kanban/ by the dashboard plugin system. Every handler
is a thin wrapper around ``hermes_cli.kanban_db`` (the same code paths the CLI
and gateway ``/kanban`` command use, so the three surfaces cannot drift).

Live updates arrive via the ``/events`` WebSocket, which tails the append-only
``task_events`` table on a short poll (WAL mode lets reads run alongside the
dispatcher's IMMEDIATE write transactions).

Security: plugin HTTP routes sit behind the dashboard's session-token auth
middleware like core API routes. The ``/events`` WebSocket carries its
credential in the query string (browsers cannot set ``Authorization`` on an
upgrade request) and is gated by the dashboard's canonical WS auth check.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator, Optional

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect, status as http_status
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from hermes_cli import kanban_db
from hermes_cli import kanban_diagnostics as kd
from hermes_cli.kanban_db import (
    KANBAN_ATTACHMENT_MAX_BYTES,
    _collision_free_path,
    _safe_attachment_name,
)

log = logging.getLogger(__name__)

router = APIRouter()


# --- Connection / board helpers ---------------------------------------------

def _ws_upgrade_authorized(ws: "WebSocket") -> bool:
    """Authorize a WebSocket upgrade via the dashboard's canonical WS gate.

    Delegating to ``web_server._ws_auth_ok`` means this endpoint accepts whatever
    the core gate accepts in each mode (loopback ``?token=``, OAuth single-use
    ``?ticket=``, server-internal ``?internal=``) and can never drift from core
    auth. When ``web_server`` isn't importable (bare-FastAPI test harness) we
    accept so the tail loop stays testable.
    """
    try:
        from hermes_cli import web_server as _ws
    except Exception:
        return True
    return bool(_ws._ws_auth_ok(ws))


def _resolve_board(board: Optional[str]) -> Optional[str]:
    """Validate/normalise a board slug query param (400 malformed, 404 unknown).

    Returns ``None`` when the param was omitted so ``kb.connect()`` falls through
    to the active board.
    """
    if board is None or board == "":
        return None
    try:
        normed = kanban_db._normalize_board_slug(board)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if normed and normed != kanban_db.DEFAULT_BOARD and not kanban_db.board_exists(normed):
        raise HTTPException(status_code=404, detail=f"board {normed!r} does not exist")
    return normed


def _existing_board_slug(slug: str) -> str:
    """Normalise a path slug and require the board to exist (400 / 404)."""
    try:
        normed = kanban_db._normalize_board_slug(slug)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if not normed or not kanban_db.board_exists(normed):
        raise HTTPException(status_code=404, detail=f"board {slug!r} does not exist")
    return normed


def _conn(board: Optional[str] = None):
    """Open a kanban_db connection, creating the schema on first use.

    ``init_db`` is idempotent; running it here means a fresh install self-heals
    (no "no such table" if POST /tasks arrives before GET /board). ``board`` is
    the already-normalised slug; ``None`` resolves to the active board.
    """
    try:
        kanban_db.init_db(board=board)
    except Exception as exc:
        log.warning("kanban init_db failed: %s", exc)
    return kanban_db.connect(board=board)


@contextmanager
def _board_conn(board: Optional[str]) -> Iterator[tuple[Optional[str], sqlite3.Connection]]:
    """Resolve the ``board`` query param, open a connection, close it on exit."""
    board = _resolve_board(board)
    conn = _conn(board=board)
    try:
        yield board, conn
    finally:
        conn.close()


def _require_task(conn: sqlite3.Connection, task_id: str) -> kanban_db.Task:
    task = kanban_db.get_task(conn, task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"task {task_id} not found")
    return task


@contextmanager
def _value_error_400() -> Iterator[None]:
    """Map domain-layer ``ValueError`` (validation refusals) to a 400 response."""
    try:
        yield
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# --- Serialization helpers --------------------------------------------------

# Dashboard columns, left-to-right; "archived" is a filter toggle, not a column.
# Keep in sync with kanban_db.VALID_STATUSES — a status missing here (e.g.
# ``scheduled``) gets mis-bucketed into ``todo`` by the board fallback.
BOARD_COLUMNS: list[str] = [
    "triage", "todo", "scheduled", "ready", "running", "blocked", "review", "done",
]

_CARD_SUMMARY_PREVIEW_CHARS = 200


def _task_dict(task: kanban_db.Task, *, latest_summary: Optional[str] = None) -> dict[str, Any]:
    d = asdict(task)
    # Derived age metrics so the UI can colour stale cards without client deltas.
    try:
        d["age"] = kanban_db.task_age(task)
    except Exception:
        d["age"] = {"created_age_seconds": None, "started_age_seconds": None, "time_to_complete_seconds": None}
    # Latest non-null run summary (workers hand off via ``task_runs.summary``,
    # not ``tasks.result``); None until a run has produced one.
    d["latest_summary"] = latest_summary
    return d


def _event_dict(event: kanban_db.Event) -> dict[str, Any]:
    return asdict(event)


def _comment_dict(c: kanban_db.Comment) -> dict[str, Any]:
    return asdict(c)


def _attachment_dict(a: kanban_db.Attachment) -> dict[str, Any]:
    """``stored_path`` is the absolute on-disk path workers read; UI downloads by ``id``."""
    return {
        "id": a.id, "task_id": a.task_id, "filename": a.filename,
        "content_type": a.content_type, "size": a.size, "uploaded_by": a.uploaded_by,
        "stored_path": a.stored_path, "created_at": a.created_at,
    }


def _run_dict(r: kanban_db.Run) -> dict[str, Any]:
    return asdict(r)


def _compute_task_diagnostics(
    conn: sqlite3.Connection,
    task_ids: Optional[list[str]] = None,
) -> dict[str, list[dict]]:
    """Run the diagnostic rule engine (``kanban_diagnostics``) and return
    ``{task_id: [diagnostic_dict, ...]}``; tasks with no diagnostics are omitted.

    Three aggregate queries (tasks, events, runs) instead of N per-task lookups.
    Slurps every event/run for the board — fine for the dashboard's typical
    working set (hundreds of tasks); paginate if profiling shows a hotspot.
    """
    from hermes_cli.config import load_config

    diag_config = kd.config_from_runtime_config(load_config())

    if task_ids is not None:
        if not task_ids:
            return {}
        placeholders = ",".join(["?"] * len(task_ids))
        rows = conn.execute(
            f"SELECT * FROM tasks WHERE id IN ({placeholders})",
            tuple(task_ids),
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM tasks WHERE status != 'archived'").fetchall()

    if not rows:
        return {}

    row_ids = [r["id"] for r in rows]
    placeholders = ",".join(["?"] * len(row_ids))
    events_by_task: dict[str, list] = {tid: [] for tid in row_ids}
    for ev_row in conn.execute(
        f"SELECT * FROM task_events WHERE task_id IN ({placeholders}) ORDER BY id",
        tuple(row_ids),
    ).fetchall():
        events_by_task.setdefault(ev_row["task_id"], []).append(ev_row)
    runs_by_task: dict[str, list] = {tid: [] for tid in row_ids}
    for run_row in conn.execute(
        f"SELECT * FROM task_runs WHERE task_id IN ({placeholders}) ORDER BY id",
        tuple(row_ids),
    ).fetchall():
        runs_by_task.setdefault(run_row["task_id"], []).append(run_row)

    graph_by_task = kanban_db.task_graph_contexts(conn, row_ids)
    out: dict[str, list[dict]] = {}
    for r in rows:
        tid = r["id"]
        diags = kd.compute_task_diagnostics(
            r,
            events_by_task.get(tid, []),
            runs_by_task.get(tid, []),
            config=diag_config,
            graph=graph_by_task.get(tid),
        )
        if diags:
            out[tid] = [d.to_dict() for d in diags]
    return out


def _warnings_summary_from_diagnostics(diagnostics: list[dict]) -> Optional[dict]:
    """Compact card badge summary ``{count, kinds, latest_at, highest_severity}``;
    None when ``diagnostics`` is empty."""
    if not diagnostics:
        return None
    kinds: dict[str, int] = {}
    latest = 0
    highest_idx = -1
    highest_sev: Optional[str] = None
    count = 0
    for d in diagnostics:
        kinds[d["kind"]] = kinds.get(d["kind"], 0) + d.get("count", 1)
        count += d.get("count", 1)
        la = d.get("last_seen_at") or 0
        if la > latest:
            latest = la
        sev = d.get("severity")
        if sev in kd.SEVERITY_ORDER:
            idx = kd.SEVERITY_ORDER.index(sev)
            if idx > highest_idx:
                highest_idx = idx
                highest_sev = sev
    return {"count": count, "kinds": kinds, "latest_at": latest, "highest_severity": highest_sev}


def _attach_diagnostics(task_d: dict, diags: Optional[list[dict]]) -> None:
    """Full list goes in the payload (drawer renders without a second round-trip);
    the card badge only needs the summary."""
    if diags:
        task_d["diagnostics"] = diags
        task_d["warnings"] = _warnings_summary_from_diagnostics(diags)


def _links_for(conn: sqlite3.Connection, task_id: str) -> dict[str, list[str]]:
    """Return {'parents': [...], 'children': [...]} for a task."""
    parents = [
        r["parent_id"]
        for r in conn.execute(
            "SELECT parent_id FROM task_links WHERE child_id = ? ORDER BY parent_id",
            (task_id,),
        )
    ]
    children = [
        r["child_id"]
        for r in conn.execute(
            "SELECT child_id FROM task_links WHERE parent_id = ? ORDER BY child_id",
            (task_id,),
        )
    ]
    return {"parents": parents, "children": children}


# --- GET /board -------------------------------------------------------------

@router.get("/board")
def get_board(
    tenant: Optional[str] = Query(None, description="Filter to a single tenant"),
    include_archived: bool = Query(False),
    board: Optional[str] = Query(None, description="Kanban board slug (omit for current)"),
    workflow_template_id: Optional[str] = Query(
        None, description="Restrict to tasks using this workflow template id",
    ),
    current_step_key: Optional[str] = Query(None, description="Restrict to tasks at this workflow step key"),
):
    """Return the full board grouped by status column.

    Omitting ``board`` falls through to the active board (``HERMES_KANBAN_BOARD``
    env → on-disk ``current`` pointer → ``default``).
    """
    with _board_conn(board) as (board, conn):
        tasks = kanban_db.list_tasks(
            conn,
            tenant=tenant,
            include_archived=include_archived,
            workflow_template_id=workflow_template_id,
            current_step_key=current_step_key,
        )
        # Link / comment / progress rollups are each one aggregate query rather
        # than N per-task lookups.
        link_counts: dict[str, dict[str, int]] = {}
        for row in conn.execute("SELECT parent_id, child_id FROM task_links").fetchall():
            link_counts.setdefault(row["parent_id"], {"parents": 0, "children": 0})["children"] += 1
            link_counts.setdefault(row["child_id"], {"parents": 0, "children": 0})["parents"] += 1

        comment_counts: dict[str, int] = {
            r["task_id"]: r["n"]
            for r in conn.execute(
                "SELECT task_id, COUNT(*) AS n FROM task_comments GROUP BY task_id"
            )
        }

        # Per parent: children done / total, rendered as "N/M".
        progress: dict[str, dict[str, int]] = {}
        for row in conn.execute(
            "SELECT l.parent_id AS pid, t.status AS cstatus "
            "FROM task_links l JOIN tasks t ON t.id = l.child_id"
        ).fetchall():
            p = progress.setdefault(row["pid"], {"done": 0, "total": 0})
            p["total"] += 1
            if row["cstatus"] == "done":
                p["done"] += 1

        diagnostics_per_task = _compute_task_diagnostics(conn, task_ids=None)

        latest_event_id = conn.execute(
            "SELECT COALESCE(MAX(id), 0) AS m FROM task_events"
        ).fetchone()["m"]

        columns: dict[str, list[dict]] = {c: [] for c in BOARD_COLUMNS}
        if include_archived:
            columns["archived"] = []

        # One window-function query for latest summaries (avoids N+1); cards get
        # a truncated preview, the full text comes from /tasks/:id.
        summary_map = kanban_db.latest_summaries(conn, [t.id for t in tasks])

        for t in tasks:
            full = summary_map.get(t.id)
            preview = (full[:_CARD_SUMMARY_PREVIEW_CHARS] if full else None)
            d = _task_dict(t, latest_summary=preview)
            d["link_counts"] = link_counts.get(t.id, {"parents": 0, "children": 0})
            d["comment_count"] = comment_counts.get(t.id, 0)
            d["progress"] = progress.get(t.id)  # None when the task has no children
            _attach_diagnostics(d, diagnostics_per_task.get(t.id))
            col = t.status if t.status in columns else "todo"
            columns[col].append(d)

        # Per-column ordering (priority DESC, created_at ASC) comes from list_tasks.
        tenants = [
            r["tenant"]
            for r in conn.execute(
                "SELECT DISTINCT tenant FROM tasks WHERE tenant IS NOT NULL ORDER BY tenant"
            )
        ]
        assignees = [
            r["assignee"]
            for r in conn.execute(
                "SELECT DISTINCT assignee FROM tasks WHERE assignee IS NOT NULL "
                "AND status != 'archived' ORDER BY assignee"
            )
        ]

        return {
            "columns": [{"name": name, "tasks": columns[name]} for name in columns],
            "tenants": tenants,
            "assignees": assignees,
            "latest_event_id": int(latest_event_id),
            "now": int(time.time()),
        }


# --- GET /tasks/:id ---------------------------------------------------------

@router.get("/tasks/{task_id}")
def get_task(
    task_id: str,
    board: Optional[str] = Query(None),
    run_state_type: Optional[str] = Query(
        None, description="With run_state_name: filter runs by column 'status' or 'outcome'",
    ),
    run_state_name: Optional[str] = Query(
        None, description="With run_state_type: exact value for that run column",
    ),
):
    with _board_conn(board) as (board, conn):
        if (run_state_type is None) ^ (run_state_name is None):
            raise HTTPException(
                status_code=400,
                detail="run_state_type and run_state_name must be passed together or omitted",
            )
        if run_state_type is not None and run_state_type not in ("status", "outcome"):
            raise HTTPException(status_code=400, detail="run_state_type must be 'status' or 'outcome'")
        task = _require_task(conn, task_id)
        # Drawer returns the FULL summary (cards on /board carry a 200-char preview).
        full_summary = kanban_db.latest_summary(conn, task_id)
        task_d = _task_dict(task, latest_summary=full_summary)
        links = _links_for(conn, task_id)
        child_ids = links["children"]
        child_summaries = kanban_db.latest_summaries(conn, child_ids)
        child_results = []
        for child_id in child_ids:
            child = kanban_db.get_task(conn, child_id)
            if child is None:
                continue
            child_results.append({
                "id": child.id,
                "title": child.title,
                "status": child.status,
                "latest_summary": child_summaries.get(child.id),
                "result": child.result,
            })
        diags = _compute_task_diagnostics(conn, task_ids=[task_id])
        _attach_diagnostics(task_d, diags.get(task_id) or [])
        return {
            "task": task_d,
            "comments": [_comment_dict(c) for c in kanban_db.list_comments(conn, task_id)],
            "events": [_event_dict(e) for e in kanban_db.list_events(conn, task_id)],
            "attachments": [_attachment_dict(a) for a in kanban_db.list_attachments(conn, task_id)],
            "links": links,
            "child_results": child_results,
            "runs": [
                _run_dict(r)
                for r in kanban_db.list_runs(
                    conn,
                    task_id,
                    state_type=run_state_type,
                    state_name=run_state_name,
                )
            ],
        }


# --- POST /tasks ------------------------------------------------------------

class CreateTaskBody(BaseModel):
    title: str
    body: Optional[str] = None
    assignee: Optional[str] = None
    tenant: Optional[str] = None
    priority: int = 0
    workspace_kind: str = "scratch"
    workspace_path: Optional[str] = None
    parents: list[str] = Field(default_factory=list)
    triage: bool = False
    idempotency_key: Optional[str] = None
    max_runtime_seconds: Optional[int] = None
    skills: Optional[list[str]] = None
    goal_mode: bool = False
    goal_max_turns: Optional[int] = None
    model_override: Optional[str] = None
    provider_override: Optional[str] = None
    # Thinking depth (none|minimal|…|ultra); None inherits the profile's own level.
    reasoning_effort: Optional[str] = None
    # When omitted, create_task inherits the board's scoped project (if any).
    project_id: Optional[str] = None


@router.post("/tasks")
def create_task(payload: CreateTaskBody, board: Optional[str] = Query(None)):
    with _board_conn(board) as (board, conn), _value_error_400():
        # CreateTaskBody field names match create_task's keyword parameters.
        task_id = kanban_db.create_task(
            conn, created_by="dashboard", board=board, **payload.model_dump(),
        )
        task = kanban_db.get_task(conn, task_id)
        body: dict[str, Any] = {"task": _task_dict(task) if task else None}
        # Dispatcher-presence warning so the UI can banner a ready+assigned task
        # that would otherwise sit idle (no gateway / dispatch_in_gateway=false).
        # triage/todo are expected to wait; unassigned tasks can't dispatch anyway.
        if task and task.status == "ready" and task.assignee:
            try:
                from hermes_cli.kanban import _check_dispatcher_presence
                from hermes_constants import get_hermes_home

                # Probe the request's active home: the dashboard backend may run
                # under a different HERMES_HOME than the board's profile.
                running, message = _check_dispatcher_presence(hermes_home=get_hermes_home())
                if not running and message:
                    body["warning"] = message
            except Exception:
                pass  # probe failure must never block the create itself
        return body


# --- Attachments — upload / list / download / delete ------------------------
# Size cap, filename sanitiser, and collision resolver live in ``kanban_db`` so
# the dashboard, agent toolset, and CLI share one implementation.
# ``_safe_attachment_name`` raises ``ValueError`` → mapped to 400 below.

@router.get("/tasks/{task_id}/attachments")
def list_task_attachments(task_id: str, board: Optional[str] = Query(None)):
    with _board_conn(board) as (board, conn):
        _require_task(conn, task_id)
        return {
            "attachments": [_attachment_dict(a) for a in kanban_db.list_attachments(conn, task_id)]
        }


@router.post("/tasks/{task_id}/attachments")
async def upload_task_attachment(
    task_id: str,
    file: UploadFile = File(...),
    board: Optional[str] = Query(None),
    uploaded_by: Optional[str] = Form(None),
):
    """Store an upload under ``attachments_root(board)/<task_id>/`` with a
    sanitised, collision-resolved name and record its metadata."""
    with _board_conn(board) as (board, conn), _value_error_400():
        _require_task(conn, task_id)
        safe_name = _safe_attachment_name(file.filename or "")

        dest_dir = kanban_db.task_attachments_dir(task_id, board=board)
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = _collision_free_path(dest_dir, safe_name)  # foo.pdf → foo (1).pdf …
        candidate = dest_path.name

        # Stream in chunks with a hard size cap so one upload can't fill the disk.
        total = 0
        try:
            with open(dest_path, "wb") as out:
                while True:
                    chunk = await file.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > KANBAN_ATTACHMENT_MAX_BYTES:
                        out.close()
                        dest_path.unlink(missing_ok=True)
                        raise HTTPException(
                            status_code=413,
                            detail=(
                                f"attachment exceeds {KANBAN_ATTACHMENT_MAX_BYTES // (1024 * 1024)} MB limit"
                            ),
                        )
                    out.write(chunk)
        except HTTPException:
            raise
        except OSError as exc:
            raise HTTPException(status_code=500, detail=f"failed to store attachment: {exc}")

        att_id = kanban_db.add_attachment(
            conn,
            task_id,
            filename=candidate,
            stored_path=str(dest_path.resolve()),
            content_type=file.content_type,
            size=total,
            uploaded_by=(uploaded_by or "dashboard"),
        )
        att = kanban_db.get_attachment(conn, att_id)
        return {"attachment": _attachment_dict(att) if att else None}


@router.get("/attachments/{attachment_id}")
def download_attachment(attachment_id: int, board: Optional[str] = Query(None)):
    with _board_conn(board) as (board, conn):
        att = kanban_db.get_attachment(conn, attachment_id)
        if att is None:
            raise HTTPException(status_code=404, detail="attachment not found")
        # Defense in depth against a tampered DB row: the blob must still live
        # under the board's attachments root.
        root = kanban_db.attachments_root(board=board).resolve()
        try:
            stored = Path(att.stored_path).resolve()
            stored.relative_to(root)
        except (ValueError, OSError):
            raise HTTPException(status_code=404, detail="attachment file unavailable")
        if not stored.is_file():
            raise HTTPException(status_code=404, detail="attachment file missing on disk")
        return FileResponse(
            path=str(stored),
            filename=att.filename,
            media_type=att.content_type or "application/octet-stream",
        )


@router.delete("/attachments/{attachment_id}")
def remove_attachment(attachment_id: int, board: Optional[str] = Query(None)):
    with _board_conn(board) as (board, conn):
        att = kanban_db.delete_attachment(conn, attachment_id)
        if att is None:
            raise HTTPException(status_code=404, detail="attachment not found")
        return {"ok": True, "id": attachment_id}


# --- PATCH /tasks/:id  and  POST /tasks/bulk ---------------------------------

class UpdateTaskBody(BaseModel):
    status: Optional[str] = None
    assignee: Optional[str] = None
    priority: Optional[int] = None
    title: Optional[str] = None
    body: Optional[str] = None
    result: Optional[str] = None
    block_reason: Optional[str] = None
    # Structured handoff fields forwarded to complete_task on -> 'done'
    # (parity with ``hermes kanban complete --summary/--metadata``).
    summary: Optional[str] = None
    metadata: Optional[dict] = None
    # Model/provider override. In a PATCH ``None`` means "field not sent", so
    # ``clear_model_override=True`` is the explicit clear signal.
    model_override: Optional[str] = None
    provider_override: Optional[str] = None
    clear_model_override: bool = False
    # Thinking depth: ``"none"`` is a VALUE (thinking off); clear separately so
    # dropping a model override doesn't silently reset the depth.
    reasoning_effort: Optional[str] = None
    clear_reasoning_effort: bool = False


class BulkTaskBody(BaseModel):
    ids: list[str]
    status: Optional[str] = None
    assignee: Optional[str] = None  # "" or None = unassign
    priority: Optional[int] = None
    archive: bool = False
    result: Optional[str] = None
    summary: Optional[str] = None
    metadata: Optional[dict] = None
    reclaim_first: bool = False
    # Same semantics as UpdateTaskBody.
    model_override: Optional[str] = None
    provider_override: Optional[str] = None
    clear_model_override: bool = False
    reasoning_effort: Optional[str] = None
    clear_reasoning_effort: bool = False


class _StatusRejected(Exception):
    """A status the dashboard may never set directly (message is user-facing)."""


class _UnknownStatus(_StatusRejected):
    pass


_RUNNING_DIRECT_MSG = "Cannot set status to 'running' directly; use the dispatcher/claim path"


def _reopen_if_review(conn, task_id: str, current) -> Optional[bool]:
    """Route a task leaving ``review`` through ``reopen_review_task`` (stale-run
    recovery, parent re-gate, ``review_reopened`` event) instead of a raw status
    write. Returns None when the task isn't in review so callers fall through."""
    if current is not None and getattr(current, "status", None) == "review":
        return kanban_db.reopen_review_task(conn, task_id)
    return None


def _to_ready(conn, task_id, p) -> bool:
    # Re-open blocked/scheduled via unblock; "changes requested" (review -> ready)
    # via reopen_review_task; otherwise a direct drag-drop write (todo -> ready).
    current = kanban_db.get_task(conn, task_id)
    if current and current.status in ("blocked", "scheduled"):
        return kanban_db.unblock_task(conn, task_id)
    reopened = _reopen_if_review(conn, task_id, current)
    return reopened if reopened is not None else _set_status_direct(conn, task_id, "ready")


def _to_todo_or_triage(conn, task_id, p, s) -> bool:
    # Only review -> todo needs the reopen transition; triage skips the query.
    current = kanban_db.get_task(conn, task_id) if s == "todo" else None
    reopened = _reopen_if_review(conn, task_id, current)
    return reopened if reopened is not None else _set_status_direct(conn, task_id, s)


# Status verb dispatch shared by PATCH /tasks/{id} and POST /tasks/bulk. Each
# handler is (conn, task_id, payload) -> ok. ``review`` routes through
# request_review (never a block, so it can't trip unblock-loop detection);
# ``force=True`` because a dashboard action is an explicit human override of a
# live worker claim.
_STATUS_HANDLERS: dict[str, Any] = {
    "done": lambda conn, tid, p: kanban_db.complete_task(
        conn, tid, result=p.result, summary=p.summary, metadata=p.metadata,
    ),
    "blocked": lambda conn, tid, p: kanban_db.block_task(
        conn, tid, reason=getattr(p, "block_reason", None),
    ),
    "scheduled": lambda conn, tid, p: kanban_db.schedule_task(
        conn, tid, reason=getattr(p, "block_reason", None),
    ),
    "review": lambda conn, tid, p: kanban_db.request_review(
        conn, tid, summary=p.summary, metadata=p.metadata,
        reviewer=(p.assignee or None), force=True,
    ),
    "ready": _to_ready,
    "todo": lambda conn, tid, p: _to_todo_or_triage(conn, tid, p, "todo"),
    "triage": lambda conn, tid, p: _to_todo_or_triage(conn, tid, p, "triage"),
}


def _apply_status(conn, task_id: str, s: str, p) -> bool:
    if s == "running":
        raise _StatusRejected(_RUNNING_DIRECT_MSG)
    handler = _STATUS_HANDLERS.get(s)
    if handler is None:
        raise _UnknownStatus(s)
    return handler(conn, task_id, p)


def _set_priority(conn, task_id: str, priority: int, board: Optional[str]) -> None:
    with kanban_db.write_txn(conn):
        conn.execute("UPDATE tasks SET priority = ? WHERE id = ?", (int(priority), task_id))
        conn.execute(
            "INSERT INTO task_events (task_id, kind, payload, created_at) "
            "VALUES (?, 'reprioritized', ?, ?)",
            (task_id, json.dumps({"priority": int(priority)}), int(time.time())),
        )
    # Mutation-boundary observer: this direct-SQL write bypasses every kanban_db
    # mutator, so report it here — after the txn commits.
    kanban_db.notify_task_updated(conn, task_id, ("priority",), board=board)


def _apply_model_override(conn, task_id: str, p) -> bool:
    """Raises ValueError/RuntimeError from kanban_db for the caller to map."""
    new_model = (None if p.clear_model_override else (p.model_override or "").strip() or None)
    return kanban_db.set_model_override(conn, task_id, new_model, provider=p.provider_override)


def _apply_reasoning_effort(conn, task_id: str, p) -> bool:
    new_effort = None if p.clear_reasoning_effort else p.reasoning_effort
    return kanban_db.set_reasoning_effort(conn, task_id, new_effort)


def _wants_model_override(p) -> bool:
    return p.clear_model_override or p.model_override is not None


def _wants_reasoning_effort(p) -> bool:
    return p.clear_reasoning_effort or p.reasoning_effort is not None


@router.patch("/tasks/{task_id}")
def update_task(task_id: str, payload: UpdateTaskBody, board: Optional[str] = Query(None)):
    with _board_conn(board) as (board, conn):
        _require_task(conn, task_id)

        # For a combined assignee+review patch, request_review must capture the
        # current implementer before the task is routed to the reviewer.
        review_assignee_deferred = (payload.status == "review" and payload.assignee is not None)
        if payload.assignee is not None and not review_assignee_deferred:
            try:
                ok = kanban_db.assign_task(conn, task_id, payload.assignee or None)
            except RuntimeError as e:
                raise HTTPException(status_code=409, detail=str(e))
            if not ok:
                raise HTTPException(status_code=404, detail="task not found")

        if payload.status is not None:
            s = payload.status
            if s == "archived":
                ok = kanban_db.archive_task(conn, task_id)
            else:
                try:
                    ok = _apply_status(conn, task_id, s, payload)
                except _UnknownStatus:
                    raise HTTPException(status_code=400, detail=f"unknown status: {s}")
                except _StatusRejected as e:
                    raise HTTPException(status_code=400, detail=str(e))
                if s == "review" and ok and review_assignee_deferred and not payload.assignee:
                    ok = kanban_db.assign_task(conn, task_id, None)
            if not ok:
                # For ``ready``, name the blocking parent(s) so the dashboard can
                # render an actionable toast instead of a silent no-op.
                if s == "ready":
                    blockers = _parents_blocking_ready(conn, task_id)
                    if blockers:
                        names = ", ".join(
                            f"{p['title']!r} ({p['id']}, status={p['status']})"
                            for p in blockers
                        )
                        raise HTTPException(
                            status_code=409,
                            detail=(
                                f"Cannot move to 'ready': blocked by parent(s) "
                                f"not done — {names}"
                            ),
                        )
                raise HTTPException(
                    status_code=409,
                    detail=f"status transition to {s!r} not valid from current state",
                )

        for wanted, apply in (
            (_wants_model_override, _apply_model_override),
            (_wants_reasoning_effort, _apply_reasoning_effort),
        ):
            if wanted(payload):
                try:
                    ok = apply(conn, task_id, payload)
                except (ValueError, RuntimeError) as e:
                    raise HTTPException(status_code=400, detail=str(e))
                if not ok:
                    raise HTTPException(status_code=404, detail="task not found")

        if payload.priority is not None:
            _set_priority(conn, task_id, payload.priority, board)

        if payload.title is not None or payload.body is not None:
            with kanban_db.write_txn(conn):
                sets, vals = [], []
                if payload.title is not None:
                    if not payload.title.strip():
                        raise HTTPException(status_code=400, detail="title cannot be empty")
                    sets.append("title = ?")
                    vals.append(payload.title.strip())
                if payload.body is not None:
                    sets.append("body = ?")
                    vals.append(payload.body)
                vals.append(task_id)
                conn.execute(f"UPDATE tasks SET {', '.join(sets)} WHERE id = ?", vals)
                conn.execute(
                    "INSERT INTO task_events (task_id, kind, payload, created_at) "
                    "VALUES (?, 'edited', NULL, ?)",
                    (task_id, int(time.time())),
                )
            # Post-commit mutation observer; field names only — values never
            # leave the DB via this payload.
            kanban_db.notify_task_updated(
                conn, task_id,
                [f for f in ("title", "body") if getattr(payload, f) is not None],
                board=board,
            )

        updated = kanban_db.get_task(conn, task_id)
        return {"task": _task_dict(updated) if updated else None}


@router.delete("/tasks/{task_id}")
def delete_task(task_id: str, board: Optional[str] = Query(None)):
    with _board_conn(board) as (board, conn):
        ok = kanban_db.delete_task(conn, task_id)
        if not ok:
            raise HTTPException(status_code=404, detail=f"task {task_id} not found")
        return {"deleted": True, "task_id": task_id}


def _parents_blocking_ready(conn: sqlite3.Connection, task_id: str) -> list:
    """Parent rows (id, title, status) not ``done`` that block promotion to ``ready``."""
    rows = conn.execute(
        "SELECT t.id, t.title, t.status FROM tasks t "
        "JOIN task_links l ON l.parent_id = t.id "
        "WHERE l.child_id = ? AND t.status != 'done'",
        (task_id,),
    ).fetchall()
    return [{"id": r["id"], "title": r["title"], "status": r["status"]} for r in rows]


def _set_status_direct(conn: sqlite3.Connection, task_id: str, new_status: str) -> bool:
    """Direct status write for drag-drop moves not covered by the structured
    complete/block/unblock/archive verbs (todo<->ready, running<->ready).
    Appends a ``status`` event for the live feed.

    Leaving ``running`` closes the active run with outcome='reclaimed' so attempt
    history isn't orphaned, and the worker is terminated only AFTER the txn
    commits (events must be durable before the kill).
    """
    terminations: list[tuple[Optional[int], Optional[str]]] = []
    effective_status = new_status
    with kanban_db.write_txn(conn):
        prev = conn.execute(
            "SELECT status, current_run_id, worker_pid, claim_lock "
            "FROM tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
        if prev is None:
            return False

        if prev["status"] == "running" and new_status == "ready":
            resume_status = kanban_db._retry_status_for_run(conn, task_id, prev["current_run_id"])
            if resume_status == "review":
                effective_status = (
                    "review"
                    if kanban_db._parents_satisfied(conn, task_id)
                    else "todo"
                )

        # Never promote to 'ready' unless all parents are done — otherwise the
        # dispatcher spawns a child whose upstream work hasn't completed.
        if effective_status == "ready":
            parent_statuses = conn.execute(
                "SELECT t.status FROM tasks t "
                "JOIN task_links l ON l.parent_id = t.id "
                "WHERE l.child_id = ?",
                (task_id,),
            ).fetchall()
            if parent_statuses and not all(
                p["status"] in {"done", "archived"} for p in parent_statuses
            ):
                return False

        was_running = prev["status"] == "running"
        reopening_satisfied_parent = (
            prev["status"] in {"done", "archived"}
            and effective_status not in {"done", "archived"}
        )

        cur = conn.execute(
            "UPDATE tasks SET status = ?, "
            "  claim_lock = CASE WHEN ? = 'running' THEN claim_lock ELSE NULL END, "
            "  claim_expires = CASE WHEN ? = 'running' THEN claim_expires ELSE NULL END, "
            "  worker_pid = CASE WHEN ? = 'running' THEN worker_pid ELSE NULL END "
            "WHERE id = ?",
            (effective_status,) * 4 + (task_id,),
        )
        if cur.rowcount != 1:
            return False
        run_id = None
        if was_running and effective_status != "running" and prev["current_run_id"]:
            run_id = kanban_db._end_run(
                conn, task_id,
                outcome="reclaimed", status="reclaimed",
                summary=f"status changed to {effective_status} (dashboard/direct)",
            )
            terminations.append((prev["worker_pid"], prev["claim_lock"]))
        conn.execute(
            "INSERT INTO task_events (task_id, run_id, kind, payload, created_at) "
            "VALUES (?, ?, 'status', ?, ?)",
            (
                task_id,
                run_id,
                json.dumps({"status": effective_status, "requested_status": new_status}),
                int(time.time()),
            ),
        )
        if reopening_satisfied_parent:
            # Domain-layer invalidation composes via a savepoint inside our txn
            # and hands back worker terminations to perform post-commit.
            result = kanban_db.invalidate_descendants_for_parent_reopen(
                conn, task_id, author="dashboard",
            )
            terminations.extend(result["terminations"])
    for pid, claim_lock in terminations:
        kanban_db._terminate_reclaimed_worker(pid, claim_lock)
    # Re-opening something may have made children stale.
    if effective_status in {"done", "ready", "review"}:
        kanban_db.recompute_ready(conn)
    return True


# --- Comments / links -------------------------------------------------------

class CommentBody(BaseModel):
    body: str
    author: Optional[str] = "dashboard"


@router.post("/tasks/{task_id}/comments")
def add_comment(task_id: str, payload: CommentBody, board: Optional[str] = Query(None)):
    if not payload.body.strip():
        raise HTTPException(status_code=400, detail="body is required")
    with _board_conn(board) as (board, conn):
        _require_task(conn, task_id)
        kanban_db.add_comment(
            conn, task_id, author=payload.author or "dashboard", body=payload.body,
        )
        return {"ok": True}


class LinkBody(BaseModel):
    parent_id: str
    child_id: str


@router.post("/links")
def add_link(payload: LinkBody, board: Optional[str] = Query(None)):
    with _board_conn(board) as (board, conn), _value_error_400():
        kanban_db.link_tasks(conn, payload.parent_id, payload.child_id)
        return {"ok": True}


@router.delete("/links")
def delete_link(
    parent_id: str = Query(...),
    child_id: str = Query(...),
    board: Optional[str] = Query(None),
):
    with _board_conn(board) as (board, conn):
        ok = kanban_db.unlink_tasks(conn, parent_id, child_id)
        return {"ok": bool(ok)}


@router.post("/tasks/bulk")
def bulk_update(payload: BulkTaskBody, board: Optional[str] = Query(None)):
    """Apply the same patch to every id. Independent iteration — per-task
    failures don't abort siblings; returns per-id outcome for partials."""
    ids = [i for i in (payload.ids or []) if i]
    if not ids:
        raise HTTPException(status_code=400, detail="ids is required")
    results: list[dict] = []
    with _board_conn(board) as (board, conn):
        for tid in ids:
            entry: dict[str, Any] = {"id": tid, "ok": True}
            try:
                if kanban_db.get_task(conn, tid) is None:
                    entry.update(ok=False, error="not found")
                    results.append(entry)
                    continue
                if payload.archive and not kanban_db.archive_task(conn, tid):
                    entry.update(ok=False, error="archive refused")
                if payload.status is not None and not payload.archive:
                    s = payload.status
                    try:
                        ok = _apply_status(conn, tid, s, payload)
                    except _UnknownStatus:
                        entry.update(ok=False, error=f"unknown status {s!r}")
                        results.append(entry)
                        continue
                    except _StatusRejected as e:
                        entry.update(ok=False, error=str(e))
                        results.append(entry)
                        continue
                    if not ok:
                        entry.update(ok=False, error=f"transition to {s!r} refused")
                if payload.assignee is not None:
                    try:
                        if payload.reclaim_first:
                            ok = kanban_db.reassign_task(
                                conn, tid, payload.assignee or None,
                                reclaim_first=True,
                            )
                        else:
                            ok = kanban_db.assign_task(conn, tid, payload.assignee or None)
                        if not ok:
                            entry.update(ok=False, error="assign refused")
                    except RuntimeError as e:
                        entry.update(ok=False, error=str(e))
                if payload.priority is not None:
                    _set_priority(conn, tid, payload.priority, board)
                for wanted, apply, refused in (
                    (_wants_model_override, _apply_model_override, "model override refused"),
                    (_wants_reasoning_effort, _apply_reasoning_effort, "reasoning override refused"),
                ):
                    if wanted(payload):
                        try:
                            if not apply(conn, tid, payload):
                                entry.update(ok=False, error=refused)
                        except (ValueError, RuntimeError) as e:
                            entry.update(ok=False, error=str(e))
            except Exception as e:  # one bad id shouldn't kill the batch
                entry.update(ok=False, error=str(e))
            results.append(entry)
        return {"results": results}


# --- Diagnostics — fleet-wide distress signals (see kanban_diagnostics) ------

@router.get("/diagnostics")
def list_diagnostics(
    board: Optional[str] = Query(None, description="Kanban board slug (omit for current)"),
    severity: Optional[str] = Query(None, description="Filter by severity: warning|error|critical"),
):
    """Return ``[{task_id, task_title, task_status, task_assignee, diagnostics}]``
    for every task with an active diagnostic, highest severity first then most
    recent. Also consumed by ``hermes kanban diagnostics`` when the dashboard runs."""
    with _board_conn(board) as (board, conn):
        diags_by_task = _compute_task_diagnostics(conn, task_ids=None)
        if severity and diags_by_task:
            diags_by_task = {
                tid: keep
                for tid, dl in diags_by_task.items()
                if (keep := [d for d in dl if kd.severity_at_or_above(d.get("severity"), severity)])
            }
        if not diags_by_task:
            return {"diagnostics": [], "count": 0}

        ids = list(diags_by_task.keys())
        placeholders = ",".join(["?"] * len(ids))
        rows = {
            r["id"]: r
            for r in conn.execute(
                f"SELECT id, title, status, assignee FROM tasks WHERE id IN ({placeholders})",
                tuple(ids),
            ).fetchall()
        }

        out = []
        for tid, dl in diags_by_task.items():
            r = rows.get(tid)
            out.append({
                "task_id": tid,
                "task_title": r["title"] if r else None,
                "task_status": r["status"] if r else None,
                "task_assignee": r["assignee"] if r else None,
                "diagnostics": dl,
            })
        sev_idx = {s: i for i, s in enumerate(kd.SEVERITY_ORDER)}

        def _sort_key(row):
            top = row["diagnostics"][0]
            return (-sev_idx.get(top.get("severity"), -1), -(top.get("last_seen_at") or 0))
        out.sort(key=_sort_key)

        return {"diagnostics": out, "count": sum(len(d["diagnostics"]) for d in out)}


# --- Worker visibility — active-worker list, per-run inspect/terminate -------

try:
    import psutil as _psutil
except ImportError:
    _psutil = None  # type: ignore[assignment]


@router.get("/workers/active")
def list_active_workers(
    board: Optional[str] = Query(None, description="Kanban board slug (omit for current)"),
):
    """Every running worker: an open ``task_runs`` row with a ``worker_pid`` whose
    task is ``running``. Returns ``{workers, count, checked_at}``."""
    with _board_conn(board) as (board, conn):
        rows = conn.execute(
            """
            SELECT
                r.id          AS run_id,
                r.task_id,
                t.title       AS task_title,
                t.status      AS task_status,
                t.assignee    AS task_assignee,
                r.profile,
                r.worker_pid,
                r.started_at,
                r.claim_lock,
                r.claim_expires,
                r.last_heartbeat_at,
                r.max_runtime_seconds
            FROM task_runs r
            JOIN tasks t ON t.id = r.task_id
            WHERE r.ended_at IS NULL
              AND r.worker_pid IS NOT NULL
              AND t.status = 'running'
            ORDER BY r.started_at ASC
            """,
        ).fetchall()
        workers = [dict(row) for row in rows]
        return {"workers": workers, "count": len(workers), "checked_at": int(time.time())}


def _require_run(conn, run_id: int) -> kanban_db.Run:
    r = kanban_db.get_run(conn, run_id)
    if r is None:
        raise HTTPException(status_code=404, detail=f"run {run_id} not found")
    return r


@router.get("/runs/{run_id}")
def get_run_endpoint(
    run_id: int,
    board: Optional[str] = Query(None, description="Kanban board slug (omit for current)"),
):
    """``{run: {...}}`` with the same serialisation as ``GET /tasks/{id}``; 404 if unknown."""
    with _board_conn(board) as (board, conn):
        return {"run": _run_dict(_require_run(conn, run_id))}


@router.get("/runs/{run_id}/inspect")
def inspect_run_endpoint(
    run_id: int,
    board: Optional[str] = Query(None, description="Kanban board slug (omit for current)"),
):
    """Live psutil stats for a run's worker process.

    ``{alive: false, reason}`` when the run ended, has no pid, the process is
    gone, or psutil is missing; ``access_denied`` style errors are reported
    inline rather than as a 500.
    """
    with _board_conn(board) as (board, conn):
        r = _require_run(conn, run_id)

    if r.ended_at is not None:
        return {"run_id": run_id, "alive": False, "reason": "run already ended"}
    if r.worker_pid is None:
        return {"run_id": run_id, "alive": False, "reason": "no worker_pid recorded"}

    pid = r.worker_pid

    if _psutil is None:
        return {"run_id": run_id, "alive": False, "pid": pid, "reason": "psutil not available"}

    try:
        proc = _psutil.Process(pid)
        info = proc.as_dict(attrs=[
            "cpu_percent", "memory_info", "num_threads",
            "status", "create_time", "cmdline",
        ])
        try:
            num_fds = proc.num_fds()
        except AttributeError:  # POSIX-only
            num_fds = None
        mem = info.get("memory_info")
        return {
            "run_id": run_id,
            "alive": True,
            "pid": pid,
            "cpu_percent": info.get("cpu_percent"),
            "memory_rss_bytes": mem.rss if mem else None,
            "memory_vms_bytes": mem.vms if mem else None,
            "num_threads": info.get("num_threads"),
            "num_fds": num_fds,
            "status": info.get("status"),
            "create_time": info.get("create_time"),
            "cmdline": info.get("cmdline"),
        }
    except _psutil.NoSuchProcess:
        return {"run_id": run_id, "alive": False, "pid": pid, "reason": "process not found"}
    except _psutil.AccessDenied:
        return {"run_id": run_id, "alive": True, "pid": pid, "error": "access denied"}


class TerminateRunBody(BaseModel):
    reason: Optional[str] = None


@router.post("/runs/{run_id}/terminate")
def terminate_run_endpoint(
    run_id: int,
    payload: TerminateRunBody,
    board: Optional[str] = Query(None, description="Kanban board slug (omit for current)"),
):
    """Terminate the worker behind an in-flight run via ``reclaim_task`` so the
    SIGTERM->SIGKILL flow, run bookkeeping, and event log match
    ``POST /tasks/{id}/reclaim``. 404 unknown run; 409 already ended / not reclaimable."""
    with _board_conn(board) as (board, conn):
        r = _require_run(conn, run_id)
        if r.ended_at is not None:
            raise HTTPException(status_code=409, detail=f"run {run_id} already ended")
        ok = kanban_db.reclaim_task(conn, r.task_id, reason=payload.reason)
        if not ok:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"cannot terminate run {run_id}: task {r.task_id} is no "
                    "longer in a reclaimable state"
                ),
            )
        return {"ok": True, "run_id": run_id, "task_id": r.task_id}


# --- Recovery actions — reclaim / specify / reassign / estimate -------------

class ReclaimBody(BaseModel):
    reason: Optional[str] = None


@router.post("/tasks/{task_id}/reclaim")
def reclaim_task_endpoint(task_id: str, payload: ReclaimBody, board: Optional[str] = Query(None)):
    """Release an active worker claim without waiting for the claim TTL
    (``hermes kanban reclaim <task_id> --reason ...``)."""
    with _board_conn(board) as (board, conn):
        ok = kanban_db.reclaim_task(conn, task_id, reason=payload.reason)
        if not ok:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"cannot reclaim {task_id}: not in a claimable state "
                    "(not running, or unknown id)"
                ),
            )
        return {"ok": True, "task_id": task_id}


class SpecifyBody(BaseModel):
    """Only the author is configurable; model + prompt come from
    ``auxiliary.triage_specifier`` in config.yaml, same as the CLI."""

    author: Optional[str] = None


@router.post("/tasks/{task_id}/specify")
def specify_task_endpoint(task_id: str, payload: SpecifyBody, board: Optional[str] = Query(None)):
    """Flesh out a triage task via the auxiliary LLM and promote it to ``todo``
    (``hermes kanban specify``). Returns ``{ok, task_id, reason, new_title}``; a
    non-OK outcome is NOT an HTTP error — the UI renders the reason inline.

    Sync ``def`` so the slow LLM call runs in FastAPI's threadpool.
    """
    board = _resolve_board(board)
    # Context-local board pin (not the process-global HERMES_KANBAN_BOARD env
    # var): concurrent threadpool requests for different boards would otherwise
    # race on the shared env var and cross-write.
    with kanban_db.scoped_current_board(board or kanban_db.DEFAULT_BOARD):
        from hermes_cli import kanban_specify  # lazy: missing aux client must not break plugin load

        outcome = kanban_specify.specify_task(task_id, author=(payload.author or None))

    return {
        "ok": bool(outcome.ok),
        "task_id": outcome.task_id,
        "reason": outcome.reason,
        "new_title": outcome.new_title,
    }


class ReassignBody(BaseModel):
    profile: Optional[str] = None  # "" or None = unassign
    reclaim_first: bool = False
    reason: Optional[str] = None


@router.post("/tasks/{task_id}/reassign")
def reassign_task_endpoint(task_id: str, payload: ReassignBody, board: Optional[str] = Query(None)):
    """Reassign to another profile, optionally reclaiming first
    (``hermes kanban reassign <task_id> <profile> [--reclaim]``)."""
    with _board_conn(board) as (board, conn):
        ok = kanban_db.reassign_task(
            conn, task_id,
            payload.profile or None,
            reclaim_first=bool(payload.reclaim_first),
            reason=payload.reason,
        )
        if not ok:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"cannot reassign {task_id}: unknown id, or still "
                    "running (pass reclaim_first=true to release the claim first)"
                ),
            )
        return {"ok": True, "task_id": task_id, "assignee": payload.profile or None}


# Estimate: rough token/complexity read via the auxiliary model. NOT a dollar
# cost — providers don't report cost reliably.
_ESTIMATE_SYSTEM_PROMPT = (
    "You estimate how much work an autonomous coding agent will spend on a "
    "kanban task. Given the task title and description, respond with STRICT "
    "JSON only (no prose, no code fence):\n"
    '{"est_tokens": <integer total tokens across the whole run>, '
    '"complexity": "S"|"M"|"L", '
    '"rationale": "<one short sentence>"}\n'
    "Base the token figure on a realistic multi-turn agent run (reading files, "
    "tool calls, edits, retries) — not a single reply. S≈small/localized, "
    "M≈multi-file, L≈broad or ambiguous. Be honest that this is a rough guess."
)


class EstimateBody(BaseModel):
    title: str = ""
    body: Optional[str] = None


@router.post("/estimate")
def estimate_text_endpoint(payload: EstimateBody):
    """Estimate from raw title/body (create dialog, before a task exists)."""
    return _run_estimate(payload.title, payload.body)


@router.post("/tasks/{task_id}/estimate")
def estimate_task_endpoint(task_id: str, board: Optional[str] = Query(None)):
    """Estimate for an existing task; ``{ok, est_tokens, complexity, rationale, model}``."""
    with _board_conn(board) as (board, conn):
        task = kanban_db.get_task(conn, task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"task {task_id} not found")
    return _run_estimate(task.title, task.body)


def _run_estimate(title: str, body: Optional[str]) -> dict:
    """Never raises — config/parse/API errors become ``{"ok": False, "reason"}``
    so the UI renders them inline."""
    if not (title or "").strip():
        return {"ok": False, "reason": "a title is required to estimate"}

    try:
        from agent.auxiliary_client import call_llm
    except Exception:
        return {"ok": False, "reason": "auxiliary client unavailable"}

    def _cap(s: Optional[str], n: int) -> str:
        s = (s or "").strip()
        return s if len(s) <= n else s[:n] + "…"

    user_msg = (f"Title: {_cap(title, 400)}\n\n" f"Description:\n{_cap(body, 4000) or '(none)'}")
    try:
        resp = call_llm(
            task="kanban_estimator",
            messages=[
                {"role": "system", "content": _ESTIMATE_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
            max_tokens=300,
            timeout=60,
        )
    except Exception as exc:
        return {"ok": False, "reason": f"LLM error: {type(exc).__name__}"}

    try:
        raw = (resp.choices[0].message.content or "").strip()
        model = getattr(resp, "model", None)
    except Exception:
        raw, model = "", None

    # Same tolerant JSON-blob extraction the specifier uses.
    parsed: Optional[dict] = None
    try:
        blob = raw
        if not blob.lstrip().startswith("{"):
            m = re.search(r"\{.*\}", blob, re.DOTALL)
            blob = m.group(0) if m else blob
        obj = json.loads(blob)
        if isinstance(obj, dict):
            parsed = obj
    except Exception:
        parsed = None

    if not parsed:
        return {"ok": False, "reason": "could not parse an estimate from the model"}

    try:
        est_tokens = int(parsed.get("est_tokens") or 0)
    except (TypeError, ValueError):
        est_tokens = 0
    complexity = str(parsed.get("complexity") or "").strip().upper()
    if complexity not in {"S", "M", "L"}:
        complexity = None
    rationale = str(parsed.get("rationale") or "").strip() or None

    return {
        "ok": True,
        "est_tokens": est_tokens,
        "complexity": complexity,
        "rationale": rationale,
        "model": model,
    }


# --- Plugin config ----------------------------------------------------------

def _load_config_or_empty() -> dict:
    try:
        from hermes_cli.config import load_config
        return load_config() or {}
    except Exception:
        return {}


@router.get("/config")
def get_config():
    """Kanban dashboard preferences from the ``dashboard.kanban`` config section."""
    k_cfg = (_load_config_or_empty().get("dashboard") or {}).get("kanban") or {}
    return {
        "default_tenant": k_cfg.get("default_tenant") or "",
        "lane_by_profile": bool(k_cfg.get("lane_by_profile", True)),
        "include_archived_by_default": bool(k_cfg.get("include_archived_by_default", False)),
        "render_markdown": bool(k_cfg.get("render_markdown", True)),
    }


# --- Home-channel subscriptions (per-task, per-platform toggles) -------------
# Each gateway platform has at most one "home" (chat_id, thread_id, name). A
# toggle-on writes exactly the notify_subs row ``/kanban create`` would, so the
# existing gateway notifier delivers completed/blocked/gave_up with no extra plumbing.

def _configured_home_channels() -> list[dict]:
    """Every platform with a home_channel, from the live GatewayConfig (so env
    overlays like ``TELEGRAM_HOME_CHANNEL`` are honored), sorted by platform."""
    try:
        from gateway.config import load_gateway_config
        gw_cfg = load_gateway_config()
    except Exception:
        return []
    result: list[dict] = []
    for platform, pcfg in gw_cfg.platforms.items():
        if not pcfg or not pcfg.home_channel:
            continue
        hc = pcfg.home_channel
        result.append({
            "platform": platform.value,
            "chat_id": hc.chat_id,
            "thread_id": hc.thread_id or "",
            "name": hc.name or "Home",
        })
    result.sort(key=lambda r: r["platform"])
    return result


def _active_profile_name() -> str:
    """Current Hermes profile name for notify-sub ownership."""
    try:
        from hermes_cli.profiles import get_active_profile_name
        return get_active_profile_name() or "default"
    except Exception:
        return "default"


def _home_for_platform(platform: str, detail: str) -> dict:
    home = next((h for h in _configured_home_channels() if h["platform"] == platform), None)
    if not home:
        raise HTTPException(status_code=404, detail=detail)
    return home


@router.get("/home-channels")
def get_home_channels(task_id: Optional[str] = Query(None), board: Optional[str] = Query(None)):
    """Every platform with a home channel plus whether *task_id* (if given) is
    subscribed to it; without ``task_id`` every ``subscribed`` is false."""
    homes = _configured_home_channels()
    subscribed_homes: set[tuple[str, str, str]] = set()
    if task_id:
        with _board_conn(board) as (board, conn):
            subs = kanban_db.list_notify_subs(conn, task_id)
        for sub in subs:
            subscribed_homes.add((
                str(sub.get("platform") or ""),
                str(sub.get("chat_id") or ""),
                str(sub.get("thread_id") or ""),
            ))
    result = []
    for home in homes:
        key = (home["platform"], home["chat_id"], home["thread_id"])
        result.append({**home, "subscribed": key in subscribed_homes})
    return {"home_channels": result}


@router.post("/tasks/{task_id}/home-subscribe/{platform}")
def subscribe_home(task_id: str, platform: str, board: Optional[str] = Query(None)):
    """Subscribe *task_id* to *platform*'s home channel. Idempotent at the DB
    layer; 404 when the platform has no home or the task doesn't exist."""
    home = _home_for_platform(
        platform,
        f"No home channel configured for platform {platform!r}. "
        f"Set one from the messenger via /sethome, or configure "
        f"gateway.platforms.{platform}.home_channel in config.yaml.",
    )
    with _board_conn(board) as (board, conn):
        _require_task(conn, task_id)
        kanban_db.add_notify_sub(
            conn,
            task_id=task_id,
            platform=platform,
            chat_id=home["chat_id"],
            thread_id=home["thread_id"] or None,
            notifier_profile=_active_profile_name(),
        )
        return {"ok": True, "task_id": task_id, "home_channel": home}


@router.delete("/tasks/{task_id}/home-subscribe/{platform}")
def unsubscribe_home(task_id: str, platform: str, board: Optional[str] = Query(None)):
    """Remove any notify subscription on *task_id* matching *platform*'s home."""
    home = _home_for_platform(platform, f"No home channel configured for platform {platform!r}.")
    with _board_conn(board) as (board, conn):
        kanban_db.remove_notify_sub(
            conn,
            task_id=task_id,
            platform=platform,
            chat_id=home["chat_id"],
            thread_id=home["thread_id"] or None,
        )
        return {"ok": True, "task_id": task_id, "home_channel": home}


# --- Stats / assignees / worker log / dispatch / model options ---------------

@router.get("/stats")
def get_stats(board: Optional[str] = Query(None)):
    """Per-status + per-assignee counts + oldest-ready age (HUD and router profiles)."""
    with _board_conn(board) as (board, conn):
        return kanban_db.board_stats(conn)


@router.get("/assignees")
def get_assignees(board: Optional[str] = Query(None)):
    """Union of on-disk profiles and assignees used on the board, so a fresh
    profile appears in the picker before it has any task."""
    with _board_conn(board) as (board, conn):
        return {"assignees": kanban_db.known_assignees(conn)}


@router.get("/tasks/{task_id}/log")
def get_task_log(
    task_id: str,
    tail: Optional[int] = Query(None, ge=1, le=2_000_000),
    board: Optional[str] = Query(None),
):
    """Worker stdout/stderr log. ``tail`` caps the response bytes; 404 if the
    task never spawned. On-disk log rotates at 2 MiB with one ``.log.1`` kept."""
    with _board_conn(board) as (board, conn):
        task = kanban_db.get_task(conn, task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"task {task_id} not found")
    content = kanban_db.read_worker_log(task_id, tail_bytes=tail, board=board)
    log_path = kanban_db.worker_log_path(task_id, board=board)
    size = log_path.stat().st_size if log_path.exists() else 0
    return {
        "task_id": task_id,
        "path": str(log_path),
        "exists": content is not None,
        "size_bytes": size,
        "content": content or "",
        "truncated": bool(tail and size > tail),
    }


@router.post("/dispatch")
def dispatch(
    dry_run: bool = Query(False),
    max_n: int = Query(8, alias="max"),
    board: Optional[str] = Query(None),
):
    """Dispatch nudge so the UI doesn't wait out the 60 s dispatcher tick."""
    with _board_conn(board) as (board, conn):
        result = kanban_db.dispatch_once(conn, dry_run=dry_run, max_spawn=max_n, board=board)
        try:
            return asdict(result)  # DispatchResult is a dataclass
        except TypeError:
            return {"result": str(result)}


@router.get("/model-options")
def model_options():
    """Authenticated providers + curated models for the model-override dropdown.

    Thin wrapper over ``inventory.build_models_payload`` (same substrate as the
    Models page / TUI picker) so the dropdown can't offer a pair the rest of
    Hermes rejects. Skips pricing enrichment and custom-provider probes: a
    slow/offline local endpoint must not hang the drawer.
    """
    try:
        from hermes_cli.inventory import build_models_payload, load_picker_context

        payload = build_models_payload(
            load_picker_context(),
            explicit_only=True,
            canonical_order=True,
            probe_custom_providers=False,
        )
        return {
            "providers": [
                {
                    "slug": row.get("slug", ""),
                    "label": row.get("label") or row.get("slug", ""),
                    "models": list(row.get("models") or []),
                }
                for row in payload.get("providers", [])
                if row.get("models")
            ],
        }
    except Exception:
        log.exception("kanban model-options failed")
        # Empty catalog → the UI falls back to a free-text input.
        return {"providers": []}


# --- Boards CRUD (multi-project support) --------------------------------------

class CreateBoardBody(BaseModel):
    slug: str
    name: Optional[str] = None
    description: Optional[str] = None
    icon: Optional[str] = None
    color: Optional[str] = None
    default_workdir: Optional[str] = None
    # Project (id or slug) scoping the board: default_workdir mirrors the
    # project's primary repo and new tasks inherit the project.
    project_id: Optional[str] = None
    switch: bool = False


class RenameBoardBody(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    icon: Optional[str] = None
    color: Optional[str] = None
    # For both fields: ``None`` = leave unchanged; "" = clear; value = validate/resolve + set.
    default_workdir: Optional[str] = None
    project_id: Optional[str] = None


# Board transfer exchanges filesystem PATHS, not bytes (same contract as profile
# export/import): the desktop/dashboard clients run the native save/open dialog
# on the machine hosting the backend, so a path is all either side needs.

class ExportBoardBody(BaseModel):
    output: str = ""  # empty → staging path under the kanban root
    attachments: bool = True
    logs: bool = False


class ImportBoardBody(BaseModel):
    archive: str  # path to a board .tar.gz on the backend's filesystem
    slug: Optional[str] = None  # override the archive's slug; collisions auto-suffix
    switch: bool = False


def _resolve_project(ref: Optional[str]) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Resolve a project id/slug to ``(id, name, primary_path)``; ``(None,)*3``
    for a falsy ref, 400 when a non-empty ref doesn't resolve."""
    if not ref or not ref.strip():
        return None, None, None
    try:
        from hermes_cli import projects_db as pdb
        with pdb.connect_closing() as pconn:
            proj = pdb.get_project(pconn, ref.strip())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"projects unavailable: {exc}")
    if proj is None:
        raise HTTPException(status_code=400, detail=f"project {ref!r} does not exist")
    return proj.id, proj.name, (proj.primary_path or None)


def _projects_by_id() -> dict[str, Any]:
    """Map every project id -> Project (archived included) for annotation."""
    try:
        from hermes_cli import projects_db as pdb
        with pdb.connect_closing() as pconn:
            return {p.id: p for p in pdb.list_projects(pconn, include_archived=True)}
    except Exception:
        return {}


def _board_counts(slug: str) -> dict[str, int]:
    """``{status: count}`` for a board; ``{}`` on a missing/empty DB."""
    try:
        path = kanban_db.kanban_db_path(board=slug)
        if not path.exists():
            return {}
        conn = kanban_db.connect(board=slug)
        try:
            rows = conn.execute(
                "SELECT status, COUNT(*) AS n FROM tasks GROUP BY status"
            ).fetchall()
            return {r["status"]: int(r["n"]) for r in rows}
        finally:
            conn.close()
    except Exception:
        return {}


def _default_workspace_kind(board: dict[str, Any]) -> str:
    """Recommend a non-destructive task workspace from board metadata."""
    workdir = str(board.get("default_workdir") or "").strip()
    if not workdir:
        return "scratch"
    try:
        return "worktree" if kanban_db._git_toplevel(Path(workdir)) else "dir"
    except (OSError, ValueError):
        return "dir"


def _annotate_board_meta(meta: dict) -> dict:
    meta["default_workspace_kind"] = _default_workspace_kind(meta)
    _, meta["project_name"], _ = _resolve_project(meta.get("project_id"))
    return meta


@router.get("/projects")
def list_kanban_projects():
    """Live (non-archived) projects available for board scoping."""
    try:
        from hermes_cli import projects_db as pdb
        with pdb.connect_closing() as pconn:
            projects = pdb.list_projects(pconn, include_archived=False)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"failed to list projects: {exc}")
    return {
        "projects": [
            {
                "id": p.id,
                "slug": p.slug,
                "name": p.name,
                "primary_path": p.primary_path or "",
                "icon": p.icon or "",
                "color": p.color or "",
            }
            for p in projects
        ]
    }


@router.get("/boards")
def list_boards(include_archived: bool = Query(False)):
    """Every board on disk with task counts and the active slug."""
    boards = kanban_db.list_boards(include_archived=include_archived)
    current = kanban_db.get_current_board()
    proj_map = _projects_by_id()
    for b in boards:
        b["is_current"] = (b["slug"] == current)
        b["counts"] = _board_counts(b["slug"])
        # Live cards only — archived tasks are hidden from every default board
        # view, so counting them in the switcher badge would visibly disagree.
        b["total"] = sum(n for status, n in b["counts"].items() if status != "archived")
        b["default_workspace_kind"] = _default_workspace_kind(b)
        pid = b.get("project_id") or None
        b["project_id"] = pid
        proj = proj_map.get(pid) if pid else None
        b["project_name"] = proj.name if proj else None
    return {"boards": boards, "current": current}


def _validate_workdir(raw: str) -> str:
    """Board default_workdir must be an absolute, existing directory (400 otherwise)."""
    requested = Path(raw).expanduser()
    if not requested.is_absolute():
        raise HTTPException(status_code=400, detail="Project directory must be an absolute path.")
    if not requested.is_dir():
        raise HTTPException(status_code=400, detail="Project directory must be an existing directory.")
    return str(requested.resolve())


@router.post("/boards")
def create_board_endpoint(payload: CreateBoardBody):
    """Create a board. Idempotent — ``slug`` collision returns the existing one."""
    default_workdir = None
    if payload.default_workdir:
        default_workdir = _validate_workdir(payload.default_workdir)
    # A chosen project's primary repo becomes the default workdir unless one was
    # passed explicitly.
    project_id, _pname, primary_path = _resolve_project(payload.project_id)
    if primary_path and not default_workdir:
        default_workdir = primary_path
    with _value_error_400():
        meta = kanban_db.create_board(
            payload.slug,
            name=payload.name,
            description=payload.description,
            icon=payload.icon,
            color=payload.color,
            default_workdir=default_workdir,
            project_id=project_id,
        )
    if payload.switch:
        with _value_error_400():
            kanban_db.set_current_board(meta["slug"])
    return {"board": _annotate_board_meta(meta), "current": kanban_db.get_current_board()}


@router.patch("/boards/{slug}")
def rename_board(slug: str, payload: RenameBoardBody):
    """Update display metadata / default workdir / project scope (slug is immutable)."""
    normed = _existing_board_slug(slug)
    # write_board_metadata treats a falsy value as "clear", so pass "" through.
    default_workdir: Optional[str] = None
    if payload.default_workdir is not None:
        raw = payload.default_workdir.strip()
        default_workdir = _validate_workdir(raw) if raw else ""
    # A resolved project mirrors its repo into default_workdir unless the caller
    # set default_workdir explicitly.
    project_id: Optional[str] = None
    if payload.project_id is not None:
        if payload.project_id.strip():
            project_id, _pname, primary_path = _resolve_project(payload.project_id)
            if primary_path and default_workdir is None:
                default_workdir = primary_path
        else:
            project_id = ""  # clear the scope
    meta = kanban_db.write_board_metadata(
        normed,
        name=payload.name,
        description=payload.description,
        icon=payload.icon,
        color=payload.color,
        default_workdir=default_workdir,
        project_id=project_id,
    )
    return {"board": _annotate_board_meta(meta)}


@router.delete("/boards/{slug}")
def delete_board(slug: str, delete: bool = Query(False, description="Hard-delete instead of archive")):
    """Archive (default) or hard-delete a board."""
    with _value_error_400():
        res = kanban_db.remove_board(slug, archive=not delete)
    return {"result": res, "current": kanban_db.get_current_board()}


async def _run_transfer(fn, log_label: str):
    """Run a blocking kanban_transfer call off the event loop, mapping its errors
    to 404 (missing path) / 400 (invalid) / 500 (logged)."""
    try:
        return await asyncio.get_running_loop().run_in_executor(None, fn)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        log.exception("%s failed", log_label)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/boards/{slug}/export")
async def export_board_endpoint(slug: str, body: ExportBoardBody):
    """Write ``slug`` to a portable archive; return the path written."""
    from hermes_cli import kanban_transfer

    output = (body.output or "").strip()
    if not output:
        staging = kanban_db.kanban_home() / "kanban" / "board-exports"
        try:
            staging.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise HTTPException(status_code=500, detail=f"Could not create export directory: {exc}")
        stamp = time.strftime("%Y%m%d-%H%M%S")
        output = str(staging / f"{slug}-{stamp}.tar.gz")

    return await _run_transfer(
        lambda: kanban_transfer.export_board(
            slug, output,
            include_attachments=body.attachments,
            include_logs=body.logs,
        ),
        f"POST /boards/{slug}/export",
    )


@router.post("/boards/import")
async def import_board_endpoint(body: ImportBoardBody):
    """Import a board archive as a NEW board; return the landed board."""
    from hermes_cli import kanban_transfer

    archive = (body.archive or "").strip()
    if not archive:
        raise HTTPException(status_code=400, detail="archive path is required")

    result = await _run_transfer(
        lambda: kanban_transfer.import_board(
            archive, (body.slug or "").strip() or None, activate=body.switch
        ),
        "POST /boards/import",
    )
    return {**result, "current": kanban_db.get_current_board()}


@router.post("/boards/{slug}/switch")
def switch_board(slug: str):
    """Persist ``slug`` as the active board for CLI / slash-command parity
    (dashboard users pick boards client-side via localStorage)."""
    normed = _existing_board_slug(slug)
    kanban_db.set_current_board(normed)
    return {"current": normed}


# Poll interval for the event tail loop. SQLite WAL + 300 ms polling is the
# simplest robust approach: negligible CPU, no shared state across workers.
_EVENT_POLL_SECONDS = 0.3


# --- Profile metadata & description editing (kanban orchestrator) ------------

class DescribeBody(BaseModel):
    description: Optional[str] = None  # explicit user-authored text


class DescribeAutoBody(BaseModel):
    overwrite: bool = False


@router.get("/profiles")
def list_profile_roster():
    """Every installed profile with its description (profiles without one are
    still routable on name alone, just less precisely)."""
    try:
        from hermes_cli import profiles as profiles_mod
        profiles = profiles_mod.list_profiles()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"failed to list profiles: {exc}")
    return {
        "profiles": [
            {
                "name": p.name,
                "is_default": bool(p.is_default),
                "model": p.model or "",
                "provider": p.provider or "",
                "description": p.description or "",
                "description_auto": bool(p.description_auto),
                "skill_count": int(p.skill_count or 0),
            }
            for p in profiles
        ],
    }


@router.patch("/profiles/{profile_name}")
def update_profile_description(profile_name: str, payload: DescribeBody):
    """Set (``description_auto: false`` so the auto-describer won't overwrite it
    without ``--overwrite``) or clear (empty string) a profile's description."""
    try:
        from hermes_cli import profiles as profiles_mod
        canon = profiles_mod.normalize_profile_name(profile_name)
        if canon == "default":
            from hermes_constants import get_hermes_home  # type: ignore
            profile_dir = Path(get_hermes_home())
        else:
            profile_dir = profiles_mod.get_profile_dir(canon)
        if not profile_dir.is_dir():
            raise HTTPException(status_code=404, detail=f"profile '{profile_name}' not found")
        text = (payload.description or "").strip()
        profiles_mod.write_profile_meta(profile_dir, description=text, description_auto=False)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"failed to update profile: {exc}")
    return {"ok": True, "profile": canon, "description": text}


@router.post("/profiles/{profile_name}/describe-auto")
def auto_describe_profile(profile_name: str, payload: DescribeAutoBody):
    """Generate a description via ``auxiliary.profile_describer`` and persist it
    with ``description_auto: true`` (``hermes profile describe <name> --auto``).
    Non-OK outcomes are NOT HTTP errors — the UI renders the reason inline."""
    try:
        from hermes_cli import profile_describer
        outcome = profile_describer.describe_profile(
            profile_name,
            overwrite=bool(payload.overwrite),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"describer crashed: {exc}")
    return {
        "ok": bool(outcome.ok),
        "profile": outcome.profile_name,
        "reason": outcome.reason,
        "description": outcome.description,
    }


# --- Decompose (built-in decomposer fan-out) ----------------------------------

class DecomposeBody(BaseModel):
    author: Optional[str] = None


@router.post("/tasks/{task_id}/decompose")
def decompose_task_endpoint(
    task_id: str,
    payload: DecomposeBody,
    board: Optional[str] = Query(None),
):
    """Fan a triage task out into child tasks via the auxiliary LLM, routed to
    specialist profiles by description (``hermes kanban decompose``). Returns
    ``{ok, task_id, reason, fanout, child_ids, new_title}``; non-OK is NOT an
    HTTP error. Sync ``def`` so the slow LLM call runs in the threadpool."""
    board = _resolve_board(board)
    # Context-local board pin — see specify_task_endpoint for the race rationale.
    with kanban_db.scoped_current_board(board or kanban_db.DEFAULT_BOARD):
        from hermes_cli import kanban_decompose
        outcome = kanban_decompose.decompose_task(task_id, author=(payload.author or None))

    return {
        "ok": bool(outcome.ok),
        "task_id": outcome.task_id,
        "reason": outcome.reason,
        "fanout": bool(outcome.fanout),
        "child_ids": outcome.child_ids or [],
        "new_title": outcome.new_title,
    }


# --- Orchestration settings (kanban.orchestrator_profile / default_assignee /
#     auto_decompose / auto_promote_children) ----------------------------------

class OrchestrationSettingsBody(BaseModel):
    orchestrator_profile: Optional[str] = None
    default_assignee: Optional[str] = None
    auto_decompose: Optional[bool] = None
    auto_promote_children: Optional[bool] = None


@router.get("/orchestration")
def get_orchestration_settings():
    """Current orchestration knobs from config.yaml plus the resolved effective
    values (fallbacks filled the same way the decomposer does)."""
    cfg = _load_config_or_empty()
    kanban_cfg = (cfg.get("kanban") or {}) if isinstance(cfg, dict) else {}
    explicit_orch = (kanban_cfg.get("orchestrator_profile") or "").strip()
    explicit_default = (kanban_cfg.get("default_assignee") or "").strip()
    auto_decompose = bool(kanban_cfg.get("auto_decompose", True))
    auto_promote_children = bool(kanban_cfg.get("auto_promote_children", True))

    resolved_orch = explicit_orch
    resolved_default = explicit_default
    try:
        from hermes_cli import profiles as profiles_mod
        active_default = profiles_mod.get_active_profile_name() or "default"
        if not resolved_orch or not profiles_mod.profile_exists(resolved_orch):
            resolved_orch = active_default
        if not resolved_default or not profiles_mod.profile_exists(resolved_default):
            resolved_default = active_default
    except Exception:
        active_default = "default"
        resolved_orch = resolved_orch or active_default
        resolved_default = resolved_default or active_default

    return {
        "orchestrator_profile": explicit_orch,
        "default_assignee": explicit_default,
        "auto_decompose": auto_decompose,
        "auto_promote_children": auto_promote_children,
        "resolved_orchestrator_profile": resolved_orch,
        "resolved_default_assignee": resolved_default,
        "active_profile": active_default,
    }


def _validated_profile_name(raw: Optional[str], profiles_mod) -> str:
    """Strip a profile name; 400 if non-empty and unknown. Fails open when the
    lookup itself errors."""
    name = (raw or "").strip()
    if name and profiles_mod is not None:
        try:
            if not profiles_mod.profile_exists(name):
                raise HTTPException(status_code=400, detail=f"profile '{name}' does not exist")
        except HTTPException:
            raise
        except Exception:
            pass
    return name


@router.put("/orchestration")
def set_orchestration_settings(payload: OrchestrationSettingsBody):
    """Update orchestration knobs in config.yaml. Only fields explicitly passed
    are written; empty profile strings clear the override."""
    try:
        from hermes_cli.config import load_config, save_config
        cfg = load_config() or {}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"failed to load config: {exc}")

    kanban_section = cfg.setdefault("kanban", {})
    if not isinstance(kanban_section, dict):
        kanban_section = {}
        cfg["kanban"] = kanban_section

    try:
        from hermes_cli import profiles as profiles_mod
    except Exception:
        profiles_mod = None  # type: ignore

    if payload.orchestrator_profile is not None:
        kanban_section["orchestrator_profile"] = _validated_profile_name(
            payload.orchestrator_profile, profiles_mod,
        )
    if payload.default_assignee is not None:
        kanban_section["default_assignee"] = _validated_profile_name(
            payload.default_assignee, profiles_mod,
        )
    if payload.auto_decompose is not None:
        kanban_section["auto_decompose"] = bool(payload.auto_decompose)
    if payload.auto_promote_children is not None:
        kanban_section["auto_promote_children"] = bool(payload.auto_promote_children)

    try:
        save_config(cfg)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"failed to save config: {exc}")

    return get_orchestration_settings()  # callers re-render from the resolved state


# --- WebSocket: /events?since=<event_id>&board=<slug> ------------------------

@router.websocket("/events")
async def stream_events(ws: WebSocket):
    if not _ws_upgrade_authorized(ws):
        await ws.close(code=http_status.WS_1008_POLICY_VIOLATION)
        return
    await ws.accept()

    # One SQLite connection per socket, used (and closed) only on a dedicated
    # single worker thread: sqlite connections are thread-affine, and reusing it
    # avoids churning WAL/SHM sidecars while an idle dashboard polls.
    event_conn: Optional[sqlite3.Connection] = None
    event_executor: Optional[ThreadPoolExecutor] = None

    def _close_event_conn() -> None:
        nonlocal event_conn
        if event_conn is not None:
            event_conn.close()
            event_conn = None

    try:
        try:
            cursor = int(ws.query_params.get("since", "0"))
        except ValueError:
            cursor = 0

        # Board is pinned at the handshake; the UI opens a new WS on board change
        # rather than reconciling two cursors mid-stream.
        ws_board_raw = ws.query_params.get("board")
        try:
            ws_board = kanban_db._normalize_board_slug(ws_board_raw) if ws_board_raw else None
        except ValueError:
            ws_board = None

        def _fetch_new(cursor_val: int) -> tuple[int, list[dict]]:
            nonlocal event_conn
            if event_conn is None:
                event_conn = kanban_db.connect(board=ws_board)
            rows = event_conn.execute(
                "SELECT id, task_id, run_id, kind, payload, created_at "
                "FROM task_events WHERE id > ? ORDER BY id ASC LIMIT 200",
                (cursor_val,),
            ).fetchall()
            out: list[dict] = []
            new_cursor = cursor_val
            for r in rows:
                try:
                    payload = json.loads(r["payload"]) if r["payload"] else None
                except Exception:
                    payload = None
                out.append({**dict(r), "payload": payload})
                new_cursor = r["id"]
            return new_cursor, out

        while True:
            # Race receive() against the poll interval so a client disconnect is
            # detected even when no events are flowing; otherwise an idle board
            # leaks zombie poll tasks until the next send_json() fails.
            try:
                msg = await asyncio.wait_for(ws.receive(), timeout=_EVENT_POLL_SECONDS)
                if msg["type"] == "websocket.disconnect":
                    return
                # Other client messages (pong, text) are ignored.
            except asyncio.TimeoutError:
                pass  # no client message — poll the DB

            if event_executor is None:
                event_executor = ThreadPoolExecutor(
                    max_workers=1,
                    thread_name_prefix="kanban-events",
                )
            cursor, events = await asyncio.get_running_loop().run_in_executor(
                event_executor,
                _fetch_new,
                cursor,
            )
            if events:
                await ws.send_json({"events": events, "cursor": cursor})
    except WebSocketDisconnect:
        return
    except asyncio.CancelledError:
        # Normal shutdown (Ctrl-C cancels the task mid-poll). CancelledError is a
        # BaseException, so the Exception handler below wouldn't quiet it and
        # Uvicorn would print an application traceback.
        return
    except Exception as exc:  # never crash the dashboard worker
        log.warning("Kanban event stream error: %s", exc)
        try:
            await ws.close()
        except Exception:
            pass
    finally:
        if event_executor is not None:
            try:
                await asyncio.get_running_loop().run_in_executor(event_executor, _close_event_conn)
            except Exception as exc:
                log.warning("Kanban event stream connection cleanup failed: %s", exc)
            finally:
                event_executor.shutdown(wait=True, cancel_futures=True)
