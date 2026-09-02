"""Kanban tools — structured tool-call surface for worker + orchestrator agents.

Registered into the model's schema only when running under the dispatcher
(``HERMES_KANBAN_TASK`` set) or when the active profile enables the ``kanban``
toolset; a plain ``hermes chat`` session sees zero kanban tools.

Why tools rather than shelling out to ``hermes kanban``: tools run in the
agent's Python process, so they reach ``~/.hermes/kanban.db`` even when the
terminal backend is a container/SSH host without ``hermes`` installed; they
avoid shlex/argparse quoting of JSON metadata; and failures come back as
structured JSON the model can reason about. Humans keep using the CLI,
dashboard, and ``/kanban`` slash command, which bypass the agent entirely.
"""
from __future__ import annotations

import functools
import json
import logging
import os
from contextlib import contextmanager
from typing import Any, Callable, Optional

from agent.redact import redact_sensitive_text
from hermes_cli.goals import judge_goal
from tools.registry import registry, tool_error
from hermes_cli.config import cfg_get, load_config
from tools.kanban_tools_schemas import (  # noqa: F401 - re-exported for callers/tests
    _DESC_BOARD,
    _DESC_TASK_ID_DEFAULT,
    _board_schema_prop,
    KANBAN_ATTACH_SCHEMA,
    KANBAN_ATTACH_URL_SCHEMA,
    KANBAN_ATTACHMENTS_SCHEMA,
    KANBAN_BLOCK_SCHEMA,
    KANBAN_COMMENT_SCHEMA,
    KANBAN_COMPLETE_SCHEMA,
    KANBAN_CREATE_SCHEMA,
    KANBAN_HEARTBEAT_SCHEMA,
    KANBAN_LINK_SCHEMA,
    KANBAN_LIST_SCHEMA,
    KANBAN_REQUEST_CHANGES_SCHEMA,
    KANBAN_REQUEST_REVIEW_SCHEMA,
    KANBAN_SHOW_SCHEMA,
    KANBAN_UNBLOCK_SCHEMA,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Gating
# ---------------------------------------------------------------------------

KANBAN_LIST_DEFAULT_LIMIT = 50
KANBAN_LIST_MAX_LIMIT = 200


def _profile_has_kanban_toolset() -> bool:
    # load_config() is mtime-cached and check_fn results are TTL-cached (~30s)
    # by the registry, so this is cheap.
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
        return "kanban" in cfg.get("toolsets", [])
    except Exception:
        return False


def _is_delegated_child_context() -> bool:
    try:
        from agent.delegation_context import is_delegated_child_context

        return is_delegated_child_context()
    except Exception:
        return False


def _is_dispatcher_owned_worker() -> bool:
    """False for delegate_task children AND for cron jobs fired in-process from
    a worker — i.e. whenever HERMES_KANBAN_* is present but not ours."""
    try:
        from agent.delegation_context import is_dispatcher_owned_worker_context

        return is_dispatcher_owned_worker_context()
    except Exception:
        return True


def _is_env_worker() -> bool:
    """True only for a dispatcher-spawned worker scoped to HERMES_KANBAN_TASK."""
    return bool(os.environ.get("HERMES_KANBAN_TASK")) and _is_dispatcher_owned_worker()


def _reject_delegated_child_mutation(tool_name: str) -> Optional[str]:
    """Deny Kanban mutations from delegate_task children.

    A child runs in the same process as its parent, so inherited HERMES_KANBAN_*
    env vars are not proof of dispatcher ownership. It may report findings to
    the parent but must not mutate board state directly.
    """
    if not _is_delegated_child_context():
        return None
    return tool_error(
        f"{tool_name} refused: delegate_task child agents are not Kanban "
        "run owners. Return findings to the parent agent; the dispatcher "
        "worker or an explicitly configured Kanban orchestrator must perform "
        "board mutations."
    )


def _check_kanban_mode() -> bool:
    """Lifecycle tools: visible to dispatcher-spawned workers and to profiles
    that enable the ``kanban`` toolset (orchestrators); never to delegate children."""
    if _is_delegated_child_context():
        return False
    if _is_env_worker():
        return True
    return _profile_has_kanban_toolset()


def _check_kanban_orchestrator_mode() -> bool:
    """Board-routing tools (kanban_list, kanban_unblock): hidden from task workers.

    Workers close their own task via complete/block/heartbeat; only profiles
    that opt into the toolset and are NOT scoped to a single task route work.
    """
    if _is_delegated_child_context():
        return False
    if _is_env_worker():
        return False
    return _profile_has_kanban_toolset()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_TASK_ID_REQUIRED = "task_id is required (or set HERMES_KANBAN_TASK in the env)"


def _default_task_id(arg: Optional[str]) -> Optional[str]:
    """Resolve ``task_id`` arg or fall back to the env var the dispatcher set.

    A delegate child or a cron job fired in-process from a worker must never
    inherit the worker's task id as an implicit default.
    """
    if arg:
        return arg
    if _is_delegated_child_context() or not _is_dispatcher_owned_worker():
        return None
    return os.environ.get("HERMES_KANBAN_TASK") or None


def _worker_run_id(task_id: str) -> Optional[int]:
    """Return this worker's dispatcher run id when it is scoped to task_id."""
    if os.environ.get("HERMES_KANBAN_TASK") != task_id:
        return None
    raw = os.environ.get("HERMES_KANBAN_RUN_ID")
    try:
        return int(raw) if raw else None
    except ValueError:
        return None


def _stamp_worker_session_metadata(task_id: str, metadata: Optional[dict]) -> Optional[dict]:
    """Add trusted worker session id metadata for this worker's own task."""
    session_id = os.environ.get("HERMES_SESSION_ID")
    if os.environ.get("HERMES_KANBAN_TASK") != task_id or not session_id:
        return metadata
    return {**(metadata or {}), "worker_session_id": session_id}


def _enforce_worker_task_ownership(tid: str) -> Optional[str]:
    """Reject worker-driven destructive calls on foreign task IDs.

    A dispatcher-spawned worker has ``HERMES_KANBAN_TASK`` set to its own task;
    a buggy or prompt-injected explicit ``task_id`` must not corrupt sibling or
    cross-tenant runs. Orchestrators (toolset enabled, no env task) are exempt:
    routing legitimately closes or reopens child tasks.
    """
    env_tid = os.environ.get("HERMES_KANBAN_TASK")
    if env_tid and tid != env_tid:
        return tool_error(
            f"worker is scoped to task {env_tid}; refusing to mutate "
            f"{tid}. Use kanban_comment to hand off information to other "
            f"tasks, or kanban_create to spawn follow-up work."
        )
    return None


def _worker_guard(tool_name: str, args: dict) -> tuple[str, Optional[str]]:
    """Common preamble for worker mutation tools: ``(task_id, error)``.

    Order matters: delegate-child rejection, then task id resolution, then
    task-scope ownership. ``task_id`` is only meaningful when ``error`` is None.
    """
    err = _reject_delegated_child_mutation(tool_name)
    if err:
        return "", err
    tid = _default_task_id(args.get("task_id"))
    if not tid:
        return "", tool_error(_TASK_ID_REQUIRED)
    return tid, _enforce_worker_task_ownership(tid)


def _connect(board: Optional[str] = None):
    """Import + connect lazily so the module imports cleanly in non-kanban contexts.

    ``board=None`` keeps the legacy resolution chain (``HERMES_KANBAN_DB`` →
    ``HERMES_KANBAN_BOARD`` → current symlink → ``default``); an explicit slug
    lets e.g. a Telegram-side agent override the env-pinned board per call.
    """
    from hermes_cli import kanban_db as kb
    return kb, kb.connect(board=board)


@contextmanager
def _board(board: Optional[str]):
    """``with _board(slug) as (kb, conn)`` — connection closed on exit."""
    kb, conn = _connect(board=board)
    try:
        yield kb, conn
    finally:
        conn.close()


def _close_quietly(conn) -> None:
    try:
        conn.close()
    except Exception:
        pass


def _kanban_handler(tool_name: str) -> Callable:
    """Wrap a handler so every failure is a structured tool error.

    ``ValueError`` (invalid board slug, DB validation such as cycle/self-link,
    ``AttachmentTooLarge``) is reported without a traceback; anything else is
    logged with ``logger.exception``.
    """
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(args: dict, **kw) -> str:
            try:
                return fn(args, **kw)
            except ValueError as e:
                return tool_error(f"{tool_name}: {e}")
            except Exception as e:
                logger.exception(f"{tool_name} failed")
                return tool_error(f"{tool_name}: {e}")
        return wrapper
    return deco


def _ok(**fields: Any) -> str:
    return json.dumps({"ok": True, **fields})


def _redact(value: Any) -> str:
    return redact_sensitive_text(str(value), force=True)


def _redact_metadata(metadata: dict) -> Optional[dict]:
    """Redact a metadata dict via a JSON round-trip; None if it can't be re-parsed."""
    try:
        return json.loads(redact_sensitive_text(json.dumps(metadata), force=True))
    except json.JSONDecodeError:
        return None


def _coerce_str_list(
    value: Any, name: str, what: str, *, strip: bool = False
) -> tuple[Any, Optional[str]]:
    """Accept a single string (convenience) or a list/tuple; ``(value, error)``.

    With ``strip`` the items are stringified, stripped, and empties dropped.
    """
    if value is None:
        return None, None
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)):
        return None, tool_error(
            f"{name} must be a list of {what}, got {type(value).__name__}"
        )
    if strip:
        value = [str(x).strip() for x in value if str(x).strip()]
    return value, None


def _parse_bool_arg(args: dict, name: str, *, default: bool = False):
    value = args.get(name)
    if value is None:
        return default, None
    if isinstance(value, bool):
        return value, None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True, None
    if text in {"false", "0", "no"}:
        return False, None
    return default, f"{name} must be a boolean or 'true'/'false'"


def _require_orchestrator_tool(tool_name: str) -> Optional[str]:
    """Runtime guard for orchestrator-only handlers.

    The check_fn already hides these from the worker schema; this catches a
    stale registration or test harness routing a worker here anyway.
    """
    if os.environ.get("HERMES_KANBAN_TASK"):
        return tool_error(
            f"{tool_name} is orchestrator-only; dispatcher-spawned workers "
            "must use kanban_complete, kanban_block, kanban_heartbeat, or "
            "kanban_comment for their assigned task."
        )
    return None


_TASK_FIELDS = (
    "id", "title", "body", "assignee", "status", "tenant", "priority",
    "workspace_kind", "workspace_path", "created_by", "created_at",
    "started_at", "completed_at", "result", "current_run_id",
    "model_override", "provider_override",
)
_TASK_SUMMARY_FIELDS = (
    "id", "title", "assignee", "status", "priority", "tenant",
    "workspace_kind", "workspace_path", "project_id", "created_by",
    "created_at", "started_at", "completed_at", "current_run_id",
    "model_override", "provider_override",
)
_RUN_FIELDS = (
    "id", "profile", "status", "outcome", "summary", "error", "metadata",
    "started_at", "ended_at",
)
_COMMENT_FIELDS = ("author", "body", "created_at")
_EVENT_FIELDS = ("kind", "payload", "created_at", "run_id")
_ATTACHMENT_FIELDS = (
    "id", "filename", "content_type", "size", "uploaded_by", "stored_path",
    "created_at",
)


def _fields(obj: Any, names: tuple[str, ...]) -> dict[str, Any]:
    return {n: getattr(obj, n) for n in names}


def _task_summary_dict(kb, conn, task) -> dict[str, Any]:
    """Compact task shape for board-listing tools."""
    parents = kb.parent_ids(conn, task.id)
    children = kb.child_ids(conn, task.id)
    return {
        **_fields(task, _TASK_SUMMARY_FIELDS),
        "parents": parents,
        "children": children,
        "parent_count": len(parents),
        "child_count": len(children),
    }


# ---------------------------------------------------------------------------
# Goal-mode judge gate
# ---------------------------------------------------------------------------

_GOAL_MODE_BLOCK_ALLOWED_KINDS = frozenset({"dependency", "needs_input"})


def _goal_judge_available() -> bool:
    """True when an auxiliary client is configured for the goal judge.

    ``judge_goal`` fails open: with no reachable auxiliary model it returns
    ``"continue"``, indistinguishable from a real "not done yet". Treating that
    as a rejection would wedge every goal_mode worker, so the completion gate
    is enforced only when a judge is actually reachable (same client lookup
    ``judge_goal`` performs internally).
    """
    try:
        from agent.auxiliary_client import get_text_auxiliary_client
        client, model = get_text_auxiliary_client("goal_judge")
    except Exception:
        return False
    return client is not None and bool(model)


def _goal_mode_handoff_rejection(task, evidence: str):
    """Return ``(verdict, reason_or_None)`` for a goal-mode terminal handoff.

    ``("done", None)`` allows the handoff. Otherwise the verdict picks the
    guidance: ``continue`` = not done yet, ``blocked`` = judged unachievable.
    A broken judge fails open (logged) so it cannot permanently wedge work.
    """
    if not task or not task.goal_mode or not _goal_judge_available():
        return ("done", None)
    verdict = "done"
    reason = ""
    try:
        verdict, reason, _, _, _ = judge_goal(
            goal=f"{task.title}\n\n{task.body or ''}".strip(),
            last_response=evidence.strip(),
        )
    except Exception as judge_exc:
        logger.warning(
            "goal judge check failed, allowing lifecycle handoff: %s",
            judge_exc,
            exc_info=True,
        )
    return (verdict, None if verdict == "done" else reason)


# ---------------------------------------------------------------------------
# Runtime-activity → board bridges (auto-heartbeat, live comment injection)
# ---------------------------------------------------------------------------
# The dispatcher watchdog reads ``tasks.last_heartbeat_at``, not the agent's
# in-process activity timestamp, so normal work (tool calls, stream chunks) is
# mirrored onto the board here; the explicit ``kanban_heartbeat`` tool stays
# for attaching a note or pre-extending a claim across a known-long op.
# Constraints: best-effort (never raise into the agent loop), rate-limited
# per process, no-op outside dispatcher-spawned worker context, no durable
# note on auto-heartbeats.

_AUTO_HEARTBEAT_MIN_INTERVAL_SECONDS = 60.0
_auto_heartbeat_last_attempt: float = 0.0


def heartbeat_current_worker_from_env() -> bool:
    """Best-effort: extend the claim + bump board heartbeat for the current worker.

    Returns True if a write was attempted, False if skipped (not a worker,
    rate-limited, or failed) — informational only. Identity from env:
    ``HERMES_KANBAN_TASK`` (required), ``HERMES_KANBAN_RUN_ID`` (pins the run
    row so a reclaimed stale run is not heartbeated), ``HERMES_KANBAN_CLAIM_LOCK``
    (falls back to the default claimer for locally-driven workers). The
    monotonic rate limit is not strictly thread-safe; a race costs one extra
    harmless DB write.
    """
    global _auto_heartbeat_last_attempt
    tid = os.environ.get("HERMES_KANBAN_TASK")
    if not tid:
        return False
    import time as _time
    now = _time.monotonic()
    if (now - _auto_heartbeat_last_attempt) < _AUTO_HEARTBEAT_MIN_INTERVAL_SECONDS:
        return False
    _auto_heartbeat_last_attempt = now
    try:
        kb, conn = _connect()
        try:
            try:
                kb.heartbeat_claim(conn, tid, claimer=os.environ.get("HERMES_KANBAN_CLAIM_LOCK"))
            except Exception:
                logger.debug("auto-heartbeat: heartbeat_claim failed", exc_info=True)
            try:
                kb.heartbeat_worker(conn, tid, note=None, expected_run_id=_worker_run_id(tid))
            except Exception:
                logger.debug("auto-heartbeat: heartbeat_worker failed", exc_info=True)
        finally:
            _close_quietly(conn)
        return True
    except Exception:
        logger.debug("auto-heartbeat: bridge failed", exc_info=True)
        return False


# Live operator-note injection: poll the worker's task for new comments and
# fold them in via the OUT-OF-BAND steer channel, so a user can talk to a
# running task without block → comment → unblock (or a restart). Polled
# tighter than the heartbeat so notes land within seconds; watermarked per task.
_COMMENT_POLL_MIN_INTERVAL_SECONDS = 6.0
_comment_poll_last_attempt: float = 0.0
# task_id -> highest comment id already seen (seeded on first poll so history
# already present in build_worker_context isn't re-injected).
_comment_watermark: dict[str, int] = {}


def inject_new_comments_from_env(agent: Any) -> bool:
    """Fold new operator comments on the current worker's task into ``agent``.

    Self-gating no-op unless ``HERMES_KANBAN_TASK`` is set and ``agent`` exposes
    ``steer``; returns True iff a steer was injected; never raises. The first
    poll only seeds the watermark (those comments are already in context), and
    the worker's own comments (matched by ``HERMES_PROFILE``) are skipped.
    """
    tid = os.environ.get("HERMES_KANBAN_TASK")
    if not tid or agent is None or not hasattr(agent, "steer"):
        return False
    global _comment_poll_last_attempt
    import time as _time
    now = _time.monotonic()
    if (now - _comment_poll_last_attempt) < _COMMENT_POLL_MIN_INTERVAL_SECONDS:
        return False
    _comment_poll_last_attempt = now

    seen = _comment_watermark.get(tid)
    try:
        kb, conn = _connect()
        try:
            rows = kb.list_comments_after(conn, tid, after_id=seen or 0)
        finally:
            _close_quietly(conn)
    except Exception:
        logger.debug("comment-inject: bridge failed", exc_info=True)
        return False

    if seen is None:
        _comment_watermark[tid] = max((c.id for c in rows), default=0)
        return False
    if not rows:
        return False
    # Advance past everything read (including our own notes) so nothing is re-injected.
    _comment_watermark[tid] = max(c.id for c in rows)

    own = (os.environ.get("HERMES_PROFILE") or "").strip()
    fresh = [c for c in rows if (c.author or "").strip() != own and (c.body or "").strip()]
    if not fresh:
        return False

    lines = [f"- {c.author or 'operator'}: {c.body.strip()}" for c in fresh]
    note = (
        "New note"
        + ("s" if len(fresh) > 1 else "")
        + " on your kanban task from the operator (delivered mid-run). "
        + "Take it into account for the work you're doing right now:\n"
        + "\n".join(lines)
    )
    try:
        return bool(agent.steer(note))
    except Exception:
        logger.debug("comment-inject: steer failed", exc_info=True)
        return False


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

@_kanban_handler("kanban_show")
def _handle_show(args: dict, **kw) -> str:
    """Read a task's full state: row, parents, children, comments, runs, last 50 events."""
    tid = _default_task_id(args.get("task_id"))
    if not tid:
        return tool_error(_TASK_ID_REQUIRED)
    with _board(args.get("board")) as (kb, conn):
        task = kb.get_task(conn, tid)
        if task is None:
            return tool_error(f"task {tid} not found")
        return json.dumps({
            "task": _fields(task, _TASK_FIELDS),
            "parents": kb.parent_ids(conn, tid),
            "children": kb.child_ids(conn, tid),
            "comments": [_fields(c, _COMMENT_FIELDS) for c in kb.list_comments(conn, tid)],
            # Capped; full log via CLI.
            "events": [_fields(e, _EVENT_FIELDS) for e in kb.list_events(conn, tid)[-50:]],
            "runs": [_fields(r, _RUN_FIELDS) for r in kb.list_runs(conn, tid)],
            # Same string build_worker_context hands the dispatcher at spawn time.
            "worker_context": kb.build_worker_context(conn, tid),
        })


@_kanban_handler("kanban_list")
def _handle_list(args: dict, **kw) -> str:
    """List task summaries with the same core filters as the CLI."""
    guard = _require_orchestrator_tool("kanban_list")
    if guard:
        return guard
    include_archived, bool_error = _parse_bool_arg(args, "include_archived")
    if bool_error:
        return tool_error(bool_error)
    limit = args.get("limit")
    if limit is None:
        limit = KANBAN_LIST_DEFAULT_LIMIT
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        return tool_error("limit must be an integer")
    if limit < 1:
        return tool_error("limit must be >= 1")
    if limit > KANBAN_LIST_MAX_LIMIT:
        return tool_error(f"limit must be <= {KANBAN_LIST_MAX_LIMIT}")
    with _board(args.get("board")) as (kb, conn):
        # Match CLI list: dependencies cleared since the last dispatcher tick
        # should be visible to orchestrators immediately.
        promoted = kb.recompute_ready(conn)
        # One extra row lets the output report truncation without dumping the board.
        rows = kb.list_tasks(
            conn,
            assignee=args.get("assignee"),
            status=args.get("status"),
            tenant=args.get("tenant"),
            include_archived=include_archived,
            limit=limit + 1,
        )
        truncated = len(rows) > limit
        tasks = rows[:limit]
        return json.dumps({
            "tasks": [_task_summary_dict(kb, conn, t) for t in tasks],
            "count": len(tasks),
            "limit": limit,
            "truncated": truncated,
            "next_limit": (
                min(limit * 2, KANBAN_LIST_MAX_LIMIT)
                if truncated and limit < KANBAN_LIST_MAX_LIMIT else None
            ),
            "promoted": promoted,
        })


@_kanban_handler("kanban_complete")
def _handle_complete(args: dict, **kw) -> str:
    """Mark the current task done with a structured handoff."""
    tid, err = _worker_guard("kanban_complete", args)
    if err:
        return err
    summary = args.get("summary")
    metadata = args.get("metadata")
    result = args.get("result")
    if summary:
        summary = _redact(summary)
    if result:
        result = _redact(result)
    if isinstance(metadata, dict):
        # Keep the unredacted dict if the redacted JSON cannot be re-parsed.
        redacted = _redact_metadata(metadata)
        if redacted is not None:
            metadata = redacted
    created_cards, err = _coerce_str_list(
        args.get("created_cards"), "created_cards", "task ids", strip=True
    )
    if err:
        return err
    artifacts, err = _coerce_str_list(
        args.get("artifacts"), "artifacts", "file paths", strip=True
    )
    if err:
        return err
    if artifacts:
        # Artifacts ride inside metadata so the completed-event payload needs
        # no DB schema change; the gateway notifier reads payload['artifacts']
        # and uploads each path as a native attachment. Merge with (never
        # overwrite) a metadata.artifacts the worker passed manually.
        if metadata is None:
            metadata = {}
        elif not isinstance(metadata, dict):
            return tool_error(
                f"metadata must be an object/dict, got {type(metadata).__name__}"
            )
        existing = metadata.get("artifacts")
        if isinstance(existing, (list, tuple)):
            merged = (str(item).strip() for item in [*existing, *artifacts])
            metadata["artifacts"] = list(dict.fromkeys(s for s in merged if s))
        else:
            metadata["artifacts"] = artifacts
    if not (summary or result):
        return tool_error("provide at least one of: summary (preferred), result")
    if metadata is not None and not isinstance(metadata, dict):
        return tool_error(
            f"metadata must be an object/dict, got {type(metadata).__name__}"
        )
    metadata = _stamp_worker_session_metadata(tid, metadata)
    with _board(args.get("board")) as (kb, conn):
        # Goal-mode pre-completion judge gate: a worker must not bypass the
        # auxiliary judge by completing before acceptance criteria are met.
        task = kb.get_task(conn, tid)
        gate_verdict, rejection = _goal_mode_handoff_rejection(
            task, (summary or result or "").strip()
        )
        if gate_verdict == "blocked":
            return tool_error(
                f"Goal completion rejected: judge ruled the goal "
                f"unachievable — {rejection}. The task will NOT complete "
                f"silently. Either re-scope the task with kanban_edit, "
                f"or record the block with kanban_block and hand the "
                f"decision to a human / reviewer."
            )
        if rejection is not None:
            return tool_error(
                f"Goal completion rejected by judge: {rejection}. "
                f"To proceed, either: (1) provide explicit acceptance "
                f"evidence in your summary matching the task's criteria, "
                f"or (2) create continuation tasks with parents=[{tid}] "
                f"and keep this task alive."
            )
        try:
            ok = kb.complete_task(
                conn, tid,
                result=result, summary=summary, metadata=metadata,
                created_cards=created_cards,
                expected_run_id=_worker_run_id(tid),
            )
        except kb.ArtifactPreservationError as artifact_err:
            return tool_error(
                f"kanban_complete could not preserve the declared artifacts: "
                f"{artifact_err}. Your task is still in-flight and its "
                f"scratch workspace was kept. Fix the artifact path or "
                f"storage error, then retry kanban_complete with the same handoff."
            )
        except kb.HallucinatedCardsError as hall_err:
            # The gate runs before the write txn, so the task was NOT mutated;
            # say so explicitly or the model treats the error as terminal and
            # blocks/crashes instead of retrying. Audit event already landed.
            return tool_error(
                f"kanban_complete blocked: the following created_cards "
                f"do not exist or were not created by this worker: "
                f"{', '.join(hall_err.phantom)}. "
                f"Your task is still in-flight (no state change). "
                f"Retry kanban_complete with the same summary/metadata "
                f"and either drop these ids from created_cards, or pass "
                f"created_cards=[] to skip the card-claim check entirely."
            )
        if not ok:
            return tool_error(
                f"could not complete {tid} (unknown id or already terminal)"
            )
        run = kb.latest_run(conn, tid)
        return _ok(task_id=tid, run_id=run.id if run else None)


@_kanban_handler("kanban_block")
def _handle_block(args: dict, **kw) -> str:
    """Transition the task to blocked with a reason a human will read."""
    tid, err = _worker_guard("kanban_block", args)
    if err:
        return err
    reason = args.get("reason")
    if not reason or not str(reason).strip():
        return tool_error("reason is required — explain what input you need")
    reason = _redact(reason)
    kind = args.get("kind")
    with _board(args.get("board")) as (kb, conn):
        if kind is not None and kind not in kb.VALID_BLOCK_KINDS:
            return tool_error(
                f"kind must be one of {sorted(kb.VALID_BLOCK_KINDS)} (or omit it)"
            )
        # Goal-mode block gate: the goal loop treats ANY blocked status as
        # terminal, so kanban_block would be an escape hatch around the
        # completion judge. Restrict goal_mode tasks to kinds that are genuine
        # external blockers; everything else routes back through kanban_complete.
        task = kb.get_task(conn, tid)
        if task and task.goal_mode and kind not in _GOAL_MODE_BLOCK_ALLOWED_KINDS:
            return tool_error(
                f"goal_mode tasks can only block with kind in "
                f"{sorted(_GOAL_MODE_BLOCK_ALLOWED_KINDS)} (got {kind!r}). "
                f"If the task is actually finished or cannot proceed for "
                f"another reason, call kanban_complete instead — the "
                f"completion judge will evaluate it."
            )
        ok = kb.block_task(
            conn, tid, reason=reason, kind=kind, expected_run_id=_worker_run_id(tid),
        )
        if not ok:
            return tool_error(
                f"could not block {tid} (unknown id or not in running/ready)"
            )
        run = kb.latest_run(conn, tid)
        # Report where the task actually landed; routing may not leave it in 'blocked'.
        landed = kb.get_task(conn, tid)
        return _ok(
            task_id=tid,
            run_id=run.id if run else None,
            status=landed.status if landed else "blocked",
            block_kind=kind,
        )


@_kanban_handler("kanban_request_review")
def _handle_request_review(args: dict, **kw) -> str:
    """Move implementation into the first-class review phase."""
    tid, err = _worker_guard("kanban_request_review", args)
    if err:
        return err
    summary = args.get("summary")
    if not summary or not str(summary).strip():
        return tool_error(
            "summary is required — describe what was implemented and how it "
            "was verified so the reviewer has context"
        )
    summary = _redact(summary)
    metadata = args.get("metadata")
    if metadata is not None and not isinstance(metadata, dict):
        return tool_error(
            f"metadata must be an object/dict, got {type(metadata).__name__}"
        )
    if metadata is not None:
        metadata = _redact_metadata(metadata)
        if metadata is None:
            return tool_error("metadata could not be safely serialized")
    metadata = _stamp_worker_session_metadata(tid, metadata)
    reviewer = args.get("reviewer") or None
    if reviewer:
        # Model-supplied free text stored durably on the event payload.
        reviewer = _redact(reviewer)
    with _board(args.get("board")) as (kb, conn):
        task = kb.get_task(conn, tid)
        gate_verdict, rejection = _goal_mode_handoff_rejection(task, summary)
        if gate_verdict == "blocked":
            return tool_error(
                f"Goal review handoff rejected: judge ruled the goal "
                f"unachievable — {rejection}. Record the block with "
                f"kanban_block instead of requesting review."
            )
        if rejection is not None:
            return tool_error(
                f"Goal review handoff rejected by judge: {rejection}. "
                "Provide acceptance evidence matching the card before "
                "requesting review."
            )
        ok, fail_reason = kb.request_review(
            conn, tid,
            summary=summary,
            metadata=metadata,
            reviewer=reviewer,
            expected_run_id=_worker_run_id(tid),
            with_reason=True,
        )
        if not ok:
            detail = fail_reason or "unknown id or not in running/ready"
            return tool_error(f"could not request review for {tid}: {detail}")
        run = kb.latest_run(conn, tid)
        landed = kb.get_task(conn, tid)
        return _ok(
            task_id=tid,
            run_id=run.id if run else None,
            status=landed.status if landed else "review",
        )


@_kanban_handler("kanban_request_changes")
def _handle_request_changes(args: dict, **kw) -> str:
    """Return a reviewer-owned running task to its implementer."""
    tid, err = _worker_guard("kanban_request_changes", args)
    if err:
        return err
    reason = args.get("reason")
    if not reason or not str(reason).strip():
        return tool_error("reason is required — describe the changes needed")
    reason = _redact(reason)
    with _board(args.get("board")) as (kb, conn):
        ok, detail = kb.request_changes(
            conn, tid, reason=reason, expected_run_id=_worker_run_id(tid),
        )
        if not ok:
            return tool_error(
                f"could not request changes for {tid}: {detail or 'invalid review state'}"
            )
        landed = kb.get_task(conn, tid)
        run = kb.latest_run(conn, tid)
        return _ok(
            task_id=tid,
            run_id=run.id if run else None,
            status=landed.status if landed else "ready",
            implementer=detail,
        )


@_kanban_handler("kanban_heartbeat")
def _handle_heartbeat(args: dict, **kw) -> str:
    """Signal liveness during a long operation.

    Extends the claim TTL (``heartbeat_claim``) AND records a heartbeat event
    (``heartbeat_worker``). Without the claim half, a worker looping this tool
    while one tool call blocks longer than the claim TTL still gets reclaimed
    by ``release_stale_claims``.
    """
    tid, err = _worker_guard("kanban_heartbeat", args)
    if err:
        return err
    with _board(args.get("board")) as (kb, conn):
        # The dispatcher pins HERMES_KANBAN_CLAIM_LOCK at spawn; the default
        # claimer covers locally-driven workers that bypassed the dispatcher.
        kb.heartbeat_claim(conn, tid, claimer=os.environ.get("HERMES_KANBAN_CLAIM_LOCK"))
        ok = kb.heartbeat_worker(
            conn, tid, note=args.get("note"), expected_run_id=_worker_run_id(tid),
        )
        if not ok:
            return tool_error(f"could not heartbeat {tid} (unknown id or not running)")
        return _ok(task_id=tid)


@_kanban_handler("kanban_comment")
def _handle_comment(args: dict, **kw) -> str:
    """Append a comment to a task's thread."""
    delegated_err = _reject_delegated_child_mutation("kanban_comment")
    if delegated_err:
        return delegated_err
    tid = args.get("task_id")
    if not tid:
        return tool_error(
            "task_id is required (use the current task id if that's what "
            "you mean — pulls from env but kept explicit here)"
        )
    body = args.get("body")
    if not body or not str(body).strip():
        return tool_error("body is required")
    body = _redact(body)
    # Author comes from the worker's runtime identity, never caller args:
    # comments are injected into future workers' system prompts as
    # ``**{author}** (timestamp): {body}``, so an args["author"] override
    # could forge a directive from an authoritative-looking name like
    # ``hermes-system``. Cross-task commenting stays unrestricted — it is the
    # deliberate handoff channel between tasks.
    author = os.environ.get("HERMES_PROFILE") or "worker"
    with _board(args.get("board")) as (kb, conn):
        cid = kb.add_comment(conn, tid, author=author, body=str(body))
        return _ok(task_id=tid, comment_id=cid)


def _store_attachment(kb, board, tid, filename, data, content_type) -> str:
    with _board(board) as (_, conn):
        att_id = kb.store_attachment_bytes(
            conn, tid, str(filename), data,
            content_type=content_type, uploaded_by="agent", board=board,
        )
        return _ok(task_id=tid, attachment_id=att_id, size=len(data))


@_kanban_handler("kanban_attach")
def _handle_attach(args: dict, **kw) -> str:
    """Attach an inline (base64) file to a task.

    Goes through ``kanban_db.store_attachment_bytes`` (decode, shared size cap,
    per-task attachments dir, metadata row) so agent, dashboard, and CLI
    surfaces stay in lockstep.
    """
    from hermes_cli import kanban_db as kb

    tid, err = _worker_guard("kanban_attach", args)
    if err:
        return err
    filename = args.get("filename")
    if not filename or not str(filename).strip():
        return tool_error("filename is required")
    content_b64 = args.get("content_base64")
    if not content_b64 or not str(content_b64).strip():
        return tool_error("content_base64 is required")
    import base64
    import binascii
    try:
        data = base64.b64decode(str(content_b64), validate=True)
    except (binascii.Error, ValueError) as e:
        return tool_error(f"content_base64 is not valid base64: {e}")
    return _store_attachment(kb, args.get("board"), tid, filename, data, args.get("content_type"))


_MAX_ATTACH_URL_REDIRECTS = 5


def _download_url_with_cap(url: str, max_bytes: int) -> tuple[bytes, Optional[str]]:
    """Fetch ``url`` over http(s) with SSRF guarding, capped at ``max_bytes``.

    Every hop (initial URL and each redirect target) is validated with
    ``tools.url_safety.is_safe_url`` before fetching, so a model-controlled URL
    (or a public host 302ing to one) cannot reach loopback, private/CGNAT
    ranges, or cloud metadata. Redirects are followed manually so each
    Location is re-checked. Returns ``(data, content_type)``; raises
    ``ValueError`` for a bad scheme, blocked target, too many redirects, or a
    body over the cap (checked while streaming, so nothing oversize is buffered).
    """
    from urllib.parse import urljoin, urlparse

    import httpx

    from tools.url_safety import is_safe_url

    current_url = url
    for _ in range(_MAX_ATTACH_URL_REDIRECTS + 1):
        scheme = (urlparse(current_url).scheme or "").lower()
        if scheme not in ("http", "https"):
            raise ValueError(
                f"unsupported URL scheme {scheme!r}; only http/https are allowed"
            )
        if not is_safe_url(current_url):
            raise ValueError(
                f"URL blocked by SSRF protection (private/internal address): {current_url}"
            )
        chunks: list[bytes] = []
        total = 0
        with httpx.stream(
            "GET",
            current_url,
            headers={"User-Agent": "hermes-kanban/attach"},
            timeout=30,
            follow_redirects=False,
        ) as resp:
            if resp.is_redirect:
                location = resp.headers.get("location")
                if not location:
                    raise ValueError(f"redirect without Location header from {current_url}")
                current_url = urljoin(current_url, location)
                continue
            resp.raise_for_status()
            content_type = (resp.headers.get("content-type") or "").split(";")[0].strip() or None
            for chunk in resp.iter_bytes(1024 * 1024):
                total += len(chunk)
                if total > max_bytes:
                    raise ValueError(
                        f"attachment exceeds {max_bytes // (1024 * 1024)} MB limit"
                    )
                chunks.append(chunk)
        return b"".join(chunks), content_type
    raise ValueError(f"too many redirects fetching {url}")


@_kanban_handler("kanban_attach_url")
def _handle_attach_url(args: dict, **kw) -> str:
    """Attach a file fetched server-side from an http(s) URL (shared size cap)."""
    from hermes_cli import kanban_db as kb

    tid, err = _worker_guard("kanban_attach_url", args)
    if err:
        return err
    url = args.get("url")
    if not url or not str(url).strip():
        return tool_error("url is required")
    url = str(url).strip()
    filename = args.get("filename") or args.get("title")
    if not filename or not str(filename).strip():
        # Derive a name from the URL path's leaf component.
        from urllib.parse import unquote, urlparse
        leaf = unquote(urlparse(url).path.rsplit("/", 1)[-1]).strip()
        filename = leaf or "download"
    try:
        data, fetched_ct = _download_url_with_cap(url, kb.KANBAN_ATTACHMENT_MAX_BYTES)
    except ValueError as e:
        return tool_error(f"kanban_attach_url: {e}")
    except Exception as e:
        logger.exception("kanban_attach_url download failed")
        return tool_error(f"kanban_attach_url: failed to fetch {url}: {e}")
    return _store_attachment(
        kb, args.get("board"), tid, filename, data, args.get("content_type") or fetched_ct
    )


@_kanban_handler("kanban_attachments")
def _handle_attachments(args: dict, **kw) -> str:
    """List a task's attachments (read-only; no ownership restriction)."""
    tid = _default_task_id(args.get("task_id"))
    if not tid:
        return tool_error(_TASK_ID_REQUIRED)
    with _board(args.get("board")) as (kb, conn):
        if kb.get_task(conn, tid) is None:
            return tool_error(f"task {tid} not found")
        return json.dumps({
            "ok": True,
            "task_id": tid,
            "attachments": [_fields(a, _ATTACHMENT_FIELDS) for a in kb.list_attachments(conn, tid)],
        })


@_kanban_handler("kanban_create")
def _handle_create(args: dict, **kw) -> str:
    """Create a (child) task; orchestrator workers use this to fan out."""
    delegated_err = _reject_delegated_child_mutation("kanban_create")
    if delegated_err:
        return delegated_err
    title = args.get("title")
    if not title or not str(title).strip():
        return tool_error("title is required")
    assignee = args.get("assignee")
    if not assignee:
        return tool_error(
            "assignee is required — name the profile that should execute this "
            "task (the dispatcher will only spawn tasks with an assignee)"
        )
    body = args.get("body")
    parents = args.get("parents") or []
    tenant = args.get("tenant") or os.environ.get("HERMES_TENANT")
    # Prefer the request-scoped api_server origin binding over HERMES_SESSION_ID:
    # the env var is clobbered with a subagent's internal id whenever a child
    # agent is constructed in-process, which would stamp — and later wake —
    # the wrong session. NULL on CLI/dashboard paths that set neither.
    from tools.async_delegation import _current_origin_session_id

    session_id = (
        args.get("session_id")
        or _current_origin_session_id()
        or os.environ.get("HERMES_SESSION_ID")
    )
    priority = args.get("priority")
    # Workspace sharing is always explicit: omitted fields mean a fresh scratch
    # workspace even for a dispatcher-spawned creator — reusing the parent's
    # literal path would let a child mutate review evidence or race its
    # checkout. Project identity is the one safe thing to inherit implicitly
    # (the DB turns it into a fresh per-task worktree).
    workspace_kind = args.get("workspace_kind")
    workspace_path = args.get("workspace_path")
    project_id = args.get("project") or args.get("project_id")
    project_source_task_id = None
    _inherit_project = workspace_kind is None and workspace_path is None
    if workspace_kind is None:
        workspace_kind = "scratch"
    triage, bool_error = _parse_bool_arg(args, "triage")
    if bool_error:
        return tool_error(bool_error)
    idempotency_key = args.get("idempotency_key")
    max_runtime_seconds = args.get("max_runtime_seconds")
    initial_status = args.get("initial_status") or "running"
    skills, err = _coerce_str_list(args.get("skills"), "skills", "skill names")
    if err:
        return err
    goal_mode, goal_bool_error = _parse_bool_arg(args, "goal_mode")
    if goal_bool_error:
        return tool_error(goal_bool_error)
    goal_max_turns = args.get("goal_max_turns")
    model_override = args.get("model")
    provider_override = args.get("provider")
    if provider_override and not model_override:
        return tool_error("'provider' requires 'model' to be set as well")
    parents, err = _coerce_str_list(parents, "parents", "task ids")
    if err:
        return err
    with _board(args.get("board")) as (kb, conn):
        if _inherit_project and project_id is None:
            _self_tid = os.environ.get("HERMES_KANBAN_TASK")
            if _self_tid:
                _self_task = kb.get_task(conn, _self_tid)
                if _self_task is not None and _self_task.project_id:
                    project_id = _self_task.project_id
                    project_source_task_id = _self_task.id
        new_tid = kb.create_task(
            conn,
            title=str(title).strip(),
            body=body,
            assignee=str(assignee),
            parents=tuple(parents),
            tenant=tenant,
            priority=int(priority) if priority is not None else 0,
            workspace_kind=str(workspace_kind),
            workspace_path=workspace_path,
            project_id=project_id,
            project_source_task_id=project_source_task_id,
            triage=triage,
            idempotency_key=idempotency_key,
            max_runtime_seconds=(
                int(max_runtime_seconds) if max_runtime_seconds is not None else None
            ),
            skills=skills,
            model_override=model_override,
            provider_override=provider_override,
            goal_mode=goal_mode,
            goal_max_turns=int(goal_max_turns) if goal_max_turns is not None else None,
            initial_status=str(initial_status),
            created_by=os.environ.get("HERMES_PROFILE") or "worker",
            session_id=session_id,
        )
        new_task = kb.get_task(conn, new_tid)
        subscribed = _maybe_auto_subscribe(conn, new_tid)
        return _ok(
            task_id=new_tid,
            status=new_task.status if new_task else None,
            workspace_kind=new_task.workspace_kind if new_task else None,
            workspace_path=new_task.workspace_path if new_task else None,
            project_id=new_task.project_id if new_task else None,
            subscribed=subscribed,
        )


def _maybe_auto_subscribe(conn: Any, task_id: str) -> bool:
    """Auto-subscribe the calling session to task completion / block events.

    Returns True iff a subscription row was written; surfaced as ``subscribed``
    on kanban_create so an orchestrator can fall back to an explicit
    ``kanban_notify-subscribe`` or polling. Gated by
    ``kanban.auto_subscribe_on_create`` (default True; unreadable config also
    means True).

    Delivery targets:
    - Gateway (telegram/discord/...): ``HERMES_SESSION_PLATFORM`` /
      ``HERMES_SESSION_CHAT_ID`` ContextVars set before dispatch.
    - TUI/desktop: those ContextVars are cleared, but the subprocess inherits
      ``HERMES_SESSION_KEY``; subscribe as ``platform="tui"``, ``chat_id=<key>``
      for the TUI notification poller. ``HERMES_SESSION_ID`` is deliberately
      NOT a fallback — it is set for every CLI/ACP invocation for telemetry and
      would auto-subscribe every CLI run.
    - CLI / cron / tests: no persistent channel, no-op.

    Any failure is logged at WARNING and swallowed: notification bookkeeping
    must never fail the kanban_create the agent is mid-conversation about.
    """
    try:
        cfg = load_config()
        if not cfg_get(cfg, "kanban", "auto_subscribe_on_create", default=True):
            return False
    except Exception:
        pass

    platform = ""
    chat_id = ""
    try:
        from gateway.session_context import get_session_env
        platform = get_session_env("HERMES_SESSION_PLATFORM", "")
        chat_id = get_session_env("HERMES_SESSION_CHAT_ID", "")
        if not platform or not chat_id:
            session_key = (
                get_session_env("HERMES_SESSION_KEY", "")
                or os.environ.get("HERMES_SESSION_KEY", "")
            )
            if not session_key:
                return False  # CLI / cron / test — no persistent channel
            platform = "tui"
            chat_id = session_key
        is_gateway_session = platform != "tui"
        chat_type = get_session_env("HERMES_SESSION_CHAT_TYPE", "") or None
        delivery_mode = "notify+wake" if is_gateway_session else None
        thread_id = get_session_env("HERMES_SESSION_THREAD_ID", "") or None
        user_id = get_session_env("HERMES_SESSION_USER_ID", "") or None
        user_id_alt = get_session_env("HERMES_SESSION_USER_ID_ALT", "") or None
        message_id = get_session_env("HERMES_SESSION_MESSAGE_ID", "") or ""
        notifier_profile = (
            get_session_env("HERMES_SESSION_PROFILE", "")
            or os.environ.get("HERMES_PROFILE")
        )
        if not notifier_profile:
            try:
                from hermes_cli.profiles import get_active_profile_name
                notifier_profile = get_active_profile_name() or "default"
            except Exception:
                notifier_profile = "default"
        delivery_metadata: dict[str, Any] = {}
        if thread_id:
            delivery_metadata["thread_id"] = thread_id
        if chat_type:
            delivery_metadata["chat_type"] = chat_type
        if (
            platform.lower() == "telegram"
            and thread_id
            and (chat_type or "").lower() in {"dm", "direct", "private"}
        ):
            delivery_metadata["telegram_dm_topic_reply_fallback"] = True
            if str(thread_id) not in {"", "1"}:
                delivery_metadata["direct_messages_topic_id"] = str(thread_id)
            if message_id:
                delivery_metadata["telegram_reply_to_message_id"] = str(message_id)

        from hermes_cli import kanban_db as _kb
        _kb.add_notify_sub(
            conn, task_id=task_id,
            platform=platform, chat_id=chat_id,
            thread_id=thread_id, user_id=user_id, user_id_alt=user_id_alt,
            chat_type=chat_type,
            notifier_profile=notifier_profile,
            delivery_mode=delivery_mode,
            delivery_metadata=delivery_metadata or None,
        )
        return True
    except Exception as _exc:
        logger.warning(
            "_maybe_auto_subscribe failed: %r (platform=%r key_set=%r)",
            _exc, platform, bool(chat_id),
        )
        return False


@_kanban_handler("kanban_unblock")
def _handle_unblock(args: dict, **kw) -> str:
    """Transition a blocked task to ready, or todo while parents remain open."""
    delegated_err = _reject_delegated_child_mutation("kanban_unblock")
    if delegated_err:
        return delegated_err
    guard = _require_orchestrator_tool("kanban_unblock")
    if guard:
        return guard
    tid = args.get("task_id")
    if not tid:
        return tool_error("task_id is required")
    ownership_err = _enforce_worker_task_ownership(str(tid))
    if ownership_err:
        return ownership_err
    with _board(args.get("board")) as (kb, conn):
        ok = kb.unblock_task(conn, str(tid))
        if not ok:
            return tool_error(f"could not unblock {tid} (not blocked or unknown)")
        task = kb.get_task(conn, str(tid))
        return _ok(task_id=str(tid), status=task.status if task else None)


@_kanban_handler("kanban_link")
def _handle_link(args: dict, **kw) -> str:
    """Add a parent→child dependency edge after the fact (cycles/self-links → ValueError)."""
    delegated_err = _reject_delegated_child_mutation("kanban_link")
    if delegated_err:
        return delegated_err
    parent_id = args.get("parent_id")
    child_id = args.get("child_id")
    if not parent_id or not child_id:
        return tool_error("both parent_id and child_id are required")
    with _board(args.get("board")) as (kb, conn):
        kb.link_tasks(conn, parent_id=parent_id, child_id=child_id)
        return _ok(parent_id=parent_id, child_id=child_id)


# ---------------------------------------------------------------------------
# Registration (order preserved: it is the order tools appear in the schema)
# ---------------------------------------------------------------------------

_TOOLS = (
    ("kanban_show", KANBAN_SHOW_SCHEMA, _handle_show, _check_kanban_mode, "📋"),
    ("kanban_list", KANBAN_LIST_SCHEMA, _handle_list, _check_kanban_orchestrator_mode, "📋"),
    ("kanban_complete", KANBAN_COMPLETE_SCHEMA, _handle_complete, _check_kanban_mode, "✔"),
    ("kanban_block", KANBAN_BLOCK_SCHEMA, _handle_block, _check_kanban_mode, "⏸"),
    ("kanban_request_review", KANBAN_REQUEST_REVIEW_SCHEMA, _handle_request_review, _check_kanban_mode, "👀"),
    ("kanban_request_changes", KANBAN_REQUEST_CHANGES_SCHEMA, _handle_request_changes, _check_kanban_mode, "↩"),
    ("kanban_heartbeat", KANBAN_HEARTBEAT_SCHEMA, _handle_heartbeat, _check_kanban_mode, "💓"),
    ("kanban_comment", KANBAN_COMMENT_SCHEMA, _handle_comment, _check_kanban_mode, "💬"),
    ("kanban_attach", KANBAN_ATTACH_SCHEMA, _handle_attach, _check_kanban_mode, "📎"),
    ("kanban_attach_url", KANBAN_ATTACH_URL_SCHEMA, _handle_attach_url, _check_kanban_mode, "📎"),
    ("kanban_attachments", KANBAN_ATTACHMENTS_SCHEMA, _handle_attachments, _check_kanban_mode, "📎"),
    ("kanban_create", KANBAN_CREATE_SCHEMA, _handle_create, _check_kanban_mode, "➕"),
    ("kanban_unblock", KANBAN_UNBLOCK_SCHEMA, _handle_unblock, _check_kanban_orchestrator_mode, "▶"),
    ("kanban_link", KANBAN_LINK_SCHEMA, _handle_link, _check_kanban_mode, "🔗"),
)

for _name, _sch, _handler, _check_fn, _emoji in _TOOLS:
    registry.register(
        name=_name,
        toolset="kanban",
        schema=_sch,
        handler=_handler,
        check_fn=_check_fn,
        emoji=_emoji,
    )
