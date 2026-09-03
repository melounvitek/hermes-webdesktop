"""SQLite-backed Kanban board for multi-profile, multi-project collaboration.

The board lives under the **shared Hermes root** ``<root>`` (the parent of any
active profile; ``HERMES_HOME`` itself in Docker / custom deployments).
Profiles intentionally collapse onto a shared board — it IS the cross-profile
coordination primitive: a worker spawned with ``hermes -p <profile>`` joins the
same board as the dispatcher that claimed its task.

**Boards:** each extra board is ``<root>/kanban/boards/<slug>/`` with its own
``kanban.db``, ``workspaces/`` and ``logs/``; a worker on one board cannot see
or enumerate others and its dispatcher ticks never touch their DBs. The first
board is ``default`` and, for back-compat, its DB stays at ``<root>/kanban.db``
so pre-boards installs need zero migration (see :func:`kanban_db_path`).

Board resolution order (highest precedence first, all optional):

* ``board=`` argument to :func:`connect` / :func:`init_db` (CLI ``--board``,
  dashboard ``?board=``).
* ``HERMES_KANBAN_BOARD`` env var (dispatcher pins workers to their board).
* ``HERMES_KANBAN_DB`` env var (pins the DB file path directly; legacy
  override, wins when the file path itself is what the caller forces).
* ``<root>/kanban/current`` — one-line slug file written by
  ``hermes kanban boards switch``; absent → ``default``.

Legacy overrides ``HERMES_KANBAN_DB`` / ``HERMES_KANBAN_WORKSPACES_ROOT`` /
``HERMES_KANBAN_HOME`` (umbrella root; tests and unusual deployments) still
work. The dispatcher injects the DB, workspaces-root and board env vars into
worker subprocesses so they converge on the exact DB it claimed from, even
under unusual symlink or Docker layouts.

Schema: tasks, task_links, task_comments, task_events (+ runs, attachments,
notify subs). ``workspace_kind`` decouples coordination from git worktrees so
research / ops workloads run alongside coding. See
``docs/hermes-kanban-v1-spec.pdf``.

Concurrency: WAL + ``BEGIN IMMEDIATE`` write transactions + compare-and-swap
updates on ``tasks.status`` / ``tasks.claim_lock``. SQLite serializes writers,
so at most one claimer wins a task; losers see zero affected rows and move on
— no retry loops, no distributed locks. CAS is per-board (one DB per board).
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import secrets
import sqlite3
import subprocess
import sys
import logging
import time
from contextvars import ContextVar, Token
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from toolsets import get_toolset_names

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared micro-helpers (row access, JSON, env, git)
# ---------------------------------------------------------------------------

def _row_get(row: Any, col: str, default: Any = None) -> Any:
    """``row[col]`` tolerant of the column being absent from the SELECT / schema."""
    if row is None or col not in row.keys():
        return default
    return row[col]


def _json_or(value: Any, default: Any = None) -> Any:
    """Decode a JSON text column; any decode failure or empty value yields ``default``."""
    if not value:
        return default
    try:
        return json.loads(value)
    except Exception:
        return default


def _json_dict(value: Any) -> dict:
    """Decode a JSON text column that must be an object; anything else yields ``{}``."""
    parsed = _json_or(value, {})
    return parsed if isinstance(parsed, dict) else {}


def _env_int(name: str, default: int, *, minimum: int = 0) -> int:
    """Integer env override: absent/empty/non-integer/below ``minimum`` falls back to ``default``."""
    raw = os.environ.get(name, "").strip()
    if raw:
        try:
            parsed = int(raw)
        except ValueError:
            return default
        if parsed >= minimum:
            return parsed
    return default


def _git_out(cwd: Path, *args: str, timeout: int = 30) -> Optional[str]:
    """Run ``git -C cwd args`` and return stripped stdout, or ``None`` on any failure / empty output."""
    try:
        result = subprocess.run(
            ["git", "-C", str(cwd), *args],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=timeout, check=False,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    return (result.stdout or "").strip() or None


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_STATUSES = {"triage", "todo", "scheduled", "ready", "running", "blocked", "review", "done", "archived"}
VALID_INITIAL_STATUSES = {"running", "blocked"}

# Typed block reasons (routing rules live on ``block_task``): ``dependency``
# -> ``todo`` (parent gating promotes it, no human/cron/retry storm);
# ``needs_input`` / ``capability`` -> ``blocked`` for a human; ``transient`` =
# may clear on retry. ``None`` = legacy un-typed block (generic human blocker).
VALID_BLOCK_KINDS = {"dependency", "needs_input", "capability", "transient"}

# Same-reason block -> unblock -> re-block cycles tolerated before the loop
# breaker stops trusting the unblocker (usually a cron) and routes to ``triage``
# for a human decision. Counts manual unblock recurrences, NOT dispatcher
# spawn/crash/timeout failures (that is ``DEFAULT_FAILURE_LIMIT``).
BLOCK_RECURRENCE_LIMIT = 2
VALID_WORKSPACE_KINDS = {"scratch", "worktree", "dir"}


def normalize_reasoning_effort(effort: Optional[str]) -> Optional[str]:
    """Normalize a per-task reasoning effort into a storable level.

    Accepts any level in ``hermes_constants.VALID_REASONING_EFFORTS`` plus
    ``"none"`` (thinking disabled), case-insensitively. Empty / None means
    "inherit the worker profile's own ``agent.reasoning_effort``" and stores
    NULL. Anything else is rejected rather than silently dropped — a typo'd
    level must not quietly hand the task back to the profile default.
    """
    from hermes_constants import VALID_REASONING_EFFORTS

    value = str(effort or "").strip().lower()
    if not value:
        return None
    if value == "none" or value in VALID_REASONING_EFFORTS:
        return value
    allowed = ", ".join(("none", *VALID_REASONING_EFFORTS))
    raise ValueError(
        f"reasoning_effort must be one of {allowed}, got {effort!r}"
    )


KNOWN_TOOLSET_NAMES = frozenset(name.casefold() for name in get_toolset_names())
_IS_WINDOWS = sys.platform == "win32"
KANBAN_ATTACHMENT_MAX_BYTES = 25 * 1024 * 1024


def _assert_not_delegated_child_mutation() -> None:
    """Reject Kanban state mutations from ``delegate_task`` child contexts.

    The structured kanban tools and CLI dispatch layer both have fast-fail
    guards for better UX, but neither is a trust boundary: a delegated child can
    still shell out to the CLI or import this module directly. The actual
    invariant belongs at the DB/filesystem mutation layer so every public
    mutator that uses ``write_txn`` (tasks, runs, comments, attachments,
    dispatcher claims, repair events, subscriptions, GC, etc.) and every board
    metadata mutator fails closed before touching durable state.
    """
    try:
        from agent.delegation_context import is_delegated_child_process_context

        delegated = is_delegated_child_process_context()
    except Exception:
        delegated = bool(os.environ.get("HERMES_DELEGATED_CHILD_CONTEXT"))
    if delegated:
        raise PermissionError(
            "delegate_task child contexts cannot mutate Kanban tasks or boards"
        )


def _fire_kanban_lifecycle_hook(event: str, task_id: str, **fields: Any) -> None:
    """Fire a kanban lifecycle plugin hook, fully best-effort.

    Called by the claim/complete/block transitions AFTER their write txn has
    committed, so plugin code never runs while a SQLite write lock is held and
    always observes durable board state. Any failure (plugins unavailable,
    a plugin raising, import error) is swallowed — a misbehaving observer must
    never break a board state transition.

    ``profile_name`` is resolved from the active HERMES_HOME so dispatcher- and
    worker-side hooks both carry the right profile without the caller plumbing
    it through.
    """
    try:
        from hermes_cli.lifecycle import invoke_hook

        invoke_hook(event, task_id=task_id, profile_name=_hook_profile_name(), **fields)
    except Exception as exc:  # pragma: no cover - defensive
        _log.debug("kanban lifecycle hook %s failed: %s", event, exc)


def _hook_profile_name() -> str:
    """Active profile for hook payloads; ``"default"`` when it cannot be resolved."""
    from hermes_cli.profiles import get_active_profile_name

    try:
        return get_active_profile_name()
    except Exception:
        return "default"


def _kanban_observer_consumed(event: str) -> bool:
    """Return whether any first-party observer or plugin consumes *event*.

    Hot-path short-circuit for the worker-lifecycle / task-mutation /
    dispatch-tick observers (RFC #58548): those fire on every dispatcher
    tick and every task write, so call sites skip payload assembly entirely
    when nothing subscribes. Best-effort — if inspection fails the event is
    treated as unconsumed (the invoke path would fail the same way, and
    these are observers, so dropping is always safe).
    """
    try:
        from hermes_cli.lifecycle import has_hook

        return has_hook(event)
    except Exception:  # pragma: no cover - defensive
        return False


def _fire_worker_spawned_hook(
    conn: sqlite3.Connection,
    task: "Task",
    workspace_path: str,
    pid: Optional[int],
    *,
    board: Optional[str] = None,
) -> None:
    """Fire ``on_kanban_worker_spawned`` for one dispatched spawn.

    Called by the dispatch loop AFTER ``spawn_fn`` returned and the worker
    PID (when one was reported) has been durably persisted — the RFC #58548
    timing contract. Fully best-effort: any failure is swallowed so a
    misbehaving observer can never break the dispatch loop.
    """
    if not _kanban_observer_consumed("on_kanban_worker_spawned"):
        return
    try:
        _fire_kanban_lifecycle_hook(
            "on_kanban_worker_spawned",
            task.id,
            board=board or get_current_board(),
            assignee=task.assignee,
            run_id=_current_run_id(conn, task.id),
            worker_pid=int(pid) if pid else None,
            workspace_path=str(workspace_path),
        )
    except Exception as exc:  # pragma: no cover - defensive
        _log.debug("kanban worker spawned hook failed: %s", exc)


def notify_task_updated(
    conn: sqlite3.Connection,
    task_id: str,
    changed_fields: Iterable[str],
    *,
    board: Optional[str] = None,
) -> None:
    """Fire ``on_kanban_task_updated`` for a committed task-row mutation.

    Task-mutation boundary primitive from RFC #58548: a surface that mutates
    a task row outside the claim/complete/block lifecycle calls this AFTER
    its write txn has committed — including surfaces that write with direct
    SQL and bypass every ``kanban_db`` mutator (the dashboard plugin API's
    priority/title/body editors). ``changed_fields`` carries field NAMES
    only, never values. Observer-only and fully best-effort: it can never
    fail a task mutation, and it costs one ``has_hook`` probe when nothing
    subscribes.
    """
    if not _kanban_observer_consumed("on_kanban_task_updated"):
        return
    try:
        row = conn.execute(
            "SELECT assignee, current_run_id FROM tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
        _fire_kanban_lifecycle_hook(
            "on_kanban_task_updated",
            task_id,
            board=board or get_current_board(),
            assignee=row["assignee"] if row else None,
            run_id=row["current_run_id"] if row else None,
            changed_fields=list(changed_fields),
        )
    except Exception as exc:  # pragma: no cover - defensive
        _log.debug("kanban task updated hook failed: %s", exc)


def _fire_dispatch_tick_hook(
    result: "DispatchResult",
    *,
    board: Optional[str] = None,
    dry_run: bool = False,
) -> None:
    """Fire ``on_kanban_dispatch_tick`` after one dispatcher tick.

    Re-port of PR #56066 per the #64231 batch disposition: renamed to the
    taxonomy form and called by ``dispatch_once`` strictly AFTER
    ``_dispatch_tick_lock`` has been released — the original fired inside
    the lock, so a slow subscriber could extend the single-writer critical
    section and stall a sibling dispatcher's tick. Observer-only and fully
    best-effort: any subscriber failure is swallowed.
    """
    if not _kanban_observer_consumed("on_kanban_dispatch_tick"):
        return
    try:
        from hermes_cli.lifecycle import invoke_hook

        profile_name = _hook_profile_name()
        if board is None:
            try:
                board = get_current_board()
            except Exception:
                board = None
        outcome = "ok"
        if result.skipped_locked:
            outcome = "skipped_locked"
        elif not any((
            result.spawned,
            result.reclaimed,
            result.promoted,
            result.reconciled_orphans,
            result.crashed,
            result.stale,
            result.timed_out,
            result.auto_blocked,
            result.rate_limited,
            result.auto_assigned_default,
            result.respawn_guarded,
            result.skipped_per_profile_capped,
            result.skipped_unassigned,
            result.skipped_nonspawnable,
        )):
            outcome = "idle"
        invoke_hook(
            "on_kanban_dispatch_tick",
            board=board,
            profile_name=profile_name,
            dry_run=bool(dry_run),
            outcome=outcome,
            result=result,
        )
    except Exception as exc:  # pragma: no cover - defensive
        _log.debug("kanban dispatch tick hook failed: %s", exc)


# A running task's claim is valid for 15 minutes by default; after that the
# next dispatcher tick reclaims it. Workers that outlive this window should
# call ``heartbeat_claim(task_id)`` periodically. In practice most kanban
# workloads either finish within 15m, set a longer claim explicitly, or use
# ``HERMES_KANBAN_CLAIM_TTL_SECONDS`` to raise the default claim window for
# long single-call MCP workflows.
DEFAULT_CLAIM_TTL_SECONDS = 15 * 60

# If a worker's PID is still alive but its ``last_heartbeat_at`` is
# older than this when ``release_stale_claims`` runs, treat the worker
# as wedged and reclaim regardless of PID liveness (#29747 gap 3).
# This catches the logic-loop case where the process is technically
# running but not making observable progress.  ``_touch_activity``
# bridges chunk-level liveness into ``last_heartbeat_at`` via #31752,
# so any genuinely active worker keeps its heartbeat fresh as a side
# effect of normal API traffic.
DEFAULT_CLAIM_HEARTBEAT_MAX_STALE_SECONDS = 60 * 60

# Grace added to a claim when a reclaim is deferred because the previous
# host-local worker is still alive after a termination attempt. Releasing the
# claim in that state would spawn a duplicate alongside the surviving worker —
# the runaway seen when a cgroup memory.high throttle parks a worker in
# uninterruptible (D) state, where a pending SIGKILL cannot be delivered until
# the throttle lifts. Holding the claim a short grace and retrying next tick
# stops the duplication; once no duplicate is spawned the pressure eases, the
# signal lands, and the following tick reclaims cleanly.
RECLAIM_DEFER_GRACE_SECONDS = 120


def _resolve_claim_ttl_seconds(ttl_seconds: Optional[int] = None) -> int:
    """Return the effective claim TTL, honoring the kanban env override.

    Explicit call-site values win. Otherwise a positive integer from
    ``HERMES_KANBAN_CLAIM_TTL_SECONDS`` overrides the built-in default.
    Invalid or non-positive env values fall back silently so existing
    installs keep working.
    """
    if ttl_seconds is not None:
        return max(1, int(ttl_seconds))

    return _env_int("HERMES_KANBAN_CLAIM_TTL_SECONDS", DEFAULT_CLAIM_TTL_SECONDS, minimum=1)


# Grace period after a task transitions to ``running`` during which
# ``detect_crashed_workers`` skips the ``_pid_alive`` check. Covers the
# fork() → /proc-visibility window where liveness can transiently report
# False for a freshly-spawned worker. The 15-minute claim TTL still
# catches genuinely-crashed workers; this only suppresses false positives
# during the launch window.
DEFAULT_CRASH_GRACE_SECONDS = 30


# Sentinel exit code a kanban worker uses to signal "I bailed because the
# provider rate-limited / exhausted quota, not because the task failed."
# The dispatcher's reap classifier maps this to a ``rate_limited`` exit kind
# so ``detect_crashed_workers`` can release the task back to ``ready``
# WITHOUT counting a failure (the circuit breaker must never trip on a
# transient throttle). 75 == BSD ``EX_TEMPFAIL`` (sysexits.h) — the
# conventional "temporary failure, retry later" code, and well clear of the
# 0/1/2 codes the worker uses for success / generic failure / usage error.
KANBAN_RATE_LIMIT_EXIT_CODE = 75


def _resolve_crash_grace_seconds() -> int:
    """Return the crash-detection grace period in seconds.

    Reads ``HERMES_KANBAN_CRASH_GRACE_SECONDS`` from the environment;
    falls back to ``DEFAULT_CRASH_GRACE_SECONDS`` when absent, empty,
    non-integer, or negative. A value of 0 restores immediate-reclaim
    behaviour (useful for tests).
    """
    return _env_int("HERMES_KANBAN_CRASH_GRACE_SECONDS", DEFAULT_CRASH_GRACE_SECONDS)


def _resolve_rate_limit_cooldown_seconds() -> int:
    """Return the rate-limit requeue cooldown in seconds.

    Reads ``HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS`` from the environment;
    falls back to ``DEFAULT_RATE_LIMIT_COOLDOWN_SECONDS`` when absent, empty,
    non-integer, or negative. A value of 0 disables the cooldown (re-spawn on
    the next tick) — useful for tests that want to assert the task becomes
    spawnable again immediately.
    """
    return _env_int("HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS", DEFAULT_RATE_LIMIT_COOLDOWN_SECONDS)


# Worker-context caps so build_worker_context() stays bounded on
# pathological boards (retry-heavy tasks, comment storms, giant
# summaries). Values chosen to fit a typical 100k-char LLM prompt with
# plenty of headroom. Each constant is tuned independently so users
# who need to relax one don't have to relax all of them.
_CTX_MAX_PRIOR_ATTEMPTS = 10      # most recent N prior runs shown in full
_CTX_MAX_COMMENTS       = 30      # most recent N comments shown in full
_CTX_MAX_FIELD_BYTES    = 4 * 1024   # 4 KB per summary/error/metadata/result
_CTX_MAX_BODY_BYTES     = 8 * 1024   # 8 KB per task.body (opening post)
_CTX_MAX_COMMENT_BYTES  = 2 * 1024   # 2 KB per comment


def _relative_age(ts: Optional[int], now: Optional[int] = None) -> str:
    """Render the age of an epoch-seconds timestamp as a coarse, human-
    readable string like ``just now``, ``18h ago``, ``3d ago``.

    Workers read parent handoffs, comments, and prior-attempt summaries as
    if they describe *current* state. A bare absolute timestamp
    (``2026-06-25 14:30``) doesn't make an LLM reason about staleness — it
    reads the content as fact regardless of how old it is. A relative age
    ("18h ago") is the signal that prompts the worker to re-verify against
    the live source before acting on stale sibling work. Returns an empty
    string for missing/invalid timestamps so callers can append
    unconditionally.
    """
    if ts is None:
        return ""
    try:
        ts = int(ts)
    except (TypeError, ValueError):
        return ""
    if now is None:
        now = int(time.time())
    delta = now - ts
    if delta < 0:
        # Clock skew across machines/profiles — don't claim "in the future".
        return "just now"
    if delta < 60:
        return "just now"
    if delta < 3600:
        m = delta // 60
        return f"{m}m ago"
    if delta < 86400:
        h = delta // 3600
        return f"{h}h ago"
    d = delta // 86400
    return f"{d}d ago"


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DEFAULT_BOARD = "default"
_CURRENT_BOARD_OVERRIDE: ContextVar[str | None] = ContextVar(
    "hermes_kanban_current_board_override",
    default=None,
)


@contextlib.contextmanager
def scoped_current_board(slug: str):
    """Temporarily pin the active board for the current context only."""
    token: Token[str | None] = _CURRENT_BOARD_OVERRIDE.set(slug)
    try:
        yield
    finally:
        _CURRENT_BOARD_OVERRIDE.reset(token)

# Slug validator: lowercase alphanumerics, digits, hyphens; 1–64 chars.
# Strict enough to stop traversal (`..`) and embedded path separators, loose
# enough that kebab-case names like ``atm10-server`` or ``hermes-agent``
# pass without fuss. Board names with display formatting (spaces, emoji)
# live in ``board.json``; the slug is just the directory name.
_BOARD_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9\-_]{0,63}$")


def _normalize_board_slug(slug: Optional[str]) -> Optional[str]:
    """Lowercase + strip a slug; validate; return ``None`` for empty."""
    if slug is None:
        return None
    s = str(slug).strip().lower()
    if not s:
        return None
    if not _BOARD_SLUG_RE.match(s):
        raise ValueError(
            f"invalid board slug {slug!r}: must be 1-64 chars, lowercase "
            f"alphanumerics / hyphens / underscores, not starting with '-' or '_'"
        )
    return s


def kanban_home() -> Path:
    """Return the shared Hermes root that anchors the kanban board.

    Resolution order:

    1. ``HERMES_KANBAN_HOME`` env var when set and non-empty (explicit
       override for tests and unusual deployments).
    2. ``get_default_hermes_root()``, which already returns ``<root>``
       when ``HERMES_HOME`` is ``<root>/profiles/<name>``, and returns
       ``HERMES_HOME`` directly for Docker / custom deployments.

    The kanban board is shared across profiles **by design** (see the
    module docstring). Resolving the kanban paths through the active
    profile's ``HERMES_HOME`` would silently fork the board per profile,
    which breaks the dispatcher / worker handoff.
    """
    override = os.environ.get("HERMES_KANBAN_HOME", "").strip()
    if override:
        return Path(override).expanduser()
    from hermes_constants import get_default_hermes_root
    return get_default_hermes_root()


def boards_root() -> Path:
    """Return ``<root>/kanban/boards`` — the parent of non-default board dirs.

    ``default`` is intentionally NOT under this directory — its DB lives at
    ``<root>/kanban.db`` for back-compat with pre-boards installs. This
    function returns the directory where *additional* named boards live,
    used by :func:`list_boards` to enumerate them.
    """
    return kanban_home() / "kanban" / "boards"


def current_board_path() -> Path:
    """Return the path to ``<root>/kanban/current``.

    One-line text file written by ``hermes kanban boards switch <slug>``
    to persist the user's board selection across CLI invocations. Absent
    by default (meaning: active board is ``default``).
    """
    return kanban_home() / "kanban" / "current"


def get_current_board() -> str:
    """Return the active board slug, honouring the resolution chain.

    Order (highest precedence first):

    1. ``HERMES_KANBAN_BOARD`` env var (set by the dispatcher on worker
       spawn, or manually for ad-hoc overrides).
    2. ``<root>/kanban/current`` on disk (set by ``hermes kanban boards
       switch``), but only when that board still exists.
    3. ``DEFAULT_BOARD`` (``"default"``).

    A malformed or stale slug at any step falls through to the next layer
    with a best-effort warning — the dispatcher must never crash because a
    user hand-edited a file or removed a board directory.
    """
    def _existing(candidate: str) -> Optional[str]:
        if not candidate:
            return None
        try:
            normed = _normalize_board_slug(candidate)
        except ValueError:
            return None
        return normed if normed and board_exists(normed) else None

    for candidate in (
        (_CURRENT_BOARD_OVERRIDE.get() or "").strip(),
        os.environ.get("HERMES_KANBAN_BOARD", "").strip(),
    ):
        found = _existing(candidate)
        if found:
            return found
    try:
        f = current_board_path()
        if f.exists():
            found = _existing(f.read_text(encoding="utf-8").strip())
            if found:
                return found
    except OSError:
        pass
    return DEFAULT_BOARD


def set_current_board(slug: str) -> Path:
    """Persist ``slug`` as the active board. Returns the file written.

    Writes ``<root>/kanban/current``. The caller should validate the slug
    exists first (via :func:`board_exists`) — this function does not —
    so that ``hermes kanban boards switch <typo>`` returns an error
    instead of silently pointing at nothing.
    """
    _assert_not_delegated_child_mutation()
    normed = _normalize_board_slug(slug)
    if not normed:
        raise ValueError("board slug is required")
    path = current_board_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(normed + "\n", encoding="utf-8")
    return path


def clear_current_board() -> None:
    """Remove ``<root>/kanban/current`` so the active board reverts to ``default``."""
    _assert_not_delegated_child_mutation()
    with contextlib.suppress(FileNotFoundError):
        current_board_path().unlink()


def board_dir(board: Optional[str] = None) -> Path:
    """Return the on-disk directory for ``board``.

    ``default`` is ``<root>/kanban/boards/default/`` **for metadata only**
    (board.json + workspaces/ + logs/). Its DB file stays at
    ``<root>/kanban.db`` for back-compat — see :func:`kanban_db_path`.

    All other boards live at ``<root>/kanban/boards/<slug>/`` with
    everything inside that directory including the ``kanban.db``.
    """
    slug = _normalize_board_slug(board) or DEFAULT_BOARD
    return boards_root() / slug


def board_exists(board: Optional[str] = None) -> bool:
    """Return True if the board has persisted metadata or a DB on disk.

    ``default`` is considered to always exist — its DB is created
    on first :func:`connect` and there's no way for it to be missing
    in a configuration where the kanban feature is usable at all.
    """
    slug = _normalize_board_slug(board) or DEFAULT_BOARD
    if slug == DEFAULT_BOARD:
        return True
    d = board_dir(slug)
    return (d / "board.json").exists() or (d / "kanban.db").exists()


def _board_path(
    env_var: Optional[str], board: Optional[str], default_parts: tuple[str, ...], leaf: str,
) -> Path:
    """Shared resolver: ``env_var`` override, else legacy ``<root>/<default_parts>``
    for the ``default`` board, else ``board_dir(slug)/leaf``."""
    if env_var:
        override = os.environ.get(env_var, "").strip()
        if override:
            return Path(override).expanduser()
    slug = _normalize_board_slug(board)
    if slug is None:
        slug = get_current_board()
    if slug == DEFAULT_BOARD:
        return kanban_home().joinpath(*default_parts)
    return board_dir(slug) / leaf


def kanban_db_path(board: Optional[str] = None) -> Path:
    """Return the path to the ``kanban.db`` for ``board``.

    Resolution (highest precedence first):

    1. ``HERMES_KANBAN_DB`` env var — pins the path directly. Honoured for
       back-compat and for the dispatcher→worker handoff (defense in
       depth: dispatcher injects this into worker env so workers are
       immune to any path-resolution disagreement).
    2. When ``board`` arg is None, the active board from
       :func:`get_current_board` is used.
    3. Board ``default`` → ``<root>/kanban.db`` (back-compat path).
       Other boards → ``<root>/kanban/boards/<slug>/kanban.db``.
    """
    return _board_path("HERMES_KANBAN_DB", board, ("kanban.db",), "kanban.db")


def workspaces_root(board: Optional[str] = None) -> Path:
    """Return the directory under which ``scratch`` workspaces are created.

    Anchored per-board so workspaces don't leak between projects.
    ``HERMES_KANBAN_WORKSPACES_ROOT`` pins the path directly (highest
    precedence) — the dispatcher injects this into worker env.

    ``default`` keeps the legacy path ``<root>/kanban/workspaces/`` so
    that existing scratch workspaces from before the boards feature are
    preserved. Other boards use ``<root>/kanban/boards/<slug>/workspaces/``.
    """
    return _board_path("HERMES_KANBAN_WORKSPACES_ROOT", board, ("kanban", "workspaces"), "workspaces")


def attachments_root(board: Optional[str] = None) -> Path:
    """Return the directory under which task file attachments are stored.

    Mirrors :func:`worker_logs_dir` / :func:`workspaces_root`: anchored
    per-board so attachments don't leak between projects. Each task gets
    its own ``<root>/.../attachments/<task_id>/`` subdirectory.

    ``HERMES_KANBAN_ATTACHMENTS_ROOT`` pins the path directly (highest
    precedence) for tests and unusual deployments.

    ``default`` uses ``<root>/kanban/attachments/``; other boards use
    ``<root>/kanban/boards/<slug>/attachments/``.

    Workers (which run with full file-tool access) read attached files
    by the absolute path surfaced in :func:`build_worker_context`. On the
    local terminal backend — the default for kanban — that path resolves
    directly. Remote backends (Docker/Modal) need this directory mounted;
    see the kanban docs.
    """
    return _board_path("HERMES_KANBAN_ATTACHMENTS_ROOT", board, ("kanban", "attachments"), "attachments")


def task_attachments_dir(task_id: str, board: Optional[str] = None) -> Path:
    """Return the per-task attachment directory ``<root>/<task_id>/``."""
    return attachments_root(board=board) / task_id


def worker_logs_dir(board: Optional[str] = None) -> Path:
    """Return the directory under which per-task worker logs are written.

    ``default`` keeps the legacy path ``<root>/kanban/logs/``. Other
    boards use ``<root>/kanban/boards/<slug>/logs/``. Logs follow the
    board — makes ``hermes kanban log`` unambiguous even when multiple
    boards have tasks with the same id.
    """
    return _board_path(None, board, ("kanban", "logs"), "logs")


def board_metadata_path(board: Optional[str] = None) -> Path:
    """Return the path to ``board.json`` for ``board``.

    Stores display metadata (display name, description, icon, color,
    created_at). The on-disk slug is the canonical identity; this file
    is purely for presentation in the CLI / dashboard.
    """
    slug = _normalize_board_slug(board) or DEFAULT_BOARD
    return board_dir(slug) / "board.json"


def _default_board_display_name(slug: str) -> str:
    """Turn a slug into a reasonable default display name.

    ``atm10-server`` → ``Atm10 Server``. Users can override via
    ``board.json`` but the default should look presentable in the
    dashboard without any follow-up editing.
    """
    return " ".join(part.capitalize() for part in slug.replace("_", "-").split("-") if part) or slug


def read_board_metadata(board: Optional[str] = None) -> dict:
    """Return ``board.json`` contents (or synthesized defaults).

    Never raises — a missing / malformed ``board.json`` falls back to a
    synthesised entry so the dashboard always has something to render.
    Includes the canonical ``slug`` and ``db_path`` so the caller
    doesn't need to reconstruct them.
    """
    slug = _normalize_board_slug(board) or DEFAULT_BOARD
    meta: dict[str, Any] = {
        "slug": slug,
        "name": _default_board_display_name(slug),
        "description": "",
        "icon": "",
        "color": "",
        "default_workdir": None,
        # Optional first-class Project this board is scoped to. When set, new
        # tasks inherit it (deterministic worktree + branch under the project's
        # primary repo) and ``default_workdir`` mirrors the project's primary
        # path so the persistent-workspace inheritance path keeps working.
        "project_id": None,
        "created_at": None,
        "archived": False,
    }
    try:
        p = board_metadata_path(slug)
        if p.exists():
            raw = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                # Never let the metadata file claim a different slug than
                # its directory — trust the filesystem.
                raw["slug"] = slug
                meta.update(raw)
    except (OSError, json.JSONDecodeError):
        pass
    meta["db_path"] = str(kanban_db_path(slug))
    return meta


def write_board_metadata(
    board: Optional[str],
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    icon: Optional[str] = None,
    color: Optional[str] = None,
    archived: Optional[bool] = None,
    default_workdir: Optional[str] = None,
    project_id: Optional[str] = None,
) -> dict:
    """Create / update ``board.json`` for ``board``.

    Preserves any existing fields not mentioned in the call. Sets
    ``created_at`` on first write. Returns the resulting metadata dict.

    ``project_id``: ``None`` leaves it unchanged; empty string clears the
    project scope; a value sets it (not validated here — the caller resolves
    it against ``projects_db``).
    """
    _assert_not_delegated_child_mutation()
    slug = _normalize_board_slug(board) or DEFAULT_BOARD
    meta = read_board_metadata(slug)
    # Preserve existing DB-derived fields — they get re-computed each
    # read but shouldn't be written into board.json.
    meta.pop("db_path", None)
    if name is not None:
        meta["name"] = str(name).strip() or _default_board_display_name(slug)
    if description is not None:
        meta["description"] = str(description)
    if icon is not None:
        meta["icon"] = str(icon)
    if color is not None:
        meta["color"] = str(color)
    if archived is not None:
        meta["archived"] = bool(archived)
    if default_workdir is not None:
        meta["default_workdir"] = str(default_workdir) if default_workdir else None
    if project_id is not None:
        meta["project_id"] = str(project_id) if project_id else None
    if not meta.get("created_at"):
        meta["created_at"] = int(time.time())
    path = board_metadata_path(slug)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    meta["db_path"] = str(kanban_db_path(slug))
    return meta


def create_board(
    slug: str,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    icon: Optional[str] = None,
    color: Optional[str] = None,
    default_workdir: Optional[str] = None,
    project_id: Optional[str] = None,
) -> dict:
    """Create a new board directory + DB + metadata. Idempotent.

    Returns the resulting metadata. Raises :class:`ValueError` for a
    malformed slug; returns the existing metadata (not an error) if the
    board already exists — matching ``mkdir -p`` semantics.
    """
    normed = _normalize_board_slug(slug)
    if not normed:
        raise ValueError("board slug is required")
    meta = write_board_metadata(
        normed,
        name=name,
        description=description,
        icon=icon,
        color=color,
        default_workdir=default_workdir,
        project_id=project_id,
    )
    # Touch the DB so list_boards() sees it immediately.
    init_db(board=normed)
    return meta


def list_boards(*, include_archived: bool = True) -> list[dict]:
    """Enumerate all boards that exist on disk.

    Always includes ``default`` (even when the ``boards/default/``
    metadata dir doesn't exist, because its DB is at the legacy path).
    Other boards are discovered by scanning ``boards/`` for subdirectories
    that either contain a ``kanban.db`` or a ``board.json``.

    Returns a list of metadata dicts, sorted with ``default`` first and
    the rest alphabetically.
    """
    entries: list[dict] = []
    seen: set[str] = set()

    # Default board is always first.
    entries.append(read_board_metadata(DEFAULT_BOARD))
    seen.add(DEFAULT_BOARD)

    root = boards_root()
    if root.is_dir():
        for child in sorted(root.iterdir(), key=lambda p: p.name.lower()):
            if not child.is_dir():
                continue
            slug = child.name
            # Keep slug normalisation soft for discovery — but skip dirs
            # that don't parse as valid slugs so we don't surface junk.
            try:
                normed = _normalize_board_slug(slug)
            except ValueError:
                continue
            if not normed or normed in seen:
                continue
            has_db = (child / "kanban.db").exists()
            has_meta = (child / "board.json").exists()
            if not (has_db or has_meta):
                continue
            meta = read_board_metadata(normed)
            if meta.get("archived") and not include_archived:
                continue
            entries.append(meta)
            seen.add(normed)
    return entries


def remove_board(slug: str, *, archive: bool = True) -> dict:
    """Remove or archive a board.

    ``archive=True`` (default) moves the board's directory to
    ``<root>/kanban/boards/_archived/<slug>-<timestamp>/`` so the data
    is recoverable. ``archive=False`` deletes the directory outright.

    The ``default`` board cannot be removed — raises :class:`ValueError`.
    Returns a summary dict describing what happened (``{"slug", "action",
    "new_path"}``).
    """
    _assert_not_delegated_child_mutation()
    normed = _normalize_board_slug(slug)
    if not normed:
        raise ValueError("board slug is required")
    if normed == DEFAULT_BOARD:
        raise ValueError("the 'default' board cannot be removed")
    d = board_dir(normed)
    if not d.exists():
        raise ValueError(f"board {normed!r} does not exist")

    # If the user removed the currently-active board, revert to default.
    if get_current_board() == normed:
        clear_current_board()

    # A concurrent connect(board=normed) after the rename/delete recreates
    # an empty sqlite file via mkdir(exist_ok=True); the cache entry must be
    # dropped first so the schema init pass re-runs on that fresh file.
    _INITIALIZED_PATHS.discard(str((d / "kanban.db").resolve()))

    if archive:
        archive_root = boards_root() / "_archived"
        archive_root.mkdir(parents=True, exist_ok=True)
        ts = int(time.time())
        target = archive_root / f"{normed}-{ts}"
        # Avoid collision on rapid double-archives.
        suffix = 1
        while target.exists():
            target = archive_root / f"{normed}-{ts}-{suffix}"
            suffix += 1
        d.rename(target)
        return {"slug": normed, "action": "archived", "new_path": str(target)}
    import shutil
    shutil.rmtree(d)
    return {"slug": normed, "action": "deleted", "new_path": ""}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Task:
    """In-memory view of a row from the ``tasks`` table."""

    id: str
    title: str
    body: Optional[str]
    assignee: Optional[str]
    status: str
    priority: int
    created_by: Optional[str]
    created_at: int
    started_at: Optional[int]
    completed_at: Optional[int]
    workspace_kind: str
    workspace_path: Optional[str]
    claim_lock: Optional[str]
    claim_expires: Optional[int]
    tenant: Optional[str]
    branch_name: Optional[str] = None
    project_id: Optional[str] = None
    result: Optional[str] = None
    idempotency_key: Optional[str] = None
    # Column semantics are documented on SCHEMA_SQL. Pre-rename columns:
    # ``spawn_failures`` -> consecutive_failures, ``last_spawn_error`` ->
    # last_failure_error (see ``from_row`` fallbacks).
    consecutive_failures: int = 0
    worker_pid: Optional[int] = None
    last_failure_error: Optional[str] = None
    max_runtime_seconds: Optional[int] = None
    last_heartbeat_at: Optional[int] = None
    current_run_id: Optional[int] = None
    workflow_template_id: Optional[str] = None
    current_step_key: Optional[str] = None
    skills: Optional[list] = None            # None = defaults only; [] = explicitly none
    model_override: Optional[str] = None
    provider_override: Optional[str] = None  # provider ``model_override`` belongs to
    reasoning_effort: Optional[str] = None   # VALID_REASONING_EFFORTS | "none"; NULL = profile's
    # Failure count at which the breaker trips (1 = block on first failure);
    # None -> ``kanban.failure_limit`` config -> DEFAULT_FAILURE_LIMIT.
    max_retries: Optional[int] = None
    # Ralph-style goal loop (same engine as ``/goal``): a judge model re-checks
    # the worker's response against title/body each turn and feeds a
    # continuation prompt IN THE SAME SESSION until done, budget exhausted
    # (-> kanban_block) or explicit block/complete. ``goal_max_turns`` None ->
    # ``goals.DEFAULT_MAX_TURNS``.
    goal_mode: bool = False
    goal_max_turns: Optional[int] = None
    # Originating agent session (``HERMES_SESSION_ID``); NULL from CLI/dashboard.
    session_id: Optional[str] = None
    # Typed block reason (VALID_BLOCK_KINDS) or None for legacy blocks; kept
    # across unblock so a same-kind re-block is recognisable as a loop.
    block_kind: Optional[str] = None
    block_recurrences: int = 0               # unblock-loop counter, see BLOCK_RECURRENCE_LIMIT

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Task":
        g = lambda col, default=None: _row_get(row, col, default)  # noqa: E731
        parsed = _json_or(g("skills"))
        skills_value = [str(s) for s in parsed if s] if isinstance(parsed, list) else None
        return cls(
            id=row["id"],
            title=row["title"],
            body=row["body"],
            assignee=row["assignee"],
            status=row["status"],
            priority=row["priority"],
            created_by=row["created_by"],
            created_at=row["created_at"],
            started_at=row["started_at"],
            completed_at=row["completed_at"],
            workspace_kind=row["workspace_kind"],
            workspace_path=row["workspace_path"],
            branch_name=g("branch_name"),
            project_id=g("project_id"),
            claim_lock=row["claim_lock"],
            claim_expires=row["claim_expires"],
            tenant=g("tenant"),
            result=g("result"),
            idempotency_key=g("idempotency_key"),
            # Pre-migration fallbacks (spawn_failures / last_spawn_error) are only
            # reachable on a DB never opened since the rename migration landed.
            consecutive_failures=g("consecutive_failures", g("spawn_failures", 0)),
            worker_pid=g("worker_pid"),
            last_failure_error=g("last_failure_error", g("last_spawn_error")),
            max_runtime_seconds=g("max_runtime_seconds"),
            last_heartbeat_at=g("last_heartbeat_at"),
            current_run_id=g("current_run_id"),
            workflow_template_id=g("workflow_template_id"),
            current_step_key=g("current_step_key"),
            skills=skills_value,
            model_override=g("model_override") or None,
            provider_override=g("provider_override") or None,
            reasoning_effort=g("reasoning_effort") or None,
            max_retries=g("max_retries"),
            goal_mode=bool(g("goal_mode")),
            goal_max_turns=g("goal_max_turns") or None,
            session_id=g("session_id"),
            block_kind=g("block_kind") or None,
            block_recurrences=(
                int(g("block_recurrences")) if g("block_recurrences") is not None else 0
            ),
        )


@dataclass
class Run:
    """In-memory view of a ``task_runs`` row.

    A run is one attempt to execute a task — created on claim, closed
    on complete/block/crash/timeout/spawn_failure/reclaim. Multiple runs
    per task when retries happen. Carries the claim machinery, PID,
    heartbeat, and the structured handoff summary that downstream workers
    read via ``build_worker_context``.
    """

    id: int
    task_id: str
    profile: Optional[str]
    step_key: Optional[str]
    status: str
    claim_lock: Optional[str]
    claim_expires: Optional[int]
    worker_pid: Optional[int]
    max_runtime_seconds: Optional[int]
    last_heartbeat_at: Optional[int]
    started_at: int
    ended_at: Optional[int]
    outcome: Optional[str]
    summary: Optional[str]
    metadata: Optional[dict]
    error: Optional[str]

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Run":
        return cls(
            id=int(row["id"]),
            task_id=row["task_id"],
            profile=row["profile"],
            step_key=row["step_key"],
            status=row["status"],
            claim_lock=row["claim_lock"],
            claim_expires=row["claim_expires"],
            worker_pid=row["worker_pid"],
            max_runtime_seconds=row["max_runtime_seconds"],
            last_heartbeat_at=row["last_heartbeat_at"],
            started_at=int(row["started_at"]),
            ended_at=_opt_int(row["ended_at"]),
            outcome=row["outcome"],
            summary=row["summary"],
            metadata=_json_or(row["metadata"]),
            error=row["error"],
        )


@dataclass
class Comment:
    id: int
    task_id: str
    author: str
    body: str
    created_at: int

    @classmethod
    def from_row(cls, r: sqlite3.Row) -> "Comment":
        return cls(
            id=r["id"], task_id=r["task_id"], author=r["author"],
            body=r["body"], created_at=r["created_at"],
        )


@dataclass
class Attachment:
    """In-memory view of a row from the ``task_attachments`` table."""

    id: int
    task_id: str
    filename: str
    stored_path: str
    content_type: Optional[str]
    size: int
    uploaded_by: Optional[str]
    created_at: int

    @classmethod
    def from_row(cls, r: sqlite3.Row) -> "Attachment":
        return cls(
            id=r["id"], task_id=r["task_id"], filename=r["filename"],
            stored_path=r["stored_path"], content_type=r["content_type"],
            size=r["size"] or 0, uploaded_by=r["uploaded_by"],
            created_at=r["created_at"],
        )


@dataclass
class Event:
    id: int
    task_id: str
    kind: str
    payload: Optional[dict]
    created_at: int
    run_id: Optional[int] = None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Event":
        run_id = _row_get(row, "run_id")
        return cls(
            id=row["id"], task_id=row["task_id"], kind=row["kind"],
            payload=_json_or(row["payload"]), created_at=row["created_at"],
            run_id=_opt_int(run_id),
        )


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS tasks (
    id                   TEXT PRIMARY KEY,
    title                TEXT NOT NULL,
    body                 TEXT,
    assignee             TEXT,
    status               TEXT NOT NULL,
    priority             INTEGER DEFAULT 0,
    created_by           TEXT,
    created_at           INTEGER NOT NULL,
    started_at           INTEGER,
    completed_at         INTEGER,
    workspace_kind       TEXT NOT NULL DEFAULT 'scratch',
    workspace_path       TEXT,
    branch_name          TEXT,
    -- Optional link to a first-class Project (hermes_cli/projects_db). When set,
    -- the task's worktree is anchored under the project's primary repo with a
    -- deterministic branch name instead of a random wt/<task-id> fallback.
    project_id           TEXT,
    claim_lock           TEXT,
    claim_expires        INTEGER,
    tenant               TEXT,
    result               TEXT,
    idempotency_key      TEXT,
    -- Unified consecutive-failure counter. Incremented on spawn
    -- failure, timeout, or crash; reset only on successful completion.
    -- The circuit breaker in _record_task_failure trips when this
    -- exceeds DEFAULT_FAILURE_LIMIT consecutive non-successes.
    consecutive_failures INTEGER NOT NULL DEFAULT 0,
    worker_pid           INTEGER,
    -- Short excerpt of the most recent failure's error text.
    last_failure_error   TEXT,
    max_runtime_seconds  INTEGER,
    last_heartbeat_at    INTEGER,
    -- Pointer into task_runs for the currently-active run (NULL if no
    -- run is in-flight). Denormalised for cheap reads.
    current_run_id       INTEGER,
    -- Forward-compat for v2 workflow routing. In v1 the kernel writes
    -- these when the task is opted into a template but otherwise ignores
    -- them; the dispatcher doesn't consult them for routing yet.
    workflow_template_id TEXT,
    current_step_key     TEXT,
    -- Force-loaded skills for the worker on this task, stored as JSON.
    -- Passed to the worker via `--skills`. NULL or empty array = no extras.
    skills               TEXT,
    -- Per-task model override. When set, the dispatcher passes -m <model>
    -- to the worker, overriding the profile's default model. NULL = use
    -- the profile default.
    model_override       TEXT,
    -- Provider the model override belongs to. When set (alongside
    -- model_override), the dispatcher passes --provider <name> so the
    -- worker resolves the model against the right backend instead of the
    -- profile's configured provider. NULL = profile provider.
    provider_override    TEXT,
    -- Per-task reasoning effort for the worker (minimal|low|medium|high|
    -- xhigh|max|ultra, or 'none' for thinking off). When set, the dispatcher
    -- passes --reasoning <level> so the worker runs at that depth regardless
    -- of the profile's agent.reasoning_effort. NULL = profile setting.
    reasoning_effort     TEXT,
    -- Per-task override for the consecutive-failure circuit breaker.
    -- The value is the failure count at which the breaker trips — e.g.
    -- ``max_retries=1`` blocks on the first failure. NULL (the common
    -- case) falls through to the dispatcher-level ``kanban.failure_limit``
    -- config and then ``DEFAULT_FAILURE_LIMIT``.
    max_retries          INTEGER,
    -- When 1, the dispatched worker runs in a Ralph-style goal loop: an
    -- auxiliary judge re-evaluates the worker's response against the
    -- card title/body after each turn and feeds a continuation prompt
    -- back into the SAME session until the judge agrees the work is done
    -- or ``goal_max_turns`` is exhausted. NULL/0 = classic single-shot
    -- worker (the default).
    goal_mode            INTEGER NOT NULL DEFAULT 0,
    -- Goal-loop turn budget for ``goal_mode`` workers. NULL = use the
    -- goals-engine default.
    goal_max_turns       INTEGER,
    -- Originating chat/agent session id when the task was created from
    -- inside an agent loop that propagated ``HERMES_SESSION_ID``. NULL
    -- for tasks created from the CLI, dashboard, or any path that doesn't
    -- set the env var. Indexed so per-session list queries stay cheap on
    -- larger boards.
    session_id           TEXT,
    -- Typed block reason set by ``block_task`` (one of VALID_BLOCK_KINDS, or
    -- NULL for legacy/un-typed blocks). Drives routing: ``dependency`` never
    -- sits in ``blocked`` (goes to ``todo`` for parent-gating); the others go
    -- to ``blocked`` for a human. Preserved across unblock so a re-block for
    -- the SAME kind can be recognised as a loop.
    block_kind           TEXT,
    -- Unblock-loop counter. Incremented each time a task is re-blocked for the
    -- same truly-blocked reason after having been unblocked. When it reaches
    -- BLOCK_RECURRENCE_LIMIT the task is routed to ``triage`` instead of
    -- ``blocked`` so a cron can't spin it forever. Reset to 0 only on a
    -- successful completion — NOT on unblock (resetting on unblock is exactly
    -- the amnesia that let the loop run unbounded).
    block_recurrences    INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS task_links (
    parent_id  TEXT NOT NULL,
    child_id   TEXT NOT NULL,
    PRIMARY KEY (parent_id, child_id)
);

CREATE TABLE IF NOT EXISTS task_comments (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id    TEXT NOT NULL,
    author     TEXT NOT NULL,
    body       TEXT NOT NULL,
    created_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS task_events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id    TEXT NOT NULL,
    run_id     INTEGER,
    kind       TEXT NOT NULL,
    payload    TEXT,
    created_at INTEGER NOT NULL
);

-- Historical attempt record. Each time the dispatcher claims a task, a
-- new row is created here; claim state, PID, heartbeat, runtime cap,
-- and structured summary all live on the run, not the task. Multiple
-- rows per task id when the task was retried after crash/timeout/block.
-- v2 of the kanban schema will use ``step_key`` to drive per-stage
-- workflow routing; in v1 the column is nullable and unused (kernel
-- ignores it).
CREATE TABLE IF NOT EXISTS task_runs (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id             TEXT NOT NULL,
    profile             TEXT,
    step_key            TEXT,
    status              TEXT NOT NULL,
    -- status: running | done | blocked | crashed | timed_out | failed | released
    claim_lock          TEXT,
    claim_expires       INTEGER,
    worker_pid          INTEGER,
    max_runtime_seconds INTEGER,
    last_heartbeat_at   INTEGER,
    started_at          INTEGER NOT NULL,
    ended_at            INTEGER,
    outcome             TEXT,
    -- outcome: completed | blocked | crashed | timed_out | spawn_failed |
    --          gave_up | reclaimed | (null while still running)
    summary             TEXT,
    metadata            TEXT,
    error               TEXT
);

-- Files attached to a task (PDFs, images, source documents). The blob
-- lives on disk under ``attachments_root(board)/<task_id>/<stored_name>``;
-- this row carries metadata + the absolute ``stored_path`` so the
-- dashboard can list/download and ``build_worker_context`` can surface
-- the absolute path to the worker (which has full file-tool access). See
-- #35338.
CREATE TABLE IF NOT EXISTS task_attachments (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id      TEXT NOT NULL,
    filename     TEXT NOT NULL,
    stored_path  TEXT NOT NULL,
    content_type TEXT,
    size         INTEGER NOT NULL DEFAULT 0,
    uploaded_by  TEXT,
    created_at   INTEGER NOT NULL
);

-- Subscription from a gateway source (platform + chat + thread) to a
-- task. The gateway's kanban-notifier watcher tails task_events and
-- pushes ``completed`` / ``blocked`` / ``spawn_auto_blocked`` events to
-- the original requester so human-in-the-loop workflows close the loop.
CREATE TABLE IF NOT EXISTS kanban_notify_subs (
    task_id       TEXT NOT NULL,
    platform      TEXT NOT NULL,
    chat_id       TEXT NOT NULL,
    thread_id     TEXT NOT NULL DEFAULT '',
    user_id       TEXT,
    user_id_alt   TEXT,
    chat_type     TEXT,
    notifier_profile TEXT,
    delivery_mode TEXT NOT NULL DEFAULT 'notify',
    delivery_metadata TEXT,
    created_at    INTEGER NOT NULL,
    last_event_id INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (task_id, platform, chat_id, thread_id)
);

CREATE INDEX IF NOT EXISTS idx_tasks_assignee_status ON tasks(assignee, status);
CREATE INDEX IF NOT EXISTS idx_tasks_status          ON tasks(status);
CREATE INDEX IF NOT EXISTS idx_links_child           ON task_links(child_id);
CREATE INDEX IF NOT EXISTS idx_links_parent          ON task_links(parent_id);
CREATE INDEX IF NOT EXISTS idx_comments_task         ON task_comments(task_id, created_at);
CREATE INDEX IF NOT EXISTS idx_events_task           ON task_events(task_id, created_at);
CREATE INDEX IF NOT EXISTS idx_runs_task             ON task_runs(task_id, started_at);
CREATE INDEX IF NOT EXISTS idx_runs_status           ON task_runs(status);
CREATE INDEX IF NOT EXISTS idx_attachments_task      ON task_attachments(task_id, created_at);
CREATE INDEX IF NOT EXISTS idx_notify_task           ON kanban_notify_subs(task_id);
"""


# ---------------------------------------------------------------------------
# ID generation
# ---------------------------------------------------------------------------

def _new_task_id() -> str:
    """Generate a short, URL-safe task id.

    4 hex bytes = ~4.3B possibilities. At 10k tasks the collision
    probability is ~1.2e-5; at 100k it's ~1.2e-3. Previously we used 2
    hex bytes (65k possibilities) which hit the birthday paradox hard:
    ~5% collision probability at 1k tasks, ~50% at 10k. Callers that
    care about idempotency should pass ``idempotency_key`` to
    :func:`create_task` rather than rely on id uniqueness.
    """
    return "t_" + secrets.token_hex(4)


def _claimer_id() -> str:
    """Return a ``host:pid`` string that identifies this claimer."""
    import socket
    try:
        host = socket.gethostname() or "unknown"
    except Exception:
        host = "unknown"
    return f"{host}:{os.getpid()}"


def _host_prefix() -> str:
    """``"<host>:"`` prefix shared by every claim lock issued from this host."""
    return f"{_claimer_id().split(':', 1)[0]}:"


# ---------------------------------------------------------------------------
# Task creation / mutation
# ---------------------------------------------------------------------------

def _canonical_assignee(assignee: Optional[str]) -> Optional[str]:
    """Lowercase-assignee normalization for Kanban rows (dashboard/CLI parity)."""
    if assignee is None:
        return None
    from hermes_cli.profiles import normalize_profile_name

    return normalize_profile_name(assignee)


def _resolve_project_link(
    conn: sqlite3.Connection,
    project_id: Optional[str],
    project_source_task_id: Optional[str],
    workspace_kind: str,
    workspace_path: Optional[str],
) -> tuple[Optional[str], Any, Optional[str], str]:
    """Resolve the optional first-class Project link for ``create_task``.

    Returns ``(project_id, project_obj, project_repo, workspace_kind)``. A
    project-linked task is anchored to the project's primary repo as a git
    worktree so its branch can be named deterministically (project slug + task
    id) instead of the random ``wt/<task-id>`` worker fallback. Projects live in
    the creator's per-profile projects.db; the repo path is absolute and the
    branch name pure, so the cross-profile dispatcher needs no projects.db
    access at dispatch time. ``project_repo`` is the primary repo of a
    project-linked worktree task whose path still has to be derived once the
    task id exists. An unresolvable id/slug drops the link (never a dangling
    reference, never a crash).
    """
    project_obj = None
    project_repo: Optional[str] = None
    if project_id is not None:
        project_id = str(project_id).strip() or None
    if project_id:
        from hermes_cli import projects_db as _pdb

        try:
            with _pdb.connect_closing() as _pconn:
                project_obj = _pdb.get_project(_pconn, project_id)
        except Exception:
            project_obj = None
        if project_obj is None and project_source_task_id:
            # Worker profiles have their own projects.db, while the Kanban DB is
            # intentionally shared. Recover routing only from a canonical
            # project-linked source task in this same board. This carries the
            # repo + project branch convention forward without copying or
            # opening the creator profile's project store, and without reusing
            # the source task's literal worktree path.
            source_task = get_task(conn, str(project_source_task_id))
            if (
                source_task is not None
                and source_task.project_id == project_id
                and source_task.workspace_kind == "worktree"
                and source_task.workspace_path
            ):
                source_path = Path(source_task.workspace_path)
                if (
                    source_path.is_absolute()
                    and source_path.name == source_task.id
                    and source_path.parent.name == ".worktrees"
                ):
                    project_slug = None
                    if source_task.branch_name:
                        prefix, separator, leaf = source_task.branch_name.partition("/")
                        if separator and (
                            leaf == source_task.id
                            or leaf.startswith(f"{source_task.id}-")
                        ):
                            try:
                                project_slug = _pdb.normalize_slug(prefix)
                            except ValueError:
                                project_slug = None
                    if project_slug is None:
                        try:
                            project_slug = _pdb.normalize_slug(project_id)
                        except ValueError:
                            project_slug = None
                    if project_slug:
                        project_repo = str(source_path.parent.parent)
                        project_obj = _pdb.Project(
                            id=project_id,
                            slug=project_slug,
                            name=project_slug,
                            created_at=0,
                            primary_path=project_repo,
                        )
                        if workspace_kind == "scratch":
                            workspace_kind = "worktree"

        if project_obj is None:
            # A project id/slug that doesn't resolve must not crash task
            # creation or persist a dangling reference — drop the link and
            # create the task as an ordinary (scratch) task.
            project_id = None
        else:
            # Canonicalise (a slug may have been passed) and anchor the
            # worktree under the project's primary repo.
            project_id = project_obj.id
            if workspace_kind == "scratch" and project_obj.primary_path:
                workspace_kind = "worktree"
            if (
                workspace_kind == "worktree"
                and workspace_path is None
                and project_obj.primary_path
            ):
                # Defer the concrete path to the insert loop: it's a fresh
                # ``<repo>/.worktrees/<task-id>`` dir keyed on the new task id.
                project_repo = str(project_obj.primary_path)
    return project_id, project_obj, project_repo, workspace_kind


def _normalize_task_skills(skills: Optional[Iterable[str]]) -> Optional[list[str]]:
    """Strip, drop empties, dedupe (order-preserving) a per-task skills list.

    Refuses commas inside a single name so a comma-joined string is never
    splattered into one argv slot (the ``hermes --skills X,Y`` comma syntax is
    handled in the dispatcher, not here). Toolset names are rejected all at
    once — agents that confuse skills with toolsets usually pass several
    (``["web", "browser", "terminal"]``) and serial-correcting wastes tokens.
    """
    if skills is None:
        return None
    cleaned: list[str] = []
    seen: set[str] = set()
    toolset_typos: list[str] = []
    for s in skills:
        if not s:
            continue
        name = str(s).strip()
        if not name:
            continue
        if "," in name:
            raise ValueError(
                f"skill name cannot contain comma: {name!r} "
                f"(pass a list of separate names instead of a comma-joined string)"
            )
        if name.casefold() in KNOWN_TOOLSET_NAMES:
            toolset_typos.append(name)
            continue
        if name in seen:
            continue
        seen.add(name)
        cleaned.append(name)
    if toolset_typos:
        quoted = ", ".join(repr(n) for n in toolset_typos)
        noun = "is a toolset name" if len(toolset_typos) == 1 else "are toolset names"
        raise ValueError(
            f"{quoted} {noun}, not skill name(s). "
            "Put toolsets in the assignee profile's `toolsets:` config "
            "instead of per-task skills. Skills are named skill bundles "
            "(e.g. `blogwatcher`, `github-code-review`); toolsets are runtime "
            "capabilities (e.g. `web`, `browser`, `terminal`)."
        )
    return cleaned


def create_task(
    conn: sqlite3.Connection,
    *,
    title: str,
    body: Optional[str] = None,
    assignee: Optional[str] = None,
    created_by: Optional[str] = None,
    workspace_kind: str = "scratch",
    workspace_path: Optional[str] = None,
    branch_name: Optional[str] = None,
    tenant: Optional[str] = None,
    priority: int = 0,
    parents: Iterable[str] = (),
    triage: bool = False,
    idempotency_key: Optional[str] = None,
    max_runtime_seconds: Optional[int] = None,
    skills: Optional[Iterable[str]] = None,
    max_retries: Optional[int] = None,
    model_override: Optional[str] = None,
    provider_override: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
    goal_mode: bool = False,
    goal_max_turns: Optional[int] = None,
    initial_status: str = "running",
    session_id: Optional[str] = None,
    board: Optional[str] = None,
    project_id: Optional[str] = None,
    project_source_task_id: Optional[str] = None,
) -> str:
    """Create a new task and optionally link it under parent tasks; returns its id.

    Status is ``ready`` with no parents (or all parents ``done``), else ``todo``;
    ``triage=True`` forces ``triage`` regardless of parents (a specifier promotes
    it later); ``initial_status="blocked"`` parks it for human-ops review.

    ``idempotency_key``: if a non-archived task with the same key exists its id
    is returned instead of creating a duplicate (retried webhooks/automation).
    ``max_runtime_seconds``: cap before the dispatcher SIGTERMs (then SIGKILLs
    after a grace window) and re-queues; ``None`` = no cap.
    ``skills``: skill names force-loaded into the worker (``hermes --skills``);
    see ``_normalize_task_skills``. ``model_override``/``provider_override`` pin
    the worker model (``-m <model> [--provider <name>]``); provider requires
    model. ``reasoning_effort`` pins thinking depth (``--reasoning <level>``),
    independent of the model override.
    ``project_source_task_id``: internal cross-profile fallback for a
    worker-created child — when the active profile cannot resolve
    ``project_id`` in its own projects.db, a canonical project-linked task in
    this board supplies the repo and branch convention (its literal worktree is
    never reused). See ``_resolve_project_link``.
    """
    model_override = (model_override or "").strip() or None
    provider_override = (provider_override or "").strip() or None
    reasoning_effort = normalize_reasoning_effort(reasoning_effort)
    if provider_override and not model_override:
        raise ValueError("provider_override requires a model_override")
    assignee = _canonical_assignee(assignee)
    if not title or not title.strip():
        raise ValueError("title is required")
    if initial_status not in VALID_INITIAL_STATUSES:
        raise ValueError(
            f"initial_status must be one of {sorted(VALID_INITIAL_STATUSES)}"
        )
    if workspace_kind not in VALID_WORKSPACE_KINDS:
        raise ValueError(
            f"workspace_kind must be one of {sorted(VALID_WORKSPACE_KINDS)}, "
            f"got {workspace_kind!r}"
        )
    if branch_name is not None:
        branch_name = str(branch_name).strip() or None
    if branch_name and workspace_kind != "worktree":
        raise ValueError("branch_name is only valid for worktree workspaces")

    # Inherit the board's scoped project when the caller didn't name one, so a
    # project-scoped board anchors every new task to that project's repo
    # (deterministic worktree + branch) without each surface repeating it.
    if project_id is None:
        try:
            _bmeta = read_board_metadata(board if board else get_current_board())
            _board_project = (_bmeta.get("project_id") or "").strip()
            if _board_project:
                project_id = _board_project
        except Exception:
            pass

    project_id, project_obj, project_repo, workspace_kind = _resolve_project_link(
        conn, project_id, project_source_task_id, workspace_kind, workspace_path
    )

    parents = tuple(p for p in parents if p)

    skills_list = _normalize_task_skills(skills)

    # Idempotency check — return the existing task instead of creating a
    # duplicate. Done BEFORE entering write_txn to keep the fast path fast
    # and to avoid holding a write lock during the lookup. Race is
    # acceptable: two concurrent creators with the same key might both
    # insert, at which point both rows exist but the next lookup stabilises.
    if idempotency_key:
        row = conn.execute(
            "SELECT id FROM tasks WHERE idempotency_key = ? "
            "AND status != 'archived' "
            "ORDER BY created_at DESC LIMIT 1",
            (idempotency_key,),
        ).fetchone()
        if row:
            return row["id"]

    now = int(time.time())

    # Resolve workspace_path from board-level default_workdir when the
    # caller did not specify one explicitly. Board defaults represent
    # persistent project checkouts, so only persistent workspace kinds may
    # inherit them. Scratch workspaces are auto-deleted on completion and
    # must stay under the per-board scratch root created by
    # ``resolve_workspace``; inheriting ``default_workdir`` for a scratch
    # task would point cleanup at the user's source tree (#28818). The
    # containment guard in ``_cleanup_workspace`` is the safety rail, but
    # we also stop the bad state from being created in the first place.
    if (
        workspace_path is None
        and project_repo is None
        and workspace_kind in {"dir", "worktree"}
    ):
        board_slug = board if board else get_current_board()
        board_meta = read_board_metadata(board_slug)
        board_default = board_meta.get("default_workdir")
        if board_default:
            workspace_path = str(board_default)

    # Retry once on the extremely unlikely id collision.
    for attempt in range(2):
        task_id = _new_task_id()
        try:
            # ``allow_nested=True``: graph builders (kanban_swarm.create_swarm)
            # compose create_task calls under one outer commit so the
            # dispatcher can never observe a partially constructed graph.
            with write_txn(conn, allow_nested=True):
                # Parent ids are validated in every mode (even triage) so the
                # eventual link rows don't dangle.
                if parents:
                    missing = _find_missing_parents(conn, parents)
                    if missing:
                        raise ValueError(f"unknown parent task(s): {', '.join(missing)}")
                # Determine task status from parent status, unless the caller
                # parks it directly in blocked for human-ops review or in
                # triage for a specifier.
                if initial_status == "blocked":
                    task_status = "blocked"
                elif triage:
                    task_status = "triage"
                else:
                    task_status = "ready"
                    if parents:
                        # If any parent is not yet done, we're todo.
                        rows = conn.execute(
                            "SELECT status FROM tasks WHERE id IN "
                            "(" + ",".join("?" * len(parents)) + ")",
                            parents,
                        ).fetchall()
                        if any(r["status"] != "done" for r in rows):
                            task_status = "todo"

                # Project-linked worktree: a fresh worktree dir under the repo
                # plus a deterministic branch (project slug + task id). Together
                # these kill the random ``wt/<task-id>`` worker fallback and the
                # unanchored ``.worktrees/<id>`` under the dispatcher's cwd.
                if project_obj is not None and workspace_kind == "worktree":
                    if project_repo and not workspace_path:
                        workspace_path = os.path.join(
                            project_repo, ".worktrees", task_id
                        )
                    if not branch_name:
                        from hermes_cli import projects_db as _pdb

                        try:
                            branch_name = _pdb.branch_name_for(
                                project_obj, task_id, title=title or ""
                            )
                        except Exception:
                            branch_name = None

                conn.execute(
                    """
                    INSERT INTO tasks (
                        id, title, body, assignee, status, priority,
                        created_by, created_at, workspace_kind, workspace_path,
                        branch_name, project_id, tenant, idempotency_key,
                        max_runtime_seconds,
                        skills, max_retries, model_override, provider_override,
                        reasoning_effort,
                        goal_mode, goal_max_turns, session_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        task_id,
                        title.strip(),
                        body,
                        assignee,
                        task_status,
                        priority,
                        created_by,
                        now,
                        workspace_kind,
                        workspace_path,
                        branch_name,
                        project_id,
                        tenant,
                        idempotency_key,
                        _opt_int(max_runtime_seconds),
                        json.dumps(skills_list) if skills_list is not None else None,
                        _opt_int(max_retries),
                        model_override,
                        provider_override,
                        reasoning_effort,
                        1 if goal_mode else 0,
                        _opt_int(goal_max_turns),
                        session_id,
                    ),
                )
                for pid in parents:
                    _link(conn, pid, task_id)
                # Notify-sub inheritance (ACK-edge: the originating channel
                # still hears about a child that BLOCKs, not just the final
                # fan-in) is handled by the single-owner helper below —
                # _inherit_notify_subs copies every routing/delivery column.
                _append_event(
                    conn,
                    task_id,
                    "created",
                    {
                        "assignee": assignee,
                        "status": task_status,
                        "parents": list(parents),
                        "tenant": tenant,
                        "workspace_kind": workspace_kind,
                        "workspace_path": workspace_path,
                        "branch_name": branch_name,
                        "project_id": project_id,
                        "skills": list(skills_list) if skills_list else None,
                        "goal_mode": bool(goal_mode) or None,
                        "model_override": model_override,
                        "provider_override": provider_override,
                    },
                )
                _inherit_notify_subs(conn, task_id, parents, created_at=now)
            return task_id
        except sqlite3.IntegrityError:
            if attempt == 1:
                raise
            # Retry with a fresh id.
            continue
    raise RuntimeError("unreachable")


def _link(conn: sqlite3.Connection, parent_id: str, child_id: str) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO task_links (parent_id, child_id) VALUES (?, ?)",
        (parent_id, child_id),
    )


def _find_missing_parents(conn: sqlite3.Connection, parents: Iterable[str]) -> list[str]:
    parents = list(parents)
    if not parents:
        return []
    placeholders = ",".join("?" * len(parents))
    rows = conn.execute(
        f"SELECT id FROM tasks WHERE id IN ({placeholders})",
        parents,
    ).fetchall()
    present = {r["id"] for r in rows}
    return [p for p in parents if p not in present]


def _inherit_notify_subs(
    conn: sqlite3.Connection,
    child_id: str,
    parents: Iterable[str],
    *,
    created_at: Optional[int] = None,
) -> None:
    """Copy gateway notification subscriptions from parent tasks to a child.

    The inherited subscription starts caught up to the child's current event
    cursor. This makes manual `link_tasks(parent, existing_child)` safe: the
    parent chat receives future child terminal events without replaying the
    child's pre-link history.

    Copies EVERY routing/delivery column (chat_type, user_id_alt,
    delivery_mode, delivery_metadata included) — this helper is the single
    owner of subscription inheritance for create_task, link_tasks, and triage
    decomposition. Omitting columns here silently degrades routing: a
    DM-originated child completion falls back to chat_type='group' and wakes
    a fresh group-scoped session instead of the originating DM (issue #73030).
    """
    parent_ids = tuple(dict.fromkeys(p for p in parents if p))
    if not parent_ids:
        return
    row = conn.execute(
        "SELECT COALESCE(MAX(id), 0) AS cursor FROM task_events WHERE task_id = ?",
        (child_id,),
    ).fetchone()
    cursor = int(row["cursor"] if row is not None else 0)
    placeholders = ",".join("?" * len(parent_ids))
    conn.execute(
        f"""
        INSERT OR IGNORE INTO kanban_notify_subs
            (task_id, platform, chat_id, thread_id, user_id, user_id_alt,
             chat_type, notifier_profile, delivery_mode, delivery_metadata,
             created_at, last_event_id)
        SELECT ?, platform, chat_id, thread_id, user_id, user_id_alt,
               COALESCE(chat_type, 'dm'), notifier_profile,
               COALESCE(delivery_mode, 'notify'), delivery_metadata, ?, ?
          FROM kanban_notify_subs
         WHERE task_id IN ({placeholders})
        """,
        (
            child_id,
            int(created_at if created_at is not None else time.time()),
            cursor,
            *parent_ids,
        ),
    )


def get_task(conn: sqlite3.Connection, task_id: str) -> Optional[Task]:
    row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
    return Task.from_row(row) if row else None


# Canonical sort-order mappings for ``hermes kanban list --sort``.
# Each value is a raw SQL fragment appended after ``ORDER BY``.
VALID_SORT_ORDERS: dict[str, str] = {
    "created": "created_at ASC, id ASC",
    "created-desc": "created_at DESC, id DESC",
    "priority": "priority DESC, created_at ASC",
    "priority-desc": "priority ASC, created_at ASC",
    "status": "status ASC, created_at ASC",
    "assignee": "assignee ASC, created_at ASC",
    "title": "title ASC, id ASC",
    "updated": "started_at DESC NULLS LAST, created_at DESC",
}


def list_tasks(
    conn: sqlite3.Connection,
    *,
    assignee: Optional[str] = None,
    status: Optional[str] = None,
    tenant: Optional[str] = None,
    session_id: Optional[str] = None,
    include_archived: bool = False,
    limit: Optional[int] = None,
    order_by: Optional[str] = None,
    workflow_template_id: Optional[str] = None,
    current_step_key: Optional[str] = None,
) -> list[Task]:
    query = "SELECT * FROM tasks WHERE 1=1"
    params: list[Any] = []
    if assignee is not None:
        query += " AND assignee = ?"
        params.append(_canonical_assignee(assignee))
    if status is not None:
        if status not in VALID_STATUSES:
            raise ValueError(f"status must be one of {sorted(VALID_STATUSES)}")
        query += " AND status = ?"
        params.append(status)
    if tenant is not None:
        query += " AND tenant = ?"
        params.append(tenant)
    if session_id is not None:
        query += " AND session_id = ?"
        params.append(session_id)
    if workflow_template_id is not None:
        query += " AND workflow_template_id = ?"
        params.append(workflow_template_id)
    if current_step_key is not None:
        query += " AND current_step_key = ?"
        params.append(current_step_key)
    if not include_archived and status != "archived":
        query += " AND status != 'archived'"
    if order_by is not None:
        order_by = order_by.strip().lower()
        if order_by not in VALID_SORT_ORDERS:
            raise ValueError(
                f"order_by must be one of {sorted(VALID_SORT_ORDERS.keys())}"
            )
        query += f" ORDER BY {VALID_SORT_ORDERS[order_by]}"
    else:
        query += " ORDER BY priority DESC, created_at ASC"
    if limit:
        query += f" LIMIT {int(limit)}"
    rows = conn.execute(query, params).fetchall()
    return [Task.from_row(r) for r in rows]


def assign_task(conn: sqlite3.Connection, task_id: str, profile: Optional[str]) -> bool:
    """Assign or reassign a task.  Returns True on success.

    Refuses to reassign a task that's currently running (claim_lock set).
    Reassign after the current run completes if needed.
    """
    profile = _canonical_assignee(profile)
    with write_txn(conn):
        row = conn.execute(
            "SELECT status, claim_lock, assignee FROM tasks WHERE id = ?", (task_id,)
        ).fetchone()
        if not row:
            return False
        if row["claim_lock"] is not None and row["status"] == "running":
            raise RuntimeError(
                f"cannot reassign {task_id}: currently running (claimed). "
                "Wait for completion or reclaim the stale lock first."
            )
        if row["assignee"] != profile:
            # The retry guard is scoped to the task/profile combination. A
            # human reassigning the task is an explicit recovery action, so the
            # new profile should not inherit the previous profile's streak.
            conn.execute(
                "UPDATE tasks SET assignee = ?, consecutive_failures = 0, "
                "last_failure_error = NULL WHERE id = ?",
                (profile, task_id),
            )
        else:
            conn.execute("UPDATE tasks SET assignee = ? WHERE id = ?", (profile, task_id))
        _append_event(conn, task_id, "assigned", {"assignee": profile})
    # Task-mutation observer (RFC #58548), fired AFTER the assignment txn
    # has committed so subscribers always observe durable board state.
    notify_task_updated(conn, task_id, ("assignee",))
    return True


def set_model_override(
    conn: sqlite3.Connection,
    task_id: str,
    model: Optional[str],
    provider: Optional[str] = None,
) -> bool:
    """Set (or clear) the per-task model/provider override.

    ``model=None`` (or empty) clears BOTH overrides — the worker falls back
    to its profile's configured model. ``provider`` without ``model`` is
    rejected: a bare provider switch has no defined meaning for the worker
    spawn (``--provider`` alone would re-resolve the profile's model name
    against a different backend, which is exactly the mismatch class this
    feature exists to kill).

    Allowed on any non-archived task, including ``running`` ones — the
    override only takes effect on the NEXT dispatch, so setting it on a
    running task that's about to be reclaimed/retried is the primary
    rate-limit-recovery flow. Returns True on success.
    """
    model = (model or "").strip() or None
    provider = (provider or "").strip() or None
    if provider and not model:
        raise ValueError("provider_override requires a model_override")
    if not model:
        provider = None
    with write_txn(conn):
        status = _task_status(conn, task_id)
        if status is None:
            return False
        if status == "archived":
            raise RuntimeError(f"cannot set model override on archived task {task_id}")
        conn.execute(
            "UPDATE tasks SET model_override = ?, provider_override = ? WHERE id = ?",
            (model, provider, task_id),
        )
        _append_event(
            conn, task_id, "model_override_set",
            {"model": model, "provider": provider},
        )
    # Task-mutation observer (RFC #58548), fired AFTER the txn commits.
    notify_task_updated(conn, task_id, ("model_override", "provider_override"))
    return True


def set_reasoning_effort(
    conn: sqlite3.Connection,
    task_id: str,
    effort: Optional[str],
) -> bool:
    """Set (or clear) the per-task reasoning effort.

    ``effort=None`` (or empty) clears the override — the worker falls back to
    its profile's own ``agent.reasoning_effort``. ``"none"`` is a real value,
    not a clear: it pins thinking OFF for this task.

    Deliberately independent of :func:`set_model_override`: a task may run the
    profile's own model at a different depth, and clearing a model override
    must not silently reset the depth the operator chose. Like the model
    override, it takes effect on the NEXT dispatch, so it is settable on a
    running task. Returns True on success.
    """
    effort = normalize_reasoning_effort(effort)
    with write_txn(conn):
        status = _task_status(conn, task_id)
        if status is None:
            return False
        if status == "archived":
            raise RuntimeError(
                f"cannot set reasoning effort on archived task {task_id}"
            )
        conn.execute(
            "UPDATE tasks SET reasoning_effort = ? WHERE id = ?",
            (effort, task_id),
        )
        _append_event(
            conn, task_id, "reasoning_effort_set", {"reasoning_effort": effort}
        )
    # Task-mutation observer (RFC #58548), fired AFTER the txn commits.
    notify_task_updated(conn, task_id, ("reasoning_effort",))
    return True


# ---------------------------------------------------------------------------
# Links
# ---------------------------------------------------------------------------

def link_tasks(conn: sqlite3.Connection, parent_id: str, child_id: str) -> None:
    if parent_id == child_id:
        raise ValueError("a task cannot depend on itself")
    with write_txn(conn):
        missing = _find_missing_parents(conn, [parent_id, child_id])
        if missing:
            raise ValueError(f"unknown task(s): {', '.join(missing)}")
        if _would_cycle(conn, parent_id, child_id):
            raise ValueError(
                f"linking {parent_id} -> {child_id} would create a cycle"
            )
        _link(conn, parent_id, child_id)
        # If child was ready but parent is not yet done, demote child to todo.
        if _task_status(conn, parent_id) != "done":
            conn.execute(
                "UPDATE tasks SET status = 'todo' WHERE id = ? AND status = 'ready'",
                (child_id,),
            )
        _append_event(
            conn, child_id, "linked",
            {"parent": parent_id, "child": child_id},
        )
        _inherit_notify_subs(conn, child_id, (parent_id,))


def _would_cycle(conn: sqlite3.Connection, parent_id: str, child_id: str) -> bool:
    """Return True if adding parent->child creates a cycle.

    A cycle exists iff ``parent_id`` is already a descendant of
    ``child_id`` via existing parent->child links.  We walk downward
    from ``child_id`` and check whether we reach ``parent_id``.
    """
    seen = set()
    stack = [child_id]
    while stack:
        node = stack.pop()
        if node == parent_id:
            return True
        if node in seen:
            continue
        seen.add(node)
        rows = conn.execute(
            "SELECT child_id FROM task_links WHERE parent_id = ?", (node,)
        ).fetchall()
        stack.extend(r["child_id"] for r in rows)
    return False


def unlink_tasks(conn: sqlite3.Connection, parent_id: str, child_id: str) -> bool:
    with write_txn(conn):
        cur = conn.execute(
            "DELETE FROM task_links WHERE parent_id = ? AND child_id = ?",
            (parent_id, child_id),
        )
        if cur.rowcount:
            _append_event(
                conn, child_id, "unlinked",
                {"parent": parent_id, "child": child_id},
            )
        removed = cur.rowcount > 0
    if removed:
        # Dependency edge removed — re-evaluate promotion eligibility for the
        # child immediately.  Matches the contract of complete_task and
        # unblock_task; without this the child stays stuck in todo until the
        # next dispatcher tick or a manual `hermes kanban recompute` (issue #22459).
        recompute_ready(conn)
    return removed


def _linked_ids(conn: sqlite3.Connection, want: str, where: str, task_id: str) -> list[str]:
    rows = conn.execute(
        f"SELECT {want} FROM task_links WHERE {where} = ? ORDER BY {want}", (task_id,)
    ).fetchall()
    return [r[want] for r in rows]


def parent_ids(conn: sqlite3.Connection, task_id: str) -> list[str]:
    return _linked_ids(conn, "parent_id", "child_id", task_id)


def child_ids(conn: sqlite3.Connection, task_id: str) -> list[str]:
    return _linked_ids(conn, "child_id", "parent_id", task_id)


def task_graph_contexts(
    conn: sqlite3.Connection, task_ids: Iterable[str]
) -> dict[str, dict]:
    """Bulk-load compact direct graph state for graph-aware diagnostics."""
    ordered_ids = list(dict.fromkeys(str(task_id) for task_id in task_ids if task_id))
    contexts = {
        task_id: {"parents": [], "children": []}
        for task_id in ordered_ids
    }
    if not ordered_ids:
        return contexts

    placeholders = ",".join("?" for _ in ordered_ids)
    for row in conn.execute(
        "SELECT l.child_id AS owner_id, t.id, t.title, t.status "
        "FROM task_links l JOIN tasks t ON t.id = l.parent_id "
        f"WHERE l.child_id IN ({placeholders}) ORDER BY l.child_id, t.id",
        tuple(ordered_ids),
    ).fetchall():
        contexts[row["owner_id"]]["parents"].append({
            "id": row["id"],
            "title": row["title"],
            "status": row["status"],
        })
    for row in conn.execute(
        "SELECT l.parent_id AS owner_id, t.id, t.title, t.status "
        "FROM task_links l JOIN tasks t ON t.id = l.child_id "
        f"WHERE l.parent_id IN ({placeholders}) ORDER BY l.parent_id, t.id",
        tuple(ordered_ids),
    ).fetchall():
        contexts[row["owner_id"]]["children"].append({
            "id": row["id"],
            "title": row["title"],
            "status": row["status"],
        })
    return contexts


def task_graph_context(conn: sqlite3.Connection, task_id: str) -> dict:
    """Return compact direct parent/child state for one task."""
    return task_graph_contexts(conn, [task_id])[task_id]


# ---------------------------------------------------------------------------
# Comments & events
# ---------------------------------------------------------------------------

def add_comment(
    conn: sqlite3.Connection, task_id: str, author: str, body: str
) -> int:
    if not body or not body.strip():
        raise ValueError("comment body is required")
    if not author or not author.strip():
        raise ValueError("comment author is required")
    now = int(time.time())
    # ``allow_nested=True``: graph builders (kanban_swarm blackboard seeding)
    # compose comment writes under one outer commit.
    with write_txn(conn, allow_nested=True):
        if not conn.execute(
            "SELECT 1 FROM tasks WHERE id = ?", (task_id,)
        ).fetchone():
            raise ValueError(f"unknown task {task_id}")
        cur = conn.execute(
            "INSERT INTO task_comments (task_id, author, body, created_at) "
            "VALUES (?, ?, ?, ?)",
            (task_id, author.strip(), body.strip(), now),
        )
        _append_event(conn, task_id, "commented", {"author": author, "len": len(body)})
        return int(cur.lastrowid or 0)


def _task_rows(conn: sqlite3.Connection, table: str, task_id: str, order: str) -> list[sqlite3.Row]:
    return conn.execute(
        f"SELECT * FROM {table} WHERE task_id = ? ORDER BY {order}", (task_id,)
    ).fetchall()


def list_comments(conn: sqlite3.Connection, task_id: str) -> list[Comment]:
    return [Comment.from_row(r) for r in _task_rows(conn, "task_comments", task_id, "created_at ASC")]


def list_comments_after(
    conn: sqlite3.Connection, task_id: str, *, after_id: int = 0
) -> list[Comment]:
    """Return comments on ``task_id`` with ``id > after_id`` (ascending).

    Keyed on the monotonic rowid rather than ``created_at`` so a same-second
    burst can't be skipped. Used by the live worker bridge to fold new
    operator notes into a running task without a restart (see
    ``tools.kanban_tools.inject_new_comments_from_env``).
    """
    rows = conn.execute(
        "SELECT id, task_id, author, body, created_at FROM task_comments "
        "WHERE task_id = ? AND id > ? ORDER BY id ASC",
        (task_id, int(after_id)),
    ).fetchall()
    return [Comment.from_row(r) for r in rows]


# ---------------------------------------------------------------------------
# Attachments
# ---------------------------------------------------------------------------

# The attachment size cap is the module-level ``KANBAN_ATTACHMENT_MAX_BYTES``
# (defined near the top of this file) — one constant shared by the dashboard
# HTTP endpoint, the agent toolset, and the CLI so the limit cannot drift
# between surfaces.


class AttachmentTooLarge(ValueError):
    """Raised when an attachment exceeds the configured size cap.

    Subclasses :class:`ValueError` so generic ``except ValueError`` handlers
    (e.g. the dashboard's 400 fallback) still catch it, while callers that
    want a distinct user-facing message (the tool/CLI 413-equivalent) can
    catch it specifically.
    """


def _safe_attachment_name(raw: str) -> str:
    """Reduce a client-supplied filename to a safe basename.

    Strips any directory components (both separators) so a malicious
    ``../../etc/passwd`` or ``C:\\x`` collapses to its leaf. Drops control
    chars and leading dots so we never write a dotfile or a name with
    embedded NULs/newlines. Rejects empty / dotfile-only names. The result
    is only ever joined under the per-task attachments dir, never used
    verbatim as a path from the client.

    Raises :class:`ValueError` on an unusable name; HTTP callers map that
    to a 400.
    """
    name = (raw or "").replace("\\", "/").split("/")[-1].strip()
    name = "".join(ch for ch in name if ch.isprintable() and ch not in "\x00").strip()
    name = name.lstrip(".").strip()
    if not name:
        raise ValueError("invalid attachment filename")
    return name[:200]


def _collision_free_path(dest_dir: Path, safe_name: str) -> Path:
    """Return a path under ``dest_dir`` that doesn't clobber an existing file.

    ``foo.pdf`` → ``foo.pdf``, then ``foo (1).pdf``, ``foo (2).pdf``, …
    ``safe_name`` must already be sanitised via :func:`_safe_attachment_name`.
    """
    stem, dot, ext = safe_name.partition(".")
    candidate = safe_name
    n = 1
    while (dest_dir / candidate).exists():
        candidate = f"{stem} ({n}){dot}{ext}"
        n += 1
    return dest_dir / candidate


def store_attachment_bytes(
    conn: sqlite3.Connection,
    task_id: str,
    filename: str,
    data: bytes,
    *,
    content_type: Optional[str] = None,
    uploaded_by: Optional[str] = None,
    board: Optional[str] = None,
    max_bytes: Optional[int] = None,
) -> int:
    """Validate, size-check, persist a blob, and record its metadata row.

    This is the single write path shared by the dashboard endpoint, the
    agent toolset (``kanban_attach`` / ``kanban_attach_url``), and the CLI
    (``hermes kanban attach``) so name-sanitisation, the size cap, and the
    collision-resolution all behave identically everywhere.

    Steps: enforce ``max_bytes``, sanitise ``filename`` to a safe basename,
    write the bytes under :func:`task_attachments_dir` with a
    collision-free name, then insert the ``task_attachments`` row via
    :func:`add_attachment`. Returns the new attachment id.

    Raises :class:`AttachmentTooLarge` when ``data`` exceeds ``max_bytes``,
    or :class:`ValueError` for a bad filename / unknown task. On any failure
    after the blob is written (e.g. the task disappeared) the orphaned blob
    is removed before re-raising.
    """
    if max_bytes is None:
        max_bytes = KANBAN_ATTACHMENT_MAX_BYTES
    if len(data) > max_bytes:
        raise AttachmentTooLarge(
            f"attachment exceeds {max_bytes // (1024 * 1024)} MB limit"
        )
    safe_name = _safe_attachment_name(filename)
    dest_dir = task_attachments_dir(task_id, board=board)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = _collision_free_path(dest_dir, safe_name)
    dest_path.write_bytes(data)
    try:
        return add_attachment(
            conn,
            task_id,
            filename=dest_path.name,
            stored_path=str(dest_path.resolve()),
            content_type=content_type,
            size=len(data),
            uploaded_by=uploaded_by,
        )
    except Exception:
        # Don't leave an orphan blob if the metadata insert fails (most
        # commonly: the task id doesn't exist).
        with contextlib.suppress(OSError):
            dest_path.unlink(missing_ok=True)
        raise


def add_attachment(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    filename: str,
    stored_path: str,
    content_type: Optional[str] = None,
    size: int = 0,
    uploaded_by: Optional[str] = None,
) -> int:
    """Record a file attachment for a task. Returns the new attachment id.

    The caller is responsible for writing the blob to ``stored_path``
    first (under :func:`task_attachments_dir`); this only persists the
    metadata row and appends an ``attached`` event.
    """
    if not filename or not filename.strip():
        raise ValueError("attachment filename is required")
    if not stored_path or not stored_path.strip():
        raise ValueError("attachment stored_path is required")
    now = int(time.time())
    with write_txn(conn):
        if not conn.execute(
            "SELECT 1 FROM tasks WHERE id = ?", (task_id,)
        ).fetchone():
            raise ValueError(f"unknown task {task_id}")
        cur = conn.execute(
            "INSERT INTO task_attachments "
            "(task_id, filename, stored_path, content_type, size, uploaded_by, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                task_id,
                filename.strip(),
                stored_path,
                content_type,
                int(size),
                uploaded_by,
                now,
            ),
        )
        _append_event(
            conn,
            task_id,
            "attached",
            {"filename": filename.strip(), "size": int(size), "by": uploaded_by},
        )
        return int(cur.lastrowid or 0)


def list_attachments(conn: sqlite3.Connection, task_id: str) -> list[Attachment]:
    return [Attachment.from_row(r) for r in _task_rows(conn, "task_attachments", task_id, "created_at ASC, id ASC")]


def get_attachment(conn: sqlite3.Connection, attachment_id: int) -> Optional[Attachment]:
    r = conn.execute(
        "SELECT * FROM task_attachments WHERE id = ?", (attachment_id,)
    ).fetchone()
    return None if r is None else Attachment.from_row(r)


def delete_attachment(conn: sqlite3.Connection, attachment_id: int) -> Optional[Attachment]:
    """Delete an attachment row and its on-disk blob. Returns the removed row.

    Returns ``None`` when no row matched. The blob is removed best-effort
    (a missing file is not an error); the metadata row is the source of
    truth for whether an attachment "exists".
    """
    with write_txn(conn):
        att = get_attachment(conn, attachment_id)
        if att is None:
            return None
        conn.execute("DELETE FROM task_attachments WHERE id = ?", (attachment_id,))
        _append_event(
            conn, att.task_id, "attachment_removed", {"filename": att.filename}
        )
    try:
        p = Path(att.stored_path)
        if p.is_file():
            p.unlink()
    except OSError:
        pass
    return att


def list_events(conn: sqlite3.Connection, task_id: str) -> list[Event]:
    return [Event.from_row(r) for r in _task_rows(conn, "task_events", task_id, "created_at ASC, id ASC")]


def _insert_comment(
    conn: sqlite3.Connection, task_id: str, author: str, body: str, created_at: int,
) -> None:
    """Raw ``task_comments`` INSERT for callers already inside a write txn.

    ``add_comment`` opens its own ``write_txn`` (raises on nesting) and emits
    a ``commented`` event; transitions that record their own event use this.
    """
    conn.execute(
        "INSERT INTO task_comments (task_id, author, body, created_at) "
        "VALUES (?, ?, ?, ?)",
        (task_id, author, body, created_at),
    )


def _append_event(
    conn: sqlite3.Connection,
    task_id: str,
    kind: str,
    payload: Optional[dict] = None,
    *,
    run_id: Optional[int] = None,
) -> None:
    """Record an event row.  Called from within an already-open txn.

    ``run_id`` is optional: pass the current run id so UIs can group
    events by attempt. For events that aren't scoped to a single run
    (task created/edited/archived, dependency promotion) leave it None
    and the row carries NULL.
    """
    now = int(time.time())
    pl = json.dumps(payload, ensure_ascii=False) if payload else None
    conn.execute(
        "INSERT INTO task_events (task_id, run_id, kind, payload, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (task_id, run_id, kind, pl, now),
    )


def _end_run(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    outcome: str,
    summary: Optional[str] = None,
    error: Optional[str] = None,
    metadata: Optional[dict] = None,
    status: Optional[str] = None,
) -> Optional[int]:
    """Close the currently-active run for ``task_id`` and clear the pointer.

    ``outcome`` is the semantic result (completed / blocked / crashed /
    timed_out / spawn_failed / gave_up / reclaimed). ``status`` is the
    run-row status (usually just ``outcome``, but callers can pass it
    explicitly). Returns the closed run_id or ``None`` if no active run
    existed (e.g. a CLI user calling ``hermes kanban complete`` on a
    task that was never claimed).
    """
    now = int(time.time())
    run_id = _current_run_id(conn, task_id)
    if run_id is None:
        return None
    conn.execute(
        """
        UPDATE task_runs
           SET status        = ?,
               outcome       = ?,
               summary       = ?,
               error         = ?,
               metadata      = ?,
               ended_at      = ?,
               claim_lock    = NULL,
               claim_expires = NULL,
               worker_pid    = NULL
         WHERE id = ?
           AND ended_at IS NULL
        """,
        (
            status or outcome,
            outcome,
            summary,
            error,
            json.dumps(metadata, ensure_ascii=False) if metadata else None,
            now,
            run_id,
        ),
    )
    conn.execute(
        "UPDATE tasks SET current_run_id = NULL WHERE id = ?", (task_id,),
    )
    return run_id


def _opt_int(value: Any) -> Optional[int]:
    """``int(value)`` or ``None`` when ``value`` is ``None`` (NULL column passthrough)."""
    return int(value) if value is not None else None


def _task_status(conn: sqlite3.Connection, task_id: str) -> Optional[str]:
    """Current ``tasks.status`` for ``task_id``, or ``None`` when no such row."""
    row = conn.execute("SELECT status FROM tasks WHERE id = ?", (task_id,)).fetchone()
    return row["status"] if row else None


def _current_run_id(conn: sqlite3.Connection, task_id: str) -> Optional[int]:
    row = conn.execute(
        "SELECT current_run_id FROM tasks WHERE id = ?", (task_id,),
    ).fetchone()
    return int(row["current_run_id"]) if row and row["current_run_id"] else None


def _synthesize_ended_run(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    outcome: str,
    summary: Optional[str] = None,
    error: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> int:
    """Insert a zero-duration, already-closed run row.

    Used when a terminal transition happens on a task that was never
    claimed (CLI user calling ``hermes kanban complete <ready-task>
    --summary X``, or dashboard "mark done" on a ready task). Without
    this, the handoff fields (summary / metadata / error) would be
    silently dropped: ``_end_run`` is a no-op because there's no
    current run.

    The synthetic run has ``started_at == ended_at == now`` so it
    shows up in attempt history as "instant" and doesn't skew elapsed
    stats. Caller is responsible for leaving ``current_run_id`` NULL
    (or for clearing it elsewhere in the same txn) since this
    function does NOT touch the tasks row.
    """
    now = int(time.time())
    trow = conn.execute(
        "SELECT assignee, current_step_key FROM tasks WHERE id = ?",
        (task_id,),
    ).fetchone()
    profile = trow["assignee"] if trow else None
    step_key = trow["current_step_key"] if trow else None
    cur = conn.execute(
        """
        INSERT INTO task_runs (
            task_id, profile, step_key,
            status, outcome,
            summary, error, metadata,
            started_at, ended_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            task_id, profile, step_key,
            outcome, outcome,
            summary, error,
            json.dumps(metadata, ensure_ascii=False) if metadata else None,
            now, now,
        ),
    )
    return int(cur.lastrowid or 0)


# ---------------------------------------------------------------------------
# Dependency resolution (todo -> ready)
# ---------------------------------------------------------------------------

def _has_sticky_block(conn: sqlite3.Connection, task_id: str) -> bool:
    """Return True when ``task_id`` is sticky-blocked by an explicit
    worker/operator ``kanban_block`` call.

    A ``blocked`` status has two sources: a deliberate worker/operator
    ``kanban_block`` handoff (emits a ``"blocked"`` event; must stay blocked
    until an operator unblocks it) and a circuit-breaker trip in
    ``_record_task_failure`` (emits ``"gave_up"``, NOT ``"blocked"``; meant to
    recover once conditions change). The cheapest discriminator is the most
    recent ``"blocked"``/``"unblocked"`` event: if it is ``"blocked"`` the task
    is sticky and ``recompute_ready`` must not auto-promote it. No such event
    at all (breaker trip, direct DB edit) returns ``False`` — the legacy
    auto-recover path.
    """
    row = conn.execute(
        "SELECT kind FROM task_events "
        "WHERE task_id = ? AND kind IN ('blocked', 'unblocked') "
        "ORDER BY id DESC LIMIT 1",
        (task_id,),
    ).fetchone()
    return bool(row) and row["kind"] == "blocked"


def _latest_event(
    conn: sqlite3.Connection, task_id: str, kind: str, run_id: Optional[int] = None,
) -> Optional[sqlite3.Row]:
    """Newest ``task_events`` row of ``kind`` (optionally scoped to one run)."""
    sql = "SELECT payload FROM task_events WHERE task_id = ? AND kind = ?"
    params: tuple[Any, ...] = (task_id, kind)
    if run_id is not None:
        sql += " AND run_id = ?"
        params = (*params, int(run_id))
    return conn.execute(sql + " ORDER BY id DESC LIMIT 1", params).fetchone()


def _resume_status_from_events(conn: sqlite3.Connection, task_id: str) -> str:
    """Return the durable phase a blocked/dependency-wait task should resume.

    Events written by review workers carry ``source_status``/``retry_status``;
    an explicit unblock that must wait for parents carries ``resume_status``.
    Legacy events omit these fields and therefore retain the historical
    ``ready`` behavior.
    """
    row = conn.execute(
        "SELECT payload FROM task_events "
        "WHERE task_id = ? AND kind IN ("
        "'blocked', 'block_loop_detected', 'dependency_wait', 'gave_up', "
        "'unblocked', 'changes_requested', 'review_reopened', 'status', 'reclaimed', "
        "'stale', 'timed_out', 'crashed', 'spawn_failed', 'rate_limited'"
        ") ORDER BY id DESC LIMIT 1",
        (task_id,),
    ).fetchone()
    payload = _json_dict(_row_get(row, "payload"))
    for key in ("resume_status", "retry_status", "source_status"):
        if payload.get(key) == "review":
            return "review"
    return "ready"


def recompute_ready(
    conn: sqlite3.Connection, failure_limit: int = None,
) -> int:
    """Promote ``todo`` tasks to ``ready`` when all parents are ``done`` or ``archived``.

    Returns the number of tasks promoted.  Opens its own IMMEDIATE txn, so it
    MUST be called OUTSIDE any open write transaction (plain ``write_txn``
    raises on nesting); call it after the enclosing txn commits.

    ``blocked`` tasks are also considered (a task blocked purely by a parent
    dependency unblocks itself when the parent completes), *except* when the
    most recent block event was a worker-initiated ``kanban_block`` (stays
    blocked until explicit ``kanban_unblock``) or ``consecutive_failures`` has
    reached the effective limit (otherwise the counter would reset on every
    recovery cycle and the breaker could never trip).

    The effective limit resolves in the same order as ``_record_task_failure``
    so the two never disagree: per-task ``max_retries``, then the caller's
    ``failure_limit`` (``kanban.failure_limit`` via ``dispatch_once``), then
    ``DEFAULT_FAILURE_LIMIT``.
    """
    if failure_limit is None:
        failure_limit = DEFAULT_FAILURE_LIMIT
    promoted = 0
    with write_txn(conn):
        todo_rows = conn.execute(
            "SELECT id, status, consecutive_failures, max_retries "
            "FROM tasks WHERE status IN ('todo', 'blocked')"
        ).fetchall()
        for row in todo_rows:
            task_id = row["id"]
            cur_status = row["status"]
            if cur_status == "blocked" and _has_sticky_block(conn, task_id):
                # Worker / operator asked for explicit human intervention — do not
                # silently auto-recover.  ``unblock_task`` is the only
                # legitimate exit (it emits ``"unblocked"`` which flips
                # this predicate back).
                continue
            parents = conn.execute(
                "SELECT t.status FROM tasks t "
                "JOIN task_links l ON l.parent_id = t.id "
                "WHERE l.child_id = ?",
                (task_id,),
            ).fetchall()
            if all(p["status"] in ("done", "archived") for p in parents):
                resume_status = _resume_status_from_events(conn, task_id)
                if cur_status == "blocked":
                    # Don't auto-recover tasks that have hit the
                    # circuit-breaker failure limit.  Without this
                    # guard, a task that repeatedly exhausts its
                    # iteration budget would cycle forever:
                    # block → auto-recover → respawn → budget
                    # exhausted → block → …  The counter must also
                    # be preserved so the breaker can accumulate
                    # across recovery cycles.
                    failures = int(row["consecutive_failures"] or 0)
                    task_limit = row["max_retries"]
                    effective_limit = (
                        int(task_limit) if task_limit is not None
                        else int(failure_limit)
                    )
                    if failures >= effective_limit:
                        continue
                    conn.execute(
                        "UPDATE tasks SET status = ? "
                        "WHERE id = ? AND status = 'blocked'",
                        (resume_status, task_id),
                    )
                else:
                    conn.execute(
                        "UPDATE tasks SET status = ? WHERE id = ? AND status = 'todo'",
                        (resume_status, task_id),
                    )
                _append_event(
                    conn, task_id, "promoted",
                    {"status": resume_status} if resume_status != "ready" else None,
                )
                promoted += 1
    return promoted


# ---------------------------------------------------------------------------
# Claim / complete / block
# ---------------------------------------------------------------------------

def _parents_satisfied(conn: sqlite3.Connection, task_id: str) -> bool:
    """Return whether every direct parent is terminal for dependency gating."""
    return conn.execute(
        "SELECT 1 FROM task_links l "
        "JOIN tasks p ON p.id = l.parent_id "
        "WHERE l.child_id = ? "
        "AND p.status NOT IN ('done', 'archived') LIMIT 1",
        (task_id,),
    ).fetchone() is None


def _claim_and_open_run(
    conn: sqlite3.Connection,
    task_id: str,
    source_status: str,
    lock: str,
    expires: int,
    now: int,
    *,
    event_extra: Optional[dict] = None,
) -> Optional[int]:
    """CAS ``source_status -> running``, open a ``task_runs`` row and emit ``claimed``.

    Caller holds the write transaction. Returns the new run id, or ``None``
    when the CAS lost (task already claimed / not in ``source_status``).
    """
    cur = conn.execute(
        f"""
        UPDATE tasks
           SET status        = 'running',
               claim_lock    = ?,
               claim_expires = ?,
               started_at    = COALESCE(started_at, ?)
         WHERE id = ?
           AND status = '{source_status}'
           AND claim_lock IS NULL
        """,
        (lock, expires, now, task_id),
    )
    if cur.rowcount != 1:
        return None
    # Populate the run with the task's assignee / step / runtime cap.
    trow = conn.execute(
        "SELECT assignee, max_runtime_seconds, current_step_key "
        "FROM tasks WHERE id = ?",
        (task_id,),
    ).fetchone()
    run_cur = conn.execute(
        """
        INSERT INTO task_runs (
            task_id, profile, step_key, status,
            claim_lock, claim_expires, max_runtime_seconds,
            started_at
        ) VALUES (?, ?, ?, 'running', ?, ?, ?, ?)
        """,
        (
            task_id,
            trow["assignee"] if trow else None,
            trow["current_step_key"] if trow else None,
            lock,
            expires,
            trow["max_runtime_seconds"] if trow else None,
            now,
        ),
    )
    run_id = run_cur.lastrowid
    conn.execute(
        "UPDATE tasks SET current_run_id = ? WHERE id = ?",
        (run_id, task_id),
    )
    _append_event(
        conn, task_id, "claimed",
        {"lock": lock, "expires": expires, "run_id": run_id, **(event_extra or {})},
        run_id=run_id,
    )
    return run_id


def claim_task(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    ttl_seconds: Optional[int] = None,
    claimer: Optional[str] = None,
) -> Optional[Task]:
    """Atomically transition ``ready -> running``.

    Returns the claimed ``Task`` on success, ``None`` if the task was
    already claimed (or is not in ``ready`` status).
    """
    now = int(time.time())
    lock = claimer or _claimer_id()
    expires = now + _resolve_claim_ttl_seconds(ttl_seconds)
    with write_txn(conn):
        # Structural invariant: never transition ready -> running while any
        # parent is not yet 'done'. This is the single enforcement point
        # regardless of which writer (create_task, link_tasks, unblock_task,
        # release_stale_claims, manual SQL) set status='ready'. If a racy
        # writer promoted a task with undone parents, demote it back to
        # 'todo' here — recompute_ready will re-promote when the parents
        # actually finish. See RCA at
        # kanban/boards/cookai/workspaces/t_a6acd07d/root-cause.md.
        if not _parents_satisfied(conn, task_id):
            conn.execute(
                "UPDATE tasks SET status = 'todo' "
                "WHERE id = ? AND status = 'ready'",
                (task_id,),
            )
            _append_event(
                conn, task_id, "claim_rejected",
                {"reason": "parents_not_done"},
            )
            return None
        # Defensive: close a leaked prior run as 'reclaimed' so the CAS below
        # doesn't strand it. No-op when the runs invariant holds.
        _reclaim_dangling_run(
            conn, task_id, statuses=("ready",), now=now,
            note="invariant recovery on re-claim",
        )
        run_id = _claim_and_open_run(conn, task_id, "ready", lock, expires, now)
        if run_id is None:
            return None
        claimed = get_task(conn, task_id)
    _fire_kanban_lifecycle_hook(
        "kanban_task_claimed",
        task_id,
        board=get_current_board(),
        assignee=claimed.assignee if claimed else None,
        run_id=run_id,
    )
    return claimed


def claim_review_task(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    ttl_seconds: Optional[int] = None,
    claimer: Optional[str] = None,
) -> Optional[Task]:
    """Atomically transition ``review -> running``.

    Returns the claimed ``Task`` on success, ``None`` if the task was
    already claimed (or is not in ``review`` status).

    Parent dependencies are re-checked because a previously completed parent
    may have been reopened while this task waited in review.

    Creates a new run entry so the review agent's lifecycle is tracked
    independently from the original worker run.
    """
    now = int(time.time())
    lock = claimer or _claimer_id()
    expires = now + _resolve_claim_ttl_seconds(ttl_seconds)
    with write_txn(conn):
        if not _parents_satisfied(conn, task_id):
            demoted = conn.execute(
                "UPDATE tasks SET status = 'todo' "
                "WHERE id = ? AND status = 'review' AND claim_lock IS NULL",
                (task_id,),
            )
            if demoted.rowcount == 1:
                _append_event(
                    conn,
                    task_id,
                    "dependency_wait",
                    {
                        "reason": "parent_reopened",
                        "source_status": "review",
                    },
                )
            return None
        run_id = _claim_and_open_run(
            conn, task_id, "review", lock, expires, now,
            event_extra={"source_status": "review"},
        )
        if run_id is None:
            return None
        return get_task(conn, task_id)


def _retry_status_for_run(
    conn: sqlite3.Connection,
    task_id: str,
    run_id: Optional[int] = None,
) -> str:
    """Return the non-running phase an interrupted run must resume from.

    Review claims record ``source_status=review`` on their claimed event. All
    other and legacy runs retry from ``ready``. Keeping this decision in one
    place prevents crash/timeout/reclaim paths from silently converting a
    reviewer run into an implementation run.
    """
    if run_id is None:
        run_id = _current_run_id(conn, task_id)
    if run_id is None:
        return "ready"
    event = _latest_event(conn, task_id, "claimed", run_id)
    payload = _json_dict(_row_get(event, "payload"))
    return "review" if payload.get("source_status") == "review" else "ready"


def goal_run_status(
    conn: sqlite3.Connection,
    task_id: str,
    expected_run_id: Optional[int] = None,
) -> Optional[str]:
    """Resolve lifecycle status from the perspective of one worker run.

    A successor may claim the task immediately after this run hands it off.
    Returning the task's live ``running`` status in that case lets the old goal
    loop mutate the successor.  Bind terminal handoffs to the original run and
    report any other ownership loss as ``superseded``.
    """
    task = get_task(conn, task_id)
    if task is None:
        return None
    if expected_run_id is not None:
        row = conn.execute(
            "SELECT outcome FROM task_runs WHERE id = ? AND task_id = ?",
            (int(expected_run_id), task_id),
        ).fetchone()
        outcome = (
            str(row["outcome"])
            if row and row["outcome"] is not None
            else None
        )
        terminal_status = (
            {
                "completed": "done",
                "review_requested": "review",
                "changes_requested": "changes_requested",
                "blocked": "blocked",
                "dependency_wait": "blocked",
            }.get(outcome)
            if outcome is not None
            else None
        )
        if terminal_status is not None:
            return terminal_status
        if outcome is not None or task.current_run_id != int(expected_run_id):
            return "superseded"
    if task.status in {"ready", "todo"}:
        event = conn.execute(
            "SELECT kind FROM task_events WHERE task_id = ? "
            "ORDER BY id DESC LIMIT 1",
            (task_id,),
        ).fetchone()
        if event and event["kind"] == "changes_requested":
            return "changes_requested"
    return task.status


def heartbeat_claim(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    ttl_seconds: Optional[int] = None,
    claimer: Optional[str] = None,
) -> bool:
    """Extend a running claim.  Returns True if we still own it.

    Workers that know they'll exceed 15 minutes should call this every
    few minutes to keep ownership.
    """
    expires = int(time.time()) + _resolve_claim_ttl_seconds(ttl_seconds)
    lock = claimer or _claimer_id()
    with write_txn(conn):
        cur = conn.execute(
            "UPDATE tasks SET claim_expires = ? "
            "WHERE id = ? AND status = 'running' AND claim_lock = ?",
            (expires, task_id, lock),
        )
        if cur.rowcount == 1:
            run_id = _current_run_id(conn, task_id)
            if run_id is not None:
                conn.execute(
                    "UPDATE task_runs SET claim_expires = ? WHERE id = ?",
                    (expires, run_id),
                )
            return True
        return False


def release_stale_claims(
    conn: sqlite3.Connection,
    *,
    signal_fn=None,
) -> int:
    """Reset any ``running`` task whose claim has expired.

    A stale-by-TTL claim whose host-local worker PID is still alive is
    *extended* (``claim_extended`` event) instead of reclaimed: reclaiming a
    live worker mid-flight causes a spawn-then-reclaim loop on slow models
    that spend longer than ``DEFAULT_CLAIM_TTL_SECONDS`` inside one tool-free
    LLM call (no tool calls means no ``kanban_heartbeat``).

    Backstop: a live PID whose ``last_heartbeat_at`` is older than
    ``DEFAULT_CLAIM_HEARTBEAT_MAX_STALE_SECONDS`` is reclaimed anyway — the
    wedged-in-a-logic-loop case. ``_touch_activity`` (run_agent.py) bridges
    chunk-level liveness into ``last_heartbeat_at``, so any genuinely active
    worker stays fresh via normal API traffic. ``enforce_max_runtime`` and
    ``detect_crashed_workers`` remain the upper bounds for wedged/dead workers.

    Returns the number of stale claims actually reclaimed (live-pid
    extensions don't count). Safe to call often.
    """
    now = int(time.time())
    reclaimed = 0
    host_prefix = _host_prefix()
    stale = conn.execute(
        "SELECT id, claim_lock, worker_pid, claim_expires, last_heartbeat_at, "
        "       assignee "
        "FROM tasks "
        "WHERE status = 'running' AND claim_expires IS NOT NULL "
        "  AND claim_expires < ?",
        (now,),
    ).fetchall()
    for row in stale:
        lock = row["claim_lock"] or ""
        host_local = lock.startswith(host_prefix)
        hb = row["last_heartbeat_at"]
        # Heartbeat staleness backstop: if we have a heartbeat at all
        # and it's older than the max-stale threshold, the worker is
        # not making observable progress.  Reclaim instead of extending,
        # even if the PID is still alive (it's likely in a logic loop).
        heartbeat_stale = (
            hb is not None
            and (now - int(hb)) > DEFAULT_CLAIM_HEARTBEAT_MAX_STALE_SECONDS
        )
        if (
            host_local
            and row["worker_pid"]
            and _pid_alive(row["worker_pid"])
            and not heartbeat_stale
        ):
            new_expires = now + _resolve_claim_ttl_seconds()
            with write_txn(conn):
                cur = conn.execute(
                    "UPDATE tasks SET claim_expires = ? "
                    "WHERE id = ? AND status = 'running' "
                    "  AND claim_lock IS ? "
                    "  AND claim_expires IS NOT NULL "
                    "  AND claim_expires < ?",
                    (new_expires, row["id"], row["claim_lock"], now),
                )
                if cur.rowcount != 1:
                    continue
                run_id = _current_run_id(conn, row["id"])
                if run_id is not None:
                    conn.execute(
                        "UPDATE task_runs SET claim_expires = ? WHERE id = ?",
                        (new_expires, run_id),
                    )
                _append_event(
                    conn, row["id"], "claim_extended",
                    {
                        "reason": "pid_alive",
                        "worker_pid": int(row["worker_pid"]),
                        "claim_lock": row["claim_lock"],
                        "claim_expires_was": int(row["claim_expires"]),
                        "claim_expires_now": new_expires,
                        "last_heartbeat_at": _opt_int(row["last_heartbeat_at"]),
                    },
                    run_id=run_id,
                )
            continue

        termination = _terminate_reclaimed_worker(
            row["worker_pid"], row["claim_lock"], signal_fn=signal_fn,
        )
        # Never release a claim while our own worker is still alive: that would
        # spawn a duplicate beside it. Hold the claim and retry next tick.
        if _worker_survived_termination(termination):
            _defer_reclaim_for_live_worker(
                conn, row["id"], row["claim_lock"], now, termination,
                reason="ttl_expired_worker_alive",
            )
            continue
        with write_txn(conn):
            retry_status = _retry_status_for_run(conn, row["id"])
            cur = conn.execute(
                "UPDATE tasks SET status = ?, claim_lock = NULL, "
                "claim_expires = NULL, worker_pid = NULL "
                "WHERE id = ? AND status = 'running' AND claim_lock IS ? "
                "AND claim_expires IS NOT NULL AND claim_expires < ?",
                (retry_status, row["id"], row["claim_lock"], now),
            )
            if cur.rowcount != 1:
                continue
            run_id = _end_run(
                conn, row["id"],
                outcome="reclaimed", status="reclaimed",
                error=f"stale_lock={row['claim_lock']}",
                metadata=termination,
            )
            payload = {
                "stale_lock": row["claim_lock"],
                "worker_pid": _opt_int(row["worker_pid"]),
                "claim_expires": int(row["claim_expires"]),
                "last_heartbeat_at": _opt_int(row["last_heartbeat_at"]),
                "now": now,
                "host_local": host_local,
                "heartbeat_stale": bool(heartbeat_stale),
                "retry_status": retry_status,
            }
            payload.update(termination)
            _append_event(
                conn, row["id"], "reclaimed",
                payload,
                run_id=run_id,
            )
            reclaimed += 1
        # Worker-lifecycle observer (RFC #58548): the reclaim txn above has
        # committed. The ``continue`` branches (rowcount mismatch, claim
        # extension, deferred reclaim) never reach this point, so only a
        # genuinely reclaimed stale claim fires.
        if _kanban_observer_consumed("on_kanban_worker_stale_claim"):
            _fire_kanban_lifecycle_hook(
                "on_kanban_worker_stale_claim",
                row["id"],
                board=get_current_board(),
                assignee=row["assignee"],
                run_id=run_id,
                worker_pid=_opt_int(row["worker_pid"]),
                heartbeat_stale=bool(heartbeat_stale),
                retry_status=retry_status,
            )
    return reclaimed


def reclaim_task(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    reason: Optional[str] = None,
    signal_fn=None,
) -> bool:
    """Operator-driven reclaim: release the claim and restore its source phase.

    Unlike :func:`release_stale_claims` which only acts on tasks whose
    ``claim_expires`` has passed, this function reclaims immediately
    regardless of TTL. Intended for the dashboard/CLI recovery flow
    when an operator wants to abort a running worker without waiting
    for the TTL to expire (e.g. after seeing a hallucination warning).

    Returns True if a reclaim happened, False if the task isn't in a
    reclaimable state (not running, or doesn't exist).
    """
    row = conn.execute(
        "SELECT status, claim_lock, worker_pid FROM tasks WHERE id = ?",
        (task_id,),
    ).fetchone()
    if not row:
        return False
    if row["status"] != "running" and row["claim_lock"] is None:
        # Nothing to reclaim — already ready / blocked / done.
        return False
    prev_lock = row["claim_lock"]
    termination = _terminate_reclaimed_worker(
        row["worker_pid"], prev_lock, signal_fn=signal_fn,
    )
    with write_txn(conn):
        retry_status = _retry_status_for_run(conn, task_id)
        cur = conn.execute(
            "UPDATE tasks SET status = ?, claim_lock = NULL, "
            "claim_expires = NULL, worker_pid = NULL "
            "WHERE id = ? AND status IN ('running', 'ready', 'blocked') "
            "AND claim_lock IS ?",
            (retry_status, task_id, prev_lock),
        )
        if cur.rowcount != 1:
            return False
        run_id = _end_run(
            conn, task_id,
            outcome="reclaimed", status="reclaimed",
            error=(
                f"manual_reclaim: {reason}" if reason
                else f"manual_reclaim lock={prev_lock}"
            ),
            metadata=termination,
        )
        payload = {
            "manual": True,
            "reason": reason,
            "prev_lock": prev_lock,
            "retry_status": retry_status,
        }
        payload.update(termination)
        _append_event(
            conn, task_id, "reclaimed",
            payload,
            run_id=run_id,
        )
    # Operator intervention — they've looked at the task, so the
    # consecutive-failures counter is now stale. Give the next retry
    # a fresh budget. (_clear_failure_counter opens its own write_txn,
    # so it runs after the enclosing one commits.)
    _clear_failure_counter(conn, task_id)
    return True


def reassign_task(
    conn: sqlite3.Connection,
    task_id: str,
    profile: Optional[str],
    *,
    reclaim_first: bool = False,
    reason: Optional[str] = None,
) -> bool:
    """Reassign a task, optionally reclaiming a stuck running worker first.

    This is the recovery path for "this profile's model is broken, try
    a different one". If ``reclaim_first`` is True, any active claim is
    released (via :func:`reclaim_task`) before the reassign happens;
    otherwise the function refuses to reassign a currently-running task
    and returns False (caller can retry with ``reclaim_first=True``).

    Returns True if the reassign landed. ``profile`` may be ``None`` to
    unassign entirely.
    """
    if reclaim_first:
        # Safe to call even if nothing to reclaim.
        reclaim_task(conn, task_id, reason=reason or "reassign")
    # assign_task handles its own txn + the still-running guard.
    try:
        return assign_task(conn, task_id, profile)
    except RuntimeError:
        # Task is still running and reclaim_first was False; caller
        # needs to decide whether to retry with reclaim.
        return False


def _verify_created_cards(
    conn: sqlite3.Connection,
    completing_task_id: str,
    claimed_ids: Iterable[str],
) -> tuple[list[str], list[str]]:
    """Partition ``claimed_ids`` into (verified, phantom).

    A card is "verified" iff a row exists in ``tasks`` AND at least one
    of the following holds:

    * ``created_by`` matches the completing task's ``assignee`` profile
      (the common case: worker A spawns a card via ``kanban_create``,
      which stamps ``created_by=A``).
    * ``created_by`` matches the completing task's id (edge case where
      a worker passed its own task id as the ``created_by`` value).
    * The card is linked as a ``task_links.child`` of the completing
      task — i.e. the worker explicitly called ``kanban_create`` with
      ``parents=[<current_task>]``. This accepts cards created through
      the dashboard/CLI by a different principal but then attached to
      the completing task by the worker.

    ``phantom`` returns ids that either don't exist at all, or exist
    but don't satisfy any of the three trust conditions. The caller
    decides what to do with each bucket; this helper never mutates.
    """
    claimed = [str(x).strip() for x in (claimed_ids or []) if str(x).strip()]
    if not claimed:
        return [], []
    # Dedupe while preserving order.
    seen: set[str] = set()
    ordered: list[str] = []
    for cid in claimed:
        if cid not in seen:
            seen.add(cid)
            ordered.append(cid)

    row = conn.execute(
        "SELECT assignee FROM tasks WHERE id = ?", (completing_task_id,),
    ).fetchone()
    if row is None:
        # Completing task not found — nothing resolves.
        return [], ordered
    completing_assignee = row["assignee"]

    # Batch-fetch existence + created_by in one query.
    placeholders = ",".join(["?"] * len(ordered))
    rows = conn.execute(
        f"SELECT id, created_by FROM tasks WHERE id IN ({placeholders})",
        tuple(ordered),
    ).fetchall()
    found = {r["id"]: r["created_by"] for r in rows}

    # Pull the set of cards linked as children of the completing task.
    # Cheap: one query, indexed on parent_id.
    linked_children: set[str] = set(child_ids(conn, completing_task_id))

    verified: list[str] = []
    phantom: list[str] = []
    for cid in ordered:
        created_by = found.get(cid)
        if created_by is None:
            phantom.append(cid)
            continue
        # Accept if any of the three trust conditions holds.
        if (
            (completing_assignee and created_by == completing_assignee)
            or created_by == completing_task_id
            or cid in linked_children
        ):
            verified.append(cid)
        else:
            phantom.append(cid)
    return verified, phantom


# Task-id pattern used both by ``kanban_create`` (``t_<12 hex>``) and
# ``_new_task_id`` below. Kept permissive on length for forward compat:
# accept 8+ hex chars after the ``t_`` prefix.
_TASK_ID_PROSE_RE = re.compile(r"\bt_[a-f0-9]{8,}\b")


def _scan_prose_for_phantom_ids(
    conn: sqlite3.Connection,
    text: str,
) -> list[str]:
    """Regex-scan free-form text for ``t_<hex>`` references; return the
    ones that don't exist in ``tasks``.

    Used as a non-blocking advisory check on completion summaries. An
    empty return means "no suspicious references found" — either the
    text had no IDs at all, or every ID it mentioned resolves to a real
    task. Duplicates are deduped.
    """
    if not text:
        return []
    matches = _TASK_ID_PROSE_RE.findall(text)
    if not matches:
        return []
    # Dedupe preserving order.
    seen: set[str] = set()
    unique: list[str] = []
    for m in matches:
        if m not in seen:
            seen.add(m)
            unique.append(m)
    placeholders = ",".join(["?"] * len(unique))
    rows = conn.execute(
        f"SELECT id FROM tasks WHERE id IN ({placeholders})",
        tuple(unique),
    ).fetchall()
    existing = {r["id"] for r in rows}
    return [m for m in unique if m not in existing]


class HallucinatedCardsError(ValueError):
    """Raised by ``complete_task`` when ``created_cards`` contains ids
    that don't exist or weren't created by the completing worker.

    The phantom list is attached as ``.phantom`` for callers that want
    structured access. Kept as ``ValueError`` subclass so existing
    tool-error handlers treat it as a recoverable user error.
    """

    def __init__(self, phantom: list[str], completing_task_id: str):
        self.phantom = list(phantom)
        self.completing_task_id = completing_task_id
        super().__init__(
            f"completion blocked: claimed created_cards that do not exist "
            f"or were not created by this worker: {', '.join(phantom)}"
        )


class ArtifactPreservationError(RuntimeError):
    """Raised when a declared scratch deliverable cannot be preserved."""


def complete_task(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    result: Optional[str] = None,
    summary: Optional[str] = None,
    metadata: Optional[dict] = None,
    created_cards: Optional[Iterable[str]] = None,
    expected_run_id: Optional[int] = None,
    fire_lifecycle_hook: bool = True,
) -> bool:
    """Transition ``running|ready|blocked|review -> done`` and record ``result``.

    Accepts a task that is merely ``ready`` too, so a manual CLI
    completion (``hermes kanban complete <id>``) works without requiring
    a claim/start/complete sequence. ``review`` is accepted so a human
    (or reviewer) can approve a task parked in the review lane by
    :func:`request_review` — even when it has no active run
    (``current_run_id IS NULL``), the handoff fields are preserved via
    :func:`_synthesize_ended_run`.

    ``summary`` and ``metadata`` are stored on the closing run (if any)
    and surfaced to downstream children via :func:`build_worker_context`.
    When ``summary`` is omitted we fall back to ``result`` so single-run
    callers do not have to pass both. ``metadata`` is a free-form dict
    (e.g. ``{"changed_files": [...], "tests_run": [...]}``) — workers
    are encouraged to use it for structured handoff facts.

    ``created_cards`` is an optional list of task ids the completing
    worker claims to have created. Each id is verified against
    ``tasks.created_by``. If any id is phantom (does not exist or was
    not created by this worker's assignee profile), completion is blocked
    with a ``HallucinatedCardsError`` and a
    ``completion_blocked_hallucination`` event is emitted so the rejected
    attempt is auditable. When all ids verify, they are recorded on the
    ``completed`` event payload.

    After a successful completion, ``summary`` and ``result`` are scanned
    for prose references like ``t_deadbeefcafe`` that do not resolve.
    Any suspected phantom references are recorded as a
    ``suspected_hallucinated_references`` event. This pass is advisory
    and never blocks.
    """
    now = int(time.time())
    # Fail before validating cards or staging artifacts; re-check inside the
    # final write transaction below to close the parent-reopen race.
    if not _parents_satisfied(conn, task_id):
        return False

    # Gate: verify created_cards BEFORE the main write txn. A rejected
    # completion still needs an auditable event, so we emit it in a
    # tiny dedicated txn, then raise. The caller is responsible for
    # surfacing HallucinatedCardsError to the worker; this function
    # never mutates task state on a phantom-card rejection.
    if created_cards:
        verified_cards, phantom_cards = _verify_created_cards(
            conn, task_id, created_cards
        )
        if phantom_cards:
            with write_txn(conn):
                _append_event(
                    conn, task_id, "completion_blocked_hallucination",
                    {
                        "phantom_cards": phantom_cards,
                        "verified_cards": verified_cards,
                        "summary_preview": (
                            (summary or result or "").strip().splitlines()[0][:200]
                            if (summary or result)
                            else None
                        ),
                    },
                )
            raise HallucinatedCardsError(phantom_cards, task_id)
    else:
        verified_cards = []

    metadata = _merge_completion_prose_artifacts(
        conn, task_id, metadata, summary=summary, result=result,
    )
    with write_txn(conn):
        # Parent completion is a hard invariant even for direct human review
        # approval. A parent may have been reopened after this task entered
        # ``review`` or ``running``.
        if not _parents_satisfied(conn, task_id):
            return False
        prior_status = _task_status(conn, task_id)
        sql = """
                UPDATE tasks
                   SET status       = 'done',
                       result       = ?,
                       completed_at = ?,
                       claim_lock   = NULL,
                       claim_expires= NULL,
                       worker_pid   = NULL,
                       block_kind   = NULL,
                       block_recurrences = 0
                 WHERE id = ?
                   AND status IN ('running', 'ready', 'blocked', 'review')
                """
        params: tuple = (result, now, task_id)
        if expected_run_id is not None:
            sql += " AND current_run_id = ?"
            params = (*params, int(expected_run_id))
        cur = conn.execute(sql, params)
        if cur.rowcount != 1:
            return False
        if isinstance(metadata, dict):
            _persist_scratch_completion_artifacts(conn, task_id, metadata)
            for stored_path in metadata.pop("_staged_artifacts", []):
                path = Path(stored_path)
                _insert_completion_attachment(
                    conn,
                    task_id,
                    filename=path.name,
                    stored_path=str(path),
                    size=path.stat().st_size,
                    created_at=now,
                )
        run_id = _end_run(
            conn, task_id,
            outcome="completed", status="done",
            summary=summary if summary is not None else result,
            metadata=metadata,
        )
        # If complete_task was called on a never-claimed task (ready or
        # blocked → done with no run in flight), synthesize a
        # zero-duration run so the handoff fields are persisted in
        # attempt history instead of silently lost.
        if run_id is None and (
            summary or metadata or result or prior_status == "review"
        ):
            synth_summary = summary if summary is not None else result
            synth_metadata = metadata
            if prior_status == "review" and not synth_summary and not synth_metadata:
                synth_summary = "Review approved without additional evidence."
                synth_metadata = {
                    "source_status": "review",
                    "approval": "manual",
                }
            run_id = _synthesize_ended_run(
                conn, task_id,
                outcome="completed",
                summary=synth_summary,
                metadata=synth_metadata,
            )
        # Carry the handoff summary in the event payload so gateway
        # notifiers and dashboard WS consumers can render it without a
        # second SQL round-trip. First line only, 400 char cap — the
        # full summary stays on the run row.
        event_summary = summary if summary is not None else result
        if prior_status == "review" and not event_summary:
            event_summary = "Review approved without additional evidence."
        _ev_lines = (event_summary or "").strip().splitlines()
        ev_summary = _ev_lines[0][:400] if _ev_lines else ""
        completed_payload: dict = {
            "result_len": len(result) if result else 0,
            "summary": ev_summary or None,
        }
        if verified_cards:
            completed_payload["verified_cards"] = verified_cards
        # Carry artifact paths in the event payload so the gateway
        # notifier can upload them as native attachments alongside the
        # completion message. Workers pass these via
        # ``kanban_complete(artifacts=[...])`` which stashes the list in
        # ``metadata["artifacts"]`` — we promote it onto the event so
        # consumers don't have to fetch the run row to find it.
        if isinstance(metadata, dict):
            md_artifacts = metadata.get("artifacts")
            if isinstance(md_artifacts, (list, tuple)):
                cleaned_artifacts = [
                    str(p).strip() for p in md_artifacts if isinstance(p, str) and str(p).strip()
                ]
                if cleaned_artifacts:
                    completed_payload["artifacts"] = cleaned_artifacts
        _append_event(
            conn, task_id, "completed",
            completed_payload,
            run_id=run_id,
        )
    # Prose-scan the summary + result for t_<hex> references that do
    # not resolve. Advisory — does not block the completion. Runs in
    # its own txn so the completion itself is already durable by the
    # time we emit the warning.
    scan_text = " ".join(filter(None, [summary, result]))
    if scan_text:
        phantom_refs = _scan_prose_for_phantom_ids(conn, scan_text)
        # Drop any phantom refs that were already flagged as verified
        # above (shouldn't happen — verified means they exist — but
        # belt-and-suspenders).
        phantom_refs = [p for p in phantom_refs if p not in set(verified_cards)]
        if phantom_refs:
            with write_txn(conn):
                _append_event(
                    conn, task_id, "suspected_hallucinated_references",
                    {
                        "phantom_refs": phantom_refs,
                        "source": "completion_summary",
                    },
                    run_id=run_id,
                )
    # Successful completion — wipe the consecutive-failures counter.
    # Failure history stays on the event log for audit; the counter
    # just tracks "is there a current pathology the breaker should
    # care about", and a success resets that question.
    _clear_failure_counter(conn, task_id)
    # Recompute ready status for dependents (separate txn so children see done).
    recompute_ready(conn)
    # Clean up the scratch workspace and any stale tmux session for the worker.
    _cleanup_workspace(conn, task_id)
    _done_task = get_task(conn, task_id)
    if fire_lifecycle_hook:
        _fire_kanban_lifecycle_hook(
            "kanban_task_completed",
            task_id,
            board=get_current_board(),
            assignee=_done_task.assignee if _done_task else None,
            run_id=run_id,
            summary=(summary if summary is not None else result),
        )
    return True


# ---------------------------------------------------------------------------
# Workspace / tmux cleanup
# ---------------------------------------------------------------------------


def _merge_completion_prose_artifacts(
    conn: sqlite3.Connection,
    task_id: str,
    metadata: Optional[dict],
    *,
    summary: Optional[str],
    result: Optional[str],
) -> Optional[dict]:
    """Promote existing scratch files named in legacy completion prose.

    ``artifacts=[...]`` is preferred. Older workers only wrote an absolute
    deliverable path in ``summary``/``result``; discover it while scratch still
    exists so cleanup cannot erase the file the user was promised.
    """
    workspace = _scratch_workspace(conn, task_id)
    if workspace is None:
        return metadata
    if not _is_managed_scratch_path(workspace):
        return metadata
    text = "\n".join(part for part in (summary, result) if part)
    if not text:
        return metadata
    prefix = re.escape(str(workspace))
    discovered: list[str] = []
    for match in re.finditer(prefix + r"(?:[/\\][^\s`\"'<>]+)", text):
        raw = match.group(0).rstrip(".,;:!?)]}")
        candidate = Path(raw)
        if candidate.is_file():
            discovered.append(str(candidate))
    if not discovered:
        return metadata
    updated = dict(metadata) if isinstance(metadata, dict) else {}
    existing = updated.get("artifacts")
    merged = list(existing) if isinstance(existing, (list, tuple)) else []
    seen = {str(path) for path in merged}
    for path in discovered:
        if path not in seen:
            merged.append(path)
            seen.add(path)
    updated["artifacts"] = merged
    return updated


def _persist_scratch_completion_artifacts(
    conn: sqlite3.Connection,
    task_id: str,
    metadata: dict,
) -> None:
    """Copy scratch-workspace completion artifacts before cleanup removes them."""
    raw_artifacts = metadata.get("artifacts")
    if not isinstance(raw_artifacts, (list, tuple)):
        return

    workspace = _scratch_workspace(conn, task_id)
    if workspace is None:
        return
    is_managed, board = _managed_scratch_path_info(workspace)
    if not is_managed:
        return

    try:
        workspace_root = workspace.resolve()
    except OSError:
        return

    attachment_dir = task_attachments_dir(task_id, board=board)
    persisted: list[str] = []
    used_destinations: set[Path] = set()
    changed = False

    def _discard_copies() -> None:
        for copied in used_destinations:
            with contextlib.suppress(OSError):
                copied.unlink(missing_ok=True)
        with contextlib.suppress(OSError):
            attachment_dir.rmdir()

    for item in raw_artifacts:
        artifact = str(item).strip() if isinstance(item, str) else ""
        if not artifact:
            continue
        src = Path(artifact).expanduser()
        try:
            resolved_src = src.resolve()
        except OSError:
            persisted.append(artifact)
            continue

        if not resolved_src.is_relative_to(workspace_root):
            persisted.append(artifact)
            continue

        if not src.is_file():
            _discard_copies()
            raise ArtifactPreservationError(
                f"declared scratch artifact is unavailable or not a regular file: {artifact}"
            )

        size = resolved_src.stat().st_size
        if size > KANBAN_ATTACHMENT_MAX_BYTES:
            _discard_copies()
            raise ArtifactPreservationError(
                f"declared scratch artifact exceeds the "
                f"{KANBAN_ATTACHMENT_MAX_BYTES}-byte limit: {artifact}"
            )

        dest: Optional[Path] = None
        try:
            attachment_dir.mkdir(parents=True, exist_ok=True)
            dest = _unique_attachment_path(attachment_dir, resolved_src.name, used_destinations)
            with resolved_src.open("rb") as source_file, dest.open("xb") as destination_file:
                copied = 0
                while chunk := source_file.read(1024 * 1024):
                    copied += len(chunk)
                    if copied > KANBAN_ATTACHMENT_MAX_BYTES:
                        raise ArtifactPreservationError(
                            f"declared scratch artifact grew beyond the size limit: {artifact}"
                        )
                    destination_file.write(chunk)
        except Exception as exc:
            if dest is not None:
                with contextlib.suppress(OSError):
                    dest.unlink(missing_ok=True)
            _discard_copies()
            if isinstance(exc, ArtifactPreservationError):
                raise
            raise ArtifactPreservationError(
                f"could not preserve declared scratch artifact {artifact}: {exc}"
            ) from exc

        used_destinations.add(dest)
        persisted.append(str(dest.resolve()))
        changed = True

    if changed:
        metadata["artifacts"] = persisted
        metadata["_staged_artifacts"] = [
            path for path in persisted if path.startswith(str(attachment_dir.resolve()))
        ]


def _insert_completion_attachment(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    filename: str,
    stored_path: str,
    size: int,
    created_at: int,
) -> None:
    """Record a worker-produced artifact in the existing attachment table."""
    conn.execute(
        "INSERT INTO task_attachments "
        "(task_id, filename, stored_path, content_type, size, uploaded_by, created_at) "
        "VALUES (?, ?, ?, NULL, ?, 'kanban_complete', ?)",
        (task_id, filename, stored_path, size, created_at),
    )
    _append_event(
        conn,
        task_id,
        "attached",
        {"filename": filename, "size": size, "by": "kanban_complete"},
    )


def _unique_attachment_path(directory: Path, filename: str, used: set[Path]) -> Path:
    """Return a non-conflicting path under ``directory`` for ``filename``."""
    safe_name = Path(filename).name or "artifact"
    candidate = directory / safe_name
    if candidate not in used and not candidate.exists():
        return candidate

    stem = Path(safe_name).stem or "artifact"
    suffix = Path(safe_name).suffix
    idx = 1
    while True:
        candidate = directory / f"{stem}_{idx}{suffix}"
        if candidate not in used and not candidate.exists():
            return candidate
        idx += 1


# ---------------------------------------------------------------------------
# First-use tip for scratch workspaces
# ---------------------------------------------------------------------------
#
# Scratch workspaces are intentionally ephemeral (``_cleanup_workspace`` removes
# them on ``complete_task``); new users lose worker output without warning. On
# the FIRST scratch materialization per install: log a warning, append a
# ``tip_scratch_workspace`` event on the task, and touch a sentinel under
# ``kanban_home()`` so the tip never repeats. Per-install, not per-board.


def edit_completed_task_result(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    result: str,
    summary: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> bool:
    """Backfill the user-visible result for an already completed task."""
    handoff_summary = summary if summary is not None else result
    with write_txn(conn):
        if _task_status(conn, task_id) != "done":
            return False
        conn.execute(
            "UPDATE tasks SET result = ? WHERE id = ?",
            (result, task_id),
        )
        run = conn.execute(
            """
            SELECT id FROM task_runs
             WHERE task_id = ?
               AND outcome = 'completed'
             ORDER BY COALESCE(ended_at, started_at, 0) DESC, id DESC
             LIMIT 1
            """,
            (task_id,),
        ).fetchone()
        run_id = int(run["id"]) if run else None
        if run_id is None:
            run_id = _synthesize_ended_run(
                conn, task_id,
                outcome="completed",
                summary=handoff_summary,
                metadata=metadata,
            )
        else:
            conn.execute(
                "UPDATE task_runs SET summary = ? WHERE id = ?",
                (handoff_summary, run_id),
            )
            if metadata is not None:
                conn.execute(
                    "UPDATE task_runs SET metadata = ? WHERE id = ?",
                    (json.dumps(metadata, ensure_ascii=False), run_id),
                )
        _ev_lines = (handoff_summary or "").strip().splitlines()
        ev_summary = _ev_lines[0][:400] if _ev_lines else ""
        _append_event(
            conn, task_id, "edited",
            {
                "fields": (
                    ["result", "summary"]
                    + (["metadata"] if metadata is not None else [])
                ),
                "result_len": len(result) if result else 0,
                "summary": ev_summary or None,
            },
            run_id=run_id,
        )
    return True


def block_task(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    reason: Optional[str] = None,
    kind: Optional[str] = None,
    expected_run_id: Optional[int] = None,
) -> bool:
    """Transition ``running``/``ready`` → ``blocked`` (or route elsewhere).

    ``kind`` (one of :data:`VALID_BLOCK_KINDS`, or ``None`` for a legacy
    un-typed block) drives routing instead of every block landing in one
    undifferentiated ``blocked`` bucket:

    * ``dependency`` — the task is only waiting on another task. It does NOT
      sit in ``blocked`` (where a cron would keep "unblocking" it); it goes to
      ``todo`` so the existing parent-gating / ``recompute_ready`` machinery
      promotes it automatically once its parents finish. No human, no cron, no
      retry storm. This is Dale's "Type 2 — dependency blocked".

    * ``needs_input`` / ``capability`` / ``None`` — "truly blocked" (Dale's
      "Type 1"). Lands in ``blocked`` for a human. BUT: each time such a task
      is re-blocked for the SAME kind after having been unblocked, the
      unblock-loop counter (``block_recurrences``) increments. When it reaches
      :data:`BLOCK_RECURRENCE_LIMIT`, the task is routed to ``triage`` instead
      of ``blocked`` — breaking the cron-unblock ↔ worker-re-block loop and
      forcing a human-in-the-loop triage decision.

    * ``transient`` — treated like a generic block for routing, but a worker
      can use it to signal "this might clear on its own"; it still participates
      in the loop breaker so a forever-flaky task eventually escalates.

    Returns True on any successful transition (to ``blocked``, ``todo``, or
    ``triage``), False when the task wasn't in a blockable state.
    """
    if kind is not None and kind not in VALID_BLOCK_KINDS:
        raise ValueError(
            f"block kind must be one of {sorted(VALID_BLOCK_KINDS)} or None"
        )
    with write_txn(conn):
        cur_row = conn.execute(
            "SELECT status, block_kind, block_recurrences FROM tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
        if cur_row is None:
            return False
        source_status = (
            _retry_status_for_run(conn, task_id)
            if cur_row["status"] == "running"
            else "ready"
        )
        prev_kind = _row_get(cur_row, "block_kind")
        prev_recurrences = _row_get(cur_row, "block_recurrences")
        prev_recurrences = int(prev_recurrences) if prev_recurrences is not None else 0

        # Dependency blocks never enter the human ``blocked`` bucket — they
        # wait in ``todo`` and let ``recompute_ready`` gate on parents. Routing
        # here (rather than ``blocked``) is what keeps a cron from ever seeing
        # a dependency-wait as something to "unblock".
        if kind == "dependency":
            new_status, event_kind = "todo", "dependency_wait"
            set_sql, params = "block_kind    = ?", (kind,)
            payload = {"reason": reason, "kind": kind, "source_status": source_status}
        else:
            # Truly-blocked kinds. Increment the unblock-loop counter when this
            # is a re-block for the SAME reason after a prior unblock: block_task
            # only fires from running/ready (i.e. AFTER an unblock returned the
            # task to the work pool), so a stored block_kind matching the
            # incoming kind means blocked → unblocked → re-block, same cause.
            # An un-typed (None) block compares as "same" to a prior un-typed one.
            recurrences = prev_recurrences + 1 if prev_kind == kind else 1
            set_sql = "block_kind    = ?,\n                       block_recurrences = ?"
            params = (kind, recurrences)
            payload = {
                "reason": reason,
                "kind": kind,
                "recurrences": recurrences,
                "source_status": source_status,
            }
            if recurrences >= BLOCK_RECURRENCE_LIMIT:
                # Loop detected — route to triage for a human-in-the-loop
                # decision instead of letting the unblocker spin this task.
                new_status, event_kind = "triage", "block_loop_detected"
                payload["limit"] = BLOCK_RECURRENCE_LIMIT
            else:
                new_status, event_kind = "blocked", "blocked"
        sql = f"""
                UPDATE tasks
                   SET status        = '{new_status}',
                       claim_lock    = NULL,
                       claim_expires = NULL,
                       worker_pid    = NULL,
                       {set_sql}
                 WHERE id = ?
                   AND status IN ('running', 'ready')
                """
        params = (*params, task_id)
        if expected_run_id is not None:
            sql += " AND current_run_id = ?"
            params = (*params, int(expected_run_id))
        cur = conn.execute(sql, params)
        if cur.rowcount != 1:
            return False
        run_id = _end_run(
            conn, task_id,
            outcome="blocked", status="blocked",
            summary=reason,
        )
        # Synthesize a run when blocking a never-claimed task so the reason
        # is preserved in attempt history.
        if run_id is None and reason:
            run_id = _synthesize_ended_run(
                conn, task_id, outcome="blocked", summary=reason,
            )
        _append_event(conn, task_id, event_kind, payload, run_id=run_id)
        _blocked_task = get_task(conn, task_id)

        def _fire_blocked_hook() -> None:
            _fire_kanban_lifecycle_hook(
                "kanban_task_blocked",
                task_id,
                board=get_current_board(),
                assignee=_blocked_task.assignee if _blocked_task else None,
                run_id=run_id,
                reason=reason,
            )

        if kind == "dependency":
            # Historical ordering: the dependency lane fires inside the txn.
            _fire_blocked_hook()
            return True
    _fire_blocked_hook()
    return True


def redact_review_value(value: Any) -> Any:
    """Redact secrets at the domain boundary for durable review handoffs."""
    if isinstance(value, str):
        from agent.redact import redact_sensitive_text

        return redact_sensitive_text(value, force=True)
    if isinstance(value, dict):
        return {key: redact_review_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [redact_review_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_review_value(item) for item in value)
    return value


def request_review(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    summary: Optional[str] = None,
    metadata: Optional[dict] = None,
    reviewer: Optional[str] = None,
    expected_run_id: Optional[int] = None,
    force: bool = False,
    with_reason: bool = False,
):
    """Transition implementation work into the first-class review phase.

    Unlike :func:`block_task`, this transition never touches block recurrence
    accounting.  The current implementer and resolved reviewer are recorded on
    the event so an autonomous reviewer can route requested changes back to the
    right profile.  Supplying ``reviewer`` reassigns the task before it is
    exposed to the review dispatcher.  On re-review, omitting it reuses the
    reviewer provenance persisted by the latest ``changes_requested`` event.

    When the task is ``running`` under a live claim, a caller that supplies no
    ``expected_run_id`` must pass ``force=True`` (explicit human/CLI override)
    — otherwise the request is refused instead of silently clearing the live
    worker's ``claim_lock``/``worker_pid``. Workers prove ownership by passing
    their own run id as ``expected_run_id`` (unchanged).

    Returns ``bool`` by default. With ``with_reason=True`` returns
    ``(ok, reason)`` mirroring :func:`request_changes` — ``reason`` is a
    diagnostic string on failure, ``None`` on success.
    """

    def _ret(ok: bool, reason: Optional[str] = None):
        return (ok, reason) if with_reason else ok

    summary = redact_review_value(summary)
    metadata = redact_review_value(metadata)
    with write_txn(conn):
        if not _parents_satisfied(conn, task_id):
            return _ret(False, "parent dependencies are not satisfied")
        trow = conn.execute(
            "SELECT assignee, status, claim_lock, current_run_id "
            "FROM tasks WHERE id = ?", (task_id,),
        ).fetchone()
        if trow is None:
            return _ret(False, "task not found")
        # Refuse to clear a live worker's claim without proof of ownership
        # (expected_run_id) or an explicit human override (force=True).
        if (
            expected_run_id is None
            and not force
            and trow["status"] == "running"
            and trow["claim_lock"] is not None
        ):
            return _ret(
                False,
                "task is running under a live claim; pass expected_run_id "
                "(worker ownership) or force=True (explicit operator "
                "override) instead of clearing the live run's claim",
            )
        implementer = trow["assignee"]
        if reviewer is None:
            changes_run = conn.execute(
                "SELECT id FROM task_runs "
                "WHERE task_id = ? AND outcome = 'changes_requested' "
                "ORDER BY id DESC LIMIT 1",
                (task_id,),
            ).fetchone()
            changes_event = None
            if changes_run is not None:
                changes_event = _latest_event(
                    conn, task_id, "changes_requested", changes_run["id"],
                )
            prior_reviewer = _json_dict(_row_get(changes_event, "payload")).get("reviewer")
            if changes_run is not None:
                if not isinstance(prior_reviewer, str) or not prior_reviewer.strip():
                    return _ret(
                        False,
                        "re-review has no durable reviewer provenance (the "
                        "latest changes_requested event is missing or "
                        "malformed); pass reviewer= explicitly",
                    )
                reviewer = prior_reviewer
        reviewer = _canonical_assignee(reviewer) if reviewer is not None else None
        assignee_sql = ", assignee = ?" if reviewer is not None else ""
        params: tuple[Any, ...]
        if expected_run_id is None:
            params = (reviewer, task_id) if reviewer is not None else (task_id,)
            run_guard = ""
        else:
            params = (
                (reviewer, task_id, int(expected_run_id))
                if reviewer is not None
                else (task_id, int(expected_run_id))
            )
            run_guard = " AND current_run_id = ?"
        cur = conn.execute(
            """
            UPDATE tasks
               SET status        = 'review',
                   claim_lock    = NULL,
                   claim_expires = NULL,
                   worker_pid    = NULL
            """ + assignee_sql + """
             WHERE id = ?
               AND status IN ('running', 'ready')
            """ + run_guard,
            params,
        )
        if cur.rowcount != 1:
            return _ret(
                False,
                "task is not in running/ready (or expected_run_id did not "
                "match the current run)",
            )
        run_id = _end_run(
            conn,
            task_id,
            outcome="review_requested",
            status="review",
            summary=summary,
            metadata=metadata,
        )
        if run_id is None and (summary or metadata):
            run_id = _synthesize_ended_run(
                conn,
                task_id,
                outcome="review_requested",
                summary=summary,
                metadata=metadata,
            )
        lines = (summary or "").strip().splitlines()
        event_summary = lines[0][:400] if lines else ""
        _append_event(
            conn,
            task_id,
            "review_requested",
            {
                "summary": event_summary or None,
                "implementer": implementer,
                "reviewer": reviewer,
            },
            run_id=run_id,
        )
    return _ret(True)


def request_changes(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    reason: str,
    expected_run_id: Optional[int] = None,
) -> tuple[bool, Optional[str]]:
    """Finish an active review run and route the task back for rework.

    The transition is valid only for a run claimed from ``review``.  It closes
    that reviewer run, restores the implementer recorded by the latest
    ``review_requested`` event, reapplies parent gating, and emits an auditable
    ``changes_requested`` event.  The second tuple item is the implementer on
    success or a diagnostic reason on failure.
    """
    reason = str(redact_review_value(reason or "")).strip()
    if not reason:
        return False, "reason is required"

    with write_txn(conn):
        task_row = conn.execute(
            "SELECT status, assignee, current_run_id FROM tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
        if task_row is None:
            return False, "task not found"
        current_run_id = task_row["current_run_id"]
        if task_row["status"] != "running" or current_run_id is None:
            return False, "task is not in an active review run"
        if expected_run_id is not None and int(current_run_id) != int(expected_run_id):
            return False, "run_id mismatch"

        claimed_event = _latest_event(conn, task_id, "claimed", current_run_id)
        claimed_payload = _json_dict(_row_get(claimed_event, "payload"))
        if claimed_payload.get("source_status") != "review":
            return False, "active run was not claimed from review"

        requested_event = _latest_event(conn, task_id, "review_requested")
        if requested_event is None:
            return False, "no prior review_requested event"
        implementer = _json_dict(requested_event["payload"]).get("implementer")
        if not isinstance(implementer, str) or not implementer.strip():
            return False, "review handoff has no valid implementer provenance"
        reviewer = task_row["assignee"]
        if isinstance(reviewer, str) and reviewer.strip():
            reviewer = _canonical_assignee(reviewer)
        else:
            reviewer = None

        new_status = _landing_status_after_parents(conn, task_id)
        # NOTE: consecutive_failures is deliberately PRESERVED (neither
        # reset nor incremented). Review transitions are not evidence the
        # pathology cleared — only complete_task's success path resets the
        # breaker counter (mirrors unblock_task, #35072).
        cur = conn.execute(
            """
            UPDATE tasks
               SET status = ?,
                   assignee = COALESCE(?, assignee),
                   claim_lock = NULL,
                   claim_expires = NULL,
                   worker_pid = NULL
             WHERE id = ? AND status = 'running' AND current_run_id = ?
            """,
            (new_status, implementer, task_id, int(current_run_id)),
        )
        if cur.rowcount != 1:
            return False, "task changed during review handoff"
        run_id = _end_run(
            conn,
            task_id,
            outcome="changes_requested",
            status=new_status,
            summary=reason,
        )
        _append_event(
            conn,
            task_id,
            "changes_requested",
            {
                "reason": reason,
                "implementer": implementer,
                "reviewer": reviewer,
                "status": new_status,
            },
            run_id=run_id,
        )
    return True, implementer


def promote_task(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    actor: str,
    reason: Optional[str] = None,
    force: bool = False,
    dry_run: bool = False,
) -> tuple[bool, Optional[str]]:
    """Manually promote a `todo` or `blocked` task to `ready`.

    Mirrors the automatic promotion done by ``recompute_ready`` but
    drives it from a deliberate operator action with an audit-trail
    entry. Refuses to promote if any parent dep is not in a terminal
    state (`done`/`archived`) unless ``force=True``. Does NOT change
    assignee or claim state. Returns ``(True, None)`` on success and
    ``(False, reason)`` if refused. ``dry_run=True`` validates the
    promotion would succeed without mutating state.
    """
    cur_status = _task_status(conn, task_id)
    if cur_status is None:
        return False, f"task {task_id} not found"

    if cur_status not in ("todo", "blocked"):
        return False, (
            f"task {task_id} is {cur_status!r}; promote only applies to "
            f"'todo' or 'blocked'"
        )

    if not force:
        parents = conn.execute(
            "SELECT t.id, t.status FROM tasks t "
            "JOIN task_links l ON l.parent_id = t.id "
            "WHERE l.child_id = ?",
            (task_id,),
        ).fetchall()
        unsatisfied = [
            p["id"] for p in parents
            if p["status"] not in ("done", "archived")
        ]
        if unsatisfied:
            return False, (
                f"unsatisfied parent dependencies: "
                f"{', '.join(unsatisfied)} (use --force to override)"
            )

    if dry_run:
        return True, None

    with write_txn(conn):
        upd = conn.execute(
            "UPDATE tasks SET status = 'ready' "
            "WHERE id = ? AND status IN ('todo', 'blocked')",
            (task_id,),
        )
        if upd.rowcount != 1:
            return False, f"task {task_id} status changed during promotion"
        _append_event(
            conn,
            task_id,
            "promoted_manual",
            {"actor": actor, "reason": reason, "forced": force},
        )

    return True, None


def _reclaim_dangling_run(
    conn: sqlite3.Connection, task_id: str, *, statuses, now: int, note: str,
) -> None:
    """Close a leaked ``current_run_id`` (run row still open) before a status
    flip, preserving the runs invariant (``current_run_id IS NULL`` ⇔ run row
    terminal). No-op in the common path where the prior transition already
    closed the run. Shared by :func:`unblock_task` and
    :func:`reopen_review_task` so the recovery can't drift.
    """
    placeholders = ", ".join("?" for _ in statuses)
    stale = conn.execute(
        f"SELECT current_run_id FROM tasks WHERE id = ? AND status IN ({placeholders})",
        (task_id, *statuses),
    ).fetchone()
    if stale and stale["current_run_id"]:
        conn.execute(
            """
            UPDATE task_runs
               SET status = 'reclaimed', outcome = 'reclaimed',
                   summary = COALESCE(summary, ?),
                   ended_at = ?,
                   claim_lock = NULL, claim_expires = NULL, worker_pid = NULL
             WHERE id = ? AND ended_at IS NULL
            """,
            (note, now, int(stale["current_run_id"])),
        )


def _landing_status_after_parents(conn: sqlite3.Connection, task_id: str) -> str:
    """Return ``'todo'`` if any parent isn't ``done`` yet, else ``'ready'``.

    The parent-completion re-gate shared by :func:`unblock_task` and
    :func:`reopen_review_task`: flipping straight to ``ready`` would bypass the
    parent-completion invariant the dispatcher trusts (it would spawn a child
    whose upstream work isn't finished). If parents are still in progress the
    task waits in ``todo`` until ``recompute_ready`` picks it up. RCA: Bug 2 at
    kanban/boards/cookai/workspaces/t_a6acd07d/root-cause.md. Kept in one place
    so the two transitions can't drift.
    """
    return "ready" if _parents_satisfied(conn, task_id) else "todo"


def unblock_task(conn: sqlite3.Connection, task_id: str) -> bool:
    """Transition ``blocked``/``scheduled`` to its safe resumable phase.

    Defensively closes any stale ``current_run_id`` pointer before flipping
    status. In the common path (``block_task`` closed the run already) this
    is a no-op. If a future or external write left the pointer dangling,
    the leaked run is closed as ``reclaimed`` inside the same txn so the
    runs invariant (``current_run_id IS NULL`` ⇔ run row in terminal
    state) holds for the rest of this function's lifetime.
    """
    now = int(time.time())
    with write_txn(conn):
        resume_status = (
            _resume_status_from_events(conn, task_id)
            if _task_status(conn, task_id) == "blocked"
            else "ready"
        )
        _reclaim_dangling_run(
            conn, task_id, statuses=("blocked", "scheduled"), now=now,
            note="invariant recovery on unblock",
        )
        # Re-gate on parent completion before restoring the source phase.
        landing_status = _landing_status_after_parents(conn, task_id)
        new_status = (
            "review"
            if landing_status == "ready" and resume_status == "review"
            else landing_status
        )
        # NOTE: deliberately does NOT touch ``block_recurrences`` or
        # ``block_kind``. Resetting the recurrence counter on unblock is exactly
        # the amnesia that let a cron unblock → worker re-block loop run
        # unbounded (Dale's report). The counter survives the unblock so that a
        # subsequent same-cause ``block_task`` can detect the loop and route to
        # triage at ``BLOCK_RECURRENCE_LIMIT``. It is reset to 0 only on a
        # successful completion (see ``complete_task``). ``consecutive_failures``
        # (the *dispatcher* spawn/crash/timeout counter — a different signal) is
        # still reset here, which is correct: a deliberate unblock is a fresh
        # start for the dispatcher's retry budget.
        cur = conn.execute(
            "UPDATE tasks SET status = ?, current_run_id = NULL, "
            "consecutive_failures = 0, last_failure_error = NULL "
            "WHERE id = ? AND status IN ('blocked', 'scheduled')",
            (new_status, task_id),
        )
        if cur.rowcount != 1:
            return False
        _append_event(
            conn, task_id, "unblocked",
            (
                {"status": new_status, "resume_status": resume_status}
                if new_status != "ready" or resume_status != "ready"
                else None
            ),
        )
        return True


def reopen_review_task(conn: sqlite3.Connection, task_id: str) -> bool:
    """Transition ``review`` -> ready (or todo) so the implementer re-runs.

    The "changes requested" counterpart of :func:`request_review`: sends the
    task back out of the review lane so the dispatcher re-runs the implementer
    on the new comments. Mirrors :func:`unblock_task` (parent re-gating,
    defensive stale-run close, ``consecutive_failures`` preserved) and emits a
    ``review_reopened`` event.

    Deliberately does NOT touch ``block_recurrences``/``block_kind``: review is
    not a block, so there is no loop counter to reset. (A stale counter from a
    genuine block *before* review is left intact — only :func:`complete_task`
    clears it.) Returns False when the task is missing or not in ``review``.
    """
    now = int(time.time())
    with write_txn(conn):
        _reclaim_dangling_run(
            conn, task_id, statuses=("review",), now=now,
            note="invariant recovery on review reopen",
        )
        new_status = _landing_status_after_parents(conn, task_id)
        review_event = _latest_event(conn, task_id, "review_requested")
        handoff = _json_dict(_row_get(review_event, "payload"))
        implementer = handoff.get("implementer")
        if not isinstance(implementer, str) or not implementer.strip():
            implementer = None
        assignee_sql = ", assignee = ?" if implementer else ""
        params: tuple[Any, ...] = (
            (new_status, implementer, task_id)
            if implementer
            else (new_status, task_id)
        )
        cur = conn.execute(
            "UPDATE tasks SET status = ?, current_run_id = NULL, "
            "claim_lock = NULL, claim_expires = NULL, worker_pid = NULL "
            # consecutive_failures deliberately PRESERVED: review reopen is
            # not a success signal; only complete_task resets the breaker
            # counter (mirrors unblock_task, #35072).
            + assignee_sql
            + " WHERE id = ? AND status = 'review'",
            params,
        )
        if cur.rowcount != 1:
            return False
        payload: dict[str, Any] = {"status": new_status}
        if implementer:
            payload["implementer"] = implementer
        _append_event(
            conn,
            task_id,
            "review_reopened",
            payload if payload != {"status": "ready"} else None,
        )
        return True


def invalidate_descendants_for_parent_reopen(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    author: str,
) -> dict[str, Any]:
    """Retract every dispatchable/completed descendant of a reopened ancestor.

    THE single implementation of done-reopen descendant invalidation. When a
    ``done``/``archived`` ancestor is reopened, every descendant whose state
    assumed its result — ``ready``, ``review``, ``running`` or ``done`` — is
    demoted to ``todo`` and re-gated on the graph. The CLI has no done-reopen
    verb (``reopen-review`` is the review-phase transition), so every surface
    that reopens a done task (dashboard drag-drop / PATCH via
    ``_set_status_direct``) must route through here.

    Transactionality: composes under the caller's open transaction via
    ``write_txn(conn, allow_nested=True)`` so the ancestor's status flip and
    the retractions commit atomically; standalone it opens its own. All SQL
    is inline (no txn-opening helpers).

    Non-silent contract — every invalidated descendant gets a
    ``descendant_invalidated`` event (``ancestor, prior_status, new_status,
    resume_status``), the legacy ``status`` event
    (``reason=ancestor_reopened``) the live feed renders, and a comment
    naming the ancestor so operators see WHY a card moved.

    Live ``running`` descendants are wasted spend: their run is closed
    ``reclaimed`` and the worker killed via :func:`_terminate_reclaimed_worker`
    strictly post-commit, so the audit trail exists BEFORE the worker dies.
    When composing under a caller's transaction the caller MUST drain the
    returned ``terminations`` after its own commit.

    ``consecutive_failures`` resets to 0: ancestor reopen is a deliberate
    operator action, so demoted work gets a fresh breaker budget. This is the
    OPPOSITE of :func:`reopen_review_task` (preserves the counter) because the
    autonomous review loop must not launder its own failure streak.

    Returns ``{"invalidated": [...], "terminations": [...]}`` where each
    invalidated entry is ``{id, prior_status, new_status, resume_status}``
    and each termination is a ``(worker_pid, claim_lock)`` tuple.
    """
    caller_owns_txn = bool(getattr(conn, "in_transaction", False))
    now = int(time.time())
    invalidated: list[dict[str, Any]] = []
    terminations: list[tuple[Optional[int], Optional[str]]] = []
    with write_txn(conn, allow_nested=True):
        rows = conn.execute(
            """
            WITH RECURSIVE descendants(id) AS (
                SELECT child_id FROM task_links WHERE parent_id = ?
                UNION
                SELECT l.child_id
                FROM task_links l
                JOIN descendants d ON d.id = l.parent_id
            )
            SELECT t.id, t.status, t.current_run_id, t.worker_pid, t.claim_lock
            FROM descendants d
            JOIN tasks t ON t.id = d.id
            ORDER BY t.id
            """,
            (task_id,),
        ).fetchall()
        for row in rows:
            previous_status = row["status"]
            if previous_status not in {"ready", "review", "running", "done"}:
                continue
            resume_status = "ready"
            run_id = None
            if previous_status == "review":
                resume_status = "review"
            elif previous_status == "running":
                resume_status = _retry_status_for_run(
                    conn, row["id"], row["current_run_id"]
                )
                terminations.append((row["worker_pid"], row["claim_lock"]))
                run_id = _end_run(
                    conn,
                    row["id"],
                    outcome="reclaimed",
                    status="todo",
                    summary=f"ancestor {task_id} reopened",
                )
            # consecutive_failures = 0: deliberate operator reset — see
            # docstring for why this diverges from reopen_review_task.
            conn.execute(
                "UPDATE tasks SET status = 'todo', completed_at = NULL, "
                "claim_lock = NULL, claim_expires = NULL, worker_pid = NULL, "
                "current_run_id = NULL, consecutive_failures = 0 WHERE id = ?",
                (row["id"],),
            )
            _append_event(
                conn,
                row["id"],
                "descendant_invalidated",
                {
                    "ancestor": task_id,
                    "prior_status": previous_status,
                    "new_status": "todo",
                    "resume_status": resume_status,
                },
                run_id=run_id,
            )
            # Legacy 'status' event kept so existing live-feed consumers
            # still see the move without learning the new event kind.
            _append_event(
                conn,
                row["id"],
                "status",
                {
                    "status": "todo",
                    "reason": "ancestor_reopened",
                    "parent": task_id,
                    "previous_status": previous_status,
                    "resume_status": resume_status,
                },
                run_id=run_id,
            )
            _insert_comment(
                conn, row["id"], author,
                f"Invalidated: ancestor {task_id} was reopened; "
                f"retracted from '{previous_status}' to 'todo' "
                f"(will resume via '{resume_status}').",
                now,
            )
            invalidated.append(
                {
                    "id": row["id"],
                    "prior_status": previous_status,
                    "new_status": "todo",
                    "resume_status": resume_status,
                }
            )
    if not caller_owns_txn:
        # Standalone call: we committed above, so the audit trail is durable
        # — safe to kill workers now. Composed calls leave this to the
        # caller (post-commit), preserving events-before-termination.
        for pid, claim_lock in terminations:
            _terminate_reclaimed_worker(pid, claim_lock)
    return {"invalidated": invalidated, "terminations": terminations}


def specify_triage_task(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    title: Optional[str] = None,
    body: Optional[str] = None,
    assignee: Optional[str] = None,
    author: Optional[str] = None,
) -> bool:
    """Flesh out a triage task and promote it to ``todo``.

    Atomically updates ``title`` / ``body`` / ``assignee`` (when provided)
    and transitions ``status: triage -> todo`` in a single write txn. Returns
    False when the task is missing or not in the ``triage`` column — callers
    should surface that as "nothing to specify" rather than an error.

    ``todo`` (not ``ready``) is the correct landing column: ``recompute_ready``
    promotes parent-free / parent-done todos to ``ready`` on the next
    dispatcher tick, which keeps the normal parent-gating behaviour intact
    for specified tasks that happen to have open parents.

    ``author`` is recorded on an audit comment only when at least one of
    ``title`` / ``body`` / ``assignee`` actually changed — avoids noisy
    comment spam for status-only promotions.
    """
    if title is not None and not title.strip():
        raise ValueError("title cannot be blank")
    assignee = _canonical_assignee(assignee)
    with write_txn(conn):
        existing = conn.execute(
            "SELECT title, body, assignee FROM tasks WHERE id = ? AND status = 'triage'",
            (task_id,),
        ).fetchone()
        if existing is None:
            return False
        sets: list[str] = ["status = 'todo'"]
        params: list[Any] = []
        changed_fields: list[str] = []
        if title is not None and title.strip() != (existing["title"] or ""):
            sets.append("title = ?")
            params.append(title.strip())
            changed_fields.append("title")
        if body is not None and (body or "") != (existing["body"] or ""):
            sets.append("body = ?")
            params.append(body)
            changed_fields.append("body")
        if assignee is not None and assignee != (existing["assignee"] or None):
            sets.append("assignee = ?")
            params.append(assignee)
            changed_fields.append("assignee")
        params.append(task_id)
        cur = conn.execute(
            f"UPDATE tasks SET {', '.join(sets)} "
            f"WHERE id = ? AND status = 'triage'",
            tuple(params),
        )
        if cur.rowcount != 1:
            return False
        if changed_fields and author and author.strip():
            # Inline INSERT (rather than ``add_comment``) because we're
            # already inside this function's write_txn — nested BEGIN
            # IMMEDIATE would raise OperationalError. We also skip the
            # 'commented' event that ``add_comment`` emits, since the
            # 'specified' event below already records the change.
            _insert_comment(
                conn, task_id, author.strip(),
                "Specified — updated " + ", ".join(changed_fields) + " and promoted to todo.",
                int(time.time()),
            )
        _append_event(
            conn,
            task_id,
            "specified",
            {"changed_fields": changed_fields} if changed_fields else None,
        )
    # Outside the write_txn above, so we don't nest BEGIN IMMEDIATE — the
    # ready-promotion pass opens its own IMMEDIATE txn. This runs the same
    # logic the dispatcher would on its next tick, so a specified task
    # with no open parents flips straight to 'ready' here instead of
    # idling in 'todo' until the next sweep.
    recompute_ready(conn)
    return True


def decompose_triage_task(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    root_assignee: Optional[str],
    children: list[dict],
    author: Optional[str] = None,
    auto_promote: bool = True,
) -> Optional[list[str]]:
    """Fan a triage task out into child tasks and promote the root to ``todo``.

    The root task stays alive and becomes the parent of every child —
    when all children reach ``done``, the root promotes to ``ready`` and
    its assignee (typically the orchestrator profile) wakes back up to
    judge completion or spawn more work.

    ``children`` is a list of dicts, each shaped like::

        {
            "title": "...",
            "body": "...",                     # optional
            "assignee": "profile-name",        # optional, None -> default fallback
            "parents": [0, 2],                 # indices into this same children list
        }

    Returns the list of created child task ids (in input order) on
    success. Returns ``None`` when:
      - The root task does not exist
      - The root task is not in ``triage``
      - A cycle would result (caller built a bad graph)

    Validation of titles/assignees happens inside the same write_txn as
    the inserts so a malformed entry aborts the whole decomposition
    cleanly (no orphan children).
    """
    if not children:
        return None
    if root_assignee is not None:
        root_assignee = _canonical_assignee(root_assignee)

    # Pre-validate the children list shape outside the txn. Cheap checks
    # that don't need DB access. Bad input aborts before we touch the DB.
    for idx, child in enumerate(children):
        if not isinstance(child, dict):
            raise ValueError(f"child[{idx}] is not a dict")
        title = child.get("title")
        if not isinstance(title, str) or not title.strip():
            raise ValueError(f"child[{idx}].title is required")
        parents_idx = child.get("parents") or []
        if not isinstance(parents_idx, list):
            raise ValueError(f"child[{idx}].parents must be a list")
        for p in parents_idx:
            if not isinstance(p, int) or p < 0 or p >= len(children):
                raise ValueError(
                    f"child[{idx}].parents[{p}] is not a valid index into children"
                )
            if p == idx:
                raise ValueError(f"child[{idx}] cannot list itself as a parent")

    # Detect cycles in the sibling parent graph (Kahn's topological sort).
    # link_tasks() calls _would_cycle() for every new edge; here we check
    # the entire sibling graph before touching the DB.  A cycle silently
    # deadlocks every involved child in 'todo' because recompute_ready()
    # can never promote them.
    _in_deg = [0] * len(children)
    _adj: list[list[int]] = [[] for _ in range(len(children))]
    for _i, _c in enumerate(children):
        for _p in (_c.get("parents") or []):
            _adj[_p].append(_i)
            _in_deg[_i] += 1
    _queue = [_i for _i in range(len(children)) if _in_deg[_i] == 0]
    _seen = 0
    while _queue:
        _node = _queue.pop()
        _seen += 1
        for _nb in _adj[_node]:
            _in_deg[_nb] -= 1
            if _in_deg[_nb] == 0:
                _queue.append(_nb)
    if _seen != len(children):
        raise ValueError("cyclic dependency detected in decomposed children list")

    # We do the full decomposition in a SINGLE write_txn so it's
    # atomic: either every child is created AND the root flips to
    # ``todo``, or nothing changes. We deliberately do NOT call any
    # kb helper that opens its own write_txn (create_task, link_tasks,
    # add_comment) from inside this block — see architecture.md
    # write_txn pitfalls. Instead we inline the INSERTs and
    # _append_event calls.
    now = int(time.time())
    child_ids: list[str] = []
    with write_txn(conn):
        root_row = conn.execute(
            "SELECT id, status, tenant, workspace_kind, workspace_path "
            "FROM tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
        if root_row is None:
            return None
        if root_row["status"] != "triage":
            return None
        tenant = root_row["tenant"]
        # Children inherit the root's workspace by default so a fan-out
        # of a code-gen task lands in the parent's project dir/worktree
        # rather than throwaway scratch tmp dirs. A child dict can still
        # override with its own 'workspace_kind' / 'workspace_path'.
        root_ws_kind = root_row["workspace_kind"] or "scratch"
        root_ws_path = root_row["workspace_path"]

        # Create children. Status is 'todo' regardless of parents — we
        # link them under the root AFTER creation so the dispatcher
        # sees a coherent state, and recompute_ready() at the end
        # promotes parent-free children to 'ready'.
        for idx, child in enumerate(children):
            new_id = _new_task_id()
            title = child["title"].strip()
            body = child.get("body")
            assignee = _canonical_assignee(child.get("assignee"))
            # Per-child override wins; otherwise inherit the root's
            # workspace. A child that sets workspace_kind without a path
            # falls back to the root path only when kinds match (so a
            # child can't accidentally point a 'dir' at the root's
            # worktree path or vice versa).
            child_ws_kind = child.get("workspace_kind") or root_ws_kind
            if child.get("workspace_path"):
                child_ws_path = child.get("workspace_path")
            elif child_ws_kind == "worktree":
                # Never share one worktree checkout between siblings: the
                # root's literal path would put every child in the same
                # directory on the first-dispatched sibling's branch, with
                # no lock — siblings can be promoted and dispatched
                # concurrently. Leave the path unset so dispatch
                # materializes a fresh <repo>/.worktrees/<child-id> per
                # child from the board anchor.
                child_ws_path = None
            elif child_ws_kind == root_ws_kind:
                child_ws_path = root_ws_path
            else:
                child_ws_path = None
            conn.execute(
                "INSERT INTO tasks "
                "(id, title, body, assignee, status, workspace_kind, "
                " workspace_path, tenant, created_at, created_by) "
                "VALUES (?, ?, ?, ?, 'todo', ?, ?, ?, ?, ?)",
                (
                    new_id,
                    title,
                    body if isinstance(body, str) else None,
                    assignee,
                    child_ws_kind,
                    child_ws_path,
                    tenant,
                    now,
                    (author or "decomposer"),
                ),
            )
            _append_event(
                conn, new_id, "created",
                {"by": author or "decomposer", "from_decompose_of": task_id},
            )
            _inherit_notify_subs(conn, new_id, (task_id,), created_at=now)
            child_ids.append(new_id)

        # Link children to their sibling parents (within the decomposed graph).
        for idx, child in enumerate(children):
            for p_idx in child.get("parents") or []:
                parent_id = child_ids[p_idx]
                child_id = child_ids[idx]
                _link(conn, parent_id, child_id)
                _append_event(
                    conn, child_id, "linked",
                    {"parent": parent_id, "child": child_id},
                )

        # Link the ROOT task as a child of every leaf child — i.e. the
        # root waits for the whole graph. Simpler than computing leaves:
        # link root under every child. Cycle-free because the root is
        # only ever a child here, never a parent of children.
        for cid in child_ids:
            _link(conn, cid, task_id)

        # Flip the root: triage -> todo, set assignee to the orchestrator.
        sets = ["status = 'todo'"]
        params: list[Any] = []
        if root_assignee is not None:
            sets.append("assignee = ?")
            params.append(root_assignee)
        params.append(task_id)
        conn.execute(
            f"UPDATE tasks SET {', '.join(sets)} WHERE id = ?",
            tuple(params),
        )

        # Audit comment + event on the root so the timeline shows the fan-out.
        if author and author.strip():
            _insert_comment(
                conn, task_id, author.strip(),
                "Decomposed into " + ", ".join(child_ids)
                + ". Root will wake when all children complete.",
                now,
            )
        _append_event(
            conn, task_id, "decomposed",
            {
                "child_ids": child_ids,
                "root_assignee": root_assignee,
            },
        )

    # Outside the write_txn: promote parent-free children to 'ready'
    # so the dispatcher picks them up on its next tick. Same pattern
    # specify_triage_task uses.  When auto_promote is False children
    # stay in 'todo' until the user manually promotes them — useful
    # for manual-review-first workflows.
    if auto_promote:
        recompute_ready(conn)
    return child_ids


def archive_task(conn: sqlite3.Connection, task_id: str) -> bool:
    with write_txn(conn):
        cur = conn.execute(
            "UPDATE tasks SET status = 'archived', "
            "    claim_lock = NULL, claim_expires = NULL, worker_pid = NULL "
            "WHERE id = ? AND status != 'archived'",
            (task_id,),
        )
        if cur.rowcount != 1:
            return False
        # If archive happened while a run was still in flight (e.g. user
        # archived a running task from the dashboard), close that run with
        # outcome='reclaimed' so attempt history isn't orphaned.
        run_id = _end_run(
            conn, task_id,
            outcome="reclaimed", status="reclaimed",
            summary="task archived with run still active",
        )
        _append_event(conn, task_id, "archived", None, run_id=run_id)
    # ``archived`` parents no longer block children, same as ``done``.
    # Promote newly-unblocked dependents immediately instead of waiting
    # for a later dispatcher tick.
    recompute_ready(conn)
    # Reap the workspace on archive too — tasks archived without ever
    # completing previously kept their scratch dir / worktree forever.
    _cleanup_workspace(conn, task_id)
    return True


def _delete_task_relations(conn: sqlite3.Connection, task_id: str) -> None:
    """Delete every row referencing ``task_id`` (schema has no ON DELETE CASCADE)."""
    conn.execute(
        "DELETE FROM task_links WHERE parent_id = ? OR child_id = ?", (task_id, task_id),
    )
    for table in ("task_comments", "task_events", "task_runs", "kanban_notify_subs"):
        conn.execute(f"DELETE FROM {table} WHERE task_id = ?", (task_id,))


def delete_archived_task(conn: sqlite3.Connection, task_id: str) -> bool:
    """Permanently remove an already-archived task and its related rows.

    Safety guard: only archived tasks can be deleted. Active / blocked / done
    tasks must be explicitly archived first so accidental data loss requires a
    second deliberate action.
    """
    with write_txn(conn):
        if _task_status(conn, task_id) != "archived":
            return False
        _delete_task_relations(conn, task_id)
        cur = conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
        return cur.rowcount == 1


def delete_task(conn: sqlite3.Connection, task_id: str) -> bool:
    """Hard-delete a task and cascade to all related rows.

    Because the schema does not use ``ON DELETE CASCADE`` foreign keys,
    we explicitly delete from child tables first, then the task row.
    This keeps the operation atomic (single ``write_txn``).

    Returns ``True`` if the task existed and was deleted, ``False``
    if the task was not found.
    """
    with write_txn(conn):
        cur = conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
        if cur.rowcount != 1:
            return False
        _delete_task_relations(conn, task_id)
    recompute_ready(conn)
    return True


# ---------------------------------------------------------------------------
def schedule_task(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    reason: Optional[str] = None,
    expected_run_id: Optional[int] = None,
) -> bool:
    """Park a task in ``scheduled`` so it is waiting on time, not human input.

    ``scheduled`` tasks are intentionally not dispatchable; an external cron,
    human action, or automation can later call ``unblock_task`` to re-gate them
    to ``ready`` (or ``todo`` if parents are still incomplete).
    """
    with write_txn(conn):
        params: list[Any] = [task_id]
        sql = """
            UPDATE tasks
               SET status       = 'scheduled',
                   claim_lock   = NULL,
                   claim_expires= NULL,
                   worker_pid   = NULL
             WHERE id = ?
               AND status IN ('todo', 'ready', 'running', 'blocked')
        """
        if expected_run_id is not None:
            sql += " AND current_run_id = ?"
            params.append(int(expected_run_id))
        cur = conn.execute(sql, params)
        if cur.rowcount != 1:
            return False
        run_id = _end_run(
            conn, task_id,
            outcome="scheduled", status="scheduled",
            summary=reason,
        )
        if run_id is None and reason:
            run_id = _synthesize_ended_run(
                conn, task_id,
                outcome="scheduled",
                summary=reason,
            )
        _append_event(conn, task_id, "scheduled", {"reason": reason}, run_id=run_id)
        return True


# Dispatcher (one-shot pass)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Worker context builder (what a spawned worker sees)
# ---------------------------------------------------------------------------

def build_worker_context(conn: sqlite3.Connection, task_id: str) -> str:
    """Return the full text a worker should read to understand its task.

    Order:
      1. Task title (mandatory).
      2. Task body (optional opening post, capped at 8 KB).
      3. Prior attempts on THIS task (most recent ``_CTX_MAX_PRIOR_ATTEMPTS``
         shown; older attempts collapsed into a one-line summary).
         Each attempt's ``summary`` / ``error`` / ``metadata`` capped at
         ``_CTX_MAX_FIELD_BYTES`` each.
      4. Structured handoff results of every done parent task. Prefers
         ``run.summary`` / ``run.metadata`` when the parent was executed
         via a run; falls back to ``task.result`` for older data. Same
         per-field cap.
      5. Cross-task role history for the assignee (most recent 5
         completed runs on other tasks).
      6. Comment thread (most recent ``_CTX_MAX_COMMENTS`` shown, older
         collapsed).

    All caps exist so worker prompts stay bounded even on pathological
    boards (retry-heavy tasks, comment storms). The per-field char cap
    prevents a single 1 MB summary from dominating context.
    """
    task = get_task(conn, task_id)
    if not task:
        raise ValueError(f"unknown task {task_id}")

    # Single clock reading shared by every relative-age stamp below, so all
    # ages in one rendering are consistent ("3h ago" / "3h ago", not drifting
    # by the seconds it takes to build the block).
    _now = int(time.time())

    def _cap(s: Optional[str], limit: int = _CTX_MAX_FIELD_BYTES) -> str:
        """Truncate a string to `limit` chars with a visible ellipsis."""
        if not s:
            return ""
        s = s.strip()
        if len(s) <= limit:
            return s
        return s[:limit] + f"… [truncated, {len(s) - limit} chars omitted]"

    def _stamp(ts: int) -> str:
        """``YYYY-MM-DD HH:MM`` plus a relative age when one is available."""
        disp = time.strftime("%Y-%m-%d %H:%M", time.localtime(ts))
        age = _relative_age(ts, _now)
        return f"{disp}, {age}" if age else disp

    def _metadata_line(metadata: Any) -> Optional[str]:
        if not metadata:
            return None
        try:
            return f"_metadata_: `{_cap(json.dumps(metadata, ensure_ascii=False, sort_keys=True))}`"
        except Exception:
            return None

    def _tail(items: list, cap: int, noun: str) -> tuple[list, Optional[str]]:
        """Keep the newest ``cap`` items; describe the omitted head, if any."""
        omitted = max(0, len(items) - cap)
        if not omitted:
            return items, None
        return items[-cap:], (
            f"_({omitted} earlier {noun}{'s' if omitted != 1 else ''} "
            f"omitted; showing most recent {cap})_"
        )

    lines: list[str] = []
    lines.append(f"# Kanban task {task.id}: {task.title}")
    lines.append("")
    lines.append(f"Assignee: {task.assignee or '(unassigned)'}")
    lines.append(f"Status:   {task.status}")
    if task.tenant:
        lines.append(f"Tenant:   {task.tenant}")
    lines.append(f"Workspace: {task.workspace_kind} @ {task.workspace_path or '(unresolved)'}")
    if task.max_runtime_seconds is not None:
        terminal_timeout = _worker_terminal_timeout_env(
            task.max_runtime_seconds,
            os.environ.get("TERMINAL_TIMEOUT"),
        )
        effective_terminal_timeout = terminal_timeout or os.environ.get("TERMINAL_TIMEOUT")
        lines.append(f"Max runtime: {task.max_runtime_seconds}s")
        if effective_terminal_timeout:
            lines.append(f"Terminal timeout: {effective_terminal_timeout}s")
    if task.branch_name:
        lines.append(f"Branch:   {task.branch_name}")
    lines.append("")

    if task.body and task.body.strip():
        lines.append("## Body")
        lines.append(_cap(task.body, _CTX_MAX_BODY_BYTES))
        lines.append("")

    # Attachments — files uploaded to this task (PDFs, source docs,
    # images). Surface the absolute on-disk path so the worker, which has
    # full file-tool access, can read them directly (read_file, terminal
    # `pdftotext`, etc.). On the local terminal backend the path resolves
    # as-is; remote backends need the kanban attachments dir mounted.
    attachments = list_attachments(conn, task_id)
    if attachments:
        lines.append("## Attachments")
        lines.append(
            "Files attached to this task. Read them with the file/terminal "
            "tools at the absolute paths below:"
        )
        for att in attachments:
            size_kb = max(1, (att.size + 1023) // 1024) if att.size else 0
            size_str = f", {size_kb} KB" if size_kb else ""
            ctype = f", {att.content_type}" if att.content_type else ""
            lines.append(f"- `{att.filename}`{ctype}{size_str} → `{att.stored_path}`")
        lines.append("")

    # Prior attempts — show closed runs so a retrying worker sees the
    # history. Skip the currently-active run (that's this worker).
    # Cap at _CTX_MAX_PRIOR_ATTEMPTS most-recent closed runs; older
    # attempts get collapsed into a one-line marker so the worker knows
    # more exist without bloating the prompt.
    all_prior = [r for r in list_runs(conn, task_id) if r.ended_at is not None]
    # list_runs returns ascending by started_at; "most recent" = last N
    shown, omitted_note = _tail(all_prior, _CTX_MAX_PRIOR_ATTEMPTS, "attempt")
    first_shown_idx = len(all_prior) - len(shown) + 1
    if shown:
        lines.append("## Prior attempts on this task")
        if omitted_note:
            lines.append(omitted_note)
        for offset, run in enumerate(shown):
            idx = first_shown_idx + offset
            profile = run.profile or "(unknown)"
            outcome = run.outcome or run.status
            lines.append(f"### Attempt {idx} — {outcome} ({profile}, {_stamp(run.started_at)})")
            if run.summary and run.summary.strip():
                lines.append(_cap(run.summary))
            if run.error and run.error.strip():
                lines.append(f"_error_: {_cap(run.error)}")
            meta_line = _metadata_line(run.metadata)
            if meta_line:
                lines.append(meta_line)
            lines.append("")

    # Parents: prefer the most-recent 'completed' run's summary + metadata,
    # fall back to ``task.result`` when no run rows exist (legacy DBs,
    # or tasks completed before the runs table landed).
    parent_rows = conn.execute(
        "SELECT parent_id FROM task_links WHERE child_id = ? ORDER BY parent_id",
        (task_id,),
    ).fetchall()
    parent_ids = [r["parent_id"] for r in parent_rows]

    if parent_ids:
        wrote_header = False
        for pid in parent_ids:
            pt = get_task(conn, pid)
            if not pt or pt.status != "done":
                continue
            runs = [r for r in list_runs(conn, pid) if r.outcome == "completed"]
            runs.sort(key=lambda r: r.started_at, reverse=True)
            run = runs[0] if runs else None

            if not wrote_header:
                lines.append("## Parent task results")
                lines.append(
                    "_Handoffs from upstream tasks, captured when each parent "
                    "completed (see age below). These are point-in-time "
                    "snapshots, not live state — if a result drives your "
                    "current work and it's not recent, re-verify against the "
                    "source before acting on it as current._"
                )
                wrote_header = True

            # When did this parent's result get produced? Prefer the
            # completed run's end time; fall back to the task's completed_at.
            done_ts = None
            if run is not None and getattr(run, "ended_at", None):
                done_ts = run.ended_at
            elif pt.completed_at:
                done_ts = pt.completed_at
            age = _relative_age(done_ts, _now)
            lines.append(f"### {pid}" + (f" (completed {age})" if age else ""))

            body_lines: list[str] = []
            if run is not None and run.summary and run.summary.strip():
                body_lines.append(_cap(run.summary))
            elif pt.result:
                body_lines.append(_cap(pt.result))
            else:
                body_lines.append("(no result recorded)")

            meta_line = _metadata_line(run.metadata) if run is not None else None
            if meta_line:
                body_lines.append(meta_line)
            lines.extend(body_lines)
            lines.append("")

    # Cross-task role history: what else has THIS assignee completed
    # recently? Gives the worker implicit continuity — "I'm the reviewer
    # and my last three reviews focused on security" — without forcing
    # the user to wire anything into SOUL.md / MEMORY.md. Bounded to the
    # most recent 5 completed runs, excluding this task so the retry
    # section above isn't duplicated. Safe on assignee=None (skipped).
    if task.assignee:
        role_rows = conn.execute(
            "SELECT t.id, t.title, r.summary, r.ended_at "
            "FROM task_runs r JOIN tasks t ON r.task_id = t.id "
            "WHERE r.profile = ? AND r.task_id != ? "
            "  AND r.outcome = 'completed' "
            "ORDER BY r.ended_at DESC LIMIT 5",
            (task.assignee, task_id),
        ).fetchall()
        if role_rows:
            lines.append(f"## Recent work by @{task.assignee}")
            for row in role_rows:
                s = (row["summary"] or "").strip().splitlines()
                first = s[0][:200] if s else "(no summary)"
                lines.append(
                    f"- {row['id']} — {row['title']} ({_stamp(int(row['ended_at']))}): {first}"
                )
            lines.append("")

    # Comments: cap at the most-recent _CTX_MAX_COMMENTS so
    # comment-storm tasks don't blow out the worker's prompt. Older
    # comments summarised in a one-line marker like prior attempts.
    shown_c, omitted_note = _tail(list_comments(conn, task_id), _CTX_MAX_COMMENTS, "comment")
    if shown_c:
        lines.append("## Comment thread")
        if omitted_note:
            lines.append(omitted_note)
        for c in shown_c:
            # Explicit "comment from worker" framing so operator-controlled
            # HERMES_PROFILE values like "hermes-system" or "operator" can't be
            # misread by the next worker as a system directive above the
            # (attacker-influenceable) comment body. Defense-in-depth on top of
            # the closed LLM-controlled author-forgery surface.
            safe_author = (c.author or "").replace("`", "")
            lines.append(f"comment from worker `{safe_author}` at {_stamp(c.created_at)}:")
            lines.append(_cap(c.body, _CTX_MAX_COMMENT_BYTES))
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Stats + SLA helpers
# ---------------------------------------------------------------------------

def board_stats(conn: sqlite3.Connection) -> dict:
    """Per-status + per-assignee counts, plus the oldest ``ready`` age in
    seconds (the clearest staleness signal for a router or HUD).
    """
    by_status: dict[str, int] = {}
    for row in conn.execute(
        "SELECT status, COUNT(*) AS n FROM tasks "
        "WHERE status != 'archived' GROUP BY status"
    ):
        by_status[row["status"]] = int(row["n"])

    by_assignee: dict[str, dict[str, int]] = {}
    for row in conn.execute(
        "SELECT assignee, status, COUNT(*) AS n FROM tasks "
        "WHERE status != 'archived' AND assignee IS NOT NULL "
        "GROUP BY assignee, status"
    ):
        by_assignee.setdefault(row["assignee"], {})[row["status"]] = int(row["n"])

    oldest_row = conn.execute(
        "SELECT MIN(created_at) AS ts FROM tasks WHERE status = 'ready'"
    ).fetchone()
    now = int(time.time())
    oldest_ready_age = (
        (now - int(oldest_row["ts"]))
        if oldest_row and oldest_row["ts"] is not None else None
    )

    return {
        "by_status": by_status,
        "by_assignee": by_assignee,
        "oldest_ready_age_seconds": oldest_ready_age,
        "now": now,
    }


def _to_epoch(val) -> Optional[int]:
    """Normalise a timestamp to unix epoch seconds.

    Accepts ints (pass-through), numeric strings, and ISO-8601 strings.
    Returns ``None`` for ``None`` / empty values.
    """
    if val is None:
        return None
    if isinstance(val, int):
        return val
    if isinstance(val, float):
        return int(val)
    s = str(val).strip()
    if not s:
        return None
    try:
        return int(s)
    except ValueError:
        pass
    # ISO-8601 fallback (e.g. '2026-05-10T15:00:00Z')
    try:
        from datetime import datetime
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return int(dt.timestamp())
    except (ValueError, OSError):
        return None


def task_age(task: Task) -> dict:
    """Return age metrics for a single task. All values are seconds or None."""
    now = int(time.time())
    _c = _to_epoch(task.created_at)
    _s = _to_epoch(task.started_at)
    _co = _to_epoch(task.completed_at)
    age_since_created = now - _c if _c is not None else None
    age_since_started = now - _s if _s is not None else None
    time_to_complete = (
        _co - (_s or _c) if _co is not None else None
    )
    return {
        "created_age_seconds": age_since_created,
        "started_age_seconds": age_since_started,
        "time_to_complete_seconds": time_to_complete,
    }


# ---------------------------------------------------------------------------
# Retention + garbage collection
# ---------------------------------------------------------------------------

def gc_events(
    conn: sqlite3.Connection, *, older_than_seconds: int = 30 * 24 * 3600,
) -> int:
    """Delete task_events rows older than ``older_than_seconds`` for tasks
    in a terminal state (``done`` or ``archived``). Returns the number of
    rows deleted. Running / ready / blocked tasks keep their full event
    history."""
    cutoff = int(time.time()) - int(older_than_seconds)
    with write_txn(conn):
        cur = conn.execute(
            "DELETE FROM task_events WHERE created_at < ? AND task_id IN "
            "(SELECT id FROM tasks WHERE status IN ('done', 'archived'))",
            (cutoff,),
        )
    return int(cur.rowcount or 0)


def gc_worker_logs(
    *, older_than_seconds: int = 30 * 24 * 3600,
    board: Optional[str] = None,
) -> int:
    """Delete worker log files older than ``older_than_seconds``. Returns
    the number of files removed. Kept separate from ``gc_events`` because
    log files live on disk, not in SQLite. Scoped to ``board`` (defaults
    to the active board) — per-board isolation means deleting logs from
    board A cannot touch board B's logs."""
    log_dir = worker_logs_dir(board=board)
    if not log_dir.exists():
        return 0
    cutoff = time.time() - older_than_seconds
    removed = 0
    for p in log_dir.iterdir():
        try:
            if p.is_file() and p.stat().st_mtime < cutoff:
                p.unlink()
                removed += 1
        except OSError:
            continue
    return removed


# ---------------------------------------------------------------------------
# Worker log accessor
# ---------------------------------------------------------------------------

def worker_log_path(task_id: str, *, board: Optional[str] = None) -> Path:
    """Return the path to a worker's log file. The file may not exist
    (task never spawned, or log already GC'd).

    When ``board`` is None, resolves via the active board (env var →
    current-board file → default). The dispatcher always passes the
    board explicitly to avoid any resolution ambiguity when multiple
    boards exist."""
    return worker_logs_dir(board=board) / f"{task_id}.log"


def read_worker_log(
    task_id: str, *, tail_bytes: Optional[int] = None,
    board: Optional[str] = None,
) -> Optional[str]:
    """Read the worker log for ``task_id``. Returns None if the file
    doesn't exist. If ``tail_bytes`` is set, only the last N bytes are
    returned (useful for the dashboard drawer which shouldn't page megabytes)."""
    path = worker_log_path(task_id, board=board)
    if not path.exists():
        return None
    try:
        if tail_bytes is None:
            return path.read_text(encoding="utf-8", errors="replace")
        size = path.stat().st_size
        with open(path, "rb") as f:
            if size > tail_bytes:
                f.seek(size - tail_bytes)
                # Skip a partial line if we tailed mid-line. But if the
                # window has no newline at all (one giant log line),
                # readline() would eat everything — in that case don't
                # skip and return the raw tail.
                probe = f.tell()
                partial = f.readline()
                if not partial.endswith(b"\n") and f.tell() >= size:
                    f.seek(probe)
            data = f.read()
        return data.decode("utf-8", errors="replace")
    except OSError:
        return None


# ---------------------------------------------------------------------------
# Assignee enumeration (known profiles + per-profile board stats)
# ---------------------------------------------------------------------------

def list_profiles_on_disk() -> list[str]:
    """Return the set of assignee/profile names discovered on disk.

    Includes:
    - named profiles under ``<default-root>/profiles/<name>/config.yaml``
    - the implicit ``default`` profile when the default Hermes root exists

    Reads profile paths directly so this module has no import dependency on
    ``hermes_cli.profiles`` (which pulls in a large chunk of the CLI startup
    path).
    """
    try:
        from hermes_constants import get_default_hermes_root
        default_root = get_default_hermes_root()
        profiles_dir = default_root / "profiles"
    except Exception:
        return []

    names: set[str] = set()
    if default_root.exists():
        names.add("default")

    if profiles_dir.is_dir():
        try:
            for entry in sorted(profiles_dir.iterdir()):
                if not entry.is_dir():
                    continue
                if (entry / "config.yaml").is_file():
                    names.add(entry.name)
        except OSError:
            pass

    return sorted(names)


def known_assignees(conn: sqlite3.Connection) -> list[dict]:
    """Return every assignee name known to the board or on disk.

    Each entry is ``{"name": str, "on_disk": bool, "counts": {status: n}}``.
    A name is included when it's a configured profile on disk OR when
    any non-archived task has it as the assignee. Used by:

    - ``hermes kanban assignees`` for the terminal.
    - The dashboard assignee dropdown (so a fresh profile appears in
      the picker even before it's been given any task).
    - Router-profile heuristics ("who's overloaded?") without scanning
      the whole board.
    """
    on_disk = set(list_profiles_on_disk())

    # Count tasks per (assignee, status), excluding archived.
    counts: dict[str, dict[str, int]] = {}
    for row in conn.execute(
        "SELECT assignee, status, COUNT(*) AS n FROM tasks "
        "WHERE status != 'archived' AND assignee IS NOT NULL "
        "GROUP BY assignee, status"
    ):
        counts.setdefault(row["assignee"], {})[row["status"]] = int(row["n"])

    names = sorted(on_disk | set(counts.keys()))
    return [
        {
            "name": name,
            "on_disk": name in on_disk,
            "counts": counts.get(name, {}),
        }
        for name in names
    ]


# ---------------------------------------------------------------------------
# Runs (attempt history on a task)
# ---------------------------------------------------------------------------

def list_runs(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    include_active: bool = True,
    state_type: Optional[str] = None,
    state_name: Optional[str] = None,
) -> list[Run]:
    """Return all runs for ``task_id`` in start order.

    ``include_active=True`` (default) includes the currently-running
    attempt if any. Set False to return only closed runs (useful for
    "how many prior attempts have there been?" checks).

    When ``state_type`` and ``state_name`` are set, restrict to rows
    where that column equals ``state_name`` (``state_type`` is
    ``status`` or ``outcome``). Both must be passed together.
    """
    if (state_type is None) ^ (state_name is None):
        raise ValueError("state_type and state_name must both be set or both omitted")
    if state_type is not None and state_type not in ("status", "outcome"):
        raise ValueError("state_type must be 'status' or 'outcome'")
    q = "SELECT * FROM task_runs WHERE task_id = ?"
    params: list[Any] = [task_id]
    if not include_active:
        q += " AND ended_at IS NOT NULL"
    if state_type is not None:
        q += f" AND {state_type} = ?"
        params.append(state_name)
    q += " ORDER BY started_at ASC, id ASC"
    rows = conn.execute(q, params).fetchall()
    return [Run.from_row(r) for r in rows]


def get_run(conn: sqlite3.Connection, run_id: int) -> Optional[Run]:
    row = conn.execute(
        "SELECT * FROM task_runs WHERE id = ?", (int(run_id),),
    ).fetchone()
    return Run.from_row(row) if row else None


def latest_run(conn: sqlite3.Connection, task_id: str) -> Optional[Run]:
    """Return the most recent run regardless of outcome (active or closed)."""
    row = conn.execute(
        "SELECT * FROM task_runs WHERE task_id = ? "
        "ORDER BY started_at DESC, id DESC LIMIT 1",
        (task_id,),
    ).fetchone()
    return Run.from_row(row) if row else None


def latest_summary(conn: sqlite3.Connection, task_id: str) -> Optional[str]:
    """Return the latest non-null ``task_runs.summary`` for ``task_id``.

    The worker writes its handoff to ``task_runs.summary``
    via ``complete_task(summary=...)``; ``tasks.result`` is left empty
    unless the caller passes ``result=`` explicitly. Dashboards and CLI
    "show" views need this value to surface what a worker actually did
    — without it, ``tasks.result`` is NULL and the task looks like a
    no-op even when the run completed.

    Picks the most recent run by ``ended_at`` (falling back to ``id``
    for ties or unfinished rows). Returns None if no run has a summary.
    """
    row = conn.execute(
        "SELECT summary FROM task_runs "
        "WHERE task_id = ? AND summary IS NOT NULL AND summary != '' "
        "ORDER BY COALESCE(ended_at, started_at) DESC, id DESC LIMIT 1",
        (task_id,),
    ).fetchone()
    return row["summary"] if row else None


def latest_summaries(
    conn: sqlite3.Connection, task_ids: Iterable[str]
) -> dict[str, str]:
    """Batch-fetch latest non-null summaries for a list of task ids.

    Used by the dashboard board endpoint to attach ``latest_summary`` to
    every card in a single SQL query, avoiding the N+1 pattern of
    calling :func:`latest_summary` per task. Returns a dict mapping
    ``task_id`` → summary string, omitting tasks with no summary.

    Approach: a window function picks the newest non-null-summary row
    per ``task_id``; works against SQLite ≥ 3.25 (default on every
    supported platform).
    """
    ids = list(task_ids)
    if not ids:
        return {}
    placeholders = ",".join("?" for _ in ids)
    rows = conn.execute(
        f"""
        SELECT task_id, summary FROM (
            SELECT task_id, summary,
                   ROW_NUMBER() OVER (
                       PARTITION BY task_id
                       ORDER BY COALESCE(ended_at, started_at) DESC, id DESC
                   ) AS rn
              FROM task_runs
             WHERE task_id IN ({placeholders})
               AND summary IS NOT NULL AND summary != ''
        ) WHERE rn = 1
        """,
        ids,
    ).fetchall()
    return {r["task_id"]: r["summary"] for r in rows}


# ---------------------------------------------------------------------------
# Split modules — re-exported so ``kanban_db.<name>`` keeps resolving (and
# stays the single monkeypatch target).
# ---------------------------------------------------------------------------
from hermes_cli.kanban_db_connect import (  # noqa: E402,F401
    DEFAULT_BUSY_TIMEOUT_MS,
    KanbanDbCorruptError,
    RepairResult,
    _BUSY_MAX_RETRIES,
    _BUSY_RETRY_MAX_S,
    _BUSY_RETRY_MIN_S,
    _CORRUPT_BACKUP_RETENTION,
    _EARLY_TASK_COLUMNS,
    _INITIALIZED_PATHS,
    _INIT_LOCK,
    _INIT_LOCK_POLL_SECONDS,
    _INIT_LOCK_TIMEOUT_SECONDS,
    _LAST_WAL_CHECKPOINT,
    _LATER_TASK_COLUMNS,
    _REBUILD_SPECS,
    _RENAMED_TASK_COLUMNS,
    _REPAIRABLE_INDEX_ERROR_PATTERNS,
    _SQLITE_HEADER,
    _WAL_CHECKPOINT_INTERVAL_SECONDS,
    _WAL_CHECKPOINT_LOCK,
    _attempt_index_reindex_repair,
    _backup_corrupt_db,
    _check_file_length_invariant,
    _cross_process_init_lock,
    _dispatch_tick_lock,
    _execute_boundary_with_retry,
    _guard_existing_db_is_healthy,
    _integrity_messages_ok,
    _is_busy_error,
    _looks_like_tls_record_at,
    _maybe_checkpoint_wal,
    _migrate_add_optional_columns,
    _open_configured,
    _probe_integrity,
    _prune_corrupt_backups,
    _rebuild_drifted_tables,
    _repairable_index_names,
    _resolve_busy_timeout_ms,
    _run_integrity_check,
    _schema_is_present,
    _sqlite_connect,
    _table_has_drifted,
    _try_lock_nb,
    _unlock,
    _validate_sqlite_header,
    connect,
    connect_closing,
    init_db,
    repair_db,
    write_txn,
)
from hermes_cli.kanban_db_workspace import (  # noqa: E402,F401
    _SCRATCH_TIP_MESSAGE,
    _SCRATCH_TIP_SENTINEL_NAME,
    _cleanup_worker_tmux,
    _cleanup_workspace,
    _cleanup_worktree_workspace,
    _ensure_git_worktree,
    _git_branch_exists,
    _git_common_dir,
    _git_current_branch,
    _git_dir,
    _git_toplevel,
    _is_linked_worktree_checkout,
    _is_managed_scratch_path,
    _managed_scratch_path_info,
    _mark_scratch_tip_shown,
    _maybe_emit_scratch_tip,
    _nearest_existing_path,
    _repo_root_for_worktree_target,
    _resolve_worktree_workspace,
    _scratch_tip_sentinel_path,
    _scratch_tip_shown,
    _scratch_workspace,
    _try_cleanup_parent_workspaces,
    resolve_workspace,
    set_branch_name,
    set_workspace_path,
)
from hermes_cli.kanban_db_dispatch import (  # noqa: E402,F401
    DEFAULT_FAILURE_LIMIT,
    DEFAULT_LOG_BACKUP_COUNT,
    DEFAULT_LOG_ROTATE_BYTES,
    DEFAULT_RATE_LIMIT_COOLDOWN_SECONDS,
    DEFAULT_SPAWN_FAILURE_LIMIT,
    DERIVED_MAX_IN_PROGRESS_CEILING,
    DERIVED_MAX_IN_PROGRESS_FLOOR,
    DispatchResult,
    KANBAN_TERMINAL_TIMEOUT_GRACE_SECONDS,
    MEMORY_GUARD_MB_PER_WORKER,
    _PROTOCOL_VIOLATION_FAILURE_LIMIT,
    _PROTOCOL_VIOLATION_SCAN_LIMIT,
    _RECENT_WORKER_EXITS_MAX,
    _RECENT_WORKER_EXIT_TTL_SECONDS,
    _RESPAWN_BLOCKER_RE,
    _RESPAWN_GUARD_PR_URL_RE,
    _RESPAWN_GUARD_PR_WINDOW,
    _RESPAWN_GUARD_SUCCESS_WINDOW,
    _STALE_HEARTBEAT_GAP_SECONDS,
    _absolute_hermes_path,
    _apply_default_assignee,
    _classify_worker_exit,
    _clear_failure_counter,
    _default_spawn,
    _defer_reclaim_for_live_worker,
    _dispatch_lane_task,
    _dispatch_once_locked,
    _error_fingerprint,
    _has_spawnable,
    _hermes_path_argv,
    _is_windows_batch_shim,
    _looks_like_path,
    _memory_pressure_level,
    _module_hermes_argv,
    _path_search_names,
    _pid_alive,
    _positive_int,
    _protocol_violation_streak,
    _recent_worker_exits,
    _record_spawn_failure,
    _record_task_failure,
    _record_worker_exit,
    _resolve_hermes_argv,
    _resolve_worker_cli_toolsets,
    _retag_legacy_worker_sessions,
    _retagged_workspace_roots,
    _rotate_worker_log,
    _rotated_log_path,
    _safe_which_no_cwd,
    _set_worker_pid,
    _system_memory_sample,
    _terminate_reclaimed_worker,
    _worker_survived_termination,
    _worker_terminal_timeout_env,
    check_respawn_guard,
    configured_max_in_progress,
    count_running_tasks,
    count_running_tasks_other_boards,
    derive_default_max_in_progress,
    detect_crashed_workers,
    detect_stale_running,
    dispatch_once,
    enforce_max_runtime,
    has_spawnable_ready,
    has_spawnable_review,
    heartbeat_worker,
    reap_worker_zombies,
    reconcile_orphaned_running,
    resolve_max_in_progress,
    review_dispatch_enabled,
    run_daemon,
    worker_log_rotation_config,
)
from hermes_cli.kanban_db_notify import (  # noqa: E402,F401
    _NOTIFY_DELIVERY_MODES,
    _decode_notify_delivery_metadata,
    _encode_notify_delivery_metadata,
    _notify_cursor,
    _notify_profile_filter,
    add_notify_sub,
    advance_notify_cursor,
    claim_unseen_events_for_sub,
    count_notify_subs,
    list_notify_subs,
    purge_stale_done_notify_subs,
    remove_notify_sub,
    rewind_notify_cursor,
    unseen_events_for_sub,
)
