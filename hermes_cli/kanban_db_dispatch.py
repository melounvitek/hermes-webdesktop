"""Dispatcher: crash/stale/orphan detection, failure accounting and the respawn circuit breaker, memory-aware concurrency caps, the one-shot ``dispatch_once`` pass, worker spawning (``_default_spawn``), worker-log rotation and the long-lived ``run_daemon`` loop.

Split out of ``hermes_cli.kanban_db``; every name is re-exported there, and
origin-resident helpers are reached late-bound via ``_kb`` so monkeypatching
``kanban_db.<name>`` keeps working.
"""

from __future__ import annotations

import contextlib
import logging
import os
import re
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any
from typing import Mapping
from typing import Optional
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hermes_cli.kanban_db import Task

# Log-record parity with the origin module.
_log = logging.getLogger("hermes_cli.kanban_db")


# After this many consecutive non-success attempts on a task/profile, the
# dispatcher stops retrying and parks the task in ``blocked`` with a reason so
# a human can investigate. Prevents retry storms when a worker repeatedly times
# out, crashes, or cannot spawn.
DEFAULT_FAILURE_LIMIT = 2


# Legacy alias — callers / tests still reference the old name.
DEFAULT_SPAWN_FAILURE_LIMIT = DEFAULT_FAILURE_LIMIT


# Max bytes to keep in a single worker log file. The dispatcher truncates
# and rotates on spawn if the file is larger than this at spawn time.
DEFAULT_LOG_ROTATE_BYTES = 2 * 1024 * 1024   # 2 MiB


DEFAULT_LOG_BACKUP_COUNT = 1


# Keep a little wall-clock budget for the worker to observe a terminal timeout
# and call kanban_block/kanban_complete before max_runtime_seconds kills it.
KANBAN_TERMINAL_TIMEOUT_GRACE_SECONDS = 30


# ---------------------------------------------------------------------------
# Respawn guard constants
# ---------------------------------------------------------------------------

# Patterns in last_failure_error that indicate a quota / auth blocker.
# These errors won't resolve by retrying immediately — auto-block instead.
_RESPAWN_BLOCKER_RE = re.compile(
    r"\b(quota|rate[\s_\-]?limit|429|403|auth\w*|"
    r"unauthorized|forbidden|billing|subscription|"
    r"access[\s_]denied|permission[\s_]denied|"
    r"invalid[\s_]api[\s_]key)\b",
    re.IGNORECASE,
)


# Within this window a completed run counts as "recent proof"; don't re-spawn.
_RESPAWN_GUARD_SUCCESS_WINDOW = 3600  # 1 hour


# Cooldown after a rate-limited (quota-wall) requeue before the dispatcher
# re-spawns the worker. Without this, a task released by the rate-limit path
# would be re-spawned on the very next tick and immediately bounce off the
# same quota wall, burning a worker slot every tick for hours. The cooldown
# spaces retries out so the board keeps cheaply probing whether quota is back
# without thrashing. Overridable via ``HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS``
# for operators who want a tighter/looser probe cadence.
DEFAULT_RATE_LIMIT_COOLDOWN_SECONDS = 300  # 5 minutes


# Within this window a GitHub PR URL in a comment blocks re-spawn.
_RESPAWN_GUARD_PR_WINDOW = 86400  # 24 hours


# Pattern matching a GitHub PR URL in task comments.
_RESPAWN_GUARD_PR_URL_RE = re.compile(
    r"https?://github\.com/[^/\s]+/[^/\s]+/pull/\d+",
    re.IGNORECASE,
)


@dataclass
class DispatchResult:
    """Outcome of a single ``dispatch`` pass."""

    reclaimed: int = 0
    promoted: int = 0
    reconciled_orphans: list[str] = field(default_factory=list)
    """Task ids requeued by :func:`reconcile_orphaned_running` this tick —
    ``running`` cards whose claim bookkeeping was broken (no valid claim,
    dead/gone worker). See the reconciliation pass for details."""
    spawned: list[tuple[str, str, str]] = field(default_factory=list)
    """List of ``(task_id, assignee, workspace_path)`` triples."""
    skipped_unassigned: list[str] = field(default_factory=list)
    """Ready task ids skipped because they have no assignee at all.
    Operator-actionable — usually a misfiled task waiting for routing."""
    auto_assigned_default: list[str] = field(default_factory=list)
    """Task ids that were unassigned in the DB and had
    ``kanban.default_assignee`` applied this tick before spawning (#27145).
    Surfaces the auto-assignment to telemetry / CLI / dashboard so the
    operator can see when the dispatcher is acting on the fallback rule
    rather than on explicit per-task assignments."""
    skipped_nonspawnable: list[str] = field(default_factory=list)
    """Ready task ids skipped because their assignee names a control-plane
    lane (a Claude Code terminal like ``orion-cc``) rather than a Hermes
    profile. Expected steady-state on multi-lane setups; NOT an
    operator-actionable failure. Tracked separately so health telemetry
    can distinguish "real stuck" (nothing spawned but spawnable work
    available) from "correctly idle" (nothing spawnable in the queue)."""
    skipped_per_profile_capped: list[tuple[str, str, int]] = field(default_factory=list)
    """Tasks deferred this tick because their assignee is already at
    ``kanban.max_in_progress_per_profile`` (#21582). Each entry is
    ``(task_id, assignee, current_running_count)``. NOT an
    operator-actionable failure — the task will be picked up on a
    subsequent tick when the assignee has capacity. Separate bucket so
    telemetry / dashboards can show "this profile is busy" vs
    "task is genuinely stuck"."""
    crashed: list[str] = field(default_factory=list)
    """Task ids reclaimed because their worker PID disappeared."""
    auto_blocked: list[str] = field(default_factory=list)
    """Task ids auto-blocked by the spawn-failure circuit breaker."""
    timed_out: list[str] = field(default_factory=list)
    """Task ids whose workers exceeded ``max_runtime_seconds``."""
    stale: list[str] = field(default_factory=list)
    """Task ids reclaimed because no progress (heartbeat) was seen
    within ``dispatch_stale_timeout_seconds``."""
    respawn_guarded: list[tuple[str, str]] = field(default_factory=list)
    """Tasks skipped by the respawn guard, as ``(task_id, reason)`` pairs.

    Reasons: ``"blocker_auth"`` (quota/auth error — also auto-blocked),
    ``"recent_success"`` (completed run within guard window),
    ``"active_pr"`` (GitHub PR URL in a recent comment)."""
    rate_limited: list[str] = field(default_factory=list)
    """Task ids whose workers bailed on a provider rate-limit / quota wall
    (EX_TEMPFAIL sentinel exit) and were released back to ``ready`` WITHOUT
    counting a failure. These never trip the circuit breaker — a long quota
    window just makes the task bounce cheaply until the window clears."""
    skipped_locked: bool = False
    """True when this tick was skipped because another process already held
    the board's dispatch lock (issue #35240). A losing dispatcher does no
    DB writes this tick — the lock holder is making progress on the same
    board. This is the steady-state signal that a single-writer guard is
    actively preventing two dispatchers from racing on ``kanban.db``."""
    memory_pressure: Optional[str] = None
    """System memory pressure observed at spawn time when the memory guard
    restricted this tick (OOF-30/OOF-77): ``"critical"`` — no new workers
    were spawned this tick; ``"elevated"`` — at most one new worker was
    spawned. ``None`` when memory was fine/unknown and the guard imposed
    no restriction. Reclaim/promotion bookkeeping still ran either way;
    deferred tasks stay queued for the next tick."""


# Bounded registry of recently-reaped worker child exits, populated by the
# reap loop at the top of ``dispatch_once`` and consulted by
# ``detect_crashed_workers`` to classify a dead-pid task.
#
# Entry: ``pid -> (raw_wait_status, reaped_at_epoch)``. We keep raw status
# so both ``os.WIFEXITED`` / ``os.WEXITSTATUS`` and ``os.WIFSIGNALED`` can
# be consulted. Entries are trimmed by age (and total size cap as a
# belt-and-braces against unbounded growth on exotic platforms).
_RECENT_WORKER_EXIT_TTL_SECONDS = 600


_RECENT_WORKER_EXITS_MAX = 4096


_recent_worker_exits: "dict[int, tuple[int, float]]" = {}


def _record_worker_exit(pid: int, raw_status: int) -> None:
    """Record a reaped child's exit status for later classification.

    Called from the reap loop in ``dispatch_once``. Safe to call many
    times; duplicate pids overwrite (pids can cycle, latest wins).
    """
    if not pid or pid <= 0:
        return
    now = time.time()
    _recent_worker_exits[int(pid)] = (int(raw_status), now)
    # Age-based trim: drop entries older than the TTL.
    if len(_recent_worker_exits) > _RECENT_WORKER_EXITS_MAX // 2:
        cutoff = now - _RECENT_WORKER_EXIT_TTL_SECONDS
        for _pid in [p for p, (_s, t) in _recent_worker_exits.items() if t < cutoff]:
            _recent_worker_exits.pop(_pid, None)
    # Size cap as a final guard.
    if len(_recent_worker_exits) > _RECENT_WORKER_EXITS_MAX:
        # Drop oldest half.
        ordered = sorted(_recent_worker_exits.items(), key=lambda kv: kv[1][1])
        for _pid, _ in ordered[: len(ordered) // 2]:
            _recent_worker_exits.pop(_pid, None)


def _classify_worker_exit(pid: int) -> "tuple[str, Optional[int]]":
    """Return ``(kind, code)`` for a reaped worker PID.

    ``clean_exit`` (status 0 — a still-``running`` task is a protocol
    violation: worker exited without ``kanban_complete``/``kanban_block``, so
    auto-block, retrying just loops); ``rate_limited`` (status
    ``KANBAN_RATE_LIMIT_EXIT_CODE`` — provider quota wall, released back to
    ``ready`` WITHOUT counting a failure so a long quota window can't trip the
    breaker); ``nonzero_exit`` (real error); ``signaled`` (OOM/SIGKILL, real
    crash; ``code`` is the signal); ``unknown`` (pid not in the reap registry
    — reaped elsewhere or died between reap tick and liveness check — fall
    back to the crashed-counter path; ``code`` is None).
    """
    entry = _recent_worker_exits.get(int(pid))
    if entry is None:
        return ("unknown", None)
    raw, _ = entry
    try:
        if os.WIFEXITED(raw):
            code = os.WEXITSTATUS(raw)
            if code == 0:
                return ("clean_exit", 0)
            if code == _kb.KANBAN_RATE_LIMIT_EXIT_CODE:
                return ("rate_limited", code)
            return ("nonzero_exit", code)
        if os.WIFSIGNALED(raw):
            return ("signaled", os.WTERMSIG(raw))
    except Exception:
        pass
    return ("unknown", None)


def reap_worker_zombies() -> "list[int]":
    """Reap all zombie children of this process without blocking.

    Returns the list of reaped PIDs. Safe to call when there are no
    children (returns []). No-op on Windows.
    """
    reaped: "list[int]" = []
    if os.name != "nt":
        try:
            while True:
                try:
                    pid, status = os.waitpid(-1, os.WNOHANG)
                except ChildProcessError:
                    break
                if pid == 0:
                    break
                _record_worker_exit(pid, status)
                reaped.append(pid)
        except Exception:
            pass
    return reaped


def _pid_alive(pid: Optional[int]) -> bool:
    """Return True if ``pid`` is still running on this host.

    Cross-platform: uses ``OpenProcess`` + ``WaitForSingleObject`` on
    Windows (via ``gateway.status._pid_exists``) and ``os.kill(pid, 0)``
    on POSIX. Returns False for falsy PIDs or on any OS error.

    **DO NOT** use ``os.kill(pid, 0)`` directly on Windows — Python's
    Windows ``os.kill`` treats ``sig=0`` as ``CTRL_C_EVENT`` (bpo-14484)
    and will broadcast it to the target's console group, potentially
    killing unrelated processes.

    **Zombie handling:** the existence check succeeds against zombie
    processes (post-exit, pre-reap) because the process table entry
    still exists. A worker that exits without being reaped by its
    parent would stay "alive" to the dispatcher forever. Dispatcher
    workers are started via ``start_new_session=True`` + intentional
    Popen handle abandonment, so init reaps them quickly — but during
    the window between exit and reap, we'd otherwise see stale "alive"
    signals. On Linux we peek at ``/proc/<pid>/status`` and treat
    ``State: Z`` as dead. On macOS we ask ``ps`` for the BSD ``stat``
    field and treat values containing ``Z`` as dead.
    """
    if not pid or pid <= 0:
        return False
    from gateway.status import _pid_exists
    if not _pid_exists(int(pid)):
        return False
    # Still here → process exists. Check for zombie on platforms
    # where we have a cheap, deterministic process-state probe.
    if sys.platform == "linux":
        try:
            with open(f"/proc/{int(pid)}/status", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("State:"):
                        # "State:\tZ (zombie)" → dead
                        if "Z" in line.split(":", 1)[1]:
                            return False
                        break
        except (FileNotFoundError, PermissionError, OSError):
            # proc entry gone → already reaped; treat as dead.
            # PermissionError shouldn't happen for our own children but
            # be defensive.
            pass
    elif sys.platform == "darwin":
        try:
            proc = subprocess.run(
                ["ps", "-o", "stat=", "-p", str(int(pid))],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True, encoding='utf-8', errors='replace',
                timeout=1,
                check=False,
            )
            if proc.returncode != 0:
                return False
            if "Z" in (proc.stdout or "").strip():
                return False
        except (OSError, subprocess.SubprocessError, TimeoutError):
            # If the secondary probe fails, keep the kill(0) answer.
            pass
    return True


def _terminate_reclaimed_worker(
    pid: Optional[int],
    claim_lock: Optional[str],
    *,
    signal_fn=None,
) -> dict[str, Any]:
    """Best-effort host-local worker termination for reclaim paths."""
    import signal

    info: dict[str, Any] = {
        "prev_pid": int(pid) if pid else None,
        "host_local": False,
        "termination_attempted": False,
        "terminated": False,
        "sigkill": False,
    }
    if not pid or pid <= 0 or not claim_lock:
        return info

    host_prefix = _kb._host_prefix()
    if not str(claim_lock).startswith(host_prefix):
        return info
    info["host_local"] = True

    kill = signal_fn if signal_fn is not None else (
        os.kill if hasattr(os, "kill") else None
    )
    if kill is None:
        return info

    info["termination_attempted"] = True
    try:
        kill(int(pid), signal.SIGTERM)
    except ProcessLookupError:
        # Process is already gone — that's a successful termination, not a
        # survival. Leaving terminated=False here would make the reclaim guard
        # misread a dead worker as still-alive and defer forever.
        info["terminated"] = True
        return info
    except OSError:
        return info

    for _ in range(10):
        if not _kb._pid_alive(pid):
            info["terminated"] = True
            return info
        time.sleep(0.5)

    if _kb._pid_alive(pid):
        try:
            # signal.SIGKILL doesn't exist on Windows; fall back to SIGTERM
            # (which maps to TerminateProcess via the stdlib shim).
            _sigkill = getattr(signal, "SIGKILL", signal.SIGTERM)
            kill(int(pid), _sigkill)
            info["sigkill"] = True
        except (ProcessLookupError, OSError):
            return info

    info["terminated"] = not _kb._pid_alive(pid)
    return info


def _worker_survived_termination(termination: dict) -> bool:
    """True when we tried to kill our own host-local worker and it is still alive.

    Reclaiming in this state would release the claim and let the dispatcher
    spawn a second worker while the first is still running — the duplication
    loop. Only host-local workers we actually signalled count: a non-local
    claim lock or a no-op attempt (no ``os.kill`` available) must fall through
    to the normal release path, since we cannot manage that worker anyway.
    """
    return bool(
        termination.get("termination_attempted")
        and termination.get("host_local")
        and not termination.get("terminated")
    )


def _defer_reclaim_for_live_worker(
    conn: sqlite3.Connection,
    task_id: str,
    claim_lock: Optional[str],
    now: int,
    termination: dict,
    *,
    reason: str,
) -> None:
    """Hold a claim whose worker survived termination instead of releasing it.

    Extends ``claim_expires`` by ``RECLAIM_DEFER_GRACE_SECONDS`` so the task
    stays ``running`` (no duplicate spawn) and records a ``reclaim_deferred``
    event so the hold is visible in ``hermes kanban tail``. The next dispatch
    tick retries the kill; this is self-correcting because not spawning a
    duplicate is what lets the throttled worker finally die.
    """
    grace = now + _kb.RECLAIM_DEFER_GRACE_SECONDS
    with _kb.write_txn(conn):
        cur = conn.execute(
            "UPDATE tasks SET claim_expires = ? "
            "WHERE id = ? AND status = 'running' AND claim_lock IS ?",
            (grace, task_id, claim_lock),
        )
        if cur.rowcount != 1:
            return
        run_id = _kb._current_run_id(conn, task_id)
        if run_id is not None:
            conn.execute(
                "UPDATE task_runs SET claim_expires = ? WHERE id = ?",
                (grace, run_id),
            )
        payload = {
            "reason": reason,
            "claim_lock": claim_lock,
            "claim_expires_now": grace,
        }
        payload.update(termination)
        _kb._append_event(conn, task_id, "reclaim_deferred", payload, run_id=run_id)


def heartbeat_worker(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    note: Optional[str] = None,
    expected_run_id: Optional[int] = None,
) -> bool:
    """Record a ``heartbeat`` event + touch ``last_heartbeat_at``.

    Called by long-running workers as a liveness signal orthogonal to
    the PID check. A worker that forks a long-lived child (train loop,
    video encode, web crawl) can have its Python still alive while the
    actual work process is stuck; periodic heartbeats catch that.

    Returns True on success, False if the task is not in a state that
    should be heartbeating (not running, or claim expired).
    """
    now = int(time.time())
    with _kb.write_txn(conn):
        if expected_run_id is None:
            cur = conn.execute(
                "UPDATE tasks SET last_heartbeat_at = ? "
                "WHERE id = ? AND status = 'running'",
                (now, task_id),
            )
        else:
            cur = conn.execute(
                "UPDATE tasks SET last_heartbeat_at = ? "
                "WHERE id = ? AND status = 'running' AND current_run_id = ?",
                (now, task_id, int(expected_run_id)),
            )
        if cur.rowcount != 1:
            return False
        run_id = (
            int(expected_run_id)
            if expected_run_id is not None
            else _kb._current_run_id(conn, task_id)
        )
        if run_id is not None:
            conn.execute(
                "UPDATE task_runs SET last_heartbeat_at = ? WHERE id = ?",
                (now, run_id),
            )
        _kb._append_event(
            conn, task_id, "heartbeat",
            {"note": note} if note else None,
            run_id=run_id,
        )
    return True


def enforce_max_runtime(
    conn: sqlite3.Connection,
    *,
    signal_fn=None,
) -> list[str]:
    """Terminate workers whose per-task ``max_runtime_seconds`` has elapsed.

    Sends SIGTERM, waits a short grace window, then SIGKILL. Emits a
    ``timed_out`` event and restores the task's source phase so the next
    dispatcher tick re-spawns the same kind of worker — unless the circuit
    breaker has already given up, in which case the task stays blocked
    where ``_record_spawn_failure`` parked it.

    Runs host-local: only tasks claimed by this host are candidates
    (same reasoning as ``detect_crashed_workers``). ``signal_fn`` is a
    test hook; defaults to ``os.kill`` on POSIX.
    """
    import signal
    timed_out: list[str] = []
    now = int(time.time())
    host_prefix = _kb._host_prefix()

    rows = conn.execute(
        "SELECT t.id, t.worker_pid, "
        "       COALESCE(r.started_at, t.started_at) AS active_started_at, "
        "       t.max_runtime_seconds, t.claim_lock "
        "FROM tasks t "
        "LEFT JOIN task_runs r ON r.id = t.current_run_id "
        "WHERE t.status = 'running' AND t.max_runtime_seconds IS NOT NULL "
        "  AND COALESCE(r.started_at, t.started_at) IS NOT NULL "
        "  AND t.worker_pid IS NOT NULL"
    ).fetchall()
    for row in rows:
        lock = row["claim_lock"] or ""
        if not lock.startswith(host_prefix):
            continue
        # Runtime is per attempt, not lifetime-of-task. ``tasks.started_at``
        # intentionally records the first time a task ever started, so retries
        # must be measured from the active task_runs row when present.
        elapsed = now - int(row["active_started_at"])
        if elapsed < int(row["max_runtime_seconds"]):
            continue

        pid = int(row["worker_pid"])
        tid = row["id"]
        # SIGTERM then SIGKILL. Keep it simple: 5 s grace. Workers that
        # want a cleaner shutdown can install their own SIGTERM handler
        # before the grace expires.
        killed = False
        kill = signal_fn if signal_fn is not None else (
            os.kill if hasattr(os, "kill") else None
        )
        if kill is not None:
            with contextlib.suppress(ProcessLookupError, OSError):
                kill(pid, signal.SIGTERM)
            # Short polling wait — no time.sleep on the write txn.
            for _ in range(10):
                if not _kb._pid_alive(pid):
                    break
                time.sleep(0.5)
            if _kb._pid_alive(pid):
                try:
                    # signal.SIGKILL doesn't exist on Windows.
                    _sigkill = getattr(signal, "SIGKILL", signal.SIGTERM)
                    kill(pid, _sigkill)
                    killed = True
                except (ProcessLookupError, OSError):
                    pass

        with _kb.write_txn(conn):
            retry_status = _kb._retry_status_for_run(conn, tid)
            cur = conn.execute(
                "UPDATE tasks SET status = ?, claim_lock = NULL, "
                "claim_expires = NULL, worker_pid = NULL, "
                "last_heartbeat_at = NULL "
                "WHERE id = ? AND status = 'running' "
                "  AND worker_pid = ? AND claim_lock IS ?",
                (retry_status, tid, pid, row["claim_lock"]),
            )
            if cur.rowcount == 1:
                payload = {
                    "pid": pid,
                    "elapsed_seconds": int(elapsed),
                    "limit_seconds": int(row["max_runtime_seconds"]),
                    "sigkill": killed,
                    "retry_status": retry_status,
                }
                run_id = _kb._end_run(
                    conn, tid,
                    outcome="timed_out", status="timed_out",
                    error=f"elapsed {int(elapsed)}s > limit {int(row['max_runtime_seconds'])}s",
                    metadata=payload,
                )
                _kb._append_event(
                    conn, tid, "timed_out", payload, run_id=run_id,
                )
                timed_out.append(tid)
        # Increment the unified failure counter. Outside the write_txn
        # above because ``_record_task_failure`` opens its own. If the
        # breaker trips, this flips the retried task to ``blocked`` and
        # emits a ``gave_up`` event on top of the ``timed_out`` we
        # already emitted.
        if cur.rowcount == 1:
            _kb._record_task_failure(
                conn, tid,
                error=f"elapsed {int(elapsed)}s > limit {int(row['max_runtime_seconds'])}s",
                outcome="timed_out",
                release_claim=False,
                end_run=False,
                event_payload_extra={
                    "pid": pid,
                    "sigkill": killed,
                    "retry_status": retry_status,
                },
            )
    return timed_out


# Heartbeat staleness heartbeat gap — if a running task hasn't sent a
# heartbeat in this many seconds it's considered inactive regardless of
# the ``dispatch_stale_timeout_seconds`` threshold.  Hardcoded at 1 hour
# to match the original spec (">4h started + no commits in 1h").
_STALE_HEARTBEAT_GAP_SECONDS = 3600


def detect_stale_running(
    conn: sqlite3.Connection,
    *,
    stale_timeout_seconds: int = 0,
    signal_fn=None,
) -> list[str]:
    """Reclaim ``running`` tasks with no heartbeat progress; returns their ids.

    Stale = running longer than ``stale_timeout_seconds`` (from the active
    run's ``started_at``, else ``tasks.started_at``) AND ``last_heartbeat_at``
    older than ``_STALE_HEARTBEAT_GAP_SECONDS`` or NULL. The task returns to
    its source phase, the run closes ``outcome='stale'`` and a live host-local
    worker is terminated. Blocked tasks are never candidates;
    ``stale_timeout_seconds=0`` disables the check. ``signal_fn`` is a test
    hook (default ``os.kill`` on POSIX).
    """
    if stale_timeout_seconds <= 0:
        return []


    now = int(time.time())
    reclaimed: list[str] = []

    rows = conn.execute(
        "SELECT t.id, t.worker_pid, t.last_heartbeat_at, t.claim_lock, "
        "       COALESCE(r.started_at, t.started_at) AS active_started_at "
        "FROM tasks t "
        "LEFT JOIN task_runs r ON r.id = t.current_run_id "
        "WHERE t.status = 'running'"
    ).fetchall()

    for row in rows:
        # Skip if no started_at (shouldn't happen for running, but be safe).
        if row["active_started_at"] is None:
            continue

        elapsed = now - int(row["active_started_at"])
        if elapsed < stale_timeout_seconds:
            continue  # not old enough to check

        last_hb = row["last_heartbeat_at"]
        hb_age = (now - int(last_hb)) if last_hb is not None else None
        if hb_age is not None and hb_age < _STALE_HEARTBEAT_GAP_SECONDS:
            continue  # recent heartbeat → still alive

        pid = row["worker_pid"]
        tid = row["id"]
        lock = row["claim_lock"] or ""

        # Terminate the worker if it's still host-local.
        termination = _kb._terminate_reclaimed_worker(
            pid, lock, signal_fn=signal_fn,
        )

        # Never release a claim while our own worker is still alive: that would
        # spawn a duplicate beside it. Hold the claim and retry next tick.
        if _worker_survived_termination(termination):
            _defer_reclaim_for_live_worker(
                conn, tid, lock, now, termination,
                reason="heartbeat_stale_worker_alive",
            )
            continue

        with _kb.write_txn(conn):
            retry_status = _kb._retry_status_for_run(conn, tid)
            cur = conn.execute(
                "UPDATE tasks SET status = ?, claim_lock = NULL, "
                "claim_expires = NULL, worker_pid = NULL, "
                "last_heartbeat_at = NULL "
                "WHERE id = ? AND status = 'running' "
                "  AND claim_lock IS ?",
                (retry_status, tid, row["claim_lock"]),
            )
            if cur.rowcount != 1:
                continue

            payload = {
                "elapsed_seconds": int(elapsed),
                "last_heartbeat_at": _kb._opt_int(last_hb),
                "heartbeat_age_seconds": _kb._opt_int(hb_age),
                "timeout_seconds": stale_timeout_seconds,
                "pid": int(pid) if pid else None,
                "retry_status": retry_status,
            }
            payload.update(termination)

            run_id = _kb._end_run(
                conn, tid,
                outcome="stale", status="stale",
                error=(
                    f"no heartbeat for {int(hb_age)}s "
                    if hb_age is not None
                    else "no heartbeat ever"
                ) + f" after {int(elapsed)}s running",
                metadata=payload,
            )
            _kb._append_event(
                conn, tid, "stale", payload, run_id=run_id,
            )
            reclaimed.append(tid)

        # Intentionally NOT calling _record_task_failure here. Stale reclaim
        # is dispatcher-side detection of an absent heartbeat; the task is
        # going straight back to its source phase for re-dispatch. Counting it as
        # a worker failure would let two legitimately-long-running tasks
        # (>4h without explicit heartbeat) trip the circuit breaker and
        # auto-block, even though no worker actually failed. The 'stale'
        # event already lives in task_events for auditability; that's the
        # right surface for "this happened" without conflating with the
        # spawn_failed / timed_out / crashed counters.

    return reclaimed


def reconcile_orphaned_running(
    conn: sqlite3.Connection,
) -> list[str]:
    """Requeue ``running`` cards with broken claim bookkeeping; returns their ids.

    A task can sit ``running`` with ``claim_lock``/``claim_expires`` NULL
    (crash mid-claim, manual SQL, DB restore) and no other recovery path
    touches it — ``release_stale_claims`` needs ``claim_expires``,
    ``detect_crashed_workers`` needs a host-local lock + pid,
    ``detect_stale_running`` is off by default — so it is a zombie forever.
    Orphans go back to ``ready`` with an explanatory comment, a leaked run is
    closed and a ``reconciled`` event appended; a row still recording a live
    host-local PID is deferred to a later tick so no duplicate is spawned
    beside a possibly-alive worker. Safe to call every tick.
    """
    now = int(time.time())
    reconciled: list[str] = []
    rows = conn.execute(
        "SELECT id, claim_lock, claim_expires, worker_pid FROM tasks "
        "WHERE status = 'running' "
        "  AND (claim_lock IS NULL OR claim_expires IS NULL)"
    ).fetchall()
    for row in rows:
        tid = row["id"]
        pid = row["worker_pid"]
        if pid and _kb._pid_alive(pid):
            # The recorded worker may still be doing real work — never
            # requeue beside a live process. Retry next tick.
            _kb._log.debug(
                "kanban reconcile: task %s has broken claim bookkeeping but "
                "pid %s is alive on this host — deferring", tid, pid,
            )
            continue
        with _kb.write_txn(conn):
            cur = conn.execute(
                "UPDATE tasks SET status = 'ready', claim_lock = NULL, "
                "claim_expires = NULL, worker_pid = NULL, "
                "last_heartbeat_at = NULL "
                "WHERE id = ? AND status = 'running' "
                "  AND claim_lock IS ? AND claim_expires IS ?",
                (tid, row["claim_lock"], row["claim_expires"]),
            )
            if cur.rowcount != 1:
                continue
            payload = {
                "reason": "orphaned_running",
                "claim_lock": row["claim_lock"],
                "claim_expires": _kb._opt_int(row["claim_expires"]),
                "worker_pid": int(pid) if pid else None,
                "now": now,
            }
            run_id = _kb._end_run(
                conn, tid,
                outcome="reclaimed", status="reclaimed",
                error="orphaned running card (broken claim bookkeeping)",
                metadata=payload,
            )
            _kb._insert_comment(
                conn, tid, "dispatcher",
                "reconciliation: card was 'running' with no valid claim "
                "(dead/gone worker) — requeued to ready",
                now,
            )
            _kb._append_event(conn, tid, "reconciled", payload, run_id=run_id)
            reconciled.append(tid)
        _kb._log.info(
            "kanban reconcile: requeued orphaned running task %s "
            "(claim_lock=%r, worker_pid=%r)", tid, row["claim_lock"], pid,
        )
    return reconciled


def _error_fingerprint(error_text: str) -> str:
    """Normalize an error message for grouping identical failures.

    Strips host-specific details (PIDs, timestamps) so that errors
    with the same root cause produce the same fingerprint.
    """
    fp = re.sub(r'\bpid \d+\b', 'pid N', error_text[:80])
    fp = re.sub(r'\b\d{10,}\b', '<TS>', fp)
    return fp.lower().strip()


# ~96% of "clean exit without a terminal tool call" tasks complete on a later
# run, so a protocol violation is NOT deterministic — bounded retry before the
# breaker trips. The budget is a violation-only STREAK
# (``_protocol_violation_streak``), independent of ``consecutive_failures``:
# other failure kinds neither consume nor extend it, and a below-budget
# violation doesn't tick the unified counter. Per-task ``max_retries``
# overrides it, as for every other failure kind.
_PROTOCOL_VIOLATION_FAILURE_LIMIT = 3


# Closed runs to walk when counting the streak; the streak trips at a handful,
# so anything past a few dozen rows means "way past the bound" anyway.
_PROTOCOL_VIOLATION_SCAN_LIMIT = 50


def _protocol_violation_streak(conn: sqlite3.Connection, task_id: str) -> int:
    """Count the task's trailing run of clean-exit protocol violations.

    Walks the task's closed runs newest-first — including the violation run
    ``detect_crashed_workers`` just closed — and counts how many in a row were
    clean-exit protocol violations:

    * ``rate_limited`` runs are neutral and skipped: a quota wall says nothing
      about the task, exactly as it is neutral for the unified
      ``consecutive_failures`` counter.
    * Any other closed run (completed, plain crash, timeout, spawn failure,
      reclaim, …) breaks the streak, so the bounded retry budget counts ONLY
      protocol violations — mixed failure kinds can neither consume nor
      extend it.

    Violation runs are recognized by the ``protocol_violation`` marker that
    ``detect_crashed_workers`` stamps into the run metadata; the violation
    error text is matched as a fallback for runs recorded before the marker
    existed.
    """
    streak = 0
    rows = conn.execute(
        "SELECT outcome, error, metadata FROM task_runs "
        "WHERE task_id = ? AND ended_at IS NOT NULL "
        "ORDER BY id DESC LIMIT ?",
        (task_id, _PROTOCOL_VIOLATION_SCAN_LIMIT),
    ).fetchall()
    for row in rows:
        outcome = row["outcome"] or ""
        if outcome == "rate_limited":
            continue
        if outcome == "crashed":
            is_violation = bool(_kb._json_dict(row["metadata"]).get("protocol_violation"))
            if not is_violation:
                is_violation = "protocol violation" in (row["error"] or "")
            if is_violation:
                streak += 1
                continue
        break
    return streak


def detect_crashed_workers(conn: sqlite3.Connection) -> list[str]:
    """Reclaim ``running`` tasks whose worker PID is no longer alive.

    Appends a ``crashed`` event and restores the task's source phase.
    Different from ``release_stale_claims``: this checks liveness
    immediately rather than waiting for the claim TTL.

    Only tasks claimed by *this host* are considered — PIDs from other hosts
    are meaningless, and ``_default_spawn`` always runs workers on the
    dispatcher's host.

    A clean exit (rc=0) with the task still ``running`` is a protocol
    violation (worker answered conversationally without ``kanban_complete`` /
    ``kanban_block``); it gets a bounded violation-only retry budget before
    the breaker trips (see ``_protocol_violation_streak``).

    An exit with ``KANBAN_RATE_LIMIT_EXIT_CODE`` is a provider quota wall,
    NOT a task failure: released to its source phase WITHOUT counting a
    failure (a long quota window must not trip the breaker) and stamped with
    a quota-blocker error so ``check_respawn_guard`` defers the respawn. Those
    ids surface via the ``_last_rate_limited`` function attribute; the public
    return stays the crashed-only ``list[str]``.
    """
    crashed: list[str] = []
    rate_limited: list[str] = []
    # Per-crash details collected inside the main txn, used after it
    # closes to run ``_record_task_failure`` (which needs its own
    # write_txn so can't nest). ``protocol_violation`` flags the
    # clean-exit-but-still-running case, which is accounted against its
    # own bounded violation streak instead of the unified failure
    # counter (see the post-txn loop below).
    crash_details: list[tuple[str, int, str, bool, str]] = []
    # (task_id, pid, claimer, protocol_violation, error_text)
    # Worker-exit observer payloads (RFC #58548), collected inside the main
    # txn and fired only after every reclaim/accounting txn has committed.
    exited_hook_payloads: list[dict] = []
    with _kb.write_txn(conn):
        rows = conn.execute(
            "SELECT id, worker_pid, claim_lock, started_at, assignee "
            "FROM tasks "
            "WHERE status = 'running' AND worker_pid IS NOT NULL"
        ).fetchall()
        host_prefix = _kb._host_prefix()
        for row in rows:
            # Only check liveness for claims owned by this host.
            lock = row["claim_lock"] or ""
            if not lock.startswith(host_prefix):
                continue
            # Skip liveness check inside the launch-window grace period
            # so a freshly-spawned worker isn't reclaimed before its PID
            # is visible on /proc.
            started_at = _kb._row_get(row, "started_at")
            if started_at is not None:
                grace = _kb._resolve_crash_grace_seconds()
                if time.time() - started_at < grace:
                    continue
            if _kb._pid_alive(row["worker_pid"]):
                continue

            pid = int(row["worker_pid"])
            kind, code = _kb._classify_worker_exit(pid)
            rate_limited_exit = False
            if kind == "clean_exit":
                # rc=0 with the task still ``running``: exited without
                # ``kanban_complete`` / ``kanban_block``. Usually the work
                # succeeded and only the paperwork was skipped; the corrective
                # sentence reaches the retry worker via ``build_worker_context``.
                protocol_violation = True
                error_text = (
                    "worker exited cleanly (rc=0) without calling "
                    "kanban_complete or kanban_block — protocol violation. "
                    "If the prior run already did the work, verify it and "
                    "report the result via kanban_complete; a run that ends "
                    "without a terminal kanban call counts as failed no "
                    "matter what it did."
                )
                event_kind = "protocol_violation"
                event_payload = {
                    "pid": pid,
                    "claimer": row["claim_lock"],
                    "exit_code": code,
                    # Durable marker for _protocol_violation_streak: _end_run
                    # copies this payload into the run metadata, which is how
                    # the violation-only retry budget is derived later.
                    "protocol_violation": True,
                }
            elif kind == "rate_limited":
                # EX_TEMPFAIL quota wall — NOT a task failure. Release to the
                # source phase (respawn guard defers it) and do NOT count a
                # failure so a long quota window can't trip the breaker.
                protocol_violation = False
                rate_limited_exit = True
                error_text = (
                    f"pid {pid} exited rate-limited (quota wall) — "
                    f"requeued without counting a failure"
                )
                event_kind = "rate_limited"
                event_payload = {
                    "pid": pid,
                    "claimer": row["claim_lock"],
                    "exit_code": code,
                }
            else:
                protocol_violation = False
                if kind == "nonzero_exit":
                    error_text = f"pid {pid} exited with code {code}"
                elif kind == "signaled":
                    error_text = f"pid {pid} killed by signal {code}"
                else:
                    error_text = f"pid {pid} not alive"
                event_kind = "crashed"
                event_payload = {"pid": pid, "claimer": row["claim_lock"]}
                if code is not None and kind != "unknown":
                    event_payload["exit_kind"] = kind
                    event_payload["exit_code"] = code

            retry_status = _kb._retry_status_for_run(conn, row["id"])
            event_payload["retry_status"] = retry_status
            cur = conn.execute(
                "UPDATE tasks SET status = ?, claim_lock = NULL, "
                "claim_expires = NULL, worker_pid = NULL "
                "WHERE id = ? AND status = 'running' "
                "  AND worker_pid = ? AND claim_lock IS ?",
                (retry_status, row["id"], pid, row["claim_lock"]),
            )
            if cur.rowcount == 1:
                # Rate-limited requeues are a clean release, not a crash —
                # record the run outcome as ``rate_limited`` so the board
                # history doesn't show a phantom crash for a quota wall.
                _run_outcome = "rate_limited" if rate_limited_exit else "crashed"
                run_id = _kb._end_run(
                    conn, row["id"],
                    outcome=_run_outcome, status=_run_outcome,
                    error=error_text,
                    metadata=dict(event_payload),
                )
                _kb._append_event(
                    conn, row["id"], event_kind,
                    event_payload,
                    run_id=run_id,
                )
                exited_hook_payloads.append({
                    "task_id": row["id"],
                    "assignee": row["assignee"],
                    "run_id": run_id,
                    "worker_pid": pid,
                    "exit_kind": kind,
                    "exit_code": code,
                    "outcome": _run_outcome,
                    "retry_status": retry_status,
                })
                if rate_limited_exit:
                    # Stamp last_failure_error so ``check_respawn_guard`` sees a
                    # quota blocker — WITHOUT touching ``consecutive_failures``.
                    conn.execute(
                        "UPDATE tasks SET last_failure_error = ? WHERE id = ?",
                        (error_text[:500], row["id"]),
                    )
                    rate_limited.append(row["id"])
                else:
                    if protocol_violation:
                        # A below-budget violation never reaches
                        # ``_record_task_failure`` (which stamps this column
                        # for every other kind), yet the board UI and the retry
                        # worker's context need the corrective message.
                        conn.execute(
                            "UPDATE tasks SET last_failure_error = ? "
                            "WHERE id = ?",
                            (error_text[:500], row["id"]),
                        )
                    crashed.append(row["id"])
                    crash_details.append(
                        (row["id"], pid, row["claim_lock"],
                         protocol_violation, error_text)
                    )
    # Outside the main txn: account each crash and maybe trip the breaker
    # (blocked + ``gave_up`` on top of the event already emitted).
    # Protocol violations get a BOUNDED violation-only retry budget
    # (``_protocol_violation_streak``; see ``_PROTOCOL_VIOLATION_FAILURE_LIMIT``
    # for why) — independent of ``consecutive_failures``, with per-task
    # ``max_retries`` taking top precedence. Systemic same-error crashes still
    # trip immediately.
    auto_blocked: list[str] = []
    if crash_details:
        # Fingerprint errors to detect systemic failures.
        _fp_counts: dict[str, int] = {}
        for _, _, _, _, err_text in crash_details:
            fp = _error_fingerprint(err_text)
            _fp_counts[fp] = _fp_counts.get(fp, 0) + 1
        for tid, pid, claimer, protocol_violation, error_text in crash_details:
            if protocol_violation:
                streak = _protocol_violation_streak(conn, tid)
                trow = conn.execute(
                    "SELECT max_retries FROM tasks WHERE id = ?", (tid,),
                ).fetchone()
                if trow is None:
                    continue  # task deleted mid-loop
                task_override = _kb._row_get(trow, "max_retries")
                violation_limit = (
                    int(task_override)
                    if task_override is not None
                    else _PROTOCOL_VIOLATION_FAILURE_LIMIT
                )
                if streak < violation_limit:
                    # Below budget: already back at ``ready`` with
                    # ``last_failure_error`` stamped. Deliberately no
                    # ``_record_task_failure`` — must not consume the unified
                    # failure budget.
                    continue
                # ``force_trip``: the decision (incl. per-task ``max_retries``)
                # was already made against the violation streak above.
                tripped = _kb._record_task_failure(
                    conn, tid,
                    error=error_text,
                    outcome="crashed",
                    failure_limit=violation_limit,
                    force_trip=True,
                    release_claim=False,
                    end_run=False,
                    event_payload_extra={
                        "pid": pid,
                        "claimer": claimer,
                        "protocol_violations": streak,
                        "protocol_violation_limit": violation_limit,
                    },
                )
                if tripped:
                    auto_blocked.append(tid)
                continue
            fp = _error_fingerprint(error_text)
            is_systemic = _fp_counts.get(fp, 0) >= 3
            tripped = _kb._record_task_failure(
                conn, tid,
                error=error_text,
                outcome="crashed",
                failure_limit=1 if is_systemic else None,
                release_claim=False,
                end_run=False,
                event_payload_extra={"pid": pid, "claimer": claimer},
            )
            if tripped:
                auto_blocked.append(tid)
    # Stash auto-blocked ids on the function for the dispatch loop to pick up.
    # Keeps the public return type (``list[str]``) stable for direct callers
    # and tests that destructure the result; ``dispatch_once`` reads this
    # side-channel attribute to populate ``DispatchResult.auto_blocked``.
    detect_crashed_workers._last_auto_blocked = auto_blocked  # type: ignore[attr-defined]
    # Same side-channel for rate-limited requeues — these did NOT count a
    # failure and are NOT crashes, so they stay out of the ``crashed`` return.
    detect_crashed_workers._last_rate_limited = rate_limited  # type: ignore[attr-defined]
    # Worker-lifecycle observer (RFC #58548): exit events are tick-derived
    # from this reclaim pass — fired only now, after the main reclaim txn
    # AND the breaker accounting above have committed, so subscribers always
    # observe fully durable board state.
    if exited_hook_payloads and _kb._kanban_observer_consumed("on_kanban_worker_exited"):
        _board = _kb.get_current_board()
        for hook_fields in exited_hook_payloads:
            hook_fields = dict(hook_fields)
            _kb._fire_kanban_lifecycle_hook(
                "on_kanban_worker_exited",
                hook_fields.pop("task_id"),
                board=_board,
                **hook_fields,
            )
    return crashed


def _record_task_failure(
    conn: sqlite3.Connection,
    task_id: str,
    error: str,
    *,
    outcome: str,
    failure_limit: int = None,
    force_trip: bool = False,
    release_claim: bool = False,
    end_run: bool = False,
    event_payload_extra: Optional[dict] = None,
) -> bool:
    """Record a non-success outcome (spawn_failed / crashed / timed_out)
    and maybe trip the circuit breaker.

    Every non-success path funnels through here so ``consecutive_failures``
    and the auto-block threshold stay consistent. Returns True when the task
    was auto-blocked, False when just updated in place.

    Modes: ``release_claim=True, end_run=True`` is the spawn-failure path
    (task still running with an open run: restore its source phase — or
    ``blocked`` on trip — release the claim, close the run with
    ``outcome=<outcome>``). ``release_claim=False, end_run=False`` is the
    timeout/crash path (caller ALREADY restored the phase and closed the run;
    only the counter moves, and a trip re-transitions to ``blocked`` with a
    ``gave_up`` event). ``event_payload_extra`` merges into that payload.

    Effective threshold: per-task ``max_retries`` (nothing overrides it), then
    the caller's ``failure_limit`` (``kanban.failure_limit``), then
    ``DEFAULT_FAILURE_LIMIT``. ``force_trip=True`` trips unconditionally —
    the caller already applied its own bounded-retry policy (the
    protocol-violation streak in ``detect_crashed_workers``); the order above
    is then only reported in the payload. The failure is still counted.
    """
    if failure_limit is None:
        failure_limit = DEFAULT_FAILURE_LIMIT
    blocked = False
    with _kb.write_txn(conn):
        row = conn.execute(
            "SELECT consecutive_failures, status, max_retries, current_run_id "
            "FROM tasks WHERE id = ?", (task_id,),
        ).fetchone()
        if row is None:
            return False
        retry_status = (
            _kb._retry_status_for_run(conn, task_id, row["current_run_id"])
            if release_claim
            else ("review" if row["status"] == "review" else "ready")
        )
        failures = int(row["consecutive_failures"]) + 1

        # Per-task override wins over both caller-supplied and default
        # thresholds. None (the common case) falls through.
        task_override = _kb._row_get(row, "max_retries")
        if task_override is not None:
            effective_limit = int(task_override)
            limit_source = "task"
        else:
            effective_limit = int(failure_limit)
            limit_source = "dispatcher"

        if force_trip or failures >= effective_limit:
            # Trip the breaker. Spawn path (release_claim) is still running
            # and also clears claim state; the timeout/crash path already
            # restored the source phase with the claim cleared.
            conn.execute(
                "UPDATE tasks SET status = 'blocked', "
                + ("claim_lock = NULL, claim_expires = NULL, worker_pid = NULL, "
                   if release_claim else "")
                + "consecutive_failures = ?, last_failure_error = ? "
                "WHERE id = ? AND status IN ('running', 'ready', 'review')",
                (failures, error[:500], task_id),
            )
            run_id = None
            if end_run:
                # Only the spawn path has an open run to close.
                run_id = _kb._end_run(
                    conn, task_id,
                    outcome="gave_up", status="gave_up",
                    error=error[:500],
                    metadata={
                        "failures": failures,
                        "trigger_outcome": outcome,
                        "effective_limit": effective_limit,
                        "limit_source": limit_source,
                        "retry_status": retry_status,
                    },
                )
            payload = {
                "failures": failures,
                "effective_limit": effective_limit,
                "limit_source": limit_source,
                "error": error[:500],
                "trigger_outcome": outcome,
                "retry_status": retry_status,
            }
            if event_payload_extra:
                payload.update(event_payload_extra)
            _kb._append_event(
                conn, task_id, "gave_up", payload, run_id=run_id,
            )
            blocked = True
        else:
            # Below threshold.
            if release_claim:
                # Spawn path: restore the claimed source phase + clear claim.
                conn.execute(
                    "UPDATE tasks SET status = ?, claim_lock = NULL, "
                    "claim_expires = NULL, worker_pid = NULL, "
                    "consecutive_failures = ?, last_failure_error = ? "
                    "WHERE id = ? AND status = 'running'",
                    (retry_status, failures, error[:500], task_id),
                )
            else:
                # Timeout/crash path: caller already restored the source phase.
                conn.execute(
                    "UPDATE tasks SET consecutive_failures = ?, "
                    "last_failure_error = ? WHERE id = ?",
                    (failures, error[:500], task_id),
                )
            if end_run:
                # Spawn path: close the open run with outcome.
                run_id = _kb._end_run(
                    conn, task_id,
                    outcome=outcome, status=outcome,
                    error=error[:500],
                    metadata={
                        "failures": failures,
                        "retry_status": retry_status,
                    },
                )
                _kb._append_event(
                    conn, task_id, outcome,
                    {
                        "error": error[:500],
                        "failures": failures,
                        "retry_status": retry_status,
                    },
                    run_id=run_id,
                )
            # Timeout/crash path's caller already emitted its own event.
    return blocked


# Backward-compat alias. Old name is referenced from tests and possibly
# third-party callers. New code should call ``_record_task_failure``.
def _record_spawn_failure(
    conn: sqlite3.Connection,
    task_id: str,
    error: str,
    *,
    failure_limit: int = None,
) -> bool:
    return _kb._record_task_failure(
        conn, task_id, error,
        outcome="spawn_failed",
        failure_limit=failure_limit,
        release_claim=True,
        end_run=True,
    )


def _set_worker_pid(conn: sqlite3.Connection, task_id: str, pid: int) -> None:
    """Record the spawned child's pid + emit a ``spawned`` event.

    The event's payload carries the pid so a human reading ``hermes kanban
    tail`` can correlate log lines with OS-level traces without opening
    the drawer.
    """
    with _kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET worker_pid = ? WHERE id = ?",
            (int(pid), task_id),
        )
        run_id = _kb._current_run_id(conn, task_id)
        if run_id is not None:
            conn.execute(
                "UPDATE task_runs SET worker_pid = ? WHERE id = ?",
                (int(pid), run_id),
            )
        _kb._append_event(conn, task_id, "spawned", {"pid": int(pid)}, run_id=run_id)


def _clear_failure_counter(conn: sqlite3.Connection, task_id: str) -> None:
    """Reset the unified consecutive-failures counter.

    Called from ``complete_task`` on successful completion — a fresh
    success means the task + profile combination is working and any
    past failures are history. NOT called on spawn success anymore:
    a successful spawn proves the worker could start but says nothing
    about whether the run will succeed, so we need to let timeouts and
    crashes accumulate across spawn boundaries.
    """
    with _kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET consecutive_failures = 0, "
            "last_failure_error = NULL WHERE id = ?",
            (task_id,),
        )


# Legacy alias for test-code and anything else that still imports it.
_clear_spawn_failures = _clear_failure_counter


def check_respawn_guard(
    conn: sqlite3.Connection, task_id: str, *, lane: str = "ready",
) -> Optional[str]:
    """Return a guard reason if ``task_id`` should NOT be re-spawned, else None.

    Called per ready/review task in ``dispatch_once`` before any claim attempt.
    Returning a reason defers the spawn this tick; the task stays in its
    source phase and gets another chance on the next dispatcher tick.

    ``lane`` is the dispatch column (``"ready"`` / ``"review"``). The review
    lane skips ``recent_success`` and ``active_pr``: a recent PR comment (and
    often a completed run) is the *precondition* of a review handoff, not a
    duplicate-work signal. Cooldown and auth-blocker apply in every lane.

    Checks in priority order:

    ``"rate_limit_cooldown"``
        Latest run ended ``rate_limited`` (EX_TEMPFAIL quota wall) within
        ``_resolve_rate_limit_cooldown_seconds()``; defer until it elapses,
        then allow a cheap probe. Checked BEFORE ``blocker_auth`` because the
        rate-limit requeue stamps a quota-flavored ``last_failure_error`` that
        would otherwise match the auth regex and park the task forever (that
        path never increments ``consecutive_failures``, so the breaker can't
        free it).

    ``"blocker_auth"``
        ``last_failure_error`` matches a quota / auth pattern; retrying now
        won't help. ``consecutive_failures`` still trips the breaker after
        ``failure_limit``, so a persistent auth error eventually blocks while
        a transient 429 gets a few ticks of recovery.

    ``"recent_success"``
        A completed run within ``_RESPAWN_GUARD_SUCCESS_WINDOW``; wait for an
        explicit re-queue. Bypassed when a re-queue event (status, promote,
        unblock, reclaim) arrives AFTER that completion — a deliberate re-run.

    ``"active_pr"``
        A GitHub PR URL in a comment within ``_RESPAWN_GUARD_PR_WINDOW``;
        re-spawning risks a duplicate PR.

    Stale / dead claim locks are NOT a guard reason — ``release_stale_claims``
    and ``detect_crashed_workers`` reset those only after verifying the lock
    is genuinely dead (no live PID on this host).
    """
    row = conn.execute(
        "SELECT last_failure_error FROM tasks WHERE id = ?",
        (task_id,),
    ).fetchone()
    if row is None:
        return None

    now = int(time.time())

    # 1. Rate-limit cooldown. The most recent run ended ``rate_limited``
    #    (quota wall) — defer while inside the cooldown window, then allow a
    #    cheap probe. Must run BEFORE the blocker_auth regex check, because a
    #    rate-limit requeue stamps a quota-flavored last_failure_error that
    #    the regex would otherwise match → defer forever (no failure counter
    #    increment on this path means the breaker can never free it).
    #
    #    We look at the LATEST run only (ORDER BY ended_at DESC LIMIT 1): if a
    #    newer crash/completion superseded the rate-limit run, this guard
    #    no longer applies and the normal paths take over.
    rl_cooldown = _kb._resolve_rate_limit_cooldown_seconds()
    latest_run = conn.execute(
        "SELECT outcome, ended_at FROM task_runs "
        "WHERE task_id = ? AND ended_at IS NOT NULL "
        "ORDER BY ended_at DESC LIMIT 1",
        (task_id,),
    ).fetchone()
    if (
        latest_run is not None
        and latest_run["outcome"] == "rate_limited"
    ):
        if rl_cooldown <= 0:
            # Cooldown disabled — respawn immediately, and skip the
            # blocker_auth regex so the stamped rate-limit text doesn't
            # re-trap the task.
            return None
        ended_at = latest_run["ended_at"]
        if ended_at is not None and (now - int(ended_at)) < rl_cooldown:
            return "rate_limit_cooldown"
        # Cooldown elapsed — allow the respawn. Return early so the
        # blocker_auth check below doesn't catch the rate-limit text we
        # stamped on the task; this path intentionally retries forever
        # (cheaply, spaced by the cooldown) until quota returns or a real
        # crash/completion supersedes it.
        return None

    # 2. Quota / auth blocker: retrying immediately will not help.
    err = row["last_failure_error"]
    if err and _RESPAWN_BLOCKER_RE.search(err):
        return "blocker_auth"

    # Review-lane spawns stop here: a recent completed run and a fresh PR
    # URL comment are the canonical *inputs* to a review handoff (worker
    # opened a PR, then requested review), not signals of duplicate work.
    if lane == "review":
        return None

    # 3. Completed run within guard window — proof of recent success.
    #    Exception: an explicit re-queue AFTER that success (an operator
    #    dragging done→ready, a dependency re-promotion, an unblock, a
    #    reclaim) is a deliberate "run it again" — honor it instead of
    #    deferring. Without this, a manual done→ready just sits there,
    #    silently held by the guard, until the window elapses.
    cutoff = now - _RESPAWN_GUARD_SUCCESS_WINDOW
    recent_completed = conn.execute(
        "SELECT ended_at FROM task_runs "
        "WHERE task_id = ? AND outcome = 'completed' AND ended_at >= ? "
        "ORDER BY ended_at DESC LIMIT 1",
        (task_id, cutoff),
    ).fetchone()
    if recent_completed:
        completed_at = int(recent_completed["ended_at"] or 0)
        requeued_after = conn.execute(
            "SELECT 1 FROM task_events "
            "WHERE task_id = ? AND created_at >= ? "
            "AND kind IN ('status', 'promoted', 'unblocked', 'reclaimed') "
            "LIMIT 1",
            (task_id, completed_at),
        ).fetchone()
        if not requeued_after:
            return "recent_success"

    # 4. GitHub PR URL in a recent comment — prior worker already opened a PR.
    pr_cutoff = now - _RESPAWN_GUARD_PR_WINDOW
    for c in conn.execute(
        "SELECT body FROM task_comments WHERE task_id = ? AND created_at >= ?",
        (task_id, pr_cutoff),
    ).fetchall():
        if c["body"] and _RESPAWN_GUARD_PR_URL_RE.search(c["body"]):
            return "active_pr"

    return None


def _has_spawnable(conn: sqlite3.Connection, status: str) -> bool:
    rows = conn.execute(
        "SELECT DISTINCT assignee FROM tasks "
        "WHERE status = ? AND assignee IS NOT NULL AND claim_lock IS NULL",
        (status,),
    ).fetchall()
    if not rows:
        return False
    try:
        from hermes_cli.profiles import profile_exists  # local import: avoids cycle
    except Exception:
        # Can't introspect — assume spawnable, preserve legacy behavior.
        return True
    return any(profile_exists(row["assignee"]) for row in rows)


def has_spawnable_ready(conn: sqlite3.Connection) -> bool:
    """True iff a ready+assigned+unclaimed task maps to a real Hermes profile.

    Health telemetry uses it to tell "stuck" (``0 spawned`` with spawnable
    work) from "correctly idle" (only control-plane lanes waiting on terminals
    that pull via ``claim_task``). Falls back to "any assigned" when
    ``profile_exists`` is unimportable (partial install) so the warning still
    fires when degraded.
    """
    return _has_spawnable(conn, "ready")


def has_spawnable_review(conn: sqlite3.Connection) -> bool:
    """:func:`has_spawnable_ready` for the review column."""
    return _has_spawnable(conn, "review")


def review_dispatch_enabled() -> bool:
    """Return whether first-class review tasks should dispatch automatically.

    The default is true because Hermes ships the ``sdlc-review`` skill and the
    review lifecycle includes a supported reviewer-owned changes-requested
    transition. Operators can disable it for human-only review boards.
    """
    try:
        from hermes_cli.config import load_config
        return bool(
            (load_config() or {}).get("kanban", {}).get("review_dispatch", True)
        )
    except Exception:
        return True


# ---------------------------------------------------------------------------
# Memory-aware dispatch guard
#
# With no ``kanban.max_in_progress`` on a busy board the dispatcher fanned out
# 26-31 workers on a 1 GiB VM and OOM'd the host. Two safeguards: a
# memory-DERIVED default cap when the operator never set one
# (``resolve_max_in_progress``, sized from MemTotal), and a live
# memory-PRESSURE guard inside the tick (``_memory_pressure_level``) because
# a static cap can't see other tenants of the box. Both fail open: non-Linux
# or any read error → empty sample → no cap / "unknown" (no restriction).
# ---------------------------------------------------------------------------

# Assumed per-worker memory footprint for the derived default cap. Hermes
# workers are full agent processes (Python + model client + tool subprocesses);
# ~512 MiB is a deliberately conservative planning number so the derived cap
# errs toward fewer workers on small VMs.
MEMORY_GUARD_MB_PER_WORKER = 512


# Bounds for the derived default: never below 2 (a board must still make
# progress on the smallest hosted VM) and never above 8 (operators who want
# more fan-out on big iron should say so explicitly in config).
DERIVED_MAX_IN_PROGRESS_FLOOR = 2


DERIVED_MAX_IN_PROGRESS_CEILING = 8


def _system_memory_sample() -> dict:
    """Best-effort system memory snapshot (KiB values), ``{}`` when unknown.

    Delegates to :func:`gateway.lifecycle_ledger.sample_memory` (pure /proc
    reads, Linux-only, never raises). Local import keeps ``kanban_db``
    importable in stripped-down environments without the gateway package.
    Module-level indirection is also the test seam — the shared conftest
    patches this to ``{}`` so suite results don't depend on the CI runner's
    live memory state.
    """
    try:
        from gateway.lifecycle_ledger import sample_memory
        return sample_memory() or {}
    except Exception:
        return {}


def derive_default_max_in_progress(sample: Optional[Mapping[str, Any]] = None) -> Optional[int]:
    """Memory-derived default for ``kanban.max_in_progress`` when unset.

    ``clamp(MemTotal / MEMORY_GUARD_MB_PER_WORKER, FLOOR, CEILING)`` — e.g.
    a 1 GiB VM derives 2, a 4 GiB VM derives 8. Returns ``None`` (no cap,
    pre-fix behaviour) when total memory can't be determined, so dev
    machines on macOS/Windows are unaffected.
    """
    if sample is None:
        sample = _kb._system_memory_sample()
    total_kib = sample.get("mem_total_kib")
    if isinstance(total_kib, bool) or not isinstance(total_kib, int) or total_kib <= 0:
        return None
    workers = (total_kib // 1024) // MEMORY_GUARD_MB_PER_WORKER
    return max(
        DERIVED_MAX_IN_PROGRESS_FLOOR,
        min(workers, DERIVED_MAX_IN_PROGRESS_CEILING),
    )


def resolve_max_in_progress(configured: Optional[int]) -> Optional[int]:
    """Return the effective global concurrency cap for a dispatch tick.

    An explicit operator-configured value always wins. When unset, fall back
    to the memory-derived default (see :func:`derive_default_max_in_progress`).
    Callers that parse config (gateway dispatcher, ``hermes kanban dispatch``)
    should route through this so both paths agree.
    """
    if configured is not None:
        return configured
    return _kb.derive_default_max_in_progress()


def configured_max_in_progress() -> Optional[int]:
    """Read ``kanban.max_in_progress`` from config, or None when unset/invalid.

    Small shared parser so every dispatch entry point (gateway watcher, CLI
    dispatch, standalone daemon) agrees on what "explicitly configured"
    means: a positive integer wins, anything else falls through to the
    memory-derived default via :func:`resolve_max_in_progress`.
    """
    try:
        from hermes_cli.config import load_config_readonly
        raw = (load_config_readonly() or {}).get("kanban", {}).get(
            "max_in_progress"
        )
    except Exception:
        return None
    if raw is None:
        return None
    try:
        ival = int(raw)
    except (TypeError, ValueError):
        return None
    return ival if ival >= 1 else None


def count_running_tasks(conn: sqlite3.Connection) -> int:
    """Return the number of tasks currently in ``status='running'``.

    Used by the gateway's multi-board sweep to account for workers on
    OTHER boards against the host-level concurrency budget (OOF-30): the
    memory-derived cap bounds the machine, so each board's tick must see
    the machine's total, not just its own. Fails open to 0 — a broken
    board must not brick dispatch on healthy ones (corruption is handled
    separately by the watcher's quarantine logic).
    """
    try:
        return int(
            conn.execute(
                "SELECT COUNT(*) FROM tasks WHERE status = 'running'"
            ).fetchone()[0]
        )
    except Exception:
        return 0


def count_running_tasks_other_boards(board: Optional[str] = None) -> int:
    """Total ``running`` tasks across every board EXCEPT ``board``.

    The concurrency caps bound the HOST (workers are OS processes sharing
    one machine's memory), but each board's dispatch tick only sees its own
    DB. Without this, a memory-derived cap of N gets multiplied by the
    number of active boards — reproduced in review of OOF-30: two boards
    each spawned N workers on a derived N-worker host budget.

    Boards are matched by resolved DB path, so the ``HERMES_KANBAN_DB``
    override (which pins every board to one file) naturally yields 0.
    Fails open per board: one broken/corrupt board must not brick dispatch
    on the healthy ones.
    """
    try:
        current_path = str(_kb.kanban_db_path(board=board).expanduser().resolve())
    except Exception:
        current_path = None
    try:
        boards = _kb.list_boards(include_archived=False)
    except Exception:
        return 0
    total = 0
    for meta in boards:
        slug = meta.get("slug") or _kb.DEFAULT_BOARD
        try:
            path = _kb.kanban_db_path(board=slug).expanduser()
            resolved = str(path.resolve())
            if current_path is not None and resolved == current_path:
                continue
            if not path.exists():
                continue
            other = _kb.connect(board=slug)
            try:
                total += count_running_tasks(other)
            finally:
                with contextlib.suppress(Exception):
                    other.close()
        except Exception:
            continue
    return total


def _memory_pressure_level(sample: Optional[Mapping[str, Any]] = None) -> str:
    """Classify current system memory pressure: ok/elevated/critical/unknown.

    Reuses :func:`gateway.memory_status.classify_pressure` so the dispatcher's
    idea of "critical" matches the memory banner users see on the dashboard
    and the lifecycle ledger's OOM-suspicion heuristics (NS-608/NS-656).
    ``unknown`` (non-Linux, read failure) imposes no restriction — the guard
    must never brick dispatch on hosts where /proc isn't available.
    """
    if sample is None:
        sample = _kb._system_memory_sample()
    if not sample:
        return "unknown"
    try:
        from gateway.memory_status import classify_pressure
        return classify_pressure(
            sample.get("mem_available_kib"), sample.get("mem_total_kib")
        )
    except Exception:
        return "unknown"


def dispatch_once(
    conn: sqlite3.Connection,
    *,
    spawn_fn=None,
    ttl_seconds: Optional[int] = None,
    dry_run: bool = False,
    max_spawn: Optional[int] = None,
    max_in_progress: Optional[int] = None,
    failure_limit: int = DEFAULT_SPAWN_FAILURE_LIMIT,
    stale_timeout_seconds: int = 0,
    board: Optional[str] = None,
    default_assignee: Optional[str] = None,
    max_in_progress_per_profile: Optional[int] = None,
    reconcile_orphans: bool = True,
) -> DispatchResult:
    """Run one dispatcher tick under the board's single-writer lock.

    Wraps :func:`_dispatch_once_locked` in the non-blocking board-scoped
    :func:`_dispatch_tick_lock` so two dispatchers on one ``kanban.db`` (the
    service-managed gateway plus an orphan that escaped its cgroup) never
    race a write tick on WAL frames. The loser returns an empty
    ``DispatchResult`` with ``skipped_locked=True`` and writes nothing; the
    lock is keyed on the resolved DB path so unrelated boards tick in parallel.
    """
    try:
        db_path = _kb.kanban_db_path(board=board)
    except Exception:
        # Path resolution should never fail, but if it somehow does we
        # must not lose the tick — fall through to an unguarded dispatch
        # rather than dropping work.
        result = _dispatch_once_locked(
            conn,
            spawn_fn=spawn_fn,
            ttl_seconds=ttl_seconds,
            dry_run=dry_run,
            max_spawn=max_spawn,
            max_in_progress=max_in_progress,
            failure_limit=failure_limit,
            stale_timeout_seconds=stale_timeout_seconds,
            board=board,
            default_assignee=default_assignee,
            max_in_progress_per_profile=max_in_progress_per_profile,
            reconcile_orphans=reconcile_orphans,
        )
        _kb._fire_dispatch_tick_hook(result, board=board, dry_run=dry_run)
        return result
    with _kb._dispatch_tick_lock(db_path) as held:
        if not held:
            result = DispatchResult(skipped_locked=True)
        else:
            result = _dispatch_once_locked(
                conn,
                spawn_fn=spawn_fn,
                ttl_seconds=ttl_seconds,
                dry_run=dry_run,
                max_spawn=max_spawn,
                max_in_progress=max_in_progress,
                failure_limit=failure_limit,
                stale_timeout_seconds=stale_timeout_seconds,
                board=board,
                default_assignee=default_assignee,
                max_in_progress_per_profile=max_in_progress_per_profile,
                reconcile_orphans=reconcile_orphans,
            )
            # Still under the dispatch lock: run the periodic PASSIVE WAL
            # checkpoint (see _maybe_checkpoint_wal; the -wal file size is
            # bounded by journal_size_limit on the writer's natural reset).
            _kb._maybe_checkpoint_wal(conn, db_path)
    # The dispatch lock has been released here. Fire the tick observer
    # strictly OUTSIDE the single-writer critical section (#56066 sweeper
    # finding / #64231 disposition): a slow subscriber must never extend
    # the lock hold and stall a sibling dispatcher's tick.
    _kb._fire_dispatch_tick_hook(result, board=board, dry_run=dry_run)
    return result


def _dispatch_lane_task(
    conn: sqlite3.Connection,
    row: sqlite3.Row,
    assignee: str,
    result: "DispatchResult",
    *,
    lane: str,
    dry_run: bool,
    ttl_seconds: Optional[int],
    board: Optional[str],
    failure_limit: int,
    spawn_fn,
    per_profile_cap: Optional[int],
    per_profile_running: dict[str, int],
) -> bool:
    """Guard, claim, resolve the workspace and spawn one ready/review row.

    Shared tail of both dispatch lanes. Returns True when a spawn slot was
    consumed (real spawn, or a would-be spawn under ``dry_run``); every skip
    is recorded on ``result``.
    """
    task_id = row["id"]
    # Skip tasks whose assignee is not a real Hermes profile: ``_default_spawn``
    # runs ``hermes -p <assignee>``, which fails on startup for control-plane
    # lanes (interactive terminals that pull via ``claim_task``), and the task
    # would loop ready→crash→ready forever. Bucketed apart from
    # skipped_unassigned: the operator cannot fix it by assigning a profile,
    # and health telemetry uses the distinction to suppress "stuck" warnings.
    try:
        from hermes_cli.profiles import profile_exists  # local import: avoids cycle
    except Exception:
        profile_exists = None  # type: ignore[assignment]
    if profile_exists is not None and not profile_exists(assignee):
        result.skipped_nonspawnable.append(task_id)
        return False
    # Per-profile concurrency cap: even with global headroom, refuse to spawn
    # for an assignee already at its in-flight cap so one profile's local
    # model / API quota / browser pool isn't overwhelmed by a fan-out.
    if per_profile_cap is not None:
        current = per_profile_running.get(assignee, 0)
        if current >= per_profile_cap:
            result.skipped_per_profile_capped.append((task_id, assignee, current))
            return False
    # Respawn guard: defer this tick when useful work is already in-flight /
    # recent or the last failure is a deterministic blocker (quota / auth).
    # consecutive_failures still trips the auto-block breaker eventually.
    guard_reason = _kb.check_respawn_guard(conn, task_id, lane=lane)
    if guard_reason is not None:
        result.respawn_guarded.append((task_id, guard_reason))
        # Event so ``hermes kanban tail`` shows why the task looks stuck.
        if not dry_run:
            with _kb.write_txn(conn):
                _kb._append_event(conn, task_id, "respawn_guarded", {"reason": guard_reason})
        return False
    if dry_run:
        result.spawned.append((task_id, assignee, ""))
        # Count the would-be spawn so the cap check sees it on later rows;
        # otherwise dry_run under-reports the capped subset.
        if per_profile_cap is not None and assignee:
            per_profile_running[assignee] = per_profile_running.get(assignee, 0) + 1
        return True
    claim = _kb.claim_review_task if lane == "review" else _kb.claim_task
    claimed = claim(conn, task_id, ttl_seconds=ttl_seconds)
    if claimed is None:
        return False
    try:
        resolved_branch_name = None
        if claimed.workspace_kind == "worktree":
            workspace, resolved_branch_name = _kb._resolve_worktree_workspace(claimed, board=board)
        else:
            workspace = _kb.resolve_workspace(claimed, board=board)
    except Exception as exc:
        if _record_spawn_failure(conn, claimed.id, f"workspace: {exc}", failure_limit=failure_limit):
            result.auto_blocked.append(claimed.id)
        return False
    # Persist the resolved workspace path so the worker can cd there.
    _kb.set_workspace_path(conn, claimed.id, str(workspace))
    if claimed.workspace_kind == "worktree":
        _kb.set_branch_name(conn, claimed.id, resolved_branch_name or (claimed.branch_name or "").strip() or f"wt/{claimed.id}")
    _kb._maybe_emit_scratch_tip(conn, claimed.id, claimed.workspace_kind)
    if lane == "review":
        # Force-load the sdlc-review skill (AC verification, merge, ...). The
        # mandatory kanban lifecycle is already in every worker's system
        # prompt via KANBAN_GUIDANCE, so this is the only extra skill needed.
        claimed.skills = list(dict.fromkeys([*(claimed.skills or []), "sdlc-review"]))
    _spawn = spawn_fn if spawn_fn is not None else _default_spawn
    try:
        # Back-compat: older spawn_fn signatures (and test stubs) accept only
        # (task, workspace); pass ``board`` only when the callable supports it.
        import inspect
        try:
            sig = inspect.signature(_spawn)
            if "board" in sig.parameters:
                pid = _spawn(claimed, str(workspace), board=board)
            else:
                pid = _spawn(claimed, str(workspace))
        except (TypeError, ValueError):
            pid = _spawn(claimed, str(workspace))
        if pid:
            _set_worker_pid(conn, claimed.id, int(pid))
        # Worker-lifecycle observer: fires AFTER spawn_fn returned and the PID
        # (when reported) is durably persisted. Best-effort.
        _kb._fire_worker_spawned_hook(conn, claimed, str(workspace), pid, board=board)
        # consecutive_failures is deliberately NOT reset here: a successful
        # spawn doesn't prove the run will succeed, and resetting on spawn
        # would let a task that keeps timing out loop forever. Cleared only
        # on successful completion (complete_task).
        result.spawned.append((claimed.id, claimed.assignee or "", str(workspace)))
        # Track the new in-flight count so later rows in this tick respect
        # the per-profile cap; subsequent ticks re-query from the DB.
        if per_profile_cap is not None and claimed.assignee:
            per_profile_running[claimed.assignee] = per_profile_running.get(claimed.assignee, 0) + 1
        return True
    except Exception as exc:
        if _record_spawn_failure(conn, claimed.id, str(exc), failure_limit=failure_limit):
            result.auto_blocked.append(claimed.id)
        return False


def _apply_default_assignee(
    conn: sqlite3.Connection, task_id: str, assignee: str, *, dry_run: bool,
) -> bool:
    """Persist ``kanban.default_assignee`` on an unassigned ready row.

    Mutating the row (not just the in-memory view) keeps diagnostics and board
    state consistent: the task is now legitimately owned by the default, not
    "unassigned but secretly routed". ``dry_run`` reports the would-be
    assignment without writing. Returns False when the write failed.
    """
    if dry_run:
        return True
    try:
        with _kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET assignee = ? WHERE id = ? "
                "AND (assignee IS NULL OR assignee = '')",
                (assignee, task_id),
            )
            _kb._append_event(
                conn, task_id, "assigned",
                {"assignee": assignee, "source": "kanban.default_assignee"},
            )
    except Exception:
        _kb._log.debug(
            "kanban dispatch: failed to apply default_assignee=%r to task %s",
            assignee, task_id, exc_info=True,
        )
        return False
    return True


def _dispatch_once_locked(
    conn: sqlite3.Connection,
    *,
    spawn_fn=None,
    ttl_seconds: Optional[int] = None,
    dry_run: bool = False,
    max_spawn: Optional[int] = None,
    max_in_progress: Optional[int] = None,
    failure_limit: int = DEFAULT_SPAWN_FAILURE_LIMIT,
    stale_timeout_seconds: int = 0,
    board: Optional[str] = None,
    default_assignee: Optional[str] = None,
    max_in_progress_per_profile: Optional[int] = None,
    reconcile_orphans: bool = True,
) -> DispatchResult:
    """Run one dispatcher tick: reclaim stale (TTL / heartbeat) and crashed
    (dead host-local PID) running tasks, promote todo -> ready where all
    parents are done, then for each ready task with an assignee atomically
    claim and call ``spawn_fn(task, workspace_path, board) -> Optional[int]``,
    recording the returned PID as ``worker_pid`` so later ticks catch crashes
    before the TTL expires.

    After ``failure_limit`` consecutive per-task failures the task is
    auto-blocked with the last error as reason (no thrashing on an unfixable
    task). ``max_spawn`` is a live per-board concurrency cap (running + this
    tick's spawns), not a per-tick budget — a per-tick reading would grow
    concurrency by N every tick. ``max_in_progress`` is a host-level cap over
    every active board plus this tick's spawns: workers are OS processes
    sharing one machine's memory, so a per-board reading would multiply it by
    the board count. ``spawn_fn`` defaults to ``_default_spawn`` (tests stub
    it); ``board`` pins workspace/log/db resolution for this tick, else the
    current-board chain is used.
    """
    # Reap zombie children from previously spawned workers. See
    # reap_worker_zombies() for the full rationale.
    reap_worker_zombies()

    result = DispatchResult()
    result.reclaimed = _kb.release_stale_claims(conn)
    if reconcile_orphans:
        # Orphaned-card reconciliation: requeue 'running' cards whose claim
        # bookkeeping is broken (no valid claim, dead/gone worker) that the
        # TTL/crash/stale paths can never see. See reconcile_orphaned_running.
        result.reconciled_orphans = reconcile_orphaned_running(conn)
    result.stale = detect_stale_running(
        conn, stale_timeout_seconds=stale_timeout_seconds,
    )
    result.crashed = detect_crashed_workers(conn)
    # detect_crashed_workers stashes protocol-violation auto-blocks on
    # itself so the public list-return stays stable. Pull them into the
    # DispatchResult here so telemetry / tests see the trip.
    result.auto_blocked.extend(getattr(detect_crashed_workers, "_last_auto_blocked", []))
    # Rate-limited requeues (quota wall, no failure counted) — surface for
    # telemetry / tests. These tasks went back to ``ready`` and the respawn
    # guard will defer them until the quota window clears.
    result.rate_limited.extend(getattr(detect_crashed_workers, "_last_rate_limited", []))
    result.timed_out = enforce_max_runtime(conn)
    result.promoted = _kb.recompute_ready(conn, failure_limit=failure_limit)

    # Count tasks already running so max_spawn enforces concurrency rather
    # than a per-tick spawn budget. See the docstring above for the full
    # rationale; the short version is that a 60-second tick interval with a
    # per-tick budget of N would grow concurrency by N every tick on a busy
    # board, since "running" tasks aren't reclaimed by completion alone —
    # they sit in status='running' until the worker calls
    # kanban_complete/kanban_block (or the dispatcher TTL-reclaims them).
    running_count = 0
    spawn_budget: Optional[int] = None
    if max_spawn is not None or max_in_progress is not None:
        running_count = count_running_tasks(conn)

    # Convert any concurrency caps into a shared additional-spawns budget
    # for this tick. Both ready and review loops consume from the same
    # budget so the total number of new workers stays bounded.
    if max_spawn is not None:
        if running_count >= max_spawn:
            return result
        spawn_budget = max_spawn - running_count

    # Honour kanban.max_in_progress across both ready and review queues: if
    # the board already has enough running tasks, skip this tick entirely.
    # When there is room left, intersect the remaining in-progress budget
    # with any explicit max_spawn cap above.
    #
    # max_in_progress is a HOST-level cap, not a per-board one (OOF-30):
    # workers are OS processes sharing one machine's memory, so running
    # workers on every other board count against the same budget. Without
    # this, N active boards multiply the cap by N — exactly the fan-out
    # the memory-derived default exists to prevent.
    if max_in_progress is not None:
        total_running = running_count + count_running_tasks_other_boards(board)
        if total_running >= max_in_progress:
            return result
        remaining = max_in_progress - total_running
        if spawn_budget is None or spawn_budget > remaining:
            spawn_budget = remaining

    # Memory-pressure guard (OOF-30/OOF-77): even a well-chosen static cap
    # can't see the host's actual memory state (other tenants, bloated
    # long-lived workers, dashboard growth). Under observed pressure the
    # dispatcher stops adding load: critical -> spawn nothing this tick;
    # elevated -> at most one new worker. Reclaim/promotion above already
    # ran, so board bookkeeping stays live either way, and deferred tasks
    # simply wait for a later tick. "unknown" imposes no restriction.
    pressure = _memory_pressure_level()
    if pressure == "critical":
        result.memory_pressure = pressure
        _kb._log.warning(
            "kanban dispatch: system memory pressure is critical; "
            "spawning no new workers this tick (deferred, not dropped)"
        )
        return result
    if pressure == "elevated":
        result.memory_pressure = pressure
        if spawn_budget is None or spawn_budget > 1:
            _kb._log.warning(
                "kanban dispatch: system memory pressure is elevated; "
                "limiting to at most 1 new worker this tick"
            )
            spawn_budget = 1

    ready_rows = conn.execute(
        "SELECT id, assignee FROM tasks "
        "WHERE status = 'ready' AND claim_lock IS NULL "
        "ORDER BY priority DESC, created_at ASC"
    ).fetchall()
    # Review rows are enumerated up front (not after the ready loop) so the
    # budget split below can see whether review work exists at all.
    review_rows = []
    if review_dispatch_enabled():
        review_rows = conn.execute(
            "SELECT id, assignee FROM tasks "
            "WHERE status = 'review' AND claim_lock IS NULL "
            "ORDER BY priority DESC, created_at ASC"
        ).fetchall()
    # Review-lane reservation: the ready loop runs first and used to consume
    # the ENTIRE shared budget, starving autonomous reviews under a sustained
    # ready backlog. When spawnable review work exists and there is any
    # budget, hold one slot back from the ready loop. Per-tick and
    # self-releasing: no spawnable review work (or no cap) → full budget.
    # "Spawnable" mirrors the review loop's own gate (assigned + real
    # profile) so human-pulled control-plane lanes don't tax ready throughput.
    def _any_spawnable_review() -> bool:
        if not review_rows:
            return False
        try:
            from hermes_cli.profiles import profile_exists as _rpe
        except Exception:
            # Profiles module unavailable (test stubs, exotic envs) —
            # assume spawnable, matching the review loop's own fallback.
            return any(row["assignee"] for row in review_rows)
        return any(
            row["assignee"] and _rpe(row["assignee"]) for row in review_rows
        )

    ready_budget = spawn_budget
    if spawn_budget is not None and spawn_budget > 0 and _any_spawnable_review():
        ready_budget = max(spawn_budget - 1, 0)
    spawned = 0
    # Per-profile concurrency cap: track in-flight workers per assignee and
    # refuse spawns past the cap so a fan-out can't melt one profile's local
    # model / API quota / browser pool. Deferred tasks go to
    # skipped_per_profile_capped, not skipped_unassigned — "busy, retry
    # later" is a different operator signal from "needs routing".
    _per_profile_cap = max_in_progress_per_profile if (
        isinstance(max_in_progress_per_profile, int)
        and max_in_progress_per_profile > 0
    ) else None
    _per_profile_running: dict[str, int] = {}
    if _per_profile_cap is not None:
        for prow in conn.execute(
            "SELECT assignee, COUNT(*) AS n FROM tasks "
            "WHERE status = 'running' AND assignee IS NOT NULL "
            "GROUP BY assignee"
        ):
            _per_profile_running[prow["assignee"]] = int(prow["n"])
    # Normalize default_assignee once (empty → None) and resolve
    # profile_exists once. When the profiles module isn't importable (test
    # stubs, exotic envs) trust the operator's config: the downstream
    # profile_exists check on the assigned row still buckets a missing
    # profile as nonspawnable with the existing diagnostic.
    _default_assignee = (default_assignee or "").strip() or None
    if _default_assignee:
        try:
            from hermes_cli.profiles import profile_exists as _pe
            if not _pe(_default_assignee):
                _default_assignee = None
        except Exception:
            pass
    for row in ready_rows:
        if ready_budget is not None and spawned >= ready_budget:
            break
        row_assignee = row["assignee"]
        if not row_assignee:
            # Honour kanban.default_assignee so a task created without an
            # assignee doesn't park in 'ready' forever when the operator's
            # intent ("default") was clear.
            if not _default_assignee or not _apply_default_assignee(
                conn, row["id"], _default_assignee, dry_run=dry_run,
            ):
                result.skipped_unassigned.append(row["id"])
                continue
            row_assignee = _default_assignee
            result.auto_assigned_default.append(row["id"])
        if _dispatch_lane_task(
            conn, row, row_assignee, result, lane="ready", dry_run=dry_run,
            ttl_seconds=ttl_seconds, board=board, failure_limit=failure_limit,
            spawn_fn=spawn_fn, per_profile_cap=_per_profile_cap,
            per_profile_running=_per_profile_running,
        ):
            spawned += 1

    # ---- review column dispatch ----
    # A worker moved the task to 'review' after creating a PR; the dispatcher
    # spawns a review agent (sdlc-review skill) that approves (→ done) or
    # requests changes (→ ready/todo for the implementer). Review spawns share
    # max_spawn with ready tasks so total running workers stay bounded; enabled
    # by default, disable with ``kanban.review_dispatch`` for human-only boards.
    # The review loop checks the FULL shared ``spawn_budget`` — the
    # reservation above caps the ready lane, it grants no extra capacity here.
    for row in review_rows:
        if spawn_budget is not None and spawned >= spawn_budget:
            break
        if not row["assignee"]:
            result.skipped_unassigned.append(row["id"])
            continue
        if _dispatch_lane_task(
            conn, row, row["assignee"], result, lane="review", dry_run=dry_run,
            ttl_seconds=ttl_seconds, board=board, failure_limit=failure_limit,
            spawn_fn=spawn_fn, per_profile_cap=_per_profile_cap,
            per_profile_running=_per_profile_running,
        ):
            spawned += 1
    return result


def _positive_int(value: Any, default: int, *, minimum: int = 1) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= minimum else default


def worker_log_rotation_config(kanban_cfg: Optional[dict] = None) -> tuple[int, int]:
    """Return ``(rotate_bytes, backup_count)`` for worker log rotation.

    Defaults preserve the historical behavior: rotate at 2 MiB and keep one
    backup generation (``.log.1``). Operators with long-running workers can
    raise either value from ``config.yaml`` without changing dispatcher code.
    """
    if kanban_cfg is None:
        try:
            from hermes_cli.config import load_config

            kanban_cfg = (load_config().get("kanban") or {})
        except Exception:
            kanban_cfg = {}
    max_bytes = _positive_int(
        (kanban_cfg or {}).get("worker_log_rotate_bytes"),
        DEFAULT_LOG_ROTATE_BYTES,
        minimum=1,
    )
    backup_count = _positive_int(
        (kanban_cfg or {}).get("worker_log_backup_count"),
        DEFAULT_LOG_BACKUP_COUNT,
        minimum=0,
    )
    return max_bytes, backup_count


def _rotated_log_path(log_path: Path, generation: int) -> Path:
    return log_path.with_suffix(log_path.suffix + f".{generation}")


def _rotate_worker_log(
    log_path: Path,
    max_bytes: int,
    backup_count: int = DEFAULT_LOG_BACKUP_COUNT,
) -> None:
    """Rotate ``<log>`` when it exceeds ``max_bytes``.

    ``backup_count=1`` preserves the legacy single-generation behavior:
    ``<log>`` moves to ``<log>.1`` and any previous ``.1`` is replaced.
    Higher values shift older generations up to ``backup_count``.
    """
    try:
        if not log_path.exists():
            return
        if log_path.stat().st_size <= max_bytes:
            return
        backup_count = _positive_int(
            backup_count,
            DEFAULT_LOG_BACKUP_COUNT,
            minimum=0,
        )
        if backup_count == 0:
            log_path.unlink()
            return
        oldest = _rotated_log_path(log_path, backup_count)
        try:
            if oldest.exists():
                oldest.unlink()
        except OSError:
            pass
        for generation in range(backup_count - 1, 0, -1):
            src = _rotated_log_path(log_path, generation)
            if not src.exists():
                continue
            with contextlib.suppress(OSError):
                src.rename(_rotated_log_path(log_path, generation + 1))
        log_path.rename(_rotated_log_path(log_path, 1))
    except OSError:
        pass


def _module_hermes_argv() -> list[str]:
    """Return the interpreter-bound Hermes CLI invocation."""
    # ``hermes_cli.main`` is the console-script target declared in
    # pyproject.toml, NOT a top-level ``hermes`` package — there is no
    # ``hermes`` package to import.
    return [sys.executable, "-m", "hermes_cli.main"]


def _absolute_hermes_path(path: str) -> str:
    """Return an absolute filesystem path for a resolved Hermes shim."""
    expanded = os.path.expanduser(path)
    return expanded if os.path.isabs(expanded) else os.path.abspath(expanded)


def _looks_like_path(value: str) -> bool:
    """Return true when a command override is an explicit path, not a name."""
    expanded = os.path.expanduser(value)
    return (
        expanded.startswith("~")
        or os.path.isabs(expanded)
        or bool(os.path.dirname(expanded))
        or "\\" in expanded
        or bool(re.match(r"^[A-Za-z]:", expanded))
    )


def _is_windows_batch_shim(path: str) -> bool:
    """Return true for Windows shell/batch shims that should not be argv[0]."""
    return path.lower().endswith((".cmd", ".bat"))


def _path_search_names(command: str) -> list[str]:
    """Return executable names to try for an unqualified command."""
    if not _kb._IS_WINDOWS or os.path.splitext(command)[1]:
        return [command]
    raw = os.environ.get("PATHEXT") or ".COM;.EXE;.BAT;.CMD"
    exts = [ext for ext in raw.split(";") if ext]
    return [command + ext for ext in exts]


def _safe_which_no_cwd(command: str) -> Optional[str]:
    """Resolve a bare command from PATH without implicit current-dir search.

    ``shutil.which`` follows platform search behavior. On Windows that can
    include the current directory before PATH for bare names, which is not a
    safe dispatcher primitive. This resolver only considers explicit PATH
    entries and skips empty / ``.`` entries.
    """
    path_env = os.environ.get("PATH", "")
    for raw_dir in path_env.split(os.pathsep):
        if not raw_dir or raw_dir == ".":
            continue
        directory = os.path.expanduser(raw_dir)
        for name in _path_search_names(command):
            candidate = os.path.join(directory, name)
            if not os.path.isfile(candidate):
                continue
            if _kb._IS_WINDOWS or os.access(candidate, os.X_OK):
                return candidate
    return None


def _hermes_path_argv(path: str) -> list[str]:
    """Return argv for a resolved Hermes executable path.

    Windows batch shims (`.cmd` / `.bat`) are not safe as argv[0] for
    worker launches because the argument vector includes task-derived
    values. Prefer the interpreter-bound module form whenever the resolved
    executable is only a shell shim.
    """
    if _kb._IS_WINDOWS and _is_windows_batch_shim(path):
        return _module_hermes_argv()
    return [_absolute_hermes_path(path)]


def _resolve_hermes_argv() -> list[str]:
    """Resolve the ``hermes`` invocation as argv parts for ``Popen``.

    Tries in order:

    1. ``$HERMES_BIN`` — explicit operator override. Path-like values are
       normalized to absolute paths; bare command names keep normal PATH
       semantics and never prefer a same-directory file before ``PATH``.
    2. ``shutil.which("hermes")`` — the console-script shim, normalized to
       an absolute path. On Windows, ``which`` can return a relative
       ``.\\hermes.CMD`` when the current directory is on ``PATH``; directly
       launching batch shims is also unsafe with task-derived argv. The
       dispatcher therefore falls back to the interpreter-bound module form
       for implicit ``.cmd`` / ``.bat`` shims.
    3. ``sys.executable -m hermes_cli.main`` — fallback for setups where
       Hermes is launched from a venv and the ``hermes`` shim is not on
       the dispatcher's ``$PATH`` (cron, systemd ``User=`` services,
       launchd jobs, detached processes, etc.). Goes through the running
       interpreter so the result is independent of ``$PATH``.

    Mirrors ``gateway.run._resolve_hermes_bin`` for the same reason. Kept
    local (not imported from gateway) because ``hermes_cli`` sits below
    ``gateway`` in the dependency order.
    """
    import shutil

    env_bin = os.environ.get("HERMES_BIN", "").strip()
    if env_bin:
        if _looks_like_path(env_bin):
            return _hermes_path_argv(env_bin)
        resolved_env_bin = _safe_which_no_cwd(env_bin)
        if resolved_env_bin:
            return _hermes_path_argv(resolved_env_bin)
        return _module_hermes_argv()

    hermes_bin = _safe_which_no_cwd("hermes") if _kb._IS_WINDOWS else shutil.which("hermes")
    if hermes_bin:
        return _hermes_path_argv(hermes_bin)
    return _module_hermes_argv()


def _worker_terminal_timeout_env(
    max_runtime_seconds: Optional[int],
    current_timeout: Optional[str],
) -> Optional[str]:
    """Return a worker-scoped TERMINAL_TIMEOUT override, if needed.

    Kanban's ``max_runtime_seconds`` bounds the whole worker attempt. The
    terminal tool has its own default timeout via ``TERMINAL_TIMEOUT``; when
    the worker runtime is longer, raise only the child process default so a
    long command is not killed by the generic terminal default first.
    """
    if max_runtime_seconds is None:
        return None
    try:
        runtime = int(max_runtime_seconds)
    except (TypeError, ValueError):
        return None
    if runtime <= 0:
        return None

    desired = max(1, runtime - KANBAN_TERMINAL_TIMEOUT_GRACE_SECONDS)
    try:
        existing = int(str(current_timeout).strip()) if current_timeout else 0
    except (TypeError, ValueError):
        existing = 0
    if existing >= desired:
        return None
    return str(desired)


def _resolve_worker_cli_toolsets(hermes_home: Optional[str]) -> Optional[list[str]]:
    """Return the assigned profile's effective CLI toolsets for a worker.

    Dispatcher-spawned workers are launched from a long-lived gateway process,
    then the child re-enters the CLI with ``-p <assignee>``. Resolve the
    assignee profile's CLI tool surface at dispatch time and pass it as an
    explicit ``--toolsets`` pin so worker startup cannot fall back to a stale
    root/active-profile config or a profile whose top-level ``toolsets`` entry
    is only the kanban orchestrator surface. ``model_tools`` still appends the
    task-scoped kanban lifecycle tools when ``HERMES_KANBAN_TASK`` is set.
    """
    if not hermes_home:
        return None
    try:
        from hermes_constants import reset_hermes_home_override, set_hermes_home_override
        from hermes_cli.config import load_config
        from hermes_cli.tools_config import _get_platform_tools

        token = set_hermes_home_override(hermes_home)
        try:
            cfg = load_config()
            toolsets = sorted(_get_platform_tools(cfg, "cli"))
        finally:
            reset_hermes_home_override(token)
        return toolsets or None
    except Exception as exc:
        _kb._log.debug(
            "kanban worker: could not resolve CLI toolsets for HERMES_HOME=%r (%s)",
            hermes_home,
            exc,
        )
        return None


_retagged_workspace_roots: set[str] = set()


def _retag_legacy_worker_sessions(workspaces_root_path: str) -> None:
    """Reclaim pre-tag worker rows in state.db so they leave the session lists.

    Best-effort and gated — the durable ``state_meta`` gate lives in
    ``retag_kanban_worker_sessions``; the in-process set keeps a busy
    dispatcher from reopening state.db on every spawn just to read it. A
    dispatcher tick must never fail because a session DB was busy or missing.
    """
    if workspaces_root_path in _retagged_workspace_roots:
        return
    try:
        from hermes_state import SessionDB

        db = SessionDB()
        try:
            db.retag_kanban_worker_sessions(workspaces_root_path)
        finally:
            db.close()
        _retagged_workspace_roots.add(workspaces_root_path)
    except Exception as exc:
        _kb._log.debug("kanban worker: legacy session retag skipped (%s)", exc)


def _default_spawn(
    task: Task,
    workspace: str,
    *,
    board: Optional[str] = None,
) -> Optional[int]:
    """Fire-and-forget ``hermes -p <profile> chat -q ...`` subprocess.

    Returns the spawned child's PID so the dispatcher can detect crashes
    before the claim TTL expires. The child's completion is still observed
    via the ``complete`` / ``block`` transitions the worker writes itself;
    the PID check is a safety net for crashes, OOM kills, and Ctrl+C.

    ``board`` pins the child's kanban context to that board: the child's
    ``HERMES_KANBAN_DB`` / ``HERMES_KANBAN_BOARD`` / workspaces_root env
    vars all resolve to the same board the dispatcher claimed the task
    from. Workers cannot accidentally see other boards.
    """
    import subprocess
    if not task.assignee:
        raise ValueError(f"task {task.id} has no assignee")

    from hermes_cli.profiles import normalize_profile_name

    profile_arg = normalize_profile_name(task.assignee)

    prompt = f"work kanban task {task.id}"
    env = dict(os.environ)
    # The dispatcher is detached from every conversation. Its worker must never
    # inherit routing mirrored by a previous gateway turn, even before the first
    # session binds ContextVars in this process.
    from gateway.session_context import _VAR_MAP
    for key in _VAR_MAP:
        env.pop(key, None)

    # Inject HERMES_HOME so the worker reads the profile-scoped config.yaml
    # (fallback_providers, toolsets, ...). Without it the child's
    # get_hermes_home() falls back to the DEFAULT profile root because
    # `hermes -p` applies its override before hermes_constants is imported.
    from hermes_cli.profiles import resolve_profile_env
    try:
        env["HERMES_HOME"] = resolve_profile_env(profile_arg)
    except FileNotFoundError:
        # No profile dir (isolated test fixtures) — the CLI resolves it from
        # HERMES_PROFILE (set below) instead.
        pass
    if task.tenant:
        env["HERMES_TENANT"] = task.tenant
    env["HERMES_KANBAN_TASK"] = task.id
    env["HERMES_KANBAN_WORKSPACE"] = workspace
    # Tag the session as `kanban` (not an untitled `cli` row): a worker is a
    # dispatcher-owned run read on the board / `hermes kanban log`, so every
    # session-browsing surface filters it out by source instead of rendering
    # one sidebar row per attempt.
    env["HERMES_SESSION_SOURCE"] = "kanban"
    # Pin TERMINAL_CWD to the workspace: it takes precedence over the process
    # cwd in file_tools._resolve_base_dir and build_context_files_prompt, so
    # without it relative writes land in the gateway user's home and workers
    # load the gateway's AGENTS.md. Only a real absolute directory — file_tools
    # rejects relative / sentinel values, so leave the inherited value otherwise.
    if workspace and os.path.isabs(workspace) and os.path.isdir(workspace):
        env["TERMINAL_CWD"] = workspace
    if task.branch_name:
        env["HERMES_KANBAN_BRANCH"] = task.branch_name
    if task.current_run_id is not None:
        env["HERMES_KANBAN_RUN_ID"] = str(task.current_run_id)
    if task.claim_lock:
        env["HERMES_KANBAN_CLAIM_LOCK"] = task.claim_lock
    # Goal-loop mode: the worker reads these and wraps its run in the
    # Ralph-style /goal judge loop (see cli.py quiet-mode path). Only set
    # when enabled so non-goal tasks keep a clean env.
    if task.goal_mode:
        env["HERMES_KANBAN_GOAL_MODE"] = "1"
        if task.goal_max_turns is not None:
            env["HERMES_KANBAN_GOAL_MAX_TURNS"] = str(int(task.goal_max_turns))
    terminal_timeout = _worker_terminal_timeout_env(
        task.max_runtime_seconds,
        env.get("TERMINAL_TIMEOUT"),
    )
    if terminal_timeout is not None:
        env["TERMINAL_TIMEOUT"] = terminal_timeout
    foreground_timeout = _worker_terminal_timeout_env(
        task.max_runtime_seconds,
        env.get("TERMINAL_MAX_FOREGROUND_TIMEOUT"),
    )
    if foreground_timeout is not None:
        env["TERMINAL_MAX_FOREGROUND_TIMEOUT"] = foreground_timeout
    # Pin the board DB + workspaces root the dispatcher resolved so the
    # worker's kanban paths still match after `hermes -p` rewrites
    # HERMES_HOME (belt-and-braces for symlink / Docker layouts).
    env["HERMES_KANBAN_DB"] = str(_kb.kanban_db_path(board=board))
    env["HERMES_KANBAN_WORKSPACES_ROOT"] = str(_kb.workspaces_root(board=board))
    _kb._retag_legacy_worker_sessions(env["HERMES_KANBAN_WORKSPACES_ROOT"])
    # Board slug — final defense-in-depth pin if a path is ever resolved
    # without the DB / workspaces env vars.
    resolved_board = _kb._normalize_board_slug(board) or _kb.get_current_board()
    env["HERMES_KANBAN_BOARD"] = resolved_board
    # kanban_comment reads HERMES_PROFILE for its default author; `-p` alone
    # doesn't set the env var.
    env["HERMES_PROFILE"] = profile_arg

    # A worker must NEVER boot the interactive TUI: its no-TTY bail-out exits
    # 0 without doing the task → "protocol violation" every attempt. `--cli`
    # is the highest-precedence override; dropping HERMES_TUI covers older
    # hermes builds on PATH that predate the flag's precedence.
    env.pop("HERMES_TUI", None)

    cmd = [
        *_kb._resolve_hermes_argv(),
        "-p", profile_arg,
        "--cli",
        # Worker subprocesses switch to a profile-scoped HERMES_HOME above,
        # so they see that profile's shell-hook allowlist instead of the
        # dispatcher's root allowlist. Pass --accept-hooks explicitly so
        # profile-local worker sessions still register configured hooks.
        "--accept-hooks",
    ]
    # Per-task force-loaded skills. Each name goes in its own
    # `--skills X` pair rather than a single comma-joined arg: the CLI
    # accepts both forms (action='append' + comma-split), but
    # per-name pairs are easier to read in `ps` output and avoid any
    # quoting ambiguity if a skill name ever contains unusual chars.
    if task.skills:
        for sk in task.skills:
            if sk:
                cmd.extend(["--skills", sk])
    if task.model_override:
        cmd.extend(["-m", task.model_override])
        # Pin the provider too when the override names one, so the worker
        # resolves the model against the intended backend instead of the
        # profile's configured provider (mixing model X with provider Y is
        # the classic mis-set that stalls a board).
        if task.provider_override:
            cmd.extend(["--provider", task.provider_override])
    # Per-task thinking depth. Independent of the model override — a task can
    # run the profile's own model at a different depth — so this is its own
    # branch, not a nested one.
    if task.reasoning_effort:
        cmd.extend(["--reasoning", task.reasoning_effort])
    worker_toolsets = _resolve_worker_cli_toolsets(env.get("HERMES_HOME"))
    if worker_toolsets:
        cmd.extend(["--toolsets", ",".join(worker_toolsets)])
    cmd.extend([
        "chat",
        "-q", prompt,
    ])
    if task.goal_mode:
        # Goal-mode workers must take the fully-quiet single-query path:
        # the kanban goal-loop hook (_run_kanban_goal_loop_q) only runs in
        # cli.py's quiet branch. Without -Q the worker gets exactly one
        # turn, prints text, exits rc=0, and the dispatcher records a
        # protocol violation (incident 2026-06-09 t_d9cbe312).
        cmd.append("-Q")
    # Redirect output to a per-task log under <board-root>/logs/.
    # Anchored at the board root (not the shared kanban root), so
    # `hermes kanban log` on a specific board reads its own file and
    # logs don't collide across boards that happen to share task ids.
    log_dir = _kb.worker_logs_dir(board=board)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{task.id}.log"
    rotate_bytes, backup_count = worker_log_rotation_config()
    _rotate_worker_log(log_path, rotate_bytes, backup_count)

    # Use 'a' so a re-run on unblock appends rather than overwrites.
    log_f = open(log_path, "ab")
    try:
        proc = subprocess.Popen(  # noqa: S603 -- argv is a fixed list built above
            cmd,
            cwd=workspace if os.path.isdir(workspace) else None,
            stdin=subprocess.DEVNULL,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
            creationflags=subprocess.CREATE_NO_WINDOW if _kb._IS_WINDOWS else 0,
        )
    except FileNotFoundError:
        log_f.close()
        raise RuntimeError(
            "`hermes` executable not found on PATH. "
            "Install Hermes Agent or activate its venv before running the kanban dispatcher."
        )
    # NOTE: we intentionally do NOT close log_f here — we want Popen's
    # child process to keep writing after this function returns.  The
    # handle is kept alive by the child's inheritance.  The parent's
    # reference goes out of scope and is GC'd, but the OS-level FD stays
    # open in the child until the child exits.
    return proc.pid


# ---------------------------------------------------------------------------
# Long-lived dispatcher daemon
# ---------------------------------------------------------------------------

def run_daemon(
    *,
    interval: float = 60.0,
    max_spawn: Optional[int] = None,
    failure_limit: int = DEFAULT_SPAWN_FAILURE_LIMIT,
    stop_event=None,
    on_tick=None,
) -> None:
    """Run the dispatcher in a loop until interrupted.

    Calls :func:`dispatch_once` every ``interval`` seconds. Exits cleanly
    on SIGINT / SIGTERM so ``hermes kanban daemon`` is systemd-friendly.
    ``stop_event`` (a :class:`threading.Event`) and ``on_tick`` (a
    callable receiving the :class:`DispatchResult`) are test hooks.

    Each tick resolves ``kanban.max_in_progress`` (explicit config, else
    the memory-derived default) exactly like the gateway-embedded
    dispatcher and ``hermes kanban dispatch`` — the standalone daemon must
    not be the one uncapped entry point (OOF-30).
    """
    import signal
    import threading

    if stop_event is None:
        stop_event = threading.Event()

    def _handle(_signum, _frame):
        stop_event.set()

    # Install handlers only when running on the main thread — tests call
    # this inline from worker threads and signal() would raise there.
    if threading.current_thread() is threading.main_thread():
        for sig_name in ("SIGINT", "SIGTERM"):
            sig = getattr(signal, sig_name, None)
            if sig is not None:
                with contextlib.suppress(ValueError, OSError):
                    signal.signal(sig, _handle)

    while not stop_event.is_set():
        try:
            # Resolve the global concurrency cap the same way the gateway
            # dispatcher and `hermes kanban dispatch` do (OOF-30): explicit
            # kanban.max_in_progress wins, otherwise the memory-derived
            # default applies. The standalone daemon previously passed no
            # cap at all — the shipped systemd path could still fan out an
            # entire backlog in one tick even with the derived default in
            # place everywhere else. Re-resolved every tick (config load is
            # mtime-cached) so operator edits apply without a restart.
            max_in_progress = resolve_max_in_progress(
                _kb.configured_max_in_progress()
            )
            with contextlib.closing(_kb.connect()) as conn:
                res = _kb.dispatch_once(
                    conn,
                    max_spawn=max_spawn,
                    max_in_progress=max_in_progress,
                    failure_limit=failure_limit,
                )
            if on_tick is not None:
                with contextlib.suppress(Exception):
                    on_tick(res)
        except Exception:
            # Don't let any single tick kill the daemon.
            import traceback
            traceback.print_exc()
        stop_event.wait(timeout=interval)


# Late-bound origin namespace (see module docstring). Imported LAST so this
# module is fully populated before ``kanban_db`` re-exports from it.
from hermes_cli import kanban_db as _kb  # noqa: E402
