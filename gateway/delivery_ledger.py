"""Durable delivery-obligation ledger for gateway final responses.

A final agent response generated but not yet confirmed-delivered is the one
artifact the gateway can lose without a trace: the turn already burned its
tokens, the text exists only in a Python local, and a crash / restart between
finalize and platform ACK drops it silently.

Each outbound final response gets a small durable row in the shared ``state.db``
(same conventions as ``tools.async_delegation`` — WAL, owner pid + process-start
liveness, bounded retention). The gateway writes three checkpoints:

    record_obligation()   state='pending'     before any send attempt
    mark_attempting()     state='attempting'  immediately before the await
    mark_delivered() /    state='delivered'   only on SendResult.success
    mark_failed()         state='failed'      on a definitive rejection

On startup ``sweep_recoverable()`` claims rows whose owner is dead and hands
them back for redelivery. After an adapter reconnects without a restart,
``sweep_failed_for_runtime()`` may claim only the same live process's
allowlisted transient failures. Crash semantics are explicit about ambiguity
(never silently resend an ambiguous send):

- ``pending``     — send never started: redeliver plainly, no dup risk.
- ``attempting``  — crashed mid-await: the platform MAY already have it.
  Redelivered WITH a visible recovered-reply marker (honest at-least-once).
- ``failed``      — definitively rejected once; restart is a natural retry
  boundary. Also carries the marker.
- ``delivered``   — nothing to do; retention prunes.

Poison rows cannot spin: attempts are capped and stale rows expire, both
transitioning to ``abandoned`` (kept briefly, then pruned).

Everything here is best-effort: ledger failures must never block or delay an
actual send. Callers wrap every call in try/except.
"""

from __future__ import annotations

import hashlib
import logging
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

_DB_LOCK = threading.Lock()

# Redelivery policy knobs (deliberately not config — the ledger is gated by
# ``gateway.delivery_ledger`` and these only matter in the rare recovery path).
MAX_ATTEMPTS = 3
STALE_AFTER_SECONDS = 24 * 60 * 60
_RETENTION_SECONDS = 7 * 24 * 60 * 60
_MAX_ROWS = 500

# Visible prefix for redeliveries that might duplicate an already-received
# message (crash mid-send / post-rejection retry). Honest at-least-once.
RECOVERED_MARKER = (
    "♻️ Recovered reply — the gateway restarted during delivery, "
    "so this may be a duplicate:\n\n"
)

# Runtime recovery uses a distinct marker: no restart occurred, but a network
# rejection's acknowledgement can still have been lost independently.
RECONNECTED_MARKER = (
    "♻️ Recovered reply — the messaging platform reconnected after the original "
    "delivery failed, so this may be a duplicate:\n\n"
)

# Runtime replay is fail-closed: only errors whose send contract proves they are
# transient reconnect failures. Permanent rejects (blocked bot, bad auth, missing
# chat) must not be retried merely because an adapter reconnected.
_RUNTIME_RETRYABLE_ERRORS = frozenset({"send_path_degraded"})


def _db_path():
    return get_hermes_home() / "state.db"


def _connect() -> sqlite3.Connection:
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=10)
    try:
        _initialize_schema(conn)
    except Exception:
        # A PRAGMA/DDL failure after connect() must not leak the connection.
        conn.close()
        raise
    return conn


def _initialize_schema(conn: sqlite3.Connection) -> None:
    from hermes_state import apply_wal_with_fallback

    apply_wal_with_fallback(conn, db_label="state.db (delivery_ledger)")
    conn.execute(
        """CREATE TABLE IF NOT EXISTS delivery_obligations (
            obligation_id TEXT PRIMARY KEY,
            session_key TEXT NOT NULL,
            platform TEXT NOT NULL,
            chat_id TEXT NOT NULL,
            thread_id TEXT,
            content TEXT NOT NULL,
            state TEXT NOT NULL,
            attempts INTEGER NOT NULL DEFAULT 0,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            owner_pid INTEGER,
            owner_started_at INTEGER,
            last_error TEXT,
            adapter_profile TEXT
        )"""
    )
    columns = {row[1] for row in conn.execute("PRAGMA table_info(delivery_obligations)")}
    if "adapter_profile" not in columns:
        try:
            conn.execute("ALTER TABLE delivery_obligations ADD COLUMN adapter_profile TEXT")
        except sqlite3.OperationalError as exc:
            # Concurrent first-use connections can both observe the old schema.
            if "duplicate column" not in str(exc).lower():
                raise


@contextmanager
def _transaction() -> Iterator[sqlite3.Connection]:
    """Open a connection, commit/rollback on exit, and ALWAYS close it.

    ``sqlite3.Connection`` as a context manager only commits/rolls back; it does
    not close. ``with _connect()`` alone would leak a connection (and its WAL/SHM
    fds) per call until GC — ``record_obligation`` runs on every outbound final
    response, so this ledger would exhaust ``RLIMIT_NOFILE`` on a long-running gateway.
    """
    conn = _connect()
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def _owner_stamp() -> tuple[int, Optional[int]]:
    pid = os.getpid()
    try:
        from gateway.status import get_process_start_time

        return pid, get_process_start_time(pid)
    except Exception:
        return pid, None


def _owner_alive(pid: Any, started_at: Any) -> bool:
    """True when the recorded owning process still exists (pid + start time)."""
    if not pid:
        return False
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    try:
        from gateway.status import get_process_start_time

        current_start = get_process_start_time(pid)
    except Exception:
        current_start = None
    if current_start is None:
        # Start time unreadable: alive iff the pid exists. Route through the
        # cross-platform probe — ``os.kill(pid, 0)`` on Windows is NOT a no-op
        # (bpo-14484: it maps to ``GenerateConsoleCtrlEvent(0, pid)`` and could
        # Ctrl+C the gateway's own console group). ``_pid_exists`` keeps the
        # EPERM-means-alive semantics (pid exists but is owned by another user).
        try:
            from gateway.status import _pid_exists
        except Exception:
            if os.name == "nt":
                return False  # never fall back to a raw sig-0 probe on Windows
            try:
                os.kill(pid, 0)  # windows-footgun: ok — POSIX-only fallback branch
            except ProcessLookupError:
                return False
            except PermissionError:
                return True
            except OSError:
                return False
            return True
        try:
            return bool(_pid_exists(pid))
        except Exception:
            return False
    if started_at is None:
        return True
    try:
        return int(current_start) == int(started_at)
    except (TypeError, ValueError):
        return True


def compute_obligation_id(session_key: str, message_ref: str, content: str) -> str:
    """Stable id: same turn + same content re-records idempotently, while distinct
    threads/topics on one chat never collide (session_key carries platform, chat and
    thread; ``message_ref`` is the triggering inbound message id)."""
    payload = f"{session_key}|{message_ref}|{content}"
    return hashlib.sha256(payload.encode("utf-8", "replace")).hexdigest()[:24]


def record_obligation(
    *,
    obligation_id: str,
    session_key: str,
    platform: str,
    chat_id: str,
    thread_id: Optional[str],
    content: str,
    adapter_profile: Optional[str] = None,
) -> None:
    """Record a final response as owed to the platform (state='pending')."""
    now = time.time()
    stored_profile = str(adapter_profile).strip() if adapter_profile else "default"
    pid, started = _owner_stamp()
    with _DB_LOCK, _transaction() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO delivery_obligations
               (obligation_id, session_key, platform, chat_id, thread_id,
                content, state, attempts, created_at, updated_at,
                owner_pid, owner_started_at, adapter_profile)
               VALUES (?, ?, ?, ?, ?, ?, 'pending', 0, ?, ?, ?, ?, ?)""",
            (obligation_id, session_key, platform, str(chat_id),
             str(thread_id) if thread_id else None, content, now, now,
             pid, started, stored_profile),
        )
    _prune()


def mark_attempting(obligation_id: str) -> None:
    _update_state(obligation_id, "attempting")


def mark_delivered(obligation_id: str) -> None:
    _update_state(obligation_id, "delivered")


def mark_failed(obligation_id: str, error: str = "") -> None:
    _update_state(obligation_id, "failed", error=error)


def release_runtime_claim(obligation_id: str, error: str = "") -> bool:
    """Return an unsent runtime claim to ``failed`` without spending an attempt.

    Runtime recovery claims before clearing ``resume_pending`` so two reconnect
    paths cannot send the same row. If the session flag cannot be cleared, no
    send was attempted and the claim must not consume the redelivery budget.
    Fail-closed to the exact current process instance and ``attempting`` state.
    """
    pid, started = _owner_stamp()
    if started is None:
        return False
    with _DB_LOCK, _transaction() as conn:
        cursor = conn.execute(
            """UPDATE delivery_obligations
               SET state='failed', attempts=CASE
                       WHEN attempts > 0 THEN attempts - 1 ELSE 0 END,
                   updated_at=?, last_error=?
               WHERE obligation_id=? AND state='attempting'
                 AND owner_pid IS ? AND owner_started_at IS ?""",
            (time.time(), error[:500] if error else None,
             obligation_id, pid, started),
        )
    return bool(cursor.rowcount)


def _update_state(obligation_id: str, state: str, error: str = "") -> None:
    with _DB_LOCK, _transaction() as conn:
        conn.execute(
            """UPDATE delivery_obligations
               SET state=?, updated_at=?, last_error=?
               WHERE obligation_id=?""",
            (state, time.time(), error[:500] if error else None, obligation_id),
        )


def _exhausted(attempts: int, created_at: float, now: float) -> bool:
    return attempts >= MAX_ATTEMPTS or (now - created_at) > STALE_AFTER_SECONDS


def sweep_recoverable(
    now: Optional[float] = None,
    *,
    deliverable_platforms: Optional[set] = None,
    deliverable_targets: Optional[set] = None,
) -> List[Dict[str, Any]]:
    """Claim undelivered rows owned by dead processes; return them for redelivery.

    Claiming atomically re-stamps the owner to THIS process and increments
    ``attempts`` (the UPDATE is guarded on the previous owner stamp, so a second
    gateway racing the same sweep cannot double-claim). Rows over the attempts
    cap or stale cutoff transition to 'abandoned' instead of being returned.

    ``deliverable_platforms`` restricts claiming to platforms the caller can
    actually send on this boot: ``attempts`` is the redelivery budget and must
    only be spent on a real send, else a platform that failed to connect burns
    one attempt per boot and hits the cap having never been sent once. Rows for
    absent platforms are left untouched (the stale cutoff still bounds them).

    ``deliverable_targets`` further scopes multiplexed gateways by exact
    ``(platform, adapter_profile)`` so one connected bot cannot spend another
    disconnected bot's retry budget.
    """
    now = now if now is not None else time.time()
    pid, started = _owner_stamp()
    claimed: List[Dict[str, Any]] = []
    with _DB_LOCK, _transaction() as conn:
        rows = conn.execute(
            """SELECT obligation_id, session_key, platform, chat_id, thread_id,
                      content, state, attempts, created_at,
                      owner_pid, owner_started_at, adapter_profile
               FROM delivery_obligations
               WHERE state IN ('pending', 'attempting', 'failed')"""
        ).fetchall()
        for (oid, session_key, platform, chat_id, thread_id, content, state,
             attempts, created_at, owner_pid, owner_started_at,
             adapter_profile) in rows:
            if _owner_alive(owner_pid, owner_started_at):
                continue  # a live gateway still owns this row
            if _exhausted(attempts, created_at, now):
                conn.execute(
                    """UPDATE delivery_obligations
                       SET state='abandoned', updated_at=? WHERE obligation_id=?""",
                    (now, oid),
                )
                continue
            if deliverable_platforms is not None and platform not in deliverable_platforms:
                continue  # no adapter this boot — claiming would spend an attempt on a no-op
            if deliverable_targets is not None and (platform, adapter_profile) not in deliverable_targets:
                continue
            cursor = conn.execute(
                """UPDATE delivery_obligations
                   SET owner_pid=?, owner_started_at=?, attempts=attempts+1,
                       updated_at=?
                   WHERE obligation_id=? AND (owner_pid IS ? OR owner_pid=?)""",
                (pid, started, now, oid, owner_pid, owner_pid),
            )
            if cursor.rowcount:
                claimed.append({
                    "obligation_id": oid,
                    "session_key": session_key,
                    "platform": platform,
                    "chat_id": chat_id,
                    "thread_id": thread_id,
                    "content": content,
                    # pending = never started, redeliver plainly; else carry marker.
                    "needs_marker": state != "pending",
                    "profile": adapter_profile,
                    "attempts": attempts + 1,
                })
    return claimed


def sweep_failed_for_runtime(
    platform: str,
    now: Optional[float] = None,
    *,
    profile: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Claim this process's reconnect-retryable failed rows for one adapter.

    ``profile`` scopes multiplexed gateways to the bot identity that owned the
    failed send; ``None`` means the primary/default adapter. The persisted
    adapter owner is independent of the routed session namespace. Unowned rows
    and rows owned by another process are left for the startup/dead-owner sweep.

    Startup recovery ignores rows owned by a live gateway, so a final response
    rejected with ``send_path_degraded`` would stay stranded when only the
    adapter reconnects. This sweep closes that gap without weakening ownership:
    only rows stamped to this exact process instance, only allowlisted transient
    errors, same attempts/staleness bounds, every update guarded by the prior
    owner stamp and ``failed`` state. Claimed rows always carry the reconnect
    marker because the failed send's acknowledgement is not safe to infer.
    """
    now = now if now is not None else time.time()
    pid, started = _owner_stamp()
    if started is None:
        # PID alone cannot distinguish this process from a stale row left after
        # PID reuse. Runtime replay is optional, so fail closed; startup
        # recovery remains the durable fallback.
        return []
    expected_profile = "default" if not profile or profile == "default" else str(profile)
    claimed: List[Dict[str, Any]] = []
    with _DB_LOCK, _transaction() as conn:
        rows = conn.execute(
            """SELECT obligation_id, session_key, platform, chat_id, thread_id,
                      content, attempts, created_at, owner_pid,
                      owner_started_at, last_error, adapter_profile
               FROM delivery_obligations
               WHERE state='failed' AND platform=?""",
            (platform,),
        ).fetchall()
        for (oid, session_key, row_platform, chat_id, thread_id, content,
             attempts, created_at, owner_pid, owner_started_at, last_error,
             adapter_profile) in rows:
            if adapter_profile != expected_profile:
                continue
            # Exact process-start matching prevents PID reuse from stealing work.
            if owner_pid != pid or owner_started_at != started:
                continue
            if str(last_error or "").strip().lower() not in _RUNTIME_RETRYABLE_ERRORS:
                continue
            owner_guard = (oid, owner_pid, owner_started_at)
            if _exhausted(attempts, created_at, now):
                conn.execute(
                    """UPDATE delivery_obligations
                       SET state='abandoned', updated_at=?
                       WHERE obligation_id=? AND state='failed'
                         AND owner_pid IS ? AND owner_started_at IS ?""",
                    (now, *owner_guard),
                )
                continue
            cursor = conn.execute(
                """UPDATE delivery_obligations
                   SET state='attempting', attempts=attempts+1, updated_at=?
                   WHERE obligation_id=? AND state='failed'
                     AND owner_pid IS ? AND owner_started_at IS ?""",
                (now, *owner_guard),
            )
            if cursor.rowcount:
                claimed.append({
                    "obligation_id": oid,
                    "session_key": session_key,
                    "platform": row_platform,
                    "chat_id": chat_id,
                    "thread_id": thread_id,
                    "content": content,
                    "needs_marker": True,
                    "marker": RECONNECTED_MARKER,
                    "profile": adapter_profile,
                    "runtime_recovery": True,
                    "attempts": attempts + 1,
                })
    return claimed


def _prune(now: Optional[float] = None) -> None:
    now = now if now is not None else time.time()
    cutoff = now - _RETENTION_SECONDS
    try:
        with _transaction() as conn:
            conn.execute(
                """DELETE FROM delivery_obligations
                   WHERE state IN ('delivered', 'abandoned') AND updated_at < ?""",
                (cutoff,),
            )
            total = conn.execute("SELECT COUNT(*) FROM delivery_obligations").fetchone()[0]
            excess = max(0, total - _MAX_ROWS)
            if excess:
                conn.execute(
                    """DELETE FROM delivery_obligations WHERE obligation_id IN (
                         SELECT obligation_id FROM delivery_obligations
                         ORDER BY CASE state
                                    WHEN 'delivered' THEN 0
                                    WHEN 'abandoned' THEN 1
                                    ELSE 2
                                  END, updated_at ASC
                         LIMIT ?)""",
                    (excess,),
                )
    except Exception:
        logger.debug("delivery ledger prune failed", exc_info=True)


def ledger_enabled(config: Optional[Dict[str, Any]] = None) -> bool:
    """Read the ``gateway.delivery_ledger`` config gate (default on)."""
    try:
        if config is None:
            from hermes_cli.config import load_config

            config = load_config()
        gw = config.get("gateway") or {}
        value = gw.get("delivery_ledger", True)
        if isinstance(value, str):
            return value.strip().lower() not in {"false", "0", "no", "off"}
        return bool(value)
    except Exception:
        return True
