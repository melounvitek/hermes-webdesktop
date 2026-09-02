"""Flush pending messages and agent transcripts to disk before shutdown to prevent data loss.

When FTS5 index corruption prevents ``INSERT INTO messages``, the gateway
accumulates messages in ``_pending_messages`` (memory-only) and the live
``agent._session_messages`` cannot be flushed via ``_flush_messages_to_session_db``.
On shutdown, ``.clear()`` would discard the only surviving copy.

Hooks (all write atomic JSON payloads under ``<hermes_home>/pending_messages/``):

* ``flush_pending_to_file()`` / ``flush_overflow_to_file()`` — called BEFORE
  ``_pending_messages.clear()`` during shutdown (queue head / FIFO tail).
* ``recover_pending_to_db()`` — called AFTER ``runner.start()`` on startup;
  replays payloads via ``SessionDB.append_message`` (so FTS indexing, session
  metadata and display_kind are handled), deleting each file on success.
* ``flush_agent_history_to_file()`` — called from ``_finalize_shutdown_agents``
  when ``_flush_messages_to_session_db`` raises; dumps ``agent._session_messages``.
* ``spool_dropped_transcript_message()`` / ``drain_transcript_spool()`` — live
  spool for transcript messages evicted by the in-memory pending cap.
"""

from __future__ import annotations

import itertools
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Reason tag for transcript messages dropped by the in-memory pending cap during
# live operation. Payloads carry the full transcript message dict for verbatim replay.
TRANSCRIPT_CAP_DROP_REASON = "transcript_cap_drop"

# Monotonic tiebreaker so same-second spool files replay in drop order.
_TRANSCRIPT_SPOOL_SEQ = itertools.count()


def _get_flush_dir():
    """Return the pending-messages flush directory under the active HERMES_HOME."""
    from hermes_constants import get_hermes_home

    flush_dir = get_hermes_home() / "pending_messages"
    flush_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    if os.name == "posix":
        os.chmod(flush_dir, 0o700)
    return flush_dir


def _fsync_directory(path: Path) -> None:
    """Persist a directory entry on platforms that support directory fsync."""
    if os.name != "posix":
        return
    directory_fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _write_payload(flush_dir: Path, payload: Dict[str, Any]) -> Path:
    """Atomically write one private, uniquely named recovery payload; return its path."""
    from utils import atomic_json_write

    final_path = flush_dir / f"pending-{uuid.uuid4().hex}.json"
    atomic_json_write(final_path, payload, mode=0o600, default=str)
    try:
        _fsync_directory(flush_dir)
    except OSError as exc:
        # The published file is still the only recovery copy; keep it even if
        # this filesystem cannot persist directory entries.
        logger.debug("Failed to fsync pending-message directory: %s", exc)
    return final_path


def _flush_value(flush_dir: Path, kind: str, session_key: str, value: Any, **extra: Any) -> bool:
    """Serialise and write one pending value; return True when a payload was written."""
    try:
        serialised = _serialise_value(value)
        if serialised is None:
            return False
        _write_payload(flush_dir, {"session_key": session_key, **extra, "data": serialised})
        return True
    except Exception as exc:
        logger.debug("Failed to flush %s message for %s: %s", kind, session_key, exc)
        return False


def flush_pending_to_file(
    pending: Dict[str, Any],
    *,
    reason: str = "shutdown",
) -> int:
    """Serialise non-empty ``_pending_messages`` slots to disk.

    ``pending`` values may be ``MessageEvent`` objects (adapter) or plain
    strings (runner).  Returns the number of sessions flushed.
    """
    if not pending:
        return 0

    flush_dir = _get_flush_dir()
    ts = int(time.time())
    flushed = 0
    for session_key, value in list(pending.items()):
        if value is None:
            continue
        flushed += _flush_value(flush_dir, "pending", session_key, value, reason=reason, ts=ts)

    if flushed:
        logger.info("Flushed %d pending message(s) to %s (reason=%s)", flushed, flush_dir, reason)
    return flushed


def flush_overflow_to_file(
    overflow_by_session: Dict[str, Any],
    *,
    reason: str = "shutdown",
) -> int:
    """Serialise the FIFO overflow tails (``queued_events``) to disk.

    The adapter slot holds the queue head and ``SessionState.conversation
    .queued_events`` holds the tail; both must survive restart.  Each event
    is its own payload in the slot-flush shape so ``recover_pending_to_db``
    replays them unchanged; ``seq`` preserves arrival order within a session.
    Returns the number of events flushed.
    """
    if not overflow_by_session:
        return 0

    flush_dir = _get_flush_dir()
    ts = int(time.time())
    flushed = 0
    for session_key, events in list(overflow_by_session.items()):
        if not session_key or not events:
            continue
        for seq, value in enumerate(list(events)):
            if value is None:
                continue
            flushed += _flush_value(flush_dir, "overflow", session_key, value, reason=reason, ts=ts, seq=seq)

    if flushed:
        logger.info("Flushed %d queued overflow message(s) to %s (reason=%s)", flushed, flush_dir, reason)
    return flushed


def spool_dropped_transcript_message(session_id: str, message: Dict[str, Any]) -> Optional[Path]:
    """Spool a transcript message evicted by the runtime pending cap.

    Returns the written spool path, or ``None`` when spooling failed —
    callers must degrade to drop-and-log.
    """
    try:
        return _write_payload(
            _get_flush_dir(),
            {
                "session_key": session_id,
                "reason": TRANSCRIPT_CAP_DROP_REASON,
                "ts": int(time.time()),
                "seq": next(_TRANSCRIPT_SPOOL_SEQ),
                "data": {"session_id": session_id, "message": message},
            },
        )
    except Exception as exc:
        logger.debug("Failed to spool cap-dropped transcript message for %s: %s", session_id, exc)
        return None


def drain_transcript_spool(session_id: str, replay) -> tuple[int, int]:
    """Replay cap-dropped transcript messages spooled for *session_id*.

    ``replay(message_dict)`` runs per message in drop order; a spool file is
    deleted only after its replay succeeds.  The first failure stops the
    drain (the DB is likely still unhealthy) and keeps the rest for retry.
    Returns ``(replayed, remaining)``.
    """
    try:
        candidates = list(_get_flush_dir().glob("pending-*.json"))
    except Exception as exc:
        logger.debug("Cannot scan transcript spool: %s", exc)
        return 0, 0

    entries = []
    for path in candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if (
            payload.get("reason") != TRANSCRIPT_CAP_DROP_REASON
            or payload.get("session_key") != session_id
        ):
            continue
        message = (payload.get("data") or {}).get("message")
        if not isinstance(message, dict):
            logger.warning("Removing structurally invalid transcript spool file %s", path)
            path.unlink(missing_ok=True)
            continue
        entries.append((payload.get("ts", 0), payload.get("seq", 0), path.name, path, message))

    ordered = sorted(entries, key=lambda e: e[:3])
    replayed = remaining = 0
    for idx, (_ts, _seq, _name, path, message) in enumerate(ordered):
        try:
            replay(message)
        except Exception as exc:
            logger.warning(
                "Replay of spooled transcript message %s for %s failed; "
                "keeping spool file for retry: %s",
                path, session_id, exc,
            )
            remaining = len(ordered) - idx
            break
        path.unlink(missing_ok=True)
        replayed += 1

    if replayed:
        logger.info("Replayed %d spooled transcript message(s) for %s after DB recovery", replayed, session_id)
    return replayed, remaining


def _serialise_value(value: Any) -> Optional[dict]:
    """Convert a pending message value to a JSON-serialisable dict."""
    if hasattr(value, "text"):  # MessageEvent-like object
        result: Dict[str, Any] = {"text": getattr(value, "text", "")}
        for attr in ("session_id", "platform", "sender_id", "sender_name",
                      "reply_to", "media", "raw_event"):
            val = getattr(value, attr, None)
            if val is not None:
                try:
                    json.dumps(val)
                    result[attr] = val
                except (TypeError, ValueError):
                    result[attr] = str(val)
        return result
    if isinstance(value, str):  # runner-level _pending_messages
        return {"text": value}
    if isinstance(value, dict):
        try:
            json.dumps(value)
            return value
        except (TypeError, ValueError):
            return {"text": str(value)}
    return {"text": str(value)}


def recover_pending_to_db(session_db=None) -> int:
    """Recover flushed pending messages into state.db via SessionDB.

    Reads all ``*.json`` files from the flush directory, inserts messages via
    ``SessionDB.append_message`` and deletes each flush file on success.
    ``session_db=None`` opens (and afterwards releases) the shared default
    ``state.db``.  Returns the number of messages recovered.
    """
    flush_dir = _get_flush_dir()
    flush_files = sorted(flush_dir.glob("*.json"))
    if not flush_files:
        return 0

    own_db = session_db is None
    if own_db:
        from hermes_state import get_shared_session_db
        session_db = get_shared_session_db()

    recovered = 0
    try:
        for path in flush_files:
            payload = json.loads(path.read_text(encoding="utf-8"))
            # Agent-history snapshots (reason + messages list) are for manual
            # operator recovery, not automatic DB insertion.
            if payload.get("reason") == "shutdown-with-unpersisted-agent-history":
                continue
            # Cap-dropped transcript payloads carry the full message dict keyed
            # by session_id — replay directly (spool never drained before restart).
            if payload.get("reason") == TRANSCRIPT_CAP_DROP_REASON:
                data = payload.get("data", {}) or {}
                spooled_sid = data.get("session_id", "")
                message = data.get("message")
                if not spooled_sid or not isinstance(message, dict):
                    logger.warning(
                        "Cannot recover structurally invalid transcript spool "
                        "file %s; preserved for manual inspection",
                        path,
                    )
                    continue
                session_db.append_message(
                    session_id=spooled_sid,
                    role=message.get("role", "unknown"),
                    content=message.get("content") or "",
                    timestamp=message.get("timestamp") or payload.get("ts"),
                )
                recovered += 1
                path.unlink(missing_ok=True)
                continue
            session_key = payload.get("session_key", "")
            data = payload.get("data", {})
            text = data.get("text", "")
            if not text or not session_key:
                logger.warning(
                    "Cannot recover structurally invalid pending message from %s; "
                    "the flush file has been preserved",
                    path,
                )
                continue

            # session_key is a gateway routing key (e.g. "agent:main:telegram:...");
            # appending a row needs the real session_id, which only the
            # serialised data can supply at this recovery stage.
            session_id = data.get("session_id", "")
            if not session_id:
                logger.warning(
                    "Cannot recover pending message for %s: no session_id "
                    "in flush file and session_key-to-id resolution is not "
                    "available at this recovery stage. The message text is "
                    "preserved in %s",
                    session_key, path,
                )
                continue

            session_db.append_message(
                session_id=session_id, role="user", content=text,
                timestamp=payload.get("ts", int(time.time())),
            )
            recovered += 1
            path.unlink(missing_ok=True)
    finally:
        # Shutdown cancellation/interrupt must not strand an owned DB.
        if own_db:
            try:
                from hermes_state import release_or_close
                release_or_close(session_db)
            except Exception:
                pass

    if recovered:
        logger.info("Recovered %d pending message(s) from shutdown flush", recovered)
    return recovered


def flush_agent_history_to_file(session_id: Optional[str], history: list) -> None:
    """Best-effort dump of an agent's in-memory transcript before teardown.

    Used when ``_flush_messages_to_session_db`` raises (e.g. FTS/SQLite
    corruption): the transcript is written outside the broken DB so an
    operator can salvage it after repairing state.db.  Failures are
    swallowed — shutdown must never block on a best-effort backup.
    """
    if not history:
        return
    try:
        flush_dir = _get_flush_dir()
        snapshot = []
        for _m in history:
            try:
                snapshot.append(
                    _m if isinstance(_m, (dict, list, str, int, float, bool, type(None)))
                    else str(_m)
                )
            except Exception:
                continue
        _write_payload(flush_dir, {
            "reason": "shutdown-with-unpersisted-agent-history",
            "issue": "#72680",
            "session_id": session_id,
            "count": len(snapshot),
            "messages": snapshot,
        })
        logger.warning(
            "Preserved %d in-memory message(s) for session %s "
            "(possible FTS corruption — recover after repairing state.db)",
            len(snapshot), session_id,
        )
    except Exception as _e:
        logger.warning("Agent-history shutdown preservation failed for session %s: %s", session_id, _e)
