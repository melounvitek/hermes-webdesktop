"""Transcript repair for SessionDB batch appends: reconcile in-memory assistant
rows with committed SQLite rows (blank-row in-place update, concurrent-winner
adoption, watermark-compaction clone lookup) and sync markers after commit.
"""

from __future__ import annotations

import sqlite3
from typing import Any, Callable, Dict, List

from agent.context_compressor import _DB_PERSISTED_MARKER


def is_content_blank(content: Any) -> bool:
    """True when decoded message content is None, whitespace-only, or has no visible text parts."""
    if content is None:
        return True
    if isinstance(content, str):
        return not content.strip()
    if isinstance(content, list):
        if not content:
            return True
        return not "".join(
            p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"
        ).strip()
    return False


def resolve_and_repair_transcript_batch(
    conn: sqlite3.Connection,
    session_id: str,
    messages: List[Dict[str, Any]],
    encode_content_fn: Callable[[Any], Any],
    decode_content_fn: Callable[[Any], Any],
) -> List[Dict[str, Any]]:
    """Partition a message batch within an active write transaction.

    An assistant message carrying an existing integer ``_row_id`` targets its active SQLite row (or
    the active clone a watermark compaction made of it): a blank row is updated in place; a
    non-blank one (concurrent winner) has its canonical content adopted without overwrite.
    Returns the messages that must be inserted as fresh rows.
    """
    inserted_rows: List[Dict[str, Any]] = []
    for msg in messages:
        repaired = False
        existing_row_id = msg.get("_row_id") if isinstance(msg, dict) else None
        if isinstance(existing_row_id, int) and msg.get("role", "unknown") == "assistant":
            row = conn.execute(
                "SELECT id, role, active, timestamp, content FROM messages "
                "WHERE id = ? AND session_id = ?",
                (existing_row_id, session_id),
            ).fetchone()
            target_row = None
            if row is not None and row["role"] == "assistant":
                if int(row["active"] or 0) == 1:
                    target_row = row
                else:
                    # Watermark compaction soft-archived the concurrent tail and cloned it.
                    target_row = conn.execute(
                        "SELECT id, role, active, timestamp, content FROM messages "
                        "WHERE session_id = ? AND active = 1 AND role = 'assistant' "
                        "AND timestamp IS ? AND id != ? "
                        "ORDER BY id DESC LIMIT 1",
                        (session_id, row["timestamp"], row["id"]),
                    ).fetchone()
            if target_row is not None:
                target_id = int(target_row["id"])
                decoded = decode_content_fn(target_row["content"])
                msg["_row_id"] = target_id
                if is_content_blank(decoded):
                    conn.execute(
                        "UPDATE messages SET content = ? "
                        "WHERE id = ? AND session_id = ? AND active = 1",
                        (encode_content_fn(msg.get("content")), target_id, session_id),
                    )
                else:
                    msg["_canonical_content"] = decoded  # concurrent winner: adopt, don't overwrite
                repaired = True
        if not repaired:
            inserted_rows.append(msg)
    return inserted_rows


def sync_flushed_message_markers(batch_msgs: List[Dict[str, Any]], batch_rows: List[Dict[str, Any]]) -> None:
    """Stamp _DB_PERSISTED_MARKER and sync canonical row ID / content onto live dicts after commit."""
    for written, row in zip(batch_msgs, batch_rows):
        written[_DB_PERSISTED_MARKER] = True
        if isinstance(row.get("_row_id"), int):
            written["_row_id"] = row["_row_id"]
        if "_canonical_content" in row:
            written["content"] = row["_canonical_content"]
