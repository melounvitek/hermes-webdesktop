"""Shared leaf helpers for the gateway hosted-room modules.

Every hosted-room module validates identifiers, bounded integers, exact field
sets and canonical JSON with its own error class and its own error strings
(tests pin those strings). These helpers take the error class and message
templates as parameters so each caller keeps byte-identical failures while the
logic lives once. This module must stay a leaf: never import a hosted_room*
origin module from here (import cycle).
"""

from __future__ import annotations

import json
import re
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")


def identifier(
    value: Any,
    *,
    label: str,
    error: type[Exception],
    max_chars: int = 128,
    pattern: re.Pattern[str] = IDENTIFIER_RE,
    invalid: str | None = None,
) -> str:
    """Strip and validate a bounded identifier; ``invalid`` overrides the fail message."""
    if not isinstance(value, str):
        raise error(f"{label} must be a string")
    value = value.strip()
    if not value or len(value) > max_chars or not pattern.fullmatch(value):
        raise error(invalid or f"invalid {label}")
    return value


def positive_int(value: Any, *, error: type[Exception], message: str) -> int:
    """Reject bools, non-ints and values below 1 (``message`` is the exact error text)."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise error(message)
    return value


def non_negative_int(value: Any, *, error: type[Exception], message: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise error(message)
    return value


def exact_fields(
    value: Any,
    *,
    label: str,
    required: frozenset[str] | set[str],
    optional: frozenset[str] | set[str] = frozenset(),
    error: type[Exception],
    not_object: str | None = None,
    missing_fmt: str = "{label} is missing fields: {fields}",
    unknown_fmt: str = "{label} has unknown fields: {fields}",
) -> Mapping[str, Any]:
    """Require exactly ``required`` (+ any ``optional``) keys; formats name the offenders sorted."""
    if not isinstance(value, Mapping):
        raise error(not_object or f"{label} must be an object")
    keys = frozenset(value)
    missing = required - keys
    unknown = keys - required - optional
    if missing:
        raise error(missing_fmt.format(label=label, fields=", ".join(sorted(missing))))
    if unknown:
        raise error(unknown_fmt.format(label=label, fields=", ".join(sorted(unknown))))
    return value


def canonical_json(
    value: Any,
    *,
    error: type[Exception],
    label: str,
    max_bytes: int,
    ensure_ascii: bool,
) -> str:
    """Sorted, compact JSON bounded by ``max_bytes`` of UTF-8."""
    try:
        encoded = json.dumps(
            value, ensure_ascii=ensure_ascii, sort_keys=True, separators=(",", ":")
        )
    except (TypeError, ValueError, RecursionError) as exc:
        raise error(f"{label} must be JSON-serializable") from exc
    if len(encoded.encode("utf-8")) > max_bytes:
        raise error(f"{label} is too large")
    return encoded


def utf8_len(*parts: str) -> int:
    return len("".join(parts).encode("utf-8"))


def open_sqlite(path: Path | str, *, timeout: float = 10) -> sqlite3.Connection:
    """Row-factory connection with foreign keys on; no journal or schema work."""
    conn = sqlite3.connect(path, timeout=timeout)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone()
    return row is not None


def table_columns(conn: sqlite3.Connection, table: str) -> frozenset[str]:
    return frozenset(row[1] for row in conn.execute(f"PRAGMA table_info({table})"))


@contextmanager
def transaction(
    connect: Callable[[Path | str], sqlite3.Connection],
    db_path: Path | str,
    *,
    immediate: bool,
) -> Iterator[sqlite3.Connection]:
    """Open via ``connect``, optionally ``BEGIN IMMEDIATE``, commit on success, always close."""
    conn = connect(db_path)
    try:
        if immediate:
            conn.execute("BEGIN IMMEDIATE")
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
