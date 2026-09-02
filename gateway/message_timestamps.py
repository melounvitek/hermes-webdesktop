"""Helpers for rendering gateway message timestamps exactly once.

Gateway messages need timestamps in the LLM context for temporal awareness, but
persisted message content should stay clean so replay does not accumulate
``[timestamp] [timestamp] ...`` prefixes across turns.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Optional, Tuple


# Current gateway format: [Tue 2026-04-28 13:40:53 CEST]
_HUMAN_TIMESTAMP_RE = re.compile(
    r"^\[(?P<dow>[A-Z][a-z]{2}) "
    r"(?P<date>\d{4}-\d{2}-\d{2}) "
    r"(?P<time>\d{2}:\d{2}:\d{2})"
    r"(?: (?P<tz>[A-Za-z0-9_+\-/:]+))?\]\s*"
)

# Older gateway format: [2026-04-13T17:02:06+0200] or [+02:00]
_ISO_TIMESTAMP_RE = re.compile(r"^\[(?P<iso>\d{4}-\d{2}-\d{2}T[^\]]+)\]\s*")


def _localize(dt: datetime, tz) -> float:
    """Epoch for ``dt``; naive values take ``tz`` if given, else the local zone."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=tz) if tz is not None else dt.astimezone()
    return float(dt.timestamp())


def _parse_iso(text: str, tz=None) -> Optional[float]:
    """Parse an ISO-8601 string (incl. ``+0200`` offsets fromisoformat rejects)."""
    try:
        dt = datetime.fromisoformat(text)
    except (TypeError, ValueError):
        try:
            dt = datetime.strptime(text, "%Y-%m-%dT%H:%M:%S%z")
        except (TypeError, ValueError):
            return None
    return _localize(dt, tz)


def _parse_timestamp_match(match: re.Match, tz=None) -> Optional[float]:
    if match.groupdict().get("iso"):
        return _parse_iso(match.group("iso"), tz)
    try:
        dt = datetime.strptime(f"{match.group('date')} {match.group('time')}", "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None
    return _localize(dt, tz)


def _match_timestamp_prefix(text: str) -> Optional[re.Match]:
    return _HUMAN_TIMESTAMP_RE.match(text) or _ISO_TIMESTAMP_RE.match(text)


def coerce_message_timestamp(ts_value: Any, tz=None) -> Optional[float]:
    """Coerce a timestamp-like value to Unix epoch seconds.

    Accepts epoch numbers, datetime objects, ISO strings, and the gateway's
    bracketed human-readable format. Returns ``None`` when uninterpretable.
    """
    if ts_value is None:
        return None
    if isinstance(ts_value, (int, float)):
        return float(ts_value)
    if hasattr(ts_value, "timestamp"):
        try:
            return float(ts_value.timestamp())
        except Exception:
            return None
    if isinstance(ts_value, str):
        text = ts_value.strip()
        if not text:
            return None
        match = _match_timestamp_prefix(text)
        if match is not None:
            parsed = _parse_timestamp_match(match, tz=tz)
            if parsed is not None:
                return parsed
        try:
            return float(text)
        except (TypeError, ValueError):
            pass
        return _parse_iso(text, tz)
    return None


def format_message_timestamp(ts_value: Any, tz=None) -> str:
    """Format a timestamp value as ``[Tue 2026-04-28 13:40:53 CEST]``."""
    epoch = coerce_message_timestamp(ts_value, tz=tz)
    if epoch is None:
        return ""
    dt = datetime.fromtimestamp(epoch, tz=tz) if tz is not None else datetime.fromtimestamp(epoch).astimezone()
    return "[" + dt.strftime("%a %Y-%m-%d %H:%M:%S %Z") + "]"


def strip_leading_message_timestamps(content: str, tz=None) -> Tuple[str, Optional[float]]:
    """Strip one or more leading gateway timestamp prefixes from ``content``.

    Returns ``(clean_content, embedded_epoch)``. With multiple prefixes the one
    closest to the message text wins, preserving the original platform-send
    time for legacy contaminated rows like ``[processing time] [platform time] [sender] message``.
    """
    if not isinstance(content, str) or not content:
        return content, None
    text = content
    embedded_epoch: Optional[float] = None
    while (match := _match_timestamp_prefix(text)) is not None:
        parsed = _parse_timestamp_match(match, tz=tz)
        if parsed is not None:
            embedded_epoch = parsed
        text = text[match.end():]
    return text, embedded_epoch


def render_user_content_with_timestamp(content: str, ts_value: Any = None, tz=None) -> str:
    """Render a user message for LLM context with exactly one timestamp prefix.

    An existing leading prefix is stripped and its parsed time wins over
    ``ts_value``. If no timestamp is available the cleaned content is returned.
    """
    clean_content, embedded_epoch = strip_leading_message_timestamps(content, tz=tz)
    effective_ts = embedded_epoch if embedded_epoch is not None else ts_value
    prefix = format_message_timestamp(effective_ts, tz=tz)
    if not prefix:
        return clean_content
    return f"{prefix} {clean_content}" if clean_content else prefix
