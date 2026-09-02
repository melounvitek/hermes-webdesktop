"""Shared time/filter parsing for `hermes sessions prune` / `archive`."""

from __future__ import annotations

import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

_DURATION_RE = re.compile(
    r"^(\d+(?:\.\d+)?)\s*"
    r"(s|sec|secs|second|seconds|"
    r"m|min|mins|minute|minutes|"
    r"h|hr|hrs|hour|hours|"
    r"d|day|days|"
    r"w|wk|wks|week|weeks)$"
)

_UNIT_SECONDS = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}


def parse_duration_seconds(value: str) -> Optional[float]:
    """Parse ``5h`` / ``30m`` / ``2d`` / ``1w`` / ``90`` (bare = days) into
    seconds. Returns None when the value doesn't look like a duration."""
    s = str(value).strip().lower()
    if not s:
        return None
    if re.fullmatch(r"\d+(?:\.\d+)?", s):
        # Bare number = days (backward compatible with --older-than 90)
        return float(s) * 86400
    m = _DURATION_RE.match(s)
    if not m:
        return None
    return float(m.group(1)) * _UNIT_SECONDS[m.group(2)[0]]


def parse_point_in_time(value: str, flag: str) -> float:
    """Parse a CLI time value into an epoch timestamp.

    Durations mean "that long ago" (``5h`` = now minus 5 hours); ISO timestamps are taken as-is
    (naive = local time). Raises ``ValueError`` with a user-facing message on bad input.
    """
    s = str(value).strip()
    dur = parse_duration_seconds(s)
    if dur is not None:
        return time.time() - dur
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        raise ValueError(
            f"Invalid value for {flag}: '{value}'. Use a duration like '5h', "
            f"'30m', '2d', '1w', a bare number of days, or an ISO timestamp "
            f"like '2026-07-05' or '2026-07-05 14:30'."
        ) from None
    if dt.tzinfo is None:
        return dt.timestamp()
    return dt.astimezone(timezone.utc).timestamp()


def format_epoch(ts: Optional[float]) -> str:
    """Render an epoch timestamp as a short local-time string."""
    if ts is None:
        return "-"
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")


def build_prune_filters(args: Any) -> Dict[str, Any]:
    """Translate argparse Namespace flags into SessionDB filter kwargs.

    Understands: ``--older-than``, ``--newer-than``, ``--before``, ``--after``, ``--source``,
    ``--title``, ``--end-reason``, ``--cwd``, ``--min-messages``, ``--max-messages``,
    ``--archived``/``--no-archived``.

    ``--older-than`` / ``--newer-than`` bound last activity, while ``--before`` / ``--after``
    explicitly bound session start time. Last activity is the latest message timestamp, falling back
    to ``started_at`` for empty sessions.
    """
    bounds: Dict[str, Optional[float]] = {}
    for key, attr, flag in _TIME_BOUNDS:
        raw = getattr(args, attr, None)
        bounds[key] = None if raw is None else parse_point_in_time(raw, flag)

    for lo, hi, label, lo_flag, hi_flag in _WINDOWS:
        if bounds[hi] is not None and bounds[lo] is not None and bounds[lo] >= bounds[hi]:
            raise ValueError(
                f"Empty {label} window: the {lo_flag} bound "
                f"({format_epoch(bounds[lo])}) is not earlier than the "
                f"{hi_flag} bound ({format_epoch(bounds[hi])})."
            )

    # older_than_days=None: the epoch bounds above are the whole story.
    # Without this, prune_sessions' default 90-day cutoff would silently
    # cap an --after/--newer-than-only window.
    filters: Dict[str, Any] = {"older_than_days": None, **bounds}
    for key, attr in _ARG_FILTERS:
        filters[key] = getattr(args, attr, None)
    return filters


# (filter key, argparse attr, CLI flag) for the four epoch bounds.
_TIME_BOUNDS = (
    ("last_active_before", "older_than", "--older-than"),
    ("last_active_after", "newer_than", "--newer-than"),
    ("started_before", "before", "--before"),
    ("started_after", "after", "--after"),
)
# (lower key, upper key, window label, lower flag, upper flag); checked in this order.
_WINDOWS = (
    ("started_after", "started_before", "start-time", "--after", "--before"),
    ("last_active_after", "last_active_before", "activity", "--newer-than", "--older-than"),
)
_ARG_FILTERS = (
    ("source", "source"),
    ("title_like", "title"),
    ("end_reason", "end_reason"),
    ("cwd_prefix", "cwd"),
    ("min_messages", "min_messages"),
    ("max_messages", "max_messages"),
    ("model_like", "model"),
    ("provider", "provider"),
    ("user_id", "user"),
    ("chat_id", "chat_id"),
    ("chat_type", "chat_type"),
    ("branch_like", "branch"),
    ("min_tokens", "min_tokens"),
    ("max_tokens", "max_tokens"),
    ("min_cost", "min_cost"),
    ("max_cost", "max_cost"),
    ("min_tool_calls", "min_tool_calls"),
    ("max_tool_calls", "max_tool_calls"),
)

# (filter key, description template, include when: "set" == `is not None`, "truthy" == bool(v)).
_DESCRIBE = (
    ("last_active_before", "last active before {e}", "set"),
    ("last_active_after", "last active after {e}", "set"),
    ("started_before", "started before {e}", "set"),
    ("started_after", "started after {e}", "set"),
    ("source", "source '{v}'", "truthy"),
    ("title_like", "title contains '{v}'", "truthy"),
    ("end_reason", "end reason '{v}'", "truthy"),
    ("cwd_prefix", "cwd under '{v}'", "truthy"),
    ("min_messages", ">= {v} messages", "set"),
    ("max_messages", "<= {v} messages", "set"),
    ("model_like", "model contains '{v}'", "truthy"),
    ("provider", "provider '{v}'", "truthy"),
    ("user_id", "user '{v}'", "truthy"),
    ("chat_id", "chat '{v}'", "truthy"),
    ("chat_type", "chat type '{v}'", "truthy"),
    ("branch_like", "git branch contains '{v}'", "truthy"),
    ("min_tokens", ">= {v} tokens", "set"),
    ("max_tokens", "<= {v} tokens", "set"),
    ("min_cost", ">= ${v}", "set"),
    ("max_cost", "<= ${v}", "set"),
    ("min_tool_calls", ">= {v} tool calls", "set"),
    ("max_tool_calls", "<= {v} tool calls", "set"),
)


def describe_filters(filters: Dict[str, Any]) -> str:
    """Human-readable summary of active filters for confirmation prompts."""
    parts = []
    for key, template, mode in _DESCRIBE:
        value = filters.get(key)
        if (value is not None) if mode == "set" else bool(value):
            shown = format_epoch(value) if "{e}" in template else value
            parts.append(template.replace("{e}", "{v}").format(v=shown))
    return ", ".join(parts) if parts else "no filters (all ended sessions)"
