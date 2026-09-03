"""Memory status rollup for ``/api/status``.

Read side for signals the gateway already persists: the 30s
``state/gateway.heartbeat`` (RSS + MemAvailable/MemTotal + swap) and the
lifecycle sentinel's ``suspected_oom`` flag.  Two small file reads, no IPC.
``/api/status`` is unauthenticated, so the block carries only coarse numbers
(MB), enums and booleans.  Best-effort: a missing/corrupt file degrades to
``pressure="unknown"`` rather than raising into the status endpoint.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Thresholds on system MemAvailable.  ``critical`` doubles as the lifecycle
# ledger's OOM-suspicion heuristic: a level that makes a later unclean death
# "suspected OOM" already warns while the process is alive.
_CRITICAL_AVAILABLE_KIB = 64 * 1024  # < 64 MiB available
_CRITICAL_AVAILABLE_FRACTION = 0.05  # < 5% of MemTotal
_ELEVATED_AVAILABLE_KIB = 128 * 1024  # < 128 MiB available
_ELEVATED_AVAILABLE_FRACTION = 0.15  # < 15% of MemTotal

# Writer cadence is 30s; 150s tolerates a briefly stalled loop without letting
# a long-dead gateway's last sample pose as current.
_HEARTBEAT_FRESH_TTL_S = 150.0


def _nonneg_int(value: Any) -> Optional[int]:
    """Return *value* if it is a non-negative int (bools rejected), else None."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _mb(kib: Any) -> Optional[int]:
    kib = _nonneg_int(kib)
    return None if kib is None else kib // 1024


def _parse_iso(value: Any) -> Optional[datetime]:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


def classify_pressure(available_kib: Any, total_kib: Any) -> str:
    """Map a MemAvailable/MemTotal pair to ``ok``/``elevated``/``critical``.

    ``unknown`` when the sample is missing or malformed — the caller must
    not treat "we could not read it" as "memory is fine".
    """
    available = _nonneg_int(available_kib)
    if available is None:
        return "unknown"
    total = _nonneg_int(total_kib)
    fraction = available / total if total else None
    for level, kib_floor, frac_floor in (
        ("critical", _CRITICAL_AVAILABLE_KIB, _CRITICAL_AVAILABLE_FRACTION),
        ("elevated", _ELEVATED_AVAILABLE_KIB, _ELEVATED_AVAILABLE_FRACTION),
    ):
        if available < kib_floor or (fraction is not None and fraction < frac_floor):
            return level
    return "ok"


def _read_state_files(home: Optional[Path]) -> tuple:
    """``(heartbeat, sentinel)`` dicts, each ``None`` when unreadable."""
    try:
        from gateway.lifecycle_ledger import _read_json, get_lifecycle_sentinel_path
        from gateway.shutdown_watchdog import get_loop_heartbeat_path

        return _read_json(get_loop_heartbeat_path(home)), _read_json(get_lifecycle_sentinel_path(home))
    except Exception:
        return None, None


def collect_memory_status(
    home: Optional[Path] = None,
    *,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Build the ``memory`` block for ``/api/status``.

    ``home`` scopes the read to a profile's HERMES_HOME (``None`` = active
    profile); ``now`` is injectable for tests.  Always returns a dict and never
    raises — a down gateway or corrupt files yield ``{"pressure": "unknown", ...}``
    plus whatever fields could be recovered.
    """
    moment = now or datetime.now(timezone.utc)
    status: Dict[str, Any] = {
        "pressure": "unknown",
        "gateway_rss_mb": None,
        "system_total_mb": None,
        "system_available_mb": None,
        "swap_used_mb": None,
        "sampled_at": None,
        "last_boot_unclean": False,
        "last_boot_suspected_oom": False,
        # Identity of the CURRENT life (sentinel started_at): the dashboard keys
        # banner dismissal on it so acknowledging one OOM restart does not mute the NEXT.
        "boot_id": None,
    }

    heartbeat, sentinel = _read_state_files(home)
    if heartbeat:
        sampled_at = _parse_iso(heartbeat.get("updated_at"))
        mem = heartbeat.get("mem")
        if isinstance(mem, dict):
            status["gateway_rss_mb"] = _mb(mem.get("rss_kib"))
            status["system_total_mb"] = _mb(mem.get("mem_total_kib"))
            status["system_available_mb"] = _mb(mem.get("mem_available_kib"))
            status["swap_used_mb"] = _mb(mem.get("swap_used_kib"))
            if sampled_at is not None:
                status["sampled_at"] = sampled_at.isoformat()
                # Stale sample: numbers still reported (sampled_at says when) but
                # pressure stays "unknown" so a dead gateway's final gasp cannot
                # render a live "critical" banner forever.
                if 0 <= (moment - sampled_at).total_seconds() <= _HEARTBEAT_FRESH_TTL_S:
                    status["pressure"] = classify_pressure(mem.get("mem_available_kib"), mem.get("mem_total_kib"))

    if sentinel:
        status["last_boot_unclean"] = bool(sentinel.get("prior_unclean_exit"))
        status["last_boot_suspected_oom"] = bool(sentinel.get("prior_suspected_oom"))
        started_at = sentinel.get("started_at")
        if isinstance(started_at, str) and started_at:
            status["boot_id"] = started_at

    return status
