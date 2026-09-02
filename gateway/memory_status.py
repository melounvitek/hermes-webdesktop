"""Memory status rollup for ``/api/status``.

Read side for memory-pressure signals the gateway already persists but only
logs: the 30s ``state/gateway.heartbeat`` (RSS + MemAvailable/MemTotal + swap,
from :func:`gateway.shutdown_watchdog.write_loop_heartbeat`) and the lifecycle
sentinel's ``suspected_oom`` flag (:func:`gateway.lifecycle_ledger.record_startup`).
Distills them into a compact block the dashboard SPA and the NAS availability
sweep consume — no new sampling, no IPC, two small file reads.

Public-safety: ``/api/status`` is unauthenticated (``PUBLIC_API_PATHS``), so this
block carries only coarse numbers (MB granularity), enums, and booleans — the
same disclosure class as ``active_agents`` / ``nous_session_valid``.

Best-effort and read-only: a missing/corrupt file degrades to
``pressure="unknown"`` rather than raising into the status endpoint.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Thresholds on system MemAvailable.  ``critical`` mirrors the lifecycle ledger's
# OOM-suspicion heuristics (_LOW_MEM_AVAILABLE_KIB / _LOW_MEM_AVAILABLE_FRACTION):
# a level that would make a later unclean death "suspected OOM" should already
# warn while the process is alive.
_CRITICAL_AVAILABLE_KIB = 64 * 1024  # < 64 MiB available
_CRITICAL_AVAILABLE_FRACTION = 0.05  # < 5% of MemTotal
_ELEVATED_AVAILABLE_KIB = 128 * 1024  # < 128 MiB available
_ELEVATED_AVAILABLE_FRACTION = 0.15  # < 15% of MemTotal

# Writer cadence is 30s (DEFAULT_HEARTBEAT_INTERVAL_S); 150s tolerates a briefly
# stalled loop without letting a long-dead gateway's last sample pose as current.
_HEARTBEAT_FRESH_TTL_S = 150.0

_KIB_PER_MB = 1024


def _nonneg_int(value: Any) -> Optional[int]:
    """Return *value* if it is a non-negative int (bools rejected), else None."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _mb(kib: Any) -> Optional[int]:
    kib = _nonneg_int(kib)
    return None if kib is None else kib // _KIB_PER_MB


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


def _read_heartbeat(home: Optional[Path]) -> Optional[Dict[str, Any]]:
    try:
        from gateway.lifecycle_ledger import _read_json
        from gateway.shutdown_watchdog import get_loop_heartbeat_path

        return _read_json(get_loop_heartbeat_path(home))
    except Exception:
        return None


def _read_sentinel(home: Optional[Path]) -> Optional[Dict[str, Any]]:
    try:
        from gateway.lifecycle_ledger import _read_json, get_lifecycle_sentinel_path

        return _read_json(get_lifecycle_sentinel_path(home))
    except Exception:
        return None


def collect_memory_status(
    home: Optional[Path] = None,
    *,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Build the ``memory`` block for ``/api/status``.

    ``home`` scopes the read to a profile's HERMES_HOME (``None`` = active
    profile); ``now`` is injectable for tests.  Always returns a dict and never
    raises — a down/never-started gateway or corrupt files yield
    ``{"pressure": "unknown", ...}`` plus whatever fields could be recovered.
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
        # Identity of the CURRENT gateway life (sentinel started_at) — changes on
        # every boot, so the dashboard can key banner dismissal on it and
        # acknowledging one OOM restart does not mute the NEXT one.
        "boot_id": None,
    }

    heartbeat = _read_heartbeat(home)
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

    sentinel = _read_sentinel(home)
    if sentinel:
        status["last_boot_unclean"] = bool(sentinel.get("prior_unclean_exit"))
        status["last_boot_suspected_oom"] = bool(sentinel.get("prior_suspected_oom"))
        started_at = sentinel.get("started_at")
        if isinstance(started_at, str) and started_at:
            status["boot_id"] = started_at

    return status
