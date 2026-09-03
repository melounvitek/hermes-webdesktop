"""Disk-usage rollup for ``/api/status``.

Companion to :mod:`gateway.memory_status`: a hosted agent can fill its data
volume (SQLite writes failing, config saves lost) while its dashboard looks
healthy.  Sampled live via one ``statvfs`` call, so there is no ``sampled_at``.
``/api/status`` is unauthenticated: only coarse numbers (MB, one-decimal
percent) and an enum.  Best-effort: an unreadable filesystem degrades to
``pressure="unknown"`` rather than raising into the status endpoint.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

from gateway.memory_status import _nonneg_int

logger = logging.getLogger(__name__)

# Percent alone misleads both ways: 90% used on 100 GB leaves 10 GB, while 50%
# on a tiny volume is one download from write failures.  Percent triggers are
# gated on absolute headroom also being low, and a hard absolute floor applies
# regardless of size (below it SQLite journaling / config writes are at risk).
_CRITICAL_FREE_MB = 256  # < 256 MB free: critical on any volume
_CRITICAL_PERCENT = 95.0  # >= 95% used AND < 1 GB free: critical
_CRITICAL_HEADROOM_MB = 1024
_ELEVATED_FREE_MB = 512  # < 512 MB free: elevated on any volume
_ELEVATED_PERCENT = 85.0  # >= 85% used AND < 4 GB free: elevated
_ELEVATED_HEADROOM_MB = 4096

_BYTES_PER_MB = 1024 * 1024


def classify_disk_pressure(free_mb: Any, total_mb: Any) -> str:
    """Map free/total MB to ``ok``/``elevated``/``critical``.

    ``unknown`` when the sample is missing or malformed — the caller must
    not treat "we could not read it" as "disk is fine".
    """
    free = _nonneg_int(free_mb)
    total = _nonneg_int(total_mb)
    if free is None or not total:
        return "unknown"
    used_percent = (1 - free / total) * 100.0
    for level, free_floor, percent_floor, headroom in (
        ("critical", _CRITICAL_FREE_MB, _CRITICAL_PERCENT, _CRITICAL_HEADROOM_MB),
        ("elevated", _ELEVATED_FREE_MB, _ELEVATED_PERCENT, _ELEVATED_HEADROOM_MB),
    ):
        if free < free_floor or (used_percent >= percent_floor and free < headroom):
            return level
    return "ok"


def collect_disk_status(home: Optional[Path] = None) -> Dict[str, Any]:
    """Build the ``disk`` block for ``/api/status``.

    ``home`` scopes the sample to a profile's HERMES_HOME (same contract as the
    ``memory`` block).  Always returns a dict and never raises — an unreadable
    or unmounted filesystem yields ``{"pressure": "unknown", ...}``.
    """
    status: Dict[str, Any] = {"pressure": "unknown", "total_mb": None, "free_mb": None, "used_percent": None}
    try:
        if home is None:
            from hermes_constants import get_hermes_home

            home = get_hermes_home()
        usage = shutil.disk_usage(home)
    except Exception:
        return status
    if usage.total <= 0:
        return status
    total_mb = usage.total // _BYTES_PER_MB
    free_mb = usage.free // _BYTES_PER_MB
    status.update(
        total_mb=total_mb,
        free_mb=free_mb,
        used_percent=round((usage.used / usage.total) * 100, 1),
        pressure=classify_disk_pressure(free_mb, total_mb),
    )
    return status
