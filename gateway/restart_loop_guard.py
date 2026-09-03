"""Auto-resume restart-loop breaker (defense-3).

Defenses 1 and 2 (the ``_HERMES_GATEWAY`` guard on ``hermes gateway
stop|restart`` + ``terminal_tool``, and the cron-creation lifecycle filter) stop
the agent scheduling its own restart.  They do NOT cover every SIGTERM source
(raw ``launchctl kickstart``, a bad external monitor, any repeated crash): the
supervisor respawns, the gateway auto-resumes the restart-interrupted session,
whose next turn re-runs the offending logic.

Last-resort circuit breaker: each boot with restart-interrupted sessions pending
is timestamped and persisted to ``<HERMES_HOME>/gateway/restart_loop.json``
(each boot is a fresh process).  Boots CHAIN while consecutive gaps stay within
``max_gap_seconds``, so a slow crash cycle (liveness watchdog every ~150s) trips
exactly like a fast ~10s respawn loop.  When tripped, the caller SKIPS
auto-resume for that boot — real inbound messages are still served.
Best-effort: any read/write failure fails OPEN (a broken breaker must never
wedge a healthy gateway).
"""

from __future__ import annotations

import contextlib
import json
import logging
import time
from typing import List, Optional

from hermes_constants import get_hermes_home

logger = logging.getLogger("gateway.run")

# A legitimate operator restart (or two) never trips; a ~10s respawn loop does
# within a few cycles.
DEFAULT_MAX_RESTARTS = 3
DEFAULT_WINDOW_SECONDS = 60

# Longest gap between consecutive restart-interrupted boots that still counts
# them as the SAME loop.  A fixed-window prune only sees cycles faster than the
# window (a slower loop drops its own history every boot and never trips);
# chaining on the inter-boot gap is period-agnostic, and real quiet resets it.
DEFAULT_MAX_GAP_SECONDS = 300

# Cap the persisted chain; only the newest ``max_restarts`` entries can change
# a verdict, the rest are forensics.
_MAX_STORED_BOOTS = 50


def _state_path():
    return get_hermes_home() / "gateway" / "restart_loop.json"


def _load_boots() -> List[float]:
    try:
        data = json.loads(_state_path().read_text(encoding="utf-8"))
        return [float(t) for t in data.get("boots", []) if isinstance(t, (int, float))]
    except (OSError, ValueError, TypeError):
        return []


def _save_boots(boots: List[float]) -> None:
    with contextlib.suppress(OSError):
        path = _state_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"boots": boots}), encoding="utf-8")


def _chain_gap(window_seconds: int, max_gap_seconds: int) -> float:
    """Inter-boot gap that still links two boots.  Floored by ``window_seconds`` so
    widening the window never makes the breaker *less* sensitive."""
    return float(max(1, window_seconds, max_gap_seconds))


def _chain_ending_at(boots: List[float], ts: float, gap: float) -> List[float]:
    """Unbroken chain of boots leading up to ``ts`` (oldest first).

    Walks backwards while each successive gap stays within ``gap``; the first
    wider gap ends the chain (older boots belong to a resolved episode).
    Nothing recent enough -> empty list: how a healthy gateway forgets a loop.
    """
    chain: List[float] = []
    prev = ts
    for t in sorted(boots, reverse=True):
        if t > ts:
            # Clock moved backwards (NTP step, restored state file): treat the
            # future entry as adjacent rather than dropping the whole chain.
            chain.append(t)
            continue
        if prev - t > gap:
            break
        chain.append(t)
        prev = t
    chain.reverse()
    return chain


def record_restart_interrupted_boot(
    window_seconds: int = DEFAULT_WINDOW_SECONDS,
    *,
    now: Optional[float] = None,
    max_gap_seconds: int = DEFAULT_MAX_GAP_SECONDS,
) -> List[float]:
    """Record a restart-interrupted boot; return the pruned chain + now (most recent last).

    Best-effort — a persistence failure returns the in-memory list without raising.
    """
    ts = time.time() if now is None else now
    boots = _chain_ending_at(_load_boots(), ts, _chain_gap(window_seconds, max_gap_seconds))
    boots.append(ts)
    _save_boots(boots[-_MAX_STORED_BOOTS:])
    return boots


def clear() -> None:
    """Remove the persisted boot log (used on clean shutdown / by tests)."""
    with contextlib.suppress(OSError):
        _state_path().unlink(missing_ok=True)


def check_and_record(
    max_restarts: int = DEFAULT_MAX_RESTARTS,
    window_seconds: int = DEFAULT_WINDOW_SECONDS,
    *,
    now: Optional[float] = None,
    max_gap_seconds: int = DEFAULT_MAX_GAP_SECONDS,
) -> bool:
    """Record this boot and return True when auto-resume should be SKIPPED.

    The single entry point the gateway calls: appends the current boot, then
    checks whether the updated chain has reached ``max_restarts``.
    """
    boots = record_restart_interrupted_boot(
        window_seconds, now=now, max_gap_seconds=max_gap_seconds
    )
    tripped = max_restarts > 0 and len(boots) >= max_restarts
    if tripped:
        logger.warning(
            "Restart-loop breaker TRIPPED: %d chained restart-interrupted "
            "gateway boots (no gap wider than %ds; threshold %d). Skipping "
            "auto-resume to break a suspected SIGTERM-respawn loop (#30719, "
            "#81642). Restart-interrupted sessions stay resume-pending and "
            "will continue on the next real user message. If this is a false "
            "positive, delete %s.",
            len(boots),
            int(_chain_gap(window_seconds, max_gap_seconds)),
            max_restarts,
            _state_path(),
        )
    return tripped
