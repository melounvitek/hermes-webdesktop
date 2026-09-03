"""Gateway lifecycle ledger — durable termination-reason evidence.

Graceful shutdowns leave forensics (``gateway-exit-diag.log``); an unclean
death (SIGKILL, kernel OOM, VM death) runs no handler.  A sentinel at
``<HERMES_HOME>/state/gateway.lifecycle.json`` closes the gap:
:func:`record_startup` finds ``phase == "running"`` from the previous life →
unclean death, appended to the exit-diag log as ``gateway.previous_unclean_exit``
and logged at WARNING; :func:`mark_exited` rewrites ``phase=exited`` on every
clean exit path.  Best-effort: forensics must never affect the lifecycle.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_LIFECYCLE_RELATIVE = ("state", "gateway.lifecycle.json")
_EXIT_DIAG_RELATIVE = ("logs", "gateway-exit-diag.log")
_STATE_DB_RELATIVE = ("state.db",)


def _process_hermes_home() -> Path:
    """HERMES_HOME for process-level identity files (ignore task overrides)."""
    val = os.environ.get("HERMES_HOME", "").strip()
    if val:
        return Path(val)
    from hermes_constants import get_hermes_home

    return get_hermes_home()


def _home_path(home: Optional[Path], relative: tuple) -> Path:
    return (_process_hermes_home() if home is None else home).joinpath(*relative)


def get_lifecycle_sentinel_path(home: Optional[Path] = None) -> Path:
    """Return ``<HERMES_HOME>/state/gateway.lifecycle.json``."""
    return _home_path(home, _LIFECYCLE_RELATIVE)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _proc_fields(path: str, wanted: Dict[str, str]) -> Dict[str, int]:
    """``{dst: int}`` for each ``src: dst`` key found in a ``Key: value`` /proc file."""
    found: Dict[str, int] = {}
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                key = line.split(":", 1)[0]
                if key in wanted:
                    found[wanted[key]] = int(line.split()[1])
                    if len(found) == len(wanted):
                        break
    except (OSError, ValueError, IndexError):
        return {}
    return found


def sample_memory() -> Dict[str, Any]:
    """Cheap /proc snapshot (KiB, kernel units): own RSS + MemTotal/MemAvailable + swap used.

    Linux-only (``{}`` elsewhere), never raises; embedded in the 30s loop
    heartbeat so OOM crash cycles are classifiable from the volume alone.
    """
    sample = _proc_fields("/proc/self/status", {"VmRSS": "rss_kib"})
    mem = _proc_fields("/proc/meminfo", {
        "MemTotal": "mem_total_kib", "MemAvailable": "mem_available_kib",
        "SwapTotal": "SwapTotal", "SwapFree": "SwapFree",
    })
    swap_total, swap_free = mem.pop("SwapTotal", None), mem.pop("SwapFree", None)
    sample.update(mem)
    if swap_total is not None and swap_free is not None:
        sample["swap_used_kib"] = swap_total - swap_free
    return sample


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _write_sentinel(payload: Dict[str, Any], home: Optional[Path]) -> None:
    path = get_lifecycle_sentinel_path(home)
    try:
        from utils import atomic_json_write

        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json_write(path, payload, indent=None)
    except Exception:
        logger.debug("Failed to write lifecycle sentinel", exc_info=True)


def _append_exit_diag(record: Dict[str, Any], home: Optional[Path]) -> None:
    """Append a JSON line to gateway-exit-diag.log (same format as the CLI's ``_exit_diag``)."""
    path = _home_path(home, _EXIT_DIAG_RELATIVE)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
    except OSError:
        logger.debug("Failed to append unclean-exit record", exc_info=True)


def _pid_alive_with_start_time(pid: Any, start_time: Any) -> bool:
    """True when ``pid`` is a live process matching ``start_time`` (±2s).

    Guards the ``--replace`` takeover race: a live matching owner mid-teardown
    is a planned handover, not an unclean death.
    """
    try:
        pid_int = int(pid)
        # NOT os.kill(pid, 0): on Windows that sends CTRL_C_EVENT to the target's console group.
        from gateway.status import _pid_exists, get_process_start_time

        if pid_int <= 0 or not _pid_exists(pid_int):
            return False
    except Exception:
        return False
    if start_time is None:
        return True  # alive; can't disambiguate PID reuse — err on "alive"
    try:
        actual = get_process_start_time(pid_int)
        return actual is None or abs(float(actual) - float(start_time)) <= 2.0
    except Exception:
        return True


def detect_unclean_exit(home: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Evidence dict when the previous life died uncleanly, else ``None``. Read-only."""
    sentinel = _read_json(get_lifecycle_sentinel_path(home))
    if not sentinel or sentinel.get("phase") != "running":
        return None
    if _pid_alive_with_start_time(sentinel.get("pid"), sentinel.get("start_time")):
        return None  # live owner — planned takeover in flight, not a death

    evidence: Dict[str, Any] = {
        "prior_pid": sentinel.get("pid"),
        "prior_started_at": sentinel.get("started_at"),
        "prior_start_time": sentinel.get("start_time"),
    }
    # Enrich with the last heartbeat: last proven liveness and memory at that moment.
    try:
        from gateway.shutdown_watchdog import get_loop_heartbeat_path

        hb = _read_json(get_loop_heartbeat_path(home))
    except Exception:
        hb = None
    if hb:
        evidence["last_heartbeat_at"] = hb.get("updated_at")
        mem = hb.get("mem")
        if isinstance(mem, dict):
            evidence["last_heartbeat_mem"] = mem
            # OOM suspicion is only a hint (classification stays with the reader);
            # thresholds are memory_status' "critical" tier so a live warning and a
            # post-mortem verdict can never disagree.
            from gateway.memory_status import _CRITICAL_AVAILABLE_FRACTION, _CRITICAL_AVAILABLE_KIB

            total, avail = mem.get("mem_total_kib"), mem.get("mem_available_kib")
            if isinstance(avail, int) and (
                avail < _CRITICAL_AVAILABLE_KIB
                or (isinstance(total, int) and total > 0 and avail / total < _CRITICAL_AVAILABLE_FRACTION)
            ):
                evidence["suspected_oom"] = True
    return evidence


def check_state_db_integrity(home: Optional[Path] = None) -> str:
    """Return ``"ok"``, ``"absent"``, or the first ``quick_check`` complaint.

    Called only after an unclean death — a SIGKILL mid-WAL-checkpoint can leave
    half-written b-tree pages.  ``quick_check(1)`` stops at the first problem
    (~2s on a healthy 500MB store): cheap once per unclean boot, too costly every
    boot.  Opened normally, not read-only: a WAL store needs its -shm sidecar for
    a read-only open, and the PRAGMA writes nothing.  Never raises.
    """
    path = _home_path(home, _STATE_DB_RELATIVE)
    if not path.exists():
        return "absent"
    try:
        with closing(sqlite3.connect(str(path))) as conn:
            row = conn.execute("PRAGMA quick_check(1)").fetchone()
    except Exception as exc:
        return f"check-failed: {exc}"
    return "check-failed: no result" if not row or row[0] is None else str(row[0])


def record_startup(home: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Boot entry point: report any unclean previous exit, then claim the sentinel.

    Returns the evidence dict (also persisted to ``gateway-exit-diag.log`` and
    logged at WARNING) or ``None``.  Never raises.
    """
    evidence: Optional[Dict[str, Any]] = None
    try:
        evidence = detect_unclean_exit(home)
        if evidence is not None:
            # The death may have torn the store; this is the only moment we know to look.
            verdict = check_state_db_integrity(home=home)
            evidence["state_db_integrity"] = verdict
            if verdict not in ("ok", "absent"):
                logger.error(
                    "state.db FAILED integrity check after an unclean gateway "
                    "exit: %s — sessions may read as missing until it is "
                    "repaired. Run `hermes doctor`.",
                    verdict,
                )
            _append_exit_diag(
                {"ts": _now_iso(), "tag": "gateway.previous_unclean_exit", "pid": os.getpid(), **evidence}, home
            )
            logger.warning(
                "Previous gateway life (pid=%s, started_at=%s) exited UNCLEANLY "
                "(no exit path ran — SIGKILL / OOM / VM death). "
                "last_heartbeat_at=%s last_mem=%s suspected_oom=%s",
                evidence.get("prior_pid"), evidence.get("prior_started_at"), evidence.get("last_heartbeat_at"),
                evidence.get("last_heartbeat_mem"), evidence.get("suspected_oom", False),
            )
    except Exception:
        logger.debug("Unclean-exit detection failed", exc_info=True)

    try:
        claim: Dict[str, Any] = {"phase": "running", "pid": os.getpid(), "start_time": time.time(), "started_at": _now_iso()}
        # Carry the verdict on the PREVIOUS life on the new sentinel: it is the only
        # machine-readable copy (/api/status reads it to report an OOM restart).
        # Scoped to this life — the next clean exit or boot rewrites the sentinel.
        if evidence is not None:
            claim["prior_unclean_exit"] = True
            if evidence.get("suspected_oom"):
                claim["prior_suspected_oom"] = True
        _write_sentinel(claim, home)
    except Exception:
        logger.debug("Failed to claim lifecycle sentinel", exc_info=True)
    return evidence


def mark_exited(exit_code: Optional[int] = None, reason: str = "graceful_shutdown", home: Optional[Path] = None) -> None:
    """Mark the current life as cleanly exited.  Idempotent, never raises.

    Only rewrites a sentinel provably owned by this process: during a
    ``--replace`` takeover the replacement claims it before the old process
    finishes teardown, and the old life must not clobber the new ``running``
    phase.  ``pid=None`` / malformed sentinels have unknown ownership → left alone.
    """
    try:
        sentinel = _read_json(get_lifecycle_sentinel_path(home))
        if sentinel is not None and sentinel.get("pid") != os.getpid():
            return
        _write_sentinel({
            "phase": "exited", "pid": os.getpid(), "exit_code": exit_code,
            "exit_reason": reason, "exited_at": _now_iso(),
        }, home)
    except Exception:
        logger.debug("Failed to mark lifecycle sentinel exited", exc_info=True)


def read_prior_exit_label(profile_home: Path) -> str:
    """``clean`` / ``unclean`` / ``unknown`` for how the profile's last gateway life ended.

    Read-only, exception-free; annotates ``container-boot.log``.  ``running`` is
    unclean because the old PID namespace is gone at container boot.
    """
    try:
        phase = (_read_json(get_lifecycle_sentinel_path(profile_home)) or {}).get("phase")
        return {"exited": "clean", "running": "unclean"}.get(phase, "unknown")
    except Exception:
        return "unknown"
