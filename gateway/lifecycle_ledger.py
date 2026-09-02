"""Gateway lifecycle ledger — durable termination-reason evidence.

Graceful shutdowns already leave forensics (:mod:`gateway.shutdown_forensics`,
``gateway-exit-diag.log``); an **unclean death** (SIGKILL, kernel OOM kill, VM
death) runs no handler, so the next boot would otherwise have no record of it.

A tiny state machine persisted to ``<HERMES_HOME>/state/gateway.lifecycle.json``
closes that gap:

* :func:`record_startup` reads the previous life's sentinel.  ``phase ==
  "running"`` means no exit path ran → unclean death.  The finding (with the
  last heartbeat's memory sample) is appended to ``gateway-exit-diag.log`` as a
  ``gateway.previous_unclean_exit`` record and logged at WARNING, then the
  sentinel is rewritten as ``phase=running`` for the new life.
* :func:`mark_exited` rewrites the sentinel as ``phase=exited`` on every clean
  exit path (``_exit_after_graceful_shutdown`` and the watchdog ``os._exit`` sites).
* :func:`sample_memory` is the cheap /proc snapshot the 30s loop heartbeat
  embeds, so OOM crash cycles are classifiable from the volume alone.

Everything here is best-effort: a forensics failure must never affect the
gateway lifecycle it is observing.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_LIFECYCLE_RELATIVE = ("state", "gateway.lifecycle.json")
_EXIT_DIAG_RELATIVE = ("logs", "gateway-exit-diag.log")
_STATE_DB_RELATIVE = ("state.db",)

# Heuristic OOM-suspicion thresholds on the last heartbeat's memory sample.
# Deliberately conservative: only a hint; classification stays with the reader.
_LOW_MEM_AVAILABLE_KIB = 64 * 1024  # < 64 MiB available
_LOW_MEM_AVAILABLE_FRACTION = 0.05  # < 5% of MemTotal available


def _process_hermes_home() -> Path:
    """HERMES_HOME for process-level identity files (ignore task overrides)."""
    val = os.environ.get("HERMES_HOME", "").strip()
    if val:
        return Path(val)
    from hermes_constants import get_hermes_home

    return get_hermes_home()


def _home_path(home: Optional[Path], relative: tuple) -> Path:
    base = home if home is not None else _process_hermes_home()
    return base.joinpath(*relative)


def get_lifecycle_sentinel_path(home: Optional[Path] = None) -> Path:
    """Return ``<HERMES_HOME>/state/gateway.lifecycle.json``."""
    return _home_path(home, _LIFECYCLE_RELATIVE)


def sample_memory() -> Dict[str, Any]:
    """Cheap memory snapshot: own RSS + system availability + swap.

    Pure ``/proc`` reads, Linux-only (returns ``{}`` elsewhere), never
    raises.  Values in KiB to match the kernel's units.
    """
    sample: Dict[str, Any] = {}
    try:
        with open("/proc/self/status", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    sample["rss_kib"] = int(line.split()[1])
                    break
    except (OSError, ValueError, IndexError):
        pass
    try:
        meminfo: Dict[str, int] = {}
        wanted = {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                key = line.split(":", 1)[0]
                if key in wanted:
                    meminfo[key] = int(line.split()[1])
                    if len(meminfo) == len(wanted):
                        break
        for src, dst in (("MemTotal", "mem_total_kib"), ("MemAvailable", "mem_available_kib")):
            if src in meminfo:
                sample[dst] = meminfo[src]
        if "SwapTotal" in meminfo and "SwapFree" in meminfo:
            sample["swap_used_kib"] = meminfo["SwapTotal"] - meminfo["SwapFree"]
    except (OSError, ValueError, IndexError):
        pass
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
    """Append a JSON line to gateway-exit-diag.log (same format as the CLI's
    ``_exit_diag`` records so existing tooling greps both)."""
    path = _home_path(home, _EXIT_DIAG_RELATIVE)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
    except OSError:
        logger.debug("Failed to append unclean-exit record", exc_info=True)


def _pid_alive_with_start_time(pid: Any, start_time: Any) -> bool:
    """True when ``pid`` is a live process matching ``start_time`` (±2s).

    Guards the takeover race: during ``--replace`` the old gateway can still
    be mid-teardown when the new one boots — a live matching owner is a
    planned handover, not an unclean death.
    """
    try:
        pid_int = int(pid)
    except (TypeError, ValueError):
        return False
    if pid_int <= 0:
        return False
    try:
        # NOT os.kill(pid, 0): on Windows that sends CTRL_C_EVENT to the
        # target's console group. _pid_exists is the canonical no-kill probe.
        from gateway.status import _pid_exists

        if not _pid_exists(pid_int):
            return False
    except Exception:
        return False
    if start_time is None:
        return True  # alive; can't disambiguate PID reuse — err on "alive"
    try:
        from gateway.status import get_process_start_time

        actual = get_process_start_time(pid_int)
        return actual is None or abs(float(actual) - float(start_time)) <= 2.0
    except Exception:
        return True


def detect_unclean_exit(home: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Inspect the previous life's sentinel; return an evidence dict when it
    died uncleanly, else ``None``.  Read-only — does not rewrite the sentinel.
    """
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
            total = mem.get("mem_total_kib")
            avail = mem.get("mem_available_kib")
            if isinstance(avail, int) and (
                avail < _LOW_MEM_AVAILABLE_KIB
                or (
                    isinstance(total, int)
                    and total > 0
                    and avail / total < _LOW_MEM_AVAILABLE_FRACTION
                )
            ):
                evidence["suspected_oom"] = True
    return evidence


def check_state_db_integrity(home: Optional[Path] = None) -> str:
    """Return ``"ok"``, ``"absent"``, or the first ``quick_check`` complaint.

    Called only after an unclean death — a SIGKILL mid-WAL-checkpoint can leave
    half-written b-tree pages (see ``_enforce_macos_synchronous_full`` in
    :mod:`hermes_state`).  ``quick_check(1)`` stops at the first problem (~2s on
    a healthy 500MB store): cheap once per unclean boot, too costly every boot.
    Opened normally, not read-only: a WAL store needs its -shm sidecar for a
    read-only open, and the PRAGMA itself writes nothing.  Never raises.
    """
    path = _home_path(home, _STATE_DB_RELATIVE)
    if not path.exists():
        return "absent"
    try:
        conn = sqlite3.connect(str(path))
        try:
            row = conn.execute("PRAGMA quick_check(1)").fetchone()
        finally:
            conn.close()
    except Exception as exc:  # sqlite3.Error, OSError, anything
        return f"check-failed: {exc}"
    if not row or row[0] is None:
        return "check-failed: no result"
    return str(row[0])


def record_startup(home: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Boot-time entry point: report any unclean previous exit, then claim
    the sentinel for the current life.

    Returns the unclean-exit evidence dict (also persisted to
    ``gateway-exit-diag.log`` and logged at WARNING) or ``None``.  Never
    raises.
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
            _append_exit_diag({
                "ts": datetime.now(timezone.utc).isoformat(),
                "tag": "gateway.previous_unclean_exit",
                "pid": os.getpid(),
                **evidence,
            }, home)
            logger.warning(
                "Previous gateway life (pid=%s, started_at=%s) exited UNCLEANLY "
                "(no exit path ran — SIGKILL / OOM / VM death). "
                "last_heartbeat_at=%s last_mem=%s suspected_oom=%s",
                evidence.get("prior_pid"),
                evidence.get("prior_started_at"),
                evidence.get("last_heartbeat_at"),
                evidence.get("last_heartbeat_mem"),
                evidence.get("suspected_oom", False),
            )
    except Exception:
        logger.debug("Unclean-exit detection failed", exc_info=True)

    try:
        claim: Dict[str, Any] = {
            "phase": "running",
            "pid": os.getpid(),
            "start_time": time.time(),
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        # Carry the verdict on the PREVIOUS life forward on the new sentinel: it
        # is the only machine-readable copy (the exit-diag log is append-only
        # prose) and /api/status reads it to report an OOM restart.  Scoped to
        # this life only — the next clean exit or boot rewrites the sentinel.
        if evidence is not None:
            claim["prior_unclean_exit"] = True
            if evidence.get("suspected_oom"):
                claim["prior_suspected_oom"] = True
        _write_sentinel(claim, home)
    except Exception:
        logger.debug("Failed to claim lifecycle sentinel", exc_info=True)
    return evidence


def mark_exited(
    exit_code: Optional[int] = None,
    reason: str = "graceful_shutdown",
    home: Optional[Path] = None,
) -> None:
    """Mark the current life as cleanly exited.  Idempotent, never raises.

    Only rewrites the sentinel when it is provably owned by this process:
    during a ``--replace`` takeover the replacement claims the sentinel before
    the old process finishes teardown, and the old life must not clobber the
    new owner's ``running`` phase.  A sentinel with ``pid=None`` (or malformed)
    has unknown ownership and is likewise left alone.
    """
    try:
        sentinel = _read_json(get_lifecycle_sentinel_path(home))
        if sentinel is not None and sentinel.get("pid") != os.getpid():
            return
        _write_sentinel({
            "phase": "exited",
            "pid": os.getpid(),
            "exit_code": exit_code,
            "exit_reason": reason,
            "exited_at": datetime.now(timezone.utc).isoformat(),
        }, home)
    except Exception:
        logger.debug("Failed to mark lifecycle sentinel exited", exc_info=True)


def read_prior_exit_label(profile_home: Path) -> str:
    """Container-boot helper: ``clean`` / ``unclean`` / ``unknown`` summary of
    how the profile's last gateway life ended.  Read-only, exception-free;
    used by ``hermes_cli.container_boot`` to annotate ``container-boot.log``.
    """
    try:
        sentinel = _read_json(get_lifecycle_sentinel_path(profile_home))
        if not sentinel:
            return "unknown"
        phase = sentinel.get("phase")
        if phase == "exited":
            return "clean"
        if phase == "running":
            # Old PID namespace is gone at container boot — never exited cleanly.
            return "unclean"
    except Exception:
        pass
    return "unknown"
