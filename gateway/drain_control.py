"""External drain-control marker contract (dashboard → gateway).

There is no control channel into a running gateway, so begin/cancel-drain
writes (or removes) ``{HERMES_HOME}/.drain_request.json`` and a gateway watcher
reacts; this module owns the contract so writer and reader never disagree.
Presence of an ACTIVE marker means "external drain" (``gateway_state ->
"draining"``); absence or a stale marker means "not draining".

Staleness — two independent, individually-lenient signals (either suffices):
epoch mismatch (HERMES_HOME is a durable volume on Hermes Cloud, so a marker
survives the machine restart a drain-gated action ends in and would park the
fresh gateway in ``draining`` forever) and expiry (a same-epoch orphan is
ignored past :data:`DRAIN_REQUEST_MAX_AGE_SECONDS`; re-writing refreshes it).
Reading never raises: a malformed file reads as ``{}``, still drain-active
(fail-safe toward quiescing).  Staleness rejects only on a *definite* verdict —
no epoch/timestamp, or no ``/proc``, degrades to presence-only, never fail-closed.
"""
from __future__ import annotations

import contextlib
import functools
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from hermes_constants import get_hermes_home
from utils import atomic_json_write

_log = logging.getLogger(__name__)

_DRAIN_REQUEST_FILENAME = ".drain_request.json"
# Drain-gated lifecycle actions complete in minutes; an hour bounds the wedge a
# leaked marker can cause. Long drains refresh the marker instead of raising this.
DRAIN_REQUEST_MAX_AGE_SECONDS = 3600.0
# Dedup for the expired-marker warning (the watcher re-reads every second).
# Keyed by ``requested_at`` so a keep-alive re-write that later expires logs again.
_expiry_logged_for: Optional[str] = None


@functools.lru_cache(maxsize=1)
def current_instantiation_epoch() -> str:
    """Identity of THIS container / VM instantiation ("<boot_id>:<pid1_start>").

    Stable for the life of PID 1 (an s6 respawn of just the gateway, or a host
    ``hermes gateway restart``, keeps honouring an in-flight drain) but changes
    whenever the machine is recreated: boot_id on a VM reboot, PID 1's start
    time on a plain ``docker restart``.  ``""`` when neither source is readable
    (non-Linux, no ``/proc``), which disables the epoch check — never fail-closed.
    """
    boot_id = pid1_start = ""
    with contextlib.suppress(OSError):
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip()
    with contextlib.suppress(OSError, IndexError):
        # "<pid> (<comm>) <state> ...": comm may contain spaces/parens, so split
        # on the LAST ')'. starttime is field 22 (1-indexed) = tail index 19.
        pid1_start = Path("/proc/1/stat").read_text(encoding="utf-8").rsplit(")", 1)[1].split()[19]
    return f"{boot_id}:{pid1_start}" if (boot_id or pid1_start) else ""


def drain_request_path(home: Optional[Path] = None) -> Path:
    """Absolute path to the drain-request marker, respecting HERMES_HOME."""
    return Path(home if home is not None else get_hermes_home()) / _DRAIN_REQUEST_FILENAME


def write_drain_request(
    *, principal: str = "drain-control", suppress_notification: bool = False, home: Optional[Path] = None
) -> dict[str, Any]:
    """Write the begin-drain marker atomically. Returns the payload written.

    Idempotent: re-writing refreshes ``requested_at`` (keep-alive past the
    max-age).  ``suppress_notification`` asks the shutdown ending this drain to
    skip ONLY the home-channel "gateway shutting down" broadcast (the per-session
    interrupt ping is never suppressed); which drains are quiet is the caller's
    policy.  Stamped with :func:`current_instantiation_epoch` so a copy surviving
    a machine restart on the durable volume is recognised as stale.
    """
    payload = {
        "action": "drain",
        "requested_at": datetime.now(timezone.utc).isoformat(),
        "principal": principal,
        "epoch": current_instantiation_epoch(),
        "suppress_notification": bool(suppress_notification),
    }
    atomic_json_write(drain_request_path(home), payload)
    return payload


def clear_drain_request(*, home: Optional[Path] = None) -> bool:
    """Remove the drain marker (cancel-drain, idempotent). Returns True if one existed."""
    path = drain_request_path(home)
    try:
        path.unlink()
        return True
    except FileNotFoundError:
        return False
    except OSError as e:
        _log.warning("drain-control: failed to remove %s: %s", path, e)
        return False


def _marker_is_expired(body: dict[str, Any]) -> bool:
    """True iff ``requested_at`` parses AND is older than the max-age.

    Missing/unparseable and future-dated (clock skew) timestamps are honoured.
    Logged once per marker, not per poll — the warning is the operator's
    breadcrumb for a writer that leaked a marker.
    """
    global _expiry_logged_for
    raw = body.get("requested_at")
    if not isinstance(raw, str) or not raw:
        return False
    try:
        requested_at = datetime.fromisoformat(raw)
    except ValueError:
        return False
    if requested_at.tzinfo is None:
        requested_at = requested_at.replace(tzinfo=timezone.utc)
    age = (datetime.now(timezone.utc) - requested_at).total_seconds()
    if age <= DRAIN_REQUEST_MAX_AGE_SECONDS:
        return False
    if _expiry_logged_for != raw:
        _expiry_logged_for = raw
        _log.warning(
            "drain-control: ignoring expired drain marker (requested_at=%s, "
            "age=%.0fs > max %.0fs, principal=%s) — the drain that wrote it "
            "was never cancelled; treating as stale so the gateway keeps "
            "accepting turns.",
            raw, age, DRAIN_REQUEST_MAX_AGE_SECONDS, body.get("principal"),
        )
    return True


def _active_drain_body(home: Optional[Path]) -> Optional[dict[str, Any]]:
    """Marker body if present AND not stale (definite epoch mismatch or expired), else None."""
    body = read_drain_request(home=home)
    if body is None:
        return None
    current, marker_epoch = current_instantiation_epoch(), body.get("epoch")
    if (current and marker_epoch and marker_epoch != current) or _marker_is_expired(body):
        return None
    return body


def drain_requested(*, home: Optional[Path] = None) -> bool:
    """True iff an active (present, same-epoch, unexpired) begin-drain marker exists."""
    return _active_drain_body(home) is not None


def drain_notification_suppressed(*, home: Optional[Path] = None) -> bool:
    """True iff an ACTIVE drain marker explicitly asks to suppress the shutdown broadcast.

    Same activeness rule as :func:`drain_requested`, so an orphaned marker can
    never silence a fresh gateway's broadcast.  A legacy marker without the
    field or a contentless ``{}`` reads as False (fail toward the louder behaviour).
    """
    body = _active_drain_body(home)
    return bool(body and body.get("suppress_notification"))


def read_drain_request(*, home: Optional[Path] = None) -> Optional[dict[str, Any]]:
    """Return the marker payload, ``{}`` if present but unparseable, ``None`` if absent. Never raises."""
    path = drain_request_path(home)
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError as e:
        _log.warning("drain-control: failed to read %s: %s", path, e)
        return None
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return {}
    return data if isinstance(data, dict) else {}
