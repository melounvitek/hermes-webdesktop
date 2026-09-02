"""External drain-control marker contract (dashboard → gateway).

There is no control channel into a running gateway, so the dashboard's
begin/cancel-drain endpoint talks to it the same way ``.restart_notify.json``
does: it writes (or removes) a marker file and a gateway watcher reacts. This
module owns the contract so writer and reader can never disagree.

Contract (presence-based):
  * begin-drain  → write ``{HERMES_HOME}/.drain_request.json`` with
    ``{"action": "drain", "requested_at": <iso>, "principal": <str>,
    "epoch": <instantiation-epoch>, "suppress_notification": <bool>}``.
  * cancel-drain → remove the marker.
  * The watcher treats presence of an ACTIVE marker as "external drain": flip
    ``gateway_state -> "draining"`` and stop accepting new turns. Absence or a
    stale marker means "not draining" (revert to ``running``).

Staleness — two independent, individually-lenient signals (either suffices):
  * epoch mismatch: HERMES_HOME is a durable volume on Hermes Cloud, so a
    marker survives the machine restart that the drain-gated action (update,
    migrate, env edit) ends in; honouring it would park the fresh gateway in
    ``draining`` forever. A marker is stamped with this instantiation's epoch
    and ignored only on a *definite* mismatch.
  * expiry: a same-epoch orphan (action completed without a restart, writer
    never cancelled) is ignored once ``requested_at`` is older than
    :data:`DRAIN_REQUEST_MAX_AGE_SECONDS`. Re-calling :func:`write_drain_request`
    refreshes the timestamp — the sanctioned keep-alive for a long drain.

Reading never raises: a malformed/half-written file reads as "present but
contentless" (``{}``), which still counts as drain-active — fail-safe toward
quiescing. Both staleness checks only reject on a *definite* verdict; a marker
with no epoch/timestamp, or a host without ``/proc``, degrades to the
presence-only behaviour and never fails closed.
"""
from __future__ import annotations

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

# Dedup for the expired-marker warning: the watcher re-reads every second and an
# expired orphan sits on disk until removed. Keyed by ``requested_at`` so a
# keep-alive re-write that later expires again logs again.
_expiry_logged_for: Optional[str] = None


@functools.lru_cache(maxsize=1)
def current_instantiation_epoch() -> str:
    """Identity of THIS container / VM instantiation ("<boot_id>:<pid1_start>").

    Stable for the life of PID 1 (so an s6 respawn of just the gateway, or a
    host ``hermes gateway restart`` under systemd/launchd, keeps honouring an
    in-flight drain) but changes whenever the machine is recreated: boot_id
    changes on a VM/microVM reboot, PID 1's start time on a plain
    ``docker restart`` (same host kernel, brand-new ``/init``).

    Returns ``""`` when neither source is readable (non-Linux, no ``/proc``),
    which disables the epoch check downstream — never fail-closed.
    """
    boot_id = ""
    try:
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip()
    except OSError:
        pass

    pid1_start = ""
    try:
        # "<pid> (<comm>) <state> ...": comm may contain spaces/parens, so split
        # on the LAST ')'. starttime is field 22 (1-indexed) = tail index 19.
        stat = Path("/proc/1/stat").read_text(encoding="utf-8")
        pid1_start = stat.rsplit(")", 1)[1].split()[19]
    except (OSError, IndexError):
        pass

    if not boot_id and not pid1_start:
        return ""
    return f"{boot_id}:{pid1_start}"


def drain_request_path(home: Optional[Path] = None) -> Path:
    """Absolute path to the drain-request marker, respecting HERMES_HOME."""
    base = home if home is not None else get_hermes_home()
    return Path(base) / _DRAIN_REQUEST_FILENAME


def write_drain_request(
    *,
    principal: str = "drain-control",
    suppress_notification: bool = False,
    home: Optional[Path] = None,
) -> dict[str, Any]:
    """Write the begin-drain marker atomically. Returns the payload written.

    Idempotent: re-writing refreshes ``requested_at`` (keep-alive past the
    max-age). ``suppress_notification`` asks the shutdown that ends this drain
    to skip ONLY the home-channel "gateway shutting down" broadcast (the
    per-active-session interrupt ping is never suppressed); the policy of
    which drains are quiet lives entirely in the caller. Defaults False so
    legacy/operator drains behave exactly as before. The marker is stamped
    with :func:`current_instantiation_epoch` so a copy surviving a machine
    restart on the durable volume can be recognised as stale.
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


def _marker_epoch_is_stale(body: dict[str, Any]) -> bool:
    """True iff ``body``'s epoch is a *definite* mismatch (both epochs known and differ)."""
    current = current_instantiation_epoch()
    marker_epoch = body.get("epoch")
    return bool(current and marker_epoch and marker_epoch != current)


def _marker_is_expired(body: dict[str, Any]) -> bool:
    """True iff ``requested_at`` parses AND is older than the max-age.

    Missing/unparseable timestamps and future-dated ones (clock skew) are
    honoured. The expiry is logged once per marker, not per poll: this path
    only fires when a writer leaked a marker, and the warning is the
    operator's breadcrumb.
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
            raw,
            age,
            DRAIN_REQUEST_MAX_AGE_SECONDS,
            body.get("principal"),
        )
    return True


def _marker_is_stale(body: dict[str, Any]) -> bool:
    """True iff the marker is definitely from a drain that is already over."""
    return _marker_epoch_is_stale(body) or _marker_is_expired(body)


def _active_drain_body(home: Optional[Path]) -> Optional[dict[str, Any]]:
    """Marker body if present AND not stale, else None."""
    body = read_drain_request(home=home)
    if body is None or _marker_is_stale(body):
        return None
    return body


def drain_requested(*, home: Optional[Path] = None) -> bool:
    """True iff an active (present, same-epoch, unexpired) begin-drain marker exists."""
    return _active_drain_body(home) is not None


def drain_notification_suppressed(*, home: Optional[Path] = None) -> bool:
    """True iff an ACTIVE drain marker explicitly asks to suppress the shutdown broadcast.

    Uses the same activeness rule as :func:`drain_requested`, so an orphaned
    marker can never silence a fresh gateway's legitimate broadcast. A legacy
    marker without the field or a contentless ``{}`` body reads as False
    (fail toward the louder behaviour).
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
