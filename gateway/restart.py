"""Shared gateway restart constants and supervisor detection helpers."""

import math
import os
from collections.abc import Mapping

from hermes_cli.config import DEFAULT_CONFIG

# EX_TEMPFAIL (sysexits.h): ask the service manager to restart the gateway
# after a graceful drain/reload path completes.
GATEWAY_SERVICE_RESTART_EXIT_CODE = 75

# EX_CONFIG (sysexits.h): fatal configuration error (token collision, no
# platforms). The s6 finish script translates this into exit 125 (permanent
# failure) so the supervisor stops restarting the gateway.
GATEWAY_FATAL_CONFIG_EXIT_CODE = 78


def is_global_startup_conflict(error_code: str | None) -> bool:
    """True when an adapter's fatal error is a single-writer ownership conflict.

    Adapters emit ``{scope}_lock`` with ``retryable=True`` so a *mid-run*
    reconnect can recover once the holder exits or a stale record is cleared.
    At startup a live foreign
    holder is a configuration conflict (two gateways cannot poll one token), so
    the startup router must not treat it as a transient blip. Matches by error
    CODE only (``lock_conflict`` / ``*_lock``), never by message text.
    """
    code = (error_code or "").strip().lower()
    return bool(code) and (code == "lock_conflict" or code.endswith("_lock"))

# Set by ``hermes gateway run --external-supervisor``. Unlike systemd's
# INVOCATION_ID and launchd's XPC_SERVICE_NAME, this survives wrappers that
# replace the child environment (e.g. ``sudo env -i``).
EXTERNAL_GATEWAY_SUPERVISOR_ENV = "HERMES_GATEWAY_EXTERNAL_SUPERVISOR"

DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT = float(
    DEFAULT_CONFIG["agent"]["restart_drain_timeout"]
)
DEFAULT_GATEWAY_SIGNAL_INTERRUPT_GRACE_TIMEOUT = float(
    DEFAULT_CONFIG["gateway"]["signal_interrupt_grace_timeout"]
)
DEFAULT_GATEWAY_POST_INTERRUPT_GRACE_TIMEOUT = 5.0

# In-band restart (``/restart``, SIGUSR1, self-restart) waits for active turns
# to finish *before* ``stop()`` begins. Distinct from ``restart_drain_timeout``,
# the force-interrupt budget once ``stop()`` is running (must stay short under
# systemd TimeoutStopSec).
DEFAULT_GATEWAY_RESTART_AFTER_TURN_TIMEOUT = float(
    DEFAULT_CONFIG["agent"]["restart_after_turn_timeout"]
)

# Cron-only floor under the ``stop()`` drain. ``restart_drain_timeout`` defaults
# to 0 because interrupting a *chat* turn is cheap and recoverable (user is
# told, session pre-marked resume_pending). An interrupted *cron* run has
# neither property — nobody is waiting on it, it lands in jobs.json as a
# permanent failure and a recurring job just waits for its next schedule — so a
# zero-second drain silently destroys work.
DEFAULT_GATEWAY_CRON_DRAIN_TIMEOUT = float(
    DEFAULT_CONFIG["agent"]["cron_drain_timeout"]
)

# Seconds of the shutdown watchdog leash held back for post-drain work
# (interrupt agents, kill tool subprocesses, mark jobs interrupted, disconnect
# adapters). Waiting for cron past that trades a job killed *and recorded* for
# one SIGKILLed mid-write and wedged at ``last_status=running`` forever.
CRON_DRAIN_CLEANUP_RESERVE_S = 10.0

# systemd TimeoutStopSec headroom after the stop-path drain budget, and the
# floor when that budget is still the default immediate (0s) chat drain.
# Keep in lockstep with generate_systemd_unit().
SYSTEMD_STOP_HEADROOM_S = 30.0
SYSTEMD_TIMEOUT_STOP_SEC_FLOOR = 60.0


def is_gateway_supervisor_process(
    environ: Mapping[str, str] | None = None,
) -> bool:
    """Return whether this gateway process is owned by a supervisor."""
    env = os.environ if environ is None else environ
    if env.get("INVOCATION_ID") or env.get("HERMES_S6_SUPERVISED_CHILD"):
        return True
    xpc_service = env.get("XPC_SERVICE_NAME", "")
    if xpc_service and xpc_service != "0":
        return True
    return str(env.get(EXTERNAL_GATEWAY_SUPERVISOR_ENV, "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def is_container_restart_context() -> bool:
    """Whether the gateway runs in a container (Docker/Podman): the detached
    setsid restart path dies with the cgroup, so exit-75 service restart is the
    only viable path. Separate function so tests can mock container detection
    (a real ``/.dockerenv`` on CI otherwise flips the routing)."""
    return os.path.exists("/.dockerenv") or os.path.exists("/run/.containerenv")


def _seconds(value: object, fallback: float = 0.0) -> float:
    """Non-negative float, or ``fallback`` on non-numeric input."""
    try:
        return max(float(value), 0.0)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return fallback


def _parse_timeout_keeping_zero(raw: object, default: float) -> float:
    """Parse a timeout where ``0`` is a deliberate disable (must NOT fall
    through to ``default``), unlike None / blank / non-numeric input."""
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    return max(0.0, value)


def parse_restart_drain_timeout(raw: object) -> float:
    """Parse a configured drain timeout, falling back to the shared default."""
    try:
        value = float(raw) if str(raw or "").strip() else DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT
    except (TypeError, ValueError):
        return DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT
    return max(0.0, value)


def parse_restart_after_turn_timeout(raw: object) -> float:
    """Parse the after-turn wait cap for in-band restart (``0`` = legacy immediate drain)."""
    return _parse_timeout_keeping_zero(raw, DEFAULT_GATEWAY_RESTART_AFTER_TURN_TIMEOUT)


def parse_cron_drain_timeout(raw: object) -> float:
    """Parse the cron-only drain floor (``0`` = opt out; cron interrupted on the chat budget)."""
    return _parse_timeout_keeping_zero(raw, DEFAULT_GATEWAY_CRON_DRAIN_TIMEOUT)


def resolve_cron_drain_budget(
    drain_timeout: float,
    cron_drain_timeout: float,
    *,
    watchdog_delay: float,
    elapsed: float = 0.0,
    cleanup_reserve_s: float = CRON_DRAIN_CLEANUP_RESERVE_S,
) -> float:
    """Seconds the shutdown drain may spend waiting on in-flight cron work.

    The configured floor is clamped to what this process can honour: the
    shutdown watchdog hard-exits at ``watchdog_delay`` (and TimeoutStopSec is
    sized from the same budget), so waiting past that leash minus
    ``cleanup_reserve_s`` swaps a cleanly-interrupted job for a SIGKILL that
    leaves it wedged. Never returns less than ``drain_timeout``: the cron floor
    only ever extends the wait, so an operator who deliberately configured a
    long ``restart_drain_timeout`` keeps it.
    """
    drain = _seconds(drain_timeout)
    floor = _seconds(cron_drain_timeout)
    if floor <= 0.0:
        return drain
    ceiling = (
        _seconds(watchdog_delay)
        - _seconds(elapsed)
        - _seconds(cleanup_reserve_s, CRON_DRAIN_CLEANUP_RESERVE_S)
    )
    return max(drain, min(floor, ceiling))


def resolve_systemd_timeout_stop_sec(
    drain_timeout: float,
    cron_drain_timeout: float = DEFAULT_GATEWAY_CRON_DRAIN_TIMEOUT,
    *,
    cleanup_reserve_s: float = CRON_DRAIN_CLEANUP_RESERVE_S,
    headroom_s: float = SYSTEMD_STOP_HEADROOM_S,
    floor_s: float = SYSTEMD_TIMEOUT_STOP_SEC_FLOOR,
) -> int:
    """Seconds systemd ``TimeoutStopSec`` must cover the full stop budget.

    ``restart_drain_timeout`` is only the chat-turn interrupt budget (default 0);
    the stop path may first wait ``cron_drain_timeout`` + ``cleanup_reserve_s``
    for cron work, so sizing from drain alone lets systemd SIGKILL an in-budget
    drain. A zero cron timeout is an opt-out and does not extend the budget.
    Non-numeric inputs degrade to 0.
    """
    drain = _seconds(drain_timeout)
    cron = _seconds(cron_drain_timeout)
    cron_budget = (cron + _seconds(cleanup_reserve_s)) if cron > 0.0 else 0.0
    stop_budget = max(drain, cron_budget)
    return int(max(_seconds(floor_s), stop_budget + _seconds(headroom_s)))


def resolve_restart_exit_wait_budget(
    drain_timeout: float,
    after_turn_timeout: float,
    *,
    headroom: float = 15.0,
) -> float:
    """Seconds a CLI should wait for the gateway PID to exit after SIGUSR1.

    In-band restart may defer ``stop()`` until turns finish (``after_turn_timeout``)
    and then spend ``drain_timeout`` inside ``stop()``; callers that hard-kill on
    expiry must cover both phases.
    """
    return _seconds(drain_timeout) + _seconds(after_turn_timeout) + _seconds(headroom)


def parse_signal_interrupt_grace_timeout(raw: object) -> float:
    """Parse the unexpected-signal post-interrupt grace timeout."""
    try:
        if raw is None or (isinstance(raw, str) and not raw.strip()):
            value = DEFAULT_GATEWAY_SIGNAL_INTERRUPT_GRACE_TIMEOUT
        else:
            value = float(raw)
    except (TypeError, ValueError):
        return DEFAULT_GATEWAY_SIGNAL_INTERRUPT_GRACE_TIMEOUT
    if not math.isfinite(value):
        return DEFAULT_GATEWAY_SIGNAL_INTERRUPT_GRACE_TIMEOUT
    return max(0.0, value)
