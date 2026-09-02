"""
Process Registry -- in-memory registry for background processes spawned via
terminal(background=true): rolling 200KB output buffer, poll/log/wait/kill,
crash recovery via a JSON checkpoint, and session-scoped tracking for gateway
reset protection.

Background processes execute THROUGH the environment interface -- nothing runs
on the host unless TERMINAL_ENV=local; for Docker/Singularity/Modal/Daytona/SSH
the command runs inside the sandbox.
"""

import codecs
import json
import logging
import os
import platform
import shlex
import signal
import subprocess
import threading
import time
import uuid
from pathlib import Path

_IS_WINDOWS = platform.system() == "Windows"
# systemd transient scopes exist only on Linux. Gate every scope-path branch
# on this constant (not merely "not Windows") so macOS and other POSIX
# platforms provably never touch systemd code (#70716 cross-platform audit).
_IS_LINUX = platform.system() == "Linux"
from tools.environments.local import _find_shell, _resolve_safe_cwd, _sanitize_subprocess_env
from hermes_cli._subprocess_compat import windows_hide_flags
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from hermes_cli.config import get_hermes_home

from agent.redact import redact_sensitive_text

logger = logging.getLogger(__name__)


# Checkpoint file for crash recovery (gateway only)
CHECKPOINT_PATH = get_hermes_home() / "processes.json"

# Limits
MAX_OUTPUT_CHARS = 200_000      # 200KB rolling output buffer
FINISHED_TTL_SECONDS = 1800     # Keep finished processes for 30 minutes
MAX_PROCESSES = 64              # Max concurrent tracked processes (LRU pruning)

# Watch pattern rate limiting — PER SESSION. At most ONE watch-match notification
# every WATCH_MIN_INTERVAL_SECONDS; a match inside the cooldown is dropped and counts
# as one strike per window. After WATCH_STRIKE_LIMIT consecutive strike windows the
# session's watch_patterns are permanently disabled and it falls back to
# notify_on_complete semantics (one notification when the process exits).
WATCH_MIN_INTERVAL_SECONDS = 15
WATCH_STRIKE_LIMIT = 3

# Lifetime cap — independent of the strike counter. A pattern recurring at a cadence
# just above the cooldown never trips the strike limit yet forces a full-context agent
# turn every time; watch_patterns is documented as "ONLY for rare one-shot signals",
# so after this many delivered matches we disable it and fall back to notify_on_complete.
WATCH_LIFETIME_MAX_HITS = 8

# Global circuit breaker across all sessions — secondary safety net so concurrent
# siblings can't collectively flood the user even when each is under its own cap.
WATCH_GLOBAL_MAX_PER_WINDOW = 15
WATCH_GLOBAL_WINDOW_SECONDS = 10
WATCH_GLOBAL_COOLDOWN_SECONDS = 30


# ---------------------------------------------------------------------------
# systemd cgroup isolation for gateway-spawned local executors
# ---------------------------------------------------------------------------
# Under a systemd gateway with MemoryMax, local background commands inherit the
# gateway's cgroup, so a memory-heavy executor can get the ENTIRE gateway killed by
# systemd-oomd. Wrapping the spawn in ``systemd-run --user --scope`` gives the worker
# its own transient cgroup. Usability is probed once (the binary can exist while the
# user D-Bus session is absent — system services, containers) and cached.

_SYSTEMD_SCOPE_AVAILABLE: Optional[bool] = None
_SYSTEMD_SCOPE_PROBE_LOCK = threading.Lock()
_SYSTEMD_SCOPE_PROBED_AT = 0.0
_SYSTEMD_SCOPE_FAILURE_TTL_SECONDS = 60.0
_MIN_WORKER_MEMORY_MAX_BYTES = 64 * 1024 * 1024
_DEFAULT_WORKER_MEMORY_MAX_BYTES = 1024 * 1024 * 1024
_WORKER_MEMORY_MAX_CAP_BYTES = 4 * 1024 * 1024 * 1024


def _worker_memory_max_bytes() -> int:
    """Finite per-worker cgroup limit that can never widen host risk.

    ``TERMINAL_LOCAL_MEMORY_MAX_MB`` is honored only when it *tightens* the safe
    bound (min of the gateway's cgroup-v2 ``memory.max`` and half of physical RAM,
    capped at 4 GiB), so an oversized override cannot exceed the enclosing slice.
    """
    override_bound: Optional[int] = None
    override = os.getenv("TERMINAL_LOCAL_MEMORY_MAX_MB", "").strip()
    if override:
        try:
            parsed = int(override) * 1024 * 1024
        except ValueError:
            parsed = -1
        if parsed >= _MIN_WORKER_MEMORY_MAX_BYTES:
            override_bound = parsed
        else:
            logger.warning(
                "Ignoring invalid TERMINAL_LOCAL_MEMORY_MAX_MB=%r; "
                "expected an integer representing at least %d MiB",
                override,
                _MIN_WORKER_MEMORY_MAX_BYTES // (1024 * 1024),
            )

    candidates: List[int] = []
    try:
        for line in Path("/proc/self/cgroup").read_text(encoding="utf-8").splitlines():
            if line.startswith("0::"):
                relative = line.partition("::")[2].lstrip("/")
                raw_limit = (
                    Path("/sys/fs/cgroup") / relative / "memory.max"
                ).read_text(encoding="utf-8").strip()
                if raw_limit.isdigit() and int(raw_limit) >= _MIN_WORKER_MEMORY_MAX_BYTES:
                    candidates.append(int(raw_limit))
                break
    except (OSError, ValueError):
        pass

    try:
        physical_bytes = int(os.sysconf("SC_PHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
        candidates.append(min(
            _WORKER_MEMORY_MAX_CAP_BYTES,
            max(_MIN_WORKER_MEMORY_MAX_BYTES, physical_bytes // 2),
        ))
    except (OSError, ValueError, TypeError):
        pass

    safe_bound = min(candidates) if candidates else _DEFAULT_WORKER_MEMORY_MAX_BYTES
    return min(override_bound, safe_bound) if override_bound else safe_bound


def _systemd_scope_argv(binary: str, unit_name: str, *argv: str) -> List[str]:
    """``systemd-run --user --scope`` command line shared by the probe and real spawns.

    ``--collect`` makes the transient scope self-clean after exit; ``--unit`` gives it
    a recognisable name for ``systemctl --user status`` / journalctl.
    """
    return [
        binary, "--user", "--scope", "--quiet",
        "--unit", unit_name,
        "--collect",
        "--property", "MemoryAccounting=yes",
        "--property", f"MemoryMax={_worker_memory_max_bytes()}",
        "--property", "OOMPolicy=kill",
        "--",
        *argv,
    ]


def _systemd_scope_cached() -> Optional[bool]:
    """Cached probe verdict, or None when a (re)probe is due.

    A True verdict is permanent; a False one expires after
    ``_SYSTEMD_SCOPE_FAILURE_TTL_SECONDS`` so a transient D-Bus outage isn't sticky.
    """
    cached = _SYSTEMD_SCOPE_AVAILABLE
    if cached is True:
        return True
    if cached is False and time.monotonic() - _SYSTEMD_SCOPE_PROBED_AT < _SYSTEMD_SCOPE_FAILURE_TTL_SECONDS:
        return False
    return None


def _systemd_run_user_scope_available() -> bool:
    """Return True if ``systemd-run --user --scope`` can create a cgroup.

    ``shutil.which`` alone is insufficient: system-service deployments and containers
    may lack the user D-Bus session bus even though the binary is on PATH, so every
    spawn would fail with ``Failed to connect to user bus``. We run a cheap no-op
    probe (``systemd-run --user --scope --unit=… -- /bin/true``) and cache the outcome.
    """
    global _SYSTEMD_SCOPE_AVAILABLE, _SYSTEMD_SCOPE_PROBED_AT
    verdict = _systemd_scope_cached()
    if verdict is not None:
        return verdict

    # Double-checked locking keeps concurrent first-use spawns from observing a
    # temporary False while the definitive probe is still in flight — such a race
    # would launch the losing workload back inside the gateway cgroup.
    with _SYSTEMD_SCOPE_PROBE_LOCK:
        verdict = _systemd_scope_cached()
        if verdict is not None:
            return verdict

        available = False
        if _IS_LINUX:
            try:
                import shutil

                binary = shutil.which("systemd-run")
                if binary:
                    # A unique unit avoids collisions; the timeout bounds D-Bus.
                    probe_unit = f"hermes-probe-scope-{os.getpid()}-{uuid.uuid4().hex[:8]}"
                    result = subprocess.run(
                        _systemd_scope_argv(binary, probe_unit, "/bin/true"),
                        capture_output=True,
                        timeout=3,
                    )
                    available = result.returncode == 0
                    if not available:
                        logger.debug(
                            "systemd-run --user --scope probe failed (rc=%s): %s",
                            result.returncode,
                            (result.stderr or b"").decode("utf-8", "replace").strip(),
                        )
            except Exception as exc:
                logger.debug("systemd-run --user --scope probe error: %s", exc)

        _SYSTEMD_SCOPE_AVAILABLE = available
        _SYSTEMD_SCOPE_PROBED_AT = time.monotonic()
        return available


def _is_supervised_gateway_process() -> bool:
    """Whether this process is the live, supervised Hermes gateway itself.

    Supervisor markers and ``_HERMES_GATEWAY`` are inherited by every descendant
    (and importing ``gateway.run`` sets the latter), so also require ownership of
    the live gateway PID file — transient scopes are for the gateway, not terminal
    children or unrelated CLIs in the same supervised tree.
    """
    if os.environ.get("_HERMES_GATEWAY") != "1":
        return False

    try:
        from gateway.restart import is_gateway_supervisor_process
        from gateway.status import get_running_pid

        return (
            is_gateway_supervisor_process()
            and get_running_pid(cleanup_stale=False) == os.getpid()
        )
    except Exception as exc:
        logger.debug("Could not verify supervised gateway process identity: %s", exc)
        return False


def _build_systemd_scope_argv(
    shell_argv: List[str],
    unit_suffix: str,
) -> List[str]:
    """Wrap *shell_argv* in a ``systemd-run --user --scope`` invocation with its own
    memory accounting, so an OOM in the worker cannot kill the gateway cgroup."""
    import shutil

    binary = shutil.which("systemd-run")
    if binary is None:
        # Caller should have checked _systemd_run_user_scope_available();
        # guard anyway so we never pass None into Popen.
        return shell_argv
    return _systemd_scope_argv(binary, f"hermes-worker-{unit_suffix}", *shell_argv)


def _stop_systemd_unit(unit_name: str) -> bool:
    """Stop a transient systemd user scope by unit name.

    Reaps the *entire* cgroup — catching double-forked descendants that survive a
    plain PID signal because they were reparented to init inside the scope.
    ``systemctl --user stop`` SIGTERMs every process in the cgroup and escalates to
    SIGKILL after ``TimeoutStopSec``.

    Returns True if the unit was stopped (or was already gone), False if
    ``systemctl`` is unavailable or the stop command failed.
    """
    import shutil

    binary = shutil.which("systemctl")
    if binary is None:
        return False
    try:
        result = subprocess.run(
            [binary, "--user", "stop", unit_name],
            capture_output=True,
            timeout=15,
        )
        if result.returncode != 0:
            stderr = (result.stderr or b"").decode(errors="replace").strip()
            stderr_lower = stderr.lower()
            if any(
                marker in stderr_lower
                for marker in ("not loaded", "not found", "does not exist")
            ):
                return True
            logger.debug(
                "systemctl --user stop %s exited %d: %s",
                unit_name, result.returncode,
                stderr,
            )
            return False
        return True
    except Exception as exc:
        logger.debug("systemctl --user stop %s failed: %s", unit_name, exc)
        return False


def format_uptime_short(seconds: int) -> str:
    s = max(0, int(seconds))
    if s < 60:
        return f"{s}s"
    mins, secs = divmod(s, 60)
    if mins < 60:
        return f"{mins}m {secs}s"
    hours, mins = divmod(mins, 60)
    return f"{hours}h {mins}m"


@dataclass
class ProcessSession:
    """A tracked background process with output buffering."""
    id: str                                     # Unique session ID ("proc_xxxxxxxxxxxx")
    command: str                                 # Original command string
    task_id: str = ""                           # Task/sandbox isolation key
    owner_task_id: str = ""                     # RAW spawning task id (e.g. "sa-..."); task_id is the
                                                # CONTAINER key (may be collapsed by _resolve_container_task_id)
                                                # so ownership checks must use this field
    session_key: str = ""                       # Gateway session key (for reset protection)
    pid: Optional[int] = None                   # OS process ID
    process: Optional[subprocess.Popen] = None  # Popen handle (local only)
    env_ref: Any = None                         # Reference to the environment object
    cwd: Optional[str] = None                   # Working directory
    started_at: float = 0.0                     # time.time() of spawn (wall clock)
    host_start_time: Optional[int] = None       # kernel start ticks (/proc/<pid>/stat f22) — PID-reuse guard
    exited: bool = False                        # Whether the process has finished
    exit_code: Optional[int] = None             # Exit code (None if still running)
    completion_reason: str = "exited"           # exited|killed|lost|failed_start|already_exited
    termination_source: str = ""                # process.kill|kill_all|backend_lost|failed_start
    output_buffer: str = ""                     # Rolling output (last MAX_OUTPUT_CHARS)
    max_output_chars: int = MAX_OUTPUT_CHARS
    detached: bool = False                      # True if recovered from crash (no pipe)
    pid_scope: str = "host"                     # "host" for local/PTY PIDs, "sandbox" for env-local PIDs
    systemd_unit: str = ""                      # transient scope unit name when spawned under systemd-run (#70716)
    # Watcher/notification metadata (persisted for crash recovery)
    watcher_platform: str = ""
    watcher_chat_id: str = ""
    watcher_user_id: str = ""
    watcher_user_name: str = ""
    watcher_thread_id: str = ""
    watcher_message_id: str = ""                # Triggering message id — reply anchor for topic routing
    watcher_interval: int = 0                   # 0 = no watcher configured
    # Session-db id of the spawning conversation; lets the gateway drop completions
    # whose session was closed at a user boundary (/new) instead of injecting them
    # into the chat's NEW session.
    parent_session_id: str = ""
    notify_on_complete: bool = False             # Queue agent notification on exit
    watch_patterns: List[str] = field(default_factory=list)
    _watch_hits: int = field(default=0, repr=False)          # total matches delivered
    _watch_suppressed: int = field(default=0, repr=False)    # matches dropped by rate limit
    _watch_disabled: bool = field(default=False, repr=False) # permanently killed after strike limit
    # Per-session rate-limit state (see WATCH_* constants). A strike is a WINDOW with
    # drops, not a dropped match.
    _watch_last_emit_at: float = field(default=0.0, repr=False)
    _watch_cooldown_until: float = field(default=0.0, repr=False)
    _watch_strike_candidate: bool = field(default=False, repr=False)
    _watch_consecutive_strikes: int = field(default=0, repr=False)
    _completion_event: threading.Event = field(default_factory=threading.Event, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _reader_thread: Optional[threading.Thread] = field(default=None, repr=False)
    _pty: Any = field(default=None, repr=False)  # ptyprocess handle (when use_pty=True)

    def append_output(self, text: str) -> None:
        """Append to the rolling output buffer under the session lock, keeping the tail."""
        with self._lock:
            self.output_buffer += text
            if len(self.output_buffer) > self.max_output_chars:
                self.output_buffer = self.output_buffer[-self.max_output_chars:]


# Session fields persisted verbatim in the crash-recovery checkpoint (plus
# ``session_id``; ``command`` is redacted and ``owner_task_id`` defaulted on write).
_CHECKPOINT_FIELDS = (
    "command", "pid", "pid_scope", "host_start_time", "systemd_unit", "cwd",
    "started_at", "task_id", "owner_task_id", "session_key",
    "watcher_platform", "watcher_chat_id", "watcher_user_id", "watcher_user_name",
    "watcher_thread_id", "watcher_message_id", "watcher_interval",
    "parent_session_id", "notify_on_complete", "watch_patterns",
)
_CHECKPOINT_DEFAULTS = {
    f.name: ([] if f.name == "watch_patterns" else f.default)
    for f in ProcessSession.__dataclass_fields__.values()
    if f.name in _CHECKPOINT_FIELDS
}


class ProcessRegistry:
    """In-memory registry of running and finished background processes.

    Thread-safe: accessed from executor threads (terminal_tool, process handlers),
    the gateway asyncio loop (watchers, reset checks) and the cleanup thread.
    """

    _SHELL_NOISE_SUBSTRINGS = (
        "bash: cannot set terminal process group",
        "bash: no job control in this shell",
        "no job control in this shell",
        "cannot set terminal process group",
        "tcsetattr: Inappropriate ioctl for device",
    )

    def __init__(self):
        self._running: Dict[str, ProcessSession] = {}
        self._finished: Dict[str, ProcessSession] = {}
        self._lock = threading.Lock()

        # Side-channel for check_interval watchers (gateway reads after agent run)
        self.pending_watchers: List[Dict[str, Any]] = []

        # Unified queue for all background events (completion, watch_match,
        # async_delegation...; distinguished by "type"). CLI process_loop and the
        # gateway drain it after each agent turn to auto-trigger new turns.
        import queue as _queue_mod
        self.completion_queue: _queue_mod.Queue = _queue_mod.Queue()
        # Rehydrate durable delegation completions once, at registry startup.
        try:
            from tools.async_delegation import restore_undelivered_completions
            restore_undelivered_completions(self.completion_queue)
        except Exception as exc:
            logger.warning("Could not restore async delegation completions: %s", exc)

        # Sessions whose completion the agent already consumed via wait()/read_log()
        # — it has the output in hand, so drain loops AND gateway/tui watchers skip.
        self._completion_consumed: set = set()
        # Sessions merely *observed* exited via poll(). poll() is read-only and must
        # NOT mark consumed (a status check would suppress the watcher's autonomous
        # delivery turn), but on the CLI the poll result is inline in the same turn,
        # so drain_notifications() skips these to avoid a duplicate [SYSTEM: ...];
        # gateway/tui watchers deliberately ignore this set.
        self._poll_observed: set = set()

        # Global watch-match circuit breaker across all sessions.
        self._global_watch_lock = threading.Lock()
        self._global_watch_window_start: float = 0.0
        self._global_watch_window_hits: int = 0
        self._global_watch_tripped_until: float = 0.0
        self._global_watch_suppressed_during_trip: int = 0
        # Driver-installed sinks (desktop gateway): on_output(session, chunk) streams
        # live output from reader threads; on_close(session_or_none, process_id) drops
        # a read-only terminal tab without killing the process.
        self.on_output = None
        self.on_close = None

    @staticmethod
    def _clean_shell_noise(text: str) -> str:
        """Strip shell startup warnings from the beginning of output."""
        lines = text.split("\n")
        while lines and any(noise in lines[0] for noise in ProcessRegistry._SHELL_NOISE_SUBSTRINGS):
            lines.pop(0)
        return "\n".join(lines)

    def _emit_output(self, session: ProcessSession, chunk: str) -> None:
        """Forward a freshly-read chunk to the live-output sink, if one is set.
        Called from reader threads; never raise into the read loop."""
        sink = self.on_output
        if sink is None or not chunk:
            return
        try:
            sink(session, chunk)
        except Exception:
            pass

    def _check_watch_patterns(self, session: ProcessSession, new_text: str) -> None:
        """Scan a freshly-read chunk for watch patterns and queue notifications.

        Rate limiting per session (see the WATCH_* constants): one match per cooldown
        window, a match inside the window is one strike, WATCH_STRIKE_LIMIT
        consecutive strikes or WATCH_LIFETIME_MAX_HITS total deliveries disable
        watching and promote the session to notify_on_complete.
        """
        if not session.watch_patterns or session._watch_disabled:
            return
        # Late chunks after the reader declared exit are post-exit noise; dropping them
        # avoids stale notifications minutes after the process ended.
        if session.exited:
            return

        # Scan new text line-by-line for pattern matches
        matched_lines = []
        matched_pattern = None
        for line in new_text.splitlines():
            for pat in session.watch_patterns:
                if pat in line:
                    matched_lines.append(line.rstrip())
                    if matched_pattern is None:
                        matched_pattern = pat
                    break  # one match per line is enough

        if not matched_lines:
            return

        now = time.time()
        should_disable = False
        lifetime_exhausted = False
        with session._lock:
            # Case 1: inside the cooldown — drop, count one strike per window, and
            # disable + promote once the strike limit is hit.
            if session._watch_cooldown_until and now < session._watch_cooldown_until:
                session._watch_suppressed += len(matched_lines)
                if not session._watch_strike_candidate:
                    # First drop in this window — count one strike.
                    session._watch_strike_candidate = True
                    session._watch_consecutive_strikes += 1
                    if session._watch_consecutive_strikes >= WATCH_STRIKE_LIMIT:
                        session._watch_disabled = True
                        # Promote to notify_on_complete so the agent still gets
                        # exactly one notification when the process actually ends.
                        session.notify_on_complete = True
                        should_disable = True
                return_early = True
            else:
                # Case 2: cooldown expired. A prior window with no drops resets the
                # consecutive-strike counter (healthy cadence again).
                if session._watch_cooldown_until and not session._watch_strike_candidate:
                    session._watch_consecutive_strikes = 0
                session._watch_strike_candidate = False

                # Emit the notification and start a new cooldown window.
                session._watch_last_emit_at = now
                session._watch_cooldown_until = now + WATCH_MIN_INTERVAL_SECONDS
                session._watch_hits += 1
                suppressed = session._watch_suppressed
                session._watch_suppressed = 0
                return_early = False
                # Lifetime cap: this match is still delivered, but no further ones.
                lifetime_exhausted = session._watch_hits >= WATCH_LIFETIME_MAX_HITS
                if lifetime_exhausted:
                    session._watch_disabled = True
                    session.notify_on_complete = True

        if return_early:
            if should_disable:
                # Exactly one summary so the agent/user sees why things went quiet.
                self.completion_queue.put({
                    **self._watch_event_base(session),
                    "type": "watch_disabled",
                    "suppressed": session._watch_suppressed,
                    "message": (
                        f"Watch patterns disabled for process {session.id} — "
                        f"{WATCH_STRIKE_LIMIT} consecutive rate-limit windows triggered "
                        f"(min spacing {WATCH_MIN_INTERVAL_SECONDS}s). "
                        f"Falling back to notify_on_complete semantics; you'll get "
                        f"exactly one notification when the process exits."
                    ),
                })
            return

        # Trim matched output to a reasonable size
        output = "\n".join(matched_lines[:20])
        if len(output) > 2000:
            output = output[:2000] + "\n...(truncated)"

        if not self._global_watch_admit(now):
            # Even when the breaker drops the final match, still explain the silence.
            if lifetime_exhausted:
                self._emit_lifetime_watch_disabled(session)
            return

        notification = {
            **self._watch_event_base(session),
            "type": "watch_match",
            "pattern": matched_pattern,
            "output": output,
            "suppressed": suppressed,
        }
        _redact_process_result(notification)
        self.completion_queue.put(notification)

        if lifetime_exhausted:
            self._emit_lifetime_watch_disabled(session)

    def _emit_lifetime_watch_disabled(self, session: ProcessSession) -> None:
        """Queue the watch_disabled summary for the lifetime-cap path."""
        self.completion_queue.put({
            **self._watch_event_base(session),
            "type": "watch_disabled",
            "suppressed": 0,
            "message": (
                f"Watch patterns disabled for process {session.id} — "
                f"reached the lifetime cap of {WATCH_LIFETIME_MAX_HITS} delivered "
                f"matches. Falling back to notify_on_complete semantics; you'll get "
                f"exactly one notification when the process exits."
            ),
        })

    @staticmethod
    def _watch_event_base(session: ProcessSession) -> dict:
        """Session identity + watcher routing fields shared by every watch event."""
        return {
            "session_id": session.id,
            "session_key": session.session_key,
            "task_id": session.task_id,
            "owner_task_id": session.owner_task_id or session.task_id,
            "command": session.command,
            "platform": session.watcher_platform,
            "chat_id": session.watcher_chat_id,
            "user_id": session.watcher_user_id,
            "user_name": session.watcher_user_name,
            "thread_id": session.watcher_thread_id,
            "message_id": session.watcher_message_id,
        }

    @staticmethod
    def _global_watch_event(type_: str, message: str, **extra) -> dict:
        """Unaddressed (all-sessions) watch breaker event."""
        return {
            "session_id": "",
            "session_key": "",
            "command": "",
            "type": type_,
            **extra,
            "message": message,
            "platform": "",
            "chat_id": "",
            "user_id": "",
            "user_name": "",
            "thread_id": "",
        }

    def _global_watch_admit(self, now: float) -> bool:
        """True if this watch_match may pass the global breaker.

        In cooldown: drop and count. Otherwise slide the rolling window; exceeding
        the cap trips the breaker for WATCH_GLOBAL_COOLDOWN_SECONDS with ONE
        "tripped" summary, and the cooldown's end emits ONE "released" summary.
        """
        release_msg = None
        with self._global_watch_lock:
            # Handle cooldown expiry first so we can emit the release summary.
            if self._global_watch_tripped_until and now >= self._global_watch_tripped_until:
                suppressed = self._global_watch_suppressed_during_trip
                self._global_watch_tripped_until = 0.0
                self._global_watch_suppressed_during_trip = 0
                self._global_watch_window_start = now
                self._global_watch_window_hits = 0
                if suppressed > 0:
                    # Queued outside the lock (below).
                    release_msg = self._global_watch_event(
                        "watch_overflow_released",
                        f"Watch-pattern notifications resumed. "
                        f"{suppressed} match event(s) were suppressed during the flood.",
                        suppressed=suppressed,
                    )

            # Still in cooldown — drop and count.
            if self._global_watch_tripped_until and now < self._global_watch_tripped_until:
                self._global_watch_suppressed_during_trip += 1
                admit = False
                trip_now = None
            else:
                # Slide the window.
                if now - self._global_watch_window_start >= WATCH_GLOBAL_WINDOW_SECONDS:
                    self._global_watch_window_start = now
                    self._global_watch_window_hits = 0

                if self._global_watch_window_hits >= WATCH_GLOBAL_MAX_PER_WINDOW:
                    # Trip the breaker.
                    self._global_watch_tripped_until = now + WATCH_GLOBAL_COOLDOWN_SECONDS
                    self._global_watch_suppressed_during_trip += 1
                    trip_now = now
                    admit = False
                else:
                    self._global_watch_window_hits += 1
                    trip_now = None
                    admit = True

        # Queue summary events outside the lock.
        if release_msg is not None:
            self.completion_queue.put(release_msg)
        if trip_now is not None:
            self.completion_queue.put(self._global_watch_event(
                "watch_overflow_tripped",
                f"Watch-pattern overflow: >{WATCH_GLOBAL_MAX_PER_WINDOW} "
                f"notifications in {WATCH_GLOBAL_WINDOW_SECONDS}s across all processes. "
                f"Suppressing further watch_match events for "
                f"{WATCH_GLOBAL_COOLDOWN_SECONDS}s.",
            ))
        return admit

    @staticmethod
    def _is_host_pid_alive(pid: Optional[int]) -> bool:
        """Best-effort liveness check for host-visible PIDs."""
        if not pid:
            return False
        # ``os.kill(pid, 0)`` is NOT a no-op on Windows (bpo-14484) — use
        # the cross-platform existence check.
        from gateway.status import _pid_exists
        return _pid_exists(pid)

    @staticmethod
    def _safe_host_start_time(pid: Optional[int]) -> Optional[int]:
        """Kernel start ticks for a host PID, or None when unavailable."""
        if not pid:
            return None
        try:
            from gateway.status import get_process_start_time
            return get_process_start_time(pid)
        except Exception:
            return None

    @classmethod
    def _host_pid_is_ours(cls, pid: Optional[int], expected_start: Optional[int]) -> bool:
        """True only if ``pid`` is alive AND still the process we spawned.

        The kernel recycles PIDs, so a stored number can later name an unrelated
        process (seen in the wild: a browser's session leader tree-killed). The kernel
        start time captured at spawn must match the live one; with no baseline
        (legacy checkpoints, no ``/proc``) degrade to a bare liveness check.
        """
        if not cls._is_host_pid_alive(pid):
            return False
        if expected_start is None:
            return True
        return cls._safe_host_start_time(pid) == expected_start

    def _refresh_detached_session(self, session: Optional[ProcessSession]) -> Optional[ProcessSession]:
        """Update recovered host-PID sessions when the underlying process has exited."""
        if session is None or session.exited or not session.detached or session.pid_scope != "host":
            return session

        # A recycled PID (alive but not ours) counts as "our process exited" so a
        # later kill() can never tree-kill the stranger.
        if self._host_pid_is_ours(session.pid, session.host_start_time):
            return session

        with session._lock:
            if session.exited:
                return session
            session.exited = True
            # Recovered sessions no longer have a waitable handle, so the real
            # exit code is unavailable once the original process object is gone.
            session.exit_code = None

        self._move_to_finished(session)
        return session

    @staticmethod
    def _proc_alive(proc) -> bool:
        """True if a psutil.Process is running and not a zombie.

        A zombie is already dead (just unreaped), so there's nothing to SIGKILL.
        """
        try:
            import psutil
            if not proc.is_running():
                return False
            return proc.status() != psutil.STATUS_ZOMBIE
        except Exception:
            return False

    @staticmethod
    def _config_value(section: str, key: str, fallback):
        """``config.yaml`` value for ``section.key``, else the DEFAULT_CONFIG value.

        Raises if config is unreadable; callers wrap with their own hard fallback so
        registry code paths never crash on a broken config file.
        """
        from hermes_cli.config import DEFAULT_CONFIG, cfg_get, read_raw_config

        val = cfg_get(read_raw_config(), section, key)
        return DEFAULT_CONFIG[section][key] if val is None else val

    @staticmethod
    def _daemon_term_grace_seconds() -> float:
        """Grace window (s) between SIGTERM and escalated SIGKILL, floored at 0
        (0 disables escalation). ``terminal.daemon_term_grace_seconds``; 2.0 if
        config is unreadable."""
        try:
            return max(float(ProcessRegistry._config_value("terminal", "daemon_term_grace_seconds", 2.0)), 0.0)
        except Exception:
            return 2.0

    @classmethod
    def _terminate_host_pid(cls, pid: int, expected_start: Optional[int] = None) -> None:
        """Terminate a host-visible PID and its descendants.

        ``expected_start`` (kernel start time at spawn) is re-validated first; a
        mismatch or dead PID means the number was recycled onto a stranger and we
        refuse to touch it — a leaked orphan beats tree-killing someone's browser.

        POSIX: psutil walks the tree and SIGTERMs children before the parent so
        subprocess trees (Chromium renderers under an agent-browser daemon) aren't
        reparented to init and survive. After ``terminal.daemon_term_grace_seconds``
        any survivor is SIGKILLed (0 disables escalation).

        Windows: ``taskkill /PID <pid> /T /F`` (same primitive as
        ``gateway.status.terminate_pid``; ``/F`` is already a hard kill). The psutil
        path is unusable there: PPID links go stale so ``children(recursive=True)``
        misses orphans, and ``terminate()`` is ``TerminateProcess()`` on one handle —
        nothing cascades like a SIGTERM to a process group. The bare ``os.kill``
        fallback covers OSError/PermissionError and a missing ``taskkill.exe``.
        """
        if expected_start is not None and not cls._host_pid_is_ours(pid, expected_start):
            logger.warning(
                "Refusing to terminate host pid %d: start-time mismatch — "
                "PID was recycled onto an unrelated process.", pid,
            )
            return
        def _sigterm_quietly():
            try:
                os.kill(pid, signal.SIGTERM)
            except (OSError, ProcessLookupError, PermissionError):
                pass

        if _IS_WINDOWS:
            try:
                subprocess.run(
                    ["taskkill", "/PID", str(pid), "/T", "/F"],
                    capture_output=True,
                    text=True, encoding='utf-8', errors='replace',
                    timeout=10,
                    creationflags=windows_hide_flags(),
                    stdin=subprocess.DEVNULL,
                )
            except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
                _sigterm_quietly()
            return

        import psutil
        try:
            parent = psutil.Process(pid)
        except psutil.NoSuchProcess:
            return
        except (OSError, PermissionError):
            _sigterm_quietly()
            return

        # Snapshot the whole tree (children before parent) and SIGTERM each.
        try:
            targets = parent.children(recursive=True)
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
            targets = []
        targets.append(parent)

        for proc in targets:
            try:
                proc.terminate()
            except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
                pass

        # Escalate to SIGKILL for anything that ignored SIGTERM within the grace
        # window. We deliberately do NOT trust ``psutil.wait_procs``' gone/alive
        # partition: it reaps via ``Process.wait()`` and mis-partitions across
        # zombie transitions in a parent/child tree, leaving survivors un-killed.
        # A direct liveness re-probe of every target is deterministic.
        grace = cls._daemon_term_grace_seconds()
        if grace <= 0:
            return
        deadline = time.monotonic() + grace
        while time.monotonic() < deadline:
            if not any(cls._proc_alive(_p) for _p in targets):
                break
            time.sleep(0.05)
        for proc in targets:
            try:
                if not cls._proc_alive(proc):
                    continue
                proc.kill()  # SIGKILL on POSIX
                logger.info(
                    "Escalated to SIGKILL for pid %d (ignored SIGTERM within "
                    "%.1fs grace)", proc.pid, grace,
                )
            except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
                pass

    # ----- Spawn -----

    @staticmethod
    def _env_temp_dir(env: Any) -> str:
        """Return the writable sandbox temp dir for env-backed background tasks."""
        get_temp_dir = getattr(env, "get_temp_dir", None)
        if callable(get_temp_dir):
            try:
                temp_dir = get_temp_dir()
                if isinstance(temp_dir, str) and temp_dir.startswith("/"):
                    return temp_dir.rstrip("/") or "/"
            except Exception as exc:
                logger.debug("Could not resolve environment temp dir: %s", exc)
        return "/tmp"

    def _scope_argv(self, session: ProcessSession, argv: List[str], unit_suffix: str, label: str):
        """Wrap *argv* in a transient systemd scope when we are the supervised gateway.

        Returns ``(argv, scoped)``. A scoped worker gets its own cgroup so an OOM kills
        only the worker, not the gateway (and its messaging control plane).
        """
        in_supervised_gateway = _IS_LINUX and _is_supervised_gateway_process()
        if in_supervised_gateway and _systemd_run_user_scope_available():
            session.systemd_unit = f"hermes-worker-{unit_suffix}.scope"
            return _build_systemd_scope_argv(argv, unit_suffix=unit_suffix), True
        if in_supervised_gateway:
            # Under a supervisor but no private cgroup: an OOM in the worker can
            # still take the whole gateway down.
            logger.debug(
                "%s background executor not isolated in a systemd scope "
                "(systemd-run --user unavailable); worker shares the gateway cgroup.",
                label,
            )
        return argv, False

    def _track_started(self, session: ProcessSession, reader_target, reader_name: str) -> None:
        """Start the output reader thread, register the session and checkpoint it."""
        reader = threading.Thread(
            target=reader_target, args=(session,), daemon=True, name=reader_name,
        )
        session._reader_thread = reader
        reader.start()
        with self._lock:
            self._prune_if_needed()
            self._running[session.id] = session
        self._write_checkpoint()

    def _spawn_local_pty(self, session: ProcessSession, safe_command: str, env_vars: dict) -> ProcessSession:
        """PTY spawn for interactive CLI tools (Codex, Claude Code, REPLs).

        Raises ImportError when no PTY backend is installed and re-raises any spawn
        failure; ``spawn_local`` falls back to pipe mode in both cases.
        """
        if _IS_WINDOWS:
            from winpty import PtyProcess as _PtyProcessCls
        else:
            from ptyprocess import PtyProcess as _PtyProcessCls
        user_shell = _find_shell()
        pty_env = _sanitize_subprocess_env(os.environ, env_vars)
        pty_env["PYTHONUNBUFFERED"] = "1"
        # A PTY is a real TTY, so pager-happy tools (git log/diff, man) WILL page and
        # hang waiting for `q` — default them to cat, honoring any pager the user set.
        pty_env.setdefault("GIT_PAGER", "cat")
        pty_env.setdefault("PAGER", "cat")
        pty_argv, _ = self._scope_argv(
            session, [user_shell, "-lic", f"set +m; {safe_command}"], session.id, "PTY",
        )
        pty_proc = _PtyProcessCls.spawn(
            pty_argv,
            cwd=session.cwd,
            env=pty_env,
            dimensions=(30, 120),
        )
        session.pid = pty_proc.pid
        session.host_start_time = self._safe_host_start_time(session.pid)
        session._pty = pty_proc
        self._track_started(session, self._pty_reader_loop, f"proc-pty-reader-{session.id}")
        return session

    def spawn_local(
        self,
        command: str,
        cwd: str = None,
        task_id: str = "",
        session_key: str = "",
        env_vars: dict = None,
        use_pty: bool = False,
        owner_task_id: str = "",
    ) -> ProcessSession:
        """Spawn a background process locally (TERMINAL_ENV=local only; other
        backends use spawn_via_env()).

        ``use_pty`` requests a pseudo-terminal via ptyprocess/pywinpty for interactive
        CLI tools; it falls back to a plain pipe when that is unavailable or fails.
        """
        # Bash parses ``A && B &`` as ``(A && B) &`` — a subshell that holds our stdout
        # pipe open forever when B is a long-running server. The rewriter turns it into
        # ``A && { B & }``. Lazy import: terminal_tool imports this module.
        from tools.terminal_tool import _rewrite_compound_background as _rewrite_bg

        safe_command = _rewrite_bg(command)

        session = ProcessSession(
            id=f"proc_{uuid.uuid4().hex[:12]}",
            command=command,
            task_id=task_id,
            owner_task_id=owner_task_id or task_id,
            session_key=session_key,
            cwd=_resolve_safe_cwd(cwd or os.getcwd()),
            started_at=time.time(),
        )

        pty_scope_attempted = False
        if use_pty:
            try:
                return self._spawn_local_pty(session, safe_command, env_vars)
            except ImportError:
                logger.warning("ptyprocess not installed, falling back to pipe mode")
            except Exception as e:
                logger.warning("PTY spawn failed (%s), falling back to pipe mode", e)
                if session.systemd_unit:
                    pty_scope_attempted = True
                    if not _stop_systemd_unit(session.systemd_unit):
                        raise RuntimeError(
                            "PTY scope could not be reaped; refusing pipe fallback "
                            "to avoid duplicate command execution"
                        ) from e
                    session.systemd_unit = ""

        # Pipe path (non-PTY or PTY fallback). The user's login shell keeps parity with
        # LocalEnvironment (rc files sourced, user tools on PATH). PYTHONUNBUFFERED so
        # tqdm/datasets-style buffering doesn't hide progress from process(action="poll").
        user_shell = _find_shell()
        bg_env = _sanitize_subprocess_env(os.environ, env_vars)
        bg_env["PYTHONUNBUFFERED"] = "1"
        _popen_kwargs = {"creationflags": windows_hide_flags()} if _IS_WINDOWS else {}

        unit_suffix = f"{session.id}-pipe-fallback" if pty_scope_attempted else session.id
        spawn_argv, _ = self._scope_argv(
            session, [user_shell, "-lic", f"set +m; {safe_command}"], unit_suffix, "Local",
        )

        # start_new_session is REQUIRED with systemd-run --scope too: the scope does not
        # give the worker a new session, so from an interactive TUI the worker would
        # share the foreground process group and background spawns would stop the whole
        # session (observed as dead TUIs in state T). Cgroup isolation is unaffected —
        # the scope attaches to the invoked process, not the spawning session.
        proc = subprocess.Popen(
            spawn_argv,
            text=True,
            cwd=session.cwd,
            env=bg_env,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            **_popen_kwargs,
        )

        session.process = proc
        session.pid = proc.pid
        session.host_start_time = self._safe_host_start_time(session.pid)

        try:
            self._track_started(session, self._reader_loop, f"proc-reader-{session.id}")
        except Exception:
            # Post-Popen setup failed — kill the orphaned subprocess (and any setsid
            # descendants) before re-raising so nothing leaks untracked.
            try:
                if session.systemd_unit:
                    # Scope teardown is the authoritative cleanup for the worker cgroup
                    # (never killpg here); the wrapper PID is terminated as fallback.
                    _stop_systemd_unit(session.systemd_unit)
                    self._terminate_host_pid(proc.pid, session.host_start_time)
                elif not _IS_WINDOWS:
                    try:
                        kill_signal = getattr(signal, "SIGKILL", signal.SIGTERM)
                        os.killpg(os.getpgid(proc.pid), kill_signal)  # windows-footgun: ok - guarded by _IS_WINDOWS above
                    except (ProcessLookupError, PermissionError, OSError):
                        proc.kill()
                else:
                    proc.kill()
            except Exception:
                pass
            try:
                proc.wait(timeout=5)
            except Exception:
                pass
            raise

        return session

    def spawn_via_env(
        self,
        env: Any,
        command: str,
        cwd: str = None,
        task_id: str = "",
        session_key: str = "",
        timeout: int = 10,
        owner_task_id: str = "",
    ) -> ProcessSession:
        """Spawn a background process inside a non-local backend's sandbox.

        The command is wrapped to capture its in-sandbox PID and redirect output to
        a log file, which later execute() calls poll. Less capable than local spawn
        (no live pipe, no stdin) but runs in the correct sandbox context.
        """
        session = ProcessSession(
            id=f"proc_{uuid.uuid4().hex[:12]}",
            command=command,
            task_id=task_id,
            owner_task_id=owner_task_id or task_id,
            session_key=session_key,
            cwd=cwd,
            started_at=time.time(),
            env_ref=env,
            pid_scope="sandbox",
        )

        # Run the command in the sandbox with output capture
        temp_dir = self._env_temp_dir(env)
        log_path = f"{temp_dir}/hermes_bg_{session.id}.log"
        pid_path = f"{temp_dir}/hermes_bg_{session.id}.pid"
        exit_path = f"{temp_dir}/hermes_bg_{session.id}.exit"
        quoted_command = shlex.quote(command)
        quoted_temp_dir = shlex.quote(temp_dir)
        quoted_log_path = shlex.quote(log_path)
        quoted_pid_path = shlex.quote(pid_path)
        quoted_exit_path = shlex.quote(exit_path)
        bg_command = (
            f"mkdir -p {quoted_temp_dir} && "
            f"( nohup bash -lc {quoted_command} > {quoted_log_path} 2>&1; "
            f"rc=$?; printf '%s\\n' \"$rc\" > {quoted_exit_path} ) & "
            f"echo $! > {quoted_pid_path} && cat {quoted_pid_path}"
        )

        try:
            result = env.execute(
                bg_command,
                timeout=timeout,
                rewrite_compound_background=False,
            )
            output = result.get("output", "").strip()
            # Try to extract the PID from the output
            for line in output.splitlines():
                line = line.strip()
                if line.isdigit():
                    session.pid = int(line)
                    break
            # No PID from the wrapper (syntax error, broken redirect): a failed launch,
            # not a fake running session.
            if session.pid is None:
                session.exited = True
                session.exit_code = int(result.get("returncode", -1))
                if session.exit_code == 0:
                    session.exit_code = -1
                session.completion_reason = "failed_start"
                session.termination_source = "failed_start"
                session.output_buffer = result.get("output", "").strip()
        except Exception as e:
            session.exited = True
            session.exit_code = -1
            session.completion_reason = "failed_start"
            session.termination_source = "failed_start"
            session.output_buffer = f"Failed to start: {e}"

        if not session.exited:
            # Start a poller thread that periodically reads the log file
            reader = threading.Thread(
                target=self._env_poller_loop,
                args=(session, env, log_path, pid_path, exit_path),
                daemon=True,
                name=f"proc-poller-{session.id}",
            )
            session._reader_thread = reader
            reader.start()

        with self._lock:
            self._prune_if_needed()
            if not session.exited:
                self._running[session.id] = session

        if not session.exited:
            self._write_checkpoint()

        return session

    # ----- Reader / Poller Threads -----

    def _reader_loop(self, session: ProcessSession):
        """Background thread: read stdout from a local Popen process.

        Uses ``buffer.read1(4096)`` not ``TextIOWrapper.read(4096)``: on pipes the
        latter blocks until EOF, landing "live" output in one burst at exit.

        Orphaned-pipe guard: a backgrounded grandchild (``node server.js &``)
        inherits our pipe's write end, so EOF never arrives while it lives — a
        blocking read would park this thread, ``session.exited`` would never flip
        and ``notify_on_complete`` never fire (``_reconcile_local_exit`` only runs
        lazily from poll()/wait()). On POSIX we ``select()`` with a short interval
        and stop draining shortly after the direct child exits, mirroring
        ``tools/environments/base.py::_wait_for_process``. Windows pipes lack
        select(); the blocking path stays and the lazy reconcile is the safety net.
        """
        first_chunk = True
        # A multibyte UTF-8 char split across read1() chunks would become U+FFFD
        # mojibake with stateless decoding; the incremental decoder holds the partial
        # sequence until the continuation bytes arrive.
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

        def _append_chunk(chunk: str):
            nonlocal first_chunk
            if first_chunk:
                chunk = self._clean_shell_noise(chunk)
                first_chunk = False
            self._ingest_output(session, chunk)

        try:
            proc = session.process
            if proc is None or proc.stdout is None:
                return
            stdout = proc.stdout

            raw_read = getattr(getattr(stdout, "buffer", None), "read1", None)

            # select() needs a real OS fd; mocked streams (tests, adapters) may lack
            # fileno() and use the blocking loop instead.
            fd = None
            if raw_read is not None and not _IS_WINDOWS:
                fileno = getattr(stdout, "fileno", None)
                try:
                    candidate = fileno() if callable(fileno) else None
                except Exception:
                    candidate = None
                if isinstance(candidate, int) and candidate >= 0:
                    fd = candidate

            if fd is not None:
                import select as _select

                idle_after_exit = 0
                while True:
                    try:
                        ready, _, _ = _select.select([fd], [], [], 0.2)
                    except (ValueError, OSError):
                        break  # fd already closed
                    if ready:
                        raw = raw_read(4096)
                        if not raw:
                            break  # true EOF — all writers closed
                        chunk = decoder.decode(raw)
                        if chunk:
                            _append_chunk(chunk)
                        idle_after_exit = 0
                    elif proc.poll() is not None:
                        # Direct child gone and pipe idle ~200ms: a few more cycles
                        # for a buffered tail, then stop rather than wait forever on
                        # an orphaned grandchild's pipe.
                        idle_after_exit += 1
                        if idle_after_exit >= 3:
                            break
            else:
                while True:
                    if raw_read is not None:
                        raw = raw_read(4096)
                        if not raw:
                            break
                        chunk = decoder.decode(raw)
                        if not chunk:
                            continue  # partial multibyte sequence — wait for more bytes
                    else:
                        # Mocked/alternate streams without a raw buffer: less "live".
                        chunk = stdout.read(4096)
                        if not chunk:
                            break

                    _append_chunk(chunk)
        except Exception as e:
            logger.debug("Process stdout reader ended: %s", e)
        finally:
            # Flush the decoder: a truncated multibyte sequence at EOF becomes one
            # U+FFFD instead of vanishing.
            try:
                tail = decoder.decode(b"", final=True)
                if tail:
                    _append_chunk(tail)
            except Exception:
                pass
            # Always reap the child to prevent zombie processes.
            try:
                session.process.wait(timeout=5)
            except Exception as e:
                logger.debug("Process wait timed out or failed: %s", e)
            self._finish_exited(session, session.process.returncode)

    def _env_poller_loop(
        self, session: ProcessSession, env: Any, log_path: str, pid_path: str, exit_path: str
    ):
        """Background thread: poll a sandbox log file for non-local backends."""
        quoted_log_path = shlex.quote(log_path)
        quoted_pid_path = shlex.quote(pid_path)
        quoted_exit_path = shlex.quote(exit_path)
        prev_output_len = 0  # track delta for watch pattern scanning
        while not session.exited:
            time.sleep(2)  # Poll every 2 seconds
            try:
                # Read new output from the log file
                result = env.execute(f"cat {quoted_log_path} 2>/dev/null", timeout=10)
                new_output = result.get("output", "")
                if new_output:
                    # Delta since the previous read feeds watch-pattern scanning.
                    delta = new_output[prev_output_len:] if len(new_output) > prev_output_len else ""
                    prev_output_len = len(new_output)
                    with session._lock:
                        session.output_buffer = new_output[-session.max_output_chars:]
                    if delta:
                        self._check_watch_patterns(session, delta)
                        self._emit_output(session, delta)

                # Check if process is still running
                check = env.execute(
                    f"kill -0 \"$(cat {quoted_pid_path} 2>/dev/null)\" 2>/dev/null; echo $?",
                    timeout=5,
                )
                check_output = check.get("output", "").strip()
                if check_output and check_output.splitlines()[-1].strip() != "0":
                    # Process has exited -- get exit code captured by the wrapper shell.
                    exit_result = env.execute(
                        f"cat {quoted_exit_path} 2>/dev/null",
                        timeout=5,
                    )
                    exit_str = exit_result.get("output", "").strip()
                    try:
                        session.exit_code = int(exit_str.splitlines()[-1].strip())
                    except (ValueError, IndexError):
                        session.exit_code = -1
                    session.exited = True
                    if session.completion_reason != "killed":
                        session.completion_reason = "exited"
                    self._move_to_finished(session)
                    return

            except Exception:
                # Environment might be gone (sandbox reaped, etc.)
                session.exited = True
                session.exit_code = -1
                session.completion_reason = "lost"
                session.termination_source = "backend_lost"
                self._move_to_finished(session)
                return

    def _pty_reader_loop(self, session: ProcessSession):
        """Background thread: read output from a PTY process."""
        pty = session._pty
        # Same split-multibyte handling as _reader_loop.
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

        _append_text = lambda text: self._ingest_output(session, text)  # noqa: E731

        try:
            while pty.isalive():
                try:
                    chunk = pty.read(4096)
                    if chunk:
                        # ptyprocess returns bytes; pywinpty returns str
                        text = chunk if isinstance(chunk, str) else decoder.decode(chunk)
                        if text:
                            _append_text(text)
                except Exception:  # EOFError included
                    break
        except Exception as e:
            logger.debug("PTY stdout reader ended: %s", e)

        # Flush any partial multibyte sequence held by the decoder.
        try:
            tail = decoder.decode(b"", final=True)
            if tail:
                _append_text(tail)
        except Exception:
            pass

        try:
            pty.wait()
        except Exception as e:
            logger.debug("PTY wait timed out or failed: %s", e)
        self._finish_exited(session, pty.exitstatus if hasattr(pty, 'exitstatus') else -1)

    def _ingest_output(self, session: ProcessSession, text: str) -> None:
        """Buffer a freshly-read chunk, then scan watch patterns and stream it live."""
        session.append_output(text)
        self._check_watch_patterns(session, text)
        self._emit_output(session, text)

    def _finish_exited(self, session: ProcessSession, exit_code) -> None:
        """Mark a reader-observed exit and move the session to finished.

        A kill that raced the reader already recorded its own exit_code/reason;
        don't overwrite it.
        """
        session.exited = True
        if session.completion_reason != "killed":
            session.exit_code = exit_code
            session.completion_reason = "exited"
        self._move_to_finished(session)

    def _move_to_finished(self, session: ProcessSession):
        """Move a session from running to finished.

        Idempotent: kill_process() and the reader thread can both call this; only
        the FIRST move enqueues the completion notification, so no duplicates.
        """
        with self._lock:
            was_running = self._running.pop(session.id, None) is not None
            self._finished[session.id] = session
        session._completion_event.set()
        self._write_checkpoint()

        if was_running and session.notify_on_complete:
            from tools.ansi_strip import strip_ansi
            output_tail = strip_ansi(session.output_buffer[-2000:]) if session.output_buffer else ""
            notification = {
                "type": "completion",
                "session_id": session.id,
                "session_key": session.session_key,
                "task_id": session.task_id,
                "owner_task_id": session.owner_task_id or session.task_id,
                "command": session.command,
                "exit_code": session.exit_code,
                "completion_reason": session.completion_reason,
                "termination_source": session.termination_source,
                "output": output_tail,
                # Stable producer identity across checkpoint recovery (unlike a
                # consumer-observed completion timestamp).
                "started_at": session.started_at,
            }
            _redact_process_result(notification)
            self.completion_queue.put(notification)

    # ----- Query Methods -----

    def is_completion_consumed(self, session_id: str) -> bool:
        """Check if a completion notification was already consumed via wait/log."""
        return session_id in self._completion_consumed

    def is_session_waiting(self, session_id: str) -> bool:
        """Whether a goal loop (``hermes_cli.goals`` wait barrier) should stay parked
        on this session: still running AND, if it has ``watch_patterns``, none has
        matched yet (a long-lived watcher unblocks on its trigger, not on exit).
        Unknown/exited/already-fired sessions return False so a stale barrier can
        never wedge the loop."""
        if not session_id:
            return False
        with self._lock:
            session = self._running.get(session_id) or self._finished.get(session_id)
        if session is None:
            return False
        try:
            self._refresh_detached_session(session)
        except Exception:
            pass
        if session.exited:
            return False
        return not (session.watch_patterns and not session._watch_disabled and session._watch_hits > 0)

    def wait_for_pending_completions(
        self,
        task_id: Optional[str] = None,
        *,
        timeout: float | None = None,
        poll_interval: float = 1.0,
    ) -> dict:
        """Bounded linger for ``notify_on_complete`` background processes at one-shot exit.

        A one-shot CLI run (``hermes -q/-Q/-z``) exits when its turn ends; any
        background process it spawned still holds a stdout pipe owned by the dying
        parent and dies of SIGPIPE seconds later (Bot Mode handoff replies sent via
        message_agent/bot_relay were the visible casualty). Only ``notify_on_complete``
        processes carry a completion contract — servers/daemons/watchers aren't the
        parent's to wait for.

        ``task_id=None`` waits on every tracked process (a one-shot process hosts one
        agent). ``timeout=None`` reads ``terminal.oneshot_completion_wait_seconds``;
        ``<= 0`` disables. Each ``poll_interval`` pass re-reconciles child state so an
        orphaned-pipe exit can't wedge the linger. Returns
        ``{"waited": [...], "completed": [...], "timed_out": [...]}`` of session ids.
        """
        if timeout is None:
            timeout = self._oneshot_completion_wait_seconds()
        result: dict = {"waited": [], "completed": [], "timed_out": []}
        with self._lock:
            pending = [
                s
                for s in self._running.values()
                if s.notify_on_complete
                and not s.exited
                and (task_id is None or s.task_id == task_id)
            ]
        if not pending or timeout <= 0:
            return result
        result["waited"] = [s.id for s in pending]
        logger.info(
            "One-shot exit lingering (bounded %ss) for %d notify_on_complete "
            "background process(es): %s",
            timeout,
            len(pending),
            ", ".join(s.id for s in pending),
        )
        deadline = time.monotonic() + max(float(timeout), 0.0)
        interval = max(float(poll_interval), 0.05)
        try:
            from tools.interrupt import is_interrupted as _is_interrupted
        except Exception:
            def _is_interrupted() -> bool:
                return False
        interrupted = False
        for session in pending:
            try:
                while not session.exited:
                    if interrupted or _is_interrupted():
                        interrupted = True
                        break
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    # Reconcile first so orphaned-pipe and detached exits fire the event.
                    try:
                        self._reconcile_local_exit(session)
                        self._refresh_detached_session(session)
                    except Exception:
                        pass
                    if session.exited:
                        break
                    session._completion_event.wait(min(remaining, interval))
            except KeyboardInterrupt:
                # Stop waiting, but never let the interrupt skip the caller's durable
                # teardown (session flush, end_session) that follows.
                interrupted = True
            if session.exited:
                result["completed"].append(session.id)
            else:
                result["timed_out"].append(session.id)
        if result["timed_out"]:
            logger.warning(
                "One-shot exit linger timed out after %ss with %d background "
                "process(es) still running: %s — they may be killed when this "
                "process exits.",
                timeout,
                len(result["timed_out"]),
                ", ".join(result["timed_out"]),
            )
        return result

    @staticmethod
    def _oneshot_completion_wait_seconds() -> float:
        """Bounded linger (s) for one-shot exits with pending notify_on_complete
        processes: ``terminal.oneshot_completion_wait_seconds`` (0 disables), 600
        if config is unreadable."""
        try:
            return max(float(ProcessRegistry._config_value("terminal", "oneshot_completion_wait_seconds", 600.0)), 0.0)
        except Exception:
            return 600.0

    def _drain_should_skip(
        self, session_id: str, *, skip_poll_observed: bool = True
    ) -> bool:
        """Skip a completion the CLI agent already has this turn — consumed via
        wait/log or observed inline via poll(). Gateway/tui watchers check only
        ``is_completion_consumed`` so a read-only poll never suppresses their
        autonomous delivery turn."""
        return session_id in self._completion_consumed or (
            skip_poll_observed and session_id in self._poll_observed
        )

    @staticmethod
    def _surface_child_process_notifications() -> bool:
        """Whether subagent-owned process notifications surface in the parent
        (``delegation.surface_child_process_notifications``; suppress on any config
        error — never crash the drain loop)."""
        try:
            return bool(ProcessRegistry._config_value("delegation", "surface_child_process_notifications", False))
        except Exception:
            return False

    def drain_notifications(
        self,
        session_key: str = "",
        owns_event=None,
        *,
        skip_poll_observed: bool = True,
    ) -> "list[tuple[dict, str]]":
        """Pop all pending events and return ``(raw_event, formatted_text)`` pairs.

        Skips completions per ``_drain_should_skip``; gateway/TUI callers pass
        ``skip_poll_observed=False``.

        Routing: async-delegation events always need ownership proof; ordinary
        events need it once they carry ``session_key`` or ``origin_ui_session_id``.
        ``owns_event(evt)`` (strongest; the TUI passes a compression-chain-aware
        check so a post-compression session still claims its pre-compression
        dispatches) consumes ONLY on True; ``session_key`` uses plain equality.
        Non-owned routed events are re-queued for their owner. With no filter every
        event is consumed (legacy single-session), except restored delegation
        payloads, which stay fail-closed.
        """
        results: "list[tuple[dict, str]]" = []
        requeue: "list[dict]" = []
        # delegation.surface_child_process_notifications, read at most once per drain
        # and only when an sa- event shows up.
        surface_child: "bool | None" = None
        while not self.completion_queue.empty():
            try:
                evt = self.completion_queue.get_nowait()
            except Exception:
                break
            is_async_delegation = evt.get("type") == "async_delegation"
            evt_session_key = str(evt.get("session_key") or "")
            evt_origin_sid = str(evt.get("origin_ui_session_id") or "")
            requires_positive_proof = is_async_delegation or bool(
                evt_session_key or evt_origin_sid
            )
            if owns_event is not None and requires_positive_proof:
                try:
                    owned = bool(owns_event(evt))
                except Exception:
                    owned = False  # fail closed — never leak on a broken check
                if not owned:
                    requeue.append(evt)
                    continue
            elif session_key and requires_positive_proof:
                if evt_session_key != session_key:
                    requeue.append(evt)
                    continue
            elif is_async_delegation and evt.get("restored"):
                # Restored payloads from a previous process: an unfiltered drain
                # cannot prove ownership, so leave them for the owner.
                requeue.append(evt)
                continue
            # Routing happened first so a foreign session cannot drop the owner's
            # event via its own consumed/observed state.
            _evt_sid = evt.get("session_id", "")
            if evt.get("type") == "completion" and self._drain_should_skip(
                _evt_sid, skip_poll_observed=skip_poll_observed
            ):
                continue

            # Subagent-owned process notifications are suppressed by default — the
            # child's delegation result is the deliverable. Judge ownership on
            # owner_task_id (RAW spawning id; task_id is the container key, collapsed
            # by _resolve_container_task_id). Dropped, NOT requeued: children never
            # drain, so a requeue would pin the event forever. 'async_delegation'
            # is the result itself and is NEVER suppressed.
            _evt_task_id = str(
                evt.get("owner_task_id") or evt.get("task_id") or ""
            )
            if not is_async_delegation and _evt_task_id.startswith("sa-"):
                if surface_child is None:
                    surface_child = self._surface_child_process_notifications()
                if not surface_child:
                    logger.debug(
                        "Suppressed subagent-owned process notification "
                        "(delegation.surface_child_process_notifications=false): "
                        "type=%s session_id=%s task_id=%s",
                        evt.get("type", "completion"),
                        _evt_sid,
                        _evt_task_id,
                    )
                    continue

            text = format_process_notification(evt)
            if text:
                results.append((evt, text))
        for evt in requeue:
            self.completion_queue.put(evt)
        return results

    # Minimum characters of the random suffix required for prefix resolution.
    # Short prefixes ("p", "pr", "proc_1") are too collision-prone to act on.
    _MIN_PREFIX_CHARS = 4

    def get(self, session_id: str) -> Optional[ProcessSession]:
        """Get a session by full ID or unique prefix (``proc_4dae`` / bare ``4dae``,
        like git short hashes). Ambiguous or too-short prefixes resolve to None,
        never to an arbitrary pick."""
        with self._lock:
            session = self._running.get(session_id) or self._finished.get(session_id)
        if session is None:
            session = self._resolve_prefix(session_id)
        return self._refresh_detached_session(session)

    def _resolve_prefix(self, session_id: str) -> Optional[ProcessSession]:
        """Resolve a unique session-ID prefix (prefix-only, unique hit; a bare hex
        tail is normalized to ``proc_<tail>``). :meth:`get` tries exact first."""
        if not session_id or not isinstance(session_id, str):
            return None
        query = session_id.strip()
        if not query:
            return None
        # Allow the bare suffix form: "4dae56" -> "proc_4dae56".
        if not query.startswith("proc_"):
            query = f"proc_{query}"
        suffix = query[len("proc_"):]
        if len(suffix) < self._MIN_PREFIX_CHARS:
            return None
        with self._lock:
            matches = [
                s
                for store in (self._running, self._finished)
                for sid, s in store.items()
                if sid.startswith(query)
            ]
        if len(matches) == 1:
            return matches[0]
        return None

    def _reconcile_local_exit(self, session: "ProcessSession") -> None:
        """Reconcile ``session.exited`` against the real child state.

        The reader flips ``exited`` only at EOF; when the direct child has exited but
        a descendant (e.g. a daemon from ``hermes update``) holds the pipe open, poll()
        would report "running" forever. If ``Popen.poll()`` has an exit code, drain
        readable bytes non-blocking and flip ``exited``; the stuck daemon reader thread
        is reaped with the process. No-op for env/PTY, exited and detached sessions.
        """
        if session is None or session.exited:
            return
        proc = getattr(session, "process", None)
        if proc is None:
            return
        try:
            rc = proc.poll()
        except Exception:
            return
        if rc is None:
            return  # Direct child still running — reader block is legitimate.

        # Best-effort non-blocking drain of whatever the reader hasn't consumed.
        drained = ""
        stdout = getattr(proc, "stdout", None)
        if stdout is not None and not _IS_WINDOWS:
            try:
                import fcntl
                fd = stdout.fileno()
                flags = fcntl.fcntl(fd, fcntl.F_GETFL)
                fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
                try:
                    chunk = stdout.read()
                    if chunk:
                        drained = chunk if isinstance(chunk, str) else chunk.decode("utf-8", errors="replace")
                except (BlockingIOError, OSError, ValueError):
                    pass
                finally:
                    try:
                        fcntl.fcntl(fd, fcntl.F_SETFL, flags)
                    except Exception:
                        pass
            except Exception as e:
                logger.debug("Non-blocking drain failed for %s: %s", session.id, e)

        with session._lock:
            if drained:
                session.output_buffer += drained
                if len(session.output_buffer) > session.max_output_chars:
                    session.output_buffer = session.output_buffer[-session.max_output_chars:]
            session.exited = True
            if session.completion_reason != "killed":
                session.exit_code = rc
                session.completion_reason = "exited"
        logger.info(
            "Reconciled session %s: direct child exited with code %s but reader "
            "was still blocked (orphaned pipe). Flipped to exited.",
            session.id, rc,
        )
        self._move_to_finished(session)

    def poll(self, session_id: str) -> dict:
        """Check status and get new output for a background process."""
        from tools.ansi_strip import strip_ansi

        session = self.get(session_id)
        if session is None:
            return {"status": "not_found", "error": f"No process with ID {session_id}"}

        self._reconcile_local_exit(session)  # orphaned-pipe reader guard

        with session._lock:
            output_preview = strip_ansi(session.output_buffer[-1000:]) if session.output_buffer else ""

        result = {
            "session_id": session.id,
            "command": session.command,
            "status": "exited" if session.exited else "running",
            "pid": session.pid,
            "uptime_seconds": int(time.time() - session.started_at),
            "output_preview": output_preview,
        }
        if session.exited:
            result["exit_code"] = session.exit_code
            result["completion_reason"] = session.completion_reason
            result["termination_source"] = session.termination_source
            # Read-only: record in _poll_observed (CLI inline dedup) but NOT in
            # _completion_consumed, or a status check would suppress the watcher's
            # autonomous delivery turn. See __init__.
            self._poll_observed.add(session_id)
        if session.detached:
            result["detached"] = True
            result["note"] = "Process recovered after restart -- output history unavailable"
        return result

    def read_log(self, session_id: str, offset: int | None = None, limit: int = 200) -> dict:
        """Read the full output log with optional pagination by lines."""
        from tools.ansi_strip import strip_ansi

        session = self.get(session_id)
        if session is None:
            return {"status": "not_found", "error": f"No process with ID {session_id}"}

        with session._lock:
            full_output = strip_ansi(session.output_buffer)

        lines = full_output.splitlines()
        total_lines = len(lines)

        # offset=None -> last N lines; an explicit offset=0 means the HEAD (don't
        # conflate the two via falsiness).
        if offset is None and limit > 0:
            selected = lines[-limit:]
            observed_completion_output = bool(selected) or total_lines == 0
        else:
            offset = offset or 0
            selected = lines[offset:offset + limit]
            stop = slice(offset, offset + limit).indices(total_lines)[1]
            observed_completion_output = (
                total_lines == 0 or (bool(selected) and stop == total_lines)
            )

        result = {
            "session_id": session.id,
            "command": session.command,
            "status": "exited" if session.exited else "running",
            "output": "\n".join(selected),
            "total_lines": total_lines,
            "showing": f"{len(selected)} lines",
        }
        if session.exited and observed_completion_output:
            self._completion_consumed.add(session_id)
        return result

    def wait(self, session_id: str, timeout: int = None) -> dict:
        """Block until the process exits, the timeout elapses, or the user interrupts.

        ``timeout`` defaults to (and is clamped by) TERMINAL_TIMEOUT. Returns a dict
        with status exited|timeout|interrupted|not_found|error and an output snapshot.
        """
        from tools.ansi_strip import strip_ansi
        from tools.interrupt import is_interrupted as _is_interrupted

        try:
            default_timeout = int(os.getenv("TERMINAL_TIMEOUT", "180"))
        except (ValueError, TypeError):
            default_timeout = 180
        max_timeout = default_timeout
        requested_timeout = timeout
        timeout_note = None

        # The schema says minimum=1 but not every caller enforces it; timeout=0 is
        # falsy and would silently fall through to the default wait.
        if requested_timeout is not None and requested_timeout <= 0:
            return {
                "status": "error",
                "error": f"timeout must be positive (got {requested_timeout})",
            }

        if requested_timeout and requested_timeout > max_timeout:
            effective_timeout = max_timeout
            timeout_note = (
                f"Requested wait of {requested_timeout}s was clamped "
                f"to configured limit of {max_timeout}s"
            )
        else:
            effective_timeout = requested_timeout or max_timeout

        session = self.get(session_id)
        if session is None:
            return {"status": "not_found", "error": f"No process with ID {session_id}"}

        deadline = time.monotonic() + effective_timeout

        while time.monotonic() < deadline:
            session = self._refresh_detached_session(session)
            if session is None:
                return {"status": "not_found", "error": f"No process with ID {session_id}"}
            self._reconcile_local_exit(session)  # orphaned-pipe reader guard
            if session.exited:
                self._completion_consumed.add(session_id)
                result = self._exit_snapshot(session, "exited")
            elif _is_interrupted():
                result = {
                    "status": "interrupted",
                    "command": session.command,
                    "output": strip_ansi(session.output_buffer[-1000:]),
                    "note": "User sent a new message -- wait interrupted",
                }
            else:
                result = None
            if result is not None:
                if timeout_note:
                    result["timeout_note"] = timeout_note
                return result

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            session._completion_event.wait(timeout=min(1.0, remaining))

        result = {
            "status": "timeout",
            "command": session.command,
            "output": strip_ansi(session.output_buffer[-1000:]),
            # Not a failure — models re-issued identical waits after misreading
            # this result as an error.
            "process_running": True,
        }
        uptime = time.time() - session.started_at if session.started_at else None
        base_note = (
            f"Wait window of {effective_timeout}s elapsed — the process is "
            "still running. This is not an error."
        )
        if uptime is not None:
            base_note += f" Uptime: {int(uptime)}s."
        if session.notify_on_complete:
            base_note += (
                " notify_on_complete is set: you will be notified on exit — "
                "do more work instead of waiting again."
            )
        else:
            base_note += (
                " Poll again later or use terminal(background=true, "
                "notify_on_complete=true) next time for automatic notification."
            )
        if timeout_note:
            result["timeout_note"] = f"{timeout_note}. {base_note}"
        else:
            result["timeout_note"] = base_note
        return result

    @staticmethod
    def _exit_snapshot(session: ProcessSession, status: str) -> dict:
        """Result dict for an exited session: exit metadata + last 2000 chars of output."""
        from tools.ansi_strip import strip_ansi

        return {
            "status": status,
            "command": session.command,
            "exit_code": session.exit_code,
            "completion_reason": session.completion_reason,
            "termination_source": session.termination_source,
            "output": strip_ansi(session.output_buffer[-2000:]),
        }

    def kill_process(
        self,
        session_id: str,
        *,
        source: str = "process.kill",
        consume_output: bool = True,
    ) -> dict:
        """Kill a background process and return its output snapshot.

        ``consume_output`` is true for explicit tool/RPC kills (the caller sees the
        output). Bulk cleanup passes false so it doesn't suppress an autonomous
        completion notification — except abandoned-turn reaping
        (``kill_started_since``), which passes true so a killed abandoned process
        can't enqueue a follow-up reviving work the timeout stopped.
        """
        from tools.ansi_strip import strip_ansi

        session = self.get(session_id)
        if session is None:
            return {"status": "not_found", "error": f"No process with ID {session_id}"}

        if session.exited:
            # A double-forked descendant may still be alive in the systemd scope even
            # though the main process exited — stop the scope to reap survivors.
            if session.systemd_unit:
                _stop_systemd_unit(session.systemd_unit)
            with session._lock:
                result = self._exit_snapshot(session, "already_exited")
            # Only suppress the autonomous turn after its output is present in
            # the explicit kill result, matching wait/log consumption.
            if consume_output:
                self._completion_consumed.add(session_id)
            return result

        try:
            early = self._signal_kill(session, session_id, consume_output)
            if early is not None:
                return early

            # Additive to the PID kill: stopping the scope reaps double-forked
            # descendants reparented inside the cgroup.
            if session.systemd_unit:
                _stop_systemd_unit(session.systemd_unit)
            # Capture output, mark consumed, THEN expose ``exited`` to watcher tasks —
            # closes the delayed-notification race without losing the transcript.
            with session._lock:
                output = strip_ansi(session.output_buffer[-2000:])
                if consume_output:
                    self._completion_consumed.add(session_id)
                session.exited = True
                session.exit_code = -15  # SIGTERM
                session.completion_reason = "killed"
                session.termination_source = source
            self._move_to_finished(session)
            self._write_checkpoint()
            return {
                "status": "killed",
                "session_id": session.id,
                "completion_reason": session.completion_reason,
                "termination_source": session.termination_source,
                "output": output,
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def _signal_kill(self, session: ProcessSession, session_id: str, consume_output: bool) -> Optional[dict]:
        """Deliver the kill via PTY, local Popen tree, sandbox exec or recovered host
        PID. Returns a final result dict when the kill cannot proceed (recycled/dead
        recovered PID, or no runtime handle), else None."""
        from tools.ansi_strip import strip_ansi

        if session._pty:
            try:
                session._pty.terminate(force=True)
            except Exception:
                if session.pid:
                    os.kill(session.pid, signal.SIGTERM)
        elif session.process:
            # Tree kill: on Windows Popen.terminate() only kills the shell wrapper and
            # leaves Git Bash descendants behind.
            self._terminate_host_pid(session.process.pid, session.host_start_time)
        elif session.env_ref and session.pid:
            session.env_ref.execute(f"kill {session.pid} 2>/dev/null", timeout=5)
        elif session.detached and session.pid_scope == "host" and session.pid:
            # Identity check, not bare liveness: a gone/recycled PID means our
            # process exited — never tree-kill the stranger. Still stop an owned
            # scope: a daemonized descendant may survive the wrapper PID.
            if not self._host_pid_is_ours(session.pid, session.host_start_time):
                if session.systemd_unit:
                    _stop_systemd_unit(session.systemd_unit)
                with session._lock:
                    session.exited = True
                    session.exit_code = None
                    output = strip_ansi(session.output_buffer[-2000:])
                if consume_output:
                    self._completion_consumed.add(session_id)
                self._move_to_finished(session)
                return {"status": "already_exited", "exit_code": session.exit_code, "output": output}
            self._terminate_host_pid(session.pid, session.host_start_time)
        else:
            return {
                "status": "error",
                "error": (
                    "Recovered process cannot be killed after restart because "
                    "its original runtime handle is no longer available"
                ),
            }
        return None

    def _live_session(self, session_id: str):
        """``(session, None)`` for a running session, else ``(None, error_result)``."""
        session = self.get(session_id)
        if session is None:
            return None, {"status": "not_found", "error": f"No process with ID {session_id}"}
        if session.exited:
            return None, {"status": "already_exited", "error": "Process has already finished"}
        return session, None

    def write_stdin(self, session_id: str, data: str) -> dict:
        """Send raw data to a running process's stdin (no newline appended)."""
        session, err = self._live_session(session_id)
        if err:
            return err

        # PTY mode -- write through pty handle.
        if session._pty:
            try:
                # pywinpty expects str on Windows; ptyprocess expects bytes on POSIX.
                if _IS_WINDOWS:
                    pty_data = data.decode("utf-8") if isinstance(data, bytes) else str(data)
                else:
                    # surrogateescape: a PTY is a byte stream — round-trip the
                    # original bytes instead of crashing on surrogate content.
                    pty_data = data.encode("utf-8", "surrogateescape") if isinstance(data, str) else data
                session._pty.write(pty_data)
                return {"status": "ok", "bytes_written": len(data)}
            except Exception as e:
                return {"status": "error", "error": str(e)}

        # Popen mode -- write through stdin pipe
        if not session.process or not session.process.stdin:
            return {"status": "error", "error": "Process stdin not available (non-local backend or stdin closed)"}
        try:
            session.process.stdin.write(data)
            session.process.stdin.flush()
            return {"status": "ok", "bytes_written": len(data)}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def submit_stdin(self, session_id: str, data: str = "") -> dict:
        """Send data + newline to stdin (like pressing Enter).

        On a Windows PTY, Enter is a carriage return: ConPTY treats ``\\r`` as
        end-of-line and a bare ``\\n`` through pywinpty is NOT a line terminator —
        the child's blocking line read (``readline()``, Go ``bufio.Scanner`` in
        ``gh auth login``) never returns and the process hangs looking healthy.
        ``\\r\\n`` gives it both; POSIX PTYs and pipes keep ``\\n``.
        """
        session = self.get(session_id)
        is_windows_pty = bool(_IS_WINDOWS and session is not None and session._pty)
        return self.write_stdin(session_id, data + ("\r\n" if is_windows_pty else "\n"))

    def request_close_terminal(self, session_id: str) -> dict:
        """Ask the desktop GUI to close this process's read-only terminal tab.

        Does NOT kill the process — output keeps buffering and the tab can be
        reopened from the status stack. Errors when no UI close sink is wired."""
        sink = self.on_close
        if sink is None:
            return {
                "status": "error",
                "error": "close_terminal is only available in the Hermes desktop app.",
            }
        # The session may already be finished (or pruned) — the tab can still
        # linger and be closed, so a missing session is not an error here.
        session = self.get(session_id)
        try:
            sink(session, session_id)
        except Exception as e:
            return {"status": "error", "error": str(e)}
        return {
            "status": "ok",
            "closed": session_id,
            "note": (
                "Closed the read-only terminal tab. The process was not killed; "
                "its output remains available and the user can reopen the tab "
                "from the status stack."
            ),
        }

    def close_stdin(self, session_id: str) -> dict:
        """Close a running process's stdin / send EOF without killing the process."""
        session, err = self._live_session(session_id)
        if err:
            return err

        if session._pty:
            try:
                session._pty.sendeof()
                return {"status": "ok", "message": "EOF sent"}
            except Exception as e:
                return {"status": "error", "error": str(e)}

        if not session.process or not session.process.stdin:
            return {"status": "error", "error": "Process stdin not available (non-local backend or stdin closed)"}
        try:
            session.process.stdin.close()
            return {"status": "ok", "message": "stdin closed"}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def count_running(self) -> int:
        """O(1) count of running processes for status-bar polling; CPython dict
        ``len()`` is atomic so no lock is needed."""
        try:
            return len(self._running)
        except Exception:
            return 0

    def list_sessions(self, task_id: str = None, session_key: str = None) -> list:
        """List running and recently-finished processes for ``task_id`` and/or
        ``session_key``. Cross-task entries that share the gateway session (a
        forgotten preview server blocking session reset) are flagged
        ``"session_scoped": true``."""
        with self._lock:
            all_sessions = list(self._running.values()) + list(self._finished.values())

        all_sessions = [self._refresh_detached_session(s) for s in all_sessions]

        if task_id or session_key:
            all_sessions = [
                s for s in all_sessions
                if (task_id and s.task_id == task_id)
                or (session_key and s.session_key == session_key)
            ]

        result = []
        for s in all_sessions:
            entry = {
                "session_id": s.id,
                "command": s.command[:200],
                "cwd": s.cwd,
                "pid": s.pid,
                "started_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(s.started_at)),
                "uptime_seconds": int(time.time() - s.started_at),
                "status": "exited" if s.exited else "running",
                "output_preview": s.output_buffer[-200:] if s.output_buffer else "",
            }
            if task_id and session_key and s.task_id != task_id and s.session_key == session_key:
                entry["session_scoped"] = True
            # Trigger metadata for goal-loop judges (a watcher may never exit).
            if s.watch_patterns and not s._watch_disabled:
                entry["watch_patterns"] = list(s.watch_patterns)
                entry["watch_hit"] = s._watch_hits > 0
            if s.notify_on_complete:
                entry["notify_on_complete"] = True
            if s.exited:
                entry["exit_code"] = s.exit_code
            if s.detached:
                entry["detached"] = True
            result.append(entry)
        return result

    # ----- Session/Task Queries (for gateway integration) -----

    def _any_running(self, predicate) -> bool:
        """True if any still-running session satisfies *predicate*, after refreshing
        detached sessions so a finished-but-unreaped process reads as inactive."""
        with self._lock:
            sessions = list(self._running.values())
        for session in sessions:
            self._refresh_detached_session(session)
        with self._lock:
            return any(not s.exited and predicate(s) for s in self._running.values())

    def has_active_processes(self, task_id: str) -> bool:
        """Check if there are active (running) processes for a task_id."""
        return self._any_running(lambda s: s.task_id == task_id)

    def has_active_for_session(
        self, session_key: str, max_active_age: Optional[float] = None,
    ) -> bool:
        """Active processes for a gateway session key. Processes older than
        ``max_active_age`` seconds are ignored as stale so a forgotten
        ``http.server`` can't freeze session idle/daily reset forever; ``None``
        keeps legacy behaviour (any running process blocks)."""
        now = time.time()
        return self._any_running(
            lambda s: s.session_key == session_key
            and (max_active_age is None or (now - s.started_at) < max_active_age)
        )

    def has_any_active(self) -> bool:
        """Whether ANY background process is running — scale-to-zero must not
        suspend a gateway with live background work or the process is lost."""
        return self._any_running(lambda s: True)

    def snapshot_running_ids(self, task_id: str) -> frozenset[str]:
        """Running IDs owned by ``task_id`` — a turn-boundary marker: on timeout
        only processes absent from the starting snapshot belong to the abandoned
        turn; older ones intentionally span turns and must survive."""
        with self._lock:
            return frozenset(
                s.id
                for s in self._running.values()
                if s.task_id == task_id and not s.exited
            )

    def kill_started_since(
        self,
        task_id: str,
        baseline_ids,
        *,
        source: str,
    ) -> int:
        """Kill ``task_id`` processes created after ``baseline_ids``. Output is
        consumed so an abandoned turn can't enqueue a follow-up reviving work the
        timeout deliberately stopped."""
        return self.kill_all(
            task_id,
            exclude_ids=frozenset(baseline_ids or ()),
            source=source,
            consume_output=True,
        )

    def kill_all(
        self,
        task_id: Optional[str] = None,
        *,
        exclude_ids: frozenset = frozenset(),
        source: str = "kill_all",
        consume_output: bool = False,
    ) -> int:
        """Kill all running processes, optionally filtered by task_id. Returns count killed."""
        with self._lock:
            targets = [
                s for s in self._running.values()
                if (task_id is None or s.task_id == task_id)
                and s.id not in exclude_ids
                and not s.exited
            ]

        killed = 0
        for session in targets:
            result = self.kill_process(
                session.id,
                source=source,
                consume_output=consume_output,
            )
            if result.get("status") in {"killed", "already_exited"}:
                killed += 1
        return killed

    # ----- Cleanup / Pruning -----

    def _prune_if_needed(self):
        """Remove oldest finished sessions if over MAX_PROCESSES. Must hold _lock."""
        # First prune expired finished sessions
        now = time.time()
        expired = [
            sid for sid, s in self._finished.items()
            if (now - s.started_at) > FINISHED_TTL_SECONDS
        ]
        if len(self._running) + len(self._finished) - len(expired) >= MAX_PROCESSES:
            # Still over the limit: also drop the oldest surviving finished session.
            survivors = [sid for sid in self._finished if sid not in expired]
            if survivors:
                expired.append(min(survivors, key=lambda sid: self._finished[sid].started_at))
        for sid in expired:
            del self._finished[sid]
        # Belt-and-suspenders against module-lifetime growth: forget consumed /
        # poll-observed marks for any session no longer tracked at all.
        tracked = self._running.keys() | self._finished.keys()
        self._completion_consumed &= tracked
        self._poll_observed &= tracked

    # ----- Checkpoint (crash recovery) -----

    def _write_checkpoint(
        self,
        extra_entries: Optional[List[Dict[str, Any]]] = None,
    ):
        """Write running process metadata to checkpoint file atomically."""
        try:
            with self._lock:
                entries = []
                for s in self._running.values():
                    if not s.exited:
                        # Backfill the start time so recovery can detect PID recycling
                        # even for sessions spawned before this field existed.
                        if s.host_start_time is None and s.pid_scope == "host" and s.pid:
                            s.host_start_time = self._safe_host_start_time(s.pid)
                        entry = {"session_id": s.id, **{f: getattr(s, f) for f in _CHECKPOINT_FIELDS}}
                        # Redact inline credentials before persisting: the file lives
                        # at ~/.hermes/processes.json. Recovery uses command only for
                        # display (adoption re-validates the PID, never re-runs it),
                        # so masking is lossless.
                        entry["command"] = redact_sensitive_text(s.command, code_file=True)
                        entry["owner_task_id"] = s.owner_task_id or s.task_id
                        entries.append(entry)
                if extra_entries:
                    tracked_ids = {item.get("session_id") for item in entries}
                    entries.extend(
                        item
                        for item in extra_entries
                        if item.get("session_id") not in tracked_ids
                    )
            
            # Atomic write to avoid corruption on crash
            from utils import atomic_json_write
            atomic_json_write(CHECKPOINT_PATH, entries)
        except Exception as e:
            logger.debug("Failed to write checkpoint file: %s", e, exc_info=True)

    def recover_from_checkpoint(self) -> int:
        """
        On gateway startup, probe PIDs from checkpoint file.

        Returns the number of processes recovered as detached.
        """
        if not CHECKPOINT_PATH.exists():
            return 0

        try:
            entries = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
        except Exception:
            return 0

        recovered = 0
        unresolved_scope_entries: List[Dict[str, Any]] = []
        for entry in entries:
            pid = entry.get("pid")
            if not pid:
                continue

            pid_scope = entry.get("pid_scope", "host")
            if pid_scope != "host":
                # In-sandbox PIDs mean nothing once the environment handle is gone.
                logger.info(
                    "Skipping recovery for non-host process: %s (pid=%s, scope=%s)",
                    entry.get("command", "unknown")[:60],
                    pid,
                    pid_scope,
                )
                continue

            # Alive AND the same process: across a restart the kernel may have
            # recycled the PID onto a stranger, and adopting it would let a later
            # kill tree-kill e.g. a browser.
            recorded_start = entry.get("host_start_time")
            if not self._host_pid_is_ours(pid, recorded_start):
                if self._is_host_pid_alive(pid):
                    logger.info(
                        "Not recovering session %s: pid %d is alive but its "
                        "start time no longer matches — PID was recycled onto "
                        "an unrelated process; refusing to adopt it.",
                        entry.get("session_id", "?"), pid,
                    )
                systemd_unit = entry.get("systemd_unit", "")
                if systemd_unit and not _stop_systemd_unit(systemd_unit):
                    logger.warning(
                        "Could not reap persisted scope %s for dead wrapper pid %s; "
                        "retaining checkpoint entry for the next startup",
                        systemd_unit,
                        pid,
                    )
                    unresolved_scope_entries.append(entry)
                continue

            fields = {f: entry.get(f, _CHECKPOINT_DEFAULTS[f]) for f in _CHECKPOINT_FIELDS}
            fields.update(
                command=entry.get("command", "unknown"),
                owner_task_id=entry.get("owner_task_id", "") or entry.get("task_id", ""),
                pid=pid,
                host_start_time=recorded_start,
                pid_scope=pid_scope,
                started_at=entry.get("started_at", time.time()),
            )
            session = ProcessSession(
                id=entry["session_id"],
                detached=True,  # Can't read output, but can report status + kill
                **fields,
            )
            with self._lock:
                self._running[session.id] = session
            recovered += 1
            logger.info("Recovered detached process: %s (pid=%d)", session.command[:60], pid)

            # Re-enqueue watcher so gateway can resume notifications
            if session.watcher_interval > 0:
                self.pending_watchers.append({
                    "session_id": session.id,
                    "check_interval": session.watcher_interval,
                    "session_key": session.session_key,
                    "platform": session.watcher_platform,
                    "chat_id": session.watcher_chat_id,
                    "user_id": session.watcher_user_id,
                    "user_name": session.watcher_user_name,
                    "thread_id": session.watcher_thread_id,
                    "message_id": session.watcher_message_id,
                    "notify_on_complete": session.notify_on_complete,
                    "parent_session_id": session.parent_session_id,
                })

        self._write_checkpoint(extra_entries=unresolved_scope_entries)

        return recovered


# Module-level singleton
process_registry = ProcessRegistry()


# Notification rendering lives in tools.process_registry_notifications; the names are
# re-exported here so `from tools.process_registry import format_process_notification`
# and `patch("tools.process_registry._x")` keep resolving.
from tools.process_registry_notifications import (  # noqa: F401,E402
    _delegation_attribution_line,
    _delegation_config,
    _delegation_model_not_found,
    _delegation_model_not_found_notice,
    _format_age,
    _format_async_delegation,
    _model_not_found_patterns,
    format_process_notification,
)


# ---------------------------------------------------------------------------
# Registry -- the "process" tool schema + handler
# ---------------------------------------------------------------------------
from tools.registry import registry, tool_error

PROCESS_SCHEMA = {
    "name": "process_manage",
    # The enum names the verbs; the description keeps only non-obvious semantics
    # (write-vs-submit is the one real trap: a lone \n on a Windows PTY is not Enter).
    "description": (
        "Poll, wait on, or kill background terminal processes (from "
        "terminal(background=true)). "
        "poll: status + new output. log: full output, paged. wait: block "
        "until exit or timeout (partial output on timeout). write vs "
        "submit: submit appends Enter — use it to answer prompts; write "
        "sends raw bytes, no newline. close: EOF stdin. kill: terminate."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["list", "poll", "log", "wait", "kill", "write", "submit", "close"]
            },
            "session_id": {
                "type": "string",
                "description": "From terminal background output; any unique prefix works ('4dae' for proc_4dae56ca81f6). Required except for 'list'."
            },
            "data": {
                "type": "string",
                "description": "Stdin text for write/submit."
            },
            "timeout": {
                "type": "integer",
                "description": "Max seconds for 'wait'.",
                "minimum": 1
            },
            "offset": {
                "type": "integer",
                "description": "Log line offset (default: last 200)."
            },
            "limit": {
                "type": "integer",
                "description": "Max log lines.",
                "minimum": 1
            }
        },
        "required": ["action"]
    }
}


def _redact_process_result(result: dict) -> dict:
    """Redact secrets from background-process output before it reaches the model,
    session.db and CLI, mirroring the foreground ``terminal`` redaction so the two
    surfaces can't diverge. Respects ``security.redact_secrets``; ``redact_terminal_output``
    picks ``code_file`` from the recorded command. The command itself is redacted too.
    """
    if not isinstance(result, dict):
        return result
    from agent.redact import redact_sensitive_text, redact_terminal_output

    command = result.get("command") or ""
    for field in ("output", "output_preview"):
        value = result.get(field)
        if isinstance(value, str) and value:
            result[field] = redact_terminal_output(value, command)
    if isinstance(result.get("command"), str) and result["command"]:
        result["command"] = redact_sensitive_text(result["command"], code_file=True)
    return result


def _list_processes(task_id) -> dict:
    # Surface session-scoped background processes (e.g. a forgotten preview
    # server) in addition to this task's own — they share the gateway
    # session_key and can block session reset.
    try:
        from tools.approval import get_current_session_key
        session_key = get_current_session_key(default="") or ""
    except Exception:
        session_key = ""
    return {
        "processes": [
            _redact_process_result(p)
            for p in process_registry.list_sessions(task_id=task_id, session_key=session_key or None)
        ]
    }


# action -> (handler(session_id, args) -> dict, redact output?). Output-bearing
# actions are redacted; stdin actions return only status.
_SESSION_ACTIONS = {
    "poll": (lambda sid, a: process_registry.poll(sid), True),
    "log": (lambda sid, a: process_registry.read_log(sid, offset=a.get("offset"), limit=a.get("limit", 200)), True),
    "wait": (lambda sid, a: process_registry.wait(sid, timeout=a.get("timeout")), True),
    "kill": (lambda sid, a: process_registry.kill_process(sid), True),
    "write": (lambda sid, a: process_registry.write_stdin(sid, str(a.get("data", ""))), False),
    "submit": (lambda sid, a: process_registry.submit_stdin(sid, str(a.get("data", ""))), False),
    "close": (lambda sid, a: process_registry.close_stdin(sid), False),
}


def _handle_process(args, **kw):
    action = args.get("action", "")
    # Coerce to string — some models send session_id as an integer
    session_id = str(args.get("session_id", "")) if args.get("session_id") is not None else ""

    if action == "list":
        return json.dumps(_list_processes(kw.get("task_id")), ensure_ascii=False)
    if action in _SESSION_ACTIONS:
        if not session_id:
            return tool_error(f"session_id is required for {action}")
        handler, redact = _SESSION_ACTIONS[action]
        result = handler(session_id, args)
        return json.dumps(_redact_process_result(result) if redact else result, ensure_ascii=False)
    return tool_error(f"Unknown process action: {action}. Use: list, poll, log, wait, kill, write, submit, close")


registry.register(
    name="process_manage",
    toolset="terminal",
    schema=PROCESS_SCHEMA,
    handler=_handle_process,
    emoji="⚙️",
)
