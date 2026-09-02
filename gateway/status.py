"""Gateway runtime status helpers.

PID-file based detection of whether the gateway daemon is running (used by
send_message's check_fn to gate CLI availability). The PID file lives at
``{HERMES_HOME}/gateway.pid``, so separate homes/profiles get separate files.
"""

import copy
import hashlib
import json
import logging
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from hermes_constants import get_hermes_home, _get_platform_default_hermes_home
from typing import Any, Callable, NamedTuple, Optional
from utils import atomic_json_write
import contextlib

if sys.platform == "win32":
    import msvcrt
else:
    import fcntl

_GATEWAY_KIND = "hermes-gateway"
_RUNTIME_STATUS_FILE = "gateway_state.json"
_LOCKS_DIRNAME = "gateway-locks"
_IS_WINDOWS = sys.platform == "win32"
_UNSET = object()
_GATEWAY_LOCK_FILENAME = "gateway.lock"
_gateway_lock_handle = None
# Windows byte-range locks are mandatory for other readers. Lock a byte well
# past the JSON payload so runtime status / PID readers can still read the file
# while another process holds the mutual-exclusion lock.
_WINDOWS_LOCK_OFFSET = 1024 * 1024
_GATEWAY_RUNNING_PID_CACHE_TTL_SECONDS = 1.0
_gateway_running_pid_cache_lock = threading.Lock()
_gateway_running_pid_cache: dict[tuple[str, bool, bool], tuple[float, tuple[Any, ...], Optional[int]]] = {}

logger = logging.getLogger(__name__)


class StormInfo(NamedTuple):
    """Respawn-storm check result: start count, window, and backoff to sleep."""

    count: int
    window_s: float
    backoff_s: float


def _get_starts_log_path() -> Path:
    """Append-only start ledger for the respawn-storm breaker (distinct from ``restart_loop.json``)."""
    return get_hermes_home() / "gateway-starts.log"


def record_start_and_check_storm(
    max_starts: int = 5, window_s: float = 120.0, *, backoff_cap_s: float = 300.0
) -> Optional[StormInfo]:
    """Record this start; return :class:`StormInfo` when > ``max_starts`` landed in ``window_s``.

    Best-effort: bookkeeping failures are logged and swallowed so a broken
    ledger can never crash gateway startup.
    """
    try:
        path = _get_starts_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)

        now = datetime.now(timezone.utc).timestamp()

        existing: list[float] = []
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    existing.append(float(line))
                except ValueError:
                    continue

        existing.append(now)

        recent = [ts for ts in existing if now - ts <= window_s]
        # Ring-buffer the persisted file so it stays bounded.
        keep = max(max_starts * 4, 40)
        to_write = existing[-keep:]

        tmp = path.with_suffix(".tmp")
        tmp.write_text(
            "\n".join(repr(ts) for ts in to_write) + "\n", encoding="utf-8"
        )
        os.replace(tmp, path)

        if len(recent) > max_starts:
            backoff = min(
                backoff_cap_s, 5.0 * (2 ** min(len(recent) - max_starts, 6))
            )
            return StormInfo(count=len(recent), window_s=window_s, backoff_s=backoff)
        return None
    except Exception as _e:
        logger.debug(
            "respawn-storm breaker bookkeeping failed (non-fatal): %s", _e
        )
        return None


def _get_process_hermes_home() -> Path:
    """Process-level HERMES_HOME, skipping context-local overrides.

    Gateway identity files (PID, lock, runtime status, markers) must live in the
    home the process was launched with; ``get_hermes_home()`` honors the
    per-session ``_HERMES_HOME_OVERRIDE`` contextvar and would misroute them.
    """
    val = os.environ.get("HERMES_HOME", "").strip()
    if val:
        return Path(val)
    return _get_platform_default_hermes_home()


def _canonical_hermes_home(path: Path | str) -> Path:
    """Return a stable absolute HERMES_HOME path for persisted identity data."""
    return Path(path).expanduser().resolve(strict=False)


def _same_hermes_home(left: Path | str, right: Path | str) -> bool:
    """Compare HERMES_HOME paths with the host platform's case semantics."""
    return os.path.normcase(str(_canonical_hermes_home(left))) == os.path.normcase(
        str(_canonical_hermes_home(right))
    )


def recorded_gateway_home_conflicts(
    record: Optional[dict[str, Any]],
    *,
    expected_home: Optional[Path | str] = None,
) -> bool:
    """True when a persisted gateway record names a DIFFERENT HERMES_HOME.

    Cross-profile kill refusal: a contaminated PID record in one profile's home
    can truthfully name another profile's live gateway. Destructive callers must
    refuse when the record proves foreign ownership, or profile B's stop/restart
    SIGTERMs profile A's gateway and the supervisors enter a mutual restart loop.

    ``expected_home`` overrides the comparison base (e.g. ``profile delete`` on a
    target profile). Legacy records without ``hermes_home`` return False (they
    prove nothing; callers pair this with PID + start-time guards). A comparison
    failure returns True: destructive action + unprovable ownership => fail closed.
    """
    if not isinstance(record, dict):
        return False
    recorded_home = record.get("hermes_home")
    if not isinstance(recorded_home, str) or not recorded_home.strip():
        return False
    try:
        base = expected_home if expected_home is not None else _get_process_hermes_home()
        return not _same_hermes_home(recorded_home, base)
    except Exception:
        return True


# Mirrors hermes_cli.profiles._PROFILE_ID_RE — duplicated here because gateway
# identity code must stay import-light (hermes_constants + stdlib only).
_PROFILE_LABEL_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


def _profile_label_for_home(home: Path | str) -> Optional[str]:
    """Best-effort profile label: ``<root>/profiles/<name>`` -> name, root home -> "default", else None.

    Never raises -- diagnostics only.
    """
    try:
        canonical = _canonical_hermes_home(home)
    except Exception:
        return None
    if canonical.parent.name == "profiles" and _PROFILE_LABEL_RE.match(canonical.name):
        return canonical.name
    try:
        from hermes_constants import get_default_hermes_root

        if _same_hermes_home(canonical, get_default_hermes_root()):
            return "default"
    except Exception:
        pass
    try:
        if _same_hermes_home(canonical, _get_platform_default_hermes_home()):
            return "default"
    except Exception:
        pass
    return None


def scoped_lock_owner_label(record: Optional[dict[str, Any]]) -> Optional[str]:
    """Profile label for the gateway owning a (machine-global) scoped lock.

    Prefers the ``profile`` field stamped by :func:`acquire_scoped_lock`, then
    infers from ``hermes_home`` for older locks. None for legacy/malformed
    records so callers keep PID-only wording.
    """
    if not isinstance(record, dict):
        return None
    profile = record.get("profile")
    if isinstance(profile, str) and _PROFILE_LABEL_RE.match(profile.strip()):
        # Validated: lock files are plain JSON and this flows into log lines
        # and a suggested CLI command.
        return profile.strip()
    home = record.get("hermes_home")
    if isinstance(home, str) and home.strip():
        return _profile_label_for_home(home)
    return None


def _get_pid_path() -> Path:
    """Return the path to the gateway PID file, respecting HERMES_HOME."""
    return _get_process_hermes_home() / "gateway.pid"


def _get_gateway_lock_path(pid_path: Optional[Path] = None) -> Path:
    """Return the path to the runtime gateway lock file."""
    if pid_path is not None:
        return pid_path.with_name(_GATEWAY_LOCK_FILENAME)
    return _get_process_hermes_home() / _GATEWAY_LOCK_FILENAME


def _get_runtime_status_path() -> Path:
    """Return the persisted runtime health/status file path."""
    return _get_pid_path().with_name(_RUNTIME_STATUS_FILE)


def _get_lock_dir() -> Path:
    """Return the machine-local directory for token-scoped gateway locks."""
    override = os.getenv("HERMES_GATEWAY_LOCK_DIR")
    if override:
        return Path(override)
    state_home = Path(os.getenv("XDG_STATE_HOME", Path.home() / ".local" / "state"))
    return state_home / "hermes" / _LOCKS_DIRNAME


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# Epochs before 2000-01-01 are corrupt/hand-edited state (e.g. an accidental 0).
_EPOCH_MIN_PLAUSIBLE = 946684800.0  # 2000-01-01T00:00:00Z


def normalize_updated_at(value: Any) -> Optional[str]:
    """Coerce a persisted ``updated_at`` value to an RFC3339 string or ``None``.

    ``/api/status`` and ``/health/detailed`` promise ``string | null`` (see
    ``web/src/lib/api.ts``), but the file may come from legacy gateways (epoch
    floats), hand edits, or corruption. ``str``: accepted iff fromisoformat
    parses it (trailing ``Z`` tolerated; naive -> UTC). ``int``/``float``:
    epoch seconds; before 2000-01-01, > 1 day in the future, or non-finite ->
    None. ``bool`` (an int subclass) and anything else -> None.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, str):
        raw = value.strip()
        # Python < 3.11 fromisoformat rejects a trailing 'Z'; tolerate it.
        if raw.endswith(("Z", "z")):
            raw = raw[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(raw)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.isoformat()
    if isinstance(value, (int, float)):
        seconds = float(value)
        if seconds != seconds or seconds in (float("inf"), float("-inf")):
            return None
        now = datetime.now(timezone.utc).timestamp()
        if seconds < _EPOCH_MIN_PLAUSIBLE or seconds > now + 86400:
            return None
        try:
            return datetime.fromtimestamp(seconds, tz=timezone.utc).isoformat()
        except (OverflowError, OSError, ValueError):
            return None
    return None


def _assert_process_start_time_matches(
    pid: int, expected_start_time: Optional[float]
) -> None:
    """Fail closed unless ``pid`` still names the recorded process object."""
    if expected_start_time is None:
        raise OSError(
            f"refusing to force-kill PID {pid} without a process start-time guard"
        )
    current_start_time = _get_process_start_time(pid)
    if current_start_time is None:
        raise OSError(
            f"refusing to force-kill PID {pid}; process start time is unavailable"
        )
    try:
        expected = float(expected_start_time)
        current = float(current_start_time)
    except (TypeError, ValueError) as exc:
        raise OSError(f"refusing to force-kill PID {pid}; malformed start time") from exc
    if expected <= 0 or current <= 0 or abs(expected - current) > 0.001:
        raise OSError(f"refusing to force-kill PID {pid}; process identity changed")


def terminate_pid(
    pid: int,
    *,
    force: bool = False,
    expected_start_time: Optional[float] = None,
) -> None:
    """Terminate a PID; POSIX SIGTERM/SIGKILL, Windows taskkill /T /F for force.

    Identity guard: on Windows ``force=True`` REQUIRES a matching
    ``expected_start_time`` (taskkill /T /F on a recycled PID has killed
    svchost.exe). On POSIX it is optional, but a provided, mismatched
    fingerprint refuses the kill everywhere -- the PID was recycled.
    """
    if force and (_IS_WINDOWS or expected_start_time is not None):
        _assert_process_start_time_matches(pid, expected_start_time)
    if force and _IS_WINDOWS:
        # Hide flags: a bare taskkill spawn from windowless pythonw.exe would
        # flash a conhost window on every force-kill.
        from hermes_cli._subprocess_compat import windows_hide_flags

        try:
            result = subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                capture_output=True,
                text=True, encoding='utf-8', errors='replace',
                timeout=10,
                creationflags=windows_hide_flags(),
            )
        except FileNotFoundError:
            os.kill(pid, signal.SIGTERM)
            return

        if result.returncode != 0:
            details = (result.stderr or result.stdout or "").strip()
            raise OSError(details or f"taskkill failed for PID {pid}")
        return

    sig = signal.SIGTERM if not force else getattr(signal, "SIGKILL", signal.SIGTERM)
    os.kill(pid, sig)


def _scope_hash(identity: str) -> str:
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]


def _get_scope_lock_path(scope: str, identity: str) -> Path:
    return _get_lock_dir() / f"{scope}-{_scope_hash(identity)}.lock"


def _get_process_start_time(pid: int) -> Optional[int]:
    """Stable per-process start-time fingerprint (PID-reuse guard), or None.

    Linux: field 22 of ``/proc/<pid>/stat`` (clock ticks since boot). Without
    ``/proc`` (macOS/Windows): psutil ``create_time()`` quantized to
    centiseconds for stable equality. Units differ per platform but the guard
    only ever compares same-host, same-source values.
    """
    stat_path = Path(f"/proc/{pid}/stat")
    try:
        return int(stat_path.read_text(encoding="utf-8").split()[21])
    except (FileNotFoundError, IndexError, PermissionError, ValueError, OSError):
        pass
    try:
        import psutil  # type: ignore
        return int(round(psutil.Process(pid).create_time() * 100))
    except Exception:
        return None


def get_process_start_time(pid: int) -> Optional[int]:
    """Public wrapper for retrieving a process start time when available."""
    return _get_process_start_time(pid)


def _read_process_cmdline(pid: int) -> Optional[str]:
    """Process command line as one string: /proc, then ``ps``, then psutil (Windows)."""
    cmdline_path = Path(f"/proc/{pid}/cmdline")
    try:
        raw = cmdline_path.read_bytes()
    except (FileNotFoundError, PermissionError, OSError):
        pass
    else:
        if raw:
            return raw.replace(b"\x00", b" ").decode("utf-8", errors="ignore").strip()

    if not _IS_WINDOWS:
        try:
            result = subprocess.run(
                ["ps", "-p", str(pid), "-o", "command="],
                capture_output=True,
                text=True, encoding='utf-8', errors='replace',
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            pass

    try:
        import psutil  # type: ignore
        proc = psutil.Process(pid)
        cmdline_parts = proc.cmdline()
        if cmdline_parts:
            return " ".join(cmdline_parts)
    except Exception:
        pass

    return None


def _gateway_command_subcommand(command: str | None) -> str | None:
    """Return the Hermes gateway lifecycle subcommand from a command line.

    Lifecycle decisions must not fire on loose substring matches: ``"gateway"
    in cmdline`` also matched ``gateway status`` and ``python -m tui_gateway``,
    making ``restart()`` race a draining process and ``status`` report false
    positives. Requires a Hermes entrypoint plus the ``gateway`` subcommand (or
    a gateway-dedicated entrypoint). Tokenizes quote-aware so Windows paths with
    spaces survive, and strips ``--profile``/``-p`` selectors from anywhere in
    argv -- ``_apply_profile_override`` removes them before argparse, so they
    (and a profile literally named ``gateway``) can appear on either side.
    """
    if not command:
        return None

    try:
        raw_tokens = shlex.split(command, posix=False)
    except ValueError:
        raw_tokens = command.split()
    # Strip surrounding quotes, normalize slashes + case per token.
    tokens = [t.strip("\"'").replace("\\", "/").lower() for t in raw_tokens]
    if not tokens:
        return None

    # Gateway-dedicated entrypoints carry no subcommand to inspect.
    for token in tokens:
        if token == "gateway/run.py" or token.endswith("/gateway/run.py"):
            return "run"
        if token.rsplit("/", 1)[-1] in ("hermes-gateway", "hermes-gateway.exe"):
            return "run"

    joined = " ".join(tokens)
    has_gateway_entry = (
        "hermes_cli.main" in joined
        or "hermes_cli/main.py" in joined
        or any(t.rsplit("/", 1)[-1] in ("hermes", "hermes.exe") for t in tokens)
    )
    if not has_gateway_entry:
        return None

    # Drop --profile X / -p X / --profile=X / -p=X (consumes a VALUE of "gateway" too).
    filtered: list[str] = []
    skip_next = False
    for token in tokens:
        if skip_next:
            skip_next = False
            continue
        if token in ("--profile", "-p"):
            skip_next = True
            continue
        if token.startswith("--profile=") or token.startswith("-p="):
            continue
        filtered.append(token)

    for i, token in enumerate(filtered):
        if token != "gateway":
            continue
        if i + 1 >= len(filtered):
            return "run"  # bare `hermes gateway` defaults to `run`
        return filtered[i + 1]
    return None


def looks_like_gateway_command_line(command: str | None) -> bool:
    """Return True only for a real ``gateway run`` process command line."""
    return _gateway_command_subcommand(command) == "run"


def looks_like_gateway_runtime_command_line(command: str | None) -> bool:
    """True for command lines that can host the gateway runtime (``run`` or ``restart``).

    Without a service manager the manual restart fallback runs ``run_gateway()``
    in-process, so argv stays ``gateway restart`` while it owns the runtime.
    Use only for validating Hermes-owned records / no-supervisor cleanup scans;
    ``looks_like_gateway_command_line()`` stays strict.
    """
    return _gateway_command_subcommand(command) in {"run", "restart"}


def _looks_like_gateway_process(pid: int) -> bool:
    """Return True when the live PID still looks like the Hermes gateway."""
    cmdline = _read_process_cmdline(pid)
    return bool(cmdline) and looks_like_gateway_command_line(cmdline)


def _record_looks_like_gateway(record: dict[str, Any]) -> bool:
    """Validate gateway identity from PID-file metadata when cmdline is unavailable."""
    if record.get("kind") != _GATEWAY_KIND:
        return False
    argv = record.get("argv")
    if not isinstance(argv, list) or not argv:
        return False
    return looks_like_gateway_runtime_command_line(" ".join(str(part) for part in argv))


def _profile_name_for_home(profile_home: Path) -> Optional[str]:
    """Profile id for ``<root>/profiles/<name>``; None for the root/default home (bare gateway)."""
    if profile_home.parent.name == "profiles":
        return profile_home.name
    return None


def _command_line_belongs_to_profile(command: str, profile_home: Path) -> bool:
    """True when a gateway command line belongs to ``profile_home``.

    Mirrors ``hermes_cli.gateway._matches_current_profile``: a stale state file
    can record a PID the OS recycled onto a DIFFERENT profile's live gateway,
    which still looks like a gateway -- so the dead profile would read running.
    Named profiles carry ``-p``/``--profile <name>`` (or explicit
    ``HERMES_HOME=``) on argv; the default gateway runs bare.
    """
    # Normalize separators: Windows str(Path) uses backslashes while an argv
    # HERMES_HOME= value may use forward slashes (Git Bash, JSON) -- or vice versa.
    command_lc = command.lower().replace("\\", "/")
    profile_name = _profile_name_for_home(profile_home)
    home_lc = str(profile_home).lower().replace("\\", "/")

    if profile_name is not None and profile_name != "default":
        profile_lc = profile_name.lower()
        return (
            f"--profile {profile_lc}" in command_lc
            or f"-p {profile_lc}" in command_lc
            or f"hermes_home={home_lc}" in command_lc
        )

    # Default profile: accept unless argv names some other profile or a
    # conflicting explicit HERMES_HOME= (its absence is not disqualifying --
    # HERMES_HOME usually arrives via the environment).
    if "--profile " in command_lc or " -p " in command_lc:
        return False
    return not ("hermes_home=" in command_lc and f"hermes_home={home_lc}" not in command_lc)


def _record_matches_live_gateway_pid(
    record: dict[str, Any],
    pid: int,
    *,
    expected_home: Optional[Path] = None,
) -> bool:
    """True when a live PID still identifies as this gateway record.

    Prefer the live command line: a stale record's argv must not make an
    unrelated process (PID reuse) count as a gateway. With ``expected_home``
    the live command line must also belong to that profile. When the command
    line is unreadable (Windows/permission), fall back to the persisted record.
    """
    live_cmdline = _read_process_cmdline(pid)
    if live_cmdline:
        if not looks_like_gateway_runtime_command_line(live_cmdline):
            return False
        return not (expected_home is not None and not _command_line_belongs_to_profile(live_cmdline, expected_home))
    return _record_looks_like_gateway(record)


def _build_pid_record() -> dict:
    return {
        "pid": os.getpid(),
        "kind": _GATEWAY_KIND,
        "argv": list(sys.argv),
        "start_time": _get_process_start_time(os.getpid()),
        # Scoped locks are machine-global; the owner's home lets a cross-profile
        # --replace place its takeover marker where the target will read it.
        "hermes_home": str(_canonical_hermes_home(_get_process_hermes_home())),
    }


def _get_code_identity_fields() -> dict[str, Any]:
    """Code identity of THIS process, stamped into ``gateway_state.json``.

    Lets ``hermes update``/the dashboard prove a restarted gateway picked up new
    code. Lazy import keeps ``gateway.status`` free of ``hermes_cli`` at import
    time. Never raises; degrades to absent fields.
    """
    try:
        from hermes_cli.build_info import get_code_identity

        identity = get_code_identity()
        return {
            "code_sha": identity.get("sha"),
            "code_version": identity.get("version"),
        }
    except Exception:
        return {}


def _pid_record_belongs_to_current_profile(
    record: Optional[dict[str, Any]],
) -> bool:
    """True when the record's ``hermes_home`` matches the current process.

    A record written under a different HERMES_HOME belongs to another profile
    and must be ignored, or the default gateway assumes that profile's identity.
    Legacy records without the field are accepted conservatively.
    """
    if not isinstance(record, dict):
        return False
    record_home = record.get("hermes_home")
    if not record_home:
        return True
    return _same_hermes_home(record_home, _get_process_hermes_home())


def _build_runtime_status_record() -> dict[str, Any]:
    payload = _build_pid_record()
    payload.update({
        "gateway_state": "starting",
        "exit_reason": None,
        "restart_requested": False,
        "active_agents": 0,
        "platforms": {},
        "session_store": {"status": "unknown"},
        "updated_at": _utc_now_iso(),
    })
    payload.update(_get_code_identity_fields())
    return payload


def _read_text_file(path: Path) -> Optional[str]:
    """Stripped file text, or None when absent/empty/unreadable.

    OSError: vanished or permission flipped between exists() and read.
    UnicodeDecodeError: non-UTF-8 / binary garbage (truncated or clobbered file).
    """
    if not path.exists():
        return None
    try:
        raw = path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError):
        return None
    return raw or None


def _read_json_file(path: Path) -> Optional[dict[str, Any]]:
    raw = _read_text_file(path)
    if raw is None:
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _write_json_file(path: Path, payload: dict[str, Any]) -> None:
    atomic_json_write(path, payload, indent=None, separators=(",", ":"))


def _unlink_quietly(path: Path) -> None:
    with contextlib.suppress(OSError):
        path.unlink(missing_ok=True)


def _read_pid_record(pid_path: Optional[Path] = None) -> Optional[dict]:
    """PID record as a dict; legacy bare-integer files become ``{"pid": N}``."""
    raw = _read_text_file(pid_path or _get_pid_path())
    if raw is None:
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        try:
            return {"pid": int(raw)}
        except ValueError:
            return None
    if isinstance(payload, int):
        return {"pid": payload}
    if isinstance(payload, dict):
        return payload
    return None


def _read_gateway_lock_record(lock_path: Optional[Path] = None) -> Optional[dict[str, Any]]:
    return _read_pid_record(lock_path or _get_gateway_lock_path())


def _pid_from_record(record: Optional[dict[str, Any]]) -> Optional[int]:
    if not record:
        return None
    try:
        return int(record["pid"])
    except (KeyError, TypeError, ValueError):
        return None


def _start_time_conflicts(record: dict[str, Any], pid: int) -> bool:
    """PID-reuse guard: True only when BOTH start times are known and differ."""
    recorded_start = record.get("start_time")
    current_start = _get_process_start_time(pid)
    return recorded_start is not None and current_start is not None and current_start != recorded_start


def _clear_running_pid_cache() -> None:
    with _gateway_running_pid_cache_lock:
        _gateway_running_pid_cache.clear()


def _file_cache_signature(path: Path) -> tuple[bool, Optional[int], Optional[int]]:
    try:
        st = path.stat()
    except OSError:
        return (False, None, None)
    return (True, st.st_mtime_ns, st.st_size)


def _running_pid_cache_signature(
    pid_path: Path,
    *,
    include_runtime_status: bool,
) -> tuple[Any, ...]:
    parts: list[Any] = [
        _file_cache_signature(pid_path),
        _file_cache_signature(_get_gateway_lock_path(pid_path)),
    ]
    if include_runtime_status:
        parts.append(_file_cache_signature(_get_runtime_status_path()))
    return tuple(parts)


def _cleanup_invalid_pid_path(pid_path: Path, *, cleanup_stale: bool) -> None:
    """Force-unlink a stale PID file and its sibling lock file.

    Called only after the runtime lock is confirmed inactive (dead owner), so
    unlike ``remove_pid_file()`` it does not check the recorded pid.
    """
    if not cleanup_stale:
        return
    _clear_running_pid_cache()
    with contextlib.suppress(Exception):
        pid_path.unlink(missing_ok=True)
    with contextlib.suppress(Exception):
        _get_gateway_lock_path(pid_path).unlink(missing_ok=True)


def _write_gateway_lock_record(handle) -> None:
    handle.seek(0)
    handle.truncate()
    json.dump(_build_pid_record(), handle)
    handle.flush()
    with contextlib.suppress(OSError):
        os.fsync(handle.fileno())


def _try_acquire_file_lock(handle) -> bool:
    try:
        if _IS_WINDOWS:
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write("\n")
                handle.flush()
            handle.seek(_WINDOWS_LOCK_OFFSET)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except (BlockingIOError, OSError):
        return False


def _pid_exists(pid: int) -> bool:
    """Cross-platform "is this PID alive" check that does NOT kill the target.

    CRITICAL on Windows: ``os.kill(pid, 0)`` is NOT a no-op -- CPython maps
    ``sig=0`` to ``CTRL_C_EVENT`` and ``GenerateConsoleCtrlEvent`` sends Ctrl+C
    to the target's whole console group (bpo-14484). Prefer psutil; fall back
    to ctypes ``OpenProcess``/``WaitForSingleObject`` on Windows and
    ``os.kill(pid, 0)`` on POSIX when psutil is unavailable (stripped install /
    scaffold phase).
    """
    try:
        import psutil  # type: ignore

        # Zombies are still in the process table (pid_exists() -> True) but are
        # dead: treating one as alive makes --replace wait forever under systemd
        # Restart=always, which respawns before reaping. Report them as dead.
        # Best-effort: status-read failures fall through to pid_exists().
        try:
            if psutil.Process(int(pid)).status() == psutil.STATUS_ZOMBIE:
                return False
        except getattr(psutil, "NoSuchProcess", ()):
            return False
        except Exception:
            pass
        return bool(psutil.pid_exists(int(pid)))

    except ImportError:
        pass  # Fall through to stdlib fallback.
    if _IS_WINDOWS:
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            # Pin restypes: default c_int mangles WAIT_* DWORDs into negatives.
            kernel32.OpenProcess.restype = ctypes.c_void_p
            kernel32.WaitForSingleObject.restype = ctypes.c_uint
            kernel32.GetLastError.restype = ctypes.c_uint
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            SYNCHRONIZE = 0x100000  # required for WaitForSingleObject
            WAIT_TIMEOUT = 0x00000102
            ERROR_INVALID_PARAMETER = 87
            ERROR_ACCESS_DENIED = 5
            handle = kernel32.OpenProcess(
                PROCESS_QUERY_LIMITED_INFORMATION | SYNCHRONIZE, False, int(pid)
            )
            if not handle:
                err = kernel32.GetLastError()
                if err == ERROR_INVALID_PARAMETER:
                    return False  # PID definitely gone
                if err == ERROR_ACCESS_DENIED:
                    return True   # Exists but owned by another user/session
                return False      # Conservative default for unknown errors
            try:
                # WAIT_TIMEOUT = still running; anything else = gone.
                return kernel32.WaitForSingleObject(handle, 0) == WAIT_TIMEOUT
            finally:
                kernel32.CloseHandle(handle)
        except (OSError, AttributeError):
            return False
    else:
        # Same zombie case as the psutil path: a zombie answers os.kill(pid, 0).
        try:
            stat_fields = (
                Path(f"/proc/{int(pid)}/stat").read_text(encoding="utf-8").split()
            )
            if len(stat_fields) > 2 and stat_fields[2] == "Z":
                return False
        except FileNotFoundError:  # No /proc (macOS/BSD): use ps state.
            try:
                r = subprocess.run(
                    ["ps", "-o", "state=", "-p", str(int(pid))],
                    capture_output=True,
                    text=True, encoding='utf-8', errors='replace',
                    timeout=5,
                )
                if r.returncode == 0 and r.stdout.strip().startswith("Z"):
                    return False
            except Exception:
                pass
        except (IndexError, PermissionError, OSError):
            pass
        try:
            os.kill(int(pid), 0)  # windows-footgun: ok — POSIX-only branch (the whole point of _pid_exists)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True  # Exists but we can't signal it.
        except OSError:
            return False


def _lock_is_held(handle) -> bool:
    """Probe: True when another process holds the lock (a won probe is released)."""
    if _try_acquire_file_lock(handle):
        _release_file_lock(handle)
        return False
    return True


def _release_file_lock(handle) -> None:
    try:
        if _IS_WINDOWS:
            handle.seek(_WINDOWS_LOCK_OFFSET)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except OSError:
        pass


def acquire_gateway_runtime_lock() -> bool:
    """Claim the cross-process runtime lock; the OS releases it if the process dies."""
    global _gateway_lock_handle
    if _gateway_lock_handle is not None:
        return True

    path = _get_gateway_lock_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        handle = open(path, "a+", encoding="utf-8")
    except PermissionError:
        # Stale root-owned lock from a launchd Background session that ran as
        # root. The directory owner can unlink it; retry once with a fresh file.
        try:
            path.unlink()
        except OSError:
            return False
        try:
            handle = open(path, "a+", encoding="utf-8")
        except OSError:
            return False
    if not _try_acquire_file_lock(handle):
        handle.close()
        return False
    _write_gateway_lock_record(handle)
    _gateway_lock_handle = handle
    _clear_running_pid_cache()
    return True


def release_gateway_runtime_lock() -> None:
    """Release the gateway runtime lock when owned by this process."""
    global _gateway_lock_handle
    handle = _gateway_lock_handle
    if handle is None:
        return
    _gateway_lock_handle = None
    _release_file_lock(handle)
    with contextlib.suppress(OSError):
        handle.close()
    _clear_running_pid_cache()


def owns_gateway_runtime_lock() -> bool:
    """True when THIS process holds the runtime lock.

    ``is_gateway_runtime_lock_active`` answers "does anyone hold it?" and a
    file re-probe cannot tell self-ownership apart (a probe on our own flock
    succeeds on POSIX); the in-process handle is the only discriminator.
    """
    return _gateway_lock_handle is not None


def is_gateway_runtime_lock_active(lock_path: Optional[Path] = None) -> bool:
    """Return True when some process currently owns the gateway runtime lock."""
    global _gateway_lock_handle
    resolved_lock_path = lock_path or _get_gateway_lock_path()
    if _gateway_lock_handle is not None and resolved_lock_path == _get_gateway_lock_path():
        return True

    if not resolved_lock_path.exists():
        return False

    try:
        handle = open(resolved_lock_path, "a+", encoding="utf-8")
    except PermissionError:
        # Stale root-owned lock (launchd session that ran as root): the
        # directory owner can unlink it; report inactive so a fresh one is made.
        with contextlib.suppress(OSError):
            resolved_lock_path.unlink()
        return False
    try:
        return _lock_is_held(handle)
    finally:
        with contextlib.suppress(OSError):
            handle.close()


def _strict_path_exists(path: Path, label: str) -> bool:
    try:
        path.stat()
        return True
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise RuntimeError(f"{label} metadata is not inspectable: {exc}") from exc


def _is_gateway_runtime_lock_active_strict(lock_path: Path) -> bool:
    """Probe ownership without treating access failures as absence."""
    try:
        handle = open(lock_path, "r+", encoding="utf-8")
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise RuntimeError(f"gateway runtime lock is not inspectable: {exc}") from exc
    try:
        return _lock_is_held(handle)
    except OSError as exc:
        raise RuntimeError(f"gateway runtime lock probe failed: {exc}") from exc
    finally:
        with contextlib.suppress(OSError):
            handle.close()


def write_pid_file() -> None:
    """Write this process's PID record via O_CREAT|O_EXCL (concurrent racers get FileExistsError)."""
    path = _get_pid_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    record = json.dumps(_build_pid_record())
    # FileExistsError propagates: another gateway is racing us; caller decides.
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(record)
        _clear_running_pid_cache()
    except Exception:
        _unlink_quietly(path)
        raise


def write_runtime_status(
    *,
    gateway_state: Any = _UNSET,
    exit_reason: Any = _UNSET,
    restart_requested: Any = _UNSET,
    active_agents: Any = _UNSET,
    platform: Any = _UNSET,
    platform_state: Any = _UNSET,
    error_code: Any = _UNSET,
    error_message: Any = _UNSET,
    needs_attention: Any = _UNSET,
    retrying_since: Any = _UNSET,
    served_profiles: Any = _UNSET,
    session_store: Any = _UNSET,
    clear_profile_platforms: bool = False,
) -> None:
    """Persist gateway runtime health information for diagnostics/status."""
    path = _get_runtime_status_path()
    payload = _read_json_file(path) or _build_runtime_status_record()
    previous_payload = copy.deepcopy(payload)
    current_record = _build_pid_record()
    payload.setdefault("platforms", {})
    if clear_profile_platforms:
        # Secondary-profile entries are keyed ``<profile>:<platform>``. A fresh
        # process must not inherit them or /api/status stays degraded until
        # every old adapter re-emits (removed profiles: forever).
        platforms = payload["platforms"]
        if not isinstance(platforms, dict):
            platforms = {}
        payload["platforms"] = {
            key: value
            for key, value in platforms.items()
            if not isinstance(key, str) or ":" not in key
        }
    for key in ("kind", "pid", "argv", "start_time"):
        payload[key] = current_record[key]
    payload["updated_at"] = _utc_now_iso()
    # Re-stamp on every write: the file can outlive its creator and the top-level
    # record must describe the CURRENT writer's code.
    payload.update(_get_code_identity_fields())

    for key, value in (("gateway_state", gateway_state), ("exit_reason", exit_reason)):
        if value is not _UNSET:
            payload[key] = value
    if restart_requested is not _UNSET:
        payload["restart_requested"] = bool(restart_requested)
    if active_agents is not _UNSET:
        payload["active_agents"] = parse_active_agents(active_agents)
    if served_profiles is not _UNSET:
        # Multiplexed profiles; absent/empty for a single-profile gateway.
        payload["served_profiles"] = list(served_profiles or [])
    if session_store is not _UNSET:
        state = "unknown"
        if isinstance(session_store, dict):
            candidate = str(session_store.get("status") or "unknown")
            if candidate in {"ok", "unavailable", "retrying", "unknown"}:
                state = candidate
        payload["session_store"] = {"status": state}

    if platform is not _UNSET:
        platform_payload = payload["platforms"].get(platform, {})
        for key, value in (
            ("state", platform_state),
            ("error_code", error_code),
            ("error_message", error_message),
        ):
            if value is not _UNSET:
                platform_payload[key] = value
        if needs_attention is not _UNSET:
            # Reconnect-loop escalation past the attention threshold: a signal
            # for owners/fleet monitoring, not a circuit breaker (retry never
            # stops). Cleared on successful reconnect.
            platform_payload["needs_attention"] = bool(needs_attention)
        if retrying_since is not _UNSET:
            # ISO start of the current retry episode; None clears it.
            platform_payload["retrying_since"] = retrying_since
        platform_payload["updated_at"] = _utc_now_iso()
        # Per-entry writer provenance: top-level pid/start_time only identify
        # the most recent writer, so /api/status distinguishes "written by the
        # live process" from "preserved from a prior one" by exact
        # (pid, start_time) equality rather than clock heuristics.
        platform_payload["writer_pid"] = current_record["pid"]
        platform_payload["writer_start_time"] = current_record["start_time"]
        payload["platforms"][platform] = platform_payload

    _write_json_file(path, payload)
    try:
        from agent.monitoring.gateway_health import emit_runtime_status_transition
        emit_runtime_status_transition(previous_payload, payload)
    except Exception:
        pass


def read_runtime_status(path: Optional[Path] = None) -> Optional[dict[str, Any]]:
    """Read ``gateway_state.json``; ``path`` lets callers inspect another profile's file."""
    return _read_json_file(path or _get_runtime_status_path())


# Max age of a ``gateway_state.json`` snapshot before its liveness claim is
# suspect: an older record with a dead PID outlived an ungracefully-killed
# writer (taskkill /F, OOM, power loss) that never ran its shutdown handler.
_RUNTIME_STATUS_STALE_TTL_S = 120


def runtime_status_is_stale(
    record: Optional[dict[str, Any]],
    ttl_s: int = _RUNTIME_STATUS_STALE_TTL_S,
) -> bool:
    """True when the snapshot's ``updated_at`` is older than ``ttl_s`` (missing/unparseable => stale)."""
    if not isinstance(record, dict):
        return True
    return _marker_is_stale(record.get("updated_at") or "", ttl_s)


def runtime_status_pid_is_live(record: Optional[dict[str, Any]]) -> bool:
    """True when the snapshot's PID is alive and passes the start-time PID-reuse guard."""
    pid = _pid_from_record(record)
    if pid is None or not _pid_exists(pid):
        return False
    return not _start_time_conflicts(record, pid)


def parse_active_agents(raw: Any) -> int:
    """Coerce ``active_agents`` to a non-negative int (shared by the writer and both HTTP readers)."""
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return 0


# Only a live ``running`` gateway is a valid begin-drain target.
_DRAINABLE_GATEWAY_STATES = frozenset({"running"})


def derive_gateway_busy(
    *, gateway_running: bool, gateway_state: Any, active_agents: Any
) -> bool:
    """Busy iff live, in ``running`` state, and ``active_agents > 0`` (the contract NAS gates on).

    Degrades to False on unknown liveness / other state / unparseable count.
    Liveness keys off ``gateway_running``, NEVER ``updated_at`` -- a healthy
    idle gateway never advances that timestamp.
    """
    if not gateway_running or gateway_state not in _DRAINABLE_GATEWAY_STATES:
        return False
    try:
        return int(active_agents) > 0
    except (TypeError, ValueError):
        return False


def derive_gateway_drainable(*, gateway_running: bool, gateway_state: Any) -> bool:
    """Drainable iff live and ``running`` (independent of ``active_agents``; an idle drain completes at once)."""
    return bool(gateway_running) and gateway_state in _DRAINABLE_GATEWAY_STATES


@dataclass(frozen=True)
class GatewayLiveness:
    """Resolved gateway liveness for one dashboard surface.

    ``source``: which ladder rung answered (logging/tests only -- never branch
    product behavior on it). ``probe_error``: a rung raised instead of
    answering; lets fail-open callers (kanban dispatcher warning) tell "down"
    from "could not tell".
    """

    running: bool
    pid: Optional[int]
    source: str
    health_body: Optional[dict[str, Any]] = None
    probe_error: bool = False


def resolve_gateway_liveness(
    *,
    profile_dir: Optional[Path] = None,
    runtime: Any = _UNSET,
    health_probe: Optional[Callable[[], tuple[bool, Optional[dict[str, Any]]]]] = None,
    use_cache: bool = True,
    pid_probe: Optional[Callable[..., Optional[int]]] = None,
    runtime_reader: Optional[Callable[..., Optional[dict[str, Any]]]] = None,
    runtime_pid_probe: Optional[Callable[..., Optional[int]]] = None,
) -> GatewayLiveness:
    """Single source of truth for "is the gateway up?" across dashboard surfaces.

    Ladder, most to least authoritative:
    1. PID file + runtime lock (scoped to ``profile_dir``; cached by default so
       high-frequency polling does not re-flock ``gateway.lock`` per request).
    2. Caller-supplied HTTP health probe (covers a gateway in another container).
    3. Runtime status PID, validated against the live process table with
       ``expected_home`` so a recycled PID of a different profile never counts.
    Rung 3 only runs against the LOCAL state record -- the probe body's PID
    belongs to another host. Pass ``runtime`` if the state file is already read.
    ``pid_probe``/``runtime_reader``/``runtime_pid_probe`` let the dashboard
    inject its ``hermes_cli.web_server`` bindings (test monkeypatch seam).
    """
    _pid_probe = pid_probe or (
        get_running_pid_cached if use_cache else get_running_pid
    )
    _runtime_reader = runtime_reader or read_runtime_status
    _runtime_pid_probe = runtime_pid_probe or get_runtime_status_running_pid

    pid_path = (profile_dir / "gateway.pid") if profile_dir is not None else None
    probe_error = False
    try:
        # Zero-arg call when unscoped: callers monkeypatch with zero-arg lambdas
        # and /api/status's cache signature is keyed on the call shape.
        pid = _pid_probe(pid_path) if pid_path is not None else _pid_probe()
    except Exception:
        # Degrade to the next rung; never 500 a status endpoint.
        pid = None
        probe_error = True
    if pid is not None:
        return GatewayLiveness(running=True, pid=pid, source="pid")

    health_body: Optional[dict[str, Any]] = None
    if health_probe is not None:
        try:
            alive, health_body = health_probe()
        except Exception:
            alive, health_body = False, None
            probe_error = True
        if alive:
            # Display-only PID: it belongs to the remote container.
            remote_pid = health_body.get("pid") if health_body else None
            return GatewayLiveness(
                running=True,
                pid=remote_pid,
                source="health",
                health_body=health_body,
            )

    if runtime is _UNSET:
        try:
            runtime = (
                _runtime_reader(path=profile_dir / "gateway_state.json")
                if profile_dir is not None
                else _runtime_reader()
            )
        except Exception:
            runtime = None
            probe_error = True
    try:
        runtime_pid = (
            _runtime_pid_probe(runtime, expected_home=profile_dir)
            if profile_dir is not None
            else _runtime_pid_probe(runtime)
        )
    except Exception:
        runtime_pid = None
        probe_error = True
    if runtime_pid is not None:
        return GatewayLiveness(
            running=True,
            pid=runtime_pid,
            source="runtime_status",
            health_body=health_body,
        )

    return GatewayLiveness(
        running=False,
        pid=None,
        source="none",
        health_body=health_body,
        probe_error=probe_error,
    )


def get_runtime_status_running_pid(
    runtime: Optional[dict[str, Any]] = None,
    *,
    expected_home: Optional[Path] = None,
) -> Optional[int]:
    """Live gateway PID from the runtime status record, or None.

    Conservative fallback to ``get_running_pid()`` for launch-service-managed
    gateways that have a fresh ``gateway_state.json`` but no ``gateway.pid``.
    ``expected_home`` scopes the OS-identity check to another profile's home
    (dashboard enumeration) so a PID recycled onto a different profile's
    gateway is not reported running for the dead one.
    """
    payload = runtime if runtime is not None else read_runtime_status()
    if not isinstance(payload, dict):
        return None
    if payload.get("gateway_state") in {None, "stopped", "startup_failed"}:
        return None

    pid = _pid_from_record(payload)
    if pid is None or not _pid_exists(pid) or _start_time_conflicts(payload, pid):
        return None

    # Active-profile context: the record's hermes_home must match this process
    # so a stale record cannot lend another profile's identity.
    if expected_home is None and not _pid_record_belongs_to_current_profile(payload):
        return None

    if _record_matches_live_gateway_pid(payload, pid, expected_home=expected_home):
        return pid
    return None


def remove_pid_file() -> None:
    """Remove the PID file only if it belongs to this process.

    During --replace the old process's atexit can fire AFTER the new process
    wrote its own record; blind removal would leave the gateway invisible.
    """
    try:
        path = _get_pid_path()
        record = _read_json_file(path)
        if record is not None:
            file_pid = _pid_from_record(record)
            if file_pid is not None and file_pid != os.getpid():
                return  # Belongs to a different process — leave it alone.
        path.unlink(missing_ok=True)
        _clear_running_pid_cache()
    except Exception:
        pass


def acquire_scoped_lock(scope: str, identity: str, metadata: Optional[dict[str, Any]] = None) -> tuple[bool, Optional[dict[str, Any]]]:
    """Acquire a machine-local lock keyed by scope + identity (e.g. one Telegram token across homes)."""
    lock_path = _get_scope_lock_path(scope, identity)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        **_build_pid_record(),
        "scope": scope,
        "identity_hash": _scope_hash(identity),
        "metadata": metadata or {},
        "updated_at": _utc_now_iso(),
    }
    # Profile label for cross-profile conflict diagnostics ("token already in
    # use (PID 559)" alone does not say WHICH profile). Omitted when not
    # inferable; readers fall back to hermes_home.
    profile = _profile_label_for_home(_get_process_hermes_home())
    if profile:
        record["profile"] = profile

    existing = _read_json_file(lock_path)
    if existing is None and lock_path.exists():
        # Empty/invalid JSON: previous process died between O_EXCL create and
        # json.dump(). Treat as stale.
        _unlink_quietly(lock_path)
    if existing:
        existing_pid = _pid_from_record(existing)

        # Our own PID: always self-reacquire. start_time guards reuse of OTHER
        # PIDs; requiring equality here rejects reconnects when the on-disk
        # record has start_time null (older writers / psutil failure).
        if existing_pid == os.getpid():
            _write_json_file(lock_path, record)
            return True, existing

        stale = existing_pid is None
        if not stale:
            if not _pid_exists(existing_pid):
                stale = True
            else:
                current_start = _get_process_start_time(existing_pid)
                if (
                    existing.get("start_time") is not None
                    and current_start is not None
                    and current_start != existing.get("start_time")
                ):
                    stale = True
                # Live process is not a gateway: stale when its cmdline is
                # readable and says so (also catches boot-time PID+start_time
                # collisions -- systemd spawns deterministically, so an unrelated
                # service can land on the same PID and jiffy count). When the
                # cmdline is unreadable (Windows has no ps) AND start_time is
                # unavailable on either side (no /proc, psutil failure), consult
                # the lock record's own argv -- the only identity signal there.
                if (
                    not stale
                    and not _looks_like_gateway_process(existing_pid)
                    and (
                        _read_process_cmdline(existing_pid) is not None
                        or (
                            (existing.get("start_time") is None or current_start is None)
                            and not _record_looks_like_gateway(existing)
                        )
                    )
                ):
                    stale = True
                # Stopped processes (Ctrl+Z / SIGTSTP) look alive to _pid_exists
                # but are not running; treat as stale so --replace works.
                if not stale:
                    try:
                        _proc_status = Path(f"/proc/{existing_pid}/status")
                        if _proc_status.exists():
                            for _line in _proc_status.read_text(encoding="utf-8").splitlines():
                                if _line.startswith("State:"):
                                    if _line.split()[1] in {"T", "t"}:  # stopped / tracing stop
                                        stale = True
                                    break
                    except (OSError, PermissionError):
                        pass
        if stale:
            # Rename to a tombstone instead of unlink(): with unlink()+O_EXCL two
            # racing starters could both win (the second unlink deleting the
            # first racer's fresh lock). os.replace() lets exactly one claim it.
            tombstone = lock_path.with_name(lock_path.name + ".stale")
            try:
                os.replace(lock_path, tombstone)
            except OSError:
                pass  # Another racer claimed it -- O_EXCL below decides the winner.
            else:
                _unlink_quietly(tombstone)
        else:
            return False, existing

    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False, _read_json_file(lock_path)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(record, handle)
    except Exception:
        _unlink_quietly(lock_path)
        raise
    return True, None


def release_scoped_lock(scope: str, identity: str) -> None:
    """Release a previously-acquired scope lock when owned by this process."""
    lock_path = _get_scope_lock_path(scope, identity)
    existing = _read_json_file(lock_path)
    if not existing:
        return
    if existing.get("pid") != os.getpid():
        return
    # Own PID => we own the lock. No start_time equality: on-disk null vs a live
    # fingerprint would leave the lock stuck across reconnects.
    _unlink_quietly(lock_path)


def release_all_scoped_locks(
    *,
    owner_pid: Optional[int] = None,
    owner_start_time: Optional[int] = None,
) -> int:
    """Remove scoped lock files (--replace cleanup); returns the count removed.

    With ``owner_pid`` only that gateway's records go (``owner_start_time``
    narrows against PID reuse); with no owner every lock file is removed.
    """
    lock_dir = _get_lock_dir()
    removed = 0
    if lock_dir.exists():
        for lock_file in lock_dir.glob("*.lock"):
            if owner_pid is not None:
                record = _read_json_file(lock_file)
                if not isinstance(record, dict):
                    continue
                if _pid_from_record(record) != owner_pid:
                    continue
                if (
                    owner_start_time is not None
                    and record.get("start_time") != owner_start_time
                ):
                    continue
            try:
                lock_file.unlink(missing_ok=True)
                removed += 1
            except OSError:
                pass
    return removed


# ── --replace takeover marker ─────────────────────────────────────────
#
# SIGTERM exits the gateway with code 1 so Restart=on-failure revives it after
# unexpected kills -- which would also revive a --replace target and start a
# flap loop against the replacer. The replacer therefore writes a short-lived
# marker naming the target PID + start_time BEFORE SIGTERM; the target's
# shutdown handler treats a matching marker as a planned takeover and exits 0.
# The marker is unlinked once consumed, so a stale one can grief at most one
# future shutdown on the same PID, within _TAKEOVER_MARKER_TTL_S.

_TAKEOVER_MARKER_FILENAME = ".gateway-takeover.json"
_TAKEOVER_MARKER_TTL_S = 60  # Marker older than this is treated as stale
_PLANNED_STOP_MARKER_FILENAME = ".gateway-planned-stop.json"
_PLANNED_STOP_MARKER_TTL_S = 60


def _get_takeover_marker_path(hermes_home: Optional[Path] = None) -> Path:
    """Takeover marker path; ``hermes_home`` is given only for a verified cross-home handoff."""
    home = hermes_home or _get_process_hermes_home()
    return _canonical_hermes_home(home) / _TAKEOVER_MARKER_FILENAME


def _get_planned_stop_marker_path() -> Path:
    """Return the path to the intentional gateway stop marker file."""
    return _get_process_hermes_home() / _PLANNED_STOP_MARKER_FILENAME


def _marker_is_stale(written_at: str, ttl_s: int) -> bool:
    try:
        return (datetime.now(timezone.utc) - datetime.fromisoformat(written_at)).total_seconds() > ttl_s
    except (TypeError, ValueError):
        return True


def _read_live_pid_marker(path: Path, ttl_s: int) -> Optional[tuple[dict[str, Any], int, Any]]:
    """Return ``(record, target_pid, target_start_time)`` for a usable marker.

    Malformed or expired markers can never match anyone, so they are unlinked
    here (a stale file left by a previous instance must not wedge a new one).
    """
    record = _read_json_file(path)
    if not record:
        return None
    try:
        target_pid = int(record["target_pid"])
        target_start_time = record.get("target_start_time")
        written_at = record.get("written_at") or ""
    except (KeyError, TypeError, ValueError):
        _unlink_quietly(path)
        return None
    if _marker_is_stale(written_at, ttl_s):
        _unlink_quietly(path)
        return None
    return record, target_pid, target_start_time


def _pid_marker_names_self(target_pid: int, target_start_time: Any) -> bool:
    """PID match with an optional start-time PID-reuse guard.

    ``_get_process_start_time`` returns None on platforms without /proc (macOS,
    native Windows -- where the planned-stop watcher matters most). Requiring a
    non-None match there would make every consume return False and misclassify
    a legitimate ``hermes gateway stop`` as an unexpected exit (revived by the
    service manager). So: when both start times are known they must match; when
    either is unknown, PID equality alone decides (bounded by the marker TTL).
    Shared by the watcher's non-destructive probe and the authoritative consume
    so they agree on every platform.
    """
    if target_pid != os.getpid():
        return False
    our_start_time = _get_process_start_time(target_pid)
    if target_start_time is not None and our_start_time is not None:
        return target_start_time == our_start_time
    return True


def _consume_pid_marker_for_self(path: Path, *, ttl_s: int) -> bool:
    parsed = _read_live_pid_marker(path, ttl_s)
    if parsed is None:
        return False
    record, target_pid, target_start_time = parsed

    # Cross-profile guard: new markers name the verified TARGET home, which
    # permits a deliberate cross-HERMES_HOME --replace while ignoring a marker
    # accidentally written into another profile's directory.  Legacy markers
    # have no target field, so keep the original same-replacer-home rule.
    our_home = _get_process_hermes_home()
    target_home = record.get("target_hermes_home")
    if target_home is not None:
        if not isinstance(target_home, str) or not _same_hermes_home(
            target_home, our_home
        ):
            return False
    else:
        replacer_home = record.get("replacer_hermes_home")
        if replacer_home is not None and not _same_hermes_home(
            replacer_home, our_home
        ):
            return False

    matches = _pid_marker_names_self(target_pid, target_start_time)
    _unlink_quietly(path)
    return matches


def write_takeover_marker(
    target_pid: int,
    *,
    target_home: Optional[Path] = None,
    target_start_time: Any = _UNSET,
) -> bool:
    """Record that ``target_pid`` is being replaced by this process; True on success.

    Captures the target's ``start_time`` (PID-reuse guard) and a timestamp for
    TTL checks. A verified cross-home handoff passes ``target_home`` and the
    validated ``target_start_time`` so the marker lands in the target's home;
    such callers must fail closed on False (the target's supervisor could
    otherwise revive it).
    """
    try:
        marker_home = _canonical_hermes_home(
            target_home or _get_process_hermes_home()
        )
        if target_start_time is _UNSET:
            target_start_time = _get_process_start_time(target_pid)
        record = {
            "target_pid": target_pid,
            "target_start_time": target_start_time,
            "target_hermes_home": str(marker_home),
            "replacer_pid": os.getpid(),
            "replacer_hermes_home": str(
                _canonical_hermes_home(_get_process_hermes_home())
            ),
            "written_at": _utc_now_iso(),
        }
        _write_json_file(_get_takeover_marker_path(marker_home), record)
        return True
    except (OSError, PermissionError):
        return False


def consume_takeover_marker_for_self() -> bool:
    """Consume the takeover marker; True means this SIGTERM is a planned takeover (exit 0).

    Always unlinks on match or staleness so later unrelated signals don't re-trigger.
    """
    return _consume_pid_marker_for_self(_get_takeover_marker_path(), ttl_s=_TAKEOVER_MARKER_TTL_S)


def clear_takeover_marker(target_home: Optional[Path] = None) -> None:
    """Remove the takeover marker unconditionally. Safe to call repeatedly."""
    _unlink_quietly(_get_takeover_marker_path(target_home))


def _validated_scoped_lock_gateway_owner(
    record: dict[str, Any],
) -> Optional[tuple[int, int, Path]]:
    """Resolve a live scoped-lock owner to a verified ``(pid, start_time, home)``.

    A lock file is only a claim: the record, the target home's PID record, and
    the live process must agree on PID, start-time, gateway identity, and home.
    Missing legacy metadata fails closed (normal retryable conflict path).
    """
    if not isinstance(record, dict) or not _record_looks_like_gateway(record):
        return None

    owner_pid = _pid_from_record(record)
    if owner_pid is None or owner_pid <= 0 or owner_pid == os.getpid():
        return None

    owner_start_time = record.get("start_time")
    if not isinstance(owner_start_time, int) or isinstance(owner_start_time, bool):
        return None

    raw_home = record.get("hermes_home")
    if not isinstance(raw_home, str) or not raw_home.strip():
        return None
    if not Path(raw_home).expanduser().is_absolute():
        return None
    target_home = _canonical_hermes_home(raw_home)

    if not _pid_exists(owner_pid):
        return None
    live_start_time = _get_process_start_time(owner_pid)
    if live_start_time is None or live_start_time != owner_start_time:
        return None

    live_cmdline = _read_process_cmdline(owner_pid)
    if live_cmdline is not None and not looks_like_gateway_runtime_command_line(
        live_cmdline
    ):
        return None

    pid_record = _read_json_file(target_home / "gateway.pid")
    if not isinstance(pid_record, dict) or not _record_looks_like_gateway(pid_record):
        return None
    if _pid_from_record(pid_record) != owner_pid or pid_record.get("start_time") != owner_start_time:
        return None

    pid_record_home = pid_record.get("hermes_home")
    if not isinstance(pid_record_home, str) or not _same_hermes_home(
        pid_record_home, target_home
    ):
        return None

    return owner_pid, owner_start_time, target_home


def _scoped_lock_owner_state(owner_pid: int, owner_start_time: int) -> str:
    """Return ``same``, ``exited``, or ``unknown`` for a validated owner."""
    if not _pid_exists(owner_pid):
        return "exited"
    live_start_time = _get_process_start_time(owner_pid)
    if live_start_time is None:
        return "unknown"
    if live_start_time != owner_start_time:
        return "exited"  # PID recycled; never signal the replacement process.
    return "same"


def _wait_for_scoped_lock_owner_exit(
    owner_pid: int,
    owner_start_time: int,
    *,
    attempts: int,
    delay: float,
) -> tuple[bool, bool]:
    """Return ``(exited, safe_to_force)`` after bounded identity-aware waits."""
    for _ in range(max(0, attempts)):
        state = _scoped_lock_owner_state(owner_pid, owner_start_time)
        if state == "exited":
            return True, False
        if state == "unknown":
            return False, False
        time.sleep(max(0.0, delay))
    return False, _scoped_lock_owner_state(owner_pid, owner_start_time) == "same"


def _snapshot_gateway_children(pid: int) -> list:
    """Best-effort snapshot of ``pid``'s live descendants (POSIX only; never raises).

    Take it while the parent is alive -- once it exits the children are
    reparented and undiscoverable. ``[]`` on Windows (taskkill /T tree-kills).
    """
    if _IS_WINDOWS:
        return []
    try:
        import psutil  # type: ignore

        return psutil.Process(int(pid)).children(recursive=True)
    except Exception:
        logger.debug(
            "Could not snapshot children of gateway PID %d", pid, exc_info=True
        )
        return []


def reap_gateway_children(children: list, *, parent_pid: int, timeout: float = 5.0) -> int:
    """Best-effort reap of a dead gateway's orphaned descendants (POSIX); returns count signalled.

    Surviving adapter subprocesses keep holding token locks. Call only AFTER the
    parent is confirmed dead with a snapshot from :func:`_snapshot_gateway_children`.
    ``is_running()`` is identity-aware so a recycled child PID is never
    signalled; a child whose ppid still equals ``parent_pid`` is skipped (parent
    alive => not an orphan). SIGTERM, bounded wait, SIGKILL survivors. Never raises.
    """
    if _IS_WINDOWS or not children:
        return 0
    reaped = 0
    try:
        import psutil  # type: ignore

        live = []
        for child in children:
            try:
                if not child.is_running():
                    continue
                if child.status() == psutil.STATUS_ZOMBIE:
                    continue
                if child.ppid() == parent_pid:
                    logger.debug(
                        "Skipping child PID %d of old gateway %d: parent "
                        "still appears alive",
                        child.pid,
                        parent_pid,
                    )
                    continue
                child.terminate()
                live.append(child)
            except psutil.NoSuchProcess:
                continue
            except Exception:
                logger.debug(
                    "Could not terminate child PID %s of old gateway %d",
                    getattr(child, "pid", "?"),
                    parent_pid,
                    exc_info=True,
                )
        if not live:
            return 0
        gone, alive = psutil.wait_procs(live, timeout=max(0.0, timeout))
        reaped = len(gone)
        for child in alive:
            try:
                child.kill()
                reaped += 1
            except Exception:
                logger.debug(
                    "Could not force-kill child PID %s of old gateway %d",
                    getattr(child, "pid", "?"),
                    parent_pid,
                    exc_info=True,
                )
        if reaped:
            logger.info(
                "Reaped %d orphaned child process(es) of replaced gateway PID %d.",
                reaped,
                parent_pid,
            )
    except Exception:
        logger.debug(
            "Child reap for replaced gateway PID %d failed", parent_pid, exc_info=True
        )
    return reaped


def take_over_scoped_lock_holder(
    record: dict[str, Any],
    *,
    graceful_attempts: int = 20,
    force_attempts: int = 20,
) -> Optional[int]:
    """Terminate one verified scoped-lock holder for explicit ``--replace``.

    Returns the owner PID only after that exact PID/start-time identity exited;
    validation or marker-write failure returns None without signalling. Stricter
    than same-home replacement: a cross-home handoff must place a consumable
    marker in the target's home or its supervisor could revive it (flap loop).
    On POSIX the owner's snapshotted children are then reaped best-effort.
    """
    owner = _validated_scoped_lock_gateway_owner(record)
    if owner is None:
        return None
    owner_pid, owner_start_time, target_home = owner
    # Snapshot while the owner is alive; afterwards children are reparented.
    owner_children = _snapshot_gateway_children(owner_pid)

    replaced = _terminate_scoped_lock_owner_once(
        owner_pid,
        owner_start_time,
        target_home,
        graceful_attempts=graceful_attempts,
        force_attempts=force_attempts,
    )
    if replaced is not None:
        reap_gateway_children(owner_children, parent_pid=owner_pid)
    return replaced


def _terminate_scoped_lock_owner_once(
    owner_pid: int,
    owner_start_time: int,
    target_home: Path,
    *,
    graceful_attempts: int = 20,
    force_attempts: int = 20,
) -> Optional[int]:
    """Marker-write + bounded identity-aware termination of a verified owner."""
    if not write_takeover_marker(
        owner_pid,
        target_home=target_home,
        target_start_time=owner_start_time,
    ):
        return None

    try:
        state = _scoped_lock_owner_state(owner_pid, owner_start_time)
        if state == "exited":
            return owner_pid
        if state != "same":
            return None

        try:
            terminate_pid(owner_pid, force=False)
        except ProcessLookupError:
            return owner_pid
        except (PermissionError, OSError):
            return None

        exited, safe_to_force = _wait_for_scoped_lock_owner_exit(
            owner_pid,
            owner_start_time,
            attempts=graceful_attempts,
            delay=0.5,
        )
        if exited:
            return owner_pid
        if not safe_to_force:
            return None

        try:
            terminate_pid(
                owner_pid,
                force=True,
                expected_start_time=owner_start_time,
            )
        except ProcessLookupError:
            return owner_pid
        except (PermissionError, OSError):
            return None

        exited, _ = _wait_for_scoped_lock_owner_exit(
            owner_pid,
            owner_start_time,
            attempts=force_attempts,
            delay=0.25,
        )
        return owner_pid if exited else None
    finally:
        # The target normally consumes the marker; clean up any remainder.
        clear_takeover_marker(target_home)


def write_planned_stop_marker(target_pid: int) -> bool:
    """Record that ``target_pid`` is being stopped intentionally.

    Unexpected SIGTERM exits non-zero so service managers revive the gateway;
    the CLI writes this marker first so a deliberate stop exits cleanly.
    """
    try:
        target_start_time = _get_process_start_time(target_pid)
        record = {
            "target_pid": target_pid,
            "target_start_time": target_start_time,
            "stopper_pid": os.getpid(),
            "written_at": _utc_now_iso(),
        }
        _write_json_file(_get_planned_stop_marker_path(), record)
        return True
    except (OSError, PermissionError):
        return False


def consume_planned_stop_marker_for_self() -> bool:
    """Return True when the current process is being intentionally stopped."""
    return _consume_pid_marker_for_self(_get_planned_stop_marker_path(), ttl_s=_PLANNED_STOP_MARKER_TTL_S)


def planned_stop_marker_targets_self() -> bool:
    """Non-destructive probe: True when a live planned-stop marker names this process.

    Used by the watcher thread (``gateway/run.py:_run_planned_stop_watcher``).
    Unlike :func:`consume_planned_stop_marker_for_self` it never unlinks a
    matching marker -- the shutdown handler does the authoritative consume.
    Malformed/expired markers are still cleaned up; markers naming another
    PID are left for that process and report False here.
    """
    parsed = _read_live_pid_marker(_get_planned_stop_marker_path(), _PLANNED_STOP_MARKER_TTL_S)
    if parsed is None:
        return False
    _, target_pid, target_start_time = parsed
    return _pid_marker_names_self(target_pid, target_start_time)


def clear_planned_stop_marker() -> None:
    """Remove the planned-stop marker unconditionally."""
    _unlink_quietly(_get_planned_stop_marker_path())


def get_running_pid(
    pid_path: Optional[Path] = None,
    *,
    cleanup_stale: bool = True,
) -> Optional[int]:
    """PID of a running gateway (lock + PID file verified against the live process), or None."""
    resolved_pid_path = pid_path or _get_pid_path()
    resolved_lock_path = _get_gateway_lock_path(resolved_pid_path)
    if is_gateway_runtime_lock_active(resolved_lock_path):
        for record in (_read_pid_record(resolved_pid_path), _read_gateway_lock_record(resolved_lock_path)):
            pid = _pid_from_record(record)
            if pid is None or not _pid_exists(pid) or _start_time_conflicts(record, pid):
                continue
            if not _pid_record_belongs_to_current_profile(record):
                continue
            if _record_matches_live_gateway_pid(record, pid):
                return pid
        _cleanup_invalid_pid_path(resolved_pid_path, cleanup_stale=cleanup_stale)
        return get_runtime_status_running_pid() if pid_path is None else None
    # Lock inactive: the runtime-status fallback runs BEFORE cleanup here.
    if pid_path is None:
        runtime_pid = get_runtime_status_running_pid()
        if runtime_pid is not None:
            return runtime_pid
    _cleanup_invalid_pid_path(resolved_pid_path, cleanup_stale=cleanup_stale)
    return None


def get_running_pid_identity_strict(pid_path: Path) -> Optional[tuple[int, float]]:
    """Return a verified process identity or fail on ambiguous runtime state."""
    resolved_pid_path = Path(pid_path)
    resolved_lock_path = _get_gateway_lock_path(resolved_pid_path)
    pid_exists = _strict_path_exists(resolved_pid_path, "gateway PID")
    lock_exists = _strict_path_exists(resolved_lock_path, "gateway lock")
    if not lock_exists:
        return None  # A stale PID file without a lock is not a live gateway.
    if not _is_gateway_runtime_lock_active_strict(resolved_lock_path):
        return None  # Lock probe is authoritative for absence.
    if not pid_exists:
        raise RuntimeError("active gateway lock has no PID metadata")
    pid_record = _read_pid_record(resolved_pid_path)
    lock_record = _read_gateway_lock_record(resolved_lock_path)
    if not pid_record or not lock_record:
        raise RuntimeError("gateway PID or lock metadata is malformed")
    pid = _pid_from_record(pid_record)
    if pid is None or pid <= 0 or _pid_from_record(lock_record) != pid:
        raise RuntimeError("gateway PID and lock identities disagree")
    if not _pid_exists(pid):
        raise RuntimeError("gateway identity is not live")
    current_start = _get_process_start_time(pid)
    starts = (pid_record.get("start_time"), lock_record.get("start_time"))
    if current_start is None or any(start is None for start in starts):
        raise RuntimeError("gateway creation time is unavailable")
    try:
        current = float(current_start)
        recorded = tuple(float(start) for start in starts)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("gateway creation time is malformed") from exc
    if current <= 0 or any(start <= 0 or abs(start - current) > 0.001 for start in recorded):
        raise RuntimeError("gateway process identity changed")
    if not all(_record_matches_live_gateway_pid(record, pid) for record in (pid_record, lock_record)):
        raise RuntimeError("runtime metadata does not identify a live gateway")
    # Windows persists a centisecond fingerprint; SCM checks need the exact psutil
    # epoch. Re-read only after validation and prove it rounds to the same value.
    if _IS_WINDOWS:
        try:
            import psutil  # type: ignore

            exact_create_time = float(psutil.Process(pid).create_time())
        except Exception as exc:
            raise RuntimeError("exact gateway creation time is unavailable") from exc
        if int(round(exact_create_time * 100)) != int(current):
            raise RuntimeError("gateway process identity changed")
        return pid, exact_create_time
    return pid, current


def get_running_pid_cached(
    pid_path: Optional[Path] = None,
    *,
    cleanup_stale: bool = True,
    ttl_seconds: float = _GATEWAY_RUNNING_PID_CACHE_TTL_SECONDS,
) -> Optional[int]:
    """Cached ``get_running_pid()`` for high-frequency dashboard polling.

    Short TTL, invalidated on PID/lock/runtime-status file changes, so status
    endpoints do not re-flock ``gateway.lock`` hundreds of times a minute.
    """
    if ttl_seconds <= 0:
        return get_running_pid(pid_path, cleanup_stale=cleanup_stale)

    resolved_pid_path = pid_path or _get_pid_path()
    include_runtime_status = pid_path is None
    signature = _running_pid_cache_signature(
        resolved_pid_path,
        include_runtime_status=include_runtime_status,
    )
    key = (str(resolved_pid_path), bool(cleanup_stale), include_runtime_status)
    now = time.monotonic()

    with _gateway_running_pid_cache_lock:
        cached = _gateway_running_pid_cache.get(key)
        if cached is not None:
            cached_at, cached_signature, cached_pid = cached
            if now - cached_at <= ttl_seconds and cached_signature == signature:
                return cached_pid

    pid = get_running_pid(pid_path, cleanup_stale=cleanup_stale)
    refreshed_signature = _running_pid_cache_signature(
        resolved_pid_path,
        include_runtime_status=include_runtime_status,
    )
    with _gateway_running_pid_cache_lock:
        _gateway_running_pid_cache[key] = (
            time.monotonic(),
            refreshed_signature,
            pid,
        )
    return pid


def is_gateway_running(
    pid_path: Optional[Path] = None,
    *,
    cleanup_stale: bool = True,
) -> bool:
    """Check if the gateway daemon is currently running."""
    return get_running_pid(pid_path, cleanup_stale=cleanup_stale) is not None
