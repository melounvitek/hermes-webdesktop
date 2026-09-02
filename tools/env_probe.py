"""Local-environment toolchain probe for the system prompt.

When the terminal backend is local, surface one deterministic line about
Python tooling state (python3/python versions, missing pip module, pip bound
to a different Python than ``python3``, PEP 668 externally-managed) so models
don't discover it by hitting walls.  The probe is cheap (~50ms), cached for
the process lifetime, and emits nothing when the environment is clean.

Remote terminal backends (docker, modal, ssh, …) are skipped: the host's
Python state is irrelevant when tools run inside a sandbox, which has its own
probe (``_probe_remote_backend`` in ``agent/prompt_builder.py``).

Toggle via ``agent.environment_probe`` in config.yaml (default True).
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
import threading
from typing import Optional

from hermes_cli._subprocess_compat import windows_hide_flags

logger = logging.getLogger(__name__)

# Concurrency model: the probe runs in exactly ONE background worker thread;
# ``_PROBE_DONE`` signals completion.  Callers never execute the probe
# themselves and block at most ``_PROBE_WAIT_TIMEOUT`` seconds on the event
# before failing open with "" — a stuck probe (e.g. a Windows pipe wedged open
# by an orphaned pip descendant) can degrade only the probe line, never
# system-prompt construction.
_CACHE_LOCK = threading.Lock()
_CACHED_LINE: Optional[str] = None  # None = not probed yet; "" = probed, nothing to say.
_PROBE_DONE = threading.Event()
_PROBE_THREAD: Optional[threading.Thread] = None
# Generation counter — bumped on every reset so a stale worker (started
# before a test reset) can't publish its result into the fresh generation.
_PROBE_GEN = 0

# Upper bound a prompt build will wait for the probe.  Generous vs the ~0.5s
# healthy runtime, but finite: prompt construction must always proceed.
_PROBE_WAIT_TIMEOUT = 10.0
# Once one caller has burned the full wait, later callers only peek at the
# event.  If the stuck worker ever finishes, the line resumes appearing.
_WAIT_ALREADY_TIMED_OUT = False

# Keep in sync with agent/prompt_builder.py:_REMOTE_TERMINAL_BACKENDS.
# Duplicated rather than imported to avoid a circular import.
_REMOTE_BACKENDS = frozenset({
    "docker", "singularity", "modal", "daytona", "ssh", "managed_modal",
    "vercel_sandbox",
})


def _plugin_backend_is_remote(backend: str) -> bool:
    """Whether a plugin-registered terminal backend is remote (fail-soft)."""
    if not backend or backend in _REMOTE_BACKENDS or backend == "local":
        return False
    try:
        from agent.terminal_env_registry import provider_flag

        return bool(provider_flag(backend, "is_remote", False))
    except Exception:
        return False


def _run(cmd: list[str], timeout: float = 3.0) -> tuple[int, str, str]:
    """Run a short subprocess.  Returns (returncode, stdout, stderr).

    Failures (binary missing, timeout, OSError) return (-1, "", "<reason>").

    Output is captured through temp files rather than pipes so ``timeout``
    bounds the *whole* call, even on native Windows: a console-script launcher
    (e.g. ``pip.exe``) can spawn a descendant that inherits the captured
    handles and outlives its parent.  With OS pipes, ``communicate()``'s reader
    threads block until that grandchild closes the write end — which the
    timeout does not cover, since killing the direct child leaves the
    grandchild holding the pipe (a warm probe could hang ~28 min holding
    ``_CACHE_LOCK``).  Temp files have no reader threads, so ``wait()`` only
    waits on the direct child and the probe genuinely fails open on timeout.
    """
    try:
        with tempfile.TemporaryFile() as out_f, tempfile.TemporaryFile() as err_f:
            try:
                result = subprocess.run(
                    cmd,
                    stdout=out_f,
                    stderr=err_f,
                    timeout=timeout,
                    check=False,
                    stdin=subprocess.DEVNULL,
                    # CREATE_NO_WINDOW (0 on POSIX): windowless hosts (pythonw
                    # gateway / kanban workers) would otherwise flash a console
                    # window per probe subprocess.
                    creationflags=windows_hide_flags(),
                )
            except subprocess.TimeoutExpired:
                return -1, "", "timeout"
            out_f.seek(0)
            err_f.seek(0)
            out = out_f.read().decode("utf-8", "replace").strip()
            err = err_f.read().decode("utf-8", "replace").strip()
            return result.returncode, out, err
    except FileNotFoundError:
        return -1, "", "not found"
    except OSError as exc:
        return -1, "", f"oserror: {exc}"


def _python_version_of(binary: str) -> Optional[str]:
    """Return a short version string like ``3.12.4`` for ``binary``, or None."""
    if not shutil.which(binary):
        return None
    rc, out, err = _run([binary, "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')"])
    return out if rc == 0 and out else None


def _has_pip_module(binary: str) -> bool:
    """True if ``<binary> -m pip --version`` succeeds."""
    if not shutil.which(binary):
        return False
    rc, _out, _err = _run([binary, "-m", "pip", "--version"])
    return rc == 0


def _detect_pep668(binary: str) -> bool:
    """True when ``<binary>`` is PEP-668 externally-managed (``EXTERNALLY-MANAGED``
    marker next to the stdlib, as Debian/Ubuntu ship)."""
    if not shutil.which(binary):
        return False
    code = (
        "import sys, os;"
        "stdlib = os.path.dirname(os.__file__);"
        "marker = os.path.join(stdlib, 'EXTERNALLY-MANAGED');"
        "print('yes' if os.path.exists(marker) else 'no')"
    )
    rc, out, _err = _run([binary, "-c", code])
    return rc == 0 and out.strip() == "yes"


def _pip_python_version() -> Optional[str]:
    """If ``pip`` is on PATH, return the Python version it's bound to.

    Parses the trailing ``(python X.Y)`` of ``pip --version`` output, e.g.
    ``pip 24.0 from /usr/lib/python3/dist-packages/pip (python 3.12)`` → ``"3.12"``.
    """
    if not shutil.which("pip"):
        return None
    rc, out, _err = _run(["pip", "--version"])
    if rc != 0 or not out:
        return None
    if "(python " in out and out.endswith(")"):
        return out.rsplit("(python ", 1)[1][:-1].strip()
    return None


def _resolve_terminal_backend() -> str:
    """Scope-aware terminal backend name (``local`` when unresolvable)."""
    try:
        from tools.terminal_scope import terminal_env

        return (terminal_env("TERMINAL_ENV") or "local").strip().lower()
    except Exception:  # never let policy resolution break prompt building
        logger.debug("terminal backend resolution failed", exc_info=True)
        return "local"


def _build_probe_line() -> str:
    """Build the one-liner.  Returns "" when nothing notable is detected —
    the goal is to save the model from an avoidable wall, not narrate a
    healthy environment."""
    py3_ver = _python_version_of("python3")
    py_ver = _python_version_of("python")  # for systems with a `python` alias
    py3_has_pip = _has_pip_module("python3") if py3_ver else False
    pip_bound_to = _pip_python_version()
    py3_pep668 = _detect_pep668("python3") if py3_ver else False
    # Bare which() is correct here, unlike Hermes's own uv call sites: this
    # reports the environment *the model will see* in the terminal tool, whose
    # PATH (via local.py) includes the Hermes-managed $HERMES_HOME/bin.
    # Claiming uv the model cannot invoke would be worse than claiming none.
    has_uv = shutil.which("uv") is not None

    mismatch = bool(pip_bound_to and py3_ver and not py3_ver.startswith(pip_bound_to))
    silent_conditions = (
        py3_ver is not None
        and py3_has_pip
        and not mismatch
        and (not py3_pep668 or has_uv)
    )
    if silent_conditions:
        return ""

    # Compact factual summary; ONE line so it doesn't dominate the prompt.
    bits: list[str] = []
    if py3_ver:
        py3_bit = f"python3={py3_ver}"
        if not py3_has_pip:
            py3_bit += " (no pip module)"
        bits.append(py3_bit)
    else:
        bits.append("python3=missing")

    if py_ver and py_ver != py3_ver:
        bits.append(f"python={py_ver}")
    elif not py_ver and py3_ver:
        # Common on Debian/Ubuntu — stop the model typing `python`.
        bits.append("python=missing (use python3)")

    if pip_bound_to:
        if mismatch:
            bits.append(f"pip→python{pip_bound_to} (mismatch)")
        elif not py3_has_pip:
            # pip script works but `python3 -m pip` doesn't.
            bits.append(f"pip→python{pip_bound_to}")
    elif not py3_has_pip:
        # (when `pip` is off PATH but `python3 -m pip` works, say nothing)
        bits.append("pip=missing")

    if py3_pep668:
        bits.append("PEP 668=yes (use venv or uv)")

    if has_uv:
        bits.append("uv=installed")

    return "Python toolchain: " + ", ".join(bits) + "."


def get_environment_probe_line(*, force_refresh: bool = False) -> str:
    """Return the cached probe line (building it on first call).

    Returns "" when the environment is clean — the system prompt assembler
    should drop the section rather than emit an empty heading.  The probe runs
    in a single background worker; this waits at most ``_PROBE_WAIT_TIMEOUT``
    seconds on its completion event and then fails open with "", so a wedged
    probe subprocess can never block system-prompt construction.

    ``force_refresh`` is for tests; real callers should never need it.
    """
    global _WAIT_ALREADY_TIMED_OUT
    if force_refresh:
        _reset_cache_for_tests()

    # Resolve the backend HERE, in the caller's context: under gateway
    # multiplexing the routed profile's backend lives in the per-turn terminal
    # scope, which the bare probe worker thread does not inherit.  A remote
    # backend answers "" without consulting the cache — the cached line
    # describes the HOST toolchain, not where that profile's tools run.
    backend = _resolve_terminal_backend()
    if backend in _REMOTE_BACKENDS or _plugin_backend_is_remote(backend):
        return ""

    if _PROBE_DONE.is_set():
        return _CACHED_LINE or ""

    _ensure_probe_started()
    wait_timeout = 0.05 if _WAIT_ALREADY_TIMED_OUT else _PROBE_WAIT_TIMEOUT
    if not _PROBE_DONE.wait(timeout=wait_timeout):
        # Probe stuck or pathologically slow: the line is a nice-to-have,
        # blocking prompt construction is an outage.  Fail open.
        if not _WAIT_ALREADY_TIMED_OUT:
            _WAIT_ALREADY_TIMED_OUT = True
            logger.warning(
                "env_probe did not finish within %.0fs; building the system "
                "prompt without the Python toolchain line",
                _PROBE_WAIT_TIMEOUT,
            )
        return ""
    return _CACHED_LINE or ""


def _probe_worker(gen: int) -> None:
    """Body of the single probe thread — computes and publishes the line."""
    global _CACHED_LINE
    try:
        line = _build_probe_line()
    except Exception as exc:  # never let probe failure propagate
        logger.debug("env_probe failed: %s", exc)
        line = ""
    with _CACHE_LOCK:
        if gen != _PROBE_GEN:
            return  # superseded by a reset (tests) — discard stale result
        _CACHED_LINE = line
        _PROBE_DONE.set()


def _ensure_probe_started() -> None:
    """Start the probe worker if it isn't running and hasn't finished."""
    global _PROBE_THREAD
    with _CACHE_LOCK:
        if _PROBE_DONE.is_set():
            return
        if _PROBE_THREAD is not None and _PROBE_THREAD.is_alive():
            return
        _PROBE_THREAD = threading.Thread(
            target=_probe_worker,
            args=(_PROBE_GEN,),
            name="env-probe",
            daemon=True,
        )
        _PROBE_THREAD.start()


def warm_environment_probe_async() -> None:
    """Start the probe in the background so the first system-prompt build
    doesn't pay the ~0.5s of subprocess calls on the time-to-first-token path.

    Idempotent and fail-safe; ``get_environment_probe_line`` waits (bounded)
    on the same worker instead of recomputing.  Called from agent init.
    """
    _ensure_probe_started()


def _reset_cache_for_tests() -> None:
    """Test helper — clear the cache between probe scenarios."""
    global _CACHED_LINE, _PROBE_THREAD, _PROBE_GEN, _WAIT_ALREADY_TIMED_OUT
    with _CACHE_LOCK:
        _CACHED_LINE = None
        _PROBE_DONE.clear()
        _PROBE_THREAD = None
        _PROBE_GEN += 1
        _WAIT_ALREADY_TIMED_OUT = False
