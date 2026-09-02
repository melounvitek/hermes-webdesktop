"""Windows subprocess compatibility helpers.

* ``["npm", "install", ...]`` — on Windows ``npm`` is ``npm.cmd``, a batch shim.
``subprocess.Popen(["npm", ...])`` fails with WinError 193 ("not a valid Win32 application") because
CreateProcessW can't run a ``.cmd`` file without ``shell=True`` or PATHEXT resolution.

* ``start_new_session=True`` — on POSIX, this maps to ``os.setsid()`` and actually detaches the
child. On Windows it's silently ignored; the Windows equivalent is the ``CREATE_NEW_PROCESS_GROUP |
CREATE_NO_WINDOW`` creationflags bundle, which Python only applies when you pass it explicitly.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from typing import Mapping, Sequence

__all__ = [
    "IS_WINDOWS",
    "resolve_node_command",
    "split_command_line",
    "suppress_platform_ver_console",
    "windows_detach_flags",
    "windows_detach_flags_without_breakaway",
    "windows_hide_flags",
    "windows_detach_popen_kwargs",
    "bounded_git_probe",
    "bounded_probe_run",
    "noninteractive_git_env",
    "NO_DRIVER_DIFF_FLAGS",
    "pid_is_hermes",
]

# Flags that neutralize *attribute-scoped* diff drivers on any diff-rendering
# git command (``diff``, ``log -p``, ``show``, ``blame``). A malicious repo can
# name a driver in ``.gitattributes`` (``* diff=evil``) and point it at an
# arbitrary program via ``[diff "evil"] command=/textconv=`` in ``.git/config``.
# Because the attacker chooses the driver name, ``GIT_CONFIG_KEY`` overrides in
# ``noninteractive_git_env`` cannot enumerate and disable it — only these
# command-line flags do. ``--no-ext-diff`` kills ``command=``; ``--no-textconv``
# kills ``textconv=``. Both are required (verified empirically: each alone
# leaves the other live). Smudge/clean filters are neutralized by the env
# layer's ``core.hooksPath`` + running against the index without checkout.
NO_DRIVER_DIFF_FLAGS = ("--no-ext-diff", "--no-textconv")

# Subcommands that render diffs and therefore invoke ``.gitattributes``-scoped
# diff/textconv drivers. Only these accept ``NO_DRIVER_DIFF_FLAGS`` — ``status``
# and friends reject the flags (``unknown option``), so the helper must gate on
# this set rather than blanket-prepending.
_DIFF_RENDERING_SUBCOMMANDS = frozenset({"diff", "show", "log", "blame"})


def harden_git_argv(args: Sequence[str]) -> list[str]:
    """Return a copy of subcommand-first git *args* with diff-driver flags
    inserted for diff-rendering subcommands.

    *args* is the argument list WITHOUT the leading ``"git"`` (e.g.
    ``["diff", "HEAD"]`` or ``["-c", "core.quotePath=false", "diff", ...]``).
    The first non-option token is treated as the subcommand; if it is one of
    :data:`_DIFF_RENDERING_SUBCOMMANDS`, :data:`NO_DRIVER_DIFF_FLAGS` is
    inserted immediately after it. Non-diff subcommands are returned unchanged.

    Pair with :func:`noninteractive_git_env`: the env layer disables
    fsmonitor/hooks/pager/editor/credential sinks, this closes the one class
    (attacker-named attribute drivers) env overrides cannot reach.
    """
    out = list(args)
    # Options that consume the FOLLOWING token as their value, so that value is
    # never mistaken for the subcommand (``-C diff`` is a path; ``-c diff=x`` is
    # a config pair — neither is the diff subcommand).
    _value_opts = {"-C", "-c", "--git-dir", "--work-tree", "--namespace", "--exec-path"}
    i = 0
    while i < len(out):
        tok = out[i]
        if tok in _value_opts:
            i += 2
            continue
        if tok.startswith("-"):
            i += 1
            continue
        if tok in _DIFF_RENDERING_SUBCOMMANDS:
            return out[: i + 1] + list(NO_DRIVER_DIFF_FLAGS) + out[i + 1 :]
        # First non-option token is the subcommand; if it isn't a diff renderer
        # there is nothing to harden.
        return out
    return out


IS_WINDOWS = sys.platform == "win32"

# Private launcher-to-child metadata. This is diagnostic state, not user config.
_WINDOWS_GATEWAY_BREAKAWAY_ENV = "_HERMES_GATEWAY_BREAKAWAY"


def split_command_line(line: str) -> list[str]:
    """Split a user-supplied command line into tokens, Windows-safely.

    ``shlex.split`` (posix=True) treats every backslash as an escape, silently mangling Windows
    paths (the separators vanish, leaving a wrong relative name). On Windows use ``posix=False``
    and strip one layer of matching double quotes per token; on POSIX this is exactly
    ``shlex.split``. Raises ValueError on unbalanced quotes.
    """
    if not IS_WINDOWS:
        import shlex

        return shlex.split(line)
    import shlex

    tokens = shlex.split(line, posix=False)
    out: list[str] = []
    for tok in tokens:
        if len(tok) >= 2 and tok[0] == tok[-1] and tok[0] in ("'", '"'):
            tok = tok[1:-1]
        out.append(tok)
    return out


# -----------------------------------------------------------------------------
# Node ecosystem launcher resolution
# -----------------------------------------------------------------------------


def resolve_node_command(name: str, argv: Sequence[str]) -> list[str]:
    """Resolve a Node-ecosystem command name to an absolute-path argv.

    On Windows, commands like ``npm``, ``npx``, ``yarn``, ``pnpm``, ``playwright``, ``prettier``
    ship as ``.cmd`` files (batch shims). ``subprocess.Popen(["npm", "install"])`` fails with
    WinError 193 because CreateProcessW doesn't execute batch files directly.

    ``shutil.which(name)`` *does* resolve ``.cmd`` via PATHEXT and returns the fully-qualified path
    — which CreateProcessW accepts because the extension tells Windows to route through ``cmd.exe
    /c``.
    """
    resolved = shutil.which(name)
    if resolved:
        return [resolved, *argv]
    return [name, *argv]


# -----------------------------------------------------------------------------
# Detached / hidden process creation
# -----------------------------------------------------------------------------


# Win32 CreationFlags — defined here rather than imported from subprocess
# because CREATE_NO_WINDOW and DETACHED_PROCESS aren't guaranteed to be
# present on stdlib subprocess on older Pythons or non-Windows builds.
_CREATE_NEW_PROCESS_GROUP = 0x00000200
# DETACHED_PROCESS is intentionally NOT part of any flag bundle here — do not
# re-add it.  Two reasons (the recurring console-flash bug #54220 / #56747):
#
# 1. MSDN (Process Creation Flags): CREATE_NO_WINDOW "is ignored if used with
#    either CREATE_NEW_CONSOLE or DETACHED_PROCESS".  Combining them means
#    DETACHED_PROCESS governs and the no-window bit is dead.
# 2. A DETACHED_PROCESS child has NO console at all, so every console-subsystem
#    descendant it ever spawns (git, gh, cmd, node, wmic, powershell, …) must
#    allocate its OWN console — a visible flash per spawn, including spawns
#    inside third-party libraries that no per-call-site CREATE_NO_WINDOW sweep
#    can reach.  A CREATE_NO_WINDOW child instead OWNS a hidden console that
#    all descendants inherit, making "no flashing windows" a property of the
#    one daemon launch.  Root cause isolated + A/B verified on Windows 11 by
#    the desktop backend fix (commit aa2ae36c3f): with per-site hide flags
#    neutered, naive git/gh/cmd spawns don't flash under a hidden-console
#    parent and do flash under a console-less one.
_DETACHED_PROCESS = 0x00000008  # kept for reference; must stay out of bundles
_CREATE_NO_WINDOW = 0x08000000
# Escape any Win32 job object the parent process belongs to. Without this,
# a detached child still inherits its parent's job object membership, and
# when that parent (Electron, Tauri, Windows Terminal, the Desktop GUI's
# bootstrap-installer) dies, the OS tears down the whole job — taking the
# "detached" child with it. Critical for the post-update gateway watcher:
# Electron spawns the Tauri updater inside its own job, the updater spawns
# the watcher subprocess; without BREAKAWAY the watcher dies the instant
# Electron exits, so the gateway never gets respawned after a `hermes
# update` triggered from the GUI. See fix/windows-gateway-reliability.
_CREATE_BREAKAWAY_FROM_JOB = 0x01000000


def windows_detach_flags() -> int:
    """Return Win32 creationflags detaching a child from the parent console/group; 0 elsewhere.

    Pair with the default ``start_new_session=False`` (POSIX uses ``start_new_session=True``).
    CREATE_NEW_PROCESS_GROUP stops Ctrl+C propagating; CREATE_NO_WINDOW gives the child a hidden
    console that descendants inherit, avoiding per-descendant console flashes (DETACHED_PROCESS
    would make CREATE_NO_WINDOW ignored and re-create that bug); CREATE_BREAKAWAY_FROM_JOB escapes
    Electron/Tauri job objects that would otherwise kill the child with the parent. A job that
    forbids breakaway yields PermissionError from Popen -- callers catch OSError and fall back.
    """
    if not IS_WINDOWS:
        return 0
    return (
        _CREATE_NEW_PROCESS_GROUP
        | _CREATE_NO_WINDOW
        | _CREATE_BREAKAWAY_FROM_JOB
    )


def windows_detach_flags_without_breakaway() -> int:
    """Same as :func:`windows_detach_flags` minus ``CREATE_BREAKAWAY_FROM_JOB``.

    Retry with this when the breakaway variant raises OSError (job disallows breakaway), instead
    of hand-coding the bit mask at every site. Returns 0 on non-Windows.
    """
    if not IS_WINDOWS:
        return 0
    return _CREATE_NEW_PROCESS_GROUP | _CREATE_NO_WINDOW


def windows_hide_flags() -> int:
    """Return Win32 creationflags hiding the child's console without detaching it; 0 elsewhere.

    For short-lived helpers (``taskkill``, ``where``, version probes) run synchronously: no
    flash, but the child stays in the parent's process group and job so Ctrl+C and job teardown
    still propagate. Stdio is inherited, so ``capture_output=True`` works.
    """
    if not IS_WINDOWS:
        return 0
    return _CREATE_NO_WINDOW


def suppress_platform_ver_console() -> None:
    """Stub ``platform._syscmd_ver`` on Windows so it never flashes a console. No-op elsewhere.

    ``platform.win32_ver()`` shells out ``cmd /c ver`` without CREATE_NO_WINDOW, so a windowless
    parent (pythonw gateway, kanban workers) flashes a visible cmd window whenever a dependency
    touches ``platform.uname()`` at import. With the stub, ``win32_ver()`` takes its documented
    fallback to ``sys.getwindowsversion()`` -- same data, in-process. Call before heavy imports.
    """
    if not IS_WINDOWS:
        return
    try:
        import platform

        if hasattr(platform, "_syscmd_ver"):
            def _quiet_syscmd_ver(system="", release="", version="",
                                  supported_platforms=("win32", "win16", "dos")):
                return system, release, version

            platform._syscmd_ver = _quiet_syscmd_ver
    except Exception:
        # Purely cosmetic hardening — never let it break startup.
        pass


def windows_detach_popen_kwargs() -> dict:
    """Return Popen kwargs detaching a child on Windows, or ``start_new_session=True`` on POSIX.

    Replaces bare ``start_new_session=True``, which is accepted but has no effect on Windows: the
    child stays attached to the parent console and dies when it closes.
    """
    if IS_WINDOWS:
        return {"creationflags": windows_detach_flags()}
    return {"start_new_session": True}


# -----------------------------------------------------------------------------
# Non-interactive git environment (credential-prompt hang guard)
# -----------------------------------------------------------------------------


def noninteractive_git_env(
    base: "Mapping[str, str] | None" = None,
) -> dict[str, str]:
    """Environment for *internal* git invocations that must never prompt.

    * ``GIT_TERMINAL_PROMPT=0`` — git fails with "terminal prompts disabled" instead of prompting
    for credentials. * ``GCM_INTERACTIVE=Never`` — Git Credential Manager (the default credential
    helper on Windows installs) never pops its own dialog.

    Returns a copy of ``base`` (default ``os.environ``) with ``GIT_TERMINAL_PROMPT=0`` (fail instead
    of prompting), ``GCM_INTERACTIVE=Never`` (no Git Credential Manager dialog), and isolated git
    config: inherited ``GIT_CONFIG_*`` overrides, global/system config, pagers, editors, fsmonitor,
    external diff, and hooks are all disabled for the child so a user's repo/global config cannot
    hang or mutate Hermes's internal plumbing calls.

    ``GIT_ASKPASS`` / ``SSH_ASKPASS`` are deliberately left alone: when the user has a *working*
    askpass helper or ssh-agent configured, auth should still succeed non-interactively. The env
    only disables paths that block on a human. Pair with ``stdin=subprocess.DEVNULL``. Internal
    plumbing only — the agent-facing terminal tool has its own policy layer and visible PTY.
    """
    env = dict(base if base is not None else os.environ)
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GCM_INTERACTIVE"] = "Never"

    # Do not inherit caller-supplied config injection. We rebuild the
    # GIT_CONFIG_COUNT block below so ambient -c values cannot re-enable
    # pagers, hooks, fsmonitor, editors, or credential prompts.
    for key in list(env):
        if (
            key == "GIT_CONFIG_PARAMETERS"
            or key.startswith("GIT_CONFIG_KEY_")
            or key.startswith("GIT_CONFIG_VALUE_")
        ):
            env.pop(key, None)
    env.pop("GIT_CONFIG_COUNT", None)

    devnull = os.devnull
    env["GIT_CONFIG_GLOBAL"] = devnull
    env["GIT_CONFIG_SYSTEM"] = devnull
    env["GIT_CONFIG_NOSYSTEM"] = "1"
    env["GIT_PAGER"] = "cat"
    env["PAGER"] = "cat"
    env["GIT_EDITOR"] = "true"

    config_overrides = {
        "credential.helper": "",
        "core.askPass": "",
        "core.fsmonitor": "false",
        "core.untrackedCache": "false",
        "core.hooksPath": devnull,
        "core.pager": "cat",
        "core.editor": "true",
        "sequence.editor": "true",
        "diff.external": "",
    }
    env["GIT_CONFIG_COUNT"] = str(len(config_overrides))
    for idx, (key, value) in enumerate(config_overrides.items()):
        env[f"GIT_CONFIG_KEY_{idx}"] = key
        env[f"GIT_CONFIG_VALUE_{idx}"] = value

    return env


# -----------------------------------------------------------------------------
# Bounded, fail-open git probing (Windows post-kill deadlock guard)
# -----------------------------------------------------------------------------


def _process_start_time(pid: int) -> int | None:
    """Return the repository's stable process-start fingerprint, if available."""
    try:
        from gateway.status import get_process_start_time

        return get_process_start_time(pid)
    except Exception:
        return None


def _text_names_hermes(text: str) -> bool:
    r"""True when *text* names Hermes at a path-segment / token boundary.

    A bare ``"hermes" in text`` substring test would also match unrelated processes whose paths
    merely contain the letters (``...\shermesa\...``), which is exactly the false-positive class
    this guard exists to prevent.
    """
    for token in re.split(r"[\\/\s=,;\"']+", text.lower()):
        if token.startswith("hermes") or token.startswith(".hermes"):
            return True
    return False


def _process_command_is_hermes(pid: int) -> bool:
    """Best-effort check that *pid* currently runs Hermes code."""
    try:
        import psutil

        process = psutil.Process(pid)
        command = " ".join(process.cmdline() or [])
        executable = process.exe() or ""
        return _text_names_hermes(f"{command} {executable}")
    except Exception:
        return False


def pid_is_hermes(
    pid: int,
    *,
    expected_start_time: int | None = None,
) -> bool:
    """Return whether it is safe to use ``taskkill`` for *pid*.

    The PID must be valid, currently exist, and identify a Hermes process. When the caller captured
    a start-time fingerprint before the destructive action, the live process must still have the
    same ``(pid, start_time)`` identity. Any ambiguity fails closed.
    """
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return False
    if not IS_WINDOWS:
        if expected_start_time is None:
            return True
        try:
            return _process_start_time(pid) == expected_start_time
        except Exception:
            return False

    try:
        current_start_time = _process_start_time(pid)
    except Exception:
        return False
    if current_start_time is None:
        return False
    if (
        expected_start_time is not None
        and current_start_time != expected_start_time
    ):
        return False
    try:
        return _process_command_is_hermes(pid)
    except Exception:
        return False


def kill_process_tree(proc: "subprocess.Popen") -> None:
    """Best-effort terminate *proc* and its descendants on both platforms.

    ``proc.kill()`` alone only terminates the direct child. On Windows a suspended descendant (e.g.

    All failures are swallowed — this is cleanup on an already-failing path, and the caller's
    contract is to fail open. ``kill()`` can raise (access denied, already reaped); an unhandled
    raise here would escape the caller's ``except`` handler and break that contract.
    """
    try:
        from agent.deadline import kill_process_tree as _deadline_kill_tree

        _deadline_kill_tree(proc.pid)
    except Exception:
        _legacy_kill_process_tree(proc)
        return
    # Ensure Popen's own bookkeeping sees the exit (matches the legacy body:
    # a direct kill() so communicate()/wait() cannot hang on a stale handle).
    try:
        proc.kill()
    except OSError:
        pass


def _legacy_kill_process_tree(proc: "subprocess.Popen") -> None:
    """Pre-#85125 local tree-kill — fallback when agent.deadline is unavailable.

    Kept verbatim so ``kill_process_tree`` can honor its swallow-everything contract even when the
    delegation path itself fails (partial install, import cycle during teardown).
    """
    if not IS_WINDOWS:
        # Group-kill first: verify the child actually leads its own process
        # group before signalling it, so we never blast a shared group.
        try:
            import signal as _signal

            pgid = os.getpgid(proc.pid)
            if pgid == proc.pid:
                os.killpg(pgid, _signal.SIGKILL)  # windows-footgun: ok — inside `if not IS_WINDOWS` gate
        except Exception:
            pass
    try:
        proc.kill()
    except OSError:
        pass
    if IS_WINDOWS:
        # No identity guard here on purpose: *proc* is our own retained
        # ``Popen`` handle. The child cannot be reaped (and its PID cannot be
        # recycled) while we still hold the handle, so an identity check could
        # only ever false-refuse a legitimate cleanup. The fail-closed
        # ``pid_is_hermes`` guard is for BARE pids from state files or process
        # scans, where recycling is real.
        try:
            subprocess.run(
                ["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                timeout=2,
                check=False,
                creationflags=windows_hide_flags(),
            )
        except Exception:
            pass


def bounded_probe_run(
    argv: Sequence[str],
    *,
    timeout: float,
    errors: str = "replace",
    env: "Mapping[str, str] | None" = None,
) -> "subprocess.CompletedProcess[str] | None":
    """Deadlock-safe ``subprocess.run(argv, capture_output=True, timeout=...)`` for fail-open probe
    call sites. Returns a ``CompletedProcess`` when the child finished within *timeout* (any exit
    code), or ``None`` on spawn failure or timeout.
    """
    _popen_kwargs: dict = {"creationflags": windows_hide_flags()} if IS_WINDOWS else {"process_group": 0}
    try:
        proc = subprocess.Popen(
            list(argv),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors=errors,
            env=dict(env) if env is not None else None,
            **_popen_kwargs,
        )
    except Exception:
        return None
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except Exception:
        # Timeout OR any other communicate() failure (torn-down pipe, decode
        # error): terminate the child + descendants and drain bounded. Leaving
        # it running would leak the same suspended-descendant class this guards.
        kill_process_tree(proc)
        try:
            proc.communicate(timeout=1)
        except Exception:
            pass
        return None
    return subprocess.CompletedProcess(list(argv), proc.returncode, stdout, stderr)


def bounded_git_probe(argv: Sequence[str], *, timeout: float) -> str:
    """Run a short ``git`` probe and return stripped stdout, or ``""`` on ANY failure.

    Deadlock-safe replacement for ``subprocess.run(["git", ...], timeout=...)`` at fail-open
    probe sites. On Windows ``run()``'s post-timeout cleanup calls an unbounded ``communicate()``;
    a suspended descendant git.exe holding the pipe handles then blocks forever. Here: bounded
    ``communicate``, then tree-kill plus a 1s drain, then abandon the pipes. Spawn contract
    matches ``run`` byte-for-byte; on POSIX the probe gets its own process group so cleanup
    also takes down credential helpers and remote helpers.

    Security (GHSA-7x36-8jrh-v4pw): these probes run automatically against whatever directory
    the session sits in, before any tool call or trust prompt, and an index refresh executes the
    repo-configured ``core.fsmonitor`` program (hooks/pager/editor/credential helper are sinks
    too). A repo delivered with its ``.git`` intact would get host code execution. Every probe
    therefore runs under :func:`noninteractive_git_env`; diff-rendering callers additionally pass
    :data:`NO_DRIVER_DIFF_FLAGS` (attribute-scoped drivers can't be disabled via env).
    """
    result = bounded_probe_run(argv, timeout=timeout, env=noninteractive_git_env())
    if result is None or result.returncode != 0:
        return ""
    return (result.stdout or "").strip()


# Backward-compat alias — existing call sites/tests import the historical name.
_kill_git_process_tree = kill_process_tree
