"""Local execution environment — spawn-per-call with session snapshot."""

import logging
import ntpath
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Mapping
from pathlib import Path

from hermes_constants import get_process_hermes_home
from tools.environments.base import BaseEnvironment, _pipe_stdin
from hermes_cli._subprocess_compat import windows_hide_flags
# Re-exported so ``from tools.environments.local import X`` and
# ``patch("tools.environments.local.X")`` keep working after the split.
from tools.environments.local_env_policy import (  # noqa: F401
    _ACTIVE_VENV_MARKER_VARS, _ALWAYS_STRIP_KEYS, _AWS_SDK_CREDENTIAL_ENV_VARS,
    _HERMES_PROVIDER_ENV_BLOCKLIST, _HERMES_PROVIDER_ENV_FORCE_PREFIX,
    _TERMINAL_FIRST_PARTY_ENV_PREFIXES, _build_provider_env_blocklist,
    _buzz_terminal_context_active, _is_hermes_internal_secret,
    _is_terminal_first_party_env, _matches_terminal_first_party_prefix,
    _plugin_terminal_env_strip_keys,
)
from tools.environments.local_gitbash_probe import (  # noqa: F401
    _BASH_EXTERNAL_PROGRAM_PROBE, _bash_probe_details_cache, _bash_starts,
    _bash_starts_cache, _git_bash_aslr_help, _git_root_from_bash,
    _looks_like_msys_spawn_failure, _mandatory_aslr_enabled,
)
from tools.environments.local_pythonpath import (  # noqa: F401
    _build_hermes_repo_root_aliases, _get_hermes_site_packages, _same_path,
    _strip_hermes_owned_pythonpath, _strip_hermes_owned_pythonpath_and_runtime_markers,
    _validated_runtime_venv,
)

_IS_WINDOWS = platform.system() == "Windows"

logger = logging.getLogger(__name__)

# --- Terminal temp-cache pruning ---------------------------------------------
# get_temp_dir() defaults to HERMES_HOME/cache/terminal (real storage, not tmpfs
# /tmp), so stale artifacts no longer vanish on reboot for free: the gateway
# housekeeping loop prunes hourly and a once-per-process sweep covers CLI-only
# installs.
TERMINAL_TEMP_MAX_AGE_HOURS = 72

_terminal_temp_prune_lock = threading.Lock()
_terminal_temp_pruned_once = False

# Background-process artifacts come in triplets (hermes_bg_<id>.log/.pid/.exit).
# A live server's .pid never changes mtime while its .log does, so age is judged
# per GROUP (newest mtime sharing a stem) to avoid yanking pid/exit files from
# under a still-running background session.
_BG_GROUP_RE = re.compile(r"^(hermes_bg_[A-Za-z0-9_-]+)\.(log|pid|exit)$")


def _default_terminal_temp_dir() -> "Path | None":
    """Return HERMES_HOME/cache/terminal, or None if unresolvable."""
    try:
        from hermes_constants import get_hermes_home
        return get_hermes_home() / "cache" / "terminal"
    except Exception:
        return None


def cleanup_terminal_temp_cache(
    max_age_hours: int = TERMINAL_TEMP_MAX_AGE_HOURS,
) -> int:
    """Delete session temp artifacts older than *max_age_hours*; return count
    (``cleanup_*_cache`` contract). Only the managed default dir is pruned —
    never a user-pointed ``terminal.temp_dir`` we don't own."""
    root = _default_terminal_temp_dir()
    if root is None:
        return 0
    cutoff = time.time() - (max_age_hours * 3600)
    try:
        entries = list(root.iterdir())
    except OSError:
        return 0

    mtimes: dict[Path, float] = {}
    group_newest: dict[str, float] = {}
    for f in entries:
        try:
            mtimes[f] = mt = f.stat().st_mtime
        except OSError:
            continue
        m = _BG_GROUP_RE.match(f.name)
        if m:
            group_newest[m.group(1)] = max(group_newest.get(m.group(1), 0.0), mt)

    removed = 0
    for f, mt in mtimes.items():
        m = _BG_GROUP_RE.match(f.name)
        if (group_newest[m.group(1)] if m else mt) >= cutoff:
            continue
        try:
            if f.is_dir():
                shutil.rmtree(f, ignore_errors=True)
            else:
                f.unlink()
            removed += 1
        except OSError:
            continue
    return removed


def _prune_terminal_temp_once() -> None:
    """Best-effort prune, at most once per process (CLI-only installs)."""
    global _terminal_temp_pruned_once
    with _terminal_temp_prune_lock:
        if _terminal_temp_pruned_once:
            return
        _terminal_temp_pruned_once = True
    try:
        cleanup_terminal_temp_cache()
    except Exception as exc:
        logger.debug("Terminal temp prune failed: %s", exc)


# --- Windows / MSYS path translation ---


def _msys_to_windows_path(cwd: str) -> str:
    """Translate a Git Bash / MSYS path (``/c/Users/x``, ``/cygdrive/c/..``,
    ``/mnt/c/..``) to native ``C:\\Users\\x`` so ``isdir``/``Popen(cwd=)`` find
    it. No-op off Windows, for empty input and for multi-segment POSIX paths
    like ``/home/x``; idempotent on native paths."""
    if not _IS_WINDOWS or not cwd:
        return cwd
    m = re.match(r'^/(?:(?:cygdrive|mnt)/)?([a-zA-Z])(/.*)?$', cwd)
    if not m:
        return cwd
    drive = m.group(1).upper()
    tail = (m.group(2) or "").replace('/', '\\')
    return f"{drive}:{tail or chr(92)}"  # chr(92) = backslash, avoid raw-string escape


def _resolve_local_initial_cwd(cwd: str) -> str:
    """Resolve the initial cwd to an absolute host path. A relative
    ``TERMINAL_CWD`` like ``hermes-agent`` naming the launch directory would
    otherwise make the wrapper ``cd hermes-agent`` *inside* the project; anchor
    it once so ``Popen(cwd=)`` and the in-shell ``cd`` agree."""
    expanded = os.path.expanduser(cwd) if cwd else os.getcwd()
    if _IS_WINDOWS:
        expanded = _msys_to_windows_path(expanded)
        # ntpath explicitly: with _IS_WINDOWS patched on a POSIX host,
        # os.path.isabs would reject ``C:\Users\x`` and mangle it below.
        if ntpath.isabs(expanded):
            return expanded
    if os.path.isabs(expanded):
        return expanded

    candidate = os.path.abspath(expanded)
    current = os.getcwd()

    # Relative name matching the tail of the current dir: use the current dir.
    if not os.path.isdir(candidate):
        wanted, have = Path(expanded).parts, Path(current).parts
        if wanted and len(wanted) <= len(have) and have[-len(wanted):] == wanted:
            return current
    return candidate


def _windows_to_msys_path(cwd: str) -> str:
    """Translate native ``C:\\Users\\x`` to Git Bash form ``/c/Users/x`` so
    ``builtin cd`` resolves it. No-op off Windows / for non-drive paths."""
    if not _IS_WINDOWS or not cwd:
        return cwd
    m = re.match(r'^([a-zA-Z]):[\\/]*(.*)$', cwd)
    if not m:
        return cwd
    drive = m.group(1).lower()
    tail = (m.group(2) or "").replace('\\', '/').lstrip('/')
    return f"/{drive}/{tail}" if tail else f"/{drive}/"


def _bash_safe_path(path: str) -> str:
    """Return *path* in a form safe to embed in a Git Bash script: native
    ``C:\\Users\\x`` / ``C:/Users/x`` become ``/c/Users/x`` (MSYS argument
    conversion mangles ``C:/`` forms), and leftover backslashes are normalized
    so bash does not eat ``\\U``. No-op off Windows and for empty input."""
    if not _IS_WINDOWS or not path:
        return path
    return _windows_to_msys_path(path).replace("\\", "/")


def _quote_bash_path(path: str) -> str:
    """Quote *path* for safe interpolation into a Git Bash script on Windows."""
    import shlex

    return shlex.quote(_bash_safe_path(path))


def _cwd_usable(path: str) -> bool:
    """True when *path* is a directory this process can actually chdir into.
    ``isdir`` alone is not enough: stat() on ``/root`` succeeds for a non-root
    user but ``Popen(cwd='/root')`` dies with PermissionError (a root-launched
    CLI leaking ``/root`` into a non-root gateway's shared state)."""
    return os.path.isdir(path) and os.access(path, os.X_OK)


def _resolve_safe_cwd(cwd: str) -> str:
    """Return ``cwd`` if enterable, else the nearest usable ancestor, else
    ``tempfile.gettempdir()``. MSYS paths are normalized first on Windows so a
    valid ``pwd -P`` result is not rejected as missing. Lets ``_run_bash``
    recover from a deleted/inaccessible cwd instead of ``Popen`` raising before
    bash starts and wedging every subsequent terminal call."""
    cwd = _msys_to_windows_path(cwd) if _IS_WINDOWS else cwd
    if cwd and _cwd_usable(cwd):
        return cwd
    if cwd and os.path.isdir(cwd):
        logger.warning(
            "Configured terminal cwd %r exists but is not accessible to "
            "this user (uid=%s) — falling back to the nearest usable "
            "directory. If this is a gateway/cron process, check for "
            "root-owned paths leaking into terminal.cwd / TERMINAL_CWD "
            "(#65583).",
            cwd, getattr(os, "getuid", lambda: "?")(),
        )
    parent = os.path.dirname(cwd) if cwd else ""
    while parent:
        if _cwd_usable(parent):
            return parent
        next_parent = os.path.dirname(parent)
        if next_parent == parent:
            break  # filesystem root itself is unusable
        parent = next_parent
    return tempfile.gettempdir()


# --- Child-process environment construction ---


def _inject_context_hermes_home(env: dict) -> None:
    """Bridge the context-local Hermes home override into subprocess env."""
    try:
        from hermes_constants import get_hermes_home_override

        value = get_hermes_home_override()
        if value:
            env["HERMES_HOME"] = value
    except Exception:
        pass


def _apply_profile_home(env: dict) -> None:
    """HERMES_HOME override bridge + the subprocess HOME contract."""
    _inject_context_hermes_home(env)
    from hermes_constants import apply_subprocess_home_env
    apply_subprocess_home_env(env)


def _inject_session_context_env(env: dict) -> None:
    """Bridge gateway session ContextVars (HERMES_SESSION_*) into a child env.

    Cross-session leak guard: the vars also have a last-writer-wins ``os.environ``
    mirror that, under a concurrent multi-session host, may belong to another
    turn. Once the session-context system is engaged, ContextVars are
    authoritative: a bound value (incl. "") wins and an _UNSET var is STRIPPED
    rather than inherited. A CLI that never engaged it keeps the inherited value.
    """
    try:
        from gateway.session_context import _UNSET, _VAR_MAP, session_context_engaged
    except Exception:
        return

    _engaged = session_context_engaged()
    for var_name, var in _VAR_MAP.items():
        value = var.get()
        if value is not _UNSET:
            env[var_name] = "" if value is None else str(value)
        elif _engaged:
            env.pop(var_name, None)


def _scrub_delegated_child_kanban_env(env: dict[str, str]) -> dict[str, str]:
    """Strip dispatcher-owned Kanban env from delegate_task child subprocesses."""
    try:
        from agent.delegation_context import is_delegated_child_process_context, scrub_kanban_env

        if is_delegated_child_process_context():
            return scrub_kanban_env(env)
    except Exception:
        pass
    return env


def _passthrough_hooks():
    """Return ``(is_passthrough, resolve_passthrough_value)`` from the
    env_passthrough skill registry, or inert fallbacks."""
    try:
        from tools.env_passthrough import is_env_passthrough, resolve_passthrough_value

        return is_env_passthrough, resolve_passthrough_value
    except Exception:
        return (lambda _: False), (lambda _name, fallback: fallback)


def _filter_secret_env(
    items: Mapping[str, str],
    out: dict,
    *,
    unwrap_force: bool,
    plugin_strip: frozenset = frozenset(),
) -> None:
    """Copy *items* into *out*, dropping Hermes-managed secrets.

    ``_HERMES_FORCE_<NAME>`` unwraps to ``NAME`` when ``unwrap_force`` (caller
    extras / terminal env), else is dropped. Blocklisted names survive only via
    env_passthrough registration or as context-entitled first-party ``BUZZ_*``
    vars; the latter are used directly, never scope-resolved (UnscopedSecretError
    under multiplex) — only passthrough names resolve through the secret scope.
    """
    is_passthrough, resolve_passthrough_value = _passthrough_hooks()
    for key, value in items.items():
        if key.startswith(_HERMES_PROVIDER_ENV_FORCE_PREFIX):
            if not unwrap_force:
                continue
            real_key = key[len(_HERMES_PROVIDER_ENV_FORCE_PREFIX):]
            if _is_hermes_internal_secret(real_key):
                continue
            out[real_key] = value
            continue
        if _is_hermes_internal_secret(key) or key in plugin_strip:
            continue
        first_party = _is_terminal_first_party_env(key)
        passthrough = is_passthrough(key)
        if key in _HERMES_PROVIDER_ENV_BLOCKLIST and not (passthrough or first_party):
            continue
        resolved = value
        if passthrough and not first_party:
            resolved = resolve_passthrough_value(key, value)
        if resolved is not None:
            out[key] = resolved


def _finalize_child_env(env: dict) -> dict:
    """Guards shared by every spawn surface: profile-home propagation,
    session-context bridging, Hermes-owned PYTHONPATH + venv-marker strip, MSYS
    defaults, delegate_task Kanban scrub. Returns the (possibly new) dict."""
    _apply_profile_home(env)
    _inject_session_context_env(env)
    _strip_hermes_owned_pythonpath_and_runtime_markers(env)
    _apply_windows_msys_bash_env_defaults(env)
    return _scrub_delegated_child_kanban_env(env)


def _sanitize_subprocess_env(base_env: dict | None, extra_env: dict | None = None) -> dict:
    """Filter Hermes-managed secrets from a subprocess environment.

    Background/PTY spawn path (``process_registry.spawn_local``), search workers,
    the computer-use driver, and user-script runners build their env here.
    """
    plugin_strip = _plugin_terminal_env_strip_keys()
    sanitized: dict[str, str] = {}
    _filter_secret_env(base_env or {}, sanitized, unwrap_force=False, plugin_strip=plugin_strip)
    _filter_secret_env(extra_env or {}, sanitized, unwrap_force=True, plugin_strip=plugin_strip)

    # Keep bare ``hermes`` resolvable for children even when the gateway was
    # launched by a service manager or cron without the console-script dir on
    # PATH (cron scripts use this sanitizer directly).
    path_key = _path_env_key(sanitized)
    if path_key is not None:
        sanitized[path_key] = _prepend_hermes_bin_dir(sanitized.get(path_key, ""))

    return _finalize_child_env(sanitized)


def hermes_subprocess_env(*, inherit_credentials: bool = False) -> dict[str, str]:
    """Sanitized env for the **non-terminal** spawn surface (browser, ACP/CLI
    executors, computer-use driver, TUI Node host). Tier 1 (``_ALWAYS_STRIP_KEYS``,
    plugin keys, force-prefixed hints, dynamic internal secrets) is always
    removed; Tier 2 (the provider/tool blocklist) unless ``inherit_credentials``
    — pass that **only** for children that legitimately need LLM credentials
    (user-blessed claude/codex/gemini CLI, TUI Node host); it is grep-able for
    audit. Terminal/execute_code use the skill-aware ``_sanitize_subprocess_env``.
    """
    env = os.environ.copy()

    for key in _ALWAYS_STRIP_KEYS:
        env.pop(key, None)
    for key in _plugin_terminal_env_strip_keys():
        env.pop(key, None)
    for key in list(env):
        if key.startswith(_HERMES_PROVIDER_ENV_FORCE_PREFIX) or _is_hermes_internal_secret(key):
            env.pop(key, None)

    if not inherit_credentials:
        for key in _HERMES_PROVIDER_ENV_BLOCKLIST:
            env.pop(key, None)

    # Windows UTF-8 safety for spawned processes.
    env.setdefault("PYTHONUTF8", "1")

    return _finalize_child_env(env)


def build_subprocess_env(
    base: "Mapping[str, str] | None" = None,
    *,
    inherit_profile_home: bool = True,
    scrub_secrets: bool = True,
    extra: "Mapping[str, str] | None" = None,
) -> dict[str, str]:
    """Single factory for child-process environments, so profile-home
    propagation and the secret-scrub policy have one owner.

    ``base=None`` snapshots ``os.environ``. ``scrub_secrets=True`` delegates to
    :func:`_sanitize_subprocess_env` (``extra`` -> ``extra_env``; profile home is
    inherent, ``inherit_profile_home`` ignored). ``scrub_secrets=False`` keeps the
    base byte-for-byte (git credential flows, ``bws``/``op``); then
    ``inherit_profile_home`` bridges HERMES_HOME + HOME and ``extra`` is applied
    last so caller overrides win.
    """
    if scrub_secrets:
        return _sanitize_subprocess_env(
            dict(base) if base is not None else os.environ.copy(),
            dict(extra) if extra else None,
        )

    env: dict[str, str] = dict(base) if base is not None else os.environ.copy()
    if inherit_profile_home:
        _apply_profile_home(env)
    if extra:
        env.update(extra)
    return env


# --- Shell discovery ---


def _windows_bash_candidates(custom: "str | None") -> list[str]:
    """Ordered bash.exe candidates on Windows: HERMES_GIT_BASH_PATH, our
    portable Git under %LOCALAPPDATA%\\hermes\\git (PortableGit ``bin`` and
    MinGit ``usr\\bin`` layouts), known Git-for-Windows dirs, then PATH last —
    ``shutil.which`` may return WSL's bash, which fails silently on Windows paths."""
    candidates: list[str] = []

    def add(candidate: str) -> None:
        if candidate and os.path.isfile(candidate) and candidate not in candidates:
            candidates.append(candidate)

    add(custom or "")

    local_appdata = os.environ.get("LOCALAPPDATA", "")
    if local_appdata:
        portable = os.path.join(local_appdata, "hermes", "git")
        add(os.path.join(portable, "bin", "bash.exe"))
        add(os.path.join(portable, "usr", "bin", "bash.exe"))

    add(os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"), "Git", "bin", "bash.exe"))
    add(os.path.join(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"), "Git", "bin", "bash.exe"))
    if local_appdata:
        add(os.path.join(local_appdata, "Programs", "Git", "bin", "bash.exe"))

    found = shutil.which("bash")
    if found and found not in candidates:
        candidates.append(found)
    return candidates


def _find_bash() -> str:
    """Find bash for command execution."""
    if not _IS_WINDOWS:
        return (
            shutil.which("bash")
            or ("/usr/bin/bash" if os.path.isfile("/usr/bin/bash") else None)
            or ("/bin/bash" if os.path.isfile("/bin/bash") else None)
            or os.environ.get("SHELL")
            or "/bin/sh"
        )

    custom = os.environ.get("HERMES_GIT_BASH_PATH")
    candidates = _windows_bash_candidates(custom)

    # First candidate that can actually start wins: a stale HERMES_GIT_BASH_PATH
    # pointing at a broken install must not beat a healthy portable Git.
    for candidate in candidates:
        if _bash_starts(candidate):
            if candidate != custom and custom and os.path.isfile(custom):
                logger.warning(
                    "HERMES_GIT_BASH_PATH=%s fails to start; using %s instead", custom, candidate,
                )
            return candidate

    if candidates:
        probe_details = "\n".join(
            detail for c in candidates if (detail := _bash_probe_details_cache.get(c))
        )
        if _mandatory_aslr_enabled() is True or _looks_like_msys_spawn_failure(probe_details):
            raise RuntimeError(_git_bash_aslr_help(candidates[0], probe_details))
        # Unknown failure class: return the first path so the caller sees the
        # real bash error instead of a less useful "not found".
        return candidates[0]

    raise RuntimeError(
        "Git Bash not found. Hermes Agent requires Git for Windows on Windows.\n"
        "Install it from: https://git-scm.com/download/win\n"
        "Or set HERMES_GIT_BASH_PATH to your bash.exe location."
    )


_git_bash_bin_dirs_cache: "list[str] | None" = None


def _git_bash_bin_dirs() -> list[str]:
    """Git Bash's coreutils dirs in ``/etc/profile`` order (mingw first so
    coreutils beat System32 lookalikes); ``[]`` off Windows. A non-login
    ``bash -c`` (fallback when ``bash -l`` is broken) never sources
    ``/etc/profile``, so without these ``cat``/``mktemp``/``mv`` are missing:
    ``write_file`` fails with an empty error and commands exit 127."""
    global _git_bash_bin_dirs_cache
    if _git_bash_bin_dirs_cache is None:
        _git_bash_bin_dirs_cache = _compute_git_bash_bin_dirs() if _IS_WINDOWS else []
    return _git_bash_bin_dirs_cache


def _compute_git_bash_bin_dirs() -> list[str]:
    try:
        bash = _find_bash()
    except Exception:
        return []
    bin_dir = os.path.dirname(bash)          # <root>\bin  or  <root>\usr\bin (MinGit)
    parent = os.path.dirname(bin_dir)
    root = os.path.dirname(parent) if os.path.basename(parent).lower() == "usr" else parent
    dirs: list[str] = []
    for sub in (("mingw64", "bin"), ("mingw32", "bin"), ("usr", "local", "bin"), ("usr", "bin"), ("bin",)):
        candidate = os.path.join(root, *sub)
        if os.path.isdir(candidate) and candidate not in dirs:
            dirs.append(candidate)
    return dirs


def _prepend_missing_path_entries(existing_path: str, dirs: list[str]) -> str:
    """Prepend *dirs* missing from *existing_path* (``os.pathsep``); an
    already-listed dir keeps its position and the input is returned unchanged
    when nothing is missing."""
    if not dirs:
        return existing_path
    sep = os.pathsep
    entries = [e for e in existing_path.split(sep) if e] if existing_path else []
    missing = [d for d in dirs if d not in entries]
    if not missing:
        return existing_path
    return sep.join([*missing, *entries])


def _prepend_git_bash_dirs(existing_path: str) -> str:
    """Prepend Git Bash's binary dirs if missing (no-op off Windows), so the
    non-login ``bash -c`` fallback can find coreutils when no login snapshot
    re-exports the full PATH inside the shell."""
    return _prepend_missing_path_entries(existing_path, _git_bash_bin_dirs())


# POSIX-sh-family shells that understand spawn_local's ``[shell, "-lic",
# "set +m; …"]`` invocation. fish, csh/tcsh, nushell, elvish, xonsh would error
# on that syntax, so _find_shell falls back to bash for them.
_SPAWN_COMPATIBLE_SHELLS = frozenset({"bash", "zsh", "sh", "dash", "ksh", "mksh"})


def _find_shell() -> str:
    """User's login shell for background spawning: ``$SHELL`` on POSIX when it
    is an executable sh-family shell, else ``_find_bash``. macOS's system bash
    3.2 under ``-l`` with stdin ``/dev/null`` sources ``~/.bash_profile``, which
    often ``exec /bin/zsh -l`` and drops ``-c`` — the command silently never
    runs. Non-allowlisted shells would trade that for a parse error."""
    if not _IS_WINDOWS:
        user_shell = os.environ.get("SHELL")
        if (
            user_shell
            and os.path.isfile(user_shell)
            and os.access(user_shell, os.X_OK)
            and Path(user_shell).name in _SPAWN_COMPATIBLE_SHELLS
        ):
            return user_shell
    return _find_bash()


# --- PATH completion for the terminal subshell ---

# Standard PATH entries for environments with minimal PATH.
_SANE_PATH = (
    "/opt/homebrew/bin:/opt/homebrew/sbin:"
    "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
)

# Cached directory containing the ``hermes`` console-script.
# ``_SENTINEL`` distinguishes "not resolved yet" from a resolved ``None``.
_SENTINEL = object()
_HERMES_BIN_DIR: "str | None | object" = _SENTINEL


def _resolve_hermes_bin_dir() -> str | None:
    """Directory holding the ``hermes`` console-script, or None (cached).
    Launched by systemd/cron/a desktop launcher, the gateway's PATH lacks the
    install dir (``~/.local/bin``, venv ``bin``, pipx, nix) and bare ``hermes``
    exits 127. Order: ``which``; absolute ``sys.argv[0]`` naming a real hermes
    executable; ``sys.executable``'s dir if it holds the shim."""
    global _HERMES_BIN_DIR
    if _HERMES_BIN_DIR is not _SENTINEL:
        return _HERMES_BIN_DIR  # type: ignore[return-value]

    candidate: str | None = None

    which = shutil.which("hermes")
    if which:
        candidate = os.path.dirname(which)

    if candidate is None:
        argv0 = sys.argv[0] if sys.argv else ""
        base = os.path.basename(argv0).lower()
        if (
            os.path.isabs(argv0)
            and (base == "hermes" or base.startswith("hermes."))
            and os.path.isfile(argv0)
        ):
            candidate = os.path.dirname(argv0)

    if candidate is None:
        exe_dir = os.path.dirname(sys.executable) if sys.executable else ""
        shim = "hermes.exe" if _IS_WINDOWS else "hermes"
        if exe_dir and os.path.isfile(os.path.join(exe_dir, shim)):
            candidate = exe_dir

    if candidate and not os.path.isdir(candidate):
        candidate = None

    _HERMES_BIN_DIR = candidate
    return candidate


def _prepend_hermes_bin_dir(existing_path: str) -> str:
    """Prepend the hermes install dir to ``existing_path`` if it's missing
    (unchanged when already present or unresolvable)."""
    bin_dir = _resolve_hermes_bin_dir()
    return _prepend_missing_path_entries(existing_path, [bin_dir] if bin_dir else [])


def _managed_runtime_path_entries() -> list[str]:
    """Existing Hermes-managed runtime dirs: ``$HERMES_HOME/node`` (+``/bin``)
    and ``$HERMES_HOME/bin`` (managed ``uv``). Per call, not cached: home is
    profile-scoped and a managed tree can appear mid-process."""
    try:
        from hermes_constants import get_hermes_home, iter_hermes_node_dirs

        candidates = [*iter_hermes_node_dirs(), get_hermes_home() / "bin"]
        return [str(d) for d in candidates if d.is_dir()]
    except Exception:
        return []


def _append_missing_sane_path_entries(existing_path: str) -> str:
    """Normalised POSIX PATH with missing sane entries appended: empty entries
    dropped (shells read them as cwd), duplicates collapsed (first wins), then
    missing ``_SANE_PATH`` / managed-runtime dirs appended so user entries keep
    precedence. Windows is a no-op passthrough (native ``;`` PATH untouched)."""
    if _IS_WINDOWS:
        return existing_path

    sane_entries = [entry for entry in _SANE_PATH.split(":") if entry]
    sane_entries.extend(
        entry for entry in _managed_runtime_path_entries() if entry not in sane_entries
    )
    if not existing_path:
        return ":".join(sane_entries)
    # dict preserves first-occurrence order; empty entries dropped.
    ordered = dict.fromkeys(entry for entry in existing_path.split(":") if entry)
    ordered.update(dict.fromkeys(sane_entries))
    return ":".join(ordered)


def _apply_windows_msys_bash_env_defaults(env: dict) -> None:
    """Disable MSYS argument path conversion (``/FO`` -> ``C:/.../git/FO``
    breaks tasklist/schtasks/wmic/``cmd /c``). Git for Windows honors
    ``MSYS_NO_PATHCONV``; MSYS2/Cygwin bash honor ``MSYS2_ARG_CONV_EXCL`` — set
    both. Users can override in their env."""
    if not _IS_WINDOWS:
        return
    env.setdefault("MSYS_NO_PATHCONV", "1")
    env.setdefault("MSYS2_ARG_CONV_EXCL", "*")


def _path_env_key(run_env: dict) -> str | None:
    """Return the PATH env key to update without altering Windows casing
    (``Path`` vs ``PATH``); None when a Windows env has no PATH key at all."""
    if not _IS_WINDOWS:
        return "PATH"
    for key in run_env:
        if key.upper() == "PATH":
            return key
    return None


def _make_run_env(env: dict) -> dict:
    """Build a run environment with a sane PATH and provider-var stripping."""
    run_env: dict = {}
    _filter_secret_env(dict(os.environ | env), run_env, unwrap_force=True)
    path_key = _path_env_key(run_env)
    if path_key is not None:
        new_path = _append_missing_sane_path_entries(run_env.get(path_key, ""))
        new_path = _prepend_git_bash_dirs(new_path)
        run_env[path_key] = _prepend_hermes_bin_dir(new_path)
    return _finalize_child_env(run_env)


# --- Hermes venv / repo-root detection (module-level, computed once) ---
# Owned here; read lazily by tools.environments.local_pythonpath.

#: The Hermes repository root (three levels up from this file). The Electron
#: app prepends it to PYTHONPATH so the backend can ``import tools``; other
#: subprocesses don't need it and it can shadow local packages.
_hermes_repo_root: Path = Path(__file__).resolve().parents[2]

#: Alternate repo-root spellings Hermes launchers may emit. ``resolve()``
#: canonicalizes junctions, but the Windows gateway launcher renders
#: Hermes-owned paths under the configured HERMES_HOME spelling (possibly a
#: junction to another drive); the unresolved ``Path(__file__)`` keeps it.
_hermes_repo_root_aliases: tuple[Path, ...] = _build_hermes_repo_root_aliases(
    _hermes_repo_root,
    Path(__file__).absolute().parents[2],
    get_process_hermes_home(),
)

#: Whether the interpreter runs inside a venv (``sys.real_prefix`` is the old
#: virtualenv<20 marker).
_in_venv: bool = (
    getattr(sys, "base_prefix", sys.prefix) != sys.prefix
    or hasattr(sys, "real_prefix")
)

#: Lazily-cached site-packages dirs of the running interpreter's own venv.
_hermes_site_packages: list[Path] | None = None


# --- Login-shell init files ---


def _read_terminal_shell_init_config() -> tuple[list[str], bool]:
    """Return (shell_init_files, auto_source_bashrc) from config.yaml.
    Best-effort: defaults on any failure so terminal execution never breaks."""
    try:
        from hermes_cli.config import load_config

        cfg = load_config() or {}
        terminal_cfg = cfg.get("terminal") or {}
        files = terminal_cfg.get("shell_init_files") or []
        if not isinstance(files, list):
            files = []
        auto_bashrc = bool(terminal_cfg.get("auto_source_bashrc", True))
        return [str(f) for f in files if f], auto_bashrc
    except Exception:
        return [], True


def _resolve_shell_init_files() -> list[str]:
    """Files to source before the login-shell snapshot (``~``/``${VAR}``
    expanded, missing dropped). ``auto_source_bashrc`` applies only without an
    explicit list: ~/.profile and ~/.bash_profile first (no interactivity guard;
    where n/nvm/asdf/pyenv add PATH), ~/.bashrc last (Debian's default returns
    early when non-interactive, but guard-less bashrcs keep working)."""
    explicit, auto_bashrc = _read_terminal_shell_init_config()

    candidates: list[str] = []
    if explicit:
        candidates.extend(explicit)
    elif auto_bashrc and not _IS_WINDOWS:
        candidates.extend(["~/.profile", "~/.bash_profile", "~/.bashrc"])

    resolved: list[str] = []
    for raw in candidates:
        try:
            path = os.path.expandvars(os.path.expanduser(raw))
        except Exception:
            continue
        if path and os.path.isfile(path):
            resolved.append(path)
    return resolved


def _prepend_shell_init(cmd_string: str, files: list[str]) -> str:
    """Prepend guarded, silent ``source <file>`` lines to a bash script:
    ``set +e`` keeps going on errors, ``2>/dev/null`` hides noisy prompts,
    ``|| true`` neutralises the exit status."""
    if not files:
        return cmd_string

    prelude_parts = ["set +e"]
    for path in files:
        safe = path.replace("'", "'\\''")
        prelude_parts.append(f"[ -r '{safe}' ] && . '{safe}' 2>/dev/null || true")
    prelude = "\n".join(prelude_parts) + "\n"
    return prelude + cmd_string


# --- Process-group teardown (POSIX) ---


def _group_alive(pgid: int) -> bool:
    """POSIX-only probe; callers are behind the _IS_WINDOWS gate."""
    try:
        os.killpg(pgid, 0)  # windows-footgun: ok — POSIX process-group alive probe
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, even if we cannot signal it


def _wait_for_group_exit(proc, pgid: int, timeout: float) -> bool:
    """Wait until the process group is gone, reaping the wrapper as we go
    (a dead but unreaped group leader still makes ``killpg(pgid, 0)`` succeed)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            proc.poll()
        except Exception:
            pass
        if not _group_alive(pgid):
            return True
        time.sleep(0.05)
    try:
        proc.poll()
    except Exception:
        pass
    return not _group_alive(pgid)


def _snapshot_descendants(proc) -> list:
    """psutil children snapshot; empty on any failure (must never break the kill)."""
    try:
        import psutil

        return psutil.Process(proc.pid).children(recursive=True)
    except Exception:
        return []


def _sweep_escaped_descendants(descendants: list, pgid: int) -> None:
    """SIGKILL snapshotted survivors that escaped the process group via
    ``setsid`` — after TERM→KILL so in-group members keep their grace; psutil's
    identity-aware Process skips recycled PIDs. POSIX-only (see _IS_WINDOWS gate)."""
    for child in descendants:
        try:
            if not child.is_running():
                continue
            try:
                if os.getpgid(child.pid) == pgid:
                    continue  # group-kill already covers it
            except (ProcessLookupError, PermissionError, OSError):
                pass
            child.kill()
        except Exception:
            continue


def _kill_process_group_posix(proc) -> None:
    """TERM the group, wait, KILL, then sweep setsid escapees. POSIX-only
    (_IS_WINDOWS handled by the caller). Descendants are snapshotted BEFORE the
    first signal — once the wrapper dies they reparent to init — and we wait on
    the group, not the wrapper, which can exit before grandchildren under load."""
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        pgid = getattr(proc, "_hermes_pgid", None)
        if pgid is None:
            raise

    descendants = _snapshot_descendants(proc)

    try:
        os.killpg(pgid, signal.SIGTERM)  # windows-footgun: ok — POSIX only (see _IS_WINDOWS gate in caller)
    except ProcessLookupError:
        _sweep_escaped_descendants(descendants, pgid)
        return

    if _wait_for_group_exit(proc, pgid, 1.0):
        _sweep_escaped_descendants(descendants, pgid)
        return

    try:
        os.killpg(pgid, signal.SIGKILL)  # windows-footgun: ok — POSIX only (see _IS_WINDOWS gate in caller)
    except ProcessLookupError:
        _sweep_escaped_descendants(descendants, pgid)
        return
    _wait_for_group_exit(proc, pgid, 2.0)
    try:
        proc.wait(timeout=0.2)
    except (subprocess.TimeoutExpired, OSError):
        pass
    _sweep_escaped_descendants(descendants, pgid)


def _kill_process_windows(proc) -> None:
    try:
        from gateway.status import get_process_start_time, terminate_pid

        terminate_pid(
            proc.pid,
            force=True,
            expected_start_time=get_process_start_time(proc.pid),
        )
    except Exception:
        proc.kill()
    try:
        proc.wait(timeout=2.0)
    except (subprocess.TimeoutExpired, OSError):
        pass


class LocalEnvironment(BaseEnvironment):
    """Run commands directly on the host machine.

    Spawn-per-call: every execute() spawns a fresh bash process.
    Session snapshot preserves env vars across calls.
    CWD persists via file-based read after each command.
    """

    _profile_scoped_passthrough = True

    # Commands run on the Hermes host itself — controller-side platform
    # behavior (macOS TCC pruning, etc.) legitimately applies here.
    is_local = True

    def _additional_profile_scoped_passthrough_names(self) -> tuple[str, ...]:
        """First-party ``BUZZ_*`` names present in the env, excluded from the
        shared session snapshot. env_passthrough can never list them (it refuses
        blocklisted names), so under a multiplexed gateway profile A's
        BUZZ_PRIVATE_KEY would land in the snapshot and be sourced by profile B.
        Prefix-only and monotonic on purpose: conservative even when the
        context-gated carve-out is inactive."""
        merged = dict(os.environ | self.env)
        return tuple(
            sorted(
                name
                for name in merged
                if isinstance(name, str) and _matches_terminal_first_party_prefix(name)
            )
        )

    def __init__(self, cwd: str = "", timeout: int = 60, env: dict = None):
        cwd = _resolve_local_initial_cwd(cwd)
        super().__init__(cwd=cwd, timeout=timeout, env=env)
        self.init_session()

    def get_temp_dir(self) -> str:
        """Shell-safe writable temp dir. Precedence: ``TERMINAL_TEMP_DIR``,
        POSIX TMPDIR/TMP/TEMP (Termux has no /tmp), ``HERMES_HOME/cache/terminal``,
        /tmp, POSIX ``tempfile.gettempdir()``; backend env before process env so
        terminal.env overrides work. The default is real storage because tmpfs
        /tmp fills under Hermes load (pruned by ``cleanup_terminal_temp_cache``).
        Windows: ``%TEMP%`` often has spaces that break unquoted bash, so always
        the HERMES_HOME cache dir with forward slashes (valid in bash and Python).
        """
        if _IS_WINDOWS:
            # Forward slashes: one string valid in bash interpolation AND Python open().
            try:
                from hermes_constants import get_hermes_home
                cache_dir = get_hermes_home() / "cache" / "terminal"
            except Exception:
                cache_dir = Path(tempfile.gettempdir()) / "hermes_terminal"
            cache_dir.mkdir(parents=True, exist_ok=True)
            _prune_terminal_temp_once()
            return str(cache_dir).replace("\\", "/")

        def _posix(p: str) -> str:
            return p.rstrip("/") or "/"

        configured = self.env.get("TERMINAL_TEMP_DIR") or os.environ.get("TERMINAL_TEMP_DIR")
        if configured and configured.startswith("/") and os.path.isdir(configured):
            return _posix(configured)

        for env_var in ("TMPDIR", "TMP", "TEMP"):
            candidate = self.env.get(env_var) or os.environ.get(env_var)
            if candidate and candidate.startswith("/"):
                return _posix(candidate)

        try:
            from hermes_constants import get_hermes_home
            cache_dir = get_hermes_home() / "cache" / "terminal"
            cache_dir.mkdir(parents=True, exist_ok=True)
            resolved = str(cache_dir)
            if resolved.startswith("/") and os.access(resolved, os.W_OK | os.X_OK):
                _prune_terminal_temp_once()
                return _posix(resolved)
        except Exception:
            pass

        if os.path.isdir("/tmp") and os.access("/tmp", os.W_OK | os.X_OK):
            return "/tmp"

        candidate = tempfile.gettempdir()
        return _posix(candidate) if candidate.startswith("/") else "/tmp"

    @staticmethod
    def _quote_cwd_for_cd(cwd: str) -> str:
        """Use native paths for Python, but Git Bash-friendly paths for cd."""
        return BaseEnvironment._quote_cwd_for_cd(_windows_to_msys_path(cwd))

    def _quote_shell_path(self, path: str) -> str:
        """Rewrite native/mixed Windows paths before quoting for Git Bash."""
        return _quote_bash_path(path)

    def _recover_cwd(self) -> None:
        """Swap ``self.cwd`` for a usable directory if it vanished or is
        inaccessible (e.g. a previous command ``rm -rf``'d its own cwd) —
        otherwise Popen raises before bash starts and every subsequent call
        fails. A benign MSYS→Windows normalization is not warned about."""
        safe_cwd = _resolve_safe_cwd(self.cwd)
        if safe_cwd == self.cwd:
            return
        normalized = _msys_to_windows_path(self.cwd) if _IS_WINDOWS else self.cwd
        if safe_cwd != normalized:
            logger.warning(
                "LocalEnvironment cwd %r is missing on disk; "
                "falling back to %r so terminal commands keep working.",
                self.cwd,
                safe_cwd,
            )
        self.cwd = safe_cwd

    def _run_bash(self, cmd_string: str, *, login: bool = False,
                  timeout: int = 120,
                  stdin_data: str | None = None) -> subprocess.Popen:
        bash = _find_bash()
        # Login invocations (init_session's env snapshot) source the user's rc /
        # custom init files so nvm/asdf/pyenv land on PATH in the snapshot.
        if login:
            init_files = _resolve_shell_init_files()
            if init_files:
                cmd_string = _prepend_shell_init(cmd_string, init_files)
        args = [bash, "-l", "-c", cmd_string] if login else [bash, "-c", cmd_string]
        run_env = _make_run_env(self.env)

        self._recover_cwd()

        _popen_kwargs = {"creationflags": windows_hide_flags()} if _IS_WINDOWS else {}

        proc = subprocess.Popen(
            args,
            text=True,
            env=run_env,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.PIPE if stdin_data is not None else subprocess.DEVNULL,
            start_new_session=True,
            cwd=self.cwd,
            **_popen_kwargs,
        )
        if not _IS_WINDOWS:
            try:
                proc._hermes_pgid = os.getpgid(proc.pid)
            except ProcessLookupError:
                pass

        if stdin_data is not None:
            _pipe_stdin(proc, stdin_data)

        return proc

    def _kill_process(self, proc):
        """Kill the entire process group (all children)."""
        try:
            if _IS_WINDOWS:
                _kill_process_windows(proc)
            else:
                _kill_process_group_posix(proc)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                proc.kill()
            except Exception:
                pass

    def _update_cwd(self, result: dict):
        """Update cwd from the stdout marker the base wrapper emits (``pwd -P``),
        sharing the remote backends' parser instead of re-reading a temp file."""
        self._extract_cwd_from_output(result)

    def _extract_cwd_from_output(self, result: dict):
        """Base semantics plus: Git Bash ``pwd -P`` emits MSYS form on Windows —
        normalize to native and require the dir to exist, else ``_run_bash`` would
        warn every command. A stale path rolls back to the previous cwd, which
        this command did not observe, so ``cwd_observed`` is dropped."""
        prev_cwd = self.cwd
        super()._extract_cwd_from_output(result)
        if self.cwd != prev_cwd:
            normalized = _msys_to_windows_path(self.cwd) if _IS_WINDOWS else self.cwd
            if normalized and os.path.isdir(normalized):
                self.cwd = normalized
                result["cwd"] = normalized
            else:
                self.cwd = prev_cwd
                result.pop("cwd_observed", None)
                result.pop("cwd", None)

    def cleanup(self):
        """Clean up temp files, including orphaned atomic-write snapshots
        (``snap.tmp.<bashpid>``) a failed/interrupted mv could leave behind."""
        import glob

        try:
            stale = glob.glob(f"{self._snapshot_path}.tmp.*")
        except Exception:
            stale = []
        for f in (self._snapshot_path, self._cwd_file, *stale):
            try:
                os.unlink(f)
            except OSError:
                pass
