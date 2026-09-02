"""MCP server config loading and stdio launch environment: ${VAR}/Cursor-style
interpolation, hidden-whitespace and suspicious-entry filtering, the filtered
subprocess env, command resolution, watchdog wrapping and the shared stderr log."""

import logging
import os
import re
import shutil
import sys
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple
from tools.mcp_tool_common import _env_ref_name, _prepend_path, _core

logger = logging.getLogger("tools.mcp_tool")

_mcp_stderr_log_fh: Optional[Any] = None
_mcp_stderr_log_lock = threading.Lock()


def _get_mcp_stderr_log() -> Any:
    """Shared append-mode handle for MCP subprocess stderr, opened once per
    process. Must expose a real fd (``fileno()``) because asyncio wires the
    child's stderr directly to it. Falls back to ``/dev/null``, then real stderr."""
    global _mcp_stderr_log_fh
    with _mcp_stderr_log_lock:
        if _mcp_stderr_log_fh is not None:
            return _mcp_stderr_log_fh
        try:
            from hermes_constants import get_hermes_home
            log_dir = get_hermes_home() / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            # Line-buffered so output lands promptly; errors="replace" tolerates
            # garbled binary from misbehaving servers.
            fh = open(log_dir / "mcp-stderr.log", "a", encoding="utf-8", errors="replace", buffering=1)
            fh.fileno()  # confirm a real fd before committing
            _mcp_stderr_log_fh = fh
        except Exception as exc:  # pragma: no cover — best-effort fallback
            logger.debug("Failed to open MCP stderr log, using devnull: %s", exc)
            try:
                _mcp_stderr_log_fh = open(os.devnull, "w", encoding="utf-8")
            except Exception:
                _mcp_stderr_log_fh = sys.stderr
        return _mcp_stderr_log_fh


def _write_stderr_log_header(server_name: str) -> None:
    """Write a session marker so operators can find each server's output in the
    shared log without per-line prefixes (which would need a pipe + reader thread)."""
    fh = _core._get_mcp_stderr_log()
    try:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fh.write(f"\n===== [{ts}] starting MCP server '{server_name}' =====\n")
        fh.flush()
    except Exception:
        pass


# Env vars safe to pass to stdio subprocesses (no secrets).
_SAFE_ENV_KEYS = frozenset({"PATH", "HOME", "USER", "LANG", "LC_ALL", "TERM", "SHELL", "TMPDIR"})

# Windows process/location vars needed by launcher-style tools (e.g. Docker
# Desktop's MCP plugin discovery); none carry secrets.
_SAFE_ENV_KEYS_CASE_INSENSITIVE = frozenset({
    "ALLUSERSPROFILE", "APPDATA", "COMMONPROGRAMFILES", "COMMONPROGRAMFILES(X86)",
    "COMMONPROGRAMW6432", "COMPUTERNAME", "COMSPEC", "HOMEDRIVE", "HOMEPATH",
    "LOCALAPPDATA", "NUMBER_OF_PROCESSORS", "OS", "PATHEXT", "PROCESSOR_ARCHITECTURE",
    "PROGRAMDATA", "PROGRAMFILES", "PROGRAMFILES(X86)", "PROGRAMW6432", "PUBLIC",
    "SYSTEMDRIVE", "SYSTEMROOT", "TEMP", "TMP", "USERDOMAIN", "USERNAME",
    "USERPROFILE", "WINDIR",
})

# ${VAR_NAME} interpolation; any non-} chars allowed so MY-VAR / my.var work.
_ENV_VAR_PATTERN = re.compile(r"\$\{([^}]+)\}")


def _workspace_folder() -> str:
    """Absolute workspace root for ``${workspaceFolder}``: the session's
    authoritative root (terminal cwd / task override / $TERMINAL_CWD), else cwd."""
    try:
        from tools.file_tools import _authoritative_workspace_root

        root = _authoritative_workspace_root()
        if root:
            return root
    except Exception:
        pass
    return os.getcwd()


def _workspace_basename() -> str:
    root = _core._workspace_folder()
    return os.path.basename(root.rstrip("/\\")) or root


# Cursor's case-sensitive context vars -> resolver.
_CONTEXT_VAR_RESOLVERS = {
    "userHome": lambda: os.path.expanduser("~"),
    "workspaceFolder": lambda: _core._workspace_folder(),
    "workspaceFolderBasename": _workspace_basename,
    "pathSeparator": lambda: os.sep,
    "/": lambda: os.sep,
}


def _context_var_value(ref: str) -> Optional[str]:
    """Resolve a Cursor context var; None for anything else so it falls through
    to env-var lookup."""
    resolver = _CONTEXT_VAR_RESOLVERS.get(ref)
    return resolver() if resolver else None


def _build_safe_env(user_env: Optional[dict]) -> dict:
    """Filtered env for stdio subprocesses so API keys/tokens don't leak: only
    the safe baseline keys, ``XDG_*``, vars injected by an external secret
    source (users configured that backend precisely so subprocesses can consume
    them), plus the server config's own ``env``."""
    try:
        from hermes_cli.env_loader import get_secret_source
    except Exception:  # pragma: no cover — early bootstrap/import fallback
        get_secret_source = None
    env = {
        key: value
        for key, value in os.environ.items()
        if key in _SAFE_ENV_KEYS
        or key.upper() in _SAFE_ENV_KEYS_CASE_INSENSITIVE
        or key.startswith("XDG_")
        or (get_secret_source is not None and get_secret_source(key))
    }
    if user_env:
        env.update(user_env)
    return env


def _which_with_config_pathext(command: str, path_arg, env: dict):
    """``shutil.which`` retried under the config env's PATHEXT (Windows only):
    ``which(path=...)`` uses the PARENT's PATHEXT, not the config env's."""
    cfg_pathext = next((v for k, v in env.items() if k.upper() == "PATHEXT" and isinstance(v, str) and v.strip()), None)
    if not cfg_pathext or cfg_pathext == os.environ.get("PATHEXT"):
        return None
    saved = os.environ.get("PATHEXT")
    try:
        os.environ["PATHEXT"] = cfg_pathext
        return shutil.which(command, path=path_arg)
    finally:
        if saved is None:
            os.environ.pop("PATHEXT", None)
        else:
            os.environ["PATHEXT"] = saved


def _node_fallback(command: str) -> str:
    """Well-known Node install locations for bare ``npx``/``npm``/``node`` when
    PATH lookup failed; returns *command* unchanged when none is executable."""
    home = os.path.expanduser("~")
    hermes_home = os.path.expanduser(os.getenv("HERMES_HOME", os.path.join(home, ".hermes")))
    candidates = [
        os.path.join(hermes_home, "node", "bin", command),
        os.path.join(home, ".local", "bin", command),
        # Canonical Node location for from-source Linux builds, the Hermes Docker
        # image and Intel Homebrew. Needed when a user's hand-authored env.PATH
        # omits it: npx's shebang re-execs /usr/bin/env node, so a symlink
        # workaround fails one layer deeper.
        os.path.join(os.sep, "usr", "local", "bin", command),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return command


def _resolve_stdio_command(command: str, env: dict) -> tuple[str, dict]:
    """Resolve a stdio command against the exact subprocess env, mainly so bare
    ``npx``/``npm``/``node`` work under a filtered PATH."""
    resolved_command = os.path.expanduser(str(command).strip())
    resolved_env = dict(env or {})

    if os.sep not in resolved_command:
        path_arg = resolved_env.get("PATH")
        which_hit = shutil.which(resolved_command, path=path_arg)
        if which_hit is None and sys.platform == "win32" and resolved_env:
            which_hit = _which_with_config_pathext(resolved_command, path_arg, resolved_env)
        if which_hit:
            resolved_command = which_hit
        elif resolved_command in {"npx", "npm", "node"}:
            resolved_command = _node_fallback(resolved_command)

    command_dir = os.path.dirname(resolved_command)
    if command_dir:
        resolved_env = _prepend_path(resolved_env, command_dir)
    return resolved_command, resolved_env


def _wrap_command_with_watchdog(command: str, args: list) -> tuple[str, list]:
    """Wrap a stdio command in the parent-death watchdog (POSIX only — it relies
    on process groups, same scope as the killpg-based orphan cleanup; the
    watchdog polls ``getppid()`` against our PID). Unchanged on non-POSIX or if
    the PID cannot be read — watchdog bookkeeping must never block a connection."""
    if os.name != "posix":
        return command, args
    try:
        my_pid = os.getpid()
    except Exception:
        return command, args
    watchdog = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mcp_stdio_watchdog.py")
    return sys.executable, [watchdog, "--ppid", str(my_pid), "--", command, *args]


def _interpolate_env_vars(value):
    """Recursively resolve ``${VAR}`` / Cursor ``${env:VAR}`` placeholders plus
    the Cursor context vars (``_context_var_value``). Env refs resolve from the
    active profile's secret scope when multiplexing (so ``${API_KEY}`` picks up
    the routed profile's value, not another profile's in ``os.environ``). Unset
    vars keep the literal placeholder."""
    from agent.secret_scope import get_secret as _get_secret

    if isinstance(value, str):
        def _replace(m):
            ctx = _context_var_value(m.group(1).strip())
            if ctx is not None:
                return ctx
            return _get_secret(_env_ref_name(m.group(1)), m.group(0)) or m.group(0)
        return _ENV_VAR_PATTERN.sub(_replace, value)
    if isinstance(value, dict):
        return {k: _interpolate_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate_env_vars(v) for v in value]
    return value


# (server_name, dotted key path) pairs already warned about; config loads
# happen on every discovery pass, so warn once per process.
_whitespace_warned: Set[Tuple[str, str]] = set()


def _warn_hidden_whitespace(server_name: str, config: dict) -> List[str]:
    """Warn once per (server, key path) about string values with leading/trailing
    whitespace — a pasted newline or leading space causes opaque auth/connect
    failures and is invisible in config.yaml. Advisory only: values are never
    mutated (whitespace could be intentional) and never logged (often secrets).
    Returns the flagged key paths."""
    flagged: List[str] = []

    def _walk(value: Any, path: str) -> None:
        if isinstance(value, str):
            if value != value.strip():
                flagged.append(path)
        elif isinstance(value, dict):
            for k, v in value.items():
                _walk(v, f"{path}.{k}" if path else str(k))
        elif isinstance(value, list):
            for i, v in enumerate(value):
                _walk(v, f"{path}[{i}]")

    _walk(config, "")
    for key_path in flagged:
        dedupe_key = (server_name, key_path)
        if dedupe_key in _whitespace_warned:
            continue
        _whitespace_warned.add(dedupe_key)
        logger.warning(
            "MCP server '%s': config value '%s' has hidden leading or "
            "trailing whitespace — this often causes authentication or "
            "connection failures. Check for stray spaces/newlines in "
            "config.yaml (or the referenced env var).",
            server_name, key_path,
        )
    return flagged


def _filter_suspicious_mcp_servers(servers: Dict[str, dict]) -> Dict[str, dict]:
    """Drop exfiltration-shaped MCP configs before any stdio spawn path."""
    try:
        from hermes_cli.mcp_security import validate_mcp_server_entry
    except Exception:
        return servers

    safe_servers = {}
    for name, cfg in servers.items():
        issues = validate_mcp_server_entry(name, cfg) if isinstance(cfg, dict) else None
        if issues:
            logger.warning("Skipping suspicious MCP server '%s': %s", name, "; ".join(issues))
            continue
        safe_servers[name] = cfg
    return safe_servers


def _portable_mcp_servers(safe_servers: Dict[str, dict]) -> None:
    """Merge plugin-provided (portable) MCP servers into *safe_servers*; native
    config wins on a name clash. Never raises."""
    try:
        from hermes_cli.plugins import discover_plugins, get_plugin_manager

        discover_plugins()
        portable = get_plugin_manager().get_portable_mcp_servers()
        for name, cfg in _core._filter_suspicious_mcp_servers(portable).items():
            if name in safe_servers:
                logger.warning("Portable MCP server '%s' conflicts with native config; skipping", name)
                continue
            safe_servers[name] = dict(cfg)
    except Exception:
        logger.debug("Failed to load portable MCP servers", exc_info=True)


def _load_mcp_config() -> Dict[str, dict]:
    """Read ``mcp_servers`` from config.yaml as ``{name: config}`` (empty on error
    or in safe mode). Entries carry ``command``/``args``/``env`` (stdio) or
    ``url``/``headers`` (HTTP) plus optional timeout/auth keys; ``${VAR}``
    placeholders are interpolated after ``.env`` is loaded."""
    try:
        from hermes_cli.config import load_config
        from utils import env_var_enabled as _env_enabled

        if _env_enabled("HERMES_SAFE_MODE"):
            return {}
        servers = load_config().get("mcp_servers")
        if not isinstance(servers, dict):
            servers = {}
        # Ensure .env vars are available for interpolation
        try:
            from hermes_cli.env_loader import load_hermes_dotenv
            load_hermes_dotenv()
        except Exception:
            pass
        safe_servers: Dict[str, dict] = {}
        for name, cfg in _core._filter_suspicious_mcp_servers(servers).items():
            interpolated = _interpolate_env_vars(cfg)
            if isinstance(interpolated, dict):
                _warn_hidden_whitespace(name, interpolated)
                safe_servers[name] = interpolated
        _portable_mcp_servers(safe_servers)
        return safe_servers
    except Exception as exc:
        logger.debug("Failed to load MCP config: %s", exc)
        return {}
