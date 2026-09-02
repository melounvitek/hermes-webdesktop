"""Terminal backend configuration: scope-aware TERMINAL_* reads, env-var parsing, container-cwd sanity checks, plugin-backend classification and the resolved config dict (_get_env_config).

Split out of ``tools/terminal_tool.py``; every public/patched name is re-imported there,
so ``tools.terminal_tool.<name>`` keeps resolving (and monkeypatching) as before.
"""

import logging
import json
import os
from typing import Any

# Log-record parity with the origin module.
logger = logging.getLogger("tools.terminal_tool")


def _parse_env_var(name: str, default: str, converter: Any = int, type_label: str = "integer"):
    """Parse an env var with *converter*, raising a clear ValueError on bad
    values (e.g. TERMINAL_TIMEOUT=5m) instead of an opaque crash. TERMINAL_*
    names are read scope-aware via :func:`_tenv`."""
    raw = os.getenv(name, default)
    if name.startswith("TERMINAL_"):
        raw = _tenv(name, default)
    try:
        return converter(raw)
    except (ValueError, json.JSONDecodeError):
        raise ValueError(
            f"Invalid value for {name}: {raw!r} (expected {type_label}). "
            f"Check ~/.hermes/.env or environment variables."
        )


def _safe_getcwd() -> str:
    """``os.getcwd()`` tolerant of a deleted cwd (FileNotFoundError) or a macOS
    TCC-protected one without Full Disk Access (PermissionError); falls back
    to TERMINAL_CWD, then the home directory."""
    try:
        return os.getcwd()
    except (FileNotFoundError, PermissionError):
        return _tenv("TERMINAL_CWD") or os.path.expanduser("~")


# Host-cwd prefixes that cannot exist inside a container sandbox (POSIX user
# dirs and Windows drive paths as they leak toward a Linux ``-w`` flag).
_HOST_CWD_PREFIXES = ("/Users/", "/home/", "C:\\", "C:/")


_CONTAINER_BACKENDS = frozenset({"docker", "singularity", "modal", "daytona", "vercel_sandbox"})


def _plugin_env_flag(env_type: str, attr: str, default=False):
    """Classification flag of a plugin-registered backend. Fail-soft: *default*
    when the registry is unavailable, the backend unknown, or the provider
    raises — a misbehaving plugin must never take the terminal tool down."""
    if not env_type or env_type in _CONTAINER_BACKENDS or env_type in {"local", "ssh", "managed_modal"}:
        return default
    try:
        from agent.terminal_env_registry import provider_flag

        return provider_flag(env_type, attr, default)
    except Exception:
        return default


def _is_container_backend(env_type: str) -> bool:
    """True for built-in container backends and plugins declaring ``is_container``."""
    return env_type in _CONTAINER_BACKENDS or _plugin_env_flag(env_type, "is_container")


def _get_plugin_env_provider(env_type: str):
    """Return the registered plugin provider for *env_type*, or None."""
    if not env_type or env_type in _CONTAINER_BACKENDS or env_type in {"local", "ssh", "managed_modal"}:
        return None
    try:
        from agent.terminal_env_registry import get_provider

        return get_provider(env_type)
    except Exception:
        return None


def _is_unusable_container_cwd(cwd: str) -> bool:
    """True if *cwd* is a host or relative path that can't be a container
    workdir: ``docker run -w`` needs an absolute in-sandbox path, otherwise the
    container fails to start (exit 125). Windows drive paths aren't ``isabs``
    on POSIX, so they're caught by the prefix check."""
    if not cwd:
        return False
    return cwd.startswith(_HOST_CWD_PREFIXES) or not os.path.isabs(cwd)


def _tenv(name: str, default: str = "") -> str:
    """Scope-aware read of a ``TERMINAL_*`` variable. Every terminal setting
    must go through this: under gateway multiplexing the active profile's
    config arrives via a per-turn scope, and a raw ``os.getenv`` would read
    whatever a previous turn pinned into the process env (cross-profile leak)."""
    from tools.terminal_scope import terminal_env

    return terminal_env(name, default)


def _tenv_bool(name: str, default: str) -> bool:
    """Scope-aware boolean ``TERMINAL_*`` read: true/1/yes (case-insensitive)."""
    return _tenv(name, default).lower() in {"true", "1", "yes"}
