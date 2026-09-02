"""Path resolution for the file tools: task-aware base dir, ``~`` expansion, workspace-divergence warning.

Companion to ``tools.file_tools`` (which re-imports every name here). The core
invariant: the base directory used to anchor relative paths is ALWAYS absolute
and derived from the task's terminal cwd, never from the process cwd unless no
other anchor exists. A relative or sentinel ``TERMINAL_CWD`` would otherwise
silently anchor edits to the agent process cwd (e.g. the main repo while a
worktree session is active).
"""

import os
import posixpath
import sys
from pathlib import Path, PurePosixPath

# ``TERMINAL_CWD`` values that mean "not configured", not a directory to resolve
# against ("." from a stale config; "auto"/"cwd" are setup-wizard placeholders).
# The gateway sanitizes the same set at import time (gateway/run.py).
_TERMINAL_CWD_SENTINELS = frozenset({"", ".", "./", "auto", "cwd"})
_CONTAINER_PATH_BACKENDS_FALLBACK = frozenset({"docker", "singularity", "modal", "daytona", "vercel_sandbox"})
# Backend name inferred from the live environment's class name (first match wins).
_ENV_CLASS_NAME_HINTS = ("local", "ssh", "docker", "singularity", "modal", "daytona")


def _expand_tilde(path: str) -> str:
    """Expand ``~`` using the effective profile home when available.

    In-process file tools share the gateway process's HOME, which may differ
    from the profile-specific HOME interactive CLI sessions use; mirroring
    ``hermes_constants.get_subprocess_home()`` keeps ``~`` consistent across
    interactive and gateway-driven (cron) runs.
    """
    if not path or "~" not in path:
        return path
    try:
        from hermes_constants import get_subprocess_home

        home = get_subprocess_home()
    except Exception:
        home = None
    if home and (path == "~" or path.startswith("~/")):
        return home if path == "~" else os.path.join(home, path[2:])
    return os.path.expanduser(path)


def _terminal_env_type_for_task(task_id: str = "default") -> str:
    """Best-effort terminal backend type for path-resolution decisions."""
    try:
        from tools.terminal_tool import (
            _active_environments,
            _env_lock,
            _get_env_config,
            _resolve_container_task_id,
        )

        try:
            container_key = _resolve_container_task_id(task_id)
        except Exception:
            container_key = task_id
        with _env_lock:
            env = _active_environments.get(container_key) or _active_environments.get(task_id)
        if env is not None:
            name = env.__class__.__name__.lower()
            for hint in _ENV_CLASS_NAME_HINTS:
                if hint in name:
                    return hint
            stamped = getattr(env, "_hermes_backend_name", None)
            if isinstance(stamped, str) and stamped:
                return stamped
        cfg = _get_env_config()
        return str(cfg.get("env_type") or os.getenv("TERMINAL_ENV") or "local").lower()
    except Exception:
        return str(os.getenv("TERMINAL_ENV") or "local").lower()


def _uses_container_paths(task_id: str = "default") -> bool:
    env_type = _terminal_env_type_for_task(task_id)
    try:
        from tools.terminal_tool import _is_container_backend

        return _is_container_backend(env_type)
    except Exception:
        return env_type in _CONTAINER_PATH_BACKENDS_FALLBACK


def _normalize_without_host_deref(path: str | Path | PurePosixPath) -> PurePosixPath:
    """Normalize path syntax without following host symlinks.

    Container paths are meaningful inside the sandbox; ``Path.resolve()`` on the
    host could dereference a host-side symlink (e.g. ``/workspace``) and rewrite
    the path before Docker sees it.
    """
    return PurePosixPath(posixpath.normpath(str(path)))


def _sentinel_free_abs_cwd(raw: str | None) -> str | None:
    """Return *raw* expanded when it is a non-sentinel ABSOLUTE anchor, else ``None``.

    A relative anchor is meaningless without knowing which cwd it is relative
    to — exactly the ambiguity that misroutes worktree edits.
    """
    raw = str(raw or "").strip()
    if raw.lower() in _TERMINAL_CWD_SENTINELS:
        return None
    expanded = _expand_tilde(raw)
    if not os.path.isabs(expanded):
        return None
    return expanded


def _configured_terminal_cwd() -> str | None:
    """Return ``$TERMINAL_CWD`` only when it names a real (absolute, non-sentinel) anchor.

    Scope-aware: under gateway multiplexing the routed profile's cwd lives in
    the per-turn terminal scope, not the process env.
    """
    from agent.runtime_cwd import scope_terminal_cwd

    return _sentinel_free_abs_cwd(scope_terminal_cwd() or None)


def _registered_task_cwd_override(task_id: str = "default") -> str | None:
    """Return a registered cwd override keyed by the RAW task id, when available.

    ``terminal_tool`` collapses CWD-only task overrides to the shared
    ``"default"`` environment (TUI/dashboard/ACP sessions share one sandbox),
    but the cwd value itself stays keyed by the raw session id — so read the
    raw override before falling back to the collapsed container key.
    """
    try:
        from tools.terminal_tool import resolve_task_overrides

        overrides = resolve_task_overrides(task_id)
    except Exception:
        return None

    return _sentinel_free_abs_cwd(overrides.get("cwd"))


def _authoritative_workspace_root(task_id: str = "default") -> str | None:
    """Best-effort absolute workspace root, or ``None`` when no reliable anchor exists.

    Order: (1) the session's own cwd record (written on every completed terminal
    command; per-session, so one session's ``cd`` never leaks into another);
    (2) a registered raw-keyed task/session cwd override (TUI/Desktop/ACP);
    (3) a sentinel-free absolute ``$TERMINAL_CWD`` (``-w`` sessions).
    """
    try:
        from tools.terminal_tool import get_session_cwd

        recorded = get_session_cwd(task_id)
    except Exception:
        recorded = None
    if recorded:
        return recorded
    registered = _registered_task_cwd_override(task_id)
    if registered:
        return registered
    return _configured_terminal_cwd()


def _resolve_base_dir(
    task_id: str = "default",
    *,
    container_paths: bool | None = None,
) -> Path | PurePosixPath:
    """Return the ABSOLUTE base directory for resolving relative paths.

    Uses ``_authoritative_workspace_root`` (live cwd → registered override →
    ``$TERMINAL_CWD``), falling back to the process cwd only as a last resort.
    Sentinel/relative ``TERMINAL_CWD`` values are rejected outright rather than
    anchored to the process cwd, so the result never depends on where the
    agent process happens to run.
    """
    root = _authoritative_workspace_root(task_id)
    if container_paths is None:
        container_paths = _uses_container_paths(task_id)
    base_text = _expand_tilde(root) if root else os.getcwd()
    if container_paths:
        if not posixpath.isabs(base_text):
            base_text = posixpath.join(os.getcwd(), base_text)
        return _normalize_without_host_deref(base_text)
    # Git Bash ``pwd -P`` reports ``/c/Users/...``; translate before Path so
    # relative file-tool paths don't anchor under a nonexistent ``\\c\\Users``.
    from tools.environments.local import _msys_to_windows_path

    base_text = _msys_to_windows_path(base_text)
    if sys.platform == "win32":
        import ntpath

        if not ntpath.isabs(base_text):
            base_text = ntpath.join(os.getcwd(), base_text)
        return Path(ntpath.normpath(base_text))
    base = Path(base_text)
    if not base.is_absolute():
        # A backend reporting a relative cwd is anchored to the process cwd
        # once, here, so the result no longer depends on cwd at resolve().
        base = Path(os.getcwd()) / base
    return base.resolve()


def _resolve_path_for_task(filepath: str, task_id: str = "default") -> Path | PurePosixPath:
    """Resolve *filepath* against the task's absolute base directory.

    Absolute inputs are returned resolved-but-unanchored. On native Windows,
    Git Bash / MSYS drive paths (``/c/Users/...``) are translated first so
    they aren't treated as relative ``\\c\\Users\\...`` under the process cwd;
    container/WSL Linux paths are never rewritten.
    """
    container_paths = _uses_container_paths(task_id)
    if container_paths:
        expanded = _expand_tilde(filepath)
        if posixpath.isabs(expanded):
            return _normalize_without_host_deref(expanded)
        resolved = _resolve_base_dir(task_id, container_paths=True) / expanded
        return _normalize_without_host_deref(resolved)

    from tools.environments.local import _msys_to_windows_path

    expanded = _expand_tilde(_msys_to_windows_path(filepath))
    if sys.platform == "win32":
        import ntpath

        if ntpath.isabs(expanded):
            return Path(ntpath.normpath(expanded))
        joined = ntpath.join(str(_resolve_base_dir(task_id, container_paths=False)), expanded)
        return Path(ntpath.normpath(joined))

    p = Path(expanded)
    if p.is_absolute():
        return p.resolve()
    resolved = _resolve_base_dir(task_id, container_paths=False) / p
    return resolved.resolve()


# Back-compat alias (imported by agent.context_references and tests).
_resolve_path = _resolve_path_for_task


def _path_resolution_warning(filepath: str, resolved: Path, task_id: str = "default") -> str | None:
    """Warn when a RELATIVE path resolved OUTSIDE the task's workspace root.

    Surfaces the worktree-cwd divergence the moment it matters — the edit is
    about to land in a different checkout than the terminal's cwd. ``None`` for
    absolute paths, an unknown root, or a path correctly under the root. Fires
    on the very first write even before any ``cd`` populated the cwd registry.
    """
    try:
        if Path(_expand_tilde(filepath)).is_absolute():
            return None
        workspace_root = _authoritative_workspace_root(task_id)
        if not workspace_root:
            return None
        if _uses_container_paths(task_id):
            root = _normalize_without_host_deref(Path(_expand_tilde(workspace_root)))
        else:
            root = Path(_expand_tilde(workspace_root)).resolve()
        try:
            resolved.relative_to(root)
            return None
        except ValueError:
            return (
                f"Relative path {filepath!r} resolved to {str(resolved)!r}, which is "
                f"OUTSIDE the active workspace ({str(root)!r}). The edit will land in "
                f"a different directory than the terminal's cwd. If this is not "
                f"intended (e.g. a git-worktree session writing into the main "
                f"checkout), pass an absolute path under the workspace instead."
            )
    except Exception:
        return None
