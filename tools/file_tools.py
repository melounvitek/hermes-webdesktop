#!/usr/bin/env python3
"""File Tools Module - LLM agent file manipulation tools.

Companion modules (every name is re-imported here so ``tools.file_tools.X``
keeps working for callers and test patches):
  * ``file_tools_paths``         — task-aware path resolution / ``~`` expansion.
  * ``file_tools_write_guards``  — sensitive-path, protected-instruction,
                                   approval, mirror and binary-document guards.
  * ``file_tools_read_tracking`` — per-task dedup / loop-detection / staleness state.
"""

import base64
import errno
import json
import logging
import os
import re
import stat
import threading
from pathlib import Path

from agent.file_safety import get_read_block_error
from tools.binary_extensions import has_binary_extension
from tools.file_operations import (
    ShellFileOperations,
    normalize_read_pagination,
    normalize_search_pagination,
)
from tools import file_state
from agent.redact import redact_sensitive_text
from tools.file_tools_paths import (  # noqa: F401  (re-exported)
    _CONTAINER_PATH_BACKENDS_FALLBACK,
    _TERMINAL_CWD_SENTINELS,
    _authoritative_workspace_root,
    _configured_terminal_cwd,
    _expand_tilde,
    _normalize_without_host_deref,
    _path_resolution_warning,
    _registered_task_cwd_override,
    _resolve_base_dir,
    _resolve_path,
    _resolve_path_for_task,
    _sentinel_free_abs_cwd,
    _terminal_env_type_for_task,
    _uses_container_paths,
)
from tools.file_tools_write_guards import (  # noqa: F401  (re-exported)
    _PROTECTED_INSTRUCTION_BASENAMES,
    _READ_DEDUP_STATUS_MESSAGE,
    _SENSITIVE_EXACT_PATHS,
    _SENSITIVE_PATH_PREFIXES,
    _check_approval_required_write,
    _check_binary_document_write,
    _check_cross_profile_path,
    _check_protected_instruction_write,
    _check_sensitive_path,
    _get_container_mirror_prefix_for_task,
    _get_hermes_config_resolved,
    _get_real_hermes_home,
    _is_internal_file_status_text,
    _is_internal_file_tool_content,
    _looks_like_read_file_line_numbered_content,
    _protected_instruction_config,
    _protected_instruction_reason,
    _request_protected_instruction_approval,
)
from tools.file_tools_read_tracking import (  # noqa: F401  (re-exported)
    _DEDUP_CAP,
    _NOT_FOUND_CAP,
    _NOT_FOUND_TTL_SECONDS,
    _READ_HISTORY_CAP,
    _READ_TIMESTAMPS_CAP,
    _bump_consecutive,
    _cap_read_tracker_data,
    _check_file_staleness,
    _check_not_found_cache,
    _invalidate_dedup_for_path,
    _mark_verification_stale,
    _patch_failure_lock,
    _patch_failure_tracker,
    _read_tracker,
    _read_tracker_lock,
    _record_not_found,
    _record_patch_failure,
    _reset_patch_failures,
    _task_data,
    _update_read_timestamp,
    notify_other_tool_call,
    reset_file_dedup,
)

logger = logging.getLogger(__name__)


_EXPECTED_WRITE_ERRNOS = {errno.EACCES, errno.EPERM, errno.EROFS}

# ---------------------------------------------------------------------------
# Read-size guard. Model-agnostic, so characters proxy tokens: 100K chars is
# ~25-35K tokens across typical tokenisers. Configurable: file_read_max_chars.
# ---------------------------------------------------------------------------
_DEFAULT_MAX_READ_CHARS = 100_000
_max_read_chars_cached: int | None = None


def _get_max_read_chars() -> int:
    """Return ``file_read_max_chars`` from config.yaml (cached per process; default on missing/invalid)."""
    global _max_read_chars_cached
    if _max_read_chars_cached is not None:
        return _max_read_chars_cached
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
        val = cfg.get("file_read_max_chars")
        if isinstance(val, (int, float)) and val > 0:
            _max_read_chars_cached = int(val)
            return _max_read_chars_cached
    except Exception:
        pass
    _max_read_chars_cached = _DEFAULT_MAX_READ_CHARS
    return _max_read_chars_cached


def _truncate_to_char_budget(content: str, max_chars: int) -> tuple[str, int, bool]:
    """Trim line-numbered ``read_file`` content to the last COMPLETE line within *max_chars*.

    Returns ``(kept_text, lines_kept, truncated)`` so the caller can offer a
    ``next_offset`` instead of rejecting the read. Lines are already clamped to
    ``get_max_line_length()`` upstream; the overflow handled here is the
    accumulation of many lines under the line-count limit (logs, wide CSV).
    If not even the first line fits it is clamped mid-line (Python slicing
    never splits a code point) so the read is never empty and the cursor advances.
    """
    if len(content) <= max_chars:
        return content, (content.count("\n") + 1 if content else 0), False

    lines = content.split("\n")
    kept: list[str] = []
    running = 0
    for line in lines:
        addition = len(line) + (1 if kept else 0)  # +1 for the rejoining "\n"
        if running + addition > max_chars:
            break
        kept.append(line)
        running += addition

    if not kept:
        kept.append(lines[0][:max_chars])

    return "\n".join(kept), len(kept), True


def _apply_char_budget(result_dict: dict, content: str, offset: int, total_lines, max_chars: int) -> str:
    """Trim *content* to the char budget, annotate *result_dict* with the
    continuation hint, and return the trimmed text."""
    trimmed, lines_kept, _ = _truncate_to_char_budget(content, max_chars)
    next_offset = offset + lines_kept
    shown_end = offset + lines_kept - 1
    result_dict["content"] = trimmed
    result_dict["truncated"] = True
    result_dict["truncated_by"] = "bytes"
    result_dict["next_offset"] = next_offset
    result_dict["hint"] = (
        f"Output truncated at the {max_chars:,}-char read budget after "
        f"{lines_kept} line(s) (showing lines {offset}-{shown_end} of "
        f"{total_lines}). Use offset={next_offset} to continue."
    )
    if len(trimmed.split("\n", 1)[0]) >= max_chars:
        result_dict["hint"] += (
            " Note: the first line alone exceeded the budget and was "
            "clamped mid-line; its remainder is not retrievable via "
            "offset."
        )
    return trimmed


# Above this size, a wide read (limit > 200) gets a hint toward targeted reads.
_LARGE_FILE_HINT_BYTES = 512_000

# Device/fd paths whose reads hang the process (infinite output or blocking on
# input). Checked by path only — no I/O.
_BLOCKED_DEVICE_PATHS = frozenset({
    "/dev/zero", "/dev/random", "/dev/urandom", "/dev/full",     # never reach EOF
    "/dev/stdin", "/dev/tty", "/dev/console",                    # block on input
    "/dev/stdout", "/dev/stderr",                                # nonsensical to read
    "/dev/fd/0", "/dev/fd/1", "/dev/fd/2",                       # fd aliases
})
# /proc/<pid>/... (and /proc/<pid>/task/<tid>/...) files that leak secrets,
# argv, memory layout (ASLR oracle: maps family, auxv, pagemap) or raw memory.
_BLOCKED_PROC_SUFFIXES = (
    "/fd/0", "/fd/1", "/fd/2",  # stdio aliases
    "/environ", "/cmdline", "/maps", "/smaps", "/smaps_rollup", "/numa_maps",
    "/mem", "/auxv", "/pagemap",
)


def _file_ops_uses_host_paths(file_ops) -> bool:
    """True when *file_ops* targets the same host filesystem as Hermes.

    Only then may we stat paths or rewrite V4A headers to host-absolute paths;
    a container/remote backend has its own filesystem namespace.
    """
    env = getattr(file_ops, "env", None)
    if env is None:
        return True
    try:
        from tools.environments.local import LocalEnvironment
    except ImportError:
        return True
    return isinstance(env, LocalEnvironment)


# V4A file headers. ``\s*`` after ``***`` mirrors patch_parser's leniency
# (``***Update File:`` with no space parses and applies, so it must be checked).
_V4A_SINGLE_HEADER_RE = re.compile(r'^(\*\*\*\s*(?:Update|Add|Delete)\s+File:\s*)(.+)$', re.MULTILINE)
_V4A_SINGLE_HEADER_OP_RE = re.compile(r'^\*\*\*\s*(Update|Add|Delete)\s+File:\s*(.+)$', re.MULTILINE)
_V4A_MOVE_HEADER_RE = re.compile(r'^(\*\*\*\s*Move\s+File:\s*)(.+?)\s*->\s*(.+)$', re.MULTILINE)


def _rewrite_v4a_patch_paths_for_host(
    patch: str,
    path_to_resolved: dict,
    file_ops,
) -> str:
    """Rewrite V4A ``*** Update/Add/Delete/Move File:`` headers to the resolved host paths.

    ``patch_tool`` resolves every header against the task's workspace for
    locking/staleness/reporting; the shell layer must patch those SAME files
    rather than re-resolving a relative header against its own cwd (which can
    differ — the git-worktree cwd bug). Only applied for host-filesystem backends.
    """
    if not _file_ops_uses_host_paths(file_ops):
        return patch

    def _resolved_or_original(raw: str) -> str:
        raw = raw.strip()
        return path_to_resolved.get(raw) or raw

    patch = _V4A_SINGLE_HEADER_RE.sub(
        lambda m: f"{m.group(1)}{_resolved_or_original(m.group(2))}", patch,
    )
    return _V4A_MOVE_HEADER_RE.sub(
        lambda m: f"{m.group(1)}{_resolved_or_original(m.group(2))} -> {_resolved_or_original(m.group(3))}",
        patch,
    )


def _is_blocked_device_path(path: str) -> bool:
    """Return True for concrete device/fd/proc paths that can hang reads or leak process state."""
    normalized = os.path.normpath(_expand_tilde(path))
    if normalized in _BLOCKED_DEVICE_PATHS:
        return True
    return normalized.startswith("/proc/") and normalized.endswith(_BLOCKED_PROC_SUFFIXES)


def _is_blocked_device(filepath: str, base_dir: str | Path | None = None) -> bool:
    """True if the path (literal, any symlink hop, or final realpath) is a blocked device.

    The literal path is checked first so aliases like /dev/stdin are caught
    before they resolve to terminal-specific paths; each symlink hop is checked
    so an alias to a device cannot bypass the guard.
    """
    expanded = _expand_tilde(filepath)
    if base_dir is not None and not os.path.isabs(expanded):
        expanded = os.path.join(os.fspath(base_dir), expanded)
    normalized = os.path.normpath(expanded)
    if _is_blocked_device_path(normalized):
        return True

    seen: set[str] = set()
    current = normalized
    for _ in range(20):
        try:
            target = os.readlink(current)
        except OSError:
            break
        if not os.path.isabs(target):
            target = os.path.join(os.path.dirname(current), target)
        target = os.path.normpath(target)
        if _is_blocked_device_path(target):
            return True
        if target in seen:
            break
        seen.add(target)
        current = target

    try:
        resolved = os.path.normpath(os.path.realpath(normalized))
    except (OSError, ValueError):
        return False
    return _is_blocked_device_path(resolved)


def _search_result_read_block_error(path: str, task_id: str = "default") -> str | None:
    """Read-safety error for a search result path, resolved against the task cwd
    (search backends may return cwd-relative paths; the process cwd can differ)."""
    try:
        resolved = _resolve_path_for_task(path, task_id)
    except (OSError, ValueError, RuntimeError):
        return get_read_block_error(path)
    return get_read_block_error(str(resolved))


def _filter_read_blocked_search_results(result, task_id: str = "default") -> int:
    """Remove credential/cache/env paths from a SearchResult in-place; return the omitted count."""
    omitted = 0

    def _blocked(path: str) -> bool:
        nonlocal omitted
        if _search_result_read_block_error(path, task_id):
            omitted += 1
            return True
        return False

    if getattr(result, "matches", None):
        result.matches = [m for m in result.matches if not _blocked(m.path)]
    if getattr(result, "files", None):
        result.files = [f for f in result.files if not _blocked(f)]
    if getattr(result, "counts", None):
        result.counts = {f: c for f, c in result.counts.items() if not _blocked(f)}
    return omitted


def _is_expected_write_exception(exc: Exception) -> bool:
    """Return True for expected write denials that should not hit error logs."""
    if isinstance(exc, PermissionError):
        return True
    return isinstance(exc, OSError) and exc.errno in _EXPECTED_WRITE_ERRNOS


# ---------------------------------------------------------------------------
# ShellFileOperations per terminal environment
# ---------------------------------------------------------------------------
_file_ops_lock = threading.Lock()
_file_ops_cache: dict = {}

# Per-backend config key that names the sandbox image (overridable per task).
_ENV_IMAGE_KEYS = {
    "docker": "docker_image",
    "singularity": "singularity_image",
    "modal": "modal_image",
    "daytona": "daytona_image",
}
_CONTAINER_CONFIG_DEFAULTS = (
    ("container_cpu", 1),
    ("container_memory", 5120),
    ("container_disk", 51200),
    ("container_persistent", True),
    ("vercel_runtime", ""),
    ("docker_volumes", []),
    ("docker_mount_cwd_to_workspace", False),
    ("docker_forward_env", []),
    ("docker_run_as_host_user", False),
    ("docker_network", True),
)


def _create_terminal_env_for_file_ops(raw_task_id: str, task_id: str):
    """Build the terminal environment for *task_id* from config + per-task overrides.

    Mirrors terminal_tool's own environment construction so a file tool that
    runs before any terminal command uses the configured backend (docker,
    modal, ...) rather than always defaulting to local.
    """
    from tools.terminal_tool import (
        _CONTAINER_BACKENDS,
        _create_environment,
        _get_env_config,
        _is_container_backend,
        _is_unusable_container_cwd,
        _resolve_task_host_cwd,
        get_session_cwd,
        resolve_task_overrides,
    )

    config = _get_env_config()
    env_type = config["env_type"]
    overrides = resolve_task_overrides(raw_task_id)

    image_key = _ENV_IMAGE_KEYS.get(env_type)
    image = (overrides.get(image_key) or config[image_key]) if image_key else ""

    try:
        recorded_cwd = get_session_cwd(raw_task_id)
    except Exception:
        recorded_cwd = None
    cwd = overrides.get("cwd") or recorded_cwd or config["cwd"]
    # Re-apply the container cwd guard _get_env_config() already ran on
    # config["cwd"]: a gateway/TUI/ACP cwd override is a raw HOST path, and
    # ``docker run -w <host-path>`` starts the container in a directory that
    # doesn't exist there, so search_files & co silently return nothing.
    # Valid in-container overrides (/workspace, /root, ...) pass untouched.
    if env_type in _CONTAINER_BACKENDS and _is_unusable_container_cwd(cwd):
        if cwd != config["cwd"]:
            logger.info(
                "Ignoring host/relative cwd override %r for %s backend "
                "(won't exist in sandbox). Using %r instead.",
                cwd, env_type, config["cwd"],
            )
        cwd = config["cwd"]
    logger.info("Creating new %s environment for task %s...", env_type, task_id[:8])

    container_config = None
    if _is_container_backend(env_type):
        container_config = {key: config.get(key, default) for key, default in _CONTAINER_CONFIG_DEFAULTS}

    ssh_config = None
    if env_type == "ssh":
        ssh_config = {
            "host": config.get("ssh_host", ""),
            "user": config.get("ssh_user", ""),
            "port": config.get("ssh_port", 22),
            "key": config.get("ssh_key", ""),
            "persistent": config.get("ssh_persistent", False),
        }

    local_config = None
    if env_type == "local":
        local_config = {
            "persistent": config.get("local_persistent", False),
        }

    terminal_env = _create_environment(
        env_type=env_type,
        image=image,
        cwd=cwd,
        timeout=config["timeout"],
        ssh_config=ssh_config,
        container_config=container_config,
        local_config=local_config,
        task_id=task_id,
        host_cwd=_resolve_task_host_cwd(config, raw_task_id),
    )
    return env_type, terminal_env


def _get_file_ops(task_id: str = "default") -> ShellFileOperations:
    """Get or create ShellFileOperations for the task's terminal environment.

    Thread-safe via terminal_tool's per-task creation locks, so concurrent tool
    calls never build duplicate sandboxes. Subagent task_ids collapse to
    "default" (``_resolve_container_task_id``) so delegate_task children share
    the parent's container and cached file_ops; RL/benchmark task_ids with a
    registered env override keep their isolation.
    """
    from tools.terminal_tool import (
        _active_environments, _env_lock,
        _last_activity, _start_cleanup_thread,
        _creation_locks,
        _creation_locks_lock,
        _resolve_container_task_id,
    )
    import time

    raw_task_id = task_id or "default"
    task_id = _resolve_container_task_id(raw_task_id)

    # Fast path: cached AND the environment is still alive (the cleanup thread
    # may have killed it).
    with _file_ops_lock:
        cached = _file_ops_cache.get(task_id)
    if cached is not None:
        with _env_lock:
            if task_id in _active_environments:
                _last_activity[task_id] = time.time()
                return cached
            else:
                # Environment was cleaned up: rescue the old cwd into the
                # session record before dropping the stale entry, FILL-ONLY —
                # ``cached.cwd`` is a snapshot of the SHARED env's cwd, not
                # attributable to this session, so never overwrite a record
                # the session wrote for itself.
                old_cwd = getattr(cached, "cwd", None)
                if old_cwd:
                    try:
                        from tools.terminal_tool import (
                            get_session_cwd,
                            record_session_cwd,
                        )
                        if get_session_cwd(raw_task_id) is None:
                            record_session_cwd(raw_task_id, old_cwd)
                    except Exception:
                        pass
                with _file_ops_lock:
                    _file_ops_cache.pop(task_id, None)

    with _creation_locks_lock:
        if task_id not in _creation_locks:
            _creation_locks[task_id] = threading.Lock()
        task_lock = _creation_locks[task_id]

    with task_lock:
        # Double-check: another thread may have created it while we waited.
        with _env_lock:
            if task_id in _active_environments:
                _last_activity[task_id] = time.time()
                terminal_env = _active_environments[task_id]
            else:
                terminal_env = None

        if terminal_env is None:
            env_type, terminal_env = _create_terminal_env_for_file_ops(raw_task_id, task_id)

            with _env_lock:
                _active_environments[task_id] = terminal_env
                _last_activity[task_id] = time.time()

            _start_cleanup_thread()
            logger.info("%s environment ready for task %s", env_type, task_id[:8])

    file_ops = ShellFileOperations(terminal_env)
    with _file_ops_lock:
        _file_ops_cache[task_id] = file_ops
    return file_ops


def clear_file_ops_cache(task_id: str = None):
    """Clear the file operations cache."""
    with _file_ops_lock:
        if task_id:
            _file_ops_cache.pop(task_id, None)
        else:
            _file_ops_cache.clear()


_SPECIAL_FILE_KINDS = (
    (stat.S_ISFIFO, "a FIFO (named pipe)"),
    (stat.S_ISSOCK, "a socket"),
    (stat.S_ISCHR, "a character device"),
    (stat.S_ISBLK, "a block device"),
)


def _special_file_kind(path) -> str | None:
    """Human name for a non-regular file type that would hang a read, else None.

    Stat-based sibling of the name-based ``_is_blocked_device`` guard: a FIFO
    or socket inside a workspace hangs ``read_file`` as hard as ``/dev/zero``
    but carries no recognizable name. Host filesystems only (see
    ``_file_ops_uses_host_paths``). Missing/unstattable paths return None and
    flow to the normal read path's own error handling.
    """
    try:
        st = os.stat(os.fspath(path))  # follows symlinks, matching a real read
    except OSError:
        return None
    mode = st.st_mode
    if stat.S_ISREG(mode) or stat.S_ISDIR(mode):
        return None
    for predicate, label in _SPECIAL_FILE_KINDS:
        if predicate(mode):
            return label
    return "a special (non-regular) file"


def _read_extracted_document(path: str, _resolved, offset: int, limit: int, task_id: str) -> str | None:
    """Render an extractable document (.docx/.xlsx/.pdf/...) as paginated text.

    Returns the JSON result, a tool_error for a binary document whose
    extraction failed for an actionable reason (size cap, encrypted,
    malformed), or ``None`` to fall through to the normal read path. Runs
    BEFORE the binary-extension guard so Office files can render as text.
    """
    from tools.read_extract import (
        ANYDOC_EXTENSIONS,
        EXTRACTABLE_EXTENSIONS,
        MAX_DOCUMENT_BYTES,
        ExtractionError,
        extract_document_bytes,
        is_extractable_document,
    )

    if not is_extractable_document(str(_resolved)):
        return None
    file_ops = _get_file_ops(task_id)
    try:
        binary = file_ops.read_file_bytes(
            str(_resolved), max_bytes=MAX_DOCUMENT_BYTES
        )
        if binary.error or binary.base64_content is None:
            raise ExtractionError(binary.error or "Document bytes unavailable")
        document_bytes = base64.b64decode(
            binary.base64_content, validate=True
        )
        extracted_text = extract_document_bytes(
            document_bytes, str(_resolved)
        )
    except (ExtractionError, ValueError, base64.binascii.Error) as exc:
        logger.debug("document extraction failed for %s", path, exc_info=True)
        # Binary document formats surface the specific failure: the fallthrough
        # can only yield a generic binary-file error or garbage bytes. .ipynb
        # (plain JSON) and byte-transport errors (ValueError/binascii) fall
        # through — only a specific ExtractionError carries an actionable reason.
        _doc_ext = _resolved.suffix.lower()
        _binary_doc = _doc_ext in ANYDOC_EXTENSIONS or (
            _doc_ext in EXTRACTABLE_EXTENSIONS and _doc_ext != ".ipynb"
        )
        if (
            _binary_doc
            and isinstance(exc, ExtractionError)
            and not str(exc).startswith("Unsupported document type")
        ):
            return tool_error(
                f"Cannot read '{path}' ({_doc_ext}): document "
                f"extraction failed — {exc}. Use terminal utilities "
                "to inspect or convert the file."
            )
        return None

    lines = extracted_text.splitlines()
    total_lines = len(lines)
    end_line = offset + limit - 1
    page_text = "\n".join(lines[offset - 1:end_line])
    result_dict = {
        "content": file_ops._add_line_numbers(page_text, offset) if page_text else "",
        "total_lines": total_lines,
        "file_size": binary.file_size,
        "truncated": total_lines > end_line,
        "extracted_document": True,
    }
    if result_dict["truncated"]:
        result_dict["hint"] = (
            f"Use offset={end_line + 1} to continue reading "
            f"(showing {offset}-{min(end_line, total_lines)} of {total_lines} lines)"
        )
    max_chars = _get_max_read_chars()
    if len(result_dict["content"]) > max_chars:
        _apply_char_budget(result_dict, result_dict["content"], offset, total_lines, max_chars)
    if result_dict["content"]:
        result_dict["content"] = redact_sensitive_text(result_dict["content"], file_read=True)
    return json.dumps(result_dict, ensure_ascii=False)


def _dedup_stub_or_block(task_data: dict, dedup_key: tuple, path: str) -> str:
    """Return the "unchanged" stub for a repeated identical read, escalating to a
    hard BLOCK after 2 stubs so weak tool-followers don't loop forever."""
    with _read_tracker_lock:
        hits = task_data["dedup_hits"].get(dedup_key, 0) + 1
        task_data["dedup_hits"][dedup_key] = hits
        _cap_read_tracker_data(task_data)

    if hits >= 2:
        return tool_error(
            f"BLOCKED: You have called read_file on this "
            f"exact region {hits + 1} times and the file "
            "has NOT changed. STOP calling read_file for "
            "this path — the content from your earlier "
            "read_file result in this conversation is "
            "still current. Proceed with your task using "
            "the information you already have.",
            path=path,
            already_read=hits + 1,
        )

    return json.dumps({
        "status": "unchanged",
        "message": _READ_DEDUP_STATUS_MESSAGE,
        "path": path,
        "dedup": True,
        "content_returned": False,
    }, ensure_ascii=False)


def _record_successful_read(task_data: dict, task_id: str, path: str, resolved_str: str,
                            offset: int, limit: int, dedup_key: tuple, *, partial: bool) -> int:
    """Bookkeeping after a real (non-stub) read; returns the consecutive-read count.

    Per-task tracker (under the lock): clear this key's stub-loop counter, add to
    history, bump the consecutive counter, and store the mtime — it feeds both
    dedup and the write/patch staleness warning. Then, OUTSIDE our lock so the
    registry's own locking isn't nested under it, the cross-agent registry
    (lets write/patch detect sibling-subagent writes after our read) and the
    background-review read-mark (a FULL read of a skill file counts like
    skill_view so a follow-up skill_manage(patch) is accepted; no-op outside
    review forks).
    """
    with _read_tracker_lock:
        task_data["dedup_hits"].pop(dedup_key, None)
        task_data["read_history"].add((path, offset, limit))
        count = _bump_consecutive(task_data, ("read", path, offset, limit))
        try:
            _mtime_now = os.path.getmtime(resolved_str)
            task_data["dedup"][dedup_key] = _mtime_now
            task_data.setdefault("read_timestamps", {})[resolved_str] = _mtime_now
        except OSError:
            pass
        _cap_read_tracker_data(task_data)

    try:
        file_state.record_read(task_id, resolved_str, partial=partial)
    except Exception:
        logger.debug("file_state.record_read failed", exc_info=True)

    if not partial:
        try:
            from tools.skill_manager_tool import mark_background_review_skill_read

            mark_background_review_skill_read(Path(resolved_str))
        except Exception:
            logger.debug("background-review read-mark failed", exc_info=True)
    return count


def read_file_tool(path: str, offset: int = 1, limit: int = 2000, task_id: str = "default") -> str:
    """Read a file with pagination and line numbers.

    Guard order: device-path blocklist (no I/O) → stat-based special-file
    guard (host only) → document extraction → binary-extension guard → Hermes
    internal denylist → negative-result cache → dedup stub → real read.
    """
    try:
        offset, limit = normalize_read_pagination(offset, limit)

        device_base = None if Path(path).expanduser().is_absolute() else _resolve_base_dir(task_id)
        if _is_blocked_device(path, base_dir=device_base):
            return tool_error(
                f"Cannot read '{path}': this is a device file that would "
                "block or produce infinite output."
            )

        _resolved = _resolve_path_for_task(path, task_id)

        # A read on a FIFO/socket blocks until the exec timeout: a self-shipped DoS.
        if _file_ops_uses_host_paths(_get_file_ops(task_id)):
            kind = _special_file_kind(_resolved)
            if kind is not None:
                return json.dumps({
                    "success": False,
                    "note": (
                        f"'{path}' is {kind}, not a regular file — reading "
                        "it would block indefinitely, so no read was "
                        "attempted. Use terminal utilities if you need to "
                        "interact with it."
                    ),
                })

        extracted = _read_extracted_document(path, _resolved, offset, limit, task_id)
        if extracted is not None:
            return extracted

        # The extension is a claim, so this message names only the extension;
        # the content-sniffing path names the actual magic-byte type.
        if has_binary_extension(str(_resolved)):
            _ext = _resolved.suffix.lower()
            return tool_error(
                f"Cannot read binary file '{path}' ({_ext}). "
                "Use vision_analyze for images, or terminal to inspect binary files."
            )

        # Hermes internal denylist: blocks prompt injection via catalog/hub
        # metadata and credential stores under HERMES_HOME. Pass the resolved
        # path: get_read_block_error's own resolve() runs against the process
        # cwd, so a relative "auth.json" read with TERMINAL_CWD == HERMES_HOME
        # would otherwise miss the denylist.
        block_error = get_read_block_error(str(_resolved))
        if block_error:
            return tool_error(block_error)

        resolved_str = str(_resolved)
        cached_not_found = _check_not_found_cache("read", resolved_str, task_id)
        if cached_not_found is not None:
            return cached_not_found

        # Dedup: identical (path, offset, limit) on an unchanged file returns a
        # lightweight stub instead of re-sending the content.
        dedup_key = (resolved_str, offset, limit)
        with _read_tracker_lock:
            task_data = _task_data(task_id)
            cached_mtime = task_data.get("dedup", {}).get(dedup_key)

        if cached_mtime is not None:
            try:
                if os.path.getmtime(resolved_str) == cached_mtime:
                    return _dedup_stub_or_block(task_data, dedup_key, path)
            except OSError:
                pass  # stat failed — fall through to full read

        file_ops = _get_file_ops(task_id)
        result = file_ops.read_file(path, offset, limit)
        result_dict = result.to_dict()

        # Cache a not-found result for retries. Deliberately NO early return:
        # error results still flow through the tracking block below
        # (consecutive-loop detection, dedup bookkeeping) exactly as before the
        # cache existed — serving from the cache is the optimization, recording
        # must stay side-effect-identical.
        _err = result_dict.get("error") or ""
        if isinstance(_err, str) and _err.startswith("File not found:"):
            _record_not_found("read", resolved_str, task_id, json.dumps(result_dict, ensure_ascii=False))

        # Char budget is checked on the FORMATTED content (that is what enters
        # context) and BEFORE redaction, to skip the regex pass on huge content.
        # Graceful truncation instead of rejection: the model gets what fits plus
        # a next_offset, rather than guessing a smaller limit and burning a turn.
        file_size = result_dict.get("file_size", 0)
        max_chars = _get_max_read_chars()
        if len(result.content or "") > max_chars:
            result.content = _apply_char_budget(
                result_dict, result.content or "", offset,
                result_dict.get("total_lines", "unknown"), max_chars,
            )

        if result.content:
            result.content = redact_sensitive_text(result.content, file_read=True)
            result_dict["content"] = result.content

        if (file_size and file_size > _LARGE_FILE_HINT_BYTES
                and limit > 200
                and result_dict.get("truncated")):
            result_dict.setdefault("_hint", (
                f"This file is large ({file_size:,} bytes). "
                "Consider reading only the section you need with offset and limit "
                "to keep context usage efficient."
            ))

        count = _record_successful_read(task_data, task_id, path, resolved_str, offset, limit,
                                        dedup_key, partial=(offset > 1) or bool(result_dict.get("truncated")))

        if count >= 4:
            return tool_error(
                f"BLOCKED: You have read this exact file region {count} times in a row. "
                "The content has NOT changed. You already have this information. "
                "STOP re-reading and proceed with your task.",
                path=path,
                already_read=count,
            )
        elif count >= 3:
            result_dict["_warning"] = (
                f"You have read this exact file region {count} times consecutively. "
                "The content has not changed since your last read. Use the information you already have. "
                "If you are stuck in a loop, stop reading and proceed with writing or responding."
            )

        return json.dumps(result_dict, ensure_ascii=False)
    except Exception as e:
        return tool_error(str(e))


def _resolve_or_none(filepath: str, task_id: str) -> str | None:
    """Task-resolved path string, or None when resolution fails for any reason."""
    try:
        return str(_resolve_path_for_task(filepath, task_id))
    except Exception:
        return None


def _write_precheck_error(paths: list[str], content_paths: list[str], task_id: str,
                          cross_profile: bool) -> str | None:
    """Run the shared write/patch guards in order; return the first error string.

    Order matters: hard denies (sensitive path, mirror) and the corruption
    guard run before anything that could prompt the user, and ONE approval
    prompt covers every path of a multi-file patch.
    """
    for p in paths:
        err = _check_sensitive_path(p, task_id)
        if err:
            return err
        if not cross_profile:
            err = _check_cross_profile_path(p, task_id)
            if err:
                return err
    for p in content_paths:
        err = _check_binary_document_write(p, task_id)
        if err:
            return err
    return (_check_protected_instruction_write(paths, task_id)
            or _check_approval_required_write(paths, task_id))


def write_file_tool(path: str, content: str, task_id: str = "default",
                    cross_profile: bool = False,
                    session_id: str | None = None) -> str:
    """Write content to a file.

    ``cross_profile`` bypasses the sandbox-mirror lost-write guards only
    (unadvertised in the schema; the mirror rejection error teaches it — the
    cross-PROFILE guard it was named for no longer exists).
    """
    # write_file checks the binary-document guard before the mirror guard.
    sensitive_err = _check_sensitive_path(path, task_id)
    if sensitive_err:
        return tool_error(sensitive_err)
    binary_doc_err = _check_binary_document_write(path, task_id)
    if binary_doc_err:
        return tool_error(binary_doc_err)
    protected_err = _check_protected_instruction_write([path], task_id)
    if protected_err:
        return tool_error(protected_err)
    approval_err = _check_approval_required_write([path], task_id)
    if approval_err:
        return tool_error(approval_err)
    if not cross_profile:
        cross_warning = _check_cross_profile_path(path, task_id)
        if cross_warning:
            return tool_error(cross_warning)
    if _is_internal_file_tool_content(content):
        return tool_error(
            "Refusing to write internal read_file display text as file content. "
            "Strip read_file line-number prefixes or reconstruct the intended "
            "file contents before writing."
        )
    try:
        # Resolution failure falls back to the legacy unlocked path (the write
        # still proceeds; the per-task staleness check still runs).
        _resolved = _resolve_or_none(path, task_id)

        if _resolved is None:
            stale_warning = _check_file_staleness(path, task_id)
            file_ops = _get_file_ops(task_id)
            result = file_ops.write_file(path, content)
            result_dict = result.to_dict()
            if stale_warning:
                result_dict["_warning"] = stale_warning
            if not result_dict.get("error"):
                _mark_verification_stale(task_id, [path], session_id=session_id)
            _update_read_timestamp(path, task_id)
            return json.dumps(result_dict, ensure_ascii=False)

        # Per-path lock serializes read→modify→write across concurrent
        # subagents; different paths stay fully parallel.
        with file_state.lock_path(_resolved):
            # Warning priority: cross-agent (names the sibling subagent) >
            # per-task staleness > workspace divergence (relative path resolving
            # outside the terminal's cwd — the worktree-cwd bug).
            cross_warning = file_state.check_stale(task_id, _resolved)
            stale_warning = _check_file_staleness(path, task_id)
            cwd_warning = _path_resolution_warning(path, Path(_resolved), task_id)
            file_ops = _get_file_ops(task_id)
            result = file_ops.write_file(_resolved, content)
            result_dict = result.to_dict()
            effective_warning = cross_warning or stale_warning or cwd_warning
            if effective_warning:
                result_dict["_warning"] = effective_warning
            # Always report the ABSOLUTE path written so a wrong-cwd mismatch is
            # visible in the response instead of silently landing elsewhere.
            result_dict["resolved_path"] = _resolved
            if not result_dict.get("error"):
                result_dict["files_modified"] = [_resolved]
                _mark_verification_stale(task_id, [_resolved], session_id=session_id)
            # Refresh stamps after the write so consecutive edits by the same
            # task don't trigger false staleness warnings.
            _update_read_timestamp(path, task_id)
            if not result_dict.get("error"):
                file_state.note_write(task_id, _resolved)
        return json.dumps(result_dict, ensure_ascii=False)
    except Exception as e:
        if _is_expected_write_exception(e):
            logger.debug("write_file expected denial: %s: %s", type(e).__name__, e)
        else:
            logger.error("write_file error: %s: %s", type(e).__name__, e, exc_info=True)
        return tool_error(str(e))


def _collect_v4a_header_paths(patch: str) -> tuple[list[str], list[str]] | str:
    """Extract every path named in V4A headers, rejecting ``..`` traversal.

    Returns ``(all_paths, content_write_paths)`` or a tool_error string.
    Header paths come from patch CONTENT (skill text, web extract, prompt
    injection) so they are more attacker-influenceable than the explicit
    ``path=`` arg, which keeps its legitimate ``..`` use. Move headers check
    BOTH endpoints (a Move onto /etc/crontab must hit the sensitive-path check).
    Delete/Move don't write text, so only Update/Add feed the binary-document guard.
    """
    from tools.path_security import has_traversal_component

    def _reject_v4a_traversal(v4a_path: str) -> str | None:
        if has_traversal_component(v4a_path):
            return tool_error(
                f"V4A patch header contains '..' traversal: {v4a_path!r}. "
                "Use the agent's cwd-relative path (no '..') or an absolute "
                "path in '*** Update File:' / '*** Add File:' / "
                "'*** Delete File:' / '*** Move File:' headers."
            )
        return None

    paths: list[str] = []
    content_paths: list[str] = []
    for _m in _V4A_SINGLE_HEADER_OP_RE.finditer(patch):
        _op = _m.group(1)
        v4a_path = _m.group(2).strip()
        _err = _reject_v4a_traversal(v4a_path)
        if _err:
            return _err
        paths.append(v4a_path)
        if _op in ("Update", "Add"):
            content_paths.append(v4a_path)
    for _m in _V4A_MOVE_HEADER_RE.finditer(patch):
        for v4a_path in (_m.group(2).strip(), _m.group(3).strip()):
            _err = _reject_v4a_traversal(v4a_path)
            if _err:
                return _err
            paths.append(v4a_path)
    return paths, content_paths


def patch_tool(mode: str = "replace", path: str = None, old_string: str = None,
               new_string: str = None, replace_all: bool = False, patch: str = None,
               task_id: str = "default", cross_profile: bool = False,
               session_id: str | None = None) -> str:
    """Patch a file using replace mode or V4A patch format.

    ``cross_profile``: same semantics as ``write_file``'s flag (mirror-guard
    bypass only; unadvertised).
    """
    _paths_to_check = [path] if path else []
    _content_write_paths = list(_paths_to_check)
    if mode == "patch" and patch:
        collected = _collect_v4a_header_paths(patch)
        if isinstance(collected, str):
            return collected
        _paths_to_check += collected[0]
        _content_write_paths += collected[1]
    precheck_err = _write_precheck_error(_paths_to_check, _content_write_paths, task_id, cross_profile)
    if precheck_err:
        return tool_error(precheck_err)
    try:
        # Lock paths in sorted, deduplicated order so concurrent callers with
        # overlapping multi-file patches can't deadlock (every caller locks in
        # the same order). An unresolvable path is simply not locked.
        _path_to_resolved: dict[str, str] = {
            _p: _resolve_or_none(_p, task_id) for _p in _paths_to_check
        }
        _resolved_paths = sorted({_r for _r in _path_to_resolved.values() if _r})

        # ExitStack: one lock per path; degenerates to a single lock (or none
        # for an unresolvable path) without special-casing.
        from contextlib import ExitStack
        with ExitStack() as _locks:
            for _r in _resolved_paths:
                _locks.enter_context(file_state.lock_path(_r))

            # Warning priority per path: cross-agent registry (names the
            # sibling) > per-task staleness > workspace divergence.
            stale_warnings: list[str] = []
            for _p in _paths_to_check:
                _r = _path_to_resolved[_p]
                _cross = file_state.check_stale(task_id, _r) if _r else None
                _sw = _cross or _check_file_staleness(_p, task_id)
                if not _sw and _r:
                    _sw = _path_resolution_warning(_p, Path(_r), task_id)
                if _sw:
                    stale_warnings.append(_sw)

            file_ops = _get_file_ops(task_id)

            # Hand the shell layer the RESOLVED targets so both layers agree on
            # which file is edited even when the shell's cwd differs.
            if mode == "replace":
                if not path:
                    return tool_error("path required")
                if old_string is None or new_string is None:
                    return tool_error("old_string and new_string required")
                _replace_target = _path_to_resolved.get(path) or path
                result = file_ops.patch_replace(_replace_target, old_string, new_string, replace_all)
            elif mode == "patch":
                if not patch:
                    return tool_error("patch content required")
                patch_for_ops = _rewrite_v4a_patch_paths_for_host(
                    patch, _path_to_resolved, file_ops
                )
                result = file_ops.patch_v4a(patch_for_ops)
            else:
                return tool_error(f"Unknown mode: {mode}")

            result_dict = result.to_dict()
            if stale_warnings:
                result_dict["_warning"] = stale_warnings[0] if len(stale_warnings) == 1 else " | ".join(stale_warnings)
            # Report the ABSOLUTE path(s) actually patched so a wrong-cwd
            # mismatch is visible instead of silently landing elsewhere.
            _resolved_modified = [
                _path_to_resolved.get(_p) or _p for _p in _paths_to_check
            ]
            if not result_dict.get("error"):
                result_dict["files_modified"] = _resolved_modified
                if len(_resolved_modified) == 1:
                    result_dict["resolved_path"] = _resolved_modified[0]
                _mark_verification_stale(task_id, _resolved_modified, session_id=session_id)
                # Refresh stamps for every patched path (no false staleness on
                # the next edit) and clear failure counters so a future miss
                # starts a fresh count.
                for _p in _paths_to_check:
                    _update_read_timestamp(_p, task_id)
                    _r = _path_to_resolved.get(_p)
                    if _r:
                        file_state.note_write(task_id, _r)
                _reset_patch_failures(task_id, [
                    _r for _r in (_path_to_resolved.get(_p) for _p in _paths_to_check) if _r
                ])
        # old_string-not-found hint. Per-file failure escalation is tracked for
        # replace mode only (V4A failures are rare; the generic hint suffices).
        # The generic hint is suppressed when patch_replace already attached a
        # richer "Did you mean?" snippet. The escalating hint after 3 failures
        # exists so the model recognises the loop and changes approach.
        if result_dict.get("error") and "Could not find" in str(result_dict["error"]):
            failure_count = 0
            if mode == "replace" and path:
                resolved = _path_to_resolved.get(path) or path
                failure_count = _record_patch_failure(task_id, resolved)

            if failure_count >= 3:
                result_dict["_hint"] = (
                    f"This is failure #{failure_count} patching {path!r}. "
                    "Stop retrying with variations of the same old_string. "
                    "Either: (1) re-read the file fresh to verify current "
                    "content, (2) use a longer / more unique old_string with "
                    "surrounding context lines, or (3) use write_file to "
                    "replace the entire file if the targeted region is hard "
                    "to anchor."
                )
            elif "Did you mean one of these sections?" not in str(result_dict["error"]):
                result_dict["_hint"] = (
                    "old_string not found. Use read_file to verify the current "
                    "content, or search_files to locate the text."
                )
        return json.dumps(result_dict, ensure_ascii=False)
    except Exception as e:
        return tool_error(str(e))


def search_tool(pattern: str, target: str = "content", path: str = ".",
                file_glob: str = None, limit: int = 50, offset: int = 0,
                output_mode: str = "content", context: int = 0,
                task_id: str = "default") -> str:
    """Search for content or files."""
    try:
        offset, limit = normalize_search_pagination(offset, limit)

        # Pagination args are part of the key so paging through truncated
        # results doesn't trip the repeated-search guard.
        search_key = (
            "search",
            pattern,
            target,
            str(path),
            file_glob or "",
            limit,
            offset,
        )
        with _read_tracker_lock:
            task_data = _read_tracker.setdefault(task_id, {
                "last_key": None, "consecutive": 0, "read_history": set(),
            })
            count = _bump_consecutive(task_data, search_key)

        if count >= 4:
            return tool_error(
                f"BLOCKED: You have run this exact search {count} times in a row. "
                "The results have NOT changed. You already have this information. "
                "STOP re-searching and proceed with your task.",
                pattern=pattern,
                already_searched=count,
            )

        try:
            resolved_path = _resolve_path_for_task(path, task_id)
        except (OSError, ValueError, RuntimeError):
            resolved_path = None
        block_error = get_read_block_error(str(resolved_path) if resolved_path else path)
        if block_error:
            return tool_error(block_error)

        # A missing search root costs two shells (search + parent listing for
        # "Similar paths"); cache the miss so a retry skips both.
        try:
            resolved_search_path = str(_resolve_path_for_task(path, task_id))
        except (OSError, ValueError):
            resolved_search_path = path
        cached_search_nf = _check_not_found_cache("search", resolved_search_path, task_id)
        if cached_search_nf is not None:
            return cached_search_nf

        file_ops = _get_file_ops(task_id)
        result = file_ops.search(
            pattern=pattern, path=path, target=target, file_glob=file_glob,
            limit=limit, offset=offset, output_mode=output_mode, context=context
        )
        omitted = _filter_read_blocked_search_results(result, task_id)
        if hasattr(result, 'matches'):
            for m in result.matches:
                if hasattr(m, 'content') and m.content:
                    m.content = redact_sensitive_text(m.content, file_read=True)
        result_dict = result.to_dict(densify=True)

        if omitted:
            result_dict["_omitted"] = (
                f"{omitted} result(s) omitted because they target credential, "
                "token, cache, or secret-bearing environment files."
            )

        # No early return on a cached miss — same rationale as the read path.
        _search_err = result_dict.get("error") or ""
        if isinstance(_search_err, str) and _search_err.startswith("Path not found:"):
            _record_not_found("search", resolved_search_path, task_id, json.dumps(result_dict, ensure_ascii=False))

        if count >= 3:
            result_dict["_warning"] = (
                f"You have run this exact search {count} times consecutively. "
                "The results have not changed. Use the information you already have."
            )

        result_json = json.dumps(result_dict, ensure_ascii=False)
        if result_dict.get("truncated"):
            next_offset = offset + limit
            result_json += f"\n\n[Hint: Results truncated. Use offset={next_offset} to see more, or narrow with a more specific pattern or file_glob.]"
        return result_json
    except Exception as e:
        return tool_error(str(e))


# ---------------------------------------------------------------------------
# Schemas + Registry
# ---------------------------------------------------------------------------
from tools.registry import registry, tool_error


def _check_file_reqs():
    """Lazy wrapper to avoid circular import with tools/__init__.py."""
    from tools import check_file_requirements
    return check_file_requirements()

READ_FILE_SCHEMA = {
    "name": "read_file",
    # Document formats are stated unconditionally: firecrawl-anydoc is a
    # core dependency (bundled), so its absence is a broken install, not a
    # configuration — the teaching error in read_extract handles that rare
    # case with the pip-install fix. The ONE dynamic word: "PDF (text
    # layer)" upgrades to "PDF (scanned or text)" when hosted OCR has a
    # route we trust (_read_file_schema_overrides). Scanned-page coverage
    # teaching lives in the response-time NEEDS-OCR warning
    # (read_extract.py); the schema doesn't pre-teach it.
    "description": "Read a text file with line numbers and pagination. Use this instead of cat/head/tail in terminal. Output format: 'LINE_NUM|CONTENT'. Suggests similar filenames if not found. Use offset and limit for large files. Reads exceeding ~100K characters are truncated on a line boundary and return a next_offset; continue with offset to read the rest. Documents auto-extract to readable text: .ipynb, Office (.docx/.xlsx/.pptx and legacy .doc/.ppt/.xls), PDF (text layer), OpenDocument, RTF, EPUB. Cannot read images/binary — use vision_analyze for images.",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file to read (absolute, relative, or ~/path)"},
            "offset": {"type": "integer", "description": "Line number to start reading from (1-indexed, default: 1)", "default": 1, "minimum": 1},
            "limit": {"type": "integer", "description": "Maximum number of lines to read (default: 2000, max: 2000). Reads are additionally capped at a ~100K-character budget with a next_offset continuation.", "default": 2000, "maximum": 2000}
        },
        "required": ["path"]
    }
}

WRITE_FILE_SCHEMA = {
    "name": "write_file",
    "description": "Write content to a file, completely replacing existing content. Use this instead of echo/cat heredoc in terminal. Creates parent directories automatically. OVERWRITES the entire file — use 'patch' for targeted edits. Auto-runs syntax checks on .py/.json/.yaml/.toml and other linted languages; only NEW errors introduced by this write are surfaced (pre-existing errors are filtered out). The result's verified:true means the on-disk content hash was confirmed — do NOT re-read the file to check the write landed.",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file to write (will be created if it doesn't exist, overwritten if it does)"},
            "content": {"type": "string", "description": "Complete content to write to the file"},
            # NOTE: the handler still accepts `cross_profile` (bool) — it now
            # bypasses only the #32049 sandbox-mirror lost-write guards, whose
            # rejection error teaches it. Unadvertised: the cross-PROFILE
            # guard it was named for was removed (profiles are not isolated,
            # maintainer decision), and mirror hits are rare + self-teaching.
        },
        "required": ["path", "content"]
    }
}

PATCH_SCHEMA = {
    "name": "patch",
    # BASE = replace-only (what nearly every model family was trained on).
    # The V4A patch mode (mode + patch params, dual-mode description) is
    # LAYERED ON dynamically for OpenAI-family mains only — V4A is the
    # OpenAI apply_patch dialect their models emit natively; advertising
    # it to everyone cost every other session ~148 tok/call
    # (_patch_schema_overrides below). The handler accepts BOTH shapes
    # from any model regardless (replay compat + strong models that know
    # V4A anyway): mode defaults to 'replace' when omitted.
    "description": (
        "Targeted find-and-replace edits in files. Use this instead of sed/awk in terminal. "
        "Uses fuzzy matching (9 strategies) so minor whitespace/indentation differences won't break it. "
        "Returns a unified diff. Auto-runs syntax checks after editing. "
        "Finds a unique string and replaces it."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "File path to edit.",
            },
            "old_string": {
                "type": "string",
                "description": "Exact text to find and replace. Must be unique in the file unless replace_all=true. Include surrounding context lines to ensure uniqueness.",
            },
            "new_string": {
                "type": "string",
                "description": "Changed replacement text; it must differ from old_string. Pass empty string '' to delete the matched text.",
            },
            "replace_all": {
                "type": "boolean",
                "description": "Replace all occurrences instead of requiring a unique match (default: false)",
                "default": False,
            },
            # NOTE: handler still accepts `cross_profile` — see write_file's
            # NOTE (mirror-guard bypass only; unadvertised by design).
            # NOTE: handler still accepts `mode` + `patch` (V4A) from ANY
            # model — the schema just doesn't advertise them off-family.
        },
        "required": ["path", "old_string", "new_string"],
    },
}


# V4A layer, rendered only for OpenAI-family main models (see PATCH_SCHEMA
# comment). Kept as data so the override composes it deterministically.
_PATCH_V4A_DESCRIPTION = (
    "Targeted find-and-replace edits in files. Use this instead of sed/awk in terminal. "
    "Uses fuzzy matching (9 strategies) so minor whitespace/indentation differences won't break it. "
    "Returns a unified diff. Auto-runs syntax checks after editing.\n\n"
    "REPLACE MODE (mode='replace', default): find a unique string and replace it. "
    "REQUIRED PARAMETERS: mode, path, old_string, new_string.\n"
    "PATCH MODE (mode='patch'): apply V4A multi-file patches for bulk changes. "
    "REQUIRED PARAMETERS: mode, patch."
)

_PATCH_V4A_PARAMS = {
    "mode": {
        "type": "string",
        "enum": ["replace", "patch"],
        "description": "Edit mode. 'replace' (default): requires path + old_string + new_string. 'patch': requires patch content only.",
        "default": "replace",
    },
    "patch": {
        "type": "string",
        "description": "REQUIRED when mode='patch'. V4A format patch content. Format:\n*** Begin Patch\n*** Update File: path/to/file\n@@ context hint @@\n context line\n-removed line\n+added line\n*** End Patch",
    },
}


def _is_openai_family_main() -> bool:
    """Whether the active main provider/model is the OpenAI/codex family —
    the population trained on the V4A apply_patch dialect.

    Provider-family-coarse on purpose (no per-model training-diet table to
    go stale): direct OpenAI providers always qualify; on aggregators
    (openrouter/nous/azure...) the MODEL slug decides (gpt-*/o-series/
    codex). Fail-closed to the universal replace-only schema.
    """
    try:
        from agent.auxiliary_client import _read_main_model, _read_main_provider

        provider = (_read_main_provider() or "").strip().lower()
        model = (_read_main_model() or "").strip().lower()
    except Exception:  # noqa: BLE001
        return False
    if provider in {"openai", "openai-chat", "openai-codex", "azure-openai", "codex"}:
        return True
    # Aggregators: the model slug carries the family.
    slug = model.split("/", 1)[-1]
    if slug.startswith(("gpt-", "gpt.", "chatgpt", "codex", "o1", "o3", "o4", "o5")):
        return True
    return "openai/" in model


SEARCH_FILES_SCHEMA = {
    "name": "search_files",
    "description": "Search file contents or find files by name. Use this instead of grep/rg/find/ls in terminal. Ripgrep-backed, faster than shell equivalents. On macOS, broad searches above the user home automatically skip TCC-protected folders (Desktop, Documents, Downloads, Library, Movies, Music, Pictures); target one directly when access is intentional.\n\nContent search (target='content'): Regex search inside files. Output modes: full matches with line numbers, file paths only, or match counts.\n\nFile search (target='files'): Find files by glob pattern (e.g., '*.py', '*config*'). Also use this instead of ls — results sorted by modification time.",
    "parameters": {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Regex pattern for content search, or glob pattern (e.g., '*.py') for file search"},
            "target": {"type": "string", "enum": ["content", "files"], "description": "'content' searches inside file contents, 'files' searches for files by name", "default": "content"},
            "path": {"type": "string", "description": "Directory or file to search in (default: current working directory)", "default": "."},
            "file_glob": {"type": "string", "description": "Filter files by pattern in grep mode (e.g., '*.py' to only search Python files)"},
            "limit": {"type": "integer", "description": "Maximum number of results to return (default: 50)", "default": 50},
            "offset": {"type": "integer", "description": "Skip first N results for pagination (default: 0)", "default": 0},
            "output_mode": {"type": "string", "enum": ["content", "files_only", "count"], "description": "Output format for grep mode: 'content' shows matching lines with line numbers, 'files_only' lists file paths, 'count' shows match counts per file", "default": "content"},
            "context": {"type": "integer", "description": "Number of context lines before and after each match (grep mode only)", "default": 0}
        },
        "required": ["pattern"]
    }
}


def _handle_read_file(args, **kw):
    tid = kw.get("task_id") or "default"
    return read_file_tool(path=args.get("path", ""), offset=args.get("offset", 1), limit=args.get("limit", 500), task_id=tid)


def _handle_write_file(args, **kw):
    tid = kw.get("task_id") or "default"
    if not args.get("path") or not isinstance(args.get("path"), str):
        return tool_error(
            "write_file: missing required field 'path'. Re-emit the tool call with "
            "both 'path' and 'content' set."
        )
    if "content" not in args:
        return tool_error(
            "write_file: missing required field 'content'. The tool call included a "
            "path but no content argument — this is almost always a dropped-arg bug "
            "under context pressure. Re-emit the tool call with the full content "
            "payload, or use execute_code with hermes_tools.write_file() for very "
            "large files."
        )
    if not isinstance(args["content"], str):
        return tool_error(
            f"write_file: 'content' must be a string, got "
            f"{type(args['content']).__name__}."
        )
    return write_file_tool(
        path=args["path"], content=args["content"], task_id=tid,
        cross_profile=bool(args.get("cross_profile", False)),
        session_id=kw.get("session_id"),
    )


def _handle_patch(args, **kw):
    tid = kw.get("task_id") or "default"
    return patch_tool(
        mode=args.get("mode", "replace"), path=args.get("path"),
        old_string=args.get("old_string"), new_string=args.get("new_string"),
        replace_all=args.get("replace_all", False), patch=args.get("patch"), task_id=tid,
        cross_profile=bool(args.get("cross_profile", False)),
        session_id=kw.get("session_id"),
    )


def _handle_search_files(args, **kw):
    tid = kw.get("task_id") or "default"
    target_map = {"grep": "content", "find": "files"}
    raw_target = args.get("target", "content")
    target = target_map.get(raw_target, raw_target)
    return search_tool(
        pattern=args.get("pattern", ""), target=target, path=args.get("path", "."),
        file_glob=args.get("file_glob"), limit=args.get("limit", 50), offset=args.get("offset", 0),
        output_mode=args.get("output_mode", "content"), context=args.get("context", 0), task_id=tid)


def _read_file_schema_overrides():
    """One-word capability upgrade: "PDF (text layer)" → "PDF (scanned or
    text)" when hosted OCR has a trusted route (see
    read_extract.hosted_ocr_available). Config/env probe only — no
    network at schema-build time. Compaction's tool refresh (#97073)
    picks up a key added mid-session.
    """
    try:
        from tools.read_extract import hosted_ocr_available

        if hosted_ocr_available():
            return {
                "description": READ_FILE_SCHEMA["description"].replace(
                    "PDF (text layer)", "PDF (scanned or text)"
                )
            }
    except Exception:  # noqa: BLE001
        pass
    return {}


registry.register(name="read_file", toolset="file", schema=READ_FILE_SCHEMA, handler=_handle_read_file, check_fn=_check_file_reqs, emoji="📖", max_result_size_chars=100_000, dynamic_schema_overrides=_read_file_schema_overrides)
registry.register(name="write_file", toolset="file", schema=WRITE_FILE_SCHEMA, handler=_handle_write_file, check_fn=_check_file_reqs, emoji="✍️", max_result_size_chars=100_000)
def _patch_schema_overrides():
    """Layer the V4A patch mode onto the base replace-only schema for
    OpenAI-family mains (see PATCH_SCHEMA comment). Config/context probe
    only — no I/O at schema-build time; compaction's tool refresh
    (#97073) re-evaluates on model switches."""
    try:
        if not _is_openai_family_main():
            return {}
        params = {
            "type": "object",
            "properties": {
                "mode": _PATCH_V4A_PARAMS["mode"],
                **PATCH_SCHEMA["parameters"]["properties"],
                "patch": _PATCH_V4A_PARAMS["patch"],
            },
            "required": ["mode"],
        }
        return {"description": _PATCH_V4A_DESCRIPTION, "parameters": params}
    except Exception:  # noqa: BLE001
        return {}


registry.register(name="patch", toolset="file", schema=PATCH_SCHEMA, handler=_handle_patch, check_fn=_check_file_reqs, emoji="🔧", max_result_size_chars=100_000, dynamic_schema_overrides=_patch_schema_overrides)
registry.register(name="search_files", toolset="file", schema=SEARCH_FILES_SCHEMA, handler=_handle_search_files, check_fn=_check_file_reqs, emoji="🔎", max_result_size_chars=100_000)
