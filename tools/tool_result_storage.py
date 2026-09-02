"""Tool result persistence -- preserves large outputs instead of truncating.

Three layers of defense against context-window overflow:

1. **Per-tool output cap** (inside each tool, e.g. search_files pre-truncates).
2. **Per-result persistence** (``maybe_persist_tool_result``): output over the
   tool's registered threshold is persisted and replaced in-context by a
   preview + file path. The canonical home is ALWAYS host-side,
   ``$HERMES_HOME/cache/spillover/{tool_use_id}.txt``, so it works for sessions
   that never ran a terminal (MCP-only, cron, gateway). Local backend / no env:
   the model sees the host path. Remote backends (docker/ssh/modal/daytona):
   ``cache/spillover`` is in the auto-mounted/synced cache-dir list
   (tools/credential_files.py), so the model sees the translated in-sandbox
   path — probed for readability first, falling back to a copy written into
   the sandbox temp dir (e.g. a persistent container created before spillover
   joined the mount list). The dir is pruned hourly by gateway housekeeping
   and once per process on first spill (CLI-only installs never run housekeeping).
3. **Per-turn aggregate budget** (``enforce_turn_budget``): if all tool results
   in one turn exceed the turn budget, the largest non-persisted results are
   spilled until under budget.
"""

import hashlib
import logging
import os
import re
import shlex
import threading
import time

from tools.budget_config import (
    DEFAULT_PREVIEW_SIZE_CHARS,
    BudgetConfig,
    DEFAULT_BUDGET,
)

logger = logging.getLogger(__name__)
PERSISTED_OUTPUT_TAG = "<persisted-output>"
PERSISTED_OUTPUT_CLOSING_TAG = "</persisted-output>"
STORAGE_DIR = "/tmp/hermes-results"
SPILLOVER_SUBDIR = "cache/spillover"
SPILLOVER_MAX_AGE_HOURS = 24
_BUDGET_TOOL_NAME = "__budget_enforcement__"
_UNSAFE_RESULT_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9_.-]+")
_MAX_RESULT_FILENAME_STEM = 120

_spillover_prune_lock = threading.Lock()
_spillover_pruned_once = False


def get_spillover_dir():
    """Return $HERMES_HOME/cache/spillover as a Path (not created)."""
    from hermes_constants import get_hermes_home

    return get_hermes_home() / SPILLOVER_SUBDIR


def cleanup_spillover_cache(max_age_hours: int = SPILLOVER_MAX_AGE_HOURS) -> int:
    """Delete spillover files older than *max_age_hours*; returns count removed.

    Same contract as the ``cleanup_*_cache`` helpers in ``gateway.platforms.base``
    so the gateway housekeeping loop can prune this dir on the hourly cadence.
    """
    cutoff = time.time() - (max_age_hours * 3600)
    removed = 0
    try:
        entries = list(get_spillover_dir().iterdir())
    except OSError:
        return 0
    for f in entries:
        try:
            if f.is_file() and f.stat().st_mtime < cutoff:
                f.unlink()
                removed += 1
        except OSError:
            continue
    return removed


def _prune_spillover_once() -> None:
    """Best-effort prune, at most once per process (CLI-only installs never run housekeeping)."""
    global _spillover_pruned_once
    with _spillover_prune_lock:
        if _spillover_pruned_once:
            return
        _spillover_pruned_once = True
    try:
        removed = cleanup_spillover_cache()
        if removed:
            logger.debug("Pruned %d expired spillover file(s)", removed)
    except Exception as exc:
        logger.debug("Spillover prune failed: %s", exc)


def _is_host_side_env(env) -> bool:
    """True when this process should write the spill file directly.

    ``env=None`` (no sandbox yet) and the local backend qualify; remote backends
    resolve ``read_file`` inside the sandbox, so the spill must be written there.
    """
    if env is None:
        return True
    try:
        from tools.environments.local import LocalEnvironment

        return isinstance(env, LocalEnvironment)
    except Exception:
        return False


def _write_to_spillover(content: str, filename: str):
    """Write host-side to $HERMES_HOME/cache/spillover; returns path str or None."""
    try:
        spill_dir = get_spillover_dir()
        spill_dir.mkdir(parents=True, exist_ok=True)
        path = spill_dir / filename
        path.write_text(content, encoding="utf-8", errors="replace")
    except OSError as exc:
        logger.warning("Spillover write failed for %s: %s", filename, exc)
        return None
    _prune_spillover_once()
    return str(path)


def _sandbox_visible_spillover_path(host_path: str, env) -> str | None:
    """Return the path where a remote backend can read *host_path*, or None.

    Translates via the same helper the image tools use, forces a sync for synced
    backends, then PROBES readability — a persistent docker container created
    before spillover joined the mount list lacks the bind mount and must fall
    back to the in-sandbox write.
    """
    try:
        from tools.credential_files import to_agent_visible_cache_path

        visible = to_agent_visible_cache_path(host_path)
    except Exception as exc:
        logger.debug("Spillover path translation failed: %s", exc)
        return None

    sync_manager = getattr(env, "_sync_manager", None)
    if sync_manager is not None:
        try:
            sync_manager.sync(force=True)
        except Exception as exc:
            logger.debug("Spillover sync failed: %s", exc)

    try:
        result = env.execute(f"test -r {shlex.quote(visible)}", timeout=15)
        if result.get("returncode", 1) == 0:
            return visible
    except Exception as exc:
        logger.debug("Spillover readability probe failed: %s", exc)
    return None


def _resolve_storage_dir(env) -> str:
    """Return the best temp-backed storage dir for this environment."""
    if env is not None:
        get_temp_dir = getattr(env, "get_temp_dir", None)
        if callable(get_temp_dir):
            try:
                temp_dir = get_temp_dir()
            except Exception as exc:
                logger.debug("Could not resolve env temp dir: %s", exc)
            else:
                if temp_dir:
                    temp_dir = temp_dir.rstrip("/") or "/"
                    return f"{temp_dir}/hermes-results"
    return STORAGE_DIR


def _safe_result_filename(tool_use_id: str) -> str:
    """Return a single safe filename for a tool result id."""
    raw_id = str(tool_use_id or "tool_result")
    safe_stem = _UNSAFE_RESULT_FILENAME_CHARS.sub("_", raw_id).strip("._-")
    changed = safe_stem != raw_id

    if not safe_stem:
        safe_stem = "tool_result"
        changed = True

    if changed or len(safe_stem) > _MAX_RESULT_FILENAME_STEM:
        digest = hashlib.sha256(raw_id.encode("utf-8")).hexdigest()[:12]
        safe_stem = safe_stem[:_MAX_RESULT_FILENAME_STEM].rstrip("._-") or "tool_result"
        safe_stem = f"{safe_stem}_{digest}"

    return f"{safe_stem}.txt"


def generate_preview(content: str, max_chars: int = DEFAULT_PREVIEW_SIZE_CHARS) -> tuple[str, bool]:
    """Truncate at last newline within max_chars. Returns (preview, has_more)."""
    if len(content) <= max_chars:
        return content, False
    truncated = content[:max_chars]
    last_nl = truncated.rfind("\n")
    if last_nl > max_chars // 2:
        truncated = truncated[:last_nl + 1]
    return truncated, True


def _write_to_sandbox(content: str, remote_path: str, env) -> bool:
    """Write content into the sandbox via env.execute(). Returns True on success.

    Content goes through stdin, not the command string: Linux ``MAX_ARG_STRLEN``
    caps a single argv element at 128 KB, so a heredoc-in-command silently failed
    for exactly the oversized results persistence exists to handle.
    """
    storage_dir = os.path.dirname(remote_path)
    cmd = f"mkdir -p {shlex.quote(storage_dir)} && cat > {shlex.quote(remote_path)}"
    result = env.execute(cmd, timeout=30, stdin_data=content)
    return result.get("returncode", 1) == 0


def _build_persisted_message(
    preview: str,
    has_more: bool,
    original_size: int,
    file_path: str,
) -> str:
    """Build the <persisted-output> replacement block."""
    size_kb = original_size / 1024
    size_str = f"{size_kb / 1024:.1f} MB" if size_kb >= 1024 else f"{size_kb:.1f} KB"

    msg = f"{PERSISTED_OUTPUT_TAG}\n"
    msg += f"This tool result was too large ({original_size:,} characters, {size_str}).\n"
    msg += f"Full output saved to: {file_path}\n"
    msg += "Use the read_file tool with offset and limit to access specific sections of this output.\n"
    msg += (
        "Recovery: page through the saved file with read_file (offset/limit) or "
        "process it with execute_code — do NOT re-request the same data from the "
        "remote API; the full result is already on disk.\n\n"
    )
    msg += f"Preview (first {len(preview)} chars):\n"
    msg += preview
    if has_more:
        msg += "\n..."
    msg += f"\n{PERSISTED_OUTPUT_CLOSING_TAG}"
    return msg


_PERSISTED_PATH_RE = re.compile(r"^Full output saved to: (.+)$", re.MULTILINE)


def extract_persisted_path(content: str) -> str | None:
    """Return the file path from a <persisted-output> block, or None.

    Lets the result-reference stubbing guard (agent/tool_guardrails.py) carry
    the spillover path in a stub instead of leaving it dangling.
    """
    if not isinstance(content, str) or PERSISTED_OUTPUT_TAG not in content:
        return None
    match = _PERSISTED_PATH_RE.search(content)
    return match.group(1).strip() if match else None


def maybe_persist_tool_result(
    content: str,
    tool_name: str,
    tool_use_id: str,
    env=None,
    config: BudgetConfig = DEFAULT_BUDGET,
    threshold: int | float | None = None,
) -> str:
    """Layer 2: persist an oversized result, return preview + path.

    ``threshold`` overrides ``config.resolve_threshold(tool_name)``. Falls back
    to inline truncation when no write location succeeds.
    """
    effective_threshold = threshold if threshold is not None else config.resolve_threshold(tool_name)

    if effective_threshold == float("inf"):
        return content

    if len(content) <= effective_threshold:
        return content

    filename = _safe_result_filename(tool_use_id)
    preview, has_more = generate_preview(content, max_chars=config.preview_size)

    def _persisted(path: str, host_suffix: str = "") -> str:
        logger.info(
            "Persisted large tool result: %s (%s, %d chars -> %s%s)",
            tool_name, tool_use_id, len(content), path, host_suffix,
        )
        return _build_persisted_message(preview, has_more, len(content), path)

    # Always persist host-side first: cache/spillover is the single canonical
    # home for spilled results regardless of backend.
    host_path = _write_to_spillover(content, filename)

    if _is_host_side_env(env):
        if host_path is not None:
            return _persisted(host_path)
    elif env is not None:
        # Remote backend: reference the mounted/synced path when the sandbox
        # can actually read it ...
        if host_path is not None:
            visible = _sandbox_visible_spillover_path(host_path, env)
            if visible is not None:
                return _persisted(visible, f" [host: {host_path}]")
        # ... else write into the sandbox temp dir (pre-existing containers
        # without the spillover mount, translation/probe failures).
        remote_path = f"{_resolve_storage_dir(env)}/{filename}"
        try:
            if _write_to_sandbox(content, remote_path, env):
                return _persisted(remote_path)
        except Exception as exc:
            logger.warning("Sandbox write failed for %s: %s", tool_use_id, exc)

    logger.info(
        "Inline-truncating large tool result: %s (%d chars, no sandbox write)",
        tool_name, len(content),
    )
    return (
        f"{preview}\n\n"
        f"[Truncated: tool response was {len(content):,} chars. "
        f"Full output could not be saved to sandbox.]"
    )


def enforce_turn_budget(
    tool_messages: list[dict],
    env=None,
    config: BudgetConfig = DEFAULT_BUDGET,
) -> list[dict]:
    """Layer 3: enforce the aggregate budget across all tool results in a turn.

    Persists the largest non-persisted results first until under budget.
    Mutates the list in-place and returns it.
    """
    candidates = []
    total_size = 0
    for i, msg in enumerate(tool_messages):
        content = msg.get("content", "")
        size = len(content)
        total_size += size
        if PERSISTED_OUTPUT_TAG not in content:
            candidates.append((i, size))

    if total_size <= config.turn_budget:
        return tool_messages

    candidates.sort(key=lambda x: x[1], reverse=True)

    for idx, size in candidates:
        if total_size <= config.turn_budget:
            break
        msg = tool_messages[idx]
        content = msg["content"]
        tool_use_id = msg.get("tool_call_id", f"budget_{idx}")

        replacement = maybe_persist_tool_result(
            content=content,
            tool_name=_BUDGET_TOOL_NAME,
            tool_use_id=tool_use_id,
            env=env,
            config=config,
            threshold=0,
        )
        if replacement != content:
            total_size -= size
            total_size += len(replacement)
            tool_messages[idx]["content"] = replacement
            logger.info(
                "Budget enforcement: persisted tool result %s (%d chars)",
                tool_use_id, size,
            )

    return tool_messages
