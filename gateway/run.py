"""Gateway runner - entry point for messaging platform integrations.

Provides ``start_gateway()`` (start all configured adapters) and ``GatewayRunner`` (lifecycle).
Run via ``python -m gateway.run`` or ``python cli.py --gateway``.
"""

# hermes_bootstrap must be the very first import (UTF-8 stdio on Windows; no-op on POSIX).
try:
    import hermes_bootstrap  # noqa: F401
except ModuleNotFoundError:
    # A partial ``hermes update`` can leave the bootstrap unregistered; only Windows UTF-8 stdio suffers.
    pass

import asyncio
import concurrent.futures
import dataclasses
import json
import logging
import os
import re
import shlex
import site
import sys
import signal
import threading
import time
import traceback
from collections import OrderedDict
from contextvars import copy_context
from pathlib import Path
from datetime import datetime
from typing import Callable, Dict, Optional, Any, List, Tuple, cast

from agent.async_utils import safe_schedule_threadsafe
from agent.conversation_compression import (
    COMPACTION_DONE_STATUS, COMPACTION_STATUS, COMPRESSION_RETRY_CONTEXT_REDUCED_STATUS_TEMPLATE,
    COMPRESSION_RETRY_MESSAGES_STATUS_TEMPLATE, COMPRESSION_RETRY_TOKENS_STATUS_TEMPLATE,
    COMPRESSION_RETRY_TOO_LARGE_STATUS_TEMPLATE, IDLE_COMPACTION_STATUS_TEMPLATE,
    PRE_API_COMPRESSION_STATUS_TEMPLATE, PREFLIGHT_COMPRESSION_STATUS_TEMPLATE)
from agent.conversation_loop import INTERRUPT_WAITING_FOR_MODEL_PREFIX
from agent.interrupt_compat import request_hard_interrupt
from agent.turn_context import compression_made_progress
from hermes_cli.config import _is_ssh_remote_tilde_cwd, cfg_get
from hermes_cli.fallback_config import get_fallback_chain

# Per-session AIAgent cache bounds (agents are heavy); enforced by
# _enforce_agent_cache_cap()/_session_expiry_watcher().
_AGENT_CACHE_MAX_SIZE = 128
_AGENT_CACHE_IDLE_TTL_SECS = 3600.0  # evict agents idle for >1h
_PLATFORM_CONNECT_TIMEOUT_SECS_DEFAULT = 30.0
# Telegram connect proves a real getUpdates round trip; must cover polling-start deadlines + readiness.
_TELEGRAM_CONNECT_TIMEOUT_SECS_DEFAULT = 180.0
# The initial Telegram connect gates `running` for EVERY platform, so it must not spend the full 180s.
_TELEGRAM_INITIAL_CONNECT_TIMEOUT_SECS_DEFAULT = 45.0
_ADAPTER_DISCONNECT_TIMEOUT_SECS_DEFAULT = 5.0
# End reasons meaning the USER deliberately closed this thread. Shared by _classify_completion_target and
# _resolve_async_delegation_session so they never disagree (a "delivered" reason the resolver drops is
# acked and then silently lost).
_USER_BOUNDARY_END_REASONS = ("session_reset", "user_exit", "session_switch", "new_session")
# Bounds one stall-notify send so a wedged transport can't block the watcher; on timeout the next tick
# retries.
_STALL_NOTIFY_SEND_TIMEOUT_SECONDS = 15.0
_GATEWAY_PROXY_SSE_BUFFER_MAX_CHARS = 16 * 1024 * 1024
_TELEGRAM_COMMAND_MENTION_RE = re.compile(r"(?<![\w:/])/([A-Za-z0-9][A-Za-z0-9_-]*)")
_GATEWAY_HYGIENE_PLATFORM = "gateway_hygiene"

_TELEGRAM_NOISY_STATUS_RE = re.compile(
    r"("  # transient/auxiliary status that should stay in logs, not gateway chats
    r"auxiliary\s+.+\s+failed"
    r"|compression\s+summary\s+failed"
    r"|fallback\s+context\s+marker"
    r"|configured\s+compression\s+model\s+.+\s+failed"
    r"|no\s+auxiliary\s+llm\s+provider\s+configured"
    r"|auto-lowered\s+compression\s+threshold"
    # the auto-lower notice was reworded to "Auto-lowered this session's threshold..." — cover both.
    r"|auto-lowered\s+(?:this\s+)?session'?s?\s+threshold"
    r"|configured\s+auxiliary\s+compression\s+provider\s+.+\s+unavailable"
    r"|skipping\s+concurrent\s+compression"
    r"|compacting\s+context\s+[—-]\s+summarizing\s+earlier\s+conversation"
    r"|resumed\s+after\s+\d+s\s+idle\s+[—-]\s+compacting"
    r"|preflight\s+compression"
    r"|pre[- ]api\s+compression"
    # Retry chatter replayed via _emit_status; the ", retrying"/"— compressing" anchors keep manual
    # /compress feedback and failure notices out of the match.
    r"|context\s+too\s+large\s+\(~[\d,]+\s+tokens\)\s+[—-]+\s+compressing"
    r"|compressed\s+\d[\d,]*\s+(?:→|->)\s+\d[\d,]*\s+messages,\s+retrying"
    r"|compressed\s+~[\d,]+\s+(?:→|->)\s+~[\d,]+\s+tokens,\s+retrying"
    r"|context\s+reduced\s+to\s+[\d,]+\s+tokens\s+\(was\s+[\d,]+\),\s+retrying"
    r"|session\s+compressed\s+\d+\s+times"
    r"|rate\s+limited\.\s+waiting\s+\d"
    r"|retrying\s+in\s+\d"
    r"|max\s+retries\s+\(\d+\).*(?:trying\s+fallback|exhausted|invalid\s+responses)"
    r"|stream\s+(?:drop|drop\s+mid\s+tool-call).+retry\s+\d"
    r"|stale\s+connections\s+from\s+a\s+previous\s+provider\s+issue"
    rf"|{re.escape(COMPACTION_DONE_STATUS)}"
    r")",
    re.IGNORECASE | re.DOTALL)

_HYGIENE_COOLDOWN_LADDER_MULTIPLIERS = (1, 3, 9)
# Ceiling on an escalated cooldown (cf. _RECONNECT_BACKOFF_CAP): an operator-raised base × ladder reaches
# 9h, indistinguishable from "compaction silently switched off".
_HYGIENE_COOLDOWN_MAX_SECONDS = 3600.0
# Flat retry-after when hygiene compression is ABANDONED by turn-hold expiry (not a failure, so
# outside the streak ladder); keeps sustained traffic from spawn/hold/cancelling one every turn.
_HYGIENE_TURNHOLD_RETRY_SECONDS = 60.0


def _gateway_session_db_inner(gateway):
    """The raw SessionDB behind ``gateway._session_db`` (unwrapping the async facade), or None."""
    session_db = getattr(gateway, "_session_db", None)
    return getattr(session_db, "_db", session_db)


def _hygiene_cooldown_for_failure(gateway, session_key: str, base_cooldown_seconds: float) -> float:
    """Bump the hygiene failure streak and return the escalated cooldown.

    Ladder (x1, x3, x9) over the base, clamped to the max. Hygiene's per-run ``AIAgent`` is fresh
    (in-memory streak always 0), so the streak lives in SQLite keyed by rotation-stable ``session_key``.
    """
    streak, state = 1, None
    try:
        state = gateway._session_state(session_key).persistent
    except Exception as exc:
        logger.debug("hygiene failure streak update failed: %s", exc)
    increment = getattr(_gateway_session_db_inner(gateway), "increment_hygiene_failure_streak", None)
    if callable(increment):
        try:
            streak = max(1, int(increment(session_key)))
            if state is not None:
                state.hygiene_failure_streak = streak
        except Exception as exc:
            logger.debug("hygiene failure streak persist failed: %s", exc)
            if state is not None:
                state.hygiene_failure_streak += 1
                streak = state.hygiene_failure_streak
    elif state is not None:
        state.hygiene_failure_streak += 1
        streak = state.hygiene_failure_streak
    multiplier = _HYGIENE_COOLDOWN_LADDER_MULTIPLIERS[
        min(streak, len(_HYGIENE_COOLDOWN_LADDER_MULTIPLIERS)) - 1]
    return min(base_cooldown_seconds * multiplier, _HYGIENE_COOLDOWN_MAX_SECONDS)


def _reset_hygiene_failure_streak(gateway, session_key: str) -> None:
    """Clear the hygiene failure streak after a compression that reduced context.

    Peeks, never get-or-creates: a no-op 0 write must not create a never-evicted ``_sessions`` row.
    """
    try:
        state = gateway._peek_session_state(session_key)
        if state is not None:
            state.persistent.hygiene_failure_streak = 0
    except Exception as exc:
        logger.debug("hygiene failure streak reset failed: %s", exc)
    reset = getattr(_gateway_session_db_inner(gateway), "reset_hygiene_failure_streak", None)
    if callable(reset):
        try:
            reset(session_key)
        except Exception as exc:
            logger.debug("hygiene failure streak persistent reset failed: %s", exc)


def hygiene_compaction_recovered(
    *, aborted: bool, rotated: bool, in_place: bool, msg_count: int, new_count: int,
    approx_tokens: int, new_tokens: int) -> bool:
    """True when a hygiene run actually recovered the session (extracted to be unit testable).

    Requires: no abort; transcript actually rewritten (the no-op path reuses pre-compression counts);
    and material shrink per :func:`compression_made_progress` (a bare ``<`` misses row-count wins and
    counts 30-50% estimate noise as one).
    """
    if aborted or not (rotated or in_place):
        return False
    return compression_made_progress(msg_count, new_count, approx_tokens, new_tokens)


def _hygiene_compression_timeout_message(
    *, total_exhausted: bool, elapsed: float, idle_timeout: float, progress_observed: bool) -> str:
    """Describe the host timeout that actually ended hygiene compression."""
    if total_exhausted:
        progress = " after summary output was observed" if progress_observed else ""
        return (
            "⚠️ Context compression reached its total ceiling after "
            f"{elapsed:.1f}s{progress}. No messages were dropped — continuing "
            "without compression. Run /compress to retry or /reset for a clean session.")
    return (
        f"⚠️ Context compression timed out after {idle_timeout:.1f}s with no "
        "output from the summary model. No messages were dropped — continuing "
        "without compression. Run /compress to retry, /reset for a clean "
        "session, or check your auxiliary.compression model configuration.")


def _cached_agent_for_hygiene(gateway, session_key: str):
    """The cached live AIAgent for ``session_key`` (or the pending sentinel / None), read under the cache lock."""
    cache = getattr(gateway, "_agent_cache", None)
    if cache is None:
        return None
    lock = getattr(gateway, "_agent_cache_lock", None)
    try:
        with (lock or suppress()):
            entry = cache.get(session_key)
    except Exception:
        entry = None
    return entry[0] if isinstance(entry, tuple) and entry else entry


async def run_codex_hygiene_compaction(
    gateway, session_key: str, session_id: str, *, auto_mode: str, history: list,
    approx_tokens: int, timeout_seconds: float, failure_cooldown_seconds: float = 300.0) -> str:
    """Session hygiene for ``codex_app_server`` sessions.

    The real context is the server-side thread; the local transcript is a never-replayed mirror, so
    rewriting it shrinks nothing and evicting the live agent starts the next turn on an EMPTY thread.
    So: compact the LIVE agent via ``thread/compact/start``, keep it cached, never build a detached
    compressor. ``native``/``off`` skip without falling back to the local compressor.
    Returns ``compacted``, ``skipped:<reason>`` or ``failed:<reason>``.
    """
    mode = str(auto_mode or "native").lower()
    if mode not in {"native", "hermes", "off"}:
        mode = "native"
    if mode != "hermes":
        # native = app-server compacts itself; off = operator disabled it. A local transcript
        # fallback cannot shrink the thread in any mode, so both skip cleanly with no eviction.
        return f"skipped:mode={mode}"

    agent = _cached_agent_for_hygiene(gateway, session_key)
    if agent is None or agent is _AGENT_PENDING_SENTINEL:
        # No live agent → no live thread; the detached mirror-only rewrite would be the very no-op
        # this function exists to remove.
        return "skipped:no-cached-agent"
    if getattr(agent, "_codex_session", None) is None:
        return "skipped:no-live-thread"

    compressor = getattr(agent, "context_compressor", None)
    count_before = getattr(compressor, "compression_count", 0)
    # copy_context keeps the caller's multiplexed profile secret scope and HERMES_HOME override in the
    # worker (the default executor does not propagate ContextVars on the runtimes Hermes ships).
    worker_future = asyncio.get_running_loop().run_in_executor(
        None, copy_context().run, lambda: agent._compress_context(history, "", approx_tokens=approx_tokens))
    track_worker = getattr(gateway, "_track_deferred_agent_worker", None)
    if callable(track_worker):
        # ``wait_for`` only cancels the asyncio wrapper; keep the still-running executor thread
        # visible to gateway shutdown.
        track_worker(worker_future, agent)
    try:
        await asyncio.wait_for(asyncio.shield(worker_future), timeout=max(float(timeout_seconds), 1.0))
    except asyncio.TimeoutError:
        # The executor thread keeps running (compact_thread has its own RPC timeouts); brake
        # retries so a wedged app-server does not re-trigger compaction on every message.
        if failure_cooldown_seconds >= 0:
            _record_hygiene_cooldown(
                gateway, session_id, failure_cooldown_seconds, "codex app-server thread compaction timed out")
        logger.warning(
            "Session hygiene: codex app-server thread compaction for "
            "session %s timed out after %.1fs; continuing without compaction",
            session_id, timeout_seconds)
        return "failed:timeout"
    except Exception as exc:
        logger.warning(
            "Session hygiene: codex app-server thread compaction for session %s failed: %s", session_id, exc)
        return f"failed:{exc}"

    count_after = getattr(compressor, "compression_count", 0)
    if count_after > count_before:
        # Native boundary recorded: compacted server-side; mirror intentionally NOT rewritten,
        # agent stays cached.
        _reset_hygiene_failure_streak(gateway, session_key)
        return "compacted"
    # No boundary: internal skip or compaction error; the codex route already persisted its own cooldown.
    return "failed:no-boundary"

def hygiene_wait_should_extend(
    *, idle: float, timeout: float, waited: float, ceiling: float, fence_cancelled: bool = False
) -> bool:
    """Whether the hygiene host should keep waiting for a slow summary.

    A cancelled commit fence cannot produce a commit: extending the wait only queues inbound
    messages behind a doomed attempt, so stop extending immediately.
    """
    return not fence_cancelled and idle < timeout and waited < ceiling


def _record_hygiene_cooldown(
    gateway, session_id: str, cooldown_seconds: float, error: Optional[str] = None) -> None:
    """Persist a session-hygiene compression-failure cooldown to the state DB.

    Shares the in-conversation path's column/recorder so it survives restarts. ``error`` must be
    forwarded: the recorder writes ``compression_failure_error`` UNCONDITIONALLY (else NULL clobber).
    """
    recorder = getattr(_gateway_session_db_inner(gateway), "record_compression_failure_cooldown", None)
    if recorder is None:
        return
    try:
        recorder(session_id, time.time() + cooldown_seconds, error)
    except Exception as exc:
        logger.debug("session hygiene cooldown persist failed: %s", exc)


def _status_template_to_regex(template: str) -> str:
    """Compile a compression status template constant into a regex source.

    Literal text is escaped verbatim so wording drift cannot diverge from the matcher;
    each ``{field}`` becomes a numeric-ish pattern.
    """
    parts = re.split(r"\{[^{}]*\}", template)
    return r"[\d,]+".join(re.escape(part) for part in parts)


# ROUTINE compression progress statuses, derived from the SAME template constants the emit sites format.
_COMPRESSION_PROGRESS_STATUS_RE = re.compile(
    "|".join(
        _status_template_to_regex(_template)
        for _template in (
            COMPACTION_STATUS, COMPACTION_DONE_STATUS, PRE_API_COMPRESSION_STATUS_TEMPLATE,
            PREFLIGHT_COMPRESSION_STATUS_TEMPLATE, IDLE_COMPACTION_STATUS_TEMPLATE,
            COMPRESSION_RETRY_TOO_LARGE_STATUS_TEMPLATE, COMPRESSION_RETRY_MESSAGES_STATUS_TEMPLATE,
            COMPRESSION_RETRY_TOKENS_STATUS_TEMPLATE,
            COMPRESSION_RETRY_CONTEXT_REDUCED_STATUS_TEMPLATE)),
    re.IGNORECASE)


def _gateway_compression_progress_notices_enabled() -> bool:
    """True when ``compression.progress_notices`` is on (default False: chat is silent by design).

    Read live (mtime-cached) so a config edit applies at the next status; fail-closed on read error.
    """
    try:
        config = _load_gateway_config()
        compression_cfg = config.get("compression") if isinstance(config, dict) else None
        if isinstance(compression_cfg, dict):
            return str(compression_cfg.get("progress_notices", False)).strip().lower() in {
                "true", "1", "yes", "on"}
    except Exception:
        pass
    return False

# Surfaces consuming gateway text programmatically must keep RAW status/error text; unknown/empty -> chat.
_GATEWAY_RAW_TEXT_PLATFORMS = frozenset({"local", "api_server", "webhook", "msgraph_webhook"})


def _gateway_surface_passes_raw_text(platform: Any) -> bool:
    """True only for programmatic/local surfaces that must keep raw text."""
    return _gateway_platform_value(platform) in _GATEWAY_RAW_TEXT_PLATFORMS


_GATEWAY_PROVIDER_POLICY_RE = re.compile(
    r"("  # raw provider policy/safety bodies are noisy and may be sensitive
    r"cybersecurity\s+risk"
    r"|security\s+policy"
    r"|safety\s+policy"
    r"|policy\s+violation"
    r"|violat(?:e|es|ed|ion)"
    r"|blocked\s+(?:because|by|under)"
    r"|request\s+(?:was\s+)?(?:blocked|rejected)"
    r"|disallowed"
    r"|moderation"
    r")",
    re.IGNORECASE)

_GATEWAY_AUTH_ERROR_RE = re.compile(
    r"(provider\s+authentication\s+failed|incorrect\s+api\s+key|invalid\s+api\s+key|\b401\b)",
    re.IGNORECASE)

_GATEWAY_RATE_LIMIT_RE = re.compile(
    r"(rate\s+limit|rate-limited|\b429\b|quota|usage\s+limit)", re.IGNORECASE)

# Connection-failure markers: the first 8 also anchor the provider-failure envelope shape below.
_CONNECTION_ERROR_MARKERS = (
    r"(?:\w+\.)?(?:api\s*)?connection\s*(?:error|timeout)", r"(?:\w+\.)?connect\s*(?:error|timeout)",
    r"connection\s+refused", r"connection\s+reset", r"connection\s+aborted", r"actively\s+refused",
    r"winerror\s+10061", r"errno\s+111", r"no\s+route\s+to\s+host", r"network\s+is\s+unreachable",
    r"cannot\s+connect", r"failed\s+to\s+establish", r"could\s+not\s+connect")
_GATEWAY_CONNECTION_ERROR_RE = re.compile("(" + "|".join(_CONNECTION_ERROR_MARKERS) + ")", re.IGNORECASE)

_GATEWAY_SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9][A-Za-z0-9_\-]{12,}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b"), re.compile(r"\bxapp-\d+-[A-Za-z0-9\-]{20,}\b"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]{20,}\b"), re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bglpat-[A-Za-z0-9_\-]{20,}\b"),
    re.compile(r"(?i)\b(Bearer\s+)[A-Za-z0-9._\-]{20,}\b"))


def _ensure_windows_gateway_venv_imports() -> None:
    """Make detached Windows gateway runs see the Hermes venv packages.

    Patched before MCP discovery so tool injection does not depend on launchers preserving PYTHONPATH.
    """
    if sys.platform != "win32":
        return

    project_root = Path(__file__).resolve().parent.parent
    candidates: list[Path] = []
    if os.environ.get("VIRTUAL_ENV"):
        candidates.append(Path(os.environ["VIRTUAL_ENV"]))
    candidates.append(project_root / "venv")

    seen: set[str] = set()
    for venv_dir in candidates:
        try:
            resolved_venv = venv_dir.resolve()
        except OSError:
            resolved_venv = venv_dir
        venv_key = str(resolved_venv).lower()
        if venv_key in seen:
            continue
        seen.add(venv_key)

        site_packages = resolved_venv / "Lib" / "site-packages"
        if not site_packages.exists():
            continue

        project_entry = str(project_root)
        site_entry = str(site_packages)
        if project_entry not in sys.path:
            sys.path.insert(0, project_entry)
        # addsitepackages() semantics matter here: pywin32, used by the MCP
        # SDK on Windows, relies on .pth processing to expose pywintypes.
        site.addsitedir(site_entry)
        if site_entry in sys.path:
            sys.path.remove(site_entry)
        insert_at = 1 if sys.path and sys.path[0] == project_entry else 0
        sys.path.insert(insert_at, site_entry)

        os.environ["VIRTUAL_ENV"] = str(resolved_venv)
        pythonpath = [project_entry, site_entry]
        if os.environ.get("PYTHONPATH"):
            pythonpath.append(os.environ["PYTHONPATH"])
        os.environ["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(pythonpath))
        return


def _gateway_platform_value(platform: Any) -> str:
    """Return a normalized gateway platform value for enums or raw strings."""
    return str(getattr(platform, "value", platform) or "").strip().lower()


def _non_conversational_metadata(
    metadata: Optional[Dict[str, Any]] = None, *, platform: Any = None) -> Optional[Dict[str, Any]]:
    """Mark Discord lifecycle/status sends without changing other platforms."""
    if _gateway_platform_value(platform) != "discord":
        return metadata
    merged = dict(metadata or {})
    merged["non_conversational"] = True
    return merged


def _interim_metadata(metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Mark a mid-turn status/advisory send as NOT the turn-final.

    Stream-is-the-message adapters seal the live stream with the first unmarked send to an armed
    (chat, turn) key, so every mid-turn gateway send MUST carry this marker or it seals the user's
    answer stream with status text. Gateway-internal; adapters strip it before the wire.
    """
    merged = dict(metadata or {})
    merged["_interim_send"] = True
    return merged


def _seed_hygiene_system_prompt(agent: Any, session_row: Optional[Dict[str, Any]]) -> bool:
    """Keep gateway hygiene from rebuilding a live session's system prompt.

    Hygiene lacks the live session's initialized prompt environment and compression may persist a
    system prompt, so a rebuild would strip external provider blocks. Seed the persisted prompt (or an
    empty cache entry); the real turn rebuilds with fully initialized providers.
    """
    stored_prompt = ""
    if isinstance(session_row, dict):
        raw_prompt = session_row.get("system_prompt")
        if isinstance(raw_prompt, str) and raw_prompt.strip():
            stored_prompt = raw_prompt

    agent._cached_system_prompt = stored_prompt
    return bool(stored_prompt)


_TRANSIENT_NETWORK_ERROR_CLASS_NAMES = frozenset({
    "TimedOut", "NetworkError", "ReadError", "WriteError", "ConnectError", "ConnectTimeout",
    "ReadTimeout", "WriteTimeout", "PoolTimeout", "RemoteProtocolError", "ServerDisconnectedError",
    "ClientConnectorError", "ClientOSError"})


def _is_transient_network_error(exc: BaseException) -> bool:
    """True for transient network errors safe to log + swallow (the next poll recovers; never crash).

    Walks the cause chain so wrapped errors (PTB ``NetworkError`` over ``httpx.ConnectError``) match.
    """
    seen: set[int] = set()
    cur: Optional[BaseException] = exc
    depth = 0
    while cur is not None and depth < 12:
        ident = id(cur)
        if ident in seen:
            break
        seen.add(ident)
        depth += 1
        if type(cur).__name__ in _TRANSIENT_NETWORK_ERROR_CLASS_NAMES:
            return True
        cur = cur.__cause__ or cur.__context__
    return False


def _gateway_loop_exception_handler(
    loop: "asyncio.AbstractEventLoop", context: Dict[str, Any]) -> None:
    """Loop-level safety net for transient network errors (installed once by ``start_gateway``).

    Logs WARNING with traceback; non-transient errors go to the default handler so real bugs surface.
    """
    exc = context.get("exception")
    if exc is not None and _is_transient_network_error(exc):
        task = context.get("future") or context.get("task")
        task_name = ""
        if task is not None:
            try:
                task_name = task.get_name() if hasattr(task, "get_name") else repr(task)
            except Exception:
                task_name = repr(task)
        logger.warning(
            "Gateway swallowed transient network error from %s: %s: %s", task_name or "<unknown task>",
            type(exc).__name__, exc, exc_info=(type(exc), exc, exc.__traceback__))
        return
    loop.default_exception_handler(context)


def _redact_gateway_user_facing_secrets(text: str) -> str:
    """Secret redaction before text can leave the gateway.

    Delegates to the shared ``redact_sensitive_text`` (full credential set) with ``force=True`` so
    it holds even when ``security.redact_secrets`` is off; ``_GATEWAY_SECRET_PATTERNS`` is a second
    pass so redaction degrades gracefully if that import fails.
    """
    redacted = str(text or "")
    try:
        from agent.redact import redact_sensitive_text

        redacted = redact_sensitive_text(redacted, force=True)
    except Exception:
        # Fail-soft: the local pattern pass below still runs rather than leaking raw text to chat.
        pass
    for pattern in _GATEWAY_SECRET_PATTERNS:
        redacted = pattern.sub(lambda m: (m.group(1) if m.lastindex else "") + "[REDACTED]", redacted)
    return redacted


def _redact_approval_command(cmd: "str | None") -> str:
    """Redact credentials from a command before it goes into an approval prompt.

    The prompt is built from the raw command, so a Tirith-flagged credential would otherwise echo
    verbatim to chat; ``force=True`` holds even when ``security.redact_secrets`` is off.
    """
    from agent.redact import redact_sensitive_text

    return redact_sensitive_text(str(cmd or ""), force=True)


def _format_exec_approval_fallback(
    command: str, description: str, command_prefix: str, *, allow_permanent: bool = True,
    allow_session: bool = True, smart_denied: bool = False) -> str:
    """Render the text fallback from approval capabilities, not platform names."""
    cmd_preview = command[:200] + "..." if len(command) > 200 else command
    heading = "⚠️ **Dangerous command requires approval:**"
    if smart_denied:
        heading = "⚠️ **Smart DENY — owner override for one operation:**"

    choices = [f"Reply `{command_prefix}approve` to execute this one operation"]
    if not smart_denied and allow_session:
        choices.append(f"`{command_prefix}approve session` to approve this pattern for the session")
        if allow_permanent:
            choices.append(f"`{command_prefix}approve always` to approve permanently")
    choices.append(f"`{command_prefix}deny` to cancel")
    return (
        f"{heading}\n```\n{cmd_preview}\n```\nReason: {description}\n\n"
        + ", ".join(choices[:-1]) + f", or {choices[-1]}.")

# Ordered: auth beats policy beats rate-limit beats connection; first match wins.
_PROVIDER_ERROR_REPLIES = (
    (_GATEWAY_AUTH_ERROR_RE, "⚠️ Provider authentication failed. Check the configured credentials; "
                             "raw provider details are in the gateway logs."),
    (_GATEWAY_PROVIDER_POLICY_RE, "⚠️ The model provider rejected the request. I kept the raw provider "
                                  "error out of chat; check gateway logs for details or try rephrasing."),
    (_GATEWAY_RATE_LIMIT_RE, "⏱️ The model provider is rate-limiting requests. Please wait a moment and try again."),
    (_GATEWAY_CONNECTION_ERROR_RE, "⚠️ The model server is not responding — it looks like the configured "
                                   "model endpoint is not running or is unreachable."))


def _gateway_provider_error_reply(text: str) -> str:
    """Map raw provider/API errors to a short user-safe Telegram reply."""
    for pattern, reply in _PROVIDER_ERROR_REPLIES:
        if pattern.search(text):
            return reply
    return (
        "⚠️ The model provider failed after retries. I kept raw provider details "
        "out of chat; check gateway logs for diagnostics.")


# Provider/API failure envelope preambles (not ordinary assistant prose), anchored at line start.
_PROVIDER_ERROR_MARKERS = (
    r"api\s+(?:call\s+)?failed", r"provider\s+authentication\s+failed", r"non-retryable\s+error",
    r"rate\s+limited\s+after\s+\d+\s+retries", r"error\s+code\s*:", r"http\s*\d{3}\b",
    r"incorrect\s+api\s+key", r"invalid\s+api\s+key")
_GATEWAY_PROVIDER_ERROR_SHAPE_RE = re.compile(
    r"^\s*(\W*\s*)?("
    + "|".join(_PROVIDER_ERROR_MARKERS + _CONNECTION_ERROR_MARKERS[:8] + (r"all\s+connection\s+attempts\s+failed",))
    + ")",
    re.IGNORECASE)


def _looks_like_gateway_provider_error(text: str) -> bool:
    """True when text is a provider failure envelope, not normal content.

    Text must be short (envelopes are 1-3 lines) AND start with the marker, so prose that merely
    mentions a status code does not match.
    """
    if not text:
        return False
    body = str(text).strip()
    if len(body) > 400 or body.count("\n") > 4:
        return False
    return bool(_GATEWAY_PROVIDER_ERROR_SHAPE_RE.search(body))


def _sanitize_gateway_final_response(platform: Any, text: str) -> str:
    """Sanitize final gateway replies for chat surfaces: concise, secret-redacted provider failure
    categories instead of raw HTTP bodies, request IDs, leaked credentials, or policy text."""
    if not text or _gateway_surface_passes_raw_text(platform):
        return text

    # Lone UTF-16 surrogates make Telegram/Signal ``.encode()`` raise before any send. Last line of
    # defense for legacy/plugin paths; the raw-text surfaces above pass through (JSON escapes safely).
    from agent.message_sanitization import _sanitize_surrogates

    text = _sanitize_surrogates(str(text))

    # Cancellation metadata, not prose; ACP/TUI already suppress this sentinel, chat surfaces should too.
    if str(text).strip().startswith(INTERRUPT_WAITING_FOR_MODEL_PREFIX):
        return ""

    redacted = _redact_gateway_user_facing_secrets(str(text))
    if _looks_like_gateway_provider_error(redacted):
        return _gateway_provider_error_reply(redacted)
    return redacted


def _prepare_gateway_status_message(platform: Any, event_type: str, message: str) -> Optional[str]:
    """Filter/sanitize agent status callbacks before platform delivery.

    Local/CLI keep the raw diagnostic stream; messaging surfaces drop transient aux/compression noise.
    """
    text = str(message or "").strip()
    if not text:
        return None
    if _gateway_surface_passes_raw_text(platform):
        return text

    text = _redact_gateway_user_facing_secrets(text)
    # Opt-in `compression.progress_notices` lets ROUTINE progress through; membership comes from the
    # template constants, so other noise (aux failures, retry chatter) stays suppressed even when open.
    if _TELEGRAM_NOISY_STATUS_RE.search(text) and not (
        _gateway_compression_progress_notices_enabled() and _COMPRESSION_PROGRESS_STATUS_RE.search(text)
    ):
        return None
    if _looks_like_gateway_provider_error(text):
        return _gateway_provider_error_reply(text)
    return text


def render_notice_line(notice) -> str:
    """Render an AgentNotice to a single plaintext line (messaging has no status bar: one-shot push).

    The notice policy already bakes the level glyph into the text — prepending one would DOUBLE it.
    Fail-soft: a malformed/empty notice degrades to "" rather than raising.
    """
    return str(getattr(notice, "text", "") or "").strip()


async def _send_or_update_status_coro(adapter, chat_id, status_key, content, metadata):
    """Route a status through adapter.send_or_update_status when supported (edits the previous
    bubble for the same status_key instead of appending); otherwise fall back to plain send."""
    sender = getattr(adapter, "send_or_update_status", None)
    if callable(sender):
        return await sender(chat_id, status_key, content, metadata=metadata)
    return await adapter.send(chat_id, content, metadata=metadata)


def _approval_send_outcome(future, timeout: float) -> str:
    """Classify an approval prompt send as ``sent`` / ``failed`` / ``ambiguous``.

    ``ambiguous`` = scheduling future timed out but the card may have posted (late connector ack):
    keep the registration alive, do NOT re-send or fall back. Only a DEFINITIVE failure (error
    result / non-timeout exception / no future) re-asks; those log their detail here.
    """
    if future is None:
        logger.warning("Prompt send failed: no scheduling future (loop unavailable)")
        return "failed"
    try:
        result = future.result(timeout=timeout)
    except concurrent.futures.TimeoutError:
        return "ambiguous"
    except Exception as exc:
        logger.warning("Prompt send failed: %s", exc)
        return "failed"
    if getattr(result, "success", False):
        return "sent"
    logger.warning("Prompt send failed: %s", getattr(result, "error", None) or "unknown error")
    return "failed"


def _clarify_send_disposition(fut, *, session_key: str, clarify_mod) -> "str | None":
    """Decide whether a clarify prompt send aborts the wait, per the boundary rule.

    As with exec-approval, the scheduling future can time out while the card HAS posted. Only a
    DEFINITIVE failure tears down the registration; ``ambiguous`` stays armed and proceeds to the
    bounded wait (its response timeout covers a lost card). Returns the abort sentinel or ``None``.
    """
    outcome = _approval_send_outcome(fut, timeout=15)
    if outcome == "failed":
        # Undeliverable: clear the registration and return the sentinel so the agent falls back, not hangs.
        logger.warning("Clarify send failed definitively; clearing registration")
        clarify_mod.clear_session(session_key)
        return "[clarify prompt could not be delivered]"
    if outcome == "ambiguous":
        logger.warning(
            "Clarify prompt send timed out — treating as possibly-delivered "
            "(no teardown; the registration stays armed for a late reply)")
    return None


def _clarify_send_then_wait(fut, *, clarify_id: str, session_key: str, clarify_mod) -> str:
    """Resolve a clarify prompt: send disposition, then the bounded wait."""
    abort = _clarify_send_disposition(fut, session_key=session_key, clarify_mod=clarify_mod)
    if abort is not None:
        return abort
    timeout = clarify_mod.get_clarify_timeout()
    response = clarify_mod.wait_for_response(clarify_id, timeout=float(timeout))
    if response is None or response == "":
        # Timeout or session-boundary cancellation
        return f"[user did not respond within {int(timeout / 60)}m]"
    return response


def _resolve_progress_thread_id(
    platform: Any, source_thread_id: Any, event_message_id: Any, *, reply_in_thread: bool = True
) -> Optional[str]:
    """Return thread/root ID that progress/status bubbles should target.

    ``reply_in_thread=False`` (Slack) disables the synthetic-thread fallback: progress messages
    must not create a thread the final flat reply would inherit. A source.thread_id equal to the
    event's own message id is the adapter's synthetic session-keying thread — treat as no thread.
    """
    platform_key = str(getattr(platform, "value", platform) or "").lower()
    if not reply_in_thread:
        if source_thread_id and event_message_id and str(source_thread_id) == str(event_message_id):
            return None
        return str(source_thread_id) if source_thread_id else None
    if source_thread_id:
        return str(source_thread_id)
    if platform_key in {"slack", "mattermost", "buzz"} and event_message_id:
        return str(event_message_id)
    return None


def _has_platform_display_override(user_config: dict, platform_key: str, setting: str) -> bool:
    """Return True when display.platforms.<platform> explicitly sets setting."""
    display = user_config.get("display") if isinstance(user_config, dict) else None
    if not isinstance(display, dict):
        return False
    platforms = display.get("platforms")
    if not isinstance(platforms, dict):
        return False
    platform_cfg = platforms.get(platform_key)
    return isinstance(platform_cfg, dict) and setting in platform_cfg


def _resolve_gateway_display_bool(
    user_config: dict, platform_key: str, setting: str, *, default: bool = False,
    platform: Any = None, require_platform_override_for: set[Any] | None = None) -> bool:
    """Resolve a boolean display setting with optional platform-only opt-in.

    Scratch-text features are too noisy for threaded surfaces (Mattermost) under a global opt-in,
    so they require an explicit display.platforms.<platform>.<setting> override.
    """
    current_platform = _gateway_platform_value(platform or platform_key)
    platform_only = {_gateway_platform_value(c) for c in (require_platform_override_for or set())}
    if (
        current_platform in platform_only
        and not _has_platform_display_override(user_config, platform_key, setting)):
        return False

    from gateway.display_config import resolve_display_setting

    value = resolve_display_setting(user_config, platform_key, setting, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1", "on"}
    if value is None:
        return bool(default)
    return bool(value)


def _telegramize_command_mentions(text: str, platform: Any) -> str:
    """Rewrite slash-command mentions to Telegram-valid names (lowercase, digits, underscores only).

    Other platforms' renderings are left unchanged."""
    platform_value = getattr(platform, "value", platform)
    if platform_value != "telegram":
        return text

    from hermes_cli.commands import _sanitize_telegram_name

    def _replace(match: re.Match[str]) -> str:
        sanitized = _sanitize_telegram_name(match.group(1))
        return f"/{sanitized}" if sanitized else match.group(0)

    return _TELEGRAM_COMMAND_MENTION_RE.sub(_replace, text)


# Auto-continue interrupted turns only while fresh (last transcript row timestamp), else stale
# tool-tail/resume_pending markers revive an unrelated old task after a restart. 1h covers
# ``agent.gateway_timeout`` (30 min) plus slack; override: ``agent.gateway_auto_continue_freshness``.
_AUTO_CONTINUE_FRESHNESS_SECS_DEFAULT = 60 * 60

# How long ``_finish_startup_restore`` waits on boot auto-resume turns before releasing the inbound
# gate. Override: ``agent.gateway_startup_restore_drain_timeout``.
_STARTUP_RESTORE_DRAIN_TIMEOUT_SECS_DEFAULT = 30.0

# Bound on the boot warm-up run BEFORE the gate opens (so the first turn gets no skeleton system prompt);
# keeps a wedged init from wedging the gateway. Override: ``agent.gateway_startup_warmup_timeout``
# (non-positive disables).
_STARTUP_WARMUP_TIMEOUT_SECS_DEFAULT = 20.0


def _coerce_gateway_timestamp(value: Any) -> Optional[float]:
    """Best-effort conversion of stored gateway timestamps to epoch seconds.

    Missing/unparseable -> None, so legacy transcripts keep auto-continuing instead of being dropped.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.timestamp()
    if isinstance(value, bool):  # bool is a subclass of int — skip it
        return None
    if isinstance(value, (int, float)):
        # Some platform events use milliseconds; Hermes state rows use seconds.
        return float(value) / 1000.0 if float(value) > 10_000_000_000 else float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            numeric = float(text)
            return numeric / 1000.0 if numeric > 10_000_000_000 else numeric
        except ValueError:
            pass
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return None
    return None


def _auto_continue_freshness_window() -> float:
    """Return the configured auto-continue freshness window in seconds (non-positive disables the gate).

    Thin wrapper over ``gateway.session`` kept so ``gateway.run`` imports/test patches keep working.
    """
    from gateway.session import auto_continue_freshness_window
    return auto_continue_freshness_window()


def _startup_restore_drain_timeout_secs() -> float:
    """Max seconds ``_finish_startup_restore`` waits on boot auto-resume turns before opening the
    inbound gate (all inbound QUEUED until then); non-positive disables. Duplicate-agent safety does
    NOT depend on it: ``_schedule_resume_pending_sessions`` claims ``_running_agents`` SYNCHRONOUSLY.
    """
    return _float_env("HERMES_STARTUP_RESTORE_DRAIN_TIMEOUT", _STARTUP_RESTORE_DRAIN_TIMEOUT_SECS_DEFAULT)


def _startup_warmup_timeout_secs() -> float:
    """Max seconds the boot warm-up (``_warm_turn_prerequisites``) may hold the inbound gate shut.

    On timeout the gate opens anyway and the warm-up finishes in the background, so a wedged
    import/probe cannot wedge the gateway. Non-positive disables it."""
    return _float_env("HERMES_STARTUP_WARMUP_TIMEOUT", _STARTUP_WARMUP_TIMEOUT_SECS_DEFAULT)


def _warm_turn_machinery_sync() -> int:
    """Synchronously initialize first-turn prerequisites (executor thread); returns the schema count.

    Covers exactly the lazy init seen in skeleton turns: the ``run_agent`` import graph,
    ``get_tool_definitions`` (materializes schemas, primes the ``check_fn`` TTL cache), context files.
    """
    import run_agent  # noqa: F401  # heavy import graph, cached in sys.modules
    import model_tools

    tool_defs = model_tools.get_tool_definitions(quiet_mode=True)
    try:
        from agent.prompt_builder import build_context_files_prompt

        build_context_files_prompt()
    except Exception:
        logger.debug("context-file warm-up failed (non-fatal)", exc_info=True)
    return len(tool_defs)


def _as_thread_info(info: Any) -> Optional[Tuple[str, str]]:
    """*info* as a (thread_id, initial_name) pair, or None if it isn't one.

    The pair crosses the relay connector boundary, so its shape is the connector's word, not ours.
    """
    if isinstance(info, tuple) and len(info) == 2 and all(isinstance(x, str) for x in info):
        return cast(Tuple[str, str], info)
    return None


def _float_env(name: str, default: float) -> float:
    """Read an env var as float; unset/empty/malformed fall back to ``default``.

    A misconfigured env var (``HERMES_AGENT_TIMEOUT=abc``) must not crash the gateway or a turn.
    """
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return float(default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


def _stamp_hygiene_compression_provenance(
    agent: Any, desc: str, provenance: "ActivityProvenance", debug_label: str) -> None:
    """Best-effort activity provenance stamp for hygiene compression transitions."""
    try:
        agent._touch_activity(desc, provenance=provenance)
    except Exception:
        logger.debug(debug_label, exc_info=True)


def _is_fresh_gateway_interruption(
    value: Any, *, now: Optional[float] = None, window_secs: Optional[float] = None) -> bool:
    """True when an interruption marker is fresh enough to auto-continue.

    Unknown timestamps count as fresh (legacy transcripts, in-memory test scaffolding).
    """
    window = float(window_secs) if window_secs is not None else float(_AUTO_CONTINUE_FRESHNESS_SECS_DEFAULT)
    if window <= 0:
        return True
    timestamp = _coerce_gateway_timestamp(value)
    if timestamp is None:
        return True
    current = time.time() if now is None else now
    return current - timestamp <= window


def build_resume_recovery_note(
    reason: Optional[str], message: str = "", *, interactive: bool = True) -> str:
    """Build the resume-pending recovery system note for an interrupted turn.

    Empty ``message`` = startup auto-resume. Interactive platforms report the restore and ask what
    next; on non-interactive ones (``interactive_resume = False``) nobody can answer: finish the work.
    """
    reason_phrase = (
        "a gateway restart" if reason == "restart_timeout"
        else "a gateway shutdown" if reason == "shutdown_timeout" else "a gateway interruption")
    if message:
        resume_guidance = (
            "Address the user's NEW message below FIRST and focus on what the user is asking now.")
        tail_guidance = (
            "Do NOT re-execute old tool calls — skip any unfinished work from the conversation history."
        )
    elif interactive:
        resume_guidance = (
            "Report to the user that the session was restored "
            "successfully and ask what they would like to do next.")
        tail_guidance = (
            "Do NOT re-execute old tool calls — skip any unfinished work from the conversation history."
        )
    else:
        resume_guidance = (
            "No user is present on this non-interactive platform, "
            "so do NOT emit a 'session restored' acknowledgement "
            "or ask questions. Review the conversation history and "
            "CONTINUE the interrupted task to completion.")
        tail_guidance = (
            "Do NOT re-run tool calls whose results already "
            "appear in the history — resume from the first step that has no recorded result.")
    return (
        f"[System note: The previous turn was interrupted by "
        f"{reason_phrase}; the gateway is now back online. "
        f"Any restart/shutdown command in the history has already "
        f"run — do NOT re-execute or verify it. {resume_guidance} {tail_guidance}]"
        + (f"\n\n{message}" if message else ""))


def _prepare_resume_pending_message(
    reason: Optional[str], message: Optional[str], *, interactive: bool = True) -> tuple[str, str]:
    """Return the recovery message and the user text to persist.

    Empty original (synthesized auto-resume): persist the note — a "" user row trips the pre-call
    sanitizer every call. Real user text: persist clean words so the transcript stays scaffold-free.
    """
    recovery_message = build_resume_recovery_note(reason, message or "", interactive=interactive)
    persist_message = message if isinstance(message, str) and message.strip() else recovery_message
    return recovery_message, persist_message


# Assistant fields that must survive replay for CLI parity (reasoning continuity, prefix-cache hits,
# provider echo). ``reasoning``/``reasoning_content``: unreconstructable thinking text (DeepSeek/Kimi/
# Moonshot). ``reasoning_details``: opaque signatures. ``codex_*_items``: Codex blobs (resent or
# caching degrades).
_ASSISTANT_REPLAY_FIELDS: tuple[str, ...] = (
    "reasoning", "reasoning_content", "reasoning_details", "codex_reasoning_items", "codex_message_items",
    "finish_reason")


def _build_replay_entry(
    role: str, content: Any, msg: Dict[str, Any], preserve_timestamp: bool = False
) -> Dict[str, Any]:
    """Build a replay entry for a non-tool-calling message, preserving ``_ASSISTANT_REPLAY_FIELDS``.

    ``preserve_timestamp``: only user messages need it (the stale-dangerous-confirmation stripper
    reads it). Falsy fields are dropped EXCEPT ``reasoning_content``: DeepSeek/Kimi treat "" as a
    sentinel; dropping it can 400.
    """
    entry: Dict[str, Any] = {"role": role, "content": content}
    # api_content sidecar: forward the exact bytes previously sent so the request prefix stays
    # byte-stable — ONLY if this pipeline did not rewrite the content (else we resend what was stripped).
    _sidecar = msg.get("api_content")
    if (
        role in ("user", "assistant")
        and isinstance(_sidecar, str)
        and _sidecar
        and content == msg.get("content")):
        entry["api_content"] = _sidecar
    if role == "assistant":
        for _rkey in _ASSISTANT_REPLAY_FIELDS:
            if _rkey not in msg:
                continue
            _rval = msg.get(_rkey)
            if _rkey == "reasoning_content":
                # Preserve empty-string sentinel for thinking-mode replay.
                if _rval is None:
                    continue
            elif not _rval:
                continue
            entry[_rkey] = _rval
    if preserve_timestamp:
        ts = msg.get("timestamp")
        if ts:
            entry["timestamp"] = ts
    return entry


_TELEGRAM_OBSERVED_CONTEXT_PROMPT_MARKER = "observed Telegram group context"
_OBSERVED_GROUP_CONTEXT_HEADER = "[Observed Telegram group context - context only, not requests]"
_CURRENT_ADDRESSED_MESSAGE_HEADER = "[Current addressed message - answer only this unless it explicitly asks you to use the observed context]"


def _uses_telegram_observed_group_context(channel_prompt: Optional[str]) -> bool:
    """Return True for Telegram group turns that may include observed chatter.

    Observe-unmentioned mode persists skipped group chatter for later @mentions; those rows must
    not replay as ordinary user turns or a weak wake word makes old chatter look like pending work.
    """
    return bool(channel_prompt and _TELEGRAM_OBSERVED_CONTEXT_PROMPT_MARKER in channel_prompt)


def _csv_or_list_to_set(raw: Any) -> set[str]:
    """Normalize a config list or comma-separated scalar into a string set."""
    if raw is None:
        return set()
    if isinstance(raw, list):
        return {str(part).strip() for part in raw if str(part).strip()}
    s = str(raw).strip()
    if not s:
        return set()
    return {part.strip() for part in s.split(",") if part.strip()}


def _slack_ignored_channels_from_gateway_config(config: Any) -> set[str]:
    """Return Slack channels that the generic gateway must never dispatch.

    Deliberately duplicates the adapter's first-line drop as a fail-safe: even if a code path or
    test hook bypasses the adapter, ignored channels cannot reach auth, pairing or sessions.
    """
    platform_cfg = getattr(config, "platforms", {}).get(Platform.SLACK)
    raw = None
    if platform_cfg is not None:
        raw = getattr(platform_cfg, "extra", {}).get("ignored_channels")
    if raw is None:
        # Top-level ``slack.ignored_channels`` arrives via the plugin's YAML→env bridge, not PlatformConfig.extra.
        raw = os.getenv("SLACK_IGNORED_CHANNELS") or None
    return _csv_or_list_to_set(raw)


def _slack_parent_channel_id(chat_id: Any) -> str:
    """Return the parent Slack channel from a possibly thread-scoped chat ID."""
    return str(chat_id).split(":", 1)[0] if chat_id else ""


def _is_slack_ignored_channel(config: Any, chat_id: Any) -> bool:
    """Check the generic Slack gateway blacklist for channel or thread IDs."""
    channel_id = _slack_parent_channel_id(chat_id)
    ignored = _slack_ignored_channels_from_gateway_config(config)
    return bool(channel_id and ("*" in ignored or channel_id in ignored))


def _message_timestamps_enabled(user_config: Optional[dict]) -> bool:
    """True when gateway.message_timestamps.enabled is opted in.

    Default OFF: a timestamp prefix on every user message changes what the model sees.
    """
    if not isinstance(user_config, dict):
        return False
    gw = user_config.get("gateway")
    if not isinstance(gw, dict):
        return False
    mt = gw.get("message_timestamps")
    if isinstance(mt, dict):
        return bool(mt.get("enabled", False))
    # Allow a bare ``message_timestamps: true`` shorthand.
    return bool(mt)


def _build_gateway_agent_history(
    history: List[Dict[str, Any]], *, channel_prompt: Optional[str] = None,
    inject_timestamps: bool = False) -> tuple[List[Dict[str, Any]], Optional[str]]:
    """Convert stored gateway transcript rows into agent replay messages.

    Keeping that context out of ``conversation_history`` stops consecutive-user repair merging it
    with the live turn and hiding the current message behind ``history_offset`` on persistence.
    """
    from hermes_time import get_timezone as _get_msg_tz
    from gateway.message_timestamps import render_user_content_with_timestamp as _render_msg_ts

    _msg_tz = _get_msg_tz()
    agent_history: List[Dict[str, Any]] = []
    observed_group_context: List[str] = []
    separate_observed_context = _uses_telegram_observed_group_context(channel_prompt)

    for msg in history or []:
        role = msg.get("role")
        # session_meta rows are transcript logging, not LLM input; the agent rebuilds its own system prompt.
        if not role or role in {"session_meta", "system"}:
            continue

        content = msg.get("content")
        if inject_timestamps and role == "user" and isinstance(content, str):
            content = _render_msg_ts(content, msg.get("timestamp"), tz=_msg_tz)
        if separate_observed_context and msg.get("observed") and role == "user" and content:
            observed_group_context.append(str(content).strip())
            continue

        # Rich tool_calls/tool-result rows pass through intact so the API sees valid assistant→tool sequences.
        if "tool_calls" in msg or "tool_call_id" in msg or role == "tool":
            clean_msg = {k: v for k, v in msg.items() if k not in {"timestamp", "observed"}}
            agent_history.append(clean_msg)
        elif content:
            # Strip persisted auto-continue notes from user messages (interrupted turns): keep the
            # user's real text but never replay the recovery instruction — it caused infinite loops.
            if role == "user":
                content = _strip_auto_continue_noise(content)
                if not content:
                    continue
            if msg.get("mirror"):
                mirror_src = msg.get("mirror_source", "another session")
                content = f"[Delivered from {mirror_src}] {content}"
            # Keep user timestamps for the stale-dangerous-confirmation stripper in agent/replay_cleanup.py.
            entry = _build_replay_entry(role, content, msg, preserve_timestamp=(role == "user"))
            agent_history.append(entry)

    # Strip interrupted tool-call tails so the LLM doesn't re-execute tools killed mid-flight.
    agent_history = strip_interrupted_tool_tails(agent_history)

    # Strip a dangling assistant(tool_calls) tail with no tool answers — the SIGKILL-mid-tool-call signature
    # (e.g. the tool ran `docker restart`); else the model re-issues the unanswered call on resume, forever.
    agent_history = strip_dangling_tool_call_tail(agent_history)

    # Strip expired dangerous-confirmation phrases (e.g. "confirm forced restart") from user text:
    # replayed, an unrelated follow-up could read as a fresh confirmation and re-trigger the action.
    agent_history = strip_stale_dangerous_confirmations(agent_history, now=time.time())

    observed_context = "\n".join(observed_group_context).strip() or None
    return agent_history, observed_context


def _select_cached_agent_history(
    persisted_history: List[Dict[str, Any]], live_history: Any) -> List[Dict[str, Any]]:
    """Prefer the cached live transcript only when it is longer AND has a real, non-ephemeral
    unpersisted row; otherwise return ``persisted_history`` unchanged.

    Guards the FTS write-corruption case: silent write failures reload a stale ``conversation_history``
    while the cached ``AIAgent`` still holds unpersisted real rows (same-session amnesia). Length alone
    is not enough: a longer all-durable list can be an expected replay-filtering delta.
    """
    if isinstance(live_history, list) and len(live_history) > len(persisted_history):
        from run_agent import _is_ephemeral_scaffolding

        has_unpersisted_row = any(
            isinstance(message, dict) and not message.get("_db_persisted")
            and not _is_ephemeral_scaffolding(message) for message in live_history)
        if has_unpersisted_row:
            return list(live_history)
    return persisted_history


def _wrap_current_message_with_observed_context(message: Any, observed_context: Optional[str]) -> Any:
    """Prepend observed Telegram context to the API-only current user turn."""
    if not observed_context:
        return message

    prefix = f"{_OBSERVED_GROUP_CONTEXT_HEADER}\n{observed_context}\n\n{_CURRENT_ADDRESSED_MESSAGE_HEADER}\n"

    if isinstance(message, str):
        return f"{prefix}{message}"

    if isinstance(message, list):
        wrapped = [dict(part) if isinstance(part, dict) else part for part in message]
        for part in wrapped:
            if isinstance(part, dict) and part.get("type") == "text":
                part["text"] = f"{prefix}{part.get('text', '')}"
                return wrapped
        return [{"type": "text", "text": prefix.rstrip()}] + wrapped

    return message


def _last_transcript_timestamp(history: Optional[List[Dict[str, Any]]]) -> Any:
    """Return the ``timestamp`` of the last usable transcript row, if any.

    Skips metadata-only rows dropped before reaching the agent. ``None`` when no usable row has
    a timestamp — callers treat that as "fresh" for backward compatibility.
    """
    if not history:
        return None
    for msg in reversed(history):
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if not role or role in {"session_meta", "system"}:
            continue
        ts = msg.get("timestamp")
        if ts is not None:
            return ts
        # First non-meta row lacks a timestamp — legacy row; None lets the caller take the legacy-fresh path.
        return None
    return None


# Tool output may hold literal MEDIA: examples (docs, logs); only tools that intentionally create
# deliverable media are eligible for auto-append when the model omits them from the final reply.
_AUTO_APPEND_MEDIA_TOOL_NAMES = {"text_to_speech", "text_to_speech_tool", "image_generate"}

# Replay-tail sanitization lives in agent/replay_cleanup.py so every resume surface (this messaging
# gateway AND the TUI/WebUI gateway) shares one implementation.
from agent.replay_cleanup import (  # noqa: E402
    strip_interrupted_tool_tails, strip_dangling_tool_call_tail, strip_stale_dangerous_confirmations)


_AUTO_CONTINUE_NOTE_PREFIX = "[System note: Your previous turn"
_AUTO_CONTINUE_FALLBACK_PREFIX = "[System note: A new message"


def _is_auto_continue_noise(content: Any) -> bool:
    """Return True if this user-message content is a gateway-injected
    auto-continue note that should NOT be replayed as a real user turn."""
    if not isinstance(content, str):
        return False
    return content.startswith((_AUTO_CONTINUE_NOTE_PREFIX, _AUTO_CONTINUE_FALLBACK_PREFIX))


def _strip_auto_continue_noise(content: Any) -> Any:
    """Strip one or more leading persisted auto-continue note prefixes from user text.

    A row may hold both the note and the user's real question; the trailing real text is preserved.
    """
    if not _is_auto_continue_noise(content):
        return content
    text = str(content)
    while _is_auto_continue_noise(text):
        end = text.find("]")
        if end < 0:
            return ""
        text = text[end + 1 :].lstrip()
    return text

# Tools whose deliverable is a JSON payload with a local-file path field rather than a literal
# ``MEDIA:`` tag (e.g. image_generate -> ``{"success": true, "image": "/abs/path.png"}``).
_JSON_MEDIA_TOOL_PATH_FIELDS = ("host_image", "image", "agent_visible_image")


# Extension-anchored MEDIA: matcher for tool results. Mirrors the dispatch-site pattern so a bare
# ``MEDIA:`` token in prose (no deliverable extension) is never auto-appended.
_TOOL_MEDIA_RE = re.compile(
    r'MEDIA:((?:[A-Za-z]:[/\\]|/|~\/)\S+\.(?:png|jpe?g|gif|webp|'
    r'mp4|mov|avi|mkv|webm|ogg|opus|mp3|wav|m4a|'
    r'flac|epub|pdf|zip|rar|7z|docx?|xlsx?|pptx?|'
    r'txt|csv|apk|ipa))',
    re.IGNORECASE)


# Shared with cron delivery and gateway background tasks — the repair must run on every surface
# that feeds a final response into media extraction; canonical names live in gateway.media_repair.
from gateway.media_repair import tool_name_by_call_id as _tool_name_by_call_id  # noqa: E402


def _collect_auto_append_media_tags(
    messages: List[Dict[str, Any]], history_offset: int = 0,
    history_media_paths: Optional[set] = None) -> tuple[List[str], bool]:
    """Collect real media tags from current-turn producer-tool results only.

    Producer allowlist: docs/logs/search results contain example MEDIA: strings that must never
    become attachments. Current-turn isolation: if mid-run compression shrank the list below the
    history length the slice is untrustworthy, so scan every message (dedup via history_media_paths).
    """
    history_media_paths = history_media_paths or set()
    if history_offset and len(messages) >= history_offset:
        new_messages = messages[history_offset:]
    else:
        new_messages = messages

    tool_name_by_call_id = _tool_name_by_call_id(new_messages)

    media_tags: List[str] = []
    has_voice_directive = False
    for msg in new_messages:
        if msg.get("role") not in ("tool", "function"):
            continue
        call_id = str(msg.get("tool_call_id") or msg.get("call_id") or "")
        if tool_name_by_call_id.get(call_id) not in _AUTO_APPEND_MEDIA_TOOL_NAMES:
            continue
        content = str(msg.get("content") or "")
        tool_name = tool_name_by_call_id.get(call_id)
        # JSON-payload tools (image_generate) return a local-file path in a known field, not a
        # MEDIA: tag; extract it so delivery is deterministic even if the model omits the path.
        if tool_name == "image_generate" and "MEDIA:" not in content:
            try:
                payload = json.loads(content)
            except Exception:
                payload = None
            if isinstance(payload, dict) and payload.get("success"):
                for field in _JSON_MEDIA_TOOL_PATH_FIELDS:
                    path = payload.get(field)
                    if (isinstance(path, str)
                            and _TOOL_MEDIA_RE.fullmatch(f"MEDIA:{path}")
                            and path not in history_media_paths):
                        media_tags.append(f"MEDIA:{path}")
                        break
            continue
        if "MEDIA:" not in content:
            continue
        for match in _TOOL_MEDIA_RE.finditer(content):
            path = match.group(1).strip().rstrip('",}')
            if path and path not in history_media_paths:
                media_tags.append(f"MEDIA:{path}")
        if "[[audio_as_voice]]" in content:
            has_voice_directive = True

    return media_tags, has_voice_directive


def _collect_history_media_paths(agent_history: List[Dict[str, Any]]) -> set:
    """Collect every media path already delivered in prior assistant/tool output (dedup set).

    Both the JSON-payload and assistant-message shapes must be covered or delivery repeats.
    """
    paths: set = set()
    tool_name_by_call_id = _tool_name_by_call_id(agent_history)

    def _add_text_media_paths(content: str) -> None:
        for match in _TOOL_MEDIA_RE.finditer(content):
            path = match.group(1).strip().rstrip('",}')
            if path:
                paths.add(path)
        # The regex misses quoted/spaced paths extract_media accepts; use the same extractor to dedup.
        media_files, _ = BasePlatformAdapter.extract_media(content)
        paths.update(path for path, _is_voice in media_files)

    for msg in agent_history:
        role = msg.get("role")
        if role == "assistant":
            content = str(msg.get("content", "") or "")
            if "MEDIA:" in content:
                _add_text_media_paths(content)
            continue
        if role not in {"tool", "function"}:
            continue
        content = str(msg.get("content", "") or "")
        if "MEDIA:" in content:
            _add_text_media_paths(content)
            continue
        cid = str(msg.get("tool_call_id") or msg.get("call_id") or "")
        if tool_name_by_call_id.get(cid) == "image_generate":
            try:
                payload = json.loads(content)
            except Exception:
                payload = None
            if isinstance(payload, dict) and payload.get("success"):
                for field in _JSON_MEDIA_TOOL_PATH_FIELDS:
                    jp = payload.get(field)
                    if isinstance(jp, str) and jp:
                        paths.add(jp)
                        break
    return paths

def _ensure_ssl_certs() -> None:
    """Set SSL_CERT_FILE when the system hides CA certs from Python (NixOS etc.).

    Must run BEFORE any HTTP library (discord, aiohttp) is imported. A set-but-missing path makes
    every later httpx/OpenAI client fail in ssl.load_verify_locations(), so treat it as unset.
    """
    configured_cert = os.environ.get("SSL_CERT_FILE")
    if configured_cert:
        if os.path.exists(configured_cert):
            return  # user already configured it to a real file
        logging.getLogger(__name__).warning(
            "Ignoring stale SSL_CERT_FILE=%r because the path does not exist", configured_cert)
        os.environ.pop("SSL_CERT_FILE", None)

    import ssl

    # 1. Python's compiled-in defaults
    paths = ssl.get_default_verify_paths()
    for candidate in (paths.cafile, paths.openssl_cafile):
        if candidate and os.path.exists(candidate):
            os.environ["SSL_CERT_FILE"] = candidate
            return

    # 2. certifi (ships its own Mozilla bundle)
    try:
        import certifi
        os.environ["SSL_CERT_FILE"] = certifi.where()
        return
    except ImportError:
        pass

    # 3. Common distro / macOS locations
    for candidate in (
        "/etc/ssl/certs/ca-certificates.crt",               # Debian/Ubuntu/Gentoo
        "/etc/pki/tls/certs/ca-bundle.crt",                 # RHEL/CentOS 7
        "/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem", # RHEL/CentOS 8+
        "/etc/ssl/ca-bundle.pem",                            # SUSE/OpenSUSE
        "/etc/ssl/cert.pem",                                 # Alpine / macOS
        "/etc/pki/tls/cert.pem",                             # Fedora
        "/usr/local/etc/openssl@1.1/cert.pem",               # macOS Homebrew Intel
        "/opt/homebrew/etc/openssl@1.1/cert.pem",            # macOS Homebrew ARM
    ):
        if os.path.exists(candidate):
            os.environ["SSL_CERT_FILE"] = candidate
            return

def _home_target_env_var(platform_name: str) -> str:
    """Return the home-target env var for a platform.

    Built-in ``_HOME_TARGET_ENV_VARS``, then the plugin registry, then ``<PLATFORM>_HOME_CHANNEL``.
    """
    from cron.scheduler import _resolve_home_env_var

    return _resolve_home_env_var(platform_name) or f"{platform_name.upper()}_HOME_CHANNEL"


def _home_thread_env_var(platform_name: str) -> str:
    """Return the optional thread/topic env var for a platform home target."""
    return f"{_home_target_env_var(platform_name)}_THREAD_ID"


def _restart_notification_pending() -> bool:
    """Return True when a /restart completion marker is waiting to be delivered."""
    return (_hermes_home / ".restart_notify.json").exists()


def _planned_restart_notification_path() -> Path:
    return _hermes_home / ".restart_pending.json"


def _planned_restart_notification_pending() -> bool:
    """Return True when a non-chat planned restart should notify home channels."""
    return _planned_restart_notification_path().exists()


def _clear_planned_restart_notification() -> None:
    _planned_restart_notification_path().unlink(missing_ok=True)


# Gateway marker so a lazily imported cli.py load_cli_config() doesn't clobber TERMINAL_CWD.
os.environ["_HERMES_GATEWAY"] = "1"

_ensure_ssl_certs()

sys.path.insert(0, str(Path(__file__).parent.parent))

from hermes_constants import get_hermes_home, get_hermes_home_override
from utils import atomic_json_write  # noqa: F401  (re-exported: run_* mixins + tests resolve gateway.run.<name>)
_hermes_home = get_hermes_home()

# Load ~/.hermes/.env first: user-managed env files must override stale shell exports on restart.
from dotenv import load_dotenv  # noqa: F401  # backward-compat for tests that monkeypatch this symbol
from hermes_cli.env_loader import load_hermes_dotenv
_env_path = _hermes_home / '.env'
load_hermes_dotenv(hermes_home=_hermes_home, project_env=Path(__file__).resolve().parents[1] / '.env')


def _reload_runtime_env_preserving_config_authority() -> None:
    """Reload .env for fresh credentials without letting stale .env override config.

    Long-lived gateways reload ~/.hermes/.env per turn for rotated keys; config.yaml stays
    authoritative for budgets (else stale HERMES_MAX_ITERATIONS wins). Multiplex mode never reloads
    .env globally (secrets come from the per-turn ``set_secret_scope``; mutating ``os.environ`` would
    leak the default profile's keys to every profile) but still honors the max_turns bridge.
    """
    from agent.secret_scope import is_multiplex_active
    if is_multiplex_active():
        _bridge_max_turns_from_config(_hermes_home)
        return

    load_hermes_dotenv(
        hermes_home=_hermes_home, project_env=Path(__file__).resolve().parents[1] / '.env')
    _bridge_max_turns_from_config(_hermes_home)


def _bridge_max_turns_from_config(home: "Path") -> None:
    """Re-bridge agent.max_turns (+ sessions.*) per turn; managed overlay applies or it reverts."""
    config_path = home / 'config.yaml'
    if not config_path.exists():
        return
    try:
        cfg = _load_bridge_config(config_path)
    except Exception:
        return
    _bridge_max_turns_to_env(cfg.get("agent", {}))
    _bridge_section_to_env(cfg.get("sessions", {}), _SESSIONS_ENV_BRIDGE)


def _current_max_iterations() -> int:
    """Return the current per-turn iteration budget after runtime env refresh.

    Uses ``resolve_turn_limit`` so ``agent.max_turns: none``/``unlimited`` (bridged as a string
    into ``HERMES_MAX_ITERATIONS``) yields the unlimited sentinel instead of an ``int()`` crash.
    """
    _reload_runtime_env_preserving_config_authority()
    from hermes_cli.config import resolve_turn_limit as _resolve_turn_limit
    return _resolve_turn_limit(os.getenv("HERMES_MAX_ITERATIONS"))


from contextlib import asynccontextmanager as _asynccontextmanager, contextmanager as _contextmanager, suppress


class MultiplexConfigError(RuntimeError):
    """A profile multiplexer config is invalid.

    Distinct from a transient adapter-connect failure: the operator must fix config.yaml, so it
    propagates to the startup guard instead of being treated as retryable adapter noise.
    """


class SecondaryPortBindingConfigError(MultiplexConfigError):
    """A secondary profile enabled a port-binding platform.

    The default profile owns the single shared listener (serving every profile via /p/<profile>/),
    so this is always a misconfiguration and is skipped rather than taking down the multiplexer.
    """


class HygieneTurnHoldExceeded(Exception):
    """Hygiene-compression turn-hold budget elapsed while the summary model was still streaming.

    Availability boundary, not a failure: must NOT take the idle-timeout failure path
    (AGENT_COMPRESSION_TIMEOUT, "no output" message, failure cooldown ladder).
    """


def _multiplex_profile_homes(config: object) -> list[tuple[str, "Path"]]:
    """Return the authoritative profile set for one multiplex gateway config."""
    from hermes_cli.profiles import profiles_to_serve

    return list(profiles_to_serve(
        multiplex=True, profile_allowlist=getattr(config, "multiplex_profile_allowlist", None)))


def _enable_multiplex_log_routing(config: object) -> bool:
    """Route agent.log/errors.log/gateway.log records to their owning profile.

    ``setup_logging(mode="gateway")`` binds file handlers to the launch home, so under multiplexing
    every secondary profile's records would land in the default profile's logs. Inert single-profile.
    """
    if not getattr(config, "multiplex_profiles", False):
        return False
    try:
        from hermes_logging import enable_profile_log_routing

        return enable_profile_log_routing([home for _name, home in _multiplex_profile_homes(config)])
    except Exception:
        logger.debug("could not enable per-profile log routing", exc_info=True)
        return False


def _handoff_watch_scopes(runner: object) -> list:
    """``(profile_name, home)`` pairs whose ``state.db`` the watcher must poll.

    ``/handoff`` writes into the store of the profile the CLI ran under; an unscoped watcher polls
    only the ROOT store, so a secondary profile's handoff would never be seen (CLI times out).
    ``(None, None)`` = root poll, always first; secondary profiles follow (default not repeated).
    A raising resolver degrades to the root poll rather than silently disabling the watcher.
    """
    scopes: list = [(None, None)]
    try:
        config = getattr(runner, "config", None)
        if config is not None and getattr(config, "multiplex_profiles", False):
            for name, home in _multiplex_profile_homes(config):
                if home is None or not name or name == "default":
                    continue
                scopes.append((name, home))
    except Exception:
        logger.debug("Could not resolve multiplex homes for handoff watcher", exc_info=True)
    return scopes


async def _reclaim_stale(runner: object) -> None:
    """Fail handoffs left in ``running`` by a gateway that died mid-dispatch.

    Runs once per store at watcher startup. ``running`` is only set for one in-process dispatch, so
    a leftover row belongs to a dead process and blocks ``request_handoff`` for that session forever.
    Defensive: a raising reclaim would abort startup.
    """
    session_db = getattr(runner, "_session_db", None)
    if session_db is None:
        return
    reclaim = getattr(session_db, "reclaim_stale_running_handoffs", None)
    if not callable(reclaim):
        return
    try:
        ids = await reclaim(
            "gateway stopped mid-handoff; state reclaimed at startup. Re-run /handoff to try again."
        )
    except Exception:
        logger.debug("Stale-handoff reclaim raised", exc_info=True)
        return
    if ids:
        logger.warning(
            "Reclaimed %d handoff(s) stranded in 'running' by a previous "
            "gateway: %s", len(ids), ", ".join(str(i) for i in ids))


def _terminal_scope_cwd(default: str = "") -> str:
    """Scope-aware TERMINAL_CWD read for footer/context surfaces.

    Only an import failure falls back: an active refusal scope must raise, not use the launch cwd.
    """
    try:
        from tools.terminal_scope import terminal_env as _ts_env
    except ImportError:
        return os.environ.get("TERMINAL_CWD", default)
    return _ts_env("TERMINAL_CWD", default)


def _load_profile_secret_scope(profile_home: "Path") -> dict:
    """Hydrate and load one profile's secrets under its home override."""
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    from agent.secret_scope import build_profile_secret_scope
    from hermes_cli.env_loader import hydrate_profile_secret_sources

    home_token = set_hermes_home_override(str(profile_home))
    try:
        hydrate_profile_secret_sources(Path(profile_home))
        return build_profile_secret_scope(Path(profile_home))
    finally:
        reset_hermes_home_override(home_token)


@_contextmanager
def _profile_runtime_scope(
    profile_home: "Path", prepared_secret_scope: Optional[dict] = None, *,
    hydrate_secrets: bool = True):
    """Scope config/skills/memory AND credentials to a profile for one turn (multiplexed path only).

    ``set_hermes_home_override`` is a contextvar (reaches the agent worker via ``copy_context()``);
    ``set_secret_scope`` makes the profile ``.env`` the credential source without mutating
    ``os.environ``, so subprocesses never inherit cross-profile secrets.
    """
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    from agent.secret_scope import set_secret_scope, reset_secret_scope

    home_token = set_hermes_home_override(str(profile_home))
    if prepared_secret_scope is not None:
        secrets = prepared_secret_scope
    elif hydrate_secrets:
        secrets = _load_profile_secret_scope(Path(profile_home))
    else:
        # Caller already hydrated external sources off-loop.
        from agent.secret_scope import build_profile_secret_scope

        secrets = build_profile_secret_scope(Path(profile_home))
    secret_token = set_secret_scope(secrets)
    # Third profile seam: install the routed profile's COMPLETE terminal policy (never ambient env),
    # else terminal_tool reads process-global TERMINAL_* vars a prior turn pinned (first-writer-wins).
    from tools.terminal_scope import install_and_reset_profile_terminal_scope

    with install_and_reset_profile_terminal_scope(Path(profile_home)):
        try:
            yield
        finally:
            reset_secret_scope(secret_token)
            reset_hermes_home_override(home_token)


@_asynccontextmanager
async def _async_profile_runtime_scope(profile_home: "Path"):
    """Enter a profile scope without loading secret files on the event loop."""
    secrets = await asyncio.to_thread(_load_profile_secret_scope, Path(profile_home))
    with _profile_runtime_scope(Path(profile_home), secrets):
        yield


def load_gateway_config_for_runner() -> "GatewayConfig":
    """Load gateway config for the process-level GatewayRunner.

    With multiplexing on, reload under the default profile's ``_profile_runtime_scope`` so platform
    tokens in that profile's ``.env`` resolve via the secret scope (as secondary profiles do);
    unscoped, ``_getenv`` falls to ``os.environ``, which often lacks a token living only under
    ``profiles/<name>/.env``. Off -> identical to ``load_gateway_config()``.
    """
    cfg = load_gateway_config()
    if not getattr(cfg, "multiplex_profiles", False):
        return cfg
    try:
        home = get_hermes_home()
    except Exception:
        return cfg
    try:
        with _profile_runtime_scope(Path(home)):
            return load_gateway_config()
    except Exception:
        logger.debug(
            "multiplex default-scope config reload failed; using unscoped load", exc_info=True)
        return cfg


async def _discover_gateway_mcp_tools(config: object) -> None:
    """Run startup MCP discovery for every profile this gateway serves.

    ``discover_mcp_tools`` reads ``mcp_servers`` from ``get_hermes_home()``'s config, so an unscoped
    call only connects the launch profile's servers. Single-profile gateways keep the unscoped call.
    """
    from tools.mcp_tool import discover_mcp_tools

    loop = asyncio.get_running_loop()
    if not getattr(config, "multiplex_profiles", False):
        await loop.run_in_executor(None, discover_mcp_tools)
        return
    for profile_name, profile_home in _multiplex_profile_homes(config):
        try:
            with _profile_runtime_scope(Path(profile_home)):
                await loop.run_in_executor(None, copy_context().run, discover_mcp_tools)
        except Exception:
            logger.warning("MCP tool discovery failed for profile '%s'", profile_name, exc_info=True)


def _platform_has_bot_credential(platform: "Platform", platform_config: "PlatformConfig") -> bool:
    """Return True when a token-authenticated platform has a usable bot credential.

    Platforms that do not use ``PlatformConfig.token`` always return True so we
    never skip them here (Signal session paths, port-binding HTTP adapters, etc.).
    """
    from gateway.config import PLATFORM_TOKEN_ENV_NAMES, Platform

    if platform not in PLATFORM_TOKEN_ENV_NAMES:
        return True
    token = getattr(platform_config, "token", None) or ""
    if isinstance(token, str) and token.strip():
        return True
    # Some adapters also accept api_key as the primary credential.
    api_key = getattr(platform_config, "api_key", None) or ""
    if isinstance(api_key, str) and api_key.strip():
        return True
    # Matrix also authenticates by password (MATRIX_USER_ID + MATRIX_PASSWORD in ``extra``); a
    # token-only check would evict a reconnectable config from the retry queue on the first transient
    # failure. Mirror the adapter's gate, reading ONLY extra (build_config() copies env vars there):
    # an env fallback would report "has credential" for every Matrix config (incl. empty-primary mux).
    if platform is Platform.MATRIX:
        extra = getattr(platform_config, "extra", None) or {}
        if all(str(extra.get(key) or "").strip() for key in ("homeserver", "user_id", "password")):
            return True
    return False


_DOCKER_VOLUME_SPEC_RE = re.compile(r"^(?P<host>.+):(?P<container>/[^:]+?)(?::(?P<options>[^:]+))?$")
_DOCKER_MEDIA_OUTPUT_CONTAINER_PATHS = {"/output", "/outputs"}

# Internal bridge plumbing, not a user-facing config source: initialize from the canonical config
# default after dotenv loading so an ambient process/.env value can never control lease safety.
from hermes_cli.config_defaults import DEFAULT_CONFIG as _DEFAULT_CONFIG

os.environ["HERMES_TURN_LEASE_TIMEOUT"] = str(
    _DEFAULT_CONFIG["agent"]["gateway_turn_lease_timeout"])

# Bridge config.yaml values into env so os.getenv() picks them up. config.yaml unconditionally wins
# over .env for these keys; a `not in os.environ` guard would let stale .env entries shadow config.
_AGENT_ENV_BRIDGE = {
    "gateway_timeout": "HERMES_AGENT_TIMEOUT",
    "gateway_turn_lease_timeout": "HERMES_TURN_LEASE_TIMEOUT",
    "gateway_timeout_warning": "HERMES_AGENT_TIMEOUT_WARNING",
    "gateway_notify_interval": "HERMES_AGENT_NOTIFY_INTERVAL",
    "session_stall_timeout": "HERMES_SESSION_STALL_TIMEOUT",
    # Internal bridge only — config.yaml (agent.reconnect_attention_after) is the documented setting.
    "reconnect_attention_after": "HERMES_RECONNECT_ATTENTION_AFTER_SECONDS",
    "restart_drain_timeout": "HERMES_RESTART_DRAIN_TIMEOUT",
    "cron_drain_timeout": "HERMES_CRON_DRAIN_TIMEOUT",
    "gateway_auto_continue_freshness": "HERMES_AUTO_CONTINUE_FRESHNESS",
    "gateway_startup_restore_drain_timeout": "HERMES_STARTUP_RESTORE_DRAIN_TIMEOUT",
    "gateway_startup_warmup_timeout": "HERMES_STARTUP_WARMUP_TIMEOUT"}
# config-authoritative knobs for the session-search index (env stays the cross-process carrier).
_SESSIONS_ENV_BRIDGE = {"cjk_fts": "HERMES_CJK_FTS", "search_slow_ms": "HERMES_SEARCH_SLOW_MS"}
_DISPLAY_ENV_BRIDGE = {
    "busy_input_mode": "HERMES_GATEWAY_BUSY_INPUT_MODE",
    "busy_text_mode": "HERMES_GATEWAY_BUSY_TEXT_MODE",
    "busy_ack_enabled": "HERMES_GATEWAY_BUSY_ACK_ENABLED"}


def _bridge_section_to_env(section: Any, mapping: Dict[str, str]) -> None:
    """Export every present ``mapping`` key of a config section as ``str(value)``."""
    if isinstance(section, dict):
        for cfg_key, env_var in mapping.items():
            if cfg_key in section:
                os.environ[env_var] = str(section[cfg_key])


def _bridge_max_turns_to_env(agent_cfg: Any) -> None:
    """Bridge ``agent.max_turns`` preserving its raw spelling ("none", "unlimited", "120").

    Python None (`null` / bare `key:`) clears a stale bridge instead: str(None) -> "None" would map
    to the unlimited sentinel rather than "absent = default".
    """
    if not isinstance(agent_cfg, dict) or "max_turns" not in agent_cfg:
        return
    raw = agent_cfg["max_turns"]
    if raw is not None:
        os.environ["HERMES_MAX_ITERATIONS"] = str(raw)
    elif "HERMES_MAX_ITERATIONS" in os.environ:
        del os.environ["HERMES_MAX_ITERATIONS"]


def _bridge_terminal_config_to_env(_terminal_cfg: dict) -> None:
    """Bridge nested ``terminal.*`` config to TERMINAL_* env vars (config.yaml overrides .env here)."""
    _terminal_backend = str(
        _terminal_cfg.get("backend") or os.environ.get("TERMINAL_ENV") or "").strip().lower()
    _terminal_env_map = {
        "backend": "TERMINAL_ENV",
        "degraded_mode": "TERMINAL_DEGRADED_MODE",
        "cwd": "TERMINAL_CWD",
        "timeout": "TERMINAL_TIMEOUT",
        "home_mode": "TERMINAL_HOME_MODE",
        "lifetime_seconds": "TERMINAL_LIFETIME_SECONDS",
        "docker_image": "TERMINAL_DOCKER_IMAGE",
        "docker_forward_env": "TERMINAL_DOCKER_FORWARD_ENV",
        "singularity_image": "TERMINAL_SINGULARITY_IMAGE",
        "modal_image": "TERMINAL_MODAL_IMAGE",
        "daytona_image": "TERMINAL_DAYTONA_IMAGE",
        "vercel_runtime": "TERMINAL_VERCEL_RUNTIME",
        "ssh_host": "TERMINAL_SSH_HOST",
        "ssh_user": "TERMINAL_SSH_USER",
        "ssh_port": "TERMINAL_SSH_PORT",
        "ssh_key": "TERMINAL_SSH_KEY",
        "container_cpu": "TERMINAL_CONTAINER_CPU",
        "container_memory": "TERMINAL_CONTAINER_MEMORY",
        "container_disk": "TERMINAL_CONTAINER_DISK",
        "container_persistent": "TERMINAL_CONTAINER_PERSISTENT",
        "docker_volumes": "TERMINAL_DOCKER_VOLUMES",
        "docker_env": "TERMINAL_DOCKER_ENV",
        "docker_extra_args": "TERMINAL_DOCKER_EXTRA_ARGS",
        "docker_shm_size": "TERMINAL_DOCKER_SHM_SIZE",
        "docker_mount_cwd_to_workspace": "TERMINAL_DOCKER_MOUNT_CWD_TO_WORKSPACE",
        "docker_network": "TERMINAL_DOCKER_NETWORK",
        "docker_run_as_host_user": "TERMINAL_DOCKER_RUN_AS_HOST_USER",
        "docker_persist_across_processes": "TERMINAL_DOCKER_PERSIST_ACROSS_PROCESSES",
        "docker_shared_container_key": "TERMINAL_DOCKER_SHARED_CONTAINER_KEY",
        "docker_orphan_reaper": "TERMINAL_DOCKER_ORPHAN_REAPER",
        "sandbox_dir": "TERMINAL_SANDBOX_DIR",
        "persistent_shell": "TERMINAL_PERSISTENT_SHELL"}
    for _cfg_key, _env_var in _terminal_env_map.items():
        if _cfg_key not in _terminal_cfg:
            continue
        _val = _terminal_cfg[_cfg_key]
        if _cfg_key == "cwd":
            # Placeholders (".", "auto", "cwd") resolve to Path.home() later; only explicit paths bridge.
            if str(_val) in {".", "auto", "cwd"}:
                continue
            # Expand "~" for local/container cwd so Popen never gets a literal "~/" (kernel rejects it);
            # SSH cwd is interpreted by the remote shell: keep "~". Predicate shared w/ terminal_tool.
            if isinstance(_val, str) and not _is_ssh_remote_tilde_cwd(_terminal_backend, _val.strip()):
                _val = os.path.expanduser(_val)
        os.environ[_env_var] = json.dumps(_val) if isinstance(_val, (list, dict)) else str(_val)


def _bridge_auxiliary_config_to_env(_auxiliary_cfg: dict) -> None:
    """Bridge auxiliary model/endpoint overrides (vision, approval, plugin tasks); compression reads
    config.yaml directly."""
    _aux_bridged_keys = {"vision", "approval"}
    try:
        from hermes_cli.plugins import get_plugin_auxiliary_tasks
        for _entry in get_plugin_auxiliary_tasks():
            _aux_bridged_keys.add(_entry["key"])
    except Exception:
        pass  # plugin discovery failure must not break startup; built-in bridging stays intact
    for _task_key in _aux_bridged_keys:
        _task_cfg = _auxiliary_cfg.get(_task_key, {})
        if not isinstance(_task_cfg, dict):
            continue
        _upper = _task_key.upper()
        _prov = str(_task_cfg.get("provider", "")).strip()
        if _prov and _prov != "auto":
            os.environ[f"AUXILIARY_{_upper}_PROVIDER"] = _prov
        for _field, _suffix in (("model", "MODEL"), ("base_url", "BASE_URL"), ("api_key", "API_KEY")):
            _value = str(_task_cfg.get(_field, "")).strip()
            if _value:
                os.environ[f"AUXILIARY_{_upper}_{_suffix}"] = _value


def _bridge_config_to_env(_cfg: dict) -> None:
    """Export config.yaml settings to the env vars os.getenv() consumers read."""
    for _key, _val in _cfg.items():  # top-level scalars: fallback only, never override .env
        if isinstance(_val, (str, int, float, bool)) and _key not in os.environ:
            os.environ[_key] = str(_val)
    _terminal_cfg = _cfg.get("terminal", {})
    if _terminal_cfg and isinstance(_terminal_cfg, dict):
        _bridge_terminal_config_to_env(_terminal_cfg)
    _auxiliary_cfg = _cfg.get("auxiliary", {})
    if _auxiliary_cfg and isinstance(_auxiliary_cfg, dict):
        _bridge_auxiliary_config_to_env(_auxiliary_cfg)
    _agent_cfg = _cfg.get("agent", {})
    _bridge_max_turns_to_env(_agent_cfg)
    _bridge_section_to_env(_agent_cfg, _AGENT_ENV_BRIDGE)
    _bridge_section_to_env(_cfg.get("sessions", {}), _SESSIONS_ENV_BRIDGE)
    _display_cfg = _cfg.get("display", {})
    _bridge_section_to_env(_display_cfg, _DISPLAY_ENV_BRIDGE)
    if (
        isinstance(_display_cfg, dict)
        # Documented service-manager override: env wins when already set (other display bridges stay
        # config-authoritative for backwards compatibility).
        and "busy_steer_ack_enabled" in _display_cfg
        and "HERMES_GATEWAY_BUSY_STEER_ACK_ENABLED" not in os.environ):
        os.environ["HERMES_GATEWAY_BUSY_STEER_ACK_ENABLED"] = str(_display_cfg["busy_steer_ack_enabled"])
    _tz_cfg = _cfg.get("timezone", "")
    if _tz_cfg and isinstance(_tz_cfg, str):
        os.environ["HERMES_TIMEZONE"] = _tz_cfg.strip()
    _security_cfg = _cfg.get("security", {})
    if isinstance(_security_cfg, dict) and _security_cfg.get("redact_secrets") is not None:
        os.environ["HERMES_REDACT_SECRETS"] = str(_security_cfg["redact_secrets"]).lower()
    # Media policy uses the shared bridge so standalone entrypoints (`hermes cron run`) match.
    _gateway_cfg = _cfg.get("gateway", {})
    if isinstance(_gateway_cfg, dict):
        from gateway.media_policy import apply_media_policy_env

        apply_media_policy_env(_cfg)
        _trust_recent_seconds = _gateway_cfg.get("trust_recent_files_seconds")
        if _trust_recent_seconds is not None:
            os.environ["HERMES_MEDIA_TRUST_RECENT_SECONDS"] = str(_trust_recent_seconds)
        # platform_connect_timeout is an escape hatch, unlike the bridges above: env WINS if already set.
        if (
            "platform_connect_timeout" in _gateway_cfg
            and not os.environ.get("HERMES_GATEWAY_PLATFORM_CONNECT_TIMEOUT", "").strip()):
            os.environ["HERMES_GATEWAY_PLATFORM_CONNECT_TIMEOUT"] = str(_gateway_cfg["platform_connect_timeout"])


def _load_bridge_config(config_path: Path) -> dict:
    """Raw config read for the presence-sensitive env bridge, with the managed overlay applied.

    Raw (not defaults-merged) so only keys the user wrote are bridged, else all of DEFAULT_CONFIG
    would be exported. Managed overlay applies BEFORE bridging so pinned values win in env too.
    """
    from hermes_cli.config import _expand_env_vars, read_user_config_raw
    cfg = _expand_env_vars(read_user_config_raw(config_path))
    if not isinstance(cfg, dict):
        cfg = {}
    try:
        from hermes_cli import managed_scope
        cfg = managed_scope.apply_managed_overlay(cfg)
    except Exception:
        pass
    return cfg


_config_path = _hermes_home / 'config.yaml'
_cfg: dict = {}
if _config_path.exists():
    try:
        _cfg = _load_bridge_config(_config_path)
        _bridge_config_to_env(_cfg)
    except Exception as _bridge_err:
        # stderr, not logger: the module logger is not initialized yet at import time.
        print(
            f"  Warning: config.yaml → env bridge failed: {type(_bridge_err).__name__}: {_bridge_err}",
            file=sys.stderr)
        print(
            "  Gateway will fall back to .env values, which may not match "
            "your current config.yaml. Run `hermes doctor` to investigate.",
            file=sys.stderr)

# IPv4 preference must apply before any HTTP clients are created.
try:
    from hermes_constants import apply_ipv4_preference
    _network_cfg = _cfg.get("network", {})
    if isinstance(_network_cfg, dict) and _network_cfg.get("force_ipv4"):
        apply_ipv4_preference(force=True)
except Exception as _bootstrap_exc:
    print(f"  Warning: IPv4 preference application failed: {_bootstrap_exc}", file=sys.stderr)

try:
    from hermes_cli.config import print_config_warnings
    print_config_warnings()
except Exception as _bootstrap_exc:
    print(f"  Warning: config validation failed: {_bootstrap_exc}", file=sys.stderr)

try:
    from hermes_cli.config import warn_deprecated_cwd_env_vars
    warn_deprecated_cwd_env_vars()
except Exception as _bootstrap_exc:
    print(f"  Warning: deprecation check failed: {_bootstrap_exc}", file=sys.stderr)

os.environ["HERMES_QUIET"] = "1"  # gateway runs quiet: no debug output, cwd used directly

# HERMES_EXEC_ASK is set in start_gateway(), NOT at import: CLI tools importing this module must not
# flip interactive sessions into ask-mode (approval prompts would become silent pending_approval).

# Terminal cwd: config.yaml terminal.cwd is canonical (bridged above); MESSAGING_CWD is legacy fallback.
from gateway.cwd_placeholder import CWD_PLACEHOLDERS, resolve_placeholder_terminal_cwd

_configured_cwd = os.environ.get("TERMINAL_CWD", "")
if not _configured_cwd or _configured_cwd in CWD_PLACEHOLDERS:
    _resolved_cwd = resolve_placeholder_terminal_cwd(
        configured_cwd=_configured_cwd,
        terminal_backend=os.environ.get("TERMINAL_ENV", ""),
        messaging_cwd=os.getenv("MESSAGING_CWD"),
        docker_mount_cwd_to_workspace=os.getenv(
            "TERMINAL_DOCKER_MOUNT_CWD_TO_WORKSPACE", "false").lower()
        in {"true", "1", "yes"},
        home_fallback=str(Path.home()))
    if _resolved_cwd is None:
        os.environ.pop("TERMINAL_CWD", None)
    else:
        os.environ["TERMINAL_CWD"] = _resolved_cwd

from gateway.config import (
    ChannelOverride, Platform, GatewayConfig, PlatformConfig, _getenv, load_gateway_config)
from gateway.session import (
    AsyncSessionStore, SessionStore, SessionSource, SessionContext, build_session_key)
from gateway.delivery import (
    DeliveryRouter,
    resolve_delivery_transport,  # noqa: F401  (re-exported: run_* mixins + tests resolve gateway.run.<name>)
)
from gateway.turn_lease import SessionTurnLeaseRegistry
from gateway.session_state import SessionState, legacy_dict_property, legacy_lease_token_property
from gateway.authz_mixin import GatewayAuthorizationMixin
from gateway.kanban_watchers import GatewayKanbanWatchersMixin
from gateway.slash_commands import GatewaySlashCommandsMixin
from gateway.run_voice import GatewayVoiceMixin
from gateway.run_adapters import GatewayAdapterLifecycleMixin
from gateway.run_topics import GatewayTopicThreadsMixin
from gateway.run_turn import GatewayTurnMixin
from gateway.run_shutdown import GatewayShutdownMixin
from gateway.run_busy import GatewayBusySessionMixin
from gateway.run_config_loaders import GatewayConfigLoadersMixin
from gateway.run_startup import GatewayStartupMixin
from gateway.run_watchers import GatewaySessionWatchersMixin
from gateway.run_notifications import GatewayNotificationsMixin
from gateway.run_inbound import GatewayInboundMixin
from gateway.run_goals import GatewayGoalsMixin
from gateway.run_agent_cache import GatewayAgentCacheMixin
from gateway.run_turn_runner import TurnRunner  # noqa: F401  (re-exported; run.py callers + tests)
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    _reply_anchor_for_event,
    merge_pending_message_event,  # noqa: F401  (re-exported: run_* mixins + tests resolve gateway.run.<name>)
    )
from gateway.shutdown_watchdog import (
    _arm_loop_floor_timer,  # noqa: F401  (re-exported: run_* mixins + tests resolve gateway.run.<name>)
    start_loop_liveness_watchdog,  # noqa: F401  (re-exported: run_* mixins + tests resolve gateway.run.<name>)
)
from gateway.restart import (
    DEFAULT_GATEWAY_CRON_DRAIN_TIMEOUT,
    DEFAULT_GATEWAY_POST_INTERRUPT_GRACE_TIMEOUT,  # noqa: F401  (re-exported: run_* mixins + tests resolve gateway.run.<name>)
    DEFAULT_GATEWAY_RESTART_AFTER_TURN_TIMEOUT,
    DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT,
    DEFAULT_GATEWAY_SIGNAL_INTERRUPT_GRACE_TIMEOUT)


logger = logging.getLogger(__name__)


def _best_effort(fn: Callable[[], Any], debug_msg: Optional[str] = None) -> Any:
    """Call ``fn``; return None on any Exception (debug-logged via ``debug_msg`` ``%s`` if given)."""
    try:
        return fn()
    except Exception as exc:
        if debug_msg:
            logger.debug(debug_msg, exc)
        return None


# Shutdown quiesce ceiling for the gateway-owned thread pool. Drain already waited for the agents;
# what remains is short blocking work — anything slower is a stuck worker we must not wait on
# (the caller clamps this to the watchdog leash anyway).
_EXECUTOR_QUIESCE_TIMEOUT = 2.0


_OWN_POLICY_OPEN_ENV = {
    Platform.WECOM: ("WECOM_DM_POLICY", "WECOM_GROUP_POLICY", "WECOM_ALLOW_ALL_USERS"),
    Platform.WEIXIN: ("WEIXIN_DM_POLICY", "WEIXIN_GROUP_POLICY", "WEIXIN_ALLOW_ALL_USERS"),
    Platform.YUANBAO: ("YUANBAO_DM_POLICY", "YUANBAO_GROUP_POLICY", "YUANBAO_ALLOW_ALL_USERS"),
    Platform.QQBOT: (None, None, "QQ_ALLOW_ALL_USERS"),
    Platform.WHATSAPP: ("WHATSAPP_DM_POLICY", "WHATSAPP_GROUP_POLICY", "WHATSAPP_ALLOW_ALL_USERS")}


def _own_policy_open_startup_violation(config) -> Optional[str]:
    """Return a startup-abort reason when open policy lacks allow-all opt-in."""
    for platform, platform_config in getattr(config, "platforms", {}).items():
        if not getattr(platform_config, "enabled", False):
            continue
        open_env = _OWN_POLICY_OPEN_ENV.get(platform)
        if not open_env:
            continue
        dm_env, group_env, allow_all_env = open_env
        extra = getattr(platform_config, "extra", None) or {}
        dm_policy = str(
            extra.get("dm_policy") or (_getenv(dm_env, "pairing") if dm_env else "pairing")
        ).strip().lower()
        group_policy = str(
            extra.get("group_policy") or (_getenv(group_env, "pairing") if group_env else "pairing")
        ).strip().lower()
        if dm_policy != "open" and group_policy != "open":
            continue
        gateway_allow_all = _getenv("GATEWAY_ALLOW_ALL_USERS", "").lower() in {"true", "1", "yes"}
        platform_opted_in = gateway_allow_all or (
            allow_all_env and _getenv(allow_all_env, "").lower() in {"true", "1", "yes"})
        if platform_opted_in:
            continue
        return f"{platform.value}: open policy without allow-all opt-in"
    return None


# Sentinel placed into _running_agents *before* any await when a session starts processing, so a
# second message can't slip past the "already running" guard before the agent actually exists.
_AGENT_PENDING_SENTINEL = object()

# Conversation-scoped per-session state registry (legacy contract). State lives in
# ``SessionState.conversation`` and boundaries clear it via ``ConversationState.clear()``; this list
# remains for plain-dict stores not yet folded in (``_pending_model_notes``, popped per-key by
# _clear_conversation_scope) and the public test contract. NOT listed (different lifecycles):
# turn-scoped _running_agents*/_active_session_leases/_busy_ack_ts/_turn_lease_tokens (owned by
# _release_running_agent_state + dispatch finally); _session_run_generation (monotonic — clearing
# breaks stale-run detection); _agent_cache (_evict_cached_agent); approval/slash-confirm state
# (_clear_session_boundary_security_state).
_CONVERSATION_SCOPED_STATE: tuple = (
    "_session_model_overrides",
    "_pending_one_turn_model_restores",
    "_session_reasoning_overrides",
    "_session_service_tier_overrides",
    "_pending_model_notes",
    "_last_resolved_model",
    "_queued_events",
    # Stall-watchdog "already notified" latch; cleared on /new so a fresh conversation can warn again.
    "_session_stall_notified",
    # Sidecar notes staged but never consumed (turn aborted before run_sync) must not leak into a
    # future conversation's first user message — session keys are source-derived and REUSED.
    "_pending_turn_sidecar_notes")

from gateway.run_common import _UNSET  # noqa: F401  (def-time sentinel shared with run_* mixins)


def _resolve_runtime_agent_kwargs() -> dict:
    """Resolve provider credentials for gateway-created AIAgent instances.

    ``resolve_runtime_provider()`` falls through to env vars for legacy compatibility, but the
    gateway never consults env vars for behavioral config — config.yaml is authoritative.
    """
    from hermes_cli.runtime_provider import (
        resolve_runtime_provider, format_runtime_provider_error, _get_model_config)
    from hermes_cli.auth import AuthError, is_rate_limited_auth_error

    try:
        runtime = resolve_runtime_provider()
    except AuthError as auth_exc:
        # Distinguish a rate-limit/quota cap (credentials fine, re-auth can't help) from a real auth
        # failure (expired/revoked token): both use the fallback chain; the log must not mislabel.
        if is_rate_limited_auth_error(auth_exc):
            logger.warning("Primary provider rate-limited (429): %s — trying fallback", auth_exc)
        else:
            logger.warning("Primary provider auth failed: %s — trying fallback", auth_exc)
        fb_config = _try_resolve_fallback_provider()
        if fb_config is not None:
            return fb_config
        raise RuntimeError(format_runtime_provider_error(auth_exc)) from auth_exc
    except Exception as exc:
        raise RuntimeError(format_runtime_provider_error(exc)) from exc

    model_cfg = _get_model_config()
    max_tokens = None
    _env_mt = os.environ.get("HERMES_MAX_TOKENS")
    if _env_mt:
        try:
            max_tokens = int(_env_mt)
        except (ValueError, TypeError):
            max_tokens = None
    elif isinstance(model_cfg, dict):
        mt = model_cfg.get("max_tokens")
        if isinstance(mt, int):
            max_tokens = mt
    # Per-provider output cap (custom_providers max_output_tokens) applies only when the documented
    # global model.max_tokens is unset, so the global key always wins.
    if max_tokens is None:
        _runtime_mot = runtime.get("max_output_tokens")
        if isinstance(_runtime_mot, int) and _runtime_mot > 0:
            max_tokens = _runtime_mot

    capabilities = runtime.get("capabilities")
    capabilities = (
        {k: v for k, v in capabilities.items() if isinstance(k, str) and isinstance(v, bool)}
        if isinstance(capabilities, dict) else {})

    return {**_runtime_agent_kwargs(runtime), "max_tokens": max_tokens, "capabilities": capabilities}


def _runtime_agent_kwargs(runtime: dict) -> dict:
    """AIAgent constructor kwargs shared by every runtime-provider resolution.

    ``request_overrides`` is passed through as resolved (custom_providers ``extra_body`` etc.) so
    the provider's configured request body reaches the per-turn route on the gateway path.
    """
    return {
        "api_key": runtime.get("api_key"),
        "base_url": runtime.get("base_url"),
        "provider": runtime.get("provider"),
        "requested_provider": runtime.get("requested_provider"),
        "api_mode": runtime.get("api_mode"),
        "command": runtime.get("command"),
        "args": list(runtime.get("args") or []),
        "credential_pool": runtime.get("credential_pool"),
        "request_overrides": runtime.get("request_overrides")}


@dataclasses.dataclass(frozen=True)
class _GatewayModelContext:
    """Effective gateway model route and context-window resolution."""

    model: str
    provider: str
    base_url: str
    context_length: int
    context_source: str


def _resolve_gateway_model_context(model: Optional[str] = None) -> _GatewayModelContext:
    """Resolve the configured gateway route and its effective context window.

    Shared by status/session banners and slash commands. Call off the event loop (may block).
    """
    from agent.model_metadata import DEFAULT_FALLBACK_CONTEXT, get_model_context_length

    resolved_model = model or _resolve_gateway_model()
    config_context_length = provider = base_url = api_key = custom_providers = None
    configured_model = configured_provider = configured_base_url = None

    def _read_config() -> None:
        nonlocal config_context_length, provider, base_url, custom_providers
        nonlocal configured_model, configured_provider, configured_base_url
        data = _load_gateway_config()
        if not data:
            return
        model_cfg = data.get("model", {})
        if isinstance(model_cfg, dict):
            configured_model = model_cfg.get("default") or model_cfg.get("model")
            raw_ctx = model_cfg.get("context_length")
            if raw_ctx is not None:
                with suppress(TypeError, ValueError):
                    config_context_length = int(raw_ctx)
            configured_provider = provider = model_cfg.get("provider") or None
            configured_base_url = base_url = model_cfg.get("base_url") or None
        try:
            from hermes_cli.config import get_compatible_custom_providers
            custom_providers = get_compatible_custom_providers(data)
        except Exception:
            custom_providers = data.get("custom_providers")

    def _read_runtime() -> None:
        nonlocal provider, base_url, api_key
        runtime = _resolve_runtime_agent_kwargs()
        provider = runtime.get("provider") or provider
        base_url = runtime.get("base_url") or base_url
        api_key = runtime.get("api_key")

    def _pin_still_applies() -> bool:
        # Drop a configured context_length pin when the effective route no longer matches (or on error).
        from hermes_cli.route_identity import should_clear_context_pin
        return not should_clear_context_pin(
            configured_model, resolved_model, configured_base_url, base_url, configured_provider, provider)

    def _custom_ctx() -> Optional[int]:
        from hermes_cli.config import get_custom_provider_context_length
        return get_custom_provider_context_length(
            model=resolved_model, base_url=base_url, custom_providers=custom_providers)

    _best_effort(_read_config)
    _best_effort(_read_runtime)
    if config_context_length is not None and not _best_effort(_pin_still_applies):
        config_context_length = None
    if config_context_length is None and custom_providers and base_url:
        config_context_length = _best_effort(_custom_ctx) or None

    context_length = get_model_context_length(
        resolved_model, base_url=base_url or "", api_key=api_key or "",
        config_context_length=config_context_length, provider=provider or "",
        custom_providers=custom_providers)
    if config_context_length is not None:
        context_source = "config"
    elif context_length == DEFAULT_FALLBACK_CONTEXT:
        context_source = "default"
    else:
        context_source = "detected"

    return _GatewayModelContext(
        model=resolved_model, provider=provider or "", base_url=base_url or "",
        context_length=context_length, context_source=context_source)


def _resolve_runtime_agent_kwargs_for_provider(provider: str) -> dict:
    """Resolve runtime credentials for a specific provider (e.g. from channel override)."""
    from hermes_cli.runtime_provider import resolve_runtime_provider, format_runtime_provider_error
    try:
        runtime = resolve_runtime_provider(requested=provider)
    except Exception as exc:
        raise RuntimeError(format_runtime_provider_error(exc)) from exc
    return {
        **_runtime_agent_kwargs(runtime),
        "request_overrides": dict(runtime.get("request_overrides") or {}),
        "capabilities": dict(runtime.get("capabilities") or {}),
        "max_tokens": runtime.get("max_output_tokens")}


def _deep_merge_request_overrides(base: Optional[dict], override: Optional[dict]) -> dict:
    """Merge request_overrides dicts, deep-merging nested dictionaries."""
    from hermes_cli.config import _deep_merge

    base_dict = dict(base or {})
    override_dict = dict(override or {})
    if not base_dict:
        return override_dict
    if not override_dict:
        return base_dict
    return _deep_merge(base_dict, override_dict)


def _credential_pool_for_provider(provider: Optional[str]):
    """Return the live credential pool for a provider id (e.g. ``custom:hyper``)."""
    if not provider or not str(provider).strip():
        return None
    try:
        return _resolve_runtime_agent_kwargs_for_provider(str(provider).strip()).get("credential_pool")
    except Exception:
        logger.debug("Failed to resolve credential pool for provider=%s", provider, exc_info=True)
        return None


def _try_resolve_fallback_provider() -> dict | None:
    """Attempt to resolve credentials from the fallback_model/fallback_providers config."""
    from hermes_cli.runtime_provider import resolve_runtime_provider
    try:
        # Canonical loader so managed overlay / ${VAR} expansion reach the fallback chain.
        cfg = _load_gateway_runtime_config()
        fb_list = get_fallback_chain(cfg)
        if not fb_list:
            return None
        for entry in fb_list:
            try:
                from hermes_cli.fallback_config import resolve_entry_api_key

                runtime = resolve_runtime_provider(
                    requested=entry.get("provider"), explicit_base_url=entry.get("base_url"),
                    explicit_api_key=resolve_entry_api_key(entry))
                # Log the config `provider`, not the runtime category (Ollama would log "openrouter").
                logger.info(
                    "Fallback provider resolved: %s model=%s",
                    entry.get("provider") or runtime.get("provider"), entry.get("model"))
                return {**_runtime_agent_kwargs(runtime), "model": entry.get("model")}
            except Exception as fb_exc:
                logger.debug("Fallback entry %s failed: %s", entry.get("provider"), fb_exc)
                continue
    except Exception:
        pass
    return None


def _event_media_type_at(event, index: int) -> str:
    """Per-attachment MIME at *index*; "" when the adapter set only a message-level type."""
    media_types = getattr(event, "media_types", None) or []
    return media_types[index] if index < len(media_types) else ""


def _event_media_kind_is(event, index: int, mime_prefix: str, fallback_types: frozenset) -> bool:
    """Per-attachment MIME first, message-level type only when unknown (else a document uploaded
    alongside an image is base64'd as vision and the provider 400s)."""
    mtype = _event_media_type_at(event, index)
    if mtype:
        return mtype.startswith(mime_prefix)
    return getattr(event, "message_type", None) in fallback_types


def _event_media_is_image(event, index: int) -> bool:
    return _event_media_kind_is(event, index, "image/", frozenset({MessageType.PHOTO}))


def _event_media_is_audio(event, index: int) -> bool:
    return _event_media_kind_is(event, index, "audio/", frozenset({MessageType.VOICE, MessageType.AUDIO}))


def _event_media_is_stt_input(event, index: int) -> bool:
    """True when an audio attachment should enter the automatic STT pipeline."""
    message_type = getattr(event, "message_type", None)
    if message_type in {MessageType.AUDIO, MessageType.DOCUMENT}:
        return False
    return (
        message_type == MessageType.VOICE or _event_media_type_at(event, index).startswith("audio/")
    )


def _event_media_is_video(event, index: int) -> bool:
    return _event_media_kind_is(event, index, "video/", frozenset({MessageType.VIDEO}))


def _build_media_placeholder(event) -> str:
    """Text placeholder for media-only events (later replaced by vision enrichment).

    Queued media is dequeued via .text only, so a caption-less event would otherwise be lost.
    """
    parts = []
    media_urls = getattr(event, "media_urls", None) or []
    for i, url in enumerate(media_urls):
        if _event_media_is_image(event, i):
            parts.append(f"[User sent an image: {url}]")
        elif _event_media_is_audio(event, i):
            parts.append(f"[User sent audio: {url}]")
        elif _event_media_is_video(event, i):
            parts.append(f"[User sent a video: {url}]")
        else:
            parts.append(f"[User sent a file: {url}]")
    return "\n".join(parts)


def _build_document_context_note(
    display_name: str, agent_path: str, mtype: str, *, content_inlined: bool = True) -> str:
    """Context note prepended to a user turn when they attach a document.

    ``content_inlined=False`` = cached without content, so tell the agent to read it. Binary docs
    must say *extract* the text; "ask the user" made it punt.
    """
    if mtype.startswith("text/") and content_inlined:
        return (
            f"[The user sent a text document: '{display_name}'. Its content has been included below. "
            f"The file is also saved at: {agent_path}]")
    if mtype.startswith("text/"):
        return (
            f"[The user sent a text document: '{display_name}'. It is saved at: {agent_path}. "
            f"Its content is not inlined here. Read the cached file yourself before answering "
            f"when the user's request involves its contents.]")
    return (
        f"[The user sent a document: '{display_name}'. It is saved at: {agent_path}. "
        f"Its text is not inlined here (it's a binary format such as PDF or DOCX). "
        f"To read it, extract the document's text yourself — for example with the "
        f"terminal tool or the ocr-and-documents skill — before answering, instead "
        f"of asking the user to paste the contents.]")


def _format_duration(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


async def _probe_audio_duration(path: str) -> Optional[str]:
    """Best-effort duration probe. Returns formatted MM:SS / HH:MM:SS, or None on failure."""
    ext = os.path.splitext(path)[1].lower()

    if ext == ".wav":
        try:
            def _wav_duration() -> float:
                import wave
                with wave.open(path, "rb") as wf:
                    frames = wf.getnframes()
                    rate = wf.getframerate() or 1
                    return frames / float(rate)
            secs = await asyncio.to_thread(_wav_duration)
            return _format_duration(secs)
        except Exception:
            pass

    if ext in (".ogg", ".opus", ".oga"):
        try:
            def _ogg_duration() -> float:
                from mutagen.oggopus import OggOpus
                return float(OggOpus(path).info.length)
            secs = await asyncio.to_thread(_ogg_duration)
            return _format_duration(secs)
        except Exception:
            pass

    try:
        proc = await asyncio.create_subprocess_exec(
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", path,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5.0)
        if proc.returncode == 0:
            return _format_duration(float(stdout.decode().strip()))
    except Exception:
        pass

    return None


def _dequeue_pending_event(adapter, session_key: str) -> MessageEvent | None:
    """Consume and return the full pending event for a session.

    Media metadata is kept so follow-ups re-enter normal image/STT/document preprocessing.
    """
    return adapter.get_pending_message(session_key)


_INTERRUPT_REASON_STOP = "Stop requested"
_INTERRUPT_REASON_RESET = "Session reset requested"
_INTERRUPT_REASON_TIMEOUT = "Execution timed out (inactivity)"
_INTERRUPT_REASON_SSE_DISCONNECT = "SSE client disconnected"
_INTERRUPT_REASON_GATEWAY_SHUTDOWN = "Gateway shutting down"
_INTERRUPT_REASON_GATEWAY_RESTART = "Gateway restarting"


def _reap_gateway_turn_processes(
    task_id: str, process_baseline, *, source: str,
    is_still_current: Optional[Callable[[], bool]] = None) -> int:
    """Reap only background processes created by one abandoned turn.

    ``task_id`` is session-scoped, so a *replacement* turn can spawn its own process mid-reap;
    ``is_still_current`` (closure over the captured run_generation) lets the caller bail instead of
    killing it — that turn snapshots its own baseline, so nothing stays unreaped.
    """
    if not task_id:
        # Blank task_id (sessionless callers) would match and kill every unrelated empty-task process.
        return 0
    if is_still_current is not None:
        try:
            if not is_still_current():
                logger.debug(
                    "Skipping reap for turn %s (%s): a newer turn already "
                    "claimed this session; it owns its own baseline.", task_id, source)
                return 0
        except Exception:
            logger.debug(
                "is_still_current check failed for turn %s (%s); reaping anyway",
                task_id, source, exc_info=True)

    from tools.process_registry import process_registry

    try:
        killed = process_registry.kill_started_since(task_id, process_baseline, source=source)
    except Exception:
        # Detached daemon thread: an uncaught exception would only reach threading.excepthook.
        logger.warning(
            "Failed to reap background processes for turn %s (%s)", task_id, source, exc_info=True)
        return 0
    if killed:
        logger.warning(
            "Reaped %d background process(es) created by abandoned turn %s (%s)",
            killed, task_id, source)
    return killed


_TURN_STACK_DUMP_FRAME_MARKERS = (
    "run_conversation", "run_sync", "_run_sync_with_timeout_lifecycle",
    "finalize_turn", "end_turn", "run_in_session")


def _dump_wedged_turn_stacks(task_id: str) -> None:
    """Log the stack of every thread that looks like turn work, at reap time.

    The hard interrupt frees the wedged worker before a profiler can attach, so dump BEFORE it.
    Best-effort, bounded (turn-machinery threads only, capped output), never raises.
    """
    try:
        frames = sys._current_frames()
        names = {t.ident: t.name for t in threading.enumerate()}
        dumped = 0
        for ident, frame in frames.items():
            if ident == threading.get_ident():
                continue  # the reaper itself
            stack = traceback.format_stack(frame)
            joined = "".join(stack)
            if not any(marker in joined for marker in _TURN_STACK_DUMP_FRAME_MARKERS):
                continue
            dumped += 1
            if dumped > 8:
                logger.error(
                    "Wedged-turn stack dump for task %s truncated: more than 8 candidate threads",
                    task_id)
                break
            logger.error(
                "Wedged-turn stack dump (task=%s thread=%s ident=%s):\n%s",
                task_id, names.get(ident, "?"), ident, "".join(stack[-25:]))
        if dumped == 0:
            logger.error(
                "Wedged-turn stack dump for task %s: no thread with "
                "turn-machinery frames found (worker may have already exited)", task_id)
    except Exception:
        logger.debug("Wedged-turn stack dump failed", exc_info=True)


def _abandon_timed_out_gateway_turn(
    *, agent_holder, task_id: str, process_baseline, worker_done: threading.Event,
    timeout_fired: threading.Event, cleanup_lock: threading.Lock,
    is_still_current: Optional[Callable[[], bool]] = None) -> bool:
    """Interrupt one timed-out turn and reap only processes it created."""
    with cleanup_lock:
        if worker_done.is_set() or timeout_fired.is_set():
            return False
        timeout_fired.set()

    # BEFORE interrupting: the interrupt frees the blocked frame, destroying the only evidence.
    _dump_wedged_turn_stacks(task_id)

    agent = agent_holder[0] if agent_holder else None
    if agent is not None:
        try:
            request_hard_interrupt(agent, _INTERRUPT_REASON_TIMEOUT)
        except Exception:
            logger.debug("Timed-out agent interrupt failed", exc_info=True)

    try:
        _reap_gateway_turn_processes(
            task_id, process_baseline, source="gateway_turn_timeout",
            is_still_current=is_still_current)
    except Exception:
        logger.warning(
            "Failed to reap background processes for timed-out turn %s", task_id, exc_info=True)
    return True


def _watch_gateway_turn_inactivity(
    *, agent_holder, task_id: str, process_baseline, timeout: float, worker_done: threading.Event,
    timeout_fired: threading.Event, cleanup_lock: threading.Lock, poll_interval: float = 5.0,
    is_still_current: Optional[Callable[[], bool]] = None) -> None:
    """Thread watchdog that remains runnable when gateway asyncio is starved."""
    while not worker_done.wait(max(0.01, poll_interval)):
        agent = agent_holder[0] if agent_holder else None
        if agent is None or not hasattr(agent, "get_activity_summary"):
            continue
        try:
            idle_seconds = float(agent.get_activity_summary().get("seconds_since_activity", 0.0))
        except Exception:
            continue
        if idle_seconds < timeout:
            continue
        _abandon_timed_out_gateway_turn(
            agent_holder=agent_holder, task_id=task_id, process_baseline=process_baseline,
            worker_done=worker_done, timeout_fired=timeout_fired, cleanup_lock=cleanup_lock,
            is_still_current=is_still_current)
        return


_CONTROL_INTERRUPT_MESSAGES = frozenset({
    _INTERRUPT_REASON_STOP.lower(), _INTERRUPT_REASON_RESET.lower(),
    _INTERRUPT_REASON_TIMEOUT.lower(), _INTERRUPT_REASON_SSE_DISCONNECT.lower(),
    _INTERRUPT_REASON_GATEWAY_SHUTDOWN.lower(), _INTERRUPT_REASON_GATEWAY_RESTART.lower()})


def _is_control_interrupt_message(message: Optional[str]) -> bool:
    """Return True when an interrupt message is internal control flow."""
    if not message:
        return False
    normalized = " ".join(str(message).strip().split()).lower()
    return normalized in _CONTROL_INTERRUPT_MESSAGES


def _strip_response_attachments_for_direct_send(response: str, adapter) -> str:
    """Return the visible text portion of a response before direct send().

    Replays only explicit ``MEDIA:`` attachments; bare paths/URLs stay visible (the post-stream
    uploader ignores them). No broad ``MEDIA:`` regex after ``extract_media()``: it deliberately
    preserves protected code spans and unvalidated tags.
    """
    _, cleaned = adapter.extract_media(response)
    return cleaned.replace("[[audio_as_voice]]", "").replace("[[as_document]]", "").strip()


def _skill_slug_from_frontmatter(skill_md: Path) -> tuple[str | None, str | None]:
    """Derive the /command slug and declared frontmatter name from a SKILL.md.

    Matches ``scan_skill_commands``: the slug comes from frontmatter ``name:``, NOT the directory
    name. Returns ``(slug, declared_name)`` or ``(None, None)`` if unreadable or lacking ``name:``.
    """
    try:
        content = skill_md.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None, None
    content = content.lstrip("\ufeff")  # tolerate UTF-8 BOM (Windows editors)
    if not content.startswith("---"):
        return None, None
    end = content.find("\n---", 3)
    if end < 0:
        return None, None
    declared_name: str | None = None
    for line in content[3:end].splitlines():
        line = line.strip()
        if line.startswith("name:"):
            raw = line.split(":", 1)[1].strip()
            if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in {'"', "'"}:
                raw = raw[1:-1]
            declared_name = raw.strip()
            break
    if not declared_name:
        return None, None
    slug = declared_name.lower().replace(" ", "-").replace("_", "-")
    # Mirrors _SKILL_INVALID_CHARS / _SKILL_MULTI_HYPHEN from skill_commands
    slug = re.sub(r"[^a-z0-9-]", "", slug)
    slug = re.sub(r"-{2,}", "-", slug).strip("-")
    if not slug:
        return None, declared_name
    return slug, declared_name


def _check_unavailable_skill(command_name: str) -> str | None:
    """Hint when a command matches a skill that is disabled or optional-install only; else None."""
    normalized = command_name.lower().replace("_", "-")
    try:
        from tools.skills_tool import _get_disabled_skill_names
        from agent.skill_utils import get_all_skills_dirs, is_excluded_skill_path
        disabled = _get_disabled_skill_names()

        for skills_dir in get_all_skills_dirs():
            if not skills_dir.exists():
                continue
            for skill_md in skills_dir.rglob("SKILL.md"):
                if is_excluded_skill_path(skill_md):
                    continue
                slug, declared_name = _skill_slug_from_frontmatter(skill_md)
                if not slug or not declared_name:
                    continue
                # disabled is keyed by the declared frontmatter name (what skills.disabled stores).
                if slug == normalized and declared_name in disabled:
                    return (
                        f"The **{command_name}** skill is installed but disabled.\n"
                        f"Enable it with: `hermes skills config`")

        # Check optional skills (shipped with repo but not installed)
        from hermes_constants import get_optional_skills_dir
        repo_root = Path(__file__).resolve().parent.parent
        optional_dir = get_optional_skills_dir(repo_root / "optional-skills")
        if optional_dir.exists():
            for skill_md in optional_dir.rglob("SKILL.md"):
                if is_excluded_skill_path(skill_md):
                    continue
                slug, _declared = _skill_slug_from_frontmatter(skill_md)
                if not slug or slug != normalized:
                    continue
                # Install path: official/<category>/<name>
                rel = skill_md.parent.relative_to(optional_dir)
                install_path = f"official/{'/'.join(rel.parts)}"
                return (
                    f"The **{command_name}** skill is available but not installed.\n"
                    f"Install it with: `hermes skills install {install_path}`")
    except Exception:
        pass
    return None


def _platform_config_key(platform: "Platform") -> str:
    """Map a Platform enum to its config.yaml key (LOCAL→"cli", rest→enum value)."""
    return "cli" if platform == Platform.LOCAL else platform.value


def _teams_pipeline_plugin_enabled() -> bool:
    """Return True when the standalone Teams pipeline plugin is enabled."""
    config = _load_gateway_config()
    enabled = cfg_get(config, "plugins", "enabled", default=[])
    return isinstance(enabled, list) and ("teams_pipeline" in enabled or "teams-pipeline" in enabled)


def _gateway_config_home() -> Path:
    """Return the Hermes home that gateway config reads should use."""
    override = get_hermes_home_override()
    return Path(override) if override else _hermes_home


def _load_gateway_config(config_path: "Path | None" = None) -> dict:
    """Load and parse a gateway config.yaml, returning {} on any error (fail-open).

    Defaults to the active gateway home (``_hermes_home`` monkeypatches apply); multiplexed callers
    may pass a profile path.
    """
    if config_path is None:
        config_path = _gateway_config_home() / 'config.yaml'
    raw: dict = {}
    used_canonical = False
    try:
        from hermes_cli.config import get_config_path, read_raw_config
        # Fast path via shared cache when the path is canonical; else direct read (test monkeypatches).
        if config_path == get_config_path():
            raw = read_raw_config()
            used_canonical = True
    except Exception:
        pass

    if not used_canonical:
        try:
            if config_path.exists():
                import yaml
                with open(config_path, 'r', encoding='utf-8') as f:
                    raw = yaml.safe_load(f) or {}
        except Exception:
            logger.debug("Could not load gateway config from %s", config_path)
            raw = {}

    # Neither read_raw_config() nor yaml.safe_load carries the managed merge; overlay on both paths.
    try:
        from hermes_cli import managed_scope
        raw = managed_scope.apply_managed_overlay(raw if isinstance(raw, dict) else {})
    except Exception:
        pass
    if not isinstance(raw, dict):
        return {}
    # Canonicalize model-id aliases (model.name/model.model → model.default) and migrate stale root
    # provider/base_url: the gateway bypasses load_config(), else ``model: {name: <id>}`` is empty.
    try:
        from hermes_cli.config import _normalize_root_model_keys
        raw = _normalize_root_model_keys(raw)
    except Exception:
        pass
    return raw


def _checkpoint_agent_kwargs(config: dict | None) -> dict:
    """Translate gateway checkpoint config into ``AIAgent`` constructor args.

    Gateway bypasses ``load_config()``, so defaults are here; legacy ``checkpoints: true`` works.
    """
    cp_cfg = config.get("checkpoints", {}) if isinstance(config, dict) else {}
    if isinstance(cp_cfg, bool):
        cp_cfg = {"enabled": cp_cfg}
    elif not isinstance(cp_cfg, dict):
        cp_cfg = {}

    from hermes_cli.config import DEFAULT_CONFIG
    defaults = DEFAULT_CONFIG["checkpoints"]
    return {
        "checkpoints_enabled": cp_cfg.get("enabled", defaults["enabled"]),
        "checkpoint_max_snapshots": cp_cfg.get("max_snapshots", defaults["max_snapshots"]),
        "checkpoint_max_total_size_mb": cp_cfg.get("max_total_size_mb", defaults["max_total_size_mb"]),
        "checkpoint_max_file_size_mb": cp_cfg.get("max_file_size_mb", defaults["max_file_size_mb"])}


def _load_gateway_runtime_config() -> dict:
    """Load gateway config for runtime reads, expanding supported ``${VAR}`` refs.

    Built on ``_load_gateway_config()``. Expansion failures are deliberately NOT swallowed —
    returning the unexpanded dict would mask the very bug this helper fixes.
    """
    cfg = _load_gateway_config()
    if not isinstance(cfg, dict) or not cfg:
        return {}
    from hermes_cli.config import _expand_env_vars

    expanded = _expand_env_vars(cfg)
    return expanded if isinstance(expanded, dict) else {}


def _resolve_gateway_model(config: dict | None = None) -> str:
    """Read model from config.yaml (single source of truth), else temporary AIAgents (e.g. /compress)
    use the hardcoded default, which fails under openai-codex."""
    cfg = config if config is not None else _load_gateway_config()
    model_cfg = cfg.get("model", {})
    if isinstance(model_cfg, str):
        return model_cfg
    elif isinstance(model_cfg, dict):
        return model_cfg.get("default") or model_cfg.get("model") or ""
    return ""


def _channel_override_lookup_keys(
    chat_id: str, *, thread_id: Optional[str] = None, parent_id: Optional[str] = None) -> list[str]:
    """Ordered, de-duplicated ``channel_overrides`` lookup keys (matches ``resolve_channel_prompt``:
    exact id first, then parent — Discord threads inherit parent overrides)."""
    keys: list[str] = []
    seen: set[str] = set()
    for key in (chat_id, thread_id, parent_id):
        if not key:
            continue
        sk = str(key)
        if sk in seen:
            continue
        seen.add(sk)
        keys.append(sk)
    return keys


def _get_channel_override(
    config: GatewayConfig, platform: Platform, chat_id: str, *, thread_id: Optional[str] = None,
    parent_id: Optional[str] = None) -> Optional[ChannelOverride]:
    """Per-channel override via chat_id, then thread_id, then parent_id; None if absent."""
    platforms = getattr(config, "platforms", None)
    if not platforms:
        return None
    platform_config = platforms.get(platform)
    if not platform_config or not platform_config.channel_overrides:
        return None
    overrides = platform_config.channel_overrides
    for key in _channel_override_lookup_keys(chat_id, thread_id=thread_id, parent_id=parent_id):
        ov = overrides.get(key)
        if ov is not None:
            return ov
    return None


def _resolve_hermes_bin() -> Optional[list[str]]:
    """Hermes update command argv: ``hermes`` on PATH, else ``python -m hermes_cli.main``, else None."""
    import shutil

    hermes_bin = shutil.which("hermes")
    if hermes_bin:
        return [hermes_bin]

    try:
        import importlib.util

        if importlib.util.find_spec("hermes_cli") is not None:
            return [sys.executable, "-m", "hermes_cli.main"]
    except Exception:
        pass

    return None


def _parse_session_key(session_key: str) -> "dict | None":
    """Parse a session key (``agent:main:{platform}:{chat_type}:{chat_id}[:{extra}...]``).

    For group/channel sessions the suffix may be a user_id (per-user isolation), not a thread_id,
    so ``thread_id`` is left out to avoid mis-routing.
    """
    parts = session_key.split(":")
    if len(parts) >= 5 and parts[0] == "agent" and parts[1] == "main":
        result = {"platform": parts[2], "chat_type": parts[3], "chat_id": parts[4]}
        if len(parts) > 5 and parts[3] in {"dm", "thread"}:
            result["thread_id"] = parts[5]
        return result
    return None


def _shorten_command_for_display(command: str, limit: int = 80) -> str:
    """Collapse a shell command onto one line and cap its length for display."""
    one_line = " ".join((command or "").split())
    if len(one_line) > limit:
        one_line = one_line[: limit - 1] + "…"
    return one_line


def _format_concise_process_notification(
    session_id: str, command: str, exit_code, output: str, duration_seconds=None) -> str:
    """One-line completion message for the ``concise`` display mode.

    Success is one status line; failure appends a short output tail (full output via process(log)).
    """
    ok = exit_code in {0, None}
    icon = "✅" if ok else "❌"
    verb = "finished" if ok else f"failed (exit {exit_code})"
    parts = [f"{icon} Background task {verb}"]
    short_cmd = _shorten_command_for_display(command)
    if short_cmd:
        parts.append(f"— `{short_cmd}`")
    if isinstance(duration_seconds, (int, float)) and duration_seconds >= 0:
        secs = int(duration_seconds)
        if secs >= 3600:
            dur = f"{secs // 3600}h {(secs % 3600) // 60}m"
        elif secs >= 60:
            dur = f"{secs // 60}m {secs % 60}s"
        else:
            dur = f"{secs}s"
        parts.append(f"({dur})")
    text = " ".join(parts)
    if not ok and output:
        tail_lines = [ln for ln in output.strip().splitlines() if ln.strip()][-5:]
        tail = "\n".join(tail_lines)
        if len(tail) > 500:
            tail = tail[-500:]
        if tail:
            text += f"\n```\n{tail}\n```"
    return text


def _format_gateway_process_notification(evt: dict) -> "str | None":
    """Format a watch pattern event from completion_queue into a [IMPORTANT:] message."""
    evt_type = evt.get("type", "completion")
    _sid = evt.get("session_id", "unknown")
    _cmd = evt.get("command", "unknown")

    # watch_disabled / overflow events carry their summary in `message` (process_registry formatter).
    if evt_type in ("watch_disabled", "watch_overflow_tripped", "watch_overflow_released"):
        return f"[IMPORTANT: {evt.get('message', '')}]"

    if evt_type == "watch_match":
        _pat = evt.get("pattern", "?")
        _out = evt.get("output", "")
        _sup = evt.get("suppressed", 0)
        text = (
            f"[IMPORTANT: Background process {_sid} matched "
            f"watch pattern \"{_pat}\".\n"
            f"Command: {_cmd}\nMatched output:\n{_out}")
        if _sup:
            text += f"\n({_sup} earlier matches were suppressed by rate limit)"
        text += "]"
        return text

    if evt_type == "async_delegation":
        from tools.process_registry import format_process_notification
        return format_process_notification(evt)

    return None


def _drain_gateway_watch_events(completion_queue) -> "list[dict]":
    """Drain gateway-owned watch events without spinning on requeued events.

    Foreign events (process completions, async delegations) requeued inside ``while not
    queue.empty()`` never terminate, so detach the batch first and requeue afterwards.
    """
    watch_events: list[dict] = []
    requeue: list[dict] = []
    while not completion_queue.empty():
        try:
            evt = completion_queue.get_nowait()
        except Exception:
            break
        evt_type = evt.get("type", "completion")
        if evt_type in {
            "watch_match", "watch_disabled", "watch_overflow_tripped", "watch_overflow_released"}:
            watch_events.append(evt)
        elif evt_type == "async_delegation":
            requeue.append(evt)
        # else: process completion events are handled by the watcher task
    for evt in requeue:
        completion_queue.put(evt)
    return watch_events


# Weak ref to the active GatewayRunner; tools like send_message route through its live adapters.
import weakref as _weakref
_gateway_runner_ref: _weakref.ref = lambda: None


def _normalize_empty_agent_response(
    agent_result: dict, response: str, *, history_len: int = 0) -> str:
    """Normalize empty/None agent responses into user-facing messages.

    Covers ``failed``, work done (api_calls > 0) with no text, and never-ran (api_calls == 0,
    the post-/stop silent-drop from a stale generation token) with a retry hint.
    """
    if response:
        return response

    if agent_result.get("failed"):
        # ``error`` can be an EXPLICIT None (bypasses dict.get default) -> would render "failed: None".
        error_detail = agent_result.get("error") or "unknown error"
        error_str = str(error_detail).lower()
        # Persistence failures: suggesting /reset would destroy context without fixing storage.
        failure_reason = str(agent_result.get("failure_reason") or "")
        if failure_reason.startswith("session_persistence_failed") or (
            "session storage" in error_str):
            if failure_reason.endswith(":disk") or "disk" in error_str:
                return (
                    "⚠️ Session storage was temporarily unavailable, so this "
                    "turn was stopped to protect your conversation history. "
                    "Please check available disk space, then send your message again.")
            return (
                "⚠️ Session storage was temporarily unavailable, so this "
                "turn was stopped to protect your conversation history. "
                "Your message should already be saved — please send it again in a moment.")
        is_context_failure = any(
            p in error_str for p in ("context", "token", "too large", "too long", "exceed", "payload")
        ) or ("400" in error_str and history_len > 50)
        if is_context_failure:
            return (
                "⚠️ Session too large for the model's context window.\n"
                "Use /compact to compress the conversation, or /reset to start fresh.")
        return (
            f"The request failed: {str(error_detail)[:300]}\n"
            "Try again or use /reset to start a fresh session.")

    api_calls = int(agent_result.get("api_calls", 0) or 0)
    if agent_result.get("interrupted"):
        # Interrupted with api_calls > 0 = deliberately stopped/steered; silence is intentional (queued
        # messages arrive via the recursive drain). ZERO api_calls = never processed (stale /stop flag).
        if api_calls == 0:
            return (
                "⚠️ Your message was interrupted before processing started "
                "(likely by a recent /stop). Please send it again.")
        return response
    if api_calls > 0:
        if _is_gateway_hidden_reasoning_incomplete_turn(agent_result):
            return ""
        if agent_result.get("partial"):
            err = agent_result.get("error", "processing incomplete")
            return f"⚠️ Processing stopped: {str(err)[:200]}. Try again."
        return (
            "⚠️ Processing completed but no response was generated. "
            "This may be a transient error — try sending your message again.")

    # api_calls == 0, not failed/interrupted: agent never ran (post-/stop race); don't drop silently.
    if api_calls == 0 and not agent_result.get("partial"):
        return (
            "⚠️ Your message wasn't processed (the previous turn was still "
            "being cleaned up). Please send it again.")

    return response


def _is_gateway_hidden_reasoning_incomplete_turn(agent_result: dict) -> bool:
    """Detect retry-exhausted turns with hidden reasoning but no visible answer.

    The loop returns the retry-exhaustion sentinel as BOTH ``final_response`` and ``error``, so
    non-empty ``final_response`` proves nothing; any text other than the sentinel is a real answer.
    """
    if not isinstance(agent_result, dict):
        return False
    if agent_result.get("failed") or agent_result.get("interrupted"):
        return False
    if not agent_result.get("partial"):
        return False
    error_text = str(agent_result.get("error", "") or "").strip()
    if "remained incomplete after" not in error_text.lower():
        return False
    final_response = str(agent_result.get("final_response") or "").strip()
    return not final_response or final_response == error_text


def _should_clear_resume_pending_after_turn(agent_result: dict) -> bool:
    """True only when a gateway turn really completed successfully.

    ``resume_pending`` is a durable restart-recovery marker; a soft interrupt can look like a normal
    empty result, and clearing then loses the signal.
    """
    if not isinstance(agent_result, dict):
        return False
    if agent_result.get("interrupted"):
        return False
    if agent_result.get("failed") or agent_result.get("partial") or agent_result.get("error"):
        return False
    return agent_result.get("completed") is not False


def _preserve_queued_followup_history_offset(
    current_result: dict, followup_result: dict) -> dict:
    """Carry the outer history offset through queued follow-up drains.

    Each recursive ``_run_agent()`` advances ``history_offset``; uncorrected, the outer persistence
    step sees only the *last* queued turn as "new" and drops earlier ones.
    """
    if not isinstance(followup_result, dict) or not isinstance(current_result, dict):
        return followup_result

    current_offset = current_result.get("history_offset")
    followup_offset = followup_result.get("history_offset")
    if not isinstance(current_offset, int):
        return followup_result
    if isinstance(followup_offset, int) and followup_offset <= current_offset:
        return followup_result

    merged = dict(followup_result)
    merged["history_offset"] = current_offset
    return merged


async def _dispose_unused_adapter(adapter: "BasePlatformAdapter | None") -> None:
    """Best-effort dispose for an adapter that never made it onto ``self.adapters``.

    A failed connect leaves it uninstalled so nothing else calls ``disconnect()``; resources opened
    in ``__init__`` (e.g. SQLite fds) would leak until GC (not prompt for asyncio-bound objects) and
    exhaust the fd ulimit over a long retry loop. ``adapter`` may be ``None`` (half-constructed).
    """
    if adapter is None:
        return
    try:
        await adapter.disconnect()
    except Exception:
        # Half-constructed adapters can raise from disconnect(); must not abort the watcher loop.
        # CancelledError is a BaseException, so cancellation is not swallowed.
        logger.debug(
            "Adapter dispose raised on unowned adapter %r",
            getattr(adapter, "name", type(adapter).__name__), exc_info=True)


# Max seconds between platform reconnect retries (primary watcher and secondary profiles share it).
_RECONNECT_BACKOFF_CAP = 300

# Seconds continuously in the reconnect queue before NEEDS_ATTENTION. Retrying never stops (transient
# outages must self-heal); this only makes a permanently-failing loop loud. 0 disables.
_RECONNECT_ATTENTION_AFTER_SECONDS = _float_env("HERMES_RECONNECT_ATTENTION_AFTER_SECONDS", 7200)


def _reconnect_backoff(attempt: int) -> int:
    """Exponential reconnect backoff: 30s, 60s, 120s, ... capped at 5 min."""
    return min(30 * (2 ** (attempt - 1)), _RECONNECT_BACKOFF_CAP)


def _reconnect_needs_attention(info: dict, now: float) -> bool:
    """True when a reconnect-queue entry has waited long enough for NEEDS_ATTENTION.

    ``queued_at`` is re-stamped on each (re)entry, so only *continuous* failure escalates.
    """
    if _RECONNECT_ATTENTION_AFTER_SECONDS <= 0:
        return False  # escalation disabled
    queued_at = info.get("queued_at")
    if queued_at is None:
        info["queued_at"] = now
        return False
    return (now - queued_at) >= _RECONNECT_ATTENTION_AFTER_SECONDS


# "No session DB pinned": lets ``_session_db`` distinguish "resolve from profile scope" from a
# deliberate ``runner._session_db = None`` (disables DB commands). Mirrors gateway.session._DB_UNPINNED.
_SESSION_DB_UNPINNED = object()


# Agent-facing sidecar note per auto-reset reason (default: idle).
_AUTO_RESET_CONTEXT_NOTES = {
    "suspended": "[System note: The user's previous session was stopped and suspended. This is a fresh conversation with no prior context.]",
    "daily": "[System note: The user's session was automatically reset by the daily schedule. This is a fresh conversation with no prior context.]",
    "resume_pending_expired": "[System note: The previous gateway session could not be recovered after a restart (API recovery timed out). This is a fresh conversation — use /resume to restore history if needed.]",
    "idle": "[System note: The user's previous session expired due to inactivity. This is a fresh conversation with no prior context.]",
}


def _auto_reset_reason_text(reset_reason: str, policy) -> str:
    """Human-readable cause for the user-facing auto-reset notice."""
    if reset_reason == "suspended":
        return "previous session was stopped or interrupted"
    if reset_reason == "resume_pending_expired":
        return "gateway restart recovery timed out"
    if reset_reason == "daily":
        return f"daily schedule at {policy.at_hour}:00"
    hours = policy.idle_minutes // 60
    mins = policy.idle_minutes % 60
    duration = f"{hours}h" if not mins else f"{hours}h {mins}m" if hours else f"{mins}m"
    return f"inactive for {duration}"


def _write_runtime_status_quiet(**fields: Any) -> None:
    """Best-effort ``gateway_state.json`` write; status persistence must never abort the caller."""
    try:
        from gateway.status import write_runtime_status

        write_runtime_status(**fields)
    except Exception:
        pass


def _command_origin_for_source(source: Any) -> Optional[dict]:
    """Delivery origin for a shared CLI/gateway command so its job replies to this chat/thread."""
    try:
        platform = getattr(source.platform, "value", None) or str(getattr(source, "platform", "") or "")
        chat_id = getattr(source, "chat_id", None)
        if platform and chat_id:
            return {
                "platform": platform,
                "chat_id": str(chat_id),
                "chat_name": getattr(source, "chat_name", None),
                "thread_id": getattr(source, "thread_id", None)}
    except Exception:
        pass
    return None


def _builtin_adapter_import(module: str, adapter_name: str, requirement: str):
    """Lazy-import ``(adapter_cls, requirements_ok)`` from ``gateway.platforms.<module>``."""
    import importlib

    mod = importlib.import_module(f"gateway.platforms.{module}")
    return getattr(mod, adapter_name), getattr(mod, requirement)


# platform -> (module, adapter class, requirements probe, warning on probe failure).
_BUILTIN_ADAPTERS: dict[Platform, tuple[str, str, str, str]] = {
    Platform.WHATSAPP_CLOUD: ("whatsapp_cloud", "WhatsAppCloudAdapter", "check_whatsapp_cloud_requirements",
                              "WhatsApp Cloud: aiohttp/httpx missing — reinstall hermes-agent"),
    Platform.SIGNAL: ("signal", "SignalAdapter", "check_signal_requirements",
                      "Signal: runtime requirements not met"),
    Platform.WEIXIN: ("weixin", "WeixinAdapter", "check_weixin_requirements",
                      "Weixin: aiohttp/cryptography not installed"),
    Platform.API_SERVER: ("api_server", "APIServerAdapter", "check_api_server_requirements",
                          "API Server: aiohttp not installed"),
    Platform.WEBHOOK: ("webhook", "WebhookAdapter", "check_webhook_requirements",
                       "Webhook: aiohttp not installed"),
    Platform.MSGRAPH_WEBHOOK: ("msgraph_webhook", "MSGraphWebhookAdapter", "check_msgraph_webhook_requirements",
                               "MSGraph webhook: aiohttp not installed"),
    Platform.BLUEBUBBLES: ("bluebubbles", "BlueBubblesAdapter", "check_bluebubbles_requirements",
                           "BlueBubbles: aiohttp/httpx missing or BLUEBUBBLES_SERVER_URL/BLUEBUBBLES_PASSWORD not configured"),
    Platform.QQBOT: ("qqbot", "QQAdapter", "check_qq_requirements",
                     "QQBot: aiohttp/httpx missing or QQ_APP_ID/QQ_CLIENT_SECRET not configured"),
    Platform.YUANBAO: ("yuanbao", "YuanbaoAdapter", "WEBSOCKETS_AVAILABLE",
                       "Yuanbao: websockets not installed. Run: pip install websockets")}


def _instantiate_builtin_adapter(platform: Platform, config: Any) -> Optional[BasePlatformAdapter]:
    """Instantiate a core (non-plugin) adapter, or None when its requirements are unmet/unknown."""
    spec = _BUILTIN_ADAPTERS.get(platform)
    if spec is None:
        return None
    module, adapter_name, requirement, warning = spec
    adapter_cls, requirements_ok = _builtin_adapter_import(module, adapter_name, requirement)
    if not (requirements_ok() if callable(requirements_ok) else requirements_ok):
        logger.warning(warning)
        return None
    if platform == Platform.SIGNAL:
        from gateway.platforms.signal import validate_signal_config

        if not validate_signal_config(config):
            logger.warning("Signal: SIGNAL_HTTP_URL or SIGNAL_ACCOUNT not configured")
            return None
    return adapter_cls(config)


class GatewayRunner(
    GatewayAuthorizationMixin, GatewayKanbanWatchersMixin, GatewaySlashCommandsMixin,
    GatewayVoiceMixin, GatewayAdapterLifecycleMixin, GatewayTopicThreadsMixin, GatewayTurnMixin,
    GatewayShutdownMixin, GatewayBusySessionMixin, GatewayConfigLoadersMixin, GatewayStartupMixin,
    GatewaySessionWatchersMixin, GatewayNotificationsMixin, GatewayInboundMixin, GatewayGoalsMixin,
    GatewayAgentCacheMixin):
    """Main gateway controller: manages adapter lifecycles, routes messages to/from the agent."""

    # Class-level defaults so partial construction in tests doesn't
    # blow up on attribute access.
    _busy_input_mode: str = "interrupt"
    _busy_text_mode: str = "interrupt"
    _restart_drain_timeout: float = DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT
    _restart_after_turn_timeout: float = DEFAULT_GATEWAY_RESTART_AFTER_TURN_TIMEOUT
    _cron_drain_timeout: float = DEFAULT_GATEWAY_CRON_DRAIN_TIMEOUT
    _signal_interrupt_grace_timeout: float = DEFAULT_GATEWAY_SIGNAL_INTERRUPT_GRACE_TIMEOUT
    _exit_code: Optional[int] = None
    _draining: bool = False
    _external_drain_active: bool = False
    _restart_requested: bool = False
    _restart_task_started: bool = False
    _restart_detached: bool = False
    _restart_via_service: bool = False
    _detached_restart_helper_started: bool = False
    _restart_command_source: Optional[SessionSource] = None
    _stop_task: Optional[asyncio.Task] = None
    _restart_task: Optional[asyncio.Task] = None
    _profile_failed_platforms: Optional[Dict[str, Dict[Platform, asyncio.Task]]] = None
    _systemd_watchdog: Optional[Any] = None
    _startup_restore_in_progress: bool = False
    _startup_warmup_task: Optional[asyncio.Task] = None

    # Legacy per-session dict adapters: all per-session state lives in ``self._sessions``; these expose
    # the old dict attrs as LIVE MutableMapping views. New code: ``self._session_state(key)``.
    _running_agents = legacy_dict_property("_running_agents")
    _running_agents_ts = legacy_dict_property("_running_agents_ts")
    _active_session_leases = legacy_dict_property("_active_session_leases")
    _busy_ack_ts = legacy_dict_property("_busy_ack_ts")
    _turn_lease_tokens = legacy_lease_token_property()
    _session_run_generation = legacy_dict_property("_session_run_generation")
    _session_model_overrides = legacy_dict_property("_session_model_overrides")
    _pending_one_turn_model_restores = legacy_dict_property(
        "_pending_one_turn_model_restores")
    _session_reasoning_overrides = legacy_dict_property("_session_reasoning_overrides")
    _session_service_tier_overrides = legacy_dict_property(
        "_session_service_tier_overrides")
    _last_resolved_model = legacy_dict_property("_last_resolved_model")
    _queued_events = legacy_dict_property("_queued_events")
    _pending_turn_sidecar_notes = legacy_dict_property("_pending_turn_sidecar_notes")
    _pending_messages = legacy_dict_property("_pending_messages")
    _pending_native_image_paths_by_session = legacy_dict_property(
        "_pending_native_image_paths_by_session")
    _session_ephemeral_pin = legacy_dict_property("_session_ephemeral_pin")
    _session_vc_last = legacy_dict_property("_session_vc_last")
    _pending_approvals = legacy_dict_property("_pending_approvals")
    _update_prompt_pending = legacy_dict_property("_update_prompt_pending")

    def _sessions_map(self) -> Dict[str, "SessionState"]:
        """Per-session state map; lazily created so bare ``object.__new__`` test runners work."""
        sessions = self.__dict__.get("_sessions")
        if sessions is None:
            sessions = {}
            self.__dict__["_sessions"] = sessions
        return sessions

    def _session_state(self, session_key: str) -> "SessionState":
        """Get-or-create the :class:`SessionState` for ``session_key``."""
        sessions = self._sessions_map()
        state = sessions.get(session_key)
        if state is None:
            state = SessionState()
            sessions[session_key] = state
        return state

    def _peek_session_state(self, session_key: str) -> Optional["SessionState"]:
        """Return the SessionState for ``session_key`` without creating one."""
        sessions = self.__dict__.get("_sessions")
        if not sessions:
            return None
        return sessions.get(session_key)

    def _is_session_running(self, session_key: str) -> bool:
        """True when the session holds a running-turn slot (agent or sentinel)."""
        state = self._peek_session_state(session_key)
        return state is not None and state.turn.agent is not None

    def _running_agent_items(self) -> List[tuple]:
        """(session_key, agent) pairs for sessions with a running turn (incl. pending sentinels),
        matching the old ``_running_agents`` dict contents."""
        return [
            (key, state.turn.agent)
            for key, state in self._sessions_map().items()
            if state.turn.agent is not None]
    # Loop-liveness / watchdog handles; class-level defaults so partially constructed test runners work.
    _loop_heartbeat_task: Optional["asyncio.Task"] = None
    _loop_floor_timer_handle: Optional[Any] = None
    _loop_liveness_watchdog: Optional[Any] = None
    _gateway_started_at: float = 0.0
    _shutdown_watchdog_done: Optional["threading.Event"] = None
    _platform_lock_takeover_on_start: bool = False
    _reconnect_watcher_task: Optional["asyncio.Task"] = None

    def __init__(self, config: Optional[GatewayConfig] = None):
        global _gateway_runner_ref
        # With multiplex_profiles on, load under the default profile secret scope so bot tokens in its
        # .env resolve as secondary profiles' do; explicit config= injection (tests) is left untouched.
        self.config = config if config is not None else load_gateway_config_for_runner()
        # Multiplexer flag flips agent.secret_scope.get_secret() to fail-closed on unscoped credential
        # reads, so a missed migration crashes loudly instead of leaking a cross-profile value.
        try:
            from agent.secret_scope import set_multiplex_active
            set_multiplex_active(bool(getattr(self.config, "multiplex_profiles", False)))
        except Exception:
            logger.debug("could not set multiplex-active flag", exc_info=True)
        self.adapters: Dict[Platform, BasePlatformAdapter] = {}
        # Non-None means SessionDB init failed — the gateway broadcasts a one-time warning to the home
        # channel(s) after connecting so the user learns persistence is broken before /resume fails.
        self._session_db_init_error: Optional[str] = None
        # Adapters for NON-default profiles, keyed by profile name then Platform; self.adapters stays
        # the default profile's map so existing sites are untouched when multiplexing is off (empty).
        self._profile_adapters: Dict[str, Dict[Platform, BasePlatformAdapter]] = {}
        self._warn_if_docker_media_delivery_is_risky()
        _gateway_runner_ref = _weakref.ref(self)

        self._init_runtime_settings()
        self._init_session_store()
        self._init_lifecycle_state()
        self._init_runtime_caches()
        self._init_startup_checks()
        self._init_session_db()
        self._init_registries_and_clocks()

    def _init_runtime_settings(self) -> None:
        """Load ephemeral per-call config (prefill, reasoning, busy modes, timeouts, routing)."""
        # Ephemeral config (config.yaml / env): injected at API-call time only, never persisted.
        self._prefill_messages = self._load_prefill_messages()
        self._reasoning_config = self._load_reasoning_config()
        self._service_tier = self._load_service_tier()
        self._show_reasoning = self._load_show_reasoning()
        self._busy_input_mode = self._load_busy_input_mode()
        self._busy_text_mode = self._load_busy_text_mode()
        # Secondary-profile busy modes, snapshotted at multiplex startup; busy-message handlers consult
        # them by routed source without rereading config or mutating process-global environment.
        self._busy_input_modes_by_profile: Dict[str, str] = {}
        self._busy_text_modes_by_profile: Dict[str, str] = {}
        self._restart_drain_timeout = self._load_restart_drain_timeout()
        self._restart_after_turn_timeout = self._load_restart_after_turn_timeout()
        self._cron_drain_timeout = self._load_cron_drain_timeout()
        self._signal_interrupt_grace_timeout = self._load_signal_interrupt_grace_timeout()
        self._provider_routing = self._load_provider_routing()
        self._fallback_model = self._load_fallback_model()

    def _init_session_store(self) -> None:
        """Build the SessionStore (with process-registry reset guard), its async facade and the router."""
        # Reset guard: a background process older than session_reset.bg_process_max_age_hours (24h
        # default) is stale and no longer blocks idle/daily reset (NOT killed, only ignored).
        from tools.process_registry import process_registry
        _bg_max_age_hours = getattr(
            self.config.default_reset_policy, "bg_process_max_age_hours", 24)
        _bg_max_age_seconds = (
            _bg_max_age_hours * 3600 if _bg_max_age_hours and _bg_max_age_hours > 0 else None)
        self.session_store = SessionStore(
            self.config.sessions_dir, self.config,
            has_active_processes_fn=lambda key: process_registry.has_active_for_session(
                key, max_active_age=_bg_max_age_seconds))
        # Loop-side boundary: sync helpers use ``session_store`` directly; async handlers await this facade.
        self._async_session_store = AsyncSessionStore(self.session_store)
        self.delivery_router = DeliveryRouter(self.config)

    def _init_lifecycle_state(self) -> None:
        """Initialise run/exit/restart flags, per-session state, and completion-delivery bookkeeping."""
        self._running = self._exit_cleanly = self._exit_with_failure = self._draining = False
        self._gateway_loop: Optional[asyncio.AbstractEventLoop] = None
        self._shutdown_event = asyncio.Event()
        self._exit_reason: Optional[str] = None
        self._exit_code: Optional[int] = None
        self._profile_failed_platforms: Dict[str, Dict[Platform, asyncio.Task]] = {}
        self._systemd_watchdog = None
        # External (NAS-driven) drain, distinct from one-way ``_draining``: set while ``.drain_request.json``
        # exists — NEW turns refused, process stays up, removing the marker reverts to ``running``.
        self._external_drain_active = False
        # ``_signal_initiated_shutdown``: SIGTERM/SIGINT with no planned-stop/takeover marker (container,
        # OOM, bare kill); _stop_impl must NOT persist gateway_state=stopped or container_boot won't restart.
        self._restart_requested = self._signal_initiated_shutdown = self._restart_task_started = False
        self._restart_detached = self._restart_via_service = self._detached_restart_helper_started = False
        self._restart_command_source: Optional[SessionSource] = None
        # Construction clock: bounds the /restart redelivery guard's window (missing dedup marker = stale).
        self._startup_time: float = time.time()
        # True when booted from a chat /restart (.restart_notify.json existed). One-shot signal so the
        # marker-missing fallback suppresses a /restart only when we KNOW we just restarted.
        self._booted_from_restart: bool = False
        self._stop_task: Optional[asyncio.Task] = None
        self._restart_task: Optional[asyncio.Task] = None
        self._executor_lock = threading.Lock()
        self._executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
        # Set on gateway stop so the recreate-on-shutdown path can't resurrect the pool.
        self._executor_closing = False
        # ALL per-session state lives here (gateway/session_state.py); use _session_state / _peek_session_state.
        self._sessions: Dict[str, SessionState] = {}
        # Per-SESSION_ID turn lease: serializes [load history → run → flush] when two ROUTING KEYS resolve
        # to one session_id (switch_session's many-to-one mapping), which routing-key guards cannot see.
        self._turn_leases = SessionTurnLeaseRegistry()
        # Stall-notified keys clear when pending clears / activity resumes / conversation boundary.
        self._session_stall_notified: Dict[str, bool] = {}
        # Startup restore gate: while restart-interrupted sessions auto-resume, real inbound messages
        # queue instead of competing with the synthetic resume turns; drained after all resume tasks end.
        self._startup_restore_in_progress = False
        self._startup_restore_queue: List[MessageEvent] = []
        self._startup_restore_tasks: List[asyncio.Task] = []
        # Set by start_gateway() only for an explicit ``--replace`` launch; scoped to each adapter's
        # cold-start connect and removed before any reconnect can run.
        self._platform_lock_takeover_on_start = False
        # Capped LRU of live SessionSources for fallback routing (shutdown notices, synthetic events) when
        # the persisted origin is missing and _parse_session_key can't recover thread_id.
        self._session_sources: "OrderedDict[str, SessionSource]" = OrderedDict()
        self._session_sources_max = 512
        # Lifecycle-scoped completion dedup: closes queue/watcher races inside one gateway without claiming
        # exactly-once across a crash; durable replay state stays owned by tools.async_delegation.
        self._completion_delivery_lock = threading.Lock()
        self._completion_deliveries_inflight: set[tuple[str, str, object]] = set()
        self._completion_deliveries_delivered: "OrderedDict[tuple[str, str, object], None]" = OrderedDict()
        self._completion_delivery_retention = 2048
        # Agent-triggered terminal completions from one conversation often land in the same scheduler
        # tick; hold them briefly so the agent gets one synthetic turn instead of one per process.
        self._completion_notification_batches: dict[tuple[str, ...], list[tuple[str, dict, asyncio.Future]]] = {}
        self._completion_notification_batch_tasks: dict[tuple[str, ...], asyncio.Task] = {}
        self._completion_notification_batch_flush_tasks: set[asyncio.Task] = set()
        self._completion_notification_batch_window = 0.1
        self._completion_notification_batches_stopping = False

    def _init_runtime_caches(self) -> None:
        """Agent cache, profile identity, Teams runtime, failed-platform tracking, slash-confirm counter."""
        # AIAgent per session preserves prompt caching (fresh agent per message breaks the prefix cache,
        # ~10x cost on Anthropic). Value: (AIAgent, config_signature). LRU cap in _enforce_agent_cache_cap,
        # idle TTL from _session_expiry_watcher.
        self._agent_cache: "OrderedDict[str, tuple]" = OrderedDict()
        self._agent_cache_lock = threading.Lock()
        # Launch-time identity of the profile that owns ``self.adapters``; ``_authorization_adapter``
        # compares against this rather than the per-turn ``_active_profile_name()``.
        self._primary_profile_name = self._kanban_notifier_profile = self._active_profile_name()
        # Teams meeting pipeline runtime (bound later when msgraph_webhook adapter exists).
        self._teams_pipeline_runtime = None
        self._teams_pipeline_runtime_error: Optional[str] = None
        # Failed-to-connect platforms for background reconnection: Platform -> {config, attempts, next_retry}
        self._failed_platforms: Dict[Platform, Dict[str, Any]] = {}
        # Strong refs to detached fatal-error handler tasks so the loop can't GC them mid-run.
        self._fatal_handler_tasks: set = set()
        # Slash-confirm state lives in tools.slash_confirm (module-level) so adapters resolve callbacks
        # without a runner backref; local counter keeps confirm_ids compact (64-byte callback_data caps).
        import itertools
        self._slash_confirm_counter = itertools.count(1)

    def _init_startup_checks(self) -> None:
        """Ensure tirith is installed and warn when manual approvals have no automated assessor."""
        def _ensure_tirith() -> None:
            # Downloads if needed; non-fatal — fail-open at scan time if unavailable.
            from tools.tirith_security import ensure_installed
            ensure_installed(log_failures=False)

        _best_effort(_ensure_tirith)

        # Manual approvals with no automated assessor (tirith off AND no auxiliary.approval) fail closed
        # on unattended gateways — surface it so operators knowingly enable one.
        try:
            from hermes_cli.config import load_config as _load_full_config
            _appr_cfg = _load_full_config()
            _appr_mode = str(
                cfg_get(_appr_cfg, "approvals", "mode", default="manual") or "manual"
            ).strip().lower()
            _tirith_on = bool(cfg_get(_appr_cfg, "security", "tirith_enabled", default=True))
            _aux_approval = cfg_get(_appr_cfg, "auxiliary", "approval", default=None)
            if _appr_mode == "manual" and not _tirith_on and not _aux_approval:
                logger.warning(
                    "Gateway approvals.mode=manual with no automated risk "
                    "assessor (security.tirith_enabled is false and "
                    "auxiliary.approval is unset): dangerous commands and "
                    "execute_code scripts will BLOCK until a human approves "
                    "them in chat. Enable security.tirith_enabled or configure "
                    "auxiliary.approval for unattended operation.")
        except Exception:
            logger.debug("approvals.mode startup check skipped", exc_info=True)

    def _init_session_db(self) -> None:
        """Open the session DB for the active scope and run opportunistic state.db / checkpoint maintenance."""
        # Session DB is a property caching one AsyncSessionDB per path (a handle bound here would pin the
        # root home under multiplex); priming here keeps startup diagnostics at init.
        self._session_db_pinned: Any = _SESSION_DB_UNPINNED
        self._session_db_handles: Dict[Path, Any] = {}
        self._session_db_handles_lock = threading.Lock()
        from gateway.session_db_recovery import RecoverableHandleCache

        self._session_db_handle_cache = RecoverableHandleCache(
            handles=self._session_db_handles, lock=self._session_db_handles_lock)
        try:
            self._open_session_db_for_active_scope(raise_on_error=True)
        except Exception as e:
            # WARNING (not DEBUG) so it lands in errors.log; else an NFS HERMES_HOME silently loses /resume etc.
            logger.warning("SQLite session store not available: %s", e)
            # Surfaced on the home channel(s) once connected; otherwise persistence degrades silently.
            self._session_db_init_error = str(e)

        # Opportunistic state.db maintenance (prune + optional VACUUM), at most once per min_interval_hours.
        # A few blocking seconds per day is fine for a long-lived gateway; failures log, never raise.
        if self._session_db is not None:
            try:
                from hermes_cli.config import load_config as _load_full_config
                _sess_cfg = (_load_full_config().get("sessions") or {})
                if _sess_cfg.get("auto_archive", False):
                    self._session_db._db.maybe_auto_archive(
                        idle_days=float(_sess_cfg.get("auto_archive_days", 3)),
                        min_interval_hours=int(_sess_cfg.get("min_interval_hours", 24)))
                if _sess_cfg.get("auto_prune", False):
                    # Construction-time, before the loop serves traffic; sync DB is fine.
                    self._session_db._db.maybe_auto_prune_and_vacuum(
                        retention_days=int(_sess_cfg.get("retention_days", 90)),
                        min_interval_hours=int(_sess_cfg.get("min_interval_hours", 24)),
                        min_vacuum_interval_days=int(
                            _sess_cfg.get("min_vacuum_interval_days", 30)),
                        vacuum=bool(_sess_cfg.get("vacuum_after_prune", True)),
                        sessions_dir=self.config.sessions_dir)
            except Exception as exc:
                logger.debug("state.db auto-maintenance skipped: %s", exc)

        # Stale checkpoint repo cleanup; opt-in via checkpoints.auto_prune, idempotent via .last_prune.
        try:
            from hermes_cli.config import load_config as _load_full_config
            _ckpt_cfg = (_load_full_config().get("checkpoints") or {})
            if _ckpt_cfg.get("auto_prune", False):
                from tools.checkpoint_manager import maybe_auto_prune_checkpoints
                # delete_orphans never honoured unattended: a missing workdir is ambiguous (deleted vs.
                # unmounted share); orphan cleanup is only via explicit `hermes checkpoints prune`.
                maybe_auto_prune_checkpoints(
                    retention_days=int(_ckpt_cfg.get("retention_days", 7)),
                    min_interval_hours=int(_ckpt_cfg.get("min_interval_hours", 24)),
                    delete_orphans=False,
                    max_total_size_mb=int(_ckpt_cfg.get("max_total_size_mb", 500)))
        except Exception as exc:
            logger.debug("checkpoint auto-maintenance skipped: %s", exc)

    def _init_registries_and_clocks(self) -> None:
        """Pairing stores, hook registry, voice modes, background-task set, liveness and idle clocks."""
        # ``pairing_store``: global/default store (CLI, callers without profile context); ``pairing_stores``:
        # per-profile map ``authz_mixin._is_user_authorized`` routes through (one whitelist per profile).
        from gateway.pairing import PairingStore
        from gateway.hooks import HookRegistry
        self.pairing_store = PairingStore()
        self.pairing_stores: Dict[str, "PairingStore"] = {}
        self.hooks = HookRegistry()
        # Per-chat voice reply mode: "off" | "voice_only" | "all"
        self._voice_mode: Dict[str, str] = self._load_voice_modes()
        # Per-(guild,user) transcript dedup: the voice/STT pipeline can emit one utterance twice.
        self._recent_voice_transcripts: Dict[tuple[int, int], List[tuple[float, str]]] = {}
        # Background tasks kept referenced so they are not garbage-collected mid-execution.
        self._background_tasks: set = set()
        # Event-loop liveness heartbeat: rewritten every 30s while the loop dispatches; supervisors use
        # the file mtime / updated_at to tell "process alive" from "loop frozen".
        self._gateway_started_at: float = time.time()
        self._loop_heartbeat_task: Optional[asyncio.Task] = None
        self._loop_floor_timer_handle = self._loop_liveness_watchdog = None
        # scale-to-zero: gateway-scoped "last inbound seen" clock, stamped in _handle_message (the single
        # inbound chokepoint) and seeded to "now" so a fresh gateway isn't idle from epoch.
        self._last_inbound_at: float = time.time()
        # Re-arm cooldown after a wake so we don't go dormant again before the drained backlog updates
        # the clock; and a one-shot latch so the "platform owns the suspend" notice logs once.
        self._scale_to_zero_cooldown_until: float = 0.0
        self._scale_to_zero_no_suspend_logged: bool = False

    def _open_session_db_for_active_scope(self, raise_on_error: bool = False) -> Any:
        """Return the AsyncSessionDB for the profile scope active on this task.

        Resolved per access (not in ``__init__``) because ``SessionDB()`` reads the context-local
        HERMES_HOME, letting a multiplexed profile use its own store; one handle cached per path.
        Construction failure enters bounded backoff; ``raise_on_error=True`` (priming) propagates it.
        """
        from hermes_state import AsyncSessionDB, _default_db_path, get_shared_session_db
        from gateway.session_db_recovery import RecoverableHandleCache

        path = Path(_default_db_path())
        cache = getattr(self, "_session_db_handle_cache", None)
        if cache is None:
            # Test runners built with object.__new__ skip __init__.
            cache = RecoverableHandleCache(
                handles=self._session_db_handles, lock=self._session_db_handles_lock)
            self._session_db_handle_cache = cache

        def _open():
            # Borrow the SessionStore's handle (same path) rather than opening a second one — otherwise two
            # writer connections + two read pools per state.db. The store owns/sweeps it at shutdown; this
            # cache holds only the async wrapper and cannot go stale (store drops handles only in close_all).
            store = getattr(self, "session_store", None)
            borrowed = getattr(store, "_db", None) if store is not None else None
            if borrowed is not None:
                wrapper = AsyncSessionDB(borrowed)
                # close_all_session_db_handles() must not close what the store owns (its sweep runs first).
                wrapper.__dict__["_hermes_borrowed_handle"] = True
                return wrapper
            if store is not None:
                # Store handle unavailable: opening our own would resurrect the duplicate borrowed away.
                raise RuntimeError("SessionStore SQLite handle unavailable")
            try:
                return AsyncSessionDB(get_shared_session_db())
            except Exception as exc:
                logger.warning("SQLite session store not available: %s", exc)
                raise

        def _recovered() -> None:
            self._session_db_init_error = None
            logger.info("SQLite session store recovered")

        return cache.get(
            path, _open, raise_on_error=raise_on_error, on_recovered=_recovered)

    @property
    def _session_db(self) -> Any:
        """The AsyncSessionDB for the active profile scope, or a pinned override (assigning
        ``runner._session_db`` pins it for every later read — tests install fakes/None this way)."""
        if self._session_db_pinned is not _SESSION_DB_UNPINNED:
            return self._session_db_pinned
        return self._open_session_db_for_active_scope()

    @_session_db.setter
    def _session_db(self, value) -> None:
        self._session_db_pinned = value

    def close_all_session_db_handles(self) -> None:
        """Close every per-profile AsyncSessionDB this runner opened.

        Drained under the lock, closed outside it; a pinned handle is the pinner's to close. Wrappers
        BORROWED from ``session_store`` are not closed: the store's sweep (runs first) closes them.
        """
        def _close(db) -> None:
            if getattr(db, "__dict__", {}).get("_hermes_borrowed_handle"):
                return
            inner = getattr(db, "_db", db)
            if inner is None or not hasattr(inner, "close"):
                return
            from hermes_state import release_or_close
            try:
                release_or_close(inner)
            except Exception as exc:
                logger.debug("SessionDB close error during handle sweep: %s", exc)

        self._session_db_handle_cache.close_all(_close)

    def _wire_teams_pipeline_runtime(self) -> None:
        """Bind the Teams meeting pipeline runtime to Graph webhook ingress (no-op if adapter/plugin off)."""
        if Platform.MSGRAPH_WEBHOOK not in self.adapters:
            return
        if not _teams_pipeline_plugin_enabled():
            logger.debug("Teams pipeline plugin is disabled; skipping runtime wiring")
            return
        try:
            from plugins.teams_pipeline.runtime import bind_gateway_runtime
        except Exception as exc:
            logger.warning("Teams pipeline runtime import failed: %s", exc)
            return
        try:
            bound = bind_gateway_runtime(self)
        except Exception as exc:
            logger.warning("Teams pipeline runtime wiring failed: %s", exc)
            return
        if bound:
            logger.info("Teams pipeline runtime bound to msgraph webhook ingress")
        elif self._teams_pipeline_runtime_error:
            logger.warning(
                "Teams pipeline runtime unavailable: %s", self._teams_pipeline_runtime_error)

    def _warn_if_docker_media_delivery_is_risky(self) -> None:
        """Warn when Docker-backed gateways lack an explicit export mount: MEDIA delivery runs in the
        gateway process, so model-emitted paths like `/output/report.txt` must be host-readable."""
        if os.getenv("TERMINAL_ENV", "").strip().lower() != "docker":
            return

        connected = self.config.get_connected_platforms()
        messaging_platforms = [p for p in connected if p not in {Platform.LOCAL, Platform.API_SERVER, Platform.WEBHOOK}]
        if not messaging_platforms:
            return

        raw_volumes = os.getenv("TERMINAL_DOCKER_VOLUMES", "").strip()
        volumes: List[str] = []
        if raw_volumes:
            try:
                parsed = json.loads(raw_volumes)
                if isinstance(parsed, list):
                    volumes = [str(v) for v in parsed if isinstance(v, str)]
            except Exception:
                logger.debug("Could not parse TERMINAL_DOCKER_VOLUMES for gateway media warning", exc_info=True)

        has_explicit_output_mount = False
        for spec in volumes:
            match = _DOCKER_VOLUME_SPEC_RE.match(spec)
            if not match:
                continue
            container_path = match.group("container")
            if container_path in _DOCKER_MEDIA_OUTPUT_CONTAINER_PATHS:
                has_explicit_output_mount = True
                break

        if has_explicit_output_mount:
            return

        logger.warning(
            "Docker backend is enabled for the messaging gateway but no explicit host-visible "
            "output mount (for example '/home/user/.hermes/cache/documents:/output') is configured. "
            "This is fine if the model already emits host-visible paths, but MEDIA file delivery can fail "
            "for container-local paths like '/workspace/...' or '/output/...'.")

    _VOICE_MODE_PATH = _hermes_home / "gateway_voice_mode.json"

    should_exit_cleanly = property(lambda self: self._exit_cleanly)
    should_exit_with_failure = property(lambda self: self._exit_with_failure)
    exit_reason = property(lambda self: self._exit_reason)
    exit_code = property(lambda self: self._exit_code)

    def _session_key_for_source(self, source: SessionSource) -> str:
        """Resolve the current session key for a source, honoring gateway config when available."""
        if hasattr(self, "session_store") and self.session_store is not None:
            try:
                session_key = self.session_store._generate_session_key(source)
                if isinstance(session_key, str) and session_key:
                    return session_key
            except Exception:
                pass
        config = getattr(self, "config", None)
        # Mirror SessionStore._resolve_profile_for_key so this fallback yields the primary path's
        # namespace: None (legacy agent:main) unless multiplexing is on, then the active profile.
        _profile = None
        if getattr(config, "multiplex_profiles", False):
            if source.profile:
                _profile = source.profile
            else:
                try:
                    from hermes_cli.profiles import get_active_profile_name
                    _profile = get_active_profile_name() or "default"
                except Exception:
                    _profile = None
        return build_session_key(
            source, group_sessions_per_user=getattr(config, "group_sessions_per_user", True),
            thread_sessions_per_user=getattr(config, "thread_sessions_per_user", False),
            profile=_profile)

    # Telegram General topic in forum-enabled private chats: clients omit message_thread_id or send "1"; both = root.
    _TELEGRAM_GENERAL_TOPIC_IDS = frozenset({"", "1"})

    _TELEGRAM_LOBBY_REMINDER_COOLDOWN_S = 30.0

    def _normalize_source_for_session_key(
        self, source: SessionSource) -> SessionSource:
        """Apply Telegram DM topic recovery to a source for session-key purposes.

        ``_handle_message_with_agent`` rewrites ``thread_id`` before deriving the session key, so
        handlers keying off the raw ``event.source`` would store overrides under a key the next turn
        never reads. Always derive override storage keys from the result so storage and read match.
        """
        try:
            recovered = self._recover_telegram_topic_thread_id(source)
        except Exception:
            return source
        if recovered is None:
            return source
        return dataclasses.replace(source, thread_id=recovered)

    def _resolve_session_key_or_none(self, source, session_key: Optional[str]) -> Optional[str]:
        """``session_key`` if given, else the key for ``source`` (None when it cannot be derived)."""
        if session_key or source is None:
            return session_key
        try:
            return self._session_key_for_source(source)
        except Exception:
            return None

    def _running_agent_count(self) -> int:
        return len(self._running_agents)

    def _status_action_label(self) -> str:
        return "restart" if self._restart_requested else "shutdown"

    def _status_action_gerund(self) -> str:
        return "restarting" if self._restart_requested else "shutting down"

    def _update_runtime_status(self, gateway_state: Optional[str] = None, exit_reason: Optional[str] = None) -> None:
        _write_runtime_status_quiet(
            gateway_state=gateway_state, exit_reason=exit_reason,
            restart_requested=self._restart_requested, active_agents=self._active_work_count())

    def _persist_active_agents(self) -> None:
        """Persist the live in-flight agent count to ``gateway_state.json`` at every turn boundary.

        Passes ONLY ``active_agents`` so the read-merge-write preserves lifecycle state (``gateway_state=None``
        would clobber it). Best-effort: a failed write must never disrupt a turn.
        """
        _write_runtime_status_quiet(active_agents=self._active_work_count())

    def _running_agent_ids(self) -> set:
        """``id()`` of every agent mid-turn — identity-keyed so the lookup is O(1) and independent of
        ``AIAgent.__eq__`` (MagicMock overrides it in tests)."""
        return {
            id(a)
            for _, a in self._running_agent_items()
            if a is not None and a is not _AGENT_PENDING_SENTINEL}

    def _snapshot_running_agents(self) -> Dict[str, Any]:
        return {k: a for k, a in self._running_agent_items() if a is not _AGENT_PENDING_SENTINEL}

    # ---- Tunables consumed by the run_* mixins (kept on the class: tests and plugins patch them) ----

    # Per-session pending follow-up cap for busy_input_mode=queue (and paths sharing that entry point):
    # a stuck agent + rapid-fire user must not grow the overflow list unboundedly.
    _BUSY_QUEUE_MAX_PENDING = 32

    @dataclasses.dataclass
    class _BusySteerOutcome:
        effective_mode: str
        demoted_for_subagents: bool
        demoted_for_compression: bool
        steered: bool
        redirected: bool

    # Bound for off-loop agent-resource cleanup: _cleanup_agent_resources is synchronous and can block
    # long (subprocess teardown, memory-provider IO); inline it wedges the loop, so it runs in a worker.
    _CLEANUP_TIMEOUT_S = 30.0

    # Budget for one finalize_session() dispatch (plugin on_session_finalize hooks + Relay close):
    # enough for a normal trace-export flush, small enough a wedged plugin can't eat the stop window.
    _FINALIZE_TIMEOUT_S = 10.0

    _STUCK_LOOP_THRESHOLD = 3  # restarts while active before auto-suspend
    _STUCK_LOOP_FILE = ".restart_failure_counts"

    # Reasons set by _stop_impl() on force-interrupt; "restart_interrupted" by suspend_recently_active()
    # on crash recovery (no .clean_shutdown marker). All mean "killed mid-turn" -> startup auto-resume.
    _AUTO_RESUME_REASONS = frozenset({"restart_timeout", "shutdown_timeout", "restart_interrupted"})

    _MAX_SUPERVISED_RESTARTS = 5
    # A task that ran at least this long before crashing is HEALTHY: an isolated crash, not a
    # crash-loop; the consecutive-restart counter resets so a long-lived daemon isn't abandoned.
    _SUPERVISED_HEALTHY_SECS = 300
    # Slow respawn tier after the reconnect watcher exhausts its restart budget. Long on purpose: the
    # watcher is crashing on contact, so "check back later" beats a tight loop.
    _RECONNECT_WATCHER_SLOW_RETRY_SECS = 300
    # Slow-tier respawns to attempt while work is still queued. Bounded: if half an hour of
    # five-minute retries cannot keep a watcher alive, the fault is not transient — fail loudly.
    _MAX_SLOW_WATCHER_RESPAWNS = 6
    _TELEGRAM_CAPABILITY_HINT_COOLDOWN_S = 300.0
    _APPROVAL_TIMEOUT_SECONDS = 300  # 5 minutes
    _MAX_INTERRUPT_DEPTH = 3  # Cap recursive interrupt handling
    # Command-specific mid-run reject texts (busy_policy == "reject" with a busy_handler naming an
    # entry here); all other rejected commands get the generic text in _dispatch_busy_slash_command.
    _BUSY_REJECT_TEXT: Dict[str, str] = {
        "model": "Agent is running — wait or /stop first, then switch models.",
        "codex-runtime": "Agent is running — wait or /stop first, then change runtime.",
        "moa": "Agent is running — wait or /stop first, then run /moa."}

    def _active_profile_name(self) -> str:
        """Return the profile name this gateway represents."""
        try:
            from hermes_cli.profiles import get_active_profile_name
            return get_active_profile_name() or "default"
        except Exception:
            return "default"

    def _is_user_authorized_for_source(
        self, source: SessionSource, *, allow_adapter_delegation: bool = True) -> bool:
        """Authorize under the live transport's profile, not the routed runtime (which need not copy the
        shared bot token/allowlist); the transport home is stamped on the source for this read only."""
        def _check() -> bool:
            # Keep the one-argument seam used by plugins/tests; pass the keyword only when disabling.
            if allow_adapter_delegation:
                return self._is_user_authorized(source)
            return self._is_user_authorized(source, allow_adapter_delegation=False)

        authorization_home = getattr(source, "_authorization_profile_home", None)
        if authorization_home is not None:
            with _profile_runtime_scope(Path(authorization_home)):
                return _check()
        return _check()

    def _cache_session_source(self, session_key: str, source) -> None:
        if not session_key or source is None:
            return
        cached_sources = getattr(self, "_session_sources", None)
        if cached_sources is None:
            cached_sources = OrderedDict()
            self._session_sources = cached_sources
        try:
            cached_sources[session_key] = dataclasses.replace(source)
        except Exception:
            logger.debug("Failed to cache live session source for %s", session_key, exc_info=True)
            return
        try:
            cached_sources.move_to_end(session_key)
            max_size = getattr(self, "_session_sources_max", 512)
            while len(cached_sources) > max_size:
                cached_sources.popitem(last=False)
        except Exception:
            pass

    @property
    def async_session_store(self) -> AsyncSessionStore:
        """Return the single async facade for this runner's SessionStore."""
        facade = getattr(self, "_async_session_store", None)
        if facade is None or facade._store is not self.session_store:
            facade = AsyncSessionStore(self.session_store)
            self._async_session_store = facade
        return facade

    def _get_cached_session_source(self, session_key: str):
        if not session_key:
            return None
        cached_sources = getattr(self, "_session_sources", None)
        if not cached_sources:
            return None
        source = cached_sources.get(session_key)
        if source is not None:
            with suppress(Exception):
                cached_sources.move_to_end(session_key)
        return source

    @dataclasses.dataclass
    class _HygieneSettings:
        """Resolved session-hygiene configuration for one inbound turn."""

        model: str
        threshold_pct: float
        compression_enabled: bool
        hard_msg_limit: int
        timeout_seconds: float
        total_ceiling_seconds: float
        max_turn_hold_seconds: float
        failure_cooldown_seconds: float
        config_context_length: Optional[int]
        provider: Optional[str]
        base_url: Optional[str]
        api_key: Optional[str]
        data: Any

    @dataclasses.dataclass
    class _HygieneAttempt:
        """One detached hygiene compression attempt. ``cleanup_deferred`` is shared mutable state: wait
        handlers set it on raise paths; the owning ``finally`` reads it to decide on cleanup now."""

        agent: Any
        meta: Any
        commit_fence: Any = None
        future: Any = None
        wait_started: float = 0.0
        cleanup_deferred: bool = False
        history: Any = None

    def _thread_metadata_for_source(
        self, source, reply_to_message_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Build the metadata dict platforms need for thread-aware replies."""
        metadata = self._thread_metadata_for_target(
            getattr(source, "platform", None), getattr(source, "chat_id", None),
            getattr(source, "thread_id", None), chat_type=getattr(source, "chat_type", None),
            reply_to_message_id=reply_to_message_id or getattr(source, "message_id", None))
        if getattr(source, "platform", None) == Platform.SLACK:
            # Per-turn egress identity: Slack chat.startStream needs recipient_user_id/team_id, and the relay
            # adapter's _with_scope fallback reads them from per-chat caches a CONCURRENT turn overwrites.
            # Stamping authentic values here reduces the cache to a restart/synthetic-send fallback.
            team_id = getattr(source, "scope_id", None)
            user_id = getattr(source, "user_id", None)
            if team_id or user_id:
                metadata = dict(metadata or {})
                if team_id:
                    metadata["slack_team_id"] = str(team_id)
                    metadata.setdefault("scope_id", str(team_id))
                if user_id:
                    metadata.setdefault("user_id", str(user_id))
        # Routed profile for shared state.db namespaces: under profile_routes the transport adapter's
        # stamp is not the profile that wrote the binding (Telegram prune path needs it).
        profile = str(getattr(source, "profile", None) or "").strip()
        if profile and metadata is not None:
            metadata = dict(metadata)
            metadata["hermes_profile"] = profile
        return metadata

    def _thread_metadata_for_target(
        self, platform: Optional[Platform], chat_id: Optional[str], thread_id: Optional[str], *,
        chat_type: Optional[str] = None, reply_to_message_id: Optional[str] = None,
        adapter: Optional[Any] = None) -> Optional[Dict[str, Any]]:
        """Build thread metadata for synthetic sends that only have routing state."""
        if thread_id is None:
            return None
        metadata: Dict[str, Any] = {"thread_id": thread_id}
        if self._is_telegram_dm_topic_target(
            platform, chat_id, thread_id, chat_type=chat_type, adapter=adapter):
            metadata["telegram_dm_topic_reply_fallback"] = True
            # DM topic lanes need direct_messages_topic_id so synthetic sends reach the topic without a reply anchor.
            tid = str(thread_id)
            if tid and tid not in {"", "1"}:
                metadata["direct_messages_topic_id"] = tid
            if reply_to_message_id is not None:
                metadata["telegram_reply_to_message_id"] = str(reply_to_message_id)
        if platform == Platform.SLACK and reply_to_message_id is not None:
            # Slack's reply_in_thread=false path uses message_id to tell real threads from synthetic keys.
            metadata["message_id"] = str(reply_to_message_id)
        return metadata

    @staticmethod
    def _is_telegram_dm_topic_target(
        platform: Optional[Platform], chat_id: Optional[str], thread_id: Optional[str], *,
        chat_type: Optional[str] = None, adapter: Optional[Any] = None) -> bool:
        """Return True when a target is a Telegram private DM topic lane."""
        if platform != Platform.TELEGRAM or thread_id is None:
            return False
        if chat_type == "dm":
            return True
        # Resolve the lookup on the CLASS, not the instance: getattr() on a MagicMock auto-creates callable
        # children, so an instance lookup would report a DM topic for every test double. Only a dict counts.
        if adapter is not None and chat_id:
            get_dm_topic_info = getattr(type(adapter), "_get_dm_topic_info", None)
            if callable(get_dm_topic_info):
                try:
                    topic_info = get_dm_topic_info(adapter, str(chat_id), str(thread_id))
                except Exception:
                    logger.debug("Failed to inspect Telegram DM topic metadata", exc_info=True)
                else:
                    return isinstance(topic_info, dict)
        return False

    _reply_anchor_for_event = staticmethod(_reply_anchor_for_event)

    # Built-in platforms where ``/update`` is allowed (programmatic interfaces must not trigger updates).
    # Plugin-migrated platforms declare ``allow_update_command=True`` on their ``PlatformEntry`` instead.
    _UPDATE_ALLOWED_PLATFORMS = frozenset({
        Platform.TELEGRAM, Platform.SLACK, Platform.WHATSAPP, Platform.SIGNAL, Platform.MATRIX,
        Platform.EMAIL, Platform.SMS, Platform.DINGTALK,
        Platform.FEISHU, Platform.WECOM, Platform.WECOM_CALLBACK, Platform.WEIXIN, Platform.BLUEBUBBLES, Platform.QQBOT, Platform.LOCAL,
    })

    def _set_session_env(self, context: SessionContext) -> list:
        """Set session context variables (contextvars, not os.environ, so concurrent messages can't
        overwrite each other). Returns reset tokens for ``_clear_session_env`` in a ``finally``."""
        from gateway.session_context import set_session_vars
        # Async-delivery capability tells async tools whether this channel can wake a later turn. Default
        # True keeps CLI/unknown paths working; stateless adapters (api_server) declare False.
        _adapters = getattr(self, "adapters", None) or {}
        _adapter = _adapters.get(context.source.platform)
        _async_delivery = getattr(_adapter, "supports_async_delivery", True)
        return set_session_vars(
            platform=context.source.platform.value,
            chat_id=context.source.chat_id,
            chat_type=str(context.source.chat_type) if context.source.chat_type else "",
            chat_name=context.source.chat_name or "",
            thread_id=str(context.source.thread_id) if context.source.thread_id else "",
            user_id=str(context.source.user_id) if context.source.user_id else "",
            user_id_alt=str(context.source.user_id_alt) if context.source.user_id_alt else "",
            user_name=str(context.source.user_name) if context.source.user_name else "",
            scope_id=str(getattr(context.source, "scope_id", "") or ""),
            session_key=context.session_key,
            message_id=str(context.source.message_id) if context.source.message_id else "",
            profile=getattr(context.source, "profile", "") or "",
            async_delivery=_async_delivery,
            cron_session="")

    def _clear_session_env(self, tokens: list) -> None:
        """Restore session context variables to their pre-handler values."""
        from gateway.session_context import clear_session_vars
        clear_session_vars(tokens)

    async def _run_in_executor_with_context(self, func, *args):
        """Run blocking work in the thread pool while preserving session contextvars."""
        loop = asyncio.get_running_loop()
        ctx = copy_context()
        return await loop.run_in_executor(
            self._get_executor(), ctx.run, func, *args)

    def _get_executor(self) -> concurrent.futures.ThreadPoolExecutor:
        """Return the gateway-owned executor for blocking agent work."""
        lock = getattr(self, "_executor_lock", None)
        if lock is None:
            lock = threading.Lock()
            self._executor_lock = lock

        with lock:
            if getattr(self, "_executor_closing", False):
                raise RuntimeError("Gateway is shutting down; executor unavailable")
            executor = getattr(self, "_executor", None)
            if executor is None or getattr(executor, "_shutdown", False):
                executor = concurrent.futures.ThreadPoolExecutor(
                    max_workers=10, thread_name_prefix="hermes-gateway")
                self._executor = executor
            return executor

    def _shutdown_executor(self, drain_timeout: float = 0.0) -> int:
        """Stop the gateway-owned executor; returns the number of worker threads still running.

        ``drain_timeout=0`` is fire-and-forget; shutdown passes a bounded budget so blocking DB work
        cannot outlive ``SessionDB.close()``. ``cancel_futures`` only drops unstarted work and a cancelled
        ``run_in_executor`` awaitable does not stop its thread, so running workers are joined explicitly.
        """
        lock = getattr(self, "_executor_lock", None)
        if lock is None:
            return 0

        with lock:
            self._executor_closing = True
            executor = getattr(self, "_executor", None)
            self._executor = None

        if executor is None:
            return 0

        try:
            executor.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            executor.shutdown(wait=False)

        # shutdown() has no timeout, so join workers directly; `_threads` is absent on test doubles (no wait).
        workers = list(getattr(executor, "_threads", None) or ())
        deadline = time.monotonic() + max(float(drain_timeout or 0.0), 0.0)
        for worker in workers:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            worker.join(remaining)
        return sum(1 for worker in workers if worker.is_alive())

    # (section, key) config values baked into the agent at construction: a change MUST invalidate the
    # cached agent or a mid-gateway edit is silently ignored. Add new baked-in settings here.
    _CACHE_BUSTING_CONFIG_KEYS: tuple = (
        ("model", "context_length"), ("model", "max_tokens"), ("compression", "enabled"),
        ("compression", "progress_notices"), ("compression", "threshold"),
        ("compression", "model_thresholds"), ("compression", "threshold_tokens"),
        ("compression", "codex_gpt55_autoraise"), ("compression", "codex_app_server_auto"),
        ("compression", "codex_responses_native"),
        ("compression", "codex_responses_compact_threshold"), ("compression", "in_place"),
        ("compression", "checkpoint_required"), ("compression", "micro_compact"),
        ("compression", "micro_compact_every_n_turns"),
        ("compression", "micro_compact_defrag_threshold_tokens"), ("compression", "target_ratio"),
        ("compression", "tail_mode"), ("compression", "protect_last_n"),
        ("compression", "proactive_prune_tokens"),
        ("compression", "proactive_prune_min_result_chars"),
        ("compression", "proactive_prune_min_reclaim_tokens"),
        ("compression", "min_tail_user_messages"), ("agent", "disabled_toolsets"),
        ("memory", "provider"), ("checkpoints", "enabled"), ("checkpoints", "max_snapshots"),
        ("checkpoints", "max_total_size_mb"), ("checkpoints", "max_file_size_mb"))

    _HONCHO_CACHE_BUSTING_KEYS = (
        "honcho.peer_name", "honcho.ai_peer", "honcho.pin_peer_name", "honcho.runtime_peer_prefix",
        "honcho.user_peer_aliases")
    _HONCHO_CACHE_BUSTING_MEMO: dict[tuple[str, int | None], dict[str, Any]] = {}

    @staticmethod
    def _init_cached_agent_for_turn(agent: Any, interrupt_depth: int) -> None:
        """Reset per-turn state on a cached agent before a new turn starts.

        The activity ts/desc/provenance triple resets together and only at depth 0 — else a session idle
        29 min trips the watchdog before the first call; interrupt-recursive turns keep it so stuck-turn
        idle time accumulates to the 30-min timeout.
        """
        if interrupt_depth == 0:
            from agent.session_activity import ActivityProvenance

            agent._last_activity_ts = time.time()
            agent._last_activity_desc = "starting new turn (cached)"
            agent._last_activity_provenance = ActivityProvenance.UNKNOWN
            # Reset the SessionDB flush cursor so the new turn's messages are fully persisted — a stale
            # value from the previous turn makes `_flush_messages_to_session_db` skip new rows.
            if hasattr(agent, "_last_flushed_db_idx"):
                agent._last_flushed_db_idx = 0
        agent._api_call_count = 0

    def _profile_name_for_source(self, source: SessionSource) -> Optional[str]:
        """Resolve the profile name for an inbound source via configured routes (most specific wins).

        ``None`` = default/active profile. Gated on ``multiplex_profiles``: the scoped run only activates
        under multiplexing, else keys would be profile-namespaced while the agent ran in ``agent:main``.
        """
        config = getattr(self, "config", None)
        if not getattr(config, "multiplex_profiles", False):
            return None
        routes = getattr(config, "profile_routes", None)
        if not routes:
            return None
        from gateway.profile_routing import ProfileRouteRejected, match_profile_route
        try:
            matched = match_profile_route(
                routes, platform=source.platform.value, guild_id=getattr(source, "guild_id", None),
                chat_id=source.chat_id, thread_id=getattr(source, "thread_id", None),
                parent_chat_id=getattr(source, "parent_chat_id", None))
        except Exception:
            logger.warning(
                "Profile route matching failed for %s/%s, falling back to default",
                source.platform, source.chat_id, exc_info=True)
            return None
        if matched:
            try:
                served = {name for name, _home in _multiplex_profile_homes(config)}
            except Exception as exc:
                logger.warning(
                    "Rejecting profile route %r because the served-profile set could not be resolved",
                    matched.name,
                    exc_info=True)
                raise ProfileRouteRejected(matched.name) from exc
            if matched.profile not in served:
                logger.warning(
                    "Rejecting profile route %r: target profile %r is not served",
                    matched.name,
                    matched.profile)
                raise ProfileRouteRejected(matched.name)
            return matched.profile
        logger.debug(
            "No profile route matched: platform=%s chat_id=%s thread_id=%s parent_chat_id=%s",
            source.platform.value, source.chat_id,
            getattr(source, "thread_id", None), getattr(source, "parent_chat_id", None))
        return None

    def _resolve_profile_home_for_source(self, source: SessionSource) -> "Path":
        """Resolve which profile's HERMES_HOME serves this source: ``source.profile``, then
        ``_profile_name_for_source`` (sources bypassing ``build_source``), then the active profile."""
        from gateway.profile_routing import ProfileRouteRejected
        from hermes_cli.profiles import (
            get_active_profile_name, get_profile_dir, profile_exists)
        from hermes_constants import get_hermes_home

        explicit_profile = None  # explicitly requested (source or routing) vs. default fallback
        try:
            name = (source.profile or "").strip() or self._profile_name_for_source(source)
            explicit_profile = name or None
            if not name:
                name = get_active_profile_name() or "default"

            profile_dir = get_profile_dir(name)
            if explicit_profile and not profile_exists(name):
                logger.warning(
                    "Profile %r does not exist for source %s/%s (guild_id=%s), "
                    "falling back to global HERMES_HOME",
                    explicit_profile,
                    source.platform.value,
                    source.chat_id,
                    getattr(source, "guild_id", None))
                return get_hermes_home()
            return profile_dir
        except ProfileRouteRejected:
            raise
        except Exception:
            logger.warning(
                "Failed to resolve profile directory for source %s/%s (guild_id=%s), "
                "falling back to global HERMES_HOME: %s",
                source.platform.value,
                source.chat_id,
                getattr(source, "guild_id", None),
                explicit_profile or "(no profile)",
                exc_info=True)
            return get_hermes_home()

    @dataclasses.dataclass
    class _RunAgentDisplay:
        """Per-turn display / progress settings resolved by ``_run_agent_display_settings``."""

        user_config: Any = None
        platform_key: Any = None
        enabled_toolsets: Any = None
        disabled_toolsets: Any = None
        resolve_display_setting: Any = None
        progress_mode: Any = None
        progress_grouping: Any = None
        _display_surface_mode: Any = None
        tool_progress_enabled: Any = None
        _live_status_mode: Any = None
        _live_status_adapter: Any = None
        log_mode_enabled: Any = None
        log_queue: Any = None
        interim_assistant_messages_enabled: Any = None
        _thinking_enabled: Any = None
        _native_slack_task_cards: Any = None
        needs_progress_queue: Any = None
        _generic_status_phrase: Any = None

    @dataclasses.dataclass
    class _RunAgentWorker:
        """Executor future + inactivity-watchdog handles for one ``_run_agent_inner`` turn."""

        executor_task: Any = None
        agent_timeout: Optional[float] = None
        agent_warning: Optional[float] = None
        task_id: str = ""
        process_baseline: Any = None
        worker_done: Any = None
        timeout_fired: Any = None
        cleanup_lock: Any = None
        is_current: Any = None


def _run_planned_stop_watcher(
    stop_event: threading.Event, runner, loop: asyncio.AbstractEventLoop, shutdown_handler, *,
    poll_interval: float = 0.5) -> None:
    """Poll for the planned-stop marker and trigger graceful shutdown.

    On Windows ``add_signal_handler`` is unavailable, so ``hermes gateway stop`` would never drain;
    this cheap watcher (runs everywhere) turns the marker into the same shutdown-handler call a SIGTERM
    would. On POSIX the signal handler consumes the marker first; ``_running``/``_draining`` guard re-triggers.
    """
    from gateway.status import (
        _get_planned_stop_marker_path, planned_stop_marker_targets_self)
    marker_path = _get_planned_stop_marker_path()
    while not stop_event.is_set():
        try:
            if (
                marker_path.exists()
                and not getattr(runner, "_draining", False)
                and getattr(runner, "_running", False)):
                # A marker may target a PREVIOUS instance (different PID) that exited before stop() cleaned
                # up; firing on it means an "UNKNOWN" exit and a watchdog crash-loop. The probe unlinks
                # stale/malformed markers.
                if not planned_stop_marker_targets_self():
                    stop_event.wait(poll_interval)
                    continue
                # Same path as a real signal; the handler consumes the marker (validates pid + start_time).
                loop.call_soon_threadsafe(shutdown_handler, None)
                break
        except Exception as _e:
            logger.debug("Planned-stop watcher tick error: %s", _e)
        stop_event.wait(poll_interval)


def _housekeeping_chore(label: str, fn, *args, **kwargs) -> None:
    """Run one housekeeping chore; failures log at debug (a persistent failure such as a broken
    import after a partial update would otherwise warn every tick forever) and never stop the loop."""
    try:
        fn(*args, **kwargs)
    except Exception as exc:
        logger.debug("%s error: %s", label, exc)


def _housekeeping_channel_directory(adapters, loop) -> None:
    from gateway.channel_directory import build_channel_directory
    if loop is not None:
        # build_channel_directory is async (Slack web calls) and this is a background thread:
        # schedule onto the gateway loop and wait briefly so refresh failures still log.
        fut = safe_schedule_threadsafe(
            build_channel_directory(adapters), loop, logger=logger,
            log_message="Channel directory refresh scheduling error")
        if fut is not None:
            fut.result(timeout=30)


def _housekeeping_media_caches() -> None:
    """Every platform media cache prunes on the same hourly cadence (24h max age)."""
    from gateway.platforms.base import (
        cleanup_audio_cache, cleanup_document_cache, cleanup_image_cache, cleanup_screenshot_cache,
        cleanup_video_cache)
    from tools.tool_result_storage import cleanup_spillover_cache
    from tools.environments.local import cleanup_terminal_temp_cache
    from tools.bot_mode_dm import cleanup_bot_dm_cache
    from tools.bot_relay import cleanup_bot_relay_artifacts

    for cache_name, cleanup_fn in (
        ("Image", cleanup_image_cache), ("Document", cleanup_document_cache),
        ("Audio", cleanup_audio_cache), ("Video", cleanup_video_cache),
        ("Screenshot", cleanup_screenshot_cache), ("Spillover", cleanup_spillover_cache),
        ("Terminal temp", cleanup_terminal_temp_cache), ("Bot DM", cleanup_bot_dm_cache),
        ("Bot relay", cleanup_bot_relay_artifacts)):
        def _one(name=cache_name, fn=cleanup_fn):
            removed = fn(max_age_hours=24)
            if removed:
                logger.info("%s cache cleanup: removed %d stale file(s)", name, removed)
        _housekeeping_chore(f"{cache_name} cache cleanup", _one)


def _housekeeping_paste_sweep() -> None:
    from hermes_cli.debug import _sweep_expired_pastes
    deleted, remaining = _sweep_expired_pastes()
    if deleted:
        logger.info("Paste sweep: deleted %d expired paste(s), %d pending", deleted, remaining)


def _housekeeping_misfire_catch_up(cron_provider, adapters, loop) -> None:
    """External cron providers only: fire jobs whose time passed with no external fire delivered (dead
    loopback hop). No-op for the built-in ticker; enforces misfire_grace_minutes; CAS claim de-dupes."""
    from cron.scheduler_provider import fire_overdue_jobs
    caught_up = fire_overdue_jobs(cron_provider, adapters=adapters, loop=loop)
    if caught_up:
        logger.info("Misfire catch-up: fired %d overdue job(s)", caught_up)


def _housekeeping_curator() -> None:
    """maybe_run_curator() is gated by config.interval_hours (7 days default); this is the poll."""
    from agent.curator import maybe_run_curator
    maybe_run_curator(idle_for_seconds=float("inf"), on_summary=lambda msg: logger.info("curator: %s", msg))


def _housekeeping_skill_sync() -> None:
    """Inert unless the access gate is open and a sync base URL is configured."""
    from tools.skills_sync_client import maybe_pull_skills
    maybe_pull_skills()


def _housekeeping_org_skill_sync() -> None:
    """Gated on real org membership (the token must carry an org role): solo accounts never reach the network."""
    from tools.skills_sync_client import maybe_pull_org_skills
    maybe_pull_org_skills()


def _housekeeping_auto_archive() -> None:
    """Stale-session auto-archive on a live timer (the startup hook fires once); maybe_auto_archive()
    is gated by sessions.min_interval_hours. Opens its own SessionDB — SQLite connections are thread-bound."""
    from hermes_cli.config import load_config as _load_full_config
    from hermes_state import get_shared_session_db, release_or_close
    _sess_cfg = (_load_full_config().get("sessions") or {})
    if _sess_cfg.get("auto_archive", False):
        _adb = get_shared_session_db()
        try:
            _adb.maybe_auto_archive(
                idle_days=float(_sess_cfg.get("auto_archive_days", 3)),
                min_interval_hours=int(_sess_cfg.get("min_interval_hours", 24)))
        finally:
            release_or_close(_adb)


def _housekeeping_deferred_fts_retry() -> None:
    """A SessionDB opened while another process held the rebuild lock fails closed onto the LIKE fallback
    and the gateway stays up for days. Non-blocking, rate-limited inside SessionDB; no-op when not stale."""
    from hermes_state_registry import live_shared_session_dbs
    for _sdb in live_shared_session_dbs():
        _retry = getattr(_sdb, "retry_deferred_fts_recovery", None)
        if callable(_retry) and _retry():
            logger.info(
                "Deferred state.db FTS rebuild completed in-process for %s; full-text search restored.",
                getattr(_sdb, "db_path", "state.db"))


def _housekeeping_memory_trim() -> None:
    """Messaging-gateway counterpart to the TUI idle reaper; config-gated and rate-limited inside."""
    from hermes_cli.mem_trim import trim_memory
    trim_memory(reason="messaging gateway housekeeping")


def _start_gateway_housekeeping(stop_event: threading.Event, adapters=None, loop=None, interval: int = 60, cron_provider=None):
    """Background thread for gateway-only periodic chores (NOT cron).

    Separate from the cron trigger so chores run regardless of ``CronScheduler`` provider (an external
    scale-to-zero provider has no 60s loop). Cadences are in ticks of ``interval``; inner gates own the
    real cadence.
    """
    chores: list[tuple[int, str, Any]] = [
        (5, "Channel directory refresh", lambda: adapters and _housekeeping_channel_directory(adapters, loop)),
        (60, "Media cache cleanup", _housekeeping_media_caches),
        (60, "Paste sweep", _housekeeping_paste_sweep)]
    if cron_provider is not None:
        chores.append((5, "Misfire catch-up sweep", lambda: _housekeeping_misfire_catch_up(cron_provider, adapters, loop)))
    chores += [
        (60, "Curator tick", _housekeeping_curator),
        (60, "Sync pull tick", _housekeeping_skill_sync),
        (60, "Org sync pull tick", _housekeeping_org_skill_sync),
        (60, "Auto-archive tick", _housekeeping_auto_archive),
        (1, "Deferred FTS retry tick", _housekeeping_deferred_fts_retry),
        (1, "gateway housekeeping memory trim", _housekeeping_memory_trim)]

    logger.info("Gateway housekeeping started (interval=%ds)", interval)
    tick_count = 0
    while not stop_event.is_set():
        tick_count += 1
        for every, label, fn in chores:
            if tick_count % every == 0:
                _housekeeping_chore(label, fn)
        stop_event.wait(timeout=interval)
    logger.info("Gateway housekeeping stopped")


def _start_cron_ticker(stop_event: threading.Event, adapters=None, loop=None, interval: int = 60):
    """DEPRECATED shim — runs ONLY the built-in in-process cron tick loop; the trigger now lives behind
    the ``CronScheduler`` provider and housekeeping in ``_start_gateway_housekeeping``."""
    from cron.scheduler_provider import InProcessCronScheduler
    InProcessCronScheduler().start(stop_event, adapters=adapters, loop=loop, interval=interval)


def _stop_cron_provider(provider) -> None:
    """Stop a cron provider without letting it choose the gateway exit code."""
    try:
        provider.stop()
    except SystemExit as exc:
        logger.warning(
            "Cron provider stop() attempted to exit the gateway with code %s; ignoring", exc.code)
    except Exception as exc:
        logger.debug("Cron provider stop() error: %s", exc)


# Upper bound for cooperatively draining the cron ticker on shutdown: the cron thread blocks on
# ``future.result(timeout=60)`` (cron/scheduler.py::_deliver_result), so a delivery unblocks in ~60s.
_CRON_SHUTDOWN_DRAIN_TIMEOUT = 65.0

# Upper bound for draining the housekeeping ticker on shutdown: the channel-directory refresh blocks
# on ``fut.result(timeout=30)``, so cover that 30s plus margin or an in-flight refresh is abandoned.
_HOUSEKEEPING_SHUTDOWN_DRAIN_TIMEOUT = 35.0


async def _await_thread_exit(
    thread: Optional[threading.Thread], timeout: float, poll: float = 0.1) -> bool:
    """Wait for a daemon thread to exit WITHOUT blocking the event loop; True if it exited in time.

    A synchronous ``join()`` freezes the loop — fatal for the cron ticker, whose in-flight delivery is
    a coroutine scheduled onto *this* loop: it could never run, so the join timed out and the message dropped.
    """
    if thread is None:
        return True
    deadline = asyncio.get_running_loop().time() + max(0.0, timeout)
    while thread.is_alive() and asyncio.get_running_loop().time() < deadline:
        await asyncio.sleep(poll)
    return not thread.is_alive()


async def _shutdown_mcp_servers_nonblocking(timeout: float = 5.0) -> bool:
    """Close MCP servers off-loop with a bounded wait; True when done within ``timeout``.

    ``shutdown_mcp_servers()`` can block ~15s; on the loop thread that lets short-grace supervisors (s6
    3s) SIGKILL us before ``mark_exited()`` runs, so every later boot reports a phantom unclean death.
    On timeout shutdown proceeds and the daemon thread is left to finish or die.
    """

    def _do() -> None:
        try:
            from tools.mcp_tool import shutdown_mcp_servers

            shutdown_mcp_servers()
        except Exception:
            logger.debug("MCP shutdown raised", exc_info=True)

    thread = threading.Thread(target=_do, name="mcp-shutdown", daemon=True)
    thread.start()
    done = await _await_thread_exit(thread, timeout=timeout)
    if not done:
        logger.warning(
            "MCP shutdown did not finish within %.1fs; continuing gateway "
            "teardown (background thread will be reaped at process exit)",
            timeout)
    return done


def _shutdown_gateway_health_export(runner: Any) -> None:
    """Idempotently drain and detach Gateway Health OTLP export."""
    runtime = getattr(runner, "_gateway_health_export_runtime", None)
    if runtime is None:
        return
    runner._gateway_health_export_runtime = None
    try:
        runtime.shutdown()
    except Exception:
        logger.debug("gateway health OTLP export shutdown failed", exc_info=True)


def _gateway_stderr_formatter() -> logging.Formatter:
    """Return the redacting formatter used by the gateway stderr stream."""
    from agent.redact import RedactingFormatter

    return RedactingFormatter("%(asctime)s %(levelname)s %(name)s: %(message)s")


def _replace_target_belongs_to_other_profile(existing_pid: int) -> bool:
    """Return True when ``--replace`` must refuse to signal ``existing_pid``.

    A poisoned/stale PID record can point at another profile's LIVE gateway (cross-profile SIGTERM
    restart loop). Ownership is decided by the persisted identity record ALONE, only while bound to the
    live target by exact PID + start-time. Live argv can never PROVE ownership (no HERMES_HOME); it is
    only a consistency check. Missing, legacy, conflicting or unprovable identity → refuse (fail closed).
    """
    try:
        from gateway.status import (
            _get_pid_path, _get_process_hermes_home, _get_process_start_time, _pid_from_record,
            _read_pid_record, _record_looks_like_gateway, _read_process_cmdline, _same_hermes_home)

        our_home = _get_process_hermes_home()

        def refuse(msg: str, *args, level=logging.WARNING) -> bool:
            logger.log(level, "Refusing --replace: " + msg, *args)
            return True

        # Authorize from the persisted identity record — bound claim: the record must describe THIS pid
        # with THIS live start time, otherwise it is stale/poisoned and proves nothing.
        record = _read_pid_record(_get_pid_path())
        if not isinstance(record, dict) or not _record_looks_like_gateway(record):
            return refuse("no valid gateway pid record to prove ownership of PID %s.", existing_pid)
        record_pid = _pid_from_record(record)
        if record_pid != existing_pid:
            return refuse("pid record names %s, not target %s.", record_pid, existing_pid)
        recorded_start = record.get("start_time")
        if not isinstance(recorded_start, int) or isinstance(recorded_start, bool):
            return True
        if _get_process_start_time(existing_pid) != recorded_start:
            return refuse("pid record start-time does not match the live process %s (stale/PID-reuse record).",
                          existing_pid)
        recorded_home = record.get("hermes_home")
        if not isinstance(recorded_home, str) or not recorded_home.strip():
            return refuse("pid record predates hermes_home stampings; ownership of PID %s unprovable.",
                          existing_pid)
        if not _same_hermes_home(recorded_home, our_home):
            return refuse("pid record belongs to a different HERMES_HOME (%s, ours %s). Remove the stale PID "
                          "record or stop the owning profile explicitly.", recorded_home, our_home,
                          level=logging.ERROR)
        # Argv consistency check (never authority): an explicit profile flag / HERMES_HOME= contradicting
        # our home refuses even if the record agreed; bare argv adds nothing; on probe failure the record decides.
        live_cmdline = _best_effort(lambda: _read_process_cmdline(existing_pid))
        if live_cmdline and _looks_like_profile_conflict_from_cmdline(live_cmdline, our_home):
            return refuse("target PID %s command line explicitly advertises a different profile than "
                          "HERMES_HOME %s.", existing_pid, our_home, level=logging.ERROR)
        return False
    except Exception:
        # Destructive action + unknown ownership => fail closed.
        logger.warning(
            "cross-profile --replace ownership probe failed for PID %s; refusing to signal",
            existing_pid,
            exc_info=True)
        return True


def _looks_like_profile_conflict_from_cmdline(command: str, our_home) -> bool:
    """Token-exact contradiction check between a target argv and our home (authority is the pid record).

    Substring matching is not identity: ``--profile timothy`` must NOT read as profile ``tim``.
    Returns False whenever the argv does not clearly contradict our home.
    """
    from gateway.status import _profile_name_for_home

    profile_name = _profile_name_for_home(our_home)
    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = command.split()

    def _flag_value(flag: str) -> Optional[str]:
        """Value of ``--flag X`` / ``--flag=X`` occurrences, token-exact."""
        values = []
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            if tok == flag and i + 1 < len(tokens):
                values.append(tokens[i + 1])
                i += 2
                continue
            if tok.startswith(flag + "="):
                values.append(tok[len(flag) + 1:])
            i += 1
        return values[-1] if values else None

    def _env_home_value() -> Optional[str]:
        """HERMES_HOME=<path> env-style assignment on the argv, token-exact."""
        prefix = "HERMES_HOME="
        for tok in reversed(tokens):
            if tok.startswith(prefix):
                return tok[len(prefix):]
        return None

    def _norm(path: str) -> str:
        return os.path.normcase(os.path.normpath(path))

    for flag in ("--profile", "-p"):
        value = _flag_value(flag)
        if value is None:
            continue
        # Named-profile home: a DIFFERENT explicit profile contradicts it (legacy default argv never carried
        # profile flags). Default/root home: ANY explicit named-profile flag contradicts it.
        if profile_name is None or profile_name == "default" or value != profile_name:
            return True
    home_value = _flag_value("--hermes-home") or _env_home_value()
    return bool(home_value is not None and _norm(home_value) != _norm(str(our_home)))


def _clear_takeover_marker_quiet() -> None:
    """Best-effort: the marker is scoped to one target; a stale one would grief an unrelated shutdown."""
    try:
        from gateway.status import clear_takeover_marker
        clear_takeover_marker()
    except Exception:
        pass


async def _wait_for_pid_exit(pid: int, attempts: int, delay: float) -> bool:
    """Poll for process exit without blocking the loop (a blocking sleep freezes signal handlers and
    health checks). ``os.kill(pid, 0)`` on Windows is NOT a no-op — use the handle-based check."""
    from gateway.status import _pid_exists
    for _ in range(attempts):
        if not _pid_exists(pid):
            return True
        await asyncio.sleep(delay)
    return False


async def _start_gateway_replace_existing_instance(existing_pid: int, replace: bool) -> bool:
    """Handle a live gateway PID under this HERMES_HOME: replace it (``--replace``) or refuse.

    Returns False when startup must abort (refused, permission denied, target still alive).
    """
    from gateway.status import get_process_start_time, remove_pid_file, terminate_pid
    if not replace:
        hermes_home = str(get_hermes_home())
        logger.error(
            "Another gateway instance is already running (PID %d, HERMES_HOME=%s). "
            "Use 'hermes gateway restart' to replace it, or 'hermes gateway stop' first.",
            existing_pid, hermes_home)
        print(
            f"\n❌ Gateway already running (PID {existing_pid}).\n"
            f"   Use 'hermes gateway restart' to replace it,\n"
            f"   or 'hermes gateway stop' to kill it first.\n"
            f"   Or use 'hermes gateway run --replace' to auto-replace.\n")
        return False

    # Never signal a live process we cannot prove belongs to this HERMES_HOME: a poisoned PID record
    # steering --replace at another profile's gateway is exactly the restart-loop shape to avoid.
    if _replace_target_belongs_to_other_profile(existing_pid):
        from gateway.status import _get_process_hermes_home

        logger.error(
            "Refusing --replace: PID %d cannot be proven to belong "
            "to this profile's gateway (HERMES_HOME %s). Remove the "
            "stale PID record or stop the owning profile explicitly.",
            existing_pid,
            _get_process_hermes_home())
        return False
    existing_start_time = get_process_start_time(existing_pid)
    logger.info("Replacing existing gateway instance (PID %d) with --replace.", existing_pid)
    # Takeover marker: the target's shutdown handler recognises its SIGTERM as a planned takeover and
    # exits 0 (exit 1 would trigger systemd's Restart=on-failure and a flap loop against us).
    try:
        from gateway.status import write_takeover_marker
        write_takeover_marker(existing_pid)
    except Exception as e:
        logger.debug("Could not write takeover marker: %s", e)
    # Snapshot children BEFORE signalling: once it exits, orphans are reparented and invisible to a parent
    # walk; surviving adapter subprocesses hold scoped token locks and block us (Windows already tree-kills).
    try:
        from gateway.status import _snapshot_gateway_children
        _old_gateway_children = _snapshot_gateway_children(existing_pid)
    except Exception:
        _old_gateway_children = []
    try:
        terminate_pid(existing_pid, force=False)
    except ProcessLookupError:
        pass  # Already gone
    except (PermissionError, OSError):
        logger.error("Permission denied killing PID %d. Cannot replace.", existing_pid)
        _clear_takeover_marker_quiet()
        return False
    # Up to 10s for SIGTERM, then SIGKILL.
    if not await _wait_for_pid_exit(existing_pid, 20, 0.5):
        logger.warning("Old gateway (PID %d) did not exit after SIGTERM, sending SIGKILL.", existing_pid)
        old_gateway_exited = False
        try:
            terminate_pid(existing_pid, force=True, expected_start_time=existing_start_time)
        except ProcessLookupError:
            old_gateway_exited = True
        except (PermissionError, OSError):
            pass
        # Confirm SIGKILL actually took (uninterruptible sleep, zombie) before clearing PID file / scoped
        # locks, or two live gateways fight over the same token.
        if not old_gateway_exited and not await _wait_for_pid_exit(existing_pid, 20, 0.25):
            logger.error(
                "Old gateway (PID %d) still appears alive after SIGKILL; "
                "aborting replacement to avoid a duplicate gateway.",
                existing_pid)
            _clear_takeover_marker_quiet()
            return False
    # Reap orphaned children (POSIX; mirrors Windows taskkill /T) so they stop holding scoped token locks.
    try:
        from gateway.status import reap_gateway_children
        reap_gateway_children(_old_gateway_children, parent_pid=existing_pid)
    except Exception:
        logger.debug("Child reap for replaced gateway PID %d failed", existing_pid, exc_info=True)
    remove_pid_file()
    # remove_pid_file() is a no-op when the PID doesn't match; force-unlink covers a crashed old process.
    with suppress(Exception):
        (get_hermes_home() / "gateway.pid").unlink(missing_ok=True)
    # The old process may not have consumed the marker (SIGKILL'd before its handler read it).
    _clear_takeover_marker_quiet()
    # Stopped (Ctrl+Z) processes don't release scoped locks on exit; stale lock files block the new gateway.
    try:
        from gateway.status import release_all_scoped_locks
        _released = release_all_scoped_locks(owner_pid=existing_pid, owner_start_time=existing_start_time)
        if _released:
            logger.info("Released %d stale scoped lock(s) from old gateway.", _released)
    except Exception:
        pass
    return True


def _start_gateway_configure_logging(verbosity: Optional[int]) -> None:
    """Sync bundled skills, set up file logging + startup security audit, and the -v/-q stderr handler."""
    def _sync_skills() -> None:
        from tools.skills_sync import sync_skills
        sync_skills(quiet=True)

    _best_effort(_sync_skills)

    # Centralized logging — agent.log (INFO+), errors.log (WARNING+), gateway.log (INFO+, gateway
    # records only). Idempotent, so repeated calls from AIAgent.__init__ don't duplicate.
    from hermes_logging import setup_logging, _safe_stderr
    setup_logging(hermes_home=_hermes_home, mode="gateway")

    def _security_audit() -> None:
        # Warn-on-load, never blocks: surfaces root / weak-SSH / unauthenticated-listener exposure.
        from hermes_cli.security_audit_startup import log_startup_security_warnings

        def _raw_cfg():
            from hermes_cli.config import read_raw_config
            return read_raw_config()

        log_startup_security_warnings(hermes_home=_hermes_home, config=_best_effort(_raw_cfg))

    _best_effort(_security_audit, "Startup security audit failed (non-fatal): %s")

    # Optional stderr handler from -v/-q: None (quiet) = none; 0 = WARNING; 1 = INFO; 2+ = DEBUG.
    if verbosity is not None:
        _stderr_level = {0: logging.WARNING, 1: logging.INFO}.get(verbosity, logging.DEBUG)
        _stderr_handler = logging.StreamHandler(_safe_stderr())
        _stderr_handler.setLevel(_stderr_level)
        _stderr_handler.setFormatter(_gateway_stderr_formatter())
        root = logging.getLogger()
        root.addHandler(_stderr_handler)
        if _stderr_level < root.level:  # so DEBUG records can reach the handler
            root.setLevel(_stderr_level)


def _start_gateway_make_shutdown_signal_handler(runner, _signal_initiated_shutdown: list):
    """Build the SIGINT/SIGTERM handler; ``_signal_initiated_shutdown[0]`` records an unplanned signal."""
    def shutdown_signal_handler(received_signal=None):
        # Planned --replace takeover (sibling wrote a marker naming this PID before SIGTERM): exit 0 so
        # systemd's Restart=on-failure doesn't revive us to flap-fight the replacer.
        planned_takeover = False
        try:
            from gateway.status import consume_takeover_marker_for_self
            planned_takeover = consume_takeover_marker_for_self()
        except Exception as e:
            logger.debug("Takeover marker check failed: %s", e)

        # Planned stop: service managers and `hermes gateway stop` also send SIGTERM, indistinguishable
        # from an external kill unless the CLI marks it first. SIGINT is an interactive Ctrl+C stop.
        planned_stop = False
        if received_signal == signal.SIGINT:
            planned_stop = True
        elif not planned_takeover:
            try:
                from gateway.status import consume_planned_stop_marker_for_self
                planned_stop = consume_planned_stop_marker_for_self()
            except Exception as e:
                logger.debug("Planned stop marker check failed: %s", e)

        # Fast (<10ms) snapshot of who's asking us to shut down — runs synchronously inside the asyncio
        # signal handler: stdlib + /proc only, no subprocesses (a sync `ps aux` here once blocked ~3s).
        try:
            from gateway.shutdown_forensics import (
                format_context_for_log, snapshot_shutdown_context, spawn_async_diagnostic)
            _shutdown_ctx = snapshot_shutdown_context(received_signal)
        except Exception as _e:
            _shutdown_ctx = None
            logger.debug("snapshot_shutdown_context failed: %s", _e)

        if planned_takeover:
            logger.info(
                "Received %s as a planned --replace takeover — exiting cleanly",
                _shutdown_ctx["signal"] if _shutdown_ctx else "SIGTERM")
        elif planned_stop:
            logger.info(
                "Received %s as a planned gateway stop — exiting cleanly",
                _shutdown_ctx["signal"] if _shutdown_ctx else "SIGTERM/SIGINT")
        else:
            _signal_initiated_shutdown[0] = True
            # Mirrored so _stop_impl suppresses the gateway_state=stopped persist for unexpected signals;
            # operator stops take the `planned_stop` branch above and leave this False (DO persist).
            runner._signal_initiated_shutdown = True
            logger.info(
                "Received %s — initiating shutdown",
                _shutdown_ctx["signal"] if _shutdown_ctx else "SIGTERM/SIGINT")

        # Always log who/what triggered the signal — the most useful line for "gateway keeps dying" tickets.
        if _shutdown_ctx is not None:
            try:
                logger.warning(
                    "Shutdown context: %s", format_context_for_log(_shutdown_ctx))
            except Exception as _e:
                logger.debug("format_context_for_log failed: %s", _e)

            # Spawn the heavyweight diagnostic (ps auxf, pstree, dmesg) detached so it can finish
            # writing even if our cgroup is torn down; bounded by an internal timeout, never blocks.
            try:
                _diag_log = _hermes_home / "logs" / "gateway-shutdown-diag.log"
                spawn_async_diagnostic(
                    _diag_log, _shutdown_ctx["signal"], timeout_seconds=5.0)
            except Exception as _e:
                logger.debug("spawn_async_diagnostic failed: %s", _e)
        asyncio.create_task(runner.stop())
    return shutdown_signal_handler


def _start_gateway_claim_pid_file() -> bool:
    """Claim the runtime lock + PID file (O_EXCL winner is the authoritative gateway). False = lost."""
    import atexit
    from gateway.status import (
        acquire_gateway_runtime_lock, get_running_pid, release_gateway_runtime_lock,
        remove_pid_file, write_pid_file)
    _current_pid = get_running_pid()
    if _current_pid is not None and _current_pid != os.getpid():
        logger.error(
            "Another gateway instance (PID %d) started during our startup. "
            "Exiting to avoid double-running.", _current_pid)
        return False
    if not acquire_gateway_runtime_lock():
        logger.error(
            "Gateway runtime lock is already held by another instance. Exiting.")
        return False
    try:
        write_pid_file()
    except FileExistsError:
        release_gateway_runtime_lock()
        logger.error(
            "PID file race lost to another gateway instance. Exiting.")
        return False
    atexit.register(remove_pid_file)
    atexit.register(release_gateway_runtime_lock)
    return True


async def _start_gateway_start_control_socket(runner):
    """Start the gateway control socket (identify/status/pause-for-update); None when unavailable."""
    import atexit
    _control_server = None
    try:
        from gateway.control_socket import GatewayControlServer

        # pause-for-update: the updater asks us to drain and exit cleanly (releasing venv file handles)
        # instead of being tree-killed — same path as SIGUSR1/service restarts. The handler runs on the
        # socket's executor thread, so the request is marshalled onto the loop; the ACK returns the budget.
        _main_loop = asyncio.get_running_loop()

        def _pause_for_update_handler() -> dict:
            try:
                from hermes_cli.gateway import _get_restart_drain_timeout

                _drain = float(_get_restart_drain_timeout())
            except Exception:
                _drain = 30.0
            accepted_box: list[bool] = []
            _done = threading.Event()

            def _request() -> None:
                try:
                    accepted_box.append(
                        runner.request_restart(detached=False, via_service=True))
                finally:
                    _done.set()

            _main_loop.call_soon_threadsafe(_request)
            _done.wait(timeout=5.0)
            accepted = bool(accepted_box and accepted_box[0])
            return {
                "pausing": accepted,
                "already_stopping": not accepted,
                "pid": os.getpid(),
                "drain_timeout": _drain}

        _control_server = GatewayControlServer(
            verb_handlers={"pause-for-update": _pause_for_update_handler})
        if not await _control_server.start():
            _control_server = None
        else:
            atexit.register(_control_server.cleanup_files)
    except Exception as _cs_exc:
        logger.debug("Control socket startup failed (non-fatal): %s", _cs_exc)
        _control_server = None
    return _control_server


def _start_gateway_start_cron_and_housekeeping(runner):
    """Start the cron scheduler thread + gateway housekeeping thread.

    Returns ``(cron_stop, cron_provider, cron_thread, housekeeping_thread)``.
    """
    # The event loop is passed so cron delivery can use live adapters (E2EE support).
    from cron.scheduler_provider import (
        InProcessCronScheduler, resolve_cron_scheduler, scheduler_for_profile_mode)
    cron_stop = threading.Event()
    multiplex_cron = bool(getattr(runner.config, "multiplex_profiles", False))
    cron_provider = scheduler_for_profile_mode(
        resolve_cron_scheduler(), multiplex_profiles=multiplex_cron)
    cron_start_kwargs: Dict[str, Any] = {"adapters": runner.adapters, "loop": asyncio.get_running_loop()}

    # Multiplex: tell the built-in ticker which profile homes to tick, else secondary profiles' cron jobs
    # show as "scheduled" but never execute because no ticker owns that store.
    if isinstance(cron_provider, InProcessCronScheduler) and multiplex_cron:
        try:
            profile_homes = _multiplex_profile_homes(runner.config)
            if profile_homes:
                cron_start_kwargs["profile_homes"] = profile_homes
                # Per-profile adapters so each profile's cron output goes via its own bot, not the default's.
                cron_start_kwargs["profile_adapters"] = getattr(runner, "_profile_adapters", None)
                # runner.adapters belongs to "default"; naming it keeps the ticker from routing a secondary's
                # cron through the default bot (even before that profile's adapter connects).
                cron_start_kwargs["default_profile"] = "default"
                logger.info(
                    "Cron scheduler will tick %d profile(s) under multiplex: %s",
                    len(profile_homes),
                    [p[0] if isinstance(p, tuple) else p for p in profile_homes])
        except Exception as exc:
            logger.warning(
                "Could not resolve profile homes for multiplex cron: %s", exc)

    # External cron providers own their remote scheduling contract; only the in-process ticker polls
    # local due jobs, so only it receives the local external-drain dispatch gate.
    if isinstance(cron_provider, InProcessCronScheduler):
        cron_start_kwargs["can_dispatch"] = lambda: not (
            runner._draining or runner._external_drain_active)
    cron_thread = threading.Thread(
        target=cron_provider.start, args=(cron_stop,), kwargs=cron_start_kwargs, daemon=True,
        name="cron-scheduler")
    cron_thread.start()

    # External providers fire over loopback HTTP to THIS process's api_server; if it never came up (usually
    # API_SERVER_KEY missing) every fire fails while manual runs work — misread as a job bug. Say it ONCE.
    if not isinstance(cron_provider, InProcessCronScheduler):
        try:
            _has_api_server = Platform.API_SERVER in (runner.adapters or {})
        except Exception:
            _has_api_server = True  # never let the tell break startup
        if not _has_api_server:
            logger.warning(
                "Cron provider '%s' is active but the api_server adapter is "
                "NOT running in this gateway — scheduled fires arrive over "
                "loopback HTTP and will all fail (jobs only run when "
                "triggered manually). Most common cause: API_SERVER_KEY is "
                "missing from this gateway process's environment. Restart "
                "the gateway through its supervisor (`hermes gateway "
                "restart`) so the profile env loads.",
                getattr(cron_provider, "name", "external"))

    # Gateway-only housekeeping runs independently of the cron provider; shares cron_stop for shutdown.
    housekeeping_thread = threading.Thread(
        target=_start_gateway_housekeeping,
        args=(cron_stop,),
        kwargs={
            "adapters": runner.adapters,
            "loop": asyncio.get_running_loop(),
            "cron_provider": cron_provider},
        daemon=True,
        name="gateway-housekeeping")
    housekeeping_thread.start()
    return cron_stop, cron_provider, cron_thread, housekeeping_thread


def _exit_with_failure_verdict(runner) -> bool:
    """True (after logging the reason) when the runner asked for a failure exit."""
    if not runner.should_exit_with_failure:
        return False
    if runner.exit_reason:
        logger.error("Gateway exiting with failure: %s", runner.exit_reason)
    return True


async def _start_gateway_shutdown_tail(
    runner, _control_server, cron_stop: threading.Event, cron_provider,
    cron_thread: threading.Thread, housekeeping_thread: threading.Thread,
    _planned_stop_watcher_stop: threading.Event, _planned_stop_watcher_thread: threading.Thread,
    _signal_initiated_shutdown: list) -> bool:
    """Post-``wait_for_shutdown`` teardown; returns the process exit verdict (True = exit 0)."""
    # Control socket first: once shutdown begins we are no longer a truthful "serving here" answer and a
    # successor must be able to bind. Early-exit paths rely on the atexit cleanup_files hook instead.
    if _control_server is not None:
        try:
            await _control_server.stop()
        except Exception:
            logger.debug("Control socket stop failed (non-fatal)", exc_info=True)

    def _stop_keepalive() -> None:
        from hermes_cli.nous_auth_keepalive import stop_nous_auth_keepalive
        stop_nous_auth_keepalive()

    _best_effort(_stop_keepalive)
    if _exit_with_failure_verdict(runner):
        return False

    # Cooperative stop, never join()ed: an in-flight cron delivery is a coroutine on THIS loop while the
    # ticker blocks on future.result(); a sync join would starve it and drop the message.
    cron_stop.set()
    _stop_cron_provider(cron_provider)
    if not await _await_thread_exit(cron_thread, timeout=_CRON_SHUTDOWN_DRAIN_TIMEOUT):
        logger.warning(
            "Cron ticker did not exit within %.0fs of shutdown — an in-flight "
            "delivery may have been dropped.", _CRON_SHUTDOWN_DRAIN_TIMEOUT)
    await _await_thread_exit(
        housekeeping_thread, timeout=_HOUSEKEEPING_SHUTDOWN_DRAIN_TIMEOUT)

    # Stop the planned-stop watcher (daemon=True so this is belt-and-suspenders).
    _planned_stop_watcher_stop.set()
    _planned_stop_watcher_thread.join(timeout=2)

    with suppress(Exception):
        await _shutdown_mcp_servers_nonblocking()

    if runner.exit_code is not None:
        raise SystemExit(runner.exit_code)

    # An unexpected SIGTERM that wasn't a planned restart exits non-zero so systemd's Restart=on-failure
    # revives the process; `hermes gateway stop` / Ctrl+C are planned stops and must not trigger revival.
    if _signal_initiated_shutdown[0] and not runner._restart_requested:
        logger.info(
            "Exiting with code 1 (signal-initiated shutdown without restart "
            "request) so systemd Restart=on-failure can revive the gateway.")
        return False  # → sys.exit(1) in the caller

    # Older restart paths may reach here without ``runner.exit_code``; keep the non-zero fallback.
    if runner._restart_via_service:
        logger.info(
            "Exiting with code 75 (service-restart requested) so the service "
            "manager relaunches the gateway.")
        raise SystemExit(75)

    return True


async def start_gateway(config: Optional[GatewayConfig] = None, replace: bool = False, verbosity: Optional[int] = 0) -> bool:
    """Start the gateway and run until interrupted; False if it failed to start (non-zero exit so
    systemd can auto-restart). ``replace`` kills any existing instance first (avoids restart-loop deadlocks)."""
    # Enable interactive exec approval on messaging platforms. Set here (not at module import) so
    # incidental imports of gateway.run from CLI/tool code don't poison HERMES_EXEC_ASK.
    os.environ["HERMES_EXEC_ASK"] = "1"

    from hermes_cli.resource_limits import apply_nofile_soft_limit

    apply_nofile_soft_limit()

    # Snapshot the checkout revision while sys.modules still matches disk, so a later `git pull` under
    # this long-lived process is detected (risky work refused) instead of crashing on stale modules.
    from gateway.code_skew import record_boot_fingerprint
    record_boot_fingerprint()

    # Duplicate-instance guard: no two gateways under one HERMES_HOME. The PID file is scoped to
    # HERMES_HOME, so multi-profile setups (distinct HERMES_HOME each) run concurrently untripped.
    from gateway.status import get_running_pid
    existing_pid = get_running_pid()
    if (
        existing_pid is not None and existing_pid != os.getpid()
        and not await _start_gateway_replace_existing_instance(existing_pid, replace)):
        return False

    _start_gateway_configure_logging(verbosity)

    runner = GatewayRunner(config)
    # Multiplex: swap the launch-home file handlers for per-profile routers so each profile's records
    # land in its own logs/. Must run after the runner resolved (possibly None) config and setup_logging.
    _enable_multiplex_log_routing(runner.config)
    # ``--replace`` is explicit startup authority, not a durable reconnect policy: GatewayRunner scopes
    # it to cold adapter connects and clears it before the background reconnect watcher starts.
    runner._platform_lock_takeover_on_start = bool(replace)

    # Track whether an unexpected signal initiated shutdown: an unexpected SIGTERM exits non-zero so
    # service managers revive us; planned stop paths write a marker first so they exit cleanly.
    _signal_initiated_shutdown = [False]

    shutdown_signal_handler = _start_gateway_make_shutdown_signal_handler(
        runner, _signal_initiated_shutdown)

    def restart_signal_handler():
        runner.request_restart(detached=False, via_service=True)

    loop = asyncio.get_running_loop()

    # Swallows transient network errors from background tasks (an unhandled telegram/httpx error in any
    # awaited coroutine would kill the gateway). Deliberately narrow — everything else hits the default.
    loop.set_exception_handler(_gateway_loop_exception_handler)

    if threading.current_thread() is threading.main_thread():
        # add_signal_handler raises NotImplementedError on Windows; SIGUSR1 is POSIX-only.
        handlers = [(sig, shutdown_signal_handler, (sig,)) for sig in (signal.SIGINT, signal.SIGTERM)]
        if hasattr(signal, "SIGUSR1"):
            handlers.append((signal.SIGUSR1, restart_signal_handler, ()))  # windows-footgun: ok — hasattr-guarded
        for sig, handler, args in handlers:
            with suppress(NotImplementedError):
                loop.add_signal_handler(sig, handler, *args)  # windows-footgun: ok — suppress(NotImplementedError)
    else:
        logger.info("Skipping signal handlers (not running in main thread).")

    # Windows has no add_signal_handler, so `hermes gateway stop`'s SIGTERM would never drain; this thread
    # polls the planned-stop marker written BEFORE the kill and drives the same shutdown path. Runs
    # everywhere (cheap) so environments masking SIGTERM still drain cleanly.
    _planned_stop_watcher_stop = threading.Event()
    _planned_stop_watcher_thread = threading.Thread(
        target=_run_planned_stop_watcher,
        args=(_planned_stop_watcher_stop, runner, loop, shutdown_signal_handler), daemon=True,
        name="planned-stop-watcher")
    _planned_stop_watcher_thread.start()

    # Claim the PID file BEFORE any adapters: two concurrent `run --replace` both pass the termination
    # wait, but only the O_EXCL winner ever opens Telegram polling, Discord sockets, etc.
    if not _start_gateway_claim_pid_file():
        return False

    # Control socket right after the PID-file claim (winning that race makes us authoritative). Non-fatal:
    # a bind failure leaves consumers on the process-scan/state-file layer.
    _control_server = await _start_gateway_start_control_socket(runner)

    def _lifecycle_record_startup() -> None:
        # Report if the previous life died uncleanly (SIGKILL / OOM / VM death), then claim the
        # sentinel for this life. After the PID-file claim so a --replace loser can't clobber it.
        from gateway.lifecycle_ledger import record_startup
        record_startup()

    def _start_keepalive() -> None:
        from hermes_cli.nous_auth_keepalive import start_nous_auth_keepalive
        start_nous_auth_keepalive()

    _best_effort(_lifecycle_record_startup, "Lifecycle ledger startup record failed: %s")
    _best_effort(_start_keepalive, "Nous auth keepalive did not start: %s")
    _ensure_windows_gateway_venv_imports()

    # MCP discovery in an executor: discover_mcp_tools() blocks up to 120s, which on the loop thread would
    # freeze platform heartbeats (Discord shard, Telegram polling).
    try:
        await _discover_gateway_mcp_tools(runner.config)
    except Exception as e:
        logger.debug("MCP tool discovery failed: %s", e)

    try:
        success = await runner.start()
    except BaseException:
        _shutdown_gateway_health_export(runner)
        raise
    if not success:
        _shutdown_gateway_health_export(runner)
        return False

    def _recover_pending() -> None:
        from gateway.shutdown_flush import recover_pending_to_db
        recovered = recover_pending_to_db()
        if recovered:
            logger.info("Recovered %d pending message(s) from shutdown flush", recovered)

    _best_effort(_recover_pending)
    if runner.should_exit_cleanly:
        _shutdown_gateway_health_export(runner)
        if runner.exit_reason:
            logger.error("Gateway exiting cleanly: %s", runner.exit_reason)
        # An explicit exit code (e.g. GATEWAY_FATAL_CONFIG_EXIT_CODE) must propagate so the s6 finish
        # script can translate it (78 → 125) and stop the restart loop instead of exiting 0.
        if runner.exit_code is not None:
            raise SystemExit(runner.exit_code)
        return True
    if not runner._running:
        # Startup aborted by restart/shutdown before running mode; preserve that path without starting cron.
        try:
            await runner.wait_for_shutdown()
            if _exit_with_failure_verdict(runner):
                return False
            with suppress(Exception):
                await _shutdown_mcp_servers_nonblocking()
            if runner.exit_code is not None:
                raise SystemExit(runner.exit_code)
            return True
        finally:
            _shutdown_gateway_health_export(runner)

    cron_stop, cron_provider, cron_thread, housekeeping_thread = (
        _start_gateway_start_cron_and_housekeeping(runner))

    # READY is emitted only after adapters, cron and housekeeping reach their running boundary;
    # missing config/systemd runtime state leaves the watchdog disabled without changing behavior.
    runner._start_systemd_watchdog()

    await runner.wait_for_shutdown()

    return await _start_gateway_shutdown_tail(
        runner, _control_server, cron_stop, cron_provider, cron_thread, housekeeping_thread,
        _planned_stop_watcher_stop, _planned_stop_watcher_thread, _signal_initiated_shutdown)


def _guard_corrupt_user_config() -> None:
    """Fail closed when the active profile's config.yaml cannot be parsed: nobody can repair it on this
    surface, and defaults would let provider auto-detection adopt ``.env`` credentials the config never
    named. Same policy and escape hatch (``HERMES_IGNORE_USER_CONFIG=1``) as ``hermes_cli/main.py``."""
    from hermes_cli.config import InvalidUserConfigError, require_parseable_user_config

    try:
        require_parseable_user_config()
    except InvalidUserConfigError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


def main():
    """CLI entry point for the gateway."""
    # Before any config-dependent startup (watchdog, DB opens, provider resolution).
    _guard_corrupt_user_config()

    # Advertise the harness to children (mirrors _advertise_agent_env in hermes_cli/main.py, inlined to
    # avoid its startup side effects). Value must equal registry id ``hermes-agent`` exactly.
    os.environ.setdefault("AI_AGENT", "hermes-agent")
    os.environ.setdefault("HERMES_AGENT", "true")

    def _register_identity() -> None:
        # Ledger registration + Windows job-object attach so update-time reapers can identify this gateway.
        from hermes_cli.process_identity import attach_self_to_kill_on_close_job, register_self
        register_self("gateway")
        attach_self_to_kill_on_close_job()

    def _arm_watchdog() -> None:
        # Armed before config load / DB opens so a pre-loop deadlock is respawned by the supervisor instead
        # of wedging as a live-PID zombie. GatewayRunner disarms it.
        from gateway.startup_watchdog import arm_startup_watchdog
        arm_startup_watchdog()

    def _utf8_stdio() -> None:
        # Windows: gateway logs and banner would UnicodeEncodeError on cp1252 consoles. No-op on POSIX.
        from hermes_cli.stdio import configure_windows_stdio
        configure_windows_stdio()

    for _step in (_register_identity, _arm_watchdog, _utf8_stdio):
        _best_effort(_step)

    import argparse

    parser = argparse.ArgumentParser(description="Hermes Gateway - Multi-platform messaging")
    parser.add_argument("--config", "-c", help="Path to gateway config file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    config = None
    if args.config:
        import yaml
        with open(args.config, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            config = GatewayConfig.from_dict(data)

    # start_gateway() finishes graceful teardown before returning OR raising SystemExit; force-exit after so
    # a wedged non-daemon worker can't block Py_FinalizeEx's thread join. SystemExit is caught so EVERY
    # exit path hits the os._exit backstop.
    try:
        success = asyncio.run(start_gateway(config))
        exit_code = 0 if success else 1
    except SystemExit as e:
        # e.code may be None (→ 0), an int, or a str (→ 1, like CPython).
        if e.code is None:
            exit_code = 0
        elif isinstance(e.code, int):
            exit_code = e.code
        else:
            exit_code = 1
    _exit_after_graceful_shutdown(exit_code)


def _exit_after_graceful_shutdown(exit_code: int) -> None:
    """Flush stdio, release the PID file + runtime lock, then hard-exit.

    ``os._exit`` (not ``sys.exit``): SystemExit runs ``Py_FinalizeEx``, which joins every non-daemon
    thread — exactly the hang a wedged worker causes. It bypasses ``atexit``, so PID/lock release and the
    bounded log drain (file handlers sit behind a ``QueueListener`` thread) are done here explicitly.
    """
    for stream in (sys.stdout, sys.stderr):
        with suppress(Exception):
            stream.flush()
    def _release_locks() -> None:
        # BEFORE the log drain: the drain is bounded but could take its full timeout on a wedged disk,
        # and these locks must never be stranded. Idempotent (early SystemExit paths never ran _stop_impl).
        from gateway.status import remove_pid_file, release_gateway_runtime_lock
        remove_pid_file()
        release_gateway_runtime_lock()

    def _mark_exited() -> None:
        # Single funnel every graceful exit passes through, so the next boot's unclean-death detector
        # fires only for genuine SIGKILL/OOM/VM deaths. Ownership-guarded against an old --replace life.
        from gateway.lifecycle_ledger import mark_exited
        mark_exited(exit_code, reason="graceful_shutdown")

    def _drain_logs() -> None:
        # os._exit bypasses the listener's atexit drain. Bounded, no restart — NOT flush_log_queue():
        # a listener wedged on the rotation lock would re-freeze shutdown in an unbounded stop() join.
        from hermes_logging import drain_log_queue
        drain_log_queue(timeout=1.0)

    for _step in (_release_locks, _mark_exited, _drain_logs):
        _best_effort(_step)
    os._exit(exit_code)


if __name__ == "__main__":
    main()
