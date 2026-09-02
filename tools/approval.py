"""Dangerous command approval -- gate flow, prompting, and per-session state.

Single source of truth for the dangerous command system:
- Pattern detection lives in :mod:`tools.approval_detection` (re-exported here)
- Per-session approval state (thread-safe, keyed by session_key)
- Approval prompting (CLI interactive + gateway async, see
  :mod:`tools.approval_gateway_wait`) and plugin transports
- Smart approval via auxiliary LLM (:mod:`tools.approval_smart`)
- Human-wait accounting (:mod:`tools.approval_human_wait`)
- Permanent allowlist persistence (config.yaml)
"""

import contextvars
import fnmatch
import hashlib
import logging
import os
import re
import sys
import threading
import time
import uuid
from typing import Optional
from hermes_cli.config import cfg_get

from tools.interrupt import is_interrupted
from utils import env_var_enabled, is_truthy_value

logger = logging.getLogger(__name__)

# Frozen at import: reading os.environ per call would let any skill running in
# the process set this and bypass every approval check (prompt-injection
# escalation path).
_YOLO_MODE_FROZEN: bool = is_truthy_value(os.getenv("HERMES_YOLO_MODE", ""))

# Per-thread/per-task gateway session identity. Gateway runs agent turns
# concurrently in executor threads, so a process-global env var is racy; the
# env fallback stays for legacy single-threaded callers.
_approval_session_key: contextvars.ContextVar[str] = contextvars.ContextVar(
    "approval_session_key", default="",
)
_approval_turn_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "approval_turn_id", default="",
)
_approval_tool_call_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "approval_tool_call_id", default="",
)
# Hermes session id (observability identity, distinct from the gateway routing
# session_key). Forwarded to approval hooks so observer plugins attach marks to
# the REAL session scope — without it they fall back to a synthetic "default"
# session whose scope never closes, so close-time exporters never ship them.
_approval_session_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "approval_session_id", default="",
)

# Interactive-CLI flag. Concurrent ACP sessions share a ThreadPoolExecutor, so
# mutating os.environ["HERMES_INTERACTIVE"] races: one session's `finally`
# restore can clobber another's set mid-run, dropping it onto the
# non-interactive auto-approve path so a dangerous command runs without the
# approval callback firing (GHSA-96vc-wcxf-jjff). None = unset → fall back to
# the env var for legacy single-threaded CLI callers.
_hermes_interactive_ctx: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "hermes_interactive", default=None,
)


def set_hermes_interactive_context(interactive: bool) -> contextvars.Token:
    """Bind interactive mode for the current context instead of mutating os.environ."""
    return _hermes_interactive_ctx.set("1" if interactive else "")


def reset_hermes_interactive_context(token: contextvars.Token) -> None:
    """Restore the prior value from :func:`set_hermes_interactive_context`."""
    _hermes_interactive_ctx.reset(token)


def _is_interactive_cli() -> bool:
    """True for an interactive CLI/ACP session (contextvar first, env fallback)."""
    ctx_val = _hermes_interactive_ctx.get()
    if ctx_val is not None:
        return is_truthy_value(ctx_val)
    return env_var_enabled("HERMES_INTERACTIVE")


def _fire_approval_hook(hook_name: str, **kwargs) -> None:
    """Invoke a plugin lifecycle hook (pre_approval_request / post_approval_response).

    Lazy-imports the plugin manager (approval.py is imported long before plugins
    are discovered). Never raises: approval flow is safety-critical, plugin
    observability is not.
    """
    try:
        from hermes_cli.lifecycle import invoke_hook
    except Exception:
        return  # plugin system unavailable (bare tool-only imports, minimal tests)
    try:
        kwargs.setdefault("turn_id", _approval_turn_id.get())
        kwargs.setdefault("tool_call_id", _approval_tool_call_id.get())
        _session_id = _approval_session_id.get()
        if _session_id:
            kwargs.setdefault("session_id", _session_id)
        invoke_hook(hook_name, **kwargs)
    except Exception as exc:
        # invoke_hook() swallows per-callback errors; reaching here means the
        # dispatch layer itself failed.
        logger.debug("Approval hook %s dispatch failed: %s", hook_name, exc)


def _prepare_smart_approval_observer(
    *,
    command: str,
    description: str,
    pattern_key: str,
    pattern_keys: list[str],
    session_key: str,
) -> dict | None:
    """Redact and emit the pre-decision smart approval observer hook.

    Redaction is observer-payload preparation, not approval policy: if it fails,
    skip observability rather than leak raw data or block the LLM decision.
    """
    try:
        from agent.redact import redact_sensitive_text

        hook_command = redact_sensitive_text(command, force=True)
        hook_description = redact_sensitive_text(description, force=True)
    except Exception as exc:
        logger.debug("Smart approval hook redaction failed: %s", exc)
        return None

    payload = {
        "command": hook_command,
        "description": hook_description,
        "pattern_key": pattern_key,
        "pattern_keys": list(pattern_keys),
        "session_key": session_key,
        "surface": "smart",
    }
    _fire_approval_hook("pre_approval_request", **payload)
    return payload


def _observe_smart_approval_verdict(payload: dict | None, verdict: str) -> None:
    """Emit a smart verdict after the auxiliary LLM decision, if safe."""
    if payload is None or verdict not in {"approve", "deny"}:
        return
    _fire_approval_hook(
        "post_approval_response",
        **payload,
        choice=f"smart_{verdict}",
        decided_by="aux_llm",
    )


def set_current_session_key(session_key: str) -> contextvars.Token[str]:
    """Bind the active approval session key to the current context."""
    return _approval_session_key.set(session_key or "")


def reset_current_session_key(token: contextvars.Token[str]) -> None:
    """Restore the prior approval session key context."""
    _approval_session_key.reset(token)


def set_current_observability_context(
    *,
    turn_id: str = "",
    tool_call_id: str = "",
    session_id: str = "",
) -> tuple[
    contextvars.Token[str], contextvars.Token[str], contextvars.Token[str]
]:
    """Bind active tool correlation IDs to approval hooks."""
    return (
        _approval_turn_id.set(turn_id or ""),
        _approval_tool_call_id.set(tool_call_id or ""),
        _approval_session_id.set(session_id or ""),
    )


def reset_current_observability_context(
    tokens: tuple[
        contextvars.Token[str], contextvars.Token[str], contextvars.Token[str]
    ],
) -> None:
    """Restore prior approval hook correlation IDs."""
    turn_token, tool_token, session_token = tokens
    _approval_session_id.reset(session_token)
    _approval_tool_call_id.reset(tool_token)
    _approval_turn_id.reset(turn_token)


def get_current_session_key(default: str = "default") -> str:
    """Return the active session key: approval contextvar → session_context → os.environ."""
    session_key = _approval_session_key.get()
    if session_key:
        return session_key
    from gateway.session_context import get_session_env
    return get_session_env("HERMES_SESSION_KEY", default)


def _session_env_flag(name: str) -> bool:
    """Truthy session-scoped env flag, contextvar-first so one cron/-q job cannot
    taint unrelated gateway/API/TUI turns in the same process; process env is
    the fallback for CLI tests and older entrypoints."""
    try:
        from gateway.session_context import get_session_env

        return is_truthy_value(get_session_env(name, ""))
    except Exception:
        return env_var_enabled(name)


def _get_session_platform() -> str:
    """Return the current gateway platform from contextvars/env fallback."""
    try:
        from gateway.session_context import get_session_env

        return get_session_env("HERMES_SESSION_PLATFORM", "") or ""
    except Exception:
        return os.getenv("HERMES_SESSION_PLATFORM", "") or ""


def _is_cron_approval_context() -> bool:
    """True when the current approval decision is running inside cron."""
    return _session_env_flag("HERMES_CRON_SESSION")


#: Programmatic/unattended platforms: no human can answer a prompt and the
#: adapter has no ``send_exec_approval`` / ``/approve`` surface. Governed by
#: ``approvals.unattended_mode`` (default deny), mirroring ``cron_mode`` —
#: never an interactive round-trip that blocks for the full timeout with
#: nobody to answer (#37284, #87509).
_UNATTENDED_APPROVAL_PLATFORMS = frozenset({
    "webhook",
    "msgraph_webhook",
    "api_server",
})


def _is_unattended_platform_approval_context() -> bool:
    """True when the session platform is a programmatic/unattended surface."""
    return _get_session_platform() in _UNATTENDED_APPROVAL_PLATFORMS


def _is_single_query_approval_context() -> bool:
    """True for a single-query (-q) session.

    ``hermes chat -q`` exports ``HERMES_INTERACTIVE=1`` (so sudo password
    prompts work) but nobody is waiting to answer approvals; without this
    marker the gate would wait the full timeout for a human who never comes,
    then fail closed and push the agent toward workarounds (e.g. execute_code).
    ``approvals.single_query_mode`` makes the path deterministic.
    """
    return _session_env_flag("HERMES_SINGLE_QUERY_SESSION")


def _is_gateway_approval_context() -> bool:
    """True inside a gateway/API session that can answer an approval.

    Legacy integrations set HERMES_GATEWAY_SESSION; concurrent paths bind
    HERMES_SESSION_PLATFORM via contextvars. Cron is NEVER a gateway approval
    context even when it originated from a platform (cron binds the platform
    for delivery routing): falling through would submit a pending approval with
    no listener and block the job indefinitely. Unattended platforms are
    excluded for the same reason (#37284, #87509).
    """
    if _is_cron_approval_context():
        return False
    if _is_unattended_platform_approval_context():
        return False
    if env_var_enabled("HERMES_GATEWAY_SESSION"):
        return True
    return bool(_get_session_platform())


def _resolve_cli_approval_callback(approval_callback=None):
    """Explicit callback, else the per-thread one from ``terminal_tool.set_approval_callback``."""
    if approval_callback is not None:
        return approval_callback
    try:
        from tools.terminal_tool import _get_approval_callback
        return _get_approval_callback()
    except Exception:
        return None


def _should_fall_through_to_cli_approval(
    *,
    is_cli: bool,
    approval_callback,
    notify_cb,
) -> bool:
    """Prefer the CLI Dangerous Command panel over a silent pending approval.

    ``HERMES_EXEC_ASK`` (or a platform marker) can leak into an interactive CLI
    process — historically via ``import gateway.run``. Without a gateway notify
    listener the ask branch used to return ``pending_approval`` immediately and
    skip the panel the user can actually answer.
    """
    return bool(is_cli and approval_callback is not None and notify_cb is None)

from tools.approval_detection import (  # noqa: F401 -- re-exported for callers/tests
    _CREDENTIAL_FILES,
    _SYSTEM_CONFIG_PATH,
    _COMMAND_TAIL,
    HARDLINE_PATTERNS,
    _check_sudo_stdin_guard,
    detect_hardline_command,
    DANGEROUS_PATTERNS,
    _approval_key_aliases,
    _normalize_command_for_detection,
    _rewrite_resolved_user_home,
    _rewrite_resolved_hermes_home,
    _MAX_SEPARATOR_FREE_COMMAND_CHARS,
    _PARSER_LIMIT_DESCRIPTION,
    _MALFORMED_EXEC_DESCRIPTION,
    _bash_exec_payload,
    _read_shell_word,
    _deobfuscate_shell_word_for_detection,
    _iter_shell_command_starts,
    _command_detection_variants,
    detect_dangerous_command,
)
from tools.approval_human_wait import (  # noqa: F401 -- re-exported for callers/tests
    _HumanWaitState,
    _human_wait_lock,
    _human_wait_states,
    _HUMAN_WAIT_MAX_SESSIONS,
    HUMAN_WAIT_MARGIN_S,
    human_wait_ceiling,
    _clamped_window_seconds,
    _human_wait_state,
    human_wait_window,
    human_wait_seconds,
)
from tools.approval_smart import (  # noqa: F401 -- re-exported for callers/tests
    _strip_shell_comments,
    _strip_line_comment,
    _get_smart_policy,
    _smart_approve,
)
from tools.approval_gateway_wait import (  # noqa: F401 -- re-exported for callers/tests
    _ApprovalEntry,
    _await_coalesced_leader,
    _await_gateway_decision,
)


def _match_user_deny_rule(command: str) -> str | None:
    """Return the matching ``approvals.deny`` glob, or None.

    User-defined fnmatch globs that block unconditionally — like the hardline
    floor, a match fires BEFORE the yolo / mode=off bypass ("never let the
    agent run this, even under yolo"). Case-insensitive, run over the same
    normalized/deobfuscated variants the dangerous-pattern detector uses so
    quoting tricks (``r\\m``, ``git st""atus``) can't sidestep a rule.
    """
    try:
        deny_patterns = _get_approval_config().get("deny") or []
    except Exception:
        return None
    globs = [p.strip() for p in deny_patterns
             if isinstance(p, str) and p.strip()]
    if not globs:
        return None
    for command_variant in _command_detection_variants(command):
        candidate = command_variant.lower().strip()
        for pattern in globs:
            if fnmatch.fnmatchcase(candidate, pattern.lower()):
                return pattern
    return None


def _user_deny_block_result(pattern: str) -> dict:
    """Build the standard block result for an ``approvals.deny`` match."""
    return {
        "approved": False,
        "user_deny": True,
        "message": (
            f"BLOCKED: this command matches the user-defined deny rule "
            f"'{pattern}' (approvals.deny in config.yaml). It cannot be "
            "executed via the agent — not even with --yolo, /yolo, or "
            "approvals.mode=off. Do NOT retry or rephrase this command; "
            "the user has explicitly forbidden it."
        ),
    }


def _save_blocked_payload(command: str) -> Optional[str]:
    """Persist a parser-limit-blocked command as a runnable script.

    The parser-limit block fires on payload SIZE/shape, not the operation —
    usually a legitimate script the model inlined. Saving it makes recovery one
    turn (`bash <file>`) instead of two, and is strictly safer than the
    hint-only path: the file goes through the normal execution pipeline
    (including the referenced-script content guard) and nothing runs here.
    Returns the path, or None on any failure (hint falls back to write_file).
    """
    try:
        from hermes_constants import get_hermes_home
        script_dir = get_hermes_home() / "cache" / "blocked-scripts"
        script_dir.mkdir(parents=True, exist_ok=True)
        # Opportunistic cleanup: blocked payloads older than 7 days.
        cutoff = time.time() - 7 * 86400
        for old in script_dir.glob("blocked-*.sh"):
            try:
                if old.stat().st_mtime < cutoff:
                    old.unlink()
            except OSError:
                pass
        path = script_dir / f"blocked-{int(time.time())}-{uuid.uuid4().hex[:8]}.sh"
        path.write_text(
            "#!/bin/bash\n"
            "# Auto-saved by Hermes: this command exceeded the inline command\n"
            "# parser limit and was blocked from direct execution. Review it,\n"
            "# then run it via: bash " + str(path) + "\n"
            + command
            + ("\n" if not command.endswith("\n") else ""),
            encoding="utf-8", errors="replace",
        )
        return str(path)
    except Exception:
        logger.debug("failed to save blocked payload", exc_info=True)
        return None


def _hardline_block_result(description: str, command: str = "") -> dict:
    """Build the standard block result for a hardline match."""
    message = (
        f"BLOCKED (hardline): {description}. "
        "This command is on the unconditional blocklist and cannot "
        "be executed via the agent — not even with --yolo, /yolo, "
        "approvals.mode=off, or cron approve mode. If you genuinely "
        "need to run it, run it yourself in a terminal outside the "
        "agent."
    )
    # The parser-limit block is almost always a giant inline payload, not a
    # forbidden operation, and is typically followed by blind rephrase
    # retries — point at the saved script (or the write_file recipe).
    if description in (_PARSER_LIMIT_DESCRIPTION, _MALFORMED_EXEC_DESCRIPTION):
        saved = _save_blocked_payload(command) if command else None
        if saved:
            message += (
                " RECOVERY: this block fires on oversized/unparseable inline "
                "command payloads (heredocs, giant one-liners), not on the "
                f"operation itself. Your command was saved to {saved} — "
                f"review it, then run: terminal(command=\"bash {saved}\"). "
                "Do not retry inline."
            )
        else:
            message += (
                " RECOVERY: this block fires on oversized/unparseable inline "
                "command payloads (heredocs, giant one-liners), not on the "
                "operation itself. Write the script to a file with write_file, "
                "then run it: terminal(command=\"bash /path/script.sh\") or "
                "\"python3 /path/script.py\". Do not retry inline."
            )
    return {
        "approved": False,
        "hardline": True,
        "message": message,
    }


def _sudo_stdin_block_result(description: str) -> dict:
    """Build the standard block result for sudo stdin guard."""
    return {
        "approved": False,
        "message": (
            f"BLOCKED: {description}. "
            "Do not pipe passwords to 'sudo -S' — this is a brute-force "
            "attack vector. Set SUDO_PASSWORD in your .env file if the "
            "agent needs passwordless sudo, or run the sudo command "
            "manually in your own terminal."
        ),
    }


# =========================================================================
# Per-session approval state (thread-safe)
# =========================================================================

_lock = threading.Lock()
_pending: dict[str, dict] = {}
_session_approved: dict[str, set] = {}
_session_yolo: set[str] = set()
_permanent_approved: set = set()

# =========================================================================
# Consecutive-denial circuit breaker for smart approvals
# =========================================================================
# Nothing stops the model from retrying variants of a smart-denied command —
# each retry burns another guardian LLM call. After
# ``approvals.denial_breaker_threshold`` consecutive guardian DENY verdicts in
# one session (default 3; 0 disables) the deny message escalates to a
# hard-stop instruction; any approval resets the tally. Only TOOL RESULT text
# changes — no history surgery, no interrupts — so it is prompt-cache-invariant.
_denial_tally: dict[str, int] = {}
# Small cap so an army of short-lived session keys cannot grow it without
# bound; oldest (least recently denied) entries are evicted.
_DENIAL_TALLY_MAX_SESSIONS = 256


def _get_denial_breaker_threshold() -> int:
    """``approvals.denial_breaker_threshold``: default 3; 0 or negative disables."""
    try:
        return int(_get_approval_config().get("denial_breaker_threshold", 3))
    except (ValueError, TypeError):
        return 3


def _record_denial(session_key: str) -> int:
    """Increment and return the session's consecutive guardian-denial count.

    Pop-and-reinsert keeps actively-denying sessions at the most-recent end
    so insertion-ordered eviction drops genuinely idle keys.
    """
    with _lock:
        count = _denial_tally.pop(session_key, 0) + 1
        _denial_tally[session_key] = count
        while len(_denial_tally) > _DENIAL_TALLY_MAX_SESSIONS:
            _denial_tally.pop(next(iter(_denial_tally)))
        return count


def _reset_denials(session_key: str) -> None:
    """Clear the session's consecutive-denial tally (an approval happened)."""
    with _lock:
        _denial_tally.pop(session_key, None)


def _denial_breaker_addendum(session_key: str) -> str:
    """Escalated hard-stop text once the breaker has tripped, else ''.

    Read-only: callers increment via :func:`_record_denial` on the guardian
    DENY verdict. The result is appended verbatim to the deny message.
    """
    with _lock:
        count = _denial_tally.get(session_key, 0)
    threshold = _get_denial_breaker_threshold()
    if threshold <= 0 or count < threshold:
        return ""
    logger.warning(
        "Smart-approval circuit breaker tripped for session %s: "
        "%d consecutive denials (threshold %d)",
        session_key, count, threshold,
    )
    return (
        f" CIRCUIT BREAKER: {count} consecutive commands were blocked by "
        "the security reviewer. STOP attempting variations of this "
        "operation. Report the blocked operation to the user and either "
        "ask them to run it manually or use /approve."
    )

# =========================================================================
# Gateway approval queue (the blocking wait loop lives in approval_gateway_wait)
# =========================================================================

_gateway_queues: dict[str, list] = {}        # session_key → [_ApprovalEntry, …]
_gateway_notify_cbs: dict[str, object] = {}  # session_key → callable(approval_data)


def register_gateway_notify(session_key: str, cb) -> None:
    """Register ``cb(approval_data: dict) -> None`` for sending approval requests.

    The callback bridges sync→async: it runs in the agent thread and must
    schedule the actual send on the event loop.
    """
    with _lock:
        _gateway_notify_cbs[session_key] = cb


def unregister_gateway_notify(session_key: str) -> None:
    """Unregister the callback and wake ALL blocked threads for this session so
    they don't hang forever (agent run finished or interrupted)."""
    with _lock:
        _gateway_notify_cbs.pop(session_key, None)
        entries = _gateway_queues.pop(session_key, [])
    for entry in entries:
        entry.event.set()


def resolve_gateway_approval(session_key: str, choice: str,
                             resolve_all: bool = False,
                             reason: Optional[str] = None,
                             request_id: Optional[str] = None) -> int:
    """Unblock waiting agent thread(s) from the gateway's /approve or /deny handler.

    *resolve_all* resolves every pending approval (``/approve all``); otherwise
    the oldest (FIFO) or the one matching *request_id*. *reason* is the free
    text from ``/deny <reason>``, relayed to the agent in the BLOCKED message.
    Returns the number resolved (0 = nothing pending).
    """
    with _lock:
        queue = _gateway_queues.get(session_key)
        if not queue:
            return 0
        if request_id:
            targets = [entry for entry in queue if entry.data.get("request_id") == request_id]
            if not targets:
                return 0
            queue[:] = [entry for entry in queue if entry not in targets]
        elif resolve_all:
            targets = list(queue)
            queue.clear()
        else:
            targets = [queue.pop(0)]
        if not queue:
            _gateway_queues.pop(session_key, None)

    for entry in targets:
        entry.result = choice
        if reason:
            entry.reason = reason
        entry.event.set()
    return len(targets)


def list_gateway_approvals(session_key: str) -> list[dict]:
    """Return replay-safe snapshots of unresolved approvals for one session."""
    with _lock:
        return [dict(entry.data) for entry in _gateway_queues.get(session_key, [])]


def ack_gateway_approval(session_key: str, request_id: str) -> bool:
    """Record that a client received a particular pending approval request."""
    with _lock:
        for entry in _gateway_queues.get(session_key, []):
            if entry.data.get("request_id") == request_id:
                entry.acknowledged = True
                return True
    return False


def has_blocking_approval(session_key: str) -> bool:
    """Check if a session has one or more blocking gateway approvals waiting."""
    with _lock:
        return bool(_gateway_queues.get(session_key))


def get_pending_gateway_approval(session_key: str) -> dict | None:
    """Copy of the oldest unresolved gateway approval, for reconnecting clients
    to restore a prompt. Read-only snapshot — the queue stays authoritative."""
    if not session_key:
        return None
    with _lock:
        queue = _gateway_queues.get(session_key)
        if not queue:
            return None
        return dict(queue[0].data)


def submit_pending(session_key: str, approval: dict):
    """Store a pending approval request for a session."""
    with _lock:
        _pending[session_key] = approval


def approve_session(session_key: str, pattern_key: str):
    """Approve a pattern for this session only."""
    with _lock:
        _session_approved.setdefault(session_key, set()).add(pattern_key)


def _release_permission_mode_dependents(session_key: str) -> None:
    """Drop resources whose immutable mode derives from Hermes YOLO.

    Lazy import so approval-only sessions never load computer-use. Releasing on
    both edges makes enabling YOLO replace a standard backend and disabling it
    revoke a private unrestricted daemon immediately.
    """
    try:
        from tools.computer_use import release_computer_use_session

        release_computer_use_session(session_key)
    except Exception:
        logger.debug(
            "Failed to release permission-mode dependent resources for %s",
            session_key,
            exc_info=True,
        )


def enable_session_yolo(session_key: str) -> None:
    """Enable YOLO bypass for a single session key."""
    if not session_key:
        return
    with _lock:
        _session_yolo.add(session_key)
    _release_permission_mode_dependents(session_key)


def disable_session_yolo(session_key: str) -> None:
    """Disable YOLO bypass for a single session key."""
    if not session_key:
        return
    with _lock:
        _session_yolo.discard(session_key)
    _release_permission_mode_dependents(session_key)


def clear_session(session_key: str) -> None:
    """Remove all approval and yolo state for a given session."""
    if not session_key:
        return
    with _lock:
        _session_approved.pop(session_key, None)
        _session_yolo.discard(session_key)
        _pending.pop(session_key, None)
        entries = _gateway_queues.pop(session_key, [])
    for entry in entries:
        # Cancel blocked waits now so the old run unwinds instead of idling
        # until timeout.
        entry.result = "deny"
        entry.event.set()
    _release_permission_mode_dependents(session_key)
    # Session-persistent code kernels (local and remote) share this owner key
    # and die at the same boundary so a finished conversation cannot leak a
    # live interpreter.
    try:
        from tools.code_kernel import shutdown_kernels_for_owner

        shutdown_kernels_for_owner(session_key)
    except Exception:
        pass
    try:
        from tools.code_kernel_remote import shutdown_remote_kernels_for_owner

        shutdown_remote_kernels_for_owner(session_key)
    except Exception:
        pass


def is_session_yolo_enabled(session_key: str) -> bool:
    """Return True when YOLO bypass is enabled for a specific session."""
    if not session_key:
        return False
    with _lock:
        return session_key in _session_yolo


def is_current_session_yolo_enabled() -> bool:
    """Return True when the active approval session has YOLO bypass enabled."""
    return is_session_yolo_enabled(get_current_session_key(default=""))


def is_approved(session_key: str, pattern_key: str) -> bool:
    """Check if a pattern is approved (session-scoped or permanent).

    Accepts the canonical key and the legacy regex-derived key so existing
    command_allowlist entries keep working after key migrations.
    """
    aliases = _approval_key_aliases(pattern_key)
    with _lock:
        if any(alias in _permanent_approved for alias in aliases):
            return True
        session_approvals = _session_approved.get(session_key, set())
        return any(alias in session_approvals for alias in aliases)


def approve_permanent(pattern_key: str):
    """Add a pattern to the permanent allowlist."""
    with _lock:
        _permanent_approved.add(pattern_key)


def load_permanent(patterns: set):
    """Bulk-load permanent allowlist entries from config."""
    with _lock:
        _permanent_approved.update(patterns)


def _persist_choice(session_key: str, choice: str, warnings: list[tuple]) -> None:
    """Persist a human ``session``/``always`` choice for each ``(key, _, is_tirith)``.

    Tirith findings are session-max by design: no broad permanent allowlisting
    of content-level security findings, so ``always`` downgrades them to session.
    ``once`` (or any other choice) persists nothing.
    """
    for key, _, is_tirith in warnings:
        if choice == "session" or (choice == "always" and is_tirith):
            approve_session(session_key, key)
        elif choice == "always":
            approve_session(session_key, key)
            approve_permanent(key)
            save_permanent_allowlist(_permanent_approved)


# Shell control characters that make a command compound when they appear
# OUTSIDE quotes. Inside quotes they are literal to the outer shell — but they
# become executable again if an option like `-c`/`-e`/`--eval` (or a git
# `-c alias.x=!...`) hands the quoted argument to another interpreter, so quoted
# control chars only disqualify a command when such an option is present.
_SHELL_CONTROL_CHARS = frozenset("\n\r;&|<>`$()")
_REINTERPRETED_ARGUMENT_RE = re.compile(
    r"(?:^|[ \t])(?:-[^-\s]*[ce]|--(?:command|eval))(?:[= \t]|$)"
)


def _has_allowlist_shell_operator(command: str) -> bool:
    """Return True when a command is too compound for the allowlist shortcut.

    Quote-aware: metacharacters inside quotes or behind a backslash are literal
    arguments (``cargo bench -- '^a(b|c)$'``), not shell syntax. Still
    disqualifying: ``$`` or backtick inside DOUBLE quotes (expansion stays
    active), and any quoted/escaped control character when the command also
    carries a ``-c``/``-e``/``--command``/``--eval``-style option that would
    hand the quoted text to another interpreter.
    """
    command = command or ""
    quote = None  # None | "'" | '"'
    has_reinterpretable = False
    i = 0
    n = len(command)
    while i < n:
        ch = command[i]
        if quote == "'":
            if ch == "'":
                quote = None
            elif ch in _SHELL_CONTROL_CHARS:
                has_reinterpretable = True
            i += 1
            continue
        if ch == "\\":
            nxt = command[i + 1] if i + 1 < n else ""
            if nxt in _SHELL_CONTROL_CHARS:
                has_reinterpretable = True
            i += 2
            continue
        if quote == '"':
            if ch == '"':
                quote = None
            elif ch in ("`", "$"):
                return True  # expansion is active inside double quotes
            elif ch in _SHELL_CONTROL_CHARS:
                has_reinterpretable = True
            i += 1
            continue
        if ch in ("'", '"'):
            quote = ch
            i += 1
            continue
        if ch == "$":
            # Unquoted $ is only compound when it opens a substitution
            # ("$HOME" stays simple, matching the historical `\$\(` behavior).
            if i + 1 < n and command[i + 1] == "(":
                return True
            i += 1
            continue
        if ch in _SHELL_CONTROL_CHARS and ch not in "()":
            return True
        i += 1
    # An unterminated quote means we can't reason about the command shape.
    if quote is not None:
        return True
    return has_reinterpretable and bool(_REINTERPRETED_ARGUMENT_RE.search(command))


def _command_matches_permanent_allowlist(command: str) -> bool:
    """True when command_allowlist holds this exact command text or a matching glob.

    Permanent approvals historically store dangerous-pattern keys such as
    ``recursive delete``; manual entries are command text, possibly with
    shell-style wildcards like ``podman *``.
    """
    command = (command or "").strip()
    if not command or _has_allowlist_shell_operator(command):
        return False

    with _lock:
        patterns = tuple(_permanent_approved)

    for pattern in patterns:
        if not isinstance(pattern, str):
            continue
        pattern = pattern.strip()
        if not pattern:
            continue
        if command == pattern:
            return True
        if any(ch in pattern for ch in "*?[") and fnmatch.fnmatchcase(command, pattern):
            return True
    return False


# =========================================================================
# Config persistence for permanent allowlist
# =========================================================================

def load_permanent_allowlist() -> set:
    """Load ``command_allowlist`` from config and sync it into the approval state
    so is_approved() honors 'always' choices from previous sessions."""
    try:
        from hermes_cli.config import load_config_readonly
        config = load_config_readonly()
        patterns = set(config.get("command_allowlist", []) or [])
        if patterns:
            load_permanent(patterns)
        return patterns
    except Exception as e:
        logger.warning("Failed to load permanent allowlist: %s", e)
        return set()


def save_permanent_allowlist(patterns: set):
    """Save permanently allowed command patterns to config."""
    try:
        from hermes_cli.config import load_config, save_config
        config = load_config()
        config["command_allowlist"] = list(patterns)
        save_config(config)
    except Exception as e:
        logger.warning("Could not save allowlist: %s", e)


# =========================================================================
# Approval prompting (CLI)
# =========================================================================

def prompt_dangerous_approval(command: str, description: str,
                              timeout_seconds: int | None = None,
                              allow_permanent: bool = True,
                              approval_callback=None,
                              *, allow_session: bool = True,
                              smart_denied: bool = False) -> str:
    """Prompt the user to approve a dangerous command (CLI only).

    Args:
        allow_permanent: When False, hide [a]lways (tirith warnings present:
            broad permanent allowlisting is wrong for content-level findings).
        allow_session: When False, hide [s]ession too — the caller grants one
            operation and re-asks next time (the protected agent-instruction
            gate in ``tools/file_tools.py``). Offering a scope the caller
            discards makes every later write re-prompt and reads as broken.
        smart_denied: Owner override of a Smart DENY: offer only once/deny.
        approval_callback: CLI prompt_toolkit callback,
            ``(command, description, *, allow_permanent=True,
            allow_session=True, smart_denied=False) -> str``. Legacy
            signatures keep working while both keywords hold their defaults.

    Returns: 'once', 'session', 'always', 'deny', or 'timeout'. 'timeout'
        means no user response — still blocked (fail-closed), but callers
        report "no response" rather than an explicit denial.
    """
    if timeout_seconds is None:
        timeout_seconds = _get_approval_timeout()

    # Everything below is a human prompt (callback panel or input() fallback,
    # both bounded by the approval deadline): record it as human-wait time so
    # the concurrent batch deadline excludes it (#79719).
    with human_wait_window():
        return _prompt_dangerous_approval_inner(
            command,
            description,
            timeout_seconds,
            allow_permanent,
            approval_callback,
            allow_session=allow_session,
            smart_denied=smart_denied,
        )


_CLI_CHOICE_ALIASES = {
    "o": "once", "once": "once",
    "s": "session", "session": "session",
    "a": "always", "always": "always",
}
_CLI_CHOICE_I18N = {
    "once": "approval.allowed_once",
    "session": "approval.allowed_session",
    "always": "approval.allowed_always",
    "deny": "approval.denied",
}


def _prompt_dangerous_approval_inner(command: str, description: str,
                                     timeout_seconds: int,
                                     allow_permanent: bool = True,
                                     approval_callback=None,
                                     *, allow_session: bool = True,
                                     smart_denied: bool = False) -> str:
    # Redact before any user-visible rendering; the original `command` is
    # still what executes after approval. Same redactor as memory/log
    # sanitization so tokens mask consistently across surfaces.
    from agent.redact import redact_sensitive_text
    display_command = redact_sensitive_text(command)
    display_description = redact_sensitive_text(description)

    # Smart DENY and a session-less gate both reduce the menu to once/deny.
    once_only = smart_denied or not allow_session

    if approval_callback is not None:
        try:
            callback_kwargs = {"allow_permanent": allow_permanent}
            if not allow_session:
                callback_kwargs["allow_session"] = False
            if smart_denied:
                callback_kwargs["smart_denied"] = True
            return approval_callback(
                display_command, display_description, **callback_kwargs
            )
        except Exception as e:
            logger.error("Approval callback failed: %s", e, exc_info=True)
            return "deny"

    # Fail-closed guard: when prompt_toolkit owns the terminal and no callback
    # is registered on this thread, the input() fallback would spawn a daemon
    # thread whose read never sees Enter (keystrokes go to prompt_toolkit) —
    # an invisible deadlock (#15216). Deny loudly instead; threads needing
    # interactive approval must install a callback via
    # tools.terminal_tool.set_approval_callback() first.
    try:
        from prompt_toolkit.application.current import get_app_or_none
        if get_app_or_none() is not None:
            logger.warning(
                "Dangerous-command approval requested on a thread with no "
                "approval callback while prompt_toolkit is active; denying "
                "to avoid stdin deadlock. command=%r description=%r",
                command, description,
            )
            return "deny"
    except Exception:
        pass  # prompt_toolkit absent or detection failed: legacy input() path is safe

    os.environ["HERMES_SPINNER_PAUSE"] = "1"
    try:
        from agent.i18n import t
        if once_only:
            prompt = t("approval.prompt_smart_deny")
        else:
            prompt = t("approval.prompt_long") if allow_permanent else t("approval.prompt_short")
        while True:
            print()
            print(f"  {t('approval.dangerous_header', description=display_description)}")
            print(f"      {display_command}")
            print()
            if once_only:
                print(t("approval.choose_smart_deny"))
            elif allow_permanent:
                print(t("approval.choose_long"))
            else:
                print(t("approval.choose_short"))
            print()
            sys.stdout.flush()

            result = {"choice": ""}

            def get_input():
                try:
                    result["choice"] = input(prompt).strip().lower()
                except (EOFError, OSError):
                    result["choice"] = ""

            thread = threading.Thread(target=get_input, daemon=True)
            thread.start()
            thread.join(timeout=timeout_seconds)

            if thread.is_alive():
                print("\n" + t("approval.timeout"))
                return "timeout"  # distinct from deny: the user never answered

            choice = result["choice"]
            if once_only:
                choice_map = {
                    **{
                        value: "once"
                        for value in t("approval.smart_deny_once_inputs").split(",")
                    },
                    **{
                        value: "deny"
                        for value in t("approval.smart_deny_deny_inputs").split(",")
                    },
                }
                decision = choice_map.get(choice, "deny")
            else:
                decision = _CLI_CHOICE_ALIASES.get(choice, "deny")
                if decision == "always" and not allow_permanent:
                    decision = "session"
            print(t(_CLI_CHOICE_I18N[decision]))
            return decision

    except (EOFError, KeyboardInterrupt):
        print("\n" + t("approval.cancelled"))
        return "deny"
    finally:
        if "HERMES_SPINNER_PAUSE" in os.environ:
            del os.environ["HERMES_SPINNER_PAUSE"]
        print()
        sys.stdout.flush()


# =========================================================================
# Config readers
# =========================================================================

def _normalize_approval_mode(mode) -> str:
    """Normalize approval mode values loaded from YAML/config.

    YAML 1.1 parses a bare ``off`` as False, so ``mode: off`` arrives as a bool;
    treat it as the intended string mode. Unknown strings (e.g. 'auto') warn and
    fall back to 'manual' instead of silently failing every mode check.
    """
    _VALID_MODES = ("manual", "smart", "off")
    if isinstance(mode, bool):
        return "off" if mode is False else "manual"
    if isinstance(mode, str):
        normalized = mode.strip().lower()
        if not normalized:
            return "manual"
        if normalized in _VALID_MODES:
            return normalized
        logger.warning(
            "Unknown approvals.mode %r — defaulting to 'manual'. "
            "Valid values: %s",
            mode,
            ", ".join(_VALID_MODES),
        )
        return "manual"
    return "manual"


def _get_approval_config() -> dict:
    """Read the approvals config block.

    Returns the LIVE config-cache sub-dict (load_config_readonly contract) —
    callers must not mutate it or any nested structure.
    """
    try:
        from hermes_cli.config import load_config_readonly
        config = load_config_readonly()
        return config.get("approvals", {}) or {}
    except Exception as e:
        logger.warning("Failed to load approval config: %s", e)
        return {}


def _get_approval_mode() -> str:
    """Return 'manual', 'smart', or 'off' (a hosted-room policy overrides config)."""
    try:
        from gateway.hosted_room_execution_policy import (
            current_room_execution_policy,
        )

        room_policy = current_room_execution_policy()
        if room_policy is not None:
            return room_policy.approval_mode
    except Exception:
        pass
    mode = _get_approval_config().get("mode", "manual")
    return _normalize_approval_mode(mode)


def is_approval_bypass_active_for_session(session_key: str) -> bool:
    """Canonical three-source bypass check: process ``--yolo`` (frozen at
    import), the session-scoped gateway ``/yolo`` toggle, ``approvals.mode: off``.

    Pure bypass sub-expression only — hardline blocklist / permanent allowlist
    are the caller's job.
    """
    return (
        _YOLO_MODE_FROZEN
        or is_session_yolo_enabled(session_key)
        or _get_approval_mode() == "off"
    )


def is_approval_bypass_active() -> bool:
    """Return whether the current approval context has bypass enabled."""
    return is_approval_bypass_active_for_session(
        get_current_session_key(default="")
    )


def _get_approval_timeout() -> int:
    """Read ``approvals.timeout`` (default 300s: gateway push notifications may
    not be seen for minutes; 60s failed closed before Telegram taps landed).

    Clamped to ``agent.deadline.MAX_SAFE_TIMEOUT_S`` (~1 year): a larger value
    overflows ``time_t`` inside ``Thread.join`` / ``Lock.acquire`` on macOS and
    crashed every parallel tool batch (#83220). Clamping at the single
    config-read site keeps every consumer platform-safe at once.
    """
    try:
        raw = int(_get_approval_config().get("timeout", 300))
    except (ValueError, TypeError):
        return 300
    try:
        from agent.deadline import MAX_SAFE_TIMEOUT_S

        safe_cap = int(MAX_SAFE_TIMEOUT_S)
    except Exception:
        # Fail CLOSED: the raw value would re-open the overflow this prevents.
        safe_cap = 365 * 24 * 3600
    if raw > safe_cap:
        logger.warning(
            "approvals.timeout=%s exceeds the platform-safe maximum; "
            "clamping to %ss",
            raw,
            safe_cap,
        )
        return safe_cap
    return raw


def _binary_approval_mode(key: str) -> str:
    """Read ``approvals.<key>`` as 'approve' or 'deny' (default deny)."""
    try:
        from hermes_cli.config import load_config_readonly
        config = load_config_readonly()
        mode = str(cfg_get(config, "approvals", key, default="deny")).lower().strip()
        if mode in {"approve", "off", "allow", "yes"}:
            return "approve"
        return "deny"
    except Exception:
        return "deny"


def _get_cron_approval_mode() -> str:
    """Read the cron approval mode from config. Returns 'deny' or 'approve'."""
    return _binary_approval_mode("cron_mode")


def _get_single_query_approval_mode() -> str:
    """Read the single-query (-q) approval mode from config. Returns 'deny' or 'approve'."""
    return _binary_approval_mode("single_query_mode")


def _get_unattended_approval_mode() -> str:
    """Approval mode for webhook / msgraph_webhook / api_server sessions; default
    deny — an unattended session never silently runs a flagged action unless the
    operator explicitly trusts it."""
    return _binary_approval_mode("unattended_mode")


def _tirith_fail_open() -> bool:
    """``security.tirith_fail_open`` (default True; True when config is unreadable).

    False means the operator opted into fail-closed: an un-importable scanner
    must not silently grant access (#20733).
    """
    try:
        from hermes_cli.config import load_config_readonly as _load_cfg
        _sec = (_load_cfg() or {}).get("security", {}) or {}
        if _sec.get("tirith_enabled", True):
            return bool(_sec.get("tirith_fail_open", True))
    except Exception:
        pass
    return True


# =========================================================================
# Result builders and decision helpers shared by the gates
# =========================================================================

_APPROVED = {"approved": True, "message": None}


def _approved() -> dict:
    return dict(_APPROVED)


def _denied(message: str, *, pattern_key: str, description: str,
            outcome: str, **extra) -> dict:
    """Standard non-consent result: the agent must not retry or rephrase."""
    result = {
        "approved": False,
        "message": message,
        "pattern_key": pattern_key,
        "description": description,
        "outcome": outcome,
        "user_consent": False,
    }
    result.update(extra)
    return result


def _blocked(message: str, *, pattern_key: str, description: str) -> dict:
    """Non-interactive block (cron / -q / unattended / no-human): no consent keys."""
    return {
        "approved": False,
        "message": message,
        "pattern_key": pattern_key,
        "description": description,
    }


def _user_approved(session_key: str, description: str) -> dict:
    """A human approval (incl. ESCALATE-then-approve or a smart-DENY owner
    override) resets the consecutive-denial tally."""
    _reset_denials(session_key)
    return {"approved": True, "message": None,
            "user_approved": True, "description": description}


def _gateway_refusal(decision: dict):
    """``(reason, reason_addendum, timeout_addendum, outcome, deny_reason)`` when
    the gateway decision is not an approval, else None.

    Consent contract: silence is NOT consent, and an explicit deny is a hard
    halt — both produce a BLOCKED outcome. ``/deny <reason>`` free text is
    relayed verbatim so the agent can adapt rather than only hearing "denied".
    """
    resolved, choice = decision["resolved"], decision["choice"]
    deny_reason = decision.get("reason")
    if resolved and choice is not None and choice != "deny":
        return None
    if not resolved:
        return ("timed out without user response", "", " Silence is not consent.",
                "timeout", deny_reason)
    reason_addendum = f' Reason given by the user: "{deny_reason}".' if deny_reason else ""
    return ("denied by user", reason_addendum, "", "denied", deny_reason)


def _gateway_notify_cb(session_key: str):
    with _lock:
        return _gateway_notify_cbs.get(session_key)


def _gateway_approval_data(display_command: str, display_description: str,
                           pattern_key: str, pattern_keys: list[str], *,
                           allow_permanent: bool, smart_denied: bool) -> dict:
    """Payload the gateway renders to Discord/Slack/etc. (screenshottable — the
    caller passes REDACTED copies; the raw command still executes after
    approval and persistence keys off pattern_key, so redaction is display-only).

    Smart DENY overrides are one-operation decisions, so the UI must not offer
    a permanent scope. Session approval is safe for every non-Smart-DENY prompt
    — including pure-tirith ones, where the persistence layer already caps
    scope at session; adapters render the session tier independently.
    """
    data = {
        "command": display_command,
        "pattern_key": pattern_key,
        "pattern_keys": pattern_keys,
        "description": display_description,
        "allow_permanent": allow_permanent and not smart_denied,
        "allow_session": not smart_denied,
    }
    if smart_denied:
        data["smart_denied"] = True
    return data


def _pending_result(session_key: str, *, display_command: str,
                    display_description: str, pattern_key: str,
                    pattern_keys: list[str], body: str, noun: str,
                    smart_denied: bool) -> dict:
    """Queue a pending approval (no gateway notifier registered) and return the
    backward-compatible ``pending_approval`` result."""
    pending_data = {
        "command": display_command,
        "pattern_key": pattern_key,
        "pattern_keys": pattern_keys,
        "description": display_description,
    }
    if smart_denied:
        pending_data.update(smart_denied=True, allow_permanent=False)
    submit_pending(session_key, pending_data)
    result = {
        "approved": False,
        "pattern_key": pattern_key,
        "status": "pending_approval",
        "approval_pending": True,
        "command": display_command,
        "description": display_description,
        "message": (
            f"⚠️ {display_description}. Asking the user for approval.\n\n{body}\n\n"
            f"STOP: do NOT re-run, rephrase, or re-issue this {noun} — each "
            "variant sends the user ANOTHER approval card. Wait for the "
            "user's decision; if this turn must end, report that approval "
            "is pending."
        ),
    }
    if smart_denied:
        result.update(smart_denied=True, allow_permanent=False)
    return result


def _prompt_cli_with_hooks(command: str, description: str, pattern_key: str,
                           pattern_keys: list[str], session_key: str,
                           **prompt_kwargs) -> str:
    """CLI prompt wrapped in the pre/post approval plugin hooks."""
    hook_kwargs = dict(
        command=command, description=description, pattern_key=pattern_key,
        pattern_keys=list(pattern_keys), session_key=session_key, surface="cli",
    )
    _fire_approval_hook("pre_approval_request", **hook_kwargs)
    choice = prompt_dangerous_approval(command, description, **prompt_kwargs)
    _fire_approval_hook("post_approval_response", **hook_kwargs, choice=choice)
    return choice


def _smart_verdict(command: str, description: str, pattern_key: str,
                   pattern_keys: list[str], session_key: str) -> str:
    """Run the guardian LLM with observer hooks; 'approve' | 'deny' | 'escalate'."""
    observer_payload = _prepare_smart_approval_observer(
        command=command,
        description=description,
        pattern_key=pattern_key,
        pattern_keys=pattern_keys,
        session_key=session_key,
    )
    verdict = _smart_approve(command, description)
    _observe_smart_approval_verdict(observer_payload, verdict)
    return verdict


def _run_approval_gate(
    *,
    pattern_key: str,
    description: str,
    display_target: str,
    approval_callback=None,
    cron_deny_message: str,
    single_query_deny_message: str,
    unattended_deny_message: str = "",
    autoapprove_log_prefix: str,
    fail_closed_when_no_human: bool = False,
    no_human_block_message: str = "",
) -> dict:
    """Shared human-approval gate for a flagged action (command or tool).

    Single decision core for :func:`check_dangerous_command` and
    :func:`request_tool_approval`, so the fail-closed / cron / gateway /
    persist policy cannot drift between them. Order: yolo bypass →
    session-cache short-circuit → interactive/gateway/cron branch → prompt →
    persistence. Input-shape-specific checks (hardline, command allowlist,
    pattern detection) are the caller's job BEFORE this gate.

    ``fail_closed_when_no_human``: a non-interactive, non-gateway, non-cron
    context BLOCKS instead of auto-approving. The dangerous-command path keeps
    its historical fail-open default; plugin escalation opts in so a
    plugin-flagged action never runs ungated without a human.
    """
    # Hardline blocks are handled by the caller BEFORE this gate, so yolo here
    # only skips the recoverable approval layer.
    if _YOLO_MODE_FROZEN or is_current_session_yolo_enabled():
        return _approved()

    session_key = get_current_session_key()
    if is_approved(session_key, pattern_key):
        return _approved()

    approval_callback = _resolve_cli_approval_callback(approval_callback)

    is_cli = _is_interactive_cli()
    is_gateway = _is_gateway_approval_context()

    # Single-query (-q) exports HERMES_INTERACTIVE=1 but nobody answers
    # prompts: treat as deterministic non-interactive under single_query_mode.
    if _is_single_query_approval_context():
        is_cli = False
        is_gateway = False

    if not is_cli and not is_gateway:
        if _is_single_query_approval_context():
            if _get_single_query_approval_mode() == "deny":
                return _blocked(single_query_deny_message,
                                pattern_key=pattern_key, description=description)
            # Must return here rather than fall through: the fail_closed
            # branch below would otherwise block the very action
            # single_query_mode: approve just authorized.
            logger.warning(
                "%s (pattern: %s): %s — single-query auto-approve "
                "(approvals.single_query_mode: approve).",
                autoapprove_log_prefix, pattern_key, description,
            )
            return _approved()
        if _is_cron_approval_context():
            if _get_cron_approval_mode() == "deny":
                return _blocked(cron_deny_message,
                                pattern_key=pattern_key, description=description)
            # cron_mode: approve — fall through to auto-approve below.
        elif _is_unattended_platform_approval_context():
            # Resolves instantly — never a pending approval nobody can answer.
            if _get_unattended_approval_mode() == "deny":
                return _blocked(
                    unattended_deny_message or (
                        f"BLOCKED: approval required ({description}) but this "
                        "session runs on an unattended platform "
                        f"({_get_session_platform()}) with no user present to "
                        "approve it. Find an alternative approach that avoids "
                        "this action. To allow flagged actions on unattended "
                        "platforms, set approvals.unattended_mode: approve in "
                        "config.yaml."
                    ),
                    pattern_key=pattern_key, description=description,
                )
        elif fail_closed_when_no_human:
            logger.warning(
                "%s (pattern: %s): %s — no interactive user/gateway present; "
                "BLOCKED (fail-closed). Set HERMES_INTERACTIVE or "
                "HERMES_GATEWAY_SESSION to answer the prompt.",
                autoapprove_log_prefix, pattern_key, description,
            )
            return _blocked(
                no_human_block_message or (
                    f"BLOCKED: approval required ({description}) but no "
                    "interactive user or gateway is present to approve it."
                ),
                pattern_key=pattern_key, description=description,
            )
        logger.warning(
            "%s (pattern: %s): %s — set HERMES_INTERACTIVE or "
            "HERMES_GATEWAY_SESSION to require approval.",
            autoapprove_log_prefix, pattern_key, description,
        )
        return _approved()

    if is_gateway or env_var_enabled("HERMES_EXEC_ASK"):
        # Interactive gateway round-trip: blocks the agent thread until the
        # user answers; the agent gets a definitive approved/BLOCKED outcome.
        notify_cb = _gateway_notify_cb(session_key)

        if notify_cb is not None:
            from agent.redact import redact_sensitive_text
            approval_data = {
                "command": redact_sensitive_text(display_target),
                "pattern_key": pattern_key,
                "pattern_keys": [pattern_key],
                "description": redact_sensitive_text(description),
                "allow_permanent": True,
                "allow_session": True,
            }
            decision = _await_gateway_decision(
                session_key, notify_cb, approval_data, surface="gateway"
            )
            if decision.get("notify_failed"):
                return _denied(
                    "BLOCKED: Failed to send approval request to user. Do NOT retry.",
                    pattern_key=pattern_key, description=description,
                    outcome="notify_failed",
                )
            refusal = _gateway_refusal(decision)
            if refusal is not None:
                reason, reason_addendum, timeout_addendum, outcome, deny_reason = refusal
                return _denied(
                    f"BLOCKED: Action {reason}.{reason_addendum} The user "
                    f"has NOT consented to this action. Do NOT retry it, "
                    f"do NOT rephrase it, and do NOT attempt the same "
                    f"outcome via a different path.{timeout_addendum}",
                    pattern_key=pattern_key, description=description,
                    outcome=outcome, deny_reason=deny_reason,
                )
            _persist_choice(session_key, decision["choice"], [(pattern_key, None, False)])
            return _approved()

        # No notify callback: an interactive CLI with a panel callback should
        # still prompt locally instead of queuing a pending approval nobody
        # can see (HERMES_EXEC_ASK / platform-marker leaks into CLI).
        if not _should_fall_through_to_cli_approval(
            is_cli=is_cli,
            approval_callback=approval_callback,
            notify_cb=notify_cb,
        ):
            # e.g. API server without an attached chat: queue for /approve
            # /deny review, agent sees approval_required.
            submit_pending(session_key, {
                "command": display_target,
                "pattern_key": pattern_key,
                "description": description,
            })
            return {
                "approved": False,
                "pattern_key": pattern_key,
                "status": "approval_required",
                "command": display_target,
                "description": description,
                "message": (
                    f"⚠️ This action is potentially dangerous ({description}). "
                    f"Asking the user for approval.\n\n**Target:**\n```\n{display_target}\n```"
                ),
            }

    choice = _prompt_cli_with_hooks(
        display_target, description, pattern_key, [pattern_key], session_key,
        approval_callback=approval_callback,
    )

    if choice == "timeout":
        return _denied(
            "BLOCKED: Action timed out without user response. The user "
            "has NOT consented to this action. Do NOT retry it, do NOT "
            "rephrase it, and do NOT attempt the same outcome via a "
            "different path. Silence is not consent.",
            pattern_key=pattern_key, description=description, outcome="timeout",
        )
    if choice == "deny":
        return _denied(
            f"BLOCKED: User denied this potentially dangerous action "
            f"(matched '{description}'). Do NOT retry — the user has "
            "explicitly rejected it.",
            pattern_key=pattern_key, description=description, outcome="denied",
        )
    _persist_choice(session_key, choice, [(pattern_key, None, False)])
    return _approved()


def _should_skip_container_guards(env_type: str, has_host_access: bool = False) -> bool:
    """True when the backend is isolated enough to skip dangerous-command prompts.

    Docker is the exception once host paths are bind-mounted: ``rm -rf
    /workspace`` then reaches host files, so it goes through normal approval.
    """
    if env_type == "docker":
        return not has_host_access
    return env_type in ("singularity", "modal", "daytona", "vercel_sandbox")


def check_dangerous_command(command: str, env_type: str,
                            approval_callback=None,
                            has_host_access: bool = False) -> dict:
    """Detect a dangerous command and handle approval (terminal_tool entry point).

    ``has_host_access``: a Docker sandbox bind-mounts host paths, so its
    commands can reach the host and must not skip approval.
    Returns ``{"approved": True/False, "message": str or None, ...}``.
    """
    if _should_skip_container_guards(env_type, has_host_access=has_host_access):
        return _approved()

    # Hardline floor: no-recovery commands (rm -rf /, mkfs, dd to raw device,
    # shutdown, fork bomb, kill -1) are blocked BEFORE the yolo bypass. Yolo
    # trusts the agent with your files, not to wipe the disk.
    is_hardline, hardline_desc = detect_hardline_command(command)
    if is_hardline:
        logger.warning("Hardline block: %s (command: %s)", hardline_desc, command[:200])
        return _hardline_block_result(hardline_desc, command)

    # approvals.deny also fires BEFORE yolo — "never, even under yolo".
    deny_pattern = _match_user_deny_rule(command)
    if deny_pattern is not None:
        logger.warning("User deny rule %r blocked command: %s",
                       deny_pattern, command[:200])
        return _user_deny_block_result(deny_pattern)

    # Gateway /yolo is session-scoped; CLI --yolo is process-scoped.
    if _YOLO_MODE_FROZEN or is_current_session_yolo_enabled():
        return _approved()

    if _command_matches_permanent_allowlist(command):
        return _approved()

    is_dangerous, pattern_key, description = detect_dangerous_command(command)
    if not is_dangerous:
        return _approved()

    return _run_approval_gate(
        pattern_key=pattern_key,
        description=description,
        display_target=command,
        approval_callback=approval_callback,
        cron_deny_message=(
            f"BLOCKED: Command flagged as dangerous ({description}) "
            "but cron jobs run without a user present to approve it. "
            "Find an alternative approach that avoids this command. "
            "To allow dangerous commands in cron jobs, set "
            "approvals.cron_mode: approve in config.yaml."
        ),
        single_query_deny_message=(
            f"BLOCKED: Command flagged as dangerous ({description}) but "
            "single-query mode (-q) runs without a user present to approve "
            "it. Find an alternative approach that avoids this command. "
            "To allow dangerous commands in single-query mode, set "
            "approvals.single_query_mode: approve in config.yaml."
        ),
        autoapprove_log_prefix=(
            "AUTO-APPROVED dangerous command in non-interactive non-gateway context"
        ),
    )


def request_tool_approval(
    tool_name: str,
    reason: str,
    *,
    rule_key: str = "",
    approval_callback=None,
) -> dict:
    """Escalate an arbitrary tool call to the human-approval gate.

    Entry point for a plugin ``pre_tool_call`` hook returning
    ``{"action": "approve", "message": ...}``: it asks the SAME human gate as
    Tier-2 dangerous shell patterns (session/permanent allowlist, CLI prompt,
    gateway pending, once/session/always/deny, timeout fail-closed), so the LLM
    cannot skip it. Cron honors ``approvals.cron_mode``; any OTHER
    non-interactive non-gateway context fails CLOSED.

    ``rule_key`` controls the ``[a]lways`` allowlist grain. When empty, the key
    is ``tool_name`` + a hash of ``reason`` so DISTINCT reasons on the same tool
    persist independently ("write to ~/.ssh" does not auto-approve a later
    "send email" rule). Returns the ``check_dangerous_command`` result shape.
    """
    description = reason or f"Plugin requires approval for {tool_name}"
    if rule_key:
        key_suffix = rule_key
    else:
        _reason_hash = hashlib.sha256(description.encode("utf-8")).hexdigest()[:12]
        key_suffix = f"{tool_name}:{_reason_hash}"
    # Namespaced so plugin-rule approvals share the allowlist machinery without
    # ever colliding with a real command pattern key.
    pattern_key = f"plugin_rule:{key_suffix}"
    # Synthetic label for the display/allowlist layer; it never executes.
    display_target = f"<{tool_name}> (plugin approval rule)"

    return _run_approval_gate(
        pattern_key=pattern_key,
        description=description,
        display_target=display_target,
        approval_callback=approval_callback,
        cron_deny_message=(
            f"BLOCKED: Tool '{tool_name}' requires approval ({description}) "
            "but cron jobs run without a user present to approve it. Find an "
            "alternative approach. To allow flagged actions in cron jobs, set "
            "approvals.cron_mode: approve in config.yaml."
        ),
        single_query_deny_message=(
            f"BLOCKED: Tool '{tool_name}' requires approval ({description}) "
            "but single-query mode (-q) runs without a user present to "
            "approve it. Find an alternative approach. To allow flagged "
            "actions in single-query mode, set "
            "approvals.single_query_mode: approve in config.yaml."
        ),
        autoapprove_log_prefix=(
            f"plugin-escalated tool call '{tool_name}' in "
            "non-interactive non-gateway context"
        ),
        fail_closed_when_no_human=True,
        no_human_block_message=(
            f"BLOCKED: Tool '{tool_name}' requires approval ({description}) "
            "but no interactive user or gateway is present to approve it. "
            "A plugin flagged this action for human confirmation."
        ),
    )


# =========================================================================
# Combined pre-exec guard (tirith + dangerous command detection)
# =========================================================================

def _format_tirith_description(tirith_result: dict) -> str:
    """Human-readable severity/title/description summary of tirith findings."""
    findings = tirith_result.get("findings") or []
    parts = []
    for f in findings:
        severity = f.get("severity", "")
        title = f.get("title", "")
        desc = f.get("description", "")
        if title and desc:
            parts.append(f"[{severity}] {title}: {desc}" if severity else f"{title}: {desc}")
        elif title:
            parts.append(f"[{severity}] {title}" if severity else title)
    if not parts:
        summary = tirith_result.get("summary") or "security issue detected"
        return f"Security scan: {summary}"
    return "Security scan — " + "; ".join(parts)


def get_plugin_manager():
    """Lazy plugin-manager seam used by tests and early tool-only imports."""
    from hermes_cli.plugins import discover_plugins, get_plugin_manager as _get_manager

    # Approval can be imported before model_tools (which triggers discovery);
    # make an explicitly selected transport available on the first approval
    # instead of treating the undiscovered registry as unavailable.
    discover_plugins()
    return _get_manager()


def _get_approval_transport_config() -> tuple[str, str | None]:
    """Return explicitly selected transport and fail-closed fallback mode."""
    try:
        from hermes_cli.config import load_config_readonly

        config = load_config_readonly() or {}
        approval_config = ((config.get("security") or {}).get("approval") or {})
        selected = str(approval_config.get("transport") or "builtin").strip().lower()
        fallback = str(approval_config.get("transport_fallback") or "").strip().lower()
    except Exception:
        # An unreadable/malformed selection must not silently materialize a
        # prompt on a built-in surface the operator may not be watching.
        return "config-error", None
    return selected or "builtin", "builtin" if fallback == "builtin" else None


def _present_with_selected_transport(
    *,
    command: str,
    description: str,
    pattern_key: str,
    pattern_keys: list[str],
    session_key: str,
    surface: str,
    allow_session: bool,
    allow_permanent: bool,
) -> dict:
    """Present through an explicitly selected plugin transport, if any.

    A selected transport replaces every built-in prompt surface; detection,
    allowed scopes, persistence, timeout, and final authorization stay
    host-owned. A failed transport reaches a built-in surface only under the
    explicit ``transport_fallback: builtin`` opt-in.
    """
    name, fallback = _get_approval_transport_config()
    if name == "builtin":
        return {"selected": False}

    try:
        registered = get_plugin_manager().get_approval_transport(name)
    except Exception:
        # Plugin/discovery exception text may contain plugin-owned secrets.
        logger.warning("Could not resolve selected approval transport %r", name)
        registered = None
    if registered is None:
        logger.warning("Selected approval transport %r is unavailable", name)
        return {
            "selected": True,
            "choice": "deny",
            "failure": "unavailable",
            "fallback": fallback,
            "name": name,
        }

    try:
        from agent.redact import redact_sensitive_text
        from hermes_cli.approval_transport import ApprovalRequest, invoke_approval_transport

        timeout_seconds = _get_approval_timeout()
        request = ApprovalRequest.create(
            command=redact_sensitive_text(command, force=True),
            description=redact_sensitive_text(description, force=True),
            pattern_key=pattern_key,
            pattern_keys=tuple(pattern_keys),
            session_key=session_key,
            surface=surface,
            allow_session=allow_session,
            allow_permanent=allow_permanent,
            timeout_seconds=timeout_seconds,
        )
    except Exception:
        # Never fall back to raw text if redaction or request construction
        # fails: fail closed without calling the plugin or leaking the
        # unredacted payload to logs/hooks.
        logger.warning("Could not build redacted plugin approval request")
        return {
            "selected": True,
            "choice": "deny",
            "failure": "error",
            "fallback": None,
            "name": name,
        }
    hook_kwargs = dict(
        command=request.command,
        description=request.description,
        pattern_key=pattern_key,
        pattern_keys=list(pattern_keys),
        session_key=session_key,
        surface=f"transport:{name}",
        request_id=request.request_id,
        request_digest=request.digest,
    )
    _fire_approval_hook("pre_approval_request", **hook_kwargs)
    try:
        from tools.environments.base import touch_activity_if_due
    except Exception:  # pragma: no cover - minimal tool-only environments
        touch_activity_if_due = None
    now = time.monotonic()
    activity_state = {"last_touch": now, "start": now}

    def _poll() -> None:
        if touch_activity_if_due is not None:
            touch_activity_if_due(activity_state, "waiting for plugin approval transport")

    with human_wait_window(session_key):
        result = invoke_approval_transport(
            registered.present,
            request,
            timeout_seconds=timeout_seconds,
            on_poll=_poll,
            is_interrupted=is_interrupted,
        )
    hook_choice = result.choice if result.failure is None else f"transport_{result.failure}"
    _fire_approval_hook("post_approval_response", **hook_kwargs, choice=hook_choice)
    return {
        "selected": True,
        "choice": result.choice,
        "failure": result.failure,
        "fallback": fallback,
        "name": name,
    }


def _transport_denied_result(
    *, pattern_key: str, description: str, failure: str
) -> dict:
    breaker_addendum = _denial_breaker_addendum(get_current_session_key())
    return _denied(
        f"BLOCKED: Selected approval transport failed ({failure}); the user "
        "has NOT consented to this action. Do NOT retry this command or "
        "attempt the same outcome through another route."
        f"{breaker_addendum}",
        pattern_key=pattern_key, description=description,
        outcome=f"transport_{failure}",
    )


def _transport_choice(attempt: dict, *, pattern_key: str, description: str):
    """Interpret a ``_present_with_selected_transport`` attempt.

    Returns ``(choice, denied_result)``: both None when the built-in surfaces
    should run (no transport selected, or a failure with the explicit builtin
    fallback); a denied result for any other failure; else the user's choice.
    """
    if not attempt.get("selected"):
        return None, None
    failure = attempt.get("failure")
    if failure and attempt.get("fallback") == "builtin":
        logger.warning(
            "Approval transport %r failed (%s); using explicit builtin fallback",
            attempt.get("name"),
            failure,
        )
        return None, None
    if failure:
        return None, _transport_denied_result(
            pattern_key=pattern_key, description=description, failure=failure,
        )
    return attempt.get("choice"), None


# Non-interactive contexts with nobody to answer a prompt, in evaluation order:
# (context predicate, mode getter, config key, "why nobody can approve" clause,
# whether the dangerous-pattern block carries pattern_key/description).
def _unattended_contexts() -> list[tuple]:
    contexts = []
    if _is_single_query_approval_context():
        contexts.append((
            _get_single_query_approval_mode, "single_query_mode",
            "single-query mode (-q) runs without a user present to approve it",
            "in single-query mode", True,
        ))
    if _is_cron_approval_context():
        contexts.append((
            _get_cron_approval_mode, "cron_mode",
            "cron jobs run without a user present to approve it",
            "in cron jobs", False,
        ))
    elif _is_unattended_platform_approval_context():
        contexts.append((
            _get_unattended_approval_mode, "unattended_mode",
            "this session runs on an unattended platform "
            f"({_get_session_platform()}) with no user present to approve it",
            "on unattended platforms", False,
        ))
    return contexts


def _unattended_deny(command: str, mode_getter, cfg_key: str, clause: str,
                     scope: str, with_keys: bool) -> dict | None:
    """Deny-mode handling for one unattended context (cron / -q / webhook).

    Pattern detection first, then tirith so content-level threats (homograph
    URLs, pipe-to-interpreter, terminal injection) are caught even when the
    pattern detector misses. An un-importable tirith honours
    ``security.tirith_fail_open``: fail-closed means block, since nobody can
    approve (#20733). Returns None to allow.
    """
    if mode_getter() != "deny":
        return None
    allow_hint = (f"Find an alternative approach that avoids this command. To allow "
                  f"dangerous commands {scope}, set approvals.{cfg_key}: approve in config.yaml.")
    is_dangerous, _pk, description = detect_dangerous_command(command)
    if is_dangerous:
        result = {
            "approved": False,
            "message": f"BLOCKED: Command flagged as dangerous ({description}) but {clause}. {allow_hint}",
        }
        if with_keys:
            result.update(pattern_key=_pk, description=description)
        return result
    try:
        from tools.tirith_security import check_command_security
        tirith = check_command_security(command)
    except ImportError:
        if _tirith_fail_open():
            return None
        return {
            "approved": False,
            "message": (
                "BLOCKED: the Tirith security scanner could not be "
                "imported and security.tirith_fail_open is false, "
                f"so this command cannot be silently allowed — and {clause}. "
                f"Find an alternative approach, install tirith, or set "
                f"approvals.{cfg_key}: approve in config.yaml."
            ),
        }
    if tirith.get("action") in ("block", "warn"):
        return {
            "approved": False,
            "message": f"BLOCKED: {_format_tirith_description(tirith)} but {clause}. {allow_hint}",
        }
    return None


def _tirith_scan(command: str) -> dict:
    """Tirith result for the interactive flow; an un-importable scanner allows
    (default) or, under fail-closed, synthesizes a HIGH warn finding that goes
    through the normal approval flow (#20733)."""
    try:
        from tools.tirith_security import check_command_security
        return check_command_security(command)
    except ImportError:
        if _tirith_fail_open():
            return {"action": "allow", "findings": [], "summary": ""}
        return {
            "action": "warn",
            "findings": [
                {
                    "rule_id": "tirith-import-error",
                    "severity": "HIGH",
                    "title": "Tirith security module unavailable",
                    "description": (
                        "The Tirith security scanner could not be imported. "
                        "Because security.tirith_fail_open is false, this "
                        "command cannot be silently allowed. Approve only if "
                        "you have verified the command is safe."
                    ),
                }
            ],
            "summary": "Tirith unavailable (fail-closed)",
        }


def check_all_command_guards(command: str, env_type: str,
                             approval_callback=None,
                             has_host_access: bool = False) -> dict:
    """Run all pre-exec security checks and return a single approval decision.

    Tirith and dangerous-command findings are presented as ONE combined
    approval request, so a gateway force=True replay cannot bypass one check
    when only the other was shown to the user. ``has_host_access``: a Docker
    sandbox with bind-mounted host paths is no longer isolated and takes the
    normal flow instead of the container fast-path.
    """
    if _should_skip_container_guards(env_type, has_host_access=has_host_access):
        return _approved()

    # Unconditional floors, BEFORE yolo / mode=off / cron approve-mode so no
    # session-level setting can bypass them: hardline catastrophic commands,
    # password-piping to sudo -S with no SUDO_PASSWORD configured, and the
    # user's own approvals.deny rules ("never, even under yolo").
    is_hardline, hardline_desc = detect_hardline_command(command)
    if is_hardline:
        logger.warning("Hardline block: %s (command: %s)", hardline_desc, command[:200])
        return _hardline_block_result(hardline_desc, command)

    is_sudo_guess, sudo_guess_desc = _check_sudo_stdin_guard(command)
    if is_sudo_guess:
        logger.warning("Sudo stdin guard block: %s (command: %s)",
                       sudo_guess_desc, command[:200])
        return _sudo_stdin_block_result(sudo_guess_desc)

    deny_pattern = _match_user_deny_rule(command)
    if deny_pattern is not None:
        logger.warning("User deny rule %r blocked command: %s",
                       deny_pattern, command[:200])
        return _user_deny_block_result(deny_pattern)

    # Gateway /yolo is session-scoped; CLI --yolo remains process-scoped.
    approval_mode = _get_approval_mode()
    if _YOLO_MODE_FROZEN or is_current_session_yolo_enabled() or approval_mode == "off":
        return _approved()

    if _command_matches_permanent_allowlist(command):
        return _approved()

    approval_callback = _resolve_cli_approval_callback(approval_callback)
    is_cli = _is_interactive_cli()
    is_gateway = _is_gateway_approval_context()
    is_ask = env_var_enabled("HERMES_EXEC_ASK")

    # Single-query (-q) exports HERMES_INTERACTIVE=1 but nobody answers
    # prompts; HERMES_EXEC_ASK has no human either — ignore both so
    # single_query_mode actually takes effect.
    if _is_single_query_approval_context():
        is_cli = is_gateway = is_ask = False

    # Outside CLI/gateway/ask flows we never block on approvals: each
    # unattended context applies its configured deny/approve mode, else allow.
    if not is_cli and not is_gateway and not is_ask:
        for mode_getter, cfg_key, clause, scope, with_keys in _unattended_contexts():
            result = _unattended_deny(command, mode_getter, cfg_key, clause, scope, with_keys)
            if result is not None:
                return result
        return _approved()

    # --- Phase 1: gather findings ---
    tirith_result = _tirith_scan(command)
    is_dangerous, pattern_key, description = detect_dangerous_command(command)

    # --- Phase 2: decide --- warnings = [(pattern_key, description, is_tirith)]
    warnings = []
    session_key = get_current_session_key()

    # Tirith block AND warn both go through the approval flow (block used to
    # be a hard stop) so users can inspect the findings and approve.
    if tirith_result["action"] in {"block", "warn"}:
        findings = tirith_result.get("findings") or []
        rule_id = findings[0].get("rule_id", "unknown") if findings else "unknown"
        tirith_key = f"tirith:{rule_id}"
        if not is_approved(session_key, tirith_key):
            warnings.append((tirith_key, _format_tirith_description(tirith_result), True))

    if is_dangerous and not is_approved(session_key, pattern_key):
        warnings.append((pattern_key, description, False))

    if not warnings:
        return _approved()

    combined_desc = "; ".join(desc for _, desc, _ in warnings)
    primary_key = warnings[0][0]
    all_keys = [key for key, _, _ in warnings]

    # --- Phase 2.5: smart approval (auxiliary LLM) ---
    smart_denied_for_owner = False
    if approval_mode == "smart":
        verdict = _smart_verdict(command, combined_desc, primary_key, all_keys, session_key)
        if verdict == "approve":
            # Approve this command only: pattern-level persistence would let
            # one benign command suppress review of later commands in the
            # same broad detector category.
            _reset_denials(session_key)
            logger.debug("Smart approval: auto-approved '%s' (%s)",
                         command[:60], combined_desc)
            return {"approved": True, "message": None,
                    "smart_approved": True,
                    "description": combined_desc}
        if verdict == "deny":
            # A guardian DENY counts toward the consecutive-denial breaker even
            # when an interactive owner may override it for this one operation.
            _record_denial(session_key)
            if not (is_cli or is_gateway or is_ask):
                breaker_addendum = _denial_breaker_addendum(session_key)
                return {
                    "approved": False,
                    "message": f"BLOCKED by smart approval: {combined_desc}. "
                               "The command was assessed as genuinely dangerous. "
                               f"Do NOT retry.{breaker_addendum}",
                    "smart_denied": True,
                }
            smart_denied_for_owner = True
        # ESCALATE follows the normal, potentially persistent manual behavior.

    # --- Phase 3: approval ---
    # "Always" is offered when at least one warning is a dangerous-pattern key
    # the persistence layer would actually allowlist permanently. Pure-tirith
    # findings are session-max by design, so a tirith-only prompt hides Always;
    # mixed prompts offer it (the pattern key persists, tirith downgrades to
    # session — see _persist_choice).
    has_permanent_capable = any(not is_t for _, _, is_t in warnings)
    allow_permanent = has_permanent_capable and not smart_denied_for_owner

    transport_attempt = _present_with_selected_transport(
        command=command,
        description=combined_desc,
        pattern_key=primary_key,
        pattern_keys=all_keys,
        session_key=session_key,
        surface="gateway" if (is_gateway or is_ask) else "cli",
        allow_session=not smart_denied_for_owner,
        allow_permanent=allow_permanent,
    )
    transport_choice, denied = _transport_choice(
        transport_attempt, pattern_key=primary_key, description=combined_desc,
    )
    if denied is not None:
        return denied
    if transport_choice is not None:
        if transport_choice == "deny":
            _record_denial(session_key)
            breaker_addendum = _denial_breaker_addendum(session_key)
            return _denied(
                "BLOCKED: User denied this command through the selected "
                "approval transport. The user has NOT consented to this "
                "action. Do NOT retry or attempt the same outcome through "
                f"another route.{breaker_addendum}",
                pattern_key=primary_key, description=combined_desc, outcome="denied",
            )
        if not smart_denied_for_owner:
            _persist_choice(session_key, transport_choice, warnings)
        return _user_approved(session_key, combined_desc)

    # Gateway/async approval: block the agent thread until /approve or /deny,
    # mirroring the CLI's synchronous input() flow. The agent never sees
    # "approval_required" here — it gets output or a definitive BLOCKED.
    if is_gateway or is_ask:
        notify_cb = _gateway_notify_cb(session_key)

        if notify_cb is not None:
            from agent.redact import redact_sensitive_text
            approval_data = _gateway_approval_data(
                redact_sensitive_text(command), redact_sensitive_text(combined_desc),
                primary_key, all_keys,
                allow_permanent=has_permanent_capable, smart_denied=smart_denied_for_owner,
            )
            decision = _await_gateway_decision(
                session_key, notify_cb, approval_data, surface="gateway"
            )
            if decision.get("notify_failed"):
                return _denied(
                    "BLOCKED: Failed to send approval request to user. Do NOT retry.",
                    pattern_key=primary_key, description=combined_desc,
                    outcome="notify_failed",
                )
            refusal = _gateway_refusal(decision)
            if refusal is not None:
                reason, reason_addendum, timeout_addendum, outcome, deny_reason = refusal
                breaker_addendum = _denial_breaker_addendum(session_key)
                return _denied(
                    f"BLOCKED: Command {reason}.{reason_addendum} The user "
                    f"has NOT consented to this action. Do NOT retry this "
                    f"command, do NOT rephrase it, and do NOT attempt the "
                    f"same outcome via a different command. Stop the "
                    f"current workflow and wait for the user to respond "
                    f"before taking any further destructive or "
                    f"irreversible action.{timeout_addendum}{breaker_addendum}",
                    pattern_key=primary_key, description=combined_desc,
                    outcome=outcome, deny_reason=deny_reason,
                )
            # A smart-DENY owner override is always one operation, even if an
            # older client returns "session" or "always".
            if not smart_denied_for_owner:
                _persist_choice(session_key, decision["choice"], warnings)
            return _user_approved(session_key, combined_desc)

        # No gateway callback (cron, batch, or ask-mode leaked into an
        # interactive CLI): paint the local panel when possible instead of a
        # pending_approval that makes the agent look "auto-blocked".
        if not _should_fall_through_to_cli_approval(
            is_cli=is_cli,
            approval_callback=approval_callback,
            notify_cb=notify_cb,
        ):
            from agent.redact import redact_sensitive_text
            _disp_command = redact_sensitive_text(command)
            return _pending_result(
                session_key,
                display_command=_disp_command,
                display_description=redact_sensitive_text(combined_desc),
                pattern_key=primary_key, pattern_keys=all_keys,
                body=f"**Command:**\n```\n{_disp_command}\n```", noun="command",
                smart_denied=smart_denied_for_owner,
            )

    # CLI interactive: single combined prompt.
    choice = _prompt_cli_with_hooks(
        command, combined_desc, primary_key, all_keys, session_key,
        allow_permanent=allow_permanent,
        smart_denied=smart_denied_for_owner,
        approval_callback=approval_callback,
    )

    if choice == "timeout":
        breaker_addendum = _denial_breaker_addendum(session_key)
        return _denied(
            "BLOCKED: Command timed out without user response. The user "
            "has NOT consented to this action. Do NOT retry this "
            "command, do NOT rephrase it, and do NOT attempt the same "
            "outcome via a different command. Stop the current workflow "
            "and wait for the user to respond before taking any further "
            "destructive or irreversible action. Silence is not "
            f"consent.{breaker_addendum}",
            pattern_key=primary_key, description=combined_desc, outcome="timeout",
        )
    if choice == "deny":
        breaker_addendum = _denial_breaker_addendum(session_key)
        return _denied(
            "BLOCKED: User denied this command. The user has NOT consented "
            "to this action. Do NOT retry this command, do NOT rephrase "
            "it, and do NOT attempt the same outcome via a different "
            "command. Stop the current workflow and wait for the user "
            f"to respond before taking any further destructive or "
            f"irreversible action.{breaker_addendum}",
            pattern_key=primary_key, description=combined_desc, outcome="denied",
        )
    if not smart_denied_for_owner:
        _persist_choice(session_key, choice, warnings)
    return _user_approved(session_key, combined_desc)


_EXECUTE_CODE_DESCRIPTION = (
    "execute_code script execution. The script can spawn subprocesses or "
    "mutate files without passing through terminal command approval; "
    "approval is one-shot for this run."
)
_EXECUTE_CODE_UNATTENDED_TAILS = {
    "single_query": (
        "Single-query mode (-q) runs without a user present to approve it. Use "
        "normal tools instead, or set approvals.single_query_mode: approve only "
        "if this single-query run is intentionally trusted."
    ),
    "cron": (
        "Cron jobs run without a user present to approve it. Use normal tools "
        "instead, or set approvals.cron_mode: approve only if this cron profile "
        "is intentionally trusted."
    ),
}


def _execute_code_unattended_block(tail: str) -> dict:
    return _denied(
        "BLOCKED: execute_code runs arbitrary local Python (including "
        "subprocess calls that bypass shell-string approval checks). " + tail,
        pattern_key="execute_code", description=_EXECUTE_CODE_DESCRIPTION,
        outcome="blocked",
    )


def check_execute_code_guard(code: str, env_type: str,
                             has_host_access: bool = False) -> dict:
    """Approve an execute_code script before its child process is spawned.

    The script can call ``subprocess``/``os.system``/``ctypes`` directly, none
    of which pass through ``terminal()`` / ``DANGEROUS_PATTERNS``. In
    gateway/ask contexts we fail closed by approving the script as a whole
    (#30882). Same dict contract as ``check_all_command_guards``.

    Documented limitation: a purely local non-interactive non-gateway session
    (no TTY, not gateway, not cron-deny) returns approved — matching the
    terminal auto-approve contract. The hardline floor still blocks
    catastrophic ``terminal()`` commands the script issues.
    """
    pattern_key = "execute_code"
    description = _EXECUTE_CODE_DESCRIPTION

    # Isolated backends already sandbox the child. vercel_sandbox has no
    # host-bind concept so it stays always-skipped.
    if env_type == "vercel_sandbox":
        return _approved()
    if _should_skip_container_guards(env_type, has_host_access=has_host_access):
        return _approved()

    approval_mode = _get_approval_mode()
    if _YOLO_MODE_FROZEN or is_current_session_yolo_enabled() or approval_mode == "off":
        return _approved()

    is_gateway = _is_gateway_approval_context()
    is_ask = env_var_enabled("HERMES_EXEC_ASK")
    is_cli = _is_interactive_cli()
    approval_callback = _resolve_cli_approval_callback()

    # No user is present to approve arbitrary code in -q / cron / unattended
    # sessions: each resolves instantly from its configured mode.
    if _is_single_query_approval_context():
        if _get_single_query_approval_mode() == "deny":
            return _execute_code_unattended_block(_EXECUTE_CODE_UNATTENDED_TAILS["single_query"])
        return _approved()
    if _is_cron_approval_context():
        if _get_cron_approval_mode() == "deny":
            return _execute_code_unattended_block(_EXECUTE_CODE_UNATTENDED_TAILS["cron"])
        return _approved()
    if _is_unattended_platform_approval_context():
        if _get_unattended_approval_mode() == "deny":
            return _execute_code_unattended_block(
                "This session runs on an unattended "
                f"platform ({_get_session_platform()}) with no user "
                "present to approve it. Use normal tools instead, or set "
                "approvals.unattended_mode: approve only if sessions on "
                "this surface are intentionally trusted."
            )
        return _approved()

    # Only gateway/ask contexts get the one-shot whole-script approval. In an
    # interactive CLI the script's terminal() calls are guarded per-call
    # (context propagates into the RPC thread, #33057), so a whole-script
    # prompt would fire on every execute_code call. Ask-mode still takes this
    # path even with INTERACTIVE set (how gateway/smart tests and messaging
    # ask-mode drive whole-script approval); when that leaks into a CLI with no
    # notify callback, the notify_cb-less branch below falls through to the
    # CLI Dangerous Command panel instead of a silent pending_approval.
    if not is_gateway and not is_ask:
        return _approved()

    session_key = get_current_session_key()
    # Built only past the early-return gates so common paths don't copy a
    # potentially-large script into this string.
    command = f"execute_code <<'PY'\n{code}\nPY"

    # Without this, "Approve session" / "Always" choices are stored but never
    # consulted, so every execute_code call re-prompts (#39275).
    if is_approved(session_key, pattern_key):
        return _approved()

    # Smart mode: an APPROVE only suppresses the redundant whole-script prompt;
    # the per-call terminal() guards still run independently.
    smart_denied_for_owner = False
    if approval_mode == "smart":
        verdict = _smart_verdict(command, description, pattern_key, [pattern_key], session_key)
        if verdict == "approve":
            _reset_denials(session_key)
            logger.debug("Smart approval: auto-approved execute_code for session %s",
                         session_key)
            return {"approved": True, "message": None,
                    "smart_approved": True, "description": description}
        if verdict == "deny":
            _record_denial(session_key)
            if not (is_gateway or is_ask):
                breaker_addendum = _denial_breaker_addendum(session_key)
                return _denied(
                    "BLOCKED by smart approval: execute_code script "
                    "execution was assessed as genuinely dangerous. "
                    f"Do NOT retry.{breaker_addendum}",
                    pattern_key=pattern_key, description=description,
                    outcome="denied", smart_denied=True,
                )
            smart_denied_for_owner = True
        # ESCALATE retains the normal manual approval behavior.

    # Redacted copies for user-visible rendering only: a script can embed
    # credentials and the gateway renders this payload to Discord/Slack. The
    # raw `command`/`code` are what get assessed and executed.
    from agent.redact import redact_sensitive_text
    display_command = redact_sensitive_text(command)
    display_code = redact_sensitive_text(code)
    display_description = redact_sensitive_text(description)

    transport_attempt = _present_with_selected_transport(
        command=command,
        description=description,
        pattern_key=pattern_key,
        pattern_keys=[pattern_key],
        session_key=session_key,
        surface="gateway",
        allow_session=not smart_denied_for_owner,
        allow_permanent=not smart_denied_for_owner,
    )
    transport_choice, denied = _transport_choice(
        transport_attempt, pattern_key=pattern_key, description=description,
    )
    if denied is not None:
        return denied
    if transport_choice is not None:
        if transport_choice == "deny":
            _record_denial(session_key)
            return _denied(
                "BLOCKED: User denied execute_code through the selected "
                "approval transport. The user has NOT consented.",
                pattern_key=pattern_key, description=description, outcome="denied",
            )
        if not smart_denied_for_owner:
            _persist_choice(session_key, transport_choice, [(pattern_key, None, False)])
        return _user_approved(session_key, description)

    notify_cb = _gateway_notify_cb(session_key)

    if notify_cb is None:
        # HERMES_EXEC_ASK (or a platform marker) leaked into an interactive CLI
        # (commonly via `import gateway.run`): show the CLI panel the user can
        # answer rather than silently queueing a pending approval.
        if _should_fall_through_to_cli_approval(
            is_cli=is_cli,
            approval_callback=approval_callback,
            notify_cb=notify_cb,
        ):
            choice = _prompt_cli_with_hooks(
                display_command, display_description, pattern_key, [pattern_key],
                session_key,
                allow_permanent=not smart_denied_for_owner,
                approval_callback=approval_callback,
                smart_denied=smart_denied_for_owner,
            )
            if choice == "timeout":
                breaker_addendum = _denial_breaker_addendum(session_key)
                return _denied(
                    "BLOCKED: Action timed out without user response. The "
                    "user has NOT consented to this action. Do NOT retry "
                    "it, do NOT rephrase it, and do NOT attempt the same "
                    "outcome via a different path. Silence is not "
                    f"consent.{breaker_addendum}",
                    pattern_key=pattern_key, description=description, outcome="timeout",
                )
            if choice == "deny":
                # No _record_denial(): the breaker counts consecutive guardian
                # LLM DENY verdicts, not deliberate human denials — matching
                # the sibling CLI tails.
                breaker_addendum = _denial_breaker_addendum(session_key)
                return _denied(
                    "BLOCKED: User denied execute_code script execution "
                    f"(matched '{description}'). Do NOT retry — the user "
                    f"has explicitly rejected it.{breaker_addendum}",
                    pattern_key=pattern_key, description=description, outcome="denied",
                )
            if not smart_denied_for_owner:
                _persist_choice(session_key, choice, [(pattern_key, None, False)])
            return _user_approved(session_key, description)

        return _pending_result(
            session_key,
            display_command=display_command,
            display_description=display_description,
            pattern_key=pattern_key, pattern_keys=[pattern_key],
            body=f"**Code:**\n```python\n{display_code}\n```", noun="code",
            smart_denied=smart_denied_for_owner,
        )

    approval_data = _gateway_approval_data(
        display_command, display_description, pattern_key, [pattern_key],
        allow_permanent=True, smart_denied=smart_denied_for_owner,
    )
    decision = _await_gateway_decision(
        session_key, notify_cb, approval_data, surface="gateway"
    )
    if decision.get("notify_failed"):
        return _denied(
            "BLOCKED: Failed to send execute_code approval request "
            "to user. Do NOT retry.",
            pattern_key=pattern_key, description=description, outcome="notify_failed",
        )

    refusal = _gateway_refusal(decision)
    if refusal is not None:
        reason, reason_addendum, addendum, outcome, deny_reason = refusal
        breaker_addendum = _denial_breaker_addendum(session_key)
        return _denied(
            f"BLOCKED: execute_code script {reason}.{reason_addendum} The "
            f"user has NOT consented to running this code. Do NOT retry, "
            f"do NOT rephrase the script, and do NOT attempt the same "
            f"outcome via a different tool.{addendum}{breaker_addendum}",
            pattern_key=pattern_key, description=description,
            outcome=outcome, deny_reason=deny_reason,
        )

    # Never persist a smart-DENY override under the coarse execute_code key;
    # that would approve unrelated future scripts. "once" persists nothing.
    if not smart_denied_for_owner:
        _persist_choice(session_key, decision["choice"], [(pattern_key, None, False)])
    return _user_approved(session_key, description)


# =========================================================================
# MCP elicitation entry point
# =========================================================================

def request_elicitation_consent(
    message: str,
    description: str,
    *,
    timeout_seconds: int | None = None,
    surface: str = "mcp-elicitation",
) -> str:
    """Route an MCP elicitation request to the surface owning the active session.

    Gateway sessions go through ``_await_gateway_decision``; CLI/TUI through
    ``prompt_dangerous_approval``. Always fails closed: a missing notify_cb in
    a gateway session, timeouts, and exceptions map to ``"decline"`` so a server
    treats them as "user did not approve" rather than retrying or hanging.
    Returns ``"accept" | "decline" | "cancel"``.
    """
    try:
        session_key = get_current_session_key()
    except Exception as exc:  # pragma: no cover -- defensive
        logger.warning("Elicitation consent: session lookup failed: %s", exc)
        return "decline"

    if _is_gateway_approval_context():
        notify_cb = _gateway_notify_cb(session_key)
        if notify_cb is None:
            logger.warning(
                "Elicitation requested in gateway session %s but no "
                "notify_cb is registered — failing closed",
                session_key,
            )
            return "decline"

        approval_data = {
            "command": message,
            "description": description,
            "pattern_key": "mcp_elicitation",
            "pattern_keys": ["mcp_elicitation"],
        }
        try:
            decision = _await_gateway_decision(
                session_key, notify_cb, approval_data, surface=surface,
            )
        except Exception as exc:
            logger.error(
                "Elicitation gateway dispatch failed: %s", exc, exc_info=True,
            )
            return "decline"

        if decision.get("notify_failed"):
            return "decline"
        if not decision.get("resolved"):
            return "cancel"
        choice = decision.get("choice")
        if choice in ("once", "session", "always"):
            return "accept"
        return "decline"

    # allow_permanent=False: elicitation is a per-call confirmation — there
    # is no pattern to remember.
    try:
        choice = prompt_dangerous_approval(
            message,
            description,
            timeout_seconds=timeout_seconds,
            allow_permanent=False,
        )
    except Exception as exc:
        logger.error(
            "Elicitation CLI prompt failed: %s", exc, exc_info=True,
        )
        return "decline"

    if choice in ("once", "session", "always"):
        return "accept"
    if choice == "timeout":
        return "cancel"  # mirror the gateway's unresolved outcome
    return "decline"


# Load permanent allowlist from config on module import
load_permanent_allowlist()
