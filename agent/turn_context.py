"""Per-turn setup for ``run_conversation`` (the turn prologue).

``build_turn_context`` runs the once-per-turn setup (stdio guard, sanitization, prompt
restore-or-build, session row, preflight compression, pre_llm_call hook, prefetch,
persistence) mutating ``agent`` exactly as the inline code did, and returns a
``TurnContext`` carrying only the locals the loop reads back."""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

from agent.conversation_compression import (
    IDLE_COMPACTION_STATUS_TEMPLATE,
    PREFLIGHT_COMPRESSION_STATUS_TEMPLATE,
    compression_skipped_due_to_lock,
    conversation_history_after_compression,
    recover_rotated_compression_session,
)
from agent.context_engine import automatic_compaction_status_message
from agent.iteration_budget import IterationBudget
from agent.memory_manager import build_memory_context_block
from agent.memory_provider import is_trivial_prompt
from agent.message_metadata import append_message, stamp_message_timestamp
from agent.model_metadata import (
    anchored_context_tokens,
    estimate_messages_tokens_rough,
    estimate_request_tokens_rough,
)

logger = logging.getLogger(__name__)


def _preflight_request_tokens(
    agent: Any,
    messages: List[Dict[str, Any]],
    system_prompt: str,
) -> int:
    """Token estimate for automatic preflight compression.

    Prefers a valid provider usage anchor; on native-compaction-eligible requests counts
    the checkpoint-pruned wire payload; otherwise uses the generic estimator."""
    anchored = anchored_context_tokens(
        messages, getattr(agent, "_usage_anchor", None)
    )
    if anchored is not None:
        return anchored
    tools = getattr(agent, "tools", None) or None
    try:
        from agent.codex_responses_adapter import (
            estimate_native_responses_preflight_tokens,
        )

        native = estimate_native_responses_preflight_tokens(
            agent,
            messages,
            system_prompt=system_prompt or "",
            tools=tools,
        )
        if isinstance(native, int) and not isinstance(native, bool) and native >= 0:
            return native
    except Exception:
        logger.debug(
            "native Responses preflight estimate unavailable; "
            "using generic transcript estimate",
            exc_info=True,
        )
    if _agent_stale_thinking_on_wire(agent):
        return estimate_request_tokens_rough(
            messages,
            system_prompt=system_prompt or "",
            tools=tools,
        )
    return estimate_request_tokens_rough(
        messages,
        system_prompt=system_prompt or "",
        tools=tools,
        charge_stale_thinking=False,
    )


def _agent_stale_thinking_on_wire(agent: Any) -> bool:
    """Whether the agent's active route replays stale thinking text (#84371).

    Returns ``True`` (conservative full charge) when route facts are unavailable."""
    try:
        from agent.message_sanitization import stale_thinking_reaches_wire

        return stale_thinking_reaches_wire(
            getattr(agent, "api_mode", "") or "",
            getattr(agent, "provider", "") or "",
            getattr(agent, "model", "") or "",
            getattr(agent, "base_url", "") or "",
        )
    except Exception:
        return True


def compose_user_api_content(
    content: Any,
    ext_prefetch_cache: str,
    plugin_user_context: str,
) -> Optional[str]:
    """Compose the API-bound content of the current turn's user message.

    Single source for the ``api_content`` sidecar and the wire bytes, so they never
    drift — the prompt-cache invariant: what turn N sends is what turn N+1 replays.
    Returns ``None`` when nothing is injected (message is sent as-is)."""
    if not isinstance(content, str):
        return None
    injections = []
    if ext_prefetch_cache:
        fenced = build_memory_context_block(ext_prefetch_cache)
        if fenced:
            injections.append(fenced)
    if plugin_user_context:
        injections.append(plugin_user_context)
    if not injections:
        return None
    return content + "\n\n" + "\n\n".join(injections)


def substitute_api_content(api_msg: Dict[str, Any]) -> Optional[str]:
    """Pop the ``api_content`` sidecar and substitute it into ``content``.

    Keeps the provider prompt-cache prefix byte-stable across turns.
    Returns the popped sidecar string, or ``None`` when absent."""
    sidecar = api_msg.pop("api_content", None)
    if (
        isinstance(sidecar, str)
        and sidecar
        and api_msg.get("role") in ("user", "assistant")
    ):
        api_msg["content"] = sidecar
    return sidecar


def drop_stale_api_content(msg: Dict[str, Any]) -> None:
    """Drop the ``api_content`` sidecar from a message whose content was rewritten.

    Replaying it would resend what the rewrite removed; cost is one cache miss."""
    msg.pop("api_content", None)


def extract_api_content_sidecar(msg: Mapping[str, Any]) -> Optional[str]:
    """Extract the ``api_content`` sidecar; ``None`` when absent/non-string."""
    v = msg.get("api_content")
    return v if isinstance(v, str) else None


def consume_gateway_turn_context_notes(agent: Any) -> str:
    """Pop the gateway's per-turn must-deliver notes off the agent (one-shot).

    Staged on ``agent._gateway_turn_context_notes``; consuming them keeps the system
    prompt byte-stable and prevents a cached agent replaying a stale note."""
    notes = getattr(agent, "_gateway_turn_context_notes", "") or ""
    if hasattr(agent, "_gateway_turn_context_notes"):
        try:
            agent._gateway_turn_context_notes = ""
        except Exception:
            pass
    return notes if isinstance(notes, str) else ""


def append_notes_to_multimodal_content(content: Any, notes: str) -> bool:
    """Deliver must-deliver notes on a multimodal (list) user message.

    Appends a durable text part in place, since the sidecar path returns ``None``
    for non-string content. Returns ``True`` when a part was appended."""
    if not notes or not isinstance(content, list):
        return False
    try:
        content.append({"type": "text", "text": notes})
        return True
    except Exception:
        return False


# Surfaces whose sessions must not be auto-titled: cron names its own session and
# its opener is a delivery hint; subagent sessions are hidden from every picker.
_UNTITLED_PLATFORMS = frozenset({"cron", "subagent"})


def _maybe_title_session_at_turn_start(agent: Any, messages: List[Any]) -> None:
    """Kick off auto-titling for the session's first user message; never fatal."""
    session_db = getattr(agent, "_session_db", None)
    session_id = getattr(agent, "session_id", None)
    if not session_db or not session_id:
        return

    if str(getattr(agent, "platform", "") or "").lower() in _UNTITLED_PLATFORMS:
        return

    try:
        from agent.message_content import flatten_message_text
        from agent.title_generator import maybe_auto_title

        # Turn's user message as text; image-only turns yield "" and are skipped.
        user_text = ""
        for msg in reversed(messages or []):
            if isinstance(msg, dict) and msg.get("role") == "user":
                user_text = flatten_message_text(msg.get("content")).strip()
                break
        if not user_text:
            return

        # Session row is created lazily later; force it now or the title write matches
        # zero rows.
        if not getattr(agent, "_session_db_created", False):
            ensure = getattr(agent, "_ensure_db_session", None)
            if callable(ensure):
                ensure()
            if not getattr(agent, "_session_db_created", False):
                return

        # Snapshot runtime identity so the background titler can skip if the user
        # switches models before it fires (#19027).
        _model = getattr(agent, "model", None)
        _provider = getattr(agent, "provider", None)

        maybe_auto_title(
            session_db,
            session_id,
            user_text,
            conversation_history=messages,
            failure_callback=(
                getattr(agent, "_title_failure_callback", None)
                or getattr(agent, "_emit_auxiliary_failure", None)
            ),
            main_runtime={
                "model": _model,
                "provider": _provider,
                "base_url": getattr(agent, "base_url", None),
                "api_key": getattr(agent, "api_key", None),
                "api_mode": getattr(agent, "api_mode", None),
            },
            title_callback=getattr(agent, "_on_session_title", None),
            runtime_validator=lambda: (
                getattr(agent, "model", None) == _model
                and getattr(agent, "provider", None) == _provider
            ),
        )
    except Exception:
        logger.debug("Turn-start auto-title dispatch failed", exc_info=True)


def reanchor_current_turn_user_idx(messages: List[Any], user_message: Any) -> int:
    """Locate this turn's user message after compaction rebuilt ``messages``.

    Prefers the LAST user message whose content exactly matches this turn's text, else
    the last user-originated turn; compaction handoffs are never the fallback (#80622).
    Returns -1 when there is no user-originated message."""
    from agent.context_compressor import user_originated_turn_view

    fallback = -1
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if not (isinstance(msg, dict) and msg.get("role") == "user"):
            continue
        # Typed synthetic current events keep their persistence anchor when raw
        # content is unchanged; not eligible for the human-only fallback below.
        if msg.get("content") == user_message:
            return i
        live_view = user_originated_turn_view(msg)
        if live_view is None:
            continue
        if live_view.get("content") == user_message:
            return i
        # Prefer a real human turn over a synthetic handoff / continuation
        # marker when the exact content was rewritten by merge-into-tail.
        if fallback < 0:
            fallback = i
    return fallback


def compression_made_progress(
    orig_len: int, new_len: int, orig_tokens: int, new_tokens: int
) -> bool:
    """Return ``True`` if a compression pass materially reduced the request.

    Counts a >5% token reduction as progress even when the row count is unchanged
    (size-only wins, #39548); same floor as the overflow-handler retry path."""
    if new_len < orig_len:
        return True
    return orig_tokens > 0 and new_tokens < orig_tokens * 0.95


# Back-compat alias: gateway callers and tests patch ``_compression_made_progress``
# (#79624).
_compression_made_progress = compression_made_progress


class PreflightCompressionTimedOut(RuntimeError):
    """Raised when an oversized turn cannot safely finish preflight."""


def _fail_closed_after_preflight_timeout(agent, request_tokens: int) -> None:
    """Stop an oversized turn instead of sending its unchanged provider payload."""
    from agent.conversation_compression import context_compression_timed_out

    if not context_compression_timed_out(agent):
        return
    raise PreflightCompressionTimedOut(
        "Context compression timed out before it could commit while the request "
        f"was still approximately {request_tokens:,} tokens. The provider call "
        "was not sent. Run /compress and wait for it to finish, then retry."
    )


def _review_fork_first_request_pending(agent: Any) -> bool:
    """Whether a detached review fork has yet to send its first provider request.

    The fork replays the parent's FULL snapshot as a warm cache read, so compaction must
    wait until that first response arrives. Dormant without the attribute (#93057)."""
    return bool(
        getattr(agent, "_review_defer_compaction_before_first_response", False)
        and not getattr(agent, "_turn_received_provider_response", False)
    )


def _compression_warrants_another_preflight_pass(
    orig_tokens: int, new_tokens: int, threshold_tokens: int
) -> bool:
    """Whether an over-threshold request merits another immediate summary.

    Continue only if still over threshold AND the previous pass cut tokens by >5%."""
    return (
        new_tokens >= threshold_tokens
        and orig_tokens > 0
        and new_tokens < orig_tokens * 0.95
    )


def _should_run_preflight_estimate(
    messages: List[Dict[str, Any]],
    protect_first_n: int,
    protect_last_n: int,
    threshold_tokens: int,
) -> bool:
    """Cheap gate for the (expensive) full preflight token estimate.

    ``True`` when message count exceeds the protected ranges OR a rough char-based
    estimate crosses the threshold — the few-but-huge case (#27405). The estimator
    undercounts by design (omits system/tools) so one large base64 image is not
    mistaken for ~250K tokens."""
    if len(messages) > protect_first_n + protect_last_n + 1:
        return True
    return estimate_messages_tokens_rough(messages) >= threshold_tokens


def _should_idle_compact(
    *,
    enabled: bool,
    idle_after_seconds: int,
    idle_gap_seconds: float,
    tokens: int,
    floor_tokens: int,
    cooldown_active: bool,
) -> bool:
    """Decide whether an idle-triggered compaction should run this turn.

    Fires after a wall-clock gap of ``idle_after_seconds`` (opt-in, <= 0 disables),
    independent of ``threshold_tokens``; skips at/below ``floor_tokens`` and during a
    compression-failure cooldown. Pure predicate."""
    if not enabled or idle_after_seconds <= 0:
        return False
    if idle_gap_seconds < idle_after_seconds:
        return False
    if cooldown_active:
        return False
    return tokens > floor_tokens


@dataclass
class TurnContext:
    """Values produced by the turn prologue and consumed by the turn loop."""

    # Sanitized inbound message (surrogates stripped).
    user_message: str
    # Clean message preserved for transcripts / memory queries (no nudge injection).
    original_user_message: Any
    # Working message list for this turn (loop appends to it).
    messages: List[Dict[str, Any]]
    # May be reset to None by preflight compression (new session created).
    conversation_history: Optional[List[Dict[str, Any]]]
    # Cached system prompt active for this turn (may be rebuilt by compression).
    active_system_prompt: Optional[str]
    # Task / turn identifiers.
    effective_task_id: str
    turn_id: str
    # Index of the current user turn within ``messages``.
    current_turn_user_idx: int
    # Whether the post-turn memory review should fire.
    should_review_memory: bool = False
    # Context contributed by ``pre_llm_call`` plugins (appended to user message).
    plugin_user_context: str = ""
    # External-memory prefetch result, reused across loop iterations.
    ext_prefetch_cache: str = ""
    # Turn-start preflight already proved an immediate retry ineffective.
    preflight_compression_blocked: bool = False


def build_turn_context(
    agent,
    user_message: Any,
    system_message: Optional[str],
    conversation_history: Optional[List[Dict[str, Any]]],
    task_id: Optional[str],
    stream_callback,
    persist_user_message: Optional[Any],
    persist_user_timestamp: Optional[float] = None,
    persist_user_platform_id: Optional[str] = None,
    *,
    persist_user_display_kind: Optional[str] = None,
    persist_user_display_metadata: Optional[Dict[str, Any]] = None,
    restore_or_build_system_prompt,
    install_safe_stdio,
    sanitize_surrogates,
    summarize_user_message_for_log,
    set_session_context,
    set_current_write_origin,
    ra,
    moa_active: bool = False,
) -> TurnContext:
    """Run the once-per-turn setup and return the loop's input context.

    Helpers are passed in to avoid an import cycle with ``agent.conversation_loop``."""
    # Guard stdio against OSError from broken pipes (systemd/headless/daemon).
    install_safe_stdio()

    # Recover a rotated session before binding log/turn ids or copying client history so
    # everything in this turn belongs to the canonical child.
    recovered_history = recover_rotated_compression_session(agent)
    if recovered_history is not None:
        conversation_history = recovered_history

    # NOTE: the DB session row is created later, after the system prompt is built;
    # creating it now would persist system_prompt=NULL and cost a cache miss (#45499).

    # Tag log records on this thread with the session ID for ``hermes logs``.
    set_session_context(agent.session_id)

    # Bind the skill write-origin ContextVar for this thread.
    set_current_write_origin(getattr(agent, "_memory_write_origin", "assistant_tool"))

    # Restore the primary runtime if the previous turn activated fallback.
    agent._restore_primary_runtime()

    # Tell auxiliary_client what the live main provider/model are for this turn
    # after primary restoration has settled the runtime.
    try:
        from agent.auxiliary_client import set_runtime_main
        from agent.prompt_cache_scope import resolve_prompt_cache_scope_safe
        # Rotation-stable prompt-cache scope (lineage root), memoized per segment; a
        # new session uses the physical id until build_api_kwargs re-resolves (#79017).
        # Never-raising variant, outside the argument list: failure loses only scope.
        _cache_scope = resolve_prompt_cache_scope_safe(agent) or ""
        set_runtime_main(
            getattr(agent, "provider", "") or "",
            getattr(agent, "model", "") or "",
            requested_provider=getattr(agent, "requested_provider", "") or "",
            base_url=getattr(agent, "base_url", "") or "",
            api_key=getattr(agent, "api_key", "") or "",
            api_mode=getattr(agent, "api_mode", "") or "",
            auth_mode=getattr(agent, "auth_mode", "") or "",
            session_id=getattr(agent, "session_id", "") or "",
            cache_scope=_cache_scope,
        )
    except Exception:
        pass

    # Between-turns MCP refresh: late-connecting servers land in THIS turn's snapshot,
    # before the first API call assembles ``tools=``. ``preserve_prefix`` keeps the
    # tool array append-only so a flapping ``check_fn`` can't fork the cache (#100336).
    try:
        if not getattr(agent, "_skip_mcp_refresh", False):
            # Import-cost gate: MCP tools are only registered by code that already
            # imported ``tools.mcp_tool`` (~0.4s); not in sys.modules => nothing to do.
            import sys as _sys
            if "tools.mcp_tool" in _sys.modules:
                from tools.mcp_tool import has_registered_mcp_tools, refresh_agent_mcp_tools
                if has_registered_mcp_tools():
                    refresh_agent_mcp_tools(
                        agent, quiet_mode=True, preserve_prefix=True,
                    )
    except Exception:
        logger.debug("between-turns MCP tool refresh skipped", exc_info=True)

    # Sanitize surrogate characters from user input.
    if isinstance(user_message, str):
        user_message = sanitize_surrogates(user_message)
    if isinstance(persist_user_message, str):
        persist_user_message = sanitize_surrogates(persist_user_message)

    # Store stream callback for _interruptible_api_call to pick up.
    agent._stream_callback = stream_callback
    agent._persist_user_message_idx = None
    agent._persist_user_message_override = persist_user_message
    agent._persist_user_message_timestamp = persist_user_timestamp
    agent._persist_user_message_platform_id = persist_user_platform_id
    # Generate unique task_id if not provided to isolate VMs between tasks.
    effective_task_id = task_id or str(uuid.uuid4())
    agent._current_task_id = effective_task_id
    turn_id = str(getattr(agent, "_relay_pending_turn_id", "") or "")
    if not turn_id:
        turn_id = (
            f"{agent.session_id or 'session'}:{effective_task_id}:{uuid.uuid4().hex[:8]}"
        )
    agent._relay_pending_turn_id = None
    agent._current_turn_id = turn_id
    agent._current_api_request_id = ""
    # Tripwire: warn when this turn starts before the previous turn-end persist
    # (concurrent turns interleave transcript writes). Cleared in _persist_session.
    from agent.agent_runtime_helpers import note_turn_start
    note_turn_start(agent, turn_id)

    # Reset retry counters and iteration budget at the start of each turn.
    agent._invalid_tool_retries = 0
    agent._invalid_json_retries = 0
    agent._empty_content_retries = 0
    agent._incomplete_scratchpad_retries = 0
    agent._codex_incomplete_retries = 0
    agent._thinking_prefill_retries = 0
    agent._post_tool_empty_retried = False
    agent._last_content_with_tools = None
    agent._last_content_tools_all_housekeeping = False
    agent._mute_post_response = False
    agent._unicode_sanitization_passes = 0
    agent._tool_guardrails.reset_for_turn()
    agent._tool_guardrail_halt_decision = None
    _reset_consol = getattr(agent._memory_store, "reset_consolidation_failures", None)
    if callable(_reset_consol):
        _reset_consol()
    agent._vision_supported = True

    # Pre-turn connection health check: clean up dead TCP connections.
    if agent.api_mode != "anthropic_messages":
        try:
            if agent._cleanup_dead_connections():
                agent._emit_status(
                    "🔌 Detected stale connections from a previous provider "
                    "issue — cleaned up automatically. Proceeding with fresh "
                    "connection."
                )
        except Exception:
            pass
    # Replay compression warning through status_callback for gateway platforms.
    if agent._compression_warning:
        agent._replay_compression_warning()
        agent._compression_warning = None  # send once

    # NOTE: _turns_since_memory and _iters_since_skill are NOT reset here.
    agent.iteration_budget = IterationBudget(agent.max_iterations)

    # Wall-clock run budget: stamped only when configured; the wrap-up latch resets per
    # turn (one notice per run).
    if getattr(agent, "run_budget_seconds", None):
        agent._run_budget_started_at = time.time()
    else:
        agent._run_budget_started_at = None
    agent._run_budget_wrapup_injected = False

    # Log conversation turn start for debugging/observability.
    _preview_text = summarize_user_message_for_log(user_message)
    _msg_preview = (_preview_text[:80] + "...") if len(_preview_text) > 80 else _preview_text
    _msg_preview = _msg_preview.replace("\n", " ")
    logger.info(
        "conversation turn: session=%s model=%s provider=%s platform=%s history=%d msg=%r",
        agent.session_id or "none", agent.model, agent.provider or "unknown",
        agent.platform or "unknown", len(conversation_history or []),
        _msg_preview,
    )

    # Initialize conversation (copy to avoid mutating the caller's list).
    messages = list(conversation_history) if conversation_history else []

    # Reuse CLI-staged input only when its clean text matches this turn; a stale
    # handoff must not replace later input. Voice turns compare the clean override.
    pending_cli_message = getattr(agent, "_pending_cli_user_message", None)
    expected_persist_content = (
        persist_user_message if persist_user_message is not None else user_message
    )
    if (
        isinstance(pending_cli_message, dict)
        and pending_cli_message.get("content") == expected_persist_content
    ):
        user_msg = pending_cli_message
        # CLI-staged value is the clean text; restore the API-facing variant (e.g. voice
        # prefix) on the same dict, keeping any close-path durable marker.
        user_msg["content"] = user_message
    else:
        user_msg = stamp_message_timestamp(
            {"role": "user", "content": user_message},
            timestamp=persist_user_timestamp,
        )
        if isinstance(pending_cli_message, dict):
            agent._pending_cli_user_message = None
    # CLI input is stamped when staged. Gateway input may carry the platform
    # event time. Preserve either value and cover any legacy unstamped handoff.
    stamp_message_timestamp(user_msg, timestamp=persist_user_timestamp)

    # Hydrate todo store from conversation history.
    if conversation_history and not agent._todo_store.has_items():
        agent._hydrate_todo_store(conversation_history)

    # Hydrate per-session nudge counters from persisted history (issue #22357).
    if conversation_history and agent._user_turn_count == 0:
        prior_user_turns = sum(
            1 for m in conversation_history if m.get("role") == "user"
        )
        if prior_user_turns > 0:
            agent._user_turn_count = prior_user_turns
            if agent._memory_nudge_interval > 0 and agent._turns_since_memory == 0:
                agent._turns_since_memory = prior_user_turns % agent._memory_nudge_interval

    # Append the user message now that close persistence is safe. Synthesized turns
    # stamp their transcript type so the crash persist writes a typed row; the model
    # still receives role/content unchanged (api_messages strips both fields).
    if persist_user_display_kind:
        user_msg["display_kind"] = persist_user_display_kind
        if persist_user_display_metadata:
            user_msg["display_metadata"] = persist_user_display_metadata

    # Stamp the platform message id so it survives the turn-start flush; restart
    # drain-window recovery dedups via ``has_platform_message_id`` against this row.
    if persist_user_platform_id is not None:
        user_msg["platform_message_id"] = persist_user_platform_id
    append_message(messages, user_msg)
    current_turn_user_idx = len(messages) - 1
    agent._persist_user_message_idx = current_turn_user_idx

    # Track user turns for memory flush and periodic nudge logic.
    agent._user_turn_count += 1
    # Copilot x-initiator: the first API call of this user turn is
    # user-initiated; tool-loop follow-ups revert to "agent" (#3040).
    agent._is_user_initiated_turn = True

    # Reset the streaming context scrubber at the top of each turn.
    scrubber = getattr(agent, "_stream_context_scrubber", None)
    if scrubber is not None:
        scrubber.reset()
    # Reset the think scrubber for the same reason.
    think_scrubber = getattr(agent, "_stream_think_scrubber", None)
    if think_scrubber is not None:
        think_scrubber.reset()

    # Preserve the original user message (no nudge injection).
    original_user_message = persist_user_message if persist_user_message is not None else user_message

    # Track memory nudge trigger (turn-based, checked here).
    should_review_memory = False
    if (agent._memory_nudge_interval > 0
            and "memory" in agent.valid_tool_names
            and agent._memory_store):
        agent._turns_since_memory += 1
        if agent._turns_since_memory >= agent._memory_nudge_interval:
            should_review_memory = True
            agent._turns_since_memory = 0

    # Cosmetic side-signal: detect an affection reaction so the host can play hearts.
    # Token-free, never touches the conversation, never fatal.
    reaction_callback = getattr(agent, "reaction_callback", None)
    if reaction_callback is not None:
        try:
            from agent.reactions import detect_reaction

            kind = detect_reaction(original_user_message)
            if kind:
                reaction_callback(kind)
        except Exception:
            pass

    if not agent.quiet_mode:
        _print_preview = summarize_user_message_for_log(user_message)
        agent._safe_print(
            f"💬 Starting conversation: '{_print_preview[:60]}"
            f"{'...' if len(_print_preview) > 60 else ''}'"
        )

    # ── System prompt (cached per session for prefix caching) ──
    if agent._cached_system_prompt is None:
        restore_or_build_system_prompt(agent, system_message, conversation_history)

    active_system_prompt = agent._cached_system_prompt

    # Bot Mode DM tool — injected ONLY into a bot's canonical "Bot Chat" session
    # (same gate as the protocol section); gate is session-stable, so cache-safe.
    try:
        from tools.bot_mode_dm import ensure_message_agent_tool

        ensure_message_agent_tool(agent)
    except Exception:
        logger.debug("message_agent injection skipped", exc_info=True)

    # Create the DB row now (system prompt populated => non-NULL, #45499) and BEFORE
    # preflight compression: compaction/rotation INSERTs reference this row under
    # PRAGMA foreign_keys=ON. Idempotent; the user-turn crash persist runs later.
    persist_lock = getattr(agent, "_session_persist_lock", None)
    try:
        if persist_lock is None:
            agent._ensure_db_session()
        else:
            with persist_lock:
                agent._ensure_db_session()
    except Exception:
        logger.warning(
            "Turn-start session row creation failed for session=%s",
            agent.session_id or "none",
            exc_info=True,
        )
    finally:
        # Clear staged CLI input eagerly so a crash in preflight compression doesn't
        # leave a stale _pending_cli_user_message for the next turn.
        if not isinstance(pending_cli_message, dict) or pending_cli_message.get("_db_persisted"):
            agent._pending_cli_user_message = None

    # ── Idle-triggered compaction (opt-in; ``idle_compact_after_seconds``) ──
    # Fires on wall-clock gap since ``_last_activity_ts``, complementing the token
    # gate; a cheap gap check gates the estimate (cf. _should_run_preflight_estimate).
    _idle_after = getattr(agent, "compression_idle_compact_after_seconds", 0)
    if agent.compression_enabled and _idle_after > 0 and messages:
        _idle_gap = time.time() - getattr(agent, "_last_activity_ts", time.time())
        if _idle_gap >= _idle_after:
            _compressor = agent.context_compressor
            # Route-aware pressure: on compacted native-Codex sessions the durable
            # figure overstates the wire; reuse the preflight estimator (#96995).
            _idle_tokens = _preflight_request_tokens(
                agent,
                messages,
                active_system_prompt or "",
            )
            # Post-compression target size: don't summarise a thread already
            # below what compaction would reduce it to.
            _idle_floor = int(
                _compressor.threshold_tokens * _compressor.summary_target_ratio
            )
            _idle_cooldown = getattr(
                _compressor, "get_active_compression_failure_cooldown", lambda: None
            )()
            if _should_idle_compact(
                enabled=agent.compression_enabled,
                idle_after_seconds=_idle_after,
                idle_gap_seconds=_idle_gap,
                tokens=_idle_tokens,
                floor_tokens=_idle_floor,
                cooldown_active=bool(_idle_cooldown),
            ):
                logger.info(
                    "Idle compaction: %ss idle >= %ss, ~%s tokens > %s floor "
                    "(session %s)",
                    int(_idle_gap),
                    _idle_after,
                    f"{_idle_tokens:,}",
                    f"{_idle_floor:,}",
                    agent.session_id or "none",
                )
                _idle_status = automatic_compaction_status_message(
                    _compressor,
                    phase="idle",
                    default_message=IDLE_COMPACTION_STATUS_TEMPLATE.format(
                        idle_seconds=int(_idle_gap), tokens=_idle_tokens
                    ),
                    approx_tokens=_idle_tokens,
                    idle_seconds=int(_idle_gap),
                    model=agent.model,
                )
                if _idle_status:
                    agent._emit_status(_idle_status)
                _idle_input = messages
                messages, active_system_prompt = agent._compress_context(
                    messages, system_message, approx_tokens=_idle_tokens,
                    task_id=effective_task_id,
                )
                # ``_compress_context`` returns the INPUT list object when it skips;
                # only re-baseline and re-anchor after a real compaction.
                if messages is not _idle_input:
                    conversation_history = conversation_history_after_compression(
                        agent, messages, conversation_history
                    )
                    # Compaction rebuilt the list; re-anchor this turn's user index.
                    current_turn_user_idx = reanchor_current_turn_user_idx(
                        messages, user_message
                    )
                    agent._persist_user_message_idx = current_turn_user_idx

    # ── Preflight context compression ──
    # Cheap pre-check gates the full estimate; see ``_should_run_preflight_estimate``
    # for the OR semantics (#27405).
    _preflight_compressed = False
    _preflight_compression_blocked = False
    agent._turn_received_provider_response = False
    agent._turn_preflight_display_snapshot = None
    if (
        agent.compression_enabled
        and not _review_fork_first_request_pending(agent)
        and _should_run_preflight_estimate(
            messages,
            agent.context_compressor.protect_first_n,
            agent.context_compressor.protect_last_n,
            agent.context_compressor.threshold_tokens,
        )
    ):
        _preflight_tokens = _preflight_request_tokens(
            agent,
            messages,
            active_system_prompt or "",
        )
        _compressor = agent.context_compressor
        # getattr guard: compressor doubles and plugin engines lack this method —
        # absence means no snapshot and the finalizer's rollback stays disarmed.
        _snapshot_fn = getattr(
            _compressor, "snapshot_preflight_display_tokens", None
        )
        if callable(_snapshot_fn):
            _snapshot_val = _snapshot_fn()
            # Type pin: MagicMock compressors return truthy Mock objects —
            # only a real int snapshot may arm the interrupted-turn rollback.
            if isinstance(_snapshot_val, int) and not isinstance(
                _snapshot_val, bool
            ):
                agent._turn_preflight_display_snapshot = _snapshot_val
        _defer_preflight = getattr(
            _compressor,
            "should_defer_preflight_to_real_usage",
            lambda _tokens: False,
        )
        _preflight_deferred = _defer_preflight(_preflight_tokens)
        # Codex app-server threads are compacted by the codex agent itself;
        # Hermes only initiates compaction in "hermes" mode (#36801).
        _codex_native_auto = (
            getattr(agent, "api_mode", None) == "codex_app_server"
            and str(
                getattr(
                    agent,
                    "codex_app_server_auto_compaction",
                    "native",
                )
                or "native"
            ).lower()
            in {"native", "off"}
        )

        if not _preflight_deferred:
            # Display-only seed: a real provider reading wins and the -1 sentinel
            # stays protected (#36718). Also feeds the tool-loop gate on usage-less
            # responses.
            _maybe_seed = getattr(
                _compressor, "maybe_seed_preflight_display_tokens", None
            )
            if callable(_maybe_seed):
                _maybe_seed(_preflight_tokens)

        _compression_cooldown = getattr(
            _compressor,
            "get_active_compression_failure_cooldown",
            lambda: None,
        )()

        _should_compress_now = False
        _compress_block_reason = None
        if _preflight_deferred:
            logger.info(
                "Skipping preflight compression: rough estimate ~%s >= %s, "
                "but last real provider prompt was %s after compression",
                f"{_preflight_tokens:,}",
                f"{_compressor.threshold_tokens:,}",
                f"{_compressor.last_real_prompt_tokens:,}",
            )
        elif _compression_cooldown:
            logger.info(
                "Skipping preflight compression: same-session cooldown active "
                "(~%s seconds remaining, session %s)",
                int(_compression_cooldown.get("remaining_seconds", 0.0)),
                agent.session_id or "none",
            )
            if _preflight_tokens >= _compressor.threshold_tokens:
                # Context is over threshold but compression is blocked by the
                # summary-LLM cooldown — surface a warning (see block below).
                _cooldown_secs = _compression_cooldown.get("remaining_seconds", 0.0)
                _compress_block_reason = f"cooldown:{_cooldown_secs:.0f}"
        elif _codex_native_auto:
            logger.info(
                "Skipping Hermes preflight compression for codex app-server "
                "(mode=%s); Hermes will not start thread compaction here.",
                getattr(agent, "codex_app_server_auto_compaction", "native"),
            )
        else:
            _should_compress_now = _compressor.should_compress(_preflight_tokens)
            if not _should_compress_now:
                # Over threshold but blocked: ask should_compress_info for the reason
                # to surface below. getattr guard: doubles/older engines lack it.
                _info = getattr(_compressor, "should_compress_info", None)
                if callable(_info):
                    try:
                        _compress_block_reason = _info(_preflight_tokens)[1]
                    except Exception:
                        _compress_block_reason = None
        if _should_compress_now:
            # Managed local runtime: growing the window beats compressing (ladder
            # order; same seam as _maybe_grow_local_window in the loop).
            try:
                from agent.conversation_loop import _maybe_grow_local_window

                _grown = _maybe_grow_local_window(
                    agent, _compressor, _preflight_tokens
                )
            except Exception:
                _grown = None
            if _grown:
                _compressor.update_model(
                    agent.model,
                    _grown,
                    base_url=getattr(agent, "base_url", "") or "",
                    api_key=getattr(agent, "api_key", "") or "",
                    provider=getattr(agent, "provider", "") or "",
                    api_mode=getattr(agent, "api_mode", "") or "",
                )
                agent._buffer_status(
                    f"📈 Context window grown to {_grown // 1024}K "
                    f"(local model; conversation continues uncompressed)"
                )
                _should_compress_now = _compressor.should_compress(
                    _preflight_tokens
                )
        if _should_compress_now:
            _preflight_compressed = True
            # Compression is actually running — reset the dedup so a future blocked
            # turn can warn again. getattr guard: object.__new__ doubles lack it.
            _clear_warn = getattr(agent, "_clear_context_overflow_warn", None)
            if callable(_clear_warn):
                _clear_warn()
            logger.info(
                "Preflight compression: ~%s tokens >= %s threshold (model %s, ctx %s)",
                f"{_preflight_tokens:,}",
                f"{_compressor.threshold_tokens:,}",
                agent.model,
                f"{_compressor.context_length:,}",
            )
            _preflight_status = automatic_compaction_status_message(
                _compressor,
                phase="preflight",
                default_message=PREFLIGHT_COMPRESSION_STATUS_TEMPLATE.format(
                    tokens=_preflight_tokens,
                    threshold=_compressor.threshold_tokens,
                ),
                approx_tokens=_preflight_tokens,
                threshold_tokens=_compressor.threshold_tokens,
                context_length=_compressor.context_length,
                model=agent.model,
            )
            if _preflight_status:
                agent._emit_status(_preflight_status)
            # Preflight passes honor compression.max_attempts like the loop's sites
            # (default 3).
            _max_preflight_passes = max(
                1, int(getattr(agent, "max_compression_attempts", 3) or 3)
            )
            for _pass in range(_max_preflight_passes):
                _orig_len = len(messages)
                _orig_tokens = _preflight_tokens
                _preflight_input = messages
                messages, active_system_prompt = agent._compress_context(
                    messages, system_message, approx_tokens=_preflight_tokens,
                    task_id=effective_task_id,
                )
                if (
                    messages is _preflight_input
                    and compression_skipped_due_to_lock(agent)
                ):
                    # Lock-skip (#69870): another path holds the lock, so this is a
                    # DEFER, not proof of incompressibility — don't arm the blocker;
                    # stop preflight passes for this turn.
                    logger.info(
                        "Preflight compression deferred: compression lock "
                        "held by another path (session %s)",
                        agent.session_id or "none",
                    )
                    break
                # Re-estimate so size-only compression (same rows, fewer tokens)
                # counts as progress (#39548).
                _preflight_tokens = _preflight_request_tokens(
                    agent,
                    messages,
                    active_system_prompt or "",
                )
                if not _compression_made_progress(
                    _orig_len, len(messages), _orig_tokens, _preflight_tokens
                ):
                    _fail_closed_after_preflight_timeout(agent, _preflight_tokens)
                    _preflight_compression_blocked = True
                    break  # Cannot compress further: neither rows nor tokens moved
                conversation_history = conversation_history_after_compression(
                    agent, messages, conversation_history
                )
                agent._empty_content_retries = 0
                agent._thinking_prefill_retries = 0
                agent._last_content_with_tools = None
                agent._last_content_tools_all_housekeeping = False
                agent._mute_post_response = False
                if not _compressor.should_compress(_preflight_tokens):
                    break
                if not _compression_warrants_another_preflight_pass(
                    _orig_tokens,
                    _preflight_tokens,
                    _compressor.threshold_tokens,
                ):
                    _preflight_compression_blocked = True
                    logger.warning(
                        "Preflight compression made insufficient progress: "
                        "~%s -> ~%s request tokens; skipping additional passes",
                        f"{_orig_tokens:,}",
                        f"{_preflight_tokens:,}",
                    )
                    break
        elif _compress_block_reason:
            # Over threshold but compression blocked: surface a deduped warning so
            # the user can /new or /compress instead of a silent provider limit.
            agent._warn_context_overflow_blocked(
                _compress_block_reason,
                _preflight_tokens,
                _compressor.threshold_tokens,
            )
        else:
            # Sub-threshold and unblocked — re-arm the overflow warning. getattr guard:
            # object.__new__ test doubles lack the method.
            _clear_warn = getattr(agent, "_clear_context_overflow_warn", None)
            if callable(_clear_warn):
                _clear_warn()
            # Engine maintenance only when NO skip-branch fired: cooldown, deferred
            # estimate, or codex-native route keep the engine hook unconsulted (#20316).
            if _compression_cooldown or _preflight_deferred or _codex_native_auto:
                _engine_preflight = None
            else:
                _engine_preflight = getattr(
                    _compressor, "should_compress_preflight", None
                )
            # ── Engine-driven sub-threshold preflight maintenance (#20316) ──
            # Engines overriding ``should_compress_preflight()`` get exactly ONE
            # ``compress()`` pass; a no-op never touches _preflight_compression_blocked.
            _wants_engine_preflight = False
            if callable(_engine_preflight):
                try:
                    _wants_engine_preflight = bool(_engine_preflight(messages))
                except Exception as _preflight_exc:
                    # A buggy engine must never break an otherwise-healthy
                    # turn: swallow at debug level and skip maintenance.
                    logger.debug(
                        "should_compress_preflight raised %s; skipping "
                        "engine-driven preflight maintenance",
                        _preflight_exc,
                    )
                    _wants_engine_preflight = False
            if _wants_engine_preflight:
                logger.info(
                    "Engine-driven preflight maintenance: %s requested "
                    "compress() at ~%s tokens (below %s threshold)",
                    getattr(_compressor, "name", type(_compressor).__name__),
                    f"{_preflight_tokens:,}",
                    f"{getattr(_compressor, 'threshold_tokens', 0):,}",
                )
                _engine_input = messages
                messages, active_system_prompt = agent._compress_context(
                    messages, system_message, approx_tokens=_preflight_tokens,
                    task_id=effective_task_id,
                )
                # ``_compress_context`` returns the INPUT list on every skip path and an
                # engine may no-op; re-baseline/re-anchor only after a REAL compaction.
                if messages is not _engine_input:
                    _preflight_compressed = True
                    conversation_history = conversation_history_after_compression(
                        agent, messages
                    )
                    agent._empty_content_retries = 0
                    agent._thinking_prefill_retries = 0
                    agent._last_content_with_tools = None
                    agent._last_content_tools_all_housekeeping = False
                    agent._mute_post_response = False
    elif not agent.compression_enabled:
        # Uncompressed session guard (#89297): the warning fires from the loop's
        # pre-API site; here we only RE-ARM the dedup once back under the window.
        _ctx_len = getattr(
            getattr(agent, "context_compressor", None), "context_length", None
        )
        if isinstance(_ctx_len, int) and _ctx_len > 0:
            _raw_chars = 0
            for _m in messages:
                if not isinstance(_m, dict):
                    continue
                _c = _m.get("content")
                if isinstance(_c, str):
                    _raw_chars += len(_c)
                elif _c:
                    # Non-string, non-empty content defeats a char count — force the
                    # real estimate. None/"" contribute nothing.
                    _raw_chars = _ctx_len + 1
                    break
            # Cheap gate: raw text under ~1/4 of the window (4 chars/token) cannot
            # be over it; non-string (multimodal) content forces the real estimate.
            if _raw_chars <= _ctx_len:
                _clear_warn = getattr(
                    agent, "_clear_context_overflow_warn", None
                )
                if callable(_clear_warn):
                    _clear_warn()
            else:
                # Re-arm with the same route-aware (checkpoint-pruned wire) figure the
                # warn site measures, else a compacted session never clears the dedup
                # and genuine overflow warnings stay suppressed (#96995/#97602).
                _uncompressed_tokens = _preflight_request_tokens(
                    agent,
                    messages,
                    active_system_prompt or "",
                )
                if _uncompressed_tokens <= _ctx_len:
                    _clear_warn = getattr(
                        agent, "_clear_context_overflow_warn", None
                    )
                    if callable(_clear_warn):
                        _clear_warn()

    if _preflight_compressed:
        # Compression rebuilt the list, so the pre-compression user index is stale.
        # Re-anchor so the api_content stamp, injection site, and persist-override row
        # hit the same dict; exact-content match first so a todo-snapshot can't steal it
        current_turn_user_idx = reanchor_current_turn_user_idx(
            messages, user_message
        )
        agent._persist_user_message_idx = current_turn_user_idx

    # Plugin hook: pre_llm_call (context injected into user message, not system prompt).
    plugin_user_context = ""
    try:
        from hermes_cli.lifecycle import invoke_hook as _invoke_hook
        _pre_results = _invoke_hook(
            "pre_llm_call",
            session_id=agent.session_id,
            task_id=effective_task_id,
            turn_id=turn_id,
            user_message=original_user_message,
            conversation_history=list(messages),
            is_first_turn=(not bool(conversation_history)),
            model=agent.model,
            platform=getattr(agent, "platform", None) or "",
            parent_session_id=getattr(agent, "_parent_session_id", None) or "",
            sender_id=getattr(agent, "_user_id", None) or "",
        )
        _ctx_parts: list[str] = []
        # Spill oversized per-hook context to disk so a runaway plugin can't inflate
        # every subsequent turn's prompt.
        try:
            from tools.hook_output_spill import (
                get_spill_config as _spill_cfg,
                spill_if_oversized as _spill_if_oversized,
            )
            _spill_config_cached = _spill_cfg()
        except Exception:
            _spill_if_oversized = None  # type: ignore[assignment]
            _spill_config_cached = None
        for r in _pre_results:
            _piece: str = ""
            if isinstance(r, dict) and r.get("context"):
                _piece = str(r["context"])
            elif isinstance(r, str) and r.strip():
                _piece = r
            else:
                continue
            if _spill_if_oversized is not None:
                try:
                    _piece = _spill_if_oversized(
                        _piece,
                        session_id=agent.session_id,
                        source="plugin hook",
                        config=_spill_config_cached,
                    )
                except Exception as _spill_exc:
                    logger.warning("hook context spill failed: %s", _spill_exc)
            _ctx_parts.append(_piece)
        if _ctx_parts:
            plugin_user_context = "\n\n".join(_ctx_parts)
    except Exception as exc:
        logger.warning("pre_llm_call hook failed: %s", exc)

    # Gateway must-deliver notes ride the user-message injection channel (one-shot,
    # gateway-staged) so the ephemeral system prompt stays byte-stable. Multimodal
    # (list) content can't take the string sidecar — append a durable text part.
    _gateway_notes = consume_gateway_turn_context_notes(agent)
    if _gateway_notes:
        _gw_turn_content = (
            messages[current_turn_user_idx].get("content")
            if 0 <= current_turn_user_idx < len(messages)
            and isinstance(messages[current_turn_user_idx], dict)
            else None
        )
        if isinstance(_gw_turn_content, list):
            append_notes_to_multimodal_content(_gw_turn_content, _gateway_notes)
        else:
            plugin_user_context = (
                plugin_user_context + "\n\n" + _gateway_notes
                if plugin_user_context
                else _gateway_notes
            )

    # Per-turn file-mutation verifier state.
    agent._turn_failed_file_mutations = {}
    agent._turn_file_mutation_paths = set()
    agent._verification_stop_nudges = 0
    agent._pre_verify_nudges = 0

    # Record the execution thread so interrupt()/clear_interrupt() can scope
    # the tool-level interrupt signal to THIS agent's thread only.
    agent._execution_thread_id = threading.current_thread().ident

    # Clear stale per-thread interrupt state, preserving a pending interrupt.
    ra()._set_interrupt(False, agent._execution_thread_id)
    if agent._interrupt_requested:
        ra()._set_interrupt(
            True,
            agent._execution_thread_id,
            reason=getattr(agent, "_tool_interrupt_reason", None),
        )
        agent._interrupt_thread_signal_pending = False
    else:
        agent._interrupt_message = None
        agent._tool_interrupt_reason = None
        agent._interrupt_thread_signal_pending = False

    # Notify memory providers of the new turn (BEFORE prefetch_all).
    if agent._memory_manager:
        try:
            _turn_msg = original_user_message if isinstance(original_user_message, str) else ""
            agent._memory_manager.on_turn_start(agent._user_turn_count, _turn_msg)
        except Exception:
            pass

    # External memory provider: prefetch once before the tool loop. Skipped on
    # trivial prompts (greetings, acks) that carry no semantic signal.
    ext_prefetch_cache = ""
    if agent._memory_manager:
        try:
            _query = original_user_message if isinstance(original_user_message, str) else ""
            if not is_trivial_prompt(_query):
                ext_prefetch_cache = agent._memory_manager.prefetch_all(_query) or ""
        except Exception:
            pass
        # Deterministic recall indicator: rendered by Hermes via _emit_status when
        # memory was injected, so the model can't silently drop it.
        if ext_prefetch_cache:
            try:
                _recall_indicator = agent._memory_manager.describe_recall()
                if _recall_indicator:
                    agent._emit_status(_recall_indicator)
            except Exception:
                pass

    # ── api_content sidecar: persist what you send ──
    # Injected context lives only in the API copy; stamp the exact sent bytes on the
    # live dict so replay reproduces the prefix. Skipped for codex_app_server/MoA.
    if (
        not moa_active
        and getattr(agent, "api_mode", None) != "codex_app_server"
        and 0 <= current_turn_user_idx < len(messages)
        and messages[current_turn_user_idx].get("role") == "user"
    ):
        _turn_user_msg = messages[current_turn_user_idx]
        _api_content = compose_user_api_content(
            _turn_user_msg.get("content", ""), ext_prefetch_cache, plugin_user_context
        )
        if _api_content is not None and _api_content != _turn_user_msg.get("content"):
            _turn_user_msg["api_content"] = _api_content
            # In-place preflight compaction already inserted this turn's user row and
            # the crash persist identity-skips compacted dicts, so backfill the stamp
            # onto the row directly. Rotation mode flushes to the child session later.
            if _preflight_compressed and bool(
                getattr(agent, "_last_compaction_in_place", False)
            ):
                _db = getattr(agent, "_session_db", None)
                if _db is not None:
                    try:
                        _db.set_latest_user_api_content(
                            agent.session_id,
                            _turn_user_msg.get("content"),
                            _api_content,
                        )
                    except Exception:
                        logger.warning(
                            "in-place compaction api_content backfill failed "
                            "for session=%s",
                            agent.session_id or "none",
                            exc_info=True,
                        )

    # Crash-resilience: persist the inbound user turn once, with final api_content,
    # before the first LLM call. Same critical section as CLI close persistence;
    # retries the row create if the pre-compression attempt failed transiently.
    def _ensure_and_persist() -> None:
        agent._ensure_db_session()
        agent._persist_session(messages, conversation_history)

    try:
        if persist_lock is None:
            _ensure_and_persist()
        else:
            with persist_lock:
                _ensure_and_persist()
    except Exception:
        logger.warning(
            "Early turn-start session persistence failed for session=%s",
            agent.session_id or "none",
            exc_info=True,
        )
    finally:
        # Keep an unmarked staged input for a later close retry if persistence failed;
        # once marked, the close path must not treat it as a pre-worker UI input.
        if not isinstance(pending_cli_message, dict) or pending_cli_message.get("_db_persisted"):
            agent._pending_cli_user_message = None

    # Title the session now: the row exists and titling depends only on the user's
    # ask, so it runs concurrently with the turn. Daemon thread, no-op once titled.
    _maybe_title_session_at_turn_start(agent, messages)

    return TurnContext(
        user_message=user_message,
        original_user_message=original_user_message,
        messages=messages,
        conversation_history=conversation_history,
        active_system_prompt=active_system_prompt,
        effective_task_id=effective_task_id,
        turn_id=turn_id,
        current_turn_user_idx=current_turn_user_idx,
        should_review_memory=should_review_memory,
        plugin_user_context=plugin_user_context,
        ext_prefetch_cache=ext_prefetch_cache,
        preflight_compression_blocked=_preflight_compression_blocked,
    )
