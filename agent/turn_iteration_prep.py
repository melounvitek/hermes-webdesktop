"""Per-iteration transcript preparation for the conversation turn loop, run before the request
is assembled: the ``agent:step`` callback, skill-nudge counter, pre-API ``/steer`` drain into
the newest tool result (never a user message — alternation), run-budget wrap-up notice,
tool_call argument sanitization, legacy interrupt-scaffold ghost-row drop and role-alternation
repair. Extracted from ``run_conversation``; nothing here imports
``agent.conversation_loop`` at module level (cycle).
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import random
from typing import Any, Dict, Optional

from agent.display import KawaiiSpinner

from agent.turn_context import reanchor_current_turn_user_idx
logger = logging.getLogger("agent.conversation_loop")


@dataclass
class IterationPrep:
    """Always ``action == "fallthrough"``. ``messages`` is the (possibly filtered) transcript
    and ``request_logger`` the per-request logger the caller keeps using."""

    action: str
    messages: Any
    request_logger: Any


def prepare_iteration(
    agent: Any,
    *,
    messages: Any,
    api_call_count: Any,
) -> IterationPrep:
    """Prepare ``messages`` for this iteration in the original order. Every mutation here is
    cache-safe by construction: steer text lands in the newest tool result, the ghost-row
    filter only drops hidden scaffold placeholders, and repair runs BEFORE the request build."""
    from agent.conversation_loop import (
        _INTERRUPT_SCAFFOLD_MARKER,
        _maybe_inject_run_budget_wrapup,
    )

    def _verdict(action: str, result: Optional[Dict[str, Any]] = None) -> IterationPrep:
        return IterationPrep(
            action=action,
            messages=messages,
            request_logger=request_logger,

        )

    # Fire step_callback for gateway hooks (agent:step event)
    if agent.step_callback is not None:
        try:
            prev_tools = []
            for _idx, _m in enumerate(reversed(messages)):
                if _m.get("role") == "assistant" and _m.get("tool_calls"):
                    _fwd_start = len(messages) - _idx
                    _results_by_id = {}
                    for _tm in messages[_fwd_start:]:
                        if _tm.get("role") != "tool":
                            break
                        _tcid = _tm.get("tool_call_id")
                        if _tcid:
                            _results_by_id[_tcid] = _tm.get("content", "")
                    prev_tools = [
                        {
                            "name": tc["function"]["name"],
                            "result": _results_by_id.get(tc.get("id")),
                            "arguments": tc["function"].get("arguments"),
                        }
                        for tc in _m["tool_calls"]
                        if isinstance(tc, dict)
                    ]
                    break
            agent.step_callback(api_call_count, prev_tools)
        except Exception as _step_err:
            logger.debug("step_callback error (iteration %s): %s", api_call_count, _step_err)

    # Track tool-calling iterations for skill nudge.
    # Counter resets whenever skill_manage is actually used.
    if (agent._skill_nudge_interval > 0
            and "skill_manage" in agent.valid_tool_names):
        agent._iters_since_skill += 1

    # ── Pre-API-call /steer drain ──────────────────────────────────
    # Drain a /steer sent during the last API call into the newest tool message so
    # it lands THIS iteration. Never put in a user message (breaks alternation).
    _pre_api_steer = agent._drain_pending_steer()
    if _pre_api_steer:
        _injected = False
        for _si in range(len(messages) - 1, -1, -1):
            _sm = messages[_si]
            if isinstance(_sm, dict) and _sm.get("role") == "tool":
                from agent.prompt_builder import format_steer_marker
                marker = format_steer_marker(_pre_api_steer)
                existing = _sm.get("content", "")
                if isinstance(existing, str):
                    _sm["content"] = existing + marker
                else:
                    # Multimodal content blocks — append text block
                    try:
                        blocks = list(existing) if existing else []
                        blocks.append({"type": "text", "text": marker})
                        _sm["content"] = blocks
                    except Exception:
                        pass
                _injected = True
                logger.debug(
                    "Pre-API-call steer drain: injected into tool msg at index %d",
                    _si,
                )
                break
        if not _injected:
            # No tool message to inject into — put it back so
            # the post-tool-execution drain picks it up later.
            _lock = getattr(agent, "_pending_steer_lock", None)
            if _lock is not None:
                with _lock:
                    if agent._pending_steer:
                        agent._pending_steer = agent._pending_steer + "\n" + _pre_api_steer
                    else:
                        agent._pending_steer = _pre_api_steer
            else:
                existing = getattr(agent, "_pending_steer", None)
                agent._pending_steer = (existing + "\n" + _pre_api_steer) if existing else _pre_api_steer

    # ── Wall-clock run-budget wrap-up notice ───────────────────────
    # One-shot at 80% of agent.run_budget_seconds: ask the model to wrap up via the
    # same cache-safe channel as /steer (newest tool result); off with no budget.
    if getattr(agent, "run_budget_seconds", None):
        _maybe_inject_run_budget_wrapup(agent, messages)

    # Reasoning lives in content via <think> tags for trajectory storage, but some
    # providers (Moonshot) also need a 'reasoning_content' field; handle both here.
    request_logger = getattr(agent, "logger", None) or logger  # same name as the origin module
    # Per-agent validation cursor skips re-parsing tool_call args already validated.
    # Identity-keyed; a rewritten list breaks the prefix match and forces a re-scan.
    _sanitize_cursor = getattr(agent, "_sanitize_args_cursor", None)
    if _sanitize_cursor is None:
        _sanitize_cursor = {}
        try:
            agent._sanitize_args_cursor = _sanitize_cursor
        except Exception:
            pass
    repaired_tool_calls = agent._sanitize_tool_call_arguments(
        messages,
        logger=request_logger,
        session_id=agent.session_id,
        cursor=_sanitize_cursor,
    )
    if repaired_tool_calls > 0:
        request_logger.info(
            "Sanitized %s corrupted tool_call arguments before request (session=%s)",
            repaired_tool_calls,
            agent.session_id or "-",
        )

    # Drop legacy hidden assistant placeholders carrying the raw interrupt scaffold
    # before repair: replayed, the model echoes/self-replicates (#81841).
    messages = [
        msg for msg in messages
        if not (
            msg.get("display_kind") == "hidden"
            and msg.get("role") == "assistant"
            and (
                (
                    isinstance(msg.get("content"), str)
                    and msg["content"].strip() == _INTERRUPT_SCAFFOLD_MARKER
                )
                or (
                    isinstance(msg.get("api_content"), str)
                    and msg["api_content"].strip() == _INTERRUPT_SCAFFOLD_MARKER
                )
            )
        )
    ]

    # Repair malformed role alternation (tool→user / user→user tails): providers
    # return empty content on them and the empty-retry loop spins. The _with_cursor
    # variant also recomputes the SessionDB flush cursor after compaction (#44837).
    from agent.agent_runtime_helpers import repair_message_sequence_with_cursor
    repaired_seq = repair_message_sequence_with_cursor(agent, messages)
    if repaired_seq > 0:
        request_logger.info(
            "Repaired %s message-alternation violations before request (session=%s)",
            repaired_seq,
            agent.session_id or "-",
        )
    return _verdict("fallthrough")


@dataclass
class ApiCallAnnouncement:
    """Always ``action == "fallthrough"``; ``thinking_spinner`` is the started raw spinner or
    None (TUI widget / streaming consumers / verbose mode)."""

    action: str
    thinking_spinner: Any


def announce_api_call(
    agent: Any,
    *,
    messages: Any,
    api_messages: Any,
    api_call_count: Any,
    approx_tokens: Any,
    total_chars: Any,
) -> ApiCallAnnouncement:
    """Print the request summary (verbose) or start the quiet-mode thinking indicator."""


    def _verdict(action: str, result: Optional[Dict[str, Any]] = None) -> ApiCallAnnouncement:
        return ApiCallAnnouncement(
            action=action,
            thinking_spinner=thinking_spinner,

        )

    # Thinking spinner for quiet mode (animated during API call)
    thinking_spinner = None

    if not agent.quiet_mode:
        agent._vprint(f"\n{agent.log_prefix}🔄 Making API call #{api_call_count}/{agent.max_iterations}...")
        agent._vprint(f"{agent.log_prefix}   📊 Request size: {len(api_messages)} messages, ~{approx_tokens:,} tokens (~{total_chars:,} chars)")
        agent._vprint(f"{agent.log_prefix}   🔧 Available tools: {len(agent.tools) if agent.tools else 0}")
    else:
        # Animated thinking spinner in quiet mode
        face = random.choice(KawaiiSpinner.get_thinking_faces())
        verb = random.choice(KawaiiSpinner.get_thinking_verbs())
        if agent.thinking_callback:
            # CLI TUI mode: use prompt_toolkit widget instead of raw spinner
            # (works in both streaming and non-streaming modes)
            agent.thinking_callback(f"{face} {verb}...")
        elif not agent._has_stream_consumers() and agent._should_start_quiet_spinner():
            # Raw KawaiiSpinner only when no streaming consumers and the
            # spinner output has a safe sink.
            spinner_type = random.choice(['brain', 'sparkle', 'pulse', 'moon', 'star'])
            thinking_spinner = KawaiiSpinner(f"{face} {verb}...", spinner_type=spinner_type, print_fn=agent._print_fn)
            thinking_spinner.start()

    # Log request details if verbose
    if agent.verbose_logging:
        logging.debug(f"API Request - Model: {agent.model}, Messages: {len(messages)}, Tools: {len(agent.tools) if agent.tools else 0}")
        logging.debug(f"Last message role: {messages[-1]['role'] if messages else 'none'}")
        logging.debug(f"Total message size: ~{approx_tokens:,} tokens")
    return _verdict("fallthrough")


@dataclass
class IterationStart:
    """``action``: ``"fallthrough"`` (run the iteration) or ``"break"`` (turn ends: interrupt,
    review input budget or iteration budget exhausted — ``_turn_exit_reason`` set)."""

    action: str
    original_user_message: Any
    api_call_count: Any
    interrupted: Any
    _turn_exit_reason: Any


def begin_iteration(
    agent: Any,
    *,
    messages: Any,
    conversation_history: Any,
    original_user_message: Any,
    api_call_count: Any,
    interrupted: Any,
    _turn_exit_reason: Any,
) -> IterationStart:
    """Iteration entry in the original order: apply a pending redirect, reset the checkpoint
    dedup, then the interrupt / review-budget / iteration-budget exits. ``api_call_count`` is
    incremented here (the grace call consumes its flag instead of the budget)."""
    from agent.conversation_loop import (
        _apply_active_turn_redirect,
        _review_input_budget_exhausted,
    )

    def _verdict(action: str, result: Optional[Dict[str, Any]] = None) -> IterationStart:
        return IterationStart(
            action=action,
            original_user_message=original_user_message,
            api_call_count=api_call_count,
            interrupted=interrupted,
            _turn_exit_reason=_turn_exit_reason,

        )

    _redirect_text = agent._drain_pending_redirect()
    if _redirect_text:
        _apply_active_turn_redirect(agent, messages, _redirect_text)
        if isinstance(original_user_message, str):
            original_user_message = (
                f"{original_user_message}\n\n"
                f"User correction during the turn: {_redirect_text}"
            )
        agent._persist_session(messages, conversation_history)

    # Reset per-turn checkpoint dedup so each iteration can take one snapshot
    agent._checkpoint_mgr.new_turn()

    # Check for interrupt request (e.g., user sent new message)
    if agent._interrupt_requested:
        interrupted = True
        _turn_exit_reason = "interrupted_by_user"
        if not agent.quiet_mode:
            agent._safe_print("\n⚡ Breaking out of tool loop due to interrupt...")
        return _verdict("break")

    # Aggregate input budget for detached auxiliary forks: bounds the whole review,
    # not each request. Checked between iterations so the crossing request's writes
    # have landed, mirroring the iteration-budget exit (#93057).
    if _review_input_budget_exhausted(agent):
        _turn_exit_reason = "review_input_budget_exhausted"
        if not agent.quiet_mode:
            agent._safe_print(
                f"\n⏹️  Review input budget exhausted "
                f"({int(agent.session_input_tokens):,} tokens) — stopping "
                f"the review tool loop before the next provider call."
            )
        return _verdict("break")

    api_call_count += 1
    agent._api_call_count = api_call_count
    agent._touch_activity(f"starting API call #{api_call_count}")

    # Grace call: budget exhausted but the model gets one more call. Consume the
    # flag so the loop exits after this iteration regardless of outcome.
    if agent._budget_grace_call:
        agent._budget_grace_call = False
    elif not agent.iteration_budget.consume():
        _turn_exit_reason = "budget_exhausted"
        if not agent.quiet_mode:
            agent._safe_print(f"\n⚠️  Iteration budget exhausted ({agent.iteration_budget.used}/{agent.iteration_budget.max_total} iterations used)")
        return _verdict("break")
    return _verdict("fallthrough")


@dataclass
class RetryRestartVerdict:
    """``action``: ``"fallthrough"`` (a response is ready — process it), ``"continue"``
    (a restart flag re-issues the iteration: redirect / compressed / rebuilt-for-fallback /
    length continuation) or ``"break"`` (turn ends: interrupted, non-actionable compaction
    handoff, or every retry exhausted without a response)."""

    action: str
    current_turn_user_idx: Any
    final_response: Any
    retry_count: Any
    api_call_count: Any
    _preflight_compression_blocked: Any
    _turn_exit_reason: Any


def apply_retry_restarts(
    agent: Any,
    *,
    _retry: Any,
    response: Any,
    interrupted: Any,
    messages: Any,
    conversation_history: Any,
    user_message: Any,
    api_kwargs: Any,
    current_turn_user_idx: Any,
    final_response: Any,
    retry_count: Any,
    api_call_count: Any,
    length_continue_retries: Any,
    _preflight_compression_blocked: Any,
    _turn_exit_reason: Any,
) -> RetryRestartVerdict:
    """Consume the ``TurnRetryState`` restart flags after the retry loop, in the original
    priority order. Refunds the iteration budget/count for restarts that produced no valid
    assistant item; ``restart_with_rebuilt_messages`` is the single consumer that clears
    ``_preflight_compression_blocked`` so the fallback gets a fresh preflight (#84733)."""
    from agent.conversation_loop import (
        _HANDOFF_SKIP_FINAL_RESPONSE,
        _should_skip_model_call_for_reference_handoff,
    )

    def _verdict(action: str, result: Optional[Dict[str, Any]] = None) -> RetryRestartVerdict:
        return RetryRestartVerdict(
            action=action,
            current_turn_user_idx=current_turn_user_idx,
            final_response=final_response,
            retry_count=retry_count,
            api_call_count=api_call_count,
            _preflight_compression_blocked=_preflight_compression_blocked,
            _turn_exit_reason=_turn_exit_reason,

        )

    if _retry.restart_with_redirected_messages:
        # Cancelled request produced no valid assistant item: reuse the same logical
        # iteration after the outer loop appends partial context + correction.
        api_call_count -= 1
        agent.iteration_budget.refund()
        _retry.restart_with_redirected_messages = False
        return _verdict("continue")

    # If the API call was interrupted, skip response processing
    if interrupted:
        _turn_exit_reason = "interrupted_during_api_call"
        return _verdict("break")

    if _retry.restart_with_compressed_messages:
        api_call_count -= 1
        agent.iteration_budget.refund()
        # Compression restarts count toward the retry limit so a compression that
        # shrinks messages but not enough can't loop forever.
        retry_count += 1
        _retry.restart_with_compressed_messages = False
        if _should_skip_model_call_for_reference_handoff(
            messages, user_message
        ):
            logger.info(
                "Skipping compressed-restart model call: reference-only "
                "handoff would be the sole active user turn (#80622)"
            )
            if not final_response:
                final_response = _HANDOFF_SKIP_FINAL_RESPONSE
            _turn_exit_reason = "compaction_handoff_not_actionable"
            return _verdict("break")
        # In-loop compression rebuilt `messages`; re-anchor the current-turn index
        # like the prologue, AFTER the handoff guard (it may re-append this turn's
        # ask). A stale anchor injects prefetch into a historical row.
        current_turn_user_idx = reanchor_current_turn_user_idx(
            messages, user_message
        )
        agent._persist_user_message_idx = current_turn_user_idx
        return _verdict("continue")

    if _retry.restart_with_rebuilt_messages:
        # A stall/failure escalated to the fallback chain: re-issue against the
        # active fallback provider, refunding budget/count for the stalled attempt.
        api_call_count -= 1
        agent.iteration_budget.refund()
        _retry.restart_with_rebuilt_messages = False
        # Failover shrank the compressor window: clear the preflight block so
        # preflight re-runs before the first fallback call. Hoisted to the single
        # consumer. (#84733)
        _preflight_compression_blocked = False
        return _verdict("continue")

    if _retry.restart_with_length_continuation:
        # Boost output budget per retry: 2×, 4×, 8×, 16× base, capped at 32 768, via
        # _ephemeral_max_output_tokens. Keep a larger original provider/model
        # default as the floor so retries never downshift.
        _boost_base = agent.max_tokens if agent.max_tokens else 4096
        _boost = _boost_base * (2 ** length_continue_retries)
        _requested_cap = agent._requested_output_cap_from_api_kwargs(api_kwargs)
        if _requested_cap is not None:
            _boost = max(_boost, _requested_cap)
        _boost_cap = max(32768, _requested_cap or 0)
        agent._ephemeral_max_output_tokens = min(_boost, _boost_cap)
        return _verdict("continue")

    # All retries may exhaust with `response` still None; break out cleanly.
    if response is None:
        _turn_exit_reason = "all_retries_exhausted_no_response"
        print(f"{agent.log_prefix}❌ All API retries exhausted with no successful response.")
        agent._persist_session(messages, conversation_history)
        return _verdict("break")
    return _verdict("fallthrough")
