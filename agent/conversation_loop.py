"""The agent conversation loop — extracted from ``run_agent.AIAgent``.

``run_conversation(agent, ...)`` drives one user turn (model call, tool dispatch,
retries, fallbacks, compression, post-turn hooks). Symbols that callers patch on
``run_agent`` (``handle_function_call``, ``_set_interrupt``, ``OpenAI``) resolve via
``_ra`` so those patches keep working."""

from __future__ import annotations

import json
import logging
import os
import random
import re
import ssl
import sys
import time
from typing import Any, Dict, List, Optional

from agent.codex_responses_adapter import _summarize_user_message_for_log
from agent.conversation_compression import (
    conversation_history_after_compression,  # noqa: F401 — resolved lazily by turn_overflow/turn_preflight/turn_recovery (tests patch it here)
)
from agent.display import KawaiiSpinner
from agent.error_classifier import FailoverReason, classify_api_error
from agent.fast_mode import begin_turn as begin_fast_mode_turn
from agent.message_metadata import append_message
from agent.turn_context import (
    PreflightCompressionTimedOut,
    _compression_warrants_another_preflight_pass,
    build_turn_context,
    compose_user_api_content,
    reanchor_current_turn_user_idx,
)
from agent.turn_retry_state import TurnRetryState
from agent.turn_usage import record_response_usage
from agent.turn_overflow import recover_from_overflow
from agent.turn_empty_response import recover_empty_response
from agent.turn_stop_gates import apply_stop_gates
from agent.turn_tool_validation import validate_tool_calls
from agent.turn_truncation import (
    continue_codex_incomplete,
    handle_content_policy_refusal,
    recover_from_truncation,
)
from agent.turn_preflight import compress_after_tool_results, run_preflight_compression
from agent.turn_recovery import (
    route_classified_error,
    describe_invalid_response,
    validate_response_shape,
    compute_error_backoff,
    interruptible_backoff_sleep,
    log_api_error_attempt,
    max_retries_exhausted_result,
    nonretryable_client_error_result,
    recover_after_classification,
    recover_before_classification,
)
from agent.runtime_cwd import resolve_agent_cwd
from agent.message_sanitization import (
    close_interrupted_tool_sequence,
    _repair_tool_call_arguments,
    coalesce_tool_call_id,
    _sanitize_messages_surrogates,
    _sanitize_structure_non_ascii,
    _sanitize_structure_surrogates,
    _sanitize_surrogates,
)
# Must mirror _STALE_TOOL_CALL_MARKER_RE in hermes_state.py; kept local so importing
# hermes_state (module-level DEFAULT_DB_PATH) is not forced at load time.
_STALE_MARKER_RE = re.compile(r"^\[[A-Za-z_][A-Za-z0-9_.-]*\]$")
from agent.model_metadata import (
    MINIMUM_CONTEXT_LENGTH,
    _estimate_tools_tokens_rough,
    anchored_context_tokens,
    estimate_messages_tokens_rough,
    estimate_request_tokens_rough,  # noqa: F401 — resolved lazily by turn_overflow/turn_preflight/turn_recovery (tests patch it here)
    save_context_length,  # noqa: F401 — resolved lazily by agent.turn_overflow (tests patch it here)
)
from agent.process_bootstrap import _install_safe_stdio
from agent.prompt_caching import (
    build_prompt_cache_plan,
    effective_cache_ttl,
    strip_anthropic_cache_control,
    strip_anthropic_tool_cache_control,
)
from agent.provider_projection import splice_provider_projection
from agent.retry_utils import (
    adaptive_rate_limit_backoff,  # noqa: F401 — resolved lazily by agent.turn_recovery (tests patch it here)
    jittered_backoff,
)
from agent.trajectory import has_incomplete_scratchpad
# Bind before the turn starts so a source-tree swap cannot load a skewed
# finalizer at turn end.
from agent.turn_finalizer import finalize_turn
from hermes_logging import set_session_context
from tools.skill_provenance import set_current_write_origin
from utils import base_url_host_matches, env_var_enabled

logger = logging.getLogger(__name__)


# Scaffold marker used by _apply_active_turn_redirect and the ghost-row filter
# in the api_messages loop. Module-level so both sites can never drift.
_INTERRUPT_SCAFFOLD_MARKER = "[This response was interrupted by a user correction.]"


# One-time wrap-up notice appended when a wall-clock run budget crosses 80%
# (agent.run_budget_seconds / --run-budget): stop new work, deliver current state.
RUN_BUDGET_WRAPUP_NOTICE = (
    "[SYSTEM NOTICE — run time budget nearly exhausted] "
    "Run time budget nearly exhausted. Stop new discovery/verification work "
    "now. Produce the required final deliverable (answer/JSON/summary) from "
    "the state you already have, completing only mandatory writes."
)


def _midturn_request_pressure_tokens(
    agent: Any,
    api_messages: List[Dict[str, Any]],
    effective_system: str,
    approx_tokens: int,
) -> int:
    """Token figure the mid-turn pre-API compression guard compares.

    Returns the pruned native-Responses estimate when native compaction eligibility is
    proven (the generic estimate overstates the wire on compacted sessions, #96995),
    else the generic message+tools figure. System prompt is counted exactly once."""
    try:
        from agent.codex_responses_adapter import (
            estimate_native_responses_preflight_tokens,
        )

        native = estimate_native_responses_preflight_tokens(
            agent,
            api_messages,
            system_prompt=effective_system or "",
            tools=getattr(agent, "tools", None) or None,
        )
        if isinstance(native, int) and not isinstance(native, bool) and native >= 0:
            return native
    except Exception:
        logger.debug(
            "native Responses mid-turn estimate unavailable; "
            "using generic transcript estimate",
            exc_info=True,
        )
    return approx_tokens + (
        _estimate_tools_tokens_rough(agent.tools) if agent.tools else 0
    )


def _review_input_budget_exhausted(agent: Any) -> bool:
    """True when a detached review fork has replayed its aggregate input budget.

    Only forks with an explicit ``_review_input_token_budget`` are gated (#93057). Fires
    at the top of the NEXT iteration, so the budget-crossing request completes first."""
    budget = getattr(agent, "_review_input_token_budget", None)
    if not isinstance(budget, int) or isinstance(budget, bool) or budget <= 0:
        return False
    used = getattr(agent, "session_input_tokens", 0)
    return isinstance(used, int) and not isinstance(used, bool) and used >= budget


def _maybe_inject_run_budget_wrapup(agent: Any, messages: List[Dict[str, Any]]) -> bool:
    """Inject the one-time wall-clock wrap-up notice when past 80% of budget.

    Appends to the NEWEST ``role:"tool"`` message (cache-safe, like /steer); latches
    ``_run_budget_wrapup_injected`` only on a successful append. Returns True when
    injected. Dormant unless ``run_budget_seconds`` + ``_run_budget_started_at`` set."""
    budget = getattr(agent, "run_budget_seconds", None)
    if not budget:
        return False
    if getattr(agent, "_run_budget_wrapup_injected", False):
        return False
    started = getattr(agent, "_run_budget_started_at", None)
    if not started:
        return False
    if (time.time() - started) < 0.8 * float(budget):
        return False
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if isinstance(msg, dict) and msg.get("role") == "tool":
            existing = msg.get("content", "")
            if isinstance(existing, str):
                msg["content"] = existing + f"\n\n{RUN_BUDGET_WRAPUP_NOTICE}"
            else:
                # Multimodal content blocks — append a text block.
                try:
                    blocks = list(existing) if existing else []
                    blocks.append({"type": "text", "text": RUN_BUDGET_WRAPUP_NOTICE})
                    msg["content"] = blocks
                except Exception:
                    return False
            agent._run_budget_wrapup_injected = True
            logger.info(
                "Run budget wrap-up notice injected (budget=%.0fs, elapsed=%.0fs)",
                float(budget),
                time.time() - started,
            )
            return True
    return False


def _restore_user_after_reference_handoff(
    messages: List[Dict[str, Any]], user_message: Any
) -> bool:
    """Re-append this turn's real user ask when compaction left only a handoff.

    Returns True when a restore append happened; only decides whether a restorable
    ask exists (#80622)."""
    if user_message is None:
        return False
    if isinstance(user_message, str):
        if not user_message.strip():
            return False
        content: Any = user_message
    elif isinstance(user_message, list):
        if not user_message:
            return False
        content = user_message
    else:
        return False
    if (
        messages
        and isinstance(messages[-1], dict)
        and messages[-1].get("role") == "user"
        and messages[-1].get("content") == content
    ):
        return False
    append_message(messages, {"role": "user", "content": content})
    return True


def _should_skip_model_call_for_reference_handoff(
    messages: List[Dict[str, Any]], user_message: Any
) -> bool:
    """Guard post-compaction continues against sole-handoff active turns (#80622)."""
    from agent.context_compressor import reference_handoff_would_drive_next_model_call

    if not reference_handoff_would_drive_next_model_call(messages):
        return False
    if _restore_user_after_reference_handoff(messages, user_message):
        # The restored ask is an actionable non-synthetic user row appended
        # after the handoff — by construction the handoff no longer drives.
        return False
    return True


# Fallback final_response for the sole-handoff skip (#80622). Not a replay of the
# last assistant text: finalize_turn appends final_response as a fresh assistant row.
_HANDOFF_SKIP_FINAL_RESPONSE = (
    "Context was compacted. The previous response is complete — "
    "awaiting your next message."
)

# Terminal final_response when compression hit its host timeout while the request
# was still oversized; resending would only bounce off the overflow error (#98722).
_COMPRESSION_TIMEOUT_FINAL_RESPONSE = (
    "Context compression timed out without reducing this conversation. "
    "No messages were dropped. Start a fresh session with /new, or check "
    "auxiliary.compression before retrying /compress."
)


# Stable prefix of the local interrupt status string; surfaces (ACP, TUI) match on
# it to treat the text as cancellation metadata rather than assistant prose.
INTERRUPT_WAITING_FOR_MODEL_PREFIX = "Operation interrupted: waiting for model response ("


def _should_rearm_compression_budget(
    compression_attempts: int,
    *,
    completed_compaction_pending: bool,
    prompt_tokens: int,
    threshold_tokens: int,
) -> bool:
    """Return True after a provider proves a completed compaction worked.

    Rough estimates cannot rearm the anti-thrash budget; require the completed-
    compaction latch and a positive normalized prompt count below the threshold."""
    return bool(
        compression_attempts
        and completed_compaction_pending
        and threshold_tokens > 0
        and 0 < prompt_tokens < threshold_tokens
    )


# Modules whose presence in a traceback (without any API-call module) marks a
# deterministic local bug not worth retrying. NEVER add "conversation_loop" or
# "run_agent": every exception passes through them; _hit_local would be True (#66267)
_LOCAL_PROCESSING_MODULES = frozenset({
    "agent_runtime_helpers",
    "message_content",
    "message_sanitization",
    "chat_completion_helpers",  # only local when NOT also an API-call module
})
_API_CALL_MODULES = frozenset({
    "chat_completion_helpers",
})

# Max outer-loop exceptions per user turn before giving up; only exceptions that
# ESCAPE the inner retry/fallback machinery count, so this can be small (#92450).
_MAX_OUTER_LOOP_ERRORS = 8


def _is_interpreter_shutdown_error(exc: Exception) -> bool:
    """Check if *exc* is a fatal interpreter-shutdown failure.

    Delegates to ``tools.interpreter_shutdown`` (one text-matching site for the
    shutdown-race bug class) but keeps the RuntimeError type gate: a ValueError
    carrying similar text must not match (#93269)."""
    if isinstance(exc, RuntimeError):
        from tools.interpreter_shutdown import interpreter_shutting_down

        return interpreter_shutting_down(exc)
    return False


def _moa_client_consumes_prepared_request(client: Any) -> bool:
    """True when ``client`` is the in-process MoA facade.

    Only ``MoAChatCompletions`` exposes ``prepare()``; other clients raise TypeError on
    ``_moa_prepared_request`` even while ``agent.provider`` stays ``"moa"``."""
    completions = getattr(getattr(client, "chat", None), "completions", None)
    return callable(getattr(completions, "prepare", None))


def _join_truncated_parts(parts: List[str]) -> str:
    """Join continuation fragments, adding a newline where two would glue together (#78577)."""
    joined = ""
    for part in parts:
        if joined and not joined[-1].isspace() and part and not part[0].isspace():
            joined += "\n"
        joined += part
    return joined


def _moa_reference_metrics_for_hook(agent: Any) -> Any:
    """Per-advisor metrics for post_api_request, or None off the MoA path.

    MoA returns only the aggregator response, so a plugin sees one generation for
    the whole fan-out; this carries the per-slot advisor spend across the hook boundary."""
    client = getattr(agent, "client", None)
    getter = getattr(client, "last_reference_metrics", None)
    if not callable(getter):
        return None
    try:
        return getter()
    except Exception:
        return None


def _apply_active_turn_redirect(agent: Any, messages: List[Dict[str, Any]], text: str) -> None:
    """Append a provider-safe checkpoint and correction to the live turn.

    Keeps only the *visible* text (demoted to plain text) then adds the correction as a
    real user message, so role alternation holds and cached messages stay byte-identical.
    INVARIANT: raw chain-of-thought never enters replayable content — inlined CoT reads
    as a prefill jailbreak and bricks the session with empty-response storms.
    INVARIANT: the interruption scaffold is replay text, carried only in the user
    correction's ``api_content``; an on-screen-empty placeholder is ``display_kind=hidden``."""
    visible = agent._strip_think_blocks(
        getattr(agent, "_current_streamed_assistant_text", "") or ""
    ).strip()

    checkpoint_parts = [_INTERRUPT_SCAFFOLD_MARKER]
    if visible:
        checkpoint_parts.extend(
            ["Visible response before the interruption:", visible]
        )
    checkpoint = "\n\n".join(checkpoint_parts)
    correction = (
        "[Context from the interrupted assistant response]\n"
        f"{checkpoint}\n\n"
        f"{text}"
    )

    # The live tail is normally user or tool, so an assistant placeholder + correction
    # keeps strict alternation; if the tail is already assistant, fold the checkpoint
    # into the user correction instead of creating assistant→assistant.
    if messages and messages[-1].get("role") == "assistant":
        # Transcript shows the user's own words; the provider replays the
        # scaffolded form so it still sees the interrupted context.
        append_message(
            messages,
            {"role": "user", "content": text, "api_content": correction},
        )
    else:
        # Placeholder preserves role alternation only. Scaffold bytes must never land
        # here: api_content is substituted back into content on replay (#81841).
        placeholder: Dict[str, Any] = {
            "role": "assistant",
            "content": visible or "",
        }
        if not visible:
            placeholder["display_kind"] = "hidden"
            # Hidden row, but a non-empty neutral api_content so the pre-call
            # sanitizer does not re-heal it every call (#88955). Never
            # _INTERRUPT_SCAFFOLD_MARKER: as assistant text the model echoes it (#81841)
            from agent.agent_runtime_helpers import _INTERRUPTED_PLACEHOLDER

            placeholder["api_content"] = _INTERRUPTED_PLACEHOLDER
        append_message(messages, placeholder)
        append_message(
            messages,
            {"role": "user", "content": text, "api_content": correction},
        )

    agent._current_streamed_assistant_text = ""
    agent._stream_needs_break = True


def _is_copilot_provider(agent: Any) -> bool:
    """Delegate to ``AIAgent._is_copilot_provider`` (single owner of the check).

    ``agent.provider`` may hold the aliases ``github-copilot`` / ``github``; a bare
    ``provider == "copilot"`` gate would skip credential recovery for them."""
    try:
        return bool(agent._is_copilot_provider())
    except Exception:
        return (getattr(agent, "provider", "") or "").strip().lower() in {
            "copilot",
            "github-copilot",
            "github",
        }


def _is_stale_copilot_credential_error(status_code: Optional[int], error_message: str) -> bool:
    """Detect a Copilot 400 that is really a STALE / DEGRADED credential.

    Matches status 400 AND ``model_not_available_for_integrator`` or
    ``model_not_supported`` / "the requested model is not supported", so a wrong model
    name never triggers the single-shot re-exchange. Caller enforces scoping/guard."""
    lowered = (error_message or "").lower()
    is_400 = status_code == 400 or "error code: 400" in lowered
    if not is_400:
        return False
    return (
        "model_not_available_for_integrator" in lowered
        or "not available for integrator" in lowered
        or "model_not_supported" in lowered
        or "the requested model is not supported" in lowered
    )



def _ollama_context_limit_error(agent: Any, request_tokens: int) -> Optional[str]:
    """Return a user-facing error when Ollama is loaded with too little context."""
    if not getattr(agent, "tools", None):
        return None

    runtime_ctx = getattr(agent, "_ollama_num_ctx", None)
    if not isinstance(runtime_ctx, int) or runtime_ctx <= 0:
        return None
    if runtime_ctx >= MINIMUM_CONTEXT_LENGTH:
        return None

    model = getattr(agent, "model", "") or "the selected model"
    base_url = getattr(agent, "base_url", "") or "unknown base URL"
    provider = getattr(agent, "provider", "") or "unknown"
    tool_count = len(getattr(agent, "tools", None) or [])

    logger.warning(
        "Ollama runtime context too small for Hermes tool use: "
        "model=%s provider=%s base_url=%s runtime_context=%d "
        "minimum_context=%d estimated_request_tokens=%d tool_count=%d "
        "session=%s",
        model,
        provider,
        base_url,
        runtime_ctx,
        MINIMUM_CONTEXT_LENGTH,
        request_tokens,
        tool_count,
        getattr(agent, "session_id", None) or "none",
    )

    return (
        f"Ollama loaded `{model}` with only {runtime_ctx:,} tokens of runtime "
        f"context, but Hermes needs at least {MINIMUM_CONTEXT_LENGTH:,} tokens "
        "for reliable tool use.\n\n"
        "Increase the Ollama context for this model and restart/reload the "
        "model before trying again. A known-good starting point is 65,536 "
        "tokens. In Hermes config, set `model.ollama_num_ctx: 65536` "
        "(and `model.context_length: 65536` if you also override the displayed "
        "model context). If you manage the model through an Ollama Modelfile, "
        "set `PARAMETER num_ctx 65536` there instead."
    )


def _maybe_grow_local_window(agent: Any, compressor: Any,
                             request_tokens: int) -> Optional[int]:
    """Try growing the managed local model's context window before compressing.

    Returns the new window when the ladder granted one, else None (hold / at native /
    not a managed local session). Cheap for non-local providers: one compare."""
    provider = (getattr(agent, "provider", "") or "").strip().lower()
    if provider not in ("llamacpp", "llama.cpp", "llama-cpp", "custom"):
        return None
    base_url = getattr(agent, "base_url", "") or ""
    if "127.0.0.1" not in base_url and "localhost" not in base_url:
        return None
    try:
        from hermes_cli.local_runtime.growth import maybe_grow_window

        current_window = int(getattr(compressor, "context_length", 0) or 0)
        if current_window <= 0:
            return None
        return maybe_grow_window(
            getattr(agent, "model", "") or "",
            base_url=base_url,
            session_tokens=int(request_tokens),
            current_window=current_window,
        )
    except Exception as exc:  # noqa: BLE001 — growth must never break a turn
        logger.debug("local window growth check failed: %s", exc)
        return None


def _ra():
    """Lazy ``run_agent`` reference so patches on ``run_agent.handle_function_call`` /
    ``run_agent._set_interrupt`` / ``run_agent.OpenAI`` reach this code path."""
    import run_agent
    return run_agent


def _nous_entitlement_message(capability: str) -> str:
    try:
        from hermes_cli.nous_account import (
            format_nous_portal_entitlement_message,
            get_nous_portal_account_info,
        )

        account_info = get_nous_portal_account_info(force_fresh=True)
        message = format_nous_portal_entitlement_message(
            account_info,
            capability=capability,
        )
        return message or ""
    except Exception:
        return ""


def _print_nous_entitlement_guidance(agent, capability: str) -> bool:
    message = _nous_entitlement_message(capability)
    if not message:
        return False
    for line in message.splitlines():
        agent._vprint(f"{agent.log_prefix}   💡 {line}", force=True)
    return True


def _system_prompt_for_hooks(api_kwargs: Any, request_messages: Any) -> Any:
    """System prompt as actually sent to the provider, for observability hooks.

    Checks ``system`` (Anthropic), ``instructions`` (Responses/Codex), then
    ``messages[0]``. Returns None when the request carries no system prompt."""
    system_prompt = api_kwargs.get("system")
    if system_prompt is None:
        system_prompt = api_kwargs.get("instructions")
    if system_prompt is None and isinstance(request_messages, list) and request_messages:
        first = request_messages[0]
        if isinstance(first, dict) and first.get("role") == "system":
            system_prompt = first.get("content")
    return system_prompt


def _is_nous_inference_route(provider: str, base_url: str) -> bool:
    provider = (provider or "").strip().lower()
    if provider == "nous":
        return True
    base = str(base_url or "")
    return (
        base_url_host_matches(base, "inference-api.nousresearch.com")
    )


def _billing_or_entitlement_message(
    *,
    capability: str,
    provider: str,
    base_url: str,
    model: str,
    unverified: bool = False,
) -> str:
    if _is_nous_inference_route(provider, base_url):
        return _nous_entitlement_message(capability)

    provider_label = (provider or "").strip() or "the selected provider"
    model_label = (model or "").strip() or "the selected model"

    # Anthropic Pro/Max OAuth surfaces exhaustion of the "extra usage" bucket as a hard
    # 400; point at the settings page and cycle reset — "add credits" does not apply.
    if (provider or "").strip().lower() == "anthropic":
        # ``unverified`` (#82154): the "out of extra usage" 400 is also returned for a
        # server-side content-filter rejection, so hedge and name the other cause.
        if unverified:
            lines = [
                (
                    f"{provider_label} reported that your Claude subscription usage may be "
                    f"exhausted for {model_label} (included quota + extra-usage credits) — "
                    "but this specific error is not proof of a billing problem."
                ),
                "If https://claude.ai/settings/usage still shows quota remaining, this is "
                "probably NOT a billing problem: on a Claude subscription (OAuth) token "
                "Anthropic returns this same message when its content filter rejects part "
                "of the request — typically a phrase in the system prompt.",
                "If usage really is exhausted: wait for the billing cycle to reset, or add "
                "extra usage at https://claude.ai/settings/usage",
                "You can also switch to an Anthropic API key or another provider with "
                "/model <model> --provider <provider>.",
                # The exhaustion latch replays the stored error without issuing
                # a request, so a real fix looks like it didn't work.
                "Retry with a fresh credential state: `hermes auth reset anthropic`. Until "
                "that cooldown clears, this error can be replayed from cache without "
                "contacting the API.",
            ]
        else:
            lines = [
                (
                    f"{provider_label} reported that your Claude subscription usage is "
                    f"exhausted for {model_label} (included quota + extra-usage credits)."
                ),
                "Options: wait for the billing cycle to reset, or add extra usage at "
                "https://claude.ai/settings/usage",
                "You can also switch to an Anthropic API key or another provider with "
                "/model <model> --provider <provider>.",
            ]
        return "\n".join(lines)

    # Provider-agnostic billing URL so every text surface (CLI, gateway, TUI) shows the
    # same actionable link, not just OpenRouter.
    try:
        from agent.billing_links import build_billing_block

        _link = build_billing_block(provider=provider, base_url=base_url, model=model)
        if _link.provider_label:
            provider_label = _link.provider_label
        billing_url = _link.billing_url
    except Exception:
        billing_url = None

    lines = [
        (
            f"{provider_label} reported that billing, credits, or account "
            f"entitlement is exhausted for {model_label}."
        ),
        "Add credits or update billing with that provider, then retry.",
    ]
    if billing_url:
        lines.append(f"{provider_label} billing: {billing_url}")
    lines.append("You can switch providers temporarily with /model <model> --provider <provider>.")
    return "\n".join(lines)


def _billing_block_dict(
    provider, base_url, model, message="", *, unverified: bool = False
) -> Optional[dict]:
    """Best-effort structured billing descriptor (None if billing_links is unavailable)."""
    try:
        from agent.billing_links import build_billing_block

        block = build_billing_block(
            provider=provider, base_url=str(base_url), model=model, message=message
        ).to_dict()
    except Exception:
        return None
    if block is not None and unverified:
        # Carry the classifier's ambiguity into the structured descriptor so
        # every surface rendering the block can hedge too (#82154).
        block["unverified"] = True
    return block


def _billing_terminal_label(summary: str, unverified: bool) -> str:
    """Terminal-failure prefix for a billing-classified error.

    ``unverified`` (#82154): the Anthropic "out of extra usage" 400 can be a
    content-filter rejection, so the line must not assert exhaustion as fact."""
    if unverified:
        return (
            "Provider reported usage/credit exhaustion (unverified — the same "
            f"error can be a content-filter rejection, not billing): {summary}"
        )
    return f"Billing or credits exhausted: {summary}"


def _billing_failure_result(
    *,
    classified,
    summary: str,
    messages,
    api_call_count: int,
    provider: str,
    base_url,
    model: str,
    guidance: Optional[str] = None,
) -> dict:
    """Structured terminal result for a billing-classified failure.

    Single construction point so label, guidance, structured block and ambiguity flag
    stay consistent across the non-retryable abort and max-retries paths (#82154)."""
    unverified = bool(getattr(classified, "billing_unverified", False))
    if guidance is None:
        guidance = _billing_or_entitlement_message(
            capability="model access",
            provider=provider,
            base_url=str(base_url),
            model=model,
            unverified=unverified,
        )
    final = _billing_terminal_label(summary, unverified)
    if guidance:
        final += f"\n\n{guidance}"
    return {
        "final_response": final,
        "messages": messages,
        "api_calls": api_call_count,
        "completed": False,
        "failed": True,
        "error": summary,
        "failure_reason": classified.reason.value,
        # Classifier's own retry verdict so UI (agent/error_surface.py) shows Retry
        # only when a re-run can differ, not re-derived from a second taxonomy.
        "failure_retryable": bool(classified.retryable),
        # The billing verdict may rest on an ambiguous body (#82154) — carry
        # that through the structured result, not just the prose.
        "billing_unverified": unverified,
        "billing_block": _billing_block_dict(
            provider, base_url, model, guidance, unverified=unverified
        ),
    }


def _print_billing_or_entitlement_guidance(
    agent,
    *,
    capability: str,
    provider: str,
    base_url: str,
    model: str,
    unverified: bool = False,
) -> bool:
    message = _billing_or_entitlement_message(
        capability=capability,
        provider=provider,
        base_url=base_url,
        model=model,
        unverified=unverified,
    )
    if not message:
        return False
    for line in message.splitlines():
        agent._vprint(f"{agent.log_prefix}   💡 {line}", force=True)
    return True



def _restore_or_build_system_prompt(agent, system_message, conversation_history):
    """Restore the cached system prompt from the session DB or build it fresh.

    Mutates ``agent._cached_system_prompt`` and persists a freshly-built prompt on first
    build. Row states ``missing``/``null``/``empty``/``present`` are logged and DB
    failures log at WARNING so silent prefix-cache misses show in ``agent.log``."""
    stored_prompt = None
    stored_state = "missing"
    session_row = None
    if conversation_history and agent._session_db:
        try:
            session_row = agent._session_db.get_session(agent.session_id)
            if session_row is not None:
                raw_prompt = session_row.get("system_prompt")
                if raw_prompt is None:
                    stored_state = "null"
                elif raw_prompt == "":
                    stored_state = "empty"
                else:
                    stored_prompt = raw_prompt
                    stored_state = "present"
        except Exception as exc:
            logger.warning(
                "Session DB get_session failed for system-prompt restore "
                "(session=%s): %s. Falling back to fresh build — prefix "
                "cache will miss for this turn.",
                agent.session_id, exc,
            )

    if stored_prompt and _stored_prompt_matches_runtime(agent, stored_prompt):
        # Bot Chat capability epoch: the stored prompt embeds a capability fingerprint;
        # a mismatch is a deliberate once-per-change rebuild. Unstamped prompts never
        # take this branch; probe failures fail closed to "reuse" so cache is kept.
        _bot_stale = False
        try:
            from tools.bot_mode_probe import (
                BOT_CHAT_TITLE,
                stored_bot_chat_prompt_needs_upgrade,
                stored_prompt_capability_stale,
            )

            _home_for_epoch = None
            try:
                from agent.system_prompt import _agent_home

                _home_for_epoch = _agent_home(agent)
            except Exception:
                pass
            _bot_stale = stored_prompt_capability_stale(stored_prompt, _home_for_epoch)
            if not _bot_stale and getattr(agent, "_bot_mode_protocol", True):
                # Legacy upgrade: a Bot Chat prompt predating the epoch mechanism gets
                # ONE title-gated migration rebuild; the stamped result cannot re-fire.
                _t = str(getattr(agent, "_session_title_hint", "") or "").strip()
                if not _t and agent._session_db and agent.session_id:
                    try:
                        _t = str(agent._session_db.get_session_title(agent.session_id) or "").strip()
                    except Exception:
                        _t = ""
                if _t == BOT_CHAT_TITLE:
                    _bot_stale = stored_bot_chat_prompt_needs_upgrade(stored_prompt, _home_for_epoch)
        except Exception:
            _bot_stale = False
        if _bot_stale:
            logger.info(
                "Bot Chat capability epoch changed for session %s; rebuilding "
                "system prompt to adopt the new capability surface (one-time "
                "prefix-cache break).",
                agent.session_id,
            )
            agent._session_title_hint = "Bot Chat"
            # The skills index cache (LRU + disk snapshot) does not watch the skills
            # dir; a capability refresh must rebuild THROUGH it or new skills are lost.
            try:
                from agent.prompt_builder import clear_skills_system_prompt_cache

                clear_skills_system_prompt_cache(clear_snapshot=True)
            except Exception:
                pass
            agent._cached_system_prompt = agent._build_system_prompt(system_message)
            # Persist so the NEXT turn restores the new bytes verbatim (cache break is
            # once per capability change). on_session_start not re-fired: continuation.
            if agent._session_db:
                try:
                    agent._session_db.update_system_prompt(
                        agent.session_id, agent._cached_system_prompt
                    )
                except Exception as exc:
                    logger.warning(
                        "Session DB update_system_prompt failed after Bot Chat "
                        "capability refresh (session=%s): %s. The refresh will "
                        "re-fire next turn.",
                        agent.session_id, exc,
                    )
            return
        # Continuing session — reuse the exact system prompt from the
        # previous turn so the Anthropic cache prefix matches.
        agent._cached_system_prompt = stored_prompt
        # Same contract for tools[]: pin the array to the order this session already
        # sent (tools freeze) instead of re-probing every check_fn on a fresh AIAgent.
        try:
            saved_tools = session_row.get("tool_names") if session_row else None
            if saved_tools:
                from tools.mcp_tool import restore_agent_tool_prefix

                restore_agent_tool_prefix(agent, json.loads(saved_tools))
        except Exception:
            logger.debug("tool prefix restore skipped", exc_info=True)
        # Prompt-section callbacks are new-session-only; recover their frozen bytes
        # from the persisted prompt so a compression rebuild keeps them.
        from agent.system_prompt import restore_plugin_prompt_sections

        restore_plugin_prompt_sections(agent, stored_prompt)
        # The static prefix is not persisted; rebuild it for the early cache breakpoint
        # or fresh-per-turn gateway agents fall back to the single-breakpoint layout.
        # reconstruct_static_prefix gates on _use_prompt_caching, fails open to legacy.
        from agent.system_prompt import reconstruct_static_prefix

        reconstruct_static_prefix(agent, system_message=system_message)
        return
    if stored_prompt:
        stored_state = "stale_runtime"
        logger.info(
            "Stored system prompt for session %s has stale runtime identity; "
            "rebuilding for model=%s provider=%s.",
            agent.session_id,
            getattr(agent, "model", "") or "",
            getattr(agent, "provider", "") or "",
        )

    if conversation_history and stored_state in ("null", "empty"):
        # Continuing session with an unusable stored prompt: every turn now rebuilds
        # and the prefix cache misses every time.
        logger.warning(
            "Stored system prompt for session %s is %s; rebuilding "
            "from scratch this turn. Prefix cache will miss until "
            "the rebuild persists. Investigate the previous turn's "
            "update_system_prompt write path.",
            agent.session_id, stored_state,
        )

    # First turn of a new session (or recovering from a broken stored
    # prompt) — build from scratch.
    agent._cached_system_prompt = agent._build_system_prompt(system_message)

    # Plugin hook: on_session_start — fired once for a brand-new session, not on
    # continuation.
    try:
        from hermes_cli.lifecycle import invoke_hook as _invoke_hook
        _invoke_hook(
            "on_session_start",
            session_id=agent.session_id,
            model=agent.model,
            platform=getattr(agent, "platform", None) or "",
        )
    except Exception as exc:
        logger.warning("on_session_start hook failed: %s", exc)

    # Cold-start credits seed (L3) fallback for the first-turn path; TUI/desktop seed at
    # session open, so this is idempotent (skips when _credits_state exists). Fail-open.
    try:
        from agent.credits_tracker import seed_credits_at_session_start

        seed_credits_at_session_start(agent)
    except Exception:
        logger.debug("cold-start credits seed failed (fail-open)", exc_info=True)

    # Persist the system prompt snapshot; the gateway path (fresh AIAgent per turn)
    # reads this row every turn, so a failure here breaks prefix-cache reuse.
    if agent._session_db:
        try:
            agent._session_db.update_system_prompt(agent.session_id, agent._cached_system_prompt)
            from tools.mcp_tool import persist_agent_tool_names

            persist_agent_tool_names(agent)
        except Exception as exc:
            logger.warning(
                "Session DB update_system_prompt failed for session %s: "
                "%s. Subsequent turns will rebuild the system prompt and "
                "miss the prefix cache.",
                agent.session_id, exc,
            )


def _stored_prompt_matches_runtime(agent, prompt: str) -> bool:
    """Return False when the persisted runtime-identity lines are stale."""

    def line_value(label: str) -> str:
        """Last matching line wins.

        Safe ONLY for fields in the volatile tier at the END of the prompt; embedded
        project context could shadow earlier fields — see ``host_info_value``."""
        prefix = f"{label}:"
        value = ""
        for line in prompt.splitlines():
            if line.startswith(prefix):
                value = line[len(prefix):].strip()
        return value

    def host_info_value(label: str) -> str:
        """Read a field from the prompt's own host-info block.

        Anchors on the FIRST ``User home directory:`` line so a user's ``AGENTS.md`` row
        cannot match; a false mismatch would rebuild the prompt every turn."""
        prefix = f"{label}:"
        lines = prompt.splitlines()
        for idx, line in enumerate(lines):
            if not line.startswith("User home directory:"):
                continue
            for candidate in lines[idx + 1: idx + 4]:
                if candidate.startswith(prefix):
                    return candidate[len(prefix):].strip()
        return ""

    stored_model = line_value("Model")
    current_model = str(getattr(agent, "model", "") or "").strip()
    if stored_model and current_model and stored_model != current_model:
        return False

    stored_provider = line_value("Provider")
    current_provider = str(getattr(agent, "provider", "") or "").strip()
    if stored_provider and current_provider and stored_provider != current_provider:
        return False

    # cwd drift check. Compare against resolve_agent_cwd() — the SAME resolver used to
    # build the prompt — so TERMINAL_CWD sessions are not falsely rejected.
    stored_cwd = host_info_value("Current working directory")
    if stored_cwd:
        if stored_cwd != str(resolve_agent_cwd()):
            return False

    # Runtime-surface drift: reusing a desktop-built prompt on a terminal session (or
    # vice versa) would inject the wrong runtime hints.
    stored_platform = line_value("Platform")
    current_platform = str(getattr(agent, "platform", "") or "").strip()
    if stored_platform and current_platform and stored_platform != current_platform:
        return False

    return True


# Named constants for the _get_continuation_prompt variants so
# _is_synthetic_compression_user_turn can recognize them by content after a crash
# persists one; SessionDB projection strips the _length_continuation_nudge tag.
_LENGTH_CONTINUATION_NETWORK_STUB = (
    "[System: The previous response was cut off by a "
    "network error mid-stream. Continue exactly where "
    "you left off. Do not restart or repeat prior text. "
    "Finish the answer directly.]"
)
_LENGTH_CONTINUATION_OUTPUT_LIMIT = (
    "[System: Your previous response was truncated by the output "
    "length limit. Continue exactly where you left off. Do not "
    "restart or repeat prior text. Finish the answer directly.]"
)
# The dropped-tools variant interpolates tool names, so
# _is_synthetic_compression_user_turn matches this prefix with str.startswith.
_LENGTH_CONTINUATION_DROPPED_TOOLS_PREFIX = "[System: Your previous tool call "


def _get_continuation_prompt(is_partial_stub: bool, dropped_tools: Optional[List[str]] = None) -> str:
    if is_partial_stub and dropped_tools:
        tool_list = ", ".join(dropped_tools[:3])
        return (
            f"{_LENGTH_CONTINUATION_DROPPED_TOOLS_PREFIX}"
            f"({tool_list}) was too large and "
            "the stream timed out before it "
            "could be delivered. Do NOT retry "
            "the same tool call with the same "
            "large content. Instead, break the "
            "content into multiple smaller tool "
            "calls (e.g. use multiple patch calls "
            "or write smaller files). Each tool "
            "call's arguments must be under ~8K "
            "tokens to avoid stream timeouts.]"
        )
    elif is_partial_stub:
        return _LENGTH_CONTINUATION_NETWORK_STUB
    else:
        return _LENGTH_CONTINUATION_OUTPUT_LIMIT


# Nudge for Codex/Responses turns that returned only internal reasoning: a bare retry
# would be byte-identical (nothing replayable emitted), so the model repeats it.
_CODEX_INCOMPLETE_NUDGE = (
    "[System: Your previous response contained only internal reasoning and "
    "never produced a visible answer or tool call. Do not keep thinking. "
    "Produce your final answer as plain text now (or make the tool call "
    "you were planning).]"
)


# Re-prompt after an acknowledgment-only Codex/Responses reply; named so
# _is_synthetic_compression_user_turn can recognize it like _CODEX_INCOMPLETE_NUDGE.
_CODEX_ACK_CONTINUATION_NUDGE = (
    "[System: Continue now. Execute the required tool calls and only "
    "send your final answer after completing the task.]"
)

# Re-prompt for finish_reason="tool_calls" with empty tool_calls. Named like
# _CODEX_ACK_CONTINUATION_NUDGE: an interrupt mid-retry can persist it.
_DROPPED_TOOLCALL_NUDGE_CONTENT = (
    "Your previous turn indicated a tool call but none was "
    "included. Do not narrate a plan or restate intent — issue "
    "the actual tool call now to continue the task."
)

# Re-prompt for an empty response after tool calls (#9400). Named because its
# _empty_recovery_synthetic metadata flag does not survive SessionDB projection.
_EMPTY_TOOL_RESPONSE_NUDGE = (
    "You just executed tool calls but returned an "
    "empty response. Please process the tool "
    "results above and continue with the task."
)


# Shared recovery trailer for both content-policy refusal paths (HTTP-200
# content_filter and the content_policy_blocked exception) so guidance cannot drift.
_CONTENT_POLICY_RECOVERY_HINT = (
    "Try rephrasing the request, narrowing the context, or "
    "adding a fallback provider with `hermes fallback add`."
)


# Memo for send-path tool-call argument canonicalization, which re-runs on every
# historical call each iteration. Sound: canonicalization is pure and deterministic;
# malformed strings raise before being stored, so the repair fallback is never memoized.
_CANON_ARGS_CACHE: Dict[str, str] = {}
_CANON_ARGS_CACHE_MAX = 4096
# Count bound alone does not bound MEMORY: argument strings can run 100KB+, so a byte
# budget bounds the worst case while keeping the memo effective for ~0.5-2KB args.
_CANON_ARGS_CACHE_MAX_BYTES = 32 * 1024 * 1024
_canon_args_cache_bytes = 0


def _canonicalize_tool_call_arguments(arg_str: str) -> str:
    """Return the canonical wire form of a tool-call arguments JSON string.

    Raises whatever ``json.loads`` raises on malformed input; the caller falls back to
    ``_repair_tool_call_arguments``."""
    global _canon_args_cache_bytes
    cached = _CANON_ARGS_CACHE.get(arg_str)
    if cached is not None:
        return cached
    canonical = json.dumps(
        json.loads(arg_str), separators=(",", ":"), sort_keys=True,
    )
    _CANON_ARGS_CACHE[arg_str] = canonical
    _canon_args_cache_bytes += len(arg_str) + len(canonical)
    while len(_CANON_ARGS_CACHE) > _CANON_ARGS_CACHE_MAX or (
        _canon_args_cache_bytes > _CANON_ARGS_CACHE_MAX_BYTES
        and len(_CANON_ARGS_CACHE) > 1
    ):
        try:
            evicted_key = next(iter(_CANON_ARGS_CACHE))
            evicted_val = _CANON_ARGS_CACHE.pop(evicted_key)
            _canon_args_cache_bytes -= len(evicted_key) + len(evicted_val)
        except (StopIteration, KeyError, RuntimeError):
            break
    return canonical


def _clone_message_for_send(msg):
    """Structural clone of a history message for the per-call API copy.

    Clones every dict/list recursively while sharing immutable leaves, so in-place
    send-path rewrites can never reach the persisted transcript (#80498). Cheaper than
    copy.deepcopy; messages are JSON-shaped and acyclic, tuples are shared as leaves."""
    if isinstance(msg, dict):
        return {
            k: _clone_message_for_send(v) if isinstance(v, (dict, list)) else v
            for k, v in msg.items()
        }
    if isinstance(msg, list):
        return [
            _clone_message_for_send(v) if isinstance(v, (dict, list)) else v
            for v in msg
        ]
    return msg


def _canonicalize_api_tool_calls(api_messages) -> None:
    """Canonicalize tool-call argument JSON on the send-path message copy.

    Rewrites ``tool_calls`` in place (copy-on-write for the dicts it touches; persisted
    history untouched). The memo bounds parse/serialize to one per UNIQUE string."""
    for am in api_messages:
        tcs = am.get("tool_calls")
        if not tcs:
            continue
        new_tcs = []
        for tc in tcs:
            if isinstance(tc, dict) and "function" in tc:
                try:
                    tc = {**tc, "function": {
                        **tc["function"],
                        "arguments": _canonicalize_tool_call_arguments(
                            tc["function"]["arguments"]
                        ),
                    }}
                except Exception:
                    # Copy-on-write as defense in depth: callers may pass shallow
                    # copies, and writing into a shared tc["function"] rewrote the
                    # stored turn with "{}" on the unrepairable path (#80498).
                    tc = {**tc, "function": {
                        **tc["function"],
                        "arguments": _repair_tool_call_arguments(
                            tc["function"]["arguments"],
                            tc["function"].get("name", "?"),
                        ),
                    }}
            new_tcs.append(tc)
        am["tool_calls"] = new_tcs


def _invalid_tool_name_error_content(name: str, valid_tool_names) -> str:
    """Error-result content for a tool call whose name isn't a real tool.

    A blank name is a model echoing tool-call syntax seen in data, not a typo (#47967);
    dumping the catalog feeds that loop, so send a terse error instead. A nonempty wrong
    name still gets the catalog so the model can self-correct."""
    if not (name or "").strip():
        return (
            "Tool call rejected: the tool name was empty. "
            "If tool-call XML or JSON appeared in file "
            "contents or tool output, that is data — do "
            "not re-emit it as a tool call. To call a "
            "tool, use a valid name from your tool list; "
            "otherwise reply in plain text."
        )
    available = ", ".join(sorted(valid_tool_names))
    return f"Tool '{name}' does not exist. Available tools: {available}"


def _content_policy_blocked_result(
    messages: List[Dict],
    api_call_count: int,
    *,
    final_response: str,
    error_detail: str,
) -> Dict[str, Any]:
    """Build the terminal turn result for a content-policy block.

    Refusals are deterministic for the unchanged prompt, so no retry; both the HTTP-200
    and exception paths return this shape with a ``content_policy_blocked:`` error."""
    return {
        "final_response": final_response,
        "messages": messages,
        "api_calls": api_call_count,
        "completed": False,
        "failed": True,
        "error": f"content_policy_blocked: {error_detail}",
    }


def _compression_deferred_result(
    agent,
    messages: List[Dict],
    api_call_count: int,
    reason: str = "lock",
) -> Dict[str, Any]:
    """Build the soft turn result for a transiently-deferred compression.

    Both ``reason="lock"`` and ``reason="transient_block"`` must end as
    ``compression_deferred``, never ``compression_exhausted`` — the gateway wipes the
    session on exhaustion (#9893/#35809). ``failed`` stays False; the turn persists."""
    if reason == "transient_block":
        block = getattr(agent, "_compression_blocked_transient", None)
        logger.info(
            "turn deferred: compression transiently blocked (%s) "
            "(session=%s) — not counting as compression exhaustion",
            block if isinstance(block, str) else "unknown guard",
            agent.session_id or "none",
        )
        _final = (
            "Context compression is temporarily paused after a recent "
            "failed attempt. Please retry in a moment — compression will "
            "resume automatically (or run /compress to force a retry now)."
        )
    else:
        holder = getattr(agent, "_compression_skipped_due_to_lock", None)
        logger.info(
            "turn deferred: compression lock held by another path "
            "(session=%s holder=%s) — not counting as compression exhaustion",
            agent.session_id or "none",
            holder if isinstance(holder, str) else "unconfirmed",
        )
        _final = (
            "Context compression is already running for this session. "
            "Please retry in a moment — your next message will be processed "
            "once the concurrent compression finishes."
        )
    try:
        agent._flush_status_buffer()
    except Exception:
        pass
    return {
        "final_response": _final,
        "messages": messages,
        "completed": False,
        "api_calls": api_call_count,
        "error": _final,
        "partial": True,
        "failed": False,
        "compression_deferred": True,
        "session_id": agent.session_id,
    }


def _provider_overflow_exhausted_result(
    agent,
    messages: List[Dict],
    conversation_history,
    api_call_count: int,
    request_pressure_tokens: int,
    max_compression_attempts: int,
) -> Dict[str, Any]:
    """Fail closed when a rebuilt request is still too large after recovery."""
    agent._flush_status_buffer()
    logger.error(
        "%sContext compression failed after %d attempts; rebuilt request "
        "remains over threshold at ~%s tokens.",
        agent.log_prefix,
        max_compression_attempts,
        f"{request_pressure_tokens:,}",
    )
    agent._persist_session(messages, conversation_history)
    final_response = (
        "Context length exceeded: compression could not reduce the rebuilt "
        "request below the safe threshold."
    )
    return {
        "final_response": final_response,
        "messages": messages,
        "completed": False,
        "api_calls": api_call_count,
        "error": final_response,
        "partial": True,
        "failed": True,
        "compression_exhausted": True,
        "turn_exit_reason": "context_compression_exhausted",
    }


def _rewrite_system_content_blocks(system_message: dict, effective: str) -> bool:
    """Rewrite a cache-decorated system message in place, keeping its blocks.

    Assigning a bare string over the ``[static prefix, volatile tail]`` block list drops
    both cache_control breakpoints. Only the LAST ``Model:``/``Provider:`` lines change.
    Returns False when the shape cannot be safely patched."""
    content = system_message.get("content")
    if not isinstance(content, list) or not content:
        return False
    if not all(
        isinstance(part, dict) and part.get("type") == "text" for part in content
    ):
        return False
    if len(content) == 1:
        content[0]["text"] = effective
        return True
    if len(content) == 2:
        head = content[0].get("text") or ""
        if head and effective.startswith(head):
            tail = effective[len(head):]
            if tail:
                content[1]["text"] = tail
                return True
    return False


def _sync_failover_system_message(agent, api_messages, active_system_prompt):
    """Refresh the in-flight system message after a provider failover.

    ``try_activate_fallback`` rewrites the identity lines on ``_cached_system_prompt``,
    but this call block's ``api_messages`` were built pre-failover and are reused each
    retry. Mutates ``api_messages[0]`` in place; returns the new ``active_system_prompt``."""
    sp = getattr(agent, "_cached_system_prompt", None)
    if not isinstance(sp, str) or not sp:
        return active_system_prompt
    if api_messages and api_messages[0].get("role") == "system":
        effective = sp
        if agent.ephemeral_system_prompt:
            effective = (effective + "\n\n" + agent.ephemeral_system_prompt).strip()
        if not _rewrite_system_content_blocks(api_messages[0], effective):
            api_messages[0]["content"] = effective
    return sp


def _arm_fallback_restart(agent, api_messages, active_system_prompt, _retry):
    """After ``_try_activate_fallback`` succeeded: sync the system message to the new
    provider and arm ``restart_with_rebuilt_messages`` (re-issue against the fallback,
    refunding the stalled attempt). Callers also reset ``retry_count`` /
    ``compression_attempts`` to 0 and ``break`` the retry loop."""
    active_system_prompt = _sync_failover_system_message(
        agent, api_messages, active_system_prompt)
    _retry.primary_recovery_attempted = False
    _retry.restart_with_rebuilt_messages = True
    return active_system_prompt


def _ensure_cached_system_prompt_static(agent, system_message=None) -> None:
    """Rebuild ``_cached_system_prompt_static`` when caching becomes active (#72626).

    Sessions restored under a cache-off primary skip the static-prefix rebuild; a later
    failover to a cache-on provider would otherwise silently fall back to the legacy
    system-plus-3 layout. Wraps ``reconstruct_static_prefix`` (memoizes failures)."""
    from agent.system_prompt import reconstruct_static_prefix

    reconstruct_static_prefix(
        agent, system_message=system_message, log_label="failover redecoration"
    )


def _peel_moa_guidance(
    messages: List[Dict[str, Any]],
    guidance: Any,
) -> List[Dict[str, Any]]:
    """Remove MoA reference guidance attached by ``_attach_reference_guidance``.

    Kept adjacent to the attach so the forward/inverse shapes evolve together."""
    from agent.moa_loop import peel_reference_guidance

    return peel_reference_guidance(messages, guidance)


def _redecorate_prompt_cache_for_provider(
    agent,
    api_messages: List[Dict[str, Any]],
    *,
    system_message=None,
    moa_prepared: Optional[Dict[str, Any]] = None,
    tools_for_api: Optional[List[Dict[str, Any]]] = None,
) -> tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]] | tuple[List[Dict[str, Any]], Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    """Strip and re-apply cache_control for the *current* provider policy.

    Decoration runs once per call block for the primary provider, but failover
    ``continue`` paths reuse ``api_messages`` (#72626), so reshape at the top of each
    retry from the mutated in-flight request. MoA guidance is peeled and rebased."""
    messages: List[Dict[str, Any]] = [
        dict(m) if isinstance(m, dict) else m for m in (api_messages or [])
    ]
    prepared = moa_prepared
    guidance = prepared.get("guidance") if isinstance(prepared, dict) else None
    if guidance:
        messages = _peel_moa_guidance(messages, guidance)

    strip_anthropic_cache_control(messages)
    planned_tools = strip_anthropic_tool_cache_control(
        tools_for_api if tools_for_api is not None else getattr(agent, "tools", [])
    )

    if prepared is not None and getattr(agent, "provider", None) == "moa":
        # Prepared MoA state is canonical: the synchronous acting-aggregator
        # sender owns its destination-local cache plan after it resolves the slot.
        completions = getattr(getattr(agent.client, "chat", None), "completions", None)
        rebase = getattr(completions, "rebase_prepared_request", None)
        if callable(rebase):
            prepared = rebase(prepared, messages)
            messages = prepared["messages"]
        if tools_for_api is None:
            return messages, prepared
        return messages, prepared, planned_tools

    # Direct attribute access, not getattr: the flags are always initialized on
    # AIAgent, and a default would mask a real init bug as silent cache-off.
    if agent._use_prompt_caching:
        _ensure_cached_system_prompt_static(agent, system_message=system_message)
        static = getattr(agent, "_cached_system_prompt_static", None)
        direct_tool_cache = getattr(
            agent,
            "_direct_native_anthropic_tool_cache_capability",
            lambda: False,
        )()
        from agent.prompt_caching import envelope_tool_part_cache_markers_supported

        plan = build_prompt_cache_plan(
            messages,
            planned_tools,
            # Clamp per-destination: a configured 1h regresses to 5m on
            # Qwen/Alibaba routes, whose context cache is 5m-only (#84733).
            cache_ttl=effective_cache_ttl(
                agent._cache_ttl,
                provider=agent.provider,
                model=agent.model,
            ),
            native_anthropic=agent._use_native_cache_layout,
            static_system_prefix=static if isinstance(static, str) else None,
            direct_native_tool_cache=direct_tool_cache,
            # LiteLLM-style envelope routes forward part-level markers into
            # tool_result.content[] → non-retryable 400 (#89886).
            tool_part_markers=envelope_tool_part_cache_markers_supported(
                getattr(agent, "provider", ""), getattr(agent, "base_url", "")
            ),
        )
        messages = plan.messages
        planned_tools = plan.tools

    if tools_for_api is None:
        return messages, prepared
    return messages, prepared, planned_tools


def _apply_context_engine_selection(
    agent: Any,
    api_messages: List[Dict[str, Any]],
    conversation_messages: List[Dict[str, Any]],
    incoming_message: Optional[Dict[str, Any]],
    *,
    logger: Any,
) -> List[Dict[str, Any]]:
    """Run the optional per-turn ``ContextEngine.select_context()`` hook.

    Returns the (possibly replaced) request list. Fail-open: a missing hook, exception,
    or invalid return yields ``api_messages`` unchanged; history is never mutated."""
    engine = getattr(agent, "context_compressor", None)
    if engine is None or not hasattr(engine, "select_context"):
        return api_messages

    # Skip the no-op base ``select_context`` so non-implementing engines pay nothing;
    # ``hasattr`` is not enough: the ABC defines a default. Lazy import avoids a cycle.
    try:
        from agent.context_engine import ContextEngine as _CE
        if getattr(engine.select_context, "__func__", None) is _CE.select_context:
            return api_messages
    except Exception:
        pass

    session_label = getattr(agent, "session_id", None) or "-"
    # Structural clones: the engine must not be able to write through nested
    # containers into persisted history; only the request list is acted on (#80498).
    _conv_copy = [_clone_message_for_send(m) for m in conversation_messages] \
        if conversation_messages is not None else None
    _incoming_copy = _clone_message_for_send(incoming_message) if isinstance(incoming_message, dict) else incoming_message
    try:
        selected = engine.select_context(
            api_messages,
            conversation_messages=_conv_copy,
            incoming_message=_incoming_copy,
            budget_tokens=getattr(engine, "context_length", 0) or 0,
        )
    except Exception:
        logger.warning(
            "Context engine select_context hook failed; using unmodified "
            "request messages (session=%s)",
            session_label,
            exc_info=True,
        )
        return api_messages

    if selected is None:
        return api_messages
    # Require a NON-EMPTY list of dicts: ``all([])`` is ``True``, so a ``[]`` from a
    # buggy engine would otherwise replace the request instead of failing open.
    if isinstance(selected, list) and selected and all(isinstance(m, dict) for m in selected):
        return selected

    logger.warning(
        "Context engine select_context returned an invalid value "
        "(not a non-empty list of dicts); ignoring (session=%s)",
        session_label,
    )
    return api_messages


def _notify_context_engine_turn_complete(
    agent: Any,
    messages: List[Dict[str, Any]],
    *,
    usage: Optional[Dict[str, Any]] = None,
    logger: Any,
    **meta: Any,
) -> None:
    """Notify the active context engine that a user turn has finished.

    Fail-open: a missing/no-op hook or any exception is swallowed. ``messages`` is
    passed as a copy so the engine cannot mutate the persisted transcript."""
    engine = getattr(agent, "context_compressor", None)
    hook = getattr(engine, "on_turn_complete", None)
    if engine is None or not callable(hook):
        return

    # Skip the no-op base ``on_turn_complete`` so non-implementing engines pay nothing
    # per turn. Lazy import avoids an import cycle with agent.context_engine.
    try:
        from agent.context_engine import ContextEngine as _CE
        if getattr(hook, "__func__", None) is _CE.on_turn_complete:
            return
    except Exception:
        pass

    try:
        hook(
            # Structural clones: dict(m) would let a hook write into nested containers
            # of the persisted transcript (#80498).
            [_clone_message_for_send(m) for m in messages],
            usage=usage,
            **meta,
        )
    except Exception:
        logger.warning(
            "Context engine on_turn_complete hook failed (session=%s)",
            getattr(agent, "session_id", None) or "-",
            exc_info=True,
        )


def run_conversation(
    agent,
    user_message: Any,
    system_message: str = None,
    conversation_history: List[Dict[str, Any]] = None,
    task_id: str = None,
    stream_callback: Optional[callable] = None,
    persist_user_message: Optional[Any] = None,
    persist_user_timestamp: Optional[float] = None,
    persist_user_display_kind: Optional[str] = None,
    persist_user_display_metadata: Optional[Dict[str, Any]] = None,
    persist_user_platform_id: Optional[str] = None,
    moa_config: Optional[dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run a complete conversation with tool calling until completion.

    Args:
        stream_callback: per-text-delta callback (TTS); None uses the non-streaming path.
        persist_user_message: clean text to store when ``user_message`` carries API-only
            synthetic prefixes; ``persist_user_timestamp`` / ``persist_user_platform_id``
            are stored as metadata (platform id lets restart drain recovery dedup).
        persist_user_display_kind/metadata: display-only event rendering (``auto_continue``,
            ``model_switch``); the model still receives the message unchanged.

    Returns: dict with the final response and message history."""
    if moa_config is None:
        try:
            from hermes_cli.moa_config import decode_moa_turn

            _decoded_message, _decoded_moa_config = decode_moa_turn(user_message)
            if _decoded_moa_config is not None:
                user_message = _decoded_message
                moa_config = _decoded_moa_config
                if persist_user_message is None:
                    persist_user_message = _decoded_message
        except Exception:
            pass

    # The gateway caches agents across turns; compression state is per-turn, or a stale
    # in-place boundary would make a later uncompressed result look compacted.
    agent._last_compaction_in_place = False
    agent._last_compression_attempt_recorded = False
    agent._last_compression_attempt_in_place = None
    begin_fast_mode_turn(agent, conversation_history)

    # Adopt ~/.hermes/.env credential/base-url edits made since the last turn — a
    # Settings save updates .env, not this worker's client (#67821). No-op if unchanged.
    try:
        agent._try_refresh_env_client_credentials()
    except Exception:
        logger.debug("per-turn env credential refresh failed", exc_info=True)

    # ── Per-turn setup (the prologue) ──
    # All once-per-turn setup lives in ``build_turn_context`` (agent/turn_context.py);
    # it mutates ``agent`` as the inline code did and returns the locals the loop reads.
    try:
        _ctx = build_turn_context(
            agent,
            user_message,
            system_message,
            conversation_history,
            task_id,
            stream_callback,
            persist_user_message,
            persist_user_timestamp,
            persist_user_display_kind=persist_user_display_kind,
            persist_user_display_metadata=persist_user_display_metadata,
            persist_user_platform_id=persist_user_platform_id,
            restore_or_build_system_prompt=_restore_or_build_system_prompt,
            install_safe_stdio=_install_safe_stdio,
            sanitize_surrogates=_sanitize_surrogates,
            summarize_user_message_for_log=_summarize_user_message_for_log,
            set_session_context=set_session_context,
            set_current_write_origin=set_current_write_origin,
            ra=_ra,
            # MoA turns append per-call aggregated context to the API copy of the
            # user message, so no byte-stable api_content sidecar can be stamped.
            moa_active=bool(moa_config),
        )
    except PreflightCompressionTimedOut as _preflight_timeout_exc:
        # Preflight compression timed out; no provider call sent (#98424). Return the
        # typed recovery result: surfaces hide raw exception text, which would bury the
        # actionable guidance and skip the compression_exhausted recovery contract.
        logger.warning(
            "Turn-start preflight compression timed out — ending turn with "
            "typed recovery result: %s",
            _preflight_timeout_exc,
        )
        # Clear the tripwire slot note_turn_start registered; the early return skips the
        # persist funnel that clears it. The user row is deliberately NOT persisted:
        # the gateway skips persistence for compression_exhausted results (#7100).
        from agent.agent_runtime_helpers import note_turn_persisted

        note_turn_persisted(agent)
        # Not _COMPRESSION_TIMEOUT_FINAL_RESPONSE — that describes a different state
        # (compression ran, could not reduce); the exception text carries the guidance.
        _final_response = str(_preflight_timeout_exc)
        return {
            "final_response": _final_response,
            "messages": list(conversation_history or []),
            "completed": False,
            "api_calls": 0,
            "error": _final_response,
            "partial": True,
            "failed": True,
            "compression_exhausted": True,
            "turn_exit_reason": "context_compression_timeout",
        }
    user_message = _ctx.user_message
    original_user_message = _ctx.original_user_message
    messages = _ctx.messages
    conversation_history = _ctx.conversation_history
    active_system_prompt = _ctx.active_system_prompt
    effective_task_id = _ctx.effective_task_id
    turn_id = _ctx.turn_id
    current_turn_user_idx = _ctx.current_turn_user_idx
    _should_review_memory = _ctx.should_review_memory
    _plugin_user_context = _ctx.plugin_user_context
    _ext_prefetch_cache = _ctx.ext_prefetch_cache

    # Commentary deduplication spans all provider continuations and tool calls
    # within one user turn, but must not suppress the same phrase next turn.
    agent._delivered_interim_texts = set()
    # A configured SessionDB append failure halts only the affected turn. A
    # cached gateway agent must recover on the next message if storage did.
    agent._incremental_persistence_failed = False
    # Cause of the last persistence failure this turn ('locked'/'disk'/'unknown', see
    # hermes_state.classify_persistence_error). Reset so a prior diagnosis cannot leak.
    agent._last_persistence_error_cause = None
    # Per-turn diagnostic: a failed compression-tip adoption in a previous
    # turn's flush must not be reported against this turn.
    agent._compression_adoption_failed = False

    # Main conversation loop counters (pure locals consumed by the loop below).
    api_call_count = 0
    final_response = None
    interrupted = False
    failed = False
    codex_ack_continuations = 0
    length_continue_retries = 0
    # Turn-scoped one-shot: armed by a thinking-only truncation, consumed by
    # build_api_kwargs; must not survive an interrupted turn into the next one.
    agent._ephemeral_reasoning_off = False
    # Total outer-loop exceptions this turn (#92450) — see _MAX_OUTER_LOOP_ERRORS.
    _outer_error_count = 0
    truncated_tool_call_retries = 0
    truncated_response_parts: List[str] = []
    compression_attempts = 0
    # Per-turn compression attempt cap shared by the pre-API gate, 413 handlers and
    # post-tool compaction; a consecutive-ineffective-attempt backstop, rearmed only
    # after a provider response reports a prompt below threshold. Default 3 if unset.
    max_compression_attempts = getattr(agent, "max_compression_attempts", 3)
    _last_preflight_pressure: Optional[int] = None
    _preflight_compression_blocked = _ctx.preflight_compression_blocked
    # A provider overflow outweighs the rough-estimate calibration that defers preflight
    # after compaction: stay armed until the rebuilt request is below the threshold.
    _provider_overflow_recovery_pending = False
    # Armed when a compression host-timeout ends the turn; finalize reuses the gateway
    # context-recovery contract (error/partial/compression_exhausted) (#98722).
    _compression_timeout_exhausted = False
    _turn_exit_reason = "unknown"  # Diagnostic: why the loop ended
    # Last answer held back by a verification gate: if the continuation exhausts the
    # budget this is the best user-facing result, distinct from error/recovery text.
    _pending_verification_response = None
    # Whether the pending verification candidate was already streamed as interim.
    # ``_response_was_previewed`` is set ONLY if it becomes the final response (#65919).
    _pending_verification_response_previewed = False
    # If pre-API compression fires after MoA advisors ran, retain their guidance and
    # rebase it onto the compacted transcript next iteration — no second fan-out.
    pending_moa_prepared_request = None

    # Per-turn tally of credential-pool refreshes by (provider, pool-entry-id): caps
    # same-entry refreshes on a persistent 401 so fallback takes over (#26080).
    agent._auth_pool_refresh_counts = {}

    # Per-turn usage forwarded to the context engine's on_turn_complete() hook; left
    # None on turns that never reach a response so the hook never sees stale usage.
    agent._last_turn_usage = None

    # Opt-in runtime: api_mode == codex_app_server hands the whole turn to the codex
    # app-server subprocess (see agent/transports/codex_app_server_session.py).
    if agent.api_mode == "codex_app_server":
        return agent._run_codex_app_server_turn(
            user_message=user_message,
            original_user_message=original_user_message,
            messages=messages,
            effective_task_id=effective_task_id,
            should_review_memory=_should_review_memory,
        )

    while (api_call_count < agent.max_iterations and agent.iteration_budget.remaining > 0) or agent._budget_grace_call:
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
            break

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
            break
        
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
            break

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
        request_logger = getattr(agent, "logger", None) or logging.getLogger(__name__)
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
        from agent.agent_runtime_helpers import (
            fill_empty_non_final_wire_payload,
            repair_message_sequence_with_cursor,
        )
        repaired_seq = repair_message_sequence_with_cursor(agent, messages)
        if repaired_seq > 0:
            request_logger.info(
                "Repaired %s message-alternation violations before request (session=%s)",
                repaired_seq,
                agent.session_id or "-",
            )

        api_messages = []
        for idx, msg in enumerate(messages):

            # Structural clone, NOT msg.copy(): in-place transforms below must not reach
            # persisted history via nested containers; see _clone_message_for_send.
            api_msg = _clone_message_for_send(msg)

            # api_content is the persistence sidecar of the exact bytes sent to the API;
            # bookkeeping, never a provider field — pop it from EVERY outgoing copy.
            _api_content = api_msg.pop("api_content", None)

            # Display-only timeline metadata, never a provider field: strict OpenAI
            # backends reject unknown keys once a typed event row enters live history.
            api_msg.pop("display_kind", None)
            api_msg.pop("display_metadata", None)

            # Durable row id from _rows_to_conversation (desktop reactions); only the
            # chat-completions transport strips underscore keys, so drop it centrally.
            api_msg.pop("_row_id", None)

            # Inject ephemeral context (memory prefetch + pre_llm_call user hooks)
            # at API time only; `messages` is untouched beyond the api_content stamp.
            if idx == current_turn_user_idx and msg.get("role") == "user":
                if isinstance(_api_content, str) and _api_content:
                    # Reuse the prologue's stamp so sidecar and wire cannot drift
                    # and every pass this turn sends identical bytes.
                    api_msg["content"] = _api_content
                else:
                    # Callers that bypass the prologue stamping: compose live.
                    _composed = compose_user_api_content(
                        api_msg.get("content", ""),
                        _ext_prefetch_cache,
                        _plugin_user_context,
                    )
                    if _composed is not None:
                        api_msg["content"] = _composed
            elif (
                isinstance(_api_content, str)
                and _api_content
                and msg.get("role") in ("user", "assistant")
            ):
                # Historical row: replay the exact bytes sent live so the prompt-cache
                # prefix stays byte-stable. User rows carry the injection sidecar; user
                # and assistant rows may carry a sanitize-divergence sidecar.
                api_msg["content"] = _api_content

            # For ALL assistant messages, pass reasoning back to the API
            # This ensures multi-turn reasoning context is preserved
            agent._copy_reasoning_content_for_api(msg, api_msg)

            # Remove 'reasoning' field - it's for trajectory storage only
            # We've copied it to 'reasoning_content' for the API above
            if "reasoning" in api_msg:
                api_msg.pop("reasoning")
            # Remove finish_reason - not accepted by strict APIs (e.g. Mistral)
            if "finish_reason" in api_msg:
                api_msg.pop("finish_reason")
            # Fill empty non-final user/assistant wire copies so the pre-call sanitizer
            # stops re-healing and flooding errors.log; durable history is untouched.
            # After the reasoning copy so thinking-only turns keep payload (#96870).
            fill_empty_non_final_wire_payload(
                api_msg, is_final=(idx == len(messages) - 1)
            )
            # _thinking_prefill survives intentionally: the drop pass below needs it.
            # Strip length-continuation marks; some transports keep underscore keys.
            api_msg.pop("_length_continuation_fragment", None)
            api_msg.pop("_length_continuation_nudge", None)
            # Strip Codex Responses fields (call_id, response_item_id): strict providers
            # reject unknown fields. New dicts keep the internal list intact for Codex.
            if agent._should_sanitize_tool_calls():
                # In MoA mode agent.model is the virtual preset name; use the resolved
                # aggregator so Gemini keeps thought_signature (extra_content).
                _sanitize_model = agent.model
                if agent.provider == "moa":
                    if moa_config:
                        _agg = moa_config.get("aggregator") or {}
                        if _agg.get("model"):
                            _sanitize_model = _agg["model"]
                    if _sanitize_model == agent.model:
                        # Virtual-provider mode: no moa_config is threaded through; ask
                        # the facade for the aggregator slot from the previous create().
                        _moa_client = getattr(agent, "client", None)
                        _agg_slot = getattr(_moa_client, "last_aggregator_slot", None)
                        if _agg_slot and _agg_slot.get("model"):
                            _sanitize_model = _agg_slot["model"]
                agent._sanitize_tool_calls_for_strict_api(api_msg, model=_sanitize_model)
            # Keep 'reasoning_details' - OpenRouter uses this for multi-turn reasoning context
            # The signature field helps maintain reasoning continuity
            api_messages.append(api_msg)

        # Final system message = cached prompt + ephemeral additions (API-time only).
        # Plugin/recall context goes into the user message, never the system prompt: the
        # prompt is built ONCE per session and replayed verbatim (stable cache prefix).
        effective_system = active_system_prompt or ""
        if agent.ephemeral_system_prompt:
            effective_system = (effective_system + "\n\n" + agent.ephemeral_system_prompt).strip()
        if effective_system:
            api_messages = [{"role": "system", "content": effective_system}] + api_messages

        if moa_config:
            try:
                from agent.message_content import flatten_message_text as _flatten_mt
                from agent.moa_loop import _preset_temperature, aggregate_moa_context

                _moa_context = aggregate_moa_context(
                    user_prompt=(
                        original_user_message
                        if isinstance(original_user_message, str)
                        # Multimodal content list: extract visible text rather than
                        # str()-ing parts, which would leak base64 image payloads.
                        else _flatten_mt(original_user_message)
                    ),
                    api_messages=api_messages,
                    reference_models=moa_config.get("reference_models") or [],
                    aggregator=moa_config.get("aggregator") or {},
                    temperature=_preset_temperature(moa_config, "reference_temperature"),
                    aggregator_temperature=_preset_temperature(moa_config, "aggregator_temperature"),
                    reference_max_tokens=moa_config.get("reference_max_tokens"),
                    # None = no per-preset override; inherit
                    # auxiliary.moa_reference.timeout via call_llm.
                    reference_timeout=(
                        float(moa_config["reference_timeout"])
                        if moa_config.get("reference_timeout")
                        else None
                    ),
                    degraded_reference_policy=str(
                        moa_config.get("degraded_reference_policy") or "loud"
                    ),
                    agent=agent,
                )
                if _moa_context:
                    for _msg in reversed(api_messages):
                        if _msg.get("role") == "user":
                            _base = _msg.get("content", "")
                            if isinstance(_base, str):
                                _msg["content"] = _base + "\n\n" + _moa_context
                            elif isinstance(_base, list):
                                # Multimodal turn: append MoA context as a trailing text
                                # part instead of silently dropping it.
                                _msg["content"] = [
                                    *_base,
                                    {"type": "text", "text": "\n\n" + _moa_context},
                                ]
                            break
            except Exception as _moa_exc:
                logger.warning("MoA context aggregation failed: %s", _moa_exc)

        # Inject ephemeral prefill messages right after the system prompt
        # but before conversation history. Same API-call-time-only pattern.
        if agent.prefill_messages:
            sys_offset = 1 if (api_messages and api_messages[0].get("role") == "system") else 0
            for idx, pfm in enumerate(agent.prefill_messages):
                # Structural clone: the in-place sanitizers below must not write
                # through into agent.prefill_messages' nested containers.
                api_messages.insert(sys_offset + idx, _clone_message_for_send(pfm))

        # Per-turn context selection hook: an engine may select/replace context for THIS
        # call only — request-only, fail-open, and independent of should_compress().
        _sel_incoming = (
            messages[current_turn_user_idx]
            if 0 <= current_turn_user_idx < len(messages)
            else None
        )
        api_messages = _apply_context_engine_selection(
            agent,
            api_messages,
            messages,
            _sel_incoming,
            logger=request_logger,
        )

        # Runs unconditionally (not gated on context_compressor) so orphaned tool
        # results from session loading or manual message edits are always caught.
        api_messages = agent._sanitize_api_messages(api_messages)

        # One-time repeated-heal notice goes out via the status/warning callback, NEVER
        # appended to messages: the cached prompt prefix stays byte-identical (#96870).
        try:
            from agent.agent_runtime_helpers import (
                consume_pending_sanitizer_heal_notice,
            )

            _heal_notice = consume_pending_sanitizer_heal_notice()
            if _heal_notice:
                agent._emit_warning(_heal_notice)
        except Exception:
            # A notice hiccup must never break the send path.
            logger.debug("sanitizer heal notice delivery failed", exc_info=True)

        # Drop thinking-only assistant turns + merge adjacent users, API copy only:
        # Anthropic-style backends 400 on a trailing `thinking` block; history keeps it.
        api_messages = agent._drop_thinking_only_and_merge_users(
            api_messages,
            drop_codex_reasoning_items=agent.api_mode != "codex_responses",
        )

        # Normalize whitespace and tool-call JSON for bit-perfect prefixes across turns
        # (KV-cache reuse on local servers, better cloud cache hits); API copy only.
        for am in api_messages:
            if isinstance(am.get("content"), str):
                am["content"] = am["content"].strip()
        _canonicalize_api_tool_calls(api_messages)

        # Strip lone surrogates (U+D800-U+DFFF) that some Ollama-served models emit;
        # they crash json.dumps() inside the OpenAI SDK and trigger the 3-retry cycle.
        _sanitize_messages_surrogates(api_messages)

        # No send-time pad loop here: ``repair_empty_non_final_messages`` (inside
        # ``_sanitize_api_messages``) is the single owner of empty-turn repair, and its
        # non-whitespace placeholder survives normalization regardless of ordering.

        # Build the request-local cache sections LAST, after every transcript mutation;
        # the canonical tool registry stays undecorated. Marked ``content`` becomes text
        # blocks the whitespace pass skips, so the same row's bytes vary across turns.
        tools_for_api = agent.tools
        if agent._use_prompt_caching and agent.provider != "moa":
            from agent.prompt_caching import (
                envelope_tool_part_cache_markers_supported,
            )

            _static_system_prefix = getattr(agent, "_cached_system_prompt_static", None)
            _initial_cache_plan = build_prompt_cache_plan(
                api_messages,
                tools_for_api,
                # Clamp per-destination: a configured 1h regresses to 5m on
                # Qwen/Alibaba routes, whose context cache is 5m-only (#84733).
                cache_ttl=effective_cache_ttl(
                    agent._cache_ttl,
                    provider=agent.provider,
                    model=agent.model,
                ),
                native_anthropic=agent._use_native_cache_layout,
                static_system_prefix=(
                    _static_system_prefix
                    if isinstance(_static_system_prefix, str)
                    else None
                ),
                direct_native_tool_cache=agent._direct_native_anthropic_tool_cache_capability(),
                # LiteLLM-style envelope routes forward part-level markers into
                # tool_result.content[] → non-retryable 400 (#89886).
                tool_part_markers=envelope_tool_part_cache_markers_supported(
                    getattr(agent, "provider", ""), getattr(agent, "base_url", "")
                ),
            )
            api_messages = _initial_cache_plan.messages
            tools_for_api = _initial_cache_plan.tools

        # Prepare the persistent-MoA request before measuring compression pressure: the
        # ephemeral advisor output is absent from ``messages``; ``create()`` reuses the
        # prepared request instead of running the advisors again.
        _moa_prepared_request = None
        if agent.provider == "moa":
            _moa_completions = getattr(getattr(agent.client, "chat", None), "completions", None)
            if pending_moa_prepared_request is not None:
                _rebase_moa_request = getattr(_moa_completions, "rebase_prepared_request", None)
                if callable(_rebase_moa_request):
                    _moa_prepared_request = _rebase_moa_request(
                        pending_moa_prepared_request, api_messages
                    )
                pending_moa_prepared_request = None
            if _moa_prepared_request is None:
                _prepare_moa_request = getattr(_moa_completions, "prepare", None)
                if callable(_prepare_moa_request):
                    _moa_prepared_request = _prepare_moa_request(api_messages)
            if _moa_prepared_request is not None:
                api_messages = _moa_prepared_request["messages"]

        # One image-stripped estimate feeds both figures; tools counted separately (50+
        # tools ≈ 20-30K tokens); total_chars is a rough proxy for logs/hooks only.
        # Charge stale thinking only when the active route replays it (#84371).
        from agent.turn_context import _agent_stale_thinking_on_wire

        if _agent_stale_thinking_on_wire(agent):
            approx_tokens = estimate_messages_tokens_rough(api_messages)
        else:
            approx_tokens = estimate_messages_tokens_rough(
                api_messages, charge_stale_thinking=False
            )
        # Route-aware: native Responses compaction prunes the wire payload, so the raw
        # history figure overstates it and fires needless local compression (#96995).
        request_pressure_tokens = _midturn_request_pressure_tokens(
            agent, api_messages, effective_system or "", approx_tokens
        )
        # Usage-anchored override: real prompt_tokens (incl. system + tool schemas) +
        # delta estimate replaces the whole-history heuristic when the anchor is fresh.
        _anchored_pressure = anchored_context_tokens(
            messages, getattr(agent, "_usage_anchor", None)
        )
        if _anchored_pressure is not None:
            request_pressure_tokens = _anchored_pressure
        total_chars = approx_tokens * 4
        # Stash the rough estimate so update_from_response() can pair it with the real
        # count (should_defer_preflight_to_real_usage). getattr: test doubles lack it.
        _note_rough = getattr(
            agent.context_compressor, "note_request_rough_estimate", None
        )
        if callable(_note_rough):
            _note_rough(request_pressure_tokens)

        _runtime_context_error = _ollama_context_limit_error(
            agent, request_pressure_tokens
        )
        if _runtime_context_error:
            final_response = _runtime_context_error
            failed = True
            _turn_exit_reason = "ollama_runtime_context_too_small"
            append_message(messages, {"role": "assistant", "content": final_response})
            agent._emit_status("❌ Ollama runtime context is too small for Hermes tool use")
            api_call_count -= 1
            agent._api_call_count = api_call_count
            try:
                agent.iteration_budget.refund()
            except Exception:
                pass
            break

        # Pre-API pressure check: tool results grow a turn and last_prompt_tokens lags
        # them. Mirror the turn-prologue guard chain: defer on noisy estimate, skip in
        # failure cooldown, then should_compress() (#11529).
        _compressor = agent.context_compressor
        _preflight_threshold = int(
            getattr(_compressor, "threshold_tokens", 0) or 0
        )
        _provider_overflow_preflight = (
            _provider_overflow_recovery_pending
            and (
                _preflight_threshold <= 0
                or request_pressure_tokens >= _preflight_threshold
            )
        )
        if (
            _provider_overflow_recovery_pending
            and not _provider_overflow_preflight
        ):
            # The outer-loop rebuild includes system prompt, request-only injections and
            # tool schemas; only that full request with output runway may be sent.
            _provider_overflow_recovery_pending = False
        # Compare fully assembled requests, not raw ``messages`` (which omit
        # api_content, plugin injections, prefills, MoA context, ephemeral system text).
        _previous_preflight_pressure = _last_preflight_pressure
        _last_preflight_pressure = None
        if (
            _previous_preflight_pressure is not None
            and request_pressure_tokens >= _preflight_threshold
            and not _compression_warrants_another_preflight_pass(
                _previous_preflight_pressure,
                request_pressure_tokens,
                _preflight_threshold,
            )
        ):
            # Stop proactive retries this turn without consuming the shared overflow-
            # recovery budget; the provider's error handler may still compact.
            _preflight_compression_blocked = True
            logger.warning(
                "Pre-API compression made insufficient progress: ~%s -> "
                "~%s request tokens; skipping additional preflight passes",
                f"{_previous_preflight_pressure:,}",
                f"{request_pressure_tokens:,}",
            )
        _defer_preflight = getattr(
            _compressor, "should_defer_preflight_to_real_usage", lambda _t: False
        )
        _pf = run_preflight_compression(
            agent,
            compressor=_compressor,
            request_pressure_tokens=request_pressure_tokens,
            provider_overflow_preflight=_provider_overflow_preflight,
            preflight_compression_blocked=_preflight_compression_blocked,
            defer_preflight=_defer_preflight,
            moa_prepared_request=_moa_prepared_request,
            pending_moa_prepared_request=pending_moa_prepared_request,
            messages=messages,
            system_message=system_message,
            user_message=user_message,
            active_system_prompt=active_system_prompt,
            conversation_history=conversation_history,
            api_call_count=api_call_count,
            compression_attempts=compression_attempts,
            max_compression_attempts=max_compression_attempts,
            effective_task_id=effective_task_id,
            final_response=final_response,
            failed=failed,
            compression_timeout_exhausted=_compression_timeout_exhausted,
            turn_exit_reason=_turn_exit_reason,
        )
        messages = _pf.messages
        active_system_prompt = _pf.active_system_prompt
        conversation_history = _pf.conversation_history
        api_call_count = _pf.api_call_count
        compression_attempts = _pf.compression_attempts
        pending_moa_prepared_request = _pf.pending_moa_prepared_request
        final_response = _pf.final_response
        failed = _pf.failed
        _compression_timeout_exhausted = _pf.compression_timeout_exhausted
        _turn_exit_reason = _pf.turn_exit_reason
        if _pf.last_preflight_pressure is not None:
            _last_preflight_pressure = _pf.last_preflight_pressure
        if _pf.action == "return":
            return _pf.result
        if _pf.action == "break":
            break
        if _pf.action == "continue":
            continue

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
        
        api_start_time = time.time()
        retry_count = 0
        max_retries = agent._api_max_retries
        _retry = TurnRetryState()

        finish_reason = "stop"
        response = None  # Guard against UnboundLocalError if all retries fail
        api_kwargs = None  # Guard against UnboundLocalError in except handler
        api_request_id = f"{turn_id}:api:{api_call_count}"
        agent._current_api_request_id = api_request_id

        while retry_count < max_retries:
            # ── Nous Portal rate limit guard ──────────────────────
            # Skip the call if another session recorded a rate limit: every attempt
            # (incl. SDK retries) counts against RPH.
            if agent.provider == "nous":
                try:
                    from agent.nous_rate_guard import (
                        nous_rate_limit_remaining,
                        format_remaining as _fmt_nous_remaining,
                    )
                    _nous_remaining = nous_rate_limit_remaining()
                    if _nous_remaining is not None and _nous_remaining > 0:
                        _nous_msg = (
                            f"Nous Portal rate limit active — "
                            f"resets in {_fmt_nous_remaining(_nous_remaining)}."
                        )
                        agent._buffer_vprint(
                            f"⏳ {_nous_msg} Trying fallback..."
                        )
                        agent._buffer_status(f"⏳ {_nous_msg}")
                        if agent._try_activate_fallback():
                            active_system_prompt = _arm_fallback_restart(
                                agent, api_messages, active_system_prompt, _retry)
                            retry_count = 0
                            compression_attempts = 0
                            break
                        # No fallback available — surface buffered context
                        # so user sees the rate-limit message that led here.
                        agent._flush_status_buffer()
                        agent._persist_session(messages, conversation_history)
                        return {
                            "final_response": (
                                f"⏳ {_nous_msg}\n\n"
                                "No fallback provider available. "
                                "Try again after the reset, or add a "
                                "fallback provider in config.yaml."
                            ),
                            "messages": messages,
                            "api_calls": api_call_count,
                            "completed": False,
                            "failed": True,
                            "error": _nous_msg,
                        }
                except ImportError:
                    pass
                except Exception:
                    pass  # Never let rate guard break the agent loop

            try:
                agent._reset_stream_delivery_tracking()
                # Per-attempt first-chunk timestamp so a stale value never leaks into
                # post_api_request.
                agent._last_api_first_chunk_at = None
                # api_messages was built for the primary; a fallback (DeepSeek / Kimi /
                # MiMo) may require reasoning_content. Re-apply the echo-back pad
                # (idempotent).
                agent._reapply_reasoning_echo_for_provider(api_messages)
                # Same for prompt-cache decoration (#72626): strip the primary's
                # breakpoints and re-render for the current provider.
                api_messages, _moa_prepared_request, tools_for_api = (
                    _redecorate_prompt_cache_for_provider(
                        agent,
                        api_messages,
                        system_message=system_message,
                        moa_prepared=_moa_prepared_request,
                        tools_for_api=tools_for_api,
                    )
                )
                if tools_for_api == agent.tools:
                    api_kwargs = agent._build_api_kwargs(api_messages)
                else:
                    api_kwargs = agent._build_api_kwargs(
                        api_messages,
                        tools_for_api=tools_for_api,
                    )
                # Surrogate chokepoint (#50959): tool descriptions, extra_body and
                # kwargs strings can carry invalid code points (HTTP 400). One walk
                # makes the payload json.dumps()-safe.
                _sanitize_structure_surrogates(api_kwargs)
                if agent._force_ascii_payload:
                    _sanitize_structure_non_ascii(api_kwargs)
                if agent.api_mode == "codex_responses":
                    api_kwargs = agent._get_transport().preflight_kwargs(
                        api_kwargs,
                        allow_stream=False,
                        is_github_responses=agent._is_copilot_url(),
                        sanitize_harmony_tokens=agent._is_codex_backend(),
                    )
                # OpenRouter caching replays identical responses, even empty ones; an
                # empty-response retry must bypass the cache.
                if agent._empty_content_retries > 0 and agent._is_openrouter_url():
                    _xh = dict(api_kwargs.get("extra_headers") or {})
                    _xh["X-OpenRouter-Cache"] = "false"
                    api_kwargs["extra_headers"] = _xh
                # Copilot x-initiator: first call of a user turn is "user" (billed
                # premium); tool-loop follow-ups keep the default "agent" (#3040).
                if getattr(agent, "_is_user_initiated_turn", False) and agent._is_copilot_url():
                    _xh = dict(api_kwargs.get("extra_headers") or {})
                    _xh["x-initiator"] = "user"
                    api_kwargs["extra_headers"] = _xh
                    agent._is_user_initiated_turn = False
                try:
                    from hermes_cli.middleware import apply_llm_request_middleware

                    _llm_request_mw = apply_llm_request_middleware(
                        api_kwargs,
                        task_id=effective_task_id,
                        turn_id=turn_id,
                        api_request_id=api_request_id,
                        session_id=agent.session_id or "",
                        platform=agent.platform or "",
                        model=agent.model,
                        provider=agent.provider,
                        base_url=agent.base_url,
                        api_mode=agent.api_mode,
                        api_call_count=api_call_count,
                    )
                    api_kwargs = _llm_request_mw.payload
                    _original_api_kwargs = _llm_request_mw.original_payload
                    _llm_middleware_trace = _llm_request_mw.trace
                except Exception:
                    _original_api_kwargs = dict(api_kwargs)
                    _llm_middleware_trace = []

                try:
                    from hermes_cli.lifecycle import (
                        has_hook,
                        invoke_hook as _invoke_hook,
                    )
                    if has_hook("pre_api_request"):
                        request_messages = api_kwargs.get("messages")
                        if not isinstance(request_messages, list):
                            request_messages = api_kwargs.get("input")
                        if not isinstance(request_messages, list):
                            request_messages = api_messages
                        # Shallow copy: plugins may retain the list; deepcopy is costly.
                        # ``request_messages``/``conversation_history`` are raw langfuse
                        # passthroughs.
                        _request_payload = agent._api_request_payload_for_hook(api_kwargs)
                        # Anthropic (``system``) and Responses/Codex (``instructions``)
                        # move the system prompt out of messages; pass it for
                        # observability.
                        system_prompt_for_hooks = _system_prompt_for_hooks(
                            api_kwargs, request_messages
                        )
                        _invoke_hook(
                            "pre_api_request",
                            task_id=effective_task_id,
                            turn_id=turn_id,
                            api_request_id=api_request_id,
                            session_id=agent.session_id or "",
                            user_message=original_user_message,
                            conversation_history=list(messages),
                            platform=agent.platform or "",
                            model=agent.model,
                            provider=agent.provider,
                            base_url=agent.base_url,
                            api_mode=agent.api_mode,
                            api_call_count=api_call_count,
                            retry_count=retry_count,
                            request_messages=list(request_messages)
                            if isinstance(request_messages, list)
                            else [],
                            system_prompt=system_prompt_for_hooks,
                            message_count=len(api_messages),
                            tool_count=len(agent.tools or []),
                            approx_input_tokens=approx_tokens,
                            request_char_count=total_chars,
                            max_tokens=agent.max_tokens,
                            started_at=api_start_time,
                            middleware_trace=list(_llm_middleware_trace),
                            request=_request_payload,
                        )
                except Exception:
                    pass

                if env_var_enabled("HERMES_DUMP_REQUESTS"):
                    agent._dump_api_request_debug(api_kwargs, reason="preflight")

                # Private to the in-process MoA facade; add after middleware/hooks/debug
                # dumps so none serializes it into the provider payload.
                if _moa_prepared_request is not None and agent.provider == "moa":
                    # Re-read the live client: rotation/fallback/cleanup rebuild
                    # agent.client between attempts; a native OpenAI client rejects this
                    # key (TypeError).
                    if _moa_client_consumes_prepared_request(agent.client):
                        api_kwargs["_moa_prepared_request"] = _moa_prepared_request
                    else:
                        logger.warning(
                            "MoA client replaced mid-turn (client=%s); sending the "
                            "prepared prompt without the MoA handshake",
                            type(agent.client).__name__,
                        )

                # Always prefer streaming even without consumers: it gives stale-
                # stream/read-timeout health checks that quiet callers otherwise lack.
                # Falls back if unsupported.
                def _stop_spinner():
                    nonlocal thinking_spinner
                    if thinking_spinner:
                        thinking_spinner.stop("")
                        thinking_spinner = None
                    if agent.thinking_callback:
                        agent.thinking_callback("")

                _use_streaming = True
                # Provider signaled "stream not supported": stay non-streaming for the
                # session.
                if getattr(agent, "_disable_streaming", False):
                    _use_streaming = False
                # ACP clients (`acp://` scheme, any vendor) return a plain
                # SimpleNamespace, not a stream; mirrors the Responses API exclusion.
                elif (
                    agent.provider in {"copilot-acp"}
                    or str(agent.base_url or "").lower().startswith("acp://")
                    or str(agent.base_url or "").lower().startswith("acp+tcp://")
                ):
                    _use_streaming = False
                # MoA streams only with a display/TTS consumer
                # (MoAChatCompletions.create() honors stream=True); else complete-
                # response path.
                elif agent.provider == "moa" and not agent._has_stream_consumers():
                    _use_streaming = False
                elif not agent._has_stream_consumers():
                    # No consumer: still stream for health checking, except Mock clients
                    # in tests (SimpleNamespace, not stream iterators).
                    from unittest.mock import Mock
                    if isinstance(getattr(agent, "client", None), Mock):
                        _use_streaming = False

                def _perform_api_call(next_api_kwargs):
                    if agent.api_mode == "codex_responses":
                        next_api_kwargs = agent._get_transport().preflight_kwargs(
                            next_api_kwargs,
                            allow_stream=False,
                            is_github_responses=agent._is_copilot_url(),
                            sanitize_harmony_tokens=agent._is_codex_backend(),
                        )
                    if _use_streaming:
                        return agent._interruptible_streaming_api_call(
                            next_api_kwargs, on_first_delta=_stop_spinner
                        )
                    from agent import relay_llm

                    return relay_llm.execute(
                        next_api_kwargs,
                        agent._interruptible_api_call,
                        session_id=str(agent.session_id or ""),
                        name=str(agent.provider or "provider"),
                        model_name=str(agent.model or ""),
                        metadata={
                            "api_mode": agent.api_mode,
                            "api_request_id": api_request_id,
                            "call_role": (
                                "delegated"
                                if getattr(agent, "is_subagent", False)
                                else "fallback"
                                if int(getattr(agent, "_fallback_index", 0) or 0) > 0
                                else "primary"
                            ),
                            "retry_count": retry_count,
                        },
                        defer_logical_completion=True,
                    )

                from hermes_cli.middleware import run_llm_execution_middleware

                _model_request_active = getattr(agent, "_model_request_active", None)
                _redirect_lock = getattr(agent, "_pending_redirect_lock", None)
                if _redirect_lock is not None:
                    with _redirect_lock:
                        if _model_request_active is not None:
                            _model_request_active.set()
                elif _model_request_active is not None:
                    _model_request_active.set()
                _redirect_crossed_response = False
                try:
                    response = run_llm_execution_middleware(
                        api_kwargs,
                        _perform_api_call,
                        original_request=_original_api_kwargs,
                        task_id=effective_task_id,
                        turn_id=turn_id,
                        api_request_id=api_request_id,
                        session_id=agent.session_id or "",
                        platform=agent.platform or "",
                        model=agent.model,
                        provider=agent.provider,
                        base_url=agent.base_url,
                        api_mode=agent.api_mode,
                        api_call_count=api_call_count,
                        middleware_trace=list(_llm_middleware_trace),
                    )
                finally:
                    if _redirect_lock is not None:
                        with _redirect_lock:
                            if _model_request_active is not None:
                                _model_request_active.clear()
                            _redirect_crossed_response = bool(
                                agent._pending_redirect
                            )
                    else:
                        if _model_request_active is not None:
                            _model_request_active.clear()
                        _redirect_crossed_response = agent._has_pending_redirect()
                if _redirect_crossed_response:
                    # Response and redirect can cross threads: discard the now-stale
                    # response and rebuild from the correction rather than lose it.
                    if thinking_spinner:
                        thinking_spinner.stop("")
                        thinking_spinner = None
                    if agent.thinking_callback:
                        agent.thinking_callback("")
                    if agent.clear_interrupt(preserve_redirect=True):
                        _retry.restart_with_redirected_messages = True
                    else:
                        interrupted = True
                    break
                
                api_duration = time.time() - api_start_time
                
                # Stop thinking spinner silently -- the response box or tool
                # execution messages that follow are more informative.
                if thinking_spinner:
                    thinking_spinner.stop("")
                    thinking_spinner = None
                if agent.thinking_callback:
                    agent.thinking_callback("")
                
                if not agent.quiet_mode:
                    agent._vprint(f"{agent.log_prefix}⏱️  API call completed in {api_duration:.2f}s")
                
                if agent.verbose_logging:
                    # Log response with provider info if available
                    resp_model = getattr(response, 'model', 'N/A') if response else 'N/A'
                    logging.debug(f"API Response received - Model: {resp_model}, Usage: {response.usage if hasattr(response, 'usage') else 'N/A'}")
                
                # Validate response shape before proceeding
                response_invalid, error_details = validate_response_shape(agent, response)

                if response_invalid:
                    agent._invoke_api_request_error_hook(
                        task_id=effective_task_id,
                        turn_id=turn_id,
                        api_request_id=api_request_id,
                        api_call_count=api_call_count,
                        api_start_time=api_start_time,
                        api_kwargs=api_kwargs,
                        error_type="InvalidAPIResponse",
                        error_message=", ".join(error_details) or "Invalid API response",
                        status_code=getattr(getattr(response, "error", None), "code", None),
                        retry_count=retry_count,
                        max_retries=max_retries,
                        retryable=True,
                        reason="invalid_response",
                    )
                    # Stop spinner silently — retry status is now buffered
                    # and only surfaced if every retry+fallback exhausts.
                    if thinking_spinner:
                        thinking_spinner.stop("")
                        thinking_spinner = None
                    if agent.thinking_callback:
                        agent.thinking_callback("")
                    
                    # Invalid response — could be rate limiting, provider timeout,
                    # upstream server error, or malformed response.
                    retry_count += 1
                    
                    # Eager fallback: empty/malformed responses often mean rate limiting
                    # — switch now instead of extended backoff.
                    if agent._fallback_index < len(agent._fallback_chain):
                        agent._buffer_status("⚠️ Empty/malformed response — switching to fallback...")
                    if agent._try_activate_fallback():
                        active_system_prompt = _arm_fallback_restart(
                            agent, api_messages, active_system_prompt, _retry)
                        retry_count = 0
                        compression_attempts = 0
                        break

                    error_msg, provider_name, _failure_hint = describe_invalid_response(
                        agent, response, api_duration
                    )

                    agent._buffer_vprint(f"⚠️  Invalid API response (attempt {retry_count}/{max_retries}): {', '.join(error_details)}")
                    agent._buffer_vprint(f"   🏢 Provider: {provider_name}")
                    cleaned_provider_error = agent._clean_error_message(error_msg)
                    agent._buffer_vprint(f"   📝 Provider message: {cleaned_provider_error}")
                    agent._buffer_vprint(f"   ⏱️  {_failure_hint}")
                    
                    if retry_count >= max_retries:
                        # Try fallback before giving up
                        if agent._has_pending_fallback():
                            agent._buffer_status(f"⚠️ Max retries ({max_retries}) for invalid responses — trying fallback...")
                        if agent._try_activate_fallback():
                            active_system_prompt = _arm_fallback_restart(
                                agent, api_messages, active_system_prompt, _retry)
                            retry_count = 0
                            compression_attempts = 0
                            break
                        # Terminal — flush buffered retry trace so user sees what happened.
                        agent._flush_status_buffer()
                        agent._emit_status(f"❌ Max retries ({max_retries}) exceeded for invalid responses. Giving up.")
                        logger.error("%sInvalid API response after %d retries.", agent.log_prefix, max_retries)
                        agent._persist_session(messages, conversation_history)
                        _final_response = f"Invalid API response after {max_retries} retries: {_failure_hint}"
                        return {
                            "final_response": _final_response,
                            "messages": messages,
                            "completed": False,
                            "api_calls": api_call_count,
                            "error": _final_response,
                            "failed": True  # Mark as failure for filtering
                        }
                    
                    # Backoff before retry — jittered exponential: 5s base, 120s cap
                    wait_time = jittered_backoff(retry_count, base_delay=5.0, max_delay=120.0)
                    agent._buffer_vprint(f"⏳ Retrying in {wait_time:.1f}s ({_failure_hint})...")
                    logger.warning("Invalid API response (retry %d/%d): %s | Provider: %s", retry_count, max_retries, ', '.join(error_details), provider_name)
                    
                    # A redirect cancels only the live request; the helper preserves the
                    # pending correction (restart_with_redirected_messages) instead of
                    # destroying it with clear_interrupt().
                    _interrupted = interruptible_backoff_sleep(
                        agent, wait_time, _retry,
                        messages=messages,
                        conversation_history=conversation_history,
                        api_call_count=api_call_count,
                        abort_message="Interrupt detected during retry wait, aborting.",
                        interrupt_text=f"Operation interrupted during retry ({_failure_hint}, attempt {retry_count}/{max_retries}).",
                        activity_label=f"retry backoff ({retry_count}/{max_retries})",
                    )
                    if _interrupted is not None:
                        return _interrupted
                    if _retry.restart_with_redirected_messages:
                        break  # rebuild this iteration from the correction
                    continue  # Retry the API call

                agent._turn_received_provider_response = True

                # Check finish_reason before proceeding
                if agent.api_mode == "codex_responses":
                    status = getattr(response, "status", None)
                    if isinstance(status, str):
                        status = status.strip().lower()
                    incomplete_details = getattr(response, "incomplete_details", None)
                    incomplete_reason = None
                    if isinstance(incomplete_details, dict):
                        incomplete_reason = incomplete_details.get("reason")
                    else:
                        incomplete_reason = getattr(incomplete_details, "reason", None)
                    if incomplete_reason is not None:
                        incomplete_reason = str(incomplete_reason).strip().lower()
                    if status == "incomplete" and incomplete_reason in {"max_output_tokens", "length"}:
                        # Responses API max-output exhaustion is a normal Codex
                        # incomplete turn: use the Codex continuation path, not the
                        # length rollback.
                        finish_reason = "incomplete"
                    elif status == "incomplete" and incomplete_reason == "content_filter":
                        finish_reason = "content_filter"
                    else:
                        finish_reason = "stop"
                elif agent.api_mode == "anthropic_messages":
                    _tfr = agent._get_transport()
                    finish_reason = _tfr.map_finish_reason(response.stop_reason)
                elif agent.api_mode == "bedrock_converse":
                    # Bedrock response already normalized at dispatch — use transport
                    _bt_fr = agent._get_transport()
                    _bedrock_result = _bt_fr.normalize_response(response)
                    finish_reason = _bedrock_result.finish_reason
                else:
                    _cc_fr = agent._get_transport()
                    _finish_result = _cc_fr.normalize_response(response)
                    finish_reason = _finish_result.finish_reason
                    assistant_message = _finish_result
                    if agent._should_treat_stop_as_truncated(
                        finish_reason,
                        assistant_message,
                        messages,
                    ):
                        agent._vprint(
                            f"{agent.log_prefix}⚠️  Treating suspicious Ollama/GLM stop response as truncated",
                            force=True,
                        )
                        finish_reason = "length"

                # ── Content-policy refusal (HTTP 200) ──────────────────
                # Refusal finish reasons (``content_filter``, ``guardrail_intervened``)
                # are deterministic: one fallback try, else return the refusal.
                if finish_reason == "content_filter":
                    _rv = handle_content_policy_refusal(
                        agent,
                        response,
                        _retry,
                        thinking_spinner=thinking_spinner,
                        messages=messages,
                        api_messages=api_messages,
                        api_kwargs=api_kwargs,
                        active_system_prompt=active_system_prompt,
                        conversation_history=conversation_history,
                        api_call_count=api_call_count,
                        effective_task_id=effective_task_id,
                        turn_id=turn_id,
                        api_request_id=api_request_id,
                        api_start_time=api_start_time,
                        retry_count=retry_count,
                        max_retries=max_retries,
                    )
                    thinking_spinner = None
                    active_system_prompt = _rv.active_system_prompt
                    if _rv.action == "return":
                        return _rv.result
                    retry_count = 0
                    compression_attempts = 0
                    break

                if finish_reason == "length":
                    _tv = recover_from_truncation(
                        agent,
                        response,
                        finish_reason,
                        _retry,
                        messages=messages,
                        conversation_history=conversation_history,
                        api_kwargs=api_kwargs,
                        api_call_count=api_call_count,
                        effective_task_id=effective_task_id,
                        current_turn_user_idx=current_turn_user_idx,
                        length_continue_retries=length_continue_retries,
                        truncated_response_parts=truncated_response_parts,
                        truncated_tool_call_retries=truncated_tool_call_retries,
                        retry_count=retry_count,
                        compression_attempts=compression_attempts,
                    )
                    messages = _tv.messages
                    length_continue_retries = _tv.length_continue_retries
                    truncated_response_parts = _tv.truncated_response_parts
                    truncated_tool_call_retries = _tv.truncated_tool_call_retries
                    retry_count = _tv.retry_count
                    compression_attempts = _tv.compression_attempts
                    if _tv.action == "return":
                        return _tv.result
                    if _tv.action == "break":
                        break
                    if _tv.action == "continue":
                        continue
                
                # Fold provider usage into compressor / anchors / session counters / state.db
                # (agent/turn_usage.py). A rearmed budget also clears the preflight-block latch.
                _usage_outcome = record_response_usage(
                    agent,
                    response,
                    messages=messages,
                    api_call_count=api_call_count,
                    api_duration=api_duration,
                    compression_attempts=compression_attempts,
                    max_compression_attempts=max_compression_attempts,
                )
                compression_attempts = _usage_outcome.compression_attempts
                if _usage_outcome.rearmed:
                    _preflight_compression_blocked = False
                    _last_preflight_pressure = None
                
                _retry.has_retried_429 = False  # Reset on success
                # Don't clear the retry buffer: bytes back != usable content; it is
                # cleared once genuine content lands. Clearing Nous rate-limit state
                # proves the limit reset so other sessions may resume.
                if agent.provider == "nous":
                    try:
                        from agent.nous_rate_guard import clear_nous_rate_limit
                        clear_nous_rate_limit()
                    except Exception:
                        pass
                from agent import relay_llm

                relay_llm.complete_logical_call(
                    api_request_id,
                    outcome="success",
                )
                agent._touch_activity(f"API call #{api_call_count} completed")
                break  # Success, exit retry loop

            except InterruptedError:
                if thinking_spinner:
                    thinking_spinner.stop("")
                    thinking_spinner = None
                if agent.thinking_callback:
                    agent.thinking_callback("")
                if agent._has_pending_redirect():
                    # redirect() cancelled only this request: keep the correction
                    # queued, clear the cancellation bit, let the outer loop rebuild.
                    # Never materialize incomplete signed/encrypted reasoning items.
                    if agent.clear_interrupt(preserve_redirect=True):
                        _retry.restart_with_redirected_messages = True
                        break
                api_elapsed = time.time() - api_start_time
                agent._vprint(f"{agent.log_prefix}⚡ Interrupted during API call.", force=True)
                interrupted = True
                # Keep assistant text already streamed before the stop, else the next
                # turn has no record of the half-finished reply.
                _partial = agent._strip_think_blocks(
                    getattr(agent, "_current_streamed_assistant_text", "") or ""
                ).strip()
                if _partial:
                    append_message(messages, {"role": "assistant", "content": _partial})
                    final_response = _partial
                else:
                    final_response = f"{INTERRUPT_WAITING_FOR_MODEL_PREFIX}{api_elapsed:.1f}s elapsed)."
                agent._persist_session(messages, conversation_history)
                break

            except Exception as api_error:
                # Stop spinner silently — retry status is buffered and
                # only flushed when every retry+fallback is exhausted.
                if thinking_spinner:
                    thinking_spinner.stop("")
                    thinking_spinner = None
                if agent.thinking_callback:
                    agent.thinking_callback("")

                # Pre-classification recovery (encoding sanitization, image rejection,
                # Bedrock SDK streaming fallback) — see agent/turn_recovery.py.
                _recovered, active_system_prompt = recover_before_classification(
                    agent,
                    api_error,
                    messages=messages,
                    api_messages=api_messages,
                    api_kwargs=api_kwargs,
                    active_system_prompt=active_system_prompt,
                )
                if _recovered:
                    continue

                status_code = getattr(api_error, "status_code", None)
                error_context = agent._extract_api_error_context(api_error)

                # ── Interpreter finalization: abandon immediately ──
                # Process is exiting mid-flight: retries/rotation/fallbacks are futile
                # and the retry trace spams the shell. One log line; shared predicate.
                from tools.interpreter_shutdown import interpreter_shutting_down

                if interpreter_shutting_down(api_error):
                    logger.warning(
                        "%sInterpreter is shutting down — abandoning turn "
                        "during API call #%d (%s)",
                        agent.log_prefix, api_call_count, api_error,
                    )
                    _shutdown_summary = (
                        "Turn abandoned: the process was shutting down "
                        "before the model call could complete."
                    )
                    return {
                        "final_response": _shutdown_summary,
                        "messages": messages,
                        "api_calls": api_call_count,
                        "completed": False,
                        "failed": True,
                        "error": _shutdown_summary,
                        "failure_reason": "interpreter_shutdown",
                        "failure_retryable": False,
                    }

                # ── Classify the error for structured recovery decisions ──
                _compressor = getattr(agent, "context_compressor", None)
                _ctx_len = getattr(_compressor, "context_length", 200000) if _compressor else 200000
                classified = classify_api_error(
                    api_error,
                    provider=getattr(agent, "provider", "") or "",
                    model=getattr(agent, "model", "") or "",
                    approx_tokens=approx_tokens,
                    context_length=_ctx_len,
                    num_messages=len(api_messages) if api_messages else 0,
                )
                logger.debug(
                    "Error classified: reason=%s status=%s retryable=%s compress=%s rotate=%s fallback=%s",
                    classified.reason.value, classified.status_code,
                    classified.retryable, classified.should_compress,
                    classified.should_rotate_credential, classified.should_fallback,
                )
                agent._invoke_api_request_error_hook(
                    task_id=effective_task_id,
                    turn_id=turn_id,
                    api_request_id=api_request_id,
                    api_call_count=api_call_count,
                    api_start_time=api_start_time,
                    api_kwargs=api_kwargs,
                    error_type=type(api_error).__name__,
                    error_message=str(api_error),
                    status_code=status_code,
                    retry_count=retry_count,
                    max_retries=max_retries,
                    retryable=classified.retryable,
                    reason=classified.reason.value,
                )

                # One-shot post-classification recovery chain (entitlement refresh, credential
                # pool, image/multimodal strips, per-provider 401 refresh, format-recovery
                # strips) — see agent/turn_recovery.py.
                _recovered, recovered_with_pool = recover_after_classification(
                    agent,
                    api_error,
                    classified,
                    _retry,
                    status_code=status_code,
                    error_context=error_context,
                    messages=messages,
                    api_messages=api_messages,
                )
                if _recovered:
                    continue

                retry_count += 1
                elapsed_time = time.time() - api_start_time
                agent._touch_activity(
                    f"API error recovery (attempt {retry_count}/{max_retries})"
                )
                
                error_type, error_msg, _provider, _base, _model = log_api_error_attempt(
                    agent,
                    api_error,
                    retry_count=retry_count,
                    max_retries=max_retries,
                    status_code=status_code,
                    elapsed_time=elapsed_time,
                    api_messages=api_messages,
                    approx_tokens=approx_tokens,
                )

                # Check for interrupt before deciding to retry
                if agent._interrupt_requested:
                    # Preserve a pending redirect: the user is steering, not stopping
                    # — rebuild the turn from the correction instead of aborting.
                    if agent.clear_interrupt(preserve_redirect=True):
                        _retry.restart_with_redirected_messages = True
                        break
                    agent._vprint(f"{agent.log_prefix}⚡ Interrupt detected during error handling, aborting retries.", force=True)
                    _interrupt_text = f"Operation interrupted: handling API error ({error_type}: {agent._clean_error_message(str(api_error))})."
                    close_interrupted_tool_sequence(messages, _interrupt_text)
                    agent._persist_session(messages, conversation_history)
                    agent.clear_interrupt()
                    return {
                        "final_response": _interrupt_text,
                        "messages": messages,
                        "api_calls": api_call_count,
                        "completed": False,
                        "interrupted": True,
                    }
                
                _ce = route_classified_error(
                    agent,
                    api_error,
                    classified,
                    _retry,
                    error_msg=error_msg,
                    error_context=error_context,
                    recovered_with_pool=recovered_with_pool,
                    base_url=_base,
                    model=_model,
                    messages=messages,
                    api_messages=api_messages,
                    system_message=system_message,
                    active_system_prompt=active_system_prompt,
                    conversation_history=conversation_history,
                    retry_count=retry_count,
                    max_retries=max_retries,
                    compression_attempts=compression_attempts,
                    max_compression_attempts=max_compression_attempts,
                    api_call_count=api_call_count,
                    effective_task_id=effective_task_id,
                )
                status_code = _ce.status_code
                messages = _ce.messages
                active_system_prompt = _ce.active_system_prompt
                conversation_history = _ce.conversation_history
                retry_count = _ce.retry_count
                max_retries = _ce.max_retries
                compression_attempts = _ce.compression_attempts
                is_rate_limited = _ce.is_rate_limited
                _wrapped_output_cap_budget = _ce.wrapped_output_cap_budget
                _is_zai_coding_overload = _ce.is_zai_coding_overload
                if _ce.provider_overflow_recovery_pending:
                    _provider_overflow_recovery_pending = True
                if _ce.action == "return":
                    return _ce.result
                if _ce.action == "break":
                    break
                if _ce.action == "continue":
                    continue

                _ov = recover_from_overflow(
                    agent,
                    api_error,
                    classified,
                    _retry,
                    status_code=status_code,
                    error_msg=error_msg,
                    wrapped_output_cap_budget=_wrapped_output_cap_budget,
                    messages=messages,
                    api_messages=api_messages,
                    system_message=system_message,
                    active_system_prompt=active_system_prompt,
                    conversation_history=conversation_history,
                    approx_tokens=approx_tokens,
                    compression_attempts=compression_attempts,
                    max_compression_attempts=max_compression_attempts,
                    api_call_count=api_call_count,
                    effective_task_id=effective_task_id,
                )
                messages = _ov.messages
                active_system_prompt = _ov.active_system_prompt
                conversation_history = _ov.conversation_history
                approx_tokens = _ov.approx_tokens
                compression_attempts = _ov.compression_attempts
                is_context_length_error = _ov.is_context_length_error
                if _ov.provider_overflow_recovery_pending:
                    _provider_overflow_recovery_pending = True
                if _ov.action == "return":
                    return _ov.result
                if _ov.action == "break":
                    break
                if _ov.action == "continue":
                    continue

                # Non-retryable: ValueError/TypeError are local bugs, except
                # UnicodeEncodeError (surrogate path above) and json.JSONDecodeError, a
                # transient provider/network failure that must be retried (#14782).
                is_local_validation_error = (
                    isinstance(api_error, (ValueError, TypeError))
                    and not isinstance(
                        api_error, (UnicodeEncodeError, json.JSONDecodeError)
                    )
                    # ssl.SSLError inherits from OSError *and* ValueError, so the
                    # ValueError check would misclassify a TLS failure as a local bug;
                    # keep it retryable.
                    and not isinstance(api_error, ssl.SSLError)
                    # "NoneType is not iterable" TypeErrors are upstream shape
                    # mismatches (e.g. Codex response.completed.output=null), reachable
                    # via shims/mocks — retryable so the fallback path runs.
                    and not (
                        isinstance(api_error, TypeError)
                        and "nonetype" in str(api_error).lower()
                        and "not iterable" in str(api_error).lower()
                    )
                )
                # ``FailoverReason.billing`` (402) is deliberately NOT excluded: pool
                # rotation and eager fallback already gave up, so retrying only burns
                # paid requests on a depleted balance. Mirrors 401/403. (#31273)
                is_client_error = (
                    is_local_validation_error
                    or (
                        not classified.retryable
                        and not classified.should_compress
                        and classified.reason not in {
                            FailoverReason.rate_limit,
                            FailoverReason.overloaded,
                            FailoverReason.context_overflow,
                            FailoverReason.payload_too_large,
                            FailoverReason.long_context_tier,
                            FailoverReason.thinking_signature,
                        }
                    )
                ) and not is_context_length_error

                if is_client_error:
                    # Copilot self-heal BEFORE fallback: a stale credential yields a 400
                    # ``model_not_available_for_integrator`` / ``model_not_supported``,
                    # not a 401. Fresh token + client rebuild, one retry, SAME provider.
                    if (
                        _is_copilot_provider(agent)
                        and not _retry.copilot_stale_cred_retry_attempted
                        and _is_stale_copilot_credential_error(
                            status_code, str(getattr(api_error, "message", "") or api_error)
                        )
                    ):
                        _retry.copilot_stale_cred_retry_attempted = True
                        if agent._try_recover_stale_copilot_credential():
                            agent._buffer_vprint(
                                "🔐 Copilot credential re-exchanged after "
                                "model_not_available 400. Retrying request..."
                            )
                            retry_count = 0
                            continue
                    # Try fallback before aborting; announce it only when a fallback
                    # chain exists, else "trying fallback..." lies before a silent abort
                    # (#35314).
                    if agent._has_pending_fallback():
                        if classified.reason == FailoverReason.content_policy_blocked:
                            agent._buffer_status("⚠️ Provider safety filter blocked this request — trying fallback...")
                        elif classified.reason == FailoverReason.ssl_cert_verification:
                            agent._buffer_status("⚠️ TLS certificate verification failed — trying fallback...")
                        else:
                            agent._buffer_status(f"⚠️ Non-retryable error (HTTP {status_code}) — trying fallback...")
                    if agent._try_activate_fallback():
                        active_system_prompt = _arm_fallback_restart(
                            agent, api_messages, active_system_prompt, _retry)
                        retry_count = 0
                        compression_attempts = 0
                        break
                    return nonretryable_client_error_result(
                        agent,
                        api_error,
                        classified,
                        status_code=status_code,
                        api_kwargs=api_kwargs,
                        api_messages=api_messages,
                        messages=messages,
                        conversation_history=conversation_history,
                        api_call_count=api_call_count,
                        approx_tokens=approx_tokens,
                        provider=_provider,
                        base_url=_base,
                        model=_model,
                    )

                if retry_count >= max_retries:
                    # Before fallback, rebuild the primary client once for transient
                    # transport errors (stale pool, TCP reset). Once per API call block.
                    if not _retry.primary_recovery_attempted and agent._try_recover_primary_transport(
                        api_error, retry_count=retry_count, max_retries=max_retries,
                    ):
                        _retry.primary_recovery_attempted = True
                        retry_count = 0
                        # Transport recovery starts a fresh attempt cycle: re-open
                        # fallback state so a follow-on 429 can still activate
                        # fallback_providers.
                        _retry.has_retried_429 = False
                        agent._fallback_index = 0
                        agent._fallback_activated = False
                        continue
                    # Try fallback before giving up entirely
                    if agent._has_pending_fallback():
                        agent._buffer_status(f"⚠️ Max retries ({max_retries}) exhausted — trying fallback...")
                    if agent._try_activate_fallback():
                        active_system_prompt = _arm_fallback_restart(
                            agent, api_messages, active_system_prompt, _retry)
                        retry_count = 0
                        compression_attempts = 0
                        break
                    return max_retries_exhausted_result(
                        agent,
                        api_error,
                        classified,
                        max_retries=max_retries,
                        is_rate_limited=is_rate_limited,
                        error_msg=error_msg,
                        api_kwargs=api_kwargs,
                        api_messages=api_messages,
                        messages=messages,
                        conversation_history=conversation_history,
                        api_call_count=api_call_count,
                        approx_tokens=approx_tokens,
                        provider=_provider,
                        base_url=_base,
                        model=_model,
                    )

                wait_time = compute_error_backoff(
                    agent,
                    api_error,
                    retry_count=retry_count,
                    max_retries=max_retries,
                    is_rate_limited=is_rate_limited,
                    is_zai_coding_overload=_is_zai_coding_overload,
                    base_url=_base,
                    model=_model,
                )
                # Same preserve-redirect rule as the invalid-response wait: a steering
                # correction must survive backoff, not die as "Operation interrupted".
                _interrupted = interruptible_backoff_sleep(
                    agent, wait_time, _retry,
                    messages=messages,
                    conversation_history=conversation_history,
                    api_call_count=api_call_count,
                    abort_message="Interrupt detected during retry wait, aborting.",
                    interrupt_text=f"Operation interrupted: retrying API call after error (retry {retry_count}/{max_retries}).",
                    activity_label=f"error retry backoff ({retry_count}/{max_retries})",
                )
                if _interrupted is not None:
                    return _interrupted
                if _retry.restart_with_redirected_messages:
                    # Leave the retry loop — the check below rebuilds this iteration
                    # from the correction instead of re-firing the stale request.
                    break
        
        if _retry.restart_with_redirected_messages:
            # Cancelled request produced no valid assistant item: reuse the same logical
            # iteration after the outer loop appends partial context + correction.
            api_call_count -= 1
            agent.iteration_budget.refund()
            _retry.restart_with_redirected_messages = False
            continue

        # If the API call was interrupted, skip response processing
        if interrupted:
            _turn_exit_reason = "interrupted_during_api_call"
            break

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
                break
            # In-loop compression rebuilt `messages`; re-anchor the current-turn index
            # like the prologue, AFTER the handoff guard (it may re-append this turn's
            # ask). A stale anchor injects prefetch into a historical row.
            current_turn_user_idx = reanchor_current_turn_user_idx(
                messages, user_message
            )
            agent._persist_user_message_idx = current_turn_user_idx
            continue

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
            continue

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
            continue

        # All retries may exhaust with `response` still None; break out cleanly.
        if response is None:
            _turn_exit_reason = "all_retries_exhausted_no_response"
            print(f"{agent.log_prefix}❌ All API retries exhausted with no successful response.")
            agent._persist_session(messages, conversation_history)
            break

        try:
            _transport = agent._get_transport()
            _normalize_kwargs = {}
            if agent.api_mode == "anthropic_messages":
                _normalize_kwargs["strip_tool_prefix"] = agent._is_anthropic_oauth
            normalized = _transport.normalize_response(response, **_normalize_kwargs)
            assistant_message = normalized
            finish_reason = normalized.finish_reason
            
            # Some OpenAI-compatible servers (llama-server) return content as dict/list,
            # which crashes downstream .strip(); normalize to str.
            if assistant_message.content is not None and not isinstance(assistant_message.content, str):
                raw = assistant_message.content
                if isinstance(raw, dict):
                    assistant_message.content = raw.get("text", "") or raw.get("content", "") or json.dumps(raw)
                elif isinstance(raw, list):
                    # Multimodal content list — extract text parts
                    parts = []
                    for part in raw:
                        if isinstance(part, str):
                            parts.append(part)
                        elif isinstance(part, dict) and part.get("type") == "text":
                            parts.append(part.get("text", ""))
                        elif isinstance(part, dict) and "text" in part:
                            parts.append(str(part["text"]))
                    assistant_message.content = "\n".join(parts)
                else:
                    assistant_message.content = str(raw)

            # ── Agent-as-provider projection ──────────────────────────────
            # Splice the provider-agent's own tool work in as call/result rows before
            # this turn's assistant message; no-op for ordinary providers.
            splice_provider_projection(agent, response, messages)

            try:
                from hermes_cli.lifecycle import (
                    has_hook,
                    invoke_hook as _invoke_hook,
                )
                if has_hook("post_api_request"):
                    _assistant_tool_calls = (
                        getattr(assistant_message, "tool_calls", None) or []
                    )
                    _assistant_text = assistant_message.content or ""
                    _api_ended_at = api_start_time + api_duration
                    _invoke_hook(
                        "post_api_request",
                        task_id=effective_task_id,
                        turn_id=turn_id,
                        api_request_id=api_request_id,
                        session_id=agent.session_id or "",
                        platform=agent.platform or "",
                        model=agent.model,
                        provider=agent.provider,
                        base_url=agent.base_url,
                        api_mode=agent.api_mode,
                        api_call_count=api_call_count,
                        api_duration=api_duration,
                        started_at=api_start_time,
                        ended_at=_api_ended_at,
                        # First stream chunk time (epoch s) from
                        # interruptible_streaming_api_call; None if not streamed / no
                        # chunk. TTFB = first_chunk_at - started_at.
                        first_chunk_at=getattr(
                            agent, "_last_api_first_chunk_at", None
                        ),
                        finish_reason=finish_reason,
                        message_count=len(api_messages),
                        response_model=getattr(response, "model", None),
                        response=agent._api_response_payload_for_hook(
                            response,
                            assistant_message,
                            finish_reason=finish_reason,
                        ),
                        usage=agent._usage_summary_for_api_request_hook(response),
                        assistant_message=assistant_message,
                        assistant_content_chars=len(_assistant_text),
                        assistant_tool_call_count=len(_assistant_tool_calls),
                        moa_references=_moa_reference_metrics_for_hook(agent),
                    )
            except Exception:
                pass

            # Handle assistant response
            if assistant_message.content and not agent.quiet_mode:
                if agent.verbose_logging:
                    agent._vprint(f"{agent.log_prefix}🤖 Assistant: {assistant_message.content}")
                else:
                    agent._vprint(f"{agent.log_prefix}🤖 Assistant: {assistant_message.content[:100]}{'...' if len(assistant_message.content) > 100 else ''}")

            # Notify progress callback of model's thinking (used by subagent
            # delegation to relay the child's reasoning to the parent display).
            if (assistant_message.content and agent.tool_progress_callback):
                _think_text = assistant_message.content.strip()
                # Strip reasoning XML tags that shouldn't leak to parent display
                _think_text = re.sub(
                    r'</?(?:REASONING_SCRATCHPAD|think|reasoning)>', '', _think_text
                ).strip()
                # For subagents: relay first line to parent display (existing behaviour).
                # For all agents with a structured callback: emit reasoning.available event.
                first_line = _think_text.split('\n')[0][:80] if _think_text else ""
                if first_line and getattr(agent, '_delegate_depth', 0) > 0:
                    try:
                        agent.tool_progress_callback("_thinking", first_line)
                    except Exception:
                        pass
                elif _think_text:
                    try:
                        agent.tool_progress_callback("reasoning.available", "_thinking", _think_text[:500], None)
                    except Exception:
                        pass
            
            # Check for incomplete <REASONING_SCRATCHPAD> (opened but never closed)
            # This means the model ran out of output tokens mid-reasoning — retry up to 2 times
            if has_incomplete_scratchpad(assistant_message.content or ""):
                agent._incomplete_scratchpad_retries += 1
                
                agent._buffer_vprint("⚠️  Incomplete <REASONING_SCRATCHPAD> detected (opened but never closed)")
                
                if agent._incomplete_scratchpad_retries <= 2:
                    agent._buffer_vprint(f"🔄 Retrying API call ({agent._incomplete_scratchpad_retries}/2)...")
                    # Don't add the broken message, just retry
                    continue
                else:
                    # Max retries - discard this turn and save as partial
                    agent._flush_status_buffer()
                    agent._vprint(f"{agent.log_prefix}❌ Max retries (2) for incomplete scratchpad. Saving as partial.", force=True)
                    agent._incomplete_scratchpad_retries = 0
                    
                    rolled_back_messages = agent._get_messages_up_to_last_assistant(messages)
                    agent._cleanup_task_resources(effective_task_id)
                    agent._persist_session(messages, conversation_history)
                    
                    return {
                        "final_response": "Incomplete REASONING_SCRATCHPAD after 2 retries",
                        "messages": rolled_back_messages,
                        "api_calls": api_call_count,
                        "completed": False,
                        "partial": True,
                        "error": "Incomplete REASONING_SCRATCHPAD after 2 retries"
                    }
            
            # Reset incomplete scratchpad counter on clean response
            agent._incomplete_scratchpad_retries = 0

            if agent.api_mode == "codex_responses" and finish_reason == "incomplete":
                _codex_result = continue_codex_incomplete(
                    agent,
                    assistant_message,
                    finish_reason,
                    messages=messages,
                    conversation_history=conversation_history,
                    api_call_count=api_call_count,
                )
                if _codex_result is not None:
                    return _codex_result
                continue
            elif hasattr(agent, "_codex_incomplete_retries"):
                agent._codex_incomplete_retries = 0
            
            # Check for tool calls
            if assistant_message.tool_calls:
                if not agent.quiet_mode:
                    agent._vprint(f"{agent.log_prefix}🔧 Processing {len(assistant_message.tool_calls)} tool call(s)...")
                
                if agent.verbose_logging:
                    for tc in assistant_message.tool_calls:
                        raw_args = tc.function.arguments
                        args_preview = raw_args[:200] if isinstance(raw_args, str) else repr(raw_args)[:200]
                        logging.debug("Tool call: %s with args: %s...", tc.function.name, args_preview)
                
                _tvv = validate_tool_calls(
                    agent,
                    assistant_message,
                    finish_reason,
                    messages=messages,
                    conversation_history=conversation_history,
                    api_call_count=api_call_count,
                    effective_task_id=effective_task_id,
                )
                _mixed_invalid_batch = _tvv.mixed_invalid_batch
                if _tvv.action == "return":
                    return _tvv.result
                if _tvv.action == "continue":
                    continue

                # ── Post-call guardrails ──────────────────────────
                assistant_message.tool_calls = agent._cap_delegate_task_calls(
                    assistant_message.tool_calls
                )
                assistant_message.tool_calls = agent._deduplicate_tool_calls(
                    assistant_message.tool_calls
                )

                # Collect invalid calls so the assistant message keeps EVERY emitted
                # call (each tool_call needs a matching result) while only valid ones
                # dispatch.
                _invalid_batch_calls = []
                if _mixed_invalid_batch:
                    _invalid_batch_calls = [
                        tc for tc in assistant_message.tool_calls
                        if tc.function.name not in agent.valid_tool_names
                    ]

                assistant_msg = agent._build_assistant_message(assistant_message, finish_reason)

                turn_content = assistant_message.content or ""

                # A bare bracketed token (e.g. ``[memory]``) beside a function call is
                # protocol scaffolding; persisting it lets the post-tool fallback replay
                # it forever (#78148).
                if (
                    assistant_message.tool_calls
                    and _STALE_MARKER_RE.fullmatch(turn_content.strip())
                ):
                    logger.warning(
                        "Discarding bare tool-call marker from assistant content: %s",
                        turn_content,
                    )
                    turn_content = ""
                    assistant_msg["content"] = ""

                # Classify tools regardless of visible content: a substantive tool-only
                # turn must invalidate any older housekeeping fallback.
                _HOUSEKEEPING_TOOLS = frozenset({
                    "memory", "todo_list", "skill_manage", "session_search",
                })
                _all_housekeeping = all(
                    tc.function.name in _HOUSEKEEPING_TOOLS
                    for tc in assistant_message.tool_calls
                )

                # Substantive tools clear any older fallback so a two-turn-old
                # housekeeping narration isn't attributed to the preceding tool turn.
                if assistant_message.tool_calls and not _all_housekeeping:
                    agent._last_content_with_tools = None
                    agent._last_content_tools_all_housekeeping = False
                    # Also clear the mute flag a prior housekeeping turn may have set,
                    # else _vprint suppresses this turn's tool progress until the
                    # no-tool-call branch clears it.
                    agent._mute_post_response = False

                # Content + tool_calls in one turn: keep the content as a fallback final
                # response in case the follow-up turn after tools is empty.
                if turn_content and agent._has_content_after_think_block(turn_content):
                    agent._last_content_with_tools = turn_content
                    # Mute only when EVERY tool call is post-response housekeeping
                    # (memory, todo, skill_manage); substantive tools keep output on.
                    agent._last_content_tools_all_housekeeping = _all_housekeeping
                    if _all_housekeeping and agent._has_stream_consumers():
                        agent._mute_post_response = True
                    elif agent._should_emit_quiet_tool_messages():
                        clean = agent._strip_think_blocks(turn_content).strip()
                        if clean:
                            agent._vprint(f"  ┊ 💬 {clean}")
                
                # Pop thinking-only prefill message(s) before appending
                # (tool-call path — same rationale as the final-response path).
                _had_prefill = False
                while (
                    messages
                    and isinstance(messages[-1], dict)
                    and messages[-1].get("_thinking_prefill")
                ):
                    messages.pop()
                    _had_prefill = True

                # Tool calls after a prefill recovery reset the prefill counter, so
                # each tool-call success is a fresh start, not a cumulative burn.
                if _had_prefill:
                    agent._thinking_prefill_retries = 0
                    agent._empty_content_retries = 0
                # Re-arm the post-tool nudge so it can fire on a LATER tool round.
                agent._post_tool_empty_retried = False
                # A landed tool call recovers any dropped-tool-call stall; refresh that
                # budget so it guards each stall independently, not the whole run.
                agent._dropped_toolcall_retries = 0

                previous_msg = messages[-1] if messages else None
                current_interim_visible = agent._interim_assistant_visible_text(assistant_msg)
                previous_interim_visible = (
                    agent._interim_assistant_visible_text(previous_msg)
                    if isinstance(previous_msg, dict)
                    else ""
                )
                duplicate_previous_interim = (
                    bool(current_interim_visible)
                    and isinstance(previous_msg, dict)
                    and previous_msg.get("role") == "assistant"
                    and previous_msg.get("finish_reason") == "incomplete"
                    and previous_interim_visible == current_interim_visible
                )
                append_message(messages, assistant_msg)

                # Mixed batch: error-result invalid calls and drop them from execution.
                # The assistant message keeps all calls so tool_call/result pairs hold.
                if _invalid_batch_calls:
                    for tc in _invalid_batch_calls:
                        append_message(messages, {
                            "role": "tool",
                            "name": tc.function.name,
                            "tool_call_id": coalesce_tool_call_id(tc),
                            "content": _invalid_tool_name_error_content(
                                tc.function.name, agent.valid_tool_names
                            ),
                        })
                    assistant_message.tool_calls = [
                        tc for tc in assistant_message.tool_calls
                        if tc.function.name in agent.valid_tool_names
                    ]

                _tool_turn_persisted = None
                try:
                    # Persist the tool-call turn before any tool side effects so resume
                    # sees the executed block if a destructive tool restarts Hermes.
                    _tool_turn_persisted = agent._flush_messages_to_session_db(
                        messages, conversation_history
                    )
                except Exception as exc:
                    _tool_turn_persisted = False
                    from hermes_state import classify_persistence_error
                    agent._last_persistence_error_cause = (
                        classify_persistence_error(exc)
                    )
                    logger.warning(
                        "Incremental tool-call persistence failed before execution "
                        "(session=%s): %s",
                        agent.session_id or "none",
                        exc,
                    )

                if _tool_turn_persisted is False:
                    # Canonical append failed: never project the row or run tools from
                    # process-only state; break rather than retry the unpersisted turn.
                    # If the flush recorded no cause, the cause is genuinely unknown.
                    if getattr(agent, "_last_persistence_error_cause", None) is None:
                        agent._last_persistence_error_cause = "unknown"
                    _turn_exit_reason = "session_persistence_failed"
                    final_response = ""
                    failed = True
                    break

                # A UI must never observe an assistant/tool-call row that is only an
                # in-memory projection: emit interim commentary after the DB append.
                if not duplicate_previous_interim:
                    agent._emit_interim_assistant_message(assistant_msg)

                # Flush open streaming boxes before tools so early content doesn't wrap
                # tool feed lines. Display callback only — TTS (_stream_callback) must
                # NOT receive None (its end-of-stream marker).
                if agent.stream_delta_callback:
                    try:
                        agent.stream_delta_callback(None)
                    except Exception:
                        pass

                agent._execute_tool_calls(assistant_message, messages, effective_task_id, api_call_count)

                if getattr(agent, "_incremental_persistence_failed", False):
                    # Tool result could not be made canonical: never send the in-memory
                    # result to the model or project later events from this turn.
                    _turn_exit_reason = "session_persistence_failed"
                    final_response = ""
                    failed = True
                    break

                if agent._tool_guardrail_halt_decision is not None:
                    decision = agent._tool_guardrail_halt_decision
                    _turn_exit_reason = "guardrail_halt"
                    final_response = agent._toolguard_controlled_halt_response(decision)
                    agent._emit_status(
                        f"⚠️ Tool guardrail halted {decision.tool_name}: {decision.code}"
                    )
                    append_message(messages, {"role": "assistant", "content": final_response})
                    # Emit the halt message so it isn't mistaken for a crash; the stream
                    # callback is still alive, so SSE/TUI clients see the explanation.
                    if final_response:
                        agent._safe_print(f"\n{final_response}\n")
                        if agent.stream_delta_callback:
                            try:
                                agent.stream_delta_callback(final_response)
                                agent.stream_delta_callback(None)
                            except Exception:
                                pass
                    break

                # Reset per-turn retry counters so one truncation can't poison the turn.
                truncated_tool_call_retries = 0

                # Defer the paragraph break: _fire_stream_delta() prepends one "\n\n"
                # when real text arrives, so tool iterations don't stack blank lines.
                agent._stream_needs_break = True

                # Refund the iteration when the ONLY tool was execute_code (programmatic
                # tool calling) — cheap RPC-style calls shouldn't eat the budget.
                _tc_names = {tc.function.name for tc in assistant_message.tool_calls}
                if _tc_names == {"execute_code"}:
                    agent.iteration_budget.refund()
                
                _ptc = compress_after_tool_results(
                    agent,
                    messages=messages,
                    system_message=system_message,
                    user_message=user_message,
                    active_system_prompt=active_system_prompt,
                    conversation_history=conversation_history,
                    compression_attempts=compression_attempts,
                    max_compression_attempts=max_compression_attempts,
                    effective_task_id=effective_task_id,
                    final_response=final_response,
                    turn_exit_reason=_turn_exit_reason,
                )
                messages = _ptc.messages
                active_system_prompt = _ptc.active_system_prompt
                conversation_history = _ptc.conversation_history
                compression_attempts = _ptc.compression_attempts
                final_response = _ptc.final_response
                _turn_exit_reason = _ptc.turn_exit_reason
                if _ptc.end_turn:
                    break
                
                # Save session log incrementally (so progress is visible even if interrupted)
                agent._session_messages = messages
                
                # Touch activity so slow post-tool work plus a slow follow-up API call
                # can't exceed the gateway inactivity timeout (HERMES_AGENT_TIMEOUT).
                agent._touch_activity(f"tool results posted, continuing iteration #{api_call_count}")
                # Continue loop for next response
                continue
            
            else:
                # No tool calls — final response. (Dropped tool-call recovery lives at
                # the finalization chokepoint below so it catches every path.)
                final_response = assistant_message.content or ""
                
                # Unmute: _mute_post_response from a housekeeping tool turn must not
                # silence empty-response warnings on the final response path.
                agent._mute_post_response = False
                
                # Check if response only has think block with no actual content after it
                if not agent._has_content_after_think_block(final_response):
                    _ev = recover_empty_response(
                        agent,
                        assistant_message,
                        response,
                        finish_reason,
                        final_response=final_response,
                        messages=messages,
                        api_messages=api_messages,
                        conversation_history=conversation_history,
                        active_system_prompt=active_system_prompt,
                        api_call_count=api_call_count,
                        turn_exit_reason=_turn_exit_reason,
                        preflight_compression_blocked=_preflight_compression_blocked,
                    )
                    final_response = _ev.final_response
                    _turn_exit_reason = _ev.turn_exit_reason
                    active_system_prompt = _ev.active_system_prompt
                    _preflight_compression_blocked = _ev.preflight_compression_blocked
                    if _ev.action == "return":
                        return _ev.result
                    if _ev.action == "break":
                        break
                    continue
                
                # Reset retry counter/signature on successful content
                agent._empty_content_retries = 0
                agent._thinking_prefill_retries = 0
                # Surface the one-shot fallback switch notice before dropping the retry
                # buffer so a provider/model switch stays visible on success.
                agent._emit_pending_fallback_notice()
                agent._clear_status_buffer()

                from agent.agent_runtime_helpers import (
                    intent_ack_continuation_mode,
                    trailing_continue_intent,
                )

                _ack_mode = intent_ack_continuation_mode(agent)
                # Said-continue-but-stopped guard: no tool calls but the short reply
                # TAILS with an announced next action. Fires mid-task too; reuses the
                # SAME bounded continuation path and counter (max 2 per turn).
                _stall_continue_intent = (
                    bool(getattr(agent, "_stall_guards", True))
                    and agent.valid_tool_names
                    and codex_ack_continuations < 2
                    and trailing_continue_intent(
                        agent._strip_think_blocks(final_response or "")
                    )
                )
                if _stall_continue_intent or (
                    _ack_mode != "off"
                    and agent.valid_tool_names
                    and codex_ack_continuations < 2
                    and agent._looks_like_codex_intermediate_ack(
                        user_message=user_message,
                        assistant_content=final_response,
                        messages=messages,
                        require_workspace=(_ack_mode == "codex_only"),
                    )
                ):
                    if _stall_continue_intent:
                        logger.info(
                            "Stall guard: turn ending on trailing continue-"
                            "intent with no tool calls — re-prompting to act "
                            "(%d/2)", codex_ack_continuations + 1,
                        )
                    codex_ack_continuations += 1
                    interim_msg = agent._build_assistant_message(assistant_message, "incomplete")
                    append_message(messages, interim_msg)
                    agent._emit_interim_assistant_message(interim_msg)

                    continue_msg = {
                        "role": "user",
                        "content": _CODEX_ACK_CONTINUATION_NUDGE,
                    }
                    append_message(messages, continue_msg)
                    agent._session_messages = messages
                    # An acknowledgment is non-final: its text must not suppress
                    # iteration-limit summarization if the continuation exhausts budget.
                    final_response = None
                    continue

                codex_ack_continuations = 0

                if truncated_response_parts:
                    final_response = _join_truncated_parts([*truncated_response_parts, final_response])
                    truncated_response_parts = []
                    length_continue_retries = 0
                    # The continuation recovered, so the fragments stay in the transcript.
                    for _frag in messages:
                        if isinstance(_frag, dict):
                            _frag.pop("_length_continuation_fragment", None)
                            _frag.pop("_length_continuation_nudge", None)
                
                final_response = agent._strip_think_blocks(final_response).strip()
                
                final_msg = agent._build_assistant_message(assistant_message, finish_reason)

                # ── Dropped tool-call recovery (copilot/Claude) ────────
                # finish_reason="tool_calls" with empty tool_calls would end the turn
                # unstarted; re-prompt (max 3 CONSECUTIVE stalls, reset per tool round).
                if (
                    finish_reason == "tool_calls"
                    and not assistant_message.tool_calls
                    and getattr(agent, "_dropped_toolcall_retries", 0) < 3
                ):
                    agent._dropped_toolcall_retries = getattr(agent, "_dropped_toolcall_retries", 0) + 1
                    logger.warning(
                        "finish_reason=tool_calls with empty tool_calls array "
                        "(narration only) — re-prompting to emit the call "
                        "(retry %d/3, model=%s provider=%s)",
                        agent._dropped_toolcall_retries, agent.model, agent.provider,
                    )
                    agent._emit_status(
                        "↻ Model signaled a tool call but sent none — "
                        f"re-prompting ({agent._dropped_toolcall_retries}/3)"
                    )
                    # Both halves of the re-prompt pair are ephemeral scaffolding; flag
                    # them so the flush never persists them and the finalization pop
                    # can strip an unanswered tail pair.
                    final_msg["_dropped_toolcall_nudge"] = True
                    append_message(messages, final_msg)
                    append_message(messages, {
                        "role": "user",
                        "content": _DROPPED_TOOLCALL_NUDGE_CONTENT,
                        "_dropped_toolcall_nudge": True,
                    })
                    agent._session_messages = messages
                    final_response = None
                    continue

                # Genuine turn end (no dropped-tool-call mismatch): clear stall budget.
                agent._dropped_toolcall_retries = 0

                # Pop prefill / empty-retry scaffolding before the final response or
                # verification follow-up; it must not become durable transcript.
                while (
                    messages
                    and isinstance(messages[-1], dict)
                    and (
                        messages[-1].get("_thinking_prefill")
                        or messages[-1].get("_empty_recovery_synthetic")
                        or messages[-1].get("_empty_terminal_sentinel")
                        or messages[-1].get("_dropped_toolcall_nudge")
                    )
                ):
                    messages.pop()

                _sg = apply_stop_gates(
                    agent,
                    final_msg,
                    final_response=final_response,
                    messages=messages,
                    conversation_history=conversation_history,
                    pending_verification_response=_pending_verification_response,
                    pending_verification_response_previewed=_pending_verification_response_previewed,
                )
                _pending_verification_response = _sg.pending_verification_response
                _pending_verification_response_previewed = _sg.pending_verification_response_previewed
                if _sg.continue_turn:
                    final_response = None
                    continue

                append_message(messages, final_msg)
                # Make the answer durable before leaving the loop; _DB_PERSISTED_MARKER
                # keeps _persist_session idempotent. Failure must NOT abort the turn:
                # _persist_session retries the write. (#81641)
                try:
                    agent._flush_messages_to_session_db(messages, conversation_history)
                except Exception:
                    logger.warning(
                        "final text-turn flush failed (session=%s) — reply is "
                        "not yet durable; relying on finalize_turn retry",
                        getattr(agent, "session_id", None) or "none",
                        exc_info=True,
                    )

                _turn_exit_reason = f"text_response(finish_reason={finish_reason})"
                if not agent.quiet_mode:
                    agent._safe_print(f"🎉 Conversation completed after {api_call_count} OpenAI-compatible API call(s)")
                break
            
        except Exception as e:
            # Count every escaped exception before classification so permanent
            # failures terminate even with an unlimited turn budget. (#92450)
            _outer_error_count += 1

            # Phase-aware classification: deterministic local post-processing bugs
            # (traceback via local helpers, never API helpers) aren't retried (#66267).
            # Interpreter shutdown makes every executor op raise: break. (#93217)
            if sys.is_finalizing() or _is_interpreter_shutdown_error(e):
                error_msg = (
                    f"Interpreter is shutting down — cannot continue "
                    f"(API call #{api_call_count}): {e}"
                )
                try:
                    agent._safe_print(f"❌ {error_msg}")
                except (OSError, ValueError):
                    pass
                logger.warning(error_msg)
                # Best-effort persist — the dying executor may raise the same error;
                # don't let it mask the shutdown exit. finalize_turn retries.
                try:
                    agent._persist_session(messages, conversation_history)
                except Exception:
                    pass
                _turn_exit_reason = "interpreter_shutdown"
                final_response = (
                    "Session is shutting down. Your conversation can be "
                    "resumed with: hermes --resume <session-id>"
                )
                # Don't append: a prefill/interim assistant may already be the tail
                # (assistant→assistant). finalize_turn appends only when safe.
                break

            tb_module_names: set[str] = set()
            _tb = e.__traceback__
            while _tb is not None:
                _fname = os.path.splitext(os.path.basename(_tb.tb_frame.f_code.co_filename))[0]
                tb_module_names.add(_fname)
                _tb = _tb.tb_next

            _hit_local = bool(tb_module_names & _LOCAL_PROCESSING_MODULES)
            _hit_api = bool(tb_module_names & _API_CALL_MODULES)

            _is_local_processing_error = _hit_local and not _hit_api

            if _is_local_processing_error:
                error_msg = (
                    f"Error during local message processing after "
                    f"OpenAI-compatible API call #{api_call_count}: {str(e)}"
                )
            else:
                error_msg = f"Error during OpenAI-compatible API call #{api_call_count}: {str(e)}"
            # Honor the _vprint contract: suppress_status_output silences hard
            # failures; quiet_mode -q still shows them. Traceback is logged below.
            if getattr(agent, "suppress_status_output", False):
                logger.error(error_msg)
            else:
                try:
                    print(f"❌ {error_msg}")
                except (OSError, ValueError):
                    logger.error(error_msg)

            # ERROR level with traceback so outer-loop failures land in agent.log
            # AND errors.log and stay reproducible.
            logger.exception("Outer loop error in API call #%d", api_call_count)
            
            # An appended assistant tool_calls message needs a role="tool" result
            # per tool_call_id; fill in error results for unanswered ones.
            for idx in range(len(messages) - 1, -1, -1):
                msg = messages[idx]
                if not isinstance(msg, dict):
                    break
                if msg.get("role") == "tool":
                    continue
                if msg.get("role") == "assistant" and msg.get("tool_calls"):
                    answered_ids = {
                        m["tool_call_id"]
                        for m in messages[idx + 1:]
                        if isinstance(m, dict) and m.get("role") == "tool"
                    }
                    for tc in msg["tool_calls"]:
                        if not tc or not isinstance(tc, dict): continue
                        if tc["id"] not in answered_ids:
                            err_msg = {
                                "role": "tool",
                                "name": _ra().AIAgent._get_tool_call_name_static(tc),
                                "tool_call_id": tc["id"],
                                "content": f"Error executing tool: {error_msg}",
                            }
                            append_message(messages, err_msg)
                break
            
            # Non-tool errors are already printed; a synthetic message would pollute
            # history and risk breaking role alternation.

            # Local errors are deterministic: stop early instead of retrying until the
            # budget is gone; a small per-turn cap prevents infinite spinning (#92450).
            _outer_error_cap = min(_MAX_OUTER_LOOP_ERRORS, max(1, agent.max_iterations))
            if (
                _is_local_processing_error
                or api_call_count >= agent.max_iterations - 1
                or _outer_error_count >= _outer_error_cap
            ):
                if _is_local_processing_error:
                    _turn_exit_reason = f"local_processing_error({error_msg[:80]})"
                    final_response = f"I apologize, but I encountered an error while processing the model response: {error_msg}"
                elif _outer_error_count >= _outer_error_cap:
                    failed = True
                    _turn_exit_reason = f"repeated_outer_errors({error_msg[:80]})"
                    final_response = f"I apologize, but I encountered repeated errors: {error_msg}"
                else:
                    _turn_exit_reason = f"error_near_max_iterations({error_msg[:80]})"
                    final_response = f"I apologize, but I encountered repeated errors: {error_msg}"
                # Don't append the assistant message: a prefill/interim assistant may be
                # the tail. finalize_turn appends only when _tail_role != "assistant".
                break
    
    # Post-loop finalization lives in agent/turn_finalizer.finalize_turn.
    result = finalize_turn(
        agent,
        final_response=final_response,
        api_call_count=api_call_count,
        interrupted=interrupted,
        failed=failed,
        messages=messages,
        conversation_history=conversation_history,
        effective_task_id=effective_task_id,
        turn_id=turn_id,
        user_message=user_message,
        original_user_message=original_user_message,
        _should_review_memory=_should_review_memory,
        _turn_exit_reason=_turn_exit_reason,
        _pending_verification_response=_pending_verification_response,
        _pending_verification_response_previewed=_pending_verification_response_previewed,
    )
    if _compression_timeout_exhausted:
        # Reuse the gateway's context-recovery contract: transcript stays intact while
        # future input can move to a clean session (#98722).
        result["error"] = _COMPRESSION_TIMEOUT_FINAL_RESPONSE
        result["partial"] = True
        result["compression_exhausted"] = True
    return result



__all__ = ["run_conversation"]
