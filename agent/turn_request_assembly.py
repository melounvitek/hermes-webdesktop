"""Per-iteration API request assembly for the conversation turn loop: build ``api_messages``
from the transcript, append MoA context, inject prefills, run the context-engine selection
hook and the send-time sanitizers, canonicalize for bit-perfect cache prefixes, build the
request-local prompt-cache plan LAST (after every transcript mutation), prepare the
persistent-MoA request, then measure request pressure. Extracted from
``run_conversation``; nothing here imports ``agent.conversation_loop`` at module level
(cycle) — loop-internal helpers resolve lazily so ``patch("agent.conversation_loop.X")``
sites keep intercepting.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Dict, Optional

from agent.message_sanitization import _sanitize_messages_surrogates
from agent.model_metadata import anchored_context_tokens
from agent.prompt_caching import build_prompt_cache_plan, effective_cache_ttl
from agent.turn_context import build_api_messages

logger = logging.getLogger("agent.conversation_loop")


@dataclass
class AssembledRequest:
    """Always ``action == "fallthrough"``; the fields are the iteration locals the assembly
    produces (``api_messages``/``tools_for_api`` are the decorated request copies — the
    canonical ``messages``/``agent.tools`` stay undecorated)."""

    action: str
    api_messages: Any
    tools_for_api: Any
    _moa_prepared_request: Any
    pending_moa_prepared_request: Any
    approx_tokens: Any
    request_pressure_tokens: Any
    total_chars: Any


def assemble_api_request(
    agent: Any, *, messages: Any, current_turn_user_idx: Any, _ext_prefetch_cache: Any,
    _plugin_user_context: Any, moa_config: Any, active_system_prompt: Any,
    original_user_message: Any, pending_moa_prepared_request: Any, request_logger: Any,
) -> AssembledRequest:
    """Assemble the request in the original order. ORDER IS LOAD-BEARING: cache breakpoints
    are injected only after whitespace normalization, the orphan sweep, thinking-only drop /
    user merge and surrogate stripping, so the same row's bytes never vary across turns."""
    from agent.conversation_loop import (
        _apply_context_engine_selection, _canonicalize_api_tool_calls, _clone_message_for_send,
        _midturn_request_pressure_tokens, estimate_messages_tokens_rough,
    )

    def _verdict(action: str, result: Optional[Dict[str, Any]] = None) -> AssembledRequest:
        return AssembledRequest(
            action=action,
            api_messages=api_messages,
            tools_for_api=tools_for_api,
            _moa_prepared_request=_moa_prepared_request,
            pending_moa_prepared_request=pending_moa_prepared_request,
            approx_tokens=approx_tokens,
            request_pressure_tokens=request_pressure_tokens,
            total_chars=total_chars,

        )

    api_messages, effective_system = build_api_messages(
        agent, messages, current_turn_user_idx=current_turn_user_idx,
        ext_prefetch_cache=_ext_prefetch_cache, plugin_user_context=_plugin_user_context,
        moa_config=moa_config, active_system_prompt=active_system_prompt,
    )

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
                                *_base, {"type": "text", "text": "\n\n" + _moa_context}
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
        messages[current_turn_user_idx] if 0 <= current_turn_user_idx < len(messages) else None
    )
    api_messages = _apply_context_engine_selection(
        agent, api_messages, messages, _sel_incoming, logger=request_logger
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
        api_messages, drop_codex_reasoning_items=agent.api_mode != "codex_responses"
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
                agent._cache_ttl, provider=agent.provider, model=agent.model
            ),
            native_anthropic=agent._use_native_cache_layout,
            static_system_prefix=(
                _static_system_prefix if isinstance(_static_system_prefix, str) else None
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
    return _verdict("fallthrough")
