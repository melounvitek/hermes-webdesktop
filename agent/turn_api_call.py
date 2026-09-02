"""The provider call itself for the conversation turn's retry loop: streaming decision,
MoA prepared-request handshake, the LLM execution middleware wrapper, the redirect
``_model_request_active`` bracket and the response-vs-redirect crossing check. Extracted
from ``run_conversation``; nothing here imports ``agent.conversation_loop`` at module level.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional


logger = logging.getLogger("agent.conversation_loop")


@dataclass
class ApiCallVerdict:
    """``action``: ``"fallthrough"`` (``response`` is ready for verification) or ``"break"``
    (a redirect crossed the response — rebuild armed on ``_retry`` or ``interrupted``)."""

    action: str
    response: Any
    thinking_spinner: Any
    interrupted: Any


def perform_api_call(
    agent: Any,
    *,
    api_kwargs: Any,
    _original_api_kwargs: Any,
    _llm_middleware_trace: Any,
    _moa_prepared_request: Any,
    _retry: Any,
    thinking_spinner: Any,
    retry_count: Any,
    api_call_count: Any,
    api_request_id: Any,
    effective_task_id: Any,
    turn_id: Any,
    interrupted: Any,
) -> ApiCallVerdict:
    """Issue the request. Streaming is preferred even without consumers (stale-stream /
    read-timeout health checks) and disabled per provider signal, ACP schemes, MoA without a
    display consumer, or Mock clients in tests."""
    response = None

    def _verdict(action: str, result: Optional[Dict[str, Any]] = None) -> ApiCallVerdict:
        return ApiCallVerdict(
            action=action,
            response=response,
            thinking_spinner=thinking_spinner,
            interrupted=interrupted,
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
        return _verdict("break")
    return _verdict("fallthrough")
