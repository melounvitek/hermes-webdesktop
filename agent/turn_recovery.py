"""Recovery-branch handlers for the conversation turn's inner retry loop.

``run_conversation`` wraps every model API call in ``while retry_count < max_retries``.
When the call raises, a long chain of one-shot recovery branches runs before the generic
retry/backoff path: payload sanitization (surrogates / ASCII codec), image rejection,
per-provider 401 credential refresh, format-recovery strips (thinking signatures,
encrypted reasoning, native compaction, llama.cpp grammar), etc. Each handler here owns
one contiguous chain and returns a verdict the loop acts on:

* ``True``  → the request was repaired in place; the loop ``continue``s (re-issues the
  call with the same ``retry_count``, exactly as the inline ``continue`` did).
* ``False`` → nothing applied; the loop falls through to the generic retry path.

One-shot guards live on ``TurnRetryState`` (``agent/turn_retry_state.py``); handlers set
them exactly where the inline code did. Handlers mutate ``agent`` / ``messages`` /
``api_messages`` in place — side effects are the point; only locals moved.

Logger name stays ``agent.conversation_loop`` so caplog pins and log routing are
unchanged. Nothing here imports ``agent.conversation_loop`` at module level (cycle);
the few loop-internal helpers are passed in or imported lazily.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from agent.error_classifier import FailoverReason
from agent.message_sanitization import (
    _looks_like_image_content_rejection,
    _sanitize_messages_non_ascii,
    _sanitize_messages_surrogates,
    _sanitize_structure_non_ascii,
    _sanitize_structure_surrogates,
    _sanitize_tools_non_ascii,
    _strip_images_from_messages,
    _strip_non_ascii,
)
from agent.turn_retry_state import TurnRetryState

logger = logging.getLogger("agent.conversation_loop")


def recover_before_classification(
    agent: Any,
    api_error: Exception,
    *,
    messages: List[Dict[str, Any]],
    api_messages: Any,
    api_kwargs: Any,
    active_system_prompt: Any,
) -> Tuple[bool, Any]:
    """Recovery branches that run BEFORE ``classify_api_error``: UnicodeEncodeError
    sanitization (surrogates, then ASCII codec), provider image-content rejection
    (switch session to text-only), and the Bedrock AnthropicBedrock SDK streaming
    fallback. Returns ``(retry_now, active_system_prompt)``; the prompt may be
    ASCII-sanitized in place."""
    # UnicodeEncodeError recovery: lone surrogates (clipboard paste) or an
    # ASCII codec under a non-UTF-8 locale. Sanitize in-place; at most two
    # retries (surrogate strip, then ASCII-only).
    if isinstance(api_error, UnicodeEncodeError) and getattr(agent, '_unicode_sanitization_passes', 0) < 2:
        _err_str = str(api_error).lower()
        _is_ascii_codec = "'ascii'" in _err_str or "ascii" in _err_str
        # Surrogate errors: utf-8 refusing U+D800..U+DFFF
        # ("surrogates not allowed").
        _is_surrogate_error = (
            "surrogate" in _err_str
            or ("'utf-8'" in _err_str and not _is_ascii_codec)
        )
        # Sanitize `messages` AND `api_messages` (which may carry
        # `reasoning_content`/`reasoning_details`), plus `api_kwargs` and
        # `prefill_messages` if present. Mirrors the ASCII recovery below.
        _surrogates_found = _sanitize_messages_surrogates(messages)
        if isinstance(api_messages, list):
            if _sanitize_messages_surrogates(api_messages):
                _surrogates_found = True
        if isinstance(api_kwargs, dict):
            if _sanitize_structure_surrogates(api_kwargs):
                _surrogates_found = True
        if isinstance(getattr(agent, "prefill_messages", None), list):
            if _sanitize_messages_surrogates(agent.prefill_messages):
                _surrogates_found = True
        # Gate the retry on the error type, not on whether anything was
        # found — a new transformed field could slip through. Bounded by
        # _unicode_sanitization_passes < 2 (outer guard).
        if _surrogates_found or _is_surrogate_error:
            agent._unicode_sanitization_passes += 1
            if _surrogates_found:
                agent._buffer_vprint(
                    "⚠️  Stripped invalid surrogate characters from messages. Retrying..."
                )
            else:
                agent._buffer_vprint(
                    "⚠️  Surrogate encoding error — retrying after full-payload sanitization..."
                )
            return True, active_system_prompt
        if _is_ascii_codec:
            agent._force_ascii_payload = True
            # ASCII codec: strip all non-ASCII from messages/tool schemas
            # and retry — both `messages` and `api_messages` (which may
            # carry extra fields like reasoning_content).
            _messages_sanitized = _sanitize_messages_non_ascii(messages)
            if isinstance(api_messages, list):
                _sanitize_messages_non_ascii(api_messages)
            # Also sanitize the last api_kwargs so a non-ASCII transformed
            # field doesn't survive via _build_api_kwargs cache paths.
            if isinstance(api_kwargs, dict):
                _sanitize_structure_non_ascii(api_kwargs)
            _prefill_sanitized = False
            if isinstance(getattr(agent, "prefill_messages", None), list):
                _prefill_sanitized = _sanitize_messages_non_ascii(agent.prefill_messages)

            _tools_sanitized = False
            if isinstance(getattr(agent, "tools", None), list):
                _tools_sanitized = _sanitize_tools_non_ascii(agent.tools)

            _system_sanitized = False
            if isinstance(active_system_prompt, str):
                _sanitized_system = _strip_non_ascii(active_system_prompt)
                if _sanitized_system != active_system_prompt:
                    active_system_prompt = _sanitized_system
                    agent._cached_system_prompt = _sanitized_system
                    _system_sanitized = True
            if isinstance(getattr(agent, "ephemeral_system_prompt", None), str):
                _sanitized_ephemeral = _strip_non_ascii(agent.ephemeral_system_prompt)
                if _sanitized_ephemeral != agent.ephemeral_system_prompt:
                    agent.ephemeral_system_prompt = _sanitized_ephemeral
                    _system_sanitized = True

            _headers_sanitized = False
            _default_headers = (
                agent._client_kwargs.get("default_headers")
                if isinstance(getattr(agent, "_client_kwargs", None), dict)
                else None
            )
            if isinstance(_default_headers, dict):
                _headers_sanitized = _sanitize_structure_non_ascii(_default_headers)

            # Sanitize the API key: non-ASCII in credentials makes httpx
            # fail encoding the Authorization header — the usual persistent
            # cause after message/tool sanitization (#6843).
            _credential_sanitized = False
            _raw_key = getattr(agent, "api_key", None) or ""
            # Entra ID bearer providers are callables minting ASCII JWTs;
            # skip (``_strip_non_ascii`` would crash on a callable).
            if _raw_key and isinstance(_raw_key, str):
                _clean_key = _strip_non_ascii(_raw_key)
                if _clean_key != _raw_key:
                    agent.api_key = _clean_key
                    if isinstance(getattr(agent, "_client_kwargs", None), dict):
                        agent._client_kwargs["api_key"] = _clean_key
                    # Also update the live client — auth_headers reads its
                    # own api_key copy on every request.
                    if getattr(agent, "client", None) is not None and hasattr(agent.client, "api_key"):
                        agent.client.api_key = _clean_key
                    _credential_sanitized = True
                    agent._vprint(
                        f"{agent.log_prefix}⚠️  API key contained non-ASCII characters "
                        f"(bad copy-paste?) — stripped them. If auth fails, "
                        f"re-copy the key from your provider's dashboard.",
                        force=True,
                    )

            # Always retry on ASCII codec detection: _force_ascii_payload
            # sanitizes the full api_kwargs next iteration even when
            # checks above find nothing. Bounded by passes < 2.
            agent._unicode_sanitization_passes += 1
            _any_sanitized = (
                _messages_sanitized
                or _prefill_sanitized
                or _tools_sanitized
                or _system_sanitized
                or _headers_sanitized
                or _credential_sanitized
            )
            if _any_sanitized:
                agent._vprint(
                    f"{agent.log_prefix}⚠️  System encoding is ASCII — stripped non-ASCII characters from request payload. Retrying...",
                    force=True,
                )
            else:
                agent._vprint(
                    f"{agent.log_prefix}⚠️  System encoding is ASCII — enabling full-payload sanitization for retry...",
                    force=True,
                )
            return True, active_system_prompt

    # ── Image-rejection recovery ──────────────────────────────
    # Some providers 4xx on image_url content: strip images, mark session
    # vision-unsupported, retry text-only. English phrase match; extend it.
    _err_body = ""
    try:
        _err_body = str(getattr(api_error, "body", None) or
                        getattr(api_error, "message", None) or
                        str(api_error))
    except Exception:
        pass
    _err_status = getattr(api_error, "status_code", None)
    _looks_like_image_rejection = _looks_like_image_content_rejection(_err_body)
    # 4xx-only gate: 5xx/timeouts are transient and take the retry path.
    _status_ok = _err_status is None or (400 <= int(_err_status) < 500)
    if (
        getattr(agent, "_vision_supported", True)
        and _looks_like_image_rejection
        and _status_ok
    ):
        agent._vision_supported = False
        _imgs_removed = _strip_images_from_messages(messages)
        if isinstance(api_messages, list):
            _strip_images_from_messages(api_messages)
        agent._vprint(
            f"{agent.log_prefix}⚠️  Server rejected image content — "
            f"switching to text-only mode for this session"
            + (". Stripped images from history and retrying." if _imgs_removed else "."),
            force=True,
        )
        return True, active_system_prompt

    # ── Bedrock AnthropicBedrock SDK streaming failure ──
    # SDK raises "Unexpected event order" when Bedrock errors before
    # message_start; fall back to native Converse for this session (#28156).
    if (
        isinstance(api_error, RuntimeError)
        and "unexpected event order" in str(api_error).lower()
        and getattr(agent, "provider", "") == "bedrock"
        and agent.api_mode == "anthropic_messages"
        and not getattr(agent, "_bedrock_converse_fallback_attempted", False)
    ):
        agent._bedrock_converse_fallback_attempted = True
        agent.api_mode = "bedrock_converse"
        agent._bedrock_region = getattr(agent, "_bedrock_region", None) or "us-east-1"
        agent.client = None  # Drop the AnthropicBedrock client
        agent._client_kwargs = {}
        agent._vprint(
            f"{agent.log_prefix}⚠️  AnthropicBedrock SDK streaming failed — "
            f"falling back to native Converse API for this session.",
            force=True,
        )
        return True, active_system_prompt
    return False, active_system_prompt
