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
import re
import time
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
    close_interrupted_tool_sequence,
)
from agent.turn_retry_state import TurnRetryState
from utils import base_url_host_matches

logger = logging.getLogger("agent.conversation_loop")


def _image_error_max_dimension(error: Exception) -> Optional[int]:
    """Extract a provider-reported image dimension ceiling, if present."""
    parts = []
    for value in (
        error,
        getattr(error, "message", None),
        getattr(error, "body", None),
    ):
        if value:
            try:
                parts.append(str(value))
            except Exception:
                pass
    text = " ".join(parts).lower()
    if "image" not in text or "dimension" not in text or "max allowed size" not in text:
        return None

    match = re.search(r"max allowed size(?:\s+for [^:]+)?:\s*(\d{3,5})\s*pixels?", text)
    if not match:
        return None
    try:
        max_dimension = int(match.group(1))
    except ValueError:
        return None
    if 512 <= max_dimension <= 8000:
        return max_dimension
    return None


def _try_refresh_nous_paid_entitlement_credentials(agent) -> bool:
    """Refresh Nous runtime credentials after a fresh paid-entitlement check."""
    try:
        from hermes_cli.nous_account import get_nous_portal_account_info

        account_info = get_nous_portal_account_info(force_fresh=True)
        if account_info.paid_service_access is not True:
            return False
        return agent._try_refresh_nous_client_credentials(
            force=True,
        )
    except Exception:
        return False


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


def recover_after_classification(
    agent: Any,
    api_error: Exception,
    classified: Any,
    _retry: TurnRetryState,
    *,
    status_code: Optional[int],
    error_context: Any,
    messages: List[Dict[str, Any]],
    api_messages: Any,
) -> Tuple[bool, bool]:
    """One-shot recovery chain that runs AFTER ``classify_api_error`` and before the
    generic retry path. Order is load-bearing (each branch may ``return`` early):
    Nous paid-entitlement refresh → credential-pool rotation → image shrink →
    multimodal-tool-content strip → corrupt-image strip → Anthropic OAuth 1M-beta
    disable → per-provider 401 credential refresh (codex/xai, vertex, nous, copilot,
    anthropic, with user-facing diagnostics when refresh fails) → thinking-signature
    strip → invalid-encrypted-content replay disable → native-compaction reject →
    llama.cpp grammar strip. Returns ``(retry_now, recovered_with_pool)``;
    ``recovered_with_pool`` is read later by the Nous rate-limit guard."""
    # Shared with the billing/entitlement helpers that stay in the loop module;
    # lazy so this module never imports agent.conversation_loop at load time.
    from agent.conversation_loop import (
        _is_copilot_provider,
        _is_nous_inference_route,
        _print_nous_entitlement_guidance,
    )

    if (
        classified.reason == FailoverReason.billing
        and _is_nous_inference_route(
            getattr(agent, "provider", "") or "",
            getattr(agent, "base_url", "") or "",
        )
        and not _retry.nous_paid_entitlement_refresh_attempted
    ):
        _retry.nous_paid_entitlement_refresh_attempted = True
        if _try_refresh_nous_paid_entitlement_credentials(agent):
            agent._vprint(
                f"{agent.log_prefix}🔐 Nous paid access verified — "
                "refreshed runtime credentials and retrying request...",
                force=True,
            )
            return True, False

    recovered_with_pool, _retry.has_retried_429 = agent._recover_with_credential_pool(
        status_code=status_code,
        has_retried_429=_retry.has_retried_429,
        classified_reason=classified.reason,
        error_context=error_context,
        billing_unverified=classified.billing_unverified,
    )
    if recovered_with_pool:
        return True, recovered_with_pool

    # Image-too-large recovery: shrink oversized native image parts
    # in-place and retry once; otherwise fall through to normal handling.
    if (
        classified.reason == FailoverReason.image_too_large
        and not _retry.image_shrink_retry_attempted
    ):
        _retry.image_shrink_retry_attempted = True
        image_max_dimension = _image_error_max_dimension(api_error) or 8000
        if agent._try_shrink_image_parts_in_messages(
            api_messages,
            max_dimension=image_max_dimension,
        ):
            agent._vprint(
                f"{agent.log_prefix}📐 Image(s) exceeded provider size limit — "
                f"shrank and retrying...",
                force=True,
            )
            return True, recovered_with_pool
        else:
            logger.info(
                "image-shrink recovery: no data-URL image parts found "
                "or shrink didn't reduce size; surfacing original error."
            )

    # Multimodal-tool-content recovery: strict OpenAI-spec providers 400
    # on list-type tool content. Strip images, mark (provider, model)
    # no-list-tool-content for the session, retry once (#27344).
    if (
        classified.reason == FailoverReason.multimodal_tool_content_unsupported
        and not _retry.multimodal_tool_content_retry_attempted
    ):
        _retry.multimodal_tool_content_retry_attempted = True
        if agent._try_strip_image_parts_from_tool_messages(api_messages):
            agent._vprint(
                f"{agent.log_prefix}📐 Provider rejected list-type tool content — "
                f"downgraded screenshots to text and retrying...",
                force=True,
            )
            return True, recovered_with_pool
        else:
            logger.info(
                "multimodal-tool-content recovery: no list-type tool "
                "messages with image parts found; surfacing original error."
            )

    # Image-corrupt recovery: provider rejected the image bytes; shrinking
    # can't help, so strip image parts and retry once (#69078).
    if classified.reason == FailoverReason.image_corrupt:
        # Strip ONLY the per-call copy: replacing msg["content"] on the
        # shallow api_messages rows keeps canonical history's images
        # (copy-on-write; transient rejection must not erase history).
        _imgs_removed = False
        if isinstance(api_messages, list):
            _imgs_removed = _strip_images_from_messages(api_messages)
        if _imgs_removed:
            agent._vprint(
                f"{agent.log_prefix}⚠️  Provider rejected a corrupted image — "
                f"stripped images from the retry payload and retrying...",
                force=True,
            )
            return True, recovered_with_pool
        else:
            logger.info(
                "image-corrupt recovery: no image parts found to "
                "strip; surfacing original error."
            )

    # Anthropic OAuth subscription rejected the 1M-context beta: disable it
    # for this session, rebuild the client, retry once. Reactive so capable
    # subscriptions keep full 1M context (#17680).
    if (
        classified.reason == FailoverReason.oauth_long_context_beta_forbidden
        and agent.api_mode == "anthropic_messages"
        and agent._is_anthropic_oauth
        and not _retry.oauth_1m_beta_retry_attempted
    ):
        _retry.oauth_1m_beta_retry_attempted = True
        if not getattr(agent, "_oauth_1m_beta_disabled", False):
            agent._oauth_1m_beta_disabled = True
            try:
                agent._anthropic_client.close()
            except Exception:
                pass
            agent._rebuild_anthropic_client()
            agent._vprint(
                f"{agent.log_prefix}🔕 OAuth subscription doesn't support "
                f"the 1M-context beta — disabled for this session and retrying...",
                force=True,
            )
            return True, recovered_with_pool

    if (
        agent.api_mode == "codex_responses"
        and agent.provider in {"openai-codex", "xai-oauth"}
        and status_code == 401
        and not _retry.codex_auth_retry_attempted
    ):
        _retry.codex_auth_retry_attempted = True
        if agent._try_refresh_codex_client_credentials(force=True):
            _label = "xAI OAuth" if agent.provider == "xai-oauth" else "Codex"
            agent._buffer_vprint(f"🔐 {_label} auth refreshed after 401. Retrying request...")
            return True, recovered_with_pool
    if (
        agent.api_mode == "chat_completions"
        and agent.provider == "vertex"
        and status_code == 401
        and not _retry.vertex_auth_retry_attempted
    ):
        _retry.vertex_auth_retry_attempted = True
        if agent._try_refresh_vertex_client_credentials():
            agent._buffer_vprint("🔐 Vertex AI token refreshed after 401. Retrying request...")
            return True, recovered_with_pool
    if (
        agent.api_mode in ("chat_completions", "anthropic_messages")
        and agent.provider == "nous"
        and status_code == 401
        and not _retry.nous_auth_retry_attempted
    ):
        _retry.nous_auth_retry_attempted = True
        if agent._try_refresh_nous_client_credentials(force=True):
            agent._buffer_vprint("🔐 Nous agent key refreshed after 401. Retrying request...")
            return True, recovered_with_pool
        # Refresh didn't help: likely Portal OAuth expired/revoked,
        # no credits, or agent key blocked.
        from hermes_constants import display_hermes_home as _dhh_fn
        _dhh = _dhh_fn()
        _body_text = ""
        try:
            _body = getattr(api_error, "body", None) or getattr(api_error, "response", None)
            if _body is not None:
                _body_text = str(_body)[:200]
        except Exception:
            pass
        print(f"{agent.log_prefix}🔐 Nous 401 — Portal authentication failed.")
        if _body_text:
            print(f"{agent.log_prefix}   Response: {_body_text}")
        if not _print_nous_entitlement_guidance(agent, "Nous model access"):
            print(f"{agent.log_prefix}   Most likely: Portal OAuth expired, account out of credits, or agent key revoked.")
        print(f"{agent.log_prefix}   Troubleshooting:")
        print(f"{agent.log_prefix}     • Re-authenticate: hermes auth add nous")
        print(f"{agent.log_prefix}     • Check credits / billing: https://portal.nousresearch.com")
        print(f"{agent.log_prefix}     • Verify stored credentials: {_dhh}/auth.json")
        print(f"{agent.log_prefix}     • Switch providers temporarily: /model <model> --provider openrouter")
    if (
        _is_copilot_provider(agent)
        and status_code == 401
        and not _retry.copilot_auth_retry_attempted
    ):
        _retry.copilot_auth_retry_attempted = True
        if agent._try_refresh_copilot_client_credentials():
            agent._buffer_vprint("🔐 Copilot credentials refreshed after 401. Retrying request...")
            return True, recovered_with_pool
    if (
        agent.api_mode == "anthropic_messages"
        and status_code == 401
        and hasattr(agent, '_anthropic_api_key')
        and not _retry.anthropic_auth_retry_attempted
    ):
        _retry.anthropic_auth_retry_attempted = True
        from agent.anthropic_adapter import _is_oauth_token
        from agent.azure_identity_adapter import is_token_provider
        if agent._try_refresh_anthropic_client_credentials():
            print(f"{agent.log_prefix}🔐 Anthropic credentials refreshed after 401. Retrying request...")
            return True, recovered_with_pool
        # Credential refresh didn't help — show diagnostic info
        key = agent._anthropic_api_key
        print(f"{agent.log_prefix}🔐 Anthropic 401 — authentication failed.")
        if is_token_provider(key):
            # Azure Foundry Entra ID: JWT minted per-request by an httpx
            # hook; 401 = Azure rejected it (RBAC, az login, IMDS).
            print(f"{agent.log_prefix}   Auth method: Microsoft Entra ID (httpx event hook)")
            print(f"{agent.log_prefix}   Run `hermes doctor` for credential-chain diagnostics, or")
            print(f"{agent.log_prefix}   `az login` if your developer session expired.")
        else:
            auth_method = "Bearer (OAuth/setup-token)" if _is_oauth_token(key) else "x-api-key (API key)"
            print(f"{agent.log_prefix}   Auth method: {auth_method}")
            print(f"{agent.log_prefix}   Token prefix: {key[:12]}..." if isinstance(key, str) and len(key) > 12 else f"{agent.log_prefix}   Token: (empty or short)")
        print(f"{agent.log_prefix}   Troubleshooting:")
        from hermes_constants import display_hermes_home as _dhh_fn
        _dhh = _dhh_fn()
        print(f"{agent.log_prefix}     • Check ANTHROPIC_TOKEN in {_dhh}/.env for Hermes-managed OAuth/setup tokens")
        print(f"{agent.log_prefix}     • Check ANTHROPIC_API_KEY in {_dhh}/.env for API keys or legacy token values")
        print(f"{agent.log_prefix}     • For API keys: verify at https://platform.claude.com/settings/keys")
        print(f"{agent.log_prefix}     • For Claude Code: run 'claude /login' to refresh, then retry")
        print(f"{agent.log_prefix}     • Legacy cleanup: hermes config set ANTHROPIC_TOKEN \"\"")
        print(f"{agent.log_prefix}     • Clear stale keys: hermes config set ANTHROPIC_API_KEY \"\"")

    # Thinking block signature recovery: upstream mutation invalidates
    # Anthropic's signature (400). Strip ``reasoning_details`` from
    # ``api_messages`` only, never ``messages`` (state.db). One-shot.
    if (
        classified.reason == FailoverReason.thinking_signature
        and not _retry.thinking_sig_retry_attempted
    ):
        _retry.thinking_sig_retry_attempted = True
        _api_stripped = 0
        for _m in api_messages:
            if isinstance(_m, dict) and "reasoning_details" in _m:
                _m.pop("reasoning_details", None)
                _api_stripped += 1
        agent._vprint(
            f"{agent.log_prefix}⚠️  Thinking block signature invalid, "
            f"stripped reasoning_details from api_messages for retry...",
            force=True,
        )
        logger.warning(
            "%sThinking block signature recovery: stripped "
            "reasoning_details from %d api_messages "
            "(canonical messages unchanged)",
            agent.log_prefix, _api_stripped,
        )
        return True, recovered_with_pool

    # ── Invalid encrypted reasoning replay recovery ───────
    # 400 ``invalid_encrypted_content`` on a stale ``codex_reasoning_items``
    # blob: disable replay for the session, strip cached items, retry once.
    if (
        classified.reason == FailoverReason.invalid_encrypted_content
        and not _retry.invalid_encrypted_content_retry_attempted
        and agent.api_mode == "codex_responses"
        and bool(getattr(agent, "_codex_reasoning_replay_enabled", True))
        and any(
            isinstance(_m, dict)
            and _m.get("role") == "assistant"
            and isinstance(_m.get("codex_reasoning_items"), list)
            and _m.get("codex_reasoning_items")
            for _m in messages
        )
    ):
        _retry.invalid_encrypted_content_retry_attempted = True
        replay_stats = agent._disable_codex_reasoning_replay(messages)
        agent._vprint(
            f"{agent.log_prefix}⚠️  Encrypted reasoning replay was rejected by the provider — "
            f"disabled replay and stripped {replay_stats['items']} item(s) from "
            f"{replay_stats['messages']} message(s), retrying...",
            force=True,
        )
        logger.warning(
            "%sInvalid encrypted reasoning recovery: disabled replay and stripped %d items from %d messages",
            agent.log_prefix,
            replay_stats["items"],
            replay_stats["messages"],
        )
        return True, recovered_with_pool

    # ── Native compaction rejection recovery ──────────────
    # Structured 400 naming ``context_management``: disable native
    # compaction for the session, retry once; local compression takes over.
    if (
        agent.api_mode == "codex_responses"
        and not _retry.native_compaction_reject_retry_attempted
        and bool(getattr(agent, "codex_responses_native_compaction", False))
    ):
        from agent.native_compaction import is_native_compaction_rejection
        if is_native_compaction_rejection(
            api_error, getattr(api_error, "status_code", None)
        ):
            _retry.native_compaction_reject_retry_attempted = True
            agent.codex_responses_native_compaction = False
            agent._vprint(
                f"{agent.log_prefix}⚠️  Provider rejected native compaction "
                f"(context_management) — disabled for this session, "
                f"local compression stays active. Retrying...",
                force=True,
            )
            logger.warning(
                "%sNative compaction rejection recovery: disabled "
                "codex_responses_native for this session and retrying",
                agent.log_prefix,
            )
            return True, recovered_with_pool

    # ── llama.cpp grammar-parse recovery ──────────────────
    # ``json-schema-to-grammar`` rejects regex escapes and most ``format``
    # values: strip ``pattern``/``format`` from ``agent.tools``, retry once.
    if (
        classified.reason == FailoverReason.llama_cpp_grammar_pattern
        and not _retry.llama_cpp_grammar_retry_attempted
    ):
        _retry.llama_cpp_grammar_retry_attempted = True
        try:
            from tools.schema_sanitizer import strip_pattern_and_format
            _, _stripped = strip_pattern_and_format(agent.tools)
        except Exception as _strip_exc:  # pragma: no cover — defensive
            logger.warning(
                "%sllama.cpp grammar recovery: strip helper failed: %s",
                agent.log_prefix, _strip_exc,
            )
            _stripped = 0
        if _stripped:
            agent._vprint(
                f"{agent.log_prefix}⚠️  llama.cpp rejected tool schema grammar — "
                f"stripped {_stripped} pattern/format keyword(s), retrying...",
                force=True,
            )
            logger.warning(
                "%sllama.cpp grammar recovery: stripped %d "
                "pattern/format keyword(s) from tool schemas",
                agent.log_prefix, _stripped,
            )
            return True, recovered_with_pool
        # No keywords found to strip — fall through to normal
        # retry path rather than loop forever on the same error.
        logger.warning(
            "%sllama.cpp grammar error but no pattern/format "
            "keywords to strip — falling through to normal retry",
            agent.log_prefix,
        )
    return False, recovered_with_pool


def nonretryable_client_error_result(
    agent: Any,
    api_error: Exception,
    classified: Any,
    *,
    status_code: Optional[int],
    api_kwargs: Any,
    api_messages: Any,
    messages: List[Dict[str, Any]],
    conversation_history: Any,
    api_call_count: int,
    approx_tokens: int,
    provider: Any,
    base_url: Any,
    model: Any,
) -> Dict[str, Any]:
    """Terminal path for a non-retryable 4xx once fallback is exhausted: dump the
    request for debugging, flush the buffered retry trace, print actionable auth /
    billing / content-policy / TLS guidance, persist the session (skipped for likely
    context-overflow 400s so the failure does not grow the session, #1630) and build
    the failed-turn result dict."""
    # Result/guidance helpers stay in the loop module (tests import + patch them
    # there); lazy import avoids a load-time cycle.
    from agent.conversation_loop import (
        _CONTENT_POLICY_RECOVERY_HINT,
        _billing_failure_result,
        _content_policy_blocked_result,
        _print_billing_or_entitlement_guidance,
        _print_nous_entitlement_guidance,
    )

    if api_kwargs is not None:
        agent._dump_api_request_debug(
            api_kwargs, reason="non_retryable_client_error", error=api_error,
        )
    # Terminal — flush buffered context so the user sees
    # what was tried before the abort.
    agent._flush_status_buffer()
    # Summarize once: Cloudflare/proxy HTML pages and raw provider
    # bodies must be collapsed here or they leak verbatim via the
    # ``error`` field.
    _nonretryable_summary = agent._summarize_api_error(api_error)
    if classified.reason == FailoverReason.content_policy_blocked:
        agent._emit_status(
            f"❌ Provider safety filter blocked this request: "
            f"{_nonretryable_summary}"
        )
    elif classified.reason == FailoverReason.ssl_cert_verification:
        agent._emit_status(
            f"❌ TLS certificate verification failed: "
            f"{_nonretryable_summary}"
        )
    else:
        agent._emit_status(
            f"❌ Non-retryable error (HTTP {status_code}): "
            f"{_nonretryable_summary}"
        )
    agent._vprint(f"{agent.log_prefix}❌ Non-retryable client error (HTTP {status_code}). Aborting.", force=True)
    agent._vprint(f"{agent.log_prefix}   🔌 Provider: {provider}  Model: {model}", force=True)
    agent._vprint(f"{agent.log_prefix}   🌐 Endpoint: {base_url}", force=True)
    # Actionable guidance for common auth errors
    if classified.is_auth or classified.reason == FailoverReason.billing:
        if classified.reason == FailoverReason.billing and _print_billing_or_entitlement_guidance(
            agent,
            capability="model access",
            provider=provider,
            base_url=str(base_url),
            model=model,
            unverified=classified.billing_unverified,
        ):
            pass
        elif provider == "nous" and _print_nous_entitlement_guidance(
            agent,
            "Nous model access",
        ):
            pass
        elif provider in {"openai-codex", "xai-oauth", "nous"} and status_code == 401:
            if provider == "openai-codex":
                agent._vprint(f"{agent.log_prefix}   💡 Codex OAuth token was rejected (HTTP 401). Your token may have been", force=True)
                agent._vprint(f"{agent.log_prefix}      refreshed by another client (Codex CLI, VS Code). To fix:", force=True)
                agent._vprint(f"{agent.log_prefix}      1. Run `codex` in your terminal to generate fresh tokens.", force=True)
                agent._vprint(f"{agent.log_prefix}      2. Then run `hermes auth` to re-authenticate.", force=True)
            elif provider == "xai-oauth":
                agent._vprint(f"{agent.log_prefix}   💡 xAI OAuth token was rejected (HTTP 401). To fix:", force=True)
                agent._vprint(f"{agent.log_prefix}      re-authenticate with xAI Grok OAuth (SuperGrok / Premium+) from `hermes model`.", force=True)
            else:  # nous
                agent._vprint(f"{agent.log_prefix}   💡 Nous Portal OAuth token was rejected (HTTP 401). Your token may be", force=True)
                agent._vprint(f"{agent.log_prefix}      expired, revoked, or your account may be out of credits. To fix:", force=True)
                agent._vprint(f"{agent.log_prefix}      1. Re-authenticate: hermes portal", force=True)
                agent._vprint(f"{agent.log_prefix}      2. Check your portal account: https://portal.nousresearch.com", force=True)
                # ``:free`` is OpenRouter slug syntax; Nous Portal will reject
                # the model name even after a successful re-auth.
                if isinstance(model, str) and model.endswith(":free"):
                    agent._vprint(f"{agent.log_prefix}      ⚠️  Note: `{model}` looks like an OpenRouter slug (`:free` suffix).", force=True)
                    agent._vprint(f"{agent.log_prefix}         Nous Portal won't recognize that model name. Either switch to a", force=True)
                    agent._vprint(f"{agent.log_prefix}         Nous catalog model, or run `/model openrouter:{model}` to use OpenRouter.", force=True)
        else:
            agent._vprint(f"{agent.log_prefix}   💡 Your API key was rejected by the provider. Check:", force=True)
            agent._vprint(f"{agent.log_prefix}      • Is the key valid? Run: hermes setup", force=True)
            agent._vprint(f"{agent.log_prefix}      • Does your account have access to {model}?", force=True)
            if base_url_host_matches(str(base_url), "openrouter.ai"):
                agent._vprint(f"{agent.log_prefix}      • Check credits: https://openrouter.ai/settings/credits", force=True)
    else:
        agent._vprint(f"{agent.log_prefix}   💡 This type of error won't be fixed by retrying.", force=True)
    # Content-policy blocks get their own guidance: the provider refused
    # this prompt, so recovery is a rephrase or another model, not
    # key/retry advice.
    if classified.reason == FailoverReason.content_policy_blocked:
        agent._vprint(
            f"{agent.log_prefix}   💡 The provider's safety filter rejected this specific prompt.",
            force=True,
        )
        agent._vprint(
            f"{agent.log_prefix}      • Try rephrasing the request, narrowing the context, or splitting into smaller steps.",
            force=True,
        )
        agent._vprint(
            f"{agent.log_prefix}      • Configure a fallback provider so future blocks route automatically:",
            force=True,
        )
        agent._vprint(
            f"{agent.log_prefix}        hermes fallback add   (interactive picker — same as `hermes model`)",
            force=True,
        )
    # TLS certificate failures are environment problems — name the knobs
    # that fix each common cause.
    if classified.reason == FailoverReason.ssl_cert_verification:
        agent._vprint(
            f"{agent.log_prefix}   💡 The TLS certificate chain could not be verified. This fails the same",
            force=True,
        )
        agent._vprint(
            f"{agent.log_prefix}      way on every retry — fix the environment, then try again:",
            force=True,
        )
        agent._vprint(
            f"{agent.log_prefix}      • Corporate TLS-inspecting proxy? Point Python at its CA bundle:",
            force=True,
        )
        agent._vprint(
            f"{agent.log_prefix}        export SSL_CERT_FILE=/path/to/corp-ca.pem  (also REQUESTS_CA_BUNDLE)",
            force=True,
        )
        agent._vprint(
            f"{agent.log_prefix}      • Missing/stale system CA store? Install/refresh it:",
            force=True,
        )
        agent._vprint(
            f"{agent.log_prefix}        pip install --upgrade certifi   (macOS: run 'Install Certificates.command')",
            force=True,
        )
        agent._vprint(
            f"{agent.log_prefix}      • Self-signed local endpoint (llama.cpp, LM Studio, vLLM)? Use http://",
            force=True,
        )
        agent._vprint(
            f"{agent.log_prefix}        for localhost, or add the server's cert to your trust store.",
            force=True,
        )
    logger.error("%sNon-retryable client error: %s", agent.log_prefix, api_error)
    # Skip persistence on likely context-overflow (400 + large session):
    # persisting the failed message grows the session and repeats the
    # failure. (#1630)
    if status_code == 400 and (approx_tokens > 50000 or len(api_messages) > 80):
        agent._vprint(
            f"{agent.log_prefix}⚠️  Skipping session persistence "
            f"for large failed session to prevent growth loop.",
            force=True,
        )
    else:
        agent._persist_session(messages, conversation_history)
    if classified.reason == FailoverReason.content_policy_blocked:
        _policy_response = (
            "⚠️  The model provider's safety filter blocked this request "
            "(not a Hermes/gateway failure).\n\n"
            f"Provider message: {_nonretryable_summary}\n\n"
            f"{_CONTENT_POLICY_RECOVERY_HINT}"
        )
        return _content_policy_blocked_result(
            messages,
            api_call_count,
            final_response=_policy_response,
            error_detail=_nonretryable_summary,
        )
    # Billing walls get the same structured recovery descriptor as the
    # max-retries path so every surface renders one consistent signal.
    if classified.reason == FailoverReason.billing:
        return _billing_failure_result(
            classified=classified,
            summary=_nonretryable_summary,
            messages=messages,
            api_call_count=api_call_count,
            provider=provider,
            base_url=base_url,
            model=model,
        )
    return {
        "final_response": _nonretryable_summary,
        "messages": messages,
        "api_calls": api_call_count,
        "completed": False,
        "failed": True,
        "error": _nonretryable_summary,
    }


def max_retries_exhausted_result(
    agent: Any,
    api_error: Exception,
    classified: Any,
    *,
    max_retries: int,
    is_rate_limited: bool,
    error_msg: str,
    api_kwargs: Any,
    api_messages: Any,
    messages: List[Dict[str, Any]],
    conversation_history: Any,
    api_call_count: int,
    approx_tokens: int,
    provider: Any,
    base_url: Any,
    model: Any,
) -> Dict[str, Any]:
    """Terminal path once ``retry_count >= max_retries`` and transport recovery +
    fallback both failed: flush the buffered trace, emit the billing / rate-limit /
    generic status line, print stream-drop or thinking-timeout guidance (the latter
    wins, #52310), persist, and build the failed-turn result dict carrying the
    classified ``failure_reason`` / ``failure_retryable`` / ``billing_block``."""
    # Result/guidance helpers stay in the loop module (tests import + patch them
    # there); lazy import avoids a load-time cycle.
    from agent.conversation_loop import (
        _billing_block_dict,
        _billing_or_entitlement_message,
        _billing_terminal_label,
        _print_billing_or_entitlement_guidance,
    )

    # Terminal — flush buffered retry/fallback trace.
    agent._flush_status_buffer()
    _final_summary = agent._summarize_api_error(api_error)
    _billing_guidance = ""
    if classified.reason == FailoverReason.billing:
        if classified.billing_unverified:
            # Ambiguous body (#82154) — hedge the terminal line.
            agent._emit_status(
                "❌ Provider reported usage/credit exhaustion "
                f"(unverified — may be a content-filter rejection) — {_final_summary}"
            )
        else:
            agent._emit_status(f"❌ Billing or credits exhausted — {_final_summary}")
        _billing_guidance = _billing_or_entitlement_message(
            capability="model access",
            provider=provider,
            base_url=str(base_url),
            model=model,
            unverified=classified.billing_unverified,
        )
        _print_billing_or_entitlement_guidance(
            agent,
            capability="model access",
            provider=provider,
            base_url=str(base_url),
            model=model,
            unverified=classified.billing_unverified,
        )
    elif is_rate_limited:
        agent._emit_status(f"❌ Rate limited after {max_retries} retries — {_final_summary}")
    else:
        agent._emit_status(f"❌ API failed after {max_retries} retries — {_final_summary}")
    agent._vprint(f"{agent.log_prefix}   💀 Final error: {_final_summary}", force=True)

    # SSE stream-drop (e.g. "Network connection lost"): usually a
    # proxy/CDN cutting a very large tool call mid-response; give
    # actionable guidance.
    _is_stream_drop = (
        not getattr(api_error, "status_code", None)
        and any(p in error_msg for p in (
            "connection lost", "connection reset",
            "connection closed", "network connection",
            "network error", "terminated",
        ))
    )
    if _is_stream_drop:
        agent._vprint(
            f"{agent.log_prefix}   💡 The provider's stream "
            f"connection keeps dropping. This often happens "
            f"when the model tries to write a very large "
            f"file in a single tool call.",
            force=True,
        )
        agent._vprint(
            f"{agent.log_prefix}      Try asking the model "
            f"to use execute_code with Python's open() for "
            f"large files, or to write the file in smaller "
            f"sections.",
            force=True,
        )

    # Thinking-timeout: a known reasoning model hit a transport error
    # before the first content token. Distinct from _is_stream_drop;
    # detection lives in agent.thinking_timeout_guidance. (#52310)
    from agent.thinking_timeout_guidance import (
        is_thinking_timeout,
    )
    _is_thinking_timeout = is_thinking_timeout(
        classified,
        model,
        error_msg,
    )
    if _is_thinking_timeout:
        agent._vprint(
            f"{agent.log_prefix}   💡 The model's thinking "
            f"phase exceeded the upstream proxy's idle "
            f"timeout before the first content token "
            f"arrived. This is a known issue with "
            f"reasoning models behind cloud gateways "
            f"(NVIDIA NIM, OpenAI, Anthropic, DeepSeek).",
            force=True,
        )
        agent._vprint(
            f"{agent.log_prefix}      Workarounds in priority order:",
            force=True,
        )
        agent._vprint(
            f"{agent.log_prefix}      1. Set "
            f"`providers.{provider}.models.{model}.stale_timeout_seconds: 900` "
            f"in `~/.hermes/config.yaml` to extend the per-call "
            f"timeout. (Hermes's built-in floor is 600s for "
            f"known reasoning models — if you still see this "
            f"after raising, the upstream cap is even shorter.)",
            force=True,
        )
        agent._vprint(
            f"{agent.log_prefix}      2. Lower `reasoning_budget` or set "
            f"`reasoning_effort: medium` on this model if the provider supports it.",
            force=True,
        )
        agent._vprint(
            f"{agent.log_prefix}      3. Use a smaller / faster reasoning "
            f"model if the task doesn't require deep thinking.",
            force=True,
        )

    logger.error(
        "%sAPI call failed after %s retries. %s | provider=%s model=%s msgs=%s tokens=~%s",
        agent.log_prefix, max_retries, _final_summary,
        provider, model, len(api_messages), f"{approx_tokens:,}",
    )
    if api_kwargs is not None:
        agent._dump_api_request_debug(
            api_kwargs, reason="max_retries_exhausted", error=api_error,
        )
    agent._persist_session(messages, conversation_history)
    _billing_block = None
    _billing_unverified = False
    if classified.reason == FailoverReason.billing:
        _billing_unverified = classified.billing_unverified
        _final_response = _billing_terminal_label(
            _final_summary, _billing_unverified
        )
        if _billing_guidance:
            _final_response += f"\n\n{_billing_guidance}"
        # Structured recovery descriptor so every surface renders
        # the same link + label from one signal (see helper).
        _billing_block = _billing_block_dict(
            provider, base_url, model, _billing_guidance,
            unverified=_billing_unverified,
        )
    else:
        _final_response = f"API call failed after {max_retries} retries: {_final_summary}"
    if _is_thinking_timeout:
        # Thinking-timeout guidance overrides stream-drop guidance,
        # which would wrongly suggest splitting large file writes.
        from agent.thinking_timeout_guidance import (
            build_thinking_timeout_guidance,
        )
        _final_response += build_thinking_timeout_guidance(
            provider=provider,
            model=model,
        )
    elif _is_stream_drop:
        _final_response += (
            "\n\nThe provider's stream connection keeps "
            "dropping — this often happens when generating "
            "very large tool call responses (e.g. write_file "
            "with long content). Try asking me to use "
            "execute_code with Python's open() for large "
            "files, or to write in smaller sections."
        )
    return {
        "final_response": _final_response,
        "messages": messages,
        "api_calls": api_call_count,
        "completed": False,
        "failed": True,
        "error": _final_summary,
        # Expose the classified reason so callers (kanban worker in
        # cli.py) can tell a quota wall (``rate_limit`` / ``billing``)
        # from a task failure.
        "failure_reason": classified.reason.value,
        # The classifier's own retry verdict — UI surfaces use
        # this instead of re-deriving from the reason string.
        "failure_retryable": bool(classified.retryable),
        # True when the billing verdict rests on an ambiguous
        # body (#82154) — may be a content-filter rejection.
        "billing_unverified": _billing_unverified,
        # Present only for billing walls: structured recovery
        # descriptor (provider, billing_url, is_nous, message).
        "billing_block": _billing_block,
    }


def log_api_error_attempt(
    agent: Any,
    api_error: Exception,
    *,
    retry_count: int,
    max_retries: int,
    status_code: Optional[int],
    elapsed_time: float,
    api_messages: Any,
    approx_tokens: int,
) -> Tuple[str, str, Any, Any, Any]:
    """Log one failed API attempt: the ``API call failed`` warning plus the buffered
    retry trace (provider/endpoint/error/4xx body/elapsed), the OpenRouter
    "no tool endpoints" hint and the bare-404 missing-vendor-prefix hint (#78796).
    The buffer only surfaces if every retry+fallback exhausts.

    Returns ``(error_type, error_msg, provider, base_url, model)`` — the loop's
    ``error_type`` / ``error_msg`` / ``_provider`` / ``_base`` / ``_model`` locals."""
    error_type = type(api_error).__name__
    error_msg = str(api_error).lower()
    _error_summary = agent._summarize_api_error(api_error)
    logger.warning(
        "API call failed (attempt %s/%s) error_type=%s %s summary=%s",
        retry_count,
        max_retries,
        error_type,
        agent._client_log_context(),
        _error_summary,
    )

    _provider = getattr(agent, "provider", "unknown")
    _base = getattr(agent, "base_url", "unknown")
    _model = getattr(agent, "model", "unknown")
    _status_code_str = f" [HTTP {status_code}]" if status_code else ""
    agent._buffer_vprint(f"⚠️  API call failed (attempt {retry_count}/{max_retries}): {error_type}{_status_code_str}")
    agent._buffer_vprint(f"   🔌 Provider: {_provider}  Model: {_model}")
    agent._buffer_vprint(f"   🌐 Endpoint: {_base}")
    agent._buffer_vprint(f"   📝 Error: {_error_summary}")
    if status_code and status_code < 500:
        _err_body = getattr(api_error, "body", None)
        _err_body_str = str(_err_body)[:300] if _err_body else None
        if _err_body_str:
            agent._buffer_vprint(f"   📋 Details: {_err_body_str}")
    agent._buffer_vprint(f"   ⏱️  Elapsed: {elapsed_time:.2f}s  Context: {len(api_messages)} msgs, ~{approx_tokens:,} tokens")

    # OpenRouter "no tool endpoints" hint, buffered with the retry trace
    # so it only surfaces if every retry+fallback exhausts.
    if (
        agent._is_openrouter_url()
        and "support tool use" in error_msg
    ):
        agent._buffer_vprint(
            f"   💡 No OpenRouter providers for {_model} support tool calling with your current settings."
        )
        if agent.providers_allowed:
            agent._buffer_vprint(
                "      Your provider_routing.only restriction is filtering out tool-capable providers."
            )
            agent._buffer_vprint(
                "      Try removing the restriction or adding providers that support tools for this model."
            )
        agent._buffer_vprint(
            f"      Check which providers support tools: https://openrouter.ai/models/{_model}"
        )

    # Bare 404 on a ``vendor/model`` catalogue usually means the id lost its
    # prefix; the provider never names the model, so we do (#78796).
    if getattr(api_error, "status_code", None) == 404:
        try:
            from hermes_cli.model_normalize import suggest_prefixed_model_id

            _suggestion = suggest_prefixed_model_id(_provider, _model)
        except Exception:
            _suggestion = None
        if _suggestion:
            agent._buffer_vprint(
                f"   💡 Model '{_model}' is not a valid id for provider {_provider} — "
                f"it is missing its vendor prefix."
            )
            agent._buffer_vprint(
                f"      Did you mean '{_suggestion}'?  Re-pick it with `hermes model`."
            )
    return error_type, error_msg, _provider, _base, _model


def interruptible_backoff_sleep(
    agent: Any,
    wait_time: float,
    _retry: Optional[TurnRetryState],
    *,
    messages: List[Dict[str, Any]],
    conversation_history: Any,
    api_call_count: int,
    abort_message: str,
    interrupt_text: str,
    activity_label: str,
) -> Optional[Dict[str, Any]]:
    """Sleep ``wait_time`` in 200 ms slices so interrupts are honoured promptly, touching
    activity every ~30 s so the gateway's inactivity monitor knows we are alive.

    On interrupt: when ``_retry`` is given and a redirect is pending, preserve it
    (``clear_interrupt(preserve_redirect=True)``), set
    ``_retry.restart_with_redirected_messages`` and return ``None`` — the caller
    rebuilds the turn from the correction. Otherwise close any open tool sequence,
    persist, clear the interrupt and return the ``interrupted`` result dict.
    Returns ``None`` when the wait completed."""
    sleep_end = time.time() + wait_time
    _touch_counter = 0
    while time.time() < sleep_end:
        if agent._interrupt_requested:
            if _retry is not None and agent.clear_interrupt(preserve_redirect=True):
                _retry.restart_with_redirected_messages = True
                return None
            agent._vprint(f"{agent.log_prefix}⚡ {abort_message}", force=True)
            close_interrupted_tool_sequence(messages, interrupt_text)
            agent._persist_session(messages, conversation_history)
            agent.clear_interrupt()
            return {
                "final_response": interrupt_text,
                "messages": messages,
                "api_calls": api_call_count,
                "completed": False,
                "interrupted": True,
            }
        time.sleep(0.2)
        _touch_counter += 1
        if _touch_counter % 150 == 0:  # 150 × 0.2s = 30s
            agent._touch_activity(
                f"{activity_label}, {int(sleep_end - time.time())}s remaining"
            )
    return None


def compute_error_backoff(
    agent: Any,
    api_error: Exception,
    *,
    retry_count: int,
    max_retries: int,
    is_rate_limited: bool,
    is_zai_coding_overload: bool,
    base_url: Any,
    model: Any,
) -> float:
    """Pick the wait before the next API retry and announce it.

    Retry-After header wins for rate limits (capped at 600s: Anthropic Tier 1 buckets
    reset in ~171s, so a 120s cap retried early and re-tripped the limit, #26293);
    otherwise jittered exponential backoff, replaced by the adaptive rate-limit policy
    for 429s / Z.AI overloads. Normal retries are buffered to avoid chatter; long Z.AI
    Coding waits can last minutes, so those surface immediately."""
    # Resolved through the loop module so tests that patch
    # ``agent.conversation_loop.jittered_backoff`` / ``adaptive_rate_limit_backoff``
    # (incl. the run_agent conftest fast-backoff fixture) keep intercepting.
    from agent.conversation_loop import adaptive_rate_limit_backoff, jittered_backoff

    _retry_after = None
    if is_rate_limited:
        _resp_headers = getattr(getattr(api_error, "response", None), "headers", None)
        if _resp_headers and hasattr(_resp_headers, "get"):
            _ra_raw = _resp_headers.get("retry-after") or _resp_headers.get("Retry-After")
            if _ra_raw:
                try:
                    _retry_after = min(float(_ra_raw), 600)
                except (TypeError, ValueError):
                    pass
    wait_time = _retry_after if _retry_after else jittered_backoff(retry_count, base_delay=2.0, max_delay=60.0)
    _backoff_policy = None
    if (is_rate_limited or is_zai_coding_overload) and not _retry_after:
        wait_time, _backoff_policy = adaptive_rate_limit_backoff(
            retry_count,
            base_url=str(base_url),
            model=model,
            error=api_error,
            default_wait=wait_time,
        )
    if is_rate_limited or is_zai_coding_overload:
        _policy_note = ""
        if _backoff_policy == "zai_coding_overload_long":
            _policy_note = " (Z.AI Coding overload adaptive long backoff)"
        elif _backoff_policy == "zai_coding_overload_short":
            _policy_note = " (Z.AI Coding overload short retry)"
        _wait_reason = "Provider overloaded" if is_zai_coding_overload and not is_rate_limited else "Rate limited"
        _rate_limit_status = f"⏱️ {_wait_reason}. Waiting {wait_time:.1f}s (attempt {retry_count + 1}/{max_retries}){_policy_note}..."
        if _backoff_policy == "zai_coding_overload_long":
            agent._emit_status(_rate_limit_status)
        else:
            agent._buffer_status(_rate_limit_status)
    else:
        agent._buffer_status(f"⏳ Retrying in {wait_time:.1f}s (attempt {retry_count}/{max_retries})...")
    logger.warning(
        "Retrying API call in %ss (attempt %s/%s) %s policy=%s error=%s",
        wait_time,
        retry_count,
        max_retries,
        agent._client_log_context(),
        _backoff_policy or "default",
        api_error,
    )
    return wait_time
