"""Multi-provider authentication system for Hermes Agent.

Architecture: - ProviderConfig registry defines known OAuth providers - Auth store (auth.json) holds
per-provider credential state - resolve_provider() picks the active provider via priority chain -
resolve_*_runtime_credentials() handles token refresh and runtime keys - logout_command() is the CLI
entry point for clearing auth
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import shlex
import stat
import threading
import time
import uuid
import webbrowser  # noqa: F401  (tests patch auth_mod.webbrowser.open; same module object)

from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, FrozenSet, Iterable, List, Optional, Tuple
from urllib.parse import urlparse

from hermes_cli.config import (
    get_hermes_home,
    get_config_path,
    read_raw_config,
    require_readable_config_before_write,
)
from hermes_constants import OPENROUTER_BASE_URL, secure_parent_dir
from agent.credential_persistence import sanitize_borrowed_credential_payload
from utils import atomic_replace, atomic_yaml_write, env_float, is_truthy_value  # noqa: F401  (env_float: agent.credential_pool reads auth_mod.env_float)
from hermes_cli.auth_zai_kimi import (  # noqa: F401  (re-exported; callers/tests use hermes_cli.auth.<name>)
    KIMI_CODE_BASE_URL,
    ZAI_ENDPOINTS,
    _normalize_lmstudio_runtime_base_url,
    _probe_single_zai_endpoint,
    _resolve_kimi_base_url,
    _resolve_zai_base_url,
    detect_zai_endpoint,
)
from hermes_cli.auth_model_picker import (  # noqa: F401  (re-exported; callers/tests use hermes_cli.auth.<name>)
    _ModelPickerRows,
    _confirm_selection_guards,
    _prompt_model_selection,
    _save_model_choice,
)
from hermes_cli.auth_device_flow import (  # noqa: F401  (re-exported; callers/tests use hermes_cli.auth.<name>)
    _CONSOLE_BROWSER_NAMES,
    _can_open_graphical_browser,
    _default_verify,
    _is_remote_session,
    _nous_device_auth_timeout_message,
    _offer_existing_oauth_credentials,
    _poll_device_token_generic,
    _poll_for_token,
    _print_device_code_instructions,
    _print_login_success,
    _print_loopback_ssh_hint,
    _prompt_yes_no,
    _request_device_code,
    _resolve_verify,
    _ssh_user_at_host,
)
from hermes_cli.auth_oauth_grants import (  # noqa: F401  (re-exported; callers/tests use hermes_cli.auth.<name>)
    SINGLE_USE_OAUTH_SINGLETON_FILES,
    SINGLE_USE_REFRESH_POOL_PROVIDERS,
    _OAUTH_TOKEN_FIELDS,
    _adopt_oauth_material,
    _find_root_counterpart,
    _heal_forked_provider_block,
    _heal_forked_single_use_oauth_grants,
    _is_oauth_pool_payload,
    _oauth_freshness,
    _oauth_heal_clean_marks,
    _oauth_heal_notices,
    _oauth_identity,
    _singleton_as_row,
    consume_oauth_heal_notices,
    heal_forked_single_use_oauth_grants,
    strip_cloned_single_use_oauth_grants,
)
from hermes_cli.auth_nous import (  # noqa: F401  (re-exported; callers/tests use hermes_cli.auth.<name>)
    NOUS_SESSION_TERMINAL,
    NOUS_SESSION_UNKNOWN,
    NOUS_SESSION_VALID,
    NOUS_SHARED_STORE_FILENAME,
    _ALLOWED_NOUS_INFERENCE_HOSTS,
    _NOUS_EFFECTIVE_STATE_IGNORED_KEYS,
    _NOUS_EMPTY_AGENT_KEY_FIELDS,
    _NOUS_SHARED_STATE_KEYS,
    _NOUS_STALE_PORTAL_HOSTS,
    _NousStatePersister,
    _OAUTH_GRANT_DEAD_CODES,
    _TERMINAL_REFRESH_ERROR_CODES,
    _agent_key_is_usable,
    _apply_nous_refreshed_tokens,
    _assert_nous_inference_jwt_usable,
    _clear_shared_nous_state,
    _compute_nous_auth_status,
    _empty_nous_auth_status,
    _format_nous_entitlement_auth_error,
    _healed_nous_inference_url,
    _is_terminal_codex_oauth_refresh_error,
    _is_terminal_nous_refresh_error,
    _is_terminal_refresh_error,
    _is_terminal_xai_oauth_refresh_error,
    _iso_after,
    _log_nous_invoke_jwt_selected,
    _login_nous,
    _merge_shared_nous_oauth_state,
    _migrate_stale_nous_portal_url,
    _mirror_nous_state_best_effort,
    _nous_device_code_login,
    _nous_effective_provider_state,
    _nous_effective_routing,
    _nous_inference_env_override,
    _nous_invoke_jwt_is_usable,
    _nous_invoke_jwt_status,
    _nous_jwt_expires_at,
    _nous_portal_env_override,
    _nous_shared_auth_dir,
    _nous_shared_lock_holder,
    _nous_shared_shape,
    _nous_shared_store_lock,
    _nous_shared_store_path,
    _nous_status_from_state,
    _oauth_trace,
    _oauth_trace_enabled,
    _offer_shared_nous_import,
    _pick_nous_model_after_login,
    _pool_first_oauth_status,
    _quarantine_nous_oauth_state,
    _quarantine_nous_pool_entries,
    _read_shared_nous_state,
    _refresh_access_token,
    _refresh_nous_or_quarantine,
    _scope_values,
    _select_nous_invoke_jwt,
    _set_nous_agent_key_from_invoke_jwt,
    _snapshot_nous_pool_status,
    _sync_nous_pool_from_auth_store,
    _token_fingerprint,
    _try_import_shared_nous_state,
    _validate_nous_inference_url_from_network,
    _write_shared_nous_state,
    fetch_nous_models,
    get_nous_auth_status_local,
    get_nous_session_validity,
    persist_nous_credentials,
    refresh_nous_oauth_from_state,
    refresh_nous_oauth_pure,
    resolve_nous_runtime_credentials,
    step_up_nous_billing_scope,
)
from hermes_cli.auth_minimax import (  # noqa: F401  (re-exported; callers/tests use hermes_cli.auth.<name>)
    _MINIMAX_OAUTH_ERROR_BODY_LIMIT,
    _login_minimax_oauth,
    _minimax_expired_in_looks_like_unix_ms,
    _minimax_expiry_fields,
    _minimax_fresh_state,
    _minimax_oauth_login,
    _minimax_oauth_quarantine_on_terminal_refresh,
    _minimax_pkce_pair,
    _minimax_poll_token,
    _minimax_post_form,
    _minimax_request_user_code,
    _minimax_resolve_token_expiry_unix,
    _minimax_response_error_text,
    _minimax_save_auth_state,
    _refresh_minimax_oauth_state,
    build_minimax_oauth_token_provider,
    resolve_minimax_oauth_runtime_credentials,
)
from hermes_cli.auth_xai import (  # noqa: F401  (re-exported; callers/tests use hermes_cli.auth.<name>)
    _is_xai_origin_host,
    _login_xai_oauth,
    _quarantine_xai_oauth_tokens,
    _read_xai_oauth_tokens,
    _refresh_xai_oauth_tokens,
    _save_xai_oauth_tokens,
    _write_through_xai_oauth_to_global_root,
    _xai_access_token_is_expiring,
    _xai_oauth_device_code_login,
    _xai_oauth_discovery,
    _xai_oauth_inference_base_url,
    _xai_oauth_poll_device_token,
    _xai_oauth_request_device_code,
    _xai_oauth_state_from_store,
    _xai_oauth_state_has_usable_tokens,
    _xai_proactive_refresh_skew_seconds,
    _xai_tokens_from_payload,
    _xai_validate_inference_base_url,
    _xai_validate_oauth_endpoint,
    refresh_xai_oauth_pure,
    resolve_xai_oauth_runtime_credentials,
)
from hermes_cli.auth_codex import (  # noqa: F401  (re-exported; callers/tests use hermes_cli.auth.<name>)
    CODEX_QUOTA_PROBE_MIN_INTERVAL_SECONDS,
    _clear_pool_entry_status,
    _codex_access_token_is_expiring,
    _codex_base_url,
    _codex_device_code_login,
    _codex_exchange_authorization_code,
    _codex_http_client,
    _codex_login_rate_limited_error,
    _codex_poll_authorization_code,
    _codex_pool_rate_limit_status,
    _codex_quota_exhausted_error,
    _codex_quota_probe_cache,
    _codex_quota_probe_lock,
    _codex_refresh_failure_error,
    _codex_request_device_code,
    _codex_runtime_result,
    _codex_usage_probe_url,
    _import_codex_cli_tokens,
    _is_codex_rate_limit_shaped,
    _load_auth_store_maybe_locked,
    _login_openai_codex,
    _parse_retry_after_seconds,
    _pool_codex_access_token,
    _pool_entries,
    _probe_codex_quota_restored,
    _read_codex_tokens,
    _recover_codex_tokens_from_cli,
    _refresh_codex_auth_tokens,
    _refresh_payload_access_token,
    _save_codex_tokens,
    _sync_codex_pool_entries,
    clear_codex_pool_quota_cooldowns,
    refresh_codex_oauth_pure,
    resolve_codex_runtime_credentials,
)
from hermes_cli.auth_spotify import (  # noqa: F401  (re-exported; callers/tests use hermes_cli.auth.<name>)
    _make_spotify_callback_handler,
    _refresh_spotify_oauth_state,
    _spotify_accounts_base_url,
    _spotify_api_base_url,
    _spotify_build_authorize_url,
    _spotify_client_id,
    _spotify_code_challenge,
    _spotify_code_verifier,
    _spotify_exchange_code_for_tokens,
    _spotify_interactive_setup,
    _spotify_redirect_uri,
    _spotify_scope_list,
    _spotify_scope_string,
    _spotify_setting,
    _spotify_token_payload_to_state,
    _spotify_token_post,
    _spotify_validate_redirect_uri,
    _spotify_wait_for_callback,
    get_spotify_auth_status,
    login_spotify_command,
    resolve_spotify_runtime_credentials,
)
from hermes_cli.auth_qwen import (  # noqa: F401  (re-exported; callers/tests use hermes_cli.auth.<name>)
    _mark_qwen_oauth_active,
    _qwen_access_token_is_expiring,
    _qwen_cli_auth_path,
    _read_qwen_cli_tokens,
    _refresh_qwen_cli_tokens,
    _save_qwen_cli_tokens,
    get_qwen_auth_status,
    resolve_qwen_runtime_credentials,
)
from hermes_cli.auth_constants import (  # noqa: F401  (re-exported; callers/tests use hermes_cli.auth.<name>)
    _decode_jwt_claims,
    AUTH_STORE_VERSION,
    AUTH_LOCK_TIMEOUT_SECONDS,
    DEFAULT_NOUS_PORTAL_URL,
    DEFAULT_NOUS_INFERENCE_URL,
    DEFAULT_NOUS_CLIENT_ID,
    NOUS_INFERENCE_INVOKE_SCOPE,
    NOUS_BILLING_MANAGE_SCOPE,
    DEFAULT_NOUS_SCOPE,
    NOUS_DEVICE_CODE_SOURCE,
    NOUS_AUTH_PATH_INVOKE_JWT,
    ACCESS_TOKEN_REFRESH_SKEW_SECONDS,
    NOUS_INVOKE_JWT_MIN_TTL_SECONDS,
    DEVICE_AUTH_POLL_INTERVAL_CAP_SECONDS,
    DEVICE_CODE_GRANT_TYPE,
    _FORM_JSON_HEADERS,
    DEFAULT_CODEX_BASE_URL,
    DEFAULT_XAI_OAUTH_BASE_URL,
    MINIMAX_OAUTH_CLIENT_ID,
    MINIMAX_OAUTH_SCOPE,
    MINIMAX_OAUTH_GRANT_TYPE,
    MINIMAX_OAUTH_GLOBAL_BASE,
    MINIMAX_OAUTH_CN_BASE,
    MINIMAX_OAUTH_GLOBAL_INFERENCE,
    MINIMAX_OAUTH_CN_INFERENCE,
    MINIMAX_OAUTH_REFRESH_SKEW_SECONDS,
    DEFAULT_QWEN_BASE_URL,
    DEFAULT_GITHUB_MODELS_BASE_URL,
    DEFAULT_COPILOT_ACP_BASE_URL,
    DEFAULT_OLLAMA_CLOUD_BASE_URL,
    DEFAULT_ACTUAL_BASE_URL,
    DEFAULT_ACTUAL_LOCAL_BASE_URL,
    STEPFUN_STEP_PLAN_INTL_BASE_URL,
    STEPFUN_STEP_PLAN_CN_BASE_URL,
    CODEX_OAUTH_CLIENT_ID,
    CODEX_OAUTH_TOKEN_URL,
    _HERMES_CLI_VERSION,
    CODEX_OAUTH_USER_AGENT,
    CODEX_ACCESS_TOKEN_REFRESH_SKEW_SECONDS,
    XAI_OAUTH_ISSUER,
    XAI_OAUTH_DISCOVERY_URL,
    XAI_OAUTH_CLIENT_ID,
    XAI_OAUTH_SCOPE,
    XAI_OAUTH_DEVICE_CODE_URL,
    XAI_ACCESS_TOKEN_REFRESH_SKEW_SECONDS,
    QWEN_OAUTH_CLIENT_ID,
    QWEN_OAUTH_TOKEN_URL,
    QWEN_ACCESS_TOKEN_REFRESH_SKEW_SECONDS,
    DEFAULT_SPOTIFY_ACCOUNTS_BASE_URL,
    DEFAULT_SPOTIFY_API_BASE_URL,
    DEFAULT_SPOTIFY_REDIRECT_URI,
    SPOTIFY_DOCS_URL,
    SPOTIFY_DASHBOARD_URL,
    SPOTIFY_ACCESS_TOKEN_REFRESH_SKEW_SECONDS,
    OAUTH_OVER_SSH_DOCS_URL,
    DEFAULT_SPOTIFY_SCOPE,
    SERVICE_PROVIDER_NAMES,
    LMSTUDIO_NOAUTH_PLACEHOLDER,
    ACTUAL_LOCAL_NOAUTH_PLACEHOLDER,
    CODEX_RATE_LIMITED_CODE,
    AuthError,
    _provider_error_factory,
    _nous_err,
    _xai_err,
    _codex_err,
    _spotify_err,
    _qwen_err,
    _minimax_err,
    httpx,
)

logger = logging.getLogger(__name__)

try:
    import fcntl
except Exception:
    fcntl = None
try:
    import msvcrt
except Exception:
    msvcrt = None

def is_actual_local_base_url(base_url: str) -> bool:
    """Return True for Actual's loopback local API endpoint."""
    try:
        host = (urlparse(base_url or "").hostname or "").lower().rstrip(".")
    except Exception:
        return False
    return host in {"localhost", "127.0.0.1", "::1", "0.0.0.0"}


def normalize_actual_base_url(base_url: str) -> str:
    """Return Actual's OpenAI-compatible base URL.

    Hosted inference lives at api.actual.inc; the Actual client's offline local server binds a
    loopback host. Both expose a /v1 surface for the Responses transport.
    """
    url = str(base_url or "").strip().rstrip("/")
    if not url:
        return DEFAULT_ACTUAL_BASE_URL
    try:
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower().rstrip(".")
        path = parsed.path.rstrip("/")
    except Exception:
        return url
    if host == "api.actual.inc" and path in {"", "/"}:
        return url + "/v1"
    if is_actual_local_base_url(url) and path in {"", "/"}:
        return url + "/v1"
    return url


# =============================================================================
# Provider Registry
# =============================================================================

@dataclass
class ProviderConfig:
    """Describes a known inference provider."""
    id: str
    name: str
    auth_type: str  # "oauth_device_code", "oauth_external", "oauth_minimax", or "api_key"
    portal_base_url: str = ""
    inference_base_url: str = ""
    client_id: str = ""
    scope: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)
    # For API-key providers: env vars to check (in priority order)
    api_key_env_vars: tuple = ()
    # Optional env var for base URL override
    base_url_env_var: str = ""


def _api_key_provider(
    id: str,
    name: str,
    inference_base_url: str,
    api_key_env_vars: tuple,
    base_url_env_var: str = "",
    *,
    auth_type: str = "api_key",
) -> ProviderConfig:
    """Compact constructor for the common env-var-keyed provider shape."""
    return ProviderConfig(
        id=id,
        name=name,
        auth_type=auth_type,
        inference_base_url=inference_base_url,
        api_key_env_vars=api_key_env_vars,
        base_url_env_var=base_url_env_var,
    )


PROVIDER_REGISTRY: Dict[str, ProviderConfig] = {
    "nous": ProviderConfig(
        id="nous",
        name="Nous Portal",
        auth_type="oauth_device_code",
        portal_base_url=DEFAULT_NOUS_PORTAL_URL,
        inference_base_url=DEFAULT_NOUS_INFERENCE_URL,
        client_id=DEFAULT_NOUS_CLIENT_ID,
        scope=DEFAULT_NOUS_SCOPE,
    ),
    "openai-codex": ProviderConfig(
        id="openai-codex",
        name="OpenAI Codex",
        auth_type="oauth_external",
        inference_base_url=DEFAULT_CODEX_BASE_URL,
    ),
    "openai-api": _api_key_provider(
        "openai-api", "OpenAI API", "https://api.openai.com/v1",
        ("OPENAI_API_KEY",), "OPENAI_BASE_URL",
    ),
    "xai-oauth": ProviderConfig(
        id="xai-oauth",
        name="xAI Grok OAuth (SuperGrok / Premium+)",
        auth_type="oauth_external",
        inference_base_url=DEFAULT_XAI_OAUTH_BASE_URL,
    ),
    "qwen-oauth": ProviderConfig(
        id="qwen-oauth",
        name="Qwen OAuth",
        auth_type="oauth_external",
        inference_base_url=DEFAULT_QWEN_BASE_URL,
    ),
    "lmstudio": _api_key_provider(
        "lmstudio", "LM Studio", "http://127.0.0.1:1234/v1",
        ("LM_API_KEY",), "LM_BASE_URL",
    ),
    "copilot": _api_key_provider(
        "copilot", "GitHub Copilot", DEFAULT_GITHUB_MODELS_BASE_URL,
        ("COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN"), "COPILOT_API_BASE_URL",
    ),
    "copilot-acp": ProviderConfig(
        id="copilot-acp",
        name="GitHub Copilot ACP",
        auth_type="external_process",
        inference_base_url=DEFAULT_COPILOT_ACP_BASE_URL,
        base_url_env_var="COPILOT_ACP_BASE_URL",
    ),
    "gemini": _api_key_provider(
        "gemini", "Google AI Studio", "https://generativelanguage.googleapis.com/v1beta",
        ("GOOGLE_API_KEY", "GEMINI_API_KEY"), "GEMINI_BASE_URL",
    ),
    "zai": _api_key_provider(
        "zai", "Z.AI / GLM", "https://api.z.ai/api/paas/v4",
        ("GLM_API_KEY", "ZAI_API_KEY", "Z_AI_API_KEY"), "GLM_BASE_URL",
    ),
    # Legacy platform.moonshot.ai keys use this endpoint (OpenAI-compat).
    # sk-kimi- (Kimi Code) keys are auto-redirected to api.kimi.com/coding
    # by _resolve_kimi_base_url() below.
    "kimi-coding": _api_key_provider(
        "kimi-coding", "Kimi / Moonshot", "https://api.moonshot.ai/v1",
        ("KIMI_API_KEY", "KIMI_CODING_API_KEY"), "KIMI_BASE_URL",
    ),
    "kimi-coding-cn": _api_key_provider(
        "kimi-coding-cn", "Kimi / Moonshot (China)", "https://api.moonshot.cn/v1",
        ("KIMI_CN_API_KEY",),
    ),
    "stepfun": _api_key_provider(
        "stepfun", "StepFun Step Plan", STEPFUN_STEP_PLAN_INTL_BASE_URL,
        ("STEPFUN_API_KEY",), "STEPFUN_BASE_URL",
    ),
    "arcee": _api_key_provider(
        "arcee", "Arcee AI", "https://api.arcee.ai/api/v1",
        ("ARCEEAI_API_KEY",), "ARCEE_BASE_URL",
    ),
    "gmi": _api_key_provider(
        "gmi", "GMI Cloud", "https://api.gmi-serving.com/v1",
        ("GMI_API_KEY",), "GMI_BASE_URL",
    ),
    "actual": _api_key_provider(
        "actual", "Actual Computer", DEFAULT_ACTUAL_BASE_URL,
        ("ACTUAL_API_KEY",), "ACTUAL_BASE_URL",
    ),
    "minimax": _api_key_provider(
        "minimax", "MiniMax", "https://api.minimax.io/anthropic",
        ("MINIMAX_API_KEY",), "MINIMAX_BASE_URL",
    ),
    "minimax-oauth": ProviderConfig(
        id="minimax-oauth",
        name="MiniMax (OAuth \u00b7 minimax.io)",
        auth_type="oauth_minimax",
        portal_base_url=MINIMAX_OAUTH_GLOBAL_BASE,
        inference_base_url=MINIMAX_OAUTH_GLOBAL_INFERENCE,
        client_id=MINIMAX_OAUTH_CLIENT_ID,
        scope=MINIMAX_OAUTH_SCOPE,
        extra={"region": "global", "cn_portal_base_url": MINIMAX_OAUTH_CN_BASE,
               "cn_inference_base_url": MINIMAX_OAUTH_CN_INFERENCE},
    ),
    # CLAUDE_CODE_OAUTH_TOKEN is NOT an API key, despite auth_type="api_key"
    # and its place in this tuple (#82154). `claude setup-token` yields an
    # `sk-ant-oat01…` OAuth token: sent as `x-api-key` it 401s, and sent as a
    # bare Bearer it 429s. It is listed here because this tuple doubles as the
    # credential-DISCOVERY list (agent/credential_pool.py builds its env scan
    # from it), so removing it would stop Hermes finding a setup-token
    # credential at all. The adapter routes such a value down the OAuth path
    # on the strength of its prefix, not on this entry. Only ANTHROPIC_API_KEY
    # and ANTHROPIC_TOKEN are usable as literal API keys.
    "anthropic": _api_key_provider(
        "anthropic", "Anthropic", "https://api.anthropic.com",
        ("ANTHROPIC_API_KEY", "ANTHROPIC_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN"), "ANTHROPIC_BASE_URL",
    ),
    "alibaba": _api_key_provider(
        "alibaba", "Qwen Cloud", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        ("DASHSCOPE_API_KEY",), "DASHSCOPE_BASE_URL",
    ),
    "alibaba-coding-plan": _api_key_provider(
        "alibaba-coding-plan", "Alibaba Cloud (Coding Plan)", "https://coding-intl.dashscope.aliyuncs.com/v1",
        ("ALIBABA_CODING_PLAN_API_KEY", "DASHSCOPE_API_KEY"), "ALIBABA_CODING_PLAN_BASE_URL",
    ),
    "minimax-cn": _api_key_provider(
        "minimax-cn", "MiniMax (China)", "https://api.minimaxi.com/anthropic",
        ("MINIMAX_CN_API_KEY",), "MINIMAX_CN_BASE_URL",
    ),
    "deepseek": _api_key_provider(
        "deepseek", "DeepSeek", "https://api.deepseek.com/v1",
        ("DEEPSEEK_API_KEY",), "DEEPSEEK_BASE_URL",
    ),
    "xai": _api_key_provider("xai", "xAI", "https://api.x.ai/v1", ("XAI_API_KEY",), "XAI_BASE_URL"),
    "nvidia": _api_key_provider(
        "nvidia", "NVIDIA NIM", "https://integrate.api.nvidia.com/v1",
        ("NVIDIA_API_KEY",), "NVIDIA_BASE_URL",
    ),
    "ai-gateway": _api_key_provider(
        "ai-gateway", "Vercel AI Gateway", "https://ai-gateway.vercel.sh/v1",
        ("AI_GATEWAY_API_KEY",), "AI_GATEWAY_BASE_URL",
    ),
    "opencode-zen": _api_key_provider(
        "opencode-zen", "OpenCode Zen", "https://opencode.ai/zen/v1",
        ("OPENCODE_ZEN_API_KEY",), "OPENCODE_ZEN_BASE_URL",
    ),
    # OpenCode Go mixes API surfaces by model:
    # - GLM / Kimi use OpenAI-compatible chat completions under /v1
    # - MiniMax models use Anthropic Messages under /v1/messages
    # - Qwen 3.7 uses Anthropic Messages under /v1/messages
    # Keep the provider base at /v1 and select api_mode per-model.
    "opencode-go": _api_key_provider(
        "opencode-go", "OpenCode Go", "https://opencode.ai/zen/go/v1",
        ("OPENCODE_GO_API_KEY",), "OPENCODE_GO_BASE_URL",
    ),
    # Deliberately NO api_key_env_vars: the free tier is served
    # anonymously (any unrecognized bearer is a 401), so there is no
    # secret to configure. Select via `hermes model` / `/model free`.
    "opencode-free": _api_key_provider("opencode-free", "OpenCode Free", "https://opencode.ai/zen/v1", ()),
    "kilocode": _api_key_provider(
        "kilocode", "Kilo Code", "https://api.kilo.ai/api/gateway",
        ("KILOCODE_API_KEY",), "KILOCODE_BASE_URL",
    ),
    "huggingface": _api_key_provider(
        "huggingface", "Hugging Face", "https://router.huggingface.co/v1",
        ("HF_TOKEN",), "HF_BASE_URL",
    ),
    "xiaomi": _api_key_provider(
        "xiaomi", "Xiaomi MiMo", "https://api.xiaomimimo.com/v1",
        ("XIAOMI_API_KEY",), "XIAOMI_BASE_URL",
    ),
    "tencent-tokenhub": _api_key_provider(
        "tencent-tokenhub", "Tencent TokenHub", "https://tokenhub.tencentmaas.com/v1",
        ("TOKENHUB_API_KEY",), "TOKENHUB_BASE_URL",
    ),
    "tencent-tokenplan": _api_key_provider(
        "tencent-tokenplan", "Tencent TokenPlan", "https://api.lkeap.cloud.tencent.com/plan/anthropic",
        ("TOKENPLAN_API_KEY",), "TOKENPLAN_BASE_URL",
    ),
    "ollama-cloud": _api_key_provider(
        "ollama-cloud", "Ollama Cloud", DEFAULT_OLLAMA_CLOUD_BASE_URL,
        ("OLLAMA_API_KEY",), "OLLAMA_BASE_URL",
    ),
    "bedrock": _api_key_provider(
        "bedrock", "AWS Bedrock", "https://bedrock-runtime.us-east-1.amazonaws.com",
        (), "BEDROCK_BASE_URL", auth_type="aws_sdk",
    ),
    # No static inference_base_url: Vertex's endpoint is computed per
    # request from project_id + region (agent/vertex_adapter.py's
    # build_vertex_base_url), not a fixed host like the other entries.
    "vertex": _api_key_provider("vertex", "Google Vertex AI", "", (), auth_type="vertex"),
    "azure-foundry": _api_key_provider(
        "azure-foundry", "Azure Foundry", "",
        ("AZURE_FOUNDRY_API_KEY",), "AZURE_FOUNDRY_BASE_URL",
    ),
}

# Auto-extend PROVIDER_REGISTRY with any api-key provider registered in
# providers/ that is not already declared above.  New providers only need a
# plugins/model-providers/<name>/ plugin — no edits to this file required.
try:
    from providers import list_providers as _list_providers_for_registry
    for _pp in _list_providers_for_registry():
        if _pp.name in PROVIDER_REGISTRY:
            continue
        if _pp.auth_type == "external_process":
            # An external-process provider (an ACP CLI driven over stdio) has no
            # API-key env vars to resolve — its credentials come from
            # resolve_external_process_provider_credentials(), keyed on this
            # auth_type. Registering it here is what lets a provider shipped
            # outside this tree pass resolve_provider()'s known-provider gate;
            # without it, `hermes -m <that provider>` dies with
            # "Unknown provider" before any client is ever built.
            PROVIDER_REGISTRY[_pp.name] = ProviderConfig(
                id=_pp.name,
                name=_pp.display_name or _pp.name,
                auth_type="external_process",
                inference_base_url=_pp.base_url,
            )
            for _alias in _pp.aliases:
                if _alias not in PROVIDER_REGISTRY:
                    PROVIDER_REGISTRY[_alias] = PROVIDER_REGISTRY[_pp.name]
            continue
        if _pp.auth_type != "api_key" or not _pp.env_vars:
            continue
        # Skip providers that need custom token resolution or are special-cased
        # in resolve_provider() (copilot/kimi/zai have bespoke token refresh;
        # openrouter/custom are aggregator/user-supplied and handled outside
        # the registry — adding them here breaks runtime_provider resolution
        # that relies on `openrouter not in PROVIDER_REGISTRY`).
        if _pp.name in {"copilot", "kimi-coding", "kimi-coding-cn", "zai", "openrouter", "custom"}:
            continue
        _api_key_vars = tuple(v for v in _pp.env_vars if not v.endswith("_BASE_URL") and not v.endswith("_URL"))
        _base_url_var = next((v for v in _pp.env_vars if v.endswith("_BASE_URL") or v.endswith("_URL")), None)
        PROVIDER_REGISTRY[_pp.name] = ProviderConfig(
            id=_pp.name,
            name=_pp.display_name or _pp.name,
            auth_type="api_key",
            inference_base_url=_pp.base_url,
            api_key_env_vars=_api_key_vars or _pp.env_vars,
            base_url_env_var=_base_url_var or "",
        )
        # Also register aliases so resolve_provider() resolves them
        for _alias in _pp.aliases:
            if _alias not in PROVIDER_REGISTRY:
                PROVIDER_REGISTRY[_alias] = PROVIDER_REGISTRY[_pp.name]
except Exception:
    pass


# =============================================================================
# Anthropic Key Helper
# =============================================================================

def get_anthropic_key() -> str:
    """Return the first usable Anthropic credential, or ``""``.

    Checks both the ``.env`` file and the process environment, preferring ``~/.hermes/.env`` so a
    deliberate key rotation isn't shadowed by a stale shell export (matches the api-key resolution
    path — see #20591). The order mirrors the ``PROVIDER_REGISTRY["anthropic"].api_key_env_vars``
    tuple:
    """
    from hermes_cli.config import get_env_value_prefer_dotenv

    for var in PROVIDER_REGISTRY["anthropic"].api_key_env_vars:
        value = get_env_value_prefer_dotenv(var) or ""
        if value:
            return value
    return ""


# =============================================================================
# Kimi Code Endpoint Detection
# =============================================================================


_PLACEHOLDER_SECRET_VALUES = {
    "*",
    "**",
    "***",
    "changeme",
    "your_api_key",
    "your_api_key_here",
    "your-api-key",
    "placeholder",
    "example",
    "dummy",
    "null",
    "none",
}


def has_usable_secret(value: Any, *, min_length: int = 4) -> bool:
    """Return True when a configured secret looks usable, not empty/placeholder."""
    if not isinstance(value, str):
        return False
    cleaned = value.strip()
    if len(cleaned) < min_length:
        return False
    return cleaned.lower() not in _PLACEHOLDER_SECRET_VALUES


# Known API-key prefixes per provider.  Only providers listed here get
# prefix validation; everyone else is fail-open (unknown formats pass).
# This exists so an obviously malformed key in .env (truncated paste, wrong
# provider's key in the wrong var, etc.) doesn't silently shadow a valid
# credential-pool entry and produce opaque 401s (#93593).
KNOWN_PROVIDER_KEY_PREFIXES: Dict[str, tuple] = {
    # All OpenRouter keys are issued as sk-or-... (currently sk-or-v1-).
    "openrouter": ("sk-or-",),
}


def _secret_matches_declared_prefix(provider_id: str, value: str) -> bool:
    """Return False only when the provider declares key prefixes and none match.

    Providers without a declared prefix always pass (fail-open): we never hard-reject unknown key
    formats, only skip values that provably don't belong to a provider whose key format we know.
    """
    prefixes = KNOWN_PROVIDER_KEY_PREFIXES.get(provider_id)
    if not prefixes:
        return True
    return any(value.startswith(p) for p in prefixes)


def _warn_malformed_secret(provider_id: str, source: str) -> None:
    prefixes = KNOWN_PROVIDER_KEY_PREFIXES.get(provider_id, ())
    logger.warning(
        "Ignoring %s for provider %r: value does not match the expected key "
        "prefix (%s). Falling back to the next credential source. Fix or "
        "remove the malformed key to silence this warning.",
        source,
        provider_id,
        " or ".join(prefixes),
    )


def _resolve_api_key_provider_secret(
    provider_id: str, pconfig: ProviderConfig
) -> tuple[str, str]:
    """Resolve an API-key provider's token and indicate where it came from."""
    if provider_id == "copilot":
        # Use the dedicated copilot auth module for proper token validation
        try:
            from hermes_cli.copilot_auth import resolve_copilot_token, get_copilot_api_token
            token, source = resolve_copilot_token()
            if token:
                api_token, _base_url = get_copilot_api_token(token)
                return api_token, source
        except ValueError as exc:
            logger.warning("Copilot token validation failed: %s", exc)
        except Exception:
            pass
        return "", ""

    from hermes_cli.config import get_env_value_prefer_dotenv
    for env_var in pconfig.api_key_env_vars:
        # Prefer ~/.hermes/.env over os.environ so a deliberate key rotation
        # in the user's .env file isn't shadowed by a stale shell export
        # inherited from a parent process (Codex CLI, test runners, etc.).
        val = (get_env_value_prefer_dotenv(env_var) or "").strip()
        if not has_usable_secret(val):
            continue
        if not _secret_matches_declared_prefix(provider_id, val):
            # A provably malformed key (declared prefix mismatch) must not
            # shadow a valid credential-pool entry (#93593). Warn and keep
            # looking instead of returning it.
            _warn_malformed_secret(provider_id, env_var)
            continue
        return val, env_var

    # Fallback: try credential pool (e.g. zai key stored via auth.json)
    try:
        from agent.credential_pool import load_pool
        pool = load_pool(provider_id)
        if pool and pool.has_credentials():
            # Prefer the pool's own selection (peek), but iterate the rest of
            # the entries too so one malformed entry doesn't block a valid one.
            candidates = []
            entry = pool.peek()
            if entry is not None:
                candidates.append(entry)
            try:
                for extra in pool.entries():
                    if extra is not None and all(extra is not c for c in candidates):
                        candidates.append(extra)
            except Exception:
                pass
            for entry in candidates:
                key = getattr(entry, "access_token", "") or getattr(entry, "runtime_api_key", "")
                key = str(key).strip()
                if not has_usable_secret(key):
                    continue
                if not _secret_matches_declared_prefix(provider_id, key):
                    _warn_malformed_secret(provider_id, f"credential_pool:{provider_id}")
                    continue
                return key, f"credential_pool:{provider_id}"
    except Exception:
        pass

    return "", ""


# =============================================================================
# Z.AI Endpoint Detection
# =============================================================================

# Z.AI has separate billing for general vs coding plans, and global vs China
# endpoints.  A key that works on one may return "Insufficient balance" on
# another.  We probe at setup time and store the working endpoint.
# Each entry lists candidate models to try in order — newer coding plan accounts
# may only have access to recent models (glm-5.1, glm-5v-turbo) while older
# ones still use glm-4.7.


# =============================================================================
# Error Types
# =============================================================================

def is_rate_limited_auth_error(error: Exception) -> bool:
    """True when an :class:`AuthError` represents upstream rate-limiting / quota

    These failures are transient and re-authenticating cannot fix them, so callers should show a
    "retry later" notice and prefer a fallback chain instead of suggesting ``hermes auth``.
    """
    return (
        isinstance(error, AuthError)
        and not error.relogin_required
        and error.code == CODEX_RATE_LIMITED_CODE
    )


def format_auth_error(error: Exception) -> str:
    """Map auth failures to concise user-facing guidance."""
    if not isinstance(error, AuthError):
        return str(error)

    # Rate-limit / quota errors are not credential problems — never append the
    # "re-authenticate" remediation, which would mislead the operator.
    if is_rate_limited_auth_error(error):
        return str(error)

    if error.relogin_required:
        return f"{error} Run `hermes model` to re-authenticate."

    if error.code in _ENTITLEMENT_ERROR_CODES:
        if error.provider == "nous":
            return _format_nous_entitlement_auth_error(error)
        generic = _GENERIC_ENTITLEMENT_MESSAGES.get(error.code)
        if generic:
            return generic

    if error.code == "temporarily_unavailable":
        return f"{error} Please retry in a few seconds."

    return str(error)


# Entitlement failures: Nous gets a Portal-aware message; other providers a fixed
# generic one (or the raw error when no generic text exists for the code).
_GENERIC_ENTITLEMENT_MESSAGES = {
    "subscription_required": "No active paid subscription found. Please purchase/activate a subscription, then retry.",
    "insufficient_credits": "Subscription credits are exhausted. Top up/renew credits, then retry.",
}
_ENTITLEMENT_ERROR_CODES = frozenset(_GENERIC_ENTITLEMENT_MESSAGES) | {
    "subscription_expired", "no_usable_credits", "account_missing", "member_spend_cap_exceeded",
}


def _nonempty_str(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


# =============================================================================
# Auth Store — persistence layer for ~/.hermes/auth.json
# =============================================================================

def _auth_file_path() -> Path:
    path = get_hermes_home() / "auth.json"
    # Seat belt: if pytest is running and HERMES_HOME resolves to the real
    # user's auth store, refuse rather than silently corrupt it. This catches
    # tests that forgot to monkeypatch HERMES_HOME, tests invoked without the
    # hermetic conftest, or sandbox escapes via threads/subprocesses. In
    # production (no PYTEST_CURRENT_TEST) this is a single dict lookup.
    if os.environ.get("PYTEST_CURRENT_TEST"):
        real_home_auth = (Path.home() / ".hermes" / "auth.json").resolve(strict=False)
        try:
            resolved = path.resolve(strict=False)
        except Exception:
            resolved = path
        if resolved == real_home_auth:
            raise RuntimeError(
                f"Refusing to touch real user auth store during test run: {path}. "
                "Set HERMES_HOME to a tmp_path in your test fixture, or run "
                "via scripts/run_tests.sh for hermetic CI-parity env."
            )
    return path


def _global_auth_file_path() -> Optional[Path]:
    """Return the global-root auth.json when the process is in profile mode.

    Returns ``None`` when the profile and global root resolve to the same directory (classic mode,
    or custom HERMES_HOME that is not a profile). Used by read-only fallback paths so providers
    authed at the root are visible to profile processes that haven't configured them locally.
    """
    try:
        from hermes_constants import get_default_hermes_root
        global_root = get_default_hermes_root()
    except Exception:
        return None
    profile_home = get_hermes_home()
    try:
        if profile_home.resolve(strict=False) == global_root.resolve(strict=False):
            return None
    except Exception:
        if profile_home == global_root:
            return None
    # No pytest seat belt here: this is a pure read-only path, and
    # ``_load_global_auth_store()`` wraps the read in a try/except so an
    # unreadable global file can never break the profile process.  The
    # write-side seat belt still lives on ``_auth_file_path()`` where it
    # belongs (that's what protects the real user's auth store from being
    # corrupted by a mis-configured test).
    return global_root / "auth.json"


def _load_global_auth_store() -> Dict[str, Any]:
    """Load the global-root auth store (read-only fallback).

    Returns an empty dict when no global fallback exists (classic mode, or the global auth.json is
    absent). Never raises on missing file.
    """
    global _global_auth_store_cache
    global_path = _global_auth_file_path()
    if global_path is None or not global_path.exists():
        _global_auth_store_cache = None
        return {}
    try:
        resolved_path = str(global_path.resolve(strict=False))
        mtime_ns = global_path.stat().st_mtime_ns
        cache_key: Optional[Tuple[str, int]] = (resolved_path, mtime_ns)
    except Exception:
        cache_key = None
    if cache_key is not None and _global_auth_store_cache is not None:
        cached_path, cached_mtime, cached_store = _global_auth_store_cache
        if cached_path == cache_key[0] and cached_mtime == cache_key[1]:
            return cached_store
    if os.environ.get("PYTEST_CURRENT_TEST"):
        real_home_env = os.environ.get("HOME", "")
        if real_home_env:
            real_root = Path(real_home_env) / ".hermes" / "auth.json"
            try:
                if global_path.resolve(strict=False) == real_root.resolve(strict=False):
                    _global_auth_store_cache = None
                    return {}
            except Exception:
                pass
    try:
        store = _load_auth_store(global_path)
    except Exception:
        # A malformed global store must not break profile reads. The
        # profile's own auth store is still authoritative.
        _global_auth_store_cache = None
        return {}
    if cache_key is not None:
        _global_auth_store_cache = (cache_key[0], cache_key[1], store)
    return store


def _auth_lock_path() -> Path:
    return _auth_file_path().with_suffix(".lock")


_auth_target_lock_holders: Dict[str, threading.local] = {}
_auth_target_lock_holders_guard = threading.Lock()


def _same_path(left: Path, right: Path) -> bool:
    try:
        return left.resolve(strict=False) == right.resolve(strict=False)
    except Exception:
        return left == right


def _auth_lock_holder_for(target_path: Path) -> threading.local:
    """Return a reentrancy tracker keyed to one canonical auth-store path."""
    try:
        key = str(target_path.resolve(strict=False))
    except Exception:
        key = str(target_path)
    with _auth_target_lock_holders_guard:
        return _auth_target_lock_holders.setdefault(key, threading.local())


@contextmanager
def _file_lock(
    lock_path: Path,
    holder: threading.local,
    timeout_seconds: float,
    timeout_message: str,
):
    """Cross-process advisory flock helper.

    Reentrant per-thread via ``holder.depth``. Falls back to a depth-only guard when neither
    ``fcntl`` nor ``msvcrt`` is available (rare). Callers supply their own ``threading.local`` so
    independent locks (e.g.
    """
    if getattr(holder, "depth", 0) > 0:
        holder.depth += 1
        try:
            yield
        finally:
            holder.depth -= 1
        return

    lock_path.parent.mkdir(parents=True, exist_ok=True)

    if fcntl is None and msvcrt is None:
        holder.depth = 1
        try:
            yield
        finally:
            holder.depth = 0
        return

    # On Windows, msvcrt.locking needs the file to have content and the
    # file pointer at position 0. Ensure the lock file has at least 1 byte.
    # Under real concurrency (many threads/processes racing this same
    # ensure-content check) this write can collide with another holder's
    # msvcrt byte-range lock on the same file and raise PermissionError --
    # uncaught, since it happens before the retry loop below even starts.
    # A stress test with 20 concurrent Hermes processes reproduced this
    # deterministically on Windows. It's a best-effort convenience write
    # (whoever gets there first wins); losing the race here just means the
    # lock file already has content, so swallow the failure and proceed
    # straight to the acquire-with-retry loop.
    if msvcrt and (not lock_path.exists() or lock_path.stat().st_size == 0):
        try:
            lock_path.write_text(" ", encoding="utf-8")
        except (OSError, PermissionError):
            pass

    with lock_path.open("r+" if msvcrt else "a+", encoding="utf-8") as lock_file:
        deadline = time.monotonic() + max(1.0, timeout_seconds)
        while True:
            try:
                if fcntl:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                else:
                    lock_file.seek(0)
                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
                break
            except (BlockingIOError, OSError, PermissionError):
                if time.monotonic() >= deadline:
                    raise TimeoutError(timeout_message)
                time.sleep(0.05)

        holder.depth = 1
        try:
            yield
        finally:
            holder.depth = 0
            if fcntl:
                try:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                except (OSError, IOError):
                    pass
            elif msvcrt:
                try:
                    lock_file.seek(0)
                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
                except (OSError, IOError):
                    pass


@contextmanager
def _auth_store_lock(
    timeout_seconds: float = AUTH_LOCK_TIMEOUT_SECONDS,
    *,
    target_path: Optional[Path] = None,
):
    """Cross-process advisory lock for one auth.json read/write transaction.

    ``target_path`` is required for profile-to-global write-throughs. A profile lock does not
    protect the distinct global auth store; each path therefore uses its own reentrancy tracker and
    kernel lock.

    Lock ordering invariant: when this lock is held together with ``_nous_shared_store_lock``,
    acquire ``_auth_store_lock`` FIRST (outer) and the shared Nous lock SECOND (inner). All runtime
    refresh paths follow this order; violating it risks deadlock against a concurrent import on the
    shared store.
    """
    auth_path = target_path if target_path is not None else _auth_file_path()
    lock_path = auth_path.with_suffix(".lock") if target_path is not None else _auth_lock_path()
    with _file_lock(
        lock_path,
        _auth_lock_holder_for(auth_path),
        timeout_seconds,
        "Timed out waiting for auth store lock",
    ):
        yield


def _load_auth_store(auth_file: Optional[Path] = None) -> Dict[str, Any]:
    auth_file = auth_file or _auth_file_path()
    if not auth_file.exists():
        return {"version": AUTH_STORE_VERSION, "providers": {}}

    try:
        raw = json.loads(auth_file.read_text(encoding="utf-8-sig"))
    except OSError:
        # The file exists (checked above) but could not be READ: EMFILE under
        # fd exhaustion, EACCES, EIO, a stalled network mount. None of those
        # mean the contents are bad, and this module does read-modify-write in
        # ~15 places, so degrading to an empty store here is one
        # _save_auth_store() away from erasing every stored credential.
        # Fail loudly instead and leave the file on disk untouched.
        logger.warning(
            "auth: could not read %s, leaving the store on disk untouched "
            "rather than degrading to an empty one",
            auth_file, exc_info=True,
        )
        raise
    except Exception as exc:
        # Genuine corruption: unparseable JSON, or bytes that are not UTF-8.
        corrupt_path = auth_file.with_suffix(".json.corrupt")
        preserved = False
        try:
            import shutil
            shutil.copy2(auth_file, corrupt_path)
            preserved = True
        except Exception:
            logger.debug(
                "auth: could not preserve a copy of the corrupt store at %s",
                corrupt_path, exc_info=True,
            )
        if preserved:
            logger.warning(
                "auth: failed to parse %s (%s), starting with empty store. "
                "Corrupt file preserved at %s",
                auth_file, exc, corrupt_path,
            )
        else:
            # Do not advertise a backup that was never written.
            logger.warning(
                "auth: failed to parse %s (%s), starting with empty store. "
                "A copy could NOT be preserved at %s",
                auth_file, exc, corrupt_path,
            )
        return {"version": AUTH_STORE_VERSION, "providers": {}}

    if isinstance(raw, dict) and (
        isinstance(raw.get("providers"), dict)
        or isinstance(raw.get("credential_pool"), dict)
    ):
        raw.setdefault("providers", {})
        if isinstance(raw.get("providers"), dict):
            _migrate_stale_nous_portal_url(raw["providers"])
        return raw

    # Migrate from PR's "systems" format if present
    if isinstance(raw, dict) and isinstance(raw.get("systems"), dict):
        systems = raw["systems"]
        providers = {}
        if "nous_portal" in systems:
            providers["nous"] = systems["nous_portal"]
        return {"version": AUTH_STORE_VERSION, "providers": providers,
                "active_provider": "nous" if providers else None}

    return {"version": AUTH_STORE_VERSION, "providers": {}}


def _write_private_file_atomic(
    target: Path,
    payload: str,
    *,
    replace: Optional[Callable[[Any, Any], Any]] = None,
    fsync_dir: bool = False,
) -> None:
    """Write *payload* to *target* via a 0o600 temp file + atomic rename.

    Creating the temp with ``os.open(O_EXCL, 0o600)`` closes the TOCTOU window where
    ``write_text()`` + post-write ``chmod`` briefly exposed tokens at process umask (often 0o644).
    Mirrors agent/google_oauth.py (#19673) and tools/mcp_oauth.py (#21148). The per-process random
    temp suffix avoids collisions between concurrent writers and stale leftovers from a crashed
    prior write.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    # secure_parent_dir refuses to chmod /, top-level dirs, or the
    # hermes-agent install tree (#25821, #93050).
    secure_parent_dir(target)
    tmp_path = target.with_name(f"{target.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    try:
        fd = os.open(
            str(tmp_path),
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            stat.S_IRUSR | stat.S_IWUSR,
        )
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        (replace or atomic_replace)(tmp_path, target)
        if fsync_dir:
            try:
                dir_fd = os.open(str(target.parent), os.O_RDONLY)
            except OSError:
                dir_fd = None
            if dir_fd is not None:
                try:
                    os.fsync(dir_fd)
                finally:
                    os.close(dir_fd)
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass


def _save_auth_store(auth_store: Dict[str, Any], target_path: Optional[Path] = None) -> Path:
    # target_path=None preserves the existing contract (write the active
    # store at _auth_file_path()). An explicit path lets callers persist a
    # specific store — e.g. the global-root write-through for rotating xAI
    # OAuth grants (#43589) — reusing this function's atomic O_EXCL + 0o600
    # write so the root auth.json gets the same TOCTOU-safe treatment.
    auth_file = target_path if target_path is not None else _auth_file_path()
    auth_store["version"] = AUTH_STORE_VERSION
    auth_store["updated_at"] = datetime.now(timezone.utc).isoformat()
    # Parent dir is tightened to 0o700 inside the writer so siblings can't
    # traverse to creds (no-op on Windows; failures ignored).
    _write_private_file_atomic(auth_file, json.dumps(auth_store, indent=2) + "\n", fsync_dir=True)
    # Restrict file permissions to owner only
    try:
        auth_file.chmod(stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        pass
    return auth_file


def _load_provider_state_with_source(
    auth_store: Dict[str, Any],
    provider_id: str,
) -> tuple[Optional[Dict[str, Any]], Optional[Path]]:
    """Return a provider state plus the auth.json path it came from.

    Most callers only need the state, but refresh paths that rotate single-use OAuth refresh tokens
    must write the updated token chain back to the same store they read.
    """
    state = _provider_state_in(auth_store, provider_id)
    if state is not None:
        return state, _auth_file_path()
    global_state = _provider_state_in(_load_global_auth_store(), provider_id)
    if global_state is not None:
        return global_state, _global_auth_file_path()
    return None, None


def _provider_state_in(store: Dict[str, Any], provider_id: str) -> Optional[Dict[str, Any]]:
    """Shallow copy of ``store["providers"][provider_id]`` when it is a dict, else None."""
    providers = store.get("providers") if store else None
    if isinstance(providers, dict):
        state = providers.get(provider_id)
        if isinstance(state, dict):
            return dict(state)
    return None


@contextmanager
def _provider_state_transaction(provider_id: str):
    """Lock the active auth store and any global fallback source in order.

    Profile-backed refresh paths must take the global auth-store lock before any provider-specific
    shared-store lock. Re-reading the source after the target lock is acquired prevents both stale
    refreshes and whole-file lost updates without inverting the documented auth -> shared lock
    order.
    """
    with _auth_store_lock():
        auth_store = _load_auth_store()
        state, source_path = _load_provider_state_with_source(
            auth_store,
            provider_id,
        )
        active_path = _auth_file_path()
        if source_path is None or _same_path(source_path, active_path):
            yield auth_store, state, source_path
            return

        with _auth_store_lock(target_path=source_path):
            source_state = _provider_state_in(_load_auth_store(source_path), provider_id)
            yield auth_store, source_state, source_path


def _load_provider_state(auth_store: Dict[str, Any], provider_id: str) -> Optional[Dict[str, Any]]:
    """Return a provider's persisted state.

    In profile mode, falls back to the global-root ``auth.json`` when the profile has no entry for
    ``provider_id``. This mirrors the per-provider shadowing already used by
    ``read_credential_pool``: workers spawned in a profile can see providers (e.g. ``nous``) that
    were only authenticated at global scope.
    """
    state, _source_path = _load_provider_state_with_source(auth_store, provider_id)
    return state


def _save_provider_state(auth_store: Dict[str, Any], provider_id: str, state: Dict[str, Any]) -> None:
    """Write *state* under ``providers`` and make *provider_id* the active provider."""
    _store_provider_state(auth_store, provider_id, state, set_active=True)


def _save_active_provider_state(provider_id: str, state: Dict[str, Any]) -> Path:
    """Lock, load, write *state* as the active provider, save. Returns the auth store path."""
    with _auth_store_lock():
        auth_store = _load_auth_store()
        _save_provider_state(auth_store, provider_id, state)
        return _save_auth_store(auth_store)


def _save_provider_state_to_source(
    auth_store: Dict[str, Any],
    provider_id: str,
    state: Dict[str, Any],
    source_path: Optional[Path],
) -> None:
    """Persist provider state back to the auth store it was read from."""
    active_path = _auth_file_path()
    if source_path is None or _same_path(source_path, active_path):
        _save_provider_state(auth_store, provider_id, state)
        _save_auth_store(auth_store)
        return

    _persist_provider_state_to_store(
        provider_id,
        state,
        source_path,
        set_active=True,
    )


def _store_provider_state(
    auth_store: Dict[str, Any],
    provider_id: str,
    state: Dict[str, Any],
    *,
    set_active: bool = True,
) -> None:
    providers = auth_store.setdefault("providers", {})
    if not isinstance(providers, dict):
        auth_store["providers"] = {}
        providers = auth_store["providers"]
    providers[provider_id] = state
    if set_active:
        auth_store["active_provider"] = provider_id


def _persist_provider_state_to_store(
    provider_id: str,
    state: Dict[str, Any],
    target_path: Path,
    *,
    set_active: bool = False,
) -> Path:
    """Merge one provider into a specific auth store under that store's lock."""
    with _auth_store_lock(target_path=target_path):
        auth_store = _load_auth_store(target_path)
        _store_provider_state(
            auth_store,
            provider_id,
            dict(state),
            set_active=set_active,
        )
        return _save_auth_store(auth_store, target_path=target_path)


def mark_provider_active_if_unset(provider_id: str) -> None:
    """Set ``active_provider`` to *provider_id* only when none is set yet.

    Used by ``hermes auth add`` OAuth paths that write pool entries directly: the first credential
    for a provider must make it active so the setup wizard's credential check does not report
    "No inference provider configured". Later adds leave the user's chosen provider untouched.
    """
    with _auth_store_lock():
        auth_store = _load_auth_store()
        if not (auth_store.get("active_provider") or "").strip():
            auth_store["active_provider"] = provider_id
            _save_auth_store(auth_store)


def is_known_auth_provider(provider_id: str) -> bool:
    normalized = (provider_id or "").strip().lower()
    return normalized in PROVIDER_REGISTRY or normalized in SERVICE_PROVIDER_NAMES


def get_auth_provider_display_name(provider_id: str) -> str:
    normalized = (provider_id or "").strip().lower()
    if normalized in PROVIDER_REGISTRY:
        return PROVIDER_REGISTRY[normalized].name
    return SERVICE_PROVIDER_NAMES.get(normalized, provider_id)


def is_runtime_provider_routable(provider_id: str) -> bool:
    """Return whether runtime resolution recognizes a provider identity.

    A capability check, not a credential check: same alias/plugin-aware normalization as
    ``resolve_provider`` while preserving special runtime identities that live outside the registry.
    """
    normalized = (provider_id or "").strip().lower()
    if not normalized:
        return False
    if normalized in {"auto", "openrouter", "custom", "moa"}:
        return True
    if normalized.startswith("custom:"):
        return True
    try:
        resolve_provider(normalized)
    except AuthError:
        return False
    return True


# ── One-time heal for installs that ALREADY forked a single-use grant ────────
#
# Fleets created before the clone-strip / root-write-through above have
# profile-local copies of the root grant. Those copies are the same credential
# with several owners: whichever profile rotated last holds the only live
# refresh token and every other copy (root included) is spent. Upgrading alone
# does not fix that — the first load in each profile would keep using its own
# doomed copy. ``heal_forked_single_use_oauth_grants`` runs at profile
# ``load_pool()`` time: it finds the profile rows that share LINEAGE with a
# root row (same pool id — clone-all and the old borrowed-persist both kept
# it — or the same account identity / token material), keeps the copy most
# likely to still be live (freshest rotation), writes that copy into ROOT when
# root's is older, and strips the profile's copy so the profile borrows root
# from then on. Idempotent (a healed profile has no matched rows), never
# touches API-key rows, never deletes a row that has no root counterpart
# (an independent ``hermes -p <p> auth add`` grant, or the only surviving
# copy), and reads only the two auth.json files the existing root fallback
# already reads — no environ / secret-scope reads.


def read_credential_pool(provider_id: Optional[str] = None) -> Dict[str, Any]:
    """Return the persisted credential pool, or one provider slice.

    In profile mode, the profile's credential pool is authoritative. If a provider has no entries in
    the profile, entries from the global-root ``auth.json`` are used as a read-only fallback — so
    workers spawned in a profile can see providers that were only authenticated at global scope.

    Profile entries always win: the global fallback only applies per-provider when the profile has
    zero entries for that provider. Once the user runs ``hermes auth add <provider>`` inside the
    profile, profile entries fully shadow global for that provider on the next read.
    """
    auth_store = _load_auth_store()
    pool = auth_store.get("credential_pool")
    if not isinstance(pool, dict):
        pool = {}

    global_pool: Dict[str, Any] = {}
    global_store = _load_global_auth_store()
    maybe_global_pool = global_store.get("credential_pool") if global_store else None
    if isinstance(maybe_global_pool, dict):
        global_pool = maybe_global_pool

    if provider_id is None:
        merged = dict(pool)
        for gp_key, gp_entries in global_pool.items():
            if not isinstance(gp_entries, list) or not gp_entries:
                continue
            # Per-provider shadowing: profile wins whenever it has ANY entries.
            existing = merged.get(gp_key)
            if isinstance(existing, list) and existing:
                continue
            merged[gp_key] = list(gp_entries)
        return merged

    provider_entries = pool.get(provider_id)
    if isinstance(provider_entries, list) and provider_entries:
        return list(provider_entries)
    # Profile has no entries for this provider — fall back to global.
    global_entries = global_pool.get(provider_id)
    return list(global_entries) if isinstance(global_entries, list) else []


_POOL_STATUS_FIELDS = (
    "last_status",
    "last_status_at",
    "last_error_code",
    "last_error_reason",
    "last_error_message",
    "last_error_reset_at",
)


def _merge_disk_cooldown_state(
    entry: Dict[str, Any],
    disk_entry: Optional[Dict[str, Any]],
    provider_id: str,
) -> Dict[str, Any]:
    """Keep a newer on-disk cooldown/quarantine over a stale in-memory one.

    ``write_credential_pool`` callers persist an in-memory snapshot that may predate another process
    marking the same credential exhausted or dead (last-writer-wins lost update). Without this
    merge, process B's later rewrite resurrects a rate-limited key as healthy and both processes
    resume hammering it.
    """
    if not isinstance(disk_entry, dict):
        return entry
    try:
        from agent.credential_pool import (
            PooledCredential,
            STATUS_DEAD,
            STATUS_EXHAUSTED,
            _exhausted_until,
            _parse_absolute_timestamp,
        )

        disk_status = disk_entry.get("last_status")
        if disk_status not in (STATUS_DEAD, STATUS_EXHAUSTED):
            return entry
        # A token change means the caller re-authed/refreshed this entry and
        # intentionally cleared its status (e.g. _sync_codex_entry_from_
        # auth_store after a fresh device-code login) — never resurrect the
        # old cooldown onto fresh credentials.
        mem_access = entry.get("access_token") or ""
        disk_access = disk_entry.get("access_token") or ""
        if mem_access and disk_access and mem_access != disk_access:
            return entry
        disk_ts = _parse_absolute_timestamp(disk_entry.get("last_status_at")) or 0.0
        mem_ts = _parse_absolute_timestamp(entry.get("last_status_at")) or 0.0
        if disk_ts <= mem_ts:
            return entry
        if disk_status == STATUS_EXHAUSTED:
            until = _exhausted_until(
                PooledCredential.from_dict(provider_id, disk_entry)
            )
            if until is None or until <= time.time():
                return entry
        merged_entry = dict(entry)
        for status_field in _POOL_STATUS_FIELDS:
            merged_entry[status_field] = disk_entry.get(status_field)
        return merged_entry
    except Exception:  # pragma: no cover - best-effort merge
        return entry


def write_credential_pool(
    provider_id: str,
    entries: List[Dict[str, Any]],
    *,
    removed_ids: Optional[Iterable[str]] = None,
) -> Path:
    """Persist one provider's credential pool under auth.json.

    This is the final disk-boundary guard for borrowed/reference-only credentials. Callers may pass
    raw dictionaries, so sanitize here even when ``PooledCredential.to_dict()`` already did the same
    work upstream.

    Re-read the on-disk pool under the same lock and merge entries present on disk but missing from
    ``entries``. Those were added by another process after the caller loaded its in-memory snapshot;
    without this merge a later rotation/exhaustion rewrite drops the concurrent credential.
    """
    removed = {rid for rid in (removed_ids or ()) if rid}
    with _auth_store_lock():
        auth_store = _load_auth_store()
        pool = auth_store.get("credential_pool")
        if not isinstance(pool, dict):
            pool = {}
            auth_store["credential_pool"] = pool
        sanitized_entries = [
            sanitize_borrowed_credential_payload(entry, provider_id)
            if isinstance(entry, dict) else entry
            for entry in entries
        ]
        existing = pool.get(provider_id)
        existing_list = existing if isinstance(existing, list) else []
        existing_by_id = {
            entry.get("id"): entry
            for entry in existing_list
            if isinstance(entry, dict) and entry.get("id")
        }
        new_ids = {
            entry.get("id")
            for entry in sanitized_entries
            if isinstance(entry, dict) and entry.get("id")
        }
        merged: List[Dict[str, Any]] = [
            _merge_disk_cooldown_state(
                entry, existing_by_id.get(entry.get("id")), provider_id
            )
            if isinstance(entry, dict)
            else entry
            for entry in sanitized_entries
        ]
        for disk_entry in existing_list:
            if not isinstance(disk_entry, dict):
                continue
            disk_id = disk_entry.get("id")
            if not disk_id or disk_id in new_ids or disk_id in removed:
                continue
            merged.append(sanitize_borrowed_credential_payload(disk_entry, provider_id))
        pool[provider_id] = merged
        return _save_auth_store(auth_store)


def suppress_credential_source(provider_id: str, source: str) -> None:
    """Mark a credential source as suppressed so it won't be re-seeded.

    Older auth stores may represent a provider's suppressed sources as a mapping. Treat its keys as
    source names and migrate the value to the canonical list form before appending the requested
    source.
    """
    with _auth_store_lock():
        auth_store = _load_auth_store()
        suppressed = auth_store.get("suppressed_sources")
        if not isinstance(suppressed, dict):
            suppressed = {}
            auth_store["suppressed_sources"] = suppressed
        provider_list = _suppressed_source_list(suppressed, provider_id)
        if provider_list is None:
            provider_list = []
            suppressed[provider_id] = provider_list
        if source not in provider_list:
            provider_list.append(source)
        _save_auth_store(auth_store)


def _suppressed_source_list(suppressed: Dict[str, Any], provider_id: str) -> Optional[List[str]]:
    """Canonical (list-form) suppressed sources for *provider_id*, migrating a legacy mapping in place."""
    raw_sources = suppressed.get(provider_id)
    if isinstance(raw_sources, list):
        return raw_sources
    if isinstance(raw_sources, dict):
        provider_list = [str(name) for name in raw_sources]
        suppressed[provider_id] = provider_list
        return provider_list
    return None


def is_source_suppressed(provider_id: str, source: str) -> bool:
    """Check if a credential source has been suppressed by the user."""
    try:
        auth_store = _load_auth_store()
        suppressed = auth_store.get("suppressed_sources", {})
        return source in suppressed.get(provider_id, [])
    except Exception:
        return False


def unsuppress_credential_source(provider_id: str, source: str) -> bool:
    """Clear a suppression marker so the source will be re-seeded on the next load."""
    with _auth_store_lock():
        auth_store = _load_auth_store()
        suppressed = auth_store.get("suppressed_sources")
        if not isinstance(suppressed, dict):
            return False
        provider_list = _suppressed_source_list(suppressed, provider_id)
        if provider_list is None or source not in provider_list:
            return False
        provider_list.remove(source)
        if not provider_list:
            suppressed.pop(provider_id, None)
        if not suppressed:
            auth_store.pop("suppressed_sources", None)
        _save_auth_store(auth_store)
        return True


def get_provider_auth_state(provider_id: str) -> Optional[Dict[str, Any]]:
    """Return persisted auth state for a provider, or None.

    In profile mode, ``_load_provider_state`` already falls back to the global-root ``auth.json``
    per-provider when the profile has no entry — so this is now a thin convenience wrapper. Profile
    state always wins when present.
    """
    auth_store = _load_auth_store()
    return _load_provider_state(auth_store, provider_id)


def get_active_provider() -> Optional[str]:
    """Return the currently active provider ID from auth store."""
    auth_store = _load_auth_store()
    return auth_store.get("active_provider")


def is_provider_explicitly_configured(provider_id: str) -> bool:
    """Return True only if the user has explicitly configured this provider.

    Claude Code's ~/.claude/.credentials.json) so they are never used without the user's explicit
    choice.
    """
    normalized = (provider_id or "").strip().lower()

    # 1. Check auth.json active_provider
    try:
        auth_store = _load_auth_store()
        active = (auth_store.get("active_provider") or "").strip().lower()
        if active and active == normalized:
            return True
    except Exception:
        pass

    # 2. Check config.yaml model.provider and other explicit provider slots.
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
        model_cfg = cfg.get("model")
        if isinstance(model_cfg, dict):
            cfg_provider = (model_cfg.get("provider") or "").strip().lower()
            if cfg_provider == normalized:
                return True

        # MoA presets are explicit model selections too.  A user who configured
        # ``provider: anthropic`` as a MoA advisor/aggregator has opted Hermes
        # into using Anthropic credentials for that slot even when the main
        # session model is another provider.  Without this, Claude Code OAuth
        # entries are pruned/ignored by credential_pool.load_pool("anthropic"),
        # so MoA Anthropic advisors fail with "no ANTHROPIC_API_KEY" while the
        # normal model picker says Anthropic is logged in.
        def _slot_matches_provider(slot):
            return (
                isinstance(slot, dict)
                and (slot.get("provider") or "").strip().lower() == normalized
            )

        def _moa_block_matches(block: Any) -> bool:
            return isinstance(block, dict) and (
                any(_slot_matches_provider(s) for s in block.get("reference_models") or [])
                or _slot_matches_provider(block.get("aggregator"))
            )

        moa_cfg = cfg.get("moa")
        if isinstance(moa_cfg, dict):
            if _moa_block_matches(moa_cfg):
                return True
            presets = moa_cfg.get("presets")
            if isinstance(presets, dict) and any(_moa_block_matches(p) for p in presets.values()):
                return True
    except Exception:
        pass

    # 3. Provider-specific env vars (explicit secrets only).
    if _explicit_env_credentials_present(normalized):
        return True

    # 4. Check persisted credential-pool entries that came from EXPLICIT flows
    # the user initiated inside Hermes (manual add / device-code / PKCE), plus
    # env-backed pool entries. This intentionally excludes ambient borrowed
    # sources like gh_cli / claude_code / qwen-cli.
    try:
        if any(_pool_entry_is_explicit(entry) for entry in read_credential_pool(normalized)):
            return True
    except Exception:
        pass

    # 5. OAuth-token / cloud-SDK providers (Vertex AI, Bedrock) have NO API-key
    # env var to detect in step 3 and mint short-lived tokens from ADC / a
    # service account / the AWS SDK chain. The user "explicitly configures"
    # them by writing non-secret routing settings into config.yaml
    # (``vertex.project_id`` / a credentials path, ``bedrock.region``) rather
    # than by pasting a key — so without this branch such a provider is only
    # ever "explicitly configured" while it is the *current* provider, and it
    # silently vanishes from explicit-only pickers (desktop chat model menu)
    # otherwise. Treat the presence of that deliberate config as explicit.
    try:
        if _keyless_provider_has_explicit_config(normalized):
            return True
    except Exception as exc:
        logger.debug("Failed checking keyless provider explicit config for %s: %s", provider_id, exc)

    return False


# Set by Claude Code itself, not by the user explicitly configuring anthropic in Hermes.
_IMPLICIT_ENV_VARS = frozenset({"CLAUDE_CODE_OAUTH_TOKEN"})
_EXPLICIT_POOL_SOURCES = frozenset({"device_code", "loopback_pkce", "hermes_pkce", "manual"})
_VERTEX_PROVIDER_IDS = ("vertex", "google-vertex", "vertex-ai", "gcp-vertex", "vertexai")


def _explicit_env_credentials_present(normalized: str) -> bool:
    """True when the user has pasted an explicit credential env var for *normalized*.

    Falls back to the models.dev ``ProviderDef`` when the provider isn't in PROVIDER_REGISTRY
    (e.g. openrouter) — both expose ``.auth_type`` / ``.api_key_env_vars`` with the same shape.
    AWS SDK providers (Bedrock) have empty ``api_key_env_vars``, so check their explicit env
    credentials directly — NOT boto3's full chain: ambient sources like EC2 IMDS / SSO profiles
    must not auto-surface, but AWS_BEARER_TOKEN_BEDROCK or an access-key pair in .env is as
    explicit as pasting ANTHROPIC_API_KEY.
    """
    pconfig = PROVIDER_REGISTRY.get(normalized)
    if pconfig is None:
        from hermes_cli.providers import get_provider
        pconfig = get_provider(normalized)
    if not pconfig:
        return False
    if pconfig.auth_type == "api_key":
        return any(
            has_usable_secret(os.getenv(env_var, ""))
            for env_var in pconfig.api_key_env_vars
            if env_var not in _IMPLICIT_ENV_VARS
        )
    if pconfig.auth_type == "aws_sdk":
        return has_usable_secret(os.getenv("AWS_BEARER_TOKEN_BEDROCK", "")) or (
            has_usable_secret(os.getenv("AWS_ACCESS_KEY_ID", ""))
            and has_usable_secret(os.getenv("AWS_SECRET_ACCESS_KEY", ""))
        )
    return False


def _pool_entry_is_explicit(entry: Any) -> bool:
    """True for pool rows the user created via an explicit Hermes flow (or a still-live env key)."""
    if not isinstance(entry, dict):
        return False
    source = str(entry.get("source") or "").strip().lower()
    if not source:
        return False
    if source.startswith("env:"):
        # A stale env-seeded pool entry survives in auth.json after
        # the user deletes the env var (#55790) — only count it when
        # the referenced var still resolves to a usable secret NOW.
        env_var = entry.get("source", "").split(":", 1)[1].strip()
        return bool(env_var and has_usable_secret(os.getenv(env_var, "")))
    return source in _EXPLICIT_POOL_SOURCES or source.startswith("manual:")


def _keyless_provider_has_explicit_config(normalized: str) -> bool:
    """Vertex / Bedrock count as explicit when Hermes-scoped routing config is present.

    Uses has_explicit_vertex_config(), NOT has_vertex_credentials() — the latter also counts an
    ambient GOOGLE_APPLICATION_CREDENTIALS path (commonly set globally for unrelated GCP work),
    which would mark Vertex explicit for users who never set Hermes up for it. Only Hermes-scoped
    signals (VERTEX_PROJECT_ID / vertex.project_id / VERTEX_CREDENTIALS_PATH) count here.
    """
    if normalized in _VERTEX_PROVIDER_IDS:
        from agent.vertex_adapter import has_explicit_vertex_config

        return bool(has_explicit_vertex_config())
    if normalized == "bedrock":
        from hermes_cli.config import load_config as _load_cfg

        bedrock_cfg = _load_cfg().get("bedrock")
        return isinstance(bedrock_cfg, dict) and bool(str(bedrock_cfg.get("region") or "").strip())
    return False


def clear_provider_auth(provider_id: Optional[str] = None) -> bool:
    """Clear auth state for a provider. Used by `hermes logout`. If provider_id is None, clears the
    active provider. Returns True if something was cleared.
    """
    with _auth_store_lock():
        auth_store = _load_auth_store()
        target = provider_id or auth_store.get("active_provider")
        if not target:
            return False

        providers = auth_store.get("providers", {})
        if not isinstance(providers, dict):
            providers = {}
            auth_store["providers"] = providers

        pool = auth_store.get("credential_pool")
        if not isinstance(pool, dict):
            pool = {}
            auth_store["credential_pool"] = pool

        cleared = False
        if target in providers:
            del providers[target]
            cleared = True
        if target in pool:
            del pool[target]
            cleared = True

        if auth_store.get("active_provider") == target:
            auth_store["active_provider"] = None
            cleared = True

        if not cleared:
            return False
        _save_auth_store(auth_store)
    return True


def deactivate_provider() -> None:
    """Clear active_provider in auth.json without deleting credentials. Used when the user switches to
    a non-OAuth provider (OpenRouter, custom) so auto-resolution doesn't keep picking the OAuth
    provider.
    """
    with _auth_store_lock():
        auth_store = _load_auth_store()
        auth_store["active_provider"] = None
        _save_auth_store(auth_store)


# =============================================================================
# Provider Resolution — picks which provider to use
# =============================================================================


def _get_config_hint_for_unknown_provider(provider_name: str) -> str:
    """Return a helpful hint string when provider resolution fails."""
    try:
        from hermes_cli.config import validate_config_structure
        issues = validate_config_structure()
        if not issues:
            return ""

        lines = ["Config issue detected — run 'hermes doctor' for full diagnostics:"]
        for ci in issues:
            prefix = "ERROR" if ci.severity == "error" else "WARNING"
            lines.append(f"  [{prefix}] {ci.message}")
            # Show first line of hint
            first_hint = ci.hint.splitlines()[0] if ci.hint else ""
            if first_hint:
                lines.append(f"    → {first_hint}")
        return "\n".join(lines)
    except Exception:
        return ""


def _refuse_env_adoption_if_config_corrupt() -> None:
    """Refuse env-key/pool auto-adoption of openrouter while config.yaml is corrupt.

    When ``~/.hermes/config.yaml`` EXISTS but fails to parse, ``load_config()`` falls back to
    ``DEFAULT_CONFIG`` — so the tier-2 config check above finds no ``model.provider`` and the env-
    var sniff / pool probe silently adopts the PAID openrouter provider, even though the user's real
    (broken) config may name a completely different provider (e.g.

    This probe fires ONLY on the auto path — explicitly requested providers never reach it — and
    clears itself as soon as the file changes (a fixed config resolves normally on the next call).
    """
    try:
        from hermes_cli.config import get_active_config_parse_failure, get_config_path

        err = get_active_config_parse_failure()
        if not err:
            return
        path = get_config_path()
    except Exception as e:
        logger.debug("Could not probe config parse-failure state: %s", e)
        return
    raise AuthError(
        f"config.yaml at {path} is corrupt ({err}) — refusing to auto-select "
        f"an inference provider from environment keys. Fix the YAML (a backup "
        f"was saved next to it) or run hermes setup.",
        code="corrupt_config",
    )


# Provider aliases accepted by resolve_provider(). Plugin-declared aliases
# (plugins/model-providers/<name>/) are layered on at call time; this hardcoded
# table remains authoritative for existing names.
_PROVIDER_ALIASES: Dict[str, str] = {
    "glm": "zai", "z-ai": "zai", "z.ai": "zai", "zhipu": "zai",
    "google": "gemini", "google-gemini": "gemini", "google-ai-studio": "gemini",
    "x-ai": "xai", "x.ai": "xai", "grok": "xai",
    "xai-oauth": "xai-oauth", "x-ai-oauth": "xai-oauth",
    "grok-oauth": "xai-oauth", "xai-grok-oauth": "xai-oauth",
    "kimi": "kimi-coding", "kimi-for-coding": "kimi-coding", "moonshot": "kimi-coding",
    "kimi-cn": "kimi-coding-cn", "moonshot-cn": "kimi-coding-cn",
    "step": "stepfun", "stepfun-coding-plan": "stepfun",
    "arcee-ai": "arcee", "arceeai": "arcee",
    "gmi-cloud": "gmi", "gmicloud": "gmi",
    "actual-computer": "actual", "actualcomputer": "actual", "aci": "actual",
    "minimax-china": "minimax-cn", "minimax_cn": "minimax-cn",
    "minimax-portal": "minimax-oauth", "minimax-global": "minimax-oauth", "minimax_oauth": "minimax-oauth",
    "alibaba_coding": "alibaba-coding-plan", "alibaba-coding": "alibaba-coding-plan",
    "alibaba_coding_plan": "alibaba-coding-plan",
    "claude": "anthropic", "claude-code": "anthropic",
    "github": "copilot", "github-copilot": "copilot",
    "github-models": "copilot", "github-model": "copilot",
    "github-copilot-acp": "copilot-acp", "copilot-acp-agent": "copilot-acp",
    "aigateway": "ai-gateway", "vercel": "ai-gateway", "vercel-ai-gateway": "ai-gateway",
    "opencode": "opencode-zen", "zen": "opencode-zen",
    "free": "opencode-free", "opencode_free": "opencode-free",
    "qwen-portal": "qwen-oauth", "qwen-cli": "qwen-oauth", "qwen-oauth": "qwen-oauth",
    "hf": "huggingface", "hugging-face": "huggingface", "huggingface-hub": "huggingface",
    "mimo": "xiaomi", "xiaomi-mimo": "xiaomi",
    "tencent": "tencent-tokenhub", "tokenhub": "tencent-tokenhub",
    "tencent-cloud": "tencent-tokenhub", "tencentmaas": "tencent-tokenhub",
    "tokenplan": "tencent-tokenplan", "tencent-lkeap": "tencent-tokenplan",
    "aws": "bedrock", "aws-bedrock": "bedrock", "amazon-bedrock": "bedrock", "amazon": "bedrock",
    "go": "opencode-go", "opencode-go-sub": "opencode-go",
    "kilo": "kilocode", "kilo-code": "kilocode", "kilo-gateway": "kilocode",
    "lmstudio": "lmstudio", "lm-studio": "lmstudio", "lm_studio": "lmstudio",
    # Local server aliases — route through the generic custom provider
    "ollama": "custom", "ollama_cloud": "ollama-cloud",
    "vllm": "custom", "llamacpp": "custom",
    "llama.cpp": "custom", "llama-cpp": "custom",
}


def _scoped_key_env_reader() -> Callable[[str], str]:
    """Scope-aware key reader for provider auto-detection.

    Under multiplex a secondary profile's API keys live only in its secret scope, not os.environ —
    a bare getenv would find nothing and auto-resolution would report "No LLM provider configured"
    for every secondary profile (same class as #86905). Catch ONLY ImportError: any other failure
    inside auxiliary_client must propagate — silently falling back to os.getenv would reintroduce
    the very fail-open this removes, with zero trace.
    """
    try:
        from agent.auxiliary_client import _scoped_key_env
        return _scoped_key_env
    except ImportError:
        logger.warning(
            "agent.auxiliary_client unavailable (%s); provider auto-detection "
            "will read keys from the process environment only — under "
            "multiplex, secondary profiles may report 'No LLM provider'.",
            "import failed",
        )
        return lambda name: os.getenv(name) or ""


def _openrouter_auto_detected(scoped_key_env: Callable[[str], str]) -> bool:
    """True when an OpenRouter credential exists via env key or the credential pool.

    The pool check covers a key added via `hermes auth add openrouter` (manual pool entry, no env
    var). Without it, a pool-only key is invisible to auto-detection — `hermes auth list` shows the
    credential while requests go out with no Authorization header (#42130).
    """
    if has_usable_secret(scoped_key_env("OPENAI_API_KEY")) or has_usable_secret(
        scoped_key_env("OPENROUTER_API_KEY")
    ):
        return True
    try:
        from agent.credential_pool import load_pool as _load_pool

        return bool(_load_pool("openrouter").has_credentials())
    except Exception as e:
        logger.debug("Could not check OpenRouter credential pool: %s", e)
        return False


def _logged_in_oauth_active_provider() -> Optional[str]:
    """auth.json ``active_provider`` when it is a registry provider that reports logged in."""
    try:
        _store = _load_auth_store()
        _maybe = _store.get("active_provider")
        if _maybe and _maybe in PROVIDER_REGISTRY and get_auth_status(_maybe).get("logged_in"):
            return _maybe
    except Exception as e:
        logger.debug("Could not pre-read active auth provider: %s", e)
    return None


def resolve_provider(
    requested: Optional[str] = None,
    *,
    explicit_api_key: Optional[str] = None,
    explicit_base_url: Optional[str] = None,
) -> str:
    """Determine which inference provider to use.

    Priority (when requested="auto" or None) — explicit user intent wins over a stale logged-in
    OAuth provider (#29285): 1. Explicit CLI api_key/base_url -> "openrouter" 2. config.yaml
    `model.provider` 3. OPENAI_API_KEY / OPENROUTER_API_KEY env vars -> "openrouter" 4. OpenRouter
    credential pool 5.
    """
    normalized = (requested or "auto").strip().lower()

    # Normalize provider aliases. Extend with aliases declared in
    # plugins/model-providers/<name>/ that aren't already mapped.
    aliases = dict(_PROVIDER_ALIASES)
    try:
        from providers import list_providers as _lp
        for _pp in _lp():
            for _alias in _pp.aliases:
                if _alias not in aliases:
                    aliases[_alias] = _pp.name
    except Exception:
        pass
    normalized = aliases.get(normalized, normalized)

    if normalized == "openrouter":
        return "openrouter"
    if normalized == "custom":
        return "custom"
    if normalized in PROVIDER_REGISTRY:
        return normalized
    if normalized != "auto":
        # Check for common config.yaml issues that cause this error
        _config_hint = _get_config_hint_for_unknown_provider(normalized)
        msg = f"Unknown provider '{normalized}'."
        if _config_hint:
            msg += f"\n\n{_config_hint}"
        else:
            msg += " Check 'hermes model' for available providers, or run 'hermes doctor' to diagnose config issues."
        raise AuthError(msg, code="invalid_provider")

    # Explicit one-off CLI creds always mean openrouter/custom
    if explicit_api_key or explicit_base_url:
        return "openrouter"

    # Provider precedence for the auto-path (#29285): explicit user intent must
    # win over a stale logged-in OAuth `active_provider`. Order matches the
    # docstring: 1. explicit CLI creds  2. config.yaml `model.provider`
    # 3. OPENAI/OPENROUTER env keys  4. OpenRouter pool  5. provider-specific
    # env keys  6. auth.json `active_provider` (OAuth)  7. Bedrock  8. error.
    # The normal chat/gateway path resolves config.provider upstream in
    # resolve_requested_provider() before ever reaching "auto"; this duplicate
    # check is the safety net for the lone direct caller (main.py resolve_provider
    # ("auto")) and any future bypass of that stage.
    _model_cfg: Any = None
    try:
        from hermes_cli.config import load_config

        _model_cfg = (load_config() or {}).get("model")
        if isinstance(_model_cfg, dict):
            _cfg_provider = _model_cfg.get("provider")
            if isinstance(_cfg_provider, str) and _cfg_provider.strip().lower() in PROVIDER_REGISTRY:
                return _cfg_provider.strip().lower()
    except Exception as e:
        logger.debug("Could not read config.yaml model.provider for auto-resolution: %s", e)

    _scoped_key_env = _scoped_key_env_reader()

    # Tiers 3-4: OPENAI/OPENROUTER env keys, then the OpenRouter credential pool.
    if _openrouter_auto_detected(_scoped_key_env):
        _refuse_env_adoption_if_config_corrupt()
        return "openrouter"

    # Determine the logged-in OAuth provider up front so the env-key loop below
    # can WARN when an exported API key preempts it (#29285 transparency). The
    # actual OAuth fallback (tier 6) still happens later if nothing else matches.
    _oauth_active = _logged_in_oauth_active_provider()

    # Auto-detect API-key providers by checking their env vars
    for pid, pconfig in PROVIDER_REGISTRY.items():
        if pconfig.auth_type != "api_key":
            continue
        # GitHub tokens are commonly present for repo/tool access but should not
        # hijack inference auto-selection unless the user explicitly chooses
        # Copilot/GitHub Models as the provider. LM Studio is a local server
        # whose availability isn't implied by LM_API_KEY presence (it may be
        # offline, and the no-auth setup uses a placeholder value), so it
        # also requires explicit selection.
        if pid in {"copilot", "lmstudio"}:
            continue
        for env_var in pconfig.api_key_env_vars:
            if has_usable_secret(_scoped_key_env(env_var)):
                # An exported API key now wins over a logged-in OAuth provider
                # (the #29285 fix). Surface that so a user who deliberately uses
                # OAuth but has a stale key in ~/.hermes/.env isn't silently
                # switched without knowing why.
                if _oauth_active and _oauth_active != pid:
                    logger.warning(
                        "Provider resolved to %r via %s, preempting your "
                        "logged-in OAuth provider %r. If you meant to use the "
                        "OAuth login, unset %s or set `model.provider` "
                        "explicitly.",
                        pid, env_var, _oauth_active, env_var,
                    )
                return pid

    # Logged-in OAuth provider (auth.json `active_provider`) — a LAST-RESORT
    # fallback, chosen only when the user expressed no other preference above.
    # Previously this sat ABOVE the env-var/config checks, so a stale OAuth
    # login silently overrode an explicit `model.provider` or an exported API
    # key (#29285). Demoted here so explicit intent always wins.
    if _oauth_active:
        # Surface the silent-override case the issue reported: a populated
        # `model` config that lacks a `provider` key falls through to OAuth.
        if isinstance(_model_cfg, dict) and _model_cfg and not _model_cfg.get("provider"):
            logger.warning(
                "Provider resolved to logged-in OAuth provider %r because "
                "config.yaml `model` has no `provider` key. If you meant a "
                "different provider, set `model.provider` explicitly.",
                _oauth_active,
            )
        return _oauth_active

    # AWS Bedrock — detect via boto3 credential chain (IAM roles, SSO, env vars).
    # This runs after API-key providers so explicit keys always win.
    try:
        from agent.bedrock_adapter import has_aws_credentials
        if has_aws_credentials():
            return "bedrock"
    except ImportError:
        pass  # boto3 not installed — skip Bedrock auto-detection

    raise AuthError(
        "No inference provider configured. Run 'hermes model' to choose a "
        "provider and model, or set an API key (OPENROUTER_API_KEY, "
        "OPENAI_API_KEY, etc.) in ~/.hermes/.env.",
        code="no_provider_configured",
    )


# =============================================================================
# Timestamp / TTL helpers
# =============================================================================

def _utc_now_z() -> str:
    """Current UTC time as an ISO-8601 string with a ``Z`` suffix (last_refresh format)."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_iso_timestamp(value: Any) -> Optional[float]:
    if not isinstance(value, str) or not value:
        return None
    text = value.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _is_expiring(expires_at_iso: Any, skew_seconds: int) -> bool:
    expires_epoch = _parse_iso_timestamp(expires_at_iso)
    if expires_epoch is None:
        return True
    return expires_epoch <= (time.time() + skew_seconds)


def _tls_state_from_verify(verify: Any) -> Dict[str, Any]:
    """Persistable ``tls`` block derived from an httpx ``verify`` value."""
    return {
        "insecure": verify is False,
        "ca_bundle": verify if isinstance(verify, str) else None,
    }


def _last_auth_error_marker(
    provider: str,
    error: "AuthError",
    *,
    reason: str,
    default_code: Optional[str] = None,
) -> Dict[str, Any]:
    """The ``last_auth_error`` record persisted when dead OAuth material is quarantined."""
    return {
        "provider": provider,
        "code": error.code if default_code is None else (error.code or default_code),
        "message": str(error),
        "reason": reason,
        "relogin_required": True,
        "at": datetime.now(timezone.utc).isoformat(),
    }


_FLAT_OAUTH_TOKEN_KEYS = ("access_token", "refresh_token", "expires_at", "expires_in", "obtained_at")

def _quarantine_flat_oauth_state(state: Dict[str, Any], provider: str, exc: "AuthError") -> None:
    """Strip dead tokens from a flat OAuth state after a terminal runtime refresh failure.

    Mirrors the Nous / xAI / Codex quarantine pattern so subsequent calls fail fast without a
    network retry.
    """
    for _k in _FLAT_OAUTH_TOKEN_KEYS:
        state.pop(_k, None)
    state["last_auth_error"] = _last_auth_error_marker(
        provider, exc, reason="runtime_refresh_failure", default_code="refresh_failed",
    )


def _coerce_ttl_seconds(expires_in: Any) -> int:
    try:
        ttl = int(expires_in)
    except Exception:
        ttl = 0
    return max(0, ttl)


def _optional_base_url(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    cleaned = value.strip().rstrip("/")
    return cleaned if cleaned else None


# Allowlist of valid Nous Portal hosts. A portal_base_url outside this
# set is treated as a misconfiguration and falls back to the default.
# "localhost" / "127.0.0.1" are valid for local development and testing.
_NOUS_PORTAL_ALLOWED_HOSTS: FrozenSet[str] = frozenset({
    "portal.nousresearch.com",
    "localhost",
    "127.0.0.1",
})


# =============================================================================
# Spotify auth — PKCE tokens stored in ~/.hermes/auth.json
# =============================================================================


# =============================================================================
# SSH / remote session detection
# =============================================================================


# =============================================================================
# OpenAI Codex auth — tokens stored in ~/.hermes/auth.json (not ~/.codex/)
#
# Hermes maintains its own Codex OAuth session separate from the Codex CLI
# and VS Code extension. This prevents refresh token rotation conflicts
# where one app's refresh invalidates the other's session.
# =============================================================================


# =============================================================================
# xAI Grok OAuth — tokens stored in ~/.hermes/auth.json
# =============================================================================


# =============================================================================
# TLS verification helper
# =============================================================================


# =============================================================================
# OAuth Device Code Flow — generic, parameterized by provider
# =============================================================================


# =============================================================================
# Nous Portal — token refresh and model discovery
# =============================================================================

# -----------------------------------------------------------------------------
# Shared Nous token store — lets OAuth credentials persist across profiles
# so a new `hermes --profile <name> auth add nous --type oauth` can one-tap
# import instead of running the full device-code flow every time.
#
# File lives at ${HERMES_SHARED_AUTH_DIR}/nous_auth.json, defaulting to
# ``<hermes-root>/shared/nous_auth.json`` where ``<hermes-root>`` is what
# ``get_default_hermes_root()`` returns — ``~/.hermes`` on Linux/macOS,
# ``%LOCALAPPDATA%\hermes`` on native Windows, or the Docker/custom root.
# It is OUTSIDE any named profile's HERMES_HOME so named profiles (which
# typically live under ``<hermes-root>/profiles/<name>/``) all see the
# same file.
#
# Written on successful login and on every runtime refresh so the stored
# refresh_token stays current even if one profile refreshes and rotates it.
# If ever the stored refresh_token does go stale server-side, import fails
# gracefully and the user falls back to the normal device-code flow.
# -----------------------------------------------------------------------------


# Per-process memo for resolve_nous_access_token. Startup runs
# check_tool_availability once per managed-tool check_fn (browser, image_gen,
# etc.), and each one independently triggers a ~15s blocking token-refresh
# network call when the stored token is expired. On a slow/constrained host that
# serial burst stretches startup to many minutes. A short-TTL memo collapses the
# burst into a single network round-trip; callers that need freshness use
# separate flows (force_fresh / refresh_nous_oauth_pure) and are unaffected.
_RESOLVE_TOKEN_CACHE_LOCK = threading.Lock()
_RESOLVE_TOKEN_CACHE: "tuple[float, str] | None" = None
_RESOLVE_TOKEN_CACHE_TTL_S = 5.0


def resolve_nous_access_token(
    *,
    timeout_seconds: float = 15.0,
    insecure: Optional[bool] = None,
    ca_bundle: Optional[str] = None,
    refresh_skew_seconds: int = ACCESS_TOKEN_REFRESH_SKEW_SECONDS,
) -> str:
    """Resolve a refresh-aware Nous Portal access token for managed tool gateways."""
    global _RESOLVE_TOKEN_CACHE
    # Memo: collapse the startup burst of managed-tool check_fns into one
    # network refresh. Only cache a successful, non-forced resolution for a
    # short window; force_fresh / error paths bypass and don't populate it.
    if not insecure and ca_bundle is None:
        with _RESOLVE_TOKEN_CACHE_LOCK:
            if _RESOLVE_TOKEN_CACHE is not None:
                cached_at, cached_token = _RESOLVE_TOKEN_CACHE
                if (time.monotonic() - cached_at) < _RESOLVE_TOKEN_CACHE_TTL_S:
                    return cached_token
    with _provider_state_transaction("nous") as (
        auth_store,
        state,
        state_source_path,
    ):

        if not state:
            raise _nous_err("Hermes is not logged into Nous Portal.", relogin=True)

        # HERMES_PORTAL_BASE_URL / NOUS_PORTAL_BASE_URL is the trusted
        # operator/deployment override (mirrors NOUS_INFERENCE_BASE_URL) and
        # must win OUTRIGHT — including over a stored value — and bypass the
        # host allowlist entirely, since the allowlist exists to reject an
        # untrusted network-provided value, not one the operator configured.
        # Only fall through to the stored/default value + allowlist gate when
        # no override is set.
        env_portal_override = _nous_portal_env_override()
        if env_portal_override:
            portal_base_url = env_portal_override.rstrip("/")
        else:
            portal_base_url = (
                _optional_base_url(state.get("portal_base_url"))
                or DEFAULT_NOUS_PORTAL_URL
            ).rstrip("/")

            parsed_portal_url = urlparse(portal_base_url)
            if parsed_portal_url.hostname and parsed_portal_url.hostname not in _NOUS_PORTAL_ALLOWED_HOSTS:
                logger.warning(
                    "auth: ignoring invalid portal_base_url %r (host %r not in allowlist), using default",
                    portal_base_url, parsed_portal_url.hostname,
                )
                portal_base_url = DEFAULT_NOUS_PORTAL_URL

        client_id = str(state.get("client_id") or DEFAULT_NOUS_CLIENT_ID)
        verify = _resolve_verify(insecure=insecure, ca_bundle=ca_bundle, auth_state=state)

        with _nous_shared_store_lock(timeout_seconds=max(timeout_seconds + 5.0, AUTH_LOCK_TIMEOUT_SECONDS)):
            merged_shared = _merge_shared_nous_oauth_state(state)
            access_token = state.get("access_token")
            refresh_token = state.get("refresh_token")
            if not isinstance(access_token, str) or not access_token:
                raise _nous_err(
                    "No access token found for Nous Portal login.",
                    relogin=True,
                )

            if not _is_expiring(state.get("expires_at"), refresh_skew_seconds):
                if merged_shared:
                    _save_provider_state_to_source(auth_store, "nous", state, state_source_path)
                # Populate the memo on the valid-token fast path too: the
                # startup burst usually finds a *valid* token, but each
                # check_fn call still pays two cross-process file locks and
                # state reads to reach this return. The token has at least
                # refresh_skew_seconds (>= 120s) of life here, so a 5s memo
                # can never serve an expired token.
                if not insecure and ca_bundle is None:
                    with _RESOLVE_TOKEN_CACHE_LOCK:
                        _RESOLVE_TOKEN_CACHE = (time.monotonic(), access_token)
                return access_token

            if not isinstance(refresh_token, str) or not refresh_token:
                raise _nous_err(
                    "Session expired and no refresh token is available.",
                    relogin=True,
                )

            timeout = httpx.Timeout(timeout_seconds if timeout_seconds else 15.0)
            with httpx.Client(
                timeout=timeout,
                headers={"Accept": "application/json"},
                verify=verify,
            ) as client:
                refreshed = _refresh_nous_or_quarantine(
                    client=client,
                    auth_store=auth_store,
                    state=state,
                    portal_base_url=portal_base_url,
                    client_id=client_id,
                    refresh_token=refresh_token,
                    reason="managed_access_token_refresh_failure",
                    persist=lambda: _save_provider_state_to_source(
                        auth_store, "nous", state, state_source_path
                    ),
                )

            _apply_nous_refreshed_tokens(state, refreshed, refresh_token)
            state["portal_base_url"] = portal_base_url
            state["client_id"] = client_id
            state["tls"] = _tls_state_from_verify(verify)
            _save_provider_state_to_source(auth_store, "nous", state, state_source_path)
            _write_shared_nous_state(state)
            resolved = state["access_token"]
            if not insecure and ca_bundle is None:
                with _RESOLVE_TOKEN_CACHE_LOCK:
                    _RESOLVE_TOKEN_CACHE = (time.monotonic(), resolved)
            return resolved


# =============================================================================
# Status helpers
# =============================================================================


# ── Process-level memo for get_nous_auth_status() ──
# get_nous_auth_status() validates state by calling resolve_nous_runtime_credentials(),
# which does a synchronous OAuth refresh POST to portal.nousresearch.com. That can take
# ~350ms even on the failure path, and read-only UI surfaces (`hermes tools`, status panels,
# subscription-feature checks) call it many times per render — `hermes tools` → "All Platforms"
# was firing the refresh ~31× during one menu paint, racking up >13s of HTTP and burning
# single-use refresh tokens. Cache the snapshot for a few seconds, keyed on the auth.json
# path + mtime so that profile switches do not share a process memo and
# `hermes auth login/logout/add/remove` invalidate naturally on the next call.
_NOUS_AUTH_STATUS_CACHE_TTL = 15.0  # seconds
_nous_auth_status_cache: Optional[Tuple[float, str, Optional[float], Dict[str, Any]]] = None

# mtime-keyed memo for _load_global_auth_store(): (path, mtime_ns, store).
# Same invalidation contract as _nous_auth_status_cache — the global auth
# file changes only when a global-scope auth write touches it.
_global_auth_store_cache: Optional[Tuple[str, int, Dict[str, Any]]] = None


def _auth_file_cache_key() -> Tuple[str, Optional[float]]:
    auth_file = _auth_file_path()
    try:
        auth_file_key = str(auth_file.resolve(strict=False))
    except Exception:
        auth_file_key = str(auth_file)
    try:
        return auth_file_key, auth_file.stat().st_mtime
    except FileNotFoundError:
        return auth_file_key, None
    except Exception:
        return auth_file_key, None


def invalidate_nous_auth_status_cache() -> None:
    """Clear the get_nous_auth_status() process-level memo.

    Call from code paths that mutate Nous auth state without going through
    ``resolve_nous_runtime_credentials()`` (e.g. tests). Login/logout touch auth.json, so the
    mtime check invalidates them automatically; this is the belt-and-braces option.
    """
    global _nous_auth_status_cache
    _nous_auth_status_cache = None


def get_nous_auth_status() -> Dict[str, Any]:
    """Status snapshot for Nous auth.

    Prefer the auth-store provider state, because that is the live source of truth for refresh
    operations. When provider state exists, validate it by resolving runtime credentials so revoked
    refresh sessions do not show up as a healthy login.

    The returned snapshot is memoised for ~15s keyed on the auth.json mtime, so menu/status surfaces
    that ask repeatedly don't trigger one refresh POST per call. Login/logout flows write to
    auth.json and therefore invalidate the cache automatically; tests can also call
    ``invalidate_nous_auth_status_cache()`` explicitly.
    """
    global _nous_auth_status_cache
    now = time.monotonic()
    auth_file_key, mtime = _auth_file_cache_key()
    cached = _nous_auth_status_cache
    if cached is not None:
        cached_at, cached_auth_file_key, cached_mtime, cached_status = cached
        if (
            cached_auth_file_key == auth_file_key
            and cached_mtime == mtime
            and (now - cached_at) < _NOUS_AUTH_STATUS_CACHE_TTL
        ):
            return dict(cached_status)

    status = _compute_nous_auth_status()
    _nous_auth_status_cache = (now, auth_file_key, mtime, dict(status))
    return status


def _codex_pool_rate_limited_status() -> Optional[Dict[str, Any]]:
    rate_limit = _codex_pool_rate_limit_status()
    if not rate_limit:
        return None
    return {
        "logged_in": True,
        "auth_store": str(_auth_file_path()),
        "last_refresh": rate_limit.get("last_refresh"),
        "auth_mode": "chatgpt",
        "source": f"pool:{rate_limit.get('label') or 'unknown'}",
        "rate_limited": True,
        "error_code": CODEX_RATE_LIMITED_CODE,
        "error": (
            rate_limit.get("message")
            or "Codex provider quota exhausted; retry after the usage limit resets."
        ),
        "reset_at": rate_limit.get("reset_at"),
    }


def get_codex_auth_status() -> Dict[str, Any]:
    """Status snapshot for Codex auth (pool first, then legacy provider state)."""
    return _pool_first_oauth_status(
        "openai-codex",
        is_expiring=_codex_access_token_is_expiring,
        auth_mode="chatgpt",
        resolve=resolve_codex_runtime_credentials,
        on_pool_miss=_codex_pool_rate_limited_status,
    )


def get_xai_oauth_auth_status() -> Dict[str, Any]:
    return _pool_first_oauth_status(
        "xai-oauth",
        is_expiring=_xai_access_token_is_expiring,
        # Display/telemetry only. Device-code is the only xAI OAuth flow, so report it
        # unconditionally (auth.json may still carry a legacy ``oauth_pkce`` label).
        auth_mode="oauth_device_code",
        resolve=resolve_xai_oauth_runtime_credentials,
    )


def _provider_env_base_url(pconfig: ProviderConfig) -> str:
    return os.getenv(pconfig.base_url_env_var, "").strip() if pconfig.base_url_env_var else ""


def get_api_key_provider_status(provider_id: str) -> Dict[str, Any]:
    """Status snapshot for API-key providers (z.ai, Kimi, MiniMax)."""
    pconfig = PROVIDER_REGISTRY.get(provider_id)
    if not pconfig or pconfig.auth_type != "api_key":
        return {"configured": False}

    # Keyless providers (opencode-free) are served anonymously: no credential
    # exists, so every install counts as configured/logged in. Derived from
    # the HermesOverlay keyless flag — the same source the provider catalog
    # and GUI contract tests use.
    try:
        from hermes_cli.providers import HERMES_OVERLAYS
        _overlay = HERMES_OVERLAYS.get(provider_id)
    except Exception:
        _overlay = None
    if _overlay is not None and getattr(_overlay, "keyless", False):
        return {
            "configured": True,
            "provider": provider_id,
            "name": pconfig.name,
            "key_source": "keyless",
            "base_url": pconfig.inference_base_url,
            "logged_in": True,
        }

    api_key, key_source = _resolve_api_key_provider_secret(provider_id, pconfig)
    env_url = _provider_env_base_url(pconfig)

    if provider_id in {"kimi-coding", "kimi-coding-cn"}:
        base_url = _resolve_kimi_base_url(api_key, pconfig.inference_base_url, env_url)
    elif env_url:
        base_url = env_url
    else:
        base_url = pconfig.inference_base_url

    if provider_id == "actual":
        base_url = normalize_actual_base_url(base_url)

    actual_local_noauth = (
        provider_id == "actual"
        and not api_key
        and is_actual_local_base_url(base_url)
    )

    return {
        "configured": bool(api_key) or actual_local_noauth,
        "provider": provider_id,
        "name": pconfig.name,
        "key_source": key_source or ("local-offline" if actual_local_noauth else ""),
        "base_url": base_url,
        "logged_in": bool(api_key) or actual_local_noauth,  # compat with OAuth status shape
    }


def _external_process_auth_evidence(provider_id: str) -> tuple[bool, Optional[str]]:
    """Best-effort POSITIVE evidence that an external-process provider's CLI
    is authenticated.

    Returns ``(verified, source)``. ``verified`` is only ever True on hard
    evidence (a supported env token, or a known on-disk credential store).
    False means "not verifiable from here", NOT "signed out" — the Copilot
    CLI may hold its session in an OS keychain Hermes can't read. Callers
    must therefore treat False as unknown, never as proof of absence.

    Deliberately subprocess-free: this runs from status endpoints and pickers,
    and spawning ``gh auth token`` there re-creates the cold-start stall
    (#60800) that copilot_auth.py works to avoid.
    """
    if provider_id != "copilot-acp":
        return False, None
    # 1. Supported env tokens — the same vars the Copilot CLI itself honors.
    try:
        from hermes_cli.copilot_auth import COPILOT_ENV_VARS, validate_copilot_token
        for env_var in COPILOT_ENV_VARS:
            val = os.getenv(env_var, "").strip()
            if val and validate_copilot_token(val)[0]:
                return True, f"env: {env_var}"
    except Exception as exc:
        logger.debug("copilot-acp env token evidence check failed: %s", exc)
    # 2. The Copilot CLI's own plaintext token store (~/.copilot/config.json,
    #    written by `copilot login` when no OS keychain is available). The file
    #    is JSONC — strip //-comment lines before parsing.
    try:
        cli_config = os.path.expanduser("~/.copilot/config.json")
        if os.path.isfile(cli_config):
            with open(cli_config, "r", encoding="utf-8", errors="ignore") as fh:
                raw = "\n".join(
                    line for line in fh.read().splitlines()
                    if not line.lstrip().startswith("//")
                )
            data = json.loads(raw) if raw.strip() else {}
            tokens = data.get("copilotTokens")
            if isinstance(tokens, dict) and any(
                isinstance(v, str) and v.strip() for v in tokens.values()
            ):
                return True, "~/.copilot/config.json"
    except Exception as exc:
        logger.debug("copilot-acp CLI config evidence check failed: %s", exc)
    # 3. Known on-disk GitHub Copilot credential stores (the same locations
    #    models.py already fingerprints as external credential files).
    for cred_path in (
        "~/.config/github-copilot/hosts.json",
        "~/.config/github-copilot/apps.json",
    ):
        try:
            expanded = os.path.expanduser(cred_path)
            if os.path.isfile(expanded) and os.path.getsize(expanded) > 2:
                return True, cred_path
        except OSError:
            continue
    return False, None


def _external_process_spec(
    pconfig: ProviderConfig,
) -> tuple[str, List[str], str, Optional[str], tuple[str, ...]]:
    """``(command, args, base_url, resolved_command, command_env_vars)`` for a
    subprocess-backed (ACP) provider.

    How to launch the CLI comes from the provider's own profile, so a provider
    shipped outside this tree describes its binary/args instead of inheriting
    another vendor's. copilot-acp's values live in its profile, which is why
    HERMES_COPILOT_ACP_COMMAND / COPILOT_CLI_PATH / HERMES_COPILOT_ACP_ARGS
    keep working unchanged.
    """
    base_url = os.getenv(pconfig.base_url_env_var, "").strip() if pconfig.base_url_env_var else ""
    if not base_url:
        base_url = pconfig.inference_base_url

    try:
        from providers import get_provider_profile as _get_provider_profile

        profile = _get_provider_profile(pconfig.id)
    except Exception:
        profile = None

    command_env_vars = tuple(getattr(profile, "process_command_env_vars", ()) or ())
    args_env_var = str(getattr(profile, "process_args_env_var", "") or "")

    command = next((v for v in (os.getenv(var, "").strip() for var in command_env_vars) if v), "")
    if not command:
        command = str(getattr(profile, "process_command", "") or "")
    raw_args = os.getenv(args_env_var, "").strip() if args_env_var else ""
    args = shlex.split(raw_args) if raw_args else list(getattr(profile, "process_args", ()) or [])
    resolved_command = shutil.which(command) if command else None
    return command, args, base_url, resolved_command, command_env_vars


def get_external_process_provider_status(provider_id: str) -> Dict[str, Any]:
    """Status snapshot for providers that run a local subprocess.

    ``configured``/``logged_in`` stay structural (the executable resolves or a
    TCP endpoint is set) because the spawned subprocess owns its real auth.
    ``auth_verified``/``auth_source`` carry positive credential evidence when
    Hermes can actually see some — absence of evidence is not absence of auth.
    """
    pconfig = PROVIDER_REGISTRY.get(provider_id)
    if not pconfig or pconfig.auth_type != "external_process":
        return {"configured": False}

    command, args, base_url, resolved_command, _ = _external_process_spec(pconfig)
    available = bool(resolved_command or base_url.startswith("acp+tcp://"))
    auth_verified, auth_source = _external_process_auth_evidence(provider_id)
    return {
        "configured": available,
        "provider": provider_id,
        "name": pconfig.name,
        "command": command,
        "args": args,
        "resolved_command": resolved_command,
        "base_url": base_url,
        "logged_in": available,
        "auth_verified": auth_verified,
        "auth_source": auth_source,
    }


def _get_aws_sdk_auth_status(target: str) -> Dict[str, Any]:
    """AWS SDK providers (Bedrock) — check via boto3 credential chain."""
    try:
        from agent.bedrock_adapter import has_aws_credentials
        return {"logged_in": has_aws_credentials(), "provider": target}
    except ImportError:
        return {"logged_in": False, "provider": target, "error": "boto3 not installed"}


def get_auth_status(provider_id: Optional[str] = None) -> Dict[str, Any]:
    """Generic auth status dispatcher."""
    target = (provider_id or get_active_provider() or "").strip().lower()
    if not target:
        return {"logged_in": False}
    # Bespoke status builders win over the auth_type-keyed fallbacks. Looked up
    # at call time so tests that patch ``hermes_cli.auth.get_*_auth_status`` still apply.
    bespoke: Dict[str, Callable[[], Dict[str, Any]]] = {
        "spotify": get_spotify_auth_status,
        "nous": get_nous_auth_status,
        "openai-codex": get_codex_auth_status,
        "xai-oauth": get_xai_oauth_auth_status,
        "qwen-oauth": get_qwen_auth_status,
        "minimax-oauth": get_minimax_oauth_auth_status,
        "azure-foundry": _get_azure_foundry_auth_status,
    }
    if target in bespoke:
        return bespoke[target]()
    # External-process providers (copilot-acp today; other ACP backends tomorrow)
    # dispatch on auth_type, not a hardcoded slug, so every provider of this
    # class gets a real status instead of the ``{"logged_in": False}`` fallthrough.
    by_auth_type: Dict[str, Callable[[str], Dict[str, Any]]] = {
        "external_process": get_external_process_provider_status,
        "api_key": get_api_key_provider_status,
        "aws_sdk": _get_aws_sdk_auth_status,
    }
    pconfig = PROVIDER_REGISTRY.get(target)
    if pconfig and pconfig.auth_type in by_auth_type:
        return by_auth_type[pconfig.auth_type](target)
    return {"logged_in": False}


def _get_azure_foundry_auth_status() -> Dict[str, Any]:
    """Return structural auth status for Azure Foundry.

    * ``auth_mode == "entra_id"`` AND ``azure-identity`` is importable (we do NOT mint a token here;
    ``hermes doctor`` runs the live probe and reports whether the credential chain can acquire one).
    * ``auth_mode == "api_key"`` (default) AND ``AZURE_FOUNDRY_API_KEY`` is set with a usable value.

    Never invokes the Entra credential chain — keeps CLI startup latency flat regardless of token-
    service / az login state.
    """
    info: Dict[str, Any] = {"provider": "azure-foundry"}
    try:
        from hermes_cli.config import load_config, get_env_value_prefer_dotenv
        cfg = load_config()
    except Exception:
        cfg = {}

    model_cfg = cfg.get("model") if isinstance(cfg, dict) else None
    auth_mode = "api_key"
    base_url = ""
    if isinstance(model_cfg, dict):
        auth_mode = str(model_cfg.get("auth_mode") or "api_key").strip().lower() or "api_key"
        base_url = str(model_cfg.get("base_url") or "").strip()
    info["auth_mode"] = auth_mode
    info["base_url"] = base_url

    if auth_mode == "entra_id":
        try:
            from agent.azure_identity_adapter import (
                EntraIdentityConfig,
                SCOPE_AI_AZURE_DEFAULT,
                has_azure_identity_installed,
            )
            installed = has_azure_identity_installed()
            entra_cfg = {}
            if isinstance(model_cfg, dict) and isinstance(model_cfg.get("entra"), dict):
                entra_cfg = model_cfg["entra"]
            identity_config = EntraIdentityConfig.from_dict(
                entra_cfg,
                default_scope=SCOPE_AI_AZURE_DEFAULT,
            )
            info["azure_identity_installed"] = installed
            info["scope"] = identity_config.scope
            info["credential_probe"] = "not_run"
            info["credential_verified"] = False
            info["logged_in"] = bool(installed)
            if not installed:
                info["hint"] = (
                    "azure-identity not installed. Install with: "
                    "pip install azure-identity  (or rely on Hermes' "
                    "lazy-install at first use)."
                )
            else:
                info["hint"] = (
                    "azure-identity is installed; live credential validation "
                    "is skipped here. Run `hermes doctor` to verify token acquisition."
                )
            return info
        except Exception as exc:
            info["logged_in"] = False
            info["error"] = f"azure-identity check failed: {exc}"
            return info

    # api_key mode (default)
    try:
        api_key = get_env_value_prefer_dotenv("AZURE_FOUNDRY_API_KEY") or ""
    except Exception:
        api_key = os.getenv("AZURE_FOUNDRY_API_KEY", "")
    info["logged_in"] = has_usable_secret(api_key)
    return info


def resolve_api_key_provider_credentials(provider_id: str) -> Dict[str, Any]:
    """Resolve API key and base URL for an API-key provider."""
    pconfig = PROVIDER_REGISTRY.get(provider_id)
    if not pconfig or pconfig.auth_type != "api_key":
        raise AuthError(
            f"Provider '{provider_id}' is not an API-key provider.",
            provider=provider_id,
            code="invalid_provider",
        )

    api_key, key_source = _resolve_api_key_provider_secret(provider_id, pconfig)

    # No-auth LM Studio: substitute a placeholder so runtime / auxiliary_client
    # see the local server as configured. doctor still reports unconfigured
    # because get_api_key_provider_status uses the raw secret resolver.
    if not api_key and provider_id == "lmstudio":
        api_key = LMSTUDIO_NOAUTH_PLACEHOLDER
        key_source = key_source or "default"

    env_url = _provider_env_base_url(pconfig)

    if provider_id in {"kimi-coding", "kimi-coding-cn"}:
        base_url = _resolve_kimi_base_url(api_key, pconfig.inference_base_url, env_url)
    elif provider_id == "zai":
        base_url = _resolve_zai_base_url(api_key, pconfig.inference_base_url, env_url)
    elif provider_id == "copilot":
        # Resolve the Copilot API base URL from the token-exchange response
        # (endpoints.api, with a proxy-ep fallback), which is authoritative
        # for Enterprise / proxied accounts. Falls back to the registry
        # default and is guarded non-empty below so chat inference never
        # resolves an empty base URL (#50252).
        base_url = env_url.rstrip("/") if env_url else pconfig.inference_base_url
        try:
            from hermes_cli.copilot_auth import (
                resolve_copilot_token,
                get_copilot_api_token,
            )
            raw_token, _ = resolve_copilot_token()
            if raw_token:
                _, resolved = get_copilot_api_token(raw_token)
                resolved = (resolved or "").strip()
                if resolved:
                    base_url = resolved
        except Exception as exc:
            logger.debug("Copilot base URL resolution fell back to default: %s", exc)
    elif env_url:
        base_url = env_url.rstrip("/")
    else:
        base_url = pconfig.inference_base_url

    if provider_id == "lmstudio":
        base_url = _normalize_lmstudio_runtime_base_url(base_url)

    if provider_id == "actual":
        base_url = normalize_actual_base_url(base_url)

    # Last-resort guard: an API-key provider must never hand back an empty
    # base URL (a set-but-empty COPILOT_API_BASE_URL or similar env override
    # otherwise wedges chat inference — #50252).
    if not _nonempty_str(base_url):
        base_url = pconfig.inference_base_url

    if not api_key and provider_id == "actual" and is_actual_local_base_url(base_url):
        api_key = ACTUAL_LOCAL_NOAUTH_PLACEHOLDER
        key_source = key_source or "local-offline"

    return {
        "provider": provider_id,
        "api_key": api_key,
        "base_url": base_url.rstrip("/"),
        "source": key_source or "default",
    }


def resolve_external_process_provider_credentials(provider_id: str) -> Dict[str, Any]:
    """Resolve runtime details for local subprocess-backed providers."""
    pconfig = PROVIDER_REGISTRY.get(provider_id)
    if not pconfig or pconfig.auth_type != "external_process":
        raise AuthError(
            f"Provider '{provider_id}' is not an external-process provider.",
            provider=provider_id,
            code="invalid_provider",
        )

    command, args, base_url, resolved_command, command_env_vars = _external_process_spec(pconfig)
    if not resolved_command and not base_url.startswith("acp+tcp://"):
        _hint = (
            " or set " + "/".join(command_env_vars) if command_env_vars else ""
        )
        raise AuthError(
            f"Could not find the '{provider_id}' CLI command "
            f"'{command or '(none configured)'}'. Install it{_hint}.",
            provider=provider_id,
            code="missing_external_process_cli",
        )

    return {
        "provider": provider_id,
        # Placeholder credential: the subprocess owns real auth. Keyed on the
        # provider id so each external-process provider gets a distinct value.
        "api_key": pconfig.id or provider_id,
        "base_url": base_url.rstrip("/"),
        "command": resolved_command or command,
        "args": args,
        "source": "process",
    }


# =============================================================================
# CLI Commands — login / logout
# =============================================================================

def _update_config_for_provider(
    provider_id: str,
    inference_base_url: str,
    default_model: Optional[str] = None,
) -> Path:
    """Update config.yaml and auth.json to reflect the active provider.

    When *default_model* is provided the function also writes it as the ``model.default`` value.
    This prevents a race condition where the gateway (which re-reads config per-message) picks up
    the new provider before the caller has finished model selection, resulting in a mismatched
    model/provider (e.g.
    """
    # Set active_provider in auth.json so auto-resolution picks this provider
    with _auth_store_lock():
        auth_store = _load_auth_store()
        auth_store["active_provider"] = provider_id
        _save_auth_store(auth_store)

    # Update config.yaml model section
    config_path = get_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    require_readable_config_before_write(config_path)

    config = read_raw_config()

    current_model = config.get("model")
    if isinstance(current_model, dict):
        model_cfg = dict(current_model)
    elif _nonempty_str(current_model):
        model_cfg = {"default": current_model.strip()}
    else:
        model_cfg = {}

    model_cfg["provider"] = provider_id
    if inference_base_url and inference_base_url.strip():
        model_cfg["base_url"] = inference_base_url.rstrip("/")
    else:
        # Clear stale base_url to prevent contamination when switching providers
        model_cfg.pop("base_url", None)

    # Clear stale endpoint credentials left over from a previous custom provider.
    # Built-in providers resolve credentials from env/auth state, not inline
    # model.api_key.
    from hermes_cli.config import clear_model_endpoint_credentials

    clear_model_endpoint_credentials(model_cfg)

    # When switching to a non-OpenRouter provider, ensure model.default is
    # valid for the new provider.  An OpenRouter-formatted name like
    # "anthropic/claude-opus-4.6" will fail on direct-API providers.
    if default_model:
        cur_default = model_cfg.get("default", "")
        if not cur_default or "/" in cur_default:
            model_cfg["default"] = default_model

    config["model"] = model_cfg

    atomic_yaml_write(config_path, config, sort_keys=False)
    return config_path


def _get_config_provider() -> Optional[str]:
    """Return model.provider from config.yaml, normalized, if present."""
    try:
        config = read_raw_config()
    except Exception:
        return None
    if not config:
        return None
    model = config.get("model")
    if not isinstance(model, dict):
        return None
    provider = model.get("provider")
    if not isinstance(provider, str):
        return None
    provider = provider.strip().lower()
    return provider or None


def _config_provider_matches(provider_id: Optional[str]) -> bool:
    """Return True when config.yaml currently selects *provider_id*."""
    if not provider_id:
        return False
    return _get_config_provider() == provider_id.strip().lower()


def _should_reset_config_provider_on_logout(provider_id: Optional[str]) -> bool:
    """Return True when logout should reset the model provider config."""
    if not provider_id:
        return False
    normalized = provider_id.strip().lower()
    return normalized in PROVIDER_REGISTRY and _config_provider_matches(normalized)


def _logout_default_provider_from_config() -> Optional[str]:
    """Fallback logout target when auth.json has no active provider.

    That left users stuck when auth state had already been cleared but config.yaml still selected an
    OAuth provider such as openai-codex for the agent model: there was no active auth provider to
    target, so logout printed "No provider is currently logged in" and never reset model.provider.
    """
    provider = _get_config_provider()
    if provider in {"nous", "openai-codex", "xai-oauth"}:
        return provider
    return None


def _reset_config_provider() -> Path:
    """Reset config.yaml provider back to auto after logout."""
    config_path = get_config_path()
    if not config_path.exists():
        return config_path
    require_readable_config_before_write(config_path)

    config = read_raw_config()
    if not config:
        return config_path

    model = config.get("model")
    if isinstance(model, dict):
        model["provider"] = "auto"
        if "base_url" in model:
            model["base_url"] = OPENROUTER_BASE_URL
    atomic_yaml_write(config_path, config, sort_keys=False)
    return config_path


def login_command(args) -> None:
    """Deprecated: use 'hermes model' or 'hermes setup' instead."""
    print("The 'hermes login' command has been removed.")
    print("Use 'hermes auth' to manage credentials,")
    print("'hermes model' to select a provider, or 'hermes setup' for full setup.")
    raise SystemExit(0)


# ==================== MiniMax Portal OAuth ====================


def get_minimax_oauth_auth_status() -> Dict[str, Any]:
    """Return auth status dict for MiniMax OAuth provider."""
    state = get_provider_auth_state("minimax-oauth")
    if not state or not state.get("access_token"):
        return {"logged_in": False, "provider": "minimax-oauth"}
    try:
        expires_at = datetime.fromisoformat(state.get("expires_at", "")).timestamp()
        token_valid = (expires_at - time.time()) > 0
    except Exception:
        token_valid = bool(state.get("access_token"))
    return {
        "logged_in": token_valid,
        "provider": "minimax-oauth",
        "region": state.get("region", "global"),
        "expires_at": state.get("expires_at"),
    }


def logout_command(args) -> None:
    """Clear auth state for a provider."""
    provider_id = getattr(args, "provider", None)

    if provider_id and not is_known_auth_provider(provider_id):
        print(f"Unknown provider: {provider_id}")
        raise SystemExit(1)

    active = get_active_provider()
    target = provider_id or active or _logout_default_provider_from_config()

    if not target:
        print("No provider is currently logged in.")
        return

    should_reset_config = _should_reset_config_provider_on_logout(target)
    provider_name = get_auth_provider_display_name(target)

    if clear_provider_auth(target) or should_reset_config:
        if should_reset_config:
            _reset_config_provider()
        print(f"Logged out of {provider_name}.")
        if should_reset_config and os.getenv("OPENROUTER_API_KEY"):
            print("Hermes will use OpenRouter for inference.")
        elif should_reset_config:
            print("Run `hermes model` or configure an API key to use Hermes.")
        else:
            print("Model provider configuration was unchanged.")
    else:
        print(f"No auth state found for {provider_name}.")
