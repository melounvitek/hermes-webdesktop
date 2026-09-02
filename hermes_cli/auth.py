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
import ssl
import stat
import sys
import base64
import hashlib
import subprocess
import threading
import time
import uuid
import webbrowser

from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, FrozenSet, Iterable, List, Optional, Tuple
from urllib.parse import parse_qs, urlencode, urlparse

from hermes_cli.config import (
    get_hermes_home,
    get_config_path,
    read_raw_config,
    require_readable_config_before_write,
)
from hermes_constants import OPENROUTER_BASE_URL, secure_parent_dir
from agent.credential_persistence import sanitize_borrowed_credential_payload
from utils import atomic_replace, atomic_yaml_write, env_float, is_truthy_value
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

# Kimi Code (kimi.com/code) issues keys prefixed "sk-kimi-" that only work
# on api.kimi.com/coding.  Legacy keys from platform.moonshot.ai work on
# api.moonshot.ai/v1 (the old default).  Auto-detect when user hasn't set
# KIMI_BASE_URL explicitly.
#
# Note: the base URL intentionally has NO /v1 suffix.  The /coding endpoint
# speaks the Anthropic Messages protocol, and the anthropic SDK appends
# "/v1/messages" internally — so "/coding" + SDK suffix → "/coding/v1/messages"
# (the correct target). Using "/coding/v1" here would produce
# "/coding/v1/v1/messages" (a 404).
KIMI_CODE_BASE_URL = "https://api.kimi.com/coding"


def _resolve_kimi_base_url(api_key: str, default_url: str, env_override: str) -> str:
    """Return the correct Kimi base URL based on the API key prefix.

    If the user has explicitly set KIMI_BASE_URL, that always wins. Otherwise, sk-kimi- prefixed
    keys route to api.kimi.com/coding/v1.
    """
    if env_override:
        return env_override
    # No key → nothing to infer from.  Return default without inspecting.
    if not api_key:
        return default_url
    if api_key.startswith("sk-kimi-"):
        return KIMI_CODE_BASE_URL
    return default_url


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

ZAI_ENDPOINTS = [
    # (id, base_url, probe_models, label)
    ("global",        "https://api.z.ai/api/paas/v4",        ["glm-5"],   "Global"),
    ("cn",            "https://open.bigmodel.cn/api/paas/v4", ["glm-5"],   "China"),
    ("coding-global", "https://api.z.ai/api/coding/paas/v4",  ["glm-5.3", "glm-5.3-flash", "glm-5.2", "glm-5.1", "glm-5v-turbo", "glm-4.7"], "Global (Coding Plan)"),
    ("coding-cn",     "https://open.bigmodel.cn/api/coding/paas/v4", ["glm-5.3", "glm-5.3-flash", "glm-5.2", "glm-5.1", "glm-5v-turbo", "glm-4.7"], "China (Coding Plan)"),
]


def _probe_single_zai_endpoint(
    api_key: str, endpoint: tuple, timeout: float,
) -> Optional[Dict[str, str]]:
    """Probe a single Z.AI endpoint. Returns endpoint info dict or None.

    Preserves the per-endpoint candidate-model loop: endpoints carry a ``probe_models`` LIST and
    each model is tried in order until one succeeds (some plans only accept newer/older GLM slugs).
    """
    ep_id, base_url, probe_models, label = endpoint
    for model in probe_models:
        try:
            resp = httpx.post(
                f"{base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "stream": False,
                    "max_tokens": 1,
                    "messages": [{"role": "user", "content": "ping"}],
                },
                timeout=timeout,
            )
            if resp.status_code == 200:
                logger.debug("Z.AI endpoint probe: %s (%s) model=%s OK", ep_id, base_url, model)
                return {
                    "id": ep_id,
                    "base_url": base_url,
                    "model": model,
                    "label": label,
                }
            logger.debug("Z.AI endpoint probe: %s model=%s returned %s", ep_id, model, resp.status_code)
        except Exception as exc:
            logger.debug("Z.AI endpoint probe: %s model=%s failed: %s", ep_id, model, exc)
    return None


def detect_zai_endpoint(api_key: str, timeout: float = 8.0) -> Optional[Dict[str, str]]:
    """Probe z.ai endpoints in parallel to find one that accepts this API key.

    Returns {"id": ..., "base_url": ..., "model": ..., "label": ...} for the first working endpoint
    (in ZAI_ENDPOINTS priority order), or None if all fail. For endpoints with multiple candidate
    models, each worker tries its endpoint's models in order and returns the first that succeeds.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # No `with` block: a context manager would join ALL probe threads on
    # exit, defeating the early return below. shutdown(wait=False) lets the
    # surviving daemon-style probes drain in the background instead of
    # blocking the caller on slow/unreachable endpoints.
    pool = ThreadPoolExecutor(max_workers=len(ZAI_ENDPOINTS))
    try:
        futures = {
            pool.submit(_probe_single_zai_endpoint, api_key, ep, timeout): ep[0]
            for ep in ZAI_ENDPOINTS
        }
        by_id = {ep_id: f for f, ep_id in futures.items()}
        results: Dict[str, Dict[str, str]] = {}
        for future in as_completed(futures):
            ep_id = futures[future]
            try:
                result = future.result()
                if result is not None:
                    results[ep_id] = result
            except Exception:
                pass
            # Early exit in PRIORITY order: walk endpoints highest-priority
            # first; if one has succeeded and every higher-priority probe
            # has already finished (without success), no later completion
            # can win — return now instead of waiting out slow endpoints
            # (main's sequential loop also stopped at first success).
            for ep in ZAI_ENDPOINTS:
                if not by_id[ep[0]].done():
                    break  # a higher-priority probe is still in flight
                if ep[0] in results:
                    return results[ep[0]]

        # All probes finished: first match in priority order, if any.
        for ep in ZAI_ENDPOINTS:
            if ep[0] in results:
                return results[ep[0]]
        return None
    finally:
        pool.shutdown(wait=False)


def _resolve_zai_base_url(api_key: str, default_url: str, env_override: str) -> str:
    """Return the correct Z.AI base URL by probing endpoints.

    If the user has explicitly set GLM_BASE_URL, that always wins. Otherwise, probe the candidate
    endpoints to find one that accepts the key. The detected endpoint is cached in provider state
    (auth.json) keyed on a hash of the API key so subsequent starts skip the probe.
    """
    if env_override:
        return env_override

    # No API key set → don't probe (would fire N×M HTTPS requests with an
    # empty Bearer token, all returning 401).  This path is hit during
    # auxiliary-client auto-detection when the user has no Z.AI credentials
    # at all — the caller discards the result immediately, so the probe is
    # pure latency for every AIAgent construction.
    if not api_key:
        return default_url

    # Check provider-state cache for a previously-detected endpoint.
    auth_store = _load_auth_store()
    state = _load_provider_state(auth_store, "zai") or {}
    cached = state.get("detected_endpoint")
    if isinstance(cached, dict) and cached.get("base_url"):
        key_hash = cached.get("key_hash", "")
        if key_hash == hashlib.sha256(api_key.encode()).hexdigest()[:16]:
            logger.debug("Z.AI: using cached endpoint %s", cached["base_url"])
            return cached["base_url"]

    # Probe — may take up to ~8s per endpoint.
    detected = detect_zai_endpoint(api_key)
    if detected and detected.get("base_url"):
        # Persist the detection result keyed on the API key hash.
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()[:16]
        detected_endpoint = {
            "base_url": detected["base_url"],
            "endpoint_id": detected.get("id", ""),
            "model": detected.get("model", ""),
            "label": detected.get("label", ""),
            "key_hash": key_hash,
        }
        # Persist failure (disk full, permissions, lock timeout) must not
        # break resolution — detection already succeeded; worst case the
        # next start re-probes.
        try:
            with _auth_store_lock():
                # Reload auth_store under lock to avoid overwriting concurrent changes
                auth_store = _load_auth_store()
                state_under_lock = _load_provider_state(auth_store, "zai") or {}
                state_under_lock["detected_endpoint"] = detected_endpoint
                # set_active=False: this runs from credential-pool env seeding
                # (agent/credential_pool.py) for ANY user with a Z.AI key in env,
                # and caching a probe result must not flip their active provider.
                _store_provider_state(auth_store, "zai", state_under_lock, set_active=False)
                _save_auth_store(auth_store)
        except Exception as exc:
            logger.warning("Z.AI: could not persist detected endpoint (%s); will re-probe next start", exc)
        logger.info("Z.AI: auto-detected endpoint %s (%s)", detected["label"], detected["base_url"])
        return detected["base_url"]

    logger.debug("Z.AI: probe failed, falling back to default %s", default_url)
    return default_url


def _normalize_lmstudio_runtime_base_url(base_url: str) -> str:
    """Return the OpenAI-compatible LM Studio runtime base URL.

    LM Studio's native management API lives under ``/api/v1`` while its OpenAI-compatible chat
    endpoint lives under ``/v1``. Users often paste either form into ``LM_BASE_URL`` or
    ``model.base_url``; normalize before the OpenAI SDK appends ``/chat/completions``.
    """
    root = str(base_url or "").strip().rstrip("/")
    for suffix in ("/api/v1", "/api", "/v1"):
        if root.endswith(suffix):
            root = root[: -len(suffix)].rstrip("/")
            break
    return (root or "http://127.0.0.1:1234") + "/v1"


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


def _parse_retry_after_seconds(headers: Any) -> Optional[int]:
    """Best-effort parse of a ``Retry-After`` header into whole seconds."""
    from agent.retry_utils import parse_retry_after_seconds

    seconds = parse_retry_after_seconds(headers)
    return None if seconds is None else int(seconds)


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


def _format_nous_entitlement_auth_error(error: AuthError) -> str:
    try:
        from hermes_cli.nous_account import (
            format_nous_portal_entitlement_message,
            get_nous_portal_account_info,
        )

        account_info = get_nous_portal_account_info(force_fresh=True)
        message = format_nous_portal_entitlement_message(
            account_info,
            capability="Nous model access",
        )
        if message:
            return message
    except Exception:
        pass
    return f"{error} Check credits or billing in Nous Portal, then retry."


def _nonempty_str(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _token_fingerprint(token: Any) -> Optional[str]:
    """Return a short hash fingerprint for telemetry without leaking token bytes."""
    if not isinstance(token, str):
        return None
    cleaned = token.strip()
    if not cleaned:
        return None
    return hashlib.sha256(cleaned.encode("utf-8")).hexdigest()[:12]


def _oauth_trace_enabled() -> bool:
    raw = os.getenv("HERMES_OAUTH_TRACE", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _oauth_trace(event: str, *, sequence_id: Optional[str] = None, **fields: Any) -> None:
    if not _oauth_trace_enabled():
        return
    payload: Dict[str, Any] = {"event": event}
    if sequence_id:
        payload["sequence_id"] = sequence_id
    payload.update(fields)
    logger.info("oauth_trace %s", json.dumps(payload, sort_keys=True, ensure_ascii=False))


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


# Pool providers whose OAuth refresh tokens are SINGLE-USE: redeeming the
# refresh token rotates the pair and revokes the old one. A grant forked into
# two auth.json files is therefore not two credentials but one credential with
# two owners — the first owner to refresh strands the other with
# ``invalid_grant`` / ``refresh_token_reused`` (#100339; same class as the
# ``providers.<id>`` write-through hazard in #48415 / #43589). Profiles must
# never receive a copy of these grants: ONE grant lives at the global root and
# named profiles read it through the ``read_credential_pool`` root fallback.
SINGLE_USE_REFRESH_POOL_PROVIDERS = frozenset({
    "anthropic",
    "openai-codex",
    "xai-oauth",
})

# Singleton credential files that hold the same single-use grants outside
# ``auth.json``. Copying one into a profile re-seeds a forked pool row on the
# profile's next ``load_pool()``.
SINGLE_USE_OAUTH_SINGLETON_FILES = (".anthropic_oauth.json",)


def _is_oauth_pool_payload(entry: Any) -> bool:
    if not isinstance(entry, dict):
        return False
    auth_type = str(entry.get("auth_type") or "").strip().lower()
    if auth_type == "oauth":
        return True
    # Legacy rows predating ``auth_type``: an Anthropic OAuth access token or
    # any row carrying a refresh token is an OAuth grant.
    if str(entry.get("refresh_token") or "").strip():
        return True
    return str(entry.get("access_token") or "").startswith("sk-ant-oat")


def strip_cloned_single_use_oauth_grants(profile_dir: Path) -> Dict[str, Any]:
    """Remove forked single-use OAuth grants from a freshly cloned profile.

    Called after any code path that copies credential files from one profile into another (``hermes
    profile create --clone-all``, the dashboard/TUI ``mirror_credentials`` flow). API-key pool rows
    are kept — a static key is safe to duplicate.

    Returns a summary ``{"pool": [...provider ids], "providers": [...], "files": [...]}`` of what
    was stripped (empty lists when nothing was). Never raises: a clone must not fail because
    credential hygiene could not run — the caller logs the summary.
    """
    stripped: Dict[str, Any] = {"pool": [], "providers": [], "files": []}
    profile_dir = Path(profile_dir)
    for name in SINGLE_USE_OAUTH_SINGLETON_FILES:
        try:
            target = profile_dir / name
            if target.is_file() or target.is_symlink():
                target.unlink()
                stripped["files"].append(name)
        except OSError:
            logger.debug("Could not remove cloned %s from %s", name, profile_dir, exc_info=True)

    auth_path = profile_dir / "auth.json"
    if not auth_path.is_file():
        return stripped
    try:
        store = json.loads(auth_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return stripped
    if not isinstance(store, dict):
        return stripped

    changed = False
    pool = store.get("credential_pool")
    if isinstance(pool, dict):
        for provider_id in list(pool):
            if provider_id not in SINGLE_USE_REFRESH_POOL_PROVIDERS:
                continue
            entries = pool.get(provider_id)
            if not isinstance(entries, list):
                continue
            kept = [e for e in entries if not _is_oauth_pool_payload(e)]
            if len(kept) != len(entries):
                changed = True
                stripped["pool"].append(provider_id)
                if kept:
                    pool[provider_id] = kept
                else:
                    # No local rows at all → read_credential_pool falls back
                    # to the root slice for this provider.
                    del pool[provider_id]
    providers = store.get("providers")
    if isinstance(providers, dict):
        # Device-code grants for these providers live under providers.<id>;
        # _load_provider_state has the same root fallback, so dropping the
        # copy keeps the profile working while removing the fork.
        for provider_id in ("openai-codex", "xai-oauth"):
            block = providers.get(provider_id)
            if isinstance(block, dict) and block:
                del providers[provider_id]
                stripped["providers"].append(provider_id)
                changed = True
    if not changed:
        return stripped
    try:
        _save_auth_store(store, target_path=auth_path)
    except Exception:
        logger.debug(
            "Failed to strip cloned single-use OAuth grants from %s",
            auth_path,
            exc_info=True,
        )
    return stripped


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

_OAUTH_TOKEN_FIELDS = (
    "access_token",
    "refresh_token",
    "expires_at",
    "expires_at_ms",
    "last_refresh",
)

_oauth_heal_notices: List[str] = []
# provider -> (profile auth.json path, auth.json mtime_ns, singleton mtime_ns)
# of the last store verified fork-free; lets load_pool() skip the locked scan.
_oauth_heal_clean_marks: Dict[str, Tuple[str, Optional[int], Optional[int]]] = {}


def consume_oauth_heal_notices() -> List[str]:
    """Return (and clear) human-readable notes about heals run in this process.

    ``hermes auth list`` / ``hermes auth status`` print them so the user sees that a forked grant
    was consolidated rather than only finding it in logs.
    """
    notes = list(_oauth_heal_notices)
    _oauth_heal_notices.clear()
    return notes


def _oauth_identity(entry: Dict[str, Any]) -> Optional[str]:
    """Stable account identity for an OAuth row when the token carries one.

    Codex / xAI access tokens are JWTs with ``sub`` / ``email`` / ``chatgpt_account_id`` claims;
    Anthropic ``sk-ant-oat`` tokens carry none (returns None, so lineage rests on id / token
    material).
    """
    if not isinstance(entry, dict):
        return None
    for token in (entry.get("access_token"), entry.get("id_token")):
        claims = _decode_jwt_claims(token)
        if not claims:
            continue
        nested = claims.get("https://api.openai.com/auth")
        account = nested.get("chatgpt_account_id") if isinstance(nested, dict) else None
        for value in (account, claims.get("sub"), claims.get("email")):
            if _nonempty_str(value):
                return value.strip()
    return None


def _oauth_freshness(entry: Dict[str, Any]) -> float:
    """Best-effort 'how recently was this pair issued' score (epoch seconds).

    A rotation always issues a later-expiring access token, so ``expires_at`` ordering identifies
    the live copy; ``last_refresh`` and the JWT ``exp`` claim are fallbacks for rows that do not
    persist expiry.
    """
    from agent.credential_pool import _parse_absolute_timestamp

    best = 0.0
    for key in ("expires_at_ms", "expires_at", "last_refresh"):
        ts = _parse_absolute_timestamp(entry.get(key))
        if ts and ts > best:
            best = ts
    if best == 0.0:
        exp = _decode_jwt_claims(entry.get("access_token")).get("exp")
        ts = _parse_absolute_timestamp(exp)
        if ts:
            best = ts
    return best


def _find_root_counterpart(
    profile_row: Dict[str, Any], root_rows: List[Dict[str, Any]]
) -> Optional[int]:
    """Index of the root OAuth row that shares a grant lineage with *profile_row*.

    Fallback per the one-grant-at-root rule: same provider + same OAuth client — every Anthropic
    ``hermes_pkce`` grant uses one client id and carries no claims, so two Anthropic OAuth rows with
    no contrary identity are one lineage.
    """
    candidates = [i for i, r in enumerate(root_rows) if _is_oauth_pool_payload(r)]
    if not candidates:
        return None
    pid = profile_row.get("id")
    for i in candidates:
        if pid and root_rows[i].get("id") == pid:
            return i
    p_ident = _oauth_identity(profile_row)
    for i in candidates:
        r_ident = _oauth_identity(root_rows[i])
        if p_ident and r_ident and p_ident == r_ident:
            return i
    for key in ("refresh_token", "access_token"):
        p_val = profile_row.get(key)
        if not _nonempty_str(p_val):
            continue
        for i in candidates:
            if root_rows[i].get(key) == p_val:
                return i
    # Fallback: same provider + same client. Only a contradicting identity
    # (both sides carry claims and they differ from every root row) blocks it.
    if p_ident:
        for i in candidates:
            if not _oauth_identity(root_rows[i]):
                return i
        return None
    return candidates[0]


def _adopt_oauth_material(target: Dict[str, Any], winner: Dict[str, Any]) -> Dict[str, Any]:
    """Return *target* carrying *winner*'s token pair, status markers cleared."""
    merged = dict(target)
    for key in _OAUTH_TOKEN_FIELDS:
        if winner.get(key) is not None:
            merged[key] = winner[key]
        else:
            merged.pop(key, None)
    for status_field in _POOL_STATUS_FIELDS:
        merged[status_field] = None
    return merged


def _singleton_as_row(path: Path) -> Optional[Dict[str, Any]]:
    """Read a ``.anthropic_oauth.json`` as a pool-row-shaped dict, or None."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or not str(data.get("accessToken") or "").strip():
        return None
    return {
        "access_token": data.get("accessToken"),
        "refresh_token": data.get("refreshToken"),
        "expires_at_ms": data.get("expiresAt"),
    }


def heal_forked_single_use_oauth_grants(provider_id: str) -> Optional[Dict[str, Any]]:
    """Consolidate a profile's forked copy of a single-use OAuth grant into root.

    Runs only in profile mode for ``SINGLE_USE_REFRESH_POOL_PROVIDERS``. Returns a summary
    ``{"adopted": bool, "stripped_ids": [...], "files": [...], "providers_block": bool}`` when
    something was healed, else ``None``. Never raises.
    """
    if provider_id not in SINGLE_USE_REFRESH_POOL_PROVIDERS:
        return None
    try:
        return _heal_forked_single_use_oauth_grants(provider_id)
    except Exception:
        logger.debug("%s: forked-OAuth heal skipped", provider_id, exc_info=True)
        return None


def _heal_forked_provider_block(
    profile_store: Dict[str, Any], root_store: Dict[str, Any], provider_id: str,
) -> Optional[bool]:
    """Consolidate a forked ``providers.<id>`` device-code block into root.

    Returns None when nothing matched, False when the profile copy was dropped (root already
    newest), True when the profile copy was fresher and was adopted into root.
    """
    p_providers = profile_store.get("providers")
    r_providers = root_store.get("providers")
    if not (isinstance(p_providers, dict) and isinstance(r_providers, dict)):
        return None
    p_block = p_providers.get(provider_id)
    r_block = r_providers.get(provider_id)
    if not (isinstance(p_block, dict) and p_block and isinstance(r_block, dict) and r_block):
        return None

    def _flat(block: Dict[str, Any]) -> Dict[str, Any]:
        tokens = block.get("tokens") if isinstance(block.get("tokens"), dict) else {}
        return {**tokens, "last_refresh": block.get("last_refresh")}

    p_flat, r_flat = _flat(p_block), _flat(r_block)
    p_ident, r_ident = _oauth_identity(p_flat), _oauth_identity(r_flat)
    if p_ident and r_ident and p_ident != r_ident:
        return None
    adopted = _oauth_freshness(p_flat) > _oauth_freshness(r_flat)
    if adopted:
        r_providers[provider_id] = dict(p_block)
    del p_providers[provider_id]
    return adopted


def _heal_forked_single_use_oauth_grants(provider_id: str) -> Optional[Dict[str, Any]]:
    root_path = _global_auth_file_path()
    if root_path is None:
        return None  # classic mode: nothing to consolidate into
    if os.environ.get("PYTEST_CURRENT_TEST"):
        # Same seat belt as the write-through paths: never touch the real
        # user's ~/.hermes/auth.json from a test that forgot to isolate HOME.
        real_home_env = os.environ.get("HOME", "")
        if real_home_env and _same_path(root_path, Path(real_home_env) / ".hermes" / "auth.json"):
            return None
    profile_path = _auth_file_path()
    profile_home = profile_path.parent
    root_home = root_path.parent
    profile_singleton = profile_home / ".anthropic_oauth.json" if provider_id == "anthropic" else None

    # Hot-path short-circuit: load_pool() runs per model call. Once this
    # profile's store was verified clean for *provider_id*, skip the locked
    # read-modify-write until the profile's own files change (mtime key).
    def _stamp(p: Optional[Path]) -> Optional[int]:
        try:
            return p.stat().st_mtime_ns if p is not None else None
        except OSError:
            return None

    fingerprint = (str(profile_path), _stamp(profile_path), _stamp(profile_singleton))
    if _oauth_heal_clean_marks.get(provider_id) == fingerprint:
        return None
    if fingerprint[1] is None and fingerprint[2] is None:
        _oauth_heal_clean_marks[provider_id] = fingerprint
        return None

    summary: Dict[str, Any] = {"adopted": False, "stripped_ids": [], "files": [], "providers_block": False}
    log_bits: List[str] = []

    # Lock order: active (profile) store first, then the root source store —
    # the same order ``_provider_state_transaction`` uses.
    with _auth_store_lock():
        profile_store = _load_auth_store(profile_path) if profile_path.exists() else {"providers": {}}
        with _auth_store_lock(target_path=root_path):
            root_store = _load_auth_store(root_path) if root_path.exists() else {"providers": {}}
            profile_changed = False
            root_changed = False

            p_pool = profile_store.get("credential_pool")
            p_rows = p_pool.get(provider_id) if isinstance(p_pool, dict) else None
            p_rows = p_rows if isinstance(p_rows, list) else []
            r_pool = root_store.get("credential_pool")
            r_rows = r_pool.get(provider_id) if isinstance(r_pool, dict) else None
            r_rows = r_rows if isinstance(r_rows, list) else []
            r_oauth = [r for r in r_rows if _is_oauth_pool_payload(r)]

            root_singleton = root_home / ".anthropic_oauth.json" if provider_id == "anthropic" else None
            root_singleton_row = (
                _singleton_as_row(root_singleton)
                if root_singleton is not None and root_singleton.exists() else None
            )

            # ── credential_pool rows ────────────────────────────────────
            kept_rows: List[Any] = []
            for row in p_rows:
                if not _is_oauth_pool_payload(row):
                    kept_rows.append(row)  # API keys are safe to duplicate
                    continue
                match_idx = _find_root_counterpart(row, r_rows)
                if match_idx is not None:
                    root_row = r_rows[match_idx]
                    if _oauth_freshness(row) > _oauth_freshness(root_row):
                        r_rows[match_idx] = _adopt_oauth_material(root_row, row)
                        root_changed = True
                        summary["adopted"] = True
                    summary["stripped_ids"].append(row.get("id"))
                    profile_changed = True
                    continue
                # No root pool counterpart. Root's grant may live only in its
                # .anthropic_oauth.json (the ``hermes auth`` PKCE shape); a
                # profile hermes_pkce-family row is that grant's copy.
                is_pkce = str(row.get("source") or "").endswith("hermes_pkce")
                if is_pkce and root_singleton_row is not None and not r_oauth:
                    if _oauth_freshness(row) > _oauth_freshness(root_singleton_row):
                        root_singleton_row = _adopt_oauth_material(root_singleton_row, row)
                        summary["adopted"] = True
                    summary["stripped_ids"].append(row.get("id"))
                    profile_changed = True
                    continue
                # Root holds no copy of this lineage (independent account, or
                # root never had the grant): the profile's row may be the
                # only surviving copy — leave it alone.
                kept_rows.append(row)
            if profile_changed and isinstance(p_pool, dict):
                if kept_rows:
                    p_pool[provider_id] = kept_rows
                else:
                    p_pool.pop(provider_id, None)

            # ── providers.<id> device-code blocks (Codex / xAI) ─────────
            if provider_id in ("openai-codex", "xai-oauth"):
                block_result = _heal_forked_provider_block(profile_store, root_store, provider_id)
                if block_result is not None:
                    profile_changed = True
                    summary["providers_block"] = True
                    if block_result:
                        root_changed = True
                        summary["adopted"] = True

            # ── profile-local .anthropic_oauth.json singleton ───────────
            if profile_singleton is not None and profile_singleton.exists():
                p_single = _singleton_as_row(profile_singleton)
                root_has_grant = bool(r_oauth) or root_singleton_row is not None
                if p_single is not None and root_has_grant:
                    if root_singleton_row is not None:
                        if _oauth_freshness(p_single) > _oauth_freshness(root_singleton_row):
                            root_singleton_row = _adopt_oauth_material(root_singleton_row, p_single)
                            summary["adopted"] = True
                    else:
                        # Root only has pool rows: fold the singleton's pair
                        # into the freshest-matching root pkce row, if any.
                        idx = next(
                            (i for i, r in enumerate(r_rows)
                             if _is_oauth_pool_payload(r)
                             and str(r.get("source") or "").endswith("hermes_pkce")),
                            None,
                        )
                        if idx is not None and _oauth_freshness(p_single) > _oauth_freshness(r_rows[idx]):
                            r_rows[idx] = _adopt_oauth_material(r_rows[idx], p_single)
                            root_changed = True
                            summary["adopted"] = True
                    try:
                        profile_singleton.unlink()
                        summary["files"].append(profile_singleton.name)
                    except OSError:
                        logger.debug("could not remove %s", profile_singleton, exc_info=True)
                # Otherwise root has NO grant for this provider (or the file
                # is not a grant): the profile's singleton may be the only
                # surviving copy — never delete it.

            if not (profile_changed or root_changed or summary["adopted"]):
                _oauth_heal_clean_marks[provider_id] = fingerprint
                return None

            if summary["adopted"] and root_singleton is not None and root_singleton_row is not None:
                # Keep root's singleton and its ``hermes_pkce``-seeded pool row
                # in step: root's next load_pool() re-seeds that row FROM the
                # singleton file, so a stale file would resurrect the spent
                # pair (and a stale row would be overwritten by a fresh file).
                pkce_idx = next(
                    (i for i, r in enumerate(r_rows)
                     if _is_oauth_pool_payload(r) and r.get("source") == "hermes_pkce"),
                    None,
                )
                if pkce_idx is not None:
                    pkce_row = r_rows[pkce_idx]
                    if _oauth_freshness(pkce_row) > _oauth_freshness(root_singleton_row):
                        root_singleton_row = _adopt_oauth_material(root_singleton_row, pkce_row)
                    elif _oauth_freshness(root_singleton_row) > _oauth_freshness(pkce_row):
                        r_rows[pkce_idx] = _adopt_oauth_material(pkce_row, root_singleton_row)
                        root_changed = True

            if root_changed:
                if isinstance(r_pool, dict):
                    r_pool[provider_id] = r_rows
                else:
                    root_store["credential_pool"] = {provider_id: r_rows}
                _save_auth_store(root_store, target_path=root_path)
            if summary["adopted"] and root_singleton is not None and root_singleton_row is not None:
                from agent.anthropic_credentials import _write_hermes_oauth_credentials
                _write_hermes_oauth_credentials(
                    root_singleton_row.get("access_token") or "",
                    root_singleton_row.get("refresh_token"),
                    root_singleton_row.get("expires_at_ms"),
                    target=root_singleton,
                )
            if profile_changed and profile_path.exists():
                _save_auth_store(profile_store, target_path=profile_path)

    if summary["stripped_ids"]:
        log_bits.append(f"pool rows {summary['stripped_ids']}")
    if summary["providers_block"]:
        log_bits.append(f"providers.{provider_id} block")
    if summary["files"]:
        log_bits.append(", ".join(summary["files"]))
    verdict = (
        "profile copy was the live pair; root updated"
        if summary["adopted"] else "root copy already newest; profile copy dropped"
    )
    message = (
        f"profile {profile_home.name}: consolidated forked {provider_id} OAuth grant "
        f"({'; '.join(log_bits) or 'no-op'}) into the root grant — {verdict}; "
        f"this profile now borrows the root grant (#100339)"
    )
    logger.info(message)
    _oauth_heal_notices.append(message)
    return summary


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


def _clear_pool_entry_status(entry: Dict[str, Any]) -> None:
    """Reset a pool entry's cooldown / last-error metadata to healthy."""
    for status_field in _POOL_STATUS_FIELDS:
        entry[status_field] = None


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


def _iso_after(now: datetime, ttl_seconds: int) -> str:
    """ISO timestamp *ttl_seconds* after *now* (UTC)."""
    return datetime.fromtimestamp(now.timestamp() + ttl_seconds, tz=timezone.utc).isoformat()


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
# Nous agent-key slots; a fresh login persists them as None, quarantine strips them.
_NOUS_EMPTY_AGENT_KEY_FIELDS: Dict[str, Any] = {
    "agent_key": None,
    "agent_key_id": None,
    "agent_key_expires_at": None,
    "agent_key_expires_in": None,
    "agent_key_reused": None,
    "agent_key_obtained_at": None,
}


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


_NOUS_STALE_PORTAL_HOSTS: FrozenSet[str] = frozenset({
    "api.nousresearch.com",
})

# Allowlist of valid Nous Portal hosts. A portal_base_url outside this
# set is treated as a misconfiguration and falls back to the default.
# "localhost" / "127.0.0.1" are valid for local development and testing.
_NOUS_PORTAL_ALLOWED_HOSTS: FrozenSet[str] = frozenset({
    "portal.nousresearch.com",
    "localhost",
    "127.0.0.1",
})


def _migrate_stale_nous_portal_url(providers: Dict[str, Any]) -> None:
    nous = providers.get("nous")
    if not isinstance(nous, dict):
        return
    stored = (nous.get("portal_base_url") or "").strip()
    if stored:
        parsed = urlparse(stored)
        if parsed.hostname in _NOUS_STALE_PORTAL_HOSTS:
            logger.warning(
                "auth: migrating stale nous portal_base_url %s -> %s",
                stored, DEFAULT_NOUS_PORTAL_URL,
            )
            nous["portal_base_url"] = DEFAULT_NOUS_PORTAL_URL


# Allowlist of hosts the Nous Portal proxy is willing to forward inference
# JWTs to. Sending a bearer anywhere else would leak it.
#
# This is consulted only for URLs coming from the NETWORK side (Portal
# refresh responses). User-controlled env-var overrides
# (NOUS_INFERENCE_BASE_URL) bypass validation — that's the documented
# dev/staging escape hatch and the env source is already trusted (the
# user set it themselves).
_ALLOWED_NOUS_INFERENCE_HOSTS: FrozenSet[str] = frozenset({
    "inference-api.nousresearch.com",
})


def _validate_nous_inference_url_from_network(url: Optional[str]) -> Optional[str]:
    """Validate a Portal-returned inference URL against the host allowlist.

    Defense-in-depth: a compromised refresh response from the Portal API (MITM, malicious response
    injection) could otherwise redirect every subsequent proxy request — bearing the user's
    inference JWT — to an attacker-controlled endpoint.
    """
    if not isinstance(url, str):
        return None
    cleaned = url.strip()
    if not cleaned:
        return None
    try:
        parsed = urlparse(cleaned)
    except Exception:
        return None
    if parsed.scheme != "https":
        logger.warning(
            "nous: refusing non-https inference URL scheme %r from Portal response",
            parsed.scheme,
        )
        return None
    if parsed.hostname not in _ALLOWED_NOUS_INFERENCE_HOSTS:
        logger.warning(
            "nous: refusing inference URL host %r from Portal response "
            "(not in allowlist); falling back to default",
            parsed.hostname,
        )
        return None
    return cleaned.rstrip("/")


def _nous_inference_env_override() -> Optional[str]:
    """Return the user-set ``NOUS_INFERENCE_BASE_URL`` override, if any.

    Documented dev/staging escape hatch. The env source is trusted (the OS user set it), so unlike
    Portal-returned URLs it is intentionally NOT gated by the network host allowlist.
    Returns a trailing-slash-stripped string, or ``None`` when unset/blank.
    """
    return _optional_base_url(os.getenv("NOUS_INFERENCE_BASE_URL"))


def _nous_portal_env_override() -> Optional[str]:
    """Return the user/deployment-set Portal base URL override, if any.

    ``HERMES_PORTAL_BASE_URL`` / ``NOUS_PORTAL_BASE_URL`` are the documented dev/staging escape
    hatch (e.g. hosted agents on the staging Portal). Like the inference override, the env source
    is trusted and must NOT be gated by ``_NOUS_PORTAL_ALLOWED_HOSTS``: that allowlist rejects an
    untrusted NETWORK-provided value persisted to auth.json, not one the operator configured.
    """
    return _optional_base_url(
        os.getenv("HERMES_PORTAL_BASE_URL") or os.getenv("NOUS_PORTAL_BASE_URL")
    )


def _decode_jwt_claims(token: Any) -> Dict[str, Any]:
    if not isinstance(token, str) or token.count(".") != 2:
        return {}
    payload = token.split(".")[1]
    payload += "=" * ((4 - len(payload) % 4) % 4)
    try:
        raw = base64.urlsafe_b64decode(payload.encode("utf-8"))
        claims = json.loads(raw.decode("utf-8"))
    except Exception:
        return {}
    return claims if isinstance(claims, dict) else {}


def _scope_values(raw_scope: Any) -> set[str]:
    # OAuth token responses normally return a space-separated string. Keep
    # collection support for JWT ``scp`` claims and older stored test fixtures.
    scopes: set[str] = set()
    if isinstance(raw_scope, str):
        for part in raw_scope.replace(",", " ").split():
            cleaned = part.strip()
            if cleaned:
                scopes.add(cleaned)
    elif isinstance(raw_scope, (list, tuple, set, frozenset)):
        for item in raw_scope:
            if isinstance(item, str):
                scopes.update(_scope_values(item))
    return scopes


def _nous_invoke_jwt_status(
    token: Any,
    *,
    scope: Any = None,
    expires_at: Any = None,
    min_ttl_seconds: int = NOUS_INVOKE_JWT_MIN_TTL_SECONDS,
) -> Optional[str]:
    """Return None when the token can be used for inference, else a reason."""
    claims = _decode_jwt_claims(token)
    if not claims:
        return "access_token_not_jwt"
    scopes = (
        _scope_values(scope)
        | _scope_values(claims.get("scope"))
        | _scope_values(claims.get("scp"))
    )
    if NOUS_INFERENCE_INVOKE_SCOPE not in scopes:
        return "missing_inference_invoke_scope"
    exp = claims.get("exp")
    skew = max(0, int(min_ttl_seconds))
    if isinstance(exp, (int, float)):
        if float(exp) <= (time.time() + skew):
            return "invoke_jwt_expiring"
        return None
    if _is_expiring(expires_at, skew):
        return "invoke_jwt_expiry_unknown_or_expiring"
    return None


def _nous_invoke_jwt_is_usable(
    token: Any,
    *,
    scope: Any = None,
    expires_at: Any = None,
    min_ttl_seconds: int = NOUS_INVOKE_JWT_MIN_TTL_SECONDS,
) -> bool:
    return (
        _nous_invoke_jwt_status(
            token,
            scope=scope,
            expires_at=expires_at,
            min_ttl_seconds=min_ttl_seconds,
        )
        is None
    )


def _assert_nous_inference_jwt_usable(
    state: Dict[str, Any],
    *,
    access_token: Any = None,
) -> None:
    token = state.get("access_token") if access_token is None else access_token
    reason = _nous_invoke_jwt_status(
        token,
        scope=state.get("scope"),
        expires_at=state.get("expires_at"),
    )
    if reason is None:
        return
    raise _nous_err(
        "Nous Portal access token is not a usable inference JWT "
        f"({reason}). Re-authenticate with: hermes auth add nous",
        reason, relogin=True,
    )


def _log_nous_invoke_jwt_selected(
    *,
    access_token: Any,
    sequence_id: Optional[str] = None,
) -> None:
    logger.debug("Nous inference auth: using NAS invoke JWT")
    _oauth_trace(
        "nous_invoke_jwt_selected",
        sequence_id=sequence_id,
        access_token_fp=_token_fingerprint(access_token),
    )


def _nous_jwt_expires_at(token: Any, fallback_expires_at: Any = None) -> Optional[str]:
    claims = _decode_jwt_claims(token)
    exp = claims.get("exp")
    if isinstance(exp, (int, float)):
        try:
            return datetime.fromtimestamp(float(exp), tz=timezone.utc).isoformat()
        except Exception:
            pass
    return fallback_expires_at if isinstance(fallback_expires_at, str) else None


def _set_nous_agent_key_from_invoke_jwt(
    state: Dict[str, Any],
    *,
    obtained_at: Optional[str] = None,
) -> None:
    access_token = state.get("access_token")
    if not _nonempty_str(access_token):
        return
    now = datetime.now(timezone.utc)
    existing_obtained_at = state.get("agent_key_obtained_at")
    if obtained_at:
        effective_obtained_at = obtained_at
    elif (
        state.get("agent_key") == access_token
        and isinstance(existing_obtained_at, str)
        and existing_obtained_at.strip()
    ):
        effective_obtained_at = existing_obtained_at
    else:
        effective_obtained_at = now.isoformat()
    expires_at = _nous_jwt_expires_at(access_token, state.get("expires_at"))
    expires_epoch = _parse_iso_timestamp(expires_at)
    expires_in = (
        max(0, int(expires_epoch - time.time()))
        if expires_epoch is not None
        else _coerce_ttl_seconds(state.get("expires_in"))
    )
    if expires_at:
        state["expires_at"] = expires_at
        state["expires_in"] = expires_in
    state["agent_key"] = access_token
    state["agent_key_id"] = None
    state["agent_key_expires_at"] = expires_at
    state["agent_key_expires_in"] = expires_in
    state["agent_key_reused"] = False
    state["agent_key_obtained_at"] = effective_obtained_at


def _select_nous_invoke_jwt(
    state: Dict[str, Any],
    *,
    access_token: Any = None,
    sequence_id: Optional[str] = None,
) -> None:
    if _nonempty_str(access_token):
        state["access_token"] = access_token
    _set_nous_agent_key_from_invoke_jwt(state)
    _log_nous_invoke_jwt_selected(
        access_token=state.get("access_token"),
        sequence_id=sequence_id,
    )


_NOUS_EFFECTIVE_STATE_IGNORED_KEYS = frozenset({
    # These are derived from expires_at/JWT exp and naturally tick down between
    # reads. Persisting only these changes makes auth.json noisy and defeats
    # the mtime-keyed auth-status cache.
    "expires_in",
    "agent_key_expires_in",
})


def _nous_effective_provider_state(state: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: value
        for key, value in state.items()
        if key not in _NOUS_EFFECTIVE_STATE_IGNORED_KEYS
    }


def _codex_access_token_is_expiring(access_token: Any, skew_seconds: int) -> bool:
    claims = _decode_jwt_claims(access_token)
    exp = claims.get("exp")
    if not isinstance(exp, (int, float)):
        return False
    return float(exp) <= (time.time() + max(0, int(skew_seconds)))


# =============================================================================
# Spotify auth — PKCE tokens stored in ~/.hermes/auth.json
# =============================================================================


def _spotify_scope_list(raw_scope: Optional[str] = None) -> List[str]:
    scope_text = (raw_scope or DEFAULT_SPOTIFY_SCOPE).strip()
    scopes = [part for part in scope_text.split() if part]
    seen: set[str] = set()
    ordered: List[str] = []
    for scope in scopes:
        if scope not in seen:
            seen.add(scope)
            ordered.append(scope)
    return ordered


def _spotify_scope_string(raw_scope: Optional[str] = None) -> str:
    return " ".join(_spotify_scope_list(raw_scope))


def _spotify_setting(
    state: Optional[Dict[str, Any]],
    state_key: str,
    env_vars: Tuple[str, ...],
    default: str,
    *,
    explicit: Optional[str] = None,
    strip_slash: bool = False,
) -> str:
    """First non-empty of explicit arg, env vars (``.env`` aware), stored state, then *default*."""
    from hermes_cli.config import get_env_value

    candidates = (
        explicit,
        *(get_env_value(var) for var in env_vars),
        state.get(state_key) if isinstance(state, dict) else None,
        default,
    )
    for candidate in candidates:
        cleaned = str(candidate or "").strip()
        if strip_slash:
            cleaned = cleaned.rstrip("/")
        if cleaned:
            return cleaned
    return default


def _spotify_client_id(
    explicit: Optional[str] = None,
    state: Optional[Dict[str, Any]] = None,
) -> str:
    client_id = _spotify_setting(
        state, "client_id", ("HERMES_SPOTIFY_CLIENT_ID", "SPOTIFY_CLIENT_ID"), "", explicit=explicit,
    )
    if client_id:
        return client_id
    raise _spotify_err(
        "Spotify client_id is required. Set HERMES_SPOTIFY_CLIENT_ID or pass --client-id.",
        "spotify_client_id_missing",
    )


def _spotify_redirect_uri(
    explicit: Optional[str] = None,
    state: Optional[Dict[str, Any]] = None,
) -> str:
    return _spotify_setting(
        state, "redirect_uri", ("HERMES_SPOTIFY_REDIRECT_URI", "SPOTIFY_REDIRECT_URI"),
        DEFAULT_SPOTIFY_REDIRECT_URI, explicit=explicit,
    )


def _spotify_api_base_url(state: Optional[Dict[str, Any]] = None) -> str:
    return _spotify_setting(
        state, "api_base_url", ("HERMES_SPOTIFY_API_BASE_URL",),
        DEFAULT_SPOTIFY_API_BASE_URL, strip_slash=True,
    )


def _spotify_accounts_base_url(state: Optional[Dict[str, Any]] = None) -> str:
    return _spotify_setting(
        state, "accounts_base_url", ("HERMES_SPOTIFY_ACCOUNTS_BASE_URL",),
        DEFAULT_SPOTIFY_ACCOUNTS_BASE_URL, strip_slash=True,
    )


def _spotify_code_verifier(length: int = 64) -> str:
    raw = base64.urlsafe_b64encode(os.urandom(length)).decode("ascii")
    return raw.rstrip("=")[:128]


def _spotify_code_challenge(code_verifier: str) -> str:
    digest = hashlib.sha256(code_verifier.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


def _spotify_build_authorize_url(
    *,
    client_id: str,
    redirect_uri: str,
    scope: str,
    state: str,
    code_challenge: str,
    accounts_base_url: str,
) -> str:
    query = urlencode({
        "client_id": client_id,
        "response_type": "code",
        "redirect_uri": redirect_uri,
        "scope": scope,
        "state": state,
        "code_challenge_method": "S256",
        "code_challenge": code_challenge,
    })
    return f"{accounts_base_url}/authorize?{query}"


def _spotify_validate_redirect_uri(redirect_uri: str) -> tuple[str, int, str]:
    parsed = urlparse(redirect_uri)
    if parsed.scheme != "http":
        raise _spotify_err(
            "Spotify PKCE redirect_uri must use http://localhost or http://127.0.0.1.",
            "spotify_redirect_invalid",
        )
    host = parsed.hostname or ""
    if host not in {"127.0.0.1", "localhost"}:
        raise _spotify_err(
            "Spotify PKCE redirect_uri must point to localhost or 127.0.0.1.",
            "spotify_redirect_invalid",
        )
    if not parsed.port:
        raise _spotify_err(
            "Spotify PKCE redirect_uri must include an explicit localhost port.",
            "spotify_redirect_invalid",
        )
    return host, parsed.port, parsed.path or "/"


def _make_spotify_callback_handler(expected_path: str) -> tuple[type[BaseHTTPRequestHandler], dict[str, Any]]:
    result: dict[str, Any] = {
        "code": None,
        "state": None,
        "error": None,
        "error_description": None,
    }

    class _SpotifyCallbackHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path != expected_path:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b"Not found.")
                return

            params = parse_qs(parsed.query)
            result["code"] = params.get("code", [None])[0]
            result["state"] = params.get("state", [None])[0]
            result["error"] = params.get("error", [None])[0]
            result["error_description"] = params.get("error_description", [None])[0]

            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            if result["error"]:
                body = "<html><body><h1>Spotify authorization failed.</h1>You can close this tab.</body></html>"
            else:
                body = "<html><body><h1>Spotify authorization received.</h1>You can close this tab.</body></html>"
            self.wfile.write(body.encode("utf-8"))

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
            return

    return _SpotifyCallbackHandler, result


def _spotify_wait_for_callback(
    redirect_uri: str,
    *,
    timeout_seconds: float = 180.0,
) -> dict[str, Any]:
    host, port, path = _spotify_validate_redirect_uri(redirect_uri)
    handler_cls, result = _make_spotify_callback_handler(path)

    class _ReuseHTTPServer(HTTPServer):
        allow_reuse_address = True

    try:
        server = _ReuseHTTPServer((host, port), handler_cls)
    except OSError as exc:
        raise _spotify_err(
            f"Could not bind Spotify callback server on {host}:{port}: {exc}",
            "spotify_callback_bind_failed",
        ) from exc

    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True)
    thread.start()
    deadline = time.monotonic() + max(5.0, timeout_seconds)
    try:
        while time.monotonic() < deadline:
            if result["code"] or result["error"]:
                return result
            time.sleep(0.1)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1.0)
    raise _spotify_err(
        "Spotify authorization timed out waiting for the local callback.",
        "spotify_callback_timeout",
    )


def _spotify_token_payload_to_state(
    token_payload: Dict[str, Any],
    *,
    client_id: str,
    redirect_uri: str,
    requested_scope: str,
    accounts_base_url: str,
    api_base_url: str,
    previous_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    now = datetime.now(timezone.utc)
    expires_in = _coerce_ttl_seconds(token_payload.get("expires_in", 0))
    expires_at = datetime.fromtimestamp(now.timestamp() + expires_in, tz=timezone.utc)
    state = dict(previous_state or {})
    state.update({
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "accounts_base_url": accounts_base_url,
        "api_base_url": api_base_url,
        "scope": requested_scope,
        "granted_scope": str(token_payload.get("scope") or requested_scope).strip(),
        "token_type": str(token_payload.get("token_type", "Bearer") or "Bearer").strip() or "Bearer",
        "access_token": str(token_payload.get("access_token", "") or "").strip(),
        "refresh_token": str(
            token_payload.get("refresh_token")
            or state.get("refresh_token")
            or ""
        ).strip(),
        "obtained_at": now.isoformat(),
        "expires_at": expires_at.isoformat(),
        "expires_in": expires_in,
        "auth_type": "oauth_pkce",
    })
    return state


def _spotify_token_post(
    accounts_base_url: str,
    data: Dict[str, str],
    *,
    timeout_seconds: float,
    what: str,
    failed_code: str,
    invalid_code: str,
    invalid_message: str,
    failed_suffix: str = "",
    relogin_required: bool = False,
) -> Dict[str, Any]:
    """POST to Spotify's ``/api/token`` and return the JSON payload, or raise a shaped AuthError."""
    try:
        response = httpx.post(
            f"{accounts_base_url}/api/token",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data=data,
            timeout=timeout_seconds,
        )
    except Exception as exc:
        raise _spotify_err(f"Spotify {what} failed: {exc}", failed_code) from exc

    if response.status_code >= 400:
        detail = response.text.strip()
        raise _spotify_err(
            f"Spotify {what} failed.{failed_suffix}"
            + (f" Response: {detail}" if detail else ""),
            failed_code, relogin=relogin_required,
        )
    payload = response.json()
    if not isinstance(payload, dict) or not str(payload.get("access_token", "") or "").strip():
        raise _spotify_err(invalid_message, invalid_code, relogin=relogin_required)
    return payload


def _spotify_exchange_code_for_tokens(
    *,
    client_id: str,
    code: str,
    redirect_uri: str,
    code_verifier: str,
    accounts_base_url: str,
    timeout_seconds: float = 20.0,
) -> Dict[str, Any]:
    return _spotify_token_post(
        accounts_base_url,
        {
            "client_id": client_id,
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
            "code_verifier": code_verifier,
        },
        timeout_seconds=timeout_seconds,
        what="token exchange",
        failed_code="spotify_token_exchange_failed",
        invalid_code="spotify_token_exchange_invalid",
        invalid_message="Spotify token response did not include an access_token.",
    )


def _refresh_spotify_oauth_state(
    state: Dict[str, Any],
    *,
    timeout_seconds: float = 20.0,
) -> Dict[str, Any]:
    refresh_token = str(state.get("refresh_token", "") or "").strip()
    if not refresh_token:
        raise _spotify_err(
            "Spotify refresh token missing. Run `hermes auth spotify` again.",
            "spotify_refresh_token_missing", relogin=True,
        )

    client_id = _spotify_client_id(state=state)
    accounts_base_url = _spotify_accounts_base_url(state)
    payload = _spotify_token_post(
        accounts_base_url,
        {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": client_id,
        },
        timeout_seconds=timeout_seconds,
        what="token refresh",
        failed_code="spotify_refresh_failed",
        invalid_code="spotify_refresh_invalid",
        invalid_message="Spotify refresh response did not include an access_token.",
        failed_suffix=" Run `hermes auth spotify` again.",
        relogin_required=True,
    )

    return _spotify_token_payload_to_state(
        payload,
        client_id=client_id,
        redirect_uri=_spotify_redirect_uri(state=state),
        requested_scope=str(state.get("scope") or DEFAULT_SPOTIFY_SCOPE),
        accounts_base_url=accounts_base_url,
        api_base_url=_spotify_api_base_url(state),
        previous_state=state,
    )


def resolve_spotify_runtime_credentials(
    *,
    force_refresh: bool = False,
    refresh_if_expiring: bool = True,
    refresh_skew_seconds: int = SPOTIFY_ACCESS_TOKEN_REFRESH_SKEW_SECONDS,
) -> Dict[str, Any]:
    with _auth_store_lock():
        auth_store = _load_auth_store()
        state = _load_provider_state(auth_store, "spotify")
        if not state:
            raise _spotify_err(
                "Spotify is not authenticated. Run `hermes auth spotify` first.",
                "spotify_auth_missing", relogin=True,
            )

        should_refresh = bool(force_refresh)
        if not should_refresh and refresh_if_expiring:
            should_refresh = _is_expiring(state.get("expires_at"), refresh_skew_seconds)
        if should_refresh:
            try:
                state = _refresh_spotify_oauth_state(state)
                _store_provider_state(auth_store, "spotify", state, set_active=False)
                _save_auth_store(auth_store)
            except AuthError as exc:
                if exc.relogin_required and state.get("refresh_token"):
                    _quarantine_flat_oauth_state(state, "spotify", exc)
                    try:
                        _store_provider_state(auth_store, "spotify", state, set_active=False)
                        _save_auth_store(auth_store)
                    except Exception as _save_exc:
                        logger.debug("Spotify OAuth: failed to persist quarantined state: %s", _save_exc)
                raise

    access_token = str(state.get("access_token", "") or "").strip()
    if not access_token:
        raise _spotify_err(
            "Spotify access token missing. Run `hermes auth spotify` again.",
            "spotify_access_token_missing", relogin=True,
        )

    return {
        "provider": "spotify",
        "access_token": access_token,
        "api_key": access_token,
        "token_type": str(state.get("token_type", "Bearer") or "Bearer"),
        "base_url": _spotify_api_base_url(state),
        "scope": str(state.get("granted_scope") or state.get("scope") or "").strip(),
        "client_id": _spotify_client_id(state=state),
        "redirect_uri": _spotify_redirect_uri(state=state),
        "expires_at": state.get("expires_at"),
        "refresh_token": str(state.get("refresh_token", "") or "").strip(),
    }


def get_spotify_auth_status() -> Dict[str, Any]:
    state = get_provider_auth_state("spotify")
    if not state:
        return {"logged_in": False}

    expires_at = state.get("expires_at")
    refresh_token = str(state.get("refresh_token", "") or "").strip()
    return {
        "logged_in": bool(refresh_token or not _is_expiring(expires_at, 0)),
        "auth_type": state.get("auth_type", "oauth_pkce"),
        "client_id": state.get("client_id"),
        "redirect_uri": state.get("redirect_uri"),
        "scope": state.get("granted_scope") or state.get("scope"),
        "expires_at": expires_at,
        "api_base_url": state.get("api_base_url"),
        "has_refresh_token": bool(refresh_token),
    }


def _spotify_interactive_setup(redirect_uri_hint: str) -> str:
    """Walk the user through creating a Spotify developer app, persist the resulting client_id to
    ~/.hermes/.env, and return it.
    """
    from hermes_cli.config import save_env_value

    print()
    print("=" * 70)
    print("Spotify first-time setup")
    print("=" * 70)
    print()
    print("Spotify requires every user to register their own lightweight")
    print("developer app. This takes about two minutes and only has to be")
    print("done once per machine.")
    print()
    print(f"Full guide: {SPOTIFY_DOCS_URL}")
    print()
    print("Steps:")
    print(f"  1. Opening {SPOTIFY_DASHBOARD_URL} in your browser...")
    print("  2. Click 'Create app' and fill in:")
    print("       App name:     anything (e.g. hermes-agent)")
    print("       Description:  anything")
    print(f"       Redirect URI: {redirect_uri_hint}")
    print("       API/SDK:      Web API")
    print("  3. Agree to the terms, click Save.")
    print("  4. Open the app's Settings page and copy the Client ID.")
    print("  5. Paste it below.")
    print()

    if not _is_remote_session():
        try:
            webbrowser.open(SPOTIFY_DASHBOARD_URL)
        except Exception:
            pass

    from hermes_cli.cli_output import line_input

    try:
        raw = line_input("Spotify Client ID: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        raise SystemExit("Spotify setup cancelled.")

    if not raw:
        print()
        print(f"No Client ID entered. See {SPOTIFY_DOCS_URL} for the full guide.")
        raise SystemExit("Spotify setup cancelled: empty Client ID.")

    # Persist so subsequent `hermes auth spotify` runs skip the wizard.
    save_env_value("HERMES_SPOTIFY_CLIENT_ID", raw)
    # Only persist the redirect URI if it's non-default, to avoid pinning
    # users to a value the default might later change to.
    if redirect_uri_hint and redirect_uri_hint != DEFAULT_SPOTIFY_REDIRECT_URI:
        save_env_value("HERMES_SPOTIFY_REDIRECT_URI", redirect_uri_hint)

    print()
    print("Saved HERMES_SPOTIFY_CLIENT_ID to ~/.hermes/.env")
    print()
    return raw


def login_spotify_command(args) -> None:
    existing_state = get_provider_auth_state("spotify") or {}

    # Interactive wizard: if no client_id is configured anywhere, walk the
    # user through creating the Spotify developer app instead of crashing
    # with "HERMES_SPOTIFY_CLIENT_ID is required".
    explicit_client_id = getattr(args, "client_id", None)
    try:
        client_id = _spotify_client_id(explicit_client_id, existing_state)
    except AuthError as exc:
        if getattr(exc, "code", "") != "spotify_client_id_missing":
            raise
        client_id = _spotify_interactive_setup(
            redirect_uri_hint=getattr(args, "redirect_uri", None) or DEFAULT_SPOTIFY_REDIRECT_URI,
        )

    redirect_uri = _spotify_redirect_uri(getattr(args, "redirect_uri", None), existing_state)
    scope = _spotify_scope_string(getattr(args, "scope", None) or existing_state.get("scope"))
    accounts_base_url = _spotify_accounts_base_url(existing_state)
    api_base_url = _spotify_api_base_url(existing_state)
    open_browser = not getattr(args, "no_browser", False)

    code_verifier = _spotify_code_verifier()
    code_challenge = _spotify_code_challenge(code_verifier)
    state_nonce = uuid.uuid4().hex
    authorize_url = _spotify_build_authorize_url(
        client_id=client_id,
        redirect_uri=redirect_uri,
        scope=scope,
        state=state_nonce,
        code_challenge=code_challenge,
        accounts_base_url=accounts_base_url,
    )

    print("Starting Spotify PKCE login...")
    print(f"Client ID: {client_id}")
    print(f"Redirect URI: {redirect_uri}")
    print("Make sure this redirect URI is allow-listed in your Spotify app settings.")
    print()
    print("Open this URL to authorize Hermes:")
    print(authorize_url)
    print()
    print(f"Full setup guide: {SPOTIFY_DOCS_URL}")
    print()

    _print_loopback_ssh_hint(redirect_uri, docs_url=SPOTIFY_DOCS_URL)

    if open_browser and not _is_remote_session() and _can_open_graphical_browser():
        try:
            opened = webbrowser.open(authorize_url)
        except Exception:
            opened = False
        if opened:
            print("Browser opened for Spotify authorization.")
        else:
            print("Could not open the browser automatically; use the URL above.")

    callback = _spotify_wait_for_callback(
        redirect_uri,
        timeout_seconds=float(getattr(args, "timeout", None) or 180.0),
    )
    if callback.get("error"):
        detail = callback.get("error_description") or callback["error"]
        raise SystemExit(f"Spotify authorization failed: {detail}")
    if callback.get("state") != state_nonce:
        raise SystemExit("Spotify authorization failed: state mismatch.")

    token_payload = _spotify_exchange_code_for_tokens(
        client_id=client_id,
        code=str(callback.get("code") or ""),
        redirect_uri=redirect_uri,
        code_verifier=code_verifier,
        accounts_base_url=accounts_base_url,
        timeout_seconds=float(getattr(args, "timeout", None) or 20.0),
    )
    spotify_state = _spotify_token_payload_to_state(
        token_payload,
        client_id=client_id,
        redirect_uri=redirect_uri,
        requested_scope=scope,
        accounts_base_url=accounts_base_url,
        api_base_url=api_base_url,
    )

    with _auth_store_lock():
        auth_store = _load_auth_store()
        _store_provider_state(auth_store, "spotify", spotify_state, set_active=False)
        saved_to = _save_auth_store(auth_store)

    print("Spotify login successful!")
    print(f"  Auth state: {saved_to}")
    print("  Provider state saved under providers.spotify")
    print(f"  Docs: {SPOTIFY_DOCS_URL}")

# =============================================================================
# SSH / remote session detection
# =============================================================================

def _is_remote_session() -> bool:
    """Detect environments where loopback OAuth can't reach the local browser.

    These environments typically don't set ``SSH_CLIENT`` / ``SSH_TTY``, so the SSH-only check left
    them with no guidance and no fallback.
    """
    if os.getenv("SSH_CLIENT") or os.getenv("SSH_TTY"):
        return True
    # Browser-only remote IDEs / cloud shells.  Keep this list narrow
    # (well-known, documented env vars set by the host platform) so
    # we don't falsely trip on a developer's local shell.
    for var in (
        "CLOUD_SHELL",         # GCP Cloud Shell
        "CODESPACES",          # GitHub Codespaces
        "CODESPACE_NAME",      # GitHub Codespaces (alt)
        "GITPOD_WORKSPACE_ID", # Gitpod
        "REPL_ID",             # Replit
        "STACKBLITZ",          # StackBlitz
    ):
        if os.getenv(var):
            return True
    return False


# Console/text-mode browsers that ``webbrowser`` will happily launch INSIDE
# the terminal.  Opening one of these is worse than not opening anything —
# it hijacks the user's TTY with an unusable text browser (the xAI OAuth
# "Account Management" page rendered in w3m, reported May 2026) instead of
# letting them copy the URL to a real browser.  When the resolved browser is
# one of these we refuse to auto-open and fall back to the print-the-URL
# path, same as a remote session.
_CONSOLE_BROWSER_NAMES: FrozenSet[str] = frozenset(
    {
        "w3m",
        "lynx",
        "links",
        "links2",
        "elinks",
        "www-browser",
        "browsh",  # TUI browser — still hijacks the terminal
    }
)


def _can_open_graphical_browser() -> bool:
    """Return True only when a *graphical* browser is likely to open.

    ``webbrowser.open()`` resolves to whatever the platform offers, and on a headless / CLI-only
    Linux box with no GUI browser installed that is often a text-mode browser (w3m/lynx/links) which
    launches inside the terminal and takes over the user's session.

    Heuristics: * Respect ``$BROWSER`` — if it names a known console browser, refuse. * On Linux,
    require a display server (``$DISPLAY`` / ``$WAYLAND_DISPLAY``) unless ``$BROWSER`` points at
    something graphical; no display server almost always means no GUI browser.
    """
    import webbrowser as _webbrowser

    def _names_console_browser(value: str) -> bool:
        token = value.strip().split()[0] if value.strip() else ""
        base = os.path.basename(token).lower()
        return base in _CONSOLE_BROWSER_NAMES

    browser_env = os.environ.get("BROWSER", "")
    if browser_env and _names_console_browser(browser_env):
        return False

    if sys.platform.startswith("linux"):
        has_display = bool(
            os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
        )
        # An explicit graphical $BROWSER can work without $DISPLAY in odd
        # setups, but a console $BROWSER already returned False above, so the
        # only way to reach here with a $BROWSER set is a graphical one.
        if not has_display and not browser_env:
            return False

    try:
        controller = _webbrowser.get()
    except Exception:
        # No browser resolvable at all → definitely don't auto-open.
        return False

    candidate = (
        getattr(controller, "name", "")
        or getattr(controller, "basename", "")
        or ""
    )
    return not (candidate and _names_console_browser(candidate))


def _ssh_user_at_host() -> str:
    """Return best-effort 'user@hostname' for the SSH tunnel hint command.

    Falls back to placeholder tokens when the values cannot be determined so the hint is always
    syntactically valid even if not copy-pasteable.
    """
    try:
        import socket as _socket
        hostname = _socket.gethostname() or "<this-host>"
    except OSError:
        hostname = "<this-host>"
    user = os.getenv("USER") or os.getenv("LOGNAME") or "<user>"
    return f"{user}@{hostname}"


def _print_loopback_ssh_hint(redirect_uri: str, *, docs_url: str | None = None) -> None:
    """Print an SSH tunnel hint when running a loopback-redirect OAuth flow on a remote host. The auth
    server (Spotify, MCP servers, ...) will redirect the user's browser to
    ``127.0.0.1:<port>/callback``. If the browser is on a different machine than the loopback
    listener (the usual SSH case), the redirect can't reach the listener without a local port
    forward.
    """
    if not _is_remote_session():
        return
    try:
        parsed = urlparse(redirect_uri)
    except Exception:
        return
    host = parsed.hostname or ""
    port = parsed.port
    if host not in {"127.0.0.1", "::1", "localhost"} or not port:
        return
    divider = "-" * 60
    print()
    print(divider)
    print("Remote session detected — SSH tunnel required")
    print(divider)
    print(f"Hermes is waiting for the OAuth callback on {redirect_uri}")
    print("but your browser is on a different machine. Run this command")
    print("in a NEW terminal on your local machine BEFORE opening the URL:")
    print()
    print(f"  ssh -N -L {port}:127.0.0.1:{port} {_ssh_user_at_host()}")
    print()
    print("Then open the authorize URL above in your local browser.")
    if docs_url:
        print(f"Provider docs:      {docs_url}")
    print(f"SSH/jump-box guide: {OAUTH_OVER_SSH_DOCS_URL}")
    print(divider)
    print()


# =============================================================================
# OpenAI Codex auth — tokens stored in ~/.hermes/auth.json (not ~/.codex/)
#
# Hermes maintains its own Codex OAuth session separate from the Codex CLI
# and VS Code extension. This prevents refresh token rotation conflicts
# where one app's refresh invalidates the other's session.
# =============================================================================

def _codex_base_url() -> str:
    return os.getenv("HERMES_CODEX_BASE_URL", "").strip().rstrip("/") or DEFAULT_CODEX_BASE_URL


def _codex_runtime_result(api_key: str, *, source: str, last_refresh: Optional[str]) -> Dict[str, Any]:
    return {
        "provider": "openai-codex",
        "base_url": _codex_base_url(),
        "api_key": api_key,
        "source": source,
        "last_refresh": last_refresh,
        "auth_mode": "chatgpt",
    }


def _load_auth_store_maybe_locked(lock: bool) -> Dict[str, Any]:
    """Load the auth store, taking the cross-process lock unless the caller already holds it."""
    if lock:
        with _auth_store_lock():
            return _load_auth_store()
    return _load_auth_store()


def _read_codex_tokens(*, _lock: bool = True) -> Dict[str, Any]:
    """Read Codex OAuth tokens from Hermes auth store (~/.hermes/auth.json)."""
    auth_store = _load_auth_store_maybe_locked(_lock)
    state = _load_provider_state(auth_store, "openai-codex")
    if not state:
        raise _codex_err(
            "No Codex credentials stored. Run `hermes auth` to authenticate.",
            "codex_auth_missing", relogin=True,
        )
    tokens = state.get("tokens")
    if not isinstance(tokens, dict):
        raise _codex_err(
            "Codex auth state is missing tokens. Run `hermes auth` to re-authenticate.",
            "codex_auth_invalid_shape", relogin=True,
        )
    access_token = tokens.get("access_token")
    refresh_token = tokens.get("refresh_token")
    if not _nonempty_str(access_token):
        raise _codex_err(
            "Codex auth is missing access_token. Run `hermes auth` to re-authenticate.",
            "codex_auth_missing_access_token", relogin=True,
        )
    if not _nonempty_str(refresh_token):
        raise _codex_err(
            "Codex auth is missing refresh_token. Run `hermes auth` to re-authenticate.",
            "codex_auth_missing_refresh_token", relogin=True,
        )
    return {
        "tokens": tokens,
        "last_refresh": state.get("last_refresh"),
    }


def _sync_codex_pool_entries(
    auth_store: Dict[str, Any],
    tokens: Dict[str, str],
    last_refresh: Optional[str],
    previous_singleton_tokens: Optional[Dict[str, str]] = None,
) -> None:
    """Mirror a fresh Codex re-auth into the credential_pool OAuth entries.

    * ``device_code`` — the singleton-seeded entry written by the device-code OAuth flow when the
    user logged in via ``hermes setup`` / the model picker. Always synced with the fresh tokens. *
    ``manual:device_code`` — entries created by ``hermes auth add openai-codex`` that use the same
    device-code OAuth mechanism.

    * ``manual:api_key`` and any other non-device-code manual sources — those are independent
    credentials (an explicit API key, a different ChatGPT account, etc.) and must not be overwritten
    by a single re-auth.
    """
    access_token = tokens.get("access_token")
    if not access_token:
        return
    refresh_token = tokens.get("refresh_token")
    entries = _pool_entries(auth_store, "openai-codex")
    if entries is None:
        return
    # Previous singleton access_token (before this re-auth overwrote it) —
    # used to distinguish legacy singleton-aliases from independent accounts.
    # When None or empty, no manual entry can be treated as an alias (which
    # is the right default for first-ever-save or a freshly initialized
    # auth.json).
    prev_at = None
    if isinstance(previous_singleton_tokens, dict):
        prev_at = previous_singleton_tokens.get("access_token") or None
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        source = entry.get("source")
        if source == "device_code":
            # Singleton-seeded mirror — always refresh.
            refresh_this_entry = True
        elif source == "manual:device_code":
            # Refresh only if this entry's existing access_token matches the
            # previous singleton access_token (i.e. it is a true alias of the
            # singleton from the #33000 workaround era).  An entry with its
            # own distinct token material is an independent account and must
            # be left alone (#39236).
            refresh_this_entry = bool(
                prev_at and entry.get("access_token") == prev_at
            )
        else:
            # ``manual:api_key`` and any future non-device-code sources.
            refresh_this_entry = False
        if not refresh_this_entry:
            continue
        entry["access_token"] = access_token
        if refresh_token:
            entry["refresh_token"] = refresh_token
        if last_refresh:
            entry["last_refresh"] = last_refresh
        _clear_pool_entry_status(entry)


def _save_codex_tokens(tokens: Dict[str, str], last_refresh: str = None, label: str = None) -> None:
    """Save Codex OAuth tokens to Hermes auth store (~/.hermes/auth.json)."""
    if last_refresh is None:
        last_refresh = _utc_now_z()
    with _auth_store_lock():
        auth_store = _load_auth_store()
        state = _load_provider_state(auth_store, "openai-codex") or {}
        # Capture the previous singleton tokens BEFORE overwriting them.  The
        # pool-sync step uses this to distinguish legacy singleton-aliases
        # (which should be refreshed) from independent accounts that
        # ``hermes auth add openai-codex`` created (which must not be
        # overwritten — see #39236).
        previous_singleton_tokens = state.get("tokens") if isinstance(state.get("tokens"), dict) else None
        state["tokens"] = tokens
        state["last_refresh"] = last_refresh
        state["auth_mode"] = "chatgpt"
        if label and str(label).strip():
            state["label"] = str(label).strip()
        _save_provider_state(auth_store, "openai-codex", state)
        _sync_codex_pool_entries(
            auth_store,
            tokens,
            last_refresh,
            previous_singleton_tokens=previous_singleton_tokens,
        )
        _save_auth_store(auth_store)


def _recover_codex_tokens_from_cli(reason: str) -> Optional[Dict[str, str]]:
    """Adopt a valid Codex CLI token pair into Hermes auth, if available."""
    imported = _import_codex_cli_tokens()
    # Require BOTH tokens before adopting: persisting a payload without a
    # usable refresh_token would only break the next refresh cycle.
    if not (
        imported
        and str(imported.get("access_token", "") or "").strip()
        and str(imported.get("refresh_token", "") or "").strip()
    ):
        return None
    logger.info("Codex auth recovered from Codex CLI auth.json (%s).", reason)
    _save_codex_tokens(imported)
    return dict(imported)


def _refresh_payload_access_token(
    response: "httpx.Response",
    *,
    provider: str,
    invalid_json: Tuple[str, str],
    invalid_response: Optional[Tuple[str, str]],
    missing_access: Tuple[str, str],
    relogin_required: bool = True,
    invalid_json_relogin: Optional[bool] = None,
    strict_str: bool = True,
) -> Tuple[Dict[str, Any], str]:
    """Parse a 200 token-refresh response; return ``(payload, stripped access_token)``.

    Each ``(message, code)`` pair keeps the provider's historical wording; ``{exc}`` in
    *invalid_json*'s message is formatted with the JSON error. *strict_str* rejects non-string
    access tokens; otherwise they are ``str()``-coerced.
    """
    try:
        payload = response.json()
    except Exception as exc:
        raise AuthError(
            invalid_json[0].format(exc=exc),
            provider=provider,
            code=invalid_json[1],
            relogin_required=(
                relogin_required if invalid_json_relogin is None else invalid_json_relogin
            ),
        ) from exc
    if not isinstance(payload, dict):
        if invalid_response is None:
            payload = {}
        else:
            raise AuthError(
                invalid_response[0],
                provider=provider,
                code=invalid_response[1],
                relogin_required=relogin_required,
            )
    access = payload.get("access_token")
    if strict_str:
        access = access.strip() if isinstance(access, str) else ""
    else:
        access = str(access or "").strip()
    if not access:
        raise AuthError(
            missing_access[0],
            provider=provider,
            code=missing_access[1],
            relogin_required=relogin_required,
        )
    return payload, access


def _codex_http_client(**kwargs: Any) -> "httpx.Client":
    """Build an ``httpx.Client`` for Codex OAuth/probe endpoints with racing.

    Same broken-IPv6 failure mode as the chat transport (#13834): a host that advertises AAAA
    records but blackholes IPv6 makes each serial connect attempt eat the full connect timeout
    before IPv4 is tried, so token refresh / device login / usage probes time out where the official
    Codex CLI (which races families per RFC 8305) works.

    Best-effort: if the racing backend can't be installed (unexpected httpx/httpcore internals,
    mocked client in tests), the client still works with the default serial connect behavior.
    """
    client = httpx.Client(**kwargs)
    try:
        from agent.process_bootstrap import enable_happy_eyeballs_on_client

        enable_happy_eyeballs_on_client(client)
    except Exception:
        pass
    return client


def _codex_quota_exhausted_error(retry_after: Optional[int]) -> AuthError:
    if retry_after is not None:
        message = (
            f"Codex provider quota exhausted (429); retry after {retry_after}s. "
            "Credentials are still valid."
        )
    else:
        message = (
            "Codex provider quota exhausted (429). Credentials are still valid; "
            "retry after the usage limit resets."
        )
    return _codex_err(message, CODEX_RATE_LIMITED_CODE, relogin=False)


def _codex_refresh_failure_error(response: "httpx.Response") -> AuthError:
    """Decode a non-200 Codex token-refresh response into a shaped AuthError."""
    code = "codex_refresh_failed"
    message = f"Codex token refresh failed with status {response.status_code}."
    relogin_required = False
    try:
        err = response.json()
        if isinstance(err, dict):
            err_obj = err.get("error")
            # OpenAI shape: {"error": {"code": "...", "message": "...", "type": "..."}}
            if isinstance(err_obj, dict):
                nested_code = err_obj.get("code") or err_obj.get("type")
                if _nonempty_str(nested_code):
                    code = nested_code.strip()
                nested_msg = err_obj.get("message")
                if _nonempty_str(nested_msg):
                    message = f"Codex token refresh failed: {nested_msg.strip()}"
            # OAuth spec shape: {"error": "code_str", "error_description": "..."}
            elif _nonempty_str(err_obj):
                code = err_obj.strip()
                err_desc = err.get("error_description") or err.get("message")
                if _nonempty_str(err_desc):
                    message = f"Codex token refresh failed: {err_desc.strip()}"
    except Exception:
        pass
    if code in {"invalid_grant", "invalid_token", "invalid_request"}:
        relogin_required = True
    if code == "refresh_token_reused":
        message = (
            "Codex refresh token was already consumed by another client "
            "(e.g. Codex CLI or VS Code extension). "
            "Run `codex` in your terminal to generate fresh tokens, "
            "then run `hermes auth` to re-authenticate."
        )
        relogin_required = True
    # A 401/403 from the token endpoint always means the refresh token
    # is invalid/expired — force relogin even if the body error code
    # wasn't one of the known strings above.
    if response.status_code in {401, 403} and not relogin_required:
        relogin_required = True
    return _codex_err(message, code, relogin=relogin_required)


def refresh_codex_oauth_pure(
    access_token: str,
    refresh_token: str,
    *,
    timeout_seconds: float = 20.0,
) -> Dict[str, Any]:
    """Refresh Codex OAuth tokens without mutating Hermes auth state."""
    del access_token  # Access token is only used by callers to decide whether to refresh.
    if not _nonempty_str(refresh_token):
        raise _codex_err(
            "Codex auth is missing refresh_token. Run `hermes auth` to re-authenticate.",
            "codex_auth_missing_refresh_token", relogin=True,
        )

    timeout = httpx.Timeout(max(5.0, float(timeout_seconds)))
    with _codex_http_client(
        timeout=timeout,
        headers={
            "Accept": "application/json",
            "User-Agent": CODEX_OAUTH_USER_AGENT,
        },
    ) as client:
        response = client.post(
            CODEX_OAUTH_TOKEN_URL,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": CODEX_OAUTH_CLIENT_ID,
            },
        )

    if response.status_code == 429:
        # Upstream rate-limit / usage-quota exhaustion on the token endpoint.
        # The stored refresh token is still valid here — re-authenticating
        # cannot lift a quota cap. Classify distinctly from auth failures so
        # callers surface a "retry later" notice instead of a misleading
        # "run hermes auth" prompt (see issue #32790).
        raise _codex_quota_exhausted_error(
            _parse_retry_after_seconds(getattr(response, "headers", None))
        )

    if response.status_code != 200:
        raise _codex_refresh_failure_error(response)

    refresh_payload, refreshed_access = _refresh_payload_access_token(
        response,
        provider="openai-codex",
        invalid_json=("Codex token refresh returned invalid JSON.", "codex_refresh_invalid_json"),
        invalid_response=None,
        missing_access=(
            "Codex token refresh response was missing access_token.",
            "codex_refresh_missing_access_token",
        ),
    )

    updated = {
        "access_token": refreshed_access,
        "refresh_token": refresh_token.strip(),
        "last_refresh": _utc_now_z(),
    }
    next_refresh = refresh_payload.get("refresh_token")
    if _nonempty_str(next_refresh):
        updated["refresh_token"] = next_refresh.strip()
    return updated


def _refresh_codex_auth_tokens(
    tokens: Dict[str, str],
    timeout_seconds: float,
) -> Dict[str, str]:
    """Refresh Codex access token using the refresh token."""
    try:
        refreshed = refresh_codex_oauth_pure(
            str(tokens.get("access_token", "") or ""),
            str(tokens.get("refresh_token", "") or ""),
            timeout_seconds=timeout_seconds,
        )
    except AuthError as exc:
        # Self-heal cross-store refresh_token rotation. Hermes keeps its OWN
        # Codex OAuth token (per profile + top-level), separate from the Codex
        # CLI's ~/.codex/auth.json. OAuth refresh_tokens are single-use, so when
        # the Codex CLI (or another Hermes process) rotates the shared token,
        # this frozen copy's refresh_token goes stale and the refresh fails with
        # a relogin-required error (invalid_grant / refresh_token_reused / 401).
        # Before surfacing that as a hard 401 to the turn, adopt the canonical
        # fresh token from ~/.codex/auth.json (the Codex CLI keeps it current) so
        # idle profiles / desktop sessions recover automatically instead of
        # 401'ing until a manual re-auth. Transient failures (e.g. 429 quota)
        # keep relogin_required=False — the stored token is still valid there, so
        # we never self-heal those and re-raise unchanged.
        if not getattr(exc, "relogin_required", False):
            raise
        imported = _recover_codex_tokens_from_cli(
            f"refresh_token rejected: {getattr(exc, 'code', None) or 'auth_error'}"
        )
        if not imported:
            raise
        return imported

    updated_tokens = dict(tokens)
    updated_tokens["access_token"] = refreshed["access_token"]
    updated_tokens["refresh_token"] = refreshed["refresh_token"]

    _save_codex_tokens(updated_tokens)
    return updated_tokens


def _import_codex_cli_tokens() -> Optional[Dict[str, str]]:
    """Try to read tokens from ~/.codex/auth.json (Codex CLI shared file).

    Returns tokens dict if valid and not expired, None otherwise. Does NOT write to the shared file.
    """
    codex_home = os.getenv("CODEX_HOME", "").strip()
    if not codex_home:
        codex_home = str(Path.home() / ".codex")
    auth_path = Path(codex_home).expanduser() / "auth.json"
    if not auth_path.is_file():
        return None
    try:
        payload = json.loads(auth_path.read_text(encoding="utf-8-sig"))
        tokens = payload.get("tokens")
        if not isinstance(tokens, dict):
            return None
        access_token = tokens.get("access_token")
        refresh_token = tokens.get("refresh_token")
        if not access_token or not refresh_token:
            return None
        # Reject expired tokens — importing stale tokens from ~/.codex/
        # that can't be refreshed leaves the user stuck with "Login successful!"
        # but no working credentials.
        if _codex_access_token_is_expiring(access_token, 0):
            logger.debug(
                "Codex CLI tokens at %s are expired — skipping import.", auth_path,
            )
            return None
        return dict(tokens)
    except Exception:
        return None


def resolve_codex_runtime_credentials(
    *,
    force_refresh: bool = False,
    refresh_if_expiring: bool = True,
    refresh_skew_seconds: int = CODEX_ACCESS_TOKEN_REFRESH_SKEW_SECONDS,
) -> Dict[str, Any]:
    """Resolve runtime credentials from Hermes's own Codex token store.

    Falls back to the credential pool when the singleton (``providers.openai-codex.tokens``) has no
    usable access_token but the pool (``credential_pool.openai-codex``) does.
    """
    read_error: Optional[AuthError] = None
    try:
        data = _read_codex_tokens()
    except AuthError as exc:
        read_error = exc
        if getattr(exc, "relogin_required", False) and getattr(exc, "code", None) in {
            "codex_auth_missing_access_token",
            "codex_auth_missing_refresh_token",
            "codex_auth_invalid_shape",
        }:
            imported = _recover_codex_tokens_from_cli(str(getattr(exc, "code", None) or "auth_error"))
            if imported:
                data = {"tokens": imported, "last_refresh": imported.get("last_refresh")}
            else:
                data = None
        else:
            data = None

    if data is None:
        pool_token = _pool_codex_access_token()
        if pool_token:
            return _codex_runtime_result(pool_token, source="credential_pool", last_refresh=None)
        pool_rate_limit = _codex_pool_rate_limit_status()
        if pool_rate_limit:
            # Before surfacing the persisted cooldown, ask the Codex usage
            # endpoint whether the quota actually reset early (banked reset
            # redeemed, plan upgraded, window reset upstream).  The persisted
            # ``last_error_reset_at`` can be days in the future while the
            # account is already usable again — see issue #43747.
            stale_token = str(pool_rate_limit.get("access_token") or "").strip()
            if stale_token and _probe_codex_quota_restored(
                stale_token,
                base_url=pool_rate_limit.get("base_url"),
            ):
                logger.info(
                    "Codex quota restored upstream — clearing stale pool cooldown(s)."
                )
                clear_codex_pool_quota_cooldowns()
                pool_token = _pool_codex_access_token()
                if pool_token:
                    return _codex_runtime_result(pool_token, source="credential_pool", last_refresh=None)
            reset_at = pool_rate_limit.get("reset_at")
            remaining = (
                int(reset_at - time.time())
                if isinstance(reset_at, (int, float)) and reset_at > time.time()
                else None
            )
            raise _codex_quota_exhausted_error(remaining)
        if read_error is not None:
            raise read_error
        raise _codex_err(
            "No Codex credentials stored. Run `hermes auth` to authenticate.",
            "codex_auth_missing", relogin=True,
        )

    tokens = dict(data["tokens"])
    access_token = str(tokens.get("access_token", "") or "").strip()
    refresh_timeout_seconds = env_float("HERMES_CODEX_REFRESH_TIMEOUT_SECONDS", 20)

    should_refresh = bool(force_refresh)
    if (not should_refresh) and refresh_if_expiring:
        should_refresh = _codex_access_token_is_expiring(access_token, refresh_skew_seconds)
    if should_refresh:
        # Re-read under lock to avoid racing with other Hermes processes
        with _auth_store_lock(timeout_seconds=max(float(AUTH_LOCK_TIMEOUT_SECONDS), refresh_timeout_seconds + 5.0)):
            data = _read_codex_tokens(_lock=False)
            tokens = dict(data["tokens"])
            access_token = str(tokens.get("access_token", "") or "").strip()

            should_refresh = bool(force_refresh)
            if (not should_refresh) and refresh_if_expiring:
                should_refresh = _codex_access_token_is_expiring(access_token, refresh_skew_seconds)

            if should_refresh:
                tokens = _refresh_codex_auth_tokens(tokens, refresh_timeout_seconds)
                access_token = str(tokens.get("access_token", "") or "").strip()

    return _codex_runtime_result(
        access_token, source="hermes-auth-store", last_refresh=data.get("last_refresh"),
    )


def _is_codex_rate_limit_shaped(
    code: Any,
    reason: Any,
    message: Any,
) -> bool:
    """True when persisted pool-entry error metadata describes a 429/quota stop."""
    reason_l = str(reason or "").lower()
    message_l = str(message or "").lower()
    return (
        code == 429
        or "rate_limit" in reason_l
        or "usage_limit" in reason_l
        or "quota" in reason_l
        or "rate limit" in message_l
        or "usage limit" in message_l
        or "quota" in message_l
    )


# Throttle for the live Codex quota probe below.  The probe runs on the hot
# credential-selection path while the pool is exhausted, so without a floor a
# busy gateway would hammer the usage endpoint on every model/auxiliary call.
CODEX_QUOTA_PROBE_MIN_INTERVAL_SECONDS = 300  # 5 minutes
_codex_quota_probe_cache: Dict[str, Tuple[float, Optional[bool]]] = {}
_codex_quota_probe_lock = threading.Lock()


def _codex_usage_probe_url(base_url: Optional[str]) -> str:
    """Resolve the Codex usage endpoint for a probe.

    Mirrors the Codex CLI's PathStyle split: base URLs containing ``/backend-api`` use the ChatGPT
    ``/wham/usage`` path, everything else ``/api/codex/usage``. Kept local so this low-level auth
    module does not import the auxiliary account-usage module.
    """
    normalized = str(base_url or "").strip().rstrip("/")
    if not normalized:
        normalized = _codex_base_url()
    if normalized.endswith("/codex"):
        normalized = normalized[: -len("/codex")]
    prefix = normalized + ("/wham" if "/backend-api" in normalized else "/api/codex")
    return prefix + "/usage"


def _probe_codex_quota_restored(
    access_token: Any,
    *,
    base_url: Optional[str] = None,
    min_interval_seconds: float = CODEX_QUOTA_PROBE_MIN_INTERVAL_SECONDS,
) -> Optional[bool]:
    """Ask the Codex usage endpoint whether this account's quota is usable again.

    Probes are throttled per access token (module-local cache) so the hot selection path can fire
    this freely.
    """
    token = str(access_token or "").strip()
    if not token:
        return None
    # Real Codex access tokens are JWTs. Refusing to probe non-JWT tokens
    # avoids pointless network calls for corrupt/placeholder entries (and
    # keeps hermetic test fixtures with dummy tokens offline).
    if not _decode_jwt_claims(token):
        return None
    cache_key = hashlib.sha256(token.encode("utf-8")).hexdigest()[:16]
    now = time.monotonic()
    with _codex_quota_probe_lock:
        cached = _codex_quota_probe_cache.get(cache_key)
        if cached is not None and (now - cached[0]) < min_interval_seconds:
            return cached[1]
        # Reserve the slot immediately so concurrent selectors don't stampede
        # the endpoint while this probe is in flight.
        _codex_quota_probe_cache[cache_key] = (now, None)

    result: Optional[bool] = None
    try:
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
            "User-Agent": "codex-cli",
        }
        # Best-effort ChatGPT-Account-Id from the JWT (the backend requires it
        # for some account shapes; harmless to omit for others).
        claims = _decode_jwt_claims(token)
        account_id = (
            claims.get("https://api.openai.com/auth", {}).get("chatgpt_account_id")
            if isinstance(claims.get("https://api.openai.com/auth"), dict)
            else None
        )
        if _nonempty_str(account_id):
            headers["ChatGPT-Account-Id"] = account_id.strip()
        with _codex_http_client(timeout=10.0) as client:
            response = client.get(_codex_usage_probe_url(base_url), headers=headers)
        if response.status_code == 200:
            payload = response.json() or {}
            rate_limit = payload.get("rate_limit") or {}
            worst_used: Optional[float] = None
            for key in ("primary_window", "secondary_window"):
                used = (rate_limit.get(key) or {}).get("used_percent")
                if isinstance(used, (int, float)):
                    worst_used = max(worst_used or 0.0, float(used))
            if worst_used is not None:
                result = worst_used < 100.0
        elif response.status_code == 429:
            result = False
    except Exception:
        logger.debug("Codex quota probe failed", exc_info=True)
        result = None

    with _codex_quota_probe_lock:
        _codex_quota_probe_cache[cache_key] = (now, result)
    return result


def clear_codex_pool_quota_cooldowns(access_token: Optional[str] = None) -> int:
    """Clear rate-limit cooldowns on persisted openai-codex pool entries.

    Called after the upstream quota is KNOWN to be restored (a successful ``/usage reset``
    redemption, or a positive live probe) so auth.json stops freezing credentials behind a stale
    ``last_error_reset_at``.

    When *access_token* is given, only the matching entry is cleared; otherwise every rate-limited
    entry clears (a redeemed banked reset restores the whole account, and any entry that is
    genuinely still exhausted just re-freezes with fresh metadata on its next 429).
    """
    cleared = 0
    try:
        with _auth_store_lock():
            auth_store = _load_auth_store()
            entries = _pool_entries(auth_store, "openai-codex")
            if entries is None:
                return 0
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                if entry.get("last_status") != "exhausted":
                    continue
                if access_token and str(entry.get("access_token") or "") != access_token:
                    continue
                if not _is_codex_rate_limit_shaped(
                    entry.get("last_error_code"),
                    entry.get("last_error_reason"),
                    entry.get("last_error_message"),
                ):
                    continue
                _clear_pool_entry_status(entry)
                cleared += 1
            if cleared:
                _save_auth_store(auth_store)
    except Exception:
        logger.debug("Failed to clear Codex pool quota cooldowns", exc_info=True)
    return cleared


def _codex_pool_rate_limit_status() -> Optional[Dict[str, Any]]:
    """Return metadata for a pool-only Codex credential in quota cooldown."""
    def _parse_reset_at(value: Any) -> Optional[float]:
        if value is None or value == "":
            return None
        if isinstance(value, (int, float)):
            numeric = float(value)
            if numeric <= 0:
                return None
            return numeric / 1000.0 if numeric > 1_000_000_000_000 else numeric
        if isinstance(value, str):
            raw = value.strip()
            if not raw:
                return None
            try:
                numeric = float(raw)
            except ValueError:
                numeric = None
            if numeric is not None:
                return numeric / 1000.0 if numeric > 1_000_000_000_000 else numeric
            try:
                return datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()
            except ValueError:
                return None
        return None

    try:
        with _auth_store_lock():
            auth_store = _load_auth_store()
        entries = _pool_entries(auth_store, "openai-codex")
        if entries is None:
            return None
        now = time.time()
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            token = entry.get("access_token")
            if not _nonempty_str(token):
                continue
            if entry.get("last_status") != "exhausted":
                continue
            if not _is_codex_rate_limit_shaped(
                entry.get("last_error_code"),
                entry.get("last_error_reason"),
                entry.get("last_error_message"),
            ):
                continue
            reset_at = _parse_reset_at(entry.get("last_error_reset_at"))
            if reset_at is not None and reset_at <= now:
                continue
            return {
                "label": entry.get("label"),
                "last_refresh": entry.get("last_refresh"),
                "reset_at": reset_at,
                "reason": entry.get("last_error_reason"),
                "message": entry.get("last_error_message"),
                "access_token": token.strip(),
                "base_url": entry.get("base_url"),
            }
    except Exception:
        logger.debug("Codex pool rate-limit lookup failed", exc_info=True)
    return None


def _pool_entries(auth_store: Dict[str, Any], provider_id: str) -> Optional[List[Any]]:
    """``auth_store["credential_pool"][provider_id]`` when it is a list, else None."""
    pool = auth_store.get("credential_pool")
    entries = pool.get(provider_id) if isinstance(pool, dict) else None
    return entries if isinstance(entries, list) else None


def _pool_codex_access_token() -> str:
    """Return the most-recent usable access_token from the openai-codex pool.

    Used as a fallback by ``resolve_codex_runtime_credentials`` when the singleton has no creds.
    Reads ``credential_pool.openai-codex`` entries directly from auth.json and picks the first non-
    empty access_token, preferring entries that are not currently in an exhaustion cooldown.
    """
    try:
        with _auth_store_lock():
            auth_store = _load_auth_store()
        entries = _pool_entries(auth_store, "openai-codex")
        if entries is None:
            return ""

        def _entry_usable(entry: Dict[str, Any]) -> bool:
            if not isinstance(entry, dict):
                return False
            token = entry.get("access_token")
            if not _nonempty_str(token):
                return False
            # Skip entries currently in an exhaustion cooldown window.
            reset_at = entry.get("last_error_reset_at")
            return not (isinstance(reset_at, (int, float)) and reset_at > time.time())

        for entry in entries:
            if _entry_usable(entry):
                return str(entry.get("access_token", "")).strip()
    except Exception:
        logger.debug("Codex pool fallback lookup failed", exc_info=True)
    return ""


# =============================================================================
# xAI Grok OAuth — tokens stored in ~/.hermes/auth.json
# =============================================================================

def _xai_oauth_state_from_store(auth_store: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return usable xAI OAuth state from provider state or credential pool."""
    state = _load_provider_state(auth_store, "xai-oauth")
    tokens = state.get("tokens") if isinstance(state, dict) else None
    if isinstance(tokens, dict):
        access_token = str(tokens.get("access_token", "") or "").strip()
        refresh_token = str(tokens.get("refresh_token", "") or "").strip()
        if access_token and refresh_token:
            return state

    credential_pool = auth_store.get("credential_pool")
    entries = (
        credential_pool.get("xai-oauth")
        if isinstance(credential_pool, dict)
        else None
    )
    if isinstance(entries, list):
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            access_token = str(entry.get("access_token", "") or "").strip()
            refresh_token = str(entry.get("refresh_token", "") or "").strip()
            if not access_token or not refresh_token:
                continue
            merged = dict(state or {})
            merged["tokens"] = {
                "access_token": access_token,
                "refresh_token": refresh_token,
                "token_type": str(entry.get("token_type") or "Bearer"),
            }
            if entry.get("last_refresh"):
                merged["last_refresh"] = entry.get("last_refresh")
            merged.setdefault("auth_mode", "oauth_pkce")
            return merged

    return state if isinstance(state, dict) else None


def _xai_oauth_state_has_usable_tokens(state: Optional[Dict[str, Any]]) -> bool:
    tokens = state.get("tokens") if isinstance(state, dict) else None
    return (
        isinstance(tokens, dict)
        and bool(str(tokens.get("access_token", "") or "").strip())
        and bool(str(tokens.get("refresh_token", "") or "").strip())
    )


def _read_xai_oauth_tokens(*, _lock: bool = True) -> Dict[str, Any]:
    auth_store = _load_auth_store_maybe_locked(_lock)
    state = _xai_oauth_state_from_store(auth_store)
    if not _xai_oauth_state_has_usable_tokens(state):
        global_state = _xai_oauth_state_from_store(_load_global_auth_store())
        if _xai_oauth_state_has_usable_tokens(global_state):
            state = global_state
    if not state:
        raise _xai_err(
            "No xAI OAuth credentials stored. Select xAI Grok OAuth (SuperGrok / Premium+) in `hermes model`.",
            "xai_auth_missing", relogin=True,
        )
    tokens = state.get("tokens")
    if not isinstance(tokens, dict):
        raise _xai_err(
            "xAI OAuth state is missing tokens. Re-authenticate with `hermes model`.",
            "xai_auth_invalid_shape", relogin=True,
        )
    access_token = str(tokens.get("access_token", "") or "").strip()
    refresh_token = str(tokens.get("refresh_token", "") or "").strip()
    if not access_token:
        raise _xai_err(
            "xAI OAuth state is missing access_token. Re-authenticate with `hermes model`.",
            "xai_auth_missing_access_token", relogin=True,
        )
    if not refresh_token:
        raise _xai_err(
            "xAI OAuth state is missing refresh_token. Re-authenticate with `hermes model`.",
            "xai_auth_missing_refresh_token", relogin=True,
        )
    return {
        "tokens": tokens,
        "last_refresh": state.get("last_refresh"),
        "discovery": state.get("discovery") or {},
        "redirect_uri": state.get("redirect_uri"),
    }


def _write_through_xai_oauth_to_global_root(state: Dict[str, Any]) -> None:
    """Persist a rotated xAI OAuth ``state`` into the global-root auth.json.

    Best-effort write-through for the multi-profile rotation hazard (#43589): xAI rotates the
    refresh_token on every refresh, so when a profile session refreshes a grant it resolved from the
    root fallback, the rotated chain must land back in root.

    Only updates ``providers.xai-oauth`` in the root store; never touches the profile store (the
    caller already saved that). Swallows all errors — a failed write-through degrades to the pre-
    existing behavior (root stale), it must never break the profile's own successful save.
    """
    global_path = _global_auth_file_path()
    if global_path is None:
        # Classic mode (profile == root); the profile save already hit root.
        return
    # Seat belt: under pytest, refuse to write the real user's
    # ~/.hermes/auth.json even when HERMES_HOME points at a profile path
    # (mirrors the read-side guard in _load_global_auth_store). Uses the
    # unmodified HOME env, not Path.home() which fixtures may monkeypatch.
    if os.environ.get("PYTEST_CURRENT_TEST"):
        real_home_env = os.environ.get("HOME", "")
        if real_home_env:
            real_root = Path(real_home_env) / ".hermes" / "auth.json"
            try:
                if global_path.resolve(strict=False) == real_root.resolve(strict=False):
                    return
            except Exception:
                return
    try:
        _persist_provider_state_to_store(
            "xai-oauth",
            state,
            global_path,
            set_active=False,
        )
    except Exception as exc:  # pragma: no cover - best effort
        logger.debug("xAI OAuth: write-through to global root failed: %s", exc)


def _save_xai_oauth_tokens(
    tokens: Dict[str, Any],
    *,
    discovery: Optional[Dict[str, Any]] = None,
    redirect_uri: str = "",
    last_refresh: Optional[str] = None,
    auth_mode: str = "oauth_device_code",
    set_active: bool = True,
) -> None:
    """Persist xAI OAuth tokens into the auth store.

    When *set_active* is True (default), also promote ``xai-oauth`` to ``active_provider`` —
    appropriate for intentional model/auth login. Pass ``set_active=False`` for side-tool credential
    bootstrap (TTS/setup, tools config, dashboard token save, token refresh) so inference routing is
    unchanged.
    """
    if last_refresh is None:
        last_refresh = _utc_now_z()
    with _auth_store_lock():
        auth_store = _load_auth_store()
        # A profile that lacks its own xai-oauth block is reading the root
        # grant through _load_provider_state's fallback. When such a profile
        # refreshes the (rotating) grant, we must write the rotated chain back
        # to root too, or root is left holding a revoked refresh token (#43589).
        # #74339: the old key-presence check (_profile_has_own_xai_oauth_state)
        # decided write-through based on whether the profile had a
        # providers.xai-oauth key BEFORE the save — but _store_provider_state
        # unconditionally creates that key below. Use
        # _load_provider_state_with_source to learn where the grant was
        # resolved from and write back only to that source.
        state, source_path = _load_provider_state_with_source(
            auth_store, "xai-oauth"
        )
        if state is None:
            state = {}
        state["tokens"] = tokens
        state["last_refresh"] = last_refresh
        state["auth_mode"] = auth_mode
        if discovery:
            state["discovery"] = discovery
        if redirect_uri:
            state["redirect_uri"] = redirect_uri
        global_root = _global_auth_file_path()
        is_from_root = bool(
            source_path is not None
            and global_root is not None
            and _same_path(source_path, global_root)
        )
        if is_from_root:
            # Grant was resolved from root — write back to root only.
            # Do NOT call _store_provider_state on the profile auth_store
            # (it would create a shadowing providers.xai-oauth key that
            # disables write-through on the next refresh — #74339).
            _write_through_xai_oauth_to_global_root(state)
        else:
            # Profile genuinely owns this — write to profile store.
            _store_provider_state(
                auth_store, "xai-oauth", state, set_active=set_active
            )
            _save_auth_store(auth_store)


def _xai_access_token_is_expiring(access_token: str, skew_seconds: int = 0) -> bool:
    if not isinstance(access_token, str) or "." not in access_token:
        return False
    try:
        parts = access_token.split(".")
        if len(parts) < 2:
            return False
        payload_b64 = parts[1]
        payload_b64 += "=" * (-len(payload_b64) % 4)
        payload = json.loads(base64.urlsafe_b64decode(payload_b64.encode("ascii")).decode("utf-8"))
        exp = payload.get("exp")
        if not isinstance(exp, (int, float)):
            return False
        return float(exp) <= (time.time() + max(0, int(skew_seconds)))
    except Exception:
        return False


def _xai_proactive_refresh_skew_seconds(access_token: str) -> int:
    """How far before JWT ``exp`` to proactively refresh xAI OAuth tokens.

    SuperGrok sessions ship multi-hour tokens where the gateway-oriented hour-long skew makes sense,
    but device-code logins often return ~15-minute JWTs; the full skew would force a refresh on
    every credential resolution, burning single-use refresh tokens and racing concurrent callers
    into ``invalid_grant`` quarantine.
    """
    max_skew = XAI_ACCESS_TOKEN_REFRESH_SKEW_SECONDS
    if not isinstance(access_token, str) or "." not in access_token:
        return max_skew
    try:
        parts = access_token.split(".")
        if len(parts) < 2:
            return max_skew
        payload_b64 = parts[1]
        payload_b64 += "=" * (-len(payload_b64) % 4)
        payload = json.loads(base64.urlsafe_b64decode(payload_b64.encode("ascii")).decode("utf-8"))
        exp = payload.get("exp")
        if not isinstance(exp, (int, float)):
            return max_skew
        remaining = float(exp) - time.time()
        if remaining <= 0:
            return max_skew
        if remaining <= 45 * 60:
            return min(120, max_skew)
        return max_skew
    except Exception:
        return max_skew


def _is_xai_origin_host(host: str) -> bool:
    """``x.ai`` is the bare apex, so an exact match or any ``.x.ai`` suffix is accepted."""
    return host == "x.ai" or host.endswith(".x.ai")


def _xai_validate_oauth_endpoint(url: str, *, field: str) -> str:
    """Refuse any OIDC discovery endpoint that isn't HTTPS on the xAI origin.

    The discovery result is cached in auth.json, so a single MITM at login could plant a malicious
    ``token_endpoint`` that receives the refresh_token forever. Pinning scheme + host (RFC 8414 §2:
    HTTPS issuer, same-origin token_endpoint) removes that persistence; ``x.ai`` is the bare apex,
    so an exact match or any ``.x.ai`` suffix is accepted.
    """
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise _xai_err(
            f"xAI OIDC discovery returned a non-HTTPS {field}: {url!r}.",
            "xai_discovery_invalid",
        )
    host = (parsed.hostname or "").lower()
    if not host:
        raise _xai_err(
            f"xAI OIDC discovery {field} is missing a hostname: {url!r}.",
            "xai_discovery_invalid",
        )
    if not _is_xai_origin_host(host):
        raise _xai_err(
            f"xAI OIDC discovery {field} host {host!r} is not on the xAI origin "
            f"(expected x.ai or a *.x.ai subdomain). Refusing to use a cached "
            f"endpoint that may have been substituted by a MITM during initial "
            f"discovery; re-authenticate with `hermes model` to re-fetch.",
            "xai_discovery_invalid",
        )
    return url


def _xai_validate_inference_base_url(value: str, *, fallback: str) -> str:
    """Refuse a non-xAI base_url for the OAuth-authenticated inference path.

    Pin the inference origin to ``api.x.ai`` (or any ``*.x.ai`` subdomain xAI may add). On
    rejection, fall back to the default and log a warning rather than raise — a bad env var should
    not deadlock authentication, but it should also never leak the bearer.

    ``value`` is the already-stripped, trailing-slash-trimmed candidate from env. Empty input
    returns ``fallback`` unchanged.
    """
    candidate = (value or "").strip().rstrip("/")
    if not candidate:
        return fallback
    try:
        parsed = urlparse(candidate)
    except Exception:
        logger.warning(
            "Ignoring malformed xAI base_url override %r; using %s instead.",
            candidate, fallback,
        )
        return fallback
    if parsed.scheme != "https":
        logger.warning(
            "Refusing non-HTTPS xAI base_url override %r (xai-oauth bearer would "
            "be sent in cleartext); falling back to %s.",
            candidate, fallback,
        )
        return fallback
    host = (parsed.hostname or "").lower()
    if not host:
        logger.warning(
            "Ignoring xAI base_url override %r with no hostname; using %s instead.",
            candidate, fallback,
        )
        return fallback
    if not _is_xai_origin_host(host):
        logger.warning(
            "Refusing xAI base_url override %r — host %r is not on the xAI origin "
            "(expected x.ai or a *.x.ai subdomain). The xai-oauth bearer is only "
            "valid against xAI's inference API; sending it elsewhere would leak "
            "the credential. Falling back to %s.",
            candidate, host, fallback,
        )
        return fallback
    return candidate


def _xai_oauth_discovery(timeout_seconds: float = 15.0) -> Dict[str, str]:
    try:
        response = httpx.get(
            XAI_OAUTH_DISCOVERY_URL,
            headers={"Accept": "application/json"},
            timeout=timeout_seconds,
        )
    except Exception as exc:
        raise _xai_err(f"xAI OIDC discovery failed: {exc}", "xai_discovery_failed") from exc
    if response.status_code != 200:
        raise _xai_err(
            f"xAI OIDC discovery returned status {response.status_code}.",
            "xai_discovery_failed",
        )
    try:
        payload = response.json()
    except Exception as exc:
        raise _xai_err(
            f"xAI OIDC discovery returned invalid JSON: {exc}",
            "xai_discovery_invalid_json",
        ) from exc
    if not isinstance(payload, dict):
        raise _xai_err(
            "xAI OIDC discovery response was not a JSON object.",
            "xai_discovery_incomplete",
        )
    authorization_endpoint = str(payload.get("authorization_endpoint", "") or "").strip()
    token_endpoint = str(payload.get("token_endpoint", "") or "").strip()
    if not authorization_endpoint or not token_endpoint:
        raise _xai_err(
            "xAI OIDC discovery response was missing required endpoints.",
            "xai_discovery_incomplete",
        )
    _xai_validate_oauth_endpoint(authorization_endpoint, field="authorization_endpoint")
    _xai_validate_oauth_endpoint(token_endpoint, field="token_endpoint")
    return {
        "authorization_endpoint": authorization_endpoint,
        "token_endpoint": token_endpoint,
    }


def _xai_tokens_from_payload(payload: Dict[str, Any], access_token: str, fallback_refresh: str) -> Dict[str, Any]:
    """Token block persisted for xAI OAuth; falls back to *fallback_refresh* when none is rotated in."""
    return {
        "access_token": access_token,
        "refresh_token": str(payload.get("refresh_token") or fallback_refresh).strip(),
        "id_token": str(payload.get("id_token") or "").strip(),
        "expires_in": payload.get("expires_in"),
        "token_type": str(payload.get("token_type") or "Bearer").strip() or "Bearer",
    }


def refresh_xai_oauth_pure(
    access_token: str,
    refresh_token: str,
    *,
    token_endpoint: str = "",
    timeout_seconds: float = 20.0,
) -> Dict[str, Any]:
    del access_token
    if not _nonempty_str(refresh_token):
        raise _xai_err(
            "xAI OAuth is missing refresh_token. Re-authenticate with `hermes model`.",
            "xai_auth_missing_refresh_token", relogin=True,
        )
    endpoint = token_endpoint.strip() or _xai_oauth_discovery(timeout_seconds)["token_endpoint"]
    # Re-validate cached endpoints on the refresh hot path: an auth.json
    # written by an older Hermes (or hand-edited) may carry a non-xAI
    # token_endpoint that would receive every future refresh_token in
    # plaintext if we trusted it blindly. Cheap suffix check; fast-fail
    # with a clear error so the user can re-run `hermes model` to refetch.
    _xai_validate_oauth_endpoint(endpoint, field="token_endpoint")
    timeout = httpx.Timeout(max(5.0, float(timeout_seconds)))
    with httpx.Client(timeout=timeout, headers={"Accept": "application/json"}) as client:
        response = client.post(
            endpoint,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={
                "grant_type": "refresh_token",
                "client_id": XAI_OAUTH_CLIENT_ID,
                "refresh_token": refresh_token,
            },
        )
    if response.status_code != 200:
        detail = response.text.strip()
        # ``403`` from xAI's token endpoint is almost always a tier /
        # entitlement gate (the OAuth grant exists but the account isn't
        # on the allowlist for API access).  Re-running ``hermes model``
        # won't fix that — surface a separate error code so
        # ``format_auth_error`` doesn't append a misleading
        # re-authenticate hint, and point users at the ``XAI_API_KEY``
        # fallback.  See #26847.
        if response.status_code == 403:
            raise _xai_err(
                "xAI token refresh failed with HTTP 403."
                + (f" Response: {detail}" if detail else "")
                + " This OAuth account is not authorized for xAI API"
                  " access — xAI may be restricting API/OAuth use to"
                  " specific SuperGrok tiers despite the in-app"
                  " subscription being active. Re-logging in won't"
                  " change that; set ``XAI_API_KEY`` and switch to"
                  " ``provider: xai`` (API-key path) if available, or"
                  " upgrade your subscription at https://x.ai/grok.",
                "xai_oauth_tier_denied", relogin=False,
            )
        raise _xai_err(
            "xAI token refresh failed."
            + (f" Response: {detail}" if detail else ""),
            "xai_refresh_failed", relogin=response.status_code in {400, 401},
        )
    payload, refreshed_access = _refresh_payload_access_token(
        response,
        provider="xai-oauth",
        invalid_json=("xAI token refresh returned invalid JSON: {exc}", "xai_refresh_invalid_json"),
        invalid_json_relogin=False,
        strict_str=False,
        invalid_response=(
            "xAI token refresh response was not a JSON object.",
            "xai_refresh_invalid_response",
        ),
        missing_access=(
            "xAI token refresh response was missing access_token.",
            "xai_refresh_missing_access_token",
        ),
    )
    return {
        **_xai_tokens_from_payload(payload, refreshed_access, refresh_token),
        "last_refresh": _utc_now_z(),
    }


def _refresh_xai_oauth_tokens(
    tokens: Dict[str, Any],
    *,
    token_endpoint: str,
    redirect_uri: str = "",
    timeout_seconds: float,
) -> Dict[str, Any]:
    # Re-persist whatever auth_mode is already stored (legacy pre-device-code
    # logins may still carry ``oauth_pkce``): the refresh hot path must not
    # relabel how the grant was originally obtained.
    try:
        state = _load_provider_state(_load_auth_store(), "xai-oauth") or {}
        auth_mode = str(state.get("auth_mode") or "oauth_device_code")
    except Exception:
        auth_mode = "oauth_device_code"
    refreshed = refresh_xai_oauth_pure(
        str(tokens.get("access_token", "") or ""),
        str(tokens.get("refresh_token", "") or ""),
        token_endpoint=token_endpoint,
        timeout_seconds=timeout_seconds,
    )
    updated_tokens = dict(tokens)
    updated_tokens["access_token"] = refreshed["access_token"]
    updated_tokens["refresh_token"] = refreshed["refresh_token"]
    if refreshed.get("id_token"):
        updated_tokens["id_token"] = refreshed["id_token"]
    if refreshed.get("expires_in") is not None:
        updated_tokens["expires_in"] = refreshed["expires_in"]
    if refreshed.get("token_type"):
        updated_tokens["token_type"] = refreshed["token_type"]
    _save_xai_oauth_tokens(
        updated_tokens,
        discovery={"token_endpoint": token_endpoint},
        redirect_uri=redirect_uri,
        last_refresh=refreshed["last_refresh"],
        auth_mode=auth_mode,
        # Refresh must not flip active_provider — TTS/side tools can refresh
        # xAI tokens while chat still routes through another provider.
        set_active=False,
    )
    return updated_tokens


def _quarantine_xai_oauth_tokens(exc: AuthError) -> None:
    """Clear dead xAI tokens from auth.json after a terminal refresh failure.

    Terminal = HTTP 400/401/403 (invalid_grant, token revoked). Subsequent sessions then fail fast
    without a network retry. Mirrors credential_pool.py quarantine. Best-effort: persistence
    failures are logged and swallowed (caller re-raises the original error regardless).
    """
    try:
        _q_store = _load_auth_store()
        _q_state = _load_provider_state(_q_store, "xai-oauth") or {}
        _q_tokens = dict(_q_state.get("tokens") or {})
        _q_tokens.pop("access_token", None)
        _q_tokens.pop("refresh_token", None)
        _q_state["tokens"] = _q_tokens
        _q_state["last_auth_error"] = _last_auth_error_marker(
            "xai-oauth", exc,
            reason="runtime_refresh_failure", default_code="xai_refresh_failed",
        )
        _store_provider_state(_q_store, "xai-oauth", _q_state, set_active=False)
        _save_auth_store(_q_store)
    except Exception as _save_exc:
        logger.debug(
            "xAI OAuth: failed to persist quarantined state: %s", _save_exc,
        )


def _xai_oauth_inference_base_url() -> str:
    return _xai_validate_inference_base_url(
        os.getenv("HERMES_XAI_BASE_URL", "").strip().rstrip("/")
        or os.getenv("XAI_BASE_URL", "").strip().rstrip("/"),
        fallback=DEFAULT_XAI_OAUTH_BASE_URL,
    )


def resolve_xai_oauth_runtime_credentials(
    *,
    force_refresh: bool = False,
    refresh_if_expiring: bool = True,
    refresh_skew_seconds: Optional[int] = None,
) -> Dict[str, Any]:
    def _view(data: Dict[str, Any]) -> tuple[Dict[str, Any], str, str, str, bool]:
        tokens = dict(data["tokens"])
        access_token = str(tokens.get("access_token", "") or "").strip()
        discovery = dict(data.get("discovery") or {})
        token_endpoint = str(discovery.get("token_endpoint", "") or "").strip()
        redirect_uri = str(data.get("redirect_uri", "") or "").strip()
        effective_skew = (
            int(refresh_skew_seconds)
            if refresh_skew_seconds is not None
            else _xai_proactive_refresh_skew_seconds(access_token)
        )
        should_refresh = bool(force_refresh)
        if (not should_refresh) and refresh_if_expiring:
            should_refresh = _xai_access_token_is_expiring(access_token, effective_skew)
        return tokens, access_token, token_endpoint, redirect_uri, should_refresh

    data = _read_xai_oauth_tokens()
    refresh_timeout_seconds = env_float("HERMES_XAI_REFRESH_TIMEOUT_SECONDS", 20)
    tokens, access_token, token_endpoint, redirect_uri, should_refresh = _view(data)
    if should_refresh:
        with _auth_store_lock(timeout_seconds=max(float(AUTH_LOCK_TIMEOUT_SECONDS), refresh_timeout_seconds + 5.0)):
            data = _read_xai_oauth_tokens(_lock=False)
            tokens, access_token, token_endpoint, redirect_uri, should_refresh = _view(data)
            if should_refresh:
                if not token_endpoint:
                    token_endpoint = _xai_oauth_discovery(refresh_timeout_seconds)["token_endpoint"]
                try:
                    tokens = _refresh_xai_oauth_tokens(
                        tokens,
                        token_endpoint=token_endpoint,
                        redirect_uri=redirect_uri,
                        timeout_seconds=refresh_timeout_seconds,
                    )
                    access_token = str(tokens.get("access_token", "") or "").strip()
                except AuthError as exc:
                    if _is_terminal_xai_oauth_refresh_error(exc):
                        _quarantine_xai_oauth_tokens(exc)
                    raise

    base_url = _xai_oauth_inference_base_url()
    return {
        "provider": "xai-oauth",
        "base_url": base_url,
        "api_key": access_token,
        "source": "hermes-auth-store",
        "last_refresh": data.get("last_refresh"),
        # Display/telemetry only. Device-code is the only supported xAI OAuth
        # flow, so report it unconditionally — auth.json may still carry a
        # legacy ``oauth_pkce`` label, which the refresh path preserves as-is.
        "auth_mode": "oauth_device_code",
    }


# =============================================================================
# TLS verification helper
# =============================================================================

def _default_verify() -> bool | ssl.SSLContext:
    """Platform-aware default SSL verify for httpx clients.

    On macOS with Homebrew Python, the system OpenSSL cannot locate the system trust store and valid
    public certs fail verification. When certifi is importable we pin its bundle explicitly;
    elsewhere we defer to httpx's built-in default (certifi via its own dependency). Mirrors the
    weixin fix in 3a0ec1d93.
    """
    if sys.platform == "darwin":
        try:
            import certifi
            return ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            pass
    return True


def _resolve_verify(
    *,
    insecure: Optional[bool] = None,
    ca_bundle: Optional[str] = None,
    auth_state: Optional[Dict[str, Any]] = None,
) -> bool | ssl.SSLContext:
    tls_state = auth_state.get("tls") if isinstance(auth_state, dict) else {}
    tls_state = tls_state if isinstance(tls_state, dict) else {}

    effective_insecure = (
        is_truthy_value(insecure, default=False) if insecure is not None
        else is_truthy_value(tls_state.get("insecure", False), default=False)
    )
    effective_ca = (
        ca_bundle
        or tls_state.get("ca_bundle")
        or os.getenv("HERMES_CA_BUNDLE")
        or os.getenv("SSL_CERT_FILE")
        or os.getenv("REQUESTS_CA_BUNDLE")
    )

    if effective_insecure:
        return False
    if effective_ca:
        ca_path = str(effective_ca)
        if not os.path.isfile(ca_path):
            logger.warning(
                "CA bundle path does not exist: %s — falling back to default certificates",
                ca_path,
            )
            return _default_verify()
        return ssl.create_default_context(cafile=ca_path)
    return _default_verify()


# =============================================================================
# OAuth Device Code Flow — generic, parameterized by provider
# =============================================================================

def _request_device_code(
    client: httpx.Client,
    portal_base_url: str,
    client_id: str,
    scope: Optional[str],
) -> Dict[str, Any]:
    """POST to the device code endpoint. Returns device_code, user_code, etc."""
    response = client.post(
        f"{portal_base_url}/api/oauth/device/code",
        data={
            "client_id": client_id,
            **({"scope": scope} if scope else {}),
        },
    )
    response.raise_for_status()
    data = response.json()

    required_fields = [
        "device_code", "user_code", "verification_uri",
        "verification_uri_complete", "expires_in", "interval",
    ]
    missing = [f for f in required_fields if f not in data]
    if missing:
        raise ValueError(f"Device code response missing fields: {', '.join(missing)}")
    return data


def _nous_device_auth_timeout_message(portal_base_url: str) -> str:
    """Actionable timeout text for Nous device-code login failures.

    A bare "timed out" gives the user nothing to act on; the usual cause is Portal sign-in failing
    in the opened browser tab, so point at the Portal login page and the retry command.
    """
    portal = (portal_base_url or DEFAULT_NOUS_PORTAL_URL).rstrip("/")
    return (
        "Timed out waiting for device authorization.\n"
        "  Portal sign-in is required before the device code can be approved.\n"
        "  If the browser showed a CAPTCHA / 'You did not pass CAPTCHA' error,\n"
        "  finish signing in at the Portal in a normal browser tab, then retry:\n"
        "    hermes portal\n"
        f"  Portal login: {portal}/login"
    )


def _print_device_code_instructions(
    verification_url: str,
    user_code: str,
    *,
    open_browser: bool,
    failure_dash: str = "--",
    swallow_open_errors: bool = False,
) -> None:
    """Print the shared "To continue" device-code block and optionally open the browser.

    Callers decide *whether* to open (remote-session / graphical-browser gating differs per
    provider); the wording of the fallback hint is parameterized so each provider keeps its
    historical dash style.
    """
    print()
    print("To continue:")
    print(f"  1. Open: {verification_url}")
    print(f"  2. If prompted, enter code: {user_code}")
    if not open_browser:
        return
    if swallow_open_errors:
        try:
            opened = webbrowser.open(verification_url)
        except Exception:
            opened = False
    else:
        opened = webbrowser.open(verification_url)
    if opened:
        print("  (Opened browser for verification)")
    else:
        print(f"  Could not open browser automatically {failure_dash} use the URL above.")


def _poll_device_token_generic(
    post: Callable[[], "httpx.Response"],
    *,
    expires_in: int,
    poll_interval: int,
    validate_success: Callable[[Dict[str, Any]], None],
    on_non_json_error: Callable[["httpx.Response"], Exception],
    on_error: Callable[["httpx.Response", Dict[str, Any]], Exception],
    on_timeout: Callable[[], Exception],
) -> Dict[str, Any]:
    """RFC 8628 device-code polling loop shared by the Nous and xAI flows.

    ``authorization_pending`` sleeps and retries; ``slow_down`` grows the interval by 1s (cap 30s).
    Every other error, a non-JSON error body, and the deadline are turned into provider-specific
    exceptions by the supplied factories so each caller keeps its exact error contract.
    """
    deadline = time.monotonic() + max(1, expires_in)
    current_interval = poll_interval
    while time.monotonic() < deadline:
        response = post()
        if response.status_code == 200:
            payload = response.json()
            validate_success(payload)
            return payload
        try:
            error_payload = response.json()
        except Exception:
            response.raise_for_status()
            raise on_non_json_error(response)
        error_code = str(error_payload.get("error") or "")
        if error_code == "authorization_pending":
            time.sleep(current_interval)
            continue
        if error_code == "slow_down":
            current_interval = min(current_interval + 1, 30)
            time.sleep(current_interval)
            continue
        raise on_error(response, error_payload)
    raise on_timeout()


def _poll_for_token(
    client: httpx.Client,
    portal_base_url: str,
    client_id: str,
    device_code: str,
    expires_in: int,
    poll_interval: int,
) -> Dict[str, Any]:
    """Poll the Nous token endpoint until the user approves or the code expires."""
    def _validate(payload: Dict[str, Any]) -> None:
        if "access_token" not in payload:
            raise ValueError("Token response did not include access_token")

    def _error(_response, error_payload) -> Exception:
        error_code = error_payload.get("error", "")
        description = error_payload.get("error_description") or "Unknown authentication error"
        return RuntimeError(f"{error_code}: {description}")

    return _poll_device_token_generic(
        lambda: client.post(
            f"{portal_base_url}/api/oauth/token",
            data={
                "grant_type": DEVICE_CODE_GRANT_TYPE,
                "client_id": client_id,
                "device_code": device_code,
            },
        ),
        expires_in=expires_in,
        poll_interval=max(1, min(poll_interval, DEVICE_AUTH_POLL_INTERVAL_CAP_SECONDS)),
        validate_success=_validate,
        on_non_json_error=lambda _r: RuntimeError("Token endpoint returned a non-JSON error response"),
        on_error=_error,
        # Enriched at the SOURCE so every caller inherits the guidance:
        # the CLI login (_nous_device_code_login) and the dashboard/desktop
        # poller (web_server._nous_poller, which surfaces str(e) to the UI).
        on_timeout=lambda: TimeoutError(_nous_device_auth_timeout_message(portal_base_url)),
    )


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

NOUS_SHARED_STORE_FILENAME = "nous_auth.json"
_nous_shared_lock_holder = threading.local()


def _nous_shared_auth_dir() -> Path:
    """Resolve the directory that holds the shared Nous token store.

    Honors ``HERMES_SHARED_AUTH_DIR`` so tests can redirect it. Defaults to
    ``<hermes-root>/shared/`` (``~/.hermes/shared/`` on POSIX, ``%LOCALAPPDATA%\\hermes\\shared\\``
    on Windows), outside any named profile so all profiles under one root share the store.
    """
    override = os.getenv("HERMES_SHARED_AUTH_DIR", "").strip()
    if override:
        return Path(override).expanduser()
    from hermes_constants import get_default_hermes_root
    return get_default_hermes_root() / "shared"


def _nous_shared_store_path() -> Path:
    path = _nous_shared_auth_dir() / NOUS_SHARED_STORE_FILENAME
    # Seat belt: if pytest is running and this resolves to a path under the
    # real user's Hermes root, refuse rather than silently corrupt cross-profile
    # state. Tests must set HERMES_SHARED_AUTH_DIR to a tmp_path (conftest
    # does not do this automatically — mirror the _auth_file_path() guard
    # so forgetting to set it fails loudly instead of writing to the real
    # shared store).
    if os.environ.get("PYTEST_CURRENT_TEST"):
        from hermes_constants import get_default_hermes_root
        real_home_shared = (
            get_default_hermes_root() / "shared" / NOUS_SHARED_STORE_FILENAME
        ).resolve(strict=False)
        try:
            resolved = path.resolve(strict=False)
        except Exception:
            resolved = path
        if resolved == real_home_shared:
            raise RuntimeError(
                f"Refusing to touch real user shared Nous auth store during test run: "
                f"{path}. Set HERMES_SHARED_AUTH_DIR to a tmp_path in your test fixture."
            )
    return path


@contextmanager
def _nous_shared_store_lock(timeout_seconds: float = AUTH_LOCK_TIMEOUT_SECONDS):
    """Cross-profile lock for the shared Nous OAuth store.

    Lock ordering invariant: if both this and ``_auth_store_lock`` need to be held, acquire
    ``_auth_store_lock`` FIRST. All runtime refresh paths follow this order.
    """
    try:
        lock_path = _nous_shared_store_path().with_suffix(".lock")
    except RuntimeError:
        # No HERMES_HOME yet (pre-setup): fall through without locking.
        yield
        return

    with _file_lock(
        lock_path,
        _nous_shared_lock_holder,
        timeout_seconds,
        "Timed out waiting for shared Nous auth lock",
    ):
        yield


# OAuth fields mirrored between a profile's Nous state and the shared cross-profile store.
_NOUS_SHARED_STATE_KEYS = (
    "access_token",
    "refresh_token",
    "token_type",
    "scope",
    "client_id",
    "portal_base_url",
    "inference_base_url",
    "obtained_at",
    "expires_at",
)


def _merge_shared_nous_oauth_state(state: Dict[str, Any]) -> bool:
    """Copy fresher shared OAuth tokens into a profile-local Nous state."""
    shared = _read_shared_nous_state()
    if not shared:
        return False

    shared_refresh = shared.get("refresh_token")
    if not _nonempty_str(shared_refresh):
        return False

    local_refresh = state.get("refresh_token")
    shared_access_exp = _parse_iso_timestamp(shared.get("expires_at")) or 0.0
    local_access_exp = _parse_iso_timestamp(state.get("expires_at")) or 0.0
    refresh_changed = shared_refresh.strip() != str(local_refresh or "").strip()
    fresher_access = shared_access_exp > local_access_exp
    if not refresh_changed and not fresher_access:
        return False

    for key in _NOUS_SHARED_STATE_KEYS:
        value = shared.get(key)
        if value not in {None, ""}:
            state[key] = value
    return True


def _nous_shared_shape(src: Dict[str, Any]) -> Dict[str, Any]:
    """The defaulted OAuth core (tokens + routing + expiry) shared across profiles."""
    return {
        "access_token": src.get("access_token"),
        "refresh_token": src.get("refresh_token"),
        "token_type": src.get("token_type") or "Bearer",
        "scope": src.get("scope") or DEFAULT_NOUS_SCOPE,
        "client_id": src.get("client_id") or DEFAULT_NOUS_CLIENT_ID,
        "portal_base_url": src.get("portal_base_url") or DEFAULT_NOUS_PORTAL_URL,
        "inference_base_url": src.get("inference_base_url") or DEFAULT_NOUS_INFERENCE_URL,
        "obtained_at": src.get("obtained_at"),
        "expires_at": src.get("expires_at"),
    }


def _write_shared_nous_state(state: Dict[str, Any]) -> None:
    """Persist a minimal copy of the Nous OAuth state to the shared store.

    Best-effort: any failure is swallowed after logging. The shared store is a convenience layer;
    the per-profile auth.json remains the source of truth.
    """
    refresh_token = state.get("refresh_token")
    access_token = state.get("access_token")
    # No refresh_token = nothing worth sharing across profiles
    if not (_nonempty_str(refresh_token) and _nonempty_str(access_token)):
        return

    shared = {
        "_schema": 1,
        **_nous_shared_shape(state),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        with _nous_shared_store_lock():
            path = _nous_shared_store_path()
            _write_private_file_atomic(
                path, json.dumps(shared, indent=2, sort_keys=True), replace=os.replace,
            )
        _oauth_trace(
            "nous_shared_store_written",
            path=str(path),
            refresh_token_fp=_token_fingerprint(refresh_token),
        )
    except Exception as exc:
        logger.debug("Failed to write shared Nous auth store: %s", exc)


def _read_shared_nous_state() -> Optional[Dict[str, Any]]:
    """Return the shared Nous OAuth state if present and well-formed.

    Returns ``None`` when the file is missing, unreadable, malformed, or lacks required fields;
    callers treat that as "no shared credentials, fall through to device-code".
    """
    try:
        path = _nous_shared_store_path()
    except RuntimeError:
        # Test seat belt tripped — treat as missing
        return None
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        logger.debug("Shared Nous auth store at %s is unreadable: %s", path, exc)
        return None
    if not isinstance(payload, dict):
        return None
    if not (_nonempty_str(payload.get("refresh_token")) and _nonempty_str(payload.get("access_token"))):
        return None
    return payload


def _clear_shared_nous_state(reason: str) -> None:
    """Remove the shared Nous OAuth store after a terminal token failure."""
    try:
        with _nous_shared_store_lock():
            path = _nous_shared_store_path()
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        _oauth_trace("nous_shared_store_cleared", reason=reason)
    except Exception as exc:
        logger.debug("Failed to clear shared Nous auth store: %s", exc)


# Error codes per provider for which retrying the SAME refresh token cannot succeed.
# ``*_refresh_failed`` covers HTTP 400/401/403 from the token endpoint (invalid_grant, token
# revoked, refresh_token_reused); ``*_auth_missing_refresh_token`` means the pool entry has no
# refresh token at all. All must also carry ``relogin_required=True``; transient failures
# (429, 5xx) do not.
_OAUTH_GRANT_DEAD_CODES = frozenset({"invalid_grant", "invalid_token", "refresh_token_reused"})
_TERMINAL_REFRESH_ERROR_CODES: Dict[str, FrozenSet[str]] = {
    "nous": _OAUTH_GRANT_DEAD_CODES,
    "xai-oauth": frozenset({"xai_refresh_failed", "xai_auth_missing_refresh_token"}),
    "openai-codex": _OAUTH_GRANT_DEAD_CODES | {"codex_refresh_failed", "codex_auth_missing_refresh_token"},
}


def _is_terminal_refresh_error(exc: Exception, provider: str) -> bool:
    """True when retrying the same *provider* refresh token cannot succeed."""
    return (
        isinstance(exc, AuthError)
        and exc.provider == provider
        and exc.code in _TERMINAL_REFRESH_ERROR_CODES[provider]
        and bool(exc.relogin_required)
    )


def _is_terminal_nous_refresh_error(exc: Exception) -> bool:
    return _is_terminal_refresh_error(exc, "nous")


def _is_terminal_xai_oauth_refresh_error(exc: Exception) -> bool:
    return _is_terminal_refresh_error(exc, "xai-oauth")


def _is_terminal_codex_oauth_refresh_error(exc: Exception) -> bool:
    return _is_terminal_refresh_error(exc, "openai-codex")


def _quarantine_nous_oauth_state(
    state: Dict[str, Any],
    error: AuthError,
    *,
    reason: str,
) -> None:
    """Keep routing metadata but remove dead OAuth material so it is not replayed."""
    # Forensic logging BEFORE we clear the token material. A hosted agent
    # can take a terminal invalid_grant and get quarantined here silently: the
    # only downstream signal is a "No access token found" WARNING once the pool
    # is already empty, which is too late to root-cause. A managed log drain may
    # be WARNING-only, so this MUST be logger.warning (INFO never reaches it).
    #
    # Redaction safety: emit ONLY the 12-char SHA-256 hex prefix of the refresh
    # token (correlates to NAS's refreshTokenHash without leaking the secret) plus
    # sizes/booleans. NEVER pass a raw token/agent_key into the log call — Hermes
    # has a known bug class where credential-shaped literals get corrupted in logs.
    forensic: Dict[str, Any] = {
        "reason": reason,
        "error_code": error.code,
        # No session_id field exists on Nous state; provenance is client_id +
        # agent_key_id (both non-secret routing identifiers).
        "client_id": state.get("client_id"),
        "agent_key_id": state.get("agent_key_id"),
        "refresh_token_fp": _token_fingerprint(state.get("refresh_token")),
    }

    # On-disk integrity of the auth store at the moment of quarantine.
    try:
        auth_path = _auth_file_path()
        forensic["auth_json_path"] = str(auth_path)
        try:
            st = os.stat(auth_path)
            forensic["auth_json_size"] = st.st_size
            forensic["auth_json_mtime"] = st.st_mtime
            forensic["auth_json_exists"] = True
        except FileNotFoundError:
            forensic["auth_json_exists"] = False
    except Exception as exc:  # pragma: no cover - never let logging break quarantine
        forensic["auth_json_stat_error"] = repr(exc)

    # Was the token already past its own expiry when it was rejected?
    already_expired: Optional[bool] = None
    expires_at_raw = state.get("expires_at")
    if isinstance(expires_at_raw, str) and expires_at_raw:
        try:
            parsed = datetime.fromisoformat(expires_at_raw)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            already_expired = parsed < datetime.now(timezone.utc)
        except ValueError:
            already_expired = None
    forensic["token_already_expired"] = already_expired

    logger.warning(
        "Nous OAuth state quarantined (terminal auth death): %s",
        json.dumps(forensic, sort_keys=True, ensure_ascii=False),
    )

    for key in (*_FLAT_OAUTH_TOKEN_KEYS, *_NOUS_EMPTY_AGENT_KEY_FIELDS):
        state.pop(key, None)
    state["last_auth_error"] = _last_auth_error_marker("nous", error, reason=reason)
    _clear_shared_nous_state(reason)
    invalidate_nous_auth_status_cache()


def _quarantine_nous_pool_entries(
    auth_store: Dict[str, Any],
    error: AuthError,
    *,
    reason: str,
) -> bool:
    """Remove singleton-seeded Nous pool entries that contain dead OAuth state."""
    entries = _pool_entries(auth_store, "nous")
    if entries is None:
        return False
    pool = auth_store["credential_pool"]

    retained = []
    removed = False
    singleton_sources = {NOUS_DEVICE_CODE_SOURCE, f"manual:{NOUS_DEVICE_CODE_SOURCE}"}
    for entry in entries:
        if isinstance(entry, dict) and entry.get("source") in singleton_sources:
            removed = True
            continue
        retained.append(entry)

    if removed:
        pool["nous"] = retained
        _oauth_trace(
            "nous_pool_device_code_quarantined",
            reason=reason,
            error_code=error.code,
        )
    return removed


def _try_import_shared_nous_state(
    *,
    timeout_seconds: float = 15.0,
) -> Optional[Dict[str, Any]]:
    """Attempt to rehydrate Nous OAuth state from the shared store.

    Runs a forced refresh with the stored refresh_token to mint a fresh inference JWT scoped to
    this profile and returns the auth_state dict ready for ``persist_nous_credentials()``.
    Returns ``None`` on any failure (expired token, portal unreachable) so the caller falls
    through to the normal device-code flow.
    """
    try:
        with _nous_shared_store_lock(timeout_seconds=max(timeout_seconds + 5.0, AUTH_LOCK_TIMEOUT_SECONDS)):
            shared = _read_shared_nous_state()
            if not shared:
                return None

            # Build a full state dict so refresh_nous_oauth_from_state has every
            # field it needs. force_refresh=True gets us a fresh access_token
            # for this profile.
            state: Dict[str, Any] = {
                **_nous_shared_shape(shared),
                "agent_key": None,
                "agent_key_expires_at": None,
                "tls": {"insecure": False, "ca_bundle": None},
            }

            def _persist_shared_refresh(updated_state: Dict[str, Any], _reason: str) -> None:
                _write_shared_nous_state(updated_state)

            refreshed = refresh_nous_oauth_from_state(
                state,
                timeout_seconds=timeout_seconds,
                force_refresh=True,
                on_state_update=_persist_shared_refresh,
            )
            _write_shared_nous_state(refreshed)
    except AuthError as exc:
        _oauth_trace(
            "nous_shared_import_failed",
            error_type=type(exc).__name__,
            error_code=getattr(exc, "code", None),
        )
        if _is_terminal_nous_refresh_error(exc):
            _clear_shared_nous_state("shared_import_terminal_refresh_failure")
        logger.debug("Shared Nous import failed: %s", exc)
        return None
    except Exception as exc:
        _oauth_trace(
            "nous_shared_import_failed",
            error_type=type(exc).__name__,
        )
        logger.debug("Shared Nous import failed: %s", exc)
        return None

    return refreshed


def _refresh_access_token(
    *,
    client: httpx.Client,
    portal_base_url: str,
    client_id: str,
    refresh_token: str,
) -> Dict[str, Any]:
    response = client.post(
        f"{portal_base_url}/api/oauth/token",
        headers={"x-nous-refresh-token": refresh_token},
        data={
            "grant_type": "refresh_token",
            "client_id": client_id,
        },
    )

    if response.status_code == 200:
        payload = response.json()
        if "access_token" not in payload:
            raise _nous_err("Refresh response missing access_token", "invalid_token", relogin=True)
        return payload

    try:
        error_payload = response.json()
    except Exception as exc:
        raise _nous_err("Refresh token exchange failed", relogin=True) from exc

    code = str(error_payload.get("error", "invalid_grant"))
    description = str(error_payload.get("error_description") or "Refresh token exchange failed")
    relogin = code in {"invalid_grant", "invalid_token", "refresh_token_reused"}

    # Detect the OAuth 2.1 "refresh token reuse" signal from the Nous portal
    # server and surface an actionable message.  This fires when an external
    # process (health-check script, monitoring tool, custom self-heal hook)
    # called POST /api/oauth/token with Hermes's refresh_token without
    # persisting the rotated token back to auth.json — the server then
    # retires the original RT, Hermes's next refresh uses it, and the whole
    # session chain gets revoked as a token-theft signal (#15099).
    lowered = description.lower()
    if code == "refresh_token_reused" or "reuse" in lowered or "reuse detected" in lowered:
        description = (
            "Nous Portal detected refresh-token reuse and revoked this session.\n"
            "This usually means an external process (monitoring script, "
            "custom self-heal hook, or another Hermes install sharing "
            "~/.hermes/auth.json) called POST /api/oauth/token with Hermes's "
            "refresh token without persisting the rotated token back.\n"
            "Nous refresh tokens are single-use — only Hermes may call the "
            "refresh endpoint. For health checks, use `hermes auth status` "
            "instead.\n"
            "Re-authenticate with: hermes auth add nous"
        )
        relogin = True

    raise _nous_err(description, code, relogin=relogin)


def _refresh_nous_or_quarantine(
    *,
    client: httpx.Client,
    auth_store: Dict[str, Any],
    state: Dict[str, Any],
    portal_base_url: str,
    client_id: str,
    refresh_token: str,
    reason: str,
    persist: Callable[[], None],
) -> Dict[str, Any]:
    """Redeem the Nous refresh token; on a terminal failure quarantine state + pool, persist, re-raise."""
    try:
        return _refresh_access_token(
            client=client,
            portal_base_url=portal_base_url,
            client_id=client_id,
            refresh_token=refresh_token,
        )
    except AuthError as exc:
        if _is_terminal_nous_refresh_error(exc):
            _quarantine_nous_oauth_state(state, exc, reason=reason)
            _quarantine_nous_pool_entries(auth_store, exc, reason=reason)
            persist()
        raise


def _apply_nous_refreshed_tokens(
    state: Dict[str, Any],
    refreshed: Dict[str, Any],
    refresh_token: str,
    *,
    inference_base_url: Optional[str] = None,
) -> None:
    """Write a successful Nous token-refresh payload into *state* (tokens + expiry fields).

    *inference_base_url*, when given, is the healed network-provenance URL to persist alongside
    the rotated tokens (key order in auth.json is preserved from the original login shape).
    """
    now = datetime.now(timezone.utc)
    access_ttl = _coerce_ttl_seconds(refreshed.get("expires_in"))
    state["access_token"] = refreshed["access_token"]
    state["refresh_token"] = refreshed.get("refresh_token") or refresh_token
    state["token_type"] = refreshed.get("token_type") or state.get("token_type") or "Bearer"
    state["scope"] = refreshed.get("scope") or state.get("scope")
    if inference_base_url is not None:
        state["inference_base_url"] = inference_base_url
    state["obtained_at"] = now.isoformat()
    state["expires_in"] = access_ttl
    state["expires_at"] = _iso_after(now, access_ttl)


def _healed_nous_inference_url(refreshed: Dict[str, Any]) -> str:
    """Validated network-provenance inference URL from a refresh payload, healed to the default.

    When the Portal-returned URL is rejected by the allowlist (returns None), reset to the
    production default instead of leaving a previously-persisted bad host (e.g. a stale staging
    URL) in place — otherwise a poisoned auth.json keeps re-validating to None on every refresh
    and silently re-uses the dead endpoint.
    """
    return (
        _validate_nous_inference_url_from_network(refreshed.get("inference_base_url"))
        or DEFAULT_NOUS_INFERENCE_URL
    )


def fetch_nous_models(
    *,
    inference_base_url: str,
    api_key: str,
    timeout_seconds: float = 15.0,
    verify: bool | str = True,
) -> List[str]:
    """Fetch available model IDs from the Nous inference API."""
    timeout = httpx.Timeout(timeout_seconds)
    with httpx.Client(timeout=timeout, headers={"Accept": "application/json"}, verify=verify) as client:
        response = client.get(
            f"{inference_base_url.rstrip('/')}/models",
            headers={"Authorization": f"Bearer {api_key}"},
        )

    if response.status_code != 200:
        description = f"/models request failed with status {response.status_code}"
        try:
            err = response.json()
            description = str(err.get("error_description") or err.get("error") or description)
        except Exception as e:
            logger.debug("Could not parse error response JSON: %s", e)
        raise _nous_err(description, "models_fetch_failed")

    payload = response.json()
    data = payload.get("data")
    if not isinstance(data, list):
        return []

    model_ids: List[str] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        model_id = item.get("id")
        if _nonempty_str(model_id):
            mid = model_id.strip()
            # Skip Hermes models — they're not reliable for agentic tool-calling
            if "hermes" in mid.lower():
                continue
            model_ids.append(mid)

    # Sort: prefer opus > pro > haiku/flash > sonnet (sonnet is cheap/fast,
    # users who want the best model should see opus first).
    def _model_priority(mid: str) -> tuple:
        low = mid.lower()
        if "opus" in low:
            return (0, mid)
        if "pro" in low and "sonnet" not in low:
            return (1, mid)
        if "sonnet" in low:
            return (3, mid)
        return (2, mid)

    model_ids.sort(key=_model_priority)
    return list(dict.fromkeys(model_ids))


def _agent_key_is_usable(state: Dict[str, Any], min_ttl_seconds: int) -> bool:
    key = state.get("agent_key")
    if not _nonempty_str(key):
        return False
    return _nous_invoke_jwt_is_usable(
        key,
        scope=state.get("scope"),
        expires_at=state.get("agent_key_expires_at"),
        min_ttl_seconds=max(0, int(min_ttl_seconds)),
    )


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


def refresh_nous_oauth_pure(
    access_token: str,
    refresh_token: str,
    client_id: str,
    portal_base_url: str,
    inference_base_url: str,
    *,
    token_type: str = "Bearer",
    scope: str = DEFAULT_NOUS_SCOPE,
    obtained_at: Optional[str] = None,
    expires_at: Optional[str] = None,
    agent_key: Optional[str] = None,
    agent_key_expires_at: Optional[str] = None,
    timeout_seconds: float = 15.0,
    insecure: Optional[bool] = None,
    ca_bundle: Optional[str] = None,
    force_refresh: bool = False,
    on_state_update: Optional[Callable[[Dict[str, Any], str], None]] = None,
) -> Dict[str, Any]:
    """Refresh Nous OAuth state without mutating auth.json directly.

    ``on_state_update`` is called after a successful access-token refresh. Callers that own
    persistent state can use it to save the newly rotated refresh token before later validation can
    fail.
    """
    state: Dict[str, Any] = {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "client_id": client_id or DEFAULT_NOUS_CLIENT_ID,
        "portal_base_url": (portal_base_url or DEFAULT_NOUS_PORTAL_URL).rstrip("/"),
        "inference_base_url": (inference_base_url or DEFAULT_NOUS_INFERENCE_URL).rstrip("/"),
        "token_type": token_type or "Bearer",
        "scope": scope or DEFAULT_NOUS_SCOPE,
        "obtained_at": obtained_at,
        "expires_at": expires_at,
        "agent_key": agent_key,
        "agent_key_expires_at": agent_key_expires_at,
        "tls": {
            "insecure": bool(insecure),
            "ca_bundle": ca_bundle,
        },
    }
    verify = _resolve_verify(insecure=insecure, ca_bundle=ca_bundle, auth_state=state)
    timeout = httpx.Timeout(timeout_seconds if timeout_seconds else 15.0)

    with httpx.Client(timeout=timeout, headers={"Accept": "application/json"}, verify=verify) as client:
        current_invoke_jwt_status = _nous_invoke_jwt_status(
            state.get("access_token"),
            scope=state.get("scope"),
            expires_at=state.get("expires_at"),
        )
        if force_refresh or current_invoke_jwt_status is not None:
            refresh_token_value = state.get("refresh_token")
            if not isinstance(refresh_token_value, str) or not refresh_token_value:
                if current_invoke_jwt_status is not None:
                    raise _nous_err(
                        "Nous Portal access token is not a usable inference JWT "
                        f"({current_invoke_jwt_status}) and no refresh token is available. "
                        "Re-authenticate with: hermes auth add nous",
                        current_invoke_jwt_status, relogin=True,
                    )
                raise _nous_err(
                    "No refresh token is available for Nous Portal.",
                    relogin=True,
                )
            refreshed = _refresh_access_token(
                client=client,
                portal_base_url=state["portal_base_url"],
                client_id=state["client_id"],
                refresh_token=refresh_token_value,
            )
            _apply_nous_refreshed_tokens(
                state, refreshed, refresh_token_value,
                inference_base_url=_healed_nous_inference_url(refreshed),
            )
            if on_state_update is not None:
                on_state_update(dict(state), "post_refresh_access_token")

        _assert_nous_inference_jwt_usable(state)
        _select_nous_invoke_jwt(state)

    return state


def refresh_nous_oauth_from_state(
    state: Dict[str, Any],
    *,
    timeout_seconds: float = 15.0,
    force_refresh: bool = False,
    on_state_update: Optional[Callable[[Dict[str, Any], str], None]] = None,
) -> Dict[str, Any]:
    """Refresh Nous OAuth from a state dict. Thin wrapper around refresh_nous_oauth_pure."""
    tls = state.get("tls") or {}
    return refresh_nous_oauth_pure(
        state.get("access_token", ""),
        state.get("refresh_token", ""),
        state.get("client_id", "hermes-cli"),
        state.get("portal_base_url", DEFAULT_NOUS_PORTAL_URL),
        state.get("inference_base_url", DEFAULT_NOUS_INFERENCE_URL),
        token_type=state.get("token_type", "Bearer"),
        scope=state.get("scope", DEFAULT_NOUS_SCOPE),
        obtained_at=state.get("obtained_at"),
        expires_at=state.get("expires_at"),
        agent_key=state.get("agent_key"),
        agent_key_expires_at=state.get("agent_key_expires_at"),
        timeout_seconds=timeout_seconds,
        insecure=tls.get("insecure"),
        ca_bundle=tls.get("ca_bundle"),
        force_refresh=force_refresh,
        on_state_update=on_state_update,
    )


def persist_nous_credentials(
    creds: Dict[str, Any],
    *,
    label: Optional[str] = None,
):
    """Persist Nous OAuth credentials as the singleton provider state

    Nous credentials are read from two places: ``providers.nous`` (401 recovery, pool seeding) and
    ``credential_pool.nous`` (runtime ``pool.select()``). Writing only a pool entry left the
    singleton empty and made expiry recovery fail silently, so this writes the singleton and then
    ``load_pool("nous")`` upserts the canonical ``device_code`` entry in place (never duplicates).
    ``label`` is embedded in the singleton so re-seeding keeps the user's display name.
    """
    from agent.credential_pool import load_pool

    state = dict(creds)
    if label and str(label).strip():
        state["label"] = str(label).strip()

    _save_active_provider_state("nous", state)

    # Mirror to the shared store so a new profile can one-tap import
    # these credentials via `hermes auth add nous --type oauth`. Best-
    # effort: any I/O failure is logged and swallowed (the per-profile
    # auth.json is still the source of truth).
    _write_shared_nous_state(state)

    pool = load_pool("nous")
    return next(
        (e for e in pool.entries() if e.source == NOUS_DEVICE_CODE_SOURCE),
        None,
    )


def _sync_nous_pool_from_auth_store() -> None:
    """Best-effort pool reseed after providers.nous changes; never fail login."""
    try:
        from agent.credential_pool import load_pool

        load_pool("nous")
    except Exception as exc:
        logger.debug("Failed to sync Nous credential pool from auth store: %s", exc)


class _NousStatePersister:
    """Writes Nous provider state to its source store, skipping no-op writes.

    Writes where only derived TTL countdowns changed are skipped; this keeps the mtime-keyed Nous
    auth-status cache warm during read paths. Every real write is mirrored to the shared store so
    sibling profiles don't hold stale refresh_tokens after rotation (best-effort — failures are
    logged and swallowed inside ``_write_shared_nous_state``).
    """

    def __init__(
        self,
        auth_store: Dict[str, Any],
        state: Dict[str, Any],
        state_source_path: Optional[Path],
        sequence_id: str,
    ) -> None:
        self._auth_store = auth_store
        self._state = state
        self._source_path = state_source_path
        self._sequence_id = sequence_id
        self._persisted_state = dict(state)
        self.persisted_any = False

    def persist(self, reason: str) -> None:
        state = self._state
        if (
            _nous_effective_provider_state(state)
            == _nous_effective_provider_state(self._persisted_state)
        ):
            _oauth_trace(
                "nous_state_persist_skipped",
                sequence_id=self._sequence_id,
                reason=reason,
            )
            return
        try:
            _save_provider_state_to_source(self._auth_store, "nous", state, self._source_path)
        except Exception as exc:
            _oauth_trace(
                "nous_state_persist_failed",
                sequence_id=self._sequence_id,
                reason=reason,
                error_type=type(exc).__name__,
            )
            raise
        _oauth_trace(
            "nous_state_persisted",
            sequence_id=self._sequence_id,
            reason=reason,
            refresh_token_fp=_token_fingerprint(state.get("refresh_token")),
            access_token_fp=_token_fingerprint(state.get("access_token")),
        )
        self._persisted_state = dict(state)
        self.persisted_any = True
        _write_shared_nous_state(state)


def _nous_effective_routing(state: Dict[str, Any]) -> tuple[str, str, str, str]:
    """Resolve every routing value that shared OAuth state can replace.

    Returns ``(portal_url, stored_inference_url, effective_inference_url, client_id)``. The
    stored inference URL is re-validated network-provenance (persisted); the effective one layers
    the runtime-only ``NOUS_INFERENCE_BASE_URL`` override on top and must never be persisted.
    """
    portal_url = (
        _optional_base_url(state.get("portal_base_url"))
        or os.getenv("HERMES_PORTAL_BASE_URL")
        or os.getenv("NOUS_PORTAL_BASE_URL")
        or DEFAULT_NOUS_PORTAL_URL
    ).rstrip("/")

    # A persisted/stale portal_base_url is where the refresh token gets
    # POSTed on refresh — reject any host outside the allowlist so a
    # poisoned value can't exfiltrate the bearer, healing to the default.
    # Trusted operator env overrides bypass this network-value gate.
    env_portal_override = _nous_portal_env_override()
    if env_portal_override:
        portal_url = env_portal_override.rstrip("/")
    else:
        parsed_portal_url = urlparse(portal_url)
        portal_host = parsed_portal_url.hostname
        loopback_http = (
            parsed_portal_url.scheme == "http"
            and portal_host in {"localhost", "127.0.0.1"}
        )
        trusted_scheme = parsed_portal_url.scheme == "https" or loopback_http
        if (
            not portal_host
            or portal_host not in _NOUS_PORTAL_ALLOWED_HOSTS
            or not trusted_scheme
        ):
            logger.warning(
                "auth: ignoring invalid portal_base_url %r "
                "(host %r or scheme not allowed), using default",
                portal_url,
                portal_host,
            )
            portal_url = DEFAULT_NOUS_PORTAL_URL

    stored_inference_url = (
        _validate_nous_inference_url_from_network(
            _optional_base_url(state.get("inference_base_url"))
        )
        or DEFAULT_NOUS_INFERENCE_URL
    )
    return (
        portal_url,
        stored_inference_url,
        _nous_inference_env_override() or stored_inference_url,
        str(state.get("client_id") or DEFAULT_NOUS_CLIENT_ID),
    )


def resolve_nous_runtime_credentials(
    *,
    timeout_seconds: float = 15.0,
    insecure: Optional[bool] = None,
    ca_bundle: Optional[str] = None,
    force_refresh: bool = False,
    stale_access_token: Optional[str] = None,
) -> Dict[str, Any]:
    """Resolve Nous inference credentials for runtime use.

    Ensures access_token is a valid inference-scoped JWT, refreshing it when
    needed. Concurrent processes coordinate through the auth store file lock.

    ``stale_access_token`` is the bearer that just failed upstream (401). When
    set together with ``force_refresh``, the refresh POST is skipped if the
    store — re-read under the lock — already holds a *different*, usable
    token: another process won the rotation, so this caller adopts it instead
    of rotating the shared grant again (otherwise N concurrent processes at the
    same expiry issue N refreshes, each invalidating a sibling's fresh token).
    """
    sequence_id = uuid.uuid4().hex[:12]

    with _provider_state_transaction("nous") as (
        auth_store,
        state,
        state_source_path,
    ):

        if not state:
            raise _nous_err("Hermes is not logged into Nous Portal.", relogin=True)

        def _already_rotated_by_peer(token: Any) -> bool:
            return bool(
                force_refresh
                and stale_access_token
                and isinstance(token, str)
                and token
                and token != stale_access_token
                and _nous_invoke_jwt_status(
                    token,
                    scope=state.get("scope"),
                    expires_at=state.get("expires_at"),
                ) is None
            )

        persister = _NousStatePersister(auth_store, state, state_source_path, sequence_id)
        _persist_state = persister.persist

        (
            portal_base_url,
            stored_inference_base_url,
            inference_base_url,
            client_id,
        ) = _nous_effective_routing(state)

        verify = _resolve_verify(insecure=insecure, ca_bundle=ca_bundle, auth_state=state)
        timeout = httpx.Timeout(timeout_seconds if timeout_seconds else 15.0)
        _oauth_trace(
            "nous_runtime_credentials_start",
            sequence_id=sequence_id,
            refresh_token_fp=_token_fingerprint(state.get("refresh_token")),
        )

        with httpx.Client(timeout=timeout, headers={"Accept": "application/json"}, verify=verify) as client:
            access_token = state.get("access_token")
            refresh_token = state.get("refresh_token")

            if not isinstance(access_token, str) or not access_token:
                with _nous_shared_store_lock(
                    timeout_seconds=max(timeout_seconds + 5.0, AUTH_LOCK_TIMEOUT_SECONDS)
                ):
                    if _merge_shared_nous_oauth_state(state):
                        access_token = state.get("access_token")
                        refresh_token = state.get("refresh_token")
                        (
                            portal_base_url,
                            stored_inference_base_url,
                            inference_base_url,
                            client_id,
                        ) = _nous_effective_routing(state)
                        _persist_state("runtime_shared_merge_missing_access_token")

            if not isinstance(access_token, str) or not access_token:
                raise _nous_err(
                    "No access token found for Nous Portal login.",
                    relogin=True,
                )

            invoke_jwt_status = _nous_invoke_jwt_status(
                access_token,
                scope=state.get("scope"),
                expires_at=state.get("expires_at"),
            )
            # Under the store lock: if the bearer that failed upstream is no
            # longer the one on disk and the on-disk one is usable, a peer
            # already rotated — adopt, never re-POST the shared grant.
            if _already_rotated_by_peer(access_token):
                _oauth_trace(
                    "refresh_skipped_peer_rotated",
                    sequence_id=sequence_id,
                    access_token_fp=_token_fingerprint(access_token),
                )
                force_refresh = False
            if force_refresh or invoke_jwt_status is not None:
                with _nous_shared_store_lock(timeout_seconds=max(timeout_seconds + 5.0, AUTH_LOCK_TIMEOUT_SECONDS)):
                    if _merge_shared_nous_oauth_state(state):
                        access_token = state.get("access_token")
                        refresh_token = state.get("refresh_token")
                        (
                            portal_base_url,
                            stored_inference_base_url,
                            inference_base_url,
                            client_id,
                        ) = _nous_effective_routing(state)
                        invoke_jwt_status = _nous_invoke_jwt_status(
                            access_token,
                            scope=state.get("scope"),
                            expires_at=state.get("expires_at"),
                        )
                        _persist_state("post_shared_merge_access_unusable")
                        if _already_rotated_by_peer(access_token):
                            _oauth_trace(
                                "refresh_skipped_peer_rotated",
                                sequence_id=sequence_id,
                                access_token_fp=_token_fingerprint(access_token),
                            )
                            force_refresh = False

                    if force_refresh or invoke_jwt_status is not None:
                        if not isinstance(refresh_token, str) or not refresh_token:
                            reason = invoke_jwt_status or "force_refresh"
                            raise _nous_err(
                                "Nous Portal access token is not a usable inference JWT "
                                f"({reason}) and no refresh token is available. "
                                "Re-authenticate with: hermes auth add nous",
                                reason, relogin=True,
                            )

                        refresh_reason = "force_refresh" if force_refresh else (invoke_jwt_status or "access_unusable")
                        _oauth_trace(
                            "refresh_start",
                            sequence_id=sequence_id,
                            reason=refresh_reason,
                            refresh_token_fp=_token_fingerprint(refresh_token),
                        )
                        refreshed = _refresh_nous_or_quarantine(
                            client=client,
                            auth_store=auth_store,
                            state=state,
                            portal_base_url=portal_base_url,
                            client_id=client_id,
                            refresh_token=refresh_token,
                            reason="runtime_access_refresh_failure",
                            persist=lambda: _persist_state("terminal_runtime_access_refresh_failure"),
                        )
                        previous_refresh_token = refresh_token
                        # The validated, network-provenance URL is what gets persisted to
                        # auth.json (with the rotated tokens, so a later JWT validation
                        # failure cannot leave the stores on stale metadata). The
                        # NOUS_INFERENCE_BASE_URL env override is layered on for the
                        # client/return value only — it is never persisted.
                        stored_inference_base_url = _healed_nous_inference_url(refreshed)
                        inference_base_url = (
                            _nous_inference_env_override() or stored_inference_base_url
                        )
                        _apply_nous_refreshed_tokens(
                            state, refreshed, refresh_token,
                            inference_base_url=stored_inference_base_url,
                        )
                        access_token = state["access_token"]
                        refresh_token = state["refresh_token"]
                        _oauth_trace(
                            "refresh_success",
                            sequence_id=sequence_id,
                            reason=refresh_reason,
                            previous_refresh_token_fp=_token_fingerprint(previous_refresh_token),
                            new_refresh_token_fp=_token_fingerprint(refresh_token),
                        )
                        # Persist immediately so validation failures cannot drop rotated refresh tokens.
                        _persist_state("post_refresh_access_token")

            _assert_nous_inference_jwt_usable(
                state,
                access_token=access_token,
            )
            _select_nous_invoke_jwt(
                state,
                access_token=access_token,
                sequence_id=sequence_id,
            )

            # Persist routing and TLS metadata for non-interactive refresh.
            # Persist the validated, network-provenance URL — NEVER the env
            # override (which is a runtime-only overlay; persisting it would
            # leak a dev/staging host into auth.json and survive unsetting it).
            state["portal_base_url"] = portal_base_url
            state["inference_base_url"] = stored_inference_base_url
            state["client_id"] = client_id
            state["tls"] = _tls_state_from_verify(verify)

        _persist_state("resolve_nous_runtime_credentials_final")

    if persister.persisted_any:
        _sync_nous_pool_from_auth_store()

    api_key = state.get("agent_key")
    if not isinstance(api_key, str) or not api_key:
        raise _nous_err("Failed to resolve a Nous inference API key", "server_error")

    expires_at = state.get("agent_key_expires_at")
    expires_epoch = _parse_iso_timestamp(expires_at)
    expires_in = (
        max(0, int(expires_epoch - time.time()))
        if expires_epoch is not None
        else _coerce_ttl_seconds(state.get("agent_key_expires_in"))
    )

    return {
        "provider": "nous",
        "base_url": inference_base_url,
        "api_key": api_key,
        "key_id": state.get("agent_key_id"),
        "expires_at": expires_at,
        "expires_in": expires_in,
        "source": NOUS_AUTH_PATH_INVOKE_JWT,
        # Preserve the public semantic source label while exposing the concrete
        # store separately for diagnostics. Refresh persistence uses
        # state_source_path internally and must not overload this field.
        "auth_path": NOUS_AUTH_PATH_INVOKE_JWT,
        "state_path": str(state_source_path or _auth_file_path()),
    }


# =============================================================================
# Status helpers
# =============================================================================

def _empty_nous_auth_status() -> Dict[str, Any]:
    return {
        "logged_in": False,
        "portal_base_url": None,
        "inference_base_url": None,
        "access_expires_at": None,
        "agent_key_expires_at": None,
        "has_refresh_token": False,
        "inference_credential_present": False,
        "credential_source": None,
    }


def _snapshot_nous_pool_status() -> Dict[str, Any]:
    """Best-effort status from the credential pool.

    This is a fallback only. The auth-store provider state is the runtime source of truth because it
    is what ``resolve_nous_runtime_credentials()`` refreshes.
    """
    try:
        from agent.credential_pool import load_pool

        pool = load_pool("nous")
        if not pool or not pool.has_credentials():
            return _empty_nous_auth_status()

        entries = list(pool.entries())
        if not entries:
            return _empty_nous_auth_status()

        def _entry_sort_key(entry: Any) -> tuple[float, float, int]:
            agent_exp = _parse_iso_timestamp(getattr(entry, "agent_key_expires_at", None)) or 0.0
            access_exp = _parse_iso_timestamp(getattr(entry, "expires_at", None)) or 0.0
            priority = int(getattr(entry, "priority", 0) or 0)
            return (agent_exp, access_exp, -priority)

        entry = max(entries, key=_entry_sort_key)
        runtime_key = getattr(entry, "runtime_api_key", None)
        if not runtime_key:
            return _empty_nous_auth_status()
        access_token = getattr(entry, "access_token", None)
        auth_type = str(getattr(entry, "auth_type", "") or "").strip().lower()
        refresh_token = getattr(entry, "refresh_token", None)
        is_portal_oauth = bool(access_token) and (
            auth_type.startswith("oauth") or bool(refresh_token)
        )
        label = getattr(entry, "label", "unknown")
        portal_status_url = None
        if is_portal_oauth:
            portal_status_url = (
                getattr(entry, "portal_base_url", None)
                or DEFAULT_NOUS_PORTAL_URL
            )

        return {
            "logged_in": is_portal_oauth,
            "portal_base_url": portal_status_url,
            "inference_base_url": getattr(entry, "inference_base_url", None)
            or getattr(entry, "runtime_base_url", None)
            or getattr(entry, "base_url", None),
            "access_token": access_token if is_portal_oauth else None,
            "access_expires_at": getattr(entry, "expires_at", None),
            "agent_key_expires_at": getattr(entry, "agent_key_expires_at", None),
            "has_refresh_token": bool(refresh_token),
            "inference_credential_present": True,
            "credential_source": f"pool:{label}",
            "source": f"pool:{label}",
        }
    except Exception:
        return _empty_nous_auth_status()


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


def _nous_status_from_state(state: Dict[str, Any], *, logged_in: bool, source: str) -> Dict[str, Any]:
    """Auth-store-backed Nous status snapshot (shared by the live and refresh-free variants)."""
    access_token = state.get("access_token")
    return {
        "logged_in": logged_in,
        "portal_base_url": state.get("portal_base_url"),
        "inference_base_url": state.get("inference_base_url"),
        "access_expires_at": state.get("expires_at"),
        "agent_key_expires_at": state.get("agent_key_expires_at"),
        "has_refresh_token": bool(state.get("refresh_token")),
        "access_token": access_token,
        "inference_credential_present": bool(access_token or state.get("agent_key")),
        "credential_source": "auth_store",
        "source": source,
    }


def _compute_nous_auth_status() -> Dict[str, Any]:
    """Uncached implementation of get_nous_auth_status(). See that function."""
    state = get_provider_auth_state("nous")
    if state:
        base_status = _nous_status_from_state(
            state, logged_in=bool(state.get("access_token")), source="auth_store",
        )
        try:
            creds = resolve_nous_runtime_credentials()
            refreshed_state = get_provider_auth_state("nous") or state
            base_status.update(
                {
                    "logged_in": True,
                    "portal_base_url": refreshed_state.get("portal_base_url") or base_status.get("portal_base_url"),
                    "inference_base_url": creds.get("base_url")
                    or refreshed_state.get("inference_base_url")
                    or base_status.get("inference_base_url"),
                    "access_expires_at": refreshed_state.get("expires_at") or base_status.get("access_expires_at"),
                    "agent_key_expires_at": creds.get("expires_at")
                    or refreshed_state.get("agent_key_expires_at")
                    or base_status.get("agent_key_expires_at"),
                    "has_refresh_token": bool(refreshed_state.get("refresh_token")),
                    "inference_credential_present": True,
                    "credential_source": "auth_store",
                    "source": f"runtime:{creds.get('source', 'portal')}",
                    "key_id": creds.get("key_id"),
                }
            )
            return base_status
        except AuthError as exc:
            base_status.update({
                "logged_in": False,
                "error": str(exc),
                "relogin_required": bool(getattr(exc, "relogin_required", False)),
                "error_code": getattr(exc, "code", None),
            })
            return base_status

    return _snapshot_nous_pool_status()


def get_nous_auth_status_local() -> Dict[str, Any]:
    """Refresh-free Nous auth snapshot for read-only display surfaces.

    Unlike :func:`get_nous_auth_status`, this NEVER calls ``resolve_nous_runtime_credentials()`` and
    therefore never performs an OAuth refresh POST or consumes a single-use refresh token. It
    reports the persisted auth-store state, classifying the access token with a local invoke-JWT
    decode only.

    ``logged_in`` here means "a persisted login exists that the runtime can use or refresh": a
    currently-usable invoke JWT, or a refresh token that has not been terminally quarantined. It
    does not prove the refresh token is still accepted server-side — only a live resolve can do
    that.
    """
    try:
        state = get_provider_auth_state("nous")
    except Exception:
        state = None

    if not state:
        return _snapshot_nous_pool_status()

    access_token = state.get("access_token")
    jwt_reason = _nous_invoke_jwt_status(
        access_token,
        scope=state.get("scope"),
        expires_at=state.get("expires_at"),
    )
    last_err = state.get("last_auth_error")
    terminal = bool(
        isinstance(last_err, dict)
        and last_err.get("relogin_required")
        and not (access_token or state.get("refresh_token"))
    )
    logged_in = (jwt_reason is None) or (
        bool(state.get("refresh_token")) and not terminal
    )

    status = _nous_status_from_state(state, logged_in=logged_in, source="auth_store_local")
    if terminal and isinstance(last_err, dict):
        status["relogin_required"] = True
        status["error_code"] = last_err.get("code")
        status["error"] = last_err.get("message") or "re-login required"
    return status


# Enum values reported on the dashboard /api/status as ``nous_session_valid``.
# NAS's health sweep re-mints the bootstrap session ONLY on "terminal"; "valid"
# and "unknown" are no-ops. Keep this set small and stable — NAS parses it with
# a permissive schema, so new members are non-breaking but should stay rare.
NOUS_SESSION_VALID = "valid"
NOUS_SESSION_TERMINAL = "terminal"
NOUS_SESSION_UNKNOWN = "unknown"


def get_nous_session_validity() -> str:
    """Classify the Nous bootstrap session for the dashboard /api/status probe.

    Determinable with NO working token — it reads local auth-store state only, which is exactly the
    condition a dead hosted box is in. This function is called by the frequently-polled public
    ``/api/status`` endpoint, so it must never resolve credentials or perform an OAuth refresh.

    ANTI-FLAP CONTRACT: only a *terminal* failure maps to "terminal". A normal mid-rotation blip, a
    transient network error, or a merely-expiring token must NOT report "terminal" (that would
    trigger a spurious NAS re-mint on a healthy box).
    """
    # A persisted quarantine marker is the strongest, most stable terminal
    # signal: the refresh path writes `last_auth_error.relogin_required=True`
    # into the Nous provider state when it clears dead tokens (the exact path
    # that produced the incident's "No access token found"). Read it directly
    # so we report "terminal" even after the in-memory AuthError is long gone.
    try:
        state = get_provider_auth_state("nous")
    except Exception:
        return NOUS_SESSION_UNKNOWN

    if not state:
        return NOUS_SESSION_UNKNOWN

    last_err = state.get("last_auth_error")
    # Only terminal while there is no usable credential left. If a later
    # successful login repopulated tokens, the stale marker must not
    # keep reporting terminal.
    if (
        isinstance(last_err, dict)
        and last_err.get("relogin_required")
        and not (state.get("access_token") or state.get("refresh_token"))
    ):
        return NOUS_SESSION_TERMINAL

    if _nous_invoke_jwt_status(
        state.get("access_token"),
        scope=state.get("scope"),
        expires_at=state.get("expires_at"),
    ) is None:
        return NOUS_SESSION_VALID

    # Missing, malformed, expired, or merely expiring credentials are not proof
    # of a terminal session. Runtime inference/keepalive paths own refreshes;
    # the health endpoint remains side-effect free and reports indeterminate.
    return NOUS_SESSION_UNKNOWN


def _pool_first_oauth_status(
    provider_id: str,
    *,
    is_expiring: Callable[[str, int], bool],
    auth_mode: str,
    resolve: Callable[[], Dict[str, Any]],
    on_pool_miss: Optional[Callable[[], Optional[Dict[str, Any]]]] = None,
) -> Dict[str, Any]:
    """Status snapshot for a store-backed OAuth provider (Codex, xAI).

    Checks the credential pool first (where `hermes auth` / `hermes model` store device_code
    tokens), optionally consults *on_pool_miss* for a pool-derived degraded status, then falls
    back to the legacy provider state via *resolve*.
    """
    try:
        from agent.credential_pool import load_pool

        pool = load_pool(provider_id)
        if pool and pool.has_credentials():
            entry = pool.select()
            if entry is not None:
                api_key = (
                    getattr(entry, "runtime_api_key", None)
                    or getattr(entry, "access_token", "")
                )
                if api_key and not is_expiring(api_key, 0):
                    return {
                        "logged_in": True,
                        "auth_store": str(_auth_file_path()),
                        "last_refresh": getattr(entry, "last_refresh", None),
                        "auth_mode": auth_mode,
                        "source": f"pool:{getattr(entry, 'label', 'unknown')}",
                        "api_key": api_key,
                    }
            if on_pool_miss is not None:
                degraded = on_pool_miss()
                if degraded:
                    return degraded
    except Exception:
        pass

    try:
        creds = resolve()
        return {
            "logged_in": True,
            "auth_store": str(_auth_file_path()),
            "last_refresh": creds.get("last_refresh"),
            "auth_mode": creds.get("auth_mode"),
            "source": creds.get("source"),
            "api_key": creds.get("api_key"),
        }
    except AuthError as exc:
        return {
            "logged_in": False,
            "auth_store": str(_auth_file_path()),
            "error": str(exc),
        }


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


def _confirm_selection_guards(
    model_id: str,
    *,
    provider: str = "",
    base_url: str = "",
    api_key: str = "",
    include_kinds: Optional[List[str]] = None,
) -> bool:
    """Prompt before saving a model that trips any selection guard.

    Runs the unified guard registry (cost, data-policy, future guards) and shows one [y/N] confirm
    listing every warning that fired. Returns True to proceed, False to cancel.
    """
    try:
        from hermes_cli.model_selection_guards import (
            combined_message,
            selection_warnings,
        )

        warnings = selection_warnings(
            model_id,
            provider=provider,
            base_url=base_url,
            api_key=api_key,
            include_kinds=include_kinds,
        )
    except Exception:
        warnings = []
    if not warnings:
        return True

    print()
    print("=" * 72)
    print(combined_message(warnings))
    print("=" * 72)
    try:
        response = input("Switch anyway? [y/N]: ").strip().lower()
    except (KeyboardInterrupt, EOFError):
        print()
        return False
    return response in {"y", "yes"}


class _ModelPickerRows:
    """Column-aligned model rows (name + $/Mtok prices + Nous sale chrome) for the model picker.

    Sale chrome (★ / -N% / was) is drawn as curses/ANSI segments (yellow % / dim "was"), not baked
    into one plain string — curses addnstr would otherwise render escape bytes literally.
    """

    def __init__(
        self,
        all_models: List[str],
        pricing: Optional[Dict[str, Dict[str, str]]],
        *,
        current_model: str,
        sale_chrome: bool,
    ) -> None:
        from hermes_cli.models import _format_price_per_mtok, compute_sale_discount

        self.current_model = current_model
        self.has_pricing = bool(pricing and any(pricing.get(m) for m in all_models))
        # Leave room for a leading "★ " on sale rows (Nous only).
        name_pad = 3 if sale_chrome else 2
        self.name_col = (
            max((len(m) for m in all_models), default=0) + name_pad
            if self.has_pricing
            else 0
        )
        # (inp, out, cache, pct|None, was_inp, was_out)
        self._price_cache: dict[str, tuple[str, str, str, int | None, str, str]] = {}
        self.price_col = 3  # minimum width
        self.cache_col = 0  # only set if any model has cache pricing
        self.has_cache = False
        self.any_on_sale = False
        if not self.has_pricing:
            return
        for mid in all_models:
            p = pricing.get(mid)  # type: ignore[union-attr]
            pct: int | None = None
            was_inp = was_out = ""
            if p:
                inp = _format_price_per_mtok(p.get("prompt", ""))
                out = _format_price_per_mtok(p.get("completion", ""))
                cache_read = p.get("input_cache_read", "")
                cache = _format_price_per_mtok(cache_read) if cache_read else ""
                if cache:
                    self.has_cache = True
                if sale_chrome:
                    sale = compute_sale_discount(
                        p.get("prompt", ""),
                        p.get("completion", ""),
                        p.get("original"),
                    )
                    if sale is not None:
                        self.any_on_sale = True
                        pct, was_prompt_raw, was_out_raw = sale
                        # Natively-free models (no gateway original) carry
                        # empty was_* raws — leave them empty so the row
                        # shows bare "-100%" with no "was ?/?" suffix.
                        if was_prompt_raw == "" and was_out_raw == "":
                            was_inp = was_out = ""
                        else:
                            was_inp = (
                                _format_price_per_mtok(was_prompt_raw)
                                if was_prompt_raw != ""
                                else "?"
                            )
                            was_out = (
                                _format_price_per_mtok(was_out_raw)
                                if was_out_raw != ""
                                else "?"
                            )
            else:
                inp, out, cache = "", "", ""
            self._price_cache[mid] = (inp, out, cache, pct, was_inp, was_out)
            self.price_col = max(self.price_col, len(inp), len(out))
            self.cache_col = max(self.cache_col, len(cache))
        if self.has_cache:
            self.cache_col = max(self.cache_col, 5)  # minimum: "Cache" header

    def segments(self, mid: str) -> list[tuple[str, str | None]]:
        """Build a rich radiolist row: yellow ★/% , dim was, plain prices."""
        if not self.has_pricing:
            segs: list[tuple[str, str | None]] = [(mid, None)]
            if mid == self.current_model:
                segs.append(("  ← currently in use", None))
            return segs

        inp, out, cache, pct, was_inp, was_out = self._price_cache.get(
            mid, ("", "", "", None, "", "")
        )
        on_sale = pct is not None
        # Reserve 2 columns for "★ " so sale and non-sale names share alignment.
        star_w = 2
        if on_sale:
            name_segs: list[tuple[str, str | None]] = [
                ("★ ", "yellow"),
                (f"{mid:<{self.name_col - star_w}}", None),
            ]
        else:
            name_segs = [(f"{mid:<{self.name_col}}", None)]

        price_part = f" {inp:>{self.price_col}}  {out:>{self.price_col}}"
        if self.has_cache:
            price_part += f"  {cache:>{self.cache_col}}"
        segs = [*name_segs, (price_part, None)]
        if on_sale:
            segs.append((f"  -{pct}%", "yellow"))
            if was_inp or was_out:
                segs.append((f"  was {was_inp}/{was_out}", "dim"))
        if mid == self.current_model:
            segs.append(("  ← currently in use", None))
        return segs

    def label(self, mid: str) -> str:
        return "".join(text for text, _style in self.segments(mid))

    def menu_title(self) -> str:
        """``Select default model:`` plus an aligned pricing header hint when priced."""
        title = "Select default model:"
        if self.has_pricing:
            # Align the header with the model column.
            # Each choice is "  {label}" (2 spaces) and we prepend
            # a 3-char cursor region ("-> " or "   "), so content starts at col 5.
            pad = " " * 5
            header = f"\n{pad}{'':>{self.name_col}} {'In':>{self.price_col}}  {'Out':>{self.price_col}}"
            if self.has_cache:
                header += f"  {'Cache':>{self.cache_col}}"
            # Legend lives on the column-header line so it reads as a key
            # (★ = on sale), not a fake menu row.
            title += header + "  $/Mtok"
            if self.any_on_sale:
                title += "  ★ = on sale"
        return title


def _prompt_model_selection(
    model_ids: List[str],
    current_model: str = "",
    pricing: Optional[Dict[str, Dict[str, str]]] = None,
    unavailable_models: Optional[List[str]] = None,
    portal_url: str = "",
    unavailable_message: str = "",
    confirm_provider: str = "",
    confirm_base_url: str = "",
    confirm_api_key: str = "",
) -> Optional[str]:
    """Interactive model picker; current_model listed first. Returns the chosen model ID or None.

    With *pricing* (``{model_id: {prompt, completion}}``) a compact price column is shown; models in
    *unavailable_models* render grayed out and unselectable with an upgrade link to *portal_url*.
    """
    from hermes_cli.cli_output import line_input

    _unavailable = unavailable_models or []
    # Sale chrome (★ / -N% / was) is Nous Portal-only — never for OpenRouter
    # or other providers even if pricing.original is somehow present.
    sale_chrome = (confirm_provider or "").strip().lower() == "nous"

    def _confirmed_selection(mid: str) -> Optional[str]:
        if not mid:
            return None
        # Unified guard registry (hermes_cli.model_selection_guards): the cost
        # guard only runs when a provider is known (pricing lookups need one);
        # id-keyed guards like the data-policy guard always run — they must
        # fire even via a custom endpoint or gateway.
        _kinds = None if confirm_provider else ["data_policy"]
        if not _confirm_selection_guards(
            mid,
            provider=confirm_provider,
            base_url=confirm_base_url,
            api_key=confirm_api_key,
            include_kinds=_kinds,
        ):
            return None
        return mid

    # Reorder: current model first, then the rest (deduplicated)
    ordered = []
    if current_model and current_model in model_ids:
        ordered.append(current_model)
    for mid in model_ids:
        if mid not in ordered:
            ordered.append(mid)

    # All models for column-width computation (selectable + unavailable)
    rows = _ModelPickerRows(
        list(ordered) + list(_unavailable), pricing,
        current_model=current_model, sale_chrome=sale_chrome,
    )
    _DIM = "\033[2m"
    _RESET = "\033[0m"

    # Default cursor on the current model (index 0 if it was reordered to top)
    default_idx = 0
    menu_title = rows.menu_title()
    _upgrade_url = (portal_url or DEFAULT_NOUS_PORTAL_URL).rstrip("/")

    # Try arrow-key menu first, fall back to number input.
    try:
        from hermes_cli.curses_ui import curses_radiolist

        choices = [rows.segments(mid) for mid in ordered]
        choices.append("Enter custom model name")
        choices.append("Skip (keep current)")

        unavailable_footer = unavailable_message.strip()
        if not unavailable_footer and _unavailable:
            unavailable_footer = f"Upgrade at {_upgrade_url} for paid models"

        # The pricing column header (and any unavailable-models block) is shown
        # as a multi-line description above the list so it survives the curses
        # screen clear. menu_title already embeds the aligned price header.
        desc_lines: list[str] = []
        if rows.has_pricing:
            # menu_title is "Select default model:\n<pad><header>  $/Mtok\n…"
            # Keep only the header/legend portion for the description.
            header_part = menu_title.split("\n", 1)
            if len(header_part) > 1:
                desc_lines.extend(header_part[1].splitlines())
        if _unavailable:
            for mid in _unavailable:
                desc_lines.append(f"   {rows.label(mid)}")
            desc_lines.append(f"  ── {unavailable_footer} ──")
        description = "\n".join(desc_lines) if desc_lines else None

        # Search haystacks keep pricing labels visible while adding aliases
        # for brand-less wire ids (e.g. Kimi Coding `k3` ↔ query "kimi").
        from hermes_cli.model_search import model_search_text

        model_search_labels = []
        for mid in ordered:
            label = rows.label(mid)
            haystack = model_search_text(mid)
            # model_search_text always starts with the wire id; only append when
            # aliases add tokens beyond the bare id already in the label.
            model_search_labels.append(
                label if haystack == mid else f"{label} {haystack}"
            )
        model_search_labels.append("Enter custom model name")
        model_search_labels.append("Skip (keep current)")

        idx = curses_radiolist(
            "Select default model:",
            choices,
            selected=default_idx,
            cancel_returns=-1,
            description=description,
            searchable=True,
            search_labels=model_search_labels,
        )
        if idx < 0:
            return None
        print()
        if idx < len(ordered):
            return _confirmed_selection(ordered[idx])
        elif idx == len(ordered):
            try:
                custom = line_input("Enter model name: ").strip()
            except (EOFError, KeyboardInterrupt):
                return None
            return _confirmed_selection(custom) if custom else None
        return None
    except (ImportError, NotImplementedError, OSError, subprocess.SubprocessError):
        pass

    # Fallback: numbered list (ANSI colors for sale chrome)
    from hermes_cli.curses_ui import format_radio_item_ansi
    from hermes_cli.colors import Colors, color

    for line in menu_title.splitlines():
        if "★" in line:
            print(line.replace("★", color("★", Colors.YELLOW), 1))
        else:
            print(line)
    num_width = len(str(len(ordered) + 2))
    for i, mid in enumerate(ordered, 1):
        print(f"  {i:>{num_width}}. {format_radio_item_ansi(rows.segments(mid))}")
    n = len(ordered)
    print(f"  {n + 1:>{num_width}}. Enter custom model name")
    print(f"  {n + 2:>{num_width}}. Skip (keep current)")

    if _unavailable:
        unavailable_footer = unavailable_message.strip() or (
            f"Unavailable models (requires paid tier — upgrade at {_upgrade_url})"
        )
        print()
        print(f"  {_DIM}── {unavailable_footer} ──{_RESET}")
        for mid in _unavailable:
            print(f"  {'':>{num_width}}  {_DIM}{rows.label(mid)}{_RESET}")
    print()

    while True:
        try:
            choice = input(f"Choice [1-{n + 2}] (default: skip): ").strip()
            if not choice:
                return None
            idx = int(choice)
            if 1 <= idx <= n:
                return _confirmed_selection(ordered[idx - 1])
            elif idx == n + 1:
                custom = line_input("Enter model name: ").strip()
                return _confirmed_selection(custom) if custom else None
            elif idx == n + 2:
                return None
            print(f"Please enter 1-{n + 2}")
        except ValueError:
            print("Please enter a number")
        except (KeyboardInterrupt, EOFError):
            return None


def _save_model_choice(model_id: str) -> None:
    """Save the selected model to config.yaml (single source of truth).

    The model is stored in config.yaml only — NOT in .env. This avoids conflicts in multi-agent
    setups where env vars would stomp each other.
    """
    from hermes_cli.config import save_config, load_config

    config = load_config()
    # Always use dict format so provider/base_url can be stored alongside
    if isinstance(config.get("model"), dict):
        config["model"]["default"] = model_id
    else:
        config["model"] = {"default": model_id}
    save_config(config)


def login_command(args) -> None:
    """Deprecated: use 'hermes model' or 'hermes setup' instead."""
    print("The 'hermes login' command has been removed.")
    print("Use 'hermes auth' to manage credentials,")
    print("'hermes model' to select a provider, or 'hermes setup' for full setup.")
    raise SystemExit(0)


def _prompt_yes_no(prompt: str, *, default: str) -> bool:
    """``input()`` a [Y/n]-style question; EOF/Ctrl-C count as *default*."""
    try:
        answer = input(prompt).strip().lower()
    except (EOFError, KeyboardInterrupt):
        answer = default
    return answer in {"", "y", "yes"} if default == "y" else answer in {"y", "yes"}


def _print_login_success(provider_id: str, config_path: Path, *, show_auth_state: bool = False) -> None:
    print()
    print("Login successful!")
    if show_auth_state:
        from hermes_constants import display_hermes_home as _dhh
        print(f"  Auth state: {_dhh()}/auth.json")
    print(f"  Config updated: {config_path} (model.provider={provider_id})")


def _offer_existing_oauth_credentials(
    provider_id: str,
    *,
    resolve: Callable[[], Dict[str, Any]],
    is_expiring: Callable[[str, int], bool],
    display_name: str,
    default_base_url: str,
    expired_notice: Optional[str] = None,
) -> bool:
    """Offer to reuse still-valid stored OAuth credentials. Returns True when the user accepted.

    *resolve* attempts a refresh, so a resolved token should be valid — but double-check the
    expiry before telling the user "Login successful!".
    """
    try:
        existing = resolve()
        api_key = existing.get("api_key", "")
        if isinstance(api_key, str) and api_key and not is_expiring(api_key, 60):
            print(f"Existing {display_name} credentials found in Hermes auth store.")
            if _prompt_yes_no("Use existing credentials? [Y/n]: ", default="y"):
                config_path = _update_config_for_provider(
                    provider_id, existing.get("base_url", default_base_url),
                )
                _print_login_success(provider_id, config_path)
                return True
        elif expired_notice:
            print(expired_notice)
    except AuthError:
        pass
    return False


def _login_openai_codex(
    args,
    pconfig: ProviderConfig,
    *,
    force_new_login: bool = False,
) -> None:
    """OpenAI Codex login via device code flow. Tokens stored in ~/.hermes/auth.json."""

    del args, pconfig  # kept for parity with other provider login helpers

    # Check for existing Hermes-owned credentials
    if not force_new_login and _offer_existing_oauth_credentials(
        "openai-codex",
        resolve=resolve_codex_runtime_credentials,
        is_expiring=_codex_access_token_is_expiring,
        display_name="Codex",
        default_base_url=DEFAULT_CODEX_BASE_URL,
        expired_notice="Existing Codex credentials are expired. Starting fresh login...",
    ):
        return

    # Check for existing Codex CLI tokens we can import
    if not force_new_login:
        cli_tokens = _import_codex_cli_tokens()
        if cli_tokens:
            print("Found existing Codex CLI credentials at ~/.codex/auth.json")
            print("Hermes will create its own session to avoid conflicts with Codex CLI / VS Code.")
            if _prompt_yes_no(
                "Import these credentials? (a separate login is recommended) [y/N]: ", default="n",
            ):
                _save_codex_tokens(cli_tokens)
                config_path = _update_config_for_provider("openai-codex", _codex_base_url())
                print()
                print("Credentials imported. Note: if Codex CLI refreshes its token,")
                print("Hermes will keep working independently with its own session.")
                print(f"  Config updated: {config_path} (model.provider=openai-codex)")
                return

    # Run a fresh device code flow — Hermes gets its own OAuth session
    print()
    print("Signing in to OpenAI Codex...")
    print("(Hermes creates its own session — won't affect Codex CLI or VS Code)")
    print()

    creds = _codex_device_code_login()

    # Save tokens to Hermes auth store
    _save_codex_tokens(creds["tokens"], creds.get("last_refresh"))
    config_path = _update_config_for_provider("openai-codex", creds.get("base_url", DEFAULT_CODEX_BASE_URL))
    _print_login_success("openai-codex", config_path, show_auth_state=True)


def _login_xai_oauth(
    args,
    pconfig: ProviderConfig,
    *,
    force_new_login: bool = False,
) -> None:
    del pconfig

    if not force_new_login and _offer_existing_oauth_credentials(
        "xai-oauth",
        resolve=resolve_xai_oauth_runtime_credentials,
        is_expiring=_xai_access_token_is_expiring,
        display_name="xAI OAuth",
        default_base_url=DEFAULT_XAI_OAUTH_BASE_URL,
    ):
        return

    print()
    print("Signing in to xAI Grok OAuth (SuperGrok / Premium+)...")
    print("(Hermes creates its own local OAuth session)")
    print()

    timeout_seconds = float(getattr(args, "timeout", None) or 20.0)
    open_browser = not getattr(args, "no_browser", False)
    if _is_remote_session():
        open_browser = False

    creds = _xai_oauth_device_code_login(
        timeout_seconds=timeout_seconds,
        open_browser=open_browser,
    )
    _save_xai_oauth_tokens(
        creds["tokens"],
        discovery=creds.get("discovery"),
        redirect_uri=creds.get("redirect_uri", ""),
        last_refresh=creds.get("last_refresh"),
        auth_mode="oauth_device_code",
    )
    # An explicit interactive re-login is a strong signal the user wants the
    # xAI credential re-enabled. ``hermes auth remove xai-oauth`` leaves a
    # ``device_code`` suppression marker that otherwise stops the singleton
    # seed from re-creating the pool entry, so ``hermes auth list`` would show
    # nothing even though the agent still works via the singleton fallback.
    # Clear it here (same helper ``auth_add_command`` uses). This is kept OUT
    # of ``_save_xai_oauth_tokens`` on purpose — that helper is shared with the
    # refresh hot path, which must never mutate suppression state.
    unsuppress_credential_source("xai-oauth", "device_code")
    config_path = _update_config_for_provider("xai-oauth", creds.get("base_url", DEFAULT_XAI_OAUTH_BASE_URL))
    _print_login_success("xai-oauth", config_path, show_auth_state=True)


def _xai_oauth_request_device_code(
    client: httpx.Client,
    *,
    scope: str = XAI_OAUTH_SCOPE,
) -> Dict[str, Any]:
    response = client.post(
        XAI_OAUTH_DEVICE_CODE_URL,
        headers=_FORM_JSON_HEADERS,
        data={
            "client_id": XAI_OAUTH_CLIENT_ID,
            "scope": scope,
        },
    )
    if response.status_code != 200:
        raise _xai_err(
            f"xAI device-code request failed (HTTP {response.status_code})."
            + (f" Response: {response.text.strip()}" if response.text else ""),
            "device_code_request_failed",
        )
    payload = response.json()
    required = (
        "device_code",
        "user_code",
        "verification_uri",
        "verification_uri_complete",
        "expires_in",
        "interval",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise _xai_err(
            f"xAI device-code response missing fields: {', '.join(missing)}",
            "device_code_invalid",
        )
    return payload


def _xai_oauth_poll_device_token(
    client: httpx.Client,
    *,
    token_endpoint: str,
    device_code: str,
    expires_in: int,
    poll_interval: int,
) -> Dict[str, Any]:
    def _validate(payload: Dict[str, Any]) -> None:
        for field_name, article in (("access_token", "an"), ("refresh_token", "a")):
            if not payload.get(field_name):
                raise _xai_err(
                    f"xAI device-code token response did not include {article} {field_name}.",
                    "xai_device_token_invalid",
                )

    def _error(response, error_payload) -> Exception:
        description = (
            error_payload.get("error_description")
            or error_payload.get("error")
            or response.text
        )
        return _xai_err(
            f"xAI device-code token polling failed: {description}",
            "xai_device_token_failed",
        )

    return _poll_device_token_generic(
        lambda: client.post(
            token_endpoint,
            headers=_FORM_JSON_HEADERS,
            data={
                "grant_type": DEVICE_CODE_GRANT_TYPE,
                "client_id": XAI_OAUTH_CLIENT_ID,
                "device_code": device_code,
            },
        ),
        expires_in=int(expires_in),
        poll_interval=max(1, int(poll_interval)),
        validate_success=_validate,
        on_non_json_error=lambda _r: _xai_err(
            "xAI device-code token polling returned a non-JSON error response.",
            "xai_device_token_failed",
        ),
        on_error=_error,
        on_timeout=lambda: _xai_err(
            "Timed out waiting for xAI device authorization.",
            "device_code_timeout",
        ),
    )


def _xai_oauth_device_code_login(
    *,
    timeout_seconds: float = 20.0,
    open_browser: bool = True,
) -> Dict[str, Any]:
    discovery = _xai_oauth_discovery(timeout_seconds)
    token_endpoint = discovery["token_endpoint"]
    timeout = httpx.Timeout(max(20.0, timeout_seconds))
    with httpx.Client(timeout=timeout, headers={"Accept": "application/json"}) as client:
        device_data = _xai_oauth_request_device_code(client)
        verification_url = str(
            device_data.get("verification_uri_complete")
            or device_data["verification_uri"]
        )
        user_code = str(device_data["user_code"])
        expires_in = int(device_data["expires_in"])
        interval = int(device_data["interval"])

        _print_device_code_instructions(
            verification_url,
            user_code,
            open_browser=open_browser and not _is_remote_session() and _can_open_graphical_browser(),
            swallow_open_errors=True,
        )
        print(f"Waiting for approval (polling every {max(1, interval)}s)...")

        payload = _xai_oauth_poll_device_token(
            client,
            token_endpoint=token_endpoint,
            device_code=str(device_data["device_code"]),
            expires_in=expires_in,
            poll_interval=interval,
        )

    access_token = str(payload.get("access_token", "") or "").strip()
    refresh_token = str(payload.get("refresh_token", "") or "").strip()
    if not access_token or not refresh_token:
        raise _xai_err(
            "xAI device-code token response was missing required tokens.",
            "xai_device_token_invalid",
        )
    base_url = _xai_oauth_inference_base_url()
    return {
        "tokens": _xai_tokens_from_payload(payload, access_token, refresh_token),
        "discovery": discovery,
        "redirect_uri": "",
        "base_url": base_url,
        "last_refresh": _utc_now_z(),
        "source": "oauth-device-code",
    }


def _codex_login_rate_limited_error(response: "httpx.Response", *, during: str = "") -> AuthError:
    """AuthError for a 429 from OpenAI's device-auth endpoints (a throttle, not a credential fault)."""
    retry_after = _parse_retry_after_seconds(getattr(response, "headers", None))
    wait_hint = (
        f" Try again in about {retry_after}s."
        if retry_after is not None
        else " Wait a minute and run the login again."
    )
    return _codex_err(
        f"OpenAI is rate-limiting Codex login requests (HTTP 429){during}. "
        "This is a temporary throttle on OpenAI's side, not a credential "
        f"problem.{wait_hint}",
        CODEX_RATE_LIMITED_CODE,
    )


def _codex_request_device_code(issuer: str, client_id: str) -> Dict[str, Any]:
    """Step 1 of the Codex device flow: request a user code, retrying capped on HTTP 429."""
    # OpenAI's auth endpoint rate-limits this request (HTTP 429) when login is
    # attempted too often from the same IP/account — retry with capped backoff
    # (honoring ``Retry-After``) before surfacing a clear, actionable message.
    resp = None
    max_attempts = 4
    for attempt in range(1, max_attempts + 1):
        try:
            with _codex_http_client(timeout=httpx.Timeout(15.0)) as client:
                resp = client.post(
                    f"{issuer}/api/accounts/deviceauth/usercode",
                    json={"client_id": client_id},
                    headers={"Content-Type": "application/json"},
                )
        except Exception as exc:
            raise _codex_err(f"Failed to request device code: {exc}", "device_code_request_failed")

        if resp.status_code != 429:
            break

        if attempt < max_attempts:
            retry_after = _parse_retry_after_seconds(
                getattr(resp, "headers", None)
            )
            # Exponential backoff (2s, 4s, 8s) capped, preferring the
            # server-provided Retry-After when present.
            delay = retry_after if retry_after is not None else 2 ** attempt
            delay = max(1, min(int(delay), 60))
            print(
                "OpenAI is rate-limiting login requests "
                f"(429); retrying in {delay}s..."
            )
            time.sleep(delay)

    if resp is not None and resp.status_code == 429:
        raise _codex_login_rate_limited_error(resp)

    if resp is None or resp.status_code != 200:
        status = resp.status_code if resp is not None else "unknown"
        raise _codex_err(
            f"Device code request returned status {status}.",
            "device_code_request_error",
        )

    device_data = resp.json()
    device_data["interval"] = max(3, int(device_data.get("interval", "5")))
    if not device_data.get("user_code", "") or not device_data.get("device_auth_id", ""):
        raise _codex_err("Device code response missing required fields.", "device_code_incomplete")
    return device_data


def _codex_poll_authorization_code(
    issuer: str, *, device_auth_id: str, user_code: str, poll_interval: int,
) -> Dict[str, Any]:
    """Step 3 of the Codex device flow: poll until sign-in completes (403/404 = still pending)."""
    max_wait = 15 * 60  # 15 minutes
    start = time.monotonic()
    code_resp = None

    try:
        with _codex_http_client(timeout=httpx.Timeout(15.0)) as client:
            while time.monotonic() - start < max_wait:
                time.sleep(poll_interval)
                poll_resp = client.post(
                    f"{issuer}/api/accounts/deviceauth/token",
                    json={"device_auth_id": device_auth_id, "user_code": user_code},
                    headers={"Content-Type": "application/json"},
                )

                if poll_resp.status_code == 200:
                    code_resp = poll_resp.json()
                    break
                elif poll_resp.status_code in {403, 404}:
                    continue  # User hasn't completed login yet
                else:
                    raise _codex_err(
                        f"Device auth polling returned status {poll_resp.status_code}.",
                        "device_code_poll_error",
                    )
    except KeyboardInterrupt:
        print("\nLogin cancelled.")
        raise SystemExit(130)

    if code_resp is None:
        raise _codex_err("Login timed out after 15 minutes.", "device_code_timeout")
    return code_resp


def _codex_exchange_authorization_code(
    issuer: str, client_id: str, code_resp: Dict[str, Any],
) -> Dict[str, Any]:
    """Step 4 of the Codex device flow: swap the authorization code for tokens."""
    authorization_code = code_resp.get("authorization_code", "")
    code_verifier = code_resp.get("code_verifier", "")
    redirect_uri = f"{issuer}/deviceauth/callback"

    if not authorization_code or not code_verifier:
        raise _codex_err(
            "Device auth response missing authorization_code or code_verifier.",
            "device_code_incomplete_exchange",
        )

    try:
        with _codex_http_client(timeout=httpx.Timeout(15.0)) as client:
            token_resp = client.post(
                CODEX_OAUTH_TOKEN_URL,
                data={
                    "grant_type": "authorization_code",
                    "code": authorization_code,
                    "redirect_uri": redirect_uri,
                    "client_id": client_id,
                    "code_verifier": code_verifier,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
    except Exception as exc:
        raise _codex_err(f"Token exchange failed: {exc}", "token_exchange_failed")

    if token_resp.status_code == 429:
        raise _codex_login_rate_limited_error(token_resp, during=" during token exchange")

    if token_resp.status_code != 200:
        raise _codex_err(
            f"Token exchange returned status {token_resp.status_code}.",
            "token_exchange_error",
        )

    tokens = token_resp.json()
    if not tokens.get("access_token", ""):
        raise _codex_err(
            "Token exchange did not return an access_token.",
            "token_exchange_no_access_token",
        )
    return tokens


def _codex_device_code_login() -> Dict[str, Any]:
    """Run the OpenAI device code login flow and return credentials dict."""
    issuer = "https://auth.openai.com"
    client_id = CODEX_OAUTH_CLIENT_ID

    device_data = _codex_request_device_code(issuer, client_id)
    user_code = device_data["user_code"]
    device_auth_id = device_data["device_auth_id"]
    poll_interval = device_data["interval"]

    # Step 2: Show user the code
    print("To continue, follow these steps:\n")
    print("  1. Open this URL in your browser:")
    print(f"     \033[94m{issuer}/codex/device\033[0m\n")
    print("  2. Enter this code:")
    print(f"     \033[94m{user_code}\033[0m\n")
    print("Waiting for sign-in... (press Ctrl+C to cancel)")

    code_resp = _codex_poll_authorization_code(
        issuer, device_auth_id=device_auth_id, user_code=user_code, poll_interval=poll_interval,
    )
    tokens = _codex_exchange_authorization_code(issuer, client_id, code_resp)

    # Return tokens for the caller to persist (no longer writes to ~/.codex/)
    return {
        "tokens": {
            "access_token": tokens.get("access_token", ""),
            "refresh_token": tokens.get("refresh_token", ""),
        },
        "base_url": _codex_base_url(),
        "last_refresh": _utc_now_z(),
        "auth_mode": "chatgpt",
        "source": "device-code",
    }


# ==================== MiniMax Portal OAuth ====================

_MINIMAX_OAUTH_ERROR_BODY_LIMIT = 16 * 1024


def _minimax_response_error_text(
    response: httpx.Response,
    *,
    limit: int = _MINIMAX_OAUTH_ERROR_BODY_LIMIT,
) -> str:
    """Return a bounded error body from a streamed MiniMax OAuth response."""
    limit = max(0, int(limit))
    chunks: list[bytes] = []
    total = 0
    truncated = False
    try:
        if getattr(response, "is_stream_consumed", False):
            text = response.text
            return text[:limit] + ("...[truncated]" if len(text) > limit else "")

        for chunk in response.iter_bytes():
            if not chunk:
                continue
            remaining = limit + 1 - total
            if remaining <= 0:
                truncated = True
                break
            if len(chunk) > remaining:
                chunks.append(chunk[:remaining])
                total += remaining
                truncated = True
                break
            chunks.append(chunk)
            total += len(chunk)
        raw = b"".join(chunks)
        if len(raw) > limit:
            raw = raw[:limit]
            truncated = True
        encoding = response.encoding or "utf-8"
        text = raw.decode(encoding, errors="replace")
        return text + ("...[truncated]" if truncated else "")
    finally:
        response.close()


def _minimax_post_form(
    client: httpx.Client,
    url: str,
    *,
    data: Dict[str, Any],
    headers: Dict[str, str],
) -> httpx.Response:
    """POST a MiniMax OAuth form without eagerly reading error bodies."""
    request = client.build_request(
        "POST",
        url,
        data=data,
        headers=headers,
    )
    response = client.send(request, stream=True)
    if response.status_code == 200:
        response.read()
    return response

def _minimax_pkce_pair() -> tuple:
    """Generate (code_verifier, code_challenge_S256, state) for MiniMax OAuth."""
    import secrets
    verifier = secrets.token_urlsafe(64)[:96]
    challenge = base64.urlsafe_b64encode(
        hashlib.sha256(verifier.encode()).digest()
    ).decode().rstrip("=")
    state = secrets.token_urlsafe(16)
    return verifier, challenge, state


def _minimax_request_user_code(
    client: httpx.Client, *, portal_base_url: str, client_id: str,
    code_challenge: str, state: str,
) -> Dict[str, Any]:
    response = _minimax_post_form(
        client,
        f"{portal_base_url}/oauth/code",
        data={
            "response_type": "code",
            "client_id": client_id,
            "scope": MINIMAX_OAUTH_SCOPE,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
            "state": state,
        },
        headers={**_FORM_JSON_HEADERS, "x-request-id": str(uuid.uuid4())},
    )
    if response.status_code != 200:
        body = _minimax_response_error_text(response)
        raise _minimax_err(
            f"MiniMax OAuth authorization failed: {body or response.reason_phrase}",
            "authorization_failed",
        )
    payload = response.json()
    for field in ("user_code", "verification_uri", "expired_in"):
        if field not in payload:
            raise _minimax_err(
                f"MiniMax OAuth response missing field: {field}",
                "authorization_incomplete",
            )
    if payload.get("state") != state:
        raise _minimax_err("MiniMax OAuth state mismatch (possible CSRF).", "state_mismatch")
    return payload


def _minimax_expired_in_looks_like_unix_ms(expired_in: int, *, now_ms: int) -> bool:
    """True if ``expired_in`` is plausibly a unix-ms absolute time (vs TTL seconds)."""
    return int(expired_in) > (now_ms // 2)


def _minimax_resolve_token_expiry_unix(expired_in: int, *, now: datetime) -> float:
    """Return access-token expiry as unix seconds (MiniMax uses ms epoch or TTL seconds)."""
    raw = int(expired_in)
    now_ms = int(now.timestamp() * 1000)
    if _minimax_expired_in_looks_like_unix_ms(raw, now_ms=now_ms):
        return raw / 1000.0
    return now.timestamp() + max(1, raw)


def _minimax_expiry_fields(expired_in: Any) -> Dict[str, Any]:
    """``obtained_at`` / ``expires_at`` / ``expires_in`` derived from a MiniMax ``expired_in``."""
    now = datetime.now(timezone.utc)
    expires_at_unix = _minimax_resolve_token_expiry_unix(int(expired_in), now=now)
    return {
        "obtained_at": now.isoformat(),
        "expires_at": datetime.fromtimestamp(expires_at_unix, tz=timezone.utc).isoformat(),
        "expires_in": max(0, int(expires_at_unix - now.timestamp())),
    }


def _minimax_poll_token(
    client: httpx.Client, *, portal_base_url: str, client_id: str,
    user_code: str, code_verifier: str, expired_in: int, interval_ms: Optional[int],
) -> Dict[str, Any]:
    # OpenClaw treats expired_in as a unix-ms timestamp (Date.now() < expireTimeMs).
    # Defensive parsing: if it's small enough to be a duration, treat as seconds.
    deadline = _minimax_resolve_token_expiry_unix(expired_in, now=datetime.now(timezone.utc))
    interval = max(2.0, (interval_ms or 2000) / 1000.0)

    while time.time() < deadline:
        response = _minimax_post_form(
            client,
            f"{portal_base_url}/oauth/token",
            data={
                "grant_type": MINIMAX_OAUTH_GRANT_TYPE,
                "client_id": client_id,
                "user_code": user_code,
                "code_verifier": code_verifier,
            },
            headers=_FORM_JSON_HEADERS,
        )
        error_text = ""
        if response.status_code != 200:
            error_text = _minimax_response_error_text(response)
            try:
                payload = json.loads(error_text) if error_text else {}
            except Exception:
                payload = {}
            msg = (payload.get("base_resp", {}) or {}).get("status_msg") or error_text
            raise _minimax_err(f"MiniMax OAuth error: {msg or 'unknown'}", "token_exchange_failed")
        try:
            payload = response.json() if response.text else {}
        except Exception:
            payload = {}

        status = payload.get("status")
        if status == "error":
            raise _minimax_err(
                "MiniMax OAuth reported an error. Please try again later.",
                "authorization_denied",
            )
        if status == "success":
            if not all(payload.get(k) for k in ("access_token", "refresh_token", "expired_in")):
                raise _minimax_err(
                    "MiniMax OAuth success payload missing required token fields.",
                    "token_incomplete",
                )
            return payload
        # "pending" or any other status -> keep polling
        time.sleep(interval)

    raise _minimax_err("MiniMax OAuth timed out before authorization completed.", "timeout")


def _minimax_save_auth_state(auth_state: Dict[str, Any]) -> None:
    """Persist MiniMax OAuth state to Hermes auth store (~/.hermes/auth.json)."""
    _save_active_provider_state("minimax-oauth", auth_state)


def _minimax_oauth_login(
    *, region: str = "global", open_browser: bool = True,
    timeout_seconds: float = 15.0,
) -> Dict[str, Any]:
    """Run MiniMax OAuth flow, persist tokens, return auth state dict."""
    pconfig = PROVIDER_REGISTRY["minimax-oauth"]
    if region == "cn":
        portal_base_url = pconfig.extra["cn_portal_base_url"]
        inference_base_url = pconfig.extra["cn_inference_base_url"]
    else:
        portal_base_url = pconfig.portal_base_url
        inference_base_url = pconfig.inference_base_url

    verifier, challenge, state = _minimax_pkce_pair()

    if _is_remote_session():
        open_browser = False

    print(f"Starting Hermes login via MiniMax ({region}) OAuth...")
    print(f"Portal: {portal_base_url}")

    with httpx.Client(timeout=httpx.Timeout(timeout_seconds),
                      headers={"Accept": "application/json"},
                      follow_redirects=True) as client:
        code_data = _minimax_request_user_code(
            client, portal_base_url=portal_base_url,
            client_id=pconfig.client_id,
            code_challenge=challenge, state=state,
        )
        verification_url = str(code_data["verification_uri"])
        user_code = str(code_data["user_code"])

        _print_device_code_instructions(
            verification_url,
            user_code,
            open_browser=open_browser and _can_open_graphical_browser(),
        )

        interval_raw = code_data.get("interval")
        interval_ms = int(interval_raw) if interval_raw is not None else None
        print("Waiting for approval...")

        token_data = _minimax_poll_token(
            client, portal_base_url=portal_base_url,
            client_id=pconfig.client_id,
            user_code=user_code, code_verifier=verifier,
            expired_in=int(code_data["expired_in"]),
            interval_ms=interval_ms,
        )

    auth_state = {
        "provider": "minimax-oauth",
        "region": region,
        "portal_base_url": portal_base_url,
        "inference_base_url": inference_base_url,
        "client_id": pconfig.client_id,
        "scope": MINIMAX_OAUTH_SCOPE,
        "token_type": token_data.get("token_type", "Bearer"),
        "access_token": token_data["access_token"],
        "refresh_token": token_data["refresh_token"],
        "resource_url": token_data.get("resource_url"),
        **_minimax_expiry_fields(token_data["expired_in"]),
    }

    _minimax_save_auth_state(auth_state)
    print("\u2713 MiniMax OAuth login successful.")
    if msg := token_data.get("notification_message"):
        print(f"Note from MiniMax: {msg}")
    return auth_state


def _refresh_minimax_oauth_state(
    state: Dict[str, Any], *, timeout_seconds: float = 15.0,
    force: bool = False,
) -> Dict[str, Any]:
    """Refresh MiniMax OAuth access token if close to expiry (or forced)."""
    if not state.get("refresh_token"):
        raise _minimax_err(
            "MiniMax OAuth state has no refresh_token; please re-login.",
            "no_refresh_token", relogin=True,
        )
    try:
        expires_at = datetime.fromisoformat(state.get("expires_at", "")).timestamp()
    except Exception:
        expires_at = 0.0
    now = time.time()
    if not force and (expires_at - now) > MINIMAX_OAUTH_REFRESH_SKEW_SECONDS:
        return state

    portal_base_url = state["portal_base_url"]
    with httpx.Client(timeout=httpx.Timeout(timeout_seconds),
                      follow_redirects=True) as client:
        response = _minimax_post_form(
            client,
            f"{portal_base_url}/oauth/token",
            data={
                "grant_type": "refresh_token",
                "client_id": state["client_id"],
                "refresh_token": state["refresh_token"],
            },
            headers=_FORM_JSON_HEADERS,
        )
        # The non-200 branch reads a STREAMED body, so it must run while
        # the client is still open — iter_bytes() after the client context
        # closes raises (StreamClosed).  The 200 path was already read by
        # _minimax_post_form, so response.json() below is safe outside.
        if response.status_code != 200:
            body = _minimax_response_error_text(response)
            body_lower = body.lower()
            relogin = any(m in body_lower for m in
                          ("invalid_grant", "refresh_token_reused", "invalid_refresh_token"))
            raise _minimax_err(
                f"MiniMax OAuth refresh failed: {body or response.reason_phrase}",
                "refresh_failed", relogin=relogin,
            )
    payload = response.json()
    if payload.get("status") != "success":
        raise _minimax_err(
            "MiniMax OAuth refresh did not return success.",
            "refresh_failed", relogin=True,
        )
    new_state = dict(state)
    new_state.update({
        "access_token": payload["access_token"],
        "refresh_token": payload.get("refresh_token", state["refresh_token"]),
        **_minimax_expiry_fields(payload["expired_in"]),
    })
    _minimax_save_auth_state(new_state)
    return new_state


def _minimax_oauth_quarantine_on_terminal_refresh(state: Dict[str, Any], exc: AuthError) -> None:
    """Wipe dead tokens from auth.json after a terminal refresh failure.

    Shared by the eager-resolve path and the lazy per-request token provider. Mirrors the
    Nous / xAI / Codex quarantine pattern so subsequent calls fail fast without a network retry.
    """
    if not (exc.relogin_required and state.get("refresh_token")):
        return
    _quarantine_flat_oauth_state(state, "minimax-oauth", exc)
    try:
        _minimax_save_auth_state(state)
    except Exception as _save_exc:
        logger.debug("MiniMax OAuth: failed to persist quarantined state: %s", _save_exc)


def _minimax_fresh_state() -> Dict[str, Any]:
    """Load the MiniMax OAuth state and refresh it if near expiry; quarantine on terminal failure."""
    state = get_provider_auth_state("minimax-oauth")
    if not state or not state.get("access_token"):
        raise _minimax_err(
            "Not logged into MiniMax OAuth. Run `hermes model` and select "
            "MiniMax (OAuth).",
            "not_logged_in", relogin=True,
        )
    try:
        return _refresh_minimax_oauth_state(state)
    except AuthError as exc:
        _minimax_oauth_quarantine_on_terminal_refresh(state, exc)
        raise


def build_minimax_oauth_token_provider() -> Callable[[], str]:
    """Return a zero-arg callable that yields a fresh MiniMax access token.

    The Anthropic SDK caches ``api_key`` as a static string at construction time, so a session that
    resolves credentials once at startup will keep sending the same bearer until MiniMax's server
    returns 401 — typically ~15 minutes in, because MiniMax issues short-lived access tokens.
    """
    def _provide() -> str:
        state = _minimax_fresh_state()
        token = state.get("access_token")
        if not token:
            raise _minimax_err(
                "MiniMax OAuth state has no access_token after refresh.",
                "no_access_token", relogin=True,
            )
        return token

    return _provide


def resolve_minimax_oauth_runtime_credentials(
    *, min_token_ttl_seconds: int = MINIMAX_OAUTH_REFRESH_SKEW_SECONDS,
    as_token_provider: bool = False,
) -> Dict[str, Any]:
    """Return {provider, api_key, base_url, source} for minimax-oauth.

    The default (string ``api_key``) preserves the historical contract for diagnostic call sites
    like ``hermes status`` that just want to know whether a valid token exists right now.
    """
    state = _minimax_fresh_state()
    if as_token_provider:
        api_key: Any = build_minimax_oauth_token_provider()
    else:
        api_key = state["access_token"]
    return {
        "provider": "minimax-oauth",
        "api_key": api_key,
        "base_url": state["inference_base_url"].rstrip("/"),
        "source": "oauth",
    }


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


def _login_minimax_oauth(args, pconfig: ProviderConfig) -> None:
    """CLI entry for MiniMax OAuth login."""
    region = getattr(args, "region", None) or "global"
    open_browser = not getattr(args, "no_browser", False)
    timeout = getattr(args, "timeout", None) or 15.0
    try:
        _minimax_oauth_login(
            region=region, open_browser=open_browser, timeout_seconds=timeout,
        )
    except AuthError as exc:
        print(format_auth_error(exc))
        raise SystemExit(1)


def _nous_device_code_login(
    *,
    portal_base_url: Optional[str] = None,
    inference_base_url: Optional[str] = None,
    client_id: Optional[str] = None,
    scope: Optional[str] = None,
    open_browser: bool = True,
    timeout_seconds: float = 15.0,
    insecure: bool = False,
    ca_bundle: Optional[str] = None,
    on_verification: Optional[Callable[[str, str], None]] = None,
) -> Dict[str, Any]:
    """Run the Nous device-code flow and return full OAuth state without persisting."""
    pconfig = PROVIDER_REGISTRY["nous"]
    portal_base_url = (
        portal_base_url
        or os.getenv("HERMES_PORTAL_BASE_URL")
        or os.getenv("NOUS_PORTAL_BASE_URL")
        or pconfig.portal_base_url
    ).rstrip("/")
    requested_inference_url = (
        inference_base_url
        or os.getenv("NOUS_INFERENCE_BASE_URL")
        or pconfig.inference_base_url
    ).rstrip("/")
    client_id = client_id or pconfig.client_id
    scope = scope or pconfig.scope
    timeout = httpx.Timeout(timeout_seconds)
    verify: bool | str = False if insecure else (ca_bundle if ca_bundle else True)

    if _is_remote_session():
        open_browser = False

    print(f"Starting Hermes login via {pconfig.name}...")
    print(f"Portal: {portal_base_url}")
    if insecure:
        print("TLS verification: disabled (--insecure)")
    elif ca_bundle:
        print(f"TLS verification: custom CA bundle ({ca_bundle})")

    with httpx.Client(timeout=timeout, headers={"Accept": "application/json"}, verify=verify) as client:
        device_data = _request_device_code(
            client=client,
            portal_base_url=portal_base_url,
            client_id=client_id,
            scope=scope,
        )

        verification_url = str(device_data["verification_uri_complete"])
        user_code = str(device_data["user_code"])
        expires_in = int(device_data["expires_in"])
        interval = int(device_data["interval"])

        _print_device_code_instructions(
            verification_url, user_code, open_browser=open_browser, failure_dash="—",
        )

        # Surface the verification URL/code to an out-of-band consumer (e.g. the
        # TUI gateway, whose stdout is a JSON-RPC pipe — a plain print() there is
        # dropped). Fired AFTER the print/browser block and BEFORE polling blocks,
        # so the consumer can render the link while we wait. Best-effort.
        if on_verification is not None:
            try:
                on_verification(verification_url, user_code)
            except Exception:
                pass

        effective_interval = max(1, min(interval, DEVICE_AUTH_POLL_INTERVAL_CAP_SECONDS))
        print(f"Waiting for approval (polling every {effective_interval}s)...")

        token_data = _poll_for_token(
            client=client,
            portal_base_url=portal_base_url,
            client_id=client_id,
            device_code=str(device_data["device_code"]),
            expires_in=expires_in,
            poll_interval=interval,
        )

    now = datetime.now(timezone.utc)
    token_expires_in = _coerce_ttl_seconds(token_data.get("expires_in", 0))
    expires_at = now.timestamp() + token_expires_in
    resolved_inference_url = (
        _optional_base_url(token_data.get("inference_base_url"))
        or requested_inference_url
    )
    if resolved_inference_url != requested_inference_url:
        print(f"Using portal-provided inference URL: {resolved_inference_url}")

    auth_state = {
        "portal_base_url": portal_base_url,
        "inference_base_url": resolved_inference_url,
        "client_id": client_id,
        "scope": token_data.get("scope") or scope,
        "token_type": token_data.get("token_type", "Bearer"),
        "access_token": token_data["access_token"],
        "refresh_token": token_data.get("refresh_token"),
        "obtained_at": now.isoformat(),
        "expires_at": datetime.fromtimestamp(expires_at, tz=timezone.utc).isoformat(),
        "expires_in": token_expires_in,
        "tls": _tls_state_from_verify(verify),
        **_NOUS_EMPTY_AGENT_KEY_FIELDS,
    }
    try:
        return refresh_nous_oauth_from_state(
            auth_state,
            timeout_seconds=timeout_seconds,
            force_refresh=False,
        )
    except AuthError as exc:
        if exc.code == "subscription_required":
            portal_url = auth_state.get(
                "portal_base_url", DEFAULT_NOUS_PORTAL_URL
            ).rstrip("/")
            message = format_auth_error(exc)
            print()
            print(message)
            print(f"  Subscribe here: {portal_url}/billing")
            print()
            print("After subscribing, run `hermes model` again to finish setup.")
            raise SystemExit(1)
        raise


def _mirror_nous_state_best_effort(auth_state: Dict[str, Any]) -> None:
    """Mirror to the shared store + reseed the pool, swallowing all errors (same as _login_nous)."""
    try:
        _write_shared_nous_state(auth_state)
    except Exception:
        pass
    try:
        _sync_nous_pool_from_auth_store()
    except Exception:
        pass


def step_up_nous_billing_scope(
    *,
    open_browser: bool = True,
    timeout_seconds: float = 15.0,
    on_verification: Optional[Callable[[str, str], None]] = None,
) -> bool:
    """Re-run the device flow requesting ``billing:manage`` and persist the result.

    Lazy step-up triggered by ``403 insufficient_scope``. The user must be ADMIN/OWNER and select
    "Allow Remote Spending" in the portal, otherwise the server silently downscopes and this returns
    False. Reuses the held credential's portal/inference URLs + client_id so the step-up targets the
    same deployment, and persists like ``_login_nous`` but WITHOUT the model picker.
    """
    prior = get_provider_auth_state("nous") or {}
    pconfig = PROVIDER_REGISTRY["nous"]

    # Build the step-up scope: existing scopes (if any) + billing:manage, deduped,
    # order-stable. Fall back to the standard inference+tool+billing set.
    _raw_scope = prior.get("scope")
    prior_scope = _raw_scope if isinstance(_raw_scope, str) else ""
    requested: list[str] = []
    for tok in (prior_scope.split() or [NOUS_INFERENCE_INVOKE_SCOPE, "tool:invoke"]):
        if tok and tok not in requested:
            requested.append(tok)
    if NOUS_BILLING_MANAGE_SCOPE not in requested:
        requested.append(NOUS_BILLING_MANAGE_SCOPE)
    scope = " ".join(requested)

    auth_state = _nous_device_code_login(
        portal_base_url=prior.get("portal_base_url") or None,
        inference_base_url=prior.get("inference_base_url") or None,
        client_id=prior.get("client_id") or pconfig.client_id,
        scope=scope,
        open_browser=open_browser,
        timeout_seconds=timeout_seconds,
        on_verification=on_verification,
    )

    _save_active_provider_state("nous", auth_state)
    _mirror_nous_state_best_effort(auth_state)

    granted = auth_state.get("scope")
    return isinstance(granted, str) and NOUS_BILLING_MANAGE_SCOPE in granted.split()


def _pick_nous_model_after_login(auth_state: Dict[str, Any], inference_base_url: str) -> Optional[str]:
    """Fetch the curated Nous model list (tier/policy-filtered) and run the interactive picker.

    Returns the selected model id, or None when the user skipped / nothing was selectable.
    Raises on any fetch failure so the caller can print the "Login succeeded, but..." notice.
    """
    runtime_key = auth_state.get("agent_key") or auth_state.get("access_token")
    if not isinstance(runtime_key, str) or not runtime_key:
        raise _nous_err("No runtime API key available to fetch models", "invalid_token")

    from hermes_cli.models import (
        get_curated_nous_model_ids, get_pricing_for_provider,
        check_nous_free_tier, partition_nous_models_by_tier,
        nous_policy_allowed_ids, restrict_to_nous_policy,
        union_with_portal_free_recommendations,
        union_with_portal_paid_recommendations,
    )
    model_ids = get_curated_nous_model_ids()

    print()
    unavailable_models: list = []
    unavailable_message = ""
    if model_ids:
        pricing = get_pricing_for_provider("nous")
        # Force fresh account data for model selection so recent credit
        # purchases are reflected immediately.
        free_tier = check_nous_free_tier(force_fresh=True)
        _portal_for_recs = auth_state.get("portal_base_url", "")
        # Narrow before the tier split, so a rescued id still has to
        # pass the free/paid predicate.
        _policy_allowed = nous_policy_allowed_ids()
        _policy_narrowed = False
        if free_tier:
            try:
                from hermes_cli.nous_account import (
                    format_nous_portal_entitlement_message,
                    get_nous_portal_account_info,
                )

                _account_info = get_nous_portal_account_info(force_fresh=True)
                unavailable_message = (
                    format_nous_portal_entitlement_message(
                        _account_info,
                        capability="paid Nous models",
                    )
                    or ""
                )
            except Exception:
                unavailable_message = ""
        # The Portal's free/paidRecommendedModels endpoint is the source of
        # truth for what's available *right now*. Augment the curated list with
        # anything new the Portal flags so users on older Hermes builds still
        # see newly-launched models without a CLI release.
        union = (
            union_with_portal_free_recommendations
            if free_tier
            else union_with_portal_paid_recommendations
        )
        model_ids, pricing = union(model_ids, pricing, _portal_for_recs)
        _before_policy = model_ids
        model_ids = restrict_to_nous_policy(
            model_ids, _policy_allowed, rescue_empty=True,
        )
        _policy_narrowed = model_ids != _before_policy
        if free_tier:
            model_ids, unavailable_models = partition_nous_models_by_tier(
                model_ids, pricing, free_tier=True,
            )
    _portal = auth_state.get("portal_base_url", "")
    if model_ids:
        from hermes_cli.nous_account import nous_policy_notice

        _policy_notice = nous_policy_notice(removed=_policy_narrowed)
        if _policy_notice:
            print(_policy_notice)
        print(f"Showing {len(model_ids)} curated models — use \"Enter custom model name\" for others.")
        return _prompt_model_selection(
            model_ids, pricing=pricing,
            unavailable_models=unavailable_models,
            portal_url=_portal,
            unavailable_message=unavailable_message,
            confirm_provider="nous",
            confirm_base_url=inference_base_url,
            confirm_api_key=runtime_key,
        )
    elif unavailable_models:
        _url = (_portal or DEFAULT_NOUS_PORTAL_URL).rstrip("/")
        print("No free models currently available.")
        print(unavailable_message or f"Upgrade at {_url} to access paid models.")
    else:
        print("No curated models available for Nous Portal.")
    return None


def _offer_shared_nous_import(timeout_seconds: float) -> Optional[Dict[str, Any]]:
    """Codex-style auto-import: offer to rehydrate a Nous credential from another profile.

    Checks the shared store before launching a fresh device-code flow. Returns the refreshed
    auth state when the user accepted and the import succeeded, else None.
    """
    shared = _read_shared_nous_state()
    if not shared:
        return None
    try:
        shared_path = _nous_shared_store_path()
    except RuntimeError:
        shared_path = None
    print()
    if shared_path:
        print(f"Found existing Nous OAuth credentials at {shared_path}")
    else:
        print("Found existing shared Nous OAuth credentials")
    if not _prompt_yes_no("Import these credentials? [Y/n]: ", default="y"):
        return None
    print("Rehydrating Nous session from shared credentials...")
    auth_state = _try_import_shared_nous_state(timeout_seconds=timeout_seconds)
    if auth_state is None:
        print("Could not refresh shared credentials — falling back to device-code login.")
    return auth_state


def _login_nous(args, pconfig: ProviderConfig) -> None:
    """Nous Portal device authorization flow."""
    timeout_seconds = getattr(args, "timeout", None) or 15.0
    insecure = bool(getattr(args, "insecure", False))
    ca_bundle = (
        getattr(args, "ca_bundle", None)
        or os.getenv("HERMES_CA_BUNDLE")
        or os.getenv("SSL_CERT_FILE")
    )

    try:
        auth_state = _offer_shared_nous_import(timeout_seconds)
        if auth_state is None:
            auth_state = _nous_device_code_login(
                portal_base_url=getattr(args, "portal_url", None),
                inference_base_url=getattr(args, "inference_url", None),
                client_id=getattr(args, "client_id", None) or pconfig.client_id,
                scope=getattr(args, "scope", None),
                open_browser=not getattr(args, "no_browser", False),
                timeout_seconds=timeout_seconds,
                insecure=insecure,
                ca_bundle=ca_bundle,
            )

        inference_base_url = auth_state["inference_base_url"]

        # Snapshot the prior active_provider BEFORE _save_provider_state
        # overwrites it to "nous".  If the user picks "Skip (keep current)"
        # during model selection below, we restore this so the user's previous
        # provider (e.g. openrouter) is preserved.
        with _auth_store_lock():
            _prior_store = _load_auth_store()
            prior_active_provider = _prior_store.get("active_provider")

        saved_to = _save_active_provider_state("nous", auth_state)

        # Mirror to the shared store so other profiles can one-tap import
        # these credentials. Best-effort: any I/O failure is logged and
        # swallowed inside the helper.
        _write_shared_nous_state(auth_state)
        _sync_nous_pool_from_auth_store()

        print()
        print("Login successful!")
        print(f"  Auth state: {saved_to}")

        # Resolve model BEFORE writing provider to config.yaml so we never
        # leave the config in a half-updated state (provider=nous but model
        # still set to the previous provider's model, e.g. opus from
        # OpenRouter).  The auth.json active_provider was already set above.
        selected_model = None
        try:
            selected_model = _pick_nous_model_after_login(auth_state, inference_base_url)
        except Exception as exc:
            message = format_auth_error(exc) if isinstance(exc, AuthError) else str(exc)
            print()
            print(f"Login succeeded, but could not fetch available models. Reason: {message}")

        # Write provider + model atomically so config is never mismatched.
        # If no model was selected (user picked "Skip (keep current)",
        # model list fetch failed, or no curated models were available),
        # preserve the user's previous provider — don't silently switch
        # them to Nous with a mismatched model.  The Nous OAuth tokens
        # stay saved for future use.
        if not selected_model:
            # Restore the prior active_provider that _save_provider_state
            # overwrote to "nous".  config.yaml model.provider is left
            # untouched, so the user's previous provider is fully preserved.
            with _auth_store_lock():
                auth_store = _load_auth_store()
                if prior_active_provider:
                    auth_store["active_provider"] = prior_active_provider
                else:
                    auth_store.pop("active_provider", None)
                _save_auth_store(auth_store)
            print()
            print("No provider change. Nous credentials saved for future use.")
            print("  Run `hermes model` again to switch to Nous Portal.")
            return

        config_path = _update_config_for_provider(
            "nous", inference_base_url, default_model=selected_model,
        )
        if selected_model:
            _save_model_choice(selected_model)
            print(f"Default model set to: {selected_model}")
        print(f"  Config updated: {config_path} (model.provider=nous)")

    except KeyboardInterrupt:
        print("\nLogin cancelled.")
        raise SystemExit(130)
    except Exception as exc:
        print(f"Login failed: {exc}")
        raise SystemExit(1)


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
