"""Shared runtime provider resolution for CLI, gateway, cron, and helpers."""

from __future__ import annotations

import logging
import os
import re
from urllib.parse import urlparse
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

from hermes_cli import auth as auth_mod
from agent.credential_pool import (
    CredentialPool,
    PooledCredential,
    credential_pool_matches_provider,
    custom_provider_pool_key_candidates,
    load_pool,
)
from agent.secret_scope import get_secret as _get_secret
from hermes_cli.auth import (
    ACTUAL_LOCAL_NOAUTH_PLACEHOLDER,
    AuthError,
    DEFAULT_CODEX_BASE_URL,
    DEFAULT_QWEN_BASE_URL,
    DEFAULT_XAI_OAUTH_BASE_URL,
    PROVIDER_REGISTRY,
    _agent_key_is_usable,
    _nous_inference_env_override,
    format_auth_error,
    resolve_provider,
    resolve_nous_runtime_credentials,
    resolve_codex_runtime_credentials,
    resolve_xai_oauth_runtime_credentials,
    resolve_qwen_runtime_credentials,
    resolve_api_key_provider_credentials,
    resolve_external_process_provider_credentials,
    has_usable_secret,
    is_actual_local_base_url,
    normalize_actual_base_url,
)
from hermes_cli import config as _config_mod
from hermes_cli.providers import custom_provider_aliases, custom_provider_slug
from hermes_constants import OPENROUTER_BASE_URL
from hermes_cli.providers import is_official_openai_host


def load_config():
    """Late-bound delegate to :func:`hermes_cli.config.load_config`.

    Deliberately NOT a module-level ``from hermes_cli.config import load_config``: this module is
    often imported lazily (inside functions), so its first import can happen while a test has
    ``hermes_cli.config.load_config`` patched — a from-import would then bind the MagicMock
    *permanently*, poisoning every later caller in the process (the mock's fixed config shadows the
    real one long after the patch exits).
    """
    return _config_mod.load_config()


def get_compatible_custom_providers(config=None):
    """Late-bound delegate — see :func:`load_config` for why."""
    return _config_mod.get_compatible_custom_providers(config)


def normalize_extra_headers(value):
    """Late-bound delegate — see :func:`load_config` for why."""
    return _config_mod.normalize_extra_headers(value)
from utils import base_url_host_matches, base_url_hostname, env_int


def _getenv(name: str, default: str = "") -> str:
    """Profile-scoped replacement for ``os.getenv`` on credential/provider reads.

    Routes through the secret scope: identical to ``os.getenv`` when multiplexing is off, scope-
    aware (and fail-closed on an unscoped read) when on. Genuinely-global vars are handled
    inside ``get_secret`` and still read ``os.environ``. Keeps the ``(name, default) -> str``
    contract.
    """
    val = _get_secret(name, default)
    return val if val is not None else default


def _normalize_custom_provider_name(value: str) -> str:
    return value.strip().lower().replace(" ", "-")


def _loopback_hostname(host: str) -> bool:
    h = (host or "").lower().rstrip(".")
    return h in {"localhost", "127.0.0.1", "::1", "0.0.0.0"}


def _config_base_url_trustworthy_for_bare_custom(cfg_base_url: str, cfg_provider: str) -> bool:
    """Decide whether ``model.base_url`` may back bare ``custom`` runtime resolution.

    The model picker can select Custom while ``model.provider`` still reflects a previous
    provider. Non-loopback URLs are rejected unless the YAML provider is already ``custom`` (or
    a local-server alias like ollama/vllm/llamacpp), so a stale OpenRouter/Z.ai base_url cannot
    hijack local ``custom`` sessions.
    """
    cfg_provider_norm = (cfg_provider or "").strip().lower()
    bu = (cfg_base_url or "").strip()
    if not bu:
        return False
    if cfg_provider_norm == "custom":
        return True
    # Aliases resolving to "custom" (ollama, vllm, llamacpp, …) are trusted like "custom",
    # else a legit LAN/WireGuard ollama endpoint silently falls through to OpenRouter (#27132).
    if _resolves_to_custom(cfg_provider_norm):
        return True
    if base_url_host_matches(bu, "openrouter.ai"):
        return False
    return _loopback_hostname(base_url_hostname(bu))


# Hosts that only speak one wire protocol. Mirrors host_mandated_api_mode in
# hermes_cli/providers.py so the runtime resolver stays in lockstep.
#  - api.meta.ai: prompt caching only on Responses API (0% on chat/completions).
#  - api.router.com (Ramp Router): /v1/chat/completions is a minimal shim;
#    reasoning and caching live on /v1/responses.
#  - api.anthropic.com: native Messages API (realigns with providers.determine_api_mode).
_HOST_MANDATED_API_MODES = {
    "api.x.ai": "codex_responses",
    "api.meta.ai": "codex_responses",
    "api.actual.inc": "codex_responses",
    "api.router.com": "codex_responses",
    "api.anthropic.com": "anthropic_messages",
}


def _detect_api_mode_for_url(base_url: str) -> Optional[str]:
    """Auto-detect api_mode from the resolved base URL.

    - Direct api.openai.com endpoints need the Responses API for GPT-5.x tool calls with reasoning
    (chat/completions returns 400). - Direct api.anthropic.com endpoints must use the native
    Messages API (``/v1/messages``).
    """
    normalized = (base_url or "").strip().lower().rstrip("/")
    hostname = base_url_hostname(base_url)
    # Exact-hostname matches reject lookalike subdomains (api.anthropic.com.attacker.test)
    # and path-segment spoofing (proxy.test/api.anthropic.com/v1) (#32243).
    mandated = _HOST_MANDATED_API_MODES.get(hostname)
    if mandated:
        return mandated
    # Official OpenAI host family: canonical api.openai.com plus the
    # data-residency regional hosts (us./eu.api.openai.com). Same API
    # surface, same Responses-API mandate. Shared predicate — see
    # providers.is_official_openai_host for the spoof-rejection contract.
    if is_official_openai_host(base_url):
        return "codex_responses"
    path = urlparse(normalized).path.rstrip("/")
    if path.endswith("/anthropic") or path.endswith("/anthropic/v1"):
        return "anthropic_messages"
    if hostname == "api.kimi.com" and "/coding" in normalized:
        return "anthropic_messages"
    return None


def _fallback_api_mode(provider: str, base_url: str, model: str = "") -> str:
    """Resolve api_mode when no explicit/persisted mode applies.

    Precedence: URL detection (host-mandated wire shapes) first, then the transport the provider
    overlay itself declares via ``providers.determine_api_mode`` — which already handles host
    mandates, dual-wire providers, and the registry transport map — and only then the
    ``chat_completions`` default for genuinely unknown providers/endpoints.

    That is how ``openai-api`` pointed at OpenAI's data-residency hosts (``us.api.openai.com``)
    400'd on every tool-calling turn: the provider declares ``codex_responses`` but the declaration
    was never consulted. Same latent class covered the other non-chat overlays (MiniMax family,
    copilot-acp).
    """
    detected = _detect_api_mode_for_url(base_url)
    if detected:
        return detected
    from hermes_cli.providers import determine_api_mode

    return determine_api_mode(provider, base_url, model) or "chat_completions"


def _resolve_plain_custom_api_mode(model_cfg: Dict[str, Any], base_url: str) -> str:
    """Resolve api_mode for legacy/plain ``provider: custom`` endpoints.

    Custom endpoints should stay conservative by default. Only direct OpenAI/xAI URLs imply
    Responses API automatically; named custom providers can opt in via their own ``api_mode`` field.
    """
    configured_mode = _parse_api_mode(model_cfg.get("api_mode"))
    # Note: api.meta.ai is handled by _detect_api_mode_for_url (returns codex_responses), so the suppression guard below does not fire for Meta.
    detected_mode = _detect_api_mode_for_url(base_url)

    if configured_mode == "codex_responses" and detected_mode != "codex_responses":
        logger.info(
            "Ignoring persisted custom api_mode=codex_responses for non-OpenAI endpoint %s",
            base_url or "(unknown)",
        )
        configured_mode = None

    return configured_mode or detected_mode or "chat_completions"


def _host_derived_api_key(base_url: str) -> str:
    """Look up `<VENDOR>_API_KEY` in the env, derived from the base URL host.

    Returns the env value (stripped) or "". Never returns env vars whose names are already
    explicitly checked elsewhere — those are handled by their own host-gated paths
    (OPENAI/OPENROUTER/OLLAMA).

    The vendor label is the *registrable* portion of the hostname: strip ``api.`` / ``www.``
    prefixes, then take the second-to-last label (``api.deepseek.com`` → ``deepseek``). Falls back
    to "" for hostnames that don't yield a usable vendor label (IPs, loopback, single-label hosts).
    """
    hostname = base_url_hostname(base_url)
    if not hostname:
        return ""
    # Reject IPv4 / IPv6 / loopback — no meaningful vendor label.
    if any(ch.isdigit() for ch in hostname.split(".")[-1]):
        # Last label starts with a digit → likely IP. (TLDs are never numeric.)
        return ""
    if hostname in ("localhost",) or ":" in hostname:
        return ""
    labels = [lbl for lbl in hostname.split(".") if lbl]
    # Strip common API/CDN prefixes.
    while labels and labels[0] in ("api", "www"):
        labels.pop(0)
    if len(labels) < 2:
        return ""
    # Registrable (second-to-last) label = "the vendor" (api.groq.com → groq). Lookalike hosts
    # pick the ATTACKER's label (api.deepseek.com.attacker.test → "attacker"), so DEEPSEEK_API_KEY
    # stays put — mirrors how `base_url_host_matches` resists the same attack for explicit hosts.
    vendor = labels[-2]
    # Sanitize to env var charset: A-Z, 0-9, underscore.
    sanitized = "".join(ch if ch.isalnum() else "_" for ch in vendor).upper()
    if not sanitized or not sanitized[0].isalpha():
        return ""
    # Don't re-derive env vars already handled by explicit host-gated paths.
    if sanitized in ("OPENAI", "OPENROUTER", "OLLAMA"):
        return ""
    env_name = f"{sanitized}_API_KEY"
    return (_getenv(env_name, "") or "").strip()


def _anthropic_base_url_override_ok(base_url: str) -> bool:
    """Decide whether a configured ``model.base_url`` may back native Anthropic.

    Native ``provider: anthropic`` resolution honors ``model.base_url`` so users can point at
    Anthropic-compatible endpoints (official Anthropic/Claude hosts, Azure Foundry,
    MiniMax/Zhipu/LiteLLM-style ``/anthropic`` proxies, Kimi's ``/coding`` route). But a config can
    carry a *stale* non-Anthropic URL — e.g.

    Returns True only when the URL plausibly speaks the Anthropic Messages protocol; otherwise the
    caller falls back to ``https://api.anthropic.com``.
    """
    candidate = (base_url or "").strip()
    if not candidate:
        return False

    hostname = (base_url_hostname(candidate) or "").lower()
    if not hostname:
        return False

    # Official Anthropic / Claude hosts.
    if hostname == "api.anthropic.com" or hostname.endswith(".anthropic.com") or hostname.endswith(".claude.com"):
        return True
    # Azure Foundry Anthropic endpoints (handled specially downstream).
    if hostname.endswith(".azure.com"):
        return True
    # Anthropic-compatible proxies conventionally expose the native Messages
    # protocol under a ``/anthropic`` suffix, and Kimi under ``/coding`` — same
    # signal _detect_api_mode_for_url() uses to pick anthropic_messages. Bare
    # api.kimi.com without the /coding path is not an Anthropic endpoint.
    return _detect_api_mode_for_url(candidate) == "anthropic_messages"


_ANTHROPIC_DEFAULT_BASE_URL = "https://api.anthropic.com"
_NO_ANTHROPIC_CREDENTIALS_MSG = (
    "No Anthropic credentials found. Set ANTHROPIC_TOKEN or ANTHROPIC_API_KEY, "
    "run 'claude setup-token', or authenticate with 'claude /login'."
)


def _runtime(provider: str, api_mode: str, base_url: Any, api_key: Any, **extra: Any) -> Dict[str, Any]:
    """Build a resolved-runtime dict; ``extra`` carries source/requested_provider/provider-specific keys."""
    return {"provider": provider, "api_mode": api_mode, "base_url": base_url, "api_key": api_key, **extra}


def _cfg_provider(model_cfg: Dict[str, Any]) -> str:
    return str(model_cfg.get("provider") or "").strip().lower()


def _config_base_url_for_provider(model_cfg: Dict[str, Any], provider: str) -> str:
    """``model.base_url`` (stripped, no trailing slash) only when ``model.provider`` is ``provider``.

    Prevents a stale base_url from a previously selected provider leaking into another one.
    """
    if _cfg_provider(model_cfg) != provider:
        return ""
    return str(model_cfg.get("base_url") or "").strip().rstrip("/")


def _anthropic_cfg_base_url(model_cfg: Dict[str, Any]) -> str:
    """Config base_url for native Anthropic, or "" when absent/untrustworthy."""
    cfg_base_url = _config_base_url_for_provider(model_cfg, "anthropic")
    return cfg_base_url if _anthropic_base_url_override_ok(cfg_base_url) else ""


def _resolves_to_custom(name: str) -> bool:
    """True when a provider alias (ollama, vllm, llamacpp, …) resolves to ``custom``."""
    try:
        return auth_mod.resolve_provider(name) == "custom"
    except Exception:
        return False


def _host_gated_env_key_candidates(base_url: str, *, ollama: bool) -> list:
    """Env API keys gated on their authoritative hosts, then the host-derived ``<VENDOR>_API_KEY``.

    Sending OPENAI_API_KEY / OPENROUTER_API_KEY / OLLAMA_API_KEY to an unrelated endpoint leaks
    credentials (#28660, GHSA-76xc-57q6-vm5m); match on HOST, not substring. ``_host_derived_api_key``
    deliberately skips OLLAMA, so callers that want it opt in via ``ollama``.
    """
    is_openai = base_url_host_matches(base_url, "openai.com") or base_url_host_matches(base_url, "openai.azure.com")
    candidates = []
    if ollama:
        candidates.append(_getenv("OLLAMA_API_KEY", "").strip() if base_url_host_matches(base_url, "ollama.com") else "")
    candidates += [
        _getenv("OPENAI_API_KEY", "").strip() if is_openai else "",
        _getenv("OPENROUTER_API_KEY", "").strip() if base_url_host_matches(base_url, "openrouter.ai") else "",
        _host_derived_api_key(base_url),
    ]
    return candidates


def _pool_entry_api_key(entry: Any) -> str:
    return getattr(entry, "runtime_api_key", None) or getattr(entry, "access_token", "")


def _pool_entry_base_url(entry: Any) -> str:
    return getattr(entry, "runtime_base_url", None) or getattr(entry, "base_url", None) or ""


def _nous_pool_state(entry: Any) -> Dict[str, Any]:
    return {
        "agent_key": getattr(entry, "agent_key", None),
        "agent_key_expires_at": getattr(entry, "agent_key_expires_at", None),
        "scope": getattr(entry, "scope", None),
    }


def _registry_base_url(provider: str) -> str:
    pconfig = PROVIDER_REGISTRY.get(provider)
    return pconfig.inference_base_url if pconfig else ""


def _azure_inferred_api_mode(effective_model: str, api_mode: str) -> str:
    """Upgrade api_mode for GPT-5.x / codex / o1-o4 deployments on Azure Foundry.

    Azure rejects /chat/completions on these with 400 "operation unsupported" (see
    ``azure_foundry_model_api_mode``). Skipped when the user explicitly picked anthropic_messages.
    """
    if not effective_model or api_mode == "anthropic_messages":
        return api_mode
    try:
        from hermes_cli.models import azure_foundry_model_api_mode

        inferred = azure_foundry_model_api_mode(effective_model)
    except Exception:
        inferred = None
    return inferred or api_mode


def _configured_or_fallback_api_mode(
    provider: str,
    model_cfg: Dict[str, Any],
    base_url: str,
    effective_model: Any,
    *,
    opencode_by_model: bool,
) -> str:
    """Persisted ``model.api_mode`` when it belongs to this provider, else URL/transport fallback.

    OpenCode Zen/Go serve both anthropic_messages and chat_completions models, so (when
    ``opencode_by_model``) their mode is always re-derived from the effective model rather than
    the stale persisted api_mode (#16878).
    """
    if opencode_by_model:
        from hermes_cli.models import opencode_provider_family

        if opencode_provider_family(provider) is not None:
            from hermes_cli.models import opencode_model_api_mode

            return opencode_model_api_mode(provider, effective_model)
    configured_mode = _parse_api_mode(model_cfg.get("api_mode"))
    if configured_mode and _provider_supports_explicit_api_mode(provider, _cfg_provider(model_cfg)):
        return configured_mode
    # URL detection first (Anthropic /anthropic suffix, Kimi /coding, official
    # OpenAI hosts / api.x.ai → codex_responses), then the provider's declared transport.
    return _fallback_api_mode(provider, base_url, effective_model)


def _api_key_provider_api_mode(
    provider: str,
    model_cfg: Dict[str, Any],
    api_key: str,
    base_url: str,
    effective_model: Any,
    *,
    opencode_by_model: bool,
) -> str:
    """api_mode for a registry ``api_key`` provider (explicit and env/config paths)."""
    if provider == "copilot":
        return _copilot_runtime_api_mode(model_cfg, api_key, target_model=effective_model)
    if provider in ("xai", "actual"):
        return "codex_responses"
    return _configured_or_fallback_api_mode(
        provider, model_cfg, base_url, effective_model, opencode_by_model=opencode_by_model
    )


def _normalize_opencode_runtime_base_url(provider: str, api_mode: str, base_url: str) -> str:
    """OpenCode base URLs end with /v1 for OpenAI-compatible models, but the Anthropic SDK
    prepends its own /v1/messages: strip /v1 for anthropic_messages, re-append otherwise."""
    from hermes_cli.models import opencode_provider_family

    if opencode_provider_family(provider) is None:
        return base_url
    from hermes_cli.models import normalize_opencode_base_url

    return normalize_opencode_base_url(provider, api_mode, base_url)


def _auto_detect_local_model(base_url: str) -> str:
    """Query a local server for its model name when only one model is loaded."""
    if not base_url:
        return ""
    try:
        import requests
        url = base_url.rstrip("/")
        if not url.endswith("/v1"):
            url += "/v1"
        resp = requests.get(url + "/models", timeout=(2, 3))
        if resp.ok:
            models = resp.json().get("data", [])
            if len(models) == 1:
                model_id = models[0].get("id", "")
                if model_id:
                    return model_id
    except Exception as exc:
        # Log instead of silently swallowing — aids debugging when
        # local model auto-detection fails unexpectedly.
        logger.debug("Auto-detect model from %s failed: %s", base_url, exc)
    return ""


def _get_model_config() -> Dict[str, Any]:
    config = load_config()
    model_cfg = config.get("model")
    if isinstance(model_cfg, dict):
        cfg = dict(model_cfg)
        # Accept "model" as alias for "default" (users intuitively write model.model)
        if not cfg.get("default") and cfg.get("model"):
            cfg["default"] = cfg["model"]
        # Handle model.default being a dict {provider: ..., model: ...} rather than a string
        _default = cfg.get("default")
        if isinstance(_default, dict):
            from hermes_cli.config import split_model_config_default
            cfg_model, cfg_provider = split_model_config_default(_default)
            cfg_provider = cfg_provider or str(model_cfg.get("provider") or "")
            cfg["default"] = cfg_model
            if cfg_provider and not cfg.get("provider"):
                cfg["provider"] = cfg_provider
            _default = cfg_model
        default = (str(_default or "")).strip()
        base_url = (cfg.get("base_url") or "").strip()
        is_local = base_url_hostname(base_url) in ("localhost", "127.0.0.1")
        is_fallback = not default
        if is_local and is_fallback and base_url:
            detected = _auto_detect_local_model(base_url)
            if detected:
                cfg["default"] = detected
        return cfg
    if isinstance(model_cfg, str) and model_cfg.strip():
        return {"default": model_cfg.strip()}
    return {}


def _provider_supports_explicit_api_mode(provider: Optional[str], configured_provider: Optional[str] = None) -> bool:
    """Check whether a persisted api_mode should be honored for a given provider.

    Prevents stale api_mode from a previous provider leaking into a different one after a
    model/provider switch. Only applies the persisted mode when the config's provider matches the
    runtime provider (or when no configured provider is recorded).
    """
    normalized_provider = (provider or "").strip().lower()
    normalized_configured = (configured_provider or "").strip().lower()
    if not normalized_configured:
        return True
    if normalized_provider == "custom":
        return normalized_configured == "custom" or normalized_configured.startswith("custom:")
    return normalized_configured == normalized_provider


def _copilot_runtime_api_mode(
    model_cfg: Dict[str, Any],
    api_key: str,
    *,
    target_model: Optional[str] = None,
) -> str:
    configured_provider = str(model_cfg.get("provider") or "").strip().lower()
    configured_mode = _parse_api_mode(model_cfg.get("api_mode"))
    if configured_mode and _provider_supports_explicit_api_mode("copilot", configured_provider):
        return configured_mode

    # Use the model being resolved, not the persisted default: MoA slots / fallbacks / mid-session
    # switches target a different model, and a Claude slot inheriting codex_responses from a
    # GPT-5 default fails with "model ... does not support Responses API".
    model_name = str(target_model or model_cfg.get("default") or "").strip()
    if not model_name:
        return "chat_completions"

    try:
        from hermes_cli.models import copilot_model_api_mode

        return copilot_model_api_mode(model_name, api_key=api_key)
    except Exception:
        return "chat_completions"


_VALID_API_MODES = {
    "chat_completions",
    "codex_responses",
    "anthropic_messages",
    "bedrock_converse",
    # Opt-in: hand the whole turn to a `codex app-server` subprocess (Codex's own tool runtime).
    # Gated on `model.openai_runtime == "codex_app_server"` AND provider in {openai, openai-codex}.
    "codex_app_server",
}


def _parse_api_mode(raw: Any) -> Optional[str]:
    """Validate an api_mode value from config. Returns None if invalid.

    Legacy/alias spellings (``openai``, ``anthropic``, ``responses``, …) are canonicalized via the
    shared alias map before validation, so configs written against older releases keep selecting the
    transport they named instead of silently falling through to hostname-based detection.
    """
    if isinstance(raw, str):
        from hermes_cli.config import _canonical_api_mode

        normalized = _canonical_api_mode(raw).lower()
        if normalized in _VALID_API_MODES:
            return normalized
    return None


def _nous_inference_base_url_override() -> str:
    """Return the trusted Nous runtime base URL override, if configured.

    Delegates to ``auth._nous_inference_env_override`` so every ``NOUS_INFERENCE_BASE_URL`` read
    shares one normalization path. The env source is trusted and intentionally bypasses the
    network host allowlist.
    """
    return _nous_inference_env_override() or ""


def _maybe_apply_codex_app_server_runtime(
    *,
    provider: str,
    api_mode: str,
    model_cfg: Optional[Dict[str, Any]],
) -> str:
    """Opt-in rewrite of api_mode → "codex_app_server" via ``model.openai_runtime`` in config.yaml.

    No-op when ``model.openai_runtime`` is unset, "auto", or empty. Only ``openai`` and
    ``openai-codex`` are eligible — other providers cannot be rerouted through codex. Returns
    the (possibly rewritten) api_mode.
    """
    if not model_cfg:
        return api_mode
    if provider not in {"openai", "openai-codex"}:
        return api_mode
    runtime = str(model_cfg.get("openai_runtime") or "").strip().lower()
    if runtime == "codex_app_server":
        return "codex_app_server"
    return api_mode


# Pool-entry providers whose api_mode is fixed: provider -> (api_mode, default base_url when the
# pool entry carries none). Callables are evaluated lazily (registry lookups). MiniMax OAuth tokens
# are valid only against the Anthropic Messages endpoint, so a stale model.api_mode from a prior
# OpenAI-compatible provider is never honoured for it (it would 404 on /chat/completions).
_POOL_ENTRY_SIMPLE_MODES: Dict[str, tuple] = {
    "openai-codex": ("codex_responses", DEFAULT_CODEX_BASE_URL),
    "xai-oauth": ("codex_responses", DEFAULT_XAI_OAUTH_BASE_URL),
    "qwen-oauth": ("chat_completions", DEFAULT_QWEN_BASE_URL),
    "minimax-oauth": ("anthropic_messages", lambda: _registry_base_url("minimax-oauth")),
    "openrouter": ("chat_completions", OPENROUTER_BASE_URL),
    "xai": ("codex_responses", ""),
}


def _resolve_runtime_from_pool_entry(
    *,
    provider: str,
    entry: PooledCredential,
    requested_provider: str,
    model_cfg: Optional[Dict[str, Any]] = None,
    pool: Optional[CredentialPool] = None,
    target_model: Optional[str] = None,
) -> Dict[str, Any]:
    model_cfg = model_cfg or _get_model_config()
    # Prefer the caller's target model (e.g. /model switch) over the persisted default, else
    # api_mode is computed from a stale default (opencode-zen /v1 stripped while config.default
    # was still a Claude model).
    effective_model = (target_model or model_cfg.get("default") or "")
    base_url = _pool_entry_base_url(entry).rstrip("/")
    api_key = _pool_entry_api_key(entry)
    if provider in _POOL_ENTRY_SIMPLE_MODES:
        api_mode, default_url = _POOL_ENTRY_SIMPLE_MODES[provider]
        base_url = base_url or (default_url() if callable(default_url) else default_url)
    elif provider == "anthropic":
        api_mode = "anthropic_messages"
        base_url = _anthropic_cfg_base_url(model_cfg) or base_url or _ANTHROPIC_DEFAULT_BASE_URL
    elif provider == "nous":
        api_mode = _nous_api_mode(effective_model)
        base_url = _nous_inference_base_url_override() or base_url
    elif provider == "copilot":
        api_mode = _copilot_runtime_api_mode(
            model_cfg,
            getattr(entry, "runtime_api_key", ""),
            target_model=effective_model,
        )
        base_url = base_url or PROVIDER_REGISTRY["copilot"].inference_base_url
    elif provider == "azure-foundry":
        # Azure Foundry: read api_mode and base_url from config
        api_mode = "chat_completions"
        if _cfg_provider(model_cfg) == "azure-foundry":
            base_url = _config_base_url_for_provider(model_cfg, "azure-foundry") or base_url
            api_mode = _parse_api_mode(model_cfg.get("api_mode")) or api_mode
        api_mode = _azure_inferred_api_mode(effective_model, api_mode)
        # For Anthropic-style endpoints, strip /v1 suffix
        if api_mode == "anthropic_messages":
            base_url = re.sub(r"/v1/?$", "", base_url)
    else:
        # Honour model.base_url only when the pool entry carries no explicit base_url (i.e. it
        # fell back to the registry default). Env var overrides win (#6039).
        pconfig = PROVIDER_REGISTRY.get(provider)
        pool_url_is_default = pconfig and base_url.rstrip("/") == pconfig.inference_base_url.rstrip("/")
        if pool_url_is_default:
            base_url = _config_base_url_for_provider(model_cfg, provider) or base_url
        api_mode = _configured_or_fallback_api_mode(
            provider, model_cfg, base_url, effective_model, opencode_by_model=True
        )

    base_url = _normalize_opencode_runtime_base_url(provider, api_mode, base_url)

    # Optional opt-in: route OpenAI/Codex turns through `codex app-server`.
    # Inert when `model.openai_runtime` is unset or "auto".
    api_mode = _maybe_apply_codex_app_server_runtime(
        provider=provider, api_mode=api_mode, model_cfg=model_cfg
    )

    if provider == "lmstudio":
        base_url = auth_mod._normalize_lmstudio_runtime_base_url(base_url)

    return _runtime(
        provider,
        api_mode,
        base_url,
        api_key,
        source=getattr(entry, "source", "pool"),
        credential_pool=pool,
        requested_provider=requested_provider,
    )


def resolve_requested_provider(requested: Optional[str] = None) -> str:
    """Resolve provider request from explicit arg, config, then env."""
    if requested and requested.strip():
        return requested.strip().lower()

    model_cfg = _get_model_config()
    cfg_provider = model_cfg.get("provider")
    if isinstance(cfg_provider, str) and cfg_provider.strip():
        return cfg_provider.strip().lower()

    # Prefer the persisted config selection over any stale shell/.env
    # provider override so chat uses the endpoint the user last saved.
    env_provider = _getenv("HERMES_INFERENCE_PROVIDER", "").strip().lower()
    if env_provider:
        return env_provider

    return "auto"


def _try_resolve_from_custom_pool(
    base_url: str,
    provider_label: str,
    api_mode_override: Optional[str] = None,
    provider_name: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Check if a credential pool exists for a custom endpoint and return a runtime dict if so."""
    try:
        raw_keys = list(custom_provider_pool_key_candidates(base_url, provider_name))
    except Exception:
        raw_keys = []
    # Order-preserving dedupe of normalized keys.
    candidates = list(dict.fromkeys(k for k in (str(key or "").strip().lower() for key in raw_keys) if k))
    if not candidates:
        return None

    for pool_key in candidates:
        try:
            pool = load_pool(pool_key)
            if not pool.has_credentials():
                continue
            entry = pool.select()
            if entry is None:
                continue
            pool_api_key = _pool_entry_api_key(entry)
            if not pool_api_key:
                continue
            if not has_usable_secret(pool_api_key) and _loopback_hostname(base_url_hostname(base_url)):
                # Legacy configs used short placeholder keys ('123', 'm') for local no-auth
                # services; has_usable_secret's later 4-char floor rejects them. Every other
                # resolution path substitutes "no-key-required" for a loopback endpoint with
                # no usable secret — this pool path was the one gap (#86864).
                pool_api_key = "no-key-required"
            return _runtime(
                provider_label,
                api_mode_override or _detect_api_mode_for_url(base_url) or "chat_completions",
                base_url,
                pool_api_key,
                source=f"pool:{pool_key}",
                credential_pool=pool,
            )
        except Exception:
            continue
    return None


def _filter_capabilities(value: Any) -> Dict[str, bool]:
    """Return the string-keyed boolean capabilities accepted at runtime."""
    if not isinstance(value, dict):
        return {}
    return {
        key: enabled
        for key, enabled in value.items()
        if isinstance(key, str) and isinstance(enabled, bool)
    }


def _lift_model_capabilities(
    entry: Dict[str, Any], model: Optional[str], result: Dict[str, Any]
) -> None:
    """Copy explicit boolean per-model capabilities into the runtime."""
    capabilities = _filter_capabilities(entry.get("capabilities"))
    models = entry.get("models")
    model_config = models.get(model) if isinstance(models, dict) and model else None
    if isinstance(model_config, dict):
        capabilities.update(_filter_capabilities(model_config))
    if capabilities:
        result["capabilities"] = capabilities


def _lift_max_output_tokens(entry: Dict[str, Any], result: Dict[str, Any]) -> None:
    """Propagate a per-provider output cap onto the resolved runtime dict.

    Accepts ``max_output_tokens`` or ``max_tokens`` on a ``custom_providers`` entry so a provider
    block can pin its own output limit. Gateway and CLI map this onto ``AIAgent.max_tokens`` only
    when the top-level ``model.max_tokens`` isn't set, so the documented global key still wins.
    """
    for _k in ("max_output_tokens", "max_tokens"):
        _v = entry.get(_k)
        if isinstance(_v, int) and _v > 0:
            result["max_output_tokens"] = _v
            return


def _lift_extra_headers(entry: Dict[str, Any], result: Dict[str, Any]) -> None:
    """Copy a validated ``extra_headers`` dict from a provider entry.

    SECURITY: header values routinely carry credentials (Cloudflare Access service tokens, proxy
    auth, custom bearer schemes). Never log them.
    """
    extra_headers = normalize_extra_headers(entry.get("extra_headers"))
    if extra_headers:
        result["extra_headers"] = extra_headers


def _lift_common_custom_fields(
    entry: Dict[str, Any],
    result: Dict[str, Any],
    *,
    provider_key: str,
    key_env: str,
    api_mode: Optional[str],
) -> None:
    """Copy the optional fields shared by ``providers:`` and legacy ``custom_providers:`` entries."""
    if key_env:
        result["key_env"] = key_env
    if provider_key:
        result["provider_key"] = provider_key
    extra_body = entry.get("extra_body")
    if isinstance(extra_body, dict):
        result["extra_body"] = dict(extra_body)
    _lift_extra_headers(entry, result)
    if api_mode:
        result["api_mode"] = api_mode
    _lift_max_output_tokens(entry, result)
    capabilities = _filter_capabilities(entry.get("capabilities"))
    if capabilities:
        result["capabilities"] = capabilities


def _get_named_custom_provider(requested_provider: str) -> Optional[Dict[str, Any]]:
    requested_norm = _normalize_custom_provider_name(requested_provider or "")
    if not requested_norm:
        return None

    # Bare "custom" is normally owned by the model.base_url trust path, but a user may literally
    # name a ``providers:`` entry "custom"; returning None before the config scan made such cron
    # jobs fail with ``auth_unavailable: providers=codex``. So fall through to the scan (still
    # None if no entry is named "custom"). Raw names map to custom providers only when they are
    # not canonical built-ins; explicit ``custom:<name>`` keys always target the saved entry, and
    # bare "custom" is exempt from the shadow check.
    if requested_norm == "auto":
        return None
    if requested_norm != "custom" and not requested_norm.startswith("custom:"):
        try:
            canonical = auth_mod.resolve_provider(requested_norm)
        except AuthError:
            pass
        else:
            # Defer to the built-in only when the raw name IS the canonical provider (``nous``);
            # an entry matching merely an alias (``kimi`` → ``kimi-coding``) is the user's target.
            if (canonical or "").strip().lower() == requested_norm:
                return None

    config = load_config()
    
    # First check providers: dict (new-style user-defined providers)
    providers = config.get("providers")
    if isinstance(providers, dict):
        from hermes_cli.config import is_provider_enabled
        for ep_name, entry in providers.items():
            if not isinstance(entry, dict):
                continue
            # ``providers.<name>.enabled: false`` entries stay in config but are invisible here.
            if not is_provider_enabled(entry):
                continue
            # Resolve the API key from the env var name stored in key_env
            key_env = str(
                entry.get("key_env") or entry.get("api_key_env") or ""
            ).strip()
            resolved_api_key = _getenv(key_env, "").strip() if key_env else ""
            # Fall back to inline api_key when key_env is absent or unresolvable
            if not resolved_api_key:
                resolved_api_key = str(entry.get("api_key", "") or "").strip()

            display_name = entry.get("name", "")
            if requested_norm in custom_provider_aliases(
                str(display_name or ep_name),
                str(ep_name),
            ):
                # Found match by provider key
                base_url = entry.get("api") or entry.get("url") or entry.get("base_url") or ""
                if base_url:
                    result: Dict[str, Any] = {
                        "name": entry.get("name", ep_name),
                        "base_url": base_url.strip(),
                        "api_key": resolved_api_key,
                        "model": entry.get("default_model", ""),
                    }
                    # Command that PRINTS a short-lived credential; wrapped in a
                    # per-request token provider at resolution.
                    key_cmd = str(entry.get("key_cmd", "") or "").strip()
                    if key_cmd:
                        result["key_cmd"] = key_cmd
                    # v12 migration writes ``transport``; hand-edited configs may still use
                    # ``api_mode``. Accept both or migrated configs silently downgrade to
                    # chat_completions.
                    _lift_common_custom_fields(
                        entry, result,
                        provider_key=str(ep_name or "").strip(),
                        key_env=key_env,
                        api_mode=_parse_api_mode(entry.get("api_mode") or entry.get("transport")),
                    )
                    return result

    # Fall back to custom_providers: list (legacy format)
    custom_providers = config.get("custom_providers")
    if isinstance(custom_providers, dict):
        logger.warning(
            "custom_providers in config.yaml is a dict, not a list. "
            "Each entry must be prefixed with '-' in YAML. "
            "Run 'hermes doctor' for details."
        )
        return None

    custom_providers = get_compatible_custom_providers(config)
    if not custom_providers:
        return None

    for entry in custom_providers:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        base_url = entry.get("base_url")
        if not isinstance(name, str) or not isinstance(base_url, str):
            continue
        provider_key = str(entry.get("provider_key", "") or "").strip()
        if requested_norm not in custom_provider_aliases(name, provider_key):
            continue
        result = {
            "name": name.strip(),
            "base_url": base_url.strip(),
            "api_key": str(entry.get("api_key", "") or "").strip(),
        }
        model_name = str(entry.get("model", "") or "").strip()
        if model_name:
            result["model"] = model_name
        _lift_common_custom_fields(
            entry, result,
            provider_key=provider_key,
            key_env=str(entry.get("key_env", "") or "").strip(),
            api_mode=_parse_api_mode(entry.get("api_mode")),
        )
        return result

    return None


def has_named_custom_provider(requested_provider: str) -> bool:
    """Return True when config defines a custom provider matching the request.

    Public wrapper around :func:`_get_named_custom_provider` so other modules (e.g. the cronjob
    tool) can check whether a provider name resolves to a configured ``providers:`` /
    ``custom_providers:`` entry without reaching into a private helper.
    """
    try:
        return _get_named_custom_provider(requested_provider) is not None
    except Exception:
        return False


def _find_custom_identity(matches) -> Optional[str]:
    """Scan ``providers:`` then legacy ``custom_providers:`` for the first entry where
    ``matches(entry)`` holds and return its canonical ``custom:<name>`` slug."""
    try:
        config = load_config()
    except Exception:
        return None

    providers = config.get("providers")
    if isinstance(providers, dict):
        for ep_name, entry in providers.items():
            if isinstance(entry, dict) and matches(entry):
                return custom_provider_slug(str(ep_name), str(ep_name))

    try:
        custom_providers = get_compatible_custom_providers(config)
    except Exception:
        custom_providers = None
    for entry in custom_providers or []:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        if matches(entry):
            return custom_provider_slug(name, str(entry.get("provider_key", "") or ""))

    return None


def find_custom_provider_identity(base_url: str) -> Optional[str]:
    """Map an endpoint URL back to its canonical ``custom:<name>`` menu key.

    Session persistence stores the agent's *resolved* provider, and for every named custom endpoint
    that is the literal string ``"custom"`` — the entry name is lost, and the api_key is
    deliberately never persisted.
    """
    target = _normalize_base_url_for_match(base_url)
    if not target:
        return None

    def _entry_owns_url(entry: Dict[str, Any]) -> bool:
        entry_url = entry.get("api") or entry.get("url") or entry.get("base_url") or ""
        return _normalize_base_url_for_match(entry_url) == target

    return _find_custom_identity(_entry_owns_url)


def find_custom_provider_identity_by_model(model: str) -> Optional[str]:
    """Map a model id back to the ``custom:<name>`` entry that serves it.

    Companion to :func:`find_custom_provider_identity` (URL reverse-lookup) for the persistence
    paths where no base_url survived the round-trip: the session row always stores the model name,
    and a custom endpoint's model ids (e.g.
    """
    target = str(model or "").strip().lower()
    if not target:
        return None

    def _entry_serves_model(entry: Dict[str, Any]) -> bool:
        for key in ("model", "default_model"):
            value = entry.get(key)
            if isinstance(value, str) and value.strip().lower() == target:
                return True
        models = entry.get("models")
        if isinstance(models, dict):
            return any(str(mid).strip().lower() == target for mid in models)
        if isinstance(models, list):
            for item in models:
                if isinstance(item, str) and item.strip().lower() == target:
                    return True
                if isinstance(item, dict):
                    mid = item.get("id") or item.get("name")
                    if isinstance(mid, str) and mid.strip().lower() == target:
                        return True
        return False

    return _find_custom_identity(_entry_serves_model)


def canonical_custom_identity(
    *,
    base_url: Optional[str] = None,
    config_provider: Optional[str] = None,
    model: Optional[str] = None,
) -> Optional[str]:
    """Recover a routable ``custom:<name>`` identity for a bare custom provider.

    Any code path that persists or restores a session's provider override must run the resolved
    provider through this helper so a bare ``"custom"`` is upgraded back to its durable
    ``custom:<name>`` menu key. Three recovery sources, in priority order:

    1. ``base_url`` — reverse-lookup the entry that owns the endpoint URL (the one fact that always
    survives the persistence round-trip when a URL was recorded). 2. ``model`` — reverse-lookup the
    entry that serves the session's model (``model``/``default_model``/``models`` catalog). 3.
    """
    # 1. Reverse-lookup by endpoint URL.
    if base_url:
        identity = find_custom_provider_identity(base_url)
        if identity:
            return identity

    # 2. Reverse-lookup by the session's model name.
    if model:
        identity = find_custom_provider_identity_by_model(model)
        if identity:
            return identity

    # 3. Fall back to the configured provider when it names a real entry.
    candidate = str(config_provider or "").strip()
    if not candidate:
        try:
            candidate = str(_get_model_config().get("provider") or "").strip()
        except Exception:
            candidate = ""
    if not candidate:
        candidate = os.environ.get("HERMES_INFERENCE_PROVIDER", "").strip()

    candidate_norm = _normalize_custom_provider_name(candidate)
    # A bare/non-routable candidate cannot heal a bare custom override.
    if not candidate_norm or candidate_norm in {"custom", "auto", "openrouter"}:
        return None
    # Only return it when it actually resolves to a configured custom entry,
    # so we never invent a `custom:<x>` that resolution can't honor.
    try:
        entry = _get_named_custom_provider(candidate)
        if entry is not None:
            # ``candidate`` may be the entry's DISPLAY NAME, which is not the durable identity of
            # a keyed ``providers:`` entry — re-resolve via its endpoint so every path returns the
            # same config-key slug (else it heals to ``custom:<display-name>`` and stops matching).
            identity = find_custom_provider_identity(str(entry.get("base_url") or ""))
            if identity:
                return identity
            if candidate_norm.startswith("custom:"):
                return candidate_norm
            return f"custom:{candidate_norm}"
    except Exception:
        pass
    return None


def is_routable_provider(provider: Optional[str]) -> bool:
    """Whether a provider name currently resolves to a routable route.

    Empty/None is vacuously routable: agent build falls back to the configured default instead of
    failing. A name that resolves through the full chain (built-in -> user ``providers:`` ->
    ``custom_providers:`` -> models.dev) is routable; anything else would fail agent init with
    "Unknown provider '<name>'".
    """
    name = str(provider or "").strip()
    if not name or name.lower() == "auto":
        return True
    if name.lower() == "custom":
        # The bare string is the resolved billing class shared by every
        # named custom entry — not a routable identity. restore paths must
        # heal it (canonical_custom_identity) or fall back, never hand it
        # straight to agent init.
        return False
    try:
        from hermes_cli.providers import resolve_provider_full

        config = load_config()
        return (
            resolve_provider_full(
                name,
                config.get("providers"),
                get_compatible_custom_providers(config),
            )
            is not None
        )
    except Exception:
        return False


def _normalize_base_url_for_match(value) -> str:
    return str(value or "").strip().rstrip("/").lower()


def _custom_provider_request_overrides(custom_provider: Dict[str, Any]) -> Dict[str, Any]:
    extra_body = custom_provider.get("extra_body")
    if not isinstance(extra_body, dict) or not extra_body:
        return {}
    return {"extra_body": dict(extra_body)}


def _apply_custom_provider_extras(
    custom_provider: Dict[str, Any], target_model: Optional[str], result: Dict[str, Any]
) -> None:
    """Copy model / capabilities / max_output_tokens / extra_headers / request_overrides onto a
    resolved custom runtime.

    An explicit ``target_model`` wins over the provider's configured default (auxiliary slots /
    background-review resolve a concrete model and must not fall back to ``default_model``).
    ``extra_headers`` may carry credentials — NEVER log them.
    """
    model_name = target_model or custom_provider.get("model")
    if model_name:
        result["model"] = model_name
    _lift_model_capabilities(custom_provider, model_name, result)
    if isinstance(custom_provider.get("max_output_tokens"), int):
        result["max_output_tokens"] = custom_provider["max_output_tokens"]
    if custom_provider.get("extra_headers"):
        result["extra_headers"] = dict(custom_provider["extra_headers"])
    request_overrides = _custom_provider_request_overrides(custom_provider)
    if request_overrides:
        result["request_overrides"] = {
            **dict(result.get("request_overrides") or {}),
            **request_overrides,
        }


def _resolve_named_custom_runtime(
    *,
    requested_provider: str,
    explicit_api_key: Optional[str] = None,
    explicit_base_url: Optional[str] = None,
    target_model: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    # Bare `provider="custom"` with an explicit base_url (e.g. from a `model_aliases:` direct
    # alias) builds a runtime directly so the alias's base_url takes effect. Aliases resolving to
    # "custom" (ollama, vllm, llamacpp, …) are treated identically (#27132).
    requested_norm = (requested_provider or "").strip().lower()

    # Managed llama.cpp runtime: a llamacpp alias with no explicit base_url resolves to the
    # supervised (or detected external) server first. Explicit base_url always wins.
    if requested_norm in ("llamacpp", "llama.cpp", "llama-cpp") and not explicit_base_url:
        try:
            from hermes_cli.local_runtime.endpoint import resolve_llamacpp_endpoint

            endpoint = resolve_llamacpp_endpoint()
        except Exception:  # noqa: BLE001 — resolution is best-effort
            endpoint = None
        if endpoint:
            return _runtime(
                "custom",
                "chat_completions",
                endpoint["base_url"],
                (explicit_api_key or "").strip()
                or endpoint["api_key"] or "no-key-required",
                source="local-runtime",
                requested_provider=requested_provider,
            )
        # No server: say so and stop — falling through to the generic custom path surfaces
        # "local server is off" as OpenRouter's baffling "401 Invalid API key". The switch's
        # state picks the message (server off → point at the switch; else the setup pane).
        try:
            _lr_enabled = bool((load_config().get("local_runtime") or {}).get("enabled"))
        except Exception:  # noqa: BLE001
            _lr_enabled = False
        if _lr_enabled:
            raise ValueError(
                "The local model server isn't running. It may still be "
                "starting — try again in a moment, or check Settings → "
                "Providers → Local models."
            )
        raise ValueError(
            "The local model server is turned off. Turn it back on in "
            "Settings → Providers → Local models, or switch to another "
            "model."
        )

    if requested_norm and requested_norm != "custom" and _resolves_to_custom(requested_norm):
        requested_norm = "custom"
    if requested_norm == "custom" and explicit_base_url:
        base_url = explicit_base_url.strip().rstrip("/")
        # Check credential pool first — mirrors the named-custom-provider path
        # so bare `provider: custom` with a configured custom_providers entry
        # also gets its api_key from the pool instead of env var fallbacks.
        pool_result = _try_resolve_from_custom_pool(base_url, "custom", None)
        if pool_result:
            pool_result["source"] = "direct-alias"
            return pool_result
        # OLLAMA_API_KEY gets its own gate here: without it a `model_aliases:`
        # entry pointing at Ollama Cloud resolved no key at all.
        api_key_candidates = [
            (explicit_api_key or "").strip(),
            *_host_gated_env_key_candidates(base_url, ollama=True),
        ]
        api_key = next((c for c in api_key_candidates if has_usable_secret(c)), "") or "no-key-required"
        return _runtime(
            "custom",
            _detect_api_mode_for_url(base_url) or "chat_completions",
            base_url,
            api_key,
            source="direct-alias",
            requested_provider=requested_provider,
        )

    custom_provider = _get_named_custom_provider(requested_provider)
    if not custom_provider:
        return None

    base_url = (
        (explicit_base_url or "").strip()
        or custom_provider.get("base_url", "")
    ).rstrip("/")
    if not base_url:
        return None

    # Check if a credential pool exists for this custom endpoint
    pool_result = _try_resolve_from_custom_pool(
        base_url,
        "custom",
        custom_provider.get("api_mode"),
        provider_name=custom_provider.get("provider_key") or custom_provider.get("name"),
    )
    if pool_result:
        # Propagate the model name / capabilities / headers even when using
        # pooled credentials — the pool doesn't know about the custom_providers
        # fields. An explicit ``target_model`` wins (same rule as the non-pool path).
        _apply_custom_provider_extras(custom_provider, target_model, pool_result)
        return pool_result

    api_key_candidates = [
        (explicit_api_key or "").strip(),
        str(custom_provider.get("api_key", "") or "").strip(),
        _getenv(str(custom_provider.get("key_env", "") or "").strip(), "").strip(),
        *_host_gated_env_key_candidates(base_url, ollama=False),
    ]
    api_key = next((candidate for candidate in api_key_candidates if has_usable_secret(candidate)), "")

    # ``key_cmd`` credentials are minted per request (short-lived bearers would go stale
    # mid-session); both wire clients accept a callable api_key (the Entra ID contract).
    # An explicit --api-key still wins as the one-off recovery escape hatch.
    key_cmd = str(custom_provider.get("key_cmd", "") or "").strip()
    if key_cmd and not has_usable_secret((explicit_api_key or "").strip()):
        from agent.command_token_source import build_command_token_provider

        token_provider = build_command_token_provider(
            key_cmd,
            str(custom_provider.get("name", requested_provider) or "custom"),
        )
        if token_provider is not None:
            api_key = token_provider

    result = _runtime(
        "custom",
        custom_provider.get("api_mode")
        or _detect_api_mode_for_url(base_url)
        or "chat_completions",
        base_url,
        api_key or "no-key-required",
        source=f"custom_provider:{custom_provider.get('name', requested_provider)}",
        requested_provider=requested_provider,
    )
    _apply_custom_provider_extras(custom_provider, target_model, result)

    # OpenCode-family custom providers (opencode-go/zen names, or opencode.ai hosts) serve
    # models on different API surfaces — a static api_mode 503s for /v1/responses-only models
    # (#85589). Re-derive api_mode from the model and normalize /v1 like the built-in paths.
    from hermes_cli.models import opencode_provider_family

    _oc_family = opencode_provider_family(requested_provider)
    if _oc_family is None:
        try:
            if base_url_hostname(base_url).lower() == "opencode.ai":
                _oc_family = "opencode-go" if "/zen/go" in base_url.lower() else "opencode-zen"
        except Exception:
            _oc_family = None
    if _oc_family is not None and not custom_provider.get("api_mode"):
        from hermes_cli.models import (
            normalize_opencode_base_url,
            opencode_model_api_mode,
        )

        _effective_model = str(
            target_model
            or custom_provider.get("model")
            or _get_model_config().get("default")
            or ""
        ).strip()
        if _effective_model:
            result["api_mode"] = opencode_model_api_mode(_oc_family, _effective_model)
        result["base_url"] = normalize_opencode_base_url(
            _oc_family, result["api_mode"], result["base_url"]
        )
    return result


def _resolve_openrouter_runtime(
    *,
    requested_provider: str,
    explicit_api_key: Optional[str] = None,
    explicit_base_url: Optional[str] = None,
) -> Dict[str, Any]:
    model_cfg = _get_model_config()
    cfg_base_url = model_cfg.get("base_url") if isinstance(model_cfg.get("base_url"), str) else ""
    cfg_provider = model_cfg.get("provider") if isinstance(model_cfg.get("provider"), str) else ""
    cfg_api_key = next(
        (v.strip() for v in (model_cfg.get("api_key"), model_cfg.get("api")) if isinstance(v, str) and v.strip()),
        "",
    )
    requested_norm = (requested_provider or "").strip().lower()
    cfg_provider = cfg_provider.strip().lower()
    # Aliases resolving to "custom" (ollama, vllm, …) follow bare-custom trust + routing rules;
    # normalising here keeps every check below alias-aware (#27132).
    if requested_norm and requested_norm != "custom" and _resolves_to_custom(requested_norm):
        requested_norm = "custom"

    env_openrouter_base_url = _getenv("OPENROUTER_BASE_URL", "").strip()
    env_custom_base_url = _getenv("CUSTOM_BASE_URL", "").strip()

    # Use config base_url when available and the provider context matches.
    # OPENAI_BASE_URL env var is no longer consulted — config.yaml is
    # the single source of truth for endpoint URLs.
    use_config_base_url = bool(cfg_base_url.strip()) and not explicit_base_url and (
        (requested_norm == "auto" and cfg_provider in ("", "auto"))
        or (
            requested_norm == "custom"
            and _config_base_url_trustworthy_for_bare_custom(cfg_base_url, cfg_provider)
        )
    )

    base_url = (
        (explicit_base_url or "").strip()
        or env_custom_base_url
        or (cfg_base_url.strip() if use_config_base_url else "")
        or env_openrouter_base_url
        or OPENROUTER_BASE_URL
    ).rstrip("/")

    # OpenRouter endpoints prefer OPENROUTER_API_KEY (#289); custom endpoints must not receive
    # the OpenRouter key (#420, #560).
    _is_openrouter_url = base_url_host_matches(base_url, "openrouter.ai")
    # Explicitly-configured OpenRouter mirrors (OPENROUTER_BASE_URL + provider=openrouter)
    # still count as OpenRouter for key selection.
    _is_openrouter_context = _is_openrouter_url or (
        requested_norm == "openrouter"
        and (env_openrouter_base_url or base_url == env_openrouter_base_url)
        and base_url == (env_openrouter_base_url or "").rstrip("/")
    )
    if _is_openrouter_context:
        api_key_candidates = [
            explicit_api_key,
            _getenv("OPENROUTER_API_KEY"),
            _getenv("OPENAI_API_KEY"),
        ]
    else:
        # Custom endpoint: use api_key from config when using config base_url (#1760),
        # then env keys gated on their authoritative hosts (Ollama Cloud, OpenAI,
        # OpenRouter) and the host-derived `<VENDOR>_API_KEY`.
        api_key_candidates = [
            explicit_api_key,
            (cfg_api_key if use_config_base_url else ""),
            *_host_gated_env_key_candidates(base_url, ollama=True),
        ]
    api_key = next((str(c or "").strip() for c in api_key_candidates if has_usable_secret(c)), "")

    source = "explicit" if (explicit_api_key or explicit_base_url) else "env/config"

    # Explicit "custom" stays "custom" rather than relabeling to "openrouter" (#2562). Local
    # no-auth servers get a placeholder key — the OpenAI SDK requires a non-empty string.
    effective_provider = "custom" if requested_norm == "custom" else "openrouter"

    # For custom endpoints, check if a credential pool exists
    if effective_provider == "custom" and base_url:
        # Pass requested_provider so pool lookup prefers name match over base_url,
        # fixing credential mix-ups when multiple custom providers share a base_url.
        pool_result = _try_resolve_from_custom_pool(
            base_url, effective_provider, _parse_api_mode(model_cfg.get("api_mode")),
            provider_name=requested_provider if requested_norm != "custom" else None,
        )
        if pool_result:
            return pool_result

    if effective_provider == "custom" and not api_key and not _is_openrouter_url:
        api_key = "no-key-required"

    return _runtime(
        effective_provider,
        _resolve_plain_custom_api_mode(model_cfg, base_url)
        if effective_provider == "custom"
        else _parse_api_mode(model_cfg.get("api_mode"))
        or _detect_api_mode_for_url(base_url)
        or "chat_completions",
        base_url,
        api_key,
        source=source,
    )


def _resolve_azure_foundry_runtime(
    *,
    requested_provider: str,
    model_cfg: Dict[str, Any],
    explicit_api_key: Optional[str] = None,
    explicit_base_url: Optional[str] = None,
    target_model: Optional[str] = None,
) -> Dict[str, Any]:
    """Resolve an Azure Foundry runtime entry.

    Reads ``model.base_url`` + ``model.api_mode`` from config.yaml (or explicit overrides), pulls
    the API key from ``.env`` / env var, and strips a trailing ``/v1`` for Anthropic-style endpoints
    because the Anthropic SDK appends ``/v1/messages`` internally.
    """
    explicit_api_key = str(explicit_api_key or "").strip()
    explicit_base_url_clean = str(explicit_base_url or "").strip().rstrip("/")

    cfg_base_url = ""
    cfg_api_mode = "chat_completions"
    cfg_auth_mode = "api_key"
    cfg_entra: Dict[str, Any] = {}
    if _cfg_provider(model_cfg) == "azure-foundry":
        cfg_base_url = _config_base_url_for_provider(model_cfg, "azure-foundry")
        cfg_api_mode = _parse_api_mode(model_cfg.get("api_mode")) or "chat_completions"
        cfg_auth_mode = str(model_cfg.get("auth_mode") or "api_key").strip().lower() or "api_key"
        _entra = model_cfg.get("entra")
        if isinstance(_entra, dict):
            cfg_entra = _entra

    # Model-family inference: Azure Foundry deploys GPT-5.x / codex / o1-o4
    # reasoning models as Responses-API-only (see _azure_inferred_api_mode).
    effective_model = str(target_model or model_cfg.get("default") or "").strip()
    cfg_api_mode = _azure_inferred_api_mode(effective_model, cfg_api_mode)

    env_base_url = _getenv("AZURE_FOUNDRY_BASE_URL", "").strip().rstrip("/")
    base_url = explicit_base_url_clean or cfg_base_url or env_base_url
    if not base_url:
        raise AuthError(
            "Azure Foundry requires a base URL. Set it via 'hermes model' or "
            "the AZURE_FOUNDRY_BASE_URL environment variable."
        )

    # Anthropic SDK appends /v1/messages itself, so strip any trailing /v1
    # we inherited from the configured base_url to avoid double-/v1 paths.
    if cfg_api_mode == "anthropic_messages":
        base_url = re.sub(r"/v1/?$", "", base_url)

    # ── Entra ID (Microsoft Foundry recommended path) ──────────────────
    # Return a callable api_key that mints a fresh JWT per request: the OpenAI SDK accepts it
    # natively; for Anthropic-style endpoints ``build_anthropic_client`` injects the bearer via an
    # httpx request hook. Both modes look identical from here.
    if cfg_auth_mode == "entra_id":
        if explicit_api_key:
            # User passed --api-key on the CLI while config says entra_id —
            # honour the explicit string (escape hatch for one-off testing).
            api_key: Any = explicit_api_key
            source = "explicit"
            auth_mode = "api_key"
        else:
            try:
                from agent.azure_identity_adapter import (
                    EntraIdentityConfig,
                    SCOPE_AI_AZURE_DEFAULT,
                    build_token_provider,
                )
            except Exception as exc:
                raise AuthError(
                    "Azure Foundry Entra ID auth requires the 'azure-identity' "
                    "package. Install it with: pip install azure-identity "
                    f"(import failed: {exc})"
                ) from exc

            scope = (
                str(cfg_entra.get("scope") or "").strip()
                or SCOPE_AI_AZURE_DEFAULT
            )
            try:
                entra_config = EntraIdentityConfig(
                    scope=scope,
                )
                token_provider = build_token_provider(config=entra_config)
            except ImportError as exc:
                raise AuthError(str(exc)) from exc
            api_key = token_provider
            source = "entra_id"
            auth_mode = "entra_id"

        clean_entra = {}
        configured_scope = str(cfg_entra.get("scope") or "").strip()
        if auth_mode == "entra_id" and configured_scope:
            clean_entra["scope"] = configured_scope

        return _runtime(
            "azure-foundry",
            cfg_api_mode,
            base_url,
            api_key,
            auth_mode=auth_mode,
            entra=clean_entra,
            source=source,
            requested_provider=requested_provider,
        )

    # ── Static API key (legacy / default) ──────────────────────────────
    api_key = explicit_api_key
    if not api_key:
        try:
            from hermes_cli.config import get_env_value
            api_key = get_env_value("AZURE_FOUNDRY_API_KEY") or ""
        except Exception:
            api_key = ""
        api_key = api_key or _getenv("AZURE_FOUNDRY_API_KEY", "").strip()
    if not api_key:
        raise AuthError(
            "Azure Foundry requires an API key. Set AZURE_FOUNDRY_API_KEY in "
            "~/.hermes/.env or run 'hermes model' to configure. To use "
            "keyless Microsoft Entra ID auth instead, set "
            "model.auth_mode: entra_id in config.yaml (or pick "
            "'Microsoft Entra ID' in 'hermes model')."
        )

    source = "explicit" if (explicit_api_key or explicit_base_url) else "config"
    return _runtime(
        "azure-foundry",
        cfg_api_mode,
        base_url,
        api_key,
        auth_mode="api_key",
        source=source,
        requested_provider=requested_provider,
    )


def _resolve_explicit_runtime(
    *,
    provider: str,
    requested_provider: str,
    model_cfg: Dict[str, Any],
    explicit_api_key: Optional[str] = None,
    explicit_base_url: Optional[str] = None,
    target_model: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    explicit_api_key = str(explicit_api_key or "").strip()
    explicit_base_url = str(explicit_base_url or "").strip().rstrip("/")
    if not explicit_api_key and not explicit_base_url:
        return None

    if provider == "anthropic":
        base_url = explicit_base_url or _anthropic_cfg_base_url(model_cfg) or _ANTHROPIC_DEFAULT_BASE_URL
        api_key = explicit_api_key
        if not api_key:
            from agent.anthropic_adapter import resolve_anthropic_token

            api_key = resolve_anthropic_token()
            if not api_key:
                raise AuthError(_NO_ANTHROPIC_CREDENTIALS_MSG)
        return _runtime(
            "anthropic",
            "anthropic_messages",
            base_url,
            api_key,
            source="explicit",
            requested_provider=requested_provider,
        )

    if provider == "openai-codex":
        base_url = explicit_base_url or DEFAULT_CODEX_BASE_URL
        api_key = explicit_api_key
        last_refresh = None
        if not api_key:
            creds = resolve_codex_runtime_credentials()
            api_key = creds.get("api_key", "")
            last_refresh = creds.get("last_refresh")
            base_url = explicit_base_url or creds.get("base_url", "").rstrip("/") or base_url
        return _runtime(
            "openai-codex",
            "codex_responses",
            base_url,
            api_key,
            source="explicit",
            last_refresh=last_refresh,
            requested_provider=requested_provider,
        )

    if provider == "nous":
        state = auth_mod.get_provider_auth_state("nous") or {}
        base_url = (
            explicit_base_url
            or _nous_inference_base_url_override()
            or str(state.get("inference_base_url") or auth_mod.DEFAULT_NOUS_INFERENCE_URL).strip().rstrip("/")
        )
        # Only use the agent_key compatibility field for inference when it
        # contains a NAS invoke JWT; raw OAuth access_token fallback is handled
        # by resolve_nous_runtime_credentials().
        api_key = explicit_api_key or (
            str(state.get("agent_key") or "").strip()
            if _agent_key_is_usable(
                state,
                max(60, env_int("HERMES_NOUS_MIN_KEY_TTL_SECONDS", 1800)),
            )
            else ""
        )
        expires_at = state.get("agent_key_expires_at") or state.get("expires_at")
        if not api_key:
            creds = _resolve_nous_creds()
            api_key = creds.get("api_key", "")
            expires_at = creds.get("expires_at")
            base_url = explicit_base_url or creds.get("base_url", "").rstrip("/") or base_url
        return _runtime(
            "nous",
            _nous_api_mode(target_model or model_cfg.get("default") or ""),
            base_url,
            api_key,
            source="explicit",
            expires_at=expires_at,
            requested_provider=requested_provider,
        )

    # Azure Foundry: user-configured endpoint with selectable API mode
    if provider == "azure-foundry":
        return _resolve_azure_foundry_runtime(
            requested_provider=requested_provider,
            model_cfg=model_cfg,
            explicit_api_key=explicit_api_key,
            explicit_base_url=explicit_base_url,
        )

    pconfig = PROVIDER_REGISTRY.get(provider)
    if pconfig and pconfig.auth_type == "api_key":
        base_url = explicit_base_url
        if not base_url:
            if provider in {"kimi-coding", "kimi-coding-cn"}:
                creds = resolve_api_key_provider_credentials(provider)
                base_url = creds.get("base_url", "").rstrip("/")
            else:
                env_url = _getenv(pconfig.base_url_env_var, "").strip().rstrip("/") if pconfig.base_url_env_var else ""
                base_url = env_url or pconfig.inference_base_url

        if provider == "actual":
            base_url = normalize_actual_base_url(base_url)

        api_key = explicit_api_key
        if not api_key:
            creds = resolve_api_key_provider_credentials(provider)
            api_key = creds.get("api_key", "")
            if not base_url:
                base_url = creds.get("base_url", "").rstrip("/")
                if provider == "actual":
                    base_url = normalize_actual_base_url(base_url)

        api_mode = _api_key_provider_api_mode(
            provider, model_cfg, api_key, base_url,
            target_model or model_cfg.get("default", ""), opencode_by_model=False,
        )

        if provider == "actual" and not api_key and is_actual_local_base_url(base_url):
            api_key = ACTUAL_LOCAL_NOAUTH_PLACEHOLDER

        return _runtime(
            provider,
            api_mode,
            base_url.rstrip("/"),
            api_key,
            source="explicit",
            requested_provider=requested_provider,
        )

    return None


def _is_external_process_provider(provider: str) -> bool:
    """Whether ``provider`` is declared as an external-process (CLI) provider.

    Reads the CLI provider registry first (which now absorbs registered
    ProviderProfiles, in-tree and out), then falls back to the profile registry
    directly so the check works before the CLI registry has been extended.
    """
    name = (provider or "").strip().lower()
    if not name:
        return False
    try:
        from hermes_cli.auth import PROVIDER_REGISTRY

        pconfig = PROVIDER_REGISTRY.get(name)
        if pconfig is not None:
            return pconfig.auth_type == "external_process"
    except Exception:
        pass
    try:
        from providers import get_provider_profile

        profile = get_provider_profile(name)
    except Exception:
        return False
    return profile is not None and getattr(profile, "auth_type", "") == "external_process"


def _resolve_nous_creds() -> Dict[str, Any]:
    return resolve_nous_runtime_credentials(
        timeout_seconds=float(_getenv("HERMES_NOUS_TIMEOUT_SECONDS", "15")),
    )


def _nous_api_mode(model: str) -> str:
    from hermes_cli.providers import nous_api_mode

    return nous_api_mode(model)


@dataclass(frozen=True)
class _OAuthRuntimeSpec:
    """Env/auth-store OAuth providers resolved by a single credential call."""

    resolve: Callable[[], Dict[str, Any]]
    api_mode: Any  # str, or callable(model) -> str
    default_source: str
    expiry_key: str
    failure_msg: str
    default_base_url: str = ""


# ``resolve`` entries are late-bound lambdas so tests can monkeypatch the module-level
# ``resolve_*_runtime_credentials`` names.
_OAUTH_RUNTIME_PROVIDERS: Dict[str, _OAuthRuntimeSpec] = {
    "nous": _OAuthRuntimeSpec(
        _resolve_nous_creds, _nous_api_mode, "portal", "expires_at",
        "Auto-detected Nous provider but credentials failed",
    ),
    "openai-codex": _OAuthRuntimeSpec(
        lambda: resolve_codex_runtime_credentials(), "codex_responses", "hermes-auth-store", "last_refresh",
        "Auto-detected Codex provider but credentials failed",
    ),
    "xai-oauth": _OAuthRuntimeSpec(
        lambda: resolve_xai_oauth_runtime_credentials(), "codex_responses", "hermes-auth-store", "last_refresh",
        "Auto-detected xAI OAuth provider but credentials failed",
        default_base_url=DEFAULT_XAI_OAUTH_BASE_URL,
    ),
    "qwen-oauth": _OAuthRuntimeSpec(
        lambda: resolve_qwen_runtime_credentials(), "chat_completions", "qwen-cli", "expires_at_ms",
        "Qwen OAuth credentials failed",
    ),
}


def resolve_runtime_provider(
    *,
    requested: Optional[str] = None,
    explicit_api_key: Optional[str] = None,
    explicit_base_url: Optional[str] = None,
    target_model: Optional[str] = None,
) -> Dict[str, Any]:
    """Resolve runtime provider credentials for agent execution.

    target_model: Optional override for model_cfg.get("default") when computing provider-specific
    api_mode (e.g. OpenCode Zen/Go where different models route through different API surfaces).
    """
    requested_provider = resolve_requested_provider(requested)

    # Honour ``providers.<name>.enabled: false`` for built-in providers too (the
    # ``_get_named_custom_provider`` gate only covers custom blocks). Fail fast with a typed
    # error so the fallback chain advances instead of using a disabled provider.
    from hermes_cli.config import is_provider_enabled, load_config
    _full_cfg = load_config()
    _provs_cfg = _full_cfg.get("providers") if isinstance(_full_cfg, dict) else None
    if isinstance(_provs_cfg, dict):
        _block = _provs_cfg.get(requested_provider)
        if isinstance(_block, dict) and not is_provider_enabled(_block):
            raise ValueError(
                f"provider {requested_provider!r} is disabled in config "
                f"(providers.{requested_provider}.enabled: false)"
            )

    if requested_provider == "moa":
        return _runtime(
            "moa",
            "chat_completions",
            "moa://local",
            "moa-virtual-provider",
            source="moa-virtual-provider",
            requested_provider=requested_provider,
        )

    # Azure Anthropic short-circuit: an explicit Azure endpoint with provider="anthropic" must
    # bypass _resolve_named_custom_runtime (which would yield custom/chat_completions/no key).
    _eff_base = (explicit_base_url or "").strip()
    if requested_provider == "anthropic" and base_url_host_matches(_eff_base, "azure.com"):
        _azure_key = (
            (explicit_api_key or "").strip()
            or _getenv("AZURE_ANTHROPIC_KEY", "").strip()
            or _getenv("ANTHROPIC_API_KEY", "").strip()
        )
        return _runtime(
            "anthropic",
            "anthropic_messages",
            _eff_base.rstrip("/"),
            _azure_key,
            source="azure-explicit",
            requested_provider=requested_provider,
        )

    # Azure Foundry resolves before the custom-runtime / pool / generic paths so its config is
    # always picked up from model.base_url + model.api_mode, with or without explicit_* args.
    if requested_provider == "azure-foundry":
        return _resolve_azure_foundry_runtime(
            requested_provider=requested_provider,
            model_cfg=_get_model_config(),
            explicit_api_key=explicit_api_key,
            explicit_base_url=explicit_base_url,
            target_model=target_model,
        )

    # Vertex AI (OAuth2): resolve BEFORE the pool / generic paths — the credential *path*
    # (GOOGLE_APPLICATION_CREDENTIALS) must never be treated as a static API key. A short-lived
    # token is minted per call by get_vertex_config(); mid-session expiry is recovered on 401 by
    # run_agent._try_refresh_vertex_client_credentials().
    if requested_provider in ("vertex", "google-vertex", "vertex-ai", "gcp-vertex", "vertexai"):
        from agent.vertex_adapter import get_vertex_config

        token, base_url = get_vertex_config()
        if not token or not base_url:
            raise AuthError(
                "Vertex AI credentials could not be resolved. Vertex uses "
                "OAuth2 (not a static API key): provide a service-account JSON "
                "via GOOGLE_APPLICATION_CREDENTIALS (or VERTEX_CREDENTIALS_PATH) "
                "in ~/.hermes/.env, or run 'gcloud auth application-default "
                "login' for ADC. Set the GCP project/region under vertex: in "
                "config.yaml if they aren't embedded in the credentials. "
                "Run `hermes setup` to install Vertex support."
            )
        return _runtime(
            "vertex",
            "chat_completions",
            base_url.rstrip("/"),
            token,
            source="vertex-oauth",
            requested_provider=requested_provider,
        )

    custom_runtime = _resolve_named_custom_runtime(
        requested_provider=requested_provider,
        explicit_api_key=explicit_api_key,
        explicit_base_url=explicit_base_url,
        target_model=target_model,
    )
    if custom_runtime:
        custom_runtime["requested_provider"] = requested_provider
        return custom_runtime

    # provider "auto"/unset with a config base_url at a custom/local endpoint routes through the
    # OpenAI-compatible resolver, so resolve_provider() cannot pick up an env ANTHROPIC/OPENAI
    # key and send the request to a cloud API (#3846).
    if not explicit_base_url and not explicit_api_key:
        model_cfg = _get_model_config()
        cfg_provider = str(model_cfg.get("provider") or "").strip().lower()
        cfg_base_url = str(model_cfg.get("base_url") or "").strip()
        if cfg_base_url and cfg_provider in ("auto", ""):
            # Only non-cloud roots (Ollama, LM Studio, vLLM, …) take the bypass. Match on HOST,
            # not substring, so a look-alike (api.anthropic.com.attacker.test) cannot leak a
            # cloud credential.
            if not any(
                base_url_host_matches(cfg_base_url, host)
                for host in ("openrouter.ai", "anthropic.com", "openai.com")
            ):
                runtime = _resolve_openrouter_runtime(
                    requested_provider=requested_provider,
                    explicit_api_key=explicit_api_key,
                    explicit_base_url=explicit_base_url,
                )
                runtime["requested_provider"] = requested_provider
                return runtime

    provider = resolve_provider(
        requested_provider,
        explicit_api_key=explicit_api_key,
        explicit_base_url=explicit_base_url,
    )
    model_cfg = _get_model_config()

    # OpenCode Zen free tier (*-free slugs) is served ANONYMOUSLY on the Zen relay only: unknown
    # bearers 401 and the Go relay rejects free models. Route free slugs through the keyless Zen
    # runtime BEFORE the pool / explicit / api_key paths.
    from hermes_cli.models import (
        opencode_provider_family as _oc_family_fn,
        opencode_zen_free_runtime as _oc_free_runtime_fn,
    )
    if _oc_family_fn(provider) is not None:
        _oc_model = str(
            target_model or model_cfg.get("default") or model_cfg.get("model") or ""
        ).strip()
        _free_runtime = _oc_free_runtime_fn(provider, _oc_model)
        if _free_runtime is not None:
            _free_runtime["requested_provider"] = requested_provider
            return _free_runtime

    explicit_runtime = _resolve_explicit_runtime(
        provider=provider,
        requested_provider=requested_provider,
        model_cfg=model_cfg,
        explicit_api_key=explicit_api_key,
        explicit_base_url=explicit_base_url,
        target_model=target_model,
    )
    if explicit_runtime:
        return explicit_runtime

    should_use_pool = provider != "openrouter"
    if provider == "openrouter":
        cfg_provider = str(model_cfg.get("provider") or "").strip().lower()
        cfg_base_url = str(model_cfg.get("base_url") or "").strip()
        env_openai_base_url = _getenv("OPENAI_BASE_URL", "").strip()
        env_openrouter_base_url = _getenv("OPENROUTER_BASE_URL", "").strip()
        has_custom_endpoint = bool(
            explicit_base_url
            or env_openai_base_url
            or env_openrouter_base_url
        )
        if cfg_base_url and cfg_provider in {"auto", "custom"}:
            has_custom_endpoint = True
        has_runtime_override = bool(explicit_api_key or explicit_base_url)
        should_use_pool = (
            requested_provider in {"openrouter", "auto"}
            and not has_custom_endpoint
            and not has_runtime_override
        )

    try:
        pool = load_pool(provider) if should_use_pool else None
    except Exception:
        pool = None
    if pool and pool.has_credentials():
        entry = pool.select()
        pool_api_key = _pool_entry_api_key(entry) if entry is not None else ""
        # Nous pool entries carry the agent_key (an invoke JWT) which the pool does not refresh
        # on selection (avoids network calls in `hermes auth list`); refresh it here before
        # falling back to singleton auth resolution.
        if provider == "nous" and entry is not None:
            min_ttl = max(60, env_int("HERMES_NOUS_MIN_KEY_TTL_SECONDS", 1800))
            if not _agent_key_is_usable(_nous_pool_state(entry), min_ttl):
                logger.debug("Nous pool entry agent_key expired/missing, refreshing selected pool entry")
                try:
                    refreshed = pool.try_refresh_current()
                except Exception as exc:
                    logger.debug("Nous pool entry refresh failed: %s", exc)
                    refreshed = None
                if refreshed is not None:
                    entry = refreshed
                    pool_api_key = _pool_entry_api_key(entry)
                if not pool_api_key or not _agent_key_is_usable(_nous_pool_state(entry), min_ttl):
                    logger.debug("Nous pool entry agent_key still unavailable, falling through to runtime resolution")
                    pool_api_key = ""
        if (
            entry is not None
            and pool_api_key
            and credential_pool_matches_provider(
                pool,
                provider,
                base_url=_pool_entry_base_url(entry),
            )
        ):
            return _resolve_runtime_from_pool_entry(
                provider=provider,
                entry=entry,
                requested_provider=requested_provider,
                model_cfg=model_cfg,
                pool=pool,
                target_model=target_model,
            )

    if provider in _OAUTH_RUNTIME_PROVIDERS:
        spec = _OAUTH_RUNTIME_PROVIDERS[provider]
        try:
            creds = spec.resolve()
        except AuthError:
            if requested_provider != "auto":
                raise
            # Auto-detected but credentials are stale/revoked — fall through
            # to env-var providers (e.g. OpenRouter).
            logger.info("%s; falling through to next provider.", spec.failure_msg)
        else:
            api_mode = spec.api_mode
            if callable(api_mode):
                api_mode = api_mode(target_model or model_cfg.get("default") or "")
            return _runtime(
                provider,
                api_mode,
                (creds.get("base_url") or "").rstrip("/") or spec.default_base_url,
                creds.get("api_key", ""),
                source=creds.get("source", spec.default_source),
                **{spec.expiry_key: creds.get(spec.expiry_key)},
                requested_provider=requested_provider,
            )

    if provider == "minimax-oauth":
        pconfig = PROVIDER_REGISTRY.get(provider)
        if pconfig and pconfig.auth_type == "oauth_minimax":
            from hermes_cli.auth import resolve_minimax_oauth_runtime_credentials
            creds = resolve_minimax_oauth_runtime_credentials()
            return _runtime(
                provider,
                "anthropic_messages",
                creds["base_url"],
                creds["api_key"],
                source=creds.get("source", "oauth"),
                requested_provider=requested_provider,
            )

    # External-process providers (an agent CLI driven over stdio, e.g. ACP).
    # Keyed on the registered provider's auth_type rather than on one name, so a
    # provider shipped outside this tree lands on the same credential path.
    if _is_external_process_provider(provider):
        creds = resolve_external_process_provider_credentials(provider)
        return _runtime(
            provider,
            "chat_completions",
            creds.get("base_url", "").rstrip("/"),
            creds.get("api_key", ""),
            command=creds.get("command", ""),
            args=list(creds.get("args") or []),
            source=creds.get("source", "process"),
            requested_provider=requested_provider,
        )

    # Anthropic (native Messages API)
    if provider == "anthropic":
        # Allow base URL override from config.yaml model.base_url, but only
        # when the configured provider is anthropic — otherwise a non-Anthropic
        # base_url (e.g. Codex endpoint) would leak into Anthropic requests.
        cfg_base_url = _anthropic_cfg_base_url(model_cfg)
        base_url = cfg_base_url or _ANTHROPIC_DEFAULT_BASE_URL

        # Microsoft Foundry endpoints reject Claude Code OAuth tokens, which
        # resolve_anthropic_token() would return first — use the env key directly.
        _is_azure_endpoint = base_url_host_matches(base_url, "azure.com") or (
            cfg_base_url and base_url_host_matches(cfg_base_url, "azure.com")
        )
        if _is_azure_endpoint:
            # Env var hints on the model config first: `key_env` (Hermes canonical) and
            # `api_key_env` (Azure Foundry guide / importers).
            token = ""
            for hint_key in ("key_env", "api_key_env"):
                env_var = str(model_cfg.get(hint_key) or "").strip()
                if env_var:
                    token = _getenv(env_var, "").strip()
                    if token:
                        break
            # Then an inline api_key on the model config (multi-profile setups),
            # finally the historical fixed names.
            token = (
                token
                or str(model_cfg.get("api_key") or "").strip()
                or _getenv("AZURE_ANTHROPIC_KEY", "").strip()
                or _getenv("ANTHROPIC_API_KEY", "").strip()
            )
            if not token:
                raise AuthError(
                    "No Azure Anthropic API key found. Set AZURE_ANTHROPIC_KEY or "
                    "ANTHROPIC_API_KEY, or point key_env/api_key_env in your "
                    "config.yaml model section at a custom env var."
                )
        else:
            from agent.anthropic_adapter import resolve_anthropic_token
            token = resolve_anthropic_token()
            if not token:
                raise AuthError(_NO_ANTHROPIC_CREDENTIALS_MSG)
        return _runtime(
            "anthropic",
            "anthropic_messages",
            base_url,
            token,
            source="env",
            requested_provider=requested_provider,
        )

    # AWS Bedrock (native Converse API via boto3)
    if provider == "bedrock":
        from agent.bedrock_adapter import (
            has_aws_credentials,
            resolve_aws_auth_env_var,
            resolve_bedrock_runtime_region,
            is_anthropic_bedrock_model,
            is_openai_bedrock_model,
            bedrock_openai_base_url,
            resolve_bedrock_bearer_token,
        )
        # Explicitly selected bedrock trusts boto3's credential chain (IMDS, ECS/Lambda roles,
        # SSO) which our env-var check can't detect.
        is_explicit = requested_provider in {"bedrock", "aws", "aws-bedrock", "amazon-bedrock", "amazon"}
        if not is_explicit and not has_aws_credentials():
            raise AuthError(
                "No AWS credentials found for Bedrock. Configure one of:\n"
                "  - AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY\n"
                "  - AWS_PROFILE (for SSO / named profiles)\n"
                "  - IAM instance role (EC2, ECS, Lambda)\n"
                "Or run 'aws configure' to set up credentials.",
                code="no_aws_credentials",
            )
        # Read bedrock-specific config from config.yaml
        _bedrock_cfg = load_config().get("bedrock", {})
        # Region priority: config.yaml bedrock.region → env var → us-east-1.
        # resolve_bedrock_runtime_region() is the canonical implementation of
        # this priority; auxiliary resolution uses the same helper.
        region = resolve_bedrock_runtime_region({"bedrock": _bedrock_cfg})
        auth_source = resolve_aws_auth_env_var() or "aws-sdk-default-chain"
        # Build guardrail config if configured
        _gr = _bedrock_cfg.get("guardrail", {})
        guardrail_config = None
        if _gr.get("guardrail_identifier") and _gr.get("guardrail_version"):
            guardrail_config = {
                "guardrailIdentifier": _gr["guardrail_identifier"],
                "guardrailVersion": _gr["guardrail_version"],
            }
            for src_key, dst_key in (("stream_processing_mode", "streamProcessingMode"), ("trace", "trace")):
                if _gr.get(src_key):
                    guardrail_config[dst_key] = _gr[src_key]
        # Triple-path routing: OpenAI models → Bedrock Mantle's Responses endpoint; Claude →
        # AnthropicBedrock SDK (prompt caching, thinking budgets); others → Converse API.
        # Exception: AWS_BEARER_TOKEN_BEDROCK auth is unsupported by AnthropicBedrock (SigV4
        # only), so bearer users go through Converse regardless of model (#28156).
        _current_model = str(target_model or model_cfg.get("default") or "").strip()
        _has_bearer_token = bool(os.environ.get("AWS_BEARER_TOKEN_BEDROCK", "").strip())
        runtime = _runtime(
            "bedrock",
            "bedrock_converse",
            f"https://bedrock-runtime.{region}.amazonaws.com",
            "aws-sdk",
            source=auth_source,
            region=region,
            requested_provider=requested_provider,
        )
        if is_openai_bedrock_model(_current_model):
            bearer = resolve_bedrock_bearer_token()
            runtime.update(
                api_mode="codex_responses",
                base_url=bedrock_openai_base_url(region),
                api_key=bearer or "aws-sdk",
                source="AWS_BEARER_TOKEN_BEDROCK" if bearer else auth_source,
                model=_current_model,
                bedrock_openai=True,
            )
        elif is_anthropic_bedrock_model(_current_model) and not _has_bearer_token:
            # Claude on Bedrock → AnthropicBedrock SDK → anthropic_messages path
            runtime.update(api_mode="anthropic_messages", bedrock_anthropic=True)
        # else: Non-Claude/OpenAI (Nova, DeepSeek, Llama, GPT-OSS, etc.) → Converse API
        if guardrail_config:
            runtime["guardrail_config"] = guardrail_config
        return runtime

    # API-key providers (z.ai/GLM, Kimi, MiniMax, MiniMax-CN)
    pconfig = PROVIDER_REGISTRY.get(provider)
    if pconfig and pconfig.auth_type == "api_key":
        creds = resolve_api_key_provider_credentials(provider)
        # Actual Computer: a loopback model_cfg base_url selects the daemon's no-auth local API;
        # inject the placeholder BEFORE the usable-secret gate (mirrors the env-driven path).
        if provider == "actual" and not has_usable_secret(creds.get("api_key")):
            _cfg_provider = str(model_cfg.get("provider") or "").strip().lower()
            _cfg_url = ""
            if _cfg_provider == provider:
                _cfg_url = (model_cfg.get("base_url") or "").strip().rstrip("/")
            _effective_url = normalize_actual_base_url(
                _cfg_url or creds.get("base_url", "").rstrip("/")
            )
            if is_actual_local_base_url(_effective_url):
                creds = dict(creds)
                creds["api_key"] = ACTUAL_LOCAL_NOAUTH_PLACEHOLDER
                creds["source"] = creds.get("source") or "local-offline"
        # An explicitly selected API-key provider is authoritative: an empty key would defer
        # failure to the first request and make a later fallback look like a silent provider
        # switch. LM Studio's no-auth path supplies a placeholder in the credential resolver.
        if not has_usable_secret(creds.get("api_key")):
            env_names = ", ".join(pconfig.api_key_env_vars)
            hint = f" Set {env_names}." if env_names else ""
            raise AuthError(
                f"No usable credentials found for provider '{provider}'.{hint}",
                provider=provider,
                code="missing_api_key",
            )
        # Honour model.base_url when the configured provider matches (e.g. the
        # api.minimaxi.com China endpoint instead of the hardcoded default, #6039).
        base_url = _config_base_url_for_provider(model_cfg, provider) or creds.get("base_url", "").rstrip("/")
        if provider == "actual":
            base_url = normalize_actual_base_url(base_url)
        api_mode = _api_key_provider_api_mode(
            provider, model_cfg, creds.get("api_key", ""), base_url,
            target_model or model_cfg.get("default", ""), opencode_by_model=True,
        )
        base_url = _normalize_opencode_runtime_base_url(provider, api_mode, base_url)
        if provider == "lmstudio":
            base_url = auth_mod._normalize_lmstudio_runtime_base_url(base_url)
        api_key = creds.get("api_key", "")
        if provider == "actual" and not api_key and is_actual_local_base_url(base_url):
            api_key = ACTUAL_LOCAL_NOAUTH_PLACEHOLDER
        return _runtime(
            provider,
            api_mode,
            base_url,
            api_key,
            source=creds.get("source", "env"),
            requested_provider=requested_provider,
        )

    runtime = _resolve_openrouter_runtime(
        requested_provider=requested_provider,
        explicit_api_key=explicit_api_key,
        explicit_base_url=explicit_base_url,
    )
    runtime["requested_provider"] = requested_provider
    return runtime


def format_runtime_provider_error(error: Exception) -> str:
    if isinstance(error, AuthError):
        return format_auth_error(error)
    return str(error)
