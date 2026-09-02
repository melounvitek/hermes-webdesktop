"""Model metadata, context lengths, and token estimation utilities.

Pure utility functions with no AIAgent dependency. Used by ContextCompressor
and run_agent.py for pre-flight context checks.
"""

import base64
import hashlib
import ipaddress
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
from urllib.parse import urlparse

import yaml

if TYPE_CHECKING:  # pragma: no cover — runtime import is lazy (see below)
    import requests

from utils import atomic_json_write, atomic_yaml_write, base_url_host_matches, base_url_hostname

from hermes_constants import OPENROUTER_MODELS_URL
from agent.message_metadata import PERSISTENCE_ONLY_MESSAGE_FIELDS

logger = logging.getLogger(__name__)

# ``requests`` costs ~27 ms of the `import cli` waterfall, so it is resolved
# lazily: ``_ensure_requests()`` on the runtime path, PEP 562 ``__getattr__``
# for external access (``patch("agent.model_metadata.requests.get")``).


def _ensure_requests():
    if "requests" not in globals():
        import requests as _requests
        globals()["requests"] = _requests
    return globals()["requests"]


def __getattr__(name: str):
    if name == "requests":
        return _ensure_requests()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _resolve_requests_verify(base_url: str = "") -> bool | str:
    """SSL ``verify`` for ``requests`` probes; mirrors ``agent.ssl_verify.resolve_httpx_verify``.

    Priority: per-provider ``ssl_verify: false`` -> per-provider ``ssl_ca_cert``
    (otherwise probes log spurious CERTIFICATE_VERIFY_FAILED while the httpx
    chat path succeeds) -> HERMES_CA_BUNDLE / REQUESTS_CA_BUNDLE / SSL_CERT_FILE
    -> ``True`` (certifi). Callers without a ``base_url`` keep env-only behavior.
    """
    if base_url:
        try:
            from hermes_cli.config import get_custom_provider_tls_settings
            tls = get_custom_provider_tls_settings(base_url)
            if tls.get("ssl_verify") is False:
                return False
            ca = tls.get("ssl_ca_cert")
            if isinstance(ca, str) and ca and os.path.isfile(ca):
                return ca
        except Exception:
            pass  # fall through to env vars — never break a probe on config lookup
    for env_var in ("HERMES_CA_BUNDLE", "REQUESTS_CA_BUNDLE", "SSL_CERT_FILE"):
        val = os.getenv(env_var)
        if val and os.path.isfile(val):
            return val
    return True

# Compatibility snapshot for callers that inspect this private constant.
# Prefix routing below queries the registry live so later registrations work.
try:
    from providers import list_providers as _list_providers
except Exception:
    def _list_providers():
        return []

_PROVIDER_PREFIXES: frozenset[str] = frozenset(
    value.lower()
    for profile in _list_providers()
    for value in (profile.name, *profile.aliases)
)


_OLLAMA_TAG_PATTERN = re.compile(
    r"^(\d+\.?\d*b|latest|stable|q\d|fp?\d|instruct|chat|coder|vision|text)",
    re.IGNORECASE,
)


# Tailscale CGNAT (RFC 6598): `ipaddress.is_private` excludes it, yet Ollama
# reached over Tailscale must count as local (timeout auto-bumps).
_TAILSCALE_CGNAT = ipaddress.IPv4Network("100.64.0.0/10")


def _strip_provider_prefix(model: str) -> str:
    """Strip a registry-known provider prefix: ``"local:m"`` -> ``"m"``.

    Ollama ``model:tag`` ids are preserved even when the model half is a
    provider name (``qwen:0.5b``, ``deepseek:latest``).
    """
    if ":" not in model or model.startswith("http"):
        return model
    prefix, suffix = model.split(":", 1)
    prefix_lower = prefix.strip().lower()
    try:
        from providers import get_provider_profile

        is_provider = get_provider_profile(prefix_lower) is not None
    except Exception:
        is_provider = False
    if is_provider:
        # Don't strip if suffix looks like an Ollama tag (e.g. "7b", "latest", "q4_0")
        if _OLLAMA_TAG_PATTERN.match(suffix.strip()):
            return model
        return suffix
    return model

_model_metadata_cache: Dict[str, Dict[str, Any]] = {}
_model_metadata_cache_time: float = 0
_novita_metadata_cache: Dict[str, Dict[str, Any]] = {}
_novita_metadata_cache_time: float = 0
_MODEL_CACHE_TTL = 3600
_endpoint_model_metadata_cache: Dict[str, Dict[str, Dict[str, Any]]] = {}
_endpoint_model_metadata_cache_time: Dict[str, float] = {}
_ENDPOINT_MODEL_CACHE_TTL = 300
# Server-type verdicts: (server_type, monotonic_ts). Positive verdicts live an
# hour (so a server swap on the same port is eventually re-detected); a None
# verdict gets the short TTL so a transient failure (server starting, key being
# fixed) recovers in minutes while still not re-running the waterfall each turn.
_ENDPOINT_PROBE_TTL_SECONDS = 3600.0
_ENDPOINT_PROBE_FAILURE_TTL_SECONDS = 300.0
_endpoint_probe_path_cache: Dict[str, tuple] = {}

# Routable-but-dead endpoints (corp LAN off-VPN) blackhole TCP: every probe
# waits out its full connect timeout and startup stalls for a minute. Once ANY
# probe observed a connect timeout, later probes short-circuit for a while.
# Pure bookkeeping — no network I/O, fires only after a real timeout was paid.
_ENDPOINT_BLACKHOLE_TTL_SECONDS = 30.0
_endpoint_blackhole_cache: Dict[str, float] = {}  # host:port -> monotonic ts


def _endpoint_host_key(base_url: str) -> Optional[str]:
    """``host:port`` key (None without a host) so every probe path for one server shares an entry."""
    normalized = _normalize_base_url(base_url)
    if not normalized:
        return None
    url = normalized if "://" in normalized else f"http://{normalized}"
    try:
        parsed = urlparse(url)
        host = parsed.hostname
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
    except Exception:
        return None
    return f"{host}:{port}" if host else None


def _note_endpoint_blackholed(base_url: str) -> None:
    """Record that a probe to ``base_url`` timed out during TCP connect."""
    key = _endpoint_host_key(base_url)
    if key is None:
        return
    _endpoint_blackhole_cache[key] = time.monotonic()
    logger.debug(
        "Endpoint %s timed out connecting — skipping further probes for %.0fs",
        key, _ENDPOINT_BLACKHOLE_TTL_SECONDS,
    )


def _endpoint_blackholed(base_url: str) -> bool:
    """True if a recent probe to ``base_url`` timed out during TCP connect (cache lookup only)."""
    if _ENDPOINT_BLACKHOLE_TTL_SECONDS <= 0:
        return False
    key = _endpoint_host_key(base_url)
    if key is None:
        return False
    seen = _endpoint_blackhole_cache.get(key)
    if seen is None:
        return False
    if (time.monotonic() - seen) >= _ENDPOINT_BLACKHOLE_TTL_SECONDS:
        del _endpoint_blackhole_cache[key]
        return False
    return True


def _is_connect_timeout(exc: BaseException) -> bool:
    """True for connect-phase timeouts raised by httpx or requests.

    Read timeouts are deliberately excluded: those mean the server accepted
    the connection, which is the opposite of the blackhole this guards.
    """
    try:
        import httpx
        if isinstance(exc, httpx.ConnectTimeout):
            return True
    except Exception:
        pass
    try:
        from requests.exceptions import ConnectTimeout
        if isinstance(exc, ConnectTimeout):
            return True
    except Exception:
        pass
    return False

# Disk L2 for local-endpoint probes so back-to-back CLI cold starts skip the
# waterfall. Only SUCCESSFUL probes are persisted (a down server must not pin
# a negative verdict); the TTL is shorter than the 1 h in-process one.
_LOCAL_PROBE_DISK_TTL_SECONDS = 300.0


def _local_probe_disk_cache_path() -> Path:
    from hermes_constants import get_hermes_home
    return get_hermes_home() / "cache" / "local_endpoint_probes.json"


def _load_local_probe_disk_cache() -> Dict[str, Any]:
    try:
        with _local_probe_disk_cache_path().open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _local_probe_disk_get(kind: str, key: str) -> Optional[Any]:
    """Return a fresh cached value for ``kind:key``, else None."""
    entry = _load_local_probe_disk_cache().get(f"{kind}:{key}")
    if not isinstance(entry, dict):
        return None
    try:
        if (time.time() - float(entry["ts"])) >= _LOCAL_PROBE_DISK_TTL_SECONDS:
            return None
        return entry["value"]
    except Exception:
        return None


def _local_probe_disk_put(kind: str, key: str, value: Any) -> None:
    """Persist a successful probe result. Best-effort; prunes stale entries."""
    try:
        now = time.time()
        data = _load_local_probe_disk_cache()
        data = {
            k: v
            for k, v in data.items()
            if isinstance(v, dict)
            and (now - float(v.get("ts", 0))) < _LOCAL_PROBE_DISK_TTL_SECONDS
        }
        data[f"{kind}:{key}"] = {"value": value, "ts": now}
        atomic_json_write(
            _local_probe_disk_cache_path(),
            data,
            indent=0,
            separators=(",", ":"),
        )
    except Exception as e:
        logger.debug("Failed to save local probe disk cache: %s", e)


def _get_model_metadata_cache_path() -> Path:
    """Return path to the OpenRouter model metadata disk cache."""
    from hermes_constants import get_hermes_home
    return get_hermes_home() / "cache" / "openrouter_model_metadata.json"


def _model_metadata_disk_cache_age_seconds() -> Optional[float]:
    """Return disk-cache age in seconds, or None if freshness is unknown."""
    try:
        cache_path = _get_model_metadata_cache_path()
        if not cache_path.exists():
            return None
        age = time.time() - cache_path.stat().st_mtime
        if age < 0:
            return None
        return age
    except Exception:
        return None


def _load_model_metadata_disk_cache() -> Dict[str, Dict[str, Any]]:
    """Load processed OpenRouter metadata cache from disk."""
    try:
        cache_path = _get_model_metadata_cache_path()
        with cache_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {}
        return {
            str(key): value
            for key, value in data.items()
            if isinstance(value, dict)
        }
    except Exception as e:
        logger.debug("Failed to load OpenRouter model metadata disk cache: %s", e)
        return {}


def _save_model_metadata_disk_cache(data: Dict[str, Dict[str, Any]]) -> None:
    """Save processed OpenRouter metadata cache to disk atomically."""
    try:
        atomic_json_write(
            _get_model_metadata_cache_path(),
            data,
            indent=0,
            separators=(",", ":"),
        )
    except Exception as e:
        logger.debug("Failed to save OpenRouter model metadata disk cache: %s", e)

def _get_endpoint_metadata_cache_path() -> Path:
    """On-disk memo of remote ``/models`` probes (see ``_endpoint_disk_cache_get``)."""
    from hermes_constants import get_hermes_home
    return get_hermes_home() / "cache" / "endpoint_model_metadata.json"


def _endpoint_disk_cache_get(normalized: str) -> Optional[Dict[str, Dict[str, Any]]]:
    """Fresh (``_ENDPOINT_MODEL_CACHE_TTL``) cross-process memo of a remote ``/models`` probe.

    One-shot runs (``hermes -q``, cron, Bot Mode hops) start cold; Nous bypasses
    the persistent context cache by design, so without this every launch paid
    the live probe. Same TTL as the in-memory cache keeps the portal
    authoritative. Local endpoints are never memoized (transient loaded context).
    """
    try:
        with _get_endpoint_metadata_cache_path().open("r", encoding="utf-8") as f:
            data = json.load(f)
        entry = data.get(normalized) if isinstance(data, dict) else None
        if not isinstance(entry, dict):
            return None
        if (time.time() - float(entry.get("at", 0))) >= _ENDPOINT_MODEL_CACHE_TTL:
            return None
        models = entry.get("models")
        return models if isinstance(models, dict) else None
    except Exception:
        return None


def _endpoint_disk_cache_put(normalized: str, cache: Dict[str, Dict[str, Any]]) -> None:
    """Memoize a successful remote ``/models`` probe; expired siblings are dropped."""
    try:
        path = _get_endpoint_metadata_cache_path()
        data: Dict[str, Any] = {}
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                now = time.time()
                data = {
                    k: v for k, v in loaded.items()
                    if isinstance(v, dict) and (now - float(v.get("at", 0))) < _ENDPOINT_MODEL_CACHE_TTL
                }
        data[normalized] = {"at": time.time(), "models": cache}
        atomic_json_write(path, data, indent=0, separators=(",", ":"))
    except Exception as e:
        logger.debug("Failed to save endpoint model metadata disk cache: %s", e)


# Descending probe tiers for unknown models; tier[0] is also the default fallback.
CONTEXT_PROBE_TIERS = [
    256_000,
    128_000,
    64_000,
    32_000,
    16_000,
    8_000,
]

# Default context length when no detection method succeeds.
DEFAULT_FALLBACK_CONTEXT = CONTEXT_PROBE_TIERS[0]

# The fallback result is never cached, so dedupe its warning per (model, base_url).
_FALLBACK_WARNED: set = set()


def _warn_context_length_fallback(model: str, base_url: str) -> None:
    """Warn (once per model+endpoint) that context detection failed and the
    hard default is being used, so small-context models (8K, 32K) don't
    silently get 256K and cause hard-to-debug API failures."""
    key = (model, base_url or "")
    if key in _FALLBACK_WARNED:
        return
    _FALLBACK_WARNED.add(key)
    logger.warning(
        "Could not determine context length for model %r (base_url=%s) "
        "— falling back to %s tokens. Set model.context_length in "
        "config.yaml to override.",
        model, base_url or "default", f"{DEFAULT_FALLBACK_CONTEXT:,}",
    )

# Sessions, model switches and cron jobs reject models below this: too little
# working memory for tool-calling workflows.
MINIMUM_CONTEXT_LENGTH = 64_000

# In-process cache for local-server context probes, (model, base_url) ->
# (result, monotonic_ts): one startup resolves the same model several times
# (banner, /model switch, compressor update_model). Never persisted.
_LOCAL_CTX_PROBE_TTL_SECONDS = 30.0
_LOCAL_CTX_PROBE_CACHE: Dict[tuple, tuple] = {}

# Family-pattern fallbacks, used only when provider-aware sources all miss.
# Lookups are longest-key-first substring matches, so dict order is cosmetic
# and a specific key must be STRICTLY longer than its catch-all.
DEFAULT_CONTEXT_LENGTHS = {
    # Anthropic — bare ids only (prefixed ids resolve via OpenRouter/models.dev
    # and would collide: "anthropic/claude-sonnet-4" ⊂ "anthropic/claude-sonnet-4.6").
    "claude-fable-5": 1000000,
    "claude-fable": 1000000,
    "claude-opus-5": 1000000,
    "claude-sonnet-5": 1000000,
    "claude-opus-4-8": 1000000,
    "claude-opus-4.8": 1000000,
    "claude-opus-4-7": 1000000,
    "claude-opus-4.7": 1000000,
    "claude-opus-4-6": 1000000,
    "claude-sonnet-4-6": 1000000,
    "claude-opus-4.6": 1000000,
    "claude-sonnet-4.6": 1000000,
    # Catch-all for older Claude models (must sort after specific entries)
    "claude": 200000,
    # OpenAI — direct-API windows (Codex OAuth caps gpt-5.4+/5.5/5.6 at 272K,
    # resolved by its own branch). https://developers.openai.com/api/docs/models
    "gpt-5.6-luna": 1050000,
    "gpt-5.6-terra": 1050000,
    "gpt-5.6-sol": 1050000,
    "gpt-5.5": 1050000,
    "gpt-5.4-nano": 400000,           # 400k (not 1.05M like full 5.4)
    "gpt-5.4-mini": 400000,           # 400k (not 1.05M like full 5.4)
    "gpt-5.4": 1050000,               # GPT-5.4, GPT-5.4 Pro (1.05M context)
    "gpt-5.3-codex-spark": 128000,    # Codex-OAuth-only; keeps "gpt-5" (400k) from winning
    "gpt-5.1-chat": 128000,           # Chat variant has 128k context
    "gpt-5": 400000,                  # GPT-5.x base, mini, codex variants (400k)
    "gpt-4.1": 1047576,
    "gpt-4": 128000,
    # Google
    "gemini": 1048576,
    # Gemma (open models served via AI Studio)
    "gemma-4": 256000,  # Gemma 4 family
    "gemma4": 256000,  # Ollama-style naming (e.g. gemma4:31b-cloud)
    "gemma-4-31b": 256000,
    "gemma-3": 131072,
    "gemma": 8192,  # fallback for older gemma models
    # DeepSeek — V4 family is 1M; deepseek-chat/-reasoner alias v4-flash modes.
    # https://api-docs.deepseek.com/zh-cn/quick_start/pricing
    "deepseek-v4-pro": 1_000_000,
    "deepseek-v4-flash": 1_000_000,
    "deepseek-chat": 1_000_000,
    "deepseek-reasoner": 1_000_000,
    "deepseek": 128000,
    # Meta
    "llama": 131072,
    # Thinking Machines — covers inkling-small and :free/:batch variants (the
    # :batch SKU's smaller live window comes from provider metadata).
    "inkling": 1_048_576,
    # Qwen — specific model families before the catch-all.
    # Official docs: https://help.aliyun.com/zh/model-studio/developer-reference/
    "qwen3.8-max": 1_000_000,     # 1M context (OpenRouter & Nous portal, verified 2026-08-03)
    "qwen3.8-flash": 1_000_000,   # 1M context (OpenRouter & Nous portal, verified 2026-08-28)
    "qwen3.6-plus": 1048576,      # 1M context (DashScope/Alibaba & OpenRouter)
    "qwen3.7-plus": 1048576,      # 1M context (DashScope/Alibaba)
    "qwen3-coder-plus": 1000000,  # 1M context
    "qwen3-coder": 262144,        # 256K context
    "qwen3-max": 262144,          # 256K context (qwen3-max-2026-01-23 snapshot, Coding Plan)
    "qwen": 131072,
    # MiniMax — M3 is 1M; M2.x is 204,800. https://platform.minimax.io/docs/api-reference/text-chat-openai
    "minimax-m3": 1000000,
    "minimax": 204800,
    # GLM — 5.2/5.3 are 1M (5.2 verified empirically at 789K on api.z.ai);
    # older GLM (5, 5.1, 5-turbo) ~202K.
    "glm-5.2": 1_048_576,
    "glm-5.2:free": 256_000,      # OpenRouter free variant is capped; longer key wins
    "glm-5.3": 1_048_576,
    "glm": 202752,
    # xAI — /v1/models returns no context_length, so these prevent probe-down
    # on api.x.ai custom providers. grok-composer is OAuth-only: 200k usable
    # (the /v1/responses ~262144 input+output budget is a separate limit).
    "grok-composer": 200000,    # grok-composer-2.5-fast (Grok Build CLI)
    "grok-build-latest": 500000,  # alias of grok-4.5 (early access)
    "grok-build": 256000,       # grok-build-0.1
    "grok-code-fast": 256000,   # grok-code-fast-1
    "grok-2-vision": 8192,      # grok-2-vision, -1212, -latest
    "grok-4-fast": 2000000,     # grok-4-fast-(non-)reasoning, also matches -reasoning
    "grok-4.20": 2000000,       # grok-4.20-0309-(non-)reasoning, -multi-agent-0309
    "grok-4.6": 500000,         # grok-4.6 — 500K context (OpenRouter / docs.x.ai)
    "grok-4.5": 500000,         # grok-4.5, grok-4.5-latest — 500K context per docs.x.ai
    "grok-4.3": 1000000,        # grok-4.3, grok-4.3-latest — 1M context per docs.x.ai
    "grok-4": 256000,           # grok-4, grok-4-0709
    "grok-3": 131072,           # grok-3, grok-3-mini, grok-3-fast, grok-3-mini-fast
    "grok-2": 131072,           # grok-2, grok-2-1212, grok-2-latest
    "grok": 131072,             # catch-all (grok-beta, unknown grok-*)
    # Kimi — K3 is 1 Mi (matches the endpoint-scoped override); older Kimi 256K.
    "kimi-k3": 1_048_576,
    "kimi": 262144,
    # Upstage Solar — /v1/models returns no context_length; dated variants
    # (solar-pro3-250127) resolve via the family prefix.
    "solar-open2": 262144,  # 256K
    "solar-pro3": 131072,
    "solar-pro2": 65536,
    "solar-mini": 32768,
    # Tencent Hunyuan (262144 = 256 × 1024, aligned with OpenRouter live metadata)
    "hy4-preview": 1_048_576,
    "hy3-preview": 262144,
    "hy3": 262144,
    # "Ox Alpha" stealth model — OpenCode Zen slug and OpenRouter slug
    "x-preview-f": 1_048_576,
    "ox-alpha": 1_048_576,
    # NVIDIA Nemotron — 128K across sizes except 3.5 Lightning (1M)
    "nemotron-3.5-lightning": 1_000_000,
    "nemotron": 131072,
    # Poolside Laguna 2.1 (covers :free and OpenCode Zen -free slugs)
    "laguna-s-2.1": 262144,
    "laguna-xs-2.1": 262144,
    # Arcee
    "trinity": 262144,
    # OpenRouter
    "elephant": 262144,
    # Hugging Face Inference Providers — model IDs use org/name format
    "Qwen/Qwen3.5-397B-A17B": 131072,
    "Qwen/Qwen3.5-35B-A3B": 131072,
    "deepseek-ai/DeepSeek-V3.2": 65536,
    "moonshotai/Kimi-K2.5": 262144,
    "moonshotai/Kimi-K2.6": 262144,
    "moonshotai/Kimi-K2-Thinking": 262144,
    "MiniMaxAI/MiniMax-M2.5": 204800,
    "XiaomiMiMo/MiMo-V2-Flash": 262144,
    "mimo-v2-pro": 1048576,
    "mimo-v2.5-pro": 1048576,
    "mimo-v2.5": 1048576,
    "mimo-v2-omni": 262144,
    "mimo-v2-flash": 262144,
    "zai-org/GLM-5": 202752,
}

# xAI Grok models that ACCEPT `reasoning.effort` (verified live against
# /v1/responses). Unlisted Grok models still reason natively but 400 on the
# parameter ("Model X does not support parameter reasoningEffort"), so callers
# must send no `reasoning` key rather than a default `medium`.
_GROK_EFFORT_CAPABLE_PREFIXES = (
    "grok-3-mini",
    "grok-4.20-multi-agent",
    "grok-4.3",
    "grok-4.5",  # accepts low/medium/high (default high) but REJECTS "none", unlike grok-4.3
    "grok-4.6",  # same effort dial as grok-4.5
)


def grok_supports_reasoning_effort(model: str) -> bool:
    """Allowlist check (aggregator prefixes like ``x-ai/`` stripped); unknown Grok models get no effort dial."""
    name = (model or "").strip().lower()
    if not name:
        return False
    if "/" in name:
        name = name.rsplit("/", 1)[-1]
    return any(name.startswith(prefix) for prefix in _GROK_EFFORT_CAPABLE_PREFIXES)


def is_grok_46_family(model: str) -> bool:
    """Return whether *model* is a Grok 4.6 family identifier."""
    name = (model or "").strip().lower().replace("_", "-")
    if "/" in name:
        name = name.rsplit("/", 1)[-1]
    return name == "grok-4.6" or name.startswith("grok-4.6-")


_CONTEXT_LENGTH_KEYS = (
    "context_length",
    "context_window",
    "context_size",
    "max_context_length",
    "max_position_embeddings",
    "max_model_len",
    "max_input_tokens",
    "max_sequence_length",
    "max_seq_len",
    "n_ctx_train",
    "n_ctx",
    "ctx_size",
)

_MAX_COMPLETION_KEYS = (
    "max_completion_tokens",
    "max_output_tokens",
    "max_tokens",
)

# Local server hostnames / address patterns
_LOCAL_HOSTS = ("localhost", "127.0.0.1", "::1", "0.0.0.0")
# Docker / Podman / Lima DNS names that resolve to the host machine
_CONTAINER_LOCAL_SUFFIXES = (
    ".docker.internal",
    ".containers.internal",
    ".lima.internal",
)


def _normalize_base_url(base_url: str) -> str:
    return (base_url or "").strip().rstrip("/")


def _auth_headers(api_key: str = "") -> Dict[str, str]:
    token = str(api_key or "").strip()
    if not token:
        return {}
    return {"Authorization": f"Bearer {token}"}


def _is_openrouter_base_url(base_url: str) -> bool:
    return base_url_host_matches(base_url, "openrouter.ai")


def _is_custom_endpoint(base_url: str) -> bool:
    normalized = _normalize_base_url(base_url)
    return bool(normalized) and not _is_openrouter_base_url(normalized)


_URL_TO_PROVIDER: Dict[str, str] = {
    "api.openai.com": "openai",
    "chatgpt.com": "openai",
    "api.anthropic.com": "anthropic",
    "api.z.ai": "zai",
    "open.bigmodel.cn": "zai",
    "api.moonshot.ai": "kimi-coding",
    "api.moonshot.cn": "kimi-coding-cn",
    "api.kimi.com": "kimi-coding",
    "api.stepfun.ai": "stepfun",
    "api.stepfun.com": "stepfun",
    "api.arcee.ai": "arcee",
    "api.minimax": "minimax",
    "dashscope.aliyuncs.com": "alibaba",
    "dashscope-intl.aliyuncs.com": "alibaba",
    "portal.qwen.ai": "qwen-oauth",
    "openrouter.ai": "openrouter",
    "generativelanguage.googleapis.com": "gemini",
    "inference-api.nousresearch.com": "nous",
    "api.deepseek.com": "deepseek",
    "api.githubcopilot.com": "copilot",
    ".githubcopilot.com": "copilot",  # api.enterprise./api.business. Copilot hosts
    "models.github.ai": "copilot",
    # GitHub Models free tier: ~8K per-request cap makes it unusable, but
    # mapping it lets us emit a targeted hint instead of the custom-endpoint path.
    "models.inference.ai.azure.com": "copilot",
    "api.fireworks.ai": "fireworks",
    "opencode.ai": "opencode-go",
    "api.x.ai": "xai",
    "integrate.api.nvidia.com": "nvidia",
    "api.xiaomimimo.com": "xiaomi",
    "xiaomimimo.com": "xiaomi",
    "api.gmi-serving.com": "gmi",
    "api.novita.ai": "novita",
    "tokenhub.tencentmaas.com": "tencent-tokenhub",
    "api.lkeap.cloud.tencent.com": "tencent-tokenplan",
    "ollama.com": "ollama-cloud",
}

# Auto-extend with provider-profile hostnames not already mapped.
try:
    for _pp in _list_providers():
        _host = _pp.get_hostname()
        if _host and _host not in _URL_TO_PROVIDER:
            _URL_TO_PROVIDER[_host] = _pp.name
except Exception:
    pass


def _infer_provider_from_url(base_url: str) -> Optional[str]:
    """models.dev provider name for a base URL (custom endpoints need no explicit provider)."""
    normalized = _normalize_base_url(base_url)
    if not normalized:
        return None
    parsed = urlparse(normalized if "://" in normalized else f"https://{normalized}")
    host = parsed.netloc.lower() or parsed.path.lower()
    for url_part, provider in _URL_TO_PROVIDER.items():
        if url_part in host:
            return provider
    return None


def _lmstudio_server_root(base_url: str) -> str:
    """Return the LM Studio server root for native ``/api/v1`` endpoints."""
    root = _normalize_base_url(base_url).rstrip("/")
    for suffix in ("/api/v1", "/api", "/v1"):
        if root.endswith(suffix):
            root = root[: -len(suffix)].rstrip("/")
            break
    return root


def _is_known_provider_base_url(base_url: str) -> bool:
    return _infer_provider_from_url(base_url) is not None


def _server_root(base_url: str) -> str:
    """Probe root for a local server: IPv4-resolved, ``/v1`` suffix stripped."""
    server_url = _localhost_to_ipv4(base_url.rstrip("/"))
    if server_url.endswith("/v1"):
        server_url = server_url[:-3]
    return server_url


def _longest_key_match(table: Dict[str, int], model_lower: str) -> Optional[Tuple[str, int]]:
    """First ``(key, value)`` whose key is a substring of ``model_lower``, longest key first.

    Longest-first makes specific entries (``gpt-5.4-mini``) win over their
    family catch-all (``gpt-5``); ties keep table order (stable sort).
    """
    for key, value in sorted(table.items(), key=lambda x: len(x[0]), reverse=True):
        if key in model_lower:
            return key, value
    return None


def _ollama_show_context(data: Dict[str, Any], *, gguf_first: bool, minimum: Optional[int] = None) -> Optional[int]:
    """Context length from an Ollama ``/api/show`` payload.

    ``parameters`` -> ``num_ctx`` is the Modelfile override (the RUNTIME window
    Ollama allocates KV cache for); ``model_info.*.context_length`` is the GGUF
    training max, which can exceed num_ctx. Local users control num_ctx, so
    local probes prefer it; hosted Ollama operators may cap num_ctx
    arbitrarily, so hosted probes prefer the GGUF value (``gguf_first``).
    """
    def _num_ctx() -> Optional[int]:
        for line in data.get("parameters", "").split("\n"):
            if "num_ctx" in line:
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        ctx = int(parts[-1])
                    except ValueError:
                        continue
                    if minimum is None or ctx >= minimum:
                        return ctx
        return None

    def _gguf() -> Optional[int]:
        for key, value in data.get("model_info", {}).items():
            if "context_length" in key and isinstance(value, (int, float)):
                ctx = int(value)
                if minimum is None or ctx >= minimum:
                    return ctx
        return None

    for reader in ((_gguf, _num_ctx) if gguf_first else (_num_ctx, _gguf)):
        ctx = reader()
        if ctx is not None:
            return ctx
    return None


# (host, canonical paths, model ids, context) — see _endpoint_scoped_context_length.
_ENDPOINT_SCOPED_CONTEXT = (
    ("api.kimi.com", {"/coding", "/coding/v1"}, {"k3", "kimi-k3", "kimi-k3-cot"}, 1_048_576),
    ("integrate.api.nvidia.com", {"/v1"}, {"deepseek-ai/deepseek-v4-pro"}, 262_144),
)


def _endpoint_scoped_context_length(model: str, base_url: str) -> Optional[int]:
    """Context confirmed for one provider endpoint only (see _ENDPOINT_SCOPED_CONTEXT).

    Kimi Coding serves K3 (aliases kimi-k3, kimi-k3-cot) at 1 Mi only on the
    canonical ``https://api.kimi.com/coding`` host — legacy Moonshot keys do
    not. NVIDIA NIM serves deepseek-v4-pro at 262,144 while DeepSeek's native
    endpoint is 1M; the lower limit stays scoped to NVIDIA.
    """
    normalized = _normalize_base_url(base_url)
    try:
        parsed = urlparse(normalized)
        port = parsed.port
    except ValueError:
        return None
    # Only canonical https://host[:443]/path with no credentials/query/fragment.
    if (
        parsed.scheme.lower() != "https"
        or port not in (None, 443)
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        return None
    host = (parsed.hostname or "").lower()
    path = parsed.path.rstrip("/")
    model_key = model.strip().lower()
    for scoped_host, paths, models, ctx in _ENDPOINT_SCOPED_CONTEXT:
        if host == scoped_host and path in paths and model_key in models:
            return ctx
    return None


def _skip_persistent_context_cache(base_url: str, provider: str) -> bool:
    """Providers whose on-disk context cache must not short-circuit probing.

    LM Studio: loaded context is transient (the user can reload with another
    context_length). Codex OAuth: the window is account/entitlement-specific,
    and a fallback persisted after a transient failure would suppress revalidation.
    """
    return (provider or "").strip().lower() in {"lmstudio", "openai-codex"}


def _maybe_cache_local_context_length(
    model: str,
    base_url: str,
    length: int,
) -> None:
    """Persist a probed local window only at/above MINIMUM_CONTEXT_LENGTH.

    Sub-minimum windows are still returned so agent_init can reject them, but
    must not be blessed into the on-disk cache as valid operating limits.
    """
    if length >= MINIMUM_CONTEXT_LENGTH:
        save_context_length(model, base_url, length)


def _probe_local_context_length(model: str, base_url: str, api_key: str, provider: str) -> Optional[int]:
    """Live local probe; persists a positive result unless the provider opts out of the disk cache."""
    local_ctx = _query_local_context_length(model, base_url, api_key=api_key)
    if local_ctx and local_ctx > 0:
        if not _skip_persistent_context_cache(base_url, provider):
            _maybe_cache_local_context_length(model, base_url, local_ctx)
        return local_ctx
    return None


def _reconcile_local_cached_context_length(
    model: str,
    base_url: str,
    cached: int,
    api_key: str = "",
) -> int:
    """Return *cached* unless a live local probe reports a different limit.

    Operators restart vLLM/Ollama with a new --max-model-len / num_ctx under
    the same model id; a reachable server wins over the disk entry, a failed
    probe keeps it. Sub-minimum live windows invalidate but are not persisted.
    """
    live_ctx = _query_local_context_length(model, base_url, api_key=api_key)
    if live_ctx and live_ctx > 0 and live_ctx != cached:
        if live_ctx < MINIMUM_CONTEXT_LENGTH:
            logger.info(
                "Live local probe for %s@%s reports %s (< minimum %s); "
                "invalidating stale cache — agent init should reject",
                model, base_url, f"{live_ctx:,}", f"{MINIMUM_CONTEXT_LENGTH:,}",
            )
            _invalidate_cached_context_length(model, base_url)
            return live_ctx
        logger.info(
            "Reconciling stale local cache entry %s@%s: %s -> %s (live probe)",
            model, base_url, f"{cached:,}", f"{live_ctx:,}",
        )
        _invalidate_cached_context_length(model, base_url)
        _maybe_cache_local_context_length(model, base_url, live_ctx)
        return live_ctx
    return cached


def is_local_endpoint(base_url: str) -> bool:
    """True for loopback, container-internal DNS, unqualified hosts, RFC-1918,
    link-local and Tailscale CGNAT (so a trusted Ollama box over Tailscale gets
    the same timeout auto-bumps as localhost)."""
    normalized = _normalize_base_url(base_url)
    if not normalized:
        return False
    url = normalized if "://" in normalized else f"http://{normalized}"
    try:
        parsed = urlparse(url)
        host = parsed.hostname or ""
    except Exception:
        return False
    if host in _LOCAL_HOSTS:
        return True
    # Docker / Podman / Lima internal DNS names (e.g. host.docker.internal)
    if any(host.endswith(suffix) for suffix in _CONTAINER_LOCAL_SUFFIXES):
        return True
    # Unqualified hostnames (no dots) are local by definition — Docker
    # Compose service names, /etc/hosts entries, or mDNS names.
    if host and "." not in host:
        return True
    # RFC-1918 private ranges, link-local, and Tailscale CGNAT
    try:
        addr = ipaddress.ip_address(host)
        if addr.is_private or addr.is_loopback or addr.is_link_local:
            return True
        if isinstance(addr, ipaddress.IPv4Address) and addr in _TAILSCALE_CGNAT:
            return True
    except ValueError:
        pass
    # Bare IP that looks like a private range (e.g. 172.26.x.x for WSL)
    # or Tailscale CGNAT (100.64.x.x–100.127.x.x).
    parts = host.split(".")
    if len(parts) == 4:
        try:
            first, second = int(parts[0]), int(parts[1])
        except ValueError:
            return False
        return (
            first == 10
            or (first == 172 and 16 <= second <= 31)
            or (first == 192 and second == 168)
            or (first == 100 and 64 <= second <= 127)
        )
    return False


def _localhost_to_ipv4(url: str) -> str:
    """Rewrite a ``localhost`` HOST to ``127.0.0.1`` in a probe URL.

    Windows dual-stack resolves localhost to ::1 first and pays a ~2s IPv6
    connect timeout when the server only listens on IPv4. Anchored at the
    scheme so an embedded ``?upstream=http://localhost`` is untouched.
    """
    if not url or not isinstance(url, str):
        return url  # non-string values (test doubles, lazy config) pass through
    return re.sub(
        r"^(https?://)localhost(?=[:/]|$)",
        r"\g<1>127.0.0.1",
        url,
        count=1,
    )


def detect_local_server_type(base_url: str, api_key: str = "") -> Optional[str]:
    """Probe known endpoints: "ollama", "lm-studio", "vllm", "llamacpp", or None (TTL-cached)."""
    import httpx

    # IPv4-resolve BEFORE deriving server/LM Studio URLs and the cache lookup,
    # so localhost and 127.0.0.1 share a cache entry.
    normalized = _localhost_to_ipv4(_normalize_base_url(base_url))
    server_url = _server_root(normalized)
    lmstudio_url = _lmstudio_server_root(normalized)

    cached = _endpoint_probe_path_cache.get(server_url)
    if cached is not None:
        ttl = (
            _ENDPOINT_PROBE_TTL_SECONDS
            if cached[0] is not None
            else _ENDPOINT_PROBE_FAILURE_TTL_SECONDS
        )
        if (time.monotonic() - cached[1]) < ttl:
            return cached[0]

    # Blackholed host: skip the waterfall. Deliberately NOT written to the
    # hour-long verdict cache, which would pin "undetected" after it comes back.
    if _endpoint_blackholed(server_url):
        return None

    disk_hit = _local_probe_disk_get("server_type", server_url)
    if isinstance(disk_hit, str):
        _endpoint_probe_path_cache[server_url] = (disk_hit, time.monotonic())
        return disk_hit

    headers = _auth_headers(api_key)

    def _probe_failed(exc: Exception) -> None:
        """Swallow a probe error; on a connect timeout re-raise so the remaining legs are skipped."""
        if _is_connect_timeout(exc):
            _note_endpoint_blackholed(server_url)
            raise exc

    def _lm_studio(client) -> bool:
        return client.get(f"{lmstudio_url}/api/v1/models").status_code == 200

    def _ollama(client) -> bool:
        # LM Studio answers /api/tags with {"error": ...} and status 200, so
        # the body must actually carry "models".
        r = client.get(f"{server_url}/api/tags")
        return r.status_code == 200 and "models" in r.json()

    def _llamacpp(client) -> bool:
        r = client.get(f"{server_url}/v1/props")
        if r.status_code != 200:
            r = client.get(f"{server_url}/props")  # older builds: no /v1 prefix
        return r.status_code == 200 and "default_generation_settings" in r.text

    def _vllm(client) -> bool:
        r = client.get(f"{server_url}/version")
        return r.status_code == 200 and "version" in r.json()

    # Most specific first: LM Studio's native API, then Ollama, llama.cpp, vLLM.
    waterfall = (("lm-studio", _lm_studio), ("ollama", _ollama), ("llamacpp", _llamacpp), ("vllm", _vllm))
    result: Optional[str] = None
    try:
        with httpx.Client(timeout=2.0, headers=headers) as client:
            for name, probe in waterfall:
                try:
                    if probe(client):
                        result = name
                        break
                except Exception as exc:
                    _probe_failed(exc)
    except Exception:
        pass

    if result is not None:
        _endpoint_probe_path_cache[server_url] = (result, time.monotonic())
        _local_probe_disk_put("server_type", server_url, result)
    else:
        # Negative verdict in memory only (never on disk — failures are often transient).
        _endpoint_probe_path_cache[server_url] = (None, time.monotonic())
    return result


def _iter_nested_dicts(value: Any):
    if isinstance(value, dict):
        yield value
        for nested in value.values():
            yield from _iter_nested_dicts(nested)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_nested_dicts(item)


def _coerce_reasonable_int(value: Any, minimum: int = 1024, maximum: int = 10_000_000) -> Optional[int]:
    try:
        if isinstance(value, bool):
            return None
        if isinstance(value, str):
            value = value.strip().replace(",", "")
        result = int(value)
    except (TypeError, ValueError):
        return None
    if minimum <= result <= maximum:
        return result
    return None


def _extract_first_int(payload: Dict[str, Any], keys: tuple[str, ...]) -> Optional[int]:
    keyset = {key.lower() for key in keys}
    for mapping in _iter_nested_dicts(payload):
        for key, value in mapping.items():
            if str(key).lower() not in keyset:
                continue
            coerced = _coerce_reasonable_int(value)
            if coerced is not None:
                return coerced
    return None


def _extract_flat_context_length(payload: Dict[str, Any]) -> Optional[int]:
    """Top-level-only context WINDOW read (no nested walk, so an unrelated nested
    section can't leak a same-named key). ``max_tokens`` is deliberately NOT a
    window key: on OpenAI-compatible passthroughs it is the max OUTPUT."""
    for key in _CONTEXT_LENGTH_KEYS:
        coerced = _coerce_reasonable_int(payload.get(key))
        if coerced is not None:
            return coerced
    return None


def _extract_context_length(payload: Dict[str, Any]) -> Optional[int]:
    return _extract_first_int(payload, _CONTEXT_LENGTH_KEYS)


def _extract_max_completion_tokens(payload: Dict[str, Any]) -> Optional[int]:
    return _extract_first_int(payload, _MAX_COMPLETION_KEYS)


def _context_length_from_model_payload(payload: Dict[str, Any]) -> Optional[int]:
    """Context window from a ``/v1/models`` object: window keys first, ``max_tokens`` last.

    Anthropic-shaped payloads carry both ``max_input_tokens`` (1M window) and
    ``max_tokens`` (128k OUTPUT cap); reading max_tokens first would persist a
    stale window and fire the compressor at 75% of 128k instead of 1M.
    """
    if not isinstance(payload, dict):
        return None
    ctx = _extract_flat_context_length(payload)
    if ctx is not None:
        return ctx
    raw = payload.get("max_tokens")
    if isinstance(raw, (int, float)):
        ivalue = int(raw)
        if ivalue > 0:
            return ivalue
    return None


def _extract_pricing(payload: Dict[str, Any]) -> Dict[str, Any]:
    def _per_token(source: Dict[str, Any], fields: Dict[str, str], scale) -> Dict[str, Any]:
        # Provider $/MTok (or Novita's 1/10_000-$ per M) -> per-token strings so
        # usage_pricing consumes them through the same path as OpenRouter.
        return {
            target: str(scale(float(source[key])))
            for target, key in fields.items()
            if source.get(key) is not None
        }

    novita_fields = {"prompt": "input_token_price_per_m", "completion": "output_token_price_per_m"}
    if any(payload.get(k) is not None for k in novita_fields.values()):
        return _per_token(payload, novita_fields, lambda v: v / 10_000 / 1_000_000)

    # DeepInfra ships pricing under ``metadata.pricing`` in $/MTok.
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else None
    deepinfra_pricing = metadata.get("pricing") if metadata else None
    deepinfra_fields = {"prompt": "input_tokens", "completion": "output_tokens", "cache_read": "cache_read_tokens"}
    if isinstance(deepinfra_pricing, dict) and any(k in deepinfra_pricing for k in deepinfra_fields.values()):
        return _per_token(deepinfra_pricing, deepinfra_fields, lambda v: v / 1_000_000)

    alias_map = {
        "prompt": ("prompt", "input", "input_cost_per_token", "prompt_token_cost"),
        "completion": ("completion", "output", "output_cost_per_token", "completion_token_cost"),
        "request": ("request", "request_cost"),
        "cache_read": ("cache_read", "cached_prompt", "input_cache_read", "cache_read_cost_per_token"),
        "cache_write": ("cache_write", "cache_creation", "input_cache_write", "cache_write_cost_per_token"),
    }
    for mapping in _iter_nested_dicts(payload):
        normalized = {str(key).lower(): value for key, value in mapping.items()}
        if not any(any(alias in normalized for alias in aliases) for aliases in alias_map.values()):
            continue
        pricing: Dict[str, Any] = {}
        for target, aliases in alias_map.items():
            for alias in aliases:
                if alias in normalized and normalized[alias] not in {None, ""}:
                    pricing[target] = normalized[alias]
                    break
        if pricing:
            return pricing
    return {}


def _add_model_aliases(cache: Dict[str, Dict[str, Any]], model_id: str, entry: Dict[str, Any]) -> None:
    cache[model_id] = entry
    if "/" in model_id:
        bare_model = model_id.split("/", 1)[1]
        cache.setdefault(bare_model, entry)


def fetch_model_metadata(force_refresh: bool = False) -> Dict[str, Dict[str, Any]]:
    """Fetch model metadata from OpenRouter (cached for 1 hour)."""
    global _model_metadata_cache, _model_metadata_cache_time

    if not force_refresh and _model_metadata_cache and (time.time() - _model_metadata_cache_time) < _MODEL_CACHE_TTL:
        return _model_metadata_cache

    if not force_refresh:
        disk_age = _model_metadata_disk_cache_age_seconds()
        if disk_age is not None and disk_age < _MODEL_CACHE_TTL:
            disk_cache = _load_model_metadata_disk_cache()
            if disk_cache:
                _model_metadata_cache = disk_cache
                _model_metadata_cache_time = time.time() - disk_age
                return _model_metadata_cache

    try:
        _ensure_requests()
        # (connect, read) tuple: a flat timeout lets urllib3 block per retry
        # stage through proxies that 403 CONNECT, ballooning to minutes.
        response = requests.get(OPENROUTER_MODELS_URL, timeout=(5, 10), verify=_resolve_requests_verify())
        response.raise_for_status()
        data = response.json()

        cache = {}
        for model in data.get("data", []):
            model_id = model.get("id", "")
            entry = {
                "context_length": model.get("context_length", 128000),
                "max_completion_tokens": model.get("top_provider", {}).get("max_completion_tokens", 4096),
                "name": model.get("name", model_id),
                "pricing": model.get("pricing", {}),
            }
            _add_model_aliases(cache, model_id, entry)
            canonical = model.get("canonical_slug", "")
            if canonical and canonical != model_id:
                _add_model_aliases(cache, canonical, entry)

        _model_metadata_cache = cache
        _model_metadata_cache_time = time.time()
        _save_model_metadata_disk_cache(cache)
        logger.debug("Fetched metadata for %s models from OpenRouter", len(cache))
        return cache

    except Exception as e:
        logger.warning("Failed to fetch model metadata from OpenRouter: %s", e)
        if _model_metadata_cache:
            return _model_metadata_cache
        disk_cache = _load_model_metadata_disk_cache()
        if disk_cache:
            _model_metadata_cache = disk_cache
            disk_age = _model_metadata_disk_cache_age_seconds()
            if disk_age is not None:
                _model_metadata_cache_time = time.time() - min(disk_age, _MODEL_CACHE_TTL)
            else:
                _model_metadata_cache_time = time.time() - _MODEL_CACHE_TTL + 1
            return _model_metadata_cache
        return {}


def _endpoint_model_entry(model: Dict[str, Any], model_id: str, context_length: Optional[int]) -> Dict[str, Any]:
    """Cache entry for one ``/models`` item; optional keys are set only when known."""
    entry: Dict[str, Any] = {"name": model.get("name", model_id)}
    if context_length is not None:
        entry["context_length"] = context_length
    max_completion_tokens = _extract_max_completion_tokens(model)
    if max_completion_tokens is not None:
        entry["max_completion_tokens"] = max_completion_tokens
    pricing = _extract_pricing(model)
    if pricing:
        entry["pricing"] = pricing
    return entry


def fetch_endpoint_model_metadata(
    base_url: str,
    api_key: str = "",
    force_refresh: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """Model metadata from an OpenAI-compatible ``/models`` endpoint (cached per base URL)."""
    normalized = _normalize_base_url(base_url)
    if not normalized or _is_openrouter_base_url(normalized):
        return {}
    _ensure_requests()

    if not force_refresh:
        cached = _endpoint_model_metadata_cache.get(normalized)
        cached_at = _endpoint_model_metadata_cache_time.get(normalized, 0)
        if cached is not None and (time.time() - cached_at) < _ENDPOINT_MODEL_CACHE_TTL:
            return cached
        if not is_local_endpoint(normalized):
            memo = _endpoint_disk_cache_get(normalized)
            if memo is not None:
                _endpoint_model_metadata_cache[normalized] = memo
                _endpoint_model_metadata_cache_time[normalized] = time.time()
                return memo

    # Blackholed: return empty WITHOUT caching so it is retried once the entry expires.
    if _endpoint_blackholed(normalized):
        return {}

    candidates = [normalized]
    if normalized.endswith("/v1"):
        alternate = normalized[:-3].rstrip("/")
    else:
        alternate = normalized + "/v1"
    if alternate and alternate not in candidates:
        candidates.append(alternate)

    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    last_error: Optional[Exception] = None

    if is_local_endpoint(normalized):
        try:
            if detect_local_server_type(normalized, api_key=api_key) == "lm-studio":
                server_url = _lmstudio_server_root(normalized)
                response = requests.get(
                    server_url.rstrip("/") + "/api/v1/models",
                    headers=headers,
                    timeout=(5, 10),
                    verify=_resolve_requests_verify(normalized),
                )
                response.raise_for_status()
                payload = response.json()
                cache: Dict[str, Dict[str, Any]] = {}
                for model in payload.get("models", []):
                    if not isinstance(model, dict):
                        continue
                    model_id = model.get("key") or model.get("id")
                    if not model_id:
                        continue
                    context_length = None
                    for inst in model.get("loaded_instances", []) or []:
                        if not isinstance(inst, dict):
                            continue
                        cfg = inst.get("config", {})
                        ctx = cfg.get("context_length") if isinstance(cfg, dict) else None
                        if isinstance(ctx, int) and ctx > 0:
                            context_length = ctx
                            break
                    entry = _endpoint_model_entry(model, model_id, context_length)
                    _add_model_aliases(cache, model_id, entry)
                    alt_id = model.get("id")
                    if isinstance(alt_id, str) and alt_id and alt_id != model_id:
                        _add_model_aliases(cache, alt_id, entry)

                _endpoint_model_metadata_cache[normalized] = cache
                _endpoint_model_metadata_cache_time[normalized] = time.time()
                return cache
        except Exception as exc:
            last_error = exc
            if _is_connect_timeout(exc):
                _note_endpoint_blackholed(normalized)

    for candidate in candidates:
        # A connect timeout condemns the host, not the path.
        if _endpoint_blackholed(normalized):
            break
        # Cache keys stay unrewritten; only the outbound target is IPv4-resolved.
        request_candidate = _localhost_to_ipv4(candidate)
        url = request_candidate.rstrip("/") + "/models"
        response = None
        try:
            response = requests.get(
                url,
                headers=headers,
                timeout=(5, 10),
                verify=_resolve_requests_verify(normalized),
                stream=True,
            )
            if response.status_code in (401, 403):
                logger.debug(
                    "Model metadata probe received HTTP %s from %s; stopping candidate probing",
                    response.status_code,
                    url,
                )
                break
            response.raise_for_status()
            payload = response.json()
            cache: Dict[str, Dict[str, Any]] = {}
            for model in payload.get("data", []):
                if not isinstance(model, dict):
                    continue
                model_id = model.get("id")
                if not model_id:
                    continue
                _add_model_aliases(cache, model_id, _endpoint_model_entry(model, model_id, _extract_context_length(model)))

            # llama.cpp: /props carries the actually allocated context.
            is_llamacpp = any(
                m.get("owned_by") == "llamacpp"
                for m in payload.get("data", []) if isinstance(m, dict)
            )
            if is_llamacpp:
                try:
                    # Try /v1/props first (current llama.cpp); fall back to /props for older builds
                    base = request_candidate.rstrip("/").replace("/v1", "")
                    _verify = _resolve_requests_verify(normalized)
                    props_resp = requests.get(base + "/v1/props", headers=headers, timeout=5, verify=_verify)
                    if not props_resp.ok:
                        props_resp = requests.get(base + "/props", headers=headers, timeout=5, verify=_verify)
                    if props_resp.ok:
                        props = props_resp.json()
                        gen_settings = props.get("default_generation_settings", {})
                        n_ctx = gen_settings.get("n_ctx")
                        model_alias = props.get("model_alias", "")
                        if n_ctx and model_alias and model_alias in cache:
                            cache[model_alias]["context_length"] = n_ctx
                    else:
                        # Router mode: bare /props 400s; read each LOADED
                        # child's granted window via /props?model=. Unloaded
                        # children are skipped — probing could autoload them.
                        native = requests.get(base + "/models", headers=headers, timeout=5, verify=_verify)
                        if native.ok:
                            children = (native.json() or {}).get("data", [])
                            for child in children[:16]:
                                if not isinstance(child, dict):
                                    continue
                                child_id = child.get("id")
                                status = (child.get("status") or {}).get("value")
                                if not child_id or child_id not in cache or status not in ("loaded", "ready"):
                                    continue
                                pr = requests.get(
                                    base + "/v1/props", params={"model": child_id},
                                    headers=headers, timeout=5, verify=_verify)
                                if not pr.ok:
                                    pr = requests.get(
                                        base + "/props", params={"model": child_id},
                                        headers=headers, timeout=5, verify=_verify)
                                if pr.ok:
                                    child_ctx = (pr.json().get("default_generation_settings") or {}).get("n_ctx")
                                    if child_ctx:
                                        cache[child_id]["context_length"] = child_ctx
                except Exception:
                    pass

            _endpoint_model_metadata_cache[normalized] = cache
            _endpoint_model_metadata_cache_time[normalized] = time.time()
            if cache and not is_local_endpoint(normalized):
                _endpoint_disk_cache_put(normalized, cache)
            return cache
        except Exception as exc:
            last_error = exc
            if _is_connect_timeout(exc):
                _note_endpoint_blackholed(normalized)
        finally:
            if response is not None:
                response.close()

    if last_error:
        logger.debug("Failed to fetch model metadata from %s/models: %s", normalized, last_error)
    _endpoint_model_metadata_cache[normalized] = {}
    _endpoint_model_metadata_cache_time[normalized] = time.time()
    return {}


def _resolve_endpoint_context_length(
    model: str,
    base_url: str,
    api_key: str = "",
) -> Optional[int]:
    """Resolve context length from an endpoint's live ``/models`` metadata."""
    endpoint_metadata = fetch_endpoint_model_metadata(base_url, api_key=api_key)
    matched = endpoint_metadata.get(model)
    if not matched:
        if len(endpoint_metadata) == 1:
            matched = next(iter(endpoint_metadata.values()))
        elif model:
            # Substring match; "" would match EVERY key and poison the window.
            for key, entry in endpoint_metadata.items():
                if model in key or key in model:
                    matched = entry
                    break
    if matched:
        context_length = matched.get("context_length")
        if isinstance(context_length, int):
            return context_length
    return None


def _get_context_cache_path() -> Path:
    """Return path to the persistent context length cache file."""
    from hermes_constants import get_hermes_home
    return get_hermes_home() / "context_length_cache.yaml"


def _load_context_cache() -> Dict[str, int]:
    """Load the model+provider -> context_length cache from disk."""
    path = _get_context_cache_path()
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data.get("context_lengths") or {}
    except Exception as e:
        logger.debug("Failed to load context length cache: %s", e)
        return {}


def _context_cache_key(model: str, base_url: str) -> str:
    """Canonical ``model@base_url`` key for the persistent context cache.

    Trailing slashes are stripped so ``http://host/v1`` and
    ``http://host/v1/`` share one entry instead of creating duplicates
    that can go stale independently.
    """
    return f"{model}@{(base_url or '').rstrip('/')}"


def save_context_length(model: str, base_url: str, length: int) -> None:
    """Persist a discovered context length for a model+provider combo.

    Cache key is ``model@base_url`` so the same model name served from
    different providers can have different limits.
    """
    # Never persist non-positive values — a 0 or negative context length
    # is always a bug and would poison the cache, causing downstream
    # `get_model_context_length()` to return 0 (since `0 is not None`).
    if length <= 0:
        logger.warning(
            "Refusing to cache non-positive context length %s -> %s tokens",
            f"{model}@{base_url}", length,
        )
        return
    key = _context_cache_key(model, base_url)
    cache = _load_context_cache()
    if cache.get(key) == length:
        return  # already stored
    cache[key] = length
    path = _get_context_cache_path()
    try:
        # Atomic write (temp file + fsync + os.replace): a plain truncating
        # ``open(path, "w")`` leaves the file empty/partial if the process is
        # killed mid-dump, and the next _load_context_cache() swallows the
        # resulting YAML error and returns {} — silently wiping EVERY cached
        # context length. It also exposes torn reads to a concurrent process
        # reading between truncate and dump-complete.
        atomic_yaml_write(path, {"context_lengths": cache})
        logger.info("Cached context length %s -> %s tokens", key, f"{length:,}")
    except Exception as e:
        logger.debug("Failed to save context length cache: %s", e)


def get_cached_context_length(model: str, base_url: str) -> Optional[int]:
    """Look up a previously discovered context length for model+provider."""
    key = _context_cache_key(model, base_url)
    cache = _load_context_cache()
    hit = cache.get(key)
    if hit is not None:
        return hit
    # Legacy rows written before key normalization may carry a trailing
    # slash — honor them rather than re-probing. Checked regardless of the
    # caller's slash form: the row's shape and the caller's shape can differ
    # in either direction (old slashed row + new normalized config, or the
    # reverse), so probe the literal form and the slashed canonical form.
    for legacy_key in (f"{model}@{base_url}", f"{key}/"):
        if legacy_key != key:
            hit = cache.get(legacy_key)
            if hit is not None:
                return hit
    return None


def _invalidate_cached_context_length(model: str, base_url: str) -> None:
    """Drop a stale cache entry so it gets re-resolved on the next lookup."""
    key = _context_cache_key(model, base_url)
    cache = _load_context_cache()
    # Invalidation must also drop the in-memory TTL probe entries for this
    # pair — otherwise the next resolution inside the TTL window reuses the
    # very value we just declared stale and re-persists it.
    bare = _strip_provider_prefix(model)
    stripped = (base_url or "").rstrip("/")
    _LOCAL_CTX_PROBE_CACHE.pop((bare, stripped), None)
    _LOCAL_CTX_PROBE_CACHE.pop(("ollama_show", bare, stripped), None)
    # Clear every key shape for this pair: canonical, the caller's literal
    # form, and the slashed legacy form — same set get_cached_context_length
    # consults, so a lookup can never resurrect a row invalidation missed.
    stale_keys = {key, f"{model}@{base_url}", f"{key}/"}
    if not any(k in cache for k in stale_keys):
        return
    for k in stale_keys:
        cache.pop(k, None)
    path = _get_context_cache_path()
    try:
        # Atomic write — see save_context_length() for why a plain truncating
        # open() here risks wiping the entire cache on an interrupted dump.
        atomic_yaml_write(path, {"context_lengths": cache})
    except Exception as e:
        logger.debug("Failed to invalidate context length cache entry %s: %s", key, e)


def get_next_probe_tier(current_length: int) -> Optional[int]:
    """Return the next lower probe tier, or None if already at minimum."""
    for tier in CONTEXT_PROBE_TIERS:
        if tier < current_length:
            return tier
    return None


def parse_context_limit_from_error(error_msg: str) -> Optional[int]:
    """Context limit quoted in a provider error ("maximum context length is 32768 tokens"), if any."""
    error_lower = error_msg.lower()
    patterns = [
        r'max_model_len\s*(?:is\s*)?[:=(]?\s*(\d{4,})',  # vLLM: "max_model_len 32768", "=32768", ": 32768", "(32768)", "is 32768"
        r'maximum model length\s*(?:is\s*)?[:=(]?\s*(\d{4,})',  # vLLM alt: "maximum model length 131072", "... is 131072"
        r'(?:max(?:imum)?|limit)\s*(?:context\s*)?(?:length|size|window)?\s*(?:is|of|:)?\s*(\d{4,})',
        r'context\s*(?:length|size|window)\s*(?:is|of|:)?\s*(\d{4,})',
        r'(\d{4,})\s*(?:token)?\s*(?:context|limit)',
        r'>\s*(\d{4,})\s*(?:max|limit|token)',  # "250000 tokens > 200000 maximum"
        r'(\d{4,})\s*(?:max(?:imum)?)\b',  # "200000 maximum"
        # Gemini: "input token count is 32825 but model only supports up to
        # 32768" — anchor on the phrase so the input count isn't captured.
        r'supports?\s+(?:only\s+)?up\s+to\s+(\d{4,})',
    ]
    for pattern in patterns:
        match = re.search(pattern, error_lower)
        if match:
            limit = int(match.group(1))
            # Sanity check: must be a reasonable context length
            if 1024 <= limit <= 10_000_000:
                return limit
    return None


def get_context_length_from_provider_error(
    error_msg: str,
    current_context_length: int,
) -> Optional[int]:
    """Provider-reported limit LOWER than the current window, else None.

    Overflow recovery must not invent a window: when the provider only says
    the input is too long, callers keep the configured length and compress
    rather than stepping down guessed probe tiers.
    """
    parsed_limit = parse_context_limit_from_error(error_msg)
    if parsed_limit is not None and parsed_limit < current_context_length:
        return parsed_limit
    return None


def parse_available_output_tokens_from_error(error_msg: str) -> Optional[int]:
    """Available OUTPUT tokens from a "max_tokens too large" error, or None.

    Distinct from "prompt too long" (input exceeds the window -> compress):
    here input + requested_output > window, so the fix is a smaller max_tokens
    for this call and context_length must NOT be touched. E.g. Anthropic:
    "max_tokens: 32768 > context_window: 200000 - input_tokens: 190000 = available_tokens: 10000" -> 10000.
    """
    error_lower = error_msg.lower()
    if not _any_phrase_group(error_lower, _PARSEABLE_OUTPUT_CAP_SIGNALS):
        return None

    # Direct cap figures, most specific first:
    #   "max_tokens (98304) exceeds model's maximum output tokens (65536)"
    #   "Range of max_tokens should be [1, 65536]"  (upper bound is the cap)
    #   "... = available_tokens: 10000"             (Anthropic)
    #   "200000 - 190000 = 10000"                   (last number after "=")
    for pattern in (
        r'exceeds model(?:\'s)? maximum output tokens\s*\(?\s*(\d+)\s*\)?',
        r'range of max_tokens should be\s*\[\s*\d+\s*,\s*(\d+)\s*\]',
        r'available_tokens[:\s]+(\d+)',
        r'available\s+tokens[:\s]+(\d+)',
        r'=\s*(\d+)\s*$',
    ):
        match = re.search(pattern, error_lower)
        if match and int(match.group(1)) >= 1:
            return int(match.group(1))

    # OpenRouter/Nous format: "maximum context length is N … (A of text input,
    # B of tool input, C in the output)". Available output = ctx - text - tool.
    _m_ctx = re.search(r'maximum context length is (\d+)', error_lower)
    _m_parts = re.search(
        r'\((\d+)\s+of text input,\s*(\d+)\s+of tool input,\s*(\d+)\s+in the output\)',
        error_lower,
    )
    if _m_ctx and _m_parts:
        _available = int(_m_ctx.group(1)) - int(_m_parts.group(1)) - int(_m_parts.group(2))
        if _available >= 1:
            return _available

    # LM Studio / llama.cpp: window in tokens, prompt in CHARACTERS. ~3
    # chars/token over-reserves the input so the retried cap stays inside the window.
    _m_ctx_tok = re.search(r'maximum context length is (\d+)\s*token', error_lower)
    _m_chars = re.search(r'prompt contains (\d+)\s*character', error_lower)
    if _m_ctx_tok and _m_chars:
        _ctx = int(_m_ctx_tok.group(1))
        _est_input = (int(_m_chars.group(1)) + 2) // 3
        _available = _ctx - _est_input
        if _available >= 1:
            return _available

    # vLLM: window and prompt both in TOKENS; available = window - input (None
    # when the input alone overflows, so the caller compresses instead).
    # When max_tokens is the BINDING constraint vLLM reports "at least N input
    # tokens" with N == window + 1 - requested_output, so window - N is always
    # requested_output - 1 and each retry walks the cap down by the safety
    # margin without ever fitting. Detect that and halve the cap instead —
    # still strictly below what was rejected, converges in one or two retries.
    _m_vllm_input = re.search(
        r'prompt contains (?:at least )?(\d+)\s*input tokens', error_lower
    )
    if _m_ctx_tok and _m_vllm_input:
        _available = int(_m_ctx_tok.group(1)) - int(_m_vllm_input.group(1))
        _m_requested_out = re.search(r'requested (\d+)\s*output tokens', error_lower)
        if 'at least' in error_lower and _m_requested_out:
            _requested_out = int(_m_requested_out.group(1))
            if _available >= _requested_out - 1:
                # The budget is derived from the constraint, not measured.
                return max(1, _requested_out // 2)
        if _available >= 1:
            return _available

    return None


# Each entry is a phrase group; the group matches when ALL phrases are present.
_OUTPUT_CAP_SIGNALS = (
    ("range of max_tokens should be",),               # DashScope / Alibaba
    ("available_tokens",),                            # Anthropic
    ("available tokens",),
    ("in the output", "maximum context length"),      # OpenRouter / Nous
    ("requested", "output tokens"),                   # LM Studio / llama.cpp
    ("should be",),                                   # generic "max_tokens should be <= N"
    ("less than or equal",),
    ("must be",),
    ("exceeds model", "maximum output tokens"),       # OpenAI-compatible relays
)
_INPUT_OVERFLOW_SIGNALS = (
    "prompt is too long", "prompt too long", "input is too long", "input token",
    "prompt length", "prompt contains", "reduce the length",
)
# Narrower than _OUTPUT_CAP_SIGNALS: only phrasings we can extract a number from.
_PARSEABLE_OUTPUT_CAP_SIGNALS = (
    ("max_tokens", "available_tokens"),               # Anthropic
    ("max_tokens", "available tokens"),
    ("in the output", "maximum context length"),      # OpenRouter / Nous
    # "requested N output tokens" means the OUTPUT cap is the problem (the
    # input itself fits) — reduce max_tokens, don't compress.
    ("maximum context length", "requested", "output tokens"),  # LM Studio / llama.cpp
    # DashScope rejects an over-cap output request with a bounded range whose
    # upper bound IS the real max-output cap: "Range of max_tokens should be [1, 65536]".
    ("range of max_tokens should be",),
    ("exceeds model", "maximum output tokens"),       # "max_tokens (98304) exceeds model's maximum output tokens (65536)"
)


def _any_phrase_group(text: str, groups: tuple) -> bool:
    return any(all(p in text for p in group) for group in groups)


def is_output_cap_error(error_msg: str) -> bool:
    """Yes/no sibling of :func:`parse_available_output_tokens_from_error` for wordings we can't parse a number from.

    An output-cap 400 is deterministic: misclassified as a context overflow it
    death-loops the compressor (same max_tokens, same rejection) until "cannot
    compress further". Signal: talks about max_tokens as a cap/range/limit and
    NOT about the input being too long; when both appear, defer to overflow.
    """
    error_lower = error_msg.lower()
    if not any(p in error_lower for p in ("max_tokens", "max_output_tokens", "max_completion_tokens")):
        return False
    if not _any_phrase_group(error_lower, _OUTPUT_CAP_SIGNALS):
        return False
    # An error that ALSO describes an oversized INPUT is a genuine context
    # overflow that happens to mention max_tokens — compression can fix it.
    return not any(p in error_lower for p in _INPUT_OVERFLOW_SIGNALS)


def _model_id_matches(candidate_id: str, lookup_model: str) -> bool:
    """Exact match, or ``publisher/slug`` (LM Studio native ids) whose slug equals the configured name."""
    return candidate_id == lookup_model or (
        "/" in candidate_id and candidate_id.rsplit("/", 1)[1] == lookup_model
    )


def query_ollama_num_ctx(model: str, base_url: str, api_key: str = "") -> Optional[int]:
    """Ollama ``/api/show`` context (Modelfile num_ctx, else GGUF max); the value to send as ``num_ctx``."""
    import httpx

    bare_model = _strip_provider_prefix(model)
    server_url = _server_root(base_url)

    try:
        server_type = detect_local_server_type(base_url, api_key=api_key)
    except Exception:
        return None
    if server_type != "ollama":
        return None

    _disk_key = f"{server_url}|{bare_model}"
    disk_hit = _local_probe_disk_get("ollama_num_ctx", _disk_key)
    if isinstance(disk_hit, int) and disk_hit > 0:
        return disk_hit

    headers = _auth_headers(api_key)

    try:
        with httpx.Client(timeout=3.0, headers=headers) as client:
            resp = client.post(f"{server_url}/api/show", json={"name": bare_model})
            if resp.status_code != 200:
                return None
            ctx = _ollama_show_context(resp.json(), gguf_first=False)
            if ctx is not None:
                _local_probe_disk_put("ollama_num_ctx", _disk_key, ctx)
                return ctx
    except Exception:
        pass
    return None


def query_ollama_supports_vision(model: str, base_url: str, api_key: str = "") -> Optional[bool]:
    """Return True/False when Ollama ``/api/show`` reports vision support.

    Uses the ``capabilities`` field on Ollama 0.6.0+ and falls back to
    ``model_info.*.vision.block_count`` on older servers. Returns None when
    the server is unreachable, not Ollama, or the model is unknown.
    """
    import httpx

    bare_model = _strip_provider_prefix(model)
    if not bare_model or not base_url:
        return None

    try:
        if detect_local_server_type(base_url, api_key=api_key) != "ollama":
            return None
    except Exception:
        return None

    server_url = _server_root(base_url)
    headers = _auth_headers(api_key)

    try:
        with httpx.Client(timeout=3.0, headers=headers) as client:
            resp = client.post(f"{server_url}/api/show", json={"name": bare_model})
            if resp.status_code != 200:
                return None
            data = resp.json()
    except Exception:
        return None

    caps = data.get("capabilities")
    if isinstance(caps, list):
        if any(str(cap).lower() == "vision" for cap in caps):
            return True
        if caps:
            return False

    model_info = data.get("model_info")
    if isinstance(model_info, dict):
        for key in model_info:
            if "vision.block_count" in str(key).lower():
                return True

    return None


def _query_ollama_api_show(model: str, base_url: str, api_key: str = "") -> Optional[int]:
    """Provider-agnostic Ollama ``/api/show`` context probe (any hostname; non-Ollama servers 404 fast).

    GGUF-first (hosted users can't set num_ctx) — the reverse of
    query_ollama_num_ctx(). Positive results share _LOCAL_CTX_PROBE_CACHE under
    a namespaced key (the two probes can differ for the same (model, url)).
    """
    import time as _time

    cache_key = ("ollama_show", _strip_provider_prefix(model), base_url.rstrip("/"))
    now = _time.monotonic()
    cached = _LOCAL_CTX_PROBE_CACHE.get(cache_key)
    if cached is not None and (now - cached[1]) < _LOCAL_CTX_PROBE_TTL_SECONDS:
        return cached[0]

    result = _query_ollama_api_show_uncached(model, base_url, api_key=api_key)
    if result:  # positive-only — never memoize a failed probe
        _LOCAL_CTX_PROBE_CACHE[cache_key] = (result, now)
    return result


def _query_ollama_api_show_uncached(model: str, base_url: str, api_key: str = "") -> Optional[int]:
    """Uncached body of ``_query_ollama_api_show`` — one POST to ``/api/show``."""
    import httpx

    server_url = _server_root(base_url)
    if _endpoint_blackholed(server_url):
        return None

    headers = _auth_headers(api_key)

    try:
        with httpx.Client(timeout=5.0, headers=headers) as client:
            resp = client.post(f"{server_url}/api/show", json={"name": model})
            if resp.status_code != 200:
                return None
            # Hosted Ollama: the GGUF max is authoritative (the operator may
            # have capped num_ctx arbitrarily).
            ctx = _ollama_show_context(resp.json(), gguf_first=True, minimum=1024)
            if ctx is not None:
                return ctx
    except Exception as exc:
        if _is_connect_timeout(exc):
            _note_endpoint_blackholed(server_url)
    return None


def _model_name_suggests_kimi(model: str) -> bool:
    """Kimi family (``kimi-*``, ``moonshotai/*``) — guard against stale 32K underreports."""
    lower = model.lower()
    return lower.startswith("kimi") or "moonshot" in lower


def _model_name_suggests_minimax_m3(model: str) -> bool:
    """MiniMax M3 on any surface — models.dev underreport guard and agent_runtime_helpers cache-control gating."""
    return "minimax-m3" in model.lower()


# Catalog keys added AFTER the model was reachable via a shorter catch-all (or
# the 256K fallback): older builds persisted that smaller value and the step-1
# cache hit would pin it forever. A cached value at or below what the old path
# could produce is dropped and re-resolved. Only list keys whose catalog value
# is STRICTLY ABOVE every shorter matching key and the 256K fallback — the
# threshold is inferred from those shorter keys.
_PRE_CATALOG_STALE_KEYS = frozenset({
    "minimax-m3",    # 1M; "minimax" catch-all persisted 204,800
    "grok-4.3",      # 1M; "grok-4" catch-all persisted 256,000
    "grok-4.6",      # 500K; "grok-4" catch-all persisted 256,000
    "grok-4-fast",   # 2M; fell through to the 256K fallback
    "grok-4.20",     # 2M; fell through to the 256K fallback
    "qwen3.6-plus",  # 1M; "qwen" catch-all persisted 131,072
})


def _stale_pre_catalog_cache_entry(model: str, cached: int) -> bool:
    """True when a persisted window is a pre-catalog leftover (see _PRE_CATALOG_STALE_KEYS).

    The model must resolve (longest-key-first, as step 8) to a listed key and
    the cached value must be <= the largest shorter matching catch-all (or the
    256K fallback). Values above that — genuine probe results — are kept.
    """
    model_lower = model.lower()
    matches = [
        (key, value)
        for key, value in DEFAULT_CONTEXT_LENGTHS.items()
        if key in model_lower
    ]
    if not matches:
        return False
    specific_key, specific_value = max(matches, key=lambda kv: len(kv[0]))
    if specific_key not in _PRE_CATALOG_STALE_KEYS:
        return False
    if cached >= specific_value:
        return False
    shorter_values = [v for k, v in matches if len(k) < len(specific_key)]
    threshold = max(shorter_values, default=DEFAULT_FALLBACK_CONTEXT)
    return cached <= threshold


def _model_name_suggests_minimax(model: str) -> bool:
    """MiniMax family (``minimax*``, ``minimaxai/*``) — guard against stale 32K underreports (real: 204.8K)."""
    lower = model.lower()
    return lower.startswith("minimax") or "minimaxai/" in lower


def _model_name_suggests_stale_32k_underreport(model: str) -> bool:
    """Return True for model families known to be wrongly underreported as 32K."""
    return _model_name_suggests_kimi(model) or _model_name_suggests_minimax(model)


def _query_local_context_length(model: str, base_url: str, api_key: str = "") -> Optional[int]:
    """Local-server context probe, short-TTL cached (see _LOCAL_CTX_PROBE_CACHE)."""
    import time as _time

    cache_key = (_strip_provider_prefix(model), base_url.rstrip("/"))
    now = _time.monotonic()
    cached = _LOCAL_CTX_PROBE_CACHE.get(cache_key)
    if cached is not None and (now - cached[1]) < _LOCAL_CTX_PROBE_TTL_SECONDS:
        return cached[0]

    result = _query_local_context_length_uncached(model, base_url, api_key=api_key)
    # Positive-only: a failure during a startup race must not suppress the
    # retry seconds later once the server is up.
    if result:
        _LOCAL_CTX_PROBE_CACHE[cache_key] = (result, now)
    return result


def _query_local_context_length_uncached(model: str, base_url: str, api_key: str = "") -> Optional[int]:
    """Query a local server for the model's context length."""
    import httpx

    model = _strip_provider_prefix(model)

    server_url = _server_root(base_url)
    lmstudio_url = _localhost_to_ipv4(_lmstudio_server_root(base_url))

    if _endpoint_blackholed(server_url):
        return None

    headers = _auth_headers(api_key)

    try:
        server_type = detect_local_server_type(base_url, api_key=api_key)
    except Exception:
        server_type = None

    try:
        with httpx.Client(timeout=3.0, headers=headers) as client:
            # Ollama: num_ctx (runtime window) before the GGUF training max —
            # using the max would let conversations grow past what Ollama
            # allocated and it would silently truncate. Matches query_ollama_num_ctx().
            if server_type == "ollama":
                resp = client.post(f"{server_url}/api/show", json={"name": model})
                if resp.status_code == 200:
                    ctx = _ollama_show_context(resp.json(), gguf_first=False)
                    if ctx is not None:
                        return ctx

            # LM Studio native /api/v1/models (the OpenAI-compat list omits
            # context); loaded-instance config is the runtime value.
            if server_type == "lm-studio":
                resp = client.get(f"{lmstudio_url}/api/v1/models")
                if resp.status_code == 200:
                    data = resp.json()
                    for m in data.get("models", []):
                        if _model_id_matches(m.get("key", ""), model) or _model_id_matches(m.get("id", ""), model):
                            # Prefer loaded instance context (actual runtime value)
                            for inst in m.get("loaded_instances", []):
                                cfg = inst.get("config", {})
                                ctx = cfg.get("context_length")
                                if ctx and isinstance(ctx, (int, float)):
                                    return int(ctx)
                            break

            # llama.cpp /props: the RUNTIME n_ctx, answered by the router even
            # for a not-yet-loaded model (while /v1/models has meta=null), so a
            # lazily-loaded model doesn't fall to a family catch-all.
            if server_type == "llamacpp":
                for props_path in (f"/props?model={model}", "/props"):
                    try:
                        resp = client.get(f"{server_url}{props_path}")
                    except httpx.HTTPError:
                        break
                    if resp.status_code != 200:
                        continue
                    n_ctx = (resp.json().get("default_generation_settings")
                             or {}).get("n_ctx")
                    if isinstance(n_ctx, (int, float)) and n_ctx:
                        return int(n_ctx)

            # LM Studio / vLLM / llama.cpp / Anthropic-compat proxies:
            # try /v1/models/{model}
            resp = client.get(f"{server_url}/v1/models/{model}")
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, dict):
                    ctx = _context_length_from_model_payload(data)
                    if ctx is not None:
                        return ctx

            # Try /v1/models and find the model in the list.
            # Use _model_id_matches to handle "publisher/slug" vs bare "slug".
            resp = client.get(f"{server_url}/v1/models")
            if resp.status_code == 200:
                data = resp.json()
                models_list = data.get("data", [])
                # Match by id; on single-model servers (e.g. llama.cpp) the
                # configured name rarely equals the reported id (a GGUF path),
                # so fall back to the sole model when nothing matches.
                matched = None
                for m in models_list:
                    if not isinstance(m, dict):
                        continue
                    if _model_id_matches(m.get("id", ""), model):
                        matched = m
                        break
                if matched is None and len(models_list) == 1:
                    matched = models_list[0]
                if matched is not None:
                    # Runtime n_ctx (llama.cpp nests it under meta) beats
                    # n_ctx_train, which can exceed what the server allocates.
                    sources = [
                        s
                        for s in (matched, matched.get("meta") or {})
                        if isinstance(s, dict)
                    ]
                    for source in sources:
                        val = source.get("n_ctx")
                        if isinstance(val, (int, float)) and val:
                            return int(val)
                    for source in sources:
                        ctx = _context_length_from_model_payload(source)
                        if ctx is not None:
                            return ctx
    except Exception as exc:
        if _is_connect_timeout(exc):
            _note_endpoint_blackholed(server_url)

    return None


def _normalize_model_version(model: str) -> str:
    """Dots -> dashes so Nous ids (claude-opus-4-6) compare with OpenRouter's (claude-opus-4.6)."""
    return model.replace(".", "-")


def _query_anthropic_context_length(model: str, base_url: str, api_key: str) -> Optional[int]:
    """Anthropic /v1/models max_input_tokens; OAuth tokens (sk-ant-oat*) 401 and are skipped."""
    if not api_key or api_key.startswith("sk-ant-oat"):
        return None
    try:
        base = base_url.rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3]
        url = f"{base}/v1/models?limit=1000"
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        }
        _ensure_requests()
        resp = requests.get(url, headers=headers, timeout=(5, 10), verify=_resolve_requests_verify(base_url))
        if resp.status_code != 200:
            return None
        data = resp.json()
        for m in data.get("data", []):
            if m.get("id") == model:
                ctx = m.get("max_input_tokens")
                if isinstance(ctx, int) and ctx > 0:
                    return ctx
    except Exception as e:
        logger.debug("Anthropic /v1/models query failed: %s", e)
    return None


# Codex OAuth `context_window` values (what Codex enforces — lower than the
# direct API for the same slugs). Fallback when the live probe fails;
# longest-key-first substring match.
_CODEX_OAUTH_CONTEXT_FALLBACK: Dict[str, int] = {
    "gpt-5.1-codex-max": 272_000,
    "gpt-5.1-codex-mini": 272_000,
    "gpt-5.3-codex": 272_000,
    "gpt-5.3-codex-spark": 128_000,  # smaller window; listed so "gpt-5.3-codex" doesn't win
    "gpt-5.2-codex": 272_000,
    "gpt-5.4-mini": 272_000,
    "gpt-5.6-sol": 272_000,
    "gpt-5.6-terra": 272_000,
    "gpt-5.6-luna": 272_000,
    "gpt-daybreak-blue-latest": 272_000,
    "gpt-5.5": 272_000,
    "gpt-5.4": 272_000,
    "gpt-5.2": 272_000,
    "gpt-5": 272_000,
}

# Codex OAuth advertises 272K for these families but ACCEPTS ~900K+ (verified
# live: 911,276 input tokens OK on gpt-5.6-sol; terra/luna/gpt-5.4 completed
# 900,026; gpt-5.5 and gpt-5.4-mini genuinely reject >272K and are NOT
# listed). 900K keeps ≥11K margin under the observed ceiling.
#
# OPT-IN ONLY: the large window is exposed via explicit ``-900k`` picker
# variants; base slugs keep 272K so the cheaper limit is the default (a 900K
# default burned subscription usage for people who never asked). The suffix is
# a Hermes-side alias stripped before the wire (strip_codex_context_variant_suffix).
#
# The bump fires ONLY when the resolved value is exactly the stale 272,000
# advertisement; any other advertised number is trusted and the table is
# inert. ``gpt-5.6`` is a FAMILY PREFIX (``-pro`` slugs aren't routable on
# Codex, so over-matching is moot); ``gpt-5.4`` is EXACT because gpt-5.4-mini
# enforces 272K.
_CODEX_OAUTH_VERIFIED_ABOVE_ADVERTISED_PREFIXES: Dict[str, int] = {
    "gpt-5.6": 900_000,   # sol / terra / luna
}
_CODEX_OAUTH_VERIFIED_ABOVE_ADVERTISED_EXACT: Dict[str, int] = {
    "gpt-5.4": 900_000,
    "gpt-daybreak-blue-latest": 900_000,  # Daybreak/Sol alias
}
_CODEX_OAUTH_STALE_ADVERTISED_CTX = 272_000  # the only advertised value the bump may override

# Picker suffix opting a Codex slug into the verified large window; never sent on the wire.
CODEX_CONTEXT_VARIANT_SUFFIX = "-900k"

# The ONLY bases eligible for ``-900k``: routable, live-verified. No family
# prefixing here — it would synthesize dead ``-pro`` variants and accept
# unprobed descendants. Dated snapshots of the 5.6 bases are allowed.
_CODEX_900K_ELIGIBLE_BASES = frozenset({
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-5.6-luna",
    "gpt-5.4",                    # exact; gpt-5.4-mini enforces 272K
    "gpt-daybreak-blue-latest",   # verified Sol alias
})
_CODEX_900K_SNAPSHOT_BASES = ("gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna")
_CODEX_900K_SNAPSHOT_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _bare_codex_slug(model: Optional[str]) -> str:
    """Lowercased slug without ``vendor/`` (display/auxiliary callers pass ``openai/gpt-5.6-sol-900k``)."""
    return (model or "").strip().lower().rsplit("/", 1)[-1]


def is_codex_900k_base(model: Optional[str]) -> bool:
    """Single source of truth for ``-900k`` eligibility (picker, resolution, /model validation, wire stripping)."""
    slug = _bare_codex_slug(model)
    if not slug or slug.endswith(CODEX_CONTEXT_VARIANT_SUFFIX):
        return False
    if slug in _CODEX_900K_ELIGIBLE_BASES:
        return True
    # Dated snapshots of the routable 5.6 bases (gpt-5.6-sol-2026-07-09).
    for base in _CODEX_900K_SNAPSHOT_BASES:
        if slug.startswith(base + "-") and _CODEX_900K_SNAPSHOT_RE.match(
            slug[len(base) + 1:]
        ):
            return True
    return False


def is_codex_context_variant(model: Optional[str]) -> bool:
    """Suffix AND eligible base — ``gpt-5.5-900k`` is an invalid alias, not a variant."""
    slug = _bare_codex_slug(model)
    if not slug.endswith(CODEX_CONTEXT_VARIANT_SUFFIX):
        return False
    return is_codex_900k_base(slug[: -len(CODEX_CONTEXT_VARIANT_SUFFIX)])


def strip_codex_context_variant_suffix(model: Optional[str]) -> str:
    """Wire-safe slug with a VALID ``-900k`` suffix removed (vendor prefix kept).

    An ineligible alias (``gpt-5.5-900k``) is returned unchanged so it fails
    honestly at the API instead of silently running as a different model.
    """
    raw = (model or "").strip()
    if not raw.lower().endswith(CODEX_CONTEXT_VARIANT_SUFFIX):
        return raw
    base = raw[: -len(CODEX_CONTEXT_VARIANT_SUFFIX)]
    if is_codex_900k_base(base):
        return base
    return raw


def has_codex_context_variant(model_bare: str) -> bool:
    """Picker-side alias of :func:`is_codex_900k_base`."""
    return is_codex_900k_base(model_bare)


def _verified_codex_ctx_for_slug(model_bare: str) -> Optional[int]:
    """Live-verified cap for a VALID ``-900k`` variant only; base slugs and ineligible aliases -> None."""
    slug = _bare_codex_slug(model_bare)
    if not slug.endswith(CODEX_CONTEXT_VARIANT_SUFFIX):
        return None
    base = slug[: -len(CODEX_CONTEXT_VARIANT_SUFFIX)]
    if not is_codex_900k_base(base):
        return None
    exact = _CODEX_OAUTH_VERIFIED_ABOVE_ADVERTISED_EXACT.get(base)
    if exact is not None:
        return exact
    for key, ctx in _CODEX_OAUTH_VERIFIED_ABOVE_ADVERTISED_PREFIXES.items():
        if base == key or base.startswith(key + "-") or base.startswith(key + "."):
            return ctx
    return None


_codex_oauth_context_cache: Dict[str, Tuple[Dict[str, int], float]] = {}
_CODEX_OAUTH_CONTEXT_CACHE_TTL = 3600  # 1 hour


def _codex_oauth_token_fingerprint(access_token: str) -> str:
    """Return a non-secret cache key for a Codex OAuth access token."""
    return hashlib.sha256(access_token.encode("utf-8")).hexdigest()[:16]


def _extract_chatgpt_account_id(access_token: str) -> Optional[str]:
    """``chatgpt_account_id`` from the Codex OAuth JWT, or None on any parse error.

    Without the ``ChatGPT-Account-Id`` header /backend-api/codex/models returns
    ``{"models":[]}`` (HTTP 200) and the probe silently falls back. Mirrors auxiliary_client.py.
    """
    try:
        parts = access_token.split(".")
        if len(parts) < 2:
            return None
        payload_b64 = parts[1] + "=" * (-len(parts[1]) % 4)
        claims = json.loads(base64.urlsafe_b64decode(payload_b64))
        if not isinstance(claims, dict):
            return None
        acct_id = claims.get("https://api.openai.com/auth", {}).get("chatgpt_account_id")
        return acct_id if isinstance(acct_id, str) and acct_id else None
    except Exception:
        return None


def _fetch_codex_oauth_context_lengths_with_source(
    access_token: str,
) -> Tuple[Dict[str, int], bool]:
    """Codex catalogue ``{slug: context_window}`` plus whether it came from HTTP.

    Cached per token fingerprint (windows vary by entitlement; the raw token is
    never a key). An in-process hit reports False: it is not a fresh provider
    confirmation and must not drive persistent writes.
    """
    global _codex_oauth_context_cache
    now = time.time()
    cache_key = _codex_oauth_token_fingerprint(access_token)
    cached = _codex_oauth_context_cache.get(cache_key)
    if cached is not None:
        cached_models, cached_at = cached
        if now - cached_at < _CODEX_OAUTH_CONTEXT_CACHE_TTL:
            return cached_models, False

    headers = {"Authorization": f"Bearer {access_token}"}
    acct_id = _extract_chatgpt_account_id(access_token)
    if acct_id:
        headers["ChatGPT-Account-Id"] = acct_id

    try:
        _ensure_requests()
        resp = requests.get(
            "https://chatgpt.com/backend-api/codex/models?client_version=1.0.0",
            headers=headers,
            timeout=(5, 10),
            verify=_resolve_requests_verify(),
        )
        if resp.status_code != 200:
            logger.debug(
                "Codex /models probe returned HTTP %s; falling back to hardcoded defaults",
                resp.status_code,
            )
            return {}, False
        data = resp.json()
    except Exception as exc:
        logger.debug("Codex /models probe failed: %s", exc)
        return {}, False

    entries = data.get("models", []) if isinstance(data, dict) else []
    result: Dict[str, int] = {}
    for item in entries:
        if not isinstance(item, dict):
            continue
        slug = item.get("slug")
        ctx = item.get("context_window")
        if isinstance(slug, str) and isinstance(ctx, int) and ctx > 0:
            result[slug.strip()] = ctx

    if result:
        _codex_oauth_context_cache[cache_key] = (result, now)
    return result, True


def _resolve_codex_oauth_context_length_with_source(
    model: str, access_token: str = ""
) -> Tuple[Optional[int], str]:
    """``(context_length, source)`` for a Codex OAuth slug.

    source: "live" (fresh authenticated probe — the only one eligible for
    persistent writes), "memory" (same-token in-process hit), "fallback"
    (static table), or "" when unresolved.
    """
    model_bare = _strip_provider_prefix(model).strip()
    if not model_bare:
        return None, ""

    def _apply_verified_bump(ctx: int, source: str) -> Tuple[int, str]:
        """Lift an EXACT stale 272K advertisement to the verified cap for opted-in ``-900k`` variants only."""
        bumped = _verified_codex_ctx_for_slug(model_bare)
        if bumped is not None and ctx == _CODEX_OAUTH_STALE_ADVERTISED_CTX:
            logger.debug(
                "Codex OAuth context for %s: advertised %d raised to "
                "live-verified %d", model_bare, ctx, bumped,
            )
            return bumped, source
        return ctx, source

    # The Codex catalog only knows the base slug (no -900k, no vendor/).
    lookup_bare = _bare_codex_slug(strip_codex_context_variant_suffix(model_bare))

    if access_token:
        live, fresh_probe = _fetch_codex_oauth_context_lengths_with_source(access_token)
        live_source = "live" if fresh_probe else "memory"
        if lookup_bare in live:
            return _apply_verified_bump(live[lookup_bare], live_source)
        # Case-insensitive match in case casing drifts
        model_lower = lookup_bare.lower()
        for slug, ctx in live.items():
            if slug.lower() == model_lower:
                return _apply_verified_bump(ctx, live_source)

    hit = _longest_key_match(_CODEX_OAUTH_CONTEXT_FALLBACK, lookup_bare.lower())
    if hit:
        return _apply_verified_bump(hit[1], "fallback")
    return None, ""


def _resolve_nous_context_length(
    model: str,
    base_url: str = "",
    api_key: str = "",
) -> Tuple[Optional[int], str]:
    """``(context_length, source)`` for a Nous Portal model.

    Portal /v1/models is authoritative ("portal") and may differ from OR (OR
    says 1M for qwen3.6-plus; the portal 262144). Fallback matches OR's
    prefixed ids against the bare Nous id with dot/dash normalisation
    ("openrouter" — callers must NOT persist it, or a portal blip freezes the
    wrong value forever). "" when unresolved.
    """
    if base_url:
        portal_ctx = _resolve_endpoint_context_length(model, base_url, api_key=api_key)
        if portal_ctx is not None:
            return portal_ctx, "portal"

    metadata = fetch_model_metadata()

    def _safe_ctx(or_id: str, entry: dict) -> Optional[int]:
        """Context length minus the known stale 32K underreports (same guard as step 6)."""
        ctx = entry.get("context_length")
        if ctx is None:
            return None
        if ctx <= 32768 and _model_name_suggests_stale_32k_underreport(or_id):
            logger.info(
                "Rejecting OpenRouter metadata context=%s for %r "
                "(known 32K underreport, Nous path); falling through to hardcoded defaults",
                ctx, or_id,
            )
            return None
        return ctx

    if model in metadata:
        ctx = _safe_ctx(model, metadata[model])
        if ctx is not None:
            return ctx, "openrouter"

    normalized = _normalize_model_version(model).lower()

    for or_id, entry in metadata.items():
        bare = or_id.split("/", 1)[1] if "/" in or_id else or_id
        if bare.lower() == model.lower() or _normalize_model_version(bare).lower() == normalized:
            ctx = _safe_ctx(or_id, entry)
            if ctx is not None:
                return ctx, "openrouter"

    model_lower = model.lower()
    for or_id, entry in metadata.items():
        bare = or_id.split("/", 1)[1] if "/" in or_id else or_id
        for candidate, query in [(bare.lower(), model_lower), (_normalize_model_version(bare).lower(), normalized)]:
            if candidate.startswith(query) and (
                len(candidate) == len(query) or candidate[len(query)] in "-:."
            ):
                ctx = _safe_ctx(or_id, entry)
                if ctx is not None:
                    return ctx, "openrouter"

    return None, ""


def _validate_cached_context_length(
    model: str, base_url: str, cached: int, is_bedrock_context: bool, *, api_key: str = "",
) -> Optional[int]:
    """Step 1 of get_model_context_length: accept, repair, or drop a persisted entry.

    Returns the value to use, or None to fall through to live resolution
    (the stale entry is invalidated first where noted). Order matters: a
    value must be rejected as bogus before any provider-specific handling.
    """
    # 0/negative is always a bug (corrupt cache, failed probe, manual edit);
    # `0 is not None` would short-circuit the chain and hand the compressor a
    # zero window, breaking every status-bar and /usage display downstream.
    if cached <= 0:
        logger.warning(
            "Dropping non-positive cache entry %s@%s -> %s; re-resolving",
            model, base_url, cached,
        )
        _invalidate_cached_context_length(model, base_url)
        return None
    # Families stale third-party metadata underreports as 32K (Kimi, MiniMax).
    if cached <= 32768 and _model_name_suggests_stale_32k_underreport(model):
        logger.info(
            "Dropping stale cached context entry %s@%s -> %s (known 32K underreport); "
            "re-resolving via hardcoded defaults",
            model, base_url, f"{cached:,}",
        )
        _invalidate_cached_context_length(model, base_url)
        return None
    # Pre-catalog leftovers: a shorter catch-all (or the 256K fallback) was
    # persisted before the specific entry existed (see _PRE_CATALOG_STALE_KEYS).
    if _stale_pre_catalog_cache_entry(model, cached):
        logger.info(
            "Dropping stale pre-catalog cache entry %s@%s -> %s; "
            "re-resolving via hardcoded defaults",
            model, base_url, f"{cached:,}",
        )
        _invalidate_cached_context_length(model, base_url)
        return None
    # Nous Portal: /v1/models is authoritative. Bypass (don't drop) the cache so
    # step 5b reconciles pre-fix OR-seeded entries without touching the on-disk
    # file when the portal is unreachable; the 300s in-memory endpoint cache
    # makes the per-call cost ~0 within a process.
    if _infer_provider_from_url(base_url) == "nous":
        logger.debug(
            "Bypassing persistent cache for %s@%s (Nous portal authoritative)",
            model, base_url,
        )
        return None
    # Bedrock: the static table is a FLOOR, not an override — probe-derived
    # entries may legitimately exceed it (real window read from Bedrock's
    # length-validation error), so only under-reporting entries are dropped.
    if is_bedrock_context:
        try:
            from agent.bedrock_adapter import get_bedrock_context_length
            bedrock_ctx = get_bedrock_context_length(model)
            if cached < bedrock_ctx:
                logger.info(
                    "Dropping stale Bedrock cache entry %s@%s -> %s; "
                    "using static Bedrock table value %s",
                    model,
                    base_url,
                    f"{cached:,}",
                    f"{bedrock_ctx:,}",
                )
                _invalidate_cached_context_length(model, base_url)
                return bedrock_ctx
        except ImportError:
            pass
        return cached
    if is_local_endpoint(base_url):
        return _reconcile_local_cached_context_length(model, base_url, cached, api_key=api_key)
    return cached


def _resolve_bedrock_context_length(model: str, base_url: str) -> Optional[int]:
    """Step 1b: Bedrock static table + one cached live probe; None when boto3 is absent.

    Bedrock exposes no context window via metadata APIs, so
    get_bedrock_context_length() probes the live endpoint (one fast
    pre-inference length rejection). The result is cached per model — keyed by
    base_url, else a synthetic bedrock:// key so display/offline paths share it.
    """
    try:
        from agent.bedrock_adapter import (
            get_bedrock_context_length,
            resolve_bedrock_region,
        )
    except ImportError:
        return None  # boto3 not installed — fall through to generic resolution
    cache_key_url = base_url or "bedrock://"
    cached = get_cached_context_length(model, cache_key_url)
    if cached is not None:
        return cached
    # Region from the base_url host first, then the standard AWS chain. An
    # empty region disables probing (table only).
    region = ""
    if base_url:
        _m = re.search(r"bedrock-runtime\.([a-z0-9-]+)\.", base_url)
        if _m:
            region = _m.group(1)
    if not region:
        try:
            region = resolve_bedrock_region()
        except Exception:
            region = ""
    ctx = get_bedrock_context_length(model, region=region, probe=bool(region))
    # Only persist probe-derived values (region present); a pure table fallback
    # must not poison the cache against a later successful probe.
    if ctx and region:
        save_context_length(model, cache_key_url, ctx)
    return ctx


def _resolve_custom_endpoint_context_length(model: str, base_url: str, api_key: str, provider: str) -> int:
    """Steps 2-3 for a truly custom endpoint: /models, local probes, Ollama /api/show, catalog, default."""
    context_length = _resolve_endpoint_context_length(model, base_url, api_key=api_key)
    if context_length is not None:
        return context_length
    # Local endpoints: the Modelfile-aware probe first. _query_local_context_length
    # prefers num_ctx, while _query_ollama_api_show returns the GGUF training max
    # first, which can be larger and would create a false-safe compression window.
    if is_local_endpoint(base_url):
        local_ctx = _probe_local_context_length(model, base_url, api_key, provider)
        if local_ctx:
            return local_ctx
    # 2b. Ollama native /api/show (GGUF-first for non-local). Non-Ollama servers 404/405 quickly.
    ctx = _query_ollama_api_show(model, base_url, api_key=api_key)
    if ctx is not None:
        if not _skip_persistent_context_cache(base_url, provider):
            save_context_length(model, base_url, ctx)
        return ctx
    # 3. Probe-down fallback after endpoint-specific detection failed
    logger.info(
        "Could not detect context length for model %r at %s — "
        "defaulting to %s tokens (probe-down). Set model.context_length "
        "in config.yaml to override.",
        model, base_url, f"{DEFAULT_FALLBACK_CONTEXT:,}",
    )
    # 3b. Hardcoded catalog as a last resort: a proxied Anthropic gateway fails
    # the probes above but its model name still matches DEFAULT_CONTEXT_LENGTHS
    # (e.g. "claude-opus-4-8" -> 1M); without this the early return would
    # silently cap context at 256K.
    hit = _longest_key_match(DEFAULT_CONTEXT_LENGTHS, model.lower())
    if hit:
        logger.info(
            "Using hardcoded context length %s for model %r "
            "(custom endpoint, catalog match on %r)",
            f"{hit[1]:,}", model, hit[0],
        )
        return hit[1]
    # Same silent-256K bug class as the step-9 fallback — warn here too.
    _warn_context_length_fallback(model, base_url)
    return DEFAULT_FALLBACK_CONTEXT


def get_model_context_length(
    model: str,
    base_url: str = "",
    api_key: str = "",
    config_context_length: int | None = None,
    provider: str = "",
    custom_providers: list | None = None,
) -> int:
    """Get the context length for a model.

    Resolution order:
    0. Explicit config override (model.context_length or custom_providers per-model)
    0b. model_overrides config (per-provider+model context_window override)
    0c. Endpoint-scoped metadata for models validated on one multiplexed endpoint
    1. Persistent cache (previously discovered via probing).  Nous URLs,
       LM Studio, and Codex OAuth bypass the cache here so their provider
       metadata can be reconciled against the authoritative live source.
    1b. AWS Bedrock static table (must precede custom-endpoint probe)
    2. Active endpoint metadata (/models for explicit custom endpoints)
    3. Local server query (for local endpoints)
    4. Anthropic /v1/models API (API-key users only, not OAuth)
    5. Provider-aware lookups (before generic OpenRouter cache):
       a. Copilot live /models API
       b. Nous: live /v1/models probe first (authoritative), then OR
          cache fallback with suffix/version normalisation.  Only
          portal-derived values are persisted to disk.
       c. Codex OAuth /models probe
       d. GMI /models endpoint
       e. Ollama native /api/show probe (any base_url, provider-agnostic)
       f. models.dev registry lookup (with :cloud/-cloud suffix fallback)
    6. OpenRouter live API metadata (Kimi-family 32k guard)
    7. Local server query (before hardcoded defaults for local endpoints)
    8. Hardcoded defaults (broad family patterns, longest-key-first)
    9. Default fallback (256K)"""
    # 0. Explicit config override — user knows best
    if config_context_length is not None and isinstance(config_context_length, int) and config_context_length > 0:
        return config_context_length

    # 0a. MoA virtual provider: ``model`` is a preset name and ``base_url`` the
    # local virtual endpoint, so every probe would miss. The aggregator is the
    # acting model — resolve its real provider+model (references are advisory
    # and never bound the acting context). Falls through on failure.
    if (provider or "").strip().lower() == "moa":
        try:
            from hermes_cli.config import (
                get_compatible_custom_providers,
                load_config,
            )
            from hermes_cli.moa_config import resolve_moa_preset
            from hermes_cli.runtime_provider import resolve_runtime_provider

            config = load_config()
            effective_custom_providers = custom_providers
            if effective_custom_providers is None:
                effective_custom_providers = get_compatible_custom_providers(config)
            preset = resolve_moa_preset(config.get("moa") or {}, model)
            agg = preset.get("aggregator") or {}
            agg_provider = str(agg.get("provider") or "").strip()
            agg_model = str(agg.get("model") or "").strip()
            if agg_model and agg_provider and agg_provider.lower() != "moa":
                rt = resolve_runtime_provider(requested=agg_provider, target_model=agg_model)
                return get_model_context_length(
                    agg_model,
                    base_url=rt.get("base_url", "") or "",
                    api_key=rt.get("api_key", "") or "",
                    provider=rt.get("provider") or agg_provider,
                    custom_providers=effective_custom_providers,
                )
        except Exception:
            logger.debug("MoA aggregator context-length resolution failed", exc_info=True)

    # 0b. model_overrides: EXPLICIT per-provider+model context_window only.
    # Fill-gap _default entries apply later inside lookup_models_dev_context
    # (step 5f) once the catalog has missed, so a _default can never preempt
    # custom_providers or live probes. Config-read only; never touches the network.
    if provider and model:
        try:
            from agent.models_dev import _override_context_window
            mo_ctx = _override_context_window(provider, model)
            if mo_ctx is not None and mo_ctx > 0:
                return mo_ctx
        except Exception:
            pass  # fall through to other resolution paths

    # 0c. custom_providers per-model override — before any probe, so /model
    # switch and display paths honour a per-model context_length.
    if custom_providers and base_url and model:
        try:
            from hermes_cli.config import get_custom_provider_context_length
            cp_ctx = get_custom_provider_context_length(
                model=model,
                base_url=base_url,
                custom_providers=custom_providers,
            )
            if cp_ctx:
                return cp_ctx
        except Exception:
            pass  # fall through to probing

    # Malformed URLs (e.g. unmatched IPv6 bracket) make urllib.parse raise;
    # treat them as an unknown endpoint so the inference layer reports the
    # configuration error itself.
    if base_url:
        try:
            parsed_base_url = urlparse(_normalize_base_url(base_url))
            _ = parsed_base_url.port
        except ValueError:
            base_url = ""

    # A blank model id would fuzzy-match an arbitrary catalog entry (`"" in key`
    # is vacuously true) and persist it under a junk "@<base_url>" cache key.
    if not str(model or "").strip():
        logger.info(
            "No model id provided for context length resolution — defaulting to %s tokens.",
            f"{DEFAULT_FALLBACK_CONTEXT:,}",
        )
        return DEFAULT_FALLBACK_CONTEXT

    # Bare id for cache lookups and server queries ("local:x" -> "x"; Ollama
    # "model:tag" colons preserved).
    model = _strip_provider_prefix(model)

    # Endpoint-scoped metadata goes AHEAD of the persistent cache so a value
    # learned on a multiplexed provider's other endpoint cannot override the
    # endpoint where the model was actually validated.
    endpoint_context = _endpoint_scoped_context_length(model, base_url)
    if endpoint_context is not None:
        return endpoint_context

    is_bedrock_context = provider == "bedrock" or (
        base_url
        and base_url_hostname(base_url).startswith("bedrock-runtime.")
        and base_url_host_matches(base_url, "amazonaws.com")
    )

    # 1. Persistent cache (LM Studio / Codex OAuth excluded — see
    # _skip_persistent_context_cache).
    if base_url and not _skip_persistent_context_cache(base_url, provider):
        cached = get_cached_context_length(model, base_url)
        if cached is not None:
            validated = _validate_cached_context_length(
                model, base_url, cached, is_bedrock_context, api_key=api_key,
            )
            if validated is not None:
                return validated

    # 1b. AWS Bedrock static table + probe. Must run BEFORE the custom-endpoint
    # step: bedrock-runtime.<region>.amazonaws.com is not in _URL_TO_PROVIDER,
    # so it would be treated as a custom endpoint, fail the /models probe and
    # fall back to the default.
    if is_bedrock_context:
        ctx = _resolve_bedrock_context_length(model, base_url)
        if ctx is not None:
            return ctx

    if provider == "novita" or (base_url and base_url_host_matches(base_url, "api.novita.ai")):
        ctx = _resolve_endpoint_context_length(model, base_url or "https://api.novita.ai/openai/v1", api_key=api_key)
        if ctx is not None:
            if base_url:
                save_context_length(model, base_url, ctx)
            return ctx

    # 2. Live /models for truly custom endpoints. Known providers skip this:
    # their /models may report a provider-imposed limit (Copilot: 128k) rather
    # than the model's window (400k); models.dev is consulted at step 5+.
    if _is_custom_endpoint(base_url) and not _is_known_provider_base_url(base_url):
        return _resolve_custom_endpoint_context_length(model, base_url, api_key, provider)

    # 4. Anthropic /v1/models API (only for regular API keys, not OAuth)
    if provider == "anthropic" or (
        base_url and base_url_hostname(base_url) == "api.anthropic.com"
    ):
        ctx = _query_anthropic_context_length(model, base_url or "https://api.anthropic.com", api_key)
        if ctx:
            return ctx

    # 5. Provider-aware lookups — before the generic OR cache, since the same
    # model has different limits per provider (claude-opus-4.6: 1M on
    # Anthropic, 128K on Copilot). Generic providers are inferred from the URL.
    effective_provider = provider
    if not effective_provider or effective_provider in {"openrouter", "custom"}:
        if base_url:
            inferred = _infer_provider_from_url(base_url)
            if inferred:
                effective_provider = inferred

    # 5a. Copilot live /models — account-specific models (claude-opus-4.6-1m)
    # absent from models.dev, and the provider-enforced limit for the rest.
    if effective_provider in {"copilot", "copilot-acp", "github-copilot"}:
        try:
            from hermes_cli.models import get_copilot_model_context
            ctx = get_copilot_model_context(model, api_key=api_key)
            if ctx:
                return ctx
        except Exception:
            pass  # Fall through to models.dev

    if effective_provider == "nous":
        ctx, source = _resolve_nous_context_length(
            model, base_url=base_url or "", api_key=api_key or ""
        )
        if ctx:
            # Persist ONLY portal-derived values: an OR-fallback value cached
            # on a portal blip would be frozen in by step 1 forever.
            if base_url and source == "portal":
                save_context_length(model, base_url, ctx)
            return ctx
    if effective_provider == "openai-codex":
        # Codex OAuth enforces lower limits than the direct API for the same
        # slug (gpt-5.5: 1.05M vs 272K); its own /models is authoritative.
        codex_ctx, codex_source = _resolve_codex_oauth_context_length_with_source(
            model, access_token=api_key or "",
        )
        if codex_ctx:
            # Only a fresh authenticated catalogue response may be persisted;
            # the static fallback must not poison future probes.
            if base_url and codex_source == "live":
                save_context_length(model, base_url, codex_ctx)
            return codex_ctx
    if effective_provider == "gmi" and base_url:
        # GMI exposes authoritative context_length via /models, but it is not
        # in models.dev yet. Preserve that higher-fidelity endpoint lookup.
        ctx = _resolve_endpoint_context_length(model, base_url, api_key=api_key)
        if ctx is not None:
            return ctx
    # 5e. Ollama native /api/show for any base_url that is not a known
    # non-Ollama provider (OpenAI-compat /v1/models omits context_length; the
    # GGUF model_info is authoritative). Known hosted providers are skipped:
    # the POST always 404s and cost ~300ms on the first-turn critical path.
    if base_url:
        _inferred_for_probe = _infer_provider_from_url(base_url)
        _skip_ollama_probe = (
            _inferred_for_probe is not None
            and "ollama" not in _inferred_for_probe
        )
        if not _skip_ollama_probe:
            ctx = _query_ollama_api_show(model, base_url, api_key=api_key)
            if ctx is not None:
                if not _skip_persistent_context_cache(base_url, provider):
                    save_context_length(model, base_url, ctx)
                return ctx
    # 5f. OpenRouter live /models — authoritative for OR-routed models and
    # refreshed as new slugs ship, so it must win over models.dev (5g) and the
    # family catch-all (8): otherwise a brand-new slug (claude-fable-5, 1M)
    # falls through to the generic "claude": 200K entry.
    if effective_provider == "openrouter":
        metadata = fetch_model_metadata()
        entry = metadata.get(model)
        if entry:
            or_ctx = entry.get("context_length")
            # Guard against the known OpenRouter Kimi-family 32k underreport
            # (same class the hardcoded overrides exist to mitigate).
            if isinstance(or_ctx, int) and or_ctx > 0 and not (
                or_ctx == 32768 and _model_name_suggests_kimi(model)
            ):
                return or_ctx

    if effective_provider:
        from agent.models_dev import lookup_models_dev_context
        ctx = lookup_models_dev_context(effective_provider, model)
        if ctx:
            # MiniMax M3: models.dev reports 512K but actual context is 1M.
            # Prefer hardcoded catalog over stale probe value.
            if _model_name_suggests_minimax_m3(model):
                catalog = DEFAULT_CONTEXT_LENGTHS.get("minimax-m3")
                if catalog and ctx < catalog:
                    logger.info(
                        "Rejecting models.dev context=%s for %r "
                        "(MiniMax-M3 underreport); using hardcoded default %s",
                        ctx, model, f"{catalog:,}",
                    )
                    ctx = catalog
            return ctx

    # 6. OpenRouter metadata, provider-unaware fallback — only when the
    # provider is unknown (OR data is community-maintained).
    if not effective_provider:
        metadata = fetch_model_metadata()
        if model in metadata:
            or_ctx = metadata[model].get("context_length", DEFAULT_FALLBACK_CONTEXT)
            # Guard against stale OpenRouter metadata for model families
            # known to be underreported as 32K.
            if or_ctx == 32768 and _model_name_suggests_stale_32k_underreport(model):
                logger.info(
                    "Rejecting OpenRouter metadata context=%s for %r "
                    "(known 32K underreport); falling through to hardcoded defaults",
                    or_ctx, model,
                )
            else:
                return or_ctx

    # 7. Query local server before hardcoded defaults — model names like
    # ``Hermes-3-Llama-3.1-70B`` substring-match ``llama`` (131072) even when
    # vLLM is running at a lower ``--max-model-len`` (e.g. 32768 on limited VRAM).
    if base_url and is_local_endpoint(base_url):
        local_ctx = _probe_local_context_length(model, base_url, api_key, provider)
        if local_ctx:
            return local_ctx

    # 8. Hardcoded defaults: `key in model` only — the reverse would let
    # "claude-sonnet-4" match "claude-sonnet-4-6" and return 1M.
    hit = _longest_key_match(DEFAULT_CONTEXT_LENGTHS, model.lower())
    if hit:
        return hit[1]

    # 9. Default fallback — warn (deduped per model+endpoint) so small-context
    # models don't silently get 256K.
    _warn_context_length_fallback(model, base_url)
    return DEFAULT_FALLBACK_CONTEXT


async def get_model_context_length_async(
    model: str,
    base_url: str = "",
    api_key: str = "",
    config_context_length: int | None = None,
    provider: str = "",
    custom_providers: list | None = None,
) -> int:
    """get_model_context_length on a worker thread (its blocking HTTP would stall the event loop)."""
    import asyncio
    return await asyncio.to_thread(
        get_model_context_length,
        model,
        base_url=base_url,
        api_key=api_key,
        config_context_length=config_context_length,
        provider=provider,
        custom_providers=custom_providers,
    )


def _is_cjk_token_dense_char(ch: str) -> bool:
    code = ord(ch)
    return (
        0x1100 <= code <= 0x11FF  # Hangul Jamo
        or 0x2E80 <= code <= 0x9FFF  # CJK radicals/ideographs
        or 0xA960 <= code <= 0xA97F  # Hangul Jamo Extended-A
        or 0xAC00 <= code <= 0xD7AF  # Hangul Syllables
        or 0xF900 <= code <= 0xFAFF  # CJK compatibility ideographs
        or 0xFF00 <= code <= 0xFFEF  # Fullwidth forms / halfwidth kana
    )


# Same ranges as _is_cjk_token_dense_char (MUST stay in sync) so dense-char
# counting runs in C rather than a per-char Python loop.
_CJK_DENSE_RE = re.compile(
    "[\u1100-\u11ff"  # Hangul Jamo
    "\u2e80-\u9fff"  # CJK radicals/ideographs
    "\ua960-\ua97f"  # Hangul Jamo Extended-A
    "\uac00-\ud7af"  # Hangul Syllables
    "\uf900-\ufaff"  # CJK compatibility ideographs
    "\uff00-\uffef]"  # Fullwidth forms / halfwidth kana
)


def estimate_tokens_rough(text: str) -> int:
    """Rough token estimate: ceil(chars/4), CJK/Hangul/Kana codepoints ~1 token each.

    Ceiling division keeps short texts from estimating 0 (systematic
    undercount with many short tool results). Runs on every message of every
    preflight walk, so the all-ASCII case must stay O(1): ``str.isascii()`` is
    a flag check on CPython and the CJK count is a single C-level regex pass.
    """
    if not text:
        return 0
    text = str(text)
    if text.isascii():
        return (len(text) + 3) // 4
    dense = len(text) - len(_CJK_DENSE_RE.sub("", text))
    if not dense:  # non-ASCII but no CJK (accents, Cyrillic, emoji)
        return (len(text) + 3) // 4
    sparse = len(text) - dense
    return dense + ((sparse + 3) // 4)


def estimate_messages_tokens_rough(
    messages: List[Dict[str, Any]], *, charge_stale_thinking: bool = True,
) -> int:
    """Rough token estimate for a message list (pre-flight only).

    Images cost a flat ~1500 tokens each (Anthropic's model) rather than their
    base64 length, which would put a 1MB screenshot at ~250K.

    ``charge_stale_thinking=False`` mirrors the tail-budget walk
    (``context_compressor._estimate_msg_budget_tokens``): on routes that don't
    echo stale reasoning, ``reasoning``/``reasoning_content`` ride the wire only
    for the NEWEST assistant turn, so excluding them elsewhere keeps the
    compaction TRIGGER in the same size class as the walk — otherwise
    reasoning-heavy sessions fire preflight forever while the walk finds
    nothing to compact. Default True is the conservative full charge.

    Per-message results are memoized on an identity fingerprint (see
    ``_estimate_message_tokens_cached``); equal fingerprints imply identical
    leaves and structure, hence identical estimates.
    """
    _IMAGE_TOKEN_COST = 1500
    if not charge_stale_thinking:
        messages = _strip_stale_thinking_for_estimate(messages)
    total = 0
    for msg in messages:
        total += _estimate_message_tokens_cached(msg, _IMAGE_TOKEN_COST)
    return total


# Generic thinking-text keys replayed for at most the newest assistant turn
# on non-echo routes — must stay in lockstep with
# ``context_compressor._NEWEST_TURN_ONLY_BUDGET_KEYS``.
_STALE_THINKING_ESTIMATE_KEYS = ("reasoning", "reasoning_content")


def _strip_stale_thinking_for_estimate(
    messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Copy of ``messages`` with stale thinking keys removed (newest kept).

    Shallow stripped copies share the original value objects, so the
    per-message memo still hits for the stripped shape on subsequent walks.
    """
    newest = -1
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if isinstance(m, dict) and m.get("role") == "assistant":
            newest = i
            break
    out: List[Dict[str, Any]] = []
    for i, m in enumerate(messages):
        if (
            i != newest
            and isinstance(m, dict)
            and m.get("role") == "assistant"
            and any(m.get(k) for k in _STALE_THINKING_ESTIMATE_KEYS)
        ):
            m = {
                k: v for k, v in m.items()
                if k not in _STALE_THINKING_ESTIMATE_KEYS
            }
        out.append(m)
    return out


# Per-message token-estimate memo. The estimate is a pure function of the
# message value, so a fingerprint that uniquely determines the value is exact:
#   * strings by ``id()`` AND pinned (strong ref in the entry) — while the
#     entry lives the id can't be reused, and strings are immutable, so
#     id-equality implies value-equality;
#   * ints/floats/bools/None by value; dicts/lists structurally, preserving
#     key order (``str(shadow)`` depends on it); any other type aborts the memo.
# api_messages shallow-copies history dicts each turn but shares the content
# strings, so unchanged messages still hit.
_MSG_TOKENS_CACHE: Dict[Any, Tuple[list, int]] = {}
_MSG_TOKENS_CACHE_MAX = 4096


def _msg_fingerprint(value: Any, pins: list) -> Any:
    if value is None or value is True or value is False:
        return value
    t = type(value)
    if t is str:
        pins.append(value)
        return ("s", id(value))
    if t is int or t is float:
        return ("n", t.__name__, value)
    if t is dict:
        return ("d", tuple(
            (_msg_fingerprint(k, pins), _msg_fingerprint(v, pins))
            for k, v in value.items()
        ))
    if t is list:
        return ("l", tuple(_msg_fingerprint(v, pins) for v in value))
    if t is tuple:
        return ("t", tuple(_msg_fingerprint(v, pins) for v in value))
    raise ValueError("unfingerprintable message value")


def _estimate_message_tokens_cached(msg: Any, image_cost: int) -> int:
    try:
        pins: list = []
        key = _msg_fingerprint(msg, pins)
        hash(key)
    except Exception:
        return (
            _estimate_message_tokens_without_images(msg)
            + _count_image_tokens(msg, image_cost)
        )
    cached = _MSG_TOKENS_CACHE.get(key)
    if cached is not None:
        return cached[1]
    tokens = (
        _estimate_message_tokens_without_images(msg)
        + _count_image_tokens(msg, image_cost)
    )
    _MSG_TOKENS_CACHE[key] = (pins, tokens)
    while len(_MSG_TOKENS_CACHE) > _MSG_TOKENS_CACHE_MAX:
        try:
            _MSG_TOKENS_CACHE.pop(next(iter(_MSG_TOKENS_CACHE)))
        except (StopIteration, KeyError, RuntimeError):
            break
    return tokens


def _count_image_tokens(msg: Dict[str, Any], cost_per_image: int) -> int:
    """Count image-like content parts in a message; return their token cost."""
    count = 0
    content = msg.get("content") if isinstance(msg, dict) else None
    if isinstance(content, list):
        for part in content:
            if not isinstance(part, dict):
                continue
            ptype = part.get("type")
            if ptype in {"image", "image_url", "input_image"}:
                count += 1
    stashed = msg.get("_anthropic_content_blocks") if isinstance(msg, dict) else None
    if isinstance(stashed, list):
        for part in stashed:
            if isinstance(part, dict) and part.get("type") == "image":
                count += 1
    # Multimodal tool results that haven't been converted yet.
    if isinstance(content, dict) and content.get("_multimodal"):
        inner = content.get("content")
        if isinstance(inner, list):
            for part in inner:
                if isinstance(part, dict) and part.get("type") in {"image", "image_url"}:
                    count += 1
    return count * cost_per_image


def _wire_message_shadow(msg: Dict[str, Any]) -> Dict[str, Any]:
    """Shadow of a message holding only what the provider actually receives.

    * ``api_content`` SUBSTITUTES ``content`` (``turn_context.substitute_api_content``
      pops it and overwrites content at every API-bound build), so exactly one
      is counted. The guard is mirrored exactly: only a non-empty STRING sidecar
      on a user/assistant row displaces content; any other shape is discarded on
      the wire, and substituting it would UNDERcount — the dangerous direction
      (compaction fires too late, the turn dies on a hard context error).
    * Base64 images become a placeholder; ``_count_image_tokens`` charges them flat.
    """
    sidecar = msg.get("api_content")
    sidecar_wins = (
        isinstance(sidecar, str)
        and bool(sidecar)
        and msg.get("role") in ("user", "assistant")
    )
    # ``reasoning`` never ships as-is: request builds pop it after optionally
    # promoting it into ``reasoning_content``. When both exist (reasoning-echo
    # providers pin reasoning_content; reasoning holds the same text for the
    # trajectory) counting both inflated estimates up to +53%; keep
    # ``reasoning`` only as the promotion proxy when nothing displaces it.
    _rc = msg.get("reasoning_content")
    drop_reasoning_dup = isinstance(_rc, str) and bool(_rc.strip())
    shadow: Dict[str, Any] = {}
    for k, v in msg.items():
        if k in ("_anthropic_content_blocks", "reasoning_details") or k in PERSISTENCE_ONLY_MESSAGE_FIELDS:
            continue
        if k == "reasoning" and drop_reasoning_dup:
            continue
        if k == "api_content":
            if sidecar_wins:
                shadow["content"] = v
            continue
        if k == "content":
            if sidecar_wins:
                continue
            if isinstance(v, list):
                cleaned = []
                for part in v:
                    if isinstance(part, dict):
                        if part.get("type") in {"image", "image_url", "input_image"}:
                            cleaned.append({"type": part.get("type"), "image": "[stripped]"})
                        else:
                            cleaned.append(part)
                    else:
                        cleaned.append(part)
                shadow[k] = cleaned
            elif isinstance(v, dict) and v.get("_multimodal"):
                shadow[k] = v.get("text_summary", "")
            else:
                shadow[k] = v
        else:
            shadow[k] = v
    return shadow


def _estimate_message_tokens_without_images(msg: Dict[str, Any]) -> int:
    """Token estimate for a message shadow with image payloads stripped."""
    if not isinstance(msg, dict):
        return estimate_tokens_rough(str(msg))
    return estimate_tokens_rough(str(_wire_message_shadow(msg)))


def estimate_request_tokens_rough(
    messages: List[Dict[str, Any]],
    *,
    system_prompt: str = "",
    tools: Optional[List[Dict[str, Any]]] = None,
    charge_stale_thinking: bool = True,
) -> int:
    """Rough token estimate for a full request: system prompt + messages + tool
    schemas (50+ tools add 20-30K on their own). ``charge_stale_thinking``
    is forwarded — pass False when the route provably strips stale thinking
    (``message_sanitization.stale_thinking_reaches_wire``)."""
    total = 0
    if system_prompt:
        total += estimate_tokens_rough(system_prompt)
    if messages:
        if charge_stale_thinking:
            # Positional call: test seams and plugin engines monkeypatch
            # estimate_messages_tokens_rough with (messages)-only signatures.
            total += estimate_messages_tokens_rough(messages)
        else:
            total += estimate_messages_tokens_rough(
                messages, charge_stale_thinking=False
            )
    if tools:
        total += _estimate_tools_tokens_rough(tools)
    return total


# Usage-anchored context accounting: ``usage.prompt_tokens`` is EXACT ground
# truth for everything sent on that request, so anchoring on the last real
# usage shrinks chars/4 estimation to the messages appended since and the
# error self-corrects at every response. Anchor dict fields:
#   prompt_tokens / completion_tokens — provider usage at capture.
#   base_count — len(messages) at capture; the reply for that response is not
#       yet appended and its cost is covered by completion_tokens, so the delta
#       walk skips it when it appears at index base_count.
#   base_last_id / base_last_role — identity of the last message at capture;
#       compaction/splices/rewrites replace it and fall back to full estimation.


def capture_usage_anchor(
    prompt_tokens: Any,
    completion_tokens: Any,
    messages: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Build a usage anchor from provider-reported usage, or None."""
    try:
        pt = int(prompt_tokens or 0)
        ct = int(completion_tokens or 0)
    except (TypeError, ValueError):
        return None
    if pt <= 0 or not isinstance(messages, list):
        return None  # no usable usage (some endpoints omit it) — caller keeps its anchor
    base_count = len(messages)
    last = messages[-1] if base_count else None
    return {
        "prompt_tokens": pt,
        "completion_tokens": max(0, ct),
        "base_count": base_count,
        "base_last_id": id(last) if last is not None else None,
        "base_last_role": last.get("role") if isinstance(last, dict) else None,
    }


def anchored_context_tokens(
    messages: List[Dict[str, Any]],
    anchor: Optional[Dict[str, Any]],
    *,
    charge_stale_thinking: bool = True,
) -> Optional[int]:
    """Anchored prompt+completion tokens plus a rough estimate of ONLY the
    messages appended since; None when the anchor is missing or stale. The
    anchored response's own reply is skipped (already in completion_tokens).
    ``charge_stale_thinking`` is forwarded to the delta estimate."""
    if not isinstance(anchor, dict) or not isinstance(messages, list):
        return None
    base_count = anchor.get("base_count") or 0
    if base_count <= 0 or len(messages) < base_count:
        return None
    base_msg = messages[base_count - 1]
    if id(base_msg) != anchor.get("base_last_id"):
        return None
    base_role = base_msg.get("role") if isinstance(base_msg, dict) else None
    if base_role != anchor.get("base_last_role"):
        return None
    total = int(anchor["prompt_tokens"]) + int(anchor.get("completion_tokens") or 0)
    delta = messages[base_count:]
    if delta:
        first = delta[0]
        if isinstance(first, dict) and first.get("role") == "assistant":
            delta = delta[1:]
    if delta:
        total += estimate_messages_tokens_rough(
            delta, charge_stale_thinking=charge_stale_thinking
        )
    return total


# Keyed by ``id(tools)``; bounded, evicts oldest-first. Avoids repeated
# ``str(tools)`` on large schemas, which stalls GUI event loops under GIL pressure.
_TOOLS_TOKENS_CACHE: dict[int, Tuple[int, str, str, int]] = {}
_TOOLS_TOKENS_CACHE_MAX = 256


def _tool_name_for_cache(tool: Any) -> str:
    if not isinstance(tool, dict):
        return ""
    fn = tool.get("function")
    if isinstance(fn, dict):
        name = fn.get("name")
        if isinstance(name, str):
            return name
    name = tool.get("name")
    return name if isinstance(name, str) else ""


def _estimate_tools_tokens_rough(tools: List[Dict[str, Any]]) -> int:
    if not tools:
        return 0

    key = id(tools)
    n = len(tools)
    first = _tool_name_for_cache(tools[0]) if n else ""
    last = _tool_name_for_cache(tools[-1]) if n else ""

    cached = _TOOLS_TOKENS_CACHE.get(key)
    if cached is not None:
        cached_n, cached_first, cached_last, cached_tokens = cached
        if cached_n == n and cached_first == first and cached_last == last:
            return cached_tokens

    # Sum the major schema fields (descriptions + parameters dominate).
    total_chars = 0
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function")
        if isinstance(fn, dict):
            name = fn.get("name") or ""
            desc = fn.get("description") or ""
            params = fn.get("parameters") or {}
        else:
            name = tool.get("name") or ""
            desc = tool.get("description") or ""
            params = tool.get("parameters") or {}

        if isinstance(name, str):
            total_chars += len(name)
        if isinstance(desc, str):
            total_chars += len(desc)
        try:  # JSON is closer to wire size than repr()
            total_chars += len(json.dumps(params, ensure_ascii=False, separators=(",", ":")))
        except Exception:
            total_chars += len(str(params))

    tokens = (total_chars + 3) // 4
    if len(_TOOLS_TOKENS_CACHE) >= _TOOLS_TOKENS_CACHE_MAX:
        _TOOLS_TOKENS_CACHE.pop(next(iter(_TOOLS_TOKENS_CACHE)), None)
    _TOOLS_TOKENS_CACHE[key] = (n, first, last, tokens)
    return tokens
