"""Canonical model catalogs and lightweight validation helpers."""

from __future__ import annotations

import copy
import json
import logging
import os
import re
import threading
import urllib.parse
import urllib.request
import urllib.error
import time
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from typing import TypeGuard

from hermes_cli import __version__ as _HERMES_VERSION
from hermes_cli.urllib_security import open_credentialed_url
from hermes_cli.models_catalog_static import (  # noqa: F401  (re-exported; tests patch hermes_cli.models.<name>)
    CANONICAL_PROVIDERS,
    OPENROUTER_MODELS,
    PREFERRED_SILENT_DEFAULT_MODEL,
    PROVIDER_GROUPS,
    ProviderEntry,
    VERCEL_AI_GATEWAY_MODELS,
    _AGGREGATOR_PROVIDERS,
    _AZURE_FOUNDRY_RESPONSES_PREFIXES,
    _BORROWED_MODEL_PROVIDERS,
    _COPILOT_MODEL_ALIASES,
    _KEYLESS_STABLE_CACHE_PROVIDERS,
    _LIVE_FIRST_PICKER_PROVIDERS,
    _MODELS_DEV_PREFERRED,
    _OPENAI_FAST_MODE_PREFIXES,
    _OPENROUTER_VARIANT_SUFFIXES,
    _PROVIDER_ALIASES,
    _PROVIDER_LABELS,
    _PROVIDER_MODELS,
    _PROVIDER_RETIRED_ALIASES,
    _SILENT_DEFAULT_PROVIDERS,
    _SLUG_TO_GROUP,
    _XAI_CURATED_EXTRAS,
    _XAI_STATIC_FALLBACK,
    _XAI_TOP_MODEL,
    _codex_curated_models,
    _xai_curated_models,
    _xai_finalize_catalog,
    _xai_merge_curated_extras,
    _xai_promote_top,
    group_providers,
    provider_group_for_slug,
)
from hermes_cli.models_reasoning_caps import (  # noqa: F401  (re-exported; tests patch hermes_cli.models.<name>)
    _OPENROUTER_CATALOG_URL,
    _REASONING_CAPS_DISK_TTL_SECONDS,
    _fetch_reasoning_caps_catalog,
    _hydrate_reasoning_caps_from_disk,
    _load_reasoning_caps_disk,
    _read_reasoning_caps_disk,
    _reasoning_caps_disk_path,
    _save_reasoning_caps_disk,
    _seed_reasoning_caps,
    _warm_reasoning_caps_async,
    nous_catalog_url,
    nous_model_reasoning_capabilities,
    openrouter_model_reasoning_capabilities,
    parse_openrouter_reasoning_capabilities,
    warm_nous_reasoning_caps_async,
    warm_openrouter_reasoning_caps_async,
)
from hermes_cli.models_local import (  # noqa: F401  (re-exported; tests patch hermes_cli.models.<name>)
    LMStudioLoadResult,
    _OLLAMA_CLOUD_CACHE_TTL,
    _OLLAMA_LOCAL_CACHE_MAX_ENTRIES,
    _OLLAMA_LOCAL_MODELS_CACHE,
    _OLLAMA_LOCAL_MODELS_CACHE_TTL,
    _OLLAMA_LOCAL_PROBE_FAILURE_CACHE,
    _OLLAMA_LOCAL_PROBE_FAILURE_TTL,
    _OLLAMA_LOCAL_PROBE_REACHABLE,
    _evict_related_ollama_cache_entries,
    _get_ollama_base_url,
    _get_ollama_native_headers,
    _get_ollama_request_headers,
    _lmstudio_fetch_raw_models,
    _lmstudio_request_headers,
    _lmstudio_server_root,
    _load_ollama_cloud_cache,
    _normalize_openai_base_url,
    _ollama_cloud_cache_path,
    _ollama_local_catalog,
    _ollama_probe_cache_key,
    _remember_ollama_cache,
    _root_for_ollama_native_api,
    _same_ollama_native_root,
    _save_ollama_cloud_cache,
    _strip_ollama_cloud_suffix,
    ensure_lmstudio_model_loaded,
    fetch_lmstudio_models,
    fetch_ollama_cloud_models,
    fetch_ollama_local_models,
    lmstudio_model_reasoning_options,
    ollama_model_supports_thinking,
    probe_lmstudio_models,
    probe_ollama_local_models,
    should_use_ollama_native_catalog,
)
from hermes_cli.models_pricing import (  # noqa: F401  (re-exported; tests patch hermes_cli.models.<name>)
    _DEFAULT_NOUS_INFERENCE_BASE,
    _FAILED_CATALOG_TTL_SECONDS,
    _NOUS_CATALOG_TTL_SECONDS,
    _NOUS_POLICY_APPEND_MAX,
    _PRICING_AUTH_KEY_PREFIX,
    _cache_catalog,
    _cached_catalog,
    _fetch_deepinfra_pricing,
    _fetch_novita_pricing,
    _fireworks_pricing_from_models_dev,
    _format_price_per_mtok,
    _pricing_auth_fingerprint,
    _pricing_cache,
    _pricing_cache_retry_after,
    _resolve_nous_pricing_credentials,
    _resolve_openrouter_api_key,
    compute_sale_discount,
    fetch_ai_gateway_pricing,
    fetch_models_with_pricing,
    get_pricing_for_provider,
    nous_policy_allowed_ids,
    peek_cached_pricing,
    restrict_to_nous_policy,
)
from hermes_cli.models_validate import validate_requested_model  # noqa: F401  (re-exported)

logger = logging.getLogger(__name__)

# Identify ourselves so endpoints fronted by Cloudflare's Browser Integrity
# Check (error 1010) don't reject the default ``Python-urllib/*`` signature.
_HERMES_USER_AGENT = f"hermes-cli/{_HERMES_VERSION}"

COPILOT_BASE_URL = "https://api.githubcopilot.com"
COPILOT_MODELS_URL = f"{COPILOT_BASE_URL}/models"
COPILOT_EDITOR_VERSION = "vscode/1.104.1"
COPILOT_REASONING_EFFORTS_GPT5 = ["minimal", "low", "medium", "high"]
COPILOT_REASONING_EFFORTS_O_SERIES = ["low", "medium", "high"]

def _urlopen_model_catalog_request(req: urllib.request.Request, *, timeout: float, ssl_context=None):
    """Open catalog requests without forwarding headers across origins."""
    return open_credentialed_url(req, timeout=timeout, ssl_context=ssl_context)


def _read_json_cache(path: Path, *, errors=Exception) -> Optional[dict]:
    """Load a JSON-object cache file; None when missing, unreadable, or not a dict."""
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except errors:
        return None
    return data if isinstance(data, dict) else None


def _write_json_cache(path: Path, data: Any, **dump_kwargs: Any) -> None:
    """Atomically persist a cache file (creating parents). Raises on failure — callers decide
    whether a failed cache write is worth logging."""
    from utils import atomic_json_write

    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json_write(path, data, **dump_kwargs)


def _custom_provider_ssl_context(base_url: str):
    """Build an ``ssl.SSLContext`` from a custom provider's TLS settings.

    Mirrors the httpx/requests TLS resolution so the urllib ``/models`` probe honors a
    provider's ``ssl_ca_cert`` / ``ssl_verify`` instead of the process-wide
    ``SSL_CERT_FILE``/certifi bundle. Returns None when no per-provider override applies, so the
    caller keeps urllib's default policy.
    """
    if not base_url:
        return None
    try:
        from hermes_cli.config import get_custom_provider_tls_settings

        tls = get_custom_provider_tls_settings(base_url)
        if not tls:
            return None
        import ssl

        if tls.get("ssl_verify") is False:
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            return ctx
        ca = tls.get("ssl_ca_cert")
        if isinstance(ca, str) and ca and os.path.isfile(ca):
            return ssl.create_default_context(cafile=ca)
    except Exception:
        return None  # never break discovery on a TLS-config lookup
    return None


_openrouter_catalog_cache: list[tuple[str, str]] | None = None


_ai_gateway_catalog_cache: list[tuple[str, str]] | None = None


# ---------------------------------------------------------------------------
# Nous Portal free-model helper
# ---------------------------------------------------------------------------
# The Nous Portal models endpoint is the source of truth for which models
# are currently offered (free or paid). We trust whatever it returns and
# surface it to users as-is — no local allowlist filtering.


def _is_model_free(model_id: str, pricing: dict[str, dict[str, str]]) -> bool:
    """Return True if *model_id* has zero-cost prompt AND completion pricing."""
    p = pricing.get(model_id)
    if not p:
        return False
    try:
        return float(p.get("prompt", "1")) == 0 and float(p.get("completion", "1")) == 0
    except (TypeError, ValueError):
        return False


def partition_nous_models_by_tier(
    model_ids: list[str],
    pricing: dict[str, dict[str, str]],
    free_tier: bool,
) -> tuple[list[str], list[str]]:
    """Split Nous models into (selectable, unavailable) based on user tier.

    For free-tier users: only free models are selectable; paid models are returned as unavailable
    (shown grayed out in the menu).
    """
    if not free_tier:
        return (model_ids, [])

    if not pricing:
        return (model_ids, [])  # can't determine, show everything

    selectable: list[str] = []
    unavailable: list[str] = []
    for mid in model_ids:
        if _is_model_free(mid, pricing):
            selectable.append(mid)
        else:
            unavailable.append(mid)
    return (selectable, unavailable)


def _union_with_portal_recommendations(
    tier_key: str,
    curated_ids: list[str],
    pricing: dict[str, dict[str, str]],
    portal_base_url: str,
    *,
    force_refresh: bool,
    synthesize_free_pricing: bool,
) -> tuple[list[str], dict[str, dict[str, str]]]:
    """Append the Portal's ``<tier_key>`` recommendations missing from ``curated_ids``.

    In-repo curated models show first and Portal-only picks follow. Failures (network, parse,
    missing field) are silent and degrade to returning the inputs unchanged — never block the
    picker on a Portal-side hiccup.
    """
    try:
        payload = fetch_nous_recommended_models(portal_base_url, force_refresh=force_refresh)
    except Exception:
        return (list(curated_ids), dict(pricing))

    block = payload.get(tier_key) if isinstance(payload, dict) else None
    if not isinstance(block, list) or not block:
        return (list(curated_ids), dict(pricing))
    portal_ids = [name for entry in block if (name := _extract_model_name(entry))]
    if not portal_ids:
        return (list(curated_ids), dict(pricing))

    augmented_pricing = dict(pricing)
    if synthesize_free_pricing:
        for mid in portal_ids:
            if mid not in augmented_pricing:
                augmented_pricing[mid] = {"prompt": "0", "completion": "0"}

    seen = set(curated_ids)
    return (list(curated_ids) + [mid for mid in portal_ids if mid not in seen], augmented_pricing)


def union_with_portal_free_recommendations(
    curated_ids: list[str],
    pricing: dict[str, dict[str, str]],
    portal_base_url: str = "",
    *,
    force_refresh: bool = False,
) -> tuple[list[str], dict[str, dict[str, str]]]:
    """Augment curated list + pricing with the Portal's ``freeRecommendedModels`` (Portal-only free
    picks get a synthetic $0 pricing entry so tier partitioning sees them as free)."""
    return _union_with_portal_recommendations(
        "freeRecommendedModels", curated_ids, pricing, portal_base_url,
        force_refresh=force_refresh, synthesize_free_pricing=True,
    )


def union_with_portal_paid_recommendations(
    curated_ids: list[str],
    pricing: dict[str, dict[str, str]],
    portal_base_url: str = "",
    *,
    force_refresh: bool = False,
) -> tuple[list[str], dict[str, dict[str, str]]]:
    """Augment curated list with the Portal's ``paidRecommendedModels``. ``pricing`` is left
    untouched — we deliberately do NOT synthesize pricing entries for paid models."""
    return _union_with_portal_recommendations(
        "paidRecommendedModels", curated_ids, pricing, portal_base_url,
        force_refresh=force_refresh, synthesize_free_pricing=False,
    )


# ---------------------------------------------------------------------------
# TTL cache for free-tier detection — avoids repeated API calls within a
# session while still picking up upgrades quickly.
# ---------------------------------------------------------------------------
_FREE_TIER_CACHE_TTL: int = 180  # seconds (3 minutes)
_free_tier_cache: tuple[bool, float] | None = None  # (result, timestamp)


def check_nous_free_tier(*, force_fresh: bool = False) -> bool:
    """Check if the current Nous Portal user is on a free (unpaid) tier.

    Results are cached for ``_FREE_TIER_CACHE_TTL`` seconds to avoid hitting the Portal API on every
    call. The cache is short-lived so that an account upgrade is reflected within a few minutes.

    Returns True only when entitlement is known to be free. Unknown/error states return False so
    this compatibility wrapper does not block users.
    """
    global _free_tier_cache
    now = time.monotonic()
    if not force_fresh and _free_tier_cache is not None:
        cached_result, cached_at = _free_tier_cache
        if now - cached_at < _FREE_TIER_CACHE_TTL:
            return cached_result

    try:
        from hermes_cli.nous_account import get_nous_portal_account_info

        account_info = get_nous_portal_account_info(force_fresh=force_fresh)
        result = account_info.is_free_tier
        _free_tier_cache = (result, now)
        return result
    except Exception:
        _free_tier_cache = (False, now)
        return False  # default to paid on error — don't block users


# ---------------------------------------------------------------------------
# Nous Portal recommended models
#
# The Portal publishes a curated list of suggested models (separated into
# paid and free tiers) plus dedicated recommendations for compaction (text
# summarisation / auxiliary) and vision tasks. We fetch it once per process
# with a TTL cache so callers can ask "what's the best aux model right now?"
# without hitting the network on every lookup.
#
# Shape of the response (fields we care about):
#   {
#     "paidRecommendedModels":     [ {modelName, ...}, ... ],
#     "freeRecommendedModels":     [ {modelName, ...}, ... ],
#     "paidRecommendedCompactionModel":  {modelName, ...} | null,
#     "paidRecommendedVisionModel":      {modelName, ...} | null,
#     "freeRecommendedCompactionModel":  {modelName, ...} | null,
#     "freeRecommendedVisionModel":      {modelName, ...} | null,
#   }
# ---------------------------------------------------------------------------

NOUS_RECOMMENDED_MODELS_PATH = "/api/nous/recommended-models"
_NOUS_RECOMMENDED_CACHE_TTL: int = 600  # seconds (10 minutes)
# (result_dict, timestamp) keyed by portal_base_url so staging vs prod don't collide.
_nous_recommended_cache: dict[str, tuple[dict[str, Any], float]] = {}


def _nous_recommended_disk_path() -> "Path":
    """Disk path for the persisted recommended-models cache."""
    from hermes_constants import get_hermes_home
    return get_hermes_home() / "cache" / "nous_recommended_cache.json"


def _read_nous_recommended_disk(base: str) -> dict[str, Any] | None:
    """Return the last-known-good payload for ``base`` from disk, or None.

    The disk file is a JSON object keyed by portal base URL so staging and prod don't collide:
    ``{"<base>": {"data": {...}, "ts": <epoch_seconds>}}``.
    """
    blob = _read_json_cache(_nous_recommended_disk_path(), errors=(OSError, json.JSONDecodeError))
    entry = (blob or {}).get(base)
    if not isinstance(entry, dict):
        return None
    data = entry.get("data")
    return data if isinstance(data, dict) and data else None


def _write_nous_recommended_disk(base: str, data: dict[str, Any]) -> None:
    """Persist ``data`` as the last-known-good payload for ``base``.

    Merges into any existing per-base map, then writes atomically. Failures are non-fatal (logged at
    debug) — the in-process cache still works.
    """
    if not data:
        return
    path = _nous_recommended_disk_path()
    try:
        blob = _read_json_cache(path, errors=(OSError, json.JSONDecodeError)) or {}
        blob[base] = {"data": data, "ts": time.time()}
        _write_json_cache(path, blob, indent=2)
    except OSError as exc:
        logger.debug("nous recommended-models disk cache write failed: %s", exc)


def fetch_nous_recommended_models(
    portal_base_url: str = "",
    timeout: float = 5.0,
    *,
    force_refresh: bool = False,
) -> dict[str, Any]:
    """Fetch the Nous Portal's curated recommended-models payload.

    Hits ``<portal>/api/nous/recommended-models``. The endpoint is public — no auth is required.
    Results are cached per portal URL for ``_NOUS_RECOMMENDED_CACHE_TTL`` seconds in process; pass
    ``force_refresh=True`` to bypass the in-process cache.

    A successful live fetch is also persisted to a per-base disk cache
    (``$HERMES_HOME/cache/nous_recommended_cache.json``) as last-known-good. Self-heals on the next
    successful fetch.
    """
    base = (portal_base_url or "https://portal.nousresearch.com").rstrip("/")
    now = time.monotonic()
    cached = _nous_recommended_cache.get(base)
    if not force_refresh and cached is not None:
        payload, cached_at = cached
        if now - cached_at < _NOUS_RECOMMENDED_CACHE_TTL:
            return payload

    url = f"{base}{NOUS_RECOMMENDED_MODELS_PATH}"
    try:
        req = urllib.request.Request(
            url,
            headers={"Accept": "application/json"},
        )
        with _urlopen_model_catalog_request(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
        if not isinstance(data, dict):
            data = {}
    except Exception:
        data = {}

    if data:
        # Live fetch succeeded — refresh both cache layers.
        _nous_recommended_cache[base] = (data, now)
        _write_nous_recommended_disk(base, data)
        return data

    # Live fetch failed. Fall back to the last-known-good disk copy so a
    # transient Portal hiccup doesn't drop the recommendations entirely.
    disk = _read_nous_recommended_disk(base)
    if disk:
        _nous_recommended_cache[base] = (disk, now)
        return disk

    _nous_recommended_cache[base] = (data, now)
    return data


def _resolve_nous_portal_url() -> str:
    """Best-effort lookup of the Portal base URL the user is authed against."""
    try:
        from hermes_cli.auth import (
            DEFAULT_NOUS_PORTAL_URL,
            get_provider_auth_state,
        )
        state = get_provider_auth_state("nous") or {}
        portal = str(state.get("portal_base_url") or "").strip()
        if portal:
            return portal.rstrip("/")
        return str(DEFAULT_NOUS_PORTAL_URL).rstrip("/")
    except Exception:
        return "https://portal.nousresearch.com"


def _extract_model_name(entry: Any) -> Optional[str]:
    """Pull the ``modelName`` field from a recommended-model entry, else None."""
    if not isinstance(entry, dict):
        return None
    model_name = entry.get("modelName")
    if isinstance(model_name, str) and model_name.strip():
        return model_name.strip()
    return None


def get_nous_recommended_aux_model(
    *,
    vision: bool = False,
    free_tier: Optional[bool] = None,
    portal_base_url: str = "",
    force_refresh: bool = False,
) -> Optional[str]:
    """Return the Portal's recommended model name for an auxiliary task.

    For paid-tier users we prefer the paid recommendation but gracefully fall back to the free
    recommendation if the Portal returned ``null`` for the paid field (common during the staged
    rollout of new paid models).
    """
    base = portal_base_url or _resolve_nous_portal_url()
    payload = fetch_nous_recommended_models(base, force_refresh=force_refresh)
    if not payload:
        return None

    if free_tier is None:
        try:
            free_tier = check_nous_free_tier()
        except Exception:
            # On any detection error, assume paid — paid users see both fields
            # anyway so this is a safe default that maximises model quality.
            free_tier = False

    if vision:
        paid_key, free_key = "paidRecommendedVisionModel", "freeRecommendedVisionModel"
    else:
        paid_key, free_key = "paidRecommendedCompactionModel", "freeRecommendedCompactionModel"

    # Preference order:
    #   free tier  → free only
    #   paid tier  → paid, then free (if paid field is null)
    candidates = [free_key] if free_tier else [paid_key, free_key]
    for key in candidates:
        name = _extract_model_name(payload.get(key))
        if name:
            return name
    return None


def get_preferred_silent_default_model(provider: str = "openrouter") -> str:
    """Return the silent-default model id — catalog label first, constant second.

    Reads the ``"default": true`` label from the cached remote catalog (never hits the network —
    safe on hot resolution paths), falling back to :data:`PREFERRED_SILENT_DEFAULT_MODEL` when no
    cached manifest exists or the provider block carries no label.
    """
    try:
        from hermes_cli.model_catalog import get_default_model_from_cache
        labeled = get_default_model_from_cache(provider)
        if labeled:
            return labeled
    except Exception:
        pass
    return PREFERRED_SILENT_DEFAULT_MODEL


def pick_silent_default_model(model_ids: list[str], provider: str = "openrouter") -> str:
    """Pick the silent default from an available-models list.

    Returns the catalog-labeled default (see :func:`get_preferred_silent_default_model`) when the
    list carries it, else the first entry, else "". Used by every surface that must choose a model
    on the user's behalf without an interactive picker (GUI onboarding recommended-default, empty-
    model runtime fallback).
    """
    preferred = get_preferred_silent_default_model(provider)
    if preferred in model_ids:
        return preferred
    return model_ids[0] if model_ids else ""


def get_default_model_for_provider(provider: str) -> str:
    """Return a cost-safe default model for a provider, or "" if unknown.

    Used as a NON-INTERACTIVE fallback when a provider is configured but no model was ever selected
    (e.g. ``hermes auth add openai-codex`` without ``hermes model``, or a profile that sets
    ``provider`` with no ``model``).
    """
    models = _PROVIDER_MODELS.get(provider, [])
    if provider in _SILENT_DEFAULT_PROVIDERS:
        preferred = get_preferred_silent_default_model(provider)
        # Trust the preferred default even when the provider has no static
        # catalog (OpenRouter's picker list is fetched live; its curated
        # snapshot carries the default).
        if preferred and (preferred in models or not models):
            return preferred
    return models[0] if models else ""


def _openrouter_model_is_free(pricing: Any) -> bool:
    """Return True when both prompt and completion pricing are zero."""
    if not isinstance(pricing, dict):
        return False
    try:
        return float(pricing.get("prompt", "0")) == 0 and float(pricing.get("completion", "0")) == 0
    except (TypeError, ValueError):
        return False


def _openrouter_model_supports_tools(item: Any) -> bool:
    """Return True when the model's ``supported_parameters`` advertise tool calling.

    hermes-agent is tool-calling-first — every provider path assumes the model can invoke tools.
    Models that don't advertise ``tools`` in their ``supported_parameters`` (e.g. image-only or
    completion-only models) cannot be driven by the agent loop and would fail at the first tool
    call.

    **Permissive when the field is missing.** Some OpenRouter-compatible gateways (Nous Portal,
    private mirrors, older catalog snapshots) don't populate ``supported_parameters`` at all. Treat
    that as "unknown capability → allow" so the picker doesn't silently empty for those users.
    """
    if not isinstance(item, dict):
        return True
    params = item.get("supported_parameters")
    if not isinstance(params, list):
        # Field absent / malformed / None — be permissive.
        return True
    return "tools" in params


# Reasoning-capability cache slots, one set per catalog (OpenRouter, Nous Portal). The logic
# lives in models_reasoning_caps and reads/writes these by name so tests can reset them here.
# ``*_cache``: model id → parsed caps for the process lifetime; ``*_failed_at``: monotonic time
# of the last failed fetch (60s re-fetch suppression); the flags are once-per-process guards.
_openrouter_reasoning_caps_cache: dict[str, Optional[dict[str, Any]]] | None = None
_openrouter_reasoning_caps_failed_at: float | None = None
_openrouter_caps_disk_checked = False
_openrouter_caps_warm_started = False
_nous_reasoning_caps_cache: dict[str, Optional[dict[str, Any]]] | None = None
_nous_reasoning_caps_failed_at: float | None = None
_nous_caps_disk_checked = False
_nous_caps_warm_started = False


from agent.reasoning_effort import clamp_effort as _clamp_effort


def clamp_reasoning_effort_to_supported(
    effort: Optional[str],
    supported_efforts: Optional[list[str]],
) -> Optional[str]:
    """Clamp a requested reasoning effort to a provider's supported levels.

    Thin wrapper over the canonical policy in :func:`agent.reasoning_effort.clamp_effort` (single
    implementation for every transport and provider profile): keep a supported level verbatim,
    otherwise nearest WEAKER supported level (never silently escalate cost), weakest supported level
    when nothing weaker exists, pass through unknown supported-sets and bespoke level names
    unchanged.
    """
    return _clamp_effort(effort, supported_efforts)


def _fetch_live_catalog_index(url: str, timeout: float, opener) -> Optional[tuple[list, dict[str, dict[str, Any]]]]:
    """GET an OpenAI-style ``/models`` listing → ``(raw data array, {id: item})``, or None when the
    endpoint is unreachable or the payload has no ``data`` list."""
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with opener(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode())
    except Exception:
        return None
    live_items = payload.get("data", [])
    if not isinstance(live_items, list):
        return None
    live_by_id: dict[str, dict[str, Any]] = {}
    for item in live_items:
        if not isinstance(item, dict):
            continue
        mid = str(item.get("id") or "").strip()
        if not mid:
            continue
        live_by_id[mid] = item
    return live_items, live_by_id


def fetch_openrouter_models(
    timeout: float = 8.0,
    *,
    force_refresh: bool = False,
) -> list[tuple[str, str]]:
    """Return the curated OpenRouter picker list, refreshed from the live catalog when possible."""
    global _openrouter_catalog_cache

    if _openrouter_catalog_cache is not None and not force_refresh:
        return list(_openrouter_catalog_cache)

    # Prefer the remotely-hosted catalog manifest; fall back to the in-repo snapshot when the
    # manifest is unreachable. Both are curated lists that drive the picker; the OpenRouter live
    # /v1/models filter (tool support, free pricing) is applied on top either way.
    try:
        from hermes_cli.model_catalog import get_curated_openrouter_models
        remote = get_curated_openrouter_models()
    except Exception:
        remote = None
    fallback = list(remote) if remote else list(OPENROUTER_MODELS)

    live = _fetch_live_catalog_index(_OPENROUTER_CATALOG_URL, timeout, _urlopen_model_catalog_request)
    if live is None:
        return list(_openrouter_catalog_cache or fallback)
    live_items, live_by_id = live

    # Free warm-up for the reasoning-capability cache: same payload the caps fetch would pull, so
    # parse it once here and hot-path callers never need their own HTTP round-trip.
    global _openrouter_reasoning_caps_cache
    seeded = _seed_reasoning_caps(_OPENROUTER_CATALOG_URL, live_items)
    if _openrouter_reasoning_caps_cache is None and seeded is not None:
        _openrouter_reasoning_caps_cache = seeded

    curated: list[tuple[str, str]] = []
    silent_default = get_preferred_silent_default_model("openrouter")
    for preferred_id, _ in fallback:
        live_item = live_by_id.get(preferred_id)
        if live_item is None:
            continue
        # Hide models that don't advertise tool-calling support — hermes-agent requires it and
        # selecting one fails at the first tool call.
        if not _openrouter_model_supports_tools(live_item):
            continue
        if preferred_id == silent_default:
            # Keep the silent-default badge through the live refresh so the picker shows which
            # model Hermes lands on when none is selected.
            desc = "default"
        else:
            desc = "free" if _openrouter_model_is_free(live_item.get("pricing")) else ""
        curated.append((preferred_id, desc))

    if not curated:
        return list(_openrouter_catalog_cache or fallback)

    first_id, first_desc = curated[0]
    if not first_desc:
        curated[0] = (first_id, "recommended")
    _openrouter_catalog_cache = curated
    return list(curated)


def model_ids(*, force_refresh: bool = False) -> list[str]:
    """Return just the OpenRouter model-id strings."""
    return [mid for mid, _ in fetch_openrouter_models(force_refresh=force_refresh)]


def get_curated_nous_model_ids() -> list[str]:
    """Return the curated Nous Portal model-id list.

    Prefers the remotely-hosted catalog manifest (published under ``website/static/api/model-
    catalog.json``); falls back to the in-repo snapshot in ``_PROVIDER_MODELS["nous"]`` when the
    manifest is unreachable. Always returns a list (never None).
    """
    try:
        from hermes_cli.model_catalog import get_curated_nous_models
        remote = get_curated_nous_models()
    except Exception:
        remote = None
    if remote:
        return list(remote)
    return list(_PROVIDER_MODELS.get("nous", []))


def _ai_gateway_model_is_free(pricing: Any) -> bool:
    """Return True if an AI Gateway model has $0 input AND output pricing."""
    if not isinstance(pricing, dict):
        return False
    try:
        return float(pricing.get("input", "0")) == 0 and float(pricing.get("output", "0")) == 0
    except (TypeError, ValueError):
        return False


def fetch_ai_gateway_models(
    timeout: float = 8.0,
    *,
    force_refresh: bool = False,
) -> list[tuple[str, str]]:
    """Return the curated AI Gateway picker list, refreshed from the live catalog when possible."""
    global _ai_gateway_catalog_cache

    if _ai_gateway_catalog_cache is not None and not force_refresh:
        return list(_ai_gateway_catalog_cache)

    from hermes_constants import AI_GATEWAY_BASE_URL

    fallback = list(VERCEL_AI_GATEWAY_MODELS)
    live = _fetch_live_catalog_index(f"{AI_GATEWAY_BASE_URL.rstrip('/')}/models", timeout, urllib.request.urlopen)
    if live is None:
        return list(_ai_gateway_catalog_cache or fallback)
    _, live_by_id = live

    curated: list[tuple[str, str]] = []
    for preferred_id, _ in fallback:
        live_item = live_by_id.get(preferred_id)
        if live_item is None:
            continue
        desc = "free" if _ai_gateway_model_is_free(live_item.get("pricing")) else ""
        curated.append((preferred_id, desc))

    if not curated:
        return list(_ai_gateway_catalog_cache or fallback)

    # If the live catalog offers a free Moonshot model, auto-promote it to position #1 as
    # "recommended" — dynamic discovery without a PR.
    free_moonshot = next(
        (
            mid
            for mid, item in live_by_id.items()
            if mid.startswith("moonshotai/")
            and _ai_gateway_model_is_free(item.get("pricing"))
        ),
        None,
    )
    if free_moonshot:
        curated = [(mid, desc) for mid, desc in curated if mid != free_moonshot]
        curated.insert(0, (free_moonshot, "recommended"))
    else:
        first_id, _ = curated[0]
        curated[0] = (first_id, "recommended")

    _ai_gateway_catalog_cache = curated
    return list(curated)


def ai_gateway_model_ids(*, force_refresh: bool = False) -> list[str]:
    """Return just the AI Gateway model-id strings."""
    return [mid for mid, _ in fetch_ai_gateway_models(force_refresh=force_refresh)]


# ---------------------------------------------------------------------------
# Pricing helpers — fetch live pricing from OpenRouter-compatible /v1/models
# ---------------------------------------------------------------------------


# All provider IDs and aliases that are valid for the provider:model syntax.
_KNOWN_PROVIDER_NAMES: set[str] = (
    set(_PROVIDER_LABELS.keys())
    | set(_PROVIDER_ALIASES.keys())
    | {"openrouter", "custom"}
)


def _configured_custom_provider_ids() -> set[str]:
    """Return routable custom-provider IDs configured by the user."""
    ids = {"custom"}
    try:
        from hermes_cli.config import load_config
        from hermes_cli.providers import custom_provider_slug

        config = load_config()
        providers = config.get("providers", {})
        if isinstance(providers, dict):
            for key, entry in providers.items():
                if isinstance(entry, dict):
                    ids.add(custom_provider_slug(str(entry.get("name") or key), str(key)))
        legacy = config.get("custom_providers", [])
        if isinstance(legacy, list):
            for entry in legacy:
                if isinstance(entry, dict):
                    ids.add(custom_provider_slug(str(entry.get("name") or "")))
    except (ImportError, OSError, RuntimeError, TypeError, ValueError, AttributeError):
        pass
    return ids

def list_available_providers() -> list[dict[str, str]]:
    """Return info about all providers the user could use with ``provider:model``.

    Each dict has ``id``, ``label``, and ``aliases``, plus whether valid credentials are
    configured. Derived from :data:`CANONICAL_PROVIDERS`, the single source of truth shared with
    ``hermes model`` and ``/model``.
    """
    # Derive display order from canonical list + custom
    provider_order = [p.slug for p in CANONICAL_PROVIDERS] + ["custom"]

    # Build reverse alias map
    aliases_for: dict[str, list[str]] = {}
    for alias, canonical in _PROVIDER_ALIASES.items():
        aliases_for.setdefault(canonical, []).append(alias)

    result = []
    for pid in provider_order:
        label = _PROVIDER_LABELS.get(pid, pid)
        alias_list = aliases_for.get(pid, [])
        # Check if this provider has credentials available
        has_creds = False
        try:
            from hermes_cli.auth import get_auth_status, has_usable_secret
            if pid == "custom":
                custom_base_url = _get_custom_base_url() or ""
                has_creds = bool(custom_base_url.strip())
            elif pid == "openrouter":
                has_creds = has_usable_secret(os.getenv("OPENROUTER_API_KEY", ""))
            else:
                status = get_auth_status(pid)
                has_creds = bool(status.get("logged_in") or status.get("configured"))
        except Exception:
            pass
        result.append({
            "id": pid,
            "label": label,
            "aliases": alias_list,
            "authenticated": has_creds,
        })
    return result


def parse_model_input(raw: str, current_provider: str) -> tuple[str, str]:
    """Parse ``/model`` input into ``(provider, model)``.

    The colon is only treated as a provider delimiter if the left side is a recognized provider name
    or alias. This avoids misinterpreting model names that happen to contain colons (e.g.
    ``anthropic/claude-3.5-sonnet:beta``).
    """
    stripped = raw.strip()
    colon = stripped.find(":")
    if colon > 0:
        provider_part = stripped[:colon].strip().lower()
        model_part = stripped[colon + 1:].strip()
        if provider_part and model_part and provider_part in _KNOWN_PROVIDER_NAMES:
            if provider_part == "custom":
                lowered = stripped.lower()
                for custom_id in sorted(
                    _configured_custom_provider_ids() - {"custom"},
                    key=len,
                    reverse=True,
                ):
                    prefix = f"{custom_id.lower()}:"
                    if lowered.startswith(prefix):
                        return custom_id, stripped[len(custom_id) + 1 :].strip()
            # Support custom:name:model triple syntax for named custom
            # providers.  ``custom:local:qwen`` → ("custom:local", "qwen").
            # Single colon ``custom:qwen`` → ("custom", "qwen") as before.
            if provider_part == "custom" and ":" in model_part:
                second_colon = model_part.find(":")
                custom_name = model_part[:second_colon].strip()
                actual_model = model_part[second_colon + 1:].strip()
                if custom_name and actual_model:
                    custom_id = f"custom:{custom_name.lower()}"
                    if custom_id in _configured_custom_provider_ids():
                        return (custom_id, actual_model)
                    return ("custom", model_part)
            return (normalize_provider(provider_part), model_part)
    return (current_provider, stripped)


def _get_custom_base_url() -> str:
    """Get the custom endpoint base_url from config.yaml."""
    model_cfg = _get_model_config_dict()
    return str(model_cfg.get("base_url", "")).strip()


def _get_provider_config_dict(provider: str) -> dict[str, Any]:
    """Return config.yaml providers.<provider>, or an empty dict."""
    key = str(provider or "").strip()
    if not key:
        return {}
    try:
        from hermes_cli.config import load_config
        config = load_config()
        providers_cfg = config.get("providers", {})
        if isinstance(providers_cfg, dict):
            entry = providers_cfg.get(key) or providers_cfg.get(key.lower())
            if isinstance(entry, dict):
                return entry
    except (ImportError, OSError, RuntimeError, TypeError, ValueError, AttributeError):
        pass
    return {}


def _get_model_config_dict() -> dict[str, Any]:
    """Return the main model config mapping, or an empty dict."""
    try:
        from hermes_cli.config import load_config
        config = load_config()
        model_cfg = config.get("model", {})
        if isinstance(model_cfg, dict):
            return model_cfg
    except Exception:
        pass
    return {}


def _base_url_looks_like_anthropic_messages(base_url: str) -> bool:
    normalized = str(base_url or "").strip().lower().rstrip("/")
    if not normalized:
        return False
    path = urllib.parse.urlparse(normalized).path.rstrip("/")
    return path.endswith("/anthropic") or path.endswith("/anthropic/v1")


def _anthropic_models_url(base_url: Optional[str] = None) -> str:
    endpoint = str(base_url or "https://api.anthropic.com").strip().rstrip("/")
    if endpoint.endswith("/v1"):
        return endpoint + "/models"
    return endpoint + "/v1/models"


def _provider_keys(provider: str) -> set[str]:
    key = (provider or "").strip().lower()
    normalized = normalize_provider(provider)
    return {k for k in (key, normalized) if k}


def _provider_catalog_names(provider: str) -> tuple[str, ...]:
    """Active picker models plus retired aliases recognized for detection."""
    active = tuple(_PROVIDER_MODELS.get(provider, []))
    retired = _PROVIDER_RETIRED_ALIASES.get(provider, ())
    return active + retired


def _model_in_provider_catalog(name_lower: str, providers: set[str]) -> bool:
    return any(
        name_lower == model.lower()
        for provider in providers
        for model in _provider_catalog_names(provider)
    )


def _openrouter_variant_base(model_id: str) -> Optional[str]:
    """Return the base model id when ``model_id`` carries a recognized OpenRouter routing-variant
    suffix (e.g. ``x-ai/grok-4:nitro`` → ``x-ai/grok-4``), else ``None``.
    """
    base, sep, suffix = (model_id or "").rpartition(":")
    if not sep or not base:
        return None
    if suffix.lower() in _OPENROUTER_VARIANT_SUFFIXES:
        return base
    return None


def _resolve_static_model_alias(
    name_lower: str,
    current_keys: set[str],
) -> Optional[tuple[str, str]]:
    """Resolve short aliases (e.g. sonnet/opus) using static catalogs only."""
    try:
        from hermes_cli.model_switch import MODEL_ALIASES
    except Exception:
        return None

    identity = MODEL_ALIASES.get(name_lower)
    if identity is None:
        return None

    vendor = identity.vendor
    family = identity.family

    def _match(provider: str) -> Optional[str]:
        models = _PROVIDER_MODELS.get(provider, [])
        if not models:
            return None
        prefix = (
            f"{vendor}/{family}"
            if provider in _AGGREGATOR_PROVIDERS
            else family
        ).lower()
        for model in models:
            if model.lower().startswith(prefix):
                return model
        return None

    for provider in current_keys:
        if matched := _match(provider):
            return provider, matched

    for provider in _PROVIDER_MODELS:
        if (
            provider in current_keys
            or provider in _AGGREGATOR_PROVIDERS
            or provider in _BORROWED_MODEL_PROVIDERS
        ):
            continue
        if matched := _match(provider):
            return provider, matched

    for provider in _AGGREGATOR_PROVIDERS:
        if provider in current_keys and (matched := _match(provider)):
            return provider, matched

    # Last resort: providers that re-expose other vendors' models. Only reached
    # when no native-vendor catalog matched — so `sonnet` resolves to anthropic.
    # None are currently defined (_BORROWED_MODEL_PROVIDERS is empty).
    for provider in _BORROWED_MODEL_PROVIDERS:
        if provider in current_keys and (matched := _match(provider)):
            return provider, matched

    return None


def detect_static_provider_for_model(
    model_name: str,
    current_provider: str,
) -> Optional[tuple[str, str]]:
    """Auto-detect a provider from static catalogs only.

    Returns ``(provider_id, model_name)``; the name may be remapped when a static alias or bare
    provider name resolves to a catalog default. Returns ``None`` when no confident match is
    found.
    """
    name = (model_name or "").strip()
    if not name:
        return None

    name_lower = name.lower()
    current_keys = _provider_keys(current_provider)

    alias_match = _resolve_static_model_alias(name_lower, current_keys)
    if alias_match:
        return alias_match

    # --- Step 0: bare provider name typed as model ---
    # If someone types `/model nous` or `/model anthropic`, treat it as a
    # provider switch and pick the first model from that provider's catalog.
    # Skip "custom" and "openrouter" — custom has no model catalog, and
    # openrouter requires an explicit model name to be useful.
    resolved_provider = _PROVIDER_ALIASES.get(name_lower, name_lower)
    if resolved_provider not in {"custom", "openrouter"}:
        default_models = _PROVIDER_MODELS.get(resolved_provider, [])
        if (
            resolved_provider in _PROVIDER_LABELS
            and default_models
            and resolved_provider not in current_keys
        ):
            # Route through the cost-safe default rather than picking
            # ``default_models[0]`` directly. For metered aggregators whose
            # curated list is ordered most-capable-first (e.g. Nous Portal),
            # entry [0] is the priciest flagship, and typing ``/model nous``
            # would silently escalate to it — the exact billing footgun the
            # catalog-labeled silent default (``_SILENT_DEFAULT_PROVIDERS``)
            # exists to prevent. For providers outside that set this is
            # unchanged (it returns ``models[0]``).
            return (
                resolved_provider,
                get_default_model_for_provider(resolved_provider) or default_models[0],
            )

    # Aggregators list other providers' models — never auto-switch TO them
    # If the model belongs to the current provider's catalog, don't suggest switching
    if _model_in_provider_catalog(name_lower, current_keys):
        return None

    # --- Step 1: check static provider catalogs for a direct match ---
    # If the current provider is a custom endpoint (custom or custom:*), never
    # auto-switch away from it based on a static catalog match — the user
    # explicitly configured their own endpoint and the same model name may be
    # served there (#48305).
    _is_custom_current = (
        current_provider == "custom"
        or current_provider.startswith("custom:")
    )
    for pid in _PROVIDER_MODELS:
        if (
            pid in current_keys
            or pid in _AGGREGATOR_PROVIDERS
            or pid in _BORROWED_MODEL_PROVIDERS
        ):
            continue
        if _is_custom_current:
            continue
        if any(name_lower == m.lower() for m in _provider_catalog_names(pid)):
            return (pid, name)

    # Borrow-list providers (re-expose other vendors' models) only after every
    # native-vendor catalog, and only when one is the current provider.
    for pid in _BORROWED_MODEL_PROVIDERS:
        if pid in current_keys:
            continue
        if any(name_lower == m.lower() for m in _provider_catalog_names(pid)):
            return (pid, name)

    return None


def _configured_provider_ids() -> set[str]:
    """Provider ids defined in the user's config ``providers:`` block.

    Includes both top-level ids (``ollama``, ``nous``) and ``custom:*`` profile ids. Returns an
    empty set when config is unreadable — callers treat that as "no user-defined providers" and fall
    through to built-in catalogs only.
    """
    try:
        from hermes_cli.config import load_config

        cfg = load_config() or {}
        providers = cfg.get("providers")
        if not isinstance(providers, dict):
            return set()
        ids: set[str] = set()
        for pid in providers:
            key = str(pid).strip().lower()
            if key:
                ids.add(key)
        return ids
    except Exception:
        return set()


def _resolve_provider_prefix(model_name: str) -> Optional[tuple[str, str]]:
    """Resolve an explicit ``vendor/model`` prefix to a configured provider.

    ``nous/deepseek-v4-pro`` or ``ollama/qwen3.5:4b`` should route to the named provider instead of
    falling back to the configured default (which silently sends non-default models to the wrong
    endpoint, #87189).

    Only vendors the user actually defined in their ``providers:`` config block (by raw name or
    alias) are routed here.
    """
    if "/" not in model_name:
        return None
    vendor, model = model_name.split("/", 1)
    vendor = vendor.strip().lower()
    model = model.strip()
    if not vendor or not model:
        return None
    configured = _configured_provider_ids()
    if not configured:
        return None
    # A provider block the user explicitly named (``ollama:``) wins over the
    # built-in alias table, which may canonicalize the same name elsewhere
    # (``ollama`` → ``custom``) and route to the wrong endpoint.
    if vendor in configured:
        return (vendor, model)
    canonical = _PROVIDER_ALIASES.get(vendor, vendor)
    if canonical in configured:
        return (canonical, model)
    return None


def detect_provider_for_model(
    model_name: str,
    current_provider: str,
) -> Optional[tuple[str, str]]:
    """Auto-detect the best provider for a model name.

    Priority: 0. Bare provider name → switch to that provider's default model 1. Direct provider
    static catalog match 2. OpenRouter catalog match
    """
    name = (model_name or "").strip()
    if not name:
        return None

    static_match = detect_static_provider_for_model(name, current_provider)
    if static_match:
        return static_match
    if _model_in_provider_catalog(name.lower(), _provider_keys(current_provider)):
        return None

    # --- Step 2: check OpenRouter catalog ---
    # First try exact match (handles provider/model format)
    or_slug = _find_openrouter_slug(name)
    if or_slug:
        if current_provider != "openrouter":
            return ("openrouter", or_slug)
        # Already on openrouter, just return the resolved slug
        if or_slug != name:
            return ("openrouter", or_slug)
        return None  # already on openrouter with matching name

    # --- Step 3: explicit ``vendor/model`` prefix naming a configured provider ---
    # Checked after the OpenRouter slug lookup so aggregator-native slugs
    # (e.g. ``deepseek/deepseek-chat``) keep their existing routing; only
    # vendors the user defined in their ``providers:`` block route here,
    # so catalog/default behavior for built-in vendor prefixes is unchanged
    # (#87189).
    prefix_match = _resolve_provider_prefix(name)
    if prefix_match is not None:
        return prefix_match

    return None


def _find_openrouter_slug(model_name: str) -> Optional[str]:
    """Find the full OpenRouter model slug for a bare or partial model name."""
    name_lower = model_name.strip().lower()
    if not name_lower:
        return None

    # Exact match (already has provider/ prefix)
    for mid in model_ids():
        if name_lower == mid.lower():
            return mid

    # Try matching just the model part (after the /)
    for mid in model_ids():
        if "/" in mid:
            _, model_part = mid.split("/", 1)
            if name_lower == model_part.lower():
                return mid

    return None


def normalize_provider(provider: Optional[str]) -> str:
    """Normalize provider aliases to Hermes' canonical provider ids.

    ``"auto"`` passes through unchanged — use ``hermes_cli.auth.resolve_provider()`` to resolve
    it to a concrete provider from credentials and environment.
    """
    normalized = (provider or "openrouter").strip().lower()
    return _PROVIDER_ALIASES.get(normalized, normalized)


def provider_label(provider: Optional[str]) -> str:
    """Return a human-friendly label for a provider id or alias."""
    original = (provider or "openrouter").strip()
    normalized = original.lower()
    if normalized == "auto":
        return "Auto"
    normalized = normalize_provider(normalized)
    return _PROVIDER_LABELS.get(normalized, original or "OpenRouter")


def _is_openai_fast_model(model_id: Optional[str]) -> bool:
    """Return True if the model is an OpenAI flagship eligible for Priority Processing."""
    raw = _strip_vendor_prefix(str(model_id or ""))
    base = raw.split(":")[0]
    if not base:
        return False
    # Exclude Codex-series — they route through the Codex Responses API
    # which doesn't accept service_tier.
    if "codex" in base:
        return False
    return any(base.startswith(prefix) for prefix in _OPENAI_FAST_MODE_PREFIXES)


# Models that support Anthropic Fast Mode (speed="fast").
# See https://platform.claude.com/docs/en/build-with-claude/fast-mode
#
# Pattern-based matching — any claude-* model is eligible. The anthropic
# adapter gates speed=fast on native Anthropic endpoints only (see
# _is_third_party_anthropic_endpoint in agent/anthropic_adapter.py), so
# third-party proxies that would reject the beta header are protected.


def _strip_vendor_prefix(model_id: str) -> str:
    """Strip vendor/ prefix from a model ID (e.g. 'anthropic/claude-opus-4-6' -> 'claude-opus-4-6')."""
    raw = str(model_id or "").strip().lower()
    if "/" in raw:
        raw = raw.split("/", 1)[1]
    return raw


def model_supports_fast_mode(model_id: Optional[str]) -> bool:
    """Return whether Hermes should expose the /fast toggle for this model."""
    from agent.model_metadata import is_grok_46_family

    return (
        _is_anthropic_fast_model(model_id)
        or _is_openai_fast_model(model_id)
        or is_grok_46_family(str(model_id or ""))
    )


def _is_anthropic_fast_model(model_id: Optional[str]) -> bool:
    """Return True if the model accepts the Anthropic Fast Mode ``speed`` param.

    This gates the *speed=fast request parameter*, which Anthropic supports on Opus 4.8 and Opus 5
    (research preview, Claude API only). It is deliberately NOT a general "is this a fast model"
    check:

    - Opus 4.7 hard-400s on the parameter. - Dedicated ``…-fast`` model ids (e.g. OpenRouter's
    ``claude-opus-4.8-fast``) select fast inference via the model field and must not also receive
    the speed parameter.
    """
    raw = _strip_vendor_prefix(str(model_id or ""))
    base = raw.split(":")[0]
    if not base.startswith("claude-"):
        return False
    if "-fast" in base:
        return False
    return any(v in base for v in ("opus-4-8", "opus-4.8", "opus-5"))


def _fast_mode_route_supported(
    model_id: Optional[str], provider: Optional[str], base_url: Optional[str]
) -> bool:
    """Only the first-party endpoint that bills for fast mode may receive its params."""
    from urllib.parse import urlparse

    from agent.model_metadata import is_grok_46_family

    if _is_anthropic_fast_model(model_id):
        allowed = {"anthropic": "api.anthropic.com"}
    elif is_grok_46_family(str(model_id or "")):
        allowed = {"xai": "api.x.ai"}
    else:
        allowed = {"openai": "api.openai.com", "openai-codex": "chatgpt.com"}
    if provider and normalize_provider(provider) not in allowed:
        return False
    host = (urlparse(str(base_url or "")).hostname or "").lower()
    return not host or host in allowed.values()


def resolve_fast_mode_overrides(
    model_id: Optional[str],
    *,
    provider: Optional[str] = None,
    base_url: Optional[str] = None,
) -> dict[str, Any] | None:
    """Return request_overrides for fast/priority mode, or None if unsupported.

    Returns provider-appropriate overrides: - OpenAI models: ``{"service_tier": "priority"}``
    (Priority Processing) - Anthropic models: ``{"speed": "fast"}`` (Anthropic Fast Mode beta) -
    Grok 4.6: ``{"service_tier": "priority"}`` (xAI Priority Processing)

    When ``provider``/``base_url`` are given the result is also gated on the route (see
    ``_fast_mode_route_supported``) so proxies never see the params. This is the single fast-mode
    gate for static ``/fast fast`` and the bounded ``auto``/``cold`` windows in ``agent.fast_mode``.
    """
    if not model_supports_fast_mode(model_id):
        return None
    if (provider or base_url) and not _fast_mode_route_supported(
        model_id, provider, base_url
    ):
        return None
    if _is_anthropic_fast_model(model_id):
        return {"speed": "fast"}
    return {"service_tier": "priority"}


def _first_exchangeable_copilot_token(raw_tokens) -> str:
    """Exchange stored GitHub tokens in order; the first that validates AND exchanges wins.

    Trying every entry (instead of stopping at the first malformed one) keeps a later valid entry
    reachable when an earlier one is unsupported.
    """
    from hermes_cli.copilot_auth import exchange_copilot_token, validate_copilot_token

    for raw in raw_tokens:
        raw = str(raw or "").strip()
        if not raw:
            continue
        valid, _ = validate_copilot_token(raw)
        if not valid:
            continue
        try:
            # exchange_copilot_token returns (api_token, expires_at, base_url).
            api_token = exchange_copilot_token(raw)[0]
        except Exception:
            continue
        if api_token:
            return api_token
    return ""


def _copilot_cli_config_tokens() -> list[str]:
    """``copilotTokens`` from the GitHub Copilot CLI's own plaintext store (JSONC — strip
    ``//``-comment lines), written by ``copilot login`` on hosts without an OS keychain."""
    cli_config = os.path.expanduser("~/.copilot/config.json")
    if not os.path.isfile(cli_config):
        return []
    with open(cli_config, "r", encoding="utf-8", errors="ignore") as fh:
        raw_text = "\n".join(
            line for line in fh.read().splitlines()
            if not line.lstrip().startswith("//")
        )
    data = json.loads(raw_text) if raw_text.strip() else {}
    tokens = data.get("copilotTokens")
    return list(tokens.values()) if isinstance(tokens, dict) else []


def _resolve_copilot_catalog_api_key() -> str:
    """Best-effort GitHub token for fetching the Copilot model catalog.

    Resolution order:
      1. ``resolve_api_key_provider_credentials("copilot")`` — env vars (``COPILOT_GITHUB_TOKEN`` /
         ``GH_TOKEN`` / ``GITHUB_TOKEN``) plus the ``gh auth token`` CLI fallback.
      2. ``read_credential_pool("copilot")`` — a token (a ``gho_*`` from device-code login, or a
         fine-grained PAT) stored in ``auth.json`` under ``credential_pool.copilot[]``.
      3. ``~/.copilot/config.json`` ``copilotTokens`` — without it, a user whose ONLY credential is
         the ACP CLI login sees the copilot-acp picker fall back to the stale curated list.

    Without (2)/(3), users without env-var credentials see the ``/model`` picker fall back to a
    stale hardcoded list because the live catalog fetch silently 401s.
    """
    try:
        from hermes_cli.auth import resolve_api_key_provider_credentials

        api_key = str(resolve_api_key_provider_credentials("copilot").get("api_key") or "").strip()
        if api_key:
            return api_key
    except Exception:
        pass
    try:
        from hermes_cli.auth import read_credential_pool

        token = _first_exchangeable_copilot_token(
            entry.get("access_token") for entry in read_credential_pool("copilot") if isinstance(entry, dict)
        )
        if token:
            return token
    except Exception:
        pass
    try:
        return _first_exchangeable_copilot_token(_copilot_cli_config_tokens())
    except Exception:
        return ""


def _model_dedup_key(model_id: str) -> str:
    """Case-insensitive dedup key that also folds picker-search aliases.

    Some providers serve one model under both a curated public slug and a bare live wire id
    (Kimi lists ``k3`` while the curated catalog carries ``kimi-k3``). Folding through the
    search-alias table keeps the curated-first merge from emitting both; the primary list's row
    survives and selection sends whichever id it carries.
    """
    key = str(model_id).strip().lower()
    try:
        from hermes_cli.model_search import model_alias_canonical
        return model_alias_canonical(key)
    except Exception:
        return key


def _merge_with_models_dev(provider: str, curated: list[str]) -> list[str]:
    """Merge curated list with fresh models.dev entries for a preferred provider.

    Returns models.dev entries first (in models.dev order), then any curated-only entries appended.
    Preserves case for curated fallbacks (e.g. ``MiniMax-M2.7``) while trusting models.dev for newer
    variants.

    If models.dev is unreachable or returns nothing, the curated list is returned unchanged — this
    is the offline/CI fallback path.
    """
    try:
        from agent.models_dev import list_agentic_models
        mdev = list_agentic_models(provider)
    except Exception:
        mdev = []

    if not mdev:
        return list(curated)

    # Case-insensitive dedup while preserving order and curated casing.
    seen_lower: set[str] = set()
    merged: list[str] = []
    for mid in mdev:
        key = str(mid).lower()
        if key in seen_lower:
            continue
        seen_lower.add(key)
        merged.append(mid)
    for mid in curated:
        key = str(mid).lower()
        if key in seen_lower:
            continue
        seen_lower.add(key)
        merged.append(mid)
    return merged


def _openai_discovery_base_url(provider: str) -> str:
    """Effective OpenAI endpoint for model discovery.

    Mirrors the runtime precedence so discovery probes the SAME endpoint inference uses:
    ``$OPENAI_BASE_URL`` (explicit env override) → ``model.base_url`` from config.yaml when the
    configured provider matches → the canonical default.
    """
    env_raw = os.getenv("OPENAI_BASE_URL", "").strip().rstrip("/")
    if env_raw:
        return env_raw
    try:
        model_cfg = _get_model_config_dict()
        cfg_provider = str(model_cfg.get("provider") or "").strip().lower()
        if cfg_provider in ("openai", "openai-api") and normalize_provider(provider) == normalize_provider(cfg_provider):
            cfg_url = str(model_cfg.get("base_url") or "").strip().rstrip("/")
            if cfg_url:
                return cfg_url
    except Exception:
        pass
    return "https://api.openai.com/v1"


def _codex_catalog(normalized: str, force_refresh: bool) -> list[str]:
    from hermes_cli.codex_models import get_codex_model_ids

    # Pass the live OAuth access token so the picker matches whatever ChatGPT lists for this
    # account right now; falls back to the hardcoded catalog without a token / when unreachable.
    access_token = None
    try:
        from hermes_cli.auth import resolve_codex_runtime_credentials

        access_token = resolve_codex_runtime_credentials(refresh_if_expiring=True).get("api_key")
    except Exception:
        access_token = None
    return get_codex_model_ids(access_token=access_token)


def _copilot_catalog(normalized: str, force_refresh: bool) -> Optional[list[str]]:
    try:
        live = _fetch_github_models(_resolve_copilot_catalog_api_key())
        if live:
            return live
    except Exception:
        pass
    if normalized == "copilot-acp":
        return list(_PROVIDER_MODELS.get("copilot", []))
    return None


def _nous_catalog(normalized: str, force_refresh: bool) -> Optional[list[str]]:
    try:
        from hermes_cli.auth import fetch_nous_models, resolve_nous_runtime_credentials

        creds = resolve_nous_runtime_credentials()
        if creds:
            live = fetch_nous_models(api_key=creds.get("api_key", ""), inference_base_url=creds.get("base_url", ""))
            if live:
                return live
    except Exception:
        pass
    # Live failed (or no creds): the docs-hosted manifest — NOT the in-repo snapshot — so newly
    # added Portal models still surface without a Hermes release.
    return get_curated_nous_model_ids() or None


def _api_key_provider_live(normalized: str, force_refresh: bool) -> Optional[list[str]]:
    """Live /v1/models for a simple api-key provider (stepfun, gmi); None on any miss."""
    try:
        from hermes_cli.auth import resolve_api_key_provider_credentials

        creds = resolve_api_key_provider_credentials(normalized)
        api_key = str(creds.get("api_key") or "").strip()
        base_url = str(creds.get("base_url") or "").strip()
        if api_key and base_url:
            return fetch_api_models(api_key, base_url) or None
    except Exception:
        pass
    return None


def _anthropic_catalog(normalized: str, force_refresh: bool) -> list[str]:
    model_cfg = _get_model_config_dict()
    cfg_base_url = cfg_api_key = ""
    if normalize_provider(str(model_cfg.get("provider", "") or "")) == "anthropic":
        cfg_base_url = str(model_cfg.get("base_url", "") or "").strip()
        cfg_api_key = str(model_cfg.get("api_key", "") or "").strip()
    live = _fetch_anthropic_models(base_url=cfg_base_url or None, api_key=cfg_api_key or None)
    curated = list(_PROVIDER_MODELS.get("anthropic", []))
    if not live:
        return curated
    if cfg_base_url:
        return live
    # The live /v1/models dump lags newly-routed curated aliases (reachable before enumerated).
    # Curated first, then live-only extras, so a fresh curated model never disappears.
    merged = list(curated)
    merged_lower = {m.lower() for m in curated}
    for m in live:
        if m.lower() not in merged_lower:
            merged.append(m)
            merged_lower.add(m.lower())
    return merged


def _openai_catalog(normalized: str, force_refresh: bool) -> Optional[list[str]]:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None
    base = _openai_discovery_base_url(normalized)
    # Custom OpenAI-compatible endpoints may serve a small curated catalog — use it verbatim.
    # Official OpenAI hosts (canonical AND data-residency regional, identical dump) return 120+
    # embeddings/whisper/tts/dall-e/moderation/legacy entries, so intersect with the curated
    # agentic catalog there so ``/model`` matches ``hermes model``.
    from hermes_cli.providers import is_official_openai_host

    is_default_openai = is_official_openai_host(base)
    try:
        live = fetch_api_models(api_key, base)
    except Exception:
        return None
    if not live:
        return None
    if not is_default_openai:
        return live
    live_lower = {m.lower() for m in live}
    curated = list(_PROVIDER_MODELS.get(normalized, []))
    # Keep curated order; only surface curated models the account actually has access to. An
    # account serving none of them (rare) falls back to curated so the picker still offers sane
    # defaults.
    filtered = [m for m in curated if m.lower() in live_lower]
    return filtered or curated or live


def _custom_catalog(normalized: str, force_refresh: bool) -> Optional[list[str]]:
    base_url = _get_custom_base_url()
    if not base_url:
        return None
    model_cfg = _get_model_config_dict()
    # Try common API key env vars for custom endpoints.
    api_key = (
        str(model_cfg.get("api_key", "") or "").strip()
        or os.getenv("CUSTOM_API_KEY", "")
        or os.getenv("OPENAI_API_KEY", "")
        or os.getenv("OPENROUTER_API_KEY", "")
    )
    api_mode = "anthropic_messages" if _base_url_looks_like_anthropic_messages(base_url) else None
    return fetch_api_models(api_key, base_url, api_mode=api_mode) or None


def _bedrock_catalog(normalized: str, force_refresh: bool) -> Optional[list[str]]:
    # Live discovery keyed by the resolved AWS region so EU/AP users see eu.*/ap.* ids instead of
    # the static us.* list. A hit skips the _MODELS_DEV_PREFERRED merge (bedrock isn't in it).
    try:
        from agent.bedrock_adapter import bedrock_model_ids_or_none

        return bedrock_model_ids_or_none()
    except Exception:
        return None


def _opencode_free_catalog(normalized: str, force_refresh: bool) -> list[str]:
    # Keyless live catalog revalidated against the Zen relay every TTL. models.dev's
    # cost.input==0 filter lags reality (a promo model kept "free" there after the relay began
    # 401ing keyless requests), so filter the live /zen/v1/models dump to the anonymous-servable
    # `*-free` tier ourselves; the curated floor only applies when the live fetch fails/is empty.
    return _fetch_opencode_free_models(force_refresh=force_refresh) or list(_PROVIDER_MODELS.get(normalized, []))


# Per-provider catalog sources tried before the generic profile fetch. A fetcher returning None
# falls through to the profile/curated path; a list is returned as-is (even empty).
_PROVIDER_CATALOG_FETCHERS: dict[str, Any] = {
    "openrouter": lambda normalized, force_refresh: model_ids(force_refresh=force_refresh),
    "openai-codex": _codex_catalog,
    "copilot": _copilot_catalog,
    "copilot-acp": _copilot_catalog,
    "nous": _nous_catalog,
    "stepfun": _api_key_provider_live,
    "gmi": _api_key_provider_live,
    "anthropic": _anthropic_catalog,
    "ai-gateway": lambda normalized, force_refresh: _fetch_ai_gateway_models() or None,
    # DeepInfra's generic /models mixes chat, image, video, speech and embedding models; the
    # tagged catalog helper is the only safe source for the chat picker, including its
    # empty/failure result.
    "deepinfra": lambda normalized, force_refresh: _fetch_deepinfra_models(force_refresh=force_refresh) or [],
    "ollama-cloud": lambda normalized, force_refresh: fetch_ollama_cloud_models(force_refresh=force_refresh) or None,
    "openai": _openai_catalog,
    "openai-api": _openai_catalog,
    "custom": _custom_catalog,
    "bedrock": _bedrock_catalog,
    "opencode-free": _opencode_free_catalog,
}


def _profile_live_catalog(normalized: str) -> Optional[list[str]]:
    """Generic live fetch for any provider registered in providers/ with ``auth_type="api_key"``.

    Live results are merged with the curated list so models the live endpoint omits (stale cache,
    partial rollout) still appear. Most providers merge curated-first so the newest curated models
    lead even when the live API lags; ``_LIVE_FIRST_PICKER_PROVIDERS`` (OpenCode Zen/Go, whose
    live API is authoritative) merge live-first so stale curated entries stop polluting the top.
    Plugin providers with no static entry use the profile's ``fallback_models`` as the curated
    list so their agentic picks lead the picker (Fireworks lists an image model first).
    """
    from providers import get_provider_profile
    from hermes_cli.auth import resolve_api_key_provider_credentials

    profile = get_provider_profile(normalized)
    if not (profile and profile.auth_type == "api_key" and profile.base_url):
        return None
    try:
        creds = resolve_api_key_provider_credentials(normalized)
        api_key = str(creds.get("api_key") or "").strip()
        base_url = str(creds.get("base_url") or "").strip()
    except Exception:
        api_key, base_url = "", profile.base_url
    if not base_url:
        base_url = profile.base_url
    if api_key:
        live = profile.fetch_models(api_key=api_key, base_url=base_url or None)
        if live:
            curated = list(_PROVIDER_MODELS.get(normalized, [])) or list(profile.fallback_models or ())
            if not curated:
                return live
            if normalized in _LIVE_FIRST_PICKER_PROVIDERS:
                primary, secondary = live, curated
            else:
                primary, secondary = curated, live
            merged = list(primary)
            merged_keys = {_model_dedup_key(m) for m in primary}
            for m in secondary:
                if _model_dedup_key(m) not in merged_keys:
                    merged.append(m)
                    merged_keys.add(_model_dedup_key(m))
            return merged
    if profile.fallback_models:
        return list(profile.fallback_models)
    return None


def provider_model_ids(provider: Optional[str], *, force_refresh: bool = False) -> list[str]:
    """Return the best known model catalog for a provider.

    Tries live API endpoints where supported (Codex, Nous), falling back to static lists. For
    providers in ``_MODELS_DEV_PREFERRED`` models.dev entries are merged on top of curated so
    new platform models appear in ``/model`` without a Hermes release.
    """
    requested = str(provider or "").strip().lower()
    if requested == "ollama":
        return _ollama_local_catalog(force_refresh)

    normalized = normalize_provider(provider)
    fetcher = _PROVIDER_CATALOG_FETCHERS.get(normalized)
    if fetcher is not None:
        models = fetcher(normalized, force_refresh)
        if models is not None:
            return models

    try:
        models = _profile_live_catalog(normalized)
        if models is not None:
            return models
    except Exception:
        pass

    curated_static = list(_PROVIDER_MODELS.get(normalized, []))
    if normalized in _MODELS_DEV_PREFERRED:
        merged = _merge_with_models_dev(normalized, curated_static)
        if normalized in {"xai", "xai-oauth"}:
            return _xai_finalize_catalog(merged)
        return merged
    return curated_static


# ---------------------------------------------------------------------------
# Generic disk cache for provider_model_ids() — keeps /model picker fast.
# ---------------------------------------------------------------------------
#
# Without this layer, every /model picker open re-fetches every authed
# provider's /v1/models endpoint. On a well-configured user (anthropic +
# openai + copilot + gemini + huggingface + ...) that's 2+ seconds of cold
# HTTP roundtrips just to render the provider list.
#
# Cache strategy:
#   - One JSON file at $HERMES_HOME/provider_models_cache.json
#   - Per-provider entries keyed by (provider, credential fingerprint)
#   - Credential fingerprint = sha256 of env-var values that the provider
#     normally reads. Swap your OPENAI_API_KEY and the entry invalidates.
#   - 1h TTL by default. `force_refresh=True` skips the cache entirely
#     and overwrites it on success.
#   - Only NON-EMPTY results are cached. An empty/None response from a
#     transient network error never gets pinned.
#   - Cache file is best-effort. Any read/write error degrades silently
#     to a live fetch — the picker keeps working.

_PROVIDER_MODELS_CACHE_TTL = 3600  # 1h
# Stale-while-revalidate window: an expired-but-same-credentials entry is
# served IMMEDIATELY (picker opens stay instant) while a background daemon
# thread re-fetches the live catalog and rewrites the disk cache for the
# next open. Beyond this bound the entry is considered too old to trust and
# the caller blocks on a live fetch as before. Rationale: the /model picker's
# provider listing runs 8-9 serial /v1/models round-trips (~2-3s) whenever
# the 1h TTL lapses mid-session — model catalogs change on release timescales,
# not hourly, so serving hour-old data while refreshing off-thread is strictly
# better than stalling every picker surface (CLI, TUI, dashboard, gateway).
_PROVIDER_MODELS_STALE_SERVE_MAX = 7 * 24 * 3600  # 7d

# Providers with a background SWR refresh currently in flight — dedupes
# concurrent refreshes so repeated picker opens during one refresh don't
# stack threads or duplicate network calls.
_swr_refresh_inflight: set = set()
_swr_refresh_lock = threading.Lock()


def _cache_entry(fp: str, models: list[str], at: Optional[float] = None) -> dict:
    """One provider row of the disk cache: credential fingerprint, write time, model ids."""
    return {"fp": fp, "at": time.time() if at is None else at, "models": list(models)}


def _ollama_native_probe_reachable() -> bool:
    """Whether the configured local Ollama root answered the native ``/api/tags`` probe (an empty
    catalog from a reachable server is authoritative; a failed probe is not)."""
    base_url = _get_ollama_base_url()
    headers = _get_ollama_native_headers(base_url) or None
    probe_key = _ollama_probe_cache_key(_root_for_ollama_native_api(base_url), headers)
    return _OLLAMA_LOCAL_PROBE_REACHABLE.get(probe_key) is True


def _spawn_swr_refresh(cache_key: str, refresh_fn=None) -> None:
    """Kick a background refresh of *cache_key*'s model-id cache entry.

    Fire-and-forget daemon thread; at most one in flight per cache key. Failures are swallowed — the
    stale entry stays served until a later refresh succeeds (same degradation the blocking path
    already had).

    ``refresh_fn`` (no-args, returns the fresh cache-entry dict or ``None``) lets non-slug keys
    (``custom:<base_url>`` entries from :func:`cached_fetch_api_models`) reuse the same inflight-
    dedupe and thread scaffolding.
    """
    with _swr_refresh_lock:
        if cache_key in _swr_refresh_inflight:
            return
        _swr_refresh_inflight.add(cache_key)

    def _default_refresh():
        live = provider_model_ids(cache_key, force_refresh=True)
        if not live and cache_key == "ollama" and _ollama_native_probe_reachable():
            return _cache_entry(_credential_fingerprint(cache_key), [])
        if not live:
            return None
        return _cache_entry(_credential_fingerprint(cache_key), live)

    def _refresh() -> None:
        try:
            entry = (refresh_fn or _default_refresh)()
            if entry:
                cache = _load_provider_models_cache()
                cache[cache_key] = entry
                _save_provider_models_cache(cache)
        except Exception:
            logger.debug("SWR refresh failed for %s", cache_key, exc_info=True)
        finally:
            with _swr_refresh_lock:
                _swr_refresh_inflight.discard(cache_key)

    threading.Thread(
        target=_refresh, daemon=True, name=f"model-cache-swr-{cache_key}"
    ).start()


def _provider_models_cache_path() -> Path:
    from hermes_constants import get_hermes_home
    return get_hermes_home() / "provider_models_cache.json"


def _credential_fingerprint(provider: str) -> str:
    """Short hash of the credentials ``provider_model_ids(provider)`` would see right now.

    Rotating any relevant env var invalidates that provider's cache entry: the api-key and base-
    url env vars from ``PROVIDER_REGISTRY`` are hashed. OAuth-backed providers keep tokens in
    ``auth.json`` and external credential files, so those files' mtimes are folded in too — re-
    auth busts the cache without parsing every file shape.
    """
    import hashlib
    import os as _os

    parts: list[str] = []

    # Keyless providers have no credential to fingerprint: the catalog is
    # served anonymously, so nothing the user rotates (env vars, auth files,
    # base URLs) should invalidate the cached entry. A stable fingerprint keeps
    # the SWR disk cache alive across unrelated re-auths and only busts on TTL
    # expiry — matching how the live catalog genuinely changes.
    if (provider or "").strip().lower() in _KEYLESS_STABLE_CACHE_PROVIDERS:
        return "keyless:" + (provider or "").strip().lower()

    # Env vars from PROVIDER_REGISTRY for this slug
    try:
        from hermes_cli.auth import PROVIDER_REGISTRY
        pcfg = PROVIDER_REGISTRY.get(provider)
        if pcfg is not None:
            for ev in getattr(pcfg, "api_key_env_vars", ()) or ():
                parts.append(f"{ev}={_os.environ.get(ev, '')}")
            bev = getattr(pcfg, "base_url_env_var", "") or ""
            if bev:
                parts.append(f"{bev}={_os.environ.get(bev, '')}")
    except Exception:
        pass

    # Effective configured endpoint: config.yaml's model.base_url changes the
    # endpoint discovery probes (data-residency hosts) without touching any
    # env var, so it must change the fingerprint too or `hermes config set
    # model.base_url ...` keeps serving the previous endpoint's cached
    # catalog until TTL expiry.
    if provider in ("openai", "openai-api"):
        try:
            parts.append(f"effective_base={_openai_discovery_base_url(provider)}")
        except Exception:
            pass

    if provider == "ollama":
        parts.append(f"OLLAMA_HOST={_os.environ.get('OLLAMA_HOST', '')}")
        provider_cfg = _get_provider_config_dict("ollama")
        parts.append(
            "providers.ollama.base_url="
            f"{provider_cfg.get('base_url', '') or provider_cfg.get('api', '') or provider_cfg.get('url', '')}"
        )
        parts.append(f"providers.ollama.api_key={provider_cfg.get('api_key', '')}")
        key_env = provider_cfg.get("key_env") or provider_cfg.get("api_key_env") or ""
        parts.append(f"providers.ollama.key_env={key_env}")
        if key_env:
            parts.append(f"{key_env}={_os.environ.get(str(key_env), '')}")
        model_cfg = _get_model_config_dict()
        parts.append(
            "model.provider="
            f"{model_cfg.get('provider', '')}|model.base_url={model_cfg.get('base_url', '')}"
        )
        parts.append(
            "providers.ollama.extra_headers="
            + json.dumps(provider_cfg.get("extra_headers", {}), sort_keys=True, default=str)
        )

    # OAuth / external-file mtimes that change on re-auth
    try:
        from hermes_constants import get_hermes_home
        for rel in ("auth.json", "credentials.json"):
            p = get_hermes_home() / rel
            try:
                parts.append(f"{rel}@{p.stat().st_mtime_ns}")
            except FileNotFoundError:
                parts.append(f"{rel}@missing")
            except Exception:
                pass
    except Exception:
        pass

    # External well-known credential file locations
    for path in (
        _os.path.expanduser("~/.codex/auth.json"),
        _os.path.expanduser("~/.claude/.credentials.json"),
        _os.path.expanduser("~/.config/github-copilot/hosts.json"),
        _os.path.expanduser("~/.minimax/credentials.json"),
    ):
        try:
            mt = _os.stat(path).st_mtime_ns
            parts.append(f"{path}@{mt}")
        except FileNotFoundError:
            parts.append(f"{path}@missing")
        except Exception:
            pass

    blob = "|".join(parts).encode("utf-8", errors="replace")
    # blake2b for cache-key fingerprinting only — not for credential storage.
    # We never reverse this hash; collisions are harmless (worst case: cache
    # miss → live re-fetch). Use blake2b instead of sha256 here because
    # CodeQL's `py/weak-sensitive-data-hashing` rule flags sha256 over env
    # vars whose names contain "API_KEY" / "TOKEN" even when the hash is
    # used as an identity fingerprint, not for password storage. blake2b
    # is a keyed-hash primitive and isn't flagged.
    return hashlib.blake2b(blob, digest_size=8).hexdigest()


def _load_provider_models_cache() -> dict:
    """Return the full cache dict, or {} on any error."""
    try:
        return _read_json_cache(_provider_models_cache_path()) or {}
    except Exception:
        return {}


_cache_write_lock = threading.Lock()


def _save_provider_models_cache(data: dict) -> None:
    """Persist the cache dict. Best-effort — silent on any error."""
    try:
        _write_json_cache(_provider_models_cache_path(), data, indent=None)
    except Exception:
        pass


def update_provider_cache_entry(provider: str, models: list[str]) -> None:
    """Thread-safe single-entry update of the provider-models disk cache.

    Used by parallel prefetch workers so concurrent fetches don't clobber each other's writes via
    read-modify-write races on the shared JSON file. Each worker loads the latest cache state under
    the lock, writes its own entry, and saves — best-effort, silent on any error.
    """
    try:
        normalized = normalize_provider(provider) or (provider or "")
        if not normalized or not models:
            return
        fp = _credential_fingerprint(normalized)
        with _cache_write_lock:
            cache = _load_provider_models_cache()
            cache[normalized] = _cache_entry(fp, models)
            _save_provider_models_cache(cache)
    except Exception:
        pass


def cached_provider_model_ids(
    provider: Optional[str],
    *,
    force_refresh: bool = False,
    ttl_seconds: int = _PROVIDER_MODELS_CACHE_TTL,
) -> list[str]:
    """Disk-cached wrapper around :func:`provider_model_ids`.

    Hits the cache when fresh; otherwise calls the live function and persists a non-empty result.
    Always returns a list (never None).
    """
    requested = str(provider or "").strip().lower()
    normalized = requested if requested == "ollama" else (normalize_provider(provider) or (provider or ""))
    if not normalized:
        return []
    if normalized == "ollama":
        ttl_seconds = min(ttl_seconds, _OLLAMA_LOCAL_MODELS_CACHE_TTL)

    cache = _load_provider_models_cache()
    fp = _credential_fingerprint(normalized)
    entry = cache.get(normalized)
    now = time.time()

    allow_empty_ollama = normalized == "ollama"
    if not force_refresh and _cache_entry_valid(entry, fp, allow_empty=allow_empty_ollama):
        age = now - entry["at"]
        if age < ttl_seconds:
            return list(entry["models"])
        # Empty native catalogs are authoritative only for the short native
        # TTL. Re-probe after expiry so newly pulled models become visible;
        # do not serve an empty row through the generic stale window.
        if entry["models"] and age < _PROVIDER_MODELS_STALE_SERVE_MAX:
            # Stale-while-revalidate: serve the expired entry immediately so
            # interactive picker opens never block on serial /v1/models
            # round-trips; refresh the cache off-thread for the next open.
            _spawn_swr_refresh(normalized)
            return list(entry["models"])

    # Cache miss / stale / forced refresh — call the live path.
    live = provider_model_ids(normalized, force_refresh=force_refresh)
    if live:
        cache[normalized] = _cache_entry(fp, live, now)
        _save_provider_models_cache(cache)
        return list(live)

    if normalized == "ollama":
        if _ollama_native_probe_reachable():
            # A reachable empty native catalog is authoritative for the short
            # native TTL; do not resurrect a stale disk catalog.
            cache[normalized] = _cache_entry(fp, [], now)
            _save_provider_models_cache(cache)
            return []

        # A failed/non-native probe is not authoritative. Preserve a stale
        # catalog rather than blanking the picker during a transient outage.
        if (
            isinstance(entry, dict)
            and entry.get("fp") == fp
            and isinstance(entry.get("models"), list)
            and entry["models"]
        ):
            return list(entry["models"])
        return []

    # Live fetch returned nothing. If we have a stale entry with the
    # SAME fingerprint, prefer it over an empty result — stale data
    # beats no data when the network is flaky.
    if _cache_entry_valid(entry, fp):
        return list(entry["models"])
    return list(live or [])


def clear_provider_models_cache(provider: Optional[str] = None) -> None:
    """Drop a single provider's cache entry, or wipe the whole cache.

    ``provider=None`` wipes everything; otherwise only that provider's entry is removed. Used by
    ``/model --refresh`` and ``hermes model --refresh``.
    """
    try:
        # Native Ollama tags are keyed by root URL rather than provider slug.
        # A targeted refresh for a custom local-Ollama endpoint cannot identify
        # the right root from the provider name alone, so clear this small
        # in-process cache on every explicit provider-cache refresh.
        _OLLAMA_LOCAL_MODELS_CACHE.clear()
        _OLLAMA_LOCAL_PROBE_FAILURE_CACHE.clear()
        _OLLAMA_LOCAL_PROBE_REACHABLE.clear()
        if provider is None:
            path = _provider_models_cache_path()
            if path.exists():
                path.unlink()
            return
        cache = _load_provider_models_cache()
        requested = str(provider or "").strip().lower()
        normalized = requested if requested == "ollama" else (normalize_provider(provider) or provider or "")
        if normalized in cache:
            del cache[normalized]
            _save_provider_models_cache(cache)
    except Exception:
        pass


def _resolve_anthropic_pool_catalog_credentials() -> tuple[str, str]:
    """Return a read-only API-key pool credential for model discovery.

    ``resolve_anthropic_token()`` intentionally ignores ``api_key`` pool entries because its runtime
    contract is OAuth-oriented.
    """
    try:
        from agent.credential_pool import AUTH_TYPE_API_KEY
        from hermes_cli.auth import read_credential_pool

        for entry in read_credential_pool("anthropic"):
            if not isinstance(entry, dict):
                continue
            if entry.get("auth_type") != AUTH_TYPE_API_KEY:
                continue
            token = str(entry.get("access_token") or "").strip()
            if not token:
                continue
            endpoint = str(
                entry.get("base_url") or entry.get("inference_base_url") or ""
            ).strip()
            return token, endpoint
    except Exception:
        pass
    return "", ""


def _fetch_anthropic_models(
    timeout: float = 5.0,
    *,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Optional[list[str]]:
    """Fetch available models from the Anthropic /v1/models endpoint.

    Uses resolve_anthropic_token() to find credentials (env vars, OAuth, or Claude Code auto-
    discovery) unless api_key is provided explicitly. If those sources are empty, a read-only API-
    key credential_pool entry is used. Returns sorted model IDs or None.
    """
    try:
        from agent.anthropic_adapter import resolve_anthropic_token, _is_oauth_token
    except ImportError:
        return None

    resolved_base_url = base_url
    token = (api_key or "").strip() or resolve_anthropic_token()
    if not token:
        # A pool credential and its endpoint are one security boundary. Never
        # pair the selected pool key with a caller-provided model endpoint.
        token, resolved_base_url = _resolve_anthropic_pool_catalog_credentials()
    if not token:
        return None

    headers: dict[str, str] = {"anthropic-version": "2023-06-01"}
    is_oauth = _is_oauth_token(token)
    if is_oauth:
        headers["Authorization"] = f"Bearer {token}"
        from agent.anthropic_adapter import _COMMON_BETAS, _OAUTH_ONLY_BETAS, _CONTEXT_1M_BETA
        headers["anthropic-beta"] = ",".join(_COMMON_BETAS + _OAUTH_ONLY_BETAS)
    else:
        headers["x-api-key"] = token

    def _do_request(h: dict[str, str]):
        req = urllib.request.Request(
            _anthropic_models_url(resolved_base_url),
            headers=h,
        )
        with _urlopen_model_catalog_request(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())

    try:
        try:
            data = _do_request(headers)
        except urllib.error.HTTPError as http_err:
            # Reactive recovery for OAuth subscriptions that reject the 1M
            # context beta with 400 "long context beta is not yet available
            # for this subscription". Retry once without the beta; re-raise
            # anything else so the outer except logs it.
            if (
                is_oauth
                and http_err.code == 400
            ):
                try:
                    body_text = http_err.read().decode(errors="ignore").lower()
                except Exception:
                    body_text = ""
                if "long context beta" in body_text and "not yet available" in body_text:
                    headers["anthropic-beta"] = ",".join(
                        [b for b in _COMMON_BETAS if b != _CONTEXT_1M_BETA]
                        + list(_OAUTH_ONLY_BETAS)
                    )
                    data = _do_request(headers)
                else:
                    raise
            else:
                raise
        models = [m["id"] for m in data.get("data", []) if m.get("id")]
        # Sort: latest/largest first (opus > sonnet > haiku, higher version first)
        return sorted(models, key=lambda m: (
            "opus" not in m,      # opus first
            "sonnet" not in m,    # then sonnet
            "haiku" not in m,     # then haiku
            m,                    # alphabetical within tier
        ))
    except Exception as e:
        import logging
        logging.getLogger(__name__).debug("Failed to fetch Anthropic models: %s", e)
        return None


def _payload_items(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        data = payload.get("data", [])
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
    return []


def copilot_default_headers(*, is_agent_turn: bool = True) -> dict[str, str]:
    """Standard headers for Copilot API requests."""
    try:
        from hermes_cli.copilot_auth import copilot_request_headers
        return copilot_request_headers(is_agent_turn=is_agent_turn)
    except ImportError:
        return {
            "Editor-Version": COPILOT_EDITOR_VERSION,
            "User-Agent": "HermesAgent/1.0",
            "Openai-Intent": "conversation-edits",
            "x-initiator": "agent" if is_agent_turn else "user",
        }


def _copilot_catalog_item_is_text_model(
    item: dict[str, Any], *, ignore_picker_flag: bool = False
) -> bool:
    model_id = str(item.get("id") or "").strip()
    if not model_id:
        return False

    if not ignore_picker_flag and item.get("model_picker_enabled") is False:
        return False

    capabilities = item.get("capabilities")
    if isinstance(capabilities, dict):
        model_type = str(capabilities.get("type") or "").strip().lower()
        if model_type and model_type != "chat":
            return False

    supported_endpoints = item.get("supported_endpoints")
    if isinstance(supported_endpoints, list):
        normalized_endpoints = {
            str(endpoint).strip()
            for endpoint in supported_endpoints
            if str(endpoint).strip()
        }
        if normalized_endpoints and not normalized_endpoints.intersection(
            {"/chat/completions", "/responses", "/v1/messages"}
        ):
            return False

    return True


# Module-level cache for the GitHub Copilot /models catalog.
# The picker path can ask for it multiple times in one process via:
#   list_authenticated_providers -> cached_provider_model_ids -> provider_model_ids -> _fetch_github_models
# and later get_copilot_model_context()/normalize helpers. Cache the raw filtered
# catalog for a short TTL so we don't pay repeated TLS handshakes on every picker open.
# Keyed by the api_key used for the successful fetch so a credential swap
# mid-process never serves the previous account's catalog. Uses a monotonic
# clock so wall-clock adjustments can't extend the TTL. Lock-free like the
# other module caches here — a racing thread at worst duplicates one fetch.
_github_model_catalog_cache: Optional[list[dict[str, Any]]] = None
_github_model_catalog_cache_key: Optional[str] = None
_github_model_catalog_cache_time: float = 0.0
_GITHUB_MODEL_CATALOG_CACHE_TTL = 300  # 5 minutes


def fetch_github_model_catalog(
    api_key: Optional[str] = None, timeout: float = 5.0
) -> Optional[list[dict[str, Any]]]:
    """Fetch the live GitHub Copilot model catalog for this account."""
    global _github_model_catalog_cache, _github_model_catalog_cache_key
    global _github_model_catalog_cache_time

    if (
        _github_model_catalog_cache is not None
        and _github_model_catalog_cache_key == api_key
        and (time.monotonic() - _github_model_catalog_cache_time) < _GITHUB_MODEL_CATALOG_CACHE_TTL
    ):
        # Deep copy: catalog items are dicts, and a shallow copy would let
        # callers mutate the cached entries in place.
        return copy.deepcopy(_github_model_catalog_cache)

    attempts: list[dict[str, str]] = []
    if api_key:
        attempts.append({
            **copilot_default_headers(),
            "Authorization": f"Bearer {api_key}",
        })
    attempts.append(copilot_default_headers())

    for headers in attempts:
        req = urllib.request.Request(COPILOT_MODELS_URL, headers=headers)
        try:
            with _urlopen_model_catalog_request(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode())
                items = _payload_items(data)
                models: list[dict[str, Any]] = []
                seen_ids: set[str] = set()
                for item in items:
                    if not _copilot_catalog_item_is_text_model(item):
                        continue
                    model_id = str(item.get("id") or "").strip()
                    if not model_id or model_id in seen_ids:
                        continue
                    seen_ids.add(model_id)
                    models.append(item)
                if not models and items:
                    # GitHub has been observed returning
                    # ``model_picker_enabled: false`` for EVERY model on some
                    # accounts/token types, which would silently reject the
                    # whole live catalog and strand the picker on the stale
                    # curated fallback. The flag is a display hint, not an
                    # availability contract — when honoring it empties the
                    # catalog, retry without it (chat/endpoint checks still
                    # apply, so embeddings and non-chat rows stay excluded).
                    for item in items:
                        if not _copilot_catalog_item_is_text_model(
                            item, ignore_picker_flag=True
                        ):
                            continue
                        model_id = str(item.get("id") or "").strip()
                        if not model_id or model_id in seen_ids:
                            continue
                        seen_ids.add(model_id)
                        models.append(item)
                if models:
                    _github_model_catalog_cache = copy.deepcopy(models)
                    _github_model_catalog_cache_key = api_key
                    _github_model_catalog_cache_time = time.monotonic()
                    return models
        except Exception:
            continue
    return None


# ─── Copilot catalog context-window helpers ─────────────────────────────────

# Module-level cache: {model_id: max_prompt_tokens}
_copilot_context_cache: dict[str, int] = {}
_copilot_context_cache_time: float = 0.0
_COPILOT_CONTEXT_CACHE_TTL = 3600  # 1 hour


def get_copilot_model_context(model_id: str, api_key: Optional[str] = None) -> Optional[int]:
    """Look up max_prompt_tokens for a Copilot model from the live /models API.

    Results are cached in-process for 1 hour to avoid repeated API calls. Returns the token limit or
    None if not found.
    """
    global _copilot_context_cache, _copilot_context_cache_time

    # Serve from cache if fresh
    if _copilot_context_cache and (time.time() - _copilot_context_cache_time < _COPILOT_CONTEXT_CACHE_TTL):
        if model_id in _copilot_context_cache:
            return _copilot_context_cache[model_id]
        # Cache is fresh but model not in it — don't re-fetch
        return None

    # Fetch and populate cache
    catalog = fetch_github_model_catalog(api_key=api_key)
    if not catalog:
        return None

    cache: dict[str, int] = {}
    for item in catalog:
        mid = str(item.get("id") or "").strip()
        if not mid:
            continue
        caps = item.get("capabilities") or {}
        limits = caps.get("limits") or {}
        max_prompt = limits.get("max_prompt_tokens")
        if isinstance(max_prompt, int) and max_prompt > 0:
            cache[mid] = max_prompt

    _copilot_context_cache = cache
    _copilot_context_cache_time = time.time()

    return cache.get(model_id)


def _is_github_models_base_url(base_url: Optional[str]) -> bool:
    normalized = (base_url or "").strip().rstrip("/").lower()
    return (
        normalized.startswith(COPILOT_BASE_URL)
        or normalized.startswith("https://models.github.ai/inference")
        or normalized.startswith("https://models.inference.ai.azure.com")
    )


def _fetch_github_models(api_key: Optional[str] = None, timeout: float = 5.0) -> Optional[list[str]]:
    catalog = fetch_github_model_catalog(api_key=api_key, timeout=timeout)
    if not catalog:
        return None
    return [item.get("id", "") for item in catalog if item.get("id")]


def _copilot_catalog_ids(
    catalog: Optional[list[dict[str, Any]]] = None,
    api_key: Optional[str] = None,
) -> set[str]:
    if catalog is None and api_key:
        catalog = fetch_github_model_catalog(api_key=api_key)
    if not catalog:
        return set()
    return {
        str(item.get("id") or "").strip()
        for item in catalog
        if str(item.get("id") or "").strip()
    }


def normalize_copilot_model_id(
    model_id: Optional[str],
    *,
    catalog: Optional[list[dict[str, Any]]] = None,
    api_key: Optional[str] = None,
) -> str:
    raw = str(model_id or "").strip()
    if not raw:
        return ""

    catalog_ids = _copilot_catalog_ids(catalog=catalog, api_key=api_key)
    alias = _COPILOT_MODEL_ALIASES.get(raw)
    if alias:
        return alias

    candidates = [raw]
    if "/" in raw:
        candidates.append(raw.split("/", 1)[1].strip())

    if raw.endswith("-mini"):
        candidates.append(raw[:-5])
    if raw.endswith("-nano"):
        candidates.append(raw[:-5])
    if raw.endswith("-chat"):
        candidates.append(raw[:-5])

    seen: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        if candidate in _COPILOT_MODEL_ALIASES:
            return _COPILOT_MODEL_ALIASES[candidate]
        if candidate in catalog_ids:
            return candidate

    if "/" in raw:
        return raw.split("/", 1)[1].strip()
    return raw


def _github_reasoning_efforts_for_model_id(model_id: str) -> list[str]:
    raw = (model_id or "").strip().lower()
    if raw.startswith(("openai/o1", "openai/o3", "openai/o4", "o1", "o3", "o4")):
        return list(COPILOT_REASONING_EFFORTS_O_SERIES)
    normalized = normalize_copilot_model_id(model_id).lower()
    if normalized.startswith("gpt-5"):
        return list(COPILOT_REASONING_EFFORTS_GPT5)
    return []


def _should_use_copilot_responses_api(model_id: str) -> bool:
    """Decide whether a Copilot model should use the Responses API.

    Replicates opencode's ``shouldUseCopilotResponsesApi``: GPT-5+ models use the Responses API
    except ``gpt-5-mini``; all non-GPT models (Claude, Gemini, ...) use Chat Completions.
    """
    import re

    match = re.match(r"^gpt-(\d+)", model_id)
    if not match:
        return False
    major = int(match.group(1))
    return major >= 5 and not model_id.startswith("gpt-5-mini")


def copilot_model_api_mode(
    model_id: Optional[str],
    *,
    catalog: Optional[list[dict[str, Any]]] = None,
    api_key: Optional[str] = None,
) -> str:
    """Determine the API mode for a Copilot model.

    Uses the model ID pattern (matching opencode's approach) as the primary signal. Falls back to
    the catalog's ``supported_endpoints`` only for models not covered by the pattern check.
    """
    # Fetch the catalog once so normalize + endpoint check share it
    # (avoids two redundant network calls for non-GPT-5 models).
    if catalog is None and api_key:
        catalog = fetch_github_model_catalog(api_key=api_key)

    normalized = normalize_copilot_model_id(model_id, catalog=catalog, api_key=api_key)
    if not normalized:
        return "chat_completions"

    # Primary: model ID pattern (matches opencode's shouldUseCopilotResponsesApi)
    if _should_use_copilot_responses_api(normalized):
        return "codex_responses"

    # Copilot's Claude models are exposed through its OpenAI-compatible chat
    # endpoint, not through Hermes' native Anthropic adapter. The live catalog may
    # advertise /v1/messages, but the Copilot token/header scheme is handled by
    # the OpenAI client path; selecting anthropic_messages would send the wrong
    # auth/wire shape. Keep non-GPT Copilot slots on chat_completions.
    return "chat_completions"


def azure_foundry_model_api_mode(model_name: Optional[str]) -> Optional[str]:
    """Infer Azure Foundry api_mode from a deployment/model name.

    Returns ``"codex_responses"`` when the model name matches a family that only accepts the
    Responses API on Azure Foundry (GPT-5.x, codex, o1/o3/o4 reasoning models).
    """
    raw = str(model_name or "").strip().lower()
    if not raw:
        return None
    # Strip any vendor/ prefix a user may have copied from OpenRouter / Copilot.
    if "/" in raw:
        raw = raw.rsplit("/", 1)[-1]
    # gpt-5-mini speaks chat completions on Copilot but Azure Foundry deploys
    # the full gpt-5 family uniformly on Responses API — don't carve an
    # exception here.
    for prefix in _AZURE_FOUNDRY_RESPONSES_PREFIXES:
        if raw.startswith(prefix):
            return "codex_responses"
    return None


def opencode_provider_family(provider_id: Optional[str]) -> Optional[str]:
    """Resolve a provider id to its OpenCode family, or None.

    ``opencode-go`` is checked before ``opencode-zen`` but the two slugs are not prefixes of each
    other, so order is cosmetic.
    """
    raw = str(provider_id or "").strip().lower()
    if not raw:
        return None
    canonical = normalize_provider(provider_id)
    if canonical in {"opencode-zen", "opencode-go", "opencode-free"}:
        return canonical
    if raw.startswith("opencode-free"):
        return "opencode-free"
    if raw.startswith("opencode-go"):
        return "opencode-go"
    if raw.startswith("opencode-zen"):
        return "opencode-zen"
    return None


def normalize_opencode_model_id(provider_id: Optional[str], model_id: Optional[str]) -> str:
    """Normalize OpenCode config IDs to the bare model slug used in API requests."""
    family = opencode_provider_family(provider_id)
    current = str(model_id or "").strip()
    if not current or family is None:
        return current

    prefix = f"{provider_id}/" if provider_id else f"{family}/"
    if current.lower().startswith(prefix.lower()):
        return current[len(prefix):]
    fallback_prefix = f"{family}/"
    if current.lower().startswith(fallback_prefix.lower()):
        return current[len(fallback_prefix):]
    return current


# OpenCode Zen free-tier models (``*-free`` slugs, e.g. x-preview-f-free /
# "Ox Alpha", plus unsuffixed free models like big-pickle) are served
# ANONYMOUSLY on the Zen relay: a request with no Authorization header
# succeeds, while ANY non-empty bearer the relay doesn't recognize is
# rejected with 401 "Invalid API key" — including our "no-key-required"
# placeholder and OpenCode GO subscription keys (the Go relay doesn't serve
# the free tier at all: "Model x is not supported").
# Verified live 2026-08-21 against POST /zen/v1/chat/completions.
OPENCODE_ZEN_FREE_KEYLESS_PLACEHOLDER = "opencode-zen-free-keyless"
_OPENCODE_ZEN_FREE_BASE_URL = "https://opencode.ai/zen/v1"

# Models whose slug carries ``-free`` but are NOT anonymous-servable: they are
# KEYED (Go-subscription) models and must be excluded from the keyless free
# catalog even though the suffix looks free. ox-alpha-free is the Go relay's
# subscription twin of the Zen keyless Ox Alpha (verified 2026-08-21).
_OPENCODE_FREE_KEYED_SUFFIX_MODELS = frozenset({"ox-alpha-free"})

# In-process memo for _fetch_opencode_free_models(): (fetched_at, ids-or-None).
# Direct provider_model_ids("opencode-free") callers (model validation, healing)
# can run several times per resolution — without this each would block on a
# network round-trip. Failures are memoized too (negative caching) so an
# unreachable relay doesn't stall every validation for `timeout` seconds.
_opencode_free_live_memo: Optional[tuple[float, Optional[list[str]]]] = None
_OPENCODE_FREE_LIVE_MEMO_TTL = 300.0  # 5 min; SWR disk cache handles the rest


def opencode_zen_free_headers() -> dict:
    """Client default_headers for anonymous OpenCode Zen free-tier requests.

    ``Authorization: ""`` overrides the OpenAI SDK's ``Bearer <api_key>`` header so the placeholder
    key never reaches the wire — the Zen relay accepts anonymous requests for free models but 401s
    any unknown bearer. Attribution headers mirror the opencode provider profile.
    """
    try:
        from hermes_cli import __version__ as _v
    except Exception:
        _v = "0"
    return {
        "Authorization": "",
        "HTTP-Referer": "https://hermes-agent.nousresearch.com",
        "X-Title": "Hermes Agent",
        "User-Agent": f"HermesAgent/{_v}",
    }


def _fetch_opencode_free_models(
    timeout: float = 8.0, *, force_refresh: bool = False
) -> Optional[list[str]]:
    """Fetch the live keyless OpenCode Free catalog from the Zen relay.

    The Zen ``/models`` dump also lists paid/subscription IDs (e.g. Go ``ox-alpha-free`` is KEYED
    despite the suffix), so a bare ``*-free`` suffix filter is not safe on its own — this mirrors
    the existing ``opencode_zen_free_runtime`` contract, which uses membership in the verified
    keyless catalog as the routing criterion.
    """
    import urllib.request

    from hermes_cli.urllib_security import open_credentialed_url

    now = time.time()
    if not force_refresh:
        memo = _opencode_free_live_memo
        if memo is not None and now - memo[0] < _OPENCODE_FREE_LIVE_MEMO_TTL:
            return list(memo[1]) if memo[1] else None

    url = f"{_OPENCODE_ZEN_FREE_BASE_URL.rstrip('/')}/models"
    req = urllib.request.Request(url)
    req.add_header("Accept", "application/json")
    for k, v in opencode_zen_free_headers().items():
        if k.lower() != "authorization":  # never send a bearer keylessly
            req.add_header(k, v)
    try:
        with open_credentialed_url(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
        items = data if isinstance(data, list) else data.get("data", [])
    except Exception:
        _set_opencode_free_live_memo(None)
        return None
    ids = [m["id"] for m in items if isinstance(m, dict) and isinstance(m.get("id"), str)]
    # Filter to the anonymous-servable free tier. The Zen dump can contain
    # keyed/Go IDs; only the verified free set belongs in the keyless picker.
    live_free = [
        mid
        for mid in ids
        if mid.lower().endswith("-free")
        and mid.lower() not in _OPENCODE_FREE_KEYED_SUFFIX_MODELS
    ]
    result = live_free if live_free else None
    _set_opencode_free_live_memo(result)
    return result


def _set_opencode_free_live_memo(ids: Optional[list[str]]) -> None:
    global _opencode_free_live_memo
    _opencode_free_live_memo = (time.time(), list(ids) if ids else None)


def _opencode_free_known_model_slugs() -> set[str]:
    """Lowercased keyless free-tier slugs known right now — WITHOUT network I/O.

    Union of the static ``_PROVIDER_MODELS["opencode-free"]`` floor, the in-process live memo, and
    the SWR disk-cache entry. Used by the ``opencode_zen_free_runtime`` healing path, which runs
    during model resolution and must never block on a live fetch.
    """
    known = {m.lower() for m in _PROVIDER_MODELS.get("opencode-free", [])}
    memo = _opencode_free_live_memo
    if memo is not None and memo[1]:
        known.update(m.lower() for m in memo[1])
    try:
        entry = _load_provider_models_cache().get("opencode-free") or {}
        known.update(str(m).lower() for m in entry.get("models", []) or [])
    except Exception:
        pass
    return known


def opencode_zen_free_runtime(provider_id: Optional[str], model_id: Optional[str]) -> Optional[dict]:
    """Keyless runtime entry for an OpenCode Zen free-tier model, or None.

    - ``provider_id`` is ``opencode-free`` (the dedicated keyless provider — EVERY model on it
    routes anonymously; that is the provider's contract), or - ``provider_id`` is any other
    OpenCode-family provider and ``model_id`` is in the VERIFIED keyless catalog
    (``_PROVIDER_MODELS["opencode-free"]``) — heals a free-model selection made under opencode-
    zen/opencode-go, whose keys the free tier rejects.

    Membership means the union of the cached LIVE keyless catalog (in-process memo / SWR disk cache
    — never a blocking fetch on this hot path) and the static floor, so a newly-live free model
    heals without a release.
    """
    family = opencode_provider_family(provider_id)
    if family is None:
        return None
    if family != "opencode-free":
        bare = normalize_opencode_model_id(provider_id, model_id).strip().lower()
        if bare not in _opencode_free_known_model_slugs():
            return None
    normalized = normalize_opencode_model_id(provider_id, model_id)
    api_mode = opencode_model_api_mode("opencode-zen", normalized)
    base_url = normalize_opencode_base_url(
        "opencode-zen", api_mode, _OPENCODE_ZEN_FREE_BASE_URL
    )
    return {
        "provider": family,
        "api_mode": api_mode,
        "base_url": base_url,
        "api_key": OPENCODE_ZEN_FREE_KEYLESS_PLACEHOLDER,
        "default_headers": opencode_zen_free_headers(),
        "source": "opencode-zen-free-keyless",
    }


def opencode_model_api_mode(provider_id: Optional[str], model_id: Optional[str]) -> str:
    """Determine the API mode for an OpenCode Zen / Go model.

    OpenCode routes models behind different surfaces per its Zen/Go docs: GPT/Codex/Grok and
    Muse Spark use ``/v1/responses`` (Muse Spark 503s on chat/completions); Claude and Qwen on
    Zen and MiniMax/Qwen on Go use ``/v1/messages``; everything else uses
    ``/v1/chat/completions``.
    """
    family = opencode_provider_family(provider_id)
    # opencode-free is Zen-hosted (the free tier lives on the Zen relay),
    # so it shares Zen's per-model endpoint routing.
    if family == "opencode-free":
        family = "opencode-zen"
    normalized = normalize_opencode_model_id(provider_id, model_id).lower()
    if not normalized:
        return "chat_completions"

    if family == "opencode-go":
        if normalized.startswith("gpt-") or normalized.startswith("grok-"):
            # GPT and Grok models on Go (gpt-5.6-luna, grok-4.5) are served
            # via /v1/responses per the published Go endpoint table, same as
            # GPT/Grok on Zen: https://opencode.ai/docs/go/#endpoints
            return "codex_responses"
        if normalized.startswith("muse-spark"):
            # Muse Spark (standard + contributor) is Responses-only on Go.
            # /v1/chat/completions returns HTTP 503 with an empty assistant
            # message; /v1/responses completes. See opencode.ai/docs/go.
            return "codex_responses"
        if normalized.startswith("minimax-"):
            return "anthropic_messages"
        if normalized.startswith("qwen"):
            # All Qwen models on Go (qwen3.7-max, qwen3.7-plus, qwen3.6-plus)
            # are served via /v1/messages per the published Go endpoint table.
            return "anthropic_messages"
        return "chat_completions"

    if family == "opencode-zen":
        if normalized.startswith("claude-"):
            return "anthropic_messages"
        if normalized.startswith("gpt-") or normalized.startswith("grok-"):
            # GPT-5/Codex and all Grok models on Zen (grok-4.6, grok-4.5,
            # grok-build-0.1) are served via /v1/responses per the Zen
            # endpoint table.
            return "codex_responses"
        if normalized.startswith("muse-spark"):
            # Standard Muse Spark on Zen is served via /v1/responses:
            # https://opencode.ai/docs/zen/#endpoints
            return "codex_responses"
        if normalized.startswith("qwen"):
            # Qwen models on Zen moved to /v1/messages per the published
            # Zen endpoint table.
            return "anthropic_messages"
        return "chat_completions"

    return "chat_completions"


def normalize_opencode_base_url(
    provider_id: Optional[str], api_mode: Optional[str], base_url: Optional[str]
) -> str:
    """Normalize an OpenCode Zen / Go base URL for the target API mode.

    Crucially this must be SYMMETRIC. The stripped URL gets persisted to config (``model.base_url``)
    by the TUI/desktop and gateway after switching into an anthropic-routed model (e.g. minimax-m2.7
    on Go).

    Only opencode.ai-hosted URLs are re-suffixed; custom proxy overrides via ``OPENCODE_*_BASE_URL``
    are left alone unless they already carry ``/v1``.
    """
    url = str(base_url or "").strip().rstrip("/")
    if not url:
        return url
    if opencode_provider_family(provider_id) is None:
        return url

    import re as _re

    if api_mode == "anthropic_messages":
        return _re.sub(r"/v1$", "", url)

    # chat_completions / codex_responses: ensure the /v1 suffix is present on
    # official opencode.ai hosts (heals a persisted anthropic-stripped URL).
    if url.endswith("/v1"):
        return url
    try:
        host = urllib.parse.urlparse(url).netloc.lower()
    except Exception:
        host = ""
    if host == "opencode.ai" or host.endswith(".opencode.ai"):
        return url + "/v1"
    return url


def github_model_reasoning_efforts(
    model_id: Optional[str],
    *,
    catalog: Optional[list[dict[str, Any]]] = None,
    api_key: Optional[str] = None,
) -> list[str]:
    """Return supported reasoning-effort levels for a Copilot-visible model."""
    normalized = normalize_copilot_model_id(model_id, catalog=catalog, api_key=api_key)
    if not normalized:
        return []

    catalog_entry = None
    if catalog is not None:
        catalog_entry = next((item for item in catalog if item.get("id") == normalized), None)
    elif api_key:
        fetched_catalog = fetch_github_model_catalog(api_key=api_key)
        if fetched_catalog:
            catalog_entry = next((item for item in fetched_catalog if item.get("id") == normalized), None)

    if catalog_entry is not None:
        capabilities = catalog_entry.get("capabilities")
        if isinstance(capabilities, dict):
            supports = capabilities.get("supports")
            if isinstance(supports, dict):
                efforts = supports.get("reasoning_effort")
                if isinstance(efforts, list):
                    normalized_efforts = [
                        str(effort).strip().lower()
                        for effort in efforts
                        if str(effort).strip()
                    ]
                    return list(dict.fromkeys(normalized_efforts))
            return []
        legacy_capabilities = {
            str(capability).strip().lower()
            for capability in catalog_entry.get("capabilities", [])
            if str(capability).strip()
        }
        if "reasoning" not in legacy_capabilities:
            return []

    return _github_reasoning_efforts_for_model_id(str(model_id or normalized))


def _probe_result(models, probed_url, resolved_base_url, suggested_base_url=None, used_fallback=False) -> dict[str, Any]:
    return {
        "models": models,
        "probed_url": probed_url,
        "resolved_base_url": resolved_base_url,
        "suggested_base_url": suggested_base_url,
        "used_fallback": used_fallback,
    }


def probe_api_models(
    api_key: Optional[str],
    base_url: Optional[str],
    timeout: float = 5.0,
    api_mode: Optional[str] = None,
    request_headers: Optional[dict[str, str]] = None,
) -> dict[str, Any]:
    """Probe a ``/models`` endpoint with light URL heuristics (``base`` then ``base±/v1``).

    For ``anthropic_messages`` mode, sends ``x-api-key`` and ``anthropic-version`` headers
    instead of ``Authorization: Bearer``; the response shape (``data[].id``) is identical so one
    parser serves both. ``models`` is None when no candidate answered.
    """
    normalized = (base_url or "").strip().rstrip("/")
    if not normalized:
        return _probe_result(None, None, "")

    if _is_github_models_base_url(normalized):
        return _probe_result(_fetch_github_models(api_key=api_key, timeout=timeout), COPILOT_MODELS_URL, COPILOT_BASE_URL)

    if normalized.endswith("/v1"):
        alternate_base = normalized[:-3].rstrip("/")
    else:
        alternate_base = normalized + "/v1"

    candidates: list[tuple[str, bool]] = [(normalized, False)]
    if alternate_base and alternate_base != normalized:
        candidates.append((alternate_base, True))

    tried: list[str] = []
    headers: dict[str, str] = {"User-Agent": _HERMES_USER_AGENT}
    if urllib.parse.urlparse(normalized).hostname == "generativelanguage.googleapis.com":
        headers["X-Goog-Api-Client"] = f"hermes-agent/{_HERMES_VERSION}"
    if api_key and api_mode == "anthropic_messages":
        headers["x-api-key"] = api_key
        headers["anthropic-version"] = "2023-06-01"
    elif api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if normalized.startswith(COPILOT_BASE_URL):
        headers.update(copilot_default_headers())
    if isinstance(request_headers, dict):
        # Per-provider custom headers can contain auth/proxy secrets. Merge last so
        # endpoint-specific config wins, and never log the values.
        from hermes_cli.config import normalize_extra_headers

        headers.update(normalize_extra_headers(request_headers))

    # Only thread ssl_context when a per-provider TLS override applies; public/unconfigured
    # endpoints keep the original 2-arg call so existing call-seam mocks stay valid.
    _open_kwargs: dict[str, Any] = {"timeout": timeout}
    _ssl_context = _custom_provider_ssl_context(normalized)
    if _ssl_context is not None:
        _open_kwargs["ssl_context"] = _ssl_context
    for candidate_base, is_fallback in candidates:
        url = candidate_base.rstrip("/") + "/models"
        tried.append(url)
        req = urllib.request.Request(url, headers=headers)
        try:
            with _urlopen_model_catalog_request(req, **_open_kwargs) as resp:
                data = json.loads(resp.read().decode())
                return _probe_result(
                    [m.get("id", "") for m in data.get("data", [])],
                    url,
                    candidate_base.rstrip("/"),
                    alternate_base if alternate_base != candidate_base else normalized,
                    is_fallback,
                )
        except Exception:
            continue

    return _probe_result(
        None,
        tried[0] if tried else normalized.rstrip("/") + "/models",
        normalized,
        alternate_base if alternate_base != normalized else None,
    )


# Legacy filter — used when an item has no surface tag (rolling out
# 2026-05). Once every model returned by the catalog endpoint carries an
# explicit surface tag (``chat``/``embed``/``image-gen``/``tts``/``stt``)
# the regex path becomes unreachable and can be removed.
_DEEPINFRA_EXCLUDE_RE = re.compile(
    r"(?i)(embed|rerank|whisper|stable-diffusion|flux|sdxl|"
    r"tts|bark|speech|image-gen|clip|vit-|dpt-)",
)

# Surface tags announce *what kind of model* this is. When none of these
# are present on a catalog entry, the tags array only carries capability
# tags (``reasoning``, ``vision``, ``prompt_cache``, …) and we have to
# fall back to id-regex inference for the chat surface.
_DEEPINFRA_SURFACE_TAGS: frozenset[str] = frozenset({
    "chat", "embed", "image-gen", "tts", "stt", "video-gen",
})

_DEEPINFRA_DEFAULT_BASE_URL = "https://api.deepinfra.com/v1/openai"
_DEEPINFRA_MODELS_QUERY = "filter=true&sort_by=hermes"

# Module-level cache for the full tagged catalog response, keyed by base URL.
# Each value is the parsed ``data`` list. Surface-specific filters read from
# this cache so a single network round-trip serves chat / image-gen / tts /
# stt callers across the whole process lifetime.
_deepinfra_catalog_cache: dict[str, list[dict]] = {}

# Negative cache: monotonic timestamp of the last failed fetch, keyed by base
# URL. Without this, an unreachable catalog (offline / DNS / firewall) makes
# every surface helper (chat picker, pricing, image/video/tts/stt defaults,
# vision) re-attempt a fresh blocking fetch that eats the full timeout each
# time — several sequential stalls in one user-visible operation. A short TTL
# lets connectivity recover without a process restart.
_deepinfra_catalog_neg_cache: dict[str, float] = {}
_DEEPINFRA_CATALOG_NEG_TTL = 60.0  # seconds


def _deepinfra_catalog_url() -> tuple[str, str]:
    """Return ``(cache_key, full_url)`` for the DeepInfra catalog endpoint."""
    base = os.getenv("DEEPINFRA_BASE_URL", "").strip() or _DEEPINFRA_DEFAULT_BASE_URL
    cache_key = base.rstrip("/")
    return cache_key, f"{cache_key}/models?{_DEEPINFRA_MODELS_QUERY}"


def _fetch_deepinfra_catalog(
    *,
    timeout: float = 5.0,
    force_refresh: bool = False,
) -> Optional[list[dict]]:
    """Fetch the raw DeepInfra catalog list with module-level caching.

    The endpoint serves chat, embed, image-gen, TTS and STT models in one response. Auth is
    optional but a Bearer token is attached when available so user-scoped catalogs (private
    fine-tunes) show.
    """
    cache_key, url = _deepinfra_catalog_url()
    if not force_refresh:
        if cache_key in _deepinfra_catalog_cache:
            return _deepinfra_catalog_cache[cache_key]
        last_fail = _deepinfra_catalog_neg_cache.get(cache_key)
        if last_fail is not None and (time.monotonic() - last_fail) < _DEEPINFRA_CATALOG_NEG_TTL:
            return None

    headers: dict[str, str] = {"User-Agent": _HERMES_USER_AGENT}
    api_key = os.getenv("DEEPINFRA_API_KEY", "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(url, headers=headers)
    try:
        with _urlopen_model_catalog_request(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode())
    except Exception:
        _deepinfra_catalog_neg_cache[cache_key] = time.monotonic()
        return None

    data = payload.get("data")
    if not isinstance(data, list):
        _deepinfra_catalog_neg_cache[cache_key] = time.monotonic()
        return None

    _deepinfra_catalog_cache[cache_key] = data
    _deepinfra_catalog_neg_cache.pop(cache_key, None)
    return data


def _fetch_deepinfra_models_by_tag(
    tag: str,
    *,
    timeout: float = 5.0,
    force_refresh: bool = False,
) -> Optional[list[dict]]:
    """Return DeepInfra models whose ``metadata.tags`` includes *tag*.

    Each item is ``{"id", "metadata"}`` so callers can inspect context length, pricing and
    units. For the chat surface, items with no ``tags`` field fall through to the legacy name-
    regex exclusion so this keeps working while the tag rollout is in flight. Returns ``None``
    on network failure.
    """
    data = _fetch_deepinfra_catalog(timeout=timeout, force_refresh=force_refresh)
    if data is None:
        return None

    matched: list[dict] = []
    for item in data:
        mid = item.get("id")
        if not mid:
            continue
        # ``metadata is None`` means DeepInfra returns a stub without
        # pricing/context — typically a model that's listed but not
        # served. Skip those for every surface.
        raw_metadata = item.get("metadata")
        if raw_metadata is None:
            continue
        metadata = raw_metadata if isinstance(raw_metadata, dict) else {}
        raw_tags = metadata.get("tags")
        tags = raw_tags if isinstance(raw_tags, list) else []
        has_surface_tag = any(t in _DEEPINFRA_SURFACE_TAGS for t in tags)

        if has_surface_tag:
            if tag in tags:
                matched.append({"id": mid, "metadata": metadata})
            continue
        # Surface-tag rollout incomplete — fall back to id-regex inference.
        # Only meaningful for the chat surface; embed/image-gen/tts/stt
        # cannot be safely inferred from an id alone.
        if tag == "chat" and not _DEEPINFRA_EXCLUDE_RE.search(mid):
            matched.append({"id": mid, "metadata": metadata})

    return matched


def _fetch_deepinfra_models(
    timeout: float = 5.0,
    *,
    force_refresh: bool = False,
) -> Optional[list[str]]:
    """Return DeepInfra chat-model ids (tag-aware, regex fallback).

    Thin wrapper over :func:`_fetch_deepinfra_models_by_tag` so historical callers in
    :func:`provider_model_ids` keep their string-list contract. Returns ``None`` on network failure,
    an empty list if the catalog contains no chat-tagged ids (which would itself be surprising).
    """
    items = _fetch_deepinfra_models_by_tag(
        "chat", timeout=timeout, force_refresh=force_refresh
    )
    if items is None:
        return None
    return [item["id"] for item in items] or None


def deepinfra_model_ids(tag: str, *, force_refresh: bool = False) -> list[str]:
    """Return DeepInfra model ids carrying surface *tag* (``[]`` on failure)."""
    items = _fetch_deepinfra_models_by_tag(tag, force_refresh=force_refresh)
    return [item["id"] for item in items] if items else []


def deepinfra_base_url(section: Optional[dict] = None) -> str:
    """Resolve the DeepInfra OpenAI-compatible base URL, normalized.

    Precedence: config-section ``base_url`` → ``DEEPINFRA_BASE_URL`` env → default. Always stripped
    with any trailing slash removed.
    """
    candidate = section.get("base_url") if isinstance(section, dict) else None
    value = candidate or os.getenv("DEEPINFRA_BASE_URL") or _DEEPINFRA_DEFAULT_BASE_URL
    return str(value).strip().rstrip("/")


def _fetch_ai_gateway_models(timeout: float = 5.0) -> Optional[list[str]]:
    """Fetch available language models with tool-use from AI Gateway."""
    api_key = os.getenv("AI_GATEWAY_API_KEY", "").strip()
    if not api_key:
        return None
    base_url = os.getenv("AI_GATEWAY_BASE_URL", "").strip()
    if not base_url:
        from hermes_constants import AI_GATEWAY_BASE_URL
        base_url = AI_GATEWAY_BASE_URL

    url = base_url.rstrip("/") + "/models"
    headers: dict[str, str] = {
        "Authorization": f"Bearer {api_key}",
        "User-Agent": _HERMES_USER_AGENT,
    }
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
            return [
                m["id"]
                for m in data.get("data", [])
                if m.get("id")
                and m.get("type") == "language"
                and "tool-use" in (m.get("tags") or [])
            ]
    except Exception:
        return None


def fetch_api_models(
    api_key: Optional[str],
    base_url: Optional[str],
    timeout: float = 5.0,
    api_mode: Optional[str] = None,
    headers: Optional[dict[str, str]] = None,
) -> Optional[list[str]]:
    """Fetch the list of available model IDs from the provider's ``/models`` endpoint."""
    return probe_api_models(
        api_key,
        base_url,
        timeout=timeout,
        api_mode=api_mode,
        request_headers=headers,
    ).get("models")


def _custom_endpoint_fingerprint(
    api_key: Optional[str],
    api_mode: Optional[str],
    headers: Optional[dict[str, str]],
) -> str:
    """Fingerprint the credentials/wire-shape used to probe a custom endpoint.

    Custom OpenAI-compatible endpoints have no ``PROVIDER_REGISTRY`` slug to key off (unlike
    ``_credential_fingerprint``), so this hashes exactly the values callers pass to
    :func:`fetch_api_models`: a rotated ``api_key``, a changed ``api_mode``, or an edited
    ``extra_headers`` block each bust the cache entry on their own.
    """
    import hashlib

    blob = "|".join((
        api_key or "",
        api_mode or "",
        json.dumps(headers or {}, sort_keys=True),
    )).encode("utf-8", errors="replace")
    # blake2b for cache-key fingerprinting only, same rationale as
    # _credential_fingerprint (avoids CodeQL's sha256-over-secrets rule).
    return hashlib.blake2b(blob, digest_size=8).hexdigest()


def _cache_entry_valid(
    entry: Any,
    fp: str,
    *,
    allow_empty: bool = False,
) -> "TypeGuard[dict[str, Any]]":
    """True when *entry* is a well-formed cache row for fingerprint *fp*.

    Requires a numeric ``at`` so corrupt disk state (hand-edited JSON with ``"at": "yesterday"`` or
    ``null``) degrades to a cache miss / live fetch instead of raising out of the wrapper. Empty
    model lists are valid only for callers that explicitly opt into an authoritative empty catalog.
    """
    return (
        isinstance(entry, dict)
        and entry.get("fp") == fp
        and isinstance(entry.get("models"), list)
        and (allow_empty or bool(entry["models"]))
        and isinstance(entry.get("at"), (int, float))
        and not isinstance(entry.get("at"), bool)
    )


def cached_fetch_api_models(
    api_key: Optional[str],
    base_url: Optional[str],
    *,
    timeout: float = 5.0,
    api_mode: Optional[str] = None,
    headers: Optional[dict[str, str]] = None,
    force_refresh: bool = False,
    cache_only: bool = False,
    ttl_seconds: int = _PROVIDER_MODELS_CACHE_TTL,
) -> Optional[list[str]]:
    """Disk-cached wrapper around :func:`fetch_api_models` for custom endpoints.

    Callers that deliberately skip live probing for latency reasons (GUI picker opens, which must
    not block on a stopped local endpoint) use this so a warm catalog still reaches the picker
    instead of collapsing to the config-declared subset.
    """
    normalized_url = str(base_url or "").strip().rstrip("/").lower()
    if not normalized_url:
        if cache_only:
            return None
        # No base_url means nothing to key the cache on — fall through to a
        # live call so callers keep getting fetch_api_models' own behavior.
        return fetch_api_models(
            api_key, base_url, timeout=timeout, api_mode=api_mode, headers=headers
        )

    cache_key = f"custom:{normalized_url}"
    fp = _custom_endpoint_fingerprint(api_key, api_mode, headers)
    cache = _load_provider_models_cache()
    entry = cache.get(cache_key)
    now = time.time()

    if cache_only:
        # Same trust window as the stale-while-revalidate tier below, minus
        # the revalidation: an entry this side of the bound is good enough to
        # render, and anything older is treated as a miss so the caller falls
        # back to its configured list rather than showing a stale catalog.
        if force_refresh or not _cache_entry_valid(entry, fp):
            return None
        if now - entry["at"] >= _PROVIDER_MODELS_STALE_SERVE_MAX:
            return None
        return list(entry["models"])

    if not force_refresh and _cache_entry_valid(entry, fp):
        age = now - entry["at"]
        if age < ttl_seconds:
            return list(entry["models"])
        if age < _PROVIDER_MODELS_STALE_SERVE_MAX:
            # Stale-while-revalidate: serve the expired entry immediately so
            # picker opens never block on a live /v1/models round-trip
            # (#72762's stall class, which a plain TTL would reintroduce an
            # hour into the session); refresh off-thread for the next open.
            def _refresh_custom():
                live = fetch_api_models(
                    api_key, base_url,
                    timeout=timeout, api_mode=api_mode, headers=headers,
                )
                if not live:
                    return None
                return {"fp": fp, "at": time.time(), "models": list(live)}

            _spawn_swr_refresh(cache_key, _refresh_custom)
            return list(entry["models"])

    live = fetch_api_models(
        api_key, base_url, timeout=timeout, api_mode=api_mode, headers=headers
    )
    if live:
        cache[cache_key] = {"fp": fp, "at": now, "models": list(live)}
        _save_provider_models_cache(cache)
        return list(live)

    # Live fetch returned nothing (offline endpoint, timeout, auth hiccup).
    # A stale same-fingerprint entry beats an empty result.
    if _cache_entry_valid(entry, fp):
        return list(entry["models"])
    return live


# ---------------------------------------------------------------------------
# Ollama Cloud — merged model discovery with disk cache
# ---------------------------------------------------------------------------


