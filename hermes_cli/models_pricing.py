"""Live model pricing: OpenRouter-compatible /v1/models pricing fetch + per-endpoint/credential cache, Nous Portal sale chrome and org-policy filtering, Vercel AI Gateway / Novita / Fireworks / DeepInfra pricing adapters.

Split out of ``hermes_cli.models``; every moved name is re-imported there, so
``hermes_cli.models.<name>`` keeps resolving (and monkeypatching) as before.
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from typing import Any, Optional
from hermes_cli.models_reasoning_caps import _seed_reasoning_caps


# Cache: maps model_id → {"prompt": str, "completion": str} per endpoint
_pricing_cache: dict[str, dict[str, dict[str, str]]] = {}


# A failed fetch caches its empty result too, so an unreachable endpoint isn't
# re-dialed on every call — but only until this deadline. Cached forever, one
# bad moment (a blip during startup, a key that hadn't been written yet) turns
# into no live model discovery for the life of the process, and the processes
# that read this most are the ones that run for weeks: the gateway, the desktop
# backend. Every caller falls back to a curated list meanwhile, so the cost of
# the stale entry is silent and invisible.
_FAILED_CATALOG_TTL_SECONDS = 120.0


_pricing_cache_retry_after: dict[str, float] = {}


def _cached_catalog(cache_key: str) -> Optional[dict[str, dict[str, Any]]]:
    """The cached catalog for *cache_key*, or None to go fetch it."""
    from hermes_cli.models import _pricing_cache, _pricing_cache_retry_after
    cached = _pricing_cache.get(cache_key)
    if cached is None:
        return None
    retry_after = _pricing_cache_retry_after.get(cache_key)
    if retry_after is not None and time.monotonic() >= retry_after:
        _pricing_cache.pop(cache_key, None)
        _pricing_cache_retry_after.pop(cache_key, None)
        return None
    return cached


def _cache_catalog(
    cache_key: str,
    result: dict[str, dict[str, Any]],
    ttl_seconds: Optional[float] = None,
) -> dict[str, dict[str, Any]]:
    """Cache a catalog result, giving an empty one an expiry.

    *ttl_seconds* expires a non-empty result too. Only a catalog whose contents depend on server-
    side state the client cannot observe needs it — an org's model policy can change while a long-
    lived process holds the entry.
    """
    from hermes_cli.models import _pricing_cache, _pricing_cache_retry_after
    _pricing_cache[cache_key] = result
    if result:
        if ttl_seconds:
            _pricing_cache_retry_after[cache_key] = time.monotonic() + ttl_seconds
        else:
            _pricing_cache_retry_after.pop(cache_key, None)
    else:
        _pricing_cache_retry_after[cache_key] = (
            time.monotonic() + _FAILED_CATALOG_TTL_SECONDS
        )
    return result


# NUL cannot appear in a URL, so this cannot collide with a real base URL.
_PRICING_AUTH_KEY_PREFIX = "\x00auth:"


def _pricing_auth_fingerprint(api_key: str | None) -> str:
    """Key suffix identifying the credential a catalog was read with.

    A governed endpoint answers each token with the catalog its org may reach, so two credentials
    cannot share an entry. blake2b for cache-key fingerprinting only, same rationale as
    :func:`_custom_endpoint_fingerprint`.
    """
    if not api_key:
        return ""
    import hashlib

    digest = hashlib.blake2b(api_key.encode("utf-8", errors="replace"), digest_size=8)
    return _PRICING_AUTH_KEY_PREFIX + digest.hexdigest()


def peek_cached_pricing(base_url: str) -> dict[str, dict[str, Any]]:
    """Pricing already cached for *base_url*, or ``{}``. Never fetches.

    Accepts a ``/v1``-suffixed URL as well as the pre-``/v1`` root the fetchers key on, and
    prefers an authenticated catalog. Scans rather than rebuilding a key because callers hold no
    credential — newest first, skipping expired entries, so a rotated credential does not keep
    answering from the catalog its predecessor read.
    """
    from hermes_cli.models import _pricing_cache
    root = (base_url or "").rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3].rstrip("/")
    authed_prefix = root + _PRICING_AUTH_KEY_PREFIX
    for key in reversed(list(_pricing_cache)):
        if key.startswith(authed_prefix):
            cached = _cached_catalog(key)
            if cached:
                return cached
    return _cached_catalog(root) or {}


def _format_price_per_mtok(per_token_str: str) -> str:
    """Convert a per-token price string to a human-friendly $/Mtok string.

    Always uses 2 decimal places so that prices align vertically when right-justified in a column
    (the decimal point stays in the same position).

    Sub-cent prices (e.g. deep-discount cache-hit promos) extend precision instead of collapsing to
    "$0.00": the smallest decimal place that makes the value non-zero is found, then one extra digit
    is kept and trailing zeros trimmed.
    """
    try:
        val = float(per_token_str)
    except (TypeError, ValueError):
        return "?"
    if val == 0:
        return "free"
    per_m = val * 1_000_000
    text = f"{per_m:.2f}"
    if per_m < 0.01:
        # Non-zero price below one cent per Mtok — widen precision until the
        # value shows, keep one extra significant digit, trim trailing zeros.
        prec = 3
        while prec < 12 and round(per_m, prec) == 0:
            prec += 1
        text = f"{per_m:.{min(prec + 1, 12)}f}".rstrip("0").rstrip(".")
    return f"${text}"


def compute_sale_discount(
    prompt: str,
    completion: str,
    original: Any,
) -> tuple[int, str, str] | None:
    """Derive sale chrome from gateway ``pricing.original`` when cheaper.

    Nous Portal-only feature: callers gate on the provider; this helper only sees ``original``
    because the Nous fetch path opted in via ``include_sale_original=True``.

    Returns ``(discount_percent, was_prompt_raw, was_completion_raw)`` only when ``original`` is a
    dict and the current prompt (fallback: completion) rate is strictly below the corresponding
    original.
    """
    def _finite(raw: Any) -> float | None:
        try:
            n = float(raw)
        except (TypeError, ValueError):
            return None
        return n if n > 0 and n == n else None  # n == n rejects NaN

    def _nonneg(raw: Any) -> float | None:
        try:
            n = float(raw)
        except (TypeError, ValueError):
            return None
        return n if n >= 0 and n == n else None

    orig_dict = original if isinstance(original, dict) else {}
    was_prompt = orig_dict.get("prompt")
    was_completion = orig_dict.get("completion")

    # Free / $0 models: flat 100% off, with "was" prices only when the
    # gateway actually served an original (e.g. a :free sibling); a
    # natively-free model (stealth/ox-alpha) gets bare "-100%" chrome.
    cur_prompt_any = _nonneg(prompt) if prompt not in (None, "") else None
    cur_comp_any = _nonneg(completion) if completion not in (None, "") else None
    if cur_prompt_any == 0 and cur_comp_any in (0, None):
        return (
            100,
            str(was_prompt) if was_prompt not in (None, "") else "",
            str(was_completion) if was_completion not in (None, "") else "",
        )

    if not isinstance(original, dict):
        return None

    if was_prompt in (None, "") and was_completion in (None, ""):
        return None

    cur_prompt = _finite(prompt) if prompt not in (None, "") else None
    orig_prompt = _finite(was_prompt) if was_prompt not in (None, "") else None
    if cur_prompt is not None and orig_prompt is not None and cur_prompt < orig_prompt:
        pct = int(round((1.0 - (cur_prompt / orig_prompt)) * 100))
        if pct < 1:
            return None
        return (
            pct,
            str(was_prompt),
            str(was_completion) if was_completion not in (None, "") else "",
        )

    cur_comp = _finite(completion) if completion not in (None, "") else None
    orig_comp = _finite(was_completion) if was_completion not in (None, "") else None
    if cur_comp is not None and orig_comp is not None and cur_comp < orig_comp:
        pct = int(round((1.0 - (cur_comp / orig_comp)) * 100))
        if pct < 1:
            return None
        return (
            pct,
            str(was_prompt) if was_prompt not in (None, "") else "",
            str(was_completion),
        )

    return None


def fetch_models_with_pricing(
    api_key: str | None = None,
    base_url: str = "https://openrouter.ai/api",
    timeout: float = 8.0,
    *,
    force_refresh: bool = False,
    include_sale_original: bool = False,
    cache_ttl_seconds: Optional[float] = None,
) -> dict[str, dict[str, Any]]:
    """Fetch ``/v1/models`` and return ``{model_id: {prompt, completion, ...}}``.

    Results are cached per *base_url* and per credential, so repeated calls are free and one
    caller's catalog never answers another's read. Works with any OpenRouter-compatible endpoint
    (OpenRouter, Nous Portal).

    When *include_sale_original* is true (Nous Portal only) and the gateway advertises a global
    discount under ``pricing.original``, those pre-discount rates are copied through as a nested
    ``original`` dict so pickers can show sale chrome.
    """
    from hermes_cli.models import _HERMES_USER_AGENT, _urlopen_model_catalog_request
    url_root = (base_url or "").rstrip("/")
    cache_key = url_root + _pricing_auth_fingerprint(api_key)
    if not force_refresh:
        cached = _cached_catalog(cache_key)
        if cached is not None:
            return cached

    url = url_root + "/v1/models"
    headers: dict[str, str] = {
        "Accept": "application/json",
        "User-Agent": _HERMES_USER_AGENT,
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        req = urllib.request.Request(url, headers=headers)
        with _urlopen_model_catalog_request(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode())
    except Exception:
        return _cache_catalog(cache_key, {})

    # Same document the reasoning-capability fetch would pull, and every
    # picker/pricing surface goes through here — mirror it so a later hot-path
    # lookup (and the next process) has an answer without its own round-trip.
    _seed_reasoning_caps(url, payload.get("data"))

    result: dict[str, dict[str, Any]] = {}
    for item in payload.get("data", []):
        mid = item.get("id")
        pricing = item.get("pricing")
        if mid and isinstance(pricing, dict):
            entry: dict[str, Any] = {
                "prompt": str(pricing.get("prompt", "")),
                "completion": str(pricing.get("completion", "")),
            }
            if pricing.get("input_cache_read"):
                entry["input_cache_read"] = str(pricing["input_cache_read"])
            if pricing.get("input_cache_write"):
                entry["input_cache_write"] = str(pricing["input_cache_write"])
            # Sale chrome is Nous Portal-only. Never copy pricing.original for
            # OpenRouter / other OpenAI-compatible catalogs.
            if include_sale_original:
                original = pricing.get("original")
                if isinstance(original, dict):
                    orig_entry: dict[str, str] = {}
                    for key in (
                        "prompt",
                        "completion",
                        "input_cache_read",
                        "input_cache_write",
                    ):
                        if original.get(key) not in (None, ""):
                            orig_entry[key] = str(original[key])
                    if orig_entry.get("prompt") or orig_entry.get("completion"):
                        entry["original"] = orig_entry
            result[mid] = entry

    return _cache_catalog(cache_key, result, cache_ttl_seconds)


def fetch_ai_gateway_pricing(
    timeout: float = 8.0,
    *,
    force_refresh: bool = False,
) -> dict[str, dict[str, str]]:
    """Fetch Vercel AI Gateway /v1/models and return hermes-shaped pricing.

    Vercel uses ``input`` / ``output`` field names; hermes's picker expects ``prompt`` /
    ``completion``. This translates. Cache read/write field names already match.
    """
    from hermes_constants import AI_GATEWAY_BASE_URL

    cache_key = AI_GATEWAY_BASE_URL.rstrip("/")
    if not force_refresh:
        cached = _cached_catalog(cache_key)
        if cached is not None:
            return cached

    try:
        req = urllib.request.Request(
            f"{cache_key}/models",
            headers={"Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode())
    except Exception:
        return _cache_catalog(cache_key, {})

    result: dict[str, dict[str, str]] = {}
    for item in payload.get("data", []):
        if not isinstance(item, dict):
            continue
        mid = item.get("id")
        pricing = item.get("pricing")
        if not (mid and isinstance(pricing, dict)):
            continue
        entry: dict[str, str] = {
            "prompt": str(pricing.get("input", "")),
            "completion": str(pricing.get("output", "")),
        }
        if pricing.get("input_cache_read"):
            entry["input_cache_read"] = str(pricing["input_cache_read"])
        if pricing.get("input_cache_write"):
            entry["input_cache_write"] = str(pricing["input_cache_write"])
        result[mid] = entry

    return _cache_catalog(cache_key, result)


def _resolve_openrouter_api_key() -> str:
    """Best-effort OpenRouter API key for pricing fetch."""
    return os.getenv("OPENROUTER_API_KEY", "").strip()


_DEFAULT_NOUS_INFERENCE_BASE = "https://inference-api.nousresearch.com"


def _resolve_nous_pricing_credentials() -> tuple[str, str]:
    """Return ``(api_key, base_url)`` for Nous Portal pricing.

    Base URL precedence (mirrors runtime credential resolution): 1. ``NOUS_INFERENCE_BASE_URL`` env
    override (staging / preview) 2. Resolved runtime credential ``base_url`` 3. Production default

    Without (1), a staging profile's sale ``pricing.original`` never reaches the pickers — the
    anonymous fallback would hit prod, which has no ``original`` field.
    """
    env_base = None
    try:
        from hermes_cli.auth import _nous_inference_env_override

        env_base = _nous_inference_env_override()
    except Exception:
        env_base = None

    api_key = ""
    creds_base = ""
    try:
        from hermes_cli.auth import resolve_nous_runtime_credentials

        creds = resolve_nous_runtime_credentials()
        if creds:
            api_key = creds.get("api_key", "") or ""
            creds_base = (creds.get("base_url", "") or "").strip()
    except Exception:
        pass

    base_url = (env_base or creds_base or _DEFAULT_NOUS_INFERENCE_BASE).rstrip("/")
    # Credential bases arrive with or without the ``/v1`` suffix. Callers
    # append their own path, so hand back the bare origin.
    if base_url.endswith("/v1"):
        base_url = base_url[:-3]
    return (api_key, base_url)


def nous_policy_allowed_ids(*, force_refresh: bool = False) -> Optional[set[str]]:
    """The Nous model ids the caller's org may reach, or ``None`` to not filter.

    The gateway omits policy-blocked rows from an authenticated ``GET /v1/models``, so that
    response's keys are the reachable set.

    ``None`` means "leave the caller's list alone", for the three states that cannot support
    narrowing one: no policy (or a token too old to say), an anonymous read whose catalog is
    unfiltered, and an empty read, which is a fetch failure rather than an org that may reach
    nothing.
    """
    from hermes_cli.models import _resolve_nous_pricing_credentials, fetch_models_with_pricing
    try:
        from hermes_cli.nous_account import nous_policy_present

        if nous_policy_present() is not True:
            return None
    except Exception:
        return None

    api_key, base_url = _resolve_nous_pricing_credentials()
    if not api_key or not base_url:
        return None

    # Same arguments as get_pricing_for_provider's nous branch, so a caller
    # asking for pricing too shares this entry instead of paying for a second
    # request.
    pricing = fetch_models_with_pricing(
        api_key=api_key,
        base_url=base_url,
        force_refresh=force_refresh,
        include_sale_original=True,
        cache_ttl_seconds=_NOUS_CATALOG_TTL_SECONDS,
    )
    return set(pricing) or None


# Past this size an allowed set reads as a whole catalog rather than an
# allowlist, and is not worth showing in place of an empty picker.
_NOUS_POLICY_APPEND_MAX = 64


# How long a Nous catalog stays trusted. Its contents depend on the org's
# policy, which an admin can change at any time and the client cannot observe,
# so a long-lived process must re-ask instead of holding the first answer for
# its whole life. Other providers' catalogs carry no such state and keep the
# default no-expiry caching.
_NOUS_CATALOG_TTL_SECONDS = 300.0


def restrict_to_nous_policy(
    model_ids: list[str],
    allowed: Optional[set[str]],
    *,
    rescue_empty: bool = False,
) -> list[str]:
    """*model_ids* narrowed to *allowed*, preserving the caller's order.

    A ``:free`` sibling is kept when its base model is reachable, mirroring the gateway, which
    admits a row when any of its requestable ids passes. Prefer over-listing: that costs a 403 from
    the authoritative gate, while hiding a row the gate would serve is unrecoverable from the
    client.
    """
    if not allowed:
        return list(model_ids)
    kept = [
        mid
        for mid in model_ids
        if mid in allowed or mid.split(":", 1)[0] in allowed
    ]

    # An allowlist can name only models the curated manifest lacks, leaving an
    # empty picker — worse than no filter, since the models the org may use are
    # the ones dropped. Opt-in per list: an already-empty list (a paid tier's
    # gated models) means "nothing to gate", not "nothing survived".
    if rescue_empty and not kept and len(allowed) <= _NOUS_POLICY_APPEND_MAX:
        return sorted(allowed)
    return kept


def get_pricing_for_provider(provider: str, *, force_refresh: bool = False) -> dict[str, dict[str, str]]:
    """Return live pricing for providers that support it (openrouter, nous, ai-gateway, novita)."""
    from hermes_cli.models import _resolve_nous_pricing_credentials, fetch_models_with_pricing, normalize_provider
    normalized = normalize_provider(provider)
    if normalized == "openrouter":
        return fetch_models_with_pricing(
            api_key=_resolve_openrouter_api_key(),
            base_url="https://openrouter.ai/api",
            force_refresh=force_refresh,
        )
    if normalized == "ai-gateway":
        return fetch_ai_gateway_pricing(force_refresh=force_refresh)
    if normalized == "novita":
        return _fetch_novita_pricing(force_refresh=force_refresh)
    if normalized == "deepinfra":
        return _fetch_deepinfra_pricing(force_refresh=force_refresh)
    if normalized == "fireworks":
        return _fireworks_pricing_from_models_dev(force_refresh=force_refresh)
    if normalized == "nous":
        api_key, base_url = _resolve_nous_pricing_credentials()
        if base_url:
            return fetch_models_with_pricing(
                api_key=api_key,
                base_url=base_url,
                force_refresh=force_refresh,
                # Sale chrome (pricing.original) is Nous Portal-only.
                include_sale_original=True,
                cache_ttl_seconds=_NOUS_CATALOG_TTL_SECONDS,
            )
    return {}


def _fireworks_pricing_from_models_dev(
    *,
    force_refresh: bool = False,
) -> dict[str, dict[str, str]]:
    """Derive Fireworks picker pricing from the models.dev registry cache.

    No dedicated network fetch: ``fetch_models_dev()`` already maintains an in-memory + disk cache
    (1h TTL) that every picker surface shares, so this is a pure dict transform on the picker path —
    no added latency and no per-render network call.
    """
    cache_key = "models.dev/fireworks"
    if not force_refresh:
        cached = _cached_catalog(cache_key)
        if cached is not None:
            return cached

    result: dict[str, dict[str, str]] = {}
    try:
        from agent.models_dev import _get_provider_models

        models = _get_provider_models("fireworks") or {}
        for mid, entry in models.items():
            if not isinstance(entry, dict):
                continue
            cost = entry.get("cost")
            if not isinstance(cost, dict):
                continue
            inp = cost.get("input")
            out = cost.get("output")
            if inp is None and out is None:
                continue
            row: dict[str, str] = {
                "prompt": str(float(inp or 0) / 1_000_000),
                "completion": str(float(out or 0) / 1_000_000),
            }
            cache_read = cost.get("cache_read")
            if cache_read:
                row["input_cache_read"] = str(float(cache_read) / 1_000_000)
            result[str(mid)] = row
    except Exception:
        result = {}

    return _cache_catalog(cache_key, result)


def _fetch_novita_pricing(
    timeout: float = 8.0,
    *,
    force_refresh: bool = False,
) -> dict[str, dict[str, str]]:
    """Fetch pricing from NovitaAI /v1/models.

    NovitaAI reports per-million-token prices in units of 0.0001 USD; they are converted to the
    per-token strings the shared pricing formatter expects. Results are cached in
    ``_pricing_cache`` keyed on the resolved base URL so menu renders don't re-hit the network.
    """
    from hermes_cli.models import _HERMES_USER_AGENT, _urlopen_model_catalog_request
    api_key = os.getenv("NOVITA_API_KEY", "").strip()
    if not api_key:
        return {}

    base_url = os.getenv("NOVITA_BASE_URL", "").strip() or "https://api.novita.ai/openai/v1"
    cache_key = base_url.rstrip("/")
    if not force_refresh:
        cached = _cached_catalog(cache_key)
        if cached is not None:
            return cached

    url = cache_key + "/models"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
        "User-Agent": _HERMES_USER_AGENT,
    }

    try:
        req = urllib.request.Request(url, headers=headers)
        with _urlopen_model_catalog_request(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode())
    except Exception:
        return _cache_catalog(cache_key, {})

    result: dict[str, dict[str, str]] = {}
    for item in payload.get("data", []):
        if not isinstance(item, dict):
            continue
        mid = item.get("id")
        if not mid:
            continue
        inp = item.get("input_token_price_per_m")
        out = item.get("output_token_price_per_m")
        if inp is None and out is None:
            continue
        result[str(mid)] = {
            "prompt": str(float(inp or 0) / 10_000 / 1_000_000),
            "completion": str(float(out or 0) / 10_000 / 1_000_000),
        }

    return _cache_catalog(cache_key, result)


def _fetch_deepinfra_pricing(
    timeout: float = 5.0,
    *,
    force_refresh: bool = False,
) -> dict[str, dict[str, str]]:
    """Return picker-shape pricing for DeepInfra chat models.

    DeepInfra publishes ``input_tokens``/``output_tokens``/``cache_read_tokens`` in $/MTok; the
    picker expects per-token strings under ``prompt``/``completion``/``input_cache_read``
    (OpenRouter shape). Cached via the catalog helper so repeated picker renders are free.
    """
    from hermes_cli.models import _fetch_deepinfra_models_by_tag
    items = _fetch_deepinfra_models_by_tag(
        "chat", timeout=timeout, force_refresh=force_refresh
    )
    if not items:
        return {}

    result: dict[str, dict[str, str]] = {}
    for item in items:
        metadata = item.get("metadata") or {}
        pricing = metadata.get("pricing") if isinstance(metadata, dict) else None
        if not isinstance(pricing, dict):
            continue
        entry: dict[str, str] = {}
        inp = pricing.get("input_tokens")
        out = pricing.get("output_tokens")
        cache_read = pricing.get("cache_read_tokens")
        if inp is not None:
            entry["prompt"] = str(float(inp) / 1_000_000)
        if out is not None:
            entry["completion"] = str(float(out) / 1_000_000)
        if cache_read is not None:
            entry["input_cache_read"] = str(float(cache_read) / 1_000_000)
        if entry:
            result[item["id"]] = entry
    return result
