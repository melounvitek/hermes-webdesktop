"""Per-provider model name normalization."""

from __future__ import annotations

import re
from typing import Optional

# ---------------------------------------------------------------------------
# Vendor prefix mapping
# ---------------------------------------------------------------------------
# Maps the first hyphen-delimited token of a bare model name to the vendor
# slug used by aggregator APIs (OpenRouter, Nous, etc.).
#
# Example: "claude-sonnet-4.6" -> first token "claude" -> vendor "anthropic"
#          -> aggregator slug: "anthropic/claude-sonnet-4.6"

_VENDOR_PREFIXES: dict[str, str] = {
    "claude": "anthropic",
    "gpt": "openai",
    "o1": "openai",
    "o3": "openai",
    "o4": "openai",
    "gemini": "google",
    "gemma": "google",
    "deepseek": "deepseek",
    "glm": "z-ai",
    "kimi": "moonshotai",
    "minimax": "minimax",
    "grok": "x-ai",
    "qwen": "qwen",
    "mimo": "xiaomi",
    "trinity": "arcee-ai",
    "nemotron": "nvidia",
    "llama": "meta-llama",
    "step": "stepfun",
    "trinity": "arcee-ai",
}

# Providers whose APIs consume vendor/model slugs.
_AGGREGATOR_PROVIDERS: frozenset[str] = frozenset({
    "openrouter",
    "nous",
    "ai-gateway",
    "kilocode",
})

# Providers that want bare names with dots replaced by hyphens.
_DOT_TO_HYPHEN_PROVIDERS: frozenset[str] = frozenset({
    "anthropic",
})

# Providers that want bare names with dots preserved.
_STRIP_VENDOR_ONLY_PROVIDERS: frozenset[str] = frozenset({
    "copilot",
    "copilot-acp",
    "openai-codex",
})

# Providers whose native naming is authoritative -- pass through unchanged.
_AUTHORITATIVE_NATIVE_PROVIDERS: frozenset[str] = frozenset({
    "huggingface",
})

# Direct providers that accept bare native names but should repair a matching
# provider/ prefix when users copy the aggregator form into config.yaml.
_MATCHING_PREFIX_STRIP_PROVIDERS: frozenset[str] = frozenset({
    "zai",
    "kimi-coding",
    "kimi-coding-cn",
    "minimax",
    "minimax-oauth",
    "minimax-cn",
    "alibaba",
    "qwen-oauth",
    "xiaomi",
    "arcee",
    "ollama-cloud",
    "nebius-token-factory",
    "custom",
    "gemini",
    "xai",
})

# Providers whose API serves ``vendor/model`` ids but whose endpoint can also
# front arbitrary self-hosted models, so a bare name cannot be prefixed
# blindly. A bare id is repaired only when the curated catalogue for that
# provider holds exactly one entry ending in ``/<name>`` — a lookup, not a
# guess. NVIDIA NIM is the case in hand: build.nvidia.com serves
# ``nvidia/nemotron-…`` (and third-party ``z-ai/glm-…``), while the same
# provider id also points at local NIM containers with their own naming.
# Without this repair a bare ``nemotron-3-ultra-550b-a55b`` reaches the API
# and returns a bare ``404 page not found`` that never names the model (#78796).
_CATALOGUE_PREFIX_REPAIR_PROVIDERS: frozenset[str] = frozenset({
    "nvidia",
})

# Providers whose APIs require lowercase model IDs.  Xiaomi's
# ``api.xiaomimimo.com`` rejects mixed-case names like ``MiMo-V2.5-Pro``
# that users might copy from marketing docs — it only accepts
# ``mimo-v2.5-pro``.  After stripping a matching provider prefix, these
# providers also get ``.lower()`` applied.
_LOWERCASE_MODEL_PROVIDERS: frozenset[str] = frozenset({
    "xiaomi",
})

# ---------------------------------------------------------------------------
# DeepSeek special handling
# ---------------------------------------------------------------------------
# DeepSeek's direct API only accepts first-class V-series IDs after the
# 2026-07-24 cut-off.  Legacy aliases and fuzzy names are remapped here so
# saved configs / picker leftovers cannot keep sending retired IDs.

_DEEPSEEK_REASONER_KEYWORDS: frozenset[str] = frozenset({
    "reasoner",
    "r1",
    "think",
    "reasoning",
    "cot",
})

# Retired on 2026-07-24 15:59 UTC. Official docs: both aliases mapped to
# deepseek-v4-flash (chat = non-thinking, reasoner = thinking). Thinking
# mode itself is controlled by extra_body.thinking on the DeepSeek profile.
_DEEPSEEK_RETIRED_ALIASES: frozenset[str] = frozenset({
    "deepseek-chat",
    "deepseek-reasoner",
})

_DEEPSEEK_CANONICAL_MODELS: frozenset[str] = frozenset({
    "deepseek-v4-pro",     # V4 Pro — first-class model ID
    "deepseek-v4-flash",   # V4 Flash — first-class model ID
})

# First-class V-series IDs (``deepseek-v4-pro``, ``deepseek-v4-flash``,
# future ``deepseek-v5-*``, dated variants like ``deepseek-v4-flash-20260423``).
# Verified empirically 2026-04-24: DeepSeek's Chat Completions API returns
# ``provider: DeepSeek`` / ``model: deepseek-v4-flash-20260423`` when called
# with ``model=deepseek/deepseek-v4-flash``, so these names are not aliases
# of ``deepseek-chat`` and must not be folded into it.
_DEEPSEEK_V_SERIES_RE = re.compile(r"^deepseek-v\d+([-.].+)?$")


def _normalize_for_deepseek(model_name: str) -> str:
    """Map a model input to a DeepSeek-accepted identifier.

    Retired aliases ``deepseek-chat``/``deepseek-reasoner`` and known canonicals map as expected;
    anything matching ``deepseek-v<digit>...`` passes through so future V-series ids work without a
    release; reasoner keywords and everything else fall back to ``deepseek-v4-flash``.
    """
    bare = _strip_vendor_prefix(model_name).lower()

    # Retired aliases must rewrite — DeepSeek returns HTTP 400 after the
    # 2026-07-24 cut-off if these IDs are sent on the wire.
    if bare in _DEEPSEEK_RETIRED_ALIASES:
        return "deepseek-v4-flash"

    if bare in _DEEPSEEK_CANONICAL_MODELS:
        return bare

    # V-series first-class IDs (v4-pro, v4-flash, future v5-*, dated variants)
    if _DEEPSEEK_V_SERIES_RE.match(bare):
        return bare

    # Check for reasoner-like keywords anywhere in the name
    for keyword in _DEEPSEEK_REASONER_KEYWORDS:
        if keyword in bare:
            return "deepseek-v4-flash"

    return "deepseek-v4-flash"


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _strip_vendor_prefix(model_name: str) -> str:
    """Remove a ``vendor/`` prefix if present."""
    if "/" in model_name:
        return model_name.split("/", 1)[1]
    return model_name


def _dots_to_hyphens(model_name: str) -> str:
    """Replace dots with hyphens in a model name."""
    return model_name.replace(".", "-")


def _normalize_provider_alias(provider_name: str) -> str:
    """Resolve provider aliases to Hermes' canonical ids."""
    raw = (provider_name or "").strip().lower()
    if not raw:
        return raw
    try:
        from hermes_cli.models import normalize_provider

        return normalize_provider(raw)
    except Exception:
        return raw


def _strip_matching_provider_prefix(model_name: str, target_provider: str) -> str:
    """Strip ``provider/`` only when the prefix matches the target provider.

    Prevents arbitrary slash-bearing ids from being mangled on native providers while still
    repairing config values like ``zai/glm-5.1`` for ``zai``. ``custom`` is a bucket, not a vendor:
    an alias that merely resolves to it (e.g. ``ollama``) may be a real routing prefix required by a
    proxy such as LiteLLM, so only a literal ``custom/`` prefix is treated as redundant.
    """
    if "/" not in model_name:
        return model_name

    prefix, remainder = model_name.split("/", 1)
    if not prefix.strip() or not remainder.strip():
        return model_name

    normalized_target = _normalize_provider_alias(target_provider)
    if normalized_target == "custom":
        if prefix.strip().lower() == "custom":
            return remainder.strip()
        return model_name

    normalized_prefix = _normalize_provider_alias(prefix)
    if normalized_prefix and normalized_prefix == normalized_target:
        return remainder.strip()
    return model_name


def detect_vendor(model_name: str) -> Optional[str]:
    """Detect the vendor slug from a bare model name."""
    name = model_name.strip()
    if not name:
        return None

    # If there's already a vendor/ prefix, extract it
    if "/" in name:
        return name.split("/", 1)[0].lower() or None

    name_lower = name.lower()

    # Try first hyphen-delimited token (exact match)
    first_token = name_lower.split("-")[0]
    if first_token in _VENDOR_PREFIXES:
        return _VENDOR_PREFIXES[first_token]

    # Handle patterns where the first token includes version digits,
    # e.g. "qwen3.5-plus" -> first token "qwen3.5", but prefix is "qwen"
    for prefix, vendor in _VENDOR_PREFIXES.items():
        if name_lower.startswith(prefix):
            return vendor

    return None


def _prepend_vendor(model_name: str) -> str:
    """Prepend the detected ``vendor/`` prefix if missing.

    For aggregators that require ``vendor/model``. Names already containing ``/`` or with no
    detectable vendor are returned unchanged (the aggregator may still accept them).
    """
    if "/" in model_name:
        return model_name

    vendor = detect_vendor(model_name)
    if vendor:
        return f"{vendor}/{model_name}"
    return model_name


def _repair_prefix_from_catalogue(model_name: str, provider: str) -> str:
    """Restore a dropped ``vendor/`` prefix using the provider's catalogue.

    Unlike :func:`_prepend_vendor`, this never guesses from the model's name shape — it only repairs
    a bare id that matches **exactly one** curated entry for this provider modulo the prefix.
    """
    if "/" in model_name:
        return model_name
    try:
        from hermes_cli.models import _PROVIDER_MODELS
    except Exception:
        return model_name

    catalogue = _PROVIDER_MODELS.get(provider) or []
    # Compare against the catalogue's own suffix, tag included: a bare
    # ``…:free`` id must resolve to the ``:free`` entry, not its paid sibling.
    needle = model_name.strip().lower()
    matches = {
        entry
        for entry in catalogue
        if "/" in entry and entry.split("/", 1)[1].strip().lower() == needle
    }
    if len(matches) == 1:
        return matches.pop()
    return model_name


def suggest_prefixed_model_id(provider: str, model_name: str) -> Optional[str]:
    """Return the prefixed catalogue id for a bare *model_name*, if unambiguous.

    Diagnostic counterpart to :func:`_repair_prefix_from_catalogue`, used to explain a provider's
    content-free 404 when the configured id lost its ``vendor/`` prefix. Returns ``None`` when the
    name already has a prefix, the provider has no catalogue, or nothing matches — so callers stay
    silent rather than guess.
    """
    name = (model_name or "").strip()
    if not name or "/" in name:
        return None
    try:
        canonical = _normalize_provider_alias(provider)
    except Exception:
        return None
    repaired = _repair_prefix_from_catalogue(name, canonical)
    return repaired if repaired != name else None


# ---------------------------------------------------------------------------
# Main normalisation entry point
# ---------------------------------------------------------------------------

def normalize_model_for_provider(model_input: str, target_provider: str) -> str:
    """Translate a model name into the format the target provider's API expects.

    Primary entry point for model-name normalisation. Accepts bare, vendor-prefixed or native ids;
    ``target_provider`` should already be normalised via ``normalize_provider()``. Never raises —
    always returns a best-effort string.
    """
    name = (model_input or "").strip()
    if not name:
        return name

    provider = _normalize_provider_alias(target_provider)

    # --- Aggregators: need vendor/model format ---
    if provider in _AGGREGATOR_PROVIDERS:
        return _prepend_vendor(name)

    # --- OpenCode Zen / OpenCode Go: flat-namespace resellers.
    #     Their /v1/models API returns bare IDs only (no vendor prefix), and
    #     the inference endpoint rejects vendor-prefixed names with HTTP 401
    #     "Model not supported".  Strip ANY leading ``vendor/`` so config
    #     entries like ``minimax/minimax-m2.7`` or ``deepseek/deepseek-v4-flash``
    #     — commonly copied from aggregator slugs into fallback_model lists —
    #     resolve to bare ``minimax-m2.7`` / ``deepseek-v4-flash`` the API
    #     actually serves.  See PR reviewing opencode-go fallback 401s. ---
    from hermes_cli.models import opencode_provider_family

    _oc_family = opencode_provider_family(provider)
    if _oc_family is not None:
        if "/" in name:
            _, bare_after_slash = name.split("/", 1)
            name = bare_after_slash.strip() or name
        if _oc_family == "opencode-zen" and name.lower().startswith("claude-"):
            return _dots_to_hyphens(name)
        return name

    # --- Anthropic: strip matching provider prefix, dots -> hyphens ---
    if provider in _DOT_TO_HYPHEN_PROVIDERS:
        bare = _strip_matching_provider_prefix(name, provider)
        if "/" in bare:
            return bare
        return _dots_to_hyphens(bare)

    # --- Copilot / Copilot ACP: delegate to the Copilot-specific
    #     normalizer.  It knows about the alias table (vendor-prefix
    #     stripping for Anthropic/OpenAI, dash-to-dot repair for Claude)
    #     and live-catalog lookups.  Without this, vendor-prefixed or
    #     dash-notation Claude IDs survive to the Copilot API and hit
    #     HTTP 400 "model_not_supported".  See issue #6879.
    if provider in {"copilot", "copilot-acp"}:
        try:
            from hermes_cli.models import normalize_copilot_model_id

            normalized = normalize_copilot_model_id(name)
            if normalized:
                return normalized
        except Exception:
            # Fall through to the generic strip-vendor behaviour below
            # if the Copilot-specific path is unavailable for any reason.
            pass

    # --- Copilot / Copilot ACP / openai-codex fallback:
    #     strip matching provider prefix, keep dots ---
    if provider in _STRIP_VENDOR_ONLY_PROVIDERS:
        stripped = _strip_matching_provider_prefix(name, provider)
        if stripped == name and name.startswith("openai/"):
            # openai-codex maps openai/gpt-5.4 -> gpt-5.4
            return name.split("/", 1)[1]
        return stripped

    # --- DeepSeek: map to one of two canonical names ---
    if provider == "deepseek":
        bare = _strip_matching_provider_prefix(name, provider)
        if "/" in bare:
            return bare
        return _normalize_for_deepseek(bare)

    # --- Direct providers: repair matching provider prefixes only ---
    if provider in _MATCHING_PREFIX_STRIP_PROVIDERS:
        result = _strip_matching_provider_prefix(name, provider)
        # Some providers require lowercase model IDs (e.g. Xiaomi's API
        # rejects "MiMo-V2.5-Pro" but accepts "mimo-v2.5-pro").
        if provider in _LOWERCASE_MODEL_PROVIDERS:
            result = result.lower()
        return result

    # --- Catalogue-backed prefix repair: restore a dropped ``vendor/`` on a
    #     bare id that matches exactly one curated entry.  Unknown names (a
    #     local NIM container, a proxied model) pass through untouched. ---
    if provider in _CATALOGUE_PREFIX_REPAIR_PROVIDERS:
        return _repair_prefix_from_catalogue(name, provider)

    # --- Authoritative native providers: preserve user-facing slugs as-is ---
    if provider in _AUTHORITATIVE_NATIVE_PROVIDERS:
        return name

    # --- Custom & all others: pass through as-is ---
    return name


# ---------------------------------------------------------------------------
# Batch / convenience helpers
# ---------------------------------------------------------------------------

