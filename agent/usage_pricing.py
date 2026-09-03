from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, Literal, Optional

from agent.model_metadata import fetch_endpoint_model_metadata, fetch_model_metadata
from utils import base_url_host_matches, base_url_hostname

logger = logging.getLogger(__name__)

_ZERO = Decimal("0")
_ONE_MILLION = Decimal("1000000")
_NOUS_DEFAULT_BASE_URL = "https://inference-api.nousresearch.com/v1"

# Below $0.01, render at 4 dp so cheap-model costs never display as $0.00.
_SUBCENT_THRESHOLD = Decimal("0.01")

# Attached to every CostResult with status="included" so consumers can
# distinguish "free because subscription" from "free because $0 pricing".
_INCLUDED_NOTE = "subscription-included; no provider invoice for usage"


def format_cost_label(amount: Decimal) -> str:
    """Format a cost as a display label, scaling precision to magnitude.

    Zero → "$0.00"; sub-cent → "~$0.0046" (4 dp, or "~$<0.0001" when the
    amount rounds to 0.0000 so the label never reads as zero); else "~$1.23".
    Shared by per-response cost labels and the insights cost-bucket
    formatters so sub-cent honesty cannot regress on one surface.
    """
    if amount == _ZERO:
        return "$0.00"
    if amount < _SUBCENT_THRESHOLD:
        label = f"~${amount:.4f}"
        # Compare the rendered label: a naive `< 0.00005` threshold misses
        # the exact boundary under ROUND_HALF_EVEN.
        return label if label != "~$0.0000" else "~$<0.0001"
    return f"~${amount:.2f}"

CostStatus = Literal["actual", "estimated", "included", "unknown"]
CostSource = Literal[
    "provider_cost_api",
    "provider_generation_api",
    "provider_models_api",
    "official_docs_snapshot",
    "user_override",
    "custom_contract",
    "none",
]


@dataclass(frozen=True)
class CanonicalUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0
    request_count: int = 1
    raw_usage: Optional[dict[str, Any]] = None

    @property
    def prompt_tokens(self) -> int:
        return self.input_tokens + self.cache_read_tokens + self.cache_write_tokens

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.output_tokens

    def __add__(self, other: "CanonicalUsage") -> "CanonicalUsage":
        """Sum two usage buckets. ``raw_usage`` (single-response detail) is
        dropped; ``request_count`` adds so callers see how many API calls a
        combined figure covers."""
        if not isinstance(other, CanonicalUsage):
            return NotImplemented
        return CanonicalUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
            request_count=self.request_count + other.request_count, raw_usage=None,
        )


@dataclass(frozen=True)
class BillingRoute:
    provider: str
    model: str
    base_url: str = ""
    billing_mode: str = "unknown"


@dataclass(frozen=True)
class PricingEntry:
    input_cost_per_million: Optional[Decimal] = None
    output_cost_per_million: Optional[Decimal] = None
    cache_read_cost_per_million: Optional[Decimal] = None
    cache_write_cost_per_million: Optional[Decimal] = None
    request_cost: Optional[Decimal] = None
    source: CostSource = "none"
    source_url: Optional[str] = None
    pricing_version: Optional[str] = None
    fetched_at: Optional[datetime] = None
    # Context-tiered pricing (e.g. Gemini Pro above 200k prompt tokens): when
    # ``usage.prompt_tokens`` exceeds ``tier_threshold_tokens`` the ``*_above``
    # rates replace the base rates for the WHOLE request (Google's semantics,
    # not marginal brackets). A None ``*_above`` falls back to its base rate.
    tier_threshold_tokens: Optional[int] = None
    input_cost_per_million_above: Optional[Decimal] = None
    output_cost_per_million_above: Optional[Decimal] = None
    cache_read_cost_per_million_above: Optional[Decimal] = None


@dataclass(frozen=True)
class CostResult:
    amount_usd: Optional[Decimal]
    status: CostStatus
    source: CostSource
    label: str
    fetched_at: Optional[datetime] = None
    pricing_version: Optional[str] = None
    notes: tuple[str, ...] = ()


_UTC_NOW = lambda: datetime.now(timezone.utc)


def _snap(
    inp: str, out: str, cache_read: Optional[str] = None, cache_write: Optional[str] = None, *,
    version: str, url: Optional[str] = None, **tiers: Any,
) -> PricingEntry:
    """Build an official-docs snapshot entry from per-million USD rate strings."""
    return PricingEntry(
        input_cost_per_million=Decimal(inp), output_cost_per_million=Decimal(out),
        cache_read_cost_per_million=Decimal(cache_read) if cache_read is not None else None,
        cache_write_cost_per_million=Decimal(cache_write) if cache_write is not None else None,
        source="official_docs_snapshot", source_url=url, pricing_version=version, **tiers,
    )


# (source_url, pricing_version) shared by the entries of one snapshot.
_OPENAI_56 = dict(url="https://openai.com/index/previewing-gpt-5-6-sol/", version="openai-gpt-5.6-2026-07")
_ANTHROPIC = dict(url="https://platform.claude.com/docs/en/about-claude/pricing", version="anthropic-pricing-2026-05")
_OPENAI = dict(url="https://openai.com/api/pricing/", version="openai-pricing-2026-03-16")
_DEEPSEEK = dict(url="https://api-docs.deepseek.com/quick_start/pricing", version="deepseek-pricing-2026-07")
_GOOGLE = dict(url="https://ai.google.dev/pricing", version="google-pricing-2026-07-07")
_GOOGLE_NEW = dict(url="https://ai.google.dev/gemini-api/docs/pricing", version="google-pricing-2026-07-28")
_BEDROCK_URL = "https://aws.amazon.com/bedrock/pricing/"
_BEDROCK_ANTHROPIC = dict(url=_BEDROCK_URL, version="anthropic-list-2026-07")
_BEDROCK = dict(url=_BEDROCK_URL, version="bedrock-pricing-2026-04")
_FIREWORKS = dict(url="https://docs.fireworks.ai/serverless/pricing", version="fireworks-pricing-2026-07")

# Official docs snapshot: models whose published pricing and cache semantics
# are stable enough to encode exactly. Positional rates are per 1M tokens:
# (input, output[, cache_read[, cache_write]]).
_OFFICIAL_DOCS_PRICING: Dict[tuple[str, str], PricingEntry] = {
    # OpenAI GPT-5.6 (Sol/Terra/Luna). Cache write = 1.25x input, cache read =
    # 0.10x input. "-pro" high-effort modes bill at the same per-token rates
    # (aliased below the dict); "Sol Fast mode" is a separate tier, not covered.
    ("openai", "gpt-5.6-sol"): _snap("5.00", "30.00", "0.50", "6.25", **_OPENAI_56),
    ("openai", "gpt-5.6-terra"): _snap("2.50", "15.00", "0.25", "3.125", **_OPENAI_56),
    ("openai", "gpt-5.6-luna"): _snap("1.00", "6.00", "0.10", "1.25", **_OPENAI_56),
    # Anthropic Claude 4.8; fast mode is a separate model id at a 2x premium.
    ("anthropic", "claude-opus-4-8"): _snap("5.00", "25.00", "0.50", "6.25", **_ANTHROPIC),
    ("anthropic", "claude-opus-4-8-fast"): _snap(
        "10.00", "50.00", "1.00", "12.50",
        url="https://openrouter.ai/anthropic/claude-opus-4.8-fast", version="anthropic-pricing-2026-05",
    ),
    # Claude Sonnet 5: introductory $2/$10 through 2026-08-31, then $3/$15
    # (matching Sonnet 4.6). Update this entry when the intro window closes.
    ("anthropic", "claude-sonnet-5"): _snap(
        "2.00", "10.00", "0.20", "2.50", url=_ANTHROPIC["url"], version="anthropic-pricing-2026-06-intro"
    ),
    # Claude 4.5/4.6/4.7 Opus share $5/$25 (new tokenizer, up to 35% more tokens).
    ("anthropic", "claude-opus-4-7"): _snap("5.00", "25.00", "0.50", "6.25", **_ANTHROPIC),
    ("anthropic", "claude-opus-4-7-20250507"): _snap("5.00", "25.00", "0.50", "6.25", **_ANTHROPIC),
    ("anthropic", "claude-opus-4-6"): _snap("5.00", "25.00", "0.50", "6.25", **_ANTHROPIC),
    ("anthropic", "claude-opus-4-6-20250414"): _snap("5.00", "25.00", "0.50", "6.25", **_ANTHROPIC),
    ("anthropic", "claude-sonnet-4-6"): _snap("3.00", "15.00", "0.30", "3.75", **_ANTHROPIC),
    ("anthropic", "claude-sonnet-4-6-20250414"): _snap("3.00", "15.00", "0.30", "3.75", **_ANTHROPIC),
    ("anthropic", "claude-opus-4-5"): _snap("5.00", "25.00", "0.50", "6.25", **_ANTHROPIC),
    ("anthropic", "claude-sonnet-4-5"): _snap("3.00", "15.00", "0.30", "3.75", **_ANTHROPIC),
    ("anthropic", "claude-haiku-4-5"): _snap("1.00", "5.00", "0.10", "1.25", **_ANTHROPIC),
    ("anthropic", "claude-opus-4-20250514"): _snap("15.00", "75.00", "1.50", "18.75", **_ANTHROPIC),
    ("anthropic", "claude-sonnet-4-20250514"): _snap("3.00", "15.00", "0.30", "3.75", **_ANTHROPIC),
    # OpenAI
    ("openai", "gpt-4o"): _snap("2.50", "10.00", "1.25", **_OPENAI),
    ("openai", "gpt-4o-mini"): _snap("0.15", "0.60", "0.075", **_OPENAI),
    ("openai", "gpt-4.1"): _snap("2.00", "8.00", "0.50", **_OPENAI),
    ("openai", "gpt-4.1-mini"): _snap("0.40", "1.60", "0.10", **_OPENAI),
    ("openai", "gpt-4.1-nano"): _snap("0.10", "0.40", "0.025", **_OPENAI),
    ("openai", "o3"): _snap("10.00", "40.00", "2.50", **_OPENAI),
    ("openai", "o3-mini"): _snap("1.10", "4.40", "0.55", **_OPENAI),
    # Anthropic pre-4.5 generation
    ("anthropic", "claude-3-5-sonnet-20241022"): _snap("3.00", "15.00", "0.30", "3.75", **_ANTHROPIC),
    ("anthropic", "claude-3-5-haiku-20241022"): _snap("0.80", "4.00", "0.08", "1.00", **_ANTHROPIC),
    ("anthropic", "claude-3-opus-20240229"): _snap("15.00", "75.00", "1.50", "18.75", **_ANTHROPIC),
    ("anthropic", "claude-3-haiku-20240307"): _snap("0.25", "1.25", "0.03", "0.30", **_ANTHROPIC),
    # DeepSeek. deepseek-chat / deepseek-reasoner are deprecated aliases of
    # deepseek-v4-flash's non-thinking / thinking modes — same rates.
    ("deepseek", "deepseek-chat"): _snap("0.14", "0.28", "0.0028", **_DEEPSEEK),
    ("deepseek", "deepseek-reasoner"): _snap("0.14", "0.28", "0.0028", **_DEEPSEEK),
    ("deepseek", "deepseek-v4-pro"): _snap("0.435", "0.87", "0.003625", **_DEEPSEEK),
    ("deepseek", "deepseek-v4-flash"): _snap("0.14", "0.28", "0.0028", **_DEEPSEEK),
    # Google Gemini
    ("google", "gemini-3.6-flash"): _snap("1.50", "7.50", "0.15", **_GOOGLE_NEW),
    ("google", "gemini-3.5-flash"): _snap("1.50", "9.00", "0.15", **_GOOGLE),
    ("google", "gemini-3.5-flash-lite"): _snap("0.30", "2.50", "0.03", **_GOOGLE_NEW),
    ("google", "gemini-3.1-pro"): _snap(
        "2.00", "12.00", "0.20",
        tier_threshold_tokens=200_000,
        input_cost_per_million_above=Decimal("4.00"),
        output_cost_per_million_above=Decimal("18.00"),
        cache_read_cost_per_million_above=Decimal("0.40"),
        **_GOOGLE,
    ),
    ("google", "gemini-3.1-flash-lite"): _snap("0.25", "1.50", "0.025", **_GOOGLE),
    ("google", "gemini-3-pro-preview"): _snap("2.00", "12.00", "0.20", **_GOOGLE),
    ("google", "gemini-3-flash-preview"): _snap("0.50", "3.00", "0.05", **_GOOGLE),
    ("google", "gemini-2.5-pro"): _snap(
        "1.25", "10.00", "0.125",
        tier_threshold_tokens=200_000,
        input_cost_per_million_above=Decimal("2.50"),
        output_cost_per_million_above=Decimal("15.00"),
        **_GOOGLE,
    ),
    ("google", "gemini-2.5-flash"): _snap("0.15", "0.60", "0.015", **_GOOGLE),
    ("google", "gemini-2.0-flash"): _snap("0.10", "0.40", "0.01", **_GOOGLE),
    # AWS Bedrock on-demand: same per-token rates as the model provider, billed
    # through AWS. Current-gen Claude rows are commercial-list snapshots (the AWS
    # Price List API had not published these SKUs machine-readably).
    ("bedrock", "anthropic.claude-opus-4-8"): _snap("5.00", "25.00", "0.50", "6.25", **_BEDROCK_ANTHROPIC),
    ("bedrock", "anthropic.claude-opus-4-7"): _snap("5.00", "25.00", "0.50", "6.25", **_BEDROCK_ANTHROPIC),
    ("bedrock", "anthropic.claude-opus-4-6"): _snap("5.00", "25.00", "0.50", "6.25", **_BEDROCK_ANTHROPIC),
    ("bedrock", "anthropic.claude-sonnet-5"): _snap(
        "3.00", "15.00", "0.30", "3.75", url=_BEDROCK_URL, version="bedrock-pricing-2026-06"
    ),
    ("bedrock", "anthropic.claude-sonnet-4-6"): _snap("3.00", "15.00", "0.30", "3.75", **_BEDROCK),
    ("bedrock", "anthropic.claude-sonnet-4-5"): _snap("3.00", "15.00", "0.30", "3.75", **_BEDROCK),
    ("bedrock", "anthropic.claude-haiku-4-5"): _snap("0.80", "4.00", "0.08", "1.00", **_BEDROCK),
    ("bedrock", "amazon.nova-pro"): _snap("0.80", "3.20", **_BEDROCK),
    ("bedrock", "amazon.nova-lite"): _snap("0.06", "0.24", **_BEDROCK),
    ("bedrock", "amazon.nova-micro"): _snap("0.035", "0.14", **_BEDROCK),
    # MiniMax
    ("minimax", "minimax-m2.7"): _snap("0.30", "1.20", version="minimax-pricing-2026-04"),
    ("minimax-cn", "minimax-m2.7"): _snap("0.30", "1.20", version="minimax-pricing-2026-04"),
    # Fireworks AI serverless (Standard tier). Fireworks publishes a per-model
    # cached_input rate (→ cache_read) but no separate cache_write rate.
    ("fireworks", "kimi-k2p6"): _snap("0.95", "4.00", "0.16", **_FIREWORKS),
    ("fireworks", "kimi-k2p7-code"): _snap("0.95", "4.00", "0.19", **_FIREWORKS),
    ("fireworks", "glm-5p2"): _snap("1.40", "4.40", "0.14", **_FIREWORKS),
    ("fireworks", "deepseek-v4-pro"): _snap("1.74", "3.48", "0.145", **_FIREWORKS),
    ("fireworks", "deepseek-v4-flash"): _snap("0.14", "0.28", "0.028", **_FIREWORKS),
    ("fireworks", "qwen3p7-plus"): _snap("0.40", "1.60", "0.08", **_FIREWORKS),
    ("fireworks", "minimax-m3"): _snap("0.30", "1.20", "0.06", **_FIREWORKS),
    ("fireworks", "gpt-oss-120b"): _snap("0.15", "0.60", "0.015", **_FIREWORKS),
    ("fireworks", "gpt-oss-20b"): _snap("0.07", "0.30", "0.035", **_FIREWORKS),
    ("fireworks", "glm-5p1"): _snap("1.40", "4.40", "0.26", **_FIREWORKS),
    ("fireworks", "minimax-m2p7"): _snap("0.30", "1.20", "0.06", **_FIREWORKS),
    # Fast/turbo tiers are exposed as accounts/fireworks/routers/<name>, so
    # rsplit("/", 1) yields these distinct ids with their own (higher) rates.
    ("fireworks", "kimi-k2p6-fast"): _snap("2.00", "8.00", "0.30", **_FIREWORKS),
    ("fireworks", "kimi-k2p6-turbo"): _snap("2.00", "8.00", "0.30", **_FIREWORKS),
    ("fireworks", "kimi-k2p7-code-fast"): _snap("1.90", "8.00", "0.38", **_FIREWORKS),
    ("fireworks", "glm-5p2-fast"): _snap("2.10", "6.60", "0.21", **_FIREWORKS),
    ("fireworks", "glm-5p1-fast"): _snap("2.80", "8.80", "0.52", **_FIREWORKS),
}
del _OPENAI_56, _ANTHROPIC, _OPENAI, _DEEPSEEK, _GOOGLE, _GOOGLE_NEW
del _BEDROCK_URL, _BEDROCK_ANTHROPIC, _BEDROCK, _FIREWORKS

# GPT-5.6 "-pro" high-effort variants bill at the base tier's per-token rates
# (more tokens per task, not a higher rate); the Hermes-side "-900k" Codex
# picker variants are the same model with the suffix stripped on the wire.
# Alias both onto the base entries so the snapshot stays single-source.
for _base_56 in ("gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"):
    _OFFICIAL_DOCS_PRICING[("openai", f"{_base_56}-pro")] = _OFFICIAL_DOCS_PRICING[("openai", _base_56)]
    _OFFICIAL_DOCS_PRICING[("openai", f"{_base_56}-900k")] = _OFFICIAL_DOCS_PRICING[("openai", _base_56)]
del _base_56

# The direct Gemini provider emits preview IDs for these two models; key the
# snapshot by both the documented stable name and the emitted ID.
for _alias, _canonical in {
    "gemini-3.1-pro-preview": "gemini-3.1-pro",
    "gemini-3.1-flash-lite-preview": "gemini-3.1-flash-lite",
}.items():
    _OFFICIAL_DOCS_PRICING[("google", _alias)] = _OFFICIAL_DOCS_PRICING[("google", _canonical)]
del _alias, _canonical


def _to_decimal(value: Any) -> Optional[Decimal]:
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except Exception:
        return None


def _to_int(value: Any) -> int:
    try:
        return int(value or 0)
    except Exception:
        return 0


def _usage_get(obj: Any, name: str, default: Any = 0) -> Any:
    """Read a usage field from either a dict or an attribute object.

    The Responses API returns usage as a typed SDK object OR a plain dict;
    ``getattr`` on a dict silently yields the default and zeroes every count.
    """
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _usage_count(value: Any) -> int:
    """Coerce a usage counter to a non-negative int (providers occasionally
    emit negative counters; clamp so they cannot corrupt session accounting)."""
    return max(0, _to_int(value))


def _usage_field(obj: Any, name: str, *path: str) -> int:
    """``_usage_count`` of ``obj.name[.path...]``; 0 if any hop is falsy."""
    for hop in (name, *path):
        if not obj:
            return 0
        obj = _usage_get(obj, hop, 0)
    return _usage_count(obj)


def _first_nonzero(obj: Any, *paths: tuple[str, ...]) -> int:
    """First non-zero ``_usage_field`` across candidate paths, else 0."""
    for path in paths:
        value = _usage_field(obj, *path)
        if value:
            return value
    return 0


def resolve_billing_route(
    model_name: str, provider: Optional[str] = None, base_url: Optional[str] = None
) -> BillingRoute:
    provider_name = (provider or "").strip().lower()
    base = (base_url or "").strip().lower()
    model = (model_name or "").strip()
    if not provider_name and "/" in model:
        inferred_provider, bare_model = model.split("/", 1)
        if inferred_provider in {"anthropic", "openai", "google"}:
            provider_name = inferred_provider
            model = bare_model

    url = base_url or ""
    bare = model.split("/")[-1]

    def host(name: str) -> bool:
        return base_url_host_matches(url, name)

    if provider_name == "openai-codex":
        return BillingRoute(provider="openai-codex", model=model, base_url=url, billing_mode="subscription_included")
    if provider_name == "openrouter" or host("openrouter.ai"):
        return BillingRoute(provider="openrouter", model=model, base_url=url, billing_mode="official_models_api")
    if provider_name == "nous" or host("inference-api.nousresearch.com"):
        return BillingRoute(provider="nous", model=model, base_url=base_url or _NOUS_DEFAULT_BASE_URL, billing_mode="official_models_api")
    if provider_name == "anthropic":
        return BillingRoute(provider="anthropic", model=bare, base_url=url, billing_mode="official_docs_snapshot")
    # "openai-api" is the picker slug for direct api.openai.com; it bills as
    # bare "openai", whose keys the snapshot uses.
    if provider_name in {"openai", "openai-api"}:
        return BillingRoute(provider="openai", model=bare, base_url=url, billing_mode="official_docs_snapshot")
    if provider_name in {"minimax", "minimax-cn"}:
        return BillingRoute(provider=provider_name, model=bare, base_url=url, billing_mode="official_docs_snapshot")
    # AI Studio and Vertex host the same Gemini models; the snapshot is keyed on
    # provider='google', and the Vertex "google/" vendor prefix is stripped.
    if (
        provider_name in {"google", "gemini", "vertex", "google-gemini", "google-ai-studio", "google-vertex", "vertex-ai"}
        or host("aiplatform.googleapis.com")
        or host("generativelanguage.googleapis.com")
    ):
        return BillingRoute(provider="google", model=bare, base_url=url, billing_mode="official_docs_snapshot")
    if provider_name == "fireworks" or host("api.fireworks.ai"):
        # Fireworks ids look like accounts/fireworks/models/<name>; keys use <name>.
        return BillingRoute(provider="fireworks", model=model.rsplit("/", 1)[-1], base_url=url, billing_mode="official_docs_snapshot")
    if provider_name in {"custom", "local"} or (base and base_url_hostname(base) in ("localhost", "127.0.0.1")):
        return BillingRoute(provider=provider_name or "custom", model=model, base_url=url, billing_mode="unknown")
    return BillingRoute(provider=provider_name or "unknown", model=bare if model else "", base_url=url, billing_mode="unknown")


def _normalize_bedrock_model_name(model: str) -> str:
    """Normalize a Bedrock model id to its bare foundation-model form.

    Cross-region inference profiles prefix the id with a region scope
    (``us.``/``global.``/``apac.``/``au.``/...); the pricing table is keyed on
    the bare ``anthropic.claude-*`` id, so the prefix is stripped. Also maps
    dotted versions (``4.7`` → ``4-7``) and strips only the documented
    trailing date/revision/profile components (``-20250514-v1:0``).
    """
    name = model.lower().strip()
    for prefix in ("global.", "us.", "eu.", "apac.", "ap.", "au.", "jp.", "ca.", "sa.", "me.", "af."):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    name = re.sub(r"(\d+)\.(\d+)", r"\1-\2", name)
    name = re.sub(r":\d+$", "", name)
    name = re.sub(r"-v\d+$", "", name)
    name = re.sub(r"-\d{8}$", "", name)
    return name


def _normalize_anthropic_model_name(model: str) -> str:
    """Strip an ``anthropic/`` prefix and map dotted versions (4.7 → 4-7)."""
    name = model.lower().strip()
    if name.startswith("anthropic/"):
        name = name[len("anthropic/"):]
    return re.sub(r"(\d+)\.(\d+)", r"\1-\2", name)


def _lookup_official_docs_pricing(route: BillingRoute) -> Optional[PricingEntry]:
    model = route.model.lower()
    entry = _OFFICIAL_DOCS_PRICING.get((route.provider, model))
    if entry:
        return entry
    # Anthropic dot-notation (opus-4.7) and Bedrock region-prefixed ids need
    # normalizing before a second lookup.
    normalize = {
        "anthropic": _normalize_anthropic_model_name, "bedrock": _normalize_bedrock_model_name
    }.get(route.provider)
    if normalize:
        normalized = normalize(model)
        if normalized != model:
            entry = _OFFICIAL_DOCS_PRICING.get((route.provider, normalized))
            if entry:
                return entry
    return None


def _openrouter_pricing_entry(route: BillingRoute) -> Optional[PricingEntry]:
    return _pricing_entry_from_metadata(
        fetch_model_metadata(), route.model,
        source_url="https://openrouter.ai/docs/api/api-reference/models/get-models",
        pricing_version="openrouter-models-api",
    )


def _pricing_entry_from_metadata(
    metadata: Dict[str, Dict[str, Any]], model_id: str, *, source_url: str, pricing_version: str
) -> Optional[PricingEntry]:
    if model_id not in metadata:
        return None
    pricing = metadata[model_id].get("pricing") or {}
    prompt = _to_decimal(pricing.get("prompt"))
    completion = _to_decimal(pricing.get("completion"))
    request = _to_decimal(pricing.get("request"))
    cache_read = _to_decimal(
        pricing.get("cache_read") or pricing.get("cached_prompt") or pricing.get("input_cache_read")
    )
    cache_write = _to_decimal(
        pricing.get("cache_write")
        or pricing.get("cache_creation")
        or pricing.get("input_cache_write")
    )
    if prompt is None and completion is None and request is None:
        return None

    def _per_million(value: Optional[Decimal]) -> Optional[Decimal]:
        return None if value is None else value * _ONE_MILLION

    return PricingEntry(
        input_cost_per_million=_per_million(prompt),
        output_cost_per_million=_per_million(completion),
        cache_read_cost_per_million=_per_million(cache_read),
        cache_write_cost_per_million=_per_million(cache_write), request_cost=request,
        source="provider_models_api", source_url=source_url, pricing_version=pricing_version,
        fetched_at=_UTC_NOW(),
    )


def get_pricing_entry(
    model_name: str, provider: Optional[str] = None, base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Optional[PricingEntry]:
    route = resolve_billing_route(model_name, provider=provider, base_url=base_url)
    if route.billing_mode == "subscription_included":
        return PricingEntry(
            input_cost_per_million=_ZERO, output_cost_per_million=_ZERO,
            cache_read_cost_per_million=_ZERO, cache_write_cost_per_million=_ZERO, source="none",
            pricing_version="included-route",
        )
    if route.provider == "openrouter":
        return _openrouter_pricing_entry(route)
    if route.base_url:
        entry = _pricing_entry_from_metadata(
            fetch_endpoint_model_metadata(route.base_url, api_key=api_key or ""), route.model,
            source_url=f"{route.base_url.rstrip('/')}/models",
            pricing_version="openai-compatible-models-api",
        )
        if entry:
            return entry
    return _lookup_official_docs_pricing(route)


def normalize_usage(
    response_usage: Any, *, provider: Optional[str] = None, api_mode: Optional[str] = None
) -> CanonicalUsage:
    """Normalize raw API response usage into canonical token buckets.

    Three shapes: Anthropic (input/output/cache_read_input/cache_creation_input
    tokens), Codex Responses and OpenAI Chat Completions. In the latter two the
    input/prompt total INCLUDES cached tokens and the ``*_details`` object breaks
    them out, so input_tokens is derived by subtraction.
    """
    if not response_usage:
        return CanonicalUsage()

    provider_name = (provider or "").strip().lower()
    mode = (api_mode or "").strip().lower()
    u = response_usage

    if mode == "anthropic_messages" or provider_name == "anthropic":
        input_tokens = _usage_field(u, "input_tokens")
        output_tokens = _usage_field(u, "output_tokens")
        cache_read_tokens = _usage_field(u, "cache_read_input_tokens")
        cache_write_tokens = _usage_field(u, "cache_creation_input_tokens")
    elif mode == "codex_responses":
        input_total = _usage_field(u, "input_tokens")
        output_tokens = _usage_field(u, "output_tokens")
        cache_read_tokens = _usage_field(u, "input_tokens_details", "cached_tokens")
        # OpenAI's documented GPT-5.6+ field is `cache_write_tokens` (billed at
        # 1.25x); `cache_creation_tokens` is a fallback for older endpoints.
        cache_write_tokens = _first_nonzero(
            u, ("input_tokens_details", "cache_write_tokens"),
            ("input_tokens_details", "cache_creation_tokens"),
        )
        input_tokens = max(0, input_total - cache_read_tokens - cache_write_tokens)
    else:
        # OpenAI-style names first, then Anthropic-style: local OpenAI-compatible
        # servers (e.g. mlx_vlm.server) emit input_tokens/output_tokens and the
        # OpenAI client preserves them as extra attributes.
        prompt_total = _first_nonzero(u, ("prompt_tokens",), ("input_tokens",))
        output_tokens = _first_nonzero(u, ("completion_tokens",), ("output_tokens",))
        # Cache reads: nested OpenAI shape, then Anthropic-style top-level fields
        # exposed by proxies routing Claude (OpenRouter, Vercel AI Gateway, Cline),
        # then DeepSeek's top-level prompt_cache_hit_tokens, then Kimi/Moonshot's
        # top-level cached_tokens — without these, direct sessions show 0 hits
        # and bill hits at the full input rate.
        cache_read_tokens = _first_nonzero(
            u, ("prompt_tokens_details", "cached_tokens"), ("cache_read_input_tokens",),
            ("prompt_cache_hit_tokens",), ("cached_tokens",),
        )
        cache_write_tokens = _first_nonzero(
            u, ("prompt_tokens_details", "cache_write_tokens"),
            ("prompt_tokens_details", "cache_creation_input_tokens"),
            ("cache_creation_input_tokens",), ("cache_write_tokens",),
        )
        input_tokens = max(0, prompt_total - cache_read_tokens - cache_write_tokens)

    # Responses API: output_tokens_details.reasoning_tokens. Chat Completions
    # (OpenAI, OpenRouter, DeepSeek, ...): completion_tokens_details.reasoning_tokens.
    # Hidden thinking dominates output spend on reasoning models, so read both.
    reasoning_tokens = _first_nonzero(
        u, ("output_tokens_details", "reasoning_tokens"),
        ("completion_tokens_details", "reasoning_tokens"),
    )

    # On MiniMax-M3's Anthropic wire, cache_read_input_tokens carries a constant
    # +128 floor and cache_creation is always 0, so cache_read is not a reliable
    # hit signal; the input_tokens drop between consecutive calls is.
    # Docs: https://platform.minimax.io/docs/api-reference/text-prompt-caching
    if provider_name in {"minimax", "minimax-cn"} and mode == "anthropic_messages":
        logger.debug(
            "cache_observability provider=%s mode=%s input_tokens=%s "
            "output_tokens=%s cache_read_tokens=%s cache_write_tokens=%s "
            "(note: on MiniMax-M3 cache_read carries a +128 constant "
            "floor and is not a reliable hit signal — track input_tokens "
            "drops across calls instead)",
            provider_name, mode, input_tokens, output_tokens,
            cache_read_tokens, cache_write_tokens,
        )

    return CanonicalUsage(
        input_tokens=input_tokens, output_tokens=output_tokens, cache_read_tokens=cache_read_tokens,
        cache_write_tokens=cache_write_tokens, reasoning_tokens=reasoning_tokens,
    )


def _unknown_cost(source: CostSource, *notes: str) -> CostResult:
    return CostResult(amount_usd=None, status="unknown", source=source, label="n/a", notes=notes)


def estimate_usage_cost(
    model_name: str, usage: CanonicalUsage, *, provider: Optional[str] = None,
    base_url: Optional[str] = None, api_key: Optional[str] = None,
) -> CostResult:
    route = resolve_billing_route(model_name, provider=provider, base_url=base_url)
    if route.billing_mode == "subscription_included":
        return CostResult(
            amount_usd=_ZERO, status="included", source="none", label="included",
            pricing_version="included-route", notes=(_INCLUDED_NOTE,),
        )

    entry = get_pricing_entry(model_name, provider=provider, base_url=base_url, api_key=api_key)
    if not entry:
        return _unknown_cost("none")

    # Whole-request context tier (e.g. Gemini Pro >200k prompts): above the
    # threshold the *_above rates apply to the entire request; None falls back.
    input_rate = entry.input_cost_per_million
    output_rate = entry.output_cost_per_million
    cache_read_rate = entry.cache_read_cost_per_million
    cache_write_rate = entry.cache_write_cost_per_million
    if entry.tier_threshold_tokens is not None and usage.prompt_tokens > entry.tier_threshold_tokens:
        if entry.input_cost_per_million_above is not None:
            input_rate = entry.input_cost_per_million_above
        if entry.output_cost_per_million_above is not None:
            output_rate = entry.output_cost_per_million_above
        if entry.cache_read_cost_per_million_above is not None:
            cache_read_rate = entry.cache_read_cost_per_million_above

    if usage.input_tokens and input_rate is None:
        return _unknown_cost(entry.source)
    if usage.output_tokens and output_rate is None:
        return _unknown_cost(entry.source)
    if usage.cache_read_tokens and cache_read_rate is None:
        return _unknown_cost(entry.source, "cache-read pricing unavailable for route")
    if usage.cache_write_tokens and cache_write_rate is None:
        return _unknown_cost(entry.source, "cache-write pricing unavailable for route")

    amount = _ZERO
    for tokens, rate in (
        (usage.input_tokens, input_rate), (usage.output_tokens, output_rate),
        (usage.cache_read_tokens, cache_read_rate), (usage.cache_write_tokens, cache_write_rate),
    ):
        if rate is not None:
            amount += Decimal(tokens) * rate / _ONE_MILLION
    if entry.request_cost is not None and usage.request_count:
        amount += Decimal(usage.request_count) * entry.request_cost

    notes: list[str] = []
    status: CostStatus = "estimated"
    label = format_cost_label(amount)
    if entry.source == "none" and amount == _ZERO:
        status = "included"
        label = "included"
        notes.append(_INCLUDED_NOTE)

    if route.provider == "openrouter":
        notes.append("OpenRouter cost is estimated from the models API until reconciled.")

    return CostResult(
        amount_usd=amount, status=status, source=entry.source, label=label,
        fetched_at=entry.fetched_at, pricing_version=entry.pricing_version, notes=tuple(notes),
    )


def has_known_pricing(
    model_name: str, provider: Optional[str] = None, base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> bool:
    """True if pricing data exists for this model+route (direct lookup, no dummy usage)."""
    route = resolve_billing_route(model_name, provider=provider, base_url=base_url)
    if route.billing_mode == "subscription_included":
        return True
    return get_pricing_entry(model_name, provider=provider, base_url=base_url, api_key=api_key) is not None


def format_duration_compact(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.0f}m"
    hours = minutes / 60
    if hours < 24:
        remaining_min = int(minutes % 60)
        return f"{int(hours)}h {remaining_min}m" if remaining_min else f"{int(hours)}h"
    return f"{hours / 24:.1f}d"


def format_token_count_compact(value: int) -> str:
    abs_value = abs(int(value))
    if abs_value < 1_000:
        return str(int(value))

    sign = "-" if value < 0 else ""
    for threshold, suffix in ((1_000_000_000, "B"), (1_000_000, "M"), (1_000, "K")):
        if abs_value >= threshold:
            scaled = abs_value / threshold
            text = f"{scaled:.2f}" if scaled < 10 else f"{scaled:.1f}" if scaled < 100 else f"{scaled:.0f}"
            if "." in text:
                text = text.rstrip("0").rstrip(".")
            return f"{sign}{text}{suffix}"
