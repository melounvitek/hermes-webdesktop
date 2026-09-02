"""Endpoint-family detection for Anthropic-compatible base URLs.

A dozen services speak the Anthropic Messages API but differ in auth style,
accepted beta headers, and request quirks (MiniMax, Kimi/Moonshot, DeepSeek,
OpenCode, Azure AI Foundry, Nous Portal, Bedrock). Every such difference is
decided from the configured base URL, so the predicates live together here.

Pure functions over a base-URL string - no I/O, no SDK, no credentials - so
both ``agent/anthropic_adapter.py`` and ``agent/anthropic_message_convert.py``
can depend on this module without a cycle. ``agent.anthropic_adapter``
re-exports every name below.
"""

from urllib.parse import urlparse

from utils import base_url_host_matches, base_url_hostname

_MINIMAX_ANTHROPIC_PREFIXES = ("https://api.minimax.io/anthropic", "https://api.minimaxi.com/anthropic")


def _normalize_base_url_text(base_url) -> str:
    """Coerce a base URL (str or ``httpx.URL``) to a stripped string; "" when falsy."""
    if not base_url:
        return ""
    return str(base_url).strip()


def _normalized_lower(base_url) -> str:
    """``_normalize_base_url_text`` + rstrip("/") + lower(), the shape most predicates match on."""
    return _normalize_base_url_text(base_url).rstrip("/").lower()


def _is_third_party_anthropic_endpoint(base_url: str | None) -> bool:
    """True for any non-anthropic.com endpoint (own API keys via x-api-key; skip OAuth detection).

    No base_url means the direct Anthropic API.
    """
    normalized = _normalized_lower(base_url)
    return bool(normalized) and "anthropic.com" not in normalized


def _is_kimi_coding_endpoint(base_url: str | None) -> bool:
    """True for Kimi's /coding endpoint, which requires a claude-code User-Agent."""
    return _normalized_lower(base_url).startswith("https://api.kimi.com/coding")


def _is_opencode_endpoint(base_url: str | None) -> bool:
    """True for OpenCode's Zen/Go relay (opencode.ai)."""
    return base_url_host_matches(base_url or "", "opencode.ai")


# Model-name prefixes identifying the Kimi / Moonshot family: official slugs
# (``kimi-k2.5``, ``kimi_thinking``, ``moonshot-v1-8k``) and release lines
# (``k1.5-…``, ``k2-thinking``, ``k25-…``, ``k3.x``/``k3-…``). Matched
# case-insensitively after stripping any ``vendor/`` prefix.
_KIMI_FAMILY_MODEL_PREFIXES = (
    "kimi-", "kimi_",
    "moonshot-", "moonshot_",
    "k1.", "k1-",
    "k2.", "k2-",
    "k25", "k2.5",
    "k3.", "k3-",
)

# Bare release slugs with no separator suffix (Kimi Coding Plan serves K3 as
# exactly ``k3``). Exact-match so unrelated names sharing the prefix don't match.
_KIMI_FAMILY_EXACT_SLUGS = frozenset({"k3"})


def _model_name_is_kimi_family(model: str | None) -> bool:
    if not isinstance(model, str):
        return False
    m = model.strip().lower()
    if not m:
        return False
    if "/" in m:  # ``moonshotai/kimi-k2.5`` -> ``kimi-k2.5``
        m = m.rsplit("/", 1)[-1]
    return m in _KIMI_FAMILY_EXACT_SLUGS or m.startswith(_KIMI_FAMILY_MODEL_PREFIXES)


def _is_kimi_family_endpoint(base_url: str | None, model: str | None = None) -> bool:
    """True for any Kimi / Moonshot Anthropic-Messages endpoint.

    Broader than ``_is_kimi_coding_endpoint``: also matches any api.kimi.com /
    moonshot.ai / moonshot.cn host, and any endpoint (e.g. a private gateway)
    whose *model* is in the Kimi family — the upstream still enforces Kimi's
    thinking semantics regardless of hostname. Decides whether unsigned
    reasoning_content-derived thinking blocks are preserved on replay.
    """
    if _is_kimi_coding_endpoint(base_url):
        return True
    if any(base_url_host_matches(base_url or "", d) for d in ("api.kimi.com", "moonshot.ai", "moonshot.cn")):
        return True
    return _model_name_is_kimi_family(model)


def _is_deepseek_anthropic_endpoint(base_url: str | None) -> bool:
    """True for DeepSeek's ``/anthropic`` route.

    In thinking mode DeepSeek requires prior-turn ``thinking`` blocks to round-trip
    ("The content[].thinking in the thinking mode must be passed back to the API"),
    while the generic third-party path strips them. Its blocks are unsigned, so it
    gets the same strip-signed / keep-unsigned policy as Kimi. Pinned to the
    ``/anthropic`` path so the OpenAI-compatible base URL is not misclassified.
    """
    if not base_url_host_matches(base_url or "", "api.deepseek.com"):
        return False
    return "/anthropic" in _normalized_lower(base_url)


def _is_nous_portal_endpoint(base_url: str | None) -> bool:
    """True for Nous Portal's Anthropic Messages route (Bearer JWT, verbatim catalog
    ids, native thinking-signature replay).

    Trusted hosts only: prod ``inference-api.nousresearch.com`` or the operator-set
    ``NOUS_INFERENCE_BASE_URL`` host (exact hostname equality, so neither lookalike
    domains nor sibling hosts of the override match).
    """
    if base_url_host_matches(base_url or "", "inference-api.nousresearch.com"):
        return True
    try:
        from hermes_cli.auth import _nous_inference_env_override

        override = _nous_inference_env_override()
    except Exception:
        return False
    if not override:
        return False
    override_host = base_url_hostname(override)
    return bool(override_host) and base_url_hostname(base_url or "") == override_host


def _requires_bearer_auth(base_url: str | None) -> bool:
    """True for Anthropic-compatible providers that need ``Authorization: Bearer``
    instead of ``x-api-key``: MiniMax, Azure AI Foundry, Palantir Foundry's LLM
    proxy, CommandCode, and Nous Portal.

    Palantir/CommandCode use hostname matching (not substring) so e.g.
    ``evil.com/palantirfoundry`` paths don't trigger Bearer auth.
    """
    if _is_nous_portal_endpoint(base_url):
        return True
    normalized = _normalized_lower(base_url)
    if not normalized:
        return False
    return (
        normalized.startswith(_MINIMAX_ANTHROPIC_PREFIXES)
        or "azure.com" in normalized
        or base_url_host_matches(normalized, "palantirfoundry.com")
        or base_url_host_matches(normalized, "api.commandcode.ai")
    )


def _base_url_needs_context_1m_beta(base_url: str | None) -> bool:
    """True for endpoints that still gate 1M context behind a beta (Azure)."""
    return "azure.com" in _normalize_base_url_text(base_url).lower()


def _is_minimax_anthropic_endpoint(base_url: str | None) -> bool:
    """True for MiniMax's Anthropic-compatible endpoints, which reject the
    fine-grained-tool-streaming and context-1m betas (stripped even though MiniMax
    also uses Bearer auth)."""
    return _normalized_lower(base_url).startswith(_MINIMAX_ANTHROPIC_PREFIXES)


def _is_azure_anthropic_endpoint(base_url: str | None) -> bool:
    """True for Azure-hosted Anthropic Messages endpoints serving ``/anthropic``:
    modern Foundry (``*.services.ai.azure.*``) and legacy Azure OpenAI
    (``*.openai.azure.*``) hosts. Opts them into ``api-version`` query plumbing.

    Deliberately no finite TLD allow-list, so sovereign/private clouds work.
    """
    normalized = _normalize_base_url_text(base_url)
    if not normalized:
        return False
    parsed = urlparse(normalized)
    host_padded = f".{(parsed.hostname or '').lower().rstrip('.')}."
    is_azure_host = ".services.ai.azure." in host_padded or ".openai.azure." in host_padded
    return is_azure_host and "/anthropic" in (parsed.path or "").lower()
