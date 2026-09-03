"""One-shot keyless-ring rescue for failed keyed/configured web calls.

Stateless by design: a rescue routes THIS call through the free-tier ring
(plugins/web/keyless_mcp.py); the next web_search/web_extract call attempts
the chosen backend again. Callers must never cache a rescue-served response,
or the one-shot rescue becomes sticky for a whole TTL. Names are re-imported by
tools/web_tools.py (``tools.web_tools._rescue_eligible``); logs under the origin logger.
"""

import logging

logger = logging.getLogger("tools.web_tools")

# Ring vendor -> env var holding its paid key (keyed mode ⇒ eligible for rescue).
_RING_KEY_VARS = {
    "exa": "EXA_API_KEY", "parallel": "PARALLEL_API_KEY", "firecrawl": "FIRECRAWL_API_KEY", "keenable": "KEENABLE_API_KEY",
}


def _keyless_rescue_enabled() -> bool:
    """``web.keyless_rescue`` (default on), implicitly off when the keyless tier is disabled."""
    from tools.web_tools import _load_web_config

    if not _load_web_config().get("keyless_rescue", True):
        return False
    try:
        from agent.web_search_registry import _keyless_tier_enabled

        return _keyless_tier_enabled()
    except Exception as exc:  # noqa: BLE001 — registry optional
        logger.debug("keyless rescue tier check failed: %s", exc)
        return False


def _rescue_eligible(provider) -> bool:
    """True when a failed call on *provider* should get a one-shot rescue.

    Eligible: a keyed/configured path — any non-ring backend, or a ring vendor in keyed mode. A ring
    vendor already in keyless mode is NOT eligible: its failure means the ring was already walked.
    """
    if not _keyless_rescue_enabled() or provider is None:
        return False
    try:
        from plugins.web.keyless_mcp import _KEYLESS_RING, use_keyless

        name = getattr(provider, "name", "")
        if name in _KEYLESS_RING:
            from agent.web_search_provider import get_provider_env

            key_var = _RING_KEY_VARS.get(name, "")
            api_key = get_provider_env(key_var) if key_var else ""
            return not use_keyless(name, api_key)
        return True
    except Exception as exc:  # noqa: BLE001 — rescue is best-effort
        logger.debug("rescue eligibility check failed: %s", exc)
        return False


def _rescue_search(provider_name: str, original_error: str, query: str, limit: int) -> dict:
    """Rescue a failed search via the ring; annotate the result with the original failure."""
    from plugins.web.keyless_mcp import search_with_failover

    logger.warning("web_search backend '%s' failed (%s); one-shot keyless rescue", provider_name, (original_error or "")[:200])
    rescued = search_with_failover(provider_name, query, limit)
    if rescued.get("success"):
        data = rescued.setdefault("data", {})
        data["rescued_from"] = provider_name
        data["backend_error"] = (
            f"Configured backend '{provider_name}' failed this call "
            f"({(original_error or 'unknown error')[:300]}); result served "
            "by the keyless free tier. The next call will use "
            f"'{provider_name}' again."
        )
        return rescued
    # Ring also failed: the ORIGINAL error names the user's setup, so lead with it.
    return {
        "success": False,
        "error": f"{original_error or 'search failed'} (keyless rescue also failed: {rescued.get('error', 'unknown')})",
    }


def _policy_blocked_result(result: dict) -> bool:
    """True for a website-policy refusal — intentional, never rescued (it would fetch blocked content)."""
    return bool(result.get("blocked_by_policy")) or "blocked by website policy" in str(result.get("error") or "").lower()


def _rescue_extract(provider_name: str, urls: list, results: list) -> list:
    """Rescue a whole-batch extract failure via the ring.

    Only genuine failures are re-fetched; policy-blocked entries are preserved verbatim. If the
    provider broke url/result order parity, every entry is treated as rescueable and the ring's
    list replaces the batch wholesale.
    """
    from plugins.web.keyless_mcp import extract_with_failover

    parity = len(results) == len(urls)
    rescue_idx = [i for i, r in enumerate(results) if not parity or not _policy_blocked_result(r)]
    if not rescue_idx:
        return results  # every failure is an intentional policy block

    rescue_urls = [urls[i] for i in rescue_idx] if parity else list(urls)
    original_error = next((results[i].get("error") for i in rescue_idx if results[i].get("error")), "extract failed")
    logger.warning(
        "web_extract backend '%s' failed all %d URL(s) (%s); one-shot keyless rescue",
        provider_name, len(rescue_urls), (original_error or "")[:200],
    )
    rescued = extract_with_failover(provider_name, list(rescue_urls))
    if rescued and all(r.get("error", "") for r in rescued):
        return results  # rescue also failed everywhere: keep original errors
    for r in rescued:
        if not r.get("error"):
            meta = r.setdefault("metadata", {})
            if isinstance(meta, dict):
                meta["rescued_from"] = provider_name
                meta["backend_error"] = (original_error or "")[:300]
    if parity and len(rescued) == len(rescue_idx):
        merged = list(results)
        for pos, i in enumerate(rescue_idx):
            merged[i] = rescued[pos]
        return merged
    return rescued
