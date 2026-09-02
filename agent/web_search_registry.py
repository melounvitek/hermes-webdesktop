"""
Web Search Provider Registry
============================

Central map of registered web providers. Populated by plugins at import-time
via :meth:`PluginContext.register_web_search_provider`; consumed by the
``web_search`` and ``web_extract`` tool wrappers in :mod:`tools.web_tools` to
dispatch each call to the active backend.

Active selection
----------------
The active provider is chosen by configuration with this precedence:

1. ``web.search_backend`` / ``web.extract_backend``
   (per-capability override).
2. ``web.backend`` (shared fallback).
3. If exactly one capability-eligible provider is registered AND available,
   use it.
4. Legacy preference order — ``firecrawl`` → ``parallel`` → ``tavily`` →
   ``exa`` → ``searxng`` → ``brave-free`` → ``ddgs`` — filtered by
   availability. Matches the historic ``tools.web_tools._get_backend()``
   candidate order so installs that never set a config key keep landing
   on the same provider they did before the plugin migration.
5. Otherwise ``None`` — the tool surfaces a helpful error pointing at
   ``hermes tools``.

The capability filter (``supports_search`` / ``supports_extract``) is
applied at every step so a search-only provider (``brave-free``)
configured as ``web.extract_backend`` correctly falls through to an
extract-capable backend.
"""

from __future__ import annotations

import logging
from typing import Optional

from agent.provider_registry import ProviderRegistry, is_available_safe
from agent.web_search_provider import WebSearchProvider

logger = logging.getLogger(__name__)


_registry: ProviderRegistry[WebSearchProvider] = ProviderRegistry(
    label="Web", provider_cls=WebSearchProvider, logger=logger,
)
_registry.export(globals())


# ---------------------------------------------------------------------------
# Active-provider resolution
# ---------------------------------------------------------------------------


def _read_config_key(*path: str) -> Optional[str]:
    """Resolve a dotted config key from ``config.yaml``. Returns None on miss."""
    try:
        from hermes_cli.config import load_config_readonly

        cfg = load_config_readonly()
        cur = cfg
        for segment in path:
            if not isinstance(cur, dict):
                return None
            cur = cur.get(segment)
        if isinstance(cur, str) and cur.strip():
            return cur.strip()
    except Exception as exc:
        logger.debug("Could not read config %s: %s", ".".join(path), exc)
    return None


def _configured_backend(capability: str) -> Optional[str]:
    """``web.<capability>_backend`` (preferred) or ``web.backend`` (shared fallback)."""
    return _read_config_key("web", f"{capability}_backend") or _read_config_key("web", "backend")


# Legacy preference order — preserves behaviour for users who set no
# ``web.backend`` / ``web.<capability>_backend`` config key at all. Matches
# the historic candidate order in :func:`tools.web_tools._get_backend`
# (paid providers first so existing paid setups don't get downgraded to
# a free tier on upgrade). Filtered by ``is_available()`` at walk time so
# we don't surface a provider the user has no credentials for.
_LEGACY_PREFERENCE = (
    "firecrawl",
    "parallel",
    "tavily",
    "exa",
    "searxng",
    "brave-free",
    "ddgs",
)

# Keyless free-tier walk — strictly LAST-resort, tried only after the
# availability-filtered legacy walk finds nothing (i.e. the user has zero
# web credentials and no importable ddgs). Ring vendors expose public
# anonymous free tiers (see plugins/web/keyless_mcp.py). Unpinned keyless
# traffic round-robins across the ring per request (the ring cursor lives
# in keyless_mcp; an explicit `hermes tools` pick bypasses this walk
# entirely, and rate-limited requests fail over to the next ring vendor).
# Disable the tier with ``web.keyless_fallback: false``.
_KEYLESS_PREFERENCE = (
    "exa",
    "parallel",
    "firecrawl",
    "keenable",
)


def _keyless_preference() -> tuple:
    """Return the keyless walk order for resolution.

    Delegates the entry-vendor choice to the ring cursor in
    :mod:`plugins.web.keyless_mcp` (round-robin per request, seeded by the
    per-process random session id) so resolution and dispatch agree on
    which vendor a fresh install starts at. The remaining vendors follow
    in ring order as fallbacks for registration gaps.
    """
    try:
        from plugins.web.keyless_mcp import _KEYLESS_RING, _ring_cursor

        start = _ring_cursor % len(_KEYLESS_RING)
        return tuple(
            _KEYLESS_RING[(start + i) % len(_KEYLESS_RING)]
            for i in range(len(_KEYLESS_RING))
        )
    except Exception as exc:  # noqa: BLE001 — ring optional in stripped envs
        logger.debug("keyless ring order unavailable: %s", exc)
    return _KEYLESS_PREFERENCE


def _resolve(configured: Optional[str], *, capability: str) -> Optional[WebSearchProvider]:
    """Resolve the active provider for a capability ("search" | "extract").

    Rules, in order (see module docstring): explicit config wins even when
    ``is_available()`` is False (the dispatcher surfaces a precise
    "X_API_KEY is not set" error instead of a silent switch); then the single
    available capable provider; then the availability-filtered legacy walk;
    then the keyless free-tier walk; else None.
    """
    snapshot = _registry.merged()

    def _capable(p: WebSearchProvider) -> bool:
        if capability == "search":
            return bool(p.supports_search())
        if capability == "extract":
            return bool(p.supports_extract())
        return False

    def _available(p: WebSearchProvider) -> bool:
        return is_available_safe(p, logger, "provider %s.is_available() raised %s")

    if configured:
        provider = snapshot.get(configured)
        if provider is not None and _capable(provider):
            return provider
        if provider is None:
            logger.debug(
                "web backend '%s' configured but not registered; falling back",
                configured,
            )
        else:
            logger.debug(
                "web backend '%s' configured but does not support '%s'; falling back",
                configured, capability,
            )

    # Fallbacks are availability-filtered so a registered-but-keyless provider
    # never becomes "active" on a fresh install.
    eligible = [p for p in snapshot.values() if _capable(p) and _available(p)]
    if len(eligible) == 1:
        return eligible[0]

    for legacy in _LEGACY_PREFERENCE:
        provider = snapshot.get(legacy)
        if provider is not None and _capable(provider) and _available(provider):
            return provider

    # Keyless free tier (anonymous public MCP tiers) is last-resort only: it is
    # reachable solely when the legacy walk found nothing, never pre-empting a
    # keyed setup. Disabled via ``web.keyless_fallback: false``.
    if _keyless_tier_enabled():
        for name in _keyless_preference():
            provider = snapshot.get(name)
            if provider is None or not _capable(provider):
                continue
            try:
                if provider.is_keyless_available():
                    return provider
            except Exception as exc:  # noqa: BLE001 — buggy provider skipped
                logger.debug(
                    "provider %s.is_keyless_available() raised %s", name, exc
                )

    return None


def _keyless_tier_enabled() -> bool:
    """Read ``web.keyless_fallback`` from config.yaml (default: enabled)."""
    try:
        from hermes_cli.config import load_config

        web_cfg = load_config().get("web") or {}
        return bool(web_cfg.get("keyless_fallback", True))
    except Exception as exc:  # noqa: BLE001 — config layer optional
        logger.debug("keyless_fallback config read failed: %s", exc)
        return True


def _disabled_web_plugin_for(configured: Optional[str] = None, *, capability: Optional[str] = None) -> Optional[str]:
    """Plugin key of a *disabled* bundled web plugin that would have provided
    the configured backend (``web.<capability>_backend`` → ``web.backend``),
    or None.

    Lets the dispatcher say "re-enable web-firecrawl" instead of a misleading
    "No web extract provider configured" when the backend IS configured but
    listed in ``plugins.disabled``. Resolving from config.yaml (rather than
    the resolved backend) matters because a disabled provider fails the
    availability gate and the dispatcher silently drops to the default.
    Bundled web plugins live under ``web/<vendor>`` with the provider name
    differing only by hyphen/underscore, so both sides are normalized.
    """
    def _norm(s: str) -> str:
        return s.strip().lower().replace("-", "_")

    if not configured and capability in ("search", "extract"):
        configured = _configured_backend(capability)
    if not configured:
        return None

    want = _norm(configured)
    try:
        from hermes_cli.plugins import get_plugin_manager

        pm = get_plugin_manager()
        for key, loaded in pm._plugins.items():
            if not isinstance(key, str) or not key.startswith("web/"):
                continue
            if loaded.enabled:
                continue
            if loaded.error != "disabled via config":
                continue
            vendor = key.split("/", 1)[1]
            if _norm(vendor) == want:
                return key
    except Exception as exc:  # noqa: BLE001 — diagnostics are best-effort
        logger.debug("disabled-web-plugin lookup failed: %s", exc)
    return None


def get_active_search_provider() -> Optional[WebSearchProvider]:
    """Resolve the currently-active web search provider."""
    return _resolve(_configured_backend("search"), capability="search")


def get_active_extract_provider() -> Optional[WebSearchProvider]:
    """Resolve the currently-active web extract provider."""
    return _resolve(_configured_backend("extract"), capability="extract")

