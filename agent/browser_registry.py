"""
Browser Provider Registry
=========================

Central map of registered cloud browser providers. Populated by plugins at
import-time via :meth:`PluginContext.register_browser_provider`; consumed by
:func:`tools.browser_tool._get_cloud_provider` to route each cloud-mode
``browser_*`` tool call to the active backend.

Active selection
----------------
The active provider is chosen by configuration with this precedence:

1. ``browser.cloud_provider`` in ``config.yaml`` (explicit override).
2. Legacy preference order — ``browser-use`` → ``browserbase`` — filtered by
   availability. Matches the historic auto-detect order in
   :func:`tools.browser_tool._get_cloud_provider` (Browser Use checked first
   because it covers both the managed Nous gateway and direct API key path;
   Browserbase as the older direct-credentials fallback). ``firecrawl`` is
   intentionally NOT in the legacy walk — users only get Firecrawl as a
   cloud browser when they explicitly set ``browser.cloud_provider:
   firecrawl``, matching pre-migration behaviour where Firecrawl was never
   auto-selected.
3. Otherwise ``None`` — the dispatcher falls back to local browser mode.

The explicit-config branch (rule 1) intentionally ignores ``is_available()``
so the dispatcher surfaces a typed "X_API_KEY is not set" error to the user
instead of silently switching backends. Matches the legacy
:func:`tools.browser_tool._get_cloud_provider` behaviour for configured names.

Note: there is no "capability" split here (unlike the web subsystem, which
has search/extract/crawl). Every browser provider implements the full
:class:`agent.browser_provider.BrowserProvider` lifecycle; the registry's
job is purely selection, not capability routing.
"""

from __future__ import annotations

import logging
from typing import Optional

from agent.browser_provider import BrowserProvider
from agent.provider_registry import ProviderRegistry, is_available_safe

logger = logging.getLogger(__name__)


_registry: ProviderRegistry[BrowserProvider] = ProviderRegistry(
    label="Browser", provider_cls=BrowserProvider, logger=logger,
)
_registry.export(globals())


# ---------------------------------------------------------------------------
# Active-provider resolution
# ---------------------------------------------------------------------------


# Auto-detect order when ``browser.cloud_provider`` is unset (pre-migration
# walk of :func:`tools.browser_tool._get_cloud_provider`); see :func:`_resolve`
# for why Firecrawl is absent.
_LEGACY_PREFERENCE = (
    "browser-use",
    "browserbase",
)


def _resolve(configured: Optional[str]) -> Optional[BrowserProvider]:
    """Resolve the active browser provider (rules in the module docstring).

    There is intentionally NO "single-eligible shortcut" (unlike
    :func:`agent.web_search_registry._resolve`): only ``_LEGACY_PREFERENCE``
    names are auto-eligible. Firecrawl shares its API key with the *web*
    extract plugin, so a user with ``FIRECRAWL_API_KEY`` must never be routed
    to a paid cloud browser without setting ``browser.cloud_provider``; the
    same gate applies to third-party browser-provider plugins.
    """
    snapshot = _registry.merged()

    if configured == "local":
        return None

    # Explicit config wins regardless of is_available(): the dispatcher then
    # surfaces a precise "X_API_KEY is not set" error instead of a silent switch.
    if configured:
        provider = snapshot.get(configured)
        if provider is not None:
            return provider
        logger.debug(
            "browser cloud_provider '%s' configured but not registered; "
            "falling back to auto-detect",
            configured,
        )

    for legacy in _LEGACY_PREFERENCE:
        provider = snapshot.get(legacy)
        if provider is not None and is_available_safe(
            provider, logger,
            "Browser provider %s.is_available() raised %s — treating as unavailable",
            level=logging.WARNING, exc_info=True,
        ):
            return provider

    return None

