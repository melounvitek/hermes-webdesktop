"""
Image Generation Provider Registry
==================================

Central map of registered providers. Populated by plugins at import-time via
``PluginContext.register_image_gen_provider()``; consumed by the
``image_generate`` tool to dispatch each call to the active backend.

Active selection
----------------
The active provider is chosen by ``image_gen.provider`` in ``config.yaml``.
If unset, :func:`get_active_provider` applies fallback logic:

1. If exactly one *available* provider is registered, use it.
2. Otherwise if a provider named ``fal`` is registered and available, use it
   (legacy default — matches pre-plugin behavior).
3. Otherwise return ``None`` (the tool surfaces a helpful error pointing
   the user at ``hermes tools``).
"""

from __future__ import annotations

import logging
from typing import Optional

from agent.image_gen_provider import ImageGenProvider
from agent.provider_registry import ProviderRegistry, configured_provider_name, is_available_safe

logger = logging.getLogger(__name__)


_registry: ProviderRegistry[ImageGenProvider] = ProviderRegistry(
    label="Image gen", provider_cls=ImageGenProvider, logger=logger,
)
_registry.export(globals())


def get_active_provider() -> Optional[ImageGenProvider]:
    """Resolve the currently-active provider.

    **Availability semantics** (mirrors :mod:`agent.web_search_registry`):
    an explicitly configured provider is returned even if ``is_available()``
    is False, so the dispatcher surfaces a precise "X_API_KEY is not set"
    error instead of silently switching backends. Only the unconfigured
    fallback path is filtered by availability.
    """
    configured = configured_provider_name("image_gen", logger)
    snapshot = _registry.merged()

    def _available(p: ImageGenProvider) -> bool:
        return is_available_safe(p, logger, "image_gen provider %s.is_available() raised %s")

    if configured:
        provider = snapshot.get(configured)
        if provider is not None:
            return provider
        logger.debug(
            "image_gen.provider='%s' configured but not registered; falling back",
            configured,
        )

    available = [p for p in snapshot.values() if _available(p)]
    if len(available) == 1:
        return available[0]

    fal = snapshot.get("fal")
    if fal is not None and _available(fal):
        return fal

    return None
