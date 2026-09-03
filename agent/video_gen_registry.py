"""
Video Generation Provider Registry
==================================

Central map of registered providers. Populated by plugins at import-time via
``PluginContext.register_video_gen_provider()``; consumed by the
``video_generate`` tool to dispatch each call to the active backend.

Active selection
----------------
The active provider is chosen by ``video_gen.provider`` in ``config.yaml``.
If unset, :func:`get_active_provider` applies fallback logic:

1. If exactly one *available* provider is registered, use it.
2. Otherwise return ``None`` (the tool surfaces a helpful error pointing
   the user at ``hermes tools``).

Mirrors ``agent/image_gen_registry.py``: the unconfigured fallback is
filtered by ``is_available()`` so a box with credentials for only one backend
(e.g. DeepInfra, while ``fal``/``xai`` register unconditionally) auto-selects
it instead of returning ``None``. Unlike image gen there is no legacy ``fal``
preference, and a configured-but-unregistered name fails closed.
"""

from __future__ import annotations

import logging
from typing import Optional

from agent.provider_registry import ProviderRegistry, configured_provider_name, is_available_safe
from agent.video_gen_provider import VideoGenProvider

logger = logging.getLogger(__name__)


_registry: ProviderRegistry[VideoGenProvider] = ProviderRegistry(
    label="Video gen", provider_cls=VideoGenProvider, logger=logger,
)
_registry.export(globals())


def get_active_provider() -> Optional[VideoGenProvider]:
    """Resolve the currently-active provider (see module docstring)."""
    configured = configured_provider_name("video_gen", logger)
    snapshot = _registry.merged()

    if configured:
        provider = snapshot.get(configured)
        if provider is not None:
            return provider
        logger.debug(
            "video_gen.provider='%s' configured but not registered; failing closed", configured
        )
        return None

    available = [
        p for p in snapshot.values()
        if is_available_safe(p, logger, "video_gen provider %s.is_available() raised %s")
    ]
    if len(available) == 1:
        return available[0]

    return None
