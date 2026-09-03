"""
TTS Provider Registry
=====================

Central map of registered TTS providers. Populated by plugins at import-time
via :meth:`PluginContext.register_tts_provider`; consumed by
:mod:`tools.tts_tool` to dispatch ``text_to_speech`` calls to the active
plugin backend **when** the configured ``tts.provider`` name is neither a
built-in nor a command-type provider.

Built-ins-always-win: a plugin name colliding with a built-in TTS provider is
rejected at registration with a warning (re-checked at dispatch time in
:func:`tools.tts_tool._dispatch_to_plugin_provider`).

Command-providers-win-over-plugins is enforced by the dispatcher, not here:
it checks for a same-name ``tts.providers.<name>: type: command`` entry before
consulting the registry (a name declared in the user's config.yaml is more
specific to their setup than an installed plugin).
"""

from __future__ import annotations

import logging

from agent.provider_registry import ProviderRegistry, lower_key
from agent.tts_provider import TTSProvider

logger = logging.getLogger(__name__)


# Names reserved for native built-in TTS handlers. **Kept in sync with
# ``BUILTIN_TTS_PROVIDERS`` in :mod:`tools.tts_tool`** (a regression test in
# ``tests/agent/test_tts_registry.py::TestBuiltinSync`` fails on drift);
# importing it directly would be a circular import.
_BUILTIN_NAMES = frozenset({
    "edge",
    "elevenlabs",
    "openai",
    "minimax",
    "xai",
    "mistral",
    "gemini",
    "neutts",
    "kittentts",
    "piper",
    "deepinfra",
})


def _warn_builtin_collision(key: str) -> None:
    logger.warning(
        "TTS provider '%s' shadows a built-in name; registration ignored. "
        "Built-in TTS providers (%s) always win — pick a different name.",
        key, ", ".join(sorted(_BUILTIN_NAMES)),
    )


# Case-insensitive, whitespace-tolerant keys mirror how
# ``tools.tts_tool._get_provider`` normalizes the configured ``tts.provider``.
_registry: ProviderRegistry[TTSProvider] = ProviderRegistry(
    label="TTS", provider_cls=TTSProvider, logger=logger, normalize=lower_key,
    builtin_names=_BUILTIN_NAMES, on_builtin_collision=_warn_builtin_collision,
)
_registry.export(globals())
