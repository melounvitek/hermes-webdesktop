"""
Transcription Provider Registry
================================

Central map of registered STT providers. Populated by plugins at import-time
via :meth:`PluginContext.register_transcription_provider`; consumed by
:mod:`tools.transcription_tools` to dispatch :func:`transcribe_audio` calls
to the active plugin backend **when** the configured ``stt.provider`` name is
not a built-in.

Built-ins-always-win: a plugin name colliding with a built-in STT provider is
rejected at registration with a warning (re-checked at dispatch time in
:func:`tools.transcription_tools._dispatch_to_plugin_provider`).
"""

from __future__ import annotations

import logging

from agent.provider_registry import ProviderRegistry, lower_key
from agent.transcription_provider import TranscriptionProvider

logger = logging.getLogger(__name__)


# Names reserved for native built-in STT handlers. **Kept in sync with
# ``BUILTIN_STT_PROVIDERS`` in :mod:`tools.transcription_tools`** (a regression
# test in ``tests/agent/test_transcription_registry.py::TestBuiltinSync`` fails
# on drift); importing it directly would be a circular import.
_BUILTIN_NAMES = frozenset({
    "local",
    "local_command",
    "groq",
    "openai",
    "mistral",
    "xai",
    "elevenlabs",
    "deepinfra",
})


def _warn_builtin_collision(key: str) -> None:
    logger.warning(
        "Transcription provider '%s' shadows a built-in name; registration "
        "ignored. Built-in STT providers (%s) always win — pick a different "
        "name.",
        key, ", ".join(sorted(_BUILTIN_NAMES)),
    )


# Case-insensitive, whitespace-tolerant keys mirror how
# ``tools.transcription_tools`` normalizes the configured ``stt.provider``.
_registry: ProviderRegistry[TranscriptionProvider] = ProviderRegistry(
    label="Transcription",
    provider_cls=TranscriptionProvider,
    logger=logger,
    normalize=lower_key,
    builtin_names=_BUILTIN_NAMES,
    on_builtin_collision=_warn_builtin_collision,
)
_registry.export(globals())
