"""
Transcription Provider ABC
==========================

Pluggable-backend interface for speech-to-text. Providers register via
:meth:`PluginContext.register_transcription_provider`; the one named by
``stt.provider`` services :func:`tools.transcription_tools.transcribe_audio`
**when that name is not a built-in**. Built-ins (``BUILTIN_STT_PROVIDERS`` in
:mod:`tools.transcription_tools`) always win: the registry rejects colliding
names at registration and the dispatcher re-checks at dispatch time. The
``HERMES_LOCAL_STT_COMMAND`` shell escape hatch stays on the built-in
``local_command`` path.

Response contract for :meth:`TranscriptionProvider.transcribe`::

    success      bool
    transcript   str       transcribed text (empty when success=False)
    provider     str       provider name (for diagnostics)
    error        str       only when success=False
"""

from __future__ import annotations

import abc
import logging
from typing import Any, Dict, Optional

from agent.provider_base import CatalogProviderBase

logger = logging.getLogger(__name__)


class TranscriptionProvider(CatalogProviderBase):
    """Abstract base class for a speech-to-text backend.

    Subclasses must implement :attr:`name` (rejected at registration if it
    collides with a built-in STT name) and :meth:`transcribe`.
    """

    @abc.abstractmethod
    def transcribe(
        self,
        file_path: str,
        *,
        model: Optional[str] = None,
        language: Optional[str] = None,
        **extra: Any,
    ) -> Dict[str, Any]:
        """Transcribe the audio file at ``file_path`` into the module-docstring envelope.

        Implementations should NOT raise — convert exceptions to the error
        envelope so the gateway/CLI caller always gets a consistent shape. The
        dispatcher has already validated existence + size. ``model`` None →
        :meth:`default_model`; ``language`` is an optional BCP-47 hint. The
        dispatcher forwards ``prompt`` in ``extra`` when ``stt.prompt`` or a
        ``pre_transcription`` hook sets one — prompt-capable providers may use
        it as a vocabulary hint; unknown keys must be ignored.
        """
