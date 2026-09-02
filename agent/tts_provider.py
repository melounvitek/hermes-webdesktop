"""
Text-to-Speech Provider ABC
============================

Pluggable-backend interface for TTS synthesis. Providers register via
``PluginContext.register_tts_provider()``; the one named by ``tts.provider``
services ``text_to_speech`` **only when that name is neither a built-in nor a
``tts.providers.<name>: type: command`` entry**. Resolution order:

1. Built-in providers (``BUILTIN_TTS_PROVIDERS`` in :mod:`tools.tts_tool`) —
   always win; :func:`agent.tts_registry.register_provider` rejects colliding
   names and the dispatcher re-checks at dispatch time.
2. Command-type providers from ``config.yaml`` — win over a same-name plugin
   because config is more local than a plugin install.
3. Plugin providers (this ABC) — for backends needing a Python SDK, streaming
   bytes, OAuth refresh, or voice-listing APIs the shell template can't express.

:meth:`TTSProvider.synthesize` writes audio to ``output_path`` and returns the
path; it should raise on failure — the dispatcher converts exceptions into the
standard ``{success: False, error: …}`` envelope.
"""

from __future__ import annotations

import abc
import logging
from typing import Any, Dict, Iterator, List, Optional

from agent.provider_base import CatalogProviderBase

logger = logging.getLogger(__name__)


DEFAULT_OUTPUT_FORMAT = "mp3"
VALID_OUTPUT_FORMATS = frozenset({"mp3", "wav", "ogg", "opus", "flac"})


class TTSProvider(CatalogProviderBase):
    """Abstract base class for a text-to-speech backend.

    Subclasses must implement :attr:`name` (rejected at registration if it
    collides with a built-in TTS provider name) and :meth:`synthesize`.
    """

    def list_voices(self) -> List[Dict[str, Any]]:
        """Voice catalog entries: ``{"id"}`` required; ``display`` / ``language``
        / ``gender`` / ``preview_url`` optional. Default: empty."""
        return []

    def default_voice(self) -> Optional[str]:
        """Id of the first voice entry, or None if not applicable."""
        voices = self.list_voices()
        if voices:
            return voices[0].get("id")
        return None

    @abc.abstractmethod
    def synthesize(
        self,
        text: str,
        output_path: str,
        *,
        voice: Optional[str] = None,
        model: Optional[str] = None,
        speed: Optional[float] = None,
        format: str = DEFAULT_OUTPUT_FORMAT,
        **extra: Any,
    ) -> str:
        """Synthesize ``text`` into ``output_path`` and return the written path.

        ``text`` is already truncated to the provider's max length and the
        parent directory exists. ``voice`` / ``model`` fall back to
        :meth:`default_voice` / :meth:`default_model` when None; ``speed`` is a
        rate multiplier providers may ignore. If ``format`` is unsupported, pick
        the closest equivalent and make ``output_path`` carry the right
        extension. Unknown ``extra`` keys must be ignored. Raise on failure.
        """

    def stream(
        self,
        text: str,
        *,
        voice: Optional[str] = None,
        model: Optional[str] = None,
        format: str = "opus",
        **extra: Any,
    ) -> Iterator[bytes]:
        """Stream synthesized audio bytes (optional).

        Default raises :class:`NotImplementedError`; the dispatcher then falls
        back to :meth:`synthesize` + read-whole-file. ``format`` defaults to
        ``opus`` because the primary streaming consumer is voice-bubble
        delivery (Telegram et al.), which requires Opus.
        """
        raise NotImplementedError(
            f"TTS provider {self.name!r} does not implement streaming "
            "synthesis. Use synthesize() instead, or implement stream() "
            "if your backend supports it."
        )

    def warm(self) -> None:
        """Speech output was just turned on; pre-load so the first reply is hot.

        Called from the TTS lease path (Desktop read-aloud / voice conversation)
        when this is the configured provider. Best-effort; default no-op.
        """

    def release(self) -> None:
        """Last speech-output lease released; free resident resources (counterpart
        of :meth:`warm`). Best-effort; default no-op."""

    @property
    def voice_compatible(self) -> bool:
        """Whether output suits voice-bubble delivery (mirrors
        ``tts.providers.<name>.voice_compatible``).

        True → the gateway converts to Opus via ffmpeg if needed; False →
        delivered as a regular audio attachment. Default False (opt in).
        """
        return False


def resolve_output_format(value: Optional[str]) -> str:
    """Clamp an output_format to :data:`VALID_OUTPUT_FORMATS`; invalid values
    coerce to :data:`DEFAULT_OUTPUT_FORMAT` so the tool surface forgives agent
    mistakes instead of rejecting them."""
    if not isinstance(value, str):
        return DEFAULT_OUTPUT_FORMAT
    v = value.strip().lower()
    if v in VALID_OUTPUT_FORMATS:
        return v
    return DEFAULT_OUTPUT_FORMAT
