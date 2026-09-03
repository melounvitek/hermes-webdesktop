"""Plugin-registered TTS providers for ``tools.tts_tool``.

Routes ``tts.provider: <name>`` values that are neither built-in nor a
``type: command`` entry to a :class:`agent.tts_provider.TTSProvider`
registered by a plugin. Discovery goes through
``hermes_cli.plugins._ensure_plugins_discovered`` (imported lazily so the
tool module stays importable without the plugin machinery).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from tools.tts_command_provider import (
    BUILTIN_TTS_PROVIDERS,
    DEFAULT_COMMAND_TTS_OUTPUT_FORMAT,
    _get_named_provider_config,
    _is_command_provider_config,
)

logger = logging.getLogger("tools.tts_tool")


def _lookup_plugin_provider(key: str, *, discover: bool = True, retry: bool = False):
    """The registered ``TTSProvider`` named *key*, or None.

    ``discover`` runs plugin discovery first; ``retry`` re-discovers with
    ``force=True`` on a miss (long-lived sessions may have discovered plugins
    before this one was installed/enabled). Raises on registry/discovery
    failure — callers decide whether that is fatal.
    """
    from agent.tts_registry import get_provider

    if discover:
        from hermes_cli.plugins import _ensure_plugins_discovered

        _ensure_plugins_discovered()
    plugin_provider = get_provider(key)
    if plugin_provider is None and retry:
        _ensure_plugins_discovered(force=True)
        plugin_provider = get_provider(key)
    return plugin_provider


def _dispatch_to_plugin_provider(
    text: str,
    output_path: str,
    provider: str,
    tts_config: Dict[str, Any],
) -> Optional[str]:
    """Route to a plugin-registered TTS provider; None means "fall through".

    Invariants enforced here even though the caller checks them too, so a
    caller refactor can't silently break them:

    1. Built-in names never reach the plugin registry.
    2. A same-named ``type: command`` provider wins over a plugin.
    3. Dispatch fires only for a registered :class:`TTSProvider` whose name
       equals the configured value; unknown names return None.

    Plugin exceptions propagate — the outer ``text_to_speech_tool`` converts
    them to the standard error envelope.
    """
    if not provider:
        return None
    key = provider.lower().strip()
    if key in BUILTIN_TTS_PROVIDERS:
        return None
    if _is_command_provider_config(_get_named_provider_config(tts_config, key)):
        return None
    try:
        plugin_provider = _lookup_plugin_provider(key, retry=True)
    except Exception as exc:  # noqa: BLE001 — discovery failure is non-fatal
        logger.debug("tts plugin dispatch skipped (discovery failed): %s", exc)
        return None
    if plugin_provider is None:
        return None

    # voice/model/speed/format are optional per the TTSProvider.synthesize
    # contract; providers fall back to their own defaults on None.
    cfg = tts_config if isinstance(tts_config, dict) else {}
    voice = cfg.get("voice")
    model = cfg.get("model")
    speed = cfg.get("speed")
    fmt = cfg.get("output_format", DEFAULT_COMMAND_TTS_OUTPUT_FORMAT)

    logger.info("Generating speech with plugin TTS provider '%s'...", key)
    written = plugin_provider.synthesize(
        text,
        output_path,
        voice=voice if isinstance(voice, str) and voice else None,
        model=model if isinstance(model, str) and model else None,
        speed=float(speed) if isinstance(speed, (int, float)) else None,
        format=str(fmt).lower() if fmt else "mp3",
    )
    # Contract: returns the (possibly rewritten) output path; tolerate None.
    return written if isinstance(written, str) and written else output_path


def _plugin_provider_is_voice_compatible(provider: str) -> bool:
    """True when the registered plugin provider opts into voice-bubble delivery.

    Any registry/property failure means False (safe default, like command providers).
    """
    if not provider:
        return False
    key = provider.lower().strip()
    if key in BUILTIN_TTS_PROVIDERS:
        return False
    try:
        plugin_provider = _lookup_plugin_provider(key, discover=False)
        if plugin_provider is None:
            return False
        return bool(plugin_provider.voice_compatible)
    except Exception as exc:  # noqa: BLE001
        logger.debug("tts plugin voice_compatible check failed for '%s': %s", key, exc)
        return False


def _plugin_provider_is_available(provider: str) -> bool:
    """``check_fn`` leg for plugin names: discovered provider reports ``is_available()``; any failure is False."""
    try:
        plugin = _lookup_plugin_provider(provider)
        return bool(plugin and plugin.is_available())
    except Exception:
        return False
