"""Resolve the active profile's STT/TTS config for CLIENT-DIRECT voice.

The desktop can skip the audio relay hop (mic → gateway → provider) by calling
voice providers directly with the profile's own credentials, fetched over the
authenticated REST channel at voice-session start. This is the single resolver
behind ``GET /api/audio/voice-config``; it reuses the exact provider/key/model/
language chains of ``tools.transcription_tools`` and ``tools.tts_tool`` so the
client receives byte-for-byte what the gateway itself would use.

Design rules:

* **Same-trust boundary.** The endpoint is profile-scoped and rides the same
  auth as every REST route — a client that can reach it can already drive the
  agent, so handing it the voice key is no escalation. Keys still never touch
  client disk (renderer memory only) and are never logged here.
* **Relay is the floor, not an error.** Server-host-only providers (local
  whisper, edge-tts, command providers, plugins) resolve to ``{"mode": "relay"}``
  and the desktop falls back to ``/api/audio/*``. A resolution failure also
  degrades to relay — the relay endpoint surfaces the real error.
* **No new key stores.** Everything is read through the live resolvers.

Config gate: ``voice.client_direct`` (config.yaml, default ``true``). When
false every provider reports relay and the desktop behaves as before.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Wire shapes the desktop knows how to speak. Anything else → relay.
#   openai-multipart : POST {base_url}/audio/transcriptions (multipart, Bearer)
#   xai-stt          : POST {base_url}/stt (multipart, Bearer, format=true)
#   elevenlabs-stt   : POST {base_url}/speech-to-text (multipart, xi-api-key)
#   openai-speech    : POST {base_url}/audio/speech (JSON, Bearer) → audio bytes
#   elevenlabs-tts   : POST {base_url}/text-to-speech/{voice_id} (JSON, xi-api-key)
STT_WIRE_OPENAI = "openai-multipart"
STT_WIRE_XAI = "xai-stt"
STT_WIRE_ELEVENLABS = "elevenlabs-stt"
TTS_WIRE_OPENAI = "openai-speech"
TTS_WIRE_ELEVENLABS = "elevenlabs-tts"

_RELAY: Dict[str, Any] = {"mode": "relay"}


def _client_direct_enabled() -> bool:
    try:
        from hermes_cli.config import load_config

        voice_cfg = load_config().get("voice") or {}
        if not isinstance(voice_cfg, dict):
            return True
        value = voice_cfg.get("client_direct", True)
    except Exception:
        return True
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return True


def _relay(reason: str) -> Dict[str, Any]:
    """A relay verdict that tells the client (and logs) WHY, without secrets."""
    return {"mode": "relay", "reason": reason}


def _section(config: Any, provider: str) -> Dict[str, Any]:
    """The provider's own sub-dict of an STT/TTS config, shape-guarded."""
    section = config.get(provider) if isinstance(config, dict) else None
    return section if isinstance(section, dict) else {}


def _direct(wire: str, provider: str, base_url: Any, api_key: str, model: Any, **extra: Any) -> Dict[str, Any]:
    return {"mode": "direct", "wire": wire, "provider": provider, "base_url": base_url,
            "api_key": api_key, "model": model, **extra}


def _deepinfra_model(section: Dict[str, Any], kind: str) -> Optional[str]:
    """Configured model, else the first catalog model of ``kind`` (stt/tts)."""
    from hermes_cli.models import deepinfra_model_ids

    model = section.get("model")
    if not model:
        candidates = deepinfra_model_ids(kind)
        model = candidates[0] if candidates else None
    return model


# ---------------------------------------------------------------------------
# STT
# ---------------------------------------------------------------------------

# provider -> (env var, default-model attr on transcription_tools, base_url).
# ``base_url`` is a transcription_tools attr name or a literal URL.
_STT_KEYED: Dict[str, tuple[str, str, str]] = {
    "groq": ("GROQ_API_KEY", "DEFAULT_GROQ_STT_MODEL", "GROQ_BASE_URL"),
    "mistral": ("MISTRAL_API_KEY", "DEFAULT_MISTRAL_STT_MODEL", "https://api.mistral.ai/v1"),
}


def _resolve_stt_client_config() -> Dict[str, Any]:
    from tools import transcription_tools as tt

    stt_config = tt._load_stt_config()
    if not tt.is_stt_enabled(stt_config):
        return _relay("stt disabled")

    provider = tt._get_provider(stt_config)

    # Server-host-only providers: local whisper, the env-var command escape
    # hatch, declared command providers, and anything plugin-registered.
    if tt._is_local_stt_provider(provider, stt_config):
        return _relay("local provider")
    if provider not in tt.BUILTIN_STT_PROVIDERS:
        return _relay("command/plugin provider")

    language = tt._resolve_stt_language(
        provider, stt_config,
        extra_keys=("language_code",) if provider == "elevenlabs" else (),
    )
    section = _section(stt_config, provider)

    def direct(wire: str, base_url: Any, api_key: str, model: Any) -> Dict[str, Any]:
        return _direct(wire, provider, base_url, api_key, model, language=language)

    def env_base_url(env_var: str, default: str) -> str:
        return str(section.get("base_url") or tt.get_env_value(env_var) or default).strip().rstrip("/")

    if provider in _STT_KEYED:
        env_var, default_model, base = _STT_KEYED[provider]
        api_key = tt._resolve_provider_key(env_var, provider)
        if not api_key:
            return _relay("no credentials")
        return direct(STT_WIRE_OPENAI, getattr(tt, base, base), api_key,
                      section.get("model") or getattr(tt, default_model))

    if provider == "openai":
        # Handles the Nous-managed selection too: the resolver returns the
        # user's own gateway token + managed base URL, which is exactly the
        # credential the client should use.
        try:
            api_key, base_url = tt._resolve_openai_audio_client_config()
        except ValueError as exc:
            return _relay(f"openai resolution failed: {exc}")
        return direct(STT_WIRE_OPENAI, base_url, api_key, section.get("model") or tt.DEFAULT_STT_MODEL)

    if provider == "xai":
        # API key only. An xAI OAuth bearer refreshes server-side mid-session;
        # handing it out strands the client on the first 401. Relay instead.
        api_key = str(tt.get_env_value("XAI_API_KEY") or "").strip()
        if not api_key:
            return _relay("xai oauth (server-managed) or no credentials")
        return direct(STT_WIRE_XAI, env_base_url("XAI_STT_BASE_URL", tt.XAI_STT_BASE_URL), api_key, None)

    if provider == "elevenlabs":
        api_key = tt._resolve_provider_key("ELEVENLABS_API_KEY", "elevenlabs")
        if not api_key:
            return _relay("no credentials")
        base_url = env_base_url("ELEVENLABS_STT_BASE_URL", tt.ELEVENLABS_STT_BASE_URL)
        return direct(STT_WIRE_ELEVENLABS, base_url, api_key,
                      section.get("model") or tt.DEFAULT_ELEVENLABS_STT_MODEL)

    if provider == "deepinfra":
        api_key = tt._resolve_provider_key("DEEPINFRA_API_KEY", "deepinfra")
        if not api_key:
            return _relay("no credentials")
        from hermes_cli.models import deepinfra_base_url

        model = _deepinfra_model(section, "stt")
        if not model:
            return _relay("no deepinfra stt model")
        return direct(STT_WIRE_OPENAI, deepinfra_base_url(section), api_key, model)

    return _relay(f"provider {provider!r} has no client wire")


# ---------------------------------------------------------------------------
# TTS
# ---------------------------------------------------------------------------


def _resolve_tts_client_config() -> Dict[str, Any]:
    from tools import tts_tool as tts

    tts_config = tts._load_tts_config()
    provider = tts._get_provider(tts_config)

    if provider not in tts.BUILTIN_TTS_PROVIDERS:
        return _relay("command/plugin provider")

    if provider == "openai":
        # Covers the direct-key, custom-base_url, and Nous-managed selections.
        try:
            api_key, base_url, is_managed = tts._resolve_openai_audio_client_config()
        except ValueError as exc:
            return _relay(f"openai resolution failed: {exc}")
        oai = _section(tts_config, "openai")
        model = oai.get("model") or tts.DEFAULT_OPENAI_MODEL
        config_base = oai.get("base_url")
        if config_base:
            base_url = config_base
        # The managed gateway only proxies MANAGED_OPENAI_TTS_MODELS — same
        # coercion text_to_speech applies server-side.
        if is_managed and not config_base and model not in tts.MANAGED_OPENAI_TTS_MODELS:
            model = tts.DEFAULT_OPENAI_MODEL
        speed_default = tts_config.get("speed", 1.0) if isinstance(tts_config, dict) else 1.0
        try:
            speed = float(oai.get("speed", speed_default))
        except (TypeError, ValueError):
            speed = 1.0
        return _direct(TTS_WIRE_OPENAI, "openai", base_url, api_key, model,
                       voice=oai.get("voice") or tts.DEFAULT_OPENAI_VOICE, speed=speed)

    if provider == "elevenlabs":
        api_key = tts._resolve_provider_key("ELEVENLABS_API_KEY", "elevenlabs")
        if not api_key:
            return _relay("no credentials")
        el = _section(tts_config, "elevenlabs")
        return _direct(
            TTS_WIRE_ELEVENLABS, "elevenlabs",
            str(el.get("base_url") or "https://api.elevenlabs.io/v1").rstrip("/"),
            api_key, el.get("model_id") or tts.DEFAULT_ELEVENLABS_MODEL_ID,
            voice=el.get("voice_id") or tts.DEFAULT_ELEVENLABS_VOICE_ID, speed=None,
        )

    if provider == "deepinfra":
        api_key = tts._resolve_provider_key("DEEPINFRA_API_KEY", "deepinfra")
        if not api_key:
            return _relay("no credentials")
        from hermes_cli.models import deepinfra_base_url

        di = _section(tts_config, "deepinfra")
        model = _deepinfra_model(di, "tts")
        if not model:
            return _relay("no deepinfra tts model")
        return _direct(TTS_WIRE_OPENAI, "deepinfra", deepinfra_base_url(di), api_key, model,
                       voice=di.get("voice") or "af_bella", speed=None)

    # edge / minimax / xai / mistral / gemini / neutts / kittentts / piper:
    # either server-host-only engines or wire shapes the desktop doesn't
    # speak yet. The relay path (speak-stream WS + POST fallback) serves them.
    return _relay(f"provider {provider!r} has no client wire")


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------


def resolve_client_voice_config() -> Dict[str, Any]:
    """Resolve both directions for the CURRENT profile scope.

    Callers scope the profile via ``hermes_constants.set_hermes_home_override``
    (the web server's ``_config_profile_scope``) before calling — identical to
    how ``/api/audio/transcribe`` scopes ``transcribe_recording``.
    """
    if not _client_direct_enabled():
        disabled = _relay("voice.client_direct disabled")
        return {"stt": disabled, "tts": disabled}

    out: Dict[str, Any] = {}
    for key, resolver in (("stt", _resolve_stt_client_config), ("tts", _resolve_tts_client_config)):
        try:
            out[key] = resolver()
        except Exception:
            logger.exception("client voice-config %s resolution failed", key.upper())
            out[key] = _relay("resolution error")
    return out
