#!/usr/bin/env python3
"""
Text-to-Speech Tool Module

Built-in TTS providers:
- Edge TTS (default, free, no API key): Microsoft Edge neural voices
- ElevenLabs (premium): High-quality voices, needs ELEVENLABS_API_KEY
- OpenAI TTS: Good quality, needs OPENAI_API_KEY
- MiniMax TTS: High-quality with voice cloning, needs the selected region's key
- Mistral (Voxtral TTS): Multilingual, native Opus, needs MISTRAL_API_KEY
- Google Gemini TTS: Controllable, 30 prebuilt voices, needs GEMINI_API_KEY
- xAI TTS: Grok voices, uses xAI Grok OAuth credentials or XAI_API_KEY
- NeuTTS (local, free, no API key): On-device TTS via neutts
- KittenTTS (local, free, no API key): On-device 25MB model
- Piper (local, free, no API key): OHF-Voice/piper1-gpl neural VITS, 44 languages

Custom command providers: any number of named ``type: command`` providers under
``tts.providers.<name>`` in ``~/.hermes/config.yaml``; Hermes writes the text to
a temp file and runs the shell template (see the Local Command section of
``website/docs/user-guide/features/tts.md``).

Output: Opus (.ogg) for voice-bubble platforms (Telegram etc.), MP3 elsewhere.
Configuration lives under the ``tts:`` key; the user chooses provider/voice,
the model just sends text.

Module layout: this file owns config resolution, lazy SDK importers,
built-in provider dispatch, output-path policy, the long-form tool entry point,
the ``check_fn`` and the tool registration. Sibling modules: ``tts_command_provider``
(``type: command`` providers), ``tts_tool_plugins`` (plugin-registered providers),
``tts_tool_openai`` (OpenAI/DeepInfra + managed gateway), ``tts_tool_providers``
(other cloud backends), ``tts_tool_local`` (on-device engines + model caches),
``tts_tool_lifecycle`` (warm/release leases), ``tts_tool_delivery`` (caps /
chunking / ffmpeg / packing), ``tts_tool_speaker`` (streaming speaker pipeline).
Their names are re-imported here so ``tools.tts_tool.<name>`` keeps resolving
and tests patching ``tools.tts_tool.<seam>`` still take effect (siblings
resolve those seams through ``_origin()`` at call time).

Usage:
    from tools.tts_tool import text_to_speech_tool, check_tts_requirements

    result = text_to_speech_tool(text="Hello world")
"""

import asyncio
import datetime
import importlib.util
import json
import logging
import os
import tempfile  # noqa: F401 — tests/gateway patch ``tts_tool.tempfile.NamedTemporaryFile``
from pathlib import Path
from typing import Callable, Dict, Any, List, Optional

from hermes_constants import display_hermes_home

logger = logging.getLogger(__name__)


def get_env_value(name, default=None):
    """Read env values through the live config module.

    Resolved at call time: tests monkeypatch/restore
    ``hermes_cli.config.get_env_value`` and must not leave TTS holding a stale
    function for the rest of the process.
    """
    try:
        from hermes_cli.config import get_env_value as _get_env_value
    except ImportError:
        return os.getenv(name, default)
    value = _get_env_value(name)
    return default if value is None else value


def _resolve_provider_key(env_var: str, provider_id: str) -> str:
    """Resolve a TTS provider API key via the shared voice-key resolver.

    ``tools.tool_backend_helpers.resolve_provider_secret`` is the single owner
    of STT/TTS key resolution (config > env/.env > credential pool). Resolved
    at call time so tests that reload the helpers module see the live function.
    """
    try:
        from tools.tool_backend_helpers import resolve_provider_secret
    except ImportError:  # pragma: no cover — helpers are in-repo
        return str(get_env_value(env_var) or "").strip()
    return resolve_provider_secret(env_var, provider_id, env_getter=get_env_value)


from tools.managed_tool_gateway import resolve_managed_tool_gateway  # noqa: F401 — seam patched by tests
from tools.tts_command_provider import (  # noqa: F401 — historical names re-exported
    BUILTIN_TTS_PROVIDERS,
    COMMAND_TTS_OUTPUT_FORMATS,
    DEFAULT_COMMAND_TTS_MAX_TEXT_LENGTH,
    DEFAULT_COMMAND_TTS_OUTPUT_FORMAT,
    DEFAULT_COMMAND_TTS_TIMEOUT_SECONDS,
    _configured_command_tts_output_path,
    _generate_command_tts,
    _get_command_tts_output_format,
    _get_command_tts_timeout,
    _get_named_provider_config,
    _is_command_provider_config,
    _is_command_tts_voice_compatible,
    _iter_command_providers,
    _resolve_command_provider_config,
    command_env_passthrough as _command_provider_env_passthrough,
    render_command_template as _render_command_tts_template,
    run_command_provider as _run_command_tts,
    shell_quote_context as _shell_quote_context,
)
from tools.tool_backend_helpers import (  # noqa: F401 — seams patched by tests, resolved via tts_tool_openai._origin()
    NOUS_MANAGED_PROVIDER,
    managed_nous_tools_enabled,
    read_selection,
    resolve_openai_audio_api_key,
)
from tools.tts_tool_delivery import (  # noqa: F401 — historical names re-exported
    FALLBACK_MAX_TEXT_LENGTH,
    PROVIDER_MAX_TEXT_LENGTH,
    _resolve_max_text_length,
    AudioDeliveryProfile,
    _build_audio_delivery_files,
    _concat_audio_files,
    _convert_to_opus,
    _pack_audio_files_for_delivery,
    _repair_ogg_container,
    _resolve_audio_delivery_profile,
    _sniff_audio_container,
    _split_oversized_sentence,
    _split_text_for_tts,
    _wrap_pcm_as_wav,
)
from tools.tts_tool_providers import (  # noqa: F401 — historical names re-exported
    DEFAULT_ELEVENLABS_MODEL_ID,
    DEFAULT_ELEVENLABS_VOICE_ID,
    DEFAULT_GEMINI_TTS_MODEL,
    DEFAULT_GEMINI_TTS_VOICE,
    DEFAULT_MINIMAX_BASE_URL,
    DEFAULT_MINIMAX_CN_BASE_URL,
    TTS_RESPONSE_BODY_LIMIT_BYTES,
    _XAI_FIRST_SENTENCE_RE,
    _XAI_INLINE_SPEECH_TAGS,
    _XAI_WRAPPING_SPEECH_TAGS,
    _apply_xai_auto_speech_tags,
    _elevenlabs_environment_kwargs,
    _generate_edge_tts,
    _generate_elevenlabs,
    _generate_gemini_tts,
    _generate_minimax_tts,
    _generate_mistral_tts,
    _generate_xai_tts,
    _resolve_minimax_tts_runtime,
)
from tools.tts_tool_local import (  # noqa: F401 — historical names re-exported
    DEFAULT_PIPER_VOICE,
    _LOCAL_TTS_MODEL_CACHES,
    _TTS_MODEL_CACHE_MAX,
    _generate_kittentts,
    _generate_neutts,
    _generate_piper_tts,
    _kittentts_model_cache,
    _piper_voice_cache,
    _resolve_piper_voice_path,
    _tts_cache_get_or_load,
)
from tools.tts_tool_speaker import (  # noqa: F401 — historical names re-exported
    stream_tts_to_speaker,
)
from tools.tts_text_normalize import _strip_markdown_for_tts  # noqa: F401 — historical name re-exported
from tools.tts_tool_plugins import (  # noqa: F401 — historical names re-exported
    _dispatch_to_plugin_provider,
    _plugin_provider_is_available,
    _plugin_provider_is_voice_compatible,
)
from tools.tts_tool_openai import (  # noqa: F401 — historical names re-exported
    DEFAULT_OPENAI_BASE_URL,
    DEFAULT_OPENAI_MODEL,
    DEFAULT_OPENAI_VOICE,
    MANAGED_OPENAI_TTS_MODELS,
    _generate_deepinfra_tts,
    _generate_openai_tts,
    _has_openai_audio_backend,
    _resolve_openai_audio_client_config,
)
from tools.tts_tool_lifecycle import (  # noqa: F401 — historical names re-exported
    _local_tts_warmers,
    _reset_tts_leases_for_tests,
    acquire_tts_lease,
    release_tts_lease,
    release_tts_provider,
    tts_lease_holders,
    warm_tts_provider,
)

# ---------------------------------------------------------------------------
# Lazy imports -- providers are imported only when actually used to avoid
# crashing in headless environments (SSH, Docker, WSL, no PortAudio).
# ---------------------------------------------------------------------------

def _sdk_importer(module: str, attr: Optional[str] = None, feature: Optional[str] = None) -> Callable[[], Any]:
    """Lazy SDK importer: returns ``module`` (or ``module.attr``), raising ImportError when absent.

    ``feature`` names a ``tools.lazy_deps`` feature to best-effort install
    first (users who enabled a provider by editing config.yaml never ran the
    post-setup hook); any failure there falls through so the raw import still
    raises a clean ImportError. sounddevice additionally raises OSError when
    PortAudio is unavailable.
    """
    def _import():
        if feature:
            try:
                from tools.lazy_deps import ensure
                ensure(feature, prompt=False)
            except Exception:
                pass
        mod = importlib.import_module(module)
        return getattr(mod, attr) if attr else mod
    _import.__name__ = f"_import_{module.split('.')[0]}"
    return _import


_import_edge_tts = _sdk_importer("edge_tts", feature="tts.edge")
_import_elevenlabs = _sdk_importer("elevenlabs.client", "ElevenLabs", feature="tts.elevenlabs")
_import_openai_client = _sdk_importer("openai", "OpenAI")
_import_mistral_client = _sdk_importer("mistralai.client", "Mistral", feature="tts.mistral")
_import_sounddevice = _sdk_importer("sounddevice")
_import_kittentts = _sdk_importer("kittentts", "KittenTTS")
_import_piper = _sdk_importer("piper", "PiperVoice")  # piper-tts wheels embed espeak-ng


def _package_installed(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def _check_neutts_available() -> bool:
    return _package_installed("neutts")


def _check_kittentts_available() -> bool:
    return _package_installed("kittentts")


def _check_piper_available() -> bool:
    return _package_installed("piper")


# ===========================================================================
# Defaults
# ===========================================================================
DEFAULT_PROVIDER = "edge"


def _get_default_output_dir() -> str:
    from hermes_constants import get_hermes_dir
    return str(get_hermes_dir("cache/audio", "audio_cache"))


DEFAULT_OUTPUT_DIR = _get_default_output_dir()
_DEFAULT_OUTPUT_DIR_AT_IMPORT = DEFAULT_OUTPUT_DIR


def _default_output_dir() -> str:
    """Return the active profile's audio output dir at call time.

    Long-lived multi-profile runtimes (dashboard, TUI/Desktop backend, cron)
    import this module once and later switch profiles via
    ``set_hermes_home_override()``; a frozen constant would keep writing into
    the launch profile's cache. ``DEFAULT_OUTPUT_DIR`` stays as a module
    attribute for tests/patchers and wins whenever it has been patched.
    """
    configured = DEFAULT_OUTPUT_DIR
    if configured != _DEFAULT_OUTPUT_DIR_AT_IMPORT:
        return configured
    return _get_default_output_dir()


# Back-compat alias. Prefer ``_resolve_max_text_length()`` for new code.
MAX_TEXT_LENGTH = FALLBACK_MAX_TEXT_LENGTH


# ===========================================================================
# Config loader -- reads tts: section from ~/.hermes/config.yaml
# ===========================================================================
def _load_tts_config() -> Dict[str, Any]:
    """Return the ``tts`` config section ({} when unavailable)."""
    try:
        from hermes_cli.config import load_config
        config = load_config()
        return config.get("tts") or {}
    except ImportError:
        logger.debug("hermes_cli.config not available, using default TTS config")
        return {}
    except Exception as e:
        logger.warning("Failed to load TTS config: %s", e, exc_info=True)
        return {}


def _get_provider(tts_config: Dict[str, Any]) -> str:
    """The explicitly configured TTS provider, or the free default.

    Inference credentials do not imply consent to paid speech generation:
    cloud TTS is opt-in via ``tts.provider``. The managed selection
    (``tts.provider: nous``) is serviced by the OpenAI implementation, routed
    through the managed openai-audio gateway by
    ``_resolve_openai_audio_client_config``.
    """
    provider = (tts_config.get("provider") or DEFAULT_PROVIDER).lower().strip()
    if provider == NOUS_MANAGED_PROVIDER:
        return "openai"
    return provider


# Platforms whose native voice-bubble delivery requires Ogg/Opus audio
# (MP3 renders as a broken attachment there).
OPUS_VOICE_PLATFORMS = frozenset({"telegram", "matrix", "feishu", "whatsapp", "signal"})

# Built-ins that emit Opus natively when asked for .ogg (no ffmpeg needed).
_NATIVE_OPUS_PROVIDERS = frozenset({"openai", "elevenlabs", "mistral", "gemini"})
# Built-ins whose native output (MP3/WAV) needs ffmpeg for voice-bubble delivery.
_FFMPEG_OPUS_PROVIDERS = frozenset({"edge", "neutts", "minimax", "xai", "kittentts", "piper"})


def _has_any_command_tts_provider(tts_config: Optional[Dict[str, Any]] = None) -> bool:
    """Return True when any command-type TTS provider is configured."""
    if tts_config is None:
        tts_config = _load_tts_config()
    for _name, _cfg in _iter_command_providers(tts_config):
        return True
    return False


# ===========================================================================
# Built-in provider dispatch
# ===========================================================================
# provider -> (importer-name or None, "package missing" error, log line,
# generator-name). Names are looked up in module globals at call time so
# tests that monkeypatch ``tools.tts_tool._import_x`` / ``_generate_x`` apply.
_BUILTIN_DISPATCH: Dict[str, tuple] = {
    "elevenlabs": (
        "_import_elevenlabs",
        "ElevenLabs provider selected but 'elevenlabs' package not installed. Run: pip install elevenlabs",
        "Generating speech with ElevenLabs...",
        "_generate_elevenlabs",
    ),
    "openai": (
        "_import_openai_client",
        "OpenAI provider selected but 'openai' package not installed.",
        "Generating speech with OpenAI TTS...",
        "_generate_openai_tts",
    ),
    "deepinfra": (
        "_import_openai_client",
        "DeepInfra TTS uses the 'openai' SDK but it isn't installed.",
        "Generating speech with DeepInfra TTS...",
        "_generate_deepinfra_tts",
    ),
    "minimax": (None, None, "Generating speech with MiniMax TTS...", "_generate_minimax_tts"),
    "xai": (None, None, "Generating speech with xAI TTS...", "_generate_xai_tts"),
    "mistral": (
        "_import_mistral_client",
        "Mistral provider selected but 'mistralai' package not installed. "
        "Run `hermes setup` to install Mistral support.",
        "Generating speech with Mistral Voxtral TTS...",
        "_generate_mistral_tts",
    ),
    "gemini": (None, None, "Generating speech with Google Gemini TTS...", "_generate_gemini_tts"),
    "kittentts": (
        "_import_kittentts",
        "KittenTTS provider selected but 'kittentts' package not installed. "
        "Run 'hermes setup tts' and choose KittenTTS, or install manually: "
        "pip install https://github.com/KittenML/KittenTTS/releases/download/0.8.1/kittentts-0.8.1-py3-none-any.whl",
        "Generating speech with KittenTTS (local, ~25MB)...",
        "_generate_kittentts",
    ),
    "piper": (
        "_import_piper",
        "Piper provider selected but 'piper-tts' package not installed. "
        "Run 'hermes tools' and select Piper under TTS, or install manually: "
        "pip install piper-tts",
        "Generating speech with Piper (local)...",
        "_generate_piper_tts",
    ),
}
_NEUTTS_MISSING_ERROR = (
    "NeuTTS provider selected but neutts is not installed. "
    "Run hermes setup and choose NeuTTS, or install espeak-ng and run python -m pip install -U neutts[all]."
)


def _error_json(message: str) -> str:
    return json.dumps({"success": False, "error": message}, ensure_ascii=False)


def _run_edge_tts(text: str, file_str: str, tts_config: Dict[str, Any]) -> None:
    """Run the async Edge generator from sync code (worker thread; direct run if that fails)."""
    try:
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(
                lambda: asyncio.run(_generate_edge_tts(text, file_str, tts_config))
            ).result(timeout=60)
    except RuntimeError:
        asyncio.run(_generate_edge_tts(text, file_str, tts_config))


def _select_builtin_engine(provider: str) -> tuple:
    """Check a built-in provider's SDK. Returns ``(engine, None)`` or ``(provider, error_json)``.

    Unknown names take the Edge default; when edge-tts is missing, NeuTTS is
    the local fallback (``engine`` then differs from ``provider``).
    """
    entry = _BUILTIN_DISPATCH.get(provider)
    if entry is not None:
        importer_name, missing_error = entry[0], entry[1]
        if importer_name is not None and not _importable(globals()[importer_name]):
            return provider, _error_json(missing_error)
        return provider, None
    if provider == "neutts":
        if not _check_neutts_available():
            return provider, _error_json(_NEUTTS_MISSING_ERROR)
        logger.info("Generating speech with NeuTTS (local)...")
        return provider, None
    if _importable(_import_edge_tts):
        return provider, None  # Edge default; the reported provider stays as configured
    if _check_neutts_available():
        logger.info("Edge TTS not available, falling back to NeuTTS (local)...")
        return "neutts", None
    return provider, _error_json(
        "No TTS provider available. Install edge-tts (pip install edge-tts) "
        "or set up NeuTTS for local synthesis."
    )


def _synthesize_builtin(engine: str, text: str, file_str: str, tts_config: Dict[str, Any], instructions: Optional[str]) -> None:
    """Run the already-selected built-in *engine* (the caller logs the engine-selection line)."""
    entry = _BUILTIN_DISPATCH.get(engine)
    if entry is not None:
        logger.info(entry[2])
        if engine == "openai":
            _generate_openai_tts(text, file_str, tts_config, instructions=instructions)
        else:
            globals()[entry[3]](text, file_str, tts_config)
    elif engine == "neutts":
        _generate_neutts(text, file_str, tts_config)
    else:
        logger.info("Generating speech with Edge TTS...")
        _run_edge_tts(text, file_str, tts_config)


def _finalize_voice_delivery(
    file_str: str,
    provider: str,
    command_provider_config: Optional[Dict[str, Any]],
    want_opus: bool,
) -> tuple:
    """Decide voice-bubble eligibility and Opus-convert when needed.

    Command and plugin providers are documents by default and opt in via
    ``voice_compatible``; native-Opus built-ins are voice-compatible when the
    platform wants Opus and they wrote .ogg; MP3/WAV built-ins are converted
    with ffmpeg only when the platform needs Opus. Returns ``(path, voice_compatible)``.
    """
    voice_compatible = False
    if command_provider_config is not None:
        opted_in = _is_command_tts_voice_compatible(command_provider_config)
    elif provider not in BUILTIN_TTS_PROVIDERS:
        opted_in = _plugin_provider_is_voice_compatible(provider)
    elif want_opus and provider in _FFMPEG_OPUS_PROVIDERS and not file_str.endswith(".ogg"):
        opus_path = _convert_to_opus(file_str)
        if opus_path:
            return opus_path, True
        return file_str, False
    elif provider in _NATIVE_OPUS_PROVIDERS:
        return file_str, want_opus and file_str.endswith(".ogg")
    else:
        return file_str, False

    if opted_in:
        if not file_str.endswith(".ogg"):
            opus_path = _convert_to_opus(file_str)
            if opus_path:
                file_str = opus_path
        voice_compatible = file_str.endswith(".ogg")
    return file_str, voice_compatible


# ===========================================================================
# Main tool function
# ===========================================================================

def _apply_call_overrides(tts_config: Dict[str, Any], speed: Optional[float], provider: Optional[str]):
    """Apply per-call ``speed`` (clamped, on a shallow copy) and resolve the provider name."""
    if speed is not None:
        clamped = max(0.25, min(4.0, float(speed)))
        tts_config = dict(tts_config)  # shallow copy to avoid mutating the cache
        tts_config["speed"] = clamped
    provider = provider.lower().strip() if provider else _get_provider(tts_config)
    return tts_config, provider


def _session_platform() -> tuple:
    """``(platform, wants_opus)`` — platforms delivering voice bubbles only as Ogg/Opus want Opus."""
    from gateway.session_context import get_session_env
    platform = get_session_env("HERMES_SESSION_PLATFORM", "").lower()
    return platform, platform in OPUS_VOICE_PLATFORMS


def _resolve_output_base(
    output_path: Optional[str],
    provider: str,
    command_provider_config: Optional[Dict[str, Any]],
    want_opus: bool,
) -> tuple:
    """Pick the output file. Returns ``(Path, None)`` or ``(None, error_json)``.

    A caller-supplied path is rejected on ``..`` traversal (bug or
    prompt-injection; an absolute path is fine) and on protected credential/
    system locations. Command providers get their configured extension.
    Default: ``<audio cache>/tts_<timestamp>.<ext>`` where ext is the command
    format, ``.ogg`` for native-Opus providers on Opus platforms, else ``.mp3``.
    """
    if output_path:
        from tools.path_security import has_traversal_component
        if has_traversal_component(output_path):
            return None, _error_json(
                f"output_path contains '..' traversal component: {output_path}. "
                "Use an absolute path or one relative to the current directory "
                "without '..'."
            )
        file_path = Path(output_path).expanduser()
        if command_provider_config is not None:
            file_path = _configured_command_tts_output_path(file_path, command_provider_config)
        from agent.file_safety import is_write_approval_required, is_write_denied
        if is_write_denied(str(file_path)) or is_write_approval_required(str(file_path)):
            return None, _error_json(
                f"output_path targets a protected credential or system path: "
                f"{file_path}. Choose a normal audio output location."
            )
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        out_dir = Path(_default_output_dir())
        out_dir.mkdir(parents=True, exist_ok=True)
        if command_provider_config is not None:
            ext = _get_command_tts_output_format(command_provider_config)
        elif want_opus and provider in _NATIVE_OPUS_PROVIDERS:
            ext = "ogg"
        else:
            ext = "mp3"
        file_path = out_dir / f"tts_{timestamp}.{ext}"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    return file_path, None


def _tool_failure(prefix: str, provider: str, exc: BaseException) -> str:
    """Log and wrap a synthesis failure as the standard error envelope (traceback except for config errors)."""
    error_msg = f"{prefix} ({provider}): {exc}"
    logger.error("%s", error_msg, exc_info=not isinstance(exc, ValueError))
    return tool_error(error_msg, success=False)


def _text_to_speech_single(
    text: str,
    file_str: str,
    *,
    provider: str,
    tts_config: Dict[str, Any],
    command_provider_config: Optional[Dict[str, Any]],
    want_opus: bool,
    instructions: Optional[str],
) -> str:
    """Synthesize one provider-safe text chunk into *file_str* and return one final-encoded file.

    Text arrives already normalized and the output path already validated;
    :func:`text_to_speech_tool` owns long-form splitting, delivery packing and
    size enforcement. Command providers resolve BEFORE built-in dispatch;
    built-in names short-circuit so ``tts.providers.openai.command`` can't
    shadow OpenAI. Plugin providers fire only for names that are neither
    built-in nor command; a None return falls through to built-in dispatch
    (unknown -> Edge default).
    """
    try:
        if command_provider_config is not None:
            logger.info("Generating speech with command TTS provider '%s'...", provider)
            file_str = _generate_command_tts(text, file_str, provider, command_provider_config, tts_config)
        elif provider not in BUILTIN_TTS_PROVIDERS and (
            _plugin_path := _dispatch_to_plugin_provider(text, file_str, provider, tts_config)
        ) is not None:
            file_str = _plugin_path
        else:
            provider, error = _select_builtin_engine(provider)
            if error:
                return error
            _synthesize_builtin(provider, text, file_str, tts_config, instructions)

        if not os.path.exists(file_str) or os.path.getsize(file_str) == 0:
            return _error_json(f"TTS generation produced no output (provider: {provider})")

        # Sniff once for every provider: MP3/WAV bytes in a .ogg path render
        # as broken 0-second voice bubbles.
        file_str = _repair_ogg_container(file_str)
        file_str, voice_compatible = _finalize_voice_delivery(
            file_str, provider, command_provider_config, want_opus,
        )

        file_size = os.path.getsize(file_str)
        logger.info("TTS audio saved: %s (%s bytes, provider: %s)", file_str, f"{file_size:,}", provider)

        media_tag = f"MEDIA:{file_str}"
        if voice_compatible:
            media_tag = f"[[audio_as_voice]]\n{media_tag}"

        return json.dumps({
            "success": True,
            "file_path": file_str,
            "media_tag": media_tag,
            "provider": provider,
            "voice_compatible": voice_compatible,
        }, ensure_ascii=False)

    except ValueError as e:
        return _tool_failure("TTS configuration error", provider, e)
    except FileNotFoundError as e:
        return _tool_failure("TTS dependency missing", provider, e)
    except Exception as e:
        return _tool_failure("TTS generation failed", provider, e)


class _ChunkFailed(Exception):
    """One chunk's synthesis returned an error envelope; message is the final tool error text."""


def _synthesize_chunks(
    chunks: List[str],
    base_path: Path,
    generated_artifacts: set,
    *,
    provider: str,
    tts_config: Dict[str, Any],
    command_provider_config: Optional[Dict[str, Any]],
    want_opus: bool,
    instructions: Optional[str],
) -> tuple:
    """Synthesize every chunk into ``<base>.chunkNNN<ext>`` (or ``base`` alone) sequentially.

    Every path touched is added to *generated_artifacts* so the caller can
    sweep non-final files. Returns ``(encoded_paths, chunk_results)``; raises
    :class:`_ChunkFailed` when a chunk reports failure and ``RuntimeError``
    when it returns garbage or no audio.
    """
    encoded_paths: List[str] = []
    chunk_results: List[Dict[str, Any]] = []
    for index, chunk in enumerate(chunks, start=1):
        if len(chunks) == 1:
            chunk_path = base_path
        else:
            chunk_path = base_path.with_name(
                f"{base_path.stem}.chunk{index:03d}{base_path.suffix}"
            )
        generated_artifacts.add(str(chunk_path))
        raw_result = _text_to_speech_single(
            chunk,
            str(chunk_path),
            provider=provider,
            tts_config=tts_config,
            command_provider_config=command_provider_config,
            want_opus=want_opus,
            instructions=instructions,
        )
        try:
            chunk_result = json.loads(raw_result)
        except (json.JSONDecodeError, TypeError):
            raise RuntimeError(
                f"TTS chunk {index} returned invalid JSON: {str(raw_result)[:200]}"
            )
        if not chunk_result.get("success"):
            error_msg = chunk_result.get("error", "unknown error")
            raise _ChunkFailed(f"TTS chunk {index} failed ({provider}): {error_msg}")
        actual_path = str(chunk_result.get("file_path") or chunk_path)
        if not os.path.isfile(actual_path) or os.path.getsize(actual_path) <= 0:
            raise RuntimeError(
                f"TTS chunk {index} produced no final audio: {actual_path}"
            )
        generated_artifacts.add(actual_path)
        encoded_paths.append(actual_path)
        chunk_results.append(chunk_result)
    return encoded_paths, chunk_results


def text_to_speech_tool(
    text: str,
    output_path: Optional[str] = None,
    speed: Optional[float] = None,
    instructions: Optional[str] = None,
    provider: Optional[str] = None,
) -> str:
    """Convert text to speech audio with long-form chunking.

    Text is normalized, split into provider-safe chunks, synthesized
    sequentially (each chunk final-encoded), then packed against the
    destination platform's upload limit. Multi-chunk voice output is
    re-encoded when combined; a failed combine keeps separate valid files;
    no over-limit artifact is returned. On messaging platforms the
    ``MEDIA:<path>`` tag is delivered as a native voice message.

    Args:
        text: Text to speak; longer input is split into ordered chunks, never
            silently truncated.
        output_path: Optional custom save path.
        speed: Optional playback speed multiplier (0.25-4.0).
        instructions: Optional voice-design guidance (tone, emotion, pacing).
        provider: Optional TTS provider override.

    Returns:
        str: JSON result with success, file_path, file_paths, and MEDIA tag.
    """
    if not text or not text.strip():
        return tool_error("Text is required", success=False)

    # Shared cleaner: markdown, emoji, think blocks, verifier footer, units, newlines.
    try:
        from tools.tts_text_normalize import prepare_spoken_text
        text = prepare_spoken_text(text, max_chars=None)
    except Exception:
        text = text.strip()
    if not text:
        return tool_error("Text is empty after TTS cleanup", success=False)

    tts_config, provider = _apply_call_overrides(_load_tts_config(), speed, provider)

    command_provider_config = _resolve_command_provider_config(provider, tts_config)
    max_len = _resolve_max_text_length(provider, tts_config)
    chunks = _split_text_for_tts(text, max_len)
    if not chunks:
        return tool_error("Text is required", success=False)
    if len(chunks) > 1:
        logger.info(
            "TTS text for provider %s split into %d chunks (input=%d chars, cap=%d)",
            provider,
            len(chunks),
            len(text),
            max_len,
        )

    platform, want_opus = _session_platform()
    delivery_profile = _resolve_audio_delivery_profile(platform, tts_config)

    base_path, error = _resolve_output_base(output_path, provider, command_provider_config, want_opus)
    if error:
        return error

    generated_artifacts: set[str] = set()
    final_paths: List[str] = []
    try:
        encoded_paths, chunk_results = _synthesize_chunks(
            chunks, base_path, generated_artifacts,
            provider=provider,
            tts_config=tts_config,
            command_provider_config=command_provider_config,
            want_opus=want_opus,
            instructions=instructions,
        )

        voice_compatible = bool(chunk_results) and all(
            bool(result.get("voice_compatible")) for result in chunk_results
        )
        delivery_base = base_path.with_suffix(Path(encoded_paths[0]).suffix)
        final_paths, combined_chunks = _build_audio_delivery_files(
            encoded_paths,
            str(delivery_base),
            delivery_profile,
            voice_compatible=voice_compatible,
        )

        for path in final_paths:
            logger.info(
                "TTS audio saved: %s (%s bytes, provider: %s)",
                path,
                f"{os.path.getsize(path):,}",
                provider,
            )
        media_tag = "\n".join(f"MEDIA:{path}" for path in final_paths)
        if voice_compatible:
            media_tag = f"[[audio_as_voice]]\n{media_tag}"

        return json.dumps({
            "success": True,
            "file_path": final_paths[0],
            "file_paths": final_paths,
            "media_tag": media_tag,
            "provider": chunk_results[0].get("provider", provider),
            "voice_compatible": voice_compatible,
            "chunk_count": len(chunks),
            "delivery_file_count": len(final_paths),
            "combined_chunks": bool(combined_chunks),
            "delivery_profile": {
                "platform": delivery_profile.platform,
                "max_file_bytes": delivery_profile.max_file_bytes,
                "target_file_bytes": delivery_profile.target_file_bytes,
            },
        }, ensure_ascii=False)
    except _ChunkFailed as exc:
        return tool_error(str(exc), success=False)
    except ValueError as exc:
        return _tool_failure("TTS delivery error", provider, exc)
    except Exception as exc:
        return _tool_failure("TTS long-form generation failed", provider, exc)
    finally:
        final_absolute = {os.path.abspath(path) for path in final_paths}
        for artifact in generated_artifacts:
            if os.path.abspath(artifact) in final_absolute:
                continue
            try:
                os.unlink(artifact)
            except OSError:
                pass


def _importable(importer: Callable[[], Any]) -> bool:
    try:
        importer()
        return True
    except ImportError:
        return False


def _minimax_requirements() -> bool:
    try:
        _resolve_minimax_tts_runtime(_load_tts_config())
    except ValueError:
        return False
    return True


def _xai_requirements() -> bool:
    try:
        from tools.xai_http import resolve_xai_http_credentials

        return bool(resolve_xai_http_credentials().get("api_key"))
    except Exception:
        return False


# Must mirror text_to_speech_tool dispatch: unrelated cloud credentials never
# make the Edge default usable, and an explicit provider is checked on its own.
_BUILTIN_REQUIREMENTS: Dict[str, Callable[[], bool]] = {
    "edge": lambda: _importable(_import_edge_tts) or _check_neutts_available(),
    "elevenlabs": lambda: _importable(_import_elevenlabs) and bool(_resolve_provider_key("ELEVENLABS_API_KEY", "elevenlabs")),
    "openai": lambda: _package_installed("openai") and _has_openai_audio_backend(),
    "deepinfra": lambda: _package_installed("openai") and bool(_resolve_provider_key("DEEPINFRA_API_KEY", "deepinfra")),
    "minimax": _minimax_requirements,
    "xai": _xai_requirements,
    "gemini": lambda: bool(_resolve_provider_key("GEMINI_API_KEY", "gemini") or _resolve_provider_key("GOOGLE_API_KEY", "gemini")),
    "mistral": lambda: _importable(_import_mistral_client) and bool(_resolve_provider_key("MISTRAL_API_KEY", "mistral")),
    "neutts": lambda: _check_neutts_available(),
    "kittentts": lambda: _check_kittentts_available(),
    "piper": lambda: _check_piper_available(),
}


def check_tts_requirements() -> bool:
    """Return whether the explicitly resolved TTS provider can run."""
    tts_config = _load_tts_config()
    provider = _get_provider(tts_config)
    if _resolve_command_provider_config(provider, tts_config) is not None:
        return True

    check = _BUILTIN_REQUIREMENTS.get(provider)
    if check is not None:
        return check()

    return _plugin_provider_is_available(provider)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
from tools.registry import registry, tool_error

TTS_SCHEMA = {
    "name": "text_to_speech",
    "description": "Convert text to speech audio. Returns a MEDIA: path that the platform delivers as native audio. Compatible providers render as a voice bubble on Telegram; otherwise audio is sent as a regular attachment. In CLI mode, saves to ~/voice-memos/. Voice and provider are user-configured (built-in providers like edge/openai or custom command providers under tts.providers.<name>), not model-selected.",
    "parameters": {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "The text to convert to speech. Provider-specific per-request character caps apply automatically (OpenAI 4096, xAI 15000, MiniMax 10000, ElevenLabs 5k-40k depending on model); longer input is split into ordered chunks without silent truncation."
            },
            "output_path": {
                "type": "string",
                "description": f"Optional custom file path to save the audio. Defaults to {display_hermes_home()}/audio_cache/<timestamp>.mp3"
            },
            "speed": {
                "type": "number",
                "description": "Playback speed multiplier. 1.0 = normal, 0.5 = very slow (language learning), 2.0 = fast. Range: 0.25-4.0. Overrides the speed configured in config.yaml."
            },
            "instructions": {
                "type": "string",
                "description": (
                    "Optional voice-design guidance: tone, emotion, pacing, accent, "
                    "whispering, impressions (e.g. 'Speak in a cheerful, excited whisper'). "
                    "Forwarded to the OpenAI backend (gpt-4o-mini-tts and OpenAI-compatible "
                    "voice-design servers). Silently ignored by backends that don't support it."
                )
            },
            "provider": {
                "type": "string",
                "description": (
                    "Optional TTS provider override. Accepts built-in names "
                    "(edge, openai, elevenlabs, minimax, xai, mistral, gemini, "
                    "neutts, kittentts, piper), user-declared command provider "
                    "names from tts.providers.<name>, or plugin-registered names. "
                    "When omitted, the configured tts.provider from config.yaml is used."
                )
            }
        },
        "required": ["text"]
    }
}

registry.register(
    name="text_to_speech",
    toolset="tts",
    schema=TTS_SCHEMA,
    handler=lambda args, **kw: text_to_speech_tool(
        text=args.get("text", ""),
        output_path=args.get("output_path"),
        speed=args.get("speed"),
        instructions=args.get("instructions"),
        provider=args.get("provider")),
    check_fn=check_tts_requirements,
    emoji="🔊",
)
