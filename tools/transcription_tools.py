#!/usr/bin/env python3
"""Speech-to-text transcription used by the gateway for voice messages.

Built-in providers: local (faster-whisper, default/free), local_command, groq,
openai (also serves the managed ``nous`` selection), mistral, xai, elevenlabs,
deepinfra; plus user-declared command providers and plugin providers.

    result = transcribe_audio("/path/to/audio.ogg")   # {"success", "transcript", "error"?, "provider"?}
"""

import logging
import os
import platform
import re
import shlex
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional, Dict, Any
from urllib.parse import urljoin

from hermes_cli._subprocess_compat import windows_hide_flags
from utils import is_truthy_value
from tools.managed_tool_gateway import resolve_managed_tool_gateway
from tools.tts_command_provider import (  # noqa: F401 — aliases are patched by tests
    command_env_passthrough as _command_stt_env_passthrough,
    quote_command_placeholder as _quote_command_stt_placeholder,
    render_command_template as _render_command_stt_template,
    run_command_provider as _run_command_stt,
    shell_quote_context as _shell_quote_context_stt,
    terminate_command_process_tree as _terminate_command_stt_process_tree,
)
from tools.tool_backend_helpers import (
    managed_nous_tools_enabled,
    nous_tool_gateway_unavailable_message,
    resolve_openai_audio_api_key,
)

from tools.transcription_common import (  # noqa: F401  (re-exported; tests patch tools.transcription_tools.<name>)
    BUILTIN_STT_PROVIDERS,
    CLOUD_STT_PROVIDERS,
    COMMON_LOCAL_BIN_DIRS,
    DEFAULT_ELEVENLABS_STT_MODEL,
    DEFAULT_GROQ_STT_MODEL,
    DEFAULT_LOCAL_MODEL,
    DEFAULT_LOCAL_STT_LANGUAGE,
    DEFAULT_MISTRAL_STT_MODEL,
    DEFAULT_PROVIDER,
    DEFAULT_STT_MODEL,
    ELEVENLABS_STT_BASE_URL,
    GROQ_BASE_URL,
    GROQ_MODELS,
    LOCAL_NATIVE_AUDIO_FORMATS,
    LOCAL_STT_COMMAND_ENV,
    LOCAL_STT_LANGUAGE_ENV,
    MAX_FILE_SIZE,
    OPENAI_BASE_URL,
    OPENAI_MODELS,
    SUPPORTED_FORMATS,
    XAI_STT_BASE_URL,
    _config_number,
    _error_result,
    _get_stt_section,
    _log_prompt_unsupported,
    _ok_result,
)

from tools.transcription_audio import (  # noqa: F401  (re-exported; tests patch tools.transcription_tools.<name>)
    _CLOUD_TRIM_KEEP_MS_DEFAULT,
    _CLOUD_TRIM_MIN_INPUT_SECONDS,
    _CLOUD_TRIM_MIN_RESULT_SECONDS,
    _CLOUD_TRIM_MIN_SAVING,
    _CLOUD_TRIM_THRESHOLD_DB_DEFAULT,
    _STT_M4A_ENCODE_ARGS,
    _cloud_trim_settings,
    _convert_caf_to_wav,
    _find_binary,
    _find_ffmpeg_binary,
    _find_ffprobe_binary,
    _find_whisper_binary,
    _prepare_audio_for_transcription,
    _prepare_local_audio,
    _probe_audio_duration,
    _run_ffmpeg_stt_encode,
    _run_quiet,
    _transcode_audio_for_stt,
    _trim_silence_for_cloud_stt,
    _validate_audio_file,
    _validate_audio_file_size,
    _validate_audio_source_file,
)

from tools.transcription_local import (  # noqa: F401  (re-exported; tests patch tools.transcription_tools.<name>)
    _CUDA_LIB_ERROR_MARKERS,
    _LOGPROB_THRESHOLD_DEFAULT,
    _NO_SPEECH_PROB_THRESHOLD_DEFAULT,
    _VAD_MIN_SILENCE_MS_DEFAULT,
    _confidence_thresholds,
    _get_idle_unload_seconds,
    _get_local_command_template,
    _has_local_command,
    _is_hallucinated_segment,
    _join_confident_segments,
    _load_local_whisper_model,
    _looks_like_cuda_lib_error,
    _normalize_local_command_model,
    _normalize_local_model,
    _should_force_faster_whisper_cpu,
    _sysctl_value,
    _transcribe_local_command,
    _try_lazy_install_stt,
    build_local_transcribe_kwargs,
)

from tools.transcription_cloud import (  # noqa: F401  (re-exported; tests patch tools.transcription_tools.<name>)
    _close_client,
    _direct_openai_credentials,
    _elevenlabs_error_detail,
    _extract_transcript_text,
    _has_xai_stt_credentials,
    _http_error_detail,
    _is_local_or_private_url,
    _openai_sdk_failure,
    _post_audio_multipart,
    _resolve_openai_audio_client_config,
    _transcribe_deepinfra,
    _transcribe_elevenlabs,
    _transcribe_groq,
    _transcribe_mistral,
    _transcribe_openai,
    _transcribe_xai,
)

from tools.transcription_command import (  # noqa: F401  (re-exported; tests patch tools.transcription_tools.<name>)
    COMMAND_STT_OUTPUT_FORMATS,
    DEFAULT_COMMAND_STT_LANGUAGE,
    DEFAULT_COMMAND_STT_OUTPUT_FORMAT,
    DEFAULT_COMMAND_STT_TIMEOUT_SECONDS,
    _PRE_TRANSCRIPTION_MUTABLE_FIELDS,
    _PROMPT_CHARS_PER_TOKEN,
    _WHISPER_PROMPT_CAPPED_PROVIDERS,
    _WHISPER_PROMPT_TOKEN_CAP,
    _apply_pre_transcription_hook,
    _dispatch_to_plugin_provider,
    _enforce_prompt_length_limit,
    _get_command_stt_output_format,
    _get_command_stt_timeout,
    _get_named_stt_provider_config,
    _is_command_stt_provider_config,
    _read_command_stt_output,
    _resolve_command_stt_provider_config,
    _transcribe_command_stt,
    _unregistered_stt_provider_error,
)

logger = logging.getLogger(__name__)


def get_env_value(name, default=None):
    """Read env values through the live config module.

    Resolved at call time: tests monkeypatch/restore ``hermes_cli.config.get_env_value``
    around this module's import, so a cached import would go stale.
    """
    try:
        from hermes_cli.config import get_env_value as _get_env_value
    except ImportError:
        return os.getenv(name, default)
    value = _get_env_value(name)
    return default if value is None else value


def _resolve_provider_key(env_var: str, provider_id: str) -> str:
    """Resolve an STT API key via the shared voice-key resolver (config > env/.env > credential pool).

    Resolved at call time so tests that reload the helpers module see the live function.
    """
    try:
        from tools.tool_backend_helpers import resolve_provider_secret
    except ImportError:  # pragma: no cover — helpers are in-repo
        return str(get_env_value(env_var) or "").strip()
    return resolve_provider_secret(env_var, provider_id, env_getter=get_env_value)


# ---------------------------------------------------------------------------
# Optional imports — graceful degradation
# ---------------------------------------------------------------------------

import importlib.util as _ilu


def _safe_find_spec(module_name: str) -> bool:
    try:
        return _ilu.find_spec(module_name) is not None
    except (ImportError, ValueError):
        return module_name in globals() or module_name in os.sys.modules


_HAS_FASTER_WHISPER = _safe_find_spec("faster_whisper")
_HAS_OPENAI = _safe_find_spec("openai")
_HAS_MISTRAL = _safe_find_spec("mistralai")
_HAS_PILK = _safe_find_spec("pilk")


# Singleton for the local model — loaded once, reused across calls. The lock
# guards the check-then-load so two concurrent voice messages can't both
# download/load the model.
_local_model: Optional[object] = None
_local_model_name: Optional[str] = None
_local_model_lock = threading.Lock()

# Idle unload: a single daemon thread checks _last_transcription_time and
# releases the model (hundreds of MB of RAM/VRAM) after a configurable idle
# period, then exits; the next voice message reloads and restarts it.
# _idle_unload_mgmt_lock serializes the start check so two concurrent
# transcriptions can't both observe "no watcher alive" and spawn duplicates.
_last_transcription_time: float = 0.0
_idle_unload_thread: Optional[threading.Thread] = None
_idle_unload_stop = threading.Event()
_idle_unload_mgmt_lock = threading.Lock()

_IDLE_UNLOAD_CHECK_INTERVAL = 30  # seconds between idle checks


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _load_stt_config() -> dict:
    """Load the ``stt`` section from user config, falling back to defaults."""
    try:
        from hermes_cli.config import load_config
        return load_config().get("stt") or {}
    except Exception:
        return {}


def is_stt_enabled(stt_config: Optional[dict] = None) -> bool:
    """Return whether STT is enabled in config."""
    if stt_config is None:
        stt_config = _load_stt_config()
    return is_truthy_value(stt_config.get("enabled", True), default=True)


def _resolve_stt_language(
    provider_key: str,
    stt_config: Optional[Dict[str, Any]] = None,
    *,
    extra_keys: tuple = (),
) -> Optional[str]:
    """Resolve the language hint for an STT provider; first non-empty wins.

    Order: ``stt.<provider>.language`` (plus *extra_keys* aliases, e.g. ElevenLabs'
    ``language_code``) > ``stt.language`` > ``HERMES_LOCAL_STT_LANGUAGE`` env >
    None (provider auto-detects). Never returns "".
    """
    if stt_config is None:
        stt_config = _load_stt_config()
    provider_cfg = _get_stt_section(stt_config, provider_key)
    candidates = [provider_cfg.get(key) for key in ("language", *extra_keys)]
    if isinstance(stt_config, dict):
        candidates.append(stt_config.get("language"))
    candidates.append(os.getenv(LOCAL_STT_LANGUAGE_ENV))
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return None


def _has_openai_audio_backend() -> bool:
    """Return True when OpenAI audio can use config credentials, env credentials, or the managed gateway."""
    try:
        _resolve_openai_audio_client_config()
        return True
    except ValueError:
        return False


def _is_local_stt_provider(provider: str, stt_config: Dict[str, Any]) -> bool:
    """Return whether *provider* is exempt from Hermes's remote upload cap."""
    return (provider or "").lower().strip() in {"local", "local_command"}


# ---------------------------------------------------------------------------
# Provider resolution
# ---------------------------------------------------------------------------


def _has_xai_stt_credentials_quietly() -> bool:
    try:
        return _has_xai_stt_credentials()
    except Exception:
        return False


def _has_key(env_var: str, provider: str, *, needs_openai: bool = False, needs_mistral: bool = False):
    """Availability probe factory: optional SDK flag AND a resolvable API key."""
    def probe() -> bool:
        if needs_openai and not _HAS_OPENAI:
            return False
        if needs_mistral and not _HAS_MISTRAL:
            return False
        return bool(_resolve_provider_key(env_var, provider))
    return probe


_has_groq_key = _has_key("GROQ_API_KEY", "groq", needs_openai=True)
_has_mistral_key = _has_key("MISTRAL_API_KEY", "mistral", needs_mistral=True)
_has_elevenlabs_key = _has_key("ELEVENLABS_API_KEY", "elevenlabs")
_has_deepinfra_key = _has_key("DEEPINFRA_API_KEY", "deepinfra", needs_openai=True)


def _resolve_explicit_openai() -> str:
    if not _HAS_OPENAI:
        logger.warning("STT provider 'openai' configured but no API key available")
        return "none"
    # Resolve directly rather than via the boolean probe so a managed
    # openai-audio gateway outage is logged with its real reason, not a
    # generic "no API key" hint.
    try:
        _resolve_openai_audio_client_config()
        return "openai"
    except ValueError as exc:
        logger.warning("STT provider 'openai' configured but unavailable: %s", exc)
        return "none"


def _resolve_explicit_local() -> str:
    if _HAS_FASTER_WHISPER:
        return "local"
    if _has_local_command():
        return "local_command"
    if _try_lazy_install_stt():
        return "local"
    logger.warning(
        "STT provider 'local' configured but unavailable "
        "(install faster-whisper or set HERMES_LOCAL_STT_COMMAND)"
    )
    return "none"


def _resolve_explicit_local_command() -> str:
    if _has_local_command():
        return "local_command"
    if _HAS_FASTER_WHISPER:
        logger.info("Local STT command unavailable, using local faster-whisper")
        return "local"
    logger.warning("STT provider 'local_command' configured but unavailable")
    return "none"


def _explicit_cloud_resolver(name: str, probe, warning: str):
    def resolve() -> str:
        if probe():
            return name
        logger.warning(warning)
        return "none"
    return resolve


# Explicit ``stt.provider`` selections -> resolver returning the provider or "none".
_EXPLICIT_PROVIDER_RESOLVERS = {
    "local": _resolve_explicit_local,
    "local_command": _resolve_explicit_local_command,
    "openai": _resolve_explicit_openai,
    "groq": _explicit_cloud_resolver(
        "groq", _has_groq_key, "STT provider 'groq' configured but GROQ_API_KEY not set"),
    "mistral": _explicit_cloud_resolver(
        "mistral", _has_mistral_key,
        "STT provider 'mistral' configured but mistralai package "
        "not installed or MISTRAL_API_KEY not set"),
    "xai": _explicit_cloud_resolver(
        "xai", _has_xai_stt_credentials, "STT provider 'xai' configured but no xAI credentials are available"),
    "elevenlabs": _explicit_cloud_resolver(
        "elevenlabs", _has_elevenlabs_key, "STT provider 'elevenlabs' configured but ELEVENLABS_API_KEY not set"),
    "deepinfra": _explicit_cloud_resolver(
        "deepinfra", _has_deepinfra_key,
        "STT provider 'deepinfra' configured but DEEPINFRA_API_KEY not set "
        "(or openai package missing)"),
}

# Auto-detect ladder for cloud providers, in priority order: (check, name, log).
# DeepInfra is LAST so a DEEPINFRA_API_KEY set for the chat surface never
# displaces an existing xAI/ElevenLabs auto-selection. Mistral only
# auto-selects when the SDK is already present — no lazy-install during
# passive auto-detection (explicit ``provider: mistral`` installs on first use).
_AUTO_DETECT_CLOUD = (
    (_has_groq_key, "groq", "No local STT available, using Groq Whisper API"),
    (lambda: _HAS_OPENAI and _has_openai_audio_backend(),
     "openai", "No local STT available, using OpenAI Whisper API"),
    (_has_mistral_key, "mistral", "No local STT available, using Mistral Voxtral Transcribe API"),
    (_has_xai_stt_credentials_quietly, "xai", "No local STT available, using xAI Grok STT API"),
    (_has_elevenlabs_key, "elevenlabs", "No local STT available, using ElevenLabs Scribe STT API"),
    (_has_deepinfra_key, "deepinfra", "No local STT available, using DeepInfra Whisper API"),
)


def _get_provider(stt_config: dict) -> str:
    """Determine which STT provider to use.

    An explicit ``stt.provider`` is honoured — no silent cloud fallback. With no
    provider configured, auto-detect tries local > groq > openai > mistral > xai
    > elevenlabs > deepinfra.
    """
    if not is_stt_enabled(stt_config):
        return "none"

    explicit = "provider" in stt_config
    provider = stt_config.get("provider", DEFAULT_PROVIDER)

    # The managed "Nous Subscription" selection is serviced by the OpenAI
    # implementation, routed through the managed gateway by
    # _resolve_openai_audio_client_config.
    if isinstance(provider, str) and provider.strip().lower() == "nous":
        provider = "openai"

    if explicit and provider == "local":
        # Legacy DEFAULT_CONFIG seeded ``stt.provider: local`` on every install,
        # so a merged-config "local" is not proof of a user pick. Only a raw
        # config.yaml selection counts as explicit; otherwise autodetect (which
        # prefers local first anyway).
        try:
            from tools.tool_backend_helpers import read_selection

            if read_selection("stt") is None:
                explicit = False
        except Exception:  # pragma: no cover — helpers are in-repo
            pass

    if explicit:
        resolver = _EXPLICIT_PROVIDER_RESOLVERS.get(provider)
        return resolver() if resolver else provider  # Unknown — let it fail downstream

    if _HAS_FASTER_WHISPER:
        return "local"
    if _has_local_command():
        return "local_command"
    if _try_lazy_install_stt():
        return "local"
    for available, name, message in _AUTO_DETECT_CLOUD:
        if available():
            logger.info(message)
            return name
    return "none"


# ---------------------------------------------------------------------------
# Plugin provider dispatch
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# pre_transcription plugin hook (STT prompt/vocab threading)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Shared validation
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Provider: local (faster-whisper)
# ---------------------------------------------------------------------------


def _unload_local_model() -> None:
    """Release the cached local whisper model. Thread-safe via the model lock."""
    global _local_model, _local_model_name
    with _local_model_lock:
        if _local_model is not None:
            logger.info(
                "Unloading local whisper model '%s' after idle timeout",
                _local_model_name or "unknown",
            )
            _local_model = None
            _local_model_name = None


def _start_idle_unload_watcher(timeout_seconds: int) -> None:
    """Ensure the single idle-unload watcher thread is running.

    Started only when none is alive (one lock + one ``is_alive()`` per
    transcription). The loop re-reads ``stt.local.unload_after_idle_seconds``
    every cycle so config edits apply within one interval; ``timeout_seconds``
    seeds the first cycle so a just-written config is honored even if a
    concurrent read races. After unloading, when the timeout becomes 0, or when
    the model is already gone, the thread exits; the next transcription restarts it.
    """
    global _idle_unload_thread
    with _idle_unload_mgmt_lock:
        if _idle_unload_thread is not None and _idle_unload_thread.is_alive():
            return

        def _watch(initial_timeout=timeout_seconds):
            timeout = initial_timeout
            while not _idle_unload_stop.is_set():
                if _idle_unload_stop.wait(_IDLE_UNLOAD_CHECK_INTERVAL):
                    break
                if _local_model is None:
                    break
                try:
                    timeout = _get_idle_unload_seconds(
                        _load_stt_config().get("local") or {}
                    )
                except Exception:  # noqa: BLE001 - keep the seed value
                    timeout = initial_timeout
                if timeout <= 0:
                    break  # unload disabled mid-flight — stand down
                if time.monotonic() - _last_transcription_time >= timeout:
                    _unload_local_model()
                    break

        _idle_unload_stop.clear()
        _idle_unload_thread = threading.Thread(
            target=_watch, name="hermes-stt-idle-unload", daemon=True
        )
        _idle_unload_thread.start()


def _touch_transcription_time() -> None:
    """Record transcription activity (resets the idle timer)."""
    global _last_transcription_time
    _last_transcription_time = time.monotonic()


def _get_or_load_local_model(model_name: str, local_cfg: Dict[str, Any]):
    """Return the cached faster-whisper model, (re)loading under the lock when needed.

    Double-checked lock: concurrent voice messages must not both download/load.
    The returned strong reference stays valid even if the idle watcher nulls the
    module global mid-transcription.
    """
    global _local_model, _local_model_name
    model = _local_model
    if model is None or _local_model_name != model_name:
        with _local_model_lock:
            if _local_model is None or _local_model_name != model_name:
                logger.info("Loading faster-whisper model '%s' (first load downloads the model)...", model_name)
                # stt.local.device / compute_type let users pin a configuration
                # where ``auto`` mis-detects; the loader keeps the CUDA→CPU fallback.
                _local_model = _load_local_whisper_model(
                    model_name,
                    device=local_cfg.get("device", "auto"),
                    compute_type=local_cfg.get("compute_type", "auto"),
                )
                _local_model_name = model_name
            model = _local_model
    return model


def _transcribe_local(
    file_path: str,
    model_name: str,
    *,
    language: Optional[str] = None,
    prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """Transcribe using faster-whisper (local, free)."""
    global _local_model, _local_model_name

    if not _HAS_FASTER_WHISPER and not _try_lazy_install_stt():
        return _error_result("faster-whisper not installed")

    try:
        local_cfg = _load_stt_config().get("local") or {}
        # Reset the idle timer BEFORE loading/transcribing so the watcher can't
        # count a long in-flight transcription as idle time and unload mid-use.
        _touch_transcription_time()
        model = _get_or_load_local_model(model_name, local_cfg)
        if model is None:  # defensive: load failed without raising
            return _error_result("Local whisper model failed to load")

        stt_config = _load_stt_config()
        local_config = stt_config.get("local") or {}
        transcribe_kwargs = build_local_transcribe_kwargs(stt_config)
        # pre_transcription hook overrides win over config-resolved values.
        if language:
            transcribe_kwargs["language"] = language
        if prompt:
            transcribe_kwargs["initial_prompt"] = prompt

        try:
            segments, info = model.transcribe(file_path, **transcribe_kwargs)
            transcript = _join_confident_segments(segments, local_config)
        except Exception as exc:
            # CUDA libs sometimes only fail at dlopen-on-first-use, AFTER the
            # model loaded. Evict the poisoned cached model, reload on CPU and
            # retry once — otherwise every later voice message fails until restart.
            if not _looks_like_cuda_lib_error(exc):
                raise
            logger.warning(
                "faster-whisper CUDA runtime failed mid-transcribe (%s) — "
                "evicting cached model and retrying on CPU (int8).",
                exc,
            )
            from faster_whisper import WhisperModel
            model = WhisperModel(model_name, device="cpu", compute_type="int8")
            with _local_model_lock:
                _local_model = model
                _local_model_name = model_name
            segments, info = model.transcribe(file_path, **transcribe_kwargs)
            transcript = _join_confident_segments(segments, local_config)

        logger.info(
            "Transcribed %s via local whisper (%s, lang=%s, %.1fs audio)",
            Path(file_path).name, model_name, info.language, info.duration,
        )

        _touch_transcription_time()
        idle_timeout = _get_idle_unload_seconds(local_cfg)
        if idle_timeout > 0:
            _start_idle_unload_watcher(idle_timeout)

        return _ok_result(transcript, "local")

    except Exception as e:
        logger.error("Local transcription failed: %s", e, exc_info=True)
        return _error_result(f"Local transcription failed: {e}")


# ---------------------------------------------------------------------------
# OpenAI-SDK-shaped providers: groq, openai (+ deepinfra via openai)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Provider: mistral (Voxtral Transcribe API)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# REST multipart providers: xAI, ElevenLabs
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Provider: ElevenLabs (Scribe STT API)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Provider: DeepInfra (OpenAI-compatible /v1/audio/transcriptions)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Cloud pre-upload silence trim
# ---------------------------------------------------------------------------
#
# Local faster-whisper gets Silero VAD; cloud providers get the raw file, so
# every second of silence is paid for twice (upload + per-minute billing) and
# cloud Whisper hallucinates on it. Before uploading to a built-in cloud
# provider we collapse long pauses with ffmpeg's silenceremove, keeping
# ``stt.cloud_trim_keep_ms`` of each pause so word boundaries survive.
# Purely best-effort — ANY of these uploads the original untouched:
# ``stt.cloud_trim_silence: false``, ffmpeg/ffprobe missing, trim failure or
# timeout, a ~empty result (the provider, not a dB heuristic, decides "no
# speech"), or <10% saving. Command-type and plugin providers are NOT trimmed:
# they may wrap local CLIs that want the original bytes.


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _transcribe_prepared_audio(
    file_path: str,
    model: Optional[str] = None,
    source: Optional[str] = None,
) -> Dict[str, Any]:
    """Transcribe a validated audio file with the configured STT provider.

    ``model`` overrides the config/provider default; ``source`` is a caller-surface
    label (``"gateway"``, ``"voice_mode"``) forwarded to the ``pre_transcription``
    hook for observability only. Returns the standard result envelope.
    """
    # Refuse to feed a credential / secret store (auth.json, .env, OAuth
    # tokens, ...) to an STT provider, which would ship its plaintext to a
    # third-party API. Mirrors the image-gen / video-gen read guards.
    from agent.file_safety import get_read_block_error
    blocked = get_read_block_error(file_path)
    if blocked:
        return _error_result(blocked)

    # Validate before provider resolution so invalid files cannot trigger
    # provider setup or lazy installation. The remote-upload size cap is
    # enforced below, only for non-local providers.
    error = _validate_audio_file(file_path, enforce_size_limit=False)
    if error:
        return error

    stt_config = _load_stt_config()
    if not is_stt_enabled(stt_config):
        return _error_result("STT is disabled in config.yaml (stt.enabled: false).")

    provider = _get_provider(stt_config)
    if not _is_local_stt_provider(provider, stt_config):
        error = _validate_audio_file_size(Path(file_path))
        if error:
            return error

    # Convert CAF (iMessage voice notes) to WAV for cloud STT providers.
    if Path(file_path).suffix.lower() == ".caf" and provider not in ("local", "local_command"):
        converted = _convert_caf_to_wav(file_path)
        if not converted:
            return _error_result("CAF audio could not be converted to WAV.")
        file_path = converted

    # Best-effort pre-upload silence trim for built-in cloud providers.
    trim_cleanup_dir: Optional[str] = None
    if provider in CLOUD_STT_PROVIDERS:
        trimmed = _trim_silence_for_cloud_stt(file_path, stt_config)
        if trimmed:
            file_path = trimmed
            trim_cleanup_dir = os.path.dirname(trimmed)

    try:
        return _dispatch_stt_provider(file_path, provider, stt_config, model, source)
    finally:
        if trim_cleanup_dir:
            shutil.rmtree(trim_cleanup_dir, ignore_errors=True)


def _builtin_model_name(provider: str, stt_config: Dict[str, Any], model: Optional[str]) -> str:
    """Resolve the model for a built-in provider: caller override > ``stt.<provider>`` config > default."""
    if model:
        return model
    if provider in ("local", "local_command"):
        return _get_stt_section(stt_config, "local").get("model", DEFAULT_LOCAL_MODEL)
    if provider == "xai":
        return "grok-stt"  # xAI STT takes no model parameter — logging only
    cfg = _get_stt_section(stt_config, provider)
    if provider == "groq":
        return cfg.get("model") or DEFAULT_GROQ_STT_MODEL
    if provider == "openai":
        return cfg.get("model", DEFAULT_STT_MODEL)
    if provider == "mistral":
        return cfg.get("model", DEFAULT_MISTRAL_STT_MODEL)
    if provider == "elevenlabs":
        return cfg.get("model_id", DEFAULT_ELEVENLABS_STT_MODEL)
    return cfg.get("model") or ""  # deepinfra: resolved from the live catalog when empty


# Built-in provider -> handler. Looked up at call time so tests may patch the
# module-level ``_transcribe_*`` functions.
_BUILTIN_STT_HANDLERS = {
    "local": lambda *a, **kw: _transcribe_local(*a, **kw),
    "local_command": lambda *a, **kw: _transcribe_local_command(*a, **kw),
    "groq": lambda *a, **kw: _transcribe_groq(*a, **kw),
    "openai": lambda *a, **kw: _transcribe_openai(*a, **kw),
    "mistral": lambda *a, **kw: _transcribe_mistral(*a, **kw),
    "xai": lambda *a, **kw: _transcribe_xai(*a, **kw),
    "elevenlabs": lambda *a, **kw: _transcribe_elevenlabs(*a, **kw),
    "deepinfra": lambda *a, **kw: _transcribe_deepinfra(*a, **kw),
}


def _dispatch_stt_provider(
    file_path: str,
    provider: str,
    stt_config: Dict[str, Any],
    model: Optional[str] = None,
    source: Optional[str] = None,
) -> Dict[str, Any]:
    """Route *file_path* to the handler for *provider* (built-in > command > plugin)."""
    # Static ``stt.prompt`` is the base; pre_transcription hook results mutate
    # on top in registration order (last hook to set a field wins).
    prompt = stt_config.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        prompt = None

    # The hook fires after provider resolution and BEFORE any backend is
    # invoked; ``language`` stays None unless a hook overrides it.
    model, language, prompt = _apply_pre_transcription_hook(
        file_path=file_path,
        provider=provider,
        model=model,
        language=_get_stt_section(stt_config, provider).get("language"),
        prompt=prompt,
        source=source,
    )
    prompt = _enforce_prompt_length_limit(prompt, provider)

    handler = _BUILTIN_STT_HANDLERS.get(provider)
    if handler is not None:
        model_name = _builtin_model_name(provider, stt_config, model)
        if provider in ("local", "local_command"):
            model_name = _normalize_local_model(model_name)
        return handler(file_path, model_name, language=language, prompt=prompt)

    # User-declared command provider: after built-ins (so ``stt.providers.openai
    # .command`` can't override the real handler) and BEFORE plugins, because
    # config is more local than a plugin install (same precedence as TTS).
    command_provider_config = _resolve_command_stt_provider_config(provider, stt_config)
    if command_provider_config is not None:
        return _transcribe_command_stt(
            file_path,
            provider,
            command_provider_config,
            stt_config,
            model_override=model,
            language_override=language,
            prompt=prompt,
        )

    # Plugin-registered backend. Plugins read per-provider config under
    # ``stt.<provider>`` like built-ins; the ``model`` argument overrides it.
    plugin_cfg = _get_stt_section(stt_config, provider)
    plugin_result = _dispatch_to_plugin_provider(
        file_path,
        provider,
        stt_config,
        model=model or plugin_cfg.get("model"),
        language=language or _resolve_stt_language(provider, stt_config),
        prompt=prompt,
    )
    if plugin_result is not None:
        return plugin_result

    provider_key = str(provider or "").strip().lower()
    if (
        "provider" in stt_config
        and provider_key
        and provider_key not in BUILTIN_STT_PROVIDERS
        and provider_key != "none"
    ):
        return _unregistered_stt_provider_error(provider_key)

    # An explicit openai selection flattened to "none" carries a
    # selection-specific reason (e.g. managed openai-audio gateway down);
    # surface it with its remediation instead of the all-provider hint.
    if provider_key == "none" and str(stt_config.get("provider") or "") == "openai" and _HAS_OPENAI:
        try:
            _resolve_openai_audio_client_config()
        except ValueError as exc:
            return _error_result(str(exc))

    return _error_result(
        "No STT provider available. Install faster-whisper for free local "
        f"transcription, configure {LOCAL_STT_COMMAND_ENV} or install a local whisper CLI, "
        "set GROQ_API_KEY for free Groq Whisper, set MISTRAL_API_KEY for Mistral "
        "Voxtral Transcribe, configure xAI OAuth or set XAI_API_KEY for xAI Grok STT, "
        "set ELEVENLABS_API_KEY for ElevenLabs Scribe, or set VOICE_TOOLS_OPENAI_KEY "
        "or OPENAI_API_KEY for the OpenAI Whisper API."
    )


def transcribe_audio(
    file_path: str,
    model: Optional[str] = None,
    source: Optional[str] = None,
) -> Dict[str, Any]:
    """Safely validate, preprocess supported inputs, and dispatch transcription.

    ``source`` is an optional caller-surface label (``"gateway"``, ``"voice_mode"``)
    forwarded to the ``pre_transcription`` hook for observability only.
    """
    # Secret-store refusal runs before ANY validation so the error names the
    # real reason rather than a format error.
    from agent.file_safety import get_read_block_error
    blocked = get_read_block_error(file_path)
    if blocked:
        return _error_result(blocked)

    # Cap .silk sources before the decoder runs (decoder safety); for all other
    # inputs the upload cap is provider-scoped in _transcribe_prepared_audio,
    # so local whisper can handle big files.
    is_silk = Path(file_path).suffix.lower() == ".silk"
    source_error = _validate_audio_source_file(file_path, enforce_size_limit=is_silk)
    if source_error:
        return source_error

    prepared_path, cleanup_dir, prep_error = _prepare_audio_for_transcription(file_path)
    if prep_error:
        return prep_error
    if prepared_path is None:
        return _error_result("Audio preprocessing did not produce a file for transcription.")

    try:
        prepared_error = _validate_audio_file(prepared_path, enforce_size_limit=False)
        if prepared_error:
            return prepared_error
        return _transcribe_prepared_audio(prepared_path, model, source)
    finally:
        if cleanup_dir:
            shutil.rmtree(cleanup_dir, ignore_errors=True)


def transcribe_audio_local_fallback(
    file_path: str,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """Try an already-installed local STT backend without changing config.

    For passive inbound-media recovery after the configured provider failed:
    never lazy-installs or falls through to a cloud provider.
    """
    error = _validate_audio_file(file_path)
    if error:
        return error

    local_cfg = _load_stt_config().get("local") or {}
    local_model = model or local_cfg.get("model", DEFAULT_LOCAL_MODEL)

    if _HAS_FASTER_WHISPER:
        return _transcribe_local(file_path, _normalize_local_model(local_model))
    if _has_local_command():
        return _transcribe_local_command(file_path, _normalize_local_command_model(local_model))
    return _error_result("No installed local STT backend is available.", provider="local")


