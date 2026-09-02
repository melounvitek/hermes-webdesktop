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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_PROVIDER = "local"
DEFAULT_LOCAL_MODEL = "base"
DEFAULT_LOCAL_STT_LANGUAGE = "en"
DEFAULT_STT_MODEL = os.getenv("STT_OPENAI_MODEL", "whisper-1")
DEFAULT_GROQ_STT_MODEL = os.getenv("STT_GROQ_MODEL", "whisper-large-v3-turbo")
DEFAULT_MISTRAL_STT_MODEL = os.getenv("STT_MISTRAL_MODEL", "voxtral-mini-latest")
DEFAULT_ELEVENLABS_STT_MODEL = os.getenv("STT_ELEVENLABS_MODEL", "scribe_v2")
LOCAL_STT_COMMAND_ENV = "HERMES_LOCAL_STT_COMMAND"
LOCAL_STT_LANGUAGE_ENV = "HERMES_LOCAL_STT_LANGUAGE"
COMMON_LOCAL_BIN_DIRS = ("/opt/homebrew/bin", "/usr/local/bin")

GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
OPENAI_BASE_URL = os.getenv("STT_OPENAI_BASE_URL", "https://api.openai.com/v1")
XAI_STT_BASE_URL = os.getenv("XAI_STT_BASE_URL", "https://api.x.ai/v1")
ELEVENLABS_STT_BASE_URL = os.getenv("ELEVENLABS_STT_BASE_URL", "https://api.elevenlabs.io/v1")
# DeepInfra STT base URL is resolved via hermes_cli.models.deepinfra_base_url (shared).

SUPPORTED_FORMATS = {".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm", ".ogg", ".oga", ".opus", ".aac", ".flac", ".caf"}
LOCAL_NATIVE_AUDIO_FORMATS = {".wav", ".aiff", ".aif"}
MAX_FILE_SIZE = 25 * 1024 * 1024  # 25 MB

# Known model sets for auto-correction
OPENAI_MODELS = {"whisper-1", "gpt-4o-mini-transcribe", "gpt-4o-transcribe", "gpt-transcribe"}
GROQ_MODELS = {"whisper-large-v3", "whisper-large-v3-turbo", "distil-whisper-large-v3-en"}

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


def _error_result(error: str, **extra: Any) -> Dict[str, Any]:
    """Standard failure envelope shared by every provider and validator."""
    return {"success": False, "transcript": "", "error": error, **extra}


def _ok_result(transcript: str, provider: str) -> Dict[str, Any]:
    return {"success": True, "transcript": transcript, "provider": provider}


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


def _get_stt_section(stt_config: Dict[str, Any], name: str) -> Dict[str, Any]:
    """Return an stt sub-section if it's a dict, else an empty dict."""
    if not isinstance(stt_config, dict):
        return {}
    section = stt_config.get(name)
    return section if isinstance(section, dict) else {}


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


def _find_binary(binary_name: str) -> Optional[str]:
    """Find a local binary, checking common Homebrew/local prefixes as well as PATH."""
    for directory in COMMON_LOCAL_BIN_DIRS:
        candidate = Path(directory) / binary_name
        if candidate.exists() and os.access(candidate, os.X_OK):
            return str(candidate)
    return shutil.which(binary_name)


def _find_ffmpeg_binary() -> Optional[str]:
    return _find_binary("ffmpeg")


def _find_ffprobe_binary() -> Optional[str]:
    return _find_binary("ffprobe")


def _find_whisper_binary() -> Optional[str]:
    return _find_binary("whisper")


def _run_quiet(command: list, *, timeout: float, env: Optional[dict] = None) -> subprocess.CompletedProcess:
    """``subprocess.run`` for STT helper binaries: checked, captured, utf-8 text, no stdin, hidden window."""
    return subprocess.run(
        command, check=True, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=timeout,
        stdin=subprocess.DEVNULL, env=env, creationflags=windows_hide_flags(),
    )


# Shared encode profile for every STT-bound m4a (transcode and silence-trim):
# 16 kHz mono 32 kbps AAC, faststart. One owner so codec/bitrate never drift.
_STT_M4A_ENCODE_ARGS = (
    "-vn", "-ac", "1", "-ar", "16000",
    "-c:a", "aac", "-b:a", "32k", "-movflags", "+faststart",
)


def _run_ffmpeg_stt_encode(
    ffmpeg: str, input_path: str, output_path: str, *, audio_filter: Optional[str] = None
) -> None:
    """Run the shared STT m4a encode, optionally with an ``-af`` filter.

    Raises on failure — callers own the error semantics (transcode reports, trim swallows).
    """
    command = [ffmpeg, "-y", "-i", input_path]
    if audio_filter:
        command += ["-af", audio_filter]
    command += [*_STT_M4A_ENCODE_ARGS, output_path]
    _run_quiet(command, timeout=120)


def _transcode_audio_for_stt(file_path: str, work_dir: str) -> tuple[Optional[str], Optional[str]]:
    """Transcode to a compact 16 kHz mono AAC/m4a for STT upload.

    Newer OpenAI models reject containers ``whisper-1`` accepted (notably Ogg/Opus
    voice notes) and gateway downloads may carry a misleading extension.
    Returns ``(converted_path, None)`` or ``(None, error)``.
    """
    ffmpeg = _find_ffmpeg_binary()
    if not ffmpeg:
        return None, "audio needs transcoding for the STT API, but ffmpeg was not found"
    converted_path = os.path.join(work_dir, f"{Path(file_path).stem or 'audio'}-stt.m4a")
    try:
        _run_ffmpeg_stt_encode(ffmpeg, file_path, converted_path)
        return converted_path, None
    except subprocess.CalledProcessError as exc:
        details = exc.stderr.strip() or exc.stdout.strip() or str(exc)
        logger.error("ffmpeg STT transcode failed for %s: %s", file_path, details)
        return None, f"failed to transcode audio for the STT API: {details}"
    except Exception as exc:  # noqa: BLE001 - transcode is best-effort
        logger.error("unexpected STT transcode failure for %s: %s", file_path, exc, exc_info=True)
        return None, f"failed to transcode audio for the STT API: {exc}"


def _get_local_command_template() -> Optional[str]:
    configured = os.getenv(LOCAL_STT_COMMAND_ENV, "").strip()
    if configured:
        return configured
    whisper_binary = _find_whisper_binary()
    if whisper_binary:
        return (
            f"{shlex.quote(whisper_binary)} {{input_path}} --model {{model}} --output_format txt "
            "--output_dir {output_dir} --language {language}"
        )
    return None


def _has_local_command() -> bool:
    return _get_local_command_template() is not None


def _normalize_local_model(model_name: Optional[str]) -> str:
    """Return a valid faster-whisper size; cloud-only names (``whisper-1`` …) fall back to the default with a warning."""
    if not model_name:
        return DEFAULT_LOCAL_MODEL
    if model_name in OPENAI_MODELS or model_name in GROQ_MODELS:
        logger.warning(
            "STT model '%s' is a cloud-only name and cannot be used with the local "
            "provider. Falling back to '%s'. Set stt.local.model to a valid "
            "faster-whisper size (tiny, base, small, medium, large-v3).",
            model_name,
            DEFAULT_LOCAL_MODEL,
        )
        return DEFAULT_LOCAL_MODEL
    return model_name


_normalize_local_command_model = _normalize_local_model


def _try_lazy_install_stt() -> bool:
    """Lazy-install faster-whisper and re-check dynamically so it's usable without a restart."""
    try:
        from tools.lazy_deps import ensure
        # prompt=False: a bare input() deadlocks under the interactive CLI where
        # prompt_toolkit owns stdin; the install is already gated by
        # security.allow_lazy_installs, so reaching here is opt-in.
        ensure("stt.faster_whisper", prompt=False)
        if _ilu.find_spec("faster_whisper"):
            return True
        logger.warning(
            "faster-whisper was installed but importlib still cannot find it "
            "(may require Python restart)"
        )
    except Exception as exc:
        logger.warning(
            "Lazy install of faster-whisper failed: %s. "
            "This is often a permission issue: the Hermes process user cannot "
            "write to the virtual environment. Try running manually as the "
            "venv owner: `stat -c '%%u' '$(dirname $(dirname $(which python3)))'` "
            "then `su - <owner> -c 'VIRTUAL_ENV=/opt/hermes/.venv "
            "uv pip install faster-whisper==1.2.1'`",
            exc,
        )
    return False


# Providers with native handlers here. Kept in sync with
# ``agent.transcription_registry._BUILTIN_NAMES`` (a regression test fails on
# drift); plugins may not register under these names and the dispatcher
# short-circuits them before command/plugin lookup.
BUILTIN_STT_PROVIDERS = frozenset({
    "local", "local_command", "groq", "openai", "mistral", "xai", "elevenlabs", "deepinfra",
})

# Built-in providers that upload audio to a remote API.
CLOUD_STT_PROVIDERS = frozenset(BUILTIN_STT_PROVIDERS - {"local", "local_command"})


# ---------------------------------------------------------------------------
# Command-provider registry (``stt.providers.<name>: type: command``)
# ---------------------------------------------------------------------------
#
# Mirrors the TTS command-provider registry: same placeholder grammar,
# shell-quote-aware rendering and process-tree termination on timeout.
# Resolution order: built-in name (always wins) > stt.providers.<name> command
# > plugin-registered TranscriptionProvider > "No STT provider available".
# The single-env-var HERMES_LOCAL_STT_COMMAND escape hatch stays untouched via
# the built-in ``local_command`` path.
DEFAULT_COMMAND_STT_TIMEOUT_SECONDS = 300
DEFAULT_COMMAND_STT_LANGUAGE = "en"
DEFAULT_COMMAND_STT_OUTPUT_FORMAT = "txt"
COMMAND_STT_OUTPUT_FORMATS = frozenset({"txt", "json", "srt", "vtt"})


def _get_named_stt_provider_config(
    stt_config: Dict[str, Any],
    name: str,
) -> Dict[str, Any]:
    """Return the config for a user-declared STT provider, or {}.

    ``stt.providers.<name>`` is canonical; ``stt.<name>`` is accepted for
    back-compat only when *name* is not a built-in, so a user's ``stt.openai``
    block still means the OpenAI provider. Built-in sections can't be mistaken
    for command providers anyway: ``_is_command_stt_provider_config`` requires
    an explicit ``command:``.
    """
    providers = _get_stt_section(stt_config, "providers")
    section = providers.get(name)
    if isinstance(section, dict):
        return section
    if name.lower() not in BUILTIN_STT_PROVIDERS:
        return _get_stt_section(stt_config, name)
    return {}


def _is_command_stt_provider_config(config: Dict[str, Any]) -> bool:
    """Return True when *config* declares a command-type STT provider."""
    if not isinstance(config, dict):
        return False
    ptype = str(config.get("type") or "").strip().lower()
    if ptype and ptype != "command":
        return False
    command = config.get("command")
    return isinstance(command, str) and bool(command.strip())


def _resolve_command_stt_provider_config(
    provider: str,
    stt_config: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Return the provider config if *provider* is a command type; None for built-ins, ``none``, unknown."""
    if not provider:
        return None
    key = provider.lower().strip()
    if key in BUILTIN_STT_PROVIDERS or key == "none":
        return None
    config = _get_named_stt_provider_config(stt_config, key)
    return config if _is_command_stt_provider_config(config) else None


def _is_local_stt_provider(provider: str, stt_config: Dict[str, Any]) -> bool:
    """Return whether *provider* is exempt from Hermes's remote upload cap."""
    return (provider or "").lower().strip() in {"local", "local_command"}


def _get_command_stt_timeout(config: Dict[str, Any]) -> float:
    """Return timeout in seconds, falling back when invalid."""
    raw = config.get("timeout", config.get("timeout_seconds", DEFAULT_COMMAND_STT_TIMEOUT_SECONDS))
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return float(DEFAULT_COMMAND_STT_TIMEOUT_SECONDS)
    return value if value > 0 else float(DEFAULT_COMMAND_STT_TIMEOUT_SECONDS)


def _get_command_stt_output_format(config: Dict[str, Any]) -> str:
    """Return the validated output format (txt/json/srt/vtt)."""
    raw = config.get("format") or config.get("output_format") or DEFAULT_COMMAND_STT_OUTPUT_FORMAT
    fmt = str(raw).lower().strip().lstrip(".")
    return fmt if fmt in COMMAND_STT_OUTPUT_FORMATS else DEFAULT_COMMAND_STT_OUTPUT_FORMAT


def _read_command_stt_output(output_path: Path, stdout: str, fmt: str) -> str:
    """Return the transcript: non-empty output file > non-empty stdout (curl one-liners) > RuntimeError.

    JSON output is returned raw — users configure ``format: txt`` or post-process.
    """
    if output_path.exists():
        try:
            content = output_path.read_text(encoding="utf-8").strip()
        except UnicodeDecodeError:
            content = output_path.read_bytes().decode("utf-8", errors="replace").strip()
        if content:
            return content
    if stdout and stdout.strip():
        return stdout.strip()
    raise RuntimeError(
        f"Command STT provider wrote no output file at {output_path} "
        f"and produced no stdout"
    )


def _log_prompt_unsupported(label: str) -> None:
    logger.debug("%s does not support transcription prompts — proceeding without the prompt.", label)


def _transcribe_command_stt(
    file_path: str,
    provider_name: str,
    config: Dict[str, Any],
    stt_config: Dict[str, Any],
    model_override: Optional[str] = None,
    language_override: Optional[str] = None,
    prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """Transcribe via a user-declared ``stt.providers.<name>: type: command``.

    Placeholders (all shell-quote-aware; ``{{``/``}}`` stay literal):
    ``{input_path}`` original audio path, ``{output_path}`` file to write the
    transcript to, ``{output_dir}`` its parent, ``{format}`` txt/json/srt/vtt,
    ``{language}`` (default ``en``), ``{model}`` (empty when unset).
    """
    if prompt:
        _log_prompt_unsupported(f"Command STT provider '{provider_name}'")

    def fail(error: str) -> Dict[str, Any]:
        return _error_result(error, provider=provider_name)

    command_template = str(config.get("command") or "").strip()
    if not command_template:
        return fail(f"stt.providers.{provider_name}.command is not configured")

    audio = Path(file_path).expanduser()
    if not audio.exists():
        return fail(f"Audio file not found: {file_path}")

    timeout = _get_command_stt_timeout(config)
    output_format = _get_command_stt_output_format(config)
    language = (
        language_override
        or config.get("language")
        or _resolve_stt_language(provider_name, stt_config)
        or DEFAULT_COMMAND_STT_LANGUAGE
    )
    model = model_override or config.get("model") or ""

    try:
        with tempfile.TemporaryDirectory(prefix=f"hermes-cmd-stt-{provider_name}-") as tmpdir:
            output_path = Path(tmpdir) / f"transcript.{output_format}"
            placeholders = {
                "input_path": str(audio.resolve()),
                "output_path": str(output_path),
                "output_dir": str(output_path.parent),
                "format": output_format,
                "language": str(language),
                "model": str(model),
            }
            command = _render_command_stt_template(command_template, placeholders)
            logger.info(
                "Transcribing %s via command STT provider '%s'...",
                audio.name, provider_name,
            )
            try:
                result = _run_command_stt(
                    command,
                    timeout,
                    env_passthrough=_command_stt_env_passthrough(config),
                )
            except subprocess.TimeoutExpired:
                return fail(f"STT command provider '{provider_name}' timed out after {timeout:g}s")
            except subprocess.CalledProcessError as exc:
                detail_parts = []
                if exc.stderr:
                    detail_parts.append(f"stderr: {exc.stderr.strip()}")
                if exc.stdout:
                    detail_parts.append(f"stdout: {exc.stdout.strip()}")
                detail = "; ".join(detail_parts) or "no command output"
                return fail(
                    f"STT command provider '{provider_name}' exited with code "
                    f"{exc.returncode}: {detail}"
                )

            try:
                transcript_text = _read_command_stt_output(
                    output_path, result.stdout or "", output_format,
                )
            except RuntimeError as exc:
                return fail(str(exc))

    except OSError as exc:
        return fail(f"STT command provider '{provider_name}' failed: {exc}")

    logger.info(
        "Transcribed %s via command STT provider '%s' (%d chars)",
        audio.name, provider_name, len(transcript_text),
    )
    return _ok_result(transcript_text, provider_name)


# ---------------------------------------------------------------------------
# Provider resolution
# ---------------------------------------------------------------------------


def _has_xai_stt_credentials() -> bool:
    from tools.xai_http import resolve_xai_http_credentials

    return bool(resolve_xai_http_credentials().get("api_key"))


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


def _unregistered_stt_provider_error(provider: str) -> Dict[str, Any]:
    key = str(provider or "").strip()
    return _error_result(
        f"stt.provider='{key}' is set but no built-in, command, or plugin "
        "provider registered that name. Run `hermes plugins list` to see "
        "installed STT plugins, or configure a command provider under "
        f"`stt.providers.{key}.command`.",
        provider=key,
        error_type="provider_not_registered",
    )


# ---------------------------------------------------------------------------
# Plugin provider dispatch
# ---------------------------------------------------------------------------


def _dispatch_to_plugin_provider(
    file_path: str,
    provider: str,
    stt_config: Optional[Dict[str, Any]] = None,
    *,
    model: Optional[str] = None,
    language: Optional[str] = None,
    prompt: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Route to a plugin-registered transcription provider; None when no plugin claims the name.

    Invariants (re-verified here even though the caller short-circuits first,
    so a caller refactor can't silently break them): built-in names never reach
    the registry; a same-name ``stt.providers.<name>: type: command`` wins over
    a plugin. A matched plugin reporting ``is_available() == False`` returns an
    error envelope — not None — because the user explicitly opted in via
    ``stt.provider`` and the generic fall-through message would mislead.
    Provider exceptions become the standard error envelope.
    """
    if not provider:
        return None
    key = provider.lower().strip()
    if key in BUILTIN_STT_PROVIDERS or key == "none":
        return None
    if stt_config is not None and _is_command_stt_provider_config(
        _get_named_stt_provider_config(stt_config, key)
    ):
        return None
    try:
        from agent.transcription_registry import get_provider
        from hermes_cli.plugins import _ensure_plugins_discovered

        _ensure_plugins_discovered()
        plugin_provider = get_provider(key)
        if plugin_provider is None:
            # Long-lived sessions may have discovered plugins before a backend
            # was patched in or config changed — retry once with a forced refresh.
            _ensure_plugins_discovered(force=True)
            plugin_provider = get_provider(key)
    except Exception as exc:  # noqa: BLE001 — discovery failure is non-fatal
        logger.debug("STT plugin dispatch skipped (discovery failed): %s", exc)
        return None
    if plugin_provider is None:
        return None

    # ``is_available()`` MUST NOT raise per the ABC contract; defend anyway so
    # a buggy plugin can't break dispatch for everyone.
    try:
        available = plugin_provider.is_available()
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "STT plugin provider '%s' is_available() raised: %s — "
            "treating as unavailable", key, exc, exc_info=True,
        )
        available = False
    if not available:
        logger.info(
            "STT plugin provider '%s' reports not available; returning "
            "unavailability envelope.", key,
        )
        return _error_result(
            f"STT plugin '{key}' is not available — check that its "
            "required credentials / dependencies are configured.",
            provider=key,
        )

    logger.info("Transcribing with plugin STT provider '%s'...", key)
    # The prompt travels via the ABC's ``**extra`` kwargs and is only sent when
    # set, so pre-prompt providers see byte-identical calls on the no-prompt path.
    extra_kwargs: Dict[str, Any] = {}
    if prompt is not None:
        extra_kwargs["prompt"] = prompt
    try:
        result = plugin_provider.transcribe(
            file_path,
            model=model,
            language=language,
            **extra_kwargs,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "STT plugin provider '%s' raised: %s", key, exc, exc_info=True,
        )
        return _error_result(f"STT plugin '{key}' raised: {exc}", provider=key)

    if not isinstance(result, dict):
        return _error_result(f"STT plugin '{key}' returned a non-dict result", provider=key)
    result.setdefault("provider", key)
    return result


# ---------------------------------------------------------------------------
# pre_transcription plugin hook (STT prompt/vocab threading)
# ---------------------------------------------------------------------------


# Fields a pre_transcription hook may mutate. ``file_path`` is read-only —
# attempts to change it are logged and dropped.
_PRE_TRANSCRIPTION_MUTABLE_FIELDS = ("prompt", "language", "model")

# Whisper-family models only use the final ~224 tokens of the prompt; longer
# values waste upload bytes and can trip stricter OpenAI-compatible servers.
# Enforced client-side (truncate with a warning, never error), ~4 chars/token.
_WHISPER_PROMPT_TOKEN_CAP = 224
_PROMPT_CHARS_PER_TOKEN = 4
_WHISPER_PROMPT_CAPPED_PROVIDERS = frozenset(
    {"local", "openai", "groq", "deepinfra"}
)


def _enforce_prompt_length_limit(
    prompt: Optional[str], provider: str
) -> Optional[str]:
    """Truncate *prompt* to the whisper-family token cap, keeping the TAIL (fail-open).

    Whisper conditions on the final context window, so the most recently
    appended hints survive. Other providers own their own validation.
    """
    if not prompt or provider not in _WHISPER_PROMPT_CAPPED_PROVIDERS:
        return prompt
    max_chars = _WHISPER_PROMPT_TOKEN_CAP * _PROMPT_CHARS_PER_TOKEN
    if len(prompt) <= max_chars:
        return prompt
    logger.warning(
        "Transcription prompt is ~%d tokens; whisper-family provider '%s' "
        "only uses the final ~%d — truncating to the last %d characters.",
        len(prompt) // _PROMPT_CHARS_PER_TOKEN,
        provider,
        _WHISPER_PROMPT_TOKEN_CAP,
        max_chars,
    )
    return prompt[-max_chars:]


def _apply_pre_transcription_hook(
    *,
    file_path: str,
    provider: str,
    model: Optional[str],
    language: Optional[str],
    prompt: Optional[str],
    source: Optional[str],
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Fire the ``pre_transcription`` plugin hook and merge its results.

    Gated on ``has_hook`` so the no-hook path never builds hook kwargs, and
    fail-open: any hook-plumbing error leaves the dispatch untouched. Results
    arrive in registration order (plugins discovered in sorted order) and are
    applied field-by-field, so the last hook to write a field wins. Model
    values are accepted as-is and flow through the same per-backend
    normalization a caller-supplied model would.

    Returns ``(model, language_override, prompt)``; ``language_override`` is
    None unless a hook explicitly set ``language``, so backends keep their own
    config/env language resolution.
    """
    try:
        from hermes_cli.plugins import has_hook, invoke_hook

        if not has_hook("pre_transcription"):
            return model, None, prompt

        hook_results = invoke_hook(
            "pre_transcription",
            file_path=file_path,
            provider=provider,
            model=model,
            language=language,
            prompt=prompt,
            source=source,
        )
        overrides: Dict[str, Any] = {}
        for hook_result in hook_results:
            if not isinstance(hook_result, dict):
                continue
            for key, value in hook_result.items():
                if key == "file_path":
                    logger.warning(
                        "pre_transcription hook attempted to change "
                        "file_path (read-only) — ignoring the attempt."
                    )
                    continue
                if key not in _PRE_TRANSCRIPTION_MUTABLE_FIELDS:
                    logger.debug(
                        "pre_transcription hook returned unsupported field "
                        "%r — ignoring.", key,
                    )
                    continue
                if not isinstance(value, str):
                    logger.debug(
                        "pre_transcription hook returned non-string value "
                        "%r for field %r — ignoring.", value, key,
                    )
                    continue
                overrides[key] = value

        if "model" in overrides:
            model = overrides["model"]
        if "prompt" in overrides:
            # Hooks win over the static ``stt.prompt`` config; "" clears it.
            prompt = overrides["prompt"] or None
        return model, overrides.get("language") or None, prompt
    except Exception as _hook_err:  # noqa: BLE001 — hook plumbing is fail-open
        logger.debug("pre_transcription hook error: %s", _hook_err)
        return model, None, prompt


# ---------------------------------------------------------------------------
# Shared validation
# ---------------------------------------------------------------------------


def _validate_audio_file_size(audio_path: Path) -> Optional[Dict[str, Any]]:
    """Return an error when *audio_path* exceeds the remote upload cap."""
    try:
        file_size = audio_path.stat().st_size
    except OSError as e:
        return _error_result(f"Failed to access file: {e}")
    if file_size > MAX_FILE_SIZE:
        return _error_result(
            f"File too large: {file_size / (1024*1024):.1f}MB (max {MAX_FILE_SIZE / (1024*1024):.0f}MB)"
        )
    return None


def _validate_audio_source_file(
    file_path: str,
    *,
    enforce_size_limit: bool = True,
) -> Optional[Dict[str, Any]]:
    """Validate source path safety (and optionally size) before any decoder runs."""
    audio_path = Path(file_path)

    if os.path.islink(audio_path):
        return _error_result(f"Path is a symbolic link: {file_path}")
    if not audio_path.exists():
        return _error_result(f"Audio file not found: {file_path}")
    if not audio_path.is_file():
        return _error_result(f"Path is not a file: {file_path}")
    if enforce_size_limit:
        return _validate_audio_file_size(audio_path)
    try:
        audio_path.stat()
    except OSError as e:
        return _error_result(f"Failed to access file: {e}")
    return None


def _validate_audio_file(
    file_path: str,
    *,
    enforce_size_limit: bool = True,
) -> Optional[Dict[str, Any]]:
    """Validate a supported, decoder-safe audio file."""
    source_error = _validate_audio_source_file(
        file_path, enforce_size_limit=enforce_size_limit
    )
    if source_error:
        return source_error

    suffix = Path(file_path).suffix
    if suffix.lower() not in SUPPORTED_FORMATS:
        return _error_result(
            f"Unsupported format: {suffix}. Supported: {', '.join(sorted(SUPPORTED_FORMATS))}"
        )
    return None


def _prepare_audio_for_transcription(
    file_path: str,
) -> tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]:
    """Convert a decoder-safe .silk source to a temporary supported WAV file."""
    audio_path = Path(file_path)
    if audio_path.suffix.lower() != ".silk":
        return file_path, None, None
    if not _HAS_PILK:
        # pilk is a tiny silk-v3 codec binding — lazy-install on first .silk
        # voice note instead of bloating the base install.
        try:
            from tools.lazy_deps import ensure as _lazy_ensure
            _lazy_ensure("stt.silk", prompt=False)
        except Exception:
            pass
        if not _safe_find_spec("pilk"):
            return None, None, _error_result(
                "Unsupported format: .silk. Install the optional 'pilk' dependency to enable WeChat voice transcription."
            )

    temp_dir = tempfile.mkdtemp(prefix="hermes-silk-")
    converted_path = os.path.join(temp_dir, f"{audio_path.stem}.wav")
    try:
        import pilk

        pilk.silk_to_wav(file_path, converted_path)
        if not Path(converted_path).is_file() or Path(converted_path).stat().st_size == 0:
            raise RuntimeError("pilk did not produce a readable WAV file")
        return converted_path, temp_dir, None
    except Exception as exc:
        shutil.rmtree(temp_dir, ignore_errors=True)
        logger.error("Failed to convert .silk audio %s: %s", file_path, exc, exc_info=True)
        return None, None, _error_result(f"Failed to convert .silk audio for transcription: {exc}")


# ---------------------------------------------------------------------------
# Provider: local (faster-whisper)
# ---------------------------------------------------------------------------


# Substrings identifying a missing/unloadable CUDA runtime library: when
# ctranslate2 can't dlopen one of these the "auto" device picker has already
# committed to CUDA, so we fall back to CPU and reload. Deliberately narrow
# (library names + dlopen phrasing) so legitimate runtime failures like "CUDA
# out of memory" surface to the user instead of silently running on CPU.
_CUDA_LIB_ERROR_MARKERS = (
    "libcublas", "libcudnn", "libcudart", "cannot be loaded", "cannot open shared object",
    "no kernel image is available", "CUBLAS_STATUS_NOT_SUPPORTED", "no CUDA-capable device",
    "CUDA driver version is insufficient",
)


def _looks_like_cuda_lib_error(exc: BaseException) -> bool:
    """Heuristic: is this a missing/broken CUDA runtime library (not a legitimate runtime failure)?"""
    msg = str(exc)
    return any(marker in msg for marker in _CUDA_LIB_ERROR_MARKERS)


def _sysctl_value(name: str) -> str:
    """Return a sysctl value, or an empty string when unavailable."""
    try:
        return subprocess.check_output(
            ["/usr/sbin/sysctl", "-n", name],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2,
        ).strip()
    except Exception:
        return ""


def _should_force_faster_whisper_cpu() -> bool:
    """Force CPU on Apple Silicon (incl. x86_64 under Rosetta), where ctranslate2's
    ``device="auto"`` can abort inside native code before Python can catch it."""
    if platform.system() != "Darwin":
        return False
    if platform.machine().lower() in {"arm64", "aarch64"}:
        return True
    # Under Rosetta platform.machine() reports x86_64; sysctl.proc_translated
    # flags translation and hw.optional.arm64 distinguishes Apple Silicon hosts.
    if _sysctl_value("sysctl.proc_translated") == "1":
        return True
    return _sysctl_value("hw.optional.arm64") == "1"


def _get_idle_unload_seconds(local_cfg: Dict[str, Any]) -> int:
    """Resolve the idle unload timeout from config; 0 = never (default), negatives clamp to 0."""
    try:
        val = int(local_cfg.get("unload_after_idle_seconds", 0))
    except (TypeError, ValueError):
        return 0
    return max(val, 0)


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


def _load_local_whisper_model(model_name: str, device: str = "auto", compute_type: str = "auto"):
    """Load faster-whisper with graceful CUDA → CPU fallback.

    ``device="auto"`` picks CUDA whenever the ctranslate2 wheel ships CUDA libs,
    even on hosts without the NVIDIA runtime (WSL2, headless servers, CPU-only
    dev boxes). Try the requested config first; on a CUDA library load failure
    fall back to CPU + int8. Pass ``stt.local.device`` / ``compute_type`` to pin.
    """
    force_cpu = _should_force_faster_whisper_cpu()
    if force_cpu:
        # Importing ctranslate2 can itself abort on Apple Silicon/Rosetta when
        # multiple Intel OpenMP runtimes are loaded — set before the import.
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    from faster_whisper import WhisperModel
    if force_cpu:
        logger.info(
            "Apple Silicon/Rosetta detected — loading faster-whisper on CPU "
            "(int8) to avoid native device autodetection crashes"
        )
        return WhisperModel(model_name, device="cpu", compute_type="int8")

    try:
        return WhisperModel(model_name, device=device, compute_type=compute_type)
    except Exception as exc:
        if not _looks_like_cuda_lib_error(exc):
            raise
        logger.warning(
            "faster-whisper CUDA load failed (%s) — falling back to CPU (int8). "
            "Install the NVIDIA CUDA runtime (libcublas/libcudnn) to use GPU.",
            exc,
        )
        return WhisperModel(model_name, device="cpu", compute_type="int8")


# Silence-hallucination hardening for local faster-whisper (whisper decodes
# junk like "You"/"Thank you." from pure silence). Three layers, all tunable
# under ``stt.local``: Silero VAD so silence never reaches the model
# (``vad: false`` restores raw behaviour for music/ambient audio);
# condition_on_previous_text=False so one hallucinated token can't seed a run;
# and the segment confidence gate in _is_hallucinated_segment.
_VAD_MIN_SILENCE_MS_DEFAULT = 500
_NO_SPEECH_PROB_THRESHOLD_DEFAULT = 0.6
_LOGPROB_THRESHOLD_DEFAULT = -1.0


def _config_number(cfg: Dict[str, Any], key: str, default, cast=float):
    """Read ``cfg[key]`` through *cast*, falling back to *default* on bad values."""
    try:
        return cast(cfg.get(key, default))
    except (TypeError, ValueError):
        return default


def build_local_transcribe_kwargs(stt_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Build the kwargs for EVERY local faster-whisper ``model.transcribe`` call.

    Single owner for the anti-hallucination hardening — new local-whisper call
    sites must go through here instead of hand-rolling kwargs.
    """
    stt_config = stt_config if isinstance(stt_config, dict) else _load_stt_config()
    local_cfg = stt_config.get("local") or {}

    kwargs: Dict[str, Any] = {
        "beam_size": 5,
        "condition_on_previous_text": False,
    }

    vad_enabled = local_cfg.get("vad", True)
    if vad_enabled is None:
        vad_enabled = True
    if bool(vad_enabled):
        kwargs["vad_filter"] = True
        kwargs["vad_parameters"] = {
            "min_silence_duration_ms": _config_number(
                local_cfg, "vad_min_silence_ms", _VAD_MIN_SILENCE_MS_DEFAULT, int
            )
        }
    else:
        kwargs["vad_filter"] = False

    # Push the confidence gate into faster-whisper itself: its internal
    # defaults drop low-confidence segments BEFORE our post-filter sees them,
    # so without this the ``stt.local`` threshold knobs were dead for that
    # first gate (non-English speech decodes at lower avg_logprob and was
    # silently discarded). Same values feed both gates; defaults unchanged.
    no_speech_threshold, log_prob_threshold = _confidence_thresholds(local_cfg)
    kwargs["no_speech_threshold"] = no_speech_threshold
    kwargs["log_prob_threshold"] = log_prob_threshold

    forced_lang = _resolve_stt_language("local", stt_config)
    if forced_lang:
        kwargs["language"] = forced_lang

    initial_prompt = local_cfg.get("initial_prompt")
    if isinstance(initial_prompt, str) and initial_prompt.strip():
        kwargs["initial_prompt"] = initial_prompt

    return kwargs


def _confidence_thresholds(local_cfg: Dict[str, Any]) -> tuple[float, float]:
    """Resolve (no_speech_prob, avg_logprob) gate thresholds from config."""
    return (
        _config_number(local_cfg, "no_speech_prob_threshold", _NO_SPEECH_PROB_THRESHOLD_DEFAULT),
        _config_number(local_cfg, "logprob_threshold", _LOGPROB_THRESHOLD_DEFAULT),
    )


def _is_hallucinated_segment(segment: Any, no_speech_threshold: float, logprob_threshold: float) -> bool:
    """True when a segment is very likely a silence hallucination.

    Conservative AND gate (openai-whisper's own heuristic): the model must BOTH
    think the window is non-speech AND have decoded it with low confidence, so
    quiet-but-real speech survives. Unknown segment shapes are never dropped.
    """
    no_speech_prob = getattr(segment, "no_speech_prob", None)
    avg_logprob = getattr(segment, "avg_logprob", None)
    if no_speech_prob is None or avg_logprob is None:
        return False
    try:
        no_speech_prob = float(no_speech_prob)
        avg_logprob = float(avg_logprob)
    except (TypeError, ValueError):
        return False
    return no_speech_prob > no_speech_threshold and avg_logprob < logprob_threshold


def _join_confident_segments(segments: Any, local_cfg: Dict[str, Any]) -> str:
    """Join segment texts, dropping probable silence hallucinations."""
    no_speech_threshold, logprob_threshold = _confidence_thresholds(local_cfg)
    kept: list[str] = []
    for segment in segments:
        if _is_hallucinated_segment(segment, no_speech_threshold, logprob_threshold):
            logger.debug(
                "Dropping probable hallucinated segment %r (no_speech_prob=%.3f, avg_logprob=%.3f)",
                getattr(segment, "text", ""),
                getattr(segment, "no_speech_prob", float("nan")),
                getattr(segment, "avg_logprob", float("nan")),
            )
            continue
        kept.append(segment.text.strip())
    return " ".join(kept).strip()


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


def _prepare_local_audio(file_path: str, work_dir: str) -> tuple[Optional[str], Optional[str]]:
    """Normalize audio for local CLI STT when needed."""
    audio_path = Path(file_path)
    if audio_path.suffix.lower() in LOCAL_NATIVE_AUDIO_FORMATS:
        return file_path, None

    ffmpeg = _find_ffmpeg_binary()
    if not ffmpeg:
        return None, "Local STT fallback requires ffmpeg for non-WAV inputs, but ffmpeg was not found"

    converted_path = os.path.join(work_dir, f"{audio_path.stem}.wav")
    try:
        _run_quiet([ffmpeg, "-y", "-i", file_path, converted_path], timeout=300)
        return converted_path, None
    except subprocess.TimeoutExpired:
        logger.error("ffmpeg conversion timed out for %s", file_path)
        return None, "Audio conversion for local STT timed out"
    except subprocess.CalledProcessError as e:
        details = e.stderr.strip() or e.stdout.strip() or str(e)
        logger.error("ffmpeg conversion failed for %s: %s", file_path, details)
        return None, f"Failed to convert audio for local STT: {details}"


def _convert_caf_to_wav(file_path: str) -> Optional[str]:
    """Convert CAF to WAV using ffmpeg or afconvert (macOS)."""
    audio_path = Path(file_path)
    wav_path = os.path.join(audio_path.parent, f"{audio_path.stem}.wav")
    ffmpeg = _find_ffmpeg_binary()
    if ffmpeg:
        try:
            subprocess.run([ffmpeg, "-y", "-i", file_path, wav_path],
                check=True, capture_output=True, text=True,
                timeout=300, stdin=subprocess.DEVNULL,
                creationflags=windows_hide_flags())
            return wav_path
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            logger.warning("ffmpeg CAF to WAV failed for %s: %s", file_path, e)
    afconvert = shutil.which("afconvert")
    if afconvert:
        try:
            subprocess.run([afconvert, file_path, wav_path, "-d", "LEI16", "-f", "WAVE"],
                check=True, capture_output=True, text=True,
                timeout=300, stdin=subprocess.DEVNULL)
            return wav_path
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            logger.warning("afconvert CAF to WAV failed for %s: %s", file_path, e)
    return None


def _transcribe_local_command(
    file_path: str,
    model_name: str,
    *,
    language: Optional[str] = None,
    prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the configured local STT command template and read back a .txt transcript."""
    if prompt:
        _log_prompt_unsupported("STT provider 'local_command'")

    command_template = _get_local_command_template()
    if not command_template:
        return _error_result(
            f"{LOCAL_STT_COMMAND_ENV} not configured and no local whisper binary was found"
        )

    # Language: hook override > stt.local.language > stt.language > env > "en".
    language = language or _resolve_stt_language("local") or DEFAULT_LOCAL_STT_LANGUAGE
    normalized_model = _normalize_local_command_model(model_name)

    try:
        with tempfile.TemporaryDirectory(prefix="hermes-local-stt-") as output_dir:
            prepared_input, prep_error = _prepare_local_audio(file_path, output_dir)
            if prep_error:
                return _error_result(prep_error)

            command = command_template.format(
                input_path=shlex.quote(prepared_input),
                output_dir=shlex.quote(output_dir),
                language=shlex.quote(language),
                model=shlex.quote(normalized_model),
            )
            # Scrub Hermes secrets from the child env (same policy as _run_command_stt).
            from tools.environments.local import hermes_subprocess_env

            _run_quiet(
                shlex.split(command), timeout=300,
                env=hermes_subprocess_env(inherit_credentials=False),
            )

            txt_files = sorted(Path(output_dir).glob("*.txt"))
            if not txt_files:
                return _error_result("Local STT command completed but did not produce a .txt transcript")

            transcript_text = txt_files[0].read_text(encoding="utf-8").strip()
            logger.info(
                "Transcribed %s via local STT command (%s, %d chars)",
                Path(file_path).name,
                normalized_model,
                len(transcript_text),
            )
            return _ok_result(transcript_text, "local_command")

    except KeyError as e:
        return _error_result(f"Invalid {LOCAL_STT_COMMAND_ENV} template, missing placeholder: {e}")
    except subprocess.CalledProcessError as e:
        details = e.stderr.strip() or e.stdout.strip() or str(e)
        logger.error("Local STT command failed for %s: %s", file_path, details)
        return _error_result(f"Local STT failed: {details}")
    except Exception as e:
        logger.error("Unexpected error during local command transcription: %s", e, exc_info=True)
        return _error_result(f"Local transcription failed: {e}")


# ---------------------------------------------------------------------------
# OpenAI-SDK-shaped providers: groq, openai (+ deepinfra via openai)
# ---------------------------------------------------------------------------


def _close_client(client: Any) -> None:
    close = getattr(client, "close", None)
    if callable(close):
        close()


def _openai_sdk_failure(exc: BaseException, file_path: str, log_label: str) -> Dict[str, Any]:
    """Map an OpenAI-SDK-shaped exception to the shared error envelope.

    Order matters: APIConnectionError is checked before APITimeoutError (its
    subclass) so timeouts report as connection errors, as they always have.
    """
    try:
        from openai import APIError, APIConnectionError, APITimeoutError
    except ImportError:  # pragma: no cover — callers gate on _HAS_OPENAI
        APIError = APIConnectionError = APITimeoutError = ()
    if isinstance(exc, PermissionError):
        return _error_result(f"Permission denied: {file_path}")
    if isinstance(exc, APIConnectionError):
        return _error_result(f"Connection error: {exc}")
    if isinstance(exc, APITimeoutError):
        return _error_result(f"Request timeout: {exc}")
    if isinstance(exc, APIError):
        return _error_result(f"API error: {exc}")
    logger.error("%s transcription failed: %s", log_label, exc, exc_info=True)
    return _error_result(f"Transcription failed: {exc}")


def _transcribe_groq(
    file_path: str,
    model_name: str,
    *,
    language: Optional[str] = None,
    prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """Transcribe using Groq Whisper API (free tier available).

    Language: hook override > ``stt.groq.language`` > ``stt.language`` > env;
    otherwise Groq auto-detects.
    """
    api_key = _resolve_provider_key("GROQ_API_KEY", "groq")
    if not api_key:
        return _error_result("GROQ_API_KEY not set")

    if not _HAS_OPENAI:
        return _error_result("openai package not installed")

    # Auto-correct model if caller passed an OpenAI-only model
    if model_name in OPENAI_MODELS:
        logger.info("Model %s not available on Groq, using %s", model_name, DEFAULT_GROQ_STT_MODEL)
        model_name = DEFAULT_GROQ_STT_MODEL

    language = language or _resolve_stt_language("groq")

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=GROQ_BASE_URL, timeout=30, max_retries=0)
        try:
            create_kwargs = {
                "model": model_name,
                "response_format": "text",
            }
            if language:
                create_kwargs["language"] = language
            if prompt:
                # Only sent when set so the no-hook, no-config request stays byte-identical.
                create_kwargs["prompt"] = prompt
            with open(file_path, "rb") as audio_file:
                transcription = client.audio.transcriptions.create(
                    file=audio_file,
                    **create_kwargs,
                )

            transcript_text = str(transcription).strip()
            logger.info("Transcribed %s via Groq API (%s, lang=%s, %d chars)",
                         Path(file_path).name, model_name, language or "auto", len(transcript_text))

            return _ok_result(transcript_text, "groq")
        finally:
            _close_client(client)

    except Exception as e:
        return _openai_sdk_failure(e, file_path, "Groq")


def _transcribe_openai(
    file_path: str,
    model_name: str,
    *,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    provider_label: str = "openai",
    language: Optional[str] = None,
    prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """Transcribe via the OpenAI ``audio.transcriptions.create`` SDK shape.

    Shared backend for every OpenAI-compatible STT endpoint (DeepInfra etc.):
    callers pass explicit ``api_key``/``base_url`` to skip the OpenAI-only auth
    chain and a ``provider_label`` for the response's ``provider``.
    """
    if api_key is None:
        try:
            api_key, fallback_base = _resolve_openai_audio_client_config()
        except ValueError as exc:
            return _error_result(str(exc))
        base_url = base_url or fallback_base

    # Language: hook override > stt.<provider>.language > stt.language > env > auto.
    language = language or _resolve_stt_language(provider_label)

    if not _HAS_OPENAI:
        return _error_result("openai package not installed")

    # Auto-correct a Groq-only model on the native OpenAI path only —
    # third-party endpoints may legitimately serve a whisper-large-v3 variant.
    if provider_label == "openai" and model_name in GROQ_MODELS:
        logger.info("Model %s not available on OpenAI, using %s", model_name, DEFAULT_STT_MODEL)
        model_name = DEFAULT_STT_MODEL

    try:
        from openai import OpenAI, BadRequestError
        client = OpenAI(api_key=api_key, base_url=base_url, timeout=30, max_retries=0)

        def _create_transcription(path: str):
            with open(path, "rb") as audio_file:
                create_kwargs = {
                    "model": model_name,
                    "file": audio_file,
                    "response_format": "text" if model_name == "whisper-1" else "json",
                }
                if language:
                    if model_name == "gpt-transcribe":
                        # gpt-transcribe replaces ``language`` with a ``languages``
                        # list and rejects requests sending the legacy field.
                        create_kwargs["extra_body"] = {"languages": [language]}
                    else:
                        create_kwargs["language"] = language
                    logger.debug("Using language hint '%s' for OpenAI STT", language)
                if prompt:
                    # Only sent when set so the no-hook, no-config request stays byte-identical.
                    create_kwargs["prompt"] = prompt
                return client.audio.transcriptions.create(**create_kwargs)

        try:
            with tempfile.TemporaryDirectory(prefix="hermes-stt-") as work_dir:
                try:
                    transcription = _create_transcription(file_path)
                except BadRequestError as exc:
                    message = str(exc).lower()
                    if not any(k in message for k in ("unsupported", "corrupted", "invalid file")):
                        raise
                    # Newer models reject some containers whisper-1 accepted
                    # (notably Ogg/Opus voice notes): transcode to m4a, retry once.
                    converted_path, transcode_error = _transcode_audio_for_stt(file_path, work_dir)
                    if transcode_error:
                        return _error_result(transcode_error)
                    logger.info(
                        "Retrying %s STT after transcoding %s to m4a (API rejected the original container)",
                        provider_label, Path(file_path).name,
                    )
                    transcription = _create_transcription(converted_path)

            transcript_text = _extract_transcript_text(transcription)
            logger.info(
                "Transcribed %s via %s (%s, %d chars)",
                Path(file_path).name, provider_label, model_name, len(transcript_text),
            )

            return _ok_result(transcript_text, provider_label)
        finally:
            _close_client(client)

    except Exception as e:
        return _openai_sdk_failure(e, file_path, provider_label)


# ---------------------------------------------------------------------------
# Provider: mistral (Voxtral Transcribe API)
# ---------------------------------------------------------------------------


def _transcribe_mistral(
    file_path: str,
    model_name: str,
    *,
    language: Optional[str] = None,
    prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """Transcribe with the ``mistralai`` SDK (``/v1/audio/transcriptions``); requires ``MISTRAL_API_KEY``."""
    api_key = _resolve_provider_key("MISTRAL_API_KEY", "mistral")
    if not api_key:
        return _error_result("MISTRAL_API_KEY not set")

    try:
        try:
            from tools.lazy_deps import ensure as _lazy_ensure
            _lazy_ensure("stt.mistral", prompt=False)
        except Exception:
            pass
        from mistralai.client import Mistral

        with Mistral(api_key=api_key) as client:
            with open(file_path, "rb") as audio_file:
                complete_kwargs: Dict[str, Any] = {
                    "model": model_name,
                    "file": {"content": audio_file, "file_name": Path(file_path).name},
                }
                # Language: hook override > stt.mistral.language > stt.language > env > auto.
                language = language or _resolve_stt_language("mistral")
                if language:
                    complete_kwargs["language"] = language
                if prompt:
                    # Only sent when set so the no-hook, no-config request stays byte-identical.
                    complete_kwargs["prompt"] = prompt
                result = client.audio.transcriptions.complete(**complete_kwargs)

            transcript_text = _extract_transcript_text(result)
            logger.info(
                "Transcribed %s via Mistral API (%s, %d chars)",
                Path(file_path).name, model_name, len(transcript_text),
            )
            return _ok_result(transcript_text, "mistral")

    except PermissionError:
        return _error_result(f"Permission denied: {file_path}")
    except Exception as e:
        logger.error("Mistral transcription failed: %s", e, exc_info=True)
        return _error_result(f"Mistral transcription failed: {type(e).__name__}")


# ---------------------------------------------------------------------------
# REST multipart providers: xAI, ElevenLabs
# ---------------------------------------------------------------------------


def _post_audio_multipart(url: str, headers: Dict[str, str], file_path: str, data: Dict[str, str]):
    import requests

    with open(file_path, "rb") as audio_file:
        return requests.post(
            url, headers=headers, files={"file": (Path(file_path).name, audio_file)},
            data=data, timeout=120,
        )


def _http_error_detail(response, extract) -> str:
    """``extract(json_body)`` -> detail string, falling back to the first 300 chars of the body."""
    try:
        return extract(response.json()) or response.text[:300]
    except Exception:
        return response.text[:300]


def _elevenlabs_error_detail(err_body: Dict[str, Any]) -> str:
    error_value = err_body.get("detail") or err_body.get("error")
    if isinstance(error_value, dict):
        return str(error_value.get("message") or error_value)
    return str(error_value) if error_value else ""


def _transcribe_xai(
    file_path: str,
    model_name: str,
    *,
    language: Optional[str] = None,
    prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """Transcribe via xAI ``POST /v1/stt`` (multipart). Supports ITN, diarization, word timestamps."""
    from tools.xai_http import resolve_xai_http_credentials

    if prompt:
        _log_prompt_unsupported("STT provider 'xai'")

    # STT is API-billed: prefer the explicit XAI_API_KEY over the general xAI
    # OAuth/Grok-subscription credential, which may be valid for Grok yet hit
    # personal-team spending-limit errors on /v1/stt.
    direct_api_key = str(get_env_value("XAI_API_KEY") or "").strip()
    if direct_api_key:
        creds = {
            "provider": "xai",
            "api_key": direct_api_key,
            "base_url": str(
                get_env_value("XAI_BASE_URL") or "https://api.x.ai/v1"
            ).strip().rstrip("/"),
        }
    else:
        creds = resolve_xai_http_credentials()
    api_key = str(creds.get("api_key") or "").strip()
    if not api_key:
        return _error_result(
            "No xAI credentials found. Configure xAI OAuth in `hermes model` or set XAI_API_KEY"
        )

    stt_config = _load_stt_config()
    xai_config = stt_config.get("xai") or {}

    def _resolve_base_url(resolved_creds: Dict[str, str]) -> str:
        # OAuth bearers are pinned to the resolver-validated xAI origin;
        # config/env base URL overrides only apply to API-key credentials.
        if resolved_creds.get("provider") == "xai-oauth":
            return str(
                resolved_creds.get("base_url") or XAI_STT_BASE_URL
            ).strip().rstrip("/")
        return str(
            xai_config.get("base_url")
            or get_env_value("XAI_STT_BASE_URL")
            or resolved_creds.get("base_url")
            or XAI_STT_BASE_URL
        ).strip().rstrip("/")

    base_url = _resolve_base_url(creds)
    # Language: hook override > stt.xai.language > stt.language > env.
    language = language or _resolve_stt_language("xai", stt_config) or ""
    use_format = is_truthy_value(xai_config.get("format", True))
    use_diarize = is_truthy_value(xai_config.get("diarize", False))

    try:
        from tools.xai_http import hermes_xai_user_agent

        data: Dict[str, str] = {}
        if language:
            data["language"] = language
        if use_format:
            data["format"] = "true"
        if use_diarize:
            data["diarize"] = "true"

        def _post_transcription(bearer: str, endpoint_base_url: str):
            return _post_audio_multipart(
                f"{endpoint_base_url}/stt",
                {"Authorization": f"Bearer {bearer}", "User-Agent": hermes_xai_user_agent()},
                file_path, data,
            )

        response = _post_transcription(api_key, base_url)

        if (
            response.status_code in {401, 403}
            and creds.get("provider") == "xai-oauth"
        ):
            logger.info(
                "xAI STT got HTTP %d; refreshing OAuth credentials and retrying once",
                response.status_code,
            )
            try:
                refreshed_creds = resolve_xai_http_credentials(
                    force_refresh=True,
                    api_key_hint=api_key,
                )
                refreshed_key = str(refreshed_creds.get("api_key") or "").strip()
                if refreshed_key and refreshed_key != api_key:
                    response = _post_transcription(
                        refreshed_key,
                        _resolve_base_url(refreshed_creds),
                    )
            except Exception as retry_exc:
                logger.warning(
                    "xAI STT OAuth refresh-and-retry after HTTP %d failed: %s",
                    response.status_code,
                    retry_exc,
                )

        if response.status_code != 200:
            detail = _http_error_detail(response, lambda body: body.get("error", {}).get("message", ""))
            return _error_result(f"xAI STT API error (HTTP {response.status_code}): {detail}")

        result = response.json()
        transcript_text = result.get("text", "").strip()

        if not transcript_text:
            return _error_result("xAI STT returned empty transcript", no_speech=True)

        logger.info(
            "Transcribed %s via xAI Grok STT (lang=%s, %.1fs audio, %d chars)",
            Path(file_path).name,
            result.get("language", language),
            result.get("duration", 0),
            len(transcript_text),
        )

        return _ok_result(transcript_text, "xai")

    except PermissionError:
        return _error_result(f"Permission denied: {file_path}")
    except Exception as e:
        logger.error("xAI STT transcription failed: %s", e, exc_info=True)
        return _error_result(f"xAI STT transcription failed: {e}")


# ---------------------------------------------------------------------------
# Provider: ElevenLabs (Scribe STT API)
# ---------------------------------------------------------------------------


def _transcribe_elevenlabs(
    file_path: str,
    model_name: str,
    *,
    language: Optional[str] = None,
    prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """Transcribe using ElevenLabs Scribe STT API."""
    if prompt:
        _log_prompt_unsupported("STT provider 'elevenlabs'")

    api_key = _resolve_provider_key("ELEVENLABS_API_KEY", "elevenlabs")
    if not api_key:
        return _error_result("ELEVENLABS_API_KEY not set")

    stt_config = _load_stt_config()
    elevenlabs_config = stt_config.get("elevenlabs") or {}
    base_url = str(
        elevenlabs_config.get("base_url")
        or get_env_value("ELEVENLABS_STT_BASE_URL")
        or ELEVENLABS_STT_BASE_URL
    ).strip().rstrip("/")
    # Language: hook override > stt.elevenlabs.language(_code) > stt.language.
    language_code = language or _resolve_stt_language(
        "elevenlabs", stt_config, extra_keys=("language_code",)
    ) or ""
    tag_audio_events = is_truthy_value(elevenlabs_config.get("tag_audio_events", False))
    diarize = is_truthy_value(elevenlabs_config.get("diarize", False))

    try:
        data: Dict[str, str] = {
            "model_id": model_name,
            "tag_audio_events": "true" if tag_audio_events else "false",
            "diarize": "true" if diarize else "false",
        }
        if language_code:
            data["language_code"] = language_code

        response = _post_audio_multipart(
            f"{base_url}/speech-to-text", {"xi-api-key": api_key}, file_path, data,
        )

        if response.status_code != 200:
            detail = _http_error_detail(response, _elevenlabs_error_detail)
            return _error_result(f"ElevenLabs STT API error (HTTP {response.status_code}): {detail}")

        transcript_text = _extract_transcript_text(response.json())
        if not transcript_text:
            return _error_result("ElevenLabs STT returned empty transcript", no_speech=True)

        logger.info(
            "Transcribed %s via ElevenLabs Scribe (%s, %d chars)",
            Path(file_path).name,
            model_name,
            len(transcript_text),
        )

        return _ok_result(transcript_text, "elevenlabs")

    except PermissionError:
        return _error_result(f"Permission denied: {file_path}")
    except Exception as e:
        logger.error("ElevenLabs STT transcription failed: %s", e, exc_info=True)
        return _error_result(f"ElevenLabs STT transcription failed: {e}")


# ---------------------------------------------------------------------------
# Provider: DeepInfra (OpenAI-compatible /v1/audio/transcriptions)
# ---------------------------------------------------------------------------


def _transcribe_deepinfra(
    file_path: str,
    model_name: str,
    *,
    language: Optional[str] = None,
    prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """Resolve DeepInfra credentials/model (via the shared ``hermes_cli.models``
    helpers), then delegate to :func:`_transcribe_openai`."""
    api_key = _resolve_provider_key("DEEPINFRA_API_KEY", "deepinfra")
    if not api_key:
        return _error_result("DEEPINFRA_API_KEY not set")

    from hermes_cli.models import deepinfra_base_url, deepinfra_model_ids

    # ``stt.deepinfra: null`` in YAML yields None, not {} — coalesce.
    base_url = deepinfra_base_url(_get_stt_section(_load_stt_config(), "deepinfra"))

    if not model_name:
        candidates = deepinfra_model_ids("stt")
        if not candidates:
            return _error_result(
                "No DeepInfra STT model available. Pin one in "
                "config.yaml under stt.deepinfra.model, or check "
                "connectivity to api.deepinfra.com so the live catalog "
                "can be fetched."
            )
        model_name = candidates[0]

    return _transcribe_openai(
        file_path,
        model_name,
        api_key=api_key,
        base_url=base_url,
        provider_label="deepinfra",
        language=language,
        prompt=prompt,
    )


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

_CLOUD_TRIM_THRESHOLD_DB_DEFAULT = -40  # audio below this level counts as silence
_CLOUD_TRIM_KEEP_MS_DEFAULT = 300  # how much of each pause survives the trim
_CLOUD_TRIM_MIN_SAVING = 0.10  # use the trimmed file only when >=10% shorter
_CLOUD_TRIM_MIN_RESULT_SECONDS = 0.3  # all-silence guard floor: never upload ~empty audio
# Below this the trim can't pay for itself (several providers bill a 10s
# minimum per request) and the encode would sit on the synchronous voice-note path.
_CLOUD_TRIM_MIN_INPUT_SECONDS = 12.0


def _probe_audio_duration(file_path: str) -> Optional[float]:
    """Return the audio duration in seconds via ffprobe, or None.

    Canonical sync probe; ``gateway/run.py._probe_audio_duration`` and the
    Telegram adapter carry local variants — keep the command shape in sync.
    """
    ffprobe = _find_ffprobe_binary()
    if not ffprobe:
        return None
    command = [
        ffprobe, "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        file_path,
    ]
    try:
        return float(_run_quiet(command, timeout=30).stdout.strip())
    except Exception:  # noqa: BLE001 - probe is best-effort
        return None


def _cloud_trim_settings(stt_config: Dict[str, Any]) -> tuple[bool, int, int]:
    """Resolve (enabled, threshold_db, keep_ms) for the cloud silence trim."""
    cfg = stt_config if isinstance(stt_config, dict) else {}
    # is_truthy_value: a YAML string "false" must disable, exactly like is_stt_enabled.
    enabled = is_truthy_value(cfg.get("cloud_trim_silence", True), default=True)
    threshold_db = _config_number(cfg, "cloud_trim_threshold_db", _CLOUD_TRIM_THRESHOLD_DB_DEFAULT, int)
    keep_ms = _config_number(cfg, "cloud_trim_keep_ms", _CLOUD_TRIM_KEEP_MS_DEFAULT, int)
    return enabled, threshold_db, max(keep_ms, 0)


def _trim_silence_for_cloud_stt(
    file_path: str, stt_config: Dict[str, Any]
) -> Optional[str]:
    """Return a silence-trimmed copy of *file_path* for cloud upload, or None.

    ``None`` always means "upload the original" (disabled, tools missing, clip
    too short, trim failed, mostly silence, or not enough saving). On success
    the caller owns deleting the returned file's parent directory.
    """
    enabled, threshold_db, keep_ms = _cloud_trim_settings(stt_config)
    if not enabled:
        return None
    ffmpeg = _find_ffmpeg_binary()
    if not ffmpeg:
        logger.debug("Cloud STT silence trim skipped: ffmpeg not found")
        return None
    original_duration = _probe_audio_duration(file_path)
    if not original_duration or original_duration <= 0:
        logger.debug("Cloud STT silence trim skipped: could not probe %s", file_path)
        return None
    if original_duration < _CLOUD_TRIM_MIN_INPUT_SECONDS:
        logger.debug(
            "Cloud STT silence trim skipped for %s: %.1fs is below the %.0fs gate",
            Path(file_path).name, original_duration, _CLOUD_TRIM_MIN_INPUT_SECONDS,
        )
        return None

    keep_seconds = keep_ms / 1000.0
    # start_periods=1 strips leading silence; stop_periods=-1 collapses every
    # interior/trailing silence, keeping ``keep_seconds`` of each pause.
    filter_expr = (
        f"silenceremove="
        f"start_periods=1:start_threshold={threshold_db}dB:start_silence={keep_seconds}:"
        f"stop_periods=-1:stop_threshold={threshold_db}dB:stop_silence={keep_seconds}"
    )
    work_dir = tempfile.mkdtemp(prefix="hermes-stt-trim-")
    trimmed_path = os.path.join(work_dir, f"{Path(file_path).stem or 'audio'}-trimmed.m4a")
    # Scale the all-silence guard with keep_ms: an output consisting solely
    # of kept pause must never be uploaded as "speech".
    min_result_seconds = max(_CLOUD_TRIM_MIN_RESULT_SECONDS, 2 * keep_seconds)
    keep_result = False
    try:
        _run_ffmpeg_stt_encode(ffmpeg, file_path, trimmed_path, audio_filter=filter_expr)
        trimmed_duration = _probe_audio_duration(trimmed_path)
        if not trimmed_duration or trimmed_duration < min_result_seconds:
            logger.debug(
                "Cloud STT silence trim discarded for %s: trimmed result ~empty (%.2fs)",
                Path(file_path).name, trimmed_duration or 0.0,
            )
            return None
        if trimmed_duration > original_duration * (1 - _CLOUD_TRIM_MIN_SAVING):
            logger.debug(
                "Cloud STT silence trim discarded for %s: saves <%.0f%% (%.1fs -> %.1fs)",
                Path(file_path).name, _CLOUD_TRIM_MIN_SAVING * 100,
                original_duration, trimmed_duration,
            )
            return None
        logger.info(
            "Trimmed silence from %s before cloud STT upload (%.1fs -> %.1fs, -%d%%)",
            Path(file_path).name, original_duration, trimmed_duration,
            round((1 - trimmed_duration / original_duration) * 100),
        )
        keep_result = True
        return trimmed_path
    except Exception as exc:  # noqa: BLE001 - trim is best-effort
        logger.debug("Cloud STT silence trim failed for %s: %s", file_path, exc)
        return None
    finally:
        if not keep_result:
            shutil.rmtree(work_dir, ignore_errors=True)


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


def _is_local_or_private_url(url: str) -> bool:
    """True for loopback/RFC-1918/LAN-internal hosts.

    Decides whether an empty ``stt.openai.api_key`` is acceptable: local
    OpenAI-compatible STT servers ignore the auth header, so users shouldn't
    need a sham ``api_key: not-needed``.
    """
    try:
        from urllib.parse import urlparse
        import ipaddress

        host = (urlparse(url).hostname or "").lower()
        if not host:
            return False
        if host == "localhost" or host.endswith((".local", ".lan", ".internal")):
            return True
        try:
            return ipaddress.ip_address(host).is_private or ipaddress.ip_address(host).is_loopback
        except ValueError:
            return False
    except Exception:
        return False


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


def _direct_openai_credentials(cfg_api_key: str, cfg_base_url: str) -> Optional[tuple[str, str]]:
    """Direct-credential ladder: config key > keyless local base_url > env key; None if none apply.

    A local OpenAI-compatible server needs no key — send a placeholder so the
    SDK doesn't refuse to construct a client.
    """
    if cfg_api_key:
        return cfg_api_key, (cfg_base_url or OPENAI_BASE_URL)
    if cfg_base_url and _is_local_or_private_url(cfg_base_url):
        return "not-needed", cfg_base_url
    direct_api_key = resolve_openai_audio_api_key()
    if direct_api_key:
        return direct_api_key, OPENAI_BASE_URL
    return None


def _resolve_openai_audio_client_config() -> tuple[str, str]:
    """Return ``(api_key, base_url)`` for the OpenAI STT client.

    Strict selection semantics on the stored ``stt`` provider string:
    - ``"nous"`` → managed gateway ONLY; unentitled/unreachable is a
      selection-naming error (a direct OPENAI_API_KEY must NOT override it).
    - any other stored provider → direct credentials ONLY; missing credentials
      is a selection-naming error — no silent managed fallback.
    - never-configured stt section → legacy ladder: direct credentials, then
      the managed gateway.
    """
    from tools.tool_backend_helpers import (
        NOUS_MANAGED_PROVIDER,
        read_selection,
        selection_error,
    )

    openai_cfg = _load_stt_config().get("openai") or {}
    cfg_api_key = openai_cfg.get("api_key", "")
    cfg_base_url = openai_cfg.get("base_url", "")

    selected = read_selection("stt")

    if selected == NOUS_MANAGED_PROVIDER:
        managed_gateway = resolve_managed_tool_gateway("openai-audio")
        if managed_gateway is None:
            raise ValueError(selection_error(
                "stt",
                NOUS_MANAGED_PROVIDER,
                "the Nous Tool Gateway is not available (not entitled or "
                "unreachable)",
            ))
        return managed_gateway.nous_user_token, urljoin(
            f"{managed_gateway.gateway_origin.rstrip('/')}/", "v1"
        )

    direct = _direct_openai_credentials(cfg_api_key, cfg_base_url)
    if direct is not None:
        return direct

    if selected is not None:
        raise ValueError(selection_error(
            "stt",
            selected,
            "neither stt.openai.api_key in config nor "
            "VOICE_TOOLS_OPENAI_KEY/OPENAI_API_KEY is set",
        ))

    managed_gateway = resolve_managed_tool_gateway("openai-audio")
    if managed_gateway is None:
        message = "Neither stt.openai.api_key in config nor VOICE_TOOLS_OPENAI_KEY/OPENAI_API_KEY is set"
        if managed_nous_tools_enabled():
            message += (
                ". "
                + nous_tool_gateway_unavailable_message(
                    "managed OpenAI audio for transcription",
                )
            )
        raise ValueError(message)

    return managed_gateway.nous_user_token, urljoin(
        f"{managed_gateway.gateway_origin.rstrip('/')}/", "v1"
    )


def _extract_transcript_text(transcription: Any) -> str:
    """Normalize text / object / dict transcription responses to a plain string."""
    if isinstance(transcription, str):
        text = transcription.strip()
    else:
        value = getattr(transcription, "text", None)
        if not isinstance(value, str) and isinstance(transcription, dict):
            value = transcription.get("text")
        text = value.strip() if isinstance(value, str) else str(transcription).strip()

    match = re.match(
        r"\s*language\s+[\w.-]+(?:\s*<audio_language>[^<]*</audio_language>)?\s*<asr_text>\s*(?P<text>.*)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match:
        text = match.group("text").strip()

    return text
