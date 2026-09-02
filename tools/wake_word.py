"""Wake-word ("Hey Hermes") detection — hands-free session trigger.

An always-on hotword listener shared by CLI, TUI and desktop GUI (one owns it,
gated by ``wake_surface_enabled``): on wake Hermes opens a fresh session and
captures voice via the existing pipeline. Engines (openwakeword default,
sherpa open-vocabulary, porcupine premium) are all on-device and live in
:mod:`tools.wake_word_engines`; this module owns config, the capture loop and
the process-wide listener singleton.

Capture reuses voice mode's 16 kHz mono int16 ``sounddevice`` path on a daemon
thread; callers ``pause()`` while a voice turn holds the mic and ``resume()``
once idle (two input streams on one device is unreliable cross-platform).
Nothing here touches agent context or the prompt cache — on wake the caller
gets a plain string, like a transcript.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from tools.wake_word_engines import (  # noqa: F401  (re-exported for callers/tests)
    _SHERPA_KWS_MODEL_DIR, _SHERPA_KWS_MODEL_URL, _Engine, _OpenWakeWordEngine, _PorcupineEngine,
    _SherpaKwsEngine, _ensure_sherpa_model, _looks_like_path, _sherpa_model_root,
)

logger = logging.getLogger(__name__)

# 16 kHz mono int16 — Whisper-native and what every engine expects.
SAMPLE_RATE = 16000

# Minimum gap between two wake fires, so one "hey hermes" can't retrigger
# across several frames while the caller is still reacting.
_FIRE_COOLDOWN_SECONDS = 2.0
_START_TIMEOUT_SECONDS = 5.0

# Ambient-speech rejection: require N consecutive over-threshold frames before
# firing (a stray phoneme spikes one frame; a real phrase holds several).
_DEFAULT_CONFIRMATION_FRAMES = 3

# Dead-mic detection: an int16 stream whose peak stays at/below _SILENCE_PEAK
# for this many consecutive seconds is flagged silent (desktop push-to-talk and
# the backend listener use different capture paths, so one can work while the
# backend-selected stream is all zeros).
_SILENCE_PEAK = 10
_SILENCE_ALERT_SECONDS = 10

# provider alias -> (engine class name on this module, lazy_deps feature).
# Unknown providers probe as openwakeword but fail to build.
_PROVIDERS: Dict[str, tuple[str, str]] = {
    "porcupine": ("_PorcupineEngine", "wake.porcupine"),
    **{k: ("_SherpaKwsEngine", "wake.sherpa") for k in ("sherpa", "sherpa-onnx", "kws", "open")},
    **{k: ("_OpenWakeWordEngine", "wake.openwakeword") for k in ("openwakeword", "oww", "local")},
}


class WakeWordInUse(RuntimeError):
    """Raised when another surface or process owns the wake-word listener."""


# ── Config ──

_DEFAULTS: Dict[str, Any] = {
    "enabled": False,
    "surface": "auto",
    "input_device": None,
    # Where PCM is captured: "local" (PortAudio on the backend host),
    # "client" (desktop/TUI streams int16 frames via wake.feed), or
    # "auto" (local when a device exists, else client capture).
    "capture": "auto",
    "provider": "openwakeword",
    "phrase": "hey hermes",
    "sensitivity": 0.6,
    "confirmation_frames": _DEFAULT_CONFIRMATION_FRAMES,
    "start_new_session": True,
}

# Bundled "hey hermes" model (tools/wakewords/) — the default. Config names in
# _ALIASES resolve to it, not to an openWakeWord built-in.
_BUNDLED_MODEL_NAME = "hey_hermes"
_BUNDLED_MODEL_ALIASES = frozenset({"", "hey_hermes", "hey hermes", "hermes"})


def _bundled_wakeword_path(framework: str = "onnx") -> str:
    """Path to the shipped hey_hermes model (.onnx/.tflite) for ``framework``."""
    ext = "tflite" if str(framework).strip().lower() == "tflite" else "onnx"
    return os.path.join(os.path.dirname(__file__), "wakewords", f"{_BUNDLED_MODEL_NAME}.{ext}")


def _is_macos_arm64() -> bool:
    import platform

    return sys.platform == "darwin" and platform.machine() == "arm64"


def default_inference_framework() -> str:
    """tflite on macOS ARM64, onnx elsewhere: openWakeWord's ONNX *embedding*
    model scores near-zero on Apple Silicon (upstream #336) — the detector arms
    but no phrase ever crosses threshold."""
    return "tflite" if _is_macos_arm64() else "onnx"


_warned_onnx_coerced = False


def resolve_inference_framework(cfg: Dict[str, Any]) -> str:
    """Effective openWakeWord backend: explicit ``openwakeword.inference_framework``
    or the platform default. The one provably dead combination — explicit
    ``onnx`` on macOS ARM64 (upstream #336) — is coerced to tflite with a
    one-time warning so a pre-fix pin doesn't keep a wake word that never fires.
    """
    global _warned_onnx_coerced

    sub = cfg.get("openwakeword") if isinstance(cfg.get("openwakeword"), dict) else {}
    framework = str(sub.get("inference_framework") or "").strip().lower()

    if not framework:
        return default_inference_framework()

    if framework == "onnx" and _is_macos_arm64():
        if not _warned_onnx_coerced:
            _warned_onnx_coerced = True
            logger.warning(
                "wake: openwakeword.inference_framework='onnx' is set but ONNX's "
                "embedding model never fires on macOS ARM64 (openWakeWord #336) — "
                "using tflite instead. Set inference_framework to '' (auto) or "
                "'tflite' in config.yaml to silence this."
            )
        return "tflite"

    return framework


def ensure_tflite_runtime() -> bool:
    """Make ``import tflite_runtime.interpreter`` resolve, returning success.

    openWakeWord hardcodes that import but only declares ``tflite-runtime`` on
    Linux; on macOS the equivalent wheel is ``ai-edge-litert``. Alias the
    module in-process (nothing is written to site-packages).
    """
    try:
        import tflite_runtime.interpreter  # noqa: F401

        return True
    except ImportError:
        pass

    try:
        from ai_edge_litert import interpreter as _litert  # type: ignore[import-not-found]
    except ImportError:
        return False

    import types

    pkg = types.ModuleType("tflite_runtime")
    pkg.__path__ = []  # type: ignore[attr-defined]  # mark as package
    sys.modules.setdefault("tflite_runtime", pkg)
    sys.modules["tflite_runtime.interpreter"] = _litert
    logger.debug("wake word: bridged tflite_runtime -> ai_edge_litert")
    return True


def load_wake_word_config() -> Dict[str, Any]:
    """Return the ``wake_word`` config section, shape-guarded to a dict."""
    try:
        from hermes_cli.config import load_config

        cfg = load_config().get("wake_word")
    except Exception:
        cfg = None
    return cfg if isinstance(cfg, dict) else {}


def _get(cfg: Dict[str, Any], key: str) -> Any:
    val = cfg.get(key, _DEFAULTS.get(key))
    return _DEFAULTS.get(key) if val is None else val


def _clamped(cfg: Dict[str, Any], key: str, cast, lo, hi):
    """Numeric config value via ``cast``, defaulting on junk, clamped to lo..hi."""
    try:
        n = cast(_get(cfg, key))
    except (TypeError, ValueError):
        n = cast(_DEFAULTS[key])
    return min(max(n, lo), hi)


def _provider(cfg: Dict[str, Any]) -> str:
    return str(_get(cfg, "provider")).strip().lower() or "openwakeword"


def _input_device(cfg: Dict[str, Any]) -> int | str | None:
    """Configured PortAudio input selector, preserving indices and names."""
    raw = _get(cfg, "input_device")
    if raw is None or isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return raw
    return str(raw).strip() or None


def _sensitivity(cfg: Dict[str, Any]) -> float:
    return _clamped(cfg, "sensitivity", float, 0.0, 1.0)


def _confirmation_frames(cfg: Dict[str, Any]) -> int:
    """Consecutive over-threshold frames required to fire, clamped 1..10.

    ``1`` restores single-frame behaviour; higher rejects ambient blips at the
    cost of a few tens of ms of latency.
    """
    return _clamped(cfg, "confirmation_frames", int, 1, 10)


def wake_phrase(cfg: Optional[Dict[str, Any]] = None) -> str:
    """Human-facing wake phrase label (purely cosmetic; engine keys detection)."""
    cfg = cfg if cfg is not None else load_wake_word_config()
    return str(_get(cfg, "phrase")) or "hey hermes"


def resolve_capture_mode(
    cfg: Optional[Dict[str, Any]] = None,
    *,
    prefer_client: bool = False,
    force_local: bool = False,
) -> str:
    """Return ``local`` or ``client`` capture mode for this arm.

    ``prefer_client`` is set by remote desktop; ``force_local`` keeps CLI/TUI on
    the process mic. Under ``auto`` a working backend input always wins (local
    desktops keep PortAudio + ``input_device``); client is the fallback only for
    a preferring surface with no usable backend mic — CLI/TUI stay local so
    status reports the real requirement rather than a path nothing will feed.
    """
    cfg = cfg if cfg is not None else load_wake_word_config()
    if force_local:
        return "local"
    raw = str(_get(cfg, "capture") or "auto").strip().lower()
    if raw in ("client", "remote", "external"):
        return "client"
    if raw == "local":
        return "local"
    if prefer_client and not _local_input_device_ready():
        return "client"
    return "local"


def _input_channels(info: Any) -> int:
    channels = info.get("max_input_channels") if isinstance(info, dict) else None
    if channels is None:
        channels = getattr(info, "max_input_channels", 0)
    return int(channels or 0)


def _local_input_device_ready() -> bool:
    """True when PortAudio is importable and at least one input device exists."""
    try:
        sd, _ = _import_audio()
    except (ImportError, OSError):
        return False
    try:
        devices = sd.query_devices()
        if isinstance(devices, dict):
            return _input_channels(devices) > 0
        if any(_input_channels(dev) > 0 for dev in devices):
            return True
        # Also accept a resolvable default input (some hosts list devices oddly).
        return _input_channels(sd.query_devices(None, "input")) > 0
    except Exception:
        return False


def wake_surface_enabled(surface: str, cfg: Optional[Dict[str, Any]] = None) -> bool:
    """Should ``surface`` (``cli`` / ``tui`` / ``gui``) host the listener?

    True when enabled and the configured ``surface`` is ``auto`` or this exact
    surface. ``auto`` only makes a surface eligible; the process/machine
    ownership lock still permits a single claimant.
    """
    cfg = cfg if cfg is not None else load_wake_word_config()
    if not cfg.get("enabled"):
        return False
    want = str(_get(cfg, "surface")).strip().lower() or "auto"
    return want == "auto" or want == surface.strip().lower()


# ── Multi-profile phrase enrollment (open-vocabulary routing) ──

def _active_profile_name() -> str:
    try:
        from hermes_cli.profiles import get_active_profile_name

        return get_active_profile_name() or "default"
    except Exception:
        return "default"


def enrolled_profile_phrases() -> Dict[str, str]:
    """Map ``profile name -> wake phrase`` for every wake-enabled profile.

    Reads each profile's own ``config.yaml`` raw (``load_config()`` targets only
    the ACTIVE profile). Enrolled = ``wake_word.enabled`` truthy; phrase defaults
    to ``"hey <profile>"``. The sherpa engine listens for all of them at once and
    routes the wake to the matching profile. Best-effort: unreadable skipped.
    """
    phrases: Dict[str, str] = {}
    try:
        from hermes_cli.config import read_user_config_raw
        from hermes_cli.profiles import get_profile_dir, list_profiles

        for info in list_profiles():
            name = getattr(info, "name", None) or str(info)
            try:
                raw = read_user_config_raw(Path(get_profile_dir(name)) / "config.yaml")
                wc = raw.get("wake_word") or {}
                if not isinstance(wc, dict) or not wc.get("enabled"):
                    continue
                phrase = str(wc.get("phrase") or f"hey {name}").strip()
                if phrase:
                    phrases[name] = phrase
            except Exception:
                continue
    except Exception:
        pass
    return phrases


# ── Audio capture (lazy — never import sounddevice at module load) ──

def _import_audio():
    import numpy as np
    import sounddevice as sd

    return sd, np


def _audio_available() -> bool:
    try:
        _import_audio()
        return True
    except (ImportError, OSError):
        return False


def _describe_input_device(sd, selector: int | str | None) -> Dict[str, Any]:
    """Resolve a PortAudio selector into JSON-safe diagnostics.

    Diagnostic only: ``InputStream`` remains the authority on whether the
    device can actually open at the requested format.
    """
    details: Dict[str, Any] = {"selector": selector}
    try:
        info = sd.query_devices(selector, "input")
    except Exception as e:
        details["error"] = str(e)
        return details
    if not isinstance(info, dict):
        return details

    if info.get("name"):
        details["name"] = str(info["name"])
    for key, out_key, cast in (
        ("max_input_channels", "max_input_channels", int),
        ("default_samplerate", "default_samplerate", float),
        ("hostapi", "hostapi_index", int),
    ):
        if isinstance(info.get(key), (int, float)):
            details[out_key] = cast(info[key])
    if "hostapi_index" in details:
        try:
            hostapi = sd.query_hostapis(details["hostapi_index"])
            hostapi_name = hostapi.get("name") if isinstance(hostapi, dict) else None
            if hostapi_name:
                details["hostapi"] = str(hostapi_name)
        except Exception:
            pass
    return details


def _device_label(details: Dict[str, Any]) -> str:
    name = str(details.get("name") or "").strip()
    selector = details.get("selector")
    label = name or ("system default" if selector is None else str(selector))
    hostapi = str(details.get("hostapi") or "").strip()
    return f"{label} ({hostapi})" if hostapi else label


def _capture_sample_rate(details: Dict[str, Any]) -> int:
    """Use the selected device's native rate when PortAudio reports one."""
    rate = details.get("default_samplerate")
    if isinstance(rate, (int, float)) and not isinstance(rate, bool) and rate > 0:
        try:
            return int(round(rate))
        except (OverflowError, ValueError):
            pass
    return SAMPLE_RATE


def _resample_audio_frame(np, frame, output_length: int):
    """Convert one native-rate capture block to an exact engine frame."""
    source = np.asarray(frame, dtype=np.float64).reshape(-1)
    if source.size == output_length:
        return np.asarray(frame, dtype=np.int16).reshape(-1)
    if source.size == 0:
        return np.zeros(output_length, dtype=np.int16)

    if source.size > output_length:
        # Average each source window when reducing (matches the desktop wake
        # capture path) so speech energy is retained instead of decimated.
        edges = np.linspace(0, source.size, output_length + 1, dtype=np.int64)
        values = np.add.reduceat(source, edges[:-1]) / np.diff(edges)
    else:
        # Unusual low-rate devices: interpolate up to the 16 kHz frame size.
        source_positions = np.arange(source.size, dtype=np.float64)
        target_positions = np.linspace(0, source.size - 1, output_length)
        values = np.interp(target_positions, source_positions, source)

    return np.rint(values).clip(-32768, 32767).astype(np.int16)


def silent_audio_hint(details: Dict[str, Any]) -> str:
    """Platform-specific remediation for an armed stream delivering silence."""
    if sys.platform == "darwin":
        return (
            "Microphone delivers only silence. Grant the Hermes backend "
            "microphone access in System Settings > Privacy & Security > "
            "Microphone, then toggle the wake word."
        )
    if sys.platform == "win32":
        return (
            f"Microphone delivers only silence from {_device_label(details)}. "
            "Set wake_word.input_device to a different PortAudio input device, "
            "then toggle the wake word."
        )
    return (
        f"Microphone delivers only silence from {_device_label(details)}. "
        "Check the selected input device, then toggle the wake word."
    )


# ── Engines (implementations live in tools.wake_word_engines) ──

def _build_engine(cfg: Dict[str, Any]) -> _Engine:
    provider = _provider(cfg)
    if provider not in _PROVIDERS:
        raise ValueError(f"Unknown wake_word provider: {provider!r}")
    return globals()[_PROVIDERS[provider][0]](cfg)


# ── Requirements probe (for /wake status + enable path) ──

def _stt_ready() -> bool:
    """Is a speech-to-text provider configured and enabled?

    A wake without STT arms the mic but every utterance dies at transcription.
    Same standard as voice mode's ``check_voice_requirements``.
    """
    try:
        from tools.transcription_tools import _get_provider, _load_stt_config, is_stt_enabled

        stt_config = _load_stt_config()
        return is_stt_enabled(stt_config) and _get_provider(stt_config) != "none"
    except Exception:
        return False


_LAZY_TTS_FEATURES = {"edge": "tts.edge", "elevenlabs": "tts.elevenlabs", "mistral": "tts.mistral"}


def _tts_ready() -> bool:
    """Can the configured TTS provider run (or install at first use)?

    PROBE, not an installer: ``check_tts_requirements`` lazily pip-installs the
    provider SDK, which froze wake.status polls for a whole pip run (a failed
    install unmounted the desktop ear). Uninstalled deps count as ready iff
    lazy installs are allowed; pip is never touched from here.
    """
    try:
        from tools.tts_tool import _get_provider, _load_tts_config

        provider = _get_provider(_load_tts_config())
    except Exception:
        return False

    feature = _LAZY_TTS_FEATURES.get(provider)
    if feature is not None:
        try:
            from tools import lazy_deps

            if not lazy_deps.is_available(feature):
                return lazy_deps._allow_lazy_installs()
        except Exception:
            return False

    try:
        from tools.tts_tool import check_tts_requirements

        return bool(check_tts_requirements())
    except Exception:
        return False


def check_wake_word_requirements(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Report whether wake-word detection can run, with a remediation hint."""
    cfg = cfg if cfg is not None else load_wake_word_config()
    provider = _provider(cfg)
    from tools import lazy_deps

    feature = _PROVIDERS.get(provider, ("", "wake.openwakeword"))[1]
    deps_ok = lazy_deps.is_available(feature)
    lazy_ok = lazy_deps._allow_lazy_installs()
    # The audio probe imports sounddevice + numpy — packages the lazy installer
    # would fetch — so only trust it once deps are installed; on a fresh install
    # the engine constructors' ``lazy_deps.ensure()`` + stream-open surface any
    # real audio problem (gating on the probe made lazy install unreachable).
    audio_ok = _audio_available() if deps_ok else False
    key_ok = True
    # Loop is wake → record → STT → agent → TTS; without either end the mic
    # hears you and nothing perceptible happens — refuse with a hint.
    stt_ok = _stt_ready()
    tts_ok = _tts_ready()
    hint = ""

    # tflite needs a runtime openWakeWord doesn't declare off Linux; report it
    # as a remediation instead of arming a detector that can't fire.
    tflite_ok = True
    if feature == "wake.openwakeword" and resolve_inference_framework(cfg) == "tflite":
        tflite_ok = ensure_tflite_runtime() or lazy_deps.is_available("wake.openwakeword.tflite") or lazy_ok

    if provider == "porcupine" and not (os.getenv("PORCUPINE_ACCESS_KEY") or "").strip():
        key_ok = False
        hint = "Set PORCUPINE_ACCESS_KEY (free key at https://console.picovoice.ai)."
    elif not deps_ok and not lazy_ok:
        hint = lazy_deps.feature_install_command(feature) or ""
    elif not tflite_ok:
        hint = "The wake word needs the tflite runtime on this Mac: pip install ai-edge-litert"
    elif deps_ok and not audio_ok and resolve_capture_mode(cfg) == "local":
        hint = "Microphone capture needs sounddevice + numpy and a working audio device."
    elif not stt_ok or not tts_ok:
        missing = " and ".join(
            name for name, ok in (("speech-to-text", stt_ok), ("text-to-speech", tts_ok)) if not ok
        )
        hint = (f"Wake word needs {missing} configured — run `hermes tools` "
                f"(Voice section) or see the voice-mode docs.")

    capture_mode = resolve_capture_mode(cfg)
    # Client capture needs deps (engine) but not a server-side PortAudio device.
    if capture_mode == "client":
        mic_ok = deps_ok or lazy_ok
    else:
        mic_ok = (deps_ok and audio_ok) or (not deps_ok and lazy_ok)
        if deps_ok and not audio_ok and not hint:
            hint = (
                "No local microphone on this backend. Remote desktop can stream "
                "the client mic — set wake_word.capture: client or use a desktop "
                "build with client-capture wake support."
            )

    return {
        "available": key_ok and stt_ok and tts_ok and tflite_ok and mic_ok,
        "provider": provider,
        "deps_available": deps_ok,
        "audio_available": audio_ok,
        "local_input_available": _local_input_device_ready() if deps_ok else False,
        "capture": capture_mode,
        "access_key_set": key_ok,
        "stt_available": stt_ok,
        "tts_available": tts_ok,
        "phrase": wake_phrase(cfg),
        "hint": hint,
    }


# ── Detector ──

@dataclass
class _Capture:
    """One armed audio source: a PortAudio stream (local) or the feed queue (client)."""

    stream: Any = None  # sounddevice.InputStream, None in client mode
    queue: Any = None  # client-capture frame queue, None in local mode
    np: Any = None
    rate: int = SAMPLE_RATE
    frame_length: int = 1280  # samples per read at ``rate``

    def read(self):
        """One raw block; None when no client frame arrived within 250 ms.
        Stream errors propagate."""
        if self.stream is not None:
            return self.stream.read(self.frame_length)[0]
        try:
            return self.queue.get(timeout=0.25)
        except Exception:
            return None

    def close(self) -> None:
        try:
            if self.stream is not None:
                self.stream.stop()
                self.stream.close()
        except Exception:
            pass


class WakeWordDetector:
    """Background hotword listener. Fires ``on_wake()`` when the phrase is heard.

    The engine is built once and kept alive across pause/resume; only the audio
    stream + reader thread cycle, so toggling the mic for a voice turn is cheap.
    """

    def __init__(self, engine: _Engine, on_wake: Callable[[], None],
                 cooldown: float = _FIRE_COOLDOWN_SECONDS,
                 on_failure: Optional[Callable[["WakeWordDetector"], None]] = None,
                 input_device: int | str | None = None,
                 external_audio: bool = False):
        self.engine = engine
        self.on_wake = on_wake
        self.cooldown = cooldown
        self.on_failure = on_failure
        self.input_device = input_device
        self.external_audio = bool(external_audio)
        self.input_device_details: Dict[str, Any] = (
            {"selector": "client", "name": "client capture", "hostapi": "remote"}
            if self.external_audio
            else {"selector": input_device}
        )
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._callback_inflight = threading.Event()
        self._last_fire = 0.0
        self._lock = threading.Lock()
        # Client-capture PCM queue (int16 mono frames). Local mode ignores this.
        import queue as _queue

        self._audio_q: "_queue.Queue[Any]" = _queue.Queue(maxsize=64)
        # True when the stream is open but every frame is (near-)silence, so
        # status surfaces can tell "armed" from "deaf".
        self.audio_silent = False
        self._silent_frames = 0

    @property
    def running(self) -> bool:
        t = self._thread
        return t is not None and t.is_alive()

    def feed(self, pcm_int16) -> None:
        """Enqueue one int16 mono frame (or raw bytes) for client capture.

        Short frames are zero-padded to ``engine.frame_length``; long frames are
        split. On queue overflow the oldest frame is dropped to stay real-time.
        """
        if not self.external_audio:
            return
        try:
            import numpy as np
        except Exception:
            return
        if isinstance(pcm_int16, (bytes, bytearray, memoryview)):
            arr = np.frombuffer(pcm_int16, dtype=np.int16)
        else:
            arr = np.asarray(pcm_int16, dtype=np.int16).reshape(-1)
        fl = int(self.engine.frame_length)
        if fl <= 0:
            return
        for offset in range(0, int(arr.shape[0]), fl):
            chunk = arr[offset : offset + fl]
            if chunk.shape[0] < fl:
                pad = np.zeros(fl, dtype=np.int16)
                pad[: chunk.shape[0]] = chunk
                chunk = pad
            try:
                self._audio_q.put_nowait(chunk)
            except Exception:
                try:  # full: drop the oldest frame, then retry once
                    self._audio_q.get_nowait()
                    self._audio_q.put_nowait(chunk)
                except Exception:
                    pass

    def start(self) -> None:
        """Open the mic (or client feeder) and begin listening. Idempotent."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            ready = threading.Event()
            startup_errors: list[BaseException] = []
            self._thread = threading.Thread(
                target=self._run,
                args=(ready, startup_errors),
                daemon=True,
                name="wake-word",
            )
            self._thread.start()
        if not ready.wait(_START_TIMEOUT_SECONDS):
            self._halt_thread()
            raise TimeoutError("Timed out while opening the wake-word microphone.")
        if startup_errors:
            self._halt_thread()
            raise RuntimeError("Failed to open the wake-word microphone.") from startup_errors[0]

    # pause/resume keep the engine; stop tears it down.
    def pause(self) -> None:
        self._halt_thread()

    def resume(self) -> None:
        self.start()

    def stop(self) -> None:
        self._halt_thread()
        self.engine.close()

    def _halt_thread(self) -> None:
        with self._lock:
            self._stop.set()
            t = self._thread
            if t is not None and t is not threading.current_thread():
                t.join(timeout=2.0)
            if self._thread is t:
                self._thread = None

    def _dispatch_wake(self) -> None:
        try:
            self.on_wake()
        except Exception as e:
            logger.warning("wake word callback failed: %s", e)
        finally:
            self._callback_inflight.clear()

    def _open_capture(self, frame_length: int) -> _Capture:
        """Open the audio source; raises on any local-mic failure."""
        if self.external_audio:
            # Drain any stale frames from a previous arm.
            try:
                while True:
                    self._audio_q.get_nowait()
            except Exception:
                pass
            logger.info(
                "wake word: client-capture mode (frame=%d, rate=%d) — waiting for wake.feed",
                frame_length, SAMPLE_RATE,
            )
            return _Capture(queue=self._audio_q, frame_length=frame_length)

        try:
            sd, np = _import_audio()
        except (ImportError, OSError) as e:
            logger.error("wake word: audio libraries unavailable: %s", e)
            raise

        self.input_device_details = _describe_input_device(sd, self.input_device)
        cap = _Capture(np=np, rate=_capture_sample_rate(self.input_device_details))
        cap.frame_length = max(1, int(round(frame_length * cap.rate / SAMPLE_RATE)))
        logger.info(
            "wake word: opening microphone device=%s selector=%r hostapi=%s "
            "default_rate=%s capture_rate=%d engine_rate=%d",
            self.input_device_details.get("name") or "system default",
            self.input_device,
            self.input_device_details.get("hostapi") or "unknown",
            self.input_device_details.get("default_samplerate") or "unknown",
            cap.rate,
            SAMPLE_RATE,
        )
        try:
            cap.stream = sd.InputStream(
                device=self.input_device,
                samplerate=cap.rate,
                channels=1,
                dtype="int16",
                blocksize=cap.frame_length,
            )
            cap.stream.start()
        except Exception as e:
            logger.error("wake word: failed to open microphone: %s", e)
            raise
        return cap

    def _note_silence(self, frame, silent_alert_frames: int) -> None:
        """Track consecutive near-zero frames; flag/unflag ``audio_silent``."""
        try:
            peak = int(abs(frame).max()) if len(frame) else 0
        except Exception:
            peak = _SILENCE_PEAK + 1
        if peak <= _SILENCE_PEAK:
            self._silent_frames += 1
            if self._silent_frames == silent_alert_frames:
                self.audio_silent = True
                logger.warning(
                    "wake word: mic delivers only silence (peak<=%d for %ds); %s",
                    _SILENCE_PEAK, _SILENCE_ALERT_SECONDS,
                    silent_audio_hint(self.input_device_details),
                )
        elif self._silent_frames:
            if self.audio_silent:
                logger.info("wake word: mic audio detected — stream healthy")
            self._silent_frames = 0
            self.audio_silent = False

    def _fire(self) -> None:
        """Honor the cooldown, then run ``on_wake`` on its own thread (once)."""
        now = time.monotonic()
        if now - self._last_fire < self.cooldown:
            logger.debug("wake word: detection within cooldown — ignored")
            return
        self._last_fire = now
        logger.info("wake word: phrase detected — firing callback")
        if not self._callback_inflight.is_set():
            self._callback_inflight.set()
            threading.Thread(target=self._dispatch_wake, daemon=True, name="wake-word-callback").start()

    def _run(self, ready: threading.Event,
             startup_errors: list[BaseException]) -> None:
        frame_length = self.engine.frame_length
        try:
            cap = self._open_capture(frame_length)
        except Exception as e:
            startup_errors.append(e)
            ready.set()
            return

        # Drop buffered audio/feature state so a resume right after a voice turn
        # can't re-fire on audio captured before the pause (the wake → voice →
        # resume → wake runaway loop).
        try:
            self.engine.reset()
        except Exception:
            pass

        logger.info("wake word: listening (frame=%d, rate=%d, external=%s)",
                    frame_length, SAMPLE_RATE, self.external_audio)
        ready.set()
        failed = False
        silent_alert_frames = max(1, int(_SILENCE_ALERT_SECONDS * SAMPLE_RATE / max(1, frame_length)))
        try:
            while not self._stop.is_set():
                try:
                    data = cap.read()
                except Exception as e:
                    logger.warning("wake word: stream read error: %s", e)
                    failed = not self._stop.is_set()
                    break
                if data is None:
                    # No client frames yet — count as silence for status.
                    self._silent_frames += 1
                    if self._silent_frames == silent_alert_frames:
                        self.audio_silent = True
                    continue
                frame = data[:, 0] if getattr(data, "ndim", 1) == 2 else data
                if cap.rate != SAMPLE_RATE:
                    frame = _resample_audio_frame(cap.np, frame, frame_length)
                self._note_silence(frame, silent_alert_frames)
                try:
                    fired = self.engine.process(frame)
                except Exception as e:
                    logger.debug("wake word: engine error: %s", e)
                    continue
                if fired:
                    self._fire()
        finally:
            cap.close()
            logger.info("wake word: stream closed")
            if failed and self.on_failure is not None:
                self.on_failure(self)


# ── Process-wide singleton (mirrors hermes_cli.voice's continuous API) ──

_detector: Optional[WakeWordDetector] = None
_detector_owner: object | None = None
_detector_file_lock = None
_detector_lock = threading.Lock()


def _lock_path() -> Path:
    from hermes_constants import get_default_hermes_root

    return get_default_hermes_root() / "runtime" / "wake-word.lock"


def _flock(handle, acquire: bool) -> None:
    """Non-blocking exclusive lock (or unlock) of one byte / whole file, per OS."""
    if os.name == "nt":
        import msvcrt

        if acquire:  # msvcrt needs at least one byte to lock
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK if acquire else msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), (fcntl.LOCK_EX | fcntl.LOCK_NB) if acquire else fcntl.LOCK_UN)


def _acquire_machine_lock(path: Optional[Path] = None):
    """Acquire the cross-process microphone lease, or raise WakeWordInUse."""
    lock_path = path or _lock_path()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(lock_path, "a+b")
    try:
        _flock(handle, True)
    except (OSError, BlockingIOError) as e:
        handle.close()
        raise WakeWordInUse("Wake-word microphone is already owned.") from e
    return handle


def _release_machine_lock(handle) -> None:
    if handle is None:
        return
    try:
        _flock(handle, False)
    except OSError:
        pass
    finally:
        handle.close()


def _clear_singleton_locked() -> tuple[Optional[WakeWordDetector], Any]:
    """Forget the armed detector (caller holds ``_detector_lock``); returns (detector, lock handle)."""
    global _detector, _detector_owner, _detector_file_lock
    det, handle = _detector, _detector_file_lock
    _detector = None
    _detector_owner = None
    _detector_file_lock = None
    return det, handle


def _owned_detector(owner: object) -> Optional[WakeWordDetector]:
    """The armed detector iff ``owner`` holds the lease (caller holds the lock)."""
    return _detector if _detector is not None and _detector_owner is owner else None


def _detector_failed(detector: WakeWordDetector) -> None:
    """Release ownership if the active microphone stream dies unexpectedly."""
    with _detector_lock:
        if _detector is not detector:
            return
        _, lock_handle = _clear_singleton_locked()
        try:
            detector.engine.close()
        finally:
            _release_machine_lock(lock_handle)


def start_listening(
    on_wake: Callable[[], None],
    *,
    owner: object,
    config: Optional[Dict[str, Any]] = None,
    external_audio: bool = False,
) -> WakeWordDetector:
    """Claim, build, and start the detector. Idempotent for the same owner.

    Raises if engine construction fails (missing deps / access key / model);
    callers should probe :func:`check_wake_word_requirements` first. A different
    owner, including another process, receives :class:`WakeWordInUse`.
    """
    if owner is None:
        raise ValueError("wake-word owner must not be None")

    global _detector, _detector_owner, _detector_file_lock
    with _detector_lock:
        if _detector is not None:
            if _detector_owner is not owner:
                raise WakeWordInUse("Wake-word microphone is already owned.")
            _detector.on_wake = on_wake
            _detector.resume()
            return _detector
        lock_handle = _acquire_machine_lock()
        try:
            cfg = config if config is not None else load_wake_word_config()
            engine = _build_engine(cfg)
            detector = WakeWordDetector(
                engine,
                on_wake,
                on_failure=_detector_failed,
                input_device=_input_device(cfg),
                external_audio=external_audio,
            )
            _detector = detector
            _detector_owner = owner
            _detector_file_lock = lock_handle
            detector.start()
            return detector
        except Exception:
            det, _ = _clear_singleton_locked()
            try:
                if det is not None:
                    det.stop()
            except Exception:
                pass
            _release_machine_lock(lock_handle)
            raise


def owns_listener(owner: object) -> bool:
    with _detector_lock:
        return _owned_detector(owner) is not None


def _owned_call(owner: object, method: str) -> bool:
    with _detector_lock:
        det = _owned_detector(owner)
        if det is None:
            return False
        getattr(det, method)()
        return True


def pause_listening(*, owner: object) -> bool:
    """Release the microphone only when ``owner`` holds the lease."""
    return _owned_call(owner, "pause")


def resume_listening(*, owner: object) -> bool:
    """Re-open the microphone only when ``owner`` holds the lease."""
    return _owned_call(owner, "resume")


def stop_listening(*, owner: object) -> bool:
    """Fully stop the detector only when ``owner`` holds the lease."""
    with _detector_lock:
        if _owned_detector(owner) is None:
            return False
        det, lock_handle = _clear_singleton_locked()
        try:
            det.stop()
        finally:
            _release_machine_lock(lock_handle)
        return True


def _current_detector() -> Optional[WakeWordDetector]:
    with _detector_lock:
        return _detector


def is_listening() -> bool:
    det = _current_detector()
    return det is not None and det.running


def audio_is_silent() -> bool:
    """True when the armed stream has delivered only silence (dead mic).

    The stream opens fine but every frame is zeros, so detection can never
    fire; status surfaces show "listening but the microphone appears silent".
    """
    det = _current_detector()
    return det is not None and det.audio_silent


def get_input_device_status(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return configured/active PortAudio input diagnostics for status UIs."""
    det = _current_detector()
    if det is not None:
        return dict(det.input_device_details)

    cfg = cfg if cfg is not None else load_wake_word_config()
    selector = _input_device(cfg)
    try:
        sd, _ = _import_audio()
    except (ImportError, OSError) as e:
        return {"selector": selector, "error": str(e)}
    return _describe_input_device(sd, selector)


def get_last_match() -> Optional[tuple[str, str]]:
    """(matched phrase, profile) of the most recent wake fire, if the engine
    reports per-phrase matches (sherpa multi-profile routing). None otherwise."""
    det = _current_detector()
    return None if det is None else getattr(det.engine, "last_match", None)


def feed_audio(*, owner: object, pcm_int16) -> bool:
    """Push client-captured PCM into the armed detector (client capture mode).

    Returns True when the frame was accepted for ``owner``'s armed detector.
    """
    with _detector_lock:
        det = _owned_detector(owner)
        if det is None or not det.external_audio:
            return False
    det.feed(pcm_int16)
    return True


def detector_frame_info() -> Dict[str, Any]:
    """Sample rate + frame length for client capture streamers."""
    det = _current_detector()
    if det is None:
        return {"sample_rate": SAMPLE_RATE, "frame_length": 1280}
    return {
        "sample_rate": SAMPLE_RATE,
        "frame_length": int(getattr(det.engine, "frame_length", 1280) or 1280),
        "external_audio": bool(det.external_audio),
    }
