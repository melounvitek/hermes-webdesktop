"""Voice Mode -- push-to-talk recording and playback for the CLI.

Audio capture via sounddevice, WAV encoding via stdlib wave, STT dispatch via
tools.transcription_tools, TTS playback via sounddevice or system players.
Optional deps: ``pip install sounddevice numpy`` (or ``uv sync --extra voice``).
"""

import logging
import math
import os
import platform
import shlex
import shutil
import subprocess
import sys
from collections import deque
from pathlib import Path
import tempfile
import threading
import time
import wave
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

from tools.voice_mode_transcript import (  # noqa: F401 - re-exported; tests patch tools.voice_mode.<name>
    _voice_config,
    WHISPER_HALLUCINATIONS,
    _HALLUCINATION_REPEAT_RE,
    is_whisper_hallucination,
    DEFAULT_VOICE_STOP_PHRASES,
    _load_voice_stop_phrases,
    is_voice_stop_phrase,
    DEFAULT_TTS_ECHO_SIMILARITY_THRESHOLD,
    MIN_FRAGMENT_LENGTH_FOR_ECHO,
    _normalize_for_echo_compare,
    is_tts_echo,
    voice_stop_hint,
)
from hermes_constants import is_termux as _is_termux_environment

# ── Recording parameters ──
SAMPLE_RATE = 16000  # Whisper native rate
CHANNELS = 1
DTYPE = "int16"
SAMPLE_WIDTH = 2  # bytes per sample (int16)
SILENCE_RMS_THRESHOLD = 200  # RMS below this = silence (int16 range 0-32767)
SILENCE_DURATION_SECONDS = 3.0  # continuous silence before auto-stop
_TEMP_DIR = os.path.join(tempfile.gettempdir(), "hermes_voice")


# ── Lazy audio imports ──
# Never imported at module level: crashes headless environments (SSH, Docker,
# WSL, no PortAudio).

def _import_audio():
    """Lazy-import (sounddevice, numpy); raises ImportError/OSError when unavailable."""
    import sounddevice as sd
    import numpy as np
    return sd, np


def _sounddevice_output_allowed() -> bool:
    """Whether sounddevice may be used for audio OUTPUT.

    False on macOS: initializing PortAudio/CoreAudio for output triggers a
    kTCCServiceMediaLibrary prompt, so all output goes through ``afplay`` there.
    Does NOT affect input (recording), which legitimately needs mic permission.
    """
    return platform.system() != "Darwin"


def _play_int16_via_tempfile(audio, sample_rate: int) -> None:
    """Play int16 mono PCM via a temp WAV + play_audio_file (macOS: afplay, no TCC prompt)."""
    tmp_path = None
    try:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_path = tmp.name
        _write_wav_frames(tmp, audio.tobytes(), sample_rate)
        play_audio_file(tmp_path)
    except Exception as e:
        logger.debug("Tone tempfile playback failed: %s", e)
    finally:
        _unlink_quietly(tmp_path)


def _write_wav_frames(dest, frames: bytes, sample_rate: int) -> None:
    """Write raw 16-bit mono PCM *frames* as a WAV to *dest* (path or file object)."""
    with wave.open(dest, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(sample_rate)
        wf.writeframes(frames)


def _unlink_quietly(path: Optional[str]) -> None:
    """Best-effort unlink; missing/undeletable files are ignored."""
    if not path:
        return
    try:
        os.unlink(path)
    except OSError:
        pass


def _audio_available() -> bool:
    try:
        _import_audio()
        return True
    except (ImportError, OSError):
        return False


def _rms(np, data) -> float:
    return float(np.sqrt(np.mean(data.astype(np.float64) ** 2)))


def _default_input_samplerate(sd) -> int:
    """Default input device rate, else the Whisper-friendly SAMPLE_RATE."""
    try:
        info = sd.query_devices(None, "input")
        rate = info.get("default_samplerate") if isinstance(info, dict) else getattr(info, "default_samplerate", None)
        if isinstance(rate, (int, float)) and rate > 0:
            return int(round(rate))
    except Exception:
        pass
    return SAMPLE_RATE


# ── Environment detection ──
def _voice_capture_install_hint() -> str:
    if _is_termux_environment():
        return "pkg install python-numpy portaudio && python -m pip install sounddevice"
    # Inside a venv a bare `pip install` may hit whichever Python the shell
    # resolves first (macOS: often a Rosetta system Python) — use the venv's pip.
    try:
        if sys.prefix != getattr(sys, "base_prefix", sys.prefix):
            pip_in_venv = Path(sys.prefix) / "bin" / "pip"
            if pip_in_venv.exists():
                return f"{pip_in_venv} install sounddevice numpy"
    except Exception:
        pass
    return "pip install sounddevice numpy"


def _portaudio_missing_message() -> str:
    """sounddevice imports but PortAudio's .so is missing — pip can't fix that."""
    if _is_termux_environment():
        hint = "  Termux: pkg install portaudio"
    else:
        hint = "  Linux:  sudo apt-get install libportaudio2\n  macOS:  brew install portaudio"
    return f"PortAudio system library not found -- install it first:\n{hint}\nThen retry /voice on."


_TERMUX_APP_MISSING_WARNING = (
    "Termux:API Android app is not installed. Install/update the Termux:API app to use termux-microphone-record."
)


def _termux_microphone_command() -> Optional[str]:
    if not _is_termux_environment():
        return None
    return shutil.which("termux-microphone-record")


def _run_quiet(cmd: List[str], *, timeout: float, check: bool) -> subprocess.CompletedProcess:
    """subprocess.run with captured, utf-8-decoded output and no stdin."""
    return subprocess.run(
        cmd, capture_output=True, text=True, encoding='utf-8', errors='replace',
        timeout=timeout, check=check, stdin=subprocess.DEVNULL)


# `pm list packages` is canonical, but on some ROMs `pm` isn't on Termux's PATH
# while `cmd package` is, and on others `pm` returns nothing for the calling
# user even when the app is present — so both are tried.
_TERMUX_API_PACKAGE_PROBES = (
    ("pm", "list", "packages", "com.termux.api"),
    ("cmd", "package", "list", "packages", "com.termux.api"),
)


def _termux_api_app_installed() -> bool:
    """True iff the Termux:API Android app is installed.

    Any probe reporting ``package:com.termux.api`` is authoritative. If EVERY
    probe is inconclusive (binary missing, denied, timeout, non-zero exit) we
    trust the ``termux-microphone-record`` binary on PATH instead: a false
    negative blocks ``/voice on`` outright, a false positive only surfaces a
    precise runtime error. One clean probe without the package = genuinely missing.
    """
    if not _is_termux_environment():
        return False
    inconclusive = False
    for cmd in _TERMUX_API_PACKAGE_PROBES:
        try:
            result = _run_quiet(list(cmd), timeout=5, check=False)
        except (OSError, subprocess.TimeoutExpired):
            inconclusive = True
            continue
        if result.returncode != 0:
            inconclusive = True
            continue
        if "package:com.termux.api" in (result.stdout or "").lower():
            return True
    if inconclusive and shutil.which("termux-microphone-record") is not None:
        logger.debug(
            "Termux package-manager probes inconclusive; trusting "
            "termux-microphone-record binary on PATH (issue #31015).")
        return True
    return False


def _termux_voice_capture_available() -> bool:
    return _termux_microphone_command() is not None and _termux_api_app_installed()


def _pulse_socket_candidates() -> List[str]:
    """Socket paths a PulseAudio/PipeWire client would try by default."""
    candidates: List[str] = []
    # PULSE_SERVER may be "unix:/path", "unix:/path;..." or a bare path.
    for part in os.environ.get('PULSE_SERVER', '').split(';'):
        part = part.strip()
        if part.startswith('unix:'):
            candidates.append(part[len('unix:'):])
    pulse_runtime = os.environ.get('PULSE_RUNTIME_PATH')
    if pulse_runtime:
        candidates.append(os.path.join(pulse_runtime, 'native'))
    xdg_runtime = os.environ.get('XDG_RUNTIME_DIR')
    if xdg_runtime:
        candidates.append(os.path.join(xdg_runtime, 'pulse', 'native'))
        candidates.append(os.path.join(xdg_runtime, 'pipewire-0'))
    return [c for c in candidates if c]


def _pulse_socket_reachable() -> bool:
    """True if a PulseAudio/PipeWire socket on disk accepts a connection.

    Covers a sound server running locally (e.g. a remote SSH host) without
    ``PULSE_SERVER``/``PIPEWIRE_REMOTE`` set; a stale socket of a dead server does not count.
    """
    import socket
    import stat
    for path in _pulse_socket_candidates():
        try:
            if not stat.S_ISSOCK(os.stat(path).st_mode):
                continue
        except OSError:
            continue
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.settimeout(0.5)
            sock.connect(path)
            return True
        except OSError:
            continue
        finally:
            sock.close()
    return False


def _probe_audio_libraries(
    warnings: List[str], notices: List[str], *,
    has_forwarded_audio: bool, termux_mic_cmd: Optional[str], termux_app_installed: bool,
) -> None:
    """Import sounddevice and query devices; append the outcome to warnings/notices.

    Host audio forwarding or Termux:API capture downgrade "no devices" /
    "query failed" to notices — in WSL with PulseAudio device queries can fail
    even though recording/playback works fine.
    """
    termux_capture = bool(termux_mic_cmd and termux_app_installed)

    def outcome(termux_notice: str, warning: str, *, forwarded_notice: str = "", import_failed: bool = False):
        if forwarded_notice and has_forwarded_audio:
            notices.append(forwarded_notice)
        elif termux_capture:
            notices.append(termux_notice)
        elif import_failed and termux_mic_cmd and not termux_app_installed:
            warnings.append(_TERMUX_APP_MISSING_WARNING)
        else:
            warnings.append(warning)

    try:
        sd, _ = _import_audio()
    except ImportError:
        return outcome("Termux:API microphone recording available (sounddevice not required)",
                       f"Audio libraries not installed ({_voice_capture_install_hint()})", import_failed=True)
    except OSError:
        return outcome("Termux:API microphone recording available (PortAudio not required)",
                       _portaudio_missing_message(), import_failed=True)
    try:
        if sd.query_devices():
            return
        outcome("No PortAudio devices detected, but Termux:API microphone capture is available",
                "No audio input/output devices detected",
                forwarded_notice="No PortAudio devices detected but host audio forwarding is configured -- continuing")
    except Exception:
        outcome("PortAudio device query failed, but Termux:API microphone capture is available",
                "Audio subsystem error (PortAudio cannot query devices)",
                forwarded_notice="Audio device query failed but host audio forwarding is configured -- continuing")


def detect_audio_environment() -> dict:
    """Detect whether the current environment supports audio I/O.

    Returns ``{'available', 'warnings' (hard-fail, block voice mode),
    'notices' (informational)}``. SSH, containers and WSL normally have no
    audio devices, but a reachable sound server (PulseAudio/PipeWire socket or
    forwarding env vars) is honored.
    """
    warnings: List[str] = []
    notices: List[str] = []
    termux_mic_cmd = _termux_microphone_command()
    termux_app_installed = _termux_api_app_installed()
    has_forwarded_audio = bool(
        os.environ.get('PULSE_SERVER') or os.environ.get('PIPEWIRE_REMOTE') or _pulse_socket_reachable())

    def report(notice: str, warning: str) -> None:
        (notices if has_forwarded_audio else warnings).append(notice if has_forwarded_audio else warning)

    if any(os.environ.get(v) for v in ('SSH_CLIENT', 'SSH_TTY', 'SSH_CONNECTION')):
        report("Running over SSH with a reachable PulseAudio/PipeWire sound server",
               "Running over SSH -- no audio devices available.\n"
               "  If a sound server (PulseAudio/PipeWire) is running on this host,\n"
               "  point Hermes at it, e.g.:\n"
               "    export XDG_RUNTIME_DIR=/run/user/$(id -u)\n"
               "    # or: export PULSE_SERVER=unix:$XDG_RUNTIME_DIR/pulse/native")

    from hermes_constants import is_container
    if is_container():
        report("Running inside container (Docker/Podman/LXC) with host audio forwarding",
               "Running inside container (Docker/Podman/LXC) -- no audio devices.\n"
               "  Forward host audio with one of (substitute $XDG_RUNTIME_DIR for your runtime dir,\n"
               "  typically /run/user/$UID):\n"
               "    PulseAudio:  -v $XDG_RUNTIME_DIR/pulse/native:$XDG_RUNTIME_DIR/pulse/native \\\n"
               "                 -e PULSE_SERVER=unix:$XDG_RUNTIME_DIR/pulse/native\n"
               "    PipeWire:    -e PIPEWIRE_REMOTE=$XDG_RUNTIME_DIR/pipewire-0")

    # WSL: the PowerShell/Media.SoundPlayer fallback only covers OUTPUT, so when
    # it is all that's available downgrade to a notice (recording guidance stays
    # visible, TTS-only usage isn't blocked).
    if _is_wsl2_env():
        if has_forwarded_audio:
            notices.append("Running in WSL with a reachable PulseAudio/PipeWire sound server")
        elif _wsl_powershell_tts_available():
            notices.append(
                "Running in WSL without a PulseAudio bridge -- TTS playback "
                "will use the PowerShell/Media.SoundPlayer fallback. "
                "Voice INPUT (recording) still requires a PulseAudio bridge:\n"
                "  1. Set PULSE_SERVER=unix:/mnt/wslg/PulseServer\n"
                "  2. Create ~/.asoundrc pointing ALSA at PulseAudio\n"
                "  3. Verify with: arecord -d 3 /tmp/test.wav && aplay /tmp/test.wav")
        else:
            warnings.append(
                "Running in WSL -- audio requires a forwarded sound server.\n"
                "  PulseAudio: export PULSE_SERVER=unix:/mnt/wslg/PulseServer\n"
                "  PipeWire:   export PIPEWIRE_REMOTE=$XDG_RUNTIME_DIR/pipewire-0\n"
                "  Then verify: arecord -d 3 /tmp/test.wav && aplay /tmp/test.wav")

    _probe_audio_libraries(
        warnings, notices, has_forwarded_audio=has_forwarded_audio,
        termux_mic_cmd=termux_mic_cmd, termux_app_installed=termux_app_installed)
    return {"available": not warnings, "warnings": warnings, "notices": notices}


# ── Audio cues (beep tones) ──
_DEFAULT_BEEP_VOLUME = 0.3


def _get_beep_volume() -> float:
    """``voice.beep_volume`` clamped to 0.0-1.0; 0.3 when missing/invalid."""
    raw = _voice_config().get("beep_volume", _DEFAULT_BEEP_VOLUME)
    try:
        volume = float(raw)
    except (TypeError, ValueError):
        return _DEFAULT_BEEP_VOLUME
    if isinstance(raw, bool) or volume < 0.0 or volume > 1.0 or math.isnan(volume):
        return _DEFAULT_BEEP_VOLUME
    return volume


def _sd_play_blocking(sd, audio, sample_rate: int, *, timeout: float, blocksize: int = 0) -> None:
    """``sd.play`` then poll until idle or *timeout*.

    ``sd.wait()`` has no timeout and hangs forever if the device stalls.
    """
    sd.play(audio, samplerate=sample_rate, blocksize=blocksize)
    deadline = time.monotonic() + timeout
    while sd.get_stream() and sd.get_stream().active and time.monotonic() < deadline:
        time.sleep(0.01)
    sd.stop()


def play_beep(frequency: int = 880, duration: float = 0.12, count: int = 1) -> None:
    """Play *count* short beeps of *frequency* Hz, *duration* s each.

    Synthesized with numpy only (no sounddevice import on the synthesis step,
    so no macOS TCC prompt); on macOS output goes through afplay.
    """
    try:
        import numpy as np
    except ImportError:
        return
    try:
        gap = 0.06  # seconds between beeps
        samples_per_beep = int(SAMPLE_RATE * duration)
        samples_per_gap = int(SAMPLE_RATE * gap)
        beep_volume = _get_beep_volume()
        parts = []
        for i in range(count):
            t = np.linspace(0, duration, samples_per_beep, endpoint=False)
            tone = np.sin(2 * np.pi * frequency * t)
            fade_len = min(int(SAMPLE_RATE * 0.01), samples_per_beep // 4)  # avoid clicks
            tone[:fade_len] *= np.linspace(0, 1, fade_len)
            tone[-fade_len:] *= np.linspace(1, 0, fade_len)
            parts.append((tone * beep_volume * 32767).astype(np.int16))
            if i < count - 1:
                parts.append(np.zeros(samples_per_gap, dtype=np.int16))
        audio = np.concatenate(parts)
        if not _sounddevice_output_allowed():
            _play_int16_via_tempfile(audio, SAMPLE_RATE)
            return
        try:
            sd, _ = _import_audio()
        except (ImportError, OSError):
            return
        _sd_play_blocking(sd, audio, SAMPLE_RATE, timeout=2.0)
    except Exception as e:
        logger.debug("Beep playback failed: %s", e)


# ── Thinking sound — calm ambient "blub blub" while the agent works ──
# Minutes of silent tool use reads as "it died"; a quiet pair of water-bubble
# blips fills the gap. Scaled by voice.beep_volume, gated by voice.thinking_sound
# (default on). The host's *should_play* callback decides when blips are allowed;
# the output ref-count below tells hosts when real audio is actually flowing.

_audio_output_active_count = 0
_audio_output_lock = threading.Lock()


def mark_audio_output_active(active: bool) -> None:
    """Reference-count real audio output (TTS/file playback).

    Playback paths bracket their work with ``(True)`` / ``(False)`` so
    ``is_audio_output_active()`` reflects speech leaving the speakers RIGHT NOW —
    unlike per-turn TTS-done events, which stay 'busy' while waiting for text.
    """
    global _audio_output_active_count
    with _audio_output_lock:
        _audio_output_active_count = max(0, _audio_output_active_count + (1 if active else -1))


def is_audio_output_active() -> bool:
    """True while TTS/file audio is actually playing on the speakers."""
    with _audio_output_lock:
        return _audio_output_active_count > 0


_thinking_lock = threading.Lock()
_thinking_stop: Optional[threading.Event] = None


def thinking_sound_enabled() -> bool:
    """Config gate: ``voice.thinking_sound`` (default True)."""
    try:
        from utils import is_truthy_value
        return is_truthy_value(_voice_config().get("thinking_sound", True), default=True)
    except Exception:
        return True


def _synth_thinking_blip(np, frequency: float) -> "Any":
    """One soft 'blub': short sine with a downward glide and a click-free envelope."""
    duration = 0.16
    n = int(SAMPLE_RATE * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    glide = np.linspace(1.0, 0.72, n)  # water-drop feel: freq → 0.72*freq
    phase = 2 * np.pi * np.cumsum(frequency * glide) / SAMPLE_RATE
    tone = 0.8 * np.sin(phase) + 0.2 * np.sin(phase / 2.0)  # octave-down softens harmonics
    attack = int(0.02 * SAMPLE_RATE)
    env = np.ones(n)
    env[:attack] = np.linspace(0.0, 1.0, attack)
    env *= np.exp(-t * 14.0)
    volume = _get_beep_volume() * 0.5  # deliberately quieter than the beeps
    return (tone * env * volume * 32767).astype(np.int16)


def _thinking_sound_loop(stop: threading.Event, should_play) -> None:
    """Daemon loop: alternating-pitch blips every ~0.8-1.2s until *stop*.

    Skips (without stopping) whenever *should_play* returns False. macOS:
    sounddevice output is TCC-gated and per-second afplay churn is worse than
    silence, so the loop exits immediately there.
    """
    if not _sounddevice_output_allowed():
        return
    try:
        sd, np = _import_audio()
    except (ImportError, OSError):
        return
    import random
    blips = [_synth_thinking_blip(np, p) for p in (392.0, 329.6)]  # G4 / E4
    i = 0
    while not stop.is_set():
        try:
            if should_play is None or should_play():
                blip = blips[i % len(blips)]
                sd.play(blip, samplerate=SAMPLE_RATE)
                stop.wait(len(blip) / SAMPLE_RATE + 0.02)
                sd.stop()
                i += 1
        except Exception as e:
            logger.debug("Thinking sound blip failed: %s", e)
            return
        stop.wait(0.8 + random.random() * 0.4)


def start_thinking_sound(should_play=None) -> bool:
    """Start the ambient thinking sound (idempotent).

    *should_play* is polled before each blip. Returns True when running
    (or already running), False when disabled/unavailable.
    """
    global _thinking_stop
    if not thinking_sound_enabled():
        return False
    with _thinking_lock:
        if _thinking_stop is not None and not _thinking_stop.is_set():
            return True
        stop = threading.Event()
        _thinking_stop = stop
    threading.Thread(target=_thinking_sound_loop, args=(stop, should_play),
                     daemon=True, name="voice-thinking-sound").start()
    return True


def stop_thinking_sound() -> None:
    """Stop the ambient thinking sound instantly (idempotent)."""
    global _thinking_stop
    with _thinking_lock:
        stop, _thinking_stop = _thinking_stop, None
    if stop is not None:
        stop.set()


# ── Recorders ──
def _new_recording_path(ext: str) -> str:
    """Timestamped ``recording_*.<ext>`` path under _TEMP_DIR (created on demand)."""
    os.makedirs(_TEMP_DIR, exist_ok=True)
    return os.path.join(_TEMP_DIR, f"recording_{time.strftime('%Y%m%d_%H%M%S')}.{ext}")


class _RecorderBase:
    """Lock, recording flag, start time and live RMS shared by both recorder backends."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._recording = False
        self._start_time: float = 0.0
        self._current_rms: int = 0

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def elapsed_seconds(self) -> float:
        if not self._recording:
            return 0.0
        return time.monotonic() - self._start_time

    @property
    def current_rms(self) -> int:
        """Current input RMS level (0-32767), updated each audio chunk."""
        return self._current_rms


class TermuxAudioRecorder(_RecorderBase):
    """Recorder backend that uses Termux:API microphone capture commands."""

    supports_silence_autostop = False

    def __init__(self) -> None:
        super().__init__()
        self._recording_path: Optional[str] = None

    def start(self, on_silence_stop=None) -> None:
        del on_silence_stop  # Termux:API does not expose live silence callbacks.
        mic_cmd = _termux_microphone_command()
        if not mic_cmd:
            raise RuntimeError(
                "Termux voice capture requires the termux-api package and app.\n"
                "Install with: pkg install termux-api\n"
                "Then install/update the Termux:API Android app.")
        if not _termux_api_app_installed():
            raise RuntimeError(
                "Termux voice capture requires the Termux:API Android app.\n"
                "Install/update the Termux:API app, then retry /voice on.")
        with self._lock:
            if self._recording:
                return
            self._recording_path = _new_recording_path("aac")
        command = [mic_cmd, "-f", self._recording_path, "-l", "0", "-e", "aac",
                   "-r", str(SAMPLE_RATE), "-c", str(CHANNELS)]
        try:
            _run_quiet(command, timeout=15, check=True)
        except subprocess.CalledProcessError as e:
            details = (e.stderr or e.stdout or str(e)).strip()
            raise RuntimeError(f"Termux microphone start failed: {details}") from e
        except Exception as e:
            raise RuntimeError(f"Termux microphone start failed: {e}") from e
        with self._lock:
            self._start_time = time.monotonic()
            self._recording = True
            self._current_rms = 0
        logger.info("Termux voice recording started")

    def _stop_termux_recording(self) -> None:
        mic_cmd = _termux_microphone_command()
        if mic_cmd:
            _run_quiet([mic_cmd, "-q"], timeout=15, check=False)

    def _reset_state(self) -> tuple:
        """Clear recording state under the lock; return (was_recording, path, started_at)."""
        with self._lock:
            was_recording, path, started_at = self._recording, self._recording_path, self._start_time
            self._recording = False
            self._recording_path = None
            self._current_rms = 0
        return was_recording, path, started_at

    def stop(self) -> Optional[str]:
        was_recording, path, started_at = self._reset_state()
        if not was_recording:
            return None
        self._stop_termux_recording()
        if not path or not os.path.isfile(path):
            return None
        if time.monotonic() - started_at < 0.3 or os.path.getsize(path) <= 0:  # sub-0.3s taps / empty
            _unlink_quietly(path)
            return None
        logger.info("Termux voice recording stopped: %s", path)
        return path

    def cancel(self) -> None:
        _, path, _ = self._reset_state()
        try:
            self._stop_termux_recording()
        except Exception:
            pass
        if path and os.path.isfile(path):
            _unlink_quietly(path)
        logger.info("Termux voice recording cancelled")

    def shutdown(self) -> None:
        self.cancel()


class AudioRecorder(_RecorderBase):
    """Thread-safe audio recorder using sounddevice.InputStream.

    ``start(on_silence_stop=cb)`` ... ``stop()`` returns a WAV path (or None);
    ``cancel()`` discards. With ``on_silence_stop`` the recording auto-stops
    after ``silence_duration`` seconds of silence following speech.
    """

    supports_silence_autostop = True

    def __init__(self) -> None:
        super().__init__()
        self._stream: Any = None
        self._frames: List[Any] = []
        self._sample_rate: int = SAMPLE_RATE
        self._on_silence_stop = None
        self._silence_threshold: int = SILENCE_RMS_THRESHOLD
        self._silence_duration: float = SILENCE_DURATION_SECONDS
        self._min_speech_duration: float = 0.3  # seconds of speech needed to confirm
        self._max_dip_tolerance: float = 0.3  # max dip before resetting speech
        self._max_wait: float = 15.0  # max seconds to wait for speech before auto-stop
        # Hard cap on total length, wired from voice.max_recording_seconds by the
        # CLI before each recording. 0 (or unset) = no cap.
        self._max_recording_seconds: float = 0.0
        self._peak_rms: int = 0  # for the speech-presence check in stop()
        self._reset_detection_state()

    def _reset_detection_state(self) -> None:
        self._has_spoken = False
        self._speech_start: float = 0.0
        self._dip_start: float = 0.0
        self._silence_start: float = 0.0
        self._resume_start: float = 0.0  # sustained speech after silence started
        self._resume_dip_start: float = 0.0

    def _max_duration_reached(self, elapsed: float) -> bool:
        """``voice.max_recording_seconds`` cap elapsed (<= 0 / unset disables it)."""
        cap = self._max_recording_seconds
        return bool(cap and cap > 0 and elapsed >= cap)

    # -- silence detection ---------------------------------------------------

    def _track_speech(self, rms: int, now: float) -> None:
        """Advance the speech/dip trackers for one audio block.

        Speech is confirmed after ``_min_speech_duration`` above threshold,
        tolerating dips shorter than ``_max_dip_tolerance`` (micro-pauses).
        After confirmation only SUSTAINED resumed speech resets the silence
        timer — brief ambient spikes must not.
        """
        if rms > self._silence_threshold:
            self._dip_start = 0.0
            if self._speech_start == 0.0:
                self._speech_start = now
            elif not self._has_spoken and now - self._speech_start >= self._min_speech_duration:
                self._has_spoken = True
                logger.debug("Speech confirmed (%.2fs above threshold)", now - self._speech_start)
            if not self._has_spoken:
                self._silence_start = 0.0
            else:
                # Resumed speech mirrors initial detection: track, tolerate dips, confirm.
                self._resume_dip_start = 0.0
                if self._resume_start == 0.0:
                    self._resume_start = now
                elif now - self._resume_start >= self._min_speech_duration:
                    self._silence_start = 0.0
                    self._resume_start = 0.0
        elif self._has_spoken:
            if self._resume_start > 0:  # dip-tolerant resume reset
                if self._resume_dip_start == 0.0:
                    self._resume_dip_start = now
                elif now - self._resume_dip_start >= self._max_dip_tolerance:
                    self._resume_start = 0.0
                    self._resume_dip_start = 0.0
        elif self._speech_start > 0:
            # Speech attempt dipped; a long enough dip is genuine silence.
            if self._dip_start == 0.0:
                self._dip_start = now
            elif now - self._dip_start >= self._max_dip_tolerance:
                logger.debug("Speech attempt reset (dip lasted %.2fs)", now - self._dip_start)
                self._speech_start = 0.0
                self._dip_start = 0.0

    def _should_auto_stop(self, rms: int, now: float) -> bool:
        """Spoke then silent for ``_silence_duration``; no speech for ``_max_wait``;
        or the hard cap elapsed (independent of speech)."""
        elapsed = now - self._start_time
        if self._has_spoken and rms <= self._silence_threshold:
            if self._silence_start == 0.0:
                self._silence_start = now
            elif now - self._silence_start >= self._silence_duration:
                logger.info("Silence detected (%.1fs), auto-stopping", self._silence_duration)
                return True
        elif not self._has_spoken and elapsed >= self._max_wait:
            logger.info("No speech within %.0fs, auto-stopping", self._max_wait)
            return True
        if self._max_duration_reached(elapsed):
            logger.info("Max recording length reached (%.0fs), auto-stopping", self._max_recording_seconds)
            return True
        return False

    def _fire_silence_callback(self) -> None:
        """Invoke ``on_silence_stop`` once, in a daemon thread."""
        with self._lock:
            cb = self._on_silence_stop
            self._on_silence_stop = None  # fire only once
        if not cb:
            return

        def _safe_cb():
            try:
                cb()
            except Exception as e:
                logger.error("Silence callback failed: %s", e, exc_info=True)
        threading.Thread(target=_safe_cb, daemon=True).start()

    def _on_audio_block(self, np, indata) -> None:
        self._frames.append(indata.copy())
        rms = int(_rms(np, indata))
        self._current_rms = rms
        self._peak_rms = max(self._peak_rms, rms)
        if self._on_silence_stop is None:
            return
        now = time.monotonic()
        self._track_speech(rms, now)
        if self._should_auto_stop(rms, now):
            self._fire_silence_callback()

    # -- public methods ------------------------------------------------------

    def _ensure_stream(self) -> None:
        """Create the InputStream once and keep it alive.

        Between recordings the callback simply discards chunks; this avoids the
        CoreAudio bug where closing and re-opening an InputStream hangs on macOS.
        """
        if self._stream is not None:
            return
        sd, np = _import_audio()

        def _callback(indata, frames, time_info, status):  # noqa: ARG001
            if status:
                logger.debug("sounddevice status: %s", status)
            if self._recording:
                self._on_audio_block(np, indata)

        stream = None
        try:  # may block on CoreAudio (first call only)
            stream = sd.InputStream(samplerate=self._sample_rate, channels=CHANNELS, dtype=DTYPE, callback=_callback)
            stream.start()
        except Exception as e:
            if stream is not None:
                try:
                    stream.close()
                except Exception:
                    pass
            raise RuntimeError(
                f"Failed to open audio input stream: {e}. "
                "Check that a microphone is connected and accessible.") from e
        self._stream = stream

    def start(self, on_silence_stop=None) -> None:
        """Start capturing from the default input device.

        *on_silence_stop* is invoked (daemon thread, no args) when silence
        follows speech. Raises ``RuntimeError`` if sounddevice/numpy are missing.
        """
        try:
            sd, _ = _import_audio()
        except OSError as e:
            raise RuntimeError(_portaudio_missing_message()) from e
        except ImportError as e:
            raise RuntimeError(
                "Voice mode requires sounddevice and numpy.\n"
                f"Install with: {sys.executable} -m pip install sounddevice numpy") from e
        with self._lock:
            if self._recording:
                return
            self._frames = []
            self._start_time = time.monotonic()
            self._reset_detection_state()
            self._peak_rms = 0
            self._current_rms = 0
            self._on_silence_stop = on_silence_stop
        self._sample_rate = _default_input_samplerate(sd)
        self._ensure_stream()
        with self._lock:
            self._recording = True
        logger.info("Voice recording started (rate=%d, channels=%d)", self._sample_rate, CHANNELS)

    def _close_stream_with_timeout(self, timeout: float = 3.0) -> None:
        """Close the stream with a timeout to prevent CoreAudio hangs."""
        if self._stream is None:
            return
        stream, self._stream = self._stream, None

        def _do_close():
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass

        t = threading.Thread(target=_do_close, daemon=True)
        t.start()
        clock = __import__("time")  # real clock even when tests patch this module's ``time``
        deadline = clock.monotonic() + timeout
        while t.is_alive() and clock.monotonic() < deadline:  # short joins keep Ctrl+C responsive
            t.join(timeout=0.1)
        if t.is_alive():
            logger.warning("Audio stream close timed out after %.1fs — forcing ahead", timeout)

    def stop(self) -> Optional[str]:
        """Stop recording (stream stays alive) and return the WAV path, or None if unusable."""
        with self._lock:
            if not self._recording:
                return None
            self._recording = False
            self._current_rms = 0
            if not self._frames:
                return None
            _, np = _import_audio()
            audio_data = np.concatenate(self._frames, axis=0)
            self._frames = []
            elapsed = time.monotonic() - self._start_time
            logger.info("Voice recording stopped (%.1fs, %d samples)", elapsed, len(audio_data))
            if len(audio_data) < int(self._sample_rate * 0.3):
                logger.debug("Recording too short (%d samples), discarding", len(audio_data))
                return None
            # Peak RMS, not the average (which trailing silence dilutes).
            if self._peak_rms < SILENCE_RMS_THRESHOLD:
                logger.info("Recording too quiet (peak RMS=%d < %d), discarding",
                            self._peak_rms, SILENCE_RMS_THRESHOLD)
                return None
            return self._write_wav(audio_data, sample_rate=self._sample_rate)

    def _discard(self) -> None:
        with self._lock:
            self._recording = False
            self._frames = []
            self._on_silence_stop = None
            self._current_rms = 0

    def cancel(self) -> None:
        """Stop recording and discard all captured audio (stream stays alive)."""
        self._discard()
        logger.info("Voice recording cancelled")

    def shutdown(self) -> None:
        """Release the audio stream. Call when voice mode is disabled."""
        self._discard()
        self._close_stream_with_timeout()  # outside the lock: avoids deadlock with the callback
        logger.info("AudioRecorder shut down")

    @staticmethod
    def _write_wav(audio_data, *, sample_rate: int = SAMPLE_RATE) -> str:
        """Write numpy int16 audio to a WAV file; returns the path."""
        wav_path = _new_recording_path("wav")
        _write_wav_frames(wav_path, audio_data.tobytes(), sample_rate)
        logger.info("WAV written: %s (%d bytes)", wav_path, os.path.getsize(wav_path))
        return wav_path


def create_audio_recorder() -> AudioRecorder | TermuxAudioRecorder:
    """Return the best recorder backend for the current environment."""
    if _termux_voice_capture_available():
        return TermuxAudioRecorder()
    return AudioRecorder()


# ── STT dispatch ──
def transcribe_recording(wav_path: str, model: Optional[str] = None) -> Dict[str, Any]:
    """Transcribe a WAV via ``transcribe_audio()``, filtering Whisper hallucinations.

    Returns dict with ``success``, ``transcript``, and optionally ``error``.
    """
    from tools.transcription_tools import MAX_FILE_SIZE, transcribe_audio

    result = transcribe_audio(wav_path, model=model, source="voice_mode")
    # Only chunk when the provider itself reports "File too large" — local
    # providers have no upload cap and never return this error.
    if not result.get("success") and "File too large" in result.get("error", ""):
        result = _transcribe_wav_in_chunks(wav_path, model=model, max_file_size=MAX_FILE_SIZE)
    # A configured stop phrase always survives: "bye"/"okay" overlap the
    # hallucination blocklist, and swallowing them would make "bye" fail to end the chat.
    if result.get("success"):
        raw_transcript = result.get("transcript", "")
        if is_whisper_hallucination(raw_transcript) and not is_voice_stop_phrase(raw_transcript):
            logger.info("Filtered Whisper hallucination: %r", result["transcript"])
            return {"success": True, "transcript": "", "filtered": True}
    # no_speech = heard no words, not a failure: re-listen quietly instead of
    # surfacing "Transcription failed".
    if result.get("no_speech"):
        return {"success": True, "transcript": "", "no_speech": True}
    return result


def _transcribe_wav_in_chunks(wav_path: str, *, model: Optional[str], max_file_size: int) -> Dict[str, Any]:
    """Split an oversized WAV into provider-sized chunks and join transcripts."""
    from tools.transcription_tools import transcribe_audio

    chunk_paths: List[str] = []
    transcripts: List[str] = []
    try:
        chunk_paths = _split_wav_for_transcription(wav_path, max_file_size=max_file_size)
        if not chunk_paths:
            return {"success": False, "transcript": "", "error": "No audio chunks were created"}
        logger.info("Transcribing oversized WAV in %d chunks: %s", len(chunk_paths), wav_path)
        for index, chunk_path in enumerate(chunk_paths, start=1):
            result = transcribe_audio(chunk_path, model=model, source="voice_mode")
            if not result.get("success"):
                error = result.get("error", "Unknown transcription error")
                return {"success": False, "transcript": "",
                        "error": f"Chunk {index}/{len(chunk_paths)} failed: {error}"}
            transcript = result.get("transcript", "").strip()
            if transcript and not is_whisper_hallucination(transcript):
                transcripts.append(transcript)
        return {"success": True, "transcript": " ".join(transcripts).strip(),
                "provider": result.get("provider"), "chunks": len(chunk_paths)}
    except Exception as e:
        logger.error("Chunked transcription failed for %s: %s", wav_path, e, exc_info=True)
        return {"success": False, "transcript": "", "error": f"Chunked transcription failed: {e}"}
    finally:
        for chunk_path in chunk_paths:
            _unlink_quietly(chunk_path)


def _split_wav_for_transcription(wav_path: str, *, max_file_size: int) -> List[str]:
    """Write WAV chunks small enough to pass the shared STT file-size gate."""
    os.makedirs(_TEMP_DIR, exist_ok=True)
    chunk_paths: List[str] = []
    header_reserve = 64 * 1024
    with wave.open(wav_path, "rb") as source:
        params = source.getparams()
        block_align = max(1, params.nchannels * params.sampwidth)
        max_data_bytes = max_file_size - header_reserve
        if max_data_bytes < block_align:
            raise ValueError("STT max_file_size is too small for WAV chunking")
        frames_per_chunk = max(1, max_data_bytes // block_align)
        index = 0
        while True:
            frames = source.readframes(frames_per_chunk)
            if not frames:
                break
            index += 1
            temp = tempfile.NamedTemporaryFile(
                prefix=f"{os.path.splitext(os.path.basename(wav_path))[0]}_chunk{index:03d}_",
                suffix=".wav", dir=_TEMP_DIR, delete=False)
            chunk_path = temp.name
            temp.close()
            try:
                with wave.open(chunk_path, "wb") as chunk:
                    chunk.setparams(params._replace(nframes=0))
                    chunk.writeframes(frames)
                chunk_paths.append(chunk_path)
            except Exception:
                _unlink_quietly(chunk_path)
                raise
    return chunk_paths


# ── Audio playback (interruptable) ──
_active_playback: Optional[subprocess.Popen] = None  # so stop_playback can interrupt it
_playback_lock = threading.Lock()


def _set_active_playback(proc) -> None:
    global _active_playback
    with _playback_lock:
        _active_playback = proc


def stop_playback() -> None:
    """Interrupt the currently playing audio (if any)."""
    global _active_playback
    with _playback_lock:
        proc = _active_playback
        _active_playback = None
    if proc and proc.poll() is None:
        try:
            proc.terminate()
            logger.info("Audio playback interrupted")
        except Exception:
            pass
    try:  # also stop sounddevice playback if active
        sd, _ = _import_audio()
        sd.stop()
    except Exception:
        pass


def _is_wsl2_env() -> bool:
    """True inside WSL (Microsoft kernel signature in /proc/version); False on any error.

    Module-level so tests can patch it instead of ``builtins.open``.
    """
    try:
        with open("/proc/version", encoding="utf-8", errors="replace") as _fv:
            return "microsoft" in _fv.read().lower()
    except OSError:
        return False


def _wsl_powershell_tts_available() -> bool:
    """True when the WSL2 PowerShell TTS playback fallback can be used.

    OUTPUT only (Media.SoundPlayer on the Windows host) — microphone recording
    still needs a PulseAudio bridge, so callers must keep surfacing that guidance.
    """
    return bool(_is_wsl2_env() and shutil.which("powershell.exe") and shutil.which("ffmpeg"))


def play_audio_file(file_path: str) -> bool:
    """Play an audio file; returns True on success.

    WAV goes through ``sounddevice.play()`` when allowed; otherwise system
    players: ``afplay`` (macOS), the WSL2 PowerShell bridge, ``ffplay``,
    ``aplay`` (Linux). Interruptible via ``stop_playback()``.
    """
    mark_audio_output_active(True)  # ref-count real speaker output for the whole call
    try:
        return _play_audio_file_impl(file_path)
    finally:
        mark_audio_output_active(False)


def _play_wav_via_sounddevice(file_path: str) -> bool:
    """Play a WAV through sounddevice; False when unavailable/failed (caller falls through)."""
    try:
        sd, np = _import_audio()
        with wave.open(file_path, "rb") as wf:
            frames = wf.readframes(wf.getnframes())
            audio_data = np.frombuffer(frames, dtype=np.int16)
            sample_rate = wf.getframerate()
        # WSLg RDP audio needs a warmup to avoid crackling: the RDP channel takes
        # ~100 ms to stabilise and the small default blocksize worsens
        # clock-adjustment jitter (microsoft/wslg#1257).
        if _is_wsl2_env():
            silence_samples = int(0.1 * sample_rate)
            fade_samples = int(0.1 * sample_rate)
            fade = np.linspace(0.0, 1.0, fade_samples, dtype=np.float64)
            audio_float = audio_data.astype(np.float64)
            audio_float[:fade_samples] *= fade
            tail = np.zeros(int(0.05 * sample_rate), dtype=np.int16)
            audio_data = np.concatenate([
                np.zeros(silence_samples, dtype=np.int16), audio_float.astype(np.int16), tail])
            blocksize = 4096
        else:
            blocksize = 0  # default (auto)
        _sd_play_blocking(sd, audio_data, sample_rate,
                          timeout=len(audio_data) / sample_rate + 2.0, blocksize=blocksize)
        return True
    except (ImportError, OSError):
        return False
    except Exception as e:
        logger.debug("sounddevice playback failed: %s", e)
        return False


def _wsl_powershell_player_cmd(file_path: str) -> Optional[List[str]]:
    """Build the WSL2 PowerShell fallback player command, or None.

    Without a PulseAudio bridge ffplay/aplay have no device, but Media.SoundPlayer
    on the Windows host does: convert to a uniquely-named WAV in Windows %TEMP%
    (concurrent TTS calls must not collide) and play it. The WAV is deleted
    unconditionally and the ORIGINAL exit status re-raised past that cleanup
    (rm -f always exits 0) so the player loop can fall through.
    """
    if not (shutil.which("powershell.exe") and shutil.which("ffmpeg") and _is_wsl2_env()):
        return None
    try:
        import uuid

        def _out(cmd):
            return subprocess.check_output(cmd, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL,
                                           timeout=3).decode(errors="replace").strip()

        win_tmp_wsl = _out(["wslpath", "-u", _out(["cmd.exe", "/c", "echo %TEMP%"])])
        if not win_tmp_wsl:
            return None
        wsl_wav = os.path.join(win_tmp_wsl, f"hermes-tts-{uuid.uuid4().hex[:8]}.wav")
        win_wav = _out(["wslpath", "-w", wsl_wav])
        if not win_wav:
            return None
        win_wav_safe = win_wav.replace("'", "''")
        ps_script = f"(New-Object Media.SoundPlayer '{win_wav_safe}').PlaySync()"
        ps_cmd = " && ".join([
            shlex.join(["ffmpeg", "-i", file_path, "-f", "wav", wsl_wav, "-loglevel", "quiet", "-y"]),
            shlex.join(["powershell.exe", "-NoProfile", "-Command", ps_script])])
        cleanup = shlex.join(["rm", "-f", wsl_wav])
        # Full path so the which(cmd[0]) check in the player loop passes.
        return ["/bin/sh", "-c", f"( {ps_cmd} ); rc=$?; {cleanup}; exit $rc"]
    except Exception:
        return None  # WSL path resolution failed; fall through to ffplay/aplay


def _system_player_candidates(file_path: str) -> List[List[str]]:
    """Ordered system-player commands for this platform."""
    system = platform.system()
    players: List[List[str]] = []
    if system == "Darwin":
        players.append(["afplay", file_path])
    if system == "Linux":
        ps_cmd = _wsl_powershell_player_cmd(file_path)
        if ps_cmd:
            players.append(ps_cmd)
    players.append(["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", file_path])
    if system == "Linux":
        players.append(["aplay", "-q", file_path])
    return players


def _run_system_player(cmd: List[str]) -> bool:
    """Run one player to completion (interruptible via stop_playback)."""
    proc = None
    try:
        # Sibling of the TTS/STT credential scrub: players must not inherit tokens/keys.
        from tools.environments.local import hermes_subprocess_env
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                stdin=subprocess.DEVNULL, env=hermes_subprocess_env(inherit_credentials=False))
        _set_active_playback(proc)
        proc.wait(timeout=300)
        rc = proc.returncode
        if rc == 0:
            return True
        # e.g. WSL ffplay/aplay with no audio device — fall through to the next player.
        logger.debug("System player %s exited with code %d, trying next", cmd[0], rc)
    except subprocess.TimeoutExpired:
        logger.warning("System player %s timed out, killing process", cmd[0])
        if proc is not None:
            proc.kill()
            proc.wait()
    except Exception as e:
        logger.debug("System player %s failed: %s", cmd[0], e)
    finally:
        _set_active_playback(None)
    return False


def _play_audio_file_impl(file_path: str) -> bool:
    if not os.path.isfile(file_path):
        logger.warning("Audio file not found: %s", file_path)
        return False
    # macOS skips sounddevice output (TCC media-library prompt); afplay handles all formats.
    if file_path.endswith(".wav") and _sounddevice_output_allowed() and _play_wav_via_sounddevice(file_path):
        return True
    for cmd in _system_player_candidates(file_path):
        if shutil.which(cmd[0]) and _run_system_player(cmd):
            return True
    logger.warning("No audio player available for %s", file_path)
    return False


# ── Full-duplex agent-turn listener ──
# One listener for the WHOLE agent turn in continuous voice mode: armed when an
# utterance is submitted, disarmed when the turn (response + TTS) is done. It
# calibrates against the QUIET room at turn start, freezes that baseline through
# playback (never against speaker bleed), and trips on a windowed majority of
# blocks — so the user can interject during LLM generation, not just TTS.

# Minimum trigger while TTS flows: speaker bleed reaching the mic is a few
# hundred RMS (~1000-1400 with loud close speakers); direct speech is 3000-8000.
PLAYBACK_MIN_TRIGGER = 1500.0
# A noisy room must never push the trigger above what normal speech can reach.
TRIGGER_CEILING = 4000.0
# Multiplier over the quiet-room floor (typically 50-300 RMS): 3x separates
# speech from ambient while staying reachable (300 * 3 = 900 vs 3000+ speech).
DEFAULT_BARGE_MULTIPLIER = 3.0


def _vad_log(msg: str) -> None:
    """VAD diagnostic: logger.debug, plus stderr when HERMES_VOICE_DEBUG=1 (live tuning)."""
    logger.debug(msg)
    if os.environ.get("HERMES_VOICE_DEBUG", "").strip() == "1":
        try:
            print(f"[voice-vad] {msg}", file=sys.stderr, flush=True)
        except Exception:
            pass


def _capture_until_quiet(stream, np, block: int, pre_roll, *, endpoint_blocks: int, max_blocks: int) -> str:
    """After a trip, read until *endpoint_blocks* of quiet (or *max_blocks*); write
    pre-roll + capture to a WAV and return its path. Playback was cut by the
    trigger, so plain silence endpointing works."""
    frames: List[Any] = list(pre_roll)
    quiet = 0
    for _ in range(max_blocks):
        data, _ = stream.read(block)
        frames.append(data.copy())
        quiet = quiet + 1 if _rms(np, data) < SILENCE_RMS_THRESHOLD else 0
        if quiet >= endpoint_blocks:
            break
    return AudioRecorder._write_wav(np.concatenate(frames, axis=0))


class _BargeDetector:
    """Per-block barge-in state machine behind ``full_duplex_listen``."""

    def __init__(self, np, *, mult: float, calib_blocks: int, trip_blocks: int, grace_blocks: int) -> None:
        self._np = np
        self.mult = mult
        self.calib_blocks = calib_blocks
        self.grace_blocks = grace_blocks
        self.trip_needed = max(1, int(round(trip_blocks * 0.8)))
        self.ambient: deque = deque(maxlen=100)  # ~3s of quiet-phase RMS
        self.recent_above: deque = deque(maxlen=trip_blocks)
        self.quiet_floor = float(SILENCE_RMS_THRESHOLD)
        self.floor_locked = False
        self.playing_prev = False
        self.playback_seen = False
        self.grace_remaining = 0
        self.blocks_since_playback = 10_000
        self.block_idx = 0

    def _floor(self) -> tuple:
        """(pct90, floor): 90th percentile of the quiet window; floor never below the silence threshold."""
        seq = self.ambient
        pct90 = float(self._np.percentile(list(seq), 90)) if seq else float(SILENCE_RMS_THRESHOLD)
        return pct90, max(pct90, float(SILENCE_RMS_THRESHOLD))

    def _calibrate(self, rms: float, playing: bool) -> bool:
        """Pre-playback calibration: the listener arms at utterance submit, before
        any TTS exists, so the first calib_blocks sample the actual quiet room —
        NOT speaker bleed. Returns True once the floor is locked."""
        if not playing:
            self.ambient.append(rms)
        if len(self.ambient) >= self.calib_blocks or playing:
            pct90, self.quiet_floor = self._floor()
            self.floor_locked = True
            _vad_log(f"calibrated quiet floor={self.quiet_floor:.0f} "
                     f"(pct90={pct90:.0f}, {len(self.ambient)} blocks, mult={self.mult:g})")
        return self.floor_locked

    def _track_playback(self, playing: bool) -> None:
        """Grace only when playback starts after a real gap (>=1s), so inter-sentence
        flapping of the audio-active flag can't chain grace windows together and
        swallow a genuine interjection."""
        if playing and not self.playing_prev:
            if not self.playback_seen or self.blocks_since_playback > 33:
                self.grace_remaining = self.grace_blocks
                _vad_log(f"playback started (block={self.block_idx}) — grace {self.grace_blocks * 30}ms")
            self.playback_seen = True
        self.playing_prev = playing
        self.blocks_since_playback = 0 if playing else self.blocks_since_playback + 1

    def feed(self, rms: float, playing: bool) -> Optional[str]:
        """Consume one 30ms block; return the phase name when speech trips, else None."""
        self.block_idx += 1
        if not self.floor_locked and not self._calibrate(rms, playing):
            return None
        self._track_playback(playing)
        # Trigger: quiet baseline x multiplier, phase-clamped.
        trigger = max(self.quiet_floor * self.mult,
                      PLAYBACK_MIN_TRIGGER if playing else float(SILENCE_RMS_THRESHOLD) * 2)
        trigger = min(trigger, TRIGGER_CEILING)
        # Track ambient drift ONLY while nothing is playing (never absorb bleed) and the block isn't speech.
        if not playing and rms < trigger:
            self.ambient.append(rms)
            _, self.quiet_floor = self._floor()
        above = rms >= trigger
        if above and self.grace_remaining > 0:
            _vad_log(f"grace suppression: block={self.block_idx} rms={rms:.0f} "
                     f"trigger={trigger:.0f} ({self.grace_remaining} blocks left)")
            above = False
        if self.grace_remaining > 0:
            self.grace_remaining -= 1
        self.recent_above.append(above)
        phase = "playback" if playing else "generation"
        if rms >= trigger * 0.5:
            _vad_log(f"block={self.block_idx} rms={rms:.0f} floor={self.quiet_floor:.0f} "
                     f"trigger={trigger:.0f} above={above} "
                     f"window={sum(self.recent_above)}/{self.trip_needed} phase={phase}")
        if not (above and sum(self.recent_above) >= self.trip_needed):
            return None
        _vad_log(f"TRIPPED ({phase}): block={self.block_idx} rms={rms:.0f} "
                 f"floor={self.quiet_floor:.0f} trigger={trigger:.0f} "
                 f"window={sum(self.recent_above)}/{len(self.recent_above)}")
        return phase


def full_duplex_listen(
    should_stop: Callable[[], bool],
    is_playing: Optional[Callable[[], bool]] = None,
    on_trigger: Optional[Callable[[str], None]] = None,
    multiplier: Optional[float] = None,
    sustained_ms: int = 300,
    calibration_ms: int = 450,
    grace_ms: int = 500,
    pre_roll_ms: int = 1200,
    endpoint_silence_ms: int = 1250,
    max_utterance_ms: int = 30_000,
) -> Optional[str]:
    """Listen across an ENTIRE agent turn; return the captured interruption WAV path.

    Two phases, decided per 30ms block by *is_playing* (usually
    ``is_audio_output_active``): ``generation`` — the first *calibration_ms* of
    quiet room set the noise floor, trigger = floor x *multiplier*;
    ``playback`` — the quiet baseline is HELD, the trigger is clamped up to
    ``PLAYBACK_MIN_TRIGGER`` so bleed alone can't trip it, and *grace_ms*
    after playback starts suppresses onset transients. Detection is a windowed
    majority (>=80% of the last *sustained_ms* of blocks above trigger) so
    intra-word dips don't reset progress. On detection ``on_trigger(phase)``
    fires and capture continues from the rolling *pre_roll_ms* buffer until
    *endpoint_silence_ms* of quiet. Returns ``None`` when *should_stop* ends
    the turn without speech.
    """
    try:
        sd, np = _import_audio()
    except (ImportError, OSError):
        return None
    block = int(SAMPLE_RATE * 0.03)  # 30ms blocks
    detector = _BargeDetector(
        np, mult=float(multiplier) if multiplier else DEFAULT_BARGE_MULTIPLIER,
        calib_blocks=max(1, calibration_ms // 30), trip_blocks=max(1, sustained_ms // 30),
        grace_blocks=max(0, grace_ms // 30))
    pre_roll: deque = deque(maxlen=max(1, pre_roll_ms // 30))
    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16", blocksize=block) as stream:
            while not should_stop():
                data, _ = stream.read(block)
                pre_roll.append(data.copy())
                playing = bool(is_playing()) if is_playing is not None else False
                phase = detector.feed(_rms(np, data), playing)
                if phase is None:
                    continue
                if on_trigger:
                    try:
                        on_trigger(phase)
                    except Exception as e:
                        logger.debug("full-duplex trigger callback failed: %s", e)
                return _capture_until_quiet(
                    stream, np, block, pre_roll,
                    endpoint_blocks=max(1, endpoint_silence_ms // 30), max_blocks=max(1, max_utterance_ms // 30))
    except Exception as e:
        logger.debug("Full-duplex listener failed: %s", e)
    return None


# ── Requirements check ──
def _check_plugin_stt_provider(provider: str) -> bool:
    """True when *provider* resolves to an available STT plugin."""
    key = (provider or "").lower().strip()
    if not key or key == "none":
        return False
    try:
        from agent.transcription_registry import get_provider
        from hermes_cli.plugins import _ensure_plugins_discovered
        _ensure_plugins_discovered()
        plugin_provider = get_provider(key)
        if plugin_provider is None:
            # Match the transcription dispatcher: long-lived processes may need
            # one refresh after plugins or configuration change.
            _ensure_plugins_discovered(force=True)
            plugin_provider = get_provider(key)
    except Exception as exc:  # noqa: BLE001 - discovery failure is non-fatal
        logger.debug("STT plugin requirements check skipped for '%s': %s", key, exc)
        return False
    if plugin_provider is None:
        return False
    try:
        return bool(plugin_provider.is_available())
    except Exception as exc:  # noqa: BLE001 - plugins must not break status
        logger.warning(
            "STT plugin provider '%s' is_available() raised during requirements "
            "check: %s - treating as unavailable", key, exc, exc_info=True)
        return False


# STT providers handled natively by tools.transcription_tools -> status label.
_NATIVE_STT_LABELS = {
    "local": "local faster-whisper",
    "local_command": "local command",
    "groq": "Groq",
    "openai": "OpenAI",
    "mistral": "Mistral Voxtral",
    "xai": "xAI Grok STT",
    "elevenlabs": "ElevenLabs Scribe",
}


def check_voice_requirements() -> Dict[str, Any]:
    """Check voice mode requirements.

    Returns dict with ``available``, ``audio_available``, ``stt_available``,
    ``missing_packages``, ``details`` and ``environment``.
    """
    from tools.transcription_tools import (
        _get_provider, _load_stt_config, _resolve_command_stt_provider_config, is_stt_enabled)
    stt_config = _load_stt_config()
    stt_enabled = is_stt_enabled(stt_config)
    stt_provider = _get_provider(stt_config)
    stt_label = None  # "OK (...)" once a native / command / plugin provider resolves
    if stt_provider in _NATIVE_STT_LABELS:
        stt_label = f"OK ({_NATIVE_STT_LABELS[stt_provider]})"
    elif stt_enabled:
        if _resolve_command_stt_provider_config(stt_provider, stt_config) is not None:
            stt_label = f"OK (command: {stt_provider})"
        elif _check_plugin_stt_provider(stt_provider):
            stt_label = f"OK (plugin: {stt_provider})"
    stt_available = stt_enabled and stt_label is not None

    termux_capture = _termux_voice_capture_available()
    has_audio = _audio_available() or termux_capture
    env_check = detect_audio_environment()
    details = [
        "Audio capture: OK (Termux:API microphone)" if termux_capture
        else "Audio capture: OK" if has_audio
        else f"Audio capture: MISSING ({_voice_capture_install_hint()})",
        "STT provider: DISABLED in config (stt.enabled: false)" if not stt_enabled
        else f"STT provider: {stt_label}" if stt_label
        else ("STT provider: MISSING (uv pip install faster-whisper — "
              "`pip install faster-whisper` also works if pip is on PATH, "
              "or set GROQ_API_KEY / VOICE_TOOLS_OPENAI_KEY)"),
    ]
    details += [f"Environment: {w}" for w in env_check["warnings"]]
    details += [f"Environment: {n}" for n in env_check.get("notices", [])]
    return {
        "available": has_audio and stt_available and env_check["available"],
        "audio_available": has_audio,
        "stt_available": stt_available,
        "missing_packages": [] if has_audio else ["sounddevice", "numpy"],
        "details": "\n".join(details),
        "environment": env_check,
    }


# ── Temp file cleanup ──
def cleanup_temp_recordings(max_age_seconds: int = 3600) -> int:
    """Remove ``recording_*.wav`` temp files older than *max_age_seconds*; returns the count."""
    if not os.path.isdir(_TEMP_DIR):
        return 0
    deleted = 0
    now = time.time()
    for entry in os.scandir(_TEMP_DIR):
        if entry.is_file() and entry.name.startswith("recording_") and entry.name.endswith(".wav"):
            try:
                if now - entry.stat().st_mtime > max_age_seconds:
                    os.unlink(entry.path)
                    deleted += 1
            except OSError:
                pass
    if deleted:
        logger.debug("Cleaned up %d old voice recordings", deleted)
    return deleted
