"""Voice Mode -- Push-to-talk audio recording and playback for the CLI.

Provides audio capture via sounddevice, WAV encoding via stdlib wave,
STT dispatch via tools.transcription_tools, and TTS playback via
sounddevice or system audio players.

Dependencies (optional):
    pip install sounddevice numpy
    or: uv sync --extra voice
"""

import logging
import math
import os
import platform
import shlex
import shutil
import subprocess
import sys
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

# ── Lazy audio imports ──
# Never imported at module level: crashes headless environments (SSH, Docker,
# WSL, no PortAudio).

def _import_audio():
    """Lazy-import sounddevice and numpy; returns (sd, np). Raises ImportError
    or OSError when unavailable (e.g. PortAudio missing on headless servers)."""
    import sounddevice as sd
    import numpy as np
    return sd, np


def _import_numpy():
    """Lazy-import numpy only — for synthesizing samples where sounddevice
    must NOT be imported (see _sounddevice_output_allowed)."""
    import numpy as np
    return np


def _sounddevice_output_allowed() -> bool:
    """Whether sounddevice may be used for audio OUTPUT.

    False on macOS: initializing PortAudio/CoreAudio for output triggers a
    kTCCServiceMediaLibrary prompt even though playback needs no media-library
    access, so all output goes through ``afplay`` there. Does NOT affect
    *input* (recording), which legitimately needs microphone permission.
    """
    return platform.system() != "Darwin"


def _play_int16_via_tempfile(audio, sample_rate: int) -> None:
    """Write int16 mono PCM to a temp WAV and play it via play_audio_file.

    Used on macOS so tone/beep output goes through ``afplay`` instead of
    sounddevice (avoids the TCC media-library prompt).
    """
    tmp_path = None
    try:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_path = tmp.name
        _write_wav_frames(tmp, audio.tobytes(), sample_rate)
        play_audio_file(tmp_path)
    except Exception as e:
        logger.debug("Tone tempfile playback failed: %s", e)
    finally:
        if tmp_path:
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
    """Return True if audio libraries can be imported."""
    try:
        _import_audio()
        return True
    except (ImportError, OSError):
        return False


def _rms(np, data) -> float:
    """Root-mean-square level of an int16 block."""
    return float(np.sqrt(np.mean(data.astype(np.float64) ** 2)))


def _default_input_samplerate(sd) -> int:
    """Preferred capture rate for the default input device; falls back to the
    Whisper-friendly 16 kHz constant when the backend exposes no numeric rate."""
    try:
        info = sd.query_devices(None, "input")
        rate = info.get("default_samplerate") if isinstance(info, dict) else getattr(info, "default_samplerate", None)
        if isinstance(rate, (int, float)) and rate > 0:
            return int(round(rate))
    except Exception:
        pass
    return SAMPLE_RATE


from hermes_constants import is_termux as _is_termux_environment


def _voice_capture_install_hint() -> str:
    if _is_termux_environment():
        return "pkg install python-numpy portaudio && python -m pip install sounddevice"
    # Inside a venv (e.g. the bundled Hermes venv) a bare `pip install` may hit
    # whichever Python the shell resolves first (on macOS often a Rosetta
    # system Python) — point at the venv's own pip instead.
    try:
        if sys.prefix != getattr(sys, "base_prefix", sys.prefix):
            pip_in_venv = Path(sys.prefix) / "bin" / "pip"
            if pip_in_venv.exists():
                return f"{pip_in_venv} install sounddevice numpy"
    except Exception:
        pass
    return "pip install sounddevice numpy"


def _portaudio_missing_message() -> str:
    """Error text for "sounddevice imports but PortAudio's shared library is
    missing" — a pip install can't fix that, so point at the system package."""
    if _is_termux_environment():
        hint = "  Termux: pkg install portaudio"
    else:
        hint = (
            "  Linux:  sudo apt-get install libportaudio2\n"
            "  macOS:  brew install portaudio"
        )
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
        timeout=timeout, check=check, stdin=subprocess.DEVNULL,
    )


# Probes for the Termux:API Android app. `pm list packages` is the canonical
# lookup but on some ROMs `pm` isn't on Termux's PATH while `cmd package` is,
# and on others `pm` returns nothing for the calling user even when the app is
# present — so both are tried before concluding the app is missing.
_TERMUX_API_PACKAGE_PROBES = (
    ("pm", "list", "packages", "com.termux.api"),
    ("cmd", "package", "list", "packages", "com.termux.api"),
)


def _termux_api_app_installed() -> bool:
    """Return True iff the Termux:API Android app is installed.

    Any probe reporting ``package:com.termux.api`` is authoritative. If EVERY
    probe is inconclusive (binary missing, permission denied, timeout, non-zero
    exit) we cannot honestly say the app is missing, so trust the
    ``termux-microphone-record`` binary on PATH instead: a false negative here
    blocks ``/voice on`` outright, while a false positive only surfaces a
    precise runtime error from the binary. If at least one probe ran cleanly
    without mentioning the package, the app is genuinely missing.
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
            "termux-microphone-record binary on PATH (issue #31015)."
        )
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
    """True if a PulseAudio/PipeWire socket is reachable on disk.

    Covers a sound server running locally (e.g. on a remote SSH host) without
    ``PULSE_SERVER``/``PIPEWIRE_REMOTE`` set. A socket file must also accept a
    connection — a stale socket left by a dead server does not count.
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
    try:
        sd, _ = _import_audio()
    except ImportError:
        if termux_capture:
            notices.append("Termux:API microphone recording available (sounddevice not required)")
        elif termux_mic_cmd and not termux_app_installed:
            warnings.append(_TERMUX_APP_MISSING_WARNING)
        else:
            warnings.append(f"Audio libraries not installed ({_voice_capture_install_hint()})")
        return
    except OSError:
        if termux_capture:
            notices.append("Termux:API microphone recording available (PortAudio not required)")
        elif termux_mic_cmd and not termux_app_installed:
            warnings.append(_TERMUX_APP_MISSING_WARNING)
        else:
            warnings.append(_portaudio_missing_message())
        return

    try:
        if sd.query_devices():
            return
        if has_forwarded_audio:
            notices.append("No PortAudio devices detected but host audio forwarding is configured -- continuing")
        elif termux_capture:
            notices.append("No PortAudio devices detected, but Termux:API microphone capture is available")
        else:
            warnings.append("No audio input/output devices detected")
    except Exception:
        if has_forwarded_audio:
            notices.append("Audio device query failed but host audio forwarding is configured -- continuing")
        elif termux_capture:
            notices.append("PortAudio device query failed, but Termux:API microphone capture is available")
        else:
            warnings.append("Audio subsystem error (PortAudio cannot query devices)")


def detect_audio_environment() -> dict:
    """Detect if the current environment supports audio I/O.

    Returns dict with 'available' (bool), 'warnings' (hard-fail reasons that
    block voice mode), and 'notices' (informational, do NOT block). SSH,
    containers and WSL normally have no audio devices, but a reachable sound
    server (PulseAudio/PipeWire socket or forwarding env vars) is honored.
    """
    warnings: List[str] = []
    notices: List[str] = []
    termux_mic_cmd = _termux_microphone_command()
    termux_app_installed = _termux_api_app_installed()
    has_forwarded_audio = bool(
        os.environ.get('PULSE_SERVER')
        or os.environ.get('PIPEWIRE_REMOTE')
        or _pulse_socket_reachable()
    )

    if any(os.environ.get(v) for v in ('SSH_CLIENT', 'SSH_TTY', 'SSH_CONNECTION')):
        if has_forwarded_audio:
            notices.append("Running over SSH with a reachable PulseAudio/PipeWire sound server")
        else:
            warnings.append(
                "Running over SSH -- no audio devices available.\n"
                "  If a sound server (PulseAudio/PipeWire) is running on this host,\n"
                "  point Hermes at it, e.g.:\n"
                "    export XDG_RUNTIME_DIR=/run/user/$(id -u)\n"
                "    # or: export PULSE_SERVER=unix:$XDG_RUNTIME_DIR/pulse/native"
            )

    from hermes_constants import is_container
    if is_container():
        if has_forwarded_audio:
            notices.append("Running inside container (Docker/Podman/LXC) with host audio forwarding")
        else:
            warnings.append(
                "Running inside container (Docker/Podman/LXC) -- no audio devices.\n"
                "  Forward host audio with one of (substitute $XDG_RUNTIME_DIR for your runtime dir,\n"
                "  typically /run/user/$UID):\n"
                "    PulseAudio:  -v $XDG_RUNTIME_DIR/pulse/native:$XDG_RUNTIME_DIR/pulse/native \\\n"
                "                 -e PULSE_SERVER=unix:$XDG_RUNTIME_DIR/pulse/native\n"
                "    PipeWire:    -e PIPEWIRE_REMOTE=$XDG_RUNTIME_DIR/pipewire-0"
            )

    # WSL: the PowerShell/Media.SoundPlayer fallback only covers OUTPUT, so
    # when it is the only thing available we downgrade to a notice (recording
    # guidance stays visible, but TTS-only usage isn't blocked).
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
                "  3. Verify with: arecord -d 3 /tmp/test.wav && aplay /tmp/test.wav"
            )
        else:
            warnings.append(
                "Running in WSL -- audio requires a forwarded sound server.\n"
                "  PulseAudio: export PULSE_SERVER=unix:/mnt/wslg/PulseServer\n"
                "  PipeWire:   export PIPEWIRE_REMOTE=$XDG_RUNTIME_DIR/pipewire-0\n"
                "  Then verify: arecord -d 3 /tmp/test.wav && aplay /tmp/test.wav"
            )

    _probe_audio_libraries(
        warnings, notices,
        has_forwarded_audio=has_forwarded_audio,
        termux_mic_cmd=termux_mic_cmd,
        termux_app_installed=termux_app_installed,
    )

    return {
        "available": not warnings,
        "warnings": warnings,
        "notices": notices,
    }

# ── Recording parameters ──
SAMPLE_RATE = 16000  # Whisper native rate
CHANNELS = 1  # Mono
DTYPE = "int16"  # 16-bit PCM
SAMPLE_WIDTH = 2  # bytes per sample (int16)

# Silence detection defaults
SILENCE_RMS_THRESHOLD = 200  # RMS below this = silence (int16 range 0-32767)
SILENCE_DURATION_SECONDS = 3.0  # Seconds of continuous silence before auto-stop

# Temp directory for voice recordings
_TEMP_DIR = os.path.join(tempfile.gettempdir(), "hermes_voice")


# ── Audio cues (beep tones) ──
_DEFAULT_BEEP_VOLUME = 0.3   # Backward-compatible default (matches prior hardcoded value)


def _get_beep_volume() -> float:
    """``voice.beep_volume`` clamped to 0.0-1.0; 0.3 when missing/invalid so the
    audio cue never breaks the voice loop on a degenerate config."""
    raw = _voice_config().get("beep_volume", _DEFAULT_BEEP_VOLUME)
    try:
        volume = float(raw)
    except (TypeError, ValueError):
        return _DEFAULT_BEEP_VOLUME
    if isinstance(raw, bool) or volume < 0.0 or volume > 1.0 or math.isnan(volume):
        return _DEFAULT_BEEP_VOLUME
    return volume


def _sd_play_blocking(sd, audio, sample_rate: int, *, timeout: float, blocksize: int = 0) -> None:
    """``sd.play`` then poll until the stream goes idle or *timeout* passes.

    ``sd.wait()`` calls ``Event.wait()`` without a timeout and hangs forever if
    the audio device stalls, so poll with a ceiling and force-stop instead.
    """
    sd.play(audio, samplerate=sample_rate, blocksize=blocksize)
    deadline = time.monotonic() + timeout
    while sd.get_stream() and sd.get_stream().active and time.monotonic() < deadline:
        time.sleep(0.01)
    sd.stop()


def play_beep(frequency: int = 880, duration: float = 0.12, count: int = 1) -> None:
    """Play *count* short beeps of *frequency* Hz (default 880 = A5), *duration* s each.

    The tone is synthesized with numpy only, so the macOS TCC prompt is not
    triggered on the synthesis step; on macOS output goes through afplay.
    """
    try:
        np = _import_numpy()
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
            # Fade in/out to avoid click artifacts.
            fade_len = min(int(SAMPLE_RATE * 0.01), samples_per_beep // 4)
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
# The agent can think / run tools for minutes with zero audio, which reads as
# "it died"; a quiet repeating pair of soft water-bubble blips fills the gap.
# Synthesized with numpy, scaled by voice.beep_volume, gated by
# voice.thinking_sound (default on). macOS: sounddevice OUTPUT is TCC-gated
# and spawning afplay every second would churn subprocesses, so it is skipped.
#
# The host's *should_play* callback decides when blips are allowed; the
# module-level output ref-count below tracks when real audio (TTS sentences,
# file playback) is actually flowing so hosts have an accurate signal.

_audio_output_active_count = 0
_audio_output_lock = threading.Lock()


def mark_audio_output_active(active: bool) -> None:
    """Reference-count real audio output (TTS/file playback).

    Playback paths bracket their work with ``(True)`` / ``(False)`` so
    ``is_audio_output_active()`` reflects whether speech audio is leaving the
    speakers RIGHT NOW — unlike per-turn TTS-done events, which stay 'busy'
    for a whole turn even while the pipeline silently waits for text.
    """
    global _audio_output_active_count
    with _audio_output_lock:
        _audio_output_active_count = max(
            0, _audio_output_active_count + (1 if active else -1)
        )


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
    """One soft 'blub': short sine with a gentle downward pitch glide and a
    smooth attack/decay envelope (no clicks), low-volume."""
    duration = 0.16
    n = int(SAMPLE_RATE * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    # Downward glide (water-drop feel): freq → 0.72*freq over the blip.
    glide = np.linspace(1.0, 0.72, n)
    phase = 2 * np.pi * np.cumsum(frequency * glide) / SAMPLE_RATE
    tone = np.sin(phase)
    # Soften harmonics (cheap low-pass feel): add a quieter octave-down sine.
    tone = 0.8 * tone + 0.2 * np.sin(phase / 2.0)
    # Envelope: quick-but-smooth attack, long exponential-ish decay.
    attack = int(0.02 * SAMPLE_RATE)
    env = np.ones(n)
    env[:attack] = np.linspace(0.0, 1.0, attack)
    env *= np.exp(-t * 14.0)
    volume = _get_beep_volume() * 0.5  # deliberately quieter than the beeps
    return (tone * env * volume * 32767).astype(np.int16)


def _thinking_sound_loop(stop: threading.Event, should_play) -> None:
    """Daemon loop: play alternating-pitch blips every ~0.8-1.2s until *stop*.

    Skips a blip (without stopping) whenever *should_play* returns False —
    e.g. TTS audio started flowing or the mic re-armed. macOS: sounddevice
    output is TCC-gated, and per-second afplay subprocess churn is worse
    than silence, so the loop exits immediately there.
    """
    if not _sounddevice_output_allowed():
        return
    try:
        sd, np = _import_audio()
    except (ImportError, OSError):
        return

    import random

    pitches = (392.0, 329.6)  # G4 / E4 — calm, low, alternating
    blips = [_synth_thinking_blip(np, p) for p in pitches]
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

    *should_play* is polled before each blip; return False to skip while
    speech audio flows or the mic is capturing. Returns True when the loop
    was started (or already running), False when disabled/unavailable.
    """
    global _thinking_stop
    if not thinking_sound_enabled():
        return False
    with _thinking_lock:
        if _thinking_stop is not None and not _thinking_stop.is_set():
            return True  # already running
        stop = threading.Event()
        _thinking_stop = stop
    threading.Thread(
        target=_thinking_sound_loop,
        args=(stop, should_play),
        daemon=True,
        name="voice-thinking-sound",
    ).start()
    return True


def stop_thinking_sound() -> None:
    """Stop the ambient thinking sound instantly (idempotent)."""
    global _thinking_stop
    with _thinking_lock:
        stop, _thinking_stop = _thinking_stop, None
    if stop is not None:
        stop.set()


# ── Termux Audio Recorder ──
def _new_recording_path(ext: str) -> str:
    """Timestamped ``recording_*.<ext>`` path under _TEMP_DIR (created on demand)."""
    os.makedirs(_TEMP_DIR, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return os.path.join(_TEMP_DIR, f"recording_{timestamp}.{ext}")


class TermuxAudioRecorder:
    """Recorder backend that uses Termux:API microphone capture commands."""

    supports_silence_autostop = False

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._recording = False
        self._start_time = 0.0
        self._recording_path: Optional[str] = None
        self._current_rms = 0

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
        return self._current_rms

    def start(self, on_silence_stop=None) -> None:
        del on_silence_stop  # Termux:API does not expose live silence callbacks.
        mic_cmd = _termux_microphone_command()
        if not mic_cmd:
            raise RuntimeError(
                "Termux voice capture requires the termux-api package and app.\n"
                "Install with: pkg install termux-api\n"
                "Then install/update the Termux:API Android app."
            )
        if not _termux_api_app_installed():
            raise RuntimeError(
                "Termux voice capture requires the Termux:API Android app.\n"
                "Install/update the Termux:API app, then retry /voice on."
            )

        with self._lock:
            if self._recording:
                return
            self._recording_path = _new_recording_path("aac")

        command = [
            mic_cmd,
            "-f", self._recording_path,
            "-l", "0",
            "-e", "aac",
            "-r", str(SAMPLE_RATE),
            "-c", str(CHANNELS),
        ]
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
        if not mic_cmd:
            return
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
        # Discard sub-0.3s taps and empty files.
        if time.monotonic() - started_at < 0.3 or os.path.getsize(path) <= 0:
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


# ── AudioRecorder ──
class AudioRecorder:
    """Thread-safe audio recorder using sounddevice.InputStream.

    Usage::

        recorder = AudioRecorder()
        recorder.start(on_silence_stop=my_callback)
        # ... user speaks ...
        wav_path = recorder.stop()   # returns path to WAV file
        # or
        recorder.cancel()            # discard without saving

    If ``on_silence_stop`` is provided, recording automatically stops when
    the user is silent for ``silence_duration`` seconds and calls the callback.
    """

    supports_silence_autostop = True

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stream: Any = None
        self._frames: List[Any] = []
        self._recording = False
        self._start_time: float = 0.0
        self._sample_rate: int = SAMPLE_RATE
        self._on_silence_stop = None
        self._silence_threshold: int = SILENCE_RMS_THRESHOLD
        self._silence_duration: float = SILENCE_DURATION_SECONDS
        self._min_speech_duration: float = 0.3  # Seconds of speech needed to confirm
        self._max_dip_tolerance: float = 0.3  # Max dip duration before resetting speech
        self._max_wait: float = 15.0  # Max seconds to wait for speech before auto-stop
        # Hard cap on total recording length, wired from voice.max_recording_seconds
        # by the CLI before each recording. 0 (or unset) = no cap.
        self._max_recording_seconds: float = 0.0
        self._peak_rms: int = 0  # for the speech-presence check in stop()
        self._current_rms: int = 0  # live level, read by the UI
        self._reset_detection_state()

    def _reset_detection_state(self) -> None:
        """Reset per-recording silence-detection trackers."""
        self._has_spoken = False
        self._speech_start: float = 0.0  # When speech attempt began
        self._dip_start: float = 0.0  # When current below-threshold dip began
        self._silence_start: float = 0.0
        self._resume_start: float = 0.0  # Tracks sustained speech after silence starts
        self._resume_dip_start: float = 0.0  # Dip tolerance tracker for resume detection

    def _max_duration_reached(self, elapsed: float) -> bool:
        """Whether the configured hard recording-length cap has elapsed.

        ``voice.max_recording_seconds`` is applied by the CLI before each
        recording (see ``HermesCLI._voice_start_recording``). A value <= 0
        (or unset) disables the cap.
        """
        cap = self._max_recording_seconds
        return bool(cap and cap > 0 and elapsed >= cap)

    # -- public properties ---------------------------------------------------

    @property
    def elapsed_seconds(self) -> float:
        if not self._recording:
            return 0.0
        return time.monotonic() - self._start_time

    @property
    def current_rms(self) -> int:
        """Current audio input RMS level (0-32767). Updated each audio chunk."""
        return self._current_rms

    @property
    def is_recording(self) -> bool:
        """Whether audio recording is currently active."""
        return self._recording

    # -- silence detection ---------------------------------------------------

    def _track_speech(self, rms: int, now: float) -> None:
        """Advance the speech/dip trackers for one audio block.

        Speech is confirmed after ``_min_speech_duration`` above threshold,
        tolerating dips shorter than ``_max_dip_tolerance`` (micro-pauses
        between syllables). After confirmation only SUSTAINED resumed speech
        resets the silence timer — brief ambient spikes must not.
        """
        if rms > self._silence_threshold:
            self._dip_start = 0.0
            if self._speech_start == 0.0:
                self._speech_start = now
            elif not self._has_spoken and now - self._speech_start >= self._min_speech_duration:
                self._has_spoken = True
                logger.debug("Speech confirmed (%.2fs above threshold)",
                             now - self._speech_start)
            if not self._has_spoken:
                self._silence_start = 0.0
            else:
                # Resumed speech mirrors initial detection: track, tolerate
                # short dips, confirm after _min_speech_duration.
                self._resume_dip_start = 0.0
                if self._resume_start == 0.0:
                    self._resume_start = now
                elif now - self._resume_start >= self._min_speech_duration:
                    self._silence_start = 0.0
                    self._resume_start = 0.0
        elif self._has_spoken:
            # Below threshold after confirmed speech: dip-tolerant resume reset.
            if self._resume_start > 0:
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
                logger.debug("Speech attempt reset (dip lasted %.2fs)",
                             now - self._dip_start)
                self._speech_start = 0.0
                self._dip_start = 0.0

    def _should_auto_stop(self, rms: int, now: float) -> bool:
        """Auto-stop when: the user spoke then stayed silent for
        ``_silence_duration``; no speech at all for ``_max_wait``; or the
        hard ``voice.max_recording_seconds`` cap elapsed (independent of
        speech, so a continuous speaker still stops)."""
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
            logger.info("Max recording length reached (%.0fs), auto-stopping",
                        self._max_recording_seconds)
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
        """Per-block work for the InputStream callback while recording."""
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
        """Create the audio InputStream once and keep it alive.

        The stream stays open for the lifetime of the recorder; between
        recordings the callback simply discards chunks (``_recording`` is
        False). This avoids the CoreAudio bug where closing and re-opening an
        ``InputStream`` hangs indefinitely on macOS.
        """
        if self._stream is not None:
            return  # already alive

        sd, np = _import_audio()

        def _callback(indata, frames, time_info, status):  # noqa: ARG001
            if status:
                logger.debug("sounddevice status: %s", status)
            if self._recording:
                self._on_audio_block(np, indata)

        # Create stream — may block on CoreAudio (first call only).
        stream = None
        try:
            stream = sd.InputStream(
                samplerate=self._sample_rate,
                channels=CHANNELS,
                dtype=DTYPE,
                callback=_callback,
            )
            stream.start()
        except Exception as e:
            if stream is not None:
                try:
                    stream.close()
                except Exception:
                    pass
            raise RuntimeError(
                f"Failed to open audio input stream: {e}. "
                "Check that a microphone is connected and accessible."
            ) from e
        self._stream = stream

    def start(self, on_silence_stop=None) -> None:
        """Start capturing audio from the default input device.

        The InputStream is created once and kept alive across recordings;
        later calls reset detection state and toggle frame collection.
        *on_silence_stop* is invoked (in a daemon thread, no arguments) when
        silence follows speech — use it to auto-stop and transcribe.
        Raises ``RuntimeError`` if sounddevice/numpy are not installed.
        """
        try:
            sd, _ = _import_audio()
        except OSError as e:
            raise RuntimeError(_portaudio_missing_message()) from e
        except ImportError as e:
            raise RuntimeError(
                "Voice mode requires sounddevice and numpy.\n"
                f"Install with: {sys.executable} -m pip install sounddevice numpy"
            ) from e

        with self._lock:
            if self._recording:
                return  # already recording

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
        """Close the audio stream with a timeout to prevent CoreAudio hangs."""
        if self._stream is None:
            return

        stream = self._stream
        self._stream = None

        def _do_close():
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass

        t = threading.Thread(target=_do_close, daemon=True)
        t.start()
        # Poll in short intervals so Ctrl+C is not blocked
        deadline = __import__("time").monotonic() + timeout
        while t.is_alive() and __import__("time").monotonic() < deadline:
            t.join(timeout=0.1)
        if t.is_alive():
            logger.warning("Audio stream close timed out after %.1fs — forcing ahead", timeout)

    def stop(self) -> Optional[str]:
        """Stop recording and write captured audio to a WAV file.

        The stream stays alive for reuse — only frame collection stops.
        Returns the WAV path, or ``None`` if no usable audio was captured.
        """
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

            # Skip very short recordings (< 0.3s of audio)
            min_samples = int(self._sample_rate * 0.3)
            if len(audio_data) < min_samples:
                logger.debug("Recording too short (%d samples), discarding", len(audio_data))
                return None

            # Skip silent recordings using peak RMS (not overall average, which
            # gets diluted by silence at the end of the recording).
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
        """Release the audio stream.  Call when voice mode is disabled."""
        self._discard()
        # Close stream OUTSIDE the lock to avoid deadlock with audio callback
        self._close_stream_with_timeout()
        logger.info("AudioRecorder shut down")

    # -- private helpers -----------------------------------------------------

    @staticmethod
    def _write_wav(audio_data, *, sample_rate: int = SAMPLE_RATE) -> str:
        """Write numpy int16 audio data to a WAV file; returns the path."""
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
    """Transcribe a WAV via ``tools.transcription_tools.transcribe_audio()``,
    filtering Whisper hallucinations on silent audio.

    Returns dict with ``success``, ``transcript``, and optionally ``error``.
    """
    from tools.transcription_tools import MAX_FILE_SIZE, transcribe_audio

    result = transcribe_audio(wav_path, model=model, source="voice_mode")

    # Only chunk when the provider itself reports "File too large" — local
    # providers have no upload cap and never return this error.
    if not result.get("success") and "File too large" in result.get("error", ""):
        result = _transcribe_wav_in_chunks(wav_path, model=model, max_file_size=MAX_FILE_SIZE)

    # A configured voice-chat stop phrase is checked FIRST and always survives:
    # phrases like "bye" or "okay" overlap the hallucination blocklist/repeat
    # regex, and swallowing them would make saying "bye" fail to end the chat.
    if result.get("success"):
        raw_transcript = result.get("transcript", "")
        if is_whisper_hallucination(raw_transcript) and not is_voice_stop_phrase(raw_transcript):
            logger.info("Filtered Whisper hallucination: %r", result["transcript"])
            return {"success": True, "transcript": "", "filtered": True}

    # Providers that flag no_speech failed to hear words, not to transcribe —
    # treat like silence so the voice loop re-listens quietly instead of
    # surfacing "Transcription failed".
    if result.get("no_speech"):
        return {"success": True, "transcript": "", "no_speech": True}

    return result


def _transcribe_wav_in_chunks(
    wav_path: str,
    *,
    model: Optional[str],
    max_file_size: int,
) -> Dict[str, Any]:
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
                return {
                    "success": False,
                    "transcript": "",
                    "error": f"Chunk {index}/{len(chunk_paths)} failed: {error}",
                }

            transcript = result.get("transcript", "").strip()
            if transcript and not is_whisper_hallucination(transcript):
                transcripts.append(transcript)

        return {
            "success": True,
            "transcript": " ".join(transcripts).strip(),
            "provider": result.get("provider"),
            "chunks": len(chunk_paths),
        }
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
                suffix=".wav",
                dir=_TEMP_DIR,
                delete=False,
            )
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

# Global reference to the active playback process so it can be interrupted.
_active_playback: Optional[subprocess.Popen] = None
_playback_lock = threading.Lock()


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
    # Also stop sounddevice playback if active
    try:
        sd, _ = _import_audio()
        sd.stop()
    except Exception:
        pass


def _is_wsl2_env() -> bool:
    """True when running inside WSL (Microsoft kernel signature in /proc/version).

    Returns False on any error (non-WSL Linux, Docker, SSH, etc.). A
    module-level function so tests can patch it instead of ``builtins.open``.
    """
    try:
        with open("/proc/version", encoding="utf-8", errors="replace") as _fv:
            return "microsoft" in _fv.read().lower()
    except OSError:
        return False


_is_wsl = _is_wsl2_env


def _wsl_powershell_tts_available() -> bool:
    """True when the WSL2 PowerShell TTS playback fallback can be used.

    Only covers OUTPUT (Media.SoundPlayer on the Windows host) — it does NOT
    make microphone recording work, so callers relaxing the audio-environment
    gate must still surface the PulseAudio-bridge guidance for recording.
    """
    return bool(
        _is_wsl2_env()
        and shutil.which("powershell.exe")
        and shutil.which("ffmpeg")
    )


def play_audio_file(file_path: str) -> bool:
    """Play an audio file through the default output device.

    WAV files go through ``sounddevice.play()`` when allowed; otherwise system
    players: ``afplay`` (macOS), the WSL2 PowerShell bridge, ``ffplay``,
    ``aplay`` (Linux). Interruptible via ``stop_playback()``. Returns True on
    success.
    """
    # Ref-count real speaker output for the whole call so the thinking-sound
    # loop (and any other ambient cue) knows audio is flowing right now.
    mark_audio_output_active(True)
    try:
        return _play_audio_file_impl(file_path)
    finally:
        mark_audio_output_active(False)


def _play_wav_via_sounddevice(file_path: str) -> bool:
    """Play a WAV through sounddevice; False if the audio libs are unavailable
    or playback failed (caller falls through to system players)."""
    try:
        sd, np = _import_audio()
        with wave.open(file_path, "rb") as wf:
            frames = wf.readframes(wf.getnframes())
            audio_data = np.frombuffer(frames, dtype=np.int16)
            sample_rate = wf.getframerate()

        # WSLg RDP audio needs a warmup to avoid crackling at the start: the
        # RDP virtual channel takes ~100 ms to stabilise, and the small default
        # blocksize exasperates clock-adjustment jitter (microsoft/wslg#1257).
        if _is_wsl():
            silence_samples = int(0.1 * sample_rate)
            fade_samples = int(0.1 * sample_rate)
            fade = np.linspace(0.0, 1.0, fade_samples, dtype=np.float64)
            audio_float = audio_data.astype(np.float64)
            audio_float[:fade_samples] *= fade
            tail = np.zeros(int(0.05 * sample_rate), dtype=np.int16)
            audio_data = np.concatenate([
                np.zeros(silence_samples, dtype=np.int16),
                audio_float.astype(np.int16),
                tail,
            ])
            blocksize = 4096
        else:
            blocksize = 0  # default (auto)

        _sd_play_blocking(
            sd, audio_data, sample_rate,
            timeout=len(audio_data) / sample_rate + 2.0, blocksize=blocksize,
        )
        return True
    except (ImportError, OSError):
        return False  # audio libs not available, fall through to system players
    except Exception as e:
        logger.debug("sounddevice playback failed: %s", e)
        return False


def _wsl_powershell_player_cmd(file_path: str) -> Optional[List[str]]:
    """Build the WSL2 PowerShell fallback player command, or None.

    In WSL without a PulseAudio bridge ffplay/aplay have no device, but
    Media.SoundPlayer on the Windows host always does: convert to a
    uniquely-named WAV in Windows %TEMP% (so concurrent TTS calls don't
    collide) and play it. The WAV is deleted unconditionally, and the ORIGINAL
    ffmpeg/powershell exit status is re-raised past that cleanup (rm -f always
    exits 0) so the player loop can fall through to the next player.
    """
    if not (shutil.which("powershell.exe") and shutil.which("ffmpeg") and _is_wsl2_env()):
        return None
    try:
        import uuid

        def _out(cmd):
            return subprocess.check_output(cmd, stderr=subprocess.DEVNULL, timeout=3).decode(errors="replace").strip()

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
            shlex.join(["powershell.exe", "-NoProfile", "-Command", ps_script]),
        ])
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


def _set_active_playback(proc) -> None:
    global _active_playback
    with _playback_lock:
        _active_playback = proc


def _run_system_player(cmd: List[str]) -> bool:
    """Run one player to completion (interruptible via stop_playback)."""
    proc = None
    try:
        # Sibling of the TTS/STT credential scrub: system audio players must
        # not inherit gateway tokens / API keys.
        from tools.environments.local import hermes_subprocess_env

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            env=hermes_subprocess_env(inherit_credentials=False),
        )
        _set_active_playback(proc)
        proc.wait(timeout=300)
        rc = proc.returncode
        _set_active_playback(None)
        if rc == 0:
            return True
        # Non-zero exit: e.g. WSL ffplay/aplay with no audio device, or the
        # PowerShell fallback failing. Fall through to the next player.
        logger.debug("System player %s exited with code %d, trying next", cmd[0], rc)
    except subprocess.TimeoutExpired:
        logger.warning("System player %s timed out, killing process", cmd[0])
        if proc is not None:
            proc.kill()
            proc.wait()
        _set_active_playback(None)
    except Exception as e:
        logger.debug("System player %s failed: %s", cmd[0], e)
        _set_active_playback(None)
    return False


def _play_audio_file_impl(file_path: str) -> bool:
    if not os.path.isfile(file_path):
        logger.warning("Audio file not found: %s", file_path)
        return False

    # sounddevice output is skipped on macOS (PortAudio/CoreAudio init triggers
    # a TCC media-library prompt); afplay handles all formats there instead.
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
# replaced per-playback barge monitors that (1) only listened during TTS, so
# the user could not interject during LLM generation, and (2) calibrated their
# noise floor against active speaker bleed with a strict consecutive-block
# counter, making the trigger unreachable for normal speech. This listener
# calibrates against the QUIET room at turn start, freezes that baseline
# through playback, and trips on a windowed majority of blocks.

# Minimum trigger while TTS audio is flowing. Speaker bleed reaching the mic
# is typically a few hundred RMS (~1000-1400 with loud speakers close to the
# mic), while direct speech at normal distance measures 3000-8000 RMS.
PLAYBACK_MIN_TRIGGER = 1500.0

# Absolute trigger ceiling — a noisy room must never push the trigger above
# what normal speech (3000-8000 RMS) can reach.
TRIGGER_CEILING = 4000.0

# Trigger multiplier over the quiet-room floor (typically 50-300 RMS): 3x
# separates speech from ambient cleanly while staying reachable
# (300 RMS floor * 3 = 900 trigger vs 3000+ RMS speech).
DEFAULT_BARGE_MULTIPLIER = 3.0


def _vad_log(msg: str) -> None:
    """VAD decision-point diagnostic — always logger.debug, plus stderr when
    HERMES_VOICE_DEBUG=1 so live hardware tuning doesn't need a log tail."""
    logger.debug(msg)
    if os.environ.get("HERMES_VOICE_DEBUG", "").strip() == "1":
        try:
            print(f"[voice-vad] {msg}", file=sys.stderr, flush=True)
        except Exception:
            pass


def _capture_until_quiet(stream, np, block: int, pre_roll, *, endpoint_blocks: int, max_blocks: int) -> str:
    """Keep reading after a trip until *endpoint_blocks* of quiet (or
    *max_blocks*), then write pre-roll + capture to a WAV and return its path.
    Playback was cut by the trigger, so plain silence endpointing works."""
    frames: List[Any] = list(pre_roll)
    quiet = 0
    for _ in range(max_blocks):
        data, _ = stream.read(block)
        frames.append(data.copy())
        quiet = quiet + 1 if _rms(np, data) < SILENCE_RMS_THRESHOLD else 0
        if quiet >= endpoint_blocks:
            break
    return AudioRecorder._write_wav(np.concatenate(frames, axis=0))


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
    """Listen across an ENTIRE agent turn; return the captured interruption.

    Two phases, decided per 30ms block by *is_playing* (usually
    ``is_audio_output_active``): ``generation`` (no TTS) — the first
    *calibration_ms* of the quiet room set the noise floor and the trigger is
    quiet_floor x *multiplier*; ``playback`` (TTS flowing) — the quiet
    baseline is HELD (never recalibrated against speaker bleed), the trigger
    is clamped up to ``PLAYBACK_MIN_TRIGGER`` so bleed alone can't trip it,
    and a *grace_ms* window after playback starts suppresses onset transients.

    Detection is a windowed majority — >=80% of the last *sustained_ms* of
    blocks above trigger (current block included) — so intra-word dips don't
    reset progress. On detection ``on_trigger(phase)`` fires, capture
    continues from the rolling *pre_roll_ms* buffer until
    *endpoint_silence_ms* of quiet, and the WAV path is returned. Returns
    ``None`` when *should_stop* ends the turn without speech.
    """
    try:
        sd, np = _import_audio()
    except (ImportError, OSError):
        return None

    from collections import deque

    block = int(SAMPLE_RATE * 0.03)  # 30ms blocks
    calib_blocks = max(1, calibration_ms // 30)
    trip_blocks = max(1, sustained_ms // 30)
    trip_needed = max(1, int(round(trip_blocks * 0.8)))
    grace_blocks = max(0, grace_ms // 30)
    endpoint_blocks = max(1, endpoint_silence_ms // 30)
    max_blocks = max(1, max_utterance_ms // 30)
    mult = float(multiplier) if multiplier else DEFAULT_BARGE_MULTIPLIER

    ambient: "deque[float]" = deque(maxlen=100)  # ~3s of quiet-phase RMS
    pre_roll: deque = deque(maxlen=max(1, pre_roll_ms // 30))
    recent_above: "deque[bool]" = deque(maxlen=trip_blocks)
    quiet_floor = float(SILENCE_RMS_THRESHOLD)
    floor_locked = False
    playing_prev = False
    playback_seen = False
    grace_remaining = 0
    blocks_since_playback = 10_000
    block_idx = 0

    def _floor(seq) -> tuple:
        """(pct90, floor): 90th percentile of the quiet-phase RMS window, floor never below the silence threshold."""
        pct90 = float(np.percentile(list(seq), 90)) if seq else float(SILENCE_RMS_THRESHOLD)
        return pct90, max(pct90, float(SILENCE_RMS_THRESHOLD))

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="int16", blocksize=block
        ) as stream:
            while not should_stop():
                data, _ = stream.read(block)
                rms = _rms(np, data)
                pre_roll.append(data.copy())
                block_idx += 1

                playing = bool(is_playing()) if is_playing is not None else False

                # Pre-playback calibration: the listener arms at utterance
                # submit, before any TTS exists, so the first calib_blocks
                # sample the actual quiet room — NOT speaker bleed.
                if not floor_locked:
                    if not playing:
                        ambient.append(rms)
                    if len(ambient) >= calib_blocks or playing:
                        pct90, quiet_floor = _floor(ambient)
                        floor_locked = True
                        _vad_log(
                            f"calibrated quiet floor={quiet_floor:.0f} "
                            f"(pct90={pct90:.0f}, {len(ambient)} blocks, "
                            f"mult={mult:g})"
                        )
                    if not floor_locked:
                        continue

                # Playback phase transitions: grace only when playback starts
                # after a real gap (>=1s), so inter-sentence flapping of the
                # audio-active flag can't chain grace windows together and
                # swallow a genuine interjection.
                if playing and not playing_prev:
                    if not playback_seen or blocks_since_playback > 33:
                        grace_remaining = grace_blocks
                        _vad_log(
                            f"playback started (block={block_idx}) — "
                            f"grace {grace_blocks * 30}ms"
                        )
                    playback_seen = True
                playing_prev = playing
                blocks_since_playback = 0 if playing else blocks_since_playback + 1

                # Trigger: quiet baseline x multiplier, phase-clamped.
                trigger = quiet_floor * mult
                trigger = max(trigger, PLAYBACK_MIN_TRIGGER if playing else float(SILENCE_RMS_THRESHOLD) * 2)
                trigger = min(trigger, TRIGGER_CEILING)

                # Track ambient drift — ONLY while nothing is playing (never
                # absorb speaker bleed) and the block isn't speech.
                if not playing and rms < trigger:
                    ambient.append(rms)
                    _, quiet_floor = _floor(ambient)

                above = rms >= trigger
                if above and grace_remaining > 0:
                    _vad_log(
                        f"grace suppression: block={block_idx} rms={rms:.0f} "
                        f"trigger={trigger:.0f} ({grace_remaining} blocks left)"
                    )
                    above = False
                if grace_remaining > 0:
                    grace_remaining -= 1

                recent_above.append(above)
                if rms >= trigger * 0.5:
                    _vad_log(
                        f"block={block_idx} rms={rms:.0f} floor={quiet_floor:.0f} "
                        f"trigger={trigger:.0f} above={above} "
                        f"window={sum(recent_above)}/{trip_needed} "
                        f"phase={'playback' if playing else 'generation'}"
                    )

                if not (above and sum(recent_above) >= trip_needed):
                    continue

                phase = "playback" if playing else "generation"
                _vad_log(
                    f"TRIPPED ({phase}): block={block_idx} rms={rms:.0f} "
                    f"floor={quiet_floor:.0f} trigger={trigger:.0f} "
                    f"window={sum(recent_above)}/{len(recent_above)}"
                )
                if on_trigger:
                    try:
                        on_trigger(phase)
                    except Exception as e:
                        logger.debug("full-duplex trigger callback failed: %s", e)

                return _capture_until_quiet(
                    stream, np, block, pre_roll,
                    endpoint_blocks=endpoint_blocks, max_blocks=max_blocks,
                )
    except Exception as e:
        logger.debug("Full-duplex listener failed: %s", e)
    return None


# ── Requirements check ──
def _check_plugin_stt_provider(provider: str) -> bool:
    """Return True when *provider* resolves to an available STT plugin."""
    key = (provider or "").lower().strip()
    if not key or key == "none":
        return False
    try:
        from agent.transcription_registry import get_provider
        from hermes_cli.plugins import _ensure_plugins_discovered

        _ensure_plugins_discovered()
        plugin_provider = get_provider(key)
        if plugin_provider is None:
            # Match the transcription dispatcher: long-lived processes may
            # need one refresh after plugins or configuration change.
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
            "check: %s - treating as unavailable",
            key, exc, exc_info=True,
        )
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
    """Check if all voice mode requirements are met.

    Returns dict with ``available``, ``audio_available``, ``stt_available``,
    ``missing_packages``, ``details`` and ``environment``.
    """
    from tools.transcription_tools import (
        _get_provider,
        _load_stt_config,
        _resolve_command_stt_provider_config,
        is_stt_enabled,
    )
    stt_config = _load_stt_config()
    stt_enabled = is_stt_enabled(stt_config)
    stt_provider = _get_provider(stt_config)
    native_stt_available = stt_provider in _NATIVE_STT_LABELS
    command_stt_config = None
    plugin_stt_available = False
    if stt_enabled and not native_stt_available:
        command_stt_config = _resolve_command_stt_provider_config(
            stt_provider, stt_config,
        )
        if command_stt_config is None:
            plugin_stt_available = _check_plugin_stt_provider(stt_provider)
    stt_available = stt_enabled and (
        native_stt_available
        or command_stt_config is not None
        or plugin_stt_available
    )

    missing: List[str] = []
    termux_capture = _termux_voice_capture_available()
    has_audio = _audio_available() or termux_capture

    if not has_audio:
        missing.extend(["sounddevice", "numpy"])

    env_check = detect_audio_environment()

    available = has_audio and stt_available and env_check["available"]
    details_parts = []

    if termux_capture:
        details_parts.append("Audio capture: OK (Termux:API microphone)")
    elif has_audio:
        details_parts.append("Audio capture: OK")
    else:
        details_parts.append(f"Audio capture: MISSING ({_voice_capture_install_hint()})")

    if not stt_enabled:
        details_parts.append("STT provider: DISABLED in config (stt.enabled: false)")
    elif stt_provider in _NATIVE_STT_LABELS:
        details_parts.append(f"STT provider: OK ({_NATIVE_STT_LABELS[stt_provider]})")
    elif command_stt_config is not None:
        details_parts.append(f"STT provider: OK (command: {stt_provider})")
    elif plugin_stt_available:
        details_parts.append(f"STT provider: OK (plugin: {stt_provider})")
    else:
        details_parts.append(
            "STT provider: MISSING (uv pip install faster-whisper — "
            "`pip install faster-whisper` also works if pip is on PATH, "
            "or set GROQ_API_KEY / VOICE_TOOLS_OPENAI_KEY)"
        )

    for warning in env_check["warnings"]:
        details_parts.append(f"Environment: {warning}")
    for notice in env_check.get("notices", []):
        details_parts.append(f"Environment: {notice}")

    return {
        "available": available,
        "audio_available": has_audio,
        "stt_available": stt_available,
        "missing_packages": missing,
        "details": "\n".join(details_parts),
        "environment": env_check,
    }


# ── Temp file cleanup ──
def cleanup_temp_recordings(max_age_seconds: int = 3600) -> int:
    """Remove ``recording_*.wav`` temp files older than *max_age_seconds*
    (default 1 hour); returns the number deleted."""
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
