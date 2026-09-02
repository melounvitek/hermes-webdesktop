"""Long-form chunking, ffmpeg encoding, container repair and delivery packing.

Everything here is provider-agnostic post-processing for ``tools.tts_tool``:
split text under a per-request cap, wrap raw PCM as WAV, convert WAV/MP3 to
the target container, sniff/repair mislabelled ``.ogg`` files, and combine
final-encoded chunks under a destination platform's upload limit. Origin
module re-imports every name under its historical spelling.
"""

from __future__ import annotations

import logging
import os
import re
import shlex
import shutil
import struct
import subprocess
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from hermes_cli._subprocess_compat import windows_hide_flags

logger = logging.getLogger("tools.tts_tool")

# Final fallback when provider isn't recognised at all.
FALLBACK_MAX_TEXT_LENGTH = 4000

# PCM output specs for Gemini TTS (fixed by the API)
GEMINI_TTS_SAMPLE_RATE = 24000
GEMINI_TTS_CHANNELS = 1
GEMINI_TTS_SAMPLE_WIDTH = 2  # 16-bit PCM (L16)

# ffmpeg args producing the Ogg/Opus voice-bubble encoding Telegram & co expect.
_OPUS_VOICE_ARGS = [
    "-acodec", "libopus", "-ac", "1", "-b:a", "48k", "-vbr", "on",
    "-application", "voip", "-compression_level", "10",
]


# ===========================================================================
# Text chunking and delivery profiles
# ===========================================================================

@dataclass(frozen=True)
class AudioDeliveryProfile:
    """Destination-platform constraints for generated TTS audio."""

    platform: str
    max_file_bytes: int
    safety_ratio: float = 0.85

    @property
    def target_file_bytes(self) -> int:
        """Conservative packing target below the platform hard limit."""
        return max(1, int(self.max_file_bytes * self.safety_ratio))


_PLATFORM_AUDIO_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "discord": {"max_file_bytes": 10 * 1024 * 1024, "safety_ratio": 0.85},
    "telegram": {"max_file_bytes": 50 * 1024 * 1024, "safety_ratio": 0.85},
    "default": {"max_file_bytes": 10 * 1024 * 1024, "safety_ratio": 0.85},
}


def _resolve_audio_delivery_profile(
    platform: Optional[str],
    tts_config: Optional[Dict[str, Any]] = None,
) -> AudioDeliveryProfile:
    """Resolve upload constraints, including optional ``tts.delivery_profiles`` overrides."""
    key = (platform or "default").lower().strip() or "default"
    defaults = dict(_PLATFORM_AUDIO_DEFAULTS.get(key) or _PLATFORM_AUDIO_DEFAULTS["default"])
    profiles = (tts_config or {}).get("delivery_profiles")
    overrides = profiles.get(key, {}) if isinstance(profiles, dict) else {}
    if isinstance(overrides, dict):
        defaults.update({k: v for k, v in overrides.items() if v is not None})

    max_file_bytes = defaults.get("max_file_bytes")
    if isinstance(max_file_bytes, bool) or not isinstance(max_file_bytes, int) or max_file_bytes <= 0:
        max_file_bytes = _PLATFORM_AUDIO_DEFAULTS["default"]["max_file_bytes"]

    safety_ratio = defaults.get("safety_ratio", 0.85)
    if (
        isinstance(safety_ratio, bool)
        or not isinstance(safety_ratio, (int, float))
        or not 0 < safety_ratio <= 1
    ):
        safety_ratio = 0.85

    return AudioDeliveryProfile(platform=key, max_file_bytes=max_file_bytes, safety_ratio=float(safety_ratio))


def _pack_under_cap(pieces: List[str], max_chars: int) -> List[str]:
    """Greedily join *pieces* with single spaces, starting a new chunk past *max_chars*."""
    chunks: List[str] = []
    current = ""
    for piece in pieces:
        candidate = f"{current} {piece}".strip()
        if current and len(candidate) > max_chars:
            chunks.append(current)
            current = piece
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def _split_oversized_sentence(sentence: str, max_chars: int) -> List[str]:
    """Split one over-limit sentence on word boundaries, then hard boundaries.

    An over-long word flushes the running chunk and emits its slices as their
    own chunks (the tail slice is not merged with following words).
    """
    chunks: List[str] = []
    current = ""
    for word in sentence.split():
        if len(word) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(word[i:i + max_chars] for i in range(0, len(word), max_chars))
            continue
        candidate = f"{current} {word}".strip()
        if current and len(candidate) > max_chars:
            chunks.append(current)
            current = word
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def _split_text_for_tts(text: str, max_chars: int) -> List[str]:
    """Split text under a provider cap without dropping normalized content."""
    if max_chars <= 0:
        max_chars = FALLBACK_MAX_TEXT_LENGTH
    normalized = " ".join((text or "").split())
    if not normalized:
        return []
    if len(normalized) <= max_chars:
        return [normalized]

    expanded: List[str] = []
    for sentence in re.split(r"(?<=[.!?;:,])\s+", normalized):
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) <= max_chars:
            expanded.append(sentence)
        else:
            expanded.extend(_split_oversized_sentence(sentence, max_chars))
    return _pack_under_cap(expanded, max_chars)


def _pack_audio_files_for_delivery(
    audio_paths: List[str],
    profile: AudioDeliveryProfile,
) -> List[List[str]]:
    """Group already-final-encoded chunks under the conservative size target.

    A group never mixes container suffixes (they can't be concat-copied).
    """
    groups: List[List[str]] = []
    current: List[str] = []
    current_size = 0
    current_suffix = ""
    for path in audio_paths:
        size = Path(path).stat().st_size
        suffix = Path(path).suffix.lower()
        if current and (current_size + size > profile.target_file_bytes or suffix != current_suffix):
            groups.append(current)
            current, current_size = [], 0
        current.append(path)
        current_size += size
        current_suffix = suffix
    if current:
        groups.append(current)
    return groups


# ===========================================================================
# ffmpeg encoding helpers
# ===========================================================================

def _has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def _ffmpeg_run(args: List[str], *, timeout: int = 30) -> subprocess.CompletedProcess:
    """Run ``ffmpeg <args>`` headless (no stdin, hidden window on Windows)."""
    return subprocess.run(
        ["ffmpeg", *args],
        capture_output=True,
        timeout=timeout,
        stdin=subprocess.DEVNULL,
        creationflags=windows_hide_flags(),
    )


def _wav_sidecar_path(output_path: str) -> str:
    """Path a WAV-native engine writes to before conversion to *output_path*'s format."""
    if output_path.endswith(".wav"):
        return output_path
    return output_path.rsplit(".", 1)[0] + ".wav"


def _finalize_wav_output(wav_path: str, output_path: str) -> str:
    """Move a WAV-native engine's output into the caller's requested container.

    Shared by NeuTTS / Piper / KittenTTS: ffmpeg-convert when available,
    otherwise rename the WAV to the expected path so the tool stays usable
    (the extension is then misleading but the audio plays).
    """
    if wav_path == output_path:
        return output_path
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        subprocess.run(
            [ffmpeg, "-i", wav_path, "-y", "-loglevel", "error", output_path],
            check=True, timeout=30, stdin=subprocess.DEVNULL, creationflags=windows_hide_flags(),
        )
        try:
            os.remove(wav_path)
        except OSError:
            pass
    else:
        os.rename(wav_path, output_path)
    return output_path


def _wrap_pcm_as_wav(
    pcm_bytes: bytes,
    sample_rate: int = GEMINI_TTS_SAMPLE_RATE,
    channels: int = GEMINI_TTS_CHANNELS,
    sample_width: int = GEMINI_TTS_SAMPLE_WIDTH,
) -> bytes:
    """Wrap raw signed-little-endian PCM (e.g. Gemini's L16) with a minimal WAV RIFF header."""
    byte_rate = sample_rate * channels * sample_width
    block_align = channels * sample_width
    data_size = len(pcm_bytes)
    fmt_chunk = struct.pack(
        "<4sIHHIIHH", b"fmt ", 16, 1, channels, sample_rate, byte_rate, block_align, sample_width * 8,
    )
    data_chunk_header = struct.pack("<4sI", b"data", data_size)
    riff_size = 4 + len(fmt_chunk) + len(data_chunk_header) + data_size
    riff_header = struct.pack("<4sI4s", b"RIFF", riff_size, b"WAVE")
    return riff_header + fmt_chunk + data_chunk_header + pcm_bytes


def _write_wav_bytes_as(wav_bytes: bytes, output_path: str) -> str:
    """Write in-memory WAV to *output_path*, ffmpeg-converting to its container.

    ``.wav`` is written directly; ``.ogg`` is forced to Opus (ffmpeg's .ogg
    default is Vorbis, which voice bubbles reject); anything else is a plain
    ffmpeg conversion. A failed conversion raises RuntimeError. Without
    ffmpeg the raw WAV is written under the requested name (misleading
    extension, but the audio still plays).
    """
    if output_path.lower().endswith(".wav"):
        with open(output_path, "wb") as f:
            f.write(wav_bytes)
        return output_path

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(wav_bytes)
        wav_path = tmp.name
    try:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            opus = _OPUS_VOICE_ARGS if output_path.lower().endswith(".ogg") else []
            cmd = [ffmpeg, "-i", wav_path, *opus, "-y", "-loglevel", "error", output_path]
            result = subprocess.run(cmd, capture_output=True, timeout=30, stdin=subprocess.DEVNULL, creationflags=windows_hide_flags())
            if result.returncode != 0:
                stderr = result.stderr.decode("utf-8", errors="ignore")[:300]
                raise RuntimeError(f"ffmpeg conversion failed: {stderr}")
        else:
            logger.warning(
                "ffmpeg not found; writing raw WAV to %s (extension may be misleading)",
                output_path,
            )
            shutil.copyfile(wav_path, output_path)
    finally:
        try:
            os.remove(wav_path)
        except OSError:
            pass
    return output_path


def _convert_to_opus(mp3_path: str) -> Optional[str]:
    """Convert any ffmpeg-readable audio file to OGG Opus next to it; None on failure."""
    if not _has_ffmpeg():
        return None
    return _ffmpeg_transcode_to_opus(mp3_path, mp3_path.rsplit(".", 1)[0] + ".ogg")


def _ffmpeg_transcode_to_opus(input_path: str, ogg_path: str) -> Optional[str]:
    """Transcode *input_path* to real Ogg/Opus at *ogg_path* via ffmpeg.

    Safe when ``input_path == ogg_path`` (writes to a temp file, then
    replaces). Returns the output path on success, None on failure.
    """
    if not _has_ffmpeg():
        return None

    in_place = os.path.abspath(input_path) == os.path.abspath(ogg_path)
    work_path = ogg_path + ".tmp.ogg" if in_place else ogg_path
    try:
        result = _ffmpeg_run(["-i", input_path, *_OPUS_VOICE_ARGS, "-f", "ogg", work_path, "-y"])
        if result.returncode != 0:
            logger.warning("ffmpeg conversion failed with return code %d: %s",
                          result.returncode, result.stderr.decode('utf-8', errors='ignore')[:200])
            return None
        if os.path.exists(work_path) and os.path.getsize(work_path) > 0:
            if in_place:
                os.replace(work_path, ogg_path)
            return ogg_path
    except subprocess.TimeoutExpired:
        logger.warning("ffmpeg OGG conversion timed out after 30s")
    except FileNotFoundError:
        logger.warning("ffmpeg not found in PATH")
    except Exception as e:
        logger.warning("ffmpeg OGG conversion failed: %s", e, exc_info=True)
    finally:
        if in_place and os.path.exists(work_path):
            try:
                os.remove(work_path)
            except OSError:
                pass
    return None


# ===========================================================================
# Container sniffing / repair
# ===========================================================================
# Several backends silently ignore the requested opus format (Edge only emits
# MP3, Piper writes WAV, xAI writes MP3, some OpenAI-compatible servers ignore
# response_format="opus"), which breaks native voice bubbles. Sniff the magic
# bytes once after synthesis and repair when they don't match the extension.

def _sniff_audio_container(path: str) -> str:
    """Return a container id ('ogg', 'wav', 'mp3', 'flac', ...) or 'unknown'."""
    from tools.audio_container import sniff_container

    try:
        with open(path, "rb") as fh:
            head = fh.read(12)
    except OSError:
        return "unknown"
    return sniff_container(head) or "unknown"


def _repair_ogg_container(file_str: str) -> str:
    """Ensure a path claiming ``.ogg`` actually contains an Ogg container.

    MP3/WAV/FLAC bytes are transcoded in place to real Ogg/Opus. On failure
    the file is renamed to its sniffed real extension so platforms get an
    honest file instead of a 0-second voice bubble.
    """
    if not file_str.endswith(".ogg"):
        return file_str
    container = _sniff_audio_container(file_str)
    if container in ("ogg", "unknown"):
        return file_str

    logger.info(
        "TTS wrote %s bytes into a .ogg path (%s) — transcoding to real Ogg/Opus",
        container, file_str,
    )
    repaired = _ffmpeg_transcode_to_opus(file_str, file_str)
    if repaired:
        return repaired

    honest = file_str[:-4] + "." + container
    try:
        os.replace(file_str, honest)
        logger.warning(
            "Could not transcode %s to Ogg/Opus — renamed to %s so the "
            "file is delivered with its real format", file_str, honest,
        )
        return honest
    except OSError:
        return file_str


# ===========================================================================
# Long-form audio combination and delivery packing
# ===========================================================================

def _concat_audio_files(
    audio_paths: List[str],
    output_path: str,
    *,
    voice_compatible: bool = False,
) -> Optional[str]:
    """Combine independently encoded chunks with ffmpeg.

    OGG/Opus is always decoded and re-encoded (even without voice opt-in);
    matching MP3 chunks keep their encoded frames (``-c:a copy``). Structured
    containers are never byte-joined. Returns ``None`` when ffmpeg is missing
    or fails so callers keep the individually valid files.
    """
    if not audio_paths:
        raise ValueError("No audio chunks to combine")
    if len(audio_paths) == 1:
        source = audio_paths[0]
        if os.path.abspath(source) != os.path.abspath(output_path):
            shutil.copyfile(source, output_path)
        return output_path

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return None

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    concat_path = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.concat.txt")
    temp_output = destination.with_name(
        f".{destination.stem}.{uuid.uuid4().hex}.combining{destination.suffix}"
    )
    try:
        with concat_path.open("w", encoding="utf-8") as concat_file:
            for path in audio_paths:
                concat_file.write(f"file {shlex.quote(os.path.abspath(path))}\n")

        command = [
            ffmpeg, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
            "-i", str(concat_path), "-vn",
        ]
        suffix = destination.suffix.lower()
        if voice_compatible or suffix in {".ogg", ".opus"}:
            command.extend(["-c:a", "libopus", "-ac", "1", "-b:a", "64k", "-vbr", "off"])
        elif suffix == ".mp3" and all(Path(path).suffix.lower() == ".mp3" for path in audio_paths):
            command.extend(["-c:a", "copy"])
        command.append(str(temp_output))

        result = subprocess.run(
            command,
            capture_output=True,
            timeout=120,
            stdin=subprocess.DEVNULL,
            creationflags=windows_hide_flags(),
        )
        if result.returncode == 0 and temp_output.exists() and temp_output.stat().st_size > 0:
            os.replace(temp_output, destination)
            return str(destination)
        logger.warning(
            "ffmpeg audio combine failed: %s",
            result.stderr.decode("utf-8", errors="ignore")[:500],
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning("ffmpeg audio combine failed: %s", exc)
    finally:
        for path in (concat_path, temp_output):
            try:
                path.unlink()
            except OSError:
                pass
    return None


def _build_audio_delivery_files(
    audio_paths: List[str],
    output_path: str,
    profile: AudioDeliveryProfile,
    *,
    voice_compatible: bool = False,
) -> Tuple[List[str], bool]:
    """Pack final-encoded chunks and enforce the hard upload limit.

    Groups are packed against the conservative target, then every combined
    artifact is checked at its real post-encoding size; an over-limit group is
    split in half and retried. A failed combine returns the constituent files
    separately. A single chunk above the hard limit fails closed. Returns
    ``(final_paths, combined_any)``.
    """
    if not audio_paths:
        raise ValueError("No final-encoded TTS audio chunks")
    for path in audio_paths:
        size = Path(path).stat().st_size
        if size > profile.max_file_bytes:
            raise ValueError(
                f"Final-encoded TTS chunk exceeds {profile.platform} delivery "
                f"limit ({size} > {profile.max_file_bytes} bytes): {path}"
            )

    base = Path(output_path)
    scratch_outputs: List[str] = []
    combined_any = False
    combine_index = 0

    def emit(group: List[str]) -> List[str]:
        nonlocal combined_any, combine_index
        if len(group) == 1:
            return list(group)

        combine_index += 1
        scratch = base.with_name(
            f".{base.stem}.delivery{combine_index:03d}.{uuid.uuid4().hex}{base.suffix}"
        )
        combined = _concat_audio_files(group, str(scratch), voice_compatible=voice_compatible)
        if not combined:
            return list(group)
        scratch_outputs.append(combined)
        if Path(combined).stat().st_size <= profile.max_file_bytes:
            combined_any = True
            return [combined]

        try:
            Path(combined).unlink()
        except OSError:
            pass
        midpoint = max(1, len(group) // 2)
        return emit(group[:midpoint]) + emit(group[midpoint:])

    packed: List[str] = []
    for group in _pack_audio_files_for_delivery(audio_paths, profile):
        packed.extend(emit(group))

    final_paths: List[str] = []
    for index, source in enumerate(packed, start=1):
        if len(packed) == 1:
            destination = base
        else:
            source_suffix = Path(source).suffix or base.suffix
            destination = base.with_name(f"{base.stem}.part{index:02d}{source_suffix}")
        if os.path.abspath(source) != os.path.abspath(destination):
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source, destination)
        if destination.stat().st_size > profile.max_file_bytes:
            raise ValueError(
                f"Final TTS deliverable exceeds {profile.platform} delivery limit: {destination}"
            )
        final_paths.append(str(destination))

    try:
        return final_paths, combined_any
    finally:
        for scratch in scratch_outputs:
            if scratch not in final_paths:
                try:
                    Path(scratch).unlink()
                except OSError:
                    pass
