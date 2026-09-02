"""Pure-text helpers for voice mode: Whisper hallucination filter, voice-chat
stop phrases, and the TTS self-echo guard. No audio dependencies."""

import difflib
import re
from typing import Optional


def _voice_config() -> dict:
    """``voice`` section of config.yaml, or ``{}`` when missing, malformed,
    or the config system can't be imported (broken config mid-install)."""
    try:
        from hermes_cli.config import load_config
        voice_cfg = load_config().get("voice", {})
        return voice_cfg if isinstance(voice_cfg, dict) else {}
    except Exception:
        return {}


# Whisper commonly hallucinates these phrases on silent/near-silent audio
# (matched with trailing '.'/'!' stripped, so the bare form suffices).
WHISPER_HALLUCINATIONS = {
    "thank you", "thanks for watching", "subscribe to my channel", "like and subscribe",
    "please subscribe", "thank you for watching", "bye", "you", "the end",
    # Non-English hallucinations (common on silence)
    "продолжение следует", "sous-titres", "sous-titres réalisés par la communauté d'amara.org",
    "sottotitoli creati dalla comunità amara.org", "untertitel von stephanie geiges",
    "amara.org", "www.mooji.org", "ご視聴ありがとうございました",
}


# Repetitive hallucinations (e.g. "Thank you. Thank you. Thank you.")
_HALLUCINATION_REPEAT_RE = re.compile(
    r'^(?:thank you|thanks|bye|you|ok|okay|the end|\.|\s|,|!)+$',
    flags=re.IGNORECASE,
)


def is_whisper_hallucination(transcript: str) -> bool:
    """Check if a transcript is a known Whisper hallucination on silence."""
    cleaned = transcript.strip().lower()
    if not cleaned:
        return True
    return (
        cleaned.rstrip('.!') in WHISPER_HALLUCINATIONS
        or bool(_HALLUCINATION_REPEAT_RE.match(cleaned))
    )


DEFAULT_VOICE_STOP_PHRASES = ("stop",)


def _load_voice_stop_phrases() -> tuple:
    """Configured ``voice.stop_phrases`` (default ``("stop",)``); an empty tuple
    disables the feature. Malformed config (dict, list of non-strings) falls
    back to the default rather than crashing the voice loop."""
    try:
        raw = _voice_config().get("stop_phrases", DEFAULT_VOICE_STOP_PHRASES)
        if isinstance(raw, str):
            raw = [raw]
        if isinstance(raw, (list, tuple)):
            return tuple(
                str(p).strip().lower() for p in raw
                if isinstance(p, (str, int, float)) and str(p).strip()
            )
    except Exception:
        pass
    return DEFAULT_VOICE_STOP_PHRASES


def _configured_stop_phrases() -> tuple:
    """Resolve ``_load_voice_stop_phrases`` through ``tools.voice_mode`` so
    ``patch("tools.voice_mode._load_voice_stop_phrases")`` still takes effect."""
    from tools import voice_mode as _vm
    return _vm._load_voice_stop_phrases()


def is_voice_stop_phrase(transcript: str, stop_phrases: Optional[tuple] = None) -> bool:
    """True when *transcript* is EXACTLY a configured stop phrase.

    Deliberately strict: the whole utterance — lowercased, surrounding
    punctuation stripped — must equal a phrase, so "stop doing that and try
    again" still reaches the agent. ``voice.stop_phrases: []`` disables.
    """
    if not transcript:
        return False
    cleaned = transcript.strip().lower().strip(".,!?;: \t\n\"'")
    if not cleaned:
        return False
    if stop_phrases is None:
        stop_phrases = _configured_stop_phrases()
    return cleaned in stop_phrases


# Similarity ratio (difflib.SequenceMatcher) above which a playback-phase barge
# transcript is treated as a self-capture of Hermes' own TTS: the full-duplex
# listener has no echo cancellation, so speaker bleed can trip the barge
# trigger and get transcribed near-verbatim (a TTS -> STT -> TTS loop).
DEFAULT_TTS_ECHO_SIMILARITY_THRESHOLD = 0.6


# Minimum normalized-transcript length before the sliding-window fallback
# runs. Below this a genuine one-word barge-in ("yes") landing verbatim inside
# a longer reply would score a trivial 1.0 and be misread as self-capture; a
# real self-capture spans pre-roll plus time-to-silence, so it is longer.
MIN_FRAGMENT_LENGTH_FOR_ECHO = 10


def _normalize_for_echo_compare(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def is_tts_echo(
    transcript: str,
    spoken_text: str,
    threshold: float = DEFAULT_TTS_ECHO_SIMILARITY_THRESHOLD,
) -> bool:
    """True when *transcript* looks like a self-capture of *spoken_text*.

    Character-level similarity (language-agnostic, no word tokenization): a
    genuine user interjection is very unlikely to closely match Hermes' own
    words, so a high ratio signals speaker-bleed self-capture (fail-closed
    guard for the playback-phase listener, which has no echo cancellation).

    The playback-phase capture is cut when the trigger fires and only spans
    pre-roll plus time-to-silence, so for replies longer than a clause the
    transcript is a short FRAGMENT of `spoken_text` and the whole-string
    ratio dilutes toward 0. When it misses, a window sized to the transcript
    slides across `spoken_text` (character-based, so it works without word
    boundaries). Transcripts shorter than `MIN_FRAGMENT_LENGTH_FOR_ECHO` skip
    this fallback: a short interjection trivially matches a short window.
    """
    if not transcript or not spoken_text:
        return False
    a = _normalize_for_echo_compare(transcript)
    b = _normalize_for_echo_compare(spoken_text)
    if not a or not b:
        return False

    def _similar(x: str, y: str) -> bool:
        return difflib.SequenceMatcher(None, x, y).ratio() >= threshold

    if _similar(a, b):
        return True
    if len(a) < MIN_FRAGMENT_LENGTH_FOR_ECHO or len(a) >= len(b):
        return False
    return any(_similar(a, b[start : start + len(a)]) for start in range(0, len(b) - len(a) + 1))


def voice_stop_hint() -> str:
    """One-line 'Say "stop" to end the voice chat.' hint for voice-mode start.

    Uses the first ``voice.stop_phrases`` entry so a custom phrase renders
    correctly; returns "" when stop phrases are disabled so surfaces show no
    hint. Every surface announcing voice-mode start (CLI, TUI, desktop) uses
    this one owner instead of hardcoding the wording.
    """
    phrases = _configured_stop_phrases()
    if not phrases:
        return ""
    return f'Say "{phrases[0]}" to end the voice chat.'
