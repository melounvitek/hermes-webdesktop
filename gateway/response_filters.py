"""Gateway response filtering helpers.

These decide whether a completed agent turn should be delivered to the chat,
not what should be persisted in conversation history.
"""

from __future__ import annotations

import unicodedata
from typing import Any

# Exact whole-response markers meaning "the agent intentionally chose not to
# reply". Keep small and explicit; arbitrary empty output remains an
# error/empty-response path, not silence.
LIVE_GATEWAY_SILENT_MARKERS = frozenset({
    "[SILENT]",
    "SILENT",
    "NO_REPLY",
    "NO REPLY",
})

# Longer than any marker could plausibly be, even with stray punctuation.
_MARKER_LENGTH_CAP = 64


def _canonical_silence_candidate(text: str) -> str:
    return " ".join(text.strip().upper().split())


def _is_edge_punctuation(ch: str) -> bool:
    # Square brackets stay structural so malformed ``[SILENT`` cannot become ``SILENT``.
    return ch not in "[]" and unicodedata.category(ch).startswith("P")


def _strip_edge_silence_punctuation(text: str) -> str:
    """Strip stray edge punctuation (``.NO_REPLY``, ``*NO_REPLY*``) without erasing marker structure."""
    start, end = 0, len(text)
    while start < end and _is_edge_punctuation(text[start]):
        start += 1
    while end > start and _is_edge_punctuation(text[end - 1]):
        end -= 1
    return text[start:end].strip()


def _canonical_silence_candidates(text: str) -> tuple[str, ...]:
    exact = _canonical_silence_candidate(text)
    stripped = _strip_edge_silence_punctuation(text.strip())
    if stripped == text.strip():
        return (exact,)
    return (exact, _canonical_silence_candidate(stripped))


def _short_stripped(text: Any) -> str:
    """Stripped text if it is a non-empty string within the marker cap, else ''."""
    if not isinstance(text, str):
        return ""
    stripped = text.strip()
    return stripped if 0 < len(stripped) <= _MARKER_LENGTH_CAP else ""


def is_intentional_silence_response(response: Any) -> bool:
    """True only when ``response`` is exactly a silence marker.

    Prose that merely mentions ``NO_REPLY`` must be delivered normally. A blank
    response is not silence either — that is the empty-response failure path.
    """
    stripped = _short_stripped(response)
    return bool(stripped) and any(
        c in LIVE_GATEWAY_SILENT_MARKERS for c in _canonical_silence_candidates(stripped)
    )


def is_autonomous_silence_response(response: Any) -> bool:
    """Loose silence matcher for autonomous lanes (cron, webhook).

    Autonomous lanes ask for ``[SILENT]`` when a tick produced nothing worth
    attention, and models reliably bracket the marker with a short note. Unlike
    :func:`is_intentional_silence_response` (interactive rule: EXACTLY a marker),
    this suppresses when a marker is the whole response, sits on its own first or
    last line, or the bracketed sentinel opens the response (``[SILENT] No
    changes detected``). A token buried mid-sentence is still delivered.
    Shares :data:`LIVE_GATEWAY_SILENT_MARKERS` so the two sets cannot drift.
    """
    if not isinstance(response, str):
        return False
    stripped = response.strip()
    if not stripped:
        return False

    def _is_token(line: str) -> bool:
        return _canonical_silence_candidate(line) in LIVE_GATEWAY_SILENT_MARKERS

    if _is_token(stripped):
        return True
    lines = [ln for ln in stripped.splitlines() if ln.strip()]
    if lines and (_is_token(lines[0]) or _is_token(lines[-1])):
        return True
    # Bracketed form only, so a bare "Silent retry succeeded" is NOT swallowed.
    return stripped.upper().startswith("[SILENT]")


def is_intentional_silence_agent_result(agent_result: dict | None, response: Any) -> bool:
    """Silence markers suppress delivery only for successful agent turns."""
    if not isinstance(agent_result, dict) or agent_result.get("failed"):
        return False
    return is_intentional_silence_response(response)


def is_partial_silence_marker(text: Any) -> bool:
    """True while streamed ``text`` could still resolve to a silence marker.

    The streaming path must decide, before the whole response is known, whether
    to show its buffer. A buffer whose canonical form is a non-empty *prefix* of
    a marker (``"NO"`` on the way to ``"NO_REPLY"``, or an exact marker not yet
    terminated by stream-end) is held back so a raw marker is never shown and
    then retracted. Anything that has diverged from every marker, or exceeds the
    marker cap, returns False so normal streaming resumes. Shares the marker set
    and canonicalization with :func:`is_intentional_silence_response`.
    """
    stripped = _short_stripped(text)
    return bool(stripped) and any(
        c and any(marker.startswith(c) for marker in LIVE_GATEWAY_SILENT_MARKERS)
        for c in _canonical_silence_candidates(stripped)
    )
