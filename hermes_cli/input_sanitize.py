"""Sanitize user prompt text leaked from terminal / paste control sequences."""

from __future__ import annotations

import re

_BRACKETED_PASTE_BOUNDARY_START = re.compile(r"(^|[\s\n>:\]\)])\[200~")
_BRACKETED_PASTE_BOUNDARY_END = re.compile(r"\[201~(?=$|[\s\n<\[\(\):;.,!?])")
_BRACKETED_PASTE_DEGRADED_START = re.compile(r"(^|[\s\n>:\]\)])00~")
_BRACKETED_PASTE_DEGRADED_END = re.compile(r"01~(?=$|[\s\n<\[\(\):;.,!?])")

# Corruption signature from desktop bracketed-paste leaks (#62557).
_DESKTOP_PASTE_ARTIFACT = "~[[e"


def strip_leaked_bracketed_paste_wrappers(text: str) -> str:
    """Strip leaked bracketed-paste wrapper markers from user-visible text.

    Canonical wrappers are stripped unconditionally. Degraded visible forms like ``[200~`` /
    ``[201~`` and ``00~`` / ``01~`` are removed only at boundaries so embedded literals such as
    ``literal[200~tag`` stay intact.
    """
    if not text:
        return text
    for wrapper in ("\x1b[200~", "\x1b[201~", "^[[200~", "^[[201~"):
        text = text.replace(wrapper, "")
    text = _BRACKETED_PASTE_BOUNDARY_START.sub(r"\1", text)
    text = _BRACKETED_PASTE_BOUNDARY_END.sub("", text)
    text = _BRACKETED_PASTE_DEGRADED_START.sub(r"\1", text)
    return _BRACKETED_PASTE_DEGRADED_END.sub("", text)


def collapse_repeated_input_artifacts(text: str, min_repeats: int = 4) -> str:
    """Drop a trailing run of the desktop ~[[e corruption signature (#62557)."""
    if not text:
        return text
    marker = _DESKTOP_PASTE_ARTIFACT
    index = len(text)
    repeat_count = 0
    while index >= len(marker) and text[index - len(marker) : index] == marker:
        repeat_count += 1
        index -= len(marker)
    if repeat_count < min_repeats:
        return text
    start = index
    if start >= 2 and text[start - 2 : start] == "[e":
        start -= 2
    elif start >= 1 and text[start - 1] == "[":
        start -= 1
    return text[:start]


def sanitize_user_prompt_text(text: str) -> str:
    """Normalize user-authored prompt text before persistence or model input."""
    if not isinstance(text, str) or not text:
        return text
    return collapse_repeated_input_artifacts(strip_leaked_bracketed_paste_wrappers(text))
