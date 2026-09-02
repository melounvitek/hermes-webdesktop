"""Strip ANSI escape sequences from subprocess output.

Used by terminal_tool, code_execution_tool, and process_registry so ANSI codes
never enter the model's context (the root cause of models copying escape
sequences into file writes).

Covers the full ECMA-48 spec: CSI (including private-mode ``?`` prefix,
colon-separated params, intermediate bytes), OSC (BEL and ST terminators),
DCS/SOS/PM/APC string sequences, nF multi-byte escapes, Fp/Fe/Fs
single-byte escapes, and 8-bit C1 control characters.
"""

import re

_ANSI_ESCAPE_RE = re.compile(
    r"\x1b"
    r"(?:"
        r"\[[\x30-\x3f]*[\x20-\x2f]*[\x40-\x7e]"     # CSI sequence
        r"|\][\s\S]*?(?:\x07|\x1b\\)"                  # OSC (BEL or ST terminator)
        r"|[PX^_][\s\S]*?(?:\x1b\\)"                   # DCS/SOS/PM/APC strings
        r"|[\x20-\x2f]+[\x30-\x7e]"                    # nF escape sequences
        r"|[\x30-\x7e]"                                 # Fp/Fe/Fs single-byte
    r")"
    r"|\x9b[\x30-\x3f]*[\x20-\x2f]*[\x40-\x7e]"       # 8-bit CSI
    r"|\x9d[\s\S]*?(?:\x07|\x9c)"                       # 8-bit OSC
    r"|[\x80-\x9f]",                                    # Other 8-bit C1 controls
    re.DOTALL,
)

# Fast-path check — skip full regex when no escape-like bytes are present.
_HAS_ESCAPE = re.compile(r"[\x1b\x80-\x9f]")

# C0 controls (minus tab/newline/CR, handled separately) plus DEL. They survive
# strip_ansi() — it only removes well-formed *sequences* — but are dangerous when
# echoed to a terminal (BEL rings, backspace/DEL overwrite, NUL truncates).
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# Fast-path check for sanitize_display_text — any C0 control (except
# tab/newline), CR, DEL, ESC, or C1 byte triggers the slow path.
_HAS_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")

# Unicode TAG characters (U+E0000–U+E007F) render as nothing in terminals and
# chat UIs but are visible to LLM tokenizers — the "ASCII smuggling" injection
# channel. The only legitimate modern use is emoji tag sequences (TR51: U+1F3F4
# base + tag spec + U+E007F CANCEL TAG, e.g. the Scotland/Wales flags); those
# are preserved, same rationale as keeping ZWJ inside emoji sequences.
_UNICODE_TAG_SUB_RE = re.compile(
    r"(\U0001F3F4[\U000E0020-\U000E007E]+\U000E007F)"  # valid emoji tag seq (kept)
    r"|[\U000E0000-\U000E007F]"                        # any other tag char (stripped)
)

# Fast-path check — plane-14 tag chars only.
_HAS_UNICODE_TAG = re.compile(r"[\U000E0000-\U000E007F]")


def strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences; clean text passes through unchanged (fast path)."""
    if not text or not _HAS_ESCAPE.search(text):
        return text
    return _ANSI_ESCAPE_RE.sub("", text)


def sanitize_display_text(text: str) -> str:
    """Sanitize stored/untrusted text before echoing it to a terminal.

    Removes ANSI/ECMA-48 sequences AND bare control characters, keeping only
    newlines and tabs (CRs become newlines so ``\\r``-overwrite spoofing can't
    hide content). Use when re-rendering persisted text (e.g. the ``/resume``
    recap): Rich's ``Text()`` does NOT neutralize raw escape bytes, so a replayed
    message must not be able to clear the screen, retitle the window, or restyle UI.
    """
    if not text or not _HAS_CONTROL.search(text):
        return text
    text = strip_ansi(text)
    if "\r" in text:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
    return _CONTROL_CHARS_RE.sub("", text)


def strip_unicode_tags(text: str) -> str:
    """Remove invisible Unicode TAG characters (U+E0000–U+E007F) from text.

    A prompt-injection smuggling channel for untrusted tool output (MCP servers,
    web content). Valid emoji tag sequences (regional flags) are preserved;
    tag-free input is returned unchanged (fast path).
    """
    if not text or not _HAS_UNICODE_TAG.search(text):
        return text
    return _UNICODE_TAG_SUB_RE.sub(lambda m: m.group(1) or "", text)
