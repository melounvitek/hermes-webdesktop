"""Code-fence helpers shared by the stream consumer and the gateway final send."""

from __future__ import annotations

import re


def escape_code_fences_for_display(text: str) -> str:
    """Replace each ``` with \\`\\`\\` so text can be wrapped in an outer ``` block.

    Reasoning content that quotes code would otherwise break the outer fence.
    """
    if not isinstance(text, str) or "```" not in text:
        return text
    return text.replace("```", "\\`\\`\\`")


def ensure_closed_code_fences(text: str) -> str:
    """Append a closing ``` and/or ` if the text has orphaned code markers.

    Output truncated mid-code-block (token limit, finish_reason="length") would
    otherwise render everything after the orphan as one code block / inline
    span.  Trade-off: a spurious close creates a brief empty span at the end,
    far less harmful than the alternative.  Odd ``` count → append a fence on
    its own line; then, with complete ```…``` regions stripped, odd ` count →
    append a backtick.
    """
    if not isinstance(text, str) or not text:
        return text

    if text.count("```") % 2 == 1:
        text = text.rstrip("\n") + "\n```"

    # Strip complete fenced regions (and any trailing unclosed ``` that leaks
    # through) so their internal backticks don't pollute the standalone count.
    without_fences = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    without_fences = re.sub(r"```[^`]*$", "", without_fences)

    if without_fences.count("`") % 2 == 1:
        text = text + "`"

    return text
