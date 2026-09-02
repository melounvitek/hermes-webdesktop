"""Boundary-aware partial compression — "summarize up to here".

* **Role alternation.** The compressed head ends with summary/handoff content (assistant- or user-
role, possibly a trailing todo snapshot). The verbatim tail must begin with a ``user`` message so
the rejoined history keeps the user↔assistant alternation that providers validate.

* **No silent context mutation.** This is a manual, user-invoked action. It rotates the session
exactly like ``/compress`` does (via the caller), so the prompt-cache reset is explicit and
expected, not silent.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

#: Default number of recent exchanges to preserve verbatim when the user
#: runs ``/compress here`` without an explicit count.
DEFAULT_KEEP_LAST = 2

#: Hard ceiling so a fat-fingered ``/compress here 9999`` doesn't turn
#: into a no-op surprise — clamp instead.
MAX_KEEP_LAST = 100


def parse_partial_compress_args(
    raw_args: str,
) -> Tuple[bool, int, Optional[str]]:
    """Parse the argument string after ``/compress``.

    Anything else is treated as a focus topic for the existing full ``/compress <focus>`` behavior.

    * ``partial`` — True when a boundary-aware form was requested. * ``keep_last`` — exchanges to
    preserve verbatim (only meaningful when ``partial`` is True). * ``focus_topic`` — focus string
    for full compression, or None.
    """
    text = (raw_args or "").strip()
    if not text:
        return False, DEFAULT_KEEP_LAST, None

    lowered = text.lower()

    # Normalize the "up to here" alias to "here".
    if lowered.startswith("up to here"):
        lowered = lowered[len("up to ") :]
        text = text[len("up to ") :]

    tokens = lowered.split()
    head = tokens[0] if tokens else ""

    # Form: here [N]
    if head == "here":
        keep = _coerce_keep(tokens[1]) if len(tokens) >= 2 else DEFAULT_KEEP_LAST
        return True, keep, None

    # Form: --keep N  (or --keep=N)
    if head in ("--keep", "-k") and len(tokens) >= 2:
        return True, _coerce_keep(tokens[1]), None
    if head.startswith("--keep="):
        return True, _coerce_keep(head.split("=", 1)[1]), None

    # Otherwise: full compression with this as the focus topic.
    return False, DEFAULT_KEEP_LAST, text or None


def extract_compress_flags(raw_args: str) -> Tuple[str, bool, bool]:
    """Strip ``--preview``/``--dry-run``/``--aggressive`` from the ``/compress`` argument string.

    Flags may appear anywhere alongside the positional forms (``here [N]``, ``--keep N``, focus
    topic); the remainder is what :func:`parse_partial_compress_args` should see. Returns
    ``(remaining_args, preview, aggressive_requested)``. ``preview`` (``--preview``/``--dry-
    run``) means report what WOULD be compressed and change nothing. No surface implements an
    LLM-free hard-truncate path, so callers surface "not supported" for ``--aggressive`` instead
    of treating it as a focus topic.
    """
    preview = False
    aggressive = False
    kept: List[str] = []
    for tok in (raw_args or "").split():
        low = tok.lower()
        if low in ("--preview", "--dry-run", "--dryrun"):
            preview = True
        elif low == "--aggressive":
            aggressive = True
        else:
            kept.append(tok)
    return " ".join(kept), preview, aggressive


def summarize_compress_preview(
    history: List[Dict[str, Any]],
    partial: bool,
    keep_last: int,
    focus_topic: Optional[str],
    approx_tokens: int,
) -> Dict[str, Any]:
    """Build the ``/compress --preview`` report — pure, no side effects.

    Shared by the CLI and the gateway slash handler so both surfaces report the same numbers the
    real run would use. Returns ``head_count``/``tail_count``/``lines`` (ready-to-print
    strings).
    """
    total = len(history)
    head = list(history)
    tail: List[Dict[str, Any]] = []
    effective_partial = partial
    if partial:
        head, tail = split_history_for_partial_compress(history, keep_last)
        if not tail:
            # Same degenerate-split fallback the real run applies.
            effective_partial = False
            head, tail = list(history), []

    lines = [
        "Preview — no changes made.",
        f"Would compress {len(head)} of {total} message(s) "
        f"(~{approx_tokens:,} tokens currently in context).",
    ]
    if effective_partial:
        lines.append(
            f"Boundary: keeping the last {keep_last} exchange(s) "
            f"({len(tail)} message(s)) verbatim."
        )
    elif partial:
        lines.append(
            "Boundary: 'here' split would keep everything — "
            "falling back to full compression."
        )
    if focus_topic:
        lines.append(f'Focus topic: "{focus_topic}"')
    lines.append("Run the command again without --preview to apply.")

    return {
        "head_count": len(head),
        "tail_count": len(tail),
        "total": total,
        "partial": effective_partial,
        "lines": lines,
    }


def _coerce_keep(value: str) -> int:
    """Parse a keep-count token, clamping to [1, MAX_KEEP_LAST]."""
    try:
        n = int(value)
    except (TypeError, ValueError):
        return DEFAULT_KEEP_LAST
    return max(1, min(n, MAX_KEEP_LAST))


def split_history_for_partial_compress(
    history: List[Dict[str, Any]],
    keep_last: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split ``history`` into ``(head, tail)`` for partial compression.

    ``head`` is summarized; ``tail`` is the last ``keep_last`` exchanges kept verbatim.
    Exchanges are counted by ``user`` messages so the tail always starts on a user turn and
    rejoining ``compressed_head + tail`` keeps user↔assistant alternation valid. Returns
    ``(history, [])`` when the head would be empty, signaling the caller to fall back to full
    compression or "nothing to do".
    """
    if keep_last < 1:
        keep_last = 1

    n = len(history)
    if n == 0:
        return [], []

    # Walk backwards collecting the indices of the most recent `keep_last`
    # user-message starts. The tail begins at the earliest such index.
    user_starts: List[int] = []
    for idx in range(n - 1, -1, -1):
        if history[idx].get("role") == "user":
            user_starts.append(idx)
            if len(user_starts) >= keep_last:
                break

    if not user_starts:
        # No user turns at all (degenerate) — nothing sensible to keep
        # as a "recent exchange"; treat as full compression.
        return list(history), []

    boundary = user_starts[-1]  # earliest of the kept user starts

    head = history[:boundary]
    tail = history[boundary:]

    # If everything is in the tail (nothing left to compress), signal the
    # caller to fall back to full compression rather than producing a
    # no-op that rotates the session for no benefit.
    if not head:
        return list(history), []

    return head, tail


def rejoin_compressed_head_and_tail(
    compressed_head: List[Dict[str, Any]],
    tail: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Concatenate a compressed head with the verbatim tail, defending the seam's role alternation.

    The compressed head normally ends on an assistant/tool turn, but the head compressor's
    output shape isn't contractually guaranteed (a plugin engine could end on a user turn). If
    the last head message and first tail message share a user/assistant role, the tail's first
    content is folded onto the head's last message so provider role-alternation rules hold.
    ``tool`` messages are left alone — consecutive tool entries are the one legal repetition
    (parallel results).
    """
    if not tail:
        return list(compressed_head)
    if not compressed_head:
        return list(tail)

    head = list(compressed_head)
    rest = list(tail)

    last = head[-1]
    first = rest[0]
    last_role = last.get("role")
    first_role = first.get("role")

    if last_role == first_role and last_role in ("user", "assistant"):
        # Illegal adjacency. Merge the tail's first message text into the
        # head's last message so alternation is preserved. Only string
        # contents are merged inline; structured/multimodal contents fall
        # back to dropping the redundant standalone (the content is
        # preserved by concatenation when both are strings).
        last_content = last.get("content")
        first_content = first.get("content")
        if isinstance(last_content, str) and isinstance(first_content, str):
            merged = dict(last)
            merged["content"] = f"{last_content}\n\n{first_content}"
            head[-1] = merged
            rest = rest[1:]
        else:
            # Can't safely string-merge multimodal content. Insert a
            # minimal bridging turn so the seam alternates rather than
            # losing data.
            bridge_role = "assistant" if first_role == "user" else "user"
            head.append({"role": bridge_role, "content": ""})

    return head + rest
