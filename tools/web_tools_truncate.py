"""Truncate-and-store pipeline for web_extract (no LLM).

Pages at or under the char budget are returned whole; larger pages become a
head+tail window plus a footer that says how much is shown, where the full text
is stored (cache/web) and the exact read_file call that pages the omitted middle.
Inline base64 images are replaced with ``[IMAGE: alt]`` placeholders.

Extracted from tools/web_tools.py; the names are re-imported there so
``tools.web_tools.MAX_STORED_TEXT_CHARS`` etc. keep working. Logs under the
origin logger name for parity.
"""

import logging
import re
from typing import Any, List, Optional

logger = logging.getLogger("tools.web_tools")

# Per-page char budget sent to the model (override: web.extract_char_limit).
# Larger pages are head+tail truncated and the full text stored on disk.
DEFAULT_EXTRACT_CHAR_LIMIT = 15000

# Ceiling on the full-text file written to cache/web so a multi-MB page can't
# write unbounded bytes on every extract; the model only ever sees char_limit.
MAX_STORED_TEXT_CHARS = 2_000_000

_CHAR_LIMIT_FLOOR, _CHAR_LIMIT_CEILING = 2000, 500_000


def _clamp_char_limit(value: Any) -> int:
    """Clamp to [2k, 500k]; raises TypeError/ValueError for non-numeric input.

    Floor: below 2k the truncation footer dominates. Ceiling: a config typo
    must not blow up context.
    """
    return max(_CHAR_LIMIT_FLOOR, min(int(value), _CHAR_LIMIT_CEILING))


def _get_extract_char_limit() -> int:
    """``web.extract_char_limit`` clamped to a sane range, else the default."""
    from tools.web_tools import _load_web_config  # lazy: tests patch tools.web_tools._load_web_config

    try:
        configured = _load_web_config().get("extract_char_limit")
        if configured is not None:
            return _clamp_char_limit(configured)
    except (TypeError, ValueError):
        pass
    return DEFAULT_EXTRACT_CHAR_LIMIT


def convert_base64_images_to_links(text: str) -> str:
    """Replace inline base64 image blobs (token bombs) with ``[IMAGE: alt]`` placeholders.

    Handles markdown images (alt text kept), parenthesised blobs, and bare
    ``data:image/...;base64,`` payloads. Real http(s) markdown image links are
    left untouched so the agent can ``web_extract`` / ``vision_analyze`` them.
    """
    def _md_repl(m: "re.Match[str]") -> str:
        alt = (m.group("alt") or "").strip()
        return f"[IMAGE: {alt}]" if alt else "[IMAGE]"

    md_b64 = re.compile(
        r"!\[(?P<alt>[^\]]*)\]\(\s*data:image/[^;]+;base64,[A-Za-z0-9+/=\s]+\)"
    )
    out = md_b64.sub(_md_repl, text)
    out = re.sub(r"\(\s*data:image/[^;]+;base64,[A-Za-z0-9+/=\s]+\)", "[IMAGE]", out)
    out = re.sub(r"data:image/[^;]+;base64,[A-Za-z0-9+/=]+", "[IMAGE]", out)
    return out


def _store_full_text(url: str, content: str) -> Optional[str]:
    """Write the full page to cache/web; absolute path or None.

    cache/web is mounted read-only into remote backends (credential_files
    _CACHE_DIRS) so read_file can page the complete text on any backend.
    Best-effort: on failure the truncated content is still returned to the model.
    """
    try:
        import hashlib
        from hermes_constants import get_hermes_dir
        from tools.web_result_cache import _host_slug

        cache_dir = get_hermes_dir("cache/web", "web_cache")
        cache_dir.mkdir(parents=True, exist_ok=True)

        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:10]
        path = cache_dir / f"{_host_slug(url)}-{digest}.md"
        if len(content) > MAX_STORED_TEXT_CHARS:
            content = (
                content[:MAX_STORED_TEXT_CHARS]
                + f"\n\n[... stored copy truncated at {MAX_STORED_TEXT_CHARS:,} chars "
                f"of {len(content):,}; re-extract a more specific URL for the rest ...]"
            )
        from tools.spill_safety import write_text_exclusive

        # Deterministic name in a well-known dir: refuse symlinks (lstat-unlink +
        # exclusive create); same-URL re-extraction legitimately overwrites. Not
        # private: cache/web is bind-mounted into remote backends' container UID.
        write_text_exclusive(path, content, private=False, overwrite=True)
        return str(path)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Failed to store full web_extract text for %s: %s", url, exc)
        return None


def _truncate_with_footer(
    content: str,
    url: str,
    char_limit: int,
) -> tuple[str, bool]:
    """Return (model_text, was_truncated).

    Pages over ``char_limit`` become a ~75% head / ~25% tail window cut on line
    boundaries, plus a footer saying how much is shown, where the full text is
    stored, and the read_file call that pages the omitted middle. Deterministic.
    """
    if len(content) <= char_limit:
        return content, False

    head_budget = int(char_limit * 0.75)
    tail_budget = char_limit - head_budget

    head = content[:head_budget]
    tail = content[-tail_budget:]
    # Snap both cuts to line boundaries (head back, tail forward) so we never slice mid-line.
    nl = head.rfind("\n")
    if nl > head_budget * 0.5:
        head = head[:nl]
    nl = tail.find("\n")
    if 0 <= nl < tail_budget * 0.5:
        tail = tail[nl + 1:]

    total = len(content)
    stored_path = _store_full_text(url, content)

    footer_lines = [
        "",
        "─" * 8 + " [TRUNCATED] " + "─" * 8,
        f"Showing {len(head):,} chars (head) + {len(tail):,} chars (tail) "
        f"of {total:,} total clean characters.",
    ]
    if stored_path:
        # read_file is 1-indexed; +2 lands on the first line after the shown head.
        middle_start_line = head.count("\n") + 2
        footer_lines.append(f"Full text saved to: {stored_path}")
        footer_lines.append(
            f'To read the omitted middle: read_file path="{stored_path}" '
            f"offset={middle_start_line} limit=200  (the file is the complete page; "
            f"raise/lower offset to page through it)."
        )
    else:
        footer_lines.append(
            "Full text could not be stored; re-run web_extract on a more "
            "specific URL or use browser_navigate for the complete page."
        )
    footer_lines.append("─" * 29)

    model_text = head + "\n\n[... middle omitted — see footer ...]\n\n" + tail
    model_text += "\n" + "\n".join(footer_lines)
    return model_text, True


def _effective_char_limit(char_limit: Optional[int]) -> int:
    """Caller's ``char_limit`` (else config) clamped; non-numeric input falls back to the default."""
    value = char_limit if char_limit is not None else _get_extract_char_limit()
    try:
        return _clamp_char_limit(value)
    except (TypeError, ValueError):
        return DEFAULT_EXTRACT_CHAR_LIMIT


def _truncate_results(results: List[dict], char_limit: int, debug_call_data: dict) -> None:
    """In place: replace each successful entry's content with its base64-cleaned, budgeted text.

    Records per-page truncation metrics into ``debug_call_data``.
    """
    for result in results:
        if result.get("error"):
            continue
        url = result.get("url", "")
        raw_content = result.get("raw_content", "") or result.get("content", "")
        if not raw_content:
            continue
        clean = convert_base64_images_to_links(raw_content)
        model_text, truncated = _truncate_with_footer(clean, url, char_limit)
        result["content"] = model_text
        if truncated:
            debug_call_data["pages_truncated"] += 1
            debug_call_data["truncation_metrics"].append({
                "url": url,
                "original_size": len(clean),
                "sent_size": len(model_text),
            })
            logger.info("%s (truncated %d -> %d chars)", url, len(clean), len(model_text))
        else:
            logger.info("%s (%d chars, whole)", url, len(clean))


def _trim_results(results: List[dict]) -> List[dict]:
    """Keep only url/title/content/error per entry (+ blocked_by_policy when present)."""
    return [
        {
            "url": r.get("url", ""),
            "title": r.get("title", ""),
            "content": r.get("content", ""),
            "error": r.get("error"),
            **({"blocked_by_policy": r["blocked_by_policy"]} if "blocked_by_policy" in r else {}),
        }
        for r in results
    ]
