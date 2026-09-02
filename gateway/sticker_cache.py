"""Sticker description cache for Telegram.

Stickers are described via the vision tool once and cached by file_unique_id
(``~/.hermes/sticker_cache.json``) so the same image is never re-analyzed.
"""

import json
import os
import tempfile
import time
from typing import Optional

from hermes_cli.config import get_hermes_home


CACHE_PATH = get_hermes_home() / "sticker_cache.json"

# Kept concise to save tokens.
STICKER_VISION_PROMPT = (
    "Describe this sticker in 1-2 sentences. Focus on what it depicts -- "
    "character, action, emotion. Be concise and objective."
)


def _load_cache() -> dict:
    if CACHE_PATH.exists():
        try:
            return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_cache(cache: dict) -> None:
    """Write the cache atomically (temp file + fsync + replace)."""
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(CACHE_PATH.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, str(CACHE_PATH))
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def get_cached_description(file_unique_id: str) -> Optional[dict]:
    """Return ``{description, emoji, set_name, cached_at}`` or None."""
    return _load_cache().get(file_unique_id)


def cache_sticker_description(file_unique_id: str, description: str, emoji: str = "", set_name: str = "") -> None:
    """Store a vision-generated description under Telegram's stable sticker id."""
    cache = _load_cache()
    cache[file_unique_id] = {"description": description, "emoji": emoji, "set_name": set_name, "cached_at": time.time()}
    _save_cache(cache)


def build_sticker_injection(description: str, emoji: str = "", set_name: str = "") -> str:
    """Warm-style injection text, e.g.
    ``[The user sent a sticker 😀 from "MyPack"~ It shows: "A cat waving" (=^.w.^=)]``.
    ``set_name`` is only shown together with an emoji.
    """
    context = ""
    if set_name and emoji:
        context = f" {emoji} from \"{set_name}\""
    elif emoji:
        context = f" {emoji}"
    return f"[The user sent a sticker{context}~ It shows: \"{description}\" (=^.w.^=)]"


def build_animated_sticker_injection(emoji: str = "") -> str:
    """Injection text for animated/video stickers we can't analyze."""
    if emoji:
        return (
            f"[The user sent an animated sticker {emoji}~ "
            f"I can't see animated ones yet, but the emoji suggests: {emoji}]"
        )
    return "[The user sent an animated sticker~ I can't see animated ones yet]"
