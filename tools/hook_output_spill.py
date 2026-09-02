"""Spill oversized hook-injected context to disk with a preview placeholder.

Shell hooks and plugin ``pre_llm_call`` hooks can return ``{"context": ...}``
that is concatenated into the user message on EVERY subsequent API call, so a
large blob inflates every turn and breaks the prompt-cache prefix. Above a
configured budget the full text is written to a per-session directory and the
in-prompt payload becomes a head/tail preview plus the saved path.

Config (``config.yaml``)::

    hooks:
      output_spill:
        enabled: true          # default: true; set false to disable spilling
        max_chars: 10000       # default; context above this is spilled
        preview_head: 500      # chars shown at the start of the preview
        preview_tail: 500      # chars shown at the end of the preview
        directory: null        # default: <HERMES_HOME>/hook_outputs

Invariants: unchanged input when disabled or under the cap; never raises —
an I/O failure still returns a bounded preview with an in-prompt notice. Spill
files are grouped per session so ``/new`` sessions don't pile into one directory.
Ported from openai/codex PR #21069.
"""

from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from tools.tool_output_limits import _coerce_int, _coerce_positive_int

logger = logging.getLogger(__name__)


DEFAULT_MAX_CHARS = 10_000
DEFAULT_PREVIEW_HEAD = 500
DEFAULT_PREVIEW_TAIL = 500
DEFAULT_ENABLED = True


def get_spill_config() -> Dict[str, Any]:
    """Return resolved hook output-spill config. Never raises."""
    section: Dict[str, Any] = {}
    try:
        from hermes_cli.config import load_config
        cfg = load_config() or {}
        hooks = cfg.get("hooks") if isinstance(cfg, dict) else None
        if isinstance(hooks, dict):
            sub = hooks.get("output_spill")
            if isinstance(sub, dict):
                section = sub
    except Exception:
        section = {}

    enabled_raw = section.get("enabled", DEFAULT_ENABLED)
    enabled = bool(enabled_raw) if enabled_raw is not None else DEFAULT_ENABLED

    directory = section.get("directory")
    if directory is not None and not isinstance(directory, str):
        directory = None

    return {
        "enabled": enabled,
        "max_chars": _coerce_positive_int(section.get("max_chars"), DEFAULT_MAX_CHARS),
        # head/tail allow zero (empty tail), max_chars must be positive.
        "preview_head": _coerce_int(section.get("preview_head"), DEFAULT_PREVIEW_HEAD, 0),
        "preview_tail": _coerce_int(section.get("preview_tail"), DEFAULT_PREVIEW_TAIL, 0),
        "directory": directory,
    }


def _resolve_spill_dir(directory_override: Optional[str], session_id: Optional[str]) -> Path:
    """Per-session spill directory; session id is sanitised so it can't escape ``base``."""
    if directory_override:
        base = Path(os.path.expanduser(directory_override))
    else:
        from hermes_constants import get_hermes_home

        base = Path(get_hermes_home()) / "hook_outputs"

    session_segment = session_id or "no-session"
    session_segment = session_segment.replace("/", "_").replace("\\", "_").replace("..", "_")
    return base / session_segment


def _build_preview(
    text: str,
    head: int,
    tail: int,
    saved_path: Optional[str],
    *,
    source: str,
) -> str:
    """Assemble the in-prompt preview with head/tail and saved-path footer."""
    total = len(text)
    head_chunk = text[:head] if head > 0 else ""
    tail_chunk = text[-tail:] if tail > 0 and total > head else ""

    parts = [
        f"[{source} output truncated — {total:,} chars; full content "
        + (f"saved to {saved_path}]" if saved_path else "unavailable — spill write failed]"),
    ]
    if head_chunk:
        parts.append("--- head ---")
        parts.append(head_chunk)
    if tail_chunk:
        parts.append("--- tail ---")
        parts.append(tail_chunk)
    return "\n".join(parts)


def spill_if_oversized(
    text: str,
    *,
    session_id: Optional[str] = None,
    source: str = "hook",
    config: Optional[Dict[str, Any]] = None,
) -> str:
    """Spill ``text`` to disk if it exceeds the configured cap.

    Returns ``text`` unchanged (under cap, disabled, or empty) or a preview
    string pointing at the full content. Non-string input is ``str()``-coerced;
    ``source`` labels the preview header; ``config`` overrides config.yaml.
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        try:
            text = str(text)
        except Exception:
            return ""

    cfg = config if config is not None else get_spill_config()
    if not cfg.get("enabled", True):
        return text

    max_chars = int(cfg.get("max_chars") or DEFAULT_MAX_CHARS)
    if len(text) <= max_chars:
        return text

    head = int(cfg.get("preview_head") or 0)
    tail = int(cfg.get("preview_tail") or 0)
    directory_override = cfg.get("directory")

    # A disk failure must never blow up the turn — fall through to a preview
    # without a saved path.
    saved_path: Optional[str] = None
    try:
        spill_dir = _resolve_spill_dir(directory_override, session_id)
        from tools.spill_safety import ensure_spill_dir, write_text_exclusive

        # Hook context may embed raw secrets: private perms + exclusive,
        # symlink-refusing create (the per-session dir is predictable).
        ensure_spill_dir(spill_dir, private=True)
        spill_path = spill_dir / f"{uuid.uuid4().hex}.txt"
        # Trailing newline so tail readers don't report "missing newline".
        write_text_exclusive(
            spill_path,
            text if text.endswith("\n") else text + "\n",
            private=True,
        )
        saved_path = str(spill_path)
    except Exception as exc:
        logger.warning("hook output spill failed: %s", exc)
        saved_path = None

    return _build_preview(text, head, tail, saved_path, source=source)


__all__ = [
    "DEFAULT_MAX_CHARS",
    "DEFAULT_PREVIEW_HEAD",
    "DEFAULT_PREVIEW_TAIL",
    "DEFAULT_ENABLED",
    "get_spill_config",
    "spill_if_oversized",
]
