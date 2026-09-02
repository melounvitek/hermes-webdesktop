"""Hindsight plugin constants and pure config normalizers (no I/O, no origin imports)."""

from __future__ import annotations

import json
import logging
from typing import Any, List

# Log under the plugin package's own logger name (loader-path independent).
logger = logging.getLogger(__name__.rpartition(".")[0])

_DEFAULT_API_URL = "https://api.hindsight.vectorize.io"
_DEFAULT_LOCAL_URL = "http://localhost:8888"
# Keep in sync with tools/lazy_deps.py ("memory.hindsight") and plugin.yaml.
_MIN_CLIENT_VERSION = "0.6.1"
_DEFAULT_TIMEOUT = 120  # seconds — cloud API can take 30-40s per request
_DEFAULT_IDLE_TIMEOUT = 300  # seconds — Hindsight embedded daemon default
# ``metadata.source`` stamped on retained memories — OPT-IN, empty by default:
# AGENTS.md forbids on-by-default third-party attribution tags. Set via the
# ``retain_source`` config key or HINDSIGHT_RETAIN_SOURCE.
_DEFAULT_RETAIN_SOURCE = ""
# Hindsight brand mark (eye ringed by graph nodes) for the recall/retain indicators.
_HINDSIGHT_GLYPH = "👁️"
# Hindsight 0.5.0 added ``update_mode='append'`` on retain. Without it, reusing a
# stable session-scoped document_id silently overwrites prior turns server-side,
# so older APIs keep the per-process unique document_id fallback.
_MIN_VERSION_FOR_UPDATE_MODE_APPEND = "0.5.0"
_VALID_BUDGETS = {"low", "mid", "high"}
_PROVIDER_DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-haiku-4-5",
    "gemini": "gemini-3.6-flash",
    "groq": "openai/gpt-oss-120b",
    "openrouter": "qwen/qwen3.5-9b",
    "minimax": "MiniMax-M2.7",
    "ollama": "gemma3:12b",
    "lmstudio": "local-model",
    "openai_compatible": "your-model-name",
}
# The embedded daemon speaks OpenAI wire format for these providers.
_OPENAI_WIRE_PROVIDERS = {"openai_compatible", "openrouter"}
_OBSERVATION_SCOPE_KEYWORDS = {"per_tag", "combined", "all_combinations"}


def _parse_int_setting(value: Any, default: int) -> int:
    """Parse an integer config/env value, falling back on invalid input."""
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        logger.warning("Invalid integer Hindsight setting %r; using default %s", value, default)
        return default


def _daemon_llm_provider(provider: str) -> str:
    return "openai" if provider in _OPENAI_WIRE_PROVIDERS else provider


def _normalize_retain_tags(value: Any) -> List[str]:
    """Normalize tag config/tool values to a deduplicated list of strings."""
    if value is None:
        return []
    if isinstance(value, list):
        raw_items = value
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        parsed = None
        if text.startswith("["):
            try:
                parsed = json.loads(text)
            except Exception:
                parsed = None
        raw_items = parsed if isinstance(parsed, list) else text.split(",")
    else:
        raw_items = [value]
    normalized: list[str] = []
    for item in raw_items:
        tag = str(item).strip()
        if tag and tag not in normalized:
            normalized.append(tag)
    return normalized


def _normalize_observation_scopes(value: Any) -> Any:
    """Normalize an observation_scopes value to a Hindsight-accepted form.

    Returns ``None`` (nothing configured; Hindsight applies its ``combined``
    default), a keyword string, or ``list[list[str]]`` (one inner list per
    consolidation pass). Accepts a keyword, a JSON-encoded list, a flat list of
    tags (one scope), or a list of tag-lists. Anything unrecognized yields
    ``None`` so we never send an invalid payload.
    """
    if isinstance(value, str):
        text = value.strip()
        if text in _OBSERVATION_SCOPE_KEYWORDS:
            return text
        if text.startswith("["):
            try:
                return _normalize_observation_scopes(json.loads(text))
            except Exception:
                return None
        return None
    if isinstance(value, (list, tuple)):
        if all(isinstance(entry, str) for entry in value):
            inner = [entry.strip() for entry in value if entry.strip()]
            return [inner] if inner else None
        scopes: list[list[str]] = []
        for entry in value:
            if isinstance(entry, (list, tuple)):
                inner = [str(tag).strip() for tag in entry if str(tag).strip()]
                if inner:
                    scopes.append(inner)
            elif isinstance(entry, str) and entry.strip():
                scopes.append([entry.strip()])
        return scopes or None
    return None


def _sanitize_bank_segment(value: str) -> str:
    """Make a bank_id placeholder URL/filesystem safe: non ``[A-Za-z0-9_-]``
    runs become a single dash; leading/trailing dashes and underscores are stripped."""
    if not value:
        return ""
    out = []
    prev_dash = False
    for ch in str(value):
        if ch.isalnum() or ch in "-_":
            out.append(ch)
            prev_dash = False
        elif not prev_dash:
            out.append("-")
            prev_dash = True
    return "".join(out).strip("-_")


def _resolve_bank_id_template(template: str, fallback: str, **placeholders: str) -> str:
    """Render a bank_id template ({profile}, {workspace}, {platform}, {user},
    {session}); each placeholder is sanitized first. Empty placeholders render
    as "" and the dash/underscore runs they leave are collapsed, e.g.
    ``hermes-{user}`` with no user becomes ``hermes``. Empty template or an
    invalid placeholder falls back to *fallback*."""
    if not template:
        return fallback
    sanitized = {k: _sanitize_bank_segment(v) for k, v in placeholders.items()}
    try:
        rendered = template.format(**sanitized)
    except (KeyError, IndexError) as exc:
        logger.warning("Invalid bank_id_template %r: %s — using fallback %r",
                       template, exc, fallback)
        return fallback
    while "--" in rendered:
        rendered = rendered.replace("--", "-")
    while "__" in rendered:
        rendered = rendered.replace("__", "_")
    return rendered.strip("-_") or fallback
