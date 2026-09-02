"""Content capture, message serialization and usage/cost translation for the Langfuse plugin.

Everything here is pure data shaping; nothing touches the SDK client or trace state.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, Optional

# Same logger as the package so log records keep the plugin's name.
logger = logging.getLogger(__name__.rpartition(".")[0])

_READ_FILE_LINE_RE = re.compile(r"^\s*(\d+)\|(.*)$")
_READ_FILE_HEAD_LINES = 25
_READ_FILE_TAIL_LINES = 15

# (langfuse usage key, CanonicalUsage attribute / summary-dict key, PricingEntry attribute)
_USAGE_FIELDS = (
    ("input", "input_tokens", "input_cost_per_million"),
    ("output", "output_tokens", "output_cost_per_million"),
    ("cache_read_input_tokens", "cache_read_tokens", "cache_read_cost_per_million"),
    ("cache_creation_input_tokens", "cache_write_tokens", "cache_write_cost_per_million"),
    ("reasoning_tokens", "reasoning_tokens", None),
)

_CAPTURE_MODES = ("metadata", "sanitized", "full")
_DEFAULT_CAPTURE_MODE = "sanitized"
_warned_invalid_capture = False


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _debug(message: str) -> None:
    if _env("HERMES_LANGFUSE_DEBUG").lower() in {"1", "true", "yes", "on"}:
        logger.info("Langfuse tracing: %s", message)


# ---------------------------------------------------------------------------
# Capture modes
# ---------------------------------------------------------------------------

def _capture_mode() -> str:
    """Resolve ``metadata | sanitized | full``.

    Read per call so tests and long-lived processes can flip modes without a
    client reset. Invalid values warn once and fall back to the default rather
    than silently capturing more than the operator intended.
    """
    global _warned_invalid_capture
    value = _env("HERMES_LANGFUSE_CAPTURE").lower()
    if not value:
        return _DEFAULT_CAPTURE_MODE
    if value in _CAPTURE_MODES:
        return value
    if not _warned_invalid_capture:
        _warned_invalid_capture = True
        logger.warning(
            "Langfuse plugin: invalid HERMES_LANGFUSE_CAPTURE=%r, falling back "
            "to %r (valid: %s)",
            value, _DEFAULT_CAPTURE_MODE, ", ".join(_CAPTURE_MODES),
        )
    return _DEFAULT_CAPTURE_MODE


def _redact_secrets(value: str) -> str:
    # force=True: redact even if the user disabled security.redact_secrets —
    # this content is exported to an external service.
    try:
        from agent.redact import redact_sensitive_text
        return redact_sensitive_text(value, force=True)
    except Exception:
        return value


def _describe_content(value: Any) -> Any:
    """Metadata-mode stand-in for content: shape and size, never payload."""
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return {"omitted": True, "type": "number"}
    if isinstance(value, bytes):
        return {"omitted": True, "type": "bytes", "length": len(value)}
    if isinstance(value, str):
        return {"omitted": True, "type": "text", "chars": len(value)}
    if isinstance(value, dict):
        return {"omitted": True, "type": "object", "keys": [str(k) for k in list(value.keys())[:20]]}
    if isinstance(value, (list, tuple, set)):
        return {"omitted": True, "type": "array", "items": len(value)}
    return {"omitted": True, "type": type(value).__name__}


def _capture_content(value: Any, *, parse_json_strings: bool = False) -> Any:
    """Apply the active capture mode to a CONTENT value.

    Only prompt/response text, tool arguments and tool results are content;
    metadata fields (provider, model, IDs, counts) stay as-is in every mode.
    """
    if _capture_mode() == "metadata":
        return _describe_content(value)
    return _safe_value(value, parse_json_strings=parse_json_strings)


def _capture_tool_result(result: Any, *, tool_name: str, args: Any) -> Any:
    """Capture a tool result: JSON strings are parsed first so a read_file
    payload can be collapsed to a preview keyed by the call's ``args``."""
    if _capture_mode() == "metadata":
        return _describe_content(result)
    value = _maybe_parse_json_string(result) if isinstance(result, str) else result
    return _safe_value(_normalize_payload(value, tool_name=tool_name, args=args), parse_json_strings=True)


def _redact_data_uri(value: str) -> dict[str, Any]:
    header = value.split(",", 1)[0] if "," in value else "data:"
    media_type = header[5:].split(";", 1)[0] if header.startswith("data:") else ""
    return {"type": "data_uri", "media_type": media_type or None, "omitted": True, "length": len(value)}


def _truncate_text(value: str, max_chars: int) -> Any:
    # The SDK decodes data:*;base64 strings as media; a truncated one is
    # invalid base64 and logs noisily, so redact the whole URI instead.
    prefix = value[:200].lower()
    if prefix.startswith("data:") and ";base64," in prefix:
        return _redact_data_uri(value)
    # Redact BEFORE truncating so a secret straddling the cut cannot leak.
    if _capture_mode() == "sanitized":
        value = _redact_secrets(value)
    if len(value) <= max_chars:
        return value
    return value[:max_chars] + f"... [truncated {len(value) - max_chars} chars]"


def _maybe_parse_json_string(value: str) -> Any:
    stripped = value.strip()
    if len(stripped) < 2 or stripped[0] not in "{[":
        return value
    try:
        parsed, idx = json.JSONDecoder().raw_decode(stripped)
    except Exception:
        return value
    if not isinstance(parsed, (dict, list)):
        return value

    trailing = stripped[idx:].strip()
    if not trailing:
        return parsed

    hint_key = "_hint" if trailing.startswith("[Hint:") else "_trailing_text"
    if isinstance(parsed, dict):
        merged = dict(parsed)
        merged[hint_key if hint_key not in merged else "_trailing_text"] = trailing
        return merged
    return {"data": parsed, hint_key: trailing}


def _parse_read_file_lines(content: str) -> list[dict[str, Any]]:
    if not isinstance(content, str) or not content:
        return []
    lines = []
    for raw_line in content.splitlines():
        match = _READ_FILE_LINE_RE.match(raw_line)
        if not match:
            return []
        lines.append({"line": int(match.group(1)), "text": match.group(2)})
    return lines


def _normalize_read_file_payload(value: dict[str, Any], *, args: Any = None) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    if isinstance(args, dict):
        path = args.get("path")
        if isinstance(path, str) and path:
            normalized["path"] = path
        for key in ("offset", "limit"):
            if isinstance(args.get(key), int):
                normalized[key] = args[key]

    lines = _parse_read_file_lines(value.get("content", ""))
    if lines:
        normalized["returned_lines"] = {"start": lines[0]["line"], "end": lines[-1]["line"], "count": len(lines)}
        head, tail = _READ_FILE_HEAD_LINES, _READ_FILE_TAIL_LINES
        if len(lines) <= head + tail:
            normalized["content_preview"] = {"lines": lines}
        else:
            normalized["content_preview"] = {
                "head": lines[:head],
                "tail": lines[-tail:],
                "omitted_line_count": len(lines) - head - tail,
            }
    elif value.get("content"):
        normalized["content_preview"] = {"text": value.get("content", "")}

    for key in ("total_lines", "file_size", "truncated", "is_binary", "is_image", "hint",
                "_warning", "mime_type", "dimensions", "similar_files", "error"):
        if key in value:
            normalized[key] = value[key]

    base64_content = value.get("base64_content")
    if isinstance(base64_content, str) and base64_content:
        normalized["base64_content"] = {"omitted": True, "length": len(base64_content)}
    return normalized


def _normalize_payload(value: Any, *, tool_name: str = "", args: Any = None) -> Any:
    """Collapse a read_file result (line-numbered content + file metadata) into a compact preview."""
    is_read_file = (
        isinstance(value, dict)
        and isinstance(value.get("content"), str)
        and all(k in value for k in ("total_lines", "file_size", "is_binary", "is_image"))
        and not value.get("error")
    )
    if is_read_file:
        return _normalize_read_file_payload(value, args=args if tool_name == "read_file" else None)
    return value


def _safe_value(value: Any, *, max_chars: Optional[int] = None, depth: int = 0,
                parse_json_strings: bool = False) -> Any:
    max_chars = max_chars if max_chars is not None else int(_env("HERMES_LANGFUSE_MAX_CHARS", "12000") or "12000")
    if depth > 4:
        return "<max-depth>"
    if value is None or isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, bytes):
        return {"type": "bytes", "len": len(value)}
    recurse = lambda v, d: _safe_value(v, max_chars=max_chars, depth=d, parse_json_strings=parse_json_strings)  # noqa: E731
    if isinstance(value, str):
        if parse_json_strings:
            parsed = _maybe_parse_json_string(value)
            if parsed is not value:
                return recurse(parsed, depth)
        return _truncate_text(value, max_chars)
    if isinstance(value, dict):
        normalized = _normalize_payload(value)
        if normalized is not value:
            return recurse(normalized, depth)
        return {str(k): recurse(v, depth + 1) for k, v in list(value.items())[:50]}
    if isinstance(value, (list, tuple, set)):
        return [recurse(v, depth + 1) for v in list(value)[:50]]
    if hasattr(value, "__dict__"):
        return recurse(vars(value), depth + 1)
    return _truncate_text(repr(value), max_chars)


# ---------------------------------------------------------------------------
# Message serialization
# ---------------------------------------------------------------------------

def _extract_last_user_message(messages: Any) -> Any:
    if not isinstance(messages, list):
        return None
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "user":
            return {"role": "user", "content": _capture_content(message.get("content"))}
    return None


def _coerce_request_messages(*, request_messages: Any = None, messages: Any = None,
                             conversation_history: Any = None, user_message: Any = None) -> list[dict[str, Any]]:
    for candidate in (request_messages, messages, conversation_history):
        if isinstance(candidate, list):
            return candidate
    if user_message is None:
        return []
    return [{"role": "user", "content": user_message}]


def _serialize_system_prompt(system_prompt: Any) -> Optional[dict[str, Any]]:
    """Normalize Anthropic/Bedrock ``system`` param or OpenAI-style system content."""
    if isinstance(system_prompt, str):
        text = system_prompt.strip()
    elif isinstance(system_prompt, list):
        parts: list[str] = []
        for block in system_prompt:
            if isinstance(block, dict):
                # Anthropic: {"type": "text", "text": ...}; Bedrock Converse: {"text": ...}.
                block_type = block.get("type")
                if block_type == "text" or (block_type is None and "text" in block):
                    piece = block.get("text", "")
                    if isinstance(piece, str) and piece:
                        parts.append(piece)
            elif isinstance(block, str) and block:
                parts.append(block)
        text = "\n\n".join(parts)
    else:
        return None
    if not text:
        return None
    return {"role": "system", "content": _capture_content(text)}


def _messages_for_langfuse_input(*, request_messages: Any = None, messages: Any = None,
                                 conversation_history: Any = None, user_message: Any = None,
                                 system_prompt: Any = None, pre_coerced: Any = None) -> list[dict[str, Any]]:
    """Generation input, prepending ``system_prompt`` when the provider split it out of messages.

    ``pre_coerced`` lets the caller pass an already-coerced list and skip a
    second ``_coerce_request_messages`` per hook.
    """
    raw = pre_coerced if pre_coerced is not None else _coerce_request_messages(
        request_messages=request_messages, messages=messages,
        conversation_history=conversation_history, user_message=user_message,
    )
    system_msg = None if raw and raw[0].get("role") == "system" else _serialize_system_prompt(system_prompt)
    serialized = _serialize_messages(raw)
    return serialized if system_msg is None else [system_msg, *serialized]


def _serialize_messages(messages: Any) -> list[dict[str, Any]]:
    if not isinstance(messages, list):
        return []
    serialized = []
    for message in messages[-12:]:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        item = {"role": role, "content": _capture_content(message.get("content"), parse_json_strings=(role == "tool"))}
        if role == "tool":
            if message.get("tool_call_id"):
                item["tool_call_id"] = message.get("tool_call_id")
            if message.get("name"):
                item["name"] = _safe_value(message.get("name"))
        if message.get("tool_calls"):
            item["tool_calls"] = _capture_content(message.get("tool_calls"), parse_json_strings=True)
        serialized.append(item)
    return serialized


def _serialize_tool_calls(tool_calls: Any) -> list[dict[str, Any]]:
    serialized = []
    for tool_call in tool_calls or ():
        fn = getattr(tool_call, "function", None)
        name = getattr(fn, "name", None) if fn else None
        safe_arguments = _capture_content(getattr(fn, "arguments", None) if fn else None)
        serialized.append({
            "id": getattr(tool_call, "id", None),
            "type": getattr(tool_call, "type", None) or "function",
            "name": name,
            "arguments": safe_arguments,
            "function": {"name": name, "arguments": safe_arguments},
        })
    return serialized


def _serialize_assistant_message(message: Any) -> dict[str, Any]:
    reasoning = None
    for attr in ("reasoning", "reasoning_content", "reasoning_details"):
        value = getattr(message, attr, None)
        if value is not None:
            reasoning = _capture_content(value)
            break
    return {
        "content": _capture_content(getattr(message, "content", None)),
        "reasoning": reasoning,
        "tool_calls": _serialize_tool_calls(getattr(message, "tool_calls", None)),
    }


# ---------------------------------------------------------------------------
# Usage + cost
# ---------------------------------------------------------------------------

def _canonical_usage_and_cost(canonical: Any, *, provider: str, model: str,
                              base_url: str) -> tuple[dict[str, int], dict[str, float]]:
    """Translate canonical Hermes usage into Langfuse usage and cost maps."""
    usage_details: Dict[str, int] = {}
    for key, attr, _ in _USAGE_FIELDS:
        tokens = getattr(canonical, attr)
        if tokens or key in ("input", "output"):
            usage_details[key] = tokens

    cost_details: Dict[str, float] = {}
    try:
        from agent.usage_pricing import estimate_usage_cost, resolve_billing_route

        # Subscription-included routes: Langfuse treats explicit cost_details
        # (even zeros) as authoritative, so omit them and let it estimate.
        route = resolve_billing_route(model, provider=provider, base_url=base_url)
        if getattr(route, "billing_mode", "") == "subscription_included":
            return usage_details, cost_details
        cost = estimate_usage_cost(model, canonical, provider=provider, base_url=base_url, api_key="")
    except Exception as exc:  # pragma: no cover - fail-open
        _debug(f"usage pricing failed: {exc}")
        return usage_details, cost_details

    # No total (e.g. cache pricing unknown) => export no costs at all, so a
    # partial component subtotal is never mistaken for the request total.
    if cost.amount_usd is None:
        return usage_details, cost_details

    # Langfuse only derives totals from input/output keys, so cache/custom keys
    # need an explicit total (Hermes estimate also includes request pricing).
    # A zero total is not exported: Langfuse would treat it as authoritative.
    if cost.status != "included" and float(cost.amount_usd) > 0:
        cost_details["total"] = float(cost.amount_usd)

    # Per-type breakdown for dashboards; keys mirror usage_details.
    try:
        from decimal import Decimal

        from agent.usage_pricing import get_pricing_entry

        entry = get_pricing_entry(model, provider=provider, base_url=base_url)
        if entry:
            for key, attr, rate_attr in _USAGE_FIELDS:
                rate = getattr(entry, rate_attr, None) if rate_attr else None
                tokens = getattr(canonical, attr)
                if rate is not None and tokens:
                    cost_details[key] = float(Decimal(tokens) * rate / Decimal("1000000"))
    except Exception:  # pragma: no cover - canonical total remains usable
        pass

    return usage_details, cost_details


def _usage_and_cost(response: Any, *, provider: str, api_mode: str, model: str, base_url: str) -> tuple[dict[str, int], dict[str, float]]:
    raw_usage = getattr(response, "usage", None)
    if not raw_usage:
        return {}, {}
    try:
        from agent.usage_pricing import normalize_usage

        canonical = normalize_usage(raw_usage, provider=provider, api_mode=api_mode)
        return _canonical_usage_and_cost(canonical, provider=provider, model=model, base_url=base_url)
    except Exception as exc:  # pragma: no cover - fail-open
        _debug(f"usage normalization failed: {exc}")
        return {}, {}


def _summary_usage_and_cost(usage: dict, *, provider: str, model: str, base_url: str) -> tuple[dict[str, int], dict[str, float]]:
    """post_api_request path: usage arrives as a pre-built CanonicalUsage summary dict."""
    try:
        from agent.usage_pricing import CanonicalUsage

        canonical = CanonicalUsage(
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0) or usage.get("completion_tokens", 0),
            cache_read_tokens=usage.get("cache_read_tokens", 0),
            cache_write_tokens=usage.get("cache_write_tokens", 0),
            reasoning_tokens=usage.get("reasoning_tokens", 0),
            request_count=usage.get("request_count", 1),
        )
        return _canonical_usage_and_cost(canonical, provider=provider, model=model, base_url=base_url)
    except Exception:
        return {}, {}


def _moa_usage_and_cost(ref: dict) -> tuple[dict[str, int], dict[str, float]]:
    """MoA advisor reference: usage dict keyed like CanonicalUsage plus a pre-computed ``cost_usd``."""
    usage = ref.get("usage") or {}
    usage_details = {}
    if isinstance(usage, dict):
        for key, attr, _ in _USAGE_FIELDS:
            if usage.get(attr):
                usage_details[key] = usage[attr]
    cost_usd = ref.get("cost_usd")
    return usage_details, ({"total": float(cost_usd)} if isinstance(cost_usd, (int, float)) else {})
