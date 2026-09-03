"""OpenAI-style -> Anthropic Messages API request conversion.

Everything here rewrites *request payloads*: model-id normalization, tool schemas, and the
message list (content blocks, thinking blocks and their signatures, tool_use/tool_result
pairing, cache_control placement, screenshot eviction, blank-block scrubbing). Endpoint
predicates come from ``agent/anthropic_endpoints.py``, so this module never imports the adapter
and there is no cycle. ``agent.anthropic_adapter`` re-exports every name below.
"""

import copy
import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from agent.anthropic_endpoints import (
    _is_deepseek_anthropic_endpoint, _is_kimi_family_endpoint, _is_nous_portal_endpoint,
    _is_third_party_anthropic_endpoint,
)

logger = logging.getLogger(__name__)

_THINKING_TYPES = frozenset(("thinking", "redacted_thinking"))
_CACHEABLE_TYPES = frozenset(("text", "tool_use"))
_EMPTY_TEXT_PLACEHOLDER = "(empty)"
_EMPTY_SCHEMA = {"type": "object", "properties": {}}
_BEDROCK_REGION_PREFIXES = (
    "global.", "us.", "eu.", "apac.", "ap.", "au.", "jp.", "ca.", "sa.", "me.", "af.",
)


# ----- small shared predicates -----


def _block_type(b: Any) -> Any:
    """``type`` of a dict block, None for non-dicts."""
    return b.get("type") if isinstance(b, dict) else None


def _has_block_type(blocks: List[Any], types) -> bool:
    return any(_block_type(b) in types for b in blocks)


def _is_blank_text_block(b: Any) -> bool:
    """A text block whose ``text`` is not a non-whitespace string (None/int/blank all count) —
    Anthropic 400s on them ("text content blocks must contain non-whitespace text")."""
    if _block_type(b) != "text":
        return False
    text = b.get("text")
    return not (isinstance(text, str) and text.strip())


def _cache_control_of(b: Any) -> Optional[Dict[str, Any]]:
    cc = b.get("cache_control") if isinstance(b, dict) else None
    return cc if isinstance(cc, dict) else None


def _text_block(text: str) -> Dict[str, str]:
    return {"type": "text", "text": text}


def _parse_tool_args(raw: Any) -> Any:
    """JSON-decode a tool_call ``arguments`` string; non-strings pass through, bad JSON -> {}."""
    try:
        return json.loads(raw) if isinstance(raw, str) else raw
    except (json.JSONDecodeError, ValueError):
        return {}


def _strip_thinking(blocks: List[Any]) -> List[Any]:
    return [b for b in blocks if _block_type(b) not in _THINKING_TYPES]


# ----- model / tool conversion -----


def _is_bedrock_model_id(model: str) -> bool:
    """Bedrock ids (``anthropic.claude-opus-4-7``, ``us.anthropic.claude-*``) use dots as namespace
    separators that must be preserved verbatim."""
    return model.lower().startswith(_BEDROCK_REGION_PREFIXES + ("anthropic.",))


def normalize_model_name(model: str, preserve_dots: bool = False) -> str:
    """Strip the ``anthropic/`` prefix (case-insensitive) and, unless ``preserve_dots`` (DashScope:
    ``qwen3.5-plus``), convert version dots to hyphens for Claude models only (``claude-opus-4.6``
    -> ``claude-opus-4-6``). Bedrock ids keep their namespace dots; non-Anthropic names
    (``gpt-5.4``) keep dots as part of their canonical form."""
    if model.lower().startswith("anthropic/"):
        model = model[len("anthropic/"):]
    if not preserve_dots and not _is_bedrock_model_id(model) and model.lower().startswith(("claude-", "anthropic/")):
        model = model.replace(".", "-")
    return model


def _sanitize_tool_id(tool_id: str) -> str:
    """Anthropic requires ids matching [a-zA-Z0-9_-]; replace the rest, never empty."""
    if not tool_id:
        return "tool_0"
    return re.sub(r"[^a-zA-Z0-9_-]", "_", tool_id) or "tool_0"


def _normalize_tool_input_schema(schema: Any) -> Dict[str, Any]:
    """Normalize a tool schema for Anthropic's validator: collapse nullable unions (``anyOf:
    [{type: string}, {type: null}]`` from Pydantic/MCP optional fields) to the non-null branch —
    optionality is already expressed by ``required``; ``keep_nullable_hint=False`` because the
    OpenAPI ``nullable`` keyword is not recognized. Top-level oneOf/allOf/anyOf are rejected with a
    generic 400, so they are dropped in favour of a plain object."""
    if not schema:
        return dict(_EMPTY_SCHEMA)

    from tools.schema_sanitizer import strip_nullable_unions

    normalized = strip_nullable_unions(schema, keep_nullable_hint=False)
    if not isinstance(normalized, dict):
        return dict(_EMPTY_SCHEMA)
    banned = {"oneOf", "allOf", "anyOf"}
    if banned & normalized.keys():
        normalized = {k: v for k, v in normalized.items() if k not in banned}
        normalized.setdefault("type", "object")
    if normalized.get("type") == "object" and not isinstance(normalized.get("properties"), dict):
        normalized = {**normalized, "properties": {}}
    return normalized


def convert_tools_to_anthropic(tools: List[Dict]) -> List[Dict]:
    """Convert OpenAI tool definitions to Anthropic format. Duplicate names are dropped with a
    warning (Anthropic hard-400s on them); ``cache_control`` on the OpenAI tool dict is forwarded."""
    if not tools:
        return []
    result = []
    seen_names: set = set()
    for t in tools:
        fn = t.get("function", {})
        name = fn.get("name", "")
        if name and name in seen_names:
            logger.warning("convert_tools_to_anthropic: duplicate tool name '%s' — dropping second occurrence", name)
            continue
        if name:
            seen_names.add(name)
        anthropic_tool: Dict[str, Any] = {
            "name": name,
            "description": fn.get("description", ""),
            "input_schema": _normalize_tool_input_schema(fn.get("parameters") or {}),
        }
        cache_control = _cache_control_of(t)
        if cache_control is not None:
            anthropic_tool["cache_control"] = dict(cache_control)
        result.append(anthropic_tool)
    return result


# ----- content-part conversion -----


def _image_source_from_openai_url(url: str) -> Dict[str, str]:
    """OpenAI image URL / data URL -> Anthropic image ``source``."""
    url = str(url or "").strip()
    if url.startswith("data:"):
        header, _, data = url.partition(",")
        mime_part = header[len("data:"):].split(";", 1)[0].strip()
        media_type = mime_part if mime_part.startswith("image/") else "image/jpeg"
        return {"type": "base64", "media_type": media_type, "data": data}
    return {"type": "url", "url": url}


def _convert_content_part_to_anthropic(part: Any) -> Optional[Dict[str, Any]]:
    """Convert one OpenAI-style content part to an Anthropic block (None -> dropped)."""
    if part is None:
        return None
    if isinstance(part, str):
        return _text_block(part)
    if not isinstance(part, dict):
        return _text_block(str(part))

    ptype = part.get("type")
    if ptype in ("input_text", "text"):
        # Rebuild from whitelisted fields only: stored SDK text blocks carry output-only siblings
        # (parsed_output, citations=None) that the INPUT schema rejects with 400.
        block: Dict[str, Any] = _text_block(part.get("text", ""))
        cits = part.get("citations")
        if ptype == "text" and isinstance(cits, list) and cits:
            block["citations"] = cits
    elif ptype in {"image_url", "input_image"}:
        image_value = part.get("image_url", {})
        url = image_value.get("url", "") if isinstance(image_value, dict) else str(image_value or "")
        block = {"type": "image", "source": _image_source_from_openai_url(url)}
    else:
        block = dict(part)

    cache_control = _cache_control_of(part)
    if cache_control is not None:
        block.setdefault("cache_control", dict(cache_control))
    return block


def _to_plain_data(value: Any, *, _depth: int = 0, _path: Optional[set] = None) -> Any:
    """Recursively convert SDK objects to plain Python data. ``_path`` tracks ids on the *current*
    recursion path (shared but non-cyclic objects convert normally while true cycles stringify);
    depth is capped at 20."""
    if _depth > 20:
        return str(value)
    if _path is None:
        _path = set()
    obj_id = id(value)
    if obj_id in _path:
        return str(value)

    def rec(v):
        return _to_plain_data(v, _depth=_depth + 1, _path=_path)

    _path.add(obj_id)
    if hasattr(value, "model_dump"):
        try:
            # warnings=False: streaming-accumulator blocks trip pydantic's serializer-mismatch
            # UserWarning, which otherwise leaks to the terminal.
            dumped = value.model_dump(warnings=False)
        except TypeError:  # duck-typed model_dump without pydantic's signature
            dumped = value.model_dump()
        result = rec(dumped)
    elif isinstance(value, dict):
        result = {k: rec(v) for k, v in value.items()}
    elif isinstance(value, (list, tuple)):
        result = [rec(v) for v in value]
    elif hasattr(value, "__dict__"):
        result = {k: rec(v) for k, v in vars(value).items() if not k.startswith("_")}
    else:
        result = value
    _path.discard(obj_id)
    return result


def _extract_preserved_thinking_blocks(message: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Deep-copied thinking/redacted_thinking blocks from ``reasoning_details``."""
    raw_details = message.get("reasoning_details")
    if not isinstance(raw_details, list):
        return []
    return [
        copy.deepcopy(d)
        for d in raw_details
        if isinstance(d, dict) and str(d.get("type", "") or "").strip().lower() in _THINKING_TYPES
    ]


def _convert_content_to_anthropic(content: Any) -> Any:
    """Convert an OpenAI multimodal content list to Anthropic blocks (non-lists pass through)."""
    if not isinstance(content, list):
        return content
    return [b for b in map(_convert_content_part_to_anthropic, content) if b is not None]


def _content_parts_to_anthropic_blocks(parts: Any) -> List[Dict[str, Any]]:
    """Tool-message content parts -> tool_result inner blocks (text + image only, the types
    Anthropic accepts there). Used for multimodal tool results."""
    if not isinstance(parts, list):
        return []
    out: List[Dict[str, Any]] = []
    for block in map(_convert_content_part_to_anthropic, parts):
        if not block:
            continue
        btype, text_val, src = block.get("type"), block.get("text"), block.get("source")
        if btype == "text" and isinstance(text_val, str) and text_val:
            out.append(_text_block(text_val))
        elif btype == "image" and isinstance(src, dict) and src:
            out.append({"type": "image", "source": src})
    return out


def _safe_text(text: Any) -> str:
    """``text`` if non-whitespace, else the placeholder. A blank text block stored in history (e.g.
    by compression) is replayed on every turn and wedges the session with HTTP 400; the placeholder
    is self-healing. Mirrors ``bedrock_adapter._safe_text`` (kept separate on purpose)."""
    if text is None:
        return _EMPTY_TEXT_PLACEHOLDER
    if not isinstance(text, str):
        text = str(text)
    return text if text.strip() else _EMPTY_TEXT_PLACEHOLDER


# ----- replay-block sanitizing (per-type whitelist) -----


def _replay_text(b: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    # Drop blank blocks rather than coerce in place: the caller relocates any cache_control and
    # falls back to a placeholder only when nothing survives, so "(empty)" never sits as
    # model-visible noise next to real blocks.
    if _is_blank_text_block(b):
        return None
    out: Dict[str, Any] = _text_block(b["text"])
    cits = b.get("citations")  # input-valid ONLY as a non-empty list
    if isinstance(cits, list) and cits:
        out["citations"] = cits
    if _cache_control_of(b) is not None:
        out["cache_control"] = b["cache_control"]
    return out


def _replay_thinking(b: Dict[str, Any]) -> Dict[str, Any]:
    out = {"type": "thinking", "thinking": b.get("thinking", "")}
    if b.get("signature"):
        out["signature"] = b["signature"]
    return out


def _replay_redacted_thinking(b: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    return {"type": "redacted_thinking", "data": b["data"]} if b.get("data") else None


def _replay_tool_use(b: Dict[str, Any]) -> Dict[str, Any]:
    out = {
        "type": "tool_use", "id": _sanitize_tool_id(b.get("id", "")), "name": b.get("name", ""),
        "input": b.get("input", {}),
    }
    if _cache_control_of(b) is not None:
        out["cache_control"] = b["cache_control"]
    return out


def _replay_image(b: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    src = b.get("source")
    return {"type": "image", "source": src} if isinstance(src, dict) else None


_REPLAY_SANITIZERS = {
    "text": _replay_text, "thinking": _replay_thinking, "redacted_thinking": _replay_redacted_thinking,
    "tool_use": _replay_tool_use, "image": _replay_image,
}


def _sanitize_replay_block(b: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Whitelist a stored Anthropic block so it is valid as REQUEST input. SDK response blocks carry
    output-only fields the INPUT schema forbids ("Extra inputs are not permitted": ``parsed_output``,
    ``caller``, ``citations=None``), and ``_to_plain_data`` captured them verbatim. Whitelist per
    type (not blacklist) so future SDK fields can't reintroduce the bug; unknown types are dropped.
    Returns a clean block or None."""
    if not isinstance(b, dict):
        return None
    sanitizer = _REPLAY_SANITIZERS.get(b.get("type"))
    return sanitizer(b) if sanitizer else None


def _apply_assistant_cache_control_to_last_cacheable_block(blocks: List[Dict[str, Any]], cache_control: Any) -> None:
    if not isinstance(cache_control, dict):
        return
    for block in reversed(blocks):
        if _block_type(block) in _CACHEABLE_TYPES:
            block.setdefault("cache_control", dict(cache_control))
            break


# ----- per-message conversion -----


def _replay_ordered_blocks(m: Dict[str, Any], ordered_blocks: List[Any]) -> Optional[List[Dict[str, Any]]]:
    """Interleaved-thinking replay: rebuild the assistant turn from the verbatim block list
    normalize_response stored (only for turns interleaving SIGNED thinking with tool_use).
    Preserves block ORDER; returns None if nothing survives. tool_use ``input`` is re-sourced from
    ``tool_calls`` (redacted at storage time) rather than the captured block (raw API response, NOT
    redacted), so a secret the model inlined into a tool call never goes back on the wire."""
    redacted_input_by_id = {
        _sanitize_tool_id(tc.get("id", "")): _parse_tool_args((tc.get("function", {}) or {}).get("arguments", "{}"))
        for tc in m.get("tool_calls", []) or []
        if isinstance(tc, dict)
    }
    replayed: List[Dict[str, Any]] = []
    relocated_cc = None
    dropped_blank_text = False
    for b in ordered_blocks:
        clean = _sanitize_replay_block(b)
        if clean is None:
            if _block_type(b) == "text":
                dropped_blank_text = True
            if _cache_control_of(b) is not None:  # relocate a dropped block's breakpoint
                relocated_cc = b["cache_control"]
            continue
        if clean.get("type") == "tool_use":
            redacted = redacted_input_by_id.get(clean.get("id", ""))
            if redacted is not None:
                clean["input"] = redacted
        replayed.append(clean)
    # Nothing cacheable survived (e.g. signed thinking + blank text): emit the placeholder so the
    # turn stays schema-valid and a relocated marker has a carrier.
    if not _has_block_type(replayed, _CACHEABLE_TYPES) and (dropped_blank_text or relocated_cc is not None):
        replayed.append(_text_block(_EMPTY_TEXT_PLACEHOLDER))
    if not replayed:
        return None
    _apply_assistant_cache_control_to_last_cacheable_block(replayed, relocated_cc)
    _apply_assistant_cache_control_to_last_cacheable_block(replayed, m.get("cache_control"))
    # prompt_caching marks an assistant turn with text by writing cache_control INTO ``content``
    # (not top-level). This path never reads ``content``, so carry that marker over or the
    # breakpoint is burned rather than relocated.
    msg_content = m.get("content")
    if isinstance(msg_content, list):
        inline_cc = next((cc for cc in map(_cache_control_of, msg_content) if cc is not None), None)
        _apply_assistant_cache_control_to_last_cacheable_block(replayed, inline_cc)
    return replayed


def _convert_assistant_message(m: Dict[str, Any]) -> Dict[str, Any]:
    """Assistant message -> Anthropic content blocks (thinking, text, tool_use, Kimi/DeepSeek
    reasoning_content injection)."""
    content = m.get("content", "")
    ordered_blocks = m.get("anthropic_content_blocks")
    if isinstance(ordered_blocks, list) and ordered_blocks:
        replayed = _replay_ordered_blocks(m, ordered_blocks)
        if replayed:
            return {"role": "assistant", "content": replayed}

    blocks = _extract_preserved_thinking_blocks(m)
    # Blank text blocks are dropped; a cache marker riding on one is relocated onto the last
    # surviving cacheable block (prompt_caching sets cache_control on content[-1], which may be
    # exactly the blank block).
    relocated_cc = None
    if isinstance(content, list):
        for blk in _convert_content_to_anthropic(content):
            if _is_blank_text_block(blk):
                if _cache_control_of(blk) is not None:
                    relocated_cc = blk["cache_control"]
                continue
            blocks.append(blk)
    elif content and str(content).strip():
        blocks.append(_text_block(str(content)))
    for tc in m.get("tool_calls", []):
        if not tc or not isinstance(tc, dict):
            continue
        fn = tc.get("function", {})
        blocks.append({
            "type": "tool_use", "id": _sanitize_tool_id(tc.get("id", "")), "name": fn.get("name", ""),
            "input": _parse_tool_args(fn.get("arguments", "{}")),
        })
    # Kimi's /coding endpoint requires reasoning_content on replayed tool-call turns — even ""
    # (injected as a fallback upstream). Prepend, since thinking must precede text/tool_use. Skip
    # when reasoning_details already supplied (signed) thinking blocks: a duplicate unsigned one
    # would be downgraded to a spurious text block on the last assistant message.
    reasoning_content = m.get("reasoning_content")
    if isinstance(reasoning_content, str) and not _has_block_type(blocks, _THINKING_TYPES):
        blocks.insert(0, {"type": "thinking", "thinking": reasoning_content})
    # Empty assistant content is rejected. Fall back ONLY to the placeholder, never to raw
    # ``content`` — that is the unfiltered blank payload just removed. Markers are applied after
    # the fallback so one from a sole dropped blank block lands on the placeholder.
    effective = blocks or [_text_block(_EMPTY_TEXT_PLACEHOLDER)]
    _apply_assistant_cache_control_to_last_cacheable_block(effective, relocated_cc)
    _apply_assistant_cache_control_to_last_cacheable_block(effective, m.get("cache_control"))
    return {"role": "assistant", "content": effective}


def _tool_result_content(m: Dict[str, Any]) -> Any:
    """Resolve a tool message's content into tool_result content (blocks or string)."""
    content = m.get("content", "")
    multimodal_blocks: Optional[List[Dict[str, Any]]] = None
    if isinstance(content, dict) and content.get("_multimodal"):
        multimodal_blocks = _content_parts_to_anthropic_blocks(content.get("content") or [])
        if not multimodal_blocks and content.get("text_summary"):
            multimodal_blocks = [_text_block(str(content["text_summary"]))]
    elif isinstance(content, list):
        converted = _content_parts_to_anthropic_blocks(content)
        if any(b.get("type") == "image" for b in converted):
            multimodal_blocks = converted
    if multimodal_blocks is None:  # back-compat: blocks stashed under a private key
        stashed = m.get("_anthropic_content_blocks")
        if isinstance(stashed, list) and stashed:
            text_content = content if isinstance(content, str) and content.strip() else None
            multimodal_blocks = [_text_block(text_content)] + stashed if text_content else list(stashed)

    if multimodal_blocks:
        return multimodal_blocks
    if isinstance(content, str):
        return content or "(no output)"
    return json.dumps(content) if content else "(no output)"


def _convert_tool_message_to_result(result: List[Dict[str, Any]], m: Dict[str, Any]) -> None:
    """Append a tool_result to ``result``, merging into a trailing tool_result user message when
    there is one. Mutates ``result`` in place."""
    tool_result = {
        "type": "tool_result", "tool_use_id": _sanitize_tool_id(m.get("tool_call_id", "")),
        "content": _tool_result_content(m),
    }
    cache_control = _cache_control_of(m)
    if cache_control is not None:
        tool_result["cache_control"] = dict(cache_control)
    last = result[-1] if result else {}
    if last.get("role") == "user" and isinstance(last.get("content"), list) and last["content"] \
            and last["content"][0].get("type") == "tool_result":
        last["content"].append(tool_result)
    else:
        result.append({"role": "user", "content": [tool_result]})


def _convert_user_message(content: Any) -> Dict[str, Any]:
    """Validate and convert a user message to Anthropic format."""
    if isinstance(content, list):
        kept_blocks = _fix_blank_text_blocks_in_list(
            _convert_content_to_anthropic(content), placeholder_text="(empty message)",
            msg_index=-1, role="user", location="_convert_user_message",
        )
        return {"role": "user", "content": kept_blocks}
    if not content or (isinstance(content, str) and not content.strip()):
        content = "(empty message)"
    return {"role": "user", "content": content}


# ----- whole-list passes -----


def _strip_orphaned_tool_blocks(result: List[Dict[str, Any]]) -> None:
    """Strip tool_use blocks with no matching tool_result, and vice versa. Compression/truncation
    can remove either side of a pair or insert messages between them. Anthropic requires the
    tool_result in the IMMEDIATELY FOLLOWING user message — a global id match is not enough.
    Mutates ``result`` in place."""
    # Pass 1: tool_use without an adjacent result.
    for i, m in enumerate(result):
        if m.get("role") != "assistant" or not isinstance(m.get("content"), list):
            continue
        tool_use_ids_in_turn = {b.get("id") for b in m["content"] if _block_type(b) == "tool_use"}
        if not tool_use_ids_in_turn:
            continue
        adjacent_result_ids: set = set()
        if i + 1 < len(result):
            nxt = result[i + 1]
            if nxt.get("role") == "user" and isinstance(nxt.get("content"), list):
                adjacent_result_ids = {b.get("tool_use_id") for b in nxt["content"] if _block_type(b) == "tool_result"}
        orphaned = tool_use_ids_in_turn - adjacent_result_ids
        if not orphaned:
            continue
        kept = [b for b in m["content"] if not (_block_type(b) == "tool_use" and b.get("id") in orphaned)]
        # A signed thinking block on this turn was signed against the ORIGINAL content and is now
        # dead (400 "thinking blocks in the latest assistant message cannot be modified"). Flag so
        # _manage_thinking_signatures demotes it.
        if len(kept) != len(m["content"]) and _has_block_type(m["content"], _THINKING_TYPES):
            m["_thinking_signature_invalidated"] = True
        m["content"] = kept if kept else [_text_block("(tool call removed)")]

    # Pass 2: tool_result whose tool_use no longer exists anywhere.
    surviving_tool_use_ids = {
        b.get("id")
        for m in result
        if m.get("role") == "assistant" and isinstance(m.get("content"), list)
        for b in m["content"]
        if _block_type(b) == "tool_use"
    }
    for m in result:
        if m.get("role") != "user" or not isinstance(m.get("content"), list):
            continue
        new_content = [
            b for b in m["content"] if _block_type(b) != "tool_result" or b.get("tool_use_id") in surviving_tool_use_ids
        ]
        if len(new_content) != len(m["content"]):
            m["content"] = new_content if new_content else [_text_block("(tool result removed)")]


def _concat_content(prev: Any, curr: Any) -> Any:
    """Merge two message contents: str+str joined by newline, list+list concatenated, mixed shapes
    promoted to block lists."""
    if isinstance(prev, str) and isinstance(curr, str):
        return prev + "\n" + curr
    if isinstance(prev, str):
        prev = [_text_block(prev)]
    if isinstance(curr, str):
        curr = [_text_block(curr)]
    return prev + curr


def _merge_consecutive_roles(result: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge consecutive same-role messages to enforce alternation. Returns a new list."""
    fixed = []
    for m in result:
        if not (fixed and fixed[-1]["role"] == m["role"]):
            fixed.append(m)
            continue
        if m["role"] != "user":
            # Keep the orphan-strip flag visible to _manage_thinking_signatures.
            if m.get("_thinking_signature_invalidated"):
                fixed[-1]["_thinking_signature_invalidated"] = True
            # The second message's thinking blocks were signed against a different turn boundary
            # and become invalid once merged.
            if isinstance(m["content"], list):
                m["content"] = _strip_thinking(m["content"])
        fixed[-1]["content"] = _concat_content(fixed[-1]["content"], m["content"])
    return fixed


def _keep_valid_latest_thinking(content: List[Any], signature_dead: bool) -> List[Any]:
    """Latest assistant turn on direct Anthropic: keep signed thinking, demote unsigned to text so
    the reasoning isn't lost. If orphan-stripping mutated THIS turn every signature is dead (and a
    bare signed block with no tool_use is also invalid), so demote ALL of them."""
    new_content = []
    for b in content:
        if _block_type(b) not in _THINKING_TYPES:
            new_content.append(b)
            continue
        is_redacted = b.get("type") == "redacted_thinking"
        signed = b.get("data") if is_redacted else b.get("signature")  # redacted 'data' IS the signature
        if signed and not signature_dead:
            new_content.append(b)
        elif (signature_dead or not is_redacted) and b.get("thinking"):
            new_content.append(_text_block(b["thinking"]))  # demote to plain text
        # else: redacted_thinking without data — unverifiable, dropped
    return new_content


def _manage_thinking_signatures(result: List[Dict[str, Any]], base_url: str | None, model: str | None) -> None:
    """Strip or preserve thinking blocks per endpoint. Mutates ``result`` in place.

    Anthropic signs thinking blocks against the full turn; any upstream mutation invalidates them
    (400 "Invalid signature in thinking block"), so on direct Anthropic only the LATEST assistant
    turn keeps signed blocks. Signatures are proprietary: third-party endpoints strip all thinking.
    Kimi replays as-is; DeepSeek needs unsigned blocks round-tripped but rejects signed ones. Nous
    Portal proxies Claude with sticky sessions and validates the same signatures, so it takes the
    native path despite not being anthropic.com.
    """
    is_third_party = _is_third_party_anthropic_endpoint(base_url) and not _is_nous_portal_endpoint(base_url)
    is_kimi = _is_kimi_family_endpoint(base_url, model)
    is_deepseek = _is_deepseek_anthropic_endpoint(base_url)
    last_assistant_idx = next((i for i in range(len(result) - 1, -1, -1) if result[i].get("role") == "assistant"), None)

    for idx, m in enumerate(result):
        if m.get("role") != "assistant" or not isinstance(m.get("content"), list):
            continue
        if is_kimi:
            pass  # shared cleanup below still strips cache markers + the flag
        elif is_deepseek:
            # Strip signed (or redacted-with-data), keep unsigned.
            new_content = [
                b for b in m["content"]
                if _block_type(b) not in _THINKING_TYPES or not (b.get("signature") or b.get("data"))
            ]
            m["content"] = new_content or [_text_block("(empty)")]
        elif is_third_party or idx != last_assistant_idx:
            m["content"] = _strip_thinking(m["content"]) or [_text_block("(thinking elided)")]
        else:
            new_content = _keep_valid_latest_thinking(m["content"], bool(m.get("_thinking_signature_invalidated")))
            m["content"] = new_content or [_text_block("(empty)")]

        # cache_control on thinking blocks interferes with signature validation.
        for b in m["content"]:
            if _block_type(b) in _THINKING_TYPES:
                b.pop("cache_control", None)
        m.pop("_thinking_signature_invalidated", None)  # internal flag, never on the wire


def _evict_old_screenshots(result: List[Dict[str, Any]]) -> None:
    """Keep only the 3 most recent computer-use screenshots (~1,465 tokens each); older images
    become a placeholder text block. Mutates ``result`` in place."""
    image_count = 0
    for msg in reversed(result):
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if _block_type(block) != "tool_result":
                continue
            inner = block.get("content")
            if not isinstance(inner, list) or not _has_block_type(inner, {"image"}):
                continue
            image_count += 1
            if image_count > 3:
                block["content"] = [
                    b if b.get("type") != "image" else _text_block("[screenshot removed to save context]") for b in inner
                ]


def _ensure_leading_user_turn(result: List[Dict[str, Any]]) -> None:
    """Anthropic requires messages[0].role == user; prepend a placeholder turn otherwise. A second
    auto-compaction can leave a role=assistant summary first, which the API rejects (often masked
    as a misleading tool_use/tool_result 400). The filler must be non-whitespace text or it trades
    that 400 for the blank-block one."""
    if result and result[0].get("role") != "user":
        result.insert(0, {"role": "user", "content": [_text_block(_EMPTY_TEXT_PLACEHOLDER)]})


def _fix_blank_text_blocks_in_list(
    blocks: List[Any], *, placeholder_text: str, msg_index: int, role: Any, location: str
) -> List[Any]:
    """Drop blank text blocks; relocate any cache_control they carried onto the last surviving
    cacheable block; if nothing survives, substitute one placeholder block (carrying the relocated
    marker). Non-text blocks and order are untouched. Returns a new list; logs structure only."""
    kept: List[Any] = []
    relocated_cache_control = None
    for block_index, blk in enumerate(blocks):
        if _is_blank_text_block(blk):
            if _cache_control_of(blk) is not None:
                relocated_cache_control = blk["cache_control"]
            logger.warning(
                "Pre-call sanitizer: dropped blank text content block "
                "(message_index=%d role=%s location=%s block_index=%d block_type=text)",
                msg_index, role, location, block_index,
            )
            continue
        kept.append(blk)
    if not kept:
        kept.append(_text_block(placeholder_text))
    _apply_assistant_cache_control_to_last_cacheable_block(kept, relocated_cache_control)
    return kept


def _scrub_blank_text_blocks(result: List[Dict[str, Any]]) -> None:
    """Final boundary guard against blank text blocks (HTTP 400 "text content blocks must contain
    non-whitespace text"), including inside tool_result content. Runs LAST so a blank block from
    any current or future producer never reaches the wire. Mutates ``result`` in place."""
    for msg_index, msg in enumerate(result):
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        content = msg.get("content")
        if not isinstance(content, list) or not content:
            continue
        new_content = _fix_blank_text_blocks_in_list(
            content, placeholder_text=_EMPTY_TEXT_PLACEHOLDER if role == "assistant" else "(empty message)",
            msg_index=msg_index, role=role, location="content",
        )
        for blk in new_content:
            if _block_type(blk) != "tool_result":
                continue
            inner = blk.get("content")
            if isinstance(inner, list) and inner:
                blk["content"] = _fix_blank_text_blocks_in_list(
                    inner, placeholder_text="(no output)", msg_index=msg_index, role=role, location="tool_result",
                )
        msg["content"] = new_content


def _convert_system_content(content: Any) -> Any:
    """System message content -> Anthropic ``system`` param (str, or block list when cache_control
    is present). With cache markers the blocks are copied (never mutating the caller's dicts) and
    blank text is replaced by the placeholder: Anthropic rejects blank system blocks too, and a
    blank block carrying a breakpoint can't simply be dropped."""
    if not isinstance(content, list):
        return content
    if not any(p.get("cache_control") for p in content if isinstance(p, dict)):
        return "\n".join(p["text"] for p in content if p.get("type") == "text")
    system = []
    for p in content:
        if not isinstance(p, dict):
            continue
        if p.get("type") == "text" and isinstance(p.get("text"), str) and not p["text"].strip():
            p = {**p, "text": _EMPTY_TEXT_PLACEHOLDER}
        system.append(p)
    return system


def convert_messages_to_anthropic(
    messages: List[Dict], base_url: str | None = None, model: str | None = None
) -> Tuple[Optional[Any], List[Dict]]:
    """Convert OpenAI-format messages to Anthropic format -> ``(system, messages)``. System is
    extracted into its own param (a string, or a block list when cache_control is present).
    ``base_url``/``model`` drive thinking-signature policy — third-party endpoints strip signatures
    (proprietary, they 400 on them); Kimi-family endpoints/models keep unsigned
    reasoning_content-derived blocks, which Kimi requires even when empty."""
    system = None
    result: List[Dict[str, Any]] = []

    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "system":
            system = _convert_system_content(content)
        elif role == "assistant":
            result.append(_convert_assistant_message(m))
        elif role == "tool":
            _convert_tool_message_to_result(result, m)
        else:
            result.append(_convert_user_message(content))

    _strip_orphaned_tool_blocks(result)
    result = _merge_consecutive_roles(result)
    _ensure_leading_user_turn(result)
    _manage_thinking_signatures(result, base_url, model)
    _evict_old_screenshots(result)
    _scrub_blank_text_blocks(result)

    return system, result
