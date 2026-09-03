"""Message and tool-payload sanitization helpers (pure; documented in-place mutation).

Walk OpenAI-format message lists and structured payloads, repairing or stripping
characters that would crash ``json.dumps`` in the OpenAI SDK or be rejected upstream.
``run_agent`` re-exports them for old imports.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from functools import partial
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Lone surrogates are invalid UTF-8 and crash json.dumps in the OpenAI SDK; also used for
# CLI paste scrubbing.
_SURROGATE_RE = re.compile(r'[\ud800-\udfff]')

# Keys handled explicitly by _sanitize_messages; every OTHER key is swept generically.
_MESSAGE_CORE_KEYS = frozenset({"content", "name", "tool_calls", "role"})


def _sanitize_surrogates(text: str) -> str:
    """Replace lone surrogate code points with U+FFFD; no-op when none present."""
    if _SURROGATE_RE.search(text):
        return _SURROGATE_RE.sub('\ufffd', text)
    return text


def _strip_non_ascii(text: str) -> str:
    """Drop non-ASCII characters — last resort for ASCII-only system encodings (LANG=C)."""
    return text.encode('ascii', errors='ignore').decode('ascii')


def _fix_str_field(container: Any, key: Any, fix: Callable[[str], str]) -> bool:
    """Apply ``fix`` to ``container[key]`` if it is a str; True if it changed."""
    value = container.get(key) if isinstance(container, dict) else container[key]
    if isinstance(value, str):
        fixed = fix(value)
        if fixed != value:
            container[key] = fixed
            return True
    return False


def _sanitize_structure(payload: Any, fix: Callable[[str], str]) -> bool:
    """Apply ``fix`` to every str inside nested dict/list ``payload`` in-place."""
    found = False
    stack = [payload]
    while stack:
        node = stack.pop()
        items = node.items() if isinstance(node, dict) else enumerate(node) if isinstance(node, list) else ()
        for key, value in list(items):
            if isinstance(value, str):
                found |= _fix_str_field(node, key, fix)
            elif isinstance(value, (dict, list)):
                stack.append(value)
    return found


def _sanitize_messages(messages: list, fix: Callable[[str], str], *, deep: bool) -> bool:
    """Apply ``fix`` to the string fields of every message dict in-place.

    Covers content / content-part text, name, tool_call arguments, and every non-core
    top-level str field. ``deep=True`` additionally covers tool_call ids, function names,
    and NESTED non-core fields (``reasoning_details`` arrays from byte-level reasoning models).
    """
    found = False
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    found |= _fix_str_field(part, "text", fix)
        else:
            found |= _fix_str_field(msg, "content", fix)
        found |= _fix_str_field(msg, "name", fix)
        tool_calls = msg.get("tool_calls")
        for tc in tool_calls if isinstance(tool_calls, list) else ():
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function")
            fn_fields = [(fn, "name"), (fn, "arguments")] if isinstance(fn, dict) else []
            for container, key in [(tc, "id")] + fn_fields:
                if deep or key == "arguments":
                    found |= _fix_str_field(container, key, fix)
        for key, value in list(msg.items()):
            if key in _MESSAGE_CORE_KEYS:
                continue
            if isinstance(value, str):
                found |= _fix_str_field(msg, key, fix)
            elif deep and isinstance(value, (dict, list)):
                found |= _sanitize_structure(value, fix)
    return found


# In-place sanitizers; each returns True when anything changed. Surrogate repair is deep
# (tool_call ids, nested reasoning_details); the ASCII-only-locale strip is shallow.
_sanitize_structure_surrogates = partial(_sanitize_structure, fix=_sanitize_surrogates)
_sanitize_messages_surrogates = partial(_sanitize_messages, fix=_sanitize_surrogates, deep=True)
_sanitize_structure_non_ascii = partial(_sanitize_structure, fix=_strip_non_ascii)
_sanitize_messages_non_ascii = partial(_sanitize_messages, fix=_strip_non_ascii, deep=False)
_sanitize_tools_non_ascii = _sanitize_structure_non_ascii


def _escape_invalid_chars_in_json_strings(raw: str) -> str:
    """Escape literal control chars (0x00-0x1F) inside JSON string values as ``\\uXXXX``
    (for llama.cpp-style output mixing control chars with other malformations)."""
    out: list[str] = []
    in_string = False
    i = 0
    while i < len(raw):
        ch = raw[i]
        if in_string and ch == "\\" and i + 1 < len(raw):
            out.append(raw[i:i + 2])
            i += 2
            continue
        if ch == '"':
            in_string = not in_string
        out.append(f"\\u{ord(ch):04x}" if in_string and ord(ch) < 0x20 else ch)
        i += 1
    return "".join(out)


# When a repair rewrites arguments to "{}", the WARNING log is the last surviving copy of
# content that can hold real user data (a truncated write_file), so bound it generously.
_FULL_ARGS_LOG_BOUND = 100_000


def _loads_ok(text: str) -> bool:
    try:
        json.loads(text)
        return True
    except json.JSONDecodeError:
        return False


def _repair_tool_call_arguments(raw_args: str, tool_name: str = "?") -> str:
    """Repair malformed tool_call argument JSON (truncation, trailing commas, Python ``None``,
    control chars); ``"{}"`` if unrepairable so the request succeeds. Repairs log at WARNING."""
    raw_stripped = raw_args.strip() if isinstance(raw_args, str) else ""

    if not raw_stripped:
        logger.warning("Sanitized empty tool_call arguments for %s", tool_name)
        return "{}"

    if raw_stripped == "None":
        logger.warning("Sanitized Python-None tool_call arguments for %s", tool_name)
        return "{}"

    # Pass 0: strict=False accepts literal control chars inside strings (the most common
    # local-model case) and re-serialises to wire-valid JSON.
    try:
        reserialised = json.dumps(json.loads(raw_stripped, strict=False), separators=(",", ":"))
        if reserialised != raw_stripped:
            logger.warning("Repaired unescaped control chars in tool_call arguments for %s", tool_name)
        return reserialised
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    # Passes 1-3: strip trailing commas, close unclosed structures, trim excess closers (bounded).
    fixed = re.sub(r',\s*([}\]])', r'\1', raw_stripped)
    fixed += '}' * max(0, fixed.count('{') - fixed.count('}'))
    fixed += ']' * max(0, fixed.count('[') - fixed.count(']'))
    for _ in range(50):
        if _loads_ok(fixed):
            break
        if (fixed.endswith('}') and fixed.count('}') > fixed.count('{')) or (
            fixed.endswith(']') and fixed.count(']') > fixed.count('[')
        ):
            fixed = fixed[:-1]
        else:
            break

    if _loads_ok(fixed):
        logger.warning("Repaired malformed tool_call arguments for %s: %s → %s", tool_name, raw_stripped[:80], fixed[:80])
        return fixed

    # Pass 4: escape control chars inside strings (strict=False alone fails when other
    # malformations are present too), then retry.
    escaped = _escape_invalid_chars_in_json_strings(fixed)
    if escaped != fixed and _loads_ok(escaped):
        logger.warning(
            "Repaired control-char-laced tool_call arguments for %s: %s → %s", tool_name, raw_stripped[:80], escaped[:80],
        )
        return escaped

    logger.warning(
        "Unrepairable tool_call arguments for %s — replaced with empty object (was: %s)",
        tool_name, raw_stripped[:_FULL_ARGS_LOG_BOUND],
    )
    return "{}"


def close_interrupted_tool_sequence(messages: list, final_response: Any = None) -> bool:
    """Append a synthetic assistant turn when an interrupted tail is a tool result.

    A transcript ending on a raw ``tool`` message makes the next user message land as
    ``tool → user`` — an alternation violation strict providers (Gemini, Claude) answer by
    hallucinating a continuation. Mutates in place; True if a closing turn was appended.
    """
    last = messages[-1] if messages else None
    if not isinstance(last, dict) or last.get("role") != "tool":
        return False
    text = final_response if isinstance(final_response, str) else ""
    from agent.message_metadata import append_message

    append_message(messages, {"role": "assistant", "content": text.strip() or "Operation interrupted."})
    return True


def serialized_messages_bytes(messages: list) -> int:
    """Exact serialized byte size of the ``messages`` payload (HTTP 413 recovery).

    A 413 is a BYTE-size error, but the token estimator prices images at a flat cost, so
    it cannot score recovery from an image-dominated 413. Non-serializable values fall
    back to ``str()`` so a malformed message can never crash recovery.
    """
    if not isinstance(messages, list) or not messages:
        return 0
    try:
        return len(json.dumps(messages, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8"))
    except (TypeError, ValueError):
        return sum(len(str(m)) for m in messages)


_IMAGE_PART_TYPES = {"image_url", "image", "input_image"}


def _strip_images_from_messages(messages: list) -> bool:
    """Remove image content parts from all messages in-place (server rejected images).

    ``tool`` messages and assistant messages carrying ``tool_calls`` whose content was
    entirely images get a placeholder, NOT deleted (deleting orphans the paired
    ``tool_call_id`` → HTTP 400); other now-empty messages are dropped. Rewritten messages
    lose their ``api_content`` sidecar (it carries the images being removed).
    """
    from agent.turn_context import drop_stale_api_content

    found = False
    to_delete = []
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        new_parts = [p for p in content if not (isinstance(p, dict) and p.get("type") in _IMAGE_PART_TYPES)]
        if len(new_parts) < len(content):
            found = True
            if new_parts:
                msg["content"] = new_parts
            elif msg.get("role") == "tool" or msg.get("tool_calls"):
                msg["content"] = "[image content removed — server does not support images]"
            else:
                to_delete.append(i)
            drop_stale_api_content(msg)
    for i in reversed(to_delete):
        del messages[i]
    return found


# Provider error bodies (lowercased substring match) meaning "image/multimodal input
# unsupported" — the loop then strips images and retries text-only instead of cascading
# into compression / context-too-large recovery or wedging on retries.
_IMAGE_REJECTION_PHRASES = (
    "only 'text' content type is supported", "only text content type is supported",
    "image_url is not supported", "image content is not supported",
    "multimodal is not supported", "multimodal content is not supported", "multimodal input is not supported",
    "vision is not supported", "vision input is not supported",
    "does not support images", "does not support image input", "does not support multimodal",
    "does not support vision", "model does not support image",
    # DashScope-style gateways reject non-text blocks with this generic body.
    "unexpected item type in content",
    # ChatGPT-account Codex backend rejects data:image URLs in input_image; keyed on the
    # field-path apostrophe so other URL errors don't false-trip. Second: its wording for
    # corrupt/unsupported native image payloads.
    "image_url'. expected", "image data you provided does not represent a valid image",
    # DeepSeek's text-only request-body variant error.
    "unknown variant `image_url`, expected `text`", "unknown variant image_url, expected text",
    # OpenRouter HTTP 404 when no upstream endpoint accepts image input (passes the 4xx
    # gate; without this the gateway queue wedges behind the stuck turn).
    "no endpoints found that support image input",
    # Kimi/Moonshot et al. reject truncated/corrupt image bytes baked into history.
    "failed to decode image",
)


def _looks_like_image_content_rejection(error_body: str) -> bool:
    """Return True when a provider error says image/multimodal input is unsupported."""
    body = str(error_body or "").lower()
    return any(phrase in body for phrase in _IMAGE_REJECTION_PHRASES)


__all__ = [
    "_SURROGATE_RE", "close_interrupted_tool_sequence",
    "_sanitize_surrogates", "_sanitize_structure_surrogates", "_sanitize_messages_surrogates",
    "_escape_invalid_chars_in_json_strings", "_repair_tool_call_arguments",
    "_strip_non_ascii", "_sanitize_messages_non_ascii", "_sanitize_tools_non_ascii",
    "_strip_images_from_messages", "_sanitize_structure_non_ascii",
    # call_id policy owners
    "deterministic_call_id", "coalesce_tool_call_id", "tool_call_id_variants",
    "tool_result_id_variants", "uniquify_tool_call_ids",
    # reasoning_content policy owners
    "reasoning_echo_family", "matches_reasoning_echo_family", "needs_reasoning_echo",
    "stale_thinking_reaches_wire", "apply_reasoning_content_policy", "reapply_reasoning_echo",
]


# -- call_id policy: hash synthesis, ``call_id or id`` coalescing, duplicate-id repair ----
# NOT merged with codex_event_projector._deterministic_call_id (maps app-server ITEM ids,
# not chat tool-call content; merging would change ids and invalidate caches).
# HARD INVARIANT: deterministic (never uuid4) and byte-identical for existing inputs —
# these ids feed prompt-cache prefixes.


def _tc_field(tc: Any, key: str) -> Any:
    """Read ``key`` from a tool-call entry that may be a dict or an SDK object."""
    return tc.get(key) if isinstance(tc, dict) else getattr(tc, key, None)


def _tc_set(tc: Any, key: str, value: Any) -> None:
    if isinstance(tc, dict):
        tc[key] = value
    else:
        setattr(tc, key, value)


def deterministic_call_id(fn_name: str, arguments: str, index: int = 0) -> str:
    """Deterministic call_id fallback when the API omits one (random ids would break caching)."""
    seed = f"{fn_name}:{arguments}:{index}"
    return f"call_{hashlib.sha256(seed.encode('utf-8', errors='replace')).hexdigest()[:12]}"


def _expand_tool_id_variants(values: tuple[Any, ...]) -> frozenset[str]:
    """Every wire spelling of one tool-call identifier: Responses bridges may expose the pairing
    id and response-item id separately or as ``call_id|response_item_id``; all alias ONE call."""
    variants: set[str] = set()
    for raw in values:
        value = raw.strip() if isinstance(raw, str) else ""
        if not value:
            continue
        variants.add(value)
        if "|" in value:
            variants.update(p for p in (part.strip() for part in value.split("|")) if p)
    return frozenset(variants)


def tool_call_id_variants(tc: Any) -> frozenset[str]:
    """Return all pairing-id variants carried by a tool-call entry."""
    return _expand_tool_id_variants(tuple(_tc_field(tc, k) for k in ("call_id", "id", "response_item_id")))


def tool_result_id_variants(tool_call_id: Any) -> frozenset[str]:
    """Return all matching variants for a role=tool ``tool_call_id``."""
    return _expand_tool_id_variants((tool_call_id,))


def coalesce_tool_call_id(tc: Any) -> str:
    """Effective call id of a tool_call entry (dict or object); ``""`` when none.

    Codex Responses calls carry ``call_id`` (authoritative pairing key), Chat Completions
    carry ``id`` only, and bridge ids may be ``call_id|response_item_id``.
    """
    for raw in (_tc_field(tc, "call_id"), _tc_field(tc, "id")):
        value = raw.strip() if isinstance(raw, str) else ""
        if value:
            return value.split("|", 1)[0].strip() or value
    return ""


def uniquify_tool_call_ids(tool_calls: list) -> list:
    """Ensure every tool call in one assistant turn has a distinct id.

    Some providers reuse one id across a batch; the pre-API sanitizer then keeps only the
    first call/result pair per id and strict providers reject duplicates. First occurrence
    keeps its id; later collisions get a deterministic ``<id>_d<n>`` suffix (never uuid4 —
    cache-prefix stability). Mutates entries in place (SDK models / SimpleNamespace /
    dicts). Blank ids are left for the deterministic fallback in ``build_assistant_message``.
    """
    seen: set = set()
    for tc in tool_calls or []:
        # Same coalescing rule as coalesce_tool_call_id, tolerant of non-string ids.
        raw = _tc_field(tc, "call_id") or _tc_field(tc, "id") or ""
        raw = raw.strip() if isinstance(raw, str) else ""
        # Composite Responses ids ("call_x|fc_y") collide on the call half — the pairing key.
        cid = raw.split("|", 1)[0]
        if not cid:
            continue
        if cid not in seen:
            seen.add(cid)
            continue
        n = 2
        while f"{cid}_d{n}" in seen:
            n += 1
        new_id = f"{cid}_d{n}"
        seen.add(new_id)

        def _renamed(value):
            # Keep a composite id's response-item half so the provider's fc_/item id survives.
            return f"{new_id}|{value.split('|', 1)[1]}" if isinstance(value, str) and "|" in value else new_id

        try:
            _tc_set(tc, "id", _renamed(_tc_field(tc, "id")))
            if _tc_field(tc, "call_id"):
                _tc_set(tc, "call_id", new_id)
        except Exception:
            logger.warning("Could not uniquify duplicate tool call id %s", cid)
            continue
        _fn_name = _tc_field(_tc_field(tc, "function"), "name") or "?"
        logger.warning(
            "Model reused tool call id %s within one turn; renamed the "
            "duplicate to %s (tool=%s) to keep call/result pairing "
            "lossless.", cid, new_id, _fn_name,
        )
    return tool_calls


# -- reasoning_content policy: single owner of strip-vs-re-pad; adapters keep only SYNTAX --
#   require-side (echo-back enforced; replays 400 without the field):
#     kimi     — provider kimi-coding/kimi-coding-cn, or host api.kimi.com / moonshot.ai /
#                moonshot.cn. Host-driven on purpose: aggregators re-exporting kimi reject it.
#     deepseek — provider "deepseek", model contains "deepseek", or host api.deepseek.com.
#                V4 rejects empty-string pads → " " single space.
#     mimo     — provider "xiaomi", model contains "mimo", or host *.xiaomimimo.com.
#   strict side (field rejected 400/422 "Extra inputs are not permitted"): everyone else —
#     Mistral, Cerebras, Groq, SambaNova, … Strip the key entirely, even a one-space pad.

_REASONING_ECHO_RULES: tuple = (
    # (family, exact providers (raw), exact providers (lowered), model substrings (lowered), hosts)
    ("kimi", frozenset({"kimi-coding", "kimi-coding-cn"}), frozenset(), (), ("api.kimi.com", "moonshot.ai", "moonshot.cn")),
    ("deepseek", frozenset(), frozenset({"deepseek"}), ("deepseek",), ("api.deepseek.com",)),
    ("mimo", frozenset(), frozenset({"xiaomi"}), ("mimo",), ("api.xiaomimimo.com", "xiaomimimo.com")),
)
_REASONING_ECHO_RULE_BY_FAMILY = {rule[0]: rule for rule in _REASONING_ECHO_RULES}


def matches_reasoning_echo_family(family: str, provider: Any, model: Any, base_url: Any) -> bool:
    """True when (provider, model, base_url) matches one echo-back family (families can overlap;
    membership is tested independently). Raises KeyError for an unknown family."""
    from utils import base_url_host_matches

    _, raw_providers, lowered_providers, model_subs, hosts = _REASONING_ECHO_RULE_BY_FAMILY[family]
    model_lower = (model or "").lower()
    return (
        provider in raw_providers
        or (provider or "").lower() in lowered_providers
        or any(sub in model_lower for sub in model_subs)
        or any(base_url_host_matches(base_url, host) for host in hosts)
    )


def reasoning_echo_family(provider: Any, model: Any, base_url: Any) -> "str | None":
    """``"kimi"`` / ``"deepseek"`` / ``"mimo"`` (first match in table order) when the
    endpoint enforces reasoning_content echo-back, else ``None`` (strip side)."""
    return next(
        (rule[0] for rule in _REASONING_ECHO_RULES if matches_reasoning_echo_family(rule[0], provider, model, base_url)),
        None,
    )


def needs_reasoning_echo(provider: Any, model: Any, base_url: Any) -> bool:
    """True when the endpoint requires reasoning_content echo-back."""
    return reasoning_echo_family(provider, model, base_url) is not None


def stale_thinking_reaches_wire(api_mode: Any, provider: Any, model: Any, base_url: Any) -> bool:
    """True when stale assistant ``reasoning``/``reasoning_content`` text is actually replayed
    on the wire for the active route.

    The single wire-truth predicate the compaction TRIGGER estimator and the tail-budget
    walks must share: if they disagree, a reasoning-heavy session can look over-threshold
    to preflight yet fully tail-protected to the walk — an infinite compaction loop.
    ``codex_responses`` never reads the text keys (continuity rides the encrypted sidecar);
    echo-back families replay stored ``reasoning_content`` verbatim; everyone else strips.
    """
    return (api_mode or "") != "codex_responses" and needs_reasoning_echo(provider, model, base_url)


def apply_reasoning_content_policy(source_msg: dict, api_msg: dict, needs_thinking_pad: bool) -> None:
    """Copy provider-facing reasoning fields onto an API replay message (mutates ``api_msg``).
    ``needs_thinking_pad`` is the require-side flag (``needs_reasoning_echo``)."""
    if source_msg.get("role") != "assistant":
        return
    if not needs_thinking_pad:
        # Strict side: never carry the field — a reasoning primary pads history with " ",
        # then a fallback to Mistral/Cerebras/Groq replays the pad and 422s. Also drops a
        # non-string value (None after compaction): never pass null to the API.
        api_msg.pop("reasoning_content", None)
        return
    existing = source_msg.get("reasoning_content")
    if isinstance(existing, str):
        # Explicit value: preserve verbatim, upgrading legacy "" to " " (DeepSeek V4 400s on "").
        api_msg["reasoning_content"] = existing or " "
        return
    reasoning = source_msg.get("reasoning")
    if isinstance(reasoning, str) and reasoning and not source_msg.get("tool_calls"):
        # Healthy session: promote internal 'reasoning' → 'reasoning_content'.
        api_msg["reasoning_content"] = reasoning
        return
    # tool_calls + 'reasoning' but no 'reasoning_content' means the reasoning came from
    # ANOTHER provider (DeepSeek's own build pins reasoning_content for tool-call turns):
    # pad without leaking foreign CoT. No reasoning at all: every assistant turn still needs
    # the field; " " (not "") because DeepSeek V4 rejects empty string.
    api_msg["reasoning_content"] = " "


def reapply_reasoning_echo(api_messages: list, needs_thinking_pad: bool) -> int:
    """Re-pad (or strip) assistant turns' reasoning_content for the ACTIVE provider.

    ``api_messages`` is built once under the primary provider; a mid-conversation
    fallback can switch providers, so baked-in fields must be reconciled: TO a
    require-side provider re-applies the pad (else 400), TO a strict provider strips it
    (else 422). Idempotent. Returns the number of assistant turns changed.
    """
    changed = 0
    for api_msg in api_messages:
        if api_msg.get("role") != "assistant":
            continue
        if needs_thinking_pad:
            if api_msg.get("reasoning_content"):
                continue
            apply_reasoning_content_policy(api_msg, api_msg, needs_thinking_pad)
            if api_msg.get("reasoning_content"):
                changed += 1
        elif "reasoning_content" in api_msg:
            api_msg.pop("reasoning_content", None)
            changed += 1
    return changed


# Image / multimodal parts are deliberately NOT consolidated here: per-adapter handling is
# format-specific SYNTAX. The one shared image POLICY is ``_strip_images_from_messages``.
