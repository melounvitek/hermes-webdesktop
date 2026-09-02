"""Anthropic prompt caching strategy — pure functions, no AIAgent dependency.

Default layout: 4 cache_control breakpoints — the static system prefix, the end
of the system prompt, and the last 2 non-system messages. Without a static
prefix: one system breakpoint plus the last 3 messages. All markers share one
TTL (5m or 1h). This keeps intra-session caching while letting new sessions
reuse the stable system-prompt prefix.
"""

import copy
from dataclasses import dataclass
from typing import Any, Dict, List

from agent.prompt_cache_boundary import find_stable_prefix


@dataclass(frozen=True)
class PromptCachePlan:
    """Request-local message and tool sections with their cache markers."""

    messages: List[Dict[str, Any]]
    tools: List[Dict[str, Any]]


def envelope_tool_part_cache_markers_supported(
    provider: str | None, base_url: str | None
) -> bool:
    """Whether the envelope-layout route honors part-level markers on role:tool.

    OpenRouter (and Nous Portal, which proxies to it) relocate a part-level
    ``cache_control`` onto the ``tool_result`` block during OpenAI→Anthropic
    translation. LiteLLM-style proxies copy parts verbatim, so the marker lands
    at ``tool_result.content[0]`` — forbidden by the Anthropic schema, a
    non-retryable 400. On those routes tool messages carry no part markers and
    the breakpoint budget reallocates to the nearest eligible message.
    """
    from agent.agent_runtime_helpers import _is_litellm_route

    return not _is_litellm_route((provider or "").strip().lower(), base_url or "")


def _apply_cache_marker(
    msg: dict,
    cache_marker: dict,
    native_anthropic: bool = False,
    tool_part_markers: bool = True,
) -> None:
    """Add cache_control to a single message, handling all format variations."""
    role = msg.get("role", "")
    content = msg.get("content")

    if role == "tool" and native_anthropic:
        # Top-level marker; the native adapter moves it inside tool_result.
        msg["cache_control"] = cache_marker
        return
    if role == "tool" and not tool_part_markers:
        # LiteLLM-style envelope: a part marker becomes
        # tool_result.content[0].cache_control → non-retryable 400.
        return

    if content is None or content == "":
        # Envelope layout: OpenRouter rejects top-level cache_control on
        # role:tool (silent hang), and ignores it on empty assistant turns
        # (pure tool_calls) — neither has a content part to carry it.
        if role in ("tool", "assistant") and not native_anthropic:
            return
        msg["cache_control"] = cache_marker
        return

    if isinstance(content, str):
        if role == "user":
            stable_prefix = find_stable_prefix(content)
            if stable_prefix is not None:
                suffix = content[len(stable_prefix):]
                if suffix.strip():
                    # Builder-declared boundary: the scaffold carries the
                    # breakpoint and the volatile tail rides unmarked, so a
                    # changed ticket ID/timestamp no longer invalidates the
                    # skill body. Request-local only — the stored message
                    # stays a plain string.
                    msg["content"] = [
                        {"type": "text", "text": stable_prefix, "cache_control": cache_marker},
                        {"type": "text", "text": suffix},
                    ]
                    return
        msg["content"] = [
            {"type": "text", "text": content, "cache_control": cache_marker}
        ]
        return

    if isinstance(content, list) and content:
        last = content[-1]
        if isinstance(last, dict):
            last["cache_control"] = cache_marker


def _can_carry_marker(
    msg: dict, native_anthropic: bool, tool_part_markers: bool = True
) -> bool:
    """True if a marker on this message is actually honored by the provider.

    Native Anthropic honors every message (the adapter relocates top-level
    markers). The envelope layout only honors markers inside content parts, so
    empty-content messages would waste one of the four breakpoints; with
    ``tool_part_markers=False`` (LiteLLM-style routes) every role:tool message
    is excluded too, since its part marker would be rejected with a 400.
    Must agree with :func:`_apply_cache_marker`, which marks only the LAST part.
    """
    if native_anthropic:
        return True
    if msg.get("role") == "tool" and not tool_part_markers:
        return False
    content = msg.get("content")
    if content is None or content == "":
        return False
    if isinstance(content, list):
        # Mirrors _apply_cache_marker (marks only the LAST part): a list whose
        # last element isn't a dict cannot receive a marker.
        return bool(content) and isinstance(content[-1], dict)
    return isinstance(content, str)


def _build_marker(ttl: str) -> Dict[str, str]:
    """Build a cache_control marker dict for the given TTL ('5m' or '1h')."""
    marker: Dict[str, str] = {"type": "ephemeral"}
    if ttl == "1h":
        marker["ttl"] = "1h"
    return marker


# Alibaba-family providers (Qwen routes): documented five-minute context cache,
# Anthropic 1h tier rejected. Shared with
# agent_runtime_helpers.anthropic_prompt_cache_policy so the cache-policy
# opt-in and the TTL clamp never desync. Do NOT narrow this set to extend a
# TTL — it also drives the marker-layout opt-in, so narrowing DISABLES caching.
ALIBABA_FAMILY_PROVIDERS = frozenset({
    "opencode",
    "opencode-go",
    "opencode-zen",
    "alibaba",
})

# 1h-tier ALLOW-list: only routes wire-measured to retain a 1h marker (delayed
# read past 5 minutes with no intervening call — an intervening read renews the
# window and masks expiry). Other opencode routes stay clamped because they are
# UNMEASURED, not known-bad. Note opencode-go labels every write
# `ephemeral_5m_input_tokens` regardless of requested ttl; that label is not
# evidence of the retention window.
MEASURED_1H_PROVIDERS = frozenset({
    "opencode-go",
})

# Models measured to ignore the 1h tier on a MEASURED_1H_PROVIDERS route.
# Consulted only there: the same model on its own Anthropic-compatible endpoint
# is a separate cache-eligible route and must not inherit this clamp.
NO_1H_TIER_MODELS = frozenset({
    "minimax-m2.5",
})


def _flat_model(model: str) -> str:
    """Bare model id, tolerating aggregator prefixes (``vendor/model``)."""
    return (model or "").strip().rsplit("/", 1)[-1].lower()


def is_qwen_model(model: str) -> bool:
    """True when ``model`` names a Qwen-family model (case-insensitive).

    Shared with ``agent_runtime_helpers.anthropic_prompt_cache_policy`` so the
    cache-policy opt-in and the TTL clamp never desync.
    """
    return "qwen" in (model or "").lower()


def effective_cache_ttl(
    ttl: str | None,
    *,
    model: str = "",
    provider: str = "",
) -> str:
    """Clamp a requested cache TTL to what the destination route supports.

    Qwen/Alibaba routes document a five-minute window and drop the ``1h``
    tier, so a configured ``1h`` regresses to ``5m`` there instead of creating
    a false 1h-cache expectation — except on ``MEASURED_1H_PROVIDERS``, which
    keep ``1h`` minus any ``NO_1H_TIER_MODELS`` model. The measured-route check
    runs BEFORE the generic Qwen clamp, which would otherwise swallow every
    Qwen model on it. ``None`` resolves to ``5m``.
    """
    if ttl != "1h":
        return ttl or "5m"
    if (provider or "").lower() in MEASURED_1H_PROVIDERS:
        # Checked BEFORE the generic Qwen clamp (which would swallow every Qwen
        # model on this route); the per-model denial stays nested so an
        # opencode-go observation cannot reclamp the same model on another route.
        return "5m" if _flat_model(model) in NO_1H_TIER_MODELS else "1h"
    if is_qwen_model(model) or (provider or "").lower() in ALIBABA_FAMILY_PROVIDERS:
        return "5m"
    return "1h"


def _apply_system_cache_markers(
    message: dict,
    cache_marker: dict,
    static_system_prefix: str | None,
    *,
    native_anthropic: bool,
    mark_suffix: bool = True,
    fallback_to_whole: bool = True,
) -> int:
    """Mark the static system prefix (and optionally the full prompt).

    The system prompt stays one stored string; it is split only in the
    outgoing request so persistence and non-Anthropic transports are
    unchanged. ``mark_suffix=False`` is the tool-cache-plan layout (suffix
    unmarked, its budget spent on the tools array). ``fallback_to_whole=False``
    marks nothing when the prefix split is impossible. When the prompt IS the
    prefix (empty/whitespace suffix) the whole message is marked as one block —
    never a split with an empty text block, which Anthropic rejects.

    Returns the number of markers applied (0, 1, or 2).
    """
    content = message.get("content")
    if (
        isinstance(static_system_prefix, str)
        and static_system_prefix
        and isinstance(content, str)
        and content.startswith(static_system_prefix)
    ):
        suffix = content[len(static_system_prefix):]
        if suffix.strip():
            suffix_part: dict = {"type": "text", "text": suffix}
            if mark_suffix:
                suffix_part["cache_control"] = cache_marker
            message["content"] = [
                {"type": "text", "text": static_system_prefix, "cache_control": cache_marker},
                suffix_part,
            ]
            return 2 if mark_suffix else 1
        _apply_cache_marker(message, cache_marker, native_anthropic=native_anthropic)
        return 1

    if not fallback_to_whole:
        return 0
    _apply_cache_marker(message, cache_marker, native_anthropic=native_anthropic)
    return 1


def strip_anthropic_cache_control(
    api_messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Remove ``cache_control`` markers and undo decoration-produced list shapes.

    Used before re-decorating after a mid-turn provider failover, so the
    mutated undecorated shape is preserved while markers match the new
    provider's policy. Flattening back to a plain string is restricted to the
    exact shapes :func:`apply_anthropic_cache_control` produces from string
    content — a single text part, the two-part ``[static, volatile]`` system
    split, or the two-part skill split — so the ``""``-join is provably
    byte-exact; organic multi-part text and parts with extra keys keep their
    structure. Marker removal is copy-on-write on part dicts: parts can alias
    caller-held lists and stripping must never rewrite the stored transcript.

    Mutates the top-level message dicts in place and returns the same list.
    """
    for msg in api_messages:
        if not isinstance(msg, dict):
            continue
        msg.pop("cache_control", None)
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        # The builder-declared skill split is the only decoration that marks
        # the FIRST part of a user message (list content is otherwise marked
        # on the last part; the [static, volatile] split is system-only), so
        # the shape alone identifies it even after the prefix registry has
        # evicted the entry.
        skill_split_shape = (
            msg.get("role") == "user"
            and len(content) == 2
            and isinstance(content[0], dict)
            and isinstance(content[1], dict)
            and "cache_control" in content[0]
            and "cache_control" not in content[1]
        )
        if any(isinstance(part, dict) and "cache_control" in part for part in content):
            content = [
                {k: v for k, v in part.items() if k != "cache_control"}
                if isinstance(part, dict) and "cache_control" in part
                else part
                for part in content
            ]
            msg["content"] = content
        decoration_shape = content and all(
            isinstance(part, dict)
            and part.get("type", "text") == "text"
            and isinstance(part.get("text"), str)
            and set(part.keys()) <= {"type", "text"}
            for part in content
        ) and (
            len(content) == 1
            or (msg.get("role") == "system" and len(content) == 2)
            or skill_split_shape
        )
        if decoration_shape:
            msg["content"] = "".join(part["text"] for part in content)
    return api_messages


def strip_anthropic_tool_cache_control(tools: List[Dict[str, Any]] | None) -> List[Dict[str, Any]]:
    """Return copied tools without request-local Anthropic cache markers."""
    cleaned = copy.deepcopy(tools or [])
    for tool in cleaned:
        if isinstance(tool, dict):
            tool.pop("cache_control", None)
    return cleaned


def _count_cache_markers(messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]) -> int:
    """Count the wire-visible cache markers in a request-local plan."""
    count = sum(
        1
        for message in messages
        if isinstance(message, dict) and "cache_control" in message
    )
    count += sum(
        1
        for message in messages
        if isinstance(message, dict) and isinstance(message.get("content"), list)
        for part in message["content"]
        if isinstance(part, dict) and "cache_control" in part
    )
    return count + sum(
        1 for tool in tools if isinstance(tool, dict) and "cache_control" in tool
    )


def _completed_transaction_endpoint_indexes(
    messages: List[Dict[str, Any]], *, native_anthropic: bool,
) -> List[int]:
    """Select legal ends of completed tool runs and ordinary turns."""
    endpoints: List[int] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        if not isinstance(message, dict) or message.get("role") == "system":
            index += 1
            continue

        if message.get("role") == "assistant" and message.get("tool_calls"):
            result_start = index + 1
            result_end = result_start
            while result_end < len(messages):
                result = messages[result_end]
                if not isinstance(result, dict) or result.get("role") != "tool":
                    break
                result_end += 1
            if result_end > result_start:
                endpoint = result_end - 1
                if _can_carry_marker(messages[endpoint], native_anthropic):
                    endpoints.append(endpoint)
            index = result_end
            continue

        if message.get("role") == "tool":
            while index < len(messages):
                result = messages[index]
                if not isinstance(result, dict) or result.get("role") != "tool":
                    break
                index += 1
            continue

        if message.get("role") == "user" and index + 1 < len(messages):
            index += 1
            continue

        if (
            message.get("role") == "assistant"
            and message.get("content") in (None, "")
        ):
            index += 1
            continue

        if _can_carry_marker(message, native_anthropic):
            endpoints.append(index)
        index += 1
    return endpoints


def build_prompt_cache_plan(
    api_messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]] | None,
    *,
    cache_ttl: str = "5m",
    native_anthropic: bool = False,
    static_system_prefix: str | None = None,
    direct_native_tool_cache: bool = False,
    tool_part_markers: bool = True,
) -> PromptCachePlan:
    """Build isolated cache sections for one resolved request destination.

    ``tool_part_markers=False`` (LiteLLM-style envelope routes) keeps
    ``cache_control`` off role:tool content parts; breakpoints reallocate to
    the nearest eligible non-tool message.
    """
    messages = copy.deepcopy(api_messages or [])
    strip_anthropic_cache_control(messages)
    planned_tools = strip_anthropic_tool_cache_control(tools)

    if not direct_native_tool_cache or not planned_tools:
        planned_messages = apply_anthropic_cache_control(
            messages,
            cache_ttl=cache_ttl,
            native_anthropic=native_anthropic,
            static_system_prefix=static_system_prefix,
            tool_part_markers=tool_part_markers,
        )
        return PromptCachePlan(messages=planned_messages, tools=planned_tools)

    marker = _build_marker(cache_ttl)
    if (
        messages
        and isinstance(messages[0], dict)
        and messages[0].get("role") == "system"
    ):
        # Tool-cache layout: only the static prefix carries a system-side
        # marker; the volatile suffix's budget is spent on the tools array.
        _apply_system_cache_markers(
            messages[0],
            marker,
            static_system_prefix,
            native_anthropic=True,
            mark_suffix=False,
            fallback_to_whole=False,
        )
    planned_tools[-1]["cache_control"] = dict(marker)
    for endpoint in _completed_transaction_endpoint_indexes(
        messages,
        native_anthropic=True,
    )[-2:]:
        _apply_cache_marker(messages[endpoint], marker, native_anthropic=True)

    return PromptCachePlan(messages=messages, tools=planned_tools)


def apply_anthropic_cache_control(
    api_messages: List[Dict[str, Any]],
    cache_ttl: str = "5m",
    native_anthropic: bool = False,
    static_system_prefix: str | None = None,
    tool_part_markers: bool = True,
) -> List[Dict[str, Any]]:
    """Apply Anthropic cache-control markers to API messages.

    With a matching ``static_system_prefix`` the prefix gets an early marker
    and the full system prompt a trailing one; the remaining two markers go to
    the latest cacheable non-system messages. Without it, the legacy
    system-and-3 layout applies. Idempotent: pre-existing markers are stripped
    from a per-message copy first, so repeated calls never accumulate past 4
    markers; a shallow top-level copy suffices because
    :func:`strip_anthropic_cache_control` is copy-on-write on content parts.

    Returns:
        Shallow copy of message list with selective deep copies of modified messages.
    """
    if not api_messages:
        return api_messages

    messages = list(api_messages)
    marker = _build_marker(cache_ttl)

    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        has_marker = "cache_control" in msg or (
            isinstance(content, list)
            and any(isinstance(part, dict) and "cache_control" in part for part in content)
        )
        if has_marker:
            messages[i] = strip_anthropic_cache_control([dict(msg)])[0]

    breakpoints_used = 0

    if messages[0].get("role") == "system":
        messages[0] = copy.deepcopy(messages[0])
        breakpoints_used = _apply_system_cache_markers(
            messages[0],
            marker,
            static_system_prefix,
            native_anthropic=native_anthropic,
        )

    remaining = 4 - breakpoints_used
    non_sys = [
        i
        for i in range(len(messages))
        if messages[i].get("role") != "system"
        and _can_carry_marker(
            messages[i],
            native_anthropic=native_anthropic,
            tool_part_markers=tool_part_markers,
        )
    ]
    for idx in non_sys[-remaining:]:
        messages[idx] = copy.deepcopy(messages[idx])
        _apply_cache_marker(
            messages[idx],
            marker,
            native_anthropic=native_anthropic,
            tool_part_markers=tool_part_markers,
        )

    return messages
