"""Assorted AIAgent runtime helpers, moved out of run_agent.py.

Each function takes the parent ``AIAgent`` as ``agent`` except the stateless
helpers (``sanitize_tool_call_arguments``, ``drop_thinking_only_and_merge_users``).
"""

from __future__ import annotations

import copy
import json
import logging
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from hermes_cli.timeouts import get_provider_request_timeout
from agent.message_sanitization import (
    _FULL_ARGS_LOG_BOUND,
    coalesce_tool_call_id,
    tool_call_id_variants,
    tool_result_id_variants,
)
from agent.prompt_builder import format_steer_marker
from agent.tool_dispatch_helpers import _trajectory_normalize_msg, make_tool_result_message
from agent.trajectory import convert_scratchpad_to_think
from agent.credential_pool import (
    STATUS_EXHAUSTED,
    credential_pool_matches_provider,
    resolve_runtime_pool_key,
)
from agent.error_classifier import FailoverReason
from agent.turn_context import drop_stale_api_content
from utils import base_url_host_matches, base_url_hostname, env_var_enabled, atomic_json_write

logger = logging.getLogger(__name__)


# Cap consecutive same-entry OAuth token refreshes on a persistent auth failure;
# without it a single-entry pool re-mints forever and never reaches fallback (#26080).
_MAX_AUTH_REFRESH_ATTEMPTS = 2


_REASONING_TAG_NAMES = ("think", "thinking", "reasoning", "REASONING_SCRATCHPAD", "thought")
_TOOL_CALL_TAG_NAMES = ("tool_call", "tool_calls", "tool_result", "function_call", "function_calls")

_REASONING_BLOCK_PATTERNS = tuple(
    re.compile(rf"<{name}>.*?</{name}>", re.DOTALL | re.IGNORECASE)
    for name in _REASONING_TAG_NAMES
)

_TOOL_CALL_BLOCK_PATTERNS = tuple(
    re.compile(rf"<{name}\b[^>]*>.*?</{name}>", re.DOTALL | re.IGNORECASE)
    for name in _TOOL_CALL_TAG_NAMES
)

# Named <function name=...> blocks; see strip_think_blocks step 1c for the
# boundary/tempered-dot rationale.
_NAMED_FUNCTION_BLOCK_PATTERN = re.compile(
    r'(?:(?<=^)|(?<=[\n\r.!?:]))[ \t]*'
    r'<function\b[^>]*\bname\s*=[^>]*>'
    r'(?:(?:(?!</function>).)*)</function>',
    re.DOTALL | re.IGNORECASE,
)

_UNTERMINATED_REASONING_BLOCK_PATTERN = re.compile(
    rf'(?:^|\n)[ \t]*<(?:{"|".join(_REASONING_TAG_NAMES)})\b[^>]*>.*$',
    re.DOTALL | re.IGNORECASE,
)

_ORPHAN_REASONING_TAG_PATTERN = re.compile(
    rf'</?(?:{"|".join(_REASONING_TAG_NAMES)})>\s*',
    re.IGNORECASE,
)

_STRAY_TOOL_CALL_CLOSER_PATTERN = re.compile(
    rf'</(?:{"|".join(_TOOL_CALL_TAG_NAMES)}|function)>\s*',
    re.IGNORECASE,
)


def _ra():
    """Lazy ``run_agent`` reference for test-patch routing."""
    import run_agent
    return run_agent


AGENT_RUNTIME_POST_HOOK_TOOL_NAMES = frozenset(
    {"todo_list", "session_search", "memory", "clarify", "read_terminal", "desktop_preview", "drive_preview", "annotate_preview", "read_window_below", "setup_mcp", "gui_tour", "delegate_task"}
)


def convert_to_trajectory_format(agent, messages: List[Dict[str, Any]], user_query: str, completed: bool) -> List[Dict[str, Any]]:
    """Convert internal message history to trajectory format for saving."""
    # Trajectories are text-only: swap image-bearing tool messages for their
    # text_summary so ~1MB base64 blobs are not embedded.
    messages = [_trajectory_normalize_msg(m) for m in messages]
    trajectory = []
    
    system_msg = (
        "You are a function calling AI model. You are provided with function signatures within <tools> </tools> XML tags. "
        "You may call one or more functions to assist with the user query. If available tools are not relevant in assisting "
        "with user query, just respond in natural conversational language. Don't make assumptions about what values to plug "
        "into functions. After calling & executing the functions, you will be provided with function results within "
        "<tool_response> </tool_response> XML tags. Here are the available tools:\n"
        f"<tools>\n{agent._format_tools_for_system_message()}\n</tools>\n"
        "For each function call return a JSON object, with the following pydantic model json schema for each:\n"
        "{'title': 'FunctionCall', 'type': 'object', 'properties': {'name': {'title': 'Name', 'type': 'string'}, "
        "'arguments': {'title': 'Arguments', 'type': 'object'}}, 'required': ['name', 'arguments']}\n"
        "Each function call should be enclosed within <tool_call> </tool_call> XML tags.\n"
        "Example:\n<tool_call>\n{'name': <function-name>,'arguments': <args-dict>}\n</tool_call>"
    )
    
    trajectory.append({
        "from": "system",
        "value": system_msg
    })
    
    trajectory.append({
        "from": "human",
        "value": user_query
    })
    
    # Skip messages[0] (already added). Prefill is injected at API-call time
    # only, so no offset adjustment is needed.
    i = 1
    
    while i < len(messages):
        msg = messages[i]
        
        if msg["role"] == "assistant":
            if "tool_calls" in msg and msg["tool_calls"]:
                content = ""
                
                # Prepend reasoning in <think> tags if available (native thinking tokens)
                if msg.get("reasoning") and msg["reasoning"].strip():
                    content = f"<think>\n{msg['reasoning']}\n</think>\n"
                
                if msg.get("content") and msg["content"].strip():
                    # <REASONING_SCRATCHPAD> -> <think> (model reasons via XML when native thinking is off)
                    content += convert_scratchpad_to_think(msg["content"]) + "\n"
                
                for tool_call in msg["tool_calls"]:
                    if not tool_call or not isinstance(tool_call, dict): continue
                    # Arguments were validated during conversation; try/except is a safety net
                    try:
                        arguments = json.loads(tool_call["function"]["arguments"]) if isinstance(tool_call["function"]["arguments"], str) else tool_call["function"]["arguments"]
                    except json.JSONDecodeError:
                        # Should not happen (validated during the conversation); degrade to {} rather than abort.
                        logger.warning("Unexpected invalid JSON in trajectory conversion: %s", tool_call['function']['arguments'][:100])
                        arguments = {}
                    
                    tool_call_json = {
                        "name": tool_call["function"]["name"],
                        "arguments": arguments
                    }
                    content += f"<tool_call>\n{json.dumps(tool_call_json, ensure_ascii=False)}\n</tool_call>\n"
                
                # Every gpt turn gets a <think> block (empty if none) for a consistent training format
                if "<think>" not in content:
                    content = "<think>\n</think>\n" + content
                
                trajectory.append({
                    "from": "gpt",
                    "value": content.rstrip()
                })
                
                tool_responses = []
                j = i + 1
                while j < len(messages) and messages[j]["role"] == "tool":
                    tool_msg = messages[j]
                    tool_response = "<tool_response>\n"
                    
                    # Pretty-print tool content if it looks like JSON
                    tool_content = tool_msg["content"]
                    try:
                        if tool_content.strip().startswith(("{", "[")):
                            tool_content = json.loads(tool_content)
                    except (json.JSONDecodeError, AttributeError):
                        pass  # Keep as string if not valid JSON
                    
                    tool_index = len(tool_responses)
                    tool_name = (
                        msg["tool_calls"][tool_index]["function"]["name"]
                        if tool_index < len(msg["tool_calls"])
                        else "unknown"
                    )
                    tool_response += json.dumps({
                        "tool_call_id": tool_msg.get("tool_call_id", ""),
                        "name": tool_name,
                        "content": tool_content
                    }, ensure_ascii=False)
                    tool_response += "\n</tool_response>"
                    tool_responses.append(tool_response)
                    j += 1
                
                if tool_responses:
                    trajectory.append({
                        "from": "tool",
                        "value": "\n".join(tool_responses)
                    })
                    i = j - 1  # Skip the tool messages we just processed
            
            else:
                content = ""
                
                # Prepend reasoning in <think> tags if available (native thinking tokens)
                if msg.get("reasoning") and msg["reasoning"].strip():
                    content = f"<think>\n{msg['reasoning']}\n</think>\n"
                
                # <REASONING_SCRATCHPAD> -> <think> (model reasons via XML when native thinking is off)
                raw_content = msg["content"] or ""
                content += convert_scratchpad_to_think(raw_content)
                
                # Every gpt turn gets a <think> block (empty if none) for a consistent training format
                if "<think>" not in content:
                    content = "<think>\n</think>\n" + content
                
                trajectory.append({
                    "from": "gpt",
                    "value": content.strip()
                })
        
        elif msg["role"] == "user":
            trajectory.append({
                "from": "human",
                "value": msg["content"]
            })
        
        i += 1
    
    return trajectory



def sanitize_tool_call_arguments(
    messages: list,
    *,
    logger=None,
    session_id: str = None,
    cursor: Optional[dict] = None,
) -> int:
    """Repair corrupted assistant tool-call argument JSON in-place.

    ``cursor`` (optional caller-owned dict) stores under ``"prefix"`` strong
    references to the message objects validated last call; the longest
    ``is``-identical prefix is skipped on the next call. Skipping is safe
    because only the surrogate/non-ASCII sanitizers mutate arguments on live
    dicts (inside JSON string values), and every other path replaces or
    reorders dicts, breaking identity. Strong refs (not ``id()``) rule out
    address-reuse aliasing (#50372).
    """
    log = logger or logging.getLogger(__name__)
    if not isinstance(messages, list):
        return 0

    start_index = 0
    if cursor is not None:
        prev_prefix = cursor.get("prefix")
        if isinstance(prev_prefix, list):
            limit = min(len(prev_prefix), len(messages))
            while start_index < limit and messages[start_index] is prev_prefix[start_index]:
                start_index += 1

    repaired = 0
    marker = _ra().AIAgent._TOOL_CALL_ARGUMENTS_CORRUPTION_MARKER

    def _prepend_marker(tool_msg: dict) -> None:
        existing = tool_msg.get("content")
        if isinstance(existing, str):
            if not existing:
                tool_msg["content"] = marker
            elif not existing.startswith(marker):
                tool_msg["content"] = f"{marker}\n{existing}"
            return
        if existing is None:
            tool_msg["content"] = marker
            return
        try:
            existing_text = json.dumps(existing)
        except TypeError:
            existing_text = str(existing)
        tool_msg["content"] = f"{marker}\n{existing_text}"

    message_index = start_index
    while message_index < len(messages):
        msg = messages[message_index]
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            message_index += 1
            continue

        tool_calls = msg.get("tool_calls")
        if not isinstance(tool_calls, list) or not tool_calls:
            message_index += 1
            continue

        insert_at = message_index + 1
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function")
            if not isinstance(function, dict):
                continue

            arguments = function.get("arguments")
            if arguments is None or arguments == "":
                function["arguments"] = "{}"
                continue
            if isinstance(arguments, str) and not arguments.strip():
                function["arguments"] = "{}"
                continue
            if not isinstance(arguments, str):
                continue

            try:
                json.loads(arguments)
            except json.JSONDecodeError:
                # Use canonical ``call_id || id`` precedence so scan and stub share the id
                # the pipeline uses; bare ``id`` misses Codex call_id results and orphans a stub (#58168).
                tool_call_id = _ra().AIAgent._get_tool_call_id_static(tool_call) or None
                function_name = function.get("name", "?")
                # Log the FULL (bounded) argument string: we are about to overwrite the only
                # copy, which may hold real user content from a truncated write_file/patch (#80498).
                preview = arguments[:_FULL_ARGS_LOG_BOUND]
                log.warning(
                    "Corrupted tool_call arguments repaired before request "
                    "(session=%s, message_index=%s, tool_call_id=%s, function=%s, "
                    "original_arguments=%r)",
                    session_id or "-",
                    message_index,
                    tool_call_id or "-",
                    function_name,
                    preview,
                )
                function["arguments"] = "{}"

                existing_tool_msg = None
                scan_index = message_index + 1
                while scan_index < len(messages):
                    candidate = messages[scan_index]
                    if not isinstance(candidate, dict) or candidate.get("role") != "tool":
                        break
                    if (
                        tool_result_id_variants(candidate.get("tool_call_id"))
                        & tool_call_id_variants(tool_call)
                    ):
                        existing_tool_msg = candidate
                        break
                    scan_index += 1

                if existing_tool_msg is None:
                    messages.insert(
                        insert_at,
                        make_tool_result_message(
                            function_name if function_name != "?" else "",
                            marker,
                            tool_call_id,
                        ),
                    )
                    insert_at += 1
                else:
                    _prepend_marker(existing_tool_msg)

                repaired += 1

        message_index += 1

    if cursor is not None:
        # Strong refs to the objects validated this call; any divergence
        # (compression, undo, repair, steer) forces a re-scan from that index.
        cursor["prefix"] = messages[:]

    return repaired


# Session-scoped in-flight registry for note_turn_start. The gateway caches agents
# per routing key while the transcript is keyed by session_id (many-to-one), so two
# agent objects can run concurrent turns on one session unseen by per-agent state (#64934).
_INFLIGHT_TURNS_BY_SESSION: Dict[str, Tuple[str, float]] = {}
_INFLIGHT_TURNS_LOCK = threading.Lock()


def note_turn_start(agent, turn_id: str):
    """Tripwire: warn when a turn starts while a previous turn of the same agent
    or the same session (on another agent object) has not finished its persist.

    Does not prevent the overlap; it names it with both turn ids so the dispatch
    route that bypassed the busy guard can be found in logs. Returns the previous
    in-flight turn_id on overlap, else None; takes ownership of the slot either way.
    """
    prev = getattr(agent, "_inflight_turn_id", None)
    prev_started = getattr(agent, "_inflight_turn_started", 0.0)
    agent._inflight_turn_id = turn_id
    agent._inflight_turn_started = time.time()
    overlap = None
    if prev and prev != turn_id:
        logger.warning(
            "turn %s starting while turn %s (started %.0fs ago) has not "
            "completed its turn-end persist (session=%s) — concurrent turns "
            "on one session; transcript writes may interleave",
            turn_id,
            prev,
            time.time() - prev_started if prev_started else -1.0,
            getattr(agent, "session_id", None) or "-",
        )
        overlap = prev

    # Cross-agent leg: same session_id in flight under another agent object
    # (busy guard is keyed by routing key and cannot see it). Persist-disabled
    # forks share the parent's session_id but never write, so they must not
    # register or pop here (note_turn_persisted skips them symmetrically).
    session_id = getattr(agent, "session_id", None)
    if session_id and not getattr(agent, "_persist_disabled", False):
        now = time.time()
        with _INFLIGHT_TURNS_LOCK:
            entry = _INFLIGHT_TURNS_BY_SESSION.get(session_id)
            _INFLIGHT_TURNS_BY_SESSION[session_id] = (turn_id, now)
        # Record the session id registered under: compression can rotate
        # agent.session_id mid-turn and persist must pop the slot actually held.
        agent._inflight_turn_session_id = session_id
        if entry and entry[0] not in (turn_id, prev):
            logger.warning(
                "turn %s starting while turn %s (started %.0fs ago) is still "
                "in flight on session %s under a different agent object — "
                "two routing keys are mapped to one session_id; concurrent "
                "turns on one session; transcript writes may interleave",
                turn_id,
                entry[0],
                now - entry[1] if entry[1] else -1.0,
                session_id,
            )
            overlap = overlap or entry[0]
    return overlap


def note_turn_persisted(agent):
    """Clear the in-flight marker at turn-end persist (see note_turn_start).

    Unconditional by design: on a real overlap the first persist clears the
    second slot and the tripwire under-reports rather than double-reports.
    """
    agent._inflight_turn_id = None
    # Persist-disabled forks never registered a slot; popping here would
    # steal the live parent turn's slot (symmetric with note_turn_start).
    if not getattr(agent, "_persist_disabled", False):
        session_id = getattr(agent, "_inflight_turn_session_id", None) or getattr(
            agent, "session_id", None
        )
        if session_id:
            with _INFLIGHT_TURNS_LOCK:
                _INFLIGHT_TURNS_BY_SESSION.pop(session_id, None)
    agent._inflight_turn_session_id = None


def _is_codex_interim(m: Dict) -> bool:
    """Codex Responses interim turn: carries its own continuation state, replayed verbatim."""
    return bool(
        m.get("codex_reasoning_items")
        or m.get("codex_message_items")
        or m.get("finish_reason") == "incomplete"
    )


def _merge_assistant_into(prev: Dict, msg: Dict) -> None:
    """Fold a consecutive assistant ``msg`` into ``prev`` (union tool_calls, concat text)."""
    prev_calls = list(prev.get("tool_calls") or [])
    new_calls = list(msg.get("tool_calls") or [])
    if new_calls:
        prev["tool_calls"] = prev_calls + new_calls
    elif prev_calls:
        prev["tool_calls"] = prev_calls
    else:
        # Drop a stale ``tool_calls: []`` at the source: strict providers (DeepSeek v4,
        # Kimi) 400 on it and it persists into replayed history.
        prev.pop("tool_calls", None)
    # Concatenate plain-text content only; leave multimodal (list) content alone.
    prev_content = prev.get("content")
    new_content = msg.get("content")
    content_rewritten = False
    if isinstance(prev_content, str) and isinstance(new_content, str):
        joined = "\n".join(p for p in (prev_content.strip(), new_content.strip()) if p)
        prev["content"] = joined
        # A falsy new_content leaves ``joined`` == prev_content; that is not a rewrite.
        content_rewritten = joined != prev_content
    elif not prev_content and new_content is not None:
        prev["content"] = new_content
        content_rewritten = new_content != prev_content
    # Carry reasoning_content from the later turn only if the earlier lacks it
    # (strict thinking providers need one on the merged tool-call turn).
    if not prev.get("reasoning_content") and msg.get("reasoning_content"):
        prev["reasoning_content"] = msg["reasoning_content"]
    # A stale ``api_content`` sidecar overrides ``content`` at API-build time and would
    # replay pre-merge bytes; drop it only when content actually changed.
    if content_rewritten:
        drop_stale_api_content(prev)


def _merge_consecutive_assistants(messages: List[Dict]) -> Tuple[List[Dict], int]:
    """Pass 0: merge consecutive assistant turns (codex interims exempt)."""
    repairs = 0
    collapsed: List[Dict] = []
    for msg in messages:
        if (
            collapsed
            and isinstance(msg, dict)
            and msg.get("role") == "assistant"
            and isinstance(collapsed[-1], dict)
            and collapsed[-1].get("role") == "assistant"
            and not _is_codex_interim(msg)
            and not _is_codex_interim(collapsed[-1])
        ):
            prev = collapsed[-1]
            # A provisional verification candidate is superseded, not unioned.
            if prev.get("finish_reason") in {"verification_required", "verify_hook_continue"}:
                collapsed[-1] = msg
            else:
                _merge_assistant_into(prev, msg)
            repairs += 1
            continue
        collapsed.append(msg)
    return collapsed, repairs


def _drop_stray_tool_results(messages: List[Dict]) -> Tuple[List[Dict], int]:
    """Pass 1: drop tool results not following a known assistant tool call.

    Consumes the whole alias group (call_id/id/response_item_id/composite) so a
    duplicate keyed on a sibling alias is not replayed to strict providers.
    """
    repairs = 0
    known_tool_ids: Dict[str, int] = {}
    matched_tool_groups: set = set()
    next_tool_group = 0
    filtered: List[Dict] = []
    for msg in messages:
        role = msg.get("role") if isinstance(msg, dict) else None
        if role == "assistant":
            known_tool_ids = {}
            matched_tool_groups = set()
            for tc in (msg.get("tool_calls") or []):
                variants = tool_call_id_variants(tc)
                if not variants:
                    continue
                group_id = next_tool_group
                next_tool_group += 1
                for tc_id in variants:
                    known_tool_ids.setdefault(tc_id, group_id)
        elif role == "tool":
            result_variants = tool_result_id_variants(msg.get("tool_call_id"))
            candidate_groups = {
                known_tool_ids[tc_id]
                for tc_id in result_variants
                if tc_id in known_tool_ids and known_tool_ids[tc_id] not in matched_tool_groups
            }
            if result_variants and not candidate_groups:
                repairs += 1
                continue
            if candidate_groups:
                matched_tool_groups.add(min(candidate_groups))
        elif role == "user":
            # A user turn closes the tool-result run; later tool messages are orphans.
            known_tool_ids = {}
            matched_tool_groups = set()
        filtered.append(msg)
    return filtered, repairs


def _prune_unanswered_tool_calls(messages: List[Dict]) -> Tuple[List[Dict], int]:
    """Pass 2: prune tool_calls not answered in the IMMEDIATELY following tool run.

    A displaced result masks the per-call stub pass and strict providers 400.
    Payload-empty turns are dropped; codex interims exempt.
    """
    repairs = 0
    pruned: List[Dict] = []
    n = len(messages)
    i = 0
    while i < n:
        msg = messages[i]
        i += 1
        if not (
            isinstance(msg, dict)
            and msg.get("role") == "assistant"
            and msg.get("tool_calls")
            and not _is_codex_interim(msg)
        ):
            pruned.append(msg)
            continue
        answered: set = set()
        j = i
        while j < n and isinstance(messages[j], dict) and messages[j].get("role") == "tool":
            tid = (messages[j].get("tool_call_id") or "").strip()
            if tid:
                answered.update(tool_result_id_variants(tid))
            j += 1
        kept_calls = [tc for tc in msg["tool_calls"] if tool_call_id_variants(tc) & answered]
        if len(kept_calls) != len(msg["tool_calls"]):
            repairs += 1
            if not kept_calls and not _msg_has_payload(
                {k: v for k, v in msg.items() if k != "tool_calls"}
            ):
                # Pruned calls were the only payload; drop the turn (empty assistant messages 400).
                continue
            if kept_calls:
                msg["tool_calls"] = kept_calls
            else:
                msg.pop("tool_calls", None)
        pruned.append(msg)
    return pruned, repairs


def _merge_consecutive_users(messages: List[Dict]) -> Tuple[List[Dict], int]:
    """Pass 3: merge consecutive plain-text user messages (no user input lost)."""
    repairs = 0
    merged: List[Dict] = []
    for msg in messages:
        if (
            merged
            and isinstance(msg, dict)
            and msg.get("role") == "user"
            and isinstance(merged[-1], dict)
            and merged[-1].get("role") == "user"
        ):
            prev = merged[-1]
            # A summary carrier followed by a new user row is a deliberate durable shape after
            # retry/rewind; never mutate the persisted carrier (sanitizers merge copies later).
            from agent.context_compressor import split_user_originated_turn

            handoff, _ = split_user_originated_turn(prev)
            prev_content = prev.get("content", "")
            new_content = msg.get("content", "")
            # Only merge plain-text content; leave multimodal (list) content alone.
            if handoff is None and isinstance(prev_content, str) and isinstance(new_content, str):
                prev["content"] = (
                    (prev_content + "\n\n" + new_content)
                    if prev_content and new_content
                    else (prev_content or new_content)
                )
                # Merged content invalidates the api_content sidecar; drop it so replay cannot use stale bytes.
                drop_stale_api_content(prev)
                repairs += 1
                continue
        merged.append(msg)
    return merged, repairs


_SEQUENCE_REPAIR_PASSES = (
    _merge_consecutive_assistants,
    _drop_stray_tool_results,
    _prune_unanswered_tool_calls,
    _merge_consecutive_users,
)


def repair_message_sequence(agent, messages: List[Dict]) -> int:
    """Collapse malformed role-alternation left in the live history; returns repair count.

    Providers require strict alternation after the system message; violations cause
    silent empty responses or HTTP 400s. Runs right before the API call as a defensive
    belt for host-fed, resumed, or replayed histories. Passes, in order: merge
    consecutive assistant turns (BEFORE orphan detection so the merged tool_call-id
    union is known); drop stray tool results; prune tool_calls unanswered in the
    immediately following tool run; merge consecutive user messages. A user turn
    directly after an assistant turn is valid and left alone.
    """
    if not messages:
        return 0
    repairs = 0
    current = messages
    for repair_pass in _SEQUENCE_REPAIR_PASSES:
        current, made = repair_pass(current)
        repairs += made
    if repairs > 0:
        # Rewrite in place so persistence/return value/DB flush see the repaired sequence.
        messages[:] = current
    return repairs


def repair_message_sequence_with_cursor(agent, messages: List[Dict]) -> int:
    """Run :func:`repair_message_sequence` and keep ``_last_flushed_db_idx`` consistent (#44837).

    Repair shrinks the list in place; counting survivors from the flushed prefix
    (identity-preserved) gives the exact new cursor, whereas a ``min()`` clamp
    would skip unflushed rows. Falls back to the clamp without a snapshot.
    """
    pre_repair_flushed_ids = None
    flush_cursor = getattr(agent, "_last_flushed_db_idx", None)
    if isinstance(flush_cursor, int) and flush_cursor > 0:
        pre_repair_flushed_ids = {id(m) for m in messages[:flush_cursor]}

    repairs = repair_message_sequence(agent, messages)

    if repairs > 0 and hasattr(agent, "_last_flushed_db_idx"):
        if pre_repair_flushed_ids is not None:
            agent._last_flushed_db_idx = sum(
                1 for m in messages if id(m) in pre_repair_flushed_ids
            )
        else:
            agent._last_flushed_db_idx = min(
                agent._last_flushed_db_idx, len(messages)
            )

    return repairs



def strip_think_blocks(agent, content: str) -> str:
    """Remove reasoning/thinking blocks from content, returning only visible text.

    Strips closed tag pairs, unterminated open tags at a block boundary (mirrors
    ``gateway/stream_consumer.py``), stray orphan tags, and all case-insensitive
    variants (think/thinking/reasoning/REASONING_SCRATCHPAD/thought). Also strips
    standalone tool-call XML blocks some open models emit in content (ported from
    openclaw/openclaw#67318); the ``<function>`` variant is boundary- and
    ``name=``-gated so prose mentions survive.
    """
    if not content:
        return ""
    # Flatten list/dict content (e.g. Anthropic-via-OpenRouter block lists from
    # stored history) before regex: a raw list hits re.sub, raises TypeError,
    # and the conversation loop retries forever.
    if not isinstance(content, str):
        if isinstance(content, list):
            _parts: list[str] = []
            for _part in content:
                if isinstance(_part, str):
                    _parts.append(_part)
                elif isinstance(_part, dict):
                    _ptype = str(_part.get("type") or "").strip().lower()
                    # Drop thinking/reasoning blocks outright; their text key varies per provider.
                    if _ptype in {"thinking", "reasoning", "redacted_thinking"}:
                        continue
                    _text = _part.get("text")
                    if isinstance(_text, str) and _text:
                        _parts.append(_text)
            content = "".join(_parts)
        elif isinstance(content, dict):
            content = str(content.get("text") or content.get("content") or "")
        else:
            content = str(content)
        if not content:
            return ""
    # 1. Closed tag pairs, case-insensitive so mixed-case tags do not fall
    #    through to the unterminated pass and eat trailing content.
    for _pattern in _REASONING_BLOCK_PATTERNS:
        content = _pattern.sub('', content)
    # 1b. Tool-call XML blocks (openclaw/openclaw#67318); generic tags need no attribute gating.
    for _pattern in _TOOL_CALL_BLOCK_PATTERNS:
        content = _pattern.sub('', content)
    # 1c. Gemma-style <function name="..."> block: strip only at a block boundary
    #     AND with a name attribute so prose mentions of <function> survive.
    content = _NAMED_FUNCTION_BLOCK_PATTERN.sub('', content)
    # 2. Unterminated reasoning block at a block boundary: strip to end of
    #    string (#8878, #9568: MiniMax M2.7 leaking raw reasoning).
    content = _UNTERMINATED_REASONING_BLOCK_PATTERN.sub('', content)
    # 3. Stray orphan open/close tags that slipped through.
    content = _ORPHAN_REASONING_TAG_PATTERN.sub('', content)
    # 3b. Stray tool-call closers only; bare/unterminated <function> is kept since a
    #     truncated streaming tail may still be valuable (matches OpenClaw asymmetry).
    content = _STRAY_TOOL_CALL_CLOSER_PATTERN.sub('', content)
    return content



def sync_credential_pool_entry_id(agent) -> None:
    """Rebind ``agent._credential_pool_entry_id`` from the current pool + key.

    OAuth refreshes can replace the token before recovery runs, so the key
    alone cannot attribute a failure; the stable entry ID can. Cleared when no pool is bound.
    """
    pool = getattr(agent, "_credential_pool", None)
    try:
        agent._credential_pool_entry_id = (
            pool.entry_id_for_api_key(getattr(agent, "api_key", None))
            if pool is not None
            else None
        )
    except Exception:
        agent._credential_pool_entry_id = None


def recover_with_credential_pool(
    agent,
    *,
    status_code: Optional[int],
    has_retried_429: bool,
    classified_reason: Optional[FailoverReason] = None,
    error_context: Optional[Dict[str, Any]] = None,
    billing_unverified: bool = False,
) -> tuple[bool, bool]:
    """Attempt credential recovery via pool rotation.

    Returns (recovered, has_retried_429). Rate limits: retry once, then rotate.
    Billing exhaustion: rotate immediately. Auth failures: refresh before rotating.
    ``classified_reason`` honors the structured classifier over raw HTTP codes
    (e.g. Anthropic 400 for "out of extra usage"). ``billing_unverified`` (#82154)
    persists an ambiguous billing verdict so the entry gets a short cooldown, not
    the one-hour bench.
    """
    pool = agent._credential_pool
    if pool is None:
        return False, has_retried_429

    # The pool belongs to the PRIMARY provider: acting on fallback errors would
    # corrupt its state (#33088) and reset base_url to the primary endpoint (#33163).
    current_provider = (getattr(agent, "provider", "") or "").strip().lower()
    pool_provider = (getattr(pool, "provider", "") or "").strip().lower()
    # Skip recovery when the pool is scoped to another provider. Empty pool provider
    # means unscoped; empty agent provider is a mismatch (swap would leave provider="" model="").
    if pool_provider:
        # Same fail-closed boundary predicate as runtime binding (named-custom
        # aliases, endpoint validation, fallback isolation).
        if not credential_pool_matches_provider(
            pool,
            current_provider,
            base_url=getattr(agent, "base_url", None),
        ):
            _ra().logger.warning(
                "Credential pool provider mismatch: pool=%s, agent=%s — "
                "skipping pool mutation to avoid cross-provider contamination",
                pool_provider, current_provider,
            )
            return False, has_retried_429

    # Attribute the failure to the key actually dispatched, not pool.current():
    # the shared pointer often points at a different healthy entry, and marking
    # it exhausted can take the whole pool offline from one 429 (#43747).
    _api_key_hint = getattr(agent, "api_key", None) or None
    _raw_credential_id = getattr(agent, "_credential_pool_entry_id", None)
    _credential_id = (
        _raw_credential_id
        if isinstance(_raw_credential_id, str) and _raw_credential_id
        else None
    )
    if not _api_key_hint:
        _cur = pool.current()
        if _cur:
            _api_key_hint = getattr(_cur, "runtime_api_key", None)
            if not _credential_id:
                _current_id = getattr(_cur, "id", None)
                if isinstance(_current_id, str) and _current_id:
                    _credential_id = _current_id

    def _rotate_failed_credential(rotate_status: int):
        kwargs = {
            "status_code": rotate_status,
            "error_context": error_context,
            "api_key_hint": _api_key_hint,
        }
        if _credential_id:
            kwargs["credential_id"] = _credential_id
        # Pass classified semantics, not just the status: a billing 403 and an
        # edge-throttle 403 need opposite cooldowns. ``effective_reason`` is resolved below.
        if effective_reason is not None:
            _failure_reason = effective_reason.value
            if effective_reason == FailoverReason.billing and billing_unverified:
                # Ambiguous billing body (#82154): size the cooldown as transient, not a 1-hour bench.
                from agent.credential_pool import FAILURE_REASON_BILLING_UNVERIFIED
                _failure_reason = FAILURE_REASON_BILLING_UNVERIFIED
            kwargs["failure_reason"] = _failure_reason
        return pool.mark_exhausted_and_rotate(**kwargs)

    def _rotate_and_swap(default_status: int, label: str) -> bool:
        """Rotate away from the failed credential; True when a new entry was swapped in."""
        rotate_status = status_code if status_code is not None else default_status
        next_entry = _rotate_failed_credential(rotate_status)
        if next_entry is None:
            return False
        _ra().logger.info(
            "Credential %s (%s) — rotated to pool entry %s",
            rotate_status,
            label,
            getattr(next_entry, "id", "?"),
        )
        agent._swap_credential(next_entry)
        return True

    effective_reason = classified_reason
    if effective_reason is None:
        if status_code == 402:
            effective_reason = FailoverReason.billing
        elif status_code == 429:
            effective_reason = FailoverReason.rate_limit
        elif status_code in {401, 403}:
            effective_reason = FailoverReason.auth

    if effective_reason == FailoverReason.upstream_rate_limit:
        # Upstream (e.g. DeepSeek behind OpenRouter) is throttling the aggregator; the
        # credential is healthy. Do not rotate/exhaust; let fallback switch models.
        upstream = (error_context or {}).get("upstream_provider") if error_context else None
        if upstream:
            _ra().logger.info(
                "Upstream provider %s rate-limited via aggregator — skipping "
                "credential rotation, deferring to fallback chain",
                upstream,
            )
        else:
            _ra().logger.info(
                "Upstream aggregator 429 (provider unknown) — skipping "
                "credential rotation, deferring to fallback chain"
            )
        return False, has_retried_429

    if effective_reason == FailoverReason.billing:
        # A separate pool instance may have resolved runtime credentials, leaving
        # no ``current_id``; match the key that failed, not a different account.
        if _rotate_and_swap(402, "billing"):
            return True, False
        return False, has_retried_429

    if effective_reason == FailoverReason.rate_limit:
        # Already-exhausted credential: rotate immediately. Avoids the "cancel-between-429s"
        # trap where the local has_retried_429 resets per prompt and retries forever.
        current_entry = None
        if _credential_id:
            current_entry = next(
                (e for e in pool.entries() if e.id == _credential_id),
                None,
            )
        if _api_key_hint:
            current_entry = current_entry or next(
                (e for e in pool.entries() if e.runtime_api_key == _api_key_hint),
                None,
            )
        if current_entry is None:
            current_entry = pool.current()
        current_last_status = getattr(current_entry, "last_status", None) if current_entry else None
        if current_last_status == STATUS_EXHAUSTED:
            _ra().logger.info(
                "Credential already exhausted (last_status=%s) — rotating immediately instead of retrying",
                current_last_status,
            )
            if _rotate_and_swap(429, "rate limit, pre-exhausted"):
                return True, False
            return False, True

        usage_limit_reached = False
        if error_context:
            context_reason = str(error_context.get("reason") or "").lower()
            context_message = str(error_context.get("message") or "").lower()
            usage_limit_reached = (
                "usage_limit_reached" in context_reason
                or "gousagelimit" in context_reason
                or "usage limit reached" in context_message
                or "usage limit has been reached" in context_message
            )
        if not has_retried_429 and not usage_limit_reached:
            return False, True
        if _rotate_and_swap(429, "rate limit"):
            return True, False
        return False, True

    if effective_reason == FailoverReason.auth:
        # Entitlement 403s look like auth failures but refresh cannot fix them; any
        # xai-oauth 403 is treated as entitlement (#26847) EXCEPT xAI's stale-token
        # signals (``[WKE=unauthenticated:...]``, "could not be validated"), which must
        # stay refreshable (#29344).
        is_entitlement = agent._is_entitlement_failure(error_context, status_code)
        _auth_haystack = " ".join(
            str(error_context.get(k) or "").lower()
            for k in ("message", "reason", "code", "error")
            if isinstance(error_context, dict)
        )
        if (
            not is_entitlement
            and status_code == 403
            and "oauth authentication is currently not allowed for this organization" in _auth_haystack
        ):
            is_entitlement = True
        if (
            not is_entitlement
            and status_code == 403
            and (agent.provider or "") == "anthropic"
            and getattr(agent, "api_mode", "") == "anthropic_messages"
        ):
            is_entitlement = True
        if not is_entitlement and status_code == 403 and (agent.provider or "") == "xai-oauth":
            _is_xai_auth_failure = (
                "[wke=unauthenticated:" in _auth_haystack
                or "oauth2 access token could not be validated" in _auth_haystack
            )
            if not _is_xai_auth_failure:
                is_entitlement = True
        if is_entitlement:
            _ra().logger.info(
                "Credential %s — entitlement-shaped 403 from %s; "
                "skipping pool refresh (account lacks subscription, "
                "not a transient auth failure).",
                status_code if status_code is not None else "auth",
                agent.provider or "provider",
            )
            return False, has_retried_429
        # Refresh the entry that supplied the failing key, not current(): refreshing a
        # healthy entry burns its single-use refresh token for a failure it never had.
        refresh_kwargs = {"api_key_hint": _api_key_hint}
        if _credential_id:
            refresh_kwargs["credential_id"] = _credential_id
        refreshed = pool.try_refresh_matching(**refresh_kwargs)
        if refreshed is not None:
            # try_refresh_matching() reports success even when upstream keeps rejecting;
            # cap same-entry refreshes so a single-entry pool falls through to fallback (#26080).
            refreshed_id = getattr(refreshed, "id", None)
            if refreshed_id is not None:
                refresh_counts = getattr(agent, "_auth_pool_refresh_counts", None)
                if refresh_counts is None:
                    refresh_counts = {}
                    agent._auth_pool_refresh_counts = refresh_counts
                refresh_key = (agent.provider, refreshed_id)
                refresh_counts[refresh_key] = refresh_counts.get(refresh_key, 0) + 1
                if refresh_counts[refresh_key] > _MAX_AUTH_REFRESH_ATTEMPTS:
                    _ra().logger.warning(
                        "Credential auth failure persists after %s refreshes for "
                        "pool entry %s — treating as unrecoverable and allowing "
                        "fallback to activate.",
                        refresh_counts[refresh_key] - 1,
                        refreshed_id,
                    )
                    return False, has_retried_429
            _ra().logger.info("Credential auth failure — refreshed pool entry %s", getattr(refreshed, 'id', '?'))
            agent._swap_credential(refreshed)
            return True, has_retried_429
        # Refresh failed; rotate (the failed entry is already marked exhausted).
        if _rotate_and_swap(401, "auth refresh failed"):
            return True, False

    return False, has_retried_429



def _apply_primary_runtime_fields(agent, rt: Dict[str, Any]) -> None:
    """Copy the identity/transport fields of a ``_primary_runtime`` snapshot onto ``agent``.

    Shared by transport recovery and turn-start restore; the caller rebuilds the client.
    """
    agent.model = rt["model"]
    agent.provider = rt["provider"]
    agent.requested_provider = rt.get("requested_provider", agent.provider)
    agent.base_url = rt["base_url"]           # setter updates _base_url_lower
    agent.api_mode = rt["api_mode"]
    if hasattr(agent, "_transport_cache"):
        agent._transport_cache.clear()
    agent.api_key = rt["api_key"]
    agent._reasoning_echo_flag = rt.get("reasoning_echo_flag", False)
    agent.request_overrides = dict(rt.get("request_overrides") or {})
    agent._client_kwargs = dict(rt["client_kwargs"])


def _build_anthropic_client_from_runtime(agent, rt: Dict[str, Any]) -> None:
    """Rebuild the native Anthropic client from a ``_primary_runtime`` snapshot."""
    from agent.anthropic_adapter import build_anthropic_client
    agent._anthropic_api_key = rt["anthropic_api_key"]
    agent._anthropic_base_url = rt["anthropic_base_url"]
    agent._anthropic_client = build_anthropic_client(
        rt["anthropic_api_key"], rt["anthropic_base_url"],
        timeout=get_provider_request_timeout(agent.provider, agent.model),
    )
    agent._is_anthropic_oauth = rt["is_anthropic_oauth"]
    agent.client = None


def try_recover_primary_transport(
    agent, api_error: Exception, *, retry_count: int, max_retries: int,
) -> bool:
    """Rebuild the primary client once and retry after ``max_retries`` exhaust on a transient transport error.

    Skipped for aggregator providers (OpenRouter, Nous) that already manage pools/retries server-side.
    """
    if agent._fallback_activated:
        return False

    error_type = type(api_error).__name__
    if error_type not in _TRANSIENT_TRANSPORT_ERRORS:
        return False

    # Skip for aggregator providers — they manage their own retry infra
    if agent._is_openrouter_url():
        return False
    provider_lower = (agent.provider or "").strip().lower()
    # Portal OpenAI-wire traffic rides aggregator retry infra (skip), but Portal
    # Claude on native Messages holds a local Anthropic client that needs the rebuild.
    if (
        provider_lower in {"nous", "nous-portal", "nousresearch"}
        and getattr(agent, "api_mode", None) != "anthropic_messages"
    ):
        return False

    try:
        # Never hard-close the shared client here (#70773): stale streaming workers may
        # still be unwinding on the old pool; _retire_shared_openai_client defers FD release to GC.
        if getattr(agent, "client", None) is not None:
            try:
                agent._retire_shared_openai_client(
                    agent.client, reason="primary_recovery",
                )
            except Exception:
                pass

        rt = agent._primary_runtime
        _apply_primary_runtime_fields(agent, rt)

        if agent.api_mode == "anthropic_messages":
            _build_anthropic_client_from_runtime(agent, rt)
        elif (agent.provider or "").strip().lower() == "moa":
            # MoA has empty client_kwargs; rebuild via the shared facade factory so the
            # reference_callback relay survives recovery (#53802).
            from agent.moa_loop import build_moa_facade

            agent.client = build_moa_facade(agent, agent.model)
        else:
            agent.client = agent._create_openai_client(
                dict(rt["client_kwargs"]),
                reason="primary_recovery",
                shared=True,
            )

        wait_time = min(3 + retry_count, 8)
        agent._vprint(
            f"{agent.log_prefix}🔁 Transient {error_type} on {agent.provider} — "
            f"rebuilt client, waiting {wait_time}s before one last primary attempt.",
            force=True,
        )
        time.sleep(wait_time)
        return True
    except Exception as e:
        logger.warning("Primary transport recovery failed: %s", e)
        return False

# ── End provider fallback ──────────────────────────────────────────────



def drop_thinking_only_and_merge_users(
    messages: List[Dict[str, Any]],
    *,
    drop_codex_reasoning_items: bool = True,
) -> List[Dict[str, Any]]:
    """Drop thinking-only assistant turns and merge adjacent user messages left behind.

    Operates on the per-call ``api_messages`` copy only; ``agent.messages`` is never mutated.
    Drop-and-merge (not stub text) keeps history honest and preserves role alternation
    (mirrors Claude Code's ``normalizeMessagesForAPI``).
    """
    if not messages:
        return messages

    # Pass 1: drop thinking-only assistant turns.
    kept = [
        m for m in messages
        if not _ra().AIAgent._is_thinking_only_assistant(
            m,
            drop_codex_reasoning_items=drop_codex_reasoning_items,
        )
    ]
    dropped = len(messages) - len(kept)

    # Pass 2: merge any newly-adjacent user messages.
    merged: List[Dict[str, Any]] = []
    merges = 0
    for m in kept:
        prev = merged[-1] if merged else None
        if (
            prev is not None
            and prev.get("role") == "user"
            and m.get("role") == "user"
        ):
            prev_content = prev.get("content", "")
            cur_content = m.get("content", "")
            # Copy ``prev`` so caller dicts are never mutated (safe from tests/other loops).
            prev_copy = dict(prev)
            # Only string+string content merges; list (multimodal) sides append as separate blocks.
            if isinstance(prev_content, str) and isinstance(cur_content, str):
                sep = "\n\n" if prev_content and cur_content else ""
                prev_copy["content"] = prev_content + sep + cur_content
            elif isinstance(prev_content, list) and isinstance(cur_content, list):
                prev_copy["content"] = list(prev_content) + list(cur_content)
            elif isinstance(prev_content, list) and isinstance(cur_content, str):
                if cur_content:
                    prev_copy["content"] = list(prev_content) + [
                        {"type": "text", "text": cur_content}
                    ]
                else:
                    prev_copy["content"] = list(prev_content)
            elif isinstance(prev_content, str) and isinstance(cur_content, list):
                new_blocks: List[Dict[str, Any]] = []
                if prev_content:
                    new_blocks.append({"type": "text", "text": prev_content})
                new_blocks.extend(cur_content)
                prev_copy["content"] = new_blocks
            else:
                # Unknown content shape — fall back to appending separately
                # (violates alternation, but safer than raising in a hot path).
                merged.append(m)
                continue
            merged[-1] = prev_copy
            merges += 1
        else:
            merged.append(m)

    if dropped == 0 and merges == 0:
        return messages

    _ra().logger.debug(
        "Pre-call sanitizer: dropped %d thinking-only assistant turn(s), "
        "merged %d adjacent user message(s)",
        dropped,
        merges,
    )
    return merged



def restore_primary_runtime(agent) -> bool:
    """Restore the primary runtime at the start of a new turn so fallback stays turn-scoped.

    Needed for long-lived CLI agents and the gateway's cached agents (``_agent_cache``).
    """
    if not agent._fallback_activated:
        # Reset the index even without activation: a failed _try_activate_fallback() can strand
        # _fallback_index past the chain end and silently block future fallbacks (#20465).
        agent._fallback_index = 0
        return False

    if getattr(agent, "_rate_limited_until", 0) > time.monotonic():
        return False  # primary still in rate-limit cooldown, stay on fallback

    # Reset-aware gate: when the credential pool reports a reset time still in the future
    # (subscription windows), skip the guaranteed-to-fail restore (saves two cache invalidations
    # per turn). Fails open on any error/None. The loaded primary pool is handed to the
    # rebind block below via ``prefetched_primary_pool`` so it loads at most once.
    rt = agent._primary_runtime
    primary_provider = str((rt or {}).get("provider") or "").strip().lower()
    primary_runtime_base_url = str((rt or {}).get("base_url") or "")

    def _matches_primary(candidate) -> bool:
        return credential_pool_matches_provider(
            candidate, primary_provider, base_url=primary_runtime_base_url
        )

    def _load_primary_pool():
        """Load the primary provider's pool; None when absent or provider-mismatched."""
        from agent.credential_pool import load_pool

        key = resolve_runtime_pool_key(primary_provider, primary_runtime_base_url)
        loaded = load_pool(key) if key else None
        return loaded if loaded is not None and _matches_primary(loaded) else None

    prefetched_primary_pool = None
    primary_pool_prefetched = False
    try:
        pool = getattr(agent, "_credential_pool", None)
        if not _matches_primary(pool):
            prefetched_primary_pool = pool = _load_primary_pool()
            primary_pool_prefetched = True
        next_at = getattr(pool, "next_available_at", lambda: None)()
        if next_at is not None and next_at > time.time():
            if not getattr(agent, "_restore_wait_logged", False):
                agent._restore_wait_logged = True
                logger.info(
                    "Primary %s rate-limited until %s; staying on fallback "
                    "%s/%s until the reset elapses",
                    primary_provider or "?",
                    datetime.fromtimestamp(next_at).isoformat(timespec="seconds"),
                    agent.provider,
                    agent.model,
                )
            return False
    except Exception:
        logger.debug(
            "Reset-aware restore gate failed; falling back to per-turn retry",
            exc_info=True,
        )
    agent._restore_wait_logged = False

    fallback_route = getattr(agent, "_provider_fallback_route", None)
    if (
        isinstance(fallback_route, (list, tuple))
        and len(fallback_route) == 2
    ):
        previous_model = str(fallback_route[0] or "unknown")
        previous_provider = str(fallback_route[1] or "unknown")
    else:
        previous_model = str(getattr(agent, "model", "") or "unknown")
        previous_provider = str(getattr(agent, "provider", "") or "unknown")
    provider_fallback_active = bool(
        getattr(agent, "_provider_fallback_active", False)
    )
    try:
        # ── Core runtime state ──
        _apply_primary_runtime_fields(agent, rt)
        if "runtime_capabilities" in rt:
            raw_capabilities = rt["runtime_capabilities"]
            if not isinstance(raw_capabilities, dict):
                logger.warning("Ignoring malformed runtime capabilities snapshot")
            else:
                agent.runtime_capabilities = dict(raw_capabilities)
        elif "capabilities" in rt:
            # Read snapshots written by the initial capability propagation patch.
            raw_capabilities = rt["capabilities"]
            if isinstance(raw_capabilities, dict):
                agent.runtime_capabilities = dict(raw_capabilities)
        agent._use_prompt_caching = rt["use_prompt_caching"]
        # Default to native layout for snapshots predating the native-vs-proxy split.
        agent._use_native_cache_layout = rt.get(
            "use_native_cache_layout",
            agent.api_mode == "anthropic_messages" and agent.provider == "anthropic",
        )
        # An operator cache disable (_cache_disabled) must survive snapshot restoration (#33555).
        if getattr(agent, "_cache_disabled", False):
            agent._use_prompt_caching = False
            agent._use_native_cache_layout = False

        # ── Rebuild client for the primary provider ──
        if agent.provider == "moa":
            # MoA has no real OpenAI client kwargs; rebuild via the shared facade factory so the
            # reference_callback relay stays wired (#53802).
            from agent.moa_loop import build_moa_facade

            agent.client = build_moa_facade(agent, agent.model)
            agent._anthropic_client = None
        elif agent.api_mode == "anthropic_messages":
            _build_anthropic_client_from_runtime(agent, rt)
        else:
            agent.client = agent._create_openai_client(
                dict(rt["client_kwargs"]),
                reason="restore_primary",
                shared=True,
            )

        # ── Restore context engine state ──
        cc = agent.context_compressor
        cc.update_model(
            model=rt["compressor_model"],
            context_length=rt["compressor_context_length"],
            base_url=rt["compressor_base_url"],
            api_key=rt["compressor_api_key"],
            provider=rt["compressor_provider"],
            api_mode=rt.get("compressor_api_mode", ""),
        )

        # ── Rebind and re-select the primary credential pool ──
        # A cross-provider fallback attaches the fallback's pool; leaving it would trip the
        # provider-mismatch guard on the next 401/429. Reload the primary pool, else clear it.
        pool = getattr(agent, "_credential_pool", None)
        pool_provider = str(getattr(pool, "provider", "") or "").strip().lower()
        if pool is not None and pool_provider and not _matches_primary(pool):
            agent._credential_pool = None
            agent._credential_pool_entry_id = None
            try:
                # Reuse the pool the reset-aware gate already loaded (avoids a second auth.json read).
                agent._credential_pool = (
                    prefetched_primary_pool if primary_pool_prefetched else _load_primary_pool()
                )
            except Exception as exc:
                logger.warning(
                    "Restore could not reload primary credential pool for %s: %s",
                    primary_provider,
                    exc,
                )

        # The snapshot api_key may be stale after pool rotation; re-select the pool's current
        # best entry, keeping the snapshot key when no usable entry exists (#25205).
        agent._credential_pool_entry_id = None
        pool = getattr(agent, "_credential_pool", None)
        if pool is not None and pool.has_available():
            entry = pool.select()
            if entry is not None:
                entry_provider = str(getattr(entry, "provider", "") or "").strip().lower()
                entry_key = (
                    getattr(entry, "runtime_api_key", None)
                    or getattr(entry, "access_token", "")
                )
                if entry_key and _matches_primary(entry):
                    # _swap_credential rebuilds the client and reapplies base-url-scoped headers (#33163).
                    agent._swap_credential(entry)
                    logger.info(
                        "Restore re-selected pool entry %s (%s)",
                        getattr(entry, "id", "?"),
                        getattr(entry, "label", "?"),
                    )
                elif entry_key:
                    logger.info(
                        "Restore skipped pool entry %s (%s): provider %s does not match primary provider %s",
                        getattr(entry, "id", "?"),
                        getattr(entry, "label", "?"),
                        entry_provider or "?",
                        primary_provider or "?",
                    )

        # ── Restore reasoning_config if saved (older snapshots keep the current value) ──
        saved_reasoning = rt.get("reasoning_config")
        if saved_reasoning is not None:
            agent.reasoning_config = dict(saved_reasoning)

        # ── Reset fallback chain for the new turn ──
        agent._fallback_activated = False
        agent._fallback_index = 0
        agent._rate_limit_backoff_count = 0  # reset exponential backoff counter

        # Reset the stale-call circuit breaker (#58962): its streak measured the fallback provider.
        from agent.chat_completion_helpers import _reset_stale_streak
        _reset_stale_streak(agent)

        # Undo the fallback's identity rewrite so the prompt is
        # byte-identical to the stored copy again (prefix cache match).
        from agent.chat_completion_helpers import rewrite_prompt_model_identity
        rewrite_prompt_model_identity(agent, rt["model"], rt["provider"])

        logger.info(
            "Primary runtime restored for new turn: %s (%s)",
            agent.model, agent.provider,
        )
        agent._provider_fallback_active = False
        agent._provider_fallback_route = None
        if provider_fallback_active:
            try:
                agent._emit_status(
                    f"✅ Primary model restored: {agent.model} via {agent.provider}; "
                    f"fallback {previous_model} via {previous_provider} is no longer active."
                )
            except Exception:
                # Notification surfaces are best-effort and must never undo a
                # successful runtime restoration.
                pass
        return True
    except Exception as e:
        logger.warning("Failed to restore primary runtime: %s", e)
        return False

# Which error types indicate a transient transport failure worth
# one more attempt with a rebuilt client / connection pool.
_TRANSIENT_TRANSPORT_ERRORS = frozenset({
    "ReadTimeout", "ConnectTimeout", "PoolTimeout",
    "ConnectError", "RemoteProtocolError",
    "APIConnectionError", "APITimeoutError",
})



def extract_reasoning(agent, assistant_message) -> Optional[str]:
    """Extract reasoning text from an assistant message, or None.

    Checks ``reasoning``, ``reasoning_content``, ``reasoning_details`` (OpenRouter unified),
    then inline thinking blocks in list content.
    """
    reasoning_parts = []
    
    if hasattr(assistant_message, 'reasoning') and assistant_message.reasoning:
        reasoning_parts.append(assistant_message.reasoning)
    
    if hasattr(assistant_message, 'reasoning_content') and assistant_message.reasoning_content:
        if assistant_message.reasoning_content not in reasoning_parts:
            reasoning_parts.append(assistant_message.reasoning_content)
    
    # reasoning_details: [{"type": "reasoning.summary", "summary": "...", ...}, ...]
    if hasattr(assistant_message, 'reasoning_details') and assistant_message.reasoning_details:
        for detail in assistant_message.reasoning_details:
            if isinstance(detail, dict):
                summary = (
                    detail.get('summary')
                    or detail.get('thinking')
                    or detail.get('content')
                    or detail.get('text')
                )
                if summary and summary not in reasoning_parts:
                    reasoning_parts.append(summary)

    # Fall back to reasoning embedded in content only when no structured field was found.
    content = getattr(assistant_message, "content", None)
    if not reasoning_parts and isinstance(content, list):
        # DeepSeek V4 Pro returns typed content blocks ({"type": "thinking", ...}); dropping
        # them makes the next turn fail with HTTP 400 "thinking must be passed back" (#21944).
        for block in content:
            if isinstance(block, dict) and block.get("type") == "thinking":
                thinking_text = block.get("thinking") or block.get("text") or ""
                thinking_text = thinking_text.strip()
                if thinking_text and thinking_text not in reasoning_parts:
                    reasoning_parts.append(thinking_text)
    if not reasoning_parts and isinstance(content, str) and content:
        inline_patterns = (
            r"<think>(.*?)</think>",
            r"<thinking>(.*?)</thinking>",
            r"<thought>(.*?)</thought>",
            r"<reasoning>(.*?)</reasoning>",
            r"<REASONING_SCRATCHPAD>(.*?)</REASONING_SCRATCHPAD>",
        )
        for pattern in inline_patterns:
            flags = re.DOTALL | re.IGNORECASE
            for block in re.findall(pattern, content, flags=flags):
                cleaned = block.strip()
                if cleaned and cleaned not in reasoning_parts:
                    reasoning_parts.append(cleaned)
    
    if reasoning_parts:
        return "\n\n".join(reasoning_parts)
    
    return None



def dump_api_request_debug(
    agent,
    api_kwargs: Dict[str, Any],
    *,
    reason: str,
    error: Optional[Exception] = None,
) -> Optional[Path]:
    """Dump the request body from api_kwargs (minus transport keys) for debugging provider 4xx failures."""
    try:
        body = copy.deepcopy(api_kwargs)
        body.pop("timeout", None)
        body = {k: v for k, v in body.items() if v is not None}

        api_key = None
        try:
            api_key = getattr(agent.client, "api_key", None)
        except Exception as e:
            _ra().logger.debug("Could not extract API key for debug dump: %s", e)

        dump_payload: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "session_id": agent.session_id,
            "reason": reason,
            "request": {
                "method": "POST",
                "url": f"{agent.base_url.rstrip('/')}{'/responses' if agent.api_mode == 'codex_responses' else '/chat/completions'}",
                "headers": {
                    "Authorization": f"Bearer {agent._mask_api_key_for_logs(api_key)}",
                    "Content-Type": "application/json",
                },
                "body": body,
            },
        }

        if error is not None:
            error_info: Dict[str, Any] = {
                "type": type(error).__name__,
                "message": str(error),
            }
            for attr_name in ("status_code", "request_id", "code", "param", "type"):
                attr_value = getattr(error, attr_name, None)
                if attr_value is not None:
                    error_info[attr_name] = attr_value

            body_attr = getattr(error, "body", None)
            if body_attr is not None:
                error_info["body"] = body_attr

            response_obj = getattr(error, "response", None)
            if response_obj is not None:
                try:
                    error_info["response_status"] = getattr(response_obj, "status_code", None)
                    error_info["response_text"] = response_obj.text
                except Exception as e:
                    _ra().logger.debug("Could not extract error response details: %s", e)

            dump_payload["error"] = error_info

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        # Sanitize the session ID (may come from an untrusted X-Hermes-Session-Id header)
        # so a "../"-shaped ID cannot write outside logs_dir.
        safe_sid = _ra()._safe_session_filename_component(agent.session_id)
        dump_file = agent.logs_dir / f"request_dump_{safe_sid}_{timestamp}.json"

        # Redact secrets first: this fires unconditionally on API errors and captures the full
        # request body, so context-embedded secrets would otherwise land in cleartext on disk.
        from agent.redact import redact_sensitive_text
        _serialized = json.dumps(dump_payload, ensure_ascii=False, indent=2, default=str)
        _redacted_payload = json.loads(redact_sensitive_text(_serialized, force=True))
        atomic_json_write(dump_file, _redacted_payload, default=str)

        agent._vprint(f"{agent.log_prefix}🧾 Request debug dump written to: {dump_file}")

        if env_var_enabled("HERMES_DUMP_REQUEST_STDOUT"):
            print(json.dumps(_redacted_payload, ensure_ascii=False, indent=2, default=str))

        return dump_file
    except Exception as dump_error:
        if agent.verbose_logging:
            logger.warning("Failed to dump API request debug payload: %s", dump_error)
        return None



def _direct_native_anthropic_tool_cache_capability(
    agent,
    *,
    provider: Optional[str] = None,
    base_url: Optional[str] = None,
    api_mode: Optional[str] = None,
    model: Optional[str] = None,
) -> bool:
    """Return whether this resolved destination accepts native tool markers."""
    eff_base_url = base_url if base_url is not None else (agent.base_url or "")
    eff_api_mode = api_mode if api_mode is not None else (agent.api_mode or "")
    return (
        eff_api_mode == "anthropic_messages"
        and base_url_hostname(eff_base_url) == "api.anthropic.com"
    )


def cache_ttl_means_disabled(ttl: Any) -> bool:
    """Return True when a ``prompt_caching.cache_ttl`` value means caching off.

    Single predicate shared by ``agent_init`` and the stub policy paths (#76085).
    Unknown values (``"2h"``, integers) are NOT a disable.
    """
    if ttl in ("5m", "1h"):
        return False
    if ttl is False or ttl is None:
        return True
    return str(ttl).lower() in ("off", "false", "disabled", "no", "none")


# The cache_ttl tiers accepted by config; mirrored by agent_init's live-agent snapshot.
VALID_CACHE_TTLS = ("5m", "1h")


def _raw_cache_ttl_from_config() -> Any:
    """Read the raw ``prompt_caching.cache_ttl`` config value (may raise)."""
    from hermes_cli.config import load_config_readonly

    pc_cfg = load_config_readonly().get("prompt_caching", {}) or {}
    return pc_cfg.get("cache_ttl", "5m")


def prompt_caching_disabled_from_config() -> bool:
    """Return True when ``prompt_caching.cache_ttl`` is configured as off (same detection as ``agent_init``; #76085 / #33555)."""
    try:
        ttl = _raw_cache_ttl_from_config()
    except Exception:
        return False
    return cache_ttl_means_disabled(ttl)


def configured_cache_ttl() -> Optional[str]:
    """Return the configured ``prompt_caching.cache_ttl`` tier (``5m``/``1h``), else None.

    Mirrors ``agent_init`` so stub paths don't regress a configured ``1h`` to 5m (#84733).
    """
    try:
        ttl = _raw_cache_ttl_from_config()
    except Exception:
        return None
    return ttl if ttl in VALID_CACHE_TTLS else None


def blank_cache_policy_stub(cache_disabled: Optional[bool] = None):
    """Build the destination-identity-blank stub for ``anthropic_prompt_cache_policy``.

    Sole sanctioned constructor so ``_cache_disabled`` is never omitted (#76085); when
    ``cache_disabled`` is None the global config is consulted.
    """
    from types import SimpleNamespace

    if cache_disabled is None:
        cache_disabled = prompt_caching_disabled_from_config()
    return SimpleNamespace(
        provider="",
        base_url="",
        api_mode="",
        model="",
        _cache_disabled=bool(cache_disabled),
    )


def plan_cache_sections_for_destination(
    messages: list,
    tools: Optional[list],
    *,
    provider: str,
    base_url: str,
    api_mode: str,
    model: str,
    cache_disabled: Optional[bool] = None,
    cache_ttl: Optional[str] = None,
    static_system_prefix: Optional[str] = None,
) -> Tuple[list, list]:
    """Plan request-local cache sections for one resolved destination (MoA / auxiliary senders).

    Returns stripped copies (non-caching route) or a ``build_prompt_cache_plan`` layout; never
    mutates ``messages``/``tools``. ``cache_disabled`` and ``cache_ttl`` default to live config
    so these paths honor the operator's disable (#76085) and tier (#84733);
    ``static_system_prefix`` gives the system prompt the same early breakpoint as the main loop.
    """
    from agent.prompt_caching import (
        build_prompt_cache_plan,
        effective_cache_ttl,
        envelope_tool_part_cache_markers_supported,
        strip_anthropic_cache_control,
        strip_anthropic_tool_cache_control,
    )

    stub = blank_cache_policy_stub(cache_disabled)
    should_cache, native_layout = anthropic_prompt_cache_policy(
        stub,
        provider=provider,
        base_url=base_url,
        api_mode=api_mode,
        model=model,
    )
    if not should_cache:
        canonical_messages = copy.deepcopy(messages or [])
        strip_anthropic_cache_control(canonical_messages)
        return canonical_messages, strip_anthropic_tool_cache_control(tools)
    plan = build_prompt_cache_plan(
        messages,
        tools,
        cache_ttl=effective_cache_ttl(
            # effective_cache_ttl resolves None → "5m"; cache-disabled agents never reach here.
            cache_ttl,
            provider=provider,
            model=model,
        ),
        native_anthropic=native_layout,
        static_system_prefix=(
            static_system_prefix if isinstance(static_system_prefix, str) else None
        ),
        direct_native_tool_cache=_direct_native_anthropic_tool_cache_capability(
            stub,
            provider=provider,
            base_url=base_url,
            api_mode=api_mode,
            model=model,
        ),
        # LiteLLM-style envelope routes forward part-level markers into
        # tool_result.content[] → non-retryable 400 (#89886).
        tool_part_markers=envelope_tool_part_cache_markers_supported(
            provider, base_url
        ),
    )
    return plan.messages, plan.tools


def _is_litellm_route(provider_lower: str, base_url: str) -> bool:
    """True when a route is a LiteLLM proxy, by provider id or host token.

    ``litellm`` must match as a whole delimited token (not substring) in provider or host;
    a path segment never qualifies.
    """
    if _has_litellm_token(provider_lower, ":-_/"):
        return True
    return _has_litellm_token(base_url_hostname(base_url), ".-")


def _has_litellm_token(value: str, delimiters: str) -> bool:
    """True when ``value`` contains ``litellm`` as a whole delimited token."""
    if not value:
        return False
    for delimiter in delimiters:
        value = value.replace(delimiter, " ")
    return "litellm" in value.split()


def anthropic_prompt_cache_policy(
    agent,
    *,
    provider: Optional[str] = None,
    base_url: Optional[str] = None,
    api_mode: Optional[str] = None,
    model: Optional[str] = None,
) -> tuple[bool, bool]:
    """Decide whether to apply Anthropic prompt caching; returns ``(should_cache, use_native_layout)``.

    ``use_native_layout`` puts markers on inner content blocks (native Anthropic wire);
    otherwise on the message envelope (OpenRouter / OpenAI-wire proxies). Qwen/Alibaba routes
    also honour envelope markers (pi-mono #3392). An operator disable is read from
    ``_cache_disabled`` (not ``_cache_ttl``, unset during init) so it survives switches and
    restores (#33555).
    """
    if getattr(agent, "_cache_disabled", False):
        return (False, False)

    eff_provider = (provider if provider is not None else agent.provider) or ""
    eff_base_url = base_url if base_url is not None else (agent.base_url or "")
    eff_api_mode = api_mode if api_mode is not None else (agent.api_mode or "")
    eff_model = (model if model is not None else agent.model) or ""

    # MoA virtual provider matches no caching branch, silently losing caching for the acting
    # aggregator; resolve the policy from the preset's real aggregator slot instead.
    if eff_provider.strip().lower() == "moa":
        try:
            from hermes_cli.config import load_config as _load_moa_cfg
            from hermes_cli.moa_config import resolve_moa_preset
            from hermes_cli.runtime_provider import resolve_runtime_provider

            _preset = resolve_moa_preset(
                _load_moa_cfg().get("moa") or {}, eff_model or None
            )
            _agg = _preset.get("aggregator") or {}
            _agg_provider = str(_agg.get("provider") or "").strip()
            _agg_model = str(_agg.get("model") or "").strip()
            if _agg_provider and _agg_model:
                _agg_base_url = ""
                _agg_api_mode = ""
                try:
                    _rt = resolve_runtime_provider(
                        requested=_agg_provider, target_model=_agg_model
                    )
                    _agg_base_url = _rt.get("base_url") or ""
                    _agg_api_mode = _rt.get("api_mode") or ""
                except Exception:
                    pass
                return anthropic_prompt_cache_policy(
                    agent,
                    provider=_agg_provider,
                    base_url=_agg_base_url,
                    api_mode=_agg_api_mode,
                    model=_agg_model,
                )
        except Exception as _moa_exc:  # pragma: no cover - defensive
            logger.debug("MoA aggregator cache-policy resolution failed: %s", _moa_exc)
        return False, False

    if isinstance(eff_model, dict):
        eff_model = eff_model.get('model') or eff_model.get('default') or ''
    eff_model = eff_model if isinstance(eff_model, str) else str(eff_model or '')
    model_lower = eff_model.lower()
    provider_lower = eff_provider.lower()
    is_claude = "claude" in model_lower
    # Kimi/Moonshot via OpenRouter uses the same envelope cache_control as Claude; without
    # this branch it serves ~1% cache hits (#25970). Family matcher covers bare k1./k2. slugs.
    from agent.anthropic_adapter import _model_name_is_kimi_family
    is_kimi = (
        _model_name_is_kimi_family(eff_model) or "moonshot" in model_lower
    )
    is_openrouter = base_url_host_matches(eff_base_url, "openrouter.ai")
    # Nous Portal proxies to OpenRouter; treat as OpenRouter-equivalent for cache layout.
    is_nous_portal = base_url_host_matches(eff_base_url, "nousresearch.com")
    is_anthropic_wire = eff_api_mode == "anthropic_messages"
    is_native_anthropic = (
        is_anthropic_wire
        and (eff_provider == "anthropic" or base_url_hostname(eff_base_url) == "api.anthropic.com")
    )

    # Honor a configured route's per-model ``prompt_caching`` capability (explicit false too);
    # only for the two transports this planner handles, not Responses/Bedrock.
    custom_prompt_caching = None
    _supports_anthropic_cache_markers = eff_api_mode in {
        "anthropic_messages",
        "chat_completions",
    }
    _litellm_openai_wire = (
        eff_api_mode == "chat_completions"
        and is_claude
        and _is_litellm_route(provider_lower, eff_base_url)
    )
    _custom_providers = getattr(agent, "_custom_providers", None)
    _route_may_be_custom = False
    if not _supports_anthropic_cache_markers:
        # Responses/Bedrock never consume the declaration — skip the
        # identity probe entirely for those transports.
        pass
    elif _custom_providers:
        # Cheap identity gate before the capability helper, matching its semantics
        # (normalize_route_base_url + custom_provider_aliases) so spelling differences don't drop declarations.
        from hermes_cli.providers import custom_provider_aliases
        from hermes_cli.route_identity import normalize_route_base_url

        _provider_ids = {provider_lower}
        if provider_lower.startswith("custom:"):
            _provider_ids.add(provider_lower.removeprefix("custom:"))
        _eff_url_normalized = normalize_route_base_url(eff_base_url)
        for _entry in _custom_providers:
            if not isinstance(_entry, dict):
                continue
            _entry_ids = custom_provider_aliases(
                str(_entry.get("name") or ""),
                str(_entry.get("provider_key") or ""),
            )
            if _provider_ids & _entry_ids or (
                _eff_url_normalized
                and normalize_route_base_url(_entry.get("base_url"))
                == _eff_url_normalized
            ):
                _route_may_be_custom = True
                break
    elif _custom_providers is None:
        # None = list not attached yet (early init or blank stub); an attached empty list never
        # matches. Avoid rebuilding the list for ordinary built-in routes.
        try:
            from hermes_cli.providers import get_provider

            # allow_network=False: never trigger a registry fetch from the send path;
            # a catalog miss degrades to the conservative capability lookup.
            _provider_def = get_provider(eff_provider, allow_network=False)
            _route_may_be_custom = _provider_def is None or (
                bool(_provider_def.base_url)
                and base_url_hostname(_provider_def.base_url)
                != base_url_hostname(eff_base_url)
            )
        except Exception as _pd_exc:
            logger.debug(
                "provider lookup failed during cache-policy pre-gate: %s",
                _pd_exc,
            )
            _route_may_be_custom = provider_lower.startswith("custom:")

    if _supports_anthropic_cache_markers and (
        is_anthropic_wire or _litellm_openai_wire or _route_may_be_custom
    ):
        try:
            from hermes_cli.config import get_custom_provider_model_capability

            custom_prompt_caching = get_custom_provider_model_capability(
                model=eff_model,
                base_url=eff_base_url,
                capability="prompt_caching",
                custom_providers=_custom_providers,
            )
        except Exception as _cap_exc:
            logger.debug(
                "custom-provider prompt_caching capability lookup failed: %s",
                _cap_exc,
            )
    if custom_prompt_caching is not None:
        # Layout follows the transport: native Messages → inner blocks; OpenAI wire → envelope.
        return custom_prompt_caching, custom_prompt_caching and is_anthropic_wire

    # MiniMax-M3 uses server-side automatic prefix caching; explicit markers are dead weight.
    # Checked BEFORE the native-Anthropic return since provider="anthropic" may point at a MiniMax proxy.
    is_minimax_provider = provider_lower in {"minimax", "minimax-cn"}
    is_minimax_host = (
        base_url_host_matches(eff_base_url, "api.minimax.io")
        or base_url_host_matches(eff_base_url, "api.minimaxi.com")
    )
    is_minimax_route = is_minimax_provider or is_minimax_host
    if is_anthropic_wire and is_minimax_route:
        from agent.model_metadata import _model_name_suggests_minimax_m3

        if _model_name_suggests_minimax_m3(eff_model):
            return False, False

    if is_native_anthropic:
        return True, True
    # Envelope layout is OpenAI-wire only; Portal Claude on native Messages must fall through
    # to the anthropic_messages branch (inner-block markers) or it serves 0% cache hits.
    if (
        (is_openrouter or is_nous_portal)
        and (is_claude or is_kimi)
        and not is_anthropic_wire
    ):
        return True, False
    # Nous Portal Qwen takes the envelope path too; the alibaba-family check below only matches
    # provider=opencode/alibaba and would leave Portal traffic uncached.
    if is_nous_portal and "qwen" in model_lower:
        return True, False
    if is_anthropic_wire and is_claude:
        # Third-party Anthropic-compatible gateway.
        return True, True

    # LiteLLM fronting Claude on the OpenAI-compatible wire supports cache_control but matched
    # no grant branch above (#84506). Claude-only: strict relays reject the block format for
    # other models (#77217). Envelope layout: the native layout's top-level markers are only
    # relocated by the anthropic_messages adapter and cause HTTP 400 via LiteLLM (#69512).
    # Gated on chat_completions explicitly; codex_responses/bedrock_converse have their own handling.
    if _litellm_openai_wire:
        return True, False

    # MiniMax's own models (M2.x) on its Anthropic-compatible endpoint support cache_control;
    # opt them in past the is_claude gate. M3 is excluded above.
    if is_anthropic_wire and is_minimax_route:
        return True, True

    # Qwen/Alibaba on OpenCode and DashScope accept envelope cache_control on the OpenAI wire.
    # DeepSeek on OpenCode is excluded: its relay 400s on block-array content (#77217).
    # Family set/predicate shared with the effective_cache_ttl clamp (#84733).
    from agent.prompt_caching import ALIBABA_FAMILY_PROVIDERS, is_qwen_model

    model_is_qwen = is_qwen_model(model_lower)
    provider_is_alibaba_family = provider_lower in ALIBABA_FAMILY_PROVIDERS
    if provider_is_alibaba_family and model_is_qwen:
        # Envelope layout (native_anthropic=False), matching pi-mono's "alibaba" cacheControlFormat.
        return True, False

    return False, False



def _provider_supplied_client(agent, client_kwargs: dict) -> Any | None:
    """Ask the registered ProviderProfile for a custom client, if any.

    Resolves by provider name first, then by the ``base_url`` scheme prefix so a
    runtime configured only by URL (``acp://…``) still reaches its profile.
    A profile that raises is logged and skipped: a third-party plugin must not
    be able to take the turn down, it can only fail to provide a client.
    """
    try:
        from providers import get_provider_profile
    except Exception:
        return None

    profile = None
    provider_name = (getattr(agent, "provider", "") or "").strip()
    if provider_name:
        try:
            profile = get_provider_profile(provider_name)
        except Exception:
            profile = None
    if profile is None:
        base_url = str(client_kwargs.get("base_url", "") or "").strip()
        if base_url:
            profile = _profile_for_base_url(base_url)
    if profile is None:
        return None

    try:
        return profile.create_client(**client_kwargs)
    except Exception:
        _ra().logger.warning(
            "Provider profile %r failed to create a client; falling back to the "
            "standard client path",
            getattr(profile, "name", provider_name) or "?",
            exc_info=True,
        )
        return None


def _profile_for_base_url(base_url: str) -> Any | None:
    """Find a registered profile whose own base_url matches ``base_url``.

    Only used when the provider name did not resolve. Matches on exact base_url
    so a non-HTTP scheme (``acp://copilot``) routes to its profile even when the
    caller passed no provider name.
    """
    try:
        from providers import list_providers
    except Exception:
        return None
    target = base_url.rstrip("/").lower()
    try:
        candidates = list_providers()
    except Exception:
        return None
    for candidate in candidates or []:
        own = str(getattr(candidate, "base_url", "") or "").rstrip("/").lower()
        # Prefix match, not equality: the replaced copilot-acp branch keyed on
        # ``startswith("acp://copilot")``, so a base_url carrying a path or a
        # user override under the same root must still resolve.
        if own and (target == own or target.startswith(own + "/")):
            return candidate
    return None


def create_openai_client(agent, client_kwargs: dict, *, reason: str, shared: bool) -> Any:
    from agent.auxiliary_client import _validate_base_url, _validate_proxy_env_urls
    from agent.ssl_verify import resolve_httpx_verify
    # Treat client_kwargs as read-only: callers pass agent._client_kwargs, and in-place
    # mutation leaks into later requests (#10933: a torn-down httpx transport got reused).
    client_kwargs = dict(client_kwargs)
    # The MoA virtual provider has no OpenAI wire endpoint; the facade *is* the client.
    # Rebuild the facade, never a native client (#78382 TypeError, #53802 relay re-wire).
    if (getattr(agent, "provider", "") or "").strip().lower() == "moa":
        from agent.moa_loop import build_moa_facade
        return build_moa_facade(agent, getattr(agent, "model", None) or "default")
    ssl_ca_cert = client_kwargs.pop("ssl_ca_cert", None)
    ssl_verify_cfg = client_kwargs.pop("ssl_verify", None)
    httpx_verify = resolve_httpx_verify(ca_bundle=ssl_ca_cert, ssl_verify=ssl_verify_cfg)
    _validate_proxy_env_urls()
    _validate_base_url(client_kwargs.get("base_url"))
    # ── Provider-supplied client (registration seam) ──────────────────────
    # A provider whose wire protocol is not OpenAI-over-HTTP supplies its own
    # client from its ProviderProfile.create_client(). Consulted before the
    # built-in ladder so a profile registered from ~/.hermes/plugins/ or a pip
    # entry point can ship a transport without editing this function — that is
    # what makes an out-of-tree ACP provider possible at all. Returning None
    # (the default) falls through to the paths below, so every existing
    # provider is unaffected.
    provider_client = _provider_supplied_client(agent, client_kwargs)
    if provider_client is not None:
        _ra().logger.info(
            "%s client created from provider profile (%s, shared=%s) %s",
            agent.provider,
            reason,
            shared,
            agent._client_log_context(),
        )
        return provider_client
    if agent.provider == "gemini":
        from agent.gemini_native_adapter import GeminiNativeClient, is_native_gemini_base_url

        base_url = str(client_kwargs.get("base_url", "") or "")
        if is_native_gemini_base_url(base_url):
            safe_kwargs = {
                k: v for k, v in client_kwargs.items()
                if k in {"api_key", "base_url", "default_headers", "timeout", "http_client"}
            }
            if "http_client" not in safe_kwargs:
                keepalive_http = agent._build_keepalive_http_client(
                    base_url, verify=httpx_verify,
                )
                if keepalive_http is not None:
                    safe_kwargs["http_client"] = keepalive_http
            client = GeminiNativeClient(**safe_kwargs)
            _ra().logger.info(
                "Gemini native client created (%s, shared=%s) %s",
                reason,
                shared,
                agent._client_log_context(),
            )
            return client
    # TCP keepalives so dead provider connections are detected (~60s) instead of hanging in
    # CLOSE-WAIT (#10324). Injected into the local copy only (#10933), so each client gets its
    # own httpx.Client; pinned by tests/run_agent/test_create_openai_client_reuse.py and
    # tests/run_agent/test_sequential_chats_live.py.
    if "http_client" not in client_kwargs:
        keepalive_http = agent._build_keepalive_http_client(
            client_kwargs.get("base_url", ""), verify=httpx_verify,
        )
        if keepalive_http is not None:
            client_kwargs["http_client"] = keepalive_http
    # Retries belong to the outer conversation loop (honors Retry-After); SDK retries would
    # double-retry inside it (#26293). auxiliary_client keeps SDK retries as it isn't wrapped.
    client_kwargs.setdefault("max_retries", 0)
    # Defense-in-depth: primary_recovery/restore_primary rebuild from a _primary_runtime
    # snapshot without re-running header wiring; missing Copilot-Integration-Id causes
    # model_not_available_for_integrator 400s. Only ADD missing keys, never override.
    try:
        if base_url_host_matches(str(client_kwargs.get("base_url", "")), "githubcopilot.com"):
            from hermes_cli.models import copilot_default_headers
            existing = dict(client_kwargs.get("default_headers") or {})
            existing_lower = {k.lower() for k in existing}
            for hk, hv in copilot_default_headers().items():
                if hk.lower() not in existing_lower:
                    existing[hk] = hv
            client_kwargs["default_headers"] = existing
    except Exception:
        _ra().logger.debug("Copilot default-header guard skipped", exc_info=True)
    # OpenCode Free is served anonymously: any unrecognized bearer is a 401, so an empty
    # Authorization default_header overrides the SDK's "Bearer <api_key>".
    if agent.provider == "opencode-free":
        from hermes_cli.models import opencode_zen_free_headers

        _existing = dict(client_kwargs.get("default_headers") or {})
        _existing.update(opencode_zen_free_headers())
        client_kwargs["default_headers"] = _existing

    # All primary construction and recovery paths must identify Hermes to the
    # official Codex endpoint, including snapshots with custom header overrides.
    from agent.codex_headers import apply_required_codex_headers

    apply_required_codex_headers(
        client_kwargs,
        access_token=client_kwargs.get("api_key", ""),
        base_url=str(client_kwargs.get("base_url", "")),
    )
    # Module-level `OpenAI` is resolved lazily via __getattr__; tests patch `run_agent.OpenAI`.
    client = _ra().OpenAI(**client_kwargs)
    _ra().logger.info(
        "OpenAI client created (%s, shared=%s) %s",
        reason,
        shared,
        agent._client_log_context(),
    )
    return client


def _apply_switched_provider_request_overrides(agent, new_provider):
    """Re-derive the switched-to provider's ``request_overrides`` (custom_providers ``extra_body``) onto a live agent.

    Matches by provider key, base_url AND model (same rule as
    ``agent_init._merge_custom_provider_extra_body``) so a different model at the
    same endpoint never inherits another model's ``extra_body``. Stale
    ``extra_body`` is cleared; ``service_tier`` / ``speed`` overrides are preserved.
    """
    from agent.agent_init import _custom_provider_extra_body_for_agent

    # Prefer the init-time cache (agent._custom_providers); reload only if absent.
    custom_providers = getattr(agent, "_custom_providers", None)
    if custom_providers is None:
        try:
            from hermes_cli.config import load_config, get_compatible_custom_providers
            custom_providers = get_compatible_custom_providers(load_config())
        except Exception:
            custom_providers = []

    new_extra_body = _custom_provider_extra_body_for_agent(
        provider=new_provider,
        model=getattr(agent, "model", "") or "",
        base_url=getattr(agent, "base_url", "") or "",
        custom_providers=custom_providers or [],
    )

    overrides = dict(getattr(agent, "request_overrides", {}) or {})
    overrides.pop("extra_body", None)  # always drop the previous provider's extra_body
    if new_extra_body:
        overrides["extra_body"] = dict(new_extra_body)
    agent.request_overrides = overrides


def switch_model(
    agent,
    new_model,
    new_provider,
    api_key='',
    base_url='',
    api_mode='',
    capabilities=None,
):
    """Switch the model/provider in-place for a live agent (rebuild clients, caching flags, compressor).

    Mirrors ``_try_activate_fallback()`` but also updates ``_primary_runtime`` so
    the change persists across turns.
    """
    from hermes_cli.providers import determine_api_mode
    from agent.native_compaction import resolve_native_compaction_capabilities

    old_model = agent.model
    old_provider = agent.provider
    old_norm = (old_provider or "").strip().lower()
    new_norm = (new_provider or "").strip().lower()

    # Pass model so dual-wire providers (Nous Portal anthropic/* -> Messages) resolve correctly.
    if not api_mode:
        api_mode = determine_api_mode(new_provider, base_url, model=new_model)

    if not base_url and new_norm == "openai":
        # An omitted URL means the provider's canonical direct endpoint.
        base_url = "https://api.openai.com/v1"

    # Same-provider switches may omit base_url (e.g. credential refresh); resolve
    # capabilities from the endpoint the normalization below retains.
    effective_base_url = base_url
    if not effective_base_url and old_norm == new_norm:
        effective_base_url = getattr(agent, "base_url", "")

    destination_capabilities = (
        dict(capabilities)
        if isinstance(capabilities, dict)
        else resolve_native_compaction_capabilities(
            model=new_model,
            base_url=effective_base_url,
            provider=new_provider,
            is_codex_backend=new_norm == 'openai-codex',
        )
    )

    # Guard against a trailing /v1 on OpenCode base_url reaching the anthropic_messages
    # client (double-/v1 404); model_switch already strips it, direct callers may not.
    from hermes_cli.models import opencode_provider_family

    if (
        api_mode == "anthropic_messages"
        and opencode_provider_family(new_provider) is not None
        and isinstance(base_url, str)
        and base_url
    ):
        base_url = re.sub(r"/v1/?$", "", base_url)

    # Snapshot every field the swap+rebuild mutates so a failed rebuild rolls back atomically
    # (else new model name + OLD client -> 400s next turn). Sentinel distinguishes unset from
    # None: tests build bare agents via __new__ without all fields.
    _MISSING = object()
    _snapshot = {
        name: getattr(agent, name, _MISSING)
        for name in (
            "model",
            "provider",
            "requested_provider",
            "base_url",
            "api_mode",
            "api_key",
            "client",
            "_anthropic_client",
            "_anthropic_api_key",
            "_anthropic_base_url",
            "_is_anthropic_oauth",
            "_config_context_length",
            "_reasoning_echo_flag",
            "runtime_capabilities",
        )
    }
    # Shallow-copy the dict so mutating the live one doesn't poison the rollback target.
    _snapshot["_client_kwargs"] = dict(getattr(agent, "_client_kwargs", {}) or {})
    # Pool reload is part of this switch and must be reversible on rollback (#52727).
    _snapshot["_credential_pool"] = getattr(agent, "_credential_pool", _MISSING)
    _snapshot["_credential_pool_entry_id"] = getattr(
        agent, "_credential_pool_entry_id", _MISSING
    )

    def _restore_snapshot() -> None:
        for _name, _value in _snapshot.items():
            if _value is _MISSING:
                # Attribute did not exist before the swap; don't fabricate it.
                continue
            try:
                setattr(agent, _name, _value)
            except Exception:  # noqa: BLE001
                pass

    try:
        # Clear the per-config override so the new model's context window is re-resolved.
        agent._config_context_length = None

        # ── Swap core runtime fields ──
        agent.model = new_model
        agent.provider = new_provider
        agent.requested_provider = new_provider
        # Re-read reasoning_echo so the flag reflects the new primary model (see _reasoning_echo_opt_in).
        agent._reasoning_echo_flag = agent._read_reasoning_echo_from_config()
        # Empty base_url while the provider changes means upstream resolution failed; falling
        # back to the old provider's URL pairs the wrong host and persists via _primary_runtime
        # (#47828). Fail loud. Same-provider re-select (credential refresh) may keep the URL.
        if base_url:
            agent.base_url = base_url
        elif old_norm != new_norm:
            raise ValueError(
                f"switch_model: no base_url resolved for provider "
                f"'{new_provider}' (switching from '{old_provider}'); "
                "refusing to keep the previous provider's endpoint"
            )
        agent.api_mode = api_mode
        # New api_mode may need a different transport.
        if hasattr(agent, "_transport_cache"):
            agent._transport_cache.clear()
        if api_key:
            agent.api_key = api_key

        # Reload the credential pool on provider change (#52727): a pool with a mismatched
        # provider makes recover_with_credential_pool short-circuit. Reload failure is non-fatal.
        if old_norm != new_norm or getattr(agent, "_credential_pool", None) is None:
            # A pool bound to the old provider is worse than none: the recovery guard rejects it.
            agent._credential_pool = None
            agent._credential_pool_entry_id = None
            try:
                from agent.credential_pool import load_pool
                agent._credential_pool = load_pool(new_provider)
            except Exception as _pool_exc:  # noqa: BLE001
                logger.warning(
                    "switch_model: credential pool reload failed for %s (%s); "
                    "continuing without pool rotation this turn",
                    new_provider, _pool_exc,
                )
        # ── Build new client ──
        if new_norm == "moa":
            from agent.moa_loop import build_moa_facade

            # MoA speaks only chat.completions via the MoAClient facade; the aggregator's real
            # transport is applied inside the fan-out. Pin api_mode so the loop never dispatches
            # client.responses.create against the facade (matches agent_init.py).
            agent.api_mode = "chat_completions"
            agent.api_key = api_key or "moa-virtual-provider"
            agent.base_url = "moa://local"
            agent._client_kwargs = {}
            agent.client = build_moa_facade(agent, agent.model)
        elif api_mode == "anthropic_messages":
            from agent.anthropic_adapter import (
                build_anthropic_client,
                resolve_anthropic_token,
                _is_oauth_token,
            )
            # Only fall back to ANTHROPIC_TOKEN for native Anthropic; other anthropic_messages
            # providers must never receive Anthropic credentials.
            _is_native_anthropic = new_provider == "anthropic"
            effective_key = (api_key or agent.api_key or resolve_anthropic_token() or "") if _is_native_anthropic else (api_key or agent.api_key or "")

            # MiniMax OAuth: per-request callable token provider survives 15-min expiry
            # (rationale in agent_init.py).
            if new_provider == "minimax-oauth" and isinstance(effective_key, str) and effective_key:
                try:
                    from hermes_cli.auth import build_minimax_oauth_token_provider
                    effective_key = build_minimax_oauth_token_provider()
                except Exception as _mm_exc:  # noqa: BLE001
                    logger.warning(
                        "MiniMax OAuth: failed to install per-request token provider "
                        "on switch (%s); using static bearer.",
                        _mm_exc,
                    )

            agent.api_key = effective_key
            agent._anthropic_api_key = effective_key
            agent._anthropic_base_url = base_url or getattr(agent, "_anthropic_base_url", None)
            agent._anthropic_client = build_anthropic_client(
                effective_key, agent._anthropic_base_url,
                timeout=get_provider_request_timeout(agent.provider, agent.model),
            )
            agent._is_anthropic_oauth = _is_oauth_token(effective_key) if (_is_native_anthropic and isinstance(effective_key, str)) else False
            agent.client = None
            agent._client_kwargs = {}
        else:
            effective_key = api_key or agent.api_key
            effective_base = base_url or agent.base_url
            agent._client_kwargs = {
                "api_key": effective_key,
                "base_url": effective_base,
            }
            try:
                from hermes_cli.config import (
                    apply_custom_provider_tls_to_client_kwargs,
                    get_compatible_custom_providers,
                    load_config_readonly,
                )

                # Read live config, not agent._custom_providers, so mid-session ssl_ca_cert /
                # ssl_verify edits are honored (#15779).
                apply_custom_provider_tls_to_client_kwargs(
                    agent._client_kwargs,
                    str(effective_base or ""),
                    get_compatible_custom_providers(load_config_readonly()),
                )
            except Exception:
                logger.debug("custom-provider TLS resolution skipped on switch_model", exc_info=True)
            _sm_timeout = get_provider_request_timeout(agent.provider, agent.model)
            if _sm_timeout is not None:
                agent._client_kwargs["timeout"] = _sm_timeout
            # Reapply provider headers (OpenRouter HTTP-Referer/X-Title) lost when
            # _client_kwargs was rebuilt; otherwise attribution shows "Unknown".
            agent._apply_client_headers_for_base_url(effective_base)
            agent.client = agent._create_openai_client(
                dict(agent._client_kwargs),
                reason="switch_model",
                shared=True,
            )

        sync_credential_pool_entry_id(agent)
    except Exception:
        # Roll back to the pre-swap snapshot so the agent stays consistent; callers
        # (cli.py / gateway/run.py / tui_gateway) catch the re-raised exception.
        _restore_snapshot()
        raise

    # LM Studio: preload before probing context length.
    _sm_custom_providers = None
    try:
        from hermes_cli.config import (
            get_compatible_custom_providers,
            get_custom_provider_context_length,
            load_config,
        )

        _sm_cfg = load_config()
        _sm_custom_providers = get_compatible_custom_providers(_sm_cfg)
        _destination_context_intent = get_custom_provider_context_length(
            model=agent.model,
            base_url=agent.base_url,
            custom_providers=_sm_custom_providers,
        )
    except Exception:
        _destination_context_intent = None
    agent._config_context_length = _destination_context_intent
    if hasattr(agent, "_ensure_lmstudio_runtime_loaded"):
        try:
            _runtime_context_length = agent._ensure_lmstudio_runtime_loaded(
                _destination_context_intent
            )
        except Exception:
            _restore_snapshot()
            raise
    else:
        _runtime_context_length = None
    if (
        hasattr(agent, "_lmstudio_load_was_unverified")
        and agent._lmstudio_load_was_unverified(_runtime_context_length)
    ):
        logger.warning(
            "LM Studio model activation was rejected or completed without a "
            "verifiable active context length during model switch; continuing "
            "with configured context"
        )
    if hasattr(agent, "_effective_lmstudio_context_length"):
        _effective_context_length = agent._effective_lmstudio_context_length(
            _destination_context_intent,
            _runtime_context_length,
        )
    else:
        _effective_context_length = _destination_context_intent

    # Refresh the custom-provider snapshot from the config just loaded so the prompt_caching
    # lookup sees flags added to config.yaml after session start.
    if _sm_custom_providers is not None:
        agent._custom_providers = _sm_custom_providers
    agent._use_prompt_caching, agent._use_native_cache_layout = (
        agent._anthropic_prompt_cache_policy(
            provider=new_provider,
            base_url=agent.base_url,
            api_mode=api_mode,
            model=new_model,
        )
    )

    # ── Update context compressor ──
    if hasattr(agent, "context_compressor") and agent.context_compressor:
        from agent.model_metadata import get_model_context_length
        if _sm_custom_providers is None:
            try:
                from hermes_cli.config import get_compatible_custom_providers, load_config
                _sm_custom_providers = get_compatible_custom_providers(load_config())
            except Exception:
                _sm_custom_providers = None
        # agent.api_key may be a callable (Azure Foundry Entra ID); get_model_context_length
        # expects a string for live probes, so coerce defensively.
        _ctx_api_key = agent.api_key if isinstance(agent.api_key, str) else ""
        try:
            new_context_length = get_model_context_length(
                agent.model,
                base_url=agent.base_url,
                api_key=_ctx_api_key,
                provider=agent.provider,
                config_context_length=_effective_context_length,
                custom_providers=_sm_custom_providers,
            )
            agent.context_compressor.update_model(
                model=agent.model,
                context_length=new_context_length,
                base_url=agent.base_url,
                api_key=agent.api_key,  # context_compressor forwards to call_llm; callable preserved
                provider=agent.provider,
                api_mode=agent.api_mode,
            )
        except Exception:
            _restore_snapshot()
            raise

    # Re-read the per-model reasoning_effort override so it applies immediately
    # (per-model > global; YAML False = disabled).
    try:
        from hermes_constants import resolve_reasoning_config
        from hermes_cli.config import load_config as _sm_load_config

        _reasoning_cfg = _sm_load_config() or {}
        agent.reasoning_config = resolve_reasoning_config(_reasoning_cfg, agent.model)
        logger.info(
            "switch_model: reasoning_config resolved for %s: %s",
            agent.model, agent.reasoning_config,
        )
    except Exception as _reasoning_err:
        logger.debug("switch_model: could not re-resolve reasoning_config: %s", _reasoning_err)

    # Invalidate the cached system prompt so it rebuilds next turn.
    agent._cached_system_prompt = None

    # Publish the destination capability map only after every runtime setup
    # above has succeeded. Failed switches must leave the old map intact.
    agent.runtime_capabilities = destination_capabilities

    # Reset the cross-turn stale-call circuit breaker (#58962); otherwise the latched
    # streak keeps short-circuiting the freshly selected healthy provider.
    from agent.chat_completion_helpers import _reset_stale_streak
    _reset_stale_streak(agent)

    # Update _primary_runtime so the change persists across turns.
    _cc = agent.context_compressor if hasattr(agent, "context_compressor") and agent.context_compressor else None
    agent._primary_runtime = {
        "model": agent.model,
        "provider": agent.provider,
        "requested_provider": agent.requested_provider,
        "base_url": agent.base_url,
        "api_mode": agent.api_mode,
        "api_key": getattr(agent, "api_key", ""),
        "client_kwargs": dict(agent._client_kwargs),
        "use_prompt_caching": agent._use_prompt_caching,
        "use_native_cache_layout": agent._use_native_cache_layout,
        "reasoning_config": dict(agent.reasoning_config) if getattr(agent, "reasoning_config", None) else None,
        "reasoning_echo_flag": getattr(agent, "_reasoning_echo_flag", False),
        # Overrides must travel with the switched-to identity or a later recovery/restore
        # resurrects PRE-switch overrides from the stale init snapshot (#75091).
        "request_overrides": dict(getattr(agent, "request_overrides", {}) or {}),
        "runtime_capabilities": dict(getattr(agent, "runtime_capabilities", {}) or {}),
        "compressor_model": getattr(_cc, "model", agent.model) if _cc else agent.model,
        "compressor_base_url": getattr(_cc, "base_url", agent.base_url) if _cc else agent.base_url,
        "compressor_api_key": getattr(_cc, "api_key", "") if _cc else "",
        "compressor_provider": getattr(_cc, "provider", agent.provider) if _cc else agent.provider,
        "compressor_context_length": _cc.context_length if _cc else 0,
        "compressor_api_mode": getattr(_cc, "api_mode", agent.api_mode) if _cc else agent.api_mode,
        "compressor_threshold_tokens": _cc.threshold_tokens if _cc else 0,
    }
    if api_mode == "anthropic_messages":
        agent._primary_runtime.update({
            "anthropic_api_key": agent._anthropic_api_key,
            "anthropic_base_url": agent._anthropic_base_url,
            "is_anthropic_oauth": agent._is_anthropic_oauth,
        })

    # ── Reset fallback state ──
    agent._fallback_activated = False
    agent._provider_fallback_active = False
    agent._provider_fallback_route = None
    agent._fallback_index = 0

    # On a deliberate provider swap, prune fallback entries targeting the OLD or NEW primary;
    # otherwise a failed turn silently re-activates the provider the user just rejected.
    fallback_chain = list(getattr(agent, "_fallback_chain", []) or [])
    if old_norm and new_norm and old_norm != new_norm:
        fallback_chain = [
            entry for entry in fallback_chain
            if (entry.get("provider") or "").strip().lower() not in {old_norm, new_norm}
        ]
    agent._fallback_chain = fallback_chain
    agent._fallback_model = fallback_chain[0] if fallback_chain else None

    # Apply the switched-to provider's request_overrides (custom_providers extra_body).
    try:
        _apply_switched_provider_request_overrides(agent, new_provider)
    except Exception:
        logger.debug("switch_model: request_overrides re-derivation failed", exc_info=True)

    logger.info(
        "Model switched in-place: %s (%s) -> %s (%s)",
        old_model, old_provider, new_model, new_provider,
    )

    # Persist billing route so dashboard Model cards show the post-switch provider (#48248).
    # _session_db / session_id may be unset (tests, bare agents).
    _session_db = getattr(agent, "_session_db", None)
    _session_id = getattr(agent, "session_id", None)
    if _session_db is not None and _session_id:
        try:
            _session_db.update_session_billing_route(
                _session_id,
                provider=agent.provider,
                base_url=agent.base_url,
                billing_mode=getattr(agent, "api_mode", None),
            )
        except Exception:
            logger.warning(
                "Failed to persist billing route after model switch",
                exc_info=True,
            )


def invoke_tool(agent, function_name: str, function_args: dict, effective_task_id: str,
                 tool_call_id: Optional[str] = None, messages: list = None,
                 pre_tool_block_checked: bool = False,
                 skip_tool_request_middleware: bool = False,
                 tool_request_middleware_trace: Optional[List[Dict[str, Any]]] = None,
                 skip_tool_execution_middleware: bool = False) -> str:
    """Invoke a single tool and return the result string; no display logic.

    Handles agent-level and registry-dispatched tools. Used by the concurrent
    path; the sequential path keeps its own inline invocation for display.
    """
    from agent.inline_tool_executors import (
        InlineToolContext,
        emit_terminal_post_tool_call,
        resolve_invoke_tool_executor,
        tool_hook_ids,
    )

    if not isinstance(function_args, dict):
        function_args = {}

    _tool_middleware_trace = list(tool_request_middleware_trace or [])
    try:
        from hermes_cli.middleware import apply_tool_request_middleware

        if not skip_tool_request_middleware:
            _tool_request_mw = apply_tool_request_middleware(
                function_name,
                function_args,
                **tool_hook_ids(agent, effective_task_id, tool_call_id),
            )
            function_args = _tool_request_mw.payload
            _tool_middleware_trace = _tool_request_mw.trace
    except Exception as _mw_err:
        logger.debug("tool_request middleware error: %s", _mw_err)

    # Check plugin hooks for a block or approval directive before executing.
    block_message: Optional[str] = None
    if not pre_tool_block_checked:
        try:
            from hermes_cli.plugins import _dispatch_pre_tool_call_hooks
            block_message, modified_args = _dispatch_pre_tool_call_hooks(
                function_name, function_args, task_id=effective_task_id or "",
                session_id=getattr(agent, "session_id", "") or "",
                tool_call_id=tool_call_id or "",
                turn_id=getattr(agent, "_current_turn_id", "") or "",
                api_request_id=getattr(agent, "_current_api_request_id", "") or "",
                middleware_trace=list(_tool_middleware_trace),
            )
            if modified_args is not None:
                function_args = modified_args
        except Exception:
            block_message = None
    if block_message is not None:
        result = json.dumps({"error": block_message}, ensure_ascii=False)
        emit_terminal_post_tool_call(
            agent,
            function_name=function_name,
            function_args=function_args,
            result=result,
            effective_task_id=effective_task_id,
            tool_call_id=tool_call_id,
            status="blocked",
            error_type="plugin_block",
            error_message=block_message,
            middleware_trace=_tool_middleware_trace,
        )
        return result

    tool_start_time = time.monotonic()

    def _finish_agent_tool(result: Any, observed_args: Optional[dict] = None) -> Any:
        emit_terminal_post_tool_call(
            agent,
            function_name=function_name,
            function_args=observed_args if isinstance(observed_args, dict) else function_args,
            result=result,
            effective_task_id=effective_task_id,
            tool_call_id=tool_call_id,
            duration_ms=int((time.monotonic() - tool_start_time) * 1000),
            middleware_trace=_tool_middleware_trace,
        )
        return result

    inline_executor = resolve_invoke_tool_executor(agent, function_name)
    if inline_executor is not None:
        inline_ctx = InlineToolContext(
            effective_task_id=effective_task_id,
            tool_call_id=tool_call_id,
            messages=messages,
        )

        def _execute(next_args: dict) -> Any:
            return _finish_agent_tool(inline_executor(agent, next_args, inline_ctx), next_args)
    else:
        def _execute(next_args: dict) -> Any:
            dispatch_kwargs = dict(
                tool_call_id=tool_call_id,
                session_id=agent.session_id or "",
                turn_id=getattr(agent, "_current_turn_id", "") or "",
                api_request_id=getattr(agent, "_current_api_request_id", "") or "",
                enabled_tools=list(agent.valid_tool_names) if agent.valid_tool_names else None,
                skip_pre_tool_call_hook=True,
                skip_tool_request_middleware=True,
                enabled_toolsets=getattr(agent, "enabled_toolsets", None),
                disabled_toolsets=getattr(agent, "disabled_toolsets", None),
                tool_request_middleware_trace=list(_tool_middleware_trace),
            )
            if skip_tool_execution_middleware:
                dispatch_kwargs["skip_tool_execution_middleware"] = True
            return _ra().handle_function_call(
                function_name,
                next_args,
                effective_task_id,
                **dispatch_kwargs,
            )

    if skip_tool_execution_middleware:
        return _execute(function_args)

    from hermes_cli.middleware import run_tool_execution_middleware

    return run_tool_execution_middleware(
        function_name,
        function_args,
        lambda next_args: _execute(next_args if isinstance(next_args, dict) else function_args),
        original_args=function_args,
        **tool_hook_ids(agent, effective_task_id, tool_call_id),
    )



def repair_tool_call(agent, tool_name: str) -> str | None:
    """Repair a mismatched tool name (case, separators, CamelCase, ``_tool`` suffixes, then fuzzy match) before aborting.

    Suffix stripping is applied twice so ``TodoTool_tool`` reduces fully (#14784).
    Returns the repaired name if in valid_tool_names, else None.
    """
    import re
    from difflib import get_close_matches

    if not tool_name:
        return None

    # VolcEngine api/plan (#33007) leaks XML attribute fragments into tool_use.name
    # (`terminal" parameter="command" ...`); trim at the first quote/angle bracket.
    # Do NOT split on whitespace: "write file" must reach ``_norm`` -> ``write_file``
    # (test_space_to_underscore in tests/run_agent/test_repair_tool_call_name.py).
    for _xml_sep in ('"', "'", "<", ">"):
        _idx = tool_name.find(_xml_sep)
        if _idx > 0:
            tool_name = tool_name[:_idx]
    if not tool_name:
        return None

    def _norm(s: str) -> str:
        return s.lower().replace("-", "_").replace(" ", "_")

    def _camel_snake(s: str) -> str:
        return re.sub(r"(?<!^)(?=[A-Z])", "_", s).lower()

    def _strip_tool_suffix(s: str) -> str | None:
        lc = s.lower()
        for suffix in ("_tool", "-tool", "tool"):
            if lc.endswith(suffix):
                return s[: -len(suffix)].rstrip("_-")
        return None

    # Cheap fast-paths first.
    lowered = tool_name.lower()
    if lowered in agent.valid_tool_names:
        return lowered
    normalized = _norm(tool_name)
    if normalized in agent.valid_tool_names:
        return normalized

    cands: set[str] = {tool_name, lowered, normalized, _camel_snake(tool_name)}
    # Strip trailing tool-suffix up to twice (TodoTool_tool needs it).
    for _ in range(2):
        extra: set[str] = set()
        for c in cands:
            stripped = _strip_tool_suffix(c)
            if stripped:
                extra.add(stripped)
                extra.add(_norm(stripped))
                extra.add(_camel_snake(stripped))
        cands |= extra

    for c in cands:
        if c and c in agent.valid_tool_names:
            return c

    matches = get_close_matches(lowered, agent.valid_tool_names, n=1, cutoff=0.7)
    if matches:
        return matches[0]

    return None


def _tool_call_id_variants(tc: Any) -> set:
    """Return every id a tool result might match this tool_call on.

    Thin backward-compatible forwarder; policy owner is
    ``agent.message_sanitization.tool_call_id_variants``.
    """
    return set(tool_call_id_variants(tc))


# Placeholder for an empty non-final message the provider would reject. Kept identical to
# the stub placeholder in chat_completion_helpers so healed transcripts read consistently.
_INTERRUPTED_PLACEHOLDER = "[response interrupted]"

# Escalate repeated heals once per session window, then stay quiet (#96870). Default
# threshold; tunable via ``agent.sanitizer_heal_escalation_threshold`` (<= 0 disables).
_EMPTY_HEAL_ESCALATE_AFTER = 3
_EMPTY_HEAL_WINDOW_S = 600.0
_empty_heal_log_state: Dict[str, Dict[str, Any]] = {}
_empty_heal_log_lock = threading.Lock()
# Sessions already given the one-time user notice; separate from the windowed log state
# so the user is told ONCE per session (#96870, out-of-band, never in conversation context).
_empty_heal_user_notified: set = set()
# One-shot pending notices keyed by session, drained by the conversation loop via
# ``consume_pending_sanitizer_heal_notice`` and delivered via the status/warning callback.
_empty_heal_pending_notice: Dict[str, str] = {}


def _msg_has_payload(msg: Dict[str, Any]) -> bool:
    """True if ``msg`` carries anything the API treats as non-empty content (text, multimodal blocks, tool_calls, tool_call_id, reasoning).

    Role-agnostic counterpart of ``AIAgent._is_thinking_only_assistant``.
    """
    content = msg.get("content")
    if isinstance(content, str):
        if content.strip():
            return True
    elif isinstance(content, list):
        for block in content:
            if isinstance(block, dict):
                # any typed block counts, as long as a text block is not itself blank
                if block.get("type") == "text":
                    if isinstance(block.get("text"), str) and block["text"].strip():
                        return True
                    continue
                return True
            elif block:
                return True
    elif content not in (None, ""):
        return True
    # Structural payloads that make an "empty-content" message still valid.
    if msg.get("tool_calls"):
        return True
    if isinstance(msg.get("reasoning_content"), str) and msg["reasoning_content"].strip():
        return True
    if msg.get("reasoning") or msg.get("reasoning_details"):
        return True
    # Codex Responses item carriers persist with content:"" by design (text lives in
    # codex_message_items / codex_reasoning_items and is replayed); treat as payload so
    # the repair never rewrites a designed-empty codex turn.
    return bool(msg.get("codex_message_items") or msg.get("codex_reasoning_items"))


def fill_empty_non_final_wire_payload(
    msg: Dict[str, Any], *, is_final: bool
) -> bool:
    """Write the interrupted placeholder onto an empty non-final wire copy; returns True when filled.

    Pass the per-call copy only; durable history must not be mutated
    (#88955, #96870).
    """
    if is_final or not isinstance(msg, dict):
        return False
    if msg.get("role") not in ("user", "assistant"):
        return False
    if _msg_has_payload(msg):
        return False
    msg["content"] = _INTERRUPTED_PLACEHOLDER
    return True


def _session_id_for_heal_log() -> str:
    try:
        from hermes_logging import _session_context

        return str(getattr(_session_context, "session_id", None) or "")
    except Exception:
        return ""


def _heal_escalation_threshold() -> int:
    """Escalation threshold from ``agent.sanitizer_heal_escalation_threshold``, else the module default (fail-safe on any read error)."""
    try:
        from hermes_cli.config import load_config_readonly

        raw = (load_config_readonly().get("agent", {}) or {}).get(
            "sanitizer_heal_escalation_threshold"
        )
        if raw is not None:
            return int(raw)
    except Exception:
        pass
    return _EMPTY_HEAL_ESCALATE_AFTER


def consume_pending_sanitizer_heal_notice() -> Optional[str]:
    """Drain the one-time user notice for the current session, if any (at most one per session lifetime).

    Delivered through the status/warning callback, NEVER appended to the
    conversation context.
    """
    key = _session_id_for_heal_log() or "-"
    with _empty_heal_log_lock:
        return _empty_heal_pending_notice.pop(key, None)


def get_sanitizer_heal_stats() -> Dict[str, Dict[str, Any]]:
    """Read-only snapshot of per-session sanitiser heal counters for diagnostics.

    Keyed by session id; values carry ``heal_events``, ``messages_healed`` and
    ``escalated``.
    """
    with _empty_heal_log_lock:
        return {
            k: {
                "heal_events": v.get("total_events", v.get("count", 0)),
                "messages_healed": v.get("total_healed", 0),
                "escalated": k in _empty_heal_user_notified,
            }
            for k, v in _empty_heal_log_state.items()
        }


def _log_empty_non_final_heal(healed: int) -> None:
    """WARNING on the first heals in a window, one ERROR at the threshold, then silent (#96870).

    The threshold also queues a ONE-TIME out-of-band user notice (drained by
    ``consume_pending_sanitizer_heal_notice``); never re-armed by a new window.
    """
    key = _session_id_for_heal_log() or "-"
    threshold = _heal_escalation_threshold()
    now = time.monotonic()
    with _empty_heal_log_lock:
        state = _empty_heal_log_state.get(key)
        if state is None or (now - state["window_start"]) > _EMPTY_HEAL_WINDOW_S:
            prior_events = state.get("total_events", 0) if state else 0
            prior_healed = state.get("total_healed", 0) if state else 0
            state = {
                "count": 0,
                "window_start": now,
                "escalated": False,
                "total_events": prior_events,
                "total_healed": prior_healed,
            }
            _empty_heal_log_state[key] = state
        state["count"] += 1
        state["total_events"] = state.get("total_events", 0) + 1
        state["total_healed"] = state.get("total_healed", 0) + healed
        count = state["count"]
        total_events = state["total_events"]
        total_healed = state["total_healed"]
        if threshold > 0 and count >= threshold and not state["escalated"]:
            state["escalated"] = True
            level = "error"
            if key not in _empty_heal_user_notified:
                _empty_heal_user_notified.add(key)
                _empty_heal_pending_notice[key] = (
                    "⚠️ Your session transcript required repeated repair "
                    f"({total_events} heal passes so far). Replies keep "
                    "working, but a corrupted turn is stuck in this "
                    "session's history — run /debug share or `hermes "
                    "doctor` to capture diagnostics, or /new to start a "
                    "clean session."
                )
        elif state["escalated"]:
            level = "silent"
        else:
            level = "warning"

    if level == "silent":
        return
    if level == "error":
        _ra().logger.error(
            "Pre-call sanitizer: repeated-heal escalation for session %s — "
            "healed %d empty non-final message(s) this send; heal pattern: "
            "%d heal events / %d messages healed this session "
            "(%d in the current session window, threshold %d). The transcript "
            "is being repaired on every send; /new drops the poisoned turns.",
            key,
            healed,
            total_events,
            total_healed,
            count,
            threshold,
        )
        return
    _ra().logger.warning(
        "Pre-call sanitizer: healed %d empty non-final message(s) by "
        "substituting placeholder content — an empty-content turn was in "
        "the transcript and would 400 the request ('messages must have "
        "non-empty content' / INVALID_REQUEST_BODY). Self-recovering the "
        "poisoned transcript in memory; no restart needed.",
        healed,
    )


def repair_empty_non_final_messages(
    messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Substitute a placeholder for empty-content non-final messages on the per-call copy.

    Anthropic/litellm/Bedrock 400 on any empty non-final message, and a
    persisted stub poisons every later turn; repairing the wire copy heals
    the session in memory. Substitution (not deletion) keeps role alternation
    and tool-call pairing intact. The final message is left untouched.
    """
    if not messages or len(messages) < 2:
        return messages

    repaired: List[Dict[str, Any]] = []
    healed = 0
    last_idx = len(messages) - 1
    for idx, msg in enumerate(messages):
        if (
            idx != last_idx
            and isinstance(msg, dict)
            # Tool results are checked by their own pairing pass; empty ones are a separate concern.
            and msg.get("role") in ("assistant", "user")
            and not _msg_has_payload(msg)
        ):
            # Shallow-copy so stored history / prompt caching stays byte-stable.
            fixed = dict(msg)
            fixed["content"] = _INTERRUPTED_PLACEHOLDER
            repaired.append(fixed)
            healed += 1
        else:
            repaired.append(msg)

    if healed:
        _log_empty_non_final_heal(healed)
        return repaired
    return messages


def _classify_tool_call_orphans(messages: List[Dict[str, Any]]):
    """Classify orphaned tool-call / tool-result pairs; single source of truth for GLOBAL orphan detection.

    Returns ``(surviving_call_ids, result_call_ids, orphaned_results,
    missing_tool_calls)``. Every id variant of a tool_call (``id``,
    ``call_id``, ``response_item_id``, composite bridge) is registered so a
    result matching any alias survives (#55626, #63000, #58357).
    ``orphaned_results`` are the actual dicts (filter by ``id(msg)``).
    ``sanitize_api_messages`` pairs positionally instead (#94704) but shares
    the ``tool_call_id_variants`` / ``tool_result_id_variants`` alias policy.
    """
    assistant_call_variants: List[tuple[Any, frozenset[str]]] = []
    surviving_call_ids: set[str] = set()
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls") or []:
            variants = tool_call_id_variants(tc)
            if variants:
                assistant_call_variants.append((tc, variants))
                surviving_call_ids.update(variants)

    result_entries = [
        (msg, tool_result_id_variants(msg.get("tool_call_id")))
        for msg in messages
        if msg.get("role") == "tool"
    ]
    result_call_ids: set[str] = set()
    for _, variants in result_entries:
        result_call_ids.update(variants)

    orphaned_results = [
        msg
        for msg, variants in result_entries
        if variants and not (variants & surviving_call_ids)
    ]
    orphaned_ids = {id(msg) for msg in orphaned_results}
    surviving_result_variants = [
        variants
        for msg, variants in result_entries
        if variants and id(msg) not in orphaned_ids
    ]
    missing_tool_calls = [
        tc
        for tc, variants in assistant_call_variants
        if not any(variants & rv for rv in surviving_result_variants)
    ]
    return surviving_call_ids, result_call_ids, orphaned_results, missing_tool_calls


def sanitize_api_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Fix orphaned tool_call / tool_result pairs before every LLM call; runs unconditionally (not gated on the compressor)."""
    # --- Role allowlist: drop messages with roles the API won't accept ---
    filtered = []
    for msg in messages:
        role = msg.get("role")
        if role not in _ra().AIAgent._VALID_API_ROLES:
            _ra().logger.debug(
                "Pre-call sanitizer: dropping message with invalid role %r",
                role,
            )
            continue
        filtered.append(msg)
    messages = filtered

    # --- Heal empty-content non-final messages (self-recovery) ---
    # A dead stream can leave an empty stub mid-transcript that 400s every later request;
    # repair the per-call copy so the session heals in memory. Done first so the substituted
    # turn participates in the tool-pair and dedup passes below.
    messages = repair_empty_non_final_messages(messages)

    # --- Drop empty / malformed tool_calls arrays on assistant messages ---
    # Strict providers 400 on ``tool_calls: []`` (#58755, #56980). Normalize on the
    # per-call copy (shallow-copy) so persisted history stays byte-stable.
    normalized: List[Dict[str, Any]] = []
    dropped_empty_tool_calls = 0
    for msg in messages:
        if (
            isinstance(msg, dict)
            and msg.get("role") == "assistant"
            and "tool_calls" in msg
            and not (isinstance(msg["tool_calls"], list) and msg["tool_calls"])
        ):
            msg = {k: v for k, v in msg.items() if k != "tool_calls"}
            dropped_empty_tool_calls += 1
        normalized.append(msg)
    if dropped_empty_tool_calls:
        messages = normalized
        _ra().logger.debug(
            "Pre-call sanitizer: dropped empty/invalid tool_calls on %d "
            "assistant message(s)",
            dropped_empty_tool_calls,
        )

    # --- Repair tool_calls whose function.name is empty/missing ---
    # Rename to a sentinel instead of dropping: the dispatch loop keeps empty-name calls
    # paired with an anti-priming result (#47967), and Responses adapters drop nameless calls (400).
    _EMPTY_NAME_SENTINEL = "invalid_tool_call"
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        tcs = msg.get("tool_calls") or []
        if not tcs:
            continue
        for tc in tcs:
            if isinstance(tc, dict):
                fn = tc.get("function")
                name = fn.get("name") if isinstance(fn, dict) else getattr(fn, "name", None)
            else:
                fn = getattr(tc, "function", None)
                name = getattr(fn, "name", None) if fn else None
            if isinstance(name, str) and name.strip():
                continue
            _ra().logger.warning(
                "Pre-call sanitizer: repairing tool_call with empty "
                "function.name -> %r (id=%s)",
                _EMPTY_NAME_SENTINEL,
                _ra().AIAgent._get_tool_call_id_static(tc),
            )
            if isinstance(fn, dict):
                fn["name"] = _EMPTY_NAME_SENTINEL
            elif fn is not None and hasattr(fn, "name"):
                try:
                    fn.name = _EMPTY_NAME_SENTINEL
                except Exception:
                    pass
            elif isinstance(tc, dict):
                tc["function"] = {"name": _EMPTY_NAME_SENTINEL, "arguments": "{}"}

    # --- Drop tool results with a missing/empty tool_call_id ---
    # Kept explicit (not left to the positional walk) for its own log line and so the
    # final-chokepoint guarantee holds for callers skipping ``repair_message_sequence`` (#78071).
    _pre_id_filter_count = len(messages)
    messages = [
        m for m in messages
        if not (m.get("role") == "tool" and not (m.get("tool_call_id") or "").strip())
    ]
    if len(messages) != _pre_id_filter_count:
        _ra().logger.debug(
            "Pre-call sanitizer: dropped %d tool result(s) with missing/empty tool_call_id",
            _pre_id_filter_count - len(messages),
        )

    # --- Positional tool_call <-> tool_result pairing (#94704) ---
    # Strict providers (DeepSeek v4, Kimi) require results IMMEDIATELY after their call:
    # drop positional orphans, stub unanswered declared ids; matching is alias-aware (#55626/#63000/#93251).
    paired: List[Dict[str, Any]] = []
    declared_calls: Dict[str, tuple] = {}
    dropped_positional_orphans = 0
    added_stubs = 0

    def _flush_unanswered_stubs() -> None:
        nonlocal added_stubs
        for key in sorted(declared_calls):
            tc, _variants = declared_calls[key]
            cid = coalesce_tool_call_id(tc) or key
            paired.append({
                "role": "tool",
                "name": _ra().AIAgent._get_tool_call_name_static(tc),
                "content": "[Result unavailable — see context summary above]",
                "tool_call_id": cid,
            })
            added_stubs += 1
        declared_calls.clear()

    for msg in messages:
        role = msg.get("role")
        if role == "assistant":
            # A new assistant turn closes the previous tool-result run:
            # anything still pending was never answered positionally.
            _flush_unanswered_stubs()
            declared_calls = {}
            for tc in msg.get("tool_calls") or []:
                variants = tool_call_id_variants(tc)
                if variants:
                    # Key on a stable representative of the alias group so
                    # a result matching ANY spelling can consume the call.
                    declared_calls[sorted(variants)[0]] = (tc, variants)
            paired.append(msg)
        elif role == "tool":
            result_variants = tool_result_id_variants(msg.get("tool_call_id"))
            matched = next(
                (
                    key
                    for key, (_tc, variants) in declared_calls.items()
                    if variants & result_variants
                ),
                None,
            )
            if matched is not None:
                paired.append(msg)
                # Consume so a duplicate result reusing the id is dropped (strict providers reject duplicates).
                declared_calls.pop(matched, None)
            else:
                dropped_positional_orphans += 1
        else:
            if role == "user":
                # A user turn closes the tool-result run; later tool messages are orphans.
                _flush_unanswered_stubs()
            paired.append(msg)
    # The transcript may end right after an unanswered assistant turn.
    _flush_unanswered_stubs()
    if dropped_positional_orphans or added_stubs:
        messages = paired
    if dropped_positional_orphans:
        _ra().logger.debug(
            "Pre-call sanitizer: removed %d positionally orphaned tool result(s)",
            dropped_positional_orphans,
        )
    if added_stubs:
        _ra().logger.debug(
            "Pre-call sanitizer: added %d stub tool result(s) for "
            "positionally unanswered tool call(s)",
            added_stubs,
        )

    # 3. Deduplicate tool_call_ids (strict providers 400 on duplicates, #58327): collapse
    # duplicates within an assistant message; drop results answering no OUTSTANDING call.
    # Track outstanding calls (not ids ever seen) because llama.cpp reuses one constant id,
    # and track the whole variant group so alias-keyed results are not deleted (#93251).
    seen_assistant_call_ids: set = set()
    outstanding_call_ids: set = set()
    outstanding_groups: Dict[int, frozenset] = {}
    variant_to_group: Dict[str, int] = {}
    next_group_id = 0
    deduped: List[Dict[str, Any]] = []
    removed_dupes = 0
    for msg in messages:
        role = msg.get("role")
        if role == "assistant" and msg.get("tool_calls"):
            kept_tcs = []
            for tc in msg.get("tool_calls") or []:
                variants = tool_call_id_variants(tc)
                if variants and variants & seen_assistant_call_ids:
                    removed_dupes += 1
                    continue
                if variants:
                    group_id = next_group_id
                    next_group_id += 1
                    outstanding_groups[group_id] = variants
                    for variant in variants:
                        seen_assistant_call_ids.add(variant)
                        outstanding_call_ids.add(variant)
                        variant_to_group.setdefault(variant, group_id)
                kept_tcs.append(tc)
            if kept_tcs:
                msg = {**msg, "tool_calls": kept_tcs}
            elif len(kept_tcs) != len(msg.get("tool_calls") or []):
                msg = {k: v for k, v in msg.items() if k != "tool_calls"}
            deduped.append(msg)
        elif role == "tool":
            result_variants = tool_result_id_variants(msg.get("tool_call_id"))
            candidate_groups = {
                variant_to_group[variant]
                for variant in result_variants
                if variant in variant_to_group
                and variant in outstanding_call_ids
            }
            if result_variants and not candidate_groups:
                removed_dupes += 1
                continue
            if candidate_groups:
                # Consume EVERY variant of the matched call; ids are re-armed by the next call reusing them.
                group_id = min(candidate_groups)
                group_variants = outstanding_groups.pop(group_id, frozenset())
                for variant in group_variants:
                    outstanding_call_ids.discard(variant)
                    seen_assistant_call_ids.discard(variant)
                    if variant_to_group.get(variant) == group_id:
                        variant_to_group.pop(variant, None)
            deduped.append(msg)
        else:
            deduped.append(msg)
    if removed_dupes:
        messages = deduped
        _ra().logger.debug(
            "Pre-call sanitizer: removed %d duplicate tool_call_id reference(s)",
            removed_dupes,
        )

    # 4. Align each tool result's wire ``name`` with its call's function name: Google 400s
    # on a mismatch, which is routine when tool_search bridges via ``tool_call`` (#72089).
    # Done here, provider-agnostically, on the per-call copy only.
    call_names: Dict[str, str] = {}
    for msg in messages:
        if msg.get("role") == "assistant":
            for tc in msg.get("tool_calls") or []:
                # Strip on insert to match the lookup below so padded ids still pair.
                cid = (_ra().AIAgent._get_tool_call_id_static(tc) or "").strip()
                nm = _ra().AIAgent._get_tool_call_name_static(tc)
                if cid and nm:
                    call_names[cid] = nm
    realigned: List[Tuple[str, str]] = []
    aligned: List[Dict[str, Any]] = []
    for msg in messages:
        if msg.get("role") == "tool":
            cid = (msg.get("tool_call_id") or "").strip()
            expected = call_names.get(cid)
            current = msg.get("name")
            # Only rewrite a present, disagreeing name; clean transcripts must stay byte-identical for prompt caching.
            if expected and current and current != expected:
                msg = {**msg, "name": expected}
                realigned.append((current, expected))
        aligned.append(msg)
    if realigned:
        messages = aligned
        _ra().logger.debug(
            "Pre-call sanitizer: realigned %d tool result name(s) with their "
            "tool_call function name (%s)",
            len(realigned),
            ", ".join(f"{was} -> {now}" for was, now in realigned),
        )
    return messages



def looks_like_codex_intermediate_ack(
    agent,
    user_message: Any,
    assistant_content: str,
    messages: List[Dict[str, Any]],
    require_workspace: bool = True,
) -> bool:
    """Detect a planning/ack message that should continue instead of ending the turn.

    ``require_workspace=False`` (user opted into ``agent.intent_ack_continuation``
    for all api_modes) drops the filesystem/repo reference requirement; the
    future-ack + short-content + no-prior-tools + action-verb checks always apply.
    """
    if any(isinstance(msg, dict) and msg.get("role") == "tool" for msg in messages):
        return False

    assistant_text = agent._strip_think_blocks(assistant_content or "").strip().lower()
    if not assistant_text:
        return False
    if len(assistant_text) > 1200:
        return False

    has_future_ack = bool(
        re.search(r"\b(i['’]ll|i will|let me|i can do that|i can help with that)\b", assistant_text)
    )
    if not has_future_ack:
        return False

    action_markers = (
        "look into",
        "look at",
        "inspect",
        "scan",
        "check",
        "analyz",
        "review",
        "explore",
        "read",
        "open",
        "run",
        "test",
        "fix",
        "debug",
        "search",
        "find",
        "walkthrough",
        "report back",
        "summarize",
    )
    workspace_markers = (
        "directory",
        "current directory",
        "current dir",
        "cwd",
        "repo",
        "repository",
        "codebase",
        "project",
        "folder",
        "filesystem",
        "file tree",
        "files",
        "path",
    )

    assistant_mentions_action = any(marker in assistant_text for marker in action_markers)
    if not assistant_mentions_action:
        return False

    # Opted-in (all-api_mode) path: future-ack + action verb + no prior tool call suffices.
    if not require_workspace:
        return True

    # ``user_message`` may be a multi-part content list (vision via the OpenAI-compat
    # server); a list survives ``or ""`` and ``.strip()`` raises, so flatten first.
    from agent.codex_responses_adapter import _summarize_user_message_for_log

    user_text = _summarize_user_message_for_log(user_message).strip().lower()
    user_targets_workspace = (
        any(marker in user_text for marker in workspace_markers)
        or "~/" in user_text
        or "/" in user_text
    )
    assistant_targets_workspace = any(
        marker in assistant_text for marker in workspace_markers
    )
    return user_targets_workspace or assistant_targets_workspace


# Narrow "trailing continue-intent" detector for the stall guard (agent.stall_guards):
# only the message TAIL announcing a next action, so mid-sentence "I will" never trips it.
_TRAILING_CONTINUE_INTENT_RE = re.compile(
    r"(?:\blet me now\b|\bi(?:['\u2019])?ll now\b|\bi will now\b"
    r"|\bnow i(?:['\u2019]ll| will)\b|\bnext[,:] i\b)"
    r"[^.!?\n]{0,100}[.:\u2026]?\s*$",
    re.IGNORECASE,
)

# Content longer than this is a substantive reply, not a dangling ack.
_TRAILING_CONTINUE_INTENT_MAX_CHARS = 400


def trailing_continue_intent(text: str) -> bool:
    """Whether ``text`` is a short reply ENDING on an announced next action (stall-guard re-prompt trigger)."""
    t = (text or "").strip()
    if not t or len(t) > _TRAILING_CONTINUE_INTENT_MAX_CHARS:
        return False
    return bool(_TRAILING_CONTINUE_INTENT_RE.search(t[-160:]))


def intent_ack_continuation_mode(agent) -> str:
    """Resolve the intent-ack continuation mode: ``"off"``, ``"codex_only"`` (workspace acks on codex_responses), or ``"all"``.

    Mirrors ``agent.tool_use_enforcement``: ``"auto"`` -> codex_only; true-ish
    values -> all; false-ish -> off; ``list`` -> all when a substring matches
    the active model name, else off.
    """
    mode = getattr(agent, "_intent_ack_continuation", "auto")

    if mode is True or (isinstance(mode, str) and mode.lower() in {"true", "always", "yes", "on"}):
        return "all"
    if mode is False or (isinstance(mode, str) and mode.lower() in {"false", "never", "no", "off"}):
        return "off"
    if isinstance(mode, list):
        model_lower = (agent.model or "").lower()
        return "all" if any(p.lower() in model_lower for p in mode if isinstance(p, str)) else "off"
    # "auto" or any unrecognised value — historical codex-only behavior.
    return "codex_only" if agent.api_mode == "codex_responses" else "off"


def copy_reasoning_content_for_api(agent, source_msg: dict, api_msg: dict) -> None:
    """Forward reasoning fields onto an API replay message; policy lives in ``agent.message_sanitization.apply_reasoning_content_policy``."""
    from agent.message_sanitization import apply_reasoning_content_policy

    apply_reasoning_content_policy(
        source_msg, api_msg, agent._needs_thinking_reasoning_pad()
    )


def reapply_reasoning_echo_for_provider(agent, api_messages: list) -> int:
    """Re-pad or strip assistant turns' reasoning_content for the CURRENT provider after a fallback switch.

    ``api_messages`` is shaped for the primary provider; require-side providers
    (DeepSeek/Kimi/MiMo) 400 without the pad, strict ones (Mistral, Cerebras,
    Groq, ...) 400/422 with it (#45655). Idempotent. Returns the number of
    assistant turns changed.
    """
    from agent.message_sanitization import reapply_reasoning_echo

    return reapply_reasoning_echo(
        api_messages, agent._needs_thinking_reasoning_pad()
    )


def _iter_httpx_pool_objects(http_client: Any):
    """Yield httpcore pool objects reachable from an httpx client, including mounted transports.

    Keepalive (#10324) and proxy configs put live connections on ``client._mounts``;
    walking only ``_transport`` made ``force_close_tcp_sockets`` miss them (#72975).
    """
    seen_pools: set[int] = set()

    def _emit(pool: Any):
        if pool is None:
            return
        marker = id(pool)
        if marker in seen_pools:
            return
        seen_pools.add(marker)
        yield pool

    def _pools_for_transport(transport: Any):
        if transport is None:
            return
        # Connections live under ``_pool``; a directly mounted HTTPProxy *is* a
        # ConnectionPool, so ``_connections`` may sit on the transport itself.
        pool = getattr(transport, "_pool", None)
        if pool is not None:
            yield from _emit(pool)
            return
        if getattr(transport, "_connections", None) is not None:
            yield from _emit(transport)

    try:
        yield from _pools_for_transport(getattr(http_client, "_transport", None))
        mounts = getattr(http_client, "_mounts", None) or {}
        for _pattern, mounted in list(mounts.items()):
            yield from _pools_for_transport(mounted)
    except Exception:
        return


def _connection_candidates(conn: Any):
    """Walk nested ``_connection`` wrappers (proxy tunnel → HTTP11/2)."""
    seen: set[int] = set()
    stack = [conn]
    while stack:
        candidate = stack.pop()
        if candidate is None:
            continue
        marker = id(candidate)
        if marker in seen:
            continue
        seen.add(marker)
        yield candidate
        inner = getattr(candidate, "_connection", None)
        if inner is not None and id(inner) not in seen:
            stack.append(inner)


def _iter_pool_sockets(client: Any):
    """Yield raw sockets reachable from an OpenAI/httpx client pool.

    Traversal is defensive over private httpcore internals (``conn._connection``,
    proxy tunnel wrappers) that vary by release. Also walks mount transports and
    in-flight ``PoolRequest.connection`` objects, reachable when
    ``_connections`` is empty during checkout (#85252).
    """
    try:
        http_client = getattr(client, "_client", None)
        if http_client is None:
            # Some SDK wrappers *are* the httpx client; fall through so mount-aware discovery runs.
            http_client = client
        pools = list(_iter_httpx_pool_objects(http_client))
    except Exception:
        return

    if not pools:
        return

    seen: set[int] = set()
    for pool in pools:
        # ``is None``, not falsiness: an empty ``_connections`` must still let us walk in-flight ``_requests``.
        raw_conns = getattr(pool, "_connections", None)
        if raw_conns is None:
            raw_conns = getattr(pool, "_pool", None)
        connections = list(raw_conns or [])
        for pool_req in list(getattr(pool, "_requests", None) or []):
            conn = getattr(pool_req, "connection", None)
            if conn is not None:
                connections.append(conn)
        for conn in connections:
            for candidate in _connection_candidates(conn):
                stream = (
                    getattr(candidate, "_network_stream", None)
                    or getattr(candidate, "_stream", None)
                )
                if stream is None:
                    continue
                sock = getattr(stream, "_sock", None)
                if sock is None:
                    get_extra_info = getattr(stream, "get_extra_info", None)
                    if callable(get_extra_info):
                        try:
                            sock = get_extra_info("socket")
                        except Exception:
                            sock = None
                if sock is None:
                    wrapped = getattr(stream, "stream", None)
                    if wrapped is not None:
                        sock = getattr(wrapped, "_sock", None)
                if sock is None:
                    # anyio-backed streams expose the raw socket through
                    # SocketAttribute.raw_socket when available.
                    wrapped = getattr(stream, "_stream", None)
                    extra = getattr(wrapped, "extra", None)
                    if callable(extra):
                        try:
                            from anyio.abc import SocketAttribute
                            sock = extra(SocketAttribute.raw_socket)
                        except Exception:
                            sock = None
                if sock is None:
                    continue
                marker = id(sock)
                if marker in seen:
                    continue
                seen.add(marker)
                yield sock


def cleanup_dead_connections(agent) -> bool:
    """Force-close and rebuild the primary client if its pool has dead sockets (CLOSE-WAIT, errors); returns True if cleaned."""
    client = getattr(agent, "client", None)
    if client is None:
        return False
    try:
        dead_count = 0
        for sock in _iter_pool_sockets(client):
            # Probe socket health with a non-blocking recv peek
            import socket as _socket
            try:
                sock.setblocking(False)
                data = sock.recv(1, _socket.MSG_PEEK | _socket.MSG_DONTWAIT)
                if data == b"":
                    dead_count += 1
            except BlockingIOError:
                pass  # No data available — socket is healthy
            except OSError:
                dead_count += 1
            finally:
                try:
                    sock.setblocking(True)
                except OSError:
                    pass
        if dead_count > 0:
            _ra().logger.warning(
                "Found %d dead connection(s) in client pool — rebuilding client",
                dead_count,
            )
            agent._replace_primary_openai_client(reason="dead_connection_cleanup")
            return True
    except Exception as exc:
        _ra().logger.debug("Dead connection check error: %s", exc)
    return False



def extract_api_error_context(error: Exception) -> Dict[str, Any]:
    """Extract structured rate-limit details from provider errors."""
    context: Dict[str, Any] = {}

    body = getattr(error, "body", None)
    payload = None
    if isinstance(body, dict):
        payload = body.get("error") if isinstance(body.get("error"), dict) else body
    if isinstance(payload, dict):
        reason = payload.get("code") or payload.get("type") or payload.get("error")
        if isinstance(reason, str) and reason.strip():
            context["reason"] = reason.strip()
        message = payload.get("message") or payload.get("error_description")
        if not message and isinstance(payload.get("error"), str):
            # xAI uses a top-level string ``error`` beside a structured
            # ``code`` (for example personal-team-blocked:spending-limit).
            message = payload.get("error")
        if isinstance(message, str) and message.strip():
            context["message"] = message.strip()
        for key in ("resets_at", "reset_at"):
            value = payload.get(key)
            if value not in {None, ""}:
                context["reset_at"] = value
                break
        retry_after = payload.get("retry_after")
        if retry_after not in {None, ""} and "reset_at" not in context:
            try:
                context["reset_at"] = time.time() + float(retry_after)
            except (TypeError, ValueError):
                pass

    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    if headers:
        retry_after = headers.get("retry-after") or headers.get("Retry-After")
        if retry_after and "reset_at" not in context:
            try:
                context["reset_at"] = time.time() + float(retry_after)
            except (TypeError, ValueError):
                pass
        ratelimit_reset = headers.get("x-ratelimit-reset")
        if ratelimit_reset and "reset_at" not in context:
            context["reset_at"] = ratelimit_reset

    if "message" not in context:
        raw_message = str(error).strip()
        if raw_message:
            context["message"] = raw_message[:500]

    if "reset_at" not in context:
        message = context.get("message") or ""
        if isinstance(message, str):
            delay_match = re.search(r"quotaResetDelay[:\s\"]+(\d+(?:\.\d+)?)(ms|s)", message, re.IGNORECASE)
            if delay_match:
                value = float(delay_match.group(1))
                seconds = value / 1000.0 if delay_match.group(2).lower() == "ms" else value
                context["reset_at"] = time.time() + seconds
            else:
                resets_in_match = re.search(
                    r"resets?\s+in\s+"
                    r"(?:(\d+(?:\.\d+)?)\s*(?:h|hr|hrs|hour|hours)\b\s*)?"
                    r"(?:(\d+(?:\.\d+)?)\s*(?:m|min|mins|minute|minutes)\b\s*)?"
                    r"(?:(\d+(?:\.\d+)?)\s*(?:s|sec|secs|second|seconds)\b)?",
                    message,
                    re.IGNORECASE,
                )
                if resets_in_match and any(resets_in_match.groups()):
                    hours = float(resets_in_match.group(1) or 0)
                    minutes = float(resets_in_match.group(2) or 0)
                    seconds = float(resets_in_match.group(3) or 0)
                    context["reset_at"] = time.time() + (hours * 3600) + (minutes * 60) + seconds
                else:
                    sec_match = re.search(
                        r"retry\s+(?:after\s+)?(\d+(?:\.\d+)?)\s*(?:sec|secs|seconds|s\b)",
                        message,
                        re.IGNORECASE,
                    )
                    if sec_match:
                        context["reset_at"] = time.time() + float(sec_match.group(1))

    return context



def apply_pending_steer_to_tool_results(agent, messages: list, num_tool_msgs: int) -> None:
    """Append pending /steer text to the last ``role:"tool"`` message of this batch, marked as user-origin.

    Modifies existing content only, so role alternation is preserved.
    ``num_tool_msgs`` bounds the tail slice searched.
    """
    if num_tool_msgs <= 0 or not messages:
        return
    steer_text = agent._drain_pending_steer()
    if not steer_text:
        return
    # Skip non-tool messages in the tail in case something else is appended at the boundary.
    target_idx = None
    for j in range(len(messages) - 1, max(len(messages) - num_tool_msgs - 1, -1), -1):
        msg = messages[j]
        if isinstance(msg, dict) and msg.get("role") == "tool":
            target_idx = j
            break
    if target_idx is None:
        # No tool result in this batch (e.g. all skipped by interrupt): put the steer
        # back so the caller's fallback delivers it as a next-turn user message.
        _lock = getattr(agent, "_pending_steer_lock", None)
        if _lock is not None:
            with _lock:
                if agent._pending_steer:
                    agent._pending_steer = agent._pending_steer + "\n" + steer_text
                else:
                    agent._pending_steer = steer_text
        else:
            existing = getattr(agent, "_pending_steer", None)
            agent._pending_steer = (existing + "\n" + steer_text) if existing else steer_text
        return
    marker = format_steer_marker(steer_text)
    existing_content = messages[target_idx].get("content", "")
    if not isinstance(existing_content, str):
        # Anthropic multimodal content blocks: preserve them and append a text block.
        try:
            blocks = list(existing_content) if existing_content else []
            blocks.append({"type": "text", "text": marker.lstrip()})
            messages[target_idx]["content"] = blocks
        except Exception:
            # Fall back to string replacement if content shape is unexpected.
            messages[target_idx]["content"] = f"{existing_content}{marker}"
    else:
        messages[target_idx]["content"] = existing_content + marker
    _ra().logger.info(
        "Delivered /steer to agent after tool batch (%d chars): %s",
        len(steer_text),
        steer_text[:120] + ("..." if len(steer_text) > 120 else ""),
    )



def force_close_tcp_sockets(client: Any) -> int:
    """Abort in-flight TCP I/O via ``shutdown(SHUT_RDWR)`` WITHOUT closing FDs.

    ``close()`` from a non-owner thread is unsafe: the SSL BIO caches the raw
    FD, the kernel recycles it, and a flushed TLS record lands in the wrong
    file (#29507 clobbered a SQLite header). ``shutdown()`` is FD-safe from
    any thread; the owning httpx thread releases the FD on unwind.

    Returns the number of sockets shut down (logged as ``tcp_force_closed=N``
    for backwards-compatible parsing).
    """
    import socket as _socket

    shutdown_count = 0
    try:
        for sock in _iter_pool_sockets(client):
            try:
                # Clear a blocking timeout so a hung SSL_read notices the shutdown (#85252).
                # Still no close() — that is the #29507 race.
                settimeout = getattr(sock, "settimeout", None)
                if callable(settimeout):
                    try:
                        settimeout(0)
                    except OSError:
                        pass
                sock.shutdown(_socket.SHUT_RDWR)
            except OSError:
                # Already shut down / not connected / FD invalid — all benign.
                pass
            # IMPORTANT (#29507): do NOT call sock.close() here. See docstring.
            shutdown_count += 1
    except Exception as exc:
        _ra().logger.debug("Force-close TCP sockets sweep error: %s", exc)
    return shutdown_count



__all__ = [
    "convert_to_trajectory_format",
    "sanitize_tool_call_arguments",
    "repair_message_sequence",
    "strip_think_blocks",
    "recover_with_credential_pool",
    "try_recover_primary_transport",
    "drop_thinking_only_and_merge_users",
    "restore_primary_runtime",
    "extract_reasoning",
    "dump_api_request_debug",
    "prompt_caching_disabled_from_config",
    "blank_cache_policy_stub",
    "plan_cache_sections_for_destination",
    "anthropic_prompt_cache_policy",
    "create_openai_client",
    "switch_model",
    "invoke_tool",
    "repair_tool_call",
    "sanitize_api_messages",
    "looks_like_codex_intermediate_ack",
    "copy_reasoning_content_for_api",
    "cleanup_dead_connections",
    "extract_api_error_context",
    "apply_pending_steer_to_tool_results",
    "_iter_pool_sockets",
    "force_close_tcp_sockets",
]
