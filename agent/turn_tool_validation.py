"""Tool-call validation for the conversation turn loop: unknown tool names (with
auto-repair and the 3-strike partial exit) and malformed JSON arguments (retry, then
recovery tool results).

Extracted from ``run_conversation``. Role alternation is preserved on every path: an
invalid batch is answered with tool-role error results (never a user message), and
the exits close any open tool-result tail (#48879). Nothing here imports
``agent.conversation_loop`` at module level (cycle); loop-internal helpers resolve lazily.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from agent.message_metadata import append_message
from agent.message_sanitization import close_interrupted_tool_sequence, coalesce_tool_call_id

logger = logging.getLogger("agent.conversation_loop")


@dataclass
class ToolValidationVerdict:
    """Outcome of ``validate_tool_calls``.

    ``action``: ``"ok"`` (dispatch the calls), ``"continue"`` (re-issue the API call —
    error results / retry state were recorded) or ``"return"`` (terminal partial
    result in ``result``). ``mixed_invalid_batch`` is True when the batch contains BOTH
    valid and unknown tool names: only the invalid calls get error results, the valid
    ones run."""

    action: str
    result: Optional[Dict[str, Any]]
    mixed_invalid_batch: bool


def validate_tool_calls(
    agent: Any,
    assistant_message: Any,
    finish_reason: str,
    *,
    messages: List[Dict[str, Any]],
    conversation_history: Any,
    api_call_count: int,
    effective_task_id: Any,
) -> ToolValidationVerdict:
    """Validate ``assistant_message.tool_calls`` in place (ids uniquified, names
    repaired, dict/empty args normalized to JSON strings). Strikes for invalid names
    advance only when a turn has NO valid call, so a degenerate model still halts at
    3; args cut off mid-stream (routers rewrite ``length`` → ``tool_calls``) are refused
    outright rather than retried."""
    from agent.conversation_loop import _invalid_tool_name_error_content

    _mixed_invalid_batch = False

    def _verdict(action: str, result: Optional[Dict[str, Any]] = None) -> ToolValidationVerdict:
        return ToolValidationVerdict(action=action, result=result, mixed_invalid_batch=_mixed_invalid_batch)

    # Uniquify duplicate tool-call ids BEFORE any downstream consumer: the
    # pre-API sanitizer keeps only the first call/result per id. See
    # _uniquify_tool_call_ids.
    agent._uniquify_tool_call_ids(assistant_message.tool_calls)

    # Validate tool call names - detect model hallucinations
    # Repair mismatched tool names before validating
    for tc in assistant_message.tool_calls:
        if tc.function.name not in agent.valid_tool_names:
            repaired = agent._repair_tool_call(tc.function.name)
            if repaired:
                print(f"{agent.log_prefix}🔧 Auto-repaired tool name: '{tc.function.name}' -> '{repaired}'")
                tc.function.name = repaired
    invalid_tool_calls = [
        tc.function.name for tc in assistant_message.tool_calls
        if tc.function.name not in agent.valid_tool_names
    ]
    # Mixed batch: error-result ONLY the invalid calls and run the valid
    # ones; voiding the turn discards real work. Strikes advance only when a
    # turn has NO valid call, so a degenerate model still halts at 3.
    _mixed_invalid_batch = bool(invalid_tool_calls) and any(
        tc.function.name in agent.valid_tool_names
        for tc in assistant_message.tool_calls
    )
    if _mixed_invalid_batch:
        agent._invalid_tool_retries = 0
        invalid_name = invalid_tool_calls[0]
        invalid_preview = invalid_name[:80] + "..." if len(invalid_name) > 80 else invalid_name
        _n_valid = sum(
            1 for tc in assistant_message.tool_calls
            if tc.function.name in agent.valid_tool_names
        )
        agent._buffer_vprint(
            f"⚠️  Unknown tool '{invalid_preview}' in batch — erroring that call, "
            f"executing {_n_valid} valid call(s)"
        )
    elif invalid_tool_calls:
        # Track retries for invalid tool calls
        agent._invalid_tool_retries += 1

        # Return helpful error to model — model can agent-correct next turn
        invalid_name = invalid_tool_calls[0]
        invalid_preview = invalid_name[:80] + "..." if len(invalid_name) > 80 else invalid_name
        agent._buffer_vprint(f"⚠️  Unknown tool '{invalid_preview}' — sending error to model for agent-correction ({agent._invalid_tool_retries}/3)")

        if agent._invalid_tool_retries >= 3:
            agent._flush_status_buffer()
            agent._vprint(f"{agent.log_prefix}❌ Max retries (3) for invalid tool calls exceeded. Stopping as partial.", force=True)
            agent._invalid_tool_retries = 0
            _final_response = f"Model generated invalid tool call: {invalid_preview}"
            # Prior retries or an earlier tool batch leave a tool-result
            # tail; close it as interrupt aborts do so the next turn is not
            # tool→user. (#48879)
            close_interrupted_tool_sequence(messages, _final_response)
            agent._persist_session(messages, conversation_history)
            return _verdict("return", {
                "final_response": _final_response,
                "messages": messages,
                "api_calls": api_call_count,
                "completed": False,
                "partial": True,
                "error": _final_response
            })

        assistant_msg = agent._build_assistant_message(assistant_message, finish_reason)
        append_message(messages, assistant_msg)
        for tc in assistant_message.tool_calls:
            _tc_name = tc.function.name
            if _tc_name not in agent.valid_tool_names:
                # See _invalid_tool_name_error_content for the
                # blank-name anti-priming rationale (#47967).
                content = _invalid_tool_name_error_content(
                    _tc_name, agent.valid_tool_names
                )
            else:
                content = "Skipped: another tool call in this turn used an invalid name. Please retry this tool call."
            append_message(messages, {
                "role": "tool",
                "name": tc.function.name,
                "tool_call_id": coalesce_tool_call_id(tc),
                "content": content,
            })
        return _verdict("continue")
    # Reset retry counter on successful tool call validation
    agent._invalid_tool_retries = 0

    # Validate tool call arguments are valid JSON
    # Handle empty strings as empty objects (common model quirk)
    invalid_json_args = []
    for tc in assistant_message.tool_calls:
        args = tc.function.arguments
        if isinstance(args, (dict, list)):
            tc.function.arguments = json.dumps(args)
            continue
        if args is not None and not isinstance(args, str):
            tc.function.arguments = str(args)
            args = tc.function.arguments
        # Treat empty/whitespace strings as empty object
        if not args or not args.strip():
            tc.function.arguments = "{}"
            continue
        try:
            json.loads(args)
        except json.JSONDecodeError as e:
            if (
                _mixed_invalid_batch
                and tc.function.name not in agent.valid_tool_names
            ):
                # This call never executes (invalid-name error result
                # below); don't let its broken args trigger the whole-turn
                # JSON retry.
                continue
            invalid_json_args.append((tc.function.name, str(e)))

    if invalid_json_args:
        # Routers may rewrite finish_reason "length" → "tool_calls", hiding
        # truncation; args not ending in } or ] (stripped) were cut off
        # mid-stream.
        _truncated = any(
            not (tc.function.arguments or "").rstrip().endswith(("}", "]"))
            for tc in assistant_message.tool_calls
            if tc.function.name in {n for n, _ in invalid_json_args}
        )
        if _truncated:
            agent._vprint(
                f"{agent.log_prefix}⚠️  Truncated tool call arguments detected "
                f"(finish_reason={finish_reason!r}) — refusing to execute.",
                force=True,
            )
            agent._invalid_json_retries = 0
            agent._cleanup_task_resources(effective_task_id)
            _final_response = "Response truncated due to output length limit"
            # Same tool-tail close as interrupt / invalid-tool
            # exhaustion — this path never reaches finalize_turn.
            close_interrupted_tool_sequence(messages, _final_response)
            agent._persist_session(messages, conversation_history)
            return _verdict("return", {
                "final_response": _final_response,
                "messages": messages,
                "api_calls": api_call_count,
                "completed": False,
                "partial": True,
                "error": _final_response,
            })

        # Track retries for invalid JSON arguments
        agent._invalid_json_retries += 1

        tool_name, error_msg = invalid_json_args[0]
        agent._buffer_vprint(f"⚠️  Invalid JSON in tool call arguments for '{tool_name}': {error_msg}")

        if agent._invalid_json_retries < 3:
            agent._buffer_vprint(f"🔄 Retrying API call ({agent._invalid_json_retries}/3)...")
            # Don't add anything to messages, just retry the API call
            return _verdict("continue")
        else:
            # Instead of returning partial, inject tool error results so the model can recover.
            # Using tool results (not user messages) preserves role alternation.
            agent._buffer_vprint("⚠️  Injecting recovery tool results for invalid JSON...")
            agent._invalid_json_retries = 0  # Reset for next attempt

            # Append the assistant message with its (broken) tool_calls
            recovery_assistant = agent._build_assistant_message(assistant_message, finish_reason)
            append_message(messages, recovery_assistant)

            # Respond with tool error results for each tool call
            invalid_names = {name for name, _ in invalid_json_args}
            for tc in assistant_message.tool_calls:
                if tc.function.name in invalid_names:
                    err = next(e for n, e in invalid_json_args if n == tc.function.name)
                    tool_result = (
                        f"Error: Invalid JSON arguments. {err}. "
                        f"For tools with no required parameters, use an empty object: {{}}. "
                        f"Please retry with valid JSON."
                    )
                else:
                    tool_result = "Skipped: other tool call in this response had invalid JSON."
                append_message(messages, {
                    "role": "tool",
                    "name": tc.function.name,
                    "tool_call_id": coalesce_tool_call_id(tc),
                    "content": tool_result,
                })
            return _verdict("continue")

    # Reset retry counter on successful JSON validation
    agent._invalid_json_retries = 0
    return _verdict("ok")
