"""Text-response stop gates for the conversation turn loop.

Extracted from ``run_conversation``. When the model stops with a text answer, three
gates may instead append the answer as an interim row plus a synthetic user-role nudge
and continue the turn: verify-on-stop (#65919), the ``pre_verify`` plugin hook after code
edits, and the kanban worker terminal-tool guard. Each keeps the candidate answer as a
budget-exhaustion fallback (``pending_verification_response``) and clears
``final_response`` so the finalizer can tell this gate from error exits (#61631).
Nothing here imports ``agent.conversation_loop`` at module level (cycle).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List

from agent.message_metadata import append_message

logger = logging.getLogger("agent.conversation_loop")


@dataclass
class StopGateVerdict:
    """``continue_turn`` True → a nudge was appended; re-enter the turn loop with
    ``final_response=None`` and the pending-verification fields updated."""

    continue_turn: bool
    final_response: Any
    pending_verification_response: Any
    pending_verification_response_previewed: Any


def apply_stop_gates(
    agent: Any, final_msg: Dict[str, Any], *, final_response: Any, messages: List[Dict[str, Any]],
    conversation_history: Any, pending_verification_response: Any,
    pending_verification_response_previewed: Any,
) -> StopGateVerdict:
    """Run verify-on-stop → pre_verify hook → kanban stop guard, in that order. Nudges
    are user-role rows appended only after the assistant answer row, so role alternation
    holds. Hook lookups are imported lazily from their origin modules (tests patch them
    there)."""
    _pending_verification_response = pending_verification_response
    _pending_verification_response_previewed = pending_verification_response_previewed

    def _verdict(continue_turn: bool) -> StopGateVerdict:
        return StopGateVerdict(
            continue_turn=continue_turn, final_response=None if continue_turn else final_response,
            pending_verification_response=_pending_verification_response,
            pending_verification_response_previewed=_pending_verification_response_previewed,
        )

    try:
        from agent.verification_stop import (
            build_verify_on_stop_nudge, verify_on_stop_enabled
        )

        if verify_on_stop_enabled():
            _verify_nudge = build_verify_on_stop_nudge(
                session_id=getattr(agent, "session_id", None),
                changed_paths=getattr(agent, "_turn_file_mutation_paths", set()),
                attempts=getattr(agent, "_verification_stop_nudges", 0),
            )
        else:
            _verify_nudge = None
    except Exception:
        logger.debug("verification stop-loop check failed", exc_info=True)
        _verify_nudge = None

    if _verify_nudge:
        agent._verification_stop_nudges = (
            getattr(agent, "_verification_stop_nudges", 0) + 1
        )
        final_msg["finish_reason"] = "verification_required"
        # Real content: persist and emit as interim so the user sees the
        # attempted answer; only the nudge is flagged synthetic. (#65919)
        agent._emit_interim_assistant_message(final_msg)
        append_message(messages, final_msg)
        try:
            agent._flush_messages_to_session_db(messages, conversation_history)
        except Exception:
            logger.debug("verify-on-stop interim flush failed", exc_info=True)
        append_message(messages, {
            "role": "user", "content": _verify_nudge, "_verification_stop_synthetic": True
        })
        agent._session_messages = messages
        # Internal nudge: stay silent on the terminal, debug-log only.
        logger.debug("verification stop-loop nudge issued (attempt %d)",
                     agent._verification_stop_nudges)
        # Keep the answer only as a budget-exhaustion fallback; clear
        # ``final_response`` so the finalizer can tell this gate from error
        # exits. Mark previewed only if the candidate is reused. (#61631)
        _pending_verification_response = final_response
        _pending_verification_response_previewed = (
            agent._interim_content_was_streamed(final_response or "")
        )
        return _verdict(True)

    # pre_verify hook gate: after code edits a registered hook may keep the
    # agent going one more turn; no default continuation cost.
    _verify_nudge2 = None
    _edited = sorted(getattr(agent, "_turn_file_mutation_paths", set()) or [])
    _attempt = getattr(agent, "_pre_verify_nudges", 0)
    try:
        from agent.verify_hooks import max_verify_nudges
        from hermes_cli.lifecycle import has_hook
        from hermes_cli.plugins import get_pre_verify_continue_message

        if _edited and has_hook("pre_verify") and _attempt < max_verify_nudges():
            # Posture is fixed for the session — resolve once + cache.
            coding = getattr(agent, "_resolved_is_coding", None)
            if coding is None:
                from agent.coding_context import is_coding_context
                coding = bool(is_coding_context(platform=getattr(agent, "platform", "") or ""))
                agent._resolved_is_coding = coding
            _verify_nudge2 = get_pre_verify_continue_message(
                session_id=getattr(agent, "session_id", None) or "",
                platform=getattr(agent, "platform", "") or "",
                model=getattr(agent, "model", "") or "", coding=coding, attempt=_attempt,
                final_response=final_response, changed_paths=_edited,
            )
    except Exception:
        logger.debug("pre_verify hook check failed", exc_info=True)
        _verify_nudge2 = None

    if _verify_nudge2:
        agent._pre_verify_nudges = _attempt + 1
        final_msg["finish_reason"] = "verify_hook_continue"
        # Real content: persist and emit as interim so the user sees the
        # attempted answer; only the nudge is flagged synthetic. (#65919)
        agent._emit_interim_assistant_message(final_msg)
        append_message(messages, final_msg)
        try:
            agent._flush_messages_to_session_db(messages, conversation_history)
        except Exception:
            logger.debug("pre_verify interim flush failed", exc_info=True)
        append_message(messages, {
            "role": "user", "content": _verify_nudge2, "_pre_verify_synthetic": True
        })
        agent._session_messages = messages
        logger.debug("pre_verify nudge issued (attempt %d)",
                     agent._pre_verify_nudges)
        _pending_verification_response = final_response
        _pending_verification_response_previewed = (
            agent._interim_content_was_streamed(final_response or "")
        )
        return _verdict(True)

    # ── Kanban worker terminal-tool stop guard ─────────────
    # Workers must end with kanban_complete / kanban_block; a narrated stop
    # is recorded as protocol_violation, so nudge once or twice first.
    try:
        from agent.kanban_stop import build_kanban_stop_nudge

        _kanban_nudge = build_kanban_stop_nudge(
            messages=messages, attempts=getattr(agent, "_kanban_stop_nudges", 0)
        )
    except Exception:
        logger.debug("kanban stop-loop check failed", exc_info=True)
        _kanban_nudge = None

    if _kanban_nudge:
        agent._kanban_stop_nudges = (
            getattr(agent, "_kanban_stop_nudges", 0) + 1
        )
        final_msg["finish_reason"] = "kanban_terminal_required"
        final_msg["_kanban_stop_synthetic"] = True
        append_message(messages, final_msg)
        append_message(messages, {
            "role": "user", "content": _kanban_nudge, "_kanban_stop_synthetic": True
        })
        agent._session_messages = messages
        logger.info(
            "kanban stop-loop nudge issued (attempt %d) task=%s",
            agent._kanban_stop_nudges,
            os.environ.get("HERMES_KANBAN_TASK", ""),
        )
        agent._emit_status(
            "⚠️ Kanban worker tried to exit without "
            "kanban_complete/kanban_block — nudging to finish"
        )
        # Same finalizer contract as verify-on-stop: clear final_response so
        # budget exhaustion doesn't treat the narrated stop as an answer.
        _pending_verification_response = final_response
        _pending_verification_response_previewed = (
            agent._interim_content_was_streamed(final_response or "")
        )
        return _verdict(True)
    return _verdict(False)
