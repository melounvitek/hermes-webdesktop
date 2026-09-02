"""Durable transcript persistence for ``AIAgent``.

SQLite session flush (intrinsic ``_DB_PERSISTED_MARKER`` dedup), ephemeral-scaffolding filtering, the JSON
session log and trajectory export.
Extracted from ``run_agent.py``; every method resolves through ``AIAgent``'s MRO unchanged.
"""
import hashlib
import json
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from agent.context_compressor import (
    COMPRESSED_SUMMARY_METADATA_KEY,
    _DB_PERSISTED_MARKER,
    ContextCompressor,
    user_originated_turn_view,
)
from agent.lazy_forward import forward as _forward, forward_static as _forward_static
from agent.memory_manager import sanitize_context
from agent.redact import redact_sensitive_text
from agent.tool_dispatch_helpers import _is_multimodal_tool_result, _multimodal_text_summary
from agent.trajectory import convert_scratchpad_to_think, save_trajectory as _save_trajectory_to_file
from utils import atomic_json_write

# Same logger name as the origin module so log records / caplog filters are unchanged.
logger = logging.getLogger("run_agent")


# Flags marking ephemeral empty-response/prefill recovery scaffolding. The loop pops these before
# appending the real response; persistence must skip them or a resumed session replays synthetic turns.
_EPHEMERAL_SCAFFOLDING_FLAGS = (
    "_empty_recovery_synthetic",
    "_empty_terminal_sentinel",
    "_thinking_prefill",
    # verify-on-stop / pre_verify nudges: persisting them poisons the resumed transcript and breaks
    # prompt-prefix cache reuse. The assistant candidate is NOT synthetic (#65919).
    "_verification_stop_synthetic",
    "_pre_verify_synthetic",
    # kanban worker stop-guard: narrated exit without kanban_complete/block
    "_kanban_stop_synthetic",
    # dropped tool-call re-prompt pair: internal retry instruction, must not replay as user context on resume.
    "_dropped_toolcall_nudge",
)


def _is_ephemeral_scaffolding(msg: Any) -> bool:
    """Return True when ``msg`` is internal recovery scaffolding that must never be persisted to the
    durable transcript (SQLite session store or JSON log)."""
    return isinstance(msg, dict) and any(
        msg.get(flag) for flag in _EPHEMERAL_SCAFFOLDING_FLAGS
    )


# `_DB_PERSISTED_MARKER` (agent.context_compressor) — intrinsic "already written to SQLite" marker. An id(msg)
# dedup set can alias a freed dict's address onto a new message and silently skip persisting it; a marker on
# the dict cannot. The `_` prefix is mandatory: wire sanitizers strip `_`-prefixed keys. CONTRACT (#92231):
# the marker asserts the dict's CONTENT is durable as written — any in-place mutation that must persist MUST
# pop it (see turn_finalizer, context_compressor).


def _safe_session_filename_component(session_id: str) -> str:
    """Return a stable, path-safe filename component for a session ID.

    Session IDs may be untrusted (``X-Hermes-Session-Id``) and are interpolated into ``~/.hermes/sessions/``
    filenames. Collapses non ``[A-Za-z0-9_-]`` chars to ``_``, caps length, and appends a short content
    hash when sanitization changed the string so distinct IDs cannot collide.
    """
    raw = str(session_id or "").strip()
    sanitized = re.sub(r"[^\w-]", "_", raw).strip("._")
    sanitized = sanitized[:96] or "session"
    if raw and sanitized == raw:
        return sanitized
    digest = hashlib.sha256(
        raw.encode("utf-8", errors="surrogatepass")
    ).hexdigest()[:12]
    return f"{sanitized}_{digest}"


class SessionPersistenceMixin:
    """Session DB flush, session log and trajectory persistence (see module docstring)."""

    def _apply_persist_user_message_override(self, messages: List[Dict]) -> None:
        """Rewrite the current-turn user message before persistence/return.

        Some paths use an API-only user-message variant that must not leak into transcripts or resumed
        history; mutate the in-memory list in place so both persistence and returned history stay clean.
        """
        idx = getattr(self, "_persist_user_message_idx", None)
        override = getattr(self, "_persist_user_message_override", None)
        timestamp = getattr(self, "_persist_user_message_timestamp", None)
        platform_id = getattr(self, "_persist_user_message_platform_id", None)
        if idx is None or (
            override is None and timestamp is None and platform_id is None
        ):
            return
        if 0 <= idx < len(messages):
            msg = messages[idx]
            if isinstance(msg, dict) and msg.get("role") == "user":
                # A plain-text override must not replace native image/audio blocks; a list override is the
                # clean multimodal payload and does. Preflight compaction may re-anchor this index at a
                # message MERGED with the compaction summary — overwriting it would drop the summary (see the
                # twin guard in _flush_messages_to_session_db_unlocked).
                if (
                    override is not None
                    and not msg.get(COMPRESSED_SUMMARY_METADATA_KEY)
                    and (
                        not isinstance(msg.get("content"), list)
                        or isinstance(override, list)
                    )
                ):
                    msg["content"] = override
                if timestamp is not None:
                    msg["timestamp"] = timestamp
                # Platform message id: load-bearing for restart drain-window recovery dedup
                # (has_platform_message_id). Stamped here too so it survives the override path.
                if platform_id is not None:
                    msg["platform_message_id"] = platform_id

    def _persist_session(self, messages: List[Dict], conversation_history: List[Dict] = None):
        """Save session state to both JSON log and SQLite on any exit path.

        Trailing empty-response scaffolding is dropped from the live list. The persist user-message override
        is NOT applied here — ``_flush_messages_to_session_db`` writes it to the DB row only.
        """
        # Scaffolding removal mutates the live list on purpose. Close and turn-start persistence can run on
        # separate CLI threads, so the marker test-and-append below must be one critical section.
        from agent.agent_runtime_helpers import note_turn_persisted

        persist_lock = getattr(self, "_session_persist_lock", None)

        def _persist_and_drain() -> None:
            self._drop_trailing_empty_response_scaffolding(messages)
            self._session_messages = messages
            self._save_session_log(messages)
            self._flush_messages_to_session_db(messages, conversation_history)
            # Drain async token-accounting deltas at every persist point; cheap no-op when nothing queued.
            if self._session_db is not None:
                self._session_db.flush_token_counts()
            note_turn_persisted(self)

        if persist_lock is None:
            _persist_and_drain()
            return

        with persist_lock:
            _persist_and_drain()

    def _drop_trailing_empty_response_scaffolding(self, messages: List[Dict]) -> None:
        """Remove private empty-response retry/failure scaffolding from transcript tails.

        Also rewinds a trailing tool-result / assistant(tool_calls) pair the failed iteration left hanging;
        otherwise the next user turn lands as ``...tool, user`` and providers return empty content forever.
        """
        # Pass 1: strip the flagged scaffolding messages themselves.
        dropped_scaffolding = False
        while (
            messages
            and isinstance(messages[-1], dict)
            and (
                messages[-1].get("_empty_recovery_synthetic")
                or messages[-1].get("_empty_terminal_sentinel")
            )
        ):
            messages.pop()
            dropped_scaffolding = True

        # Pass 2: after stripping scaffolding, rewind trailing tool results and the assistant(tool_calls)
        # that produced them, so role alternation holds. Only runs when scaffolding was present.
        if not dropped_scaffolding:
            return

        # Drop any trailing tool-result messages
        while (
            messages
            and isinstance(messages[-1], dict)
            and messages[-1].get("role") == "tool"
        ):
            messages.pop()

        # Drop the assistant(tool_calls) whose results were just popped — providers reject a dangling one.
        if (
            messages
            and isinstance(messages[-1], dict)
            and messages[-1].get("role") == "assistant"
            and messages[-1].get("tool_calls")
        ):
            messages.pop()

    _repair_message_sequence = _forward("agent.agent_runtime_helpers", "repair_message_sequence")

    def _flush_messages_to_session_db(
        self,
        messages: List[Dict],
        conversation_history: Optional[List[Dict]] = None,
    ):
        """Serialize direct and turn-boundary session flushes per agent."""
        persist_lock = getattr(self, "_session_persist_lock", None)
        if persist_lock is None:
            return self._flush_messages_to_session_db_unlocked(messages, conversation_history)
        with persist_lock:
            return self._flush_messages_to_session_db_unlocked(messages, conversation_history)

    def _flush_messages_to_session_db_unlocked(
        self,
        messages: List[Dict],
        conversation_history: Optional[List[Dict]] = None,
        _adoption_budget: int = 1,
    ):
        """Persist any un-flushed messages to the SQLite session store.

        Dedup is an intrinsic ``_DB_PERSISTED_MARKER`` on each written dict — not positional slices (drift
        after sequence repair) nor a retained ``id(msg)`` set (address reuse). ``_flushed_db_message_ids`` is
        only a one-shot seed translated to markers and cleared each flush.
        """
        # Persistence-isolated agents (background review fork) share the parent's session_id for cache
        # warmth; a write here would land the curator's harness turn in the user's real history. Hard-stop.
        if getattr(self, "_persist_disabled", False):
            return None
        if not self._session_db:
            return None
        # Persist user-message override (#48677): resolved here and applied ONLY to the written row, never
        # to the live dict — the early crash-resilience persist runs before the API call is built.
        _ov_idx = getattr(self, "_persist_user_message_idx", None)
        _ov_content = getattr(self, "_persist_user_message_override", None)
        _ov_timestamp = getattr(self, "_persist_user_message_timestamp", None)
        try:
            # Retry row creation if the earlier attempt failed transiently.
            if not self._session_db_created:
                self._ensure_db_session()
            # Positional slicing broke when repair_message_sequence shrank the list (#46053). Persistence is
            # tracked by an intrinsic per-message marker (see _DB_PERSISTED_MARKER); `_flushed_db_message_ids`
            # is honoured only as a one-shot seed translated to markers and then cleared.
            current_session_id = getattr(self, "session_id", None)
            flushed_session_id = getattr(self, "_flushed_db_message_session_id", None)
            if flushed_session_id != current_session_id or self._last_flushed_db_idx == 0:
                seed_ids = set()
            else:
                seed_ids = getattr(self, "_flushed_db_message_ids", None)
                if not isinstance(seed_ids, set):
                    seed_ids = set()
            self._flushed_db_message_session_id = current_session_id
            history_ids = {
                id(item) for item in (conversation_history or [])
                if isinstance(item, dict)
            }

            # Bounded scan: skip the identity-matched prefix of the previous flush's snapshot. Every message
            # in it already got its final disposition, and no live dict has its marker popped in place.
            _scan_start = 0
            _prev_prefix = getattr(self, "_db_flush_scan_prefix", None)
            if isinstance(_prev_prefix, list):
                _limit = min(len(_prev_prefix), len(messages))
                while (
                    _scan_start < _limit
                    and messages[_scan_start] is _prev_prefix[_scan_start]
                    and bool(messages[_scan_start].get(_DB_PERSISTED_MARKER))
                ):
                    _scan_start += 1

            # Collect this flush's new rows and write them in ONE transaction
            # at the end of the scan (see append_messages_batch).
            _batch_rows: List[Dict[str, Any]] = []
            _batch_msgs: List[Dict] = []
            for _msg_idx in range(_scan_start, len(messages)):
                msg = messages[_msg_idx]
                if not isinstance(msg, dict):
                    continue
                # Never write ephemeral scaffolding: the flush is append-only, so a mid-turn persist could
                # commit a synthetic turn that the end-of-turn drop cannot un-write. Skip regardless of
                # position.
                if _is_ephemeral_scaffolding(msg):
                    continue
                if msg.get(_DB_PERSISTED_MARKER):
                    continue
                # Already-durable (history copy or caller-seeded): stamp so future flushes skip without id()
                # sets.
                if id(msg) in history_ids or id(msg) in seed_ids:
                    msg[_DB_PERSISTED_MARKER] = True
                    continue
                role = msg.get("role", "unknown")
                content = msg.get("content")
                # api_content sidecar: exact bytes sent to the API when they differ from clean content, so
                # replay reproduces the sent prefix byte-for-byte.
                _row_api_content = msg.get("api_content")
                if not isinstance(_row_api_content, str):
                    _row_api_content = None
                _row_timestamp = msg.get("timestamp")
                # Apply the persist override to THIS row only. A list override replaces a noted payload; a
                # text override must not erase an image/audio summary. Also match the staged CLI dict by
                # identity — the close safety-net may flush a shortened snapshot whose turn index refers to
                # the full history.
                pending_cli_message = getattr(self, "_pending_cli_user_message", None)
                is_current_turn_user = (
                    _ov_idx == _msg_idx or msg is pending_cli_message
                )
                if is_current_turn_user and msg.get("role") == "user":
                    # Preflight compaction may have re-anchored the index at a message MERGED with the
                    # compaction summary; overwriting it with the clean text would drop the summary from the
                    # durable transcript.
                    if (
                        _ov_content is not None
                        and (not isinstance(content, list) or isinstance(_ov_content, list))
                        and not msg.get(COMPRESSED_SUMMARY_METADATA_KEY)
                    ):
                        # Live content is what the wire sent, the override is the clean transcript; keep the
                        # sent bytes in api_content so replay matches the wire (#48677).
                        if (
                            _row_api_content is None
                            and isinstance(content, str)
                            and content != _ov_content
                        ):
                            _row_api_content = content
                        content = _ov_content
                    if _ov_timestamp is not None:
                        _row_timestamp = _ov_timestamp
                # Store the sidecar only when it actually differs.
                if _row_api_content == content:
                    _row_api_content = None
                # Load-time sanitize divergence: get_messages_as_conversation replays rows through
                # sanitize_context().strip(); capture the sent bytes when they would differ (compared in wire
                # form).
                if (
                    _row_api_content is None
                    and role in ("user", "assistant")
                    and isinstance(content, str)
                    and content
                    and sanitize_context(content).strip() != content.strip()
                ):
                    _row_api_content = content
                # Persist multimodal tool results as text summary only — base64 images bloat the DB.
                if _is_multimodal_tool_result(content):
                    content = _multimodal_text_summary(content)
                elif isinstance(content, list):
                    # List of OpenAI-style content parts: strip images, keep text.
                    _txt = []
                    for p in content:
                        if isinstance(p, dict) and p.get("type") == "text":
                            _txt.append(str(p.get("text", "")))
                        elif isinstance(p, dict) and p.get("type") in {"image", "image_url", "input_image"}:
                            _txt.append("[screenshot]")
                    content = "\n".join(_txt) if _txt else None
                tool_calls_data = None
                if hasattr(msg, "tool_calls") and isinstance(msg.tool_calls, list) and msg.tool_calls:
                    tool_calls_data = [
                        {"name": tc.function.name, "arguments": tc.function.arguments}
                        for tc in msg.tool_calls
                    ]
                elif isinstance(msg.get("tool_calls"), list):
                    tool_calls_data = msg["tool_calls"]
                _row = {
                    "role": role,
                    "content": content,
                    "tool_name": msg.get("tool_name"),
                    "tool_calls": tool_calls_data,
                    "tool_call_id": msg.get("tool_call_id"),
                    "finish_reason": msg.get("finish_reason"),
                    # Reasoning/codex fields are role-gated (assistant-only)
                    # inside _insert_message_rows — pass through untouched.
                    "reasoning": msg.get("reasoning"),
                    "reasoning_content": msg.get("reasoning_content"),
                    "reasoning_details": msg.get("reasoning_details"),
                    "codex_reasoning_items": msg.get("codex_reasoning_items"),
                    "codex_message_items": msg.get("codex_message_items"),
                    "_compressed_summary": bool(msg.get(COMPRESSED_SUMMARY_METADATA_KEY)),
                    "timestamp": _row_timestamp,
                    "api_content": _row_api_content,
                    # Standalone reference handoffs are always hidden so they never occupy the active user
                    # slot in retry/undo dispatch (#80622); merge-into-tail carriers keep prior visibility.
                    "display_kind": (
                        "hidden"
                        if (
                            msg.get(COMPRESSED_SUMMARY_METADATA_KEY)
                            and user_originated_turn_view(msg) is None
                            and (
                                ContextCompressor.classify_summary_content(
                                    msg.get("content")
                                )
                                == "standalone"
                                or not msg.get(
                                    "_compressed_summary_has_user_turn"
                                )
                            )
                        )
                        else msg.get("display_kind")
                    ),
                    "display_metadata": msg.get("display_metadata"),
                    # Platform message id — load-bearing for restart drain-window recovery dedup.
                    "platform_message_id": msg.get("platform_message_id"),
                }
                if isinstance(msg.get("_row_id"), int):
                    _row["_row_id"] = msg["_row_id"]
                _batch_rows.append(_row)
                _batch_msgs.append(msg)
            # One transaction for the turn's new rows. All-or-nothing pairs with the marker stamping below:
            # on failure no rows landed and no markers were stamped, so the next flush re-writes the tail.
            if _batch_rows:
                self._session_db.append_messages_batch(
                    session_id=self.session_id,
                    messages=_batch_rows,
                    compression_lock_holder=getattr(
                        self, "_active_compression_lock_holder", None
                    ),
                    turn_lease_holder=getattr(
                        self, "_active_session_turn_lease_holder", None
                    ),
                    turn_lease_ttl_seconds=getattr(
                        self, "_active_session_turn_lease_ttl_seconds", 300.0
                    )
                    or 300.0,
                )
                from agent.transcript_repair import sync_flushed_message_markers

                sync_flushed_message_markers(_batch_msgs, _batch_rows)
            # Markers are now the sole truth; reset the one-shot seed so no id() outlives this flush.
            self._flushed_db_message_ids = set()
            self._last_flushed_db_idx = len(messages)
            # Snapshot for the bounded scan above — only on full success, so
            # a partially-processed list can never be treated as settled.
            self._db_flush_scan_prefix = messages[:]
            return True
        except Exception as e:
            # Force a full re-scan on the next flush: an exception mid-loop
            # leaves messages with mixed dispositions.
            self._db_flush_scan_prefix = None
            # The only place the SQLite error is visible before it becomes a bare False — classify it so the
            # turn-end explanation can distinguish lock contention from disk-full/read-only.
            from hermes_state import (
                CompressionSessionClosedError,
                StateDbCorruptError,
                StateDbReplacedError,
                classify_persistence_error,
                divert_session_transcript_jsonl,
            )

            self._last_persistence_error_cause = classify_persistence_error(e)
            if isinstance(e, (StateDbReplacedError, StateDbCorruptError)):
                # Replaced/quarantined handle will not take this batch again — keep it on disk, not only in
                # RAM.
                try:
                    divert_session_transcript_jsonl(
                        getattr(self, "session_id", "") or "",
                        _batch_rows,
                    )
                except Exception:
                    logger.warning(
                        "JSONL divert failed after state.db %s for %s",
                        self._last_persistence_error_cause,
                        getattr(self, "session_id", None),
                        exc_info=True,
                    )
            if isinstance(e, CompressionSessionClosedError):
                # Compression race: another path rotated this session mid-write. Adopt the continuation tip
                # (get_compression_tip) ONLY when it is a different, live row, and retry exactly once; a
                # second closed-parent write fails closed. tip == session_id means no continuation exists.
                if _adoption_budget > 0:
                    old_id = self.session_id
                    tip = None
                    try:
                        tip = self._session_db.get_compression_tip(old_id)
                    except Exception as tip_exc:
                        logger.warning(
                            "compression tip lookup failed for %s: %s",
                            old_id,
                            tip_exc,
                        )
                    if tip and tip != old_id:
                        tip_row = None
                        try:
                            tip_row = self._session_db.get_session(tip)
                        except Exception:
                            tip_row = None
                        if tip_row is not None and tip_row.get("ended_at") is None:
                            logger.warning(
                                "Adopted live compression tip %s for closed "
                                "session %s; retrying flush once",
                                tip,
                                old_id,
                            )
                            self.session_id = tip
                            self._flushed_db_message_ids = set()
                            self._last_flushed_db_idx = 0
                            self._compression_adoption_failed = False
                            return self._flush_messages_to_session_db_unlocked(
                                messages,
                                conversation_history,
                                _adoption_budget=0,
                            )
                # No live tip or budget exhausted: fail closed. The flag lets the turn explanation name
                # compression rotation instead of misleading full-disk advice.
                self._compression_adoption_failed = True
                logger.warning("Session DB append_message failed: %s", e)
                return False
            logger.warning("Session DB append_message failed: %s", e)
            return False

    def _get_messages_up_to_last_assistant(self, messages: List[Dict]) -> List[Dict]:
        """Get messages up to (but not including) the last assistant turn.

        The rollback point when the final assistant message is incomplete or malformed.
        """
        if not messages:
            return []

        # Find the index of the last assistant message
        last_assistant_idx = None
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "assistant":
                last_assistant_idx = i
                break

        if last_assistant_idx is None:
            # No assistant message found, return all messages
            return messages.copy()

        # Return everything up to (not including) the last assistant message
        return messages[:last_assistant_idx]

    _format_tools_for_system_message = _forward("agent.system_prompt", "format_tools_for_system_message")

    _convert_to_trajectory_format = _forward("agent.agent_runtime_helpers", "convert_to_trajectory_format")

    def _save_trajectory(self, messages: List[Dict[str, Any]], user_query: str, completed: bool):
        """Save conversation trajectory to JSONL file."""
        if not self.save_trajectories:
            return

        trajectory = self._convert_to_trajectory_format(messages, user_query, completed)
        _save_trajectory_to_file(trajectory, self.model, completed)

    _extract_api_error_context = _forward_static("agent.agent_runtime_helpers", "extract_api_error_context")

    _dump_api_request_debug = _forward("agent.agent_runtime_helpers", "dump_api_request_debug")

    @staticmethod
    def _clean_session_content(content: str) -> str:
        """Convert REASONING_SCRATCHPAD to think tags and clean up whitespace."""
        if not content:
            return content
        content = convert_scratchpad_to_think(content)
        content = re.sub(r'\n+(<think>)', r'\n\1', content)
        content = re.sub(r'(</think>)\n+', r'\1\n', content)
        return content.strip()

    @staticmethod
    def _redact_message_content(content):
        """Apply secret redaction to message content (str or list-of-parts).

        Only text fields pass through ``redact_sensitive_text``; image/binary parts are untouched.
        No-op when ``HERMES_REDACT_SECRETS`` disables redaction.
        """
        if content is None:
            return content
        if isinstance(content, str):
            return redact_sensitive_text(content)
        if isinstance(content, list):
            redacted = []
            for part in content:
                if isinstance(part, dict):
                    part = dict(part)
                    if isinstance(part.get("text"), str):
                        part["text"] = redact_sensitive_text(part["text"])
                    if isinstance(part.get("content"), str):
                        part["content"] = redact_sensitive_text(part["content"])
                redacted.append(part)
            return redacted
        return content

    def _save_session_log(self, messages: List[Dict[str, Any]] = None):
        """Optional per-session JSON snapshot writer (``sessions.write_json_snapshots``, default False).

        state.db is canonical; this exists for external tooling reading ``session_{sid}.json``. Rewrites the
        full list after every persistence point, never overwriting a larger log with fewer messages.
        """
        if not getattr(self, "_session_json_enabled", False):
            return
        messages = messages or self._session_messages
        if not messages:
            return

        # Re-derive the path each call so /branch and /compress land in the right file. Session IDs can be
        # untrusted (X-Hermes-Session-Id) — sanitize to a single traversal-free segment.
        try:
            safe_sid = _safe_session_filename_component(self.session_id)
            log_file = self.logs_dir / f"session_{safe_sid}.json"
        except Exception:
            return

        try:
            cleaned = []
            for msg in messages:
                # Mirror the SQLite flush: ephemeral recovery scaffolding is
                # internal retry state, never durable transcript content.
                if _is_ephemeral_scaffolding(msg):
                    continue
                if msg.get("role") == "assistant" and msg.get("content"):
                    msg = dict(msg)
                    msg["content"] = self._clean_session_content(msg["content"])
                # Defence-in-depth: redact credentials from every message before persistence; respects
                # HERMES_REDACT_SECRETS via redact_sensitive_text (#19798, #19845).
                if "content" in msg:
                    msg = dict(msg)
                    msg["content"] = self._redact_message_content(msg.get("content"))
                cleaned.append(msg)

            # Never overwrite a larger session log with fewer messages (resumed agent with partial history).
            if log_file.exists():
                try:
                    existing = json.loads(log_file.read_text(encoding="utf-8"))
                    existing_count = existing.get("message_count", len(existing.get("messages", [])))
                    if existing_count > len(cleaned):
                        logging.debug(
                            "Skipping session log overwrite: existing has %d messages, current has %d",
                            existing_count, len(cleaned),
                        )
                        return
                except Exception:
                    pass  # corrupted existing file — allow the overwrite

            entry = {
                "session_id": self.session_id,
                "model": self.model,
                "base_url": self.base_url,
                "platform": self.platform,
                "session_start": self.session_start.isoformat(),
                "last_updated": datetime.now().isoformat(),
                "system_prompt": redact_sensitive_text(self._cached_system_prompt or ""),
                "tools": self.tools or [],
                "message_count": len(cleaned),
                "messages": cleaned,
            }

            atomic_json_write(
                log_file,
                entry,
                indent=2,
                default=str,
            )

        except Exception as e:
            if self.verbose_logging:
                logging.warning(f"Failed to save session log: {e}")
