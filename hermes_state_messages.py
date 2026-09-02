"""Transcript persistence for SessionDB.

Mixin split out of ``hermes_state.py``; bound onto ``SessionDB`` via the MRO
and built on its ``_read_ctx`` / ``_execute_write`` / ``_write_sql`` /
``_read_one`` / ``_read_all`` primitives. Covers message append / replace /
rewind, reactions, resume-conversation assembly and replayed-user-message
duplicate detection.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from agent.context_compressor import _DB_PERSISTED_MARKER as _DB_PERSISTED_MARKER_KEY
from agent.memory_manager import sanitize_context
from agent.message_sanitization import _sanitize_surrogates
from hermes_state_common import (
    _RESET_END_REASONS,
    _RESET_END_REASONS_SQL,
    _legacy_reset_child_sql,
)

# Log-record parity with the origin module (caplog tests pin "hermes_state").
logger = logging.getLogger("hermes_state")


class SessionMessagesMixin:
    """Message append/replace/rewind, reactions, resume conversations, replay dedupe."""

    def _bump_conversation_generation(self, conn, session_id: str, end_reason: str) -> None:
        """Advance this peer's conversation generation past a boundary.

        Called inside the transaction that writes the boundary, so the
        generation and the ``end_reason`` that caused it commit together.

        Only ``_RESET_END_REASONS`` count: ``compression`` continues one
        conversation, and an accidental close is not a replacement.  Rows with
        no ``session_key`` have no routing peer to advance.

        The counter deliberately does NOT read the session rows.  An aggregate
        over them (COUNT/MAX of boundaries) can return a pair it already
        emitted once ``delete_session()`` or bulk pruning removes an ended row,
        which would hand a new conversation a retired affinity identity.  This
        value only ever increments, so a generation is never reused for a peer
        even if every row behind it is gone.
        """
        if end_reason not in _RESET_END_REASONS:
            return
        row = conn.execute(
            "SELECT source, session_key FROM sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
        if row is None:
            return
        source = str(row["source"] or "").strip()
        session_key = str(row["session_key"] or "").strip()
        if not source or not session_key:
            return
        conn.execute(
            """
            INSERT INTO conversation_generations (source, session_key, generation)
            VALUES (?, ?, 1)
            ON CONFLICT(source, session_key) DO UPDATE
                SET generation = conversation_generations.generation + 1
            """,
            (source, session_key),
        )

    @classmethod
    def _encode_content(cls, content: Any) -> Any:
        """Serialize structured (list/dict) message content for sqlite.

        sqlite3 can only bind ``str``, ``bytes``, ``int``, ``float``, and ``None``
        to query parameters. Multimodal messages have ``content`` as a list of
        parts (``[{"type": "text", ...}, {"type": "image_url", ...}]``), which
        raises ``ProgrammingError: Error binding parameter N: type 'list' is
        not supported`` when bound directly.

        Returns the value unchanged when it's already a safe scalar, or a
        sentinel-prefixed JSON string for lists/dicts. Paired with
        :meth:`_decode_content` on read.
        """
        if isinstance(content, str):
            # Lone UTF-16 surrogates reach here inside tool results scraped
            # from the web/social platforms (the same input that crashed the
            # guardrail hasher). The proactive sanitizer upstream only cleans
            # the *api_messages* copy, and the recovery sanitizer only runs
            # after the API call itself raises — which it no longer does — so
            # the canonical history keeps them and this write is where they
            # land. Left raw, sqlite3 raises UnicodeEncodeError, the flush is
            # abandoned, and the session silently stops persisting for the
            # rest of its life. Scrub so persistence never fails.
            return _sanitize_surrogates(content)
        if content is None or isinstance(content, (bytes, int, float)):
            return content
        try:
            # json.dumps defaults to ensure_ascii=True, which escapes any
            # surrogate as \udXXX — already safe to bind.
            return cls._CONTENT_JSON_PREFIX + json.dumps(content)
        except (TypeError, ValueError):
            # Last-resort fallback: stringify so persistence never fails.
            return _sanitize_surrogates(str(content))

    @classmethod
    def _decode_content(cls, content: Any) -> Any:
        """Reverse :meth:`_encode_content`; returns scalars unchanged."""
        if isinstance(content, str) and content.startswith(cls._CONTENT_JSON_PREFIX):
            try:
                return json.loads(content[len(cls._CONTENT_JSON_PREFIX):])
            except (json.JSONDecodeError, TypeError):
                logger.warning(
                    "Failed to decode JSON-encoded message content; "
                    "returning raw string"
                )
                return content
        return content

    @staticmethod
    def _encode_display_metadata(display_metadata: Any) -> Optional[str]:
        """Serialize ``display_metadata`` for its TEXT column without double-encoding.

        Import/replace paths can hand us an already-serialized JSON string (the
        same hazard ``tool_calls`` guards against above). ``json.dumps`` on that
        string would store a quoted JSON string, and the single ``json.loads``
        on read then yields a ``str`` instead of a dict.
        """
        if not display_metadata:
            return None
        if isinstance(display_metadata, str):
            try:
                parsed = json.loads(display_metadata)
            except (json.JSONDecodeError, TypeError):
                logger.warning("Ignoring non-JSON display metadata on write")
                return None
            if not isinstance(parsed, dict):
                logger.warning("Ignoring non-object display metadata on write")
                return None
            return json.dumps(parsed)
        if isinstance(display_metadata, dict):
            return json.dumps(display_metadata)
        logger.warning(
            "Ignoring unexpected display metadata type on write: %s",
            type(display_metadata).__name__,
        )
        return None

    def _check_transcript_write_guards(
        self,
        conn,
        session_id: str,
        compression_lock_holder: Optional[str],
        turn_lease_holder: Optional[str] = None,
        turn_lease_ttl_seconds: float = 300.0,
        reject_active_turn_lease: bool = False,
        reject_active_compression_lock: bool = False,
        allow_closed_compression_parent: bool = False,
    ) -> None:
        """Transcript-write admission checks, run INSIDE the write txn.

        Shared by :meth:`append_message` and :meth:`append_messages_batch` so
        the two writers can never diverge on these correctness invariants
        (this guard has already needed targeted fixes — see the #74478 patience
        note below). User-initiated transcript mutations may opt in to rejecting
        an active unowned turn lease in that same transaction.
        """
        from hermes_state import CompressionSessionClosedError, SessionCompressionInProgressError, SessionTurnLeaseLostError, _compression_lock_holder_process_is_dead
        # NOTE (#75316 redesign): appends do NOT check compression_locks.
        # The lock's job is to stop two COMPRESSIONS colliding, not to fence
        # ordinary transcript writes. Concurrent appends during a compression
        # are safe by construction: archive_and_compact() commits against a
        # watermark captured at compression start and clones every row that
        # arrived after it back into the live transcript, in the same write
        # transaction. Blocking appends here was the root cause of a whole
        # symptom family — turns dying as session_persistence_failed while a
        # slow provider summary held the lease (#74568, #77386), including
        # stale locks from dead PIDs blocking writes for the full TTL.
        # Destructive user mutations are different: a compressor that already
        # captured its watermark can otherwise publish the pre-rewind snapshot
        # after the mutation and resurrect the removed turn. Keep that narrow
        # fence opt-in so ordinary appends retain the watermark behavior.
        if reject_active_compression_lock:
            active_lock = conn.execute(
                "SELECT holder, expires_at FROM compression_locks "
                "WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if active_lock is not None:
                current_holder = active_lock["holder"]
                if (
                    float(active_lock["expires_at"]) <= time.time()
                    or _compression_lock_holder_process_is_dead(current_holder)
                ):
                    conn.execute(
                        "DELETE FROM compression_locks "
                        "WHERE session_id = ? AND holder = ?",
                        (session_id, current_holder),
                    )
                elif current_holder != compression_lock_holder:
                    raise SessionCompressionInProgressError(
                        f"Session {session_id!r} is being compressed by another writer"
                    )
        if turn_lease_holder or reject_active_turn_lease:
            conversation_id = self._session_turn_lease_key_on_conn(conn, session_id)
            lease = conn.execute(
                "SELECT holder, expires_at FROM session_turn_leases "
                "WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
            now = time.time()
            if turn_lease_holder:
                if lease is None or lease["holder"] != turn_lease_holder:
                    raise SessionTurnLeaseLostError(
                        f"Session turn lease lost; refusing transcript write "
                        f"for {session_id!r}"
                    )
                if float(lease["expires_at"]) <= now:
                    # Expiry makes the row reclaimable; it does not prove that a
                    # takeover occurred. BEGIN IMMEDIATE serializes this renewal
                    # with acquisition, so a still-matching owner can recover from
                    # a starved refresher without weakening the foreign-holder fence.
                    conn.execute(
                        "UPDATE session_turn_leases SET expires_at = ? "
                        "WHERE conversation_id = ? AND holder = ?",
                        (
                            now + max(0.1, float(turn_lease_ttl_seconds)),
                            conversation_id,
                            turn_lease_holder,
                        ),
                    )
            elif lease is not None:
                current_holder = lease["holder"]
                if (
                    float(lease["expires_at"]) <= now
                    or _compression_lock_holder_process_is_dead(current_holder)
                ):
                    # Match acquisition semantics: an expired or provably dead
                    # owner is reclaimable. Deleting it inside this BEGIN IMMEDIATE
                    # transaction also fences a stale late flush after the mutation.
                    conn.execute(
                        "DELETE FROM session_turn_leases "
                        "WHERE conversation_id = ? AND holder = ?",
                        (conversation_id, current_holder),
                    )
                else:
                    raise SessionTurnLeaseLostError(
                        f"Session has an active turn lease; refusing transcript "
                        f"mutation for {session_id!r}"
                    )
        session = conn.execute(
            "SELECT ended_at, end_reason FROM sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
        if (
            session is not None
            and session["ended_at"] is not None
            and session["end_reason"] == "compression"
            and not allow_closed_compression_parent
        ):
            raise CompressionSessionClosedError(session_id)

    @staticmethod
    def _decode_display_metadata(raw: Any) -> Optional[Dict[str, Any]]:
        """Decode a ``display_metadata`` column into the dict every reader expects.

        Every message read path must go through this. Returning the raw TEXT
        instead reaches the desktop as a string, where ``'task_count' in meta``
        throws and fails the whole resume. Rows written before the encode guard
        landed are double-encoded, so unwrap a second layer when we find one.
        """
        if raw is None:
            return None
        try:
            meta = json.loads(raw) if isinstance(raw, str) else raw
            if isinstance(meta, str):
                meta = json.loads(meta)
        except (json.JSONDecodeError, TypeError):
            logger.warning("Ignoring invalid display metadata on message row")
            return None
        if not isinstance(meta, dict):
            logger.warning("Ignoring non-object display metadata on message row")
            return None
        return meta

    @staticmethod
    def _reasoning_json_text(value: Any) -> Optional[str]:
        """Serialize a structured reasoning field for its TEXT column.

        ``reasoning_details`` / ``codex_reasoning_items`` / ``codex_message_items``
        arrive as list/dict structures from the live runtime, but callers that
        round-trip stored rows — ``get_messages`` straight into
        ``replace_messages``, e.g. the POST /api/sessions/{id}/fork handler —
        hand back the raw TEXT these columns already hold, because
        ``get_messages`` only deserializes ``content`` and ``tool_calls``.
        Re-dumping that TEXT double-encodes it, and the forked session's next
        ``get_messages_as_conversation`` json.loads then yields the inner
        string instead of the original list, so every reasoning-replay consumer
        (all of which check ``isinstance(..., list)``) silently drops it.
        Strings are therefore stored as-is; structures are dumped.
        """
        if not value:
            return None
        if isinstance(value, str):
            return value
        return json.dumps(value)

    def append_message(
        self,
        session_id: str,
        role: str,
        content: str = None,
        tool_name: str = None,
        tool_calls: Any = None,
        tool_call_id: str = None,
        token_count: int = None,
        finish_reason: str = None,
        reasoning: str = None,
        reasoning_content: str = None,
        reasoning_details: Any = None,
        codex_reasoning_items: Any = None,
        codex_message_items: Any = None,
        platform_message_id: str = None,
        observed: bool = False,
        effect_disposition: Optional[str] = None,
        _compressed_summary: bool = False,
        timestamp: Any = None,
        api_content: Optional[str] = None,
        display_kind: Optional[str] = None,
        display_metadata: Optional[Dict[str, Any]] = None,
        compression_lock_holder: Optional[str] = None,
        turn_lease_holder: Optional[str] = None,
        turn_lease_ttl_seconds: float = 300.0,
    ) -> int:
        """
        Append a message to a session. Returns the message row ID.

        Also increments the session's message_count (and tool_call_count
        if role is 'tool' or tool_calls is present).

        ``platform_message_id`` is the external messaging platform's own
        message ID (e.g. Telegram update_id, Yuanbao msg_id).  It is
        independent of the SQLite autoincrement primary key and is used by
        platform-specific flows like yuanbao's recall guard to redact a
        message by its platform-side identifier.

        ``api_content`` is the exact content string sent to the API for this
        message when it differs from ``content`` (ephemeral memory/plugin
        injections, persist overrides).  It is a byte-fidelity sidecar for
        prompt-cache-stable replay — stored as sent, except lone surrogates
        (which sqlite3 cannot bind and which the conversation loop scrubs
        from every outgoing payload anyway, so the scrubbed form IS the
        wire bytes).
        """
        from hermes_state import _scrub_surrogates
        # Display metadata is presentation-only and never changes the model
        # context role/content replayed to providers.
        display_metadata_json = self._encode_display_metadata(display_metadata)
        # Serialize structured fields to JSON before entering the write txn
        reasoning_details_json = self._reasoning_json_text(reasoning_details)
        codex_items_json = self._reasoning_json_text(codex_reasoning_items)
        codex_message_items_json = self._reasoning_json_text(codex_message_items)
        # tool_calls may arrive as a Python list (from the live agent) or
        # as a JSON string (from import/export). Parse first to avoid
        # double-encoding.
        if isinstance(tool_calls, str):
            try:
                tool_calls = json.loads(tool_calls)
            except (json.JSONDecodeError, TypeError):
                tool_calls = []
        tool_calls_json = json.dumps(tool_calls) if tool_calls else None
        # Multimodal content (list of parts) must be JSON-encoded: sqlite3
        # cannot bind list/dict parameters directly.
        stored_content = self._encode_content(content)

        message_timestamp = time.time()
        if timestamp is not None:
            try:
                if hasattr(timestamp, "timestamp"):
                    message_timestamp = float(timestamp.timestamp())
                else:
                    message_timestamp = float(timestamp)
            except (TypeError, ValueError):
                logger.debug("Ignoring invalid explicit message timestamp: %r", timestamp)

        # Pre-compute tool call count
        num_tool_calls = 0
        if tool_calls is not None:
            num_tool_calls = len(tool_calls) if isinstance(tool_calls, list) else 1

        def _do(conn):
            self._check_transcript_write_guards(
                conn,
                session_id,
                compression_lock_holder,
                turn_lease_holder=turn_lease_holder,
                turn_lease_ttl_seconds=turn_lease_ttl_seconds,
            )
            cursor = conn.execute(
                """INSERT INTO messages (session_id, role, content, tool_call_id,
                   tool_calls, tool_name, effect_disposition, timestamp, token_count, finish_reason,
                   reasoning, reasoning_content, reasoning_details, codex_reasoning_items,
                   codex_message_items, platform_message_id, observed, _compressed_summary, active, api_content, display_kind, display_metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    session_id,
                    role,
                    stored_content,
                    tool_call_id,
                    tool_calls_json,
                    _scrub_surrogates(tool_name),
                    effect_disposition,
                    message_timestamp,
                    token_count,
                    finish_reason,
                    _scrub_surrogates(reasoning),
                    _scrub_surrogates(reasoning_content),
                    reasoning_details_json,
                    codex_items_json,
                    codex_message_items_json,
                    platform_message_id,
                    1 if observed else 0,
                    1 if _compressed_summary else 0,
                    1,
                    _scrub_surrogates(api_content) if isinstance(api_content, str) else None,
                    _scrub_surrogates(display_kind) if isinstance(display_kind, str) else None,
                    display_metadata_json,
                ),
            )
            msg_id = cursor.lastrowid

            # Update counters
            if num_tool_calls > 0:
                conn.execute(
                    """UPDATE sessions SET message_count = message_count + 1,
                       tool_call_count = tool_call_count + ? WHERE id = ?""",
                    (num_tool_calls, session_id),
                )
            else:
                conn.execute(
                    "UPDATE sessions SET message_count = message_count + 1 WHERE id = ?",
                    (session_id,),
                )
            return msg_id

        # Transcript append is THE critical write: its failure aborts the
        # user's turn (session_persistence_failed). Use the long patience so
        # a sibling process legitimately holding the write lock for seconds
        # (VACUUM, TRUNCATE checkpoint at close, an older pre-bounded-merge
        # process's FTS optimize) can't destroy a healthy turn (#74478).
        return self._execute_write(
            _do, patience_s=self._TRANSCRIPT_WRITE_PATIENCE_S
        )

    def append_messages_batch(
        self,
        session_id: str,
        messages: List[Dict[str, Any]],
        compression_lock_holder: Optional[str] = None,
        turn_lease_holder: Optional[str] = None,
        chunk_rows: Optional[int] = None,
        turn_lease_ttl_seconds: float = 300.0,
    ) -> int:
        """Append multiple messages atomically in ONE write transaction.

        ``messages`` is a list of dicts in the same shape
        :meth:`_insert_message_rows` already consumes for replace/compact/
        import (role, content, tool_name, tool_calls, tool_call_id,
        finish_reason, reasoning*, codex_*, timestamp, api_content,
        display_kind, display_metadata, ...). Reusing that helper keeps ONE
        row-serialization path for every multi-row writer.

        A turn-boundary flush writes the whole turn (user + assistant + tool
        rows, typically 3-8 messages) as one BEGIN IMMEDIATE / commit pair
        instead of one transaction (and, off WAL, one fsync) per row.

        Atomicity contract: all rows land or none do (the caller re-flushes
        unstamped messages on the next attempt). The same admission guards
        as :meth:`append_message` run once for the batch — same session,
        same instant.

        ``chunk_rows`` bounds the transaction size for LARGE copies (branch
        seeds can be thousands of rows; measured: 10k rows ≈ 2.4s inside one
        BEGIN IMMEDIATE because the FTS triggers run per row, which would
        monopolize the write lock and starve concurrent writers). When set,
        the batch commits in chunks of at most that many rows — same
        recovery semantics as the old per-row loops (a mid-copy failure
        leaves a partial seed), just with bounded lock holds. A turn flush
        never needs it. Returns the inserted row count.
        """
        if not messages:
            return 0

        if chunk_rows is not None and len(messages) > chunk_rows:
            inserted_total = 0
            for start in range(0, len(messages), chunk_rows):
                inserted_total += self.append_messages_batch(
                    session_id,
                    messages[start:start + chunk_rows],
                    compression_lock_holder=compression_lock_holder,
                    turn_lease_holder=turn_lease_holder,
                    turn_lease_ttl_seconds=turn_lease_ttl_seconds,
                )
            return inserted_total

        def _do(conn):
            self._check_transcript_write_guards(
                conn,
                session_id,
                compression_lock_holder,
                turn_lease_holder=turn_lease_holder,
                turn_lease_ttl_seconds=turn_lease_ttl_seconds,
            )
            from agent.transcript_repair import resolve_and_repair_transcript_batch

            inserted_rows = resolve_and_repair_transcript_batch(
                conn,
                session_id,
                messages,
                encode_content_fn=self._encode_content,
                decode_content_fn=self._decode_content,
            )
            inserted = 0
            tool_calls_total = 0
            if inserted_rows:
                inserted, tool_calls_total = self._insert_message_rows(
                    conn, session_id, inserted_rows
                )

            # One aggregated counter update for the newly inserted rows.
            if tool_calls_total > 0:
                conn.execute(
                    """UPDATE sessions SET message_count = message_count + ?,
                       tool_call_count = tool_call_count + ? WHERE id = ?""",
                    (inserted, tool_calls_total, session_id),
                )
            elif inserted > 0:
                conn.execute(
                    "UPDATE sessions SET message_count = message_count + ? WHERE id = ?",
                    (inserted, session_id),
                )
            return inserted

        # Same criticality as append_message: this IS the turn's transcript.
        return self._execute_write(
            _do, patience_s=self._TRANSCRIPT_WRITE_PATIENCE_S
        )

    def set_latest_matching_message_display_kind(
        self, session_id: str, *, role: str, content: str, display_kind: str,
        display_metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Stamp presentation metadata on this turn's freshly persisted row.

        The model still receives ``role`` and ``content`` unchanged. Gateway and
        CLI synthetic inputs call this immediately after their serial turn has
        flushed, preserving producer provenance without classifying by content
        during transcript rendering.
        """
        from hermes_state import _scrub_surrogates
        if not session_id or not content or not display_kind:
            return False

        def _do(conn):
            row = conn.execute(
                "SELECT id FROM messages WHERE session_id = ? AND role = ? "
                "AND content = ? AND active = 1 ORDER BY id DESC LIMIT 1",
                (session_id, role, self._encode_content(content)),
            ).fetchone()
            if row is None:
                return False
            conn.execute(
                "UPDATE messages SET display_kind = ?, display_metadata = ? WHERE id = ?",
                (
                    _scrub_surrogates(display_kind),
                    self._encode_display_metadata(display_metadata),
                    row[0],
                ),
            )
            return True

        return bool(self._execute_write(_do))

    def set_message_reaction(
        self,
        session_id: str,
        message_row_id: int,
        emoji: Optional[str],
        *,
        author: str = "user",
    ) -> Optional[List[Dict[str, Any]]]:
        """Set (or with ``emoji=None`` clear) *author*'s reaction on one message.

        iOS Tapback semantics: one reaction per author per message. Re-sending
        the same emoji clears it, a different emoji replaces it. Returns the
        message's full reaction list after the write, or ``None`` when the row
        doesn't exist or isn't part of *session_id*.
        """
        from hermes_state import _scrub_surrogates
        if not session_id or message_row_id is None:
            return None

        def _do(conn):
            row = conn.execute(
                "SELECT display_metadata FROM messages WHERE id = ? AND session_id = ?",
                (message_row_id, session_id),
            ).fetchone()
            if row is None:
                return None

            meta = self._decode_display_metadata(row[0]) or {}
            existing = meta.get(self.REACTIONS_METADATA_KEY)
            reactions = [
                r
                for r in (existing if isinstance(existing, list) else [])
                if isinstance(r, dict) and r.get("author") != author
            ]
            previous = next(
                (
                    r
                    for r in (existing if isinstance(existing, list) else [])
                    if isinstance(r, dict) and r.get("author") == author
                ),
                None,
            )
            # Tapping the live reaction again retracts it.
            toggling_off = (
                emoji is not None and previous is not None and previous.get("emoji") == emoji
            )
            if emoji and not toggling_off:
                reactions.append(
                    {"emoji": _scrub_surrogates(emoji), "author": author, "at": time.time()}
                )

            if reactions:
                meta[self.REACTIONS_METADATA_KEY] = reactions
            else:
                meta.pop(self.REACTIONS_METADATA_KEY, None)

            conn.execute(
                "UPDATE messages SET display_metadata = ? WHERE id = ?",
                (self._encode_display_metadata(meta) if meta else None, message_row_id),
            )
            return reactions

        return self._execute_write(_do)

    def get_message_reactions(
        self, session_id: str, message_row_id: int
    ) -> List[Dict[str, Any]]:
        """Return the reaction list persisted on one message row (never ``None``)."""
        if not session_id or message_row_id is None:
            return []

        row = self._read_one(
            "SELECT display_metadata FROM messages WHERE id = ? AND session_id = ?",
            (message_row_id, session_id),
        )

        if row is None:
            return []

        meta = self._decode_display_metadata(row[0]) or {}
        reactions = meta.get(self.REACTIONS_METADATA_KEY)

        return [r for r in reactions if isinstance(r, dict)] if isinstance(reactions, list) else []

    def take_unseen_reactions(
        self, session_id: str, *, author: str = "user"
    ) -> List[Dict[str, Any]]:
        """Return *author*'s not-yet-surfaced reactions and mark them seen.

        Powers the cache-safe model-context path: reactions are announced on the
        NEXT user turn (never by rewriting the message that was reacted to), and
        the ``seen`` stamp guarantees each one is announced exactly once.
        """
        if not session_id:
            return []

        def _do(conn):
            rows = conn.execute(
                "SELECT id, role, content, display_metadata FROM messages "
                "WHERE session_id = ? AND active = 1 AND display_metadata IS NOT NULL "
                "ORDER BY id",
                (session_id,),
            ).fetchall()

            pending = []
            for row in rows:
                meta = self._decode_display_metadata(row["display_metadata"])
                if not meta:
                    continue
                reactions = meta.get(self.REACTIONS_METADATA_KEY)
                if not isinstance(reactions, list):
                    continue

                changed = False
                for reaction in reactions:
                    if (
                        not isinstance(reaction, dict)
                        or reaction.get("author") != author
                        or reaction.get("seen")
                    ):
                        continue
                    reaction["seen"] = True
                    changed = True
                    content = self._decode_content(row["content"])
                    pending.append(
                        {
                            "row_id": row["id"],
                            "role": row["role"],
                            "emoji": reaction.get("emoji") or "",
                            "text": content if isinstance(content, str) else "",
                        }
                    )

                if changed:
                    conn.execute(
                        "UPDATE messages SET display_metadata = ? WHERE id = ?",
                        (self._encode_display_metadata(meta), row["id"]),
                    )

            return pending

        return self._execute_write(_do) or []

    def latest_message_row_id(
        self, session_id: str, *, role: str = "user", offset: int = 0, require_text: bool = True
    ) -> Optional[int]:
        """Row id of the most recent active message with *role*, or ``None``.

        Two callers, same need — "the message I mean, without an id": the agent
        defaulting to the turn that triggered it, and the desktop reacting to a
        live message that hasn't round-tripped through a resume yet.
        ``offset`` steps to earlier turns (1 = the one before the latest) so a
        reaction can land retroactively — "two messages ago" is how the caller
        thinks about it.

        ``require_text`` (default) skips rows with no plain-text content —
        tool-call-only assistant turns and attachment stubs don't render as
        bubbles, so "the latest message" as a HUMAN means it must never
        resolve to one (a reaction landing on an invisible row looks dropped,
        and its annotation quotes an empty string).
        """
        if not session_id or role not in {"user", "assistant"} or offset < 0:
            return None

        text_filter = (
            "AND content IS NOT NULL AND TRIM(content) != '' " if require_text else ""
        )

        row = self._read_one(
            "SELECT id FROM messages WHERE session_id = ? AND role = ? "
            f"AND active = 1 {text_filter}ORDER BY id DESC LIMIT 1 OFFSET ?",
            (session_id, role, int(offset)),
        )

        return row[0] if row else None

    def latest_user_message_row_id(self, session_id: str) -> Optional[int]:
        """Row id of the most recent active user message, or ``None``.

        The agent's default reaction target: "the message that triggered me",
        so the model never has to thread row ids through a tool call (mirrors
        the photon adapter's ``_record_last_inbound``).
        """
        return self.latest_message_row_id(session_id, role="user")

    def get_message_role(self, session_id: str, row_id: int) -> Optional[str]:
        """Role of the active message at *row_id* in *session_id*, or ``None``.

        Lets a reaction event carry the target's role so a renderer can match
        a live message that doesn't know its durable row id yet.
        """
        if not session_id:
            return None

        row = self._read_one(
            "SELECT role FROM messages WHERE id = ? AND session_id = ? AND active = 1",
            (int(row_id), session_id),
        )

        return row[0] if row else None

    def _insert_message_rows(self, conn, session_id: str, messages: List[Dict[str, Any]]) -> tuple[int, int]:
        """Insert *messages* as fresh active rows for *session_id*.

        Shared by :meth:`replace_messages` (delete-then-insert) and
        :meth:`archive_and_compact` (soft-archive-then-insert). Runs inside the
        caller's write transaction (takes the live ``conn``). Returns
        ``(inserted_count, tool_call_count)``. Does NOT touch sessions.* counters
        — the caller owns that, since the two flows reconcile counts differently.
        """
        from hermes_state import _scrub_surrogates
        now_ts = time.time()
        inserted = 0
        tool_calls_total = 0
        for msg in messages:
            role = msg.get("role", "unknown")
            tool_calls = msg.get("tool_calls")
            message_timestamp = now_ts
            if msg.get("timestamp") is not None:
                try:
                    ts_value = msg.get("timestamp")
                    if hasattr(ts_value, "timestamp"):
                        message_timestamp = float(ts_value.timestamp())
                    else:
                        message_timestamp = float(ts_value)
                except (TypeError, ValueError):
                    logger.debug("Ignoring invalid explicit message timestamp: %r", msg.get("timestamp"))
            reasoning_details = msg.get("reasoning_details") if role == "assistant" else None
            codex_reasoning_items = (
                msg.get("codex_reasoning_items") if role == "assistant" else None
            )
            codex_message_items = (
                msg.get("codex_message_items") if role == "assistant" else None
            )
            reasoning_details_json = self._reasoning_json_text(reasoning_details)
            codex_items_json = self._reasoning_json_text(codex_reasoning_items)
            codex_message_items_json = self._reasoning_json_text(codex_message_items)
            # tool_calls may arrive as a Python list (from the live agent)
            # or as a JSON string (from import_sessions / export_session,
            # which store it as TEXT). json.dumps on an already-serialized
            # string double-encodes it, so parse first.
            if isinstance(tool_calls, str):
                try:
                    tool_calls = json.loads(tool_calls)
                except (json.JSONDecodeError, TypeError):
                    tool_calls = []
            tool_calls_json = json.dumps(tool_calls) if tool_calls else None
            # Accept either `platform_message_id` (new explicit name) or
            # `message_id` (yuanbao's existing convention on message dicts).
            platform_msg_id = (
                msg.get("platform_message_id") or msg.get("message_id")
            )

            api_content = msg.get("api_content")

            cur = conn.execute(
                """INSERT INTO messages (session_id, role, content, tool_call_id,
                   tool_calls, tool_name, effect_disposition, timestamp, token_count, finish_reason,
                   reasoning, reasoning_content, reasoning_details, codex_reasoning_items,
                   codex_message_items, platform_message_id, observed, _compressed_summary, active, api_content, display_kind, display_metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    session_id,
                    role,
                    self._encode_content(msg.get("content")),
                    msg.get("tool_call_id"),
                    tool_calls_json,
                    _scrub_surrogates(msg.get("tool_name")),
                    msg.get("effect_disposition"),
                    message_timestamp,
                    msg.get("token_count"),
                    msg.get("finish_reason"),
                    _scrub_surrogates(msg.get("reasoning")) if role == "assistant" else None,
                    _scrub_surrogates(msg.get("reasoning_content")) if role == "assistant" else None,
                    reasoning_details_json,
                    codex_items_json,
                    codex_message_items_json,
                    platform_msg_id,
                    1 if msg.get("observed") else 0,
                    1 if msg.get("_compressed_summary") else 0,
                    1,
                    _scrub_surrogates(api_content) if isinstance(api_content, str) else None,
                    _scrub_surrogates(msg.get("display_kind")) if isinstance(msg.get("display_kind"), str) else None,
                    self._encode_display_metadata(msg.get("display_metadata")),
                ),
            )
            if isinstance(msg, dict) and cur.lastrowid is not None:
                msg["_row_id"] = cur.lastrowid
            inserted += 1
            if tool_calls is not None:
                tool_calls_total += (
                    len(tool_calls) if isinstance(tool_calls, list) else 1
                )
            now_ts = max(now_ts + 1e-6, message_timestamp + 1e-6)
        return inserted, tool_calls_total

    def replace_messages(
        self,
        session_id: str,
        messages: List[Dict[str, Any]],
        active_only: bool = False,
        archive_dropped: bool = False,
        reject_active_turn_lease: bool = False,
    ) -> None:
        """Atomically replace the stored messages for a session.

        Used by transcript-rewrite flows such as /retry, /undo, and /compress.
        The delete + reinsert sequence must commit as one transaction so a
        mid-rewrite failure does not leave SQLite with a partial transcript.

        DESTRUCTIVE by default: every row for the session is DELETEd (and drops
        out of the FTS index). For compaction that must preserve the
        pre-compaction transcript under the same id, use
        :meth:`archive_and_compact` instead.

        Pass ``active_only=True`` to replace ONLY the live (``active = 1``) rows,
        leaving soft-archived rows (``active = 0`` — e.g. the ``compacted = 1``
        turns that :meth:`archive_and_compact` keeps on disk for #38763
        durability, or rewind/undo rows) untouched. Callers that share a session
        id with an agent already running in-place compaction must use this so a
        full-history rewrite doesn't wipe the rows the agent deliberately
        archived. ``message_count``/``tool_call_count`` then track the live set,
        matching :meth:`archive_and_compact`.

        Pass ``archive_dropped=True`` to SOFT-archive the live rows instead of
        DELETEing them: the replaced turns stay on disk with ``active = 0``,
        ``compacted = 0`` — the same "the user took it back" marking
        :meth:`rewind_to_message` applies — and stay readable via
        :meth:`get_messages` with ``include_inactive=True``. This is the mode a
        rewind/edit/regenerate must use: those flows overwrite a transcript the
        user may not have meant to drop, and a plain DELETE also evicts the rows
        from the FTS index, leaving nothing to recover from (#82756). It implies
        active-only handling — already-archived rows are never touched — so
        ``active_only`` is redundant with it. The rewritten set is inserted as
        fresh active rows exactly as in the destructive path, so the live view
        is identical either way; only the durability of the dropped turns
        differs.

        Pass ``reject_active_turn_lease=True`` for user-initiated rewrites that
        do not already own the cross-process turn lease. The lease check and
        transcript mutation then share one write transaction, so a second
        process cannot archive or replace a turn that is still being produced.
        """
        from hermes_state import CompressionSessionClosedError

        active_clause = " AND active = 1" if active_only else ""

        def _do(conn):
            if reject_active_turn_lease:
                self._check_transcript_write_guards(
                    conn,
                    session_id,
                    None,
                    reject_active_turn_lease=True,
                    reject_active_compression_lock=True,
                )
            else:
                session = conn.execute(
                    "SELECT ended_at, end_reason FROM sessions WHERE id = ?",
                    (session_id,),
                ).fetchone()
                if (
                    session is not None
                    and session["ended_at"] is not None
                    and session["end_reason"] == "compression"
                ):
                    raise CompressionSessionClosedError(session_id)
            if archive_dropped:
                # Content-preserving UPDATE: the rows keep their FTS entries
                # (the messages_fts triggers fire on INSERT / DELETE / UPDATE
                # of content columns, not on `active`), so the replaced turns
                # stay readable via get_messages(include_inactive=True) and
                # searchable with include_inactive=True after the rewrite.
                conn.execute(
                    "UPDATE messages SET active = 0 "
                    "WHERE session_id = ? AND active = 1",
                    (session_id,),
                )
            else:
                conn.execute(
                    f"DELETE FROM messages WHERE session_id = ?{active_clause}",
                    (session_id,),
                )
            conn.execute(
                "UPDATE sessions SET message_count = 0, tool_call_count = 0 WHERE id = ?",
                (session_id,),
            )
            total_messages, total_tool_calls = self._insert_message_rows(
                conn, session_id, messages
            )
            conn.execute(
                "UPDATE sessions SET message_count = ?, tool_call_count = ? WHERE id = ?",
                (total_messages, total_tool_calls, session_id),
            )

        self._execute_write(_do)

    def has_archived_messages(self, session_id: str) -> bool:
        """Return True if the session has any soft-archived (``active = 0``) rows.

        Cheap existence probe — does not load rows. NOTE: production rewrite
        paths no longer branch on this (they pass ``active_only=True``
        unconditionally — a probe can fail open or race a concurrent
        ``archive_and_compact``, #80216); kept for tests and diagnostics.
        """
        return self._read_one(
            "SELECT 1 FROM messages WHERE session_id = ? AND active = 0 LIMIT 1",
            (session_id,),
        ) is not None

    def get_active_message_watermark(self, session_id: str) -> int:
        """MAX(id) of the session's active rows — the compression watermark.

        Captured at compression START (before the slow provider summary call).
        Every active row with ``id > watermark`` at commit time arrived
        concurrently and must survive the compaction verbatim. Returns 0 for
        an empty/unknown session.
        """
        if not session_id:
            return 0
        row = self._read_one(
            "SELECT COALESCE(MAX(id), 0) FROM messages "
            "WHERE session_id = ? AND active = 1",
            (session_id,),
        )
        return int(row[0]) if row else 0

    def archive_and_compact(
        self,
        session_id: str,
        compacted_messages: List[Dict[str, Any]],
        model_config_patch: Optional[Dict[str, Any]] = None,
        watermark: Optional[int] = None,
        lock_holder: Optional[str] = None,
        tail_count: int = 0,
    ) -> int:
        """Non-destructive in-place compaction for a single durable session id.

        Soft-archives the active messages (``active = 0``) and inserts
        *compacted_messages* as fresh active rows — atomically, in one write
        transaction. The conversation keeps ONE session id for life (#38763)
        WITHOUT destroying history:

        - The live-context load (:meth:`get_messages_as_conversation`,
          :meth:`get_messages`) filters ``active = 1`` by default, so the model
          reloads ONLY the compacted set.
        - The archived pre-compaction turns stay on disk (active=0) and stay
          DISCOVERABLE: they are marked compacted=1, and search_messages()
          includes compacted=1 rows by default — so session_search still finds
          them, unlike rewind/undo rows (active=0, compacted=0) which stay
          hidden. They remain in the FTS index (the messages_fts* triggers
          index on INSERT / drop on DELETE and don't key on active/compacted;
          flipping to active=0 is a content-preserving UPDATE) and are
          recoverable via get_messages(..., include_inactive=True).

        Concurrent-append safety (#75316): when *watermark* is provided (the
        value of :meth:`get_active_message_watermark` captured at compression
        START), rows that arrived during the slow provider summary call
        (``id > watermark``) are NOT summarized away. They are re-sequenced
        after the compacted set by a pure-SQL column clone (every column
        except ``id`` — content, api_content, platform_message_id, token
        counts, reasoning sidecars all survive byte-exact, and the FTS
        triggers index the clones naturally), and the originals are archived.
        NOTE: re-sequencing assigns the tail rows fresh ids; consumers that
        reference durable row ids re-resolve by content (see 3e8ab0610).
        ``watermark=None`` preserves the historical archive-everything
        behavior.

        Commit-fence safety: when *lock_holder* is provided, the commit
        verifies INSIDE the transaction that the compression lock is still
        held by that holder and unexpired — a compression whose lease was
        reclaimed (crash cleanup, TTL expiry, competing writer) fails the
        commit instead of clobbering the winner's transcript.

        *tail_count* (default 0) names how many of the LAST rows of
        *compacted_messages* are the verbatim carried-forward tail the
        compressor protected rather than summarized (#86366). Those rows'
        ORIGINALS — which this call archives as a side effect of the blanket
        soft-archive — are superseded byte-identical duplicates, not
        "summarized away" content, so they are stamped rewind-style
        (``active=0, compacted=0``, hidden from search_messages) instead of
        ``compacted=1``. Without this the tail originals satisfy the recall
        filter alongside their live clones and session_search returns every
        carried-forward message once per compaction. Callers that cannot know
        their tail shape keep the historical archive-everything behavior.

        ``message_count`` is set to the ACTIVE count after commit, matching
        what the live load returns. ``model_config_patch`` is merged into the
        session's JSON config in the same transaction; a ``None`` value
        removes that key. Returns the new active count.
        """
        from hermes_state import SessionCompressionInProgressError

        def _do(conn):
            if lock_holder is not None:
                lock_row = conn.execute(
                    "SELECT holder, expires_at FROM compression_locks "
                    "WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
                if (
                    lock_row is None
                    or lock_row["holder"] != lock_holder
                    or float(lock_row["expires_at"]) <= time.time()
                ):
                    raise SessionCompressionInProgressError(
                        f"Compression lease for {session_id!r} lost before "
                        "commit; refusing to publish a stale compaction"
                    )

            patched_model_config = None
            if model_config_patch is not None:
                # on_missing="raise": a prune/compaction must not commit
                # against a vanished session row (the compressor's caller
                # converts the raised error into a safe keep-the-original
                # no-op), unlike the flag setters which tolerate missing rows.
                patched_model_config = self._merge_model_config_json(
                    conn, session_id, model_config_patch, on_missing="raise"
                )

            # Concurrent tail: active rows that arrived after the watermark.
            # Snapshot their ids and tool_calls now — the clone below needs a
            # stable id list, and the tool-call count keeps sessions.* honest.
            tail_ids: list[int] = []
            tail_tool_calls = 0
            if watermark is not None:
                for row in conn.execute(
                    "SELECT id, tool_calls FROM messages "
                    "WHERE session_id = ? AND active = 1 AND id > ? "
                    "ORDER BY id",
                    (session_id, int(watermark)),
                ).fetchall():
                    tail_ids.append(int(row["id"]))
                    raw = row["tool_calls"]
                    if raw:
                        try:
                            parsed = json.loads(raw) if isinstance(raw, str) else raw
                            tail_tool_calls += len(parsed) if isinstance(parsed, list) else 0
                        except (TypeError, ValueError):
                            pass

            # Soft-archive the live turns: active=0 hides them from the live
            # context load, compacted=1 marks them as "summarized away" (vs
            # rewind/undo's active=0+compacted=0, which means "user took it
            # back"). search_messages includes compacted=1 rows by default so
            # the pre-compaction transcript stays discoverable; live-context
            # loads (active=1 only) still exclude them. Tail originals whose
            # verbatim clones ride inside *compacted_messages* (tail_count)
            # are superseded duplicates instead (#86366): they get the
            # rewind-style flags so they stop matching the recall filter.
            # Rewind-target ids: the originals of the carried-forward tail
            # rows (tail_count), captured BEFORE any flag flips. Named apart
            # from the watermark `tail_ids` below on purpose — the two are
            # different sets (#86366): rewind targets sit AT/BELOW the
            # watermark (the compressor only saw rows up to it), while
            # `tail_ids` are concurrent appends ABOVE it. Without the bound,
            # a concurrent append would steal a LIMIT slot and leave a real
            # carried-forward original stamped compacted=1.
            rewind_tail_ids: Optional[list[int]] = None
            if tail_count > 0:
                if watermark is not None:
                    tail_rows = conn.execute(
                        "SELECT id FROM messages "
                        "WHERE session_id = ? AND active = 1 AND id <= ? "
                        "ORDER BY id DESC LIMIT ?",
                        (session_id, int(watermark), int(tail_count)),
                    ).fetchall()
                else:
                    tail_rows = conn.execute(
                        "SELECT id FROM messages "
                        "WHERE session_id = ? AND active = 1 ORDER BY id DESC LIMIT ?",
                        (session_id, int(tail_count)),
                    ).fetchall()
                rewind_tail_ids = [int(row["id"]) for row in tail_rows]

            # The watermark clone below re-inserts `tail_ids` rows byte-exact
            # as live rows — their originals are the SAME superseded-duplicate
            # class as the carried-forward tail (#86366), so they take the
            # rewind flags too instead of double-matching the recall filter.
            rewind_ids = [*(rewind_tail_ids or []), *tail_ids]

            if rewind_ids:
                placeholders = ",".join("?" for _ in rewind_ids)
                conn.execute(
                    "UPDATE messages SET active = 0, compacted = 0 "
                    f"WHERE session_id = ? AND id IN ({placeholders})",
                    [session_id, *rewind_ids],
                )
                conn.execute(
                    "UPDATE messages SET active = 0, compacted = 1 "
                    "WHERE session_id = ? AND active = 1 "
                    f"AND id NOT IN ({placeholders})",
                    [session_id, *rewind_ids],
                )
            else:
                conn.execute(
                    "UPDATE messages SET active = 0, compacted = 1 "
                    "WHERE session_id = ? AND active = 1",
                    (session_id,),
                )
            inserted, tool_calls_total = self._insert_message_rows(
                conn, session_id, compacted_messages
            )

            if tail_ids:
                # Re-sequence the concurrent tail after the compacted set via
                # a pure-SQL column clone: no decode/re-encode round trip, no
                # field drift — new id, active=1, compacted=0, all else exact.
                placeholders = ",".join("?" for _ in tail_ids)
                clone_cols = [
                    c for c in self._message_column_names(conn)
                    if c not in ("id", "active", "compacted")
                ]
                col_list = ", ".join(clone_cols)
                conn.execute(
                    f"INSERT INTO messages ({col_list}, active, compacted) "
                    f"SELECT {col_list}, 1, 0 FROM messages "
                    f"WHERE id IN ({placeholders}) ORDER BY id",
                    tail_ids,
                )
                inserted += len(tail_ids)
                tool_calls_total += tail_tool_calls

            # message_count / tool_call_count reflect the LIVE (active) set —
            # the archived rows are still on disk but not part of the live count.
            if model_config_patch is None:
                conn.execute(
                    "UPDATE sessions SET message_count = ?, tool_call_count = ? WHERE id = ?",
                    (inserted, tool_calls_total, session_id),
                )
            else:
                conn.execute(
                    "UPDATE sessions SET message_count = ?, tool_call_count = ?, "
                    "model_config = ? WHERE id = ?",
                    (inserted, tool_calls_total, patched_model_config, session_id),
                )
            return inserted

        return self._execute_write(_do)

    def _message_column_names(self, conn) -> List[str]:
        """Column names of the messages table, cached per-connection era."""
        cached = getattr(self, "_message_columns_cache", None)
        if cached:
            return cached
        cols = [r[1] for r in conn.execute("PRAGMA table_info(messages)").fetchall()]
        self._message_columns_cache = cols
        return cols

    def set_latest_user_api_content(
        self, session_id: str, content: Any, api_content: str
    ) -> int:
        """Backfill the ``api_content`` sidecar onto the newest ACTIVE user row.

        In-place preflight compaction (:meth:`archive_and_compact`) inserts the
        current turn's user row BEFORE the turn prologue composes the
        prefetch/plugin sidecar, and the subsequent crash persist identity-skips
        every compacted dict — without this backfill the stamped sidecar would
        never land in the DB and any reload would replay clean content,
        re-introducing the prompt-cache divergence the sidecar exists to close.

        The ``content`` match is a defensive guard: if the newest active user
        row is not the message the caller stamped (racing rewrite, unexpected
        tail shape), nothing is written. Returns the number of rows updated
        (0 or 1).
        """
        from hermes_state import _scrub_surrogates
        encoded = self._encode_content(content)

        return self._write_rowcount(
            "UPDATE messages SET api_content = ? WHERE id = ("
            "SELECT id FROM messages "
            "WHERE session_id = ? AND role = 'user' AND active = 1 "
            "ORDER BY id DESC LIMIT 1"
            ") AND content IS ?",
            (_scrub_surrogates(api_content), session_id, encoded),
        )

    def _dedupe_display_generations(self, rows):
        """Collapse compaction generations so each message appears once.

        Compaction epochs copy the protected tail into each new generation, so
        one logical message can exist as several rows (identical
        role/content/timestamp) with different ``active`` flags and ids. A
        display read must surface each exactly once: prefer the live row, then
        the newest generation.

        This is the ONE definition shared by every display projection —
        :meth:`get_messages` (REST), :meth:`get_resume_conversations` and
        :meth:`get_ancestor_display_prefix` (gateway resume), and
        :meth:`get_messages_as_conversation` (warm-session payload) — so the
        surfaces cannot disagree about the same transcript. *rows* must already
        be ordered by ``id``; the returned list keeps that order.
        """
        seen: Dict[Tuple[Any, ...], Any] = {}
        for row in rows:
            dedupe_content = row["content"]
            if row["role"] == "user":
                from agent.context_compressor import split_user_originated_turn

                candidate = {
                    "role": "user",
                    "content": self._decode_content(row["content"]),
                    "display_kind": row["display_kind"],
                    "display_metadata": self._decode_display_metadata(
                        row["display_metadata"]
                    ),
                }
                handoff, live_view = split_user_originated_turn(candidate)
                if handoff is not None and live_view is not None:
                    dedupe_content = self._encode_content(live_view.get("content"))
            # Tool fields participate in the dedupe key: compaction copies them
            # verbatim, so identical tool messages across generations still
            # collapse, while distinct tool calls that happen to share
            # role/content/timestamp are never merged.
            key = (
                row["role"],
                dedupe_content,
                row["timestamp"],
                row["tool_call_id"],
                row["tool_calls"],
                row["tool_name"],
            )
            cur = seen.get(key)
            if cur is None or (row["active"], row["id"]) > (cur["active"], cur["id"]):
                seen[key] = row
        return sorted(seen.values(), key=lambda r: r["id"])

    def get_messages(
        self,
        session_id: str,
        include_inactive: bool = False,
        include_compacted: bool = False,
        limit: Optional[int] = None,
        offset: int = 0,
        latest: bool = False,
        after_id: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Load messages for a session in insertion order.

        By default only active messages are returned. Pass
        ``include_inactive=True`` to load soft-deleted rows (e.g. for
        audit / debug views of rewound history). See
        :meth:`rewind_to_message` for the soft-delete mechanic.

        Pass ``include_compacted=True`` to additionally load rows preserved
        by in-place context compaction (``active=0, compacted=1``). Those are
        durable display history, not soft-deleted rows — a user-visible
        transcript read must not drop them, or earlier turns silently become
        unreachable once the UI exhausts its active-only window. Soft-deleted
        Undo/Rewind rows (``active=0, compacted=0``) stay excluded; use
        ``include_inactive`` for those.

        Ordered by AUTOINCREMENT id (true insertion order) rather than
        timestamp — see c03acca50 for the WSL2 clock-regression rationale.

        When ``limit`` is provided, returns at most ``limit`` messages
        starting from ``offset`` (0-based, in insertion order). Enables
        pagination for the API endpoint to avoid loading entire transcripts.
        With ``latest=True``, the offset is measured back from the newest
        message and the selected page is still returned in chronological
        order. ``offset`` alone (without ``limit``) also pages — SQLite
        requires a LIMIT clause for OFFSET, so it's emitted as ``LIMIT -1``
        (unbounded).

        ``after_id`` enables keyset pagination (``id > after_id``): O(1)
        page seeks on huge transcripts where OFFSET degrades to O(n) per
        page. Ascending order only (incompatible with ``latest``/``offset``).
        """
        if after_id is not None and (latest or offset):
            raise ValueError("after_id is incompatible with latest/offset paging")
        if after_id is not None and include_compacted:
            raise ValueError("after_id is incompatible with include_compacted (deduped display reads use offset paging)")
        if include_inactive:
            # Audit / debug reads: every row, including soft-deleted.
            active_clause = ""
        elif include_compacted:
            # Display history: active rows plus rows preserved by in-place
            # compaction (active=0, compacted=1), but never soft-deleted
            # Undo/Rewind rows (active=0, compacted=0).
            active_clause = " AND (active = 1 OR compacted = 1)"
        else:
            active_clause = " AND active = 1"
        keyset_clause = " AND id > ?" if after_id is not None else ""
        sql = (
            "SELECT * FROM messages WHERE session_id = ?"
            f"{active_clause}{keyset_clause} ORDER BY id {'DESC' if latest else 'ASC'}"
        )
        params: list = [session_id]
        if after_id is not None:
            params.append(after_id)
        if include_compacted:
            # Read the full display set (a session's rows are bounded; the
            # UI-level 500-row cap lives in the endpoint, not here), dedupe
            # generations, then apply paging.
            all_rows = self._read_all(
                "SELECT * FROM messages WHERE session_id = ?" + active_clause
                + " ORDER BY id ASC",
                [session_id],
            )
            rows = self._dedupe_display_generations(all_rows)
            if latest:
                rows = rows[::-1]
            rows = rows[offset:]
            if limit is not None:
                rows = rows[:limit]
            if latest:
                rows = rows[::-1]
        else:
            if limit is not None or offset:
                # SQLite's OFFSET requires LIMIT; -1 means "no limit".
                sql += " LIMIT ? OFFSET ?"
                params.extend([-1 if limit is None else limit, offset])
            rows = self._read_all(sql, params)
            if latest:
                rows.reverse()
        result = []
        for row in rows:
            msg = dict(row)
            if msg.pop("_compressed_summary", 0):
                msg["_compressed_summary"] = True
            if "content" in msg:
                msg["content"] = self._decode_content(msg["content"])
            if msg.get("tool_calls"):
                try:
                    msg["tool_calls"] = json.loads(msg["tool_calls"])
                except (json.JSONDecodeError, TypeError):
                    logger.warning("Failed to deserialize tool_calls in get_messages, falling back to []")
                    msg["tool_calls"] = []
            if msg.get("display_metadata") is not None:
                msg["display_metadata"] = self._decode_display_metadata(msg["display_metadata"])
            result.append(msg)
        return result

    def find_pr_url_messages(self, session_ids: List[str]) -> List[Dict[str, Any]]:
        """Tool results in these sessions that mention a GitHub PR url.

        A candidate scan, deliberately loose: it hands back every tool result
        containing ``/pull/`` and leaves the caller to decide which ones make a
        claim (see the desktop's PR recovery, which only accepts an output that
        is a bare PR url — the signature of ``gh pr create``). Ordered
        oldest-first per session so the caller can take the last match.
        """
        found: List[Dict[str, Any]] = []
        ids = [s for s in session_ids if s]
        for start in range(0, len(ids), 900):  # SQLite's bound-variable ceiling.
            chunk = ids[start : start + 900]
            placeholders = ",".join("?" * len(chunk))
            rows = self._read_all(
                f"""SELECT session_id, content FROM messages
                    WHERE session_id IN ({placeholders})
                      AND role = 'tool' AND content LIKE '%/pull/%'
                    ORDER BY id ASC""",
                chunk,
            )
            found.extend({"session_id": row[0], "content": row[1]} for row in rows)
        return found

    def get_messages_around(
        self,
        session_id: str,
        around_message_id: int,
        window: int = 5,
    ) -> Dict[str, Any]:
        """Load a window of messages anchored on a specific message id.

        Returns a dict with:
          - ``window``: up to ``window`` messages before the anchor, the anchor
            itself, and up to ``window`` messages after, ordered by id ascending.
          - ``messages_before``: count of messages strictly before the anchor
            still in the session (== window unless we hit the start).
          - ``messages_after``: count of messages strictly after the anchor
            still in the session (== window unless we hit the end).

        Used by ``session_search`` for both the discovery shape (anchored on the
        FTS5 match) and the scroll shape (anchored on any message id). The
        ``messages_before`` / ``messages_after`` counts let the caller detect
        session boundaries: when either is less than ``window``, the agent has
        reached one end of the session.

        Returns an empty window when ``around_message_id`` is not a real id in
        ``session_id`` — callers decide how to surface that.
        """
        if window < 0:
            window = 0
        with self._read_ctx() as conn:
            # Confirm the anchor exists in this session.
            anchor_exists = conn.execute(
                "SELECT 1 FROM messages WHERE id = ? AND session_id = ? LIMIT 1",
                (around_message_id, session_id),
            ).fetchone()
            if not anchor_exists:
                return {"window": [], "messages_before": 0, "messages_after": 0}

            # Two queries: anchor + before (DESC, take window+1), and after
            # (ASC, take window). Final order is id ASC.
            before_rows = conn.execute(
                "SELECT * FROM messages "
                "WHERE session_id = ? AND id <= ? "
                "ORDER BY id DESC LIMIT ?",
                (session_id, around_message_id, window + 1),
            ).fetchall()
            after_rows = conn.execute(
                "SELECT * FROM messages "
                "WHERE session_id = ? AND id > ? "
                "ORDER BY id ASC LIMIT ?",
                (session_id, around_message_id, window),
            ).fetchall()

        # before_rows is DESC; reverse so it's ASC, then concatenate after_rows.
        rows = list(reversed(before_rows)) + list(after_rows)
        result = []
        for row in rows:
            msg = dict(row)
            if "content" in msg:
                msg["content"] = self._decode_content(msg["content"])
            if msg.get("tool_calls"):
                try:
                    msg["tool_calls"] = json.loads(msg["tool_calls"])
                except (json.JSONDecodeError, TypeError):
                    logger.warning(
                        "Failed to deserialize tool_calls in get_messages_around, falling back to []"
                    )
                    msg["tool_calls"] = []
            if msg.get("display_metadata") is not None:
                msg["display_metadata"] = self._decode_display_metadata(msg["display_metadata"])
            result.append(msg)

        # before_rows includes the anchor itself; subtract 1 for the count of
        # messages strictly before the anchor in the returned slice.
        messages_before = max(0, len(before_rows) - 1)
        messages_after = len(after_rows)
        return {
            "window": result,
            "messages_before": messages_before,
            "messages_after": messages_after,
        }

    def resolve_resume_session_id(self, session_id: str) -> str:
        """Redirect a resume target to the descendant session that holds the messages.

        Context compression ends the current session and forks a new child session
        (linked via ``parent_session_id``). The flush cursor is reset, so the
        child is where new messages actually land — the parent ends up with
        ``message_count = 0`` rows unless messages had already been flushed to
        it before compression. See #15000.

        This helper walks ``parent_session_id`` forward from ``session_id`` and
        returns the descendant in the chain that has the **most recent** messages.
        Unlike the original logic, it does NOT short-circuit when the starting
        session already has messages — a descendant that was created by
        compression may hold the continuation content and should be preferred
        by the WebUI and gateway for ``--resume`` and session loading.

        If no descendant (including the starting session) has any messages,
        the original ``session_id`` is returned unchanged.

        The chain is always walked via the child whose ``started_at`` is
        latest; that matches the single-chain shape that compression creates.
        A depth cap (32) guards against accidental loops in malformed data.
        """
        if not session_id:
            return session_id

        # Follow the compression-continuation chain forward to the live tip
        # FIRST. Auto-compression ends the current session and forks a
        # continuation child, but a long-lived parent keeps its own flushed
        # message rows — so the empty-head walk below never redirects it, and
        # resuming the parent id reloads the pre-compression transcript while
        # the turns generated *after* compression (and their responses) sit in
        # the continuation. ``get_compression_tip`` is lineage-aware: it only
        # follows children whose parent ended with ``end_reason='compression'``
        # (created after the parent was ended), so delegation / branch children
        # never hijack the resume. This is the fix for the desktop "I came back
        # and the reply isn't there" report on large sessions.
        try:
            tip = self.get_compression_tip(session_id)
        except Exception:
            tip = session_id
        if tip and tip != session_id:
            session_id = tip

        with self._read_ctx() as conn:
            current = session_id
            seen = {current}
            best = None  # tracks the last (deepest) node with messages

            for _ in range(32):
                # Check if the current node has messages.
                try:
                    row = conn.execute(
                        "SELECT 1 FROM messages WHERE session_id = ? LIMIT 1",
                        (current,),
                    ).fetchone()
                except Exception:
                    return session_id
                if row is not None:
                    best = current

                # Walk to the most-recently-started child — but skip explicit
                # branch (`_branched_from`), delegate/subagent (`_delegate_from`),
                # reset-continuation (`_reset_from` or the legacy same-key
                # heuristic — a post-reset conversation must never be reached
                # by resuming the parent the user reset away), and tool
                # children. They also carry a ``parent_session_id`` yet
                # are NOT compression continuations; following them would hijack
                # the resume target to an unrelated session (e.g. a subagent
                # run). This mirrors the child-exclusion in ``get_compression_tip``.
                try:
                    child_row = conn.execute(
                        "SELECT id FROM sessions AS child "
                        "WHERE child.parent_session_id = ? "
                        "  AND json_extract(COALESCE(child.model_config, '{}'), '$._branched_from') IS NULL "
                        "  AND json_extract(COALESCE(child.model_config, '{}'), '$._delegate_from') IS NULL "
                        "  AND json_extract(COALESCE(child.model_config, '{}'), '$._reset_from') IS NULL "
                        f"  AND NOT {_legacy_reset_child_sql('child', _RESET_END_REASONS_SQL)} "
                        "  AND COALESCE(child.source, '') != 'tool' "
                        "ORDER BY child.started_at DESC, child.id DESC LIMIT 1",
                        (current,),
                    ).fetchone()
                except Exception:
                    return session_id
                if child_row is None:
                    break
                child_id = child_row["id"] if hasattr(child_row, "keys") else child_row[0]
                if not child_id or child_id in seen:
                    break
                seen.add(child_id)
                current = child_id

            return best if best is not None else session_id

    def get_messages_as_conversation(
        self,
        session_id: str,
        include_ancestors: bool = False,
        include_inactive: bool = False,
        repair_alternation: bool = False,
        include_row_ids: bool = False,
        include_compacted: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Load messages in the OpenAI conversation format (role + content dicts).
        Used by the gateway to restore conversation history.

        By default only active messages are returned. Pass
        ``include_inactive=True`` to load soft-deleted (rewound) rows
        as well. See :meth:`rewind_to_message`.

        ``include_compacted=True`` additionally loads rows preserved by
        in-place compaction (``active=0, compacted=1``), deduped by
        :meth:`_dedupe_display_generations`. DISPLAY reads want this; the
        model-fed restore must NOT pass it, or a resumed session regrows the
        very history compaction just summarized away.

        ``repair_alternation=True`` runs ``repair_message_sequence`` over the
        loaded list before returning it. Callers that restore a session for
        LIVE REPLAY should pass it: a durable alternation violation (e.g. a
        ``user;user`` pair left by a turn that persisted no assistant row)
        otherwise re-triggers the pre-request defensive repair on every
        single request for the rest of the session's life — the repair
        mutates only the per-request list, never the stored transcript.
        Inspection/export consumers keep the default and see the transcript
        verbatim.
        """
        session_ids = [session_id]
        if include_ancestors and not self._is_explicit_branch_session(session_id):
            session_ids = self._session_lineage_root_to_tip(session_id)

        if include_inactive:
            active_clause = ""
        elif include_compacted:
            active_clause = " AND (active = 1 OR compacted = 1)"
        else:
            active_clause = " AND active = 1"
        with self._read_ctx() as conn:
            placeholders = ",".join("?" for _ in session_ids)
            rows = conn.execute(
                f"SELECT {self._CONVERSATION_ROW_COLUMNS} "
                f"FROM messages WHERE session_id IN ({placeholders})"
                # Order by AUTOINCREMENT id (true insertion order), NOT timestamp:
                # append_message stamps rows with time.time(), which is not
                # monotonic (WSL2, NTP steps, VM/laptop sleep resume). A later
                # row can carry an earlier timestamp than its predecessor, and
                # ORDER BY timestamp would then sort an assistant tool_calls row
                # after its tool response, breaking tool-call/response adjacency
                # and triggering an HTTP 400 on replay. This matches get_messages
                # — see c03acca50 for the original fix.
                f"{active_clause} ORDER BY id",
                tuple(session_ids),
            ).fetchall()

        if include_compacted:
            rows = self._dedupe_display_generations(rows)

        return self._rows_to_conversation(
            rows,
            session_id=session_id,
            include_ancestors=include_ancestors,
            repair_alternation=repair_alternation,
            include_row_ids=include_row_ids,
        )

    def _rows_to_conversation(
        self,
        rows,
        *,
        session_id: str,
        include_ancestors: bool,
        repair_alternation: bool,
        include_row_ids: bool = False,
        include_summary_markers: bool = False,
    ) -> List[Dict[str, Any]]:
        """Decode fetched message rows into the OpenAI conversation format.

        Extracted from get_messages_as_conversation so get_resume_conversations
        can build the model-fed and display views from one SELECT. ``rows`` must
        already be ordered by ``id`` (insertion order) and filtered to the
        desired session set / active state by the caller.
        """
        from hermes_state import _strip_background_review_harness, _strip_stale_tool_call_markers
        messages = []
        # Watermark rotation column-clones concurrent tail rows into the child
        # after the new summary, so the copies need not be adjacent. Index the
        # exact durable clone identity while decoding instead of rescanning the
        # whole accumulated lineage for every user row.
        exact_user_clones: Dict[Tuple[Any, str], Dict[str, Any]] = {}
        for row in rows:
            content = self._decode_content(row["content"])
            if row["role"] in {"user", "assistant"} and isinstance(content, str):
                content = sanitize_context(content).strip()
            msg = {"role": row["role"], "content": content}
            # Born durable (#92231): this dict is materialized FROM a durable
            # row, so stamp the persistence marker at the source instead of
            # relying on every restore caller to thread the loaded list back
            # through a flush as ``conversation_history=`` — any
            # identity-losing handoff (compression's durable-snapshot
            # adoption, incremental persists with no history arg) would
            # otherwise re-append the ENTIRE transcript on flush.
            # Underscore-prefixed like ``_row_id``: every transport strips it
            # before the wire, and compression's assembly copies deliberately
            # strip it so rotated child handoffs still flush (see
            # _fresh_compaction_message_copy).
            msg[_DB_PERSISTED_MARKER_KEY] = True
            # Durable per-message identity for surfaces that need to address a
            # specific row later (desktop reactions). OPT-IN: only the gateway
            # asks for it — every other consumer (ACP restore, export,
            # inspection) gets the transcript in its historical shape.
            # Underscore-prefixed so every transport's convert_messages()
            # strips it before the wire.
            if include_row_ids and row["id"] is not None:
                msg["_row_id"] = row["id"]
            # api_content is the byte-fidelity sidecar: the exact string sent
            # to the API when it differed from the clean content. Returned
            # VERBATIM — no sanitize_context, no strip — because the replay
            # path substitutes it for content to keep the provider prompt
            # cache prefix byte-stable across turns. Cleaning it here would
            # re-introduce the divergence it exists to remove.
            if row["api_content"]:
                msg["api_content"] = row["api_content"]
            if row["display_kind"]:
                msg["display_kind"] = row["display_kind"]
            if row["display_metadata"]:
                decoded = self._decode_display_metadata(row["display_metadata"])
                if decoded is not None:
                    msg["display_metadata"] = decoded
            if include_summary_markers and row["_compressed_summary"]:
                msg["_compressed_summary"] = True
            if row["timestamp"]:
                msg["timestamp"] = row["timestamp"]
            if row["tool_call_id"]:
                msg["tool_call_id"] = row["tool_call_id"]
            if row["tool_name"]:
                msg["tool_name"] = row["tool_name"]
            if row["effect_disposition"]:
                msg["effect_disposition"] = row["effect_disposition"]
            if row["tool_calls"]:
                try:
                    msg["tool_calls"] = json.loads(row["tool_calls"])
                except (json.JSONDecodeError, TypeError):
                    logger.warning("Failed to deserialize tool_calls in conversation replay, falling back to []")
                    msg["tool_calls"] = []
            # Surface the platform-side message id (e.g. yuanbao msg_id,
            # telegram update_id) so platform-specific flows like recall
            # can match by external identifier instead of having to fall
            # back to content-match heuristics.  Exposed as ``message_id``
            # for backward compatibility with the JSONL transcript shape.
            if row["platform_message_id"]:
                msg["message_id"] = row["platform_message_id"]
            if row["observed"]:
                msg["observed"] = True
            # Restore reasoning fields on assistant messages so providers
            # that replay reasoning (OpenRouter, OpenAI, Nous) receive
            # coherent multi-turn reasoning context.
            if row["role"] == "assistant":
                if row["finish_reason"]:
                    msg["finish_reason"] = row["finish_reason"]
                if row["reasoning"]:
                    msg["reasoning"] = row["reasoning"]
                if row["reasoning_content"] is not None:
                    msg["reasoning_content"] = row["reasoning_content"]
                if row["reasoning_details"]:
                    try:
                        msg["reasoning_details"] = json.loads(row["reasoning_details"])
                    except (json.JSONDecodeError, TypeError):
                        logger.warning("Failed to deserialize reasoning_details, falling back to None")
                        msg["reasoning_details"] = None
                if row["codex_reasoning_items"]:
                    try:
                        msg["codex_reasoning_items"] = json.loads(row["codex_reasoning_items"])
                    except (json.JSONDecodeError, TypeError):
                        logger.warning("Failed to deserialize codex_reasoning_items, falling back to None")
                        msg["codex_reasoning_items"] = None
                if row["codex_message_items"]:
                    try:
                        msg["codex_message_items"] = json.loads(row["codex_message_items"])
                    except (json.JSONDecodeError, TypeError):
                        logger.warning("Failed to deserialize codex_message_items, falling back to None")
                        msg["codex_message_items"] = None
            if include_ancestors:
                canonical_content, _is_composite = (
                    self._canonical_replayed_user_content(msg)
                )
                exact_clone_key = self._exact_replayed_user_clone_key(
                    msg.get("timestamp"), canonical_content
                )
                previous_exact = (
                    exact_user_clones.get(exact_clone_key)
                    if exact_clone_key is not None
                    else None
                )
                duplicate = None
                if previous_exact is not None:
                    previous_index = next(
                        (
                            index
                            for index, candidate in enumerate(messages)
                            if candidate is previous_exact
                        ),
                        None,
                    )
                    if previous_index is not None:
                        duplicate = (previous_index, True)
                if duplicate is None:
                    duplicate = self._find_duplicate_replayed_user_message(
                        messages, msg
                    )
                if duplicate is not None:
                    duplicate_index, prefer_current = duplicate
                    if prefer_current:
                        # A rotated compression child can carry the same live
                        # ask as the parent row plus the only surviving summary
                        # scaffold. Keep the child carrier (and its durable row
                        # id), not the simpler ancestor copy.
                        messages.pop(duplicate_index)
                    else:
                        continue
            messages.append(msg)
            if include_ancestors and exact_clone_key is not None:
                exact_user_clones[exact_clone_key] = msg
        # DEFENSE-IN-DEPTH against background-review session pollution: a forked
        # skill/memory review that (in older builds, before the _persist_disabled
        # fix) shared the parent's session_id wrote its harness turn into this
        # real session. The harness is a user/system message instructing the
        # agent to "Review the conversation above and update the skill library /
        # save to memory" under a hard tool restriction; re-loading it as live
        # history makes the agent adopt the curator role and refuse the user's
        # actual task. Strip any such harness message AND the curator-mode
        # assistant reply immediately following it, so a polluted session
        # resumes clean even if stray rows exist.
        messages = _strip_background_review_harness(messages)
        # DEFENSE-IN-DEPTH against #78148: before that fix, a bare tool-call
        # marker (e.g. "[memory]") could get cached as a fallback and
        # persisted as if it were the model's real answer. Sessions written
        # before the fix can still carry those rows — clear the stray
        # content on load so replaying history doesn't re-teach the model
        # to keep emitting the marker. No-op for unaffected sessions.
        messages = _strip_stale_tool_call_markers(messages)
        if repair_alternation and messages:
            # Lazy import: hermes_state already depends on agent.* (see
            # sanitize_context above), but keep this optional path from
            # widening the import surface at module load.
            from agent.agent_runtime_helpers import repair_message_sequence

            repaired = repair_message_sequence(None, messages)
            if repaired:
                logger.info(
                    "Repaired %d message-alternation violation(s) while "
                    "restoring session %s — durable transcript kept them, "
                    "see repair_message_sequence",
                    repaired,
                    session_id,
                )
        return messages

    def get_resume_conversations(
        self, session_id: str
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Return ``(model_history, display_history)`` for a session resume in ONE SELECT.

        ``session.resume`` needs two projections of the same lineage:

        - ``model_history`` — the tip session's active rows, alternation-repaired
          (the live-replay working conversation). Equivalent to
          ``get_messages_as_conversation(session_id, repair_alternation=True)``.
        - ``display_history`` — the full compression lineage (ancestors → tip),
          verbatim, with replayed-user dedup. Explicit ``/branch`` sessions are
          excluded from this lineage because their own rows already contain the
          copied transcript; including the live parent's rows would let messages
          written to the original after the fork leak into the branch.

        The display projection also includes rows preserved by IN-PLACE
        compaction (``active=0, compacted=1``), deduped by
        :meth:`_dedupe_display_generations`. Without them a compacted
        conversation resumes showing only its summary plus the carried-forward
        tail — the user's own turns read as deleted even though every row is
        still on disk, and the REST transcript read (which has always included
        them) disagreed with this one about the same session (#92080).

        The display fetch already reads a superset of the model fetch (the tip
        rows are part of the lineage), so serving both from one lineage SELECT
        halves the resume's DB work versus two separate calls, with byte-identical
        output (see test_get_resume_conversations_matches_separate_reads).
        """
        session_ids = self._resume_lineage_ids(session_id)
        with self._read_ctx() as conn:
            placeholders = ",".join("?" for _ in session_ids)
            rows = conn.execute(
                f"SELECT session_id, {self._CONVERSATION_ROW_COLUMNS} "
                f"FROM messages WHERE session_id IN ({placeholders}) "
                # Compaction-archived rows (active=0, compacted=1) are display
                # history; Undo/Rewind rows (active=0, compacted=0) are not.
                "AND (active = 1 OR compacted = 1) "
                # ORDER BY id (insertion order) — see get_messages_as_conversation
                # for why timestamp ordering is unsafe.
                "ORDER BY id",
                tuple(session_ids),
            ).fetchall()

        # Tip rows are exactly the model-fed set (get_messages_as_conversation
        # with session_ids=[session_id]); filtering the lineage fetch preserves
        # their relative id order. The model projection stays active-only — it
        # is the compressed working context and must not regrow the history
        # compaction just summarized away.
        tip_rows = [r for r in rows if r["session_id"] == session_id and r["active"]]
        model_history = self._rows_to_conversation(
            tip_rows,
            session_id=session_id,
            include_ancestors=False,
            repair_alternation=True,
            include_row_ids=True,
            # Pre-compress checkpointing: the resumed model history must keep
            # the summary marker so checkpoint providers can exclude derivative
            # summaries after a process restart (marker survives restart).
            include_summary_markers=True,
        )
        display_history = self._rows_to_conversation(
            self._dedupe_display_generations(rows),
            session_id=session_id,
            include_ancestors=True,
            repair_alternation=False,
            include_row_ids=True,
        )
        return model_history, display_history

    def _resume_lineage_ids(self, session_id: str) -> List[str]:
        """Session ids a full (display) resume materializes for *session_id*.

        Compression continuations need their ended ancestors' rows for the
        display transcript; an explicit ``/branch`` copy already owns its
        transcript, so its lineage is itself alone. This is the ONE definition
        shared by the resume readers (``get_resume_conversations``,
        ``get_ancestor_display_prefix``) and the resume guard
        (``assert_resume_safe`` / ``get_resume_message_count``) — the guard must
        count exactly the rows a resume would load, never a superset.
        """
        if self._is_explicit_branch_session(session_id):
            return [session_id]
        return self._session_lineage_root_to_tip(session_id)

    def get_resume_message_count(
        self, session_id: str, *, tip_only: bool = False
    ) -> int:
        """Count the rows that a resume would materialize.

        ``tip_only=True`` counts the tip segment's ACTIVE rows — the set a
        model-history restore loads (``get_messages_as_conversation`` without
        ancestors, or the deferred Desktop resume that pages the display
        transcript over REST and never materializes the ancestor prefix in
        memory).

        Otherwise this counts the full-lineage DISPLAY set — active rows plus
        the compaction-archived rows ``get_resume_conversations`` now loads
        for the transcript. Counting only active rows here would let a
        heavily-compacted conversation pass a limit sized for a handful of
        live rows and then materialize tens of thousands.
        """
        session_ids = [session_id] if tip_only else self._resume_lineage_ids(session_id)
        active_clause = "active = 1" if tip_only else "(active = 1 OR compacted = 1)"
        placeholders = ",".join("?" for _ in session_ids)
        row = self._read_one(
            f"SELECT COUNT(*) FROM messages "
            f"WHERE session_id IN ({placeholders}) AND {active_clause}",
            tuple(session_ids),
        )
        return int(row[0] if row else 0)

    def assert_resume_safe(
        self,
        session_id: str,
        max_messages: Optional[int] = None,
        *,
        tip_only: bool = False,
    ) -> int:
        """Return resume row count or reject a transcript too large to load.

        ``max_messages=None`` resolves the limit from config
        (``sessions.max_resume_messages``); 0 disables the guard and returns
        the (bounded) count without raising.

        ``tip_only=True`` bounds only the tip segment's ACTIVE rows, for
        callers that never materialize the ancestor lineage or the
        compaction archive in memory (tip-only model restore, deferred
        Desktop resume whose display history is REST-paginated). A
        heavily-compressed conversation — 85 compaction segments and ~29k
        lineage rows behind a ~700-row tip — is exactly the shape compression
        is supposed to produce; counting its whole lineage against a limit
        sized for in-memory materialization rejected the healthiest sessions
        (Desktop Bot Chat stuck on "Waking up…" with code 4130) while the
        process would only ever have held the tip.

        The full (non-``tip_only``) bound counts the DISPLAY set — active plus
        compaction-archived rows — because that is what
        ``get_resume_conversations`` materializes for the transcript.
        """
        from hermes_state import SessionResumeTooLargeError, resolved_max_resume_messages
        if max_messages is None:
            max_messages = resolved_max_resume_messages()
        if max_messages < 0:
            raise ValueError("max_messages must be non-negative")
        if max_messages == 0:
            # Guard disabled by config — skip counting entirely. Every live
            # caller invokes this for its raise side effect and ignores the
            # return value, and an unbounded lineage COUNT here would do the
            # exact pathological work the disable exists to avoid.
            return 0
        session_ids = [session_id] if tip_only else self._resume_lineage_ids(session_id)
        active_clause = "active = 1" if tip_only else "(active = 1 OR compacted = 1)"
        placeholders = ",".join("?" for _ in session_ids)
        row = self._read_one(
            "SELECT COUNT(*) FROM ("
            f"SELECT 1 FROM messages WHERE session_id IN ({placeholders}) "
            f"AND {active_clause} LIMIT ?"
            ")",
            (*session_ids, max_messages + 1),
        )
        message_count = int(row[0] if row else 0)
        if message_count > max_messages:
            raise SessionResumeTooLargeError(
                message_count,
                max_messages,
                scope="in its tip segment" if tip_only else "across its lineage",
            )
        return message_count

    def get_ancestor_display_prefix(self, session_id: str) -> List[Dict[str, Any]]:
        """Return the ancestor-only display messages for a session lineage.

        These are messages from parent/grandparent sessions (compression
        ancestors) that appear in the display transcript but NOT in the
        tip session's model-fed history. Used by ``session.resume`` to
        build the ``display_history_prefix`` that ``_live_session_payload``
        prepends to the live model history.

        Previously the prefix was calculated as
        ``display_history[:len(display) - len(raw)]``, but that overcounts
        when ``repair_message_sequence`` removes messages from the MIDDLE
        of the tip history (e.g. verification candidates collapsed by the
        consecutive-assistant merge) — the length difference includes both
        ancestor messages AND repair-removed tip messages, but the slice
        only captures the first N display messages (which are tip messages
        when there are no ancestors), causing duplication. This method
        returns ONLY the genuine ancestor messages, identified by
        ``session_id != tip_session_id``. (#65919)
        """
        session_ids = self._resume_lineage_ids(session_id)
        if len(session_ids) <= 1:
            return []
        with self._read_ctx() as conn:
            placeholders = ",".join("?" for _ in session_ids)
            rows = conn.execute(
                f"SELECT session_id, {self._CONVERSATION_ROW_COLUMNS} "
                f"FROM messages WHERE session_id IN ({placeholders}) "
                # Display read: compaction-archived rows included, Undo/Rewind
                # rows excluded (see get_resume_conversations).
                "AND (active = 1 OR compacted = 1) "
                "ORDER BY id",
                tuple(session_ids),
            ).fetchall()
        rows = self._dedupe_display_generations(rows)
        ancestor_ids = {
            int(row["id"])
            for row in rows
            if row["session_id"] != session_id and row["id"] is not None
        }
        if not ancestor_ids:
            return []
        lineage = self._rows_to_conversation(
            rows,
            session_id=session_id,
            include_ancestors=True,
            repair_alternation=False,
            include_row_ids=True,
        )
        prefix: List[Dict[str, Any]] = []
        for message in lineage:
            if message.get("_row_id") not in ancestor_ids:
                continue
            projected = message.copy()
            projected.pop("_row_id", None)
            prefix.append(projected)
        return prefix

    def get_conversation_root(self, session_id: str) -> str:
        """Return the ROOT id of *session_id*'s lineage chain.

        The root is the stable "conversation id": context compression
        rotates ``session_id`` to a new segment linked via
        ``parent_session_id``, and delegate subagents hang off their
        parent the same way. Walking to the root gives every segment of
        one user-facing conversation (and its delegation tree) a single
        identifier — used for Nous Portal ``conversation=`` usage tagging.
        Returns *session_id* unchanged when it has no recorded parent.
        """
        chain = self._session_lineage_root_to_tip(session_id)
        return (chain[0] if chain and chain[0] else session_id)

    @staticmethod
    def _canonical_replayed_user_content(
        msg: Dict[str, Any],
    ) -> Tuple[Any, bool]:
        """Return canonical live content and whether *msg* is composite."""
        if msg.get("role") != "user":
            return None, False

        from agent.context_compressor import split_user_originated_turn

        handoff, live_view = split_user_originated_turn(msg)
        is_composite = handoff is not None and live_view is not None
        return (
            live_view.get("content")
            if is_composite and live_view is not None
            else msg.get("content"),
            is_composite,
        )

    @staticmethod
    def _exact_replayed_user_clone_key(
        timestamp: Any, content: Any
    ) -> Optional[Tuple[Any, str]]:
        """Return a hashable key for a column-exact rotation clone."""
        if timestamp is None or content in (None, "", []):
            return None
        try:
            encoded = json.dumps(
                content,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        except (TypeError, ValueError):
            return None
        return timestamp, encoded

    @staticmethod
    def _find_duplicate_replayed_user_message(
        messages: List[Dict[str, Any]], msg: Dict[str, Any]
    ) -> Optional[Tuple[int, bool]]:
        """Return an adjacent replay duplicate and whether *msg* must win.

        Compression rotation may persist the current ask once in the parent
        and again inside a composite child carrier. Compare the canonical live
        payload for that carrier, while retaining the historical exact-string
        dedupe for ordinary replayed users. The child carrier wins because it
        owns both the current durable row identity and the retained scaffold.
        """
        from hermes_state import SessionDB
        if msg.get("role") != "user":
            return None

        content, prefer_current = SessionDB._canonical_replayed_user_content(msg)
        if content in (None, "", []):
            return None

        for index in range(len(messages) - 1, -1, -1):
            prev = messages[index]
            if prev.get("role") == "user":
                prev_content, prev_is_composite = (
                    SessionDB._canonical_replayed_user_content(prev)
                )
                if prev_content == content and (
                    prefer_current
                    or prev_is_composite
                    or isinstance(content, str)
                ):
                    return index, prefer_current
            if prev.get("role") == "assistant" and (prev.get("content") or prev.get("tool_calls")):
                return None
        return None

    def get_active_message_ids(self, session_id: str) -> List[int]:
        """Return the ordered physical ids pinned by rewind CAS checks.

        Conversation projections intentionally omit legacy background-review
        harness rows.  Destructive rewinds must nevertheless pin every active
        physical row so the caller snapshot matches the transaction-local
        comparison in :meth:`rewind_to_message`.
        """
        rows = self._read_all(
            "SELECT id FROM messages "
            "WHERE session_id = ? AND active = 1 ORDER BY id",
            (session_id,),
        )
        return [int(row[0]) for row in rows]

    @staticmethod
    def _active_transcript_counts(conn, session_id: str) -> tuple[int, int]:
        """Return active message/tool-call counts inside the caller's txn."""
        rows = conn.execute(
            "SELECT tool_calls FROM messages "
            "WHERE session_id = ? AND active = 1",
            (session_id,),
        ).fetchall()
        tool_call_count = 0
        for row in rows:
            raw = row[0]
            if not raw:
                continue
            try:
                decoded = json.loads(raw) if isinstance(raw, str) else raw
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(decoded, list):
                tool_call_count += len(decoded)
            elif decoded:
                tool_call_count += 1
        return len(rows), tool_call_count

    def rewind_to_message(
        self,
        session_id: str,
        target_message_id: int,
        *,
        preserve_compaction_handoff: bool = False,
        expected_active_ids: Optional[List[int]] = None,
        expected_target_content: Any = None,
    ) -> Dict[str, Any]:
        """Soft-delete all messages with id >= ``target_message_id`` in *session_id*.

        The target message itself becomes inactive as well so the caller
        can pre-fill it as the next user prompt without it appearing
        twice in the replayed transcript.  Rewound rows are kept on
        disk with ``active=0`` for audit / forensic inspection — use
        :meth:`get_messages` with ``include_inactive=True`` to see them.

        Returns a dict::

            {
                "rewound_count": int,    # number of rows newly flipped to active=0
                "target_message": dict,  # full row dict of the target
                "new_head_id":   int|None  # id of the last still-active row, or None
            }

        Raises ``ValueError`` if the target message does not exist in
        *session_id* or if its role is not ``"user"``.  With
        ``preserve_compaction_handoff=True``, a composite summary carrier is
        split inside the same write transaction: its original row is archived
        and its canonical hidden handoff scaffold is inserted as the new head.
        That opt-in result also contains ``replacement_message_id``.

        ``expected_active_ids`` optionally pins the ordered active row set.
        ``expected_target_content`` additionally pins the selected canonical
        live-user payload.  Both checks run inside the write transaction before
        any row or counter mutation.  Presentation-only metadata changes (for
        example Desktop reactions) deliberately do not invalidate a rewind.
        A live cross-process turn lease always refuses the rewind; expired or
        provably dead holders are reclaimed inside the mutation transaction.

        Always increments ``sessions.rewind_count`` — even when the
        target is already inactive — so the counter accurately reflects
        the number of rewind operations performed against the session.
        Idempotent on the ``active`` flag: re-rewinding past the same
        target is a no-op on row state but still bumps the counter.
        """

        def _do(conn):
            # Rewind changes the active transcript and must honor the same
            # compression/closed-parent and cross-process turn guards as
            # append writers.
            self._check_transcript_write_guards(
                conn,
                session_id,
                None,
                reject_active_turn_lease=True,
                reject_active_compression_lock=True,
            )

            if expected_active_ids is not None:
                active_rows = conn.execute(
                    "SELECT id FROM messages "
                    "WHERE session_id = ? AND active = 1 ORDER BY id",
                    (session_id,),
                ).fetchall()
                active_ids = [int(active_row[0]) for active_row in active_rows]
                if active_ids != expected_active_ids:
                    raise RuntimeError(
                        "active transcript changed before the rewind could be persisted"
                    )

            row = conn.execute(
                "SELECT * FROM messages WHERE id = ? AND session_id = ?",
                (target_message_id, session_id),
            ).fetchone()
            if row is None:
                raise ValueError(
                    f"message {target_message_id} not found in session {session_id}"
                )
            target_row = dict(row)
            if target_row.get("role") != "user":
                raise ValueError(
                    f"rewind target must be a 'user' message (got role="
                    f"{target_row.get('role')!r}, id={target_message_id})"
                )

            replacement_message_id: Optional[int] = None
            replacement: Optional[Dict[str, Any]] = None
            if preserve_compaction_handoff or expected_target_content is not None:
                if not target_row.get("active"):
                    raise ValueError("rewind target is not active")
                from agent.context_compressor import split_user_originated_turn

                split_target = target_row.copy()
                split_target["content"] = self._decode_content(
                    split_target.get("content")
                )
                split_target["display_metadata"] = self._decode_display_metadata(
                    split_target.get("display_metadata")
                )
                handoff, live_view = split_user_originated_turn(split_target)
                if live_view is None:
                    raise ValueError("rewind target is not a user-originated turn")
                live_content = live_view.get("content")
                if isinstance(live_content, str):
                    live_content = sanitize_context(live_content).strip()
                if (
                    expected_target_content is not None
                    and live_content != expected_target_content
                ):
                    raise RuntimeError(
                        "rewind target changed before it could be persisted"
                    )
                if preserve_compaction_handoff and handoff is None:
                    raise ValueError(
                        "preserve_compaction_handoff requires an active composite carrier"
                    )
                replacement = handoff if preserve_compaction_handoff else None

            cursor = conn.execute(
                "SELECT id FROM messages "
                "WHERE session_id = ? AND id >= ? AND active = 1",
                (session_id, target_message_id),
            )
            ids = [r[0] for r in cursor.fetchall()]
            if ids:
                placeholders = ",".join("?" for _ in ids)
                conn.execute(
                    f"UPDATE messages SET active = 0 WHERE id IN ({placeholders})",
                    ids,
                )
            if replacement is not None:
                self._insert_message_rows(conn, session_id, [replacement])
                inserted = conn.execute("SELECT last_insert_rowid()").fetchone()
                replacement_message_id = int(inserted[0])
            conn.execute(
                "UPDATE sessions SET rewind_count = COALESCE(rewind_count, 0) + 1 "
                "WHERE id = ?",
                (session_id,),
            )
            message_count, tool_call_count = self._active_transcript_counts(
                conn, session_id
            )
            conn.execute(
                "UPDATE sessions SET message_count = ?, tool_call_count = ? "
                "WHERE id = ?",
                (message_count, tool_call_count, session_id),
            )
            head_row = conn.execute(
                "SELECT MAX(id) FROM messages WHERE session_id = ? AND active = 1",
                (session_id,),
            ).fetchone()
            new_head_id = (
                head_row[0] if head_row and head_row[0] is not None else None
            )
            return target_row, ids, new_head_id, replacement_message_id

        target_row, rewound, new_head_id, replacement_message_id = (
            self._execute_write(_do)
        )

        # Decode content for callers (prefill the prompt buffer) without a
        # second fallible database operation after the transaction commits.
        target_row["content"] = self._decode_content(target_row.get("content"))

        result = {
            "rewound_count": len(rewound),
            "target_message": target_row,
            "new_head_id": new_head_id,
        }
        if preserve_compaction_handoff:
            result["replacement_message_id"] = replacement_message_id
        return result

    def message_count(self, session_id: str = None) -> int:
        """Count messages, optionally for a specific session."""
        with self._read_ctx() as conn:
            if session_id:
                cursor = conn.execute(
                    "SELECT COUNT(*) FROM messages WHERE session_id = ?", (session_id,)
                )
            else:
                cursor = conn.execute("SELECT COUNT(*) FROM messages")
            return cursor.fetchone()[0]

    def has_platform_message_id(
        self, session_id: str, platform_message_id: str
    ) -> bool:
        """Check if a message with the given platform_message_id exists.

        Uses the idx_messages_platform_msg_id partial index for efficient
        lookup. Used by the gateway's transient-failure dedupe guard (#47237)
        to skip re-persisting a user message that was already saved on a
        prior retry of the same inbound platform message.
        """
        return self._read_one(
            "SELECT 1 FROM messages "
            "WHERE session_id = ? AND platform_message_id = ? LIMIT 1",
            (session_id, platform_message_id),
        ) is not None

    def _is_explicit_fork_child_row(self, session: Dict[str, Any]) -> bool:
        """True when ``session`` is a branch, delegate, or tool child of its parent.

        Markers only count as a fork when they point at ``parent_session_id``.
        Compression copies ``model_config`` onto the continuation
        (``publish_compression_child`` callers pass
        ``agent._session_init_model_config``), so a delegate's continuation
        carries ``_delegate_from=<the delegate's own parent>``. Presence-only
        matching would treat that real continuation as a fork — the same
        misclassification ``_NON_CONTINUATION_CHILD_FILTER_SQL`` already
        avoids by binding both markers to the queried parent.
        """
        if session.get("source") == "tool":
            return True
        raw = session.get("model_config")
        if not raw:
            return False
        try:
            cfg = json.loads(raw) if isinstance(raw, str) else raw
        except (TypeError, json.JSONDecodeError):
            return False
        if not isinstance(cfg, dict):
            return False
        parent_id = session.get("parent_session_id")
        branched = cfg.get("_branched_from")
        delegated = cfg.get("_delegate_from")
        if parent_id:
            return branched == parent_id or delegated == parent_id
        return branched is not None or delegated is not None

    def is_explicit_fork_child(self, session_id: str) -> bool:
        """True when ``session_id`` is a /branch, delegate, or tool child row.

        Read-only public view of :meth:`_is_explicit_fork_child_row` for
        callers that must respect the fork boundary without re-implementing
        its marker rules (``agent/prompt_cache_scope.py`` keeps a declared
        conversation key from crossing it). A missing row is not a fork.
        """
        session = self.get_session(session_id)
        return bool(session and self._is_explicit_fork_child_row(session))

    def latest_conversation_boundary(
        self, session_key: str, source: str
    ) -> Optional[int]:
        """How many conversation boundaries this routing peer has crossed.

        A boundary is a row this peer ended at an intentional conversation
        break — the ``_RESET_END_REASONS`` set (``/new``, ``/switch``, idle,
        daily, suspended, resume_pending_expired).  That is the same fence
        :meth:`find_latest_gateway_session_for_peer` refuses to reach behind,
        so the two agree on where one conversation stops and the next begins
        and cannot drift.

        The peer is ``(session_key, source)``, the SAME identity tuple recovery
        uses — never the key alone.  ``X-Hermes-Session-Key`` accepts any
        authenticated caller-supplied string, so an API conversation may
        legally carry the same key as a Telegram row in one database; keying
        on the string alone would let a ``/new`` on that unrelated row rotate
        this conversation's affinity identity while recovery correctly refuses
        to cross the same line.

        Returns the count, or ``None`` when this peer has never been reset.

        The value comes from ``conversation_generations``, which
        :meth:`_bump_conversation_generation` advances inside the transaction
        that writes each boundary — NOT from an aggregate over the session
        rows.  An aggregate cannot prove non-reuse: ``delete_session()``
        orphans children and deletes the row, and bulk prune selects ended
        rows, so ``COUNT``/``MAX`` over boundaries can return a pair it already
        emitted and hand a new conversation a retired affinity identity.  It is
        also wall-clock-free, so a backwards NTP correction cannot reorder it.

        Databases upgraded mid-conversation start at no generation and take
        their first one from the next boundary written; a conversation that
        reset before the upgrade shares its predecessor's scope once, which
        costs a warm prompt-cache bucket and never crosses an identity.

        These rows are never garbage-collected, by design: dropping one resets
        the peer to "no generation", so its next boundary writes ``1`` again
        and re-issues a scope a retired conversation already used — the ABA
        this counter exists to prevent.  See the schema comment in
        ``hermes_state_common.py``.
        """
        if not session_key or not source:
            return None
        row = self._read_one(
            "SELECT generation FROM conversation_generations "
            "WHERE source = ? AND session_key = ?",
            (source, session_key),
        )
        if row is None or row["generation"] is None:
            return None
        generation = int(row["generation"])
        return generation if generation > 0 else None

    def clear_messages(self, session_id: str) -> None:
        """Delete all messages for a session and reset its counters."""
        def _do(conn):
            conn.execute(
                "DELETE FROM messages WHERE session_id = ?", (session_id,)
            )
            conn.execute(
                "UPDATE sessions SET message_count = 0, tool_call_count = 0 WHERE id = ?",
                (session_id,),
            )
        self._execute_write(_do)

    def purge_stale_tool_call_markers(
        self, *, dry_run: bool = False, backup: bool = True
    ) -> Dict[str, Any]:
        """Permanently clear bare tool-call marker content (e.g. "[memory]")
        left in the ``messages`` table by sessions persisted before the
        #78148 fix in ``agent.conversation_loop``.

        ``_strip_stale_tool_call_markers`` already repairs this in memory on
        every session load (see ``_rows_to_conversation``), so running this
        is optional — but for long-lived sessions the same rows get
        re-scanned and re-repaired on every resume, which is wasted work
        and keeps the contaminated bytes sitting in the DB (and in any
        downstream cache/backup snapshot of it) indefinitely. This rewrites
        the affected rows once, in place.

        Only the ``content`` column is touched — ``role``, ``tool_calls``,
        and every other column on the row are left exactly as they are, so
        provider tool_call/tool_result pairing is unaffected.

        Unlike the in-memory repair, this UPDATE is permanent and can't be
        undone from within the DB. Since ``backup`` defaults to True, a
        timestamped full snapshot is taken via ``VACUUM INTO`` (safe against
        a live connection, unlike the raw-copy ``_backup_db_file`` used for
        malformed-schema repair) before any row is touched — mirroring
        ``repair_state_db_schema``'s backup-by-default convention for
        destructive state.db operations. No snapshot is taken when there is
        nothing to change.

        With ``dry_run=True``, reports the affected row count/ids without
        writing or backing up (read-only, no write lock taken).

        Returns ``{"dry_run": bool, "rows_affected": int, "row_ids": [...],
        "backup_path": str|None}``.
        """
        from hermes_state import _STALE_TOOL_CALL_MARKER_RE

        def _find_affected(conn) -> List[int]:
            cursor = conn.execute(
                "SELECT id, content FROM messages "
                "WHERE role = 'assistant' AND tool_calls IS NOT NULL AND tool_calls != ''"
            )
            affected: List[int] = []
            for row in cursor.fetchall():
                content = row["content"]
                if isinstance(content, str) and _STALE_TOOL_CALL_MARKER_RE.fullmatch(content.strip()):
                    affected.append(row["id"])
            return affected

        with self._read_ctx() as conn:
            affected_ids = _find_affected(conn)

        if dry_run:
            return {
                "dry_run": True,
                "rows_affected": len(affected_ids),
                "row_ids": affected_ids,
                "backup_path": None,
            }

        if not affected_ids:
            return {
                "dry_run": False,
                "rows_affected": 0,
                "row_ids": [],
                "backup_path": None,
            }

        backup_path: Optional[str] = None
        if backup:
            import datetime

            stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            dest = self.db_path.with_name(
                f"{self.db_path.name}.pre-clean-markers-backup-{stamp}"
            )
            with self._lock:
                self._conn.execute("VACUUM INTO ?", (str(dest),))
            backup_path = str(dest)
            logger.info("Backed up state.db to %s before clean-markers write", backup_path)

        def _do(conn):
            ids = _find_affected(conn)
            if ids:
                placeholders = ",".join("?" * len(ids))
                conn.execute(
                    f"UPDATE messages SET content = '' WHERE id IN ({placeholders})",
                    ids,
                )
            return ids

        affected_ids = self._execute_write(_do)
        if affected_ids:
            logger.info(
                "Permanently cleared %d stale tool-call marker row(s) in state.db (#78148)",
                len(affected_ids),
            )
        return {
            "dry_run": False,
            "rows_affected": len(affected_ids),
            "row_ids": affected_ids,
            "backup_path": backup_path,
        }
