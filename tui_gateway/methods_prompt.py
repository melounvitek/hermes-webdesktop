"""Prompt / attachment / respond JSON-RPC handlers.

Bodies are rebound onto server.py's globals at install time (see
method_ctx.bind_module), so they reference server.py globals bare.
"""

import contextlib

from .method_ctx import HandlerRegistry, bind_module

_registry = HandlerRegistry()
method = _registry.method
_profile_scoped = _registry.profile_scoped


_STALE_TARGET_MSG = "target user message is no longer in session history"


def _history_user_indices(history: list) -> list:
    """Indices of canonical live-user turns, including composite carriers."""
    from agent.context_compressor import user_originated_turn_view

    return [i for i, m in enumerate(history) if user_originated_turn_view(m) is not None]


def _message_row_id(msg: dict):
    """Parse durable SQLite row id from a history entry, or None."""
    raw = msg.get("_row_id")
    if raw is None:
        raw = msg.get("row_id")
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _mem_db_pair_agrees(mem, db_msg) -> bool:
    """True when a live-memory entry plausibly corresponds to a durable row.

    Positional trust needs evidence beyond equal lengths: roles must match,
    display-marker status must match (a marker on one side only shifts every
    later position), and an addressable user turn must show the same text.
    Multimodal content can't be compared cheaply — role/marker agreement suffices.
    """
    if not isinstance(mem, dict) or not isinstance(db_msg, dict):
        return False
    if mem.get("role") != db_msg.get("role"):
        return False
    if mem.get("role") == "user":
        from agent.context_compressor import user_originated_turn_view
        from agent.memory_manager import sanitize_context

        mem_view = user_originated_turn_view(mem)
        db_view = user_originated_turn_view(db_msg)
        if (mem_view is None) != (db_view is None):
            return False
        if mem_view is None:
            return bool(mem.get("display_kind")) == bool(db_msg.get("display_kind"))
        mem_content = mem_view.get("content")
        db_content = db_view.get("content")
        if isinstance(mem_content, str) and isinstance(db_content, str):
            if sanitize_context(mem_content).strip() != sanitize_context(
                db_content
            ).strip():
                return False
        return True
    if bool(mem.get("display_kind")) != bool(db_msg.get("display_kind")):
        return False
    return True


def _find_user_turn_by_row_id(history: list, target_row_id: int):
    """Return ``(user_ordinal, history_index)`` for ``target_row_id``, or None."""
    for u_ord, h_idx in enumerate(_history_user_indices(history)):
        if _message_row_id(history[h_idx]) == target_row_id:
            return u_ord, h_idx
    return None


def _load_durable_truncation_history(
    session: dict,
    fallback_sid: str = "",
    repair_alternation: bool = True,
):
    """Load the durable live-replay transcript, or None when it cannot be proven safe."""
    session_key = str(session.get("session_key") or fallback_sid or "")
    if not session_key:
        return []
    try:
        with _session_db(session) as db:
            get_conv = getattr(db, "get_messages_as_conversation", None)
            if not callable(get_conv):
                return None
            history = get_conv(
                session_key, repair_alternation=repair_alternation, include_row_ids=True,
            )
    except Exception:
        logger.debug(
            "prompt.submit: failed loading durable history for session %s", session_key,
            exc_info=True,
        )
        return None
    return history if isinstance(history, list) else None


def _resolve_truncate_row_id(session: dict, history: list, target_row_id: int):
    """Resolve ``truncate_before_row_id`` to ``(user_ordinal, history_index)``.

    Prefer in-memory ``_row_id``/``row_id`` stamps. When a live turn rewrote
    ``session["history"]`` without stamps, load the durable transcript with
    ``include_row_ids=True`` and map the matched user-turn ordinal onto the live
    list. Never falls back to a client-supplied ordinal — unknown row ids refuse.
    """
    hit = _find_user_turn_by_row_id(history, target_row_id)
    if hit is not None:
        return hit

    db_history = _load_durable_truncation_history(session)
    if db_history is None:
        return None

    # Heal missing stamps only when EVERY pair agrees (all-or-nothing). Equal
    # length alone is not alignment: the durable copy is alternation-repaired
    # (may merge/drop rows) while the live list is not and can carry
    # optimistic/marker rows; a stamp on a misaligned pair is sticky and
    # re-aims every later rewind at the wrong durable row.
    if len(db_history) == len(history) and all(
        _mem_db_pair_agrees(mem, db_msg)
        for mem, db_msg in zip(history, db_history)
    ):
        for mem, db_msg in zip(history, db_history):
            db_rid = _message_row_id(db_msg) if isinstance(db_msg, dict) else None
            if db_rid is not None and _message_row_id(mem) is None:
                mem["_row_id"] = db_rid
        hit = _find_user_turn_by_row_id(history, target_row_id)
        if hit is not None:
            return hit

    db_hit = _find_user_turn_by_row_id(db_history, target_row_id)
    if db_hit is None:
        return None
    db_ord, db_idx = db_hit
    mem_user_indices = _history_user_indices(history)
    if db_ord < 0 or db_ord >= len(mem_user_indices):
        return None
    mem_idx = mem_user_indices[db_ord]
    # Same-ordinal mapping across lists that can diverge (repair may have merged
    # a user;user pair): trust it only when the mapped live turn shows the same
    # content as the durable target — else refuse (caller fails closed, 4018).
    if not _mem_db_pair_agrees(history[mem_idx], db_history[db_idx]):
        return None
    return db_ord, mem_idx


def _coerce_truncate_int(rid, value, param_name="truncate_before_user_ordinal"):
    """``(int_value, error_response)`` for a client integer param. bool is refused
    like any non-integer: JSON ``true`` would int() to 1 and aim at the wrong turn."""
    if isinstance(value, bool):
        return None, _err(rid, 4004, f"{param_name} must be an integer")
    try:
        return int(value), None
    except (TypeError, ValueError):
        return None, _err(rid, 4004, f"{param_name} must be an integer")


def _reconcile_client_ordinal(
    rid, sid, client_ordinal, msg_ordinal, param_name, target_repr,
    prefix_user_count=0,
):
    """Cross-check a client ordinal against a resolved durable target.

    Returns ``(ordinal, error_response)``: the target's tip-relative ordinal when
    the client sent none or agreed, else the 4004/4030 refusal — a stale ordinal
    beside a *resolved* durable id is drift; never guess which the user meant.
    Client ordinals count the full displayed lineage, so after compression
    ``msg_ordinal + prefix_user_count`` is the SAME turn, not drift. The cut is
    always aimed by the durable target, so this can never re-aim a truncation.
    """
    if client_ordinal is None:
        return msg_ordinal, None
    ordinal, err = _coerce_truncate_int(rid, client_ordinal)
    if err is not None:
        return None, err
    if ordinal == msg_ordinal:
        return msg_ordinal, None
    if prefix_user_count > 0 and ordinal == msg_ordinal + prefix_user_count:
        return msg_ordinal, None
    logger.warning(
        "prompt.submit: REFUSED truncation due to ordinal mismatch for session %s "
        "(ordinal=%d, %s_ordinal=%d, %s=%s, prefix_user_count=%d). "
        "Stale truncate_before_user_ordinal detected.",
        sid,
        ordinal,
        param_name,
        msg_ordinal,
        param_name,
        target_repr,
        prefix_user_count,
    )
    return None, _err(
        rid,
        4030,
        f"truncate_before_user_ordinal ({ordinal}) does not match "
        f"{param_name} target turn ({msg_ordinal})",
    )


def _pending_reaction_notes(session: dict) -> str:
    """Note block for reactions added since the last turn, or "". Applied to the
    MODEL INPUT only, never the persisted prompt; each reaction is announced once
    (rows are stamped ``seen`` on read). Feature-gated (display.message_reactions)."""
    session_key = str(session.get("session_key") or "")
    if not session_key:
        return ""

    try:
        display = _load_cfg().get("display")
        if not (isinstance(display, dict) and bool(display.get("message_reactions", False))):
            return ""
    except Exception:
        return ""

    try:
        with _session_db(session) as db:
            if db is None:
                return ""
            pending = db.take_unseen_reactions(session_key, author="user")
    except Exception:
        logger.debug("Failed to read pending reactions", exc_info=True)
        return ""

    if not pending:
        return ""

    notes = []
    for entry in pending:
        snippet = (entry.get("text") or "").strip().replace("\n", " ")
        if len(snippet) > 120:
            snippet = snippet[:120] + "…"
        emoji = entry.get("emoji") or ""
        whose = "their own" if entry.get("role") == "user" else "your"
        if snippet:
            notes.append(f'[The user reacted {emoji} to {whose} message: "{snippet}"]')
        else:
            # Attachment-only / tool-call-only rows: no quote beats an empty quote.
            notes.append(f"[The user reacted {emoji} to {whose} earlier message]")

    return "\n".join(notes)


# ── prompt.submit pieces ────────────────────────────────────────────────────


def _typed_stop_phrase_response(rid, text):
    """End the voice chat when a bare stop phrase is TYPED while backend voice mode
    is on (typed twin of the spoken stop phrase, at the one server-side choke
    point). Returns the RPC reply, or None when this is a normal message. The
    desktop's renderer-owned voice chat never flips the backend flag and handles
    its own typed stop."""
    if not (isinstance(text, str) and _voice_mode_enabled()):
        return None
    try:
        from tools.voice_mode import is_voice_stop_phrase

        typed_stop = is_voice_stop_phrase(text)
    except Exception:
        typed_stop = False
    if not typed_stop:
        return None
    _end_voice_chat(stop_loop=True, stop_tts=True)
    _voice_emit("voice.transcript", {"stop_phrase": True, "typed": True})
    logger.info("prompt.submit: typed stop phrase — voice chat ended")
    return _ok(rid, {"voice_stopped": True})


_HOSTED_TASK_FIELDS = {"room_id", "task_id", "thread_id", "turn_id", "execution_generation"}


def _hosted_submit_error(rid, session, hosted_task, hosted_terminal_callback):
    """Validate the hosted-room turn proof carried by an internal submit."""
    if session.get("source") != "bot_room":
        return _err(rid, 4120, "hosted room turns require a bot_room session")
    if not isinstance(hosted_task, dict) or not callable(hosted_terminal_callback):
        return _err(rid, 4120, "invalid hosted room turn proof")
    if set(hosted_task) != _HOSTED_TASK_FIELDS or not all(
        isinstance(hosted_task.get(field), str) and hosted_task[field]
        for field in _HOSTED_TASK_FIELDS - {"execution_generation"}
    ) or not isinstance(hosted_task.get("execution_generation"), int):
        return _err(rid, 4120, "invalid hosted room turn proof")
    return None


def _legacy_group_fence_error(rid, session, params):
    """Older Desktop builds know the ``Group: <room-id>`` title but not the hosted
    authority marker; once a gateway owns that room a direct prompt would start a
    second renderer driver. Fence server-side instead of trusting the client."""
    title = str(session.get("title") or "")
    if not title.startswith("Group: "):
        return None
    room_id = title.removeprefix("Group: ").strip()
    if not room_id:
        return None
    try:
        from gateway.hosted_rooms import (
            HostedRoomError,
            RoomProbeUnavailableError,
            default_db_path,
            probe_hosted_room,
            probe_peer_room_reservation,
        )

        hosted = probe_hosted_room(default_db_path(), room_id=room_id)
        peer = False
        if not hosted:
            from hermes_constants import named_profile_home

            session_profile_home = named_profile_home(str(session.get("profile_home") or ""))
            requested_profile = (
                (
                    session_profile_home.name
                    if session_profile_home is not None
                    else ""
                )
                or str(params.get("profile") or "").strip()
                or str(_current_profile_name() or "default").strip()
            )
            peer = probe_peer_room_reservation(
                default_db_path(), room_id=room_id, target_profile=requested_profile,
            )
    except RoomProbeUnavailableError:
        return _err(rid, 5122, "Could not verify this group. Try again after the gateway recovers.")
    except HostedRoomError:
        # Legacy Desktop sessions used the display name after "Group: "; those
        # names are not hosted room ids.
        return None
    except Exception:
        return _err(rid, 5122, "Could not verify this group. Try again after the gateway recovers.")
    if hosted or peer:
        return _err(
            rid,
            4122,
            (
                "This room is managed by its gateway. "
                if hosted
                else "This room is managed by its home host. "
            )
            + "Update Hermes Desktop to continue it.",
        )
    return None


def _resolve_truncation_ordinal(rid, sid, session, params, history):
    """Resolve the truncation target to ``(ordinal, cut_index, err)``.

    Refusal precedence: malformed params (4004) → unconfirmed (4029, checked
    BEFORE target resolution so a leaked-state request never pays the durable
    read or heal-stamps live dicts) → unresolvable target (4018, fail closed —
    never degrade a missing row_id/message_id into an ordinal cut) → ordinal
    drift (4030) → ordinal-only on a durable session (4004).
    """
    truncate_user_ordinal = params.get("truncate_before_user_ordinal")
    truncate_message_id = params.get("truncate_before_message_id")
    truncate_row_id = params.get("truncate_before_row_id")

    target_row_id = None
    if truncate_row_id is not None:
        target_row_id, err = _coerce_truncate_int(rid, truncate_row_id, "truncate_before_row_id")
        if err is not None:
            return None, None, err
    client_ordinal = None
    if truncate_user_ordinal is not None:
        client_ordinal, err = _coerce_truncate_int(rid, truncate_user_ordinal)
        if err is not None:
            return None, None, err

    # An ordinal/id alone is not consent: a leftover ordinal on an ORDINARY
    # submit is field-for-field indistinguishable from a real rewind, and the
    # cut is a destructive replace_messages(). Only the client knows.
    if not is_truthy_value(params.get("confirm_truncate")):
        logger.warning(
            "prompt.submit: REFUSED unconfirmed truncation of session %s "
            "(%d messages held; ordinal=%s, row_id=%s, message_id=%s). "
            "The client attached truncation parameters without "
            "confirm_truncate — likely stale truncation parameters on "
            "an ordinary submit.",
            sid,
            len(history),
            client_ordinal,
            target_row_id,
            truncate_message_id,
        )
        return None, None, _err(
            rid,
            4029,
            "truncation parameters require confirm_truncate=true; "
            "an ordinary prompt.submit must not drop session history "
            "(update your Hermes client if a rewind was intended)",
        )
    # Client ordinals count the full displayed lineage; after compression the
    # tip segment is session["history"] and the ancestors live in
    # display_history_prefix. Count the ancestor user turns once so client and
    # tip-relative ordinals can translate without loading ancestors into the tip.
    prefix_user_count = len(_history_user_indices(session.get("display_history_prefix") or []))
    user_indices = _history_user_indices(history)

    def _stale(resolved_ordinal=None):
        # Structured recovery fields: Desktop resyncs + retries on a stale target
        # and shows "compressed away" when segment_ordinal < 0 (ancestor-only).
        segment = (
            client_ordinal - prefix_user_count if client_ordinal is not None else resolved_ordinal
        )
        return None, None, _err(rid, 4018, _STALE_TARGET_MSG, data={
            "user_turn_count": len(user_indices), "ordinal": client_ordinal,
            "segment_ordinal": segment, "prefix_user_count": prefix_user_count,
        })

    if target_row_id is not None:
        found_match = _resolve_truncate_row_id(session, history, target_row_id)
        if found_match is None:
            logger.warning(
                "prompt.submit: target row_id %d not found for session %s "
                "(in-memory + durable); refusing truncation without fallback",
                target_row_id,
                sid,
            )
            return _stale()
        ordinal, err = _reconcile_client_ordinal(
            rid, sid, client_ordinal, found_match[0], "truncate_before_row_id", target_row_id,
            prefix_user_count=prefix_user_count,
        )
        if err is not None:
            return None, None, err
    elif truncate_message_id is not None:
        msg_id_str = str(truncate_message_id)
        found_match = next(
            (
                (u_ord, h_idx)
                for u_ord, h_idx in enumerate(user_indices)
                if history[h_idx].get("id") == msg_id_str
                or history[h_idx].get("message_id") == msg_id_str
            ),
            None,
        )
        if found_match is None:
            logger.warning(
                "prompt.submit: target message_id %s not found in history "
                "for session %s; refusing truncation without fallback",
                msg_id_str,
                sid,
            )
            return _stale()
        ordinal, err = _reconcile_client_ordinal(
            rid, sid, client_ordinal, found_match[0], "truncate_before_message_id", msg_id_str,
            prefix_user_count=prefix_user_count,
        )
        if err is not None:
            return None, None, err
    else:
        segment_ordinal = client_ordinal - prefix_user_count
        if segment_ordinal < 0 or segment_ordinal >= len(user_indices):
            return _stale()
        # Durability is a state.db property, not an optional annotation on the
        # live copy (resume paths historically omitted _row_id stamps). If the
        # durable state cannot be read, fail closed too: absence of proof is
        # not proof of an ephemeral conversation.
        has_stamped_user = any(
            _message_row_id(history[h_idx]) is not None for h_idx in user_indices
        )
        durable_history = (
            [] if has_stamped_user else _load_durable_truncation_history(session, sid)
        )
        if has_stamped_user or durable_history is None or durable_history:
            logger.warning(
                "prompt.submit: REFUSED ordinal-only truncation of durable "
                "session %s (ordinal=%d); truncate_before_row_id required",
                sid,
                client_ordinal,
            )
            return None, None, _err(
                rid,
                4004,
                "ordinal-only truncation is unsafe for durable session history; "
                "include truncate_before_row_id",
            )
        ordinal = segment_ordinal

    # Reject out-of-range on BOTH ends: a negative ordinal would hit Python's
    # negative indexing (user_indices[-1] → the LAST user turn) and persist the loss.
    if ordinal < 0 or ordinal >= len(user_indices):
        return _stale(resolved_ordinal=ordinal)
    return ordinal, user_indices[ordinal], None


def _row_ids_of(messages) -> set:
    return {row_id for message in messages if isinstance((row_id := _message_row_id(message)), int)}


def _persist_truncation(rid, sid, session, history, truncated, ordinal, requested_rebind_ids):
    """Write the truncated transcript BEFORE touching memory (fail closed).

    If replace_messages failed after session["history"] was rewritten, the turn
    would run against the short list while state.db kept the old tail; the
    append-only agent flush would then stack the new exchange on the "undone"
    turns — zombie history on resume. Writes through ``_session_db`` (the db that
    owns this session's row), never ``_get_db()``: a profile session's transcript
    lives in its own profile's state.db.

    Returns ``(err, survivor_user_row_ids, survivor_row_id_map)``.
    """
    survivor_user_row_ids = None
    survivor_row_id_map = None
    with _session_db(session) as db:
        if db is not None:
            try:
                # session_key can be NULL for old CLI-origin sessions; fall back to
                # sid or replace_messages(None) trips an FK violation.
                truncation_key = session.get("session_key") or sid
                old_active_row_ids = _row_ids_of(history)
                if requested_rebind_ids is not None:
                    # Row-id fallback can resolve a target the live list is too
                    # misaligned to stamp, and repair can merge a user;user pair
                    # keeping only the first id: read the authoritative
                    # un-repaired pre-write active-id set so a rewritten row is
                    # never mistaken for an untouched archived/ancestor row.
                    durable_rebind_history = _load_durable_truncation_history(
                        session, truncation_key, repair_alternation=False,
                    )
                    if durable_rebind_history is None:
                        raise RuntimeError("could not load durable row identities for truncation")
                    old_active_row_ids.update(_row_ids_of(durable_rebind_history))
                old_survivor_row_ids = [_message_row_id(message) for message in truncated]
                # active_only=True: in-place compaction keeps the pre-compaction
                # transcript as active=0 rows under this key; a bare replace would
                # DELETE that archive on every edit. archive_dropped=True: this
                # write is the last step before the dropped turns are gone —
                # soft-archive (active=0, still in FTS) so a mis-aimed cut is
                # recoverable.
                db.replace_messages(
                    truncation_key, truncated, active_only=True, archive_dropped=True,
                    reject_active_turn_lease=True,
                )
            except Exception as exc:
                logger.error(
                    "prompt.submit: replace_messages failed for session %s "
                    "(ordinal=%d); refusing turn so memory and DB stay "
                    "aligned: %s",
                    sid,
                    ordinal,
                    exc,
                    exc_info=True,
                )
                return _err(rid, 5008, f"failed to persist history truncation: {exc}"), None, None
            # replace_messages re-inserted the survivors as NEW rows and stamped
            # fresh _row_id values onto these dicts. Surface the surviving
            # user-turn ids (visible-user-ordinal order) so the client rebinds
            # its cached rowIds — else a second rewind sends the pre-rewind id
            # and the fail-closed resolver refuses with 4018. None entries mean
            # the client must drop its cached id for that turn.
            survivor_user_row_ids = [
                _message_row_id(truncated[i]) for i in _history_user_indices(truncated)
            ]
            if requested_rebind_ids is not None:
                survivor_row_id_map = {
                    str(old_row_id): new_row_id
                    for old_row_id, new_row_id in zip(
                        old_survivor_row_ids,
                        (_message_row_id(message) for message in truncated),
                    )
                    if isinstance(old_row_id, int)
                    and isinstance(new_row_id, int)
                    and old_row_id in requested_rebind_ids
                }
                for dropped_row_id in requested_rebind_ids.intersection(
                    old_active_row_ids
                ):
                    survivor_row_id_map.setdefault(str(dropped_row_id), None)
    return None, survivor_user_row_ids, survivor_row_id_map


def _truncate_history_for_submit(rid, sid, session, params, requested_rebind_ids):
    """Rewind/regenerate cut, under ``history_lock``. Returns
    ``(err, survivor_user_row_ids, survivor_row_id_map)``; on success
    ``session["history"]`` is replaced and ``history_version`` bumped."""
    history = _history_without_ephemeral_scaffolding(session.get("history", []))
    ordinal, cut_index, err = _resolve_truncation_ordinal(rid, sid, session, params, history)
    if err is not None:
        return err, None, None
    from agent.context_compressor import history_before_user_originated_turn

    truncated, _live_view = history_before_user_originated_turn(history, cut_index)
    # Second gate on top of confirm_truncate: ordinal 0 → history[:0] == [] and
    # replace_messages() DELETEs every durable row. Wiping the whole transcript
    # needs its own opt-in (legitimate restore/regenerate of the first turn).
    if (
        not truncated
        and history
        and not is_truthy_value(params.get("confirm_empty_truncate"))
    ):
        logger.warning(
            "prompt.submit: REFUSED empty truncation of session %s "
            "(%d messages would be wiped; ordinal=%d).",
            sid,
            len(history),
            ordinal,
        )
        return _err(
            rid,
            4028,
            "truncation would erase the entire session transcript; "
            "resubmit with confirm_empty_truncate=true if this is intended",
        ), None, None
    log_fn = logger.warning if not truncated else logger.info
    log_fn(
        "prompt.submit: truncating session %s history %d -> %d messages (ordinal=%d)",
        sid, len(history), len(truncated), ordinal,
    )
    err, survivor_user_row_ids, survivor_row_id_map = _persist_truncation(
        rid, sid, session, history, truncated, ordinal, requested_rebind_ids
    )
    if err is not None:
        return err, None, None
    session["history"] = truncated
    session["history_version"] = int(session.get("history_version", 0)) + 1
    return None, survivor_user_row_ids, survivor_row_id_map


def _survivor_fields(survivor_user_row_ids, survivor_row_id_map, requested_rebind_ids) -> dict:
    """Client rowId-rebind payload for a submit that truncated a durable session."""
    fields = {}
    if survivor_user_row_ids is not None and requested_rebind_ids is None:
        fields["survivor_user_row_ids"] = survivor_user_row_ids
    if survivor_row_id_map is not None:
        fields["survivor_row_id_map"] = survivor_row_id_map
    return fields


def _persist_session_row_for_submit(rid, session):
    """Lazily persist the DB row now that the user actually sent a message; a
    branch becomes real here (parent transcript copied as its seed). Returns an
    error reply — the only user-visible signal; desktop maps the string to a
    toast — or None. On failure the in-flight turn is released."""
    try:
        if _ensure_session_db_row(session) is False:
            return _err(
                rid,
                5072,
                "session storage unavailable: "
                f"{_db_error or 'state.db could not be opened'} — the message "
                "was not saved; repair state.db and try again",
            )
        _persist_branch_seed(session)
    except Exception as exc:
        from hermes_state import is_disk_full_error

        with session["history_lock"]:
            session["running"] = False
            session["last_active"] = time.time()
            _clear_inflight_turn(session)
        if is_disk_full_error(exc):
            return _err(
                rid,
                5070,
                "disk full: session storage could not be written — free some disk space and try again",
            )
        logger.warning("prompt.submit: session persist failed: %s", exc, exc_info=True)
        return _err(rid, 5071, f"session storage could not be written: {exc}")
    return None


def _run_after_agent_ready(rid, sid, session, text, display_kind, hosted_terminal_callback):
    """Turn thread body: patient wait for a deferred build (the message is already
    the accepted in-flight turn, so a slow build must not eat it), then run."""
    err = _wait_agent_for_prompt(session, rid, sid)
    if err:
        # Terminal frame + retained snapshot (not a bare "error" event): if the
        # client is disconnected, the snapshot is the only way resume shows this.
        _emit_terminal_turn_error(
            sid,
            session,
            (err.get("error") or {}).get("message", "agent initialization failed"),
            # Construction never reached the provider: local-runtime failure.
            error_surface={"layer": "runtime", "code": "agent_init_failed", "retryable": True},
        )
        with session["history_lock"]:
            session["running"] = False
            session["last_active"] = time.time()
        _emit("session.info", sid, _session_info(session.get("agent"), session))
        return
    with session["history_lock"]:
        if session.get("_turn_cancel_requested") or not session.get("running"):
            session["running"] = False
            _clear_inflight_turn(session)
            # Without this emit the turn vanishes silently: the client saw
            # {"status": "streaming"} but never gets message.start or error.
            _emit(
                "error",
                sid,
                {
                    "message": "Turn cancelled before the agent was ready"
                    if session.get("_turn_cancel_requested")
                    else "Session no longer running before the agent was ready"
                },
            )
            return
    _run_prompt_submit(
        rid, sid, session, text, display_kind=display_kind,
        terminal_callback=hosted_terminal_callback,
    )


@method("prompt.submit")
def _(rid, params: dict) -> dict:
    from hermes_cli.input_sanitize import sanitize_user_prompt_text

    sid = params.get("session_id", "")
    raw_text = params.get("text", "")
    text = sanitize_user_prompt_text(raw_text) if isinstance(raw_text, str) else raw_text
    # Off-screen sends (widget intents) type the persisted row so no client
    # renders a bubble. Whitelisted to "hidden": this RPC must not mint kinds.
    display_kind = "hidden" if params.get("display_kind") == "hidden" else None
    if (stopped := _typed_stop_phrase_response(rid, text)) is not None:
        return stopped
    if params.get("interrupted"):
        # Client-side barge-in (desktop VAD / typing over playback): latch it so
        # this turn's model message carries the interruption note.
        from tools.tts_streaming import mark_speech_interrupted

        mark_speech_interrupted()
    session, err = _sess_nowait(params, rid)
    if err:
        return err
    hosted_task = params.get("_hosted_task")
    hosted_terminal_callback = params.get("_hosted_terminal_callback")
    internal_hosted_submit = hosted_task is not None or hosted_terminal_callback is not None
    if internal_hosted_submit:
        err = _hosted_submit_error(rid, session, hosted_task, hosted_terminal_callback)
    else:
        err = _legacy_group_fence_error(rid, session, params)
    if err is not None:
        return err
    if (limit_message := _ensure_active_session_slot(sid, session)) is not None:
        # Refused HERE — before the busy queue, _ensure_session_db_row and
        # _start_agent_build — so a refusal leaves the session exactly as it was.
        # The reason travels as machine-readable data ("at capacity, retry" vs
        # "live owner, your write would interleave"), never as matched prose.
        reason = getattr(limit_message, "reason", None)
        return _err(rid, 4090, str(limit_message), {"reason": reason} if reason else None)
    # Rewritten on every submit: one session can be driven from the app window
    # and the HUD in turn, and a stale "hud" misinforms the model.
    session["client_surface"] = "hud" if params.get("surface") == "hud" else ""
    has_truncation = any(
        params.get(k) is not None
        for k in ("truncate_before_user_ordinal", "truncate_before_row_id", "truncate_before_message_id")
    )
    if has_truncation and isinstance(text, str):
        # A rewind replays what the transcript shows; a skill turn shows its
        # invocation, so re-expand it or `/work fix it` sends nine literal chars.
        text = _expand_skill_invocation_for_replay(text, str(session.get("session_key") or ""))
    isolation_cfg = _load_dashboard_process_isolation_config()
    turn_isolation = _session_uses_compute_host(session, isolation_cfg)
    if internal_hosted_submit and turn_isolation:
        return _err(rid, 4121, "hosted room turns do not support isolated compute workers yet")
    # Re-bind to the current client transport so streaming stays on the active
    # websocket even if a disconnect/fallback moved the session to stdio.
    if (t := current_transport()) is not None:
        session["transport"] = t
    while True:
        busy_transport = None
        with session["history_lock"]:
            if session.get("running"):
                if internal_hosted_submit:
                    return _err(rid, 4091, "hosted room member session is busy")
                # Queue a mid-turn prompt (and by default interrupt the live turn)
                # instead of rejecting. The provider interrupt must happen after
                # this lock is released: a non-interruptible tool may hold it.
                busy_transport = t or session.get("transport")
            else:
                break
        busy_response = _handle_busy_submit(
            rid, sid, session, text, busy_transport, queued=bool(params.get("queued")),
        )
        if busy_response is not None:
            return busy_response
        # The old turn finished between the two lock acquisitions: retry the
        # claim rather than strand this prompt in a queue whose drain already ran.

    survivor_user_row_ids = None
    survivor_row_id_map = None
    raw_rebind_ids = params.get("rebind_survivor_row_ids")
    requested_rebind_ids = (
        {
            row_id
            for row_id in raw_rebind_ids
            if isinstance(row_id, int) and not isinstance(row_id, bool)
        }
        if isinstance(raw_rebind_ids, list)
        else None
    )
    with session["history_lock"]:
        # A watch session's run lives in the PARENT turn, so its own running flag
        # is False; typing mid-run would build a second agent racing the child
        # on the same stored session. After the run completes, submitting is fine.
        if session.get("lazy") and _child_run_active(str(session.get("session_key") or "")):
            return _err(rid, 4009, "subagent still running — wait for it to finish")
        if is_truthy_value(params.get("confirm_truncate")) and not has_truncation:
            return _err(
                rid,
                4004,
                "confirm_truncate requires truncate_before_user_ordinal, truncate_before_message_id, or truncate_before_row_id",
            )
        if has_truncation:
            err, survivor_user_row_ids, survivor_row_id_map = _truncate_history_for_submit(
                rid, sid, session, params, requested_rebind_ids
            )
            if err is not None:
                return err
        session["running"] = True
        session["_turn_cancel_requested"] = False
        session["last_active"] = time.time()
        if internal_hosted_submit:
            session["_hosted_room_task"] = dict(hosted_task)
        _start_inflight_turn(session, text)

    survivor_fields = _survivor_fields(
        survivor_user_row_ids, survivor_row_id_map, requested_rebind_ids
    )
    if turn_isolation:
        isolated_response = _submit_prompt_to_compute_host(
            rid, sid, session, text, display_kind=display_kind
        )
        if not isolated_response.get("error"):
            # The truncation already happened inline above (memory + DB).
            isolated_response["result"].update(survivor_fields)
            return isolated_response
        logger.warning(
            "compute-host dispatch failed for session %s; falling back inline: %s", sid,
            isolated_response["error"].get("message", "unknown error"),
        )

    if (err := _persist_session_row_for_submit(rid, session)) is not None:
        return err
    # A completed FAILED build must not wedge the session: rebuild with fresh
    # provider resolution instead of replaying the cached failure forever.
    if not _restart_completed_failed_agent_build(
        sid, session, session.get("agent_ready")
    ):
        _start_agent_build(sid, session)

    run_thread = threading.Thread(
        target=lambda: _run_after_agent_ready(
            rid, sid, session, text, display_kind, hosted_terminal_callback
        ),
        daemon=True,
    )
    # Handle lets session.interrupt tell a live turn from a stuck `running` flag.
    session["_run_thread"] = run_thread
    run_thread.start()
    return _ok(rid, {"status": "streaming", **survivor_fields})


# ── attachments ─────────────────────────────────────────────────────────────


def _attached_image_result(session, image_path, **extra) -> dict:
    """Common ``{attached, path, count, ...meta}`` reply after queuing an image."""
    return {
        "attached": True, "path": str(image_path), "count": len(session["attached_images"]),
        **extra, **_image_meta(image_path),
    }


@method("clipboard.paste")
def _(rid, params: dict) -> dict:
    session, err = _sess_building(params, rid)
    if err:
        return err
    try:
        from hermes_cli.clipboard import has_clipboard_image, save_clipboard_image
    except Exception as e:
        return _err(rid, 5027, f"clipboard unavailable: {e}")

    session["image_counter"] = session.get("image_counter", 0) + 1
    img_dir = _session_images_dir(session)
    img_dir.mkdir(parents=True, exist_ok=True)
    img_path = (
        img_dir / f"clip_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{session['image_counter']}.png"
    )

    # Save-first (CLI keybinding parity): more robust than a has_image() precheck.
    if not save_clipboard_image(img_path):
        session["image_counter"] = max(0, session["image_counter"] - 1)
        msg = (
            "Clipboard has image but extraction failed"
            if has_clipboard_image()
            else "No image found in clipboard"
        )
        return _ok(rid, {"attached": False, "message": msg})

    session.setdefault("attached_images", []).append(str(img_path))
    return _ok(rid, _attached_image_result(session, img_path))


@method("image.attach")
def _(rid, params: dict) -> dict:
    session, err = _sess_building(params, rid)
    if err:
        return err
    raw = str(params.get("path", "") or "").strip()
    if not raw:
        return _err(rid, 4015, "path required")
    try:
        from cli import (
            _IMAGE_EXTENSIONS,
            _detect_file_drop,
            _resolve_attachment_path,
            _split_path_input,
        )

        dropped = _detect_file_drop(raw)
        if dropped:
            image_path = dropped["path"]
            remainder = dropped["remainder"]
        else:
            path_token, remainder = _split_path_input(raw)
            image_path = _resolve_attachment_path(path_token)
            if image_path is None:
                return _err(rid, 4016, f"image not found: {path_token}")
        if image_path.suffix.lower() not in _IMAGE_EXTENSIONS:
            return _err(rid, 4016, f"unsupported image: {image_path.name}")
        session.setdefault("attached_images", []).append(str(image_path))
        return _ok(rid, _attached_image_result(
            session, image_path,
            remainder=remainder,
            text=remainder or f"[User attached image: {image_path.name}]",
        ))
    except Exception as e:
        return _err(rid, 5027, str(e))


@method("image.attach_bytes")
def _(rid, params: dict) -> dict:
    """Attach an image from base64 bytes (remote client: its file isn't on our disk).
    Reply shape mirrors ``image.attach``. ``content_base64``/``data`` accept a
    ``data:image/...;base64,`` prefix; ``filename``/``ext`` hint the extension, else
    magic bytes decide (PNG/JPEG/GIF/WebP/BMP, fallback ``.png``)."""
    session, err = _sess_building(params, rid)
    if err:
        return err

    raw_b64 = str(params.get("content_base64") or params.get("data") or "").strip()
    if not raw_b64:
        return _err(rid, 4015, "content_base64 required")

    img_bytes, err = _decode_attach_payload(
        rid, raw_b64, mime_prefix="image/", max_bytes=_ATTACH_BYTES_MAX_BYTES,
        label="image", empty_msg="image is empty",
    )
    if err is not None:
        return err

    filename = str(params.get("filename", "") or "")
    ext_hint = str(params.get("ext", "") or "").strip().lower()
    if ext_hint and not ext_hint.startswith("."):
        ext_hint = "." + ext_hint
    ext = _sniff_image_ext(img_bytes, filename or (f"x{ext_hint}" if ext_hint else ""))
    if ext not in _allowed_image_extensions():
        return _err(rid, 4016, f"unsupported image extension: {ext}")

    try:
        img_path = _queue_attached_image(session, img_bytes, ext, prefix="upload")
    except Exception as e:
        return _err(rid, 5027, f"write failed: {e}")

    return _ok(rid, _attached_image_result(
        session, img_path,
        remainder="",
        text=f"[User attached image: {img_path.name}]",
        bytes=len(img_bytes),
    ))


@method("pdf.attach")
def _(rid, params: dict) -> dict:
    """Attach a PDF by rendering each page to PNG (``pdftoppm`` @150 DPI, poppler-utils;
    5028 if missing) and queuing the pages as images. Accepts a host ``path`` or
    base64 ``content_base64``. Caps: 50 MB / 25 pages per call."""
    import shutil
    import subprocess
    import tempfile

    session, err = _sess_building(params, rid)
    if err:
        return err

    if shutil.which("pdftoppm") is None:
        return _err(rid, 5028, "pdftoppm not installed (poppler-utils package required)")

    raw_path = str(params.get("path", "") or "").strip()
    raw_b64 = str(params.get("content_base64") or params.get("data") or "").strip()
    if not raw_path and not raw_b64:
        return _err(rid, 4015, "path or content_base64 required")

    with tempfile.TemporaryDirectory(prefix="pdf_attach_") as td:
        td_path = Path(td)
        if raw_b64:
            pdf_bytes, err = _decode_attach_payload(
                rid, raw_b64, mime_prefix="application/pdf", max_bytes=_PDF_ATTACH_MAX_BYTES,
                label="PDF", empty_msg="decoded PDF is empty",
            )
            if err is not None:
                return err
            if pdf_bytes[:5] != b"%PDF-":
                return _err(rid, 4017, "payload is not a PDF (missing %PDF- magic bytes)")
            pdf_path = td_path / "input.pdf"
            pdf_path.write_bytes(pdf_bytes)
            display_name = str(params.get("filename", "") or "uploaded.pdf")
        else:
            try:
                from cli import _resolve_attachment_path

                resolved = _resolve_attachment_path(raw_path)
            except Exception:
                resolved = None
            if resolved is None or not Path(resolved).is_file():
                return _err(rid, 4016, f"PDF not found: {raw_path}")
            if Path(resolved).suffix.lower() != ".pdf":
                return _err(rid, 4016, f"not a PDF: {Path(resolved).name}")
            if Path(resolved).stat().st_size > _PDF_ATTACH_MAX_BYTES:
                mb = _PDF_ATTACH_MAX_BYTES // (1024 * 1024)
                return _err(rid, 4018, f"PDF too large; cap is {mb} MB")
            pdf_path = Path(resolved)
            display_name = pdf_path.name

        try:
            first_page = int(params.get("first_page") or 1)
            last_page_param = params.get("last_page")
            last_page = int(last_page_param) if last_page_param is not None else None
        except (TypeError, ValueError):
            return _err(rid, 4015, "first_page/last_page must be integers")

        if first_page < 1:
            return _err(rid, 4015, "first_page must be >= 1")
        if last_page is None:
            last_page = first_page + _PDF_ATTACH_MAX_PAGES - 1
        if last_page < first_page:
            return _err(rid, 4015, "last_page must be >= first_page")
        if last_page - first_page + 1 > _PDF_ATTACH_MAX_PAGES:
            return _err(rid, 4019, f"page range exceeds cap of {_PDF_ATTACH_MAX_PAGES} pages per attach call")

        out_prefix = td_path / "page"
        argv = [
            "pdftoppm", "-png", "-r", "150", "-f", str(first_page), "-l", str(last_page),
            str(pdf_path), str(out_prefix),
        ]
        from hermes_cli._subprocess_compat import windows_hide_flags

        try:
            res = subprocess.run(
                argv, capture_output=True, text=True, timeout=120, stdin=subprocess.DEVNULL,
                # UTF-8 + lossy decode: non-UTF-8 child output must not crash the
                # gateway thread on locale-mismatched Windows.
                encoding="utf-8", errors="replace",
                creationflags=windows_hide_flags(),
            )
        except subprocess.TimeoutExpired:
            return _err(rid, 5028, "pdftoppm timed out (>120s)")
        if res.returncode != 0:
            tail = (res.stderr or res.stdout or "").strip().splitlines()[-3:]
            return _err(rid, 5028, "pdftoppm failed: " + " | ".join(tail))

        rendered = sorted(td_path.glob("page-*.png"))
        if not rendered:
            return _err(rid, 5028, "pdftoppm produced no pages (corrupt PDF?)")

        attached_pages = []
        for src in rendered:
            page_num = src.stem.split("-", 1)[-1]
            try:
                page_int = int(page_num)
            except ValueError:
                page_int = first_page + len(attached_pages)
            dst = _queue_attached_image(session, src.read_bytes(), ".png", prefix=f"pdf_p{page_num}")
            attached_pages.append({"path": str(dst), "page": page_int, **_image_meta(dst)})

        return _ok(
            rid,
            {
                "attached": True,
                "filename": display_name,
                "pages_attached": len(attached_pages),
                "pages": attached_pages,
                "count": len(session["attached_images"]),
                "text": f"[User attached PDF: {display_name} ({len(attached_pages)} page(s))]",
            },
        )


@method("file.attach")
def _(rid, params: dict) -> dict:
    """Stage a non-image file into the session workspace and return a
    workspace-relative ``@file:`` ref the agent's file tools can read. ``path`` is
    the client/host path (naming + local resolution); ``data_url`` carries the bytes
    when the path isn't visible to the gateway; ``name`` overrides the filename."""
    session, err = _sess_building(params, rid)
    if err:
        return err
    raw = str(params.get("path", "") or "").strip()
    data_url = str(params.get("data_url", "") or "").strip()
    name = str(params.get("name", "") or "").strip()
    if not raw and not data_url:
        return _err(rid, 4015, "path or data_url required")
    try:
        stored_path, uploaded = _stage_session_file_attachment(
            session, raw_path=raw, data_url=data_url, name=name
        )
        ref_path = _attachment_ref_path(session, stored_path)
        return _ok(
            rid,
            {
                "attached": True,
                "name": stored_path.name,
                "path": str(stored_path),
                "ref_path": ref_path,
                "ref_text": f"@file:{_format_ref_value(ref_path)}",
                "uploaded": uploaded,
            },
        )
    except Exception as e:
        return _err(rid, 5028, str(e))


@method("image.detach")
def _(rid, params: dict) -> dict:
    session, err = _sess_building(params, rid)
    if err:
        return err
    raw = str(params.get("path", "") or "").strip()
    if not raw:
        return _err(rid, 4015, "path required")
    images = session.setdefault("attached_images", [])
    before = len(images)
    session["attached_images"] = [path for path in images if path != raw]
    return _ok(
        rid,
        {
            "detached": len(session["attached_images"]) != before,
            "count": len(session["attached_images"]),
        },
    )


@method("input.detect_drop")
def _(rid, params: dict) -> dict:
    session, err = _sess_nowait(params, rid)
    if err:
        return err
    try:
        from cli import _detect_file_drop

        raw = str(params.get("text", "") or "")
        dropped = _detect_file_drop(raw)
        if not dropped:
            return _ok(rid, {"matched": False})

        drop_path = dropped["path"]
        remainder = dropped["remainder"]
        if dropped["is_image"]:
            session.setdefault("attached_images", []).append(str(drop_path))
            text = remainder or f"[User attached image: {drop_path.name}]"
            return _ok(
                rid,
                {
                    "matched": True,
                    "is_image": True,
                    "path": str(drop_path),
                    "count": len(session["attached_images"]),
                    "text": text,
                    **_image_meta(drop_path),
                },
            )

        text = f"[User attached file: {drop_path}]" + (f"\n{remainder}" if remainder else "")
        return _ok(
            rid,
            {
                "matched": True,
                "is_image": False,
                "path": str(drop_path),
                "name": drop_path.name,
                "text": text,
            },
        )
    except Exception as e:
        return _err(rid, 5027, str(e))


# ── side agents (background / btw / preview.restart) ────────────────────────


@contextlib.contextmanager
def _session_profile_home_scope(session):
    """Bind the session's HERMES_HOME override for an ephemeral agent thread: the
    ContextVar set on the session-create thread doesn't propagate, so a turn under
    a non-default profile would otherwise run against the wrong home."""
    profile_home = session.get("profile_home")
    home_token = set_hermes_home_override(profile_home) if profile_home else None
    try:
        yield
    finally:
        if home_token is not None:
            reset_hermes_home_override(home_token)


def _final_response_text(result) -> str:
    return (result.get("final_response", str(result)) if isinstance(result, dict) else str(result))


@method("prompt.background")
def _(rid, params: dict) -> dict:
    session, err = _sess(params, rid)
    if err:
        return err
    text, parent = params.get("text", ""), params.get("session_id", "")
    if not text:
        return _err(rid, 4012, "text required")
    task_id = f"bg_{uuid.uuid4().hex[:6]}"

    def run():
        session_tokens = _set_session_context(task_id, cwd=_session_cwd(session))
        try:
            from run_agent import AIAgent

            with _session_profile_home_scope(session):
                result = AIAgent(
                    **_background_agent_kwargs(session["agent"], task_id)
                ).run_conversation(
                    user_message=text,
                    task_id=task_id,
                )
            _emit(
                "background.complete", parent,
                {"task_id": task_id, "text": _final_response_text(result)},
            )
        except Exception as e:
            _emit("background.complete", parent, {"task_id": task_id, "text": f"error: {e}"})
        finally:
            _clear_session_context(session_tokens)

    threading.Thread(target=run, daemon=True).start()
    return _ok(rid, {"task_id": task_id})


@method("prompt.btw")
def _(rid, params: dict) -> dict:
    """Answer a side question without touching session history: snapshot the live
    conversation (in-flight ``_session_messages`` else ``session["history"]``) and
    run a one-shot auxiliary call (``agent/side_question.py``). History, role
    alternation and prompt cache stay untouched; answer arrives as ``btw.complete``."""
    session, err = _sess(params, rid)
    if err:
        return err
    text, parent = params.get("text", ""), params.get("session_id", "")
    if not text:
        return _err(rid, 4012, "text required")
    task_id = f"btw_{uuid.uuid4().hex[:6]}"

    agent = session.get("agent")
    snapshot = list(getattr(agent, "_session_messages", None) or session.get("history") or [])
    main_runtime = {
        "model": getattr(agent, "model", None), "provider": getattr(agent, "provider", None),
        "base_url": getattr(agent, "base_url", None), "api_key": getattr(agent, "api_key", None),
        "api_mode": getattr(agent, "api_mode", None),
    }

    def run():
        session_tokens = _set_session_context(task_id, cwd=_session_cwd(session))
        try:
            from agent.side_question import answer_side_question

            with _session_profile_home_scope(session):
                answer = answer_side_question(
                    text, snapshot, parent_agent=agent, main_runtime=main_runtime,
                )
            _emit(
                "btw.complete", parent,
                {"task_id": task_id, "question": text, "text": answer or ""},
            )
        except Exception as e:
            _emit(
                "btw.complete", parent,
                {"task_id": task_id, "question": text, "text": f"error: {e}"},
            )
        finally:
            _clear_session_context(session_tokens)

    threading.Thread(target=run, daemon=True).start()
    return _ok(rid, {"task_id": task_id})


_PREVIEW_RESTART_RULES = (
    "Restart exactly the app intended for the Preview URL, not Hermes Desktop itself.",
    "The Preview URL and port are the target. Preserve that target unless you conclude it is impossible.",
    "If the prior conversation shows a specific command that bound this URL/port, prefer re-running THAT exact command (in the same cwd) over guessing a new one.",
    "First inspect what process, if any, owns the Preview URL port. If a stale server exists, inspect its cwd and prefer that cwd over the Hermes/Desktop process cwd.",
    "The Current working directory is only a hint. Do not assume it is the preview app root when the port owner or files indicate another root.",
    "If the console shows a module-script MIME error for src/main.tsx or similar, a static server is serving source files. Do not restart python -m http.server or any dumb static server for that app.",
    "For module-script MIME failures, inspect package.json/vite config in the candidate app root and start the real dev server/bundler (for example npm/pnpm/yarn dev) so module transforms happen.",
    "Before declaring success, verify the Preview URL responds with the intended app, not Hermes Desktop. If it serves Hermes/Desktop UI or another unrelated app, stop that process and report failure.",
    "Do not modify files. Do not ask the user unless blocked.",
    "Prefer existing project scripts or commands when they are clear.",
    "If a stale process owns the needed port, handle it safely.",
    "Start long-running servers detached/in the background, then return immediately.",
    "Do not run a foreground dev server command that blocks this background task.",
    "Keep the final response short: what command/server was started, or why it could not be restarted.",
)

_PREVIEW_RESTART_HISTORY_NOTE = (
    "The conversation history above is from the user's main session — including the commands you (the assistant) previously ran to start servers, edit files, or check ports. Use it to figure out exactly which server should be running at this Preview URL. The user did not start a brand new task; recover what they had working."
)


@method("preview.restart")
def _(rid, params: dict) -> dict:
    session, err = _sess(params, rid)
    if err:
        return err

    url = str(params.get("url") or "").strip()
    cwd = str(params.get("cwd") or "").strip()
    context = str(params.get("context") or "").strip()

    if not url:
        return _err(rid, 4012, "url required")

    task_id = f"preview_{uuid.uuid4().hex[:6]}"
    parent = params.get("session_id", "")
    parent_history = _preview_restart_history(session)
    prompt = "\n".join(
        line
        for line in [
            "The desktop preview pane cannot load a local server URL.",
            f"Preview URL: {url}",
            f"Current working directory: {cwd or '(unknown)'}",
            f"Preview console:\n{context}" if context else "",
            _PREVIEW_RESTART_HISTORY_NOTE if parent_history else None,
            *_PREVIEW_RESTART_RULES,
        ]
        if line
    )

    # A malformed client path (embedded NUL, etc.) must not blow up the restart:
    # treat it as "no validated cwd".
    try:
        preview_cwd = os.path.abspath(os.path.expanduser(cwd)) if cwd else ""
        if preview_cwd and not os.path.isdir(preview_cwd):
            preview_cwd = ""
    except Exception:
        preview_cwd = ""

    def run():
        # Pin the validated preview cwd, else the parent workspace — never an
        # invalid client path (which would silently fall back to the launch dir).
        session_tokens = _set_session_context(task_id, cwd=(preview_cwd or _session_cwd(session)))
        try:
            from run_agent import AIAgent
            from tools.terminal_tool import register_task_env_overrides

            if preview_cwd:
                register_task_env_overrides(task_id, {"cwd": preview_cwd})

            history_note = (
                f" (with {len(parent_history)} parent-session messages of context)"
                if parent_history
                else ""
            )
            _emit(
                "preview.restart.progress", parent,
                {"task_id": task_id, "text": f"Starting hidden restart agent{history_note}"},
            )
            # Deliberately NOT closed through task-wide process cleanup: the whole
            # point is to leave a background server running under this task_id,
            # and AIAgent.close() would kill every process for it.
            with _session_profile_home_scope(session):
                result = AIAgent(
                    **_ephemeral_preview_agent_kwargs(session["agent"], task_id),
                    **_preview_restart_callbacks(parent, task_id),
                ).run_conversation(
                    user_message=prompt, task_id=task_id,
                    conversation_history=parent_history or None,
                )
            _emit(
                "preview.restart.complete", parent,
                {"task_id": task_id, "text": _final_response_text(result)},
            )
        except Exception as e:
            _emit("preview.restart.complete", parent, {"task_id": task_id, "text": f"error: {e}"})
        finally:
            try:
                from tools.terminal_tool import clear_task_env_overrides

                clear_task_env_overrides(task_id)
            except Exception:
                pass
            _clear_session_context(session_tokens)

    threading.Thread(target=run, daemon=True).start()
    return _ok(rid, {"task_id": task_id})


# ── late-answer RPCs for tool-driven UI cards ───────────────────────────────
# All use allow_expired=True: each tool's bounded wait (read_terminal 30s,
# setup_mcp 10min, clarify ...) can expire — its _pending entry popped — while the
# card is still visible (e.g. a WS reconnect dropped tool.complete). A late answer
# must resolve gracefully instead of the raw 4009 "no pending answer request".


@method("clarify.respond")
def _(rid, params: dict) -> dict:
    if proxied := _respond_compute_host_clarify(rid, params):
        return proxied
    return _respond(rid, params, "answer", allow_expired=True)


_LATE_RESPOND_KEYS = {
    "terminal.read.respond": "text", "preview.read.respond": "text", "preview.act.respond": "text",
    "window.read.respond": "text", "tour.respond": "text", "mcp.setup.respond": "result",
    "sudo.respond": "password", "secret.respond": "value",
}


def _late_respond(key: str):
    def handler(rid, params: dict) -> dict:
        return _respond(rid, params, key, allow_expired=True)

    return handler


for _name, _key in _LATE_RESPOND_KEYS.items():
    method(_name)(_late_respond(_key))
del _name, _key


# ── approvals ───────────────────────────────────────────────────────────────


def _approval_reply(rid, result_key, call):
    """``_ok({result_key: call(tools.approval)})``, 5004 on any failure."""
    try:
        import tools.approval as approval

        return _ok(rid, {result_key: call(approval)})
    except Exception as e:
        return _err(rid, 5004, str(e))


@method("approval.pending")
def _(rid, params: dict) -> dict:
    session, err = _sess(params, rid)
    if err:
        return err
    return _approval_reply(
        rid, "approvals", lambda a: a.list_gateway_approvals(session["session_key"])
    )


@method("approval.received")
def _(rid, params: dict) -> dict:
    session, err = _sess(params, rid)
    if err:
        return err
    request_id = params.get("request_id")
    if not isinstance(request_id, str) or not request_id:
        return _err(rid, 4006, "request_id required")
    return _approval_reply(
        rid, "acknowledged", lambda a: a.ack_gateway_approval(session["session_key"], request_id),
    )


def _approval_respond_session_fallback(params: dict):
    """Durable-identity fallback for ``approval.respond``: the desktop can answer
    with a stale live sid (runtime re-minted after a reconnect while the prompt
    stayed on screen). Try (1) the approval ``request_id`` — unique across sessions
    — against every live session's pending approvals, then (2) ``session_id`` as a
    STORED id mapped to its live record. Returns the live session or None."""
    request_id = str(params.get("request_id") or "")
    if request_id:
        try:
            from tools.approval import list_gateway_approvals

            with _sessions_lock:
                live = list(_sessions.items())
            for sid, session in live:
                key = str(session.get("session_key") or "")
                if not key:
                    continue
                for pending in list_gateway_approvals(key):
                    if str(pending.get("request_id") or "") == request_id:
                        return session
        except Exception:
            logger.debug("approval.respond request_id fallback failed", exc_info=True)
    target = str(params.get("session_id") or "")
    if target:
        try:
            live = _find_live_session_by_key(target)
            if live is not None:
                return live[1]
        except Exception:
            logger.debug("approval.respond stored-id fallback failed", exc_info=True)
    return None


@method("approval.respond")
def _(rid, params: dict) -> dict:
    session, err = _sess(params, rid)
    if err:
        # Session-not-found (4001) only: resolve by durable identity before failing.
        code = (err.get("error") or {}).get("code")
        if code != 4001:
            return err
        session = _approval_respond_session_fallback(params)
        if session is None:
            return err
    return _approval_reply(
        rid, "resolved",
        lambda a: a.resolve_gateway_approval(
            session["session_key"],
            params.get("choice", "deny"),
            resolve_all=params.get("all", False),
            request_id=params.get("request_id"),
        ),
    )


def register(server) -> None:
    """Publish this module's helpers + handlers onto ``server``, rebound to its globals."""
    bind_module(globals(), server, skip=("_",))
