"""Gateway slash commands that rotate, switch, fork or rewrite the session transcript:
/new, /resume, /sessions, /branch, /title, /save, /undo, /retry, /topic, /compress.

Split out of ``gateway/slash_commands.py``; bound onto ``GatewayRunner`` through
``GatewaySlashCommandsMixin``. Origin internals are imported lazily inside the bodies to avoid
the import cycle.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import logging
import os
import shlex
from typing import Optional, Union

from agent.i18n import t
from agent.turn_context import extract_api_content_sidecar
from gateway.config import Platform
from gateway.platforms.base import EphemeralReply, MessageEvent, MessageType
from gateway.session import SessionSource, build_session_key, is_shared_multi_user_session

# Log-record parity with gateway/run.py and the origin module.
logger = logging.getLogger("gateway.run")

# Upper bound on the off-loop agent-resource cleanup during a /new or /reset. A stuck teardown must
# not block the event loop; past this the reset proceeds and the cleanup finishes (or leaks) in its
# worker thread.
_RESET_CLEANUP_TIMEOUT_S = 30.0

# chat_type values whose session key is per-user (DM-like), incl. the unknown/blank case.
_DM_CHAT_TYPES = {"dm", "direct", "private", ""}

_BRANCH_COPIED_FIELDS = (
    "content", "tool_calls", "tool_call_id", "finish_reason", "reasoning", "reasoning_content",
    "reasoning_details", "codex_reasoning_items", "codex_message_items", "timestamp",
)


def _sattr(obj, name: str) -> str:
    """``str(getattr(obj, name) or "")`` — normalized identity field for origin comparisons."""
    return str(getattr(obj, name, "") or "")


def _manual_compression_reply_lines(summary: dict, compressor, focus_topic) -> list[str]:
    """Lines for the manual /compress confirmation, surfacing summariser/aux-model failures.

    ``_last_compress_aborted`` = no usable summary, messages unchanged. Provider exception text is
    force-redacted at this UI boundary even when global redaction is off. A configured aux model
    that failed and was recovered via main is an info note so the user can fix their config.
    """
    lines = [f"🗜️ {summary['headline']}"]
    if focus_topic:
        lines.append(t("gateway.compress.focus_line", topic=focus_topic))
    lines.append(summary["token_line"])
    if summary["note"]:
        lines.append(summary["note"])
    summary_err = getattr(compressor, "_last_summary_error", None)
    if summary_err:
        from agent.redact import redact_sensitive_text
        summary_err = redact_sensitive_text(summary_err, force=True)
    aux_fail_model = getattr(compressor, "_last_aux_model_failure_model", None)
    if getattr(compressor, "_last_compress_aborted", False):
        lines.append(t("gateway.compress.aborted", error=(summary_err or "unknown error")))
    elif aux_fail_model:
        lines.append(t(
            "gateway.compress.aux_failed",
            model=aux_fail_model,
            error=(getattr(compressor, "_last_aux_model_failure_error", None) or "unknown error"),
        ))
    return lines


def _compress_preview_reply(history, partial: bool, keep_last, focus_topic, agg_note: str) -> str:
    """``/compress --preview``: report what WOULD be compressed — no agent, no writes."""
    from agent.model_metadata import estimate_request_tokens_rough
    from hermes_cli.partial_compress import summarize_compress_preview

    pv_msgs = [
        {"role": m.get("role"), "content": m.get("content")}
        for m in history
        if m.get("role") in {"user", "assistant"} and m.get("content")
    ]
    report = summarize_compress_preview(
        pv_msgs, partial, keep_last, focus_topic, estimate_request_tokens_rough(pv_msgs)
    )
    lines = [f"🗜️ {line}" for line in report["lines"]]
    if agg_note:
        lines.append(agg_note)
    return "\n".join(lines)


def _reset_process_scoped_tool_state() -> None:
    """Drop env-passthrough and credential-file state at a conversation boundary (best-effort)."""
    with contextlib.suppress(Exception):
        from tools.env_passthrough import clear_env_passthrough
        clear_env_passthrough()
    with contextlib.suppress(Exception):
        from tools.credential_files import clear_credential_files
        clear_credential_files()


def _branch_row(msg: dict) -> dict:
    """Transcript row copied into a /branch child. Keeps the api_content sidecar so the branch's
    first turn replays the parent's exact wire bytes (warm provider prompt cache), not a cold prefill."""
    row = {k: msg.get(k) for k in _BRANCH_COPIED_FIELDS}
    row["role"] = msg.get("role", "user")
    row["tool_name"] = msg.get("tool_name") or msg.get("name")
    row["api_content"] = extract_api_content_sidecar(msg)
    return row


def _strip_resume_name(parts: list[str]) -> str:
    """Join the non-flag /resume tokens; strip literal ``<...>``/``[...]``/quotes typed from the
    usage hint (mirrors the CLI)."""
    name = " ".join(p for p in parts if p not in {"--all", "--cross-room"}).strip()
    if len(name) >= 2 and (name[0], name[-1]) in {("<", ">"), ("[", "]"), ('"', '"'), ("'", "'")}:
        name = name[1:-1].strip()
    return name


class GatewaySessionCommandsMixin:
    """Session-transcript slash commands (/new, /resume, /sessions, /branch, /title, /save, /undo, /retry, /topic, /compress)."""

    # ------------------------------------------------------------------ /new, /reset

    async def _cleanup_old_agent_for_reset(self, session_key: str) -> None:
        """Close the old agent's tool resources (sandboxes, browser daemons, subprocesses) before eviction.

        _cleanup_agent_resources is blocking and this runs ON the event loop (confirm-button click),
        so it is offloaded with a bounded timeout. wait_for cancels the await but not the worker
        thread — a wedged teardown keeps running (or leaks); the reset proceeds either way.
        """
        _old_agent = self._cached_agent_for(session_key)
        if _old_agent is None:
            return
        try:
            await asyncio.wait_for(
                self._run_in_executor_with_context(self._cleanup_agent_resources, _old_agent),
                timeout=_RESET_CLEANUP_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Agent resource cleanup for session %s exceeded %ss during "
                "/new reset; proceeding with reset (the worker thread is left "
                "to finish on its own). (#35994)",
                session_key, _RESET_CLEANUP_TIMEOUT_S,
            )
        except Exception as cleanup_exc:
            logger.warning(
                "Agent resource cleanup for session %s failed during /new "
                "reset: %s (#35994)",
                session_key, cleanup_exc,
            )

    async def _fire_session_reset_hooks(
        self, source: SessionSource, session_key: str, old_sid, new_sid
    ) -> None:
        """Session-boundary hooks: plugin finalize (off-loop + bounded), session:end/reset, on_session_reset."""
        platform_value = source.platform.value if source.platform else ""
        # Finalize hooks can block arbitrarily (observability trace exports) and this handler runs
        # on the gateway event loop (see GatewayRunner._finalize_session_off_loop).
        with contextlib.suppress(Exception):
            await self._finalize_session_off_loop(
                session_id=old_sid,
                platform=platform_value,
                reason="new_session",
                old_session_id=old_sid,
                new_session_id=new_sid,
            )
        hook_payload = {"platform": platform_value, "user_id": source.user_id, "session_key": session_key}
        await self.hooks.emit("session:end", dict(hook_payload))
        await self.hooks.emit("session:reset", dict(hook_payload))

    def _invoke_session_reset_lifecycle_hook(self, source: SessionSource, old_sid, new_sid) -> None:
        """Plugin on_session_reset hook (new session guaranteed to exist); best-effort."""
        try:
            from hermes_cli.lifecycle import invoke_hook as _invoke_hook
            _invoke_hook(
                "on_session_reset",
                session_id=new_sid,
                platform=source.platform.value if source.platform else "",
                reason="new_session",
                old_session_id=old_sid,
                new_session_id=new_sid,
            )
        except Exception:
            pass

    async def _handle_reset_command(self, event: MessageEvent) -> Union[str, EphemeralReply]:
        """Handle /new or /reset command."""
        source = event.source
        session_key = self._session_key_for_source(source)
        self._invalidate_session_run_generation(session_key, reason="session_reset")
        # Evict the running-agent slot now that the generation is bumped: the in-flight run's own
        # guarded release (old generation) returns False and would leave a zombie slot that silently
        # drops all later messages. Idempotent, so the run's finally calling it again is harmless.
        self._release_running_agent_state(session_key)

        # Snapshot the old entry so on_session_finalize can report the expiring session id.
        old_entry = self.session_store._entries.get(session_key)
        await self._cleanup_old_agent_for_reset(session_key)
        self._evict_cached_agent(session_key)
        # Conversation boundary: clear ALL conversation-scoped per-session state (model/reasoning
        # overrides, one-turn restores, model notes, last-resolved cache, /queue overflow) +
        # security state in one funnel call. See _CONVERSATION_SCOPED_STATE in gateway/run.py.
        self._clear_conversation_scope(session_key, reason="session_reset")

        # The old conversation's in-flight async delegations end WITH it: once the session id rotates
        # their completions have no live owner (orphaned payload on the shared queue, wasted tokens).
        # Interrupt by expiring durable session id (parent_session_id), routing key as legacy fallback.
        try:
            from tools.async_delegation import interrupt_for_session

            interrupt_for_session(
                session_key=session_key,
                parent_session_id=str(getattr(old_entry, "session_id", "") or ""),
                reason="session_reset",
            )
        except Exception:
            pass
        _reset_process_scoped_tool_state()

        new_entry = await self.async_session_store.reset_session(session_key)
        _old_sid = old_entry.session_id if old_entry else None
        await self._fire_session_reset_hooks(
            source, session_key, _old_sid, new_entry.session_id if new_entry else None
        )

        # Session config info for the user, scoped to the profile serving this source so a
        # multiplexed /new banner reports the profile's model, not the base config's.
        try:
            session_info = await asyncio.to_thread(self._reset_notice_session_info, source)
        except Exception:
            session_info = ""

        if new_entry:
            default_header = t("gateway.reset.header_default")
        else:
            # No existing session, just create one
            new_entry = await self.async_session_store.get_or_create_session(source, force_new=True)
            default_header = t("gateway.reset.header_new")
        header = await asyncio.to_thread(self._telegram_topic_new_header, source) or default_header

        _title_arg = event.get_command_args().strip()
        if _title_arg and self._session_db and new_entry:
            header = await self._reset_titled_header(header, new_entry.session_id, _title_arg)

        # Inside a Telegram DM topic lane, rewrite the (chat_id, thread_id) → session_id binding so
        # the next message uses the fresh session instead of switching back to the old one.
        if await asyncio.to_thread(self._is_telegram_topic_lane, source) and new_entry is not None:
            try:
                await asyncio.to_thread(self._record_telegram_topic_binding, source, new_entry)
            except Exception:
                logger.debug("Failed to rebind Telegram topic after /new", exc_info=True)

        self._invoke_session_reset_lifecycle_hook(
            source, _old_sid, new_entry.session_id if new_entry else None
        )
        try:
            from hermes_cli.tips import get_random_tip
            _tip_line = t("gateway.reset.tip", tip=get_random_tip())
        except Exception:
            _tip_line = ""
        if session_info:
            return EphemeralReply(f"{header}\n\n{session_info}{_tip_line}")
        return EphemeralReply(f"{header}{_tip_line}")

    async def _reset_titled_header(self, header: str, session_id: str, title_arg: str) -> str:
        """Apply ``/new <title>``: titled header on success, else the header plus a rejection note."""
        from hermes_state import SessionDB
        note = ""
        try:
            sanitized = SessionDB.sanitize_title(title_arg)
        except ValueError as e:
            sanitized = None
            note = t("gateway.reset.title_rejected", error=str(e))
        if sanitized:
            try:
                await self._session_db.set_session_title(session_id, sanitized)
                header = t("gateway.reset.header_titled", title=sanitized)
            except ValueError as e:
                note = t("gateway.reset.title_error_untitled", error=str(e))
            except Exception:
                pass
        elif not note:
            # sanitize_title returned empty (whitespace-only / unprintable)
            note = t("gateway.reset.title_empty_untitled")
        return header + note

    # ------------------------------------------------------- origin / ownership guards

    def _gateway_session_origin_for_id(self, session_id: str) -> Optional[SessionSource]:
        """Best-effort origin lookup for gateway session IDs."""
        lookup = getattr(type(self.session_store), "lookup_by_session_id", None)
        if callable(lookup):
            entry = lookup(self.session_store, session_id)
            return getattr(entry, "origin", None) if entry is not None else None
        # Test doubles and older stores may not expose the public lookup helper. Keep the Matrix
        # resume guard fail-closed if no origin can be resolved.
        entries = getattr(self.session_store, "_entries", {}) or {}
        for entry in entries.values():
            if getattr(entry, "session_id", None) == session_id:
                return getattr(entry, "origin", None)
        return None

    @staticmethod
    def _same_matrix_room(current: SessionSource, origin: Optional[SessionSource]) -> bool:
        # thread_id is part of the session key and Matrix scopes a turn to the current room/thread,
        # so a live session in another thread of the SAME room is a DIFFERENT session. Non-threaded
        # rooms compare "" == "" unchanged.
        return (
            origin is not None
            and origin.platform == Platform.MATRIX
            and current.platform == Platform.MATRIX
            and origin.chat_id == current.chat_id
            and _sattr(current, "thread_id") == _sattr(origin, "thread_id")
        )

    def _same_origin_chat(self, current: SessionSource, origin: Optional[SessionSource]) -> bool:
        """Platform-agnostic counterpart to ``_same_matrix_room``.

        Per-participant sessions (``build_session_key`` with the default ``group_sessions_per_user``)
        must be participant-scoped here too, else a co-member could resume another member's live
        session (IDOR). Only an explicitly shared group/thread (``is_shared_multi_user_session``) shares.
        """
        if origin is None or current is None:
            return False
        if origin.platform != current.platform or origin.chat_id != current.chat_id:
            return False
        # thread_id is part of the session key for every chat type, so threads of the same parent
        # chat are DIFFERENT sessions; require thread equality before any sharing logic.
        if _sattr(current, "thread_id") != _sattr(origin, "thread_id"):
            return False
        if _sattr(current, "chat_type").lower() in _DM_CHAT_TYPES:
            # DM: an equal non-empty chat_id IS the session key. build_session_key falls back to the
            # participant (``user_id_alt or user_id`` — Signal/Feishu key on user_id_alt) only when
            # there is NO chat_id; mirror that and fail closed on a missing/different participant.
            if _sattr(current, "chat_id"):
                return True
            cur_pid = str(current.user_id_alt or current.user_id or "")
            org_pid = str(origin.user_id_alt or origin.user_id or "")
            return bool(cur_pid) and cur_pid == org_pid
        # Non-DM: a shared key is one session for every participant; a per-user key must compare the
        # participant it is built from, failing closed when either side lacks one.
        if self._is_shared_session_source(current):
            return True
        cur_pid = current.user_id_alt or current.user_id
        org_pid = origin.user_id_alt or origin.user_id
        return bool(cur_pid and org_pid) and cur_pid == org_pid

    def _is_shared_session_source(self, source: SessionSource) -> bool:
        """Whether *source*'s session key is shared by every participant (not per-user).

        Mirrors build_session_key's isolation rules exactly, so the guards stay in lock-step with the key.
        """
        return is_shared_multi_user_session(
            source,
            group_sessions_per_user=getattr(self.config, "group_sessions_per_user", True),
            thread_sessions_per_user=getattr(self.config, "thread_sessions_per_user", False),
        )

    def _resume_caller_is_admin(self, source: SessionSource) -> bool:
        """Whether *source* is an EXPLICITLY-configured admin allowed cross-origin /resume or /sessions.

        Stricter than ``SlashAccessPolicy.is_admin()``, which returns True for every allowed caller
        when slash gating is DISABLED; cross-origin DATA ACCESS needs a real configured admin, else
        the default (no admin list) config would make every caller cross-origin-capable (IDOR).
        """
        try:
            from gateway.slash_access import policy_for_source
            policy = policy_for_source(self.config, source)
            uid = getattr(source, "user_id", None)
            return bool(policy.enabled and uid and policy.is_admin(uid))
        except Exception:
            return False

    def _persisted_row_proves_owner(self, source: SessionSource, row: dict) -> bool:
        """Whether a persisted (inactive) session *row* provably belongs to *source*'s session key.

        Rows once stored only source + user_id, so a same-user row could belong to a DIFFERENT chat;
        the persisted chat/thread origin closes that gap and legacy NULL rows fail closed. The table
        has no user_id_alt column, so an alt-keyed (Signal/Feishu) caller can never be proven by
        row_uid == caller_uid alone (CWE-639). Same platform alone is never ownership proof.
        """
        caller_src = source.platform.value if source.platform else None
        row_src = row.get("source")
        caller_uid = _sattr(source, "user_id")
        if not caller_uid:
            return False
        row_thread = str(row.get("thread_id") or "")
        if not (row_src and caller_src and str(row_src) == str(caller_src)
                and row_thread == _sattr(source, "thread_id")):
            return False  # blank/legacy source cannot prove the platform; other thread = other session
        row_uid = str(row.get("user_id") or "")
        row_chat = str(row.get("chat_id") or "")
        caller_chat = _sattr(source, "chat_id")
        caller_keys_on_alt = bool(_sattr(source, "user_id_alt"))
        if _sattr(source, "chat_type").lower() in _DM_CHAT_TYPES:
            # DMs key on user_id; a no-chat_id DM is keyed PURELY on the participant (alt-keyed
            # caller fails closed); with chat_id on both sides equality is the DM key.
            if caller_keys_on_alt and not (row_chat and caller_chat):
                return False
            return bool(row_uid) and row_uid == caller_uid and row_chat == caller_chat
        # Non-DM: the key includes chat_id — both sides must carry it and match (a legacy NULL-chat
        # row fails closed even when both normalize to "").
        if not (row_chat and caller_chat and row_chat == caller_chat):
            return False
        # A SHARED group/thread is one session for every participant: same-chat proof suffices (a
        # user-id check would block co-members). A per-user key still requires the same owner.
        if self._is_shared_session_source(source):
            return True
        if caller_keys_on_alt:
            return False
        return bool(row_uid) and row_uid == caller_uid

    async def _resume_target_allowed(
        self, source: SessionSource, target_id: str, allow_override: bool = False
    ) -> bool:
        """Whether *source* may resume the persisted session *target_id*.

        Generalizes the Matrix-only room guard to every adapter so a caller cannot bind to another
        user's/room's session (IDOR). Uses the live origin when the target is active, else the DB
        row must PROVE ownership or fail closed. Admin ``--all`` bypasses.
        """
        if allow_override and self._resume_caller_is_admin(source):
            return True
        # The live origin decides only when it resolves to a real SessionSource; a store that can't
        # resolve it (or a lookup error) falls through to the deterministic DB scoping.
        try:
            origin = self._gateway_session_origin_for_id(target_id)
        except Exception:
            origin = None
        if isinstance(origin, SessionSource):
            return self._same_origin_chat(source, origin)
        try:
            row = await self._session_db.get_session(target_id) or {}
        except Exception:
            return False
        return self._persisted_row_proves_owner(source, row)

    async def _resume_row_visible(
        self, source: SessionSource, row: dict, allow_all: bool
    ) -> bool:
        """Whether a titled-session listing *row* belongs to the caller's origin.

        Prevents cross-origin enumeration of session ids/previews via the numbered /resume list;
        keeps Matrix room-scoping, scopes every other platform to the caller unless admin ``--all``.
        Cross-room enumeration is cross-ORIGIN data access, so ``--all`` needs a configured admin.
        """
        if allow_all and self._resume_caller_is_admin(source):
            return True
        sid = str(row.get("id") or "")
        if source.platform == Platform.MATRIX:
            return self._same_matrix_room(source, self._gateway_session_origin_for_id(sid))
        return await self._resume_target_allowed(source, sid, allow_override=False)

    # ------------------------------------------------------------------ /retry, /undo

    async def _handle_retry_command(self, event: MessageEvent) -> str:
        """Handle /retry command - re-send the last user message."""
        # The canonical projection excludes bookkeeping rows (role=user + display_kind) and pure
        # handoffs while still recognizing a real ask embedded in a compaction carrier.
        from agent.context_compressor import (
            history_before_user_originated_turn,
            retryable_user_text,
            split_user_originated_turn,
            user_originated_turn_view,
        )

        source = event.source
        session_entry = await self.async_session_store.get_or_create_session(source)
        history = await self.async_session_store.load_transcript(session_entry.session_id)
        last_user_idx = next(
            (i for i in range(len(history) - 1, -1, -1) if user_originated_turn_view(history[i]) is not None),
            None,
        )
        if last_user_idx is None:
            return t("gateway.retry.no_previous")

        # Resolve the live text and the scaffold-preserving prefix before any transcript write.
        # Messaging retries cannot reconstruct attachments; reject media/unknown content untouched.
        try:
            truncated, live_view = history_before_user_originated_turn(history, last_user_idx)
            last_user_msg = retryable_user_text(live_view.get("content"))
            handoff, _ = split_user_originated_turn(history[last_user_idx])
        except ValueError as exc:
            return f"Cannot retry that message safely: {exc}"

        if handoff is not None:
            # A composite carrier is one physical row holding both the retained summary and the live
            # ask: the carrier-aware rewind archives that row/tail and inserts its pure scaffold
            # atomically, reselecting the latest carrier on the same snapshot so a concurrent newer
            # turn can never be removed while stale text is resent.
            try:
                rewind_result = await self.async_session_store.rewind_session(
                    session_entry.session_id, 1, require_retryable_composite=True,
                )
            except ValueError as exc:
                return f"Cannot retry that message safely: {exc}"
            if rewind_result is None:
                return "Retry failed; transcript was not changed."
            last_user_msg = rewind_result["target_text"]
        # After in-place compaction the pre-compaction transcript lives on as active=0/compacted=1
        # rows under this id; active_only preserves that archive.
        elif not await self.async_session_store.rewrite_transcript(
            session_entry.session_id, truncated, active_only=True, reject_active_turn_lease=True,
        ):
            return "Retry failed; transcript was not changed."
        session_entry.last_prompt_tokens = 0  # transcript was truncated

        retry_event = MessageEvent(
            text=last_user_msg,
            message_type=MessageType.TEXT,
            source=source,
            raw_message=event.raw_message,
            channel_prompt=event.channel_prompt,
        )
        return await self._handle_message(retry_event)

    async def _handle_undo_command(self, event: MessageEvent) -> str:
        """Handle /undo [N] — back up N user turns (default 1), soft-deleting the truncated rows and
        echoing the backed-up text. Evicts the cached agent so the next message rebuilds context
        from the active-only transcript (gateway analogue of the CLI's history surgery).
        """
        source = event.source
        n = 1
        raw_args = event.get_command_args().strip()
        if raw_args:
            try:
                n = max(1, int(raw_args.split()[0]))
            except (ValueError, IndexError):
                return t("gateway.undo.invalid_count", arg=raw_args.split()[0])

        session_entry = await self.async_session_store.get_or_create_session(source)
        result = await self.async_session_store.rewind_session(session_entry.session_id, n)
        if result is None:
            return t("gateway.undo.nothing")
        session_entry.last_prompt_tokens = 0  # transcript was truncated
        try:
            self._evict_cached_agent(build_session_key(source))
        except Exception as e:
            logger.debug("undo: cached-agent eviction skipped: %s", e)

        target_text = result["target_text"]
        preview = target_text[:200] + "..." if len(target_text) > 200 else target_text
        return t(
            "gateway.undo.removed",
            turns=result["turns_undone"],
            count=result["rewound_count"],
            preview=preview,
        )

    # --------------------------------------------------------------------- /compress

    async def _handle_compress_command(self, event: MessageEvent) -> str:
        """Profile-scoping wrapper around manual /compress.

        Multiplexed gateways resolve credentials through the fail-closed per-profile secret scope;
        slash dispatch (unlike ``_run_agent``) does not install it, so an unscoped /compress would
        raise ``UnscopedSecretError``. Single-profile gateways skip this.
        """
        if not getattr(getattr(self, "config", None), "multiplex_profiles", False):
            return await self._handle_compress_command_inner(event)

        from gateway.run import _profile_runtime_scope

        profile_home = self._resolve_profile_home_for_source(event.source)
        with _profile_runtime_scope(profile_home):
            return await self._handle_compress_command_inner(event)

    async def _compress_codex_app_server_session(
        self, session_key: str, session_id: str
    ) -> str:
        """Manual /compress for codex_app_server sessions.

        Compacts the LIVE cached agent's app-server thread (``thread/compact/start``, ``force=True``
        bypasses the ``codex_app_server_auto`` gate) and keeps the agent cached. Never builds a
        temporary agent or rewrites the mirror: neither can shrink the server-side thread.
        """
        from gateway.run import _AGENT_PENDING_SENTINEL

        agent = self._cached_agent_for(session_key)
        if (
            agent is None
            or agent is _AGENT_PENDING_SENTINEL
            or getattr(agent, "_codex_session", None) is None
        ):
            return (
                "🗜️ Nothing to compact: this session runs on the Codex "
                "app-server runtime, whose context lives in a Codex-owned "
                "thread that only exists while the agent is active. Send a "
                "message first, then /compress — or /reset to start fresh."
            )

        compressor = getattr(agent, "context_compressor", None)
        count_before = getattr(compressor, "compression_count", 0)
        try:
            await self._run_in_executor_with_context(
                lambda: agent._compress_context([], "", force=True)
            )
        except Exception as exc:
            return t("gateway.compress.failed", error=exc)
        if getattr(compressor, "compression_count", 0) > count_before:
            return (
                "🗜️ Codex app-server thread compacted (thread/compact). "
                "The transcript mirror is unchanged by design — the "
                "app-server now carries the compacted context."
            )
        return (
            "⚠️ Codex app-server compaction did not complete — the thread "
            "is unchanged. Check the app-server logs, retry /compress, or "
            "/reset for a clean session."
        )

    async def _handle_compress_command_inner(self, event: MessageEvent) -> str:
        """Handle /compress command -- manually compress conversation context.

        Optional ``/compress <focus>`` tells the summariser what to preserve, discarding the rest.
        """
        from hermes_cli.partial_compress import extract_compress_flags, parse_partial_compress_args

        source = event.source
        session_entry = await self.async_session_store.get_or_create_session(source)
        history = await self.async_session_store.load_transcript(session_entry.session_id)
        if not history or len(history) < 4:
            return t("gateway.compress.not_enough")

        # Strip --preview/--dry-run/--aggressive before positional parsing so the flags coexist with
        # the boundary-aware "here [N]" (partial) and focus-topic (full) forms.
        _raw_args = (event.get_command_args() or "").strip()
        _raw_args, _preview, _aggressive = extract_compress_flags(_raw_args)
        partial, keep_last, focus_topic = parse_partial_compress_args(_raw_args)

        _agg_note = ""
        if _aggressive:
            # LLM-free hard truncation is not supported on this surface — it would need its own
            # transcript-persistence branch outside the guarded _compress_context rotation machinery.
            _agg_note = t("gateway.compress.aggressive_unsupported")
            if not _preview:
                return _agg_note
        if _preview:
            return _compress_preview_reply(history, partial, keep_last, focus_topic, _agg_note)
        try:
            return await self._run_manual_compression(
                source, session_entry, history, partial, keep_last, focus_topic
            )
        except Exception as e:
            logger.warning("Manual compress failed: %s", e)
            return t("gateway.compress.failed", error=e)

    async def _run_manual_compression(
        self, source, session_entry, history: list, partial: bool, keep_last, focus_topic
    ) -> str:
        """Build a temporary agent, compress the transcript, persist, and describe the outcome."""
        from agent.conversation_compression import finalize_context_engine_compression_notification
        from agent.manual_compression_feedback import summarize_manual_compression
        from agent.model_metadata import estimate_request_tokens_rough
        from gateway.run import _platform_config_key
        from hermes_cli.partial_compress import (
            rejoin_compressed_head_and_tail,
            split_history_for_partial_compress,
        )

        session_key = self._session_key_for_source(source)
        # Preserve the platform + stable gateway session identity of a normal turn so external
        # context engines bind this agent to the original conversation, not a default "cli" host.
        platform_key = _platform_config_key(source.platform) if source.platform else None
        model, runtime_kwargs = self._resolve_session_agent_runtime(source=source, session_key=session_key)
        if str(runtime_kwargs.get("api_mode") or "").lower() == "codex_app_server":
            # The model's context is the server-side thread owned by the LIVE cached agent; a
            # temporary agent has none (and finally-eviction would destroy the real context).
            return await self._compress_codex_app_server_session(session_key, session_entry.session_id)
        if not runtime_kwargs.get("api_key"):
            return t("gateway.compress.no_provider")

        # FULL transcript (tool results included), like auto-compress: user/assistant-only starves
        # tool-result pruning and can trip the protect-first/last early-return.
        msgs = [m for m in history if m.get("role") in {"user", "assistant", "tool"}]
        # Boundary-aware split: only the head is summarized; the most recent `keep_last` exchanges
        # stay verbatim. The split snaps the tail to a user-turn start so role alternation holds.
        head, tail = msgs, []
        if partial:
            head, tail = split_history_for_partial_compress(msgs, keep_last)
            if not tail:  # degenerate split — fall back to full compression
                partial, head = False, msgs

        # Assign directly (not setdefault: a resolver value would be a stale placeholder, and it
        # avoids duplicate-kwarg TypeError); platform only when known so None -> "cli" holds.
        if platform_key is not None:
            runtime_kwargs["platform"] = platform_key
        runtime_kwargs["gateway_session_key"] = session_key

        tmp_agent = await self._build_manual_compression_agent(session_entry.session_id, model, runtime_kwargs)
        try:
            # Estimate with system prompt + tool schemas so the figure reflects real request
            # pressure; computed after tmp_agent is built so _cached_system_prompt/tools exist.
            _sys_prompt = getattr(tmp_agent, "_cached_system_prompt", "") or ""
            _tools = getattr(tmp_agent, "tools", None) or None
            approx_tokens = estimate_request_tokens_rough(msgs, system_prompt=_sys_prompt, tools=_tools)

            compressor = tmp_agent.context_compressor
            if not compressor.has_content_to_compress(head):
                return t("gateway.compress.nothing_to_do")

            # Not a bare run_in_executor: the profile secret scope is a contextvar and the default-
            # executor hop would drop it, failing the aux-client credential resolution closed.
            compressed, _ = await self._run_in_executor_with_context(
                lambda: tmp_agent._compress_context(
                    head,
                    "",
                    approx_tokens=approx_tokens,
                    focus_topic=focus_topic,
                    force=True,
                    defer_context_engine_notification=True,
                )
            )
            # Unchanged because a concurrent compression lock is held: say so instead of the
            # misleading "No changes from compression" no-op text.
            _lock_skipped = getattr(tmp_agent, "_compression_skipped_due_to_lock", None)
            if _lock_skipped is True or isinstance(_lock_skipped, str):
                from agent.manual_compression_feedback import describe_compression_lock_skip
                return describe_compression_lock_skip(_lock_skipped)

            if partial and tail:
                compressed = rejoin_compressed_head_and_tail(compressed, tail)
            await self._persist_manual_compression(tmp_agent, session_entry, source, compressed)
            finalize_context_engine_compression_notification(tmp_agent, committed=True)
            new_tokens = estimate_request_tokens_rough(compressed, system_prompt=_sys_prompt, tools=_tools)
            summary = summarize_manual_compression(
                msgs, compressed, approx_tokens, new_tokens, compression_state=compressor,
            )
        finally:
            finalize_context_engine_compression_notification(tmp_agent, committed=False)
            # Evict so the next turn rebuilds the system prompt from current files (SOUL.md, memory).
            self._evict_cached_agent(session_key)
            # Off-loop + bounded: temporary-agent teardown can block on subprocess/network/SQLite.
            await self._cleanup_agent_resources_off_loop(tmp_agent, context="manual compression")
        return "\n".join(_manual_compression_reply_lines(summary, compressor, focus_topic))

    async def _build_manual_compression_agent(self, session_id: str, model, runtime_kwargs: dict):
        """Build the throwaway AIAgent that performs a manual /compress rewrite of *session_id*."""
        from run_agent import AIAgent
        from gateway.run import _GATEWAY_HYGIENE_PLATFORM, _seed_hygiene_system_prompt
        from hermes_cli.config import load_config as _load_cfg
        from utils import is_truthy_value as _is_truthy

        # This helper runs outside the live session's fully initialized prompt environment and
        # _compress_context may persist its cached system prompt — restore the exact live-session
        # prompt so provider blocks are retained.
        session_row = None
        get_session = getattr(self._session_db, "get_session", None)
        if callable(get_session):
            try:
                session_row = await get_session(session_id)
            except Exception as exc:
                logger.warning(
                    "Manual compression could not restore the system prompt "
                    "for session %s: %s. Preserving an empty prompt so the "
                    "live turn rebuilds it with its configured providers.",
                    session_id,
                    exc,
                    exc_info=True,
                )

        # Lossy rewrite: when compression.checkpoint_required is on, the memory provider must be
        # loaded so _compress_context() can write the pre-compression checkpoint; otherwise keep
        # the historical fast path (no provider init).
        _checkpoint_required = _is_truthy(
            ((_load_cfg() or {}).get("compression") or {}).get("checkpoint_required"),
            default=False,
        )
        tmp_agent = AIAgent(
            **runtime_kwargs,
            model=model,
            max_iterations=4,
            quiet_mode=True,
            skip_memory=not _checkpoint_required,
            enabled_toolsets=["memory"],
            session_id=session_id,
            session_db=getattr(self._session_db, "_db", self._session_db),
        )
        _seed_hygiene_system_prompt(tmp_agent, session_row)
        # Keep the real source platform during construction so external context engines bind
        # correctly; if compression has to rebuild the prompt, stamp that provider-less fallback as
        # stale for the next real gateway turn.
        tmp_agent.platform = _GATEWAY_HYGIENE_PLATFORM
        tmp_agent._print_fn = lambda *a, **kw: None
        # close() must not end the newly rotated session — the gateway entry now points at the new
        # id and must remain open for the next user turn.
        tmp_agent._end_session_on_close = False
        return tmp_agent

    async def _persist_manual_compression(self, tmp_agent, session_entry, source, compressed) -> None:
        """Commit a manual /compress result to the session store.

        _compress_context either rotated (new continuation id — write compressed messages into the
        NEW session so the original stays searchable) or compacted in place (same id, transcript
        replaced). Persist BEFORE repointing the live session: a failed write is fatal so old history
        stays reachable. Only rewrite when rotation produced a NEW id: in-place compaction already
        archived + inserted rows and rewrite_transcript() (active_only=False) would DELETE the
        archived turns; an unchanged id without in-place means rotation FAILED.
        """
        new_session_id = tmp_agent.session_id
        if new_session_id != session_entry.session_id:
            if not await self.async_session_store.rewrite_transcript(new_session_id, compressed):
                raise RuntimeError(
                    f"failed to persist compressed transcript for session {new_session_id}"
                )
            session_entry.session_id = new_session_id
            await self.async_session_store._save()
            await asyncio.to_thread(
                self._sync_telegram_topic_binding,
                source, session_entry, reason="compress-command",
            )
        elif not getattr(tmp_agent, "_last_compaction_in_place", False):
            logger.warning(
                "Manual /compress: session rotation did not occur "
                "(session_id unchanged) and in-place mode is off — "
                "preserving original transcript instead of overwriting "
                "it (#44794)."
            )
        # Transcript changed — the stored token count is stale.
        await self.async_session_store.update_session(session_entry.session_key, last_prompt_tokens=0)

    # ------------------------------------------------------------------------ /topic

    async def _handle_topic_command(self, event: MessageEvent, args: str = "") -> str:
        """Handle /topic for Telegram DM user-managed topic sessions."""
        source = event.source
        if source.platform != Platform.TELEGRAM or source.chat_type != "dm":
            return t("gateway.topic.not_telegram_dm")
        if not self._session_db:
            return self._session_db_unavailable_reply()

        # /topic activates multi-session mode and mutates SQLite side tables; routes already
        # authorized the message, but defense in depth against unauthorized (non-allowlist) senders.
        auth_fn = getattr(self, "_is_user_authorized", None)
        if callable(auth_fn):
            try:
                if not auth_fn(source):
                    return t("gateway.topic.unauthorized")
            except Exception:
                logger.debug("Topic auth check failed", exc_info=True)

        args = event.get_command_args().strip()
        if args.lower() in {"help", "?", "-h", "--help"}:
            return self._telegram_topic_help_text()
        if args.lower() in {"off", "disable", "stop"}:
            return await self._disable_telegram_topic_mode_for_chat(source)
        if args:
            if not source.thread_id:
                return t("gateway.topic.restore_needs_topic")
            return await self._restore_telegram_topic_session(event, args)

        capabilities = await self._get_telegram_topic_capabilities(source)
        if capabilities.get("checked"):
            blocked_key = None
            if capabilities.get("has_topics_enabled") is False:
                blocked_key = "gateway.topic.topics_disabled"
            elif capabilities.get("allows_users_to_create_topics") is False:
                blocked_key = "gateway.topic.topics_user_disallowed"
            if blocked_key:
                # Debounce the BotFather screenshot: don't re-send on every /topic while blocked.
                if self._should_send_telegram_capability_hint(source):
                    await self._send_telegram_topic_setup_image(source)
                return t(blocked_key)

        profile_name = self._telegram_topic_profile_name(source)
        try:
            await self._session_db.enable_telegram_topic_mode(
                chat_id=str(source.chat_id),
                user_id=str(source.user_id),
                profile_name=profile_name,
                has_topics_enabled=capabilities.get("has_topics_enabled"),
                allows_users_to_create_topics=capabilities.get("allows_users_to_create_topics"),
            )
        except Exception as exc:
            logger.exception("Failed to enable Telegram topic mode")
            return t("gateway.topic.enable_failed", error=exc)

        if not source.thread_id:
            await self._ensure_telegram_system_topic(source)
            return await self._telegram_topic_root_status_message(source)
        try:
            binding = await self._session_db.get_telegram_topic_binding(
                chat_id=str(source.chat_id),
                thread_id=str(source.thread_id),
                profile_name=profile_name,
            )
        except Exception:
            logger.debug("Failed to read Telegram topic binding", exc_info=True)
            binding = None
        if not binding:
            return t("gateway.topic.thread_ready")
        session_id = str(binding.get("session_id") or "")
        try:
            title = await self._session_db.get_session_title(session_id)
        except Exception:
            title = None
        return t(
            "gateway.topic.bound_status",
            label=title or t("gateway.topic.untitled_session"),
            session_id=session_id,
        )

    # ------------------------------------------------------------------ /save, /title

    async def _handle_save_command(self, event: MessageEvent) -> str:
        """Handle /save — export the current session and send it as a document."""
        import tempfile
        from hermes_cli.session_export import (
            SAVE_USAGE,
            default_save_filename,
            normalize_save_format,
            render_session_for_save,
        )

        parts = event.get_command_args().split()
        redact = bool(parts) and parts[-1].lower() in ("redact", "--redact")
        if redact:
            parts = parts[:-1]
        if not parts:
            return SAVE_USAGE
        try:
            fmt = normalize_save_format(parts[0])
        except ValueError as e:
            return f"{e}\n\n{SAVE_USAGE}"

        source = event.source
        session_entry = await self.async_session_store.get_or_create_session(source)
        session_id = session_entry.session_id
        if not self._session_db:
            return "Session database not available."
        # The filename is echoed to the platform only — never trust path separators from chat input.
        filename = parts[1] if len(parts) > 1 else default_save_filename(session_id, fmt)
        filename = os.path.basename(filename) or default_save_filename(session_id, fmt)

        # self._session_db is an AsyncSessionDB — every forwarded call is awaited.
        export_data = await self._session_db.export_session(session_id)
        if not export_data:
            return f"No stored messages found for this session ({session_id})."
        if redact:
            from hermes_cli.session_export_md import redact_session_data
            export_data = redact_session_data(export_data)

        temp_dir = tempfile.mkdtemp(prefix="hermes_save_")
        temp_path = os.path.join(temp_dir, filename)
        try:
            # Off-loop: rendering a long session and writing it are CPU/disk-bound and scale with
            # transcript size (multi-MB); inline they stall every other chat on the event loop.
            def _render_and_write() -> None:
                rendered = render_session_for_save(export_data, fmt)
                with open(temp_path, "w", encoding="utf-8") as f:
                    f.write(rendered)

            await asyncio.to_thread(_render_and_write)
            adapter = self.get_adapter(source.platform)
            if not adapter:
                return "Platform adapter not found to send the document."
            await adapter.send_document(
                chat_id=source.chat_id,
                file_path=temp_path,
                caption=f"Session export: {filename}",
                file_name=filename,
            )
            return "Export complete."
        except Exception as e:
            logger.warning("Session /save failed: %s", e)
            return f"Error exporting session: {e}"
        finally:
            with contextlib.suppress(Exception):
                os.remove(temp_path)
                os.rmdir(temp_dir)

    async def _handle_title_command(self, event: MessageEvent) -> str:
        """Handle /title command — set or show the current session's title."""
        source = event.source
        session_entry = await self.async_session_store.get_or_create_session(source)
        session_id = session_entry.session_id
        if not self._session_db:
            return self._session_db_unavailable_reply()

        # The session may only exist in session_store so far (first command in a new session).
        if await self._session_db.get_session_title(session_id) is None:
            try:
                await self._session_db.create_session(
                    session_id=session_id,
                    source=source.platform.value if source.platform else "unknown",
                    user_id=source.user_id,
                    # Persist the messaging origin so a later /resume of this titled-but-inactive
                    # session can prove it belongs to the caller's chat/thread (IDOR scoping).
                    chat_id=source.chat_id,
                    chat_type=source.chat_type,
                    thread_id=source.thread_id,
                )
            except Exception:
                pass  # Session might already exist, ignore errors

        title_arg = event.get_command_args().strip()
        if not title_arg:
            title = await self._session_db.get_session_title(session_id)
            if title:
                return t("gateway.title.current_with_title", session_id=session_id, title=title)
            return t("gateway.title.current_no_title", session_id=session_id)
        try:
            from hermes_state import SessionDB
            sanitized = SessionDB.sanitize_title(title_arg)
        except ValueError as e:
            return t("gateway.shared.warn_passthrough", error=e)
        if not sanitized:
            return t("gateway.title.empty_after_clean")
        try:
            if not await self._session_db.set_session_title(session_id, sanitized):
                return t("gateway.title.not_found")
        except ValueError as e:
            return t("gateway.shared.warn_passthrough", error=e)
        # Propagate the user-chosen title to the visible Telegram forum topic name too (auto-
        # generated titles already rename the topic).
        schedule_rename = getattr(self, "_schedule_telegram_topic_title_rename", None)
        if callable(schedule_rename):
            try:
                await asyncio.to_thread(schedule_rename, source, session_id, sanitized)
            except Exception:
                logger.debug("Failed to rename Telegram topic from /title", exc_info=True)
        return t("gateway.title.set_to", title=sanitized)

    # -------------------------------------------------------------- /resume, /sessions

    async def _list_titled_sessions(self, source, session_key: str, allow_all: bool) -> list[dict]:
        """Titled sessions visible to the caller (origin-scoped unless admin ``--all``)."""
        widen = allow_all and self._resume_caller_is_admin(source)
        sessions = await self._session_db.list_sessions_rich(
            source=source.platform.value if source.platform else None,
            session_key=None if widen else session_key,
            limit=10,
        )
        titled = [s for s in sessions if s.get("title")][:10]
        return [s for s in titled if await self._resume_row_visible(source, s, allow_all)]

    async def _resolve_resume_target(self, source, session_key: str, name: str, allow_all: bool):
        """``(target_id, display_name)`` for a numbered choice, session id or title; else an error reply."""
        if name.isdigit():
            try:
                titled = await self._list_titled_sessions(source, session_key, allow_all)
            except Exception as e:
                logger.debug("Failed to list titled sessions for numeric resume: %s", e)
                return t("gateway.resume.list_failed", error=e)
            index = int(name)
            if index < 1 or index > len(titled):
                return t("gateway.resume.out_of_range", index=index)
            target = titled[index - 1]
            target_id, name = target.get("id"), target.get("title") or name
        else:
            # Direct session ID lookup first so `/resume <session_id>` works, not just `<title>`.
            session = await self._session_db.get_session(name)
            target_id = session["id"] if session else await self._session_db.resolve_session_by_title(name)
        if not target_id:
            return t("gateway.resume.not_found", name=name)
        # Compression creates child continuations that hold the live transcript; follow the chain
        # so gateway /resume matches CLI behavior.
        try:
            target_id = await self._session_db.resolve_resume_session_id(target_id)
        except Exception as e:
            logger.debug("Failed to resolve resume continuation for %s: %s", target_id, e)
        return target_id, name

    async def _resume_access_denied_reply(
        self, source, target_id: str, name: str, allow_all: bool, allow_cross_room: bool
    ) -> Optional[str]:
        """IDOR guard: a session id/title is a routing handle, not authority. Bind /resume to the
        caller's own room (Matrix) or platform/user/chat (every other adapter)."""
        if source.platform == Platform.MATRIX:
            target_origin = self._gateway_session_origin_for_id(target_id)
            if self._same_matrix_room(source, target_origin) or allow_cross_room:
                return None
            if target_origin is None:
                return t("gateway.resume.matrix_blocked_no_origin", name=name)
            return t(
                "gateway.resume.matrix_blocked_other_room",
                room=target_origin.chat_name or target_origin.chat_id,
                name=name,
            )
        if await self._resume_target_allowed(source, target_id, allow_override=(allow_all or allow_cross_room)):
            return None
        return t("gateway.resume.blocked_not_owner", name=name)

    async def _handle_resume_command(self, event: MessageEvent) -> str:
        """Handle /resume command — list or switch to a previous session."""
        if not self._session_db:
            return self._session_db_unavailable_reply()

        source = await asyncio.to_thread(self._normalize_source_for_session_key, event.source)
        session_key = self._session_key_for_source(source)
        try:
            parts = shlex.split(event.get_command_args().strip())
        except ValueError as exc:
            return t("gateway.resume.parse_error", error=exc)
        allow_all = "--all" in parts
        allow_cross_room = "--cross-room" in parts
        name = _strip_resume_name(parts)

        if not name:
            try:
                titled = await self._list_titled_sessions(source, session_key, allow_all)
                return self._resume_listing_reply(source, titled, allow_all)
            except Exception as e:
                logger.debug("Failed to list titled sessions: %s", e)
                return t("gateway.resume.list_failed", error=e)

        resolved = await self._resolve_resume_target(source, session_key, name, allow_all)
        if isinstance(resolved, str):
            return resolved
        target_id, name = resolved
        denied = await self._resume_access_denied_reply(source, target_id, name, allow_all, allow_cross_room)
        if denied is not None:
            return denied

        current_entry = await self.async_session_store.get_or_create_session(source)
        if current_entry.session_id == target_id:
            return t("gateway.resume.already_on", name=name)

        self._release_running_agent_state(session_key)
        new_entry = await self.async_session_store.switch_session(session_key, target_id)
        if not new_entry:
            return t("gateway.resume.switch_failed")
        # Conversation boundary: clear ALL conversation-scoped per-session state (model/reasoning
        # overrides, one-turn restores, model notes, last-resolved cache, /queue overflow) +
        # security state in one funnel call.
        self._clear_conversation_scope(session_key, reason="resume")
        # Evict so the next message rebuilds with the correct session_id end-to-end — otherwise the
        # cached AIAgent (and its memory provider, which cached _session_id at initialize()) keeps
        # writing to the wrong session. Mirrors /branch and /reset.
        self._evict_cached_agent(session_key)

        title = await self._session_db.get_session_title(target_id) or name
        history = await self.async_session_store.load_transcript(target_id)
        msg_count = len([m for m in history if m.get("role") == "user"]) if history else 0
        if source.platform == Platform.MATRIX and allow_cross_room:
            msg_part = f" ({msg_count} message{'s' if msg_count != 1 else ''})" if msg_count else ""
            return t(
                "gateway.resume.matrix_cross_room_success",
                title=title,
                room=source.chat_name or source.chat_id,
                msg_part=msg_part,
            )
        if not msg_count:
            return t("gateway.resume.resumed_no_count", title=title)
        if msg_count == 1:
            return t("gateway.resume.resumed_one", title=title, count=msg_count)
        return t("gateway.resume.resumed_many", title=title, count=msg_count)

    def _resume_listing_reply(self, source, titled: list[dict], allow_all: bool) -> str:
        """Numbered /resume list. A non-admin ``--all`` silently falls back to same-origin scoping;
        say so instead of rendering an unexplained narrower list (sibling of the /sessions notice)."""
        scope_note = (
            t("gateway.resume.all_requires_admin")
            if allow_all and not self._resume_caller_is_admin(source)
            else None
        )
        if not titled:
            if source.platform == Platform.MATRIX and not allow_all:
                return t("gateway.resume.matrix_no_named_sessions")
            base = t("gateway.resume.no_named_sessions")
            return f"{base}\n{scope_note}" if scope_note else base
        lines = [t("gateway.resume.list_header")]
        for idx, s in enumerate(titled[:10], start=1):
            title = s["title"]
            if source.platform == Platform.MATRIX and allow_all:
                origin = self._gateway_session_origin_for_id(str(s.get("id") or ""))
                if origin:
                    title = f"{title} — {origin.chat_name or origin.chat_id}"
            preview = s.get("preview", "")[:40]
            preview_part = t("gateway.resume.list_preview_suffix", preview=preview) if preview else ""
            lines.append(t("gateway.resume.list_item_numbered", index=idx, title=title, preview_part=preview_part))
        if scope_note:
            lines.append(scope_note)
        lines.append(t("gateway.resume.list_footer_numbered"))
        return "\n".join(lines)

    async def _handle_sessions_command(self, event: MessageEvent) -> str:
        """Handle /sessions — list previous sessions for gateway chats."""
        if not self._session_db:
            return self._session_db_unavailable_reply()

        from hermes_cli.session_listing import (
            format_gateway_session_listing,
            parse_session_listing_args,
            query_session_listing,
        )

        try:
            include_all, include_unnamed, target, search_query = (
                parse_session_listing_args(event.get_command_args().strip())
            )
        except ValueError as exc:
            return t("gateway.resume.parse_error", error=exc)
        if search_query == "":
            return "Usage: `/sessions search <query>`"
        if target:
            resume_event = dataclasses.replace(event, text=f"/resume {target}")
            return await self._handle_resume_command(resume_event)

        source = await asyncio.to_thread(self._normalize_source_for_session_key, event.source)
        session_key = self._session_key_for_source(source)
        # A cross-origin listing (`/sessions all`) is honored only for an admin, mirroring
        # `/resume --all`; ungated, any caller could enumerate other origins' session ids/titles/
        # previews — the enumeration half of the IDOR. A non-admin `all` gets explicit feedback
        # rather than a silently scoped list that reads as "my session vanished".
        cross_origin = include_all and self._resume_caller_is_admin(source)
        scope_notice = None
        if include_all and not cross_origin:
            scope_notice = (
                "_Note: `all` (cross-chat listing) requires a configured admin; "
                "showing this chat's sessions only._"
            )
        current_entry = await self.async_session_store.get_or_create_session(source)
        rows = await asyncio.to_thread(
            query_session_listing,
            getattr(self._session_db, "_db", self._session_db),
            source=source.platform.value if source.platform else None,
            session_key=None if cross_origin else session_key,
            current_session_id=current_entry.session_id,
            include_current_session=True,
            include_all_sources=cross_origin,
            include_unnamed=include_unnamed,
            search_query=search_query,
            # Search filters at SQL level, so over-fetch before the visibility cut: origin-invisible
            # matches would otherwise consume the page.
            limit=50 if search_query else 10,
            exclude_sources=["tool"],
        )
        if not cross_origin:
            # Scope to the caller's own origin so other users'/rooms' ids/previews aren't enumerable.
            rows = [row for row in rows if await self._resume_row_visible(source, row, allow_all=False)]
        rows = rows[:10]
        if search_query:
            title = f"Sessions matching “{search_query}”"
        else:
            title = "Sessions" if include_unnamed else "Named Sessions"
        return format_gateway_session_listing(
            rows, include_source=cross_origin, title=title, notice=scope_notice,
        )

    # ----------------------------------------------------------------------- /branch

    async def _handle_branch_command(self, event: MessageEvent) -> str:
        """Handle /branch [name] — fork the current session into a new independent copy so the
        user can explore a different approach without losing the original.
        """
        import json as _json
        import uuid as _uuid
        from datetime import datetime as _dt

        if not self._session_db:
            return self._session_db_unavailable_reply()

        source = event.source
        session_key = self._session_key_for_source(source)
        current_entry = await self.async_session_store.get_or_create_session(source)
        history = await self.async_session_store.load_transcript(current_entry.session_id)
        if not history:
            return t("gateway.branch.no_conversation")

        new_session_id = f"{_dt.now().strftime('%Y%m%d_%H%M%S')}_{_uuid.uuid4().hex[:6]}"
        branch_title = event.get_command_args().strip()
        if not branch_title:
            current_title = await self._session_db.get_session_title(current_entry.session_id)
            branch_title = await self._session_db.get_next_title_in_lineage(current_title or "branch")
        parent_session_id = current_entry.session_id

        # Serialize the parent's full origin (same shape as the reset path's db_create_kwargs in
        # gateway/session.py) so the branch row carries complete identity from birth. Prefer the
        # live entry's origin (it may hold richer metadata than the triggering event's source).
        _branch_origin = current_entry.origin or source
        _branch_origin_json = None
        if _branch_origin is not None:
            with contextlib.suppress(Exception):
                _branch_origin_json = _json.dumps(_branch_origin.to_dict())

        # Persist a stable ``_branched_from`` marker in model_config so list_sessions_rich() keeps
        # the branch visible in /resume and /sessions even after the parent is reopened and
        # re-ended with a different end_reason. Forward ALL gateway routing columns at CREATE time:
        # otherwise they're NULL until switch_session() records the peer, and a crash in between
        # leaves the branch unroutable (by chat/thread lookup and by /resume's IDOR guard).
        try:
            await self._session_db.create_session(
                session_id=new_session_id,
                source=source.platform.value if source.platform else "gateway",
                model=(self.config.get("model", {}) or {}).get("default") if isinstance(self.config, dict) else None,
                model_config={"_branched_from": parent_session_id},
                parent_session_id=parent_session_id,
                user_id=source.user_id,
                session_key=session_key,
                chat_id=source.chat_id,
                chat_type=source.chat_type,
                thread_id=source.thread_id,
                origin_json=_branch_origin_json,
                display_name=current_entry.display_name,
            )
        except Exception as e:
            logger.error("Failed to create branch session: %s", e)
            return t("gateway.branch.create_failed", error=e)

        # Bounded-chunk transactions (a history can be hundreds of rows). Best-effort: a failed
        # copy still yields a usable (partial) branch.
        with contextlib.suppress(Exception):
            await self._session_db.append_messages_batch(
                new_session_id, [_branch_row(msg) for msg in history], chunk_rows=500,
            )
        with contextlib.suppress(Exception):
            await self._session_db.set_session_title(new_session_id, branch_title)

        new_entry = await self.async_session_store.switch_session(session_key, new_session_id)
        if not new_entry:
            return t("gateway.branch.switch_failed")
        self._clear_session_boundary_security_state(session_key)
        self._evict_cached_agent(session_key)

        msg_count = len([m for m in history if m.get("role") == "user"])
        key = "gateway.branch.branched_one" if msg_count == 1 else "gateway.branch.branched_many"
        return t(key, title=branch_title, count=msg_count, parent=parent_session_id, new=new_session_id)
