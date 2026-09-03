"""Inbound message pipeline (_handle_message, text/media preparation, durable-turn markers, plugin injection) for GatewayRunner.

Split out of ``gateway/run.py``; bound onto ``GatewayRunner`` via the MRO.
``gateway.run`` internals are imported lazily inside method bodies (import cycle),
so ``patch("gateway.run.X")`` keeps intercepting them at call time.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
import asyncio
import concurrent.futures
import dataclasses
import json
import os
import re
import time
from contextlib import suppress
from gateway.config import Platform
from gateway.platforms.base import EphemeralReply, MessageEvent, MessageType
from gateway.run_common import _UNSET
from gateway.session import (
    SessionSource,
    is_shared_multi_user_session,
    neutralize_untrusted_inline_text,
)
from gateway.turn_lease import TurnLeaseTimeoutError
from typing import Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:  # string annotations only; never imported at runtime (cycle)
    from gateway.run import GatewayRunner, TurnRunner  # noqa: F401

# Log-record parity with the origin module.
logger = logging.getLogger("gateway.run")


class GatewayInboundMixin:
    """Inbound message pipeline (_handle_message, text/media preparation, durable-turn markers, plugin injection) for GatewayRunner."""

    async def _hm_admit_event(
        self, event: "MessageEvent"
    ) -> Optional[Tuple["MessageEvent", SessionSource, bool]]:
        """Ingress gates for ``_handle_message``: leak guard, profile route, ignored channels,
        startup-restore queueing, ``pre_gateway_dispatch`` hook, authorization/pairing.

        Returns ``None`` when the message is dropped, else ``(event, source, is_internal)`` —
        the hook may have rewritten ``event``.
        """
        from gateway.run import _is_slack_ignored_channel
        source = event.source

        # 🔴 Cross-session leak guard. This per-message task was created via create_task(), which
        # copies the spawning context: if a concurrent message had already bound its session via
        # set_session_vars(), we inherited ITS HERMES_SESSION_* ContextVars, and until _set_session_env
        # binds ours any subprocess would read the foreign identity (the _UNSET-strip guard can't
        # help — the vars are set-to-foreign). Reset to _UNSET so that window strips safe instead.
        try:
            from gateway.session_context import reset_session_vars
            reset_session_vars()
        except Exception:
            logger.debug("reset_session_vars failed at handler entry", exc_info=True)

        # Most adapters resolve profile routes in build_source(), before they hand us the event. A
        # few internal/voice paths construct SessionSource directly, so resolve those here as the
        # shared fail-closed ingress gate before authorization, hooks, or session side effects.
        if (
            getattr(getattr(self, "config", None), "multiplex_profiles", False)
            and not getattr(source, "profile", None)
            and getattr(source, "profile_route_rejected", False) is not True
        ):
            from gateway.profile_routing import ProfileRouteRejected

            try:
                source.profile = self._profile_name_for_source(source)
            except ProfileRouteRejected:
                source.profile_route_rejected = True

        # SessionSource owns a strict boolean marker. Require the literal value
        # so duck-typed test/internal sources with dynamic attributes are not
        # mistaken for an explicit matched-route rejection.
        if getattr(source, "profile_route_rejected", False) is True:
            logger.warning(
                "Dropping inbound message because its explicit profile route "
                "targets an unserved profile"
            )
            return None

        # Internal events (e.g. background-process completion notifications)
        # are system-generated and must skip user authorization.
        is_internal = bool(getattr(event, "internal", False))

        # Ignored-channel guard runs FIRST — before startup-restore queueing, plugin hooks, auth,
        # and session setup — so an ignored channel can never reach pairing/auth/session state.
        # getattr: bare test runners construct GatewayRunner via object.__new__ without config.
        if (
            not is_internal
            and getattr(source, "platform", None) == Platform.SLACK
            and _is_slack_ignored_channel(
                getattr(self, "config", None), getattr(source, "chat_id", None)
            )
        ):
            logger.info(
                "Dropping Slack message from configured ignored channel %s",
                getattr(source, "chat_id", None),
            )
            return None

        if (
            getattr(self, "_startup_restore_in_progress", False)
            and not is_internal
            and not getattr(event, "_hermes_startup_restore_replay", False)
        ):
            self._queue_startup_restore_event(event)
            return None

        # scale-to-zero: stamp the gateway-scoped last-inbound clock (read by is_idle) for real
        # user-originated inbound only. Internal/system events are NOT traffic — counting them
        # would keep a genuinely idle gateway awake.
        if not is_internal:
            self._scale_to_zero_note_real_inbound()

        # pre_gateway_dispatch plugin hook (user-originated only). Plugins may return
        #   {"action": "skip", "reason": ...} -> drop; {"action": "rewrite", "text": ...} -> replace
        #   event.text; {"action": "allow"} / None -> normal dispatch.
        # Runs BEFORE auth so plugins can handle unauthorized senders without the pairing flow.
        if not is_internal:
            try:
                from hermes_cli.lifecycle import invoke_hook as _invoke_hook
                _hook_results = _invoke_hook(
                    "pre_gateway_dispatch",
                    event=event,
                    gateway=self,
                    # getattr: bare-runner tests build GatewayRunner via object.__new__ without
                    # __init__; the hook must not fail dispatch over a missing attribute.
                    session_store=getattr(self, "session_store", None),
                )
            except Exception as _hook_exc:
                logger.warning("pre_gateway_dispatch invocation failed: %s", _hook_exc)
                _hook_results = []

            for _result in _hook_results:
                if not isinstance(_result, dict):
                    continue
                _action = _result.get("action")
                if _action == "skip":
                    logger.info(
                        "pre_gateway_dispatch skip: reason=%s platform=%s chat=%s",
                        _result.get("reason"),
                        source.platform.value if source.platform else "unknown",
                        source.chat_id or "unknown",
                    )
                    return None
                if _action == "rewrite":
                    _new_text = _result.get("text")
                    if isinstance(_new_text, str):
                        event = dataclasses.replace(event, text=_new_text)
                        source = event.source
                    break
                if _action == "allow":
                    break

        if is_internal:
            pass
        elif source.user_id is None:
            # Messages with no user identity (Telegram service messages, channel forwards, anonymous
            # admin posts, sender_chat) can't be paired but may be authorized via a chat-scoped
            # allowlist (e.g. TELEGRAM_GROUP_ALLOWED_CHATS), so defer to _is_user_authorized.
            if not self._is_user_authorized_for_source(source):
                logger.debug("Ignoring message with no user_id from %s", source.platform.value)
                return None
        elif not self._is_user_authorized_for_source(source):
            logger.warning("Unauthorized user: %s (%s) on %s", source.user_id, source.user_name, source.platform.value)
            # In DMs: offer pairing code. In groups: silently ignore.
            if (
                source.chat_type == "dm"
                and self._get_unauthorized_dm_behavior(
                    source.platform,
                    profile=source.profile,
                )
                == "pair"
            ):
                platform_name = source.platform.value if source.platform else "unknown"
                pairing_store = self._pairing_store_for(source)
                if pairing_store is None:
                    logger.error(
                        "Cannot offer pairing code on %s: no pairing store",
                        platform_name,
                    )
                    return None
                # Rate-limit ALL pairing responses (code or rejection) so a burst of DMs doesn't
                # spam the user with repeated messages.
                if pairing_store._is_rate_limited(platform_name, source.user_id):
                    return None
                code = pairing_store.generate_code(
                    platform_name, source.user_id, source.user_name or ""
                )
                if code:
                    adapter = self._adapter_for_source(source)
                    if adapter:
                        store_profile = getattr(pairing_store, "profile", None)
                        profile_arg = (
                            f"-p {store_profile} "
                            if isinstance(store_profile, str)
                            and store_profile
                            and store_profile != "default"
                            else ""
                        )
                        await adapter.send(
                            source.chat_id,
                            f"Hi~ I don't recognize you yet!\n\n"
                            f"Here's your pairing code: `{code}`\n\n"
                            f"Ask the bot owner to run:\n"
                            f"`hermes {profile_arg}pairing approve "
                            f"{platform_name} {code}`"
                        )
                else:
                    adapter = self._adapter_for_source(source)
                    if adapter:
                        await adapter.send(
                            source.chat_id,
                            "Too many pairing requests right now~ "
                            "Please try again later!"
                        )
                    # Record rate limit so subsequent messages are silently ignored
                    pairing_store._record_rate_limit(platform_name, source.user_id)
            return None

        return event, source, is_internal

    def _hm_estop_gate(
        self, event: "MessageEvent", source: SessionSource, is_internal: bool
    ) -> Optional[str]:
        """Return the global emergency-stop notice when this turn must be blocked, else None."""
        # Global emergency stop (`hermes pause`): new turns get a brief paused notice instead of an
        # agent run. Placed after auth so unauthorized senders can't probe pause state. Pause blocks
        # NEW agent turns, never running work or control traffic, so these pass through: internal
        # events from IN-FLIGHT work; recognized slash commands (/status, /approve, ... and /pause off
        # as the in-band resume path); replies owned by in-flight work — pending update prompt,
        # clarify, slash-confirm, dangerous-command approval, or steering an already-running session.
        if not is_internal:
            try:
                from agent.estop import paused_reply as _estop_paused_reply
                _paused_notice = _estop_paused_reply()
            except ImportError:
                _paused_notice = None
            if _paused_notice is not None:
                _estop_allow = False
                _estop_cmd = None
                try:
                    _estop_cmd = event.get_command()
                except Exception:
                    _estop_cmd = None
                if _estop_cmd:
                    try:
                        from hermes_cli.commands import (
                            resolve_command as _resolve_estop_cmd,
                        )
                        _estop_allow = _resolve_estop_cmd(_estop_cmd) is not None
                    except Exception:
                        _estop_allow = False
                if not _estop_allow:
                    try:
                        _estop_key = self._session_key_for_source(source)
                        _estop_state = self._peek_session_state(_estop_key)
                        if (
                            _estop_state is not None
                            and _estop_state.persistent.update_prompt_pending
                        ):
                            _estop_allow = True
                        if not _estop_allow and self._is_session_running(_estop_key):
                            # Steering / interrupting in-flight work (also covers pending clarify +
                            # tool approvals held by the running agent).
                            _estop_allow = True
                        if not _estop_allow:
                            from tools import slash_confirm as _estop_confirm_mod
                            if _estop_confirm_mod.get_pending(_estop_key):
                                _estop_allow = True
                        if not _estop_allow:
                            from tools.approval import (
                                has_blocking_approval as _estop_has_approval,
                            )
                            if _estop_has_approval(_estop_key):
                                _estop_allow = True
                    except Exception:
                        pass
                if not _estop_allow:
                    logger.info(
                        "Gateway turn paused by global emergency stop (platform=%s chat=%s)",
                        getattr(getattr(source, "platform", None), "value", "unknown"),
                        getattr(source, "chat_id", None) or "unknown",
                    )
                    return _paused_notice
        return None

    def _hm_update_prompt_reply(
        self, event: "MessageEvent", _quick_key: str, allow_gateway_control: bool
    ) -> Optional[str]:
        """Consume a reply to a pending ``/update`` prompt; None when nothing was consumed."""
        from gateway.run import _hermes_home
        # Route replies to a pending /update prompt back to the detached update process via
        # .update_response. Recognized slash commands must bypass this or /new, /help etc. get
        # silently consumed as update answers.
        _up_state = self._peek_session_state(_quick_key)
        if (
            allow_gateway_control
            and _up_state is not None
            and _up_state.persistent.update_prompt_pending
        ):
            raw = (event.text or "").strip()
            # Accept /approve and /deny as shorthand for yes/no
            cmd = event.get_command()
            if cmd in {"approve", "yes"}:
                response_text = "y"
            elif cmd in {"deny", "no"}:
                response_text = "n"
            else:
                _recognized_cmd = None
                if cmd:
                    try:
                        from hermes_cli.commands import resolve_command as _resolve_update_cmd
                    except Exception:
                        _resolve_update_cmd = None
                    if _resolve_update_cmd is not None:
                        try:
                            _cmd_def = _resolve_update_cmd(cmd)
                            _recognized_cmd = _cmd_def.name if _cmd_def else None
                        except Exception:
                            _recognized_cmd = None
                response_text = "" if _recognized_cmd else raw
            if response_text:
                response_path = _hermes_home / ".update_response"
                prompt_path = _hermes_home / ".update_prompt.json"
                try:
                    tmp = response_path.with_suffix(".tmp")
                    tmp.write_text(response_text, encoding="utf-8")
                    tmp.replace(response_path)
                    prompt_path.unlink(missing_ok=True)
                except OSError as e:
                    logger.warning("Failed to write update response: %s", e)
                    return f"✗ Failed to send response to update process: {e}"
                _up_state.persistent.update_prompt_pending = False
                label = response_text if len(response_text) <= 20 else response_text[:20] + "…"
                return f"✓ Sent `{label}` to the update process."
            # Recognized slash command during a pending update prompt: write a blank response so the
            # detached update's ``_gateway_prompt`` returns the prompt's default (typically a safe
            # "n" / skip) and exits instead of blocking on stdin until the watcher timeout.
            if _recognized_cmd:
                response_path = _hermes_home / ".update_response"
                prompt_path = _hermes_home / ".update_prompt.json"
                try:
                    tmp = response_path.with_suffix(".tmp")
                    tmp.write_text("", encoding="utf-8")
                    tmp.replace(response_path)
                    prompt_path.unlink(missing_ok=True)
                    logger.info(
                        "Recognized /%s during pending update prompt for %s; "
                        "cancelled prompt with default and dispatching command",
                        _recognized_cmd,
                        _quick_key,
                    )
                except OSError as e:
                    logger.warning(
                        "Failed to write cancel response for pending update prompt: %s",
                        e,
                    )
                _up_state.persistent.update_prompt_pending = False
        return None

    async def _hm_clarify_reply(
        self,
        event: "MessageEvent",
        source: SessionSource,
        _quick_key: str,
        allow_gateway_control: bool,
    ) -> Optional[str]:
        """Intercept a reply to a pending clarify prompt; None when the message falls through."""
        # Intercept replies to a pending clarify: open-ended prompts and "Other" responses are free
        # text; direct replies to multi-choice prompts are accepted too ("2" → second option).
        _clarify_mod = None
        try:
            from tools import clarify_gateway as _clarify_mod
            _pending_clarify = _clarify_mod.get_pending_for_session(
                _quick_key, include_choice_prompts=True,
            )
        except Exception:
            _pending_clarify = None
        if (
            allow_gateway_control
            and _pending_clarify is not None
            and _clarify_mod is not None
        ):
            _clarify_has_audio = bool(self._pending_event_audio_paths(event))
            _raw_clarify_reply = await self._prepare_clarify_reply_text(event)
            if _clarify_has_audio and not _raw_clarify_reply:
                logger.info(
                    "Gateway retained pending clarify after voice transcription "
                    "produced no usable text (session=%s, id=%s)",
                    _quick_key,
                    _pending_clarify.clarify_id,
                )
                return ""
            # Skip slash commands — the user wanted a command, not to answer the clarify. Leave it
            # pending so they can retry; on timeout the agent unblocks with an empty response.
            if _raw_clarify_reply and not _raw_clarify_reply.startswith("/"):
                _text_outcome = _clarify_mod.attempt_text_response_for_session(
                    _quick_key, _raw_clarify_reply,
                )
                if _text_outcome == _clarify_mod.TEXT_RESOLVED:
                    logger.info(
                        "Gateway intercepted clarify text response (session=%s, id=%s)",
                        _quick_key, _pending_clarify.clarify_id,
                    )
                    # The clarify callback pauses the platform typing/status indicator while waiting
                    # so Slack users can type their answer. The active agent resumes as soon as this
                    # reply resolves the wait, so re-enable its indicator here too.
                    _clarify_adapter = self._adapter_for_source(source)
                    if _clarify_adapter:
                        try:
                            _clarify_adapter.resume_typing_for_chat(source.chat_id)
                        except Exception:
                            logger.debug(
                                "Failed to resume typing after clarify response",
                                exc_info=True,
                            )
                    # Acknowledge with empty string so adapters that emit the agent's response don't
                    # double-post; the agent itself produces the next user-facing message.
                    return ""
                if _text_outcome == _clarify_mod.TEXT_REJECTED_SELECTION:
                    # Selection-shaped but invalid (out-of-range number, bad comma-list): keep the
                    # clarify armed for retry — don't cancel, don't treat as an unrelated follow-up.
                    logger.info(
                        "Gateway retained pending clarify after invalid "
                        "selection attempt (session=%s, id=%s)",
                        _quick_key, _pending_clarify.clarify_id,
                    )
                    return ""
                if _text_outcome == _clarify_mod.TEXT_REJECTED_PROSE:
                    # Native-choice prompts deliberately reject unmatched prose so it can continue
                    # through normal busy-message routing. Release this clarify first: redirect()
                    # degrades to steer() while tools execute, and that steer cannot drain until
                    # the clarify tool returns.
                    _clarify_mod.resolve_gateway_clarify(
                        _pending_clarify.clarify_id,
                        "",
                    )
        return None

    async def _hm_slash_confirm_reply(
        self, event: "MessageEvent", _quick_key: str, allow_gateway_control: bool
    ) -> Optional[str]:
        """Resolve a reply to a pending slash-confirm prompt; None when the message falls through."""
        # Replies to a pending slash-confirm prompt (/reload-mcp etc.): /approve, /always, /cancel and
        # short aliases. Anything else falls through — a stale pending confirm does NOT block other
        # commands. A pending dangerous-command approval takes precedence: /approve there unblocks
        # the waiting tool thread; slash-confirm only catches it when no tool approval is live.
        from tools import slash_confirm as _slash_confirm_mod
        _pending_confirm = _slash_confirm_mod.get_pending(_quick_key)
        _tool_approval_live = False
        try:
            from tools.approval import has_blocking_approval
            _tool_approval_live = has_blocking_approval(_quick_key)
        except Exception:
            _tool_approval_live = False
        if allow_gateway_control and _pending_confirm and not _tool_approval_live:
            _raw_reply = (event.text or "").strip()
            # Accept bang-prefixed replies (`!always`, `!cancel`) verbatim: Slack/Matrix show the
            # `!` prefix (typed `/` is blocked in Slack threads) and adapters only rewrite
            # `!<known-command>` — confirm keywords aren't commands, so the `!` survives to here.
            _norm_reply = _raw_reply.lstrip("!/").lower()
            _cmd_reply = event.get_command()
            _confirm_choice = None
            if _cmd_reply in {"approve", "yes", "ok", "confirm"}:
                _confirm_choice = "once"
            elif _cmd_reply in {"always", "remember"}:
                _confirm_choice = "always"
            elif _cmd_reply in {"cancel", "no", "deny", "nevermind"}:
                _confirm_choice = "cancel"
            elif _norm_reply in {"approve", "approve once", "once"}:
                _confirm_choice = "once"
            elif _norm_reply in {"always", "always approve"}:
                _confirm_choice = "always"
            elif _norm_reply in {"cancel", "nevermind", "no"}:
                _confirm_choice = "cancel"
            if _confirm_choice is not None:
                _resolved = await _slash_confirm_mod.resolve(
                    _quick_key, _pending_confirm.get("confirm_id"), _confirm_choice,
                )
                return _resolved or ""
            # Stale pending + unrelated command: the user moved on, so drop the pending state rather
            # than let the confirm block normal usage indefinitely.
            _slash_confirm_mod.clear_if_stale(_quick_key)
        return None

    def _hm_evict_stale_running_agent(self, _quick_key: str) -> None:
        """Evict a leaked/reaped ``_running_agents`` slot before the busy-session fast-path."""
        from gateway.run import _AGENT_PENDING_SENTINEL, _float_env
        # Staleness eviction: detect leaked locks from hung/crashed handlers. With inactivity-based
        # timeout active tasks can run for hours, so evict only when the agent has been *idle* past
        # the threshold (or has no activity tracker and its wall-clock age is extreme).
        _raw_stale_timeout = _float_env("HERMES_AGENT_TIMEOUT", 1800)
        _quick_state = self._peek_session_state(_quick_key)
        _stale_ts = _quick_state.turn.started_ts if _quick_state else 0
        if _quick_state is not None and _quick_state.turn.agent is not None and _stale_ts:
            _stale_age = time.time() - _stale_ts
            _stale_agent = _quick_state.turn.agent
            # Never evict the pending sentinel — it was just placed during async setup before the
            # real agent exists. Sentinels have no get_activity_summary(), so the idle check would
            # read inf >= timeout and evict them immediately, racing the setup path.
            _stale_idle = float("inf")  # assume idle if we can't check
            _stale_detail = ""
            _activity_summary_valid = False
            if _stale_agent and hasattr(_stale_agent, "get_activity_summary"):
                try:
                    _sa = _stale_agent.get_activity_summary()
                    from gateway.session_stall import (
                        resolve_session_idle_seconds_from_activity,
                    )

                    _resolved_idle = resolve_session_idle_seconds_from_activity(
                        _sa if isinstance(_sa, dict) else None,
                        now=time.time(),
                    )
                    if _resolved_idle is not None:
                        _stale_idle = _resolved_idle
                        _activity_summary_valid = True
                    _stale_detail = (
                        f" | last_activity={_sa.get('last_activity_desc', 'unknown') if isinstance(_sa, dict) else 'unknown'} "
                        f"({_stale_idle:.0f}s ago) "
                        f"| iteration={_sa.get('api_call_count', 0) if isinstance(_sa, dict) else 0}/{_sa.get('max_iterations', 0) if isinstance(_sa, dict) else 0}"
                    )
                except Exception:
                    pass
            # A valid activity clock is authoritative: total age alone never
            # makes an actively progressing turn stale. The emergency wall TTL
            # is only a fallback when the agent cannot report usable activity.
            _wall_ttl = max(_raw_stale_timeout * 10, 7200) if _raw_stale_timeout > 0 else float("inf")
            _should_evict = (
                _stale_agent is not _AGENT_PENDING_SENTINEL
                and (
                    (
                        _activity_summary_valid
                        and _raw_stale_timeout > 0
                        and _stale_idle >= _raw_stale_timeout
                    )
                    or (
                        not _activity_summary_valid
                        and _stale_age > _wall_ttl
                    )
                )
            )
            if _should_evict:
                logger.warning(
                    "Evicting stale _running_agents entry for %s "
                    "(age: %.0fs, idle: %.0fs, timeout: %.0fs)%s",
                    _quick_key, _stale_age, _stale_idle,
                    _raw_stale_timeout, _stale_detail,
                )
                self._invalidate_session_run_generation(
                    _quick_key,
                    reason="stale_running_agent_eviction",
                )
                self._release_running_agent_state(_quick_key)

        # Durable-reaped guard. A session whose routing row was ended in state.db (``ws_orphan_reap``
        # / ``agent_close``) while the gateway lived keeps its in-memory turn slot, so the fast-path
        # would queue every next message into the dead runtime. Evict the stale slot so the cold
        # path re-attaches via ``get_or_create_session`` → ``reopen`` or creates a fresh session.
        if self._is_session_running(_quick_key):
            try:
                _reap_store = getattr(self, "session_store", None)
                # Use the public, lock-held accessors: peek_session_id resolves key -> session_id
                # under the store lock, and returns a non-str on stubbed stores in bare test runners
                # — both the isinstance() gate and the ``is True`` gate below keep this guard inert
                # unless a real SessionStore answers.
                _reap_peek = getattr(_reap_store, "peek_session_id", None)
                _is_ended = getattr(_reap_store, "_is_session_ended_in_db", None)
                _reap_sid = _reap_peek(_quick_key) if callable(_reap_peek) else None
                if (
                    isinstance(_reap_sid, str)
                    and _reap_sid
                    and callable(_is_ended)
                    and _is_ended(_reap_sid) is True
                ):
                    logger.warning(
                        "Evicting stale _running_agents entry for %s — "
                        "durable session %s is ended (reaped) in state.db; "
                        "healing routing on next message (#99106)",
                        _quick_key,
                        _reap_sid,
                    )
                    self._invalidate_session_run_generation(
                        _quick_key,
                        reason="reaped_session_eviction",
                    )
                    self._release_running_agent_state(_quick_key)
            except Exception:
                logger.debug("reaped-session staleness check failed", exc_info=True)

    async def _hm_handle_running_session_message(
        self, event: "MessageEvent", source: SessionSource, _quick_key: str
    ) -> Optional[str]:
        """Fast-path for a message that arrives while this session's agent is running."""
        from gateway.run import (
            _AGENT_PENDING_SENTINEL,
            _build_media_placeholder,
            merge_pending_message_event,
        )
        # PRIORITY handling when an agent is already running for this session. Default behavior is
        # to interrupt immediately so user text/stop messages are handled with minimal latency.
        # Exception: Telegram photo bursts arrive as near-simultaneous updates — do NOT interrupt
        # for photo-only follow-ups; adapter-level batching absorbs them.
        # Resolve the command once; each command's mid-run behavior is declared on its
        # CommandDef (busy_policy / busy_handler in hermes_cli/commands.py) and dispatched via
        # _dispatch_busy_slash_command below — no per-command if-chain here.
        from hermes_cli.commands import resolve_command as _resolve_cmd_inner
        _evt_cmd = event.get_command()
        _cmd_def_inner = _resolve_cmd_inner(_evt_cmd) if _evt_cmd else None

        # /status and /context are intentionally pre-gate so users
        # always see session state.
        if _cmd_def_inner and _cmd_def_inner.name == "status":
            return await self._handle_status_command(event)
        if _cmd_def_inner and _cmd_def_inner.name == "context":
            return await self._handle_context_command(event)

        # Slash command access control on the running-agent fast-path. Mirrors the cold-path
        # gate below so non-admins can't bypass gating just because an agent is busy. /status
        # above is intentionally pre-gate; /help and /whoami are the always-allowed floor.
        if _evt_cmd and _cmd_def_inner is not None:
            _denied = self._check_slash_access(source, _cmd_def_inner.name)
            if _denied is not None:
                return _denied

        # Any recognized slash command: dispatch according to its declared busy_policy (dispatch
        # / interrupt_then_dispatch / reject). Unrecognized commands and plain text fall through
        # to the interrupt/queue logic below.
        if _cmd_def_inner:
            return await self._dispatch_busy_slash_command(
                event, _cmd_def_inner, _quick_key, source,
            )

        if event.message_type == MessageType.PHOTO:
            logger.debug("PRIORITY photo follow-up for session %s — queueing without interrupt", _quick_key)
            adapter = self._adapter_for_source(source)
            if adapter:
                merge_pending_message_event(adapter._pending_messages, _quick_key, event)
            return None

        effective_busy_input_mode = self._effective_busy_input_mode(source)
        _telegram_followup_grace = float(
            os.getenv("HERMES_TELEGRAM_FOLLOWUP_GRACE_SECONDS", "3.0")
        )
        _grace_state = self._peek_session_state(_quick_key)
        _started_at = _grace_state.turn.started_ts if _grace_state else 0
        if (
            source.platform == Platform.TELEGRAM
            and event.message_type == MessageType.TEXT
            and _telegram_followup_grace > 0
            and _started_at
            and (time.time() - _started_at) <= _telegram_followup_grace
        ):
            logger.debug(
                "Telegram follow-up arrived %.2fs after run start for %s — queueing without interrupt",
                time.time() - _started_at,
                _quick_key,
            )
            adapter = self._adapter_for_source(source)
            if adapter:
                if effective_busy_input_mode == "queue":
                    self._enqueue_fifo(_quick_key, event, adapter)
                else:
                    merge_pending_message_event(
                        adapter._pending_messages,
                        _quick_key,
                        event,
                        merge_text=True,
                    )
            return None

        _ra_state = self._peek_session_state(_quick_key)
        running_agent = _ra_state.turn.agent if _ra_state else None
        if running_agent is _AGENT_PENDING_SENTINEL:
            # Agent is being set up but not ready yet.
            if event.get_command() == "stop":
                # Force-clean the sentinel so the session is unlocked.
                self._release_running_agent_state(_quick_key)
                logger.info("HARD STOP (pending) for session %s — sentinel cleared", _quick_key)
                return EphemeralReply("⚡ Force-stopped. The agent was still starting — session unlocked.")
            # Queue the message so it will be picked up after the
            # agent starts.
            adapter = self._adapter_for_source(source)
            if adapter:
                merge_pending_message_event(
                    adapter._pending_messages,
                    _quick_key,
                    event,
                    merge_text=True,
                )
            return None
        if self._draining:
            queue_during_drain = self._queue_during_drain_enabled(
                effective_busy_input_mode
            )
            if queue_during_drain:
                self._queue_or_replace_pending_event(_quick_key, event)
            return (
                f"⏳ Gateway {self._status_action_gerund()} — queued for the next turn after it comes back."
                if queue_during_drain
                else f"⏳ Gateway is {self._status_action_gerund()} and is not accepting another turn right now."
            )
        if effective_busy_input_mode == "queue":
            logger.debug("PRIORITY queue follow-up for session %s", _quick_key)
            self._queue_or_replace_pending_event(_quick_key, event)
            return None
        if effective_busy_input_mode == "steer":
            # Steer mode: inject text into the running agent mid-run via
            # agent.steer().  Falls back to queue semantics if the payload
            # is empty, the agent lacks steer(), or steer() rejects.
            steer_text = (event.text or "").strip()
            steered = False
            if (
                event.message_type == MessageType.TEXT
                and not event.media_urls
                and not event.media_types
                and steer_text
                and hasattr(running_agent, "steer")
            ):
                try:
                    steered = bool(running_agent.steer(steer_text))
                except Exception as exc:
                    logger.warning("PRIORITY steer failed for session %s: %s", _quick_key, exc)
                    steered = False
            if steered:
                logger.debug("PRIORITY steer for session %s", _quick_key)
                return None
            logger.debug("PRIORITY steer-fallback-to-queue for session %s", _quick_key)
            self._queue_or_replace_pending_event(_quick_key, event)
            return None
        # Subagent protection (PRIORITY path). Same rationale as
        # ``_handle_active_session_busy_message``: an interrupt cascades through
        # ``_active_children`` and aborts in-flight delegate_task work, so demote to queue
        # semantics while subagents run. /stop reached its handler above — still an escape hatch.
        if self._agent_has_active_subagents(running_agent):
            logger.info(
                "PRIORITY interrupt demoted to queue for session %s "
                "because the running agent has active subagents (#30170)",
                _quick_key,
            )
            self._queue_or_replace_pending_event(_quick_key, event)
            return None
        # Compression protection (PRIORITY path), as in ``_handle_active_session_busy_message``:
        # an interrupt would start a new turn on the pre-rotation parent while compression
        # rotates the id away, forking orphaned siblings. Demote to queue until rotation lands.
        if await self._session_has_compression_in_flight(_quick_key):
            logger.info(
                "PRIORITY interrupt demoted to queue for session %s "
                "because context compression is in flight (#56391)",
                _quick_key,
            )
            self._queue_or_replace_pending_event(_quick_key, event)
            return None
        # Text-only corrections redirect the live turn (preserving displayed context) when the
        # runtime supports it; media/voice and older runtimes use the interrupt path below.
        if (
            event.message_type == MessageType.TEXT
            and not event.media_urls
            and not event.media_types
            and getattr(running_agent, "_supports_active_turn_redirect", False)
            is True
            and hasattr(running_agent, "redirect")
        ):
            try:
                if running_agent.redirect((event.text or "").strip()):
                    logger.debug("PRIORITY redirect for session %s", _quick_key)
                    return None
            except Exception as exc:
                logger.warning(
                    "PRIORITY redirect failed for session %s: %s",
                    _quick_key,
                    exc,
                )
        logger.debug("PRIORITY interrupt for session %s", _quick_key)
        _interrupt_text = event.text
        _media_urls = getattr(event, "media_urls", None) or []
        if self._pending_event_audio_paths(event):
            _interrupt_text, _ = await self._transcribe_and_echo_pending_voice(
                event,
                self._adapter_for_source(source),
                source,
                event.text or "",
                log_context="Voice-priority-interrupt",
            )
        elif not _interrupt_text and _media_urls:
            _interrupt_text = _build_media_placeholder(event)
        running_agent.interrupt(_interrupt_text)
        # The interrupt message is delivered via adapter._pending_messages (read by _run_agent);
        # don't also buffer it on self — that copy was never consumed and grew unbounded.
        return None

    async def _hm_resolve_command(
        self, event: "MessageEvent", source: SessionSource, _quick_key: str
    ) -> Tuple[bool, Optional[str], Optional[str], Optional[str]]:
        """Resolve the slash command (aliases, access gate, ``pre_command`` + ``command:<name>`` hooks).

        Returns ``(handled, result, command, canonical)``; when ``handled`` the caller returns
        ``result`` as-is (it may legitimately be None).
        """
        # Check for commands
        command = event.get_command()

        from hermes_cli.commands import (
            is_gateway_known_command,
            resolve_command as _resolve_cmd,
        )

        # Resolve aliases to canonical name so dispatch and hook names
        # don't depend on the exact alias the user typed.
        _cmd_def = _resolve_cmd(command) if command else None
        canonical = _cmd_def.name if _cmd_def else command

        # Expand alias quick commands before built-in dispatch so targets like /model openai/gpt-5.5
        # --provider openrouter reach the /model handler. Preserve built-in precedence; aliases only
        # need early handling when the typed command is not already known.
        if command and _cmd_def is None:
            if isinstance(self.config, dict):
                quick_commands = self.config.get("quick_commands", {}) or {}
            else:
                quick_commands = getattr(self.config, "quick_commands", {}) or {}
            if isinstance(quick_commands, dict) and command in quick_commands:
                qcmd = quick_commands[command]
                if qcmd.get("type") == "alias":
                    target = (qcmd.get("target") or "").strip()
                    if target:
                        target = target if target.startswith("/") else f"/{target}"
                        target_command = target.lstrip("/")
                        user_args = event.get_command_args().strip()
                        event.text = f"{target} {user_args}".strip()
                        command = target_command.split()[0] if target_command else target_command
                        _cmd_def = _resolve_cmd(command) if command else None
                        canonical = _cmd_def.name if _cmd_def else command

        # Per-platform slash command access control. Only kicks in when the operator has set
        # ``allow_admin_from`` for the source's scope (DM vs group). When unset → backward-compat:
        # every allowed user can run every command. When set → non-admins get only
        # ``user_allowed_commands`` plus the /help, /whoami floor. Plain chat is never gated.
        if command and canonical and is_gateway_known_command(canonical):
            _denied = self._check_slash_access(source, canonical)
            if _denied is not None:
                return True, _denied, command, canonical

        # pre_command observer hook (returns ignored) fires for every recognized slash command
        # BEFORE core handling, mirroring cli.py. The running-agent intercept path above (/stop,
        # /approve, busy_policy) deliberately does NOT fire it — a slow or hostile plugin must not
        # interfere with the operator's escape hatches for a live agent.
        if command and is_gateway_known_command(canonical):
            try:
                from hermes_cli.plugins import fire_pre_command_hook
                fire_pre_command_hook(
                    surface="gateway",
                    command=str(canonical),
                    alias_used=str(command),
                    args_raw=event.get_command_args().strip(),
                    session_key=_quick_key,
                    platform=source.platform.value if source.platform else "",
                )
            except Exception as _pre_cmd_err:
                logger.debug(
                    "pre_command hook dispatch failed (non-fatal): %s",
                    _pre_cmd_err,
                )

        # Fire ``command:<canonical>`` for any recognized slash command (built-in or plugin).
        # Handlers may return ``{"decision": "deny" | "handled" | "rewrite", ...}`` to intercept
        # dispatch; handlers returning nothing behave as plain observers.
        if command and is_gateway_known_command(canonical):
            raw_args = event.get_command_args().strip()
            hook_ctx = {
                "platform": source.platform.value if source.platform else "",
                "user_id": source.user_id,
                "command": canonical,
                "raw_command": command,
                "args": raw_args,
                "raw_args": raw_args,
            }
            try:
                hook_results = await self.hooks.emit_collect(
                    f"command:{canonical}", hook_ctx
                )
            except Exception as _hook_err:
                logger.debug(
                    "command:%s hook dispatch failed (non-fatal): %s",
                    canonical, _hook_err,
                )
                hook_results = []

            for hook_result in hook_results:
                if not isinstance(hook_result, dict):
                    continue
                decision = str(hook_result.get("decision", "")).strip().lower()
                if not decision or decision == "allow":
                    continue
                if decision == "deny":
                    message = hook_result.get("message")
                    if isinstance(message, str) and message:
                        return True, message, command, canonical
                    return True, f"Command `/{command}` was blocked by a hook.", command, canonical
                if decision == "handled":
                    message = hook_result.get("message")
                    return True, message if isinstance(message, str) and message else None, command, canonical
                if decision == "rewrite":
                    new_command = str(
                        hook_result.get("command_name", "")
                    ).strip().lstrip("/")
                    if not new_command:
                        continue
                    new_args = str(hook_result.get("raw_args", "")).strip()
                    event.text = f"/{new_command} {new_args}".strip()
                    command = event.get_command()
                    _cmd_def = _resolve_cmd(command) if command else None
                    canonical = _cmd_def.name if _cmd_def else command
                    break

        return False, None, command, canonical

    async def _hm_dispatch_canonical_command(
        self,
        event: "MessageEvent",
        source: SessionSource,
        _quick_key: str,
        canonical: Optional[str],
    ) -> Tuple[bool, Optional[str]]:
        """Dispatch built-in idle-path commands (plain handlers, /new, /learn, /plan, /moa, ...).

        Returns ``(handled, result)``. Prompt-rewriting commands (/learn, /plan, /init, /steer,
        /moa, ...) mutate ``event.text`` and return ``(False, None)`` to fall through to the agent.
        """
        plain_handler = (
            self._gateway_plain_command_handlers().get(canonical)
            or self._gateway_idle_command_handlers().get(canonical)
        )
        if plain_handler is not None:
            return True, await plain_handler(event)

        if canonical == "new":
            if await asyncio.to_thread(self._is_telegram_topic_root_lobby, source):
                return True, self._telegram_topic_root_new_message()
            async def _do_reset():
                return await self._handle_reset_command(event)
            return True, await self._maybe_confirm_destructive_slash(
                event=event,
                command="new",
                title="/new",
                detail=(
                    "This starts a fresh session and discards the current "
                    "conversation history."
                ),
                execute=_do_reset,
            )

        if canonical == "start":
            logger.info("Ignoring /start platform ping for session %s", _quick_key)
            return True, ""

        if canonical == "egress":
            from hermes_cli.proxy_cli import format_status_text

            return True, format_status_text()

        if canonical == "learn":
            # Open-ended: rewrite the turn to a standards-guided prompt and fall through to normal
            # agent processing. Mirrors the /blueprint fall-through so role alternation is
            # preserved. No engine, works on any backend.
            from agent.learn_prompt import build_learn_prompt

            _learn_req = event.get_command_args().strip()
            _ack = (
                "Learning a skill from what you described…"
                if _learn_req
                else "Learning a skill from this conversation…"
            )
            await self._send_command_ack(source, _ack, "learn")
            try:
                event.text = build_learn_prompt(_learn_req)
                # fall through to agent processing
            except Exception:
                return True, "Could not start /learn — please try again."

        if canonical == "plan":
            # /plan: rewrite the turn to the plan-mode prompt and fall through to normal agent
            # processing (the /learn fall-through keeps role alternation). Works on any backend.
            from agent.plan_prompt import build_plan_prompt

            _plan_task = event.get_command_args().strip()
            _ack = (
                f"Planning: {_plan_task[:80]}{'…' if len(_plan_task) > 80 else ''}"
                if _plan_task
                else "Planning from this conversation's context…"
            )
            await self._send_command_ack(source, _ack, "plan")
            try:
                event.text = build_plan_prompt(_plan_task)
                # fall through to agent processing
            except Exception:
                return True, "Could not start /plan — please try again."

        if canonical == "init":
            # /init: rewrite the turn to a guidance-laden prompt and fall through to normal agent
            # processing (the /learn fall-through keeps role alternation). Works on any backend.
            from hermes_cli.init_command import build_init_prompt_for_cwd

            _init_notes = event.get_command_args().strip()
            try:
                _init_prompt = build_init_prompt_for_cwd(extra=_init_notes)
            except Exception:
                return True, "Could not start /init — please try again."
            _ack = (
                "Updating AGENTS.md from a project scan…"
                if "UPDATE the existing AGENTS.md" in _init_prompt
                else "Generating AGENTS.md from a project scan…"
            )
            await self._send_command_ack(source, _ack, "init")
            event.text = _init_prompt
            # fall through to agent processing

        if canonical == "blueprint":
            _blueprint_result = await self._handle_blueprint_command(event)
            _blueprint_seed = getattr(_blueprint_result, "agent_seed", None)
            if _blueprint_seed:
                # Blueprint matched — rewrite the turn to the seed and fall through to
                # _handle_message_with_agent so the agent collects each slot value conversationally,
                # then calls the cronjob tool (the /steer fall-through pattern).
                _ack = getattr(_blueprint_result, "text", "") or ""
                if _ack:
                    await self._send_command_ack(source, _ack, "blueprint")
                try:
                    event.text = _blueprint_seed
                except Exception:
                    return True, getattr(_blueprint_result, "text", "") or None
            else:
                return True, getattr(_blueprint_result, "text", "") or None

        if canonical == "undo":
            async def _do_undo():
                return await self._handle_undo_command(event)
            _undo_n = 1
            _undo_raw = event.get_command_args().strip()
            if _undo_raw:
                try:
                    _undo_n = max(1, int(_undo_raw.split()[0]))
                except (ValueError, IndexError):
                    _undo_n = 1
            _undo_detail = (
                "This removes the last user/assistant exchange from history."
                if _undo_n == 1
                else f"This removes the last {_undo_n} user turns from history."
            )
            return True, await self._maybe_confirm_destructive_slash(
                event=event,
                command="undo",
                title="/undo",
                detail=_undo_detail,
                execute=_do_undo,
            )

        if canonical == "queue":
            queue_payload = event.get_command_args().strip()
            if not queue_payload:
                return True, "Usage: /queue <prompt>"
            with suppress(Exception):
                event.text = queue_payload

        if canonical == "steer":
            # No active agent — /steer has nothing to inject into. Strip the prefix so downstream
            # treats it as a normal user message; an empty payload surfaces the usage hint.
            steer_payload = event.get_command_args().strip()
            if not steer_payload:
                return True, "Usage: /steer <prompt>  (no agent is running; sending as a normal message)"
            with suppress(Exception):
                event.text = steer_payload
            # Do NOT return — fall through to _handle_message_with_agent at the end of this function
            # so the rewritten text is sent to the agent as a regular user turn.

        if canonical == "moa":
            # /moa is one-shot sugar only: run a single prompt through the default MoA preset, then
            # restore the prior model. To *switch* to a MoA preset for the session, pick it from the
            # model picker (MoA presets surface as a virtual "Mixture of Agents" provider).
            from hermes_cli.moa_config import (
                moa_usage,
                normalize_moa_config,
            )
            from hermes_cli.config import load_config

            moa_payload = event.get_command_args().strip()
            if not moa_payload:
                return True, moa_usage()
            try:
                cfg = load_config()
                moa_cfg = normalize_moa_config(cfg.get("moa") if isinstance(cfg, dict) else {})
            except Exception:
                moa_cfg = normalize_moa_config({})
            preset = moa_cfg["default_preset"]
            try:
                event.text = moa_payload
                _moa_state = self._session_state(_quick_key)
                event._moa_restore_override = _moa_state.conversation.model_override
                _moa_state.conversation.model_override = {
                    "provider": "moa",
                    "model": preset,
                    "base_url": "moa://local",
                    "api_key": "moa-virtual-provider",
                    "api_mode": "chat_completions",
                }
                self._evict_cached_agent(_quick_key)
                event._moa_disable_after_turn = True
            except Exception:
                return True, "Failed to prepare MoA turn."

        return False, None

    async def _hm_dispatch_quick_and_plugin_commands(
        self, event: "MessageEvent", source: SessionSource, command: Optional[str]
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """Drain gate, user-defined quick commands (exec/alias) and plugin slash commands.

        Returns ``(handled, result, command)`` — an alias quick command rewrites ``command``.
        """
        if self._draining:
            return True, f"⏳ Gateway is {self._status_action_gerund()} and is not accepting new work right now.", command

        # User-defined quick commands (bypass agent loop, no LLM call)
        if command:
            if isinstance(self.config, dict):
                quick_commands = self.config.get("quick_commands", {}) or {}
            else:
                quick_commands = getattr(self.config, "quick_commands", {}) or {}
            if not isinstance(quick_commands, dict):
                quick_commands = {}
            if command in quick_commands:
                # Quick commands are slash capabilities too — and type:exec ones run a shell command
                # in the gateway process. The early gate only fires for registry-known commands and
                # quick commands are never in the registry, so apply the same admin/user policy to
                # the raw typed name here so non-admins can't invoke admin-only quick commands.
                _denied = self._check_slash_access(source, command)
                if _denied is not None:
                    return True, _denied, command
                qcmd = quick_commands[command]
                if qcmd.get("type") == "exec":
                    exec_cmd = qcmd.get("command", "")
                    if exec_cmd:
                        try:
                            # Sanitize env to prevent credential leakage — quick commands run in the
                            # gateway process, which has all API keys in os.environ.
                            from tools.environments.local import build_subprocess_env
                            sanitized_env = build_subprocess_env()
                            proc = await asyncio.create_subprocess_shell(
                                exec_cmd,
                                stdout=asyncio.subprocess.PIPE,
                                stderr=asyncio.subprocess.PIPE,
                                env=sanitized_env,
                            )
                            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
                            output = (stdout or stderr).decode().strip()
                            # Redact any remaining sensitive patterns in output
                            if output:
                                from agent.redact import redact_sensitive_text
                                output = redact_sensitive_text(output)
                            return True, output if output else "Command returned no output.", command
                        except asyncio.TimeoutError:
                            return True, "Quick command timed out (30s).", command
                        except Exception as e:
                            return True, f"Quick command error: {e}", command
                    else:
                        return True, f"Quick command '/{command}' has no command defined.", command
                elif qcmd.get("type") == "alias":
                    target = (qcmd.get("target") or "").strip()
                    if target:
                        target = target if target.startswith("/") else f"/{target}"
                        target_command = target.lstrip("/")
                        user_args = event.get_command_args().strip()
                        event.text = f"{target} {user_args}".strip()
                        command = target_command.split()[0] if target_command else target_command
                        # Fall through to normal command dispatch below
                    else:
                        return True, f"Quick command '/{command}' has no target defined.", command
                else:
                    return True, f"Quick command '/{command}' has unsupported type (supported: 'exec', 'alias').", command

        # Plugin-registered slash commands
        if command:
            try:
                from hermes_cli.plugins import get_plugin_command_handler
                # Normalize underscores to hyphens so Telegram's underscored autocomplete form
                # matches plugin commands registered with hyphens (see _build_telegram_menu).
                plugin_handler = get_plugin_command_handler(command.replace("_", "-"))
                if plugin_handler:
                    user_args = event.get_command_args().strip()
                    result = plugin_handler(user_args)
                    if asyncio.iscoroutine(result):
                        result = await result
                    return True, str(result) if result else None, command
            except Exception as e:
                logger.warning("Plugin command dispatch failed: %s", e)

        return False, None, command

    def _hm_skill_slash_rewrite(
        self,
        event: "MessageEvent",
        source: SessionSource,
        _quick_key: str,
        command: Optional[str],
    ) -> Optional[str]:
        """Rewrite ``/<bundle>`` / ``/<skill>`` invocations into the skill prompt on ``event.text``.

        Returns a reply string when the command is disabled/unknown/failed, else None.
        """
        from gateway.run import _check_unavailable_skill
        from hermes_cli.commands import GATEWAY_KNOWN_COMMANDS

        # Skill slash commands: /skill-name loads the skill and sends to agent.
        # resolve_skill_command_key() handles the Telegram underscore/hyphen round-trip so
        # /claude_code from Telegram autocomplete still resolves to the claude-code skill.
        if command:
            # Skill bundles take precedence over individual skill commands —
            # /<bundle> loads multiple skills at once. Mirrors CLI dispatch.
            _bundle_handled = False
            try:
                from agent.skill_bundles import (
                    build_bundle_invocation_message,
                    resolve_bundle_command_key,
                )
                bundle_key = resolve_bundle_command_key(command)
                if bundle_key is not None:
                    user_instruction = event.get_command_args().strip()
                    # Pass the platform explicitly: bundle skill loading bypasses
                    # get_skill_commands()' scan-time disabled filter, and the gateway serves
                    # multiple platforms in one process, so env-var platform resolution can't be
                    # trusted here.
                    _bundle_plat = source.platform.value if source.platform else None
                    bundle_result = build_bundle_invocation_message(
                        bundle_key, user_instruction, task_id=_quick_key,
                        platform=_bundle_plat,
                    )
                    if bundle_result:
                        msg, _loaded, missing = bundle_result
                        event.text = msg
                        _bundle_handled = True
                        if missing:
                            logger.info(
                                "Bundle %s skipped missing skills: %s",
                                bundle_key, ", ".join(missing),
                            )
                        # Fall through to normal message processing with bundle content
            except Exception as exc:
                logger.warning("Bundle dispatch failed: %s", exc)

        if command and not locals().get("_bundle_handled", False):
            try:
                from agent.skill_commands import (
                    get_skill_commands,
                    build_skill_invocation_message,
                    resolve_skill_command_key,
                )
                skill_cmds = get_skill_commands()
                cmd_key = resolve_skill_command_key(command)
                if cmd_key is not None:
                    # Check per-platform disabled status before executing. get_skill_commands() only
                    # applies the *global* disabled list at scan time; per-platform overrides need
                    # checking here because the cache is process-global across platforms.
                    _skill_name = skill_cmds[cmd_key].get("name", "")
                    _plat = source.platform.value if source.platform else None
                    if _plat and _skill_name:
                        from agent.skill_utils import get_disabled_skill_names as _get_plat_disabled
                        if _skill_name in _get_plat_disabled(platform=_plat):
                            return (
                                f"The **{_skill_name}** skill is disabled for {_plat}.\n"
                                f"Enable it with: `hermes skills config`"
                            )
                    user_instruction = event.get_command_args().strip()
                    # Stacked slash-skill invocations: `/skill-a /skill-b do XYZ` loads every
                    # leading skill (up to 5), not just the first. Mirrors CLI.
                    try:
                        from agent.skill_commands import (
                            build_stacked_skill_invocation_message as _build_stacked,
                            split_stacked_skill_commands,
                        )
                        extra_keys, stacked_instruction = (
                            split_stacked_skill_commands(user_instruction)
                        )
                    except Exception:
                        _build_stacked = None
                        extra_keys, stacked_instruction = [], user_instruction
                    if extra_keys and _plat:
                        # split_stacked_skill_commands() only resolves that each extra token is a
                        # KNOWN skill command — like get_skill_commands() itself, it has no per-
                        # platform view. Re-check every stacked skill against the same disabled list,
                        # or a skill disabled for this platform still gets loaded via the stack.
                        from agent.skill_utils import get_disabled_skill_names as _get_plat_disabled
                        _plat_disabled = _get_plat_disabled(platform=_plat)
                        _disabled_extra = [
                            skill_cmds.get(k, {}).get("name", "")
                            for k in extra_keys
                            if skill_cmds.get(k, {}).get("name", "") in _plat_disabled
                        ]
                        if _disabled_extra:
                            return (
                                f"The **{', '.join(_disabled_extra)}** skill(s) in this "
                                f"stacked invocation are disabled for {_plat}.\n"
                                f"Enable them with: `hermes skills config`"
                            )
                    if extra_keys and _build_stacked is not None:
                        stacked_result = _build_stacked(
                            [cmd_key, *extra_keys],
                            stacked_instruction,
                            task_id=_quick_key,
                        )
                        if stacked_result:
                            msg, _loaded, _missing = stacked_result
                            event.text = msg
                            # Fall through to normal message processing
                        else:
                            return f"Failed to load stacked skills for /{command}."
                    else:
                        msg = build_skill_invocation_message(
                            cmd_key, user_instruction, task_id=_quick_key
                        )
                        if msg:
                            event.text = msg
                            # Fall through to normal message processing with skill content
                else:
                    # Not an active skill — check if it's a known-but-disabled or
                    # uninstalled skill and give actionable guidance.
                    _unavail_msg = _check_unavailable_skill(command)
                    if _unavail_msg:
                        return _unavail_msg
                    # Genuinely unrecognized /command (not built-in/plugin/skill/known-inactive):
                    # warn instead of forwarding to the LLM as free text (it invents tool calls).
                    # Normalize to hyphenated form first: the quick-command block may have set an
                    # alias target, so _cmd_def can be stale.
                    if command.replace("_", "-") not in GATEWAY_KNOWN_COMMANDS:
                        logger.warning(
                            "Unrecognized slash command /%s from %s — "
                            "replying with unknown-command notice",
                            command,
                            source.platform.value if source.platform else "?",
                        )
                        return (
                            f"Unknown command `/{command}`. "
                            f"Type /commands to see what's available, "
                            f"or resend without the leading slash to send "
                            f"as a regular message."
                        )
            except Exception as e:
                logger.debug("Skill command check failed (non-fatal): %s", e)
        return None

    async def _handle_message(self, event: MessageEvent) -> Optional[str]:
        """Handle an incoming message from any platform.

        Pipeline: auth → command check → running-agent interrupt → get/create session → build
        context → run agent → return response.
        """
        from gateway.run import _AGENT_PENDING_SENTINEL
        _admitted = await self._hm_admit_event(event)
        if _admitted is None:
            return None
        event, source, is_internal = _admitted

        _paused_notice = self._hm_estop_gate(event, source, is_internal)
        if _paused_notice is not None:
            return _paused_notice

        # Replies owned by in-flight work: pending /update prompt, clarify, slash-confirm.
        _quick_key = self._session_key_for_source(source)
        allow_gateway_control = event.allow_gateway_control
        _update_reply = self._hm_update_prompt_reply(event, _quick_key, allow_gateway_control)
        if _update_reply is not None:
            return _update_reply
        _clarify_reply = await self._hm_clarify_reply(
            event, source, _quick_key, allow_gateway_control
        )
        if _clarify_reply is not None:
            return _clarify_reply
        _confirm_reply = await self._hm_slash_confirm_reply(
            event, _quick_key, allow_gateway_control
        )
        if _confirm_reply is not None:
            return _confirm_reply

        self._hm_evict_stale_running_agent(_quick_key)
        if self._is_session_running(_quick_key):
            return await self._hm_handle_running_session_message(event, source, _quick_key)

        # Idle path: resolve + dispatch slash commands; rewriting commands fall through to the agent.
        _handled, _result, command, canonical = await self._hm_resolve_command(
            event, source, _quick_key
        )
        if _handled:
            return _result
        _handled, _result = await self._hm_dispatch_canonical_command(
            event, source, _quick_key, canonical
        )
        if _handled:
            return _result
        _handled, _result, command = await self._hm_dispatch_quick_and_plugin_commands(
            event, source, command
        )
        if _handled:
            return _result
        _skill_reply = self._hm_skill_slash_rewrite(event, source, _quick_key, command)
        if _skill_reply is not None:
            return _skill_reply

        # Pending exec approvals go through /approve and /deny only — no bare-text matching, or a
        # conversational "yes" would execute a dangerous command.

        if not is_internal and await asyncio.to_thread(
            self._is_telegram_topic_root_lobby, source
        ):
            # Debounce the lobby reminder so a user who forgets about
            # topic mode and fires ten prompts doesn't get ten copies.
            if self._should_send_telegram_lobby_reminder(source):
                return self._telegram_topic_root_lobby_message()
            return None

        # ── External-drain new-turn gate ─────────────────────────────
        # When NAS engaged an external drain (.drain_request.json, seen by _drain_control_watcher),
        # refuse to START new turns so the in-flight set can only fall to zero (stop accepting
        # FIRST, then NAS polls active_agents==0). Internal/system events bypass; reversible.
        if self._external_drain_active and not is_internal:
            logger.info(
                "Refusing new turn for session %s — external drain active.",
                _quick_key,
            )
            return (
                "⏳ This agent is draining for a maintenance action and isn't "
                "accepting new turns right now. It'll be back in a moment — "
                "please resend shortly."
            )

        # ── Claim this session before any await ───────────────────────
        # Many awaits sit between here and _run_agent registering the real AIAgent; without this
        # sentinel a second message during any of them passes the "already running" guard and spins
        # up a duplicate agent for the same session, corrupting the transcript.
        _active_session_lease, _limit_message = self._claim_active_session_slot(
            _quick_key,
            source,
        )
        if _limit_message is not None:
            logger.info(
                "Rejecting new active session %s: max_concurrent_sessions reached",
                _quick_key,
            )
            return _limit_message

        # ── FIFO orphan rescue ───────────────────────────────────────
        # A session that went idle with a populated overflow (post-turn drain never promoted, e.g. a
        # compression-demoted follow-up) silently orphaned those events. Re-stage them FIFO and
        # enqueue this event behind them. Skipped for control commands and internal events.
        try:
            _orphan_adapter = self._adapter_for_source(source)
            if (
                _orphan_adapter is not None
                and not bool(getattr(event, "internal", False))
                and not event.get_command()
            ):
                _rescued = self._rescue_orphaned_overflow(
                    _quick_key, _orphan_adapter
                )
                if _rescued is not None:
                    # The oldest orphan runs as THIS turn. Park the incoming event behind the rest of
                    # the chain: into the slot when the chain was a single orphan (post-turn drain
                    # picks it up), otherwise into overflow behind the already-staged next orphan.
                    self._enqueue_fifo(_quick_key, event, _orphan_adapter)
                    event = _rescued
                    # Same session key by construction; carry the orphan's own source so reply
                    # anchors / thread metadata point at the message actually being answered.
                    _rescued_source = getattr(_rescued, "source", None)
                    if _rescued_source is not None:
                        source = _rescued_source
                    is_internal = bool(getattr(_rescued, "internal", False))
        except Exception:
            logger.debug(
                "FIFO orphan rescue pre-claim failed for %s",
                _quick_key,
                exc_info=True,
            )

        _claim_state = self._session_state(_quick_key)
        if _active_session_lease is not None:
            _claim_state.turn.lease = _active_session_lease
        _claim_state.turn.agent = _AGENT_PENDING_SENTINEL
        _claim_state.turn.started_ts = time.time()
        self._persist_active_agents()
        _run_generation = self._begin_session_run_generation(_quick_key)

        try:
            try:
                _agent_result = await self._handle_message_with_agent(
                    event, source, _quick_key, _run_generation
                )
            except TurnLeaseTimeoutError as exc:
                # A rejected message, not a completed turn: return before the /goal judge below so
                # it cannot consume the resend notice and enqueue a synthetic continuation loop.
                logger.error(
                    "Rejecting turn for routing key %s on session %s after "
                    "turn-lease timeout; transcript load was not started and "
                    "the user must resend",
                    _quick_key,
                    exc.session_id,
                )
                return (
                    "⏳ Another turn is still running on this session. To "
                    "protect the transcript, this message was not processed. "
                    "Wait for the active turn to finish, then resend it."
                )
            try:
                await self._run_post_turn_hooks(
                    agent_result=_agent_result,
                    source=source,
                    is_internal=is_internal,
                    event=event,
                )
            except Exception as _goal_exc:
                logger.debug("post-turn hook failed: %s", _goal_exc)
            return _agent_result
        finally:
            # MoA one-shot restore must run on EVERY exit path: the restore data lives on the
            # per-turn event, so a restore in the try block is skipped when the handler raises and
            # the override leaks permanently; finally covers success, exception and interrupt.
            self._restore_moa_one_shot(event, _quick_key)
            self._restore_pending_one_turn_model_override(_quick_key)
            # Normal completion/exception/interrupt clears this durable marker; SIGKILL/OOM skips
            # finally, leaving it for the next unclean startup's recovery pass.
            await self._clear_durable_active_turn(event)
            # Unconditional release covers every exit path: _release_running_agent_state is idempotent
            # and, without a run_generation guard, clears the slot whichever generation holds it. This
            # evicts the zombie left when session_reset bumps the generation mid-flight (gen-N's
            # guarded release in _run_agent returns False; a sentinel-only check would lock forever).
            self._release_running_agent_state(_quick_key)
            # Turn lease: release THIS turn's token — keyed by (routing key, run generation) so this
            # unwind can only free the lease its own turn acquired, never a newer turn's.
            self._release_turn_lease(_quick_key, _run_generation)

    def _restore_moa_one_shot(self, event: "MessageEvent", quick_key: str) -> None:
        """Revert a ``/moa <prompt>`` one-shot model override after its turn.

        Called from the message-handling ``finally`` so it fires on success, error or interrupt.
        No-op unless ``event._moa_disable_after_turn``; ``_moa_restore_override`` holds the prior
        per-session override (``None`` = clear the MoA override outright).
        """
        if not getattr(event, "_moa_disable_after_turn", False):
            return
        try:
            _restore = getattr(event, "_moa_restore_override", None)
            self._session_state(quick_key).conversation.model_override = _restore
            self._evict_cached_agent(quick_key)
        except Exception:
            pass

    def _restore_pending_one_turn_model_override(self, session_key: str) -> None:
        """Restore a per-session model override after ``/model --once`` runs."""
        if not session_key:
            return
        try:
            _otr_state = self._peek_session_state(session_key)
            snapshot = _otr_state.conversation.one_turn_restore if _otr_state else None
            if _otr_state is not None:
                _otr_state.conversation.one_turn_restore = None
            if not snapshot:
                return
            self._restore_session_model_override(session_key, snapshot)
        except Exception:
            logger.debug("Failed to restore one-turn model override", exc_info=True)

    def _prefix_inbound_sender_context(self, event: MessageEvent, source: SessionSource, message_text: str) -> str:
        """Attribute the sender in shared multi-user sessions and prepend history-backfill channel context."""
        _group_sessions_per_user = getattr(self.config, "group_sessions_per_user", True)
        _thread_sessions_per_user = getattr(self.config, "thread_sessions_per_user", False)
        _is_shared_multi_user = is_shared_multi_user_session(
            source,
            group_sessions_per_user=_group_sessions_per_user,
            thread_sessions_per_user=_thread_sessions_per_user,
        )
        if _is_shared_multi_user and source.user_name:
            # source.user_name is the platform display name — attacker-influenceable on any
            # platform that lets participants set their own name. Neutralize newlines/control chars
            # before interpolating it into every message, or a hostile name can masquerade as a
            # fake markdown section (mirrors build_session_context_prompt's treatment).
            _safe_user_name = neutralize_untrusted_inline_text(source.user_name)
            # On Slack, expose the current author's verifiable user ID next to the display name:
            # "mention me again" requests need a trusted `<@U...>` target for the CURRENT speaker —
            # display names are ambiguous and historical mentions may point at someone else. The
            # user_id comes from the Slack event envelope (not user-editable), so no neutralization.
            if source.platform == Platform.SLACK and source.user_id:
                _safe_user_name = (
                    f"{_safe_user_name} | Slack user <@{source.user_id}>"
                )
            message_text = f"[{_safe_user_name}] {message_text}"

        # Prepend history-backfill channel context after the sender-prefix so the prefix applies
        # only to the trigger message, not the backfill block.
        if getattr(event, "channel_context", None):
            message_text = f"{event.channel_context}\n\n[New message]\n{message_text}"
        return message_text

    @staticmethod
    def _classify_inbound_media(
        event: MessageEvent, pending_stt_prepared: bool
    ) -> Tuple[list, list, list, list]:
        """Split ``event.media_urls`` into (image, STT-voice, audio-file, video) paths."""
        from gateway.run import _event_media_is_audio, _event_media_is_image, _event_media_is_stt_input
        image_paths: list[str] = []
        audio_paths: list[str] = []
        audio_file_paths: list[str] = []
        video_paths: list[str] = []

        if event.media_urls:
            for i, path in enumerate(event.media_urls):
                mtype = event.media_types[i] if i < len(event.media_types) else ""
                # Classify images per-attachment: trust this attachment's own MIME, and only honour
                # the message-level PHOTO type when the per-attachment MIME is unknown. Otherwise a
                # document sent alongside an image gets mis-routed as an image and the provider 400s.
                if _event_media_is_image(event, i):
                    image_paths.append(path)
                # MessageType.AUDIO = audio file attachment (e.g. .mp3, .m4a) — never STT.
                # Mixed DOCUMENT events also preserve audio as a file path instead of
                # dropping it or treating it as a voice note.
                if _event_media_is_audio(event, i):
                    if event.message_type in {MessageType.AUDIO, MessageType.DOCUMENT}:
                        audio_file_paths.append(path)
                    elif not pending_stt_prepared and _event_media_is_stt_input(event, i):
                        audio_paths.append(path)
                if mtype.startswith("video/") or (not mtype and event.message_type == MessageType.VIDEO):
                    video_paths.append(path)
        return image_paths, audio_paths, audio_file_paths, video_paths

    async def _enrich_inbound_images(
        self, source: SessionSource, session_key: str, message_text: str, image_paths: list[str]
    ) -> str:
        # Decide routing: native (attach pixels) vs text (vision_analyze pre-run + prepend
        # description). See agent/image_routing.py. Offloaded to a thread: the decision does
        # blocking network I/O (models.dev fetch on cache miss, Ollama /api/show probe) whose
        # timeout would otherwise stall the whole gateway event loop.
        _img_mode = await asyncio.to_thread(
            self._decide_image_input_mode,
            source=source,
            session_key=session_key,
        )
        if _img_mode == "native":
            # Defer attachment to the run_conversation call site.
            self._session_state(
                session_key
            ).persistent.native_image_paths = list(image_paths)
            logger.info(
                "Image routing: native (model supports vision). %d image(s) will be attached inline.",
                len(image_paths),
            )
        else:
            logger.info(
                "Image routing: text (mode=%s). Pre-analyzing %d image(s) via vision_analyze.",
                _img_mode, len(image_paths),
            )
            # Vision enrichment runs before AIAgent.run_conversation(),
            # so bind this session's resolved runtime explicitly rather
            # than consulting process-global compatibility mirrors.
            vision_runtime = None
            try:
                turn_model, runtime_kwargs = self._resolve_session_agent_runtime(
                    source=source,
                    session_key=session_key,
                )
                vision_runtime = dict(runtime_kwargs or {})
                vision_runtime["model"] = turn_model
            except Exception:
                logger.debug(
                    "vision enrichment: session runtime resolution failed",
                    exc_info=True,
                )

            from agent.auxiliary_client import scoped_runtime_main

            with scoped_runtime_main(vision_runtime):
                message_text = await self._enrich_message_with_vision(
                    message_text,
                    image_paths,
                )
        return message_text

    async def _enrich_inbound_voice(
        self, event: MessageEvent, source: SessionSource, message_text: str, audio_paths: list[str]
    ) -> str:
        message_text, _successful_transcripts = await self._enrich_message_with_transcription(
            message_text,
            audio_paths,
        )
        # Echo each successful transcript back to the user immediately when configured. Lets
        # users verify STT quality in real-time, while allowing quiet STT for users who only
        # want the agent to receive the transcription.
        if _successful_transcripts and self._should_echo_stt_transcripts():
            _echo_adapter = self._adapter_for_source(source)
            _echo_meta = self._thread_metadata_for_source(source, self._reply_anchor_for_event(event))
            if _echo_adapter:
                for _tx in _successful_transcripts:
                    try:
                        await _echo_adapter.send(
                            source.chat_id,
                            f'🎙️ "{_tx}"',
                            metadata=_echo_meta,
                        )
                    except Exception as _echo_exc:
                        logger.debug(
                            "Transcript echo failed (non-fatal): %s", _echo_exc,
                        )
        # On transcription failure, do NOT send a hardcoded notice here: that bypassed the
        # LLM and produced two replies (one pre-canned, TTS'd in the wrong language).
        # Enrichment leaves a single neutral marker so the LLM gives one localized reply.
        return message_text

    @staticmethod
    def _prepend_inbound_media_file_notes(message_text: str, audio_file_paths: list[str], video_paths: list[str]) -> str:
        if audio_file_paths:
            from tools.credential_files import to_agent_visible_cache_path as _to_agent_path
            for _apath in audio_file_paths:
                _basename = os.path.basename(_apath)
                _parts = _basename.split("_", 2)
                _display = _parts[2] if len(_parts) >= 3 else _basename
                _display = re.sub(r'[^\w.\- ]', '_', _display)
                _agent_path = _to_agent_path(_apath)
                _note = (
                    f"[The user sent an audio file attachment: '{_display}'. "
                    f"It is saved at: {_agent_path}. "
                    f"Its content is not inlined here. If the user's request involves "
                    f"what the audio contains, transcribe or process it yourself — for "
                    f"example by passing the path to a transcription or media tool — "
                    f"instead of asking the user to describe it. Only ask what to do "
                    f"with it if their intent is genuinely unclear.]"
                )
                message_text = f"{_note}\n\n{message_text}"

        if video_paths:
            from tools.credential_files import to_agent_visible_cache_path as _to_agent_path
            for _vpath in video_paths:
                _basename = os.path.basename(_vpath)
                _parts = _basename.split("_", 2)
                _display = _parts[2] if len(_parts) >= 3 else _basename
                _display = re.sub(r'[^\w.\- ]', '_', _display)
                _agent_path = _to_agent_path(_vpath)
                _note = (
                    f"[The user sent a video attachment: '{_display}'. "
                    f"It is saved at: {_agent_path}. "
                    f"Its content is not inlined here. If the user's request involves "
                    f"what the video contains, inspect or process it yourself — for "
                    f"example by passing the path to a video analysis or media tool — "
                    f"instead of asking the user to describe it. Only ask what to do "
                    f"with it if their intent is genuinely unclear.]"
                )
                message_text = f"{_note}\n\n{message_text}"
        return message_text

    @staticmethod
    def _prepend_inbound_document_notes(event: MessageEvent, message_text: str) -> str:
        from gateway.run import (
            _build_document_context_note,
            _event_media_is_audio,
            _event_media_is_image,
            _event_media_is_video,
        )
        if event.media_urls:
            import mimetypes as _mimetypes
            from tools.credential_files import to_agent_visible_cache_path

            _TEXT_EXTENSIONS = {".txt", ".md", ".csv", ".log", ".json", ".xml", ".yaml", ".yml", ".toml", ".ini", ".cfg"}
            for i, path in enumerate(event.media_urls):
                # Per-attachment document handling: skip anything already routed as image/audio/video
                # above; only genuine non-media files get a context note. A document mixed into a
                # PHOTO/VOICE message (message-level type != DOCUMENT) thus still reaches the agent.
                if (
                    _event_media_is_image(event, i)
                    or _event_media_is_audio(event, i)
                    or _event_media_is_video(event, i)
                ):
                    continue
                mtype = event.media_types[i] if i < len(event.media_types) else ""
                if mtype in {"", "application/octet-stream"}:
                    _ext = os.path.splitext(path)[1].lower()
                    if _ext in _TEXT_EXTENSIONS:
                        mtype = "text/plain"
                    else:
                        guessed, _ = _mimetypes.guess_type(path)
                        mtype = guessed or "application/octet-stream"
                # Any accepted file gets a path-pointing context note — we accept
                # all file types now, so a non-text/non-application MIME (font/*,
                # model/*, etc.) must still tell the agent the file exists.

                basename = os.path.basename(path)
                parts = basename.split("_", 2)
                display_name = parts[2] if len(parts) >= 3 else basename
                display_name = re.sub(r'[^\w.\- ]', '_', display_name)

                # Translate host cache path to in-container path if running under Docker backend.
                # This ensures the agent receives a path it can open inside its sandbox, as the
                # cache directories are auto-mounted at /root/.hermes/cache/* by get_cache_directory_mounts().
                agent_path = to_agent_visible_cache_path(path)

                inline_flags = getattr(event, "media_text_inlined", None) or []
                inline_flag = inline_flags[i] if i < len(inline_flags) else None
                context_note = _build_document_context_note(
                    display_name,
                    agent_path,
                    mtype,
                    content_inlined=inline_flag is not False,
                )
                message_text = f"{context_note}\n\n{message_text}"
        return message_text

    @staticmethod
    def _prepend_inbound_reply_context(event: MessageEvent, source: SessionSource, message_text: str) -> str:
        # Discord: surface the triggering message id per-turn on the user message rather than in the
        # cached system prompt. message_id changes every turn, so baking it into
        # build_session_context_prompt() would bust the agent-cache signature and rebuild the
        # AIAgent every message (destroying prompt caching).
        if (
            source is not None
            and getattr(source, "platform", None) == Platform.DISCORD
            and getattr(event, "message_id", None)
        ):
            from gateway.session import _discord_tools_loaded as _disc_tools_loaded
            if _disc_tools_loaded():
                message_text = (
                    f"[Triggering message id: `{event.message_id}` — use as "
                    f"`message_id` for reply/react/pin via the discord tools.]\n\n"
                    f"{message_text}"
                )

        if getattr(event, "reply_to_text", None) and event.reply_to_message_id:
            # Always inject the reply-to pointer — even when the quoted text already appears in
            # history. The prefix isn't deduplication, it's disambiguation: it tells the agent
            # *which* prior message the user is referencing. Token overhead is minimal.
            reply_snippet = event.reply_to_text[:500]
            if getattr(event, "reply_to_is_own_message", False):
                message_text = (
                    f'[Replying to your previous message: "{reply_snippet}"]\n\n'
                    f"{message_text}"
                )
            else:
                message_text = f'[Replying to: "{reply_snippet}"]\n\n{message_text}'
        return message_text

    async def _expand_inbound_context_references(
        self, source: SessionSource, session_key: str, message_text: str
    ) -> Optional[str]:
        """Expand ``@`` context references; returns None when the injection was refused (user notified)."""
        from gateway.run import _load_gateway_config
        try:
            from agent.context_references import preprocess_context_references_async
            from agent.model_metadata import get_model_context_length_async

            try:
                from tools.terminal_scope import terminal_env as _ts_env
            except ImportError:
                _msg_cwd = os.environ.get("TERMINAL_CWD", os.path.expanduser("~"))
            else:
                _msg_cwd = _ts_env("TERMINAL_CWD", os.path.expanduser("~"))
            _msg_config_ctx = None
            _msg_cfg = None
            _msg_model_cfg = {}
            _msg_custom_providers = []
            try:
                _msg_cfg = _load_gateway_config()
                _msg_model_cfg = _msg_cfg.get("model", {})
                if isinstance(_msg_model_cfg, dict):
                    _msg_raw_ctx = _msg_model_cfg.get("context_length")
                    if _msg_raw_ctx is not None:
                        _msg_config_ctx = int(_msg_raw_ctx)
                try:
                    from hermes_cli.config import get_compatible_custom_providers

                    _msg_custom_providers = get_compatible_custom_providers(_msg_cfg)
                except Exception:
                    _msg_custom_providers = _msg_cfg.get("custom_providers") or []
            except Exception:
                pass
            # Resolve the session's actual model/provider/base_url as the hygiene compression
            # block does; GatewayRunner has no self._model/self._base_url (AttributeError,
            # silently caught below).
            _msg_model, _msg_runtime = self._resolve_session_agent_runtime(
                source=source,
                session_key=session_key,
                user_config=_msg_cfg,
            )
            _msg_base_url = _msg_runtime.get("base_url") or ""
            # A global model.context_length belongs to the configured
            # model, not a session /model or channel override. Prefer a
            # matching per-custom-provider model limit when available.
            _msg_configured_model = (
                _msg_model_cfg.get("default") or _msg_model_cfg.get("model")
                if isinstance(_msg_model_cfg, dict)
                else _msg_model_cfg
            )
            if _msg_model != _msg_configured_model:
                _msg_config_ctx = None
            if _msg_config_ctx is not None and isinstance(_msg_model_cfg, dict):
                try:
                    from hermes_cli.route_identity import should_clear_context_pin_async

                    if await should_clear_context_pin_async(
                        None,  # model match already checked above
                        None,
                        _msg_model_cfg.get("base_url"),
                        _msg_base_url,
                        _msg_model_cfg.get("provider"),
                        _msg_runtime.get("provider"),
                    ):
                        _msg_config_ctx = None
                except Exception:
                    _msg_config_ctx = None
            if _msg_custom_providers and _msg_base_url:
                try:
                    from hermes_cli.config import get_custom_provider_context_length

                    _msg_custom_ctx = get_custom_provider_context_length(
                        model=_msg_model,
                        base_url=_msg_base_url,
                        custom_providers=_msg_custom_providers,
                    )
                    if _msg_custom_ctx:
                        _msg_config_ctx = _msg_custom_ctx
                except Exception:
                    pass
            _msg_ctx_len = await get_model_context_length_async(
                _msg_model,
                base_url=_msg_base_url,
                api_key=_msg_runtime.get("api_key") or "",
                config_context_length=_msg_config_ctx,
                provider=_msg_runtime.get("provider") or "",
                custom_providers=_msg_custom_providers,
            )
            _ctx_result = await preprocess_context_references_async(
                message_text,
                cwd=_msg_cwd,
                context_length=_msg_ctx_len,
                allowed_root=_msg_cwd,
            )
            if _ctx_result.blocked:
                _adapter = self._adapter_for_source(source)
                if _adapter:
                    await _adapter.send(
                        source.chat_id,
                        "\n".join(_ctx_result.warnings) or "Context injection refused.",
                    )
                return None
            if _ctx_result.expanded:
                message_text = _ctx_result.message
        except Exception as exc:
            logger.warning("@ context reference expansion failed: %s", exc)
            logger.debug("@ context reference expansion failure detail", exc_info=True)
        return message_text

    async def _prepare_inbound_message_text(
        self,
        *,
        event: MessageEvent,
        source: SessionSource,
        history: List[Dict[str, Any]],
        session_key: Optional[str] = None,
    ) -> Optional[str]:
        """Prepare inbound event text for the agent.

        Shared by the normal inbound and queued follow-up paths so attribution, image enrichment,
        STT, document notes, reply context and @ references behave the same. Side effect: buffers
        per-session native image paths when the model supports native vision; the caller consumes
        that buffer at ``run_conversation``. Empty list means the text vision path already ran.
        """
        history = history or []
        _pending_stt_prepared = hasattr(event, "_gateway_pending_stt_text")
        message_text = (
            getattr(event, "_gateway_pending_stt_text", None)
            if _pending_stt_prepared
            else event.text
        ) or ""
        # Prefer the caller's resolved session key so this write key matches the consume key at the
        # run_conversation site; derive it here only for tests and legacy standalone callers.
        session_key = session_key or self._session_key_for_source(source)
        # Reset only this session's per-call buffer; other sessions may be
        # concurrently preparing multimodal turns on the same runner.
        self._consume_pending_native_image_paths(session_key)

        message_text = self._prefix_inbound_sender_context(event, source, message_text)
        image_paths, audio_paths, audio_file_paths, video_paths = self._classify_inbound_media(
            event, _pending_stt_prepared
        )
        if image_paths:
            message_text = await self._enrich_inbound_images(source, session_key, message_text, image_paths)
        if audio_paths:
            message_text = await self._enrich_inbound_voice(event, source, message_text, audio_paths)
        message_text = self._prepend_inbound_media_file_notes(message_text, audio_file_paths, video_paths)
        message_text = self._prepend_inbound_document_notes(event, message_text)
        message_text = self._prepend_inbound_reply_context(event, source, message_text)
        if "@" in message_text:
            message_text = await self._expand_inbound_context_references(source, session_key, message_text)
            if message_text is None:
                return None

        return message_text

    async def _prepare_profile_scoped_inbound_message_text(
        self,
        *,
        event: MessageEvent,
        source: SessionSource,
        history: List[Dict[str, Any]],
        session_key: Optional[str] = None,
    ) -> Optional[str]:
        """Run inbound preprocessing under the routed profile when multiplexed."""
        from gateway.run import _async_profile_runtime_scope
        if getattr(getattr(self, "config", None), "multiplex_profiles", False):
            async with _async_profile_runtime_scope(
                self._resolve_profile_home_for_source(source)
            ):
                return await self._prepare_inbound_message_text(
                    event=event,
                    source=source,
                    history=history,
                    session_key=session_key,
                )
        return await self._prepare_inbound_message_text(
            event=event,
            source=source,
            history=history,
            session_key=session_key,
        )

    async def _prepare_clarify_reply_text(self, event) -> str:
        """Return raw text or successful voice transcripts for a clarify reply."""
        if not self._pending_event_audio_paths(event):
            return (event.text or "").strip()

        _, successful_transcripts = await self._transcribe_pending_audio_event_once(
            event, "",
        )
        return "\n\n".join(
            transcript.strip()
            for transcript in successful_transcripts
            if transcript.strip()
        )

    def _consume_pending_native_image_paths(self, session_key: str) -> List[str]:
        state = self._peek_session_state(session_key)
        if state is None or not state.persistent.native_image_paths:
            return []
        paths = list(state.persistent.native_image_paths)
        state.persistent.native_image_paths = []
        return paths

    async def _mark_durable_active_turn(
        self,
        event: "MessageEvent",
        session_key: str,
    ) -> bool:
        """Persist the exact resolved routing key for this running turn."""
        try:
            token = await self.async_session_store.mark_turn_active(session_key)
        except Exception as exc:
            logger.warning(
                "Could not persist active-turn marker for %s: %s",
                session_key,
                exc,
            )
            return False
        if not token:
            return False
        # Private event attributes are process-local ownership state.  Keep the
        # token out of public metadata, transcripts, and platform payloads.
        setattr(event, "_gateway_active_turn_session_key", session_key)
        setattr(event, "_gateway_active_turn_token", token)
        return True

    async def _clear_durable_active_turn(self, event: "MessageEvent") -> bool:
        """Best-effort CAS clear of the marker owned by *event*."""
        session_key = getattr(event, "_gateway_active_turn_session_key", None)
        token = getattr(event, "_gateway_active_turn_token", None)
        try:
            if not session_key or not token:
                return False
            last_error: Optional[Exception] = None
            for attempt in range(1, 4):
                try:
                    return bool(
                        await self.async_session_store.clear_turn_active(
                            session_key, token
                        )
                    )
                except Exception as exc:
                    last_error = exc
                    if attempt < 3:
                        logger.debug(
                            "Retrying active-turn marker cleanup for %s (%d/3): %s",
                            session_key,
                            attempt,
                            exc,
                        )
            # Never let marker cleanup block agent/lease release; a stale marker is bounded by the
            # agent timeout and the clean-start orphan-marker discard path.
            logger.warning(
                "Could not clear active-turn marker for %s after 3 attempts: %s",
                session_key,
                last_error,
            )
            return False
        finally:
            for attr in (
                "_gateway_active_turn_session_key",
                "_gateway_active_turn_token",
            ):
                with suppress(AttributeError):
                    delattr(event, attr)

    def _install_plugin_message_injector(self) -> None:
        """Publish this live gateway's plugin message scheduler."""
        from hermes_cli.plugins import get_plugin_manager

        get_plugin_manager().set_gateway_message_injector(
            self,
            self._schedule_plugin_message_injection,
        )

    def _clear_plugin_message_injector(self) -> None:
        """Remove this runner's scheduler without clobbering a newer owner."""
        from hermes_cli.plugins import get_plugin_manager

        get_plugin_manager().clear_gateway_message_injector(self)

    def _schedule_plugin_message_injection(
        self,
        *,
        session_key: str,
        content: str,
        plugin_id: str,
    ) -> bool:
        """Schedule a plugin-triggered turn on the live gateway loop."""
        from gateway.run import safe_schedule_threadsafe
        loop = getattr(self, "_gateway_loop", None)
        if not getattr(self, "_running", False) or loop is None or loop.is_closed():
            return False

        coro = self._dispatch_plugin_message_injection(
            session_key=session_key,
            content=content,
            plugin_id=plugin_id,
        )
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None

        if current_loop is loop:
            try:
                future = loop.create_task(coro)
            except Exception:
                coro.close()
                logger.warning(
                    "Plugin message injection scheduling failed",
                    exc_info=True,
                )
                return False
            self._background_tasks.add(future)
            future.add_done_callback(self._background_tasks.discard)
        else:
            future = safe_schedule_threadsafe(
                coro,
                loop,
                logger=logger,
                log_message="Plugin message injection scheduling failed",
                log_level=logging.WARNING,
            )
            if future is None:
                return False

        def _log_result(completed) -> None:
            try:
                accepted = completed.result()
            except (asyncio.CancelledError, concurrent.futures.CancelledError):
                return
            except Exception:
                logger.warning(
                    "Plugin message injection failed: plugin=%s session=%s",
                    plugin_id,
                    session_key,
                    exc_info=True,
                )
                return
            if not accepted:
                logger.warning(
                    "Plugin message injection was not routed: plugin=%s session=%s",
                    plugin_id,
                    session_key,
                )

        future.add_done_callback(_log_result)
        return True

    async def _dispatch_plugin_message_injection(
        self,
        *,
        session_key: str,
        content: str,
        plugin_id: str,
    ) -> bool:
        """Route a plugin-triggered turn through the session's live adapter."""
        if not getattr(self, "_running", False) or getattr(self, "_draining", False):
            return False

        entry = await self.async_session_store.lookup_by_session_key(session_key)
        if entry is None or entry.origin is None:
            return False
        if not getattr(self, "_running", False) or getattr(self, "_draining", False):
            return False

        source = dataclasses.replace(entry.origin)
        try:
            if not self._is_user_authorized(
                source,
                allow_adapter_delegation=False,
            ):
                logger.warning(
                    "Plugin message injection denied by current gateway authorization: "
                    "plugin=%s session=%s",
                    plugin_id,
                    session_key,
                )
                return False
        except Exception:
            logger.warning(
                "Plugin message injection authorization check failed: "
                "plugin=%s session=%s",
                plugin_id,
                session_key,
                exc_info=True,
            )
            return False

        adapter = self._adapter_for_source(source)
        if adapter is None:
            return False

        event = MessageEvent(
            text=content,
            message_type=MessageType.TEXT,
            source=source,
            internal=True,
            allow_gateway_control=False,
            metadata={
                "hermes_plugin_id": plugin_id,
                "hermes_plugin_injection": True,
                "gateway_session_key": session_key,
                "gateway_session_id": entry.session_id,
                "gateway_session_strict": True,
            },
        )
        await adapter.handle_message(event)
        logger.info(
            "Plugin message injection dispatched: plugin=%s session=%s session_id=%s",
            plugin_id,
            session_key,
            entry.session_id,
        )
        return True

    def _decide_image_input_mode(
        self,
        *,
        source: Optional[SessionSource] = None,
        session_key: Optional[str] = None,
        user_config: Optional[dict] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> str:
        """Resolve image-input routing for the effective model this turn.

        Returns ``"native"`` (attach pixels on the user turn) or ``"text"`` (pre-analyze with
        vision_analyze and prepend the description); see agent/image_routing.py. Gateway sessions
        can carry /model overrides and image preprocessing runs before AIAgent sets the
        auxiliary_client runtime globals, so resolve the per-session runtime bundle the upcoming
        turn will use, not just the persisted default.
        """
        try:
            from agent.image_routing import decide_image_input_mode
            from agent.auxiliary_client import _read_main_model, _read_main_provider
            from hermes_cli.config import load_config

            cfg = user_config if isinstance(user_config, dict) else load_config()
            resolved_provider = (provider or "").strip()
            resolved_model = (model or "").strip()
            resolved_requested_provider = ""

            needs_session_runtime = not resolved_provider or not resolved_model
            has_session_identity = source is not None or session_key
            if needs_session_runtime and has_session_identity:
                try:
                    turn_model, runtime_kwargs = self._resolve_session_agent_runtime(
                        source=source,
                        session_key=session_key,
                        user_config=cfg,
                    )
                    if not resolved_model and isinstance(turn_model, str):
                        resolved_model = turn_model.strip()
                    runtime_provider = runtime_kwargs.get("provider") if isinstance(runtime_kwargs, dict) else None
                    runtime_requested_provider = (
                        runtime_kwargs.get("requested_provider")
                        if isinstance(runtime_kwargs, dict)
                        else None
                    )
                    if not resolved_provider and isinstance(runtime_provider, str):
                        resolved_provider = runtime_provider.strip()
                    if isinstance(runtime_requested_provider, str):
                        resolved_requested_provider = runtime_requested_provider.strip()
                except Exception as exc:
                    logger.debug(
                        "image_routing: session runtime resolution failed, falling back to config — %s",
                        exc,
                    )

            if not resolved_provider:
                resolved_provider = _read_main_provider()
            if not resolved_model:
                resolved_model = _read_main_model()

            return decide_image_input_mode(
                resolved_provider,
                resolved_model,
                cfg,
                requested_provider=resolved_requested_provider,
            )
        except Exception as exc:
            logger.debug("image_routing: decision failed, falling back to text — %s", exc)
            return "text"

    async def _enrich_message_with_vision(
        self,
        user_text: str,
        image_paths: List[str],
    ) -> str:
        """Auto-analyze user-attached images with the vision tool and prepend the descriptions to
        the message text.

        Description *and* local cache path are injected so the model understands the image without
        a tool call and can re-examine it with vision_analyze. Returns the enriched message string.
        """
        from tools.vision_tools import vision_analyze_tool
        from agent.memory_manager import sanitize_context

        analysis_prompt = (
            "Concisely describe this image in 2-4 sentences "
            "(~200 Chinese characters or ~150 English words). "
            "Cover the main subject, key visible text/data/code, and overall context. "
            "If it is a chart, diagram, or scientific figure, include the important "
            "labels, legend, and key values. Skip decorative details."
        )

        enriched_parts = []
        for path in image_paths:
            try:
                logger.debug("Auto-analyzing user image: %s", path)
                result_json = await vision_analyze_tool(
                    image_url=path,
                    user_prompt=analysis_prompt,
                )
                result = json.loads(result_json)
                if result.get("success"):
                    description = result.get("analysis", "")
                    description = sanitize_context(description)
                    enriched_parts.append(
                        f"[The user sent an image~ Here's what I can see:\n{description}]\n"
                        f"[If you need a closer look, use vision_analyze with "
                        f"image_url: {path} ~]"
                    )
                else:
                    enriched_parts.append(
                        "[The user sent an image but I couldn't quite see it "
                        "this time (>_<) You can try looking at it yourself "
                        f"with vision_analyze using image_url: {path}]"
                    )
            except Exception as e:
                logger.error("Vision auto-analysis error: %s", e)
                enriched_parts.append(
                    f"[The user sent an image but something went wrong when I "
                    f"tried to look at it~ You can try examining it yourself "
                    f"with vision_analyze using image_url: {path}]"
                )

        # Combine: vision descriptions first, then the user's original text
        if enriched_parts:
            prefix = "\n\n".join(enriched_parts)
            if user_text:
                return f"{prefix}\n\n{user_text}"
            return prefix
        return user_text

    async def _enrich_message_with_transcription(
        self,
        user_text: str,
        audio_paths: List[str],
    ) -> tuple[str, List[str]]:
        """Auto-transcribe user voice/audio messages using the configured STT provider and prepend
        the transcript to the message text.

        Returns ``(enriched_text, successful_transcripts)``: the message with transcription
        wrappers prepended, and the raw transcripts of successfully transcribed clips in input
        order (empty if every clip failed or STT is disabled) so callers can echo them back to
        the user before the agent loop.
        """
        from gateway.run import _probe_audio_duration
        seen = set()
        audio_paths = [p for p in audio_paths if p not in seen and not seen.add(p)]
        if not getattr(self.config, "stt_enabled", True):
            notes = []
            for path in audio_paths:
                abs_path = os.path.abspath(path)
                duration_str = await _probe_audio_duration(abs_path)
                if duration_str:
                    notes.append(
                        f"[The user sent a voice message: {abs_path} (duration: {duration_str})]"
                    )
                else:
                    notes.append(f"[The user sent a voice message: {abs_path}]")
            if not notes:
                return user_text, []
            prefix = "\n\n".join(notes)
            _placeholder = "(The user sent a message with no text content)"
            if user_text and user_text.strip() == _placeholder:
                return prefix, []
            if user_text:
                return f"{prefix}\n\n{user_text}", []
            return prefix, []

        try:
            from tools.transcription_tools import (
                transcribe_audio,
                transcribe_audio_local_fallback,
            )
        except ModuleNotFoundError as e:
            logger.error("Transcription module unavailable: %s", e)
            unavailable_note = "[voice message could not be transcribed]"
            _placeholder = "(The user sent a message with no text content)"
            if user_text and user_text.strip() == _placeholder:
                return unavailable_note, []
            if user_text:
                return f"{unavailable_note}\n\n{user_text}", []
            return unavailable_note, []

        enriched_parts = []
        successful_transcripts: List[str] = []
        for path in audio_paths:
            try:
                logger.debug("Transcribing user voice: %s", path)
                result = await asyncio.to_thread(
                    transcribe_audio, path, None, "gateway",
                )
                if not result.get("success"):
                    fallback = await asyncio.to_thread(
                        transcribe_audio_local_fallback,
                        path,
                    )
                    if fallback.get("success"):
                        logger.info(
                            "Configured STT failed for %s; recovered with local STT",
                            path,
                        )
                        result = fallback
                if result["success"]:
                    transcript = result["transcript"]
                    # STT may return success=True with an empty/whitespace transcript (silence, cut-off);
                    # empty quotes make the agent reply to nothing and can loop, so emit a sentinel note.
                    if not (transcript or "").strip():
                        enriched_parts.append(
                            "[The user sent a voice message but it came through "
                            "empty or inaudible — speech-to-text returned no "
                            "words. Do not guess at the content; ask the user "
                            "to resend or type it out.]"
                        )
                        continue
                    successful_transcripts.append(transcript)
                    # Pass the transcript as a plain quoted line: a "The user sent a voice message..."
                    # wrapper read as a meta-instruction and made the LLM comment on voice mode instead.
                    enriched_parts.append(f'"{transcript}"')
                else:
                    error = result.get("error", "unknown error")
                    # All failure branches: one minimal neutral marker. Never mention "no STT provider",
                    # setup steps, or a DM sent — persisted in history they poison later turns (the model
                    # keeps volunteering STT-setup advice). Cause is logged for operators, not the prompt.
                    logger.info("Voice transcription failed for %s: %s", path, error)
                    from tools.credential_files import to_agent_visible_cache_path

                    agent_path = to_agent_visible_cache_path(os.path.abspath(path))
                    enriched_parts.append(
                        "[voice message could not be transcribed automatically; "
                        f"the audio is available at: {agent_path}]"
                    )
            except Exception as e:
                logger.error("Transcription error: %s", e)
                from tools.credential_files import to_agent_visible_cache_path

                agent_path = to_agent_visible_cache_path(os.path.abspath(path))
                enriched_parts.append(
                    "[voice message could not be transcribed automatically; "
                    f"the audio is available at: {agent_path}]"
                )

        if enriched_parts:
            prefix = "\n\n".join(enriched_parts)
            # Strip the empty-content placeholder from the Discord adapter
            # when we successfully transcribed the audio — it's redundant.
            _placeholder = "(The user sent a message with no text content)"
            if user_text and user_text.strip() == _placeholder:
                return prefix, successful_transcripts
            if user_text:
                return f"{prefix}\n\n{user_text}", successful_transcripts
            return prefix, successful_transcripts
        return user_text, successful_transcripts

    def _pending_event_audio_paths(self, event) -> List[str]:
        """Return STT-eligible paths from a pending voice message."""
        from gateway.run import _event_media_is_stt_input
        audio_paths: List[str] = []
        media_urls = getattr(event, "media_urls", None) or []
        for i, path in enumerate(media_urls):
            if _event_media_is_stt_input(event, i):
                audio_paths.append(path)
        return audio_paths

    async def _transcribe_pending_audio_event_once(
        self,
        event,
        user_text: Optional[str] = None,
    ) -> tuple[str | None, List[str]]:
        """Transcribe a pending audio event once and cache the result on the event.

        The interrupt monitor and the pending-drain path both need the transcript; caching keeps
        it to one STT call and one transcript echo per platform message.
        """
        if hasattr(event, "_gateway_pending_stt_text"):
            cached_text = getattr(event, "_gateway_pending_stt_text")
            cached_transcripts = getattr(event, "_gateway_pending_stt_transcripts", []) or []
            return cached_text, list(cached_transcripts)

        audio_paths = self._pending_event_audio_paths(event)
        if not audio_paths:
            return user_text if user_text is not None else (getattr(event, "text", None) or None), []

        text = user_text if user_text is not None else (getattr(event, "text", "") or "")
        enriched_text, successful_transcripts = await self._enrich_message_with_transcription(
            text,
            audio_paths,
        )
        setattr(event, "_gateway_pending_stt_text", enriched_text)
        setattr(event, "_gateway_pending_stt_transcripts", list(successful_transcripts))
        return enriched_text, successful_transcripts

    async def _echo_pending_stt_transcripts_once(
        self,
        event,
        adapter,
        source,
        transcripts: List[str],
        *,
        metadata=None,
        log_context: str = "Transcript",
    ) -> None:
        """Echo pending-event STT transcripts to the chat at most once.

        Tracked as a COUNT (not a set — identical transcripts are distinct deliveries):
        ``merge_pending_message_event`` can append a second voice note and invalidate the cache,
        and the re-run returns earlier transcripts as a prefix, so only the unsent tail is echoed.
        """
        if (
            not transcripts
            or not self._should_echo_stt_transcripts()
            or adapter is None
        ):
            return
        already_echoed = int(getattr(event, "_gateway_pending_stt_echoed", 0) or 0)
        unsent = transcripts[already_echoed:]
        setattr(event, "_gateway_pending_stt_echoed", already_echoed + len(unsent))
        for tx in unsent:
            try:
                await adapter.send(
                    source.chat_id,
                    f'🎙️ "{tx}"',
                    metadata=metadata,
                )
            except Exception as echo_exc:
                logger.debug("%s echo failed (non-fatal): %s", log_context, echo_exc)

    async def _transcribe_and_echo_pending_voice(
        self,
        event,
        adapter,
        source,
        text: str,
        *,
        log_context: str,
        metadata=_UNSET,
    ) -> tuple[str, List[str]]:
        """Transcribe a pending voice event and echo transcripts once.

        Returns ``(enriched_text, transcripts)`` for ``agent.interrupt()`` or the pending-drain
        flow; ``(text, [])`` unchanged when there is no STT-eligible media (caller owns the
        ``_build_media_placeholder`` fallback for empty ``text`` with non-audio media).
        """
        if not self._pending_event_audio_paths(event):
            return text, []
        try:
            enriched_text, transcripts = await self._transcribe_pending_audio_event_once(
                event,
                text,
            )
            echo_meta = self._thread_metadata_for_source(
                source,
                self._reply_anchor_for_event(event),
            ) if metadata is _UNSET else metadata
            await self._echo_pending_stt_transcripts_once(
                event,
                adapter,
                source,
                transcripts,
                metadata=echo_meta,
                log_context=log_context,
            )
            return enriched_text or text, transcripts
        except Exception as trans_exc:
            logger.warning("%s transcription failed: %s", log_context, trans_exc)
            return text, []
