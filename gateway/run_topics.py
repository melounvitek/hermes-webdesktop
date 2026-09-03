"""Telegram forum-topic and Discord auto-thread binding/rename methods for GatewayRunner.

Split out of ``gateway/run.py``; bound onto ``GatewayRunner`` via the MRO.
``gateway.run`` internals are imported lazily inside method bodies (import cycle),
so ``patch("gateway.run.X")`` keeps intercepting them at call time.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Tuple

from agent.compaction_display import project_compaction_message_for_display
from agent.i18n import t
from gateway.config import Platform
from gateway.platforms.base import MessageEvent, _prefix_within_utf16_limit, utf16_len
from gateway.session import SessionSource

if TYPE_CHECKING:  # string annotations only; never imported at runtime (cycle)
    from gateway.run import GatewayRunner, TurnRunner  # noqa: F401

# Log-record parity with the origin module.
logger = logging.getLogger("gateway.run")

_TOPIC_RESTORE_STEPS = (
    "1. Create or open a topic. To create a new one, open All Messages and send any message there.",
    "2. Send /topic <session-id> inside that topic.",
)


class GatewayTopicThreadsMixin:
    """Telegram forum-topic and Discord auto-thread binding/rename methods for GatewayRunner."""

    # ── Telegram topic mode: predicates and keys ────────────────────────────────────────────

    @staticmethod
    def _telegram_topic_profile_name(source: SessionSource) -> str:
        """Profile namespace for Telegram topic-mode rows.

        Use the profile stamped on the routed event (``source.profile``), never the process-global
        active profile — under multiplex that mis-attributes topic state across bots sharing state.db.
        """
        name = str(getattr(source, "profile", None) or "").strip()
        return name if name else "default"

    def _sync_session_db(self):
        """The sync SessionDB handle, or None. Only for callers that provably run off-loop
        (asyncio.to_thread / the run_sync executor)."""
        session_db = getattr(self, "_session_db", None)
        return None if session_db is None else getattr(session_db, "_db", session_db)

    @staticmethod
    def _is_telegram_dm(source: SessionSource) -> bool:
        return source.platform == Platform.TELEGRAM and source.chat_type == "dm"

    def _telegram_topic_mode_enabled(self, source: SessionSource) -> bool:
        """Return whether Telegram DM topic mode is active for this chat."""
        if not self._is_telegram_dm(source):
            return False
        session_db = self._sync_session_db()
        if session_db is None:
            return False
        try:
            raw = session_db.is_telegram_topic_mode_enabled(
                chat_id=str(source.chat_id),
                user_id=str(source.user_id),
                profile_name=self._telegram_topic_profile_name(source),
            )
        except Exception:
            logger.debug("Failed to read Telegram topic mode state", exc_info=True)
            return False
        # Only a real True enables topic mode; anything else (including MagicMock from test
        # fixtures that didn't opt in) means off for this chat.
        return raw is True

    def _is_telegram_topic_root_lobby(self, source: SessionSource) -> bool:
        """True for the main Telegram DM (or General topic) when topic mode has made it a lobby."""
        if not self._is_telegram_dm(source) or not self._telegram_topic_mode_enabled(source):
            return False
        return str(source.thread_id or "") in self._TELEGRAM_GENERAL_TOPIC_IDS

    def _is_telegram_topic_lane(self, source: SessionSource) -> bool:
        """True for a user-created Telegram private-chat topic lane."""
        if not self._is_telegram_dm(source) or not self._telegram_topic_mode_enabled(source):
            return False
        tid = str(source.thread_id or "")
        return bool(tid) and tid not in self._TELEGRAM_GENERAL_TOPIC_IDS

    def _telegram_topic_cooldown_key(self, source: SessionSource) -> Optional[str]:
        """Cooldown key (profile, chat_id): profiles sharing a Telegram private chat_id under
        multiplex must not suppress each other's lobby reminders / capability hints."""
        chat_id = str(source.chat_id or "")
        if not chat_id:
            return None
        return f"{self._telegram_topic_profile_name(source)}:{chat_id}"

    def _telegram_cooldown_elapsed(self, source: SessionSource, attr: str, cooldown_s: float) -> bool:
        """Per-(profile, chat) debounce: True (and stamp now) when the window has elapsed."""
        if not hasattr(self, attr):
            setattr(self, attr, {})
        key = self._telegram_topic_cooldown_key(source)
        if not key:
            return True
        stamps = getattr(self, attr)
        now = time.monotonic()
        if now - stamps.get(key, 0.0) < cooldown_s:
            return False
        stamps[key] = now
        return True

    def _should_send_telegram_lobby_reminder(self, source: SessionSource) -> bool:
        """Rate-limit root-DM lobby reminders to one per cooldown window, not one per prompt typed."""
        return self._telegram_cooldown_elapsed(
            source, "_telegram_lobby_reminder_ts", self._TELEGRAM_LOBBY_REMINDER_COOLDOWN_S,
        )

    def _should_send_telegram_capability_hint(self, source: SessionSource) -> bool:
        """Rate-limit the BotFather Threads Settings screenshot: repeated /topic while Threads
        Settings are still off must not re-upload it every time."""
        return self._telegram_cooldown_elapsed(
            source, "_telegram_capability_hint_ts", self._TELEGRAM_CAPABILITY_HINT_COOLDOWN_S,
        )

    # ── Telegram topic mode: user-facing text ───────────────────────────────────────────────

    def _telegram_topic_root_lobby_message(self) -> str:
        return (
            "This main chat is reserved for system commands.\n\n"
            "To start a new Hermes chat, open the All Messages topic at the top "
            "of this bot interface and send any message there. Telegram will "
            "create a new topic for that message; each topic works as an "
            "independent Hermes session."
        )

    def _telegram_topic_root_new_message(self) -> str:
        return (
            "To start a new parallel Hermes chat, open the All Messages topic "
            "at the top of this bot interface and send any message there. "
            "Telegram will create a new topic for it.\n\n"
            "Each topic is an independent Hermes session. Use /new inside an "
            "existing topic only if you want to replace that topic's current session."
        )

    def _telegram_topic_new_header(self, source: SessionSource) -> Optional[str]:
        if not self._is_telegram_topic_lane(source):
            return None
        return (
            "Started a new Hermes session in this topic.\n\n"
            "Tip: for parallel work, open All Messages and send a message there "
            "to create a separate topic instead of using /new here. /new replaces "
            "the session attached to the current topic."
        )

    def _telegram_topic_help_text(self) -> str:
        return (
            "/topic — enable multi-session DM mode (one bot, many parallel chats)\n"
            "\n"
            "Usage:\n"
            "  /topic             Enable topic mode, or show status if already on\n"
            "  /topic help        Show this message\n"
            "  /topic off         Disable topic mode and clear topic bindings\n"
            "  /topic <id>        Inside a topic: restore a previous session by ID\n"
            "\n"
            "How it works:\n"
            "1. Run /topic once in this DM — Hermes checks BotFather Threads\n"
            "   Settings are enabled and flips on multi-session mode.\n"
            "2. Tap All Messages at the top of the bot and send any message.\n"
            "   Telegram creates a new topic for that message; each topic is\n"
            "   an independent Hermes session (fresh history, fresh context).\n"
            "3. The root DM becomes a system lobby — send /topic, /status,\n"
            "   /help, /usage there. Normal prompts go in a topic.\n"
            "4. /new inside a topic resets just that topic's session.\n"
            "5. /topic <id> inside a topic restores an old session into it."
        )

    # ── Telegram topic bindings ─────────────────────────────────────────────────────────────

    def _record_telegram_topic_binding(self, source: SessionSource, session_entry) -> None:
        """Persist the Telegram topic -> Hermes session binding for topic lanes (off-loop)."""
        session_db = self._sync_session_db()
        if session_db is None or not source.chat_id or not source.thread_id:
            return
        session_db.bind_telegram_topic(
            chat_id=str(source.chat_id),
            thread_id=str(source.thread_id),
            user_id=str(source.user_id or ""),
            session_key=session_entry.session_key,
            session_id=session_entry.session_id,
            profile_name=self._telegram_topic_profile_name(source),
        )

    def _sync_telegram_topic_binding(self, source: SessionSource, session_entry, *, reason: str) -> None:
        """Update the topic binding to point at ``session_entry.session_id``.

        Topic lanes persist (chat_id, thread_id) -> session_id so reopening a topic resumes the
        right session. When compression rotates the id mid-turn a stale binding reloads the
        oversized parent next message, retriggering preflight compression — sometimes in a loop.
        """
        if not self._is_telegram_topic_lane(source):
            return
        try:
            self._record_telegram_topic_binding(source, session_entry)
        except Exception:
            logger.debug("telegram topic binding refresh failed (%s)", reason, exc_info=True)

    def _recover_telegram_topic_thread_id(self, source: SessionSource) -> Optional[str]:
        """Pin DM-topic routing to the user's last-active topic.

        Telegram can omit ``message_thread_id`` or surface General (``1``) for topic-mode DM
        replies; in those lobby-shaped cases keep the conversation on the user's most-recent bound
        topic. Do not rewrite a non-lobby, previously-unbound thread id: a brand-new DM topic is
        also "unknown" until its first inbound message is recorded, and rewriting would send its
        answer into an older lane. Returns None to leave the source alone.
        """
        if (
            not self._is_telegram_dm(source)
            or not source.chat_id
            or not source.user_id
            or not self._telegram_topic_mode_enabled(source)
        ):
            return None
        inbound = str(source.thread_id or "")
        if inbound and inbound not in self._TELEGRAM_GENERAL_TOPIC_IDS:
            return None
        session_db = self._sync_session_db()
        if session_db is None:
            return None
        try:
            bindings = session_db.list_telegram_topic_bindings_for_chat(
                chat_id=str(source.chat_id),
                profile_name=self._telegram_topic_profile_name(source),
            )
        except Exception:
            logger.debug("topic-recover: read failed", exc_info=True)
            return None
        user_id = str(source.user_id)
        for b in bindings or ():  # newest-first
            if str(b.get("user_id") or "") == user_id:
                recovered = str(b.get("thread_id") or "")
                return recovered if recovered and recovered != inbound else None
        return None

    # ── Telegram topic mode: /topic activation helpers ──────────────────────────────────────

    async def _get_telegram_topic_capabilities(self, source: SessionSource) -> dict:
        """Read Telegram private-topic capability flags via Bot API getMe."""
        bot = getattr(self._adapter_for_source(source), "_bot", None)
        if bot is None or not hasattr(bot, "get_me"):
            return {"checked": False}
        try:
            me = await bot.get_me()
        except Exception:
            logger.debug("Failed to fetch Telegram getMe topic capabilities", exc_info=True)
            return {"checked": False}

        def _field(name: str):
            if hasattr(me, name):
                return getattr(me, name)
            api_kwargs = getattr(me, "api_kwargs", None)
            if isinstance(api_kwargs, dict) and name in api_kwargs:
                return api_kwargs.get(name)
            return me.get(name) if isinstance(me, dict) else None

        return {
            "checked": True,
            "has_topics_enabled": _field("has_topics_enabled"),
            "allows_users_to_create_topics": _field("allows_users_to_create_topics"),
        }

    async def _ensure_telegram_system_topic(self, source: SessionSource) -> None:
        """Create/pin the managed System topic after /topic activation when possible."""
        adapter = self._adapter_for_source(source)
        if adapter is None or not source.chat_id:
            return
        thread_id = None
        create_topic = getattr(adapter, "_create_dm_topic", None)
        if callable(create_topic):
            try:
                thread_id = await create_topic(int(source.chat_id), "System")
            except Exception:
                logger.debug("Failed to create Telegram System topic", exc_info=True)
        if not thread_id:
            return
        message_id = None
        try:
            send_result = await adapter.send(
                source.chat_id,
                "System topic for Hermes commands and status.",
                metadata={"thread_id": str(thread_id)},
            )
            message_id = getattr(send_result, "message_id", None)
        except Exception:
            logger.debug("Failed to send Telegram System topic intro", exc_info=True)
        if not message_id:
            return
        bot = getattr(adapter, "_bot", None)
        if bot is None or not hasattr(bot, "pin_chat_message"):
            return
        try:
            await bot.pin_chat_message(
                chat_id=int(source.chat_id),
                message_id=int(message_id),
                disable_notification=True,
            )
        except Exception:
            logger.debug("Failed to pin Telegram System topic intro", exc_info=True)

    async def _send_telegram_topic_setup_image(self, source: SessionSource) -> None:
        """Send the bundled BotFather Threads Settings screenshot when available."""
        adapter = self._adapter_for_source(source)
        if adapter is None or not source.chat_id or not hasattr(adapter, "send_image_file"):
            return
        image_path = Path(__file__).resolve().parent / "assets" / "telegram-botfather-threads-settings.jpg"
        if not image_path.exists():
            return
        try:
            await adapter.send_image_file(
                chat_id=source.chat_id,
                image_path=str(image_path),
                caption="BotFather → Bot Settings → Threads Settings",
                metadata={"thread_id": str(source.thread_id)} if source.thread_id else None,
            )
        except Exception:
            logger.debug("Failed to send Telegram topic setup image", exc_info=True)

    # ── title sanitizers ────────────────────────────────────────────────────────────────────

    def _sanitize_telegram_topic_title(self, title: str) -> str:
        """Bot API-safe forum topic name: names are 1-128 chars; keep room for multi-byte titles."""
        cleaned = re.sub(r"\s+", " ", str(title or "")).strip()
        if not cleaned:
            return "Hermes Chat"
        if len(cleaned) > 120:
            cleaned = cleaned[:117].rstrip() + "..."
        return cleaned

    def _sanitize_discord_thread_title(self, title: str) -> str:
        """Discord-safe thread title: the 100-char cap is measured in UTF-16 code units (emoji count
        double), so truncate with the UTF-16 helpers rather than Python code-point slices."""
        cleaned = re.sub(r"\s+", " ", str(title or "")).strip()
        if not cleaned:
            return "Hermes Chat"
        if utf16_len(cleaned) > 80:
            cleaned = _prefix_within_utf16_limit(cleaned, 77).rstrip() + "..."
        return cleaned

    # ── Discord auto-thread lanes ───────────────────────────────────────────────────────────

    def _is_discord_auto_thread_lane(self, source: SessionSource) -> bool:
        """Return True only for Discord threads Hermes just auto-created."""
        return (
            source.platform == Platform.DISCORD
            and source.chat_type == "thread"
            and bool(getattr(source, "auto_thread_created", False))
            and bool(source.thread_id)
            and bool(getattr(source, "auto_thread_initial_name", None))
        )

    def _is_relay_discord_channel_lane(self, source: SessionSource) -> bool:
        """Shape-only check: a relay-delivered Discord CHANNEL event whose reply the connector MAY
        auto-thread (title-turn registration gate).

        Deliberately does NOT consult the send-result cache: at registration time (before delivery)
        the feedback can't exist yet. The rename lane polls the cache at fire time instead."""
        return (
            source.platform == Platform.DISCORD
            and bool(source.chat_id)
            and not source.thread_id
            and source.chat_type in ("group", "channel")
            and getattr(source, "delivered_via_upstream_relay", False) is True
        )

    def _relay_auto_thread_info(self, source: SessionSource) -> Optional[Tuple[str, str]]:
        """(thread_id, initial_name) when the RELAY connector auto-threaded our reply to this
        source's chat — the title-turn sibling of _is_discord_auto_thread_lane.

        The marker check only matches events ARRIVING IN an auto-created thread (turn 2+); the
        auto-title fires on the FIRST exchange, whose source is the PARENT channel event with no
        markers. Preferred: the connector's ``prospective_thread_id`` stamp (anchor message id ==
        the thread it will create) — per-message, so it names the EXACT thread even when several
        auto-threads spawn from one channel; the connector's created-name guard enforces
        no-clobber. Fallback: the per-chat send-result thread_id/auto_thread_name cache (older
        connectors), which only ever renamed the FIRST thread.
        """
        from gateway.run import _as_thread_info
        if source.platform != Platform.DISCORD or not source.chat_id:
            return None
        if not getattr(source, "delivered_via_upstream_relay", False):
            return None
        prospective = getattr(source, "prospective_thread_id", None)
        if prospective:
            # Deterministic per-thread identity; the empty initial-name marker signals the caller
            # to rely on the connector-side no-clobber guard.
            return (str(prospective), "")
        info_fn = getattr(self._adapter_for_source(source), "auto_thread_info_for_chat", None)
        if not callable(info_fn):
            return None
        try:
            return _as_thread_info(info_fn(str(source.chat_id)))
        except Exception:
            return None

    async def _await_relay_auto_thread_info(self, source: SessionSource) -> Optional[Tuple[str, str]]:
        """``_relay_auto_thread_info``, waited out until this turn delivers.

        The legacy send-result path can only answer once the reply is sent, and the caller asks
        at title time — one turn early. The adapter answers on the send either way, so the
        timeout is only a backstop for a turn that never sends at all; the turn's own inactivity
        limit is exactly how long that turn could still be alive.
        """
        from gateway.run import _as_thread_info, _float_env
        # The connector-stamped prospective id is known at ingest, so most sessions answer here.
        known = self._relay_auto_thread_info(source)
        if known is not None:
            return known
        wait_fn = getattr(self._adapter_for_source(source), "wait_for_auto_thread_info", None)
        if not callable(wait_fn) or not source.chat_id:
            return None
        # 0 means the operator disabled the turn limit; the backstop still needs one.
        timeout = _float_env("HERMES_AGENT_TIMEOUT", 1800) or 1800
        try:
            return _as_thread_info(await wait_fn(str(source.chat_id), timeout))
        except Exception:
            return None

    async def _rename_discord_auto_thread_for_session_title(
        self,
        source: SessionSource,
        session_id: str,
        title: str,
        relay_info: Optional[Tuple[str, str]] = None,
    ) -> None:
        """Best-effort semantic rename of a newly auto-created Discord thread.

        ``relay_info`` is the (thread_id, initial_name) pair from the relay connector's send-
        result feedback — supplied on the title turn, where the source is the parent-channel
        event and carries no auto-thread markers (see _relay_auto_thread_info).
        """
        if relay_info is None and not await asyncio.to_thread(self._is_discord_auto_thread_lane, source):
            # Relay title turn with no feedback captured at schedule time: the title comes off the
            # user's opening message, so it beats the delivery that produces the connector's
            # send-result feedback by the whole length of the turn.
            if not self._is_relay_discord_channel_lane(source):
                return
            relay_info = await self._await_relay_auto_thread_info(source)
            if relay_info is None:
                # True miss: the connector did not auto-thread this reply.
                return
        adapter = self._adapter_for_source(source) if getattr(self, "adapters", None) else None
        rename_thread = getattr(adapter, "rename_thread", None)
        if rename_thread is None:
            return
        relay = relay_info is not None
        target_thread_id = relay_info[0] if relay else str(source.thread_id)
        thread_name = self._sanitize_discord_thread_title(title)
        if relay:
            # Ask the CONNECTOR to enforce the no-clobber guard from its own created-name memory —
            # the gateway can't reproduce the thread's initial name byte-for-byte (normalization
            # drift silently declined every rename). The connector's egress guard resolves the
            # owning tenant from caches keyed by the PARENT channel chat_id (learned at inbound),
            # not the thread id, so pass the parent channel id or the lookup misses and it declines.
            rename_kwargs = {
                "prefer_connector_created": True,
                "parent_chat_id": str(source.chat_id) if source.chat_id else None,
            }
        else:
            # Native lane: the source IS the thread (direct Discord API); guard on the initial name.
            rename_kwargs = {"only_if_current_name": getattr(source, "auto_thread_initial_name", None)}
        logger.info(
            "discord auto-thread rename: thread=%s lane=%s new_title=%r",
            target_thread_id, "relay" if relay else "native", thread_name,
        )
        try:
            renamed = await rename_thread(target_thread_id, thread_name, **rename_kwargs)
            logger.info(
                "discord auto-thread rename result: thread=%s applied=%s",
                target_thread_id, bool(renamed),
            )
        except TypeError:
            logger.warning(
                "Discord semantic thread rename raised TypeError (adapter=%s)",
                type(adapter).__name__, exc_info=True,
            )
        except Exception:
            logger.debug("Failed to rename Discord auto-thread for generated session title", exc_info=True)

    # ── title-thread → loop rename scheduling ───────────────────────────────────────────────

    def _schedule_rename_from_title_thread(self, source: SessionSource, make_coro, label: str) -> None:
        """Schedule a best-effort rename coroutine onto the gateway loop from the auto-title thread.

        The source is copied so the background thread never shares the live dataclass with the
        loop; failures are logged at debug and never propagate."""
        from gateway.run import safe_schedule_threadsafe
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = getattr(self, "_gateway_loop", None)
        if loop is None or loop.is_closed():
            return
        try:
            copied_source = dataclasses.replace(source)
        except Exception:
            copied_source = source
        future = safe_schedule_threadsafe(
            make_coro(copied_source), loop, logger=logger, log_message=f"{label} failed to schedule",
        )
        if future is None:
            return

        def _log_rename_failure(fut) -> None:
            try:
                fut.result()
            except Exception:
                logger.debug("%s failed", label, exc_info=True)

        future.add_done_callback(_log_rename_failure)

    def _schedule_discord_semantic_thread_rename(self, source: SessionSource, session_id: str, title: str) -> None:
        """Schedule Discord auto-thread rename from the auto-title background thread."""
        relay_info = None
        if not title:
            return
        if not self._is_discord_auto_thread_lane(source):
            # Relay title turn: the source is the PARENT channel event (thread didn't exist at
            # ingest). The auto-title races the delivery that fills the send-result cache, so a
            # miss HERE is not a verdict: schedule whenever the SHAPE matches; the async rename
            # lane polls the cache (bounded wait) and no-ops on a true miss.
            relay_info = self._relay_auto_thread_info(source)
            if relay_info is None and not self._is_relay_discord_channel_lane(source):
                return
        self._schedule_rename_from_title_thread(
            source,
            lambda copied: self._rename_discord_auto_thread_for_session_title(
                copied, session_id, title, relay_info=relay_info
            ),
            "Discord semantic thread rename",
        )

    def _schedule_telegram_topic_title_rename(self, source: SessionSource, session_id: str, title: str) -> None:
        """Schedule a topic rename from the auto-title background thread."""
        if not title or not self._is_telegram_topic_lane(source):
            return
        if self._telegram_topic_auto_rename_disabled(source):
            return
        self._schedule_rename_from_title_thread(
            source,
            lambda copied: self._rename_telegram_topic_for_session_title(copied, session_id, title),
            "Telegram topic title rename",
        )

    def _telegram_topic_auto_rename_disabled(self, source: SessionSource) -> bool:
        """``gateway.platforms.telegram.extra.disable_topic_auto_rename``; default False (auto-rename on)."""
        config = getattr(self, "config", None)
        platform_cfg = config.platforms.get(source.platform) if config and getattr(config, "platforms", None) else None
        if platform_cfg is None:
            return False
        value = (getattr(platform_cfg, "extra", None) or {}).get("disable_topic_auto_rename")
        if value is None:
            return False
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    async def _rename_telegram_topic_for_session_title(self, source: SessionSource, session_id: str, title: str) -> None:
        """Best-effort rename of a Telegram DM topic when Hermes auto-titles a session."""
        if not await asyncio.to_thread(self._is_telegram_topic_lane, source) or not source.chat_id or not source.thread_id:
            return
        # Operator kill-switch, e.g. user-managed topics (ad-hoc Threaded Mode) that auto-rename
        # would keep overwriting.
        if self._telegram_topic_auto_rename_disabled(source):
            return
        # Skip operator-declared topics (extra.dm_topics): fixed names chosen by the operator;
        # auto-renaming would silently mutate operator config. Check the class, not the instance —
        # getattr() on a MagicMock auto-creates attributes, so every test double would match.
        adapter = self._adapter_for_source(source)
        if adapter is not None:
            get_info = getattr(type(adapter), "_get_dm_topic_info", None)
            if callable(get_info):
                try:
                    operator_topic = get_info(adapter, str(source.chat_id), str(source.thread_id))
                except Exception:
                    operator_topic = None
                # Only dict-shaped returns count; a bare MagicMock or other sentinel shouldn't.
                if isinstance(operator_topic, dict):
                    return
        session_db = getattr(self, "_session_db", None)
        if session_db is not None:
            try:
                binding = await session_db.get_telegram_topic_binding(
                    chat_id=str(source.chat_id),
                    thread_id=str(source.thread_id),
                    profile_name=self._telegram_topic_profile_name(source),
                )
                if binding and str(binding.get("session_id") or "") != str(session_id):
                    return
            except Exception:
                logger.debug("Failed to verify Telegram topic binding before rename", exc_info=True)
                return
        if adapter is None:
            return
        topic_name = self._sanitize_telegram_topic_title(title)
        try:
            rename_topic = getattr(adapter, "rename_dm_topic", None)
            if rename_topic is not None:
                await rename_topic(chat_id=str(source.chat_id), thread_id=str(source.thread_id), name=topic_name)
                return
            bot = getattr(adapter, "_bot", None)
            edit_forum_topic = getattr(bot, "edit_forum_topic", None) or getattr(bot, "editForumTopic", None)
            if edit_forum_topic is None:
                return
            try:
                await edit_forum_topic(
                    chat_id=int(source.chat_id), message_thread_id=int(source.thread_id), name=topic_name,
                )
            except (TypeError, ValueError):
                await edit_forum_topic(
                    chat_id=source.chat_id, message_thread_id=source.thread_id, name=topic_name,
                )
        except Exception:
            logger.debug("Failed to rename Telegram topic for auto-generated title", exc_info=True)

    # ── /topic command bodies ───────────────────────────────────────────────────────────────

    async def _disable_telegram_topic_mode_for_chat(self, source: SessionSource) -> str:
        """Cleanly disable topic mode for a chat via /topic off."""
        if not self._session_db:
            from hermes_state import format_session_db_unavailable
            return format_session_db_unavailable(prefix=t("gateway.shared.session_db_unavailable_prefix"))
        chat_id = str(source.chat_id or "")
        if not chat_id:
            return "Could not determine chat ID."
        profile_name = self._telegram_topic_profile_name(source)
        try:
            currently_enabled = await self._session_db.is_telegram_topic_mode_enabled(
                chat_id=chat_id, user_id=str(source.user_id or ""), profile_name=profile_name,
            )
        except Exception:
            currently_enabled = False
        if not currently_enabled:
            return "Multi-session topic mode is not currently enabled for this chat."
        try:
            await self._session_db.disable_telegram_topic_mode(chat_id=chat_id, profile_name=profile_name)
        except Exception as exc:
            logger.exception("Failed to disable Telegram topic mode")
            return f"Failed to disable topic mode: {exc}"
        # Reset per-profile+chat debounce state so the next activation doesn't see a stale cooldown.
        cooldown_key = self._telegram_topic_cooldown_key(source)
        if cooldown_key:
            for attr in ("_telegram_lobby_reminder_ts", "_telegram_capability_hint_ts"):
                store = getattr(self, attr, None)
                if isinstance(store, dict):
                    store.pop(cooldown_key, None)
        return (
            "Multi-session topic mode is now OFF for this chat.\n\n"
            "Existing topics in Telegram aren't removed — they'll just stop "
            "being gated as independent sessions. The root DM works as a "
            "normal Hermes chat again. Run /topic to re-enable later."
        )

    async def _telegram_topic_root_status_message(self, source: SessionSource) -> str:
        lines = [
            "Telegram multi-session topics are enabled.",
            "",
            "To create a new Hermes chat, open All Messages at the top of this "
            "bot interface and send any message there. Telegram will create a "
            "new topic for it.",
            "",
        ]
        try:
            sessions = await self._session_db.list_unlinked_telegram_sessions_for_user(
                chat_id=str(source.chat_id),
                user_id=str(source.user_id),
                profile_name=self._telegram_topic_profile_name(source),
                limit=10,
            )
        except Exception:
            logger.debug("Failed to list unlinked Telegram sessions", exc_info=True)
            sessions = []
        if sessions:
            lines.append("Previous unlinked sessions:")
            for session in sessions:
                line = f"- {session.get('title') or 'Untitled session'} — `{session.get('id') or ''}`"
                preview = str(session.get("preview") or "").strip()
                if preview:
                    line += f" — {preview}"
                lines.append(line)
            lines.extend([
                "",
                "To restore one:",
                *_TOPIC_RESTORE_STEPS,
                f"Example: Send /topic {sessions[0].get('id')} inside a topic.",
            ])
        else:
            lines.extend([
                "No previous unlinked Telegram sessions found.",
                "",
                "To restore a previous session later:",
                *_TOPIC_RESTORE_STEPS,
            ])
        return "\n".join(lines)

    async def _restore_telegram_topic_session(self, event: MessageEvent, raw_session_id: str) -> str:
        """Restore an existing Telegram-owned Hermes session into this topic."""
        source = event.source
        db = self._session_db
        session_id = await db.resolve_session_id(raw_session_id.strip())
        session = await db.get_session(session_id) if session_id else None
        if not session:
            return f"Session not found: {raw_session_id.strip()}"
        if str(session.get("source") or "") != "telegram":
            return "That session is not a Telegram session and cannot be restored into this topic."
        if str(session.get("user_id") or "") != str(source.user_id):
            return "That session does not belong to this Telegram user."
        linked = await db.is_telegram_session_linked_to_topic(session_id=session_id)
        topic_profile = self._telegram_topic_profile_name(source)
        current_binding = await db.get_telegram_topic_binding(
            chat_id=str(source.chat_id), thread_id=str(source.thread_id), profile_name=topic_profile,
        )
        already_linked = "That session is already linked to another Telegram topic."
        if linked and (not current_binding or current_binding.get("session_id") != session_id):
            return already_linked
        try:
            await db.bind_telegram_topic(
                chat_id=str(source.chat_id),
                thread_id=str(source.thread_id),
                user_id=str(source.user_id),
                session_key=self._session_key_for_source(source),
                session_id=session_id,
                managed_mode="restored",
                profile_name=topic_profile,
            )
        except ValueError as exc:
            if "already linked" in str(exc):
                return already_linked
            raise
        title = await db.get_session_title(session_id) or session_id
        last_assistant = None
        try:
            for message in reversed(await db.get_messages(session_id)):
                if message.get("role") != "assistant":
                    continue
                projected = project_compaction_message_for_display(message)
                if projected is not None and projected.get("content"):
                    last_assistant = str(projected.get("content"))
                    break
        except Exception:
            last_assistant = None
        response = f"Session restored: {title}"
        if last_assistant:
            response += f"\n\nLast Hermes message:\n{last_assistant}"
        return response
