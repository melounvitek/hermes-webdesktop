"""Voice-channel / auto-TTS methods for GatewayRunner.

Split out of ``gateway/run.py``; bound onto ``GatewayRunner`` via the MRO.
``gateway.run`` internals are imported lazily inside method bodies (import cycle),
so ``patch("gateway.run.X")`` keeps intercepting them at call time.
"""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import os
import re
import sys
import time
from contextlib import suppress
from difflib import SequenceMatcher
from types import SimpleNamespace
from typing import TYPE_CHECKING, Dict, List, Optional

from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType, build_auto_tts_output_path
from gateway.session import SessionSource

if TYPE_CHECKING:  # string annotations only; never imported at runtime (cycle)
    from gateway.run import GatewayRunner, TurnRunner  # noqa: F401

# Log-record parity with the origin module.
logger = logging.getLogger("gateway.run")

# Adapter-side per-chat auto-TTS override sets (``/voice off`` vs explicit ``/voice on``/``tts``).
_OFF_SET, _ON_SET = "_auto_tts_disabled_chats", "_auto_tts_enabled_chats"
_VOICE_MODES = {"off", "voice_only", "all"}


class GatewayVoiceMixin:
    """Voice-channel / auto-TTS methods for GatewayRunner."""

    def _voice_key(self, platform: Platform, chat_id: str, profile: Optional[str] = None) -> str:
        """``<profile>:<platform>:<chat_id>`` under multiplexing (profile whose bot speaks — else
        two bots in one channel share a key and one ``/voice`` flips the other's); the default
        profile keeps ``<platform>:<chat_id>`` so persisted state stays valid."""
        base = f"{platform.value}:{chat_id}"
        profile = profile.strip() if isinstance(profile, str) else ""
        return base if not profile or profile == "default" else f"{profile}:{base}"

    def _voice_key_for_source(self, source: SessionSource) -> str:
        """Voice mode belongs to the (bot, chat) pair: namespace is the profile that OWNS the
        receiving adapter, not the routed profile."""
        profile = self._adapter_profile_for_source(source)
        return self._voice_key(source.platform, source.chat_id, profile=profile)

    def _bind_voice_input_callback(self, adapter) -> None:
        """Route voice transcripts back through the adapter that captured them."""
        if hasattr(adapter, "_voice_input_callback"):
            adapter._voice_input_callback = functools.partial(
                self._handle_voice_channel_input, adapter=adapter
            )

    def _load_voice_modes(self) -> Dict[str, str]:
        try:
            data = json.loads(self._VOICE_MODE_PATH.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return {}
        if not isinstance(data, dict):
            return {}
        items = {str(k): m for k, m in data.items() if m in _VOICE_MODES}
        for chat_id in (k for k in items if ":" not in k):  # legacy unprefixed key: warn and skip
            logger.warning(
                "Skipping legacy unprefixed voice mode key %r during migration. "
                "Re-enable voice mode on that chat to rebuild the prefixed key.", chat_id,
            )
        return {k: m for k, m in items.items() if ":" in k}

    def _save_voice_modes(self) -> None:
        try:
            self._VOICE_MODE_PATH.parent.mkdir(parents=True, exist_ok=True)
            payload = json.dumps(self._voice_mode, indent=2)
            self._VOICE_MODE_PATH.write_text(payload, encoding="utf-8")
        except OSError as e:
            logger.warning("Failed to save voice modes: %s", e)

    @staticmethod
    def _toggle_adapter_auto_tts_set(adapter, chat_id: str, on: bool, *, enable: bool) -> None:
        """Add/discard ``chat_id`` in the adapter's enabled (``enable=True``) or disabled set;
        adding also clears the other set (``/voice off`` and ``/voice on``/``tts`` override each
        other)."""
        add_to, clear_from = (_ON_SET, _OFF_SET) if enable else (_OFF_SET, _ON_SET)
        target = getattr(adapter, add_to, None)
        if not isinstance(target, set):
            return
        if not on:
            target.discard(chat_id)
            return
        target.add(chat_id)
        if isinstance(other := getattr(adapter, clear_from, None), set):
            other.discard(chat_id)

    def _set_adapter_auto_tts_disabled(self, adapter, chat_id: str, disabled: bool) -> None:
        """Update an adapter's in-memory auto-TTS suppression set if present."""
        self._toggle_adapter_auto_tts_set(adapter, chat_id, disabled, enable=False)

    def _set_adapter_auto_tts_enabled(self, adapter, chat_id: str, enabled: bool) -> None:
        """Update an adapter's per-chat auto-TTS opt-in set (works with ``voice.auto_tts`` off)."""
        self._toggle_adapter_auto_tts_set(adapter, chat_id, enabled, enable=True)

    def _sync_voice_mode_state_to_adapter(self, adapter) -> None:
        """Restore persisted /voice state into a live adapter: ``_auto_tts_default`` from
        ``voice.auto_tts``; enabled (``voice_only``/``all``) and disabled (``off``) chat sets from
        ``self._voice_mode``."""
        platform = getattr(adapter, "platform", None)
        if not isinstance(platform, Platform):
            return
        chat_sets = [
            (chats, modes)
            for name, modes in ((_OFF_SET, {"off"}), (_ON_SET, {"voice_only", "all"}))
            if isinstance(chats := getattr(adapter, name, None), set)
        ]
        if not chat_sets:
            return
        # Lazy import: no module-level dep from gateway -> hermes_cli.
        try:
            from hermes_cli.config import load_config
            auto_tts_default = bool((load_config().get("voice") or {}).get("auto_tts", False))
        except Exception:
            auto_tts_default = False
        if hasattr(adapter, "_auto_tts_default"):
            adapter._auto_tts_default = auto_tts_default
        prefix = self._voice_key(platform, "", profile=getattr(adapter, "_owner_profile", None))
        for chats, modes in chat_sets:
            chats.clear()
            chats.update(
                key[len(prefix):] for key, mode in self._voice_mode.items()
                if mode in modes and key.startswith(prefix)
            )

    @staticmethod
    def _get_guild_id(event: MessageEvent) -> Optional[int]:
        """Extract Discord guild_id from the raw message object."""
        raw = getattr(event, "raw_message", None)
        if getattr(raw, "guild_id", None):  # slash command interaction
            return int(raw.guild_id)
        return raw.guild.id if getattr(raw, "guild", None) else None  # regular message

    async def _handle_voice_channel_join(self, event: MessageEvent) -> str:
        """Join the user's current Discord voice channel."""
        adapter = self._adapter_for_source(event.source)
        if not hasattr(adapter, "join_voice_channel"):
            return "Voice channels are not supported on this platform."
        guild_id = self._get_guild_id(event)
        if not guild_id:
            return "This command only works in a Discord server."
        voice_channel = await adapter.get_user_voice_channel(guild_id, event.source.user_id)
        if not voice_channel:
            return "You need to be in a voice channel first."
        # Wire callbacks BEFORE join so voice input arriving right after connection is not lost.
        self._bind_voice_input_callback(adapter)
        voice_profile = self._adapter_profile_for_source(event.source)
        if hasattr(adapter, "_on_voice_disconnect"):
            adapter._on_voice_disconnect = functools.partial(
                self._handle_voice_timeout_cleanup, adapter=adapter
            )
        # Let the adapter's inactivity timer see the live voice-reply mode so it doesn't
        # disconnect a deliberately text-only (/voice off) session.
        if hasattr(adapter, "_voice_mode_getter"):
            adapter._voice_mode_getter = lambda chat_id: self._voice_mode.get(
                self._voice_key(Platform.DISCORD, str(chat_id), profile=voice_profile), "off"
            )
        try:
            success = await adapter.join_voice_channel(voice_channel)
        except Exception as e:
            logger.warning("Failed to join voice channel: %s", e)
            adapter._voice_input_callback = None
            if not any(tok in str(e).lower() for tok in ("pynacl", "nacl", "davey")):
                return f"Failed to join voice channel: {e}"
            return ("Voice dependencies are missing (PyNaCl / davey). "
                    f"Install with: `{sys.executable} -m pip install PyNaCl`")
        if not success:
            adapter._voice_input_callback = None
            return "Failed to join voice channel. Check bot permissions (Connect + Speak)."
        adapter._voice_text_channels[guild_id] = int(event.source.chat_id)
        if hasattr(adapter, "_voice_sources"):
            adapter._voice_sources[guild_id] = event.source.to_dict()
        self._set_voice_mode(self._voice_key_for_source(event.source), "all")
        self._set_adapter_auto_tts_enabled(adapter, event.source.chat_id, enabled=True)
        return (
            f"Joined voice channel **{voice_channel.name}**.\n"
            f"I'll speak my replies and listen to you. Use /voice leave to disconnect."
        )

    async def _handle_voice_channel_leave(self, event: MessageEvent) -> str:
        """Leave the Discord voice channel."""
        adapter = self._adapter_for_source(event.source)
        guild_id = self._get_guild_id(event)
        if not (
            guild_id and hasattr(adapter, "leave_voice_channel")
            and hasattr(adapter, "is_in_voice_channel") and adapter.is_in_voice_channel(guild_id)
        ):
            return "Not in a voice channel."
        try:
            await adapter.leave_voice_channel(guild_id)
        except Exception as e:
            logger.warning("Error leaving voice channel: %s", e)
        # Always clean up state even if leave raised an exception
        self._set_voice_mode(self._voice_key_for_source(event.source), "off")
        self._set_adapter_auto_tts_disabled(adapter, event.source.chat_id, disabled=True)
        if hasattr(adapter, "_voice_input_callback"):
            adapter._voice_input_callback = None
        return "Left voice channel."

    def _handle_voice_timeout_cleanup(self, chat_id: str, *, adapter=None) -> None:
        """Adapter callback on voice-channel timeout: clear runner-side voice_mode state.
        ``adapter`` (bound at join) is that profile's bot, not always ``self.adapters[DISCORD]``."""
        if adapter is None:
            adapter = self.adapters.get(Platform.DISCORD)
        profile = getattr(adapter, "_owner_profile", None)
        self._set_voice_mode(self._voice_key(Platform.DISCORD, chat_id, profile=profile), "off")
        self._set_adapter_auto_tts_disabled(adapter, chat_id, disabled=True)

    def _set_voice_mode(self, voice_key: str, mode: str) -> None:
        """Record ``mode`` for ``voice_key`` and persist the voice-mode file."""
        self._voice_mode[voice_key] = mode
        self._save_voice_modes()

    def _is_duplicate_voice_transcript(self, guild_id: int, user_id: int, transcript: str) -> bool:
        """Suppress repeated STT outputs for the same recent utterance (voice capture can emit an
        utterance twice a few seconds apart -> a second queued run and overlapping spoken replies).
        """
        normalized = re.sub(r"[^\w\s]", "", re.sub(r"\s+", " ", transcript).strip().lower())
        if not normalized:
            return False
        now = time.monotonic()
        key = (guild_id, user_id)
        recent_store = getattr(self, "_recent_voice_transcripts", None)
        if not isinstance(recent_store, dict):
            recent_store = self._recent_voice_transcripts = {}
        recent = [(ts, txt) for ts, txt in recent_store.get(key, []) if now - ts <= 12.0]
        if any(
            prior == normalized or (
                len(prior) >= 16 and len(normalized) >= 16
                and SequenceMatcher(None, prior, normalized).ratio() >= 0.95
            )
            for _, prior in recent
        ):
            recent_store[key] = recent
            return True
        recent_store[key] = (recent + [(now, normalized)])[-5:]
        return False

    @staticmethod
    def _voice_input_source(adapter, guild_id: int, user_id: int, text_ch_id) -> SessionSource:
        """Bound text channel's own source when available (voice shares the text conversation's
        session), else a synthetic one."""
        if source_data := getattr(adapter, "_voice_sources", {}).get(guild_id):
            source = SessionSource.from_dict(source_data)
            source.user_id = source.user_name = str(user_id)
            return source
        return SessionSource(
            platform=Platform.DISCORD, chat_id=str(text_ch_id), user_id=str(user_id),
            user_name=str(user_id), chat_type="channel",
            profile=getattr(adapter, "_owner_profile", None),
        )

    @staticmethod
    def _voice_channel_prompt(adapter, text_ch_id) -> Optional[str]:
        """Bound text channel's channel_prompt: voice input gets the same per-channel context."""
        if callable(resolver := getattr(adapter, "_resolve_channel_prompt", None)):
            with suppress(Exception):
                resolved = resolver(str(text_ch_id))
                return resolved if isinstance(resolved, str) else None
        return None

    async def _handle_voice_channel_input(
        self, guild_id: int, user_id: int, transcript: str, *, adapter=None
    ):
        """Handle transcribed voice from a voice channel. ``adapter`` captured the audio (bound
        via ``_bind_voice_input_callback``); under multiplexing each profile's bot dispatches
        through its own adapter, never the default profile's."""
        if adapter is None:
            adapter = self.adapters.get(Platform.DISCORD)
        text_ch_id = adapter._voice_text_channels.get(guild_id) if adapter else None
        if not text_ch_id:
            return
        source = self._voice_input_source(adapter, guild_id, user_id, text_ch_id)
        if not self._is_user_authorized(source):
            logger.debug("Unauthorized voice input from user %d, ignoring", user_id)
            return
        if self._is_duplicate_voice_transcript(guild_id, user_id, transcript):
            logger.info(
                "Suppressing duplicate voice transcript for guild=%s user=%s: %s",
                guild_id, user_id, transcript[:100],
            )
            return
        # Echo the transcript into the text channel (after auth, with mention sanitization).
        with suppress(Exception):
            channel = adapter._client.get_channel(text_ch_id)
            if channel:
                safe_text = transcript[:2000].replace("@everyone", "@\u200beveryone").replace("@here", "@\u200bhere")
                await channel.send(f"**[Voice]** <@{user_id}>: {safe_text}")
        # Synthetic MessageEvent for the normal pipeline; the SimpleNamespace raw_message lets
        # _get_guild_id() extract guild_id so _send_voice_reply() plays audio in the voice channel.
        event = MessageEvent(
            source=source, text=transcript, message_type=MessageType.VOICE,
            raw_message=SimpleNamespace(guild_id=guild_id, guild=None),
            channel_prompt=self._voice_channel_prompt(adapter, text_ch_id),
        )
        await adapter.handle_message(event)

    def _should_send_voice_reply(
        self, event: MessageEvent, response: str, agent_messages: list, already_sent: bool = False
    ) -> bool:
        """False when voice_mode is off for this chat, the response is empty/an error, the agent
        already called text_to_speech this turn, or voice input + base adapter auto-TTS handled
        it — UNLESS streaming consumed the response (already_sent): then the base adapter has no
        text for auto-TTS and the runner must handle it."""
        if not response or response.startswith("Error:"):
            return False
        chat_id = event.source.chat_id
        voice_mode = self._voice_mode.get(self._voice_key_for_source(event.source))
        is_voice_input = event.message_type == MessageType.VOICE
        adapter = self._adapter_for_source(event.source)
        adapter_auto_tts = False
        if adapter and hasattr(adapter, "_should_auto_tts_for_chat"):
            with suppress(Exception):
                adapter_auto_tts = bool(adapter._should_auto_tts_for_chat(chat_id))
        # ``voice.auto_tts`` (synced into the adapter at startup) is the fallback only when the
        # chat has no explicit mode; the chat-level all/voice_only/off choice takes precedence.
        if not (
            voice_mode == "all"
            or (voice_mode == "voice_only" and is_voice_input)
            or (voice_mode is None and adapter_auto_tts)
        ):
            logger.debug(
                "Auto voice reply skipped: mode=%s adapter_auto_tts=%s chat=%s platform=%s",
                voice_mode, adapter_auto_tts, chat_id, event.source.platform.value,
            )
            return False
        # Dedup: agent already called the TTS tool in THIS turn (from the last user message on).
        start = next(
            (i for i, m in reversed(list(enumerate(agent_messages))) if m.get("role") == "user"),
            0,
        )
        if any(
            (tc.get("function") or {}).get("name") == "text_to_speech"
            for msg in agent_messages[start:] if msg.get("role") == "assistant"
            for tc in (msg.get("tool_calls") or [])
        ):
            return False
        # Dedup: base adapter auto-TTS already handles voice input (play_tts plays in VC when
        # connected), so the runner can skip — unless streaming already delivered the text
        # (already_sent): then the base adapter gets None, can't run auto-TTS, and the runner must.
        return not (is_voice_input and not already_sent)

    def _should_echo_stt_transcripts(self) -> bool:
        """Return whether inbound voice/STT transcripts should be echoed to chat."""
        return bool(getattr(self.config, "stt_echo_transcripts", True))

    @staticmethod
    async def _synthesize_voice_reply(text: str, audio_path: str) -> List[str]:
        """Run the TTS tool for ``text`` into ``audio_path``; return the produced file paths (one
        combined file, or several separately valid ones when combination is unavailable / over a
        platform limit; legacy single-file results keep working) — ``[]`` on failure."""
        from tools.tts_tool import text_to_speech_tool

        result_json = await asyncio.to_thread(
            text_to_speech_tool, text=text, output_path=audio_path
        )
        try:
            result = json.loads(result_json)
        except (json.JSONDecodeError, TypeError):
            logger.warning(
                "Auto voice reply TTS returned invalid JSON: %s",
                result_json[:200] if result_json else result_json,
            )
            return []
        actual_paths = [
            str(p) for p in (result.get("file_paths") or [result.get("file_path", audio_path)])
            if p and os.path.isfile(p)
        ]
        if not result.get("success") or not actual_paths:
            logger.warning("Auto voice reply TTS failed: %s", result.get("error"))
            return []
        return actual_paths

    async def _send_voice_reply(self, event: MessageEvent, text: str) -> None:
        """Generate TTS audio and send as a voice message before the text reply."""
        audio_path = None
        actual_paths: List[str] = []
        try:
            from tools.tts_tool import _strip_markdown_for_tts

            tts_text = _strip_markdown_for_tts(text)
            if not tts_text:
                return
            # Platforms whose native voice bubbles require Ogg/Opus (OPUS_VOICE_PLATFORMS) get an
            # explicit .ogg path; the TTS tool's container repair guarantees real Ogg/Opus bytes.
            audio_path = build_auto_tts_output_path(event.source.platform)
            actual_paths = await self._synthesize_voice_reply(tts_text, audio_path)
            if actual_paths:
                await self._deliver_voice_reply(event, actual_paths)
        except Exception as e:
            logger.warning("Auto voice reply failed: %s", e, exc_info=True)
        finally:
            for p in ({audio_path, *actual_paths} - {None}):
                with suppress(OSError):
                    os.unlink(p)

    async def _deliver_voice_reply(self, event: MessageEvent, audio_paths: List[str]) -> None:
        """Play the files in the connected voice channel, else send them as voice messages."""
        adapter = self._adapter_for_source(event.source)
        guild_id = self._get_guild_id(event)
        play = getattr(adapter, "play_in_voice_channel", None)
        is_in_vc = getattr(adapter, "is_in_voice_channel", None)
        if guild_id and callable(play) and callable(is_in_vc) and is_in_vc(guild_id):
            for path in audio_paths:
                await play(guild_id, path)
            return
        if not callable(send_voice := getattr(adapter, "send_voice", None)):
            return
        reply_anchor = self._reply_anchor_for_event(event)
        # Mark the auto voice reply notify-worthy (mirrors the final-text path in platforms/base.py)
        # so adapters that gate push notifications (Telegram "important" mode) deliver it as a
        # normal notification. Clone first: the metadata is shared with typing-indicator state.
        thread_meta = dict(self._thread_metadata_for_source(event.source, reply_anchor) or {})
        thread_meta["notify"] = True
        for path in audio_paths:
            await send_voice(
                chat_id=event.source.chat_id, audio_path=path, reply_to=reply_anchor,
                metadata=thread_meta,
            )
