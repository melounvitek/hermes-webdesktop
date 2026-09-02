"""SimpleX Chat platform adapter (Hermes plugin).

Connects to a simplex-chat daemon running in WebSocket mode. Inbound messages
arrive via a persistent WebSocket connection; outbound messages use the same
WebSocket with JSON commands. The plugin loader calls ``register(ctx)`` at
startup and the platform becomes available to ``gateway/run.py`` and
``tools/send_message_tool`` through the registry.

SimpleX chat daemon setup:
    simplex-chat -p 5225          # start daemon on port 5225
    # or via Docker:
    # docker run -p 5225:5225 simplexchat/simplex-chat-cli -p 5225

Required environment variables:
    SIMPLEX_WS_URL             WebSocket URL of the daemon
                               (default: ws://127.0.0.1:5225)

Optional environment variables:
    SIMPLEX_ALLOWED_USERS      Comma-separated allowlist. Each entry may be
                               either a numeric contactId (stable across
                               renames; visible via `/contacts` in the CLI)
                               or a contact display name (what the SimpleX
                               UI shows). Both forms are accepted.
    SIMPLEX_ALLOW_ALL_USERS    Set 'true' to allow all contacts
    SIMPLEX_AUTO_ACCEPT        Set 'false' to disable contact-request auto-accept
                               (default: 'true')
    SIMPLEX_GROUP_ALLOWED      Comma-separated group IDs to monitor, or '*'
                               for any group. Omit to disable groups entirely.
    SIMPLEX_HOME_CHANNEL       Default contact/group ID for cron delivery
    SIMPLEX_HOME_CHANNEL_NAME  Human label for the home channel
    HERMES_SIMPLEX_TEXT_BATCH_DELAY
                               Quiet-period seconds (default: 0.8) used to
                               concatenate rapid-fire inbound text messages
                               into a single MessageEvent.

``websockets`` is imported lazily so the plugin stays discoverable (and
``hermes setup`` can describe it) when the package is missing;
``check_requirements()`` returns False until it is present.
"""

import asyncio
import base64
import json
import logging
import os
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from gateway.platforms._shared import get_scoped_secret as _get_scoped_secret
from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType, SendResult

logger = logging.getLogger(__name__)

MAX_MESSAGE_LENGTH = 8000  # SimpleX has no hard limit; chunk for sanity
WS_RETRY_DELAY_INITIAL = 2.0
WS_RETRY_DELAY_MAX = 60.0
HEALTH_CHECK_INTERVAL = 30.0
HEALTH_CHECK_STALE_THRESHOLD = 300.0

# Correlation ID prefix for requests we send so we can ignore our own echoes.
_CORR_PREFIX = "hermes-"

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
_AUDIO_EXTS = {".mp3", ".wav", ".ogg", ".m4a", ".aac", ".opus"}


def _parse_comma_list(value: str) -> List[str]:
    """Split a comma-separated string into a stripped list."""
    return [v.strip() for v in value.split(",") if v.strip()]


def _redact_id(contact_id: str) -> str:
    """Redact a contact/group ID for logging."""
    if not contact_id:
        return "<none>"
    s = str(contact_id)
    return s if len(s) <= 4 else s[:2] + "**" + s[-2:]


def _is_image_ext(ext: str) -> bool:
    return ext.lower() in _IMAGE_EXTS


def _is_audio_ext(ext: str) -> bool:
    return ext.lower() in _AUDIO_EXTS


def _send_cmd(chat_id: str, items: list) -> str:
    """Build a structured ``/_send`` command addressing *chat_id* by ID.

    The structured json form is used (rather than ``@name text`` / ``#[id]``)
    because the daemon parses the bare syntax as a display-name lookup and
    silently drops messages when the name doesn't resolve; json.dumps also
    escapes newlines/special chars correctly.
    """
    composed = json.dumps(items)
    if chat_id.startswith("group:"):
        return f"/_send #{chat_id[6:]} json {composed}"
    return f"/_send @{chat_id} json {composed}"


async def _cancel_task(task: Optional[asyncio.Task]) -> None:
    if task:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


class SimplexAdapter(BasePlatformAdapter):
    """SimpleX Chat adapter using the simplex-chat daemon WebSocket API."""

    MAX_MESSAGE_LENGTH = MAX_MESSAGE_LENGTH

    def __init__(self, config: PlatformConfig, **kwargs):
        super().__init__(config=config, platform=Platform("simplex"))

        extra = getattr(config, "extra", {}) or {}
        self.ws_url = extra.get("ws_url", "ws://127.0.0.1:5225").rstrip("/")

        # Auto-accept is on by default; env wins over the ``_env_enablement`` seed.
        env_auto = _get_scoped_secret("SIMPLEX_AUTO_ACCEPT")
        if env_auto is not None:
            self.auto_accept = env_auto.strip().lower() not in {"0", "false", "no", ""}
        else:
            self.auto_accept = bool(extra.get("auto_accept", True))

        # Without SIMPLEX_GROUP_ALLOWED, group messages are ignored entirely
        # (safer default). ``*`` accepts any group.
        group_allowed_str = (_get_scoped_secret("SIMPLEX_GROUP_ALLOWED", "")
                             or extra.get("group_allowed", ""))
        self.group_allow_from = set(_parse_comma_list(group_allowed_str))

        self._ws = None  # websockets connection
        self._ws_task: Optional[asyncio.Task] = None
        self._health_task: Optional[asyncio.Task] = None
        self._running = False
        self._last_ws_activity = 0.0

        # Cosmetic echo filter: corrIds we minted, bounded.
        self._pending_corr_ids: set = set()
        self._max_pending_corr = 200

        # File transfers awaiting rcvFileComplete (keyed by fileId).
        self._pending_file_transfers: Dict[int, dict] = {}

        # Futures for commands whose responses we actually await (``_send_command``).
        self._pending_responses: Dict[str, asyncio.Future] = {}
        self._corr_counter = 0

        # Text batching state consumed by BasePlatformAdapter._enqueue_text_event.
        self._text_batch_delay = float(os.getenv("HERMES_SIMPLEX_TEXT_BATCH_DELAY", "0.8"))
        self._pending_text_batches: Dict[str, MessageEvent] = {}
        self._pending_text_batch_tasks: Dict[str, asyncio.Task] = {}

        logger.info(
            "SimpleX adapter initialized: url=%s auto_accept=%s groups=%s",
            self.ws_url, self.auto_accept, "enabled" if self.group_allow_from else "disabled",
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        """Connect to the simplex-chat daemon and start the WebSocket listener."""
        try:
            import websockets as _wsclient
        except ImportError:
            logger.error("SimpleX: 'websockets' package not installed. Run: pip install websockets")
            return False

        if not self.ws_url:
            logger.error("SimpleX: SIMPLEX_WS_URL is required")
            return False

        # Quick connectivity check — open and immediately close.
        try:
            async with _wsclient.connect(self.ws_url, open_timeout=10):
                pass
        except Exception as e:
            logger.error("SimpleX: cannot reach daemon at %s: %s", self.ws_url, e)
            return False

        self._running = True
        self._last_ws_activity = time.time()
        self._ws_task = asyncio.create_task(self._ws_listener())
        self._health_task = asyncio.create_task(self._health_monitor())

        if hasattr(self, "_mark_connected"):
            self._mark_connected()
        logger.info("SimpleX: connected to %s", self.ws_url)
        self._wire_plugin_handlers(None)
        return True

    async def disconnect(self) -> None:
        """Stop WebSocket listener and clean up."""
        self._running = False
        await _cancel_task(self._ws_task)
        await _cancel_task(self._health_task)

        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

        for task in list(self._pending_text_batch_tasks.values()):
            if not task.done():
                task.cancel()
        self._pending_text_batch_tasks.clear()
        self._pending_text_batches.clear()

        for fut in self._pending_responses.values():
            if not fut.done():
                fut.cancel()
        self._pending_responses.clear()

        if hasattr(self, "_mark_disconnected"):
            self._mark_disconnected()
        logger.info("SimpleX: disconnected")

    # ------------------------------------------------------------------
    # WebSocket listener / health
    # ------------------------------------------------------------------

    async def _ws_listener(self) -> None:
        """Maintain a persistent WebSocket connection to the daemon."""
        import websockets as _wsclient
        from websockets.exceptions import ConnectionClosed

        backoff = WS_RETRY_DELAY_INITIAL
        while self._running:
            try:
                logger.debug("SimpleX WS: connecting to %s", self.ws_url)
                async with _wsclient.connect(
                    self.ws_url, ping_interval=20, ping_timeout=20, close_timeout=10
                ) as ws:
                    self._ws = ws
                    backoff = WS_RETRY_DELAY_INITIAL
                    self._last_ws_activity = time.time()
                    logger.info("SimpleX WS: connected")

                    async for raw in ws:
                        if not self._running:
                            break
                        self._last_ws_activity = time.time()
                        try:
                            await self._handle_event(json.loads(raw))
                        except json.JSONDecodeError:
                            logger.debug("SimpleX WS: invalid JSON: %.100s", raw)
                        except Exception:
                            logger.exception("SimpleX WS: error handling event")
            except asyncio.CancelledError:
                break
            except ConnectionClosed as e:
                if self._running:
                    logger.warning("SimpleX WS: connection closed: %s (reconnecting in %.0fs)",
                                   e, backoff)
            except Exception as e:
                if self._running:
                    logger.warning("SimpleX WS: unexpected error: %s (reconnecting in %.0fs)",
                                   e, backoff)
            finally:
                self._ws = None

            if self._running:
                await asyncio.sleep(backoff + backoff * 0.2 * random.random())
                backoff = min(backoff * 2, WS_RETRY_DELAY_MAX)

    async def _health_monitor(self) -> None:
        """Observe WebSocket idleness without reconnecting healthy quiet links.

        simplex-chat can legitimately stay application-silent for long periods;
        the websockets client already sends protocol pings, so idleness is only
        logged — reconnecting on it causes needless churn.
        """
        while self._running:
            await asyncio.sleep(HEALTH_CHECK_INTERVAL)
            if not self._running:
                break
            elapsed = time.time() - self._last_ws_activity
            if elapsed > HEALTH_CHECK_STALE_THRESHOLD:
                logger.debug("SimpleX: WS application-idle for %.0fs", elapsed)

    # ------------------------------------------------------------------
    # Inbound event handling
    # ------------------------------------------------------------------

    async def _handle_event(self, event: dict) -> None:
        """Dispatch a daemon event to the appropriate handler."""
        # Messages are usually {"corrId": ..., "resp": {"type": ...}}, but some
        # daemons put the response fields at top level — normalize both.
        resp = event.get("resp") if isinstance(event.get("resp"), dict) else event
        corr_id = event.get("corrId")

        if corr_id and corr_id in self._pending_responses:
            fut = self._pending_responses.pop(corr_id)
            if not fut.done():
                fut.set_result(resp)
            return

        # Cosmetic echo filter: prefixed corrIds are ours but weren't awaited.
        if corr_id and isinstance(corr_id, str) and corr_id.startswith(_CORR_PREFIX):
            self._pending_corr_ids.discard(corr_id)
            return

        resp_type = resp.get("type") or event.get("type", "")

        if resp_type == "contactRequest" and self.auto_accept:
            contact_req = resp.get("contactRequest", {}) or {}
            contact_req_id = contact_req.get("contactRequestId")
            if contact_req_id is not None:
                logger.info("SimpleX: auto-accepting contact request %s",
                            _redact_id(str(contact_req_id)))
                await self._send_command(f"/accept {contact_req_id}")
            return

        # simplex fires rcvFileDescrReady before newChatItems for some (XFTP)
        # files; start the download now, the chat item arrives later.
        if resp_type == "rcvFileDescrReady":
            rcv_file = resp.get("rcvFileTransfer", {}) or {}
            file_id = rcv_file.get("fileId") if isinstance(rcv_file, dict) else None
            if file_id is not None:
                logger.debug("SimpleX: rcvFileDescrReady for fileId=%s — sending /freceive",
                             file_id)
                await self._send_fire_and_forget(f"/freceive {file_id}")
            return

        if resp_type == "newChatItems":
            chat_items = resp.get("chatItems", []) or []
            if not isinstance(chat_items, list):
                chat_items = [chat_items]
            for item in chat_items:
                await self._safe_handle_chat_item(item, "SimpleX: error processing chat item")
            return

        # Singular variant — some daemon versions emit this
        if resp_type == "newChatItem":
            await self._safe_handle_chat_item(resp, "SimpleX: error processing chat item")
            return

        # File transfer completion — deliver any deferred chat item
        if resp_type == "rcvFileComplete":
            chat_item_data = (resp.get("chatItem", {}) or {}).get("chatItem", {}) or {}
            file_info = chat_item_data.get("file", {}) or {}
            file_id = file_info.get("fileId") if isinstance(file_info, dict) else None
            if file_id is not None and file_id in self._pending_file_transfers:
                pending = self._pending_file_transfers.pop(file_id)
                file_source = file_info.get("fileSource", {}) or {}
                file_path = file_source.get("filePath") if isinstance(file_source, dict) else None
                if file_path:
                    pending_item_data = pending.get("chatItem", {}) or {}
                    pending_item_data.setdefault("file", {})["fileSource"] = {"filePath": file_path}
                    pending["chatItem"] = pending_item_data
                    await self._safe_handle_chat_item(
                        pending, "SimpleX: error processing deferred file message")
            return

        if resp_type:
            logger.debug("SimpleX: unhandled event type: %s", resp_type)

    async def _safe_handle_chat_item(self, item: dict, err_msg: str) -> None:
        try:
            await self._handle_chat_item(item)
        except Exception:
            logger.exception(err_msg)

    async def _handle_chat_item(self, chat_item: dict) -> None:
        """Process a single chat item from a newChatItems event."""
        chat_info = chat_item.get("chatInfo", {}) or {}
        chat_item_data = chat_item.get("chatItem", {}) or {}
        chat_type = chat_info.get("type", "")
        meta = chat_item_data.get("meta", {}) or {}
        content = chat_item_data.get("content", {}) or {}
        msg_content = content.get("msgContent", {}) or {}

        # Filter out our own messages
        item_direction = chat_item_data.get("chatDir", {}) or {}
        direction_type = item_direction.get("type", "") if isinstance(item_direction, dict) else ""
        if direction_type in ("directSnd", "groupSnd"):
            return

        content_type = content.get("type", "") if isinstance(content, dict) else ""
        if content_type != "rcvMsgContent":
            return

        text = ""
        msg_type_str = msg_content.get("type", "") if isinstance(msg_content, dict) else ""
        if msg_type_str in ("text", "file", "image", "voice", "link", "video"):
            text = msg_content.get("text", "")
        if not text and msg_type_str not in ("image", "file", "voice"):
            return

        sender_id = sender_name = chat_id = ""
        is_group = False
        if chat_type == "direct":
            contact = chat_info.get("contact", {}) or {}
            sender_id = str(contact.get("contactId", ""))
            sender_name = (contact.get("localDisplayName", "")
                           or contact.get("profile", {}).get("displayName", ""))
            chat_id = sender_id
        elif chat_type == "group":
            group_info = chat_info.get("groupInfo", {}) or {}
            group_id = str(group_info.get("groupId", ""))
            chat_id = f"group:{group_id}"
            is_group = True

            member = item_direction.get("groupMember", {}) or {}
            sender_id = str(member.get("memberId", ""))
            sender_name = (member.get("localDisplayName", "")
                           or member.get("memberProfile", {}).get("displayName", ""))

            if self.group_allow_from:
                if "*" not in self.group_allow_from and group_id not in self.group_allow_from:
                    logger.debug("SimpleX: group %s not in allowlist", _redact_id(group_id))
                    return
            else:
                logger.debug("SimpleX: ignoring group message (no SIMPLEX_GROUP_ALLOWED)")
                return
        else:
            logger.debug("SimpleX: unhandled chat type: %s", chat_type)
            return

        if not sender_id:
            logger.debug("SimpleX: ignoring message with no sender")
            return

        # Attachment: chatItem.chatItem.file (sibling of meta/content/chatDir).
        media_urls: List[str] = []
        media_types: List[str] = []
        file_info = chat_item_data.get("file")
        if file_info and isinstance(file_info, dict):
            file_source = file_info.get("fileSource", {}) or {}
            file_path = file_source.get("filePath") if isinstance(file_source, dict) else None
            file_name = file_info.get("fileName", "")
            file_id = file_info.get("fileId")

            ext = Path(file_path).suffix.lower() if file_path else ""
            if not ext and file_name:
                ext = Path(file_name).suffix.lower()

            # Voice notes typically arrive before the file finishes downloading;
            # defer until rcvFileComplete. /freceive gets no corrId reply, so
            # awaiting one would block the event loop.
            if not file_path and _is_audio_ext(ext) and file_id is not None:
                logger.info("SimpleX: voice file %d not yet received, accepting transfer", file_id)
                self._pending_file_transfers[file_id] = chat_item
                await self._send_fire_and_forget(f"/freceive {file_id}")
                return

            if file_path:
                if _is_image_ext(ext):
                    mime = f"image/{ext.lstrip('.')}"
                elif _is_audio_ext(ext):
                    mime = f"audio/{ext.lstrip('.')}"
                else:
                    mime = "application/octet-stream"
                media_urls.append(file_path)
                media_types.append(mime)

        chat_name = sender_name
        if is_group:
            chat_name = group_info.get("localDisplayName", "") or group_info.get(
                "groupProfile", {}
            ).get("displayName", chat_id)
        source = self.build_source(
            chat_id=chat_id, chat_name=chat_name, chat_type="group" if is_group else "dm",
            user_id=sender_id, user_name=sender_name or sender_id,
        )

        msg_type = MessageType.TEXT
        if media_types:
            if any(mt.startswith("audio/") for mt in media_types):
                msg_type = MessageType.VOICE
            elif any(mt.startswith("image/") for mt in media_types):
                msg_type = MessageType.PHOTO
            else:
                # Non-image/non-audio files are documents so run.py's
                # document-context injection surfaces the file to the agent.
                msg_type = MessageType.DOCUMENT

        ts_str = meta.get("itemTs") or meta.get("createdAt", "")
        try:
            timestamp = (datetime.fromisoformat(ts_str.replace("Z", "+00:00")) if ts_str
                         else datetime.now(tz=timezone.utc))
        except (ValueError, AttributeError):
            timestamp = datetime.now(tz=timezone.utc)

        msg_event = MessageEvent(
            source=source, text=text or "", message_type=msg_type, media_urls=media_urls,
            media_types=media_types, timestamp=timestamp, raw_message=chat_item,
        )
        logger.debug("SimpleX: message from %s in %s: %s",
                     _redact_id(sender_id), chat_id[:20], (text or "")[:50])

        # Batch rapid-fire text so the agent sees one combined message.
        if msg_type == MessageType.TEXT and text:
            self._enqueue_text_event(msg_event)
        else:
            await self.handle_message(msg_event)

    # ------------------------------------------------------------------
    # Text message batching (enqueue lives on BasePlatformAdapter)
    # ------------------------------------------------------------------

    def _text_batch_key(self, event: MessageEvent) -> str:
        """Session-scoped key for text message batching."""
        return f"{event.source.platform.value}:{event.source.chat_id}"

    async def _flush_text_batch(self, key: str) -> None:
        """Wait for the quiet period then dispatch the aggregated text."""
        current_task = asyncio.current_task()
        try:
            await asyncio.sleep(self._text_batch_delay)
            event = self._pending_text_batches.pop(key, None)
            if not event:
                return
            logger.info("[SimpleX] Flushing text batch %s (%d chars)", key, len(event.text or ""))
            await self.handle_message(event)
        finally:
            if self._pending_text_batch_tasks.get(key) is current_task:
                self._pending_text_batch_tasks.pop(key, None)

    # ------------------------------------------------------------------
    # Command interface
    # ------------------------------------------------------------------

    def _make_corr_id(self) -> str:
        """Mint a correlation ID and remember it for echo-filtering.

        ``_pending_corr_ids`` is bounded: past ``_max_pending_corr`` the
        overflow is evicted in a single sweep.
        """
        self._corr_counter += 1
        corr_id = f"{_CORR_PREFIX}{self._corr_counter}-{int(time.time() * 1000)}"
        self._pending_corr_ids.add(corr_id)
        if len(self._pending_corr_ids) > self._max_pending_corr:
            overflow = len(self._pending_corr_ids) - self._max_pending_corr
            for _ in range(overflow):
                try:
                    self._pending_corr_ids.pop()
                except KeyError:
                    break
        return corr_id

    async def _send_ws(self, payload: dict) -> None:
        """Fire-and-forget JSON write; drops cleanly when the WS is missing/closed."""
        ws = self._ws
        if not ws:
            logger.debug("SimpleX: WS send dropped (not connected)")
            return
        try:
            await ws.send(json.dumps(payload))
        except Exception as e:
            logger.warning("SimpleX: WS send error: %s", e)

    async def _send_command(self, command: str, timeout: float = 30.0) -> Optional[dict]:
        """Send a command and await the correlated response."""
        ws = self._ws
        if not ws:
            logger.warning("SimpleX: command sent but WebSocket not connected")
            return None

        corr_id = self._make_corr_id()
        payload = json.dumps({"corrId": corr_id, "cmd": command})
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending_responses[corr_id] = fut
        try:
            await ws.send(payload)
            return await asyncio.wait_for(fut, timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning("SimpleX: command timed out: %s", command[:50])
            self._pending_responses.pop(corr_id, None)
            return None
        except Exception as e:
            logger.warning("SimpleX: command failed: %s — %s", command[:50], e)
            self._pending_responses.pop(corr_id, None)
            return None

    async def _send_fire_and_forget(self, command: str) -> None:
        """Send a command the daemon never replies to with a corrId (e.g. ``/freceive``)."""
        await self._send_ws({"corrId": self._make_corr_id(), "cmd": command})

    async def _send_items(self, chat_id: str, items: list, error: str) -> SendResult:
        """Send a structured ``/_send`` payload and await the reply."""
        result = await self._send_command(_send_cmd(chat_id, items))
        if result is not None:
            return SendResult(success=True)
        return SendResult(success=False, error=error)

    # ------------------------------------------------------------------
    # Outbound — text
    # ------------------------------------------------------------------

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send a text message.

        ``MEDIA:<path>`` tags (embedded by TTS / audio tools) are stripped from
        the body and sent as native voice notes or documents. The text send is
        fire-and-forget at the WebSocket level: the daemon doesn't always return
        a corrId reply for chat commands, and waiting would serialise all
        outbound traffic behind a 30-second timeout.
        """
        _voice_exts = {".ogg", ".mp3", ".wav", ".m4a", ".opus"}
        media_paths = re.findall(r"MEDIA:(\S+)", content)
        if media_paths:
            content = re.sub(r"MEDIA:\S+", "", content).strip()

        if content:
            corr_id = self._make_corr_id()
            cmd_str = _send_cmd(chat_id, [{"msgContent": {"type": "text", "text": content}}])
            await self._send_ws({"corrId": corr_id, "cmd": cmd_str})

        for path in media_paths:
            if os.path.splitext(path)[1].lower() in _voice_exts:
                media_result = await self.send_voice(chat_id, path)
            else:
                media_result = await self.send_document(chat_id, path)
            if not media_result.success:
                return media_result

        return SendResult(success=True)

    # ------------------------------------------------------------------
    # Channel directory enumeration
    # ------------------------------------------------------------------

    async def list_channels(self) -> Optional[List[Dict[str, Any]]]:
        """Enumerate contacts and allowed groups for the channel directory.

        Returns ``None`` (not ``[]``) when the WebSocket is down or the daemon
        is unresponsive so the directory falls back to session-history
        discovery instead of wiping known targets. Entry ``id`` values match
        the adapter's send targets: display name for DMs, ``group:<id>`` for groups.
        """
        if not self._ws:
            return None

        channels: List[Dict[str, Any]] = []

        resp = await self._send_command("/contacts", timeout=10.0)
        if resp is None:
            return None
        for contact in resp.get("contacts") or []:
            if not isinstance(contact, dict):
                continue
            contact_id = contact.get("contactId")
            name = (contact.get("localDisplayName", "")
                    or (contact.get("profile", {}) or {}).get("displayName", ""))
            if contact_id is None and not name:
                continue
            # Display name is what the DM send path addresses; fall back to contactId.
            channels.append(
                {"id": str(name or contact_id), "name": str(name or contact_id), "type": "dm"})

        resp = await self._send_command("/groups", timeout=10.0)
        if resp is not None:
            for group in resp.get("groups") or []:
                # Each group is either a groupInfo dict or a [groupInfo, groupSummary] pair.
                if isinstance(group, list) and group:
                    group = group[0]
                if not isinstance(group, dict):
                    continue
                group_id = group.get("groupId")
                if group_id is None:
                    continue
                name = (
                    group.get("localDisplayName", "")
                    or (group.get("groupProfile", {}) or {}).get("displayName", "")
                    or str(group_id)
                )
                channels.append({"id": f"group:{group_id}", "name": str(name), "type": "group"})

        return channels

    # ------------------------------------------------------------------
    # Outbound — media
    # ------------------------------------------------------------------

    @staticmethod
    def _prepare_image(file_path: str) -> tuple[str, str]:
        """Ensure *file_path* is PNG/JPEG and return ``(png_path, thumb_data_uri)``.

        SimpleX clients can't display WebP etc. inline, so convert to PNG when
        needed and build a 128px JPEG thumbnail for the ``image`` field (inline
        preview). Uses Pillow when available, else ImageMagick ``convert``.
        """
        import subprocess
        import tempfile

        p = Path(file_path)
        png_path = file_path
        thumb_uri = ""
        needs_png = p.suffix.lower() not in (".png", ".jpg", ".jpeg")

        try:
            from PIL import Image
            import io

            img = Image.open(file_path)
            if needs_png:
                png_path = str(p.with_suffix(".png"))
                img.save(png_path, "PNG")
            thumb = img.copy()
            thumb.thumbnail((128, 128))
            buf = io.BytesIO()
            thumb.save(buf, "JPEG", quality=70)
            thumb_uri = "data:image/jpg;base64," + base64.b64encode(buf.getvalue()).decode()
        except ImportError:
            try:
                if needs_png:
                    png_path = str(p.with_suffix(".png"))
                    subprocess.run(["convert", file_path, png_path],
                                   check=True, capture_output=True, timeout=30)
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                    tmp_path = tmp.name
                subprocess.run(
                    ["convert", file_path, "-resize", "128x128", "-quality", "70", tmp_path],
                    check=True, capture_output=True, timeout=30,
                )
                with open(tmp_path, "rb") as f:
                    thumb_uri = "data:image/jpg;base64," + base64.b64encode(f.read()).decode()
                os.remove(tmp_path)
            except (FileNotFoundError, subprocess.SubprocessError) as exc:
                logger.warning("SimpleX: image conversion unavailable: %s", exc)

        return png_path, thumb_uri

    async def send_image(
        self, chat_id: str, image_url: str, caption: Optional[str] = None, **kwargs
    ) -> SendResult:
        """Send an image. Supports ``file://`` URLs and ``http(s)://`` URLs."""
        from urllib.parse import unquote

        if image_url.startswith("file://"):
            file_path = unquote(image_url[7:])
        else:
            try:
                from gateway.platforms.base import cache_image_from_url

                file_path = await cache_image_from_url(image_url)
            except Exception as e:
                logger.warning("SimpleX: failed to download image: %s", e)
                return SendResult(success=False, error=str(e))

        if not file_path or not Path(file_path).exists():
            return SendResult(success=False, error="Image file not found")

        png_path, thumb_uri = self._prepare_image(file_path)
        # /_send addresses by numeric ID; /f only accepts display names.
        item = {"filePath": png_path,
                "msgContent": {"type": "image", "image": thumb_uri, "text": caption or ""}}
        return await self._send_items(chat_id, [item], "Failed to send image")

    async def send_image_file(
        self, chat_id: str, image_path: str, caption: Optional[str] = None,
        reply_to: Optional[str] = None, **kwargs,
    ) -> SendResult:
        """Send a local image file via SimpleX."""
        return await self.send_image(chat_id, f"file://{image_path}", caption=caption, **kwargs)

    async def send_video(
        self, chat_id: str, video_path: str, caption: Optional[str] = None,
        reply_to: Optional[str] = None, **kwargs,
    ) -> SendResult:
        """Send a video file via SimpleX (as a file attachment)."""
        return await self.send_document(chat_id, video_path, caption=caption)

    async def send_document(
        self, chat_id: str, file_path: str, caption: Optional[str] = None,
        filename: Optional[str] = None, **kwargs,
    ) -> SendResult:
        """Send a document/file attachment."""
        if not Path(file_path).exists():
            return SendResult(success=False, error="File not found")
        item = {"filePath": file_path, "msgContent": {"type": "file", "text": caption or ""}}
        return await self._send_items(chat_id, [item], "Failed to send document")

    async def send_voice(
        self, chat_id: str, audio_path: str, caption: Optional[str] = None,
        reply_to: Optional[str] = None, duration: int = 0, **kwargs,
    ) -> SendResult:
        """Send an audio file as an inline SimpleX voice note (``msgContent.type == "voice"``)."""
        if not Path(audio_path).exists():
            return SendResult(success=False, error="Voice file not found")
        item = {
            "msgContent": {"type": "voice", "text": caption or "", "duration": duration},
            "fileSource": {"filePath": audio_path},
        }
        return await self._send_items(chat_id, [item], "Failed to send voice message")

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        """SimpleX has no typing-indicator API — no-op."""

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Return basic chat info."""
        if chat_id.startswith("group:"):
            return {"chat_id": chat_id, "type": "group", "name": chat_id[6:]}
        return {"chat_id": chat_id, "type": "dm", "name": chat_id}


# ---------------------------------------------------------------------------
# Plugin entry-point hooks
# ---------------------------------------------------------------------------

def check_requirements() -> bool:
    """Plugin gate: require SIMPLEX_WS_URL AND the websockets package."""
    if not _get_scoped_secret("SIMPLEX_WS_URL"):
        return False
    try:
        import websockets  # noqa: F401
    except ImportError:
        return False
    return True


def validate_config(config) -> bool:
    """Validate that the platform config has enough info to connect."""
    extra = getattr(config, "extra", {}) or {}
    return bool(_get_scoped_secret("SIMPLEX_WS_URL") or extra.get("ws_url", ""))


def is_connected(config) -> bool:
    """Check whether SimpleX is configured (env or config.yaml)."""
    return validate_config(config)


def _env_enablement() -> Optional[dict]:
    """Seed ``PlatformConfig.extra`` from env vars during gateway config load.

    Runs BEFORE adapter construction so ``gateway status`` reflects env-only
    configuration. Returns ``None`` when SimpleX isn't minimally configured.
    ``home_channel`` is turned into a ``HomeChannel`` by the core hook.
    """
    ws_url = _get_scoped_secret("SIMPLEX_WS_URL", "").strip()
    if not ws_url:
        return None
    seed: dict = {"ws_url": ws_url}

    auto_accept = _get_scoped_secret("SIMPLEX_AUTO_ACCEPT", "").strip().lower()
    if auto_accept:
        seed["auto_accept"] = auto_accept not in {"0", "false", "no"}

    group_allowed = _get_scoped_secret("SIMPLEX_GROUP_ALLOWED", "").strip()
    if group_allowed:
        seed["group_allowed"] = group_allowed

    home = _get_scoped_secret("SIMPLEX_HOME_CHANNEL", "").strip()
    if home:
        seed["home_channel"] = {
            "chat_id": home,
            "name": _get_scoped_secret("SIMPLEX_HOME_CHANNEL_NAME", "").strip() or home,
        }
    return seed


async def _standalone_send(
    pconfig,
    chat_id: str,
    message: str,
    *,
    thread_id: Optional[str] = None,
    media_files: Optional[List[str]] = None,
    force_document: bool = False,
) -> Dict[str, Any]:
    """Open an ephemeral WebSocket to the daemon, send, and close.

    Used by ``tools/send_message_tool`` when the gateway runner is not in this
    process (e.g. ``hermes cron``). ``thread_id``/``force_document`` are
    signature parity only; ``media_files`` is accepted but only the text body
    is delivered — SimpleX file transfers need the daemon's filesystem-backed
    flow, which an ephemeral connection cannot drive safely.
    """
    try:
        import websockets as _wsclient
    except ImportError:
        return {"error": "websockets not installed. Run: pip install websockets"}

    extra = getattr(pconfig, "extra", {}) or {}
    ws_url = _get_scoped_secret("SIMPLEX_WS_URL") or extra.get("ws_url", "ws://127.0.0.1:5225")
    if not ws_url:
        return {"error": "SimpleX standalone send: SIMPLEX_WS_URL is required"}

    try:
        payload = {
            "corrId": f"{_CORR_PREFIX}snd-{int(time.time() * 1000)}",
            "cmd": _send_cmd(chat_id, [{"msgContent": {"type": "text", "text": message}}]),
        }
        async with _wsclient.connect(ws_url, open_timeout=10, close_timeout=5) as ws:
            await ws.send(json.dumps(payload))
            # Give the daemon a moment to process the command before closing.
            await asyncio.sleep(0.5)
        return {"success": True, "platform": "simplex", "chat_id": chat_id}
    except Exception as e:
        return {"error": f"SimpleX send failed: {e}"}


def interactive_setup() -> None:
    """Minimal stdin wizard for ``hermes setup gateway`` → SimpleX; writes ``~/.hermes/.env``."""
    print()
    print("SimpleX Chat setup")
    print("------------------")
    print("Requirements:")
    print("  1. simplex-chat daemon running (e.g. `simplex-chat -p 5225`).")
    print("  2. Python package `websockets` installed (`pip install websockets`).")
    print()

    try:
        from hermes_cli.config import get_env_value, save_env_value
    except ImportError:
        print("hermes_cli.config not available; set SIMPLEX_* vars manually in ~/.hermes/.env")
        return

    def _prompt(var: str, prompt: str) -> None:
        existing = get_env_value(var) if callable(get_env_value) else None
        suffix = " [keep current]" if existing else ""
        try:
            value = input(f"{prompt}{suffix}: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if value:
            save_env_value(var, value)

    _prompt("SIMPLEX_WS_URL", "Daemon WebSocket URL (default ws://127.0.0.1:5225)")
    _prompt("SIMPLEX_ALLOWED_USERS",
            "Allowed contactIds or display names (comma-separated; blank=skip)")
    _prompt("SIMPLEX_GROUP_ALLOWED",
            "Allowed group IDs (comma-separated, or '*' for any; blank=disable groups)")
    _prompt("SIMPLEX_AUTO_ACCEPT",
            "Auto-accept incoming contact requests? (true/false, default true)")
    _prompt("SIMPLEX_HOME_CHANNEL", "Home channel contact/group ID (or empty)")
    print("Done. Make sure the simplex-chat daemon is running before starting the gateway.")


def register(ctx) -> None:
    """Plugin entry point — called by the Hermes plugin system at startup."""
    ctx.register_platform(
        name="simplex",
        label="SimpleX Chat",
        adapter_factory=lambda cfg: SimplexAdapter(cfg),
        check_fn=check_requirements,
        validate_config=validate_config,
        is_connected=is_connected,
        required_env=["SIMPLEX_WS_URL"],
        install_hint=("pip install websockets   # SimpleX adapter requires the websockets package"),
        setup_fn=interactive_setup,
        env_enablement_fn=_env_enablement,
        cron_deliver_env_var="SIMPLEX_HOME_CHANNEL",
        standalone_sender_fn=_standalone_send,
        allowed_users_env="SIMPLEX_ALLOWED_USERS",
        allow_all_env="SIMPLEX_ALLOW_ALL_USERS",
        max_message_length=MAX_MESSAGE_LENGTH,
        emoji="🔒",
        # SimpleX uses opaque contact IDs only — nothing to redact.
        pii_safe=True,
        allow_update_command=True,
        platform_hint=(
            "You are chatting via SimpleX Chat, a private decentralised "
            "messenger. Contacts are identified by opaque internal IDs, "
            "not phone numbers or usernames. SimpleX supports standard "
            "markdown formatting. There is no typing indicator and no "
            "hard message length limit, but keep responses conversational. "
            "You can attach native images, voice notes, and arbitrary "
            "files; the adapter handles MEDIA:<path> tags by sending them "
            "as inline voice notes (audio extensions) or documents."
        ),
    )
