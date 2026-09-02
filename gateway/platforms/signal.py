"""Signal messenger platform adapter.

Connects to a signal-cli daemon running in HTTP mode. Inbound messages arrive
via SSE streaming; outbound messages and actions use JSON-RPC 2.0 over HTTP.

Requires:
  - signal-cli installed and running: signal-cli daemon --http 127.0.0.1:8080
  - SIGNAL_HTTP_URL and SIGNAL_ACCOUNT environment variables set
"""

import asyncio
import base64
import itertools
import json
import logging
import os
import random
import shutil
import subprocess
import tempfile
import time
import uuid
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote, unquote

import httpx

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    ProcessingOutcome,
    SendResult,
    cache_image_from_bytes,
    cache_audio_from_bytes,
    cache_document_from_bytes,
    cache_image_from_url,
    utf16_len,
)
from gateway.platforms.helpers import redact_phone
from gateway.platforms.media_cache import mime_for_ext
from tools.audio_container import CONTAINER_TO_EXT, sniff_container
from gateway.platforms.signal_format import markdown_to_signal
from gateway.platforms.signal_rate_limit import (
    SIGNAL_BATCH_PACING_NOTICE_THRESHOLD,
    SIGNAL_MAX_ATTACHMENTS_PER_MSG,
    SIGNAL_RATE_LIMIT_MAX_ATTEMPTS,
    SignalRateLimitError,
    _extract_retry_after_seconds,
    _format_wait,
    _is_signal_rate_limit_error,
    _signal_send_timeout,
    get_scheduler,
)
from gateway.platforms._shared import get_scoped_secret as _sig_secret

logger = logging.getLogger(__name__)

SIGNAL_MAX_ATTACHMENT_SIZE = 100 * 1024 * 1024  # 100 MB
MAX_MESSAGE_LENGTH = 8000  # Signal message size limit
SSE_RETRY_DELAY_INITIAL = 2.0
SSE_RETRY_DELAY_MAX = 60.0
HEALTH_CHECK_INTERVAL = 30.0  # seconds between health checks
HEALTH_CHECK_STALE_THRESHOLD = 120.0  # seconds without SSE activity before concern

# Magic-byte prefixes checked before delegating to the shared audio/AV sniffer.
_MAGIC_EXTENSIONS = (
    (b"\x89PNG", ".png"),
    (b"\xff\xd8", ".jpg"),
    (b"GIF8", ".gif"),
    (b"%PDF", ".pdf"),
)
_MEDIA_TYPE_BY_MIME_PREFIX = (
    ("audio/", MessageType.VOICE),
    ("image/", MessageType.PHOTO),
    ("video/", MessageType.VIDEO),
)
_OUTCOME_REACTION = {ProcessingOutcome.SUCCESS: "✅", ProcessingOutcome.FAILURE: "❌"}
_QUOTE_AUTHOR_KEYS = (
    "author", "authorNumber", "authorUuid", "authorAci", "authorServiceId", "authorServiceIdString",
)


def _parse_comma_list(value: str) -> List[str]:
    """Split a comma-separated string into a list, stripping whitespace."""
    return [v.strip() for v in value.split(",") if v.strip()]


def _guess_extension(data: bytes) -> str:
    """Guess file extension from magic bytes.

    WEBP is claimed before the shared audio/AV sniffer (it shares RIFF with WAVE);
    tools/audio_container.py owns the MP3-vs-ADTS-AAC sync-word disambiguation.
    """
    for magic, ext in _MAGIC_EXTENSIONS:
        if data.startswith(magic):
            return ext
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp"
    container = sniff_container(data)
    if container is not None:
        return CONTAINER_TO_EXT[container]
    if data[:2] == b"PK":
        return ".zip"
    return ".bin"


def _is_image_ext(ext: str) -> bool:
    return ext.lower() in {".jpg", ".jpeg", ".png", ".gif", ".webp"}


def _is_audio_ext(ext: str) -> bool:
    return ext.lower() in {".mp3", ".wav", ".ogg", ".m4a", ".aac"}


def _ext_to_mime(ext: str) -> str:
    """Map file extension to MIME type (shared table matches Signal's historical map)."""
    return mime_for_ext(ext, fallback="application/octet-stream")


def _remux_aac_to_m4a(aac_data: bytes) -> Optional[Tuple[bytes, str]]:
    """Losslessly remux raw ADTS AAC (Android voice notes, rejected by most STT APIs) to .m4a.

    Returns ``(m4a_bytes, ".m4a")`` or ``None`` when ffmpeg is missing or fails —
    callers must then pass the input through unchanged.
    """
    # Fall back to common Homebrew/local prefixes on macOS dev hosts.
    ffmpeg = shutil.which("ffmpeg") or next(
        (p for p in ("/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg")
         if os.path.isfile(p) and os.access(p, os.X_OK)), None,
    )
    if not ffmpeg:
        logger.debug("Signal: ffmpeg not found, skipping AAC→M4A remux")
        return None
    try:
        with tempfile.NamedTemporaryFile(suffix=".aac", delete=False) as src:
            src.write(aac_data)
            src_path = src.name
        dst_path = src_path[:-4] + ".m4a"
        try:
            proc = subprocess.run(
                [ffmpeg, "-y", "-loglevel", "error", "-i", src_path,
                 "-c:a", "copy", "-movflags", "+faststart", dst_path],
                capture_output=True, timeout=10,
            )
            if proc.returncode != 0:
                logger.warning(
                    "Signal: AAC→M4A remux failed (ffmpeg exit %d): %s",
                    proc.returncode, proc.stderr.decode("utf-8", "replace")[:300],
                )
                return None
            with open(dst_path, "rb") as f:
                return f.read(), ".m4a"
        finally:
            for p in (src_path, dst_path):
                try:
                    os.unlink(p)
                except OSError:
                    pass
    except subprocess.TimeoutExpired:
        logger.warning("Signal: AAC→M4A remux timed out (>10s)")
        return None
    except Exception:
        logger.exception("Signal: AAC→M4A remux error")
        return None


def _render_mentions(text: str, mentions: list) -> str:
    """Replace Signal mention placeholders (\\uFFFC + out-of-band start/length/number
    metadata) with readable @identifiers. Replace from the end so indices hold."""
    if not mentions or "\uFFFC" not in text:
        return text
    for mention in sorted(mentions, key=lambda m: m.get("start", 0), reverse=True):
        start = mention.get("start", 0)
        length = mention.get("length", 1)
        identifier = mention.get("number") or mention.get("uuid") or "user"
        text = text[:start] + f"@{identifier}" + text[start + length:]
    return text


def _is_signal_service_id(value: str) -> bool:
    """Return True if *value* already looks like a Signal service identifier."""
    if not value:
        return False
    if value.startswith("PNI:") or value.startswith("u:"):
        return True
    try:
        uuid.UUID(value)
        return True
    except (ValueError, AttributeError, TypeError):
        return False


def _looks_like_e164_number(value: str) -> bool:
    """Return True for a plausible E.164 phone number."""
    if not value or not value.startswith("+"):
        return False
    digits = value[1:]
    return digits.isdigit() and 7 <= len(digits) <= 15


def check_signal_requirements() -> bool:
    """Check if Signal runtime dependencies are available."""
    return True


def validate_signal_config(config: PlatformConfig) -> bool:
    """Check if Signal has enough config to connect."""
    extra = getattr(config, "extra", {}) or {}
    http_url = (extra.get("http_url", "") or os.getenv("SIGNAL_HTTP_URL", "")).strip()
    account = (extra.get("account", "") or os.getenv("SIGNAL_ACCOUNT", "")).strip()
    return bool(http_url and account)


class SignalAdapter(BasePlatformAdapter):
    """Signal messenger adapter using signal-cli HTTP daemon."""

    platform = Platform.SIGNAL
    MAX_MESSAGE_LENGTH = MAX_MESSAGE_LENGTH
    splits_long_messages = True  # send() chunks after markdown → Signal formatting conversion
    # Signal has no real edit API; declaring it lets streaming suppress the visible
    # cursor instead of leaving a stale tofu square behind when edits fail.
    SUPPORTS_MESSAGE_EDITING = False

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.SIGNAL)
        extra = config.extra or {}
        self.http_url = extra.get("http_url", "http://127.0.0.1:8080").rstrip("/")
        self.account = extra.get("account", "")
        self.ignore_stories = extra.get("ignore_stories", True)
        # Allowlists are per-profile: scoped reads so secondary profiles don't inherit the
        # default profile's list. Group policy derives from the group allowlist's presence.
        self.group_allow_from = set(_parse_comma_list(_sig_secret("SIGNAL_GROUP_ALLOWED_USERS", "")))
        # Mention filter — only respond in groups when the bot account is @mentioned.
        _rm_cfg = extra.get("require_mention")
        if _rm_cfg is not None:
            self.require_mention = bool(_rm_cfg)
        else:
            self.require_mention = os.getenv("SIGNAL_REQUIRE_MENTION", "false").lower() in ("true", "1", "yes", "on")
        # DM allowlist mirrors run.py's SIGNAL_ALLOWED_USERS check so the reaction hooks
        # (which fire before run.py's auth gate) can skip unauthorized senders. "*" = open.
        self.dm_allow_from = set(_parse_comma_list(_sig_secret("SIGNAL_ALLOWED_USERS", "*")))
        self.client: Optional[httpx.AsyncClient] = None
        self._sse_task: Optional[asyncio.Task] = None
        self._health_monitor_task: Optional[asyncio.Task] = None
        self._typing_tasks: Dict[str, asyncio.Task] = {}
        # Per-chat typing-indicator backoff: when signal-cli reports NETWORK_FAILURE,
        # base.py's _keep_typing loop would otherwise hammer sendTyping every ~2s.
        self._typing_failures: Dict[str, int] = {}
        self._typing_skip_until: Dict[str, float] = {}
        self._running = False
        self._last_sse_activity = 0.0
        self._sse_response: Optional[httpx.Response] = None
        self._account_normalized = self.account.strip()
        # Recently sent timestamps filter echo-backs (Note to Self / linked-device group
        # sync-sents). LRU + TTL so a still-pending echo in a chatty group isn't evicted
        # just because many outbounds happened; the cap only guards runaway producers.
        self._recent_sent_timestamps: "OrderedDict[int, float]" = OrderedDict()
        self._max_recent_timestamps = 512
        self._recent_sent_ttl_seconds = 300.0
        # Separate FIFO cache of outbound timestamps: Signal quote.id is the quoted
        # message's timestamp, so replies to this bot are recognised even after the
        # self-sync echo above was consumed.
        self._sent_message_timestamps: "OrderedDict[str, None]" = OrderedDict()
        self._max_sent_message_timestamps = 500
        # Best-effort number↔ACI/PNI UUID mapping so outbound sends can upgrade a
        # phone number to the UUID signal-cli prefers.
        self._recipient_uuid_by_number: Dict[str, str] = {}
        self._recipient_number_by_uuid: Dict[str, str] = {}
        self._recipient_cache_lock = asyncio.Lock()
        logger.info("Signal adapter initialized: url=%s account=%s groups=%s",
                     self.http_url, redact_phone(self.account),
                     "enabled" if self.group_allow_from else "disabled")

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        """Connect to signal-cli daemon and start SSE listener."""
        if not self.http_url or not self.account:
            logger.error("Signal: SIGNAL_HTTP_URL and SIGNAL_ACCOUNT are required")
            return False
        # Scoped lock prevents duplicate Signal listeners for the same phone.
        lock_acquired = False
        try:
            if not self._acquire_platform_lock('signal-phone', self.account, 'Signal account'):
                return False
            lock_acquired = True
        except Exception as e:
            logger.warning("Signal: Could not acquire phone lock (non-fatal): %s", e)
        # Tighter keepalive so idle CLOSE_WAIT drains promptly.
        from gateway.platforms._http_client_limits import platform_httpx_limits
        self.client = httpx.AsyncClient(timeout=30.0, limits=platform_httpx_limits())
        try:
            try:
                resp = await self.client.get(f"{self.http_url}/api/v1/check", timeout=10.0)
                if resp.status_code != 200:
                    logger.error("Signal: health check failed (status %d)", resp.status_code)
                    return False
            except Exception as e:
                logger.error("Signal: cannot reach signal-cli at %s: %s", self.http_url, e)
                return False
            self._running = True
            self._last_sse_activity = time.time()
            self._sse_task = asyncio.create_task(self._sse_listener())
            self._health_monitor_task = asyncio.create_task(self._health_monitor())
            logger.info("Signal: connected to %s", self.http_url)
            # Plugin-registered native handlers (ctx.register_platform_handler).
            self._wire_plugin_handlers(None)
            return True
        finally:
            if not self._running:
                if self.client:
                    await self.client.aclose()
                    self.client = None
                if lock_acquired:
                    self._release_platform_lock()

    @staticmethod
    async def _cancel_task(task: Optional[asyncio.Task]) -> None:
        if task:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def disconnect(self) -> None:
        """Stop SSE listener and clean up."""
        self._running = False
        for task in (self._sse_task, self._health_monitor_task):
            await self._cancel_task(task)
        for task in self._typing_tasks.values():
            task.cancel()
        self._typing_tasks.clear()
        if self.client:
            await self.client.aclose()
            self.client = None
        self._release_platform_lock()
        logger.info("Signal: disconnected")

    async def _sse_listener(self) -> None:
        """Listen for SSE events from signal-cli daemon."""
        url = f"{self.http_url}/api/v1/events?account={quote(self.account, safe='')}"
        backoff = SSE_RETRY_DELAY_INITIAL
        while self._running:
            try:
                logger.debug("Signal SSE: connecting to %s", url)
                async with self.client.stream(
                    "GET", url, headers={"Accept": "text/event-stream"}, timeout=None,
                ) as response:
                    self._sse_response = response
                    backoff = SSE_RETRY_DELAY_INITIAL  # Reset on successful connection
                    self._last_sse_activity = time.time()
                    logger.info("Signal SSE: connected")
                    buffer = ""
                    async for chunk in response.aiter_text():
                        if not self._running:
                            break
                        buffer += chunk
                        while "\n" in buffer:
                            line, buffer = buffer.split("\n", 1)
                            line = line.strip()
                            if not line:
                                continue
                            # Keepalive comments (":") prove the connection is alive —
                            # count them as activity so the health monitor stays quiet.
                            if line.startswith(":"):
                                self._last_sse_activity = time.time()
                                continue
                            if line.startswith("data:"):
                                data_str = line[5:].strip()
                                if not data_str:
                                    continue
                                self._last_sse_activity = time.time()
                                try:
                                    data = json.loads(data_str)
                                    await self._handle_envelope(data)
                                except json.JSONDecodeError:
                                    logger.debug("Signal SSE: invalid JSON: %s", data_str[:100])
                                except Exception:
                                    logger.exception("Signal SSE: error handling event")
            except asyncio.CancelledError:
                break
            except httpx.HTTPError as e:
                if self._running:
                    logger.warning("Signal SSE: HTTP error: %s (reconnecting in %.0fs)", e, backoff)
            except Exception as e:
                if self._running:
                    logger.warning("Signal SSE: error: %s (reconnecting in %.0fs)", e, backoff)
            if self._running:
                # 20% jitter prevents thundering herd on reconnection
                jitter = backoff * 0.2 * random.random()
                await asyncio.sleep(backoff + jitter)
                backoff = min(backoff * 2, SSE_RETRY_DELAY_MAX)
        self._sse_response = None

    async def _health_monitor(self) -> None:
        """Monitor SSE connection health and force reconnect if stale."""
        while self._running:
            await asyncio.sleep(HEALTH_CHECK_INTERVAL)
            if not self._running:
                break
            elapsed = time.time() - self._last_sse_activity
            if elapsed > HEALTH_CHECK_STALE_THRESHOLD:
                logger.warning("Signal: SSE idle for %.0fs, checking daemon health", elapsed)
                try:
                    resp = await self.client.get(f"{self.http_url}/api/v1/check", timeout=10.0)
                    if resp.status_code == 200:
                        # Daemon alive but SSE quiet — reset activity to avoid repeated warnings
                        self._last_sse_activity = time.time()
                        logger.debug("Signal: daemon healthy, SSE idle")
                    else:
                        logger.warning("Signal: health check failed (%d), forcing reconnect", resp.status_code)
                        self._force_reconnect()
                except Exception as e:
                    logger.warning("Signal: health check error: %s, forcing reconnect", e)
                    self._force_reconnect()

    def _force_reconnect(self) -> None:
        """Force SSE reconnection by closing the current response."""
        if self._sse_response and not self._sse_response.is_stream_consumed:
            try:
                task = asyncio.create_task(self._sse_response.aclose())
                self._background_tasks.add(task)
                task.add_done_callback(self._background_tasks.discard)
            except Exception:
                pass
            self._sse_response = None

    def _unwrap_sync_message(self, envelope_data: dict) -> Optional[dict]:
        """Promote a genuine "Note to Self" / group sync-sent to a dataMessage envelope;
        None for other sync events (read receipts, typing, our own outbound echoes)."""
        sync_msg = envelope_data.get("syncMessage")
        if not sync_msg or not isinstance(sync_msg, dict):
            return None
        sent_msg = sync_msg.get("sentMessage")
        if not sent_msg or not isinstance(sent_msg, dict):
            return None
        dest = sent_msg.get("destinationNumber") or sent_msg.get("destination")
        sent_msg_group_info = sent_msg.get("groupInfo") or {}
        sent_msg_group_id = sent_msg_group_info.get("groupId") if sent_msg_group_info else None
        if dest != self._account_normalized and not sent_msg_group_id:
            return None
        if self._consume_sent_timestamp(sent_msg.get("timestamp")):
            return None  # echo of our own outbound reply
        return {**envelope_data, "dataMessage": sent_msg}

    def _apply_group_mention_rules(self, text: str, data_message: dict) -> Tuple[bool, str]:
        """Gate on require_mention (False = drop) and strip the bot's own @mention.

        The self-mention is stripped from every group message so the agent doesn't
        read "@+155****4567 say hello" as a directive to contact that number.
        """
        account_norm = self._account_normalized
        if self.require_mention:
            mentioned_in_text = account_norm and (f"@{account_norm}" in (text or ""))
            mentioned_in_metadata = any(
                m.get("number") == account_norm or m.get("uuid") == account_norm
                for m in (data_message.get("mentions") or [])
            )
            if not mentioned_in_text and not mentioned_in_metadata:
                logger.debug("Signal: ignoring group message (require_mention=true, bot not mentioned)")
                return False, text
        if text and account_norm:
            text = text.replace(f"@{account_norm}", "")
            bot_uuid = self._recipient_uuid_by_number.get(account_norm)
            if bot_uuid:
                text = text.replace(f"@{bot_uuid}", "")
            # Collapse only the doubled space the removal introduced; intentional
            # newlines in multi-line messages are preserved.
            text = text.replace("  ", " ").strip()
        return True, text

    async def _collect_attachments(self, attachments_data: list) -> Tuple[List[str], List[str]]:
        """Fetch + cache inbound attachments; returns (media_urls, media_types)."""
        media_urls: List[str] = []
        media_types: List[str] = []
        for att in attachments_data:
            att_id = att.get("id")
            att_size = att.get("size", 0)
            if not att_id:
                continue
            if att_size > SIGNAL_MAX_ATTACHMENT_SIZE:
                logger.warning("Signal: attachment too large (%d bytes), skipping", att_size)
                continue
            try:
                cached_path, ext = await self._fetch_attachment(att_id)
                if cached_path:
                    media_urls.append(cached_path)
                    media_types.append(att.get("contentType") or _ext_to_mime(ext))
            except Exception:
                logger.exception("Signal: failed to fetch attachment %s", att_id)
        return media_urls, media_types

    async def _handle_envelope(self, envelope: dict) -> None:
        """Process an incoming signal-cli envelope."""
        envelope_data = envelope.get("envelope", envelope)
        is_note_to_self = False
        if "syncMessage" in envelope_data:
            envelope_data = self._unwrap_sync_message(envelope_data)
            if envelope_data is None:
                return
            is_note_to_self = True
        sender = (
            envelope_data.get("sourceNumber")
            or envelope_data.get("sourceUuid")
            or envelope_data.get("source")
        )
        sender_name = envelope_data.get("sourceName", "")
        sender_uuid = envelope_data.get("sourceUuid", "")
        self._remember_recipient_identifiers(sender, sender_uuid)
        if not sender:
            logger.debug("Signal: ignoring envelope with no sender")
            return
        # Self-message filtering prevents reply loops (Note to Self is allowed)
        if self._account_normalized and sender == self._account_normalized and not is_note_to_self:
            return
        if self.ignore_stories and envelope_data.get("storyMessage"):
            return
        # Edited messages carry their updated dataMessage inside editMessage
        data_message = (
            envelope_data.get("dataMessage")
            or (envelope_data.get("editMessage") or {}).get("dataMessage")
        )
        if not data_message:
            return
        group_info = data_message.get("groupInfo")
        group_id = group_info.get("groupId") if group_info else None
        is_group = bool(group_id)
        # Group policy derives from SIGNAL_GROUP_ALLOWED_USERS: unset → groups disabled;
        # IDs → only those groups; "*" → all. DM auth is run.py's (_is_user_authorized).
        if is_group:
            if not self.group_allow_from:
                logger.debug("Signal: ignoring group message (no SIGNAL_GROUP_ALLOWED_USERS)")
                return
            if "*" not in self.group_allow_from and group_id not in self.group_allow_from:
                logger.debug("Signal: group %s not in allowlist", group_id[:8] if group_id else "?")
                return
        chat_id = sender if not is_group else f"group:{group_id}"
        chat_type = "group" if is_group else "dm"
        text = data_message.get("message", "")
        mentions = data_message.get("mentions", [])
        if text and mentions:
            text = _render_mentions(text, mentions)
        if is_group:
            mentioned, text = self._apply_group_mention_rules(text, data_message)
            if not mentioned:
                return
        # Signal's quote.id is the quoted message's timestamp; quote.author the quoted
        # sender. Preserve both so the gateway can tell the agent which message the
        # user replied to.
        quote_data = data_message.get("quote") or {}
        reply_to_id = str(quote_data.get("id")) if quote_data.get("id") else None
        reply_to_author = self._extract_quote_author(quote_data)
        attachments_data = data_message.get("attachments", [])
        media_urls: List[str] = []
        media_types: List[str] = []
        if attachments_data and not getattr(self, "ignore_attachments", False):
            media_urls, media_types = await self._collect_attachments(attachments_data)
        # Skip contentless envelopes (profile key updates, empty messages) that still
        # carry a dataMessage wrapper — otherwise msg='' triggers a full agent turn.
        if (not text or not text.strip()) and not media_urls:
            logger.debug(
                "Signal: skipping contentless envelope from %s (%d attachments)",
                redact_phone(sender), len(media_urls) if media_urls else 0,
            )
            return
        source = self.build_source(
            chat_id=chat_id,
            chat_name=group_info.get("groupName") if group_info else sender_name,
            chat_type=chat_type,
            user_id=sender,
            user_name=sender_name or sender,
            user_id_alt=sender_uuid if sender_uuid else None,
            chat_id_alt=group_id if is_group else None,
        )
        # First matching MIME prefix wins; everything else (application/*, text/*,
        # unknown) is a DOCUMENT so run.py's document-context injection surfaces the
        # cached path to the agent.
        msg_type = MessageType.TEXT
        if media_types:
            msg_type = next(
                (mt for prefix, mt in _MEDIA_TYPE_BY_MIME_PREFIX
                 if any(m.startswith(prefix) for m in media_types)),
                MessageType.DOCUMENT,
            )
        ts_ms = envelope_data.get("timestamp", 0)  # milliseconds since epoch
        timestamp = datetime.now(tz=timezone.utc)
        if ts_ms:
            try:
                timestamp = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
            except (ValueError, OSError):
                pass
        # raw_message keeps sender + timestamp_ms so the processing hooks can build
        # sendReaction targets.
        event = MessageEvent(
            source=source,
            text=text or "",
            message_type=msg_type,
            media_urls=media_urls,
            media_types=media_types,
            timestamp=timestamp,
            raw_message={
                "sender": sender,
                "timestamp_ms": ts_ms,
                "quote": quote_data if quote_data else None,
            },
            reply_to_message_id=reply_to_id,
            reply_to_text=quote_data.get("text"),
            reply_to_author_id=reply_to_author,
            reply_to_author_name=quote_data.get("authorName") or quote_data.get("authorProfileName"),
            reply_to_is_own_message=self._quote_references_own_message(reply_to_id, reply_to_author),
        )
        logger.debug("Signal: message from %s in %s: %s",
                      redact_phone(sender), chat_id[:20], (text or "")[:50])
        await self.handle_message(event)

    def _remember_recipient_identifiers(self, number: Optional[str], service_id: Optional[str]) -> None:
        """Cache any number↔UUID mapping observed from Signal envelopes."""
        if not number or not service_id or not _is_signal_service_id(service_id):
            return
        self._recipient_uuid_by_number[number] = service_id
        self._recipient_number_by_uuid[service_id] = number

    @staticmethod
    def _extract_quote_author(quote_data: Any) -> Optional[str]:
        """Return the best available Signal sender identifier from quote metadata."""
        if not isinstance(quote_data, dict):
            return None
        for key in _QUOTE_AUTHOR_KEYS:
            value = quote_data.get(key)
            if value:
                return str(value)
        return None

    def _quote_references_own_message(self, reply_to_id: Optional[str], reply_to_author: Optional[str]) -> bool:
        """True when a Signal quote points at this adapter's outbound message."""
        if reply_to_id and str(reply_to_id) in self._sent_message_timestamps:
            return True
        if not reply_to_author:
            return False
        author = str(reply_to_author).strip()
        if self._account_normalized and author == self._account_normalized:
            return True
        cached_uuid = self._recipient_uuid_by_number.get(self._account_normalized)
        if cached_uuid and author == cached_uuid:
            return True
        cached_number = self._recipient_number_by_uuid.get(author)
        return bool(cached_number and cached_number == self._account_normalized)

    def _remember_sent_message_timestamp(self, timestamp: Any) -> None:
        """Keep a bounded cache of outbound Signal timestamps for quote matching."""
        if timestamp is None:
            return
        key = str(timestamp)
        # Re-insert to mark most-recently-used so eviction drops genuinely old entries.
        self._sent_message_timestamps.pop(key, None)
        self._sent_message_timestamps[key] = None
        while len(self._sent_message_timestamps) > self._max_sent_message_timestamps:
            self._sent_message_timestamps.popitem(last=False)

    def _extract_contact_uuid(self, contact: Any, phone_number: str) -> Optional[str]:
        """Best-effort extraction of a Signal service ID from listContacts output."""
        if not isinstance(contact, dict):
            return None
        service_id = contact.get("uuid") or contact.get("serviceId")
        if not service_id:
            profile = contact.get("profile")
            if isinstance(profile, dict):
                service_id = profile.get("serviceId") or profile.get("uuid")
        if service_id and _is_signal_service_id(service_id) and (
            contact.get("number") == phone_number or contact.get("recipient") == phone_number
        ):
            return service_id
        return None

    async def _resolve_recipient(self, chat_id: str) -> str:
        """Return the preferred Signal recipient identifier for a direct chat."""
        if (
            not chat_id
            or chat_id.startswith("group:")
            or _is_signal_service_id(chat_id)
            or not _looks_like_e164_number(chat_id)
        ):
            return chat_id
        cached = self._recipient_uuid_by_number.get(chat_id)
        if cached:
            return cached
        async with self._recipient_cache_lock:
            cached = self._recipient_uuid_by_number.get(chat_id)
            if cached:
                return cached
            contacts = await self._rpc("listContacts", {"account": self.account, "allRecipients": True})
            if isinstance(contacts, list):
                for contact in contacts:
                    number = contact.get("number") if isinstance(contact, dict) else None
                    service_id = self._extract_contact_uuid(contact, chat_id)
                    if number and service_id:
                        self._remember_recipient_identifiers(number, service_id)
            return self._recipient_uuid_by_number.get(chat_id, chat_id)

    async def _with_target(self, params: Dict[str, Any], chat_id: str, *, resolve: bool = True) -> Dict[str, Any]:
        """Add the groupId / recipient routing key for *chat_id* to *params* (in place)."""
        if chat_id.startswith("group:"):
            params["groupId"] = chat_id[6:]
        elif resolve:
            params["recipient"] = [await self._resolve_recipient(chat_id)]
        else:
            params["recipient"] = [chat_id]
        return params

    async def _fetch_attachment(self, attachment_id: str) -> tuple:
        """Fetch an attachment via JSON-RPC and cache it. Returns (path, ext)."""
        result = await self._rpc("getAttachment", {"account": self.account, "id": attachment_id})
        if not result:
            return None, ""
        # signal-cli returns {"data": "base64..."}
        if isinstance(result, dict):
            result = result.get("data")
            if not result:
                logger.warning("Signal: attachment response missing 'data' key")
                return None, ""
        raw_data = base64.b64decode(result)
        ext = _guess_extension(raw_data)
        # Android voice notes are raw ADTS AAC, which Whisper-style STT rejects; remux
        # losslessly to .m4a. If ffmpeg is absent the raw file is cached as-is (there is
        # no downstream sniff-and-remux fallback).
        if ext == ".aac":
            remuxed: Optional[Tuple[bytes, str]] = await asyncio.to_thread(_remux_aac_to_m4a, raw_data)
            if remuxed is not None:
                raw_data, ext = remuxed
        if _is_image_ext(ext):
            path = cache_image_from_bytes(raw_data, ext)
        elif _is_audio_ext(ext):
            path = cache_audio_from_bytes(raw_data, ext)
        else:
            path = cache_document_from_bytes(raw_data, ext)
        return path, ext

    async def _rpc(
        self,
        method: str,
        params: dict,
        rpc_id: str = None,
        *,
        log_failures: bool = True,
        raise_on_rate_limit: bool = False,
        timeout: float = 30.0,
    ) -> Any:
        """Send a JSON-RPC 2.0 request to signal-cli daemon.

        ``log_failures=False`` logs failures at DEBUG (typing path: silence repeated
        NETWORK_FAILURE spam). ``raise_on_rate_limit=True`` raises ``SignalRateLimitError``
        on a 429 / RateLimitException instead of swallowing it (multi-attachment sends).
        """
        if not self.client:
            logger.warning("Signal: RPC called but client not connected")
            return None
        if rpc_id is None:
            rpc_id = f"{method}_{int(time.time() * 1000)}"
        payload = {"jsonrpc": "2.0", "method": method, "params": params, "id": rpc_id}
        fail_level = logging.WARNING if log_failures else logging.DEBUG
        try:
            resp = await self.client.post(f"{self.http_url}/api/v1/rpc", json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            if "error" in data:
                err = data["error"]
                if raise_on_rate_limit and _is_signal_rate_limit_error(err):
                    err_msg = str(err.get("message", "")) if isinstance(err, dict) else str(err)
                    raise SignalRateLimitError(err_msg, retry_after=_extract_retry_after_seconds(err))
                logger.log(fail_level, "Signal RPC error (%s): %s", method, err)
                return None
            result = data.get("result")
            if isinstance(result, dict) and raise_on_rate_limit:
                results = result.get("results")
                if isinstance(results, list):
                    for r in results:
                        if isinstance(r, dict) and r.get("type") == "RATE_LIMIT_FAILURE":
                            raise SignalRateLimitError(
                                "Rate limit exceeded for recipient", retry_after=r.get("retryAfterSeconds")
                            )
            return result
        except SignalRateLimitError:
            raise
        except Exception as e:
            logger.log(fail_level, "Signal RPC %s failed: %s", method, e)
            return None

    @staticmethod
    def _markdown_to_signal(text: str) -> tuple[str, list[str]]:
        """Backward-compatible wrapper around shared Signal formatting helper."""
        return markdown_to_signal(text)

    def format_message(self, content: str) -> str:
        """Plain-text fallback for the base-class send path; send() applies rich styles itself."""
        return content

    def _validate_send_result(self, result: Any) -> tuple[bool, Optional[str]]:
        """Validate signal-cli send response results. Returns (success, error_message)."""
        if not result or not isinstance(result, dict):
            return True, None
        results = result.get("results")
        if isinstance(results, list):
            for r in results:
                if not isinstance(r, dict):
                    continue
                rtype = r.get("type")
                if rtype and rtype != "SUCCESS":
                    return False, str(rtype)
                if "success" in r and not r.get("success"):
                    return False, str(r.get("failure") or "Recipient delivery failed")
        return True, None

    @staticmethod
    def _utf16_offsets(text: str) -> list[int]:
        """Return cumulative UTF-16 offsets for every Python character boundary."""
        return [0, *itertools.accumulate(utf16_len(char) for char in text)]

    @staticmethod
    def _styles_for_chunk(text_styles: list[str], chunk_start: int, chunk_end: int) -> list[str]:
        """Translate full-message Signal styles into a chunk-local range list."""
        adjusted: list[str] = []
        for style_string in text_styles:
            try:
                start_s, length_s, style_type = style_string.split(":", 2)
                style_start = int(start_s)
                style_end = style_start + int(length_s)
            except (TypeError, ValueError):
                logger.debug("[Signal] Ignoring malformed textStyle range: %r", style_string)
                continue
            overlap_start = max(style_start, chunk_start)
            overlap_end = min(style_end, chunk_end)
            if overlap_start < overlap_end:
                adjusted.append(f"{overlap_start - chunk_start}:{overlap_end - overlap_start}:{style_type}")
        return adjusted

    @classmethod
    def _split_signal_formatted_message(
        cls, plain_text: str, text_styles: list[str], max_length: int,
    ) -> list[tuple[str, list[str]]]:
        """Split converted Signal text into chunks, translating body ranges per chunk.

        Splitting after conversion (not before) keeps styles that cross a chunk boundary
        intact instead of leaking literal Markdown markers.
        """
        if utf16_len(plain_text) <= max_length:
            return [(plain_text, text_styles)]
        indicator_reserve = 10  # Mirrors BasePlatformAdapter.truncate_message().
        body_limit = max(1, max_length - indicator_reserve)
        offsets = cls._utf16_offsets(plain_text)
        chunks: list[tuple[str, list[str]]] = []
        start_idx = 0
        total_u16 = offsets[-1]
        while offsets[start_idx] < total_u16:
            end_budget = min(total_u16, offsets[start_idx] + body_limit)
            end_idx = start_idx + 1
            while end_idx < len(offsets) and offsets[end_idx] <= end_budget:
                end_idx += 1
            end_idx -= 1
            if end_idx <= start_idx:
                end_idx = start_idx + 1
            chunk_styles = cls._styles_for_chunk(text_styles, offsets[start_idx], offsets[end_idx])
            chunks.append((plain_text[start_idx:end_idx], chunk_styles))
            start_idx = end_idx
        if len(chunks) == 1:
            return chunks
        total = len(chunks)
        return [
            (f"{chunk_text} ({idx}/{total})", chunk_styles)
            for idx, (chunk_text, chunk_styles) in enumerate(chunks, start=1)
        ]

    async def _rpc_send(self, params: Dict[str, Any], fail_error: str) -> Tuple[Any, Optional[SendResult]]:
        """Run a ``send`` RPC, validate and track it; ``(result, None)`` or ``(None, failed SendResult)``."""
        result = await self._rpc("send", params)
        if result is None:
            return None, SendResult(success=False, error=fail_error)
        success, err_msg = self._validate_send_result(result)
        if not success:
            return None, SendResult(success=False, error=err_msg, raw_response=result)
        self._track_sent_timestamp(result)
        return result, None

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send a text message with native Signal formatting."""
        await self._stop_typing_indicator(chat_id)
        if not content or not content.strip():
            return SendResult(success=True, message_id=None)
        base_params = await self._with_target({"account": self.account}, chat_id)
        plain_message, message_styles = self._markdown_to_signal(content)
        chunks = self._split_signal_formatted_message(plain_message, message_styles, self.MAX_MESSAGE_LENGTH)
        last_result = None
        for idx, (plain_text, text_styles) in enumerate(chunks, start=1):
            params: Dict[str, Any] = dict(base_params, message=plain_text)
            if text_styles:
                if len(text_styles) == 1:
                    params["textStyle"] = text_styles[0]
                else:
                    params["textStyles"] = text_styles
            logger.info(
                "[Signal] Sending response chunk %d/%d (%d chars) to %s",
                idx, len(chunks), len(plain_text), chat_id,
            )
            last_result, err = await self._rpc_send(params, "RPC send failed")
            if err:
                return err
        # Signal has no editable message identifier; message_id=None keeps the stream
        # consumer on the non-edit fallback path.
        return SendResult(success=True, message_id=None, raw_response=last_result)

    def _track_sent_timestamp(self, rpc_result) -> None:
        """Record outbound message timestamp for echo-back filtering."""
        ts = rpc_result.get("timestamp") if isinstance(rpc_result, dict) else None
        if ts:
            self._remember_sent_message_timestamp(ts)
            now = time.monotonic()
            # Re-insert to mark as most-recently-used.
            self._recent_sent_timestamps.pop(ts, None)
            self._recent_sent_timestamps[ts] = now
            # Drop entries older than TTL first, then enforce the hard cap.
            cutoff = now - self._recent_sent_ttl_seconds
            while self._recent_sent_timestamps:
                _, oldest_at = next(iter(self._recent_sent_timestamps.items()))
                if oldest_at < cutoff:
                    self._recent_sent_timestamps.popitem(last=False)
                else:
                    break
            while len(self._recent_sent_timestamps) > self._max_recent_timestamps:
                self._recent_sent_timestamps.popitem(last=False)

    def _consume_sent_timestamp(self, ts) -> bool:
        """Pop a timestamp if it matches one we sent. Returns True on echo."""
        if ts and ts in self._recent_sent_timestamps:
            self._recent_sent_timestamps.pop(ts, None)
            return True
        return False

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        """Send a typing indicator (called every ~2s by base.py's ``_keep_typing``).

        On NETWORK_FAILURE only the first consecutive failure logs at WARNING, and after
        three failures the RPC is skipped for an exponential cooldown; success resets.
        """
        now = time.monotonic()
        if now < self._typing_skip_until.get(chat_id, 0.0):
            return
        params = await self._with_target({"account": self.account}, chat_id)
        fails = self._typing_failures.get(chat_id, 0)
        result = await self._rpc("sendTyping", params, rpc_id="typing", log_failures=(fails == 0))
        if result is None:
            fails += 1
            self._typing_failures[chat_id] = fails
            # After 3 consecutive failures back off exponentially (16s, 32s, 60s cap).
            if fails >= 3:
                self._typing_skip_until[chat_id] = now + min(60.0, 16.0 * (2 ** (fails - 3)))
        else:
            self._typing_failures.pop(chat_id, None)
            self._typing_skip_until.pop(chat_id, None)

    async def _resolve_image_path(self, image_url: str) -> Tuple[Optional[str], Optional[str], Any]:
        """Resolve an http(s):// or file:// image URL to ``(path, None, None)``, or
        ``(None, reason, detail)`` with reason download (exc) / missing / oversize (size)."""
        if image_url.startswith("file://"):
            file_path = unquote(image_url[7:])
        else:
            try:
                file_path = await cache_image_from_url(image_url)
            except Exception as e:
                return None, "download", e
        if not file_path or not Path(file_path).exists():
            return None, "missing", None
        file_size = Path(file_path).stat().st_size
        if file_size > SIGNAL_MAX_ATTACHMENT_SIZE:
            return None, "oversize", file_size
        return file_path, None, None

    async def send_multiple_images(
        self,
        chat_id: str,
        images: List[Tuple[str, str]],
        metadata: Optional[Dict[str, Any]] = None,
        human_delay: float = 0.0,
    ) -> None:
        """Send a batch of images via chunked Signal RPC calls.

        Alt texts are dropped (one shared body per send). Bad images are skipped with a
        warning. ``human_delay`` is ignored: the rate-limit scheduler paces batches.
        """
        if not images:
            return
        scheduler = get_scheduler()
        logger.info(
            "Signal send_multiple_images: received %d image(s) for %s — scheduler state: %s",
            len(images), chat_id[:30], scheduler.state(),
        )
        await self._stop_typing_indicator(chat_id)
        attachments: List[str] = []
        skipped = {"download": 0, "missing": 0, "oversize": 0}
        for image_url, _alt_text in images:
            file_path, reason, detail = await self._resolve_image_path(image_url)
            if reason == "download":
                logger.warning("Signal: failed to download image %s: %s", image_url, detail)
            elif reason == "missing":
                logger.warning("Signal: image file not found for %s", image_url)
            elif reason == "oversize":
                logger.warning("Signal: image too large (%d bytes), skipping %s", detail, image_url)
            if reason:
                skipped[reason] += 1
                continue
            attachments.append(file_path)
        if not attachments:
            logger.error(
                "Signal: no valid images in batch of %d (download=%d missing=%d oversize=%d)",
                len(images), skipped["download"], skipped["missing"], skipped["oversize"],
            )
            return
        logger.info(
            "Signal send_multiple_images: %d/%d images valid, sending in chunks",
            len(attachments), len(images),
        )
        base_params = await self._with_target({"account": self.account, "message": ""}, chat_id)
        att_batches = [
            attachments[i:i + SIGNAL_MAX_ATTACHMENTS_PER_MSG]
            for i in range(0, len(attachments), SIGNAL_MAX_ATTACHMENTS_PER_MSG)
        ]
        n_batches = len(att_batches)
        for idx, att_batch in enumerate(att_batches, start=1):
            n = len(att_batch)
            estimated = scheduler.estimate_wait(n)
            logger.debug("Signal batch %d/%d: %d attachments, estimated wait=%.1fs", idx, n_batches, n, estimated)
            if estimated >= SIGNAL_BATCH_PACING_NOTICE_THRESHOLD:
                await self._notify_batch_pacing(chat_id, idx, n_batches, estimated)
            await self._send_attachment_batch(
                scheduler, dict(base_params, attachments=att_batch), n, f"{idx}/{n_batches}",
            )

    async def _send_attachment_batch(self, scheduler, params: Dict[str, Any], n: int, label: str) -> None:
        """Send one attachment batch with rate-limit pacing and a single transient retry.

        Tokens are deducted only on validated success (a None result means the server
        never accepted the batch); 429s feed the scheduler before the retry.
        """
        send_timeout = _signal_send_timeout(n)
        for attempt in range(1, SIGNAL_RATE_LIMIT_MAX_ATTEMPTS + 1):
            await scheduler.acquire(n)
            try:
                _rpc_t0 = time.monotonic()
                result = await self._rpc("send", params, raise_on_rate_limit=True, timeout=send_timeout)
                _rpc_duration = time.monotonic() - _rpc_t0
                success, err_msg = self._validate_send_result(result) if result is not None else (False, None)
                if success:
                    self._track_sent_timestamp(result)
                    await scheduler.report_rpc_duration(_rpc_duration, n)
                    logger.info(
                        "Signal batch %s: %d attachments sent in %.1fs (attempt %d/%d)",
                        label, n, _rpc_duration, attempt, SIGNAL_RATE_LIMIT_MAX_ATTEMPTS,
                    )
                    return
                logger.error(
                    "Signal: RPC send failed for batch %s (%d attachments, attempt %d/%d, rpc_duration=%.1fs)%s",
                    label, n, attempt, SIGNAL_RATE_LIMIT_MAX_ATTEMPTS, _rpc_duration,
                    f": {err_msg}" if result is not None else "",
                )
                if attempt >= SIGNAL_RATE_LIMIT_MAX_ATTEMPTS:
                    return
                backoff = 2.0 ** attempt
                logger.info("Signal: retrying batch %s after %.1fs backoff", label, backoff)
                await asyncio.sleep(backoff)
            except SignalRateLimitError as e:
                scheduler.feedback(e.retry_after, n)
                retry_after = f"{e.retry_after:.0f}s" if e.retry_after else "unknown"
                if attempt >= SIGNAL_RATE_LIMIT_MAX_ATTEMPTS:
                    logger.error(
                        "Signal: rate-limit retries exhausted on batch %s (%d attachments lost, server retry_after=%s)",
                        label, n, retry_after,
                    )
                    return
                logger.warning(
                    "Signal: rate-limited on batch %s (attempt %d/%d, server retry_after=%s); "
                    "scheduler will pace the retry",
                    label, attempt, SIGNAL_RATE_LIMIT_MAX_ATTEMPTS, retry_after,
                )

    async def _notify_batch_pacing(self, chat_id: str, next_batch_idx: int, total_batches: int, wait_s: float) -> None:
        """Tell the user about an inter-batch pacing wait over the notice threshold (best-effort)."""
        try:
            await self.send(
                chat_id,
                f"(More images coming — pausing ~{_format_wait(wait_s)} "
                f"for Signal rate limit, batch {next_batch_idx}/{total_batches}.)",
            )
        except Exception as e:
            logger.warning("Signal: failed to send pacing notice: %s", e)

    async def send_image(self, chat_id: str, image_url: str, caption: Optional[str] = None, **kwargs) -> SendResult:
        """Send an image. Supports http(s):// and file:// URLs."""
        await self._stop_typing_indicator(chat_id)
        file_path, reason, detail = await self._resolve_image_path(image_url)
        if reason == "download":
            logger.warning("Signal: failed to download image: %s", detail)
            return SendResult(success=False, error=str(detail))
        if reason == "missing":
            return SendResult(success=False, error="Image file not found")
        if reason == "oversize":
            return SendResult(success=False, error=f"Image too large ({detail} bytes)")
        return await self._send_file(chat_id, file_path, caption, "RPC send with attachment failed")

    async def _send_file(self, chat_id: str, file_path: str, caption: Optional[str], fail_error: str) -> SendResult:
        """Send one local file as a Signal attachment via the ``send`` RPC."""
        params = await self._with_target(
            {"account": self.account, "message": caption or "", "attachments": [file_path]}, chat_id
        )
        _, err = await self._rpc_send(params, fail_error)
        return err or SendResult(success=True)

    async def _send_attachment(
        self, chat_id: str, file_path: str, media_label: str, caption: Optional[str] = None,
    ) -> SendResult:
        """Send any local file as a Signal attachment (shared by send_document/image_file/voice/video)."""
        await self._stop_typing_indicator(chat_id)
        try:
            file_size = Path(file_path).stat().st_size
        except FileNotFoundError:
            return SendResult(success=False, error=f"{media_label} file not found: {file_path}")
        if file_size > SIGNAL_MAX_ATTACHMENT_SIZE:
            return SendResult(success=False, error=f"{media_label} too large ({file_size} bytes)")
        return await self._send_file(chat_id, file_path, caption, f"RPC send {media_label.lower()} failed")

    async def send_document(
        self, chat_id: str, file_path: str, caption: Optional[str] = None, filename: Optional[str] = None, **kwargs,
    ) -> SendResult:
        """Send a document/file attachment."""
        return await self._send_attachment(chat_id, file_path, "File", caption)

    async def send_image_file(
        self, chat_id: str, image_path: str, caption: Optional[str] = None, reply_to: Optional[str] = None, **kwargs,
    ) -> SendResult:
        """Send a local image file as a native Signal attachment (gateway MEDIA: delivery path)."""
        return await self._send_attachment(chat_id, image_path, "Image", caption)

    async def send_voice(
        self, chat_id: str, audio_path: str, caption: Optional[str] = None, reply_to: Optional[str] = None, **kwargs,
    ) -> SendResult:
        """Send an audio file as a Signal attachment (Signal has no distinct voice-message API)."""
        return await self._send_attachment(chat_id, audio_path, "Audio", caption)

    async def send_video(
        self, chat_id: str, video_path: str, caption: Optional[str] = None, reply_to: Optional[str] = None, **kwargs,
    ) -> SendResult:
        """Send a video file as a Signal attachment."""
        return await self._send_attachment(chat_id, video_path, "Video", caption)

    async def _stop_typing_indicator(self, chat_id: str) -> None:
        """Stop a typing indicator loop for a chat."""
        await self._cancel_task(self._typing_tasks.pop(chat_id, None))
        # Explicit stop-typing RPC so the recipient drops the indicator now instead of
        # after Signal's ~5s built-in timeout. Best-effort: any RPC or recipient-
        # resolution failure must not prevent the backoff cleanup below.
        try:
            params = await self._with_target({"account": self.account}, chat_id)
            params["stop"] = True
            await self._rpc("sendTyping", params, rpc_id="typing-stop", log_failures=False)
        except Exception:
            pass
        self._typing_failures.pop(chat_id, None)
        self._typing_skip_until.pop(chat_id, None)

    async def stop_typing(self, chat_id: str) -> None:
        """Public stop-typing hook called from the base adapter's _keep_typing finally block."""
        await self._stop_typing_indicator(chat_id)

    async def _send_reaction_rpc(self, chat_id: str, params: Dict[str, Any]) -> bool:
        """Route a ``sendReaction`` RPC to *chat_id* (no UUID upgrade — author IDs come from the envelope)."""
        await self._with_target(params, chat_id, resolve=False)
        return await self._rpc("sendReaction", params) is not None

    async def send_reaction(self, chat_id: str, emoji: str, target_author: str, target_timestamp: int) -> bool:
        """React to the message (author number/UUID, Signal ms timestamp) via signal-cli RPC."""
        ok = await self._send_reaction_rpc(chat_id, {
            "account": self.account, "emoji": emoji, "targetAuthor": target_author,
            "targetTimestamp": target_timestamp,
        })
        if not ok:
            logger.debug("Signal: sendReaction failed (chat=%s, emoji=%s)", chat_id[:20], emoji)
        return ok

    async def remove_reaction(self, chat_id: str, target_author: str, target_timestamp: int) -> bool:
        """Remove a reaction by sending an empty-string emoji."""
        return await self._send_reaction_rpc(chat_id, {
            "account": self.account, "emoji": "", "targetAuthor": target_author,
            "targetTimestamp": target_timestamp, "remove": True,
        })

    def _extract_reaction_target(self, event: MessageEvent) -> Optional[tuple]:
        """Extract (target_author, target_timestamp) from a MessageEvent, or None."""
        raw = event.raw_message
        if isinstance(raw, dict) and raw.get("sender") and raw.get("timestamp_ms"):
            return (raw["sender"], raw["timestamp_ms"])
        return None

    def _reactions_enabled(self, event: "MessageEvent" = None) -> bool:
        """SIGNAL_REACTIONS env gate, then the DM allowlist: reactions fire before run.py's
        auth gate, so an unauthorized contact's 👀 would otherwise reveal a listening bot."""
        if os.getenv("SIGNAL_REACTIONS", "true").lower() in {"false", "0", "no"}:
            return False
        if event is not None:
            sender = getattr(getattr(event, "source", None), "user_id", None)
            if sender and "*" not in self.dm_allow_from and sender not in self.dm_allow_from:
                return False
        return True

    async def on_processing_start(self, event: MessageEvent) -> None:
        """React with 👀 when processing begins."""
        if not self._reactions_enabled(event):
            return
        target = self._extract_reaction_target(event)
        if target:
            await self.send_reaction(event.source.chat_id, "👀", *target)

    async def on_processing_complete(self, event: MessageEvent, outcome: "ProcessingOutcome") -> None:
        """Swap 👀 for ✅/❌; on CANCELLED the 👀 stays to keep reflecting "in progress" (matches Telegram)."""
        if not self._reactions_enabled(event):
            return
        if outcome == ProcessingOutcome.CANCELLED:
            return
        target = self._extract_reaction_target(event)
        if not target:
            return
        chat_id = event.source.chat_id
        await self.remove_reaction(chat_id, *target)
        emoji = _OUTCOME_REACTION.get(outcome)
        if emoji:
            await self.send_reaction(chat_id, emoji, *target)

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Get information about a chat/contact."""
        if chat_id.startswith("group:"):
            return {"name": chat_id, "type": "group", "chat_id": chat_id}
        result = await self._rpc("getContact", {"account": self.account, "contactAddress": chat_id})
        name = chat_id
        if result and isinstance(result, dict):
            name = result.get("name") or result.get("profileName") or chat_id
        return {"name": name, "type": "dm", "chat_id": chat_id}
