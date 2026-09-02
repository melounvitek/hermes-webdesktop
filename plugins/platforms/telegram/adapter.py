"""
Telegram platform adapter.

Uses python-telegram-bot library for:
- Receiving messages from users/groups
- Sending responses back
- Handling media and commands
"""

import asyncio
import dataclasses
import inspect
import json
import logging
import os
import html as _html
import re
import time
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, Iterator, List, Optional, Set

logger = logging.getLogger(__name__)

from agent.deadline import run_bounded_async


def _redact_telegram_error_text(error: object) -> str:
    """Redact secrets from Telegram transport errors before logging or returning them."""
    text = "" if error is None else str(error)
    if not text:
        return text
    try:
        from agent.redact import redact_sensitive_text

        return redact_sensitive_text(text, force=True)
    except Exception:
        return "<telegram error redacted>"


def _scoped_gate_env(name: str, default: str = "") -> str:
    """Read a TELEGRAM_*/GATEWAY_* gate env var per-profile.

    Under multiplex_profiles the process env is first-writer-wins, so raw ``os.getenv`` can
    return ANOTHER profile's allowlist. Falls back to ``os.getenv`` outside multiplex.
    """
    try:
        from gateway.authz_mixin import _platform_gate_env

        return _platform_gate_env(name, default)
    except Exception:
        return (os.getenv(name) or default).strip()


def _consume_abandoned_task(task: asyncio.Task) -> None:
    """Observe a detached task's terminal exception to avoid noisy loop logs."""
    try:
        task.exception()
    except asyncio.CancelledError:
        pass
    except Exception:
        logger.debug("Abandoned Telegram init task failed after timeout", exc_info=True)


async def _await_with_thread_deadline(awaitable, timeout: float, *, on_abandon=None):
    """Await with a wall-clock (thread-timer) deadline that survives a blocked event loop.

    Wrapper over ``run_bounded_async``: abandons cancellation-shielded tasks (PTB/httpcore
    init inside anyio scopes) and runs ``on_abandon`` detached so an abandoned initialize()
    can't leak an httpx pool. Raises ``asyncio.TimeoutError`` on expiry (PTB retry ladder).
    """
    result = await run_bounded_async(
        awaitable, timeout, label="telegram-init", on_abandon=on_abandon
    )
    if result.timed_out:
        raise asyncio.TimeoutError()
    return result.value


def _iter_exception_graph(error: BaseException) -> "Iterator[BaseException]":
    """Yield ``error`` and every ``__cause__``/``__context__`` ancestor (DFS, cycle-safe).

    PTB wraps httpx exceptions and re-raises accumulate ``__context__`` chains, so
    classifiers must inspect the whole graph, not just the top frame.
    """
    seen: set[int] = set()
    stack: list[BaseException] = [error]
    while stack:
        cur = stack.pop()
        ident = id(cur)
        if ident in seen:
            continue
        seen.add(ident)
        yield cur
        cause = getattr(cur, "__cause__", None)
        context = getattr(cur, "__context__", None)
        if cause is not None:
            stack.append(cause)
        if context is not None:
            stack.append(context)


async def _first_completed(*futures: "asyncio.Future") -> None:
    """Return when the first of ``futures`` completes; losers are NOT cancelled."""
    await asyncio.wait(set(futures), return_when=asyncio.FIRST_COMPLETED)


async def _shutdown_abandoned_app(app) -> None:
    """Release a half-built PTB app's httpx transports after init was abandoned.

    ``app.shutdown()`` is gated on ``_initialized`` flags a wedged ``initialize()`` never set,
    so it no-ops and leaks the pool. ``HTTPXRequest`` builds its client eagerly and its
    ``shutdown()`` gates only on ``client.is_closed``, so we also close transports directly.
    """
    if app is None:
        return
    try:
        await app.shutdown()
    except Exception:
        logger.debug("Abandoned Telegram app.shutdown() failed", exc_info=True)
    bot = getattr(app, "bot", None)
    requests = getattr(bot, "_request", None) if bot is not None else None
    if not requests:
        return
    for request in requests:
        shutdown = getattr(request, "shutdown", None)
        if shutdown is None:
            continue
        try:
            result = shutdown()
            if asyncio.iscoroutine(result) or asyncio.isfuture(result):
                await result
        except Exception:
            logger.debug("Abandoned Telegram request shutdown failed", exc_info=True)

try:
    from telegram import Update, Bot, Message, InlineKeyboardButton, InlineKeyboardMarkup
    try:
        from telegram import LinkPreviewOptions
    except ImportError:
        LinkPreviewOptions = None
    from telegram.ext import (
        Application, CommandHandler, CallbackQueryHandler, InlineQueryHandler,
        MessageHandler as TelegramMessageHandler, ContextTypes, TypeHandler, filters,
    )
    from telegram.constants import ParseMode, ChatType
    from telegram.request import HTTPXRequest
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    Update = Any
    Bot = Any
    Message = Any
    InlineKeyboardButton = Any
    InlineKeyboardMarkup = Any
    LinkPreviewOptions = None
    Application = Any
    CommandHandler = Any
    CallbackQueryHandler = Any
    InlineQueryHandler = Any
    TypeHandler = Any
    TelegramMessageHandler = Any
    HTTPXRequest = Any
    filters = None
    ParseMode = None
    ChatType = None

    # Mock so ContextTypes.DEFAULT_TYPE annotations don't crash class definition without the lib.
    class _MockContextTypes:
        DEFAULT_TYPE = Any
    ContextTypes = _MockContextTypes

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))

from gateway.authz_mixin import _coerce_allow_set
from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter, MessageEvent, MessageType, ProcessingOutcome, SendResult,
    classify_send_error, cache_image_from_bytes, cache_audio_from_bytes, cache_video_from_bytes,
    resolve_proxy_url, SUPPORTED_VIDEO_TYPES, SUPPORTED_DOCUMENT_TYPES,
    SUPPORTED_IMAGE_DOCUMENT_TYPES, _TEXT_INJECT_EXTENSIONS, utf16_len,
)
from plugins.platforms.telegram.telegram_ids import normalize_telegram_chat_id
from plugins.platforms.telegram.telegram_network import (
    SEED_FALLBACK_IPS, TelegramFallbackTransport, discover_fallback_ips, parse_fallback_ip_env,
    tcp_keepalive_socket_options,
)
from utils import env_float, env_int

_TELEGRAM_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}

# Max seconds a send/edit may sleep inline on a flood-control RetryAfter. Longer penalties
# fail closed with a ``flood_control:{wait}`` SendResult so the caller's retry machinery
# (delivery ledger, streaming fallback) owns the wait instead of pinning a worker.
_FLOOD_INLINE_WAIT_CAP_SECS = 5.0


def _flood_cap_result(wait: float) -> "SendResult":
    """The shared fail-closed SendResult for an over-cap flood wait."""
    return SendResult(success=False, error=f"flood_control:{wait}", retry_after=float(wait))


_TELEGRAM_IMAGE_MIME_TO_EXT = {
    "image/png": ".png", "image/jpeg": ".jpg", "image/jpg": ".jpg", "image/webp": ".webp",
    "image/gif": ".gif",
}
_TELEGRAM_IMAGE_EXT_TO_MIME = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp",
    ".gif": "image/gif",
}


def _coerce_duration_seconds(value: Any) -> Optional[int]:
    """Round a raw length to whole positive seconds, or None if unusable."""
    try:
        secs = int(round(float(value)))
    except (TypeError, ValueError):
        return None
    return secs if secs > 0 else None


def _probe_voice_duration_seconds(path: str) -> Optional[int]:
    """Best-effort audio length in whole seconds for outgoing voice/audio (None if unreadable).

    Telegram only derives duration from metadata for short clips; longer ones render as 0:00,
    so we pass it explicitly. Tries wave (WAV), mutagen, then ffprobe — mirrors
    ``gateway.run._probe_audio_duration``. Blocking: call via ``asyncio.to_thread``.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".wav":
        try:
            import wave
            with wave.open(path, "rb") as wf:
                rate = wf.getframerate() or 0
                if rate:
                    secs = _coerce_duration_seconds(wf.getnframes() / float(rate))
                    if secs is not None:
                        return secs
        except Exception:
            pass

    try:
        import mutagen
        audio = mutagen.File(path)
        secs = _coerce_duration_seconds(getattr(getattr(audio, "info", None), "length", None))
        if secs is not None:
            return secs
    except Exception:
        pass

    try:
        import shutil
        import subprocess
        if shutil.which("ffprobe"):
            proc = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", path],
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=5,
            )
            if proc.returncode == 0:
                return _coerce_duration_seconds(proc.stdout.strip())
    except Exception:
        pass
    return None


def telegram_deps_present() -> bool:
    """PASSIVE probe: is python-telegram-bot importable? Must never install anything.

    Registry ``check_fn`` (status displays, config loading). The ACTIVE lazy-installer is
    ``check_telegram_requirements`` (``ensure_deps_fn``, run from ``create_adapter()``).
    """
    return TELEGRAM_AVAILABLE


def check_telegram_requirements() -> bool:
    """Lazy-install python-telegram-bot if missing, then re-import and rebind the module aliases."""
    global TELEGRAM_AVAILABLE, Update, Bot, Message, InlineKeyboardButton
    global InlineKeyboardMarkup, LinkPreviewOptions, Application
    global CommandHandler, CallbackQueryHandler, InlineQueryHandler, TelegramMessageHandler
    global ContextTypes, filters, ParseMode, ChatType, HTTPXRequest, TypeHandler
    if TELEGRAM_AVAILABLE:
        return True
    try:
        from tools.lazy_deps import ensure as _lazy_ensure
        _lazy_ensure("platform.telegram", prompt=False)
    except Exception:
        return False
    try:
        from telegram import Update as _Update, Bot as _Bot, Message as _Message
        from telegram import InlineKeyboardButton as _IKB, InlineKeyboardMarkup as _IKM
        try:
            from telegram import LinkPreviewOptions as _LPO
        except ImportError:
            _LPO = None
        from telegram.ext import (
            Application as _App, CommandHandler as _CH, CallbackQueryHandler as _CQH,
            InlineQueryHandler as _IQH, MessageHandler as _MH, ContextTypes as _CT,
            filters as _filters, TypeHandler as _TH,
        )
        from telegram.constants import ParseMode as _PM, ChatType as _CtT
        from telegram.request import HTTPXRequest as _HR
    except ImportError:
        return False
    Update = _Update
    Bot = _Bot
    Message = _Message
    InlineKeyboardButton = _IKB
    InlineKeyboardMarkup = _IKM
    LinkPreviewOptions = _LPO
    Application = _App
    CommandHandler = _CH
    CallbackQueryHandler = _CQH
    InlineQueryHandler = _IQH
    TelegramMessageHandler = _MH
    ContextTypes = _CT
    filters = _filters
    ParseMode = _PM
    ChatType = _CtT
    HTTPXRequest = _HR
    TypeHandler = _TH
    TELEGRAM_AVAILABLE = True
    return True


# Every char MarkdownV2 requires backslash-escaped outside code spans/fences.
_MDV2_ESCAPE_RE = re.compile(r'([_*\[\]()~`>#\+\-=|{}.!\\])')


def _escape_mdv2(text: str) -> str:
    """Escape Telegram MarkdownV2 special characters with a preceding backslash."""
    return _MDV2_ESCAPE_RE.sub(r'\\\1', text)


def _strip_mdv2(text: str) -> str:
    """Strip MarkdownV2 escapes and formatting markers for the plain-text fallback."""
    cleaned = re.sub(r'\\([_*\[\]()~`>#\+\-=|{}.!\\])', r'\1', text)  # escape backslashes
    cleaned = re.sub(r'\*\*([^*]+)\*\*', r'\1', cleaned)  # **bold** BEFORE MarkdownV2 *bold*
    cleaned = re.sub(r'\*([^*]+)\*', r'\1', cleaned)
    # italic: word-boundary guarded so snake_case like my_variable_name survives
    cleaned = re.sub(r'(?<!\w)_([^_]+)_(?!\w)', r'\1', cleaned)
    cleaned = re.sub(r'~([^~]+)~', r'\1', cleaned)  # strikethrough
    cleaned = re.sub(r'\|\|([^|]+)\|\|', r'\1', cleaned)  # spoiler
    return cleaned


_CHUNK_INDICATOR_ON_FENCE_RE = re.compile(r'(?m)^``` (?P<indicator>(?:\\)?\(\d+/\d+(?:\\)?\))$')


def _separate_chunk_indicator_from_fence(text: str) -> str:
    """Move ``(N/M)`` chunk markers onto their own line after a closing code fence.

    ``truncate_message()`` appends the indicator to a chunk that may end with a synthesized
    closing fence, yielding ````` \\(1/2\\)`` — Telegram rejects that as a fence and falls
    back to plain text.
    """
    return _CHUNK_INDICATOR_ON_FENCE_RE.sub(r'```\n\g<indicator>', text)


# MarkdownV2 has no table syntax ('|' is just an escaped literal), so pipe tables are
# converted to bullet groups by the shared convert_table_to_bullets().
from gateway.platforms.helpers import (
    TABLE_SEPARATOR_RE as _TABLE_SEPARATOR_RE,
    compile_mention_patterns,
    convert_table_to_bullets as _wrap_markdown_tables,
)


# Rich-message newline normalization. Protected regions whose internal newlines must stay
# bare: fenced code blocks OR GFM pipe-table blocks (header row, delimiter row, data rows).
# Telegram renders both natively; injected hard breaks would corrupt them.
_RICH_PROTECTED_REGION_RE = re.compile(
    r'(?:```[^\n]*\n[\s\S]*?```)'                       # fenced code block
    r'|(?:^[^\n]*\|[^\n]*\n'                            # table header row (has a pipe)
    r'[ \t]*\|?[ \t]*:?-+:?[ \t]*(?:\|[ \t]*:?-+:?[ \t]*)+\|?[ \t]*'  # delimiter
    r'(?:\n[^\n]*\|[^\n]*)*)',                          # data rows (newline-led, trailing \n left for prose)
    re.MULTILINE,
)


def _rich_normalize_linebreaks(text: str) -> str:
    """Convert single ``\\n`` to Markdown hard breaks (two trailing spaces) for sendRichMessage.

    Markdown treats a lone ``\\n`` as a soft break, collapsing multi-line content into one
    paragraph. ``\\n\\n``, fenced code and pipe-table blocks are left untouched.
    """
    if not text or '\n' not in text:
        return text
    out: list[str] = []
    # Inject hard breaks only in the prose between protected regions.
    pos = 0
    for m in _RICH_PROTECTED_REGION_RE.finditer(text):
        prose = text[pos:m.start()]
        out.append(re.sub(r'(?<!\n)\n(?!\n)', '  \n', prose))
        out.append(m.group(0))  # protected region kept verbatim
        pos = m.end()
    tail = text[pos:]
    out.append(re.sub(r'(?<!\n)\n(?!\n)', '  \n', tail))
    return ''.join(out)


# Internal safety bounds (not user knobs) so no reconnect/teardown path can hang on a dead
# CLOSE-WAIT socket that PTB's polling task is blocked on in epoll and never wakes from.
_UPDATER_STOP_TIMEOUT = 15.0  # `await updater.stop()`, applied identically at every site
# Other disconnect() steps: short, so a cancellation-swallowing PTB close can't burn the
# gateway's fatal-handler budget before the reconnect queue is useful.
_DISCONNECT_STEP_TIMEOUT = 2.0
# start_polling() can hang on a degraded pool after _drain_polling_connections() (both
# primary and fallback endpoints unreachable); bound it so the heartbeat can recover.
_UPDATER_START_TIMEOUT = 30.0
# Initial connect is unhealthy until getUpdates completes one round trip. Unlike reconnect,
# bootstrap must fail closed so GatewayRunner disposes the adapter and retries fresh.
_INITIAL_POLLING_PROGRESS_TIMEOUT = 60.0
# shutdown()/initialize() on the getUpdates request rebuild the pool; a wedged CLOSE-WAIT
# socket can block that forever, freezing _polling_error_task and gating every escalation
# path behind its in-flight guard. Bound the drain so the ladder reaches fatal-restart.
_DRAIN_TIMEOUT = 15.0
# Cause-agnostic wedged-recovery watchdog: every recovery path gates on
# ``_polling_error_task.done()``, so a task wedged on an unbounded await leaves the gateway
# silently deaf. Healthy worst case is stop + 2x drain + start + 60s backoff ≈ 135s, so
# 300s in flight is unambiguously stuck and the heartbeat force-escalates.
_POLLING_ERROR_TASK_STUCK_TIMEOUT = 300.0
# A generation is unhealthy until getUpdates returns successfully; exceeds one idle long-poll.
_POLLING_PROGRESS_TIMEOUT = 60.0
# Telegram answers a long-poll within ~50s, so no round-trip for ~3x that — while get_me()
# is healthy and nothing is queued server-side — means the consumer is wedged on a socket
# that never raises (CLOSE-WAIT behind a TUN/proxy route flip) and no other probe sees it.
_POLLING_STALL_TIMEOUT = 150.0
# Telegram transcodes video before answering sendVideo, outlasting the 20s read timeout the
# rest of the Bot API uses. Only media sends get this budget; kept modest because it is also
# how long a user waits to hear the attachment failed.
_MEDIA_SEND_READ_TIMEOUT = 60.0
_POLLING_GENERATION_CONTEXT: ContextVar[Optional[int]] = ContextVar(
    "telegram_polling_generation", default=None
)


class _PollingLifecycleAbort(RuntimeError):
    """Internal control flow for polling startup fenced by teardown."""


class TelegramAdapter(BasePlatformAdapter):
    """Telegram bot adapter: users/groups, MarkdownV2 replies, forum topics, media."""

    MAX_MESSAGE_LENGTH = 4096
    supports_code_blocks = True  # MarkdownV2 renders fenced code blocks
    splits_long_messages = True  # send() chunks via truncate_message(MAX_MESSAGE_LENGTH)
    # Bot API 10.1 Rich Messages cap raw text at 32,768 chars; above that use legacy chunking.
    RICH_MESSAGE_MAX_CHARS = 32768
    # Chunk near this length ⇒ a Telegram client-side split continuation is almost certain.
    _SPLIT_THRESHOLD = 4000
    MEDIA_GROUP_WAIT_SECONDS = 0.8
    # Cap on inbound events held across a disconnect window; oldest dropped first.
    HELD_INBOUND_MAX = 64
    _GENERAL_TOPIC_THREAD_ID = "1"
    # send() can race a disconnect/reconnect blip; failing "Not connected" (retryable=False)
    # parks the answer in the delivery ledger until next boot. Wait briefly for _bot (or a
    # replacement adapter) instead. Same idea as QQBot._wait_for_reconnection.
    _RECONNECT_WAIT_SECONDS = 15.0
    _RECONNECT_POLL_INTERVAL = 0.5

    # edit_message applies MarkdownV2 only on the finalize=True path; without this flag
    # stream_consumer._send_or_edit skips the final edit when raw text is unchanged.
    REQUIRES_EDIT_FINALIZE: bool = True
    # Retrying a turn-final edit burns the same flood budget while the answer sits undelivered.
    FALLBACK_ON_FINAL_EDIT_FLOOD: bool = True
    # A failed final edit can leave clients with a partial/non-durable preview; resend fresh.
    RESEND_FINAL_ON_EMPTY_STREAM_FALLBACK: bool = True

    # Adaptive text-batch ingress, tuned for "feels instant": ≤320 codepoints settle in
    # ~180ms, ≤1024 in ~240ms, longer waits the configured cap. Always clamped to
    # ``_text_batch_delay_seconds`` so an operator can lower the cap via env var.
    _TEXT_BATCH_FAST_LEN = 320
    _TEXT_BATCH_FAST_DELAY_S = 0.18
    _TEXT_BATCH_SHORT_LEN = 1024
    _TEXT_BATCH_SHORT_DELAY_S = 0.24

    @staticmethod
    def _env_float_clamped(
        name: str, default: float, *,
        min_value: Optional[float] = None, max_value: Optional[float] = None,
    ) -> float:
        """Read a float env var; non-finite → default; clamp to bounds (safe for asyncio.sleep)."""
        import math
        raw = os.getenv(name)
        try:
            value = float(raw) if raw is not None else float(default)
        except (TypeError, ValueError):
            value = float(default)
        if not math.isfinite(value):
            value = float(default)
        if min_value is not None:
            value = max(value, min_value)
        if max_value is not None:
            value = min(value, max_value)
        return value

    @property
    def _teardown_started(self) -> bool:
        """True once disconnect() fenced polling (tolerates object.__new__ test adapters)."""
        return getattr(self, "_polling_teardown_started", False)

    @property
    def message_len_fn(self):
        """Telegram measures message length in UTF-16 code units."""
        return utf16_len

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.TELEGRAM)
        self._app: Optional[Application] = None
        self._bot: Optional[Bot] = None
        self._webhook_mode: bool = False
        self._mention_patterns = self._compile_mention_patterns()
        self._reply_to_mode: str = getattr(config, 'reply_to_mode', 'first') or 'first'
        self._disable_link_previews: bool = self._coerce_bool_extra("disable_link_previews", False)
        # Bot API 10.1 Rich Messages (sendRichMessage / rich_message param) render constructs
        # MarkdownV2 degrades (tables, task lists, <details>, block math). Keep opt-in: current
        # clients make rich messages hard to copy as plain text, which is worse for command
        # snippets and mobile handoffs. Enable via platforms.telegram.extra.rich_messages.
        self._rich_messages_enabled: bool = self._coerce_bool_extra("rich_messages", False)
        # Separate opt-in: macOS/Desktop can leave rich draft frames overlaid until redraw.
        # rich_messages on + rich_drafts off keeps native draft *transport* and only skips
        # rich draft *rendering*; the final reply still lands via sendRichMessage.
        self._rich_drafts_enabled: bool = self._coerce_bool_extra("rich_drafts", False)
        # Latched off after a capability failure (e.g. older PTB without the endpoint).
        self._rich_send_disabled: bool = False
        self._rich_draft_disabled: bool = False
        # Transient sendChatAction failures recur on every keep-typing tick during a long
        # model call; back off per chat so an outage doesn't spam the API/logs.
        self._telegram_typing_cooldown_until: Dict[str, float] = {}
        self._telegram_typing_cooldown_seconds: float = self._coerce_float_extra(
            "typing_cooldown_seconds", 30.0, min_value=1.0, max_value=300.0
        )
        # Buffer album/photo bursts into a single MessageEvent instead of self-interrupting turns.
        self._media_batch_delay_seconds = env_float("HERMES_TELEGRAM_MEDIA_BATCH_DELAY_SECONDS", 0.8)
        self._pending_photo_batches: Dict[str, MessageEvent] = {}
        self._pending_photo_batch_tasks: Dict[str, asyncio.Task] = {}
        self._media_group_events: Dict[str, MessageEvent] = {}
        self._media_group_tasks: Dict[str, asyncio.Task] = {}
        # Aggregate client-side splits of long messages into one MessageEvent. Bounds are
        # conservative for Telegram's ~1 edit/s flood envelope (see _calc_text_batch_delay).
        self._text_batch_delay_seconds = self._env_float_clamped(
            "HERMES_TELEGRAM_TEXT_BATCH_DELAY_SECONDS", 0.3, min_value=0.08, max_value=2.0
        )
        self._text_batch_split_delay_seconds = self._env_float_clamped(
            "HERMES_TELEGRAM_TEXT_BATCH_SPLIT_DELAY_SECONDS", 1.0,
            min_value=self._text_batch_delay_seconds, max_value=4.0,
        )
        self._pending_text_batches: Dict[str, MessageEvent] = {}
        self._pending_text_batch_tasks: Dict[str, asyncio.Task] = {}
        self._drop_delayed_deliveries = False
        # Held across disconnect: PTB advances the polling offset before our drop-guard runs,
        # so Telegram won't redeliver — dropping is permanent loss (see _hold_inbound_event).
        self._held_inbound_events: List[MessageEvent] = []
        self._held_inbound_redispatch_task: Optional[asyncio.Task] = None
        self._polling_error_task: Optional[asyncio.Task] = None
        self._polling_conflict_count: int = 0
        self._polling_conflict_recovery_generation: Optional[int] = None
        self._polling_network_error_count: int = 0
        self._polling_generation: int = 0
        self._polling_progress_event = asyncio.Event()
        self._polling_progress_accepting: bool = False
        self._polling_progress_verifier_task: Optional[asyncio.Task] = None
        self._polling_teardown_started: bool = False
        self._polling_error_callback_ref = None
        self._polling_heartbeat_task: Optional[asyncio.Task] = None
        # Stall watchdog: generation start and last successful getUpdates (None = unknown).
        self._polling_generation_started_monotonic: Optional[float] = None
        self._polling_last_progress_monotonic: Optional[float] = None
        # Live @username. PTB caches getMe() in Bot._bot_user at initialize() and only rewrites
        # it inside get_me(), so a BotFather rename leaves self._bot.username stale. All
        # mention/routing comparisons read _current_bot_username() instead.
        self._bot_username_observed: Optional[str] = None
        # None = never checked. Must NOT be 0.0: compared against time.monotonic(), which on
        # a fresh host starts near zero, so 0.0 would suppress the first refresh for a TTL.
        self._bot_identity_checked_at: Optional[float] = None
        self._bot_identity_refresh_task: Optional[asyncio.Task] = None
        # Consecutive heartbeat probes seeing queued updates the poller isn't consuming.
        # get_me() can't see this (send path healthy, getUpdates wedged), so the heartbeat
        # probes get_webhook_info().pending_update_count and escalates after two.
        self._polling_pending_stuck_count: int = 0
        # Consecutive probes finding the updater stopped (running=False) in polling mode with
        # no reconnect in flight — the long-poll task is simply gone, so no probe or PTB
        # error_callback ever fires and the gateway silently stops receiving.
        self._polling_not_running_count: int = 0
        # Degraded until getUpdates makes progress (start_polling() return and getMe() on the
        # general path are NOT polling-health signals). While True, send() short-circuits to
        # failure so callers (cron live-adapter branch) fall through to standalone delivery.
        self._send_path_degraded: bool = False
        self._general_request_drain_lock = asyncio.Lock()
        self._dm_topics: Dict[str, int] = {}  # topic_name -> message_thread_id
        self._forum_command_registered: set[int] = set()  # forum chats with commands registered
        self._forum_lock = asyncio.Lock()
        # Status indicator: sets the bot's short description to "Online"/"Offline" on
        # connect/clean disconnect — bots have no presence dot. Off by default because it
        # mutates the GLOBAL profile; opt in via extra.status_indicator/status_online/status_offline.
        self._status_indicator_enabled: bool = bool(
            self.config.extra.get("status_indicator", False)
        )
        self._status_online_text: str = str(self.config.extra.get("status_online", "Online"))
        self._status_offline_text: str = str(self.config.extra.get("status_offline", "Offline"))
        self._dm_topics_config: List[Dict[str, Any]] = self.config.extra.get("dm_topics", [])
        # chat_ids with DM topics configured (O(1) root-DM ignore check)
        self._dm_topic_chat_ids: Set[str] = {
            str(e["chat_id"]) for e in self._dm_topics_config if "chat_id" in e
        }
        # getFile cap: 20MB on the public Bot API, 2GB on a local telegram-bot-api (base_url).
        self._max_doc_bytes: int = (
            2 * 1024 * 1024 * 1024 if self.config.extra.get("base_url") else 20 * 1024 * 1024
        )
        self._model_picker_state: Dict[str, dict] = {}  # per-chat interactive picker state
        self._choice_picker_state: Dict[str, dict] = {}
        self._approval_state: Dict[int, str] = {}  # message_id → session_key
        # confirm_id → session_key (see GatewayRunner._request_slash_confirm)
        self._slash_confirm_state: Dict[str, str] = {}
        # clarify_id → session_key (see GatewayRunner clarify_callback wiring)
        self._clarify_state: Dict[str, str] = {}
        # "important" (default): only final responses, approvals and slash confirmations
        # notify; progress/streaming/status go out with disable_notification.
        # "all": every message notifies (opt-in via display.platforms.telegram.notifications).
        self._notifications_mode: str = "important"
        # send_or_update_status(): {(chat_id, status_key) -> message_id} so repeat calls edit
        # the same bubble instead of appending.
        self._status_message_ids: Dict[tuple, str] = {}
        # Last truncated mid-stream preview per (chat_id, message_id). Once an oversized
        # stream saturates the 4096 cap every edit truncates to the SAME text; resending is a
        # no-op that still burns flood budget (200s+ penalties). Entries dropped on finalize.
        self._last_overflow_preview: Dict[tuple, str] = {}
        # Post-connect housekeeping (command menu + DM topics) runs off the connect path so a
        # slow Bot API call (set_my_commands stall) can't blow the gateway connect timeout.
        self._post_connect_task: Optional[asyncio.Task] = None

    def _mark_connected(self) -> None:
        self._drop_delayed_deliveries = False
        super()._mark_connected()
        # Drain the hold queue — PTB will not redeliver these events.
        self._schedule_held_inbound_redispatch()

    def _mark_disconnected(self) -> None:
        self._drop_delayed_deliveries = True
        super()._mark_disconnected()

    def _set_fatal_error(self, code: str, message: str, *, retryable: bool) -> None:
        self._drop_delayed_deliveries = True
        super()._set_fatal_error(code, message, retryable=retryable)
        # Permanent fatal: no reconnect will drain, so discard the hold queue and refuse
        # further holds (teardown salvage / late enqueue must not re-populate it).
        if not retryable:
            held = getattr(self, "_held_inbound_events", None)
            n = len(held) if held else 0
            if held:
                held.clear()
            if n:
                logger.warning(
                    "[Telegram] Non-retryable fatal (%s); discarding %d held inbound message(s)",
                    code, n,
                )

    def _is_permanent_fatal(self) -> bool:
        """True after non-retryable fatal — holds must discard, not queue."""
        if not getattr(self, "_fatal_error_code", None):
            return False
        return not bool(getattr(self, "_fatal_error_retryable", True))

    def _replacement_telegram_adapter(self) -> Optional["TelegramAdapter"]:
        """Return the live adapter if the reconnect watcher replaced us in ``runner.adapters``.

        An in-flight ``send()`` still holds the old instance whose ``_bot`` stays None, so
        waiting only on ``self._bot`` would drop the final reply.
        """
        runner = getattr(self, "gateway_runner", None)
        adapters = getattr(runner, "adapters", None) or {}
        live = adapters.get(self.platform)
        if live is not None and live is not self and getattr(live, "_bot", None):
            return live
        return None

    async def _wait_for_reconnection(self) -> bool:
        """Wait for ``_bot`` or a replacement adapter after a transient drop.

        Returns True if sending can proceed; False on wait expiry or permanent fatal.
        """
        if self._bot or self._replacement_telegram_adapter() is not None:
            return True
        if self._is_permanent_fatal():
            return False
        wait_s = float(getattr(self, "_RECONNECT_WAIT_SECONDS", 15.0))
        poll_s = float(getattr(self, "_RECONNECT_POLL_INTERVAL", 0.5))
        logger.info(
            "[%s] Not connected — waiting for reconnection (up to %.0fs)", self.name, wait_s
        )
        waited = 0.0
        while waited < wait_s:
            await asyncio.sleep(poll_s)
            waited += poll_s
            if self._is_permanent_fatal():
                return False
            if self._bot or self._replacement_telegram_adapter() is not None:
                logger.info("[%s] Reconnected after %.1fs", self.name, waited)
                return True
        logger.warning("[%s] Still not connected after %.0fs", self.name, wait_s)
        return False

    def _should_drop_delayed_delivery(self) -> bool:
        """True once teardown/fatal-error started — delayed flushes must not dispatch.

        Buffered flushes sit behind an asyncio.sleep(); if disconnect wins the race they'd
        spawn an agent on a torn-down session. Callers must NOT destroy the event (PTB already
        advanced the offset) — use ``_hold_inbound_event`` and redispatch on reconnect.
        """
        return bool(getattr(self, "_drop_delayed_deliveries", False))

    def _schedule_held_inbound_redispatch(self) -> None:
        """Ensure a tracked drain runs when held events exist and delivery is live.

        Triggered by ``_mark_connected``, holds created while connected (cancel-after-pop),
        and the end of a drain pass with leftovers. No-op while down or after permanent fatal.
        """
        if self._is_permanent_fatal():
            return
        if self._should_drop_delayed_delivery():
            return
        held = getattr(self, "_held_inbound_events", None)
        if not held:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        prior = getattr(self, "_held_inbound_redispatch_task", None)
        try:
            current = asyncio.current_task()
        except RuntimeError:
            current = None
        # Already draining on another task — that pass schedules any follow-up itself.
        if prior is not None and not prior.done() and prior is not current:
            return
        self._held_inbound_redispatch_task = loop.create_task(
            self._redispatch_held_inbound(prior=None if prior is current else prior)
        )

    def _hold_inbound_event(
        self, event: "MessageEvent", *, where: str, schedule: bool = True
    ) -> None:
        """Preserve an inbound event that cannot be dispatched right now.

        By enqueue/flush time PTB has already acked the update and advanced the offset, so
        destroying the event is silent permanent loss. Hold it, redispatch from
        ``_mark_connected`` (or immediately if connected). Capped, identity-deduped;
        permanent fatal discards. ``schedule=False`` inside a drain (avoids poison-event loops).
        """
        if self._is_permanent_fatal():
            logger.warning(
                "[Telegram] Discarding inbound under non-retryable fatal (%s, %d chars)",
                where, len(getattr(event, "text", None) or ""),
            )
            return
        held = getattr(self, "_held_inbound_events", None)
        if held is None:
            self._held_inbound_events = []
            held = self._held_inbound_events
        for existing in held:
            if existing is event:
                return
        max_n = int(getattr(self, "HELD_INBOUND_MAX", 64) or 64)
        while len(held) >= max_n:
            dropped = held.pop(0)
            logger.warning(
                "[Telegram] Held-inbound queue full (%d); dropping oldest (%d chars)",
                max_n, len(getattr(dropped, "text", None) or ""),
            )
        held.append(event)
        logger.warning(
            "[Telegram] Holding inbound (%s, %d chars, queue=%d)%s",
            where, len(getattr(event, "text", None) or ""), len(held),
            " - will redispatch on reconnect" if self._should_drop_delayed_delivery()
            else (" - scheduling redispatch" if schedule else ""),
        )
        # A live-path hold must not orphan the event waiting for a reconnect that never comes.
        if schedule and not self._should_drop_delayed_delivery():
            self._schedule_held_inbound_redispatch()

    async def _redispatch_held_inbound(self, prior: Optional[asyncio.Task] = None) -> None:
        """Drain the hold queue after reconnect or a connected-path hold.

        ``prior`` (previous redispatch task) is cancelled+awaited here so ``_mark_connected``
        stays synchronous while teardown can still cancel the single tracked task.
        """
        if prior is not None and prior is not asyncio.current_task() and not prior.done():
            prior.cancel()
            try:
                await prior
            except asyncio.CancelledError:
                pass
        if self._is_permanent_fatal():
            held = getattr(self, "_held_inbound_events", None)
            if held:
                n = len(held)
                held.clear()
                logger.warning(
                    "[Telegram] Redispatch aborted; discarded %d held inbound under non-retryable fatal",
                    n,
                )
            return
        held = getattr(self, "_held_inbound_events", None)
        if not held:
            return
        # Take ownership atomically; concurrent holds append to the fresh list for a follow-up.
        events = list(held)
        held.clear()
        logger.warning("[Telegram] Redispatching %d held inbound message(s)", len(events))
        allow_followup_schedule = True
        try:
            for idx, event in enumerate(events):
                if self._is_permanent_fatal() or self._should_drop_delayed_delivery():
                    # Disconnect/fatal mid-drain — re-hold current + remainder.
                    self._hold_inbound_event(event, where="redispatch-interrupted", schedule=False)
                    for rest in events[idx + 1 :]:
                        self._hold_inbound_event(
                            rest, where="redispatch-interrupted", schedule=False
                        )
                    return
                try:
                    await self.handle_message(event)
                except asyncio.CancelledError:
                    self._hold_inbound_event(event, where="redispatch-cancelled", schedule=False)
                    for rest in events[idx + 1 :]:
                        self._hold_inbound_event(rest, where="redispatch-cancelled", schedule=False)
                    raise
                except Exception:
                    # Retryable failure: re-hold current + remainder but do NOT reschedule now —
                    # a poison event would tight-loop. Next mark_connected/live hold drains.
                    logger.exception(
                        "[Telegram] Failed to redispatch held inbound (%d chars); re-holding",
                        len(getattr(event, "text", None) or ""),
                    )
                    self._hold_inbound_event(event, where="redispatch-failed", schedule=False)
                    for rest in events[idx + 1 :]:
                        self._hold_inbound_event(rest, where="redispatch-failed", schedule=False)
                    allow_followup_schedule = False
                    return
        finally:
            # Events that arrived mid-drain while still connected need another pass.
            if (
                allow_followup_schedule
                and getattr(self, "_held_inbound_events", None)
                and not self._should_drop_delayed_delivery()
                and not self._is_permanent_fatal()
            ):
                self._schedule_held_inbound_redispatch()

    def _notification_kwargs(self, metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """In "important" mode return disable_notification=True unless ``metadata["notify"]``."""
        if getattr(self, "_notifications_mode", "important") != "important":
            return {}
        if (metadata or {}).get("notify"):
            return {}
        return {"disable_notification": True}

    def _is_callback_user_authorized(
        self,
        user_id: str,
        *,
        chat_id: Optional[str] = None,
        chat_type: Optional[str] = None,
        thread_id: Optional[str] = None,
        user_name: Optional[str] = None,
    ) -> bool:
        """Return whether a Telegram inline-button caller may perform gated actions."""
        normalized_user_id = str(user_id or "").strip()
        if not normalized_user_id:
            return False
        normalized_chat_type = str(chat_type or "dm").strip().lower() or "dm"
        if normalized_chat_type == "private":
            normalized_chat_type = "dm"
        elif normalized_chat_type == "supergroup":
            normalized_chat_type = "forum" if thread_id is not None else "group"

        # Preferred: the auth callback GatewayRunner injects (set_authorization_check) → full
        # _is_user_authorized chain. Unlike the __self__ introspection below it also works for
        # a secondary multiplexed adapter whose _message_handler is a profile closure. getattr
        # tolerates partially-constructed adapters (object.__new__ in tests).
        if getattr(self, "_authorization_check", None) is not None:
            injected = self._is_sender_authorized(
                normalized_user_id,
                chat_type=normalized_chat_type,
                chat_id=str(chat_id or normalized_user_id),
                thread_id=str(thread_id) if thread_id is not None else None,
            )
            if injected is not None:
                return injected

        # Legacy: resolve the runner off the bound handler (bare-adapter tests, direct embedding).
        runner = getattr(getattr(self, "_message_handler", None), "__self__", None)
        auth_fn = getattr(runner, "_is_user_authorized", None)
        if callable(auth_fn):
            try:
                from gateway.session import SessionSource
                source = SessionSource(
                    platform=Platform.TELEGRAM,
                    chat_id=str(chat_id or normalized_user_id),
                    chat_type=normalized_chat_type,
                    user_id=normalized_user_id,
                    user_name=str(user_name).strip() if user_name else None,
                    thread_id=str(thread_id) if thread_id is not None else None,
                )
                return bool(auth_fn(source))
            except Exception:
                logger.debug(
                    "[Telegram] Falling back to env-only callback auth for user %s",
                    normalized_user_id, exc_info=True,
                )
        allowed_csv = _scoped_gate_env("TELEGRAM_ALLOWED_USERS").strip()
        if not allowed_csv:
            # Fail-closed: no allowlist means deny unless GATEWAY_ALLOW_ALL_USERS is set.
            return _scoped_gate_env("GATEWAY_ALLOW_ALL_USERS").lower() in {"true", "1", "yes"}
        allowed_ids = {uid.strip() for uid in allowed_csv.split(",") if uid.strip()}
        return "*" in allowed_ids or normalized_user_id in allowed_ids

    def _source_from_message_for_auth(self, message: Message):
        """Build the SessionSource the gateway auth path expects.

        Identity comes from ``from_user``, falling back to ``sender_chat`` for channel posts
        so an unauthorized channel cannot inject content via the broadcast path.
        """
        from gateway.session import SessionSource
        user = getattr(message, "from_user", None)
        chat = getattr(message, "chat", None)
        user_id = str(getattr(user, "id", "")).strip() or None
        # Carry is_bot so the runner's ``*_ALLOW_BOTS`` branch is reachable, as in build_source.
        is_bot = bool(getattr(user, "is_bot", False)) if user is not None else False
        user_name = (
            str(getattr(user, "username", "") or getattr(user, "full_name", "") or "").strip()
            or None
        )
        if not user_id:  # channel post — authorize the sender chat instead
            sender_chat = getattr(message, "sender_chat", None)
            if sender_chat is not None:
                user_id = str(getattr(sender_chat, "id", "")).strip() or None
                if not user_name:
                    user_name = str(getattr(sender_chat, "title", "") or "").strip() or None
        chat_id = str(getattr(chat, "id", "")).strip() or user_id
        chat_type = str(getattr(chat, "type", "dm")).strip().lower() or "dm"
        if chat_type == "private":
            chat_type = "dm"
        elif chat_type == "supergroup":
            thread_id_raw = getattr(message, "message_thread_id", None)
            is_topic_message = bool(getattr(message, "is_topic_message", False))
            is_forum_group = getattr(chat, "is_forum", False) is True
            chat_type = (
                "forum"
                if thread_id_raw is not None and (is_topic_message or is_forum_group)
                else "group"
            )
        thread_id = None
        thread_id_raw = getattr(message, "message_thread_id", None)
        if thread_id_raw is not None:
            is_topic_message = bool(getattr(message, "is_topic_message", False))
            is_forum_group = getattr(chat, "is_forum", False) is True
            if (chat_type == "forum" and (is_topic_message or is_forum_group)) or (
                chat_type == "dm" and is_topic_message
            ):
                thread_id = str(thread_id_raw)
        return SessionSource(
            platform=Platform.TELEGRAM, chat_id=chat_id or "", chat_type=chat_type, user_id=user_id,
            user_name=user_name, thread_id=thread_id, is_bot=is_bot,
        )

    def _source_from_reaction_for_auth(self, update):
        """Build the SessionSource for a ``message_reaction`` update's actor.

        Like ``_source_from_message_for_auth`` but reactions carry ``user`` (or ``actor_chat``
        for an anonymous admin) and ``chat``, no ``Message`` and no thread id.

        Raises ``ValueError`` when actor, chat or message identity is absent so the post-auth
        boundary fails closed rather than authorizing an incomplete source.
        """
        mr = getattr(update, "message_reaction", None)
        if mr is None:
            raise ValueError(
                "gateway_platform_event source extraction requires a message_reaction update"
            )
        user = getattr(mr, "user", None) or getattr(mr, "actor_chat", None)
        chat = getattr(mr, "chat", None)
        user_id = str(getattr(user, "id", "")).strip() or None
        user_name = (
            str(
                getattr(user, "username", "")
                or getattr(user, "full_name", "")
                or getattr(user, "title", "")
            ).strip()
            or None
        )
        chat_id = str(getattr(chat, "id", "")).strip() or None
        message_id = getattr(mr, "message_id", None)
        if not user_id or not chat_id or message_id is None or not str(message_id).strip():
            raise ValueError(
                "gateway_platform_event reaction requires actor, chat, and message identities"
            )
        chat_type = str(getattr(chat, "type", "dm")).strip().lower() or "dm"
        if chat_type == "private":
            chat_type = "dm"
        elif chat_type == "supergroup":
            # Reactions carry no message_thread_id; is_forum is the only forum signal.
            chat_type = "forum" if getattr(chat, "is_forum", False) is True else "group"
        return self.build_source(
            chat_id=chat_id, chat_type=chat_type, user_id=user_id, user_name=user_name,
            thread_id=None, message_id=str(message_id),
        )

    def _telegram_auth_env_configured(self) -> bool:
        """Return True when Telegram auth env vars make an early decision safe."""
        keys = (
            "TELEGRAM_ALLOWED_USERS", "TELEGRAM_GROUP_ALLOWED_USERS",
            "TELEGRAM_GROUP_ALLOWED_CHATS", "TELEGRAM_ALLOW_ALL_USERS",
            "GATEWAY_ALLOWED_USERS", "GATEWAY_ALLOW_ALL_USERS",
        )
        return any(_scoped_gate_env(key).strip() for key in keys)

    def _should_pass_unauthorized_dm_for_pairing(self, source) -> bool:
        """True when an unauthorized DM must still reach gateway pairing.

        Early auth must not short-circuit the pairing handshake when
        ``unauthorized_dm_behavior`` resolves to ``pair`` — including an allowlist plus an
        explicit platform override opting back into pairing.
        """
        if source.chat_type != "dm":
            return False
        # Bound-handler ``__self__`` is None under multiplex (profile closure); ``gateway_runner``
        # is injected on every adapter and survives that wrapping.
        runner = getattr(getattr(self, "_message_handler", None), "__self__", None) or getattr(
            self, "gateway_runner", None
        )
        behavior_fn = getattr(runner, "_get_unauthorized_dm_behavior", None)
        if callable(behavior_fn):
            try:
                return (
                    behavior_fn(
                        Platform.TELEGRAM,
                        profile=getattr(source, "profile", None)
                        or getattr(self, "_owner_profile", None),
                    )
                    == "pair"
                )
            except Exception:
                logger.debug(
                    "[Telegram] Failed to resolve unauthorized DM behavior; "
                    "falling back to adapter-local override",
                    exc_info=True,
                )
        extra = getattr(getattr(self, "config", None), "extra", None) or {}
        return str(extra.get("unauthorized_dm_behavior", "")).strip().lower() == "pair"

    def _is_user_authorized_from_message(self, message: Message) -> bool:
        """Intake auth prefilter, run BEFORE text batching/event construction/group observation.

        Only rejects when it can make the same context-aware decision the runner would.
        Unknown DMs pass through when there is no allowlist, or when pairing is the effective
        unauthorized-DM behavior, so the pairing flow can run.
        """
        source = self._source_from_message_for_auth(message)
        user_id = source.user_id
        # No identity → group service message (pin, new_chat_members…) or channel post without
        # sender_chat; nothing authorizable, defer to _should_process_message gating.
        if not user_id:
            return True
        authorized: Optional[bool] = None
        # Adapter-level allow_from (DMs) / group_allow_from (groups) are the sole authority if set.
        chat_type = source.chat_type or ""
        if chat_type in ("group", "forum", "channel"):
            adapter_allow_from = self.config.extra.get("group_allow_from")
        else:
            adapter_allow_from = self.config.extra.get("allow_from")
        if adapter_allow_from is not None:
            allowed = _coerce_allow_set(adapter_allow_from)
            authorized = user_id in allowed or "*" in allowed

        # Instance-level override only (tests). The class method _is_callback_user_authorized is
        # for inline buttons and must not become a user-id-only shortcut for real messages.
        if authorized is None:
            callback_auth = self.__dict__.get("_is_callback_user_authorized")
            if callable(callback_auth):
                try:
                    authorized = bool(
                        callback_auth(
                            user_id, chat_id=source.chat_id, chat_type=source.chat_type,
                            thread_id=source.thread_id, user_name=source.user_name,
                        )
                    )
                except Exception:
                    pass
        if authorized is None:
            # Runner's full auth chain. Prefer the set_authorization_check callback: it survives
            # multiplex handler wrapping, whereas bound-handler __self__ is None for a profile
            # closure (which silently default-denied allowlisted group members).
            runner = getattr(getattr(self, "_message_handler", None), "__self__", None)
            auth_fn = getattr(runner, "_is_user_authorized", None)
            has_callback = getattr(self, "_authorization_check", None) is not None
            if has_callback or callable(auth_fn):
                # No allowlist → unknown DMs must reach pairing, not be default-denied here.
                if not self._telegram_auth_env_configured():
                    return True
                decision = (
                    self._is_sender_authorized(
                        user_id, chat_type=source.chat_type, chat_id=source.chat_id,
                        is_bot=source.is_bot, thread_id=source.thread_id,
                    )
                    if has_callback
                    else None
                )
                if decision is not None:
                    authorized = decision
                elif callable(auth_fn):
                    try:
                        authorized = bool(auth_fn(source))
                    except Exception:
                        logger.debug(
                            "[Telegram] Falling back to env-only auth for user %s",
                            user_id, exc_info=True,
                        )
        if authorized is None:
            allowed_csv = _scoped_gate_env("TELEGRAM_ALLOWED_USERS").strip()
            if not allowed_csv:
                return True
            allowed_ids = {uid.strip() for uid in allowed_csv.split(",") if uid.strip()}
            authorized = "*" in allowed_ids or user_id in allowed_ids
        if authorized:
            return True
        # Unauthorized DM the gateway would pair: forward so pairing can run.
        return self._should_pass_unauthorized_dm_for_pairing(source)

    @classmethod
    def _metadata_thread_id(cls, metadata: Optional[Dict[str, Any]]) -> Optional[str]:
        if not metadata:
            return None
        thread_id = metadata.get("thread_id") or metadata.get("message_thread_id")
        return str(thread_id) if thread_id is not None else None

    @classmethod
    def _metadata_direct_messages_topic_id(cls, metadata: Optional[Dict[str, Any]]) -> Optional[str]:
        if not metadata:
            return None
        topic_id = metadata.get("direct_messages_topic_id") or metadata.get("telegram_direct_messages_topic_id")
        return str(topic_id) if topic_id is not None else None

    @classmethod
    def _metadata_reply_to_message_id(cls, metadata: Optional[Dict[str, Any]]) -> Optional[int]:
        if not metadata:
            return None
        reply_to = metadata.get("telegram_reply_to_message_id")
        return int(reply_to) if reply_to is not None else None

    @classmethod
    def _is_private_dm_topic_send(
        cls, chat_id: str, thread_id: Optional[str], metadata: Optional[Dict[str, Any]]
    ) -> bool:
        if cls._metadata_direct_messages_topic_id(metadata) is not None:
            return bool(
                metadata
                and metadata.get("telegram_dm_topic_reply_fallback")
                and cls._metadata_reply_to_message_id(metadata) is not None
            )
        if metadata and metadata.get("telegram_dm_topic_created_for_send"):
            return False
        return bool(thread_id and metadata and metadata.get("telegram_dm_topic_reply_fallback"))

    @staticmethod
    def _dm_topic_missing_anchor_error() -> str:
        return "Telegram DM topic delivery requires a reply anchor; refusing to send outside the requested topic"

    @classmethod
    def _reply_to_message_id_for_send(
        cls, reply_to: Optional[str], metadata: Optional[Dict[str, Any]] = None,
        reply_to_mode: Optional[str] = None,
    ) -> Optional[int]:
        if reply_to:
            return int(reply_to)
        if metadata and metadata.get("telegram_dm_topic_reply_fallback"):
            if reply_to_mode == "off":
                return None
            return cls._metadata_reply_to_message_id(metadata)
        return None

    @classmethod
    def _thread_kwargs_for_send(
        cls,
        chat_id: str,
        thread_id: Optional[str],
        metadata: Optional[Dict[str, Any]] = None,
        reply_to_message_id: Optional[int] = None,
        reply_to_mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Telegram send kwargs for forum and direct-message topic routing.

        Forum topics use ``message_thread_id``; native Bot API DM topics opt in via explicit
        ``direct_messages_topic_id`` metadata; Hermes private-chat topic lanes are marked
        ``telegram_dm_topic_reply_fallback``. Anchor-less synthetic sends (loop wakeups,
        restart-resumed follow-ups) prefer the Hermes topic's ``message_thread_id`` so they stay
        in the active lane — the native DM-topic id renders in a different chat lane.
        ``reply_to_mode="off"`` suppresses the anchor but keeps ``message_thread_id``.
        """
        if metadata and metadata.get("telegram_dm_topic_reply_fallback"):
            if reply_to_mode == "off":
                return {"message_thread_id": cls._message_thread_id_for_send(thread_id)}
            if reply_to_message_id is None:
                reply_to_message_id = cls._metadata_reply_to_message_id(metadata)
            if reply_to_message_id is None:
                # Anchor-less synthetic send: prefer the Hermes topic thread id (see docstring).
                thread_message_id = cls._message_thread_id_for_send(thread_id)
                if thread_message_id is not None:
                    return {"message_thread_id": thread_message_id}
                direct_topic_id = cls._metadata_direct_messages_topic_id(metadata)
                if direct_topic_id is not None:
                    return {
                        "message_thread_id": None, "direct_messages_topic_id": int(direct_topic_id)
                    }
                return {}
            return {"message_thread_id": cls._message_thread_id_for_send(thread_id)}
        direct_topic_id = cls._metadata_direct_messages_topic_id(metadata)
        if direct_topic_id is not None:
            return {"message_thread_id": None, "direct_messages_topic_id": int(direct_topic_id)}
        return {"message_thread_id": cls._message_thread_id_for_send(thread_id)}

    def _thread_kwargs_for_draft(
        self, chat_id: str, metadata: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Routing kwargs for ``sendMessageDraft`` / ``sendRichMessageDraft``.

        Reuses ``_thread_kwargs_for_send`` so DM topics get an integer ``message_thread_id`` —
        Telegram rejects the raw string ``thread_id``, which disabled draft streaming for the turn.
        """
        thread_id = self._metadata_thread_id(metadata)
        reply_to_id = self._reply_to_message_id_for_send(None, metadata)
        kwargs = self._thread_kwargs_for_send(
            chat_id, thread_id, metadata, reply_to_message_id=reply_to_id,
            reply_to_mode=getattr(self, "_reply_to_mode", None),
        )
        return {k: v for k, v in kwargs.items() if v is not None}

    @classmethod
    def _message_thread_id_for_send(cls, thread_id: Optional[str]) -> Optional[int]:
        if not thread_id or str(thread_id) == cls._GENERAL_TOPIC_THREAD_ID:
            return None
        return int(thread_id)

    @classmethod
    def _message_thread_id_for_typing(cls, thread_id: Optional[str]) -> Optional[int]:
        # Deliberately asymmetric with _message_thread_id_for_send: sendMessage rejects
        # message_thread_id=1 (forum General), but sendChatAction NEEDS it to place the typing
        # bubble in General — omitting it hides the bubble entirely.
        if not thread_id:
            return None
        return int(thread_id)

    @staticmethod
    def _is_thread_not_found_error(error: Exception) -> bool:
        return "thread not found" in str(error).lower()

    def _prune_stale_dm_topic_binding(
        self, chat_id: Any, thread_id: Any, *, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Drop the stale ``telegram_dm_topic_bindings`` row for a topic Telegram confirmed deleted.

        Otherwise ``gateway.run._recover_telegram_topic_thread_id`` keeps steering inbound to
        the dead thread. Best-effort: never raises from a send-fallback path. Rows are
        namespaced by profile; under ``profile_routes`` the send's ``hermes_profile`` metadata
        wins over the adapter's own stamp, falling back to ``"default"``.
        """
        if chat_id is None or thread_id is None:
            return
        store = getattr(self, "_session_store", None)
        if store is None:
            return
        db = getattr(store, "_db", None)
        if db is None or not hasattr(db, "delete_telegram_topic_binding"):
            return
        try:
            profile_name = (
                (metadata or {}).get("hermes_profile")
                or getattr(self, "_hermes_profile_name", None)
                or "default"
            )
            removed = db.delete_telegram_topic_binding(
                chat_id=str(chat_id), thread_id=str(thread_id), profile_name=profile_name
            )
        except Exception:
            logger.debug(
                "[%s] delete_telegram_topic_binding failed for chat=%s thread=%s — skipping prune",
                self.name, chat_id, thread_id, exc_info=True,
            )
            return
        if removed:
            logger.info(
                "[%s] Pruned stale Telegram DM topic binding "
                "chat=%s thread=%s (Bot API: thread not found)",
                self.name, chat_id, thread_id,
            )

    @staticmethod
    def _is_bad_request_error(error: Exception) -> bool:
        name = error.__class__.__name__.lower()
        if name == "badrequest" or name.endswith("badrequest"):
            return True
        try:
            from telegram.error import BadRequest
            return isinstance(error, BadRequest)
        except ImportError:
            return False

    @classmethod
    def _should_retry_without_dm_topic_reply_anchor(
        cls, error: Exception, metadata: Optional[Dict[str, Any]],
        reply_to_message_id: Optional[int],
    ) -> bool:
        """True when a DM-topic send should be retried with routing stripped.

        Cases: (1) stale anchor — reply target deleted ("message to be replied not found");
        (2) anchor-less synthetic send whose ``direct_messages_topic_id`` Bot API rejects with a
        topic/thread BadRequest. Retry without routing rather than drop the message.
        """
        if not (metadata and metadata.get("telegram_dm_topic_reply_fallback")):
            return False
        if not cls._is_bad_request_error(error):
            return False
        err_lower = str(error).lower()
        if reply_to_message_id is not None and "message to be replied not found" in err_lower:
            return True
        if metadata.get("direct_messages_topic_id"):  # topic id rejected → plain DM send
            topic_markers = (
                "direct_messages_topic", "message thread not found", "thread not found",
                "topic_closed", "topic_deleted", "topic not found",
            )
            if any(marker in err_lower for marker in topic_markers):
                return True
        return False

    async def _send_with_dm_topic_reply_anchor_retry(
        self,
        send_fn: Any,
        send_kwargs: Dict[str, Any],
        metadata: Optional[Dict[str, Any]],
        reply_to_message_id: Optional[int],
        media_label: str,
        reset_media: Optional[Any] = None,
    ) -> Any:
        """Retry stale private-topic media replies once without the topic anchor."""
        try:
            return await send_fn(**send_kwargs)
        except Exception as send_err:
            if not self._should_retry_without_dm_topic_reply_anchor(
                send_err, metadata, reply_to_message_id
            ):
                raise
            logger.warning(
                "[%s] Reply target deleted for Telegram %s, "
                "retrying without reply/topic anchor: %s",
                self.name, media_label, _redact_telegram_error_text(send_err),
            )
            if reset_media is not None:
                reset_media()
            retry_kwargs = dict(send_kwargs)
            retry_kwargs["reply_to_message_id"] = None
            retry_kwargs.pop("message_thread_id", None)
            retry_kwargs.pop("direct_messages_topic_id", None)
            return await send_fn(**retry_kwargs)

    def _fallback_ips(self) -> list[str]:
        """Return validated fallback IPs from config (populated by _apply_env_overrides)."""
        configured = self.config.extra.get("fallback_ips", []) if getattr(self.config, "extra", None) else []
        if isinstance(configured, str):
            configured = configured.split(",")
        return parse_fallback_ip_env(",".join(str(v) for v in configured) if configured else None)

    @staticmethod
    def _looks_like_polling_conflict(error: Exception) -> bool:
        text = str(error).lower()
        return (
            error.__class__.__name__.lower() == "conflict"
            or "terminated by other getupdates request" in text
            or "another bot instance is running" in text
        )

    @staticmethod
    def _looks_like_auth_error(error: Exception) -> bool:
        """True for terminal credential failures (InvalidToken, Forbidden) → retryable=False.

        Deliberately narrower than "not network error": BadRequest/RetryAfter are transient at
        connect time and must keep retrying. Type-based only; never match on message text.
        """
        name = error.__class__.__name__.lower()
        if name in {"invalidtoken", "forbidden"}:
            return True
        try:
            from telegram.error import Forbidden, InvalidToken
            return isinstance(error, (InvalidToken, Forbidden))
        except ImportError:
            return False

    @staticmethod
    def _looks_like_network_error(error: Exception) -> bool:
        """Return True for transient transport failures that warrant reconnect."""
        name = error.__class__.__name__.lower()
        if name in {"badrequest", "invalidtoken", "forbidden", "retryafter"}:
            return False
        if name in {"networkerror", "timedout", "connectionerror"}:
            return True
        try:
            from telegram.error import (
                BadRequest, Forbidden, InvalidToken, NetworkError, RetryAfter, TimedOut,
            )
            if isinstance(error, (BadRequest, InvalidToken, Forbidden, RetryAfter)):
                return False
            if isinstance(error, (NetworkError, TimedOut)):
                return True
        except ImportError:
            pass
        return isinstance(error, OSError)

    @staticmethod
    def _looks_like_connect_timeout(error: Exception) -> bool:
        """True when a TimedOut wraps a ConnectTimeout: TCP never connected, so re-sending is safe.

        A plain TimedOut may have reached Telegram and must not be re-sent.
        """
        for cur in _iter_exception_graph(error):
            name = cur.__class__.__name__.lower()
            text = str(cur).lower()
            if "connecttimeout" in name or "connect timeout" in text or "connect timed out" in text:
                return True
        return False

    @staticmethod
    def _looks_like_pool_timeout(error: Exception) -> bool:
        """True when a TimedOut wraps ``httpx.PoolTimeout``: the request never left the process.

        PTB's message says "Request was *not* sent to Telegram", so re-sending cannot duplicate
        (unlike a generic TimedOut). Matches the wrapped class AND the text to survive rewording.
        """
        for cur in _iter_exception_graph(error):
            name = cur.__class__.__name__.lower()
            text = str(cur).lower()
            if "pooltimeout" in name or "pool timeout" in text or (
                "connection pool" in text and "occupied" in text
            ):
                return True
        return False

    def _coerce_bool_extra(self, key: str, default: bool = False) -> bool:
        value = self.config.extra.get(key) if getattr(self.config, "extra", None) else None
        if value is None:
            return default
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "1", "yes", "on"}:
                return True
            if lowered in {"false", "0", "no", "off"}:
                return False
            return default
        return bool(value)

    def _coerce_float_extra(
        self, key: str, default: float, *,
        min_value: Optional[float] = None, max_value: Optional[float] = None,
    ) -> float:
        value = self.config.extra.get(key) if getattr(self.config, "extra", None) else None
        if value is None:
            return default
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return default
        if min_value is not None:
            parsed = max(parsed, min_value)
        if max_value is not None:
            parsed = min(parsed, max_value)
        return parsed

    def _link_preview_kwargs(self) -> Dict[str, Any]:
        if not getattr(self, "_disable_link_previews", False):
            return {}
        if LinkPreviewOptions is not None:
            return {"link_preview_options": LinkPreviewOptions(is_disabled=True)}
        return {"disable_web_page_preview": True}

    # --- Bot API 10.1 Rich Messages (sendRichMessage) ---------------------------------------
    # Final/new-message replies opportunistically send RAW agent markdown so tables, task lists,
    # <details>, math render natively; legacy MarkdownV2 send() is the fallback. Streaming edits
    # stay on the MarkdownV2 edit path; finalization may re-send rich and delete the preview.
    def _content_fits_rich_limits(self, content: str) -> bool:
        """Pre-check the 32,768-char cap only; other rich limits (blocks, nesting, columns)
        surface as BadRequest, which ``_is_rich_fallback_error`` treats as permanent."""
        return len(content) <= self.RICH_MESSAGE_MAX_CHARS

    def _bot_supports_rich(self) -> bool:
        """True when ``do_api_request`` is an *async* callable (real Bot or AsyncMock).

        Plain MagicMock (sync auto-child) and SimpleNamespace bots resolve False → legacy path.
        """
        return inspect.iscoroutinefunction(getattr(self._bot, "do_api_request", None))

    _RICH_DETAILS_RE = re.compile(r"<details\b[^>]*>.*?</details>", re.IGNORECASE | re.DOTALL)
    _RICH_MATH_IN_DETAILS_RE = re.compile(
        r"(\$\$.*?\$\$|\\\[.*?\\\]|\\\(.*?\\\)|"
        r"\\(?:sum|frac|alpha|beta|gamma|delta|theta|lambda|mu|pi|sigma|"
        r"int|prod|sqrt|lim|infty|begin\{(?:equation|align|matrix|cases)\}))",
        re.IGNORECASE | re.DOTALL,
    )
    _RICH_CJK_RE = re.compile(
        "["
        "\u3040-\u30ff"  # Hiragana, Katakana
        "\u3400-\u4dbf"  # CJK Extension A
        "\u4e00-\u9fff"  # CJK Unified Ideographs
        "\uac00-\ud7af"  # Hangul syllables
        "\uf900-\ufaff"  # CJK Compatibility Ideographs
        "\U00020000-\U000323af"  # CJK extensions and compatibility supplement
        "]"
    )

    def _has_telegram_desktop_details_math_crash_shape(self, content: str) -> bool:
        """True for math inside a <details> block — crashes Telegram Desktop 6.9.1 (tdesktop#30808).

        The Bot API accepts the payload, so we must skip rich delivery up front.
        """
        if not content:
            return False
        for details_block in self._RICH_DETAILS_RE.findall(content):
            if self._RICH_MATH_IN_DETAILS_RE.search(details_block):
                return True
        return False

    def _has_telegram_desktop_cjk_rich_garble_shape(self, content: str) -> bool:
        """True for CJK content: Telegram Mac/Desktop rich rendering leaves overlapping glyphs."""
        return bool(content and self._RICH_CJK_RE.search(content))

    def _needs_rich_rendering(self, content: str) -> bool:
        """True for constructs MarkdownV2 degrades: pipe tables, task lists, <details>, block math.

        Ordinary replies stay on MarkdownV2 so clients render consistent font weight/spacing.
        """
        if not content:
            return False
        if any(_TABLE_SEPARATOR_RE.match(line) for line in content.splitlines()):
            return True
        if re.search(r"(?m)^\s*[-*]\s+\[[ xX]\]\s+", content):
            return True
        if re.search(r"(?m)^<details\b|^</details>|^<summary\b|^</summary>", content):
            return True
        return "$$" in content

    def _rich_delivery_enabled(self) -> bool:
        """Whether rich delivery is allowed (``rich_messages`` opt-in)."""
        return bool(getattr(self, "_rich_messages_enabled", True))

    def _rich_eligible(self, content: str) -> bool:
        """Rich eligibility ignoring ``expect_edits``.

        ``_try_edit_rich`` needs this: a streamed preview carries ``expect_edits=True`` to stay
        editable mid-stream, but the FINAL edit should still upgrade to rich.
        """
        return bool(
            self._rich_delivery_enabled()
            and not getattr(self, "_rich_send_disabled", False)
            and content
            and content.strip()
            and self._needs_rich_rendering(content)
            and not self._has_telegram_desktop_details_math_crash_shape(content)
            and not self._has_telegram_desktop_cjk_rich_garble_shape(content)
            and self._content_fits_rich_limits(content)
            and self._bot_supports_rich()
        )

    def _should_attempt_rich(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        return bool(not (metadata or {}).get("expect_edits") and self._rich_eligible(content))

    def prefers_fresh_final_streaming(
        self, content: str, metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Whether to replace a streamed preview with a fresh rich final — DM topics only.

        Root DMs stay off: draft streaming has no preview message_id and an in-place rich edit
        would duplicate a live draft. DM *topics* often reject sendMessageDraft and degrade to
        edit-in-place, whose MarkdownV2 preview Telegram refuses to rich-edit — a fresh
        sendRichMessage plus deleting the preview is the only way to keep native tables.
        """
        metadata = metadata or {}
        if not (
            metadata.get("telegram_dm_topic_reply_fallback")
            or self._metadata_direct_messages_topic_id(metadata)
        ):
            return False
        return self._rich_eligible(content)

    def streaming_overflow_limit(self) -> Optional[int]:
        """Let the stream consumer accumulate up to the rich cap so a reply that fits one
        sendRichMessage isn't fragmented at 4,096. None (→ legacy limit) if rich is unavailable."""
        if (
            getattr(self, "_rich_messages_enabled", True)
            and not getattr(self, "_rich_send_disabled", False)
            and self._bot_supports_rich()
        ):
            return self.RICH_MESSAGE_MAX_CHARS
        return None

    def _rich_message_payload(
        self, content: str, *, skip_entity_detection: bool = False
    ) -> Dict[str, Any]:
        """Build the ``InputRichMessage`` from RAW markdown (single newlines → hard breaks).

        Never pass ``format_message(content)`` — MarkdownV2 escaping destroys table pipes.
        """
        payload: Dict[str, Any] = {"markdown": _rich_normalize_linebreaks(content)}
        if skip_entity_detection:
            payload["skip_entity_detection"] = True
        return payload

    def _is_rich_capability_error(self, exc: Exception) -> bool:
        """True ⇒ the rich endpoint itself is unavailable (old PTB/server); latches rich off.

        Per-message BadRequests (parser/limit) are NOT capability errors.
        """
        name = exc.__class__.__name__.lower()
        if name in {"endpointnotfound", "invalidtoken"}:
            return True
        if isinstance(exc, (AttributeError, TypeError, NotImplementedError)):
            return True
        if getattr(exc, "error_code", None) == 404:
            return True
        s = str(exc).lower()
        if ("method" in s or "endpoint" in s) and ("not found" in s or "does not exist" in s):
            return True
        return "no such method" in s

    def _is_rich_fallback_error(self, exc: Exception) -> bool:
        """True ⇒ permanent/capability error ⇒ safe to fall back to legacy.

        Conservative on purpose: anything not clearly permanent is transient — the
        rich request may have reached Telegram, so a legacy resend risks a duplicate.
        """
        if self._is_bad_request_error(exc):
            return True
        if self._is_rich_capability_error(exc):
            return True
        s = str(exc).lower()
        return "unsupported" in s or "not implemented" in s

    def _chunk_reply_routing(
        self, chat_id: str, reply_to: Optional[str], metadata: Optional[Dict[str, Any]],
        thread_id: Optional[str], index: int,
    ) -> tuple:
        """Reply-anchor routing for chunk ``index``: ``(private_dm_topic_send, anchor_off, reply_to_id)``.

        ``anchor_off``: reply_to_mode="off" on the telegram_dm_topic_reply_fallback path is an
        explicit opt-in to "message_thread_id alone is enough" — don't fail loud just because
        the anchor was suppressed by config.
        """
        metadata_reply_to = self._metadata_reply_to_message_id(metadata)
        private_dm_topic_send = self._is_private_dm_topic_send(chat_id, thread_id, metadata)
        dm_topic_reply_to_off = (
            private_dm_topic_send and self._reply_to_mode == "off"
            and bool(metadata and metadata.get("telegram_dm_topic_reply_fallback"))
        )
        reply_to_source = reply_to or (
            str(metadata_reply_to) if private_dm_topic_send and metadata_reply_to is not None else None
        )
        if private_dm_topic_send:
            should_thread = reply_to_source is not None and self._reply_to_mode != "off"
        else:
            should_thread = self._should_thread_reply(reply_to_source, index)
        reply_to_id = int(reply_to_source) if should_thread and reply_to_source else None
        return private_dm_topic_send, dm_topic_reply_to_off, reply_to_id

    def _compute_single_send_routing(
        self, chat_id: str, reply_to: Optional[str], metadata: Optional[Dict[str, Any]],
        thread_id: Optional[str],
    ) -> Optional[tuple]:
        """Routing for a single (rich) send — mirrors send()'s index-0 block.

        Returns ``(reply_to_id, thread_kwargs)``, or ``None`` = "skip rich, let legacy
        handle it" (DM-topic fail-loud case; legacy owns the refuse SendResult).
        """
        private_dm_topic_send, dm_topic_reply_to_off, reply_to_id = self._chunk_reply_routing(
            chat_id, reply_to, metadata, thread_id, 0
        )
        thread_kwargs = self._thread_kwargs_for_send(
            chat_id, thread_id, metadata, reply_to_message_id=reply_to_id, reply_to_mode=self._reply_to_mode
        )
        # Refusing to send outside the requested DM topic — defer to legacy (canonical
        # fail-loud SendResult). Synthetic/resumed sends via direct_messages_topic_id
        # need no reply anchor.
        if (
            private_dm_topic_send and reply_to_id is None and not dm_topic_reply_to_off
            and not thread_kwargs.get("direct_messages_topic_id")
        ):
            return None
        return reply_to_id, thread_kwargs

    def _rich_transient_result(self, exc: Exception, what: str, *, retry_after: Any = None) -> SendResult:
        """SendResult for a transient/unknown rich-API failure (request may have reached
        Telegram, so the caller must NOT legacy-resend); retry semantics mirror legacy send()."""
        err_str = str(exc).lower()
        try:
            from telegram.error import TimedOut as _TimedOut
        except (ImportError, AttributeError):
            _TimedOut = None
        is_timeout = (_TimedOut and isinstance(exc, _TimedOut)) or "timed out" in err_str
        is_connect_timeout = self._looks_like_connect_timeout(exc)
        safe_error = _redact_telegram_error_text(exc)
        logger.warning("[%s] %s transient failure (no legacy resend): %s", self.name, what, safe_error)
        return SendResult(
            success=False, error=safe_error, retryable=(is_connect_timeout or not is_timeout),
            retry_after=retry_after,
        )

    @staticmethod
    def _record_rich_sent(chat_id: Any, message_id: Any, content: str) -> None:
        """Index rich content we sent: Telegram won't echo it back in reply_to_message."""
        try:
            from gateway import rich_sent_store
            rich_sent_store.record(str(chat_id), str(message_id), content)
        except Exception:
            pass

    async def _try_send_rich(
        self, chat_id: str, content: str, reply_to: Optional[str], metadata: Optional[Dict[str, Any]]
    ) -> Optional[SendResult]:
        """Attempt a single ``sendRichMessage`` send.

        Returns a SendResult (success, or a transient failure the caller must NOT
        legacy-resend), or ``None`` = fall back to legacy MarkdownV2 (permanent/capability
        error or DM-topic skip).
        """
        thread_id = self._metadata_thread_id(metadata)
        routing = self._compute_single_send_routing(chat_id, reply_to, metadata, thread_id)
        if routing is None:
            return None
        reply_to_id, thread_kwargs = routing
        payload: Dict[str, Any] = {
            "chat_id": normalize_telegram_chat_id(chat_id),
            "rich_message": self._rich_message_payload(content),
        }
        # Only forward non-None routing keys: direct_messages_topic_id comes paired
        # with message_thread_id=None, which must not hit the raw endpoint.
        payload.update({k: v for k, v in thread_kwargs.items() if v is not None})
        payload.update(self._notification_kwargs(metadata))
        if getattr(self, "_disable_link_previews", False):
            payload["link_preview_options"] = {"is_disabled": True}
        if reply_to_id is not None:
            # sendRichMessage takes reply_parameters, NOT the legacy reply_to_message_id
            # scalar; the Bot API silently ignores unknown params, dropping the anchor.
            payload["reply_parameters"] = {"message_id": reply_to_id}
        try:
            # Raw Bot API result: return_type=Message would make PTB deserialize a 10.1
            # shape it doesn't fully model; a post-delivery parse error ≠ send failure.
            msg = await self._bot.do_api_request("sendRichMessage", api_kwargs=payload)
        except Exception as exc:
            if self._is_rich_fallback_error(exc):
                if self._is_rich_capability_error(exc):
                    # Endpoint missing — latch rich off to avoid a doomed roundtrip per send.
                    self._rich_send_disabled = True
                logger.debug(
                    "[%s] sendRichMessage rejected (%s) — falling back to MarkdownV2",
                    self.name, _redact_telegram_error_text(exc),
                )
                return None
            # Honor Telegram's flood-control retry_after over the base retry schedule.
            _retry_after = getattr(exc, "retry_after", None)
            if _retry_after is None:
                _m = re.search(r"retry\s+(?:in\s+)?(\d+)", str(exc).lower(), re.IGNORECASE)
                if _m:
                    _retry_after = float(_m.group(1))
            return self._rich_transient_result(exc, "sendRichMessage", retry_after=_retry_after)
        message_id = None
        if isinstance(msg, dict):
            message_id = msg.get("message_id")
            if message_id is None:
                message_id = (msg.get("result") or {}).get("message_id")
        else:
            message_id = getattr(msg, "message_id", None)
        if message_id is not None:
            self._record_rich_sent(chat_id, message_id, content)
        return SendResult(success=True, message_id=str(message_id) if message_id is not None else None)

    async def _try_edit_rich(
        self, chat_id: str, message_id: str, content: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[SendResult]:
        """Edit a message in place as rich (``editMessageText`` + ``rich_message``).

        Lets a streamed preview finalize as rich without send+delete. Same error
        contract as :meth:`_try_send_rich`: success → SendResult(True); permanent/
        capability → ``None`` (legacy edit; capability latches rich off); transient →
        SendResult(False) with retry semantics (may already be edited; no legacy resend).
        """
        payload: Dict[str, Any] = {
            "chat_id": normalize_telegram_chat_id(chat_id),
            "message_id": int(message_id),
            "rich_message": self._rich_message_payload(content),
        }
        # No topic routing on edits: message_thread_id/direct_messages_topic_id make
        # Telegram reject the rich edit and force the legacy table-to-bullets fallback.
        if getattr(self, "_disable_link_previews", False):
            payload["link_preview_options"] = {"is_disabled": True}
        try:
            # Raw result; no return_type=Message (see _try_send_rich).
            await self._bot.do_api_request("editMessageText", api_kwargs=payload)
        except Exception as exc:
            if self._is_rich_fallback_error(exc):
                if self._is_rich_capability_error(exc):
                    self._rich_send_disabled = True
                # "Message is not modified" = successful no-op; skip the redundant legacy edit.
                if "not modified" in str(exc).lower():
                    return SendResult(success=True, message_id=message_id)
                logger.debug(
                    "[%s] rich editMessageText rejected (%s) — falling back to MarkdownV2 edit",
                    self.name, _redact_telegram_error_text(exc),
                )
                return None
            if "not modified" in str(exc).lower():
                return SendResult(success=True, message_id=message_id)
            return self._rich_transient_result(exc, "rich editMessageText")
        # Mirror the fresh-send index: a streamed final finalized via edit is otherwise never recorded.
        self._record_rich_sent(chat_id, message_id, content)
        return SendResult(success=True, message_id=message_id)

    def _should_attempt_rich_draft(self, content: str) -> bool:
        return bool(
            getattr(self, "_rich_messages_enabled", True)
            and getattr(self, "_rich_drafts_enabled", False)
            and not getattr(self, "_rich_send_disabled", False)
            and not getattr(self, "_rich_draft_disabled", False)
            and content and content.strip()
            and not self._has_telegram_desktop_details_math_crash_shape(content)
            and not self._has_telegram_desktop_cjk_rich_garble_shape(content)
            and self._content_fits_rich_limits(content)
            and self._bot_supports_rich()
        )

    async def _try_send_rich_draft(
        self, chat_id: str, draft_id: int, content: str, metadata: Optional[Dict[str, Any]]
    ) -> bool:
        """Emit one ``sendRichMessageDraft`` preview frame; True on success.

        Draft frames are ephemeral (overwritten by the next frame / final send), so a
        lost or duplicate frame is harmless: any failure returns False and the caller
        renders the legacy draft. Capability failures latch ``_rich_draft_disabled``.
        """
        payload: Dict[str, Any] = {
            "chat_id": normalize_telegram_chat_id(chat_id),
            "draft_id": int(draft_id),
            "rich_message": self._rich_message_payload(content),
        }
        payload.update(self._thread_kwargs_for_draft(chat_id, metadata))
        try:
            ok = await self._bot.do_api_request("sendRichMessageDraft", api_kwargs=payload)
            return bool(ok)
        except Exception as exc:
            if self._is_rich_capability_error(exc):
                self._rich_draft_disabled = True
                logger.debug(
                    "[%s] sendRichMessageDraft unsupported (%s) — using legacy drafts",
                    self.name, _redact_telegram_error_text(exc),
                )
            else:
                logger.debug(
                    "[%s] sendRichMessageDraft transient failure (%s) — legacy draft this frame",
                    self.name, _redact_telegram_error_text(exc),
                )
            return False

    async def _drain_polling_connections(self) -> None:
        """Reset the httpx pool used for getUpdates polling before a reconnect.

        Network errors (esp. via proxies like sing-box) leave half-closed httpx
        connections occupying pool slots until ``Pool timeout: All connections in the
        connection pool are occupied.`` Only ``_request[0]`` (getUpdates) is reset;
        the general request (``_request[1]``) stays untouched so concurrent sends/edits
        are never interrupted. Relies on PTB 22.x's private ``(get_updates, general)``
        tuple — review on PTB 23+.
        """
        if not (self._app and self._app.bot):
            return
        try:
            polling_req = self._app.bot._request[0]  # noqa: SLF001
        except Exception:
            return
        shutdown_ok = False
        try:
            # Bounded: a wedged CLOSE-WAIT socket can hang this close forever and freeze
            # the reconnect ladder. Wall-clock thread deadline, not asyncio.wait_for:
            # httpcore's pool close runs under AsyncShieldCancellation and wedges wait_for.
            await _await_with_thread_deadline(polling_req.shutdown(), timeout=_DRAIN_TIMEOUT)
            shutdown_ok = True
        except Exception:
            logger.debug(
                "[%s] Polling request shutdown failed/timed out (non-fatal)", self.name, exc_info=True
            )
        if not shutdown_ok:
            # initialize() only rebuilds the client when ``client.is_closed``; an abandoned
            # aclose() leaves it false, so start_polling would reuse the CLOSE-WAIT socket
            # (alive but deaf). Swap in a fresh client first.
            self._orphan_and_rebuild_polling_client(polling_req)
        try:
            await _await_with_thread_deadline(polling_req.initialize(), timeout=_DRAIN_TIMEOUT)
            logger.debug("[%s] Polling request pool drained before reconnect", self.name)
        except Exception:
            logger.debug(
                "[%s] Polling request re-initialize failed/timed out (non-fatal)", self.name, exc_info=True
            )
            self._orphan_and_rebuild_polling_client(polling_req)

    def _orphan_and_rebuild_polling_client(self, polling_req) -> None:
        """Replace a wedged HTTPXRequest client after a hung aclose().

        ``initialize()`` only calls ``_build_client()`` when the client reports
        ``is_closed``; after an abandoned shutdown() that flag stays false and polling
        would reuse the dead connection. Swap in a fresh client and close the old one
        in a detached, bounded background task so it can't block the reconnect ladder.
        """
        old = getattr(polling_req, "_client", None)
        build = getattr(polling_req, "_build_client", None)
        if old is None or not callable(build):
            return
        if getattr(old, "is_closed", True):
            return
        try:
            polling_req._client = build()  # noqa: SLF001
        except Exception:
            logger.debug(
                "[%s] Failed to rebuild polling HTTP client after hung drain", self.name, exc_info=True
            )
            return
        logger.warning(
            "[%s] Replaced wedged getUpdates HTTP client after drain timeout (likely CLOSE-WAIT socket)",
            self.name,
        )

        async def _orphan_aclose() -> None:
            try:
                aclose = getattr(old, "aclose", None)
                if not callable(aclose):
                    return
                # Same cancellation-swallowing httpcore scope as shutdown(): wall-clock
                # deadline so this cleanup can't hang and leak one task per reconnect.
                await _await_with_thread_deadline(aclose(), timeout=_DRAIN_TIMEOUT)
            except Exception:
                logger.debug(
                    "[%s] Orphan polling client aclose failed (non-fatal)", self.name, exc_info=True
                )

        try:
            task = asyncio.ensure_future(_orphan_aclose())
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)
            task.add_done_callback(_consume_abandoned_task)
        except Exception:
            pass

    def _begin_polling_generation(self) -> tuple[int, asyncio.Event]:
        """Start accepting progress for a new getUpdates polling generation."""
        if self._teardown_started:
            self._polling_progress_accepting = False
            self._send_path_degraded = True
            progress = getattr(self, "_polling_progress_event", None)
            if progress is None:
                progress = asyncio.Event()
                self._polling_progress_event = progress
            return getattr(self, "_polling_generation", 0), progress
        verifier = getattr(self, "_polling_progress_verifier_task", None)
        if verifier is not None and not verifier.done():
            verifier.cancel()
        self._polling_progress_verifier_task = None
        self._polling_generation = getattr(self, "_polling_generation", 0) + 1
        self._polling_progress_event = asyncio.Event()
        self._polling_progress_accepting = True
        self._send_path_degraded = True
        # Reset stall-watchdog timestamps: no proven progress yet, age measured from here.
        self._polling_generation_started_monotonic = time.monotonic()
        self._polling_last_progress_monotonic = None
        return self._polling_generation, self._polling_progress_event

    def _record_polling_progress(self, generation: int) -> None:
        """Record successful getUpdates I/O for the current generation only."""
        if self._teardown_started:
            return
        if not self._polling_progress_accepting:
            return
        if generation != self._polling_generation:
            return
        if not self._polling_progress_event.is_set():
            # First confirmed round-trip resolves the "health pending" log line both
            # reconnect paths end on; otherwise "healthy" and "hung" log identically.
            logger.info(
                "[%s] Telegram polling confirmed healthy: getUpdates progressing (generation %d)",
                self.name, generation,
            )
        self._polling_progress_event.set()
        self._polling_last_progress_monotonic = time.monotonic()
        self._polling_network_error_count = 0
        if generation == self._polling_conflict_recovery_generation:
            self._polling_conflict_recovery_generation = None
        else:
            self._polling_conflict_count = 0
        self._send_path_degraded = False

    def _observe_polling_request_result(self, request, generation, result):
        """Record getUpdates progress from an observed do_request result.

        Purely observational: PTB still parses the untouched payload and owns any
        resulting exception. Separate method so it is shared and testable.
        """
        status_code, payload = result
        if generation is None or not (200 <= status_code < 300):
            return
        try:
            # The request's own parser keeps health observation in agreement with PTB
            # (UTF-8 replacement decoding, BOM rejection).
            envelope = request.parse_json_payload(payload)
        except Exception:
            return
        if isinstance(envelope, dict) and envelope.get("ok") is True and "result" in envelope:
            self._record_polling_progress(generation)

    def _instrument_polling_request(self, request):
        """Instrument one dedicated PTB getUpdates request with progress tracking.

        PTB request classes use ``__slots__`` and on Python 3.13 have no ``__dict__``,
        so ``request.do_request = wrapper`` raises AttributeError. Instead re-tag the
        instance to a thin ``__slots__ = ()`` subclass overriding ``do_request`` —
        identical layout makes the ``__class__`` swap legal; works for test doubles too.
        """
        adapter = self
        base_cls = type(request)

        class _InstrumentedPollingRequest(base_cls):
            __slots__ = ()

            async def do_request(self, *args, **kwargs):
                generation = _POLLING_GENERATION_CONTEXT.get()
                result = await super().do_request(*args, **kwargs)
                adapter._observe_polling_request_result(self, generation, result)
                return result

        request.__class__ = _InstrumentedPollingRequest
        return request

    async def _start_polling_once(
        self, app, *, drop_pending_updates: bool, error_callback,
        abandon_app_on_timeout: bool = False, schedule_verifier: bool = True,
    ) -> tuple[int, asyncio.Event]:
        """Start one generation and verify real getUpdates progress.

        Returns this generation's ``(generation, progress_event)`` so readiness-gating
        callers bind to exactly it — a concurrent recovery task may already have
        replaced ``self._polling_progress_event`` with a newer generation's event.
        """
        if self._teardown_started:
            raise _PollingLifecycleAbort("Telegram polling teardown started")
        generation, progress = self._begin_polling_generation()
        if not self._polling_progress_accepting:
            raise _PollingLifecycleAbort("Telegram polling teardown started")

        def _generation_error_callback(error: Exception) -> None:
            if self._teardown_started:
                return
            if generation != self._polling_generation:
                return
            if error_callback is not None:
                callback_context_token = _POLLING_GENERATION_CONTEXT.set(None)
                try:
                    error_callback(error)
                finally:
                    _POLLING_GENERATION_CONTEXT.reset(callback_context_token)

        context_token = _POLLING_GENERATION_CONTEXT.set(generation)
        try:
            # asyncio.wait_for can wait forever on httpcore/AnyIO shielded scopes; use the
            # wall-deadline helper and abandon the partial updater (caller rebuilds).
            await _await_with_thread_deadline(
                app.updater.start_polling(
                    allowed_updates=Update.ALL_TYPES,
                    drop_pending_updates=drop_pending_updates,
                    error_callback=_generation_error_callback,
                ),
                timeout=_UPDATER_START_TIMEOUT,
                on_abandon=(
                    (lambda app=app: _shutdown_abandoned_app(app)) if abandon_app_on_timeout else None
                ),
            )
        finally:
            _POLLING_GENERATION_CONTEXT.reset(context_token)
        if self._teardown_started:
            self._polling_progress_accepting = False
            self._send_path_degraded = True
            raise _PollingLifecycleAbort("Telegram polling teardown started")
        if schedule_verifier:
            self._schedule_polling_progress_verifier(generation, progress)
        return generation, progress

    def _schedule_polling_progress_verifier(self, generation: int, progress: asyncio.Event) -> None:
        """Own exactly one tracked verifier for the current generation."""
        if self._teardown_started:
            self._polling_progress_accepting = False
            self._send_path_degraded = True
            return
        previous = getattr(self, "_polling_progress_verifier_task", None)
        if previous is not None and not previous.done():
            previous.cancel()
        task = asyncio.get_running_loop().create_task(
            self._verify_polling_after_reconnect(generation, progress)
        )
        self._polling_progress_verifier_task = task
        self._background_tasks.add(task)

        def _clear_finished_verifier(finished: asyncio.Task) -> None:
            self._background_tasks.discard(finished)
            if self._polling_progress_verifier_task is finished:
                self._polling_progress_verifier_task = None

        task.add_done_callback(_clear_finished_verifier)

    def _get_general_request_drain_lock(self) -> asyncio.Lock:
        lock = getattr(self, "_general_request_drain_lock", None)
        if lock is None:
            lock = asyncio.Lock()
            self._general_request_drain_lock = lock
        return lock

    async def _drain_general_connections_after_pool_timeout(self) -> None:
        """Reset the general Bot API pool (``_request[1]``) after a confirmed send pool timeout.

        When httpx reports pool exhaustion PTB guarantees the request was not sent, so
        resetting the wedged pool before retrying is safe.
        """
        bot = getattr(getattr(self, "_app", None), "bot", None)
        if bot is None:
            bot = getattr(self, "_bot", None)
        if bot is None:
            return
        try:
            general_req = bot._request[1]  # noqa: SLF001
        except Exception:
            return
        async with self._get_general_request_drain_lock():
            try:
                await _await_with_thread_deadline(general_req.shutdown(), timeout=_DRAIN_TIMEOUT)
            except Exception:
                logger.debug(
                    "[%s] General request shutdown failed/timed out after pool timeout (non-fatal)",
                    self.name, exc_info=True,
                )
            try:
                await _await_with_thread_deadline(general_req.initialize(), timeout=_DRAIN_TIMEOUT)
                logger.warning("[%s] General request pool drained after Telegram pool timeout", self.name)
            except Exception:
                logger.debug(
                    "[%s] General request re-initialize failed/timed out after pool timeout (non-fatal)",
                    self.name, exc_info=True,
                )

    def _spawn_polling_recovery(self, loop, coro) -> None:
        """Start ``coro`` as the tracked in-flight recovery task (reentrancy guard)."""
        self._polling_error_task = loop.create_task(coro)
        self._background_tasks.add(self._polling_error_task)
        self._polling_error_task.add_done_callback(self._background_tasks.discard)

    def _schedule_polling_recovery(self, error: Exception, *, reason: str) -> None:
        """Schedule background polling recovery without failing gateway startup.

        A transient bootstrap failure (deleteWebhook / initial start_polling) degrades
        only this adapter; the reconnect ladder recovers in the background.
        """
        if self._teardown_started:
            return
        if self.has_fatal_error:
            return
        if self._polling_error_task and not self._polling_error_task.done():
            logger.debug(
                "[%s] Telegram polling recovery already scheduled; ignoring %s: %s",
                self.name, reason, _redact_telegram_error_text(error),
            )
            return
        self._send_path_degraded = True
        logger.warning(
            "[%s] Telegram polling degraded (%s); gateway stays alive and will retry. Error: %s",
            self.name, reason, _redact_telegram_error_text(error),
        )
        self._spawn_polling_recovery(asyncio.get_running_loop(), self._handle_polling_network_error(error))

    async def _delete_webhook_best_effort(self, *, require_success: bool = False) -> bool:
        """Clear a stale webhook; ``require_success`` fails closed on cold start.

        Reconnects recover transient errors in background; cold start raises so
        GatewayRunner disposes the partial adapter and retries with a fresh Application.
        """
        if not self._bot:
            return False
        delete_webhook = getattr(self._bot, "delete_webhook", None)
        if not callable(delete_webhook):
            return True
        try:
            # Same shielded-cancellation class as initialize/start_polling: never let a
            # wedged deleteWebhook pin initial connect.
            await _await_with_thread_deadline(
                delete_webhook(drop_pending_updates=False), timeout=_UPDATER_START_TIMEOUT
            )
            return True
        except Exception as err:
            if self._looks_like_network_error(err):
                if require_success:
                    raise OSError("Telegram deleteWebhook did not complete during initial connect") from err
                logger.warning(
                    "[%s] deleteWebhook failed with a recoverable network error; "
                    "continuing to polling so getUpdates/retry can recover: %s",
                    self.name, _redact_telegram_error_text(err),
                )
                self._send_path_degraded = True
                return False
            raise

    async def _start_polling_resilient(
        self, *, drop_pending_updates: bool, error_callback, require_progress: bool = False
    ) -> bool:
        """Start PTB polling; ``require_progress`` (initial connect) demands real readiness.

        Reconnects may recover in background. On cold start a bootstrap failure or a
        missing first getUpdates success raises so GatewayRunner disposes this partial
        adapter and retries with a fresh PTB Application.
        """
        if self._teardown_started:
            return False
        if not (self._app and self._app.updater):
            raise RuntimeError("Telegram application/updater not initialized")
        # Strict cold start: background recovery must not run while the readiness gate
        # waits, else a G1 error starts G2 on the same partial app — the cold connect
        # times out on G1 despite G2 succeeding, or G2 "heals" the partial app so
        # GatewayRunner never disposes it. Capture the first error and fail immediately.
        strict_error: list[BaseException] = []
        strict_error_event = asyncio.Event()
        strict_gate_open = True
        effective_callback = error_callback
        if require_progress:
            loop = asyncio.get_running_loop()

            def _strict_error_callback(error: Exception) -> None:
                # Registered for the whole generation: once the gate closes, delegate to
                # the real callback so later errors still reach background recovery.
                if not strict_gate_open:
                    if error_callback is not None:
                        error_callback(error)
                    return
                if not strict_error:
                    strict_error.append(error)
                # Called from the polling task; set on the loop to wake the strict waiter.
                loop.call_soon_threadsafe(strict_error_event.set)

            effective_callback = _strict_error_callback
        try:
            # Same watchdog bound as the reconnect ladders: a wedged pool can hang
            # start_polling() at bootstrap too. The TimeoutError is an OSError subclass,
            # so the except below classifies it as a network error → background recovery.
            generation, progress = await self._start_polling_once(
                self._app, drop_pending_updates=drop_pending_updates, error_callback=effective_callback,
                abandon_app_on_timeout=require_progress,
                # The strict gate IS the cold-start verifier; a background one would race it.
                schedule_verifier=not require_progress,
            )
            if require_progress:
                # Bind to THIS generation's event, not self._polling_progress_event.
                progress_wait = asyncio.ensure_future(progress.wait())
                error_wait = asyncio.ensure_future(strict_error_event.wait())
                try:
                    await _await_with_thread_deadline(
                        _first_completed(progress_wait, error_wait),
                        timeout=_INITIAL_POLLING_PROGRESS_TIMEOUT,
                    )
                except asyncio.TimeoutError as exc:
                    raise OSError(
                        "Telegram getUpdates made no progress within "
                        f"{_INITIAL_POLLING_PROGRESS_TIMEOUT:.0f}s during initial "
                        "connect — failing startup so the gateway retries with a "
                        "fresh adapter (#67498)"
                    ) from exc
                finally:
                    for fut in (progress_wait, error_wait):
                        if not fut.done():
                            fut.cancel()
                    await asyncio.gather(progress_wait, error_wait, return_exceptions=True)
                if strict_error and not progress.is_set():
                    raise OSError(
                        "Telegram polling errored before first getUpdates "
                        "success during initial connect: "
                        f"{_redact_telegram_error_text(strict_error[0])}"
                    ) from strict_error[0]
                if not progress.is_set():
                    raise OSError("Telegram getUpdates did not become ready during initial connect")
                # Readiness proven — close the gate so later errors reach background recovery.
                strict_gate_open = False
                self._polling_error_callback_ref = error_callback
            return True
        except _PollingLifecycleAbort:
            return False
        except Exception as err:
            if self._teardown_started:
                return False
            if require_progress:
                raise
            if self._looks_like_polling_conflict(err):
                logger.warning(
                    "[%s] Telegram polling bootstrap conflict; gateway stays alive "
                    "while conflict retry runs: %s",
                    self.name, _redact_telegram_error_text(err),
                )
                self._spawn_polling_recovery(asyncio.get_running_loop(), self._handle_polling_conflict(err))
                return False
            if self._looks_like_network_error(err):
                self._schedule_polling_recovery(err, reason="polling bootstrap")
                return False
            raise

    async def _stop_updater_or_go_fatal(self, app, what: str) -> bool:
        """Bounded ``updater.stop()`` before a recovery restart; False = went fatal, caller returns.

        Wall-clock deadline, not asyncio.wait_for: a CLOSE-WAIT socket wedges stop() on
        epoll and PTB/AnyIO cancellation-shielded cleanup hangs wait_for. On timeout the
        Updater's lifecycle lock may still be held, so rebuild the adapter instead.
        """
        try:
            if app and app.updater and app.updater.running:
                try:
                    await _await_with_thread_deadline(app.updater.stop(), timeout=_UPDATER_STOP_TIMEOUT)
                except asyncio.TimeoutError:
                    message = (
                        f"Telegram updater.stop() did not finish before the {what} deadline; "
                        "rebuilding the adapter instead of reusing an Updater whose lifecycle "
                        "lock may still be held."
                    )
                    logger.error("[%s] %s (likely CLOSE-WAIT socket)", self.name, message)
                    self._set_fatal_error("telegram_network_error", message, retryable=True)
                    await self._handoff_polling_fatal_error()
                    return False
        except Exception:
            pass
        return True

    async def _handle_polling_network_error(self, error: Exception) -> None:
        """Reconnect polling after a transient network interruption (NetworkError/TimedOut).

        The host losing connectivity (sleep, WiFi switch, VPN) kills the long-poll silently
        while the process lives. Exponential back-off (5s→60s cap) up to MAX_NETWORK_RETRIES,
        then retryable-fatal so the supervisor restarts the gateway.
        """
        if self._teardown_started:
            return
        if self.has_fatal_error:
            return
        MAX_NETWORK_RETRIES = 10
        BASE_DELAY = 5
        MAX_DELAY = 60
        self._polling_network_error_count += 1
        self._send_path_degraded = True
        attempt = self._polling_network_error_count
        if attempt > MAX_NETWORK_RETRIES:
            message = (
                "Telegram polling could not reconnect after %d network error retries. "
                "Escalating to gateway recovery." % MAX_NETWORK_RETRIES
            )
            logger.error("[%s] %s Last error: %s", self.name, message, _redact_telegram_error_text(error))
            self._set_fatal_error("telegram_network_error", message, retryable=True)
            await self._handoff_polling_fatal_error()
            return
        delay = min(BASE_DELAY * (2 ** (attempt - 1)), MAX_DELAY)
        safe_error = _redact_telegram_error_text(error)
        logger.warning(
            "[%s] Telegram network error (attempt %d/%d), reconnecting in %ds. Error: %s",
            self.name, attempt, MAX_NETWORK_RETRIES, delay, safe_error,
        )
        await asyncio.sleep(delay)
        if self._teardown_started:
            return
        # Stable local ref: a concurrent disconnect() may set self._app = None while we
        # await below; fail fast in one place instead of swapping in None mid-sequence.
        app = self._app
        # Unguarded stop() on a CLOSE-WAIT socket would leave _polling_error_task
        # perpetually "in-flight" so every probe skips reconnect for hours.
        if not await self._stop_updater_or_go_fatal(app, "network-recovery"):
            return
        if self._teardown_started:
            return
        # start_polling() bootstraps through the *general* pool before getUpdates; if
        # stale proxy sockets exhausted it, draining only the polling pool can't recover.
        # A confirmed pool timeout means the request was never sent, so rebuilding the
        # general pool is safe. Generic network errors stay polling-only (sends untouched).
        if self._looks_like_pool_timeout(error):
            await self._drain_general_connections_after_pool_timeout()
        if self._teardown_started:
            return
        await self._drain_polling_connections()
        if self._teardown_started:
            return
        try:
            if not app:
                raise RuntimeError("Telegram application was torn down during reconnect")
            await self._start_polling_once(
                app, drop_pending_updates=False, error_callback=self._polling_error_callback_ref,
            )
            logger.info(
                "[%s] Telegram polling restarted after network error (attempt %d); "
                "health pending getUpdates progress",
                self.name, attempt,
            )
        except _PollingLifecycleAbort:
            return
        except Exception as retry_err:
            if self._teardown_started:
                return
            safe_retry_error = _redact_telegram_error_text(retry_err)
            logger.warning("[%s] Telegram polling reconnect failed: %s", self.name, safe_retry_error)
            # Polling is dead and no more error callbacks will fire — chain the retry ourselves.
            if not self.has_fatal_error and not self._teardown_started:
                task = asyncio.ensure_future(self._handle_polling_network_error(retry_err))
                self._background_tasks.add(task)
                task.add_done_callback(self._background_tasks.discard)
                # The chained retry IS the in-flight recovery: it must replace the reentrancy
                # guard, or heartbeat/pending-probe/error_callback each start a second one.
                self._polling_error_task = task

    async def _polling_heartbeat_loop(self) -> None:
        """Detect dead Telegram TCP sockets (CLOSE-WAIT) by periodic probing.

        In CLOSE-WAIT epoll still reports the long-poll socket readable and nothing
        raises, so PTB's ``error_callback`` never fires and receiving silently stops.
        Probe ``get_me()`` every HEARTBEAT_INTERVAL on the *general* path (never the
        getUpdates pool, so a healthy long-poll is not interrupted); any connect-level
        failure feeds ``_handle_polling_network_error``. Unlike the one-shot generation
        verifier this runs for the connection's lifetime, catching steady-state wedges.
        """
        HEARTBEAT_INTERVAL = 90   # seconds between probes
        PROBE_TIMEOUT = 15        # seconds before declaring the path dead
        # Wedged-recovery watchdog, tracked locally so no _polling_error_task assignment
        # needs a timestamp: note when a recovery task is first seen in-flight and
        # force-escalate if the *same* task object still runs past the stuck timeout
        # (a healthy ladder attempt finishes or chains to a new task well before then).
        stuck_task_ref: Optional[asyncio.Task] = None
        stuck_task_since = 0.0
        while True:
            try:
                await asyncio.sleep(HEARTBEAT_INTERVAL)
                if self._teardown_started:
                    return
                if self.has_fatal_error:
                    return
                # If the recovery task hung on an unbounded await, every other recovery
                # path is gated behind it forever (alive but deaf). Force retryable-fatal
                # so the background reconnector rebuilds the adapter.
                recovery_task = self._polling_error_task
                if recovery_task is not None and not recovery_task.done():
                    now = time.monotonic()
                    if recovery_task is not stuck_task_ref:
                        stuck_task_ref = recovery_task
                        stuck_task_since = now
                    elif now - stuck_task_since > _POLLING_ERROR_TASK_STUCK_TIMEOUT:
                        stuck_for = now - stuck_task_since
                        logger.error(
                            "[%s] Telegram reconnect task wedged for %.0fs with no "
                            "ladder progress; forcing retryable-fatal so the gateway "
                            "reconnects instead of staying silently deaf.",
                            self.name, stuck_for,
                        )
                        try:
                            recovery_task.cancel()
                        except Exception:
                            pass
                        self._set_fatal_error(
                            "telegram_network_error",
                            "Telegram reconnect task wedged for %.0fs; forcing "
                            "gateway reconnect." % stuck_for,
                            retryable=True,
                        )
                        await self._handoff_polling_fatal_error()
                        return
                else:
                    stuck_task_ref = None
                bot = self._app.bot if self._app else None
                if bot is None:
                    continue
                # No get_me() ⇒ not a live polling client (torn down / test double): exit, don't spin.
                if not callable(getattr(bot, "get_me", None)):
                    return
                await asyncio.wait_for(bot.get_me(), PROBE_TIMEOUT)
                # get_me() refreshes PTB's cached bot user, so a BotFather rename is
                # picked up here — adopt the reported handle before anything routes on it.
                self._bot_identity_checked_at = time.monotonic()
                self._note_bot_username(getattr(bot, "username", None))
                # get_me() OK proves only the general/send path. PTB can report
                # updater.running=True while the long-poll is wedged and DMs queue
                # server-side; get_webhook_info().pending_update_count exposes that.
                # Escalates only after two consecutive non-zero probes so a single
                # in-flight update never trips recovery.
                await self._probe_pending_updates(bot, PROBE_TIMEOUT)
                # An empty queue can't hide a wedge forever: Telegram answers within ~50s,
                # so no round-trip past the stall threshold ⇒ dead. Pure local-state check.
                await self._check_polling_stall()
            except asyncio.CancelledError:
                return
            except (asyncio.TimeoutError, OSError) as probe_err:
                self._schedule_polling_recovery(probe_err, reason="heartbeat probe")
            except Exception as probe_err:
                if self._looks_like_network_error(probe_err):
                    self._schedule_polling_recovery(probe_err, reason="heartbeat probe")
                    continue
                # Non-connectivity errors (e.g. TelegramError 401) aren't CLOSE-WAIT symptoms.
                pass

    async def _probe_pending_updates(self, bot, probe_timeout: float) -> None:
        """Detect a wedged or stopped getUpdates consumer via pending_update_count.

        PTB can report ``updater.running`` while the long-poll is stuck (e.g. WSL2 socket
        epoll keeps reporting readable); get_me() stays healthy on the general path, yet
        DMs queue in the Bot API. A stuck queue over two consecutive probes ⇒ dead
        consumer; recovery reuses ``_handle_polling_network_error``. Also covers the
        updater having stopped entirely (``running=False``, no reconnect in flight),
        where no queue can be reported against a live consumer.
        """
        if self._teardown_started:
            return
        # Polling mode only: in webhook mode Telegram pushes and holds no server-side queue.
        if self._webhook_mode:
            return
        # An in-flight reconnect owns recovery — don't double-trigger, and don't misread its
        # brief stop()->start_polling() window (updater.running transiently False) as dead.
        if self._polling_error_task and not self._polling_error_task.done():
            self._polling_not_running_count = 0
            return
        updater = getattr(self._app, "updater", None) if self._app else None
        if updater is None:
            self._polling_pending_stuck_count = 0
            return
        if not getattr(updater, "running", False):
            # Updater stopped entirely with no reconnect in flight: the long-poll task is
            # gone, general-path calls still succeed, so no error_callback/probe ever fires.
            # Same ladder as a wedged consumer, debounced over two probes so a
            # just-starting updater never trips it.
            self._polling_pending_stuck_count = 0
            self._polling_not_running_count += 1
            logger.warning(
                "[%s] Telegram polling heartbeat: updater stopped while in "
                "polling mode (stuck probe %d/2)",
                self.name, self._polling_not_running_count,
            )
            if self._polling_not_running_count >= 2:
                self._polling_not_running_count = 0
                if self._teardown_started:
                    return
                logger.warning(
                    "[%s] Telegram updater is not running (long-poll task gone); "
                    "triggering polling restart",
                    self.name,
                )
                loop = asyncio.get_running_loop()
                self._polling_error_task = loop.create_task(
                    self._handle_polling_network_error(
                        RuntimeError("Telegram updater stopped while in polling mode")
                    )
                )
            return
        self._polling_not_running_count = 0
        get_webhook_info = getattr(bot, "get_webhook_info", None)
        if not callable(get_webhook_info):
            return
        try:
            info = await asyncio.wait_for(get_webhook_info(), probe_timeout)  # type: ignore[arg-type]
        except (asyncio.TimeoutError, OSError):
            # Connectivity symptom for the get_me() path / outer handler, not a stuck-queue signal.
            return
        pending = int(getattr(info, "pending_update_count", 0) or 0)
        if pending <= 0:
            self._polling_pending_stuck_count = 0
            return
        self._polling_pending_stuck_count += 1
        logger.warning(
            "[%s] Telegram polling heartbeat: %d update(s) queued but not consumed (stuck probe %d/2)",
            self.name, pending, self._polling_pending_stuck_count,
        )
        if self._polling_pending_stuck_count >= 2:
            self._polling_pending_stuck_count = 0
            if self._teardown_started:
                return
            logger.warning(
                "[%s] getUpdates consumer appears wedged (queue not draining); triggering polling restart",
                self.name,
            )
            loop = asyncio.get_running_loop()
            self._polling_error_task = loop.create_task(
                self._handle_polling_network_error(
                    RuntimeError("getUpdates consumer wedged: pending updates not draining")
                )
            )

    async def _check_polling_stall(self) -> None:
        """Watchdog the last successful getUpdates round-trip.

        A long-poll can wedge without raising (TCP dies mid-read, e.g. TUN/proxy route
        flip → CLOSE-WAIT): ``updater.running`` stays True, get_me() stays healthy and
        with nothing queued ``pending_update_count`` stays 0 — every other probe is blind.
        Telegram answers within ~50s, so no round-trip for ``_POLLING_STALL_TIMEOUT`` ⇒
        unambiguously wedged; escalate through the bounded reconnect ladder.
        """
        if self._webhook_mode:
            return
        if self._teardown_started:
            return
        if self.has_fatal_error:
            return
        if self._polling_error_task and not self._polling_error_task.done():
            return
        now = time.monotonic()
        last_progress = getattr(self, "_polling_last_progress_monotonic", None)
        generation_started = getattr(self, "_polling_generation_started_monotonic", None)
        if last_progress is not None:
            stalled_for = now - last_progress
        elif generation_started is not None:
            # No round-trip yet this generation: fallback for when the one-shot verifier
            # (which owns the immediate post-start window) could not run.
            stalled_for = now - generation_started
        else:
            return
        if stalled_for <= _POLLING_STALL_TIMEOUT:
            return
        logger.error(
            "[%s] Telegram polling stalled: no getUpdates progress for %.0fs "
            "(generation %d). Rebuilding the long-poll consumer through the "
            "reconnect ladder instead of staying silently deaf.",
            self.name, stalled_for, getattr(self, "_polling_generation", 0),
        )
        self._spawn_polling_recovery(
            asyncio.get_running_loop(),
            self._handle_polling_network_error(
                RuntimeError("getUpdates made no progress for %.0fs (polling stall watchdog)" % stalled_for)
            ),
        )

    async def _verify_polling_after_reconnect(
        self, generation: Optional[int] = None, progress: Optional[asyncio.Event] = None,
    ) -> None:
        """Require getUpdates progress, using getMe only to classify failure.

        The generation-bound event is set only by a successful getUpdates response; a
        general-path getMe success classifies connectivity but cannot heal polling health.
        Connectivity failures enter the guarded recovery ladder; auth/validation don't churn.
        """
        PROBE_TIMEOUT = 10
        if self._teardown_started:
            return
        if generation is None:
            generation = self._polling_generation
        if progress is None:
            progress = self._polling_progress_event
        try:
            await asyncio.wait_for(progress.wait(), timeout=_POLLING_PROGRESS_TIMEOUT)
        except asyncio.TimeoutError:
            pass
        if self._teardown_started:
            return
        if progress.is_set() or self.has_fatal_error:
            return
        if not self._polling_progress_accepting:
            return
        if generation != self._polling_generation:
            return
        if progress is not self._polling_progress_event:
            return
        app = self._app
        if not (app and app.updater and app.updater.running):
            logger.warning("[%s] Updater made no getUpdates progress and is not running", self.name)
            self._schedule_polling_recovery(
                RuntimeError("Updater not running after polling progress deadline"),
                reason="polling progress verifier: updater not running",
            )
            return
        try:
            await asyncio.wait_for(app.bot.get_me(), PROBE_TIMEOUT)
        except Exception as probe_err:
            if self._teardown_started:
                return
            if self.has_fatal_error or not self._polling_progress_accepting:
                return
            if generation != self._polling_generation:
                return
            if progress is not self._polling_progress_event or progress.is_set():
                return
            if not self._looks_like_network_error(probe_err):
                logger.warning(
                    "[%s] Polling progress verifier hit a non-connectivity error (not retrying): %s",
                    self.name, _redact_telegram_error_text(probe_err),
                )
                return
            logger.warning(
                "[%s] Polling progress verifier connectivity probe failed: %s",
                self.name, _redact_telegram_error_text(probe_err),
            )
            self._schedule_polling_recovery(
                probe_err, reason="polling progress verifier connectivity failure"
            )
            return
        if self._teardown_started:
            return
        if self.has_fatal_error or not self._polling_progress_accepting:
            return
        if generation != self._polling_generation:
            return
        if progress is not self._polling_progress_event or progress.is_set():
            return
        self._schedule_polling_recovery(
            RuntimeError("getUpdates made no progress before verifier deadline"),
            reason="polling progress verifier: general path healthy but getUpdates stalled",
        )

    def _disarm_ptb_retry_loop(self) -> None:
        """Synchronously stop PTB's internal polling retry loop.

        PTB's ``network_retry_loop`` (max_retries=-1) calls our ``error_callback``
        *synchronously* on a TelegramError (incl. 409 Conflict), then re-checks
        ``while is_running()`` and polls again. Our callback only schedules async
        recovery, so PTB keeps polling while we stop→sleep→start_polling: two sessions
        overlap and Telegram returns a fresh 409 on a ~31s cadence. Setting PTB's private
        ``stop_event`` inside the callback makes its loop exit on the next tick; our async
        ``await updater.stop()`` (idempotent) + drain + ``start_polling()`` then builds a
        fresh stop_event so the restart isn't poisoned.

        Best-effort: the attribute is name-mangled and spelled differently across PTB
        versions, so probe both; if absent, do nothing (prior racing behaviour, never
        worse). Deliberately NOT flipping ``updater._running``: stop() raises when
        running is already False and our handler guards on it, so clearing the flag
        would skip the real teardown and leave stop_event set for the next start_polling().
        """
        updater = getattr(self._app, "updater", None) if self._app else None
        if updater is None:
            return
        for attr in ("_Updater__polling_task_stop_event", "_polling_task_stop_event"):
            stop_event = getattr(updater, attr, None)
            if isinstance(stop_event, asyncio.Event):
                if not stop_event.is_set():
                    stop_event.set()
                    logger.debug("[%s] Disarmed PTB polling retry loop via %s", self.name, attr)
                return
        logger.debug(
            "[%s] Could not disarm PTB polling retry loop "
            "(stop_event not found on this PTB version); "
            "falling back to async stop()",
            self.name,
        )

    async def _handle_polling_conflict(self, error: Exception) -> None:
        if self._teardown_started:
            return
        if self.has_fatal_error and self.fatal_error_code == "telegram_polling_conflict":
            return
        # Transient 409s: the previous gateway process was killed (update / --replace
        # handoff) but Telegram holds its getUpdates session open for up to ~30s.
        # Strategy: stop the local updater, wait for the server-side session to expire
        # (RETRY_DELAY grows per attempt), drain the pool, restart — MAX_CONFLICT_RETRIES
        # times before going fatal. A failed retry must never return silently: an updater
        # that is neither running nor fatal drops messages while reporting "connected".
        self._polling_conflict_count += 1
        MAX_CONFLICT_RETRIES = 5
        # 15s, 25s, 35s, 45s, 55s — clears Telegram's ~30s session window without
        # hammering the API on fast-restart loops.
        RETRY_DELAY = 10 + (self._polling_conflict_count * 10)  # seconds
        if self._polling_conflict_count <= MAX_CONFLICT_RETRIES:
            logger.warning(
                "[%s] Telegram polling conflict (%d/%d) — previous session still "
                "held open on Telegram's servers. Waiting %ds for it to expire. "
                "Error: %s",
                self.name, self._polling_conflict_count, MAX_CONFLICT_RETRIES,
                RETRY_DELAY, _redact_telegram_error_text(error),
            )
            # Stop the updater before sleeping (no-op if PTB raised before running was set).
            if not await self._stop_updater_or_go_fatal(self._app, "conflict-retry"):
                return
            await asyncio.sleep(RETRY_DELAY)
            if self._teardown_started:
                return
            await self._drain_polling_connections()
            if self._teardown_started:
                return
            # Stable local ref: a concurrent disconnect() may null self._app across the
            # awaits above; fail fast here (where the except reschedules or escalates)
            # instead of an AttributeError deep inside start_polling.
            app = self._app
            expected_generation = self._polling_generation + 1
            if not app:
                raise RuntimeError("Telegram application was torn down during conflict reconnect")
            # drop_pending_updates=True makes Telegram terminate any other getUpdates
            # session for this token — the previous process's zombie or our own prior
            # retry's still-expiring session. Without it each retry is immediately 409'd
            # by the previous one, recreating the conflict we're recovering from.
            self._polling_conflict_recovery_generation = expected_generation
            try:
                await self._start_polling_once(
                    app, drop_pending_updates=True, error_callback=self._polling_error_callback_ref,
                )
                logger.info(
                    "[%s] Telegram polling restarted after conflict retry %d/%d; "
                    "health pending getUpdates progress",
                    self.name, self._polling_conflict_count, MAX_CONFLICT_RETRIES,
                )
                return
            except _PollingLifecycleAbort:
                return
            except Exception as retry_err:
                if self._teardown_started:
                    return
                logger.warning(
                    "[%s] Telegram polling retry %d/%d failed: %s. Scheduling next attempt.",
                    self.name, self._polling_conflict_count, MAX_CONFLICT_RETRIES,
                    _redact_telegram_error_text(retry_err),
                )
                # Never return silently: alive-and-"connected" with no polling is limbo.
                if (
                    self._polling_conflict_count < MAX_CONFLICT_RETRIES
                    and not self._teardown_started
                ):
                    # get_running_loop(): get_event_loop() raises on 3.10+ when PTB dispatches
                    # the error callback from a context without an attached loop.
                    loop = asyncio.get_running_loop()
                    self._polling_error_task = loop.create_task(self._handle_polling_conflict(retry_err))
                    return
                # Fall through to fatal on the last retry.
            finally:
                if self._polling_conflict_recovery_generation == expected_generation:
                    self._polling_conflict_recovery_generation = None
        if self._teardown_started:
            return
        # Retries exhausted — fatal so the runner surfaces it and the user knows to act.
        message = (
            "Telegram polling could not recover after %d retries (%ds total wait). "
            "The previous gateway session is still held open on Telegram's servers, "
            "or another process is using the same bot token. "
            "To recover: ensure no other Hermes or OpenClaw instance is running "
            "with this token, then restart the gateway with 'hermes gateway restart'."
            % (MAX_CONFLICT_RETRIES, sum(10 + i * 10 for i in range(1, MAX_CONFLICT_RETRIES + 1)))
        )
        logger.error("[%s] %s Original error: %s", self.name, message, _redact_telegram_error_text(error))
        # Snapshot whether WE transition to fatal: a concurrent retry task may be
        # suspended past the entry guard, and the bounded stop() await below yields the
        # loop so it reaches this branch too. Only the first transition notifies.
        _already_fatal = self.has_fatal_error and self.fatal_error_code == "telegram_polling_conflict"
        self._set_fatal_error("telegram_polling_conflict", message, retryable=False)
        try:
            if self._app and self._app.updater:
                await _await_with_thread_deadline(self._app.updater.stop(), timeout=_UPDATER_STOP_TIMEOUT)
        except asyncio.TimeoutError:
            logger.warning(
                "[%s] updater.stop() timed out after exhausting conflict "
                "retries (likely CLOSE-WAIT socket); proceeding to fatal notify",
                self.name,
            )
        except Exception as stop_error:
            logger.warning(
                "[%s] Failed stopping Telegram updater after exhausting conflict retries: %s",
                self.name, stop_error, exc_info=True,
            )
        if not _already_fatal:
            await self._handoff_polling_fatal_error()

    async def _handoff_polling_fatal_error(self) -> None:
        """Notify the runner without letting child teardown cancel this owner.

        The runner bounds adapter cleanup in a child task, and ``disconnect()`` cancels
        the tracked recovery and heartbeat tasks — so leaving the current notifier in
        either field would cancel the fatal callback mid-decision. Release only the
        current owner from whichever field tracks it.
        """
        current_task = asyncio.current_task()
        if self._polling_error_task is current_task:
            self._polling_error_task = None
        if getattr(self, "_polling_heartbeat_task", None) is current_task:
            self._polling_heartbeat_task = None
        await self._notify_fatal_error()

    async def _create_dm_topic(
        self, chat_id: int, name: str, icon_color: Optional[int] = None,
        icon_custom_emoji_id: Optional[str] = None,
    ) -> Optional[int]:
        """Create a forum topic in a private (DM) chat (Bot API 9.4+ createForumTopic).

        Returns the message_thread_id, or None on failure.
        """
        if not self._bot:
            return None
        try:
            kwargs: Dict[str, Any] = {"chat_id": chat_id, "name": name}
            if icon_color is not None:
                kwargs["icon_color"] = icon_color
            if icon_custom_emoji_id:
                kwargs["icon_custom_emoji_id"] = icon_custom_emoji_id
            topic = await self._bot.create_forum_topic(**kwargs)
            thread_id = topic.message_thread_id
            logger.info(
                "[%s] Created DM topic '%s' in chat %s -> thread_id=%s",
                self.name, name, chat_id, thread_id,
            )
            return thread_id
        except Exception as e:
            error_text = str(e).lower()
            # Telegram has no "list topics" API: an existing topic is mapped from incoming messages.
            if "topic_name_duplicate" in error_text or "already" in error_text:
                logger.info(
                    "[%s] DM topic '%s' already exists in chat %s (will be mapped from incoming messages)",
                    self.name, name, chat_id,
                )
            elif "not a forum" in error_text or "forums_disabled" in error_text:
                logger.warning(
                    "[%s] Cannot create DM topic '%s' in chat %s: Topics mode is not enabled. "
                    "The user must open the DM with this bot in Telegram, tap the bot name "
                    "at the top, and enable 'Topics' in chat settings before topics can be created.",
                    self.name, name, chat_id,
                )
            else:
                logger.warning(
                    "[%s] Failed to create DM topic '%s' in chat %s: %s",
                    self.name, name, chat_id, _redact_telegram_error_text(e),
                )
            return None

    async def create_handoff_thread(self, parent_chat_id: str, name: str) -> Optional[str]:
        """Create a forum topic for a session handoff (DM topics or forum supergroups).

        Returns the ``message_thread_id`` as a string, or ``None`` on failure.
        """
        try:
            chat_id_int = int(parent_chat_id)
        except (TypeError, ValueError):
            return None
        thread_id = await self._create_dm_topic(chat_id_int, name=name)
        return str(thread_id) if thread_id else None

    async def ensure_dm_topic(self, chat_id: str, topic_name: str, force_create: bool = False) -> Optional[str]:
        """Return a private DM topic thread id, creating and persisting it if needed."""
        name = str(topic_name or "").strip()
        if not name:
            return None
        try:
            chat_id_int = int(chat_id)
        except (TypeError, ValueError):
            return None
        cache_key = f"{chat_id_int}:{name}"
        cached = self._dm_topics.get(cache_key)
        if cached and not force_create:
            return str(cached)
        topic_conf: Optional[Dict[str, Any]] = None
        chat_entry: Optional[Dict[str, Any]] = None
        for entry in self._dm_topics_config:
            if str(entry.get("chat_id")) != str(chat_id_int):
                continue
            chat_entry = entry
            for candidate in entry.get("topics", []):
                if candidate.get("name") == name:
                    topic_conf = candidate
                    break
            break
        if topic_conf and topic_conf.get("thread_id") and not force_create:
            thread_id = int(topic_conf["thread_id"])
            self._dm_topics[cache_key] = thread_id
            return str(thread_id)
        if chat_entry is None:
            chat_entry = {"chat_id": chat_id_int, "topics": []}
            self._dm_topics_config.append(chat_entry)
        if topic_conf is None:
            topic_conf = {"name": name}
            chat_entry.setdefault("topics", []).append(topic_conf)
        thread_id = await self._create_dm_topic(
            chat_id_int, name=name, icon_color=topic_conf.get("icon_color"),
            icon_custom_emoji_id=topic_conf.get("icon_custom_emoji_id"),
        )
        if not thread_id:
            return None
        topic_conf["thread_id"] = thread_id
        self._dm_topics[cache_key] = int(thread_id)
        self._persist_dm_topic_thread_id(chat_id_int, name, int(thread_id), replace_existing=force_create)
        return str(thread_id)

    async def rename_dm_topic(self, chat_id: int, thread_id: int, name: str) -> None:
        """Rename a forum topic in a private (DM) chat."""
        if not self._bot:
            return
        try:
            chat_id_arg = int(chat_id)
        except (TypeError, ValueError):
            chat_id_arg = chat_id
        await self._bot.edit_forum_topic(chat_id=chat_id_arg, message_thread_id=int(thread_id), name=name)
        logger.info(
            "[%s] Renamed DM topic in chat %s thread_id=%s -> '%s'", self.name, chat_id, thread_id, name,
        )

    def _persist_dm_topic_thread_id(
        self, chat_id: int, topic_name: str, thread_id: int, replace_existing: bool = False,
    ) -> None:
        """Save a newly created thread_id back into config.yaml so it survives restarts."""
        try:
            from hermes_constants import get_hermes_home
            config_path = get_hermes_home() / "config.yaml"
            if not config_path.exists():
                logger.warning("[%s] Config file not found at %s, cannot persist thread_id", self.name, config_path)
                return
            import yaml as _yaml
            with open(config_path, "r", encoding="utf-8") as f:
                config = _yaml.safe_load(f) or {}
            # platforms.telegram.extra.dm_topics — create the path for topics a named
            # delivery target asks for that were not predeclared in config.yaml.
            platforms = config.setdefault("platforms", {})
            telegram_config = platforms.setdefault("telegram", {})
            extra = telegram_config.setdefault("extra", {})
            dm_topics = extra.setdefault("dm_topics", [])
            changed = False
            matching_chat_entry = None
            for chat_entry in dm_topics:
                try:
                    chat_matches = int(chat_entry.get("chat_id", 0)) == int(chat_id)
                except (TypeError, ValueError):
                    chat_matches = False
                if not chat_matches:
                    continue
                matching_chat_entry = chat_entry
                for t in chat_entry.setdefault("topics", []):
                    if t.get("name") == topic_name:
                        if (replace_existing or not t.get("thread_id")) and t.get("thread_id") != thread_id:
                            t["thread_id"] = thread_id
                            changed = True
                        break
                else:
                    chat_entry.setdefault("topics", []).append({"name": topic_name, "thread_id": thread_id})
                    changed = True
                break
            if matching_chat_entry is None:
                dm_topics.append(
                    {"chat_id": chat_id, "topics": [{"name": topic_name, "thread_id": thread_id}]}
                )
                changed = True
            if changed:
                from hermes_cli.config import atomic_config_write
                atomic_config_write(config_path, config, default_flow_style=False, sort_keys=False)
                logger.info(
                    "[%s] Persisted thread_id=%s for topic '%s' in config.yaml",
                    self.name, thread_id, topic_name,
                )
        except Exception as e:
            logger.warning("[%s] Failed to persist thread_id to config: %s", self.name, e, exc_info=True)

    async def _setup_dm_topics(self) -> None:
        """Load or create configured DM topics for specified chats.

        ``config.extra['dm_topics']`` is a list of ``{"chat_id": int, "topics": [{"name",
        "icon_color", "thread_id"?, "skill"?}, ...]}``. Topics with a persisted thread_id
        are cached without an API call; the rest are created and saved back to config.yaml.
        """
        if not self._dm_topics_config:
            return
        for chat_entry in self._dm_topics_config:
            chat_id = chat_entry.get("chat_id")
            topics = chat_entry.get("topics", [])
            if not chat_id or not topics:
                continue
            logger.info("[%s] Setting up %d DM topic(s) for chat %s", self.name, len(topics), chat_id)
            for topic_conf in topics:
                topic_name = topic_conf.get("name")
                if not topic_name:
                    continue
                cache_key = f"{chat_id}:{topic_name}"
                existing_thread_id = topic_conf.get("thread_id")
                if existing_thread_id:
                    self._dm_topics[cache_key] = int(existing_thread_id)
                    logger.info(
                        "[%s] DM topic loaded from config: %s -> thread_id=%s",
                        self.name, cache_key, existing_thread_id,
                    )
                    continue
                icon_color = topic_conf.get("icon_color")
                icon_emoji = topic_conf.get("icon_custom_emoji_id")
                thread_id = await self._create_dm_topic(
                    chat_id=normalize_telegram_chat_id(chat_id), name=topic_name,
                    icon_color=icon_color, icon_custom_emoji_id=icon_emoji,
                )
                if thread_id:
                    self._dm_topics[cache_key] = thread_id
                    logger.info("[%s] DM topic cached: %s -> thread_id=%s", self.name, cache_key, thread_id)
                    self._persist_dm_topic_thread_id(int(chat_id), topic_name, thread_id)
                    # Seed message: Telegram's client hides empty topics until they contain one.
                    try:
                        await self._bot.send_message(
                            chat_id=normalize_telegram_chat_id(chat_id), message_thread_id=thread_id,
                            text=f"\U0001f4cc {topic_name}",
                        )
                    except Exception as seed_err:
                        logger.debug(
                            "[%s] Could not send seed message to topic '%s': %s",
                            self.name, topic_name, seed_err,
                        )

    async def _bot_identity_refresh_loop(self) -> None:
        """Keep the cached @username fresh when no heartbeat is running.

        Webhook mode never calls ``get_me()`` again after ``initialize()`` (polling mode
        does via the heartbeat), so a BotFather rename would break mention routing until restart.
        """
        while True:
            try:
                await asyncio.sleep(self._BOT_IDENTITY_TTL_SECONDS)
                if self._teardown_started:
                    return
                if self.has_fatal_error:
                    return
                await self._refresh_bot_identity(force=True)
            except asyncio.CancelledError:
                return
            except Exception:
                logger.debug(
                    "[%s] Telegram identity refresh loop iteration failed", self.name, exc_info=True
                )

    def _start_post_connect_housekeeping(self) -> None:
        """Kick off deferred post-connect housekeeping; idempotent while a task is still running."""
        task = self._post_connect_task
        if task and not task.done():
            return
        self._post_connect_task = asyncio.ensure_future(self._run_post_connect_housekeeping())

    async def _run_post_connect_housekeeping(self) -> None:
        """Register the command menu, status indicator and DM topics off the connect path.

        A slow Bot API call must not blow the gateway connect timeout. Every step is non-fatal.
        """
        try:
            # Command menu derives from the central COMMAND_REGISTRY.
            try:
                from telegram import (
                    BotCommand, BotCommandScopeAllPrivateChats, BotCommandScopeAllGroupChats,
                    BotCommandScopeDefault,
                )
                from hermes_cli.commands import telegram_menu_commands, telegram_menu_max_commands
                if not self._bot:
                    return
                # Telegram allows 100 commands but has an undocumented ~4KB payload limit;
                # the default cap of 60 stays under it (tunable via extra.command_menu).
                max_commands = telegram_menu_max_commands()
                menu_commands, hidden_count = telegram_menu_commands(max_commands=max_commands)
                bot_commands = [BotCommand(name, desc) for name, desc in menu_commands]
                # Register every scope: Telegram picks the narrowest matching one per chat type.
                for scope_cls in (BotCommandScopeDefault, BotCommandScopeAllPrivateChats, BotCommandScopeAllGroupChats):
                    scope_name = getattr(scope_cls, "__name__", str(scope_cls))
                    try:
                        await self._bot.set_my_commands(bot_commands, scope=scope_cls())
                        logger.info("[%s] set_my_commands OK for scope %s (%d cmds)", self.name, scope_name, len(bot_commands))
                    except Exception as scope_err:
                        logger.warning("[%s] set_my_commands FAILED for scope %s: %s", self.name, scope_name, scope_err)
                # Forum topics don't inherit AllGroupChats; _ensure_forum_commands registers
                # BotCommandScopeChat(chat_id) lazily on the first forum-topic message.
                if hidden_count:
                    logger.info(
                        "[%s] Telegram menu: %d commands registered, %d hidden (over %d limit). Use /commands for full list.",
                        self.name, len(menu_commands), hidden_count, max_commands,
                    )
            except Exception as e:
                logger.warning(
                    "[%s] Could not register Telegram command menu: %s",
                    self.name, _redact_telegram_error_text(e), exc_info=True,
                )
            # "Online" in the bot's short description (opt-in via extra.status_indicator).
            try:
                await self._set_status_indicator(online=True)
            except Exception:
                pass
            # DM topics (Bot API 9.4 Private Chat Topics); the bot works fine without them.
            try:
                await self._setup_dm_topics()
            except Exception as topics_err:
                logger.warning(
                    "[%s] DM topics setup failed (non-fatal): %s", self.name, topics_err, exc_info=True
                )
        except asyncio.CancelledError:
            raise
        finally:
            if self._post_connect_task is asyncio.current_task():
                self._post_connect_task = None

    async def _on_platform_update(self, update, context) -> None:
        """Catch-all PTB handler firing ``gateway_platform_event`` per inbound update.

        Normalizes the update into a stable envelope (no raw SDK objects) and forwards
        it with an internal source to the gateway-owned post-auth boundary. Registered
        in a dedicated high group so it observes alongside, never displaces, the core
        handlers. Malformed updates and dispatch errors never raise into PTB's loop.
        """
        handler: Optional[Callable[[Dict[str, Any], Any], Awaitable[None]]] = getattr(
            self, "_platform_event_handler", None
        )
        if handler is None:
            return
        try:
            from hermes_cli.lifecycle import has_hook
            if not has_hook("gateway_platform_event"):
                return
            event = self._normalize_platform_event(update)
        except Exception:
            logger.debug("[%s] gateway_platform_event normalize error", self.name, exc_info=True)
            return
        if event is None:
            return
        # The gateway-owned boundary runs the full profile-scoped authorization chain
        # before plugin dispatch. No callback = no trusted auth boundary ⇒ fail closed.
        try:
            source = self._source_for_platform_event_auth(update)
            await handler(event, source)
        except Exception:
            logger.debug("[%s] gateway_platform_event dispatch error", self.name, exc_info=True)
            return

    def _source_for_platform_event_auth(self, update):
        """Route a supported update to its event-specific auth-source extractor.

        A reaction carries the reactor, an edit the editor. Raises ``ValueError`` for
        updates without a wired extractor so the boundary fails closed.
        """
        if getattr(update, "message_reaction", None) is not None:
            return self._source_from_reaction_for_auth(update)
        edited = getattr(update, "edited_message", None)
        if edited is not None:
            source = self._source_from_message_for_auth(edited)
            # Tolerates missing identities for pairing-flow callers; this boundary must not.
            if not source.user_id or not source.chat_id:
                raise ValueError(
                    "gateway_platform_event message_edited requires editor and chat identities"
                )
            return source
        raise ValueError(
            "gateway_platform_event source extraction has no extractor for this update type"
        )

    def _normalize_platform_event(self, update) -> Optional[Dict[str, Any]]:
        """Map a PTB update to a ``{platform, event_type, payload}`` envelope, or ``None``.

        Each event type has its own additive payload contract (hooks.md); raw SDK objects
        never leave this boundary. Types without a contract (forward, chat-member) → ``None``.
        """
        if getattr(update, "message_reaction", None) is not None:
            return self._normalize_reaction_event(update)
        if getattr(update, "edited_message", None) is not None:
            return self._normalize_message_edited_event(update)
        return None

    def _normalize_reaction_event(self, update) -> Optional[Dict[str, Any]]:
        """Normalize a ``message_reaction`` update (event_type ``reaction``).

        Payload: ``emojis`` (unicode), ``custom_emoji_ids`` (PTB exposes ``custom_emoji_id``
        with no ``.emoji``), ``chat_id``, ``message_id``, ``thread_id``.
        """
        mr = getattr(update, "message_reaction", None)
        if mr is None:
            return None
        chat = getattr(mr, "chat", None)
        new_reaction = getattr(mr, "new_reaction", None) or []
        if not isinstance(new_reaction, (list, tuple)):
            return None
        chat_id = getattr(chat, "id", None) if chat is not None else None
        message_id = getattr(mr, "message_id", None)
        if (
            isinstance(chat_id, bool) or not isinstance(chat_id, (str, int))
            or isinstance(message_id, bool) or not isinstance(message_id, (str, int))
        ):
            return None
        emojis: List[str] = []
        custom_emoji_ids: List[str] = []
        for r in new_reaction[:64]:
            emoji = getattr(r, "emoji", None)
            if isinstance(emoji, str) and emoji:
                emojis.append(emoji[:64])
            custom_id = getattr(r, "custom_emoji_id", None)
            if not isinstance(custom_id, bool) and isinstance(custom_id, (str, int)):
                custom_emoji_ids.append(str(custom_id)[:128])
        return {
            "platform": "telegram",
            "event_type": "reaction",
            "payload": {
                "emojis": emojis,
                "custom_emoji_ids": custom_emoji_ids,
                "chat_id": str(chat_id)[:128],
                "message_id": str(message_id)[:128],
                # Reactions carry no thread_id; don't guess or expose an adapter object.
                "thread_id": None,
            },
        }
    def _normalize_message_edited_event(self, update) -> Optional[Dict[str, Any]]:
        """Normalize an ``edited_message`` update into a ``message_edited`` event.

        Payload (v1, additive): chat_id, message_id, thread_id (forum topic), text
        (edited text or caption, bounded), edited_at (ISO 8601 UTC or None). No raw
        PTB ``Message`` leaves this boundary; malformed identities return None.
        """
        message = getattr(update, "edited_message", None)
        if message is None:
            return None
        chat = getattr(message, "chat", None)
        chat_id = getattr(chat, "id", None) if chat is not None else None
        message_id = getattr(message, "message_id", None)
        if (
            isinstance(chat_id, bool)
            or not isinstance(chat_id, (str, int))
            or isinstance(message_id, bool)
            or not isinstance(message_id, (str, int))
        ):
            return None
        text = getattr(message, "text", None) or getattr(message, "caption", None)
        if not isinstance(text, str):
            text = None
        thread_id = None
        thread_id_raw = getattr(message, "message_thread_id", None)
        if (
            not isinstance(thread_id_raw, bool)
            and isinstance(thread_id_raw, (str, int))
            and bool(getattr(message, "is_topic_message", False))
        ):
            thread_id = str(thread_id_raw)[:128]
        edited_at = None
        edit_date = getattr(message, "edit_date", None)
        try:
            if edit_date is not None and hasattr(edit_date, "isoformat"):
                edited_at = str(edit_date.isoformat())[:64]
        except Exception:
            edited_at = None
        return {
            "platform": "telegram",
            "event_type": "message_edited",
            "payload": {
                "chat_id": str(chat_id)[:128],
                "message_id": str(message_id)[:128],
                "thread_id": thread_id,
                "text": text[:8192] if text is not None else None,
                "edited_at": edited_at,
            },
        }

    def _register_handlers(self, app) -> None:
        """Register every PTB handler on ``app``.

        Single source of truth: initial connect and the transient-init rebuild both
        call this, keeping the group-99 observer in lockstep with the core handlers.
        """
        app.add_handler(TelegramMessageHandler(
            filters.TEXT & ~filters.COMMAND, self._handle_text_message
        ))
        app.add_handler(TelegramMessageHandler(filters.COMMAND, self._handle_command))
        app.add_handler(TelegramMessageHandler(
            filters.LOCATION | getattr(filters, "VENUE", filters.LOCATION),
            self._handle_location_message
        ))
        app.add_handler(TelegramMessageHandler(
            filters.PHOTO | filters.VIDEO | filters.AUDIO | filters.VOICE | filters.Document.ALL | filters.Sticker.ALL,
            self._handle_media_message
        ))
        # Inline keyboard button callbacks (update prompts)
        app.add_handler(CallbackQueryHandler(self._handle_callback_query))
        # Inline command picker (@botname <query>). Inert until the owner enables
        # inline mode via BotFather /setinline, so registering unconditionally is safe.
        app.add_handler(InlineQueryHandler(self._handle_inline_query))
        # gateway_platform_event observer (see _on_platform_update); group 99 so it
        # observes alongside, never displaces, the core handlers.
        app.add_handler(TypeHandler(Update, self._on_platform_update), group=99)

    async def _build_ptb_requests(self) -> tuple:
        """Build the (general, getUpdates) HTTPXRequest pair for the PTB app.

        Picks the fallback-IP transport, an explicit proxy, or direct DNS, and
        instruments the getUpdates request for polling-progress tracking.
        """
        # PTB's pool_timeout=1s default trips "Pool timeout: All connections in the
        # connection pool are occupied" on flaky networks; use safer defaults + env overrides.
        request_kwargs = {
            "connection_pool_size": env_int("HERMES_TELEGRAM_HTTP_POOL_SIZE", 512),
            "pool_timeout": env_float("HERMES_TELEGRAM_HTTP_POOL_TIMEOUT", 8.0),
            "connect_timeout": env_float("HERMES_TELEGRAM_HTTP_CONNECT_TIMEOUT", 10.0),
            "read_timeout": env_float("HERMES_TELEGRAM_HTTP_READ_TIMEOUT", 20.0),
            "write_timeout": env_float("HERMES_TELEGRAM_HTTP_WRITE_TIMEOUT", 20.0),
            # Not a duplicate of write_timeout: PTB routes requests carrying files to
            # media_write_timeout. httpx budgets it per socket write (stall tolerance,
            # not a size/bandwidth cap); 60s rides out congested-link buffer stalls.
            "media_write_timeout": 60.0,
        }

        # CLOSE_WAIT fd leak: PTB builds its httpx.AsyncClient with no keepalive
        # tuning, so httpx's default keepalive_expiry=5.0 applies. Behind a proxy a
        # peer FIN can sit in CLOSE_WAIT longer, leaking fds in the general pool that
        # _drain_polling_connections never resets. Inject platform_httpx_limits()
        # while preserving PTB's max_connections; httpx_kwargs is spread last into
        # PTB's client kwargs, so `limits` here wins.
        from gateway.platforms._http_client_limits import platform_httpx_limits

        _base_limits = platform_httpx_limits()
        if _base_limits is not None:
            import httpx as _httpx

            _pool_limits = _httpx.Limits(
                max_connections=request_kwargs["connection_pool_size"],
                max_keepalive_connections=_base_limits.max_keepalive_connections,
                keepalive_expiry=_base_limits.keepalive_expiry,
            )
            # A long-poll request is continuously active, so keepalive expiry can't
            # protect it from a server-side close. Never hand getUpdates a pooled
            # socket from a previous poll; ordinary requests keep the reusable pool.
            _updates_limits = _httpx.Limits(
                max_connections=request_kwargs["connection_pool_size"],
                max_keepalive_connections=0,
                keepalive_expiry=_base_limits.keepalive_expiry,
            )
        else:  # pragma: no cover — httpx always present alongside PTB
            _pool_limits = None
            _updates_limits = None

        def _with_limits(httpx_kwargs: Optional[dict] = None) -> dict:
            """Merge tuned keepalive limits into httpx client kwargs (proxy/direct
            branches only; a caller-supplied ``limits`` wins). The fallback-IP branch
            must NOT use this — see the ``_transport_kwargs`` note below."""
            kwargs = dict(httpx_kwargs or {})
            if _pool_limits is not None and "limits" not in kwargs:
                kwargs["limits"] = _pool_limits
            return kwargs

        disable_fallback = (
            os.getenv("HERMES_TELEGRAM_DISABLE_FALLBACK_IPS", "").strip().lower()
            in {"1", "true", "yes", "on"}
        )
        fallback_ips = self._fallback_ips()
        if disable_fallback:
            fallback_ips = []
        if not fallback_ips and not disable_fallback:
            discovery_timeout = self._env_float_clamped(
                "HERMES_TELEGRAM_FALLBACK_DISCOVERY_TIMEOUT", 5.0, min_value=0.0
            )
            logger.warning(
                "[%s] Discovering Telegram API fallback IPs via DNS-over-HTTPS…", self.name
            )
            try:
                fallback_ips = await _await_with_thread_deadline(
                    discover_fallback_ips(), timeout=discovery_timeout
                )
            except Exception as exc:
                logger.warning(
                    "[%s] Telegram fallback-IP discovery failed after %.0fs; "
                    "using seed IPv4 Telegram API IPs so a blackholed IPv6 "
                    "hostname path cannot hang initialize() (#87015): %s",
                    self.name, discovery_timeout, _redact_telegram_error_text(exc),
                )
                fallback_ips = list(SEED_FALLBACK_IPS)
            else:
                logger.info(
                    "[%s] Auto-discovered Telegram fallback IPs: %s",
                    self.name, ", ".join(fallback_ips),
                )

        proxy_targets = ["api.telegram.org", *fallback_ips]
        proxy_url = resolve_proxy_url("TELEGRAM_PROXY", target_hosts=proxy_targets)
        if fallback_ips and not proxy_url and not disable_fallback:
            logger.info("[%s] Telegram fallback IPs active: %s", self.name, ", ".join(fallback_ips))
            # Separate request/update pools reduce contention during polling
            # reconnect + bootstrap/delete_webhook calls. httpx ignores the
            # client-level `limits` kwarg when a custom `transport` is supplied, so
            # this branch MUST pass the tuned limits straight into
            # TelegramFallbackTransport — never via `_with_limits`.
            _transport_kwargs: dict = {}
            if _pool_limits is not None:
                _transport_kwargs["limits"] = _pool_limits
            _transport_kwargs["socket_options"] = tcp_keepalive_socket_options()
            _updates_transport_kwargs = dict(_transport_kwargs)
            if _updates_limits is not None:
                _updates_transport_kwargs["limits"] = _updates_limits
            request = HTTPXRequest(
                **request_kwargs,
                httpx_kwargs={
                    "transport": TelegramFallbackTransport(fallback_ips, **_transport_kwargs)
                },
            )
            get_updates_request = HTTPXRequest(
                **request_kwargs,
                httpx_kwargs={
                    "transport": TelegramFallbackTransport(
                        fallback_ips, **_updates_transport_kwargs
                    )
                },
            )
        elif proxy_url:
            logger.info("[%s] Proxy detected; passing explicitly to HTTPXRequest: %s", self.name, proxy_url)
            request = HTTPXRequest(**request_kwargs, proxy=proxy_url, httpx_kwargs=_with_limits())
            get_updates_request = HTTPXRequest(
                **request_kwargs, proxy=proxy_url, httpx_kwargs={"limits": _updates_limits}
            )
        else:
            if disable_fallback:
                logger.info("[%s] Telegram fallback-IP transport disabled via env", self.name)
            request = HTTPXRequest(**request_kwargs, httpx_kwargs=_with_limits())
            get_updates_request = HTTPXRequest(
                **request_kwargs, httpx_kwargs={"limits": _updates_limits}
            )

        get_updates_request = self._instrument_polling_request(get_updates_request)
        return request, get_updates_request

    async def _initialize_app_with_retries(self, builder) -> None:
        """Run ``app.initialize()`` with a bounded retry ladder for transient errors.

        Rebuilds ``self._app``/``self._bot`` from ``builder`` after each failed
        attempt; raises OSError when the per-attempt or total watchdog expires.
        """
        # Each attempt is capped by _init_timeout so one unreachable fallback-IP
        # chain can't block startup indefinitely.
        _max_connect = 8
        _init_timeout = env_float("HERMES_TELEGRAM_INIT_TIMEOUT", 30.0)
        # Total watchdog: upper bound on the whole connect loop even if the retry
        # loop itself silently stalls (per-attempt timeout plus margin for sleeps).
        _total_deadline = (
            asyncio.get_running_loop().time()
            + _init_timeout * _max_connect
            + 120.0  # extra margin for between-attempt sleeps + overhead
        )
        for _attempt in range(_max_connect):
            rebuild_app = False
            try:
                # Total watchdog: the ladder must yield even if no attempt raised.
                if asyncio.get_running_loop().time() >= _total_deadline:
                    raise OSError(
                        f"Telegram initialization timed out after {_max_connect} attempts "
                        f"({_init_timeout:.0f}s each) — total connect watchdog "
                        f"deadline ({_init_timeout * _max_connect + 120.0:.0f}s) exceeded. "
                        f"Check network connectivity to api.telegram.org "
                        f"or set HERMES_TELEGRAM_HTTP_CONNECT_TIMEOUT / "
                        f"HERMES_TELEGRAM_INIT_TIMEOUT to a lower value."
                    )
                logger.warning(
                    "[%s] Connecting to Telegram (attempt %d/%d)…",
                    self.name, _attempt + 1, _max_connect,
                )
                await _await_with_thread_deadline(
                    self._app.initialize(),
                    timeout=_init_timeout,
                    # On timeout the initialize() task is abandoned (it may be wedged
                    # in a shielded scope); best-effort release the half-built app's
                    # httpx client so it isn't leaked across the retry ladder.
                    on_abandon=lambda app=self._app: _shutdown_abandoned_app(app),
                )
                break
            except asyncio.TimeoutError:
                rebuild_app = True
                if _attempt < _max_connect - 1:
                    wait = min(2 ** _attempt, 15)
                    logger.warning(
                        "[%s] Connect attempt %d/%d timed out after %.0fs — retrying in %ds",
                        self.name, _attempt + 1, _max_connect, _init_timeout, wait,
                    )
                    await asyncio.sleep(wait)
                else:
                    raise OSError(
                        f"Telegram initialization timed out after {_max_connect} attempts "
                        f"({_init_timeout:.0f}s each). Check network connectivity to api.telegram.org "
                        f"or set HERMES_TELEGRAM_HTTP_CONNECT_TIMEOUT to a lower value."
                    )
            except Exception as init_err:
                # OSError always retries; anything else only when it looks like a network error.
                rebuild_app = True
                if not isinstance(init_err, OSError) and not self._looks_like_network_error(init_err):
                    raise
                if _attempt < _max_connect - 1:
                    wait = min(2 ** _attempt, 15)
                    logger.warning(
                        "[%s] Connect attempt %d/%d failed: %s — retrying in %ds",
                        self.name, _attempt + 1, _max_connect, init_err, wait,
                    )
                    await asyncio.sleep(wait)
                else:
                    raise
            except BaseException:
                # CancelledError etc.: log for the operator, then reraise to preserve
                # cancellation semantics. Placed LAST so the Exception handlers win.
                logger.warning(
                    "[%s] Connect attempt %d/%d interrupted by %s — propagating",
                    self.name, _attempt + 1, _max_connect,
                    "CancelledError"
                    if isinstance(sys.exc_info()[1], asyncio.CancelledError)
                    else type(sys.exc_info()[1]).__name__,
                )
                raise
            finally:
                # A failed attempt may leave the app half-initialized (closed
                # transports, half-built handlers): rebuild a fresh Application from
                # the same builder for the next attempt and discard the old one.
                if rebuild_app and _attempt < _max_connect - 1:
                    old_app = self._app
                    self._app = builder.build()
                    self._bot = self._app.bot
                    # Keep core and observer handlers in lockstep after a rebuild.
                    self._register_handlers(self._app)
                    try:
                        await _shutdown_abandoned_app(old_app)
                    except Exception:
                        pass

    async def _start_webhook_mode(self, webhook_url: str, *, is_reconnect: bool) -> None:
        """Start PTB's webhook server (Telegram pushes updates to us).

        Lets cloud platforms (Fly.io, Railway) auto-wake suspended machines on
        inbound HTTP. SECURITY: TELEGRAM_WEBHOOK_SECRET is REQUIRED — without it PTB
        passes secret_token=None and the endpoint accepts forged updates from anyone
        (GHSA-3vpc-7q5r-276h). Refuse to start rather than run fail-open.
        """
        webhook_port = env_int("TELEGRAM_WEBHOOK_PORT", 8443)
        # Bind host. Default "" → tornado opens one listening socket per address
        # family (IPv4 + IPv6); "0.0.0.0" would be unreachable over IPv6-only
        # private networks (e.g. Fly.io 6PN).
        webhook_host = (
            os.getenv("TELEGRAM_WEBHOOK_HOST", "").strip()
            or str((self.config.extra or {}).get("webhook_host") or "").strip()
        )
        # Profile-scoped read: honors the profile's own secret; only an UNSCOPED
        # read under multiplex (default-profile startup) falls back to process env.
        from agent.secret_scope import UnscopedSecretError, get_secret

        try:
            webhook_secret = (get_secret("TELEGRAM_WEBHOOK_SECRET") or "").strip()
        except UnscopedSecretError:
            webhook_secret = os.getenv("TELEGRAM_WEBHOOK_SECRET", "").strip()
        if not webhook_secret:
            raise RuntimeError(
                "TELEGRAM_WEBHOOK_SECRET is required when "
                "TELEGRAM_WEBHOOK_URL is set. Without it, the "
                "webhook endpoint accepts forged updates from "
                "anyone who can reach it — see "
                "https://github.com/NousResearch/hermes-agent/"
                "security/advisories/GHSA-3vpc-7q5r-276h.\n\n"
                "Generate a secret and set it in your .env:\n"
                "  export TELEGRAM_WEBHOOK_SECRET=\"$(openssl rand -hex 32)\"\n\n"
                "Then register it with Telegram when setting the "
                "webhook via setWebhook's secret_token parameter."
            )
        from urllib.parse import urlparse
        webhook_path = urlparse(webhook_url).path or "/telegram"

        await self._app.updater.start_webhook(
            listen=webhook_host,
            port=webhook_port,
            url_path=webhook_path,
            webhook_url=webhook_url,
            secret_token=webhook_secret,
            allowed_updates=Update.ALL_TYPES,
            # Webhooks are push-based (no server-side getUpdates queue), so this is
            # a no-op in practice; mirrors the polling path's reconnect semantics.
            drop_pending_updates=not is_reconnect,
        )
        self._webhook_mode = True
        self._polling_progress_accepting = False
        self._send_path_degraded = False
        logger.info(
            "[%s] Webhook server listening on %s:%d%s",
            self.name, webhook_host or "* (all interfaces, IPv4+IPv6)", webhook_port, webhook_path,
        )

    async def _start_polling_mode(self, *, is_reconnect: bool) -> None:
        """Clear any stale webhook and start resilient long polling."""
        # Clear any stale webhook so polling doesn't inherit it and silently stop
        # receiving updates. Best-effort: a transient Bot API error must not fail
        # gateway startup — degrade to background polling recovery instead.
        await self._delete_webhook_best_effort(require_success=not is_reconnect)

        loop = asyncio.get_running_loop()

        def _polling_error_callback(error: Exception) -> None:
            if self._teardown_started:
                return
            if self._polling_error_task and not self._polling_error_task.done():
                return
            if self._looks_like_polling_conflict(error):
                # Stop PTB's network_retry_loop synchronously BEFORE scheduling the
                # async recovery task: PTB calls this callback inside its loop and
                # keeps polling, so PTB's retry and our stop->restart would overlap
                # and produce a fresh 409. Disarming now lets recovery own polling.
                self._disarm_ptb_retry_loop()
                self._spawn_polling_recovery(loop, self._handle_polling_conflict(error))
            elif self._looks_like_network_error(error):
                logger.warning("[%s] Telegram network _redact_telegram_error_text(error), scheduling reconnect: %s", self.name, error)
                self._spawn_polling_recovery(loop, self._handle_polling_network_error(error))
            else:
                logger.error("[%s] Telegram polling _redact_telegram_error_text(error): %s", self.name, error, exc_info=True)

        # Store reference for retry use in _handle_polling_conflict
        self._polling_error_callback_ref = _polling_error_callback

        polling_started = await self._start_polling_resilient(
            # Cold first boot drops the stale Bot API queue; a watcher reconnect
            # preserves it so messages sent while offline are delivered.
            drop_pending_updates=not is_reconnect,
            error_callback=_polling_error_callback,
            require_progress=not is_reconnect,
        )
        if not polling_started:
            logger.warning(
                "[%s] Connected in degraded Telegram mode: gateway is alive, "
                "polling will be retried in the background",
                self.name,
            )

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        """Connect to Telegram via long polling, or a webhook server if
        ``TELEGRAM_WEBHOOK_URL`` is set (cloud deployments where inbound HTTP wakes
        a suspended machine).

        ``is_reconnect``: False = cold boot (drop the stale Bot API queue); True =
        watcher reconnect after an outage (preserve queued updates, otherwise every
        message sent during the outage is silently lost — matching the network-error
        ladder and 409 handler, which already pass ``drop_pending_updates=False``).

        Webhook env: TELEGRAM_WEBHOOK_URL (public HTTPS URL), TELEGRAM_WEBHOOK_PORT
        (default 8443), TELEGRAM_WEBHOOK_HOST (default: dual-stack all interfaces),
        TELEGRAM_WEBHOOK_SECRET (update verification token).
        """
        # Explicit connect() is the only operation allowed to reopen polling after a
        # completed, serialized teardown. Background recovery never clears this fence.
        self._polling_teardown_started = False
        # Mode is re-evaluated on every explicit connection.
        self._webhook_mode = False

        if not TELEGRAM_AVAILABLE:
            logger.error(
                "[%s] python-telegram-bot not installed. Run: pip install python-telegram-bot",
                self.name,
            )
            self._set_fatal_error("missing_dependency", "python-telegram-bot not installed", retryable=False)
            return False
        if not self.config.token:
            logger.error("[%s] No bot token configured", self.name)
            self._set_fatal_error("missing_credentials", "No bot token configured", retryable=False)
            return False
        try:
            if not self._acquire_platform_lock('telegram-bot-token', self.config.token, 'Telegram bot token'):
                return False

            builder = Application.builder().token(self.config.token)
            custom_base_url = self.config.extra.get("base_url")
            if custom_base_url:
                builder = builder.base_url(custom_base_url)
                builder = builder.base_file_url(
                    self.config.extra.get("base_file_url", custom_base_url)
                )
                logger.info("[%s] Using custom Telegram base_url: %s", self.name, custom_base_url)
            # Local-mode telegram-bot-api returns absolute server-side file paths;
            # PTB needs local_mode=True so download_*() reads from disk instead of
            # issuing an HTTP GET that would 404 (path must be readable by Hermes).
            if self.config.extra.get("local_mode"):
                builder = builder.local_mode(True)
                logger.info("[%s] Using Telegram local_mode (read files from disk)", self.name)

            request, get_updates_request = await self._build_ptb_requests()
            builder = builder.request(request).get_updates_request(get_updates_request)
            self._app = builder.build()
            self._bot = self._app.bot

            # Plugin PTB handlers go BEFORE the core handlers: PTB dispatches the
            # first matching handler per group, so pattern-scoped plugin handlers win
            # for their own updates and everything else falls through to core.
            self._wire_plugin_handlers(self._app)
            self._register_handlers(self._app)

            await self._initialize_app_with_retries(builder)
            await self._app.start()

            webhook_url = os.getenv("TELEGRAM_WEBHOOK_URL", "").strip()
            if webhook_url:
                await self._start_webhook_mode(webhook_url, is_reconnect=is_reconnect)
            else:
                await self._start_polling_mode(is_reconnect=is_reconnect)

            self._mark_connected()
            mode = "webhook" if self._webhook_mode else "polling"
            # WARNING, not INFO: the "Connecting…" line above is WARNING and reaches
            # the terminal (default stderr handler is WARNING-only); an INFO success
            # line made healthy startups look stalled at "attempt 1/8". A real hang
            # must be the *absence* of this line, not ambiguity.
            logger.warning("[%s] Connected to Telegram (%s mode)", self.name, mode)

            # Heartbeat loop only in polling mode: webhook mode has no long-poll
            # socket to wedge in CLOSE-WAIT.
            if not self._webhook_mode:
                if self._polling_heartbeat_task and not self._polling_heartbeat_task.done():
                    self._polling_heartbeat_task.cancel()
                self._polling_heartbeat_task = asyncio.ensure_future(self._polling_heartbeat_loop())

            # Seed the live identity from PTB's initialize() cache, then keep it
            # fresh: polling rides the heartbeat's get_me() probe; webhook mode has no
            # probe, so it gets a low-frequency refresh loop — otherwise a BotFather
            # rename breaks mention routing until restart.
            self._note_bot_username(getattr(self._bot, "username", None))
            self._bot_identity_checked_at = time.monotonic()
            if self._webhook_mode:
                identity_task = getattr(self, "_bot_identity_refresh_task", None)
                if identity_task and not identity_task.done():
                    identity_task.cancel()
                self._bot_identity_refresh_task = asyncio.ensure_future(
                    self._bot_identity_refresh_loop()
                )

            # Command-menu registration, DM-topic setup and the status indicator make
            # Bot API calls that can stall for some tokens; inside connect() (which
            # the gateway wraps in a timeout) one slow call would sink the whole
            # connect even though transport is live. Defer to a cancellable task.
            self._start_post_connect_housekeeping()

            return True
        except Exception as e:
            self._release_platform_lock()
            safe_error = _redact_telegram_error_text(e)
            # Classify by exception TYPE (never message text): auth failures
            # (InvalidToken / Forbidden) can never self-heal, so marking them
            # retryable put agents into a silent eternal reconnect loop. Reuse the
            # runtime polling path's discriminator so both agree on what's transient.
            if self._looks_like_auth_error(e):
                message = (
                    f"Telegram bot token rejected: {safe_error}. "
                    "The token is invalid or was revoked — generate a new one "
                    "with @BotFather and update TELEGRAM_BOT_TOKEN."
                )
                self._set_fatal_error("telegram_auth_error", message, retryable=False)
            else:
                message = f"Telegram startup failed: {safe_error}"
                self._set_fatal_error("telegram_connect_error", message, retryable=True)
            logger.error("[%s] Failed to connect to Telegram: %s", self.name, safe_error)
            return False

    async def _set_status_indicator(self, online: bool) -> None:
        """Set the bot's short description (profile line) to the online/offline text.

        Closest Bot API surface to presence — bots have no real online dot. No-op
        unless ``extra.status_indicator`` is enabled; best-effort, failures are
        logged at debug so they never block connect/disconnect.
        """
        if not getattr(self, "_status_indicator_enabled", False):
            return
        bot = self._bot
        if bot is None:
            return
        text = self._status_online_text if online else self._status_offline_text
        # Telegram caps short_description at 120 chars.
        text = text[:120]
        try:
            await bot.set_my_short_description(short_description=text)
            logger.info("[%s] Set bot status indicator to %r", self.name, text)
        except Exception as e:
            logger.debug(
                "[%s] Failed to set bot status indicator to %r: %s",
                self.name, text, _redact_telegram_error_text(e),
            )

    async def _cancel_pending_delivery_tasks(self) -> None:
        """Cancel every delayed-delivery task family before disconnect completes.

        Media-group, photo-batch, text-batch flush tasks plus the polling-error
        recovery task all sit behind ``asyncio.sleep()``; left running they'd
        dispatch ``handle_message`` into a torn-down session. Skips the current
        task so the teardown coroutine doesn't cancel itself.
        """
        current_task = asyncio.current_task()
        pending_tasks: list[asyncio.Task] = []
        awaitable_tasks: list[asyncio.Task] = []
        seen: set[int] = set()

        def collect(task: Optional[asyncio.Task]) -> None:
            if not task or task.done() or task is current_task:
                return
            marker = id(task)
            if marker in seen:
                return
            seen.add(marker)
            pending_tasks.append(task)
            if asyncio.isfuture(task) or asyncio.iscoroutine(task):
                awaitable_tasks.append(task)

        for task in list(self._media_group_tasks.values()):
            collect(task)
        for task in list(self._pending_photo_batch_tasks.values()):
            collect(task)
        for task in list(self._pending_text_batch_tasks.values()):
            collect(task)
        collect(getattr(self, "_polling_error_task", None))
        collect(getattr(self, "_polling_progress_verifier_task", None))
        # Hold-queue redispatch must be cancellable+awaitable on teardown so it
        # cannot dispatch handle_message into a torn-down session.
        collect(getattr(self, "_held_inbound_redispatch_task", None))

        for task in pending_tasks:
            task.cancel()
        if awaitable_tasks:
            await asyncio.gather(*awaitable_tasks, return_exceptions=True)

        # Salvage buffered inbound events before clearing maps — unless permanent
        # fatal, where no reconnect can drain and hold would re-orphan them.
        if self._is_permanent_fatal():
            n_pending = (
                len(self._pending_text_batches)
                + len(self._pending_photo_batches)
                + len(self._media_group_events)
            )
            if n_pending:
                logger.warning(
                    "[Telegram] Non-retryable fatal teardown; discarding %d pending inbound batch(es)",
                    n_pending,
                )
        else:
            for event in list(self._pending_text_batches.values()):
                self._hold_inbound_event(event, where="text-batch-teardown")
            for event in list(self._pending_photo_batches.values()):
                self._hold_inbound_event(event, where="photo-batch-teardown")
            for event in list(self._media_group_events.values()):
                self._hold_inbound_event(event, where="media-group-teardown")

        self._media_group_tasks.clear()
        self._media_group_events.clear()
        self._pending_photo_batch_tasks.clear()
        self._pending_photo_batches.clear()
        self._pending_text_batch_tasks.clear()
        self._pending_text_batches.clear()
        if getattr(self, "_polling_error_task", None) is not current_task:
            self._polling_error_task = None
        if getattr(self, "_polling_progress_verifier_task", None) is not current_task:
            self._polling_progress_verifier_task = None
        if getattr(self, "_held_inbound_redispatch_task", None) is not current_task:
            self._held_inbound_redispatch_task = None
    async def _await_disconnect_step(self, awaitable, timeout: float, step: str) -> bool:
        """Await one disconnect step; detach on timeout so teardown advances.

        ``asyncio.wait_for`` cancels an overdue child but then waits for it to exit,
        so PTB close paths that swallow ``CancelledError`` on a half-dead socket
        could wedge disconnect forever. The abandoned task is observed via
        ``_consume_abandoned_task``.
        """
        task = asyncio.ensure_future(awaitable)
        try:
            if timeout <= 0:
                done, _pending = await asyncio.wait({task})
            else:
                done, _pending = await asyncio.wait({task}, timeout=timeout)
        except asyncio.CancelledError:
            # asyncio.wait does NOT cancel its futures when itself cancelled; don't
            # orphan the inner task on outer cancellation.
            task.cancel()
            task.add_done_callback(_consume_abandoned_task)
            raise
        if task in done:
            # Intentional cancels (heartbeat / identity / lifecycle) surface as
            # CancelledError — swallow so disconnect keeps advancing.
            try:
                await task
            except asyncio.CancelledError:
                pass
            return True
        task.cancel()
        task.add_done_callback(_consume_abandoned_task)
        logger.warning(
            "[%s] %s timed out after %.1fs during disconnect; continuing teardown",
            self.name, step, timeout,
        )
        return False

    async def _cancel_task_attr(self, attr: str, label: str) -> None:
        """Cancel + bounded-await the task stored at ``self.<attr>``, then clear it.

        getattr guards the object.__new__ test pattern (attribute may be missing).
        """
        task = getattr(self, attr, None)
        if task and not task.done():
            task.cancel()
            await self._await_disconnect_step(task, _DISCONNECT_STEP_TIMEOUT, label)
        setattr(self, attr, None)

    async def disconnect(self) -> None:
        """Stop polling/webhook, cancel pending delayed deliveries, and disconnect."""
        # Mark disconnected first so the drop guard short-circuits any flush that
        # wins the race against teardown, and late handlers can't schedule tasks.
        self._mark_disconnected()
        self._polling_teardown_started = True
        self._polling_progress_accepting = False
        self._polling_generation = getattr(self, "_polling_generation", 0) + 1
        self._polling_progress_event = asyncio.Event()
        self._send_path_degraded = True

        # Release the bot-token lock immediately so a wedged close cannot block the
        # reconnect watcher. The rest of teardown is best-effort.
        self._release_platform_lock()

        # Cancel and await both polling lifecycle owners right after the fence,
        # before any other teardown await lets them start a new generation.
        current_task = asyncio.current_task()
        lifecycle_tasks: list[asyncio.Task] = []
        lifecycle_seen: set[int] = set()
        for task in (
            getattr(self, "_polling_error_task", None),
            getattr(self, "_polling_progress_verifier_task", None),
        ):
            if not task or task.done() or task is current_task:
                continue
            marker = id(task)
            if marker in lifecycle_seen:
                continue
            lifecycle_seen.add(marker)
            task.cancel()
            if asyncio.isfuture(task) or asyncio.iscoroutine(task):
                lifecycle_tasks.append(task)
        if lifecycle_tasks:
            await self._await_disconnect_step(
                asyncio.gather(*lifecycle_tasks, return_exceptions=True),
                _DISCONNECT_STEP_TIMEOUT, "lifecycle-task cancel",
            )
        if getattr(self, "_polling_error_task", None) is not current_task:
            self._polling_error_task = None
        if getattr(self, "_polling_progress_verifier_task", None) is not current_task:
            self._polling_progress_verifier_task = None

        # Cancellation callbacks may have run while awaited; the fence stays authoritative.
        self._polling_progress_accepting = False
        self._send_path_degraded = True

        # Cancel deferred post-connect housekeeping so it cannot fire into a
        # half-torn-down bot client.
        post_connect_task = getattr(self, "_post_connect_task", None)
        if post_connect_task and not post_connect_task.done():
            post_connect_task.cancel()
            await self._await_disconnect_step(
                asyncio.gather(post_connect_task, return_exceptions=True),
                _DISCONNECT_STEP_TIMEOUT, "post-connect cancel",
            )
        self._post_connect_task = None
        # Cancel the heartbeat before tearing down the app so its probe cannot fire
        # get_me() into a half-shutdown bot client; same fence for the webhook-mode
        # identity refresh loop.
        await self._cancel_task_attr("_polling_heartbeat_task", "heartbeat cancel")
        await self._cancel_task_attr("_bot_identity_refresh_task", "identity-refresh cancel")

        # Mark the bot "Offline" while its HTTP client is still alive (before app
        # shutdown closes it). Opt-in, non-fatal; a hard crash leaves the
        # last-known status — the expected limitation of a profile-text indicator.
        try:
            await self._await_disconnect_step(
                self._set_status_indicator(online=False),
                _DISCONNECT_STEP_TIMEOUT, "status-indicator update",
            )
        except Exception:
            pass

        await self._await_disconnect_step(
            self._cancel_pending_delivery_tasks(),
            _DISCONNECT_STEP_TIMEOUT, "pending-delivery cancel",
        )

        if self._app:
            try:
                # Bounded: a CLOSE-WAIT socket can wedge updater.stop() on epoll
                # forever; on timeout fall through to app.stop()/shutdown().
                if self._app.updater and self._app.updater.running:
                    try:
                        await self._await_disconnect_step(
                            self._app.updater.stop(), _UPDATER_STOP_TIMEOUT, "updater.stop()"
                        )
                    except Exception as stop_error:
                        logger.warning(
                            "[%s] updater.stop() failed during disconnect: %s",
                            self.name, _redact_telegram_error_text(stop_error),
                        )
                # app.stop()/shutdown() can also block on a half-dead httpx pool.
                if self._app.running:
                    await self._await_disconnect_step(
                        self._app.stop(), _DISCONNECT_STEP_TIMEOUT, "app.stop()"
                    )
                await self._await_disconnect_step(
                    self._app.shutdown(), _DISCONNECT_STEP_TIMEOUT, "app.shutdown()"
                )
            except Exception as e:
                logger.warning(
                    "[%s] Error during Telegram disconnect: %s",
                    self.name, _redact_telegram_error_text(e),
                )

        self._app = None
        self._bot = None
        logger.info("[%s] Disconnected from Telegram", self.name)

    def _should_thread_reply(self, reply_to: Optional[str], chunk_index: int) -> bool:
        """Whether this chunk (0 = first) should reply-thread to ``reply_to``, per reply_to_mode."""
        if not reply_to:
            return False
        mode = self._reply_to_mode
        if mode == "off":
            return False
        elif mode == "all":
            return True
        else:  # "first" (default)
            return chunk_index == 0

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> SendResult:
        """Send a message to a Telegram chat."""
        if not self._bot:
            live = self._replacement_telegram_adapter()
            if live is not None:
                return await live.send(chat_id, content, reply_to, metadata)
            if self._is_permanent_fatal() or not await self._wait_for_reconnection():
                return SendResult(
                    success=False, error="Not connected", retryable=not self._is_permanent_fatal()
                )
            live = self._replacement_telegram_adapter()
            if not self._bot and live is not None:
                return await live.send(chat_id, content, reply_to, metadata)
            if not self._bot:
                return SendResult(success=False, error="Not connected", retryable=True)

        # getattr() — tests build adapters via object.__new__() (no __init__).
        if getattr(self, "_send_path_degraded", False):
            return SendResult(success=False, error="send_path_degraded", retryable=True)

        # Skip whitespace-only text to prevent Telegram 400 empty-text errors.
        if not content or not content.strip():
            return SendResult(success=True, message_id=None)

        try:
            # Bot API 10.1 rich fast-path (sendRichMessage renders tables/task lists
            # natively). Falls through to legacy MarkdownV2 on permanent/capability
            # errors or DM-topic routing skips; returns directly on success or on a
            # transient failure (which must NOT be legacy-resent).
            if self._should_attempt_rich(content, metadata=metadata):
                rich_result = await self._try_send_rich(chat_id, content, reply_to, metadata)
                if rich_result is not None:
                    # Re-trigger typing ONLY for intermediate sends; on the final
                    # reply (metadata["notify"]) the refresh loop is already torn
                    # down and re-arming Telegram's ~5s timer leaves the bubble
                    # lingering (no Bot API call cancels it).
                    if rich_result.success and not (metadata or {}).get("notify"):
                        try:
                            await self.send_typing(chat_id, metadata=metadata)
                        except Exception:
                            pass  # Typing failures are non-fatal
                    return rich_result

            formatted = self.format_message(content)
            chunks = self.truncate_message(formatted, self.MAX_MESSAGE_LENGTH, len_fn=utf16_len)
            if len(chunks) > 1:
                # truncate_message appends a raw " (1/2)" suffix; escape the
                # MarkdownV2-special parentheses so Telegram doesn't reject the chunk.
                chunks = [
                    _separate_chunk_indicator_from_fence(
                        re.sub(r" \((\d+)/(\d+)\)$", r" \\(\1/\2\\)", chunk)
                    )
                    for chunk in chunks
                ]

            message_ids = []
            thread_id = self._metadata_thread_id(metadata)
            requested_thread_id = self._message_thread_id_for_send(thread_id)
            used_thread_fallback = False

            try:
                from telegram.error import NetworkError as _NetErr
            except ImportError:
                _NetErr = OSError  # type: ignore[misc,assignment]
            try:
                from telegram.error import BadRequest as _BadReq
            except ImportError:
                _BadReq = None  # type: ignore[assignment,misc]
            try:
                from telegram.error import TimedOut as _TimedOut
            except (ImportError, AttributeError):
                _TimedOut = None  # type: ignore[assignment,misc]

            for i, chunk in enumerate(chunks):
                retried_thread_not_found = False
                private_dm_topic_send, dm_topic_reply_to_off, reply_to_id = self._chunk_reply_routing(
                    chat_id, reply_to, metadata, thread_id, i
                )
                if private_dm_topic_send and reply_to_id is None and not dm_topic_reply_to_off:
                    return SendResult(
                        success=False, error=self._dm_topic_missing_anchor_error(), retryable=False
                    )
                thread_kwargs = self._thread_kwargs_for_send(
                    chat_id, thread_id, metadata,
                    reply_to_message_id=reply_to_id, reply_to_mode=self._reply_to_mode,
                )
                if used_thread_fallback and thread_kwargs.get("message_thread_id") is not None:
                    thread_kwargs = dict(thread_kwargs)
                    thread_kwargs["message_thread_id"] = None
                effective_thread_id = thread_kwargs.get("message_thread_id")

                msg = None
                for _send_attempt in range(3):
                    try:
                        # Try Markdown first, fall back to plain text if it fails
                        send_kwargs = {
                            "chat_id": normalize_telegram_chat_id(chat_id),
                            "reply_to_message_id": reply_to_id,
                            **thread_kwargs,
                            **self._link_preview_kwargs(),
                            **self._notification_kwargs(metadata),
                        }
                        try:
                            msg = await self._bot.send_message(
                                text=chunk, parse_mode=ParseMode.MARKDOWN_V2, **send_kwargs,
                            )
                        except Exception as md_error:
                            if "parse" in str(md_error).lower() or "markdown" in str(md_error).lower():
                                logger.warning("[%s] MarkdownV2 parse failed, falling back to plain text: %s", self.name, md_error)
                                msg = await self._bot.send_message(
                                    text=_strip_mdv2(chunk), parse_mode=None, **send_kwargs,
                                )
                            else:
                                raise
                        break  # success
                    except _NetErr as send_err:
                        # BadRequest subclasses NetworkError in PTB but is permanent;
                        # handle specific cases instead of blindly retrying.
                        if _BadReq and isinstance(send_err, _BadReq):
                            if self._is_thread_not_found_error(send_err) and effective_thread_id is not None:
                                if private_dm_topic_send or (metadata and metadata.get("telegram_dm_topic_created_for_send")):
                                    return SendResult(
                                        success=False, error=str(send_err), retryable=False
                                    )
                                # Telegram returns one-off "thread not found" flakes that
                                # recover on immediate retry: try the same thread_id once
                                # (no sleep) before falling back to a plain send.
                                if not retried_thread_not_found:
                                    retried_thread_not_found = True
                                    logger.warning(
                                        "[%s] Thread %s not found, retrying once with same thread_id",
                                        self.name, effective_thread_id,
                                    )
                                    continue
                                # Second failure: thread is genuinely gone. Retry without
                                # message_thread_id and prune the stale binding so future
                                # inbound messages aren't redirected back to it.
                                logger.warning(
                                    "[%s] Thread %s not found, retrying without message_thread_id",
                                    self.name, effective_thread_id,
                                )
                                self._prune_stale_dm_topic_binding(
                                    chat_id, effective_thread_id, metadata=metadata
                                )
                                used_thread_fallback = True
                                effective_thread_id = None
                                thread_kwargs = {"message_thread_id": None}
                                continue
                            err_lower = str(send_err).lower()
                            if "message to be replied not found" in err_lower and reply_to_id is not None:
                                if private_dm_topic_send:
                                    safe_send_error = _redact_telegram_error_text(send_err)
                                    return SendResult(
                                        success=False, error=safe_send_error, retryable=False
                                    )
                                # Reply target deleted before we could reply. For private-
                                # topic fallback sends, message_thread_id is only valid with
                                # the reply anchor, so drop both together.
                                safe_send_error = _redact_telegram_error_text(send_err)
                                logger.warning(
                                    "[%s] Reply target deleted, retrying without reply_to: %s",
                                    self.name, safe_send_error,
                                )
                                reply_to_id = None
                                if metadata and metadata.get("telegram_dm_topic_reply_fallback"):
                                    thread_kwargs = {}
                                    effective_thread_id = None
                                else:
                                    thread_kwargs = self._thread_kwargs_for_send(
                                        chat_id, thread_id, metadata,
                                        reply_to_message_id=reply_to_id,
                                        reply_to_mode=self._reply_to_mode,
                                    )
                                    effective_thread_id = thread_kwargs.get("message_thread_id")
                                continue
                            # Other BadRequest errors are permanent — don't retry
                            raise
                        # TimedOut also subclasses NetworkError. A generic timeout may have
                        # reached Telegram, so don't retry; a wrapped ConnectTimeout (no
                        # connection) or an httpx pool timeout (explicitly not sent) is
                        # safe to retry and prevents silent drops.
                        is_pool_timeout = self._looks_like_pool_timeout(send_err)
                        if (
                            _TimedOut
                            and isinstance(send_err, _TimedOut)
                            and not self._looks_like_connect_timeout(send_err)
                            and not is_pool_timeout
                        ):
                            raise
                        if is_pool_timeout:
                            await self._drain_general_connections_after_pool_timeout()
                        if _send_attempt < 2:
                            wait = 2 ** _send_attempt
                            safe_send_error = _redact_telegram_error_text(send_err)
                            logger.warning("[%s] Network error on send (attempt %d/3), retrying in %ds: %s",
                                           self.name, _send_attempt + 1, wait, safe_send_error)
                            await asyncio.sleep(wait)
                        else:
                            raise
                    except Exception as send_err:
                        retry_after = getattr(send_err, "retry_after", None)
                        if retry_after is not None or "retry after" in str(send_err).lower():
                            wait = float(retry_after) if retry_after is not None else 1.0
                            safe_send_error = _redact_telegram_error_text(send_err)
                            # Mirror the edit path: never sleep a long server RetryAfter
                            # verbatim — it once pinned send() for 97 minutes and froze
                            # inbound on every platform from the gateway boot path.
                            if wait > _FLOOD_INLINE_WAIT_CAP_SECS:
                                logger.warning(
                                    "[%s] Telegram flood control on send "
                                    "(retry_after=%.1fs > %.0fs); failing closed "
                                    "instead of sleeping: %s",
                                    self.name, wait, _FLOOD_INLINE_WAIT_CAP_SECS, safe_send_error,
                                )
                                return _flood_cap_result(wait)
                            if _send_attempt < 2:
                                logger.warning(
                                    "[%s] Telegram flood control on send (attempt %d/3), retrying in %.1fs: %s",
                                    self.name, _send_attempt + 1, wait, safe_send_error,
                                )
                                await asyncio.sleep(wait)
                                continue
                        raise
                message_ids.append(str(msg.message_id))

            # Re-trigger typing: Telegram clears typing state when a message lands,
            # so the bubble would vanish mid-response after intermediate progress
            # messages. Skip on the FINAL reply (metadata["notify"]): the refresh
            # loop is already cancelled, so re-arming Telegram's ~5s timer would
            # leave the indicator lingering (no stop-typing API exists).
            if not (metadata or {}).get("notify"):
                try:
                    await self.send_typing(chat_id, metadata=metadata)
                except Exception:
                    pass  # Typing failures are non-fatal

            return SendResult(
                success=True,
                message_id=message_ids[0] if message_ids else None,
                raw_response={
                    "message_ids": message_ids,
                    "requested_thread_id": requested_thread_id,
                    "thread_fallback": used_thread_fallback,
                },
            )
        except Exception as e:
            safe_error = _redact_telegram_error_text(e)
            logger.error("[%s] Failed to send Telegram message: %s", self.name, safe_error)
            err_str = str(e).lower()
            error_kind = classify_send_error(e)
            # Content exceeded 4096 chars: fail so the stream consumer enters
            # fallback mode and sends the remainder.
            if "message_too_long" in err_str or "too long" in err_str:
                logger.debug(
                    "[%s] send() content too long, falling back to new-message continuation",
                    self.name,
                )
                return SendResult(success=False, error="message_too_long", error_kind="too_long")
            # TimedOut usually means the request may have reached Telegram — mark
            # non-retryable so _send_with_retry() doesn't re-send. Exceptions: a
            # wrapped ConnectTimeout and an httpx pool timeout are safe to re-send.
            _to = locals().get("_TimedOut")
            is_timeout = (_to and isinstance(e, _to)) or "timed out" in err_str
            is_connect_timeout = self._looks_like_connect_timeout(e)
            is_pool_timeout = self._looks_like_pool_timeout(e)
            return SendResult(
                success=False,
                error=safe_error,
                retryable=(is_connect_timeout or is_pool_timeout or not is_timeout),
                error_kind=error_kind,
            )

    async def send_or_update_status(
        self,
        chat_id: str,
        status_key: str,
        content: str,
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send a status message, or edit the previous one with the same key.

        First call sends and remembers the message id; later calls with the same
        (chat_id, status_key) edit it in place. If the edit fails (deleted, too
        old, …) the cached id is dropped and a fresh message is sent.
        """
        key = (str(chat_id), str(status_key))
        cached_id = self._status_message_ids.get(key)
        if cached_id is not None:
            result = await self.edit_message(
                chat_id, cached_id, content, finalize=True, metadata=metadata
            )
            if result.success:
                if result.message_id:
                    self._status_message_ids[key] = str(result.message_id)
                return result
            self._status_message_ids.pop(key, None)
        result = await self.send(chat_id, content, metadata=metadata)
        if result.success and result.message_id:
            self._status_message_ids[key] = str(result.message_id)
        return result
    async def _edit_text(self, chat_id: str, message_id: str, text: str, parse_mode: Any = None) -> None:
        """``editMessageText`` with normalized ids; ``parse_mode=None`` sends plain text."""
        kwargs: Dict[str, Any] = {
            "chat_id": normalize_telegram_chat_id(chat_id), "message_id": int(message_id), "text": text,
        }
        if parse_mode is not None:
            kwargs["parse_mode"] = parse_mode
        await self._bot.edit_message_text(**kwargs)

    async def edit_message(
        self,
        chat_id: str,
        message_id: str,
        content: str,
        *,
        finalize: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Edit a previously sent Telegram message.

        Telegram caps a message at 4096 UTF-16 codeunits. Streaming replies that
        outgrow it must NOT be truncated silently nor fail (the consumer would
        re-send a duplicate): edit with the first chunk, send the rest as
        continuations, and return the final chunk's id as the next edit target.
        """
        if not self._bot:
            return SendResult(success=False, error="Not connected")

        # Rich finalize (Bot API 10.1): when content has constructs the MarkdownV2
        # edit degrades (tables, task lists, <details>, block math), edit the preview
        # IN PLACE via rich_message — no fresh send + delete, so no duplicate
        # preview. Done before the 4,096 pre-flight because the rich cap is 32,768;
        # a rich table over the MarkdownV2 limit must not be split into legacy
        # chunks. Falls back to the legacy path on capability/permanent rejection.
        if finalize and self._rich_eligible(content):
            rich_result = await self._try_edit_rich(chat_id, message_id, content, metadata=metadata)
            if rich_result is not None:
                return rich_result

        # Pre-flight: if content already exceeds the limit, split-and-deliver
        # without a doomed edit. Mid-stream (finalize=False) we truncate instead:
        # splitting moves the edit target to a continuation, and the next token
        # chunk re-edits the full text into it → infinite duplication loop. Full
        # content is delivered when finalize=True.
        _preview_key = (str(chat_id), str(message_id))
        _saturated_preview = False
        if finalize:
            # The final edit always delivers real (full) content.
            self._last_overflow_preview.pop(_preview_key, None)
        if utf16_len(content) > self.MAX_MESSAGE_LENGTH:
            if finalize:
                return await self._edit_overflow_split(
                    chat_id, message_id, content, finalize=finalize, metadata=metadata
                )
            content = self._truncate_stream_overflow_preview(content)
            _saturated_preview = True
            # Saturated-preview dedup: past the cap every progressive edit truncates
            # to the same text; re-sending is a visual no-op that still burns flood
            # budget (~1 edit/0.8s trips 200s+ penalties and hangs final delivery).
            if self._last_overflow_preview.get(_preview_key) == content:
                return SendResult(success=True, message_id=message_id)
        elif not finalize:
            # Content shrank back under the cap — clear stale saturation state so
            # dedup can't mask a real edit later.
            self._last_overflow_preview.pop(_preview_key, None)

        try:
            if not finalize:
                await self._edit_text(chat_id, message_id, content)
                if _saturated_preview:
                    self._last_overflow_preview[_preview_key] = content
                return SendResult(success=True, message_id=message_id)

            formatted = self.format_message(content)
            try:
                await self._edit_text(chat_id, message_id, formatted, ParseMode.MARKDOWN_V2)
            except Exception as fmt_err:
                # "Message is not modified" is a no-op, not an error
                if "not modified" in str(fmt_err).lower():
                    return SendResult(success=True, message_id=message_id)
                safe_format_error = _redact_telegram_error_text(fmt_err)
                logger.warning(
                    "[%s] MarkdownV2 edit failed, falling back to plain text: %s",
                    self.name, safe_format_error,
                )
                _plain = _strip_mdv2(content) if content else content
                await self._edit_text(chat_id, message_id, _plain)
            return SendResult(success=True, message_id=message_id)
        except Exception as e:
            err_str = str(e).lower()
            if "not modified" in err_str:
                return SendResult(success=True, message_id=message_id)
            # Reactive split-and-deliver: parse_mode formatting (MarkdownV2 escapes)
            # can inflate the payload past the limit even when raw text was under.
            if "message_too_long" in err_str or "too long" in err_str:
                logger.debug(
                    "[%s] edit_message overflow (%d UTF-16 > %d), splitting",
                    self.name, utf16_len(content), self.MAX_MESSAGE_LENGTH,
                )
                if finalize:
                    return await self._edit_overflow_split(
                        chat_id, message_id, content, finalize=finalize, metadata=metadata
                    )
                # Mid-stream: truncate and retry instead of splitting.
                truncated = self._truncate_stream_overflow_preview(content)
                if self._last_overflow_preview.get(_preview_key) == truncated:
                    # Saturated-preview dedup (see pre-flight path above).
                    return SendResult(success=True, message_id=message_id)
                await self._edit_text(chat_id, message_id, truncated)
                self._last_overflow_preview[_preview_key] = truncated
                return SendResult(success=True, message_id=message_id)
            # Flood control: short waits retry inline; long waits fail immediately so
            # streaming falls back to a normal final send instead of a clipped partial.
            retry_after = getattr(e, "retry_after", None)
            if retry_after is not None or "retry after" in err_str:
                wait = retry_after if retry_after else 1.0
                logger.warning("[%s] Telegram flood control, waiting %.1fs", self.name, wait)
                if wait > _FLOOD_INLINE_WAIT_CAP_SECS:
                    return _flood_cap_result(wait)
                await asyncio.sleep(wait)
                try:
                    await self._edit_text(chat_id, message_id, content)
                    return SendResult(success=True, message_id=message_id)
                except Exception as retry_err:
                    safe_retry_error = _redact_telegram_error_text(retry_err)
                    logger.error(
                        "[%s] Edit retry failed after flood wait: %s", self.name, safe_retry_error
                    )
                    return SendResult(success=False, error=safe_retry_error)
            # Transient network errors must not permanently disable progress-message
            # editing: mark retryable so the caller keeps trying next update cycle.
            _transient_markers = (
                "connecterror", "connect error", "connection error", "networkerror",
                "network error", "timed out", "readtimeout", "writetimeout",
                "server disconnected", "temporarily unavailable", "temporary failure", "httpx",
            )
            _is_transient = any(m in err_str for m in _transient_markers)
            if _is_transient:
                safe_error = _redact_telegram_error_text(e)
                logger.warning(
                    "[%s] Transient network error editing message %s (will retry): %s",
                    self.name, message_id, safe_error,
                )
                return SendResult(success=False, error=safe_error, retryable=True)
            safe_error = _redact_telegram_error_text(e)
            logger.error(
                "[%s] Failed to edit Telegram message %s: %s", self.name, message_id, safe_error
            )
            return SendResult(success=False, error=safe_error)

    def _truncate_stream_overflow_preview(self, content: str) -> str:
        """One-message preview for oversized streaming edits.

        Streaming edits must keep targeting the original message; splitting
        mid-stream would move the active id and repeat the overflow cycle. Final
        edits use ``_edit_overflow_split`` to deliver the complete response.
        """
        return self.truncate_message(content, self.MAX_MESSAGE_LENGTH, len_fn=utf16_len)[0]

    async def _edit_overflow_split(
        self,
        chat_id: str,
        message_id: str,
        content: str,
        *,
        finalize: bool,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Split an oversized edit across the existing message + continuations.

        Edit ``message_id`` with chunk 1 (``(1/N)`` suffix preserved), then send
        the remaining chunks as replies to the previous chunk. Returns
        ``SendResult(success=True, message_id=<last-chunk-id>,
        continuation_message_ids=(...))`` so the consumer keeps editing the most
        recent visible message. ``success=False`` only if the first-chunk edit
        itself fails — a real adapter problem, not an overflow.
        """
        chunks = self.truncate_message(content, self.MAX_MESSAGE_LENGTH, len_fn=utf16_len)
        if len(chunks) <= 1:
            # Defensive: caller pre-flighted, but a single chunk just edits normally.
            chunks = [content]

        # Step 1 — edit the existing message with the first chunk.
        first_chunk = chunks[0]
        try:
            if finalize:
                # Mirror edit_message's happy path: format_message + parse_mode.
                formatted = _separate_chunk_indicator_from_fence(self.format_message(first_chunk))
                try:
                    await self._edit_text(chat_id, message_id, formatted, ParseMode.MARKDOWN_V2)
                except Exception as fmt_err:
                    if "not modified" not in str(fmt_err).lower():
                        logger.warning(
                            "[%s] Overflow split: MarkdownV2 first-chunk edit "
                            "failed, falling back to plain text: %s",
                            self.name, _redact_telegram_error_text(fmt_err),
                        )
                        await self._edit_text(chat_id, message_id, _strip_mdv2(first_chunk))
            else:
                await self._edit_text(chat_id, message_id, first_chunk)
        except Exception as e:
            err_str = str(e).lower()
            if "not modified" in err_str:
                # First chunk identical to current text — still send continuations.
                pass
            else:
                logger.error(
                    "[%s] Overflow split: first-chunk edit failed: %s",
                    self.name, _redact_telegram_error_text(e), exc_info=True,
                )
                return SendResult(success=False, error=_redact_telegram_error_text(e))

        # Step 2 — send remaining chunks as reply-threaded continuations. Calls
        # self._bot.send_message directly to skip self.send's pre-chunking (chunks
        # are already sized). Best-effort MarkdownV2 with plain fallback, like send().
        continuation_ids: list[str] = []
        delivered_chunks = [first_chunk]
        prev_id = message_id
        thread_id = self._metadata_thread_id(metadata)
        for chunk in chunks[1:]:
            sent_msg = None
            reply_to_id = int(prev_id) if prev_id else None
            thread_kwargs = self._thread_kwargs_for_send(
                chat_id, thread_id, metadata, reply_to_message_id=reply_to_id
            )
            for use_markdown in (True, False) if finalize else (False,):
                try:
                    if use_markdown:
                        text = _separate_chunk_indicator_from_fence(self.format_message(chunk))
                    else:
                        # On finalize the MarkdownV2 attempt failed: degrade to stripped
                        # text, never the raw chunk (raw ** / ``` would render
                        # literally); streaming previews stay raw.
                        text = _strip_mdv2(chunk) if finalize else chunk
                    sent_msg = await self._bot.send_message(
                        chat_id=normalize_telegram_chat_id(chat_id),
                        text=text,
                        parse_mode=ParseMode.MARKDOWN_V2 if use_markdown else None,
                        reply_to_message_id=reply_to_id,
                        **thread_kwargs,
                        **self._link_preview_kwargs(),
                        **self._notification_kwargs(metadata),
                    )
                    break
                except Exception as send_err:
                    if "reply message not found" in str(send_err).lower():
                        # Drop the reply anchor and retry. Private DM topic fallback
                        # needs anchor + topic id together; forum topics keep thread id.
                        retry_thread_kwargs = (
                            {}
                            if metadata and metadata.get("telegram_dm_topic_reply_fallback")
                            else self._thread_kwargs_for_send(
                                chat_id, thread_id, metadata, reply_to_message_id=None
                            )
                        )
                        try:
                            sent_msg = await self._bot.send_message(
                                chat_id=normalize_telegram_chat_id(chat_id),
                                text=_strip_mdv2(chunk) if finalize else chunk,
                                **retry_thread_kwargs,
                                **self._link_preview_kwargs(),
                                **self._notification_kwargs(metadata),
                            )
                            break
                        except Exception as _retry_err:
                            logger.warning(
                                "[%s] Overflow continuation no-reply retry failed: %s",
                                self.name, _redact_telegram_error_text(_retry_err),
                            )
                            sent_msg = None
                            break
                    if use_markdown:
                        continue  # try plain text on next loop iteration
                    logger.warning(
                        "[%s] Overflow continuation send failed: %s",
                        self.name, _redact_telegram_error_text(send_err),
                    )
                    sent_msg = None
                    break
            if sent_msg is None:
                # Partial delivery: do NOT report success — the stream consumer treats
                # a successful edit as final delivery on got_done, which would suppress
                # fallback delivery and leave the topic clipped.
                logger.warning(
                    "[%s] Overflow split: stopped at %d/%d chunks delivered",
                    self.name, 1 + len(continuation_ids), len(chunks),
                )
                delivered_prefix = "".join(
                    re.sub(r" \(\d+/\d+\)$", "", delivered) for delivered in delivered_chunks
                )
                return SendResult(
                    success=False,
                    message_id=prev_id,
                    error="overflow_continuation_failed",
                    retryable=True,
                    raw_response={
                        "partial_overflow": True,
                        "delivered_chunks": 1 + len(continuation_ids),
                        "total_chunks": len(chunks),
                        "last_message_id": prev_id,
                        "delivered_prefix": delivered_prefix,
                        "continuation_message_ids": tuple(continuation_ids),
                    },
                    continuation_message_ids=tuple(continuation_ids),
                )
            new_id = str(getattr(sent_msg, "message_id", "")) or prev_id
            continuation_ids.append(new_id)
            delivered_chunks.append(chunk)
            prev_id = new_id

        last_id = continuation_ids[-1] if continuation_ids else message_id
        logger.debug(
            "[%s] Overflow split delivered %d chunks; last_id=%s",
            self.name, 1 + len(continuation_ids), last_id,
        )
        return SendResult(
            success=True, message_id=last_id, continuation_message_ids=tuple(continuation_ids)
        )
    async def delete_message(self, chat_id: str, message_id: str) -> bool:
        """Delete a bot-posted message (Bot API allows it within 48h).

        Used by the stream consumer's fresh-final cleanup to remove long-lived
        previews. Failures are non-fatal — the caller leaves the preview in place.
        """
        if not self._bot:
            return False
        try:
            await self._bot.delete_message(
                chat_id=normalize_telegram_chat_id(chat_id), message_id=int(message_id)
            )
            return True
        except Exception as e:
            logger.debug(
                "[%s] Failed to delete Telegram message %s: %s",
                self.name, message_id, _redact_telegram_error_text(e),
            )
            return False

    def supports_draft_streaming(
        self,
        chat_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Telegram supports sendMessageDraft for private chats only (Bot API 9.5);
        groups/supergroups/channels use the edit-based path. Also requires PTB >= 22.6
        (``send_message_draft``); older installs fall back to edits even on DMs.

        ``rich_drafts`` controls draft *format*, not availability: with rich final
        delivery but no rich drafts, keep the preview ephemeral and persist via
        ``sendRichMessage`` — an in-place edit can't be relied on to upgrade through
        ``rich_message``, and the fallback formatter turns tables into bullets.
        """
        if not self._bot or not hasattr(self._bot, "send_message_draft"):
            return False
        return (chat_type or "").lower() in {"dm", "private"}

    async def send_draft(
        self,
        chat_id: str,
        draft_id: int,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Stream a partial message via Telegram's native draft API.

        Uses ``sendRichMessageDraft`` (Bot API 10.1) when rich is enabled and
        supported, else plain ``sendMessageDraft``. Reusing ``draft_id`` animates
        the preview. The caller sends the final text via ``send``; the draft clears
        on the client (no Bot API "promotes" a draft to a real message).
        """
        if not self._bot:
            return SendResult(success=False, error="not_connected")

        # Rich draft fast-path: preview with the same raw markdown the final
        # sendRichMessage persists. Any failure degrades to the plain draft below.
        if self._should_attempt_rich_draft(content) and await self._try_send_rich_draft(
            chat_id, draft_id, content, metadata
        ):
            # Drafts have no message_id; report success without one.
            return SendResult(success=True, message_id=None)

        if not hasattr(self._bot, "send_message_draft"):
            return SendResult(success=False, error="api_unavailable")

        # Drafts share the regular-send UTF-16 length contract.
        text = content if len(content) <= self.MAX_MESSAGE_LENGTH else \
            self.truncate_message(content, self.MAX_MESSAGE_LENGTH, len_fn=utf16_len)[0]

        # Apply the same MarkdownV2 conversion as ``send`` so the draft doesn't snap
        # from raw to formatted at the end; try MarkdownV2 then plain so one bad
        # token never kills draft streaming. Exception: when the persistent
        # response will be a Rich Message but rich drafts are disabled, send a raw
        # preview — the legacy formatter would rewrite pipe tables into bullets.
        plain_rich_preview = bool(
            getattr(self, "_rich_messages_enabled", False)
            and not getattr(self, "_rich_drafts_enabled", False)
            and self._needs_rich_rendering(text)
        )
        draft_modes = (False,) if plain_rich_preview else (True, False)
        draft_thread_kwargs = self._thread_kwargs_for_draft(chat_id, metadata)
        for use_markdown in draft_modes:
            kwargs: Dict[str, Any] = {
                "chat_id": normalize_telegram_chat_id(chat_id),
                "draft_id": int(draft_id),
                "text": self.format_message(text) if use_markdown else text,
            }
            if use_markdown:
                kwargs["parse_mode"] = ParseMode.MARKDOWN_V2
            kwargs.update(draft_thread_kwargs)

            try:
                ok = await self._bot.send_message_draft(**kwargs)
                if ok:
                    # Drafts have no message_id; success means the frame landed.
                    return SendResult(success=True, message_id=None)
                return SendResult(success=False, error="draft_rejected")
            except Exception as e:
                # MarkdownV2 parse failure (BadRequest) → retry once as plain text.
                # Anything else, or a plain-text failure, returns to the caller,
                # which falls back to edit-based streaming for this response.
                if use_markdown and self._is_bad_request_error(e):
                    logger.debug(
                        "[%s] sendMessageDraft MarkdownV2 rejected, retrying "
                        "as plain text (chat=%s draft_id=%s): %s",
                        self.name, chat_id, draft_id, _redact_telegram_error_text(e),
                    )
                    continue
                logger.debug(
                    "[%s] sendMessageDraft failed (chat=%s draft_id=%s): %s",
                    self.name, chat_id, draft_id, e,
                )
                return SendResult(success=False, error=_redact_telegram_error_text(e))

        return SendResult(success=False, error="draft_rejected")

    async def _send_message_with_thread_fallback(self, **kwargs):
        """Send a message, retrying once without message_thread_id on 'Message
        thread not found'. For control-style sends (approval prompts, pickers,
        update prompts) that can carry a stale thread_id; ``send`` has its own.
        """
        if not self._bot:
            raise RuntimeError("Not connected")

        message_thread_id = kwargs.get("message_thread_id")
        try:
            return await self._bot.send_message(**kwargs)
        except Exception as send_err:
            if (
                message_thread_id is not None
                and self._is_bad_request_error(send_err)
                and self._is_thread_not_found_error(send_err)
            ):
                logger.warning(
                    "[%s] Thread %s not found for control message, retrying without message_thread_id",
                    self.name, message_thread_id,
                )
                # Same prune as the streaming send path: the topic is gone, so the
                # state.db binding must go too. Control sends carry no gateway
                # metadata, so the prune namespaces by this adapter's profile stamp.
                self._prune_stale_dm_topic_binding(kwargs.get("chat_id"), message_thread_id)
                retry_kwargs = dict(kwargs)
                retry_kwargs.pop("message_thread_id", None)
                return await self._bot.send_message(**retry_kwargs)
            raise

    async def _send_control_message(
        self,
        chat_id: str,
        text: str,
        *,
        parse_mode: Any,
        thread_id: Optional[str],
        metadata: Optional[Dict[str, Any]],
        reply_markup: Any = None,
        reply_to_mode: Optional[str] = None,
    ):
        """Send a control-style message (prompt/picker) with topic routing + thread fallback."""
        reply_to_id = self._reply_to_message_id_for_send(None, metadata, reply_to_mode=reply_to_mode)
        kwargs: Dict[str, Any] = {
            "chat_id": normalize_telegram_chat_id(chat_id),
            "text": text,
            "parse_mode": parse_mode,
            **self._link_preview_kwargs(),
        }
        if reply_markup is not None:
            kwargs["reply_markup"] = reply_markup
        kwargs["reply_to_message_id"] = reply_to_id
        kwargs.update(
            self._thread_kwargs_for_send(
                chat_id, thread_id, metadata,
                reply_to_message_id=reply_to_id, reply_to_mode=reply_to_mode,
            )
        )
        return await self._send_message_with_thread_fallback(**kwargs)

    async def send_update_prompt(
        self, chat_id: str, prompt: str, default: str = "",
        session_key: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send an inline-keyboard Yes/No prompt for the gateway ``/update`` watcher."""
        if not self._bot:
            return SendResult(success=False, error="Not connected")
        try:
            default_hint = f" (default: {default})" if default else ""
            text = self.format_message(f"⚕ *Update needs your input:*\n\n{prompt}{default_hint}")
            keyboard = InlineKeyboardMarkup([[
                InlineKeyboardButton("✓ Yes", callback_data="update_prompt:y"),
                InlineKeyboardButton("✗ No", callback_data="update_prompt:n"),
            ]])
            msg = await self._send_control_message(
                chat_id, text, parse_mode=ParseMode.MARKDOWN_V2, reply_markup=keyboard,
                thread_id=self._metadata_thread_id(metadata), metadata=metadata,
                reply_to_mode=self._reply_to_mode,
            )
            return SendResult(success=True, message_id=str(msg.message_id))
        except Exception as e:
            logger.warning("[%s] send_update_prompt failed: %s", self.name, _redact_telegram_error_text(e))
            return SendResult(success=False, error=_redact_telegram_error_text(e))

    # Template attrs for the shared _format_exec_approval core (HTML mode).
    _EA_HEADER = "⚠️ <b>Command Approval Required</b>\n\n"
    _EA_CODE_OPEN = "<pre>"
    _EA_CODE_CLOSE = "</pre>\n\n"
    _EA_SMART_DENY_LINE = "\n\n<b>Smart DENY:</b> owner override applies to this one operation only."
    _EA_CMD_BUDGET = 3800

    def _ea_escape(self, text: str) -> str:
        return _html.escape(text)

    async def send_exec_approval(
        self, chat_id: str, command: str, session_key: str,
        description: str = "dangerous command",
        metadata: Optional[Dict[str, Any]] = None,
        allow_permanent: bool = True,
        allow_session: bool = True,
        smart_denied: bool = False,
    ) -> SendResult:
        """Send an inline-keyboard approval prompt; buttons call
        ``resolve_gateway_approval()`` like the text ``/approve`` flow."""
        if not self._bot:
            return SendResult(success=False, error="Not connected")

        try:
            text = self._format_exec_approval(command, description, smart_denied)
            thread_id = self._metadata_thread_id(metadata)

            # Short monotonic ids in callback_data map back to session_key.
            import itertools
            if not hasattr(self, "_approval_counter"):
                self._approval_counter = itertools.count(1)
            approval_id = next(self._approval_counter)

            buttons = [InlineKeyboardButton("✅ Allow Once", callback_data=f"ea:once:{approval_id}")]
            if not smart_denied and allow_session:
                buttons.append(
                    InlineKeyboardButton("✅ Session", callback_data=f"ea:session:{approval_id}")
                )
                if allow_permanent:
                    buttons.append(
                        InlineKeyboardButton("✅ Always", callback_data=f"ea:always:{approval_id}")
                    )
            buttons.append(InlineKeyboardButton("❌ Deny", callback_data=f"ea:deny:{approval_id}"))
            # 2x2 rows keep labels readable on mobile (a 4-button row truncates).
            rows = [buttons[i:i + 2] for i in range(0, len(buttons), 2)]
            keyboard = InlineKeyboardMarkup(rows)

            msg = await self._send_control_message(
                chat_id, text, parse_mode=ParseMode.HTML, reply_markup=keyboard,
                thread_id=thread_id, metadata=metadata, reply_to_mode=self._reply_to_mode,
            )
            self._approval_state[approval_id] = session_key
            return SendResult(success=True, message_id=str(msg.message_id))
        except Exception as e:
            logger.warning("[%s] send_exec_approval failed: %s", self.name, _redact_telegram_error_text(e))
            return SendResult(success=False, error=_redact_telegram_error_text(e))

    async def send_slash_confirm(
        self, chat_id: str, title: str, message: str, session_key: str,
        confirm_id: str, metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Render a three-button slash-command confirmation prompt."""
        if not self._bot:
            return SendResult(success=False, error="Not connected")

        try:
            preview = self.format_message(self._truncate_preview(message, 3800))
            keyboard = InlineKeyboardMarkup([
                [
                    InlineKeyboardButton("✅ Approve Once", callback_data=f"sc:once:{confirm_id}"),
                    InlineKeyboardButton("🔒 Always Approve", callback_data=f"sc:always:{confirm_id}"),
                ],
                [InlineKeyboardButton("❌ Cancel", callback_data=f"sc:cancel:{confirm_id}")],
            ])
            msg = await self._send_control_message(
                chat_id, preview, parse_mode=ParseMode.MARKDOWN_V2, reply_markup=keyboard,
                thread_id=self._metadata_thread_id(metadata), metadata=metadata,
                reply_to_mode=self._reply_to_mode,
            )
            self._slash_confirm_state[confirm_id] = session_key
            return SendResult(success=True, message_id=str(msg.message_id))
        except Exception as e:
            logger.warning("[%s] send_slash_confirm failed: %s", self.name, _redact_telegram_error_text(e))
            return SendResult(success=False, error=_redact_telegram_error_text(e))

    async def send_clarify(
        self,
        chat_id: str,
        question: str,
        choices: Optional[list],
        clarify_id: str,
        session_key: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Render a clarify prompt.

        With ``choices``: one numbered button per option plus "✏️ Other (type
        answer)", which flips the entry into text-capture mode. Without: plain
        question, no buttons; the gateway's text-intercept captures the next message.
        """
        if not self._bot:
            return SendResult(success=False, error="Not connected")

        try:
            text = f"❓ {_html.escape(question)}"
            thread_id = self._metadata_thread_id(metadata)

            if choices:
                # Full option text goes in the body (mobile truncates button labels);
                # buttons keep short numeric labels.
                option_lines = "\n".join(
                    f"{i + 1}. {_html.escape(str(c))}" for i, c in enumerate(choices)
                )
                text += f"\n\n{option_lines}"

            keyboard = None
            if choices:
                # Telegram caps callback_data at 64 bytes; keep "cl:<id>:<idx>" short.
                rows = [
                    [InlineKeyboardButton(str(idx + 1), callback_data=f"cl:{clarify_id}:{idx}")]
                    for idx in range(len(choices))
                ]
                rows.append([InlineKeyboardButton("✏️ Other (type answer)", callback_data=f"cl:{clarify_id}:other")])
                keyboard = InlineKeyboardMarkup(rows)

            msg = await self._send_control_message(
                chat_id, text, parse_mode=ParseMode.HTML, reply_markup=keyboard,
                thread_id=thread_id, metadata=metadata,
            )
            self._clarify_state[clarify_id] = session_key
            return SendResult(success=True, message_id=str(msg.message_id))
        except Exception as e:
            logger.warning("[%s] send_clarify failed: %s", self.name, _redact_telegram_error_text(e))
            return SendResult(success=False, error=_redact_telegram_error_text(e))

    async def send_model_picker(
        self,
        chat_id: str,
        providers: list,
        current_model: str,
        current_provider: str,
        session_key: str,
        on_model_selected,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send an inline-keyboard model picker: provider → model drill-down, edited in place."""
        if not self._bot:
            return SendResult(success=False, error="Not connected")
        try:
            from hermes_cli.providers import get_label
        except ImportError:
            def get_label(slug):
                return slug
        try:
            keyboard, provider_page_info = self._build_provider_keyboard(providers, 0)
            provider_label = get_label(current_provider)
            text = self.format_message(
                f"⚙ *Model Configuration*\n\n"
                f"Current model: `{current_model or 'unknown'}`\n"
                f"Provider: {provider_label}\n\n"
                f"Select a provider:{provider_page_info}"
            )
            msg = await self._send_control_message(
                chat_id, text, parse_mode=ParseMode.MARKDOWN_V2, reply_markup=keyboard,
                thread_id=metadata.get("thread_id") if metadata else None, metadata=metadata,
                reply_to_mode=self._reply_to_mode,
            )
            self._model_picker_state[str(chat_id)] = {
                "msg_id": msg.message_id, "providers": providers, "session_key": session_key,
                "on_model_selected": on_model_selected, "current_model": current_model,
                "current_provider": current_provider, "provider_page": 0,
            }
            return SendResult(success=True, message_id=str(msg.message_id))
        except Exception as e:
            logger.warning("[%s] send_model_picker failed: %s", self.name, _redact_telegram_error_text(e))
            return SendResult(success=False, error=_redact_telegram_error_text(e))

    _PROVIDER_PAGE_SIZE = 10

    async def send_choice_picker(
        self,
        chat_id: str,
        title: str,
        choices: list,
        session_key: str,
        on_choice_selected,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Flat inline-keyboard picker (one tap → one value) for /reasoning, /fast, etc.

        Each choice dict: ``{"value": str, "label": str, "is_current": bool}``.
        """
        if not self._bot:
            return SendResult(success=False, error="Not connected")
        try:
            buttons = []
            for i, choice in enumerate(choices):
                label = str(choice.get("label") or choice.get("value") or "")
                if choice.get("is_current"):
                    label = f"✓ {label}"
                buttons.append(InlineKeyboardButton(label, callback_data=f"cp:{i}"))
            if not buttons:
                return SendResult(success=False, error="No choices")
            # Two buttons per row keeps labels readable on mobile.
            keyboard = InlineKeyboardMarkup([buttons[i:i + 2] for i in range(0, len(buttons), 2)])
            msg = await self._send_control_message(
                chat_id, self.format_message(title), parse_mode=ParseMode.MARKDOWN_V2, reply_markup=keyboard,
                thread_id=metadata.get("thread_id") if metadata else None, metadata=metadata,
                reply_to_mode=self._reply_to_mode,
            )
            self._choice_picker_state[str(chat_id)] = {
                "msg_id": msg.message_id, "choices": choices, "session_key": session_key,
                "on_choice_selected": on_choice_selected,
            }
            return SendResult(success=True, message_id=str(msg.message_id))
        except Exception as e:
            logger.warning("[%s] send_choice_picker failed: %s", self.name, _redact_telegram_error_text(e))
            return SendResult(success=False, error=_redact_telegram_error_text(e))

    async def _handle_choice_picker_callback(self, query, data: str, chat_id: str) -> None:
        """Handle choice picker button taps (cp:<index>)."""
        state = self._choice_picker_state.get(chat_id)
        if not state:
            await query.answer(text="Picker expired — run the command again.")
            return
        # Same auth gate as approval buttons: strangers in a shared group must not
        # flip session/config state via someone else's picker message.
        if not await self._callback_authorized(
            query, self._callback_ctx(query), "⛔ You are not authorized to change this setting."
        ):
            return
        try:
            idx = int(data[3:])
            choice = state["choices"][idx]
        except (ValueError, IndexError):
            await query.answer(text="Invalid selection.")
            return
        callback = state.get("on_choice_selected")
        if not callback:
            await query.answer(text="Picker expired.")
            return
        try:
            result_text = await callback(chat_id, str(choice.get("value") or ""))
        except Exception as exc:
            logger.error("Choice picker selection failed: %s", exc)
            result_text = f"Error applying selection: {exc}"
        try:
            await query.edit_message_text(
                text=self.format_message(result_text), parse_mode=ParseMode.MARKDOWN_V2, reply_markup=None
            )
        except Exception:
            try:
                await query.edit_message_text(text=result_text, parse_mode=None, reply_markup=None)
            except Exception:
                pass
        await query.answer()
        self._choice_picker_state.pop(chat_id, None)

    _MODEL_PAGE_SIZE = 8

    @staticmethod
    def _provider_button(p: dict) -> "InlineKeyboardButton":
        count = p.get("total_models", len(p.get("models", [])))
        label = f"{p['name']} ({count})"
        if p.get("is_current"):
            label = f"✓ {label}"
        return InlineKeyboardButton(label, callback_data=f"mp:{p['slug']}")

    @staticmethod
    def _picker_nav_row(page: int, total_pages: int, prefix: str) -> list:
        """``◀ Prev | n/N | Next ▶`` row (``prefix`` = ``mpv``/``mg`` page callback)."""
        nav: list = []
        if page > 0:
            nav.append(InlineKeyboardButton("◀ Prev", callback_data=f"{prefix}:{page - 1}"))
        nav.append(InlineKeyboardButton(f"{page + 1}/{total_pages}", callback_data="mx:noop"))
        if page < total_pages - 1:
            nav.append(InlineKeyboardButton("Next ▶", callback_data=f"{prefix}:{page + 1}"))
        return nav

    @staticmethod
    def _picker_back_cancel_row() -> list:
        return [
            InlineKeyboardButton("◀ Back", callback_data="mb"),
            InlineKeyboardButton("✗ Cancel", callback_data="mx"),
        ]

    def _build_provider_keyboard(self, providers: list, page: int = 0) -> tuple:
        """Paginated top-level provider keyboard, folding provider families.

        Families (Kimi/Moonshot, MiniMax, xAI...) become one ``mpg:<gid>`` button that
        drills into members; singles (and one-member groups) are direct ``mp:<slug>``.
        Uses the shared ``group_providers`` fold so it matches the CLI picker.
        """
        try:
            from hermes_cli.models import group_providers
        except Exception:
            group_providers = None
        by_slug = {p.get("slug"): p for p in providers}
        buttons: list = []
        if group_providers is not None:
            for row in group_providers([p.get("slug") for p in providers]):
                if row["kind"] == "group":
                    members = [by_slug[m] for m in row["members"] if m in by_slug]
                    count = sum(m.get("total_models", len(m.get("models", []))) for m in members)
                    label = f"{row['label']} ▸ ({count})"
                    if any(m.get("is_current") for m in members):
                        label = f"✓ {label}"
                    buttons.append(InlineKeyboardButton(label, callback_data=f"mpg:{row['group_id']}"))
                else:
                    p = by_slug.get(row["slug"])
                    if p is not None:
                        buttons.append(self._provider_button(p))
        else:
            for p in providers:
                buttons.append(self._provider_button(p))

        page_buttons, page_meta = self._format_choice_page(buttons, page, self._PROVIDER_PAGE_SIZE)
        rows = [page_buttons[i : i + 2] for i in range(0, len(page_buttons), 2)]
        if page_meta["total_pages"] > 1:
            rows.append(self._picker_nav_row(page_meta["page"], page_meta["total_pages"], "mpv"))
        rows.append([InlineKeyboardButton("✗ Cancel", callback_data="mx")])
        return InlineKeyboardMarkup(rows), page_meta["page_info"]

    def _build_model_keyboard(self, models: list, page: int) -> tuple:
        """Build paginated model buttons. Returns (keyboard, page_info_text)."""
        page_models, page_meta = self._format_choice_page(models, page, self._MODEL_PAGE_SIZE)
        start = page_meta["start"]
        buttons: list = []
        for i, model_id in enumerate(page_models):
            short = model_id.split("/")[-1] if "/" in model_id else model_id
            if len(short) > 38:
                short = short[:35] + "..."
            buttons.append(InlineKeyboardButton(short, callback_data=f"mm:{start + i}"))
        rows = [buttons[i : i + 2] for i in range(0, len(buttons), 2)]
        if page_meta["total_pages"] > 1:
            rows.append(self._picker_nav_row(page_meta["page"], page_meta["total_pages"], "mg"))
        rows.append(self._picker_back_cancel_row())
        return InlineKeyboardMarkup(rows), page_meta["page_info"]

    async def _picker_edit(self, query, text_md: str, keyboard) -> None:
        """Re-render the picker message in place (MarkdownV2) and ack the tap."""
        await query.edit_message_text(
            text=self.format_message(text_md), parse_mode=ParseMode.MARKDOWN_V2, reply_markup=keyboard,
        )
        await query.answer()

    async def _picker_show_models(self, query, state: dict, page: int) -> None:
        """Render the model page for the provider currently selected in ``state``."""
        models = state.get("model_list", [])
        state["model_page"] = page
        keyboard, page_info = self._build_model_keyboard(models, page)
        pname = state.get("selected_provider_name", "")
        provider_slug = state.get("selected_provider", "")
        provider = next((p for p in state["providers"] if p["slug"] == provider_slug), None)
        total = provider.get("total_models", len(models)) if provider else len(models)
        shown = len(models)
        extra = f"\n_{total - shown} more available — type `/model <name>` directly_" if total > shown else ""
        await self._picker_edit(
            query,
            f"⚙ *Model Configuration*\n\nProvider: *{pname}*{page_info}\nSelect a model:{extra}",
            keyboard,
        )

    async def _picker_show_providers(self, query, state: dict, page: int, get_label) -> None:
        """Render the (folded, paginated) provider list."""
        keyboard, provider_page_info = self._build_provider_keyboard(state["providers"], page)
        try:
            provider_label = get_label(state["current_provider"])
        except Exception:
            provider_label = state["current_provider"]
        await self._picker_edit(
            query,
            f"⚙ *Model Configuration*\n\n"
            f"Current model: `{state['current_model'] or 'unknown'}`\n"
            f"Provider: {provider_label}\n\n"
            f"Select a provider:{provider_page_info}",
            keyboard,
        )

    async def _picker_selection(self, query, state: dict, raw_idx: str) -> Optional[tuple]:
        """Resolve ``mm:``/``mc:`` index → ``(idx, model_id, provider_slug, callback)``; answers + None on error."""
        try:
            idx = int(raw_idx)
        except ValueError:
            await query.answer(text="Invalid selection.")
            return None
        model_list = state.get("model_list", [])
        if idx < 0 or idx >= len(model_list):
            await query.answer(text="Invalid model index.")
            return None
        callback = state.get("on_model_selected")
        if not callback:
            await query.answer(text="Picker expired.")
            return None
        return idx, model_list[idx], state.get("selected_provider", ""), callback

    async def _picker_switch(self, query, chat_id: str, model_id: str, provider_slug: str, callback) -> None:
        """Perform the model switch, render the result, and drop the picker state."""
        switch_failed = False
        try:
            result_text = await callback(chat_id, model_id, provider_slug)
        except Exception as exc:
            logger.error("Model picker switch failed: %s", exc)
            result_text = f"Error switching model: {exc}"
            switch_failed = True
        try:
            await query.edit_message_text(
                text=self.format_message(result_text), parse_mode=ParseMode.MARKDOWN_V2, reply_markup=None
            )
        except Exception:
            # Markdown parse failure — retry as plain text
            try:
                await query.edit_message_text(text=result_text, parse_mode=None, reply_markup=None)
            except Exception:
                pass
        await query.answer(text="Switch failed." if switch_failed else "Model switched!")
        self._model_picker_state.pop(chat_id, None)

    async def _handle_model_picker_callback(self, query, data: str, chat_id: str) -> None:
        """Handle model picker callbacks (mp:/mpg:/mpv:/mm:/mc:/mb/mx/mg:)."""
        state = self._model_picker_state.get(chat_id)
        if not state:
            await query.answer(text="Picker expired — use /model again.")
            return
        try:
            from hermes_cli.providers import get_label
        except ImportError:
            def get_label(slug):
                return slug

        if data.startswith("mp:"):
            # Provider selected: show model buttons (page 0)
            provider_slug = data[3:]
            provider = next((p for p in state["providers"] if p["slug"] == provider_slug), None)
            if not provider:
                await query.answer(text="Provider not found.")
                return
            state["selected_provider"] = provider_slug
            state["selected_provider_name"] = provider.get("name", provider_slug)
            state["model_list"] = provider.get("models", [])
            await self._picker_show_models(query, state, 0)

        elif data.startswith("mg:"):
            # Model page navigation
            try:
                page = int(data[3:])
            except ValueError:
                await query.answer(text="Invalid page.")
                return
            await self._picker_show_models(query, state, page)

        elif data.startswith("mpv:"):
            # Provider page navigation
            try:
                page = int(data[4:])
            except ValueError:
                await query.answer(text="Invalid page.")
                return
            state["provider_page"] = page
            await self._picker_show_providers(query, state, page, get_label)

        elif data.startswith("mc:"):
            # Expensive model confirmed: perform the switch
            sel = await self._picker_selection(query, state, data[3:])
            if sel is None:
                return
            _idx, model_id, provider_slug, callback = sel
            await self._picker_switch(query, chat_id, model_id, provider_slug, callback)

        elif data.startswith("mm:"):
            # Model selected: warn if expensive, else perform the switch
            sel = await self._picker_selection(query, state, data[3:])
            if sel is None:
                return
            idx, model_id, provider_slug, callback = sel
            try:
                from hermes_cli.model_selection_guards import combined_selection_warning
                # Pricing lookup may hit models.dev on a cache miss — keep it off the event loop.
                warning = await asyncio.to_thread(
                    combined_selection_warning, model_id, provider=provider_slug
                )
            except Exception:
                warning = None
            if warning is not None:
                keyboard = InlineKeyboardMarkup([
                    [InlineKeyboardButton("Switch anyway", callback_data=f"mc:{idx}")],
                    self._picker_back_cancel_row(),
                ])
                await query.edit_message_text(
                    text=self.format_message(f"⚠ *{warning.title}*\n\n{warning.message}"),
                    parse_mode=ParseMode.MARKDOWN_V2,
                    reply_markup=keyboard,
                )
                await query.answer(text="Confirm model selection")
                return
            await self._picker_switch(query, chat_id, model_id, provider_slug, callback)

        elif data.startswith("mpg:"):
            # Provider group selected: show member providers
            group_id = data[4:]
            try:
                from hermes_cli.models import PROVIDER_GROUPS
                _label, _desc, member_slugs = PROVIDER_GROUPS.get(group_id, ("", "", []))
            except Exception:
                _label, member_slugs = "", []
            by_slug = {p["slug"]: p for p in state["providers"]}
            members = [by_slug[m] for m in member_slugs if m in by_slug]
            if not members:
                await query.answer(text="Group not found.")
                return
            buttons = [self._provider_button(p) for p in members]
            rows = [buttons[i : i + 2] for i in range(0, len(buttons), 2)]
            rows.append(self._picker_back_cancel_row())
            await self._picker_edit(
                query,
                f"⚙ *Model Configuration*\n\nProvider family: *{_label or group_id}*\n\nSelect a provider:",
                InlineKeyboardMarkup(rows),
            )

        elif data == "mb":
            # Back to provider list (folds groups)
            page = int(state.get("provider_page", 0) or 0)
            await self._picker_show_providers(query, state, page, get_label)

        elif data == "mx":
            self._model_picker_state.pop(chat_id, None)
            await query.edit_message_text(text="Model selection cancelled.", reply_markup=None)
            await query.answer()

        else:
            await query.answer()  # e.g. page-counter button "mx:noop"

    async def _notify_clarify_expired(self, query, user_display: str) -> None:
        """Tell the user a clarify tap arrived too late (entry evicted by ``clarify_timeout``
        or gateway restarted) — otherwise the tap leaves a misleading ✓ the agent never sees."""
        try:
            await query.answer(text="⚠️ This prompt expired — please /retry.")
        except Exception:
            pass
        try:
            await query.edit_message_text(
                text=(
                    f"❓ {_html.escape(query.message.text or '')}\n\n"
                    "<i>⚠️ This question expired or the session reset — please /retry.</i>"
                ),
                parse_mode=ParseMode.HTML,
                reply_markup=None,
            )
        except Exception:
            pass

    async def _handle_inline_query(self, update: "Update", context: "ContextTypes.DEFAULT_TYPE") -> None:
        """Answer ``@botname <query>`` with a searchable command/skill picker.

        The BotCommand menu is capped (100/scope, ~4KB; 60-slot Hermes default), so most
        skill commands never fit the ``/`` menu. Inline mode is uncapped: results are computed
        per keystroke, paginated 50 at a time (Telegram's per-answer max). Tapping a result
        sends the command text into the chat as the user, so dispatch flows through the normal
        command path — this handler only *offers* text and is read-only by construction.

        Inline queries arrive from ANY chat (even ones the bot isn't in), so results are only
        served to users passing the same auth as inline-button callbacks; unauthorized users
        get an empty list so the installed-skill catalog is not leaked.
        """
        inline_query = getattr(update, "inline_query", None)
        if inline_query is None:
            return
        from_user = getattr(inline_query, "from_user", None)
        user_id = str(getattr(from_user, "id", "") or "").strip()
        try:
            # No chat context on inline queries — authorize on user identity alone, DM-shaped.
            authorized = bool(user_id) and self._is_callback_user_authorized(
                user_id, chat_id=user_id, chat_type="private", user_name=getattr(from_user, "username", None)
            )
        except Exception:
            logger.debug("[%s] inline picker auth check failed", self.name, exc_info=True)
            authorized = False
        if not authorized:
            try:
                from plugins.platforms.telegram.inline_picker import CACHE_TIME_SECONDS as _deny_cache
                await inline_query.answer([], cache_time=_deny_cache, is_personal=True)
            except Exception:
                logger.debug("[%s] inline picker empty answer failed", self.name, exc_info=True)
            return
        try:
            from telegram import InlineQueryResultArticle, InputTextMessageContent
            from plugins.platforms.telegram.inline_picker import (
                CACHE_TIME_SECONDS as _CACHE, build_inline_results,
            )
            results, next_offset = build_inline_results(
                getattr(inline_query, "query", "") or "", offset=getattr(inline_query, "offset", "") or ""
            )
            articles = [
                InlineQueryResultArticle(
                    id=r["id"], title=r["title"], description=r["description"],
                    input_message_content=InputTextMessageContent(r["message_text"]),
                )
                for r in results
            ]
            # is_personal: catalogs differ per user (auth, disabled skills) — never share cached pages.
            await inline_query.answer(articles, cache_time=_CACHE, is_personal=True, next_offset=next_offset)
        except Exception:
            logger.debug("[%s] inline picker answer failed", self.name, exc_info=True)

    @staticmethod
    def _callback_ctx(query) -> Dict[str, Any]:
        """Chat/thread/user context of a button tap, for the callback auth gate."""
        query_message = getattr(query, "message", None)
        query_chat = getattr(query_message, "chat", None)
        return {
            "chat_id": getattr(query_message, "chat_id", None),
            "chat_type": getattr(query_chat, "type", None),
            "thread_id": getattr(query_message, "message_thread_id", None),
            "user_name": getattr(query.from_user, "first_name", None),
        }

    async def _callback_authorized(self, query, cb: Dict[str, Any], denial_text: str) -> bool:
        """Gate a button tap on the callback allowlist; answers ``denial_text`` when refused."""
        caller_id = str(getattr(query.from_user, "id", ""))
        if self._is_callback_user_authorized(
            caller_id,
            chat_id=cb["chat_id"],
            chat_type=str(cb["chat_type"]) if cb["chat_type"] is not None else None,
            thread_id=str(cb["thread_id"]) if cb["thread_id"] is not None else None,
            user_name=cb["user_name"],
        ):
            return True
        await query.answer(text=denial_text)
        return False

    async def _handle_callback_query(self, update: "Update", context: "ContextTypes.DEFAULT_TYPE") -> None:
        """Dispatch inline keyboard button clicks on the callback_data prefix."""
        query = update.callback_query
        if not query or not query.data:
            return
        data = query.data
        cb = self._callback_ctx(query)
        # Model picker / generic choice picker (/reasoning, /fast) need a chat id.
        if data.startswith(("mp:", "mpg:", "mpv:", "mm:", "mc:", "mb", "mx", "mg:")):
            chat_id = str(query.message.chat_id) if query.message else None
            if chat_id:
                await self._handle_model_picker_callback(query, data, chat_id)
            return
        if data.startswith("cp:"):
            chat_id = str(query.message.chat_id) if query.message else None
            if chat_id:
                await self._handle_choice_picker_callback(query, data, chat_id)
            return
        for prefix, handler in (
            ("gt:", self._handle_gmail_triage_callback),
            ("ea:", self._handle_exec_approval_callback),
            ("sc:", self._handle_slash_confirm_callback),
            ("cl:", self._handle_clarify_callback),
            ("update_prompt:", self._handle_update_prompt_callback),
        ):
            if data.startswith(prefix):
                await handler(query, data, cb)
                return

    async def _handle_exec_approval_callback(self, query, data: str, cb: Dict[str, Any]) -> None:
        """``ea:<choice>:<approval_id>`` — resolve a pending exec approval."""
        parts = data.split(":", 2)
        if len(parts) != 3:
            return
        choice = parts[1]  # once, session, always, deny
        try:
            approval_id = int(parts[2])
        except (ValueError, IndexError):
            await query.answer(text="Invalid approval data.")
            return
        if not await self._callback_authorized(query, cb, "⛔ You are not authorized to approve commands."):
            return
        session_key = self._approval_state.pop(approval_id, None)
        if not session_key:
            await query.answer(text="This approval has already been resolved.")
            return
        user_display = getattr(query.from_user, "first_name", "User")
        # Resolve FIRST (unblocks the agent thread), render after: a tap landing after the
        # wait timed out (count == 0) must NOT claim "Approved" — the command was already denied.
        try:
            from tools.approval import resolve_gateway_approval
            count = resolve_gateway_approval(session_key, choice)
            logger.info(
                "Telegram button resolved %d approval(s) for session %s (choice=%s, user=%s)",
                count, session_key, choice, user_display,
            )
        except Exception as exc:
            logger.error("Failed to resolve gateway approval from Telegram button: %s", exc)
            count = 0
        if count:
            label_map = {
                "once": "✅ Approved once", "session": "✅ Approved for session",
                "always": "✅ Approved permanently", "deny": "❌ Denied",
            }
            label = label_map.get(choice, "Resolved")
            edit_text = f"{label} by {user_display}"
        else:
            label = "⌛ Approval expired"
            edit_text = (
                f"{label} — no command was waiting. "
                f"It already timed out (and was denied) or was resolved elsewhere."
            )
        await query.answer(text=label)
        try:
            await query.edit_message_text(
                text=self.format_message(edit_text), parse_mode=ParseMode.MARKDOWN_V2, reply_markup=None
            )
        except Exception:
            pass  # non-fatal if edit fails
        # Typing was paused when the approval was sent (gateway/run.py); the text /approve
        # and /deny paths resume it too — without this, typing stays paused for the turn.
        if count and cb["chat_id"] is not None:
            self.resume_typing_for_chat(str(cb["chat_id"]))

    async def _handle_slash_confirm_callback(self, query, data: str, cb: Dict[str, Any]) -> None:
        """``sc:<choice>:<confirm_id>`` — resolve a slash-command confirmation."""
        parts = data.split(":", 2)
        if len(parts) != 3:
            return
        choice = parts[1]  # once, always, cancel
        confirm_id = parts[2]
        if not await self._callback_authorized(query, cb, "⛔ You are not authorized to answer this prompt."):
            return
        session_key = self._slash_confirm_state.pop(confirm_id, None)
        if not session_key:
            await query.answer(text="This prompt has already been resolved.")
            return
        label_map = {"once": "✅ Approved once", "always": "🔒 Always approve", "cancel": "❌ Cancelled"}
        user_display = getattr(query.from_user, "first_name", "User")
        label = label_map.get(choice, "Resolved")
        await query.answer(text=label)
        try:
            await query.edit_message_text(
                text=self.format_message(f"{label} by {user_display}"),
                parse_mode=ParseMode.MARKDOWN_V2,
                reply_markup=None,
            )
        except Exception:
            pass
        # The runner stored a handler keyed by session_key; run it and, if it returns a
        # string, send that as a follow-up in the same chat.
        try:
            from tools import slash_confirm as _slash_confirm_mod
            result_text = await _slash_confirm_mod.resolve(session_key, confirm_id, choice)
            if result_text and query.message:
                # Inherit the prompt's topic: forums use message_thread_id; private DM-topic
                # lanes need both the topic id and the prompt reply anchor.
                thread_id = getattr(query.message, "message_thread_id", None)
                chat = getattr(query.message, "chat", None)
                chat_type = getattr(chat, "type", None)
                prompt_message_id = getattr(query.message, "message_id", None)
                send_kwargs: Dict[str, Any] = {
                    "chat_id": int(query.message.chat_id),
                    "text": self.format_message(result_text),
                    "parse_mode": ParseMode.MARKDOWN_V2,
                    **self._link_preview_kwargs(),
                }
                chat_type_value = getattr(chat_type, "value", chat_type)
                is_private_chat = str(chat_type_value).lower() in {
                    "private", str(ChatType.PRIVATE).lower(),
                    str(getattr(ChatType.PRIVATE, "value", ChatType.PRIVATE)).lower(),
                }
                if thread_id is not None and is_private_chat and prompt_message_id is not None:
                    reply_to_id = int(prompt_message_id)
                    send_kwargs["reply_to_message_id"] = reply_to_id
                    send_kwargs.update(self._thread_kwargs_for_send(
                        str(query.message.chat_id), str(thread_id),
                        {"thread_id": str(thread_id), "telegram_dm_topic_reply_fallback": True},
                        reply_to_message_id=reply_to_id, reply_to_mode=self._reply_to_mode,
                    ))
                elif thread_id is not None:
                    send_kwargs.update(self._thread_kwargs_for_send(
                        str(query.message.chat_id), str(thread_id), {"thread_id": str(thread_id)},
                        reply_to_mode=self._reply_to_mode,
                    ))
                await self._send_message_with_thread_fallback(**send_kwargs)
        except Exception as exc:
            logger.error("[%s] slash-confirm callback failed: %s", self.name, exc, exc_info=True)

    async def _handle_clarify_callback(self, query, data: str, cb: Dict[str, Any]) -> None:
        """``cl:<clarify_id>:<idx|other>`` — resolve a clarify prompt or flip to text capture."""
        parts = data.split(":", 2)
        if len(parts) != 3:
            return
        clarify_id = parts[1]
        choice_token = parts[2]
        if not await self._callback_authorized(query, cb, "⛔ You are not authorized to answer this prompt."):
            return
        session_key = self._clarify_state.get(clarify_id)
        if not session_key:
            await query.answer(text="This prompt has already been resolved.")
            return
        user_display = getattr(query.from_user, "first_name", "User")

        if choice_token == "other":
            # Flip to text-capture: the gateway's text-intercept resolves the clarify with the
            # next message in this session. Do NOT pop _clarify_state yet — still needed if
            # the user is slow and the entry gets cleared by something else.
            flipped = False
            try:
                from tools.clarify_gateway import mark_awaiting_text
                flipped = mark_awaiting_text(clarify_id)
            except Exception as exc:
                logger.warning("[%s] mark_awaiting_text failed: %s", self.name, exc)
            if not flipped:
                # Entry evicted / gateway restarted — a typed answer would go nowhere.
                self._clarify_state.pop(clarify_id, None)
                await self._notify_clarify_expired(query, user_display)
                return
            await query.answer(text="✏️ Type your answer in the chat.")
            try:
                await query.edit_message_text(
                    text=f"❓ {query.message.text or ''}\n\n<i>Awaiting typed response from {_html.escape(user_display)}…</i>",
                    parse_mode=ParseMode.HTML,
                    reply_markup=None,
                )
            except Exception:
                pass
            return

        # Numeric choice → resolve immediately with the chosen text
        try:
            idx = int(choice_token)
        except (ValueError, TypeError):
            await query.answer(text="Invalid choice.")
            return
        resolved_text: Optional[str] = None
        try:
            from tools.clarify_gateway import _entries as _clarify_entries  # type: ignore
            entry = _clarify_entries.get(clarify_id)
            if entry and entry.choices and 0 <= idx < len(entry.choices):
                resolved_text = entry.choices[idx]
        except Exception:
            resolved_text = None
        if resolved_text is None:
            # Race (timeout / session reset): entry vanished. Echo the index so the agent
            # at least sees an intentional response rather than nothing.
            resolved_text = f"choice {idx + 1}"
        self._clarify_state.pop(clarify_id, None)
        try:
            from tools.clarify_gateway import resolve_gateway_clarify
            resolved = resolve_gateway_clarify(clarify_id, resolved_text)
        except Exception as exc:
            logger.error("[%s] resolve_gateway_clarify failed: %s", self.name, exc)
            resolved = False
        if resolved:
            await query.answer(text=f"✓ {resolved_text[:60]}")
            try:
                await query.edit_message_text(
                    text=f"❓ {_html.escape(query.message.text or '')}\n\n<b>{_html.escape(user_display)}:</b> {_html.escape(resolved_text)}",
                    parse_mode=ParseMode.HTML,
                    reply_markup=None,
                )
            except Exception:
                pass
            logger.info(
                "Telegram clarify button resolved (id=%s, choice=%r, user=%s)",
                clarify_id, resolved_text, user_display,
            )
        else:
            # Entry evicted / gateway restarted between ask and tap.
            await self._notify_clarify_expired(query, user_display)
            logger.warning(
                "Telegram clarify button: resolve_gateway_clarify returned False (id=%s)", clarify_id
            )

    async def _handle_update_prompt_callback(self, query, data: str, cb: Dict[str, Any]) -> None:
        """``update_prompt:<y|n>`` — forward the answer to the update process."""
        answer = data.split(":", 1)[1]  # "y" or "n"
        if not await self._callback_authorized(query, cb, "⛔ You are not authorized to answer update prompts."):
            return
        await query.answer(text=f"Sent '{answer}' to the update process.")
        label = "Yes" if answer == "y" else "No"
        try:
            await query.edit_message_text(
                text=self.format_message(f"⚕ Update prompt answered: *{label}*"),
                parse_mode=ParseMode.MARKDOWN_V2,
                reply_markup=None,
            )
        except Exception:
            pass  # non-fatal if edit fails
        try:
            from hermes_constants import get_hermes_home
            home = get_hermes_home()
            response_path = home / ".update_response"
            tmp = response_path.with_suffix(".tmp")
            tmp.write_text(answer, encoding="utf-8")
            tmp.replace(response_path)
            logger.info(
                "Telegram update prompt answered '%s' by user %s",
                answer, getattr(query.from_user, "id", "unknown"),
            )
        except Exception as exc:
            logger.error("Failed to write update response from callback: %s", exc)

    # `gt:<verb>` -> (script in ~/.hermes/scripts/gmail-triage/, extra-args, success-label, is_state).
    # The callback `arg` is always the first positional arg. is_state=True = sticky sender-rule
    # change that keeps the keyboard tappable; False = per-email one-shot that strips it on success.
    _GT_VERB_DISPATCH = {
        "send":         ("send-draft.sh",      [],         "✓ sent draft",         False),
        "archive":      ("archive.sh",         [],         "✓ archived",           False),
        "draft":        ("draft-blank.sh",     [],         "✓ drafted reply",      False),
        "spam":         ("spam.sh",            [],         "✓ marked spam",        False),
        "mute":         ("mute-add.sh",        ["email"],  "✓ muted",              True),
        "mute-domain":  ("mute-add.sh",        ["domain"], "✓ muted domain",       True),
        "trust":        ("trusted-ops-add.sh", ["email"],  "✓ trusted",            True),
        "trust-domain": ("trusted-ops-add.sh", ["domain"], "✓ trusted domain",     True),
        "vip":          ("vip-add.sh",         ["email"],  "✓ marked VIP",         True),
        "vip-domain":   ("vip-add.sh",         ["domain"], "✓ marked VIP domain",  True),
    }

    async def _handle_gmail_triage_callback(self, query, data: str, cb: Dict[str, Any]) -> None:
        """Dispatch a gmail-triage inline-button callback (gt:verb:arg)."""
        parts = data.split(":", 2)
        if len(parts) != 3:
            await query.answer(text="Invalid gmail-triage data.")
            return
        verb, arg = parts[1], parts[2]
        if not await self._callback_authorized(query, cb, "⛔ You are not authorized to act on this email."):
            return
        entry = self._GT_VERB_DISPATCH.get(verb)
        if not entry:
            await query.answer(text=f"Unknown verb: {verb}")
            return
        script_name, extra_args, success_label, is_state_verb = entry
        script_path = _Path.home() / ".hermes" / "scripts" / "gmail-triage" / script_name
        if not script_path.exists():
            await query.answer(text=f"❌ {script_name} missing")
            logger.error("[%s] gmail-triage script missing: %s", self.name, script_path)
            return
        cmd = [str(script_path), arg, *extra_args]
        success = False
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            _stdout_bytes, stderr_bytes = await asyncio.wait_for(proc.communicate(), timeout=60)
            if proc.returncode == 0:
                label = success_label
                success = True
                logger.info("[%s] gmail-triage callback ok: verb=%s arg=%s", self.name, verb, arg)
            else:
                stderr_text = stderr_bytes.decode("utf-8", errors="replace").strip()
                last_line = stderr_text.splitlines()[-1] if stderr_text else f"exit {proc.returncode}"
                label = f"❌ {verb} failed: {last_line[:80]}"
                logger.error(
                    "[%s] gmail-triage callback failed: verb=%s arg=%s rc=%s stderr=%s",
                    self.name, verb, arg, proc.returncode, stderr_text,
                )
        except asyncio.TimeoutError:
            label = f"❌ {verb} timed out"
            logger.error("[%s] gmail-triage callback timed out: verb=%s arg=%s", self.name, verb, arg)
        except Exception as exc:
            label = f"❌ {verb} error: {exc}"
            logger.error(
                "[%s] gmail-triage callback exception: verb=%s arg=%s err=%s",
                self.name, verb, arg, exc, exc_info=True,
            )

        await query.answer(text=label)
        if not success:
            return
        user_display = getattr(query.from_user, "first_name", "User")
        original_text = (query.message.text or "") if query.message else ""
        appended = f"{original_text}\n— {label} by {user_display}"
        try:
            if is_state_verb:
                # Sticky state change: KEEP keyboard so further actions can stack on this email.
                await query.edit_message_text(text=appended)
            else:
                # One-shot: strip keyboard so the action can't fire twice.
                await query.edit_message_text(text=appended, reply_markup=None)
        except Exception:
            pass

    def _missing_media_path_error(self, label: str, path: str) -> str:
        """File-not-found error for MEDIA delivery; /workspace-style paths often exist only in the sandbox."""
        error = f"{label} file not found: {path}"
        if path.startswith(("/workspace/", "/output/", "/outputs/")):
            error += (
                " (path may only exist inside the Docker sandbox. "
                "Bind-mount a host directory and emit the host-visible "
                "path in MEDIA: for gateway file delivery.)"
            )
        return error

    def _telegram_media_too_large_note(self, label: str, file_size: Any, max_bytes: int) -> str:
        limit_mb = max(1, max_bytes // (1024 * 1024))
        try:
            size_mb = int(file_size or 0) / (1024 * 1024)
            size_text = f"{size_mb:.1f} MB"
        except (TypeError, ValueError):
            size_text = "unknown size"
        return (
            f"[Telegram {label} skipped: file size {size_text} exceeds the "
            f"{limit_mb} MB limit. Ask the user to send a smaller file.]"
        )

    def _telegram_media_size_allowed(self, source: Any, label: str) -> tuple[bool, Optional[str]]:
        """Validate Telegram media size before downloading into memory."""
        max_bytes = int(getattr(self, "_max_doc_bytes", 20 * 1024 * 1024) or 20 * 1024 * 1024)
        file_size = getattr(source, "file_size", None)
        try:
            size = int(file_size or 0)
        except (TypeError, ValueError):
            size = 0
        if size <= 0:
            return True, None
        if size <= max_bytes:
            return True, None
        return False, self._telegram_media_too_large_note(label, size, max_bytes)

    def _media_send_kwargs(
        self, chat_id: str, reply_to: Optional[str], metadata: Optional[Dict[str, Any]],
    ) -> tuple[Optional[int], Dict[str, Any]]:
        """Return ``(reply_to_id, base_kwargs)`` shared by every native media send."""
        reply_to_id = self._reply_to_message_id_for_send(reply_to, metadata, reply_to_mode=self._reply_to_mode)
        thread_kwargs = self._thread_kwargs_for_send(
            chat_id, self._metadata_thread_id(metadata), metadata,
            reply_to_message_id=reply_to_id, reply_to_mode=self._reply_to_mode,
        )
        return reply_to_id, {
            "chat_id": normalize_telegram_chat_id(chat_id),
            "reply_to_message_id": reply_to_id,
            "read_timeout": _MEDIA_SEND_READ_TIMEOUT,
            **thread_kwargs,
            **self._notification_kwargs(metadata),
        }

    async def _send_media(
        self, send_fn: Any, chat_id: str, reply_to: Optional[str], metadata: Optional[Dict[str, Any]],
        media_label: str, reset_media: Optional[Any] = None, **media_kwargs: Any,
    ) -> Any:
        """Send one native media payload with thread routing + DM-topic anchor retry."""
        reply_to_id, kwargs = self._media_send_kwargs(chat_id, reply_to, metadata)
        return await self._send_with_dm_topic_reply_anchor_retry(
            send_fn, {**kwargs, **media_kwargs}, metadata, reply_to_id, media_label, reset_media=reset_media,
        )

    async def send_voice(
        self, chat_id: str, audio_path: str, caption: Optional[str] = None, reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None, **kwargs,
    ) -> SendResult:
        """Send audio as a native Telegram voice message or audio file."""
        if not self._bot:
            return SendResult(success=False, error="Not connected")
        _transcoded_voice_path: Optional[str] = None
        try:
            if not os.path.exists(audio_path):
                return SendResult(success=False, error=self._missing_media_path_error("Audio", audio_path))
            # sendVoice only accepts Ogg/Opus: an explicit voice-bubble request (is_voice) transcodes
            # via ffmpeg; otherwise route by extension (.mp3/.m4a → sendAudio, others → document).
            _voice_ext = os.path.splitext(audio_path)[1].lower()
            if kwargs.get("is_voice") and _voice_ext not in (".ogg", ".opus"):
                from gateway.platforms.base import transcode_to_ogg_opus
                _transcoded_voice_path = await asyncio.to_thread(transcode_to_ogg_opus, audio_path)
                if _transcoded_voice_path:
                    audio_path = _transcoded_voice_path
                else:
                    logger.warning(
                        "[%s] voice transcode unavailable for %s — sending "
                        "original format (install ffmpeg for voice bubbles)",
                        self.name, os.path.basename(audio_path),
                    )
            # Telegram drops duration for long clips (~5 min+, shows 0:00).
            _duration_secs = await asyncio.to_thread(_probe_voice_duration_seconds, audio_path)
            # Auto-TTS captions carry the agent's markdown reply: MarkdownV2 when it fits the
            # 1024-char cap, plain text fallback when it overflows or Bot API rejects it.
            _caption_variants: List[tuple] = []
            if caption:
                try:
                    _formatted_caption = self.format_message(caption)
                    if utf16_len(_formatted_caption) <= 1024:
                        _caption_variants.append((_formatted_caption, ParseMode.MARKDOWN_V2))
                except Exception:
                    logger.debug(
                        "[%s] voice caption MarkdownV2 formatting failed; sending plain caption",
                        self.name, exc_info=True,
                    )
                _caption_variants.append((caption[:1024], None))
            else:
                _caption_variants.append((None, None))
            with open(audio_path, "rb") as audio_file:
                ext = os.path.splitext(audio_path)[1].lower()
                if ext in {".ogg", ".opus"}:
                    # Round playable voice bubble.
                    msg = None
                    _last_parse_error: Optional[Exception] = None
                    for _cap_text, _cap_parse_mode in _caption_variants:
                        try:
                            msg = await self._send_media(
                                self._bot.send_voice, chat_id, reply_to, metadata, "voice",
                                reset_media=lambda: audio_file.seek(0), voice=audio_file, caption=_cap_text,
                                parse_mode=_cap_parse_mode, duration=_duration_secs,
                            )
                            break
                        except Exception as _cap_error:
                            # Only retry plain on entity-parse failures; anything else is a real error.
                            if (_cap_parse_mode is not None
                                    and ("parse" in str(_cap_error).lower()
                                         or "entit" in str(_cap_error).lower())):
                                logger.warning(
                                    "[%s] voice caption MarkdownV2 rejected, retrying plain: %s",
                                    self.name, _redact_telegram_error_text(_cap_error),
                                )
                                _last_parse_error = _cap_error
                                audio_file.seek(0)
                                continue
                            raise
                    if msg is None:
                        raise _last_parse_error or RuntimeError(
                            "Telegram send_voice failed for all caption variants"
                        )
                elif ext in {".mp3", ".m4a"}:
                    # Bot API sendAudio only accepts MP3 / M4A.
                    msg = await self._send_media(
                        self._bot.send_audio, chat_id, reply_to, metadata, "audio",
                        reset_media=lambda: audio_file.seek(0), audio=audio_file,
                        caption=caption[:1024] if caption else None, duration=_duration_secs,
                    )
                else:
                    # Formats Telegram can't play natively (.wav, .flac, ...).
                    return await self.send_document(
                        chat_id=chat_id, file_path=audio_path, caption=caption,
                        reply_to=reply_to, metadata=metadata,
                    )
            return SendResult(success=True, message_id=str(msg.message_id))
        except Exception as e:
            logger.error(
                "[%s] Failed to send Telegram voice/audio, falling back to base adapter: %s",
                self.name, _redact_telegram_error_text(e), exc_info=True,
            )
            return await super().send_voice(chat_id, audio_path, caption, reply_to, metadata=metadata)
        finally:
            if _transcoded_voice_path:
                try:
                    os.unlink(_transcoded_voice_path)
                except OSError:
                    pass

    async def send_multiple_images(
        self, chat_id: str, images: List[tuple], metadata: Optional[Dict[str, Any]] = None,
        human_delay: float = 0.0,
    ) -> None:
        """Send images as Telegram albums (``send_media_group``, 10 per chunk).

        Animated GIFs can't join a media group (need ``send_animation``), so they go via the
        base per-image path; a failed chunk also falls back to the base per-image loop.
        """
        if not self._bot:
            return
        if not images:
            return
        try:
            from telegram import InputMediaPhoto
        except Exception as exc:  # pragma: no cover - missing SDK
            logger.warning(
                "[%s] InputMediaPhoto unavailable, falling back to per-image send: %s", self.name, exc
            )
            await super().send_multiple_images(chat_id, images, metadata, human_delay)
            return
        animations: List[tuple] = []
        photos: List[tuple] = []
        for image_url, alt_text in images:
            if not image_url.startswith("file://") and self._is_animation_url(image_url):
                animations.append((image_url, alt_text))
            else:
                photos.append((image_url, alt_text))
        if animations:
            await super().send_multiple_images(chat_id, animations, metadata, human_delay=human_delay)
        if not photos:
            return
        from urllib.parse import unquote as _unquote
        CHUNK = 10  # Telegram's album limit
        chunks = [photos[i:i + CHUNK] for i in range(0, len(photos), CHUNK)]
        for chunk_idx, chunk in enumerate(chunks):
            if human_delay > 0 and chunk_idx > 0:
                await asyncio.sleep(human_delay)
            media: List[Any] = []
            opened_files: List[Any] = []
            try:
                for image_url, alt_text in chunk:
                    caption = alt_text[:1024] if alt_text else None
                    if image_url.startswith("file://"):
                        local_path = _unquote(image_url[7:])
                        if not os.path.exists(local_path):
                            logger.warning(
                                "[%s] Skipping missing image in media group: %s", self.name, local_path
                            )
                            continue
                        fh = open(local_path, "rb")
                        opened_files.append(fh)
                        media.append(InputMediaPhoto(media=fh, caption=caption))
                    else:
                        media.append(InputMediaPhoto(media=image_url, caption=caption))
                if not media:
                    continue
                logger.info(
                    "[%s] Sending media group of %d photo(s) (chunk %d/%d)",
                    self.name, len(media), chunk_idx + 1, len(chunks),
                )
                reply_to_id, send_kwargs = self._media_send_kwargs(chat_id, None, metadata)

                def _reset_opened_files() -> None:
                    for fh in opened_files:
                        try:
                            fh.seek(0)
                        except Exception:
                            pass

                await self._send_with_dm_topic_reply_anchor_retry(
                    self._bot.send_media_group, {**send_kwargs, "media": media}, metadata, reply_to_id,
                    "media group", reset_media=_reset_opened_files,
                )
            except Exception as e:
                logger.warning(
                    "[%s] send_media_group failed (chunk %d/%d), falling back to per-image: %s",
                    self.name, chunk_idx + 1, len(chunks), _redact_telegram_error_text(e),
                    exc_info=True,
                )
                await super().send_multiple_images(chat_id, chunk, metadata, human_delay=human_delay)
            finally:
                for fh in opened_files:
                    try:
                        fh.close()
                    except Exception:
                        pass

    async def send_image_file(
        self, chat_id: str, image_path: str, caption: Optional[str] = None, reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None, **kwargs,
    ) -> SendResult:
        """Send a local image file natively as a Telegram photo."""
        if not self._bot:
            return SendResult(success=False, error="Not connected")
        try:
            if not os.path.exists(image_path):
                return SendResult(success=False, error=self._missing_media_path_error("Image", image_path))
            with open(image_path, "rb") as image_file:
                msg = await self._send_media(
                    self._bot.send_photo, chat_id, reply_to, metadata, "photo",
                    reset_media=lambda: image_file.seek(0), photo=image_file,
                    caption=caption[:1024] if caption else None,
                )
            return SendResult(success=True, message_id=str(msg.message_id))
        except Exception as e:
            error_str = str(e)
            # Dimension errors are expected for valid images Telegram refuses as photos
            # (screenshots, extreme aspect ratios) → INFO; anything else → WARNING.
            is_dim_error = "Photo_invalid_dimensions" in error_str or "PHOTO_INVALID_DIMENSIONS" in error_str
            if is_dim_error:
                logger.info(
                    "[%s] Image dimensions exceed Telegram photo limits, sending as document: %s",
                    self.name, image_path,
                )
            else:
                logger.warning(
                    "[%s] Failed to send Telegram local image as photo, trying document fallback: %s",
                    self.name, _redact_telegram_error_text(e), exc_info=True,
                )
            # Document has no dimension limit (50MB only); if even that fails, base adapter text.
            try:
                return await self.send_document(
                    chat_id=chat_id, file_path=image_path, caption=caption,
                    file_name=os.path.basename(image_path), reply_to=reply_to, metadata=metadata,
                )
            except Exception as doc_err:
                logger.error(
                    "[%s] Failed to send Telegram local image as document, falling back to base adapter: %s",
                    self.name, doc_err, exc_info=True,
                )
                return await super().send_image_file(chat_id, image_path, caption, reply_to, metadata=metadata)

    async def send_document(
        self, chat_id: str, file_path: str, caption: Optional[str] = None, file_name: Optional[str] = None,
        reply_to: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None, **kwargs,
    ) -> SendResult:
        """Send a document/file natively as a Telegram file attachment."""
        if not self._bot:
            return SendResult(success=False, error="Not connected")
        try:
            if not os.path.exists(file_path):
                return SendResult(success=False, error=self._missing_media_path_error("File", file_path))
            with open(file_path, "rb") as f:
                msg = await self._send_media(
                    self._bot.send_document, chat_id, reply_to, metadata, "document",
                    reset_media=lambda: f.seek(0), document=f,
                    filename=file_name or os.path.basename(file_path),
                    caption=caption[:1024] if caption else None,
                )
            return SendResult(success=True, message_id=str(msg.message_id))
        except Exception as e:
            logger.warning("[%s] Failed to send document: %s", self.name, _redact_telegram_error_text(e))
            return await super().send_document(chat_id, file_path, caption, file_name, reply_to, metadata=metadata)

    async def send_video(
        self, chat_id: str, video_path: str, caption: Optional[str] = None, reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None, **kwargs,
    ) -> SendResult:
        """Send a video natively as a Telegram video message."""
        if not self._bot:
            return SendResult(success=False, error="Not connected")
        try:
            if not os.path.exists(video_path):
                return SendResult(success=False, error=self._missing_media_path_error("Video", video_path))
            with open(video_path, "rb") as f:
                msg = await self._send_media(
                    self._bot.send_video, chat_id, reply_to, metadata, "video",
                    reset_media=lambda: f.seek(0), video=f, caption=caption[:1024] if caption else None,
                )
            return SendResult(success=True, message_id=str(msg.message_id))
        except Exception as e:
            logger.warning("[%s] Failed to send video: %s", self.name, _redact_telegram_error_text(e))
            return await super().send_video(chat_id, video_path, caption, reply_to, metadata=metadata)

    async def send_image(
        self, chat_id: str, image_url: str, caption: Optional[str] = None, reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send a URL image as a Telegram photo: URL send (<5MB) → download+upload (≤10MB) → base text."""
        if not self._bot:
            return SendResult(success=False, error="Not connected")
        from tools.url_safety import is_safe_url
        if not is_safe_url(image_url):
            logger.warning("[%s] Blocked unsafe image URL (SSRF protection)", self.name)
            return await super().send_image(chat_id, image_url, caption, reply_to, metadata=metadata)

        photo_caption = caption[:1024] if caption else None
        try:
            msg = await self._send_media(
                self._bot.send_photo, chat_id, reply_to, metadata, "URL photo",
                photo=image_url, caption=photo_caption,
            )
            return SendResult(success=True, message_id=str(msg.message_id))
        except Exception as e:
            logger.warning(
                "[%s] URL-based send_photo failed, trying file upload: %s",
                self.name, _redact_telegram_error_text(e), exc_info=True,
            )
            try:
                from gateway.platforms.base import _ssrf_redirect_guard
                from tools.url_safety import create_ssrf_safe_async_client
                async with create_ssrf_safe_async_client(
                    timeout=30.0, event_hooks={"response": [_ssrf_redirect_guard]}
                ) as client:
                    resp = await client.get(image_url)
                    resp.raise_for_status()
                    image_data = resp.content
                msg = await self._send_media(
                    self._bot.send_photo, chat_id, reply_to, metadata, "uploaded photo",
                    photo=image_data, caption=photo_caption,
                )
                return SendResult(success=True, message_id=str(msg.message_id))
            except Exception as e2:
                logger.error("[%s] File upload send_photo also failed: %s", self.name, e2, exc_info=True)
                return await super().send_image(chat_id, image_url, caption, reply_to, metadata=metadata)

    async def send_animation(
        self, chat_id: str, animation_url: str, caption: Optional[str] = None, reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send an animated GIF natively as a Telegram animation (auto-plays inline)."""
        if not self._bot:
            return SendResult(success=False, error="Not connected")
        try:
            msg = await self._send_media(
                self._bot.send_animation, chat_id, reply_to, metadata, "animation",
                animation=animation_url, caption=caption[:1024] if caption else None,
            )
            return SendResult(success=True, message_id=str(msg.message_id))
        except Exception as e:
            logger.error(
                "[%s] Failed to send Telegram animation, falling back to photo: %s",
                self.name, _redact_telegram_error_text(e), exc_info=True,
            )
            return await self.send_image(chat_id, animation_url, caption, reply_to, metadata=metadata)

    @staticmethod
    def _is_transient_typing_error(exc: Exception) -> bool:
        """Return True for Telegram typing errors worth cooling down."""
        retry_after = getattr(exc, "retry_after", None)
        if retry_after is not None:
            return True
        status_code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
        if isinstance(status_code, int) and (status_code == 429 or status_code >= 500):
            return True
        text = str(exc).lower()
        if any(marker in text for marker in ("too many requests", "rate limit", "timed out", "timeout", "temporar")):
            return True
        return isinstance(exc, (OSError, TimeoutError, ConnectionError, asyncio.TimeoutError))

    def _record_typing_cooldown(self, chat_id: str, exc: Exception) -> None:
        """Suppress Telegram typing refreshes for this chat after transient failures."""
        if not hasattr(self, "_telegram_typing_cooldown_until"):
            self._telegram_typing_cooldown_until = {}
        loop = asyncio.get_running_loop()
        retry_after = getattr(exc, "retry_after", None)
        try:
            delay = float(retry_after) if retry_after is not None else self._telegram_typing_cooldown_seconds
        except (TypeError, ValueError):
            delay = self._telegram_typing_cooldown_seconds
        delay = max(1.0, min(delay, 300.0))
        self._telegram_typing_cooldown_until[str(chat_id)] = loop.time() + delay

    def _typing_in_cooldown(self, chat_id: str) -> bool:
        if not hasattr(self, "_telegram_typing_cooldown_until"):
            self._telegram_typing_cooldown_until = {}
            self._telegram_typing_cooldown_seconds = 30.0
        until = self._telegram_typing_cooldown_until.get(str(chat_id))
        if until is None:
            return False
        if asyncio.get_running_loop().time() < until:
            return True
        self._telegram_typing_cooldown_until.pop(str(chat_id), None)
        return False

    async def send_typing(self, chat_id: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Send typing indicator."""
        if not self._bot or self._typing_in_cooldown(chat_id):
            return
        _is_dm_topic: bool = False
        message_thread_id: Optional[int] = None
        try:
            _typing_thread = self._metadata_thread_id(metadata)
            _is_dm_topic = bool(metadata and metadata.get("telegram_dm_topic_reply_fallback"))
            message_thread_id = self._message_thread_id_for_typing(_typing_thread)
            await self._bot.send_chat_action(
                chat_id=normalize_telegram_chat_id(chat_id), action="typing",
                message_thread_id=message_thread_id,
            )
            self._telegram_typing_cooldown_until.pop(str(chat_id), None)
        except Exception as e:
            # DM topic lanes: Telegram may reject message_thread_id — retry without it so the
            # indicator at least appears in the main DM view.
            if _is_dm_topic and message_thread_id is not None:
                try:
                    await self._bot.send_chat_action(
                        chat_id=normalize_telegram_chat_id(chat_id), action="typing"
                    )
                    self._telegram_typing_cooldown_until.pop(str(chat_id), None)
                    return
                except Exception as fallback_exc:
                    if self._is_transient_typing_error(fallback_exc):
                        self._record_typing_cooldown(chat_id, fallback_exc)
            elif self._is_transient_typing_error(e):
                self._record_typing_cooldown(chat_id, e)
            # Typing failures are non-fatal; debug only.
            logger.debug(
                "[%s] Failed to send Telegram typing indicator: %s",
                self.name, _redact_telegram_error_text(e), exc_info=True,
            )

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Get information about a Telegram chat."""
        if not self._bot:
            return {"name": "Unknown", "type": "dm"}
        try:
            chat = await self._bot.get_chat(normalize_telegram_chat_id(chat_id))
            chat_type = "dm"
            if chat.type == ChatType.GROUP:
                chat_type = "group"
            elif chat.type == ChatType.SUPERGROUP:
                chat_type = "group"
                if chat.is_forum:
                    chat_type = "forum"
            elif chat.type == ChatType.CHANNEL:
                chat_type = "channel"
            return {
                "name": chat.title or chat.full_name or str(chat_id),
                "type": chat_type,
                "username": chat.username,
                "is_forum": getattr(chat, "is_forum", False),
            }
        except Exception as e:
            logger.error(
                "[%s] Failed to get Telegram chat info for %s: %s",
                self.name, chat_id, _redact_telegram_error_text(e), exc_info=True,
            )
            return {"name": str(chat_id), "type": "dm", "error": str(e)}

    def format_message(self, content: str) -> str:
        """Convert standard markdown to Telegram MarkdownV2.

        Code blocks/inline code are stashed behind placeholders first so they're never
        modified; markdown constructs become MarkdownV2 syntax; everything else is escaped.
        """
        if not content:
            return content
        placeholders: dict = {}
        counter = [0]

        def _ph(value: str) -> str:
            """Stash *value* behind a placeholder token that survives escaping."""
            key = f"\x00PH{counter[0]}\x00"
            counter[0] += 1
            placeholders[key] = value
            return key

        text = content
        # 0) GFM pipe tables → Telegram-friendly row groups, before the MarkdownV2 conversions.
        text = _wrap_markdown_tables(text)

        # 1) Protect fenced code blocks; per MarkdownV2 spec \ and ` inside pre/code must be escaped.
        def _protect_fenced(m):
            raw = m.group(0)
            # Split off opening ``` (with optional language) and closing ```
            open_end = raw.index('\n') + 1 if '\n' in raw[3:] else 3
            opening = raw[:open_end]
            body_and_close = raw[open_end:]
            body = body_and_close[:-3]
            body = body.replace('\\', '\\\\').replace('`', '\\`')
            return _ph(opening + body + '```')

        text = re.sub(r'(```(?:[^\n]*\n)?[\s\S]*?```)', _protect_fenced, text)

        # 2) Protect inline code; escape \ inside it per MarkdownV2 spec.
        text = re.sub(r'(`[^`]+`)', lambda m: _ph(m.group(0).replace('\\', '\\\\')), text)

        # 3) Links: escape display text; inside the URL only ')' and '\' need escaping.
        def _convert_link(m):
            display = _escape_mdv2(m.group(1))
            url = m.group(2).replace('\\', '\\\\').replace(')', '\\)')
            return _ph(f'[{display}]({url})')

        text = re.sub(r'\[([^\]]+)\]\(([^()]*(?:\([^()]*\)[^()]*)*)\)', _convert_link, text)

        # 4) Headers (## Title) → bold *Title*, stripping redundant ** inside the header
        def _convert_header(m):
            inner = m.group(1).strip()
            inner = re.sub(r'\*\*(.+?)\*\*', r'\1', inner)
            return _ph(f'*{_escape_mdv2(inner)}*')

        text = re.sub(r'^#{1,6}\s+(.+)$', _convert_header, text, flags=re.MULTILINE)
        # 5) Bold: **text** → *text*
        text = re.sub(r'\*\*(.+?)\*\*', lambda m: _ph(f'*{_escape_mdv2(m.group(1))}*'), text)
        # 6) Italic: *text* → _text_. [^*\n]+ keeps matches on one line, or * bullet lists corrupt.
        text = re.sub(r'\*([^*\n]+)\*', lambda m: _ph(f'_{_escape_mdv2(m.group(1))}_'), text)
        # 7) Strikethrough: ~~text~~ → ~text~
        text = re.sub(r'~~(.+?)~~', lambda m: _ph(f'~{_escape_mdv2(m.group(1))}~'), text)
        # 8) Spoiler: ||text|| kept as-is (protect from | escaping)
        text = re.sub(r'\|\|(.+?)\|\|', lambda m: _ph(f'||{_escape_mdv2(m.group(1))}||'), text)

        # 9) Blockquotes: protect leading > from escaping. Also expandable quotes
        #    (**> starts, trailing || ends — that || must stay unescaped).
        def _convert_blockquote(m):
            prefix = m.group(1)  # >, >>, >>>, **>, or **>> etc.
            content = m.group(2)
            if prefix.startswith('**') and content.endswith('||'):
                return _ph(f'{prefix} {_escape_mdv2(content[:-2])}||')
            return _ph(f'{prefix} {_escape_mdv2(content)}')

        text = re.sub(r'^((?:\*\*)?>{1,3}) (.+)$', _convert_blockquote, text, flags=re.MULTILINE)
        # 10) Escape remaining special characters in plain text
        text = _escape_mdv2(text)
        # 11) Restore placeholders in reverse insertion order so nested placeholders resolve.
        for key in reversed(list(placeholders.keys())):
            text = text.replace(key, placeholders[key])
        # 12) Safety net: escape bare ( ) { } that slipped through, but never inside ``` or ` spans.
        _code_split = re.split(r'(```[\s\S]*?```|`[^`]+`)', text)
        _safe_parts = []
        for _idx, _seg in enumerate(_code_split):
            if _idx % 2 == 1:
                _safe_parts.append(_seg)  # inside code — untouched
            else:
                def _esc_bare(m, _seg=_seg):
                    s = m.start()
                    ch = m.group(0)
                    if s > 0 and _seg[s - 1] == '\\':  # already escaped
                        return ch
                    if ch == '(' and s > 0 and _seg[s - 1] == ']':  # opens a link [text](url)
                        return ch
                    if ch == ')':  # closes a link URL? walk back matching depth
                        before = _seg[:s]
                        if '](http' in before or '](' in before:
                            depth = 0
                            for j in range(s - 1, max(s - 2000, -1), -1):
                                if _seg[j] == '(':
                                    depth -= 1
                                    if depth < 0:
                                        if j > 0 and _seg[j - 1] == ']':
                                            return ch
                                        break
                                elif _seg[j] == ')':
                                    depth += 1
                    return '\\' + ch
                _safe_parts.append(re.sub(r'[(){}]', _esc_bare, _seg))
        text = ''.join(_safe_parts)
        return text

    # ── Group mention gating ──────────────────────────────────────────────

    def _extra_bool(self, key: str, env_name: str, default: str, *fallback_keys: str) -> bool:
        """Boolean gate from ``config.extra[key]`` (then ``fallback_keys``), else env var."""
        configured = self.config.extra.get(key)
        for alt in fallback_keys:
            if configured is None:
                configured = self.config.extra.get(alt)
        if configured is not None:
            if isinstance(configured, str):
                return configured.lower() in {"true", "1", "yes", "on"}
            return bool(configured)
        return os.getenv(env_name, default).lower() in {"true", "1", "yes", "on"}

    def _extra_str_set(self, key: str, env_name: str) -> set[str]:
        """Comma/list allowlist from ``config.extra[key]``, else the profile-scoped env var."""
        raw = self.config.extra.get(key)
        if raw is None:
            raw = _scoped_gate_env(env_name)
        if isinstance(raw, list):
            return {str(part).strip() for part in raw if str(part).strip()}
        return {part.strip() for part in str(raw).split(",") if part.strip()}

    def _telegram_require_mention(self) -> bool:
        """Return whether group chats should require an explicit bot trigger."""
        return self._extra_bool("require_mention", "TELEGRAM_REQUIRE_MENTION", "false")

    def _telegram_observe_unmentioned_group_messages(self) -> bool:
        """Store skipped unmentioned group messages as context (with ``require_mention``:
        observe chatter in the transcript, dispatch only when addressed)."""
        return self._extra_bool(
            "observe_unmentioned_group_messages", "TELEGRAM_OBSERVE_UNMENTIONED_GROUP_MESSAGES", "false",
            "ingest_unmentioned_group_messages",
        )

    def _telegram_guest_mode(self) -> bool:
        """Return whether non-allowlisted groups may trigger via direct @mention."""
        return self._extra_bool("guest_mode", "TELEGRAM_GUEST_MODE", "false")

    def _telegram_exclusive_bot_mentions(self) -> bool:
        """Return whether explicit @...bot mentions exclusively route group messages."""
        return self._extra_bool("exclusive_bot_mentions", "TELEGRAM_EXCLUSIVE_BOT_MENTIONS", "true")

    def _telegram_free_response_chats(self) -> set[str]:
        return self._extra_str_set("free_response_chats", "TELEGRAM_FREE_RESPONSE_CHATS")

    def _telegram_free_response_topics(self) -> set[str]:
        """Topic-level free-response entries as ``<chat_id>:<thread_id>`` (General topic = ``1``)."""
        return self._extra_str_set("free_response_topics", "TELEGRAM_FREE_RESPONSE_TOPICS")

    def _telegram_is_free_response_topic(self, message: Message) -> bool:
        """True when the message's chat/topic pair is in ``free_response_topics``."""
        topics = self._telegram_free_response_topics()
        if not topics:
            return False
        chat_id = str(getattr(getattr(message, "chat", None), "id", ""))
        if not chat_id:
            return False
        thread_id = self._effective_message_thread_id(message)
        topic_id = str(thread_id) if thread_id is not None else self._GENERAL_TOPIC_THREAD_ID
        return f"{chat_id}:{topic_id}" in topics

    def _telegram_allowed_chats(self) -> set[str]:
        """Group chat IDs the bot responds in (non-empty: others need ``guest_mode`` + @mention;
        DMs never filtered; empty = no restriction)."""
        return self._extra_str_set("allowed_chats", "TELEGRAM_ALLOWED_CHATS")

    def _telegram_group_allowed_chats(self) -> set[str]:
        """Return Telegram chats authorized at group scope."""
        return self._extra_str_set("group_allowed_chats", "TELEGRAM_GROUP_ALLOWED_CHATS")

    def _telegram_observe_allowed_chats(self) -> set[str]:
        """Chats where observed group context may use a shared source: ``group_allowed_chats``
        (auth allowlist for user-less sources) ∩ ``allowed_chats`` (response gate) when set."""
        group_allowed = self._telegram_group_allowed_chats()
        if not group_allowed:
            return set()
        response_allowed = self._telegram_allowed_chats()
        if response_allowed:
            return group_allowed & response_allowed
        return group_allowed

    def _telegram_allowed_topics(self) -> set[str]:
        """Forum topic IDs this bot handles (non-empty: other topics ignored; DMs
        never filtered; missing ``message_thread_id`` == General topic ``1``)."""
        return self._extra_str_set("allowed_topics", "TELEGRAM_ALLOWED_TOPICS")

    def _telegram_ignored_threads(self) -> set[int]:
        raw = self.config.extra.get("ignored_threads")
        if raw is None:
            raw = _scoped_gate_env("TELEGRAM_IGNORED_THREADS")
        values = raw if isinstance(raw, list) else str(raw).split(",")
        ignored: set[int] = set()
        for value in values:
            text = str(value).strip()
            if not text:
                continue
            try:
                ignored.add(int(text))
            except (TypeError, ValueError):
                logger.warning("[%s] Ignoring invalid Telegram thread id: %r", self.name, value)
        return ignored

    def _compile_mention_patterns(self) -> List[re.Pattern]:
        """Compile optional regex wake-word patterns for group triggers."""
        patterns = self.config.extra.get("mention_patterns")
        if patterns is None:
            raw = os.getenv("TELEGRAM_MENTION_PATTERNS", "").strip()
            if raw:
                try:
                    loaded = json.loads(raw)
                except Exception:
                    loaded = [part.strip() for part in raw.splitlines() if part.strip()]
                    if not loaded:
                        loaded = [part.strip() for part in raw.split(",") if part.strip()]
                patterns = loaded
        if patterns is None:
            # Return before touching ``self.name``: tests build bare adapters via object.__new__.
            return []
        return compile_mention_patterns(
            patterns, log_prefix=self.name, platform_label="telegram",
            display_label="Telegram", logger_=logger,
        )

    def _is_group_chat(self, message: Message) -> bool:
        chat = getattr(message, "chat", None)
        if not chat:
            return False
        chat_type = str(getattr(chat, "type", "")).split(".")[-1].lower()
        return chat_type in {"group", "supergroup"}

    @classmethod
    def _effective_message_thread_id(cls, message: Message) -> Optional[str]:
        """Return the routable thread id for a Telegram message.

        Forum General-topic messages arrive with ``message_thread_id=None`` but Telegram
        addresses that topic as id ``1``; conversely, plain group/DM replies carry a reply-UI
        anchor in ``message_thread_id`` that is NOT a routing id. Gating, skill binding and
        outbound routing must all agree on this one normalized value.
        """
        chat = getattr(message, "chat", None)
        chat_type = str(getattr(chat, "type", "")).split(".")[-1].lower() if chat else ""
        raw = getattr(message, "message_thread_id", None)
        is_topic_message = bool(getattr(message, "is_topic_message", False))
        is_forum_group = chat_type in ("group", "supergroup") and getattr(chat, "is_forum", False) is True
        if raw is not None:
            if is_forum_group or (chat_type in ("group", "supergroup") and is_topic_message):
                return str(raw)
            if chat_type == "private" and is_topic_message:
                return str(raw)
            return None
        if is_forum_group:
            return cls._GENERAL_TOPIC_THREAD_ID
        return None

    # Decides only whether a FOREIGN @handle is bot-shaped; our own handle is matched by
    # identity, never shape (collectible/Fragment bot usernames need not end in "bot").
    _FOREIGN_BOT_HANDLE_RE = re.compile(r"[a-z0-9_]{2,29}bot", re.IGNORECASE)
    # How long an observed identity is trusted before the heartbeat re-checks.
    _BOT_IDENTITY_TTL_SECONDS = 300.0

    def _current_bot_username(self) -> str:
        """Return this bot's live @username (lowercased, no leading ``@``).

        Prefers the last observed handle over PTB's ``get_me()`` cache: ``Bot.username`` is
        only refreshed by ``get_me()``, so after a BotFather rename it keeps the stale handle
        and every mention comparison silently stops matching.
        """
        observed = getattr(self, "_bot_username_observed", None)
        if observed:
            return observed
        return (getattr(self._bot, "username", None) or "").lstrip("@").lower()

    def _note_bot_username(self, username: Optional[str]) -> None:
        """Record the bot's current @username, logging real renames."""
        handle = (username or "").lstrip("@").lower()
        if not handle:
            return
        previous = getattr(self, "_bot_username_observed", None)
        if previous == handle:
            return
        self._bot_username_observed = handle
        self._bot_identity_checked_at = time.monotonic()
        if previous:
            logger.info(
                "[%s] Telegram bot username changed: @%s -> @%s (mention routing now follows the new handle)",
                self.name, previous, handle,
            )

    def _observe_bot_identity_from_message(self, message: Message) -> None:
        """Learn our own handle from a message Telegram says we authored.

        Telegram stamps the *current* username on our own messages and on ``reply_to_message``
        when a user replies to us, so renames are observable without getMe. Only trusted when
        the user id matches this bot, so a foreign handle can never be adopted.
        """
        bot_id = getattr(self._bot, "id", None)
        if bot_id is None:
            return
        for candidate in (
            getattr(message, "from_user", None),
            getattr(getattr(message, "reply_to_message", None), "from_user", None),
        ):
            if candidate is None:
                continue
            if getattr(candidate, "id", None) != bot_id:
                continue
            self._note_bot_username(getattr(candidate, "username", None))

    def _bot_identity_is_fresh(self) -> bool:
        """True when identity was re-read within the TTL.

        ``None`` (never checked) is always stale. Do not fold it into ``0.0``: monotonic
        clocks have an arbitrary epoch that can be smaller than the TTL on a fresh host.
        """
        checked_at = getattr(self, "_bot_identity_checked_at", None)
        if checked_at is None:
            return False
        return (time.monotonic() - checked_at) < self._BOT_IDENTITY_TTL_SECONDS

    async def _refresh_bot_identity(self, *, force: bool = False) -> None:
        """Re-read bot identity from Telegram when the cache may be stale.

        ``get_me()`` rewrites PTB's ``Bot._bot_user`` in place, repairing every consumer of
        ``self._bot.username``. Best-effort: a failed probe keeps the last known handle.
        """
        bot = self._bot
        if bot is None or not callable(getattr(bot, "get_me", None)):
            return
        if not force and self._bot_identity_is_fresh():
            return
        try:
            me = await asyncio.wait_for(bot.get_me(), self._BOT_IDENTITY_PROBE_TIMEOUT)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.debug(
                "[%s] Telegram identity refresh failed (keeping @%s): %s",
                self.name, self._current_bot_username() or "unknown", exc,
            )
            return
        self._bot_identity_checked_at = time.monotonic()
        self._note_bot_username(getattr(me, "username", None))

    _BOT_IDENTITY_PROBE_TIMEOUT = 15.0

    def _is_reply_to_bot(self, message: Message) -> bool:
        if not self._bot or not getattr(message, "reply_to_message", None):
            return False
        reply_user = getattr(message.reply_to_message, "from_user", None)
        return bool(reply_user and getattr(reply_user, "id", None) == getattr(self._bot, "id", None))

    @classmethod
    def _extract_bot_mention_usernames(cls, message: Message, self_username: str = "") -> set[str]:
        """Extract explicit bot usernames mentioned in text/captions.

        Foreign handles count only when bot-shaped (``...bot``) so human @handles never act
        as routing hints; ``self_username`` opts our OWN handle in regardless of shape
        (collectible usernames like @jarvis). Entity mentions are authoritative; the raw-text
        fallback is deliberately narrow (no emails / arbitrary substrings).
        """
        mentioned_bot_usernames: set[str] = set()
        own = (self_username or "").lstrip("@").lower()

        def _is_bot_handle(handle: str) -> bool:
            if not handle:
                return False
            if own and handle == own:
                return True
            return bool(cls._FOREIGN_BOT_HANDLE_RE.fullmatch(handle))

        def _iter_sources():
            yield getattr(message, "text", None) or "", getattr(message, "entities", None) or []
            yield getattr(message, "caption", None) or "", getattr(message, "caption_entities", None) or []
        for source_text, entities in _iter_sources():
            for entity in entities:
                entity_type = str(getattr(entity, "type", "")).split(".")[-1].lower()
                if entity_type not in {"mention", "bot_command"}:
                    continue
                offset = int(getattr(entity, "offset", -1))
                length = int(getattr(entity, "length", 0))
                if offset < 0 or length <= 0:
                    continue
                entity_text = TelegramAdapter._telegram_entity_text(source_text, offset, length).strip()
                if entity_type == "mention":
                    handle = entity_text.lstrip("@").lower()
                    if _is_bot_handle(handle):
                        mentioned_bot_usernames.add(handle)
                    continue

                # /cmd@botname is one bot_command entity (no separate mention); its suffix
                # is an explicit bot address for exclusive multi-bot routing.
                at_index = entity_text.find("@")
                if at_index < 0:
                    continue
                command_target = entity_text[at_index + 1:].strip().lower()
                if _is_bot_handle(command_target):
                    mentioned_bot_usernames.add(command_target)

        # Entity-less fallback only: if Telegram supplied entities, trust them and do not
        # regex-rescue URL/code spans the server did not mark as mentions.
        for raw_text, entities in _iter_sources():
            if not raw_text or entities:
                continue
            for match in re.finditer(r"(?i)(?<![A-Za-z0-9_`/])@([A-Za-z0-9_]{2,31})\b", raw_text):
                handle = match.group(1).lower()
                if _is_bot_handle(handle):
                    mentioned_bot_usernames.add(handle)

        return mentioned_bot_usernames

    @staticmethod
    def _telegram_entity_text(source_text: str, offset: int, length: int) -> str:
        """Return a Telegram entity span using UTF-16 code-unit offsets."""
        if offset < 0 or length <= 0:
            return ""
        try:
            raw = source_text.encode("utf-16-le")
            start = offset * 2
            end = (offset + length) * 2
            return raw[start:end].decode("utf-16-le")
        except UnicodeDecodeError:
            return ""

    def _message_mentions_bot(self, message: Message) -> bool:
        if not self._bot:
            return False
        bot_username = self._current_bot_username()
        bot_id = getattr(self._bot, "id", None)
        expected = f"@{bot_username}" if bot_username else None

        def _iter_sources():
            yield getattr(message, "text", None) or "", getattr(message, "entities", None) or []
            yield getattr(message, "caption", None) or "", getattr(message, "caption_entities", None) or []

        # Server-side MessageEntity values are authoritative (mention=@username,
        # text_mention=user without public handle): raw substrings like
        # "foo@hermes_bot.example" or handles inside URLs/code are not mentions.
        for source_text, entities in _iter_sources():
            for entity in entities:
                entity_type = str(getattr(entity, "type", "")).split(".")[-1].lower()
                if entity_type == "mention" and expected:
                    offset = int(getattr(entity, "offset", -1))
                    length = int(getattr(entity, "length", 0))
                    if offset < 0 or length <= 0:
                        continue
                    if self._telegram_entity_text(source_text, offset, length).strip().lower() == expected:
                        return True
                elif entity_type == "text_mention":
                    user = getattr(entity, "user", None)
                    if user and getattr(user, "id", None) == bot_id:
                        return True
                elif entity_type == "bot_command" and expected:
                    # ``/cmd@botname`` is a single bot_command entity with no mention entity.
                    # It is what Telegram's group command menu produces, so it must count
                    # as a direct address or require_mention groups lose slash commands.
                    offset = int(getattr(entity, "offset", -1))
                    length = int(getattr(entity, "length", 0))
                    if offset < 0 or length <= 0:
                        continue
                    command_text = self._telegram_entity_text(source_text, offset, length)
                    at_index = command_text.find("@")
                    if at_index < 0:
                        continue
                    if command_text[at_index:].strip().lower() == expected:
                        return True
        if bot_username:
            return bot_username in self._extract_bot_mention_usernames(message, bot_username)
        return False

    def _schedule_bot_identity_recheck(self) -> None:
        """Fire a TTL-guarded identity refresh in the background.

        Called when routing is about to discard a message naming other bots but not us — the
        symptom of a stale handle after a rename. TTL-bounded to one getMe per
        ``_BOT_IDENTITY_TTL_SECONDS``; fire-and-forget, the current message routes as-is.
        """
        existing = getattr(self, "_bot_identity_refresh_task", None)
        if existing is not None and not existing.done():
            return
        if self._bot_identity_is_fresh():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        task = loop.create_task(self._refresh_bot_identity())
        self._bot_identity_refresh_task = task
        tracked = getattr(self, "_background_tasks", None)
        if isinstance(tracked, set):
            tracked.add(task)
            task.add_done_callback(tracked.discard)

    def _explicit_bot_mentions_exclude_self(self, message: Message) -> bool:
        """Return True when explicit bot handles target other bots, not this one.

        Groups may hold several Hermes bots; ``@bot3 hi @bot4`` must not wake ``@bot1`` via
        reply/wake-word fallbacks. Foreign handles are limited to the ``...bot`` shape so
        human @handles never suppress us; our own handle is matched by identity.
        """
        if not self._bot:
            return False
        bot_username = self._current_bot_username()
        if not bot_username:
            return False
        mentioned_bot_usernames = self._extract_bot_mention_usernames(message, bot_username)
        excludes_self = bool(mentioned_bot_usernames) and bot_username not in mentioned_bot_usernames
        if excludes_self:
            # Either truly for another bot, or our handle is stale after a rename —
            # re-check identity out of band (TTL bounded) so the mistake self-corrects.
            self._schedule_bot_identity_recheck()
        return excludes_self

    def _message_matches_mention_patterns(self, message: Message) -> bool:
        if not self._mention_patterns:
            return False
        for candidate in (getattr(message, "text", None), getattr(message, "caption", None)):
            if not candidate:
                continue
            for pattern in self._mention_patterns:
                if pattern.search(candidate):
                    return True
        return False

    def _is_guest_mention(self, message: Message) -> bool:
        """Guest-mode bypass: explicit bot mention (caller already verified group chat)."""
        return self._telegram_guest_mode() and self._message_mentions_bot(message)

    def _clean_bot_trigger_text(self, text: Optional[str]) -> Optional[str]:
        bot_username = self._current_bot_username()
        if not text or not bot_username:
            return text
        username = re.escape(bot_username)
        cleaned = re.sub(rf"(?i)@{username}\b[,:\-]*\s*", "", text).strip()
        return cleaned or text

    def _should_observe_unmentioned_group_message(self, message: Message) -> bool:
        """Return True when a group message should be stored but not dispatched."""
        if self._is_own_message(message):
            return False
        if not self._telegram_observe_unmentioned_group_messages():
            return False
        if not self._is_group_chat(message):
            return False
        thread_id = getattr(message, "message_thread_id", None)
        allowed_topics = self._telegram_allowed_topics()
        if allowed_topics:
            topic_id = str(thread_id) if thread_id is not None else self._GENERAL_TOPIC_THREAD_ID
            if topic_id not in allowed_topics:
                return False
        if thread_id is not None:
            try:
                if int(thread_id) in self._telegram_ignored_threads():
                    return False
            except (TypeError, ValueError):
                return False
        chat_id_str = str(getattr(getattr(message, "chat", None), "id", ""))
        if self._telegram_exclusive_bot_mentions() and self._explicit_bot_mentions_exclude_self(message):
            return False
        allowed = self._telegram_observe_allowed_chats()
        # Observed context is shared at chat/topic scope, so require an explicit chat
        # allowlist: it limits shared history to operator-approved groups and lets gateway
        # auth pass once the shared source drops the per-sender user_id.
        if not allowed or chat_id_str not in allowed:
            return False
        # Only observe messages the require_mention gate would skip; anything that would be
        # processed normally belongs to the dispatcher.
        if chat_id_str in self._telegram_free_response_chats():
            return False
        if self._telegram_is_free_response_topic(message):
            return False
        if not self._telegram_require_mention():
            return False
        if self._is_reply_to_bot(message):
            return False
        if self._message_mentions_bot(message):
            return False
        return not self._message_matches_mention_patterns(message)

    def _telegram_group_observe_shared_source(self, source):
        """Return a chat/topic-scoped source for observed Telegram group context."""
        return dataclasses.replace(source, user_id=None, user_name=None, user_id_alt=None)

    def _telegram_group_observe_attributed_text(self, event: MessageEvent) -> str:
        user_id = event.source.user_id or "unknown"
        sender = event.source.user_name or user_id
        return f"[{sender}|{user_id}]\n{event.text or ''}"

    def _telegram_group_observe_channel_prompt(self) -> str:
        username = self._current_bot_username() or "unknown"
        bot_id = getattr(getattr(self, "_bot", None), "id", None) or "unknown"
        return (
            "You are handling a Telegram group chat message.\n"
            f"- Your identity: user_id={bot_id}, @-mention name in this group=@{username}\n"
            "- observed Telegram group context may be provided in a separate context-only block "
            "before the current message; it is not necessarily addressed to you.\n"
            "- Treat only the current new message as a request explicitly directed at you, "
            "and use observed context only when the current message asks for it."
        )

    def _apply_telegram_group_observe_attribution(self, event: MessageEvent) -> MessageEvent:
        """Align triggered group turns with observed-history attribution."""
        if not self._telegram_observe_unmentioned_group_messages():
            return event
        raw_message = getattr(event, "raw_message", None)
        if not raw_message or not self._is_group_chat(raw_message):
            return event
        chat_id_str = str(getattr(getattr(raw_message, "chat", None), "id", ""))
        allowed = self._telegram_observe_allowed_chats()
        if not allowed or chat_id_str not in allowed:
            return event
        shared_source = self._telegram_group_observe_shared_source(event.source)
        observe_prompt = self._telegram_group_observe_channel_prompt()
        channel_prompt = f"{event.channel_prompt}\n\n{observe_prompt}" if event.channel_prompt else observe_prompt
        if event.message_type == MessageType.COMMAND:
            # Commands keep the original source (user_id) so _check_slash_access can identify
            # the sender — SlashAccessPolicy.is_admin(None) is always False. Still inject prompt.
            return dataclasses.replace(event, channel_prompt=channel_prompt)
        return dataclasses.replace(
            event,
            text=self._telegram_group_observe_attributed_text(event),
            source=shared_source,
            channel_prompt=channel_prompt,
        )

    def _media_message_type(self, msg: Message) -> MessageType:
        """Classify a Telegram media message into a MessageType."""
        if msg.sticker:
            return MessageType.STICKER
        if msg.photo:
            return MessageType.PHOTO
        if msg.video:
            return MessageType.VIDEO
        if msg.audio:
            return MessageType.AUDIO
        if msg.voice:
            return MessageType.VOICE
        return MessageType.DOCUMENT

    _CACHED_KIND_TO_MESSAGE_TYPE = {"image": MessageType.PHOTO, "video": MessageType.VIDEO, "audio": MessageType.AUDIO}

    async def _download_observed_media(self, msg: Any, what: str):
        """Download ``msg``'s attachment into the media cache (bounded by ``_max_doc_bytes``).

        Returns ``(status, cached)`` where status is ``"none"`` (no attachment),
        ``"oversized"`` (skipped, ``cached`` is the raw file_size), ``"failed"``
        (download error, logged), ``"unreadable"`` (cache rejected it) or ``"ok"``.
        """
        from gateway.platforms.base import cache_media_bytes
        source, filename, mime, kind = self._observed_media_source(msg)
        if source is None:
            return "none", None
        max_bytes = getattr(self, "_max_doc_bytes", 20 * 1024 * 1024)
        file_size = getattr(source, "file_size", None)
        try:
            size = int(file_size or 0)
        except (TypeError, ValueError):
            size = 0
        if not (0 < size <= max_bytes):
            return "oversized", file_size
        try:
            file_obj = await source.get_file()
            data = bytes(await file_obj.download_as_bytearray())
            if not filename:
                filename = os.path.basename(getattr(file_obj, "file_path", "") or "")
            cached = cache_media_bytes(data, filename=filename, mime_type=mime, default_kind=kind)
        except Exception as exc:
            logger.warning("[Telegram] Failed to cache %s: %s", what, _redact_telegram_error_text(exc), exc_info=True)
            return "failed", None
        if cached is None:
            return "unreadable", None
        return "ok", cached

    async def _cache_observed_media(self, msg: Message, event: MessageEvent) -> None:
        """Cache an unmentioned group attachment and annotate the observed text.

        Bounded by ``_max_doc_bytes`` like the addressed document path; oversized or
        unsupported attachments are noted in the transcript without downloading.
        """
        status, cached = await self._download_observed_media(msg, "observed group media")
        if status == "oversized":
            limit_mb = getattr(self, "_max_doc_bytes", 20 * 1024 * 1024) // (1024 * 1024)
            event.text = self._append_observed_note(
                event.text, f"[Observed Telegram attachment too large or unverifiable. Maximum: {limit_mb} MB.]"
            )
            logger.info("[Telegram] Observed group attachment skipped (size=%s)", cached)
            return
        if status == "unreadable":
            # Only images that fail validation reach here; every other type is always cached.
            event.text = self._append_observed_note(event.text, "[Observed Telegram attachment could not be read, not cached.]")
            return
        if status != "ok":
            return
        event.media_urls = [cached.path]
        event.media_types = [cached.media_type]
        if cached.kind in self._CACHED_KIND_TO_MESSAGE_TYPE:
            event.message_type = self._CACHED_KIND_TO_MESSAGE_TYPE[cached.kind]
        event.text = self._append_observed_note(event.text, cached.context_note())
        logger.info("[Telegram] Cached observed group %s at %s", cached.kind, cached.path)

    async def _cache_replied_media(self, msg: Any, event: MessageEvent) -> None:
        """Cache media from the message this turn replies to, if any."""
        reply_msg = getattr(msg, "reply_to_message", None)
        if reply_msg is None:
            return
        status, cached = await self._download_observed_media(reply_msg, "replied-to media")
        if status != "ok":
            return
        event.media_urls.append(cached.path)
        event.media_types.append(cached.media_type)
        if len(event.media_urls) == 1 and cached.kind in self._CACHED_KIND_TO_MESSAGE_TYPE:
            event.message_type = self._CACHED_KIND_TO_MESSAGE_TYPE[cached.kind]
        event.text = self._append_observed_note(
            event.text, f"[Replied-to {cached.kind} '{cached.display_name}' saved at: {cached.path}]"
        )
        logger.info("[Telegram] Cached replied-to %s at %s", cached.kind, cached.path)

    def _observed_media_source(self, msg: Message):
        """Return (telegram_file_source, filename, mime, default_kind) or Nones."""
        if msg.photo:
            return msg.photo[-1], "", "", "image"
        if msg.video:
            return msg.video, "", "video/mp4", "video"
        if msg.voice:
            return msg.voice, "voice.ogg", "audio/ogg", "audio"
        if msg.audio:
            return msg.audio, getattr(msg.audio, "file_name", "") or "", "", "audio"
        if msg.document:
            doc = msg.document
            return doc, doc.file_name or "", (doc.mime_type or "").lower(), None
        return None, "", "", None

    @staticmethod
    def _append_observed_note(existing: Optional[str], note: str) -> str:
        if not note:
            return existing or ""
        if not existing:
            return note
        return f"{existing}\n\n{note}"

    async def _surface_media_cache_failure(
        self, msg: Message, event: MessageEvent, kind: str, exc: Exception, display_name: Optional[str] = None
    ) -> None:
        """Surface a failed media download/cache to BOTH the user and the agent.

        A failed download (typically a transient CDN error) leaves event.media_urls empty;
        without this the turn dispatches silently — user thinks it was delivered, agent sees
        nothing. Reply asking to retry, and append an agent-visible observed note.
        """
        named = f" ({display_name})" if display_name else ""
        try:
            await msg.reply_text(
                f"\u26a0\ufe0f Couldn't download your {kind}{named} ({exc.__class__.__name__}). Please try sending it again."
            )
        except Exception as reply_err:
            logger.warning("[Telegram] Failed to notify user about %s cache failure: %s", kind, reply_err, exc_info=True)
        agent_note = (
            f"[The user attempted to send a {kind}{named} but it could not be downloaded ({exc.__class__.__name__}); they have been asked to retry.]"
        )
        event.text = self._append_observed_note(event.text, agent_note)

    def _observe_unmentioned_group_message(
        self, message: Message, msg_type: MessageType, update_id: Optional[int] = None, event: Optional[MessageEvent] = None
    ) -> None:
        """Append skipped group chatter to the target session without dispatching."""
        store = getattr(self, "_session_store", None)
        if not store:
            return
        try:
            event = event or self._build_message_event(message, msg_type, update_id=update_id)
            shared_source = self._telegram_group_observe_shared_source(event.source)
            session_entry = store.get_or_create_session(shared_source)
            entry = {
                "role": "user",
                "content": self._telegram_group_observe_attributed_text(event),
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                "observed": True,
            }
            if event.message_id:
                entry["message_id"] = str(event.message_id)
            store.append_to_transcript(session_entry.session_id, entry)
            adapter_name = getattr(self, "name", "telegram")
            logger.info(
                "[%s] Telegram group message observed (no bot trigger): chat=%s from=%s",
                adapter_name,
                getattr(getattr(message, "chat", None), "id", "unknown"),
                event.source.user_id or "unknown",
            )
        except Exception as exc:
            adapter_name = getattr(self, "name", "telegram")
            logger.warning("[%s] Failed to observe Telegram group message: %s", adapter_name, exc)

    def _is_own_message(self, message: Message) -> bool:
        """True when sent by this bot itself.

        In groups where the bot sees its own messages, getUpdates returns them as updates;
        they must not count as incoming unread in the Hermes inbox.
        """
        if not self._bot:
            return False
        from_user = getattr(message, "from_user", None)
        if from_user is None:
            return False
        bot_id = getattr(self._bot, "id", None)
        user_id = getattr(from_user, "id", None)
        return bot_id is not None and user_id is not None and bot_id == user_id

    def _should_process_message(self, message: Message, *, is_command: bool = False) -> bool:
        """Apply Telegram group trigger rules.

        DMs are unrestricted. Group messages are accepted when the chat passes ``allowed_chats``
        (a hard gate; only the ``guest_mode`` explicit-@mention bypass crosses it) and then any
        of: ``free_response_chats``/topic, ``require_mention`` off, reply to the bot, @mention,
        or a regex wake-word match. Slash commands get no special treatment under
        ``require_mention``; ``/cmd@botname`` and ``@botname /cmd`` count as mentions.
        """
        # Learn the live handle BEFORE any mention gate routes on it (a rename would
        # otherwise make the exclusive-mention gate misread messages addressed to us),
        # then drop our own echoed messages so they never count as incoming unread.
        self._observe_bot_identity_from_message(message)
        if self._is_own_message(message):
            return False
        if not self._is_group_chat(message):
            return True
        thread_id = self._effective_message_thread_id(message)
        allowed_topics = self._telegram_allowed_topics()
        if allowed_topics:
            topic_id = str(thread_id) if thread_id is not None else self._GENERAL_TOPIC_THREAD_ID
            if topic_id not in allowed_topics:
                return False

        # ignored_threads applies to both groups and DM topics
        if thread_id is not None:
            try:
                if int(thread_id) in self._telegram_ignored_threads():
                    return False
            except (TypeError, ValueError):
                logger.warning("[%s] Ignoring non-numeric Telegram message_thread_id: %r", self.name, thread_id)
        if not self._is_group_chat(message):
            # Root DM (non-topic): ignore if ignore_root_dm is configured
            if thread_id is None and self.config.extra.get("ignore_root_dm", False):
                chat_id = str(getattr(getattr(message, "chat", None), "id", ""))
                if not is_command and chat_id in self._dm_topic_chat_ids:
                    return False
            return True
        chat_id_str = str(getattr(getattr(message, "chat", None), "id", ""))
        if self._telegram_exclusive_bot_mentions() and self._explicit_bot_mentions_exclude_self(message):
            return False

        # Resolve once; _message_mentions_bot is not re-called below in guest mode.
        guest_mention = self._is_guest_mention(message)
        # allowed_chats whitelist: outside chats pass only via the guest-mode explicit mention.
        allowed = self._telegram_allowed_chats()
        if allowed and chat_id_str not in allowed:
            return guest_mention
        if guest_mention:
            return True
        if chat_id_str in self._telegram_free_response_chats():
            return True
        if self._telegram_is_free_response_topic(message):
            return True
        if not self._telegram_require_mention():
            return True
        if self._is_reply_to_bot(message):
            return True
        if not self._telegram_guest_mode() and self._message_mentions_bot(message):
            return True
        return self._message_matches_mention_patterns(message)

    async def _ensure_forum_commands(self, message) -> None:
        """Lazy-register bot commands for forum supergroups.

        Forum topics don't inherit AllGroupChats scope (Telegram resolves via
        BotCommandScopeChat), so register on first message for the topic-view menu.
        """
        async with self._forum_lock:
            try:
                chat = getattr(message, "chat", None)
                if not chat or not getattr(chat, "is_forum", False):
                    return
                chat_id = int(chat.id)
                if chat_id in self._forum_command_registered:
                    return
                from telegram import BotCommand, BotCommandScopeChat
                from hermes_cli.commands import telegram_menu_commands, telegram_menu_max_commands
                menu_commands, _ = telegram_menu_commands(max_commands=telegram_menu_max_commands())
                bot_commands = [BotCommand(name, desc) for name, desc in menu_commands]
                await self._bot.set_my_commands(bot_commands, scope=BotCommandScopeChat(chat_id=chat_id))
                self._forum_command_registered.add(chat_id)
                logger.info("[%s] Lazy-registered %d commands for forum chat %s", self.name, len(bot_commands), chat_id)
            except Exception as e:
                logger.warning("[%s] Forum command lazy-registration failed: %s", self.name, _redact_telegram_error_text(e))

    def _effective_update_message(self, update: Update) -> Optional[Message]:
        """Message-like payload for normal messages and channel posts.

        Channel broadcasts arrive as ``update.channel_post``, not ``update.message``; using
        ``effective_message`` keeps handlers from consuming them without building an event.
        """
        return getattr(update, "effective_message", None) or getattr(update, "message", None)

    async def _handle_text_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle incoming text; buffers client-split chunks into one MessageEvent."""
        msg = self._effective_update_message(update)
        if not msg or not msg.text:
            return
        # Auth check first: blocked users must not reach batching, the observed
        # transcript, or the agent path.
        if not self._is_user_authorized_from_message(msg):
            logger.warning(
                "[Telegram] Blocked unauthorized user %s in chat %s",
                getattr(getattr(msg, "from_user", None), "id", None),
                getattr(getattr(msg, "chat", None), "id", None),
            )
            return
        if not self._should_process_message(msg):
            if self._should_observe_unmentioned_group_message(msg):
                self._observe_unmentioned_group_message(msg, MessageType.TEXT, update_id=update.update_id)
            return
        await self._ensure_forum_commands(update.message)
        event = self._build_message_event(msg, MessageType.TEXT, update_id=update.update_id)
        event.text = self._clean_bot_trigger_text(event.text)
        await self._cache_replied_media(msg, event)
        event = self._apply_telegram_group_observe_attribution(event)
        self._enqueue_text_event(event)

    async def _handle_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle incoming command messages."""
        msg = self._effective_update_message(update)
        if not msg or not msg.text:
            return
        if not self._should_process_message(msg, is_command=True):
            return
        if not self._is_user_authorized_from_message(msg):
            logger.warning(
                "[Telegram] Blocked unauthorized user %s in chat %s",
                getattr(getattr(msg, "from_user", None), "id", None),
                getattr(getattr(msg, "chat", None), "id", None),
            )
            return
        await self._ensure_forum_commands(msg)
        event = self._build_message_event(msg, MessageType.COMMAND, update_id=update.update_id)
        event.text = self._clean_bot_trigger_text(event.text)
        await self._cache_replied_media(msg, event)
        event = self._apply_telegram_group_observe_attribution(event)
        # A >4096-char command paste arrives as a near-limit COMMAND chunk plus TEXT
        # continuations; dispatching immediately would orphan them (and interrupt the
        # agent). Near-limit commands go through text batching; short ones stay immediate.
        if len(event.text or "") >= self._SPLIT_THRESHOLD:
            self._enqueue_text_event(event)
            return
        await self.handle_message(event)

    async def _handle_location_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle incoming location/venue pin messages."""
        msg = self._effective_update_message(update)
        if not msg:
            return
        if not self._is_user_authorized_from_message(msg):
            logger.warning(
                "[Telegram] Blocked unauthorized user %s in chat %s",
                getattr(getattr(msg, "from_user", None), "id", None),
                getattr(getattr(msg, "chat", None), "id", None),
            )
            return
        if not self._should_process_message(msg):
            if self._should_observe_unmentioned_group_message(msg):
                self._observe_unmentioned_group_message(msg, MessageType.LOCATION, update_id=update.update_id)
            return
        venue = getattr(msg, "venue", None)
        location = getattr(venue, "location", None) if venue else getattr(msg, "location", None)
        if not location:
            return
        lat = getattr(location, "latitude", None)
        lon = getattr(location, "longitude", None)
        if lat is None or lon is None:
            return
        parts = ["[The user shared a location pin.]"]
        if venue:
            title = getattr(venue, "title", None)
            address = getattr(venue, "address", None)
            if title:
                parts.append(f"Venue: {title}")
            if address:
                parts.append(f"Address: {address}")
        parts.append(f"latitude: {lat}")
        parts.append(f"longitude: {lon}")
        parts.append(f"Map: https://www.google.com/maps/search/?api=1&query={lat},{lon}")
        parts.append("Ask what they'd like to find nearby (restaurants, cafes, etc.) and any preferences.")
        event = self._build_message_event(msg, MessageType.LOCATION, update_id=update.update_id)
        event.text = "\n".join(parts)
        event = self._apply_telegram_group_observe_attribution(event)
        await self.handle_message(event)

    # -- Text message aggregation (handles Telegram client-side splits) --

    def _text_batch_key(self, event: MessageEvent) -> str:
        """Session-scoped batching key; topic recovery first so DM-topic batches coalesce on
        the recovered lane, not the raw inbound thread id."""
        self._apply_topic_recovery(event)
        return super()._text_batch_key(event)

    def _enqueue_text_event(self, event: MessageEvent) -> None:
        """Buffer a text chunk, or hold it while delayed delivery must be dropped."""
        if self._should_drop_delayed_delivery():
            self._hold_inbound_event(event, where="text-enqueue")
            return
        super()._enqueue_text_event(event)

    async def _flush_text_batch(self, key: str) -> None:
        """Wait for the quiet period then dispatch the aggregated text."""
        current_task = asyncio.current_task()
        event = None
        try:
            # Adaptive delay: near-split-point last chunk → long delay (continuation almost
            # certain); short/medium totals → capped fast delays; else configured cap. All
            # tiers are min()'d with the operator's ``_text_batch_delay_seconds`` override.
            pending = self._pending_text_batches.get(key)
            last_len = getattr(pending, "_last_chunk_len", 0) if pending else 0
            total_len = len(getattr(pending, "text", "") or "") if pending else 0
            if last_len >= self._SPLIT_THRESHOLD:
                delay = self._text_batch_split_delay_seconds
            elif total_len <= self._TEXT_BATCH_FAST_LEN:
                delay = min(self._text_batch_delay_seconds, self._TEXT_BATCH_FAST_DELAY_S)
            elif total_len <= self._TEXT_BATCH_SHORT_LEN:
                delay = min(self._text_batch_delay_seconds, self._TEXT_BATCH_SHORT_DELAY_S)
            else:
                delay = self._text_batch_delay_seconds
            await asyncio.sleep(delay)
            event = self._pending_text_batches.pop(key, None)
            if not event:
                return
            if self._should_drop_delayed_delivery():
                self._hold_inbound_event(event, where="text-flush")
                event = None
                return
            logger.info("[Telegram] Flushing text batch %s (%d chars)", key, len(event.text or ""))
            await self.handle_message(event)
            event = None
        except asyncio.CancelledError:
            # Cancelled after pop but before durable dispatch — hold, don't lose.
            if event is not None:
                self._hold_inbound_event(event, where="text-flush-cancelled")
            raise
        finally:
            if self._pending_text_batch_tasks.get(key) is current_task:
                self._pending_text_batch_tasks.pop(key, None)

    # -- Photo batching --

    def _photo_batch_key(self, event: MessageEvent, msg: Message) -> str:
        """Return a batching key for Telegram photos/albums."""
        from gateway.session import build_session_key
        session_key = build_session_key(
            event.source,
            group_sessions_per_user=self.config.extra.get("group_sessions_per_user", True),
            thread_sessions_per_user=self.config.extra.get("thread_sessions_per_user", False),
            profile=self._session_key_profile(event.source),
        )
        media_group_id = getattr(msg, "media_group_id", None)
        if media_group_id:
            return f"{session_key}:album:{media_group_id}"
        return f"{session_key}:photo-burst"

    async def _flush_photo_batch(self, batch_key: str) -> None:
        """Send a buffered photo burst/album as a single MessageEvent."""
        current_task = asyncio.current_task()
        event = None
        try:
            await asyncio.sleep(self._media_batch_delay_seconds)
            event = self._pending_photo_batches.pop(batch_key, None)
            if not event:
                return
            if self._should_drop_delayed_delivery():
                self._hold_inbound_event(event, where="photo-flush")
                event = None
                return
            logger.info("[Telegram] Flushing photo batch %s with %d image(s)", batch_key, len(event.media_urls))
            await self.handle_message(event)
            event = None
        except asyncio.CancelledError:
            if event is not None:
                self._hold_inbound_event(event, where="photo-flush-cancelled")
            raise
        finally:
            if self._pending_photo_batch_tasks.get(batch_key) is current_task:
                self._pending_photo_batch_tasks.pop(batch_key, None)

    def _enqueue_photo_event(self, batch_key: str, event: MessageEvent) -> None:
        """Merge photo events into a pending batch and schedule flush."""
        if self._should_drop_delayed_delivery():
            self._hold_inbound_event(event, where="photo-enqueue")
            return
        existing = self._pending_photo_batches.get(batch_key)
        if existing is None:
            self._pending_photo_batches[batch_key] = event
        else:
            existing.media_urls.extend(event.media_urls)
            existing.media_types.extend(event.media_types)
            if event.text:
                existing.text = self._merge_caption(existing.text, event.text)
        prior_task = self._pending_photo_batch_tasks.get(batch_key)
        if prior_task and not prior_task.done():
            prior_task.cancel()
        self._pending_photo_batch_tasks[batch_key] = asyncio.create_task(self._flush_photo_batch(batch_key))

    async def _route_photo_event(self, msg, event: MessageEvent) -> None:
        """Album items debounce on media_group_id; singles go through the photo burst batcher."""
        media_group_id = getattr(msg, "media_group_id", None)
        if media_group_id:
            await self._queue_media_group_event(str(media_group_id), event)
        else:
            self._enqueue_photo_event(self._photo_batch_key(event, msg), event)

    async def _cache_inbound_av(
        self, msg, event: MessageEvent, source: Any, label: str, kind: str, ext: str, mime: str,
    ) -> bool:
        """Download a voice/audio/video attachment into the local cache.

        Returns True when the event was already dispatched (oversized attachment),
        so the caller must return. Video resolves ``ext``/``mime`` from the file path.
        """
        try:
            allowed, note = self._telegram_media_size_allowed(source, label)
            if not allowed:
                event.text = self._append_observed_note(event.text, note or "")
                logger.info("[Telegram] Skipped oversized user %s (size=%s)", kind, getattr(source, "file_size", None))
                await self.handle_message(event)
                return True
            file_obj = await source.get_file()
            data = await file_obj.download_as_bytearray()
            if kind == "video":
                if getattr(file_obj, "file_path", None):
                    for candidate in SUPPORTED_VIDEO_TYPES:
                        if file_obj.file_path.lower().endswith(candidate):
                            ext = candidate
                            break
                cached_path = cache_video_from_bytes(bytes(data), ext=ext)
                mime = SUPPORTED_VIDEO_TYPES.get(ext, "video/mp4")
            else:
                cached_path = cache_audio_from_bytes(bytes(data), ext=ext)
            event.media_urls = [cached_path]
            event.media_types = [mime]
            logger.info("[Telegram] Cached user %s at %s", kind, cached_path)
        except Exception as e:
            logger.warning("[Telegram] Failed to cache %s: %s", kind, _redact_telegram_error_text(e), exc_info=True)
            await self._surface_media_cache_failure(msg, event, label, e)
        return False

    async def _handle_media_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle incoming media messages, downloading images to local cache."""
        if not update.message:
            return
        if not self._is_user_authorized_from_message(update.message):
            logger.info(
                "[Telegram] Blocked media from unauthorized user %s in chat %s",
                getattr(getattr(update.message, "from_user", None), "id", None),
                getattr(getattr(update.message, "chat", None), "id", None),
            )
            return
        if not self._should_process_message(update.message):
            if self._should_observe_unmentioned_group_message(update.message):
                _m = update.message
                _observe_type = self._media_message_type(_m)
                _event = self._build_message_event(_m, _observe_type, update_id=update.update_id)
                if _m.caption:
                    _event.text = self._clean_bot_trigger_text(_m.caption)
                await self._cache_observed_media(_m, _event)
                self._observe_unmentioned_group_message(_m, _event.message_type, update_id=update.update_id, event=_event)
            return
        msg = update.message
        msg_type = self._media_message_type(msg)
        event = self._build_message_event(msg, msg_type, update_id=update.update_id)
        if msg.caption:
            event.text = self._clean_bot_trigger_text(msg.caption)

        # Stickers: _handle_sticker overwrites event.text with its vision description, so
        # observe attribution must run after it.
        if msg.sticker:
            await self._handle_sticker(msg, event)
            event = self._apply_telegram_group_observe_attribution(event)
            await self.handle_message(event)
            return
        event = self._apply_telegram_group_observe_attribution(event)

        # Cache photo locally: Telegram's file URLs expire (~1 hour) before vision may run.
        if msg.photo:
            try:
                photo = msg.photo[-1]  # PhotoSize list sorted by size; largest last
                file_obj = await photo.get_file()
                image_bytes = await file_obj.download_as_bytearray()
                ext = ".jpg"
                if file_obj.file_path:
                    for candidate in [".png", ".webp", ".gif", ".jpeg", ".jpg"]:
                        if file_obj.file_path.lower().endswith(candidate):
                            ext = candidate
                            break
                cached_path = cache_image_from_bytes(bytes(image_bytes), ext=ext)
                event.media_urls = [cached_path]
                event.media_types = [f"image/{ext.lstrip('.')}"]
                logger.info("[Telegram] Cached user photo at %s", cached_path)
                await self._route_photo_event(msg, event)
                return
            except Exception as e:
                logger.warning("[Telegram] Failed to cache photo: %s", _redact_telegram_error_text(e), exc_info=True)
                await self._surface_media_cache_failure(msg, event, "photo", e)

        # Voice/audio cached for STT transcription; video for vision.
        if msg.voice:
            if await self._cache_inbound_av(msg, event, msg.voice, "voice message", "voice", ".ogg", "audio/ogg"):
                return
        elif msg.audio:
            if await self._cache_inbound_av(msg, event, msg.audio, "audio file", "audio", ".mp3", "audio/mp3"):
                return
        elif msg.video:
            if await self._cache_inbound_av(msg, event, msg.video, "video file", "video", ".mp4", "video/mp4"):
                return
        elif msg.document:
            doc = msg.document
            try:
                ext = ""
                original_filename = doc.file_name or ""
                if original_filename:
                    _, ext = os.path.splitext(original_filename)
                    ext = ext.lower()
                doc_mime = (doc.mime_type or "").lower()  # some clients send "IMAGE/PNG"
                if not ext and doc_mime:
                    ext = _TELEGRAM_IMAGE_MIME_TO_EXT.get(doc_mime, "")
                    if not ext:
                        mime_to_ext = {v: k for k, v in SUPPORTED_DOCUMENT_TYPES.items()}
                        ext = mime_to_ext.get(doc_mime, "")

                # Size check before the image branch so image documents can't bypass the limit.
                if not doc.file_size or doc.file_size > self._max_doc_bytes:
                    limit_mb = self._max_doc_bytes // (1024 * 1024)
                    event.text = f"The document is too large or its size could not be verified. Maximum: {limit_mb} MB."
                    logger.info("[Telegram] Document too large: %s bytes", doc.file_size)
                    await self.handle_message(event)
                    return

                # Screenshots/photos sent as documents take the image cache + batching path.
                if ext in _TELEGRAM_IMAGE_EXTENSIONS or doc_mime.startswith("image/"):
                    file_obj = await doc.get_file()
                    image_bytes = await file_obj.download_as_bytearray()
                    image_ext = ext if ext in _TELEGRAM_IMAGE_EXTENSIONS else _TELEGRAM_IMAGE_MIME_TO_EXT.get(doc_mime, ".jpg")
                    try:
                        cached_path = cache_image_from_bytes(bytes(image_bytes), ext=image_ext)
                    except ValueError as e:
                        logger.warning("[Telegram] Failed to cache image document: %s", _redact_telegram_error_text(e), exc_info=True)
                        event.text = f"Image document '{original_filename or doc_mime or ext or 'unknown'}' could not be read as an image."
                        await self.handle_message(event)
                        return
                    event.message_type = MessageType.PHOTO
                    event.media_urls = [cached_path]
                    event.media_types = [doc_mime if doc_mime.startswith("image/") else _TELEGRAM_IMAGE_EXT_TO_MIME.get(image_ext, "image/jpeg")]
                    logger.info("[Telegram] Cached user image-document at %s", cached_path)
                    await self._route_photo_event(msg, event)
                    return
                if not ext and doc.mime_type:
                    video_mime_to_ext = {v: k for k, v in SUPPORTED_VIDEO_TYPES.items()}
                    ext = video_mime_to_ext.get(doc.mime_type, "")
                if not ext and doc.mime_type:
                    # .jpg and .jpeg both map to image/jpeg; keep the first ext seen.
                    image_mime_to_ext: dict[str, str] = {}
                    for _ext, _mime in SUPPORTED_IMAGE_DOCUMENT_TYPES.items():
                        image_mime_to_ext.setdefault(_mime, _ext)
                    ext = image_mime_to_ext.get(doc.mime_type, "")
                if ext in SUPPORTED_VIDEO_TYPES:
                    file_obj = await doc.get_file()
                    video_bytes = await file_obj.download_as_bytearray()
                    cached_path = cache_video_from_bytes(bytes(video_bytes), ext=ext)
                    event.media_urls = [cached_path]
                    event.media_types = [SUPPORTED_VIDEO_TYPES[ext]]
                    event.message_type = MessageType.VIDEO
                    logger.info("[Telegram] Cached user video document at %s", cached_path)
                    await self.handle_message(event)
                    return

                # Any file type is accepted (authorization is the gate, not the extension).
                # Image documents already returned above — an ext-in-SUPPORTED_IMAGE_DOCUMENT_TYPES
                # branch here would be dead code. Unknown types get application/octet-stream.
                file_obj = await doc.get_file()
                doc_bytes = await file_obj.download_as_bytearray()
                raw_bytes = bytes(doc_bytes)
                from gateway.platforms.base import cache_media_bytes
                cached = cache_media_bytes(
                    raw_bytes, filename=original_filename or f"document{ext or '.bin'}", mime_type=doc_mime
                )
                if cached is None:
                    event.text = f"Document '{original_filename or doc_mime or ext or 'unknown'}' could not be cached."
                    await self.handle_message(event)
                    return
                event.media_urls = [cached.path]
                event.media_types = [cached.media_type]
                if cached.kind == "audio":
                    event.message_type = MessageType.AUDIO
                logger.info("[Telegram] Cached user %s at %s (%s)", cached.kind, cached.path, cached.media_type)

                # Inject text-readable content (≤100 KB). Gate on extension/MIME, NOT a blind
                # UTF-8 decode: PDF/zip/docx have decodable ASCII headers. Binary files are
                # surfaced as a cached path only.
                MAX_TEXT_INJECT_BYTES = 100 * 1024
                _is_text = ext in _TEXT_INJECT_EXTENSIONS or (doc_mime or "").startswith("text/")
                if _is_text and len(raw_bytes) <= MAX_TEXT_INJECT_BYTES:
                    try:
                        text_content = raw_bytes.decode("utf-8")
                        display_name = original_filename or f"document{ext or '.txt'}"
                        display_name = re.sub(r'[^\w.\- ]', '_', display_name)
                        injection = f"[Content of {display_name}]:\n{text_content}"
                        if event.text:
                            event.text = f"{injection}\n\n{event.text}"
                        else:
                            event.text = injection
                    except UnicodeDecodeError:
                        pass  # binary — agent has the cached path
            except Exception as e:
                logger.warning("[Telegram] Failed to cache document: %s", _redact_telegram_error_text(e), exc_info=True)
                await self._surface_media_cache_failure(
                    msg, event, "attachment", e, display_name=getattr(doc, "file_name", None) or None
                )
        media_group_id = getattr(msg, "media_group_id", None)
        if media_group_id:
            await self._queue_media_group_event(str(media_group_id), event)
            return
        await self.handle_message(event)

    async def _queue_media_group_event(self, media_group_id: str, event: MessageEvent) -> None:
        """Debounce album items (shared media_group_id) into one MessageEvent.

        Forwarding each item immediately would make the gateway treat the second image as a
        new message that interrupts the first.
        """
        if self._should_drop_delayed_delivery():
            self._hold_inbound_event(event, where="media-group-enqueue")
            return
        existing = self._media_group_events.get(media_group_id)
        if existing is None:
            self._media_group_events[media_group_id] = event
        else:
            existing.media_urls.extend(event.media_urls)
            existing.media_types.extend(event.media_types)
            if event.text:
                existing.text = self._merge_caption(existing.text, event.text)
        prior_task = self._media_group_tasks.get(media_group_id)
        if prior_task:
            prior_task.cancel()
        self._media_group_tasks[media_group_id] = asyncio.create_task(self._flush_media_group_event(media_group_id))

    async def _flush_media_group_event(self, media_group_id: str) -> None:
        current_task = asyncio.current_task()
        event = None
        try:
            await asyncio.sleep(self.MEDIA_GROUP_WAIT_SECONDS)
            event = self._media_group_events.pop(media_group_id, None)
            if event is None:
                return
            if self._should_drop_delayed_delivery():
                self._hold_inbound_event(event, where="media-group-flush")
                event = None
                return
            await self.handle_message(event)
            event = None
        except asyncio.CancelledError:
            # Cancelled after pop but before durable dispatch — hold, don't lose.
            if event is not None:
                self._hold_inbound_event(event, where="media-group-flush-cancelled")
            raise
        finally:
            if self._media_group_tasks.get(media_group_id) is current_task:
                self._media_group_tasks.pop(media_group_id, None)

    async def _handle_sticker(self, msg: Message, event: "MessageEvent") -> None:
        """Describe a sticker via vision, cached by file_unique_id.

        Animated/video stickers can't be analyzed as static images; they get an emoji placeholder.
        """
        from gateway.sticker_cache import (
            get_cached_description, cache_sticker_description, build_sticker_injection,
            build_animated_sticker_injection, STICKER_VISION_PROMPT,
        )
        sticker = msg.sticker
        emoji = sticker.emoji or ""
        set_name = sticker.set_name or ""
        if sticker.is_animated or sticker.is_video:
            event.text = build_animated_sticker_injection(emoji)
            return
        cached = get_cached_description(sticker.file_unique_id)
        if cached:
            event.text = build_sticker_injection(cached["description"], cached.get("emoji", emoji), cached.get("set_name", set_name))
            logger.info("[Telegram] Sticker cache hit: %s", sticker.file_unique_id)
            return
        try:
            file_obj = await sticker.get_file()
            image_bytes = await file_obj.download_as_bytearray()
            cached_path = cache_image_from_bytes(bytes(image_bytes), ext=".webp")
            logger.info("[Telegram] Analyzing sticker at %s", cached_path)
            from tools.vision_tools import vision_analyze_tool
            result_json = await vision_analyze_tool(image_url=cached_path, user_prompt=STICKER_VISION_PROMPT)
            result = json.loads(result_json)
            if result.get("success"):
                description = result.get("analysis", "a sticker")
                cache_sticker_description(sticker.file_unique_id, description, emoji, set_name)
                event.text = build_sticker_injection(description, emoji, set_name)
            else:
                event.text = build_sticker_injection(f"a sticker with emoji {emoji}" if emoji else "a sticker", emoji, set_name)
        except Exception as e:
            logger.warning("[Telegram] Sticker analysis error: %s", _redact_telegram_error_text(e), exc_info=True)
            event.text = build_sticker_injection(f"a sticker with emoji {emoji}" if emoji else "a sticker", emoji, set_name)

    def _reload_dm_topics_from_config(self) -> None:
        """Re-read dm_topics from config.yaml so externally created topics work without restart."""
        try:
            # Canonical loader: honors managed-scope overlay + ${VAR} expansion.
            from hermes_cli.config import load_config_readonly
            config = load_config_readonly()
            dm_topics = config.get("platforms", {}).get("telegram", {}).get("extra", {}).get("dm_topics", [])
            if not dm_topics:
                self._dm_topics_config = []
                self._dm_topic_chat_ids = set()
                return
            self._dm_topics_config = dm_topics
            # chat_id set gives O(1) root-DM ignore lookup
            self._dm_topic_chat_ids = {str(chat_entry["chat_id"]) for chat_entry in dm_topics if "chat_id" in chat_entry}
            for chat_entry in dm_topics:
                cid = chat_entry.get("chat_id")
                if not cid:
                    continue
                for t in chat_entry.get("topics", []):
                    tid = t.get("thread_id")
                    name = t.get("name")
                    if tid and name:
                        cache_key = f"{cid}:{name}"
                        if cache_key not in self._dm_topics:
                            self._dm_topics[cache_key] = int(tid)
                            logger.info("[%s] Hot-loaded DM topic from config: %s -> thread_id=%s", self.name, cache_key, tid)
        except Exception as e:
            logger.debug("[%s] Failed to reload dm_topics from config: %s", self.name, e)

    def _get_dm_topic_info(self, chat_id: str, thread_id: Optional[str]) -> Optional[Dict[str, Any]]:
        """Return the DM topic config dict (name, skill, ...) for this thread_id, or None."""
        if not thread_id:
            return None
        thread_id_int = int(thread_id)

        def _lookup() -> Optional[Dict[str, Any]]:
            for key, cached_tid in self._dm_topics.items():
                if cached_tid == thread_id_int and key.startswith(f"{chat_id}:"):
                    topic_name = key.split(":", 1)[1]
                    for chat_entry in self._dm_topics_config:
                        if str(chat_entry.get("chat_id")) == chat_id:
                            for t in chat_entry.get("topics", []):
                                if t.get("name") == topic_name:
                                    return t
                    return {"name": topic_name}
            return None

        found = _lookup()
        if found is not None:
            return found
        # Cache miss — hot-reload in case topics were added externally, then retry.
        self._reload_dm_topics_from_config()
        return _lookup()

    def _cache_dm_topic_from_message(self, chat_id: str, thread_id: str, topic_name: str) -> None:
        """Cache a thread_id -> topic_name mapping discovered from an incoming message."""
        cache_key = f"{chat_id}:{topic_name}"
        if cache_key not in self._dm_topics:
            self._dm_topics[cache_key] = int(thread_id)
            logger.info("[%s] Cached DM topic from message: %s -> thread_id=%s", self.name, cache_key, thread_id)

    @classmethod
    def _flatten_rich_inline_text(cls, value: Any) -> str:
        """Best-effort plaintext flattener for Bot API rich-message inline nodes."""
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            return "".join(cls._flatten_rich_inline_text(item) for item in value)
        if isinstance(value, dict):
            text = value.get("text")
            if text is not None:
                return cls._flatten_rich_inline_text(text)
            children = value.get("children")
            if children is not None:
                return cls._flatten_rich_inline_text(children)
        return ""

    @classmethod
    def _flatten_rich_blocks(cls, blocks: Any) -> str:
        """Best-effort plaintext flattener for Bot API rich-message blocks."""
        if not isinstance(blocks, list):
            return ""
        lines: List[str] = []
        for block in blocks:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "list":
                for item in block.get("items", []):
                    if not isinstance(item, dict):
                        continue
                    item_text = cls._flatten_rich_blocks(item.get("blocks"))
                    if not item_text:
                        continue
                    label = item.get("label")
                    item_lines = item_text.splitlines()
                    if not item_lines:
                        continue
                    first_line = item_lines[0]
                    if label:
                        first_line = f"{label} {first_line}".strip()
                    lines.append(first_line)
                    lines.extend(item_lines[1:])
                continue
            text = cls._flatten_rich_inline_text(block.get("text"))
            if text:
                lines.extend(text.splitlines())
        return "\n".join(line.rstrip() for line in lines if line)

    @classmethod
    def _extract_rich_reply_text(cls, reply_to_message: Any) -> Optional[str]:
        """Return plaintext echoed by Telegram's rich_message reply payload."""
        try:
            api_kwargs = getattr(reply_to_message, "api_kwargs", None)
            getter = getattr(api_kwargs, "get", None)
            if not callable(getter):
                return None
            rich_message = getter("rich_message")
            rich_getter = getattr(rich_message, "get", None)
            if not callable(rich_getter):
                return None
            text = cls._flatten_rich_blocks(rich_getter("blocks")).strip()
            return text or None
        except Exception:
            return None

    def _resolve_topic_binding(self, message: Message, chat_type: str, thread_id_str: Optional[str]) -> tuple:
        """Return ``(chat_topic, topic_skill)`` for a DM topic or bound forum topic (else Nones)."""
        chat = message.chat
        chat_topic = None
        topic_skill = None
        if chat_type == "dm" and thread_id_str:
            topic_info = self._get_dm_topic_info(str(chat.id), thread_id_str)
            if topic_info:
                chat_topic = topic_info.get("name")
                topic_skill = topic_info.get("skill")
            # forum_topic_created service messages also reveal topic names
            if hasattr(message, "forum_topic_created") and message.forum_topic_created:
                created_name = message.forum_topic_created.name
                if created_name:
                    self._cache_dm_topic_from_message(str(chat.id), thread_id_str, created_name)
                    if not chat_topic:
                        chat_topic = created_name
        elif chat_type == "group" and thread_id_str:
            # Forum topic skill binding via config.extra['group_topics']; accepts both
            # [{"chat_id": ..., "topics": [...]}] and legacy {"-100...": [{"thread_id": 12}]}.
            group_topics_config = self.config.extra.get("group_topics", [])
            if isinstance(group_topics_config, dict):
                group_topics_iter = [
                    {"chat_id": cfg_chat_id, "topics": topics} for cfg_chat_id, topics in group_topics_config.items()
                ]
            elif isinstance(group_topics_config, list):
                group_topics_iter = [entry for entry in group_topics_config if isinstance(entry, dict)]
            else:
                group_topics_iter = []
            for chat_entry in group_topics_iter:
                if str(chat_entry.get("chat_id", "")) == str(chat.id):
                    topics = chat_entry.get("topics", [])
                    if not isinstance(topics, list):
                        topics = []
                    for topic in topics:
                        if not isinstance(topic, dict):
                            continue
                        tid = topic.get("thread_id")
                        if tid is not None and str(tid) == thread_id_str:
                            chat_topic = topic.get("name")
                            topic_skill = topic.get("skill")
                            break
                    break
        return chat_topic, topic_skill

    def _reply_context(self, message: Message) -> tuple:
        """Return ``(reply_to_id, reply_to_text)`` for the replied-to message, if any.

        Prefers Telegram's native partial quote so quoting one substring doesn't inject the
        whole replied-to message; falls back to text/caption, rich echo, then the sent index.
        """
        if not message.reply_to_message:
            return None, None
        reply_to_id = str(message.reply_to_message.message_id)
        quote = getattr(message, "quote", None)
        quote_text = getattr(quote, "text", None) if quote is not None else None
        if quote_text:
            return reply_to_id, quote_text
        reply_to_text = message.reply_to_message.text or message.reply_to_message.caption or None
        if not reply_to_text:
            # Native rich-message echo first; local send-time index only as fallback.
            reply_to_text = self._extract_rich_reply_text(message.reply_to_message)
        if not reply_to_text:
            try:
                from gateway import rich_sent_store
                reply_to_text = rich_sent_store.lookup(str(message.chat.id), reply_to_id)
            except Exception:
                reply_to_text = None
        return reply_to_id, reply_to_text

    def _build_message_event(self, message: Message, msg_type: MessageType, update_id: Optional[int] = None) -> MessageEvent:
        """Build a MessageEvent from a Telegram message.

        ``update_id`` lets ``/restart`` record the triggering offset so the new gateway process
        advances past it (otherwise it is re-delivered when PTB's shutdown ACK fails).
        """
        chat = message.chat
        user = message.from_user
        # Normalize via str() so PTB enums (ChatType.CHANNEL) and plain-string mocks both work.
        telegram_chat_type = str(getattr(chat, "type", "")).split(".")[-1].lower()
        chat_type = "dm"
        if telegram_chat_type in {"group", "supergroup"}:
            chat_type = "group"
        elif telegram_chat_type == "channel":
            chat_type = "channel"

        # Shared normalizer so gating and session routing agree: reply-UI anchors are dropped
        # (sends against them hit 'Message thread not found'); forum General-topic messages
        # normalize to the General id so replies route back there.
        thread_id_str = self._effective_message_thread_id(message)
        chat_topic, topic_skill = self._resolve_topic_binding(message, chat_type, thread_id_str)
        source = self.build_source(
            chat_id=str(chat.id),
            chat_name=chat.title or (chat.full_name if hasattr(chat, "full_name") else None),
            chat_type=chat_type,
            user_id=(str(user.id) if user else (str(chat.id) if chat_type in {"dm", "channel"} else None)),
            user_name=(
                user.full_name
                if user
                else (
                    chat.full_name
                    if hasattr(chat, "full_name") and chat_type == "dm"
                    else (chat.title if chat_type == "channel" else None)
                )
            ),
            thread_id=thread_id_str,
            chat_topic=chat_topic,
            message_id=str(message.message_id),
            is_bot=bool(getattr(user, "is_bot", False)) if user else False,
        )
        reply_to_id, reply_to_text = self._reply_context(message)

        # Per-channel/topic ephemeral prompt
        from gateway.platforms.base import resolve_channel_prompt
        _chat_id_str = str(chat.id)
        _channel_prompt = resolve_channel_prompt(
            self.config.extra, thread_id_str or _chat_id_str, _chat_id_str if thread_id_str else None
        )
        return MessageEvent(
            text=message.text or "",
            message_type=msg_type,
            source=source,
            raw_message=message,
            message_id=str(message.message_id),
            platform_update_id=update_id,
            reply_to_message_id=reply_to_id,
            reply_to_text=reply_to_text,
            auto_skill=topic_skill,
            channel_prompt=_channel_prompt,
            timestamp=message.date,
        )

    # -- Message reactions (processing lifecycle) --

    def _reactions_enabled(self) -> bool:
        """Reactions enabled via TELEGRAM_REACTIONS env/config."""
        return os.getenv("TELEGRAM_REACTIONS", "false").lower() not in {"false", "0", "no"}

    async def _set_reaction(self, chat_id: str, message_id: str, emoji: str) -> bool:
        """Set a single emoji reaction on a Telegram message."""
        if not self._bot:
            return False
        try:
            await self._bot.set_message_reaction(
                chat_id=normalize_telegram_chat_id(chat_id), message_id=int(message_id), reaction=emoji
            )
            return True
        except Exception as e:
            logger.debug("[%s] set_message_reaction failed (%s): %s", self.name, emoji, _redact_telegram_error_text(e))
            return False

    async def _clear_reactions(self, chat_id: str, message_id: str) -> bool:
        """Clear all bot-set reactions (``reaction=None`` is the documented Bot API way)."""
        if not self._bot:
            return False
        try:
            await self._bot.set_message_reaction(
                chat_id=normalize_telegram_chat_id(chat_id), message_id=int(message_id), reaction=None
            )
            return True
        except Exception as e:
            logger.debug("[%s] clear reactions failed: %s", self.name, _redact_telegram_error_text(e))
            return False

    async def on_processing_start(self, event: MessageEvent) -> None:
        """Add an in-progress reaction when message processing begins."""
        if not self._reactions_enabled():
            return
        chat_id = getattr(event.source, "chat_id", None)
        message_id = getattr(event, "message_id", None)
        if chat_id and message_id:
            await self._set_reaction(chat_id, message_id, "\U0001f440")

    async def on_processing_complete(self, event: MessageEvent, outcome: ProcessingOutcome) -> None:
        """Swap the in-progress reaction for a final success/failure reaction.

        Telegram's set_message_reaction replaces (not adds), so no remove step. CANCELLED
        explicitly clears the 👀 so it doesn't linger when the cancel was the last activity.
        """
        if not self._reactions_enabled():
            return
        chat_id = getattr(event.source, "chat_id", None)
        message_id = getattr(event, "message_id", None)
        if not (chat_id and message_id):
            return
        if outcome == ProcessingOutcome.CANCELLED:
            await self._clear_reactions(chat_id, message_id)
        else:
            await self._set_reaction(chat_id, message_id, "\U0001f44d" if outcome == ProcessingOutcome.SUCCESS else "\U0001f44e")


# -- Plugin registration glue: register(ctx) plus the hook implementations (adapter factory,
# YAML→env/extra config, setup wizard, standalone sender) that replace the former
# per-platform core touchpoints in gateway/, hermes_cli/ and tools/send_message_tool.py.


def _resolve_notifications_mode() -> str:
    """Notification mode (all/important) from env, else config.yaml
    display.platforms.telegram.notifications, default 'important'."""
    mode = os.getenv("HERMES_TELEGRAM_NOTIFICATIONS", "")
    if not mode:
        try:
            from gateway.config import load_gateway_config
            from gateway.run import cfg_get
            _gw_cfg = load_gateway_config()
            _raw = cfg_get(_gw_cfg, "display", "platforms", "telegram", "notifications")
            if _raw not in {None, ""}:
                mode = str(_raw).strip().lower()
        except Exception:
            pass
    mode = mode or "important"
    if mode not in {"all", "important"}:
        logger.warning("Unknown telegram notifications mode '%s', defaulting to 'important' (valid: all, important)", mode)
        mode = "important"
    return mode


def _build_adapter(config):
    """Construct TelegramAdapter and apply the notification mode."""
    adapter = TelegramAdapter(config)
    try:
        adapter._notifications_mode = _resolve_notifications_mode()
    except Exception:
        adapter._notifications_mode = "important"
    return adapter


def _is_connected(config) -> bool:
    """Connected when a bot token is configured (env or PlatformConfig.token).

    check_telegram_requirements() only checks the SDK is importable; without this gate the
    plugin-enable pass would enable Telegram on any machine with the SDK installed.
    """
    token = getattr(config, "token", None)
    if not token:
        import hermes_cli.gateway as gateway_mod
        token = gateway_mod.get_env_value("TELEGRAM_BOT_TOKEN") or ""
    return bool(str(token).strip())


async def _standalone_send(pconfig, chat_id, message, *, thread_id=None, media_files=None, force_document=False):
    """Out-of-process delivery (standalone_sender_fn contract) so deliver=telegram cron jobs
    succeed without the gateway; delegates to the REST ``_send_telegram`` sender."""
    token = getattr(pconfig, "token", None)
    if not token:
        # Profile-scoped read: don't borrow another profile's env-bridged token under multiplex.
        from agent.secret_scope import get_secret
        token = get_secret("TELEGRAM_BOT_TOKEN", "") or ""
    disable_link_previews = bool(getattr(pconfig, "extra", {}) and pconfig.extra.get("disable_link_previews"))
    from tools.send_message_tool import _send_telegram
    return await _send_telegram(
        token, chat_id, message, media_files=media_files, thread_id=thread_id,
        disable_link_previews=disable_link_previews, force_document=force_document,
    )


def interactive_setup() -> None:
    """Configure Telegram credentials and allowlist via the CLI setup wizard (lazy import)."""
    from hermes_cli import setup as _setup_mod
    _setup_mod._setup_telegram()


def _apply_yaml_config(yaml_cfg: dict, telegram_cfg: dict) -> dict | None:
    """Translate config.yaml telegram: keys into TELEGRAM_* env vars and PlatformConfig.extra.

    Env vars take precedence over YAML. Returns extras to merge into PlatformConfig.extra, or None.
    """
    import json as _json
    extras: dict = {}
    # Under multiplex a secondary profile's authorization gates must NOT hit the process-global
    # env (first-writer-wins would pin them for every profile); they flow via extra/secret scope.
    try:
        from agent.secret_scope import current_secret_scope, is_multiplex_active
        _skip_env_bridge = bool(is_multiplex_active() and current_secret_scope() is not None)
    except Exception:
        _skip_env_bridge = False

    def _set_env(env: str, value: str) -> None:
        if not os.getenv(env):
            os.environ[env] = value

    def _bridge_lower(key: str, env: str) -> None:
        if key in telegram_cfg:
            _set_env(env, str(telegram_cfg[key]).lower())

    def _bridge_gate(key: str, env: str, value: Any, *, seed_extra: bool = False) -> None:
        """CSV allowlist gate: list → comma-joined; skipped under multiplex secret scope."""
        if value is None:
            return
        if seed_extra:
            extras.setdefault(key, value)
        if isinstance(value, list):
            value = ",".join(str(v) for v in value)
        if not _skip_env_bridge:
            _set_env(env, str(value))

    if "disable_topic_auto_rename" in telegram_cfg:
        extras.setdefault("disable_topic_auto_rename", telegram_cfg["disable_topic_auto_rename"])

    _effective_rm = telegram_cfg.get("require_mention", yaml_cfg.get("require_mention"))
    if _effective_rm is not None:
        _set_env("TELEGRAM_REQUIRE_MENTION", str(_effective_rm).lower())
    if "mention_patterns" in telegram_cfg:
        _set_env("TELEGRAM_MENTION_PATTERNS", _json.dumps(telegram_cfg["mention_patterns"]))
    _bridge_lower("exclusive_bot_mentions", "TELEGRAM_EXCLUSIVE_BOT_MENTIONS")
    _bridge_lower("allow_bots", "TELEGRAM_ALLOW_BOTS")
    _bridge_lower("guest_mode", "TELEGRAM_GUEST_MODE")
    _bridge_lower("observe_unmentioned_group_messages", "TELEGRAM_OBSERVE_UNMENTIONED_GROUP_MESSAGES")
    # No extras seed for allowed_chats / allowed_topics / group_allowed_chats: the shared-key
    # loop already bridges them with their original type and this merge would clobber it.
    _bridge_gate("free_response_chats", "TELEGRAM_FREE_RESPONSE_CHATS", telegram_cfg.get("free_response_chats"), seed_extra=True)
    _bridge_gate("free_response_topics", "TELEGRAM_FREE_RESPONSE_TOPICS", telegram_cfg.get("free_response_topics"))
    _bridge_gate("allowed_chats", "TELEGRAM_ALLOWED_CHATS", telegram_cfg.get("allowed_chats"))
    _bridge_gate("allowed_topics", "TELEGRAM_ALLOWED_TOPICS", telegram_cfg.get("allowed_topics"))
    _bridge_gate("ignored_threads", "TELEGRAM_IGNORED_THREADS", telegram_cfg.get("ignored_threads"), seed_extra=True)
    _bridge_lower("reactions", "TELEGRAM_REACTIONS")
    if "proxy_url" in telegram_cfg:
        _set_env("TELEGRAM_PROXY", str(telegram_cfg["proxy_url"]).strip())
    _telegram_extra = telegram_cfg.get("extra") if isinstance(telegram_cfg.get("extra"), dict) else {}
    _telegram_rtm = telegram_cfg["reply_to_mode"] if "reply_to_mode" in telegram_cfg else _telegram_extra.get("reply_to_mode")
    if _telegram_rtm is not None:
        _set_env("TELEGRAM_REPLY_TO_MODE", "off" if _telegram_rtm is False else str(_telegram_rtm).lower())
    _bridge_gate("allow_from", "TELEGRAM_ALLOWED_USERS", telegram_cfg.get("allow_from"))
    _bridge_gate(
        "group_allow_from", "TELEGRAM_GROUP_ALLOWED_USERS",
        telegram_cfg.get("group_allow_from") or _telegram_extra.get("group_allow_from"),
    )
    _bridge_gate(
        "group_allowed_chats", "TELEGRAM_GROUP_ALLOWED_CHATS",
        telegram_cfg.get("group_allowed_chats") or _telegram_extra.get("group_allowed_chats"),
    )
    for _key in ("guest_mode", "disable_link_previews", "observe_unmentioned_group_messages", "free_response_topics"):
        if _key in telegram_cfg:
            extras.setdefault(_key, telegram_cfg[_key])
    # Pass through telegram-specific extra keys but EXCLUDE generic shared-config keys:
    # _merge_platform_map already applied top-level-over-nested precedence, and our return is
    # merged via dict.update(), so re-emitting them would undo it.
    _GENERIC_MERGE_KEYS = {
        "reply_prefix", "reply_in_thread", "reply_to_mode",
        "unauthorized_dm_behavior", "notice_delivery", "require_mention",
        "channel_skill_bindings", "channel_prompts", "gateway_restart_notification",
        "allow_from", "allow_admin_from", "dm_policy", "group_policy",
    }
    for _k, _v in _telegram_extra.items():
        if _k not in _GENERIC_MERGE_KEYS:
            extras.setdefault(_k, _v)
    return extras or None


def register(ctx) -> None:
    """Plugin entry point — called by the Hermes plugin system."""
    ctx.register_platform(
        name="telegram",
        label="Telegram",
        adapter_factory=_build_adapter,
        check_fn=telegram_deps_present,
        ensure_deps_fn=check_telegram_requirements,
        is_connected=_is_connected,
        required_env=["TELEGRAM_BOT_TOKEN"],
        install_hint="Run `hermes setup` to install Telegram support.",
        setup_fn=interactive_setup,
        apply_yaml_config_fn=_apply_yaml_config,
        allowed_users_env="TELEGRAM_ALLOWED_USERS",
        allow_all_env="TELEGRAM_ALLOW_ALL_USERS",
        cron_deliver_env_var="TELEGRAM_HOME_CHANNEL",
        standalone_sender_fn=_standalone_send,
        max_message_length=4096,
        emoji="✈️",
        allow_update_command=True,
    )

